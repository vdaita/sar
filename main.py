import torch
from torch import nn, Tensor
# from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM, GPT2Tokenizer, GPTNeoConfig, GPTNeoForCausalLM
from datasets import load_dataset
import datasets
from rich import print
from rich.table import Table
from rich.console import Console
from attention_masks import generate_sar_attention_mask, generate_overlap_attention_mask, generate_default_attention_mask, generate_glue_attention_mask
import typer
import os
import time
import yaml
import wandb
from typing import Dict, Optional
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import itertools
from transformers import get_cosine_schedule_with_warmup
import transformers
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.llama.modeling_llama import repeat_kv
import inspect
import types

app = typer.Typer()
TOKENIZER_NAME = "georgeyw/TinyStories-tokenizer-10k"

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tensor_to_table(tensor, one_color="bold red", zero_color="dim"):
    table = Table(show_header=False, box=None, pad_edge=False, collapse_padding=True)
    data = tensor.tolist()
    for row in data:
        colored_row = [
            f"[{one_color}]1[/{one_color}]" if v == 1 else f"[{zero_color}]0[/{zero_color}]"
            for v in row
        ]
        table.add_row(*colored_row)
    return table

def expand_attention_mask(attention_mask: torch.Tensor, num_heads: int,  batch_size: int) -> torch.Tensor:
    mask = attention_mask.clone()
    mask[mask == 0] = -1e10
    mask[mask == 1] = 0
    return (
        mask
        .unsqueeze(0)
        .expand(num_heads, -1, -1)
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
    )

@app.command()
def sample_masks(seq_len: int = 32, k: int = 4, p_extend: float = 0.5, extend_k: int = 2):
    sar_mask = generate_sar_attention_mask(seq_len, k)
    overlap_mask = generate_overlap_attention_mask(seq_len, k, p_extend, extend_k)
    default_mask = generate_default_attention_mask(seq_len)
    glue_mask = generate_glue_attention_mask(seq_len, k)

    sar_mask, overlap_mask, default_mask, glue_mask = sar_mask.int(), overlap_mask.int(), default_mask.int(), glue_mask.int()
    sar_mask, overlap_mask, default_mask, glue_mask = tensor_to_table(sar_mask), tensor_to_table(overlap_mask), tensor_to_table(default_mask), tensor_to_table(glue_mask)

    console = Console()
    console.print("[bold underline]Glue Attention Mask:[/bold underline]")
    console.print(glue_mask)
    console.print("[bold underline]SAR Attention Mask:[/bold underline]")
    console.print(sar_mask)
    console.print("\n[bold underline]Overlap Attention Mask:[/bold underline]")
    console.print(overlap_mask)
    console.print("\n[bold underline]Default Attention Mask:[/bold underline]")
    console.print(default_mask)

def _no_causal_gpt_neo_attn(self, query, key, value, attention_mask=None):
    # Keep the attention weights computation in fp32 to avoid overflow issues
    query = query.to(torch.float32)
    key = key.to(torch.float32)

    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    # Apply sliding window masking for local attention layers
    if attention_mask is not None:  # no matter the length, we just slice it
        attention_mask_sliced = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask_sliced

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.to(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights
    
def no_causal_llama_eager_attn_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

@app.command()
def test_model(conf_path: str, checkpoint_path: str):
    with open(conf_path, "r") as f:
        configs = yaml.safe_load(f)
        
    mode = configs.get("mode")
    vocab_size = configs.get("vocab_size")
    max_length = configs.get("max_length")
    batch_size = configs.get("batch_size")
    k = configs.get("k")

    p_extend = configs.get("p_extend")
    extend_k = configs.get("extend_k")

    n_embd = configs.get("n_embd")
    n_layer = configs.get("n_layer")
    n_head = configs.get("n_head")

    tokenizer = get_tokenizer()
    model_type = configs.get("base_model", "llama")

    if model_type == "llama":
        config = LlamaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_length,
            hidden_size=n_embd,
            intermediate_size=3 * n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        model = LlamaForCausalLM(config)
        
        transformers.models.llama.modeling_llama.eager_attention_forward = no_causal_llama_eager_attn_forward
    elif model_type == "gpt-neo":
        config = GPTNeoConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_length,
            hidden_size=n_embd,
            num_layers=n_layer,
            num_heads=n_head,
            intermediate_size=3 * n_embd,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        model = GPTNeoForCausalLM(config)
        
        # perform monkeypatch
        for block in model.transformer.h:
            block.attn.attention._attn = types.MethodType(_no_causal_gpt_neo_attn, block.attn.attention)
    else:
        raise ValueError("Your model type can only either be llama (default) or gpt-neo.")

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    example_prompts = [
        "Once upon a time",
        "In a galaxy far, far away",
        "The quick brown fox",
        "To be or not to be",
        "In the beginning"
    ]

    for prompt in example_prompts:
        inputs: Dict[str, Tensor] = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length") # type: ignore
        input_ids: Tensor = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape

        if mode == "sar":
            attention_mask = generate_sar_attention_mask(max_length, k=k)
        elif mode == "overlap":
            attention_mask = generate_overlap_attention_mask(max_length, k=k, p_extend=p_extend, extend_k=extend_k)
        else:
            attention_mask = generate_default_attention_mask(max_length)

        expanded_attention_mask = expand_attention_mask(attention_mask, batch_size)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=expanded_attention_mask)
            logits = outputs.logits

        predicted_id = torch.argmax(logits[:, -1, :], dim=-1)
        predicted_token = tokenizer.decode(predicted_id)

        print(f"[bold]Prompt:[/bold] {prompt}")
        print(f"[bold]Predicted next token:[/bold] {predicted_token}\n")

@app.command()
def train_model(conf_path: str): # you can train this in default, sar, overlap. however, the eval loop for both sar and overlap will be sar
    with open(conf_path, "r") as f:
        configs = yaml.safe_load(f)
        
    mode = configs.get("mode")
    vocab_size = configs.get("vocab_size")
    checkpoint_dir = configs.get("checkpoint_dir")
    os.makedirs(checkpoint_dir, exist_ok=True)

    max_length = configs.get("max_length")
    batch_size = configs.get("batch_size")
    num_train_steps = configs.get("num_train_steps")
    eval_every = configs.get("eval_every")
    save_every = configs.get("save_every")
    lr = configs.get("lr")
    weight_decay_lambda = configs.get("weight_decay_lambda")

    k = configs.get("k")
    p_extend = configs.get("p_extend")
    extend_k = configs.get("extend_k")

    n_embed = configs.get("n_embed")
    n_layer = configs.get("n_layer")
    n_head = configs.get("n_head")
    
    num_warmup_steps = configs.get("num_warmup_steps")

    eval_num_batches = configs.get("eval_num_batches")
    model_type = configs.get("base_model", "llama")

    unix_millis = int(round(time.time() * 1000))
    model_folder = f"{checkpoint_dir}/model-{mode}-{unix_millis}/"
    
    name = f"{mode}-arch={model_type}-k={k}-p-extend={p_extend}-extend-k={extend_k}-bs={batch_size}-lr={lr}-embed={n_embed}-layer={n_layer}-head={n_head}-warmup-steps={num_warmup_steps}-wd-lambda={weight_decay_lambda}-timestamp={unix_millis}"
    tokenizer = get_tokenizer()


    os.makedirs(model_folder, exist_ok=True)

    with open(f"{model_folder}/config.yaml", "w") as f:
        yaml.safe_dump(configs, f)

    # metrics: train loss, eval loss, eval ppl, time per step
    if model_type == "llama":
        config = LlamaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_length,
            hidden_size=n_embed,
            intermediate_size=3 * n_embed,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        model = LlamaForCausalLM(config)
    elif model_type == "gpt-neo":
        if n_layer % 2 == 1:
            raise ValueError("gpt-neo models must have an even number of layers")

        config = GPTNeoConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_length,
            attention_types=[[["global", "local"], n_layer // 2]],
            window_size=max_length,
            hidden_size=n_embed,
            num_layers=n_layer,
            num_heads=n_head,
            intermediate_size=3 * n_embed,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = GPTNeoForCausalLM(config)
    else:
        raise ValueError("Your model type can only either be llama (default) or gpt-neo.")

    model.set_attn_implementation('eager')
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    device = torch.device('cpu')
    if torch.mps.is_available():
        device = torch.device('mps')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = model.to(device) # type: ignore

    train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    test_dataset = load_dataset("roneneldan/TinyStories",split="validation")

    def preprocess_dataset(dataset):
        return {"tokens": tokenizer(dataset["text"], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")["input_ids"]}

    train_dataset_path = f"./checkpoints/train_dataset_{max_length}"
    test_dataset_path = f"./checkpoints/test_dataset_{max_length}"

    if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path):
        print("Loading preprocessed datasets from disk...")
        train_dataset = datasets.load_from_disk(train_dataset_path)
        test_dataset = datasets.load_from_disk(test_dataset_path)
    else:
        print("Preprocessing datasets and saving to disk...")
        train_dataset = train_dataset.map(preprocess_dataset, batched=True, num_proc=8) # type: ignore
        test_dataset = test_dataset.map(preprocess_dataset, batched=True, num_proc=8) # type: ignore

        train_dataset.save_to_disk(train_dataset_path)
        test_dataset.save_to_disk(test_dataset_path)

    train_dataset, test_dataset = train_dataset.select_columns(["tokens"]), test_dataset.select_columns(["tokens"]) # type: ignore

    train_dataset.set_format(type="torch", columns=["tokens"])
    test_dataset.set_format(type="torch", columns=["tokens"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay_lambda)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True) # type: ignore
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=True) # type: ignore
    
    train_iterator = itertools.cycle(train_dataloader)
    test_iterator = itertools.cycle(test_dataloader)
    
    # move wandb initialization here so that you don't accidentally get runs when you crash above
    wandb.init(
        project="sar-transformer",
        config=configs,
        name=name
    )

    for step in tqdm(range(num_train_steps), desc="Train Step"):
        start_time_train = time.time()

        if mode == "sar":
            attention_mask = generate_sar_attention_mask(max_length, k=k)
        elif mode == "overlap":
            attention_mask = generate_overlap_attention_mask(max_length, k=k, p_extend=p_extend, extend_k=extend_k)
        elif mode == "base":
            attention_mask = generate_default_attention_mask(max_length)
        elif mode == "glue":
            attention_mask = generate_glue_attention_mask(max_length, k=k)
        else:
            raise ValueError(f"Unknown mode for attention mask: {mode}")

        model.train()
        input_ids: Tensor = next(train_iterator)["tokens"]
        input_ids = input_ids.to(device)
        
        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100
        # print("Input ids: ", input_ids.shape)

        batch_size, seq_len = input_ids.shape
        expanded_attention_mask = expand_attention_mask(attention_mask, n_head, batch_size)
        expanded_attention_mask = expanded_attention_mask.to(device)
        outputs = model(input_ids=input_ids, attention_mask=expanded_attention_mask, labels=labels)
        loss = outputs.loss

        # perform a training step
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
        end_time_train = time.time()

        wandb.log({"train/loss": loss.item(), "train/step_timer": (end_time_train - start_time_train), "train/lr": scheduler.get_last_lr()[0], "train/grad_norm": grad_norm.item()}, step=step)
        if step % 50 == 0:
            print(f"Step {step}: train loss {loss.item()}")

        if step % eval_every == 0:
            start_time_eval = time.time()
            model.eval()

            prompt = "Once upon a time"
            sample = model.generate(**tokenizer(prompt, return_tensors="pt").to(device), max_length=256, pad_token_id=tokenizer.pad_token_id)
            print(tokenizer.decode(sample[0]))

            eval_losses = []
            for _ in range(eval_num_batches):
                eval_input_ids = next(test_iterator)["tokens"]
                eval_input_ids = eval_input_ids.to(device)
                # print("Eval input ids shape: ", eval_input_ids.shape)
                
                eval_labels = eval_input_ids.clone()
                eval_labels[eval_input_ids == tokenizer.pad_token_id] = -100

                batch_size, seq_len = eval_input_ids.shape
                expanded_attention_mask = expand_attention_mask(attention_mask, n_head, batch_size)
                expanded_attention_mask = expanded_attention_mask.to(device)
                with torch.no_grad():
                    eval_outputs = model(input_ids=eval_input_ids, attention_mask=expanded_attention_mask, labels=eval_labels)
                    eval_loss = eval_outputs.loss
                    eval_losses.append(eval_loss)

            joint_eval_loss = torch.stack(eval_losses).mean()
            eval_ppl = torch.exp(joint_eval_loss)
            end_time_eval = time.time()

            wandb.log({"eval/loss": joint_eval_loss.item(), "eval/ppl": eval_ppl.item(), "eval/step_timer": end_time_eval - start_time_eval}, step=step)
            print(f"Step {step}: eval loss {joint_eval_loss.item()}, eval ppl {eval_ppl.item()}")

        if step % save_every == 0:
            save_path = os.path.join(model_folder, f"step-{step}")
            model.save_pretrained(save_path)
            print(f"Saved model checkpoint at step {step} to {save_path}")

    final_save_path = os.path.join(model_folder, "final")
    model.save_pretrained(final_save_path)
    print(f"Saved final model checkpoint to {final_save_path}")
    wandb.finish()

if __name__ == "__main__":
    app()