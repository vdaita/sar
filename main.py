import torch
from torch import nn, Tensor
# from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM, GPT2Tokenizer, GPTNeoConfig, GPTNeoForCausalLM
from datasets import load_dataset
import datasets
from rich import print
from rich.table import Table
from rich.console import Console
from attention_masks import preprocess_default_attention, preprocess_gist_glue, preprocess_gist_sar
import typer
import os
import time
import yaml
import wandb
from typing import Dict, Optional, Tuple
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
def sample_masks(seq_len: int = 32, k: int = 4):
    sample_input = torch.arange(start=1, end=seq_len + 1).unsqueeze(0).long()  # shape (1, seq_len)
    gist_token = -1

    sar_result = preprocess_gist_sar(sample_input, gist_token, k)
    glue_result = preprocess_gist_glue(sample_input, gist_token, k)
    default_result = preprocess_default_attention(sample_input)

    sar_mask = sar_result.mask[0]
    glue_mask = glue_result.mask[0]
    default_mask = default_result.mask[0]

    sar_mask, default_mask, glue_mask = sar_mask.int(), default_mask.int(), glue_mask.int()
    print(f"sar_mask_shape={sar_mask.shape}, glue_mask_shape={glue_mask.shape}, default_mask_shape={default_mask.shape}")

    sar_mask, default_mask, glue_mask = tensor_to_table(sar_mask), tensor_to_table(default_mask), tensor_to_table(glue_mask)

    console = Console()
    console.print("[bold underline]Glue Attention Mask:[/bold underline]")
    console.print(glue_mask)
    console.print(f"Input ids: {sar_result.input_ids}")
    console.print(f"Labels: {sar_result.labels}")
    console.print("[bold underline]SAR Attention Mask:[/bold underline]")
    console.print(sar_mask)
    console.print(f"Input ids: {sar_result.input_ids}")
    console.print(f"Labels: {sar_result.labels}")
    console.print("\n[bold underline]Default Attention Mask:[/bold underline]")
    console.print(default_mask)
    console.print(f"Input ids: {default_result.input_ids}")
    console.print(f"Labels: {default_result.labels}")

def _no_causal_gpt_neo_attn(self, query, key, value, attention_mask=None, head_mask=None):
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
def train_model(conf_path: str): # you can train this in default, sar, overlap. however, the eval loop for both sar and overlap will be sar
    with open(conf_path, "r") as f:
        configs = yaml.safe_load(f)

    mode: str = configs.get("mode")
    vocab_size: int = configs.get("vocab_size")
    checkpoint_dir: str = configs.get("checkpoint_dir")
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
    
    max_position_slots = configs.get("max_position_slots")
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
            max_position_embeddings=max_position_slots,
            hidden_size=n_embed,
            intermediate_size=3 * n_embed,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        model = LlamaForCausalLM(config)
        model.set_attn_implementation('eager')

        transformers.models.llama.modeling_llama.eager_attention_forward = no_causal_llama_eager_attn_forward
    elif model_type == "gpt-neo":
        if n_layer % 2 == 1:
            raise ValueError("gpt-neo models must have an even number of layers")

        config = GPTNeoConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_slots,
            attention_types=[[["global", "local"], n_layer // 2]],
            window_size=max_position_slots,
            hidden_size=n_embed,
            num_layers=n_layer,
            num_heads=n_head,
            intermediate_size=3 * n_embed,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = GPTNeoForCausalLM(config)
        model.set_attn_implementation('eager')
        for block in model.transformer.h:
            print(inspect.signature(block.attn.attention._attn))
            block.attn.attention._attn = types.MethodType(_no_causal_gpt_neo_attn, block.attn.attention)

    else:
        raise ValueError("Your model type can only either be llama (default) or gpt-neo.")

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

    gist_token = 9999

    def run_model(input_ids: Tensor):
        if mode == "sar":
            preprocess_result = preprocess_gist_sar(input_ids, gist_token=gist_token, k=k)
        elif mode == "glue":
            preprocess_result = preprocess_gist_glue(input_ids, gist_token=gist_token, k=k)
        elif mode == "default":
            preprocess_result = preprocess_default_attention(input_ids)
        else:
            raise ValueError("Mode must be one of 'sar', 'glue', or 'default'")

        attention_mask, input_ids, labels = preprocess_result.mask, preprocess_result.input_ids, preprocess_result.labels
        attention_mask = attention_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        attention_mask, input_ids, labels = attention_mask.to(device), input_ids.to(device), labels.to(device)
        
        # attention mask has 1 on the tokens that should be viewed
        attention_mask = attention_mask.float()
        attention_mask[attention_mask == 0] = -1e10
        attention_mask[attention_mask == 1] = 0.0
        # print(f"Attention mask shape: {attention_mask.shape}, max input id: {input_ids.max()}, min input id: {input_ids.min()}, max label id: {labels.max()}, min label id: {labels.min()}")
        
        assert torch.all(input_ids >= 0) and torch.all(input_ids < vocab_size)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    for step in tqdm(range(num_train_steps), desc="Train Step"):
        start_time_train = time.time()
        model.train()
        input_ids: Tensor = next(train_iterator)["tokens"]
        input_ids = input_ids.to(device)
        outputs = run_model(input_ids)
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

            with torch.no_grad():
                prompt = "Once upon a time"
                sample = model.generate(**tokenizer(prompt, return_tensors="pt").to(device), max_length=256, pad_token_id=tokenizer.pad_token_id)
                print(tokenizer.decode(sample[0]))

            eval_losses = []
            for _ in range(eval_num_batches):
                eval_input_ids = next(test_iterator)["tokens"]
                eval_input_ids = eval_input_ids.to(device)
                with torch.no_grad():
                    eval_outputs = run_model(eval_input_ids)
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