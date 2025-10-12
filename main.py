import torch
from torch import nn, Tensor
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import datasets
from rich import print
from rich.table import Table
from rich.console import Console
from attention_masks import generate_sar_attention_mask, generate_overlap_attention_mask, generate_default_attention_mask
import typer
import os
import time
import yaml
import wandb
from typing import Dict
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import itertools

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

def expand_attention_mask(attention_mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    return attention_mask.unsqueeze(0).expand(batch_size, -1, -1)

@app.command()
def sample_masks(seq_len: int = 32, k: int = 4, p_extend: float = 0.5, extend_k: int = 2):
    sar_mask = generate_sar_attention_mask(seq_len, k)
    overlap_mask = generate_overlap_attention_mask(seq_len, k, p_extend, extend_k)
    default_mask = generate_default_attention_mask(seq_len)

    sar_mask, overlap_mask, default_mask = sar_mask.int(), overlap_mask.int(), default_mask.int()
    sar_mask, overlap_mask, default_mask = tensor_to_table(sar_mask), tensor_to_table(overlap_mask), tensor_to_table(default_mask)

    console = Console()
    console.print("[bold underline]SAR Attention Mask:[/bold underline]")
    console.print(sar_mask)
    console.print("\n[bold underline]Overlap Attention Mask:[/bold underline]")
    console.print(overlap_mask)
    console.print("\n[bold underline]Default Attention Mask:[/bold underline]")
    console.print(default_mask)

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

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=1,
        eos_token_id=2
    )

    model = GPT2LMHeadModel(config)
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

    k = configs.get("k")
    p_extend = configs.get("p_extend")
    extend_k = configs.get("extend_k")

    n_embed = configs.get("n_embed")
    n_layer = configs.get("n_layer")
    n_head = configs.get("n_head")

    eval_num_batches = configs.get("eval_num_batches")

    timestamp = time.time()

    name = f"{mode}-k={k}-p-extend={p_extend}-extend-k={extend_k}-bs={batch_size}-lr={lr}-embed={n_embed}-layer={n_layer}-head={n_head}-timestamp={timestamp}"

    wandb.init(
        project="sar-transformer",
        config=configs,
        name=name
    )

    tokenizer = get_tokenizer()

    unix_millis = int(round(time.time() * 1000))
    model_folder = f"{checkpoint_dir}/model-{mode}-{unix_millis}/"
    os.makedirs(model_folder, exist_ok=True)

    with open(f"{model_folder}/config.yaml", "w") as f:
        yaml.safe_dump(configs, f)

    # metrics: train loss, eval loss, eval ppl, time per step
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_embd=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=1,
        eos_token_id=2,
        # resid_pdrop=0.1,
        # attn_pdrop=0.1,
        # embd_pdrop=0.1
    )

    model = GPT2LMHeadModel(config)

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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True) # type: ignore
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=True) # type: ignore
    
    train_iterator = itertools.cycle(train_dataloader)
    test_iterator = itertools.cycle(test_dataloader)

    for step in tqdm(range(num_train_steps), desc="Train Step"):
        start_time_train = time.time()

        if mode == "sar":
            attention_mask = generate_sar_attention_mask(max_length, k=k)
        elif mode == "overlap":
            attention_mask = generate_overlap_attention_mask(max_length, k=k, p_extend=p_extend, extend_k=extend_k)
        elif mode == "base":
            attention_mask = generate_default_attention_mask(max_length)
        else:
            raise ValueError(f"Unknown mode for attention mask: {mode}")

        model.train()
        input_ids: Tensor = next(train_iterator)["tokens"]
        input_ids = input_ids.to(device)
        # print("Input ids: ", input_ids.shape)

        batch_size, seq_len = input_ids.shape
        expanded_attention_mask = expand_attention_mask(attention_mask, batch_size)
        expanded_attention_mask = expanded_attention_mask.to(device)
        outputs = model(input_ids=input_ids, attention_mask=expanded_attention_mask, labels=input_ids)
        loss = outputs.loss

        # perform a training step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time_train = time.time()

        wandb.log({"train/loss": loss.item(), "train/step_timer": (end_time_train - start_time_train)}, step=step)
        print(f"Step {step}: train loss {loss.item()}")

        if step % eval_every == 0:
            start_time_eval = time.time()
            model.eval()

            prompt = "Once upon a time"
            sample = model.generate(**tokenizer(prompt, return_tensors="pt").to(device), max_length=64)
            print(tokenizer.decode(sample[0]))

            eval_losses = []
            for _ in range(eval_num_batches):
                eval_input_ids = next(test_iterator)["tokens"]
                eval_input_ids = eval_input_ids.to(device)
                # print("Eval input ids shape: ", eval_input_ids.shape)

                batch_size, seq_len = eval_input_ids.shape
                expanded_attention_mask = expand_attention_mask(attention_mask, batch_size)
                expanded_attention_mask = expanded_attention_mask.to(device)
                with torch.no_grad():
                    eval_outputs = model(input_ids=eval_input_ids, attention_mask=expanded_attention_mask, labels=eval_input_ids)
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