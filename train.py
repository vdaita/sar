import torch
from torch import nn, Tensor
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
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

app = typer.Typer()

def tensor_to_table(tensor, one_color="bold green", zero_color="dim"):
    table = Table(show_header=False, box=None, pad_edge=False, collapse_padding=True)
    data = tensor.tolist()
    for row in data:
        colored_row = [
            f"[{one_color}]1[/{one_color}]" if v == 1 else f"[{zero_color}]0[/{zero_color}]"
            for v in row
        ]
        table.add_row(*colored_row)
    return table

def get_limited_tokenizer(vocab_size: int) -> GPT2Tokenizer: 
    # Load GPT-2 tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    # Get original vocab (dict: token -> id)
    orig_vocab = tok.get_vocab()

    # Sort by ID so we keep the first 10k *in order*
    sorted_vocab = sorted(orig_vocab.items(), key=lambda x: x[1])
    limited_vocab = dict(sorted_vocab[:vocab_size])

    # Build a reverse vocab (id -> token)
    id2token = {i: t for i, (t, _) in enumerate(limited_vocab.items())}
    token2id = {t: i for i, t in id2token.items()}

    # Replace tokenizer's internal vocab
    tok.vocab = token2id
    tok._tokenizer.model.vocab = token2id
    tok._tokenizer.model.vocab_size = len(token2id)

    # You may also want to set special tokens manually
    tok.add_special_tokens({
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>"
    })

    return tok

def get_batches_from_dataset(dataset, batch_size: int, num_batches: int):
    for i in range(num_batches):
        batch = dataset.select(range(i * batch_size, (i + 1) * batch_size))
        batch = torch.stack(batch["tokens"].to_list())
        yield batch

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

    tokenizer_dir = f"./tokenizer-{vocab_size}"
    if os.path.exists(tokenizer_dir):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)    
    else:
        tokenizer = get_limited_tokenizer(vocab_size)
        tokenizer.save_pretrained(tokenizer_dir)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_embd=128,
        n_layer=4,
        n_head=8,
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
    max_length = configs.get("max_length")
    batch_size = configs.get("batch_size")
    num_train_steps = configs.get("num_train_steps")
    eval_every = configs.get("eval_every")
    save_every = configs.get("save_every")
    k = configs.get("k")
    lr = configs.get("lr")

    p_extend = configs.get("p_extend")
    extend_k = configs.get("extend_k")

    eval_num_batches = configs.get("eval_num_batches")

    wandb.init(
        project="sar-transformer",
        config=configs.to_dict()
    )

    tokenizer_dir = f"./{checkpoint_dir}/tokenizer-{vocab_size}"
    if os.path.exists(tokenizer_dir):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)    
    else:
        tokenizer = get_limited_tokenizer(vocab_size)
        tokenizer.save_pretrained(tokenizer_dir)

    unix_millis = int(round(time.time() * 1000))
    model_folder = f"{checkpoint_dir}/model-{mode}-{unix_millis}/"
    os.makedirs(model_folder, exist_ok=True)

    with open(f"{model_folder}/config.yaml", "w") as f:
        yaml.safe_dump(configs, f)

    # metrics: train loss, eval loss, eval ppl, time per step
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_embd=128,
        n_layer=4,
        n_head=8,
        bos_token_id=1,
        eos_token_id=2
    )

    model = GPT2LMHeadModel(config)
    train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    test_dataset = load_dataset("roneneldan/TinyStories",split="test")

    def preprocess_dataset(dataset):
        return {"tokens": tokenizer(dataset["text"], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")["input_ids"]}

    train_dataset, test_dataset = train_dataset.map(preprocess_dataset), test_dataset.map(preprocess_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for step in range(num_train_steps):
        if mode == "sar":
            attention_mask = generate_sar_attention_mask(max_length, k=k)
        elif mode == "overlap":
            attention_mask = generate_overlap_attention_mask(max_length, k=k, p_extend=p_extend, extend_k=extend_k)
        else:
            attention_mask = generate_default_attention_mask(max_length)

        model.train()
        batches = next(get_batches_from_dataset(train_dataset, batch_size=batch_size, num_batches=1))

        losses = []
        for input_ids in batches:
            batch_size, seq_len = input_ids.shape
            expanded_attention_mask = expand_attention_mask(attention_mask, batch_size)
            outputs = model(input_ids=input_ids, attention_mask=expanded_attention_mask, labels=input_ids)
            loss = outputs.loss
            losses.append(loss)

        # perform a training step
        joint_loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        wandb.log({"train/loss": joint_loss.item()}, step=step)
        print(f"Step {step}: train loss {joint_loss.item()}")

        if step % eval_every == 0:
            model.eval()
            eval_losses = []
            eval_batches = next(get_batches_from_dataset(test_dataset, batch_size=batch_size, num_batches=eval_num_batches))
            for eval_input_ids in eval_batches:
                batch_size, seq_len = eval_input_ids.shape
                expanded_attention_mask = expand_attention_mask(attention_mask, batch_size)
                with torch.no_grad():
                    eval_outputs = model(input_ids=eval_input_ids, attention_mask=expanded_attention_mask, labels=eval_input_ids)
                    eval_loss = eval_outputs.loss
                    eval_losses.append(eval_loss)

            joint_eval_loss = torch.stack(eval_losses).mean()
            eval_ppl = torch.exp(joint_eval_loss)
            wandb.log({"eval/loss": joint_eval_loss.item(), "eval/ppl": eval_ppl.item()}, step=step)
            print(f"Step {step}: eval loss {joint_eval_loss.item()}, eval ppl {eval_ppl.item()}")

        if step % save_every == 0:
            model.save_pretrained(f"{model_folder}/step-{step}")
            print(f"Saved model checkpoint at step {step} to {model_folder}/step-{step}")

    model.save_pretrained(f"{model_folder}/final")
    print(f"Saved final model checkpoint to {model_folder}/final")
    wandb.finish()

if __name__ == "__main__":
    app()