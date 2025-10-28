import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from transformer import SARTransformer, Transformer
from rich import print
from datasets import load_dataset
import datasets
import wandb
from tqdm import tqdm
import typer
import os
import yaml
import time
import itertools
import torch.nn.functional as F
from typing import Optional, List, Literal

app = typer.Typer()
TOKENIZER_NAME = "georgeyw/TinyStories-tokenizer-10k"
DATASET_NAME = "roneneldan/TinyStories"

# am i doing this correctly?
def calculate_perplexity(logits: Tensor, labels: Tensor):
    logprobs = F.log_softmax(logits, dim=-1)
    labels = labels.unsqueeze(-1)
    nll = -logprobs.gather(dim=-1, index=labels)
    nll = torch.squeeze(nll, -1)
    nll = nll.mean()
    return torch.exp(nll)

@app.command()
def train(conf_path: str):
    with open(conf_path, "r") as f:
        conf = yaml.safe_load(f)

    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    mode = conf.get("mode")
    vocab_size = conf.get("vocab_size")
    checkpoint_dir = conf.get("checkpoint_dir")
    os.makedirs(checkpoint_dir, exist_ok=True)

    max_seq_len = conf.get("max_seq_len")
    batch_size = conf.get("batch_size")
    num_train_steps = conf.get("num_train_steps")

    eval_every = conf.get("eval_every")
    save_every = conf.get("save_every")

    lr = conf.get("lr")

    compress_seq_len = conf.get("compress_seq_len")
    dim_embed = conf.get("dim_embed")
    dim_ffn = conf.get("dim_ffn")

    n_layer = conf.get("n_layer")
    n_head = conf.get("n_head")

    eval_num_batches = conf.get("eval_num_batches")

    unix_millis = int(round(time.time() * 1000))
    model_folder = f"{checkpoint_dir}/model-{mode}-{unix_millis}/"
    name = f"{mode}-arch={mode}-compress_seq_len={compress_seq_len}-bs={batch_size}-eval_num_batches={eval_num_batches}-lr={lr}-embed={dim_embed}-ffn={dim_ffn}-layer={n_layer}-head={n_head}-timestamp={unix_millis}"

    with open(f"{model_folder}/config.yaml", "w") as f:
        yaml.safe_dump(conf, f)

    if mode == "sar":
        model: nn.Module = SARTransformer(
            dim_embed,
            n_head,
            dim_ffn,
            max_seq_len,
            n_layer,
            vocab_size,
            compress_seq_len
        )
    else:
        model: nn.Module = Transformer(
            dim_embed,
            n_head,
            dim_ffn,
            max_seq_len,
            n_layer,
            vocab_size
        )

    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    device = torch.device('cpu')
    if torch.mps.is_available():
        device = torch.device('mps')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = model.to(device)
    train_dataset = load_dataset(DATASET_NAME, split="train")
    test_dataset = load_dataset(DATASET_NAME, split="test")

    def preprocess_dataset(dataset):
        return {"tokens": tokenizer(dataset["text"], truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")["input_ids"]}

    train_dataset_path = f"./checkpoints/train_dataset_{max_seq_len}"
    test_dataset_path = f"./checkpoints/test_dataset_{max_seq_len}"

    if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path):
        print("Loading preprocessed datasets from disk...")
        train_dataset = datasets.load_from_disk(train_dataset_path)
        test_dataset = datasets.load_from_disk(test_dataset_path)
    else:
        print("Preprocessing datasets and saving to disk...")
        train_dataset = train_dataset.map(preprocess_dataset, batched=True, num_proc=8)  # type: ignore
        test_dataset = test_dataset.map(preprocess_dataset, batched=True, num_proc=8)  # type: ignore

        train_dataset.save_to_disk(train_dataset_path)
        test_dataset.save_to_disk(test_dataset_path)

    train_dataset, test_dataset = train_dataset.select_columns(["tokens"]), test_dataset.select_columns(
        ["tokens"])  # type: ignore

    train_dataset.set_format(type="torch", columns=["tokens"])
    test_dataset.set_format(type="torch", columns=["tokens"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True) # type: ignore
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=True) # type: ignore

    train_iterator = itertools.cycle(train_dataloader)
    test_iterator = itertools.cycle(test_dataloader)
    
    wandb.init(
        project="sar_transformer",
        config=conf,
        name=name
    )

    def step_model(input_ids: Tensor):
        if mode == "sar":
            logits = model(input_ids)
            labels = input_ids[:, compress_seq_len:, :]
            shifted_logits = logits[:, :-compress_seq_len, :]
            
            labels = labels.reshape(-1, vocab_size)
            shifted_logits = shifted_logits.reshape(-1, vocab_size)
            
            loss = F.cross_entropy(labels, shifted_logits)
        else:
            logits = model(input_ids)
            labels = input_ids[:, 1:, :]
            shifted_logits = logits[:, :-1, :]
            
            labels = labels.reshape(-1, vocab_size)
            shifted_logits = shifted_logits.reshape(-1, vocab_size)
            
            loss = F.cross_entropy(labels, shifted_logits)
        ppl = calculate_perplexity(shifted_logits, labels)
        return logits, loss, ppl

    for step in tqdm(range(num_train_steps), desc="Train step"):
        start_time_train = time.perf_counter()
        model.train()

        input_ids = next(train_iterator)["tokens"]
        input_ids = input_ids.to(device)
        logits, loss, ppl = step_model(input_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time_train = time.perf_counter()
        wandb.log({"train/loss": loss.item(), "train/step_timer": (end_time_train - start_time_train), "train/ppl": ppl.item()}, step=step)
        
        if step % 50 == 0:
            print(f"Step {step}: train loss = {loss.item():.4f}, ppl = {ppl.item():.4f}, step time = {(end_time_train - start_time_train):.4f}s")
        
        if step % eval_every == 0:
            start_time_eval = time.perf_counter()
            model.eval()
            
            eval_losses: List[float] = []
            eval_ppls: List[float] = []
            
            for _ in range(eval_num_batches):
                input_ids = next(test_iterator)["tokens"]
                input_ids = input_ids.to(device)
                with torch.no_grad():
                    logits, loss, ppl = step_model(input_ids)
                eval_losses.append(loss.item())
                eval_ppls.append(ppl.item())
            
            end_time_eval = time.perf_counter()
            
            avg_eval_loss = sum(eval_losses) / len(eval_losses)
            avg_eval_ppl = sum(eval_ppls) / len(eval_ppls)
            wandb.log({"eval/loss": avg_eval_loss, "eval/step_timer": (end_time_eval - start_time_eval), "eval/ppl": avg_eval_ppl}, step=step)
            print(f"Step {step}: eval loss = {avg_eval_loss:.4f}, eval ppl = {avg_eval_ppl:.4f}, eval time = {(end_time_eval - start_time_eval):.4f}s")
        
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