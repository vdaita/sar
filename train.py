import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
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

from base_chunk_transformer import ChunkTransformer
from invertible_sar_transformer import InvertibleSARTransformer
from multi_loss_transformer import MultiLossTransformer

app = typer.Typer()
TOKENIZER_NAME = "georgeyw/TinyStories-tokenizer-10k"
DATASET_NAME = "roneneldan/TinyStories"

@app.command()
def train(conf_path: str):
    with open(conf_path, "r") as f:
        conf = yaml.safe_load(f)

    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    if not tokenizer:
        raise ValueError("Tokenizer doesn't exist")

    tokenizer.pad_token = tokenizer.eos_token # type: ignore

    mode: str = conf.get("mode")

    checkpoint_dir: str = conf.get("checkpoint_dir")
    os.makedirs(checkpoint_dir, exist_ok=True)

    d_model: int = conf.get("d_model")
    n_heads: int = conf.get("n_heads")
    d_ff: int = conf.get("d_ff")
    max_seq_len: int = conf.get("max_seq_len")
    vocab_size: int = conf.get("vocab_size")
    compress_seq_len: int = conf.get("compress_seq_len")
    compress_num_layers: int = conf.get("compress_num_layers")
    num_layers: int = conf.get("num_layers")
    repr_weight: float = conf.get("repr_weight") # only for the multiloss version

    batch_size: int = conf.get("batch_size")
    eval_num_batches = conf.get("eval_num_batches")
    
    num_train_steps: int = conf.get("num_train_steps")
    eval_every: int = conf.get("eval_every")
    save_every: int = conf.get("save_every")

    lr: float = conf.get("lr")

    unix_millis = int(round(time.time() * 1000))
    model_folder = f"{checkpoint_dir}/model-{mode}-{unix_millis}/"
    name = f"{mode}-arch={mode}-compress_seq_len={compress_seq_len}-compress_n_layer={compress_num_layers}-bs={batch_size}-eval_num_batches={eval_num_batches}-lr={lr}-embed={d_model}-ffn={d_ff}-layer={num_layers}-head={n_heads}-repr_loss={repr_weight}-timestamp={unix_millis}"

    os.makedirs(model_folder, exist_ok=False)
    with open(os.path.join(model_folder, "config.yaml"), "w") as f:
        yaml.safe_dump(conf, f)

    if mode == "base":
        model = ChunkTransformer(
            d_model,
            n_heads,
            d_ff,
            max_seq_len,
            vocab_size,
            compress_seq_len,
            compress_num_layers,
            num_layers
        )
    elif mode == "invertible":
        model = InvertibleSARTransformer(
            d_model,
            n_heads,
            d_ff,
            max_seq_len,
            vocab_size,
            compress_seq_len,
            compress_num_layers,
            num_layers
        )
    elif mode == "multi_loss":
        model = MultiLossTransformer(
            d_model,
            n_heads,
            d_ff,
            max_seq_len,
            vocab_size,
            compress_seq_len,
            compress_num_layers,
            num_layers,
            repr_weight
        )
    else:
        raise ValueError("You selected an invalid mode.")

    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    device = torch.device('cpu')
    if torch.mps.is_available():
        device = torch.device('mps')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = model.to(device)
    train_dataset = load_dataset(DATASET_NAME, split="train")
    test_dataset = load_dataset(DATASET_NAME, split="validation")

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

    for step in tqdm(range(num_train_steps), desc="Train step"):
        start_time_train = time.perf_counter()
        model.train()

        input_ids = next(train_iterator)["tokens"]
        input_ids = input_ids.to(device)
        model_out = model(input_ids)
        optimizer.zero_grad()
        model_out["loss"]["total"].backward()
        optimizer.step()

        end_time_train = time.perf_counter()

        train_report_dict = {
            "train/step_time": end_time_train - start_time_train
        }
        for loss_name in model_out["loss"].keys():
            train_report_dict[
                f"train/loss/{loss_name}"
            ] = model_out["loss"][loss_name].item()
        wandb.log(train_report_dict, step=step)
        
        if step % 50 == 0:
            print(f"Step {step}: ", train_report_dict)
        
        if step % eval_every == 0:
            start_time_eval = time.perf_counter()
            model.eval()

            eval_report_dict = {}
            for _ in range(eval_num_batches):
                input_ids = next(test_iterator)["tokens"]
                input_ids = input_ids.to(device)
                model_out = model(input_ids)
                for loss_name in model_out["loss"].keys():
                    if not f"train/loss/{loss_name}" in eval_report_dict:
                        eval_report_dict[f"train/loss/{loss_name}"] = []
                    eval_report_dict[
                        f"train/loss/{loss_name}"
                    ].append(model_out["loss"][loss_name].item())
            
            end_time_eval = time.perf_counter()

            eval_report_dict_averages = {}
            for loss_name in eval_report_dict:
                eval_report_dict_averages[loss_name] = sum(eval_report_dict[loss_name]) / len(eval_report_dict[loss_name])
            eval_report_dict_averages["eval/step_timer"] = end_time_eval - start_time_eval

            wandb.log(eval_report_dict, step=step)
            print(f"Step {step}: ", eval_report_dict_averages)
        
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
