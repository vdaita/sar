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
from base_transformer import Transformer
from block_ntp import BlockNTPTransformer, BlockNTPTransformerConfig


app = typer.Typer()
TOKENIZER_NAME = "georgeyw/TinyStories-tokenizer-10k"
DATASET_NAME = "roneneldan/TinyStories"

torch.multiprocessing.set_sharing_strategy("file_system")

def get_model(conf):
    mode: str = conf.get("mode")

    model_conf = conf.get("model_conf")
    
    if not model_conf:
        print("model_conf not found in config, using top-level config values.")
        model_conf = conf

    d_model: int = model_conf.get("d_model")
    n_heads: int = model_conf.get("n_heads")
    d_ff: int = model_conf.get("d_ff")
    max_seq_len: int = model_conf.get("max_seq_len")
    vocab_size: int = model_conf.get("vocab_size")
    compress_seq_len: int = model_conf.get("compress_seq_len")
    compress_num_layers: int = model_conf.get("compress_num_layers")
    num_layers: int = model_conf.get("num_layers")

    repr_loss_weight: float = model_conf.get("repr_loss_weight") # only for the multiloss version
    latent_loss_weight: float = model_conf.get("latent_loss_weight")

    if mode == "base":
        model = Transformer(
            d_model,
            n_heads,
            d_ff,
            max_seq_len,
            num_layers,
            vocab_size
        )
    elif mode == "block_ntp":
        config = BlockNTPTransformerConfig(
            **model_conf
        )
        model = BlockNTPTransformer(config)
    elif mode == "base_chunk":
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
    elif mode == "invertible_chunk":
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
    elif mode == "multi_loss_chunk":
        model = MultiLossTransformer(
            d_model,
            n_heads,
            d_ff,
            max_seq_len,
            vocab_size,
            compress_seq_len,
            compress_num_layers,
            num_layers,
            repr_loss_weight,
            latent_loss_weight
        )
    else:
        raise ValueError("You selected an invalid mode.")

    return model

def flatten_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_dict(v))
        else:
            out[k] = v
    return out

def generate_name_from_config(conf, exclude_from_name=[]) -> str:
    conf_name_segments = []
    flat_conf = flatten_dict(conf)
    for key in flat_conf:
        if key in exclude_from_name:
            continue
        conf_name_segments.append(f"{key}={flat_conf[key]}-")
    conf_name = "".join(conf_name_segments)
    return conf_name

@app.command()
def train(conf_path: str):
    with open(conf_path, "r") as f:
        conf = yaml.safe_load(f)

    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    if not tokenizer:
        raise ValueError("Tokenizer doesn't exist")

    tokenizer.pad_token = tokenizer.eos_token # type: ignore

    checkpoint_dir: str = conf.get("checkpoint_dir")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    batch_size: int = conf.get("batch_size")
    eval_num_batches = conf.get("eval_num_batches")
    
    num_train_steps: int = conf.get("num_train_steps")
    eval_every: int = conf.get("eval_every")
    save_every: int = conf.get("save_every")
    mode = conf.get("mode")

    max_seq_len = conf.get("max_seq_len")
    if not max_seq_len:
        max_seq_len = conf.get("model_conf").get("max_seq_len")
    
    if not max_seq_len:
        raise ValueError("You should set max_seq_len in either the base conf or the model_conf sub-variable")

    lr: float = conf.get("lr")

    unix_millis = int(round(time.time() * 1000))
    model_folder = f"{checkpoint_dir}/model-{mode}-{unix_millis}/"
    name = generate_name_from_config(conf, exclude_from_name=["checkpoint_dir", "num_train_steps", "eval_every", "save_every"])

    os.makedirs(model_folder, exist_ok=False)
    with open(os.path.join(model_folder, "config.yaml"), "w") as f:
        yaml.safe_dump(conf, f)
        
    model = get_model(conf)

    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device {device}")

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
                    if f"eval/loss/{loss_name}" not in eval_report_dict:
                        eval_report_dict[f"eval/loss/{loss_name}"] = []
                    eval_report_dict[
                        f"eval/loss/{loss_name}"
                    ].append(model_out["loss"][loss_name].item())
            
            end_time_eval = time.perf_counter()

            eval_report_dict_averages = {}
            for loss_name in eval_report_dict:
                eval_report_dict_averages[loss_name] = sum(eval_report_dict[loss_name]) / len(eval_report_dict[loss_name])
            eval_report_dict_averages["eval/step_timer"] = end_time_eval - start_time_eval

            wandb.log(eval_report_dict_averages, step=step)
            print(f"Step {step}: ", eval_report_dict_averages)
        
        if step % save_every == 0:
            save_path = os.path.join(model_folder, f"step-{step}")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint at step {step} to {save_path}")
        
    final_save_path = os.path.join(model_folder, "final")
    torch.save(model.state_dict(), final_save_path)
    print(f"Saved final model checkpoint to {final_save_path}")
    wandb.finish()

if __name__ == "__main__":
    app()
