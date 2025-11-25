import torch
from torch import Tensor, nn
import torch.nn.functional as F
from multi_loss_transformer import MultiLossTransformer
from transformers import GPT2Tokenizer
from train import get_model
import yaml
from typing import Union, Optional

import typer

app = typer.Typer()
TOKENIZER_NAME = "georgeyw/TinyStories-tokenizer-10k"

@app.command()
def test_model(model_path: str, model_conf_path: str):
    with open(model_conf_path, "r") as f:
        conf = yaml.safe_load(f)
    mode = conf.get("mode")
    compress_seq_len = conf.get("compress_seq_len")
    
    print("Mode: ", mode)

    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token # type: ignore

    model = get_model(model_conf_path)
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    
    model.eval()

    prompt = "Once upon a time"
    generation_length = 5

    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]

    for _ in range(generation_length):
        model_out = model(tokens)
        last_tokens = model_out["decoded"][:, -compress_seq_len:]
        last_tokens = torch.argmax(last_tokens)
        print("Adding last tokens: ", last_tokens)
        print("Decoded last few tokens: ", tokenizer.decode(last_tokens))
        tokens = torch.cat([tokens, last_tokens], dim=-1)
    print("baseline-style generation: ", tokenizer.decode(tokens))

        
    if mode == "multi_loss":
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
        embedded_chunks: Optional[Tensor] = None
        output_tokens = []

        for _ in range(generation_length):
            model_out = model(tokens, input_chunk_vectors=embedded_chunks)
            last_tokens = model_out["decoded"][:, -compress_seq_len:]
            last_tokens = torch.argmax(last_tokens)
            print("Adding last tokens: ", last_tokens)
            print("Decoded last few tokens: ", tokenizer.decode(last_tokens))
            
            last_head = model_out["head"][:, -1, :]
            if embedded_chunks:
                last_head = last_head.unsqueeze(1)
                embedded_chunks = torch.cat([
                    embedded_chunks,
                    last_head
                ])
            else:
                embedded_chunks = last_head.unsqueeze(1)
            output_tokens.extend(last_tokens.flatten().numpy().tolist())
        
        print("latent vector style generation: ", tokenizer.decode(output_tokens))

if __name__ == "__main__":
    app()