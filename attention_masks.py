from pydantic.networks import IPvAnyAddress
from torch import nn, Tensor
import torch
from typing import Tuple
from dataclasses import dataclass

@dataclass
class PreprocessResult:
    mask: Tensor
    input_ids: Tensor
    labels: Tensor
    is_gist: Tensor

def preprocess_gist_glue(x: Tensor, gist_token: int, k: int) -> PreprocessResult:
    sar_result = preprocess_gist_sar(x, gist_token, k)
    B, seq_len_new = sar_result.input_ids.shape
    for token in range(seq_len_new):
        if token % (k + 1) != 0:
            next_gist = token + (k + 1) - (token % (k + 1))
            if next_gist < seq_len_new:
                sar_result.mask[:, token, next_gist] = 1 # note: the first dimension is batch

    return sar_result

def preprocess_gist_sar(x: Tensor, gist_token: int, k: int) -> PreprocessResult:
    B, T = x.shape

    assert T % k == 0, "must be divisible"
    num_blocks = T // k

    reg_tokens = torch.arange(T, device=x.device)

    num_gist_tokens = num_blocks
    gist_tokens = torch.arange(num_gist_tokens, device=x.device) * (k + 1)
    shift_tokens = torch.arange(start=1, end=num_gist_tokens + 1, device=x.device)
    shift_tokens = torch.repeat_interleave(shift_tokens, repeats=k)
    reg_tokens += shift_tokens

    new_tokens = torch.zeros((B, T + num_gist_tokens), device=x.device)
    new_tokens[:, reg_tokens] = x
    new_tokens[:, gist_tokens] = gist_token

    is_gist = torch.zeros((T + num_gist_tokens), device=x.device)
    is_gist[gist_tokens] = 1

    new_labels = new_tokens.clone()
    new_labels[new_labels == gist_token] = -100

    mask = torch.zeros((T + num_gist_tokens, T + num_gist_tokens), device=x.device)
    mask[:, gist_tokens] = 1
    mask = mask.bool()
    for token in reg_tokens:
        closest_shift_token = token - (token % (k + 1)) # remove the remainder
        mask[token, closest_shift_token:token + 1] = 1

    mask = torch.tril(mask)

    mask = mask.unsqueeze(0).repeat(B, 1, 1)

    return PreprocessResult(mask=mask, input_ids=new_tokens, labels=new_labels, is_gist=is_gist)

def preprocess_default_attention(x: Tensor) -> PreprocessResult:
    B, T = x.shape
    is_gist = torch.zeros(T, device=x.device)
    mask = torch.tril(torch.ones((T, T)))
    mask = mask.unsqueeze(0).repeat(B, 1, 1)
    return PreprocessResult(mask=mask, input_ids=x, is_gist=is_gist, labels=x)