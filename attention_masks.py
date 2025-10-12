from torch import nn, Tensor
import torch

def generate_glue_attention_mask(seq_len: int, k: int) -> Tensor:
    """Generates a lower triangular attention mask for causal attention."""
    mask = torch.zeros((seq_len, seq_len))
    for i in range(seq_len):
        if i % k == 0:
            stride_positions = torch.arange(start=0, end=i + 1, step=k)
            mask[i, stride_positions] = 1
        else:
            last_stride = (i // k) * k
            next_stride: int = min(last_stride + k, seq_len - 1)
            intermediate_positions = torch.arange(start=last_stride, end=i + 1)
            mask[i, intermediate_positions] = 1
            mask[i, next_stride] = 1
    return mask

def generate_sar_attention_mask(seq_len: int, k: int) -> Tensor:
    """Generates a lower triangular attention mask for causal attention."""
    mask = torch.zeros((seq_len, seq_len))
    for i in range(seq_len):
        if i % k == 0:
            stride_positions = torch.arange(start=0, end=i + 1, step=k)
            mask[i, stride_positions] = 1
        else:
            last_stride = (i // k) * k
            intermediate_positions = torch.arange(start=last_stride, end=i + 1)
            mask[i, intermediate_positions] = 1
    return mask

def generate_overlap_attention_mask(seq_len: int, k: int, p_extend: float, extend_k: int) -> Tensor:
    
    mask = torch.zeros((seq_len, seq_len))
    k_elements = torch.arange(start=0, end=seq_len, step=k)
    mask[k_elements, k_elements] = 1

    for block_start in range(0, seq_len, k):
        extend_block = torch.rand(1).item() < p_extend
        for mod_k in range(k):
            i = block_start + mod_k
            if mod_k == 0:
                stride_positions = torch.arange(start=0, end=i + 1, step=k)
                mask[i, stride_positions] = 1
            elif mod_k < extend_k and extend_block:
                intermediate_positions = torch.arange(start=max(0, i - mod_k - k), end=i + 1)
                mask[i, intermediate_positions] = 1
            else:
                intermediate_positions = torch.arange(start=max(0, i - mod_k), end=i + 1)
                mask[i, intermediate_positions] = 1

    return mask

def generate_default_attention_mask(seq_len: int) -> Tensor:
    """Generates a standard lower triangular attention mask."""
    return torch.tril(torch.ones((seq_len, seq_len)))
