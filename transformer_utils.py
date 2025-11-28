"""
Implements basic transformer architecture for sequence modelling tasks.
"""
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn.utils.parametrize as parameterize
from torch.nn.utils.parametrizations import orthogonal

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super(SwiGLU, self).__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.gate = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x):
       return self.down(torch.nn.functional.gelu(self.up(x)) * self.gate(x))

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, use_kv_cache=False):
        super(Attention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.scale = d_model ** -0.5

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.use_kv_cache = use_kv_cache

        if self.use_kv_cache:
            raise NotImplementedError("Haven't implemented KV cache yet")

    def forward(self, x, mask=None):
        B, T_q, D = x.shape
        T_k, T_v = T_q, T_q

        q_proj = self.wq(x).view(B, T_q, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, n_heads, T_q, D_head)
        k_proj = self.wk(x).view(B, T_k, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, n_heads, T_k, D_head)
        v_proj = self.wv(x).view(B, T_v, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, n_heads, T_v, D_head)

        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale # (B, n_heads, T_q, T_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1) # (1, 1, T_q, T_k)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1) # (B, 1, T_q, T_k)
            elif mask.dim() == 4:
                mask = mask
            else:
                raise ValueError("Mask dimension is invalid.")
            scores = scores.masked_fill(mask == 0, float('-inf'))
        else:
            mask = torch.tril(
                torch.ones((T_q, T_q), device=x.device), diagonal=0
            )
            mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(0)
            mask = mask.expand(B, self.n_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1) # (B, n_heads, T_q, T_k)
        attn_output = torch.matmul(attn_weights, v_proj) # (B, n_heads, T_q, D_head)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, D) # (B, T_q, D)
        output = self.wo(attn_output) # (B, T_q, D)

        return output

def check_nan(x, step_name):
    if torch.isnan(x).any():
        raise ValueError(f"NaN after step {step_name}")
    return x

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(Block, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.attention = Attention(d_model, n_heads)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = check_nan(x, x + self.attention(self.norm1(x), mask=mask))
        x = check_nan(x, x + self.ffn(self.norm2(x)))
        return x