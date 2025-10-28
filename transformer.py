"""
Implements basic transformer architecture for sequence modelling tasks.
"""
import torch
from torch import nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super(SwiGLU, self).__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.gate = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x):
       return self.down(torch.nn.functional.gelu(self.up(x)) * self.gate(x))

class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(Attention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.scale = d_model ** -0.5

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, T_q, D = q.shape
        B, T_k, D = k.shape
        B, T_v, D = v.shape

        q_proj = self.wq(q).view(B, T_q, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, n_heads, T_q, D_head)
        k_proj = self.wk(k).view(B, T_k, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, n_heads, T_k, D_head)
        v_proj = self.wv(v).view(B, T_v, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, n_heads, T_v, D_head)

        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale # (B, n_heads, T_q, T_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1) # (1, 1, T_q, T_k)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1) # (B, 1, T_q, T_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        else:
            mask = torch.tril(
                torch.ones((T_q, T_q), device=q.device), diagonal=-1
            )
            mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, 1, 1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1) # (B, n_heads, T_q, T_k)

        attn_output = torch.matmul(attn_weights, v_proj) # (B, n_heads, T_q, D_head)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, D) # (B, T_q, D)

        output = self.wo(attn_output) # (B, T_q, D)

        return output

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
        x = self.norm1(x)
        x = self.attention(x, mask=None)
        x = self.norm2(x)
        x = self.ffn(x)
        return x

class ReversibleLinear(nn.Module):
    def __init__(self, d_model):
        super(ReversibleLinear, self).__init__()
        self.f = nn.Linear(d_model // 2, d_model // 2)
        self.g = nn.Linear(d_model // 2, d_model // 2)

    def forward(self, x, reverse=False):
        if not reverse:
            x1, x2 = x.chunk(2, dim=-1)
            y1 = x1 + self.f(x2)
            y2 = x2 + self.g(y1)
            return torch.cat((y1, y2), dim=-1)
        else:
            y1, y2 = x.chunk(2, dim=-1)
            x2 = y2 - self.g(y1)
            x1 = y1 - self.f(x2)
            return torch.cat((x1, x2), dim=-1)

class ReversibleTransformer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, chunk_size):
        super(ReversibleTransformer, self).__init__()

        self.d_model = d_model
        self.chunk_size = chunk_size
        self.half_dim = d_model // 2

        self.block_1 = nn.ModuleList(
            [
                Attention(self.half_dim, n_heads=n_heads // 2),
                SwiGLU(self.half_dim, d_ff // 2)
            ]
        )
        self.block_2 = nn.ModuleList(
            [
                Attention(self.half_dim, n_heads=n_heads // 2),
                SwiGLU(self.half_dim, d_ff // 2)
            ]
        )

        self.compressor = nn.Parameter(torch.ones(self.chunk_size * self.d_model, self.d_model), requires_grad=True)

    def forward(self, x, reverse=False):
        if reverse:
            B, _, D = x.shape
            A, _ = torch.qr(self.compressor)

            x1, x2 = x.chunk(2, dim=-1)
            y1 = x1 + self.block_1(x2)
            y2 = x2 + self.block_2(y1)
            y = torch.cat((y1, y2), dim=-1)
            y = y.reshape(B, self.chunk_size * D)
            y = y @ A

            return y
        else:
            B, D = x.shape
            A, _ = torch.qr(self.compressor)

            y = x
            y: torch.Tensor = y @ A.T
            y = y.reshape(B, self.chunk_size, D)
            y1, y2 = y.chunk(2, dim=-1)
            
            x2 = y2 - self.block_1(y1)
            x1 = y1 - self.block_2(x2)
            return torch.cat((x1, x2), dim=-1)

class SARTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, num_layers, vocab_size, compress_seq_len):
        super(SARTransformer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.num_vocab = vocab_size
        self.compress_seq_len = compress_seq_len
        self.num_layers = num_layers

        self.pos_emb = nn.Embedding(vocab_size, d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.encoder = ReversibleTransformer(d_model, d_ff, n_heads, compress_seq_len)
        self.layers = nn.ModuleList([
            Block(d_model, n_heads, d_ff) for _ in range(num_layers)
        ])

        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # embed tokens
        B, T = x.shape

        if T % self.compress_seq_len != 0:
            raise ValueError("Doesn't support part of a compressed sequence yet")

        pos_ids = torch.arange(T, device=x.device)
        pos_emb = self.pos_emb(pos_ids)
        x = self.tok_emb(x) + pos_emb

        # encode tokens
        x = x.reshape(-1, self.compress_seq_len, self.d_model)
        x = self.encoder(x, reverse=False)
        x = x.reshape(-1, T // self.compress_seq_len, self.d_model)

        # next sequence
        x = self.layers(x)

        # decompress all tokens
        x = x.reshape(-1, self.d_model)
        x = self.encoder(x, reverse=True)
        x = x.reshape(-1, T, self.d_model)

        # convert decompressed tokens to actual tokens
        x = self.proj(x)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, vocab_size, num_layers):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            Block(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # embed tokens
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device)
        pos_emb = self.pos_emb(pos_ids)
        x = self.tok_emb(x) + pos_emb

        # next token
        x = self.layers(x)

        # project to tokens
        x = self.proj(x)

        return x