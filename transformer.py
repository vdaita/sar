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
    def __init__(self, d_model):
        super(Attention, self).__init__()
        
        self.d_model = d_model
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
        
        attn_weights = F.softmax(scores, dim=-1) # (B, n_heads, T_q, T_k)
        
        attn_output = torch.matmul(attn_weights, v_proj) # (B, n_heads, T_q, D_head)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, D) # (B, T_q, D)
        
        output = self.wo(attn_output) # (B, T_q, D)
        
        return output

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, seq_len, n_steps):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.n_steps = n_steps
        
        self.attention = Attention(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = self.attention(x)
        x = self.ffn(x)
        for i in range(self.n_steps):
            x = self.attention(x)
            x = self.ffn(x)
        return x
        
class InvertibleDiffusionEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, seq_len, n_steps):
        super(InvertibleDiffusionEncoder, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.n_steps = n_steps
        
        self.encoder = Encoder(d_model // 2, n_heads // 2, d_ff // 2, seq_len, n_steps)
        self.compress = nn.Parameter(torch.randn(d_model, d_model // seq_len))
        self.o_proj = nn.Parameter(torch.randn(d_model // seq_len, d_model // seq_len))
        
    def forward(self, x, reverse=False):
        B, T, D = x.shape
        x1, x2 = x.chunk(2, dim=-1)
        if not reverse:
            y1 = x1 + self.encoder(x2)
            y2 = x2 - self.encoder(y1)
            y = torch.cat((y1, y2), dim=-1)
            
            # orthogonal compress
            A, _ = torch.qr(self.compress)
            y = torch.matmul(y, self.compress) # (B, seq_len, D // seq_len)
            y = y.reshape(B, -1) # (B, seq_len * D // seq_len = D)
            
            y1, y2 = y.chunk(2, dim=-1)
            y1 = y1 + torch.matmul(y1, self.o_proj)
            y2 = y2 - torch.matmul(y1, self.o_proj)
            y = torch.cat((y1, y2), dim=-1)
            return y
        else:
            y2 = ...
            y1 = ...

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

# actually needs to keep track of steps
# until the final state, the encoder takes (s1, h1) and predict (s2, h2)
# the last one: project dow
# then you reverse this project and go up and do whatever
# NEED TO FINISH FRIDAY NIGHT AND TRAIN UNTIL SATURDAY MORNING!!!!!!!
class InvertibleRNNEncoder(nn.Module):
    def __init__(self, d_model):
        super(InvertibleRNNEncoder, self).__init__()        
        self.state_init = nn.Parameter(torch.zeros(1, d_model))
        self.emb_token = nn.Parameter(torch.ones(1, d_model))
        
        # each state is (hidden_state, token_state)
        
        # define one transform that takes in (h, t) -> (h', t')
        # this can be defined with two works
        self.d_model = d_model
        
        self.hidden = ReversibleLinear(2 * d_model)
        self.compressor = nn.Parameter(torch.randn(2 * d_model, d_model))
        
    def forward(self, state=None, tokens=None):
        A, _ = torch.qr(self.compressor)
        if tokens:
            B, T, D = tokens.shape
            state_init = self.state_init.expand(B, 1) # (B, D)
            x = torch.cat([state_init, tokens[:, 0, :]], dim=-1)
            for _ in range(T - 1):
                x = self.hidden(x)


        if not reverse:
            x = torch.cat((state, token), dim=-1) # (B, 2 * d_model)
            x = self.hidden(x, reverse=False) # (B, 2 * d_model)
            x = torch.matmul(x, self.compressor) # (B, d_model)
            return x # new state
        else:
            x = torch.matmul(state, self.compressor.t()) # (B, 2 * d_model)
            x = self.hidden(x, reverse=True) # (B, 2 * d_model)
            state, token = x.chunk(2, dim=-1)
            return state, token
                
class RNNSARTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, num_vocab):
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.num_vocab = num_vocab


        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.tok_emb = nn.Embedding(num_vocab, d_model)

        self.encoder =

    def forward(self, x):
        # embed tokens


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.ffns = nn.ModuleList([
            
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None, tgt=None, x_pos=None):
        B, T = x.shape
        
        x_pos = x_pos if x_pos is not None else torch.arange(0, T, device=x.device).unsqueeze(0).repeat(B, 1).to(x.device)
        mask = mask if mask is not None else nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        
        x = self.tok_emb(x) + self.pos_emb(x_pos)
