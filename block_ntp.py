from torch import nn
import torch
from transformer_utils import Block
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class BlockNTPTransformerConfig:
    d_model: int
    num_heads: int
    d_ff: int
    
    max_seq_len: int
    vocab_size: int
    compress_seq_len: int
    
    decompress_num_layers: int
    num_layers: int

    
def generate_block_ntp_mask(length: int, compress_seq_len: int) -> torch.Tensor:
    mask = torch.zeros((length * 2, length * 2), dtype=torch.bool)
    for chunk_idx in range(length // compress_seq_len):
        if chunk_idx - 1 >= 0:
            mask[:(chunk_idx - 1) * compress_seq_len] = 1
        mask[length + (chunk_idx) * compress_seq_len : length + (chunk_idx + 1) * compress_seq_len] = 1
    return mask

def generate_ar_mask(length: int, compress_seq_len: int) -> torch.Tensor:
    mask = torch.zeros((length * 2, length * 2), dtype=torch.bool)
    for i in range(length):
        chunk_idx = i // compress_seq_len
        mask[:i] = 1
        mask[length + chunk_idx * compress_seq_len : length + i] = 1
    return mask
    

class BlockNTPTransformer(nn.Module):
    def __init__(self, config: BlockNTPTransformerConfig):
        super(BlockNTPTransformer, self).__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_ff = config.d_ff
        self.max_seq_len = config.max_seq_len
        self.vocab_size = config.vocab_size
        self.compress_seq_len = config.compress_seq_len
        self.decompress_num_layers = config.decompress_num_layers
        self.num_layers = config.num_layers

        self.tok_emb = nn.Parameter(torch.randn((self.vocab_size, self.d_model)), requires_grad=True)
        self.pos_emb = nn.Parameter(torch.randn((self.max_seq_len, self.d_model)), requires_grad=True)
        self.emb_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        self.decoder = nn.ModuleList([
            Block(self.d_model, self.num_heads, self.d_ff) for _ in range(self.decompress_num_layers)
        ])

        self.body = nn.ModuleList([
            Block(self.d_model, self.num_heads, self.d_ff) for _ in range(self.num_layers)
        ])
        
        self.mask_tokens = nn.Parameter(torch.randn(1, self.compress_seq_len, self.d_model), requires_grad=True)
        
    def forward(self, tok_ids):
        B, T = tok_ids.shape
        
        if T % self.compress_seq_len != 0:
            raise ValueError("Need to fix number of tokens to align with compress seq len")
        
        # concate side-by-side
        pos_ids = torch.arange(T, device=tok_ids.device)
        pos_ids = pos_ids.tile((1, 2))
        pos_ids = pos_ids.repeat(B, 1)
    
        
        emb_pos = self.pos_emb[pos_ids]
        
        mask_ids = self.mask_tokens.repeat(B, T // self.compress_seq_len, 1)
        emb_toks = self.tok_emb[tok_ids]
        emb_toks = torch.cat((emb_toks, mask_ids), dim=1)
        
        x = emb_toks + emb_pos    
        
        # transformer layer
        body_mask = generate_block_ntp_mask(T, self.compress_seq_len).to(x.device)
        for layer in self.body:
            x = layer(x, mask=body_mask)
        
        # decoder layer
        ar_mask = generate_ar_mask(T, self.compress_seq_len).to(x.device)
        for layer in self.decoder:
            x = layer(x, mask=ar_mask)
        
        x = x @ self.tok_emb.T
        
        # calculate the loss
        x = x[:, T + 1:, :]
        tgt_x = tok_ids[:, :-1]
        
        x = x.reshape(-1, self.vocab_size)
        tgt_x = tgt_x.reshape(-1)
        
        loss = F.cross_entropy(x, tgt_x)
        return {
            "outputs": {
                "decoded": x
            },
            "loss": {
                "ntp_loss": loss,
                "total": loss
            }
        }
        
if __name__ == "__main__":
    d_model = 64
    n_heads = 4
    d_ff = 256
    max_seq_len = 64
    vocab_size = 256
    compress_seq_len = 4
    decompress_num_layers = 1
    num_layers = 4
    
    config = BlockNTPTransformerConfig(
        d_model,
        n_heads,
        d_ff,
        
        max_seq_len,
        vocab_size,
        compress_seq_len,
        
        decompress_num_layers,
        num_layers
    )
    transformer = BlockNTPTransformer(config)
    
    batch_size = 4
    model_input = torch.randint(vocab_size, (batch_size, max_seq_len))
    
    transformer(model_input)
    print("Successfully processed forward pass for chunk transformer!")