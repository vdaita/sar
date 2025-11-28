from torch import nn
import torch
from transformer_utils import Block
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class BlockNTPTransformerConfig:
    d_model: int
    n_heads: int
    d_ff: int
    
    max_seq_len: int
    vocab_size: int
    compress_seq_len: int
    
    decompress_num_layers: int
    num_layers: int

    
def generate_block_ntp_mask(length: int, compress_seq_len: int) -> torch.Tensor:
    mask = torch.zeros((length * 2, length * 2), dtype=torch.bool)
    for chunk_idx in range(length // compress_seq_len):
        mask[:chunk_idx * compress_seq_len] = 1 # these are the ground truth tokens
        mask[length + (chunk_idx) * compress_seq_len : length + (chunk_idx + 1) * compress_seq_len] = 1 # these are the mask tokens being predicted
        # each mask token position is predicting the value corresponding to length + position
    return mask

def generate_ar_mask(length: int, compress_seq_len: int) -> torch.Tensor:
    mask = torch.zeros((length * 2, length * 2), dtype=torch.bool)
    for i in range(length):
        mask[:i] = 1 # everything from the preceding token gets seen
        mask[length + i] = 1 # also see the current mask token
    return mask
    

class BlockNTPTransformer(nn.Module):
    def __init__(self, config: BlockNTPTransformerConfig):
        super(BlockNTPTransformer, self).__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_ff = config.d_ff
        self.max_seq_len = config.max_seq_len
        self.vocab_size = config.vocab_size
        self.compress_seq_len = config.compress_seq_len
        self.decompress_num_layers = config.decompress_num_layers
        self.num_layers = config.num_layers

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        self.pos_emb = nn.Parameter(torch.randn((self.max_seq_len, self.d_model)), requires_grad=True)

        self.decoder = nn.ModuleList([
            Block(self.d_model, self.n_heads, self.d_ff) for _ in range(self.decompress_num_layers)
        ])

        self.body = nn.ModuleList([
            Block(self.d_model, self.n_heads, self.d_ff) for _ in range(self.num_layers)
        ])

        self.ln = nn.LayerNorm(self.d_model)
        self.proj = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.proj.weight = self.tok_emb.weight
        
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
        emb_toks = self.tok_emb(tok_ids)
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

        x = self.ln(x)
        x = self.proj(x)
        
        # calculate the loss
        full_loss = F.cross_entropy(x[:, T:, :].reshape(-1, self.vocab_size), tok_ids.reshape(-1)) # this includes the next first token 
        ntp_loss = F.cross_entropy(x[:, T + 1:, :].reshape(-1, self.vocab_size), tok_ids[:, 1:].reshape(-1)) # i should probably calculate this more efficiently...

        return {
            "outputs": {
                "decoded": x
            },
            "loss": {
                "ntp_loss": ntp_loss,
                "full_ntp_loss": full_loss,
                "total": ntp_loss
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