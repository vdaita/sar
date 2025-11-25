from torch import nn
import torch
from transformer_utils import Block
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, num_layers, vocab_size):
        super(Transformer, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Parameter(torch.randn(vocab_size, d_model), requires_grad=True)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.Sequential(*[
            Block(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

        self.proj = nn.Linear(d_model, vocab_size)

    def reconstruction_losses(self, *args):
        return {"encode_decode_loss": torch.zeros((1,)), "decode_encode_token_loss": torch.zeros((1,))}

    def forward(self, x):
        # embed tokens
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device)
        pos_emb = self.pos_emb(pos_ids)
        x = self.tok_emb[x]
        x += pos_emb

        if torch.isnan(x).any():
            raise ValueError("Transformer has NaNs after embedding")

        # next token
        x = self.layers(x)

        if torch.isnan(x).any():
            raise ValueError("Transformer has NaNs after layers")

        # project to tokens
        x = x @ self.tok_emb.T

        if torch.isnan(x).any():
            raise ValueError("Transformer has NaNs after final projection")

        x_tgt = x[:, :-1]
        loss = F.cross_entropy(x, x_tgt)
    
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
    # Test out small models
    d_model = 64
    n_heads = 4
    d_ff = 64
    max_seq_len = 32
    num_layers = 4
    vocab_size = 512

    batch_size = 32

    transformer = Transformer(
        d_model,
        n_heads,
        d_ff,
        max_seq_len,
        num_layers,
        vocab_size
    )

    random_input = torch.randint(vocab_size, (batch_size, max_seq_len))

    transformer = transformer(random_input)
    print("Finished calculating output from regular transformer")