from transformer_utils import SwiGLU, Block
import torch
from torch.nn.utils.parametrizations import orthogonal
from torch import nn
import torch.nn.functional as F

class InvertibleEncoder(nn.Module):
    def __init__(self, d_model, d_ff, chunk_size, num_layers):
        super(InvertibleEncoder, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.chunk_size = chunk_size
        self.num_layers = num_layers

        self.ffn_1 = nn.Sequential(*[
            SwiGLU(self.d_model * self.chunk_size // 2, self.d_ff // 2) for _ in range(num_layers)
        ])
        self.ffn_2 = nn.Sequential(*[
            SwiGLU(self.d_model * self.chunk_size // 2, self.d_ff // 2) for _ in range(num_layers)
        ])

        self.compress_proj = nn.Parameter(torch.randn((self.d_model * self.chunk_size, self.d_model)), requires_grad=True)
        orthogonal(self, "compress_proj")
    
    def forward(self, x, reverse=False):
        if not reverse:
            B, T, D = x.shape
            if T % self.chunk_size != 0:
                raise ValueError("T needs to be divisible by chunk_size")
            num_chunks = T // self.chunk_size
            x = x.reshape(-1, self.chunk_size * D)
            x1, x2 = x.chunk(2, dim=-1)
            y1 = x1 + self.ffn_1(x2)
            y2 = x2 + self.ffn_2(y1)
            y = torch.cat((y1, y2), dim=-1)
            y = y.reshape(-1, self.chunk_size * D)
            y = y @ self.compress_proj
            y = y.reshape(B, num_chunks, D)
            return y
        else:
            y = x
            B, num_chunks, D = y.shape
            y = y @ self.compress_proj.T
            y1, y2 = y.chunk(2, dim=-1)
            x2 = y2 - self.ffn_1(y1)
            x1 = y1 - self.ffn_2(x2)
            x = torch.cat((x1, x2), dim=-1)
            x = x.reshape(B, num_chunks * self.chunk_size, D)
            return x

class InvertibleSARTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, num_layers, vocab_size, compress_seq_len, compress_num_layers):
        super(InvertibleSARTransformer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.compress_seq_len = compress_seq_len
        self.compress_num_layers = compress_num_layers
        self.num_layers = num_layers

        self.tok_emb = nn.Parameter(torch.randn((self.vocab_size, self.d_model)), requires_grad=True)
        self.pos_emb = nn.Parameter(torch.randn((self.max_seq_len, self.d_model)), requires_grad=True)
        self.emb_proj = nn.Parameter(torch.randn((self.d_model * 2, self.d_model)), requires_grad=True)

        self.encoder = InvertibleEncoder(self.d_model, self.d_ff, self.compress_seq_len, self.compress_num_layers)
        self.transformer_blocks = nn.Sequential(*[
            Block(self.d_model, self.n_heads, self.d_ff) for _ in range(self.num_layers)
        ])

        orthogonal(self, "tok_emb")
        orthogonal(self, "pos_emb")
        orthogonal(self, "emb_proj")

    def encode(self, tok_ids, pos_ids):
        B, T = tok_ids.shape
        if T % self.compress_seq_len != 0:
            raise ValueError("Doesn't support part of a compressed sequence yet")

        emb_toks = self.tok_emb[tok_ids]
        emb_pos = self.pos_emb[pos_ids]
        x = torch.cat([emb_toks, emb_pos], dim=-1)
        x = x @ self.emb_proj

        if torch.isnan(x).any():
            raise ValueError("SARTransformer with NaNs after embedding")

        x = self.encoder(x)
        return x

    def decode(self, embedding):
        B, T, D = embedding.shape

        x = self.encoder(embedding, reverse=True)
        x = x @ self.emb_proj.T 
        
        token_x, pos_x = x.chunk(2, dim=-1)
        out_tokens = token_x @ self.tok_emb.T
        out_positions = pos_x @ self.pos_emb.T
        return out_tokens, out_positions

    def forward(self, tok_ids):
        # Reconstruction error is saying comparing E(D(x)) to x
        # embed tokens
        B, T = tok_ids.shape
        pos_ids = torch.arange(T, device=tok_ids.device)
        pos_ids = pos_ids.unsqueeze(0)
        pos_ids = pos_ids.repeat(B, 1)

        encoded_chunks = self.encode(tok_ids, pos_ids)
        next_chunks = self.transformer_blocks(encoded_chunks)
        decoded_tokens, decoded_positions = self.decode(next_chunks)

        enc_decoded_tokens, enc_decoded_positions = self.decode(encoded_chunks)
        
        decoded_tokens = decoded_tokens.reshape(B, T // self.compress_seq_len, self.compress_seq_len, self.vocab_size)
        target_tokens = tok_ids.reshape(B, T // self.compress_seq_len, self.compress_seq_len)
        target_tokens = target_tokens[:, :-1, :]
        decoded_tokens = decoded_tokens[:, 1:, :, :]
        target_tokens = target_tokens.reshape(-1)
        decoded_tokens = decoded_tokens.reshape(-1, self.vocab_size)
        
        ce_loss = F.cross_entropy(decoded_tokens, target_tokens)

        enc_decoded_tokens = enc_decoded_tokens.reshape(-1, self.vocab_size)
        tok_ids = tok_ids.reshape(-1)
        reconstruction_token_loss = F.cross_entropy(enc_decoded_tokens, tok_ids)
        reconstruction_hidden_dim_loss = F.mse_loss(encoded_chunks[:, 1:, :], next_chunks[:, :-1, :])

        return {
            "outputs": {
                "decoded": decoded_tokens
            },
            "loss": {
                "ce_loss": ce_loss,
                "reconstruction_token_loss": reconstruction_token_loss,
                "reconstruction_hidden_dim_loss": reconstruction_hidden_dim_loss,
                "total": ce_loss
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
    compress_seq_len = 4
    compress_num_layers = 1
    batch_size = 32

    transformer = InvertibleSARTransformer(
        d_model,
        n_heads,
        d_ff,
        max_seq_len,
        num_layers,
        vocab_size,
        compress_seq_len,
        compress_num_layers
    )

    random_input = torch.randint(vocab_size, (batch_size, max_seq_len))
    transformer = transformer(random_input)
    print("Finished calculating output from SAR transformer")