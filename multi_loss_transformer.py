import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from transformer_utils import Block

class MultiLossTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, vocab_size, compress_seq_len, compress_num_layers, num_layers, repr_loss_weight, latent_pred_loss_weight):
        super(MultiLossTransformer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.compress_seq_len = compress_seq_len
        self.compress_num_layers = compress_num_layers
        self.num_layers = num_layers

        self.repr_loss_weight = repr_loss_weight
        self.latent_pred_loss_weight = latent_pred_loss_weight

        self.tok_emb = nn.Parameter(torch.randn((self.vocab_size, self.d_model)), requires_grad=True)
        self.pos_emb = nn.Parameter(torch.randn((self.max_seq_len, self.d_model)), requires_grad=True)
        self.emb_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.encoder = nn.Sequential(*[
            Block(self.d_model, self.n_heads, self.d_ff) for _ in range(self.compress_num_layers)
        ])
        self.body = nn.Sequential(*[
            Block(self.d_model, self.n_heads, self.d_ff) for _ in range(self.num_layers)
        ])
        self.head = nn.Sequential(*[
            Block(self.d_model, self.n_heads, self.d_ff) for _ in range(self.compress_num_layers)
        ])
        self.decoder = nn.Sequential(*[
            Block(self.d_model, self.n_heads, self.d_ff) for _ in range(self.compress_num_layers)
        ])

    def forward(self, tok_ids):
        B, T = tok_ids.shape
        if T % self.compress_seq_len != 0:
            raise ValueError("Need to fix number of tokens to align with compress seq len")

        token_chunks = tok_ids.reshape(B, T // self.compress_seq_len, self.compress_seq_len)

        pos_ids = torch.arange(T, device=tok_ids.device)
        pos_ids = pos_ids.unsqueeze(0)
        pos_ids = pos_ids.repeat(B, 1)
        
        emb_toks = self.tok_emb[tok_ids]
        emb_pos = self.pos_emb[pos_ids]

        x = emb_toks + emb_pos
        emb_token_chunks = x.reshape(B, T // self.compress_seq_len, self.compress_seq_len, self.d_model)

        # encoder layer
        enc_x = emb_token_chunks.reshape(B * T // self.compress_seq_len, self.compress_seq_len, self.d_model)
        emb_token = self.emb_token.repeat(B * T // self.compress_seq_len, 1, 1)
        enc_x = torch.cat((enc_x, emb_token), dim=1)
        enc_x = self.encoder(enc_x)
        enc_x = enc_x[:, -1, :]
        enc_x = enc_x.reshape(B, T // self.compress_seq_len, self.d_model)

        # transformer layer
        x = self.body(enc_x)
        x = x[:, :-1, :]
        x = x.reshape(-1, 1, self.d_model)

        # decoder layer
        emb_token_chunks = emb_token_chunks[:, 1:, :, :]
        emb_token_chunks = emb_token_chunks.reshape(-1, self.compress_seq_len, self.d_model)
        dec_x = torch.cat([x, emb_token_chunks], dim=1)
        dec_x = self.decoder(dec_x)
        dec_x = dec_x[:, :-1, :] 
        dec_x = dec_x @ self.tok_emb.T

        # calculate the loss
        dec_x = dec_x.reshape(-1, self.vocab_size)
        token_chunks = token_chunks[:, 1:, :]
        token_chunks = token_chunks.reshape(-1)

        decoder_loss = F.cross_entropy(dec_x, token_chunks)

        # come up with representation and calculate loss
        x = x.reshape(B, -1, self.d_model)
        head_x = self.head(x.detach()) # this is the representation of everything but the last chunk, it corresponds to the encoded tokens shifted right
        target_repr_x = enc_x[:, 1:, :]
        repr_loss = F.mse_loss(head_x, target_repr_x) # target_repr_x comes from just the encoder, and head_x comes just from the head

        # add the next token prediction loss as well
        head_x = self.body(head_x)
        head_x = head_x[:, :-1, :]
        head_x = head_x.reshape(-1, 1, self.d_model)
    
        emb_token_chunks = emb_token_chunks.reshape(B, -1, self.compress_seq_len, self.d_model)
        emb_token_chunks = emb_token_chunks[:, 1:, :, :] # we've already shifted by 1
        emb_token_chunks = emb_token_chunks.reshape(-1, self.compress_seq_len, self.d_model)

        head_dec_x = torch.cat([head_x, emb_token_chunks], dim=1)
        head_dec_x = self.decoder(head_dec_x)
        head_dec_x = head_dec_x[:, :-1, :]
        head_dec_x = head_dec_x @ self.tok_emb.T
        head_dec_x = head_dec_x.reshape(-1, self.vocab_size)

        token_chunks = token_chunks.reshape(B, -1, self.compress_seq_len)
        token_chunks = token_chunks[:, 1:, :]
        token_chunks = token_chunks.reshape(-1)

        head_decoder_loss = F.cross_entropy(head_dec_x, token_chunks)

        total = decoder_loss + self.latent_pred_loss_weight * head_decoder_loss + self.repr_loss_weight * repr_loss 

        return {
            "outputs": {
                "decoder": dec_x, 
                "head": head_dec_x
            }, 
            "loss": {
                "decoded": decoder_loss, 
                "head_repr": repr_loss, 
                "head_token": head_decoder_loss, 
                "total": total
            }
        } 

if __name__ == "__main__":
    d_model = 64
    n_heads = 4
    d_ff = 256
    max_seq_len = 64
    vocab_size = 256
    compress_seq_len = 4
    compress_num_layers = 1
    num_layers = 4

    repr_loss_coeff = 0.5
    head_loss_coeff = 0.5

    transformer = MultiLossTransformer(
        d_model,
        n_heads,
        d_ff,
        max_seq_len,
        vocab_size,
        compress_seq_len,
        compress_num_layers,
        num_layers,
        repr_loss_coeff,
        head_loss_coeff
    )

    batch_size = 4

    model_input = torch.randint(vocab_size, (batch_size, max_seq_len))
    transformer(model_input)

    print("Successfully processed forward for multiloss transformer!")