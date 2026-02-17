import torch
import torch.nn as nn
import copy
import math


def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_positional_encoding(L, d_model, n=10000):
    pe = torch.zeros(L, d_model)
    position = torch.arange(0, L, dtype=torch.float).unsqueeze(1)  # (L,1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(n) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)  # even indeces
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indeces
    return pe.unsqueeze(0)  # (1, L, d_model)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=heads, batch_first=True, dropout=dropout
        )
        self.ff = FeedForward(d_model, d_ff=4 * d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # NORM
        x1 = self.ln1(x)
        # ATTENTION + RESIDUAL CONNECTION
        x = x + self.dropout(self.attn(x1, x1, x1)[0])
        # NORM
        x2 = self.ln2(x)
        # FEED FORWARD  + RESIDUAL CONNECTION
        x = x + self.dropout(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, heads, batch_first=True, dropout=dropout
        )
        self.ff = FeedForward(d_model, d_ff=4 * d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, memory_key_padding_mask=None):
        # Cross-attn (pre-norm)
        x_res = x
        x = self.ln_1(x)
        ca, _ = self.cross_attn(
            x, memory, memory, key_padding_mask=memory_key_padding_mask
        )
        x = x_res + self.dropout(ca)

        # FFN (pre-norm)
        x_res = x
        x = self.ln_2(x)
        x = x_res + self.dropout(self.ff(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        x = tgt
        for i in range(self.N):
            x = self.layers[i](
                x, memory, memory_key_padding_mask=memory_key_padding_mask
            )
        return x


class PINNsformer(nn.Module):
    def __init__(
        self, d_model, d_hidden, d_emb_input, d_final, N, heads, dropout, Pos_src
    ):
        super().__init__()
        self.embed_input = nn.Sequential(
            nn.Linear(d_emb_input, d_model), nn.GELU(), nn.Dropout(dropout)
        )

        self.encoder = Encoder(d_model, N, heads, dropout)
        self.decoder = Decoder(d_model, N, heads, dropout)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_final),
        )
        self.pos_src = Pos_src  # positional encoding

    def forward(self, src_state, src_variations):
        # src_state: (batch, 6)
        # src_variations: (batch, 16, 3)
        B, L_src, _ = src_variations.shape
        state_rep = src_state.unsqueeze(1).expand(B, L_src, 6)  # (B, 15, 6)

        inp = torch.cat([state_rep, src_variations], dim=2)  # (B, 15, 9)
        # 15 tokens each one composed of
        # initial state + control (for that time instant)
        src = self.embed_input(inp)  # (batch, 16, d_model)

        # Positional Encoding
        src = src + self.pos_src

        # Encoder
        e_outputs = self.encoder(src)

        d_output = self.decoder(src, e_outputs)

        output = self.linear_out(d_output)
        return output
