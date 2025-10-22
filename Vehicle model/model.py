import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn as nn
from typing import Optional
from torch import Tensor


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    embed_dim: output dimension for each position
    length: length of the sequence (L)
    return: pos_emb: (L, D)
    """
    
    assert embed_dim % 2 == 0

    # 计算 omega
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(length, dtype=np.float32).reshape(-1)  # (L,)

    out = np.einsum('m,d->md', pos, omega)  # (L, D/2), outer product

    emb_sin = np.sin(out)  # (L, D/2)
    emb_cos = np.cos(out)  # (L, D/2)

    pos_emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (L, D)
    return pos_emb

class DecoderOnlyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderOnlyTransformerBlock, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x, padding_mask=None):

        causal_mask = _generate_square_subsequent_mask(x.shape[1])

        attn_output, _ = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask = causal_mask, is_causal=True)
        x = self.norm1(x + attn_output)  


        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)  

        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self,
        in_channels,
        out_channels, 
        d_model, 
        n_heads, 
        num_layers, 
        d_ff, 
        max_len=512, 
        dropout=0.1):
        super(DecoderOnlyTransformer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.max_len = max_len

        self.x_embedder = nn.Linear(in_channels, d_model)


        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)

        self.blocks = nn.ModuleList([
            DecoderOnlyTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        

        self.final_layer = nn.Linear(d_model, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_1d_sincos_pos_embed(self.d_model, self.max_len)
        self.positional_encoding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x, padding_mask=None):
        seq_len = x.size(1)
        x = self.x_embedder(x) #(B, L, 2) -> (B, L, d_model)

        x = x + self.positional_encoding[:, :seq_len, :]

        for block in self.blocks:
            x = block(x, padding_mask)  
        out = self.final_layer(x)  # (B,L, 302)
        return out

def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )