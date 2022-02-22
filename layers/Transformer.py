from torch.nn import functional as nnf
from typing import Optional
import torch.nn as nn
import torch

from .MultiHeadAttention import MultiHeadAttention


class Transformer(nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super().__init__()

        if dim_ref is None:
            dim_ref = dim_self
        self.enc_dec = enc_dec

        if self.enc_dec:
            num_layers = num_layers * 2

        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:
                # cross
                layers.append(
                    TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer)
                )
            elif enc_dec:
                # self
                layers.append(
                    TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer)
                )
            else:
                # self or cross
                layers.append(
                    TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer)
                )

        self.layers = nn.ModuleList(layers)


    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []

        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)

        return x, attentions


    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:
                # cross
                x = layer(x, y)
            elif self.enc_dec:
                # self
                x = layer(x, x, mask)
            else:
                # self or cross
                x = layer(x, y, mask)

        return x


class MLPTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.0):
        super().__init__()

        if out_d is None:
            out_d = in_dim

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MLPTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


def geglu(x):
    x, gate = x.chunk(2, dim = -1)
    return x * nnf.gelu(gate)


def parse_act_fn(name: str):
    name = name.lower()
    if name == 'relu':
        return nnf.relu
    elif name == 'gelu':
        return nnf.gelu
    elif name == 'geglu':
        return geglu
    else:
        raise ValueError(f'Specified activation function {name} is not supported.')


class TransformerMapper(nn.Module):
    def __init__(self,
        dim_clip: int, 
        dim_embedding: int,
        prefix_length: int,
        clip_length: int,
        num_heads: int=8,
        num_layers: int=8,
        mlp_ratio: float=4.,
        prefix_init_std: float=1.,
        act_fn_name: str='relu'
    ):
        super().__init__()

        self.clip_length = clip_length

        self.transformer = Transformer(dim_embedding, num_heads, num_layers, mlp_ratio=mlp_ratio, act=parse_act_fn(act_fn_name))
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding) * prefix_init_std, requires_grad=True)

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)

        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)

        out = self.transformer(prefix)[:, self.clip_length:]

        return out

    
class TransformerMapperAllFeatures(nn.Module):
    def __init__(self,
        dim_clip: int,
        dim_embedding: int,
        prefix_length: int,
        clip_length: int, 
        use_pos_embeddings: bool,
        num_heads: int=8,
        num_layers: int=8,
        mlp_ratio: float=4.,
        prefix_init_std: float=1.,
        act_fn_name: str='relu'
    ):
        super().__init__()

        self.transformer = Transformer(dim_embedding, num_heads, num_layers, mlp_ratio=mlp_ratio, act=parse_act_fn(act_fn_name))
        self.linear = nn.Linear(dim_clip, dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding) * prefix_init_std, requires_grad=True)
        if use_pos_embeddings:
            print('Using position embeddings.')
            self.pos_embeddings = nn.Parameter(torch.randn(clip_length, dim_embedding), requires_grad=True)
        else:
            self.pos_embeddings = None

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.linear(x)

        if self.pos_embeddings is not None:
            p = self.pos_embeddings.unsqueeze(0).expand(batch_size, *self.pos_embeddings.shape)
            x = x + p

        prefix = self.prefix_const.unsqueeze(0).expand(batch_size, *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)

        clip_seq_len = x.shape[1]
        out = self.transformer(prefix)[:, clip_seq_len:]

        return out
