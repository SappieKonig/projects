import torch
from torch import nn, einsum

from einops import rearrange, repeat


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.q = nn.Linear(dim, inner_dim, bias = False)
        self.k = nn.Linear(dim, inner_dim, bias = False)
        self.v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k):

        v = k

        b, n, _, h = *q.shape, self.heads
        qkv = (self.q(q), self.k(k), self.v(v))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LunaBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=.0):
        super().__init__()
        self.p_attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.p_norm = Norm(dim)
        self.x_attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.x_attn_norm = Norm(dim)
        self.x_ffn = FeedForward(dim, hidden_dim=mlp_dim, dropout=dropout)
        self.x_ffn_norm = Norm(dim)

    def forward(self, x, p):
        p_att = self.p_attn(p, x)
        p = self.p_norm(p_att + p)
        x_att = self.x_attn(x, p_att)
        x = self.x_attn_norm(x_att + x)
        x_ff = self.x_ffn(x)
        x = self.x_ffn_norm(x_ff + x)

        return x, p


class Luna(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, p_tokens, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.p = nn.Parameter(torch.randn((p_tokens, dim)))
        for _ in range(depth):
            self.layers.append(LunaBlock(dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout))

    def forward(self, x):

        b = x.shape[0]
        p = repeat(self.p, 'n d -> b n d', b=b)

        for layer in self.layers:
            x, p = layer(x, p)

        return x



































