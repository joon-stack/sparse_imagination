import torch
from torch import nn
from einops import rearrange, repeat


NUM_FRAMES = 1
NUM_PATCHES = 1


def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    return torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.register_buffer("bias", generate_mask_matrix(NUM_PATCHES, NUM_FRAMES))

    def forward(self, x):
        _, tokens, _ = x.size()
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            qkv,
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots.masked_fill(self.bias[:, :, :tokens, :tokens] == 0, float("-inf"))
        attn = self.attend(dots)
        out = torch.matmul(self.dropout(attn), v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        for attn_layer, ff_layer in self.layers:
            attn_output = attn_layer(x)
            x = attn_output + x
            x = ff_layer(x) + x
        return self.norm(x)


class ViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.num_patches = num_patches
        self.num_frames = num_frames
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, x):
        _, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)


class ViTPredictorWithoutPE(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.num_patches = num_patches
        self.num_frames = num_frames
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, x):
        _, n, _ = x.shape
        frame_pe = repeat(self.pos_embedding, "1 f d -> 1 (f p) d", p=self.num_patches)
        x = x + frame_pe[:, :n]
        x = self.dropout(x)
        return self.transformer(x)
