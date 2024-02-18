from typing import List, Optional, Union
import segmentation_models_pytorch as smp
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from segmentation_models_pytorch.base import (ClassificationHead,SegmentationModel)
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from torch import einsum, nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def FeedForward(dim, mult=4.):
    inner_dim = int(dim * mult)
    return Residual(nn.Sequential(
        LayerNorm(dim),
        nn.GELU(),
        nn.Conv2d(dim, inner_dim, 1, bias=False),
        LayerNorm(inner_dim),   # properly credit assign normformer
        nn.GELU(),
        nn.Conv2d(inner_dim, dim, 1, bias=False)
    ))


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim = True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + eps).sqrt() * self.gamma


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=64
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm = LayerNorm(dim)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias=False)

    def forward(self, x):
        f, h, w = x.shape[-3:]

        residual = x.clone()

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) ... -> b h (...) c', h=self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out) + residual


class DynamicChannelSelector(nn.Module):
    def __init__(self, in_channels, emb_channels, channels_per_selector):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.channels_per_selector = channels_per_selector
        self.scale = 64 ** -0.5
        self.inner_dim = 32

        self.to_mapping = nn.Sequential(
            nn.Linear(emb_channels, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
            nn.GELU(),
            nn.Linear(self.inner_dim, in_channels * channels_per_selector)
        )

    def forward(self, q, v):
        # q = context, from embedding
        # v = values, from encoder
        b, c, x, y = v.shape

        # flatten spatial dims
        v = rearrange(v, 'b c x y -> b (x y) c')

        # get mapping
        mapping = self.to_mapping(q)

        # split into channels
        # n = number of selectors
        # p = channels per selector
        # c = in channels
        # this creates a matrix with selectors and channels
        mapping = rearrange(mapping, 'b n (c p) -> b c (n p)', c=self.in_channels, p=self.channels_per_selector) * self.scale
        
        # select channels for each selector
        # FIXME -1 or -2?
        # -2 means, that each output channel can get information from one input channel
        # -1 means, that each output channel can get information from all input channels
        # or so
        mapping = mapping.softmax(dim=-1)
        mapping = (v @ mapping)

        return rearrange(mapping, "b (x y) c -> b c x y", x=x, y=y)


class ContextSelfAttention(nn.Module):
    def __init__(self, dim, time_dim=8, heads=8, dim_head=64):
        super().__init__()
        self.embedding_dim = dim
        self.time_dim = time_dim
        self.absolute_positions = nn.Parameter(torch.randn(1, self.time_dim, dim_head))

        self.to_qkv = nn.Linear(dim, dim_head * 3, bias=False)
        self.attention = nn.MultiheadAttention(embed_dim=dim_head, num_heads=heads, batch_first=True)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim_head),
            nn.GELU(),
            nn.Linear(dim_head, dim),
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q + self.absolute_positions
        k = k + self.absolute_positions
        a = self.attention(q, k, v)[0]
        return self.to_out(a) + x


class TokenCollapser(nn.Module):
    def __init__(self, collapse_factor=2, in_dim=300, in_token=8):
        super().__init__()
        self.collapse = collapse_factor

        self.layer = nn.Sequential(
            nn.Linear(in_dim, in_dim // (self.collapse * 2)),
            Rearrange("b (t r) c -> b t (c r)", r=self.collapse),
            nn.Linear((in_dim // 2), in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.layer(x)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        DynamicChannelSelector(in_channels=48, emb_channels=300, channels_per_selector=12)
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.Identity()
        super().__init__(conv2d, upsampling, activation)