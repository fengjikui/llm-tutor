from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def build_rope_cache(seq_len: int, head_dim: int, *, device=None, base: float = 10_000.0):
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE.")
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    dims = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (dims / head_dim))
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a tensor shaped [batch, heads, seq_len, head_dim]."""

    if x.shape[-1] % 2 != 0:
        raise ValueError("Last dimension must be even for RoPE.")
    seq_len = x.shape[-2]
    cos = cos[:seq_len].view(1, 1, seq_len, -1)
    sin = sin[:seq_len].view(1, 1, seq_len, -1)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


def repeat_kv(x: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention.

    Input shape: [batch, kv_heads, seq_len, head_dim].
    Output shape: [batch, query_heads, seq_len, head_dim].
    """

    if num_query_heads % x.shape[1] != 0:
        raise ValueError("num_query_heads must be divisible by kv_heads.")
    repeats = num_query_heads // x.shape[1]
    return x[:, :, None, :, :].expand(-1, -1, repeats, -1, -1).reshape(
        x.shape[0], num_query_heads, x.shape[2], x.shape[3]
    )


def causal_mask(seq_len: int, *, device=None) -> torch.Tensor:
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    return mask.view(1, 1, seq_len, seq_len)


def sliding_window_causal_mask(seq_len: int, window_size: int, *, device=None) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    positions = torch.arange(seq_len, device=device)
    distance = positions[:, None] - positions[None, :]
    visible = (distance >= 0) & (distance < window_size)
    return visible.view(1, 1, seq_len, seq_len)


def grouped_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Minimal GQA attention over projected q/k/v tensors.

    q shape: [batch, q_heads, seq_len, head_dim]
    k/v shape: [batch, kv_heads, seq_len, head_dim]
    """

    key = repeat_kv(key, query.shape[1])
    value = repeat_kv(value, query.shape[1])
    scores = query @ key.transpose(-2, -1)
    scores = scores / (query.shape[-1] ** 0.5)
    if attention_mask is not None:
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1)
    return weights @ value


@dataclass
class KVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key is None:
            self.key = key
            self.value = value
        else:
            if self.value is None:
                raise RuntimeError("KVCache is corrupted: value is missing.")
            self.key = torch.cat([self.key, key], dim=-2)
            self.value = torch.cat([self.value, value], dim=-2)
        return self.key, self.value


class LatentKVProjection(nn.Module):
    """Shape-faithful sketch of MLA-style joint KV compression.

    Real MLA has more details, including separated RoPE dimensions and optimized
    absorbed computation. This module only shows the core idea: cache a compact
    latent representation, then expand it when attention needs K/V.
    """

    def __init__(self, embed_dim: int, latent_dim: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.down = nn.Linear(embed_dim, latent_dim, bias=False)
        self.up_key = nn.Linear(latent_dim, num_kv_heads * head_dim, bias=False)
        self.up_value = nn.Linear(latent_dim, num_kv_heads * head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.down(x)
        key = self.up_key(latent)
        value = self.up_value(latent)
        batch, seq_len, _ = x.shape
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return latent, key, value
