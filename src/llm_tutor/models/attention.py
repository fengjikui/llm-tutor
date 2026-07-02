from __future__ import annotations

import math

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention.

    Shape:
    - query: [batch, heads, query_len, head_dim]
    - key: [batch, heads, key_len, head_dim]
    - value: [batch, heads, key_len, head_dim]
    - attention_mask: broadcastable to [batch, heads, query_len, key_len]
      True means "can attend"; False means "masked out".
    """

    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must have the same head_dim.")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value must have the same sequence length.")

    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(query.shape[-1])
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be a boolean tensor.")
        try:
            attention_mask = attention_mask.expand(scores.shape)
        except RuntimeError as error:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} cannot broadcast "
                f"to scores shape {tuple(scores.shape)}."
            ) from error
        if not attention_mask.any(dim=-1).all():
            raise ValueError("Every query position must be able to attend to at least one key.")
        scores = scores.masked_fill(~attention_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ value
    return output, weights


def make_padding_attention_mask(
    token_ids: torch.Tensor,
    *,
    pad_id: int,
) -> torch.Tensor:
    """Build a self-attention padding mask.

    Output shape: [batch, 1, 1, seq_len]
    """

    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_attention_mask(seq_len: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Build a causal mask where position t can only see positions <= t.

    Output shape: [1, 1, seq_len, seq_len]
    """

    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril().view(
        1,
        1,
        seq_len,
        seq_len,
    )


class MultiHeadSelfAttention(nn.Module):
    """教学版 multi-head self-attention。"""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _embed_dim = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        attention_output, weights = scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.embed_dim)
        output = self.output(attention_output)
        if return_attention:
            return output, weights
        return output

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _embed_dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
