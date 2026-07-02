from __future__ import annotations

import torch
from torch import nn

from llm_tutor.models.attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block.

    Shape:
    - input: [batch, seq_len, embed_dim]
    - output: [batch, seq_len, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attention_input = self.attention_norm(x)
        if return_attention:
            attention_output, weights = self.attention(
                attention_input,
                attention_mask=attention_mask,
                return_attention=True,
            )
        else:
            attention_output = self.attention(
                attention_input,
                attention_mask=attention_mask,
                return_attention=False,
            )
            weights = None
        x = x + self.attention_dropout(attention_output)
        x = x + self.mlp(self.mlp_norm(x))
        if return_attention:
            assert weights is not None
            return x, weights
        return x
