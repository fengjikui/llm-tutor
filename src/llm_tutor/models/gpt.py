from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from llm_tutor.generation.sampling import sample_next_token
from llm_tutor.models.attention import make_causal_attention_mask
from llm_tutor.models.transformer import TransformerBlock


@dataclass(frozen=True)
class MiniGPTConfig:
    vocab_size: int
    block_size: int
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class MiniGPTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    attention_weights: list[torch.Tensor] | None = None


class MiniGPT(nn.Module):
    """A small decoder-only GPT for teaching.

    Shape:
    - input_ids: [batch, seq_len]
    - logits: [batch, seq_len, vocab_size]
    """

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        if config.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2.")
        if config.block_size < 2:
            raise ValueError("block_size must be >= 2.")
        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        target_ids: torch.Tensor | None = None,
        return_attention: bool = False,
        ) -> MiniGPTOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len].")
        batch_size, seq_len = input_ids.shape
        if seq_len < 1:
            raise ValueError("input_ids must contain at least one token.")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must have dtype torch.long.")
        if seq_len > self.config.block_size:
            raise ValueError(
                f"seq_len={seq_len} exceeds block_size={self.config.block_size}; "
                "crop the context before calling MiniGPT."
            )
        if target_ids is not None and target_ids.shape != input_ids.shape:
            raise ValueError("target_ids must have the same shape as input_ids.")
        if target_ids is not None and target_ids.dtype != torch.long:
            raise ValueError("target_ids must have dtype torch.long.")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        attention_mask = make_causal_attention_mask(seq_len, device=input_ids.device)
        attention_weights: list[torch.Tensor] | None = [] if return_attention else None

        for block in self.blocks:
            if return_attention:
                x, weights = block(x, attention_mask=attention_mask, return_attention=True)
                assert attention_weights is not None
                attention_weights.append(weights)
            else:
                x = block(x, attention_mask=attention_mask)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, self.config.vocab_size),
                target_ids.reshape(batch_size * seq_len),
            )
        return MiniGPTOutput(logits=logits, loss=loss, attention_weights=attention_weights)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        was_training = self.training
        self.eval()
        try:
            generated = input_ids
            for _ in range(max_new_tokens):
                context = generated[:, -self.config.block_size :]
                logits = self(context).logits[:, -1, :]
                next_token = sample_next_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                )
                generated = torch.cat([generated, next_token], dim=1)
            return generated
        finally:
            self.train(was_training)
