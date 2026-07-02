from __future__ import annotations

import torch
from torch import nn


class Seq2SeqGRU(nn.Module):
    """不带 attention 的最小 Encoder-Decoder 翻译模型。"""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        source_pad_id: int,
        target_pad_id: int,
        embedding_dim: int = 32,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.source_embedding = nn.Embedding(
            source_vocab_size,
            embedding_dim,
            padding_idx=source_pad_id,
        )
        self.target_embedding = nn.Embedding(
            target_vocab_size,
            embedding_dim,
            padding_idx=target_pad_id,
        )
        self.encoder = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.decoder = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, target_vocab_size)
        self.source_pad_id = source_pad_id

    def forward(self, source: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        source_embedded = self.source_embedding(source)
        encoder_outputs, _encoder_hidden = self.encoder(source_embedded)
        source_lengths = (source != self.source_pad_id).sum(dim=1).clamp(min=1)
        last_positions = source_lengths - 1
        batch_positions = torch.arange(source.shape[0], device=source.device)
        encoder_hidden = encoder_outputs[batch_positions, last_positions].unsqueeze(0)
        target_embedded = self.target_embedding(target_input)
        decoder_outputs, _decoder_hidden = self.decoder(target_embedded, encoder_hidden)
        return self.output(decoder_outputs)


class AdditiveAttention(nn.Module):
    """Bahdanau 风格 additive attention。"""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_outputs: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not source_mask.any(dim=1).all():
            raise ValueError("Each source sequence must contain at least one non-pad token.")
        query = self.query(decoder_outputs).unsqueeze(2)
        key = self.key(encoder_outputs).unsqueeze(1)
        scores = self.score(torch.tanh(query + key)).squeeze(-1)
        scores = scores.masked_fill(~source_mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ encoder_outputs
        return context, weights


class AttentionSeq2SeqGRU(nn.Module):
    """带 additive attention 的 Encoder-Decoder 翻译模型。"""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        source_pad_id: int,
        target_pad_id: int,
        embedding_dim: int = 32,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.source_embedding = nn.Embedding(
            source_vocab_size,
            embedding_dim,
            padding_idx=source_pad_id,
        )
        self.target_embedding = nn.Embedding(
            target_vocab_size,
            embedding_dim,
            padding_idx=target_pad_id,
        )
        self.encoder = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.decoder = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.attention = AdditiveAttention(hidden_size)
        self.output = nn.Linear(hidden_size * 2, target_vocab_size)
        self.source_pad_id = source_pad_id

    def forward(
        self,
        source: torch.Tensor,
        target_input: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        source_embedded = self.source_embedding(source)
        encoder_outputs, _encoder_hidden = self.encoder(source_embedded)
        source_mask = source != self.source_pad_id
        source_lengths = source_mask.sum(dim=1).clamp(min=1)
        last_positions = source_lengths - 1
        batch_positions = torch.arange(source.shape[0], device=source.device)
        encoder_hidden = encoder_outputs[batch_positions, last_positions].unsqueeze(0)

        target_embedded = self.target_embedding(target_input)
        decoder_outputs, _decoder_hidden = self.decoder(target_embedded, encoder_hidden)
        context, weights = self.attention(decoder_outputs, encoder_outputs, source_mask)
        logits = self.output(torch.cat([decoder_outputs, context], dim=-1))
        if return_attention:
            return logits, weights
        return logits
