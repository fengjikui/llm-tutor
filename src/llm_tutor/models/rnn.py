from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class CharRNNClassifier(nn.Module):
    """字符级序列分类器。

    输入 shape: [batch, seq_len]
    输出 shape: [batch, num_classes]
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 16,
        hidden_size: int = 32,
        cell_type: Literal["rnn", "gru", "lstm"] = "rnn",
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.pad_id = pad_id
        if cell_type == "rnn":
            self.recurrent: nn.RNNBase = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        elif cell_type == "gru":
            self.recurrent = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        elif cell_type == "lstm":
            self.recurrent = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        output, _hidden = self.recurrent(embedded)
        lengths = (input_ids != self.pad_id).sum(dim=1).clamp(min=1)
        last_token_positions = lengths - 1
        batch_positions = torch.arange(input_ids.shape[0], device=input_ids.device)
        last_real_output = output[batch_positions, last_token_positions]
        return self.classifier(last_real_output)
