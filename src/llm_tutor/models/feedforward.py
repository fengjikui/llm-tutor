from __future__ import annotations

import torch
from torch import nn


class LinearClassifier(nn.Module):
    """最小线性分类器：输入特征直接映射到类别 logits。"""

    def __init__(self, num_features: int, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BasicNeuralNetwork(nn.Module):
    """神经网络基础章节使用的小型全连接网络。"""

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_classes: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
