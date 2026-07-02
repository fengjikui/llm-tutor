from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearRegressionState:
    step: int
    weight: float
    bias: float
    loss: float


def make_toy_regression_data(n: int = 64, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """构造一条带噪声的直线：y = 3x - 2 + noise。"""

    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(-2.0, 2.0, n).unsqueeze(1)
    noise = 0.25 * torch.randn(n, 1, generator=generator)
    y = 3.0 * x - 2.0 + noise
    return x, y


def train_linear_regression_by_hand(
    steps: int = 80,
    lr: float = 0.05,
    seed: int = 42,
) -> list[LinearRegressionState]:
    """不用 autograd，手动计算 MSE 对 w/b 的梯度。

    对模型 y_hat = x * w + b：
    loss = mean((y_hat - y)^2)
    dloss/dw = mean(2 * (y_hat - y) * x)
    dloss/db = mean(2 * (y_hat - y))
    """

    x, y = make_toy_regression_data(seed=seed)
    w = torch.tensor([[0.0]])
    b = torch.tensor([[0.0]])
    history: list[LinearRegressionState] = []

    for step in range(1, steps + 1):
        y_hat = x @ w + b
        error = y_hat - y

        grad_w = (2.0 * error * x).mean()
        grad_b = (2.0 * error).mean()

        w -= lr * grad_w
        b -= lr * grad_b

        if step == 1 or step % 10 == 0 or step == steps:
            current_loss = ((x @ w + b - y) ** 2).mean()
            history.append(
                LinearRegressionState(
                    step=step,
                    weight=w.item(),
                    bias=b.item(),
                    loss=current_loss.item(),
                )
            )

    return history


def train_linear_regression_with_autograd(
    steps: int = 80,
    lr: float = 0.05,
    seed: int = 42,
) -> list[LinearRegressionState]:
    """用 PyTorch autograd 训练同一个 toy regression。"""

    x, y = make_toy_regression_data(seed=seed)
    w = torch.zeros(1, 1, requires_grad=True)
    b = torch.zeros(1, 1, requires_grad=True)
    history: list[LinearRegressionState] = []

    for step in range(1, steps + 1):
        y_hat = x @ w + b
        loss = ((y_hat - y) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

        if step == 1 or step % 10 == 0 or step == steps:
            with torch.no_grad():
                current_loss = ((x @ w + b - y) ** 2).mean()
            history.append(
                LinearRegressionState(
                    step=step,
                    weight=w.item(),
                    bias=b.item(),
                    loss=current_loss.item(),
                )
            )

    return history
