from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from llm_tutor.training.metrics import ClassificationMetrics, classification_metrics


@dataclass(frozen=True)
class EpochResult:
    loss: float
    metrics: ClassificationMetrics


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    positive_label: int | None = 1,
) -> EpochResult:
    _validate_loader(loader, name="loader")
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            batch_size = y.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    metrics = classification_metrics(
        torch.cat(all_logits),
        torch.cat(all_labels),
        positive_label=positive_label,
    )
    return EpochResult(loss=total_loss / total_examples, metrics=metrics)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: torch.device | None = None,
    positive_label: int | None = 1,
    optimizer_name: Literal["sgd", "adam"] = "adam",
    weight_decay: float = 0.0,
) -> list[dict[str, float]]:
    """训练一个分类器，并返回每轮的日志。

    这段代码故意保持“教学显式”：
    1. 前向传播得到 logits；
    2. 用 logits 和标签计算 loss；
    3. 清空旧梯度；
    4. 反向传播得到新梯度；
    5. optimizer 根据梯度更新参数。
    """

    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if lr <= 0:
        raise ValueError("lr must be > 0.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be >= 0.")
    _validate_loader(train_loader, name="train_loader")
    _validate_loader(val_loader, name="val_loader")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = y.shape[0]
            running_loss += loss.item() * batch_size
            total_examples += batch_size

        val = evaluate_classifier(model, val_loader, loss_fn, device, positive_label=positive_label)
        row = {
            "epoch": float(epoch),
            "train_loss": running_loss / total_examples,
            "val_loss": val.loss,
            "val_accuracy": val.metrics.accuracy,
            "val_f1": val.metrics.f1,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} "
            f"val_f1={row['val_f1']:.4f}"
        )

    return history


def _validate_loader(loader: DataLoader, name: str) -> None:
    try:
        length = len(loader)
    except TypeError:
        return
    if length == 0:
        raise ValueError(f"{name} is empty; please check dataset split and batch_size.")


def _make_optimizer(
    model: nn.Module,
    *,
    optimizer_name: Literal["sgd", "adam"],
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")
