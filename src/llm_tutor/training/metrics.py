from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    average: str
    positive_label: int | None
    accuracy: float
    precision: float
    recall: float
    f1: float


def binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_label: int = 1,
) -> ClassificationMetrics:
    """计算二分类指标。

    指标本身不关心标签语义，只关心“哪个 label 被当作正类”。在医学
    数据里这尤其重要：如果把 benign 当正类，recall 就不是癌症召回率。
    """

    if positive_label not in (0, 1):
        raise ValueError("positive_label must be 0 or 1 for binary classification.")

    preds = logits.argmax(dim=-1)
    labels = labels.to(preds.device)
    negative_label = 1 - positive_label

    tp = ((preds == positive_label) & (labels == positive_label)).sum().item()
    fp = ((preds == positive_label) & (labels == negative_label)).sum().item()
    fn = ((preds == negative_label) & (labels == positive_label)).sum().item()
    correct = (preds == labels).sum().item()
    total = labels.numel()

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return ClassificationMetrics(
        average="binary",
        positive_label=positive_label,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_label: int | None = 1,
) -> ClassificationMetrics:
    """计算分类指标。

    - `positive_label` 不为 None 时，按二分类正类语义计算 precision/recall/F1。
    - `positive_label=None` 时，按多分类 macro average 计算 precision/recall/F1。
    """

    if positive_label is not None:
        return binary_classification_metrics(logits, labels, positive_label=positive_label)

    preds = logits.argmax(dim=-1)
    labels = labels.to(preds.device)
    num_classes = logits.shape[-1]
    correct = (preds == labels).sum().item()
    total = labels.numel()

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for class_id in range(num_classes):
        tp = ((preds == class_id) & (labels == class_id)).sum().item()
        fp = ((preds == class_id) & (labels != class_id)).sum().item()
        fn = ((preds != class_id) & (labels == class_id)).sum().item()
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return ClassificationMetrics(
        average="macro",
        positive_label=None,
        accuracy=correct / total if total > 0 else 0.0,
        precision=sum(precisions) / num_classes,
        recall=sum(recalls) / num_classes,
        f1=sum(f1s) / num_classes,
    )
