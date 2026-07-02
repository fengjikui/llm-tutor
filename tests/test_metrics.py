import torch

from llm_tutor.training.metrics import binary_classification_metrics, classification_metrics


def test_binary_classification_metrics() -> None:
    logits = torch.tensor(
        [
            [4.0, 1.0],
            [0.2, 1.5],
            [0.1, 2.0],
            [2.0, 0.1],
        ]
    )
    labels = torch.tensor([0, 1, 0, 0])

    metrics = binary_classification_metrics(logits, labels, positive_label=1)

    assert metrics.average == "binary"
    assert metrics.positive_label == 1
    assert metrics.accuracy == 0.75
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert round(metrics.f1, 4) == 0.6667


def test_binary_classification_metrics_can_change_positive_label() -> None:
    logits = torch.tensor(
        [
            [4.0, 1.0],
            [0.2, 1.5],
            [0.1, 2.0],
            [2.0, 0.1],
        ]
    )
    labels = torch.tensor([0, 1, 0, 0])

    metrics = binary_classification_metrics(logits, labels, positive_label=0)

    assert metrics.average == "binary"
    assert metrics.positive_label == 0
    assert metrics.accuracy == 0.75
    assert metrics.precision == 1.0
    assert round(metrics.recall, 4) == 0.6667
    assert round(metrics.f1, 4) == 0.8


def test_classification_metrics_supports_multiclass_macro_f1() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 2.0, 5.0],
            [0.0, 3.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 2])

    metrics = classification_metrics(logits, labels, positive_label=None)

    assert metrics.average == "macro"
    assert metrics.positive_label is None
    assert metrics.accuracy == 0.75
    assert round(metrics.f1, 4) == 0.7778
