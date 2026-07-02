from __future__ import annotations

import argparse

import torch
from torch import nn

from llm_tutor.data.tabular import load_breast_cancer_data
from llm_tutor.models.feedforward import LinearClassifier
from llm_tutor.training.loop import evaluate_classifier, train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="第 01 章：训练一个线性表格分类器")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = load_breast_cancer_data(batch_size=args.batch_size, seed=args.seed)
    model = LinearClassifier(num_features=data.num_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_classifier(
        model,
        data.train_loader,
        data.val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        positive_label=0,
    )

    test = evaluate_classifier(
        model,
        data.test_loader,
        nn.CrossEntropyLoss(),
        device,
        positive_label=0,
    )
    print(
        "test "
        f"loss={test.loss:.4f} "
        f"acc={test.metrics.accuracy:.4f} "
        f"malignant_precision={test.metrics.precision:.4f} "
        f"malignant_recall={test.metrics.recall:.4f} "
        f"malignant_f1={test.metrics.f1:.4f}"
    )


if __name__ == "__main__":
    main()
