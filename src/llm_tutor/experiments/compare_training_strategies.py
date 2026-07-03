from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from llm_tutor.data.tabular import load_breast_cancer_data
from llm_tutor.experiments.artifacts import (
    ExperimentArtifacts,
    add_artifact_args,
    args_to_config,
)
from llm_tutor.models.feedforward import BasicNeuralNetwork
from llm_tutor.training.loop import evaluate_classifier, train_classifier


@dataclass(frozen=True)
class TrainingStrategy:
    name: str
    optimizer_name: Literal["sgd", "adam"]
    lr: float
    weight_decay: float = 0.0
    dropout: float = 0.0


def default_strategies() -> list[TrainingStrategy]:
    return [
        TrainingStrategy(name="adam_lr_1e-3", optimizer_name="adam", lr=1e-3),
        TrainingStrategy(name="adam_lr_1e-2", optimizer_name="adam", lr=1e-2),
        TrainingStrategy(name="sgd_momentum_lr_1e-2", optimizer_name="sgd", lr=1e-2),
        TrainingStrategy(
            name="adam_weight_decay",
            optimizer_name="adam",
            lr=1e-3,
            weight_decay=1e-3,
        ),
        TrainingStrategy(name="adam_dropout", optimizer_name="adam", lr=1e-3, dropout=0.2),
        TrainingStrategy(
            name="adam_weight_decay_dropout",
            optimizer_name="adam",
            lr=1e-3,
            weight_decay=1e-3,
            dropout=0.2,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="第 05 章：比较常见训练策略")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    add_artifact_args(parser)
    args = parser.parse_args()

    artifacts = ExperimentArtifacts.create(
        args.output_dir,
        experiment_name="compare_training_strategies",
        config=args_to_config(args),
    )
    with artifacts.capture_stdout():
        if artifacts.enabled:
            print(f"artifacts_dir={artifacts.run_dir}")
        _run(args, artifacts)


def _run(args: argparse.Namespace, artifacts: ExperimentArtifacts) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[tuple[TrainingStrategy, float, float, float]] = []
    best_strategy: TrainingStrategy | None = None
    best_val_loss = float("inf")

    for strategy in default_strategies():
        print(f"\n=== {strategy.name} ===")
        torch.manual_seed(args.seed)
        data = load_breast_cancer_data(batch_size=args.batch_size, seed=args.seed)
        model = BasicNeuralNetwork(
            num_features=data.num_features,
            hidden_size=args.hidden_size,
            dropout=strategy.dropout,
        )
        history = train_classifier(
            model,
            data.train_loader,
            data.val_loader,
            epochs=args.epochs,
            lr=strategy.lr,
            device=device,
            positive_label=0,
            optimizer_name=strategy.optimizer_name,
            weight_decay=strategy.weight_decay,
        )
        for row in history:
            artifacts.append_metric({"phase": "strategy_train", "strategy": strategy.name, **row})
        final = history[-1]
        rows.append(
            (
                strategy,
                final["val_loss"],
                final["val_accuracy"],
                final["val_f1"],
            )
        )
        if final["val_loss"] < best_val_loss:
            best_val_loss = final["val_loss"]
            best_strategy = strategy

    print("\nvalidation summary")
    print("strategy                       val_loss  val_acc  malignant_val_f1")
    for strategy, val_loss, val_acc, val_f1 in rows:
        print(f"{strategy.name:<30} {val_loss:>8.4f} {val_acc:>8.4f} {val_f1:>17.4f}")

    assert best_strategy is not None
    print(f"\nselected_by_val_loss={best_strategy.name}")
    print("Now retrain the selected strategy once and report test metrics.")

    torch.manual_seed(args.seed)
    data = load_breast_cancer_data(batch_size=args.batch_size, seed=args.seed)
    model = BasicNeuralNetwork(
        num_features=data.num_features,
        hidden_size=args.hidden_size,
        dropout=best_strategy.dropout,
    )
    train_classifier(
        model,
        data.train_loader,
        data.val_loader,
        epochs=args.epochs,
        lr=best_strategy.lr,
        device=device,
        positive_label=0,
        optimizer_name=best_strategy.optimizer_name,
        weight_decay=best_strategy.weight_decay,
    )
    for row in history:
        artifacts.append_metric(
            {"phase": "selected_retrain", "strategy": best_strategy.name, **row}
        )
    test = evaluate_classifier(
        model,
        data.test_loader,
        nn.CrossEntropyLoss(),
        device,
        positive_label=0,
    )
    print(
        "final_test "
        f"loss={test.loss:.4f} "
        f"acc={test.metrics.accuracy:.4f} "
        f"malignant_precision={test.metrics.precision:.4f} "
        f"malignant_recall={test.metrics.recall:.4f} "
        f"malignant_f1={test.metrics.f1:.4f}"
    )
    artifacts.write_summary(
        {
            "dataset": "Breast Cancer Wisconsin",
            "device": str(device),
            "hidden_size": args.hidden_size,
            "strategies": [
                {
                    "name": strategy.name,
                    "optimizer_name": strategy.optimizer_name,
                    "lr": strategy.lr,
                    "weight_decay": strategy.weight_decay,
                    "dropout": strategy.dropout,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "malignant_val_f1": val_f1,
                }
                for strategy, val_loss, val_acc, val_f1 in rows
            ],
            "selected_by_val_loss": best_strategy.name,
            "test": {
                "loss": test.loss,
                "accuracy": test.metrics.accuracy,
                "malignant_precision": test.metrics.precision,
                "malignant_recall": test.metrics.recall,
                "malignant_f1": test.metrics.f1,
            },
        }
    )


if __name__ == "__main__":
    main()
