from __future__ import annotations

import argparse

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


def main() -> None:
    parser = argparse.ArgumentParser(description="第 04 章：训练一个神经网络基础模型")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    add_artifact_args(parser)
    args = parser.parse_args()

    artifacts = ExperimentArtifacts.create(
        args.output_dir,
        experiment_name="train_neural_network_basics",
        config=args_to_config(args),
    )
    with artifacts.capture_stdout():
        if artifacts.enabled:
            print(f"artifacts_dir={artifacts.run_dir}")
        _run(args, artifacts)


def _run(args: argparse.Namespace, artifacts: ExperimentArtifacts) -> None:
    torch.manual_seed(args.seed)
    data = load_breast_cancer_data(batch_size=args.batch_size, seed=args.seed)
    model = BasicNeuralNetwork(
        num_features=data.num_features,
        hidden_size=args.hidden_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = train_classifier(
        model,
        data.train_loader,
        data.val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        positive_label=0,
    )
    artifacts.append_metrics({"phase": "train", **row} for row in history)

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
    artifacts.write_summary(
        {
            "dataset": "Breast Cancer Wisconsin",
            "device": str(device),
            "num_features": data.num_features,
            "hidden_size": args.hidden_size,
            "class_names": data.class_names,
            "final_validation": history[-1],
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
