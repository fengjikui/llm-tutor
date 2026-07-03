from __future__ import annotations

import argparse

import torch
from torch import nn

from llm_tutor.data.vision import load_fashion_mnist_data
from llm_tutor.experiments.artifacts import (
    ExperimentArtifacts,
    add_artifact_args,
    args_to_config,
)
from llm_tutor.models.cnn import SmallCNN
from llm_tutor.training.loop import evaluate_classifier, train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="第 06 章：训练一个小型 CNN 做图像分类")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int, default=2048)
    parser.add_argument("--val-limit", type=int, default=512)
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    add_artifact_args(parser)
    args = parser.parse_args()

    artifacts = ExperimentArtifacts.create(
        args.output_dir,
        experiment_name="train_cnn",
        config=args_to_config(args),
    )
    with artifacts.capture_stdout():
        if artifacts.enabled:
            print(f"artifacts_dir={artifacts.run_dir}")
        _run(args, artifacts)


def _run(args: argparse.Namespace, artifacts: ExperimentArtifacts) -> None:
    torch.manual_seed(args.seed)
    data = load_fashion_mnist_data(
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        seed=args.seed,
    )
    model = SmallCNN(num_classes=data.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = train_classifier(
        model,
        data.train_loader,
        data.val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        optimizer_name="adam",
        positive_label=None,
    )
    artifacts.append_metrics({"phase": "train", **row} for row in history)
    test = evaluate_classifier(
        model,
        data.test_loader,
        nn.CrossEntropyLoss(),
        device,
        positive_label=None,
    )
    print(
        f"test loss={test.loss:.4f} "
        f"acc={test.metrics.accuracy:.4f} "
        f"macro_f1={test.metrics.f1:.4f}"
    )
    artifacts.write_summary(
        {
            "dataset": "Fashion-MNIST",
            "device": str(device),
            "num_classes": data.num_classes,
            "limits": {
                "train": args.train_limit,
                "validation": args.val_limit,
                "test": args.test_limit,
            },
            "final_validation": history[-1],
            "test": {
                "loss": test.loss,
                "accuracy": test.metrics.accuracy,
                "macro_f1": test.metrics.f1,
            },
        }
    )


if __name__ == "__main__":
    main()
