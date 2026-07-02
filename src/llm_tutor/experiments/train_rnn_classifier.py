from __future__ import annotations

import argparse
from typing import Literal

import torch
from torch import nn

from llm_tutor.data.names import load_name_classification_data
from llm_tutor.models.rnn import CharRNNClassifier
from llm_tutor.training.loop import evaluate_classifier, train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="第 07 章：训练一个字符级 RNN 分类器")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--cell-type", choices=["rnn", "gru", "lstm"], default="rnn")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = load_name_classification_data(batch_size=args.batch_size, seed=args.seed)
    model = CharRNNClassifier(
        vocab_size=data.vocab_size,
        num_classes=data.num_classes,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        cell_type=args.cell_type,
        pad_id=data.pad_id,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_classifier(
        model,
        data.train_loader,
        data.val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        positive_label=None,
        optimizer_name="adam",
    )
    test = evaluate_classifier(
        model,
        data.test_loader,
        nn.CrossEntropyLoss(),
        device,
        positive_label=None,
    )
    cell_type: Literal["rnn", "gru", "lstm"] = args.cell_type
    print(
        f"cell_type={cell_type} "
        f"test_loss={test.loss:.4f} "
        f"test_acc={test.metrics.accuracy:.4f} "
        f"macro_f1={test.metrics.f1:.4f}"
    )


if __name__ == "__main__":
    main()
