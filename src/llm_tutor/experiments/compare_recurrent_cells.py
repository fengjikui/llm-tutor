from __future__ import annotations

import argparse
from typing import Literal

import torch
from torch import nn

from llm_tutor.data.names import load_name_classification_data
from llm_tutor.models.rnn import CharRNNClassifier
from llm_tutor.training.loop import evaluate_classifier, train_classifier

CellType = Literal["rnn", "gru", "lstm"]


def main() -> None:
    parser = argparse.ArgumentParser(description="第 08 章：比较 RNN、GRU 和 LSTM")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preview_data = load_name_classification_data(batch_size=args.batch_size, seed=args.seed)
    print(
        "note: single-seed toy dataset; use this to inspect behavior, "
        "not to rank architectures."
    )
    print(
        f"seed={args.seed} "
        f"train_batches={len(preview_data.train_loader)} "
        f"val_batches={len(preview_data.val_loader)} "
        f"test_batches={len(preview_data.test_loader)}"
    )

    rows: list[tuple[str, int, float, float, float, float]] = []
    for cell_type in ("rnn", "gru", "lstm"):
        print(f"\n=== {cell_type} ===")
        torch.manual_seed(args.seed)
        data = load_name_classification_data(batch_size=args.batch_size, seed=args.seed)
        model = CharRNNClassifier(
            vocab_size=data.vocab_size,
            num_classes=data.num_classes,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            cell_type=cell_type,
            pad_id=data.pad_id,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"parameters={parameter_count}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        history = train_classifier(
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
        rows.append(
            (
                cell_type,
                parameter_count,
                history[-1]["val_loss"],
                history[-1]["val_accuracy"],
                test.metrics.accuracy,
                test.metrics.f1,
            )
        )

    print("\nsummary")
    print("cell   params  val_loss  val_acc  test_acc  test_macro_f1")
    for cell_type, params, val_loss, val_acc, test_acc, test_f1 in rows:
        print(
            f"{cell_type:<5} {params:>6d} {val_loss:>8.4f} "
            f"{val_acc:>8.4f} {test_acc:>9.4f} {test_f1:>14.4f}"
        )


if __name__ == "__main__":
    main()
