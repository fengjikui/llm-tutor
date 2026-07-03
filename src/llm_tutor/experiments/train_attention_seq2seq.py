from __future__ import annotations

import argparse

import torch
from torch import nn

from llm_tutor.data.translation import TranslationBatch, load_toy_translation_data
from llm_tutor.experiments.artifacts import (
    ExperimentArtifacts,
    add_artifact_args,
    args_to_config,
)
from llm_tutor.models.seq2seq import AttentionSeq2SeqGRU


def main() -> None:
    parser = argparse.ArgumentParser(description="第 10 章：训练带 attention 的 Seq2Seq 翻译模型")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    add_artifact_args(parser)
    args = parser.parse_args()

    artifacts = ExperimentArtifacts.create(
        args.output_dir,
        experiment_name="train_attention_seq2seq",
        config=args_to_config(args),
    )
    with artifacts.capture_stdout():
        if artifacts.enabled:
            print(f"artifacts_dir={artifacts.run_dir}")
        _run(args, artifacts)


def _run(args: argparse.Namespace, artifacts: ExperimentArtifacts) -> None:
    torch.manual_seed(args.seed)
    data = load_toy_translation_data(batch_size=args.batch_size, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionSeq2SeqGRU(
        source_vocab_size=len(data.source_vocab.id_to_token),
        target_vocab_size=len(data.target_vocab.id_to_token),
        source_pad_id=data.source_vocab.pad_id,
        target_pad_id=data.target_vocab.pad_id,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=data.target_vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, data.train_loader, loss_fn, device, optimizer=optimizer)
        val_loss = _run_epoch(model, data.val_loader, loss_fn, device)
        row = {
            "phase": "train",
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(row)
        artifacts.append_metric(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    print("\ntranslations")
    model.eval()
    translations = []
    attention_shape = None
    logits_shape = None
    with torch.no_grad():
        for batch in data.test_loader:
            batch = _move_batch(batch, device)
            predicted_ids = _greedy_decode(
                model,
                batch.source,
                bos_id=data.target_vocab.bos_id,
                eos_id=data.target_vocab.eos_id,
                max_len=batch.target_output.shape[1] + 2,
            )
            logits, attention = model(batch.source, batch.target_input, return_attention=True)
            attention_shape = tuple(attention.shape)
            logits_shape = tuple(logits.shape)
            print(f"attention_shape={attention_shape} logits_shape={logits_shape}")
            for source_ids, target_ids, pred_ids in zip(
                batch.source.cpu().tolist(),
                batch.target_output.cpu().tolist(),
                predicted_ids.cpu().tolist(),
                strict=True,
            ):
                source_text = data.source_vocab.decode(source_ids)
                target_text = data.target_vocab.decode(target_ids)
                pred_text = data.target_vocab.decode(pred_ids)
                print(
                    f"src='{source_text}' "
                    f"target='{target_text}' "
                    f"pred='{pred_text}'"
                )
                translations.append(
                    {"source": source_text, "target": target_text, "prediction": pred_text}
                )
    artifacts.write_summary(
        {
            "dataset": "toy translation",
            "device": str(device),
            "source_vocab_size": len(data.source_vocab.id_to_token),
            "target_vocab_size": len(data.target_vocab.id_to_token),
            "final": history[-1],
            "attention_shape": attention_shape,
            "logits_shape": logits_shape,
            "translations": translations,
        }
    )


def _run_epoch(
    model: AttentionSeq2SeqGRU,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_tokens = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = _move_batch(batch, device)
            logits = model(batch.source, batch.target_input)
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                batch.target_output.reshape(-1),
            )
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            token_count = (batch.target_output != loss_fn.ignore_index).sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count
    return total_loss / total_tokens


@torch.no_grad()
def _greedy_decode(
    model: AttentionSeq2SeqGRU,
    source: torch.Tensor,
    *,
    bos_id: int,
    eos_id: int,
    max_len: int,
) -> torch.Tensor:
    batch_size = source.shape[0]
    device = source.device
    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model(source, generated)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if (next_token == eos_id).all():
            break
    return generated


def _move_batch(batch: TranslationBatch, device: torch.device) -> TranslationBatch:
    return TranslationBatch(
        source=batch.source.to(device),
        target_input=batch.target_input.to(device),
        target_output=batch.target_output.to(device),
    )


if __name__ == "__main__":
    main()
