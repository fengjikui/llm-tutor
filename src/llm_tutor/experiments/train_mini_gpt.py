from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from llm_tutor.data.language_modeling import LanguageModelingBatch, load_tiny_language_modeling_data
from llm_tutor.models.gpt import MiniGPT, MiniGPTConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="第 13 章：从零训练一个字符级 mini-GPT")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-len", type=int, default=80)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = load_tiny_language_modeling_data(
        block_size=args.block_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniGPT(
        MiniGPTConfig(
            vocab_size=data.vocab.size,
            block_size=data.block_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=0.1,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(
        f"vocab_size={data.vocab.size} block_size={data.block_size} "
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, data.train_loader, device, optimizer=optimizer)
        val_loss = _run_epoch(model, data.val_loader, device)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    prompt = "the "
    prompt_ids = torch.tensor([data.vocab.encode(prompt)], dtype=torch.long, device=device)
    model.eval()
    generated = model.generate(prompt_ids, max_new_tokens=args.sample_len, temperature=0.8)
    print("\ngeneration")
    print(data.vocab.decode(generated[0]))

    if args.checkpoint_path is not None:
        save_checkpoint(
            path=args.checkpoint_path,
            model=model,
            optimizer=optimizer,
            vocab=data.vocab,
            epoch=args.epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            training_args={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "block_size": args.block_size,
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "lr": args.lr,
                "seed": args.seed,
            },
            data_meta={
                "dataset": "TINY_STORY_CORPUS",
                "vocab_size": data.vocab.size,
                "train_token_range": data.train_token_range,
                "val_token_range": data.val_token_range,
            },
        )
        print(f"checkpoint_saved={args.checkpoint_path}")


def _run_epoch(
    model: MiniGPT,
    loader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_tokens = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(batch.input_ids, target_ids=batch.target_ids)
            assert output.loss is not None
            if optimizer is not None:
                optimizer.zero_grad()
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            token_count = batch.target_ids.numel()
            total_loss += output.loss.item() * token_count
            total_tokens += token_count
    return total_loss / total_tokens


def _move_batch(batch: LanguageModelingBatch, device: torch.device) -> LanguageModelingBatch:
    return LanguageModelingBatch(
        input_ids=batch.input_ids.to(device),
        target_ids=batch.target_ids.to(device),
    )


def save_checkpoint(
    *,
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    vocab,
    epoch: int,
    train_loss: float,
    val_loss: float,
    training_args: dict[str, Any],
    data_meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(model.config),
            "vocab_id_to_token": list(vocab.id_to_token),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "training_args": training_args,
            "data_meta": data_meta,
        },
        path,
    )


if __name__ == "__main__":
    main()
