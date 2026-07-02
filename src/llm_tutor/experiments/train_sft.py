from __future__ import annotations

import argparse

import torch

from llm_tutor.models.gpt import MiniGPT, MiniGPTConfig
from llm_tutor.post_training.sft import (
    IGNORE_INDEX,
    SFTBatch,
    load_tiny_sft_data,
    sft_cross_entropy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="第 16 章：在 tiny instruction 数据上做 SFT")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    vocab, train_loader = load_tiny_sft_data(block_size=args.block_size, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniGPT(
        MiniGPTConfig(
            vocab_size=vocab.size,
            block_size=args.block_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=0.1,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"vocab_size={vocab.size} examples={len(train_loader.dataset)}")
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, device, optimizer)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} sft_loss={train_loss:.4f}")

    model.eval()
    prompt = "Instruction: repeat: hi\nResponse: "
    prompt_ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(prompt_ids, max_new_tokens=12, temperature=0.7, top_k=8)
    print("\ngeneration")
    print(vocab.decode(generated[0]))


def _run_epoch(
    model: MiniGPT,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        output = model(batch.input_ids)
        loss = sft_cross_entropy(output.logits, batch.target_ids)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        token_count = batch.target_ids.ne(IGNORE_INDEX).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count
    return total_loss / total_tokens


def _move_batch(batch: SFTBatch, device: torch.device) -> SFTBatch:
    return SFTBatch(
        input_ids=batch.input_ids.to(device),
        target_ids=batch.target_ids.to(device),
        loss_mask=batch.loss_mask.to(device),
    )


if __name__ == "__main__":
    main()
