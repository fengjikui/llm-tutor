from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from llm_tutor.data.language_modeling import CharBlockDataset, CharVocabulary, LanguageModelingBatch
from llm_tutor.models.gpt import MiniGPT, MiniGPTConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="第 14 章：检查 GPT 的 x/y、logits 和逐 token loss"
    )
    parser.add_argument("--text", type=str, default="the model learns")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    vocab = CharVocabulary.from_text(args.text)
    token_ids = vocab.encode(args.text)
    batch = make_single_lm_batch(token_ids, block_size=args.block_size)
    model = MiniGPT(
        MiniGPTConfig(
            vocab_size=vocab.size,
            block_size=args.block_size,
            embed_dim=16,
            num_heads=4,
            num_layers=1,
        )
    )

    output = model(batch.input_ids, target_ids=batch.target_ids, return_attention=True)
    assert output.loss is not None
    token_losses = per_token_cross_entropy(output.logits, batch.target_ids)
    manual_loss = token_losses.mean()
    assert output.attention_weights is not None
    future_weight_sum = output.attention_weights[0][..., _future_positions(args.block_size)].sum()

    print(f"text={args.text!r}")
    print(f"vocab_size={vocab.size} block_size={args.block_size}")
    print(f"x_ids={batch.input_ids[0].tolist()}")
    print(f"y_ids={batch.target_ids[0].tolist()}")
    print(f"x_text={vocab.decode(batch.input_ids[0])!r}")
    print(f"y_text={vocab.decode(batch.target_ids[0])!r}")
    print(f"logits_shape={tuple(output.logits.shape)}")
    print(f"target_shape={tuple(batch.target_ids.shape)}")
    print(f"per_token_loss_shape={tuple(token_losses.shape)}")
    print(f"per_token_loss={[round(value, 4) for value in token_losses[0].tolist()]}")
    print(f"model_loss={output.loss.item():.6f}")
    print(f"manual_mean_loss={manual_loss.item():.6f}")
    print(f"loss_match={torch.allclose(output.loss, manual_loss)}")
    print(f"future_weight_sum={future_weight_sum.item():.4f}")


def make_single_lm_batch(token_ids: list[int], *, block_size: int) -> LanguageModelingBatch:
    dataset = CharBlockDataset(token_ids, block_size=block_size)
    example = dataset[0]
    return LanguageModelingBatch(
        input_ids=example.input_ids.unsqueeze(0),
        target_ids=example.target_ids.unsqueeze(0),
    )


def per_token_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq_len, vocab_size].")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError("target_ids must have shape [batch, seq_len].")
    batch_size, seq_len, vocab_size = logits.shape
    losses = F.cross_entropy(
        logits.reshape(batch_size * seq_len, vocab_size),
        target_ids.reshape(batch_size * seq_len),
        reduction="none",
    )
    return losses.view(batch_size, seq_len)


def _future_positions(seq_len: int) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool).triu(diagonal=1)


if __name__ == "__main__":
    main()
