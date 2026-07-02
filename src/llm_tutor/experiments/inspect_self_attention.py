from __future__ import annotations

import argparse

import torch

from llm_tutor.models.attention import (
    MultiHeadSelfAttention,
    make_causal_attention_mask,
    make_padding_attention_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="第 11 章：检查 self-attention 的 shape 和 mask")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x = torch.randn(args.batch_size, args.seq_len, args.embed_dim)
    token_ids = _make_demo_token_ids(args.batch_size, args.seq_len)
    attention = MultiHeadSelfAttention(embed_dim=args.embed_dim, num_heads=args.num_heads)

    padding_mask = make_padding_attention_mask(token_ids, pad_id=0)
    causal_mask = make_causal_attention_mask(args.seq_len, device=x.device)
    combined_mask = padding_mask & causal_mask
    output, weights = attention(x, attention_mask=combined_mask, return_attention=True)
    last_real_query_index = int((token_ids[0] != 0).sum().item() - 1)

    print(f"x_shape={tuple(x.shape)}")
    print(f"token_ids={token_ids.tolist()}")
    print(f"output_shape={tuple(output.shape)}")
    print(f"attention_shape={tuple(weights.shape)}")
    print(f"combined_mask_shape={tuple(combined_mask.shape)}")
    print(f"first_head_weights_row0={weights[0, 0, 0].detach().round(decimals=4).tolist()}")
    print(
        "last_real_query_can_attend_to="
        f"{combined_mask[0, 0, last_real_query_index].int().tolist()}"
    )


def _make_demo_token_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1.")

    token_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for batch_index in range(batch_size):
        real_len = max(1, seq_len - 1 - (batch_index % max(1, min(seq_len, 3))))
        token_ids[batch_index, :real_len] = torch.arange(1, real_len + 1) + batch_index * seq_len
    return token_ids


if __name__ == "__main__":
    main()
