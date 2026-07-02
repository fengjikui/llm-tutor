from __future__ import annotations

import argparse

import torch

from llm_tutor.models.attention import make_causal_attention_mask
from llm_tutor.models.transformer import TransformerBlock


def main() -> None:
    parser = argparse.ArgumentParser(description="第 12 章：检查 Transformer block 的 shape")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x = torch.randn(args.batch_size, args.seq_len, args.embed_dim)
    block = TransformerBlock(embed_dim=args.embed_dim, num_heads=args.num_heads)
    mask = make_causal_attention_mask(args.seq_len, device=x.device)
    output, weights = block(x, attention_mask=mask, return_attention=True)

    print(f"x_shape={tuple(x.shape)}")
    print(f"output_shape={tuple(output.shape)}")
    print(f"attention_shape={tuple(weights.shape)}")
    print(f"block_shape_preserved={output.shape == x.shape}")
    future_mask = torch.ones(args.seq_len, args.seq_len, device=x.device).triu(diagonal=1).bool()
    future_weight_sum = weights[..., future_mask].sum().item()
    print(f"future_weight_sum={future_weight_sum:.4f}")


if __name__ == "__main__":
    main()
