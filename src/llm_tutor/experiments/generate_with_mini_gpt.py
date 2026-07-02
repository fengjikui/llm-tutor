from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm_tutor.data.language_modeling import CharVocabulary
from llm_tutor.models.gpt import MiniGPT, MiniGPTConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="第 15 章：加载 checkpoint 并用 mini-GPT 生成文本")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="the ")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    vocab = _load_vocab(checkpoint)
    model = MiniGPT(MiniGPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    generator = torch.Generator().manual_seed(args.seed)
    prompt_ids = torch.tensor([vocab.encode(args.prompt)], dtype=torch.long)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        generator=generator,
    )
    print(f"loaded_epoch={checkpoint['epoch']} val_loss={checkpoint['val_loss']:.4f}")
    print(vocab.decode(generated[0]))


def _load_vocab(checkpoint: dict) -> CharVocabulary:
    id_to_token = tuple(checkpoint["vocab_id_to_token"])
    return CharVocabulary(
        id_to_token=id_to_token,
        token_to_id={token: idx for idx, token in enumerate(id_to_token)},
    )


if __name__ == "__main__":
    main()
