from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm_tutor.data.language_modeling import CharVocabulary
from llm_tutor.experiments.artifacts import (
    ExperimentArtifacts,
    add_artifact_args,
    args_to_config,
)
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
    add_artifact_args(parser)
    args = parser.parse_args()

    artifacts = ExperimentArtifacts.create(
        args.output_dir,
        experiment_name="generate_with_mini_gpt",
        config=args_to_config(args),
    )
    with artifacts.capture_stdout():
        if artifacts.enabled:
            print(f"artifacts_dir={artifacts.run_dir}")
        _run(args, artifacts)


def _run(args: argparse.Namespace, artifacts: ExperimentArtifacts) -> None:
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
    generated_text = vocab.decode(generated[0])
    print(f"loaded_epoch={checkpoint['epoch']} val_loss={checkpoint['val_loss']:.4f}")
    print(generated_text)
    artifacts.write_summary(
        {
            "checkpoint_path": args.checkpoint_path,
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_val_loss": checkpoint["val_loss"],
            "prompt": args.prompt,
            "sampling": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "seed": args.seed,
            },
            "generated_text": generated_text,
        }
    )


def _load_vocab(checkpoint: dict) -> CharVocabulary:
    id_to_token = tuple(checkpoint["vocab_id_to_token"])
    return CharVocabulary(
        id_to_token=id_to_token,
        token_to_id={token: idx for idx, token in enumerate(id_to_token)},
    )


if __name__ == "__main__":
    main()
