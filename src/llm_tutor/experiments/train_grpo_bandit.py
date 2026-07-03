from __future__ import annotations

import argparse

import torch

from llm_tutor.experiments.artifacts import (
    ExperimentArtifacts,
    add_artifact_args,
    args_to_config,
)
from llm_tutor.post_training.grpo import grpo_loss, sample_group_actions, verifiable_reward
from llm_tutor.post_training.ppo import TinyPromptPolicy

PROMPTS = ("2+2?", "capital of france?", "first letter of abc?")
ACTIONS = ("4", "paris", "a")
TARGET_ACTIONS = torch.tensor([0, 1, 2], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="第 19 章：GRPO 组内相对优势的最小实验")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    add_artifact_args(parser)
    args = parser.parse_args()

    artifacts = ExperimentArtifacts.create(
        args.output_dir,
        experiment_name="train_grpo_bandit",
        config=args_to_config(args),
    )
    with artifacts.capture_stdout():
        if artifacts.enabled:
            print(f"artifacts_dir={artifacts.run_dir}")
        _run(args, artifacts)


def _run(args: argparse.Namespace, artifacts: ExperimentArtifacts) -> None:
    torch.manual_seed(args.seed)
    policy = TinyPromptPolicy(num_prompts=len(PROMPTS), num_actions=len(ACTIONS))
    reference = TinyPromptPolicy(num_prompts=len(PROMPTS), num_actions=len(ACTIONS))
    reference.load_state_dict(policy.state_dict())
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    prompt_ids = torch.arange(len(PROMPTS))

    print(f"prompts={len(PROMPTS)} actions={len(ACTIONS)} group_size={args.group_size}")
    for epoch in range(1, args.epochs + 1):
        policy_logits, actions = sample_group_actions(
            policy,
            prompt_ids,
            group_size=args.group_size,
        )
        repeated_prompt_ids = prompt_ids.repeat_interleave(args.group_size)
        with torch.no_grad():
            reference_logits = reference(repeated_prompt_ids).view(
                len(PROMPTS),
                args.group_size,
                -1,
            )
            rewards = verifiable_reward(actions, TARGET_ACTIONS)
        result = grpo_loss(
            policy_logits=policy_logits,
            reference_logits=reference_logits,
            actions=actions,
            rewards=rewards,
            kl_coef=args.kl_coef,
        )
        optimizer.zero_grad()
        result.loss.backward()
        optimizer.step()
        greedy_actions = policy(prompt_ids).argmax(dim=-1)
        greedy_reward = verifiable_reward(greedy_actions.unsqueeze(-1), TARGET_ACTIONS).mean()
        artifacts.append_metric(
            {
                "phase": "grpo",
                "epoch": float(epoch),
                "mean_reward": result.mean_reward.item(),
                "greedy_reward": greedy_reward.item(),
                "loss": result.loss.item(),
                "adv_mean": result.advantages.mean().item(),
                "adv_std": result.advantages.std(unbiased=False).item(),
                "mean_kl": result.mean_kl.item(),
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} mean_reward={result.mean_reward.item():.3f} "
                f"greedy_reward={greedy_reward.item():.3f} "
                f"adv_mean={result.advantages.mean().item():+.3f} "
                f"adv_std={result.advantages.std(unbiased=False).item():.3f} "
                f"mean_kl={result.mean_kl.item():+.4f}"
            )

    print("\npolicy")
    greedy_actions = policy(prompt_ids).argmax(dim=-1)
    final_policy = []
    for prompt, action_id, target_id in zip(
        PROMPTS,
        greedy_actions.tolist(),
        TARGET_ACTIONS.tolist(),
        strict=True,
    ):
        print(f"prompt={prompt!r} action={ACTIONS[action_id]!r} target={ACTIONS[target_id]!r}")
        final_policy.append(
            {
                "prompt": prompt,
                "action": ACTIONS[action_id],
                "target": ACTIONS[target_id],
                "is_correct": action_id == target_id,
            }
        )
    artifacts.write_summary(
        {
            "prompts": list(PROMPTS),
            "actions": list(ACTIONS),
            "target_actions": [ACTIONS[index] for index in TARGET_ACTIONS.tolist()],
            "group_size": args.group_size,
            "kl_coef": args.kl_coef,
            "final_policy": final_policy,
            "final_greedy_reward": verifiable_reward(
                greedy_actions.unsqueeze(-1),
                TARGET_ACTIONS,
            )
            .mean()
            .item(),
        }
    )


if __name__ == "__main__":
    main()
