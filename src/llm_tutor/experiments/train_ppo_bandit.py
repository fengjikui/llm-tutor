from __future__ import annotations

import argparse

import torch

from llm_tutor.experiments.artifacts import (
    ExperimentArtifacts,
    add_artifact_args,
    args_to_config,
)
from llm_tutor.post_training.ppo import TinyPromptPolicy, ppo_clipped_loss, rule_reward

PROMPTS = ("say yes", "say no", "say ok")
ACTIONS = ("yes", "no", "ok")
TARGET_ACTIONS = torch.tensor([0, 1, 2], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="第 17 章：PPO clipped objective 的最小 bandit 实验"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    add_artifact_args(parser)
    args = parser.parse_args()

    artifacts = ExperimentArtifacts.create(
        args.output_dir,
        experiment_name="train_ppo_bandit",
        config=args_to_config(args),
    )
    with artifacts.capture_stdout():
        if artifacts.enabled:
            print(f"artifacts_dir={artifacts.run_dir}")
        _run(args, artifacts)


def _run(args: argparse.Namespace, artifacts: ExperimentArtifacts) -> None:
    torch.manual_seed(args.seed)
    policy = TinyPromptPolicy(num_prompts=len(PROMPTS), num_actions=len(ACTIONS))
    reference_logits = torch.zeros(len(PROMPTS), len(ACTIONS))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    prompt_ids = torch.arange(len(PROMPTS))

    print(f"prompts={len(PROMPTS)} actions={len(ACTIONS)}")
    for epoch in range(1, args.epochs + 1):
        with torch.no_grad():
            old_logits = policy(prompt_ids)
            old_dist = torch.distributions.Categorical(logits=old_logits)
            actions = old_dist.sample()
            old_log_probs = old_dist.log_prob(actions)
            rewards = rule_reward(actions, TARGET_ACTIONS)
            advantages = rewards

        for _ in range(args.ppo_epochs):
            logits = policy(prompt_ids)
            loss = ppo_clipped_loss(
                logits=logits,
                actions=actions,
                old_log_probs=old_log_probs,
                advantages=advantages,
                clip_epsilon=args.clip_epsilon,
                entropy_coef=args.entropy_coef,
                kl_coef=args.kl_coef,
                reference_logits=reference_logits,
            )
            optimizer.zero_grad()
            loss.total_loss.backward()
            optimizer.step()

        greedy_actions = policy(prompt_ids).argmax(dim=-1)
        greedy_reward = rule_reward(greedy_actions, TARGET_ACTIONS).mean().item()
        artifacts.append_metric(
            {
                "phase": "ppo",
                "epoch": float(epoch),
                "sampled_reward": rewards.mean().item(),
                "greedy_reward": greedy_reward,
                "total_loss": loss.total_loss.item(),
                "policy_loss": loss.policy_loss.item(),
                "ratio_min": loss.ratio.min().item(),
                "ratio_max": loss.ratio.max().item(),
                "clipped_fraction": loss.clipped_fraction.item(),
                "kl": loss.kl.item(),
                "entropy": loss.entropy.item(),
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} sampled_reward={rewards.mean().item():+.3f} "
                f"greedy_reward={greedy_reward:+.3f} "
                f"policy_loss={loss.policy_loss.item():+.4f} "
                f"ratio_min={loss.ratio.min().item():.3f} "
                f"ratio_max={loss.ratio.max().item():.3f} "
                f"clipped_fraction={loss.clipped_fraction.item():.3f} "
                f"kl={loss.kl.item():.4f} "
                f"entropy={loss.entropy.item():.4f}"
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
            "final_policy": final_policy,
            "final_greedy_reward": rule_reward(greedy_actions, TARGET_ACTIONS).mean().item(),
        }
    )


if __name__ == "__main__":
    main()
