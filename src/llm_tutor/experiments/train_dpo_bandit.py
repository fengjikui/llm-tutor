from __future__ import annotations

import argparse

import torch

from llm_tutor.post_training.dpo import dpo_loss, make_policy_pair, make_tiny_preference_batch

PROMPTS = ("say yes", "say no", "say ok")
ACTIONS = ("yes", "no", "ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="第 18 章：DPO 偏好优化的最小 bandit 实验")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    batch = make_tiny_preference_batch()
    policy, reference = make_policy_pair(num_prompts=len(PROMPTS), num_actions=len(ACTIONS))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    print(f"preferences={batch.prompt_ids.numel()} actions={len(ACTIONS)}")
    for epoch in range(1, args.epochs + 1):
        result = dpo_loss(
            policy_logits=policy(batch.prompt_ids),
            reference_logits=reference(batch.prompt_ids),
            chosen_actions=batch.chosen_actions,
            rejected_actions=batch.rejected_actions,
            beta=args.beta,
        )
        optimizer.zero_grad()
        result.loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} dpo_loss={result.loss.item():.4f} "
                f"pref_acc={result.preference_accuracy.item():.3f} "
                f"logit_mean={result.logits.mean().item():+.4f}"
            )

    print("\npolicy")
    greedy_actions = policy(batch.prompt_ids).argmax(dim=-1)
    for prompt, action_id, chosen_id, rejected_id in zip(
        PROMPTS,
        greedy_actions.tolist(),
        batch.chosen_actions.tolist(),
        batch.rejected_actions.tolist(),
        strict=True,
    ):
        print(
            f"prompt={prompt!r} action={ACTIONS[action_id]!r} "
            f"chosen={ACTIONS[chosen_id]!r} rejected={ACTIONS[rejected_id]!r}"
        )


if __name__ == "__main__":
    main()
