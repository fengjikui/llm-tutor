from __future__ import annotations

from dataclasses import dataclass

import torch

from llm_tutor.post_training.ppo import TinyPromptPolicy, categorical_kl, gather_action_log_probs


@dataclass(frozen=True)
class GRPOLossResult:
    loss: torch.Tensor
    mean_reward: torch.Tensor
    mean_kl: torch.Tensor
    advantages: torch.Tensor


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize rewards within each prompt group.

    Shape:
    - rewards: [num_prompts, group_size]
    """

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [num_prompts, group_size].")
    if not torch.is_floating_point(rewards):
        raise ValueError("rewards must be a floating point tensor.")
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    return torch.where(std > eps, centered / (std + eps), torch.zeros_like(centered))


def grpo_loss(
    *,
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    kl_coef: float = 0.05,
) -> GRPOLossResult:
    if kl_coef < 0:
        raise ValueError("kl_coef must be >= 0.")
    if policy_logits.shape != reference_logits.shape:
        raise ValueError("policy_logits and reference_logits must have the same shape.")
    if actions.shape != rewards.shape:
        raise ValueError("actions and rewards must have the same shape.")
    if policy_logits.shape[:2] != actions.shape:
        raise ValueError("policy_logits must have shape [num_prompts, group_size, num_actions].")

    num_prompts, group_size, num_actions = policy_logits.shape
    flat_policy_logits = policy_logits.reshape(num_prompts * group_size, num_actions)
    flat_reference_logits = reference_logits.reshape(num_prompts * group_size, num_actions)
    flat_actions = actions.reshape(num_prompts * group_size)

    policy_log_probs = gather_action_log_probs(flat_policy_logits, flat_actions).view(
        num_prompts,
        group_size,
    )
    advantages = group_relative_advantages(rewards)
    policy_loss = -(policy_log_probs * advantages.detach()).mean()
    full_kl = categorical_kl(flat_policy_logits, flat_reference_logits).view(
        num_prompts,
        group_size,
    )
    loss = policy_loss + kl_coef * full_kl.mean()
    return GRPOLossResult(
        loss=loss,
        mean_reward=rewards.mean(),
        mean_kl=full_kl.mean(),
        advantages=advantages,
    )


def sample_group_actions(
    policy: TinyPromptPolicy,
    prompt_ids: torch.Tensor,
    *,
    group_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_size < 2:
        raise ValueError("group_size must be >= 2.")
    repeated_prompt_ids = prompt_ids.repeat_interleave(group_size)
    logits = policy(repeated_prompt_ids)
    probs = torch.softmax(logits, dim=-1)
    actions = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return (
        logits.view(prompt_ids.shape[0], group_size, -1),
        actions.view(prompt_ids.shape[0], group_size),
    )


def verifiable_reward(actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim != 2:
        raise ValueError("actions must have shape [num_prompts, group_size].")
    if target_actions.shape != actions.shape[:1]:
        raise ValueError("target_actions must have shape [num_prompts].")
    targets = target_actions.unsqueeze(-1)
    positive = torch.ones_like(actions, dtype=torch.float32)
    negative = torch.zeros_like(actions, dtype=torch.float32)
    return torch.where(actions == targets, positive, negative)
