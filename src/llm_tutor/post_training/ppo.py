from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PPOLossResult:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    kl: torch.Tensor
    ratio: torch.Tensor
    clipped_fraction: torch.Tensor


class TinyPromptPolicy(nn.Module):
    """A tiny categorical policy for PPO teaching examples."""

    def __init__(self, num_prompts: int, num_actions: int) -> None:
        super().__init__()
        if num_prompts < 1:
            raise ValueError("num_prompts must be >= 1.")
        if num_actions < 2:
            raise ValueError("num_actions must be >= 2.")
        self.logits = nn.Embedding(num_prompts, num_actions)

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        return self.logits(prompt_ids)


def gather_action_log_probs(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, num_actions].")
    if actions.shape != logits.shape[:1]:
        raise ValueError("actions must have shape [batch].")
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)


def categorical_kl(policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != reference_logits.shape:
        raise ValueError("policy_logits and reference_logits must have the same shape.")
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    policy_probs = policy_log_probs.exp()
    return (policy_probs * (policy_log_probs - reference_log_probs)).sum(dim=-1)


def ppo_clipped_loss(
    *,
    logits: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.0,
    kl_coef: float = 0.0,
    reference_logits: torch.Tensor | None = None,
) -> PPOLossResult:
    if clip_epsilon <= 0:
        raise ValueError("clip_epsilon must be > 0.")
    if entropy_coef < 0:
        raise ValueError("entropy_coef must be >= 0.")
    if kl_coef < 0:
        raise ValueError("kl_coef must be >= 0.")
    if old_log_probs.shape != actions.shape:
        raise ValueError("old_log_probs must have shape [batch].")
    if advantages.shape != actions.shape:
        raise ValueError("advantages must have shape [batch].")

    new_log_probs = gather_action_log_probs(logits, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    clipped = clipped_ratio * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    clipped_fraction = ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float().mean()

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    kl = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if reference_logits is not None:
        kl = categorical_kl(logits, reference_logits).mean()

    total_loss = policy_loss + kl_coef * kl - entropy_coef * entropy
    return PPOLossResult(
        total_loss=total_loss,
        policy_loss=policy_loss,
        entropy=entropy,
        kl=kl,
        ratio=ratio,
        clipped_fraction=clipped_fraction,
    )


def rule_reward(actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
    if actions.shape != target_actions.shape:
        raise ValueError("actions and target_actions must have the same shape.")
    positive = torch.ones_like(actions, dtype=torch.float32)
    negative = torch.full_like(actions, fill_value=-0.2, dtype=torch.float32)
    return torch.where(actions == target_actions, positive, negative)
