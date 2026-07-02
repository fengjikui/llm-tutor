from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from llm_tutor.post_training.ppo import TinyPromptPolicy, gather_action_log_probs


@dataclass(frozen=True)
class PreferenceBatch:
    prompt_ids: torch.Tensor
    chosen_actions: torch.Tensor
    rejected_actions: torch.Tensor


@dataclass(frozen=True)
class DPOLossResult:
    loss: torch.Tensor
    preference_accuracy: torch.Tensor
    policy_log_ratio: torch.Tensor
    reference_log_ratio: torch.Tensor
    logits: torch.Tensor


def dpo_loss(
    *,
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    chosen_actions: torch.Tensor,
    rejected_actions: torch.Tensor,
    beta: float = 0.1,
) -> DPOLossResult:
    if beta <= 0:
        raise ValueError("beta must be > 0.")
    if policy_logits.shape != reference_logits.shape:
        raise ValueError("policy_logits and reference_logits must have the same shape.")
    if chosen_actions.shape != rejected_actions.shape:
        raise ValueError("chosen_actions and rejected_actions must have the same shape.")

    policy_chosen = gather_action_log_probs(policy_logits, chosen_actions)
    policy_rejected = gather_action_log_probs(policy_logits, rejected_actions)
    reference_chosen = gather_action_log_probs(reference_logits, chosen_actions)
    reference_rejected = gather_action_log_probs(reference_logits, rejected_actions)

    policy_log_ratio = policy_chosen - policy_rejected
    reference_log_ratio = reference_chosen - reference_rejected
    logits = beta * (policy_log_ratio - reference_log_ratio)
    loss = -F.logsigmoid(logits).mean()
    preference_accuracy = (logits > 0).float().mean()
    return DPOLossResult(
        loss=loss,
        preference_accuracy=preference_accuracy,
        policy_log_ratio=policy_log_ratio,
        reference_log_ratio=reference_log_ratio,
        logits=logits,
    )


def make_tiny_preference_batch() -> PreferenceBatch:
    return PreferenceBatch(
        prompt_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        chosen_actions=torch.tensor([0, 1, 2], dtype=torch.long),
        rejected_actions=torch.tensor([1, 0, 0], dtype=torch.long),
    )


def make_policy_pair(
    *,
    num_prompts: int,
    num_actions: int,
) -> tuple[TinyPromptPolicy, TinyPromptPolicy]:
    policy = TinyPromptPolicy(num_prompts=num_prompts, num_actions=num_actions)
    reference = TinyPromptPolicy(num_prompts=num_prompts, num_actions=num_actions)
    reference.load_state_dict(policy.state_dict())
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return policy, reference
