import pytest
import torch

from llm_tutor.post_training.grpo import (
    group_relative_advantages,
    grpo_loss,
    sample_group_actions,
    verifiable_reward,
)
from llm_tutor.post_training.ppo import TinyPromptPolicy, categorical_kl


def test_group_relative_advantages_normalizes_each_group() -> None:
    rewards = torch.tensor([[0.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    advantages = group_relative_advantages(rewards)

    torch.testing.assert_close(advantages[0].mean(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(advantages[0].std(unbiased=False), torch.tensor(1.0))
    torch.testing.assert_close(advantages[1], torch.zeros(3))


def test_group_relative_advantages_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        group_relative_advantages(torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError, match="floating point"):
        group_relative_advantages(torch.tensor([[1, 0]]))


def test_verifiable_reward_scores_actions_against_prompt_targets() -> None:
    actions = torch.tensor([[0, 1, 0], [2, 1, 1]])
    targets = torch.tensor([0, 1])

    rewards = verifiable_reward(actions, targets)

    torch.testing.assert_close(rewards, torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]))


def test_grpo_loss_returns_group_advantages() -> None:
    policy_logits = torch.zeros(2, 3, 4)
    reference_logits = torch.zeros(2, 3, 4)
    actions = torch.tensor([[0, 1, 0], [2, 1, 1]])
    rewards = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    result = grpo_loss(
        policy_logits=policy_logits,
        reference_logits=reference_logits,
        actions=actions,
        rewards=rewards,
    )

    assert result.advantages.shape == rewards.shape
    torch.testing.assert_close(result.mean_reward, rewards.mean())
    torch.testing.assert_close(result.mean_kl, torch.tensor(0.0))


def test_grpo_kl_penalty_moves_policy_toward_reference() -> None:
    policy_logits = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]], requires_grad=True)
    reference_logits = torch.zeros(1, 2, 2)
    actions = torch.tensor([[0, 1]])
    rewards = torch.ones(1, 2)
    before_kl = categorical_kl(
        policy_logits.reshape(2, 2),
        reference_logits.reshape(2, 2),
    ).mean()

    result = grpo_loss(
        policy_logits=policy_logits,
        reference_logits=reference_logits,
        actions=actions,
        rewards=rewards,
        kl_coef=1.0,
    )
    result.loss.backward()
    with torch.no_grad():
        updated_logits = policy_logits - 0.5 * policy_logits.grad
    after_kl = categorical_kl(
        updated_logits.reshape(2, 2),
        reference_logits.reshape(2, 2),
    ).mean()

    assert after_kl < before_kl


def test_sample_group_actions_shapes() -> None:
    torch.manual_seed(42)
    policy = TinyPromptPolicy(num_prompts=2, num_actions=3)

    logits, actions = sample_group_actions(policy, torch.tensor([0, 1]), group_size=4)

    assert logits.shape == (2, 4, 3)
    assert actions.shape == (2, 4)


def test_sample_group_actions_uses_generator_for_reproducibility() -> None:
    policy = TinyPromptPolicy(num_prompts=2, num_actions=3)
    prompt_ids = torch.tensor([0, 1])

    _logits_a, actions_a = sample_group_actions(
        policy,
        prompt_ids,
        group_size=4,
        generator=torch.Generator().manual_seed(42),
    )
    _logits_b, actions_b = sample_group_actions(
        policy,
        prompt_ids,
        group_size=4,
        generator=torch.Generator().manual_seed(42),
    )

    torch.testing.assert_close(actions_a, actions_b)


def test_grpo_rejects_invalid_group_size() -> None:
    policy = TinyPromptPolicy(num_prompts=2, num_actions=3)

    with pytest.raises(ValueError, match="group_size"):
        sample_group_actions(policy, torch.tensor([0, 1]), group_size=1)
