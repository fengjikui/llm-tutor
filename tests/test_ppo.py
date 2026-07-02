import pytest
import torch

from llm_tutor.post_training.ppo import (
    TinyPromptPolicy,
    categorical_kl,
    gather_action_log_probs,
    ppo_clipped_loss,
    rule_reward,
)


def test_gather_action_log_probs_matches_manual_indexing() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    actions = torch.tensor([2, 0])

    selected = gather_action_log_probs(logits, actions)
    manual = torch.log_softmax(logits, dim=-1)[torch.arange(2), actions]

    torch.testing.assert_close(selected, manual)


def test_ppo_clipped_loss_uses_lower_surrogate_for_positive_advantage() -> None:
    logits = torch.log(torch.tensor([[0.9, 0.1]]))
    actions = torch.tensor([0])
    old_log_probs = torch.log(torch.tensor([0.5]))
    advantages = torch.tensor([1.0])

    result = ppo_clipped_loss(
        logits=logits,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        clip_epsilon=0.2,
    )

    torch.testing.assert_close(result.ratio, torch.tensor([1.8]))
    torch.testing.assert_close(result.policy_loss, torch.tensor(-1.2))


def test_ppo_clipped_loss_handles_negative_advantage() -> None:
    logits = torch.log(torch.tensor([[0.9, 0.1]]))
    actions = torch.tensor([0])
    old_log_probs = torch.log(torch.tensor([0.5]))
    advantages = torch.tensor([-1.0])

    result = ppo_clipped_loss(
        logits=logits,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        clip_epsilon=0.2,
    )

    torch.testing.assert_close(result.policy_loss, torch.tensor(1.8))


def test_ppo_clipped_loss_clips_low_ratio_for_negative_advantage() -> None:
    logits = torch.log(torch.tensor([[0.1, 0.9]]))
    actions = torch.tensor([0])
    old_log_probs = torch.log(torch.tensor([0.5]))
    advantages = torch.tensor([-1.0])

    result = ppo_clipped_loss(
        logits=logits,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        clip_epsilon=0.2,
    )

    torch.testing.assert_close(result.ratio, torch.tensor([0.2]))
    torch.testing.assert_close(result.policy_loss, torch.tensor(0.8))
    torch.testing.assert_close(result.clipped_fraction, torch.tensor(1.0))


def test_ppo_loss_rejects_negative_regularization_coefficients() -> None:
    logits = torch.zeros(1, 2)
    actions = torch.tensor([0])
    old_log_probs = torch.log(torch.tensor([0.5]))
    advantages = torch.tensor([1.0])

    with pytest.raises(ValueError, match="entropy_coef"):
        ppo_clipped_loss(
            logits=logits,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            entropy_coef=-0.1,
        )
    with pytest.raises(ValueError, match="kl_coef"):
        ppo_clipped_loss(
            logits=logits,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            kl_coef=-0.1,
        )


def test_categorical_kl_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    kl = categorical_kl(logits, logits)

    torch.testing.assert_close(kl, torch.zeros(1))


def test_rule_reward_scores_matching_actions() -> None:
    rewards = rule_reward(torch.tensor([0, 1, 0]), torch.tensor([0, 2, 0]))

    torch.testing.assert_close(rewards, torch.tensor([1.0, -0.2, 1.0]))


def test_tiny_prompt_policy_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="num_actions"):
        TinyPromptPolicy(num_prompts=2, num_actions=1)
