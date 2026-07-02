import pytest
import torch
from torch.nn import functional as F

from llm_tutor.post_training.dpo import dpo_loss, make_policy_pair, make_tiny_preference_batch


def test_dpo_loss_matches_manual_formula() -> None:
    policy_logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
    reference_logits = torch.log(torch.tensor([[0.4, 0.4, 0.2]]))
    chosen = torch.tensor([0])
    rejected = torch.tensor([1])

    result = dpo_loss(
        policy_logits=policy_logits,
        reference_logits=reference_logits,
        chosen_actions=chosen,
        rejected_actions=rejected,
        beta=0.5,
    )
    manual_logit = 0.5 * ((torch.log(torch.tensor(0.7)) - torch.log(torch.tensor(0.2))) - 0.0)
    manual_loss = -F.logsigmoid(manual_logit)

    torch.testing.assert_close(result.logits, manual_logit.view(1))
    torch.testing.assert_close(result.loss, manual_loss)


def test_dpo_loss_is_log_two_when_policy_matches_reference() -> None:
    logits = torch.randn(3, 4)
    chosen = torch.tensor([0, 1, 2])
    rejected = torch.tensor([1, 2, 3])

    result = dpo_loss(
        policy_logits=logits,
        reference_logits=logits,
        chosen_actions=chosen,
        rejected_actions=rejected,
    )

    torch.testing.assert_close(result.loss, torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(result.preference_accuracy, torch.tensor(0.0))


def test_dpo_rejects_invalid_beta() -> None:
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(
            policy_logits=torch.zeros(1, 2),
            reference_logits=torch.zeros(1, 2),
            chosen_actions=torch.tensor([0]),
            rejected_actions=torch.tensor([1]),
            beta=0,
        )


def test_make_policy_pair_freezes_reference() -> None:
    policy, reference = make_policy_pair(num_prompts=2, num_actions=3)

    assert all(parameter.requires_grad for parameter in policy.parameters())
    assert not any(parameter.requires_grad for parameter in reference.parameters())


def test_reference_policy_stays_frozen_after_dpo_step() -> None:
    torch.manual_seed(42)
    batch = make_tiny_preference_batch()
    policy, reference = make_policy_pair(num_prompts=3, num_actions=3)
    before = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)

    result = dpo_loss(
        policy_logits=policy(batch.prompt_ids),
        reference_logits=reference(batch.prompt_ids),
        chosen_actions=batch.chosen_actions,
        rejected_actions=batch.rejected_actions,
    )
    optimizer.zero_grad()
    result.loss.backward()
    optimizer.step()

    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_dpo_training_step_improves_preference_direction() -> None:
    torch.manual_seed(42)
    batch = make_tiny_preference_batch()
    policy, reference = make_policy_pair(num_prompts=3, num_actions=3)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    initial = dpo_loss(
        policy_logits=policy(batch.prompt_ids),
        reference_logits=reference(batch.prompt_ids),
        chosen_actions=batch.chosen_actions,
        rejected_actions=batch.rejected_actions,
    )

    for _ in range(5):
        result = dpo_loss(
            policy_logits=policy(batch.prompt_ids),
            reference_logits=reference(batch.prompt_ids),
            chosen_actions=batch.chosen_actions,
            rejected_actions=batch.rejected_actions,
        )
        optimizer.zero_grad()
        result.loss.backward()
        optimizer.step()

    final = dpo_loss(
        policy_logits=policy(batch.prompt_ids),
        reference_logits=reference(batch.prompt_ids),
        chosen_actions=batch.chosen_actions,
        rejected_actions=batch.rejected_actions,
    )

    assert final.logits.mean() > initial.logits.mean()


def test_tiny_preference_batch_shapes() -> None:
    batch = make_tiny_preference_batch()

    assert batch.prompt_ids.shape == batch.chosen_actions.shape == batch.rejected_actions.shape
