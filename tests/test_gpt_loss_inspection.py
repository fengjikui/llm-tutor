import pytest
import torch
from torch.nn import functional as F

from llm_tutor.experiments.inspect_gpt_loss import (
    main,
    make_single_lm_batch,
    per_token_cross_entropy,
)


def test_make_single_lm_batch_uses_shifted_targets() -> None:
    batch = make_single_lm_batch([0, 1, 2, 3, 4], block_size=3)

    assert batch.input_ids.tolist() == [[0, 1, 2]]
    assert batch.target_ids.tolist() == [[1, 2, 3]]


def test_per_token_cross_entropy_matches_torch_reduction_none() -> None:
    torch.manual_seed(42)
    logits = torch.randn(2, 3, 5)
    target_ids = torch.tensor([[0, 1, 2], [3, 4, 0]])

    losses = per_token_cross_entropy(logits, target_ids)
    expected = F.cross_entropy(logits.reshape(-1, 5), target_ids.reshape(-1), reduction="none")

    torch.testing.assert_close(losses, expected.view(2, 3))


def test_per_token_cross_entropy_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="target_ids"):
        per_token_cross_entropy(torch.randn(2, 3, 5), torch.randint(0, 5, (2, 2)))


def test_inspect_gpt_loss_prints_key_fields(capsys: pytest.CaptureFixture[str]) -> None:
    main([])

    output = capsys.readouterr().out

    assert "x_text='the mode'" in output
    assert "y_text='he model'" in output
    assert "per_token_loss=[" in output
    assert "loss_match=True" in output
    assert "future_weight_sum=0.0000" in output
