import pytest
import torch

from llm_tutor.models.attention import make_causal_attention_mask
from llm_tutor.models.transformer import TransformerBlock


def test_transformer_block_preserves_shape() -> None:
    block = TransformerBlock(embed_dim=16, num_heads=4)
    x = torch.randn(2, 5, 16)

    output, weights = block(x, return_attention=True)

    assert output.shape == x.shape
    assert weights.shape == (2, 4, 5, 5)


def test_transformer_block_respects_causal_mask() -> None:
    torch.manual_seed(42)
    block = TransformerBlock(embed_dim=16, num_heads=4)
    x = torch.randn(2, 5, 16)
    mask = make_causal_attention_mask(5)

    _output, weights = block(x, attention_mask=mask, return_attention=True)

    future_mask = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)
    assert torch.all(weights[..., future_mask] == 0)


def test_transformer_block_rejects_invalid_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TransformerBlock(embed_dim=10, num_heads=4)


def test_transformer_block_residual_path_preserves_input_when_branches_are_zero() -> None:
    block = TransformerBlock(embed_dim=16, num_heads=4)
    x = torch.randn(2, 5, 16)

    for parameter in block.attention.parameters():
        parameter.data.zero_()
    for parameter in block.mlp.parameters():
        parameter.data.zero_()

    output = block(x)

    torch.testing.assert_close(output, x)
