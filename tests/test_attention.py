import pytest
import torch

from llm_tutor.models.attention import (
    MultiHeadSelfAttention,
    make_causal_attention_mask,
    make_padding_attention_mask,
    scaled_dot_product_attention,
)


def test_multi_head_self_attention_shapes() -> None:
    attention = MultiHeadSelfAttention(embed_dim=16, num_heads=4)
    x = torch.randn(2, 5, 16)

    output, weights = attention(x, return_attention=True)

    assert output.shape == x.shape
    assert weights.shape == (2, 4, 5, 5)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 4, 5))


def test_padding_mask_zeroes_padded_key_weights() -> None:
    torch.manual_seed(42)
    attention = MultiHeadSelfAttention(embed_dim=8, num_heads=2)
    x = torch.randn(1, 4, 8)
    token_ids = torch.tensor([[1, 2, 0, 0]])
    mask = make_padding_attention_mask(token_ids, pad_id=0)

    _output, weights = attention(x, attention_mask=mask, return_attention=True)

    torch.testing.assert_close(weights[..., 2:], torch.zeros_like(weights[..., 2:]))


def test_causal_mask_prevents_future_attention() -> None:
    torch.manual_seed(42)
    attention = MultiHeadSelfAttention(embed_dim=8, num_heads=2)
    x = torch.randn(1, 4, 8)
    mask = make_causal_attention_mask(4)

    _output, weights = attention(x, attention_mask=mask, return_attention=True)

    future_mask = torch.ones(4, 4, dtype=torch.bool).triu(diagonal=1)
    assert torch.all(weights[..., future_mask] == 0)


def test_combined_padding_and_causal_mask_broadcasts() -> None:
    torch.manual_seed(42)
    attention = MultiHeadSelfAttention(embed_dim=8, num_heads=2)
    x = torch.randn(2, 4, 8)
    token_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    padding_mask = make_padding_attention_mask(token_ids, pad_id=0)
    causal_mask = make_causal_attention_mask(4)

    _output, weights = attention(
        x,
        attention_mask=padding_mask & causal_mask,
        return_attention=True,
    )

    assert weights.shape == (2, 2, 4, 4)
    torch.testing.assert_close(weights[0, :, :, 3], torch.zeros_like(weights[0, :, :, 3]))
    torch.testing.assert_close(weights[1, :, :, 2:], torch.zeros_like(weights[1, :, :, 2:]))


def test_attention_rejects_fully_masked_query() -> None:
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 2, 4)
    value = torch.randn(1, 1, 2, 4)
    mask = torch.tensor([[[[True, False], [False, False]]]])

    with pytest.raises(ValueError, match="at least one key"):
        scaled_dot_product_attention(query, key, value, attention_mask=mask)


def test_attention_rejects_non_bool_mask() -> None:
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 2, 4)
    value = torch.randn(1, 1, 2, 4)
    mask = torch.ones(1, 1, 2, 2)

    with pytest.raises(ValueError, match="boolean"):
        scaled_dot_product_attention(query, key, value, attention_mask=mask)


def test_attention_rejects_unbroadcastable_mask() -> None:
    query = torch.randn(2, 1, 3, 4)
    key = torch.randn(2, 1, 3, 4)
    value = torch.randn(2, 1, 3, 4)
    mask = torch.ones(5, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match="cannot broadcast"):
        scaled_dot_product_attention(query, key, value, attention_mask=mask)
