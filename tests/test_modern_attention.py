from __future__ import annotations

import torch

from llm_tutor.models.modern_attention import (
    KVCache,
    LatentKVProjection,
    apply_rope,
    build_rope_cache,
    grouped_query_attention,
    repeat_kv,
    sliding_window_causal_mask,
)


def test_rope_preserves_pairwise_norms() -> None:
    x = torch.randn(2, 3, 5, 8)
    cos, sin = build_rope_cache(seq_len=5, head_dim=8)

    rotated = apply_rope(x, cos, sin)

    before = x[..., 0::2].square() + x[..., 1::2].square()
    after = rotated[..., 0::2].square() + rotated[..., 1::2].square()
    assert torch.allclose(before, after, atol=1e-5)


def test_repeat_kv_expands_to_query_heads() -> None:
    key = torch.randn(2, 2, 4, 8)

    expanded = repeat_kv(key, num_query_heads=6)

    assert expanded.shape == (2, 6, 4, 8)
    assert torch.equal(expanded[:, 0], key[:, 0])
    assert torch.equal(expanded[:, 3], key[:, 1])


def test_grouped_query_attention_shape() -> None:
    query = torch.randn(2, 4, 3, 8)
    key = torch.randn(2, 2, 3, 8)
    value = torch.randn(2, 2, 3, 8)

    output = grouped_query_attention(query, key, value)

    assert output.shape == query.shape


def test_kv_cache_appends_sequence_axis() -> None:
    cache = KVCache()
    key_1 = torch.randn(1, 2, 3, 4)
    value_1 = torch.randn(1, 2, 3, 4)
    key_2 = torch.randn(1, 2, 1, 4)
    value_2 = torch.randn(1, 2, 1, 4)

    cache.append(key_1, value_1)
    key, value = cache.append(key_2, value_2)

    assert key.shape == (1, 2, 4, 4)
    assert value.shape == (1, 2, 4, 4)


def test_sliding_window_mask_hides_future_and_far_past() -> None:
    mask = sliding_window_causal_mask(seq_len=5, window_size=2)[0, 0]

    assert mask[3, 3]
    assert mask[3, 2]
    assert not mask[3, 1]
    assert not mask[3, 4]


def test_latent_kv_projection_shapes() -> None:
    module = LatentKVProjection(embed_dim=16, latent_dim=6, num_kv_heads=2, head_dim=4)
    x = torch.randn(3, 5, 16)

    latent, key, value = module(x)

    assert latent.shape == (3, 5, 6)
    assert key.shape == (3, 2, 5, 4)
    assert value.shape == (3, 2, 5, 4)
