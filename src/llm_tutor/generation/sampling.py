from __future__ import annotations

import torch


def filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Filter logits before sampling.

    Shape:
    - logits: [batch, vocab_size]
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab_size].")
    filtered = logits.clone()
    vocab_size = filtered.shape[-1]

    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        kept = min(top_k, vocab_size)
        threshold = torch.topk(filtered, k=kept, dim=-1).values[:, -1].unsqueeze(-1)
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if top_p < 1:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            remove_sorted = cumulative_probs > top_p
            remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
            remove_sorted[:, 0] = False
            remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
            filtered = filtered.masked_fill(remove, float("-inf"))

    if torch.isneginf(filtered).all(dim=-1).any():
        raise ValueError("Filtering removed every token for at least one batch row.")
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token id from logits.

    Output shape: [batch, 1]
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    filtered = filter_logits(logits / temperature, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
