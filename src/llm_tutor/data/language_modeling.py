from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

TINY_STORY_CORPUS = """
the model reads tokens one by one.
the small model learns that words can follow words.
attention lets each token look back at useful earlier tokens.
the decoder never looks at future tokens during training.
loss becomes smaller when the next token prediction improves.
we train a tiny gpt so the full loop is easy to inspect.
the goal is not a smart chatbot yet.
the goal is to understand data, targets, logits, and loss.
"""


@dataclass(frozen=True)
class CharVocabulary:
    id_to_token: tuple[str, ...]
    token_to_id: dict[str, int]

    @classmethod
    def from_text(cls, text: str) -> CharVocabulary:
        tokens = tuple(sorted(set(text)))
        if not tokens:
            raise ValueError("text must contain at least one character.")
        return cls(id_to_token=tokens, token_to_id={token: idx for idx, token in enumerate(tokens)})

    def encode(self, text: str) -> list[int]:
        try:
            return [self.token_to_id[token] for token in text]
        except KeyError as error:
            raise ValueError(f"Unknown character: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return "".join(self.id_to_token[token_id] for token_id in token_ids)

    @property
    def size(self) -> int:
        return len(self.id_to_token)


@dataclass(frozen=True)
class LanguageModelingBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor


@dataclass(frozen=True)
class LanguageModelingData:
    vocab: CharVocabulary
    train_loader: DataLoader
    val_loader: DataLoader
    block_size: int
    train_token_range: tuple[int, int]
    val_token_range: tuple[int, int]


class CharBlockDataset(Dataset[LanguageModelingBatch]):
    """Return fixed-length causal language-modeling examples.

    For a token window `[t0, t1, ..., tN]`, the model input is
    `[t0, ..., tN-1]` and the target is `[t1, ..., tN]`.
    """

    def __init__(self, token_ids: list[int], *, block_size: int) -> None:
        if block_size < 2:
            raise ValueError("block_size must be >= 2.")
        if len(token_ids) <= block_size:
            raise ValueError("token_ids must contain more tokens than block_size.")
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> LanguageModelingBatch:
        chunk = self.token_ids[index : index + self.block_size + 1]
        return LanguageModelingBatch(input_ids=chunk[:-1], target_ids=chunk[1:])


def load_tiny_language_modeling_data(
    *,
    block_size: int = 48,
    batch_size: int = 16,
    seed: int = 42,
) -> LanguageModelingData:
    text = _normalize_text(TINY_STORY_CORPUS)
    vocab = CharVocabulary.from_text(text)
    token_ids = vocab.encode(text)
    train_ids, val_ids, train_range, val_range = split_token_ids_for_lm(
        token_ids,
        block_size=block_size,
        val_fraction=0.2,
    )
    train_dataset = CharBlockDataset(train_ids, block_size=block_size)
    val_dataset = CharBlockDataset(val_ids, block_size=block_size)
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_collate_lm_batches,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_lm_batches,
    )
    return LanguageModelingData(
        vocab=vocab,
        train_loader=train_loader,
        val_loader=val_loader,
        block_size=block_size,
        train_token_range=train_range,
        val_token_range=val_range,
    )


def split_token_ids_for_lm(
    token_ids: list[int],
    *,
    block_size: int,
    val_fraction: float = 0.2,
) -> tuple[list[int], list[int], tuple[int, int], tuple[int, int]]:
    """Split a language-modeling corpus by contiguous token ranges.

    The gap prevents near-duplicate sliding windows from appearing in both train and val.
    """

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    min_tokens_per_split = block_size + 2
    required_tokens = min_tokens_per_split * 2 + block_size
    if len(token_ids) < required_tokens:
        raise ValueError(
            f"Need at least {required_tokens} tokens for train/val split "
            f"with block_size={block_size}."
        )

    val_token_count = max(min_tokens_per_split, int(len(token_ids) * val_fraction))
    train_end = len(token_ids) - val_token_count - block_size
    if train_end < min_tokens_per_split:
        train_end = min_tokens_per_split
    val_start = train_end + block_size
    if len(token_ids) - val_start < min_tokens_per_split:
        raise ValueError("Not enough tokens remain for validation after applying the split gap.")

    train_ids = token_ids[:train_end]
    val_ids = token_ids[val_start:]
    return train_ids, val_ids, (0, train_end), (val_start, len(token_ids))


def _collate_lm_batches(batches: list[LanguageModelingBatch]) -> LanguageModelingBatch:
    return LanguageModelingBatch(
        input_ids=torch.stack([batch.input_ids for batch in batches]),
        target_ids=torch.stack([batch.target_ids for batch in batches]),
    )


def _normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines()) + "\n"
