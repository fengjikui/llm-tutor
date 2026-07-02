from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from llm_tutor.data.language_modeling import CharVocabulary

IGNORE_INDEX = -100


TINY_INSTRUCTION_DATA = (
    ("repeat: hi", "hi"),
    ("repeat: ok", "ok"),
    ("upper: hi", "HI"),
    ("upper: ok", "OK"),
    ("answer yes", "yes"),
    ("answer no", "no"),
)


@dataclass(frozen=True)
class InstructionExample:
    prompt: str
    response: str


@dataclass(frozen=True)
class SFTBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    loss_mask: torch.Tensor


class SFTDataset(Dataset[SFTBatch]):
    """Instruction tuning dataset with response-only labels."""

    def __init__(
        self,
        examples: list[InstructionExample],
        vocab: CharVocabulary,
        *,
        block_size: int,
    ) -> None:
        if block_size < 4:
            raise ValueError("block_size must be >= 4.")
        self.examples = examples
        self.vocab = vocab
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SFTBatch:
        example = self.examples[index]
        prompt_text = format_prompt(example.prompt)
        response_text = example.response + "\n"
        full_text = prompt_text + response_text
        token_ids = self.vocab.encode(full_text)
        prompt_len = len(self.vocab.encode(prompt_text))
        if len(token_ids) > self.block_size + 1:
            raise ValueError("Formatted example is longer than block_size + 1.")

        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        # Mask positions whose target token belongs to the prompt span.
        masked_targets = [
            target_id if position >= prompt_len - 1 else IGNORE_INDEX
            for position, target_id in enumerate(target_ids)
        ]
        loss_mask = [target_id != IGNORE_INDEX for target_id in masked_targets]
        return SFTBatch(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            target_ids=torch.tensor(masked_targets, dtype=torch.long),
            loss_mask=torch.tensor(loss_mask, dtype=torch.bool),
        )


def build_sft_vocab(examples: list[InstructionExample]) -> CharVocabulary:
    text = "".join(format_prompt(example.prompt) + example.response + "\n" for example in examples)
    return CharVocabulary.from_text(text)


def load_tiny_sft_data(
    *,
    block_size: int = 48,
    batch_size: int = 4,
) -> tuple[CharVocabulary, DataLoader]:
    examples = [InstructionExample(prompt, response) for prompt, response in TINY_INSTRUCTION_DATA]
    vocab = build_sft_vocab(examples)
    dataset = SFTDataset(examples, vocab, block_size=block_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sft_batches,
    )
    return vocab, loader


def collate_sft_batches(batches: list[SFTBatch]) -> SFTBatch:
    max_len = max(batch.input_ids.shape[0] for batch in batches)
    pad_id = 0
    input_ids = torch.full((len(batches), max_len), pad_id, dtype=torch.long)
    target_ids = torch.full((len(batches), max_len), IGNORE_INDEX, dtype=torch.long)
    loss_mask = torch.zeros((len(batches), max_len), dtype=torch.bool)

    for row, batch in enumerate(batches):
        seq_len = batch.input_ids.shape[0]
        input_ids[row, :seq_len] = batch.input_ids
        target_ids[row, :seq_len] = batch.target_ids
        loss_mask[row, :seq_len] = batch.loss_mask
    return SFTBatch(input_ids=input_ids, target_ids=target_ids, loss_mask=loss_mask)


def sft_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("target_ids must have shape [batch, seq_len].")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def format_prompt(prompt: str) -> str:
    return f"Instruction: {prompt}\nResponse: "
