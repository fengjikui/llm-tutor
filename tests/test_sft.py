import torch
from torch.nn import functional as F

from llm_tutor.post_training.sft import (
    IGNORE_INDEX,
    InstructionExample,
    SFTDataset,
    build_sft_vocab,
    collate_sft_batches,
    format_prompt,
    sft_cross_entropy,
)


def test_sft_dataset_masks_prompt_labels() -> None:
    example = InstructionExample("repeat: hi", "hi")
    vocab = build_sft_vocab([example])
    dataset = SFTDataset([example], vocab, block_size=48)

    batch = dataset[0]
    prompt_len = len(vocab.encode(format_prompt(example.prompt)))

    assert all(target == IGNORE_INDEX for target in batch.target_ids[: prompt_len - 1].tolist())
    assert batch.loss_mask[: prompt_len - 1].sum().item() == 0
    assert batch.loss_mask[prompt_len - 1 :].all()
    assert vocab.decode(batch.target_ids[batch.loss_mask]) == "hi\n"


def test_sft_collate_pads_targets_with_ignore_index() -> None:
    examples = [InstructionExample("repeat: hi", "hi"), InstructionExample("answer no", "no")]
    vocab = build_sft_vocab(examples)
    dataset = SFTDataset(examples, vocab, block_size=48)

    batch = collate_sft_batches([dataset[0], dataset[1]])

    assert batch.input_ids.ndim == 2
    assert batch.target_ids.shape == batch.input_ids.shape
    assert ((batch.target_ids == IGNORE_INDEX) | batch.loss_mask).all()
    torch.testing.assert_close(batch.loss_mask, batch.target_ids.ne(IGNORE_INDEX))


def test_sft_cross_entropy_matches_ignore_index_loss() -> None:
    torch.manual_seed(42)
    logits = torch.randn(2, 3, 5)
    target_ids = torch.tensor([[1, IGNORE_INDEX, 2], [3, 4, IGNORE_INDEX]])

    loss = sft_cross_entropy(logits, target_ids)
    expected = F.cross_entropy(logits.reshape(-1, 5), target_ids.reshape(-1), ignore_index=-100)

    torch.testing.assert_close(loss, expected)
