import pytest
import torch
from torch.nn import functional as F

from llm_tutor.data.language_modeling import CharBlockDataset, split_token_ids_for_lm
from llm_tutor.models.gpt import MiniGPT, MiniGPTConfig


def test_char_block_dataset_shifts_targets_by_one_token() -> None:
    dataset = CharBlockDataset([0, 1, 2, 3, 4], block_size=3)

    batch = dataset[0]

    assert batch.input_ids.tolist() == [0, 1, 2]
    assert batch.target_ids.tolist() == [1, 2, 3]


def test_language_modeling_split_uses_gap_between_train_and_val() -> None:
    token_ids = list(range(80))

    _train_ids, _val_ids, train_range, val_range = split_token_ids_for_lm(
        token_ids,
        block_size=8,
        val_fraction=0.2,
    )

    assert train_range[1] <= val_range[0] - 8


def test_mini_gpt_returns_logits_and_loss() -> None:
    torch.manual_seed(42)
    model = MiniGPT(
        MiniGPTConfig(vocab_size=11, block_size=6, embed_dim=16, num_heads=4, num_layers=2)
    )
    input_ids = torch.randint(0, 11, (3, 6))
    target_ids = torch.randint(0, 11, (3, 6))

    output = model(input_ids, target_ids=target_ids, return_attention=True)

    assert output.logits.shape == (3, 6, 11)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.attention_weights is not None
    assert len(output.attention_weights) == 2
    assert output.attention_weights[0].shape == (3, 4, 6, 6)


def test_mini_gpt_loss_matches_manual_cross_entropy() -> None:
    torch.manual_seed(42)
    model = MiniGPT(MiniGPTConfig(vocab_size=11, block_size=6, embed_dim=16, num_heads=4))
    input_ids = torch.randint(0, 11, (3, 6))
    target_ids = torch.randint(0, 11, (3, 6))

    output = model(input_ids, target_ids=target_ids)
    manual_loss = F.cross_entropy(output.logits.reshape(-1, 11), target_ids.reshape(-1))

    assert output.loss is not None
    torch.testing.assert_close(output.loss, manual_loss)


def test_mini_gpt_uses_causal_mask_in_attention_weights() -> None:
    torch.manual_seed(42)
    model = MiniGPT(MiniGPTConfig(vocab_size=8, block_size=5, embed_dim=16, num_heads=4))
    input_ids = torch.randint(0, 8, (2, 5))

    output = model(input_ids, return_attention=True)

    assert output.attention_weights is not None
    future_mask = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)
    assert torch.all(output.attention_weights[0][..., future_mask] == 0)


def test_mini_gpt_future_tokens_do_not_change_past_logits() -> None:
    torch.manual_seed(42)
    model = MiniGPT(MiniGPTConfig(vocab_size=12, block_size=6, embed_dim=16, num_heads=4))
    model.eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 9, 8, 7]])

    first_logits = model(first).logits
    second_logits = model(second).logits

    torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])


def test_mini_gpt_rejects_context_longer_than_block_size() -> None:
    model = MiniGPT(MiniGPTConfig(vocab_size=8, block_size=4, embed_dim=16, num_heads=4))

    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.randint(0, 8, (2, 5)))


def test_mini_gpt_rejects_empty_context() -> None:
    model = MiniGPT(MiniGPTConfig(vocab_size=8, block_size=4, embed_dim=16, num_heads=4))

    with pytest.raises(ValueError, match="at least one token"):
        model(torch.empty((2, 0), dtype=torch.long))


def test_mini_gpt_generate_extends_sequence() -> None:
    torch.manual_seed(42)
    model = MiniGPT(MiniGPTConfig(vocab_size=8, block_size=4, embed_dim=16, num_heads=4))
    input_ids = torch.tensor([[1, 2, 3]])

    generated = model.generate(input_ids, max_new_tokens=5)

    assert generated.shape == (1, 8)
    torch.testing.assert_close(generated[:, :3], input_ids)


def test_mini_gpt_generate_restores_training_mode() -> None:
    torch.manual_seed(42)
    model = MiniGPT(
        MiniGPTConfig(vocab_size=8, block_size=4, embed_dim=16, num_heads=4, dropout=0.5)
    )
    model.train()

    _generated = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=1)

    assert model.training
