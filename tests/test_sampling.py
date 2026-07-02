import pytest
import torch

from llm_tutor.data.language_modeling import CharVocabulary
from llm_tutor.experiments.train_mini_gpt import save_checkpoint
from llm_tutor.generation.sampling import filter_logits, sample_next_token
from llm_tutor.models.gpt import MiniGPT, MiniGPTConfig


def test_top_k_filter_keeps_only_best_k_tokens() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    filtered = filter_logits(logits, top_k=2)

    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
    assert torch.isfinite(filtered[0, 2])
    assert torch.isfinite(filtered[0, 3])


def test_top_p_filter_keeps_at_least_one_token() -> None:
    logits = torch.tensor([[10.0, 0.0, 0.0]])

    filtered = filter_logits(logits, top_p=0.1)

    assert torch.isfinite(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
    assert torch.isneginf(filtered[0, 2])


def test_top_p_filter_keeps_token_that_crosses_threshold() -> None:
    logits = torch.log(torch.tensor([[0.50, 0.30, 0.15, 0.05]]))

    filtered = filter_logits(logits, top_p=0.7)

    assert torch.isfinite(filtered[0, 0])
    assert torch.isfinite(filtered[0, 1])
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 3])


def test_top_k_larger_than_vocab_is_allowed() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])

    filtered = filter_logits(logits, top_k=10)

    assert torch.isfinite(filtered).all()


def test_sampling_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        filter_logits(torch.randn(1, 4), top_k=0)


def test_sampling_rejects_invalid_top_p() -> None:
    with pytest.raises(ValueError, match="top_p"):
        filter_logits(torch.randn(1, 4), top_p=0)
    with pytest.raises(ValueError, match="top_p"):
        filter_logits(torch.randn(1, 4), top_p=1.1)


def test_sample_next_token_with_top_k_one_returns_argmax() -> None:
    generator = torch.Generator().manual_seed(42)
    logits = torch.tensor([[0.0, 1.0, 5.0, 2.0]])

    token = sample_next_token(logits, top_k=1, generator=generator)

    assert token.tolist() == [[2]]


def test_sampling_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(torch.randn(1, 4), temperature=0)


def test_mini_gpt_generate_accepts_sampling_options() -> None:
    torch.manual_seed(42)
    model = MiniGPT(MiniGPTConfig(vocab_size=8, block_size=4, embed_dim=16, num_heads=4))
    generated = model.generate(
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=2,
        temperature=0.8,
        top_k=3,
        top_p=0.9,
        generator=torch.Generator().manual_seed(42),
    )

    assert generated.shape == (1, 5)


def test_checkpoint_roundtrip_contains_reproducibility_metadata(tmp_path) -> None:
    torch.manual_seed(42)
    model = MiniGPT(MiniGPTConfig(vocab_size=8, block_size=4, embed_dim=16, num_heads=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    vocab = CharVocabulary(
        id_to_token=tuple("abcdefgh"),
        token_to_id={token: idx for idx, token in enumerate("abcdefgh")},
    )
    checkpoint_path = tmp_path / "mini_gpt.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        vocab=vocab,
        epoch=3,
        train_loss=1.2,
        val_loss=1.5,
        training_args={"seed": 42, "lr": 1e-3, "batch_size": 2},
        data_meta={"dataset": "test", "train_token_range": (0, 10), "val_token_range": (14, 20)},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded_model = MiniGPT(MiniGPTConfig(**checkpoint["config"]))
    loaded_model.load_state_dict(checkpoint["model_state"])

    generated = loaded_model.generate(
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=1,
        generator=torch.Generator().manual_seed(42),
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["val_loss"] == 1.5
    assert checkpoint["training_args"]["seed"] == 42
    assert checkpoint["data_meta"]["dataset"] == "test"
    assert "optimizer_state" in checkpoint
    assert checkpoint["vocab_id_to_token"] == list("abcdefgh")
    assert generated.shape == (1, 4)
