import pytest
import torch

from llm_tutor.data.translation import TranslationDataset, load_toy_translation_data
from llm_tutor.models.seq2seq import AttentionSeq2SeqGRU, Seq2SeqGRU


def test_translation_batch_shapes() -> None:
    data = load_toy_translation_data(batch_size=32)
    batch = next(iter(data.train_loader))

    assert batch.source.ndim == 2
    assert batch.target_input.ndim == 2
    assert batch.target_output.ndim == 2
    assert batch.target_input.shape == batch.target_output.shape
    assert batch.target_input[0, 0].item() == data.target_vocab.bos_id
    assert (batch.target_output == data.target_vocab.eos_id).any()
    assert (batch.source == data.source_vocab.pad_id).any()


def test_seq2seq_output_shape() -> None:
    data = load_toy_translation_data(batch_size=4)
    batch = next(iter(data.train_loader))
    model = Seq2SeqGRU(
        source_vocab_size=len(data.source_vocab.id_to_token),
        target_vocab_size=len(data.target_vocab.id_to_token),
        source_pad_id=data.source_vocab.pad_id,
        target_pad_id=data.target_vocab.pad_id,
    )

    logits = model(batch.source, batch.target_input)

    assert logits.shape == (
        batch.target_input.shape[0],
        batch.target_input.shape[1],
        len(data.target_vocab.id_to_token),
    )
    assert torch.isfinite(logits).all()


def test_attention_seq2seq_returns_attention_weights() -> None:
    data = load_toy_translation_data(batch_size=4)
    batch = next(iter(data.train_loader))
    model = AttentionSeq2SeqGRU(
        source_vocab_size=len(data.source_vocab.id_to_token),
        target_vocab_size=len(data.target_vocab.id_to_token),
        source_pad_id=data.source_vocab.pad_id,
        target_pad_id=data.target_vocab.pad_id,
    )

    logits, attention = model(batch.source, batch.target_input, return_attention=True)

    assert logits.shape[:2] == batch.target_input.shape
    assert attention.shape == (
        batch.source.shape[0],
        batch.target_input.shape[1],
        batch.source.shape[1],
    )
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)))


def test_attention_seq2seq_masks_trailing_source_padding() -> None:
    data = load_toy_translation_data(batch_size=4)
    model = AttentionSeq2SeqGRU(
        source_vocab_size=len(data.source_vocab.id_to_token),
        target_vocab_size=len(data.target_vocab.id_to_token),
        source_pad_id=data.source_vocab.pad_id,
        target_pad_id=data.target_vocab.pad_id,
    )
    model.eval()
    source = torch.tensor(
        [
            [
                data.source_vocab.token_to_id["i"],
                data.source_vocab.token_to_id["am"],
                data.source_vocab.eos_id,
            ]
        ]
    )
    padded_source = torch.tensor(
        [
            [
                data.source_vocab.token_to_id["i"],
                data.source_vocab.token_to_id["am"],
                data.source_vocab.eos_id,
                data.source_vocab.pad_id,
                data.source_vocab.pad_id,
            ]
        ]
    )
    target_input = torch.tensor([[data.target_vocab.bos_id, data.target_vocab.token_to_id["je"]]])

    with torch.no_grad():
        logits = model(source, target_input)
        padded_logits, attention = model(padded_source, target_input, return_attention=True)

    torch.testing.assert_close(logits, padded_logits)
    torch.testing.assert_close(attention[..., -2:], torch.zeros_like(attention[..., -2:]))


def test_attention_rejects_all_pad_source() -> None:
    data = load_toy_translation_data(batch_size=4)
    model = AttentionSeq2SeqGRU(
        source_vocab_size=len(data.source_vocab.id_to_token),
        target_vocab_size=len(data.target_vocab.id_to_token),
        source_pad_id=data.source_vocab.pad_id,
        target_pad_id=data.target_vocab.pad_id,
    )
    source = torch.zeros((1, 3), dtype=torch.long)
    target_input = torch.tensor([[data.target_vocab.bos_id, data.target_vocab.token_to_id["je"]]])

    with pytest.raises(ValueError, match="non-pad"):
        model(source, target_input)


def test_seq2seq_encoder_ignores_trailing_source_padding() -> None:
    data = load_toy_translation_data(batch_size=4)
    model = Seq2SeqGRU(
        source_vocab_size=len(data.source_vocab.id_to_token),
        target_vocab_size=len(data.target_vocab.id_to_token),
        source_pad_id=data.source_vocab.pad_id,
        target_pad_id=data.target_vocab.pad_id,
    )
    model.eval()
    source = torch.tensor(
        [
            [
                data.source_vocab.token_to_id["i"],
                data.source_vocab.token_to_id["am"],
                data.source_vocab.eos_id,
            ]
        ]
    )
    padded_source = torch.tensor(
        [
            [
                data.source_vocab.token_to_id["i"],
                data.source_vocab.token_to_id["am"],
                data.source_vocab.eos_id,
                data.source_vocab.pad_id,
                data.source_vocab.pad_id,
            ]
        ]
    )
    target_input = torch.tensor([[data.target_vocab.bos_id, data.target_vocab.token_to_id["je"]]])

    with torch.no_grad():
        logits = model(source, target_input)
        padded_logits = model(padded_source, target_input)

    torch.testing.assert_close(logits, padded_logits)


def test_validation_and_test_pairs_do_not_encode_as_unknown() -> None:
    data = load_toy_translation_data(batch_size=8)

    for loader in (data.val_loader, data.test_loader):
        dataset = loader.dataset
        assert isinstance(dataset, TranslationDataset)
        for source, target in dataset.pairs:
            assert data.source_vocab.unk_id not in data.source_vocab.encode(source)
            assert data.target_vocab.unk_id not in data.target_vocab.encode(target)
