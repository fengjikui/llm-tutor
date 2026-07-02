import torch

from llm_tutor.data.names import load_name_classification_data
from llm_tutor.models.rnn import CharRNNClassifier


def test_name_dataset_batch_shapes() -> None:
    data = load_name_classification_data(batch_size=8)
    x, y = next(iter(data.train_loader))

    assert data.vocab_size == 27
    assert data.num_classes == 4
    assert x.ndim == 2
    assert x.shape[0] == 8
    assert y.shape == (8,)
    assert y.dtype == torch.long


def test_char_rnn_classifier_output_shape() -> None:
    data = load_name_classification_data(batch_size=4)
    x, _ = next(iter(data.train_loader))
    model = CharRNNClassifier(
        vocab_size=data.vocab_size,
        num_classes=data.num_classes,
        pad_id=data.pad_id,
    )

    logits = model(x)

    assert logits.shape == (4, data.num_classes)


def test_char_rnn_ignores_trailing_padding_for_classification() -> None:
    model = CharRNNClassifier(vocab_size=27, num_classes=4, pad_id=0)
    model.eval()
    short = torch.tensor([[1, 2, 3]], dtype=torch.long)
    padded = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        short_logits = model(short)
        padded_logits = model(padded)

    torch.testing.assert_close(short_logits, padded_logits)
