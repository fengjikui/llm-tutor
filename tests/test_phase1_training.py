import pytest
import torch

from llm_tutor.data.tabular import load_breast_cancer_data
from llm_tutor.models.feedforward import LinearClassifier
from llm_tutor.training.loop import train_classifier


def test_breast_cancer_data_shapes_and_names() -> None:
    data = load_breast_cancer_data(batch_size=32)
    x, y = next(iter(data.train_loader))

    assert data.num_features == 30
    assert data.class_names == ("malignant", "benign")
    assert x.shape == (32, 30)
    assert y.dtype == torch.long


def test_short_training_loop_returns_history() -> None:
    torch.manual_seed(42)
    data = load_breast_cancer_data(batch_size=128)
    model = LinearClassifier(num_features=data.num_features)

    history = train_classifier(
        model,
        data.train_loader,
        data.val_loader,
        epochs=1,
        lr=1e-2,
        device=torch.device("cpu"),
        positive_label=0,
    )

    assert len(history) == 1
    assert 0.0 <= history[0]["val_accuracy"] <= 1.0
    assert 0.0 <= history[0]["val_f1"] <= 1.0


def test_train_classifier_rejects_invalid_epochs() -> None:
    data = load_breast_cancer_data(batch_size=128)
    model = LinearClassifier(num_features=data.num_features)

    with pytest.raises(ValueError, match="epochs"):
        train_classifier(
            model,
            data.train_loader,
            data.val_loader,
            epochs=0,
            lr=1e-2,
            device=torch.device("cpu"),
        )
