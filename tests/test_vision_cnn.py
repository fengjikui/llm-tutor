import torch
from torch.utils.data import Dataset

from llm_tutor.data.vision import load_fashion_mnist_data
from llm_tutor.models.cnn import SmallCNN


class FakeFashionMNIST(Dataset):
    classes = [f"class_{i}" for i in range(10)]

    def __init__(self, root, train: bool, download: bool, transform=None) -> None:
        self.train = train
        self.transform = transform
        size = 60_000 if train else 10_000
        self.targets = torch.arange(size) % 10

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = torch.full((1, 28, 28), float(index % 255) / 255.0)
        label = int(self.targets[index])
        if self.transform is not None:
            # Fake images are already tensors with the expected shape.
            pass
        return image, label


def test_fashion_mnist_subset_shapes() -> None:
    data = load_fashion_mnist_data(
        batch_size=16,
        train_limit=32,
        val_limit=16,
        test_limit=16,
        download=False,
        dataset_cls=FakeFashionMNIST,
    )
    x, y = next(iter(data.train_loader))

    assert data.image_shape == (1, 28, 28)
    assert data.num_classes == 10
    assert x.shape == (16, 1, 28, 28)
    assert y.dtype == torch.long


def test_small_cnn_output_shape() -> None:
    model = SmallCNN(num_classes=10)
    x = torch.randn(4, 1, 28, 28)

    logits = model(x)

    assert logits.shape == (4, 10)
