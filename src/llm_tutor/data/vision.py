from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class VisionClassificationData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    class_names: tuple[str, ...]
    image_shape: tuple[int, int, int]


def load_fashion_mnist_data(
    data_dir: str | Path = "data/vision",
    batch_size: int = 64,
    train_limit: int | None = 2048,
    val_limit: int | None = 512,
    test_limit: int | None = 512,
    seed: int = 42,
    download: bool = True,
    dataset_cls: type[Any] = datasets.FashionMNIST,
) -> VisionClassificationData:
    """加载 Fashion-MNIST，并默认截取小 subset 以便 CPU 快速演示。"""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    root = Path(data_dir)
    train_full = dataset_cls(root=root, train=True, download=download, transform=transform)
    test_full = dataset_cls(root=root, train=False, download=download, transform=transform)

    train_size = 50_000
    val_size = 10_000
    generator = torch.Generator().manual_seed(seed)
    train_pool = torch.randperm(train_size, generator=generator).tolist()
    val_pool = (torch.randperm(val_size, generator=generator) + train_size).tolist()
    train_indices = train_pool
    val_indices = val_pool

    if train_limit is not None:
        train_indices = train_indices[:train_limit]
    if val_limit is not None:
        val_indices = val_indices[:val_limit]

    test_indices = torch.randperm(len(test_full), generator=generator).tolist()
    if test_limit is not None:
        test_indices = test_indices[:test_limit]

    def make_loader(dataset, indices: list[int], shuffle: bool) -> DataLoader:
        loader_generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=loader_generator,
        )

    return VisionClassificationData(
        train_loader=make_loader(train_full, train_indices, shuffle=True),
        val_loader=make_loader(train_full, val_indices, shuffle=False),
        test_loader=make_loader(test_full, test_indices, shuffle=False),
        num_classes=10,
        class_names=tuple(train_full.classes),
        image_shape=(1, 28, 28),
    )
