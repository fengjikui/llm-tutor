from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TabularClassificationData:
    """一个已经切好 train/val/test 的二分类表格数据集。"""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_features: int
    class_names: tuple[str, str]
    feature_names: tuple[str, ...]


def load_breast_cancer_data(batch_size: int = 64, seed: int = 42) -> TabularClassificationData:
    """加载 Breast Cancer Wisconsin 数据集，作为第一章的稳定离线实验。

    这个数据集不是为了追求业务新鲜度，而是为了让读者在没有网络、没有 GPU 的
    情况下也能跑通完整训练闭环：读取数据、标准化、划分数据集、训练分类器。
    后续章节会逐步换成图像、文本和生成任务。
    """

    raw = load_breast_cancer()
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        raw.data,
        raw.target,
        test_size=0.2,
        random_state=seed,
        stratify=raw.target,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.2,
        random_state=seed,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    def make_loader(features, labels, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

    return TabularClassificationData(
        train_loader=make_loader(x_train, y_train, shuffle=True),
        val_loader=make_loader(x_val, y_val, shuffle=False),
        test_loader=make_loader(x_test, y_test, shuffle=False),
        num_features=raw.data.shape[1],
        class_names=tuple(raw.target_names),
        feature_names=tuple(raw.feature_names),
    )
