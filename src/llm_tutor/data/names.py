from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

_RAW_NAMES: dict[str, list[str]] = {
    "English": [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Taylor",
        "Miller",
        "Wilson",
        "Moore",
        "Jackson",
        "Clark",
        "Lewis",
        "Walker",
        "Hall",
        "Allen",
        "Young",
        "King",
    ],
    "French": [
        "Martin",
        "Bernard",
        "Dubois",
        "Thomas",
        "Robert",
        "Richard",
        "Petit",
        "Durand",
        "Leroy",
        "Moreau",
        "Simon",
        "Laurent",
        "Lefevre",
        "Michel",
        "Garcia",
        "Roux",
    ],
    "Italian": [
        "Rossi",
        "Russo",
        "Ferrari",
        "Esposito",
        "Bianchi",
        "Romano",
        "Colombo",
        "Ricci",
        "Marino",
        "Greco",
        "Bruno",
        "Gallo",
        "Conti",
        "Mancini",
        "Costa",
        "Giordano",
    ],
    "Spanish": [
        "Garcia",
        "Rodriguez",
        "Gonzalez",
        "Fernandez",
        "Lopez",
        "Martinez",
        "Sanchez",
        "Perez",
        "Gomez",
        "Martin",
        "Jimenez",
        "Ruiz",
        "Hernandez",
        "Diaz",
        "Moreno",
        "Alvarez",
    ],
}


@dataclass(frozen=True)
class NameClassificationData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    vocab_size: int
    num_classes: int
    class_names: tuple[str, ...]
    pad_id: int


class NameDataset(Dataset):
    def __init__(
        self,
        examples: list[tuple[str, int]],
        char_to_id: dict[str, int],
    ) -> None:
        self.examples = examples
        self.char_to_id = char_to_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name, label = self.examples[index]
        ids = [self.char_to_id[char] for char in name]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def load_name_classification_data(
    batch_size: int = 16,
    seed: int = 42,
) -> NameClassificationData:
    """加载一个小型字符级姓名分类数据集。

    数据内置在代码中，目的是让 RNN 章节无需网络也能稳定运行。它借鉴了
    PyTorch 官方 RNN 教程的“名字 -> 语言”任务形式，但样本量更小，只用于
    教学和 smoke test，不用于证明真实姓名来源识别能力。
    """

    class_names = tuple(_RAW_NAMES.keys())
    examples_by_label: dict[int, list[tuple[str, int]]] = {}
    for label, class_name in enumerate(class_names):
        examples_by_label[label] = []
        for name in _RAW_NAMES[class_name]:
            examples_by_label[label].append((_normalize_name(name), label))

    alphabet = string.ascii_lowercase
    char_to_id = {char: i + 1 for i, char in enumerate(alphabet)}
    pad_id = 0

    generator = torch.Generator().manual_seed(seed)
    train_examples: list[tuple[str, int]] = []
    val_examples: list[tuple[str, int]] = []
    test_examples: list[tuple[str, int]] = []
    for label_examples in examples_by_label.values():
        permutation = torch.randperm(len(label_examples), generator=generator).tolist()
        shuffled = [label_examples[i] for i in permutation]
        train_examples.extend(shuffled[:10])
        val_examples.extend(shuffled[10:13])
        test_examples.extend(shuffled[13:])

    train_examples = _shuffle_examples(train_examples, generator)
    val_examples = _shuffle_examples(val_examples, generator)
    test_examples = _shuffle_examples(test_examples, generator)

    def make_loader(split: list[tuple[str, int]], shuffle: bool) -> DataLoader:
        loader_generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            NameDataset(split, char_to_id),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=loader_generator,
            collate_fn=lambda batch: _collate_names(batch, pad_id=pad_id),
        )

    return NameClassificationData(
        train_loader=make_loader(train_examples, shuffle=True),
        val_loader=make_loader(val_examples, shuffle=False),
        test_loader=make_loader(test_examples, shuffle=False),
        vocab_size=len(char_to_id) + 1,
        num_classes=len(class_names),
        class_names=class_names,
        pad_id=pad_id,
    )


def _collate_names(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch, strict=True)
    padded = pad_sequence(sequences, batch_first=True, padding_value=pad_id)
    return padded, torch.stack(labels)


def _normalize_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("utf-8")
    return "".join(char for char in ascii_name.lower() if char in string.ascii_lowercase)


def _shuffle_examples(
    examples: list[tuple[str, int]],
    generator: torch.Generator,
) -> list[tuple[str, int]]:
    permutation = torch.randperm(len(examples), generator=generator).tolist()
    return [examples[i] for i in permutation]
