from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

_TRAIN_PAIRS = [
    ("i am cold", "je suis froid"),
    ("i am tired", "je suis fatigue"),
    ("i am happy", "je suis heureux"),
    ("i am sad", "je suis triste"),
    ("i am here", "je suis ici"),
    ("you are cold", "tu es froid"),
    ("you are kind", "tu es gentil"),
    ("you are smart", "tu es intelligent"),
    ("you are funny", "tu es drole"),
    ("you are here", "tu es ici"),
    ("he is tall", "il est grand"),
    ("he is calm", "il est calme"),
    ("he is here", "il est ici"),
    ("she is kind", "elle est gentille"),
    ("she is smart", "elle est intelligente"),
    ("she is here", "elle est ici"),
    ("we are ready", "nous sommes prets"),
    ("we are here", "nous sommes ici"),
    ("they are ready", "ils sont prets"),
    ("they are here", "ils sont ici"),
    ("it is small", "c est petit"),
    ("it is red", "c est rouge"),
    ("it is here", "c est ici"),
    ("this is good", "ceci est bon"),
    ("this is bad", "ceci est mauvais"),
    ("that is new", "cela est nouveau"),
    ("that is good", "cela est bon"),
    ("we are very ready", "nous sommes tres prets"),
    ("they are very funny", "ils sont tres drole"),
]

_VAL_PAIRS = [
    ("you are tired", "tu es fatigue"),
    ("she is calm", "elle est calme"),
    ("this is red", "ceci est rouge"),
]

_TEST_PAIRS = [
    ("i am very happy", "je suis tres heureux"),
    ("he is smart", "il est intelligent"),
    ("that is bad", "cela est mauvais"),
]


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in text.split()]
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens: list[str] = []
        for token_id in ids:
            token = self.id_to_token[token_id]
            if token in {PAD, BOS}:
                continue
            if token == EOS:
                break
            tokens.append(token)
        return " ".join(tokens)


@dataclass(frozen=True)
class TranslationBatch:
    source: torch.Tensor
    target_input: torch.Tensor
    target_output: torch.Tensor


@dataclass(frozen=True)
class TranslationData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    source_vocab: Vocabulary
    target_vocab: Vocabulary


class TranslationDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
    ) -> None:
        self.pairs = pairs
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, target = self.pairs[index]
        source_ids = self.source_vocab.encode(source, add_bos=False, add_eos=True)
        target_input_ids = self.target_vocab.encode(target, add_bos=True, add_eos=False)
        target_output_ids = self.target_vocab.encode(target, add_bos=False, add_eos=True)
        return (
            torch.tensor(source_ids, dtype=torch.long),
            torch.tensor(target_input_ids, dtype=torch.long),
            torch.tensor(target_output_ids, dtype=torch.long),
        )


def load_toy_translation_data(batch_size: int = 8, seed: int = 42) -> TranslationData:
    source_vocab = _build_vocab(source for source, _target in _TRAIN_PAIRS)
    target_vocab = _build_vocab(target for _source, target in _TRAIN_PAIRS)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(_TRAIN_PAIRS), generator=generator).tolist()
    train_pairs = [_TRAIN_PAIRS[index] for index in permutation]

    def make_loader(split: list[tuple[str, str]], shuffle: bool) -> DataLoader:
        loader_generator = torch.Generator().manual_seed(seed)
        dataset = TranslationDataset(split, source_vocab, target_vocab)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=loader_generator,
            collate_fn=lambda batch: _collate_translation(
                batch,
                source_pad_id=source_vocab.pad_id,
                target_pad_id=target_vocab.pad_id,
            ),
        )

    return TranslationData(
        train_loader=make_loader(train_pairs, shuffle=True),
        val_loader=make_loader(_VAL_PAIRS, shuffle=False),
        test_loader=make_loader(_TEST_PAIRS, shuffle=False),
        source_vocab=source_vocab,
        target_vocab=target_vocab,
    )


def _build_vocab(texts) -> Vocabulary:
    tokens = sorted({token for text in texts for token in text.split()})
    id_to_token = (PAD, BOS, EOS, UNK, *tokens)
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return Vocabulary(token_to_id=token_to_id, id_to_token=id_to_token)


def _collate_translation(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    source_pad_id: int,
    target_pad_id: int,
) -> TranslationBatch:
    sources, target_inputs, target_outputs = zip(*batch, strict=True)
    return TranslationBatch(
        source=pad_sequence(sources, batch_first=True, padding_value=source_pad_id),
        target_input=pad_sequence(target_inputs, batch_first=True, padding_value=target_pad_id),
        target_output=pad_sequence(target_outputs, batch_first=True, padding_value=target_pad_id),
    )
