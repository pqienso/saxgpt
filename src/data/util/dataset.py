import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from typing import List, Tuple
import random

from .codes_interleaving import add_delay_interleaving


EXAMPLE_SIZE = 30 * 50  # 30s: 50 tokens = 1s


def train_test_split(
    codes: List[Tuple[Tensor, Tensor]], test: float, val: float
) -> Tuple[
    List[Tuple[Tensor, Tensor]],
    List[Tuple[Tensor, Tensor]],
    List[Tuple[Tensor, Tensor]],
]:
    assert test + val < 1

    total_len = sum([example[0].shape[-1] for example in codes])
    random.shuffle(codes)
    i = 0

    target_test_len = int(test * total_len)
    test_len = 0
    test_dataset = []
    while test_len < target_test_len:
        test_dataset.append(codes[i])
        test_len += codes[i][0].shape[-1]
        i += 1

    target_val_len = int(val * total_len)
    val_len = 0
    val_dataset = []
    while val_len < target_val_len:
        val_dataset.append(codes[i])
        val_len += codes[i][0].shape[-1]
        i += 1

    train_dataset = []
    while i < len(codes):
        train_dataset.append(codes[i])
        i += 1

    return train_dataset, val_dataset, test_dataset


def get_tensor_dataset(
    data: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device = torch.device("cpu"),
    seq_len: int = EXAMPLE_SIZE,
    stride: int = EXAMPLE_SIZE // 2,
    padding_idx: int = 2048,
) -> TensorDataset:
        src = []
        tgt = []
        for backing, lead in data:
            if backing.shape[-1] < seq_len:
                continue
            src.append(backing.unfold(-1, seq_len, stride).transpose(0, 1))
            tgt.append(lead.unfold(-1, seq_len, stride).transpose(0, 1))
        src = torch.concat(src)
        tgt = torch.concat(tgt)
        src = torch.vmap(lambda s: add_delay_interleaving(s, padding_idx))(src)
        src = src.to(device)
        tgt = torch.vmap(lambda t: add_delay_interleaving(t, padding_idx))(tgt)
        tgt = tgt.to(device)
        return TensorDataset(src, tgt)
