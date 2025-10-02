import torch
from torch import Tensor
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from typing import List, Tuple
import random

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


def add_delay_interleaving(
    codes: Tensor, padding_idx: int = 2048
) -> Tensor:
    """[num_codebooks, seq_len] -> [num_codebooks, seq_len]"""
    num_codebooks = len(codes)
    new_codes = []
    for index, stream in enumerate(codes):
        new_codes.append(
            F.pad(stream, (index + 1, num_codebooks - index), value=padding_idx)
        )
    return torch.stack(new_codes)


def remove_delay_interleaving(codes: Tensor) -> Tensor:
    """[num_codebooks, seq_len] -> [num_codebooks, seq_len]"""
    num_codebooks = codes.shape[-2]
    stream_length = codes.shape[-1]
    new_codes = []
    for index, stream in enumerate(codes):
        new_codes.append(
            torch.narrow(stream, -1, 1 + index, stream_length - (num_codebooks - 1) - 2)
        )
    return torch.stack(new_codes)


class SequenceDataset(TensorDataset):
    def __init__(
        self,
        data: List[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device = torch.device("cpu"),
        seq_len: int = EXAMPLE_SIZE,
        stride: int = EXAMPLE_SIZE // 2,
        padding_idx: int = 2048,
    ):
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
        return super().__init__(src, tgt)
