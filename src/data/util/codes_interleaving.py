import torch
from torch import Tensor
import torch.nn.functional as F


def add_delay_interleaving(
    codes: Tensor, padding_idx: int = 2048
) -> Tensor:
    """[num_codebooks, seq_len] -> [num_codebooks, seq_len + num_codebooks + 1]"""
    num_codebooks = len(codes)
    new_codes = []
    for index, stream in enumerate(codes):
        new_codes.append(
            F.pad(stream, (index + 1, num_codebooks - index), value=padding_idx)
        )
    return torch.stack(new_codes)


def remove_delay_interleaving(codes: Tensor) -> Tensor:
    """[num_codebooks, seq_len] -> [num_codebooks, seq_len - num_codebooks - 1]"""
    num_codebooks = codes.shape[-2]
    stream_length = codes.shape[-1]
    new_codes = []
    for index, stream in enumerate(codes):
        new_codes.append(
            torch.narrow(stream, -1, 1 + index, stream_length - (num_codebooks - 1) - 2)
        )
    return torch.stack(new_codes)