import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from datetime import timedelta
from pathlib import Path
import pyloudnorm


def convert_mono(audio: torch.Tensor):
    return audio.mean(dim=0, keepdim=True)


def resample(
    audio: torch.Tensor,
    old_sr: int,
    final_sr: int,
    device = torch.device("cpu"),
):
    resampler = torchaudio.transforms.Resample(orig_freq=old_sr, new_freq=final_sr).to(
        device
    )
    return resampler(audio)


def normalize_lufs(
    waveform: torch.Tensor, sr: int, target_lufs: float = -14.0
) -> torch.Tensor:
    """
    Normalizes a torchaudio waveform to a target LUFS level.

    Args:
        waveform (torch.Tensor): The input waveform tensor.
        sample_rate (int): The sample rate of the audio.
        target_lufs (float): The desired LUFS level.

    Returns:
        torch.Tensor: The loudness-normalized waveform tensor.
    """
    assert waveform.dim() == 2
    waveform_np = waveform.transpose(0, 1).numpy()

    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(waveform_np)
    normalized_waveform_np = pyloudnorm.normalize.loudness(
        waveform_np, loudness, target_lufs
    )
    normalized_waveform = torch.from_numpy(normalized_waveform_np).transpose(0, 1)

    return normalized_waveform


def mix_audio_values(audio_values: Dict, channels: List[str]):
    mixed_audio = sum([audio_values[channel] for channel in channels])
    return torch.clip(mixed_audio, -1.0, 1.0)


def trim_audio(
    file_path: Optional[Path] = None,
    waveform: Optional[torch.Tensor] = None,
    start: Optional[timedelta] = None,
    end: Optional[timedelta] = None,
    out_file_path: Optional[Path] = None,
    sample_rate: Optional[int] = None,
) -> Optional[Tensor]:
    assert (file_path is None) + (waveform is None) == 1, (
        "Provide either file path or Tensor"
    )

    if file_path is None:
        assert sample_rate is not None, (
            "Provide file path for SR or provide SR explicitly"
        )

    if file_path is not None:
        waveform, sample_rate = torchaudio.load(file_path)

    if start is not None:
        start = int(start.total_seconds() * sample_rate)
    if end is not None:
        end = int(end.total_seconds() * sample_rate)
    waveform = waveform[:, start:end]
    if out_file_path is None:
        return waveform
    torchaudio.save(out_file_path, waveform, sample_rate)


def get_audio_length(file_path: str) -> float:
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform.shape[-1] / sample_rate


def extract_windows_above_threshold(
    audio_file_path: Path,
    rms_threshold: float = 0.0175,
    min_window_size_seconds: float = 30,
    rms_frame_length_seconds: float = 10,
    rms_stride_seconds: float = 0.2,
) -> List[Tuple[float, float]]:
    waveform, sample_rate = torchaudio.load(audio_file_path)
    rms_frame_length = int(rms_frame_length_seconds * sample_rate)
    rms_stride = int(rms_stride_seconds * sample_rate)
    min_window_size = int(min_window_size_seconds / rms_stride_seconds)
    padding = rms_frame_length - rms_stride

    waveform = F.pad(waveform, (padding // 2, padding - padding // 2))
    sliding_window_rms = torch.sqrt(
        F.avg_pool1d(waveform**2, kernel_size=rms_frame_length, stride=rms_stride)
    ).squeeze()

    window_start = 0
    window_end = 1
    windows = []
    while window_end < len(sliding_window_rms):
        if sliding_window_rms[window_end] < rms_threshold:
            if window_end - window_start > min_window_size:
                windows.append(
                    (
                        round(window_start * rms_stride_seconds, ndigits=2),
                        round(window_end * rms_stride_seconds, ndigits=2),
                    )
                )
            window_start = window_end + 1
        window_end += 1
    if window_end - window_start > min_window_size:
        windows.append(
            (
                round(window_start * rms_stride_seconds, ndigits=2),
                round(len(sliding_window_rms) * rms_stride_seconds, ndigits=2),
            )
        )

    return windows
