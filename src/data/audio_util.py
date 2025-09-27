import torch
import torch.nn.functional as F
import torchaudio
from typing import Dict, List, Optional, Tuple
from datetime import timedelta


def mono_resample_audio(audio: torch.Tensor, old_sr: int, final_sr: int):
    audio = audio.mean(dim=0, keepdim=True)
    resampler = torchaudio.transforms.Resample(orig_freq=old_sr, new_freq=final_sr)
    return resampler(audio)


def mix_audio_values(audio_values: Dict, channels: List[str]):
    mixed_audio = sum([audio_values[channel] for channel in channels])
    return torch.clip(mixed_audio, -1.0, 1.0)


def trim_wav_file(
    file_path: str,
    start: Optional[timedelta] = None,
    end: Optional[timedelta] = None,
    out_file_path: Optional[str] = None,
) -> None:
    waveform, sample_rate = torchaudio.load(file_path)
    if start is not None:
        start = int(start.total_seconds() * sample_rate)
    if end is not None:
        end = int(end.total_seconds() * sample_rate)
    waveform = waveform[:, start:end]
    if out_file_path is None:
        out_file_path = file_path
    torchaudio.save(out_file_path, waveform, sample_rate)


def get_audio_length(file_path: str) -> float:
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform.shape[-1] / sample_rate


def extract_windows_above_threshold(
    audio_file_path: str,
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
                        round((window_start + 1) * rms_stride_seconds, ndigits=2),
                        round(window_end * rms_stride_seconds, ndigits=2),
                    )
                )
            window_start = window_end
        window_end += 1

    return windows
