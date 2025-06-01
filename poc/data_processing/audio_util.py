import torch
import torchaudio
from typing import Dict, List, Optional
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
