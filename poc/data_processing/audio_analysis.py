import torch
import torch.nn.functional as F
import torchaudio
from typing import Tuple, List


def extract_longest_windows_above_threshold(
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
