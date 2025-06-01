import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as T
import torchaudio
import os
from typing import Optional, Union, Tuple, List, Set

from data_processing.dataset_ingestion import extract_url_title


class AudioExample:
    def __init__(
        self,
        video_id: str,
        lead_audio: Union[str, torch.Tensor],
        backing_audio: Union[str, torch.Tensor],
        sample_rate: Optional[int] = None,
        title: Optional[str] = None,
        clip_timestamps: Optional[Tuple[float, float]] = None,
        pitch_aug: int = 0,
        tempo_aug: float = 1.0,
    ):
        if isinstance(lead_audio, str):
            lead_audio, sample_rate = torchaudio.load(lead_audio)
        self.lead = lead_audio

        if isinstance(backing_audio, str):
            backing_audio, sample_rate = torchaudio.load(backing_audio)
        self.backing = backing_audio

        if sample_rate is None:
            raise ValueError("Must specify sample rate if waveform Tensors are passed.")
        self.sr = sample_rate

        if title is None:
            title = extract_url_title(f"https://www.youtube.com/watch?v={video_id}")
        self.title = title

        self.video_id = video_id
        self.pitch_aug = pitch_aug
        self.tempo_aug = tempo_aug
        if clip_timestamps is None:
            clip_timestamps = (0, lead_audio.shape[-1] / sample_rate)
        self.clip_timestamps = clip_timestamps

    def write_to_wav_file(self, dest: str, file_name: Optional[str] = None):
        if file_name is None:
            file_name = (
                f"{self.video_id}_c{int(self.clip_timestamps[0])}"
                + f"c{int(self.clip_timestamps[1])}_p{self.pitch_aug}"
                + f"_t{str(int(self.tempo_aug * 10)).zfill(2)}"
            )
        torchaudio.save(os.path.join(dest, f"lead_{file_name}.wav"), self.lead, self.sr)
        torchaudio.save(
            os.path.join(dest, f"backing_{file_name}.wav"), self.backing, self.sr
        )


class ValidWindowClipper(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 32000,
        rms_threshold: float = 0.0175,
        min_window_size_seconds: float = 30,
        rms_frame_length_seconds: float = 10,
        rms_stride_seconds: float = 0.2,
    ):
        super().__init__()
        self.rms_threshold = rms_threshold
        self.min_window_size = min_window_size_seconds
        self.rms_frame_length = rms_frame_length_seconds
        self.rms_stride = rms_stride_seconds
        self.sample_rate = sample_rate

    def get_valid_windows(self, waveform: torch.Tensor) -> List[Tuple[float, float]]:
        rms_frame_length = int(self.rms_frame_length * self.sample_rate)
        rms_stride = int(self.rms_stride * self.sample_rate)
        min_window_size = int(self.min_window_size / self.rms_stride)
        padding = rms_frame_length - rms_stride

        waveform = F.pad(waveform, (padding // 2, padding - padding // 2))
        sliding_window_rms = torch.sqrt(
            F.avg_pool1d(waveform**2, kernel_size=rms_frame_length, stride=rms_stride)
        ).squeeze()

        window_start = 0
        window_end = 1
        windows = []
        while window_end < len(sliding_window_rms):
            if sliding_window_rms[window_end] < self.rms_threshold:
                if window_end - window_start > min_window_size:
                    windows.append(
                        (
                            round((window_start + 1) * self.rms_stride, ndigits=2),
                            round(window_end * self.rms_stride, ndigits=2),
                        )
                    )
                window_start = window_end
            window_end += 1

        return windows

    def forward(self, audio_example: AudioExample) -> List[AudioExample]:
        audio_examples = []
        valid_windows = self.get_valid_windows(audio_example.lead)
        for valid_window in valid_windows:
            start = int(valid_window[0] * self.sample_rate)
            end = int(valid_window[1] * self.sample_rate)
            clipped_lead = audio_example.lead[:, start:end]
            clipped_backing = audio_example.backing[:, start:end]
            audio_examples.append(
                AudioExample(
                    audio_example.video_id,
                    clipped_lead,
                    clipped_backing,
                    self.sample_rate,
                    audio_example.title,
                    valid_window,
                )
            )
        return audio_examples


class AudioAugmenter(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 32000,
        num_pitch_augs: int = 4,
        pitch_step_size: int = 2,
        tempo_ratios: Set[float] = {0.8, 0.9, 1.1, 1.25},
        n_fft: int = 4096,
        hop_length: int = 512,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        semitone_steps = [pitch_step_size * (i + 1) for i in range(num_pitch_augs // 2)]
        semitone_steps += [
            pitch_step_size * (-i - 1) for i in range(num_pitch_augs // 2)
        ]
        semitone_steps.append(0)
        if 1.0 not in tempo_ratios:
            tempo_ratios.add(1.0)
        self.semitone_steps = semitone_steps
        self.tempo_ratios = tempo_ratios

        self.spectrogram = T.Spectrogram(power=None, n_fft=n_fft, hop_length=hop_length)
        self.stretch = T.TimeStretch(n_freq=n_fft // 2 + 1, hop_length=hop_length)
        self.inv_spectrogram = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

    def forward(self, audio_example: AudioExample) -> List[AudioExample]:
        audio_examples = []

        for num_semitones in self.semitone_steps:
            pitch_aug_lead = AF.pitch_shift(
                audio_example.lead, self.sample_rate, num_semitones
            )
            pitch_aug_backing = AF.pitch_shift(
                audio_example.backing, self.sample_rate, num_semitones
            )

            for tempo_ratio in self.tempo_ratios:
                aug_lead = self.inv_spectrogram(
                    self.stretch(self.spectrogram(pitch_aug_lead), tempo_ratio)
                )
                aug_backing = self.inv_spectrogram(
                    self.stretch(self.spectrogram(pitch_aug_backing), tempo_ratio)
                )

                audio_examples.append(
                    AudioExample(
                        audio_example.video_id,
                        aug_lead,
                        aug_backing,
                        self.sample_rate,
                        audio_example.title,
                        audio_example.clip_timestamps,
                        pitch_aug=num_semitones,
                        tempo_aug=tempo_ratio,
                    )
                )
        return audio_examples


class AudioPipeline(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 32000,
        rms_threshold: float = 0.0175,
        min_window_size_seconds: float = 30,
        rms_frame_length_seconds: float = 10,
        rms_stride_seconds: float = 0.2,
        num_pitch_augs: int = 4,
        pitch_step_size: int = 2,
        tempo_ratios: Set[float] = {0.8, 0.9, 1.1, 1.25},
        n_fft: int = 4096,
        hop_length: int = 512,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.clipper = ValidWindowClipper(
            sample_rate,
            rms_threshold,
            min_window_size_seconds,
            rms_frame_length_seconds,
            rms_stride_seconds,
        )

        self.augmenter = AudioAugmenter(
            sample_rate,
            num_pitch_augs,
            pitch_step_size,
            tempo_ratios,
            n_fft,
            hop_length,
        )

    def forward(self, audio_example: AudioExample) -> List[AudioExample]:
        clipped_examples = self.clipper(audio_example)

        aug_examples = []
        for clipped_example in clipped_examples:
            aug_examples.extend(self.augmenter(clipped_example))

        return aug_examples
