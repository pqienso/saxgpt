import torch
import torchaudio.functional as F
import torchaudio.transforms as T
from torch import Tensor
from typing import List


class AudioAugmenter(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 32000,
        num_pitch_augs: int = 4,
        pitch_step_size: int = 2,
        tempo_ratios: List[float] = [0.8, 0.9, 1.1, 1.25],
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
            tempo_ratios.append(1.0)
        self.semitone_steps = semitone_steps
        self.tempo_ratios = tempo_ratios

        self.spectrogram = T.Spectrogram(power=None, n_fft=n_fft, hop_length=hop_length)
        self.stretch = T.TimeStretch(n_freq=n_fft // 2 + 1, hop_length=hop_length)
        self.inv_spectrogram = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

    def forward(self, audio: Tensor) -> List[Tensor]:
        audio_examples = []

        for num_semitones in self.semitone_steps:
            pitch_aug = F.pitch_shift(
                audio, self.sample_rate, num_semitones
            )
            
            for tempo_ratio in self.tempo_ratios:
                aug = self.inv_spectrogram(
                    self.stretch(self.spectrogram(pitch_aug), tempo_ratio)
                )
                audio_examples.append(aug)

        return audio_examples
