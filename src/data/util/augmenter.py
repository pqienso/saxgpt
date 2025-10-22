import torch
import torchaudio.functional as F
import torchaudio.transforms as T
from torch import Tensor
from typing import List


class AudioAugmenter(torch.nn.Module):
    """
    Augment pitch and tempo of an audio file.
    Pitch and audio are not augmented together to preserve audio quality.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        semitone_steps: List[int] = [-2, 2],
        tempo_ratios: List[float] = [0.9, 1.1],
        n_fft: int = 4096,
        hop_length: int = 512,
    ):
        super().__init__()
        assert 0 not in semitone_steps
        assert 1 not in tempo_ratios
        self.output_len = len(semitone_steps) + len(tempo_ratios) + 1
        self.aug_info = (
            [{"semitone_steps": 0, "tempo_ratio": 1}]
            + [{"semitone_steps": i, "tempo_ratio": 1} for i in semitone_steps]
            + [{"semitone_steps": 0, "tempo_ratio": i} for i in tempo_ratios]
        )

        self.sample_rate = sample_rate
        self.semitone_steps = semitone_steps
        self.tempo_ratios = tempo_ratios

        self.spectrogram = T.Spectrogram(power=None, n_fft=n_fft, hop_length=hop_length)
        self.stretch = T.TimeStretch(n_freq=n_fft // 2 + 1, hop_length=hop_length)
        self.inv_spectrogram = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

    def forward(self, audio: Tensor) -> List[Tensor]:
        audio_examples = [audio]

        for num_semitones in self.semitone_steps:
            aug = F.pitch_shift(audio, self.sample_rate, num_semitones)
            audio_examples.append(aug)

        for tempo_ratio in self.tempo_ratios:
            aug = self.inv_spectrogram(
                self.stretch(self.spectrogram(audio), tempo_ratio)
            )
            audio_examples.append(aug)

        return audio_examples
