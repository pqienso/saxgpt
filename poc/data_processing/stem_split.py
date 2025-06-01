import os
import torchaudio
import demucs.api
from glob import glob

from audio_util import mix_audio_values, mono_resample_audio


def stem_split_audio(
    wav_file_name: str,
    separator: demucs.api.Separator,
    download_folder: str,
    final_sr: int = 32000,
    repeated_splits: int = 0,
) -> None:
    audio_name = os.path.basename(wav_file_name)
    sax_wav_path = os.path.join(download_folder, f"sax_{audio_name}")
    rhythm_wav_path = os.path.join(download_folder, f"rhythm_{audio_name}")

    if os.path.exists(sax_wav_path) and os.path.exists(rhythm_wav_path):
        print(f"{wav_file_name} has already been split; skipping")
        return

    _, separated = separator.separate_audio_file(wav_file_name)
    rhythm_audio = mix_audio_values(separated, ["piano", "guitar", "drums", "bass"])
    sax_audio = mix_audio_values(separated, ["vocals", "other"])

    for _ in range(repeated_splits):
        _, separated = separator.separate_tensor(sax_audio)
        rhythm_audio += mix_audio_values(
            separated, ["piano", "guitar", "drums", "bass"]
        )
        sax_audio = mix_audio_values(separated, ["vocals", "other"])

    rhythm_audio = mono_resample_audio(rhythm_audio, separator.samplerate, final_sr)
    torchaudio.save(rhythm_wav_path, rhythm_audio, final_sr)

    sax_audio = mono_resample_audio(sax_audio, separator.samplerate, final_sr)
    torchaudio.save(sax_wav_path, sax_audio, final_sr)


def stem_split_all_in_folder(
    audio_folder: str,
    separator: demucs.api.Separator,
    destination_folder: str,
    final_sr: int = 32000,
    repeated_splits: int = 0,
) -> None:
    file_paths = glob(os.path.join(audio_folder, "*.wav"))
    for file_path in file_paths:
        stem_split_audio(
            file_path, separator, destination_folder, final_sr, repeated_splits
        )
