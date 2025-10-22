from pathlib import Path
import torchaudio
import torch
import demucs.api

from .audio_util import mix_audio_values, convert_mono, resample, normalize_lufs


def stem_split_audio(
    wav_file_path: Path,
    separator: demucs.api.Separator,
    download_folder: Path,
    final_sr: int = 32000,
    n_splits: int = 1,
    normalize_before: bool = True,
    normalize_after: bool = True,
    output_mono: bool = True
) -> None:
    audio_name = wav_file_path.name
    sax_wav_path = download_folder / f"sax_{audio_name}"
    rhythm_wav_path = download_folder / f"rhythm_{audio_name}"

    if sax_wav_path.exists() and rhythm_wav_path.exists():
        print(f"{wav_file_path} has already been split; skipping")
        return

    audio, sr = torchaudio.load(wav_file_path)
    if normalize_before:
        audio = normalize_lufs(audio, sr)
    rhythm_audio = torch.zeros_like(audio)

    sax_audio = None
    for _ in range(n_splits):
        _, separated = separator.separate_tensor(audio, sr=sr)
        rhythm_audio += mix_audio_values(
            separated, ["piano", "guitar", "drums", "bass"]
        )
        sax_audio = mix_audio_values(separated, ["vocals", "other"])

    if output_mono:
        rhythm_audio = convert_mono(rhythm_audio)
        sax_audio = convert_mono(sax_audio)
    
    rhythm_audio = resample(rhythm_audio, separator.samplerate, final_sr)
    sax_audio = resample(sax_audio, separator.samplerate, final_sr)

    if normalize_after:
        rhythm_audio = normalize_lufs(rhythm_audio, final_sr)
        sax_audio = normalize_lufs(sax_audio, final_sr)
    
    torchaudio.save(sax_wav_path, sax_audio, final_sr)
    torchaudio.save(rhythm_wav_path, rhythm_audio, final_sr)


def stem_split_all_in_folder(
    audio_folder: Path,
    separator: demucs.api.Separator,
    stems_folder: Path,
    final_sr: int = 32000,
    n_splits: int = 0,
    normalize_before: bool = True,
    normalize_after: bool = True,
) -> None:
    num_files = len(list(audio_folder.glob("*.wav")))
    for index, audio_file in enumerate(audio_folder.glob("*.wav")):
        print(f"\n\nSplitting audio file {index + 1}/{num_files}")
        stem_split_audio(
            audio_file,
            separator,
            stems_folder,
            final_sr,
            n_splits,
            normalize_before,
            normalize_after,
        )
