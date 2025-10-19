import pandas as pd
import json
import argparse
import yaml
from pathlib import Path
from typing import Optional
from tqdm.auto import tqdm

from .data_ingestion import extract_url_title
from .audio_util import get_audio_length, extract_windows_above_threshold

tqdm.pandas()


def update_index(metadata_path: Optional[Path], download_folder: Path) -> pd.DataFrame:
    if metadata_path is None:
        df = pd.DataFrame(
            {col: [] for col in ["video_id", "video_title", "valid_windows"]}
        )
    else:
        df = pd.read_csv(metadata_path).set_index("video_id", drop=True)
    audio_files = download_folder.glob("*.wav")

    new_audio_metadata = []
    for audio_file_path in audio_files:
        video_id = audio_file_path.stem
        if video_id not in df.index:
            video_title = extract_url_title(
                f"https://www.youtube.com/watch?v={video_id}"
            )
            new_audio_metadata.append(
                {
                    "video_id": video_id,
                    "video_title": video_title,
                    "valid_windows": pd.NA,
                }
            )

    new_df = pd.DataFrame(new_audio_metadata)
    df = pd.concat([new_df, df.reset_index()], axis="index").reset_index(drop=True)
    return df


def add_valid_windows(df: pd.DataFrame, stems_folder: Path, **kwargs) -> pd.DataFrame:
    def get_sax_wav_path(video_id: str) -> Path:
        return stems_folder / f"sax_{video_id}.wav"

    df["valid_windows"] = df["video_id"].progress_apply(
        lambda x: json.dumps(
            extract_windows_above_threshold(get_sax_wav_path(x), **kwargs),
        )
    )
    return df


def add_audio_lengths(df: pd.DataFrame, download_folder: Path) -> pd.DataFrame:
    def get_full_wav_path(video_id: str) -> Path:
        return download_folder / f"{video_id}.wav"

    df["audio_length"] = df["video_id"].progress_apply(
        lambda x: round(get_audio_length(get_full_wav_path(x)), ndigits=3)
    )
    return df


def update_metadata(
    metadata_path: Path,
    download_folder: Path,
    stems_folder: Path,
    **kwargs,
) -> None:
    print("Updating index...")
    df = update_index(
        metadata_path if metadata_path.exists() else None, download_folder
    )
    print("\n\nUpdating valid windows...")
    df = add_valid_windows(df, stems_folder, **kwargs)
    print("\n\nUpdating audio lengths...")
    df = add_audio_lengths(df, download_folder)

    if "index" in df.columns:
        df.drop(columns="index", inplace=True)
    df.to_csv(metadata_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Update audio metadata.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
    )
    args, _ = parser.parse_known_args()

    with open(Path(args.config), "r") as file:
        config = yaml.safe_load(file)
    try:
        dl_dest_str = config["data_paths"]["dl_dest"]
        stem_dest_str = config["data_paths"]["stem_dest"]
        metadata_path_str = config["data_paths"]["metadata_path"]

        url = config["url"]

        n_splits = config["demucs"]["n_splits"]
        n_shifts = config["demucs"]["n_shifts"]
        n_jobs = config["demucs"]["n_jobs"]

        rms_window_args = config["rms_window"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    dl_dest = Path(dl_dest_str).resolve()
    stem_dest = Path(stem_dest_str).resolve()

    update_metadata(Path(metadata_path_str), dl_dest, stem_dest, **rms_window_args)
    print("\n\nMetadata updated.")
