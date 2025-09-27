import demucs.api
import torch
import argparse
import yaml
from pathlib import Path

from data_ingestion import ingest_audio_url
from stem_split import stem_split_all_in_folder
from metadata import update_metadata


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Ingest audio, split stems and update metadata."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to the YAML configuration file",
        default="config/data/main.yaml",
    )
    args = parser.parse_args()

    with open(Path(args.config_path), "r") as file:
        config = yaml.safe_load(file)
    try:
        dl_dest_str = config["data_paths"]["dl_dest"]
        stem_dest_str = config["data_paths"]["stem_dest"]
        metadata_path_str = config["data_paths"]["metadata_path"]

        url = config["url"]

        n_splits = config["demucs"]["n_splits"]
        n_shifts = config["demucs"]["n_shifts"]
        n_jobs = config["demucs"]["n_jobs"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dl_dest = Path(dl_dest_str).resolve()
    if not dl_dest.exists():
        print(f"dl_dest {str(dl_dest)} does not exist. Creating..")
        dl_dest.mkdir(parents=True, exist_ok=True)

    stem_dest = Path(stem_dest_str).resolve()
    if not stem_dest.exists():
        print(f"stem_dest {str(stem_dest)} does not exist. Creating..")
        stem_dest.mkdir(parents=True, exist_ok=True)

    print("\n\nDownloading audio...\n")
    ingest_audio_url(url, dl_dest)

    print("\n\nAudio download complete. Commencing stem split")
    print(f"Init demucs splitter: {n_shifts} shifts, {n_jobs} jobs, {n_splits} splits")
    separator = demucs.api.Separator(
        model="htdemucs_6s",
        shifts=n_shifts,
        jobs=n_jobs,
        device=device,
        progress=True,
    )
    print("\nBegin stem split")
    stem_split_all_in_folder(
        dl_dest,
        separator,
        stem_dest,
        repeated_splits=n_splits - 1,
    )

    print("\n\nStem split complete. Updating metadata...")
    update_metadata(Path(metadata_path_str), dl_dest, stem_dest)
    print("\n\nDataset created.")
