import demucs.api
import torch
import argparse
from pathlib import Path

from data_ingestion import is_playlist_url, ingest_audio_url
from stem_split import stem_split_all_in_folder
from metadata import update_metadata

DEFAULT_DATASET_PLAYLIST_URL = "https://youtube.com/playlist?list=PLQquVh8U-z89SjTFOMgGVCuYfv9M-cBdG&si=h2-fwdCY2vQOM-lK"

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dl_dest",
        type=str,
        help="Path to folder for downloaded audio",
    )
    parser.add_argument(
        "stem_dest",
        type=str,
        help="Path to folder for split stems",
    )
    parser.add_argument(
        "metadata_path",
        type=str,
        help="Path to .csv file for audio metadata",
        default=None,
    )
    parser.add_argument(
        "--url",
        type=str,
        help="URL for youtube playlist (dataset)",
        default=DEFAULT_DATASET_PLAYLIST_URL,
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        help="Number of times track is put through stem splitter",
        default=2,
    )
    parser.add_argument(
        "--n-shifts",
        type=int,
        help="Number of shifts per split (for equivariance)",
        default=2,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Number of jobs for stem split",
        default=16,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dl_dest = Path(args.dl_dest).resolve()
    if not dl_dest.exists():
        print(f"dl_dest {str(dl_dest)} does not exist. Creating..")
        try:
            dl_dest.mkdir(exist_ok=True)
        except FileNotFoundError:
            print("Invalid dl_dest.")
            raise

    stem_dest = Path(args.stem_dest).resolve()
    if not stem_dest.exists():
        print(f"dl_dest {str(stem_dest)} does not exist. Creating..")
        try:
            stem_dest.mkdir(exist_ok=True)
        except FileNotFoundError:
            print("Invalid stem_dest.")
            raise
    
    if not is_playlist_url(args.url):
        raise ValueError(f"Invalid youtube playlist url: {args.url}")

    print("\n\nDownloading audio...\n")
    ingest_audio_url(args.url, dl_dest)

    print("\n\nAudio download complete. Commencing stem split")
    print(f"Init demucs splitter: {args.n_shifts} shifts, {args.n_jobs} jobs")
    separator = demucs.api.Separator(
        model="htdemucs_6s",
        shifts=args.n_shifts,
        jobs=args.n_jobs,
        device=device,
        progress=True,
    )
    print("\nBegin stem split")
    stem_split_all_in_folder(
        dl_dest,
        separator,
        stem_dest,
        repeated_splits=args.n_splits - 1,
    )

    print("\n\nStem split complete. Updating metadata...")
    update_metadata(Path(args.metadata_path), dl_dest, stem_dest)
    print("\n\nDataset created.")
