from pathlib import Path
import torch
import demucs.api
import argparse
import yaml

from ..util.stem_split import stem_split_all_in_folder


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Ingest audio and split stems.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA for stem splitting"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Start with clean download directory"
    )
    args, _ = parser.parse_known_args()

    print("\n\n" + "=" * 20)
    print("Pipeline Step 2: Stem Split")
    print("=" * 20 + "\n\n")

    with open(Path(args.config), "r") as file:
        config = yaml.safe_load(file)
    try:
        dl_dir_str = config["data_paths"]["dl_dir"]
        stems_dir_str = config["data_paths"]["stems_dir"]

        n_splits = config["demucs"]["n_splits"]
        n_shifts = config["demucs"]["n_shifts"]
        n_jobs = config["demucs"]["n_jobs"]
        normalize_before = config["demucs"]["normalize_before"]
        normalize_after = config["demucs"]["normalize_after"]

        keep_dl = config["intermediates"]["keep_dl"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    dl_dir = Path(dl_dir_str)
    stems_dir = Path(stems_dir_str)
    stems_dir.mkdir(exist_ok=True)

    if args.rebuild:
        for audio_file in stems_dir.glob("*.wav"):
            audio_file.unlink()

    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"\nUsing device: {device}\n")

    print("\nCommencing stem split\n")
    print(
        f"Init demucs splitter: {n_shifts} shifts, {n_jobs} jobs, {n_splits} splits\n"
    )
    separator = demucs.api.Separator(
        model="htdemucs_6s",
        shifts=n_shifts,
        jobs=n_jobs,
        device=device,
        progress=True,
    )
    print("\nBegin stem split")
    stem_split_all_in_folder(
        dl_dir,
        separator,
        stems_dir,
        n_splits=n_splits,
        normalize_before=normalize_before,
        normalize_after=normalize_after,
    )

    print("\nStem split complete.\n")

    if not keep_dl:
        for audio_file in dl_dir.glob("*.wav"):
            audio_file.unlink()
