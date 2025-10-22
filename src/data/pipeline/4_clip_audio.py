import argparse
import yaml
import torch
from typing import Dict
from tqdm import tqdm
from datetime import timedelta
import json
from pathlib import Path
import pandas as pd

from .audio_util import trim_wav_file


def create_clips(stems_dir: Path, clips_dir: Path, metadata_entry: Dict):
    video_id = metadata_entry["video_id"]

    example_path = clips_dir / f"{video_id}.pt"
    if example_path.exists():
        return

    clips = []
    windows = json.loads(metadata_entry["valid_windows"])
    for window in windows:
        start, end = window[0], window[1]
        lead_audio = trim_wav_file(
            stems_dir / f"sax_{video_id}.wav",
            timedelta(seconds=start),
            timedelta(seconds=end),
        )
        backing_audio = trim_wav_file(
            stems_dir / f"rhythm_{video_id}.wav",
            timedelta(seconds=start),
            timedelta(seconds=end),
        )
        clips.append(
            {
                "lead": lead_audio,
                "backing": backing_audio,
                "start": start,
                "end": end,
            }
        )
    
    torch.save(clips, example_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip audio based on metadata.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Start with clean clips directory"
    )
    args, _ = parser.parse_known_args()

    print("\n\n" + "=" * 20)
    print("Pipeline Step 4: Clip Audio")
    print("=" * 20 + "\n\n")

    with open(Path(args.config), "r") as file:
        config = yaml.safe_load(file)
    try:
        stems_dir_str = config["data_paths"]["stems_dir"]
        clips_dir_str = config["data_paths"]["clips_dir"]
        metadata_path_str = config["data_paths"]["metadata_path"]

        keep_stems = config["intermediates"]["keep_stems"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    clips_dir = Path(clips_dir_str)
    clips_dir.mkdir(exist_ok=True)
    stems_dir = Path(stems_dir_str)
    metadata = pd.read_csv(metadata_path_str).to_dict(orient="records")

    if args.rebuild:
        for example in clips_dir.glob("*.pt"):
            example.unlink()

    print("\nClipping & saving audio files")
    for metadata_entry in tqdm(metadata):
        examples = create_clips(stems_dir, clips_dir, metadata_entry)
    print("\nClipping complete.")

    if not keep_stems:
        for audio_file in stems_dir.glob("*.wav"):
            audio_file.unlink()
