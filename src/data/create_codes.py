from transformers import AutoProcessor
from transformers import EncodecModel
import torch
from torch import Tensor
import argparse
from tqdm import tqdm
from pathlib import Path
import yaml
import pandas as pd
from typing import List, Tuple, Dict
import json
from datetime import timedelta

from audio_util import trim_wav_file
from augmentation import AudioAugmenter, augment_examples
from tokenization import tokenize


def clip_valid_windows(metadata: List[Dict]) -> List[Tuple[Tensor, Tensor]]:
    examples = []
    for metadata_entry in tqdm(metadata):
        video_id = metadata_entry["video_id"]
        windows = json.loads(metadata_entry["valid_windows"])
        for window in windows:
            start, end = window[0], window[1]
            lead_audio = trim_wav_file(
                stem_path / f"sax_{video_id}.wav",
                timedelta(seconds=start),
                timedelta(seconds=end),
            )
            backing_audio = trim_wav_file(
                stem_path / f"rhythm_{video_id}.wav",
                timedelta(seconds=start),
                timedelta(seconds=end),
            )
            examples.append((backing_audio, lead_audio))
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Cut audio, augment pitch/tempo and tokenize audio."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
        default="config/data/main.yaml",
    )
    args = parser.parse_args()

    with open(Path(args.config), "r") as file:
        config = yaml.safe_load(file)
    try:
        stem_path_str = config["data_paths"]["stem_dest"]
        metadata_path_str = config["data_paths"]["metadata_path"]
        codes_dest_str = config["data_paths"]["codes_dest"]
        aug_cfg = config["augmentation"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    stem_path = Path(stem_path_str)
    codes_dest = Path(codes_dest_str)
    metadata = pd.read_csv(metadata_path_str).to_dict(orient="records")

    print("\n\nClipping audio files...")
    examples = clip_valid_windows(metadata)
    print(f"Got {len(examples)} clips from {len(metadata)} audio files.")

    if aug_cfg is not None:
        print("\n\nAugmenting audio clips...")
        augmenter = AudioAugmenter(**aug_cfg)
        examples = augment_examples(examples, augmenter)

    print("\n\nBeginning tokenization")
    print("Getting model and processor")
    model = EncodecModel.from_pretrained("facebook/encodec_32khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")

    codes = []
    for backing, lead in tqdm(examples):
        lead_codes = tokenize(lead, processor, model)
        backing_codes = tokenize(backing, processor, model)
        if lead_codes.shape != backing_codes.shape:
            print("\n\nWARNING: lead and backing codes have different shape")
        codes.append((backing_codes, lead_codes))
    torch.save(codes, codes_dest)
    print(f"\n\nTokenization complete. Codes saved to {codes_dest}")
