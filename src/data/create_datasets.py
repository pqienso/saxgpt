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
import random

from audio_util import trim_wav_file
from augmentation import AudioAugmenter
from dataset_util import SequenceDataset, train_test_split
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
            examples.append((lead_audio, backing_audio))
    return examples


def augment_examples(
    examples: List[Tuple[Tensor, Tensor]],
    augmenter: AudioAugmenter,
) -> List[Tuple[Tensor, Tensor]]:
    new_examples = []
    for lead_audio, backing_audio in tqdm(examples):
        augmented_leads = augmenter(lead_audio)
        augmented_backings = augmenter(backing_audio)
        new_examples.extend(
            [
                (lead, backing)
                for lead, backing in zip(augmented_leads, augmented_backings)
            ]
        )
    return new_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Cut audio, augment pitch/tempo, tokenize, and split into train/test/val."
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
        stem_path_str = config["data_paths"]["stem_dest"]
        metadata_path_str = config["data_paths"]["metadata_path"]
        codes_dest_str = config["data_paths"]["codes_dest"]
        datasets_dest_str = config["data_paths"]["datasets_dest"]
        aug_cfg = config["augmentation"]
        test_prop = config["train_test_split"]["test"]
        val_prop = config["train_test_split"]["val"]
        seed = config["train_test_split"]["seed"]
        dataset_args = config["dataset"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    stem_path = Path(stem_path_str)
    codes_dest = Path(codes_dest_str)
    datasets_dest = Path(datasets_dest_str)
    datasets_dest.mkdir(exist_ok=True)
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
        lead_codes = tokenize(processor, model, lead)
        backing_codes = tokenize(processor, model, backing)
        if lead_codes.shape != backing_codes.shape:
            print("\n\nWARNING: lead and backing codes have different shape")
        codes.append((backing_codes, lead_codes))
    torch.save(codes, codes_dest)
    print("\n\nTokenization complete.")

    print("\n\nProducing dataset")
    print("Splitting train / test")
    random.seed(seed)
    train_ds, val_ds, test_ds = train_test_split(codes, test_prop, val_prop)
    print("Constructing datasets")
    train_ds = SequenceDataset(train_ds, **dataset_args)
    val_ds = SequenceDataset(val_ds, **dataset_args)
    test_ds = SequenceDataset(test_ds, **dataset_args)
    torch.save(train_ds, datasets_dest / "train.pt")
    torch.save(val_ds, datasets_dest / "val.pt")
    torch.save(test_ds, datasets_dest / "test.pt")
    
    print("\n\nDatasets created.")
    print(f"train_ds: {len(train_ds)} examples")
    print(f"val_ds: {len(val_ds)} examples")
    print(f"test_ds: {len(test_ds)} examples")
