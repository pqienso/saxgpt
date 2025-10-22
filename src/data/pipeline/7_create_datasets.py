import torch
from torch import Tensor
import random
from pathlib import Path
import argparse
import yaml
from typing import List, Tuple

from ..util.dataset import get_tensor_dataset, train_test_split


def get_all_codes(codes_dir: Path) -> List[Tuple[Tensor, Tensor]]:
    codes = []
    for example_path in codes_dir.glob("*.pt"):
        example = torch.load(example_path, weights_only=False)
        for clip in example:
            codes.append((clip["backing"], clip["lead"]))
    return codes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Create train/test/val datasets from codes."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
    )
    args, _ = parser.parse_known_args()

    print("\n\n" + "=" * 20)
    print("Pipeline Step 7: Create Datasets")
    print("=" * 20 + "\n\n")

    with open(Path(args.config), "r") as file:
        config = yaml.safe_load(file)
    try:
        codes_dir_str = config["data_paths"]["codes_dir"]
        datasets_dir_str = config["data_paths"]["datasets_dir"]

        test_prop = config["train_test_split"]["test"]
        val_prop = config["train_test_split"]["val"]
        seed = config["train_test_split"]["seed"]

        dataset_args = config["dataset"]

        keep_codes = config["intermediates"]["keep_codes"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    codes_dir = Path(codes_dir_str)
    datasets_dir = Path(datasets_dir_str)
    datasets_dir.mkdir(exist_ok=True)

    codes = get_all_codes(codes_dir)

    print("\n\nProducing dataset")
    print("Splitting train / test")
    random.seed(seed)
    train_ds, val_ds, test_ds = train_test_split(codes, test_prop, val_prop)
    print("Constructing datasets")
    train_ds = get_tensor_dataset(train_ds, **dataset_args)
    val_ds = get_tensor_dataset(val_ds, **dataset_args)
    test_ds = get_tensor_dataset(test_ds, **dataset_args)
    torch.save(train_ds, datasets_dir / "train.pt")
    torch.save(val_ds, datasets_dir / "val.pt")
    torch.save(test_ds, datasets_dir / "test.pt")
    
    print("\n\nDatasets created.")
    print(f"train_ds: {len(train_ds)} examples")
    print(f"val_ds: {len(val_ds)} examples")
    print(f"test_ds: {len(test_ds)} examples")

    if not keep_codes:
        for example in codes_dir.glob("*.pt"):
            example.unlink()
