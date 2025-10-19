import torch
import argparse
from pathlib import Path
import yaml
import random

from .dataset_util import get_tensor_dataset, train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Split codes into train/test/val."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
    )
    args, _ = parser.parse_known_args()

    with open(Path(args.config), "r") as file:
        config = yaml.safe_load(file)
    try:
        codes_dest_str = config["data_paths"]["codes_dest"]
        datasets_dest_str = config["data_paths"]["datasets_dest"]
        test_prop = config["train_test_split"]["test"]
        val_prop = config["train_test_split"]["val"]
        seed = config["train_test_split"]["seed"]
        dataset_args = config["dataset"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    codes_dest = Path(codes_dest_str)
    datasets_dest = Path(datasets_dest_str)
    datasets_dest.mkdir(exist_ok=True)

    codes = torch.load(codes_dest, weights_only=False)

    print("\n\nProducing dataset")
    print("Splitting train / test")
    random.seed(seed)
    train_ds, val_ds, test_ds = train_test_split(codes, test_prop, val_prop)
    print("Constructing datasets")
    train_ds = get_tensor_dataset(train_ds, **dataset_args)
    val_ds = get_tensor_dataset(val_ds, **dataset_args)
    test_ds = get_tensor_dataset(test_ds, **dataset_args)
    torch.save(train_ds, datasets_dest / "train.pt")
    torch.save(val_ds, datasets_dest / "val.pt")
    torch.save(test_ds, datasets_dest / "test.pt")
    
    print("\n\nDatasets created.")
    print(f"train_ds: {len(train_ds)} examples")
    print(f"val_ds: {len(val_ds)} examples")
    print(f"test_ds: {len(test_ds)} examples")
