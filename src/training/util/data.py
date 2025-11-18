from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Tuple, Optional

from .distributed import is_distributed, get_world_size, get_rank, print_rank0
from .gcs_data import load_dataset_from_path


def load_dataset(data_path: str, dataset_name: str = "dataset") -> TensorDataset:
    """
    Load dataset from local or GCS path.

    Args:
        data_path: Path to .pt file (local or gs://...)
        dataset_name: Name for logging purposes

    Returns:
        TensorDataset
    """
    print_rank0(f"Loading {dataset_name} from: {data_path}")
    
    # Use GCS-aware loading
    data = load_dataset_from_path(data_path)

    # Handle both tuple and TensorDataset formats
    if isinstance(data, tuple):
        dataset = TensorDataset(*data)
    else:
        dataset = data

    print_rank0(f"{dataset_name.capitalize()} size: {len(dataset)}")
    return dataset


def load_datasets(config: Dict) -> Tuple[TensorDataset, TensorDataset]:
    """
    Load training and validation datasets from configuration.
    Supports both local and GCS paths.

    Args:
        config: Configuration dictionary with data paths

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_dataset = load_dataset(
        config["data"]["train_path"], dataset_name="training data"
    )
    val_dataset = load_dataset(
        config["data"]["val_path"], dataset_name="validation data"
    )

    return train_dataset, val_dataset


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    is_train: bool = True,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Create dataloader that works for both single GPU and DDP.

    Args:
        is_train: If True, use drop_last for training stability.
                 If False (validation), don't drop to see all data.

    Returns:
        Tuple of (DataLoader, sampler) where sampler is None for single GPU
    """
    if is_distributed():
        # For training: drop_last=True prevents issues with gradient sync
        # For validation: drop_last=False to evaluate on all data
        drop_last = is_train

        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return loader, sampler
    else:
        # Single GPU: no need to drop any data
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        return loader, None
