"""
Shared utilities for training and evaluation of Multi-Codebook Transformer.
"""

import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from typing import Dict, Tuple, Optional

from .distributed import is_main_process, is_distributed, get_world_size, get_rank
from ...model.transformer import EncoderDecoderTransformer


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.
    Works with both regular models and DDP-wrapped models.

    Args:
        model: Model whose parameters to optimize (can be DDP-wrapped)
        config: Configuration dictionary

    Returns:
        Optimizer instance
    """
    # Unwrap DDP if necessary
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model

    optimizer_config = config["training"]["optimizer"]
    optimizer_type = optimizer_config["type"].lower()

    optimizer_args = {
        "lr": optimizer_config["lr"],
        "betas": tuple(optimizer_config.get("betas", [0.9, 0.999])),
        "eps": optimizer_config.get("eps", 1e-8),
        "weight_decay": optimizer_config.get("weight_decay", 0.0),
    }

    embedding_lr = optimizer_config.get("embedding_lr")
    if embedding_lr is None:
        parameters = actual_model.parameters()
    else:
        embedding_parameters = []
        other_parameters = []
        for name, param in actual_model.named_parameters():
            if ".embedding." in name:
                embedding_parameters.append(param)
            else:
                other_parameters.append(param)
        parameters = [
            {"params": embedding_parameters, "lr": embedding_lr},
            {"params": other_parameters, "lr": optimizer_args["lr"]},
        ]

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(parameters, **optimizer_args)
    elif optimizer_type == "adamw":
        optimizer_args["weight_decay"] = optimizer_config.get("weight_decay", 0.01)
        optimizer = torch.optim.AdamW(parameters, **optimizer_args)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")

    if is_main_process():
        print(f"Created optimizer: {optimizer_type}")

    return optimizer


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict) -> EncoderDecoderTransformer:
    """
    Create model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized model
    """
    model_config = config["model"]
    model = EncoderDecoderTransformer(
        vocab_size=model_config["vocab_size"],
        num_codebooks=model_config["num_codebooks"],
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_encoder_layers=model_config["num_encoder_layers"],
        num_decoder_layers=model_config["num_decoder_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=model_config["dropout"],
        activation=model_config.get("activation", "relu"),
        norm_first=model_config.get("norm_first", False),
        max_seq_len=model_config.get("max_seq_len", 5000),
        padding_idx=model_config["padding_idx"],
        scale_embeddings=model_config.get("scale_embeddings", True),
    )
    return model


def load_dataset(data_path: str, dataset_name: str = "dataset") -> TensorDataset:
    """
    Load dataset from .pt file.

    Args:
        data_path: Path to .pt file
        dataset_name: Name for logging purposes

    Returns:
        TensorDataset
    """
    data_path = Path(data_path)
    print(f"Loading {dataset_name} from: {data_path}")
    data = torch.load(data_path, weights_only=False)

    # Handle both tuple and TensorDataset formats
    if isinstance(data, tuple):
        dataset = TensorDataset(*data)
    else:
        dataset = data

    print(f"{dataset_name.capitalize()} size: {len(dataset)}")
    return dataset


def load_datasets(config: Dict) -> Tuple[TensorDataset, TensorDataset]:
    """
    Load training and validation datasets from configuration.

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


def print_model_info(model: torch.nn.Module, rank: int = None):
    """
    Print model parameter information (only on rank 0 for DDP).

    Args:
        model: PyTorch model (can be DDP-wrapped or not)
        rank: Process rank (None for auto-detect)
    """
    if rank is None:
        rank = get_rank()

    if rank != 0:
        return

    # Handle DDP-wrapped models
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model

    total_params = sum(p.numel() for p in actual_model.parameters())
    trainable_params = sum(
        p.numel() for p in actual_model.parameters() if p.requires_grad
    )

    print("\nModel Information:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable params: {total_params - trainable_params:,}")

    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in actual_model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in actual_model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print(f"  Model size:           {size_mb:.2f} MB")

    if hasattr(model, "module"):
        print("  DDP wrapped:          Yes")


def calculate_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, padding_idx: int
) -> float:
    """
    Calculate token-level accuracy, ignoring padding tokens.

    Args:
        logits: Model output logits [B, C, T, V]
        targets: Target tokens [B, C, T]
        padding_idx: Index to ignore in accuracy calculation

    Returns:
        Accuracy as float between 0 and 1
    """
    predictions = logits.argmax(dim=-1)  # [B, C, T]

    # Create mask for non-padding positions
    mask = targets != padding_idx

    # Calculate accuracy only on non-padded positions
    correct = (predictions == targets) & mask
    total = mask.sum()

    if total == 0:
        return 0.0

    return (correct.sum().float() / total.float()).item()


def load_checkpoint_for_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler,
    checkpoint_path: Path,
    device: torch.device,
    rank: int = None,
):
    """
    Load checkpoint for resuming training.
    Works for both single GPU and DDP.

    Args:
        model: Model to load weights into (can be DDP-wrapped)
        rank: Process rank (None for auto-detect)

    Returns:
        Tuple of (epoch, step, metrics)
    """
    if rank is None:
        rank = get_rank()

    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    map_location = (
        {"cuda:0": f"cuda:{device.index}"}
        if device.type == "cuda" and device.index
        else device
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )

    # Unwrap DDP model if necessary
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    metrics = checkpoint.get("metrics", {})

    if rank == 0:
        print(f"Resumed from epoch {epoch}, step {step}")

    return epoch, step, metrics


def load_checkpoint_for_inference(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    """
    Load model checkpoint for inference/evaluation.
    Handles both regular and DDP-saved checkpoints.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        Checkpoint dictionary with metadata
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both DDP and non-DDP saved models
    state_dict = checkpoint["model_state_dict"]

    # If model is not wrapped but checkpoint is from DDP (keys start with 'module.')
    if not hasattr(model, "module") and any(
        k.startswith("module.") for k in state_dict.keys()
    ):
        # Remove 'module.' prefix
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # If model is wrapped but checkpoint is not from DDP
    if hasattr(model, "module") and not any(
        k.startswith("module.") for k in state_dict.keys()
    ):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    # Extract metadata
    epoch = checkpoint.get("epoch", -1)
    step = checkpoint.get("step", -1)
    metrics = checkpoint.get("metrics", {})

    print("Checkpoint info:")
    print(f"  Epoch: {epoch}")
    print(f"  Step: {step}")
    if metrics:
        print(f"  Metrics: {metrics}")

    return checkpoint


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler,
    epoch: int,
    step: int,
    metrics: dict,
    config: dict,
    checkpoint_path: Path,
    rank: int = None,
):
    """
    Save training checkpoint (only on rank 0 for DDP).

    Args:
        model: Model to save (can be DDP-wrapped)
        rank: Process rank (None for auto-detect)
    """
    if rank is None:
        rank = get_rank()

    # Only save from rank 0
    if rank != 0:
        return

    # Unwrap DDP model if necessary
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def validate_config(config: Dict):
    """
    Validate that required configuration keys are present.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required keys are missing
    """
    required_keys = {
        "model": [
            "vocab_size",
            "num_codebooks",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "num_decoder_layers",
            "dim_feedforward",
            "dropout",
            "padding_idx",
        ],
        "training": ["batch_size", "num_epochs", "optimizer", "output_dir"],
        "data": ["train_path", "val_path"],
    }

    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing required config key: {section}.{key}")

    # Validate optimizer config
    if "type" not in config["training"]["optimizer"]:
        raise ValueError("Missing required config key: training.optimizer.type")
    if "lr" not in config["training"]["optimizer"]:
        raise ValueError("Missing required config key: training.optimizer.lr")
