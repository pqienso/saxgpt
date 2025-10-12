"""
Shared utilities for training and evaluation of Multi-Codebook Transformer.
"""

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Tuple, Optional

from ..model.transformer import EncoderDecoderTransformer


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
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create DataLoader with standard settings.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for CUDA

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def setup_device() -> torch.device:
    """
    Setup and return the appropriate device (CUDA or CPU).

    Returns:
        torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    return device


def print_model_info(model: nn.Module):
    """
    Print model parameter information.

    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel Information:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable params: {total_params - trainable_params:,}")

    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print(f"  Model size:           {size_mb:.2f} MB")


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


def load_checkpoint_for_inference(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict:
    """
    Load model checkpoint for inference/evaluation.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        Checkpoint dictionary with metadata
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

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


def load_checkpoint_for_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[int, int, Dict[str, float]]:
    """
    Load checkpoint for resuming training.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scaler: GradScaler to load state into
        scheduler: LR scheduler to load state into (optional)
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        Tuple of (epoch, step, metrics)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    metrics = checkpoint.get("metrics", {})

    print(f"Resumed from epoch {epoch}, step {step}")
    return epoch, step, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    config: Dict,
    checkpoint_path: Path,
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scaler: GradScaler state to save
        scheduler: LR scheduler state to save (optional)
        epoch: Current epoch
        step: Current global step
        metrics: Current metrics
        config: Configuration dictionary
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def create_optimizer(
    model: EncoderDecoderTransformer, config: Dict
) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        model: Model whose parameters to optimize
        config: Configuration dictionary

    Returns:
        Optimizer instance
    """
    optimizer_config = config["training"]["optimizer"]
    optimizer_type = optimizer_config["type"].lower()

    optimizer_args = {
        "lr": optimizer_config["lr"],
        "betas": tuple(optimizer_config.get("betas", [0.9, 0.999])),
        "eps": optimizer_config.get("eps", 1e-8),
        "weight_decay": optimizer_config.get("weight_decay", 0.0),
    }

    embedding_lr = optimizer_config["embedding_lr"]
    if embedding_lr is None:
        parameters = model.parameters()
    else:
        embedding_parameters = []
        other_parameters = []
        for name, param in model.named_parameters():
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

    print(f"Created optimizer: {optimizer_type}")
    return optimizer


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
