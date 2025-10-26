import yaml
import torch
from typing import Dict

from .distributed import is_main_process, get_rank, print_rank0
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
        print_rank0(f"Created optimizer: {optimizer_type}")

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

    print_rank0("\nModel Information:")
    print_rank0(f"  Total parameters:     {total_params:,}")
    print_rank0(f"  Trainable parameters: {trainable_params:,}")
    print_rank0(f"  Non-trainable params: {total_params - trainable_params:,}")

    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in actual_model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in actual_model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print_rank0(f"  Model size:           {size_mb:.2f} MB")

    if hasattr(model, "module"):
        print_rank0("  DDP wrapped:          Yes")
