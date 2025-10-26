import torch
from pathlib import Path

from .distributed import get_rank, print_rank0


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
        print_rank0(f"Loading checkpoint from {checkpoint_path}")

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
        print_rank0(f"Resumed from epoch {epoch}, step {step}")

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
    print_rank0(f"Loading checkpoint from {checkpoint_path}")
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

    print_rank0("Checkpoint info:")
    print_rank0(f"  Epoch: {epoch}")
    print_rank0(f"  Step: {step}")
    if metrics:
        print_rank0(f"  Metrics: {metrics}")

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
    print_rank0(f"Checkpoint saved to {checkpoint_path}")
