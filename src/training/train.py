import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
import json
from datetime import datetime

from ..model.transformer import EncoderDecoderTransformer
from .lr_scheduling import create_scheduler


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.best_val_loss = float("inf")

    def log(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Log metrics to file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            **metrics,
        }
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def update_best(self, val_loss: float) -> bool:
        """Check if this is the best validation loss."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_datasets(config: Dict) -> Tuple[TensorDataset, TensorDataset]:
    """Load training and validation datasets from .pt files."""
    train_path = Path(config["data"]["train_path"])
    val_path = Path(config["data"]["val_path"])

    print(f"Loading training data from: {train_path}")
    train_data = torch.load(train_path, weights_only=False)

    print(f"Loading validation data from: {val_path}")
    val_data = torch.load(val_path, weights_only=False)

    # Datasets should be TensorDataset or tuple of tensors
    if isinstance(train_data, tuple):
        train_dataset = TensorDataset(*train_data)
    else:
        train_dataset = train_data

    if isinstance(val_data, tuple):
        val_dataset = TensorDataset(*val_data)
    else:
        val_dataset = val_data

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    return train_dataset, val_dataset


def create_model(config: Dict) -> EncoderDecoderTransformer:
    """Create model from config."""
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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: LRScheduler,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    config: Dict,
    checkpoint_path: Path,
    is_best: bool = False,
):
    """Save training checkpoint."""
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

    # Save regular checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_path.parent / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: LRScheduler,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[int, int, Dict[str, float]]:
    """Load training checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

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


def calculate_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, padding_idx: int
) -> float:
    """
    Calculate token-level accuracy.

    Args:
        logits: [B, C, T, V]
        targets: [B, C, T]
        padding_idx: Index to ignore

    Returns:
        Accuracy as float
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: LRScheduler,
    device: torch.device,
    config: Dict,
    epoch: int,
    metrics_tracker: MetricsTracker,
    global_step: int,
) -> Tuple[float, float, int]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    padding_idx = config["model"]["padding_idx"]
    max_grad_norm = config["training"].get("max_grad_norm", 1.0)

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Unpack batch - assuming (src, tgt) format
        # src: [B, C, src_len], tgt: [B, C, tgt_len]
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        # Use mixed precision
        with autocast():
            # Forward pass
            logits = model(src, tgt)  # [B, C, T, V]

            # Calculate loss
            B, C, T, V = logits.shape
            logits_flat = logits.reshape(B * C * T, V)
            tgt_flat = tgt.reshape(B * C * T)

            loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=padding_idx)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Accumulate metrics
        total_loss += loss.item() * gradient_accumulation_steps
        accuracy = calculate_accuracy(logits.detach(), tgt, padding_idx)
        total_accuracy += accuracy
        num_batches += 1

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1

            # Log metrics
            # Update learning rate scheduler (if not ReduceLROnPlateau)
            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

            # Log learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            if global_step % config["training"].get("log_interval", 10) == 0:
                avg_loss = total_loss / num_batches
                avg_acc = total_accuracy / num_batches

                metrics_tracker.log(
                    epoch,
                    global_step,
                    {
                        "train_loss": avg_loss,
                        "train_accuracy": avg_acc,
                        "learning_rate": current_lr,
                    },
                )

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{total_loss / num_batches:.4f}",
                "acc": f"{total_accuracy / num_batches:.4f}",
                "step": global_step,
            }
        )

    # Handle remaining gradients if batch doesn't divide evenly
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy, global_step


@torch.no_grad()
def validate(
    model: nn.Module, dataloader: DataLoader, device: torch.device, config: Dict
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    padding_idx = config["model"]["padding_idx"]

    pbar = tqdm(dataloader, desc="Validation")

    for batch in pbar:
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        with autocast():
            logits = model(src, tgt)

            B, C, T, V = logits.shape
            logits_flat = logits.reshape(B * C * T, V)
            tgt_flat = tgt.reshape(B * C * T)

            loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=padding_idx)

        total_loss += loss.item()
        accuracy = calculate_accuracy(logits, tgt, padding_idx)
        total_accuracy += accuracy
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{total_loss / num_batches:.4f}",
                "acc": f"{total_accuracy / num_batches:.4f}",
            }
        )

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def train(config_path: str):
    """Main training function."""
    # Load config
    config = load_config(config_path)
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create output directories
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(output_dir / "logs")

    # Load datasets
    train_dataset, val_dataset = load_datasets(config)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create model
    model = create_model(config)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create optimizer
    optimizer_config = config["training"]["optimizer"]
    if optimizer_config["type"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["lr"],
            betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
            eps=optimizer_config.get("eps", 1e-8),
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )
    elif optimizer_config["type"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config["lr"],
            betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
            eps=optimizer_config.get("eps", 1e-8),
            weight_decay=optimizer_config.get("weight_decay", 0.01),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")

    # Calculate total training steps, create scheduler
    steps_per_epoch = (
        len(train_loader) // config["training"]["gradient_accumulation_steps"]
    )
    total_training_steps = steps_per_epoch * config["training"]["num_epochs"]

    scheduler = create_scheduler(optimizer, config, total_training_steps)
    if scheduler:
        print(
            f"Using learning rate scheduler: {config['training']['scheduler']['type']}"
        )

    # Create GradScaler for mixed precision
    scaler = GradScaler()

    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    if config["training"].get("resume_from_checkpoint"):
        checkpoint_path = Path(config["training"]["resume_from_checkpoint"])
        if checkpoint_path.exists():
            start_epoch, global_step, _ = load_checkpoint(
                model, optimizer, scaler, scheduler, checkpoint_path, device
            )
            start_epoch += 1  # Start from next epoch
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Starting training from scratch")

    # Training loop
    num_epochs = config["training"]["num_epochs"]
    save_interval = config["training"].get("save_interval", 1)

    print(f"\nStarting training for {num_epochs} epochs...")
    print(
        f"Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}"
    )
    print(
        f"Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}"
    )

    try:
        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 80}")

            # Train
            train_loss, train_acc, global_step = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                scheduler,
                device,
                config,
                epoch,
                metrics_tracker,
                global_step,
            )

            print(f"\nTraining - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

            # Validate
            val_loss, val_acc = validate(model, val_loader, device, config)
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

            # Update ReduceLROnPlateau scheduler
            if scheduler and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)

            # Log epoch metrics
            metrics_tracker.log(
                epoch,
                global_step,
                {
                    "epoch_train_loss": train_loss,
                    "epoch_train_accuracy": train_acc,
                    "epoch_val_loss": val_loss,
                    "epoch_val_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
            )

            # Check if best model
            is_best = metrics_tracker.update_best(val_loss)
            if is_best:
                print("🌟 New best validation loss!")

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    epoch,
                    global_step,
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    },
                    config,
                    checkpoint_path,
                    is_best,
                )

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Training interrupted by user!")
        print("=" * 80)

        # Save interrupt checkpoint
        interrupt_path = checkpoint_dir / "checkpoint_interrupt.pt"
        save_checkpoint(
            model,
            optimizer,
            scaler,
            scheduler,
            epoch,
            global_step,
            {
                "train_loss": train_loss if "train_loss" in locals() else 0.0,
                "val_loss": val_loss if "val_loss" in locals() else 0.0,
            },
            config,
            interrupt_path,
            is_best=False,
        )
        print(f"\nCheckpoint saved to {interrupt_path}")
        print("You can resume training from this checkpoint.")

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {metrics_tracker.best_val_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Multi-Codebook Transformer")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )

    args = parser.parse_args()
    train(args.config)
