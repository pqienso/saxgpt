"""
Training script for Multi-Codebook Transformer.
"""

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
import json
from datetime import datetime

from .utils import (
    load_config,
    create_model,
    load_datasets,
    create_dataloader,
    setup_device,
    print_model_info,
    calculate_accuracy,
    save_checkpoint,
    load_checkpoint_for_training,
    create_optimizer,
    validate_config,
)
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

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch_idx, batch in enumerate(pbar):
        # Unpack batch - assuming (src, tgt) format
        # src: [B, C, src_len], tgt: [B, C, tgt_len]
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        # Use mixed precision
        with autocast("cuda"):
            # Forward pass
            logits = model(src, tgt[:, :, :-1])  # [B, C, T, V]

            # Calculate loss
            B, C, T, V = logits.shape
            logits_flat = logits.reshape(B * C * T, V)
            tgt_flat = tgt[:, :, 1:].reshape(B * C * T)

            loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=padding_idx)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Accumulate metrics
        total_loss += loss.item() * gradient_accumulation_steps
        accuracy = calculate_accuracy(logits.detach(), tgt[:, :, 1:], padding_idx)
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

            # Update learning rate scheduler (if not ReduceLROnPlateau)
            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

            # Log metrics
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

        with autocast("cuda"):
            logits = model(src, tgt[:, :, :-1])

            B, C, T, V = logits.shape
            logits_flat = logits.reshape(B * C * T, V)
            tgt_flat = tgt[:, :, 1:].reshape(B * C * T)

            loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=padding_idx)

        total_loss += loss.item()
        accuracy = calculate_accuracy(logits, tgt[:, :, 1:], padding_idx)
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
    # Load and validate config
    config = load_config(config_path)
    validate_config(config)

    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    # Setup device
    device = setup_device()

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
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)
    pin_memory = device.type == "cuda"

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Create model
    model = create_model(config)
    model = model.to(device)
    print_model_info(model)

    # Create optimizer
    optimizer = create_optimizer(model, config)

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
            start_epoch, global_step, _ = load_checkpoint_for_training(
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
        f"Effective batch size: {batch_size * config['training']['gradient_accumulation_steps']}"
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
                best_path = checkpoint_dir / "best_model.pt"
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
                    best_path,
                )

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
