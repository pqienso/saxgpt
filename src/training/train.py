"""
Unified training script that works with both single GPU and DDP.
Automatically detects the environment and uses the appropriate mode.

Usage:
  # Single GPU (automatic)
  python train_unified.py --config config.yaml

  # Multi-GPU with torchrun
  torchrun --nproc_per_node=4 train_unified.py --config config.yaml

  # Multi-GPU with python spawn
  python train_unified.py --config config.yaml --ddp --num-gpus 4
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime

from .util.training import (
    load_config,
    create_model,
    load_datasets,
    create_dataloader,
    print_model_info,
    calculate_accuracy,
    save_checkpoint,
    load_checkpoint_for_training,
    create_optimizer,
    validate_config,
)
from .util.distributed import (
    get_rank,
    is_distributed,
    setup_distributed,
    cleanup_distributed,
    setup_device,
    print_rank0,
    reduce_tensor,
    barrier,
)
from .lr_scheduling import create_scheduler


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, log_dir: Path):
        self.rank = get_rank()
        self.is_main = self.rank == 0
        self.best_val_loss = float("inf")

        if self.is_main:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = self.log_dir / "metrics.jsonl"

    def log(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Log metrics to file (only on rank 0)."""
        if not self.is_main:
            return

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
        if not self.is_main:
            return False

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    sampler: Optional[DistributedSampler],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: Optional[LRScheduler],
    device: torch.device,
    config: Dict,
    epoch: int,
    metrics_tracker: MetricsTracker,
    global_step: int,
) -> Tuple[float, float, int]:
    """Train for one epoch (works for both single GPU and DDP)."""
    model.train()
    rank = get_rank()

    # Set epoch for distributed sampler
    if sampler is not None:
        sampler.set_epoch(epoch)

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    padding_idx = config["model"]["padding_idx"]
    max_grad_norm = config["training"].get("max_grad_norm", 1.0)
    grad_norm = None
    optimizer.zero_grad()

    # Only show progress bar on rank 0
    pbar = tqdm(dataloader) if rank == 0 else dataloader

    for batch_idx, batch in enumerate(pbar):
        src, tgt = batch
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        with autocast("cuda" if device.type == "cuda" else "cpu"):
            logits = model(src, tgt[:, :, :-1])
            B, C, T, V = logits.shape
            logits_flat = logits.reshape(B * C * T, V)
            tgt_flat = tgt[:, :, 1:].reshape(B * C * T)
            loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=padding_idx)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        total_loss += loss.item() * gradient_accumulation_steps
        accuracy = calculate_accuracy(logits.detach(), tgt[:, :, 1:], padding_idx)
        total_accuracy += accuracy
        num_batches += 1

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

            if global_step % config["training"].get("log_interval", 10) == 0:
                # Calculate local metrics
                avg_loss = total_loss / num_batches
                avg_acc = total_accuracy / num_batches

                # Gather metrics from all GPUs
                if is_distributed():
                    loss_tensor = reduce_tensor(torch.tensor([avg_loss], device=device))
                    acc_tensor = reduce_tensor(torch.tensor([avg_acc], device=device))
                    avg_loss = loss_tensor.item()
                    avg_acc = acc_tensor.item()

                # Log only on rank 0 (but with averaged metrics from all GPUs)
                if rank == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    metrics_tracker.log(
                        epoch,
                        global_step,
                        {
                            "train_loss": avg_loss,
                            "train_accuracy": avg_acc,
                            "learning_rate": current_lr,
                        },
                    )

        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix(
                {
                    "grad": None if grad_norm is None else grad_norm.item(),
                    "loss": f"{total_loss / num_batches:.4f}",
                    "acc": f"{total_accuracy / num_batches:.4f}",
                    "step": global_step,
                }
            )

    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1

    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    # Reduce metrics across GPUs if distributed
    if is_distributed():
        loss_tensor = reduce_tensor(torch.tensor([avg_loss], device=device))
        acc_tensor = reduce_tensor(torch.tensor([avg_accuracy], device=device))
        avg_loss = loss_tensor.item()
        avg_accuracy = acc_tensor.item()

    return avg_loss, avg_accuracy, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict,
) -> Tuple[float, float]:
    """Validate the model (works for both single GPU and DDP)."""
    model.eval()
    rank = get_rank()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    padding_idx = config["model"]["padding_idx"]

    pbar = tqdm(dataloader, desc="Validation") if rank == 0 else dataloader

    for batch in pbar:
        src, tgt = batch
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        with autocast("cuda" if device.type == "cuda" else "cpu"):
            logits = model(src, tgt[:, :, :-1])
            B, C, T, V = logits.shape
            logits_flat = logits.reshape(B * C * T, V)
            tgt_flat = tgt[:, :, 1:].reshape(B * C * T)
            loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=padding_idx)

        total_loss += loss.item()
        accuracy = calculate_accuracy(logits, tgt[:, :, 1:], padding_idx)
        total_accuracy += accuracy
        num_batches += 1

        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / num_batches:.4f}",
                    "acc": f"{total_accuracy / num_batches:.4f}",
                }
            )

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    # Reduce metrics across GPUs if distributed
    if is_distributed():
        loss_tensor = reduce_tensor(torch.tensor([avg_loss], device=device))
        acc_tensor = reduce_tensor(torch.tensor([avg_accuracy], device=device))
        avg_loss = loss_tensor.item()
        avg_accuracy = acc_tensor.item()

    return avg_loss, avg_accuracy


def train(config_path: str):
    """Main training function (works for both single GPU and DDP)."""

    # Setup distributed training if applicable
    rank, local_rank, world_size = setup_distributed()

    # Load config
    config = load_config(config_path)
    validate_config(config)

    print_rank0("Configuration loaded:")
    print_rank0(yaml.dump(config, default_flow_style=False))

    # Setup device
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = setup_device()

    # Create output directories (only rank 0)
    if rank == 0:
        output_dir = Path(config["training"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

    barrier()

    output_dir = Path(config["training"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    metrics_tracker = MetricsTracker(output_dir / "logs")

    # Load datasets
    train_dataset, val_dataset = load_datasets(config)

    # Create dataloaders (automatically handles single GPU vs DDP)
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)
    pin_memory = device.type == "cuda"

    train_loader, train_sampler = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        is_train=True,  # Drop last batch for training
    )

    val_loader, val_sampler = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        is_train=False,  # Don't drop for validation
    )

    # Create model
    model = create_model(config)
    model = model.to(device)
    print_model_info(model)

    # Wrap with DDP if distributed
    if world_size > 1:
        print_rank0(f"Wrapping model with DDP (world_size={world_size})")
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        model_for_saving = model.module
    else:
        model_for_saving = model

    # Create optimizer
    optimizer = create_optimizer(model_for_saving, config)

    # Calculate total training steps
    steps_per_epoch = (
        len(train_loader) // config["training"]["gradient_accumulation_steps"]
    )
    total_training_steps = steps_per_epoch * config["training"]["num_epochs"]

    scheduler = create_scheduler(optimizer, config, total_training_steps)
    if scheduler:
        print_rank0(
            f"Using learning rate scheduler: {config['training']['scheduler']['type']}"
        )

    scaler = GradScaler()

    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    default_checkpoint_path = (
        Path(config["training"]["output_dir"]) / "checkpoints" / "interrupt.pt"
    )
    checkpoint_path = Path(
        config["training"].get("resume_from_checkpoint") or default_checkpoint_path
    )
    if checkpoint_path.exists():
        start_epoch, global_step, metrics = load_checkpoint_for_training(
            model_for_saving, optimizer, scaler, scheduler, checkpoint_path, device
        )
        metrics_tracker.best_val_loss = min(
            metrics_tracker.best_val_loss,
            metrics.get("best_val_loss", float("inf")),
        )
        start_epoch += 1
        print_rank0(f"Resumed from checkpoint: epoch {start_epoch}, step {global_step}")
    else:
        print_rank0(f"Checkpoint not found: {checkpoint_path}")
        print_rank0("Starting training from scratch")

    barrier()

    # Training loop
    num_epochs = config["training"]["num_epochs"]
    save_interval = config["training"].get("save_interval", 1)

    mode = f"DDP on {world_size} GPUs" if world_size > 1 else "Single GPU"
    print_rank0(f"\nStarting training for {num_epochs} epochs ({mode})...")
    print_rank0(f"Batch size per GPU: {batch_size}")
    effective_bs = (
        batch_size * world_size * config["training"]["gradient_accumulation_steps"]
    )
    print_rank0(f"Effective batch size: {effective_bs}")

    try:
        for epoch in range(start_epoch, num_epochs):
            print_rank0(f"\n{'=' * 80}")
            print_rank0(f"Epoch {epoch + 1}/{num_epochs}")
            print_rank0(f"{'=' * 80}")

            # Train
            train_loss, train_acc, global_step = train_epoch(
                model,
                train_loader,
                train_sampler,
                optimizer,
                scaler,
                scheduler,
                device,
                config,
                epoch,
                metrics_tracker,
                global_step,
            )

            print_rank0(
                f"\nTraining - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}"
            )

            # Validate
            val_loss, val_acc = validate(model, val_loader, device, config)
            print_rank0(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

            # Update scheduler
            if scheduler and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)

            # Log and save checkpoints (only rank 0)
            if rank == 0:
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

                is_best = metrics_tracker.update_best(val_loss)
                save_checkpoint(
                    model_for_saving,
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
                    checkpoint_dir / "latest.pt",
                )
                if is_best:
                    print_rank0("New best validation loss!")
                    save_checkpoint(
                        model_for_saving,
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
                        checkpoint_dir / "best.pt",
                    )

                if (epoch + 1) % save_interval == 0:
                    save_checkpoint(
                        model_for_saving,
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
                        checkpoint_dir / f"epoch_{epoch + 1}.pt",
                    )
            barrier()

    except KeyboardInterrupt:
        if rank == 0:
            print_rank0("\n\n" + "=" * 80)
            print_rank0("Training interrupted by user!")
            print_rank0("=" * 80)

            interrupt_path = checkpoint_dir / "interrupt.pt"
            save_checkpoint(
                model_for_saving,
                optimizer,
                scaler,
                scheduler,
                epoch,
                global_step,
                {
                    "train_loss": train_loss
                    if "train_loss" in locals()
                    else float("inf"),
                    "val_loss": val_loss if "val_loss" in locals() else float("inf"),
                    "best_val_loss": metrics_tracker.best_val_loss
                    if "metrics_tracker" in locals()
                    else float("inf"),
                },
                config,
                interrupt_path,
            )

    finally:
        cleanup_distributed()

    print_rank0("\n" + "=" * 80)
    print_rank0("Training complete!")
    print_rank0(f"Best validation loss: {metrics_tracker.best_val_loss:.4f}")
    print_rank0("=" * 80)


def train_worker(rank: int, world_size: int, config_path: Path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    train(config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Multi-Codebook Transformer (Single GPU or DDP)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Force DDP mode with torch.multiprocessing.spawn",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs (only with --ddp flag)",
    )

    args = parser.parse_args()

    if args.ddp:
        # Force DDP mode using spawn
        world_size = args.num_gpus if args.num_gpus else torch.cuda.device_count()

        if world_size < 1:
            raise RuntimeError("No GPUs available for DDP training")

        print_rank0(f"Starting DDP training with spawn on {world_size} GPUs")

        torch.multiprocessing.spawn(
            train_worker,
            args=(world_size, args.config),
            nprocs=world_size,
            join=True,
        )
    else:
        # Auto-detect: single GPU or torchrun
        train(args.config)
