"""
Evaluation script for Multi-Codebook Transformer.
"""

import torch
import torch.nn.functional as F
from torch.amp import autocast
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

from .utils import (
    load_config,
    create_model,
    load_dataset,
    create_dataloader,
    setup_device,
    print_model_info,
    load_checkpoint_for_inference,
    validate_config,
)


class EvaluationMetrics:
    """Compute and track evaluation metrics."""

    def __init__(self, num_codebooks: int, padding_idx: int):
        self.num_codebooks = num_codebooks
        self.padding_idx = padding_idx
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.correct_tokens = 0
        self.codebook_correct = defaultdict(int)
        self.codebook_total = defaultdict(int)
        self.perplexities = []
        self.num_batches = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
    ):
        """
        Update metrics with batch results.

        Args:
            logits: [B, C, T, V]
            targets: [B, C, T]
            loss: scalar loss value
        """
        B, C, T, V = logits.shape
        predictions = logits.argmax(dim=-1)  # [B, C, T]

        # Create mask for non-padding positions
        mask = targets != self.padding_idx

        # Overall accuracy
        correct = (predictions == targets) & mask
        self.correct_tokens += correct.sum().item()
        self.total_tokens += mask.sum().item()

        # Per-codebook accuracy
        for c in range(C):
            codebook_mask = mask[:, c, :]
            codebook_correct = (predictions[:, c, :] == targets[:, c, :]) & codebook_mask
            self.codebook_correct[c] += codebook_correct.sum().item()
            self.codebook_total[c] += codebook_mask.sum().item()

        # Loss and perplexity
        self.total_loss += loss
        self.perplexities.append(np.exp(loss))
        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        metrics = {
            "loss": self.total_loss / self.num_batches,
            "perplexity": np.mean(self.perplexities),
            "accuracy": self.correct_tokens / self.total_tokens if self.total_tokens > 0 else 0.0,
        }

        # Per-codebook accuracies
        for c in range(self.num_codebooks):
            if self.codebook_total[c] > 0:
                acc = self.codebook_correct[c] / self.codebook_total[c]
                metrics[f"codebook_{c}_accuracy"] = acc

        return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict,
    compute_detailed_metrics: bool = True,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Evaluate the model on test set.

    Args:
        model: The model to evaluate
        dataloader: Test data loader
        device: Device to run on
        config: Configuration dict
        compute_detailed_metrics: Whether to compute per-sample metrics

    Returns:
        metrics: Dictionary of aggregate metrics
        sample_results: List of per-sample results (if compute_detailed_metrics=True)
    """
    model.eval()
    
    num_codebooks = config["model"]["num_codebooks"]
    padding_idx = config["model"]["padding_idx"]
    
    metrics_tracker = EvaluationMetrics(num_codebooks, padding_idx)
    sample_results = []

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch_idx, batch in enumerate(pbar):
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        with autocast("cuda"):
            logits = model(src, tgt)  # [B, C, T, V]

            B, C, T, V = logits.shape
            logits_flat = logits.reshape(B * C * T, V)
            tgt_flat = tgt.reshape(B * C * T)

            loss = F.cross_entropy(
                logits_flat, tgt_flat, ignore_index=padding_idx, reduction="mean"
            )

        # Update aggregate metrics
        metrics_tracker.update(logits, tgt, loss.item())

        # Compute per-sample metrics if requested
        if compute_detailed_metrics:
            predictions = logits.argmax(dim=-1)  # [B, C, T]
            
            for i in range(B):
                sample_pred = predictions[i]  # [C, T]
                sample_tgt = tgt[i]  # [C, T]
                sample_mask = sample_tgt != padding_idx
                
                # Per-sample accuracy
                sample_correct = ((sample_pred == sample_tgt) & sample_mask).sum().item()
                sample_total = sample_mask.sum().item()
                sample_acc = sample_correct / sample_total if sample_total > 0 else 0.0
                
                # Per-codebook accuracies for this sample
                codebook_accs = {}
                for c in range(C):
                    cb_mask = sample_mask[c]
                    cb_correct = ((sample_pred[c] == sample_tgt[c]) & cb_mask).sum().item()
                    cb_total = cb_mask.sum().item()
                    codebook_accs[f"codebook_{c}"] = cb_correct / cb_total if cb_total > 0 else 0.0
                
                sample_results.append({
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "accuracy": sample_acc,
                    "num_tokens": sample_total,
                    **codebook_accs,
                })

        # Update progress bar
        current_metrics = metrics_tracker.compute()
        pbar.set_postfix({
            "loss": f"{current_metrics['loss']:.4f}",
            "acc": f"{current_metrics['accuracy']:.4f}",
            "ppl": f"{current_metrics['perplexity']:.2f}",
        })

    # Compute final metrics
    final_metrics = metrics_tracker.compute()

    return final_metrics, sample_results


@torch.no_grad()
def compute_confusion_statistics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict,
    max_batches: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute confusion matrices for each codebook.

    Args:
        model: The model to evaluate
        dataloader: Test data loader
        device: Device to run on
        config: Configuration dict
        max_batches: Maximum number of batches to process (for memory efficiency)

    Returns:
        Dictionary mapping codebook index to confusion matrix
    """
    model.eval()
    
    vocab_size = config["model"]["vocab_size"]
    num_codebooks = config["model"]["num_codebooks"]
    padding_idx = config["model"]["padding_idx"]
    
    # Initialize confusion matrices
    confusion_matrices = {
        c: np.zeros((vocab_size, vocab_size), dtype=np.int64)
        for c in range(num_codebooks)
    }

    pbar = tqdm(dataloader, desc="Computing confusion matrices", total=min(max_batches, len(dataloader)))

    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= max_batches:
            break

        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        with autocast("cuda"):
            logits = model(src, tgt)  # [B, C, T, V]

        predictions = logits.argmax(dim=-1)  # [B, C, T]

        # Update confusion matrices for each codebook
        for c in range(num_codebooks):
            pred_c = predictions[:, c, :].cpu().numpy().flatten()
            tgt_c = tgt[:, c, :].cpu().numpy().flatten()
            
            # Filter out padding
            mask = tgt_c != padding_idx
            pred_c = pred_c[mask]
            tgt_c = tgt_c[mask]
            
            # Update confusion matrix
            for true_label, pred_label in zip(tgt_c, pred_c):
                confusion_matrices[c][true_label, pred_label] += 1

    return confusion_matrices


def save_results(
    metrics: Dict[str, float],
    sample_results: List[Dict],
    config: Dict,
    output_dir: Path,
    checkpoint_name: str,
):
    """Save evaluation results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save aggregate metrics
    metrics_file = output_dir / f"eval_metrics_{timestamp}.json"
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_name,
        "config": config,
        "metrics": metrics,
    }
    
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_file}")

    # Save per-sample results
    if sample_results:
        samples_file = output_dir / f"eval_samples_{timestamp}.jsonl"
        with open(samples_file, "w") as f:
            for sample in sample_results:
                f.write(json.dumps(sample) + "\n")
        
        print(f"Sample-level results saved to: {samples_file}")


def print_metrics(metrics: Dict[str, float], num_codebooks: int):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print("\nAggregate Metrics:")
    print(f"  Loss:       {metrics['loss']:.6f}")
    print(f"  Perplexity: {metrics['perplexity']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print("\nPer-Codebook Accuracy:")
    for c in range(num_codebooks):
        key = f"codebook_{c}_accuracy"
        if key in metrics:
            acc = metrics[key]
            print(f"  Codebook {c}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("=" * 80)


def evaluate_model(
    config_path: str,
    checkpoint_path: str,
    test_data_path: str,
    output_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    compute_confusion: bool = False,
    save_sample_results: bool = True,
):
    """
    Main evaluation function.

    Args:
        config_path: Path to config YAML file
        checkpoint_path: Path to model checkpoint
        test_data_path: Path to test dataset .pt file
        output_dir: Directory to save results (default: checkpoint_dir/evaluation)
        batch_size: Batch size for evaluation (default: from config)
        compute_confusion: Whether to compute confusion matrices
        save_sample_results: Whether to save per-sample results
    """
    config = load_config(config_path)
    validate_config(config)
    print("Configuration loaded")

    device = setup_device()

    if output_dir is None:
        checkpoint_path_obj = Path(checkpoint_path)
        output_dir = checkpoint_path_obj.parent.parent / "evaluation"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    test_dataset = load_dataset(test_data_path, dataset_name="test data")

    eval_batch_size = batch_size or config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)
    pin_memory = device.type == "cuda" 

    test_loader = create_dataloader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = create_model(config)
    model = model.to(device)

    load_checkpoint_for_inference(model, Path(checkpoint_path), device)
    print_model_info(model)

    # Evaluate
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80)

    metrics, sample_results = evaluate(
        model,
        test_loader,
        device,
        config,
        compute_detailed_metrics=save_sample_results,
    )

    # Print results
    print_metrics(metrics, config["model"]["num_codebooks"])

    # Save results
    checkpoint_name = Path(checkpoint_path).name
    save_results(
        metrics,
        sample_results if save_sample_results else [],
        config,
        output_dir,
        checkpoint_name,
    )

    # Compute confusion matrices if requested
    if compute_confusion:
        print("\n" + "=" * 80)
        print("Computing confusion matrices...")
        print("=" * 80)
        
        confusion_matrices = compute_confusion_statistics(
            model, test_loader, device, config
        )
        
        # Save confusion matrices
        confusion_file = output_dir / f"confusion_matrices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez(confusion_file, **{f"codebook_{c}": mat for c, mat in confusion_matrices.items()})
        print(f"\nConfusion matrices saved to: {confusion_file}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Multi-Codebook Transformer")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test dataset .pt file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: checkpoint_dir/evaluation)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: from config)",
    )
    parser.add_argument(
        "--compute-confusion",
        action="store_true",
        help="Compute confusion matrices for each codebook",
    )
    parser.add_argument(
        "--no-save-samples",
        action="store_true",
        help="Don't save per-sample results",
    )

    args = parser.parse_args()

    evaluate_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        compute_confusion=args.compute_confusion,
        save_sample_results=not args.no_save_samples,
    )
