"""
Monitor training progress by parsing metrics.jsonl
"""

import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import yaml


def load_metrics(metrics_file: Path) -> pd.DataFrame:
    """Load metrics from JSONL file."""
    metrics = []
    with open(metrics_file, "r") as f:
        for line in f:
            metrics.append(json.loads(line))
    return pd.DataFrame(metrics)


def plot_metrics(df: pd.DataFrame, output_path: str = None):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Metrics", fontsize=16)

    # Filter epoch-level metrics
    epoch_df = df[df["epoch_train_loss"].notna()].copy()

    # Plot 1: Training Loss (per step)
    if "train_loss" in df.columns:
        train_loss_df = df[df["train_loss"].notna()]
        axes[0, 0].plot(
            train_loss_df["step"],
            train_loss_df["train_loss"],
            alpha=0.6,
            label="Train Loss",
        )
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss (per step)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Epoch Losses
    if len(epoch_df) > 0:
        axes[0, 1].plot(
            epoch_df["epoch"],
            epoch_df["epoch_train_loss"],
            label="Train Loss",
        )
        axes[0, 1].plot(
            epoch_df["epoch"], epoch_df["epoch_val_loss"], label="Val Loss"
        )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Train vs Validation Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Training Accuracy (per step)
    if "train_accuracy" in df.columns:
        train_acc_df = df[df["train_accuracy"].notna()]
        axes[1, 0].plot(
            train_acc_df["step"],
            train_acc_df["train_accuracy"],
            alpha=0.6,
            label="Train Accuracy",
        )
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_title("Training Accuracy (per step)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Epoch Accuracies
    if len(epoch_df) > 0:
        axes[1, 1].plot(
            epoch_df["epoch"],
            epoch_df["epoch_train_accuracy"],
            label="Train Acc",
        )
        axes[1, 1].plot(
            epoch_df["epoch"],
            epoch_df["epoch_val_accuracy"],
            label="Val Acc",
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].set_title("Train vs Validation Accuracy")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def print_summary(df: pd.DataFrame):
    """Print training summary."""
    epoch_df = df[df["epoch_train_loss"].notna()]

    if len(epoch_df) == 0:
        print("No epoch-level metrics found yet.")
        return

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    print(f"\nTotal epochs completed: {len(epoch_df)}")
    print(f"Total steps: {df['step'].max()}")

    latest = epoch_df.iloc[-1]
    print(f"\nLatest Epoch ({int(latest['epoch'])}):")
    print(f"  Train Loss: {latest['epoch_train_loss']:.4f}")
    print(f"  Val Loss:   {latest['epoch_val_loss']:.4f}")
    print(f"  Train Acc:  {latest['epoch_train_accuracy']:.4f}")
    print(f"  Val Acc:    {latest['epoch_val_accuracy']:.4f}")

    best_epoch = epoch_df.loc[epoch_df["epoch_val_loss"].idxmin()]
    print(f"\nBest Epoch ({int(best_epoch['epoch'])}):")
    print(f"  Val Loss:   {best_epoch['epoch_val_loss']:.4f}")
    print(f"  Val Acc:    {best_epoch['epoch_val_accuracy']:.4f}")

    # Check for overfitting
    if len(epoch_df) > 5:
        recent = epoch_df.tail(5)
        train_trend = (
            recent["epoch_train_loss"].iloc[-1] - recent["epoch_train_loss"].iloc[0]
        )
        val_trend = recent["epoch_val_loss"].iloc[-1] - recent["epoch_val_loss"].iloc[0]

        print("\nRecent Trends (last 5 epochs):")
        print(
            f"  Train Loss: {'↓ Decreasing' if train_trend < 0 else '↑ Increasing'} ({train_trend:+.4f})"
        )
        print(
            f"  Val Loss:   {'↓ Decreasing' if val_trend < 0 else '↑ Increasing'} ({val_trend:+.4f})"
        )

        if train_trend < -0.01 and val_trend > 0.01:
            print("\n  ⚠️  Warning: Possible overfitting detected!")

    print("=" * 80)


def watch_metrics(metrics_file: Path, interval: int = 10):
    """Watch metrics file and update display periodically."""
    import time
    from IPython.display import clear_output

    print(f"Watching {str(metrics_file)} (updating every {interval}s)")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            if metrics_file.exists():
                clear_output(wait=True)
                df = load_metrics(metrics_file)
                print_summary(df)
                print(f"\nLast updated: {pd.Timestamp.now()}")
            else:
                print(f"Waiting for {str(metrics_file)} to be created...")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


if __name__ == "__main__":
    matplotlib.use('QtAgg') 
    parser = argparse.ArgumentParser(description="Monitor training metrics")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training config file",
        default=None,
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="Path to metrics file",
        default=None,
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: show plot)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch metrics file and update periodically",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Update interval in seconds for watch mode",
    )

    args = parser.parse_args()
    if args.metrics:
        metrics_path = Path(args.metrics)
    elif args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        metrics_path = Path(config["training"]["output_dir"]) / "logs/metrics.jsonl"
    else:
        raise ValueError("Please specify either training config or metrics path")

    if args.watch:
        watch_metrics(metrics_path, args.interval)
    else:
        if not metrics_path.exists():
            print(f"Error: Metrics file not found: {str(metrics_path)}")
            exit(1)

        df = load_metrics(metrics_path)
        print_summary(df)

        if args.plot:
            plot_metrics(df, args.output)
