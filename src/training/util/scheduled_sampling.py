"""
Scheduled sampling for transformer training.
Gradually transitions from teacher forcing to model predictions.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Literal


class ScheduledSamplingStrategy:
    """Calculate sampling probability based on current epoch."""

    def __init__(
        self,
        start_epoch: int,
        end_epoch: int,
        start_prob: float,
        end_prob: float,
        strategy: Literal["linear", "exponential", "inverse_sigmoid"],
        padding_idx: int,
        num_codebooks: int,
    ):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.strategy = strategy
        self.padding_idx = padding_idx
        self.num_codebooks = num_codebooks

        if start_epoch >= end_epoch:
            raise ValueError("start_epoch must be less than end_epoch")
        if not 0 <= start_prob <= 1 or not 0 <= end_prob <= 1:
            raise ValueError("Probabilities must be in [0, 1]")

    def get_sampling_prob(self, epoch: int) -> float:
        """Calculate sampling probability for current epoch."""
        if epoch < self.start_epoch:
            return self.start_prob
        if epoch >= self.end_epoch:
            return self.end_prob

        # Normalize progress to [0, 1]
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

        if self.strategy == "linear":
            prob = self.start_prob + (self.end_prob - self.start_prob) * progress

        elif self.strategy == "exponential":
            prob = self.start_prob + (self.end_prob - self.start_prob) * (progress**2)

        elif self.strategy == "inverse_sigmoid":
            # Inverse sigmoid (fast start, slow end)
            k = 10  # Steepness parameter
            x = (progress - 0.5) * k
            sigmoid = 1 / (1 + math.exp(-x))
            prob = self.start_prob + (self.end_prob - self.start_prob) * sigmoid

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return prob

    def apply(
        self,
        model: nn.Module,
        src: torch.Tensor,
        tgt: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply scheduled sampling during training.

        Args:
            model: The transformer model
            src: Source tokens [B, C, T_src]
            tgt: Target tokens [B, C, T_tgt]
            epoch: Current epoch of training

        Returns:
            decoder_input: Mixed input for decoder [B, C, T_tgt-1]
            stats: Dictionary with sampling statistics
        """
        sampling_prob = self.get_sampling_prob(epoch)
        if sampling_prob == 0.0:
            # Pure teacher forcing
            return tgt[:, :, :-1], {
                "num_sampled": 0,
                "num_teacher_forced": tgt.size(2) - 1,
            }

        batch_size, _, seq_len = tgt.shape

        # Start with ground truth (excluding last token for teacher forcing)
        decoder_input = tgt[:, :, :-1].clone()  # [B, C, T-1]

        num_sampled = 0

        for t in range(1, seq_len - 1):
            if torch.rand(1).item() < sampling_prob:
                with torch.no_grad():
                    logits = model(src, decoder_input[:, :, :t])  # [B, C, T, V]
                    pred = logits[:, :, -1, :].argmax(dim=-1)  # Last token: [B, C]

                    # Handle delayed codebook pattern
                    for cb_idx in range(self.num_codebooks):
                        if t <= cb_idx:
                            pred[:, cb_idx] = self.padding_idx

                    decoder_input[:, :, t] = pred
                    num_sampled += 1

        stats = {
            "num_sampled": num_sampled,
            "sampling_prob": sampling_prob,
        }

        return decoder_input, stats


def create_scheduled_sampling_strategy(
    config: Dict,
) -> Optional[ScheduledSamplingStrategy]:
    """
    Create scheduled sampling strategy from config.

    Args:
        config: Training configuration dictionary

    Returns:
        ScheduledSamplingStrategy or None if disabled
    """
    ss_config = config["training"].get("scheduled_sampling")
    if ss_config is None:
        return None

    return ScheduledSamplingStrategy(
        start_epoch=ss_config.get("start_epoch", 0),
        end_epoch=ss_config.get("end_epoch", 100),
        start_prob=ss_config.get("start_prob", 0.0),
        end_prob=ss_config.get("end_prob", 0.35),
        strategy=ss_config.get("sampling_strategy", "linear"),
        padding_idx=config["model"]["padding_idx"],
        num_codebooks=config["model"]["num_codebooks"],
    )
