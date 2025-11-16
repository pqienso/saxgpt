"""
Comprehensive evaluation script for multi-codebook transformer models.
Evaluates models on test data with various metrics including FAD.

Usage:
    python -m src.training.evaluate --config config/model/medium.yaml --checkpoint models/medium/checkpoints/best.pt
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import json
from tqdm import tqdm
from typing import Dict, List
import numpy as np
from datetime import datetime
from frechet_audio_distance import FrechetAudioDistance
import torchaudio

from ..model.transformer import EncoderDecoderTransformer
from ..training.util.checkpointing import load_checkpoint_for_inference
from ..training.util.config_model import create_model
from ..data.util.tokenization import detokenize
from ..data.util.codes_interleaving import remove_delay_interleaving


TARGET_SR = 16000
SOURCE_SR = 32000

class ModelEvaluator:
    """Comprehensive model evaluation."""

    def __init__(
        self,
        model: EncoderDecoderTransformer,
        test_loader: DataLoader,
        device: torch.device,
        config: Dict,
    ):
        self.model = model.to(device).eval()
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.padding_idx = config["model"]["padding_idx"]
        self.num_codebooks = config["model"]["num_codebooks"]
        self.vocab_size = config["model"]["vocab_size"]

    @torch.no_grad()
    def evaluate_loss_and_accuracy(self) -> Dict[str, float]:
        """Compute test loss and accuracy."""
        print("\n" + "=" * 80)
        print("Computing Loss and Accuracy on Test Set")
        print("=" * 80)

        total_loss = 0.0
        total_accuracy = 0.0
        total_perplexity = 0.0
        num_batches = 0

        # Per-codebook metrics
        codebook_losses = [0.0] * self.num_codebooks
        codebook_accuracies = [0.0] * self.num_codebooks

        for src, tgt in tqdm(self.test_loader, desc="Evaluating"):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Forward pass
            logits = self.model(src, tgt[:, :, :-1])
            B, C, T, V = logits.shape

            # Compute overall loss
            logits_flat = logits.reshape(B * C * T, V)
            tgt_flat = tgt[:, :, 1:].reshape(B * C * T)
            loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=self.padding_idx)

            total_loss += loss.item()

            # Compute accuracy
            predictions = logits.argmax(dim=-1)
            mask = tgt[:, :, 1:] != self.padding_idx
            correct = (predictions == tgt[:, :, 1:]) & mask
            accuracy = correct.sum().float() / mask.sum().float()
            total_accuracy += accuracy.item()

            # Compute perplexity
            perplexity = torch.exp(loss)
            total_perplexity += perplexity.item()

            # Per-codebook metrics
            for cb_idx in range(self.num_codebooks):
                cb_logits = logits[:, cb_idx, :, :].reshape(B * T, V)
                cb_target = tgt[:, cb_idx, 1:].reshape(B * T)
                cb_loss = F.cross_entropy(
                    cb_logits, cb_target, ignore_index=self.padding_idx
                )
                codebook_losses[cb_idx] += cb_loss.item()

                cb_mask = cb_target != self.padding_idx
                cb_correct = (
                    predictions[:, cb_idx, :].reshape(B * T) == cb_target
                ) & cb_mask
                cb_accuracy = cb_correct.sum().float() / cb_mask.sum().float()
                codebook_accuracies[cb_idx] += cb_accuracy.item()

            num_batches += 1

        # Average metrics
        metrics = {
            "test_loss": total_loss / num_batches,
            "test_accuracy": total_accuracy / num_batches,
            "test_perplexity": total_perplexity / num_batches,
        }

        # Add per-codebook metrics
        for cb_idx in range(self.num_codebooks):
            metrics[f"codebook_{cb_idx}_loss"] = codebook_losses[cb_idx] / num_batches
            metrics[f"codebook_{cb_idx}_accuracy"] = (
                codebook_accuracies[cb_idx] / num_batches
            )

        return metrics

    @torch.no_grad()
    def evaluate_generation_quality(
        self,
        num_samples: int = 10,
        max_len: int = 1500,
        temperatures: List[float] = [0.8, 1.0, 1.2],
    ) -> Dict[str, float]:
        """Evaluate generation quality with different temperatures."""
        print("\n" + "=" * 80)
        print(f"Evaluating Generation Quality (num_samples={num_samples})")
        print("=" * 80)

        results = {}

        # Get sample batch
        src_batch, tgt_batch = next(iter(self.test_loader))
        src_batch = src_batch[:num_samples].to(self.device)
        tgt_batch = tgt_batch[:num_samples].to(self.device)

        for temp in temperatures:
            print(f"\nTemperature: {temp}")

            # Generate
            start_tokens = tgt_batch[:, :, 0]  # First token of target
            generated = self.model.generate(
                src_batch,
                max_len=max_len,
                start_tokens=start_tokens,
                temperature=temp,
                top_k=50,
                top_p=0.9,
            )

            # Compute metrics comparing generated to ground truth
            gen_len = min(generated.size(2), tgt_batch.size(2))
            generated_trimmed = generated[:, :, :gen_len]
            tgt_trimmed = tgt_batch[:, :, :gen_len]

            # Token-level accuracy
            mask = tgt_trimmed != self.padding_idx
            matches = (generated_trimmed == tgt_trimmed) & mask
            accuracy = matches.sum().float() / mask.sum().float()

            # Sequence-level accuracy (perfect matches)
            seq_matches = (generated_trimmed == tgt_trimmed).all(dim=(1, 2))
            seq_accuracy = seq_matches.float().mean()

            results[f"gen_accuracy_temp_{temp}"] = accuracy.item()
            results[f"gen_seq_accuracy_temp_{temp}"] = seq_accuracy.item()

            print(f"  Token accuracy: {accuracy.item():.4f}")
            print(f"  Sequence accuracy: {seq_accuracy.item():.4f}")

        return results

    @torch.no_grad()
    def evaluate_autoregressive_consistency(
        self,
        num_samples: int = 5,
        max_len: int = 100,
    ) -> Dict[str, float]:
        """Check consistency between teacher forcing and autoregressive generation."""
        print("\n" + "=" * 80)
        print("Evaluating Autoregressive Consistency")
        print("=" * 80)

        src_batch, tgt_batch = next(iter(self.test_loader))
        src_batch = src_batch[:num_samples].to(self.device)
        tgt_batch = tgt_batch[:num_samples, :, :max_len].to(self.device)

        # Teacher forcing predictions
        logits_tf = self.model(src_batch, tgt_batch[:, :, :-1])
        pred_tf = logits_tf.argmax(dim=-1)

        # Autoregressive generation
        start_tokens = tgt_batch[:, :, 0]
        generated = self.model.generate_greedy(
            src_batch,
            max_len=tgt_batch.size(2),
            start_tokens=start_tokens,
        )

        # Compare (skip first token and handle delayed codebooks)
        comparison_start = self.num_codebooks + 1
        pred_tf_trimmed = pred_tf[:, :, comparison_start:]
        generated_trimmed = generated[:, :, comparison_start + 1 :]

        min_len = min(pred_tf_trimmed.size(2), generated_trimmed.size(2))
        pred_tf_trimmed = pred_tf_trimmed[:, :, :min_len]
        generated_trimmed = generated_trimmed[:, :, :min_len]

        # Token-level consistency
        matches = pred_tf_trimmed == generated_trimmed
        consistency = matches.float().mean()

        # Per-position consistency (to detect where divergence starts)
        position_consistency = matches.float().mean(dim=(0, 1))
        first_divergence = None
        for pos, cons in enumerate(position_consistency):
            if cons < 0.95:  # Less than 95% match
                first_divergence = pos + comparison_start
                break

        results = {
            "autoregressive_consistency": consistency.item(),
            "first_divergence_position": first_divergence if first_divergence else -1,
        }

        print(f"Autoregressive consistency: {consistency.item():.4f}")
        if first_divergence:
            print(f"First significant divergence at position: {first_divergence}")

        return results

    @torch.no_grad()
    def analyze_token_distribution(self) -> Dict[str, any]:
        """Analyze predicted token distributions."""
        print("\n" + "=" * 80)
        print("Analyzing Token Distribution")
        print("=" * 80)

        all_predictions = []
        all_targets = []

        for src, tgt in tqdm(self.test_loader, desc="Collecting tokens"):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            logits = self.model(src, tgt[:, :, :-1])
            predictions = logits.argmax(dim=-1)

            # Filter out padding
            mask = tgt[:, :, 1:] != self.padding_idx
            pred_tokens = predictions[mask].cpu().numpy()
            tgt_tokens = tgt[:, :, 1:][mask].cpu().numpy()

            all_predictions.extend(pred_tokens)
            all_targets.extend(tgt_tokens)

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Compute statistics
        pred_unique, pred_counts = np.unique(all_predictions, return_counts=True)
        tgt_unique, tgt_counts = np.unique(all_targets, return_counts=True)

        results = {
            "num_unique_predicted": len(pred_unique),
            "num_unique_target": len(tgt_unique),
            "predicted_entropy": -np.sum(
                (pred_counts / pred_counts.sum())
                * np.log(pred_counts / pred_counts.sum() + 1e-10)
            ),
            "target_entropy": -np.sum(
                (tgt_counts / tgt_counts.sum())
                * np.log(tgt_counts / tgt_counts.sum() + 1e-10)
            ),
        }

        print(
            f"Unique predicted tokens: {results['num_unique_predicted']} / {self.vocab_size - 1}"
        )
        print(
            f"Unique target tokens: {results['num_unique_target']} / {self.vocab_size - 1}"
        )
        print(f"Predicted entropy: {results['predicted_entropy']:.4f}")
        print(f"Target entropy: {results['target_entropy']:.4f}")

        return results

    @torch.no_grad()
    def save_audio_samples(
        self,
        output_dir: Path,
        num_samples: int = 5,
        max_len: int = 1500,
        save_for_fad: bool = True,
    ):
        """Generate and save audio samples."""
        print("\n" + "=" * 80)
        print(f"Generating Audio Samples (saving to {output_dir})")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for FAD
        if save_for_fad:
            fad_gen_dir = output_dir / "fad_generated"
            fad_ref_dir = output_dir / "fad_reference"
            fad_gen_dir.mkdir(exist_ok=True)
            fad_ref_dir.mkdir(exist_ok=True)

        # Get samples
        src_batch, tgt_batch = next(iter(self.test_loader))
        
        # Only take as many samples as are available in the batch
        actual_samples = min(num_samples, src_batch.size(0))
        src_batch = src_batch[:actual_samples].to(self.device)
        tgt_batch = tgt_batch[:actual_samples].to(self.device)

        # Generate
        start_tokens = tgt_batch[:, :, 0]
        generated = self.model.generate(
            src_batch,
            max_len=max_len,
            start_tokens=start_tokens,
            temperature=0.9,
            top_k=50,
            top_p=0.9,
        )

        # Save audio for each sample - use actual number generated
        for i in range(generated.size(0)):  # ← Changed from range(num_samples)
            print(f"\nSample {i + 1}/{generated.size(0)}")

            # Generated audio
            gen_codes = remove_delay_interleaving(generated[i].cpu())
            try:
                gen_audio = detokenize(gen_codes)
                gen_path = output_dir / f"sample_{i}_generated.wav"
                torchaudio.save(str(gen_path), gen_audio, 32000)
                print(f"  Saved: sample_{i}_generated.wav")

                # Also save to FAD directory
                if save_for_fad:
                    fad_path = fad_gen_dir / f"{i:04d}.wav"
                    torchaudio.save(str(fad_path), gen_audio, 32000)

            except Exception as e:
                print(f"  Error saving generated audio: {e}")

            # Ground truth audio
            tgt_trimmed = tgt_batch[i, :, : generated.size(2)].cpu()
            tgt_codes = remove_delay_interleaving(tgt_trimmed)
            try:
                tgt_audio = detokenize(tgt_codes)
                tgt_path = output_dir / f"sample_{i}_ground_truth.wav"
                torchaudio.save(str(tgt_path), tgt_audio, 32000)
                print(f"  Saved: sample_{i}_ground_truth.wav")

                # Also save to FAD directory
                if save_for_fad:
                    fad_path = fad_ref_dir / f"{i:04d}.wav"
                    torchaudio.save(str(fad_path), tgt_audio, 32000)

            except Exception as e:
                print(f"  Error saving ground truth audio: {e}")

            # Source (backing track)
            src_codes = remove_delay_interleaving(src_batch[i].cpu())
            try:
                src_audio = detokenize(src_codes)
                src_path = output_dir / f"sample_{i}_source.wav"
                torchaudio.save(str(src_path), src_audio, 32000)
                print(f"  Saved: sample_{i}_source.wav")
            except Exception as e:
                print(f"  Error saving source audio: {e}")

        if save_for_fad:
            print("\nFAD directories created:")
            print(f"  Generated: {fad_gen_dir}")
            print(f"  Reference: {fad_ref_dir}")
            return fad_gen_dir, fad_ref_dir

        return None, None

    @torch.no_grad()
    def evaluate_fad(
        self,
        num_samples: int = 50,
        max_len: int = 1500,
        model_name: str = "vggish",
    ) -> Dict[str, float]:
        """
        Evaluate Frechet Audio Distance.

        Args:
            num_samples: Number of samples to generate for FAD
            max_len: Maximum generation length
            model_name: Embedding model to use ("vggish" recommended)

        Returns:
            Dictionary with FAD score
        """
        print("\n" + "=" * 80)
        print("Evaluating Frechet Audio Distance (FAD)")
        print(f"Generating {num_samples} samples...")
        print("=" * 80)

        # Create temporary directory for FAD audio
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="fad_eval_"))
        fad_gen_dir = temp_dir / "generated"
        fad_ref_dir = temp_dir / "reference"
        fad_gen_dir.mkdir()
        fad_ref_dir.mkdir()

        print(f"Temporary directories: {temp_dir}")

        resampler = torchaudio.transforms.Resample(
            orig_freq=SOURCE_SR, 
            new_freq=TARGET_SR
        ).to(self.device)

        # Generate samples in batches
        samples_generated = 0

        for batch_idx, (src_batch, tgt_batch) in enumerate(self.test_loader):
            if samples_generated >= num_samples:
                break

            # Only take as many samples as we need from this batch
            batch_samples = min(src_batch.size(0), num_samples - samples_generated)
            src_batch = src_batch[:batch_samples].to(self.device)
            tgt_batch = tgt_batch[:batch_samples].to(self.device)

            # Generate
            start_tokens = tgt_batch[:, :, 0]
            generated = self.model.generate(
                src_batch,
                max_len=max_len,
                start_tokens=start_tokens,
                temperature=0.9,
                top_k=50,
                top_p=0.9,
            )

            # Save audio files - iterate over actual batch size
            for i in range(generated.size(0)):
                idx = samples_generated + i

                # Generated audio
                gen_codes = remove_delay_interleaving(generated[i].cpu())
                try:
                    gen_audio = detokenize(gen_codes).to(self.device)
                    # Resample to 16kHz for VGGish
                    gen_audio_resampled = resampler(gen_audio).cpu()
                    gen_path = fad_gen_dir / f"{idx:04d}.wav"
                    torchaudio.save(str(gen_path), gen_audio_resampled, TARGET_SR)
                except Exception as e:
                    print(f"\nWarning: Failed to save generated sample {idx}: {e}")
                    continue

                # Reference audio
                tgt_trimmed = tgt_batch[i, :, : generated.size(2)].cpu()
                tgt_codes = remove_delay_interleaving(tgt_trimmed)
                try:
                    tgt_audio = detokenize(tgt_codes).to(self.device)
                    # Resample to 16kHz for VGGish
                    tgt_audio_resampled = resampler(tgt_audio).cpu()
                    ref_path = fad_ref_dir / f"{idx:04d}.wav"
                    torchaudio.save(str(ref_path), tgt_audio_resampled, TARGET_SR)
                except Exception as e:
                    print(f"\nWarning: Failed to save reference sample {idx}: {e}")
                    continue

            samples_generated += generated.size(0)
            print(f"Generated {samples_generated}/{num_samples} samples", end="\r")

        print(f"\nGenerated {samples_generated} samples total")

        # Calculate FAD
        try:
            print("\nCalculating FAD using frechet-audio-distance library...")
            frechet = FrechetAudioDistance(
                model_name=model_name,
                sample_rate=TARGET_SR,
                use_pca=False,
                use_activation=False,
                verbose=True,
            )
            fad_score = frechet.score(
                str(fad_gen_dir),
                str(fad_ref_dir),
                dtype="float32",
            )

            print(f"\nFAD Score: {fad_score:.4f}")

            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)

            return {
                "fad_score": float(fad_score),
                "fad_model": model_name,
                "fad_num_samples": samples_generated,
            }

        except Exception as e:
            print(f"\nError calculating FAD: {e}")
            import traceback
            traceback.print_exc()

            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)

            return {
                "fad_score": None,
                "error": str(e),
            }

    def run_full_evaluation(
        self,
        output_dir: Path,
        save_audio: bool = True,
        compute_fad: bool = True,
        fad_samples: int = 50,
    ) -> Dict[str, any]:
        """Run all evaluation metrics."""
        print("\n" + "=" * 80)
        print("STARTING FULL MODEL EVALUATION")
        print("=" * 80)
        print(f"Output directory: {output_dir}")
        print(f"Device: {self.device}")
        print(f"Test set size: {len(self.test_loader.dataset)}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. Loss and accuracy
        results.update(self.evaluate_loss_and_accuracy())

        # 2. Generation quality
        results.update(
            self.evaluate_generation_quality(
                num_samples=10,
                temperatures=[0.8, 1.0, 1.2],
            )
        )

        # 3. Autoregressive consistency
        results.update(self.evaluate_autoregressive_consistency(num_samples=5))

        # 4. Token distribution
        results.update(self.analyze_token_distribution())

        # 5. FAD (if enabled)
        if compute_fad:
            results.update(
                self.evaluate_fad(
                    num_samples=fad_samples,
                    model_name="vggish",
                )
            )

        # 6. Save audio samples
        if save_audio:
            self.save_audio_samples(
                output_dir / "audio_samples",
                num_samples=5,
                save_for_fad=False,  # FAD already generated its own
            )

        # Save results
        results["timestamp"] = datetime.now().isoformat()
        results["model_config"] = self.config["model"]

        results_file = output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {results_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Test Loss: {results['test_loss']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Test Perplexity: {results['test_perplexity']:.4f}")
        print(
            f"Autoregressive Consistency: {results['autoregressive_consistency']:.4f}"
        )
        if "fad_score" in results and results["fad_score"] is not None:
            print(f"FAD Score: {results['fad_score']:.4f}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate transformer model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoint_dir/evaluation)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio generation")
    parser.add_argument("--no-fad", action="store_true", help="Skip FAD computation")
    parser.add_argument(
        "--fad-samples",
        type=int,
        default=50,
        help="Number of samples for FAD computation",
    )
    parser.add_argument(
        "--fad-model",
        type=str,
        default="vggish",
        choices=["vggish", "pann"],
        help="Embedding model for FAD",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = create_model(config)
    checkpoint = load_checkpoint_for_inference(model, Path(args.checkpoint), device)

    # Load test data
    test_data_path = config["data"].get("test_path")
    print(f"Loading test data from {test_data_path}")
    test_dataset = torch.load(test_data_path, weights_only=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size or config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        checkpoint_dir = Path(args.checkpoint).parent
        output_dir = checkpoint_dir.parent / "evaluation"

    # Run evaluation
    evaluator = ModelEvaluator(model, test_loader, device, config)
    results = evaluator.run_full_evaluation(
        output_dir=output_dir,
        save_audio=not args.no_audio,
        compute_fad=not args.no_fad,
        fad_samples=args.fad_samples,
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
