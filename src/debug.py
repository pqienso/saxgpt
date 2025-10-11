"""
Debug script to compare training forward pass vs. generation behavior.
Run this to identify discrepancies between training and inference.

Usage:
    python debug_generation.py --config config/model/small.yaml --checkpoint models/small/checkpoints/best_model.pt
"""

import torch
import argparse
from pathlib import Path
import yaml

import sys
sys.path.append('src')

from src.training.utils import (
    load_config,
    create_model,
    load_dataset,
    setup_device,
    load_checkpoint_for_inference,
)


def detailed_generation_debug(
    model,
    src,
    tgt,
    device,
    max_steps=10,
):
    """
    Step through generation and print detailed information at each step.
    
    Args:
        model: The transformer model
        src: Source sequence [1, C, src_len]
        tgt: Target sequence [1, C, tgt_len] (ground truth)
        device: torch device
        max_steps: Number of generation steps to debug
    """
    model.eval()
    
    print("\n" + "="*80)
    print("DETAILED GENERATION DEBUG")
    print("="*80)
    
    batch_size = 1
    num_codebooks = model.num_codebooks
    
    # Get start tokens from ground truth
    start_tokens = tgt[:, :, 0]  # [1, C]
    
    print(f"\nStart tokens: {start_tokens[0]}")
    print(f"Ground truth sequence length: {tgt.size(2)}")
    print(f"\nGround truth first {max_steps} positions:")
    for pos in range(min(max_steps, tgt.size(2))):
        print(f"  Position {pos}: {tgt[0, :, pos].tolist()}")
    
    # Encode source
    memory = model.encode(src)
    print(f"\nMemory shape: {memory.shape}")
    
    # Initialize cache
    cache = [
        {"self_attn": None, "cross_attn": None}
        for _ in range(model.num_decoder_layers)
    ]
    
    # Start generation
    tgt_gen = start_tokens.unsqueeze(-1)  # [1, C, 1]
    
    print("\n" + "-"*80)
    print("GENERATION STEPS")
    print("-"*80)
    
    for step in range(min(max_steps - 1, tgt.size(2) - 1)):
        print(f"\n{'='*40}")
        print(f"STEP {step} (generating position {step + 1})")
        print(f"{'='*40}")
        
        current_tokens = tgt_gen[:, :, -1:]  # [1, C, 1]
        current_position = tgt_gen.size(2)
        
        print(f"Current sequence length: {current_position}")
        print(f"Current tokens (input to decoder): {current_tokens[0, :, 0].tolist()}")
        
        # Check PE offset
        pe_offset = 0
        if cache[0]["self_attn"] is not None:
            pe_offset = cache[0]["self_attn"].seq_len
        print(f"PE offset: {pe_offset}")
        
        # Decode with cache
        with torch.no_grad():
            logits, cache = model.decode(
                current_tokens,
                memory,
                cache=cache,
                return_cache=True,
                is_causal=True,
            )
        
        # logits: [1, C, 1, vocab_size-1]
        print(f"Logits shape: {logits.shape}")
        
        # Sample next tokens
        next_tokens = torch.zeros(batch_size, num_codebooks, dtype=torch.long, device=device)
        
        print(f"\nCodebook decisions:")
        for cb_idx in range(num_codebooks):
            if current_position <= cb_idx:
                # Still in delay period
                next_tokens[:, cb_idx] = model.padding_idx
                print(f"  CB{cb_idx}: PADDING (position {current_position} <= delay {cb_idx})")
            else:
                # Generate normally
                cb_logits = logits[0, cb_idx, 0, :]  # [vocab_size-1]
                
                # Get top-5 predictions
                top_probs, top_indices = torch.topk(torch.softmax(cb_logits, dim=0), k=5)
                
                predicted_token = cb_logits.argmax().item()
                next_tokens[:, cb_idx] = predicted_token
                
                ground_truth_token = tgt[0, cb_idx, step + 1].item()
                match = "✓" if predicted_token == ground_truth_token else "✗"
                
                print(f"  CB{cb_idx}: predicted={predicted_token}, ground_truth={ground_truth_token} {match}")
                print(f"         Top-5: {list(zip(top_indices.tolist(), [f'{p:.3f}' for p in top_probs.tolist()]))}")
        
        print(f"\nNext tokens: {next_tokens[0].tolist()}")
        print(f"Ground truth: {tgt[0, :, step + 1].tolist()}")
        
        # Append to sequence
        tgt_gen = torch.cat([tgt_gen, next_tokens.unsqueeze(-1)], dim=-1)
        
        # Check cache state
        if cache[0]["self_attn"] is not None:
            print(f"Cache self_attn seq_len: {cache[0]['self_attn'].seq_len}")
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    max_compare = min(max_steps, tgt.size(2))
    print(f"\nGenerated sequence (first {max_compare} positions):")
    for pos in range(max_compare):
        print(f"  Position {pos}: {tgt_gen[0, :, pos].tolist()}")
    
    print(f"\nGround truth sequence (first {max_compare} positions):")
    for pos in range(max_compare):
        print(f"  Position {pos}: {tgt[0, :, pos].tolist()}")
    
    # Calculate per-position accuracy
    matches = (tgt_gen[:, :, :max_compare] == tgt[:, :, :max_compare])
    per_position_acc = matches.float().mean(dim=1)[0]  # Average over codebooks
    
    print(f"\nPer-position accuracy:")
    for pos in range(max_compare):
        acc = matches[0, :, pos].float().mean().item()
        print(f"  Position {pos}: {acc*100:.1f}%")
    
    overall_acc = matches.float().mean().item()
    print(f"\nOverall accuracy: {overall_acc*100:.1f}%")
    
    return tgt_gen


def compare_training_vs_generation(model, src, tgt, device):
    """
    Compare training forward pass vs. generation output.
    
    Args:
        model: The transformer model
        src: Source sequence [1, C, src_len]
        tgt: Target sequence [1, C, tgt_len]
        device: torch device
    """
    model.eval()
    
    print("\n" + "="*80)
    print("TRAINING FORWARD PASS")
    print("="*80)
    
    with torch.no_grad():
        # Training-style forward (teacher forcing)
        logits_train = model(src, tgt[:, :, :-1])
        preds_train = logits_train.argmax(dim=-1)  # [1, C, T]
    
    print(f"Logits shape: {logits_train.shape}")
    print(f"Predictions shape: {preds_train.shape}")
    
    # Calculate accuracy
    targets = tgt[:, :, 1:]
    mask = targets != model.padding_idx
    correct = (preds_train == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    print(f"Training forward pass accuracy: {accuracy.item()*100:.2f}%")
    
    # Show first 10 positions
    print(f"\nFirst 10 predictions (teacher forcing):")
    for pos in range(min(10, preds_train.size(2))):
        pred = preds_train[0, :, pos].tolist()
        truth = targets[0, :, pos].tolist()
        match = "✓" if pred == truth else "✗"
        print(f"  Position {pos}: pred={pred}, truth={truth} {match}")
    
    print("\n" + "="*80)
    print("GREEDY GENERATION")
    print("="*80)
    
    start_tokens = tgt[:, :, 0]
    max_len = tgt.size(2)
    
    with torch.no_grad():
        generated = model.generate_greedy(
            src,
            max_len=max_len,
            start_tokens=start_tokens,
        )
    
    print(f"Generated shape: {generated.shape}")
    
    # Calculate accuracy
    mask = tgt != model.padding_idx
    correct = (generated == tgt) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    print(f"Generation accuracy: {accuracy.item()*100:.2f}%")
    
    # Show first 10 positions
    print(f"\nFirst 10 generated positions:")
    for pos in range(min(10, generated.size(2))):
        gen = generated[0, :, pos].tolist()
        truth = tgt[0, :, pos].tolist()
        match = "✓" if gen == truth else "✗"
        print(f"  Position {pos}: gen={gen}, truth={truth} {match}")
    
    # Find first mismatch
    print("\n" + "="*80)
    print("FIRST MISMATCHES")
    print("="*80)
    
    for cb_idx in range(model.num_codebooks):
        mismatches = (generated[0, cb_idx, :] != tgt[0, cb_idx, :])
        # Ignore padding positions
        valid_positions = (tgt[0, cb_idx, :] != model.padding_idx)
        mismatches = mismatches & valid_positions
        
        if mismatches.any():
            first_mismatch = mismatches.nonzero()[0].item()
            print(f"Codebook {cb_idx}: First mismatch at position {first_mismatch}")
            print(f"  Generated: {generated[0, cb_idx, first_mismatch].item()}")
            print(f"  Ground truth: {tgt[0, cb_idx, first_mismatch].item()}")
            
            # Show context around mismatch
            start = max(0, first_mismatch - 2)
            end = min(tgt.size(2), first_mismatch + 3)
            print(f"  Context (positions {start}-{end}):")
            print(f"    Generated:    {generated[0, cb_idx, start:end].tolist()}")
            print(f"    Ground truth: {tgt[0, cb_idx, start:end].tolist()}")
        else:
            print(f"Codebook {cb_idx}: Perfect match! ✓")


def main():
    parser = argparse.ArgumentParser(description="Debug generation vs training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num-examples", type=int, default=3, help="Number of training examples to test")
    parser.add_argument("--max-debug-steps", type=int, default=15, help="Number of steps to debug in detail")
    
    args = parser.parse_args()
    
    # Load config and setup
    config = load_config(args.config)
    device = setup_device()
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint_for_inference(model, Path(args.checkpoint), device)
    model.eval()
    
    # Load training dataset
    train_dataset = load_dataset(config["data"]["train_path"], "training data")
    
    print("\n" + "="*80)
    print(f"Testing on {args.num_examples} training examples")
    print("="*80)
    
    for idx in range(min(args.num_examples, len(train_dataset))):
        print(f"\n\n{'#'*80}")
        print(f"EXAMPLE {idx + 1}")
        print(f"{'#'*80}")
        
        # Get example
        src, tgt = train_dataset[idx]
        src = src.unsqueeze(0).to(device)
        tgt = tgt.unsqueeze(0).to(device)
        
        print(f"\nSource shape: {src.shape}")
        print(f"Target shape: {tgt.shape}")
        
        # Quick comparison
        compare_training_vs_generation(model, src, tgt, device)
        
        # Detailed step-by-step debug for first example
        if idx == 0:
            detailed_generation_debug(model, src, tgt, device, max_steps=args.max_debug_steps)
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()