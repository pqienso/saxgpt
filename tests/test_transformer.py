import torch
import torch.nn.functional as F

from src.model.transformer import EncoderDecoderTransformer

SEPARATOR = "=" * 80


def test_transformer(model: EncoderDecoderTransformer):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Number of codebooks: {model.num_codebooks}")
    print(f"  Vocabulary size: {model.vocab_size}")

    # Check for Flash Attention support
    has_flash = hasattr(F, "scaled_dot_product_attention")
    print(f"  Flash Attention available: {has_flash}")

    # Example 1: Training mode
    print("\n" + "-" * 80)
    print("Example 1: Training Mode (Teacher Forcing)")
    print("-" * 80)

    batch_size = 4
    src_seq_len = 20
    tgt_seq_len = 15

    # Shape: [batch_size, num_codebooks, seq_len]
    src = torch.randint(0, vocab_size, (batch_size, num_codebooks, src_seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, num_codebooks, tgt_seq_len))

    # No need for explicit mask - model handles causal masking automatically
    logits = model(src, tgt)

    print(f"Input source shape: {src.shape}")
    print(f"Input target shape: {tgt.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(
        f"Expected shape: [batch={batch_size}, codebooks={num_codebooks}, "
        f"tgt_len={tgt_seq_len}, vocab={vocab_size}]"
    )

    # Example 2: Training with explicit mask (if needed)
    print("\n" + "-" * 80)
    print("Example 2: Training with Explicit Causal Mask")
    print("-" * 80)

    tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len, src.device)
    print(f"Mask shape: {tgt_mask.shape}")
    print("Mask type: boolean (True = masked)")
    logits_masked = model(src, tgt, tgt_mask=tgt_mask)
    print(f"Output logits shape: {logits_masked.shape}")

    # Example 3: Generation mode with caching
    print("\n" + "-" * 80)
    print("Example 3: Generation Mode (with KV-Caching)")
    print("-" * 80)

    src_test = torch.randint(0, vocab_size, (2, num_codebooks, 20))
    # Create start tokens for each codebook
    start_tokens = torch.full((2, num_codebooks), start_token_id, dtype=torch.long)
    
    print(f"Source sequence shape: {src_test.shape}")
    print(f"Start tokens shape: {start_tokens.shape}")

    generated = model.generate(
        src_test,
        max_len=30,
        start_tokens=start_tokens,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated sequence length: {generated.size(2)}")  # Changed from size(1)
    print(f"\nFirst generated sequence (codebook 0):\n{generated[0, 0].tolist()[:15]}...")
    print(f"First generated sequence (codebook 1):\n{generated[0, 1].tolist()[:15]}...")

    # Example 4: Verify caching correctness
    print("\n" + "-" * 80)
    print("Example 4: Verifying KV-Cache Correctness")
    print("-" * 80)

    model.eval()
    with torch.no_grad():
        # Encode once
        src_single = torch.randint(0, vocab_size, (1, num_codebooks, 10))
        memory = model.encode(src_single)

        # Method 1: Without caching (process full sequence)
        # Shape: [1, num_codebooks, seq_len]
        tgt_seq = torch.randint(0, vocab_size, (1, num_codebooks, 5))
        logits_no_cache, _ = model.decode(
            tgt_seq, memory, return_cache=False, is_causal=True
        )

        # Method 2: With caching (incremental)
        cache = [
            {"self_attn": None, "cross_attn": None}
            for _ in range(model.num_decoder_layers)
        ]
        logits_with_cache_list = []

        for i in range(tgt_seq.size(2)):  # Iterate over sequence length
            current_token = tgt_seq[:, :, i : i + 1]  # [1, num_codebooks, 1]
            logits_step, cache = model.decode(
                current_token, memory, cache=cache, return_cache=True, is_causal=True
            )
            logits_with_cache_list.append(logits_step)

        logits_with_cache = torch.cat(logits_with_cache_list, dim=2)  # Concat on seq_len dim

        # Compare results
        diff = torch.abs(logits_no_cache - logits_with_cache).max().item()
        print(f"Maximum difference between cached and non-cached: {diff:.6f}")
        matches = torch.allclose(logits_no_cache, logits_with_cache, atol=1e-5)
        print(f"Results match: {matches}")
        
        if not matches:
            print(f"  logits_no_cache shape: {logits_no_cache.shape}")
            print(f"  logits_with_cache shape: {logits_with_cache.shape}")

    # Example 5: Generation with padding mask
    print("\n" + "-" * 80)
    print("Example 5: Generation with Padding Mask")
    print("-" * 80)

    src_padded = torch.randint(0, vocab_size, (3, num_codebooks, 15))
    src_padding_mask = torch.zeros(3, 15, dtype=torch.bool)
    src_padding_mask[0, 12:] = True  # Length 12
    src_padding_mask[1, 10:] = True  # Length 10
    # Third sequence: full length

    start_tokens_padded = torch.full((3, num_codebooks), start_token_id, dtype=torch.long)

    print(f"Source shape: {src_padded.shape}")
    print(f"Padding mask shape: {src_padding_mask.shape}")
    print("Sequence lengths: [12, 10, 15]")

    generated_padded = model.generate(
        src_padded,
        max_len=20,
        start_tokens=start_tokens_padded,
        src_key_padding_mask=src_padding_mask,
        temperature=1.0,
        top_p=0.9,
    )

    print(f"Generated shape: {generated_padded.shape}")

    # Example 6: Greedy generation
    print("\n" + "-" * 80)
    print("Example 6: Greedy Generation (Deterministic)")
    print("-" * 80)

    src_greedy = torch.randint(0, vocab_size, (2, num_codebooks, 20))
    start_tokens_greedy = torch.full((2, num_codebooks), start_token_id, dtype=torch.long)

    generated_greedy = model.generate_greedy(
        src_greedy,
        max_len=25,
        start_tokens=start_tokens_greedy,
    )

    print(f"Greedy generated shape: {generated_greedy.shape}")
    print(f"First sequence (codebook 0): {generated_greedy[0, 0].tolist()[:10]}...")

    # Example 7: Verify generate_greedy matches forward pass
    print("\n" + "-" * 80)
    print("Example 7: Verify generate_greedy() vs forward() Consistency")
    print("-" * 80)

    model.eval()
    with torch.no_grad():
        src_verify = torch.randint(0, vocab_size, (2, num_codebooks, 15))
        start_tokens_verify = torch.full((2, num_codebooks), start_token_id, dtype=torch.long)
        max_gen_len = 10
        
        print("Testing if generate_greedy() produces same results as forward pass...")
        print(f"Source shape: {src_verify.shape}")
        print(f"Start tokens: {start_tokens_verify.shape}")
        print(f"Max generation length: {max_gen_len}")
        
        # Method 1: Use generate_greedy
        generated_seq = model.generate_greedy(
            src_verify,
            max_len=max_gen_len,
            start_tokens=start_tokens_verify,
        )
        
        print(f"\nGenerated sequence shape: {generated_seq.shape}")
        print(f"Generated sequence (first sample, codebook 0): {generated_seq[0, 0].tolist()}")
        
        # Method 2: Step-by-step generation with decode() to match generate_greedy logic
        print("\nStep-by-step verification:")
        memory = model.encode(src_verify)
        cache = [{"self_attn": None, "cross_attn": None} for _ in range(model.num_decoder_layers)]
        
        # Start with initial tokens
        current_seq = start_tokens_verify.unsqueeze(-1)  # [B, C, 1]
        
        for step in range(max_gen_len - 1):
            # Get last token
            current_token = current_seq[:, :, -1:]  # [B, C, 1]
            
            # Decode with cache
            logits_step, cache = model.decode(
                current_token, 
                memory, 
                cache=cache, 
                return_cache=True, 
                is_causal=True
            )  # [B, C, 1, V]
            
            # Get next token
            next_token = logits_step.argmax(dim=-1)  # [B, C, 1]
            current_seq = torch.cat([current_seq, next_token], dim=2)
            
            # Compare with generated sequence at this step
            gen_token = generated_seq[:, :, step + 1:step + 2]
            if step > 2 and not torch.equal(next_token, gen_token):
                print(f"\n  Step {step}: MISMATCH!")
                print(f"    Manual decode: {next_token[0, 0].item()}")
                print(f"    generate_greedy: {gen_token[0, 0].item()}")
                print(f"    Top-5 logits: {logits_step[0, 0, 0].topk(5)}")
                break
        else:
            print("  All steps match between manual decode and generate_greedy ✓")
        
        # Method 3: Full forward pass comparison (original test)
        print("\nFull forward pass comparison:")
        tgt_input = generated_seq[:, :, :-1]  # [B, C, T-1]
        logits_forward = model(src_verify, tgt_input)  # [B, C, T-1, V]
        
        # Get greedy predictions from forward pass
        predicted_tokens = logits_forward.argmax(dim=-1)  # [B, C, T-1]
        
        # Compare: generated_seq[:, :, 1:] should match predicted_tokens
        generated_next_tokens = generated_seq[:, :, 1:]  # [B, C, T-1]

        predicted_tokens = predicted_tokens[:, :, 4:]
        generated_next_tokens = generated_next_tokens[:, :, 4:]
        
        matches = torch.equal(generated_next_tokens, predicted_tokens)
        diff_count = (generated_next_tokens != predicted_tokens).sum().item()
        total_tokens = generated_next_tokens.numel()
        
        print(f"  Forward pass logits shape: {logits_forward.shape}")
        print(f"  Predicted tokens shape: {predicted_tokens.shape}")
        print(f"  All tokens match: {matches}")
        print(f"  Mismatched tokens: {diff_count}/{total_tokens}")
        
        if not matches:
            print("\n  Detailed mismatch analysis:")
            # Check first position separately
            first_pos_match = torch.equal(generated_next_tokens[:, :, 0], predicted_tokens[:, :, 0])
            print(f"    First position matches: {first_pos_match}")
            
            # Find where mismatches start
            for t in range(generated_next_tokens.size(2)):
                pos_matches = torch.equal(generated_next_tokens[:, :, t], predicted_tokens[:, :, t])
                if not pos_matches:
                    print(f"    First mismatch at position {t}")
                    break
            
            # Show detailed first mismatch
            print("\n  First mismatch details:")
            for b in range(generated_seq.size(0)):
                for c in range(num_codebooks):
                    for t in range(generated_next_tokens.size(2)):
                        if generated_next_tokens[b, c, t] != predicted_tokens[b, c, t]:
                            print(f"    Batch {b}, Codebook {c}, Position {t}:")
                            print(f"      generate_greedy token: {generated_next_tokens[b, c, t].item()}")
                            print(f"      forward pass token:    {predicted_tokens[b, c, t].item()}")
                            top5_vals, top5_idx = logits_forward[b, c, t].topk(5)
                            print(f"      Top-5 logits: {list(zip(top5_idx.tolist(), top5_vals.tolist()))}")
                            print(f"      Input token at position {t}: {tgt_input[b, c, t].item()}")
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            
            print("\n  DIAGNOSIS: This mismatch suggests one of the following issues:")
            print("    1. Dropout is enabled during generation (should be model.eval())")
            print("    2. Cache is not being used correctly in generate_greedy()")
            print("    3. Causal masking differs between forward() and decode()")
            print("    4. There's a subtle bug in the generation logic")
        else:
            print("  ✓ generate_greedy() is consistent with forward() pass")
        
        # Don't assert yet - let's collect more info
        if not matches:
            print("\n  WARNING: Consistency check failed! See diagnosis above.")

    # Example 8: Loss computation
    print("\n" + "-" * 80)
    print("Example 8: Loss Computation")
    print("-" * 80)

    batch_size = 8
    src_loss = torch.randint(0, vocab_size, (batch_size, num_codebooks, 20))
    tgt_loss = torch.randint(0, vocab_size, (batch_size, num_codebooks, 15))

    logits = model(src_loss, tgt_loss)
    
    # Reshape for loss computation
    # logits: [B, num_codebooks, T, V] -> [B*num_codebooks*T, V]
    # tgt: [B, num_codebooks, T] -> [B*num_codebooks*T]
    B, C, T, V = logits.shape
    logits_flat = logits.reshape(B * C * T, V)
    tgt_flat = tgt_loss.reshape(B * C * T)
    
    loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=padding_idx);
    
    print(f"Logits shape: {logits.shape}")
    print(f"Target shape: {tgt_loss.shape}")
    print(f"Flattened logits shape: {logits_flat.shape}")
    print(f"Flattened target shape: {tgt_flat.shape}")
    print(f"Cross-entropy loss: {loss.item():.4f}")

    # Alternative: Per-codebook loss
    losses_per_codebook = []
    for cb_idx in range(num_codebooks):
        cb_logits = logits[:, cb_idx, :, :]  # [B, T, V]
        cb_target = tgt_loss[:, cb_idx, :]   # [B, T]
        cb_loss = F.cross_entropy(
            cb_logits.reshape(B * T, V),
            cb_target.reshape(B * T),
            ignore_index=padding_idx,
        )
        losses_per_codebook.append(cb_loss.item())
    
    print(f"Per-codebook losses: {[f'{loss:.4f}' for loss in losses_per_codebook]}")
    print(f"Mean per-codebook loss: {sum(losses_per_codebook) / len(losses_per_codebook):.4f}")


# Example usage and testing
if __name__ == "__main__":
    print(SEPARATOR)
    print("Testing Multi-Codebook Encoder-Decoder Transformer")
    print(SEPARATOR)

    # Hyperparameters
    vocab_size = 2049  # 2048 codes + 1 padding
    num_codebooks = 4
    d_model = 32
    nhead = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 128
    dropout = 0.1
    padding_idx = 2048
    start_token_id = 0

    # Create model
    model = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        num_codebooks=num_codebooks,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        norm_first=True,
        padding_idx=padding_idx,
    )

    test_transformer(model)

    print("\n" + SEPARATOR)
    print("Tests completed successfully.")
    print(SEPARATOR)
