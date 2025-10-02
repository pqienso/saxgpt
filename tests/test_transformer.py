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
    logits_masked = model(src, tgt, tgt_mask=tgt_mask)

    print(f"Mask shape: {tgt_mask.shape}")
    print("Mask type: boolean (True = masked)")
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
        end_token=end_token_id,
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
            tgt_seq, memory, use_cache=False, is_causal=True
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
                current_token, memory, cache=cache, use_cache=True, is_causal=True
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
        end_token=end_token_id,
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
        end_token=end_token_id,
    )

    print(f"Greedy generated shape: {generated_greedy.shape}")
    print(f"First sequence (codebook 0): {generated_greedy[0, 0].tolist()[:10]}...")

    # Example 7: Loss computation
    print("\n" + "-" * 80)
    print("Example 7: Loss Computation")
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
    
    loss = F.cross_entropy(logits_flat, tgt_flat)
    
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
            cb_target.reshape(B * T)
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
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    padding_idx = 2048
    start_token_id = 0
    end_token_id = 1

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
        norm_first=False,  # Post-norm (traditional)
        padding_idx=padding_idx,
    )

    test_transformer(model)

    print("\n" + SEPARATOR)
    print("Tests completed successfully.")
    print(SEPARATOR)
