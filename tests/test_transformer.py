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

    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    # No need for explicit mask - model handles causal masking automatically
    logits = model(src, tgt)

    print(f"Input source shape: {src.shape}")
    print(f"Input target shape: {tgt.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(
        f"Expected shape: [batch={batch_size}, tgt_len={tgt_seq_len}, vocab={tgt_vocab_size}]"
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

    src_test = torch.randint(0, src_vocab_size, (2, 20))
    print(f"Source sequence shape: {src_test.shape}")

    generated = model.generate(
        src_test,
        max_len=30,
        start_token=1,
        end_token=2,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated sequence length: {generated.size(1)}")
    print(f"\nFirst generated sequence:\n{generated[0].tolist()[:15]}...")

    # Example 4: Verify caching correctness
    print("\n" + "-" * 80)
    print("Example 4: Verifying KV-Cache Correctness")
    print("-" * 80)

    model.eval()
    with torch.no_grad():
        # Encode once
        src_single = torch.randint(0, src_vocab_size, (1, 10))
        memory = model.encode(src_single)

        # Method 1: Without caching (process full sequence)
        tgt_seq = torch.tensor([[1, 100, 200, 300, 400]], dtype=torch.long)
        logits_no_cache, _ = model.decode(
            tgt_seq, memory, use_cache=False, is_causal=True
        )

        # Method 2: With caching (incremental)
        cache = [
            {"self_attn": None, "cross_attn": None}
            for _ in range(model.num_decoder_layers)
        ]
        logits_with_cache_list = []

        for i in range(tgt_seq.size(1)):
            current_token = tgt_seq[:, i : i + 1]
            logits_step, cache = model.decode(
                current_token, memory, cache=cache, use_cache=True, is_causal=True
            )
            logits_with_cache_list.append(logits_step)

        logits_with_cache = torch.cat(logits_with_cache_list, dim=1)

        # Compare results
        diff = torch.abs(logits_no_cache - logits_with_cache).max().item()
        print(f"Maximum difference between cached and non-cached: {diff:.6f}")
        print(
            f"Results match: {torch.isclose(logits_no_cache, logits_with_cache, atol=1e-6).all()}"
        )

    # Example 5: Generation with padding mask
    print("\n" + "-" * 80)
    print("Example 5: Generation with Padding Mask")
    print("-" * 80)

    src_padded = torch.randint(0, src_vocab_size, (3, 15))
    src_padding_mask = torch.zeros(3, 15, dtype=torch.bool)
    src_padding_mask[0, 12:] = True  # Length 12
    src_padding_mask[1, 10:] = True  # Length 10
    # Third sequence: full length

    print(f"Source shape: {src_padded.shape}")
    print(f"Padding mask shape: {src_padding_mask.shape}")
    print("Sequence lengths: [12, 10, 15]")

    generated_padded = model.generate(
        src_padded,
        max_len=20,
        start_token=1,
        end_token=2,
        src_key_padding_mask=src_padding_mask,
        temperature=1.0,
        top_p=0.9,
    )

    print(f"Generated shape: {generated_padded.shape}")


# Example usage and testing
if __name__ == "__main__":
    print(SEPARATOR)
    print("Testing Encoder-Decoder Transformer")
    print(SEPARATOR)

    # Hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

    # Create model
    model = EncoderDecoderTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        norm_first=False,  # Post-norm (traditional)
    )

    test_transformer(model)

    print("\n" + SEPARATOR)
    print("Tests completed successfully.")
    print(SEPARATOR)
