import torch
import torch.nn.functional as F

from src.model.attention import MultiHeadSelfAttentionWithCache

SEPARATOR = "=" * 80


def test_flash_attention_usage():
    """Test to verify Flash Attention is actually being used."""
    print("\n" + SEPARATOR)
    print("Testing Flash Attention Backend Detection")
    print(SEPARATOR)

    # Check what's available
    has_sdpa = hasattr(F, "scaled_dot_product_attention")
    print(f"\nF.scaled_dot_product_attention available: {has_sdpa}")

    if has_sdpa and torch.cuda.is_available():
        # Check CUDA backends
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            ):
                print("\nSDPA backends available:")
                print(f"  Flash Attention: {torch.backends.cuda.flash_sdp_enabled()}")
                print(
                    f"  Memory Efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}"
                )
                print(f"  Math (slowest): {torch.backends.cuda.math_sdp_enabled()}")
        except Exception:
            print("\nCould not query CUDA backends (CPU mode or old PyTorch version)")

    # Test with actual model
    print("\n" + "-" * 80)
    print("Testing attention with different mask configurations")
    print("-" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    attn = MultiHeadSelfAttentionWithCache(
        d_model=128, nhead=8, dropout=0.0, use_flash=True
    ).to(device)

    x = torch.randn(2, 10, 128, device=device)

    # Test 1: No masks (should use Flash Attention)
    print("\n1. No masks:")
    with torch.no_grad():
        out1, _ = attn(x)
    print(f"   Output shape: {out1.shape}")

    # Test 2: With padding mask (should still use Flash Attention)
    print("\n2. With padding mask:")
    padding_mask = torch.zeros(2, 10, dtype=torch.bool, device=device)
    padding_mask[0, 8:] = True
    padding_mask[1, 6:] = True
    with torch.no_grad():
        out2, _ = attn(x, key_padding_mask=padding_mask)
    print(f"   Output shape: {out2.shape}")

    # Test 3: With causal mask
    print("\n3. With causal masking:")
    with torch.no_grad():
        out3, _ = attn(x, is_causal=True)
    print(f"   Output shape: {out3.shape}")

    # Test 4: With custom attention mask
    print("\n4. With custom attention mask:")
    attn_mask = torch.zeros(10, 10, dtype=torch.bool, device=device)
    attn_mask[:, 5:] = True  # Mask second half
    with torch.no_grad():
        out4, _ = attn(x, attn_mask=attn_mask)
    print(f"   Output shape: {out4.shape} ✓")

    # Test 5: Combined masks
    print("\n5. With combined padding + causal:")
    with torch.no_grad():
        out5, _ = attn(x, key_padding_mask=padding_mask, is_causal=True)
    print(f"   Output shape: {out5.shape}")

    print("\n" + SEPARATOR)
    print("Flash Attention tests complete.")
    print(SEPARATOR)


# Example usage and testing
if __name__ == "__main__":
    print(SEPARATOR)
    print("Testing Flash Attention")
    print(SEPARATOR)

    test_flash_attention_usage()

    print("\n" + SEPARATOR)
    print("Tests completed successfully.")
    print(SEPARATOR)
