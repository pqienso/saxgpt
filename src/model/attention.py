import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Structured cache entry for attention layers."""

    k: torch.Tensor  # [batch, nhead, seq, d_k]
    v: torch.Tensor  # [batch, nhead, seq, d_k]
    seq_len: int  # Track sequence length for PE offset


class MultiHeadSelfAttentionWithCache(nn.Module):
    """Multi-head self-attention with batched QKV projections and efficient KV-caching."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        bias: bool = False,
        use_flash: bool = True,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        # Batched projection for Q, K, V together
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.scale = math.sqrt(self.d_k)

    def _prepare_flash_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        kv_len: int,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Convert masks to Flash Attention format (additive float mask).

        Returns:
            Combined mask [batch, 1, seq_len, kv_len] or None
        """
        combined_mask = None

        # Handle key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [batch, kv_len], True = padding
            # Convert to [batch, 1, 1, kv_len]
            combined_mask = torch.zeros(
                (batch_size, 1, 1, kv_len), dtype=dtype, device=device
            )
            combined_mask.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Handle attention mask
        if attn_mask is not None:
            attn_mask_float = torch.zeros(
                (1, 1, seq_len, kv_len), dtype=dtype, device=device
            )

            if attn_mask.dim() == 2:
                # [seq_len, kv_len] -> [1, 1, seq_len, kv_len]
                attn_mask_float.masked_fill_(
                    attn_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )
            elif attn_mask.dim() == 3:
                # [batch, seq_len, kv_len] -> [batch, 1, seq_len, kv_len]
                attn_mask_float = torch.zeros(
                    (batch_size, 1, seq_len, kv_len), dtype=dtype, device=device
                )
                attn_mask_float.masked_fill_(attn_mask.unsqueeze(1), float("-inf"))

            # Combine with key padding mask
            if combined_mask is None:
                combined_mask = attn_mask_float
            else:
                combined_mask = combined_mask + attn_mask_float

        return combined_mask

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[CacheEntry] = None,
        use_cache: bool = False,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[CacheEntry]]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            attn_mask: Boolean mask [seq_len, seq_len] or [batch, seq_len, seq_len]
                      True indicates positions to mask (set to -inf)
            key_padding_mask: [batch_size, kv_len] - True for padding positions
            cache: CacheEntry with cached K, V tensors
            use_cache: Whether to return cache for next step
            is_causal: Whether to apply causal masking (for autoregressive)

        Returns:
            output: [batch_size, seq_len, d_model]
            new_cache: CacheEntry with updated K, V tensors (if use_cache=True)
        """
        batch_size, seq_len, _ = x.shape

        # Batched linear projection: [batch, seq_len, d_model] -> [batch, seq_len, 3*d_model]
        qkv = self.w_qkv(x)

        # Split into Q, K, V: each [batch, seq_len, d_model]
        Q, K, V = qkv.chunk(3, dim=-1)

        # Reshape to multi-head format: [batch, nhead, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # Handle caching for K and V
        if cache is not None:
            # Concatenate cached K, V with new K, V
            K = torch.cat([cache.k, K], dim=2)  # [batch, nhead, cached+new, d_k]
            V = torch.cat([cache.v, V], dim=2)

        kv_len = K.size(2)

        can_use_flash = self.use_flash
        # Special case: is_causal can only be used without cache and without custom masks
        use_is_causal = is_causal and cache is None and attn_mask is None

        if can_use_flash:
            # Prepare mask in Flash Attention format (additive)
            if use_is_causal:
                # Let Flash Attention handle causal masking efficiently
                flash_mask = None
                if key_padding_mask is not None:
                    flash_mask = self._prepare_flash_attention_mask(
                        batch_size,
                        seq_len,
                        kv_len,
                        key_padding_mask,
                        None,
                        x.device,
                        Q.dtype,
                    )

                context = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=flash_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                # Handle all masks manually (but still use Flash Attention!)
                flash_mask = self._prepare_flash_attention_mask(
                    batch_size,
                    seq_len,
                    kv_len,
                    key_padding_mask,
                    attn_mask,
                    x.device,
                    Q.dtype,
                )

                # Add causal mask if needed (when we can't use is_causal flag)
                if is_causal and cache is None and flash_mask is None:
                    # Create causal mask
                    flash_mask = torch.zeros(
                        (1, 1, seq_len, kv_len), dtype=Q.dtype, device=x.device
                    )
                    causal_mask = torch.triu(
                        torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    flash_mask.masked_fill_(causal_mask, float("-inf"))
                elif is_causal and cache is None and flash_mask is not None:
                    # Add causal mask to existing mask
                    causal_mask = torch.triu(
                        torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    flash_mask = flash_mask.masked_fill(causal_mask, float("-inf"))

                context = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=flash_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,  # We handle causal in the mask
                )
        else:
            # Manual attention computation (fallback)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            # scores: [batch, nhead, seq_len, kv_len]

            # Apply attention mask (boolean mask: True = mask out)
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    # [seq_len, kv_len] -> [1, 1, seq_len, kv_len]
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    # [batch, seq_len, kv_len] -> [batch, 1, seq_len, kv_len]
                    attn_mask = attn_mask.unsqueeze(1)
                scores = scores.masked_fill(attn_mask, float("-inf"))

            # Apply causal mask if needed
            if is_causal and cache is None:
                causal_mask = torch.triu(
                    torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(causal_mask, float("-inf"))

            # Apply key padding mask
            if key_padding_mask is not None:
                # [batch, kv_len] -> [batch, 1, 1, kv_len]
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(key_padding_mask, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            context = torch.matmul(attn_weights, V)

        # context: [batch, nhead, seq_len, d_k]
        # Reshape back: [batch, seq_len, d_model]
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        # Final linear projection
        output = self.w_o(context)

        # Prepare cache for next step (store in head-split format)
        new_cache = None
        if use_cache:
            new_cache = CacheEntry(
                k=K.contiguous(),  # [batch, nhead, kv_len, d_k]
                v=V.contiguous(),
                seq_len=kv_len,
            )

        return output, new_cache


class MultiHeadCrossAttentionWithCache(nn.Module):
    """Multi-head cross-attention with separate Q and KV projections and efficient KV-caching."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        bias: bool = False,
        use_flash: bool = True,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        # Separate projections: Q from decoder, K and V from encoder
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_kv = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.scale = math.sqrt(self.d_k)

    def _prepare_flash_attention_mask(
        self,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Convert masks to Flash Attention format (additive float mask).

        Returns:
            Combined mask [batch, 1, tgt_len, src_len] or None
        """
        combined_mask = None

        # Handle key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [batch, src_len], True = padding
            # Convert to [batch, 1, 1, src_len]
            combined_mask = torch.zeros(
                (batch_size, 1, 1, src_len), dtype=dtype, device=device
            )
            combined_mask.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Handle attention mask
        if attn_mask is not None:
            attn_mask_float = torch.zeros(
                (1, 1, tgt_len, src_len), dtype=dtype, device=device
            )

            if attn_mask.dim() == 2:
                # [tgt_len, src_len] -> [1, 1, tgt_len, src_len]
                attn_mask_float.masked_fill_(
                    attn_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )
            elif attn_mask.dim() == 3:
                # [batch, tgt_len, src_len] -> [batch, 1, tgt_len, src_len]
                attn_mask_float = torch.zeros(
                    (batch_size, 1, tgt_len, src_len), dtype=dtype, device=device
                )
                attn_mask_float.masked_fill_(attn_mask.unsqueeze(1), float("-inf"))

            # Combine with key padding mask
            if combined_mask is None:
                combined_mask = attn_mask_float
            else:
                combined_mask = combined_mask + attn_mask_float

        return combined_mask

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[CacheEntry] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[CacheEntry]]:
        """
        Args:
            query: [batch_size, tgt_seq_len, d_model] - from decoder
            key_value: [batch_size, src_seq_len, d_model] - from encoder
            attn_mask: Boolean mask
            key_padding_mask: [batch_size, src_seq_len] - True for padding positions
            cache: CacheEntry with cached K, V from encoder
            use_cache: Whether to return cache

        Returns:
            output: [batch_size, tgt_seq_len, d_model]
            new_cache: CacheEntry with K, V tensors (if use_cache=True)
        """
        batch_size, tgt_len, _ = query.shape

        Q = self.w_q(query)  # [batch, tgt_len, d_model]

        if cache is not None:
            # Use cached K, V (encoder output doesn't change during decoding)
            K = cache.k
            V = cache.v
            src_len = K.size(2)
        else:
            kv = self.w_kv(key_value)  # [batch, src_len, 2*d_model]
            K, V = kv.chunk(2, dim=-1)  # Each: [batch, src_len, d_model]
            src_len = K.size(1)

            # Reshape to multi-head format
            K = K.view(batch_size, src_len, self.nhead, self.d_k).transpose(1, 2)
            V = V.view(batch_size, src_len, self.nhead, self.d_k).transpose(1, 2)

        # Reshape Q
        Q = Q.view(batch_size, tgt_len, self.nhead, self.d_k).transpose(1, 2)

        # ===================================================================
        # NEW: Improved Flash Attention usage - handles masks properly
        # ===================================================================

        if self.use_flash:
            # Prepare mask in Flash Attention format
            flash_mask = self._prepare_flash_attention_mask(
                batch_size,
                tgt_len,
                src_len,
                key_padding_mask,
                attn_mask,
                query.device,
                Q.dtype,
            )

            context = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=flash_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,  # Cross-attention is never causal
            )
        else:
            # Manual attention computation
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

            # Apply masks
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)
                scores = scores.masked_fill(attn_mask, float("-inf"))

            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(key_padding_mask, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context = torch.matmul(attn_weights, V)

        # Reshape and project
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        )
        output = self.w_o(context)

        # Cache K, V in head-split format
        new_cache = None
        if use_cache:
            new_cache = CacheEntry(k=K.contiguous(), v=V.contiguous(), seq_len=src_len)

        return output, new_cache
