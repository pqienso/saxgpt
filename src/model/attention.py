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

    def _combine_masks(
        self,
        batch_size: int,
        seq_len: int,
        kv_len: int,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Combine all masks into a single boolean mask.
        
        Returns:
            Combined boolean mask [batch, seq_len, kv_len] where True = mask out, or None
        """
        combined_mask = None

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [seq_len, kv_len] -> [batch, seq_len, kv_len]
                combined_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # [batch, seq_len, kv_len]
                combined_mask = attn_mask.clone()

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=device, dtype=torch.bool),
                diagonal=1,
            )
            if combined_mask is None:
                combined_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                combined_mask = combined_mask | causal_mask.unsqueeze(0)

        # Add key padding mask
        if key_padding_mask is not None:
            # [batch, kv_len] -> [batch, 1, kv_len]
            padding_mask = key_padding_mask.unsqueeze(1)
            if combined_mask is None:
                combined_mask = padding_mask.expand(-1, seq_len, -1)
            else:
                combined_mask = combined_mask | padding_mask

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
            attn_mask: Boolean mask [seq_len, kv_len] or [batch, seq_len, kv_len]
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

        # Determine if we can use the is_causal flag directly
        use_is_causal_flag = (
            is_causal and cache is None and attn_mask is None and key_padding_mask is None
        )

        if self.use_flash:
            if use_is_causal_flag:
                # Let Flash Attention handle causal masking efficiently
                context = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                # Combine all masks into a single boolean mask
                combined_mask = self._combine_masks(
                    batch_size,
                    seq_len,
                    kv_len,
                    key_padding_mask,
                    attn_mask,
                    is_causal and cache is None,
                    x.device,
                )

                context = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=combined_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,
                )
        else:
            # Manual attention computation (fallback)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            # scores: [batch, nhead, seq_len, kv_len]

            # Combine masks
            combined_mask = self._combine_masks(
                batch_size,
                seq_len,
                kv_len,
                key_padding_mask,
                attn_mask,
                is_causal and cache is None,
                x.device,
            )

            if combined_mask is not None:
                # [batch, seq_len, kv_len] -> [batch, 1, seq_len, kv_len]
                scores = scores.masked_fill(combined_mask.unsqueeze(1), float("-inf"))

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

    def _combine_masks(
        self,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Combine all masks into a single boolean mask.
        
        Returns:
            Combined boolean mask [batch, tgt_len, src_len] where True = mask out, or None
        """
        combined_mask = None

        # Start with attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [tgt_len, src_len] -> [batch, tgt_len, src_len]
                combined_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # Already [batch, tgt_len, src_len]
                combined_mask = attn_mask.clone()

        # Add key padding mask
        if key_padding_mask is not None:
            # [batch, src_len] -> [batch, 1, src_len]
            padding_mask = key_padding_mask.unsqueeze(1)
            if combined_mask is None:
                combined_mask = padding_mask.expand(-1, tgt_len, -1)
            else:
                combined_mask = combined_mask | padding_mask

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

        if self.use_flash:
            # Combine masks into a single boolean mask
            combined_mask = self._combine_masks(
                batch_size, tgt_len, src_len, key_padding_mask, attn_mask, query.device
            )

            context = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=combined_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,  # Cross-attention is never causal
            )
        else:
            # Manual attention computation
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

            # Combine masks
            combined_mask = self._combine_masks(
                batch_size, tgt_len, src_len, key_padding_mask, attn_mask, query.device
            )

            if combined_mask is not None:
                # [batch, tgt_len, src_len] -> [batch, 1, tgt_len, src_len]
                scores = scores.masked_fill(combined_mask.unsqueeze(1), float("-inf"))

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
