import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .attention import (
    MultiHeadCrossAttentionWithCache,
    MultiHeadSelfAttentionWithCache,
    CacheEntry,
)

class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with efficient caching."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        norm_first: bool = False,
    ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttentionWithCache(d_model, nhead, dropout)
        self.cross_attn = MultiHeadCrossAttentionWithCache(d_model, nhead, dropout)
        self.norm_first = norm_first
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == 'relu' else F.gelu
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, CacheEntry]] = None,
        return_cache: bool = False,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, CacheEntry]]:
        """
        Args:
            tgt: [batch_size, tgt_seq_len, d_model]
            memory: [batch_size, src_seq_len, d_model] - encoder output
            cache: Dict with 'self_attn' and 'cross_attn' keys
            return_cache: Whether to return updated cache
            is_causal: Apply causal masking for self-attention
        
        Returns:
            output: [batch_size, tgt_seq_len, d_model]
            new_cache: Updated cache dict
        """
        # Extract caches
        self_attn_cache = cache.get('self_attn') if cache else None
        cross_attn_cache = cache.get('cross_attn') if cache else None
        
        if self.norm_first:
            # Pre-norm: Self-attention
            tgt2, new_self_attn_cache = self.self_attn(
                self.norm1(tgt),
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                cache=self_attn_cache,
                return_cache=return_cache,
                is_causal=is_causal,
            )
            tgt = tgt + self.dropout1(tgt2)
            
            # Cross-attention
            tgt2, new_cross_attn_cache = self.cross_attn(
                self.norm2(tgt), memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                cache=cross_attn_cache,
                return_cache=return_cache
            )
            tgt = tgt + self.dropout2(tgt2)
            
            # Feedforward
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt)))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            # Post-norm: Self-attention
            tgt2, new_self_attn_cache = self.self_attn(
                tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                cache=self_attn_cache,
                return_cache=return_cache,
                is_causal=is_causal,
            )
            tgt = self.norm1(tgt + self.dropout1(tgt2))
            
            # Cross-attention
            tgt2, new_cross_attn_cache = self.cross_attn(
                tgt, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                cache=cross_attn_cache,
                return_cache=return_cache
            )
            tgt = self.norm2(tgt + self.dropout2(tgt2))
            
            # Feedforward
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = self.norm3(tgt + self.dropout3(tgt2))
        
        # Prepare new cache
        new_cache = {}
        if return_cache:
            new_cache['self_attn'] = new_self_attn_cache
            new_cache['cross_attn'] = new_cross_attn_cache
        
        return tgt, new_cache


class TransformerDecoder(nn.Module):
    """Transformer decoder stack with full KV-caching support."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        norm_first: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, norm_first
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model) if norm_first else None
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Dict[str, CacheEntry]]] = None,
        return_cache: bool = False,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, CacheEntry]]]:
        """
        Args:
            cache: List of cache dicts, one per layer
            return_cache: Whether to return updated caches
            is_causal: Apply causal masking for autoregressive decoding
        
        Returns:
            output, new_cache_list
        """
        output = tgt
        new_cache_list = []
        
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            output, new_cache = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                cache=layer_cache,
                return_cache=return_cache,
                is_causal=is_causal,
            )
            new_cache_list.append(new_cache)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output, new_cache_list
