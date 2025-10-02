import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .attention import MultiHeadSelfAttentionWithCache

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with batched QKV self-attention."""
    
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
        self.norm_first = norm_first
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == 'relu' else F.gelu
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, d_model]
            src_mask: Boolean attention mask
            src_key_padding_mask: [batch_size, seq_len]
        """
        if self.norm_first:
            # Pre-norm architecture
            src2, _ = self.self_attn(
                self.norm1(src),
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + self.dropout1(src2)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
            src = src + self.dropout2(src2)
        else:
            # Post-norm architecture
            src2, _ = self.self_attn(
                src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = self.norm1(src + self.dropout1(src2))
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(src2))
        
        return src

class TransformerEncoder(nn.Module):
    """Transformer encoder stack."""
    
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
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, norm_first
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model) if norm_first else None
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output
