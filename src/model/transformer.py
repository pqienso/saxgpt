import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List

from .codebook_layers import PositionalEncoding, MultiCodesEmbedding, MultiCodesLinear
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .attention import CacheEntry


class EncoderDecoderTransformer(nn.Module):
    """Complete Encoder-Decoder Transformer with KV caching."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
        max_seq_len: int = 5000,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder
        self.encoder = TransformerEncoder(
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            activation,
            norm_first,
        )

        # Decoder
        self.decoder = TransformerDecoder(
            d_model,
            nhead,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            activation,
            norm_first,
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: [batch_size, src_seq_len] - token indices
            src_mask: Optional attention mask
            src_key_padding_mask: [batch_size, src_seq_len] - padding mask

        Returns:
            memory: [batch_size, src_seq_len, d_model]
        """
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(
            src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Dict[str, CacheEntry]]] = None,
        use_cache: bool = False,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, CacheEntry]]]:
        """
        Decode target sequence.

        Args:
            tgt: [batch_size, tgt_seq_len]
            memory: [batch_size, src_seq_len, d_model]
            cache: Optional list of cache dicts for each decoder layer
            use_cache: Whether to return updated caches
            is_causal: Apply causal masking

        Returns:
            logits: [batch_size, tgt_seq_len, tgt_vocab_size]
            new_cache: Updated cache list
        """
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Calculate PE offset from cache
        pe_offset = 0
        if cache and cache[0] and cache[0].get("self_attn"):
            pe_offset = cache[0]["self_attn"].seq_len

        tgt_emb = self.pos_encoder(tgt_emb, offset=pe_offset)

        output, new_cache = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            cache=cache,
            use_cache=use_cache,
            is_causal=is_causal,
        )

        logits = self.output_projection(output)
        return logits, new_cache

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass (training mode - no caching).

        Args:
            src: [batch_size, src_seq_len]
            tgt: [batch_size, tgt_seq_len]

        Returns:
            logits: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask, src_key_padding_mask)
        logits, _ = self.decode(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            use_cache=False,
            is_causal=(tgt_mask is None),  # Auto-detect causal if no mask provided
        )
        return logits

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int, device: torch.device = None
    ) -> torch.Tensor:
        """
        Generate causal mask for autoregressive decoding.

        Args:
            sz: Sequence length
            device: Device to create mask on (defaults to CPU)

        Returns:
            mask: [sz, sz] boolean mask with True in upper triangle
        """
        mask = torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1
        )
        return mask

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        start_token: int,
        end_token: int,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with full KV-caching.

        Args:
            src: [batch_size, src_seq_len]
            max_len: Maximum generation length
            start_token: Start token ID
            end_token: End token ID (stops generation)
            src_key_padding_mask: [batch_size, src_seq_len]
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            top_p: If set, nucleus sampling threshold

        Returns:
            generated: [batch_size, generated_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode source once
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)

        # Initialize cache
        cache = [
            {"self_attn": None, "cross_attn": None}
            for _ in range(self.num_decoder_layers)
        ]

        # Initialize with start token
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len - 1):
            # Only process the last token (rest are cached)
            current_token = tgt[:, -1:]

            # Decode with caching
            logits, cache = self.decode(
                current_token,
                memory,
                memory_key_padding_mask=src_key_padding_mask,
                cache=cache,
                use_cache=True,
                is_causal=True,
            )

            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(
                    next_token_logits, min(top_k, next_token_logits.size(-1))
                )
                next_token_logits[next_token_logits < v[:, [-1]]] = float("-inf")

            # Optional top-p (nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(
                    indices_to_remove, float("-inf")
                )

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Check for end token
            finished |= next_token.squeeze(1) == end_token
            if finished.all():
                break

        return tgt
