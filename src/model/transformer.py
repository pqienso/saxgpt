import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .codebook_layers import PositionalEncoding, MultiCodesEmbedding, MultiCodesLinear
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .attention import CacheEntry


class EncoderDecoderTransformer(nn.Module):
    """Complete Encoder-Decoder Transformer with KV caching for multiple codebooks."""

    def __init__(
        self,
        vocab_size: int = 2049,
        num_codebooks: int = 4,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
        max_seq_len: int = 5000,
        padding_idx: int = 2048,
        scale_embeddings: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

        self.embedding = MultiCodesEmbedding(
            num_codebooks, vocab_size, d_model, padding_idx, scale_embeddings
        )
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        self.encoder = TransformerEncoder(
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            activation,
            norm_first,
        )

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
        self.output_projection = MultiCodesLinear(
            num_codebooks, vocab_size - 1, d_model
        )

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
            src: [batch_size, num_codebooks, src_seq_len] - token indices
            src_mask: Optional attention mask
            src_key_padding_mask: [batch_size, src_seq_len] - padding mask

        Returns:
            memory: [batch_size, src_seq_len, d_model]
        """
        src_emb = self.pos_encoder(self.embedding(src))
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
            tgt: [batch_size, num_codebooks, tgt_seq_len]
            memory: [batch_size, src_seq_len, d_model]
            cache: Optional list of cache dicts for each decoder layer
            use_cache: Whether to return updated caches
            is_causal: Apply causal masking

        Returns:
            logits: [batch_size, num_codebooks, tgt_seq_len, vocab_size]
            new_cache: Updated cache list
        """
        tgt_emb = self.embedding(tgt)

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
            src: [batch_size, num_codebooks, src_seq_len]
            tgt: [batch_size, num_codebooks, tgt_seq_len]

        Returns:
            logits: [batch_size, num_codebooks, tgt_seq_len, vocab_size]
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
        start_tokens: torch.Tensor,
        end_token: int,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with full KV-caching for multiple codebooks.

        Args:
            src: [batch_size, num_codebooks, src_seq_len]
            max_len: Maximum generation length
            start_tokens: [batch_size, num_codebooks] - Start tokens for each codebook
            end_token: End token ID (stops generation for that codebook)
            src_key_padding_mask: [batch_size, src_seq_len]
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            top_p: If set, nucleus sampling threshold

        Returns:
            generated: [batch_size, num_codebooks, generated_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # Validate start_tokens shape
        assert start_tokens.shape == (batch_size, self.num_codebooks), (
            f"start_tokens must have shape [{batch_size}, {self.num_codebooks}]"
        )

        # Init memory and cache
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        cache = [
            {"self_attn": None, "cross_attn": None}
            for _ in range(self.num_decoder_layers)
        ]

        # Start with initial tokens: [batch_size, num_codebooks, 1]
        tgt = start_tokens.unsqueeze(-1)

        for step in range(max_len - 1):
            # Only process the last token (rest are cached)
            current_tokens = tgt[:, :, -1:]  # [batch_size, num_codebooks, 1]

            # Decode with caching
            logits, cache = self.decode(
                current_tokens,
                memory,
                memory_key_padding_mask=src_key_padding_mask,
                cache=cache,
                use_cache=True,
                is_causal=True,
            )

            # Get logits for last position: [batch_size, num_codebooks, vocab_size]
            next_token_logits = logits[:, :, -1, :] / temperature

            # Apply sampling to each codebook
            next_tokens = torch.zeros(
                batch_size, self.num_codebooks, dtype=torch.long, device=device
            )

            for cb_idx in range(self.num_codebooks):
                cb_logits = next_token_logits[:, cb_idx, :]  # [batch_size, vocab_size]

                # Optional top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(cb_logits, min(top_k, cb_logits.size(-1)))
                    cb_logits = cb_logits.masked_fill(
                        cb_logits < v[:, [-1]], float("-inf")
                    )

                # Optional top-p (nucleus) sampling
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        cb_logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    cb_logits = cb_logits.masked_fill(indices_to_remove, float("-inf"))

                # Sample next token
                probs = F.softmax(cb_logits, dim=-1)
                next_tokens[:, cb_idx] = torch.multinomial(
                    probs, num_samples=1
                ).squeeze(-1)

            # Add next tokens: [batch_size, num_codebooks, 1]
            tgt = torch.cat([tgt, next_tokens.unsqueeze(-1)], dim=-1)

        return tgt

    @torch.no_grad()
    def generate_greedy(
        self,
        src: torch.Tensor,
        max_len: int,
        start_tokens: torch.Tensor,
        end_token: int,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedy generation (deterministic) for multiple codebooks.

        Args:
            src: [batch_size, num_codebooks, src_seq_len]
            max_len: Maximum generation length
            start_tokens: [batch_size, num_codebooks]
            end_token: End token ID
            src_key_padding_mask: [batch_size, src_seq_len]

        Returns:
            generated: [batch_size, num_codebooks, generated_len]
        """
        self.eval()

        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        cache = [
            {"self_attn": None, "cross_attn": None}
            for _ in range(self.num_decoder_layers)
        ]

        tgt = start_tokens.unsqueeze(-1)

        for step in range(max_len - 1):
            current_tokens = tgt[:, :, -1:]

            logits, cache = self.decode(
                current_tokens,
                memory,
                memory_key_padding_mask=src_key_padding_mask,
                cache=cache,
                use_cache=True,
                is_causal=True,
            )

            # Greedy selection: [batch_size, num_codebooks]
            next_tokens = logits[:, :, -1, :].argmax(dim=-1)

            tgt = torch.cat([tgt, next_tokens.unsqueeze(-1)], dim=-1)

        return tgt
