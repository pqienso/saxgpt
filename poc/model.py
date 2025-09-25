import torch
from torch import nn
import torch.nn.functional as F

import typing as tp
import math


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        device: torch.device,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = x + self.pe[:, :T]
        return self.dropout(x)


class MHAModel(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_streams: int = 4,
        vocab_size: int = 2048,
        max_seq_len: int = 1600,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        norm_first: bool = True,
        bias: bool = False,
        dropout: float = 0.1,
        device: tp.Any | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_streams = num_streams
        self.vocab_size = vocab_size
        self.padding_index = vocab_size
        self.seq_len = max_seq_len
        self.emb = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=vocab_size + 1,
                    embedding_dim=d_model,
                    padding_idx=self.padding_index,
                    device=device,
                )
                for _ in range(num_streams)
            ]
        )
        self.pe = PositionalEncoding(d_model, device, dropout, max_seq_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            bias=bias,
            device=device,
        )
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, vocab_size, device=device) for _ in range(num_streams)]
        )
        self.device = device

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_is_causal: bool = False,
        tgt_is_causal: bool = True,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        (
            B,
            _,
            _,
        ) = src.shape
        assert (src >= self.padding_index).sum() == B * 20, (
            "Padding index only allowed at start and end of src"
        )
        transformer_out = self._get_transformer_output(
            src, tgt, src_is_causal, tgt_is_causal, memory_is_causal
        )

        B, T, C = transformer_out.shape
        logits = torch.stack(
            [linear(transformer_out) for linear in self.linears], dim=1
        )

        return logits

    def generate(
        self,
        src: torch.Tensor,
        src_is_causal: bool = False,
        tgt_is_causal: bool = True,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        if src.dim() == 2:
            src = src.unsqueeze(0)
        B, num_streams, T = src.shape

        first_token = torch.Tensor([[self.padding_index for _ in range(num_streams)]])
        tgt = torch.zeros_like(src)
        tgt[:, :, 0] = first_token
        index = 1

        while index < T:
            tokens = self._get_next_token(
                src,
                tgt[:, :, :index],
                src_is_causal=src_is_causal,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
            tgt[:, :, index] = tokens
            index += 1
        return tgt

    def _get_transformer_output(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_is_causal: bool = False,
        tgt_is_causal: bool = True,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        if src.dim() == 2:
            assert not self.training, "Use batched tensors for training"
            src = src.unsqueeze(0)
            tgt = tgt.unsqueeze(0)
        B, num_streams, T = src.shape
        assert num_streams == self.num_streams, (
            f"src must have {self.num_streams} streams of token streams, not {num_streams}"
        )
        B, num_streams, T = tgt.shape
        assert num_streams == self.num_streams, (
            f"tgt must have {self.num_streams} streams of token streams, not {num_streams}"
        )

        src_emb = sum(
            [self.emb[stream](src[:, stream]) for stream in range(self.num_streams)]
        )
        tgt_emb = sum(
            [self.emb[stream](tgt[:, stream]) for stream in range(self.num_streams)]
        )
        src_emb = self.pe(src_emb)
        tgt_emb = self.pe(tgt_emb)

        src_mask, tgt_mask, memory_mask = None, None, None
        if src_is_causal:
            src_mask = self.transformer.generate_square_subsequent_mask(
                src.shape[-1], device=self.device
            )
        if tgt_is_causal:
            tgt_mask = self.transformer.generate_square_subsequent_mask(
                tgt.shape[-1], device=self.device
            )
        if memory_is_causal:
            memory_mask = self.transformer.generate_square_subsequent_mask(
                src.shape[-1], device=self.device
            )

        transformer_out = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )  # B, T, C
        return transformer_out

    def _get_next_token(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_is_causal: bool = False,
        tgt_is_causal: bool = True,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        transformer_out = self._get_transformer_output(
            src, tgt, src_is_causal, tgt_is_causal, memory_is_causal
        )

        # (B, num_streams, T, vocab_size)
        logits = torch.stack(
            [linear(transformer_out) for linear in self.linears], dim=1
        )
        # (B, num_streams)
        return logits.argmax(-1)[:, :, -1]

    @staticmethod
    def add_delay_interleaving(
        streams: torch.Tensor, padding_idx: int = 2048
    ) -> torch.Tensor:
        num_streams = len(streams)
        new_streams = []
        for index, stream in enumerate(streams):
            new_streams.append(
                F.pad(stream, (index + 1, num_streams - index), value=padding_idx)
            )
        return torch.stack(new_streams)

    @staticmethod
    def remove_delay_interleaving(streams: torch.Tensor) -> torch.Tensor:
        num_streams = len(streams)
        stream_length = streams.shape[-1]
        new_streams = []
        for index, stream in enumerate(streams):
            new_streams.append(
                torch.narrow(
                    stream, -1, 1 + index, stream_length - (num_streams - 1) - 2
                )
            )
        return torch.stack(new_streams)


class MLASelfAttentionBlock(nn.Module):
    """
    MLA self-attention for decoder self-attention blocks, with streaming support
    """

    def __init__(
        self,
        d_model: int,
        d_latent: int,
        num_heads: int,
        causal: bool = True,
        dropout: float = 0.1,
        bias: bool = False,
        max_batch_size: int = 32,
        max_context_len: int = 1500,
    ):
        super(MLASelfAttentionBlock, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_latent < d_model, "d_latent should be a lower rank representation"

        self.w_dkv = nn.Linear(d_model, d_latent, bias=bias)
        self.w_uk = nn.Linear(d_latent, d_model, bias=bias)
        self.w_uv = nn.Linear(d_latent, d_model, bias=bias)
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.residuals_dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_latent = d_latent
        self.max_batch_size = max_batch_size
        self.num_heads = num_heads
        self.scaling_factor = math.sqrt(d_model)

        self.register_buffer(
            "kv_cache",
            torch.zeros([max_batch_size, max_context_len, d_latent]),
            persistent=False,
        )

        self.is_causal = causal
        if self.is_causal:
            self.register_buffer(
                "causal_bool_mask",
                torch.triu(
                    torch.ones(max_context_len, max_context_len),
                    diagonal=1,
                ).bool(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        assert C == self.d_model, "Dimension -1 must correspond to embedding dimesion"
        assert B <= self.max_batch_size, (
            f"Batch size greater than {self.max_batch_size}"
        )

        if self.training:
            kv_latent = self.w_dkv(x)
            k = self.w_uk(kv_latent)
            v = self.w_uv(kv_latent)
            # (B, T, C) x (C, C) -> (B, T, C)
            q = self.w_q(x)

            # (B, nh, T, hs)
            k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
            q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

            attention = torch.einsum("bhts,bhTs->bhtT", q, k) / self.scaling_factor
            if self.is_causal:
                attention = attention.masked_fill(
                    self.causal_bool_mask[:T, :T], float("-inf")
                )
            attention = F.softmax(attention, dim=-1)
            attention = self.attention_dropout(attention)

            # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = attention @ v
            # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
            y = y.transpose(1, 2).contiguous().view(B, T, C)

            o = self.residuals_dropout(self.w_o(y))
            return o
        else:
            # Batch eval mode, not streaming.
            # (B, T, C) x (C, d_latent) -> (B, T, d_latent)
            kv_latent = self.w_dkv(x)
            attention = (
                torch.einsum("btc,hcl,bTl->bhtT", x, self.w_q_w_uk, kv_latent)
                / self.scaling_factor
            )
            if self.is_causal:
                attention = attention.masked_fill(
                    self.causal_bool_mask[:T, :T], float("-inf")
                )
            attention = F.softmax(attention, dim=-1)
            v = (
                self.w_uv(kv_latent)
                .view(B, T, self.num_heads, C // self.num_heads)
                .transpose(1, 2)
            )
            y = attention @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)

            o = self.residuals_dropout(self.w_o(y))
            return o

    def train(self, mode: bool):
        if mode:
            return super().train(True)

        w_q = self.w_q.weight.T.view(
            [self.d_model, self.num_heads, self.d_model // self.num_heads]
        )
        w_uk = self.w_uk.weight.view(
            [self.num_heads, self.d_model // self.num_heads, self.d_latent]
        )
        self.register_buffer("w_q_w_uk", torch.einsum("chs,hsl->hcl", w_q, w_uk))

        self.register_buffer(
            "w_uv_w_o", torch.matmul(self.w_uv.weight.T, self.w_o.weight.T)
        )
        return super().train(False)
