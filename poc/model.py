import torch
from torch import nn
import torch.nn.functional as F

import typing as tp
import math


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
