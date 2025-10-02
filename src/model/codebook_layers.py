import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with offset support for caching."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            offset: Starting position index (>0 for cached steps)

        Returns:
            [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, offset : offset + seq_len, :]
        return self.dropout(x)


class MultiCodesEmbedding(nn.Module):
    """Embedding layer for multiple parallel codebooks."""

    def __init__(
        self,
        num_codebooks: int,
        vocab_size: int,
        d_model: int,
        padding_idx: int,
        scale_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.scale_embeddings = scale_embeddings
        self.scale_factor = math.sqrt(d_model) if scale_embeddings else 1.0
        
        self.emb = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=d_model,
                    padding_idx=padding_idx,
                )
                for _ in range(num_codebooks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_codebooks, seq_len]

        Returns:
            [batch_size, seq_len, d_model]
        """
        _, codebooks, _ = x.shape
        assert codebooks == self.num_codebooks
        
        embeddings = [self.emb[i](x[:, i]) for i in range(self.num_codebooks)]
        return torch.stack(embeddings, dim=0).sum(dim=0) * self.scale_factor


class MultiCodesLinear(nn.Module):
    """Linear projection to logits for multiple codebooks."""

    def __init__(
        self,
        num_codebooks: int,
        vocab_size: int,
        d_model: int,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, vocab_size) for _ in range(num_codebooks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, num_codebooks, seq_len, vocab_size]
        """
        return torch.stack([linear(x) for linear in self.linears], dim=1)
