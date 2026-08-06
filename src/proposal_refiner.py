"""Proposal-conditioned global permutation refiner.

The refiner is deliberately given a spatial draft of the puzzle.  A small
convolutional stem reads reliable local components in that draft and a global
transformer decides where their content belongs in the 24x24 image frame.  It
predicts one clean paired-alignment embedding per output cell; a Hungarian
assignment against the 576 dirty tile embeddings enforces an exact bijection.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from config import GRID, NFRAG


def _groups(channels: int) -> int:
    for value in range(min(8, channels), 0, -1):
        if channels % value == 0:
            return value
    return 1


class _GridBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class ProposalRefiner(nn.Module):
    """Turn a draft board of dirty embeddings into clean slot queries."""

    def __init__(
        self,
        embed_dim: int = 128,
        hidden: int = 192,
        layers: int = 4,
        heads: int = 6,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.heads = int(heads)

        self.input = nn.Linear(embed_dim, hidden)
        self.row = nn.Parameter(torch.randn(GRID, hidden) * 0.02)
        self.col = nn.Parameter(torch.randn(GRID, hidden) * 0.02)
        self.local = nn.Sequential(_GridBlock(hidden, 1), _GridBlock(hidden, 2))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_context = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(hidden)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        # Start as an exact identity refiner.  A draft embedding has cosine 1
        # with its own dirty-tile key, so Hungarian decoding reproduces the
        # incoming board until learned evidence is strong enough to move it.
        self.log_residual_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, draft: Tensor) -> Tensor:
        if draft.ndim != 3 or tuple(draft.shape[1:]) != (NFRAG, self.embed_dim):
            raise ValueError(
                f"draft must be (B,{NFRAG},{self.embed_dim}), got {tuple(draft.shape)}"
            )
        batch = draft.shape[0]
        pos = (self.row[:, None, :] + self.col[None, :, :]).reshape(1, NFRAG, self.hidden)
        x = self.input(draft) + pos
        grid = x.reshape(batch, GRID, GRID, self.hidden).permute(0, 3, 1, 2)
        x = self.local(grid).permute(0, 2, 3, 1).reshape(batch, NFRAG, self.hidden)
        x = self.global_context(x)
        delta = self.output(x)
        gain = self.log_residual_gain.exp().clamp(max=10.0)
        return F.normalize(draft + gain * delta, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    model = ProposalRefiner(embed_dim=32, hidden=48, layers=2, heads=4)
    value = model(torch.randn(2, NFRAG, 32))
    print(tuple(value.shape), count_parameters(model), value.norm(dim=-1).mean().item())
