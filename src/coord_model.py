"""Permutation-equivariant coordinate classifier for shuffled puzzle tiles.

``CoordSetNet`` is deliberately given *only* the unordered tile bag.  Every
operation before the row/column heads is either applied independently to a
tile or is equivariant to a permutation of the tile axis.  In particular,
there are no tile-index embeddings, raster-order features, or positional
encodings on the input tokens.

The model is intentionally a small gate model: a per-tile CNN supplies local
appearance features, a symmetric global summary supplies scene context, and
one or two self-attention blocks let ambiguous tiles compare themselves with
the rest of the set.  The heads predict independent row and column logits;
the one-to-one assignment is left to the caller.
"""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor, nn


IntPair = Union[int, tuple[int, int]]


def _pair(value: IntPair, name: str) -> tuple[int, int]:
    """Normalize a scalar or ``(height, width)`` pair and validate it."""
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, tuple) and len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        raise TypeError(f"{name} must be an int or a two-item tuple, got {value!r}")
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} entries must be positive, got {result}")
    return result


def _groups(channels: int, maximum: int = 8) -> int:
    """Return a small GroupNorm group count that divides ``channels``."""
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualConv(nn.Module):
    """A tiny GroupNorm residual block that is stable for small batch sizes."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _groups(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.net(x))


class _TileEncoder(nn.Module):
    """Encode every raw RGB tile independently into one token."""

    def __init__(self, d: int, width: int | None = None) -> None:
        super().__init__()
        width = int(width or max(32, d // 2))
        if width <= 0:
            raise ValueError("encoder width must be positive")

        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            _ResidualConv(width),
            nn.Conv2d(width, d, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_groups(d), d),
            nn.GELU(),
            _ResidualConv(d),
            nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_groups(d), d),
            nn.GELU(),
            _ResidualConv(d),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

    def forward(self, tiles: Tensor) -> Tensor:
        # Flatten only batch and set axes: this applies exactly the same CNN to
        # each member, hence preserves permutation equivariance.
        batch, count, channels, height, width = tiles.shape
        flat = tiles.reshape(batch * count, channels, height, width)
        return self.proj(self.net(flat)).reshape(batch, count, -1)


class _FeedForward(nn.Module):
    """Pre-norm transformer MLP."""

    def __init__(self, d: int, ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = max(d, int(round(d * ratio)))
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _SetAttentionBlock(nn.Module):
    """Self-attention over an unordered set, with no position-dependent state."""

    def __init__(self, d: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        # ``need_weights=False`` selects PyTorch's memory-efficient attention
        # path where available and avoids retaining a (B, 576, 576) output.
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d)
        self.mlp = _FeedForward(d, mlp_ratio, dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm1(tokens)
        tokens = tokens + self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        return tokens + self.mlp(self.norm2(tokens))


class _SetContext(nn.Module):
    """Inject mean and spread of the tile set into every individual token."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.summary = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, 2 * d),
        )
        self.token_norm = nn.LayerNorm(d)

    def forward(self, tokens: Tensor) -> Tensor:
        mean = tokens.mean(dim=1)
        # Population standard deviation is defined for a one-element set too.
        spread = tokens.var(dim=1, unbiased=False).add(1e-6).sqrt()
        scale, shift = self.summary(torch.cat((mean, spread), dim=-1)).chunk(2, dim=-1)
        # A bounded scale makes this initially gentle modulation rather than an
        # uncontrolled gain on already-corrupted input pixels.
        return self.token_norm(tokens) * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)


class CoordSetNet(nn.Module):
    """Predict row/column classes for an unordered bag of corrupted tiles.

    Args:
        grid: Number of output rows/columns, or a ``(rows, cols)`` pair.
        tiles: Required number of input tiles.  The standard puzzle uses 576.
        tile_size: Expected raw tile height/width.  The standard puzzle uses
            20x20 fragments.
        d: Token width; the default 96 keeps full 576-tile attention compact.
        heads: Number of attention heads.  It must divide ``d``.
        set_layers: One or two permutation-equivariant self-attention layers.
        mlp_ratio: Expansion factor in each attention block's MLP.
        dropout: Dropout in set-attention blocks and classifier heads.
        encoder_width: Optional width of the first CNN stage.

    The returned ``tile_tokens`` remain in the same order as the supplied
    tiles.  Therefore, for a permutation ``p``, logits and tokens from
    ``model(tiles[:, p])`` are the corresponding outputs from ``model(tiles)``
    indexed by ``p`` (when dropout is disabled / evaluation mode is used).
    """

    def __init__(
        self,
        grid: IntPair = 24,
        tiles: int = 576,
        tile_size: IntPair = 20,
        d: int = 96,
        heads: int = 4,
        set_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        encoder_width: int | None = None,
    ) -> None:
        super().__init__()
        self.grid = _pair(grid, "grid")
        self.tiles = int(tiles)
        self.tile_size = _pair(tile_size, "tile_size")
        self.d = int(d)

        if self.tiles <= 0:
            raise ValueError("tiles must be positive")
        if self.d <= 0:
            raise ValueError("d must be positive")
        if heads <= 0 or self.d % heads:
            raise ValueError(f"d ({self.d}) must be divisible by heads ({heads})")
        if set_layers not in (1, 2):
            raise ValueError("set_layers must be 1 or 2")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.tile_encoder = _TileEncoder(self.d, encoder_width)
        self.set_context = _SetContext(self.d)
        self.set_blocks = nn.ModuleList(
            _SetAttentionBlock(self.d, heads, mlp_ratio, dropout)
            for _ in range(set_layers)
        )
        self.final_norm = nn.LayerNorm(self.d)
        self.row_head = nn.Sequential(nn.Linear(self.d, self.grid[0]))
        self.col_head = nn.Sequential(nn.Linear(self.d, self.grid[1]))

    def forward(self, tiles: Tensor) -> dict[str, Tensor]:
        """Classify all tiles in a raw shuffled input bag.

        Args:
            tiles: Float tensor with shape ``(B, 576, 3, 20, 20)`` for the
                default geometry.  Values may be raw [0, 1] pixels or caller
                normalized pixels; normalization policy belongs to the data
                pipeline.

        Returns:
            A dictionary containing:

            * ``row_logits``: ``(B, tiles, grid_rows)``;
            * ``col_logits``: ``(B, tiles, grid_cols)``;
            * ``tile_tokens``: contextual tokens ``(B, tiles, d)``.
        """
        if tiles.ndim != 5:
            raise ValueError(
                "tiles must have shape (batch, tiles, 3, height, width), "
                f"got {tuple(tiles.shape)}"
            )
        if tiles.shape[1] != self.tiles:
            raise ValueError(f"expected {self.tiles} tiles, got {tiles.shape[1]}")
        if tiles.shape[2] != 3:
            raise ValueError(f"CoordSetNet expects RGB tiles, got {tiles.shape[2]} channels")
        if tuple(tiles.shape[-2:]) != self.tile_size:
            raise ValueError(
                f"expected tile size {self.tile_size}, got {tuple(tiles.shape[-2:])}"
            )
        if not torch.is_floating_point(tiles):
            raise TypeError(f"tiles must be floating point, got {tiles.dtype}")

        tokens = self.set_context(self.tile_encoder(tiles))
        for block in self.set_blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        return {
            "row_logits": self.row_head(tokens),
            "col_logits": self.col_head(tokens),
            "tile_tokens": tokens,
        }


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def smoke(batch_size: int = 1, device: torch.device | str = "cpu") -> dict[str, tuple[int, ...]]:
    """Run a minimal forward-pass and shape assertion for the default model."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model = CoordSetNet().to(device).eval()
    tiles = torch.rand(batch_size, 576, 3, 20, 20, device=device)
    with torch.no_grad():
        output = model(tiles)

    expected = {
        "row_logits": (batch_size, 576, 24),
        "col_logits": (batch_size, 576, 24),
        "tile_tokens": (batch_size, 576, 96),
    }
    shapes = {name: tuple(value.shape) for name, value in output.items()}
    if shapes != expected:
        raise AssertionError(f"unexpected CoordSetNet smoke shapes: {shapes}")
    return shapes


if __name__ == "__main__":
    print(smoke())
    print("parameters:", count_params(CoordSetNet()))
