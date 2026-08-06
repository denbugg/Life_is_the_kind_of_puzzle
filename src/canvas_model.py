"""Canvas-first model for reconstructing a coarse image from shuffled tiles.

``CanvasNet`` deliberately never receives a tile position.  It first encodes
each tile into a token, adds a permutation-invariant summary of the set, and
then lets a learned two-dimensional grid of output slots cross-attend to that
set.  The slots are decoded as small image patches, yielding a low-resolution
canvas suitable for a subsequent tile-to-canvas matching stage.

The default configuration is intentionally small (about 1.5M parameters) and
uses one 576-by-576 cross-attention plus one slot self-attention operation.
It is designed to fit comfortably on an 8 GB GPU with modest batch sizes.
"""

from __future__ import annotations

from typing import Literal, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


IntPair = Union[int, Tuple[int, int]]


def _pair(value: IntPair, name: str) -> Tuple[int, int]:
    """Convert an integer or a two-item tuple to a validated ``(height, width)``."""
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
    """Return a GroupNorm group count that divides ``channels``."""
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvResidual(nn.Module):
    """Small spatial residual block used inside the per-tile CNN."""

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
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class TileEncoder(nn.Module):
    """Encode individual RGB tiles into tokens while retaining a small thumbnail.

    A pure global-pooled CNN can discard the within-tile layout needed to draw
    a 4x4/5x5 output patch.  The thumbnail branch preserves that inexpensive
    signal; its projection is fused with the convolutional token.
    """

    def __init__(self, d: int, patch: IntPair, width: int | None = None) -> None:
        super().__init__()
        patch_h, patch_w = _pair(patch, "patch")
        width = int(width or max(32, d // 2))
        if width <= 0:
            raise ValueError("width must be positive")

        self.patch = (patch_h, patch_w)
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            _ConvResidual(width),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(width, d, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_groups(d), d),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            _ConvResidual(d),
            nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_groups(d), d),
            nn.GELU(),
            _ConvResidual(d),
        )
        self.cnn_proj = nn.Linear(d, d)
        self.thumb_proj = nn.Linear(3 * patch_h * patch_w, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, tiles: Tensor) -> Tensor:
        """Return one ``d``-dimensional token per tile.

        Args:
            tiles: Float tensor shaped ``(batch, tiles, 3, height, width)``.
        """
        batch, count, channels, height, width = tiles.shape
        if channels != 3:
            raise ValueError(f"CanvasNet expects RGB tiles (3 channels), got {channels}")
        flat = tiles.reshape(batch * count, channels, height, width)
        features = self.body(self.down1(self.stem(flat)))
        cnn_token = self.cnn_proj(F.adaptive_avg_pool2d(features, 1).flatten(1))
        thumbnail = F.adaptive_avg_pool2d(flat, self.patch).flatten(1)
        thumbnail_token = self.thumb_proj(thumbnail)
        return self.norm(cnn_token + thumbnail_token).view(batch, count, -1)


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


class _SelfAttentionBlock(nn.Module):
    """Standard pre-norm self-attention block for a sequence of tokens."""

    def __init__(self, d: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d)
        self.mlp = _FeedForward(d, mlp_ratio, dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class _CrossAttentionBlock(nn.Module):
    """Let output-grid slots read the unordered tile set, then refine each slot."""

    def __init__(self, d: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.slot_norm = nn.LayerNorm(d)
        self.tile_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(d)
        self.mlp = _FeedForward(d, mlp_ratio, dropout)

    def forward(self, slots: Tensor, tiles: Tensor) -> Tensor:
        query = self.slot_norm(slots)
        context = self.tile_norm(tiles)
        slots = slots + self.attn(query, context, context, need_weights=False)[0]
        return slots + self.mlp(self.mlp_norm(slots))


class CanvasNet(nn.Module):
    """Predict a coarse canvas directly from an unordered set of puzzle tiles.

    Args:
        grid: Output slot grid side length (or ``(height, width)``).  With the
            default 24 this model expects 576 input tiles.
        patch: Low-resolution patch emitted by each slot.  The default 4 gives
            a 96x96 canvas; use ``patch=5`` for 120x120.
        d: Token width.  ``d=128`` is the compact 8-GB-friendly default.
        heads: Attention head count; must divide ``d``.
        cross_layers: Number of slot-to-tile cross-attention blocks.
        slot_self_blocks: Number of optional slot self-attention blocks.  They
            mix content between nearby output positions after cross-attention.
        tile_self_blocks: Optional self-attention over the unordered tile set.
            Zero is cheapest and remains the recommended default.
        output_activation: ``"sigmoid"`` emits a canvas in [0, 1]; ``"none"``
            returns raw decoder logits for a caller that owns its loss/scale.

    Forward returns a dictionary containing at least:

    * ``canvas``: ``(B, 3, grid_h * patch_h, grid_w * patch_w)``;
    * ``tile_tokens``: ``(B, number_of_tiles, d)``;
    * ``slot_tokens``: ``(B, grid_h * grid_w, d)``.

    The model is permutation-invariant with respect to input tile order (up to
    normal floating-point reduction differences), while slot order is fixed
    row-major output-grid order.
    """

    def __init__(
        self,
        grid: IntPair = 24,
        patch: IntPair = 4,
        d: int = 128,
        heads: int = 4,
        cross_layers: int = 1,
        slot_self_blocks: int = 1,
        tile_self_blocks: int = 0,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        encoder_width: int | None = None,
        output_activation: Literal["sigmoid", "none"] = "sigmoid",
    ) -> None:
        super().__init__()
        self.grid = _pair(grid, "grid")
        self.patch = _pair(patch, "patch")
        self.d = int(d)
        self.num_slots = self.grid[0] * self.grid[1]
        self.output_activation = output_activation

        if self.d <= 0:
            raise ValueError("d must be positive")
        if heads <= 0 or self.d % heads:
            raise ValueError(f"d ({self.d}) must be divisible by heads ({heads})")
        if cross_layers <= 0:
            raise ValueError("cross_layers must be at least one")
        if slot_self_blocks < 0 or tile_self_blocks < 0:
            raise ValueError("the number of self-attention blocks cannot be negative")
        if output_activation not in {"sigmoid", "none"}:
            raise ValueError("output_activation must be 'sigmoid' or 'none'")

        self.tile_encoder = TileEncoder(self.d, self.patch, encoder_width)

        # The parameter is stored as a true 2-D grid, not a generic sequence:
        # its flattening establishes row-major canvas slots at forward time.
        self.grid_queries = nn.Parameter(
            torch.empty(1, self.grid[0], self.grid[1], self.d)
        )
        nn.init.trunc_normal_(self.grid_queries, std=0.02)

        # A set-invariant global context lets every tile token know what image
        # it belongs to without imposing any order on the tile set.
        self.set_context = nn.Sequential(
            nn.LayerNorm(self.d),
            nn.Linear(self.d, 2 * self.d),
            nn.GELU(),
            nn.Linear(2 * self.d, 2 * self.d),
        )
        self.tile_post_norm = nn.LayerNorm(self.d)
        self.tile_blocks = nn.ModuleList(
            _SelfAttentionBlock(self.d, heads, mlp_ratio, dropout)
            for _ in range(tile_self_blocks)
        )
        self.cross_blocks = nn.ModuleList(
            _CrossAttentionBlock(self.d, heads, mlp_ratio, dropout)
            for _ in range(cross_layers)
        )
        self.slot_blocks = nn.ModuleList(
            _SelfAttentionBlock(self.d, heads, mlp_ratio, dropout)
            for _ in range(slot_self_blocks)
        )

        decoder_width = max(32, self.d // 2)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.d, decoder_width, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(decoder_width), decoder_width),
            nn.GELU(),
            _ConvResidual(decoder_width),
            nn.Conv2d(
                decoder_width,
                3 * self.patch[0] * self.patch[1],
                kernel_size=1,
            ),
        )

    def _add_set_context(self, tokens: Tensor) -> Tensor:
        """Condition every tile token on an order-invariant pooled summary."""
        context = self.set_context(tokens.mean(dim=1, keepdim=True))
        scale, shift = context.chunk(2, dim=-1)
        # Tanh starts as a gentle modulation and avoids amplifying early noise.
        return self.tile_post_norm(tokens) * (1.0 + 0.1 * torch.tanh(scale)) + shift

    def _decode(self, slots: Tensor) -> Tensor:
        """Turn row-major slots into an image by emitting one patch per slot."""
        batch = slots.shape[0]
        grid_h, grid_w = self.grid
        patch_h, patch_w = self.patch
        feature_map = slots.transpose(1, 2).reshape(batch, self.d, grid_h, grid_w)
        logits = self.decoder(feature_map)
        # Channels encode (RGB, patch_y, patch_x).  This reshape supports both
        # square and rectangular grids/patches without relying on PixelShuffle.
        logits = logits.view(batch, 3, patch_h, patch_w, grid_h, grid_w)
        logits = logits.permute(0, 1, 4, 2, 5, 3).reshape(
            batch, 3, grid_h * patch_h, grid_w * patch_w
        )
        return torch.sigmoid(logits) if self.output_activation == "sigmoid" else logits

    def forward(self, frags: Tensor) -> dict[str, Tensor]:
        """Build a coarse canvas from shuffled RGB fragments.

        Args:
            frags: Floating tensor shaped ``(B, grid_h*grid_w, 3, H, W)``.
                Inputs are normally normalized to ``[0, 1]``.

        Returns:
            A dictionary with ``canvas``, ``tile_tokens``, and ``slot_tokens``.
        """
        if frags.ndim != 5:
            raise ValueError(
                "frags must have shape (batch, tiles, 3, height, width), "
                f"got {tuple(frags.shape)}"
            )
        if frags.shape[1] != self.num_slots:
            raise ValueError(
                f"grid={self.grid} needs {self.num_slots} tiles, got {frags.shape[1]}"
            )
        if frags.shape[2] != 3:
            raise ValueError(f"CanvasNet expects RGB tiles, got {frags.shape[2]} channels")

        tile_tokens = self._add_set_context(self.tile_encoder(frags))
        for block in self.tile_blocks:
            tile_tokens = block(tile_tokens)

        batch = frags.shape[0]
        slots = self.grid_queries.flatten(1, 2).expand(batch, -1, -1)
        for block in self.cross_blocks:
            slots = block(slots, tile_tokens)
        for block in self.slot_blocks:
            slots = block(slots)

        return {
            "canvas": self._decode(slots),
            "tile_tokens": tile_tokens,
            "slot_tokens": slots,
        }


if __name__ == "__main__":
    # CPU-only smoke test; device placement and mixed precision are deliberately
    # left to the caller/training loop.
    model = CanvasNet()
    with torch.no_grad():
        output = model(torch.rand(1, 24 * 24, 3, 20, 20))
    print({name: tuple(value.shape) for name, value in output.items()})
    print("parameters:", sum(parameter.numel() for parameter in model.parameters()))
