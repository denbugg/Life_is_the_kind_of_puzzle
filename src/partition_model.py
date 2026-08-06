"""Permutation-equivariant coarse partitioning model for shuffled tiles.

``MacroPartitionNet`` takes an unordered bag of raw puzzle fragments and
produces a score for assigning each fragment to one of a small number of
*latent* groups.  The groups are deliberately slots rather than absolute
coordinates: slot 0 has no required relationship to a particular image region
from one example to the next.  A training loss can therefore match predicted
slots to target macro groups independently for every image.

There are no input tile-index embeddings or raster-order features.  The tile
encoder is applied independently, the set context is symmetric in the tile
axis, and slot-to-tile cross-attention is invariant to the memory order.  As a
result, shuffling input tiles shuffles ``assignment_logits`` and
``tile_tokens`` in exactly the same way (in evaluation mode).
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


IntPair = Union[int, tuple[int, int]]


def _pair(value: IntPair, name: str) -> tuple[int, int]:
    """Normalize a scalar or a two-item spatial shape and validate it."""
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
    """Find a small GroupNorm group count that divides ``channels``."""
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualConv(nn.Module):
    """Small residual CNN block robust to the small batches used on 8 GB GPUs."""

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
    """Apply the same compact CNN to each tile, producing one token per tile."""

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
        # Only batch and set axes are flattened.  The same encoder is thus
        # applied to every input member and preserves tile-axis equivariance.
        batch, count, channels, height, width = tiles.shape
        flat = tiles.reshape(batch * count, channels, height, width)
        return self.proj(self.net(flat)).reshape(batch, count, -1)


class _FeedForward(nn.Module):
    """Pre-norm Transformer MLP."""

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


class _SetContext(nn.Module):
    """Condition every tile on permutation-invariant mean and spread statistics."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.scene = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )
        self.tile_film = nn.Linear(d, 2 * d)
        self.tile_norm = nn.LayerNorm(d)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        mean = tokens.mean(dim=1)
        # ``unbiased=False`` keeps a well-defined result for a one-tile smoke
        # configuration too, while standard 576-tile inputs are unchanged.
        spread = tokens.var(dim=1, unbiased=False).add(1e-6).sqrt()
        scene = self.scene(torch.cat((mean, spread), dim=-1))
        scale, shift = self.tile_film(scene).chunk(2, dim=-1)
        # Start with a bounded gain so noisy/raw input cannot blow up a token.
        conditioned = self.tile_norm(tokens) * (
            1.0 + 0.1 * torch.tanh(scale).unsqueeze(1)
        ) + shift.unsqueeze(1)
        return conditioned, scene


class _SetAttentionBlock(nn.Module):
    """A self-attention block over the unordered tile set."""

    def __init__(self, d: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        # ``need_weights=False`` in forward lets PyTorch use its compact
        # attention path and avoids retaining a (B, 576, 576) attention map.
        self.attention = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d)
        self.mlp = _FeedForward(d, mlp_ratio, dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm1(tokens)
        tokens = tokens + self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        return tokens + self.mlp(self.norm2(tokens))


class _SlotCrossAttentionBlock(nn.Module):
    """Let a small learned slot set read an unordered tile memory."""

    def __init__(self, d: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.slot_norm = nn.LayerNorm(d)
        self.memory_norm = nn.LayerNorm(d)
        self.cross_attention = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(d)
        self.mlp = _FeedForward(d, mlp_ratio, dropout)

    def forward(self, slots: Tensor, tile_tokens: Tensor) -> Tensor:
        # Slot queries are fixed in slot order, whereas tile tokens are only
        # keys/values.  Reordering memory rows therefore cannot change a slot
        # output (other than harmless floating-point reduction ordering).
        memory = self.memory_norm(tile_tokens)
        slots = slots + self.cross_attention(
            self.slot_norm(slots), memory, memory, need_weights=False
        )[0]
        return slots + self.mlp(self.mlp_norm(slots))


class MacroPartitionNet(nn.Module):
    """Partition an unordered bag of tiles into learned latent macro slots.

    The default 36 slots match a 6 x 6 macro partition of a 24 x 24 puzzle,
    each ultimately expected to contain sixteen tiles.  This module does not
    impose a capacity or spatial meaning on a slot: callers can use a balanced
    assignment / matching loss appropriate to their supervision instead.

    Args:
        tiles: Required number of input tiles; defaults to the 576-piece task.
        slots: Number of latent groups to score per tile.
        tile_size: Required raw tile height/width, or a ``(height, width)`` pair.
        d: Token width.  ``96`` is intentionally small enough for a 576-member
            set on an 8 GB GPU.
        heads: Attention head count dividing ``d``.
        set_layers: Number of tile self-attention layers (one is normally ample).
        slot_layers: Number of learned-query cross-attention decoder layers.
        mlp_ratio: Expansion ratio in attention-block MLPs.
        dropout: Dropout used during training; use evaluation mode to test the
            exact permutation-equivariance contract.
        encoder_width: Optional width of the first CNN stage.

    Returns from :meth:`forward`:

    * ``assignment_logits`` -- shape ``(B, tiles, slots)``;
    * ``tile_tokens`` -- contextual tile features, shape ``(B, tiles, d)``.
    """

    def __init__(
        self,
        *,
        tiles: int = 576,
        slots: int = 36,
        tile_size: IntPair = 20,
        d: int = 96,
        heads: int = 4,
        set_layers: int = 1,
        slot_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        encoder_width: int | None = None,
    ) -> None:
        super().__init__()
        self.tiles = int(tiles)
        self.slots = int(slots)
        # ``num_slots`` is a clearer read-only alias for trainer diagnostics.
        self.num_slots = self.slots
        self.tile_size = _pair(tile_size, "tile_size")
        self.d = int(d)

        if self.tiles <= 0:
            raise ValueError("tiles must be positive")
        if self.slots <= 0:
            raise ValueError("slots must be positive")
        if self.d <= 0:
            raise ValueError("d must be positive")
        if heads <= 0 or self.d % heads:
            raise ValueError(f"d ({self.d}) must be divisible by heads ({heads})")
        if set_layers < 0:
            raise ValueError("set_layers must be non-negative")
        if slot_layers <= 0:
            raise ValueError("slot_layers must be positive")
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
        self.tile_norm = nn.LayerNorm(self.d)

        # These are latent group seeds, not input positions nor fixed image
        # coordinates.  Cross-attention makes every seed image-specific.
        self.slot_queries = nn.Parameter(torch.empty(1, self.slots, self.d))
        nn.init.trunc_normal_(self.slot_queries, std=0.02)
        self.scene_to_slots = nn.Sequential(
            nn.LayerNorm(self.d),
            nn.Linear(self.d, self.d),
            nn.GELU(),
            nn.Linear(self.d, self.d),
        )
        self.slot_blocks = nn.ModuleList(
            _SlotCrossAttentionBlock(self.d, heads, mlp_ratio, dropout)
            for _ in range(slot_layers)
        )
        self.slot_norm = nn.LayerNorm(self.d)

        # Use a cosine score after separate projections.  The learned scale
        # starts at 10 rather than a near-uniform temperature, but is capped to
        # remain safe under long mixed-precision training.
        self.tile_match = nn.Linear(self.d, self.d, bias=False)
        self.slot_match = nn.Linear(self.d, self.d, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, tiles: Tensor) -> dict[str, Tensor]:
        """Return per-tile latent-slot assignment scores and tile tokens."""
        if tiles.ndim != 5:
            raise ValueError(
                "tiles must have shape (batch, tiles, 3, height, width), "
                f"got {tuple(tiles.shape)}"
            )
        if tiles.shape[1] != self.tiles:
            raise ValueError(f"expected {self.tiles} tiles, got {tiles.shape[1]}")
        if tiles.shape[2] != 3:
            raise ValueError(
                f"MacroPartitionNet expects RGB tiles, got {tiles.shape[2]} channels"
            )
        if tuple(tiles.shape[-2:]) != self.tile_size:
            raise ValueError(
                f"expected tile size {self.tile_size}, got {tuple(tiles.shape[-2:])}"
            )
        if not torch.is_floating_point(tiles):
            raise TypeError(f"tiles must be floating point, got {tiles.dtype}")

        tile_tokens = self.tile_encoder(tiles)
        tile_tokens, scene = self.set_context(tile_tokens)
        for block in self.set_blocks:
            tile_tokens = block(tile_tokens)
        tile_tokens = self.tile_norm(tile_tokens)

        batch = tiles.shape[0]
        slots = self.slot_queries.expand(batch, -1, -1)
        slots = slots + self.scene_to_slots(scene).unsqueeze(1)
        for block in self.slot_blocks:
            slots = block(slots, tile_tokens)
        slots = self.slot_norm(slots)

        tile_match = F.normalize(self.tile_match(tile_tokens), dim=-1)
        slot_match = F.normalize(self.slot_match(slots), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        assignment_logits = scale * torch.matmul(tile_match, slot_match.transpose(-1, -2))
        return {
            "assignment_logits": assignment_logits,
            "tile_tokens": tile_tokens,
        }


def count_params(model: nn.Module) -> int:
    """Return the count of trainable parameters in ``model``."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _checked_permutation(tiles: Tensor, permutation: Tensor | None) -> Tensor:
    """Create or validate a tile-axis permutation on the input's device."""
    if tiles.ndim != 5:
        raise ValueError(f"tiles must be rank 5, got shape {tuple(tiles.shape)}")
    count = tiles.shape[1]
    if permutation is None:
        permutation = torch.randperm(count, device=tiles.device)
    if permutation.ndim != 1 or permutation.numel() != count:
        raise ValueError(
            f"permutation must have shape ({count},), got {tuple(permutation.shape)}"
        )
    permutation = permutation.to(device=tiles.device, dtype=torch.long)
    if torch.unique(permutation).numel() != count or torch.any(permutation < 0) or torch.any(
        permutation >= count
    ):
        raise ValueError("permutation must contain each tile index exactly once")
    return permutation


def _permuted_outputs(
    model: MacroPartitionNet, tiles: Tensor, permutation: Tensor
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Evaluate reference and shuffled inputs while preserving train/eval state."""
    was_training = model.training
    try:
        model.eval()
        reference = model(tiles)
        shuffled = model(tiles[:, permutation])
    finally:
        model.train(was_training)
    return reference, shuffled


@torch.no_grad()
def permutation_equivariance_error(
    model: MacroPartitionNet,
    tiles: Tensor,
    permutation: Tensor | None = None,
) -> dict[str, float]:
    """Measure the tile-axis equivariance error of a model in evaluation mode.

    ``permutation`` indexes the *input* tile axis.  A correct model has
    ``model(tiles[:, permutation])[key] == model(tiles)[key][:, permutation]``
    for both returned tensors, up to normal floating-point roundoff.
    """
    permutation = _checked_permutation(tiles, permutation)
    reference, shuffled = _permuted_outputs(model, tiles, permutation)

    return {
        key: float((shuffled[key] - reference[key][:, permutation]).abs().max().item())
        for key in ("assignment_logits", "tile_tokens")
    }


@torch.no_grad()
def check_permutation_equivariance(
    model: MacroPartitionNet,
    tiles: Tensor,
    permutation: Tensor | None = None,
    *,
    atol: float = 2e-5,
    rtol: float = 2e-5,
) -> dict[str, float]:
    """Assert the model's documented input-order equivariance and return errors."""
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    permutation = _checked_permutation(tiles, permutation)
    reference, shuffled = _permuted_outputs(model, tiles, permutation)
    expected = {key: reference[key][:, permutation] for key in reference}
    errors = {
        key: float((shuffled[key] - expected[key]).abs().max().item())
        for key in ("assignment_logits", "tile_tokens")
    }
    failing = {
        key: error
        for key, error in errors.items()
        if not torch.allclose(shuffled[key], expected[key], atol=atol, rtol=rtol)
    }
    if failing:
        raise AssertionError(f"tile-axis permutation equivariance failed: {failing}")
    return errors


def smoke(
    batch_size: int = 1, device: torch.device | str = "cpu"
) -> dict[str, tuple[int, ...]]:
    """Run default shape and permutation-equivariance checks."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model = MacroPartitionNet().to(device).eval()
    tiles = torch.rand(batch_size, 576, 3, 20, 20, device=device)
    with torch.no_grad():
        output = model(tiles)

    expected = {
        "assignment_logits": (batch_size, 576, 36),
        "tile_tokens": (batch_size, 576, 96),
    }
    shapes = {name: tuple(value.shape) for name, value in output.items()}
    if shapes != expected:
        raise AssertionError(f"unexpected MacroPartitionNet smoke shapes: {shapes}")
    check_permutation_equivariance(model, tiles)
    return shapes


__all__ = [
    "MacroPartitionNet",
    "check_permutation_equivariance",
    "count_params",
    "permutation_equivariance_error",
    "smoke",
]


if __name__ == "__main__":
    print(smoke())
    print("parameters:", count_params(MacroPartitionNet()))
