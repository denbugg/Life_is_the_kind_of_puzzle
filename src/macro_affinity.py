"""Compact full-tile embedding model for coarse puzzle grouping.

``MacroAffinityNet`` deliberately describes the *contents* of each fragment,
rather than its four seams.  It is intended for a first, coarse stage such as
identifying which 4 x 4 macro-region a tile is likely to belong to.  A shared
CNN sees every pixel of every 20 x 20 tile and mixes pooled appearance and
texture features with an optional, cheap colour-statistics branch.

The module has no input-order features or cross-tile attention.  Consequently
``embed`` is exactly equivariant to a permutation of the input tile axis and
does not allocate a quadratic ``N x N`` tensor unless the caller explicitly
asks :meth:`forward` for affinities.  This keeps contrastive training on a
576-tile bag comfortable on an 8 GB GPU.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


IntPair = Union[int, tuple[int, int]]


def _pair(value: IntPair, name: str) -> tuple[int, int]:
    """Normalize and validate a square-or-pair spatial configuration."""
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
    """A small GroupNorm residual block suited to small image batches."""

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


class _TileBackbone(nn.Module):
    """Shared CNN producing pooled appearance and texture features per tile."""

    def __init__(self, embedding_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        middle = max(width, width * 2)
        final = max(embedding_dim, middle)

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
        )
        self.block1 = _ResidualConv(width)
        self.down1 = nn.Sequential(
            nn.Conv2d(width, middle, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
        )
        self.block2 = _ResidualConv(middle)
        self.down2 = nn.Sequential(
            nn.Conv2d(middle, final, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_groups(final), final),
            nn.GELU(),
        )
        self.block3 = _ResidualConv(final)

        # Mean pools broad semantic/color evidence while the channelwise
        # spatial standard deviation retains texture density.  Both operate
        # across the complete tile; no edge-only descriptor is used here.
        self.head = nn.Sequential(
            nn.LayerNorm(2 * final),
            nn.Linear(2 * final, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, tiles: Tensor) -> Tensor:
        x = self.block1(self.stem(tiles))
        x = self.block2(self.down1(x))
        x = self.block3(self.down2(x))
        spatial = x.flatten(start_dim=2)
        pooled = torch.cat(
            (
                spatial.mean(dim=-1),
                spatial.var(dim=-1, unbiased=False).add(1.0e-6).sqrt(),
            ),
            dim=-1,
        )
        return self.head(pooled)


class _ColourStats(nn.Module):
    """A small full-tile colour branch complementary to CNN texture features."""

    FEATURES = 12  # RGB mean, standard deviation, minimum, and maximum.

    def __init__(self, embedding_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(self.FEATURES),
            nn.Linear(self.FEATURES, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, tiles: Tensor) -> Tensor:
        # Each statistic spans the entire 20 x 20 fragment.  Min/max add a
        # little exposure/colour-range information at essentially no cost.
        dimensions = (-2, -1)
        stats = torch.cat(
            (
                tiles.mean(dim=dimensions),
                tiles.var(dim=dimensions, unbiased=False).add(1.0e-6).sqrt(),
                tiles.amin(dim=dimensions),
                tiles.amax(dim=dimensions),
            ),
            dim=1,
        )
        return self.net(stats)


class MacroAffinityNet(nn.Module):
    """Embed unordered RGB tiles for semantic/texture macro-region affinity.

    Args:
        tiles: Optional required number of tiles in an input bag.  Leave it as
            ``None`` (the default) to embed any number of tiles, including
            compact evaluation subsets.
        tile_size: Expected raw tile size.  The standard puzzle uses ``20``.
        embedding_dim: Size of the normalized output descriptor.
        d: Backwards-friendly alias for ``embedding_dim``.  When both are
            supplied they must agree (unless ``embedding_dim`` is its default).
        width: Base CNN channel count.  The default is below one million
            parameters and avoids all cross-tile attention.
        use_stats: Fuse full-tile RGB mean/spread/range statistics into the
            learned CNN descriptor.
        stats_hidden: Width of the optional statistics MLP.
        dropout: Dropout used only in projection heads.

    ``embed(tiles)`` accepts float tensors shaped ``(B, N, 3, H, W)`` and
    returns unit-L2 embeddings shaped ``(B, N, embedding_dim)``.  The output
    stays in the same tile order as the input.  It is therefore suitable for a
    supervised contrastive, triplet, or macro-membership objective.
    """

    def __init__(
        self,
        *,
        tiles: int | None = None,
        tile_size: IntPair = 20,
        embedding_dim: int = 128,
        d: int | None = None,
        width: int = 48,
        use_stats: bool = True,
        stats_hidden: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d is not None:
            if embedding_dim != 128 and embedding_dim != d:
                raise ValueError(
                    "embedding_dim and d disagree; specify one descriptor dimension"
                )
            embedding_dim = int(d)

        self.tiles = None if tiles is None else int(tiles)
        self.tile_size = _pair(tile_size, "tile_size")
        self.embedding_dim = int(embedding_dim)
        # Existing model modules conventionally call this field ``d``.
        self.d = self.embedding_dim
        self.use_stats = bool(use_stats)

        if self.tiles is not None and self.tiles <= 0:
            raise ValueError("tiles must be positive when specified")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        if stats_hidden <= 0:
            raise ValueError("stats_hidden must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.backbone = _TileBackbone(self.embedding_dim, int(width), float(dropout))
        if self.use_stats:
            self.stats = _ColourStats(self.embedding_dim, int(stats_hidden), float(dropout))
            self.fuse = nn.Sequential(
                nn.LayerNorm(2 * self.embedding_dim),
                nn.Linear(2 * self.embedding_dim, self.embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
            )
        else:
            self.stats = None
            self.fuse = None

    def _check_tiles(self, tiles: Tensor) -> None:
        if tiles.ndim != 5:
            raise ValueError(
                "tiles must have shape (batch, tiles, 3, height, width), "
                f"got {tuple(tiles.shape)}"
            )
        if self.tiles is not None and tiles.shape[1] != self.tiles:
            raise ValueError(f"expected {self.tiles} tiles, got {tiles.shape[1]}")
        if tiles.shape[2] != 3:
            raise ValueError(f"MacroAffinityNet expects RGB tiles, got {tiles.shape[2]} channels")
        if tuple(tiles.shape[-2:]) != self.tile_size:
            raise ValueError(
                f"expected tile size {self.tile_size}, got {tuple(tiles.shape[-2:])}"
            )
        if not torch.is_floating_point(tiles):
            raise TypeError(f"tiles must be floating point, got {tiles.dtype}")

    def embed(self, tiles: Tensor) -> Tensor:
        """Return a normalized ``(B, N, D)`` embedding for every input tile."""
        self._check_tiles(tiles)
        batch, count, channels, height, width = tiles.shape
        flat = tiles.reshape(batch * count, channels, height, width)
        descriptor = self.backbone(flat)
        if self.stats is not None:
            descriptor = self.fuse(torch.cat((descriptor, self.stats(flat)), dim=-1))
        # Epsilon protects the contract even with exotic custom initializers or
        # degenerate all-zero inputs.  In normal operation every row has norm 1.
        descriptor = F.normalize(descriptor, p=2, dim=-1, eps=1.0e-6)
        return descriptor.reshape(batch, count, self.embedding_dim)

    def affinity(self, embeddings: Tensor) -> Tensor:
        """Return unscaled cosine similarities for a ``(B, N, D)`` embedding bag."""
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                "embeddings must have shape "
                f"(batch, tiles, {self.embedding_dim}), got {tuple(embeddings.shape)}"
            )
        # Re-normalize to make this helper safe with descriptors supplied by a
        # caller, while avoiding an implicit large matrix in ``embed`` itself.
        normalized = F.normalize(embeddings, p=2, dim=-1, eps=1.0e-6)
        return normalized @ normalized.transpose(-1, -2)

    def forward(self, tiles: Tensor, *, return_affinity: bool = False) -> Tensor | dict[str, Tensor]:
        """Embed a bag, optionally also returning its quadratic cosine matrix.

        ``return_affinity=False`` is intentionally the default: contrastive
        losses generally need only selected pairs and can avoid materializing
        the full ``N x N`` matrix for every 576-tile input.
        """
        embeddings = self.embed(tiles)
        if return_affinity:
            return {"embeddings": embeddings, "affinity": self.affinity(embeddings)}
        return embeddings


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def smoke(
    batch_size: int = 1, device: torch.device | str = "cpu"
) -> dict[str, tuple[int, ...]]:
    """Run a compact shape, normalization, and optional-affinity smoke test."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model = MacroAffinityNet().to(device).eval()
    tiles = torch.rand(batch_size, 576, 3, 20, 20, device=device)
    with torch.no_grad():
        embeddings = model.embed(tiles)
        output = model(tiles, return_affinity=True)

    expected = {
        "embeddings": (batch_size, 576, 128),
        "affinity": (batch_size, 576, 576),
    }
    shapes = {
        "embeddings": tuple(embeddings.shape),
        "affinity": tuple(output["affinity"].shape),
    }
    if shapes != expected:
        raise AssertionError(f"unexpected MacroAffinityNet smoke shapes: {shapes}")
    norms = embeddings.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=3.0e-5, rtol=3.0e-5):
        raise AssertionError("MacroAffinityNet embeddings are not L2-normalized")
    return shapes


__all__ = ["MacroAffinityNet", "count_params", "smoke"]


if __name__ == "__main__":
    print(smoke())
    print("parameters:", count_params(MacroAffinityNet()))
