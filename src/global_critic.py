"""Whole-canvas structural critic and exact synthetic hard-negative utilities.

This module is deliberately *not* a puzzle solver.  It provides a scalar
energy-like score for a proposed 24x24 arrangement and the hard perturbations
needed to test whether that score contains information beyond a cheap colour
continuity heuristic.  A candidate arrangement is represented as
``(B, 576, 3, 20, 20)`` in its proposed row-major board order.

The critic combines three signals:

* a shared, per-tile encoder after per-tile luminance/contrast normalization;
* oriented descriptors of every proposed horizontal and vertical seam; and
* dilated convolution over the full 24x24 grid, so a local anomaly can be
  interpreted in the context of a much larger natural-image structure.

It has no tile ID, input-order feature, true coordinate, or recovered-label
input.  Exact synthetic labels are used only outside the model to construct a
correct board and hard negative permutations.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import FS, GRID, NFRAG


# The first four families are intentionally close to a valid arrangement.  A
# score that only separates random shuffles is not useful for reconstruction.
HARD_NEAR_FAMILIES: Final[tuple[str, ...]] = (
    "adjacent_swap",
    "nearby_swap",
    "patch_shuffle_3",
    "block_swap_2",
)
MACRO_FAMILIES: Final[tuple[str, ...]] = ("macro_swap_4", "macro_swap_6")
EVAL_NEGATIVE_FAMILIES: Final[tuple[str, ...]] = (
    *HARD_NEAR_FAMILIES,
    *MACRO_FAMILIES,
    "random_permutation",
)
# Repeating the most difficult single-swap family during training prevents the
# easy random-shuffle class from dominating the ranking loss.
TRAIN_NEGATIVE_FAMILIES: Final[tuple[str, ...]] = (
    "adjacent_swap",
    "adjacent_swap",
    "nearby_swap",
    "patch_shuffle_3",
    "block_swap_2",
    "macro_swap_4",
    "random_permutation",
)

NEGATIVE_DESCRIPTIONS: Final[dict[str, str]] = {
    "adjacent_swap": "one horizontal or vertical adjacent tile swap",
    "nearby_swap": "one non-adjacent but local tile swap (Chebyshev 2..4)",
    "patch_shuffle_3": "a shuffled contiguous 3x3 patch",
    "block_swap_2": "two intact 2x2 components swapped",
    "macro_swap_4": "two intact 4x4 macro-components swapped",
    "macro_swap_6": "two intact 6x6 macro-components swapped",
    "random_permutation": "a full random board permutation",
}

_FAMILY_ALIASES: Final[dict[str, str]] = {
    "block_swap_4": "macro_swap_4",
    "block_swap_6": "macro_swap_6",
    "component_swap_4": "macro_swap_4",
    "component_swap_6": "macro_swap_6",
}


def _groups(channels: int, maximum: int = 8) -> int:
    """Return a small GroupNorm group count that divides ``channels``."""
    for groups in range(min(int(channels), maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _check_tiles(tiles: Tensor) -> tuple[int, int, int, int, int]:
    """Validate the fixed puzzle tensor contract and return its shape."""
    if tiles.ndim != 5:
        raise ValueError(
            f"tiles must have shape (B,{NFRAG},3,{FS},{FS}), got {tuple(tiles.shape)}"
        )
    batch, count, channels, height, width = (int(value) for value in tiles.shape)
    if (count, channels, height, width) != (NFRAG, 3, FS, FS):
        raise ValueError(
            f"tiles must have shape (B,{NFRAG},3,{FS},{FS}), got {tuple(tiles.shape)}"
        )
    if batch < 1:
        raise ValueError("tiles batch must be non-empty")
    return batch, count, channels, height, width


def normalize_tiles(tiles: Tensor, *, eps: float = 1.0e-4) -> Tensor:
    """Cancel independent brightness/contrast while retaining tile chroma.

    The challenge applies a mostly scalar brightness/contrast perturbation to
    every tile.  Centering and scaling over all RGB pixels makes the learned
    seam branch much less likely to learn that nuisance.  Raw per-channel
    means and standard deviations are still fed separately to the grid branch
    because broad colour continuity is a legitimate structural cue.
    """
    _check_tiles(tiles)
    mean = tiles.mean(dim=(-1, -2, -3), keepdim=True)
    rms = tiles.sub(mean).square().mean(dim=(-1, -2, -3), keepdim=True).add(eps).sqrt()
    return tiles.sub(mean).div(rms)


def tile_mean_tv_energy(tiles: Tensor) -> Tensor:
    """Return tile-mean total variation energy for each proposed board.

    This is the deliberately weak hand-crafted baseline requested for the
    global critic gate.  Lower energy is preferred, so
    :func:`tile_mean_tv_score` simply negates this value.  It is a *ranking
    diagnostic only*: prior bounded optimization showed that minimizing this
    energy from an arbitrary board creates colour clusters rather than the
    true puzzle.
    """
    batch, _, _, _, _ = _check_tiles(tiles)
    means = tiles.float().mean(dim=(-1, -2)).reshape(batch, GRID, GRID, 3)
    horizontal = means[:, :, 1:].sub(means[:, :, :-1]).abs().mean(dim=(1, 2, 3))
    vertical = means[:, 1:].sub(means[:, :-1]).abs().mean(dim=(1, 2, 3))
    return 0.5 * (horizontal + vertical)


def tile_mean_tv_score(tiles: Tensor) -> Tensor:
    """Higher-is-better version of :func:`tile_mean_tv_energy`."""
    return tile_mean_tv_energy(tiles).neg()


def _canonical_family(family: str) -> str:
    family = _FAMILY_ALIASES.get(str(family), str(family))
    if family not in NEGATIVE_DESCRIPTIONS:
        choices = ", ".join(sorted(NEGATIVE_DESCRIPTIONS))
        raise ValueError(f"unknown negative family {family!r}; choose one of {choices}")
    return family


def _flat_block(row: int, col: int, size: int) -> np.ndarray:
    rows = np.arange(row, row + size, dtype=np.int64)[:, None]
    cols = np.arange(col, col + size, dtype=np.int64)[None, :]
    return (rows * GRID + cols).reshape(-1)


def _sample_two_nonoverlapping_blocks(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if size < 1 or size > GRID:
        raise ValueError(f"invalid block size {size}")
    first_row = int(rng.integers(0, GRID - size + 1))
    first_col = int(rng.integers(0, GRID - size + 1))
    first = _flat_block(first_row, first_col, size)
    first_mask = np.zeros(NFRAG, dtype=bool)
    first_mask[first] = True
    # 24x24 gives ample room even to choose two non-overlapping 6x6 blocks.
    for _ in range(256):
        second_row = int(rng.integers(0, GRID - size + 1))
        second_col = int(rng.integers(0, GRID - size + 1))
        second = _flat_block(second_row, second_col, size)
        if not bool(first_mask[second].any()):
            return first, second
    raise RuntimeError("failed to sample non-overlapping blocks")


def _sample_nearby_pair(rng: np.random.Generator) -> tuple[int, int]:
    """Pick a pair that is local but explicitly not a true cardinal neighbour."""
    for _ in range(1024):
        row = int(rng.integers(0, GRID))
        col = int(rng.integers(0, GRID))
        delta_row = int(rng.integers(-4, 5))
        delta_col = int(rng.integers(-4, 5))
        chebyshev = max(abs(delta_row), abs(delta_col))
        if not 2 <= chebyshev <= 4:
            continue
        other_row, other_col = row + delta_row, col + delta_col
        if 0 <= other_row < GRID and 0 <= other_col < GRID:
            return row * GRID + col, other_row * GRID + other_col
    raise RuntimeError("failed to sample nearby non-adjacent tile pair")


def sample_negative_order(family: str, rng: np.random.Generator) -> np.ndarray:
    """Return ``board_position -> source-correct-position`` for one hard negative.

    The order can be applied directly as ``correct_tiles[order]``.  All moves
    preserve every tile's orientation and use exactly the same bag as the
    positive board, eliminating colour/quality class shortcuts.
    """
    family = _canonical_family(family)
    order = np.arange(NFRAG, dtype=np.int64)

    if family == "adjacent_swap":
        horizontal = bool(rng.integers(0, 2))
        if horizontal:
            row = int(rng.integers(0, GRID))
            col = int(rng.integers(0, GRID - 1))
            first, second = row * GRID + col, row * GRID + col + 1
        else:
            row = int(rng.integers(0, GRID - 1))
            col = int(rng.integers(0, GRID))
            first, second = row * GRID + col, (row + 1) * GRID + col
        order[first], order[second] = order[second], order[first]
    elif family == "nearby_swap":
        first, second = _sample_nearby_pair(rng)
        order[first], order[second] = order[second], order[first]
    elif family == "patch_shuffle_3":
        row = int(rng.integers(0, GRID - 2))
        col = int(rng.integers(0, GRID - 2))
        cells = _flat_block(row, col, 3)
        shuffled = rng.permutation(cells)
        if np.array_equal(shuffled, cells):
            shuffled = np.roll(shuffled, 1)
        order[cells] = shuffled
    elif family == "block_swap_2":
        first, second = _sample_two_nonoverlapping_blocks(2, rng)
        saved = order[first].copy()
        order[first] = order[second]
        order[second] = saved
    elif family == "macro_swap_4":
        first, second = _sample_two_nonoverlapping_blocks(4, rng)
        saved = order[first].copy()
        order[first] = order[second]
        order[second] = saved
    elif family == "macro_swap_6":
        first, second = _sample_two_nonoverlapping_blocks(6, rng)
        saved = order[first].copy()
        order[first] = order[second]
        order[second] = saved
    elif family == "random_permutation":
        order = rng.permutation(NFRAG).astype(np.int64, copy=False)
        if np.array_equal(order, np.arange(NFRAG, dtype=np.int64)):
            order[[0, 1]] = order[[1, 0]]
    else:  # ``_canonical_family`` keeps this branch unreachable.
        raise AssertionError(f"unhandled family {family}")

    if np.array_equal(order, np.arange(NFRAG, dtype=np.int64)):
        raise AssertionError(f"{family} unexpectedly produced identity order")
    if not np.array_equal(np.sort(order), np.arange(NFRAG, dtype=np.int64)):
        raise AssertionError(f"{family} did not produce a permutation")
    return order


def sample_negative_orders(
    batch_size: int,
    families: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a ``(B, K, 576)`` bank of hard-negative board orders."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not families:
        raise ValueError("families must be non-empty")
    canonical = tuple(_canonical_family(family) for family in families)
    result = np.empty((batch_size, len(canonical), NFRAG), dtype=np.int64)
    for batch_index in range(batch_size):
        for negative_index, family in enumerate(canonical):
            result[batch_index, negative_index] = sample_negative_order(family, rng)
    return result


def apply_orders(tiles: Tensor, orders: Tensor) -> Tensor:
    """Apply batched board-position orders to a batch of correct boards.

    ``tiles`` is ``(B,576,3,20,20)`` and ``orders`` is ``(B,K,576)``.  The
    returned arrangements have shape ``(B,K,576,3,20,20)``.
    """
    batch, _, channels, height, width = _check_tiles(tiles)
    if orders.ndim != 3 or tuple(orders.shape[:1]) != (batch,) or int(orders.shape[2]) != NFRAG:
        raise ValueError(
            f"orders must have shape (B,K,{NFRAG}) matching tiles, got {tuple(orders.shape)}"
        )
    if orders.numel() == 0:
        raise ValueError("orders must contain at least one negative per board")
    if orders.device != tiles.device:
        raise ValueError("orders and tiles must be on the same device")
    if torch.any(orders < 0) or torch.any(orders >= NFRAG):
        raise ValueError("orders contain an out-of-range tile index")
    # Advanced indexing keeps the implementation simple and does not make an
    # N^2 score matrix: only K complete permutations are materialized.
    batch_index = torch.arange(batch, device=tiles.device)[:, None, None].expand_as(orders)
    arranged = tiles[batch_index, orders.long()]
    return arranged.reshape(batch, int(orders.shape[1]), NFRAG, channels, height, width)


def correct_boards_from_shuffled(shuffled_tiles: Tensor, perm: Tensor) -> Tensor:
    """Undo a synthetic input shuffle using its exact supervision label only.

    ``perm[input_tile] == clean row-major board position`` is produced by
    :class:`canvas_data.CanvasDataset` only for fresh synthetic samples.  The
    returned board is used to create positive and negative *training* examples;
    the neural critic never receives ``perm``.
    """
    batch, _, _, _, _ = _check_tiles(shuffled_tiles)
    if perm.ndim != 2 or tuple(perm.shape) != (batch, NFRAG):
        raise ValueError(f"perm must have shape (B,{NFRAG}), got {tuple(perm.shape)}")
    if perm.device != shuffled_tiles.device:
        raise ValueError("perm and shuffled_tiles must be on the same device")
    inverse = torch.argsort(perm.long(), dim=1)
    batch_index = torch.arange(batch, device=shuffled_tiles.device)[:, None]
    return shuffled_tiles[batch_index, inverse]


class _ResidualGridBlock(nn.Module):
    """Dilated residual block over the proposed 24x24 arrangement."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        groups = _groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(F.gelu(self.norm2(x)))
        return F.gelu(x + residual)


class _TileEncoder(nn.Module):
    """Small shared tile encoder; it never sees tile identity or coordinates."""

    def __init__(self, tile_width: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        middle = tile_width * 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, tile_width, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(tile_width), tile_width),
            nn.GELU(),
            nn.Conv2d(tile_width, middle, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            nn.Conv2d(middle, middle, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
            nn.Conv2d(middle, middle, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(middle, embedding_dim),
        )

    def forward(self, tiles: Tensor) -> Tensor:
        return self.proj(self.body(self.stem(tiles)))


class _EdgeEncoder(nn.Module):
    """Shared descriptor for top/bottom/left/right normalized edge bands."""

    def __init__(self, edge_width: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = edge_width * 2
        self.net = nn.Sequential(
            nn.Conv2d(3, edge_width, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(edge_width), edge_width),
            nn.GELU(),
            nn.Conv2d(edge_width, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden, edge_dim),
        )

    def forward(self, strips: Tensor) -> Tensor:
        return self.net(strips)


class GlobalStructuralCritic(nn.Module):
    """Score a complete proposed tile canvas; higher means more structurally plausible.

    The model intentionally has no absolute coordinate embedding.  The only
    spatial information arises from the candidate board arrangement itself and
    normal convolutional boundary effects, which is the same information an
    eventual optimizer would be allowed to query.
    """

    def __init__(
        self,
        *,
        tile_size: int = FS,
        tile_width: int = 24,
        embedding_dim: int = 64,
        edge_width: int = 16,
        edge_dim: int = 32,
        grid_width: int = 80,
        stats_dim: int = 16,
        edge_band: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 12, 4),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if tile_size != FS:
            raise ValueError(f"this fixed puzzle implementation expects tile_size={FS}")
        positive = {
            "tile_width": tile_width,
            "embedding_dim": embedding_dim,
            "edge_width": edge_width,
            "edge_dim": edge_dim,
            "grid_width": grid_width,
            "stats_dim": stats_dim,
            "edge_band": edge_band,
        }
        if any(int(value) <= 0 for value in positive.values()):
            raise ValueError(f"all architectural widths must be positive, got {positive}")
        if edge_band > FS:
            raise ValueError(f"edge_band must be <= tile size {FS}")
        if not dilations or any(int(value) <= 0 for value in dilations):
            raise ValueError("dilations must be a non-empty sequence of positive integers")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")

        self.tile_size = int(tile_size)
        self.tile_width = int(tile_width)
        self.embedding_dim = int(embedding_dim)
        self.edge_width = int(edge_width)
        self.edge_dim = int(edge_dim)
        self.grid_width = int(grid_width)
        self.stats_dim = int(stats_dim)
        self.edge_band = int(edge_band)
        self.dilations = tuple(int(value) for value in dilations)
        self.dropout = float(dropout)

        self.tile_encoder = _TileEncoder(self.tile_width, self.embedding_dim, self.dropout)
        self.edge_encoder = _EdgeEncoder(self.edge_width, self.edge_dim, self.dropout)
        pair_hidden = max(self.edge_dim * 2, self.grid_width)
        self.pair_fuse = nn.Sequential(
            nn.Linear(self.edge_dim * 4, pair_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity(),
            nn.Linear(pair_hidden, self.grid_width),
        )
        self.stats_project = nn.Sequential(
            nn.Linear(6, self.stats_dim),
            nn.GELU(),
            nn.Linear(self.stats_dim, self.stats_dim),
        )
        grid_input = self.embedding_dim + self.stats_dim + self.grid_width
        self.grid_stem = nn.Sequential(
            nn.Conv2d(grid_input, self.grid_width, kernel_size=1),
            nn.GroupNorm(_groups(self.grid_width), self.grid_width),
            nn.GELU(),
        )
        self.grid_blocks = nn.ModuleList(
            _ResidualGridBlock(self.grid_width, dilation, self.dropout) for dilation in self.dilations
        )
        self.local_head = nn.Conv2d(self.grid_width, 1, kernel_size=1)
        self.global_head = nn.Sequential(
            nn.LayerNorm(self.grid_width * 3),
            nn.Linear(self.grid_width * 3, self.grid_width),
            nn.GELU(),
            nn.Linear(self.grid_width, 1),
        )

    @property
    def model_kwargs(self) -> dict[str, object]:
        """Architecture metadata suitable for a self-describing checkpoint."""
        return {
            "tile_size": self.tile_size,
            "tile_width": self.tile_width,
            "embedding_dim": self.embedding_dim,
            "edge_width": self.edge_width,
            "edge_dim": self.edge_dim,
            "grid_width": self.grid_width,
            "stats_dim": self.stats_dim,
            "edge_band": self.edge_band,
            "dilations": self.dilations,
            "dropout": self.dropout,
        }

    def _edge_descriptors(self, normalized: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode oriented edge bands as ``(B,576,edge_dim)`` descriptors."""
        batch, count, channels, _, _ = _check_tiles(normalized)
        band = self.edge_band
        top = normalized[:, :, :, :band, :]
        bottom = normalized[:, :, :, -band:, :]
        # Rotating vertical bands into (band, edge-length) makes one shared
        # edge encoder applicable while preserving top-to-bottom orientation.
        left = normalized[:, :, :, :, :band].transpose(-1, -2)
        right = normalized[:, :, :, :, -band:].transpose(-1, -2)

        def encode(value: Tensor) -> Tensor:
            flat = value.reshape(batch * count, channels, band, FS).contiguous()
            return self.edge_encoder(flat).reshape(batch, count, self.edge_dim)

        return encode(top), encode(bottom), encode(left), encode(right)

    def _pair_features(self, first: Tensor, second: Tensor) -> Tensor:
        """Build an oriented learned compatibility feature for a proposed seam."""
        joined = torch.cat((first, second, first - second, first * second), dim=-1)
        return self.pair_fuse(joined)

    def forward(self, tiles: Tensor) -> Tensor:
        """Return one finite higher-is-better structural score per board."""
        batch, count, channels, height, width = _check_tiles(tiles)
        raw = tiles.float()
        # These six raw values are an intentionally transparent route for broad
        # colour continuity; the detailed image branch below is normalized.
        stats = torch.cat(
            (raw.mean(dim=(-1, -2)), raw.std(dim=(-1, -2), unbiased=False)), dim=-1
        )
        normalized = normalize_tiles(raw)
        embeddings = self.tile_encoder(
            normalized.reshape(batch * count, channels, height, width)
        ).reshape(batch, count, self.embedding_dim)
        top, bottom, left, right = self._edge_descriptors(normalized)

        tile_grid = embeddings.reshape(batch, GRID, GRID, self.embedding_dim).permute(0, 3, 1, 2)
        stat_grid = self.stats_project(stats).reshape(batch, GRID, GRID, self.stats_dim).permute(0, 3, 1, 2)
        top = top.reshape(batch, GRID, GRID, self.edge_dim)
        bottom = bottom.reshape(batch, GRID, GRID, self.edge_dim)
        left = left.reshape(batch, GRID, GRID, self.edge_dim)
        right = right.reshape(batch, GRID, GRID, self.edge_dim)

        # right(i,j) -> left(i,j+1); bottom(i,j) -> top(i+1,j).
        horizontal = self._pair_features(right[:, :, :-1], left[:, :, 1:]).permute(0, 3, 1, 2)
        vertical = self._pair_features(bottom[:, :-1], top[:, 1:]).permute(0, 3, 1, 2)
        # Give both endpoints access to an edge feature.  The additive form
        # preserves how many edges are incident at a boundary without adding a
        # separate absolute-position embedding.
        edge_grid = (
            F.pad(horizontal, (0, 1, 0, 0))
            + F.pad(horizontal, (1, 0, 0, 0))
            + F.pad(vertical, (0, 0, 0, 1))
            + F.pad(vertical, (0, 0, 1, 0))
        )

        state = self.grid_stem(torch.cat((tile_grid, stat_grid, edge_grid), dim=1))
        for block in self.grid_blocks:
            state = block(state)
        local = self.local_head(state).mean(dim=(1, 2, 3))
        average = state.mean(dim=(2, 3))
        maximum = state.amax(dim=(2, 3))
        minimum = state.amin(dim=(2, 3))
        global_score = self.global_head(torch.cat((average, maximum, minimum), dim=1)).squeeze(1)
        score = local + global_score
        if not torch.isfinite(score).all():
            raise FloatingPointError("GlobalStructuralCritic produced a non-finite score")
        return score


def count_params(model: nn.Module) -> int:
    """Return trainable parameter count for concise experiment logging."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def smoke(device: torch.device | None = None) -> None:
    """Data-free contract smoke test, including a backward pass on hard negatives."""
    device = device or torch.device("cpu")
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    model = GlobalStructuralCritic(
        tile_width=8,
        embedding_dim=16,
        edge_width=8,
        edge_dim=12,
        grid_width=24,
        stats_dim=8,
        dilations=(1, 2),
        dropout=0.0,
    ).to(device)
    tiles = torch.rand(1, NFRAG, 3, FS, FS, device=device)
    orders = torch.from_numpy(
        sample_negative_orders(1, ("adjacent_swap", "macro_swap_4"), rng)
    ).to(device)
    negatives = apply_orders(tiles, orders)
    scores = model(torch.cat((tiles, negatives.reshape(-1, NFRAG, 3, FS, FS)), dim=0))
    if tuple(scores.shape) != (3,) or not torch.isfinite(scores).all():
        raise AssertionError("critic score shape or finiteness contract failed")
    loss = F.softplus(0.10 + scores[1:] - scores[:1]).mean()
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("critic smoke backward produced no gradients")
    tv = tile_mean_tv_score(torch.cat((tiles, negatives.reshape(-1, NFRAG, 3, FS, FS)), dim=0))
    if tuple(tv.shape) != (3,) or not torch.isfinite(tv).all():
        raise AssertionError("TV baseline contract failed")


if __name__ == "__main__":
    smoke()
    print("global_critic smoke: OK", flush=True)
