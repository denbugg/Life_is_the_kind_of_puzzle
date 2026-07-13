"""Raw-tile layout plausibility model and target-free local refinement.

The model consumes a *proposed* ordered grid of independently corrupted tiles.
It never requires a denoiser or clean target at inference.  Lower global energy
means a more plausible layout.  A local head predicts incorrect positions and
a normalized displacement vector towards each tile's likely destination.

The architecture stays bounded at 24x24: a shared multi-scale tile CNN is
followed by window attention, depthwise spatial mixing across window borders,
and a small set of global cross-attention tokens.  It avoids dense global
tile-to-tile attention while preserving a whole-image plausibility signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


@dataclass(frozen=True)
class LayoutEnergyConfig:
    grid_size: int = 24
    tile_size: int = 20
    d_model: int = 256
    num_heads: int = 8
    local_layers: int = 6
    window_size: int = 6
    global_layers: int = 2
    global_tokens: int = 6
    feedforward_dim: int = 1024
    cnn_channels: int = 64
    edge_dim: int = 48
    edge_band: int = 3
    move_dim: int = 64
    dropout: float = 0.10

    @property
    def tile_count(self) -> int:
        return self.grid_size * self.grid_size

    def validate(self) -> None:
        integer_fields = {
            "grid_size": self.grid_size,
            "tile_size": self.tile_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "local_layers": self.local_layers,
            "window_size": self.window_size,
            "global_layers": self.global_layers,
            "global_tokens": self.global_tokens,
            "feedforward_dim": self.feedforward_dim,
            "cnn_channels": self.cnn_channels,
            "edge_dim": self.edge_dim,
            "edge_band": self.edge_band,
            "move_dim": self.move_dim,
        }
        if any(type(value) is not int or value <= 0 for value in integer_fields.values()):
            raise ValueError(f"positive integer configuration required: {integer_fields}")
        if self.d_model % self.num_heads:
            raise ValueError("num_heads must divide d_model")
        if self.grid_size % self.window_size:
            raise ValueError("window_size must divide grid_size")
        if self.edge_band > self.tile_size:
            raise ValueError("edge_band cannot exceed tile_size")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


class _ConvResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(_groups(channels), channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + F.silu(self.norm(self.pointwise(self.depthwise(values))))


class MultiScaleTileEncoder(nn.Module):
    """Encode tile interiors at two scales plus all four physical borders."""

    def __init__(self, config: LayoutEnergyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        channels = config.cnn_channels
        self.fine = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            _ConvResidual(channels),
        )
        self.coarse = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(channels * 2), channels * 2),
            nn.SiLU(),
            _ConvResidual(channels * 2),
            nn.AdaptiveAvgPool2d(1),
        )
        edge_input = 3 * config.edge_band * config.tile_size
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_input),
            nn.Linear(edge_input, config.edge_dim * 2),
            nn.GELU(),
            nn.Linear(config.edge_dim * 2, config.edge_dim),
        )
        feature_dim = channels * 4 + channels * 2 + 4 * config.edge_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )

    def _edges(self, tiles: torch.Tensor) -> torch.Tensor:
        band = self.config.edge_band
        # Orient every border along its natural left-to-right/top-to-bottom axis.
        edges = (
            tiles[:, :, :band, :],
            tiles[:, :, :, -band:].transpose(2, 3),
            tiles[:, :, -band:, :],
            tiles[:, :, :, :band].transpose(2, 3),
        )
        return torch.cat(
            [self.edge_encoder(edge.contiguous().flatten(1)) for edge in edges],
            dim=1,
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        expected = (
            self.config.tile_count,
            3,
            self.config.tile_size,
            self.config.tile_size,
        )
        if tiles.ndim != 5 or tuple(tiles.shape[1:]) != expected:
            raise ValueError(f"tiles must have shape Bx{expected}, got {tuple(tiles.shape)}")
        if not tiles.is_floating_point():
            raise TypeError("tiles must be floating point in [0,1]")
        batch, count = tiles.shape[:2]
        flat = tiles.reshape(-1, 3, self.config.tile_size, self.config.tile_size)
        fine = self.fine(flat)
        fine_pool = F.adaptive_avg_pool2d(fine, 2).flatten(1)
        coarse = self.coarse(fine).flatten(1)
        edges = self._edges(flat)
        encoded = self.projection(torch.cat([fine_pool, coarse, edges], dim=1))
        return encoded.reshape(batch, count, self.config.d_model)


class WindowSpatialBlock(nn.Module):
    """Local window attention plus convolutional mixing across window borders."""

    def __init__(self, config: LayoutEnergyConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.conv_norm = nn.LayerNorm(config.d_model)
        self.spatial_depthwise = nn.Conv2d(
            config.d_model,
            config.d_model,
            3,
            padding=1,
            groups=config.d_model,
        )
        self.spatial_pointwise = nn.Conv2d(config.d_model, config.d_model, 1)
        self.feedforward_norm = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def _windows(self, tiles: torch.Tensor) -> torch.Tensor:
        batch = tiles.shape[0]
        grid = self.config.grid_size
        window = self.config.window_size
        values = tiles.reshape(batch, grid, grid, self.config.d_model)
        return (
            values.reshape(
                batch,
                grid // window,
                window,
                grid // window,
                window,
                self.config.d_model,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, window * window, self.config.d_model)
        )

    def _unwindow(self, windows: torch.Tensor, batch: int) -> torch.Tensor:
        grid = self.config.grid_size
        window = self.config.window_size
        return (
            windows.reshape(
                batch,
                grid // window,
                grid // window,
                window,
                window,
                self.config.d_model,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, grid * grid, self.config.d_model)
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        batch = tiles.shape[0]
        windows = self._windows(self.pre_norm(tiles))
        attended, _ = self.attention(windows, windows, windows, need_weights=False)
        tiles = tiles + self.attention_dropout(self._unwindow(attended, batch))
        grid = self.config.grid_size
        spatial = self.conv_norm(tiles).reshape(
            batch, grid, grid, self.config.d_model
        ).permute(0, 3, 1, 2)
        spatial = self.spatial_pointwise(self.spatial_depthwise(spatial))
        tiles = tiles + spatial.permute(0, 2, 3, 1).reshape_as(tiles)
        return tiles + self.feedforward(self.feedforward_norm(tiles))


class GlobalTokenBlock(nn.Module):
    """Let a few learned tokens summarize the full grid and condition tiles."""

    def __init__(self, config: LayoutEnergyConfig) -> None:
        super().__init__()
        self.tile_norm = nn.LayerNorm(config.d_model)
        self.global_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.global_ff_norm = nn.LayerNorm(config.d_model)
        self.global_ff = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_model),
        )
        self.tile_condition = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.Sigmoid(),
        )
        self.tile_ff_norm = nn.LayerNorm(config.d_model)
        self.tile_ff = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_model),
        )

    def forward(
        self, tiles: torch.Tensor, global_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.global_norm(global_tokens)
        context = torch.cat([query, self.tile_norm(tiles)], dim=1)
        attended, _ = self.cross_attention(query, context, context, need_weights=False)
        global_tokens = global_tokens + attended
        global_tokens = global_tokens + self.global_ff(
            self.global_ff_norm(global_tokens)
        )
        summary = global_tokens.mean(dim=1)
        tiles = tiles + self.tile_condition(summary).unsqueeze(1) * summary.unsqueeze(1)
        tiles = tiles + self.tile_ff(self.tile_ff_norm(tiles))
        return tiles, global_tokens


@dataclass
class LayoutEnergyOutput:
    energy: torch.Tensor
    local_error_logits: torch.Tensor
    move_vectors: torch.Tensor
    move_queries: torch.Tensor
    move_keys: torch.Tensor
    tile_features: torch.Tensor
    global_features: torch.Tensor


class LayoutEnergyTransformer(nn.Module):
    """Multi-scale raw-only plausibility energy model for proposed layouts."""

    def __init__(self, config: LayoutEnergyConfig | None = None) -> None:
        super().__init__()
        self.config = config or LayoutEnergyConfig()
        self.config.validate()
        self.encoder = MultiScaleTileEncoder(self.config)
        self.row_embedding = nn.Embedding(
            self.config.grid_size, self.config.d_model
        )
        self.column_embedding = nn.Embedding(
            self.config.grid_size, self.config.d_model
        )
        self.local_blocks = nn.ModuleList(
            [WindowSpatialBlock(self.config) for _ in range(self.config.local_layers)]
        )
        self.global_token_seed = nn.Parameter(
            torch.empty(self.config.global_tokens, self.config.d_model)
        )
        self.global_blocks = nn.ModuleList(
            [GlobalTokenBlock(self.config) for _ in range(self.config.global_layers)]
        )
        self.final_tile_norm = nn.LayerNorm(self.config.d_model)
        self.final_global_norm = nn.LayerNorm(self.config.d_model)
        self.energy_head = nn.Sequential(
            nn.Linear(self.config.d_model * 2, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, 1),
        )
        self.local_error_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Linear(self.config.d_model // 2, 1),
        )
        self.move_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Linear(self.config.d_model // 2, 2),
            nn.Tanh(),
        )
        self.move_query = nn.Linear(self.config.d_model, self.config.move_dim)
        self.move_key = nn.Linear(self.config.d_model, self.config.move_dim)
        nn.init.normal_(self.global_token_seed, std=0.02)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.column_embedding.weight, std=0.02)

    def _position_embedding(self, device: torch.device) -> torch.Tensor:
        positions = torch.arange(self.config.tile_count, device=device)
        return self.row_embedding(positions // self.config.grid_size) + self.column_embedding(
            positions % self.config.grid_size
        )

    def _score_tokens(self, tokens: torch.Tensor) -> LayoutEnergyOutput:
        tokens = tokens + self._position_embedding(tokens.device).unsqueeze(0)
        for block in self.local_blocks:
            tokens = block(tokens)
        global_tokens = self.global_token_seed.unsqueeze(0).expand(
            len(tokens), -1, -1
        )
        for block in self.global_blocks:
            tokens, global_tokens = block(tokens, global_tokens)
        tokens = self.final_tile_norm(tokens)
        global_tokens = self.final_global_norm(global_tokens)
        global_summary = global_tokens.mean(dim=1)
        tile_summary = tokens.mean(dim=1)
        energy = self.energy_head(torch.cat([global_summary, tile_summary], dim=1)).squeeze(1)
        error_logits = self.local_error_head(tokens).squeeze(2)
        move_vectors = self.move_head(tokens)
        move_queries = F.normalize(self.move_query(tokens), dim=2)
        move_keys = F.normalize(self.move_key(tokens), dim=2)
        return LayoutEnergyOutput(
            energy=energy,
            local_error_logits=error_logits,
            move_vectors=move_vectors,
            move_queries=move_queries,
            move_keys=move_keys,
            tile_features=tokens,
            global_features=global_tokens,
        )

    def encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        """Encode each raw tile independently for candidate-layout reuse."""

        return self.encoder(tiles)

    def score_encoded_tiles(self, ordered_tokens: torch.Tensor) -> LayoutEnergyOutput:
        if ordered_tokens.ndim != 3 or ordered_tokens.shape[1:] != (
            self.config.tile_count,
            self.config.d_model,
        ):
            raise ValueError("ordered_tokens must have shape BxNxD")
        return self._score_tokens(ordered_tokens)

    def forward(
        self,
        tiles: torch.Tensor,
        candidate_layouts: torch.Tensor | None = None,
    ) -> LayoutEnergyOutput:
        encoded = self.encode_tiles(tiles)
        if candidate_layouts is None:
            return self._score_tokens(encoded)
        if candidate_layouts.ndim != 3 or candidate_layouts.shape[:1] != encoded.shape[:1]:
            raise ValueError("candidate_layouts must have shape BxCxN")
        if candidate_layouts.shape[2] != self.config.tile_count:
            raise ValueError("candidate layout tile count does not match model")
        layouts = candidate_layouts.long()
        if bool(((layouts < 0) | (layouts >= self.config.tile_count)).any()):
            raise ValueError("candidate layouts contain out-of-range tile indices")
        expected = torch.arange(self.config.tile_count, device=layouts.device)
        if not torch.equal(layouts.sort(dim=2).values, expected.view(1, 1, -1).expand_as(layouts)):
            raise ValueError("every candidate layout must be a permutation")
        batch, candidates, count = layouts.shape
        expanded = encoded.unsqueeze(1).expand(-1, candidates, -1, -1)
        ordered = expanded.gather(
            2,
            layouts.unsqueeze(3).expand(-1, -1, -1, self.config.d_model),
        )
        return self._score_tokens(ordered.reshape(batch * candidates, count, -1))


def layout_energy_losses(
    output: LayoutEnergyOutput,
    error_targets: torch.Tensor,
    move_targets: torch.Tensor,
    *,
    candidates_per_source: int,
    severity_targets: torch.Tensor | None = None,
    ranking_margin: float = 0.20,
    ranking_weight: float = 1.0,
    listwise_weight: float = 0.5,
    local_error_weight: float = 0.5,
    move_weight: float = 0.25,
    move_matching_weight: float = 0.10,
    graded_monotonic_weight: float = 0.35,
    energy_regularization: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Ranking and localization losses for positive-first candidate lists.

    Candidate index zero for every source must be the correct layout.  Lower
    energy is better.  Local targets and normalized move vectors supervise both
    the error heatmap and target-free refinement proposals.
    """

    if candidates_per_source < 2:
        raise ValueError("candidates_per_source must include one positive and negatives")
    batch_total, tile_count = output.local_error_logits.shape
    if batch_total % candidates_per_source:
        raise ValueError("output batch does not divide into candidate lists")
    if error_targets.shape != (batch_total, tile_count):
        raise ValueError("error_targets do not match local logits")
    if move_targets.shape != (batch_total, tile_count, 2):
        raise ValueError("move_targets must have shape BxNx2")
    if not torch.isfinite(move_targets).all():
        raise ValueError("move_targets contain non-finite values")
    for name, value in (
        ("ranking_margin", ranking_margin),
        ("ranking_weight", ranking_weight),
        ("listwise_weight", listwise_weight),
        ("local_error_weight", local_error_weight),
        ("move_weight", move_weight),
        ("move_matching_weight", move_matching_weight),
        ("graded_monotonic_weight", graded_monotonic_weight),
        ("energy_regularization", energy_regularization),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    source_count = batch_total // candidates_per_source
    structured_errors = error_targets.reshape(
        source_count, candidates_per_source, tile_count
    )
    structured_moves = move_targets.reshape(
        source_count, candidates_per_source, tile_count, 2
    )
    if bool((structured_errors[:, 0] != 0).any()) or bool(
        (structured_moves[:, 0] != 0).any()
    ):
        raise ValueError("candidate zero must be the exact positive layout")
    if bool((structured_errors[:, 1:].sum(dim=2) <= 0).any()):
        raise ValueError("every negative candidate must contain at least one error")
    energies = output.energy.reshape(source_count, candidates_per_source)
    positive = energies[:, :1]
    negatives = energies[:, 1:]
    ranking = F.softplus(ranking_margin + positive - negatives).mean()
    listwise = F.cross_entropy(
        -energies,
        torch.zeros(source_count, device=energies.device, dtype=torch.long),
    )
    if severity_targets is None:
        severity = structured_errors.float().mean(dim=2)
    else:
        if severity_targets.shape not in {
            (batch_total,),
            (source_count, candidates_per_source),
        }:
            raise ValueError("severity_targets must have shape B or source_count x candidates")
        severity = severity_targets.reshape(source_count, candidates_per_source).float()
        if not torch.isfinite(severity).all() or bool((severity < 0).any()):
            raise ValueError("severity_targets must be finite and non-negative")
    if bool((severity[:, 0] != 0).any()):
        raise ValueError("positive candidate severity must be exactly zero")
    severity_difference = severity.unsqueeze(2) - severity.unsqueeze(1)
    # Pair (i,j) is supervised when candidate i is strictly more wrong than j.
    graded_pairs = severity_difference > 1e-6
    energy_difference = energies.unsqueeze(2) - energies.unsqueeze(1)
    if bool(graded_pairs.any()):
        graded_monotonic = F.softplus(
            ranking_margin * severity_difference[graded_pairs]
            - energy_difference[graded_pairs]
        ).mean()
    else:
        graded_monotonic = energies.sum() * 0.0
    targets = error_targets.to(dtype=output.local_error_logits.dtype).clamp(0.0, 1.0)
    positive_count = targets.sum()
    negative_count = targets.numel() - positive_count
    pos_weight = (negative_count / positive_count.clamp_min(1.0)).clamp(1.0, 20.0)
    local_error = F.binary_cross_entropy_with_logits(
        output.local_error_logits,
        targets,
        pos_weight=pos_weight,
    )
    move_per_position = F.smooth_l1_loss(
        output.move_vectors,
        move_targets.to(dtype=output.move_vectors.dtype),
        reduction="none",
    ).mean(dim=2)
    move_weights = 0.10 + 0.90 * targets
    move = (move_per_position * move_weights).sum() / move_weights.sum().clamp_min(1.0)
    grid_size = int(round(math.sqrt(tile_count)))
    if grid_size * grid_size != tile_count:
        raise ValueError("tile count must form a square grid")
    positions = torch.arange(tile_count, device=move_targets.device)
    deltas = torch.round(
        move_targets.to(dtype=torch.float32) * float(max(grid_size - 1, 1))
    ).long()
    rows = (positions // grid_size).view(1, -1) + deltas[:, :, 0]
    columns = (positions % grid_size).view(1, -1) + deltas[:, :, 1]
    destinations = (rows.clamp(0, grid_size - 1) * grid_size + columns.clamp(0, grid_size - 1))
    move_logits = torch.einsum(
        "bnd,bmd->bnm", output.move_queries.float(), output.move_keys.float()
    ) * 10.0
    erroneous = targets > 0.5
    if bool(erroneous.any()):
        move_matching = F.cross_entropy(
            move_logits[erroneous], destinations[erroneous], reduction="mean"
        )
    else:
        move_matching = move_logits.sum() * 0.0
    regularization = output.energy.square().mean()
    total = (
        ranking_weight * ranking
        + listwise_weight * listwise
        + local_error_weight * local_error
        + move_weight * move
        + move_matching_weight * move_matching
        + graded_monotonic_weight * graded_monotonic
        + energy_regularization * regularization
    )
    return {
        "total": total,
        "ranking": ranking,
        "listwise": listwise,
        "local_error": local_error,
        "move": move,
        "move_matching": move_matching,
        "graded_monotonic": graded_monotonic,
        "energy_regularization": regularization,
    }


@dataclass(frozen=True)
class NegativeLayout:
    family: str
    position_to_tile: np.ndarray
    error_mask: np.ndarray
    move_targets: np.ndarray


NEGATIVE_FAMILIES = (
    "row_column",
    "segment",
    "block_swap",
    "component_translation",
    "sparse_swap",
    "solver_like_sparse",
    "similar_swap",
    "mixture",
)


def _validate_layout(layout: np.ndarray, tile_count: int) -> np.ndarray:
    values = np.asarray(layout)
    if values.shape != (tile_count,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"layout must be an integer permutation of length {tile_count}")
    values = values.astype(np.int32, copy=True)
    if not np.array_equal(np.sort(values), np.arange(tile_count)):
        raise ValueError("layout must be a permutation")
    return values


def _swap_rows_or_columns(layout: np.ndarray, grid: int, rng: np.random.Generator) -> None:
    values = layout.reshape(grid, grid)
    first, second = rng.choice(grid, size=2, replace=False)
    if rng.random() < 0.5:
        values[[first, second], :] = values[[second, first], :]
    else:
        values[:, [first, second]] = values[:, [second, first]]


def _segment_swap(layout: np.ndarray, grid: int, rng: np.random.Generator) -> None:
    values = layout.reshape(grid, grid)
    length = int(rng.integers(2, max(3, grid // 2 + 1)))
    length = min(length, grid)
    if rng.random() < 0.5:
        row_a, row_b = rng.choice(grid, size=2, replace=False)
        start_a = int(rng.integers(0, grid - length + 1))
        start_b = int(rng.integers(0, grid - length + 1))
        temporary = values[row_a, start_a : start_a + length].copy()
        values[row_a, start_a : start_a + length] = values[
            row_b, start_b : start_b + length
        ]
        values[row_b, start_b : start_b + length] = temporary
    else:
        col_a, col_b = rng.choice(grid, size=2, replace=False)
        start_a = int(rng.integers(0, grid - length + 1))
        start_b = int(rng.integers(0, grid - length + 1))
        temporary = values[start_a : start_a + length, col_a].copy()
        values[start_a : start_a + length, col_a] = values[
            start_b : start_b + length, col_b
        ]
        values[start_b : start_b + length, col_b] = temporary


def _block_swap(layout: np.ndarray, grid: int, rng: np.random.Generator) -> None:
    block = 2 if grid < 6 or rng.random() < 0.5 else 3
    block = min(block, grid)
    positions = [(row, col) for row in range(grid - block + 1) for col in range(grid - block + 1)]
    first = positions[int(rng.integers(len(positions)))]
    candidates = [
        value
        for value in positions
        if value[0] + block <= first[0]
        or first[0] + block <= value[0]
        or value[1] + block <= first[1]
        or first[1] + block <= value[1]
    ]
    if not candidates:
        _swap_rows_or_columns(layout, grid, rng)
        return
    second = candidates[int(rng.integers(len(candidates)))]
    values = layout.reshape(grid, grid)
    a = values[first[0] : first[0] + block, first[1] : first[1] + block].copy()
    b = values[second[0] : second[0] + block, second[1] : second[1] + block].copy()
    values[first[0] : first[0] + block, first[1] : first[1] + block] = b
    values[second[0] : second[0] + block, second[1] : second[1] + block] = a


def _component_translation(layout: np.ndarray, grid: int, rng: np.random.Generator) -> None:
    height = int(rng.integers(2, max(3, grid // 2 + 1)))
    width = int(rng.integers(2, max(3, grid // 2 + 1)))
    height, width = min(height, grid), min(width, grid)
    row = int(rng.integers(0, grid - height + 1))
    col = int(rng.integers(0, grid - width + 1))
    values = layout.reshape(grid, grid)
    region = values[row : row + height, col : col + width]
    shift = (0, 1) if rng.random() < 0.5 else (1, 0)
    values[row : row + height, col : col + width] = np.roll(
        region, shift=shift, axis=(0, 1)
    )


def _sparse_swaps(
    layout: np.ndarray,
    rng: np.random.Generator,
    *,
    pair_count: int,
    tile_features: np.ndarray | None,
    similar: bool,
) -> None:
    tile_count = len(layout)
    used: set[int] = set()
    features = None
    if tile_features is not None:
        features = np.asarray(tile_features, dtype=np.float32).reshape(tile_count, -1)
        features = (features - features.mean(axis=0, keepdims=True)) / np.maximum(
            features.std(axis=0, keepdims=True), 1e-5
        )
    for _ in range(pair_count):
        choices = [index for index in range(tile_count) if index not in used]
        if len(choices) < 2:
            break
        first = int(choices[int(rng.integers(len(choices)))])
        partners = np.asarray([index for index in choices if index != first], dtype=np.int32)
        if similar and features is not None:
            distance = np.mean((features[partners] - features[first]) ** 2, axis=1)
            top = partners[np.argsort(distance)[: min(8, len(partners))]]
            second = int(top[int(rng.integers(len(top)))])
        else:
            second = int(partners[int(rng.integers(len(partners)))])
        layout[first], layout[second] = layout[second], layout[first]
        used.update((first, second))


def make_negative_layout(
    *,
    grid_size: int,
    family: str,
    rng: np.random.Generator,
    tile_features: np.ndarray | None = None,
    severity: float = 0.10,
) -> NegativeLayout:
    """Create a hard permutation error with exact local supervision."""

    if type(grid_size) is not int or grid_size < 2:
        raise ValueError("grid_size must be an integer >=2")
    if family not in NEGATIVE_FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {NEGATIVE_FAMILIES}")
    if not math.isfinite(severity) or not 0.0 < severity <= 1.0:
        raise ValueError("severity must be in (0,1]")
    tile_count = grid_size * grid_size
    layout = np.arange(tile_count, dtype=np.int32)

    def apply(selected: str) -> None:
        if selected == "row_column":
            _swap_rows_or_columns(layout, grid_size, rng)
        elif selected == "segment":
            _segment_swap(layout, grid_size, rng)
        elif selected == "block_swap":
            _block_swap(layout, grid_size, rng)
        elif selected == "component_translation":
            _component_translation(layout, grid_size, rng)
        elif selected in {"sparse_swap", "solver_like_sparse", "similar_swap"}:
            pair_count = max(1, int(round(severity * tile_count / 2.0)))
            if selected in {"solver_like_sparse", "similar_swap"}:
                pair_count = min(pair_count, max(1, grid_size // 3))
            _sparse_swaps(
                layout,
                rng,
                pair_count=pair_count,
                tile_features=tile_features,
                similar=selected in {"solver_like_sparse", "similar_swap"},
            )
        else:
            raise AssertionError(selected)

    if family == "mixture":
        choices = ["segment", "block_swap", "component_translation", "solver_like_sparse"]
        for selected in rng.choice(choices, size=2, replace=False):
            apply(str(selected))
    else:
        apply(family)
    if np.array_equal(layout, np.arange(tile_count)):
        layout[0], layout[1] = layout[1], layout[0]
    _validate_layout(layout, tile_count)
    expected = np.arange(tile_count, dtype=np.int32)
    positions = expected
    tile_positions = layout
    scale = float(max(grid_size - 1, 1))
    move_targets = np.stack(
        [
            (tile_positions // grid_size - positions // grid_size) / scale,
            (tile_positions % grid_size - positions % grid_size) / scale,
        ],
        axis=1,
    ).astype(np.float32)
    return NegativeLayout(
        family=family,
        position_to_tile=layout,
        error_mask=(layout != expected).astype(np.float32),
        move_targets=move_targets,
    )


def classical_seam_energy(ordered_tiles: np.ndarray) -> np.ndarray:
    """Raw input-only mean L1 seam energy; lower is better."""

    values = np.asarray(ordered_tiles)
    squeeze = values.ndim == 4
    if squeeze:
        values = values[None]
    if values.ndim != 5 or values.shape[-1] != 3:
        raise ValueError("ordered_tiles must be NxHxWx3 or BxNxHxWx3")
    batch, count, height, width, _ = values.shape
    grid = int(round(math.sqrt(count)))
    if grid * grid != count or height <= 0 or width <= 0:
        raise ValueError("tile count must form a square grid")
    values = values.astype(np.float32) / (255.0 if values.dtype == np.uint8 else 1.0)
    grids = values.reshape(batch, grid, grid, height, width, 3)
    right = np.abs(grids[:, :, :-1, :, -1, :] - grids[:, :, 1:, :, 0, :]).mean(
        axis=(1, 2, 3, 4)
    )
    down = np.abs(grids[:, :-1, :, -1, :, :] - grids[:, 1:, :, 0, :, :]).mean(
        axis=(1, 2, 3, 4)
    )
    result = 0.5 * (right + down)
    return result[0] if squeeze else result


@dataclass(frozen=True)
class CandidateScoreBatch:
    energies: np.ndarray
    error_probabilities: np.ndarray
    move_vectors: np.ndarray
    move_queries: np.ndarray
    move_keys: np.ndarray


def _raw_tiles_tensor(
    raw_tiles: np.ndarray | torch.Tensor,
    *,
    config: LayoutEnergyConfig,
) -> torch.Tensor:
    if isinstance(raw_tiles, torch.Tensor):
        values = raw_tiles.detach().cpu()
        scale = 255.0 if values.dtype == torch.uint8 else 1.0
        if values.shape == (
            config.tile_count,
            3,
            config.tile_size,
            config.tile_size,
        ):
            return values.float().div(scale).clamp(0.0, 1.0)
        if values.shape == (
            config.tile_count,
            config.tile_size,
            config.tile_size,
            3,
        ):
            return values.permute(0, 3, 1, 2).float().div(scale).clamp(0.0, 1.0)
        raise ValueError("raw tile tensor has unexpected shape")
    values = np.asarray(raw_tiles)
    if values.shape != (
        config.tile_count,
        config.tile_size,
        config.tile_size,
        3,
    ):
        raise ValueError("raw_tiles must have shape NxHxWx3")
    if values.dtype != np.uint8:
        raise TypeError("numpy raw_tiles must be uint8")
    return torch.from_numpy(
        np.ascontiguousarray(values.transpose(0, 3, 1, 2))
    ).float().div_(255.0)


@torch.inference_mode()
def score_candidate_layouts(
    model: LayoutEnergyTransformer,
    raw_tiles: np.ndarray | torch.Tensor,
    candidate_layouts: np.ndarray | Sequence[np.ndarray],
    *,
    device: torch.device | str | None = None,
    batch_size: int = 4,
    autocast_dtype: torch.dtype | None = None,
) -> CandidateScoreBatch:
    """Score position-to-input-slot layouts without targets or denoising."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = model.config
    base = _raw_tiles_tensor(raw_tiles, config=config)
    layouts = np.asarray(candidate_layouts)
    if layouts.ndim == 1:
        layouts = layouts[None]
    if layouts.ndim != 2 or layouts.shape[1] != config.tile_count:
        raise ValueError("candidate_layouts must have shape MxN")
    layouts = np.stack(
        [_validate_layout(layout, config.tile_count) for layout in layouts]
    )
    resolved_device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )
    was_training = model.training
    model.eval()
    energies: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    moves: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    keys: list[np.ndarray] = []
    try:
        base_device = base.unsqueeze(0).to(resolved_device)
        use_amp = autocast_dtype is not None and resolved_device.type == "cuda"
        with torch.autocast(
            device_type=resolved_device.type,
            dtype=autocast_dtype or torch.float16,
            enabled=use_amp,
        ):
            encoded = model.encode_tiles(base_device)[0]
        for start in range(0, len(layouts), batch_size):
            rows = layouts[start : start + batch_size]
            indices = torch.from_numpy(rows).to(resolved_device, dtype=torch.long)
            ordered = encoded.unsqueeze(0).expand(len(rows), -1, -1).gather(
                1,
                indices.unsqueeze(2).expand(-1, -1, config.d_model),
            )
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=autocast_dtype or torch.float16,
                enabled=use_amp,
            ):
                output = model.score_encoded_tiles(ordered)
            energies.append(output.energy.float().cpu().numpy())
            errors.append(output.local_error_logits.float().sigmoid().cpu().numpy())
            moves.append(output.move_vectors.float().cpu().numpy())
            queries.append(output.move_queries.float().cpu().numpy())
            keys.append(output.move_keys.float().cpu().numpy())
    finally:
        model.train(was_training)
    return CandidateScoreBatch(
        energies=np.concatenate(energies),
        error_probabilities=np.concatenate(errors),
        move_vectors=np.concatenate(moves),
        move_queries=np.concatenate(queries),
        move_keys=np.concatenate(keys),
    )


@dataclass(frozen=True)
class RefinementStep:
    iteration: int
    candidates_scored: int
    best_energy: float
    accepted_improvement: bool


@dataclass(frozen=True)
class RefinementResult:
    position_to_slot: np.ndarray
    initial_energy: float
    final_energy: float
    steps: tuple[RefinementStep, ...]


def _swap_candidate(layout: np.ndarray, first: int, second: int) -> np.ndarray:
    result = layout.copy()
    result[first], result[second] = result[second], result[first]
    return result


def _refinement_proposals(
    layout: np.ndarray,
    scores: CandidateScoreBatch,
    *,
    grid_size: int,
    hot_positions: int,
    max_proposals: int,
) -> list[np.ndarray]:
    errors = scores.error_probabilities[0]
    moves = scores.move_vectors[0]
    queries = scores.move_queries[0]
    keys = scores.move_keys[0]
    hot = np.argsort(-errors)[: min(hot_positions, len(layout))]
    proposals: list[np.ndarray] = []
    seen: set[bytes] = set()

    def add_layout(candidate: np.ndarray) -> None:
        if len(proposals) >= max_proposals:
            return
        key = candidate.tobytes()
        if key not in seen:
            seen.add(key)
            proposals.append(candidate)

    def add(first: int, second: int) -> None:
        if first != second:
            add_layout(_swap_candidate(layout, int(first), int(second)))

    scale = float(max(grid_size - 1, 1))
    guided_pairs: list[tuple[int, int]] = []
    for position in hot:
        row, column = divmod(int(position), grid_size)
        delta = np.rint(moves[position] * scale).astype(np.int32)
        destination_row = int(np.clip(row + delta[0], 0, grid_size - 1))
        destination_col = int(np.clip(column + delta[1], 0, grid_size - 1))
        destination = destination_row * grid_size + destination_col
        if int(position) != destination:
            guided_pairs.append((int(position), destination))
        compatibility = queries[position] @ keys.T
        compatibility[position] = -np.inf
        guided_pairs.extend(
            (int(position), int(partner))
            for partner in np.argsort(-compatibility)[:2]
        )

    # Batched disjoint swaps let one refinement step repair a meaningful
    # fraction of a globally poor first pass.  They are still target-free:
    # every pair comes only from predicted moves/move embeddings.
    disjoint: list[tuple[int, int]] = []
    occupied: set[int] = set()
    for first, second in guided_pairs:
        if first == second or first in occupied or second in occupied:
            continue
        disjoint.append((first, second))
        occupied.update((first, second))
        if len(disjoint) >= max(1, len(hot) // 2):
            break
    for pair_count in sorted({2, 4, 8, len(disjoint)}):
        if pair_count <= 0 or pair_count > len(disjoint):
            continue
        candidate = layout.copy()
        for first, second in disjoint[:pair_count]:
            candidate[first], candidate[second] = candidate[second], candidate[first]
        add_layout(candidate)
    for first, second in guided_pairs:
        add(first, second)
        if len(proposals) >= max_proposals:
            return proposals
    for offset, first in enumerate(hot):
        for second in hot[offset + 1 :]:
            add(int(first), int(second))
            if len(proposals) >= max_proposals:
                return proposals
    return proposals


def iterative_refine_layout(
    model: LayoutEnergyTransformer,
    raw_tiles: np.ndarray | torch.Tensor,
    initial_position_to_slot: np.ndarray,
    *,
    device: torch.device | str | None = None,
    steps: int = 6,
    beam_width: int = 3,
    hot_positions: int = 32,
    proposals_per_layout: int = 64,
    score_batch_size: int = 4,
    min_improvement: float = 1e-4,
    autocast_dtype: torch.dtype | None = None,
) -> RefinementResult:
    """Target-free beam refinement guided by energy, heatmap, and move heads."""

    config = model.config
    initial = _validate_layout(initial_position_to_slot, config.tile_count)
    if min(steps, beam_width, hot_positions, proposals_per_layout, score_batch_size) <= 0:
        raise ValueError("refinement counts must be positive")
    if not math.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("min_improvement must be finite and non-negative")
    first_score = score_candidate_layouts(
        model,
        raw_tiles,
        initial,
        device=device,
        batch_size=score_batch_size,
        autocast_dtype=autocast_dtype,
    )
    initial_energy = float(first_score.energies[0])
    beam: list[tuple[float, np.ndarray]] = [(initial_energy, initial.copy())]
    history: list[RefinementStep] = []
    best_energy = initial_energy
    best_layout = initial.copy()
    for iteration in range(1, steps + 1):
        pool: dict[bytes, np.ndarray] = {layout.tobytes(): layout for _, layout in beam}
        for _, layout in beam:
            local_score = score_candidate_layouts(
                model,
                raw_tiles,
                layout,
                device=device,
                batch_size=1,
                autocast_dtype=autocast_dtype,
            )
            for proposal in _refinement_proposals(
                layout,
                local_score,
                grid_size=config.grid_size,
                hot_positions=hot_positions,
                max_proposals=proposals_per_layout,
            ):
                pool.setdefault(proposal.tobytes(), proposal)
        candidates = list(pool.values())
        scored = score_candidate_layouts(
            model,
            raw_tiles,
            candidates,
            device=device,
            batch_size=score_batch_size,
            autocast_dtype=autocast_dtype,
        )
        order = np.argsort(scored.energies)
        beam = [
            (float(scored.energies[index]), candidates[int(index)].copy())
            for index in order[:beam_width]
        ]
        candidate_energy, candidate_layout = beam[0]
        accepted = candidate_energy < best_energy - min_improvement
        if accepted:
            best_energy = candidate_energy
            best_layout = candidate_layout.copy()
        history.append(
            RefinementStep(
                iteration=iteration,
                candidates_scored=len(candidates),
                best_energy=candidate_energy,
                accepted_improvement=accepted,
            )
        )
        if not accepted:
            break
    return RefinementResult(
        position_to_slot=best_layout,
        initial_energy=initial_energy,
        final_energy=best_energy,
        steps=tuple(history),
    )
