"""Task-specific positional diffusion for corrupted jigsaw tiles.

This is a pure-PyTorch adaptation of Positional Diffusion
(https://arxiv.org/abs/2303.11120) for the fixed 576-tile restoration task.
Unlike the reference implementation, it encodes both raw and restored tiles,
accepts an optional input-only directional graph, projects with an exact
Hungarian assignment, and can warm-start deterministic DDIM from an actual
first-pass layout.  No token-index embedding is used, so the network is
permutation equivariant when tiles, positions, and graph axes are permuted
together.

The module intentionally contains no dataset or target access.  In particular,
``baseline_positions`` is an unweighted coordinate tensor: callers must never
construct it from target-derived confidence or target-selected candidates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class PositionalDiffusionConfig:
    """Architecture and diffusion parameters saved with every checkpoint."""

    model_dim: int = 128
    cnn_channels: int = 64
    layers: int = 4
    heads: int = 8
    feedforward_dim: int = 512
    dropout: float = 0.05
    diffusion_steps: int = 300
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    tile_encode_chunk: int = 192
    activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.model_dim <= 0 or self.cnn_channels <= 0:
            raise ValueError("model_dim and cnn_channels must be positive")
        if self.heads <= 0:
            raise ValueError("heads must be positive")
        if self.model_dim % self.heads:
            raise ValueError("heads must divide model_dim")
        if self.model_dim % 2:
            raise ValueError("model_dim must be even")
        if self.layers <= 0 or self.feedforward_dim <= 0:
            raise ValueError("layers and feedforward_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least two")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("expected 0 < beta_start < beta_end < 1")
        if self.tile_encode_chunk <= 0:
            raise ValueError("tile_encode_chunk must be positive")


@dataclass(frozen=True)
class ProjectionResult:
    """Exact discrete assignment of tile-indexed coordinates to grid cells."""

    position_to_tile: np.ndarray
    tile_to_position: np.ndarray
    squared_assignment_cost: float


@dataclass(frozen=True)
class DiffusionSample:
    """Continuous DDIM output and its per-example Hungarian projections."""

    initial_positions: torch.Tensor
    positions: torch.Tensor
    projections: tuple[ProjectionResult, ...]
    sampling_timesteps: tuple[int, ...]
    initialization: str


def normalized_grid_positions(
    rows: int,
    columns: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return row-major ``(x, y)`` cell centres normalized to ``[-1, 1]``."""

    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    xs = (
        torch.zeros(1, device=device, dtype=dtype)
        if columns == 1
        else torch.linspace(-1.0, 1.0, columns, device=device, dtype=dtype)
    )
    ys = (
        torch.zeros(1, device=device, dtype=dtype)
        if rows == 1
        else torch.linspace(-1.0, 1.0, rows, device=device, dtype=dtype)
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def _generic_permutation(values: np.ndarray, size: int, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be integral")
    array = array.astype(np.int32, copy=False)
    if not np.array_equal(np.sort(array), np.arange(size, dtype=np.int32)):
        raise ValueError(f"{name} must contain [0,{size - 1}] exactly once")
    return array


def layout_to_tile_positions(
    position_to_tile: np.ndarray,
    rows: int,
    columns: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert an input-only layout into tile-indexed normalized coordinates."""

    size = rows * columns
    layout = _generic_permutation(position_to_tile, size, name="position_to_tile")
    grid = normalized_grid_positions(rows, columns, device=device, dtype=dtype)
    tile_positions = torch.empty_like(grid)
    tile_indices = torch.as_tensor(layout, device=grid.device, dtype=torch.long)
    tile_positions[tile_indices] = grid
    return tile_positions


def project_positions_hungarian(
    tile_positions: torch.Tensor | np.ndarray,
    rows: int,
    columns: int,
) -> ProjectionResult:
    """Project continuous tile positions to the globally optimal grid assignment."""

    values = (
        tile_positions.detach().float().cpu().numpy()
        if isinstance(tile_positions, torch.Tensor)
        else np.asarray(tile_positions, dtype=np.float32)
    )
    size = rows * columns
    if values.shape != (size, 2):
        raise ValueError(f"tile_positions must have shape {(size, 2)}")
    if not np.isfinite(values).all():
        raise ValueError("tile_positions contain non-finite values")
    grid = normalized_grid_positions(rows, columns).numpy()
    squared_cost = np.sum((values[:, None, :] - grid[None, :, :]) ** 2, axis=2)
    tile_indices, position_indices = linear_sum_assignment(squared_cost)
    position_to_tile = np.empty(size, dtype=np.int32)
    position_to_tile[position_indices] = tile_indices.astype(np.int32, copy=False)
    tile_to_position = np.empty(size, dtype=np.int32)
    tile_to_position[tile_indices] = position_indices.astype(np.int32, copy=False)
    return ProjectionResult(
        position_to_tile=position_to_tile,
        tile_to_position=tile_to_position,
        squared_assignment_cost=float(squared_cost[tile_indices, position_indices].sum()),
    )


def compatibility_to_relative_graph(
    right_cost: np.ndarray,
    down_cost: np.ndarray,
    *,
    top_k: int = 16,
    temperature: float = 0.35,
) -> np.ndarray:
    """Turn input-only directional costs into a sparse row-stochastic graph.

    Only within-row ranks are used.  This makes the scale stable across the HBT
    scorer and corruption engines while retaining its relative-neighbour
    evidence.  The output has shape ``(2,N,N)`` for right and down messages.
    """

    right = np.asarray(right_cost, dtype=np.float32)
    down = np.asarray(down_cost, dtype=np.float32)
    if right.ndim != 2 or right.shape[0] != right.shape[1] or down.shape != right.shape:
        raise ValueError("right_cost and down_cost must be equally sized square matrices")
    size = right.shape[0]
    if not 1 <= top_k < size:
        raise ValueError("top_k must be in [1,N-1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    outputs: list[np.ndarray] = []
    for costs in (right, down):
        finite = np.isfinite(costs).copy()
        np.fill_diagonal(finite, False)
        safe = np.where(finite, costs, np.inf)
        order = np.argsort(safe, axis=1, kind="stable")[:, :top_k]
        ranks = np.arange(top_k, dtype=np.float32)[None, :]
        logits = -np.log1p(ranks) / float(temperature)
        logits = logits - logits.max(axis=1, keepdims=True)
        weights = np.exp(logits).astype(np.float32)
        weights /= weights.sum(axis=1, keepdims=True)
        graph = np.zeros((size, size), dtype=np.float32)
        graph[np.arange(size)[:, None], order] = weights
        outputs.append(graph)
    return np.stack(outputs, axis=0)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + F.silu(self.norm(self.pointwise(self.depthwise(values))))


class RawDenoisedTileEncoder(nn.Module):
    """Shared encoder for every raw/restored tile; no slot identity is present."""

    def __init__(self, cnn_channels: int, output_dim: int) -> None:
        super().__init__()
        groups = _group_count(cnn_channels)
        # raw, restored, their normalized versions, and their difference.
        self.stem = nn.Sequential(
            nn.Conv2d(15, cnn_channels, 3, padding=1),
            nn.GroupNorm(groups, cnn_channels),
            nn.SiLU(),
            _ResidualConv(cnn_channels),
            nn.Conv2d(cnn_channels, cnn_channels, 3, stride=2, padding=1),
            nn.GroupNorm(groups, cnn_channels),
            nn.SiLU(),
            _ResidualConv(cnn_channels),
            nn.Conv2d(cnn_channels, cnn_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(cnn_channels * 2), cnn_channels * 2),
            nn.SiLU(),
            _ResidualConv(cnn_channels * 2),
        )
        self.global_stats = nn.Sequential(
            nn.Linear(12, output_dim // 2), nn.SiLU()
        )
        self.side_stats = nn.Sequential(
            nn.Linear(24, output_dim // 2), nn.SiLU()
        )
        self.output = nn.Sequential(
            nn.Linear(cnn_channels * 2 + output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=(2, 3), keepdim=True)
        std = values.std(dim=(2, 3), keepdim=True).clamp_min(1.0 / 255.0)
        return ((values - mean) / std).clamp(-4.0, 4.0) / 4.0

    @staticmethod
    def _sides(values: torch.Tensor) -> torch.Tensor:
        band = max(1, min(4, values.shape[-1] // 2, values.shape[-2] // 2))
        side_means = (
            values[:, :, :, :band].mean(dim=(2, 3)),
            values[:, :, :, -band:].mean(dim=(2, 3)),
            values[:, :, :band, :].mean(dim=(2, 3)),
            values[:, :, -band:, :].mean(dim=(2, 3)),
        )
        return torch.cat(side_means, dim=1)

    def forward(self, raw: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 4 or raw.shape[1] != 3 or raw.shape[-2:] != restored.shape[-2:]:
            raise ValueError("raw and restored must be matching NCHW RGB tiles")
        if restored.shape != raw.shape:
            raise ValueError("raw and restored tile tensors must match")
        raw_values = raw.float()
        restored_values = restored.float()
        if raw_values.detach().amax() > 1.5:
            raw_values = raw_values / 255.0
        if restored_values.detach().amax() > 1.5:
            restored_values = restored_values / 255.0
        encoded = torch.cat(
            [
                raw_values,
                restored_values,
                self._normalize(raw_values),
                self._normalize(restored_values),
                raw_values - restored_values,
            ],
            dim=1,
        )
        cnn = F.adaptive_avg_pool2d(self.stem(encoded), 1).flatten(1)
        global_stats = torch.cat(
            [
                raw_values.mean(dim=(2, 3)),
                raw_values.std(dim=(2, 3)),
                restored_values.mean(dim=(2, 3)),
                restored_values.std(dim=(2, 3)),
            ],
            dim=1,
        )
        side_stats = torch.cat([self._sides(raw_values), self._sides(restored_values)], dim=1)
        return self.output(
            torch.cat(
                [cnn, self.global_stats(global_stats), self.side_stats(side_stats)], dim=1
            )
        )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape (B,)")
        half = self.dim // 2
        scale = math.log(10_000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        values = torch.cat([angles.sin(), angles.cos()], dim=1)
        if values.shape[1] < self.dim:
            values = F.pad(values, (0, self.dim - values.shape[1]))
        return values


class PositionalDiffusionNet(nn.Module):
    """Attention/GNN denoiser that predicts clean tile-indexed 2D positions."""

    def __init__(self, config: PositionalDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.tile_encoder = RawDenoisedTileEncoder(config.cnn_channels, dim)
        self.graph_messages = nn.Sequential(
            nn.Linear(dim * 4, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.graph_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.position_embedding = nn.Sequential(
            nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.input_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=config.heads,
                    dim_feedforward=config.feedforward_dim,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.layers)
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 2))

    @staticmethod
    def _validate_tiles(raw: torch.Tensor, restored: torch.Tensor) -> tuple[int, int]:
        if raw.ndim != 5 or raw.shape[2] != 3:
            raise ValueError("raw must have shape (B,N,3,H,W)")
        if restored.shape != raw.shape:
            raise ValueError("restored must match raw")
        return int(raw.shape[0]), int(raw.shape[1])

    def encode_tiles(
        self,
        raw: torch.Tensor,
        restored: torch.Tensor,
        relative_graph: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode tiles once; the result can be reused through all DDIM steps."""

        batch, size = self._validate_tiles(raw, restored)
        flat_raw = raw.reshape(batch * size, *raw.shape[2:])
        flat_restored = restored.reshape(batch * size, *restored.shape[2:])
        chunks: list[torch.Tensor] = []
        for start in range(0, batch * size, self.config.tile_encode_chunk):
            raw_chunk = flat_raw[start : start + self.config.tile_encode_chunk]
            restored_chunk = flat_restored[start : start + self.config.tile_encode_chunk]
            if (
                self.training
                and self.config.activation_checkpointing
                and torch.is_grad_enabled()
            ):
                encoded = checkpoint(
                    self.tile_encoder, raw_chunk, restored_chunk, use_reentrant=False
                )
            else:
                encoded = self.tile_encoder(raw_chunk, restored_chunk)
            chunks.append(encoded)
        features = torch.cat(chunks, dim=0).reshape(batch, size, self.config.model_dim)
        if relative_graph is None:
            messages = torch.zeros(
                batch,
                size,
                self.config.model_dim * 4,
                device=features.device,
                dtype=features.dtype,
            )
        else:
            if relative_graph.shape != (batch, 2, size, size):
                raise ValueError(
                    f"relative_graph must have shape {(batch, 2, size, size)}"
                )
            graph = relative_graph.to(device=features.device, dtype=features.dtype)
            graph = graph.clamp_min(0.0)
            graph = graph / graph.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            right, down = graph[:, 0], graph[:, 1]
            left = right.transpose(1, 2)
            up = down.transpose(1, 2)
            left = left / left.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            up = up / up.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            messages = torch.cat(
                [
                    torch.bmm(right, features),
                    torch.bmm(left, features),
                    torch.bmm(down, features),
                    torch.bmm(up, features),
                ],
                dim=2,
            )
        graph_features = self.graph_messages(messages)
        gate = self.graph_gate(torch.cat([features, graph_features], dim=2))
        return features + gate * graph_features

    def denoise_from_features(
        self,
        tile_features: torch.Tensor,
        noisy_positions: torch.Tensor,
        timesteps: torch.Tensor,
        baseline_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, size, dim = tile_features.shape
        if dim != self.config.model_dim or noisy_positions.shape != (batch, size, 2):
            raise ValueError("tile features or noisy positions have incompatible shape")
        if timesteps.shape != (batch,):
            raise ValueError("timesteps must have shape (B,)")
        if baseline_positions is None:
            baseline_positions = torch.zeros_like(noisy_positions)
        elif baseline_positions.shape != noisy_positions.shape:
            raise ValueError("baseline_positions must match noisy_positions")
        position = self.position_embedding(
            torch.cat([noisy_positions, baseline_positions], dim=2)
        )
        time = self.time_embedding(timesteps).to(tile_features.dtype).unsqueeze(1)
        values = self.input_norm(tile_features + position + time)
        for layer in self.layers:
            if (
                self.training
                and self.config.activation_checkpointing
                and torch.is_grad_enabled()
            ):
                values = checkpoint(layer, values, use_reentrant=False)
            else:
                values = layer(values)
        return self.output(values)

    def forward(
        self,
        raw: torch.Tensor,
        restored: torch.Tensor,
        noisy_positions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        relative_graph: torch.Tensor | None = None,
        baseline_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.encode_tiles(raw, restored, relative_graph)
        return self.denoise_from_features(
            features, noisy_positions, timesteps, baseline_positions
        )


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gathered = values.gather(0, timesteps)
    return gathered.reshape((len(timesteps),) + (1,) * (target.ndim - 1))


def grid_neighbor_vector_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Supervise local 2D vectors after ordering nodes by true grid cell."""

    batch, size, _ = predicted.shape
    if target.shape != predicted.shape or size != rows * columns:
        raise ValueError("predicted/target shapes do not match the requested grid")
    grid = normalized_grid_positions(
        rows, columns, device=target.device, dtype=target.dtype
    )
    ordered_predictions: list[torch.Tensor] = []
    for index in range(batch):
        # target is an exact permutation of grid centres during supervised training.
        tile_for_position = torch.cdist(grid, target[index]).argmin(dim=1)
        ordered_predictions.append(predicted[index, tile_for_position])
    ordered = torch.stack(ordered_predictions, dim=0).reshape(batch, rows, columns, 2)
    expected = grid.reshape(rows, columns, 2)
    terms: list[torch.Tensor] = []
    if columns > 1:
        terms.append(
            F.smooth_l1_loss(
                ordered[:, :, 1:] - ordered[:, :, :-1],
                (expected[:, 1:] - expected[:, :-1]).unsqueeze(0).expand(batch, -1, -1, -1),
            )
        )
    if rows > 1:
        terms.append(
            F.smooth_l1_loss(
                ordered[:, 1:] - ordered[:, :-1],
                (expected[1:] - expected[:-1]).unsqueeze(0).expand(batch, -1, -1, -1),
            )
        )
    return torch.stack(terms).mean() if terms else predicted.new_zeros(())


class GaussianPositionDiffusion(nn.Module):
    """Linear-schedule DDPM training with deterministic DDIM sampling."""

    def __init__(self, config: PositionalDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        betas = torch.linspace(
            config.beta_start, config.beta_end, config.diffusion_steps, dtype=torch.float32
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    def q_sample(
        self,
        clean_positions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clean_positions.ndim != 3 or clean_positions.shape[2] != 2:
            raise ValueError("clean_positions must have shape (B,N,2)")
        if timesteps.shape != (clean_positions.shape[0],):
            raise ValueError("timesteps must have shape (B,)")
        if noise is None:
            noise = torch.randn_like(clean_positions)
        if noise.shape != clean_positions.shape:
            raise ValueError("noise must match clean_positions")
        noisy = (
            _extract(self.sqrt_alpha_bars, timesteps, clean_positions) * clean_positions
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, clean_positions) * noise
        )
        return noisy, noise

    def training_loss(
        self,
        model: PositionalDiffusionNet,
        raw: torch.Tensor,
        restored: torch.Tensor,
        clean_positions: torch.Tensor,
        *,
        rows: int,
        columns: int,
        relative_graph: torch.Tensor | None = None,
        baseline_positions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        structure_weight: float = 0.20,
    ) -> dict[str, torch.Tensor]:
        batch = clean_positions.shape[0]
        if clean_positions.shape[1:] != (rows * columns, 2):
            raise ValueError("clean_positions do not match rows*columns")
        if timesteps is None:
            timesteps = torch.randint(
                0, self.config.diffusion_steps, (batch,), device=clean_positions.device
            )
        noisy, sampled_noise = self.q_sample(clean_positions, timesteps, noise=noise)
        predicted = model(
            raw,
            restored,
            noisy,
            timesteps,
            relative_graph=relative_graph,
            baseline_positions=baseline_positions,
        )
        position_loss = F.smooth_l1_loss(predicted, clean_positions)
        structure_loss = grid_neighbor_vector_loss(
            predicted, clean_positions, rows, columns
        )
        bounds_loss = F.relu(predicted.abs() - 1.25).square().mean()
        total = position_loss + structure_weight * structure_loss + 0.01 * bounds_loss
        return {
            "loss": total,
            "position_loss": position_loss.detach(),
            "structure_loss": structure_loss.detach(),
            "bounds_loss": bounds_loss.detach(),
            "predicted_x0": predicted,
            "noise": sampled_noise,
            "timesteps": timesteps,
        }

    def sampling_timesteps(self, steps: int) -> tuple[int, ...]:
        if not 2 <= steps <= self.config.diffusion_steps:
            raise ValueError("sampling steps must be in [2,diffusion_steps]")
        values = torch.linspace(
            self.config.diffusion_steps - 1, 0, steps, dtype=torch.float64
        ).round().to(torch.long)
        ordered: list[int] = []
        for value in values.tolist():
            integer = int(value)
            if not ordered or ordered[-1] != integer:
                ordered.append(integer)
        if ordered[-1] != 0:
            ordered.append(0)
        return tuple(ordered)

    @torch.inference_mode()
    def ddim_sample(
        self,
        model: PositionalDiffusionNet,
        raw: torch.Tensor,
        restored: torch.Tensor,
        *,
        rows: int,
        columns: int,
        relative_graph: torch.Tensor | None = None,
        baseline_positions: torch.Tensor | None = None,
        sampling_steps: int = 30,
        initialization: str = "zero",
        seed: int = 0,
        clip_x0: float = 1.5,
    ) -> DiffusionSample:
        batch, size = model._validate_tiles(raw, restored)
        if size != rows * columns:
            raise ValueError("tile count does not match rows*columns")
        if baseline_positions is not None and baseline_positions.shape != (batch, size, 2):
            raise ValueError("baseline_positions must have shape (B,N,2)")
        schedule = self.sampling_timesteps(sampling_steps)
        generator = torch.Generator(device=raw.device)
        generator.manual_seed(int(seed))
        start_t = schedule[0]
        shape = (batch, size, 2)
        if initialization == "zero":
            positions = torch.zeros(shape, device=raw.device, dtype=raw.dtype)
        elif initialization == "gaussian":
            positions = torch.randn(shape, generator=generator, device=raw.device, dtype=raw.dtype)
        elif initialization == "input_layout":
            if baseline_positions is None:
                raise ValueError("input_layout initialization requires baseline_positions")
            # Deterministic-by-seed forward sample q(x_T | input layout).  Using
            # epsilon=0 here would be a severe train/inference mismatch: with the
            # default schedule the training marginal has std ~= 0.98 while the
            # zero-epsilon layout start has std ~= 0.13.
            epsilon = torch.randn(
                shape,
                generator=generator,
                device=raw.device,
                dtype=raw.dtype,
            )
            positions = (
                self.sqrt_alpha_bars[start_t].to(raw.dtype) * baseline_positions
                + self.sqrt_one_minus_alpha_bars[start_t].to(raw.dtype) * epsilon
            )
        else:
            raise ValueError("initialization must be zero, gaussian, or input_layout")

        initial_positions = positions.detach().clone()
        features = model.encode_tiles(raw, restored, relative_graph)
        for index, timestep in enumerate(schedule):
            t = torch.full((batch,), timestep, device=raw.device, dtype=torch.long)
            predicted_x0 = model.denoise_from_features(
                features, positions, t, baseline_positions
            ).clamp(-clip_x0, clip_x0)
            previous = schedule[index + 1] if index + 1 < len(schedule) else -1
            if previous < 0:
                positions = predicted_x0
                continue
            alpha_bar = self.alpha_bars[timestep].to(dtype=positions.dtype)
            previous_alpha_bar = self.alpha_bars[previous].to(dtype=positions.dtype)
            epsilon = (positions - alpha_bar.sqrt() * predicted_x0) / (
                (1.0 - alpha_bar).sqrt().clamp_min(1.0e-6)
            )
            positions = (
                previous_alpha_bar.sqrt() * predicted_x0
                + (1.0 - previous_alpha_bar).sqrt() * epsilon
            )

        projections = tuple(
            project_positions_hungarian(positions[index], rows, columns)
            for index in range(batch)
        )
        return DiffusionSample(
            initial_positions=initial_positions,
            positions=positions,
            projections=projections,
            sampling_timesteps=schedule,
            initialization=initialization,
        )


def estimate_peak_memory_bytes(
    config: PositionalDiffusionConfig,
    *,
    batch_size: int,
    tile_count: int,
    tile_height: int = 20,
    tile_width: int = 20,
    bytes_per_value: int = 2,
    training: bool = True,
) -> dict[str, int]:
    """Conservative activation estimate for choosing a 2xT4 microbatch."""

    if batch_size <= 0 or tile_count <= 0 or bytes_per_value <= 0:
        raise ValueError("batch_size, tile_count, and bytes_per_value must be positive")
    attention = (
        batch_size
        * config.heads
        * tile_count
        * tile_count
        * config.layers
        * bytes_per_value
    )
    token_activations = (
        batch_size
        * tile_count
        * (4 * config.model_dim + 2 * config.feedforward_dim)
        * config.layers
        * bytes_per_value
    )
    graph = batch_size * 2 * tile_count * tile_count * bytes_per_value
    encoder_chunk = (
        min(config.tile_encode_chunk, batch_size * tile_count)
        * config.cnn_channels
        * tile_height
        * tile_width
        * bytes_per_value
        * 4
    )
    multiplier = 3 if training else 1
    activations = multiplier * (attention + token_activations) + graph + encoder_chunk
    return {
        "attention_logits": int(attention),
        "token_activations": int(token_activations),
        "relative_graph": int(graph),
        "checkpointed_encoder_chunk": int(encoder_chunk),
        "estimated_peak_activations": int(activations),
    }


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _load_validated_checkpoint_file(path: Path) -> dict[str, Any]:
    """Load one checkpoint path without consulting its ``.previous`` fallback."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("positional-diffusion checkpoint must contain a dictionary")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "puzzle_positional_diffusion"
    ):
        raise ValueError("unsupported positional-diffusion checkpoint")
    if payload.get("safe_for_submission") is not False:
        raise ValueError(
            "positional-diffusion checkpoint must be explicitly unsafe for submission"
        )
    if "model_config" not in payload or "model_state" not in payload:
        raise ValueError("positional-diffusion checkpoint is incomplete")
    return payload


def save_positional_diffusion_checkpoint(
    path: str | Path,
    model: PositionalDiffusionNet,
    *,
    metadata: Mapping[str, Any],
    optimizer_state: Mapping[str, Any] | None = None,
    scaler_state: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
    preserve_previous: bool = False,
) -> None:
    """Atomically save a provenance checkpoint, optionally retaining ``.previous``."""

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "puzzle_positional_diffusion",
        "safe_for_submission": False,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "metadata": dict(metadata),
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = dict(optimizer_state)
    if scaler_state is not None:
        payload["scaler_state"] = dict(scaler_state)
    if training_state is not None:
        payload["training_state"] = dict(training_state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    previous = path.with_name(f"{path.name}.previous")
    previous_temporary = previous.with_name(f".{previous.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        if preserve_previous and path.exists():
            try:
                _load_validated_checkpoint_file(path)
            except Exception:
                # A corrupt latest may have just been recovered through .previous.
                # Never rotate that corrupt file over the only known-good fallback.
                pass
            else:
                shutil.copy2(path, previous_temporary)
                os.replace(previous_temporary, previous)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        previous_temporary.unlink(missing_ok=True)


def load_positional_diffusion_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Load and validate the complete inference/training checkpoint payload."""

    requested = Path(path)
    candidates = (requested, requested.with_name(f"{requested.name}.previous"))
    failures: list[Exception] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = _load_validated_checkpoint_file(candidate)
            payload["loaded_checkpoint_path"] = str(candidate)
            payload["used_previous_fallback"] = candidate != requested
            return payload
        except Exception as error:  # a corrupt latest may recover from the previous epoch
            failures.append(error)
    if failures:
        raise ValueError(
            f"failed to load checkpoint {requested} or its previous fallback"
        ) from failures[-1]
    raise FileNotFoundError(requested)


def load_positional_diffusion_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[PositionalDiffusionNet, GaussianPositionDiffusion, dict[str, Any]]:
    payload = load_positional_diffusion_checkpoint_payload(path)
    config = PositionalDiffusionConfig(**payload["model_config"])
    model = PositionalDiffusionNet(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    diffusion = GaussianPositionDiffusion(config).to(device)
    return model, diffusion, dict(payload.get("metadata", {}))


__all__ = [
    "DiffusionSample",
    "GaussianPositionDiffusion",
    "PositionalDiffusionConfig",
    "PositionalDiffusionNet",
    "ProjectionResult",
    "RawDenoisedTileEncoder",
    "compatibility_to_relative_graph",
    "estimate_peak_memory_bytes",
    "grid_neighbor_vector_loss",
    "layout_to_tile_positions",
    "load_positional_diffusion_checkpoint",
    "load_positional_diffusion_checkpoint_payload",
    "model_parameter_count",
    "normalized_grid_positions",
    "project_positions_hungarian",
    "save_positional_diffusion_checkpoint",
]
