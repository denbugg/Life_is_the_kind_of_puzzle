"""Compact learned directional embeddings for denoised 20x20 tiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE, TILE_COUNT, inverse_permutation, validate_permutation


@dataclass(frozen=True)
class DirectionLabels:
    right_queries: np.ndarray
    right_targets: np.ndarray
    down_queries: np.ndarray
    down_targets: np.ndarray
    outside: np.ndarray


def direction_labels(slot_to_target: np.ndarray) -> DirectionLabels:
    slot_to_target = validate_permutation(slot_to_target, name="slot_to_target")
    position_to_slot = inverse_permutation(slot_to_target)
    right_positions = np.asarray(
        [position for position in range(TILE_COUNT) if position % GRID < GRID - 1],
        dtype=np.int32,
    )
    down_positions = np.arange(TILE_COUNT - GRID, dtype=np.int32)
    outside = np.zeros((TILE_COUNT, 4), dtype=np.float32)
    for slot, position in enumerate(slot_to_target.tolist()):
        row, column = divmod(position, GRID)
        outside[slot] = (column == 0, column == GRID - 1, row == 0, row == GRID - 1)
    return DirectionLabels(
        right_queries=position_to_slot[right_positions],
        right_targets=position_to_slot[right_positions + 1],
        down_queries=position_to_slot[down_positions],
        down_targets=position_to_slot[down_positions + GRID],
        outside=outside,
    )


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(values)
        values = F.silu(self.norm(self.pointwise(values)))
        return values + residual


class SideEmbeddingNet(nn.Module):
    """Encode four physical sides and emit directional query/key vectors."""

    def __init__(
        self,
        *,
        channels: int = 64,
        embedding_dim: int = 96,
        side_band: int = 4,
        tangent_bins: int = 10,
        temperature: float = 0.07,
        input_mode: str = "rgb_norm",
        edge_threshold: float = 0.12,
    ) -> None:
        super().__init__()
        if channels <= 0 or embedding_dim <= 0:
            raise ValueError("channels and embedding_dim must be positive")
        if channels % 8 != 0:
            raise ValueError("channels must be divisible by 8 for GroupNorm")
        if not 1 <= side_band <= TILE:
            raise ValueError("side_band must be in [1, 20]")
        if tangent_bins <= 0:
            raise ValueError("tangent_bins must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        input_channels = {
            "rgb_norm": 6,
            "rgb_sobel": 9,
            "sobel_only": 3,
            "binary_edges": 1,
        }
        if input_mode not in input_channels:
            raise ValueError(f"unsupported input_mode: {input_mode}")
        if not 0 < edge_threshold < 2:
            raise ValueError("edge_threshold must be in (0, 2)")
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.side_band = side_band
        self.tangent_bins = tangent_bins
        self.temperature = float(temperature)
        self.input_mode = input_mode
        self.edge_threshold = float(edge_threshold)
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
            ).unsqueeze(0)
            / 8.0,
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
            ).unsqueeze(0)
            / 8.0,
            persistent=False,
        )
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels[input_mode], channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            _ResidualBlock(channels),
            _ResidualBlock(channels),
        )
        feature_dim = channels * tangent_bins
        self.query_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embedding_dim),
        )
        self.key_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embedding_dim),
        )
        self.outside_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 1),
        )

    def _side_feature(self, features: torch.Tensor, side: str) -> torch.Tensor:
        band = self.side_band
        if side == "left":
            line = features[:, :, :, :band].mean(dim=3)
        elif side == "right":
            line = features[:, :, :, -band:].mean(dim=3)
        elif side == "up":
            line = features[:, :, :band, :].mean(dim=2)
        elif side == "down":
            line = features[:, :, -band:, :].mean(dim=2)
        else:
            raise ValueError(f"unknown side: {side}")
        line = F.adaptive_avg_pool1d(line, self.tangent_bins)
        return line.flatten(1)

    def _input_features(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=(2, 3), keepdim=True)
        std = values.std(dim=(2, 3), keepdim=True).clamp_min(1.0 / 255.0)
        normalized = ((values - mean) / std).clamp(-4.0, 4.0) / 4.0
        if self.input_mode == "rgb_norm":
            return torch.cat([values, normalized], dim=1)
        luma = (
            0.299 * values[:, 0:1]
            + 0.587 * values[:, 1:2]
            + 0.114 * values[:, 2:3]
        )
        padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")
        gradient_x = F.conv2d(padded, self.sobel_x)
        gradient_y = F.conv2d(padded, self.sobel_y)
        magnitude = torch.sqrt(
            gradient_x.square() + gradient_y.square() + 1e-8
        ).clamp_max(1.0)
        if self.input_mode == "rgb_sobel":
            return torch.cat(
                [values, normalized, gradient_x, gradient_y, magnitude], dim=1
            )
        if self.input_mode == "sobel_only":
            return torch.cat([gradient_x, gradient_y, magnitude], dim=1)
        return (magnitude >= self.edge_threshold).to(values.dtype)

    def forward(self, tiles: torch.Tensor) -> dict[str, torch.Tensor]:
        if tiles.ndim != 4 or tiles.shape[1:] != (3, TILE, TILE):
            raise ValueError(f"expected NCHW tiles with shape (*,3,20,20), got {tiles.shape}")
        values = tiles.float()
        if values.detach().amax() > 1.5:
            values = values / 255.0
        features = self.stem(self._input_features(values))
        sides = {
            side: self._side_feature(features, side)
            for side in ("left", "right", "up", "down")
        }
        outside = torch.cat(
            [self.outside_head(sides[side]) for side in ("left", "right", "up", "down")],
            dim=1,
        )
        raw_q_right = self.query_projection(sides["right"])
        raw_k_left = self.key_projection(sides["left"])
        raw_q_down = self.query_projection(sides["down"])
        raw_k_up = self.key_projection(sides["up"])
        return {
            "q_right": F.normalize(raw_q_right, dim=1),
            "k_left": F.normalize(raw_k_left, dim=1),
            "q_down": F.normalize(raw_q_down, dim=1),
            "k_up": F.normalize(raw_k_up, dim=1),
            "raw_q_right": raw_q_right,
            "raw_k_left": raw_k_left,
            "raw_q_down": raw_q_down,
            "raw_k_up": raw_k_up,
            "outside_logits": outside,
        }

    def config(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "embedding_dim": self.embedding_dim,
            "side_band": self.side_band,
            "tangent_bins": self.tangent_bins,
            "temperature": self.temperature,
            "input_mode": self.input_mode,
            "edge_threshold": self.edge_threshold,
        }


def _log_sinkhorn_iterations(
    scores: torch.Tensor,
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
    *,
    iterations: int,
) -> torch.Tensor:
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
    return scores + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(
    scores: torch.Tensor,
    bin_score: torch.Tensor,
    *,
    iterations: int,
) -> torch.Tensor:
    """SuperGlue-style partial assignment with one dustbin row and column."""
    if scores.ndim != 3 or scores.shape[1] != scores.shape[2]:
        raise ValueError("scores must have shape (B,N,N)")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    batch, rows, columns = scores.shape
    bins0 = bin_score.to(scores).expand(batch, rows, 1)
    bins1 = bin_score.to(scores).expand(batch, 1, columns)
    corner = bin_score.to(scores).expand(batch, 1, 1)
    augmented = torch.cat(
        [torch.cat([scores, bins0], dim=2), torch.cat([bins1, corner], dim=2)],
        dim=1,
    )
    norm = -torch.log(scores.new_tensor(float(rows + columns)))
    log_mu = torch.cat(
        [
            norm.expand(rows),
            (torch.log(scores.new_tensor(float(columns))) + norm).unsqueeze(0),
        ]
    ).expand(batch, -1)
    log_nu = torch.cat(
        [
            norm.expand(columns),
            (torch.log(scores.new_tensor(float(rows))) + norm).unsqueeze(0),
        ]
    ).expand(batch, -1)
    return _log_sinkhorn_iterations(
        augmented, log_mu, log_nu, iterations=iterations
    ) - norm


class GlobalSuccessorMatcher(nn.Module):
    """Permutation-equivariant global successor matcher over all 576 tiles."""

    def __init__(
        self,
        *,
        embedding_dim: int = 320,
        model_dim: int = 128,
        layers: int = 3,
        heads: int = 4,
        feedforward_dim: int = 256,
        sinkhorn_iterations: int = 20,
        dropout: float = 0.0,
        base_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or model_dim <= 0 or layers <= 0:
            raise ValueError("embedding/model dimensions and layers must be positive")
        if heads <= 0 or model_dim % heads:
            raise ValueError("heads must divide model_dim")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if base_temperature <= 0:
            raise ValueError("base_temperature must be positive")
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.layers = layers
        self.heads = heads
        self.feedforward_dim = feedforward_dim
        self.sinkhorn_iterations = sinkhorn_iterations
        self.dropout = float(dropout)
        self.base_temperature = float(base_temperature)
        self.input_projection = nn.Linear(embedding_dim + 1, model_dim)
        self.role_embedding = nn.Parameter(torch.zeros(2, model_dim))
        nn.init.normal_(self.role_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.07), dtype=torch.float32))
        self.context_gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.bin_score = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = ("q_right", "k_left", "q_down", "k_up", "outside_logits")
        if any(name not in embeddings for name in required):
            raise ValueError("missing frozen side-embedding outputs")
        outside = embeddings["outside_logits"]
        if outside.shape != (TILE_COUNT, 4):
            raise ValueError("outside_logits must have shape (576,4)")
        queries = torch.stack([embeddings["q_right"], embeddings["q_down"]], dim=0)
        keys = torch.stack([embeddings["k_left"], embeddings["k_up"]], dim=0)
        if queries.shape != (2, TILE_COUNT, self.embedding_dim):
            raise ValueError("unexpected frozen embedding shape")
        query_outside = torch.stack([outside[:, 1], outside[:, 3]], dim=0).unsqueeze(2)
        key_outside = torch.stack([outside[:, 0], outside[:, 2]], dim=0).unsqueeze(2)
        query_values = self.input_projection(torch.cat([queries, query_outside], dim=2))
        key_values = self.input_projection(torch.cat([keys, key_outside], dim=2))
        query_values = query_values + self.role_embedding[0]
        key_values = key_values + self.role_embedding[1]
        contextual = self.context(torch.cat([query_values, key_values], dim=1))
        contextual = F.normalize(self.output_norm(contextual), dim=2)
        contextual_queries = contextual[:, :TILE_COUNT]
        contextual_keys = contextual[:, TILE_COUNT:]
        scale = self.logit_scale.exp().clamp(max=100.0)
        contextual_scores = (
            torch.einsum("bnd,bmd->bnm", contextual_queries, contextual_keys) * scale
        )
        baseline_scores = torch.einsum("bnd,bmd->bnm", queries, keys) / self.base_temperature
        scores = baseline_scores + self.context_gain * contextual_scores
        diagonal = torch.arange(TILE_COUNT, device=scores.device)
        scores[:, diagonal, diagonal] = -1e4
        assignment = log_optimal_transport(
            scores, self.bin_score, iterations=self.sinkhorn_iterations
        )
        return {"scores": scores, "log_assignment": assignment}

    def config(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "model_dim": self.model_dim,
            "layers": self.layers,
            "heads": self.heads,
            "feedforward_dim": self.feedforward_dim,
            "sinkhorn_iterations": self.sinkhorn_iterations,
            "dropout": self.dropout,
            "base_temperature": self.base_temperature,
        }


class SideSequenceEmbeddingNet(nn.Module):
    """L1-v2: preserve all 20 tangent positions in directional embeddings."""

    def __init__(
        self,
        *,
        channels: int = 64,
        embedding_dim: int = 32,
        side_band: int = 4,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if channels <= 0 or channels % 8 != 0 or embedding_dim <= 0:
            raise ValueError("channels must be positive/divisible by 8 and dim positive")
        if not 1 <= side_band <= TILE:
            raise ValueError("side_band must be in [1, 20]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.side_band = side_band
        self.temperature = float(temperature)
        self.stem = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            _ResidualBlock(channels),
            _ResidualBlock(channels),
        )
        self.line_encoder = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.query_projection = nn.Conv1d(channels, embedding_dim, 1)
        self.key_projection = nn.Conv1d(channels, embedding_dim, 1)
        self.outside_head = nn.Linear(channels, 1)

    def _side_line(self, features: torch.Tensor, side: str) -> torch.Tensor:
        band = self.side_band
        if side == "left":
            line = features[:, :, :, :band].mean(dim=3)
        elif side == "right":
            line = features[:, :, :, -band:].mean(dim=3)
        elif side == "up":
            line = features[:, :, :band, :].mean(dim=2)
        elif side == "down":
            line = features[:, :, -band:, :].mean(dim=2)
        else:
            raise ValueError(f"unknown side: {side}")
        return self.line_encoder(line) + line

    def _project(self, line: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        values = projection(line).transpose(1, 2)
        return F.normalize(values, dim=2)

    def forward(self, tiles: torch.Tensor) -> dict[str, torch.Tensor]:
        if tiles.ndim != 4 or tiles.shape[1:] != (3, TILE, TILE):
            raise ValueError("expected NCHW tiles with shape (*,3,20,20)")
        values = tiles.float()
        if values.detach().amax() > 1.5:
            values = values / 255.0
        mean = values.mean(dim=(2, 3), keepdim=True)
        std = values.std(dim=(2, 3), keepdim=True).clamp_min(1.0 / 255.0)
        normalized = ((values - mean) / std).clamp(-4.0, 4.0) / 4.0
        features = self.stem(torch.cat([values, normalized], dim=1))
        sides = {
            side: self._side_line(features, side)
            for side in ("left", "right", "up", "down")
        }
        outside = torch.cat(
            [self.outside_head(sides[side].mean(dim=2)) for side in ("left", "right", "up", "down")],
            dim=1,
        )
        return {
            "q_right": self._project(sides["right"], self.query_projection),
            "k_left": self._project(sides["left"], self.key_projection),
            "q_down": self._project(sides["down"], self.query_projection),
            "k_up": self._project(sides["up"], self.key_projection),
            "outside_logits": outside,
        }

    def config(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "embedding_dim": self.embedding_dim,
            "side_band": self.side_band,
            "temperature": self.temperature,
        }


class SeamPairNet(nn.Module):
    """Compact L0 pair CNN over a canonical stitched denoised seam."""

    def __init__(self, *, channels: int = 48, side_band: int = 6) -> None:
        super().__init__()
        if channels <= 0 or channels % 8 != 0:
            raise ValueError("channels must be positive and divisible by 8")
        if not 2 <= side_band <= 10:
            raise ValueError("side_band must be in [2, 10]")
        self.channels = channels
        self.side_band = side_band
        self.encoder = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            _ResidualBlock(channels),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.SiLU(),
            _ResidualBlock(channels * 2),
            nn.AdaptiveAvgPool2d((5, 3)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2 * 5 * 3, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, 1),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 4 or patches.shape[1] != 3:
            raise ValueError("patches must have shape Nx3x20x(2*band)")
        if patches.shape[2:] != (TILE, 2 * self.side_band):
            raise ValueError(
                f"expected patch spatial shape {(TILE, 2 * self.side_band)}, got {patches.shape[2:]}"
            )
        values = patches.float()
        if values.detach().amax() > 1.5:
            values = values / 255.0
        left, right = values.chunk(2, dim=3)
        normalized_parts = []
        for half in (left, right):
            mean = half.mean(dim=(2, 3), keepdim=True)
            std = half.std(dim=(2, 3), keepdim=True).clamp_min(1.0 / 255.0)
            normalized_parts.append(((half - mean) / std).clamp(-4.0, 4.0) / 4.0)
        normalized = torch.cat(normalized_parts, dim=3)
        horizontal = F.pad(values[:, :, :, 1:] - values[:, :, :, :-1], (0, 1, 0, 0))
        vertical = F.pad(values[:, :, 1:, :] - values[:, :, :-1, :], (0, 0, 0, 1))
        features = torch.cat([values, normalized, horizontal, vertical], dim=1)
        return self.head(self.encoder(features)).squeeze(1)

    def config(self) -> dict[str, Any]:
        return {"channels": self.channels, "side_band": self.side_band}


class RankFeatureNet(nn.Module):
    """X0: rerank a sparse candidate graph from calibrated classical ranks."""

    def __init__(self, *, feature_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[2] != self.feature_dim:
            raise ValueError(
                f"features must be QxKx{self.feature_dim}, got {tuple(features.shape)}"
            )
        return self.network(features.float()).squeeze(2)

    def config(self) -> dict[str, int]:
        return {"feature_dim": self.feature_dim, "hidden_dim": self.hidden_dim}


class PositionPriorHead(nn.Module):
    """L2b: predict broad row/column priors from frozen side embeddings."""

    def __init__(self, *, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.row_head = nn.Linear(hidden_dim, GRID)
        self.column_head = nn.Linear(hidden_dim, GRID)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must be Nx{self.feature_dim}, got {tuple(features.shape)}"
            )
        values = self.trunk(features.float())
        return self.row_head(values), self.column_head(values)

    def config(self) -> dict[str, int]:
        return {"feature_dim": self.feature_dim, "hidden_dim": self.hidden_dim}


class ContextPositionTransformer(nn.Module):
    """T0: permutation-equivariant 576-tile context model for absolute positions."""

    def __init__(
        self,
        *,
        model_dim: int = 128,
        layers: int = 4,
        heads: int = 8,
        feedforward_dim: int = 512,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or model_dim % 8 != 0:
            raise ValueError("model_dim must be positive and divisible by 8")
        if layers <= 0 or heads <= 0 or model_dim % heads != 0:
            raise ValueError("invalid transformer layer/head configuration")
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive")
        self.model_dim = model_dim
        self.layers = layers
        self.heads = heads
        self.feedforward_dim = feedforward_dim
        stem_channels = max(32, model_dim // 2)
        stem_channels = int(np.ceil(stem_channels / 8) * 8)
        self.tile_encoder = nn.Sequential(
            nn.Conv2d(6, stem_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, stem_channels),
            nn.SiLU(),
            _ResidualBlock(stem_channels),
            nn.Conv2d(stem_channels, model_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, model_dim),
            nn.SiLU(),
            _ResidualBlock(model_dim),
            nn.AdaptiveAvgPool2d(1),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(model_dim)
        self.row_head = nn.Linear(model_dim, GRID)
        self.column_head = nn.Linear(model_dim, GRID)

    def forward(self, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tiles.ndim == 4:
            tiles = tiles.unsqueeze(0)
        if tiles.ndim != 5 or tiles.shape[2:] != (3, TILE, TILE):
            raise ValueError("tiles must have shape BxNx3x20x20 or Nx3x20x20")
        batch, count = tiles.shape[:2]
        values = tiles.float()
        if values.detach().amax() > 1.5:
            values = values / 255.0
        mean = values.mean(dim=(3, 4), keepdim=True)
        std = values.std(dim=(3, 4), keepdim=True).clamp_min(1.0 / 255.0)
        normalized = ((values - mean) / std).clamp(-4.0, 4.0) / 4.0
        encoded = self.tile_encoder(
            torch.cat([values, normalized], dim=2).reshape(batch * count, 6, TILE, TILE)
        ).flatten(1)
        encoded = encoded.reshape(batch, count, self.model_dim)
        contextual = self.norm(self.context(encoded))
        return self.row_head(contextual), self.column_head(contextual)

    def config(self) -> dict[str, int]:
        return {
            "model_dim": self.model_dim,
            "layers": self.layers,
            "heads": self.heads,
            "feedforward_dim": self.feedforward_dim,
        }


@torch.inference_mode()
def context_position_logits(
    model: ContextPositionTransformer,
    tiles: np.ndarray,
    *,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tiles)
    if values.shape != (TILE_COUNT, TILE, TILE, 3) or values.dtype != np.uint8:
        raise ValueError("tiles must be uint8 with shape (576,20,20,3)")
    tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    )
    model.eval()
    row_logits, column_logits = model(tensor)
    return (
        row_logits.squeeze(0).float().cpu().numpy(),
        column_logits.squeeze(0).float().cpu().numpy(),
    )


def save_context_position_checkpoint(
    path: str | Path,
    model: ContextPositionTransformer,
    *,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "puzzle_context_position_t0",
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_context_position_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[ContextPositionTransformer, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("kind") != "puzzle_context_position_t0":
        raise ValueError("unsupported context-position checkpoint")
    model = ContextPositionTransformer(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata", {}))


def embedding_position_features(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = []
    for name in ("q_right", "k_left", "q_down", "k_up"):
        values = outputs[name]
        if values.ndim == 3:
            values = values.mean(dim=1)
        if values.ndim != 2:
            raise ValueError("directional embeddings must be 2D or 3D")
        parts.append(values)
    parts.append(outputs["outside_logits"])
    return torch.cat(parts, dim=1)


@torch.inference_mode()
def learned_position_logits(
    embedding_model: SideEmbeddingNet | SideSequenceEmbeddingNet,
    position_head: PositionPriorHead,
    tiles: np.ndarray,
    *,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tiles)
    if values.shape != (TILE_COUNT, TILE, TILE, 3) or values.dtype != np.uint8:
        raise ValueError("tiles must be uint8 with shape (576,20,20,3)")
    tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    )
    embedding_model.eval()
    position_head.eval()
    features = embedding_position_features(embedding_model(tensor))
    row_logits, column_logits = position_head(features)
    return row_logits.float().cpu().numpy(), column_logits.float().cpu().numpy()


def save_position_prior_checkpoint(
    path: str | Path,
    model: PositionPriorHead,
    *,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "puzzle_position_prior_l2b",
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_position_prior_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[PositionPriorHead, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("kind") != "puzzle_position_prior_l2b":
        raise ValueError("unsupported position-prior checkpoint")
    model = PositionPriorHead(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata", {}))


def _rank_and_robust_features(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError("compatibility matrices must be 576x576")
    finite = values.copy()
    finite[~np.isfinite(finite)] = np.nan

    row_order = np.argsort(values, axis=1, kind="stable")
    row_rank = np.empty_like(row_order)
    np.put_along_axis(
        row_rank,
        row_order,
        np.broadcast_to(np.arange(TILE_COUNT), row_order.shape),
        axis=1,
    )
    column_order = np.argsort(values, axis=0, kind="stable")
    column_rank = np.empty_like(column_order)
    np.put_along_axis(
        column_rank,
        column_order,
        np.broadcast_to(np.arange(TILE_COUNT)[:, None], column_order.shape),
        axis=0,
    )

    row_median = np.nanmedian(finite, axis=1, keepdims=True)
    row_scale = np.nanpercentile(finite, 75, axis=1, keepdims=True) - np.nanpercentile(
        finite, 25, axis=1, keepdims=True
    )
    column_median = np.nanmedian(finite, axis=0, keepdims=True)
    column_scale = np.nanpercentile(finite, 75, axis=0, keepdims=True) - np.nanpercentile(
        finite, 25, axis=0, keepdims=True
    )
    row_robust = np.clip(
        (values - row_median) / np.maximum(row_scale, 1e-8), -8.0, 8.0
    ) / 8.0
    column_robust = np.clip(
        (values - column_median) / np.maximum(column_scale, 1e-8), -8.0, 8.0
    ) / 8.0
    return (
        row_rank.astype(np.float32) / (TILE_COUNT - 1),
        column_rank.astype(np.float32) / (TILE_COUNT - 1),
        row_robust.astype(np.float32),
        column_robust.astype(np.float32),
    )


def candidate_rank_features(
    score_bank: dict[str, CompatibilityMatrices],
    candidates: tuple[np.ndarray, np.ndarray],
    *,
    names: list[str],
) -> np.ndarray:
    """Return 2x576xKxF edge features without using target information."""
    if not names or any(name not in score_bank for name in names):
        raise ValueError("all requested feature score names must exist")
    if candidates[0].shape != candidates[1].shape or candidates[0].shape[0] != TILE_COUNT:
        raise ValueError("candidate arrays must have matching 576xK shapes")
    outputs = []
    for direction, direction_candidates in enumerate(candidates):
        rows = np.arange(TILE_COUNT)[:, None]
        parts = []
        gathered_row_ranks = []
        for name in names:
            matrix = getattr(score_bank[name], "right" if direction == 0 else "down")
            row_rank, column_rank, row_robust, column_robust = _rank_and_robust_features(
                matrix
            )
            parts.extend(
                [
                    row_rank[rows, direction_candidates],
                    column_rank[rows, direction_candidates],
                    row_robust[rows, direction_candidates],
                    column_robust[rows, direction_candidates],
                ]
            )
            gathered_row_ranks.append(row_rank[rows, direction_candidates])
        rank_stack = np.stack(gathered_row_ranks, axis=2)
        parts.extend(
            [
                rank_stack.min(axis=2),
                rank_stack.mean(axis=2),
                rank_stack.std(axis=2),
                np.full(direction_candidates.shape, float(direction), dtype=np.float32),
            ]
        )
        outputs.append(np.stack(parts, axis=2).astype(np.float32))
    return np.stack(outputs, axis=0)


@torch.inference_mode()
def rank_feature_compatibility(
    model: RankFeatureNet,
    features: np.ndarray,
    candidates: tuple[np.ndarray, np.ndarray],
    *,
    device: torch.device | str,
    batch_queries: int = 128,
    name: str = "denoised_x0_rank_reranker",
) -> CompatibilityMatrices:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 4 or values.shape[:2] != (2, TILE_COUNT):
        raise ValueError("features must have shape 2x576xKxF")
    matrices = []
    model.eval()
    for direction, direction_candidates in enumerate(candidates):
        logits = []
        for start in range(0, TILE_COUNT, batch_queries):
            batch = torch.from_numpy(values[direction, start : start + batch_queries]).to(
                device=device
            )
            logits.append(model(batch).float().cpu().numpy())
        logits_array = np.concatenate(logits)
        matrix = np.full((TILE_COUNT, TILE_COUNT), 1e6, dtype=np.float32)
        rows = np.arange(TILE_COUNT)[:, None]
        matrix[rows, direction_candidates] = -logits_array
        np.fill_diagonal(matrix, np.inf)
        matrices.append(matrix)
    return CompatibilityMatrices(name, matrices[0], matrices[1])


def save_rank_feature_checkpoint(
    path: str | Path,
    model: RankFeatureNet,
    *,
    feature_names: list[str],
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "puzzle_rank_feature_x0",
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "feature_names": list(feature_names),
            "metadata": metadata,
        },
        path,
    )


def load_rank_feature_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[RankFeatureNet, list[str], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("kind") != "puzzle_rank_feature_x0":
        raise ValueError("unsupported rank-feature checkpoint")
    model = RankFeatureNet(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, list(payload["feature_names"]), dict(payload.get("metadata", {}))


def seam_pair_patches(
    tiles: torch.Tensor,
    first_slots: torch.Tensor,
    second_slots: torch.Tensor,
    directions: torch.Tensor,
    *,
    side_band: int,
) -> torch.Tensor:
    """Return right/down pairs as canonical left|right stitched strips."""
    if tiles.ndim != 4 or tiles.shape[1:] != (3, TILE, TILE):
        raise ValueError("tiles must be NCHW 20x20")
    first = tiles[first_slots]
    second = tiles[second_slots]
    result = torch.empty(
        (len(first_slots), 3, TILE, 2 * side_band),
        device=tiles.device,
        dtype=tiles.dtype,
    )
    right_mask = directions == 0
    if right_mask.any():
        result[right_mask] = torch.cat(
            [first[right_mask, :, :, -side_band:], second[right_mask, :, :, :side_band]],
            dim=3,
        )
    down_mask = ~right_mask
    if down_mask.any():
        first_down = first[down_mask, :, -side_band:, :].transpose(2, 3)
        second_up = second[down_mask, :, :side_band, :].transpose(2, 3)
        result[down_mask] = torch.cat([first_down, second_up], dim=3)
    return result


def candidate_union(
    score_bank: dict[str, CompatibilityMatrices],
    *,
    names: list[str],
    per_score_top_k: int = 32,
    cap: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    if not names or per_score_top_k <= 0 or cap <= 0:
        raise ValueError("candidate union requires names and positive limits")
    directions = []
    for direction_name in ("right", "down"):
        output = np.empty((TILE_COUNT, cap), dtype=np.int32)
        for query in range(TILE_COUNT):
            reciprocal_rank: dict[int, float] = {}
            for name in names:
                matrix = getattr(score_bank[name], direction_name)
                order = np.argsort(matrix[query], kind="stable")[:per_score_top_k]
                for rank, candidate in enumerate(order.tolist(), start=1):
                    if candidate == query:
                        continue
                    reciprocal_rank[candidate] = reciprocal_rank.get(candidate, 0.0) + 1.0 / (
                        60.0 + rank
                    )
            candidates = sorted(
                reciprocal_rank,
                key=lambda candidate: (-reciprocal_rank[candidate], candidate),
            )
            seen = set(candidates)
            if len(candidates) < cap:
                fallback = np.argsort(
                    getattr(score_bank[names[0]], direction_name)[query], kind="stable"
                )
                for candidate in fallback.tolist():
                    if candidate != query and candidate not in seen:
                        seen.add(candidate)
                        candidates.append(candidate)
                    if len(candidates) >= cap:
                        break
            output[query] = np.asarray(candidates[:cap], dtype=np.int32)
        directions.append(output)
    return directions[0], directions[1]


@torch.inference_mode()
def pair_rerank_compatibility(
    model: SeamPairNet,
    tiles: np.ndarray,
    candidates: tuple[np.ndarray, np.ndarray],
    *,
    device: torch.device | str,
    batch_size: int = 4096,
    name: str = "denoised_l0_pair",
) -> CompatibilityMatrices:
    values = np.asarray(tiles)
    if values.shape != (TILE_COUNT, TILE, TILE, 3) or values.dtype != np.uint8:
        raise ValueError("tiles must be uint8 with shape (576,20,20,3)")
    tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    )
    matrices = []
    model.eval()
    for direction, direction_candidates in enumerate(candidates):
        if direction_candidates.ndim != 2 or direction_candidates.shape[0] != TILE_COUNT:
            raise ValueError("candidate arrays must have 576 rows")
        cap = direction_candidates.shape[1]
        first = np.repeat(np.arange(TILE_COUNT, dtype=np.int64), cap)
        second = direction_candidates.reshape(-1).astype(np.int64)
        direction_values = np.full(len(first), direction, dtype=np.int64)
        scores = []
        for start in range(0, len(first), batch_size):
            sl = slice(start, start + batch_size)
            patches = seam_pair_patches(
                tensor,
                torch.as_tensor(first[sl], device=device),
                torch.as_tensor(second[sl], device=device),
                torch.as_tensor(direction_values[sl], device=device),
                side_band=model.side_band,
            )
            scores.append(model(patches).float().cpu().numpy())
        scores_array = np.concatenate(scores).reshape(TILE_COUNT, cap)
        matrix = np.full((TILE_COUNT, TILE_COUNT), 1e6, dtype=np.float32)
        rows = np.arange(TILE_COUNT)[:, None]
        matrix[rows, direction_candidates] = -scores_array
        np.fill_diagonal(matrix, np.inf)
        matrices.append(matrix)
    return CompatibilityMatrices(name, matrices[0], matrices[1])


def save_pair_checkpoint(
    path: str | Path, model: SeamPairNet, *, metadata: dict[str, Any]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "puzzle_seam_pair_l0",
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_pair_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[SeamPairNet, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("kind") != "puzzle_seam_pair_l0":
        raise ValueError("unsupported seam-pair checkpoint")
    model = SeamPairNet(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata", {}))

    def config(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "embedding_dim": self.embedding_dim,
            "side_band": self.side_band,
            "tangent_bins": self.tangent_bins,
            "temperature": self.temperature,
        }


def _masked_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    query_slots: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    selected_query = query[query_slots]
    if query.ndim == 2 and key.ndim == 2:
        logits = selected_query @ key.T
    elif query.ndim == 3 and key.ndim == 3:
        logits = torch.einsum("qtd,ntd->qn", selected_query, key) / query.shape[1]
    else:
        raise ValueError("query/key embeddings must both be 2D or both be 3D")
    logits = logits / temperature
    rows = torch.arange(len(query_slots), device=logits.device)
    logits[rows, query_slots] = torch.finfo(logits.dtype).min
    return logits


def embedding_loss(
    outputs: dict[str, torch.Tensor],
    labels: DirectionLabels,
    *,
    temperature: float,
    outside_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["q_right"].device
    right_queries = torch.as_tensor(labels.right_queries, device=device, dtype=torch.long)
    right_targets = torch.as_tensor(labels.right_targets, device=device, dtype=torch.long)
    down_queries = torch.as_tensor(labels.down_queries, device=device, dtype=torch.long)
    down_targets = torch.as_tensor(labels.down_targets, device=device, dtype=torch.long)
    right_logits = _masked_logits(
        outputs["q_right"], outputs["k_left"], right_queries, temperature=temperature
    )
    down_logits = _masked_logits(
        outputs["q_down"], outputs["k_up"], down_queries, temperature=temperature
    )
    right_loss = F.cross_entropy(right_logits, right_targets)
    down_loss = F.cross_entropy(down_logits, down_targets)
    outside_targets = torch.as_tensor(labels.outside, device=device)
    outside_loss = F.binary_cross_entropy_with_logits(
        outputs["outside_logits"], outside_targets
    )
    loss = 0.5 * (right_loss + down_loss) + outside_weight * outside_loss
    with torch.no_grad():
        right_top1 = (right_logits.argmax(dim=1) == right_targets).float().mean()
        down_top1 = (down_logits.argmax(dim=1) == down_targets).float().mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "right_loss": float(right_loss.detach().cpu()),
        "down_loss": float(down_loss.detach().cpu()),
        "outside_loss": float(outside_loss.detach().cpu()),
        "recall_at_1": float((0.5 * (right_top1 + down_top1)).detach().cpu()),
    }


def embedding_hard_triplet_loss(
    outputs: dict[str, torch.Tensor],
    labels: DirectionLabels,
    *,
    temperature: float,
    margin: float = 0.2,
    cross_entropy_weight: float = 0.25,
    embedding_l2_weight: float = 1e-4,
    outside_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hardest-negative directional loss over all candidates in one puzzle."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    if cross_entropy_weight < 0 or embedding_l2_weight < 0 or outside_weight < 0:
        raise ValueError("loss weights must be non-negative")
    device = outputs["q_right"].device
    directional: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for query_name, key_name, queries_array, targets_array in (
        ("q_right", "k_left", labels.right_queries, labels.right_targets),
        ("q_down", "k_up", labels.down_queries, labels.down_targets),
    ):
        queries = torch.as_tensor(queries_array, device=device, dtype=torch.long)
        targets = torch.as_tensor(targets_array, device=device, dtype=torch.long)
        logits = _masked_logits(
            outputs[query_name], outputs[key_name], queries, temperature=temperature
        )
        rows = torch.arange(len(queries), device=device)
        positive = logits[rows, targets]
        negative_logits = logits.clone()
        negative_logits[rows, targets] = torch.finfo(logits.dtype).min
        hardest_negative = negative_logits.max(dim=1).values
        triplet = F.relu(margin + hardest_negative - positive).mean()
        cross_entropy = F.cross_entropy(logits, targets)
        top1 = (logits.argmax(dim=1) == targets).float().mean()
        directional.append((triplet, cross_entropy, top1))

    triplet_loss = 0.5 * (directional[0][0] + directional[1][0])
    cross_entropy_loss = 0.5 * (directional[0][1] + directional[1][1])
    raw_names = ("raw_q_right", "raw_k_left", "raw_q_down", "raw_k_up")
    if all(name in outputs for name in raw_names):
        embedding_l2 = torch.stack(
            [outputs[name].square().mean() for name in raw_names]
        ).mean()
    else:
        embedding_l2 = torch.zeros((), device=device, dtype=triplet_loss.dtype)
    outside_targets = torch.as_tensor(labels.outside, device=device)
    outside_loss = F.binary_cross_entropy_with_logits(
        outputs["outside_logits"], outside_targets
    )
    loss = (
        triplet_loss
        + cross_entropy_weight * cross_entropy_loss
        + embedding_l2_weight * embedding_l2
        + outside_weight * outside_loss
    )
    recall_at_1 = 0.5 * (directional[0][2] + directional[1][2])
    return loss, {
        "loss": float(loss.detach().cpu()),
        "triplet_loss": float(triplet_loss.detach().cpu()),
        "cross_entropy_loss": float(cross_entropy_loss.detach().cpu()),
        "embedding_l2": float(embedding_l2.detach().cpu()),
        "outside_loss": float(outside_loss.detach().cpu()),
        "recall_at_1": float(recall_at_1.detach().cpu()),
    }


@torch.no_grad()
def embedding_retrieval_metrics(
    outputs: dict[str, torch.Tensor],
    labels: DirectionLabels,
    *,
    temperature: float,
    ks: tuple[int, ...] = (1, 5, 10, 20, 32),
) -> dict[str, float]:
    device = outputs["q_right"].device
    directional = []
    for query_name, key_name, queries_array, targets_array in (
        ("q_right", "k_left", labels.right_queries, labels.right_targets),
        ("q_down", "k_up", labels.down_queries, labels.down_targets),
    ):
        queries = torch.as_tensor(queries_array, device=device, dtype=torch.long)
        targets = torch.as_tensor(targets_array, device=device, dtype=torch.long)
        logits = _masked_logits(
            outputs[query_name], outputs[key_name], queries, temperature=temperature
        )
        order = logits.argsort(dim=1, descending=True)
        rank = (order == targets[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        directional.append(rank)
    rank = torch.cat(directional).float()
    metrics = {
        f"recall_at_{k}": float((rank <= k).float().mean().cpu()) for k in ks
    }
    metrics["mrr"] = float((1.0 / rank).mean().cpu())
    return metrics


def global_matching_loss(
    outputs: dict[str, torch.Tensor], labels: DirectionLabels
) -> tuple[torch.Tensor, dict[str, float]]:
    assignment = outputs["log_assignment"]
    if assignment.shape != (2, TILE_COUNT + 1, TILE_COUNT + 1):
        raise ValueError("global assignment must have shape (2,577,577)")
    device = assignment.device
    outside = torch.as_tensor(labels.outside, device=device)
    directional_losses = []
    for direction, queries_array, targets_array, query_side, key_side in (
        (0, labels.right_queries, labels.right_targets, 1, 0),
        (1, labels.down_queries, labels.down_targets, 3, 2),
    ):
        queries = torch.as_tensor(queries_array, device=device, dtype=torch.long)
        targets = torch.as_tensor(targets_array, device=device, dtype=torch.long)
        unmatched_queries = torch.nonzero(outside[:, query_side] > 0.5, as_tuple=False)[:, 0]
        unmatched_keys = torch.nonzero(outside[:, key_side] > 0.5, as_tuple=False)[:, 0]
        negative_log_likelihood = torch.cat(
            [
                -assignment[direction, queries, targets],
                -assignment[direction, unmatched_queries, TILE_COUNT],
                -assignment[direction, TILE_COUNT, unmatched_keys],
            ]
        )
        directional_losses.append(negative_log_likelihood.mean())
    loss = 0.5 * (directional_losses[0] + directional_losses[1])
    metrics = global_matching_metrics(outputs, labels)
    return loss, {"loss": float(loss.detach().cpu()), **metrics}


@torch.no_grad()
def global_matching_metrics(
    outputs: dict[str, torch.Tensor],
    labels: DirectionLabels,
    *,
    ks: tuple[int, ...] = (1, 5, 10, 20, 32),
) -> dict[str, float]:
    assignment = outputs["log_assignment"][:, :TILE_COUNT, :TILE_COUNT]
    device = assignment.device
    ranks = []
    reciprocal_correct = 0
    reciprocal_total = 0
    total_matches = 0
    for direction, queries_array, targets_array in (
        (0, labels.right_queries, labels.right_targets),
        (1, labels.down_queries, labels.down_targets),
    ):
        queries = torch.as_tensor(queries_array, device=device, dtype=torch.long)
        targets = torch.as_tensor(targets_array, device=device, dtype=torch.long)
        selected = assignment[direction, queries]
        order = selected.argsort(dim=1, descending=True)
        rank = (order == targets[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        ranks.append(rank)
        best_key = assignment[direction].argmax(dim=1)
        best_query = assignment[direction].argmax(dim=0)
        reciprocal = best_query[best_key[queries]] == queries
        reciprocal_total += int(reciprocal.sum())
        reciprocal_correct += int(
            (reciprocal & (best_key[queries] == targets)).sum()
        )
        total_matches += len(queries)
    rank = torch.cat(ranks).float()
    metrics = {f"recall_at_{k}": float((rank <= k).float().mean().cpu()) for k in ks}
    metrics["mrr"] = float((1.0 / rank).mean().cpu())
    metrics["mutual_precision"] = reciprocal_correct / max(reciprocal_total, 1)
    metrics["mutual_recall"] = reciprocal_correct / max(total_matches, 1)
    return metrics


def save_global_matcher_checkpoint(
    path: str | Path,
    model: GlobalSuccessorMatcher,
    *,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "puzzle_global_successor_matcher_g0",
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_global_matcher_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[GlobalSuccessorMatcher, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "puzzle_global_successor_matcher_g0"
    ):
        raise ValueError("unsupported global-successor checkpoint")
    model = GlobalSuccessorMatcher(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata", {}))


@torch.inference_mode()
def learned_compatibility(
    model: SideEmbeddingNet | SideSequenceEmbeddingNet,
    tiles: np.ndarray,
    *,
    device: torch.device | str,
    name: str = "denoised_l1_embedding",
) -> tuple[CompatibilityMatrices, np.ndarray]:
    tiles = np.asarray(tiles)
    if tiles.shape != (TILE_COUNT, TILE, TILE, 3) or tiles.dtype != np.uint8:
        raise ValueError("tiles must be uint8 with shape (576,20,20,3)")
    tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    model.eval()
    outputs = model(tensor)
    if outputs["q_right"].ndim == 2:
        right_tensor = outputs["q_right"] @ outputs["k_left"].T
        down_tensor = outputs["q_down"] @ outputs["k_up"].T
    else:
        right_tensor = torch.einsum(
            "ntd,mtd->nm", outputs["q_right"], outputs["k_left"]
        ) / outputs["q_right"].shape[1]
        down_tensor = torch.einsum(
            "ntd,mtd->nm", outputs["q_down"], outputs["k_up"]
        ) / outputs["q_down"].shape[1]
    right = -right_tensor.float().cpu().numpy()
    down = -down_tensor.float().cpu().numpy()
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return CompatibilityMatrices(name, right, down), outputs["outside_logits"].cpu().numpy()


@torch.inference_mode()
def global_matcher_compatibility(
    matcher: GlobalSuccessorMatcher,
    encoder: SideEmbeddingNet,
    tiles: np.ndarray,
    *,
    device: torch.device | str,
    name: str = "denoised_g0_global_matcher",
) -> CompatibilityMatrices:
    tiles = np.asarray(tiles)
    if tiles.shape != (TILE_COUNT, TILE, TILE, 3) or tiles.dtype != np.uint8:
        raise ValueError("tiles must be uint8 with shape (576,20,20,3)")
    if not isinstance(encoder, SideEmbeddingNet):
        raise TypeError("global matcher requires pooled SideEmbeddingNet embeddings")
    tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(
        device=device, dtype=torch.float32
    )
    encoder.eval()
    matcher.eval()
    assignment = matcher(encoder(tensor))["log_assignment"]
    right = -assignment[0, :TILE_COUNT, :TILE_COUNT].float().cpu().numpy()
    down = -assignment[1, :TILE_COUNT, :TILE_COUNT].float().cpu().numpy()
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return CompatibilityMatrices(name, right, down)


def save_embedding_checkpoint(
    path: str | Path,
    model: SideEmbeddingNet | SideSequenceEmbeddingNet,
    *,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "puzzle_side_embedding_l1",
            "model_type": (
                "sequence" if isinstance(model, SideSequenceEmbeddingNet) else "pooled"
            ),
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_embedding_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[SideEmbeddingNet | SideSequenceEmbeddingNet, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("kind") != "puzzle_side_embedding_l1":
        raise ValueError("unsupported side-embedding checkpoint")
    model_type = payload.get("model_type", "pooled")
    if model_type == "pooled":
        model = SideEmbeddingNet(**payload["model_config"])
    elif model_type == "sequence":
        model = SideSequenceEmbeddingNet(**payload["model_config"])
    else:
        raise ValueError(f"unsupported embedding model type: {model_type}")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata", {}))
