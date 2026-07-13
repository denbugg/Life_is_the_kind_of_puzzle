"""TileNAF-latent directional embeddings and safe top-k residual fusion.

This branch deliberately keeps the established C1+HBT compatibility as the
global anchor.  A frozen TileNAF restorer exposes its final 20x20 decoder map;
the trainable model learns four side embeddings from that representation plus
raw/restored RGB and gradients.  At inference the learned score may only add a
bounded residual to a frozen candidate union.  ``alpha=0`` is therefore an
exact identity operation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .compatibility import CompatibilityMatrices
from .geometry import TILE, TILE_COUNT


_SIDES = ("left", "right", "up", "down")


class _PixelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LatentSideEmbeddingNet(nn.Module):
    """Encode TileNAF decoder features into directional side embeddings.

    The tangent dimension stays at all 20 pixels through the Transformer.  It
    is flattened only at the final projection, so the model cannot erase row
    alignment with an early global average as the old seam CNN did.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 48,
        model_dim: int = 128,
        embedding_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        feedforward_dim: int = 384,
        side_band: int = 4,
        dropout: float = 0.05,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if min(latent_channels, model_dim, embedding_dim, layers, heads) <= 0:
            raise ValueError("model dimensions must be positive")
        if model_dim % heads:
            raise ValueError("heads must divide model_dim")
        latent_dim = model_dim // 2
        visual_dim = model_dim - latent_dim
        if latent_dim % 8 or visual_dim % 8:
            raise ValueError("both model_dim halves must be divisible by 8")
        if not 1 <= side_band <= TILE:
            raise ValueError("side_band must be in [1,20]")
        if feedforward_dim < model_dim:
            raise ValueError("feedforward_dim must be at least model_dim")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.latent_channels = int(latent_channels)
        self.model_dim = int(model_dim)
        self.embedding_dim = int(embedding_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.feedforward_dim = int(feedforward_dim)
        self.side_band = int(side_band)
        self.dropout = float(dropout)
        self.temperature = float(temperature)

        # raw RGB, restored RGB, restored-raw, restored dx/dy/magnitude
        visual_channels = 12
        self.latent_stem = nn.Sequential(
            _PixelLayerNorm(latent_channels),
            nn.Conv2d(latent_channels, latent_dim, 1),
            nn.GELU(),
        )
        self.visual_stem = nn.Sequential(
            nn.Conv2d(visual_channels, visual_dim, 3, padding=1),
            nn.GroupNorm(8, visual_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, 1),
            nn.GroupNorm(8, model_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.line_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.position_embedding = nn.Parameter(torch.zeros(TILE, model_dim))
        self.side_embedding = nn.Parameter(torch.zeros(4, model_dim))
        flattened_dim = TILE * model_dim
        self.query_projection = nn.Sequential(
            nn.LayerNorm(flattened_dim),
            nn.Linear(flattened_dim, embedding_dim),
        )
        self.key_projection = nn.Sequential(
            nn.LayerNorm(flattened_dim),
            nn.Linear(flattened_dim, embedding_dim),
        )
        self.outside_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 1),
        )
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]]
            )
            / 8.0,
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]]
            )
            / 8.0,
            persistent=False,
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.side_embedding, std=0.02)

    @staticmethod
    def _unit_range(values: torch.Tensor) -> torch.Tensor:
        original_dtype = values.dtype
        values = values.float()
        if not original_dtype.is_floating_point:
            values = values / 255.0
        return values.clamp(0.0, 1.0)

    def _visual_features(
        self, raw_tiles: torch.Tensor, restored_tiles: torch.Tensor
    ) -> torch.Tensor:
        raw = self._unit_range(raw_tiles)
        restored = self._unit_range(restored_tiles)
        luma = (
            0.299 * restored[:, 0:1]
            + 0.587 * restored[:, 1:2]
            + 0.114 * restored[:, 2:3]
        )
        padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")
        gradient_x = F.conv2d(padded, self.sobel_x.to(dtype=restored.dtype))
        gradient_y = F.conv2d(padded, self.sobel_y.to(dtype=restored.dtype))
        magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-8)
        return torch.cat(
            [raw, restored, restored - raw, gradient_x, gradient_y, magnitude], dim=1
        )

    def _side_line(self, features: torch.Tensor, side: str) -> torch.Tensor:
        band = self.side_band
        if side == "left":
            return features[:, :, :, :band].mean(dim=3).transpose(1, 2)
        if side == "right":
            return features[:, :, :, -band:].mean(dim=3).transpose(1, 2)
        if side == "up":
            return features[:, :, :band, :].mean(dim=2).transpose(1, 2)
        if side == "down":
            return features[:, :, -band:, :].mean(dim=2).transpose(1, 2)
        raise ValueError(f"unknown side: {side}")

    def _validate_inputs(
        self,
        raw_tiles: torch.Tensor,
        restored_tiles: torch.Tensor,
        latent_features: torch.Tensor,
    ) -> None:
        expected_rgb = (3, TILE, TILE)
        if raw_tiles.ndim != 4 or tuple(raw_tiles.shape[1:]) != expected_rgb:
            raise ValueError("raw_tiles must have shape (N,3,20,20)")
        if restored_tiles.shape != raw_tiles.shape:
            raise ValueError("restored_tiles must match raw_tiles")
        expected_latent = (len(raw_tiles), self.latent_channels, TILE, TILE)
        if tuple(latent_features.shape) != expected_latent:
            raise ValueError(f"latent_features must have shape {expected_latent}")
        if len({raw_tiles.device, restored_tiles.device, latent_features.device}) != 1:
            raise ValueError("all inputs must share one device")

    def forward(
        self,
        raw_tiles: torch.Tensor,
        restored_tiles: torch.Tensor,
        latent_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(raw_tiles, restored_tiles, latent_features)
        latent = self.latent_stem(latent_features.float())
        visual = self.visual_stem(self._visual_features(raw_tiles, restored_tiles))
        features = self.fusion(torch.cat([latent, visual], dim=1))

        lines = torch.stack(
            [self._side_line(features, side) for side in _SIDES], dim=1
        )
        count = len(features)
        lines = lines.reshape(count * 4, TILE, self.model_dim)
        side_ids = torch.arange(4, device=features.device).repeat(count)
        lines = (
            lines
            + self.position_embedding[None, :, :]
            + self.side_embedding[side_ids, None, :]
        )
        encoded = self.line_encoder(lines).reshape(count, 4, TILE, self.model_dim)
        flattened = encoded.flatten(2)
        pooled = encoded.mean(dim=2)
        raw_q_right = self.query_projection(flattened[:, 1])
        raw_k_left = self.key_projection(flattened[:, 0])
        raw_q_down = self.query_projection(flattened[:, 3])
        raw_k_up = self.key_projection(flattened[:, 2])
        outside = self.outside_head(pooled).squeeze(-1)
        return {
            "q_right": F.normalize(raw_q_right, dim=1),
            "k_left": F.normalize(raw_k_left, dim=1),
            "q_down": F.normalize(raw_q_down, dim=1),
            "k_up": F.normalize(raw_k_up, dim=1),
            "raw_q_right": raw_q_right,
            "raw_k_left": raw_k_left,
            "raw_q_down": raw_q_down,
            "raw_k_up": raw_k_up,
            # Match the physical side order used by DirectionLabels.
            "outside_logits": outside,
        }

    def config(self) -> dict[str, Any]:
        return {
            "latent_channels": self.latent_channels,
            "model_dim": self.model_dim,
            "embedding_dim": self.embedding_dim,
            "layers": self.layers,
            "heads": self.heads,
            "feedforward_dim": self.feedforward_dim,
            "side_band": self.side_band,
            "dropout": self.dropout,
            "temperature": self.temperature,
        }


def compatibility_from_outputs(
    outputs: dict[str, torch.Tensor], *, name: str = "tilenaf_latent_edge"
) -> CompatibilityMatrices:
    """Convert normalized embedding outputs into lower-is-better costs."""

    required = ("q_right", "k_left", "q_down", "k_up")
    if any(key not in outputs for key in required):
        raise ValueError("embedding outputs are incomplete")
    right = -(outputs["q_right"] @ outputs["k_left"].T).detach().float().cpu().numpy()
    down = -(outputs["q_down"] @ outputs["k_up"].T).detach().float().cpu().numpy()
    if right.shape != (TILE_COUNT, TILE_COUNT) or down.shape != right.shape:
        raise ValueError("compatibility requires exactly 576 tile embeddings")
    right = right.astype(np.float32, copy=False)
    down = down.astype(np.float32, copy=False)
    if not np.isfinite(right).all() or not np.isfinite(down).all():
        raise ValueError("learned compatibility contains NaN or Inf")
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return CompatibilityMatrices(name, right, down)


def blend_topk_rank_residual(
    base: CompatibilityMatrices,
    learned: CompatibilityMatrices,
    candidates: tuple[np.ndarray, np.ndarray],
    *,
    alpha: float,
    name: str = "frozen_w4_plus_latent_topk",
) -> CompatibilityMatrices:
    """Add a bounded learned rank residual only inside a frozen candidate set.

    The best learned candidate receives ``-alpha/2`` and the worst receives
    ``+alpha/2``.  Non-candidates are bitwise copied from ``base``.  In
    particular, ``alpha=0`` returns an exact numerical copy of the baseline.
    """

    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    results: list[np.ndarray] = []
    for direction_name, direction_candidates in zip(
        ("right", "down"), candidates, strict=True
    ):
        candidate_array = np.asarray(direction_candidates)
        if (
            candidate_array.ndim != 2
            or candidate_array.shape[0] != TILE_COUNT
            or candidate_array.shape[1] < 2
        ):
            raise ValueError("candidate arrays must have shape (576,K), K>=2")
        if candidate_array.dtype.kind not in "iu":
            raise TypeError("candidate arrays must contain integer indices")
        if np.any((candidate_array < 0) | (candidate_array >= TILE_COUNT)):
            raise ValueError("candidate index is outside [0,576)")
        if any(len(np.unique(row)) != len(row) for row in candidate_array):
            raise ValueError("candidate rows must not contain duplicates")

        base_matrix = np.asarray(getattr(base, direction_name), dtype=np.float32)
        learned_matrix = np.asarray(getattr(learned, direction_name), dtype=np.float32)
        off_diagonal = ~np.eye(TILE_COUNT, dtype=bool)
        if not np.isfinite(base_matrix[off_diagonal]).all():
            raise ValueError("base compatibility contains non-finite off-diagonal values")
        if not np.isfinite(learned_matrix[off_diagonal]).all():
            raise ValueError("learned compatibility contains non-finite off-diagonal values")
        result = base_matrix.copy()
        if alpha:
            cap = candidate_array.shape[1]
            rows = np.arange(TILE_COUNT)[:, None]
            values = learned_matrix[rows, candidate_array]
            order = np.argsort(values, axis=1, kind="stable")
            ranks = np.empty_like(order, dtype=np.int32)
            np.put_along_axis(
                ranks,
                order,
                np.broadcast_to(np.arange(cap, dtype=np.int32), order.shape),
                axis=1,
            )
            residual = alpha * (ranks.astype(np.float32) / float(cap - 1) - 0.5)
            result[rows, candidate_array] += residual
        np.fill_diagonal(result, np.inf)
        results.append(result)
    return CompatibilityMatrices(name, results[0], results[1])


def save_latent_edge_checkpoint(
    path: str | Path,
    model: LatentSideEmbeddingNet,
    *,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "puzzle_tilenaf_latent_side_embedding",
        "safe_for_submission": False,
        "model_config": model.config(),
        "model_state": model.state_dict(),
        "metadata": {**metadata, "safe_for_submission": False},
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_latent_edge_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[LatentSideEmbeddingNet, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("kind") != "puzzle_tilenaf_latent_side_embedding"
    ):
        raise ValueError("unsupported TileNAF-latent edge checkpoint")
    model = LatentSideEmbeddingNet(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata", {}))


__all__ = [
    "LatentSideEmbeddingNet",
    "blend_topk_rank_residual",
    "compatibility_from_outputs",
    "load_latent_edge_checkpoint",
    "save_latent_edge_checkpoint",
]
