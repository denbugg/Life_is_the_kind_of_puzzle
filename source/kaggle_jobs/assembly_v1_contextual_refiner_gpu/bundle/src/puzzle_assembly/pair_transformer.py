"""Sparse full-tile pair transformer for directional puzzle compatibility.

The established HBT score is deliberately retained as a cheap proposal model.
This module only applies an expensive joint vision transformer to a bounded
candidate graph (normally HBT top-k plus the edges of the current layout).  A
complete finite compatibility matrix is returned, so existing component,
soft-cycle, annealing, and QAP solvers can consume the result unchanged.

The network sees both the raw and restored view of each complete 20x20 tile.
It also receives explicit, canonically oriented tokens from the two sides that
would touch for a proposed right/down edge.  Therefore it is materially more
expressive than the old stitched-seam CNN while remaining practical on 2xT4.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .compatibility import CompatibilityMatrices, rank_normalize
from .geometry import GRID, TILE, TILE_COUNT, validate_permutation


RIGHT = 0
DOWN = 1


@dataclass(frozen=True)
class PairCandidates:
    """Flat sparse candidate graph.

    ``first`` is the query tile, ``second`` is its proposed successor, and
    ``direction`` is 0 (right) or 1 (down).  Duplicate triples are forbidden so
    batched scores can be scattered into a matrix without ambiguous writes.
    """

    first: np.ndarray
    second: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        first = np.asarray(self.first, dtype=np.int32)
        second = np.asarray(self.second, dtype=np.int32)
        direction = np.asarray(self.direction, dtype=np.int8)
        if first.ndim != 1 or first.shape != second.shape or first.shape != direction.shape:
            raise ValueError("candidate arrays must be equally sized 1D arrays")
        if np.any((first < 0) | (first >= TILE_COUNT)):
            raise ValueError("first tile index is outside the 576-tile grid")
        if np.any((second < 0) | (second >= TILE_COUNT)):
            raise ValueError("second tile index is outside the 576-tile grid")
        if np.any((direction < RIGHT) | (direction > DOWN)):
            raise ValueError("direction must be 0 (right) or 1 (down)")
        if np.any(first == second):
            raise ValueError("self-pairs are not valid candidates")
        triples = np.stack([direction.astype(np.int32), first, second], axis=1)
        if len(np.unique(triples, axis=0)) != len(triples):
            raise ValueError("candidate triples must be unique")
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)
        object.__setattr__(self, "direction", direction)

    def __len__(self) -> int:
        return int(len(self.first))

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self),
            "right": int(np.sum(self.direction == RIGHT)),
            "down": int(np.sum(self.direction == DOWN)),
        }


@dataclass(frozen=True)
class PairScoreResult:
    compatibility: CompatibilityMatrices
    candidates: PairCandidates
    logits: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class EncodedTileBank:
    """CNN features cached once per tile set for sparse inference scoring."""

    full_tokens: torch.Tensor
    side_tokens: torch.Tensor

    def __post_init__(self) -> None:
        if self.full_tokens.ndim != 3:
            raise ValueError("full_tokens must have shape (N,P,D)")
        if self.side_tokens.ndim != 4 or self.side_tokens.shape[:2] != (
            self.full_tokens.shape[0],
            4,
        ):
            raise ValueError("side_tokens must have shape (N,4,B,D)")
        if self.side_tokens.shape[-1] != self.full_tokens.shape[-1]:
            raise ValueError("full and side token dimensions differ")


class _ConvResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(16, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.gelu(values + self.block(values))


class PairTransformerScorer(nn.Module):
    """Jointly score two full raw+denoised tiles and their proposed seam.

    Defaults are a roughly 27M-parameter model (depending on PyTorch version):
    large enough to be a serious nonlinear pair model, but with only 71 tokens
    per pair it remains a bounded 2xT4 experiment.  Gradient checkpointing is
    used per transformer block during training.
    """

    def __init__(
        self,
        *,
        model_dim: int = 512,
        layers: int = 8,
        heads: int = 8,
        feedforward_dim: int = 2048,
        cnn_channels: int = 128,
        patch_grid: int = 5,
        side_band: int = 6,
        band_tokens: int = 10,
        dropout: float = 0.10,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or heads <= 0 or model_dim % heads:
            raise ValueError("model_dim must be positive and divisible by heads")
        if layers <= 0 or feedforward_dim <= 0 or cnn_channels <= 0:
            raise ValueError("layers and feature dimensions must be positive")
        if cnn_channels % 8:
            raise ValueError("cnn_channels must be divisible by 8")
        if not 2 <= side_band <= 10:
            raise ValueError("side_band must be in [2, 10]")
        if not 2 <= patch_grid <= 10 or not 2 <= band_tokens <= TILE:
            raise ValueError("patch_grid/band_tokens are outside supported bounds")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.model_dim = int(model_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.feedforward_dim = int(feedforward_dim)
        self.cnn_channels = int(cnn_channels)
        self.patch_grid = int(patch_grid)
        self.side_band = int(side_band)
        self.band_tokens = int(band_tokens)
        self.dropout = float(dropout)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # Six physical channels (raw RGB + denoised RGB) are augmented with a
        # per-view local normalization.  Absolute colour and normalized shape
        # evidence therefore remain simultaneously available.
        input_channels = 12
        stem_channels = max(64, cnn_channels // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
            _ConvResidual(stem_channels),
            nn.Conv2d(stem_channels, cnn_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, cnn_channels),
            nn.GELU(),
            _ConvResidual(cnn_channels),
        )
        self.full_projection = nn.Linear(cnn_channels, model_dim)
        self.band_stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
            _ConvResidual(stem_channels),
            nn.Conv2d(stem_channels, cnn_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, cnn_channels),
            nn.GELU(),
        )
        self.band_projection = nn.Linear(cnn_channels, model_dim)

        full_tokens = patch_grid * patch_grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.full_position = nn.Parameter(torch.zeros(1, full_tokens, model_dim))
        self.band_position = nn.Parameter(torch.zeros(1, band_tokens, model_dim))
        # cls, first-full, second-full, first-band, second-band
        self.role_embedding = nn.Parameter(torch.zeros(5, model_dim))
        self.direction_embedding = nn.Parameter(torch.zeros(2, model_dim))
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * model_dim),
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        self.register_buffer("calibration_temperature", torch.tensor(1.0))
        self.register_buffer("calibration_bias", torch.tensor(0.0))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.full_position, std=0.02)
        nn.init.trunc_normal_(self.band_position, std=0.02)
        nn.init.trunc_normal_(self.role_embedding, std=0.02)
        nn.init.trunc_normal_(self.direction_embedding, std=0.02)

    @staticmethod
    def _normalize_views(values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        if values.numel() and values.detach().amax() > 1.5:
            values = values / 255.0
        values = values.clamp(0.0, 1.0)
        raw, denoised = values[:, :3], values[:, 3:]
        normalized = []
        for view in (raw, denoised):
            mean = view.mean(dim=(2, 3), keepdim=True)
            std = view.std(dim=(2, 3), keepdim=True).clamp_min(1.0 / 255.0)
            normalized.append(((view - mean) / std).clamp(-4.0, 4.0) / 4.0)
        return torch.cat([values, *normalized], dim=1)

    def _full_tokens(self, values: torch.Tensor) -> torch.Tensor:
        features = self.stem(values)
        pooled = F.adaptive_avg_pool2d(features, (self.patch_grid, self.patch_grid))
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.full_projection(tokens) + self.full_position

    def _canonical_bands(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        directions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(first)
        shape = (batch, first.shape[1], TILE, self.side_band)
        first_band = first.new_empty(shape)
        second_band = second.new_empty(shape)
        right = directions == RIGHT
        if right.any():
            first_band[right] = first[right, :, :, -self.side_band :]
            second_band[right] = second[right, :, :, : self.side_band]
        down = ~right
        if down.any():
            # Transpose so tangent order always runs along the 20-pixel axis
            # and the seam normal always runs across side_band.
            first_band[down] = first[down, :, -self.side_band :, :].transpose(2, 3)
            second_band[down] = second[down, :, : self.side_band, :].transpose(2, 3)
        return first_band, second_band

    def _band_tokens(self, values: torch.Tensor) -> torch.Tensor:
        features = self.band_stem(values)
        pooled = F.adaptive_avg_pool2d(features, (self.band_tokens, 1))
        tokens = pooled.squeeze(3).transpose(1, 2)
        return self.band_projection(tokens) + self.band_position

    def encode_tile_bank(self, values: torch.Tensor) -> EncodedTileBank:
        """Encode full-tile and four physical side views exactly once per tile."""

        if values.ndim != 4 or values.shape[1:] != (6, TILE, TILE):
            raise ValueError("tile bank must have shape (N,6,20,20)")
        normalized = self._normalize_views(values)
        full_tokens = self._full_tokens(normalized)
        band = self.side_band
        # Side order is physical left, right, up, down.  Horizontal seams use
        # (right,left); vertical seams use (down,up).  Up/down are transposed so
        # the first spatial axis is always the 20-pixel seam tangent.
        side_views = (
            normalized[:, :, :, :band],
            normalized[:, :, :, -band:],
            normalized[:, :, :band, :].transpose(2, 3),
            normalized[:, :, -band:, :].transpose(2, 3),
        )
        side_tokens = torch.stack(
            [self._band_tokens(side) for side in side_views], dim=1
        )
        return EncodedTileBank(full_tokens=full_tokens, side_tokens=side_tokens)

    def forward_from_encoded(
        self,
        first_bank: EncodedTileBank,
        second_bank: EncodedTileBank,
        first_indices: torch.Tensor,
        second_indices: torch.Tensor,
        directions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score indexed pairs without repeating either CNN tile encoder."""

        if first_indices.ndim != 1 or second_indices.shape != first_indices.shape:
            raise ValueError("pair indices must be equally sized vectors")
        if directions.shape != first_indices.shape:
            raise ValueError("directions must match pair indices")
        device = first_bank.full_tokens.device
        first_indices = first_indices.to(device=device, dtype=torch.long)
        second_indices = second_indices.to(device=device, dtype=torch.long)
        directions = directions.to(device=device, dtype=torch.long)
        if second_bank.full_tokens.device != device:
            raise ValueError("encoded tile banks must share a device")
        if torch.any((directions < RIGHT) | (directions > DOWN)):
            raise ValueError("directions must contain only 0/1")
        if (
            torch.any((first_indices < 0) | (first_indices >= len(first_bank.full_tokens)))
            or torch.any(
                (second_indices < 0) | (second_indices >= len(second_bank.full_tokens))
            )
        ):
            raise ValueError("pair index is outside its encoded tile bank")

        outgoing_side = directions.new_tensor([1, 3])[directions]
        incoming_side = directions.new_tensor([0, 2])[directions]
        first_full = first_bank.full_tokens[first_indices] + self.role_embedding[1]
        second_full = second_bank.full_tokens[second_indices] + self.role_embedding[2]
        first_side = (
            first_bank.side_tokens[first_indices, outgoing_side] + self.role_embedding[3]
        )
        second_side = (
            second_bank.side_tokens[second_indices, incoming_side] + self.role_embedding[4]
        )
        return self._transform_pair_tokens(
            first_full,
            second_full,
            first_side,
            second_side,
            directions,
        )

    def _transform_pair_tokens(
        self,
        first_full: torch.Tensor,
        second_full: torch.Tensor,
        first_side: torch.Tensor,
        second_side: torch.Tensor,
        directions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cls = self.cls_token.expand(len(directions), -1, -1) + self.role_embedding[0]
        tokens = torch.cat([cls, first_full, second_full, first_side, second_side], dim=1)
        tokens = tokens + self.direction_embedding[directions].unsqueeze(1)
        tokens = self.input_dropout(tokens)
        for block in self.blocks:
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.final_norm(tokens)
        cls_feature = tokens[:, 0]
        pooled_feature = tokens[:, 1:].mean(dim=1)
        pair_embedding = torch.cat([cls_feature, pooled_feature], dim=1)
        logits = self.head(pair_embedding).squeeze(1)
        calibrated_logits = logits / self.calibration_temperature.clamp_min(0.05)
        calibrated_logits = calibrated_logits + self.calibration_bias
        probability = torch.sigmoid(calibrated_logits)
        confidence = (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0)
        return {
            "logits": logits,
            "calibrated_logits": calibrated_logits,
            "probability": probability,
            "confidence": confidence,
            "pair_embedding": pair_embedding,
        }

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        directions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if first.ndim != 4 or first.shape[1:] != (6, TILE, TILE):
            raise ValueError("first must have shape (B,6,20,20)")
        if second.shape != first.shape:
            raise ValueError("second must have the same shape as first")
        if directions.ndim != 1 or len(directions) != len(first):
            raise ValueError("directions must have shape (B,)")
        directions = directions.to(device=first.device, dtype=torch.long)
        if torch.any((directions < RIGHT) | (directions > DOWN)):
            raise ValueError("directions must contain only 0/1")

        first_features = self._normalize_views(first)
        second_features = self._normalize_views(second)
        first_band, second_band = self._canonical_bands(
            first_features, second_features, directions
        )
        return self._transform_pair_tokens(
            self._full_tokens(first_features) + self.role_embedding[1],
            self._full_tokens(second_features) + self.role_embedding[2],
            self._band_tokens(first_band) + self.role_embedding[3],
            self._band_tokens(second_band) + self.role_embedding[4],
            directions,
        )

    def set_calibration(self, temperature: float, bias: float) -> None:
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("calibration temperature must be finite and positive")
        if not np.isfinite(bias):
            raise ValueError("calibration bias must be finite")
        self.calibration_temperature.fill_(float(temperature))
        self.calibration_bias.fill_(float(bias))

    def config(self) -> dict[str, Any]:
        return {
            "model_dim": self.model_dim,
            "layers": self.layers,
            "heads": self.heads,
            "feedforward_dim": self.feedforward_dim,
            "cnn_channels": self.cnn_channels,
            "patch_grid": self.patch_grid,
            "side_band": self.side_band,
            "band_tokens": self.band_tokens,
            "dropout": self.dropout,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


def _validate_tile_views(raw_tiles: np.ndarray, denoised_tiles: np.ndarray) -> None:
    expected = (TILE_COUNT, TILE, TILE, 3)
    if np.asarray(raw_tiles).shape != expected or np.asarray(raw_tiles).dtype != np.uint8:
        raise ValueError(f"raw_tiles must be uint8 with shape {expected}")
    if (
        np.asarray(denoised_tiles).shape != expected
        or np.asarray(denoised_tiles).dtype != np.uint8
    ):
        raise ValueError(f"denoised_tiles must be uint8 with shape {expected}")


def multistage_candidates(
    coarse: CompatibilityMatrices,
    *,
    top_k: int = 48,
    reverse_top_k: int = 8,
    layouts: Iterable[np.ndarray] = (),
) -> PairCandidates:
    """Build HBT proposals plus current-layout edges for iterative re-scoring.

    ``reverse_top_k`` adds candidates for which a query is among the best
    incoming predecessors.  This recovers useful asymmetric confusers at a
    small cost.  Passing first-pass or previous-pass layouts guarantees that
    every currently realized edge is explicitly validated by the transformer.
    """

    if not 1 <= top_k < TILE_COUNT or not 0 <= reverse_top_k < TILE_COUNT:
        raise ValueError("top-k limits are outside the 576-tile range")
    resolved_layouts = [
        validate_permutation(layout, name=f"layout_{index}")
        for index, layout in enumerate(layouts)
    ]
    triples: set[tuple[int, int, int]] = set()
    for direction, matrix in ((RIGHT, coarse.right), (DOWN, coarse.down)):
        values = np.asarray(matrix)
        row_order = np.argsort(values, axis=1, kind="stable")[:, : top_k + 1]
        for first in range(TILE_COUNT):
            added = 0
            for second in row_order[first].tolist():
                if second == first:
                    continue
                triples.add((direction, first, int(second)))
                added += 1
                if added == top_k:
                    break
        if reverse_top_k:
            column_order = np.argsort(values, axis=0, kind="stable")[: reverse_top_k + 1]
            for second in range(TILE_COUNT):
                added = 0
                for first in column_order[:, second].tolist():
                    if first == second:
                        continue
                    triples.add((direction, int(first), second))
                    added += 1
                    if added == reverse_top_k:
                        break

    for layout in resolved_layouts:
        grid = layout.reshape(GRID, GRID)
        triples.update(
            (RIGHT, int(first), int(second))
            for first, second in zip(
                grid[:, :-1].ravel(), grid[:, 1:].ravel(), strict=True
            )
        )
        triples.update(
            (DOWN, int(first), int(second))
            for first, second in zip(
                grid[:-1, :].ravel(), grid[1:, :].ravel(), strict=True
            )
        )
    ordered = sorted(triples)
    return PairCandidates(
        first=np.asarray([value[1] for value in ordered], dtype=np.int32),
        second=np.asarray([value[2] for value in ordered], dtype=np.int32),
        direction=np.asarray([value[0] for value in ordered], dtype=np.int8),
    )


@torch.inference_mode()
def score_pairs(
    model: PairTransformerScorer,
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    candidates: PairCandidates,
    *,
    device: torch.device | str,
    batch_size: int = 512,
    telemetry: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw logits, calibrated probabilities, and confidence."""

    _validate_tile_views(raw_tiles, denoised_tiles)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    views = np.concatenate([raw_tiles, denoised_tiles], axis=3)
    bank = torch.from_numpy(np.ascontiguousarray(views.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    )
    started = time.perf_counter()
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    model.eval()
    amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        encode_started = time.perf_counter()
        encoded = model.encode_tile_bank(bank)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encode_seconds = time.perf_counter() - encode_started
        pair_started = time.perf_counter()
        for start in range(0, len(candidates), batch_size):
            stop = min(start + batch_size, len(candidates))
            first = torch.as_tensor(candidates.first[start:stop], device=device)
            second = torch.as_tensor(candidates.second[start:stop], device=device)
            directions = torch.as_tensor(candidates.direction[start:stop], device=device)
            output = model.forward_from_encoded(
                encoded,
                encoded,
                first,
                second,
                directions,
            )
            logits.append(output["logits"].float().cpu().numpy())
            probabilities.append(output["probability"].float().cpu().numpy())
            confidence.append(output["confidence"].float().cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pair_seconds = time.perf_counter() - pair_started
    total_seconds = time.perf_counter() - started
    if telemetry is not None:
        telemetry.update(
            {
                "cached_tile_count": int(len(bank)),
                "tile_encoder_passes": 1,
                "candidate_pairs": int(len(candidates)),
                "tile_encode_seconds": float(encode_seconds),
                "pair_transformer_seconds": float(pair_seconds),
                "total_seconds": float(total_seconds),
                "pairs_per_second": float(len(candidates) / max(pair_seconds, 1.0e-9)),
                "end_to_end_pairs_per_second": float(
                    len(candidates) / max(total_seconds, 1.0e-9)
                ),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0,
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0,
            }
        )
    if not logits:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.copy(), empty.copy()
    return (
        np.concatenate(logits).astype(np.float32),
        np.concatenate(probabilities).astype(np.float32),
        np.concatenate(confidence).astype(np.float32),
    )


def _rerank_direction(
    coarse: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    probability: np.ndarray,
    *,
    blend: float,
) -> tuple[np.ndarray, dict[str, float]]:
    result = rank_normalize(coarse)
    row_margins: list[float] = []
    effective_blends: list[float] = []
    for query in range(TILE_COUNT):
        selected = np.flatnonzero(first == query)
        if not len(selected):
            continue
        target = second[selected]
        values = probability[selected]
        order = np.argsort(-values, kind="stable")
        model_rank = np.empty(len(order), dtype=np.float32)
        model_rank[order] = np.arange(len(order), dtype=np.float32)
        model_rank /= float(max(TILE_COUNT - 2, 1))
        if len(order) >= 2:
            margin = float(values[order[0]] - values[order[1]])
        else:
            margin = float(abs(values[order[0]] - 0.5) * 2.0)
        # A flat model distribution should not erase a useful HBT ordering.
        row_confidence = float(np.clip(margin / 0.20, 0.10, 1.0))
        effective = blend * row_confidence
        result[query, target] = (
            (1.0 - effective) * result[query, target] + effective * model_rank
        )
        row_margins.append(margin)
        effective_blends.append(effective)
    np.fill_diagonal(result, np.inf)
    return result.astype(np.float32), {
        "mean_top1_top2_probability_margin": float(np.mean(row_margins)) if row_margins else 0.0,
        "mean_effective_blend": float(np.mean(effective_blends)) if effective_blends else 0.0,
    }


def pair_transformer_compatibility(
    model: PairTransformerScorer,
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    coarse: CompatibilityMatrices,
    *,
    device: torch.device | str,
    top_k: int = 48,
    reverse_top_k: int = 8,
    layouts: Iterable[np.ndarray] = (),
    batch_size: int = 512,
    blend: float = 0.75,
    name: str = "pair_transformer_stage2",
) -> PairScoreResult:
    """Sparse neural re-ranking with a complete coarse fallback matrix."""

    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must lie in [0, 1]")
    resolved_layouts = list(layouts)
    candidates = multistage_candidates(
        coarse, top_k=top_k, reverse_top_k=reverse_top_k, layouts=resolved_layouts
    )
    inference_telemetry: dict[str, Any] = {}
    logits, probability, confidence = score_pairs(
        model,
        raw_tiles,
        denoised_tiles,
        candidates,
        device=device,
        batch_size=batch_size,
        telemetry=inference_telemetry,
    )
    matrices: list[np.ndarray] = []
    direction_diagnostics: dict[str, dict[str, float]] = {}
    for direction, label, matrix in (
        (RIGHT, "right", coarse.right),
        (DOWN, "down", coarse.down),
    ):
        selected = candidates.direction == direction
        reranked, diagnostics = _rerank_direction(
            matrix,
            candidates.first[selected],
            candidates.second[selected],
            probability[selected],
            blend=blend,
        )
        matrices.append(reranked)
        direction_diagnostics[label] = diagnostics
    compatibility = CompatibilityMatrices(name, matrices[0], matrices[1])
    diagnostics: dict[str, Any] = {
        "candidate_counts": candidates.counts(),
        "top_k": int(top_k),
        "reverse_top_k": int(reverse_top_k),
        "layout_count": len(resolved_layouts),
        "blend": float(blend),
        "mean_probability": float(probability.mean()) if len(probability) else 0.0,
        "mean_confidence": float(confidence.mean()) if len(confidence) else 0.0,
        "directions": direction_diagnostics,
        "inference_telemetry": inference_telemetry,
    }
    return PairScoreResult(
        compatibility=compatibility,
        candidates=candidates,
        logits=logits,
        probabilities=probability,
        confidence=confidence,
        diagnostics=diagnostics,
    )


def binary_calibration_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if logits.ndim != 1 or logits.shape != labels.shape or not len(logits):
        raise ValueError("logits/labels must be non-empty equally sized vectors")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    eps = 1e-8
    nll = -np.mean(
        labels * np.log(probabilities + eps)
        + (1.0 - labels) * np.log(1.0 - probabilities + eps)
    )
    brier = np.mean((probabilities - labels) ** 2)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (probabilities >= lower) & (probabilities < lower + 0.1)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return {"nll": float(nll), "brier": float(brier), "ece_10": float(ece)}


def fit_binary_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_iterations: int = 100,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    """Fit scalar temperature and bias on a disjoint calibration panel."""

    logits_array = np.asarray(logits, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.float32)
    if logits_array.ndim != 1 or logits_array.shape != labels_array.shape:
        raise ValueError("logits/labels must be equally sized 1D arrays")
    if len(logits_array) < 2 or len(np.unique(labels_array)) != 2:
        raise ValueError("calibration requires both classes")
    values = torch.from_numpy(logits_array).double()
    truth = torch.from_numpy(labels_array).double()
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature, bias], max_iter=max_iterations, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        calibrated = values / temperature + bias
        loss = F.binary_cross_entropy_with_logits(calibrated, truth)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
    resolved_bias = float(bias.detach())
    calibrated = logits_array / temperature + resolved_bias
    metrics = {
        "before": binary_calibration_metrics(logits_array, labels_array),
        "after": binary_calibration_metrics(calibrated, labels_array),
    }
    return temperature, resolved_bias, metrics


def save_pair_transformer_checkpoint(
    path: str | Path,
    model: PairTransformerScorer,
    *,
    metadata: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
    scaler_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
    preserve_previous: bool = False,
) -> None:
    """Atomically save a non-promoted checkpoint with an optional fallback."""

    payload_metadata = dict(metadata)
    payload_metadata["safe_for_submission"] = False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "puzzle_full_tile_pair_transformer",
        "safe_for_submission": False,
        "model_config": model.config(),
        "model_state": model.state_dict(),
        "calibration": {
            "temperature": float(model.calibration_temperature),
            "bias": float(model.calibration_bias),
        },
        "metadata": payload_metadata,
    }
    for key, value in (
        ("optimizer_state", optimizer_state),
        ("scaler_state", scaler_state),
        ("scheduler_state", scheduler_state),
        ("training_state", training_state),
    ):
        if value is not None:
            payload[key] = value

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    previous = path.with_name(f"{path.name}.previous")
    previous_temporary = previous.with_name(
        f".{previous.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(payload, temporary)
        _fsync_file(temporary)
        if preserve_previous and path.exists():
            try:
                _load_validated_pair_checkpoint_file(
                    path, require_training_state=True
                )
            except Exception:
                # A corrupt latest may have been recovered through .previous.
                # Never rotate it over the only known-good checkpoint.
                pass
            else:
                shutil.copy2(path, previous_temporary)
                _fsync_file(previous_temporary)
                os.replace(previous_temporary, previous)
                _fsync_directory(path.parent)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        previous_temporary.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_validated_pair_checkpoint_file(
    path: Path, *, require_training_state: bool = False
) -> dict[str, Any]:
    """Load exactly one checkpoint path without consulting a fallback."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("pair-transformer checkpoint must contain a dictionary")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "puzzle_full_tile_pair_transformer"
    ):
        raise ValueError("unsupported pair-transformer checkpoint")
    if payload.get("safe_for_submission") is not False:
        raise ValueError("pair-transformer checkpoint must be explicitly unsafe")
    if not isinstance(payload.get("model_config"), dict) or not isinstance(
        payload.get("model_state"), dict
    ):
        raise ValueError("pair-transformer checkpoint is incomplete")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("safe_for_submission") is not False:
        raise ValueError("pair-transformer checkpoint metadata is not fail-closed")
    try:
        validation_model = PairTransformerScorer(**payload["model_config"])
        validation_model.load_state_dict(payload["model_state"], strict=True)
    except Exception as error:
        raise ValueError("pair-transformer model payload is not strictly loadable") from error
    if require_training_state:
        required = {
            "optimizer_state",
            "scaler_state",
            "scheduler_state",
            "training_state",
        }
        missing = required - set(payload)
        if missing or any(not isinstance(payload.get(key), dict) for key in required):
            raise ValueError(
                "pair-transformer resume bundle is incomplete: "
                f"missing={sorted(missing)}"
            )
    return payload


def load_pair_transformer_checkpoint_payload(
    path: str | Path, *, require_training_state: bool = False
) -> dict[str, Any]:
    requested = Path(path)
    candidates = (requested, requested.with_name(f"{requested.name}.previous"))
    failures: list[Exception] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = _load_validated_pair_checkpoint_file(
                candidate,
                require_training_state=require_training_state,
            )
            payload["loaded_checkpoint_path"] = str(candidate)
            payload["used_previous_fallback"] = candidate != requested
            return payload
        except Exception as error:
            failures.append(error)
    if failures:
        raise ValueError(
            f"failed to load checkpoint {requested} or its previous fallback"
        ) from failures[-1]
    raise FileNotFoundError(requested)


def load_pair_transformer_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[PairTransformerScorer, dict[str, Any]]:
    payload = load_pair_transformer_checkpoint_payload(path)
    model = PairTransformerScorer(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("safe_for_submission") is not False:
        raise ValueError("research checkpoint is missing safe_for_submission=false")
    return model, metadata


__all__ = [
    "DOWN",
    "EncodedTileBank",
    "RIGHT",
    "PairCandidates",
    "PairScoreResult",
    "PairTransformerScorer",
    "binary_calibration_metrics",
    "fit_binary_temperature",
    "load_pair_transformer_checkpoint",
    "load_pair_transformer_checkpoint_payload",
    "multistage_candidates",
    "pair_transformer_compatibility",
    "save_pair_transformer_checkpoint",
    "score_pairs",
]
