"""Task-specific ViT-Sinkhorn pilot for 24x24 tile assignment.

The model consumes the *set* of 576 shuffled raw/restored tile pairs.  It has
no input-order positional embedding, so permuting the tile set only permutes
the rows of the assignment matrix.  Learned output-slot queries provide the
absolute 24x24 coordinate system.  An existing QAP solution can be supplied as
an explicitly droppable prior rather than being treated as ground truth.

This module is deliberately self-contained and grid-size configurable so its
one-to-one and equivariance contracts can be tested cheaply on small puzzles.
The production default remains the competition geometry: 576 RGB 20x20 tiles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(frozen=True)
class ViTSinkhornConfig:
    """Serializable model configuration.

    The 6-layer, d=256, 8-head defaults are the bounded 2xT4 pilot, not a claim
    that this capacity is already validated for promotion.
    """

    grid_size: int = 24
    tile_size: int = 20
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    feedforward_dim: int = 1024
    cnn_channels: int = 64
    edge_channels: int = 32
    edge_dim: int = 64
    edge_band: int = 4
    edge_bins: int = 10
    dropout: float = 0.10
    qap_prior_dropout: float = 0.35
    sinkhorn_iterations: int = 20
    sinkhorn_temperature: float = 0.10
    activation_checkpointing: bool = True

    @property
    def tile_count(self) -> int:
        return self.grid_size * self.grid_size

    def validate(self) -> None:
        integer_fields = {
            "grid_size": self.grid_size,
            "tile_size": self.tile_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "cnn_channels": self.cnn_channels,
            "edge_channels": self.edge_channels,
            "edge_dim": self.edge_dim,
            "edge_band": self.edge_band,
            "edge_bins": self.edge_bins,
            "sinkhorn_iterations": self.sinkhorn_iterations,
        }
        if any(value <= 0 for value in integer_fields.values()):
            raise ValueError(f"positive integer configuration required: {integer_fields}")
        if self.d_model % self.num_heads:
            raise ValueError("num_heads must divide d_model")
        if self.edge_band > self.tile_size:
            raise ValueError("edge_band cannot exceed tile_size")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.qap_prior_dropout <= 1.0:
            raise ValueError("qap_prior_dropout must be in [0, 1]")
        if not math.isfinite(self.sinkhorn_temperature) or self.sinkhorn_temperature <= 0:
            raise ValueError("sinkhorn_temperature must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


class _ResidualConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(values)
        values = F.silu(self.norm(self.pointwise(values)))
        return residual + values


class TileEdgeEncoder(nn.Module):
    """CNN encoder with a global tile path and four physical edge paths."""

    def __init__(self, config: ViTSinkhornConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        # Raw, restored, and their residual are all explicit inputs.
        input_channels = 9
        hidden = config.cnn_channels
        self.tile_stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(),
            _ResidualConv(hidden),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden * 2), hidden * 2),
            nn.SiLU(),
            _ResidualConv(hidden * 2),
            nn.AdaptiveAvgPool2d(1),
        )
        edge_channels = config.edge_channels
        self.edge_stem = nn.Sequential(
            nn.Conv2d(input_channels, edge_channels, 3, padding=1),
            nn.GroupNorm(_group_count(edge_channels), edge_channels),
            nn.SiLU(),
            _ResidualConv(edge_channels),
        )
        edge_feature_dim = edge_channels * config.edge_bins
        self.edge_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(edge_feature_dim),
                    nn.Linear(edge_feature_dim, config.edge_dim),
                )
                for _ in range(4)
            ]
        )
        self.tile_projection = nn.Sequential(
            nn.LayerNorm(hidden * 2 + 4 * config.edge_dim),
            nn.Linear(hidden * 2 + 4 * config.edge_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )

    def _edge_line(self, features: torch.Tensor, direction: int) -> torch.Tensor:
        band = self.config.edge_band
        if direction == TOP:
            line = features[:, :, :band, :].mean(dim=2)
        elif direction == RIGHT:
            line = features[:, :, :, -band:].mean(dim=3)
        elif direction == BOTTOM:
            line = features[:, :, -band:, :].mean(dim=2)
        elif direction == LEFT:
            line = features[:, :, :, :band].mean(dim=3)
        else:
            raise ValueError(f"unknown edge direction {direction}")
        return F.adaptive_avg_pool1d(line, self.config.edge_bins).flatten(1)

    def forward(
        self, raw_tiles: torch.Tensor, restored_tiles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            self.config.tile_count,
            3,
            self.config.tile_size,
            self.config.tile_size,
        )
        if raw_tiles.ndim != 5 or tuple(raw_tiles.shape[1:]) != expected:
            raise ValueError(
                f"raw_tiles must have shape Bx{expected}, got {tuple(raw_tiles.shape)}"
            )
        if restored_tiles.shape != raw_tiles.shape:
            raise ValueError("restored_tiles must have the same shape as raw_tiles")
        if not raw_tiles.is_floating_point() or not restored_tiles.is_floating_point():
            raise TypeError("raw/restored tile tensors must be floating point in [0,1]")
        batch, tile_count = raw_tiles.shape[:2]
        raw = raw_tiles.reshape(-1, 3, self.config.tile_size, self.config.tile_size)
        restored = restored_tiles.reshape_as(raw)
        combined = torch.cat([raw, restored, restored - raw], dim=1)
        tile_features = self.tile_stem(combined).flatten(1)
        edge_map = self.edge_stem(combined)
        edge_features = torch.stack(
            [
                projection(self._edge_line(edge_map, direction))
                for direction, projection in enumerate(self.edge_projections)
            ],
            dim=1,
        )
        tokens = self.tile_projection(
            torch.cat([tile_features, edge_features.flatten(1)], dim=1)
        )
        return (
            tokens.reshape(batch, tile_count, self.config.d_model),
            edge_features.reshape(
                batch, tile_count, 4, self.config.edge_dim
            ),
        )


@dataclass
class ViTSinkhornOutput:
    logits: torch.Tensor
    log_assignment: torch.Tensor
    edge_embeddings: torch.Tensor
    contextual_tiles: torch.Tensor
    prior_keep_mask: torch.Tensor | None


def log_sinkhorn(
    logits: torch.Tensor,
    *,
    iterations: int = 20,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Return a differentiable log doubly-stochastic square assignment.

    Normalization is performed in float32 even under autocast; this avoids the
    fp16 underflow that otherwise appears with 576 columns.
    """

    if logits.ndim != 3 or logits.shape[1] != logits.shape[2]:
        raise ValueError("logits must be a BxNxN square assignment tensor")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain non-finite values")
    values = (logits.float() / float(temperature)).clamp(-80.0, 80.0)
    for _ in range(iterations):
        values = values - torch.logsumexp(values, dim=2, keepdim=True)
        values = values - torch.logsumexp(values, dim=1, keepdim=True)
    return values


class ViTSinkhorn(nn.Module):
    """Permutation-equivariant tile set to absolute-slot assignment model."""

    def __init__(self, config: ViTSinkhornConfig | None = None) -> None:
        super().__init__()
        self.config = config or ViTSinkhornConfig()
        self.config.validate()
        self.encoder = TileEdgeEncoder(self.config)
        self.context_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.config.d_model,
                    nhead=self.config.num_heads,
                    dim_feedforward=self.config.feedforward_dim,
                    dropout=self.config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.context_norm = nn.LayerNorm(self.config.d_model)
        self.slot_query_residual = nn.Parameter(
            torch.empty(self.config.tile_count, self.config.d_model)
        )
        self.row_embedding = nn.Embedding(
            self.config.grid_size, self.config.d_model
        )
        self.column_embedding = nn.Embedding(
            self.config.grid_size, self.config.d_model
        )
        self.prior_confidence = nn.Sequential(
            nn.Linear(1, self.config.d_model),
            nn.SiLU(),
            nn.Linear(self.config.d_model, self.config.d_model),
        )
        self.prior_gain = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.qap_logit_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.assignment_logit_scale = nn.Parameter(
            torch.tensor(math.log(10.0), dtype=torch.float32)
        )
        nn.init.normal_(self.slot_query_residual, std=0.02)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.column_embedding.weight, std=0.02)

    def _slot_queries(self) -> torch.Tensor:
        grid = self.config.grid_size
        positions = torch.arange(
            self.config.tile_count, device=self.slot_query_residual.device
        )
        return (
            self.slot_query_residual
            + self.row_embedding(positions // grid)
            + self.column_embedding(positions % grid)
        )

    def _prior_features(
        self,
        qap_tile_to_position: torch.Tensor,
        qap_confidence: torch.Tensor | None,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (qap_tile_to_position.shape[0], self.config.tile_count)
        if qap_tile_to_position.ndim != 2 or tuple(qap_tile_to_position.shape) != expected:
            raise ValueError(
                f"qap_tile_to_position must have shape Bx{self.config.tile_count}"
            )
        positions = qap_tile_to_position.long()
        valid = (positions >= 0) & (positions < self.config.tile_count)
        invalid_values = (~valid) & (positions != -1)
        if invalid_values.any():
            raise ValueError("QAP positions must be -1 or valid slot indices")
        if qap_confidence is None:
            confidence = valid.to(dtype=dtype)
        else:
            if qap_confidence.shape != positions.shape:
                raise ValueError("qap_confidence must match QAP position shape")
            confidence = qap_confidence.to(dtype=dtype).clamp(0.0, 1.0)
            if not torch.isfinite(confidence).all():
                raise ValueError("qap_confidence contains non-finite values")
        keep = valid
        if self.training and self.config.qap_prior_dropout > 0:
            keep = keep & (
                torch.rand(positions.shape, device=positions.device)
                >= self.config.qap_prior_dropout
            )
        safe = positions.clamp(0, self.config.tile_count - 1)
        grid = self.config.grid_size
        prior = (
            self.row_embedding(safe // grid)
            + self.column_embedding(safe % grid)
            + self.prior_confidence(confidence.unsqueeze(2))
        )
        weight = keep.to(dtype=dtype) * confidence
        return prior * weight.unsqueeze(2), safe, weight, keep

    def forward(
        self,
        raw_tiles: torch.Tensor,
        restored_tiles: torch.Tensor,
        *,
        qap_tile_to_position: torch.Tensor | None = None,
        qap_confidence: torch.Tensor | None = None,
    ) -> ViTSinkhornOutput:
        tokens, edge_embeddings = self.encoder(raw_tiles, restored_tiles)
        prior_keep_mask: torch.Tensor | None = None
        safe_prior: torch.Tensor | None = None
        prior_weight: torch.Tensor | None = None
        if qap_tile_to_position is not None:
            prior, safe_prior, prior_weight, prior_keep_mask = self._prior_features(
                qap_tile_to_position,
                qap_confidence,
                dtype=tokens.dtype,
            )
            tokens = tokens + self.prior_gain.to(tokens.dtype) * prior
        elif qap_confidence is not None:
            raise ValueError("qap_confidence cannot be supplied without QAP positions")

        for layer in self.context_layers:
            if (
                self.config.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                tokens = checkpoint(layer, tokens, use_reentrant=False)
            else:
                tokens = layer(tokens)
        contextual = self.context_norm(tokens)
        tile_vectors = F.normalize(contextual, dim=2)
        slot_queries = F.normalize(self._slot_queries(), dim=1)
        scale = self.assignment_logit_scale.exp().clamp(1.0, 100.0)
        logits = torch.einsum("bnd,sd->bns", tile_vectors, slot_queries) * scale
        if safe_prior is not None and prior_weight is not None:
            one_hot = F.one_hot(
                safe_prior, num_classes=self.config.tile_count
            ).to(dtype=logits.dtype)
            logits = logits + (
                self.qap_logit_scale.to(logits.dtype)
                * prior_weight.unsqueeze(2)
                * one_hot
            )
        log_assignment = log_sinkhorn(
            logits,
            iterations=self.config.sinkhorn_iterations,
            temperature=self.config.sinkhorn_temperature,
        )
        return ViTSinkhornOutput(
            logits=logits,
            log_assignment=log_assignment,
            edge_embeddings=edge_embeddings,
            contextual_tiles=contextual,
            prior_keep_mask=prior_keep_mask,
        )


def hungarian_position_to_tile(
    logits: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Project tile-to-position scores to ``position_to_tile`` permutations."""

    if isinstance(logits, torch.Tensor):
        values = logits.detach().float().cpu().numpy()
    else:
        values = np.asarray(logits, dtype=np.float32)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None]
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("logits must be NxN or BxNxN")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    outputs = np.empty((values.shape[0], values.shape[1]), dtype=np.int32)
    for batch_index, matrix in enumerate(values):
        tile_indices, positions = linear_sum_assignment(-matrix.astype(np.float64))
        position_to_tile = np.empty(len(positions), dtype=np.int32)
        position_to_tile[positions] = tile_indices.astype(np.int32, copy=False)
        if not np.array_equal(np.sort(position_to_tile), np.arange(len(positions))):
            raise RuntimeError("Hungarian projection did not produce a permutation")
        outputs[batch_index] = position_to_tile
    return outputs[0] if squeeze else outputs


def position_to_tile_to_tile_to_position(
    position_to_tile: np.ndarray | torch.Tensor,
) -> np.ndarray:
    values = (
        position_to_tile.detach().cpu().numpy()
        if isinstance(position_to_tile, torch.Tensor)
        else np.asarray(position_to_tile)
    )
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("position_to_tile must be a one-dimensional integer array")
    if not np.array_equal(np.sort(values), np.arange(len(values))):
        raise ValueError("position_to_tile must be a permutation")
    inverse = np.empty(len(values), dtype=np.int32)
    inverse[values] = np.arange(len(values), dtype=np.int32)
    return inverse


def _validate_targets(
    target_tile_to_position: torch.Tensor,
    *,
    tile_count: int,
) -> torch.Tensor:
    if target_tile_to_position.ndim != 2 or target_tile_to_position.shape[1] != tile_count:
        raise ValueError(f"targets must have shape Bx{tile_count}")
    targets = target_tile_to_position.long()
    invalid = (targets < -1) | (targets >= tile_count)
    if invalid.any():
        raise ValueError("targets must be -1 (unknown) or valid positions")
    for row in targets.detach().cpu().numpy():
        known = row[row >= 0]
        if len(np.unique(known)) != len(known):
            raise ValueError("known target positions must be one-to-one per source")
    return targets


def assignment_nll_loss(
    log_assignment: torch.Tensor,
    target_tile_to_position: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    if log_assignment.ndim != 3 or log_assignment.shape[1] != log_assignment.shape[2]:
        raise ValueError("log_assignment must be BxNxN")
    targets = _validate_targets(
        target_tile_to_position, tile_count=log_assignment.shape[1]
    )
    if targets.shape[:2] != log_assignment.shape[:2]:
        raise ValueError("target batch shape does not match assignment")
    valid = targets >= 0
    safe = targets.clamp_min(0)
    losses = -log_assignment.gather(2, safe.unsqueeze(2)).squeeze(2)
    if confidence is None:
        weights = valid.to(dtype=losses.dtype)
    else:
        if confidence.shape != targets.shape:
            raise ValueError("confidence must match target shape")
        weights = confidence.to(dtype=losses.dtype).clamp(0.0, 1.0)
        if not torch.isfinite(weights).all():
            raise ValueError("confidence contains non-finite values")
        weights = weights * valid
    denominator = weights.sum()
    if denominator.detach().item() <= 0:
        return log_assignment.sum() * 0.0
    return (losses * weights).sum() / denominator


def directional_neighbor_contrast_loss(
    edge_embeddings: torch.Tensor,
    target_tile_to_position: torch.Tensor,
    *,
    grid_size: int,
    confidence: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised right/down edge InfoNCE, including partial real gold."""

    if edge_embeddings.ndim != 4 or edge_embeddings.shape[2] != 4:
        raise ValueError("edge_embeddings must have shape BxNx4xE")
    batch, tile_count = edge_embeddings.shape[:2]
    if grid_size * grid_size != tile_count:
        raise ValueError("grid_size does not match tile count")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    targets = _validate_targets(target_tile_to_position, tile_count=tile_count)
    if targets.shape != (batch, tile_count):
        raise ValueError("target shape does not match edge embeddings")
    if confidence is not None and confidence.shape != targets.shape:
        raise ValueError("confidence must match targets")
    edges = F.normalize(edge_embeddings.float(), dim=3)
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for batch_index in range(batch):
        positions = targets[batch_index]
        known_tiles = torch.nonzero(positions >= 0, as_tuple=False).flatten()
        position_to_tile = torch.full(
            (tile_count,), -1, device=positions.device, dtype=torch.long
        )
        if len(known_tiles):
            position_to_tile[positions[known_tiles]] = known_tiles
        for query_direction, key_direction, offset, boundary in (
            (RIGHT, LEFT, 1, "right"),
            (BOTTOM, TOP, grid_size, "down"),
        ):
            query_positions = positions[known_tiles]
            if boundary == "right":
                inside = query_positions.remainder(grid_size) < grid_size - 1
            else:
                inside = query_positions < tile_count - grid_size
            query_tiles = known_tiles[inside]
            neighbour_positions = positions[query_tiles] + offset
            target_tiles = position_to_tile[neighbour_positions]
            paired = target_tiles >= 0
            query_tiles = query_tiles[paired]
            target_tiles = target_tiles[paired]
            if len(query_tiles) == 0:
                continue
            logits = (
                edges[batch_index, :, query_direction]
                @ edges[batch_index, :, key_direction].T
            ) / temperature
            diagonal = torch.arange(tile_count, device=logits.device)
            logits = logits.masked_fill(
                torch.eye(tile_count, device=logits.device, dtype=torch.bool),
                -1e4,
            )
            del diagonal
            pair_losses = F.cross_entropy(
                logits[query_tiles], target_tiles, reduction="none"
            )
            if confidence is None:
                pair_weights = torch.ones_like(pair_losses)
            else:
                values = confidence[batch_index].to(pair_losses.dtype).clamp(0.0, 1.0)
                pair_weights = torch.sqrt(
                    values[query_tiles] * values[target_tiles]
                )
            losses.append((pair_losses * pair_weights).sum())
            weights.append(pair_weights.sum())
    if not losses:
        return edge_embeddings.sum() * 0.0
    denominator = torch.stack(weights).sum().clamp_min(1e-8)
    return torch.stack(losses).sum() / denominator


def _directional_consistency(
    probabilities: torch.Tensor,
    edge_embeddings: torch.Tensor,
    *,
    grid_size: int,
    query_direction: int,
    key_direction: int,
    offset: int,
    topk: int,
    temperature: float,
) -> torch.Tensor:
    batch, tile_count = probabilities.shape[:2]
    queries = F.normalize(edge_embeddings[:, :, query_direction].float(), dim=2)
    keys = F.normalize(edge_embeddings[:, :, key_direction].float(), dim=2)
    similarities = torch.einsum("bne,bme->bnm", queries, keys) / temperature
    diagonal = torch.eye(
        tile_count, device=similarities.device, dtype=torch.bool
    ).unsqueeze(0)
    similarities = similarities.masked_fill(diagonal, -1e4)
    count = min(topk, tile_count - 1)
    top_values, candidates = similarities.topk(count, dim=2)

    positions = torch.arange(tile_count, device=probabilities.device)
    if offset == 1:
        source_positions = positions[positions.remainder(grid_size) < grid_size - 1]
    else:
        source_positions = positions[positions < tile_count - grid_size]
    destination_positions = source_positions + offset
    source_mass = probabilities[:, :, source_positions]
    gather_indices = candidates.reshape(batch, tile_count * count, 1).expand(
        -1, -1, tile_count
    )
    candidate_assignments = probabilities.gather(1, gather_indices).reshape(
        batch, tile_count, count, tile_count
    )
    destination_mass = candidate_assignments[:, :, :, destination_positions]
    adjacency_scores = (
        source_mass.unsqueeze(2) * destination_mass
    ).sum(dim=3)
    epsilon = 1e-8
    assignment_distribution = (adjacency_scores + epsilon) / (
        adjacency_scores.sum(dim=2, keepdim=True) + epsilon * count
    )
    edge_distribution = F.softmax(top_values, dim=2)
    cross_assignment = -(
        edge_distribution.detach()
        * torch.log(assignment_distribution.clamp_min(epsilon))
    ).sum(dim=2)
    cross_edge = -(
        assignment_distribution.detach()
        * torch.log(edge_distribution.clamp_min(epsilon))
    ).sum(dim=2)
    # Tiles assigned to the outer boundary do not have a neighbour in this
    # direction; use their interior probability as a soft validity weight.
    validity = source_mass.sum(dim=2).detach()
    return (
        0.5 * (cross_assignment + cross_edge) * validity
    ).sum() / validity.sum().clamp_min(epsilon)


def neighbor_consistency_loss(
    log_assignment: torch.Tensor,
    edge_embeddings: torch.Tensor,
    *,
    grid_size: int,
    topk: int = 16,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Align directional edge candidates with Sinkhorn-implied neighbours.

    Only the strongest ``topk`` edge candidates are materialized.  At 576
    tiles this is O(N^2 + k*N^2), avoiding a dense O(N^3) adjacency tensor.
    """

    if log_assignment.ndim != 3 or log_assignment.shape[1] != log_assignment.shape[2]:
        raise ValueError("log_assignment must be BxNxN")
    if edge_embeddings.shape[:3] != (
        log_assignment.shape[0],
        log_assignment.shape[1],
        4,
    ):
        raise ValueError("edge embeddings do not match assignment shape")
    if grid_size * grid_size != log_assignment.shape[1]:
        raise ValueError("grid_size does not match assignment")
    if topk <= 0 or temperature <= 0:
        raise ValueError("topk and temperature must be positive")
    probabilities = log_assignment.float().exp()
    right = _directional_consistency(
        probabilities,
        edge_embeddings,
        grid_size=grid_size,
        query_direction=RIGHT,
        key_direction=LEFT,
        offset=1,
        topk=topk,
        temperature=temperature,
    )
    down = _directional_consistency(
        probabilities,
        edge_embeddings,
        grid_size=grid_size,
        query_direction=BOTTOM,
        key_direction=TOP,
        offset=grid_size,
        topk=topk,
        temperature=temperature,
    )
    return 0.5 * (right + down)


def vit_sinkhorn_losses(
    output: ViTSinkhornOutput,
    target_tile_to_position: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    grid_size: int = 24,
    assignment_weight: float = 1.0,
    directional_contrast_weight: float = 0.20,
    neighbor_consistency_weight: float = 0.05,
    contrast_temperature: float = 0.07,
    consistency_topk: int = 16,
) -> dict[str, torch.Tensor]:
    for name, value in (
        ("assignment_weight", assignment_weight),
        ("directional_contrast_weight", directional_contrast_weight),
        ("neighbor_consistency_weight", neighbor_consistency_weight),
    ):
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
    assignment = assignment_nll_loss(
        output.log_assignment,
        target_tile_to_position,
        confidence=confidence,
    )
    contrast = directional_neighbor_contrast_loss(
        output.edge_embeddings,
        target_tile_to_position,
        grid_size=grid_size,
        confidence=confidence,
        temperature=contrast_temperature,
    )
    consistency = neighbor_consistency_loss(
        output.log_assignment,
        output.edge_embeddings,
        grid_size=grid_size,
        topk=consistency_topk,
        temperature=contrast_temperature,
    )
    total = (
        assignment_weight * assignment
        + directional_contrast_weight * contrast
        + neighbor_consistency_weight * consistency
    )
    return {
        "total": total,
        "assignment": assignment,
        "directional_contrast": contrast,
        "neighbor_consistency": consistency,
    }


def permutation_metrics_from_logits(
    logits: np.ndarray | torch.Tensor,
    target_tile_to_position: np.ndarray | torch.Tensor,
    *,
    grid_size: int,
) -> dict[str, float | int | bool]:
    """Exact Hungarian layout metrics for one fully labelled source."""

    position_to_tile = hungarian_position_to_tile(logits)
    if position_to_tile.ndim != 1:
        raise ValueError("metrics expect one source")
    targets = (
        target_tile_to_position.detach().cpu().numpy()
        if isinstance(target_tile_to_position, torch.Tensor)
        else np.asarray(target_tile_to_position)
    )
    tile_count = grid_size * grid_size
    if targets.shape != (tile_count,) or not np.array_equal(
        np.sort(targets), np.arange(tile_count)
    ):
        raise ValueError("metrics require one complete target permutation")
    placed_targets = targets[position_to_tile]
    expected = np.arange(tile_count, dtype=np.int32)
    displacement = np.abs(placed_targets // grid_size - expected // grid_size) + np.abs(
        placed_targets % grid_size - expected % grid_size
    )
    grid = placed_targets.reshape(grid_size, grid_size)
    right = (grid[:, :-1] % grid_size != grid_size - 1) & (
        grid[:, 1:] == grid[:, :-1] + 1
    )
    down = grid[1:, :] == grid[:-1, :] + grid_size
    return {
        "valid_permutation": True,
        "position_accuracy": float(np.mean(placed_targets == expected)),
        "row_accuracy": float(
            np.mean(placed_targets // grid_size == expected // grid_size)
        ),
        "column_accuracy": float(
            np.mean(placed_targets % grid_size == expected % grid_size)
        ),
        "mean_manhattan": float(np.mean(displacement)),
        "median_manhattan": float(np.median(displacement)),
        "within_one_manhattan": float(np.mean(displacement <= 1)),
        "right_adjacency": float(np.mean(right)),
        "down_adjacency": float(np.mean(down)),
        "combined_adjacency": float(0.5 * (np.mean(right) + np.mean(down))),
        "exact_solved": bool(np.array_equal(placed_targets, expected)),
        "position_to_tile": position_to_tile.tolist(),
    }


def make_synthetic_smoke_batch(
    *,
    grid_size: int = 4,
    tile_size: int = 20,
    batch_size: int = 1,
    seed: int = 20260711,
) -> dict[str, torch.Tensor]:
    """Create a tiny coherent-image batch with known shuffled permutations."""

    if min(grid_size, tile_size, batch_size) <= 0:
        raise ValueError("grid_size, tile_size, and batch_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    side = grid_size * tile_size
    y = torch.linspace(0.0, 1.0, side).view(side, 1)
    x = torch.linspace(0.0, 1.0, side).view(1, side)
    base = torch.stack(
        [
            x.expand(side, side),
            y.expand(side, side),
            0.5 + 0.25 * torch.sin(8.0 * x) * torch.cos(6.0 * y),
        ],
        dim=0,
    ).clamp(0.0, 1.0)
    ordered = (
        base.reshape(3, grid_size, tile_size, grid_size, tile_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_size * grid_size, 3, tile_size, tile_size)
    )
    raw_parts: list[torch.Tensor] = []
    restored_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    tile_count = grid_size * grid_size
    for _ in range(batch_size):
        target = torch.randperm(tile_count, generator=generator)
        clean_slots = ordered[target]
        gains = 0.75 + 0.50 * torch.rand(
            tile_count, 1, 1, 1, generator=generator
        )
        offsets = 0.10 * (
            2.0 * torch.rand(tile_count, 1, 1, 1, generator=generator) - 1.0
        )
        noise = 0.06 * torch.randn(
            clean_slots.shape, generator=generator
        )
        raw = (clean_slots * gains + offsets + noise).clamp(0.0, 1.0)
        restored = (0.65 * raw + 0.35 * clean_slots).clamp(0.0, 1.0)
        raw_parts.append(raw)
        restored_parts.append(restored)
        target_parts.append(target)
    return {
        "raw_tiles": torch.stack(raw_parts),
        "restored_tiles": torch.stack(restored_parts),
        "target_tile_to_position": torch.stack(target_parts),
    }
