"""Large dirty-only Transformer reranker for ordered puzzle seams.

This architecture is deliberately different from the earlier small content
verifier.  Every ordered candidate pair is converted to a canonical 20x40
join, encoded by full Transformer depth, and scored independently.  The model
therefore learns boundary compatibility rather than content substitution and
is equivariant to shortlist order by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from aiijc_puzzle.candidate_supply import (
    DEFAULT_VIEWS,
    analytic_views,
    classical_costs,
    recover_layout,
    top_candidates,
)
from aiijc_puzzle.protocol import TILE_COUNT


@dataclass(frozen=True)
class SeamCandidateRow:
    """One dirty-only ordered-neighbour shortlist."""

    anchor: int
    candidates: np.ndarray
    features: np.ndarray
    direction: int
    baseline_choice: int
    exact_choice: int = -1
    trusted: bool = False


@dataclass(frozen=True)
class SeamCandidateBoard:
    """Dirty-only candidate board and full classical score matrices."""

    filename: str
    tiles: np.ndarray
    rows: tuple[SeamCandidateRow, ...]
    right_scores: np.ndarray
    down_scores: np.ndarray
    views: tuple[str, ...]


def _candidate_features(
    anchor: int,
    candidates: np.ndarray,
    costs: Sequence[np.ndarray],
    rankings: Sequence[np.ndarray],
    candidate_k: int,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for cost, ranking in zip(costs, rankings, strict=True):
        columns.append(np.clip(cost[anchor, candidates], -10.0, 20.0) / 10.0)
        rank = np.full(len(candidates), candidate_k, dtype=np.float32)
        for position, tile in enumerate(ranking[anchor, :candidate_k]):
            rank[candidates == tile] = position
        columns.append(rank / max(candidate_k, 1))
        columns.append((rank < candidate_k).astype(np.float32))
    return np.stack(columns, axis=1).astype(np.float32)


def build_inference_board(
    dirty_tiles: np.ndarray,
    *,
    filename: str = "",
    views: Sequence[str] = DEFAULT_VIEWS,
    candidate_k: int = 4,
) -> SeamCandidateBoard:
    """Build all candidates and features from dirty pixels only."""

    dirty = np.asarray(dirty_tiles)
    if dirty.shape != (TILE_COUNT, 20, 20, 3) or dirty.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 dirty tiles (576,20,20,3), got {dirty.dtype} {dirty.shape}"
        )
    if not 1 <= candidate_k < TILE_COUNT:
        raise ValueError(f"candidate_k must be in [1, {TILE_COUNT - 1}]")
    view_names = tuple(views)
    if not view_names:
        raise ValueError("at least one candidate view is required")

    directional_costs: list[list[np.ndarray]] = [[], []]
    directional_rankings: list[list[np.ndarray]] = [[], []]
    for transformed in analytic_views(dirty, view_names).values():
        right, down = classical_costs(transformed)
        for direction, cost in enumerate((right, down)):
            directional_costs[direction].append(cost)
            directional_rankings[direction].append(top_candidates(cost, candidate_k))

    mean_costs = [
        np.mean(np.stack(costs, axis=0), axis=0, dtype=np.float64).astype(np.float32)
        for costs in directional_costs
    ]
    rows: list[SeamCandidateRow] = []
    for direction in range(2):
        costs = directional_costs[direction]
        rankings = directional_rankings[direction]
        for anchor in range(TILE_COUNT):
            pool = np.unique(
                np.concatenate([ranking[anchor, :candidate_k] for ranking in rankings])
            ).astype(np.int64)
            ensemble_cost = mean_costs[direction][anchor, pool]
            pool = pool[np.lexsort((pool, ensemble_cost))]
            features = _candidate_features(anchor, pool, costs, rankings, candidate_k)
            rows.append(
                SeamCandidateRow(
                    anchor=anchor,
                    candidates=pool,
                    features=features,
                    direction=direction,
                    baseline_choice=0,
                )
            )
    return SeamCandidateBoard(
        filename=filename,
        tiles=dirty.copy(),
        rows=tuple(rows),
        right_scores=-mean_costs[0],
        down_scores=-mean_costs[1],
        views=view_names,
    )


def attach_exact_training_labels(
    board: SeamCandidateBoard, clean_tiles: np.ndarray
) -> SeamCandidateBoard:
    """Attach target-assisted exact-neighbour labels for train/evaluation only."""

    clean = np.asarray(clean_tiles)
    if clean.shape != board.tiles.shape or clean.dtype != np.uint8:
        raise ValueError(
            f"expected clean tiles matching dirty tiles, got {clean.dtype} {clean.shape}"
        )
    recovered = recover_layout(board.tiles, clean)
    position_of_dirty = recovered.position_of_dirty
    confidence_cut = float(np.median(recovered.margin_at_position))
    labelled: list[SeamCandidateRow] = []
    for row in board.rows:
        position = int(position_of_dirty[row.anchor])
        legal = position % 24 != 23 if row.direction == 0 else position < TILE_COUNT - 24
        if not legal:
            continue
        neighbour_position = position + (1 if row.direction == 0 else 24)
        exact_tile = int(recovered.dirty_at_position[neighbour_position])
        matches = np.flatnonzero(row.candidates == exact_tile)
        exact_choice = int(matches[0]) if len(matches) else -1
        trusted = bool(
            recovered.margin_at_position[position] >= confidence_cut
            and recovered.margin_at_position[neighbour_position] >= confidence_cut
        )
        labelled.append(replace(row, exact_choice=exact_choice, trusted=trusted))
    return replace(board, rows=tuple(labelled))


def rerank_score_matrices(
    board: SeamCandidateBoard, row_logits: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply shortlist order changes while preserving every classical score multiset.

    This scale-preserving rank substitution makes the learned edge matrices safe
    inputs to the unchanged global solver: only candidate order changes, not the
    per-row distribution or any non-candidate value.
    """

    if len(row_logits) != len(board.rows):
        raise ValueError("one logit vector is required for every board row")
    matrices = [board.right_scores.copy(), board.down_scores.copy()]
    for row, logits in zip(board.rows, row_logits, strict=True):
        values = np.asarray(logits, dtype=np.float64)
        if values.shape != (len(row.candidates),) or not np.all(np.isfinite(values)):
            raise ValueError("candidate logits have the wrong shape or contain non-finite values")
        matrix = matrices[row.direction]
        classical_values = np.sort(matrix[row.anchor, row.candidates])[::-1]
        learned_order = np.lexsort((row.candidates, -values))
        matrix[row.anchor, row.candidates[learned_order]] = classical_values
    return matrices[0], matrices[1]


def _fixed_gaussian_blur(tensor: torch.Tensor) -> torch.Tensor:
    channels = tensor.shape[1]
    kernel = tensor.new_tensor((1.0, 2.0, 1.0))
    kernel = (kernel[:, None] * kernel[None, :]) / 16.0
    weight = kernel.expand(channels, 1, 3, 3)
    padded = functional.pad(tensor, (1, 1, 1, 1), mode="reflect")
    return functional.conv2d(padded, weight, groups=channels)


def _tile_channels(tiles: torch.Tensor) -> torch.Tensor:
    raw = tiles.float() / 255.0
    mean = raw.mean(dim=(2, 3), keepdim=True)
    std = raw.std(dim=(1, 2, 3), keepdim=True).clamp_min(1.0 / 255.0)
    local = ((raw - mean) / std).clamp(-4.0, 4.0) / 4.0
    denoised = _fixed_gaussian_blur(raw)
    high_pass = ((raw - denoised) * 4.0).clamp(-1.0, 1.0)
    return torch.cat((raw * 2.0 - 1.0, local, denoised * 2.0 - 1.0, high_pass), dim=1)


class OrderedSeamTransformer(nn.Module):
    """Deep canonical-join Transformer with explicit high-resolution seam tokens."""

    def __init__(
        self,
        *,
        feature_dim: int,
        dim: int = 256,
        heads: int = 8,
        layers: int = 10,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if layers < 1 or mlp_ratio < 2:
            raise ValueError("layers must be positive and mlp_ratio must be at least two")
        self.feature_dim = feature_dim
        self.dim = dim
        self.patch_embed = nn.Conv2d(12, dim, kernel_size=4, stride=4)
        self.seam_embed = nn.Sequential(nn.Linear(12 * 4, dim), nn.LayerNorm(dim))
        self.classical_embed = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.patch_position = nn.Parameter(torch.empty(1, 50, dim))
        self.seam_position = nn.Parameter(torch.empty(1, 20, dim))
        self.token_type = nn.Parameter(torch.empty(1, 4, dim))
        self.direction = nn.Embedding(2, dim)
        nn.init.trunc_normal_(self.patch_position, std=0.02)
        nn.init.trunc_normal_(self.seam_position, std=0.02)
        nn.init.trunc_normal_(self.token_type, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(dim))
        self.residual_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        # Exact classical behaviour at initialisation; learning can only add a
        # residual after seeing supervised train seams.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        classical_features: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        if anchors.shape != candidates.shape or anchors.ndim != 4:
            raise ValueError("anchors and candidates must have equal NCHW shapes")
        if anchors.shape[1:] != (3, 20, 20):
            raise ValueError(f"expected N x 3 x 20 x 20 tiles, got {anchors.shape}")
        if classical_features.shape != (len(anchors), self.feature_dim):
            raise ValueError("classical feature shape mismatch")
        if directions.shape != (len(anchors),):
            raise ValueError("direction shape mismatch")

        # Rotate down-pairs together so anchor-bottom/candidate-top becomes the
        # same canonical right seam used for horizontal pairs.
        down = directions == 1
        if bool(down.any()):
            anchors = anchors.clone()
            candidates = candidates.clone()
            anchors[down] = torch.rot90(anchors[down], 1, dims=(-2, -1))
            candidates[down] = torch.rot90(candidates[down], 1, dims=(-2, -1))
        anchor_channels = _tile_channels(anchors)
        candidate_channels = _tile_channels(candidates)
        joined = torch.cat((anchor_channels, candidate_channels), dim=3)

        patches = self.patch_embed(joined).flatten(2).transpose(1, 2)
        if patches.shape[1] != 50:
            raise RuntimeError(f"patch embedding produced {patches.shape[1]} rather than 50 tokens")
        patches = patches + self.patch_position + self.token_type[:, 2:3]

        # Four high-resolution columns straddle the canonical boundary.  Each
        # image row becomes a separate token, retaining 2-pixel seam details
        # that the 4x4 patch stream would otherwise average away.
        seam_strip = torch.cat(
            (anchor_channels[:, :, :, -2:], candidate_channels[:, :, :, :2]), dim=3
        )
        seam_tokens = seam_strip.permute(0, 2, 1, 3).flatten(2)
        seam_tokens = self.seam_embed(seam_tokens) + self.seam_position + self.token_type[:, 3:4]

        classical = self.classical_embed(classical_features)[:, None] + self.token_type[:, 1:2]
        cls = self.cls.expand(len(anchors), -1, -1)
        cls = cls + self.direction(directions)[:, None] + self.token_type[:, 0:1]
        encoded = self.encoder(torch.cat((cls, classical, patches, seam_tokens), dim=1))
        prior = -classical_features[:, 0::3].mean(dim=1)
        return prior + self.residual_head(encoded[:, 0]).squeeze(1)


def listwise_hard_negative_loss(
    logits: torch.Tensor,
    row_ids: torch.Tensor,
    exact_flat_indices: torch.Tensor,
    *,
    margin: float = 0.25,
    margin_weight: float = 0.25,
) -> torch.Tensor:
    """Exact-neighbour listwise CE plus hardest-negative margin."""

    if logits.ndim != 1 or row_ids.shape != logits.shape:
        raise ValueError("logits and row_ids must be equal one-dimensional tensors")
    if exact_flat_indices.ndim != 1:
        raise ValueError("exact_flat_indices must be one-dimensional")
    losses = []
    for row, exact_index in enumerate(exact_flat_indices):
        mask = row_ids == row
        row_logits = logits[mask]
        local_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        local_exact = torch.nonzero(local_indices == exact_index, as_tuple=False).squeeze()
        if local_exact.numel() != 1:
            raise ValueError("every row must contain exactly one exact flat index")
        positive = row_logits[local_exact]
        ce = torch.logsumexp(row_logits, dim=0) - positive
        negative_mask = torch.arange(len(row_logits), device=logits.device) != local_exact
        hardest = row_logits[negative_mask].max()
        ranking = functional.relu(margin - positive + hardest)
        losses.append(ce + margin_weight * ranking)
    return torch.stack(losses).mean()


def augment_ordered_pairs(
    anchors: np.ndarray,
    candidates: np.ndarray,
    row_ids: np.ndarray,
    *,
    rng: np.random.Generator,
    jpeg_probability: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply strong legal corruption-matched augmentation to ordered pairs.

    Photometric corruption is independent per tile, matching the generator.
    A shared vertical flip per query preserves ordered adjacency.  JPEG is a
    genuine RGB->BGR encode/decode round trip rather than a differentiable
    approximation.
    """

    anchor_values = np.asarray(anchors)
    candidate_values = np.asarray(candidates)
    ids = np.asarray(row_ids)
    if anchor_values.ndim != 4 or anchor_values.shape[1:] != (20, 20, 3):
        raise ValueError("anchors must be B x 20 x 20 x 3")
    if candidate_values.ndim != 4 or candidate_values.shape[1:] != (20, 20, 3):
        raise ValueError("candidates must be N x 20 x 20 x 3")
    if ids.shape != (len(candidate_values),) or np.any(ids < 0) or np.any(ids >= len(anchors)):
        raise ValueError("row_ids do not index anchors")

    def photometric(batch: np.ndarray) -> np.ndarray:
        output = np.empty_like(batch)
        for index, tile in enumerate(batch):
            value = tile.astype(np.float32)
            channel_gain = rng.uniform(0.88, 1.12, size=(1, 1, 3))
            contrast = float(rng.uniform(0.75, 1.25))
            brightness = float(rng.uniform(-18.0, 18.0))
            spatial_mean = value.mean(axis=(0, 1), keepdims=True)
            value = (value - spatial_mean) * contrast + spatial_mean
            value = value * channel_gain + brightness
            if rng.random() < 0.75:
                value += rng.normal(0.0, rng.uniform(2.0, 14.0), size=value.shape)
            value = np.clip(value, 0, 255).astype(np.uint8)
            if rng.random() < 0.35:
                sigma = float(rng.uniform(0.25, 1.15))
                value = cv2.GaussianBlur(value, (3, 3), sigma)
            if rng.random() < jpeg_probability:
                quality = int(rng.integers(35, 91))
                bgr = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
                success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not success:
                    raise RuntimeError("JPEG augmentation encode failed")
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                value = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            output[index] = value
        return output

    augmented_anchors = photometric(anchor_values)
    augmented_candidates = photometric(candidate_values)
    flip_rows = rng.random(len(anchor_values)) < 0.5
    augmented_anchors[flip_rows] = augmented_anchors[flip_rows, ::-1]
    for candidate_index, row_id in enumerate(ids):
        if flip_rows[row_id]:
            augmented_candidates[candidate_index] = augmented_candidates[candidate_index, ::-1]
    return augmented_anchors, augmented_candidates
