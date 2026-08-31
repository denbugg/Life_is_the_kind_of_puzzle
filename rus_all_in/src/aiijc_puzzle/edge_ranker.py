"""Dirty-only pairwise edge reranking for strict bijective puzzle assembly.

The model scores a proposed directed join, not an absolute board position.  A
small joint CNN sees the hypothetical 20x40 two-tile collage at full seam and
downsampled whole-tile scales.  Clean targets are used only to recover exact
neighbour labels and a clean-continuity teacher during training/evaluation.

At inference, the frozen bilateral compatibility remains defined for all tile
pairs.  The network changes only candidates emitted by a deterministic union
of analytic dirty-tile views, so full-board decoding stays tractable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import log_softmax
from torch import nn

from aiijc_puzzle.candidate_supply import (
    DEFAULT_VIEWS,
    analytic_views,
    classical_costs,
    recover_layout,
    top_candidates,
)


def _cost_to_logp(cost: np.ndarray) -> np.ndarray:
    """Convert an arbitrary square dissimilarity into row log-probabilities."""

    value = np.asarray(cost, dtype=np.float32).copy()
    if value.ndim != 2 or value.shape[0] != value.shape[1] or len(value) < 2:
        raise ValueError(f"expected a square cost matrix, got {value.shape}")
    diagonal = np.eye(len(value), dtype=bool)
    off_diagonal = value[~diagonal].reshape(len(value), len(value) - 1)
    median = np.median(off_diagonal, axis=1, keepdims=True)
    mad = np.median(np.abs(off_diagonal - median), axis=1, keepdims=True)
    logits = -(value - median) / np.maximum(mad, 1e-6)
    np.fill_diagonal(logits, -1e4)
    return log_softmax(logits, axis=1).astype(np.float32)


@dataclass(frozen=True)
class EdgeRow:
    """One directed dirty-tile query, with optional target-assisted labels."""

    anchor: int
    candidates: np.ndarray
    features: np.ndarray
    baseline_scores: np.ndarray
    direction: int
    exact_candidate: int = -1
    trusted_query: bool = False
    candidate_mapping_margin: np.ndarray | None = None
    mapping_confidence_cut: float = float("nan")
    teacher_scores: np.ndarray | None = None


@dataclass(frozen=True)
class EdgeBoard:
    """Inference-visible tiles, candidate rows, and bilateral score matrices."""

    filename: str
    tiles: np.ndarray
    rows: tuple[EdgeRow, ...]
    right_baseline: np.ndarray
    down_baseline: np.ndarray
    views: tuple[str, ...]
    candidate_k: int


def _candidate_features(
    anchor: int,
    candidates: np.ndarray,
    costs: Sequence[np.ndarray],
    ranked: Sequence[np.ndarray],
    candidate_k: int,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for cost, order in zip(costs, ranked, strict=True):
        columns.append(np.clip(cost[anchor, candidates], -10.0, 20.0) / 10.0)
        rank = np.full(len(candidates), candidate_k, dtype=np.float32)
        for position, tile in enumerate(order[anchor, :candidate_k]):
            rank[candidates == tile] = position
        columns.append(rank / max(candidate_k, 1))
        columns.append((rank < candidate_k).astype(np.float32))
    return np.stack(columns, axis=1).astype(np.float32)


def build_inference_board(
    dirty: np.ndarray,
    *,
    filename: str = "",
    views: Sequence[str] = DEFAULT_VIEWS,
    candidate_k: int = 5,
) -> EdgeBoard:
    """Build candidate rows using only corrupted, shuffled input tiles."""

    dirty = np.asarray(dirty)
    if dirty.ndim != 4 or dirty.shape[-1] != 3 or dirty.shape[1:3] != (20, 20):
        raise ValueError(f"expected N x 20 x 20 x 3 dirty tiles, got {dirty.shape}")
    n = len(dirty)
    if not 1 <= candidate_k < n:
        raise ValueError(f"candidate_k must be in [1, {n - 1}], got {candidate_k}")
    view_names = tuple(views)
    if "bilateral" not in view_names:
        raise ValueError("views must contain the frozen bilateral baseline")

    direction_costs: list[list[np.ndarray]] = [[], []]
    direction_ranked: list[list[np.ndarray]] = [[], []]
    for transformed in analytic_views(dirty, view_names).values():
        right_cost, down_cost = classical_costs(transformed)
        for direction, cost in enumerate((right_cost, down_cost)):
            direction_costs[direction].append(cost)
            direction_ranked[direction].append(top_candidates(cost, candidate_k))

    bilateral_index = view_names.index("bilateral")
    bilateral_costs = (
        direction_costs[0][bilateral_index],
        direction_costs[1][bilateral_index],
    )
    baseline_matrices = tuple(_cost_to_logp(cost) for cost in bilateral_costs)
    rows: list[EdgeRow] = []
    for direction in (0, 1):
        costs = direction_costs[direction]
        ranked = direction_ranked[direction]
        baseline = baseline_matrices[direction]
        for anchor in range(n):
            candidates = np.unique(
                np.concatenate([order[anchor, :candidate_k] for order in ranked])
            ).astype(np.int64)
            ensemble_cost = np.mean(
                np.stack([cost[anchor, candidates] for cost in costs], axis=1), axis=1
            )
            candidates = candidates[np.lexsort((candidates, ensemble_cost))]
            rows.append(
                EdgeRow(
                    anchor=anchor,
                    candidates=candidates,
                    features=_candidate_features(anchor, candidates, costs, ranked, candidate_k),
                    baseline_scores=baseline[anchor, candidates].astype(np.float32),
                    direction=direction,
                )
            )
    return EdgeBoard(
        filename=filename,
        tiles=np.clip(dirty, 0, 255).astype(np.uint8),
        rows=tuple(rows),
        right_baseline=baseline_matrices[0],
        down_baseline=baseline_matrices[1],
        views=view_names,
        candidate_k=candidate_k,
    )


def _clean_continuity_cost(
    anchor: np.ndarray,
    candidates: np.ndarray,
    direction: int,
) -> np.ndarray:
    """Symmetric one-step extrapolation error on clean target tiles."""

    anchor_f = np.asarray(anchor, dtype=np.float32)
    candidates_f = np.asarray(candidates, dtype=np.float32)
    if direction == 0:
        outgoing = 2.0 * anchor_f[:, -1] - anchor_f[:, -2]
        incoming = candidates_f[:, :, 0]
        reverse_outgoing = anchor_f[:, -1]
        reverse_incoming = 2.0 * candidates_f[:, :, 0] - candidates_f[:, :, 1]
    elif direction == 1:
        outgoing = 2.0 * anchor_f[-1] - anchor_f[-2]
        incoming = candidates_f[:, 0]
        reverse_outgoing = anchor_f[-1]
        reverse_incoming = 2.0 * candidates_f[:, 0] - candidates_f[:, 1]
    else:
        raise ValueError(f"direction must be 0 or 1, got {direction}")
    forward = np.mean(np.square(incoming - outgoing[None]), axis=(1, 2))
    reverse = np.mean(np.square(reverse_outgoing[None] - reverse_incoming), axis=(1, 2))
    return 0.5 * (forward + reverse)


def attach_target_labels(board: EdgeBoard, clean: np.ndarray) -> EdgeBoard:
    """Attach target-assisted exact and clean-continuity labels to a board.

    No inference-visible field is rebuilt here.  This explicit second phase is
    used by the runner after predictions and layouts have already been frozen.
    """

    clean = np.asarray(clean)
    if clean.shape != board.tiles.shape:
        raise ValueError(f"clean shape {clean.shape} differs from dirty {board.tiles.shape}")
    n = len(clean)
    grid = round(n**0.5)
    if grid * grid != n:
        raise ValueError(f"tile count must be square, got {n}")
    recovered = recover_layout(board.tiles, clean)
    position_of_dirty = recovered.position_of_dirty
    confidence_cut = float(np.median(recovered.margin_at_position))
    labelled: list[EdgeRow] = []
    for row in board.rows:
        position = int(position_of_dirty[row.anchor])
        legal = position % grid != grid - 1 if row.direction == 0 else position < n - grid
        if not legal:
            continue
        neighbour_position = position + (1 if row.direction == 0 else grid)
        true_dirty = int(recovered.dirty_at_position[neighbour_position])
        match = np.flatnonzero(row.candidates == true_dirty)
        candidate_positions = position_of_dirty[row.candidates]
        teacher_cost = _clean_continuity_cost(
            clean[position], clean[candidate_positions], row.direction
        )
        median = float(np.median(teacher_cost))
        mad = float(np.median(np.abs(teacher_cost - median)))
        teacher_scores = -((teacher_cost - median) / max(mad, 1e-6)).astype(np.float32)
        trusted_query = bool(
            recovered.margin_at_position[position] >= confidence_cut
            and recovered.margin_at_position[neighbour_position] >= confidence_cut
        )
        labelled.append(
            replace(
                row,
                exact_candidate=int(match[0]) if len(match) else -1,
                trusted_query=trusted_query,
                candidate_mapping_margin=recovered.margin_at_position[candidate_positions].astype(
                    np.float32
                ),
                mapping_confidence_cut=confidence_cut,
                teacher_scores=teacher_scores,
            )
        )
    return replace(board, rows=tuple(labelled))


def prepare_tile_channels(tiles: np.ndarray, *, view_mode: str) -> torch.Tensor:
    """Build mandatory raw channels and an optional guarded denoised view."""

    source = np.asarray(tiles, dtype=np.uint8)
    if source.ndim != 4 or source.shape[1:] != (20, 20, 3):
        raise ValueError(f"expected N x 20 x 20 x 3 tiles, got {source.shape}")
    if view_mode not in {"raw", "dual"}:
        raise ValueError(f"view_mode must be 'raw' or 'dual', got {view_mode!r}")

    raw = torch.from_numpy(source.astype(np.float32)).permute(0, 3, 1, 2) / 127.5 - 1.0
    mean = raw.mean(dim=(1, 2, 3), keepdim=True)
    std = raw.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    raw_local = ((raw - mean) / std).clamp(-4.0, 4.0) / 4.0
    channels = [raw, raw_local]
    if view_mode == "dual":
        denoised_np = np.stack(
            [cv2.bilateralFilter(tile, 5, 25, 5) for tile in source], axis=0
        ).astype(np.float32)
        denoised = torch.from_numpy(denoised_np).permute(0, 3, 1, 2) / 127.5 - 1.0
        denoised_mean = denoised.mean(dim=(1, 2, 3), keepdim=True)
        denoised_std = denoised.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
        denoised_local = ((denoised - denoised_mean) / denoised_std).clamp(-4.0, 4.0) / 4.0
        residual = (raw - denoised).clamp(-0.5, 0.5) * 2.0
        channels.extend((denoised, denoised_local, residual))
    return torch.cat(channels, dim=1).contiguous()


class _JointEncoder(nn.Module):
    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).flatten(1)


class PairwiseEdgeRanker(nn.Module):
    """Two-scale joint CNN that predicts a residual over bilateral log-scores."""

    def __init__(
        self,
        *,
        feature_dim: int,
        view_mode: str = "dual",
        width: int = 24,
        hidden: int = 48,
    ) -> None:
        super().__init__()
        if view_mode not in {"raw", "dual"}:
            raise ValueError(f"unknown view mode {view_mode!r}")
        channels = 6 if view_mode == "raw" else 15
        self.feature_dim = int(feature_dim)
        self.view_mode = view_mode
        self.width = int(width)
        self.hidden = int(hidden)
        self.seam_encoder = _JointEncoder(channels, width)
        self.context_encoder = _JointEncoder(channels, width)
        self.head = nn.Sequential(
            nn.Linear(2 * width + feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Exact baseline identity at initialisation is a safety property: a
        # failed optimiser cannot silently invent a worse candidate ordering.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        tile_channels: torch.Tensor,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        directions: torch.Tensor,
        features: torch.Tensor,
        baseline_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = tile_channels[anchors]
        candidate = tile_channels[candidates]
        down = directions.to(torch.bool)[:, None, None, None]
        anchor = torch.where(down, anchor.transpose(-2, -1), anchor)
        candidate = torch.where(down, candidate.transpose(-2, -1), candidate)
        seam = torch.cat((anchor[..., -8:], candidate[..., :8]), dim=-1)
        context = F.avg_pool2d(torch.cat((anchor, candidate), dim=-1), kernel_size=2)
        joint = torch.cat((self.seam_encoder(seam), self.context_encoder(context), features), dim=1)
        delta = self.head(joint).squeeze(1)
        return baseline_scores + delta, delta


def pack_rows(
    rows: Sequence[EdgeRow],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pack variable candidate lists for one listwise optimisation step."""

    if not rows:
        raise ValueError("cannot pack an empty row list")
    lengths = [len(row.candidates) for row in rows]
    max_length = max(lengths)
    valid = torch.zeros((len(rows), max_length), dtype=torch.bool, device=device)
    logits_index = torch.full((len(rows), max_length), -1, dtype=torch.long, device=device)
    targets = torch.as_tensor(
        [row.exact_candidate for row in rows], dtype=torch.long, device=device
    )
    anchors: list[int] = []
    candidates: list[int] = []
    directions: list[int] = []
    features: list[np.ndarray] = []
    baseline: list[np.ndarray] = []
    teacher = torch.zeros((len(rows), max_length), dtype=torch.float32, device=device)
    teacher_valid = torch.zeros_like(valid)
    offset = 0
    for row_index, row in enumerate(rows):
        length = len(row.candidates)
        valid[row_index, :length] = True
        logits_index[row_index, :length] = torch.arange(offset, offset + length, device=device)
        anchors.extend([row.anchor] * length)
        candidates.extend(row.candidates.tolist())
        directions.extend([row.direction] * length)
        features.append(row.features)
        baseline.append(row.baseline_scores)
        if row.teacher_scores is not None and row.candidate_mapping_margin is not None:
            teacher[row_index, :length] = torch.from_numpy(row.teacher_scores).to(device)
            trusted = row.candidate_mapping_margin >= row.mapping_confidence_cut
            teacher_valid[row_index, :length] = torch.from_numpy(trusted).to(device)
        offset += length
    return {
        "anchors": torch.as_tensor(anchors, dtype=torch.long, device=device),
        "candidates": torch.as_tensor(candidates, dtype=torch.long, device=device),
        "directions": torch.as_tensor(directions, dtype=torch.long, device=device),
        "features": torch.from_numpy(np.concatenate(features)).to(device),
        "baseline": torch.from_numpy(np.concatenate(baseline)).to(device),
        "valid": valid,
        "logits_index": logits_index,
        "targets": targets,
        "teacher": teacher,
        "teacher_valid": teacher_valid,
    }


def unpack_logits(flat: torch.Tensor, packed: dict[str, torch.Tensor]) -> torch.Tensor:
    """Restore padded row logits from a flat pair-score tensor."""

    index = packed["logits_index"]
    safe = index.clamp_min(0)
    result = flat[safe]
    return result.masked_fill(~packed["valid"], -1e4)


def edge_listwise_loss(
    logits: torch.Tensor,
    packed: dict[str, torch.Tensor],
    *,
    teacher_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact-neighbour CE plus trusted clean-continuity listwise distillation."""

    targets = packed["targets"]
    if torch.any(targets < 0):
        raise ValueError("training rows must contain their exact candidate")
    exact = F.cross_entropy(logits, targets)
    teacher_terms: list[torch.Tensor] = []
    for row in range(len(logits)):
        mask = packed["teacher_valid"][row] & packed["valid"][row]
        if int(mask.sum()) < 2:
            continue
        teacher_probability = F.softmax(packed["teacher"][row, mask], dim=0)
        teacher_terms.append(-(teacher_probability * F.log_softmax(logits[row, mask], dim=0)).sum())
    teacher = torch.stack(teacher_terms).mean() if teacher_terms else exact.new_zeros(())
    loss = exact + teacher_weight * teacher
    return loss, {
        "exact_ce": float(exact.detach()),
        "teacher_ce": float(teacher.detach()),
        "total": float(loss.detach()),
    }


@torch.no_grad()
def score_board(
    model: PairwiseEdgeRanker,
    board: EdgeBoard,
    *,
    device: torch.device,
    pair_batch: int = 2048,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Rerank shortlist entries while preserving full bilateral matrices."""

    if pair_batch <= 0:
        raise ValueError("pair_batch must be positive")
    model.eval()
    channels = prepare_tile_channels(board.tiles, view_mode=model.view_mode).to(device)
    right = board.right_baseline.copy()
    down = board.down_baseline.copy()
    anchors = np.concatenate(
        [np.full(len(row.candidates), row.anchor, dtype=np.int64) for row in board.rows]
    )
    candidates = np.concatenate([row.candidates for row in board.rows])
    directions = np.concatenate(
        [np.full(len(row.candidates), row.direction, dtype=np.int64) for row in board.rows]
    )
    features = np.concatenate([row.features for row in board.rows])
    baseline = np.concatenate([row.baseline_scores for row in board.rows])
    score_values: list[np.ndarray] = []
    delta_values: list[np.ndarray] = []
    for start in range(0, len(candidates), pair_batch):
        stop = min(start + pair_batch, len(candidates))
        predicted, delta = model(
            channels,
            torch.from_numpy(anchors[start:stop]).to(device),
            torch.from_numpy(candidates[start:stop]).to(device),
            torch.from_numpy(directions[start:stop]).to(device),
            torch.from_numpy(features[start:stop]).to(device),
            torch.from_numpy(baseline[start:stop]).to(device),
        )
        score_values.append(predicted.cpu().numpy())
        delta_values.append(delta.cpu().numpy())
    scores_flat = np.concatenate(score_values)
    delta_flat = np.concatenate(delta_values)
    offset = 0
    for row in board.rows:
        length = len(row.candidates)
        matrix = right if row.direction == 0 else down
        matrix[row.anchor, row.candidates] = scores_flat[offset : offset + length]
        offset += length
    return (
        right,
        down,
        {
            "delta_mean": float(delta_flat.mean()),
            "delta_std": float(delta_flat.std()),
            "delta_abs_max": float(np.abs(delta_flat).max()),
        },
    )


def exact_edge_counts(
    board: EdgeBoard,
    right: np.ndarray,
    down: np.ndarray,
    *,
    k_values: Sequence[int] = (1, 5),
) -> list[dict[str, int | str]]:
    """Return sufficient exact-recall counts by scope and direction."""

    if any(k < 1 for k in k_values):
        raise ValueError("k values must be positive")
    output: list[dict[str, int | str]] = []
    for scope in ("all", "trusted_query"):
        for direction, name in ((0, "right"), (1, "down")):
            rows = [
                row
                for row in board.rows
                if row.direction == direction and (scope == "all" or row.trusted_query)
            ]
            for k in k_values:
                hits = 0
                matrix = right if direction == 0 else down
                for row in rows:
                    if row.exact_candidate < 0:
                        continue
                    truth = int(row.candidates[row.exact_candidate])
                    top = np.argpartition(matrix[row.anchor], -k)[-k:]
                    hits += int(truth in top)
                output.append(
                    {
                        "scope": scope,
                        "direction": name,
                        "k": int(k),
                        "edges": len(rows),
                        "hits": hits,
                    }
                )
    return output
