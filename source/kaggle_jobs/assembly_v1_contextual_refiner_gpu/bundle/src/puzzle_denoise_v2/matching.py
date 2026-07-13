"""High-purity real tile pairing for denoiser supervision.

The train input and target for one source image contain the same 576 tiles, but
the input tiles are independently corrupted and shuffled.  A single Hungarian
assignment is not accurate enough to be treated as ground truth.  This module
therefore uses two differently constructed cost matrices and keeps a pair only
when both Hungarian solutions agree and the assignment is a true mutual
nearest-neighbour cycle in both descriptor spaces.

The gate is intentionally conservative.  Rejected pairs remain usable as
unlabelled data, but they must not silently become restoration targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tiles import TILE


@dataclass(frozen=True)
class MatchingThresholds:
    """Margin/confidence floors used after consensus and mutual-NN.

    A margin is the best alternative cost minus the assigned cost.  It is
    computed independently for the input row and clean-tile column.  Positive
    margins are required in both directions, so tied or globally forced
    Hungarian assignments cannot enter the gold set.  Joint confidence
    normalizes each descriptor margin by its per-image positive median.
    """

    coarse_min_margin: float = 1e-6
    structural_min_margin: float = 1e-6
    # Synthetic-known-permutation calibration on the fixed audit split showed
    # that 0.45 retained roughly 44% of tiles with no observed false pairs in
    # the smoke panel.  Re-run the calibration CLI before changing this floor.
    joint_min_confidence: float = 0.45

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class AssignmentDiagnostics:
    mapping: np.ndarray
    assigned_cost: np.ndarray
    row_margin: np.ndarray
    column_margin: np.ndarray
    mutual_nn_cycle: np.ndarray

    @property
    def min_margin(self) -> np.ndarray:
        return np.minimum(self.row_margin, self.column_margin)


@dataclass(frozen=True)
class MatchResult:
    coarse: AssignmentDiagnostics
    structural: AssignmentDiagnostics
    consensus: np.ndarray
    selected: np.ndarray
    joint_confidence: np.ndarray
    thresholds: MatchingThresholds


def _validate_tile_pair(input_tiles: np.ndarray, clean_tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    input_tiles = np.asarray(input_tiles)
    clean_tiles = np.asarray(clean_tiles)
    expected_tail = (TILE, TILE, 3)
    if input_tiles.ndim != 4 or tuple(input_tiles.shape[1:]) != expected_tail:
        raise ValueError(f"expected input tiles Nx{expected_tail}, got {input_tiles.shape}")
    if clean_tiles.shape != input_tiles.shape:
        raise ValueError(f"clean tiles must match input shape {input_tiles.shape}, got {clean_tiles.shape}")
    if len(input_tiles) == 0:
        raise ValueError("cannot match an empty tile set")
    return input_tiles.astype(np.float32, copy=False), clean_tiles.astype(np.float32, copy=False)


def _pool_tiles(tiles: np.ndarray, bins: int) -> np.ndarray:
    if TILE % bins:
        raise ValueError(f"tile size {TILE} is not divisible by {bins}")
    block = TILE // bins
    return tiles.reshape(-1, bins, block, bins, block, 3).mean(axis=(2, 4))


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = values.reshape(len(values), -1).astype(np.float32, copy=False)
    values = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / (norms + 1e-6)


def _pairwise_mean_squared(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dimensions = left.shape[1]
    left_norm = np.mean(left * left, axis=1)[:, None]
    right_norm = np.mean(right * right, axis=1)[None, :]
    cost = left_norm + right_norm - (2.0 / dimensions) * (left @ right.T)
    return np.maximum(cost, 0.0).astype(np.float32, copy=False)


def _coarse_photometric_features(tiles: np.ndarray) -> np.ndarray:
    """Descriptor A: normalized coarse RGB plus weak absolute colour cues."""
    pooled = _pool_tiles(tiles, bins=5)
    flat = pooled.reshape(len(tiles), -1)
    normalized = flat - flat.mean(axis=1, keepdims=True)
    normalized = normalized / (flat.std(axis=1, keepdims=True) + 1e-6)
    raw = (flat / 255.0) * 0.35
    means = (tiles.mean(axis=(1, 2)) / 255.0) * 0.35
    stds = (tiles.std(axis=(1, 2)) / 255.0) * 0.35
    return np.concatenate([normalized, raw, means, stds], axis=1).astype(np.float32)


def coarse_photometric_cost(input_tiles: np.ndarray, clean_tiles: np.ndarray) -> np.ndarray:
    """Build the robust coarse-photometric squared-distance matrix."""
    input_tiles, clean_tiles = _validate_tile_pair(input_tiles, clean_tiles)
    return _pairwise_mean_squared(
        _coarse_photometric_features(input_tiles),
        _coarse_photometric_features(clean_tiles),
    )


def _structural_views(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descriptor B views: multi-scale colour, luminance, and gradients."""
    rgb_5 = _pool_tiles(tiles, bins=5)
    rgb_10 = _pool_tiles(tiles, bins=10)
    luma_10 = rgb_10 @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    gradient = np.concatenate(
        [
            np.diff(luma_10, axis=2).reshape(len(tiles), -1),
            np.diff(luma_10, axis=1).reshape(len(tiles), -1),
        ],
        axis=1,
    )
    return _unit_rows(rgb_5), _unit_rows(luma_10), _unit_rows(gradient)


def multiscale_structural_cost(input_tiles: np.ndarray, clean_tiles: np.ndarray) -> np.ndarray:
    """Build an independently normalized multi-scale cosine-distance matrix."""
    input_tiles, clean_tiles = _validate_tile_pair(input_tiles, clean_tiles)
    left = _structural_views(input_tiles)
    right = _structural_views(clean_tiles)
    similarity = (
        0.50 * (left[0] @ right[0].T)
        + 0.35 * (left[1] @ right[1].T)
        + 0.15 * (left[2] @ right[2].T)
    )
    return np.clip(1.0 - similarity, 0.0, 2.0).astype(np.float32, copy=False)


def assignment_diagnostics(cost: np.ndarray) -> AssignmentDiagnostics:
    """Solve one-to-one assignment and compute non-circular confidence data."""
    cost = np.asarray(cost, dtype=np.float32)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or cost.shape[0] == 0:
        raise ValueError(f"expected a non-empty square cost matrix, got {cost.shape}")
    if not np.isfinite(cost).all():
        raise ValueError("cost matrix contains non-finite values")

    row_indices, column_indices = linear_sum_assignment(cost)
    mapping = np.empty(len(row_indices), dtype=np.int32)
    mapping[row_indices] = column_indices.astype(np.int32)
    rows = np.arange(len(mapping))
    assigned_cost = cost[rows, mapping]

    alternatives = cost.copy()
    alternatives[rows, mapping] = np.inf
    row_margin = alternatives.min(axis=1) - assigned_cost
    column_margin = alternatives.min(axis=0)[mapping] - assigned_cost

    row_best = cost.argmin(axis=1)
    column_best = cost.argmin(axis=0)
    mutual = (row_best == mapping) & (column_best[mapping] == rows)
    return AssignmentDiagnostics(
        mapping=mapping,
        assigned_cost=assigned_cost.astype(np.float32),
        row_margin=row_margin.astype(np.float32),
        column_margin=column_margin.astype(np.float32),
        mutual_nn_cycle=mutual,
    )


def _positive_median(values: np.ndarray) -> float:
    positive = values[np.isfinite(values) & (values > 0)]
    return float(np.median(positive)) if len(positive) else 1.0


def match_tile_sets(
    input_tiles: np.ndarray,
    clean_tiles: np.ndarray,
    thresholds: MatchingThresholds | None = None,
) -> MatchResult:
    """Match one shuffled image and return only high-purity gate decisions."""
    thresholds = thresholds or MatchingThresholds()
    coarse = assignment_diagnostics(coarse_photometric_cost(input_tiles, clean_tiles))
    structural = assignment_diagnostics(multiscale_structural_cost(input_tiles, clean_tiles))
    consensus = coarse.mapping == structural.mapping

    coarse_margin = coarse.min_margin
    structural_margin = structural.min_margin
    coarse_scale = _positive_median(coarse_margin)
    structural_scale = _positive_median(structural_margin)
    joint_confidence = np.minimum(
        coarse_margin / (coarse_scale + 1e-12),
        structural_margin / (structural_scale + 1e-12),
    ).astype(np.float32)
    selected = (
        consensus
        & coarse.mutual_nn_cycle
        & structural.mutual_nn_cycle
        & (coarse_margin >= thresholds.coarse_min_margin)
        & (structural_margin >= thresholds.structural_min_margin)
        & (joint_confidence >= thresholds.joint_min_confidence)
    )
    return MatchResult(
        coarse=coarse,
        structural=structural,
        consensus=consensus,
        selected=selected,
        joint_confidence=joint_confidence,
        thresholds=thresholds,
    )


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return {f"q{int(level * 100):02d}": float(value) for level, value in zip(levels, np.quantile(values, levels), strict=True)}


def summarize_match(result: MatchResult) -> dict:
    """Compact per-image diagnostics suitable for logs and NPZ metadata."""
    total = len(result.selected)
    both_mutual = result.coarse.mutual_nn_cycle & result.structural.mutual_nn_cycle
    return {
        "tiles": total,
        "consensus": int(result.consensus.sum()),
        "consensus_coverage": float(result.consensus.mean()),
        "both_mutual": int(both_mutual.sum()),
        "both_mutual_coverage": float(both_mutual.mean()),
        "selected": int(result.selected.sum()),
        "selected_coverage": float(result.selected.mean()),
        "selected_coarse_cost": _quantiles(result.coarse.assigned_cost[result.selected]),
        "selected_structural_cost": _quantiles(result.structural.assigned_cost[result.selected]),
        "selected_coarse_margin": _quantiles(result.coarse.min_margin[result.selected]),
        "selected_structural_margin": _quantiles(result.structural.min_margin[result.selected]),
        "selected_joint_confidence": _quantiles(result.joint_confidence[result.selected]),
    }


def _stage(correct: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    selected = int(mask.sum())
    return {
        "selected": selected,
        "coverage": float(mask.mean()),
        "correct": int((correct & mask).sum()),
        "precision": float(correct[mask].mean()) if selected else None,
    }


def calibration_report(
    results: Sequence[MatchResult],
    true_mappings: Sequence[np.ndarray],
) -> dict:
    """Measure gate precision/coverage against synthetic known permutations."""
    if len(results) == 0 or len(results) != len(true_mappings):
        raise ValueError("results and true_mappings must have the same non-zero length")

    coarse_correct_parts = []
    structural_correct_parts = []
    consensus_parts = []
    coarse_mutual_parts = []
    structural_mutual_parts = []
    selected_parts = []
    confidence_parts = []
    for result, truth in zip(results, true_mappings, strict=True):
        truth = np.asarray(truth)
        if truth.shape != result.coarse.mapping.shape:
            raise ValueError(f"truth shape {truth.shape} does not match {result.coarse.mapping.shape}")
        coarse_correct_parts.append(result.coarse.mapping == truth)
        structural_correct_parts.append(result.structural.mapping == truth)
        consensus_parts.append(result.consensus)
        coarse_mutual_parts.append(result.coarse.mutual_nn_cycle)
        structural_mutual_parts.append(result.structural.mutual_nn_cycle)
        selected_parts.append(result.selected)
        confidence_parts.append(result.joint_confidence)

    coarse_correct = np.concatenate(coarse_correct_parts)
    structural_correct = np.concatenate(structural_correct_parts)
    consensus = np.concatenate(consensus_parts)
    coarse_mutual = np.concatenate(coarse_mutual_parts)
    structural_mutual = np.concatenate(structural_mutual_parts)
    selected = np.concatenate(selected_parts)
    confidence = np.concatenate(confidence_parts)
    consensus_correct = coarse_correct & structural_correct
    agreement_and_both_mutual = consensus & coarse_mutual & structural_mutual

    stages = {
        "coarse_hungarian": _stage(coarse_correct, np.ones_like(coarse_correct, dtype=bool)),
        "structural_hungarian": _stage(structural_correct, np.ones_like(structural_correct, dtype=bool)),
        "hungarian_agreement": _stage(consensus_correct, consensus),
        "agreement_and_coarse_mutual": _stage(consensus_correct, consensus & coarse_mutual),
        "agreement_and_both_mutual": _stage(consensus_correct, agreement_and_both_mutual),
        "selected": _stage(consensus_correct, selected),
    }

    confidence_sweep = []
    base_values = confidence[agreement_and_both_mutual]
    for quantile in (0.0, 0.25, 0.5, 0.75, 0.9):
        threshold = float(np.quantile(base_values, quantile)) if len(base_values) else float("inf")
        mask = agreement_and_both_mutual & (confidence >= threshold)
        confidence_sweep.append(
            {
                "base_confidence_quantile": quantile,
                "threshold": threshold,
                **_stage(consensus_correct, mask),
            }
        )

    return {
        "examples": len(results),
        "tiles": int(len(coarse_correct)),
        "thresholds": results[0].thresholds.to_dict(),
        "stages": stages,
        "confidence_sweep": confidence_sweep,
        "selected_confidence": _quantiles(confidence[selected]),
    }
