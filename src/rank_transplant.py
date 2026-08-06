"""Confidence-gated reciprocal rank transplantation.

The donor is allowed to change *which* candidate owns an existing base logit,
but it is never allowed to invent a score or change a row's finite values.  A
trusted physical relation is therefore applied as two raw-logit swaps (forward
and inverse direction) before the ordinary ``dense_rd`` conversion.

All arrays in this module are NumPy arrays.  Directional scores use the shared
``(U, D, L, R)`` convention and have shape ``(4, N, K)``; candidate ids have
shape ``(N, K)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
NUM_DIRECTIONS = 4
INVERSE_DIRECTION: tuple[int, ...] = (DOWN, UP, RIGHT, LEFT)


@dataclass(frozen=True)
class ReciprocalPair:
    """One donor relation, emitted once for an unordered physical tile pair."""

    anchor: int
    direction: int
    target: int
    reverse_direction: int
    confidence: float
    forward_margin: float
    reverse_margin: float
    changed_rows: int

    @property
    def rows(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (self.direction, self.anchor),
            (self.reverse_direction, self.target),
        )


@dataclass(frozen=True)
class RankTransplantResult:
    """Transplanted scores plus deterministic selection provenance."""

    scores: np.ndarray
    eligible_pairs: tuple[ReciprocalPair, ...]
    selected_pairs: tuple[ReciprocalPair, ...]
    swapped_rows: tuple[tuple[int, int], ...]

    @property
    def changed_row_count(self) -> int:
        return len(self.swapped_rows)


def _as_candidates(candidates: np.ndarray) -> np.ndarray:
    values = np.asarray(candidates)
    if values.ndim != 2:
        raise ValueError(f"candidates must have shape (N,K), got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("candidates must contain integer tile ids")
    count = values.shape[0]
    if values.shape[1] < 2:
        raise ValueError("candidate rows need at least two slots")
    if np.any(values < 0) or np.any(values >= count):
        raise ValueError(f"candidate ids must lie in [0,{count})")
    return values.astype(np.int64, copy=False)


def _as_scores(scores: np.ndarray, candidates: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(scores)
    expected = (NUM_DIRECTIONS, *candidates.shape)
    if values.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"{name} must be floating point")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError(f"{name} may contain finite values and -inf only")
    if not np.isfinite(values).any(axis=-1).all():
        raise ValueError(f"every {name} row needs at least one finite candidate")
    return values


def validate_candidate_rows(candidates: np.ndarray, scores: np.ndarray) -> None:
    """Validate ids, score shape, and uniqueness of each finite candidate list."""
    candidate_ids = _as_candidates(candidates)
    values = _as_scores(scores, candidate_ids, name="scores")
    finite_any = np.isfinite(values).any(axis=0)
    for anchor in range(candidate_ids.shape[0]):
        ids = candidate_ids[anchor, finite_any[anchor]]
        if len(ids) != len(np.unique(ids)):
            raise ValueError(f"finite candidate ids are duplicated for anchor {anchor}")


def row_zscore(scores: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Standardize each directional candidate row without changing its mask."""
    values = np.asarray(scores)
    if values.ndim != 3:
        raise ValueError("scores must have shape (4,N,K)")
    mask = np.isfinite(values) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("valid mask must align with scores")
    if np.any(mask & ~np.isfinite(values)):
        raise ValueError("valid entries must have finite scores")
    counts = mask.sum(axis=-1, keepdims=True)
    if np.any(counts == 0):
        raise ValueError("every row needs at least one valid score")

    work = values.astype(np.float64, copy=False)
    total = np.where(mask, work, 0.0).sum(axis=-1, keepdims=True)
    mean = total / counts
    centered = np.where(mask, work - mean, 0.0)
    variance = np.square(centered).sum(axis=-1, keepdims=True) / counts
    scale = np.maximum(np.sqrt(variance), 1.0e-4)
    out = np.full(values.shape, -np.inf, dtype=np.float32)
    standardized = (work - mean) / scale
    out[mask] = standardized[mask].astype(np.float32)
    return out


def fused_donor_scores(
    base_scores: np.ndarray,
    spatial_scores: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Return comparable row-z donor scores from base + spatial evidence.

    A final standardization does not affect the fused rank.  It only makes the
    top-two margins used for cross-row confidence comparable.
    """
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    base = np.asarray(base_scores)
    spatial = np.asarray(spatial_scores)
    if spatial.shape != base.shape:
        raise ValueError("spatial_scores must align with base_scores")
    valid = np.isfinite(base)
    if np.any(valid & ~np.isfinite(spatial)):
        raise ValueError("spatial scores must be finite at every base-valid slot")
    base_z = row_zscore(base, valid)
    spatial_z = row_zscore(spatial, valid)
    fused = np.full(base.shape, -np.inf, dtype=np.float32)
    fused[valid] = base_z[valid] + np.float32(alpha) * spatial_z[valid]
    return row_zscore(fused, valid)


def row_predictions(
    candidates: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return best candidate id, best slot, and standardized top-two margin."""
    candidate_ids = _as_candidates(candidates)
    values = _as_scores(scores, candidate_ids, name="scores")
    safe = np.where(np.isfinite(values), values, -np.inf)
    best_slot = safe.argmax(axis=-1)
    best_score = np.take_along_axis(safe, best_slot[..., None], axis=-1)[..., 0]
    if safe.shape[-1] < 2:
        raise ValueError("top-two margins require at least two candidate slots")
    second = np.partition(safe, kth=safe.shape[-1] - 2, axis=-1)[..., -2]
    margin = best_score - second
    best_target = np.take_along_axis(
        np.broadcast_to(candidate_ids[None], safe.shape),
        best_slot[..., None],
        axis=-1,
    )[..., 0]
    return best_target.astype(np.int64), best_slot.astype(np.int64), margin.astype(np.float64)


def reciprocal_physical_pairs(
    candidates: np.ndarray,
    donor_scores: np.ndarray,
    *,
    base_scores: np.ndarray | None = None,
    require_changed: bool = True,
) -> tuple[ReciprocalPair, ...]:
    """Extract deterministic reciprocal donor relations with pair confidence.

    The confidence of a physical pair is the weaker of its forward and reverse
    row margins.  If one unordered tile pair is proposed in multiple physical
    orientations, only its highest-confidence orientation survives.
    """
    candidate_ids = _as_candidates(candidates)
    donor = _as_scores(donor_scores, candidate_ids, name="donor_scores")
    donor_target, _, donor_margin = row_predictions(candidate_ids, donor)
    base_target: np.ndarray | None = None
    if base_scores is not None:
        base = _as_scores(base_scores, candidate_ids, name="base_scores")
        base_target, _, _ = row_predictions(candidate_ids, base)
    elif require_changed:
        raise ValueError("base_scores are required when require_changed=True")

    count = candidate_ids.shape[0]
    by_tiles: dict[tuple[int, int], ReciprocalPair] = {}
    for anchor in range(count):
        for direction in range(NUM_DIRECTIONS):
            target = int(donor_target[direction, anchor])
            if target <= anchor or target == anchor:
                continue
            reverse_direction = INVERSE_DIRECTION[direction]
            if int(donor_target[reverse_direction, target]) != anchor:
                continue
            forward_margin = float(donor_margin[direction, anchor])
            reverse_margin = float(donor_margin[reverse_direction, target])
            confidence = min(forward_margin, reverse_margin)
            if not np.isfinite(confidence):
                continue
            changed_rows = 0
            if base_target is not None:
                changed_rows = int(int(base_target[direction, anchor]) != target)
                changed_rows += int(int(base_target[reverse_direction, target]) != anchor)
            if require_changed and changed_rows == 0:
                continue
            relation = ReciprocalPair(
                anchor=anchor,
                direction=direction,
                target=target,
                reverse_direction=reverse_direction,
                confidence=confidence,
                forward_margin=forward_margin,
                reverse_margin=reverse_margin,
                changed_rows=changed_rows,
            )
            key = (anchor, target)
            current = by_tiles.get(key)
            if current is None or relation.confidence > current.confidence:
                by_tiles[key] = relation
            elif relation.confidence == current.confidence:
                if (relation.direction, relation.anchor, relation.target) < (
                    current.direction,
                    current.anchor,
                    current.target,
                ):
                    by_tiles[key] = relation
    return tuple(
        sorted(
            by_tiles.values(),
            key=lambda item: (-item.confidence, item.anchor, item.direction, item.target),
        )
    )


def select_trusted_pairs(
    pairs: Iterable[ReciprocalPair],
    *,
    top_m: int,
    min_confidence: float | None = None,
) -> tuple[ReciprocalPair, ...]:
    """Select the top-M reciprocal physical pairs with deterministic ties."""
    if top_m < 0:
        raise ValueError("top_m must be non-negative")
    threshold = -np.inf if min_confidence is None else float(min_confidence)
    if np.isnan(threshold):
        raise ValueError("min_confidence may not be NaN")
    ordered = sorted(
        (pair for pair in pairs if pair.confidence >= threshold),
        key=lambda item: (-item.confidence, item.anchor, item.direction, item.target),
    )
    selected = tuple(ordered[:top_m])
    used_rows: set[tuple[int, int]] = set()
    for pair in selected:
        for row in pair.rows:
            if row in used_rows:
                raise ValueError(f"a directed donor row was selected twice: {row}")
            used_rows.add(row)
    return selected


def _unique_candidate_slot(
    candidates: np.ndarray,
    scores: np.ndarray,
    *,
    anchor: int,
    direction: int,
    target: int,
) -> int:
    match = (candidates[anchor] == target) & np.isfinite(scores[direction, anchor])
    slots = np.flatnonzero(match)
    if len(slots) != 1:
        raise ValueError(
            f"expected one finite slot for row {(direction, anchor)} target {target}, got {len(slots)}"
        )
    return int(slots[0])


def transplant_raw_logits(
    candidates: np.ndarray,
    base_scores: np.ndarray,
    selected_pairs: Sequence[ReciprocalPair],
    *,
    verify: bool = True,
) -> RankTransplantResult:
    """Swap base top logits into donor slots for both sides of each pair."""
    candidate_ids = _as_candidates(candidates)
    base = _as_scores(base_scores, candidate_ids, name="base_scores")
    validate_candidate_rows(candidate_ids, base)
    original = np.array(base, copy=True, order="C")
    transplanted = np.array(base, copy=True, order="C")
    selected = tuple(selected_pairs)
    used_rows: set[tuple[int, int]] = set()
    swapped_rows: list[tuple[int, int]] = []

    relations: list[tuple[int, int, int]] = []
    for pair in selected:
        if pair.reverse_direction != INVERSE_DIRECTION[pair.direction]:
            raise ValueError("pair reverse direction is inconsistent")
        relations.extend(
            (
                (pair.direction, pair.anchor, pair.target),
                (pair.reverse_direction, pair.target, pair.anchor),
            )
        )

    for direction, anchor, donor_target in relations:
        row_key = (int(direction), int(anchor))
        if row_key in used_rows:
            raise ValueError(f"a directed row was selected twice: {row_key}")
        used_rows.add(row_key)
        row = original[direction, anchor]
        finite = np.isfinite(row)
        maximum = row[finite].max()
        top_slots = np.flatnonzero(finite & (row == maximum))
        if len(top_slots) != 1:
            raise ValueError(f"base row {row_key} does not have a unique top logit")
        base_slot = int(top_slots[0])
        donor_slot = _unique_candidate_slot(
            candidate_ids,
            original,
            anchor=anchor,
            direction=direction,
            target=donor_target,
        )
        if donor_slot == base_slot:
            continue
        value = transplanted[direction, anchor, base_slot].copy()
        transplanted[direction, anchor, base_slot] = transplanted[direction, anchor, donor_slot]
        transplanted[direction, anchor, donor_slot] = value
        swapped_rows.append(row_key)

    swapped_rows_tuple = tuple(sorted(swapped_rows))
    if verify:
        if not np.array_equal(np.isfinite(original), np.isfinite(transplanted)):
            raise AssertionError("rank transplant changed the finite score mask")
        if not np.array_equal(np.sort(original, axis=-1), np.sort(transplanted, axis=-1)):
            raise AssertionError("rank transplant changed a row value multiset")
        selected_mask = np.zeros(original.shape[:2], dtype=bool)
        for direction, anchor in used_rows:
            selected_mask[direction, anchor] = True
        if not np.array_equal(original[~selected_mask], transplanted[~selected_mask]):
            raise AssertionError("rank transplant changed an unselected row")
        for direction, anchor, donor_target in relations:
            donor_slot = _unique_candidate_slot(
                candidate_ids,
                transplanted,
                anchor=anchor,
                direction=direction,
                target=donor_target,
            )
            if transplanted[direction, anchor, donor_slot] != transplanted[direction, anchor].max():
                raise AssertionError("donor candidate did not receive the base top logit")

    return RankTransplantResult(
        scores=transplanted,
        eligible_pairs=selected,
        selected_pairs=selected,
        swapped_rows=swapped_rows_tuple,
    )


def confidence_gated_rank_transplant(
    candidates: np.ndarray,
    base_scores: np.ndarray,
    donor_scores: np.ndarray,
    *,
    top_m: int,
    min_confidence: float | None = None,
    verify: bool = True,
) -> RankTransplantResult:
    """Discover reciprocal donor pairs, select top-M, and transplant ranks."""
    eligible = reciprocal_physical_pairs(
        candidates,
        donor_scores,
        base_scores=base_scores,
        require_changed=True,
    )
    selected = select_trusted_pairs(
        eligible,
        top_m=top_m,
        min_confidence=min_confidence,
    )
    applied = transplant_raw_logits(candidates, base_scores, selected, verify=verify)
    return RankTransplantResult(
        scores=applied.scores,
        eligible_pairs=eligible,
        selected_pairs=selected,
        swapped_rows=applied.swapped_rows,
    )


def assert_disjoint_phases(
    calibration_images: Sequence[int],
    confirmation_images: Sequence[int],
) -> None:
    """Prevent confirmation from silently reusing any calibration image."""
    calibration = {int(value) for value in calibration_images}
    confirmation = {int(value) for value in confirmation_images}
    if not calibration or not confirmation:
        raise ValueError("calibration and confirmation image lists must both be non-empty")
    overlap = sorted(calibration & confirmation)
    if overlap:
        raise ValueError(f"calibration/confirmation image overlap: {overlap}")


__all__ = (
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "NUM_DIRECTIONS",
    "INVERSE_DIRECTION",
    "ReciprocalPair",
    "RankTransplantResult",
    "validate_candidate_rows",
    "row_zscore",
    "fused_donor_scores",
    "row_predictions",
    "reciprocal_physical_pairs",
    "select_trusted_pairs",
    "transplant_raw_logits",
    "confidence_gated_rank_transplant",
    "assert_disjoint_phases",
)
