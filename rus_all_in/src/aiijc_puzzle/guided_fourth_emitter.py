"""Fail-closed fourth-emitter sidecar for the immutable tri-emitter roster.

The legacy raw/adapter1600/DINO candidate pool is an input artifact and is
never rebuilt or mutated here.  A fixed guided-filter score supplies at most
``TOP_K`` additional identities in a separate slot range.  This keeps the
signed tri-emitter protocol byte-stable while making a future fourth-emitter
model possible.

Only matcher-visible arrays are accepted.  Exact references, recovered
positions and target slots are deliberately absent from this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.guided_matcher_view import (
    FIXED_CONFIG,
    guided_fused_directional_scores,
)
from aiijc_puzzle.legacy_upgrade import directional_scores
from aiijc_puzzle.tri_emitter_edge_verifier import (
    AUXILIARY_DIM,
    EMITTERS,
    TOP_K,
    CandidatePool,
    candidate_pool_digest,
)

GUIDED_EMITTER = "guided_standalone_r2_eps1600"
FOURTH_EMITTERS = (*EMITTERS, GUIDED_EMITTER)
GUIDED_AUXILIARY_DIM = 7


@dataclass(frozen=True)
class GuidedFourthEmitterPool:
    """A target-free fourth-emitter sidecar over one immutable legacy pool."""

    candidates: np.ndarray
    valid: np.ndarray
    legacy_slot: np.ndarray
    legacy_auxiliary: np.ndarray
    legacy_raw_baseline: np.ndarray
    guided_auxiliary: np.ndarray
    guided_baseline: np.ndarray
    emitter_topk: np.ndarray
    legacy_identity_digest: str
    identity_digest: str


def fixed_guided_standalone_scores(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replay the frozen guided view as a standalone fourth score emitter.

    The preregistered recipe fuses bilateral and guided score matrices with a
    fixed weight of one half.  Therefore ``2 * fused - bilateral`` recovers the
    exact standalone guided matrix already used by the frozen supply audit.
    No parameter is exposed here.
    """

    if (
        FIXED_CONFIG.radius != 2
        or FIXED_CONFIG.epsilon != 1600.0
        or FIXED_CONFIG.guided_weight != 0.5
    ):
        raise RuntimeError("the frozen guided recipe changed")
    source = np.asarray(tiles)
    if source.shape != (576, 20, 20, 3) or source.dtype != np.uint8:
        raise ValueError("tiles must be uint8 with shape 576 x 20 x 20 x 3")
    bilateral = directional_scores(source, views=("bilateral",))["bilateral"]
    fused = guided_fused_directional_scores(source, bilateral)
    standalone = tuple(
        np.ascontiguousarray(2.0 * candidate - control, dtype=np.float32)
        for control, candidate in zip(bilateral, fused, strict=True)
    )
    if any(not np.isfinite(value).all() for value in standalone):
        raise RuntimeError("guided standalone score matrix is non-finite")
    return standalone


def _validate_guided_scores(
    guided_scores: tuple[Any, Any],
    *,
    count: int,
) -> np.ndarray:
    if not isinstance(guided_scores, tuple) or len(guided_scores) != 2:
        raise ValueError("guided_scores must be a right/down tuple")
    axes = [np.asarray(axis, dtype=np.float32) for axis in guided_scores]
    if any(axis.shape != (count, count) for axis in axes) or any(
        not np.isfinite(axis).all() for axis in axes
    ):
        raise ValueError("guided score matrices must be aligned, square and finite")
    values = np.stack(axes, axis=0)
    return np.ascontiguousarray(values)


def _validate_legacy_pool(pool: CandidatePool) -> tuple[int, int, int]:
    candidates = np.asarray(pool.candidates)
    valid = np.asarray(pool.valid)
    if candidates.ndim != 3 or candidates.shape[:1] != (2,):
        raise ValueError("legacy candidates must have shape 2 x N x K")
    count, legacy_width = candidates.shape[1:]
    if count <= TOP_K or legacy_width != len(EMITTERS) * TOP_K:
        raise ValueError("legacy pool is not the fixed tri-emitter top32 roster")
    if candidates.dtype not in (np.int32, np.int64) or valid.dtype != np.bool_:
        raise ValueError("legacy candidates/valid dtypes changed")
    if valid.shape != candidates.shape:
        raise ValueError("legacy candidates and valid mask are not aligned")
    if pool.auxiliary.shape != (*candidates.shape, AUXILIARY_DIM):
        raise ValueError("legacy auxiliary shape changed")
    if pool.raw_baseline.shape != candidates.shape:
        raise ValueError("legacy raw baseline shape changed")
    if pool.emitter_topk.shape != (len(EMITTERS), 2, count, TOP_K):
        raise ValueError("legacy emitter top-k shape changed")
    for name, value in (
        ("legacy auxiliary", pool.auxiliary),
        ("legacy raw baseline", pool.raw_baseline),
    ):
        if not np.issubdtype(np.asarray(value).dtype, np.floating):
            raise ValueError(f"{name} must be floating point")
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"{name} must be finite")
    observed_digest = candidate_pool_digest(candidates, valid, pool.emitter_topk)
    if pool.identity_digest != observed_digest:
        raise ValueError("legacy pool identity digest mismatch")
    for axis in range(2):
        for source in range(count):
            row_valid = valid[axis, source]
            if np.any(row_valid[1:] & ~row_valid[:-1]):
                raise ValueError("legacy valid candidates are not prefix-packed")
            row = candidates[axis, source, row_valid]
            if len(row) != len(np.unique(row)):
                raise ValueError("legacy candidate row contains duplicates")
            if np.any((row < 0) | (row >= count)) or source in row:
                raise ValueError("legacy candidate identity is invalid")
            raw = set(pool.emitter_topk[0, axis, source].tolist())
            if not raw.issubset(set(row.tolist())):
                raise ValueError("legacy pool no longer retains raw top32")
    return count, legacy_width, TOP_K


def _stable_orders(values: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(values)
    work = np.asarray(values, dtype=np.float32).copy()
    np.fill_diagonal(work, -np.inf)
    outgoing = np.argsort(-work, axis=1, kind="stable")[:, :top_k]
    incoming = np.argsort(-work, axis=0, kind="stable")[:top_k]
    outgoing_rank = np.full((count, count), top_k, dtype=np.int16)
    incoming_rank = np.full((count, count), top_k, dtype=np.int16)
    outgoing_rank[np.arange(count)[:, None], outgoing] = np.arange(
        top_k, dtype=np.int16
    )[None]
    incoming_rank[incoming, np.arange(count)[None]] = np.arange(
        top_k, dtype=np.int16
    )[:, None]
    return outgoing_rank, incoming_rank


def _score_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(values)
    mask = ~np.eye(count, dtype=bool)
    row = np.where(mask, values, 0.0)
    row_mean = row.sum(axis=1) / (count - 1)
    row_std = np.sqrt(
        np.where(mask, (values - row_mean[:, None]) ** 2, 0.0).sum(axis=1)
        / (count - 1)
        + 1e-6
    )
    column_mean = row.sum(axis=0) / (count - 1)
    column_std = np.sqrt(
        np.where(mask, (values - column_mean[None]) ** 2, 0.0).sum(axis=0)
        / (count - 1)
        + 1e-6
    )
    work = values.copy()
    np.fill_diagonal(work, -np.inf)
    return row_mean, row_std, np.stack((column_mean, column_std, work.max(axis=1)))


def guided_fourth_pool_digest(
    candidates: np.ndarray,
    valid: np.ndarray,
    legacy_slot: np.ndarray,
    emitter_topk: np.ndarray,
) -> str:
    """Hash only target-free identities, membership and legacy slot mapping."""

    digest = hashlib.sha256()
    for value in (candidates, valid, legacy_slot, emitter_topk):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def extend_with_guided_emitter(
    legacy: CandidatePool,
    guided_scores: tuple[Any, Any],
) -> GuidedFourthEmitterPool:
    """Append fixed guided top32 identities without changing legacy slots."""

    count, legacy_width, top_k = _validate_legacy_pool(legacy)
    matrices = _validate_guided_scores(guided_scores, count=count)
    width = legacy_width + top_k
    candidates = np.full((2, count, width), -1, dtype=np.int32)
    valid = np.zeros_like(candidates, dtype=bool)
    legacy_slot = np.full_like(candidates, -1, dtype=np.int16)
    legacy_auxiliary = np.zeros(
        (*candidates.shape, AUXILIARY_DIM), dtype=np.float32
    )
    legacy_raw_baseline = np.full(candidates.shape, -1e4, dtype=np.float32)
    guided_auxiliary = np.zeros(
        (*candidates.shape, GUIDED_AUXILIARY_DIM), dtype=np.float32
    )
    guided_baseline = np.full(candidates.shape, -1e4, dtype=np.float32)
    emitter_topk = np.empty((len(FOURTH_EMITTERS), 2, count, top_k), dtype=np.int32)

    candidates[..., :legacy_width] = legacy.candidates
    valid[..., :legacy_width] = legacy.valid
    legacy_auxiliary[..., :legacy_width, :] = legacy.auxiliary
    legacy_raw_baseline[..., :legacy_width] = legacy.raw_baseline
    slot_ids = np.arange(legacy_width, dtype=np.int16)
    legacy_slot[..., :legacy_width] = np.where(legacy.valid, slot_ids, -1)
    emitter_topk[: len(EMITTERS)] = legacy.emitter_topk

    for axis in range(2):
        matrix = matrices[axis]
        outgoing_rank, incoming_rank = _stable_orders(matrix, top_k)
        row_mean, row_std, packed = _score_statistics(matrix)
        column_mean, column_std, row_top = packed
        guided_topk = np.argsort(
            -np.where(np.eye(count, dtype=bool), -np.inf, matrix),
            axis=1,
            kind="stable",
        )[:, :top_k]
        emitter_topk[-1, axis] = guided_topk
        for source in range(count):
            old = set(
                candidates[axis, source, :legacy_width][
                    valid[axis, source, :legacy_width]
                ].tolist()
            )
            novel = [int(target) for target in guided_topk[source] if int(target) not in old]
            new_slice = slice(legacy_width, legacy_width + len(novel))
            candidates[axis, source, new_slice] = novel
            valid[axis, source, new_slice] = True
            for slot in np.flatnonzero(valid[axis, source]):
                target = int(candidates[axis, source, slot])
                score = float(matrix[source, target])
                row_scale = float(row_std[source])
                column_scale = float(column_std[target])
                guided_auxiliary[axis, source, slot] = (
                    float(outgoing_rank[source, target] < top_k),
                    float(outgoing_rank[source, target]) / top_k,
                    float(incoming_rank[source, target]) / top_k,
                    float(np.clip((score - row_mean[source]) / row_scale, -8, 8)),
                    float(np.clip((score - column_mean[target]) / column_scale, -8, 8)),
                    float(np.clip((score - row_top[source]) / row_scale, -8, 0)),
                    float(slot < legacy_width),
                )
                guided_baseline[axis, source, slot] = guided_auxiliary[
                    axis, source, slot, 3
                ]

    if not np.array_equal(candidates[..., :legacy_width], legacy.candidates):
        raise RuntimeError("legacy candidate slots changed")
    if not np.array_equal(valid[..., :legacy_width], legacy.valid):
        raise RuntimeError("legacy validity slots changed")
    if not np.array_equal(emitter_topk[: len(EMITTERS)], legacy.emitter_topk):
        raise RuntimeError("legacy emitter top-k arrays changed")
    for axis in range(2):
        for source in range(count):
            row = candidates[axis, source, valid[axis, source]]
            if len(row) != len(np.unique(row)):
                raise RuntimeError("extended candidate row contains duplicates")
            raw = set(emitter_topk[0, axis, source].tolist())
            if not raw.issubset(set(row.tolist())):
                raise RuntimeError("extended pool dropped a raw candidate")
    identity_digest = guided_fourth_pool_digest(
        candidates, valid, legacy_slot, emitter_topk
    )
    return GuidedFourthEmitterPool(
        candidates=np.ascontiguousarray(candidates),
        valid=np.ascontiguousarray(valid),
        legacy_slot=np.ascontiguousarray(legacy_slot),
        legacy_auxiliary=np.ascontiguousarray(legacy_auxiliary),
        legacy_raw_baseline=np.ascontiguousarray(legacy_raw_baseline),
        guided_auxiliary=np.ascontiguousarray(guided_auxiliary),
        guided_baseline=np.ascontiguousarray(guided_baseline),
        emitter_topk=np.ascontiguousarray(emitter_topk),
        legacy_identity_digest=legacy.identity_digest,
        identity_digest=identity_digest,
    )


def pool_from_target_free_legacy_cache(
    arrays: Mapping[str, np.ndarray],
) -> CandidatePool:
    """Construct a legacy pool while refusing labels and schema drift."""

    expected = {
        "candidates",
        "valid",
        "auxiliary",
        "raw_baseline",
        "emitter_topk",
    }
    if set(arrays) != expected:
        raise ValueError("legacy sidecar inputs must contain only the five target-free pool arrays")
    digest = candidate_pool_digest(
        arrays["candidates"], arrays["valid"], arrays["emitter_topk"]
    )
    pool = CandidatePool(
        candidates=np.ascontiguousarray(arrays["candidates"]),
        valid=np.ascontiguousarray(arrays["valid"]),
        auxiliary=np.ascontiguousarray(arrays["auxiliary"]),
        raw_baseline=np.ascontiguousarray(arrays["raw_baseline"]),
        emitter_topk=np.ascontiguousarray(arrays["emitter_topk"]),
        identity_digest=digest,
    )
    _validate_legacy_pool(pool)
    return pool


__all__ = [
    "FOURTH_EMITTERS",
    "GUIDED_AUXILIARY_DIM",
    "GUIDED_EMITTER",
    "GuidedFourthEmitterPool",
    "extend_with_guided_emitter",
    "fixed_guided_standalone_scores",
    "guided_fourth_pool_digest",
    "pool_from_target_free_legacy_cache",
]
