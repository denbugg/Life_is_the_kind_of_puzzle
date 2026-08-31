"""Commutative 2x2-cycle evidence for horizontal and vertical socket edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.socket_decoder import SocketEdge


@dataclass(frozen=True)
class AxisSocketRankings:
    """Top-ranked real targets and row-conditional log scores for one axis."""

    top_targets: np.ndarray
    rank: np.ndarray
    conditional_log_score: np.ndarray


@dataclass(frozen=True)
class CommutativeCycleSupport:
    """Best and total 2x2 closures supporting one candidate edge."""

    top_k: int
    base_rank: int | None
    support_count: int
    best_total_rank: int | None
    best_total_conditional_log_score: float | None

    @property
    def supported(self) -> bool:
        return self.support_count > 0


def _assignment(value: Any, *, grid: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    expected = (grid * grid + 1, grid * grid + 1)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {result.shape}")
    usable = result.copy()
    usable[-1, -1] = 0.0
    if not np.isfinite(usable).all():
        raise ValueError(f"{name} contains non-finite usable entries")
    return result


def axis_socket_rankings(
    log_assignment: Any,
    *,
    grid: int,
    maximum_k: int,
) -> AxisSocketRankings:
    """Build deterministic row top-K lists, excluding forbidden self-pairs."""

    count = grid * grid
    if not 1 <= maximum_k < count:
        raise ValueError(f"maximum_k must be in [1, {count - 1}]")
    value = _assignment(log_assignment, grid=grid, name="log_assignment")
    real = value[:count, :count].copy()
    np.fill_diagonal(real, -np.inf)
    order = np.argsort(-real, axis=1, kind="stable")[:, :maximum_k]
    rank = np.full((count, count), -1, dtype=np.int16)
    rank[np.arange(count)[:, None], order] = np.arange(1, maximum_k + 1)

    rows = value[:count]
    maximum = rows.max(axis=1, keepdims=True)
    log_normaliser = maximum + np.log(np.exp(rows - maximum).sum(axis=1, keepdims=True))
    conditional = value[:count, :count] - log_normaliser
    return AxisSocketRankings(
        top_targets=np.ascontiguousarray(order.astype(np.int32)),
        rank=rank,
        conditional_log_score=np.ascontiguousarray(conditional),
    )


def _candidate(
    *,
    rank_sum: int,
    score: float,
) -> tuple[int, float]:
    return rank_sum, -score


def commutative_cycle_support(
    edge: SocketEdge,
    *,
    right: AxisSocketRankings,
    down: AxisSocketRankings,
    top_k: int,
) -> CommutativeCycleSupport:
    """Find closures of ``right; down == down; right`` for one hard edge.

    All four directed sides, including the candidate edge itself, must be in
    their source row's top-K list.  For a right edge ``i->j`` the other three
    sides are ``i->k`` down, ``j->l`` down, and ``k->l`` right.  The down-edge
    case is symmetric.
    """

    if edge.axis not in {"right", "down"}:
        raise ValueError("edge axis must be 'right' or 'down'")
    maximum_k = right.top_targets.shape[1]
    if down.top_targets.shape[1] != maximum_k:
        raise ValueError("right and down rankings must use the same maximum_k")
    if not 1 <= top_k <= maximum_k:
        raise ValueError(f"top_k must be in [1, {maximum_k}]")
    axis = right if edge.axis == "right" else down
    base_rank_value = int(axis.rank[edge.source, edge.target])
    base_rank = base_rank_value if 1 <= base_rank_value <= top_k else None
    if base_rank is None:
        return CommutativeCycleSupport(top_k, None, 0, None, None)

    support_count = 0
    best: tuple[int, float] | None = None
    if edge.axis == "right":
        closing_rank = {
            int(tile): rank
            for rank, tile in enumerate(down.top_targets[edge.target, :top_k], start=1)
        }
        for first_rank, lower_left in enumerate(
            down.top_targets[edge.source, :top_k], start=1
        ):
            for third_rank, lower_right in enumerate(
                right.top_targets[lower_left, :top_k], start=1
            ):
                second_rank = closing_rank.get(int(lower_right))
                if second_rank is None:
                    continue
                support_count += 1
                score = float(
                    right.conditional_log_score[edge.source, edge.target]
                    + down.conditional_log_score[edge.source, lower_left]
                    + down.conditional_log_score[edge.target, lower_right]
                    + right.conditional_log_score[lower_left, lower_right]
                )
                candidate = _candidate(
                    rank_sum=base_rank + first_rank + second_rank + third_rank,
                    score=score,
                )
                if best is None or candidate < best:
                    best = candidate
    else:
        closing_rank = {
            int(tile): rank
            for rank, tile in enumerate(right.top_targets[edge.target, :top_k], start=1)
        }
        for first_rank, upper_right in enumerate(
            right.top_targets[edge.source, :top_k], start=1
        ):
            for third_rank, lower_right in enumerate(
                down.top_targets[upper_right, :top_k], start=1
            ):
                second_rank = closing_rank.get(int(lower_right))
                if second_rank is None:
                    continue
                support_count += 1
                score = float(
                    down.conditional_log_score[edge.source, edge.target]
                    + right.conditional_log_score[edge.source, upper_right]
                    + right.conditional_log_score[edge.target, lower_right]
                    + down.conditional_log_score[upper_right, lower_right]
                )
                candidate = _candidate(
                    rank_sum=base_rank + first_rank + second_rank + third_rank,
                    score=score,
                )
                if best is None or candidate < best:
                    best = candidate

    return CommutativeCycleSupport(
        top_k=top_k,
        base_rank=base_rank,
        support_count=support_count,
        best_total_rank=best[0] if best is not None else None,
        best_total_conditional_log_score=-best[1] if best is not None else None,
    )


__all__ = [
    "AxisSocketRankings",
    "CommutativeCycleSupport",
    "axis_socket_rankings",
    "commutative_cycle_support",
]
