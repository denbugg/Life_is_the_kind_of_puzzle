"""Target-free rank calibration for SocketMatcher/edge-ranker fusion.

The two models emit scores on unrelated numerical scales.  This module keeps
only each model's within-anchor ordering, maps it to a fixed normal-score
distribution, and lets the existing exact-capacity partial OT layer enforce
the target-side one-to-one constraint.  No clean image or absolute position is
used by these helpers.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import ndtri

from aiijc_puzzle.socket_matcher import partial_log_optimal_transport


def _square_scores(value: np.ndarray, *, name: str) -> np.ndarray:
    scores = np.asarray(value, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or len(scores) < 2:
        raise ValueError(f"{name} must be a non-trivial square matrix, got {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(scores)


def robust_zscore(value: np.ndarray) -> np.ndarray:
    """Median/MAD normalisation with a finite constant-vector fallback."""

    scores = np.asarray(value, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("robust_zscore input contains non-finite values")
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    return np.ascontiguousarray((scores - median) / max(1.4826 * mad, 1e-6))


def row_rank_calibrate(value: np.ndarray) -> np.ndarray:
    """Map every directed score row to the same inverse-normal rank scale.

    Stable tie-breaking by tile index makes the transform deterministic.  The
    forbidden self-edge remains forbidden and does not consume an off-diagonal
    quantile.
    """

    scores = _square_scores(value, name="scores").copy()
    count = len(scores)
    diagonal = np.arange(count)
    scores[diagonal, diagonal] = -np.inf
    order = np.argsort(scores, axis=1, kind="mergesort")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(count, dtype=np.int64), order.shape),
        axis=1,
    )
    # The diagonal has rank zero.  The N-1 legal candidates consequently map
    # to centred quantiles (0.5/(N-1), ..., 1-0.5/(N-1)).
    quantile = (ranks.astype(np.float64) - 0.5) / float(count - 1)
    quantile = np.clip(quantile, 0.5 / (count - 1), 1.0 - 0.5 / (count - 1))
    calibrated = ndtri(quantile)
    calibrated[diagonal, diagonal] = -1e4
    return np.ascontiguousarray(calibrated.astype(np.float32))


def analytic_border_logits(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer outgoing/incoming border logits from weak best-match evidence."""

    scores = _square_scores(value, name="scores").copy()
    np.fill_diagonal(scores, -np.inf)
    outgoing = robust_zscore(-np.max(scores, axis=1))
    incoming = robust_zscore(-np.max(scores, axis=0))
    return outgoing.astype(np.float32), incoming.astype(np.float32)


def calibrated_partial_assignment(
    value: np.ndarray,
    *,
    grid: int,
    outgoing_border: np.ndarray | None = None,
    incoming_border: np.ndarray | None = None,
    iterations: int = 10,
) -> np.ndarray:
    """Rank-calibrate dense scores and apply exact-capacity partial OT."""

    scores = _square_scores(value, name="scores")
    count = len(scores)
    if grid * grid != count:
        raise ValueError(f"score count {count} is not grid^2 for grid={grid}")
    if outgoing_border is None or incoming_border is None:
        inferred_outgoing, inferred_incoming = analytic_border_logits(scores)
        outgoing_border = inferred_outgoing if outgoing_border is None else outgoing_border
        incoming_border = inferred_incoming if incoming_border is None else incoming_border
    outgoing = np.asarray(outgoing_border, dtype=np.float32)
    incoming = np.asarray(incoming_border, dtype=np.float32)
    if outgoing.shape != (count,) or incoming.shape != (count,):
        raise ValueError(f"border logits must both have shape {(count,)}")
    if not np.isfinite(outgoing).all() or not np.isfinite(incoming).all():
        raise ValueError("border logits contain non-finite values")
    calibrated = torch.from_numpy(row_rank_calibrate(scores)).unsqueeze(0)
    assignment = partial_log_optimal_transport(
        calibrated,
        torch.from_numpy(outgoing).unsqueeze(0),
        unmatched=grid,
        iterations=iterations,
        target_bin_score=torch.from_numpy(incoming).unsqueeze(0),
    )
    return np.ascontiguousarray(assignment[0].detach().cpu().numpy().astype(np.float32))


def equal_rank_fusion(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_outgoing_border: np.ndarray,
    first_incoming_border: np.ndarray,
    second_outgoing_border: np.ndarray,
    second_incoming_border: np.ndarray,
    grid: int,
    iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed 50/50 rank scores and their partial-OT assignment."""

    first_scores = _square_scores(first, name="first")
    second_scores = _square_scores(second, name="second")
    if first_scores.shape != second_scores.shape:
        raise ValueError("fusion score matrices must have identical shapes")
    count = len(first_scores)

    def border(value: np.ndarray, *, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (count,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite vector with shape {(count,)}")
        return robust_zscore(result).astype(np.float32)

    fused = 0.5 * row_rank_calibrate(first_scores) + 0.5 * row_rank_calibrate(second_scores)
    outgoing = 0.5 * border(first_outgoing_border, name="first_outgoing_border") + 0.5 * border(
        second_outgoing_border, name="second_outgoing_border"
    )
    incoming = 0.5 * border(first_incoming_border, name="first_incoming_border") + 0.5 * border(
        second_incoming_border, name="second_incoming_border"
    )
    assignment = partial_log_optimal_transport(
        torch.from_numpy(fused).unsqueeze(0),
        torch.from_numpy(outgoing).unsqueeze(0),
        unmatched=grid,
        iterations=iterations,
        target_bin_score=torch.from_numpy(incoming).unsqueeze(0),
    )
    return (
        np.ascontiguousarray(fused.astype(np.float32)),
        np.ascontiguousarray(assignment[0].detach().cpu().numpy().astype(np.float32)),
    )
