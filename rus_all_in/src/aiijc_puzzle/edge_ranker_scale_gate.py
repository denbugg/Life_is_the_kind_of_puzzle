"""Frozen promotion gate for the candidate-k16 edge-ranker scale attempt."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from aiijc_puzzle.edge_ranker import EdgeRow
from aiijc_puzzle.edge_ranker_final_tail import paired_bootstrap_ci


def candidate_coverage(rows: Sequence[EdgeRow]) -> dict[str, dict[str, float | int]]:
    """Summarise exact-neighbour inclusion in the dirty-only candidate union."""

    result: dict[str, dict[str, float | int]] = {}
    scopes = {
        "all": tuple(rows),
        "right": tuple(row for row in rows if row.direction == 0),
        "down": tuple(row for row in rows if row.direction == 1),
        "trusted_query": tuple(row for row in rows if row.trusted_query),
    }
    for name, selected in scopes.items():
        present = sum(row.exact_candidate >= 0 for row in selected)
        result[name] = {
            "rows": len(selected),
            "exact_in_candidate_union": present,
            "coverage": present / len(selected) if selected else 0.0,
        }
    return result


def scale_promotion_gate(
    learned_adjacency: Sequence[float] | np.ndarray,
    adjacency_differences: Sequence[float] | np.ndarray,
    final_ssim_differences: Sequence[float] | np.ndarray,
    translation_differences: Sequence[float] | np.ndarray,
    *,
    seed: int = 20260830,
    replicates: int = 20_000,
) -> dict[str, Any]:
    """Apply every condition fixed in the k16 scale preregistration."""

    learned = np.asarray(learned_adjacency, dtype=np.float64)
    adjacency_delta = np.asarray(adjacency_differences, dtype=np.float64)
    ssim_delta = np.asarray(final_ssim_differences, dtype=np.float64)
    translation_delta = np.asarray(translation_differences, dtype=np.float64)
    lengths = {len(learned), len(adjacency_delta), len(ssim_delta), len(translation_delta)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        raise ValueError("all metric vectors must have the same length of at least two")
    if not all(
        np.isfinite(values).all()
        for values in (learned, adjacency_delta, ssim_delta, translation_delta)
    ):
        raise ValueError("metric vectors must be finite")

    adjacency_ci = paired_bootstrap_ci(
        adjacency_delta,
        seed=seed,
        replicates=replicates,
    )
    ssim_ci = paired_bootstrap_ci(
        ssim_delta,
        seed=seed + 1,
        replicates=replicates,
    )
    observed = {
        "learned_final_adjacency_mean": float(learned.mean()),
        "adjacency_gain_ci95_lower": float(adjacency_ci["ci95_lower"]),
        "final_ssim_delta_mean": float(ssim_delta.mean()),
        "final_ssim_delta_ci95_lower": float(ssim_ci["ci95_lower"]),
        "translation_aligned_placement_delta_mean": float(translation_delta.mean()),
    }
    requirements: tuple[tuple[str, str, bool], ...] = (
        (
            "learned_final_adjacency_mean",
            ">= 0.08",
            observed["learned_final_adjacency_mean"] >= 0.08,
        ),
        (
            "adjacency_gain_ci95_lower",
            "> 0",
            observed["adjacency_gain_ci95_lower"] > 0.0,
        ),
        (
            "final_ssim_delta_mean",
            ">= -0.002",
            observed["final_ssim_delta_mean"] >= -0.002,
        ),
        (
            "final_ssim_delta_ci95_lower",
            ">= -0.006",
            observed["final_ssim_delta_ci95_lower"] >= -0.006,
        ),
        (
            "translation_aligned_placement_delta_mean",
            ">= 0",
            observed["translation_aligned_placement_delta_mean"] >= 0.0,
        ),
    )
    conditions = [
        {
            "metric": metric,
            "observed": observed[metric],
            "required": required,
            "passed": bool(passed),
        }
        for metric, required, passed in requirements
    ]
    return {
        "passed": all(condition["passed"] for condition in conditions),
        "conditions": conditions,
        "adjacency_delta": adjacency_ci,
        "final_ssim_delta": ssim_ci,
    }


__all__ = ["candidate_coverage", "scale_promotion_gate"]
