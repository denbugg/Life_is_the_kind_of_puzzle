"""Frozen helpers for the matched Socket/Pasha source-exposed diagnostic.

The only fusion admitted by this module is the preregistered equal-weight
average of row-wise rank percentiles.  Every row excludes its self-pair before
ranking.  The transform deliberately removes each model's arbitrary score
scale without consulting a layout or a target.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def validate_directional_scores(
    right: np.ndarray,
    down: np.ndarray,
    *,
    tile_count: int = 576,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one finite pair of square directional score matrices."""

    if isinstance(tile_count, bool) or not isinstance(tile_count, int) or tile_count < 3:
        raise ValueError("tile_count must be an integer of at least three")
    values = tuple(np.asarray(matrix, dtype=np.float32) for matrix in (right, down))
    expected = (tile_count, tile_count)
    if any(matrix.shape != expected for matrix in values):
        raise ValueError(f"right/down scores must both have shape {expected}")
    if any(not np.isfinite(matrix).all() for matrix in values):
        raise ValueError("right/down scores must be finite")
    return tuple(np.ascontiguousarray(matrix) for matrix in values)  # type: ignore[return-value]


def row_rank_percentiles(scores: np.ndarray) -> np.ndarray:
    """Map scores to deterministic high-is-good percentiles, masking self.

    For a board with ``N`` tiles, each row ranks only its ``N-1`` non-self
    candidates.  Stable descending ordinal ranks map to ``1 .. 0`` and the
    diagonal is fixed to ``-1``.  Continuous model scores make ties rare; the
    stable rule fixes their behaviour without reference labels.
    """

    value = np.asarray(scores, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or len(value) < 3:
        raise ValueError(
            f"scores must be a square matrix of size at least three, got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError("scores must be finite")
    count = len(value)
    output = np.full((count, count), -1.0, dtype=np.float32)
    identities = np.arange(count)
    for row in range(count):
        candidates = identities[identities != row]
        order = np.argsort(-value[row, candidates], kind="stable")
        percentiles = np.linspace(1.0, 0.0, count - 1, dtype=np.float32)
        output[row, candidates[order]] = percentiles
    return output


def fuse_pasha_socket_ot_rank_percentiles(
    pasha_right: np.ndarray,
    pasha_down: np.ndarray,
    socket_ot_right: np.ndarray,
    socket_ot_down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sole preregistered 50/50 Pasha + Socket-OT fusion."""

    tile_count = len(np.asarray(pasha_right))
    pasha = validate_directional_scores(
        pasha_right,
        pasha_down,
        tile_count=tile_count,
    )
    socket = validate_directional_scores(
        socket_ot_right,
        socket_ot_down,
        tile_count=tile_count,
    )
    fused = tuple(
        np.ascontiguousarray(
            0.5 * row_rank_percentiles(pasha_axis)
            + 0.5 * row_rank_percentiles(socket_axis),
            dtype=np.float32,
        )
        for pasha_axis, socket_axis in zip(pasha, socket, strict=True)
    )
    return fused  # type: ignore[return-value]


def mean_numeric_metrics(
    rows: list[Mapping[str, Any]],
    key: str,
) -> dict[str, float]:
    """Average the numeric fields of one identical-query-count board metric."""

    if not rows:
        raise ValueError("rows must be non-empty")
    first = rows[0].get(key)
    if not isinstance(first, Mapping):
        raise ValueError(f"row key {key!r} must contain a mapping")
    return {
        name: float(np.mean([float(row[key][name]) for row in rows]))
        for name, value in first.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


__all__ = [
    "fuse_pasha_socket_ot_rank_percentiles",
    "mean_numeric_metrics",
    "row_rank_percentiles",
    "validate_directional_scores",
]
