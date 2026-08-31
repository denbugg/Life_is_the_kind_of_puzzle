"""Use a frozen per-tile denoiser only as a target-blind edge-score view.

The rendered tiles produced by the denoiser are never assembled into the output
image.  They are used only to compute E14 directional compatibility matrices;
every returned layout is a strict permutation rendered from the original dirty
tiles by the experiment runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from aiijc_puzzle.legacy_upgrade import LayoutResult, directional_scores, solve_buddies
from aiijc_puzzle.protocol import GRID_SIZE, TILE_COUNT, TILE_SIZE

BASELINE = "baseline_dirty_bilateral"
DUALNAF_RAW = "dualnaf_match_raw"
DUALNAF_BILATERAL = "dualnaf_match_bilateral"
FUSION = "fusion_dirty_bilateral_dualnaf_raw_50_50"
MATCHER_ROSTER = (BASELINE, DUALNAF_RAW, DUALNAF_BILATERAL, FUSION)
PRIMARY_VARIANT = FUSION


@dataclass(frozen=True)
class MatchLayout:
    """One strict layout and the score source used to produce it."""

    name: str
    result: LayoutResult


def _validate_tiles(tiles: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if value.shape != expected:
        raise ValueError(f"{label} must have shape {expected}, got {value.shape}")
    if value.dtype != np.uint8:
        raise TypeError(f"{label} must be uint8, got {value.dtype}")
    return np.ascontiguousarray(value)


def _validate_score(score: np.ndarray) -> np.ndarray:
    value = np.asarray(score, dtype=np.float32)
    if value.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError(f"expected score shape {(TILE_COUNT, TILE_COUNT)}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("directional score must be finite")
    return value


def normalized_score_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Average two already row-normalized E14 log-probability matrices."""

    left = _validate_score(first)
    right = _validate_score(second)
    return np.mean((left, right), axis=0, dtype=np.float64).astype(np.float32)


def matcher_score_roster(
    dirty_tiles: np.ndarray,
    dualnaf_tiles: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build the complete preregistered dirty-only matcher roster.

    ``directional_scores`` converts every E14 cost matrix to row
    log-probabilities before it is returned.  Consequently the sole fusion arm
    is a fixed normalized 50/50 mean rather than a scale-sensitive raw-cost
    mixture.
    """

    dirty = _validate_tiles(dirty_tiles, label="dirty_tiles")
    rendered = _validate_tiles(dualnaf_tiles, label="dualnaf_tiles")
    baseline = directional_scores(dirty, views=("bilateral",))["bilateral"]
    dual = directional_scores(rendered, views=("raw", "bilateral"))
    dual_raw = dual["raw"]
    dual_bilateral = dual["bilateral"]
    fusion = (
        normalized_score_mean(baseline[0], dual_raw[0]),
        normalized_score_mean(baseline[1], dual_raw[1]),
    )
    roster = {
        BASELINE: baseline,
        DUALNAF_RAW: dual_raw,
        DUALNAF_BILATERAL: dual_bilateral,
        FUSION: fusion,
    }
    if tuple(roster) != MATCHER_ROSTER:
        raise RuntimeError("matcher roster changed from the preregistered order")
    return roster


def solve_matcher_roster(
    scores: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    edge_budget: int = 96,
) -> dict[str, MatchLayout]:
    """Solve every preregistered score arm with the same strict buddies decoder."""

    if edge_budget <= 0:
        raise ValueError("edge_budget must be positive")
    if tuple(scores) != MATCHER_ROSTER:
        raise ValueError(f"expected exact matcher roster {MATCHER_ROSTER}")
    solved: dict[str, MatchLayout] = {}
    for name in MATCHER_ROSTER:
        right, down = scores[name]
        result = solve_buddies(
            _validate_score(right),
            _validate_score(down),
            max_edges=edge_budget,
        )
        layout = np.asarray(result.layout)
        if layout.shape != (TILE_COUNT,) or not np.array_equal(
            np.sort(layout), np.arange(TILE_COUNT)
        ):
            raise RuntimeError(
                f"{name} did not return a strict {GRID_SIZE}x{GRID_SIZE} permutation"
            )
        solved[name] = MatchLayout(name=name, result=result)
    return solved
