"""Use official DRUNet40 tiles only as a target-blind bilateral matcher view."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from aiijc_puzzle.legacy_upgrade import LayoutResult, directional_scores, solve_buddies
from aiijc_puzzle.protocol import GRID_SIZE, TILE_COUNT, TILE_SIZE

BASELINE = "dirty_bilateral"
PURE_DRUNET = "drunet40_bilateral"
FUSION_WEIGHTS = (0.25, 0.5, 0.75)


def fusion_name(weight: float) -> str:
    if weight not in FUSION_WEIGHTS:
        raise ValueError(f"unsupported frozen fusion weight: {weight}")
    return f"fusion_drunet_weight_{int(round(weight * 100)):02d}"


FUSION_NAMES = tuple(fusion_name(weight) for weight in FUSION_WEIGHTS)
ARM_NAMES = (BASELINE, PURE_DRUNET, *FUSION_NAMES)


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


def normalized_score_fusion(
    dirty_score: np.ndarray,
    drunet_score: np.ndarray,
    *,
    drunet_weight: float,
) -> np.ndarray:
    """Fuse two row-normalized log-score matrices at one frozen global weight."""

    if drunet_weight not in FUSION_WEIGHTS:
        raise ValueError("drunet_weight is not in the frozen roster")
    dirty = _validate_score(dirty_score)
    drunet = _validate_score(drunet_score)
    fused = (1.0 - drunet_weight) * dirty.astype(np.float64) + drunet_weight * drunet
    return np.ascontiguousarray(fused.astype(np.float32))


def matcher_score_roster(
    dirty_tiles: np.ndarray,
    drunet_tiles: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return the exact dirty, DRUNet and normalized-fusion bilateral score roster."""

    dirty = _validate_tiles(dirty_tiles, label="dirty_tiles")
    rendered = _validate_tiles(drunet_tiles, label="drunet_tiles")
    dirty_scores = directional_scores(dirty, views=("bilateral",))["bilateral"]
    drunet_scores = directional_scores(rendered, views=("bilateral",))["bilateral"]
    roster: dict[str, tuple[np.ndarray, np.ndarray]] = {
        BASELINE: dirty_scores,
        PURE_DRUNET: drunet_scores,
    }
    for weight, name in zip(FUSION_WEIGHTS, FUSION_NAMES, strict=True):
        roster[name] = (
            normalized_score_fusion(
                dirty_scores[0],
                drunet_scores[0],
                drunet_weight=weight,
            ),
            normalized_score_fusion(
                dirty_scores[1],
                drunet_scores[1],
                drunet_weight=weight,
            ),
        )
    if tuple(roster) != ARM_NAMES:
        raise RuntimeError("DRUNet matcher roster drifted")
    return roster


def solve_matcher_roster(
    scores: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    edge_budget: int = 96,
) -> dict[str, LayoutResult]:
    """Solve all frozen score arms as strict 24x24 permutations."""

    if tuple(scores) != ARM_NAMES:
        raise ValueError(f"expected exact score roster {ARM_NAMES}")
    if isinstance(edge_budget, bool) or not isinstance(edge_budget, int) or edge_budget <= 0:
        raise ValueError("edge_budget must be a positive integer")
    output: dict[str, LayoutResult] = {}
    for name in ARM_NAMES:
        right, down = scores[name]
        solved = solve_buddies(
            _validate_score(right),
            _validate_score(down),
            max_edges=edge_budget,
        )
        layout = np.asarray(solved.layout)
        if layout.shape != (TILE_COUNT,) or not np.array_equal(
            np.sort(layout), np.arange(TILE_COUNT)
        ):
            raise RuntimeError(f"{name} did not return a strict {GRID_SIZE}x{GRID_SIZE} layout")
        output[name] = solved
    return output


__all__ = [
    "ARM_NAMES",
    "BASELINE",
    "FUSION_NAMES",
    "FUSION_WEIGHTS",
    "PURE_DRUNET",
    "fusion_name",
    "matcher_score_roster",
    "normalized_score_fusion",
    "solve_matcher_roster",
]
