"""Strict global train-population layout arms for a bounded calibration test.

The population atlas is used only to score which original dirty tile belongs
at each grid slot.  It never renders pixels or supplies replacement content.
Every returned raw board is therefore an exact one-to-one reassembly of the
576 upright input tiles; the shared restoration tail runs only after that raw
permutation has been audited.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from aiijc_puzzle.compliant_atlas_decoder import (
    PRODUCTION_EDGE_BUDGET,
    PermutationAudit,
    audit_raw_permutation,
    population_position_scores,
    solve_buddies_with_position,
)
from aiijc_puzzle.legacy_upgrade import LayoutResult, directional_scores, solve_buddies
from aiijc_puzzle.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import TILE_COUNT, assemble_tiles, split_tiles
from aiijc_puzzle.restoration_r6 import nlm_color

CONTROL_ARM = "no_atlas_buddies96"
PURE_POPULATION_ARM = "population_hungarian"
STRONG_POPULATION_WEIGHTS = (0.25, 1.0)
NLM_H = 20


def strong_arm_name(weight: float) -> str:
    """Return the stable report name for one preregistered strong unary arm."""

    return f"population_w{str(float(weight)).replace('.', 'p')}_buddies96"


FROZEN_ARMS = (
    CONTROL_ARM,
    PURE_POPULATION_ARM,
    *(strong_arm_name(weight) for weight in STRONG_POPULATION_WEIGHTS),
)


def solve_population_hungarian(position_scores: np.ndarray) -> LayoutResult:
    """Maximize the train-population unary under one global bijection."""

    scores = np.asarray(position_scores, dtype=np.float32)
    if scores.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError(f"expected {(TILE_COUNT, TILE_COUNT)} scores, got {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("population scores must be finite")
    started = perf_counter()
    tile_indices, slot_indices = linear_sum_assignment(-scores.astype(np.float64))
    layout = np.empty(TILE_COUNT, dtype=np.int32)
    layout[slot_indices] = tile_indices
    if not np.array_equal(np.sort(layout), np.arange(TILE_COUNT)):
        raise RuntimeError("Hungarian population assignment did not return a bijection")
    objective = float(scores[tile_indices, slot_indices].sum(dtype=np.float64))
    return LayoutResult(
        layout=np.ascontiguousarray(layout),
        objective=objective,
        solver="population_hungarian_train5600",
        runtime_seconds=perf_counter() - started,
    )


@dataclass(frozen=True)
class FrozenGlobalPrediction:
    """One target-blind, tile-preserving layout plus the shared safe tail."""

    layout: np.ndarray
    raw: np.ndarray
    restored: np.ndarray
    audit: PermutationAudit
    solver: str
    objective: float
    solve_seconds: float
    rgb_diagnostics: dict[str, object]
    luminance_diagnostics: dict[str, object]


def restore_frozen_tail(raw: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Apply RGB offsets, bounded luma gains, then colored NLM h20 once."""

    ordered = split_tiles(raw)
    rgb_offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered, SeamGraphConfig())
    rgb = apply_rgb_offsets(ordered, rgb_offsets)
    gains, luminance_diagnostics = seam_graph_luminance_gains(rgb, LuminanceGainConfig())
    harmonized = assemble_tiles(apply_luminance_gains(rgb, gains))
    restored = nlm_color(harmonized, NLM_H)
    return restored, {
        "rgb": rgb_diagnostics,
        "luminance": luminance_diagnostics,
    }


def predict_frozen_roster(
    dirty: np.ndarray,
    generic_tile_template: np.ndarray,
) -> dict[str, FrozenGlobalPrediction]:
    """Build the complete preregistered roster without accepting a target."""

    tiles = split_tiles(dirty)
    right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
    position = population_position_scores(tiles, generic_tile_template)
    solves: dict[str, LayoutResult] = {
        CONTROL_ARM: solve_buddies(right, down, max_edges=PRODUCTION_EDGE_BUDGET),
        PURE_POPULATION_ARM: solve_population_hungarian(position),
    }
    for weight in STRONG_POPULATION_WEIGHTS:
        solves[strong_arm_name(weight)] = solve_buddies_with_position(
            right,
            down,
            position,
            position_weight=weight,
            max_edges=PRODUCTION_EDGE_BUDGET,
        )
    if tuple(solves) != FROZEN_ARMS:
        raise RuntimeError(f"frozen arm roster mismatch: {tuple(solves)}")

    predictions: dict[str, FrozenGlobalPrediction] = {}
    for name, solved in solves.items():
        raw = assemble_tiles(tiles[solved.layout])
        audit = audit_raw_permutation(
            dirty,
            raw,
            solved.layout,
            restoration_applied_after_audit=True,
        )
        if not audit.passed:
            raise RuntimeError(f"strict permutation audit failed for {name}: {audit.as_dict()}")
        restored, diagnostics = restore_frozen_tail(raw)
        predictions[name] = FrozenGlobalPrediction(
            layout=solved.layout,
            raw=raw,
            restored=restored,
            audit=audit,
            solver=solved.solver,
            objective=solved.objective,
            solve_seconds=solved.runtime_seconds,
            rgb_diagnostics=diagnostics["rgb"],
            luminance_diagnostics=diagnostics["luminance"],
        )
    return predictions
