"""R10 global component multistart layout-only experiment.

G0 consumes only a synthetic oracle R/D matrix.  The later G1 adapter will feed
frozen canonical rank96 R/D score matrices without altering their retrieval
contract.  No tile rotations, target-derived score features, or restoration are
present here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from solve_buddies import objective, solve_buddies_from_scores, solve_buddies_multistart_from_scores

GRID = 24
NFRAG = GRID * GRID
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R10_global_component_multistart")


def oracle_scores() -> tuple[np.ndarray, np.ndarray]:
    """Perfect adjacency scores for identity tile-to-cell placement."""
    right = np.full((NFRAG, NFRAG), -10.0, np.float32)
    down = np.full((NFRAG, NFRAG), -10.0, np.float32)
    np.fill_diagonal(right, -1e6)
    np.fill_diagonal(down, -1e6)
    for row in range(GRID):
        for col in range(GRID):
            tile = row * GRID + col
            if col + 1 < GRID:
                right[tile, tile + 1] = 10.0
            if row + 1 < GRID:
                down[tile, tile + GRID] = 10.0
    return right, down


def valid_bijection(place: np.ndarray) -> bool:
    return place.shape == (NFRAG,) and np.array_equal(np.sort(place), np.arange(NFRAG, dtype=place.dtype))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=WORK / "g0_smoke")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--max-edges", type=int, default=96)
    p.add_argument("--restarts", type=int, default=32)
    p.add_argument("--repair-passes", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260814)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    if cfg.max_edges != 96 or cfg.restarts != 32 or cfg.repair_passes != 0:
        raise ValueError("R10-A G0 is pre-registered at edges=96, restarts=32, repair=0")
    right, down = oracle_scores()
    canonical, canonical_obj = solve_buddies_from_scores(right, down, max_edges=cfg.max_edges, min_margin=0.0, repair_passes=0)
    optimized, optimized_obj = solve_buddies_multistart_from_scores(right, down, max_edges=cfg.max_edges, min_margin=0.0, repair_passes=cfg.repair_passes, restarts=cfg.restarts, seed=cfg.seed, temperature=0.03, order_jitter=0.25)
    expected = np.arange(NFRAG, dtype=np.int64)
    canonical = np.asarray(canonical, dtype=np.int64)
    optimized = np.asarray(optimized, dtype=np.int64)
    report = {
        "experiment": "R10_global_component_multistart",
        "gate": "G0_oracle_smoke",
        "parameters": {"max_edges": cfg.max_edges, "restarts": cfg.restarts, "repair_passes": cfg.repair_passes, "temperature": 0.03, "order_jitter": 0.25, "variant": "R10-A_component_packing_only"},
        "oracle": {"identity_place_accuracy": float(np.mean(optimized == expected)), "canonical_place_accuracy": float(np.mean(canonical == expected)), "canonical_objective": float(canonical_obj), "optimized_objective": float(optimized_obj), "recomputed_canonical_objective": float(objective(canonical, right, down)), "recomputed_optimized_objective": float(objective(optimized, right, down))},
        "invariants": {"canonical_bijection": valid_bijection(canonical), "optimized_bijection": valid_bijection(optimized), "fixed_grid": [GRID, GRID], "orientation": "fixed_no_rotations", "score_source": "oracle_only"},
    }
    report["passes_G0"] = bool(report["invariants"]["canonical_bijection"] and report["invariants"]["optimized_bijection"] and report["oracle"]["optimized_objective"] >= report["oracle"]["canonical_objective"] and report["oracle"]["identity_place_accuracy"] == 1.0)
    report["decision"] = "advance_to_R10_G1" if report["passes_G0"] else "reject_R10_before_canonical_scores"
    cfg.work.mkdir(parents=True, exist_ok=True)
    destination = cfg.report or cfg.work / "r10_g0_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
