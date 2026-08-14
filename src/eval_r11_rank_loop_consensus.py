"""R11 rank-normalized loop-consensus layout selector.

The module generates a fixed ensemble of layouts from unchanged R/D scores and
selects layouts using only non-self row ranks and 2x2 geometric loops.  It never
needs a target except in the separately gated single-CAL lambda calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from solve_buddies import (
    _fill_board,
    _place_components_randomized,
    build_buddies_components,
    solve_buddies_from_scores,
)

GRID = 24
NFRAG = GRID * GRID
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R11_rank_loop_consensus")


def assert_layout(layout: np.ndarray) -> np.ndarray:
    value = np.asarray(layout, dtype=np.int64).reshape(-1)
    if value.shape != (NFRAG,) or not np.array_equal(np.sort(value), np.arange(NFRAG)):
        raise ValueError("layout must be a 576-tile bijection")
    return value


def nonself_ranks(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (NFRAG, NFRAG) or not np.isfinite(scores).all():
        raise ValueError("R/D matrix must be finite 576x576")
    ranks = np.full((NFRAG, NFRAG), NFRAG - 1, dtype=np.int16)
    for anchor in range(NFRAG):
        order = np.argsort(-scores[anchor], kind="stable")
        order = order[order != anchor]
        ranks[anchor, order] = np.arange(NFRAG - 1, dtype=np.int16)
    return ranks


def confidence(ranks: np.ndarray, anchor: int, candidate: int) -> float:
    return float((NFRAG - 2 - int(ranks[anchor, candidate])) / (NFRAG - 2))


def edge_and_loop_score(layout: np.ndarray, right_ranks: np.ndarray, down_ranks: np.ndarray) -> Tuple[float, float]:
    board = assert_layout(layout).reshape(GRID, GRID)
    total_edge = 0.0
    loop = 0.0
    for row in range(GRID):
        for col in range(GRID - 1):
            total_edge += confidence(right_ranks, int(board[row, col]), int(board[row, col + 1]))
    for row in range(GRID - 1):
        for col in range(GRID):
            total_edge += confidence(down_ranks, int(board[row, col]), int(board[row + 1, col]))
    for row in range(GRID - 1):
        for col in range(GRID - 1):
            top = confidence(right_ranks, int(board[row, col]), int(board[row, col + 1]))
            bottom = confidence(right_ranks, int(board[row + 1, col]), int(board[row + 1, col + 1]))
            left = confidence(down_ranks, int(board[row, col]), int(board[row + 1, col]))
            right = confidence(down_ranks, int(board[row, col + 1]), int(board[row + 1, col + 1]))
            loop += min(top, bottom, left, right)
    return float(total_edge), float(loop)


def generate_layouts(right: np.ndarray, down: np.ndarray, *, max_edges: int = 96, count: int = 32, seed: int = 20260814) -> np.ndarray:
    if max_edges != 96 or count != 32:
        raise ValueError("R11 is pre-registered at max_edges=96, count=32")
    canonical, _ = solve_buddies_from_scores(right, down, max_edges=max_edges, min_margin=0.0, repair_passes=0)
    layouts = [assert_layout(canonical)]
    components = build_buddies_components(right, down, max_edges=max_edges, min_margin=0.0)
    for index in range(1, count):
        rng = np.random.default_rng(seed + index)
        board, used = _place_components_randomized(components, right, down, rng, temperature=0.03, order_jitter=0.25)
        layout = _fill_board(board, used, right, down)
        layouts.append(assert_layout(layout))
    result = np.stack(layouts, axis=0)
    if result.shape != (count, NFRAG):
        raise RuntimeError("unexpected R11 ensemble shape")
    return result


def candidate_scores(layouts: np.ndarray, right: np.ndarray, down: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rr, dr = nonself_ranks(right), nonself_ranks(down)
    edges, loops = [], []
    for layout in layouts:
        edge, loop = edge_and_loop_score(layout, rr, dr)
        edges.append(edge)
        loops.append(loop)
    return np.asarray(edges, dtype=np.float64), np.asarray(loops, dtype=np.float64)


def select_layout(layouts: np.ndarray, edge: np.ndarray, loop: np.ndarray, lam: float) -> Tuple[int, np.ndarray]:
    objective = np.asarray(edge, dtype=np.float64) + float(lam) * np.asarray(loop, dtype=np.float64)
    index = int(np.argmax(objective))
    return index, objective


def oracle_scores() -> Tuple[np.ndarray, np.ndarray]:
    right = np.full((NFRAG, NFRAG), -10.0, dtype=np.float32)
    down = np.full((NFRAG, NFRAG), -10.0, dtype=np.float32)
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


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=WORK / "g0_smoke")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--lambda", dest="lam", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    cfg = args()
    right, down = oracle_scores()
    layouts = generate_layouts(right, down)
    edge, loop = candidate_scores(layouts, right, down)
    selected, objective = select_layout(layouts, edge, loop, cfg.lam)
    identity = np.arange(NFRAG, dtype=np.int64)
    report = {
        "experiment": "R11_rank_loop_consensus",
        "gate": "G0_oracle_smoke",
        "parameters": {"layout_count": 32, "max_edges": 96, "temperature": 0.03, "order_jitter": 0.25, "lambda": cfg.lam},
        "selected_index": selected,
        "selected_identity_accuracy": float(np.mean(layouts[selected] == identity)),
        "canonical_identity_accuracy": float(np.mean(layouts[0] == identity)),
        "all_bijections": bool(all(np.array_equal(np.sort(layout), identity) for layout in layouts)),
        "edge_range": [float(edge.min()), float(edge.max())],
        "loop_range": [float(loop.min()), float(loop.max())],
        "targets_opened": False,
        "orientation": "fixed_no_rotations",
    }
    report["passes_G0"] = bool(report["all_bijections"] and report["selected_identity_accuracy"] == 1.0)
    report["decision"] = "advance_to_R11_G1_CAL" if report["passes_G0"] else "reject_R11_before_CAL"
    cfg.work.mkdir(parents=True, exist_ok=True)
    path = cfg.report or cfg.work / "r11_g0_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
