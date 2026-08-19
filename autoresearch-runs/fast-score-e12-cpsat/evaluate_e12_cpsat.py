"""E12: sparse weighted CP-SAT large-neighborhood grid repair.

Targets and truth are evaluation-only. The candidate solver receives exactly
right/down/pos plus a seed, starts from the unchanged baseline layout, and uses
CP-SAT to exactly assign each fixed 4x4 tile set to its grid cells.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from global_solver_candidate import POSITION_WEIGHT, solve_layout

GRID, TILE, N = 24, 20, 576
WINDOW = 4
WINDOWS_PER_CASE = 3
TOP_K = 16
TIME_LIMIT_SECONDS = 1.0
SCORE_SCALE = 1000.0


def assemble(tiles: np.ndarray, layout: np.ndarray) -> np.ndarray:
    return (
        tiles[layout]
        .reshape(GRID, GRID, TILE, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(GRID * TILE, GRID * TILE, 3)
    )


def adjacency(layout: np.ndarray, truth: np.ndarray) -> float:
    target_of = np.empty(N, np.int32)
    target_of[truth] = np.arange(N)
    board = target_of[layout].reshape(GRID, GRID)
    right = (board[:, 1:] == board[:, :-1] + 1) & (
        board[:, 1:] // GRID == board[:, :-1] // GRID
    )
    down = board[1:] == board[:-1] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def full_objective(
    layout: np.ndarray, right: np.ndarray, down: np.ndarray, pos: np.ndarray
) -> float:
    board = np.asarray(layout).reshape(GRID, GRID)
    value = POSITION_WEIGHT * pos[layout, np.arange(N)].sum()
    value += right[board[:, :-1], board[:, 1:]].sum()
    value += down[board[:-1], board[1:]].sum()
    return float(value)


def _window_positions(row: int, col: int) -> np.ndarray:
    return np.asarray(
        [(row + dr) * GRID + col + dc for dr in range(WINDOW) for dc in range(WINDOW)],
        np.int32,
    )


def _incident_objective(
    layout: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    pos: np.ndarray,
    positions: np.ndarray,
) -> float:
    selected = set(int(p) for p in positions)
    value = sum(POSITION_WEIGHT * pos[int(layout[p]), p] for p in positions)
    for row in range(GRID):
        for col in range(GRID - 1):
            p, q = row * GRID + col, row * GRID + col + 1
            if p in selected or q in selected:
                value += right[int(layout[p]), int(layout[q])]
    for row in range(GRID - 1):
        for col in range(GRID):
            p, q = row * GRID + col, (row + 1) * GRID + col
            if p in selected or q in selected:
                value += down[int(layout[p]), int(layout[q])]
    return float(value)


def weakest_non_overlapping_windows(
    layout: np.ndarray, right: np.ndarray, down: np.ndarray, pos: np.ndarray
) -> list[tuple[int, int, float]]:
    candidates = []
    for row in range(GRID - WINDOW + 1):
        for col in range(GRID - WINDOW + 1):
            positions = _window_positions(row, col)
            candidates.append(
                (float(_incident_objective(layout, right, down, pos, positions)), row, col)
            )
    candidates.sort()
    chosen: list[tuple[int, int, float]] = []
    occupied: set[int] = set()
    for weakness, row, col in candidates:
        cells = set(int(p) for p in _window_positions(row, col))
        if occupied.isdisjoint(cells):
            chosen.append((row, col, weakness))
            occupied.update(cells)
        if len(chosen) == WINDOWS_PER_CASE:
            break
    return chosen


def _topk_gain(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(matrix, np.float64).copy()
    np.fill_diagonal(scores, -np.inf)
    order = np.argpartition(scores, -TOP_K, axis=1)[:, -TOP_K:]
    ordered_scores = np.take_along_axis(scores, order, axis=1)
    floor = np.partition(scores, -(TOP_K + 1), axis=1)[:, -(TOP_K + 1)]
    gain = np.maximum(ordered_scores - floor[:, None], 0.0)
    return order.astype(np.int32), gain, floor


@dataclass
class RepairResult:
    layout: np.ndarray
    status: str
    accepted: bool
    before_objective: float
    after_objective: float
    dense_objective_delta: float
    solve_seconds: float
    sparse_edges: int


def repair_window_cpsat(
    layout: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    pos: np.ndarray,
    row0: int,
    col0: int,
    seed: int,
) -> RepairResult:
    positions = _window_positions(row0, col0)
    tiles = np.asarray(layout[positions], np.int32)
    local_index = {int(tile): index for index, tile in enumerate(tiles)}
    position_index = {int(position): index for index, position in enumerate(positions)}
    right_top, right_gain, right_floor = _topk_gain(right)
    down_top, down_gain, down_floor = _topk_gain(down)
    right_sparse = np.broadcast_to(right_floor[:, None], (N, N)).copy()
    down_sparse = np.broadcast_to(down_floor[:, None], (N, N)).copy()
    np.put_along_axis(right_sparse, right_top, right_floor[:, None] + right_gain, axis=1)
    np.put_along_axis(down_sparse, down_top, down_floor[:, None] + down_gain, axis=1)
    before = full_objective(layout, right_sparse, down_sparse, pos)
    dense_before = full_objective(layout, right, down, pos)
    right_lookup = [
        {int(tile): float(gain) for tile, gain in zip(right_top[source], right_gain[source])}
        for source in range(N)
    ]
    down_lookup = [
        {int(tile): float(gain) for tile, gain in zip(down_top[source], down_gain[source])}
        for source in range(N)
    ]

    model = cp_model.CpModel()
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for pi in range(len(positions)):
        for ti in range(len(tiles)):
            x[pi, ti] = model.new_bool_var(f"x_{pi}_{ti}")
        model.add_exactly_one(x[pi, ti] for ti in range(len(tiles)))
    for ti in range(len(tiles)):
        model.add_exactly_one(x[pi, ti] for pi in range(len(positions)))

    objective_terms: list[cp_model.LinearExpr] = []
    for pi, position in enumerate(positions):
        coeff = POSITION_WEIGHT * pos[tiles, int(position)]
        coeff -= coeff.min()
        for ti, value in enumerate(coeff):
            scaled = int(round(float(value) * SCORE_SCALE))
            if scaled:
                objective_terms.append(scaled * x[pi, ti])

    sparse_edges = 0

    def add_internal_edge(
        p: int,
        q: int,
        lookup: list[dict[int, float]],
        floor: np.ndarray,
        label: str,
    ) -> None:
        nonlocal sparse_edges
        pi, qi = position_index[p], position_index[q]
        for source_i, source_tile in enumerate(tiles):
            objective_terms.append(
                int(round(float(floor[int(source_tile)]) * SCORE_SCALE))
                * x[pi, source_i]
            )
            for target_tile, gain in lookup[int(source_tile)].items():
                target_i = local_index.get(target_tile)
                if target_i is None or gain <= 0:
                    continue
                y = model.new_bool_var(f"{label}_{pi}_{source_i}_{target_i}")
                model.add(y <= x[pi, source_i])
                model.add(y <= x[qi, target_i])
                model.add(y >= x[pi, source_i] + x[qi, target_i] - 1)
                objective_terms.append(int(round(gain * SCORE_SCALE)) * y)
                sparse_edges += 1

    selected = set(int(p) for p in positions)
    for p in positions:
        p = int(p)
        row, col = divmod(p, GRID)
        if col + 1 < GRID:
            q = p + 1
            if q in selected:
                add_internal_edge(p, q, right_lookup, right_floor, "r")
            else:
                outside = int(layout[q])
                pi = position_index[p]
                for ti, tile in enumerate(tiles):
                    objective_terms.append(
                        int(round(float(right_floor[int(tile)]) * SCORE_SCALE)) * x[pi, ti]
                    )
                    gain = right_lookup[int(tile)].get(outside, 0.0)
                    if gain > 0:
                        objective_terms.append(int(round(gain * SCORE_SCALE)) * x[pi, ti])
        if row + 1 < GRID:
            q = p + GRID
            if q in selected:
                add_internal_edge(p, q, down_lookup, down_floor, "d")
            else:
                outside = int(layout[q])
                pi = position_index[p]
                for ti, tile in enumerate(tiles):
                    objective_terms.append(
                        int(round(float(down_floor[int(tile)]) * SCORE_SCALE)) * x[pi, ti]
                    )
                    gain = down_lookup[int(tile)].get(outside, 0.0)
                    if gain > 0:
                        objective_terms.append(int(round(gain * SCORE_SCALE)) * x[pi, ti])
        if col == col0 and col > 0:
            outside = int(layout[p - 1])
            pi = position_index[p]
            for ti, tile in enumerate(tiles):
                gain = right_lookup[outside].get(int(tile), 0.0)
                if gain > 0:
                    objective_terms.append(int(round(gain * SCORE_SCALE)) * x[pi, ti])
        if row == row0 and row > 0:
            outside = int(layout[p - GRID])
            pi = position_index[p]
            for ti, tile in enumerate(tiles):
                gain = down_lookup[outside].get(int(tile), 0.0)
                if gain > 0:
                    objective_terms.append(int(round(gain * SCORE_SCALE)) * x[pi, ti])

    model.maximize(sum(objective_terms))
    for pi, tile in enumerate(layout[positions]):
        model.add_hint(x[pi, local_index[int(tile)]], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(seed % (2**31 - 1))
    solver.parameters.log_search_progress = False
    started = time.perf_counter()
    status_code = solver.solve(model)
    elapsed = time.perf_counter() - started
    status = solver.status_name(status_code)
    candidate = layout.copy()
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for pi, position in enumerate(positions):
            chosen = next(ti for ti in range(len(tiles)) if solver.value(x[pi, ti]))
            candidate[int(position)] = tiles[chosen]
    after = full_objective(candidate, right_sparse, down_sparse, pos)
    dense_after = full_objective(candidate, right, down, pos)
    accepted = bool(after > before + 1e-8)
    if not accepted:
        candidate = layout.copy()
        after = before
        dense_after = dense_before
    return RepairResult(
        candidate, status, accepted, before, after, dense_after - dense_before,
        elapsed, sparse_edges
    )


def solve_layout_e12(
    right: np.ndarray, down: np.ndarray, pos: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    layout = np.asarray(solve_layout(right, down, pos, seed), np.int32)
    selected = weakest_non_overlapping_windows(layout, right, down, pos)
    repairs = []
    for repair_index, (row, col, weakness) in enumerate(selected):
        result = repair_window_cpsat(
            layout, right, down, pos, row, col, seed + 7919 * (repair_index + 1)
        )
        layout = result.layout
        repairs.append(
            {
                "row": row,
                "col": col,
                "selection_objective": weakness,
                "status": result.status,
                "accepted": result.accepted,
                "before_objective": result.before_objective,
                "after_objective": result.after_objective,
                "objective_delta": result.after_objective - result.before_objective,
                "dense_objective_delta": result.dense_objective_delta,
                "solve_seconds": result.solve_seconds,
                "sparse_edges": result.sparse_edges,
            }
        )
    return layout, {"repairs": repairs}


def summarize(values: list[float]) -> dict[str, object]:
    scores = np.asarray(values, np.float64)
    folds = np.asarray([scores[offset::4].mean() for offset in range(4)])
    return {
        "mean": float(scores.mean()),
        "robust": float(scores.mean() - 0.5 * folds.std()),
        "folds": folds.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    data = np.load(args.cache, mmap_mode="r")
    cases = min(args.limit, len(data["stems"]))
    rows = {
        "baseline": {"ssim": [], "adjacency": [], "runtime": []},
        "e12_cpsat": {"ssim": [], "adjacency": [], "runtime": []},
    }
    images = []
    repair_status: dict[str, int] = {}
    accepted_repairs = 0
    objective_gain = 0.0
    for index in range(cases):
        seed = 20260818 + index * 100 + args.seed_offset
        started = time.perf_counter()
        baseline = np.asarray(
            solve_layout(data["right"][index], data["down"][index], data["pos"][index], seed),
            np.int32,
        )
        baseline_seconds = time.perf_counter() - started
        started = time.perf_counter()
        candidate, diagnostics = solve_layout_e12(
            data["right"][index], data["down"][index], data["pos"][index], seed
        )
        candidate_seconds = time.perf_counter() - started
        for method, layout, runtime in (
            ("baseline", baseline, baseline_seconds),
            ("e12_cpsat", candidate, candidate_seconds),
        ):
            if (
                layout.shape != (N,)
                or len(np.unique(layout)) != N
                or layout.min() != 0
                or layout.max() != N - 1
            ):
                raise ValueError(f"invalid permutation from {method} at case {index}")
            score = float(
                structural_similarity(
                    data["target"][index], assemble(data["tiles"][index], layout),
                    channel_axis=2, data_range=255,
                )
            )
            adj = adjacency(layout, data["truth"][index])
            if not np.isfinite(score) or not np.isfinite(adj):
                raise FloatingPointError(f"non-finite metric from {method} at case {index}")
            rows[method]["ssim"].append(score)
            rows[method]["adjacency"].append(adj)
            rows[method]["runtime"].append(runtime)
        for repair in diagnostics["repairs"]:
            repair_status[repair["status"]] = repair_status.get(repair["status"], 0) + 1
            accepted_repairs += int(repair["accepted"])
            objective_gain += float(repair["objective_delta"])
        images.append({
            "index": index, "stem": str(data["stems"][index]),
            "baseline_ssim": rows["baseline"]["ssim"][-1],
            "e12_cpsat_ssim": rows["e12_cpsat"]["ssim"][-1],
            "baseline_adjacency": rows["baseline"]["adjacency"][-1],
            "e12_cpsat_adjacency": rows["e12_cpsat"]["adjacency"][-1],
            "baseline_runtime_seconds": baseline_seconds,
            "e12_cpsat_runtime_seconds": candidate_seconds,
            **diagnostics,
        })
        print(json.dumps({"done": index + 1, "total": cases, "stem": str(data["stems"][index]),
                          "baseline_ssim": rows["baseline"]["ssim"][-1],
                          "e12_cpsat_ssim": rows["e12_cpsat"]["ssim"][-1],
                          "accepted_repairs": sum(int(r["accepted"]) for r in diagnostics["repairs"])}), flush=True)

    baseline_ssim = np.asarray(rows["baseline"]["ssim"])
    baseline_adj = np.asarray(rows["baseline"]["adjacency"])
    methods = {}
    for method in rows:
        scores = np.asarray(rows[method]["ssim"])
        adjs = np.asarray(rows[method]["adjacency"])
        runtime = np.asarray(rows[method]["runtime"])
        methods[method] = {
            "ssim": summarize(rows[method]["ssim"]),
            "mean_adjacency": float(adjs.mean()),
            "ssim_wins_vs_baseline": int((scores > baseline_ssim).sum()),
            "adjacency_wins_vs_baseline": int((adjs > baseline_adj).sum()),
            "runtime_seconds": {"total": float(runtime.sum()), "mean": float(runtime.mean())},
        }
    report = {
        "experiment": "E12 sparse weighted CP-SAT 4x4 exact repair",
        "cases": cases, "seed_offset": args.seed_offset,
        "config": {"window": WINDOW, "windows_per_case": WINDOWS_PER_CASE,
                   "top_k": TOP_K, "time_limit_seconds_per_window": TIME_LIMIT_SECONDS,
                   "workers": 1},
        "solver_diagnostics": {"status_counts": repair_status,
                               "accepted_repairs": accepted_repairs,
                               "objective_gain": objective_gain},
        "methods": methods,
        "delta": {
            "mean_ssim": methods["e12_cpsat"]["ssim"]["mean"] - methods["baseline"]["ssim"]["mean"],
            "robust_ssim": methods["e12_cpsat"]["ssim"]["robust"] - methods["baseline"]["ssim"]["robust"],
            "mean_adjacency": methods["e12_cpsat"]["mean_adjacency"] - methods["baseline"]["mean_adjacency"],
            "runtime_seconds": methods["e12_cpsat"]["runtime_seconds"]["total"] - methods["baseline"]["runtime_seconds"]["total"],
        },
        "images": images,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "images"}, indent=2))


if __name__ == "__main__":
    main()
