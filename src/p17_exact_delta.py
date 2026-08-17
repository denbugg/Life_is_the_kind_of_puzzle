"""P17 EDSP-24 -- Exact-Delta Sparse QAP Polish.

Pre-registered in P17_PRE_REGISTRATION.md before this source file was created.
Only frozen P12 score cache data is used in G0b.  FIT labels are read only after
G0 in G1. No target PNG, CAL, DEV, held or test image is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_seeded_qap import dense_rd
from solve_buddies import GRID, NFRAG, objective, solve_buddies_from_scores
import p12_loop_consensus as p12
import p13_component_pose as p13

N = GRID * GRID
ROUNDS = 24
MAX_EDGES = 96
SEED = 20260817
CHECKPOINT_COUNT = 16
TOL_DELTA = 1e-5
TOL_TOTAL = 1e-4


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def canonical_bytes(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    return str(arr.dtype).encode() + b"\0" + repr(arr.shape).encode() + b"\0" + arr.tobytes()


def array_sha(arr: np.ndarray) -> str:
    return hashlib.sha256(canonical_bytes(arr)).hexdigest()


def assert_p8_absent(*paths: Path) -> None:
    if "p8" in "\n".join(str(p).lower() for p in paths):
        raise RuntimeError("P8 artifacts are prohibited for P17")


def validate(board: np.ndarray, grid: int = GRID) -> None:
    flat = np.asarray(board, dtype=np.int64).reshape(-1)
    expected = grid * grid
    if flat.shape != (expected,) or np.any(flat < 0) or np.any(flat >= expected) or np.unique(flat).size != expected:
        raise RuntimeError("not a strict permutation")


def all_edges(grid: int) -> list[tuple[int, int]]:
    """(direction, start cell), 0=right, 1=down."""
    out: list[tuple[int, int]] = []
    for y in range(grid):
        for x in range(grid):
            cell = y * grid + x
            if x + 1 < grid:
                out.append((0, cell))
            if y + 1 < grid:
                out.append((1, cell))
    return out


def edge_value(board: np.ndarray, edge: tuple[int, int], right: np.ndarray, down: np.ndarray, grid: int, swap_a: int | None = None, swap_b: int | None = None) -> float:
    direction, start = edge
    end = start + (1 if direction == 0 else grid)
    a = int(board[start])
    b = int(board[end])
    if swap_a is not None and swap_b is not None:
        if start == swap_a:
            a = int(board[swap_b])
        elif start == swap_b:
            a = int(board[swap_a])
        if end == swap_a:
            b = int(board[swap_b])
        elif end == swap_b:
            b = int(board[swap_a])
    return float(right[a, b] if direction == 0 else down[a, b])


def grid_objective(board: np.ndarray, right: np.ndarray, down: np.ndarray, grid: int = GRID) -> float:
    return float(sum(edge_value(board, edge, right, down, grid) for edge in all_edges(grid)))


def affected_edges(a: int, b: int, grid: int) -> tuple[tuple[int, int], ...]:
    out: set[tuple[int, int]] = set()
    for cell in (a, b):
        y, x = divmod(cell, grid)
        if x > 0:
            out.add((0, cell - 1))
        if x + 1 < grid:
            out.add((0, cell))
        if y > 0:
            out.add((1, cell - grid))
        if y + 1 < grid:
            out.add((1, cell))
    return tuple(sorted(out))


def swap_delta(board: np.ndarray, a: int, b: int, right: np.ndarray, down: np.ndarray, grid: int = GRID) -> float:
    if a == b:
        return 0.0
    edges = affected_edges(a, b, grid)
    before = sum(edge_value(board, edge, right, down, grid) for edge in edges)
    after = sum(edge_value(board, edge, right, down, grid, a, b) for edge in edges)
    return float(after - before)


def apply_swap(board: np.ndarray, a: int, b: int) -> None:
    board[a], board[b] = board[b], board[a]


def best_swap(board: np.ndarray, right: np.ndarray, down: np.ndarray, grid: int = GRID) -> tuple[float, int, int]:
    total = grid * grid
    best_delta = 0.0
    best_a, best_b = -1, -1
    for a in range(total - 1):
        for b in range(a + 1, total):
            delta = swap_delta(board, a, b, right, down, grid)
            if delta > best_delta + 1e-12:
                best_delta, best_a, best_b = delta, a, b
    return float(best_delta), best_a, best_b


def polish(board: np.ndarray, right: np.ndarray, down: np.ndarray, grid: int = GRID, rounds: int = ROUNDS) -> tuple[np.ndarray, dict[str, object]]:
    out = np.asarray(board, dtype=np.int64).copy().reshape(-1)
    validate(out, grid)
    initial = grid_objective(out, right, down, grid)
    accumulated = 0.0
    moves: list[dict[str, object]] = []
    for _ in range(rounds):
        delta, a, b = best_swap(out, right, down, grid)
        if a < 0 or delta <= 1e-12:
            break
        pre = grid_objective(out, right, down, grid)
        apply_swap(out, a, b)
        post = grid_objective(out, right, down, grid)
        exact_error = abs((post - pre) - delta)
        if exact_error > TOL_DELTA:
            raise RuntimeError(f"exact-delta mismatch {exact_error}")
        accumulated += delta
        moves.append({"a": a, "b": b, "delta": delta, "exact_error": exact_error})
    final = grid_objective(out, right, down, grid)
    total_error = abs((initial + accumulated) - final)
    if total_error > TOL_TOTAL:
        raise RuntimeError(f"accumulated-delta mismatch {total_error}")
    validate(out, grid)
    return out, {"initial_objective": initial, "final_objective": final, "objective_delta": final - initial, "moves": moves, "delta_total_error": total_error, "output_sha256": array_sha(out)}


def planted_scores(grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = grid * grid
    planted = np.arange(n, dtype=np.int64)
    right = np.full((n, n), -3.0, dtype=np.float32)
    down = np.full((n, n), -3.0, dtype=np.float32)
    for y in range(grid):
        for x in range(grid):
            tile = y * grid + x
            if x + 1 < grid:
                right[tile, tile + 1] = 5.0
            if y + 1 < grid:
                down[tile, tile + grid] = 5.0
    np.fill_diagonal(right, -np.inf)
    np.fill_diagonal(down, -np.inf)
    return planted, right, down


def g0a(args: argparse.Namespace) -> None:
    seed_all()
    began = time.perf_counter()
    # Exact delta verification on every pair of a deterministic 6x6 shuffled board.
    planted6, right6, down6 = planted_scores(6)
    board6 = planted6.copy()
    rng = np.random.default_rng(SEED)
    rng.shuffle(board6)
    maximum_error = 0.0
    for a in range(35):
        for b in range(a + 1, 36):
            delta = swap_delta(board6, a, b, right6, down6, 6)
            old = grid_objective(board6, right6, down6, 6)
            trial = board6.copy()
            apply_swap(trial, a, b)
            maximum_error = max(maximum_error, abs((grid_objective(trial, right6, down6, 6) - old) - delta))
    # A planted 24x24 board with one nonlocal swap must be repaired exactly.
    planted, right, down = planted_scores(GRID)
    disturbed = planted.copy()
    pair = (17, N - 19)
    apply_swap(disturbed, *pair)
    repaired_a, info_a = polish(disturbed, right, down)
    repaired_b, info_b = polish(disturbed, right.copy(), down.copy())
    elapsed = time.perf_counter() - began
    report = {
        "experiment": "P17_EDSP_24",
        "gate": "G0a_exact_delta_synthetic",
        "six_by_six_max_delta_error": maximum_error,
        "planted_pair": pair,
        "planted_swap_recovered": bool(np.array_equal(repaired_a, planted)),
        "strict_bijection": True,
        "deterministic_sha": bool(info_a["output_sha256"] == info_b["output_sha256"]),
        "runtime_seconds": elapsed,
        "runtime_under_30_seconds": bool(elapsed < 30.0),
        "twenty_four_info": info_a,
        "p8_imported": False,
        "labels_used": False,
        "targets_opened": False,
        "passes_G0a": bool(maximum_error <= TOL_DELTA and np.array_equal(repaired_a, planted) and info_a["output_sha256"] == info_b["output_sha256"] and elapsed < 30.0),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p17_g0a_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G0a"]:
        raise RuntimeError("P17 G0a failed")


def score_matrices(candidates: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right, down = dense_rd(candidates, scores)
    right, down = np.asarray(right, dtype=np.float32), np.asarray(down, dtype=np.float32)
    np.fill_diagonal(right, -np.inf)
    np.fill_diagonal(down, -np.inf)
    return right, down


def shuffle_candidate_axes(candidates: np.ndarray, valid: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c, v, s = candidates.copy(), valid.copy(), scores.copy()
    for anchor in range(N):
        order = np.argsort(np.array([(anchor * 131 + slot * 17) % 257 for slot in range(candidates.shape[1])]))
        c[anchor], v[anchor], s[:, anchor] = candidates[anchor, order], valid[anchor, order], scores[:, anchor, order]
    return c, v, s


def canonical(right: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, float]:
    board, value = solve_buddies_from_scores(right, down, max_edges=MAX_EDGES, min_margin=0.0, repair_passes=2)
    board = np.asarray(board, dtype=np.int64).reshape(-1)
    validate(board)
    return board, float(value)


def g0b(args: argparse.Namespace) -> None:
    seed_all()
    train, _held = p13.source_lists(args.prepare_report)
    sources = sorted(train)[:4]
    began = time.perf_counter()
    rows: list[dict[str, object]] = []
    for source in sources:
        candidates, valid, scores = p13.load_score_cache(args.score_dir, source)
        right, down = score_matrices(candidates, scores)
        seed, seed_obj = canonical(right, down)
        board, info = polish(seed, right, down)
        sc, sv, ss = shuffle_candidate_axes(candidates, valid, scores)
        sr, sd = score_matrices(sc, ss)
        seed_s, _ = canonical(sr, sd)
        board_s, info_s = polish(seed_s, sr, sd)
        rows.append({"source": source, "seed_objective": seed_obj, "final_objective": info["final_objective"], "objective_delta": info["objective_delta"], "moves": len(info["moves"]), "delta_total_error": info["delta_total_error"], "candidate_axis_invariant": bool(np.array_equal(board, board_s) and info["output_sha256"] == info_s["output_sha256"])})
    elapsed = time.perf_counter() - began
    nondecreasing = all(float(row["objective_delta"]) >= -TOL_TOTAL for row in rows)
    strictly_better = any(float(row["objective_delta"]) > TOL_TOTAL for row in rows)
    report = {"experiment": "P17_EDSP_24", "gate": "G0b_four_FIT_cache", "sources": sources, "rows": rows, "invalid_decodes": 0, "elapsed_seconds": elapsed, "runtime_under_60_seconds": bool(elapsed < 60.0), "labels_used": False, "targets_opened": False, "p8_imported": False, "passes_G0b": bool(nondecreasing and strictly_better and elapsed < 60.0 and all(bool(row["candidate_axis_invariant"]) for row in rows))}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p17_g0b_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G0b"]:
        raise RuntimeError("P17 G0b failed")


def g1(args: argparse.Namespace) -> None:
    seed_all()
    train, _held = p13.source_lists(args.prepare_report)
    rows: list[dict[str, object]] = []
    for index, source in enumerate(train[:CHECKPOINT_COUNT], 1):
        candidates, _valid, scores = p13.load_score_cache(args.score_dir, source)
        right, down = score_matrices(candidates, scores)
        seed, _obj = canonical(right, down)
        board, info = polish(seed, right, down)
        target, _ = p12.load_labels(args.cache_dir, source)
        accuracy = float(np.mean(board == np.asarray(target, dtype=np.int64).reshape(-1)))
        row = {"source": source, "index": index, "accuracy": accuracy, "moves": len(info["moves"]), "objective_delta": info["objective_delta"]}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    baseline = float(json.loads(args.baseline_report.read_text(encoding="utf-8"))["baseline_held_accuracy"])
    mean_accuracy = float(np.mean([float(row["accuracy"]) for row in rows]))
    threshold = baseline + 0.0025
    report = {"experiment": "P17_EDSP_24", "gate": "G1_16_source_checkpoint", "mean_accuracy": mean_accuracy, "baseline_accuracy": baseline, "required_accuracy": threshold, "invalid_decodes": 0, "rows": rows, "labels_used": "existing_FIT_label_cache_only", "targets_opened": False, "cal_accessed": False, "dev_accessed": False, "held_accessed": False, "test_accessed": False, "p8_imported": False, "passes_G1_checkpoint": bool(mean_accuracy >= threshold)}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p17_g1_checkpoint_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G1_checkpoint"]:
        raise RuntimeError("P17 G1 failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("g0a", "g0b", "g1"), required=True)
    parser.add_argument("--score-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache"))
    parser.add_argument("--prepare-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json"))
    parser.add_argument("--baseline-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\p12_g1_report.json"))
    parser.add_argument("--work-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P17_exact_delta"))
    args = parser.parse_args()
    assert_p8_absent(args.score_dir, args.cache_dir, args.prepare_report, args.baseline_report, args.work_dir)
    if args.mode == "g0a":
        g0a(args)
    elif args.mode == "g0b":
        g0b(args)
    else:
        g1(args)


if __name__ == "__main__":
    main()
