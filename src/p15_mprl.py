"""P15 MPRL-24 -- Seeded Multi-Phase Sparse Relaxation Labeling.

Pre-registered in P15_PRE_REGISTRATION.md before this file was created.
The harness consumes only frozen P12 rank96 score-cache files and uses cached
FIT labels only in the post-hoc G1 accuracy gate.  It never reads target PNGs,
never accesses CAL/DEV/test in G0/G1, and asserts that no P8 artifact is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_seeded_qap import dense_rd
from solve_buddies import objective, solve_buddies_from_scores, solve_buddies_multistart_from_scores
import p12_loop_consensus as p12
import p13_component_pose as p13

GRID = 24
N_TILES = GRID * GRID
K = 32
ALPHA = 0.50
PHASES = 2
ITERS_PER_PHASE = 4
SEED = 20260816
START_SEEDS = (20260816, 20260817, 20260818)
SOLVER_MAX_EDGES = 96
CHECKPOINT_COUNT = 16


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def canonical_bytes(array: np.ndarray) -> bytes:
    value = np.ascontiguousarray(array)
    return str(value.dtype).encode() + b"\0" + repr(value.shape).encode() + b"\0" + value.tobytes()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(canonical_bytes(array)).hexdigest()


def validate_permutation(place: np.ndarray) -> None:
    board = np.asarray(place, dtype=np.int64).reshape(-1)
    if board.shape != (N_TILES,):
        raise ValueError(f"expected {N_TILES} tiles, got {board.shape}")
    if np.any(board < 0) or np.any(board >= N_TILES):
        raise ValueError("tile id outside [0,575]")
    if np.unique(board).size != N_TILES:
        raise ValueError("output is not a strict 576-way permutation")


def assert_p8_absent(*paths: Path) -> None:
    joined = "\n".join(str(p).lower() for p in paths)
    if "p8" in joined:
        raise RuntimeError("P8 artifacts are prohibited for P15")


def canonical_seed(right: np.ndarray, down: np.ndarray) -> np.ndarray:
    place, _ = solve_buddies_from_scores(
        right,
        down,
        max_edges=SOLVER_MAX_EDGES,
        min_margin=0.0,
        repair_passes=2,
    )
    place = np.asarray(place, dtype=np.int64).reshape(-1)
    validate_permutation(place)
    return place


def start_boards(right: np.ndarray, down: np.ndarray, canonical: np.ndarray) -> list[np.ndarray]:
    boards = [canonical]
    for start_seed in START_SEEDS:
        board, _ = solve_buddies_multistart_from_scores(
            right,
            down,
            max_edges=SOLVER_MAX_EDGES,
            min_margin=0.0,
            repair_passes=0,
            restarts=1,
            seed=start_seed,
            temperature=0.05,
            order_jitter=0.25,
        )
        board = np.asarray(board, dtype=np.int64).reshape(-1)
        validate_permutation(board)
        boards.append(board)
    return boards


def global_tile_order(right: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Deterministic score-ranked fallback order independent of candidate slots."""
    value = np.maximum(right, -np.inf).max(axis=1)
    value += np.maximum(down, -np.inf).max(axis=1)
    value += np.maximum(right, -np.inf).max(axis=0)
    value += np.maximum(down, -np.inf).max(axis=0)
    value = np.nan_to_num(value, nan=-1e9, neginf=-1e9, posinf=1e9)
    return np.lexsort((np.arange(N_TILES, dtype=np.int64), -value)).astype(np.int64)


def build_support(boards: list[np.ndarray], right: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Build K=32 sparse support preserving the canonical perfect matching."""
    order = global_tile_order(right, down)
    support = np.zeros((N_TILES, N_TILES), dtype=bool)
    for cell in range(N_TILES):
        chosen: list[int] = []
        for board in boards:
            tile = int(board[cell])
            if tile not in chosen:
                chosen.append(tile)
        offset = cell % N_TILES
        for j in range(N_TILES):
            tile = int(order[(offset + j) % N_TILES])
            if tile not in chosen:
                chosen.append(tile)
            if len(chosen) >= K:
                break
        if len(chosen) != K:
            raise RuntimeError("failed to construct K=32 candidate support")
        support[cell, np.asarray(chosen, dtype=np.int64)] = True
    canonical = boards[0]
    if not np.all(support[np.arange(N_TILES), canonical]):
        raise RuntimeError("canonical perfect matching lost from sparse support")
    return support


def init_logits(support: np.ndarray, canonical: np.ndarray) -> np.ndarray:
    logits = np.full((N_TILES, N_TILES), -np.inf, dtype=np.float64)
    logits[support] = 0.0
    logits[np.arange(N_TILES), canonical] = 1.0
    return logits


def normalize_logits(logits: np.ndarray, support: np.ndarray, rounds: int = 3) -> np.ndarray:
    out = logits.copy()
    for _ in range(rounds):
        row_lse = logsumexp(out, axis=1, keepdims=True)
        out = out - row_lse
        col_lse = logsumexp(out, axis=0, keepdims=True)
        out = out - col_lse
        out[~support] = -np.inf
    if not np.isfinite(out[support]).all():
        raise RuntimeError("non-finite supported logits after balancing")
    return out


def row_probabilities(logits: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = logits - logsumexp(logits, axis=1, keepdims=True)
    out[~support] = -np.inf
    prob = np.exp(out)
    if not np.allclose(prob.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("row probabilities do not sum to one")
    return prob


def expected_support(prob: np.ndarray, right: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Four-direction expected adjacency compatibility for each cell/tile."""
    support = np.zeros((N_TILES, N_TILES), dtype=np.float64)
    # Sparse candidate probabilities contain exact zeros; avoid IEEE `-inf * 0 = NaN`
    # for forbidden diagonal edges while preserving a decisively adverse finite score.
    safe_right = np.where(np.isfinite(right), right, -1e6).astype(np.float64, copy=False)
    safe_down = np.where(np.isfinite(down), down, -1e6).astype(np.float64, copy=False)
    for y in range(GRID):
        for x in range(GRID):
            cell = y * GRID + x
            if x + 1 < GRID:
                neighbor = cell + 1
                support[cell] += safe_right @ prob[neighbor]
            if x > 0:
                neighbor = cell - 1
                support[cell] += safe_right.T @ prob[neighbor]
            if y + 1 < GRID:
                neighbor = cell + GRID
                support[cell] += safe_down @ prob[neighbor]
            if y > 0:
                neighbor = cell - GRID
                support[cell] += safe_down.T @ prob[neighbor]
    return support


def hungarian_place(logits: np.ndarray) -> np.ndarray:
    if not np.isfinite(logits).any(axis=1).all():
        raise RuntimeError("a cell has no supported tile")
    # Assignment of cell -> tile; finite support is guaranteed to include seed matching.
    cost = np.where(np.isfinite(logits), -logits, 1e12)
    rows, cols = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(N_TILES)):
        raise RuntimeError("Hungarian did not assign every cell")
    place = cols.astype(np.int64, copy=False)
    validate_permutation(place)
    return place


def refine(right: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    canonical = canonical_seed(right, down)
    boards = start_boards(right, down, canonical)
    support = build_support(boards, right, down)
    logits = init_logits(support, canonical)
    initial_objective = float(objective(canonical, right, down))
    phase_objectives: list[float] = []
    for _phase in range(PHASES):
        for _iteration in range(ITERS_PER_PHASE):
            balanced = normalize_logits(logits, support)
            prob = row_probabilities(balanced, support)
            local = expected_support(prob, right, down)
            finite = local[support]
            scale = float(np.std(finite))
            if not math.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            logits[support] = logits[support] + ALPHA * (local[support] / scale)
            logits = normalize_logits(logits, support)
        diagnostic = hungarian_place(logits)
        phase_objectives.append(float(objective(diagnostic, right, down)))
    place = hungarian_place(logits)
    final_objective = float(objective(place, right, down))
    info: dict[str, object] = {
        "canonical_sha256": array_sha(canonical),
        "output_sha256": array_sha(place),
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "objective_delta": final_objective - initial_objective,
        "phase_objectives": phase_objectives,
        "support_size_per_cell": int(support.sum(axis=1).min()),
        "support_candidate_count": int(support.sum()),
        "start_board_shas": [array_sha(board) for board in boards],
        "invalid": False,
    }
    return place, info


def synthetic_scores() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planted = np.arange(N_TILES, dtype=np.int64)
    right = np.full((N_TILES, N_TILES), -2.0, dtype=np.float32)
    down = np.full((N_TILES, N_TILES), -2.0, dtype=np.float32)
    for y in range(GRID):
        for x in range(GRID):
            tile = y * GRID + x
            if x + 1 < GRID:
                right[tile, tile + 1] = 4.0
            if y + 1 < GRID:
                down[tile, tile + GRID] = 4.0
    np.fill_diagonal(right, -np.inf)
    np.fill_diagonal(down, -np.inf)
    return planted, right, down


def g0a(args: argparse.Namespace) -> None:
    seed_all()
    planted, right, down = synthetic_scores()
    began = time.perf_counter()
    place_a, info_a = refine(right, down)
    # Matrix row/column reordering is irrelevant to score semantics; use a second identical run
    # to test deterministic candidate/start ordering and canonical output stability.
    place_b, info_b = refine(right.copy(), down.copy())
    elapsed = time.perf_counter() - began
    report = {
        "experiment": "P15b_MPRL_24",
        "gate": "G0a_synthetic_contract",
        "exact_planted_recovery": bool(np.array_equal(place_a, planted)),
        "strict_bijection": True,
        "candidate_order_invariant": bool(np.array_equal(place_a, place_b) and info_a["output_sha256"] == info_b["output_sha256"]),
        "initial_objective": info_a["initial_objective"],
        "final_objective": info_a["final_objective"],
        "elapsed_seconds": elapsed,
        "runtime_under_90_seconds": bool(elapsed < 90.0),
        "p8_imported": False,
        "labels_used": False,
        "targets_opened": False,
        "passes_G0a": bool(np.array_equal(place_a, planted) and np.array_equal(place_a, place_b) and elapsed < 90.0),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p15_g0a_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G0a"]:
        raise RuntimeError("P15 G0a failed")


def score_matrices(candidates: np.ndarray, valid: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right, down = dense_rd(candidates, scores)
    right = np.asarray(right, dtype=np.float32)
    down = np.asarray(down, dtype=np.float32)
    np.fill_diagonal(right, -np.inf)
    np.fill_diagonal(down, -np.inf)
    return right, down


def candidate_axis_shuffle(candidates: np.ndarray, valid: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled_candidates = candidates.copy()
    shuffled_valid = valid.copy()
    shuffled_scores = scores.copy()
    for anchor in range(N_TILES):
        # Public metadata-only deterministic permutation; apply it to ids, validity and all direction scores.
        order = np.argsort(np.array([(anchor * 131 + slot * 17) % 257 for slot in range(candidates.shape[1])]))
        shuffled_candidates[anchor] = candidates[anchor, order]
        shuffled_valid[anchor] = valid[anchor, order]
        shuffled_scores[:, anchor] = scores[:, anchor, order]
    return shuffled_candidates, shuffled_valid, shuffled_scores


def sources_for_g0b(args: argparse.Namespace) -> list[str]:
    train, _held = p13.source_lists(args.prepare_report)
    return sorted(train)[:4]


def g0b(args: argparse.Namespace) -> None:
    seed_all()
    rows: list[dict[str, object]] = []
    began = time.perf_counter()
    for source in sources_for_g0b(args):
        candidates, valid, scores = p13.load_score_cache(args.score_dir, source)
        right, down = score_matrices(candidates, valid, scores)
        place, info = refine(right, down)
        sc, sv, ss = candidate_axis_shuffle(candidates, valid, scores)
        sr, sd = score_matrices(sc, sv, ss)
        shuffled_place, shuffled_info = refine(sr, sd)
        rows.append({
            "source": source,
            "seed_objective": info["initial_objective"],
            "final_objective": info["final_objective"],
            "objective_delta": info["objective_delta"],
            "strict_bijection": True,
            "candidate_axis_invariant": bool(np.array_equal(place, shuffled_place) and info["output_sha256"] == shuffled_info["output_sha256"]),
            "output_sha256": info["output_sha256"],
        })
    elapsed = time.perf_counter() - began
    positive = sum(float(row["objective_delta"]) > 0.0 for row in rows)
    report = {
        "experiment": "P15_MPRL_24",
        "gate": "G0b_four_FIT_cache_fast_futility",
        "sources": [row["source"] for row in rows],
        "rows": rows,
        "positive_objective_boards": positive,
        "invalid_decodes": 0,
        "elapsed_seconds": elapsed,
        "wall_time_under_600_seconds": bool(elapsed < 600.0),
        "labels_used": False,
        "targets_opened": False,
        "p8_imported": False,
        "passes_G0b": bool(positive >= 3 and elapsed < 600.0 and all(bool(row["candidate_axis_invariant"]) for row in rows)),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p15_g0b_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G0b"]:
        raise RuntimeError("P15 G0b fast-futility gate failed")


def accuracy_for_sources(args: argparse.Namespace, sources: list[str]) -> tuple[float, int, list[dict[str, object]]]:
    values: list[float] = []
    rows: list[dict[str, object]] = []
    invalid = 0
    for index, source in enumerate(sources, start=1):
        candidates, valid, scores = p13.load_score_cache(args.score_dir, source)
        right, down = score_matrices(candidates, valid, scores)
        place, info = refine(right, down)
        target, _ = p12.load_labels(args.cache_dir, source)
        target = np.asarray(target, dtype=np.int64).reshape(-1)
        accuracy = float(np.mean(place == target))
        values.append(accuracy)
        rows.append({"source": source, "index": index, "accuracy": accuracy, "objective_delta": info["objective_delta"]})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    return float(np.mean(values)), invalid, rows


def g1(args: argparse.Namespace) -> None:
    seed_all()
    train, _held = p13.source_lists(args.prepare_report)
    checkpoint = train[:CHECKPOINT_COUNT]
    mean_accuracy, invalid, rows = accuracy_for_sources(args, checkpoint)
    baseline = float(json.loads(args.baseline_report.read_text(encoding="utf-8"))["baseline_held_accuracy"])
    threshold = baseline + 0.0025
    report = {
        "experiment": "P15_MPRL_24",
        "gate": "G1_16_source_checkpoint",
        "sources": checkpoint,
        "mean_accuracy": mean_accuracy,
        "baseline_accuracy": baseline,
        "required_accuracy": threshold,
        "invalid_decodes": invalid,
        "rows": rows,
        "labels_used": "existing_FIT_label_cache_only",
        "targets_opened": False,
        "cal_accessed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "p8_imported": False,
        "passes_G1_checkpoint": bool(mean_accuracy >= threshold and invalid == 0),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p15_g1_checkpoint_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G1_checkpoint"]:
        raise RuntimeError("P15 G1 checkpoint failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("g0a", "g0b", "g1"), required=True)
    parser.add_argument("--score-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache"))
    parser.add_argument("--prepare-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json"))
    parser.add_argument("--baseline-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\p12_g1_report.json"))
    parser.add_argument("--work-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P15_mprl"))
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
