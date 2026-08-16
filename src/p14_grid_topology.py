"""P14 GTPP-24 — Grid-Topology Propagation and Projection.

Pre-registered in P14_PRE_REGISTRATION.md before this source file was created.
The harness only reads frozen P12 rank96 candidate score caches. It neither loads
P8 artifacts nor accesses CAL/DEV/test targets during G0/G1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candidate_rank import DOWN, LEFT, RIGHT, UP
from eval_seeded_qap import dense_rd
from solve_buddies import solve_buddies_from_scores
import p12_loop_consensus as p12

GRID = 24
N_TILES = GRID * GRID
SEED = 20260816
SOLVER_MAX_EDGES = 96
KS = (32, 64, 96)
ITERATIONS = (1, 2, 4, 8)


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def canonical_bytes(array: np.ndarray) -> bytes:
    value = np.ascontiguousarray(array)
    return str(value.dtype).encode() + b"\0" + repr(value.shape).encode() + b"\0" + value.tobytes()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(canonical_bytes(array)).hexdigest()


def physical_edge_masks(
    candidates: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical RIGHT and DOWN adjacency masks from frozen candidate lists."""
    n, width = candidates.shape
    if valid.shape != (n, width) or scores.shape != (4, n, width):
        raise ValueError("unexpected candidate, valid, or score shape")
    limit = min(int(k), width)
    right = np.zeros((n, n), dtype=bool)
    down = np.zeros((n, n), dtype=bool)
    for direction, matrix in ((RIGHT, right), (DOWN, down)):
        finite = valid[:, :limit] & np.isfinite(scores[direction, :, :limit])
        for anchor in range(n):
            targets = candidates[anchor, :limit][finite[anchor]]
            targets = targets[(targets >= 0) & (targets < n) & (targets != anchor)]
            matrix[anchor, targets] = True
    return right, down


def propagate_2x2(
    right: np.ndarray,
    down: np.ndarray,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]]:
    """Iteratively remove directed edges that cannot belong to an oriented 2x2 cell.

    A right edge a->b survives only when there are c,d with a->c DOWN,
    b->d DOWN, and c->d RIGHT.  The DOWN rule is the directional transpose.
    Matrix products implement existential support over all c,d without labels.
    """
    if right.shape != down.shape or right.ndim != 2 or right.shape[0] != right.shape[1]:
        raise ValueError("right and down must be equal square matrices")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    current_r = np.ascontiguousarray(right.astype(bool, copy=True))
    current_d = np.ascontiguousarray(down.astype(bool, copy=True))
    history: list[dict[str, int]] = []
    fixed_point = False
    for step in range(max_iterations):
        r_u8 = current_r.astype(np.uint8, copy=False)
        d_u8 = current_d.astype(np.uint8, copy=False)
        # support_r[a,b] iff exists c,d: D[a,c] & R[c,d] & D[b,d].
        support_r = ((d_u8 @ r_u8) @ d_u8.T) > 0
        # support_d[a,c] iff exists b,d: R[a,b] & D[b,d] & R[c,d].
        support_d = ((r_u8 @ d_u8) @ r_u8.T) > 0
        next_r = current_r & support_r
        next_d = current_d & support_d
        history.append(
            {
                "iteration": step + 1,
                "right_edges_before": int(current_r.sum()),
                "down_edges_before": int(current_d.sum()),
                "right_edges_after": int(next_r.sum()),
                "down_edges_after": int(next_d.sum()),
            }
        )
        if np.array_equal(next_r, current_r) and np.array_equal(next_d, current_d):
            current_r, current_d = next_r, next_d
            fixed_point = True
            break
        current_r, current_d = next_r, next_d
    return current_r, current_d, {"iterations_run": len(history), "fixed_point": fixed_point, "history": history}


def filter_scores(
    candidates: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray,
    k: int,
    iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep only top-k physical edges surviving topology propagation."""
    before_r, before_d = physical_edge_masks(candidates, valid, scores, k)
    after_r, after_d, details = propagate_2x2(before_r, before_d, iterations)
    n, width = candidates.shape
    limit = min(int(k), width)
    filtered = np.full_like(scores, -np.inf, dtype=np.float32)
    for direction in (UP, DOWN, LEFT, RIGHT):
        for anchor in range(n):
            for slot in range(limit):
                if not valid[anchor, slot] or not np.isfinite(scores[direction, anchor, slot]):
                    continue
                target = int(candidates[anchor, slot])
                if target < 0 or target >= n or target == anchor:
                    continue
                keep = False
                if direction == RIGHT:
                    keep = bool(after_r[anchor, target])
                elif direction == LEFT:
                    keep = bool(after_r[target, anchor])
                elif direction == DOWN:
                    keep = bool(after_d[anchor, target])
                elif direction == UP:
                    keep = bool(after_d[target, anchor])
                if keep:
                    filtered[direction, anchor, slot] = scores[direction, anchor, slot]
    info: dict[str, object] = {
        "k": int(limit),
        "iterations_requested": int(iterations),
        "right_edges_before": int(before_r.sum()),
        "down_edges_before": int(before_d.sum()),
        "right_edges_after": int(after_r.sum()),
        "down_edges_after": int(after_d.sum()),
        "score_finite_before": int(np.isfinite(scores[:, :, :limit]).sum()),
        "score_finite_after": int(np.isfinite(filtered).sum()),
        "propagation": details,
    }
    return filtered, info


def board_to_tile_slot(board: np.ndarray) -> np.ndarray:
    flat = np.asarray(board, dtype=np.int32).reshape(-1)
    if flat.size != N_TILES or not np.array_equal(np.sort(flat), np.arange(N_TILES, dtype=np.int32)):
        raise RuntimeError("canonical solver did not produce a 576-way permutation")
    target = np.empty(N_TILES, dtype=np.int32)
    target[flat] = np.arange(N_TILES, dtype=np.int32)
    return target


def decode(candidates: np.ndarray, filtered_scores: np.ndarray) -> tuple[np.ndarray, float]:
    if candidates.shape != (N_TILES, candidates.shape[1]) or filtered_scores.shape[1] != N_TILES:
        raise ValueError("P14 decode requires the canonical 576-tile cache")
    tensor_c = torch.from_numpy(np.ascontiguousarray(candidates.astype(np.int64, copy=False)))
    tensor_s = torch.from_numpy(np.ascontiguousarray(filtered_scores.astype(np.float32, copy=False)))
    right, down = dense_rd(tensor_c, tensor_s)
    board, objective = solve_buddies_from_scores(
        right.detach().cpu().numpy(),
        down.detach().cpu().numpy(),
        max_edges=SOLVER_MAX_EDGES,
    )
    return board_to_tile_slot(board), float(objective)


def shuffle_candidate_axis(
    candidates: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random(candidates.shape), axis=1, kind="stable")
    shuffled_c = np.take_along_axis(candidates, order, axis=1)
    shuffled_v = np.take_along_axis(valid, order, axis=1)
    shuffled_s = np.take_along_axis(scores, order[None, :, :], axis=2)
    return shuffled_c, shuffled_v, shuffled_s


def true_adjacency_masks(tile_to_slot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if tile_to_slot.shape != (N_TILES,) or not np.array_equal(np.sort(tile_to_slot), np.arange(N_TILES, dtype=np.int32)):
        raise RuntimeError("target is not a strict tile-to-slot permutation")
    slot_to_tile = np.empty(N_TILES, dtype=np.int32)
    slot_to_tile[tile_to_slot] = np.arange(N_TILES, dtype=np.int32)
    right = np.zeros((N_TILES, N_TILES), dtype=bool)
    down = np.zeros((N_TILES, N_TILES), dtype=bool)
    for slot in range(N_TILES):
        tile = int(slot_to_tile[slot])
        row, col = divmod(slot, GRID)
        if col + 1 < GRID:
            right[tile, int(slot_to_tile[slot + 1])] = True
        if row + 1 < GRID:
            down[tile, int(slot_to_tile[slot + GRID])] = True
    return right, down


def directed_recall(candidate_r: np.ndarray, candidate_d: np.ndarray, target: np.ndarray) -> float:
    true_r, true_d = true_adjacency_masks(target)
    total = int(true_r.sum() + true_d.sum())
    hits = int((candidate_r & true_r).sum() + (candidate_d & true_d).sum())
    return float(hits / total) if total else 0.0


def g0a(args: argparse.Namespace) -> None:
    n = 5
    candidates = np.tile(np.arange(n, dtype=np.int32), (n, 1))
    valid = np.zeros((n, n), dtype=bool)
    scores = np.full((4, n, n), -np.inf, dtype=np.float32)

    def add(direction: int, anchor: int, target: int) -> None:
        valid[anchor, target] = True
        scores[direction, anchor, target] = 1.0

    # True 2x2: 0--1 / 2--3.  The right edge 0->4 has no down completion.
    add(RIGHT, 0, 1)
    add(RIGHT, 2, 3)
    add(DOWN, 0, 2)
    add(DOWN, 1, 3)
    add(RIGHT, 0, 4)
    before_r, before_d = physical_edge_masks(candidates, valid, scores, k=n)
    after_scores, info = filter_scores(candidates, valid, scores, k=n, iterations=8)
    after_r, after_d = physical_edge_masks(candidates, valid, after_scores, k=n)
    shuffled_c, shuffled_v, shuffled_s = shuffle_candidate_axis(candidates, valid, scores, seed=SEED + 14)
    shuffled_out, _ = filter_scores(shuffled_c, shuffled_v, shuffled_s, k=n, iterations=8)
    shuffled_r, shuffled_d = physical_edge_masks(shuffled_c, shuffled_v, shuffled_out, k=n)
    report = {
        "experiment": "P14_grid_topology_propagation",
        "gate": "G0a_synthetic_hard_2x2_contract",
        "true_edges_retained": bool(after_r[0, 1] and after_r[2, 3] and after_d[0, 2] and after_d[1, 3]),
        "dangling_false_removed": bool(before_r[0, 4] and not after_r[0, 4]),
        "candidate_order_invariant": bool(np.array_equal(after_r, shuffled_r) and np.array_equal(after_d, shuffled_d)),
        "finite_scores": bool(np.isfinite(after_scores).any()),
        "info": info,
        "p8_imported": False,
        "labels_used": False,
        "amp_used": False,
    }
    report["passes_G0a"] = bool(
        report["true_edges_retained"]
        and report["dangling_false_removed"]
        and report["candidate_order_invariant"]
        and report["finite_scores"]
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p14_g0a_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not report["passes_G0a"]:
        raise RuntimeError("P14 G0a contract failed")


def g0b(args: argparse.Namespace) -> None:
    train, _ = p12.source_lists(args.prepare_report)
    source = args.source or train[0]
    candidates, valid, scores = p12.load_score_cache(args.score_dir, source)
    target, _ = p12.load_labels(args.cache_dir, source)
    filtered, info = filter_scores(candidates, valid, scores, args.k, args.iterations)
    base_r, base_d = physical_edge_masks(candidates, valid, scores, args.k)
    out_r, out_d = physical_edge_masks(candidates, valid, filtered, args.k)
    base_recall = directed_recall(base_r, base_d, target)
    retained_recall = directed_recall(out_r, out_d, target)
    shuffled_c, shuffled_v, shuffled_s = shuffle_candidate_axis(candidates, valid, scores, SEED + 140)
    shuffled_filtered, _ = filter_scores(shuffled_c, shuffled_v, shuffled_s, args.k, args.iterations)
    shuffled_r, shuffled_d = physical_edge_masks(shuffled_c, shuffled_v, shuffled_filtered, args.k)
    pred, objective = decode(candidates, filtered)
    report = {
        "experiment": "P14_grid_topology_propagation",
        "gate": "G0b_one_FIT_frozen_cache",
        "source": source,
        "input_candidate_sha": array_sha(candidates),
        "input_valid_sha": array_sha(valid),
        "input_scores_sha": array_sha(scores),
        "k": int(args.k),
        "iterations": int(args.iterations),
        "candidate_order_invariant": bool(np.array_equal(out_r, shuffled_r) and np.array_equal(out_d, shuffled_d)),
        "baseline_directed_recall": base_recall,
        "retained_directed_recall": retained_recall,
        "retained_fraction_of_baseline": float(retained_recall / base_recall) if base_recall else 0.0,
        "strict_bijection": bool(np.array_equal(np.sort(pred), np.arange(N_TILES, dtype=np.int32))),
        "decoder_objective": objective,
        "filtered_score_sha": array_sha(filtered),
        "propagation": info,
        "targets_opened": "cached_FIT_labels_after_frozen_score_cache",
        "cal_target_opened": False,
        "dev_targets_opened": False,
        "test_accessed": False,
        "p8_imported": False,
        "amp_used": False,
    }
    report["passes_G0b"] = bool(
        report["candidate_order_invariant"]
        and report["strict_bijection"]
        and base_recall > 0.0
        and report["retained_fraction_of_baseline"] >= 0.95
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p14_g0b_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not report["passes_G0b"]:
        raise RuntimeError("P14 G0b contract failed")


def evaluate_sources(
    sources: list[str],
    args: argparse.Namespace,
    k: int,
    iterations: int,
) -> tuple[float, int]:
    values: list[float] = []
    invalid = 0
    for index, source in enumerate(sources):
        candidates, valid, scores = p12.load_score_cache(args.score_dir, source)
        filtered, _ = filter_scores(candidates, valid, scores, k, iterations)
        target, _ = p12.load_labels(args.cache_dir, source)
        try:
            pred, _ = decode(candidates, filtered)
            values.append(float(np.mean(pred == target)))
        except Exception:
            invalid += 1
            values.append(0.0)
        print(json.dumps({"source": source, "index": index + 1, "total": len(sources), "k": k, "iterations": iterations, "accuracy": values[-1], "invalid_so_far": invalid}, sort_keys=True), flush=True)
    return float(np.mean(values)), invalid


def g1(args: argparse.Namespace) -> None:
    train, held = p12.source_lists(args.prepare_report)
    grid: list[dict[str, object]] = []
    for k in KS:
        for iterations in ITERATIONS:
            accuracy, invalid = evaluate_sources(train, args, k, iterations)
            row = {"k": k, "iterations": iterations, "train_accuracy": accuracy, "invalid_decodes": invalid}
            grid.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    valid_grid = [row for row in grid if int(row["invalid_decodes"]) == 0]
    if not valid_grid:
        raise RuntimeError("P14 G1 grid has no valid decode configuration")
    selected = sorted(valid_grid, key=lambda row: (-float(row["train_accuracy"]), int(row["k"]), int(row["iterations"])))[0]
    held_accuracy, held_invalid = evaluate_sources(held, args, int(selected["k"]), int(selected["iterations"]))
    baseline_payload = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    baseline = float(baseline_payload["baseline_held_accuracy"])
    report = {
        "experiment": "P14_grid_topology_propagation",
        "gate": "G1_calibrate128_held32",
        "grid": grid,
        "selected": selected,
        "baseline_held_accuracy": baseline,
        "held_accuracy": held_accuracy,
        "held_delta_pp_vs_rank96": (held_accuracy - baseline) * 100.0,
        "invalid_decodes": held_invalid,
        "decision": "PASS_to_CAL" if held_accuracy >= baseline + 0.03 and held_invalid == 0 else "REJECT_before_CAL",
        "cal_target_opened": False,
        "dev_targets_opened": False,
        "test_accessed": False,
        "p8_imported": False,
        "amp_used": False,
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p14_g1_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("g0a", "g0b", "g1"), required=True)
    parser.add_argument("--work-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P14_grid_topology"))
    parser.add_argument("--score-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache"))
    parser.add_argument("--prepare-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json"))
    parser.add_argument("--baseline-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\p12_g1_report.json"))
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    seed_all()
    if args.mode == "g0a":
        g0a(args)
    elif args.mode == "g0b":
        g0b(args)
    else:
        g1(args)


if __name__ == "__main__":
    main()
