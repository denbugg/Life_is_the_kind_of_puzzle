"""P16 BCA-24 -- Deterministic Bounded Component Beam Assembly.

Pre-registered in P16_PRE_REGISTRATION.md before this file was created.
Reads only frozen P12 score caches during G0b/G1 and existing FIT labels only for
post-hoc G1 accuracy. No target PNGs, CAL, DEV, held or test files are touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_seeded_qap import dense_rd
from solve_buddies import (
    GRID,
    NFRAG,
    _fill_board,
    _repair,
    _shift_score,
    build_buddies_components,
    objective,
    solve_buddies_from_scores,
)
import p12_loop_consensus as p12
import p13_component_pose as p13

N_TILES = GRID * GRID
BEAM_WIDTH = 4
TOP_OFFSETS = 4
MAX_EDGES = 96
SEED = 20260817
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


def assert_p8_absent(*paths: Path) -> None:
    if "p8" in "\n".join(str(p).lower() for p in paths):
        raise RuntimeError("P8 artifacts are prohibited for P16")


def validate(place: np.ndarray) -> None:
    value = np.asarray(place, dtype=np.int64).reshape(-1)
    if value.shape != (N_TILES,) or np.any(value < 0) or np.any(value >= N_TILES) or np.unique(value).size != N_TILES:
        raise RuntimeError("not a strict 576-way permutation")


def normalize_component(component: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
    min_y = min(y for y, _x in component.values())
    min_x = min(x for _y, x in component.values())
    return {int(tile): (int(y - min_y), int(x - min_x)) for tile, (y, x) in component.items()}


def component_key(component: dict[int, tuple[int, int]]) -> tuple[int, int]:
    return (-len(component), min(component))


def legal_offsets(component: dict[int, tuple[int, int]], board: np.ndarray, right: np.ndarray, down: np.ndarray) -> list[tuple[float, int, int]]:
    max_y = max(y for y, _x in component.values())
    max_x = max(x for _y, x in component.values())
    choices: list[tuple[float, int, int]] = []
    for sy in range(GRID - max_y):
        for sx in range(GRID - max_x):
            if any(board[y + sy, x + sx] >= 0 for y, x in component.values()):
                continue
            value = float(_shift_score(component, board, right, down, sy, sx))
            choices.append((value, sy, sx))
    choices.sort(key=lambda item: (-item[0], item[1], item[2]))
    return choices[:TOP_OFFSETS]


def place_component(board: np.ndarray, component: dict[int, tuple[int, int]], sy: int, sx: int) -> tuple[np.ndarray, set[int]]:
    out = board.copy()
    used: set[int] = set()
    for tile, (y, x) in component.items():
        yy, xx = y + sy, x + sx
        if out[yy, xx] >= 0:
            raise RuntimeError("overlap in legal component placement")
        out[yy, xx] = tile
        used.add(tile)
    return out, used


def partial_sha(board: np.ndarray) -> str:
    return array_sha(np.asarray(board, dtype=np.int64))


@dataclass(frozen=True)
class BeamState:
    board: np.ndarray
    used: frozenset[int]
    value: float


def beam_decode(right: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    right = np.asarray(right, dtype=np.float32)
    down = np.asarray(down, dtype=np.float32)
    components = build_buddies_components(right, down, max_edges=MAX_EDGES, min_margin=0.0)
    normalized = [normalize_component(dict(comp)) for comp in components]
    normalized.sort(key=component_key)
    states = [BeamState(np.full((GRID, GRID), -1, dtype=np.int64), frozenset(), 0.0)]
    placements_evaluated = 0
    for component in normalized:
        expanded: list[BeamState] = []
        for state in states:
            choices = legal_offsets(component, state.board, right, down)
            placements_evaluated += len(choices)
            if not choices:
                expanded.append(state)
                continue
            for incremental, sy, sx in choices:
                board, used = place_component(state.board, component, sy, sx)
                expanded.append(BeamState(board, state.used.union(used), state.value + incremental))
        # A deterministic global tie-break preserves reproducibility even when local scores tie.
        expanded.sort(key=lambda state: (-state.value, partial_sha(state.board)))
        states = expanded[:BEAM_WIDTH]
        if not states:
            raise RuntimeError("beam became empty")
    candidates: list[tuple[float, str, np.ndarray]] = []
    for state in states:
        board = state.board.copy()
        used = set(state.used)
        place = _fill_board(board, used, right, down)
        place, value = _repair(place, right, down, passes=1, pool=96)
        validate(place)
        candidates.append((float(value), array_sha(place), np.asarray(place, dtype=np.int64)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    value, digest, place = candidates[0]
    info: dict[str, object] = {
        "beam_width": BEAM_WIDTH,
        "top_offsets": TOP_OFFSETS,
        "component_count": len(normalized),
        "placements_evaluated": placements_evaluated,
        "beam_final_size": len(states),
        "decoder_objective": value,
        "output_sha256": digest,
        "invalid": False,
    }
    return place, info


def canonical_decode(right: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, float]:
    place, value = solve_buddies_from_scores(right, down, max_edges=MAX_EDGES, min_margin=0.0, repair_passes=2)
    place = np.asarray(place, dtype=np.int64).reshape(-1)
    validate(place)
    return place, float(value)


def synthetic_scores() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planted = np.arange(N_TILES, dtype=np.int64)
    right = np.full((N_TILES, N_TILES), -3.0, dtype=np.float32)
    down = np.full((N_TILES, N_TILES), -3.0, dtype=np.float32)
    for y in range(GRID):
        for x in range(GRID):
            tile = y * GRID + x
            if x + 1 < GRID:
                right[tile, tile + 1] = 5.0
            if y + 1 < GRID:
                down[tile, tile + GRID] = 5.0
    np.fill_diagonal(right, -np.inf)
    np.fill_diagonal(down, -np.inf)
    return planted, right, down


def g0a(args: argparse.Namespace) -> None:
    seed_all()
    planted, right, down = synthetic_scores()
    began = time.perf_counter()
    place_a, info_a = beam_decode(right, down)
    place_b, info_b = beam_decode(right.copy(), down.copy())
    elapsed = time.perf_counter() - began
    report = {
        "experiment": "P16_BCA_24",
        "gate": "G0a_synthetic_contract",
        "exact_planted_recovery": bool(np.array_equal(place_a, planted)),
        "strict_bijection": True,
        "deterministic_sha": bool(info_a["output_sha256"] == info_b["output_sha256"]),
        "elapsed_seconds": elapsed,
        "runtime_under_90_seconds": bool(elapsed < 90.0),
        "info": info_a,
        "p8_imported": False,
        "labels_used": False,
        "targets_opened": False,
        "passes_G0a": bool(np.array_equal(place_a, planted) and info_a["output_sha256"] == info_b["output_sha256"] and elapsed < 90.0),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p16_g0a_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G0a"]:
        raise RuntimeError("P16 G0a failed")


def score_matrices(candidates: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right, down = dense_rd(candidates, scores)
    right = np.asarray(right, dtype=np.float32)
    down = np.asarray(down, dtype=np.float32)
    np.fill_diagonal(right, -np.inf)
    np.fill_diagonal(down, -np.inf)
    return right, down


def shuffle_candidate_axes(candidates: np.ndarray, valid: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_c, out_v, out_s = candidates.copy(), valid.copy(), scores.copy()
    width = candidates.shape[1]
    for anchor in range(N_TILES):
        order = np.argsort(np.array([(anchor * 131 + slot * 17) % 257 for slot in range(width)]))
        out_c[anchor] = candidates[anchor, order]
        out_v[anchor] = valid[anchor, order]
        out_s[:, anchor] = scores[:, anchor, order]
    return out_c, out_v, out_s


def g0b_sources(args: argparse.Namespace) -> list[str]:
    train, _held = p13.source_lists(args.prepare_report)
    return sorted(train)[:4]


def g0b(args: argparse.Namespace) -> None:
    seed_all()
    began = time.perf_counter()
    rows: list[dict[str, object]] = []
    for source in g0b_sources(args):
        candidates, valid, scores = p13.load_score_cache(args.score_dir, source)
        right, down = score_matrices(candidates, scores)
        seed_place, seed_objective = canonical_decode(right, down)
        place, info = beam_decode(right, down)
        sc, sv, ss = shuffle_candidate_axes(candidates, valid, scores)
        sr, sd = score_matrices(sc, ss)
        shuffled_place, shuffled_info = beam_decode(sr, sd)
        rows.append({
            "source": source,
            "seed_objective": seed_objective,
            "final_objective": info["decoder_objective"],
            "objective_delta": float(info["decoder_objective"]) - seed_objective,
            "strict_bijection": True,
            "candidate_axis_invariant": bool(np.array_equal(place, shuffled_place) and info["output_sha256"] == shuffled_info["output_sha256"]),
            "output_sha256": info["output_sha256"],
            "placements_evaluated": info["placements_evaluated"],
        })
    elapsed = time.perf_counter() - began
    nonworse = sum(float(row["objective_delta"]) >= -1e-6 for row in rows)
    report = {
        "experiment": "P16_BCA_24",
        "gate": "G0b_four_FIT_cache_fast_futility",
        "sources": [row["source"] for row in rows],
        "rows": rows,
        "nonworse_objective_boards": nonworse,
        "invalid_decodes": 0,
        "elapsed_seconds": elapsed,
        "wall_time_under_300_seconds": bool(elapsed < 300.0),
        "labels_used": False,
        "targets_opened": False,
        "p8_imported": False,
        "passes_G0b": bool(nonworse >= 3 and elapsed < 300.0 and all(bool(row["candidate_axis_invariant"]) for row in rows)),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p16_g0b_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G0b"]:
        raise RuntimeError("P16 G0b failed")


def g1(args: argparse.Namespace) -> None:
    seed_all()
    train, _held = p13.source_lists(args.prepare_report)
    rows: list[dict[str, object]] = []
    for index, source in enumerate(train[:CHECKPOINT_COUNT], start=1):
        candidates, _valid, scores = p13.load_score_cache(args.score_dir, source)
        right, down = score_matrices(candidates, scores)
        place, info = beam_decode(right, down)
        target, _ = p12.load_labels(args.cache_dir, source)
        accuracy = float(np.mean(place == np.asarray(target, dtype=np.int64).reshape(-1)))
        row = {"source": source, "index": index, "accuracy": accuracy, "objective": info["decoder_objective"]}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    baseline = float(json.loads(args.baseline_report.read_text(encoding="utf-8"))["baseline_held_accuracy"])
    mean_accuracy = float(np.mean([float(row["accuracy"]) for row in rows]))
    threshold = baseline + 0.0025
    report = {
        "experiment": "P16_BCA_24",
        "gate": "G1_16_source_checkpoint",
        "mean_accuracy": mean_accuracy,
        "baseline_accuracy": baseline,
        "required_accuracy": threshold,
        "invalid_decodes": 0,
        "rows": rows,
        "labels_used": "existing_FIT_label_cache_only",
        "targets_opened": False,
        "cal_accessed": False,
        "dev_accessed": False,
        "held_accessed": False,
        "test_accessed": False,
        "p8_imported": False,
        "passes_G1_checkpoint": bool(mean_accuracy >= threshold),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p16_g1_checkpoint_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G1_checkpoint"]:
        raise RuntimeError("P16 G1 checkpoint failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("g0a", "g0b", "g1"), required=True)
    parser.add_argument("--score-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache"))
    parser.add_argument("--prepare-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json"))
    parser.add_argument("--baseline-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\p12_g1_report.json"))
    parser.add_argument("--work-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P16_component_beam"))
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
