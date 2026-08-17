"""P18 CSED-24 -- Cached-Seed Exact-Delta evaluation.

Pre-registered in P18_PRE_REGISTRATION.md before this file was created.
Stage A uses scores only to materialize canonical seed artifacts. Stage B reuses
those artifacts and frozen scores only. Target PNGs and closed splits are never read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import p13_component_pose as p13
import p17_exact_delta as p17

SEED_COUNT = 4
STAGE_A_CAP = 180.0
STAGE_B_CAP = 30.0


def assert_p8_absent(*paths: Path) -> None:
    if "p8" in "\n".join(str(p).lower() for p in paths):
        raise RuntimeError("P8 artifacts are prohibited for P18")


def sources(args: argparse.Namespace) -> list[str]:
    train, _held = p13.source_lists(args.prepare_report)
    return sorted(train)[:SEED_COUNT]


def seed_path(seed_dir: Path, source: str) -> Path:
    return seed_dir / f"{Path(source).stem}.npz"


def materialize(args: argparse.Namespace) -> tuple[list[dict[str, object]], float, int]:
    """P18b: validate existing immutable seeds and materialize only missing sources."""
    args.seed_dir.mkdir(parents=True, exist_ok=True)
    expected = sources(args)
    existing = sorted(path.stem + ".png" for path in args.seed_dir.glob("*.npz"))
    allowed_existing = [Path(source).stem + ".png" for source in expected[:len(existing)]]
    if existing != allowed_existing:
        raise RuntimeError(f"unexpected partial seed set: {existing} vs {allowed_existing}")
    begun = time.perf_counter()
    rows: list[dict[str, object]] = []
    missing = 0
    for source in expected:
        path = seed_path(args.seed_dir, source)
        if path.exists():
            board, metadata, _right, _down = load_seed_checked(args, source)
            rows.append({"source": source, "seed_path": str(path), "resumed": True, **metadata})
            continue
        missing += 1
        candidates, valid, scores = p13.load_score_cache(args.score_dir, source)
        right, down = p17.score_matrices(candidates, scores)
        board, objective = p17.canonical(right, down)
        p17.validate(board)
        metadata = {
            "source": source,
            "board_sha256": p17.array_sha(board),
            "candidate_sha256": p13.array_sha(candidates),
            "valid_sha256": p13.array_sha(valid),
            "score_sha256": p13.array_sha(scores),
            "canonical_objective": objective,
        }
        # Never overwrite a pre-existing artifact; save exactly one missing source.
        np.savez_compressed(path, board=board.astype(np.int16), metadata=json.dumps(metadata, sort_keys=True))
        rows.append({"source": source, "seed_path": str(path), "resumed": False, **metadata})
    return rows, time.perf_counter() - begun, missing


def load_seed_checked(args: argparse.Namespace, source: str) -> tuple[np.ndarray, dict[str, object], np.ndarray, np.ndarray]:
    path = seed_path(args.seed_dir, source)
    if not path.exists():
        raise RuntimeError(f"missing immutable seed artifact: {path}")
    with np.load(path, allow_pickle=False) as raw:
        board = np.asarray(raw["board"], dtype=np.int64).reshape(-1)
        metadata = json.loads(str(np.asarray(raw["metadata"]).reshape(-1)[0]))
    p17.validate(board)
    candidates, valid, scores = p13.load_score_cache(args.score_dir, source)
    expected = {
        "candidate_sha256": p13.array_sha(candidates),
        "valid_sha256": p13.array_sha(valid),
        "score_sha256": p13.array_sha(scores),
    }
    if metadata.get("source") != source or any(metadata.get(k) != v for k, v in expected.items()):
        raise RuntimeError("seed artifact score SHA mismatch")
    if metadata.get("board_sha256") != p17.array_sha(board):
        raise RuntimeError("seed artifact board SHA mismatch")
    right, down = p17.score_matrices(candidates, scores)
    actual_objective = p17.grid_objective(board, right, down)
    if abs(actual_objective - float(metadata["canonical_objective"])) > p17.TOL_TOTAL:
        raise RuntimeError("seed artifact canonical objective mismatch")
    return board, metadata, right, down


def g0a(args: argparse.Namespace) -> None:
    p17_report = args.p17_g0a_report
    if not p17_report.exists():
        raise RuntimeError("P17 G0a persisted proof is required")
    payload = json.loads(p17_report.read_text(encoding="utf-8"))
    if not bool(payload.get("passes_G0a")):
        raise RuntimeError("P17 G0a proof did not pass")
    rows, elapsed, missing = materialize(args)
    valid = elapsed < 120.0 and missing == 1 and len(rows) == SEED_COUNT
    for row in rows:
        board, metadata, _right, _down = load_seed_checked(args, str(row["source"]))
        valid = valid and metadata["board_sha256"] == p17.array_sha(board)
    report = {
        "experiment": "P18b_CSED_24",
        "gate": "G0a_cached_seed_materialization",
        "p17_g0a_reused": True,
        "sources": [row["source"] for row in rows],
        "rows": rows,
        "elapsed_seconds": elapsed,
        "materialized_missing_count": missing,
        "runtime_under_120_seconds": elapsed < 120.0,
        "labels_used": False,
        "targets_opened": False,
        "p8_imported": False,
        "passes_G0a": bool(valid),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p18_g0a_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G0a"]:
        raise RuntimeError("P18 G0a failed")


def g0b(args: argparse.Namespace) -> None:
    begun = time.perf_counter()
    rows: list[dict[str, object]] = []
    for source in sources(args):
        board, metadata, right, down = load_seed_checked(args, source)
        out, info = p17.polish(board, right, down)
        p17.validate(out)
        objective_delta = float(info["final_objective"]) - float(metadata["canonical_objective"])
        rows.append({
            "source": source,
            "seed_sha256": metadata["board_sha256"],
            "output_sha256": info["output_sha256"],
            "canonical_objective": metadata["canonical_objective"],
            "final_objective": info["final_objective"],
            "objective_delta": objective_delta,
            "moves": len(info["moves"]),
            "delta_total_error": info["delta_total_error"],
        })
    elapsed = time.perf_counter() - begun
    exact = all(float(row["delta_total_error"]) <= p17.TOL_TOTAL for row in rows)
    nondecreasing = all(float(row["objective_delta"]) >= -p17.TOL_TOTAL for row in rows)
    better = any(float(row["objective_delta"]) > p17.TOL_TOTAL for row in rows)
    report = {
        "experiment": "P18b_CSED_24",
        "gate": "G0b_cached_seed_exact_delta",
        "sources": [row["source"] for row in rows],
        "rows": rows,
        "invalid_decodes": 0,
        "elapsed_seconds": elapsed,
        "runtime_under_60_seconds": elapsed < 60.0,
        "labels_used": False,
        "targets_opened": False,
        "p8_imported": False,
        "passes_G0b": bool(exact and nondecreasing and better and elapsed < 60.0),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p18_g0b_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G0b"]:
        raise RuntimeError("P18 G0b failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("g0a", "g0b"), required=True)
    parser.add_argument("--score-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache"))
    parser.add_argument("--prepare-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json"))
    parser.add_argument("--seed-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P18_cached_seeds"))
    parser.add_argument("--work-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P18_cached_seed_exact_delta"))
    parser.add_argument("--p17-g0a-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P17_exact_delta\p17_g0a_report.json"))
    args = parser.parse_args()
    assert_p8_absent(args.score_dir, args.prepare_report, args.seed_dir, args.work_dir, args.p17_g0a_report)
    if args.mode == "g0a":
        g0a(args)
    else:
        g0b(args)


if __name__ == "__main__":
    main()
