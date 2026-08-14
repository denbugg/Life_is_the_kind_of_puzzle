"""R10-A G1: multistart global component packing on unchanged rank96 score matrices.

The evaluator intentionally calls canonical ``infer_rank96.infer_one`` once for
each pinned raw DEV input and captures the exact dense R/D matrices passed to its
unchanged deterministic solver tail.  R10-A consumes those same in-memory arrays
with a multistart packer; it neither recomputes candidates nor accesses targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import infer_rank96 as rank96
from solve_buddies import solve_buddies_multistart_from_scores

ROOT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813")
SPLIT = ROOT / "PGA1_set_slot" / "source_disjoint_split_v1.json"
INPUTS = Path(r"E:\pazzle_data\train\inputs")
WORK = ROOT / "R10_global_component_multistart"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_dev_names(split: Path, count: int) -> List[str]:
    payload = json.loads(split.read_text(encoding="utf-8"))
    dev = sorted(payload["splits"]["dev"])
    if len(dev) < count:
        raise RuntimeError(f"pinned DEV manifest has {len(dev)} names, requires {count}")
    return dev[:count]


def make_config(device: str) -> rank96.InferenceConfig:
    checkpoints = rank96._default_checkpoints()
    return rank96.InferenceConfig(
        input_dir=INPUTS,
        output_dir=WORK / "g1_frozen_layout" / "unused_canonical_output",
        output_zip=None,
        ranker_checkpoint=checkpoints["ranker"],
        affinity_primary_checkpoint=checkpoints["affinity_primary"],
        affinity_secondary_checkpoint=checkpoints["affinity_secondary"],
        device=device,
        expected_count=0,
        pair_batch=4096,
    )


def capture_canonical_scores(image: np.ndarray, models: rank96.LoadedModels, pair_batch: int) -> Tuple[rank96.InferredImage, np.ndarray, np.ndarray]:
    original = rank96.solve_dense_tiles
    captured: Dict[str, np.ndarray] = {}

    def capture(tiles: np.ndarray, right: np.ndarray, down: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, float]:
        captured["right"] = np.ascontiguousarray(right, dtype=np.float32).copy()
        captured["down"] = np.ascontiguousarray(down, dtype=np.float32).copy()
        return original(tiles, right, down, **kwargs)

    rank96.solve_dense_tiles = capture
    try:
        inferred = rank96.infer_one(image, models, pair_batch=pair_batch)
    finally:
        rank96.solve_dense_tiles = original
    if set(captured) != {"right", "down"}:
        raise RuntimeError("R10 failed to capture canonical dense R/D scores")
    return inferred, captured["right"], captured["down"]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--pair-batch", type=int, default=4096)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--inputs", type=Path, default=INPUTS)
    p.add_argument("--work", type=Path, default=WORK / "g1_frozen_layout")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--restarts", type=int, default=32)
    p.add_argument("--max-edges", type=int, default=96)
    p.add_argument("--seed", type=int, default=20260814)
    return p.parse_args()


def main() -> None:
    cfg = args()
    if cfg.count != 8 or cfg.restarts != 32 or cfg.max_edges != 96:
        raise ValueError("R10-A G1 is pre-registered at count=8, restarts=32, max_edges=96")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg.work.mkdir(parents=True, exist_ok=True)
    report_path = cfg.report or cfg.work / "r10a_g1_report.json"
    names = pinned_dev_names(cfg.split, cfg.count)
    config = make_config(cfg.device)
    resolved_device = torch.device(cfg.device)
    rank96._set_deterministic_runtime(cfg.seed, resolved_device)
    models = rank96.load_models(config, resolved_device)
    rows: List[Dict[str, Any]] = []
    for index, name in enumerate(names):
        input_path = cfg.inputs / name
        image = rank96.load_rgb_strict(input_path)
        tiles = rank96.split_upright_tiles(image)
        inferred, right, down = capture_canonical_scores(image, models, cfg.pair_batch)
        optimized_board, optimized_objective = solve_buddies_multistart_from_scores(
            right, down, max_edges=cfg.max_edges, min_margin=0.0,
            repair_passes=0, restarts=cfg.restarts, seed=cfg.seed + index,
            temperature=0.03, order_jitter=0.25,
        )
        optimized_board = rank96._assert_board(optimized_board)
        if not np.isfinite(optimized_objective):
            raise RuntimeError("R10 optimized objective is non-finite")
        canonical_assembled = rank96.assemble_upright_tiles(tiles, inferred.board)
        optimized_assembled = rank96.assemble_upright_tiles(tiles, optimized_board)
        np.save(cfg.work / f"{Path(name).stem}_canonical_board.npy", inferred.board)
        np.save(cfg.work / f"{Path(name).stem}_r10a_board.npy", optimized_board)
        rank96._atomic_write_png(cfg.work / f"{Path(name).stem}_canonical_raw.png", canonical_assembled)
        rank96._atomic_write_png(cfg.work / f"{Path(name).stem}_r10a_raw.png", optimized_assembled)
        row = {
            "name": name,
            "candidate_ids_sha256": inferred.candidate_ids_sha256,
            "raw_scores_sha256": inferred.raw_scores_sha256,
            "shared_score_capture": True,
            "canonical_objective": float(inferred.objective),
            "r10a_objective": float(optimized_objective),
            "objective_delta": float(optimized_objective - inferred.objective),
            "canonical_bijection": bool(np.array_equal(np.sort(inferred.board), np.arange(576))),
            "r10a_bijection": bool(np.array_equal(np.sort(optimized_board), np.arange(576))),
            "targets_opened": False,
        }
        if not row["canonical_bijection"] or not row["r10a_bijection"]:
            raise RuntimeError(f"invalid board bijection for {name}")
        rows.append(row)
        print(json.dumps(row), flush=True)
    deltas = np.asarray([row["objective_delta"] for row in rows], dtype=np.float64)
    report = {
        "experiment": "R10-A_global_component_multistart",
        "gate": "G1_frozen_rank96_objective",
        "parameters": {"max_edges": cfg.max_edges, "restarts": cfg.restarts, "repair_passes": 0, "temperature": 0.03, "order_jitter": 0.25, "variant": "component_packing_only"},
        "manifest": {"path": str(cfg.split), "sha256": sha256_file(cfg.split), "pinned_dev_names": names},
        "rows": rows,
        "mean_objective_delta": float(deltas.mean()),
        "min_objective_delta": float(deltas.min()),
        "all_shared_score_capture": bool(all(row["shared_score_capture"] for row in rows)),
        "all_targets_opened_false": bool(all(not row["targets_opened"] for row in rows)),
        "passes_G1": bool(deltas.mean() > 0.0 and all(row["canonical_bijection"] and row["r10a_bijection"] for row in rows)),
        "decision": "advance_to_R10A_G2_raw_paired_SSIM" if deltas.mean() > 0.0 else "reject_R10A_before_targets",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
