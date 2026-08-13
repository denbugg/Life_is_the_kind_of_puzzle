"""R4 phase-1: direct SSIM value of frozen tile restoration under oracle train layout.

This is a capability diagnostic.  The train permutation comes only from the
pre-existing cache/perms.npz metadata, not target matching at runtime.  The
restorer receives only corrupted input tiles; clean targets are read post-hoc
for the SSIM metric.  No test directory is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity as ssim

from config import CACHE_DIR, TRAIN_INP, TRAIN_TGT
from imgio import from_frags, load, to_frags
from match_preprocess import apply_match_denoiser_np, load_match_denoiser


DEFAULT_WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R4_restoration")
DEFAULT_SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_permutations(cache_path: Path) -> dict[str, np.ndarray]:
    archive = np.load(cache_path, allow_pickle=True)
    names = archive["names"]
    inv = archive["inv"]
    if names.ndim != 1 or inv.ndim != 2 or len(names) != len(inv):
        raise RuntimeError("unexpected perms.npz schema")
    result = {str(name): np.asarray(order, dtype=np.int64) for name, order in zip(names.tolist(), inv)}
    if len(result) != len(names):
        raise RuntimeError("duplicate names in permutation cache")
    return result


def lower_confidence_bound(values: list[float]) -> float:
    if len(values) < 2:
        return values[0] if values else float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean() - 1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--partition", choices=("cal", "dev"), default="dev")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--tag", default="matchden")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.n < 2:
        parser.error("--n must be at least two for the registered lower-confidence gate")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    names = list(split["splits"][args.partition][: args.n])
    if len(names) != args.n:
        raise RuntimeError(f"requested {args.n}, found {len(names)} in split")
    cache_path = Path(CACHE_DIR) / "perms.npz"
    perms = load_permutations(cache_path)
    missing = [name for name in names if name not in perms]
    if missing:
        raise RuntimeError(f"missing cached permutation labels: {missing[:3]}")
    denoiser, checkpoint = load_match_denoiser(args.tag, device=args.device)
    if denoiser is None or checkpoint is None:
        raise FileNotFoundError("frozen MatchDenoiser checkpoint not found")
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for ordinal, name in enumerate(names, 1):
        dirty = np.ascontiguousarray(to_frags(load(str(Path(TRAIN_INP) / name))))
        target = load(str(Path(TRAIN_TGT) / name))
        order = perms[name]
        if sorted(order.tolist()) != list(range(len(order))):
            raise RuntimeError(f"non-bijective cached order for {name}")
        restored = apply_match_denoiser_np(dirty, denoiser, device=args.device)
        raw_canvas = from_frags(dirty[order])
        restored_canvas = from_frags(restored[order])
        raw_ssim = float(ssim(raw_canvas, target, channel_axis=2, data_range=255))
        restored_ssim = float(ssim(restored_canvas, target, channel_axis=2, data_range=255))
        row = {
            "name": name,
            "raw_oracle_ssim": raw_ssim,
            "restored_oracle_ssim": restored_ssim,
            "delta": restored_ssim - raw_ssim,
            "raw_l1": float(np.mean(np.abs(raw_canvas.astype(np.float32) - target.astype(np.float32))) / 255.0),
            "restored_l1": float(np.mean(np.abs(restored_canvas.astype(np.float32) - target.astype(np.float32))) / 255.0),
        }
        rows.append(row)
        deltas.append(row["delta"])
        print(json.dumps({"ordinal": ordinal, **row}), flush=True)
    report = {
        "experiment": "R4_frozen_matchden_oracle_order_ssim",
        "scope": "train-only source-disjoint held-out capability diagnostic; no test access; target is post-hoc metric only",
        "split_path": str(args.split),
        "split_sha256": sha256_file(args.split),
        "partition": args.partition,
        "permutation_cache": str(cache_path),
        "permutation_cache_sha256": sha256_file(cache_path),
        "denoiser_checkpoint": str(checkpoint),
        "denoiser_checkpoint_sha256": sha256_file(Path(checkpoint)),
        "n": len(rows),
        "rows": rows,
        "summary": {
            "raw_oracle_ssim_mean": float(np.mean([row["raw_oracle_ssim"] for row in rows])),
            "restored_oracle_ssim_mean": float(np.mean([row["restored_oracle_ssim"] for row in rows])),
            "mean_delta": float(np.mean(deltas)),
            "min_delta": float(np.min(deltas)),
            "lower_95_delta": lower_confidence_bound(deltas),
            "mean_raw_l1": float(np.mean([row["raw_l1"] for row in rows])),
            "mean_restored_l1": float(np.mean([row["restored_l1"] for row in rows])),
        },
    }
    summary = report["summary"]
    report["gate"] = {
        "condition": "mean_delta>0 and lower_95_delta>0",
        "passed": bool(summary["mean_delta"] > 0 and summary["lower_95_delta"] > 0),
        "decision": "advance_to_fixed_layout_gate" if summary["mean_delta"] > 0 and summary["lower_95_delta"] > 0 else "reject_R4_before_layout_work",
    }
    destination = args.report or (args.work / f"r4_oracle_{args.partition}_{args.n}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "gate": report["gate"], "report": str(destination)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
