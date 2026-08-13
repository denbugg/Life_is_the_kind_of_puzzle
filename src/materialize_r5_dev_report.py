"""Materialize a completed R5 rank96 DEV report from its append-only board log.

This is intentionally CPU-only: it does not rerun rank96 or R5.  It validates
that exactly the expected pinned split prefix is present, then turns already
emitted per-board JSON records into the same summary/gate format as the evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R5_restore_unet")
DEFAULT_SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
DEFAULT_CKPT = DEFAULT_WORK / "r5_capacity_fp32.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def lower_95(values: list[float]) -> float:
    values_np = np.asarray(values, dtype=np.float64)
    return float(values_np.mean() - 1.96 * values_np.std(ddof=1) / math.sqrt(len(values_np)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_WORK / "r5_rank96_layout_dev8.log")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--partition", choices=("cal", "dev"), default="dev")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--report", type=Path, default=DEFAULT_WORK / "r5_rank96_layout_dev8.json")
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    expected_names = list(split["splits"][args.partition][: args.n])
    rows = []
    for raw_line in args.log.read_text(encoding="utf-16", errors="strict").splitlines():
        start = raw_line.find('{"ordinal"')
        if start < 0:
            continue
        try:
            row = json.loads(raw_line[start:])
        except json.JSONDecodeError:
            continue
        if set(row) >= {"ordinal", "name", "raw_layout_ssim", "restored_layout_ssim", "delta", "board_sha256"}:
            rows.append(row)
    rows.sort(key=lambda row: row["ordinal"])
    if len(rows) != args.n:
        raise RuntimeError(f"expected exactly {args.n} board records, parsed {len(rows)}")
    if [row["ordinal"] for row in rows] != list(range(1, args.n + 1)):
        raise RuntimeError("board ordinal sequence is not exactly 1..n")
    if [row["name"] for row in rows] != expected_names:
        raise RuntimeError("log board names do not match pinned source-disjoint split prefix")

    deltas = [float(row["delta"]) for row in rows]
    summary = {
        "raw_layout_ssim_mean": float(np.mean([row["raw_layout_ssim"] for row in rows])),
        "restored_layout_ssim_mean": float(np.mean([row["restored_layout_ssim"] for row in rows])),
        "mean_delta": float(np.mean(deltas)),
        "min_delta": float(np.min(deltas)),
        "lower_95_delta": lower_95(deltas),
    }
    passed = bool(summary["mean_delta"] > 0 and summary["lower_95_delta"] > 0)
    report = {
        "experiment": "R5_RestoreNet_on_frozen_rank96_layout",
        "scope": "materialized from eight completed source-disjoint board records; frozen rank96 layout used input only; R5 operated on the assembled 480x480 layout; target used only for post-hoc SSIM; no test access",
        "source_log": str(args.log),
        "source_log_sha256": sha256(args.log),
        "split": str(args.split),
        "split_sha256": sha256(args.split),
        "partition": args.partition,
        "names": expected_names,
        "r5_checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
        "rows": rows,
        "summary": summary,
        "gate": {
            "condition": "mean restoration SSIM delta>0 and lower_95_delta>0 on unchanged frozen rank96 layouts",
            "passed": passed,
            "decision": "advance_R5_to_R4_comparison" if passed else "reject_R5_for_rank96_composition",
        },
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "gate": report["gate"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
