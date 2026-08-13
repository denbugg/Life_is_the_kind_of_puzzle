"""ORBIT-24 SA2: OOF confidence calibration for public-source bag retrieval.

The retrieval report is produced from dirty shuffled inputs and known public-source
index only. This script treats `rank == 1` as withheld ground truth strictly for
post-hoc evaluation. Per-fold decision thresholds use only the remaining folds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def choose_threshold(rows: list[dict[str, Any]], target_precision: float, min_accepted: int) -> tuple[float | None, dict[str, float]]:
    distances = np.array([float(row["top_distance"]) for row in rows], dtype=np.float64)
    correct = np.array([int(row["rank"]) == 1 for row in rows], dtype=bool)
    choices: list[tuple[int, float, float]] = []
    for threshold in np.unique(distances):
        accepted = distances <= threshold
        count = int(accepted.sum())
        if count < min_accepted:
            continue
        precision = float(correct[accepted].mean())
        if precision >= target_precision:
            choices.append((count, float(threshold), precision))
    if not choices:
        return None, {"calibration_accepted": 0.0, "calibration_precision": 0.0}
    count, threshold, precision = max(choices, key=lambda item: (item[0], item[2]))
    return threshold, {"calibration_accepted": float(count), "calibration_precision": precision}


def metrics(rows: list[dict[str, Any]], threshold: float | None) -> dict[str, float]:
    if threshold is None or not rows:
        return {"accepted": 0.0, "coverage": 0.0, "precision": 0.0, "correct_accepted": 0.0}
    distance = np.array([float(row["top_distance"]) for row in rows], dtype=np.float64)
    correct = np.array([int(row["rank"]) == 1 for row in rows], dtype=bool)
    accepted = distance <= threshold
    return {
        "accepted": float(accepted.sum()),
        "coverage": float(accepted.mean()),
        "precision": float(correct[accepted].mean()) if accepted.any() else 0.0,
        "correct_accepted": float((correct & accepted).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration-precision", type=float, default=0.98)
    parser.add_argument("--min-calibration-accepted", type=int, default=20)
    args = parser.parse_args()

    data = json.loads(args.benchmark.read_text(encoding="utf-8"))
    rows = [row for row in data.get("rows", []) if row.get("top_distance") is not None and row.get("rank") is not None]
    if not rows:
        raise RuntimeError("No usable retrieval rows")
    folds = sorted({int(row["fold"]) for row in rows})
    prediction_rows: list[dict[str, Any]] = []
    per_fold: list[dict[str, Any]] = []
    for fold in folds:
        train = [row for row in rows if int(row["fold"]) != fold]
        heldout = [row for row in rows if int(row["fold"]) == fold]
        threshold, calibration = choose_threshold(train, args.calibration_precision, args.min_calibration_accepted)
        heldout_metric = metrics(heldout, threshold)
        per_fold.append({"fold": fold, "threshold": threshold, **calibration, **heldout_metric})
        for row in heldout:
            copied = dict(row)
            copied["accepted_by_oof_threshold"] = bool(threshold is not None and float(row["top_distance"]) <= threshold)
            copied["oof_threshold"] = threshold
            prediction_rows.append(copied)

    accepted = [row for row in prediction_rows if row["accepted_by_oof_threshold"]]
    correct = [row for row in accepted if int(row["rank"]) == 1]
    coverage = len(accepted) / len(prediction_rows)
    precision = len(correct) / len(accepted) if accepted else 0.0
    summary = {
        "experiment": "SA2_oof_retrieval_distance_confidence",
        "input_contract": "cached event-grouped retrieval predictions from dirty shuffled inputs and public-source index; rank is post-hoc only",
        "candidate_rows": len(prediction_rows),
        "base_top1_precision": float(np.mean([int(row["rank"]) == 1 for row in prediction_rows])),
        "calibration_precision_target": args.calibration_precision,
        "oof_acceptance": {
            "accepted": len(accepted),
            "coverage": coverage,
            "precision": precision,
            "correct_accepted": len(correct),
        },
        "folds": per_fold,
        "gates": {
            "precision_ge_0_95": precision >= 0.95,
            "coverage_ge_0_50": coverage >= 0.50,
            "sa2_retrieval_confidence_pass": precision >= 0.95 and coverage >= 0.50,
        },
        "limitations": "This gate calibrates retrieval confidence; it does not replace strict spatial SIFT verification nor measure cross-catalogue source coverage.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
