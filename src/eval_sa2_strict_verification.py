"""ORBIT-24 SA2 strict spatial-verification diagnostic.

Uses only a dirty shuffled training input and a supplied public-source candidate.
For each known source, compares the existing high-precision SIFT verifier against
one deterministic wrong source. Train targets are never loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bag_source_retrieval import verify_source_candidate  # noqa: E402


def cases(found_train: Path, inputs: Path) -> list[tuple[str, Path]]:
    selected: dict[str, Path] = {}
    for source in sorted(found_train.iterdir()):
        image_id = source.name[:10]
        if source.is_file() and (inputs / f"{image_id}.png").is_file():
            selected.setdefault(image_id, source)
    return sorted(selected.items())


def split(image_id: str) -> str:
    value = int.from_bytes(hashlib.sha256(image_id.encode()).digest()[:8], "big")
    return "heldout" if value % 5 == 0 else "calibration"


def evaluate(input_path: Path, source_path: Path) -> dict[str, object]:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"Unreadable source {source_path}")
    metrics, _reconstructed = verify_source_candidate(input_path, source)
    return dict(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--found-train", type=Path, required=True)
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--wrong-offset", type=int, default=37)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_cases = cases(args.found_train, args.train_inputs)
    if args.limit is not None:
        source_cases = source_cases[:args.limit]
    if len(source_cases) < 3:
        raise RuntimeError("Need at least three source-linked dirty inputs")

    rows: list[dict[str, object]] = []
    for index, (image_id, true_source) in enumerate(source_cases):
        input_path = args.train_inputs / f"{image_id}.png"
        wrong_source = source_cases[(index + args.wrong_offset) % len(source_cases)][1]
        for label, source in (("true", true_source), ("wrong", wrong_source)):
            result = evaluate(input_path, source)
            result.update({
                "image_id": image_id,
                "split": split(image_id),
                "candidate_label": label,
                "candidate_source": source.name,
            })
            rows.append(result)
        true = rows[-2]
        wrong = rows[-1]
        print(
            f"{index + 1}/{len(source_cases)} {image_id} split={split(image_id)} "
            f"true={bool(true.get('accepted'))} wrong={bool(wrong.get('accepted'))} "
            f"true_inliers={true.get('sift_ransac_inliers')} wrong_inliers={wrong.get('sift_ransac_inliers')}",
            flush=True,
        )

    fields = sorted({field for row in rows for field in row})
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def aggregate(partition: list[dict[str, object]]) -> dict[str, object]:
        truth = [row for row in partition if row["candidate_label"] == "true"]
        wrong = [row for row in partition if row["candidate_label"] == "wrong"]
        true_accept = sum(bool(row.get("accepted")) for row in truth)
        wrong_accept = sum(bool(row.get("accepted")) for row in wrong)
        precision = true_accept / (true_accept + wrong_accept) if true_accept + wrong_accept else 0.0
        return {
            "true_cases": len(truth),
            "wrong_cases": len(wrong),
            "true_accept_rate": true_accept / len(truth) if truth else 0.0,
            "wrong_accept_rate": wrong_accept / len(wrong) if wrong else 0.0,
            "balanced_pair_precision": precision,
            "true_accepted": true_accept,
            "wrong_accepted": wrong_accept,
        }

    heldout = [row for row in rows if row["split"] == "heldout"]
    report = {
        "experiment": "SA2_strict_spatial_verification_true_vs_wrong",
        "input_contract": "dirty shuffled train input plus one candidate source; no targets loaded",
        "cases": len(source_cases),
        "metrics": {
            "calibration": aggregate([row for row in rows if row["split"] == "calibration"]),
            "heldout": aggregate(heldout),
            "all": aggregate(rows),
        },
        "gate": {
            "heldout_true_accept_ge_0_70": aggregate(heldout)["true_accept_rate"] >= 0.70,
            "heldout_wrong_accept_le_0_05": aggregate(heldout)["wrong_accept_rate"] <= 0.05,
            "heldout_balanced_precision_ge_0_95": aggregate(heldout)["balanced_pair_precision"] >= 0.95,
        },
        "note": "Wrong candidates are deterministic public-source negatives, not the full deployment candidate pool. This checks SIFT verifier discrimination, while bag-retrieval OOF confidence checks candidate ranking.",
    }
    report["gate"]["sa2_strict_verification_pass"] = all(report["gate"].values())
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
