#!/usr/bin/env python3
"""Measure the historical S1 RGB/BGR NLM discrepancy on a frozen train panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import t as student_t

from aiijc_puzzle.pixel_tails import (
    apply_nlm_color,
    assemble_tiles,
    contest_ssim,
    recover_layout,
    split_tiles,
)
from aiijc_puzzle.protocol import compute_protocol_digest, select_manifest_records, sha256_file
from aiijc_puzzle.s1_anchor import canonical_historical_nlm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/interim/validation_manifest.json",
    )
    parser.add_argument("--inputs", type=Path, default=PROJECT_ROOT / "data/raw/train/inputs")
    parser.add_argument("--targets", type=Path, default=PROJECT_ROOT / "data/raw/train/targets")
    parser.add_argument("--split", choices=("calibration", "holdout"), default="holdout")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--descriptor-bins", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/s1-nlm-channel-audit/holdout48.json",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def proper_nlm_h10(image: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def paired(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        low = high = mean
    else:
        half = float(student_t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array)))
        low, high = mean - half, mean + half
    return {
        "mean": mean,
        "ci95_low": low,
        "ci95_high": high,
        "wins": int((array > 0).sum()),
        "ties": int((array == 0).sum()),
        "losses": int((array < 0).sum()),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest digest mismatch")
    selected = select_manifest_records(manifest, args.split, limit=args.limit)

    rows: list[dict[str, object]] = []
    for ordinal, record in enumerate(selected, start=1):
        filename = str(record["filename"])
        input_path = args.inputs / filename
        target_path = args.targets / filename
        if sha256_file(input_path) != record["input_sha256"]:
            raise ValueError(f"input hash mismatch: {filename}")
        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"target hash mismatch: {filename}")
        source = load_rgb(input_path)
        target = load_rgb(target_path)
        recovery = recover_layout(source, target, descriptor_bins=args.descriptor_bins)
        layout = assemble_tiles(split_tiles(source), recovery.slot_to_input)
        historical = canonical_historical_nlm(layout)
        proper_h10 = proper_nlm_h10(layout)
        proper_h9 = apply_nlm_color(layout, h=9).image
        scores = {
            "raw": contest_ssim(target, layout),
            "historical_direct_h10": contest_ssim(target, historical),
            "proper_rgb_h10": contest_ssim(target, proper_h10),
            "proper_rgb_h9": contest_ssim(target, proper_h9),
        }
        row = {"ordinal": ordinal, "filename": filename, **scores}
        rows.append(row)
        print(json.dumps(row), flush=True)

    keys = ("raw", "historical_direct_h10", "proper_rgb_h10", "proper_rgb_h9")
    means = {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
    report = {
        "schema": "aiijc-s1-nlm-channel-audit-v1",
        "scope": (
            "target-assisted layouts from frozen train split; diagnostic only; no test access; "
            "the target is used for layout recovery and post-hoc SSIM"
        ),
        "protocol_digest": manifest["protocol_digest"],
        "split": args.split,
        "limit": len(rows),
        "descriptor_bins": args.descriptor_bins,
        "means": means,
        "paired": {
            "proper_h10_minus_historical_h10": paired(
                [float(row["proper_rgb_h10"]) - float(row["historical_direct_h10"]) for row in rows]
            ),
            "proper_h9_minus_historical_h10": paired(
                [float(row["proper_rgb_h9"]) - float(row["historical_direct_h10"]) for row in rows]
            ),
            "proper_h9_minus_proper_h10": paired(
                [float(row["proper_rgb_h9"]) - float(row["proper_rgb_h10"]) for row in rows]
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"means": means, "paired": report["paired"], "output": str(args.output)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
