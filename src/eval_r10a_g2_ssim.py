"""R10-A G2: paired raw-layout SSIM against canonical rank96 outputs.

This script may open targets only after the verified G1 report passes. It compares
the exact raw assemblies emitted from identical frozen rank96 score capture; no
R5, NLM, or test data are involved.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813")
G1 = ROOT / "R10_global_component_multistart" / "g1_frozen_layout" / "r10a_g1_report.json"
TARGETS = Path(r"E:\pazzle_data\train\targets")


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (480, 480):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def lower_95_normal(values: np.ndarray) -> float:
    """Conservative two-sided normal lower confidence bound for the paired mean."""
    if values.size < 2:
        return float(values.mean())
    return float(values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(values.size))


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g1-report", type=Path, default=G1)
    p.add_argument("--targets", type=Path, default=TARGETS)
    p.add_argument("--report", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    cfg = args()
    g1: Dict[str, Any] = json.loads(cfg.g1_report.read_text(encoding="utf-8"))
    if not bool(g1.get("passes_G1")):
        raise RuntimeError("R10-A G2 blocked: G1 did not pass")
    work = cfg.g1_report.parent
    rows: List[Dict[str, Any]] = []
    deltas: List[float] = []
    for g1_row in g1["rows"]:
        name = str(g1_row["name"])
        stem = Path(name).stem
        target = rgb(cfg.targets / name)
        canonical = rgb(work / f"{stem}_canonical_raw.png")
        r10a = rgb(work / f"{stem}_r10a_raw.png")
        base = float(sk_ssim(target, canonical, channel_axis=2, data_range=255))
        candidate = float(sk_ssim(target, r10a, channel_axis=2, data_range=255))
        delta = candidate - base
        rows.append({"name": name, "canonical_raw_ssim": base, "r10a_raw_ssim": candidate, "delta": delta, "objective_delta": g1_row["objective_delta"]})
        deltas.append(delta)
    values = np.asarray(deltas, dtype=np.float64)
    mean_delta = float(values.mean())
    lower = lower_95_normal(values)
    report = {
        "experiment": "R10-A_global_component_multistart",
        "gate": "G2_paired_raw_layout_SSIM",
        "metric": "skimage.metrics.structural_similarity(channel_axis=2,data_range=255)",
        "g1_report": str(cfg.g1_report),
        "rows": rows,
        "paired_mean_delta": mean_delta,
        "paired_lower_95_delta": lower,
        "passes_G2": bool(mean_delta > 0.0 and lower > 0.0),
        "decision": "advance_to_R10A_G3_R5NLM" if mean_delta > 0.0 and lower > 0.0 else "reject_R10A_before_R5NLM_test_submission",
        "targets_opened": True,
    }
    destination = cfg.report or (work / "r10a_g2_ssim_report.json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
