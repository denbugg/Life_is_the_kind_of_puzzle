"""Render read-only RR96/CC96/CC192 validation panels for visual diagnosis.

This utility never selects a model or changes an experiment report.  It
replays the byte-pinned E12/E14 inputs, restores each assembled corrupted
canvas with the fixed NLM(10) tail, and writes labelled PNG panels to E:.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import e14_cc192_oracle as cc192
import eval_clean_score_oracle as e12
import eval_e14_cc192_discovery as e14
from imgio import assemble


HEADER = 34


def _panel(image: np.ndarray, label: str) -> np.ndarray:
    strip = np.zeros((HEADER, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        strip,
        label,
        (7, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((strip, np.ascontiguousarray(image, dtype=np.uint8)))


def _row_by_image(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(row["image"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("E:/pazzle_work/visual_audit_e14"),
    )
    args = parser.parse_args()
    out_dir = e14._require_e_drive(args.out_dir, label="visual audit output")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = e14.E14Paths(
        raw_cache_dir=e14.DEFAULT_RAW_CACHE_DIR,
        calibration_report=e14.DEFAULT_CALIBRATION_REPORT,
        e12_report=e14.DEFAULT_E12_REPORT,
        report=e14.DEFAULT_REPORT,
    )
    e12_report, _calibration, scenes = e14.load_verified_e12_inputs(paths)
    e14_report = e14._load_json(e14.DEFAULT_REPORT, label="E14 report")
    if e14_report.get("status") != "complete" or e14_report.get("stage") != "kill_cc192":
        raise RuntimeError("visual audit requires the completed fixed E14 report")

    rr_rows = e14._e12_rr_rows(e12_report)
    cc96_rows = _row_by_image(list(e12_report["rows"]["CC"]))
    cc192_rows = _row_by_image(list(e14_report["rows"]["CC192"]))
    clean_records = e14._clean_cache_records(e12_report)
    manifest: list[dict[str, object]] = []

    for scene in scenes:
        image = int(scene.image_id)
        rr_board, _rr_objective, _ = e14._replay_rr96(scene, rr_rows[image])
        clean_cache = e14._load_cc_cache(scene, e12_report, clean_records[image])
        right, down = e12.dense_from_graph(clean_cache.cc_candidates, clean_cache.cc_scores)
        cc96_board, _cc96_objective, _ = e12.solve_dense(right, down)
        cc192_board, _cc192_objective = cc192.solve_cc192(right, down)

        canvases = {
            "RR96": e12.fixed_nlm(assemble(scene.tiles_uint8, rr_board)),
            "CC96": e12.fixed_nlm(assemble(scene.tiles_uint8, cc96_board)),
            "CC192": e12.fixed_nlm(assemble(scene.tiles_uint8, cc192_board)),
        }
        rows = {
            "RR96": rr_rows[image],
            "CC96": cc96_rows[image],
            "CC192": cc192_rows[image],
        }
        panels = [_panel(scene.target_uint8, f"target {scene.validation_name}")]
        for arm in ("RR96", "CC96", "CC192"):
            row = rows[arm]
            panels.append(
                _panel(
                    canvases[arm],
                    f"{arm} final={float(row['final_ssim']):.4f} "
                    f"neigh={float(row['neighbour']):.3f} place={float(row['placement']):.3f}",
                )
            )
        sheet = np.hstack(panels)
        output = out_dir / f"image_{image:04d}_target_rr_cc96_cc192.png"
        if not cv2.imwrite(str(output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write {output}")
        manifest.append(
            {
                "image": image,
                "validation_name": scene.validation_name,
                "panel": str(output),
                "RR96_final": float(rr_rows[image]["final_ssim"]),
                "CC96_final": float(cc96_rows[image]["final_ssim"]),
                "CC192_final": float(cc192_rows[image]["final_ssim"]),
            }
        )
        print(json.dumps(manifest[-1], sort_keys=True), flush=True)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "panels": len(manifest)}))


if __name__ == "__main__":
    main()
