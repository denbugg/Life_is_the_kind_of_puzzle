#!/usr/bin/env python3
"""Reuse the audited two-GPU renderer with the promoted luminance builder."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


INPUT = Path("/kaggle/input")


def single(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, got {values}")
    return values[0]


def main() -> None:
    orchestrator_path = single(
        list(INPUT.rglob("harmonized_submission_job/run_harmonized_submission.py")),
        "audited harmonized orchestrator",
    )
    luma_builder = single(
        list(INPUT.rglob("scripts/build_luma_harmonized_submission.py")),
        "promoted luminance builder",
    )
    spec = importlib.util.spec_from_file_location("audited_harmonized_orchestrator", orchestrator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load audited orchestrator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run_builder(
        *,
        bundle_root: Path,
        test_dir: Path,
        gpu: int,
        offset: int,
        limit: int,
        label: str,
    ) -> dict[str, Any]:
        output = module.WORKING / f"{label}.zip"
        report = module.WORKING / f"{label}.json"
        log = module.WORKING / f"{label}.log"
        for path in (output, report, log):
            if path.exists() or path.is_symlink():
                raise RuntimeError(f"builder output must be fresh: {path}")
        command = [
            sys.executable,
            str(luma_builder),
            "--input-dir", str(test_dir),
            "--selected-denoiser", str(bundle_root / "assets/selected_tilenaf_synth_50k.pt"),
            "--seam-denoiser", str(bundle_root / "assets/seam_denoiser_gpu.pt"),
            "--layout-report", str(bundle_root / "layouts/final_qap_shard_000_350.json"),
            "--layout-report", str(bundle_root / "layouts/final_qap_shard_350_700.json"),
            "--output", str(output),
            "--report", str(report),
            "--offset", str(offset),
            "--limit", str(limit),
            "--expected-count", str(module.EXPECTED_TOTAL),
            "--device", "cuda",
            "--batch-size", "512",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONPATH"] = str(bundle_root / "src")
        environment["PYTHONHASHSEED"] = "20260713"
        started = time.perf_counter()
        with log.open("wb") as handle:
            completed = subprocess.run(
                command,
                env=environment,
                cwd=str(luma_builder.parent),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(f"luma builder failed for {label} on GPU {gpu}: rc={completed.returncode}")
        payload = module.load_json(report)
        promotion = payload.get("promotion_evidence", {})
        if (
            payload.get("kind") != "luma_harmonized_frozen_qap_submission_report"
            or payload.get("status") != "test_only_candidate_not_lb_scored"
            or payload.get("offset") != offset
            or payload.get("limit") != limit
            or payload.get("count") != limit
            or payload.get("anti_leakage", {}).get("target_paths_or_pixels_read") is not False
            or payload.get("anti_leakage", {}).get("layout_recomputed") is not False
            or payload.get("archive", {}).get("sha256") != module.sha256(output)
            or promotion.get("calibration_report_sha256") != "9593b8809d2b0e6a0f928e7b0cf41e47f4406e5dd3d8151ea6b75a521c65bbee"
            or promotion.get("confirmation_report_sha256") != "aeac52ebdde35581a974ac863ccb9f7af22f4c4153b667ccb81f68c884b158c1"
            or promotion.get("source_disjoint_confirmation_passed") is not True
        ):
            raise RuntimeError(f"luma builder report contract failed: {label}")
        layout_hashes = {
            record.get("sha256")
            for record in payload.get("assets", {}).get("layout_reports", [])
        }
        if layout_hashes != module.EXPECTED_LAYOUT_REPORT_SHA256:
            raise RuntimeError("builder report layout provenance drift")
        return {
            "label": label,
            "gpu": gpu,
            "offset": offset,
            "limit": limit,
            "output": str(output),
            "output_sha256": module.sha256(output),
            "output_bytes": output.stat().st_size,
            "report": str(report),
            "report_sha256": module.sha256(report),
            "log": str(log),
            "log_sha256": module.sha256(log),
            "method_sha256": payload.get("method_sha256"),
            "source_names_sha256": payload.get("source_names_sha256"),
            "seconds": time.perf_counter() - started,
        }

    module.run_builder = run_builder
    module.main()


if __name__ == "__main__":
    main()
