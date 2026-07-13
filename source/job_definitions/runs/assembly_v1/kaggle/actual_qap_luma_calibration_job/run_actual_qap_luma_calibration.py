#!/usr/bin/env python3
"""Stage actual-QAP bounded luminance-gain calibration on Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WRAPPER = WORKING / "actual_qap_luma_calibration_wrapper.json"
EXPECTED_ASSETS = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "seam_denoiser_gpu.pt": "f973c7e606a112020c527bb72277b82586df915edc829a22305e587b35aec1b9",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exactly_one(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, got {values}")
    return values[0]


def find_asset(name: str) -> Path:
    expected = EXPECTED_ASSETS[name]
    matches = [path.resolve() for path in INPUT_ROOT.rglob(name) if sha256(path) == expected]
    if not matches:
        raise RuntimeError(f"no hash-matching asset found for {name}")
    return sorted(set(matches))[0]


def write_wrapper(payload: dict) -> None:
    WRAPPER.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def hardware_probe() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    devices = []
    for index in range(torch.cuda.device_count()):
        value = torch.randn(128, 128, device=f"cuda:{index}")
        devices.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "tensor_op": float((value @ value).mean().cpu()),
        })
    smi = subprocess.run(["nvidia-smi"], capture_output=True, check=False, text=True)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "nvidia_smi": smi.stdout,
    }


def main() -> None:
    started = time.time()
    wrapper = {
        "schema_version": 1,
        "kind": "actual_qap_luma_calibration_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "started_unix": started,
    }
    write_wrapper(wrapper)
    try:
        evaluator = exactly_one(
            list(INPUT_ROOT.rglob("scripts/evaluate_actual_qap_luma_gain.py")),
            "luma evaluator",
        )
        bundle_root = evaluator.parents[1]
        targets = exactly_one(list(INPUT_ROOT.rglob("train/targets")), "train targets")
        data_root = targets.parent.parent
        selected = find_asset("selected_tilenaf_synth_50k.pt")
        seam = find_asset("seam_denoiser_gpu.pt")
        hbt = find_asset("hbt_d320_denoised_rgb_sobel.pt")
        probe = hardware_probe()
        environment = dict(os.environ)
        environment.update({
            "PYTHONPATH": str(bundle_root / "src"),
            "PYTHONHASHSEED": "20260713",
            "PYTHONUNBUFFERED": "1",
            "OPENBLAS_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
        })
        command = [
            sys.executable,
            str(evaluator),
            "--data-root", str(data_root),
            "--selected-denoiser", str(selected),
            "--seam-denoiser", str(seam),
            "--embedding-checkpoint", str(hbt),
            "--manifest", str(bundle_root / "configs/denoise_splits_seed20260710.json"),
            "--quarantine", str(bundle_root / "configs/denoise_validation_quarantine_v1.json"),
            "--split", "assembly_cal",
            "--source-offset", "32",
            "--sources", "16",
            "--device", "cuda",
            "--batch-size", "512",
            "--denoise-batch-size", "512",
            "--baseline-iterations", "25",
            "--output", str(WORKING / "actual_qap_luma_calibration_report.json"),
            "--overwrite",
        ]
        wrapper.update({
            "status": "running",
            "hardware": probe,
            "bundle_root": str(bundle_root),
            "data_root": str(data_root),
            "assets": {
                "selected": {"path": str(selected), "sha256": sha256(selected)},
                "seam": {"path": str(seam), "sha256": sha256(seam)},
                "hbt": {"path": str(hbt), "sha256": sha256(hbt)},
            },
            "command": command,
        })
        write_wrapper(wrapper)
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode:
            raise RuntimeError(f"evaluator exited with code {completed.returncode}")
        output = WORKING / "actual_qap_luma_calibration_report.json"
        result = json.loads(output.read_text())
        wrapper.update({
            "status": "complete",
            "selected": result.get("selected"),
            "output": {"path": str(output), "sha256": sha256(output)},
            "seconds": time.time() - started,
        })
        write_wrapper(wrapper)
        print(json.dumps(wrapper, sort_keys=True), flush=True)
    except Exception as error:
        wrapper.update({
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "seconds": time.time() - started,
        })
        write_wrapper(wrapper)
        raise


if __name__ == "__main__":
    main()
