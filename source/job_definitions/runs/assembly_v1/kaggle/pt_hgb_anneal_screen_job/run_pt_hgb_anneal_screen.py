#!/usr/bin/env python3
"""Stage the small parallel-tempered HGB annealing screen on Kaggle."""

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


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WRAPPER = WORKING / "pt_hgb_anneal_screen_wrapper.json"
EXPECTED = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
    "full_union_tabular.joblib": "c5929a76c843f7541119f622bf1c5b6774006ad79e3811407e36edfe60bd0f10",
    "report.json": "d8a896dd3f22f138ed9ffac5a7d84bb04ee2af8ea653ce55750e02eacdd0ced5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected one {label}, got {values}")
    return values[0]


def asset(name: str) -> Path:
    candidates = list(INPUT.rglob(name))
    if name == "report.json":
        candidates = [path for path in candidates if path.parent.name == "v1" and "full_union_tabular" in path.parts]
    path = unique(candidates, name)
    if sha256(path) != EXPECTED[name]:
        raise RuntimeError(f"asset hash mismatch: {name}")
    return path


def write(payload: dict) -> None:
    WRAPPER.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def probe() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    devices = []
    for index in range(torch.cuda.device_count()):
        value = torch.randn(128, 128, device=f"cuda:{index}")
        devices.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "tensor_op": float((value @ value).mean().cpu()),
        })
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
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
        "kind": "pt_hgb_anneal_screen_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "started_unix": started,
    }
    write(wrapper)
    try:
        evaluator = unique(list(INPUT.rglob("scripts/evaluate_pt_hgb_anneal.py")), "PT evaluator")
        bundle_root = evaluator.parents[1]
        data_root = unique(list(INPUT.rglob("train/targets")), "train targets").parent.parent
        denoiser = asset("selected_tilenaf_synth_50k.pt")
        hbt = asset("hbt_d320_denoised_rgb_sobel.pt")
        model = asset("full_union_tabular.joblib")
        report = asset("report.json")
        environment = dict(os.environ)
        environment.update({
            "PYTHONPATH": str(bundle_root / "src"),
            "PYTHONHASHSEED": "20260713",
            "PYTHONUNBUFFERED": "1",
            "OPENBLAS_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
        })
        command = [
            sys.executable, str(evaluator),
            "--model", str(model),
            "--tabular-report", str(report),
            "--data-root", str(data_root),
            "--denoiser", str(denoiser),
            "--embedding-checkpoint", str(hbt),
            "--manifest", str(bundle_root / "configs/denoise_splits_seed20260710.json"),
            "--quarantine", str(bundle_root / "configs/denoise_validation_quarantine_v1.json"),
            "--split", "assembly_cal",
            "--source-offset", "56",
            "--sources", "4",
            "--device", "cuda",
            "--denoise-batch-size", "512",
            "--baseline-iterations", "25",
            "--replicas", "8",
            "--steps", "20000",
            "--exchange-interval", "128",
            "--output", str(WORKING / "pt_hgb_anneal_screen_report.json"),
            "--overwrite",
        ]
        wrapper.update({
            "status": "running",
            "hardware": probe(),
            "bundle_root": str(bundle_root),
            "assets": {
                "denoiser": {"path": str(denoiser), "sha256": sha256(denoiser)},
                "hbt": {"path": str(hbt), "sha256": sha256(hbt)},
                "model": {"path": str(model), "sha256": sha256(model)},
                "report": {"path": str(report), "sha256": sha256(report)},
            },
            "command": command,
        })
        write(wrapper)
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode:
            raise RuntimeError(f"PT evaluator exited {completed.returncode}")
        output = WORKING / "pt_hgb_anneal_screen_report.json"
        result = json.loads(output.read_text())
        wrapper.update({
            "status": "complete",
            "selected": result.get("selected"),
            "output": {"path": str(output), "sha256": sha256(output)},
            "seconds": time.time() - started,
        })
        write(wrapper)
        print(json.dumps(wrapper, sort_keys=True), flush=True)
    except Exception as error:
        wrapper.update({
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "seconds": time.time() - started,
        })
        write(wrapper)
        raise


if __name__ == "__main__":
    main()
