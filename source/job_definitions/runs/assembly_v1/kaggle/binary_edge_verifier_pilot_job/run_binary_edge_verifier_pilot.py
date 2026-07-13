#!/usr/bin/env python3
"""Stage and run the binary edge-verifier pilot on Kaggle T4x2."""

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


JOB_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WRAPPER = WORKING / "binary_edge_verifier_pilot_wrapper.json"
EXPECTED_ASSETS = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_wrapper(payload: dict) -> None:
    WRAPPER.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def exactly_one(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, got {values}")
    return values[0]


def find_asset(name: str) -> Path:
    path = exactly_one(list(INPUT_ROOT.rglob(name)), name)
    actual = sha256(path)
    if actual != EXPECTED_ASSETS[name]:
        raise RuntimeError(f"asset hash mismatch for {name}: {actual}")
    return path


def stage_bundle() -> Path:
    trainer = exactly_one(
        list(INPUT_ROOT.rglob("scripts/train_binary_edge_verifier.py")),
        "bundled trainer",
    )
    root = trainer.parents[1]
    if not (root / "src/puzzle_assembly/binary_edge_verifier.py").is_file():
        raise RuntimeError(f"binary verifier module is missing under {root}")
    return root


def hardware_probe() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    devices = []
    for index in range(torch.cuda.device_count()):
        value = torch.randn(128, 128, device=f"cuda:{index}")
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "tensor_op": float((value @ value).mean().cpu()),
            }
        )
    if any(tuple(device["capability"]) < (7, 0) for device in devices):
        raise RuntimeError(f"pilot requires a supported T4-class GPU, got {devices}")
    smi = subprocess.run(
        ["nvidia-smi"], capture_output=True, check=False, text=True
    )
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
        "kind": "binary_edge_verifier_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "started_unix": started,
    }
    write_wrapper(wrapper)
    try:
        bundle_root = stage_bundle()
        required = [
            bundle_root / "scripts/train_binary_edge_verifier.py",
            bundle_root / "src/puzzle_assembly/binary_edge_verifier.py",
            bundle_root / "configs/denoise_splits_seed20260710.json",
            bundle_root / "configs/denoise_validation_quarantine_v1.json",
        ]
        if any(not path.is_file() for path in required):
            raise RuntimeError(f"incomplete job bundle: {required}")
        targets = exactly_one(list(INPUT_ROOT.rglob("train/targets")), "train targets")
        data_root = targets.parent.parent
        denoiser = find_asset("selected_tilenaf_synth_50k.pt")
        hbt = find_asset("hbt_d320_denoised_rgb_sobel.pt")
        probe = hardware_probe()
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(bundle_root / "src")
        environment["PYTHONHASHSEED"] = "20260713"
        test_command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_binary_edge_verifier.py",
        ]
        test = subprocess.run(
            test_command,
            cwd=bundle_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if test.returncode:
            raise RuntimeError(f"bundle tests failed:\n{test.stdout}\n{test.stderr}")
        command = [
            sys.executable,
            str(bundle_root / "scripts/train_binary_edge_verifier.py"),
            "--data-root",
            str(data_root),
            "--denoiser",
            str(denoiser),
            "--embedding-checkpoint",
            str(hbt),
            "--manifest",
            str(bundle_root / "configs/denoise_splits_seed20260710.json"),
            "--quarantine",
            str(bundle_root / "configs/denoise_validation_quarantine_v1.json"),
            "--train-sources",
            "128",
            "--val-sources",
            "8",
            "--epochs",
            "2",
            "--channels",
            "64",
            "--side-band",
            "8",
            "--negative-ratio",
            "4",
            "--batch-size",
            "1024",
            "--score-batch-size",
            "4096",
            "--denoise-batch-size",
            "512",
            "--device",
            "cuda",
            "--data-parallel",
            "--output",
            str(WORKING / "binary_edge_verifier_pilot.pt"),
            "--report",
            str(WORKING / "binary_edge_verifier_pilot.json"),
        ]
        wrapper.update(
            {
                "status": "training",
                "hardware": probe,
                "data_root": str(data_root),
                "assets": {
                    "denoiser": {"path": str(denoiser), "sha256": sha256(denoiser)},
                    "hbt": {"path": str(hbt), "sha256": sha256(hbt)},
                },
                "bundle": {
                    "files": {
                    str(path.relative_to(bundle_root)): sha256(path)
                    for path in required
                    },
                },
                "tests": {"command": test_command, "stdout": test.stdout},
                "command": command,
            }
        )
        write_wrapper(wrapper)
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode:
            raise RuntimeError(f"training exited with code {completed.returncode}")
        report_path = WORKING / "binary_edge_verifier_pilot.json"
        checkpoint_path = WORKING / "binary_edge_verifier_pilot.pt"
        report = json.loads(report_path.read_text())
        wrapper.update(
            {
                "status": "complete",
                "training_status": report["status"],
                "report": {
                    "path": str(report_path),
                    "sha256": sha256(report_path),
                },
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": sha256(checkpoint_path),
                },
                "seconds": time.time() - started,
            }
        )
        write_wrapper(wrapper)
        print(json.dumps(wrapper, sort_keys=True), flush=True)
    except Exception as error:
        wrapper.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "seconds": time.time() - started,
            }
        )
        write_wrapper(wrapper)
        raise


if __name__ == "__main__":
    main()
