#!/usr/bin/env python3
"""Run the bounded ContinuationNet-0 training and two-stage gate on Kaggle."""

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
WRAPPER = WORKING / "continuation_net0_wrapper.json"
EXPECTED_ASSETS = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}
EXPECTED_CODE_TREE_SHA256 = "37fa325f02f4c9fc006a9cadf6a9044ef9ab6720e52e1d3f56edce7076f36f9a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    entries = [
        [path.relative_to(root).as_posix(), sha256(path)]
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    ]
    payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def unique(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected one {label}, got {values}")
    return values[0]


def asset(name: str) -> Path:
    valid = [
        path.resolve()
        for path in INPUT.rglob(name)
        if sha256(path) == EXPECTED_ASSETS[name]
    ]
    if not valid:
        raise RuntimeError(f"no hash-valid asset found: {name}")
    return sorted(set(valid), key=lambda path: (len(path.parts), str(path)))[0]


def write(payload: dict) -> None:
    temporary = WRAPPER.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(WRAPPER)


def hardware_probe() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly two T4 devices, got {torch.cuda.device_count()}")
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
        "kind": "continuation_net0_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "started_unix": started,
    }
    write(wrapper)
    try:
        evaluator = unique(
            list(INPUT.rglob("scripts/train_evaluate_continuation_net0.py")),
            "ContinuationNet evaluator",
        )
        code_root = evaluator.parents[1]
        code_tree_hash = tree_sha256(code_root)
        if code_tree_hash != EXPECTED_CODE_TREE_SHA256:
            raise RuntimeError(f"code tree hash mismatch: {code_tree_hash}")
        data_root = unique(list(INPUT.rglob("train/targets")), "train targets").parent.parent
        denoiser = asset("selected_tilenaf_synth_50k.pt")
        hbt = asset("hbt_d320_denoised_rgb_sobel.pt")
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONPATH": str(code_root / "src"),
                "PYTHONHASHSEED": "20260713",
                "PYTHONUNBUFFERED": "1",
                "OPENBLAS_NUM_THREADS": "4",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
            }
        )
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_continuation_net.py"],
            cwd=code_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        wrapper["tests"] = {
            "returncode": tests.returncode,
            "stdout": tests.stdout,
            "stderr": tests.stderr,
        }
        if tests.returncode:
            raise RuntimeError("ContinuationNet tests failed")
        command = [
            sys.executable,
            str(evaluator),
            "--data-root",
            str(data_root),
            "--denoiser",
            str(denoiser),
            "--embedding-checkpoint",
            str(hbt),
            "--manifest",
            str(code_root / "configs/denoise_splits_seed20260710.json"),
            "--quarantine",
            str(code_root / "configs/denoise_validation_quarantine_v1.json"),
            "--output-root",
            str(WORKING / "continuation_net0_gate"),
            "--device",
            "cuda",
            "--data-parallel",
            "--overwrite",
        ]
        wrapper.update(
            {
                "status": "running",
                "hardware": hardware_probe(),
                "code_root": str(code_root),
                "code_tree_sha256": code_tree_hash,
                "assets": {
                    "denoiser": {"path": str(denoiser), "sha256": sha256(denoiser)},
                    "hbt": {"path": str(hbt), "sha256": sha256(hbt)},
                },
                "command": command,
            }
        )
        write(wrapper)
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode:
            raise RuntimeError(f"ContinuationNet evaluator exited {completed.returncode}")
        report = WORKING / "continuation_net0_gate/continuation_net0_report.json"
        result = json.loads(report.read_text())
        wrapper.update(
            {
                "status": "complete",
                "scientific_status": result["status"],
                "output": {"path": str(report), "sha256": sha256(report)},
                "seconds": time.time() - started,
            }
        )
        write(wrapper)
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
        write(wrapper)
        raise


if __name__ == "__main__":
    main()
