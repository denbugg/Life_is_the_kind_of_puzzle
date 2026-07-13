#!/usr/bin/env python3
"""Run the bounded GNC-TLS synchronization calibration on Kaggle."""

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
WRAPPER = WORKING / "gnc_tls_sync_wrapper.json"
EXPECTED_ASSETS = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}
EXPECTED_CODE_TREE_SHA256 = "8712f5820143c85096262dd00a6b87f25587b327a39c1c7049b903e4d77465cf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA unavailable")
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
        "kind": "gnc_tls_sync_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "started_unix": started,
    }
    write(wrapper)
    try:
        evaluator = unique(
            list(INPUT.rglob("scripts/evaluate_gnc_tls_sync.py")),
            "GNC-TLS evaluator",
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
            [sys.executable, "-m", "pytest", "-q", "tests/test_gnc_tls_sync.py"],
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
            raise RuntimeError("GNC-TLS tests failed")
        command = [
            sys.executable,
            str(evaluator),
            "--data-root",
            str(data_root),
            "--production-config",
            str(code_root / "configs/qap_weight_confirmation_v1.json"),
            "--denoiser",
            str(denoiser),
            "--hbt-checkpoint",
            str(hbt),
            "--manifest",
            str(code_root / "configs/denoise_splits_seed20260710.json"),
            "--quarantine",
            str(code_root / "configs/denoise_validation_quarantine_v1.json"),
            "--output-dir",
            str(WORKING / "gnc_tls_sync_gate"),
            "--device",
            "cuda",
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
            raise RuntimeError(f"GNC-TLS evaluator exited {completed.returncode}")
        report = WORKING / "gnc_tls_sync_gate/gnc_tls_sync_report.json"
        raw_report = report.read_bytes()
        envelope = json.loads(raw_report)
        if set(envelope) != {"payload", "payload_sha256"}:
            raise RuntimeError("non-canonical GNC-TLS report envelope keys")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeError("invalid GNC-TLS report payload")
        payload_sha256 = hashlib.sha256(canonical_bytes(payload)).hexdigest()
        if payload_sha256 != envelope.get("payload_sha256"):
            raise RuntimeError("GNC-TLS report payload hash mismatch")
        if raw_report != canonical_bytes(envelope) + b"\n":
            raise RuntimeError("GNC-TLS report envelope is not canonical")
        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != "gnc_tls_sync_calibration_report"
            or payload.get("safe_for_submission") is not False
            or payload.get("submission_ready") is not False
        ):
            raise RuntimeError("invalid GNC-TLS report envelope")
        wrapper.update(
            {
                "status": "complete",
                "scientific_status": payload["status"],
                "selection": payload["selection"],
                "output": {
                    "path": str(report),
                    "report_file_sha256": sha256(report),
                    "payload_sha256": payload_sha256,
                },
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
