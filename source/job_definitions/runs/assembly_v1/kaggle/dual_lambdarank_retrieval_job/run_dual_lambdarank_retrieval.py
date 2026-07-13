#!/usr/bin/env python3
"""Run the bounded dual-sided LambdaRank retrieval gate on Kaggle."""

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
WRAPPER = WORKING / "dual_lambdarank_wrapper.json"
EXPECTED_ASSETS = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
    "full_union_tabular.joblib": "c5929a76c843f7541119f622bf1c5b6774006ad79e3811407e36edfe60bd0f10",
}
EXPECTED_CODE_TREE_SHA256 = "0832c395363b7795913f8af3362f37e84d7bf966d3eff14f81ba537e758d66cd"


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
    paths = list(INPUT.rglob(name))
    if name == "full_union_tabular.joblib":
        paths = [path for path in paths if "dual-lambdarank" in str(path)]
    valid = [path.resolve() for path in paths if sha256(path) == EXPECTED_ASSETS[name]]
    if not valid:
        raise RuntimeError(f"no hash-valid asset found: {name}: {paths}")
    # Kaggle may expose both the standalone file and the copy inside the
    # automatically expanded ZIP dataset.  They are byte-identical; prefer the
    # shallowest canonical mount rather than treating that as ambiguity.
    return sorted(set(valid), key=lambda path: (len(path.parts), str(path)))[0]


def write(payload: dict) -> None:
    temporary = WRAPPER.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(WRAPPER)


def hardware_probe() -> dict:
    import lightgbm
    import torch

    if not torch.cuda.is_available():
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
        "lightgbm": lightgbm.__version__,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "nvidia_smi": smi.stdout,
    }


def main() -> None:
    started = time.time()
    wrapper = {
        "schema_version": 1,
        "kind": "dual_lambdarank_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "started_unix": started,
    }
    write(wrapper)
    try:
        evaluator = unique(
            list(INPUT.rglob("scripts/train_evaluate_dual_lambdarank.py")),
            "LambdaRank evaluator",
        )
        code_root = evaluator.parents[1]
        code_tree_hash = tree_sha256(code_root)
        if code_tree_hash != EXPECTED_CODE_TREE_SHA256:
            raise RuntimeError(f"code tree hash mismatch: {code_tree_hash}")
        data_root = unique(list(INPUT.rglob("train/targets")), "train targets").parent.parent
        denoiser = asset("selected_tilenaf_synth_50k.pt")
        hbt = asset("hbt_d320_denoised_rgb_sobel.pt")
        legacy_hgb = asset("full_union_tabular.joblib")
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
        command = [
            sys.executable,
            str(evaluator),
            "--data-root",
            str(data_root),
            "--denoiser",
            str(denoiser),
            "--embedding-checkpoint",
            str(hbt),
            "--legacy-hgb",
            str(legacy_hgb),
            "--manifest",
            str(code_root / "configs/denoise_splits_seed20260710.json"),
            "--quarantine",
            str(code_root / "configs/denoise_validation_quarantine_v1.json"),
            "--fit-sources",
            "24",
            "--calibration-sources",
            "8",
            "--calibration-offset",
            "368",
            "--device",
            "cuda",
            "--denoise-batch-size",
            "512",
            "--n-estimators",
            "200",
            "--output-root",
            str(WORKING / "dual_lambdarank_retrieval"),
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
                    "legacy_hgb": {"path": str(legacy_hgb), "sha256": sha256(legacy_hgb)},
                },
                "command": command,
            }
        )
        write(wrapper)
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode:
            raise RuntimeError(f"retrieval evaluator exited {completed.returncode}")
        report = WORKING / "dual_lambdarank_retrieval/dual_lambdarank_report.json"
        result = json.loads(report.read_text())
        wrapper.update(
            {
                "status": "complete",
                "scientific_status": result["status"],
                "gate": result["gate"],
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
