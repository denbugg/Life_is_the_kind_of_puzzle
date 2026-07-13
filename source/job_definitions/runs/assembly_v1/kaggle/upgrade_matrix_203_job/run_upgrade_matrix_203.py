#!/usr/bin/env python3
"""Run the leakage-safe 0.203 QAP/axis upgrade matrix on Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")


def one(paths: list[Path], label: str) -> Path:
    values = sorted(set(paths))
    if len(values) != 1:
        raise RuntimeError(f"expected one {label}, found {values}")
    return values[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_data_root() -> Path:
    return one(
        [
            path.parent.parent
            for path in INPUT.glob("**/train/inputs")
            if path.is_dir() and (path.parent / "targets").is_dir()
        ],
        "puzzle data root",
    )


def find_runtime_root() -> Path:
    return one(
        [
            path.parent
            for path in INPUT.glob("**/selected_tilenaf_synth_50k.pt")
            if (path.parent / "hbt_d320_denoised_rgb_sobel.pt").is_file()
        ],
        "runtime asset root",
    )


def find_base_code_root() -> Path:
    return one(
        [
            path.parent.parent.parent
            for path in INPUT.glob("**/src/puzzle_assembly/qap.py")
            if (path.parent.parent.parent / "scripts" / "evaluate_real_assembly.py").is_file()
            and (path.parent.parent.parent / "configs" / "denoise_splits_seed20260710.json").is_file()
        ],
        "base code root",
    )


def find_overlay_root() -> Path:
    return one(
        [
            path.parent.parent
            for path in INPUT.glob("**/scripts/evaluate_upgrade_matrix.py")
            if (path.parent.parent / "src" / "puzzle_assembly" / "axis_refine.py").is_file()
        ],
        "upgrade overlay root",
    )


def hardware_probe() -> dict[str, object]:
    subprocess.run(["nvidia-smi"], check=False)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = torch.cuda.device_count()
    probe: dict[str, object] = {
        "python": sys.version,
        "torch": torch.__version__,
        "device_count": count,
        "devices": [torch.cuda.get_device_name(index) for index in range(count)],
        "capabilities": [
            list(torch.cuda.get_device_capability(index)) for index in range(count)
        ],
        "arch_list": torch.cuda.get_arch_list(),
        "matmul_means": [],
    }
    means: list[float] = []
    for index in range(count):
        device = torch.device(f"cuda:{index}")
        left = torch.randn(128, 128, device=device)
        right = torch.randn(128, 128, device=device)
        means.append(float((left @ right).mean().item()))
    probe["matmul_means"] = means
    return probe


def main() -> None:
    started = time.perf_counter()
    data_root = find_data_root()
    runtime_root = find_runtime_root()
    base_root = find_base_code_root()
    overlay_root = find_overlay_root()
    code_root = WORKING / "upgrade_203_code"
    if code_root.exists():
        shutil.rmtree(code_root)
    shutil.copytree(base_root, code_root)
    overlay_script = overlay_root / "scripts" / "evaluate_upgrade_matrix.py"
    overlay_axis = overlay_root / "src" / "puzzle_assembly" / "axis_refine.py"
    shutil.copy2(overlay_script, code_root / "scripts" / overlay_script.name)
    shutil.copy2(overlay_axis, code_root / "src" / "puzzle_assembly" / overlay_axis.name)

    hardware = hardware_probe()
    print(json.dumps({"event": "hardware", **hardware}, sort_keys=True), flush=True)
    script = code_root / "scripts" / "evaluate_upgrade_matrix.py"
    axis_module = code_root / "src" / "puzzle_assembly" / "axis_refine.py"
    for path in (script, axis_module):
        if not path.is_file():
            raise RuntimeError(f"overlay file missing after extraction: {path}")

    subprocess.run(
        [sys.executable, "-m", "py_compile", str(script), str(axis_module)],
        check=True,
        cwd=code_root,
    )
    output = WORKING / "upgrade_matrix_203.json"
    frozen = WORKING / "upgrade_matrix_203_frozen_layouts.npz"
    command = [
        sys.executable,
        str(script),
        "--data-root",
        str(data_root),
        "--denoiser",
        str(runtime_root / "selected_tilenaf_synth_50k.pt"),
        "--embedding-checkpoint",
        str(runtime_root / "hbt_d320_denoised_rgb_sobel.pt"),
        "--manifest",
        str(code_root / "configs" / "denoise_splits_seed20260710.json"),
        "--quarantine",
        str(code_root / "configs" / "denoise_validation_quarantine_v1.json"),
        "--exact-sources",
        "8",
        "--real-sources",
        "16",
        "--exact-select-count",
        "4",
        "--device",
        "cuda:0",
        "--batch-size",
        "512",
        "--chunk-size",
        "64",
        "--output",
        str(output),
        "--frozen-layouts",
        str(frozen),
        "--overwrite",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(code_root / "src")
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    run_started = time.perf_counter()
    completed = subprocess.run(command, cwd=code_root, env=environment, check=False)
    run_seconds = time.perf_counter() - run_started

    wrapper: dict[str, object] = {
        "schema_version": 1,
        "kind": "assembly_upgrade_203_matrix_wrapper",
        "status": "complete" if completed.returncode == 0 else "error",
        "hardware": hardware,
        "inputs": {
            "data_root": str(data_root),
            "runtime_root": str(runtime_root),
            "base_code_root": str(base_root),
            "base_qap_sha256": sha256(base_root / "src" / "puzzle_assembly" / "qap.py"),
            "overlay_root": str(overlay_root),
            "overlay_script_sha256": sha256(overlay_script),
            "overlay_axis_sha256": sha256(overlay_axis),
            "script_sha256": sha256(script),
            "axis_module_sha256": sha256(axis_module),
        },
        "command": command,
        "returncode": completed.returncode,
        "run_seconds": run_seconds,
        "seconds": time.perf_counter() - started,
    }
    if output.is_file():
        report = json.loads(output.read_text(encoding="utf-8"))
        wrapper["output"] = str(output)
        wrapper["output_sha256"] = sha256(output)
        wrapper["promotion_status"] = report.get("status")
        wrapper["promoted_label"] = report.get("real16", {}).get("promoted_label")
        wrapper["promoted_result"] = report.get("real16", {}).get("promoted_result")
    if frozen.is_file():
        wrapper["frozen_layouts"] = str(frozen)
        wrapper["frozen_layouts_sha256"] = sha256(frozen)
    wrapper_path = WORKING / "upgrade_matrix_203_wrapper.json"
    wrapper_path.write_text(
        json.dumps(wrapper, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "wrapper_complete", **wrapper}, default=str), flush=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
