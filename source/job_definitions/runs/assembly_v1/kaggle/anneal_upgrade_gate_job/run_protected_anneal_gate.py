#!/usr/bin/env python3
"""Stage and execute the exact-only protected annealing gate on Kaggle."""

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


def find_base_root() -> Path:
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
            for path in INPUT.glob("**/scripts/evaluate_anneal_gate.py")
            if (path.parent.parent / "src" / "puzzle_assembly" / "anneal_refine.py").is_file()
        ],
        "anneal overlay root",
    )


def hardware_probe() -> dict[str, object]:
    subprocess.run(["nvidia-smi"], check=False)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = torch.cuda.device_count()
    means = []
    for index in range(count):
        device = torch.device(f"cuda:{index}")
        means.append(float((torch.randn(128, 128, device=device) @ torch.randn(128, 128, device=device)).mean().item()))
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "device_count": count,
        "devices": [torch.cuda.get_device_name(index) for index in range(count)],
        "capabilities": [list(torch.cuda.get_device_capability(index)) for index in range(count)],
        "arch_list": torch.cuda.get_arch_list(),
        "matmul_means": means,
    }


def main() -> None:
    started = time.perf_counter()
    data_root = find_data_root()
    runtime_root = find_runtime_root()
    base_root = find_base_root()
    overlay_root = find_overlay_root()
    code_root = WORKING / "protected_anneal_code"
    if code_root.exists():
        shutil.rmtree(code_root)
    shutil.copytree(base_root, code_root)
    overlay_script = overlay_root / "scripts" / "evaluate_anneal_gate.py"
    overlay_module = overlay_root / "src" / "puzzle_assembly" / "anneal_refine.py"
    shutil.copy2(overlay_script, code_root / "scripts" / overlay_script.name)
    shutil.copy2(overlay_module, code_root / "src" / "puzzle_assembly" / overlay_module.name)

    hardware = hardware_probe()
    print(json.dumps({"event": "hardware", **hardware}, sort_keys=True), flush=True)
    script = code_root / "scripts" / "evaluate_anneal_gate.py"
    module = code_root / "src" / "puzzle_assembly" / "anneal_refine.py"
    subprocess.run([sys.executable, "-m", "py_compile", str(script), str(module)], check=True, cwd=code_root)

    output = WORKING / "protected_anneal_exact_gate.json"
    command = [
        sys.executable,
        str(script),
        "--data-root", str(data_root),
        "--denoiser", str(runtime_root / "selected_tilenaf_synth_50k.pt"),
        "--embedding-checkpoint", str(runtime_root / "hbt_d320_denoised_rgb_sobel.pt"),
        "--manifest", str(code_root / "configs" / "denoise_splits_seed20260710.json"),
        "--quarantine", str(code_root / "configs" / "denoise_validation_quarantine_v1.json"),
        "--exact-sources", "4",
        "--evaluations", "6000",
        "--restarts", "1",
        "--protection-strengths", "0,0.10,0.25",
        "--device", "cuda:0",
        "--output", str(output),
        "--overwrite",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(code_root / "src")
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    run_started = time.perf_counter()
    completed = subprocess.run(command, cwd=code_root, env=environment, check=False)
    wrapper: dict[str, object] = {
        "schema_version": 1,
        "kind": "protected_anneal_exact_gate_wrapper",
        "status": "complete" if completed.returncode == 0 else "error",
        "hardware": hardware,
        "returncode": completed.returncode,
        "command": command,
        "run_seconds": time.perf_counter() - run_started,
        "seconds": time.perf_counter() - started,
        "inputs": {
            "base_qap_sha256": sha256(base_root / "src" / "puzzle_assembly" / "qap.py"),
            "overlay_script_sha256": sha256(overlay_script),
            "overlay_module_sha256": sha256(overlay_module),
            "denoiser_sha256": sha256(runtime_root / "selected_tilenaf_synth_50k.pt"),
            "embedding_sha256": sha256(runtime_root / "hbt_d320_denoised_rgb_sobel.pt"),
        },
    }
    if output.is_file():
        report = json.loads(output.read_text(encoding="utf-8"))
        wrapper["output"] = str(output)
        wrapper["output_sha256"] = sha256(output)
        wrapper["gate_status"] = report.get("status")
        wrapper["decision"] = report.get("decision")
        wrapper["best_gate"] = (report.get("anneal_gates") or [None])[0]
    wrapper_path = WORKING / "protected_anneal_gate_wrapper.json"
    wrapper_path.write_text(json.dumps(wrapper, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "wrapper_complete", **wrapper}, default=str), flush=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
