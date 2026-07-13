#!/usr/bin/env python3
"""Stage, preflight, and run the bounded ViT-Sinkhorn pilot on 2xT4."""

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
            if (path.parent.parent.parent / "configs" / "denoise_splits_seed20260710.json").is_file()
            and (path.parent.parent.parent / "src" / "puzzle_denoise_v2" / "inference.py").is_file()
        ],
        "base code root",
    )


def find_overlay_root() -> Path:
    return one(
        [
            path.parent.parent
            for path in INPUT.glob("**/scripts/train_evaluate_vit_sinkhorn.py")
            if (path.parent.parent / "src" / "puzzle_assembly" / "vit_sinkhorn.py").is_file()
            and (path.parent.parent / "tests" / "test_vit_sinkhorn.py").is_file()
        ],
        "ViT overlay root",
    )


def find_pseudo_gold() -> Path:
    return one(list(INPUT.glob("**/real_gold_train_512.npz")), "real pseudo-gold archive")


def hardware_probe() -> dict[str, object]:
    subprocess.run(["nvidia-smi"], check=False)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = torch.cuda.device_count()
    if count != 2:
        raise RuntimeError(f"this pilot requires exactly two visible GPUs, found {count}")
    means: list[float] = []
    for index in range(count):
        device = torch.device(f"cuda:{index}")
        left = torch.randn(256, 256, device=device)
        right = torch.randn(256, 256, device=device)
        means.append(float((left @ right).mean().item()))
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": count,
        "devices": [torch.cuda.get_device_name(index) for index in range(count)],
        "capabilities": [list(torch.cuda.get_device_capability(index)) for index in range(count)],
        "arch_list": torch.cuda.get_arch_list(),
        "matmul_means": means,
    }


def copy_overlay(base_root: Path, overlay_root: Path, code_root: Path) -> None:
    if code_root.exists():
        shutil.rmtree(code_root)
    code_root.mkdir(parents=True)
    shutil.copytree(base_root / "src", code_root / "src")
    shutil.copytree(base_root / "configs", code_root / "configs")
    (code_root / "scripts").mkdir()
    (code_root / "tests").mkdir()
    shutil.copy2(
        overlay_root / "src" / "puzzle_assembly" / "vit_sinkhorn.py",
        code_root / "src" / "puzzle_assembly" / "vit_sinkhorn.py",
    )
    shutil.copy2(
        overlay_root / "scripts" / "train_evaluate_vit_sinkhorn.py",
        code_root / "scripts" / "train_evaluate_vit_sinkhorn.py",
    )
    shutil.copy2(
        overlay_root / "tests" / "test_vit_sinkhorn.py",
        code_root / "tests" / "test_vit_sinkhorn.py",
    )


def main() -> None:
    started = time.perf_counter()
    data_root = find_data_root()
    runtime_root = find_runtime_root()
    base_root = find_base_root()
    overlay_root = find_overlay_root()
    pseudo_gold = find_pseudo_gold()
    code_root = WORKING / "vit_sinkhorn_code"
    copy_overlay(base_root, overlay_root, code_root)
    # Two integration tests intentionally exercise the real manifest/source
    # paths through their project-default ``puzzle`` location.  Kaggle mounts
    # the dataset read-only elsewhere, so expose that same data through a
    # read-only symlink instead of skipping the tests.
    (code_root / "puzzle").symlink_to(data_root, target_is_directory=True)
    hardware = hardware_probe()
    print(json.dumps({"event": "hardware", **hardware}, sort_keys=True), flush=True)

    script = code_root / "scripts" / "train_evaluate_vit_sinkhorn.py"
    model_source = code_root / "src" / "puzzle_assembly" / "vit_sinkhorn.py"
    test_source = code_root / "tests" / "test_vit_sinkhorn.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(code_root / "src")
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(script), str(model_source), str(test_source)],
        check=True,
        cwd=code_root,
        env=environment,
    )
    tests_started = time.perf_counter()
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_source)],
        check=False,
        cwd=code_root,
        env=environment,
    )
    tests_seconds = time.perf_counter() - tests_started
    if tests.returncode != 0:
        raise RuntimeError(f"ViT unit tests failed with code {tests.returncode}")

    smoke_dir = WORKING / "vit_sinkhorn_smoke"
    smoke_command = [
        sys.executable,
        str(script),
        "--synthetic-smoke",
        "--device", "cuda:0",
        "--smoke-steps", "2",
        "--output-dir", str(smoke_dir),
        "--overwrite",
    ]
    smoke_started = time.perf_counter()
    smoke = subprocess.run(smoke_command, check=False, cwd=code_root, env=environment)
    smoke_seconds = time.perf_counter() - smoke_started
    if smoke.returncode != 0:
        raise RuntimeError(f"ViT GPU smoke failed with code {smoke.returncode}")

    output_dir = WORKING / "vit_sinkhorn_pilot"
    pilot_command = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(script),
        "--data-root", str(data_root),
        "--manifest", str(code_root / "configs" / "denoise_splits_seed20260710.json"),
        "--quarantine", str(code_root / "configs" / "denoise_validation_quarantine_v1.json"),
        "--denoiser", str(runtime_root / "selected_tilenaf_synth_50k.pt"),
        "--real-gold", str(pseudo_gold),
        "--real-gold-source-limit", "64",
        "--real-gold-probability", "0.25",
        "--qap-prior-probability", "0",
        "--train-sources", "256",
        "--dev-sources", "8",
        "--holdout-sources", "8",
        "--epochs", "3",
        "--amp", "fp16",
        "--amp-init-scale", "1024",
        "--max-consecutive-amp-skips", "8",
        "--output-dir", str(output_dir),
        "--overwrite",
    ]
    pilot_started = time.perf_counter()
    pilot = subprocess.run(pilot_command, check=False, cwd=code_root, env=environment)
    pilot_seconds = time.perf_counter() - pilot_started

    wrapper: dict[str, object] = {
        "schema_version": 1,
        "kind": "vit_sinkhorn_t4x2_pilot_wrapper",
        "status": "complete" if pilot.returncode == 0 else "error",
        "safe_for_submission": False,
        "hardware": hardware,
        "tests": {"returncode": tests.returncode, "seconds": tests_seconds},
        "smoke": {"returncode": smoke.returncode, "seconds": smoke_seconds, "command": smoke_command},
        "pilot": {"returncode": pilot.returncode, "seconds": pilot_seconds, "command": pilot_command},
        "inputs": {
            "script_sha256": sha256(script),
            "model_sha256": sha256(model_source),
            "test_sha256": sha256(test_source),
            "pseudo_gold_sha256": sha256(pseudo_gold),
            "denoiser_sha256": sha256(runtime_root / "selected_tilenaf_synth_50k.pt"),
            "manifest_sha256": sha256(code_root / "configs" / "denoise_splits_seed20260710.json"),
            "quarantine_sha256": sha256(code_root / "configs" / "denoise_validation_quarantine_v1.json"),
        },
        "seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "vit_sinkhorn_report.json"
    checkpoint_path = output_dir / "vit_sinkhorn_checkpoint.pt"
    hashes_path = output_dir / "SHA256SUMS.txt"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        wrapper["report"] = str(report_path)
        wrapper["report_sha256"] = sha256(report_path)
        wrapper["pilot_status"] = report.get("status")
        wrapper["selection_gate_passed"] = report.get("selection_gate_passed")
        wrapper["holdout_gate_passed"] = report.get("holdout_gate_passed")
        wrapper["selected_epoch"] = report.get("selected_epoch")
    for label, path in (("checkpoint", checkpoint_path), ("hashes", hashes_path)):
        if path.is_file():
            wrapper[label] = str(path)
            wrapper[f"{label}_sha256"] = sha256(path)
    smoke_report = smoke_dir / "vit_sinkhorn_report.json"
    if smoke_report.is_file():
        wrapper["smoke_report"] = str(smoke_report)
        wrapper["smoke_report_sha256"] = sha256(smoke_report)

    wrapper_path = WORKING / "vit_sinkhorn_pilot_wrapper.json"
    wrapper_path.write_text(json.dumps(wrapper, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "wrapper_complete", **wrapper}, default=str), flush=True)
    if pilot.returncode != 0:
        raise SystemExit(pilot.returncode)


if __name__ == "__main__":
    main()
