from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


# Defense in depth in addition to kernel-metadata.json enable_gpu=false.  This
# must be set before torch is imported by any project module.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

EXPECTED_MANIFEST_SHA256 = "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345"
EXPECTED_VAL_PAIRS_SHA256 = "78795a0a0ed1ee10bddac0c31222f2a9418c41d94249aa35ba183d15508928ed"
EXPECTED_LEGACY_CHECKPOINT_SHA256 = (
    "d1df5a4e4852c821d79f72063866cf1fe09fb1beff913a4fb1034466d6ead96e"
)
EXPECTED_QUARANTINE_SHA256 = (
    "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215"
)
EXPECTED_VALIDATION_PIXELS_SHA256 = (
    "fe573ed28b74b45e8b0302ad51c53ff0f7ad5ad907aa3d4d9332c87010e42bd5"
)
EXPECTED_BENCHMARK_CODE_SHA256 = (
    "2b43b9774f2f505e09c146b4a0934f40adf9c36ca3ff2c8c693b72a459de52e5"
)
EXPECTED_OPENCV_VERSION = "4.11.0"
OPENCV_PACKAGE = "opencv-python-headless==4.11.0.86"
KORNIA_PACKAGE = "kornia==0.8.3"
TORCH_PACKAGE = "torch==2.6.0"
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "pillow": "11.3.0",
    "scipy": "1.16.3",
    "skimage": "0.25.2",
    "torch": "2.6.0+cpu",
    "kornia": "0.8.3",
    "opencv": "4.11.0",
}

# Replace only after the completed synthetic-50k output has been downloaded
# and hashed.  The sentinel fails before dependency installation or inference.
EXPECTED_INIT_CHECKPOINT_SHA256 = "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"

BENCHMARK_CODE_FILES = (
    "__init__.py",
    "degradation.py",
    "inference.py",
    "legacy_baseline.py",
    "losses.py",
    "metrics.py",
    "model.py",
    "prefinetune_benchmark.py",
    "real_pairs.py",
    "real_training.py",
    "real_validation.py",
    "tiles.py",
    "training.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_package(package_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in BENCHMARK_CODE_FILES:
        path = package_dir / name
        if not path.is_file():
            raise SystemExit(f"benchmark package file is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def unique_input(pattern: str, label: str) -> Path:
    candidates = sorted(Path("/kaggle/input").rglob(pattern))
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one {label}, found {candidates}")
    return candidates[0]


if re.fullmatch(r"[0-9a-f]{64}", EXPECTED_INIT_CHECKPOINT_SHA256) is None:
    raise SystemExit(
        "EXPECTED_INIT_CHECKPOINT_SHA256 must be replaced with the completed "
        "synthetic-50k EMA checkpoint SHA256"
    )

package_file = unique_input(
    "src/puzzle_denoise_v2/prefinetune_benchmark.py", "benchmark package"
)
package_dir = package_file.parent
package_root = package_dir.parent
manifest_path = unique_input("denoise_splits_seed20260710.json", "split manifest")
quarantine_path = unique_input(
    "denoise_validation_quarantine_v1.json", "validation quarantine artifact"
)
val_pairs_path = unique_input("real_gold_val.npz", "validation real-pair artifact")
checkpoint_path = unique_input("tilenaf_synth_50k.pt", "synthetic-50k checkpoint")
legacy_path = unique_input("tile_restorer_1024_q90.pt", "legacy q90 checkpoint")

actual_inputs = {
    "manifest": sha256_file(manifest_path),
    "validation_quarantine": sha256_file(quarantine_path),
    "val_pairs": sha256_file(val_pairs_path),
    "init_checkpoint": sha256_file(checkpoint_path),
    "legacy_checkpoint": sha256_file(legacy_path),
    "benchmark_code": hash_package(package_dir),
}
expected_inputs = {
    "manifest": EXPECTED_MANIFEST_SHA256,
    "validation_quarantine": EXPECTED_QUARANTINE_SHA256,
    "val_pairs": EXPECTED_VAL_PAIRS_SHA256,
    "init_checkpoint": EXPECTED_INIT_CHECKPOINT_SHA256,
    "legacy_checkpoint": EXPECTED_LEGACY_CHECKPOINT_SHA256,
    "benchmark_code": EXPECTED_BENCHMARK_CODE_SHA256,
}
mismatches = {
    name: {"actual": actual_inputs[name], "expected": expected}
    for name, expected in expected_inputs.items()
    if actual_inputs[name] != expected
}
if mismatches:
    raise SystemExit(f"pinned benchmark input mismatch: {mismatches}")

started = time.time()
commands = [
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--upgrade",
        TORCH_PACKAGE,
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    ],
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        KORNIA_PACKAGE,
    ],
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        OPENCV_PACKAGE,
    ],
]
for command in commands:
    print(json.dumps({"event": "install", "command": command}), flush=True)
    subprocess.run(command, check=True)
print(
    json.dumps(
        {"event": "runtime_dependencies_ready", "seconds": time.time() - started},
        sort_keys=True,
    ),
    flush=True,
)

sys.path.insert(0, str(package_root))

import platform

import cv2
import kornia
import numpy as np
import PIL
import scipy
import skimage
import torch

from puzzle_denoise_v2.prefinetune_benchmark import (
    PreFineTuneBenchmarkConfig,
    prefinetune_benchmark_code_fingerprint,
    run_prefinetune_benchmark,
)


actual_runtime_versions = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pillow": PIL.__version__,
    "scipy": scipy.__version__,
    "skimage": skimage.__version__,
    "torch": torch.__version__,
    "kornia": kornia.__version__,
    "opencv": cv2.__version__,
}
runtime_mismatches = {
    name: {"actual": actual_runtime_versions[name], "expected": expected}
    for name, expected in EXPECTED_RUNTIME_VERSIONS.items()
    if actual_runtime_versions[name] != expected
}
if runtime_mismatches:
    raise SystemExit(f"pinned CPU runtime mismatch: {runtime_mismatches}")
if prefinetune_benchmark_code_fingerprint() != EXPECTED_BENCHMARK_CODE_SHA256:
    raise SystemExit("benchmark code changed between pre-import hashing and import")
if torch.cuda.is_available() or torch.version.cuda is not None:
    raise SystemExit("CPU-only benchmark unexpectedly has a CUDA-capable torch runtime")

data_root = Path("/kaggle/input/datasets/pasha883/vsos-ai-initiative-pazzle")
if not (data_root / "train" / "targets").is_dir():
    roots = sorted(
        path.parent.parent
        for path in Path("/kaggle/input").rglob("train/targets")
        if path.is_dir()
    )
    if len(roots) != 1:
        raise SystemExit(f"expected exactly one puzzle data root, found {roots}")
    data_root = roots[0]

run_probe = {
    "event": "prefinetune_calibration_start",
    "device": "cpu",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_cuda_version": torch.version.cuda,
    "runtime_versions": actual_runtime_versions,
    "inputs": actual_inputs,
    "paths": {
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "validation_quarantine": str(quarantine_path),
        "val_pairs": str(val_pairs_path),
        "checkpoint": str(checkpoint_path),
        "legacy": str(legacy_path),
    },
    "expected_validation_pixels_sha256": EXPECTED_VALIDATION_PIXELS_SHA256,
    "scope": (
        "93 quarantined sources excluded; 257 clean calibration sources evaluated; "
        "350-source sealed gate integrity-hashed only with no gate tiles passed to model/metrics"
    ),
}
print(json.dumps(run_probe, sort_keys=True), flush=True)

report = run_prefinetune_benchmark(
    PreFineTuneBenchmarkConfig(
        data_root=str(data_root),
        manifest=str(manifest_path),
        val_pairs=str(val_pairs_path),
        init_checkpoint=str(checkpoint_path),
        legacy_checkpoint=str(legacy_path),
        quarantine_artifact=str(quarantine_path),
        output="/kaggle/working/prefinetune_calibration_report.json",
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        expected_val_pairs_sha256=EXPECTED_VAL_PAIRS_SHA256,
        expected_init_checkpoint_sha256=EXPECTED_INIT_CHECKPOINT_SHA256,
        expected_legacy_checkpoint_sha256=EXPECTED_LEGACY_CHECKPOINT_SHA256,
        expected_quarantine_sha256=EXPECTED_QUARANTINE_SHA256,
        expected_validation_pixels_sha256=EXPECTED_VALIDATION_PIXELS_SHA256,
        expected_code_sha256=EXPECTED_BENCHMARK_CODE_SHA256,
        expected_opencv_version=EXPECTED_OPENCV_VERSION,
        batch_size=128,
        bootstrap_resamples=5000,
        torch_threads=4,
        max_legacy_ssim_deficit=0.01,
        gate_source_count=350,
    )
)
print(
    json.dumps(
        {
            "event": "prefinetune_calibration_complete",
            "output": "/kaggle/working/prefinetune_calibration_report.json",
            "diagnostic": report["diagnostic"],
        },
        sort_keys=True,
    ),
    flush=True,
)
