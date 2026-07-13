from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

EXPECTED_MANIFEST_SHA256 = "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345"
EXPECTED_VAL_PAIRS_SHA256 = "78795a0a0ed1ee10bddac0c31222f2a9418c41d94249aa35ba183d15508928ed"
EXPECTED_CHECKPOINT_SHA256 = "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"
EXPECTED_LEGACY_SHA256 = "d1df5a4e4852c821d79f72063866cf1fe09fb1beff913a4fb1034466d6ead96e"
EXPECTED_QUARANTINE_SHA256 = "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215"
EXPECTED_SELECTION_SHA256 = "ce244ce8c9759be859262fd16560f8318814022883ec52cdc380ad490a924080"
EXPECTED_CODE_SHA256 = "9025301a3760c2ae5d520c88bd3c92cc8befa5eb687025088b4981ce135dccde"
EXPECTED_OPENCV_VERSION = "4.11.0"
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "pillow": "11.3.0",
    "scipy": "1.16.3",
    "skimage": "0.25.2",
    "torch": "2.6.0+cpu",
    "kornia": "0.8.3",
    "opencv": EXPECTED_OPENCV_VERSION,
    "jpeg_codec": "6.2",
    "libjpeg_turbo": "True",
    "libjpeg_turbo_version": "3.1.1",
}
CODE_FILES = (
    "__init__.py",
    "degradation.py",
    "final_gate_audit.py",
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


def unique_input(pattern: str, label: str) -> Path:
    candidates = sorted(Path("/kaggle/input").rglob(pattern))
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one {label}, found {candidates}")
    return candidates[0]


def hash_code(package_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in CODE_FILES:
        path = package_dir / name
        if not path.is_file():
            raise SystemExit(f"final-gate code file is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


package_file = unique_input("src/puzzle_denoise_v2/final_gate_audit.py", "final-gate package")
package_dir = package_file.parent
package_root = package_dir.parent
manifest_path = unique_input("denoise_splits_seed20260710.json", "split manifest")
val_pairs_path = unique_input("real_gold_val.npz", "validation real-pair artifact")
checkpoint_path = unique_input("tilenaf_synth_50k.pt", "selected checkpoint")
legacy_path = unique_input("tile_restorer_1024_q90.pt", "legacy checkpoint")
quarantine_path = unique_input(
    "denoise_validation_quarantine_v1.json", "validation quarantine"
)
selection_path = unique_input("selected_model.json", "frozen selection manifest")

actual_inputs = {
    "manifest": sha256_file(manifest_path),
    "validation_pairs": sha256_file(val_pairs_path),
    "selected_checkpoint": sha256_file(checkpoint_path),
    "legacy_checkpoint": sha256_file(legacy_path),
    "validation_quarantine": sha256_file(quarantine_path),
    "selection_manifest": sha256_file(selection_path),
    "final_gate_code": hash_code(package_dir),
}
expected_inputs = {
    "manifest": EXPECTED_MANIFEST_SHA256,
    "validation_pairs": EXPECTED_VAL_PAIRS_SHA256,
    "selected_checkpoint": EXPECTED_CHECKPOINT_SHA256,
    "legacy_checkpoint": EXPECTED_LEGACY_SHA256,
    "validation_quarantine": EXPECTED_QUARANTINE_SHA256,
    "selection_manifest": EXPECTED_SELECTION_SHA256,
    "final_gate_code": EXPECTED_CODE_SHA256,
}
mismatches = {
    name: {"actual": actual_inputs[name], "expected": expected}
    for name, expected in expected_inputs.items()
    if actual_inputs[name] != expected
}
if mismatches:
    raise SystemExit(f"pinned final-gate input mismatch: {mismatches}")

started = time.time()
commands = [
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--upgrade",
        "torch==2.6.0",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    ],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "kornia==0.8.3"],
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "opencv-python-headless==4.11.0.86",
    ],
]
for command in commands:
    print(json.dumps({"event": "install", "command": command}), flush=True)
    subprocess.run(command, check=True)
print(json.dumps({"event": "runtime_ready", "seconds": time.time() - started}), flush=True)

sys.path.insert(0, str(package_root))

import cv2
import kornia
import numpy as np
import PIL
from PIL import features as pillow_features
import scipy
import skimage
import torch

from puzzle_denoise_v2.final_gate_audit import (
    FinalGateAuditConfig,
    final_gate_code_fingerprint,
    run_final_gate_audit,
)


actual_runtime = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pillow": PIL.__version__,
    "scipy": scipy.__version__,
    "skimage": skimage.__version__,
    "torch": torch.__version__,
    "kornia": kornia.__version__,
    "opencv": cv2.__version__,
    "jpeg_codec": str(pillow_features.version_codec("jpg")),
    "libjpeg_turbo": str(pillow_features.check_feature("libjpeg_turbo")),
    "libjpeg_turbo_version": str(pillow_features.version_feature("libjpeg_turbo")),
}
runtime_mismatches = {
    name: {"actual": actual_runtime[name], "expected": expected}
    for name, expected in EXPECTED_RUNTIME.items()
    if actual_runtime[name] != expected
}
if runtime_mismatches:
    raise SystemExit(f"pinned final-gate runtime mismatch: {runtime_mismatches}")
if torch.cuda.is_available() or torch.version.cuda is not None:
    raise SystemExit("CPU-only final gate unexpectedly has CUDA-capable torch")
if final_gate_code_fingerprint() != EXPECTED_CODE_SHA256:
    raise SystemExit("final-gate code changed between pre-install hash and import")

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

print(
    json.dumps(
        {
            "event": "selected_final_gate_start",
            "runtime": actual_runtime,
            "inputs": actual_inputs,
            "selection_was_frozen_before_gate": True,
            "training_or_tuning_launched": False,
            "device": "cpu",
        },
        sort_keys=True,
    ),
    flush=True,
)
report = run_final_gate_audit(
    FinalGateAuditConfig(
        data_root=str(data_root),
        manifest=str(manifest_path),
        val_pairs=str(val_pairs_path),
        checkpoint=str(checkpoint_path),
        legacy_checkpoint=str(legacy_path),
        quarantine_artifact=str(quarantine_path),
        selection_manifest=str(selection_path),
        output="/kaggle/working/selected_final_gate_report.json",
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        expected_val_pairs_sha256=EXPECTED_VAL_PAIRS_SHA256,
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        expected_legacy_checkpoint_sha256=EXPECTED_LEGACY_SHA256,
        expected_quarantine_sha256=EXPECTED_QUARANTINE_SHA256,
        expected_selection_manifest_sha256=EXPECTED_SELECTION_SHA256,
        expected_code_sha256=EXPECTED_CODE_SHA256,
        expected_opencv_version=EXPECTED_OPENCV_VERSION,
        batch_size=128,
        bootstrap_resamples=5000,
        torch_threads=4,
    )
)
print(
    json.dumps(
        {
            "event": "selected_final_gate_complete",
            "assessment": report["assessment"],
            "output": "/kaggle/working/selected_final_gate_report.json",
        },
        sort_keys=True,
    ),
    flush=True,
)
