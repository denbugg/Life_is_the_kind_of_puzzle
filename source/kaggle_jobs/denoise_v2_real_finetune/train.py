from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345"
EXPECTED_TRAIN_PAIRS_SHA256 = "70d0b7b3c15fefac62d6d1bf554f0e50a0f3473ddaf01423607b78cf0cde90c2"
EXPECTED_VAL_PAIRS_SHA256 = "78795a0a0ed1ee10bddac0c31222f2a9418c41d94249aa35ba183d15508928ed"
EXPECTED_LEGACY_CHECKPOINT_SHA256 = "d1df5a4e4852c821d79f72063866cf1fe09fb1beff913a4fb1034466d6ead96e"
EXPECTED_QUARANTINE_SHA256 = "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215"
EXPECTED_TRAINING_PIXELS_SHA256 = "d5ab28fbf439f031cbc8aed28af3b19bd5bc8a3ad253d1fa7b97b33a4bfe0fb3"
EXPECTED_VALIDATION_PIXELS_SHA256 = "fe573ed28b74b45e8b0302ad51c53ff0f7ad5ad907aa3d4d9332c87010e42bd5"
EXPECTED_OPENCV_VERSION = "4.11.0"
EXPECTED_FINE_TUNE_CODE_SHA256 = "12ba1a0f93311afd67c31080b29617daafd54ffb0a5edb2b9e2dfbff4b8fb04c"
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "pillow": "11.3.0",
    "scipy": "1.16.3",
    "skimage": "0.25.2",
    "torch": "2.6.0+cu124",
    "cuda": "12.4",
    "kornia": "0.8.3",
    "opencv": EXPECTED_OPENCV_VERSION,
    "jpeg_codec": "6.2",
    "libjpeg_turbo": "True",
    "libjpeg_turbo_version": "3.1.1",
}
# Pinned from the downloaded and independently verified synthetic-50k EMA.
# Keeping this as an explicit digest prevents an accidental unpinned launch.
EXPECTED_INIT_CHECKPOINT_SHA256 = "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"
FINE_TUNE_CODE_FILES = (
    "__init__.py",
    "degradation.py",
    "legacy_baseline.py",
    "losses.py",
    "metrics.py",
    "model.py",
    "real_pairs.py",
    "real_training.py",
    "real_validation.py",
    "tiles.py",
    "training.py",
)


for pin_name, pin_value in (
    ("EXPECTED_INIT_CHECKPOINT_SHA256", EXPECTED_INIT_CHECKPOINT_SHA256),
    ("EXPECTED_FINE_TUNE_CODE_SHA256", EXPECTED_FINE_TUNE_CODE_SHA256),
):
    if re.fullmatch(r"[0-9a-f]{64}", pin_value) is None:
        raise SystemExit(f"{pin_name} must be replaced with a lowercase SHA256 digest")


def install_runtime() -> None:
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
            "https://download.pytorch.org/whl/cu124",
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


def hash_fine_tune_package(package_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in FINE_TUNE_CODE_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((package_dir / name).read_bytes())
    return digest.hexdigest()

package_candidates = sorted(Path("/kaggle/input").rglob("src/puzzle_denoise_v2/__init__.py"))
if len(package_candidates) != 1:
    raise SystemExit(f"expected one puzzle_denoise_v2 package, found {package_candidates}")
package_init = package_candidates[0]
sys.path.insert(0, str(package_init.parents[1]))
code_sha256 = hash_fine_tune_package(package_init.parent)

manifest_candidates = sorted(Path("/kaggle/input").rglob("denoise_splits_seed20260710.json"))
if len(manifest_candidates) != 1:
    raise SystemExit(f"expected one split manifest, found {manifest_candidates}")
quarantine_candidates = sorted(
    Path("/kaggle/input").rglob("denoise_validation_quarantine_v1.json")
)
if len(quarantine_candidates) != 1:
    raise SystemExit(
        f"expected one validation-quarantine artifact, found {quarantine_candidates}"
    )
checkpoint_candidates = sorted(Path("/kaggle/input").rglob("tilenaf_synth_50k.pt"))
if len(checkpoint_candidates) != 1:
    raise SystemExit(f"expected one 50k checkpoint, found {checkpoint_candidates}")
legacy_checkpoint_candidates = sorted(
    Path("/kaggle/input").rglob("tile_restorer_1024_q90.pt")
)
if len(legacy_checkpoint_candidates) != 1:
    raise SystemExit(
        f"expected one pinned legacy checkpoint, found {legacy_checkpoint_candidates}"
    )
train_pair_candidates = sorted(Path("/kaggle/input").rglob("real_gold_train_512.npz"))
val_pair_candidates = sorted(Path("/kaggle/input").rglob("real_gold_val.npz"))
if len(train_pair_candidates) != 1 or len(val_pair_candidates) != 1:
    raise SystemExit(
        f"expected one train/val real-pair artifact, found {train_pair_candidates}, {val_pair_candidates}"
    )

manifest_sha256 = hashlib.sha256(manifest_candidates[0].read_bytes()).hexdigest()
quarantine_sha256 = hashlib.sha256(quarantine_candidates[0].read_bytes()).hexdigest()
train_pairs_sha256 = hashlib.sha256(train_pair_candidates[0].read_bytes()).hexdigest()
val_pairs_sha256 = hashlib.sha256(val_pair_candidates[0].read_bytes()).hexdigest()
checkpoint_sha256 = hashlib.sha256(checkpoint_candidates[0].read_bytes()).hexdigest()
legacy_checkpoint_sha256 = hashlib.sha256(
    legacy_checkpoint_candidates[0].read_bytes()
).hexdigest()
expected_inputs = {
    "manifest": (manifest_sha256, EXPECTED_MANIFEST_SHA256),
    "validation_quarantine": (quarantine_sha256, EXPECTED_QUARANTINE_SHA256),
    "train_pairs": (train_pairs_sha256, EXPECTED_TRAIN_PAIRS_SHA256),
    "val_pairs": (val_pairs_sha256, EXPECTED_VAL_PAIRS_SHA256),
    "init_checkpoint": (checkpoint_sha256, EXPECTED_INIT_CHECKPOINT_SHA256),
    "legacy_checkpoint": (legacy_checkpoint_sha256, EXPECTED_LEGACY_CHECKPOINT_SHA256),
    "fine_tune_code": (code_sha256, EXPECTED_FINE_TUNE_CODE_SHA256),
}
mismatched_inputs = {
    name: {"actual": actual, "expected": expected}
    for name, (actual, expected) in expected_inputs.items()
    if actual != expected
}
if mismatched_inputs:
    raise SystemExit(f"pinned input SHA256 mismatch: {mismatched_inputs}")

install_runtime()

import cv2
import kornia
import numpy as np
import PIL
from PIL import features as pillow_features
import scipy
import skimage
import torch

actual_runtime_versions = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pillow": PIL.__version__,
    "scipy": scipy.__version__,
    "skimage": skimage.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "kornia": kornia.__version__,
    "opencv": cv2.__version__,
    "jpeg_codec": str(pillow_features.version_codec("jpg")),
    "libjpeg_turbo": str(pillow_features.check_feature("libjpeg_turbo")),
    "libjpeg_turbo_version": str(
        pillow_features.version_feature("libjpeg_turbo")
    ),
}
runtime_mismatches = {
    name: {"actual": actual_runtime_versions[name], "expected": expected}
    for name, expected in EXPECTED_RUNTIME_VERSIONS.items()
    if actual_runtime_versions[name] != expected
}
if runtime_mismatches:
    raise SystemExit(f"pinned GPU runtime mismatch: {runtime_mismatches}")

from puzzle_denoise_v2.real_training import FineTuneConfig, fine_tune, fine_tune_code_fingerprint

if fine_tune_code_fingerprint() != code_sha256:
    raise SystemExit("fine-tune code changed between pre-install pinning and import")


probe = {
    "event": "gpu_probe",
    "runtime_versions": actual_runtime_versions,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "checkpoint": str(checkpoint_candidates[0]),
    "checkpoint_sha256": checkpoint_sha256,
    "legacy_checkpoint": str(legacy_checkpoint_candidates[0]),
    "legacy_checkpoint_sha256": legacy_checkpoint_sha256,
    "train_pairs": str(train_pair_candidates[0]),
    "train_pairs_sha256": train_pairs_sha256,
    "val_pairs": str(val_pair_candidates[0]),
    "val_pairs_sha256": val_pairs_sha256,
    "manifest_sha256": manifest_sha256,
    "validation_quarantine": str(quarantine_candidates[0]),
    "validation_quarantine_sha256": quarantine_sha256,
    "fine_tune_code_sha256": code_sha256,
    "expected_training_pixels_sha256": EXPECTED_TRAINING_PIXELS_SHA256,
    "expected_validation_pixels_sha256": EXPECTED_VALIDATION_PIXELS_SHA256,
    "opencv": cv2.__version__,
}
print(json.dumps(probe, sort_keys=True), flush=True)
if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (6, 0):
    raise SystemExit("compatible CUDA GPU unavailable")
try:
    left = torch.ones((64, 64), device="cuda", dtype=torch.float32)
    right = torch.eye(64, device="cuda", dtype=torch.float32)
    matmul_checksum = float((left @ right).sum().cpu())
    torch.cuda.synchronize()
except Exception as error:
    raise SystemExit(f"CUDA matmul probe failed: {error}") from error
if matmul_checksum != 4_096.0:
    raise SystemExit(f"CUDA matmul probe returned an unexpected checksum: {matmul_checksum}")
print(
    json.dumps(
        {
            "event": "cuda_matmul_probe",
            "checksum": matmul_checksum,
            "arch_list": torch.cuda.get_arch_list(),
        },
        sort_keys=True,
    ),
    flush=True,
)

data_root = Path("/kaggle/input/datasets/pasha883/vsos-ai-initiative-pazzle")
if not (data_root / "train" / "targets").is_dir():
    candidates = sorted(
        path.parent.parent
        for path in Path("/kaggle/input").rglob("train/targets")
        if path.is_dir()
    )
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one puzzle data root, found {candidates}")
    data_root = candidates[0]
required_data_directories = (
    data_root / "train" / "inputs",
    data_root / "train" / "targets",
    data_root / "test",
)
missing_data_directories = [path for path in required_data_directories if not path.is_dir()]
if missing_data_directories:
    raise SystemExit(f"puzzle data root is incomplete: {missing_data_directories}")

config = FineTuneConfig(
    data_root=str(data_root),
    manifest=str(manifest_candidates[0]),
    train_pairs=str(train_pair_candidates[0]),
    val_pairs=str(val_pair_candidates[0]),
    init_checkpoint=str(checkpoint_candidates[0]),
    legacy_checkpoint=str(legacy_checkpoint_candidates[0]),
    quarantine_artifact=str(quarantine_candidates[0]),
    output="/kaggle/working/tilenaf_real_finetune.pt",
    expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
    expected_train_pairs_sha256=EXPECTED_TRAIN_PAIRS_SHA256,
    expected_val_pairs_sha256=EXPECTED_VAL_PAIRS_SHA256,
    expected_init_checkpoint_sha256=EXPECTED_INIT_CHECKPOINT_SHA256,
    expected_legacy_checkpoint_sha256=EXPECTED_LEGACY_CHECKPOINT_SHA256,
    expected_quarantine_sha256=EXPECTED_QUARANTINE_SHA256,
    expected_training_pixels_sha256=EXPECTED_TRAINING_PIXELS_SHA256,
    expected_validation_pixels_sha256=EXPECTED_VALIDATION_PIXELS_SHA256,
    expected_opencv_version=EXPECTED_OPENCV_VERSION,
    gate_source_count=350,
    steps=4000,
    batch_size=256,
    pairs_per_real_source=32,
    synthetic_train_images=512,
    train_min_confidence=1.0,
    val_sensitivity_confidence=1.0,
    val_primary_confidence=1.5,
    val_pairs_per_source=8,
    peak_learning_rate=1e-5,
    encoder_lr_scale=0.5,
    min_lr_ratio=0.1,
    warmup_steps=100,
    eval_interval=500,
    log_interval=100,
    bootstrap_resamples=5000,
    no_gain_patience=3,
    no_gain_min_delta=1e-4,
    max_seconds=3600.0,
    seed=20260710,
    device="cuda",
)
result = fine_tune(config)
saved_output_path = Path(result["output"])
saved_checkpoint = torch.load(saved_output_path, map_location="cpu", weights_only=False)
if not isinstance(saved_checkpoint, dict):
    raise SystemExit("fine-tune output must contain a checkpoint dictionary")
if saved_checkpoint.get("safe_for_inference") is not True:
    raise SystemExit("fine-tune output is not marked safe for inference")
if result["rolled_back"]:
    rollback_contract = {
        "kind": "conservative_real_pair_fine_tune_rollback",
        "promotion_status": "rollback_safe",
        "rolled_back": True,
        "step": 0,
    }
    rollback_mismatches = {
        name: {"actual": saved_checkpoint.get(name), "expected": expected}
        for name, expected in rollback_contract.items()
        if saved_checkpoint.get(name) != expected
    }
    if rollback_mismatches:
        raise SystemExit(f"rollback checkpoint contract mismatch: {rollback_mismatches}")
else:
    promoted_step = result["best_step"]
    promoted_contract = {
        "kind": "conservative_real_pair_fine_tune",
        "promotion_status": "promoted",
        "rolled_back": False,
        "step": promoted_step,
        "best_step": promoted_step,
    }
    promoted_mismatches = {
        name: {"actual": saved_checkpoint.get(name), "expected": expected}
        for name, expected in promoted_contract.items()
        if saved_checkpoint.get(name) != expected
    }
    gate_validation = saved_checkpoint.get("gate_validation")
    if not isinstance(gate_validation, dict):
        promoted_mismatches["gate_validation"] = {
            "actual": type(gate_validation).__name__,
            "expected": "dict",
        }
    else:
        if gate_validation.get("panel") != "frozen_gate":
            promoted_mismatches["gate_validation.panel"] = {
                "actual": gate_validation.get("panel"),
                "expected": "frozen_gate",
            }
        if gate_validation.get("selected_step") != promoted_step:
            promoted_mismatches["gate_validation.selected_step"] = {
                "actual": gate_validation.get("selected_step"),
                "expected": promoted_step,
            }
        assessment = gate_validation.get("assessment")
        if not isinstance(assessment, dict) or assessment.get("eligible") is not True:
            promoted_mismatches["gate_validation.assessment.eligible"] = {
                "actual": None if not isinstance(assessment, dict) else assessment.get("eligible"),
                "expected": True,
            }
    if promoted_mismatches:
        raise SystemExit(f"promoted checkpoint contract mismatch: {promoted_mismatches}")
result["verified_output_contract"] = (
    "rollback_safe" if result["rolled_back"] else "promoted_frozen_gate"
)
result["output_sha256"] = hashlib.sha256(saved_output_path.read_bytes()).hexdigest()
Path("/kaggle/working/real_finetune_result.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps({"event": "job_complete", **result}, sort_keys=True), flush=True)
