#!/usr/bin/env python3
"""Stage 1: scientific train and calibration-B fixture preparation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import zipfile


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
EXPECTED_CODE_MANIFEST_SHA256 = "1583053a4276ba9b30368dcc6af00f24cd0fe091bfaff93090c41c01b0c3675b"
EXPECTED_CAPACITY_REPORT_SHA256 = "4fe36a4cf8fd637b519aa18da8a5b1c6ca762458fbebba3e33b226ebe3d09843"
EXPECTED_CAPACITY_WRAPPER_SHA256 = "b46d30c4486f0d2b2502f01993a181ac002bbdcc7c121ec30e60857a4fa2bb04"
ASSETS = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}
REQUIRED_LOCAL_FILES = {
    "scripts/train_evaluate_masked_gap.py",
    "src/puzzle_assembly/__init__.py",
    "src/puzzle_assembly/compatibility.py",
    "src/puzzle_assembly/components.py",
    "src/puzzle_assembly/geometry.py",
    "src/puzzle_assembly/learned.py",
    "src/puzzle_assembly/masked_gap.py",
    "src/puzzle_assembly/metrics.py",
    "src/puzzle_assembly/panels.py",
    "src/puzzle_assembly/protocol.py",
    "src/puzzle_assembly/solvers.py",
    "src/puzzle_denoise_v2/__init__.py",
    "src/puzzle_denoise_v2/degradation.py",
    "src/puzzle_denoise_v2/inference.py",
    "src/puzzle_denoise_v2/losses.py",
    "src/puzzle_denoise_v2/metrics.py",
    "src/puzzle_denoise_v2/model.py",
    "src/puzzle_denoise_v2/tiles.py",
    "src/puzzle_denoise_v2/training.py",
    "configs/denoise_splits_seed20260710.json",
    "configs/denoise_validation_quarantine_v1.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unresolved(value: str) -> bool:
    return len(value) != 64 or value.startswith("__")


def exactly_one(paths: list[Path], label: str) -> Path:
    values = sorted({path.resolve() for path in paths})
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, got {values}")
    return values[0]


def verify_code() -> tuple[Path, dict]:
    if unresolved(EXPECTED_CODE_MANIFEST_SHA256):
        raise RuntimeError("code manifest SHA placeholder is unresolved")
    manifest_path = exactly_one(
        [path for path in INPUT.rglob("masked_gap_code_manifest_v1.json") if sha256(path) == EXPECTED_CODE_MANIFEST_SHA256],
        "hash-pinned code manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "masked_gap_recursive_code_manifest_v1":
        raise RuntimeError("code manifest kind mismatch")
    files = manifest.get("file_sha256")
    if not isinstance(files, dict) or set(files) != REQUIRED_LOCAL_FILES:
        raise RuntimeError("recursive executable file closure mismatch")
    root = manifest_path.parent
    for relative, expected in files.items():
        path = (root / relative).resolve(strict=True)
        path.relative_to(root.resolve())
        if sha256(path) != expected:
            raise RuntimeError(f"recursive code hash mismatch: {relative}")
    return root, manifest


def exact_t4x2() -> dict:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("stage1 requires exactly two visible CUDA GPUs")
    devices = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
        if "T4" not in name.upper() or capability != [7, 5]:
            raise RuntimeError(f"stage1 requires exactly 2x T4 sm75: {name}, {capability}")
        value = torch.randn((64, 64), device=f"cuda:{index}")
        devices.append({"index": index, "name": name, "capability": capability, "tensor_op": float((value @ value).mean().cpu())})
    return {"devices": devices, "torch": torch.__version__, "cuda": torch.version.cuda}


def find_hash_pinned(name: str, expected: str) -> Path:
    if unresolved(expected):
        raise RuntimeError(f"unresolved expected SHA for {name}")
    return exactly_one([path for path in INPUT.rglob(name) if sha256(path) == expected], name)


def data_root() -> Path:
    return exactly_one(
        [path.parent.parent for path in INPUT.rglob("train/targets") if (path / "img_000151.png").is_file()],
        "puzzle data root",
    )


def run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    completed = subprocess.run(command, cwd=cwd, env=environment, check=False, text=True)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {command}")


def deterministic_zip(source: Path, output: Path) -> str:
    temporary = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(value for value in source.rglob("*") if value.is_file()):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 7, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=6)
    os.replace(temporary, output)
    return sha256(output)


def frozen_gate_names(code_root: Path, split: str) -> list[str]:
    import importlib.util

    protocol_path = code_root / "src/puzzle_assembly/protocol.py"
    spec = importlib.util.spec_from_file_location("masked_gap_protocol", protocol_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load hash-pinned split protocol")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    development = protocol.source_names_for_split(
        "edge_development",
        manifest_path=code_root / "configs/denoise_splits_seed20260710.json",
        quarantine_path=code_root / "configs/denoise_validation_quarantine_v1.json",
    )
    bounds = {"calibration_b": (388, 392), "holdout": (392, 400)}
    start, stop = bounds[split]
    names = development[start:stop]
    if len(names) != stop - start:
        raise RuntimeError("frozen secret-seed source selection drift")
    return names


def write_secret_seed_mapping(path: Path, *, split: str, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = secrets.SystemRandom()
    used: set[int] = set()
    records = []
    for name in names:
        for panel in ("primary_kornia", "independent_libjpeg"):
            seed = generator.getrandbits(64)
            while seed in used:
                seed = generator.getrandbits(64)
            used.add(seed)
            records.append({"name": name, "panel": panel, "seed": seed})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({
            "kind": "masked_gap_secret_panel_seed_mapping_v1",
            "split": split,
            "records": records,
        }, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    code_root, code_manifest = verify_code()
    hardware = exact_t4x2()
    selected = find_hash_pinned("selected_tilenaf_synth_50k.pt", ASSETS["selected_tilenaf_synth_50k.pt"])
    embedding = find_hash_pinned("hbt_d320_denoised_rgb_sobel.pt", ASSETS["hbt_d320_denoised_rgb_sobel.pt"])
    capacity = find_hash_pinned("masked_gap_t4_ddp_selection_v2.json", EXPECTED_CAPACITY_REPORT_SHA256)
    capacity_wrapper = find_hash_pinned(
        "masked_gap_t4_ddp_benchmark_wrapper_v2.json", EXPECTED_CAPACITY_WRAPPER_SHA256
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(code_root / "src"), str(code_root)])
    evaluator = code_root / "scripts/train_evaluate_masked_gap.py"
    training = WORKING / "training"
    run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(evaluator), "train",
        "--data-root", str(data_root()),
        "--denoiser", str(selected),
        "--embedding-checkpoint", str(embedding),
        "--manifest", str(code_root / "configs/denoise_splits_seed20260710.json"),
        "--quarantine", str(code_root / "configs/denoise_validation_quarantine_v1.json"),
        "--capacity-report", str(capacity),
        "--capacity-wrapper-report", str(capacity_wrapper),
        "--capacity-report-sha256", EXPECTED_CAPACITY_REPORT_SHA256,
        "--capacity-wrapper-report-sha256", EXPECTED_CAPACITY_WRAPPER_SHA256,
        "--output-dir", str(training),
        "--device", "cuda",
    ], cwd=code_root, environment=environment)
    input_only = WORKING / "calibration_b_input_only"
    labels = WORKING / "calibration_b_labels_only"
    secret_seed_mapping = labels / "secret_panel_seeds.json"
    write_secret_seed_mapping(
        secret_seed_mapping,
        split="calibration_b",
        names=frozen_gate_names(code_root, "calibration_b"),
    )
    run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(evaluator), "prepare",
        "--split", "calibration_b",
        "--data-root", str(data_root()),
        "--denoiser", str(selected),
        "--embedding-checkpoint", str(embedding),
        "--manifest", str(code_root / "configs/denoise_splits_seed20260710.json"),
        "--quarantine", str(code_root / "configs/denoise_validation_quarantine_v1.json"),
        "--input-dir", str(input_only), "--label-dir", str(labels),
        "--secret-seed-mapping", str(secret_seed_mapping),
        "--device", "cuda", "--require-ddp", "--overwrite",
    ], cwd=code_root, environment=environment)
    secret_seed_mapping.unlink()
    input_zip = WORKING / "calibration_b_input_only.zip"
    label_zip = WORKING / "calibration_b_labels_only.zip"
    input_archive_sha256 = deterministic_zip(input_only, input_zip)
    deterministic_zip(labels, label_zip)
    report = {
        "kind": "masked_gap_stage1_train_prepare_v1",
        "status": "complete",
        "safe_for_submission": False,
        "code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
        "code_manifest": code_manifest,
        "hardware": hardware,
        "capacity_report_sha256": EXPECTED_CAPACITY_REPORT_SHA256,
        "capacity_wrapper_report_sha256": EXPECTED_CAPACITY_WRAPPER_SHA256,
        "checkpoint_sha256": sha256(training / "masked_gap_gate.pt"),
        "training_report_sha256": sha256(training / "training_report.json"),
        "input_only_archive_sha256": input_archive_sha256,
        "label_only_archive_created": True,
        "holdout_prepared": False,
        "qap_run": False,
    }
    (WORKING / "masked_gap_stage1_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
