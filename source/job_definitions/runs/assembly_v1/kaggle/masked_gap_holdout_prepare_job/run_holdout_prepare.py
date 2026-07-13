#!/usr/bin/env python3
"""Prepare sealed holdout fixtures only after a hash-pinned calibration-B pass."""

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
EXPECTED_CALIBRATION_B_REPORT_SHA256 = "__PASSING_CALIBRATION_B_REPORT_SHA256__"
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


def find_hash_pinned(name: str, expected: str) -> Path:
    if unresolved(expected):
        raise RuntimeError(f"unresolved expected SHA for {name}")
    return exactly_one([path for path in INPUT.rglob(name) if sha256(path) == expected], name)


def verify_code() -> Path:
    manifest_path = find_hash_pinned(
        "masked_gap_code_manifest_v1.json", EXPECTED_CODE_MANIFEST_SHA256
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("file_sha256")
    if (
        payload.get("kind") != "masked_gap_recursive_code_manifest_v1"
        or not isinstance(files, dict)
        or set(files) != REQUIRED_LOCAL_FILES
    ):
        raise RuntimeError("recursive code closure mismatch")
    root = manifest_path.parent
    for relative, expected in files.items():
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root.resolve())
        if sha256(candidate) != expected:
            raise RuntimeError(f"code hash mismatch: {relative}")
    return root


def exact_t4x2() -> list[dict]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("holdout preparation requires exactly two visible GPUs")
    devices = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
        if "T4" not in name.upper() or capability != [7, 5]:
            raise RuntimeError("holdout preparation requires exactly 2x T4 sm75")
        devices.append({"index": index, "name": name, "capability": capability})
    return devices


def data_root() -> Path:
    return exactly_one(
        [path.parent.parent for path in INPUT.rglob("train/targets") if (path / "img_000151.png").is_file()],
        "puzzle data root",
    )


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    completed = subprocess.run(command, cwd=cwd, env=env, check=False, text=True)
    if completed.returncode:
        raise RuntimeError(f"command failed: {command}: {completed.returncode}")


def deterministic_zip(source: Path, output: Path) -> str:
    temporary = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(value for value in source.rglob("*") if value.is_file()):
            info = zipfile.ZipInfo(path.relative_to(source).as_posix(), date_time=(2026, 7, 13, 0, 0, 0))
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
    code = verify_code()
    calibration_b = find_hash_pinned(
        "phase_b_report.json", EXPECTED_CALIBRATION_B_REPORT_SHA256
    )
    calibration_payload = json.loads(calibration_b.read_text(encoding="utf-8"))
    if (
        calibration_payload.get("kind") != "masked_gap_phase_b_report_v1"
        or calibration_payload.get("split") != "calibration_b"
        or calibration_payload.get("decision", {}).get("passed") is not True
    ):
        raise RuntimeError("holdout remains sealed because calibration B did not pass")

    # No puzzle image or upstream model is opened until the exact passing gate
    # report above has been hash- and semantics-validated.
    hardware = exact_t4x2()
    selected = find_hash_pinned(
        "selected_tilenaf_synth_50k.pt", ASSETS["selected_tilenaf_synth_50k.pt"]
    )
    embedding = find_hash_pinned(
        "hbt_d320_denoised_rgb_sobel.pt", ASSETS["hbt_d320_denoised_rgb_sobel.pt"]
    )
    input_only = WORKING / "holdout_input_only"
    labels_only = WORKING / "holdout_labels_only"
    secret_seed_mapping = labels_only / "secret_panel_seeds.json"
    write_secret_seed_mapping(
        secret_seed_mapping,
        split="holdout",
        names=frozen_gate_names(code, "holdout"),
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(code / "src"), str(code)])
    run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(code / "scripts/train_evaluate_masked_gap.py"),
        "prepare",
        "--split", "holdout",
        "--data-root", str(data_root()),
        "--denoiser", str(selected),
        "--embedding-checkpoint", str(embedding),
        "--manifest", str(code / "configs/denoise_splits_seed20260710.json"),
        "--quarantine", str(code / "configs/denoise_validation_quarantine_v1.json"),
        "--input-dir", str(input_only),
        "--label-dir", str(labels_only),
        "--secret-seed-mapping", str(secret_seed_mapping),
        "--device", "cuda",
        "--require-ddp",
        "--overwrite",
    ], cwd=code, env=env)
    secret_seed_mapping.unlink()
    input_archive = WORKING / "holdout_input_only.zip"
    label_archive = WORKING / "holdout_labels_only.zip"
    input_archive_sha256 = deterministic_zip(input_only, input_archive)
    deterministic_zip(labels_only, label_archive)
    report = {
        "kind": "masked_gap_holdout_prepare_after_calibration_b_v1",
        "status": "complete",
        "safe_for_submission": False,
        "calibration_b_report_sha256": EXPECTED_CALIBRATION_B_REPORT_SHA256,
        "code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
        "hardware": hardware,
        "input_only_archive_sha256": input_archive_sha256,
        "label_only_archive_created": True,
        "qap_run": False,
    }
    (WORKING / "masked_gap_holdout_prepare_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
