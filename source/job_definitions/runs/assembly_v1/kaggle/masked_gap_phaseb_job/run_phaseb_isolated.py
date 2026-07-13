#!/usr/bin/env python3
"""Stage 3: label-enabled Phase B with no puzzle dataset mounted."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
EXPECTED_CODE_MANIFEST_SHA256 = "1583053a4276ba9b30368dcc6af00f24cd0fe091bfaff93090c41c01b0c3675b"
EXPECTED_CHECKPOINT_SHA256 = "79447dce4c5943abceb1ec166685a6724fb0b7c10446d20d4a1b11be74afdf48"
EXPECTED_INPUT_MANIFEST_SHA256 = "fbe74bfa263bfdea43c53941c8fddaf30468cfc89aee5ccee74697f9aef3ea8a"
EXPECTED_LABEL_MANIFEST_SHA256 = "02f9498c5c5c87ecdd241083fc5992b9820cdd45d52e1a37852295bcdfad7fb7"
EXPECTED_PHASE_A_REPORT_SHA256 = "a2dc227f7b669deab90f324abe84688264930be9416d8d2618f4baed5b20b05a"
EXPECTED_PHASE_A_ARTIFACT_SHA256 = "a1ab46b9efa81c8a76a355342d1b150bdcf84a5fae4c38d5cc93828710487f20"
EXPECTED_AUTHORIZATION_SHA256 = "7802414dac0c655102c45d0cd370dda778b13dadc7642ff8340539e4dbc05227"
EXPECTED_CALIBRATION_B_REPORT_SHA256 = "__CALIBRATION_B_REPORT_SHA256_IF_HOLDOUT__"
EXPECTED_DATASET_SOURCES = [
    "pasha883/vsos-masked-gap-gate-code",
    "pasha883/vsos-masked-gap-stage1-checkpoint-v1",
    "pasha883/vsos-masked-gap-calb-input-v1",
    "pasha883/vsos-masked-gap-calb-labels-v1",
    "pasha883/vsos-masked-gap-calb-phasea-v1",
]
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


def verify_dataset_isolation() -> dict:
    sources = EXPECTED_DATASET_SOURCES
    metadata_path = Path(__file__).with_name("kernel-metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("dataset_sources") != sources:
            raise RuntimeError("Phase B dataset source allowlist drift")
    if not isinstance(sources, list) or len(sources) not in (5, 6):
        raise RuntimeError("Phase B requires code/checkpoint/input/label/PhaseA mounts and optional calibration-B proof")
    if sources[:5] != [
        "pasha883/vsos-masked-gap-gate-code",
        "pasha883/vsos-masked-gap-stage1-checkpoint-v1",
        "pasha883/vsos-masked-gap-calb-input-v1",
        "pasha883/vsos-masked-gap-calb-labels-v1",
        "pasha883/vsos-masked-gap-calb-phasea-v1",
    ]:
        raise RuntimeError("Phase B dataset source allowlist drift")
    lowered = " ".join(sources).lower()
    if any(token in lowered for token in ("pazzle", "puzzle", "train-target", "train_target")):
        raise RuntimeError("Phase B allowlist contains a forbidden puzzle/target dataset")
    forbidden = list(INPUT.rglob("train/targets"))
    if forbidden:
        raise RuntimeError(f"Phase B sees forbidden puzzle target paths: {forbidden}")
    nested = INPUT / "datasets"
    if nested.is_dir():
        actual_sources = {
            f"{owner.name}/{dataset.name}"
            for owner in nested.iterdir()
            if owner.is_dir()
            for dataset in owner.iterdir()
            if dataset.is_dir()
        }
        if actual_sources != set(sources):
            raise RuntimeError(
                f"Phase B mounted dataset set differs from allowlist: {actual_sources}"
            )
        mounted_names = sorted(actual_sources)
        layout = "kaggle_datasets_hierarchy"
    else:
        allowed_mounts = {source.rsplit("/", 1)[-1] for source in sources}
        actual_mounts = {path.name for path in INPUT.iterdir() if path.is_dir()}
        if actual_mounts != allowed_mounts:
            raise RuntimeError(
                f"Phase B mounted dataset set differs from allowlist: {actual_mounts}"
            )
        mounted_names = sorted(actual_mounts)
        layout = "legacy_flat"
    return {"dataset_sources": sources, "mounted_names": mounted_names, "mount_layout": layout, "puzzle_mounted": False}


def exact_t4x2() -> list[dict]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Phase B requires exactly two visible GPUs")
    devices = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
        if "T4" not in name.upper() or capability != [7, 5]:
            raise RuntimeError("Phase B requires exactly 2x T4 sm75")
        devices.append({"index": index, "name": name, "capability": capability})
    return devices


def verify_manifest_records(path: Path, *, kind: str, allowed_keys: set[str]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("kind") != kind
        or payload.get("split") != "calibration_b"
        or set(payload.get("allowed_npz_keys", [])) != allowed_keys
    ):
        raise RuntimeError(f"{kind} contract mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 8:
        raise RuntimeError(f"{kind} record count mismatch")
    root = path.parent.resolve()
    for record in records:
        candidate = (root / record["file"]).resolve(strict=True)
        candidate.relative_to(root)
        if sha256(candidate) != record.get("sha256"):
            raise RuntimeError(f"record hash mismatch: {record.get('file')}")
    return payload


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    completed = subprocess.run(command, cwd=cwd, env=env, check=False, text=True)
    if completed.returncode:
        raise RuntimeError(f"command failed: {command}: {completed.returncode}")


def main() -> None:
    isolation = verify_dataset_isolation()
    code = verify_code()
    hardware = exact_t4x2()
    checkpoint = find_hash_pinned("masked_gap_gate.pt", EXPECTED_CHECKPOINT_SHA256)
    input_manifest = find_hash_pinned("input_manifest.json", EXPECTED_INPUT_MANIFEST_SHA256)
    input_payload = verify_manifest_records(
        input_manifest,
        kind="masked_gap_input_manifest_v1",
        allowed_keys={"raw_tiles", "denoised_tiles", "w4_right", "w4_down"},
    )
    if (
        input_payload.get("target_or_label_fields_attached") is not False
        or input_payload.get("panel_seed_attached") is not False
        or input_payload.get("panel_seed_derivation_available") is not False
    ):
        raise RuntimeError("input manifest isolation contract mismatch")
    phase_a_report = find_hash_pinned("phase_a_report.json", EXPECTED_PHASE_A_REPORT_SHA256)
    phase_a_artifact = find_hash_pinned("phase_a_scores.npz", EXPECTED_PHASE_A_ARTIFACT_SHA256)
    authorization = find_hash_pinned("phase_b_authorization.json", EXPECTED_AUTHORIZATION_SHA256)

    # Only after every target-blind input/code/checkpoint/authorization hash is
    # confirmed do we hash or open the label-only manifest and records.
    label_manifest = find_hash_pinned("label_manifest.json", EXPECTED_LABEL_MANIFEST_SHA256)
    verify_manifest_records(
        label_manifest,
        kind="masked_gap_label_manifest_v1",
        allowed_keys={"slot_to_target", "clean_slot_tiles"},
    )
    split = input_payload.get("split")
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(code / "scripts/train_evaluate_masked_gap.py"),
        "phase-b",
        "--input-manifest", str(input_manifest),
        "--label-manifest", str(label_manifest),
        "--phase-a-report", str(phase_a_report),
        "--phase-a-artifact", str(phase_a_artifact),
        "--authorization", str(authorization),
        "--checkpoint", str(checkpoint),
        "--output", str(WORKING / "phase_b_report.json"),
        "--device", "cuda",
        "--require-ddp",
    ]
    calibration_b_sha = None
    if split == "holdout":
        calibration_b = find_hash_pinned(
            "phase_b_report.json", EXPECTED_CALIBRATION_B_REPORT_SHA256
        )
        command.extend(["--calibration-b-report", str(calibration_b)])
        calibration_b_sha = EXPECTED_CALIBRATION_B_REPORT_SHA256
    elif split != "calibration_b":
        raise RuntimeError(f"unsupported Phase B split: {split}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(code / "src"), str(code)])
    run(command, cwd=code, env=env)
    output = WORKING / "phase_b_report.json"
    report = {
        "kind": "masked_gap_isolated_phase_b_stage_v1",
        "status": "complete",
        "split": split,
        "isolation": isolation,
        "hardware": hardware,
        "code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
        "label_manifest_sha256": EXPECTED_LABEL_MANIFEST_SHA256,
        "phase_a_report_sha256": EXPECTED_PHASE_A_REPORT_SHA256,
        "phase_a_artifact_sha256": EXPECTED_PHASE_A_ARTIFACT_SHA256,
        "authorization_sha256": EXPECTED_AUTHORIZATION_SHA256,
        "calibration_b_report_sha256": calibration_b_sha,
        "phase_b_report_sha256": sha256(output),
        "puzzle_mounted": False,
    }
    (WORKING / "masked_gap_phaseb_stage_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
