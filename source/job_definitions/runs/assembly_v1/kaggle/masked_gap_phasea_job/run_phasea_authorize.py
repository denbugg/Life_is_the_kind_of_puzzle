#!/usr/bin/env python3
"""Stage 2: physically isolated input-only dense scoring and authorization."""

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
EXPECTED_DATASET_SOURCES = [
    "pasha883/vsos-masked-gap-gate-code",
    "pasha883/vsos-masked-gap-stage1-checkpoint-v1",
    "pasha883/vsos-masked-gap-calb-input-v1",
]
REQUIRED_LOCAL_FILES = {
    "scripts/train_evaluate_masked_gap.py",
    "src/puzzle_assembly/__init__.py",
    "src/puzzle_assembly/compatibility.py", "src/puzzle_assembly/geometry.py",
    "src/puzzle_assembly/components.py",
    "src/puzzle_assembly/learned.py", "src/puzzle_assembly/masked_gap.py",
    "src/puzzle_assembly/metrics.py", "src/puzzle_assembly/panels.py",
    "src/puzzle_assembly/protocol.py", "src/puzzle_assembly/solvers.py",
    "src/puzzle_denoise_v2/__init__.py", "src/puzzle_denoise_v2/degradation.py",
    "src/puzzle_denoise_v2/inference.py", "src/puzzle_denoise_v2/losses.py",
    "src/puzzle_denoise_v2/metrics.py", "src/puzzle_denoise_v2/model.py",
    "src/puzzle_denoise_v2/tiles.py", "src/puzzle_denoise_v2/training.py",
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


def verify_code() -> Path:
    if unresolved(EXPECTED_CODE_MANIFEST_SHA256):
        raise RuntimeError("unresolved code manifest SHA")
    path = exactly_one([p for p in INPUT.rglob("masked_gap_code_manifest_v1.json") if sha256(p) == EXPECTED_CODE_MANIFEST_SHA256], "code manifest")
    payload = json.loads(path.read_text())
    files = payload.get("file_sha256")
    if payload.get("kind") != "masked_gap_recursive_code_manifest_v1" or not isinstance(files, dict) or set(files) != REQUIRED_LOCAL_FILES:
        raise RuntimeError("recursive code closure mismatch")
    root = path.parent
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
            raise RuntimeError("Phase A dataset source allowlist drift")
    lowered = " ".join(sources).lower()
    if any(token in lowered for token in ("pazzle", "puzzle", "label", "target")):
        raise RuntimeError("Phase A allowlist contains a forbidden target/label dataset")
    forbidden = list(INPUT.rglob("label_manifest.json")) + list(INPUT.rglob("train/targets"))
    if forbidden:
        raise RuntimeError(f"Phase A sees forbidden label/target paths: {forbidden}")
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
                f"Phase A mounted dataset set differs from allowlist: {actual_sources}"
            )
        mounted_names = sorted(actual_sources)
        layout = "kaggle_datasets_hierarchy"
    else:
        allowed_mounts = {source.rsplit("/", 1)[-1] for source in sources}
        actual_mounts = {path.name for path in INPUT.iterdir() if path.is_dir()}
        if actual_mounts != allowed_mounts:
            raise RuntimeError(
                f"Phase A mounted dataset set differs from allowlist: {actual_mounts}"
            )
        mounted_names = sorted(actual_mounts)
        layout = "legacy_flat"
    return {"dataset_sources": sources, "mounted_names": mounted_names, "mount_layout": layout, "puzzle_mounted": False, "labels_mounted": False}


def exact_t4x2() -> list[dict]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Phase A requires exactly 2 GPUs")
    values = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
        if "T4" not in name.upper() or capability != [7, 5]:
            raise RuntimeError("Phase A requires exactly 2x T4 sm75")
        values.append({"index": index, "name": name, "capability": capability})
    return values


def verify_input_manifest(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("kind") != "masked_gap_input_manifest_v1"
        or payload.get("split") != "calibration_b"
        or payload.get("target_or_label_fields_attached") is not False
        or payload.get("panel_seed_attached") is not False
        or payload.get("panel_seed_derivation_available") is not False
        or set(payload.get("allowed_npz_keys", []))
        != {"raw_tiles", "denoised_tiles", "w4_right", "w4_down"}
    ):
        raise RuntimeError("input manifest isolation contract mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 8:
        raise RuntimeError("input manifest record count mismatch")
    root = path.parent.resolve()
    for record in records:
        candidate = (root / record["file"]).resolve(strict=True)
        candidate.relative_to(root)
        if sha256(candidate) != record.get("sha256"):
            raise RuntimeError(f"input record hash mismatch: {record.get('file')}")


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    completed = subprocess.run(command, cwd=cwd, env=env, check=False, text=True)
    if completed.returncode:
        raise RuntimeError(f"command failed: {command}: {completed.returncode}")


def main() -> None:
    isolation = verify_dataset_isolation()
    code = verify_code()
    hardware = exact_t4x2()
    if unresolved(EXPECTED_CHECKPOINT_SHA256) or unresolved(EXPECTED_INPUT_MANIFEST_SHA256):
        raise RuntimeError("Phase A intermediate SHA placeholders unresolved")
    checkpoint = exactly_one([p for p in INPUT.rglob("masked_gap_gate.pt") if sha256(p) == EXPECTED_CHECKPOINT_SHA256], "checkpoint")
    manifest = exactly_one(
        [p for p in INPUT.rglob("input_manifest.json") if sha256(p) == EXPECTED_INPUT_MANIFEST_SHA256],
        "input manifest",
    )
    verify_input_manifest(manifest)
    output = WORKING / "phase_a"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(code / "src"), str(code)])
    evaluator = code / "scripts/train_evaluate_masked_gap.py"
    run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(evaluator), "phase-a",
        "--input-manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--output-dir", str(output), "--device", "cuda", "--require-ddp",
    ], cwd=code, env=env)
    authorization = WORKING / "phase_b_authorization.json"
    run([
        sys.executable, str(evaluator), "authorize",
        "--phase-a-report", str(output / "phase_a_report.json"),
        "--phase-a-artifact", str(output / "phase_a_scores.npz"),
        "--output", str(authorization),
    ], cwd=code, env=env)
    report = {
        "kind": "masked_gap_isolated_phase_a_v1", "status": "complete",
        "isolation": isolation, "hardware": hardware,
        "code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
        "phase_a_report_sha256": sha256(output / "phase_a_report.json"),
        "phase_a_artifact_sha256": sha256(output / "phase_a_scores.npz"),
        "authorization_sha256": sha256(authorization),
        "target_metrics_opened": False, "puzzle_mounted": False, "labels_mounted": False,
    }
    (WORKING / "masked_gap_phasea_stage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
