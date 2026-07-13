#!/usr/bin/env python3
"""Build one deterministic, self-verifying archive for the denoise-v2 release."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tempfile
from typing import Iterable
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_STEM = "denoise_v2_bundle_20260710"
BUNDLE_ROOT = PurePosixPath(BUNDLE_STEM)
RELEASE_UTC = "2026-07-10T09:24:18Z"
ZIP_TIMESTAMP = (2026, 7, 10, 9, 24, 18)


@dataclass(frozen=True)
class FileSpec:
    source: Path
    archive_path: PurePosixPath
    role: str


def _spec(source: str, destination: str, role: str) -> FileSpec:
    return FileSpec(PROJECT_ROOT / source, PurePosixPath(destination), role)


STATIC_SPECS = (
    _spec("DENOISE_V2.md", "docs/DENOISE_V2.md", "canonical_documentation"),
    _spec("ASSEMBLY_RESEARCH.md", "docs/ASSEMBLY_RESEARCH.md", "deferred_assembly_research"),
    _spec("DENOISE_PIPELINE.md", "docs/DENOISE_PIPELINE_V1_DEPRECATED.md", "deprecated_v1_documentation"),
    _spec("AGENTS.md", "reproducibility/AGENTS.md", "project_operating_rules"),
    _spec("environment.yml", "reproducibility/environment.yml", "environment_specification"),
    _spec("pyproject.toml", "reproducibility/pyproject.toml", "python_project_configuration"),
    _spec("configs/denoise_splits_seed20260710.json", "configs/denoise_splits_seed20260710.json", "source_split"),
    _spec("configs/denoise_validation_quarantine_v1.json", "configs/denoise_validation_quarantine_v1.json", "evaluation_partition"),
    _spec("runs/denoise_v2/release/selected_tilenaf_synth_50k.pt", "artifacts/model/selected_tilenaf_synth_50k.pt", "selected_checkpoint"),
    _spec("runs/denoise_v2/release/selected_model.json", "artifacts/model/selected_model.json", "frozen_selection_manifest"),
    _spec("runs/denoise_v2/release/final_audit.json", "results/final_audit.json", "final_release_audit"),
    _spec("runs/denoise_v2/release/example_test_img_000000.png", "examples/example_test_img_000000.png", "integration_example_image"),
    _spec("runs/denoise_v2/release/example_test_img_000000.json", "examples/example_test_img_000000.json", "integration_example_report"),
    _spec("runs/denoise/tile_restorer_1024_q90.pt", "artifacts/baselines/tile_restorer_1024_q90.pt", "legacy_baseline_checkpoint"),
    _spec("runs/denoise_v2/real_gold_train_512.npz", "data/derived_pairs/real_gold_train_512.npz", "derived_real_pair_train"),
    _spec("runs/denoise_v2/real_gold_val.npz", "data/derived_pairs/real_gold_val.npz", "derived_real_pair_validation"),
    _spec("runs/denoise_v2/release_readback/20260710T074500Z/synth/synth_50k_result.json", "results/synthetic_50k/synth_50k_result.json", "synthetic_training_report"),
    _spec("runs/denoise_v2/release_readback/20260710T074500Z/synth/vsos-denoise-v2-synthetic-50k.log", "results/synthetic_50k/training.log", "synthetic_training_log"),
    _spec("runs/denoise_v2/release_readback/20260710T074500Z/prefinetune_cpu_v3_current/prefinetune_calibration_report.json", "results/prefinetune_calibration/prefinetune_calibration_report.json", "clean_calibration_report"),
    _spec("runs/denoise_v2/release_readback/20260710T074500Z/prefinetune_cpu_v3_current/vsos-denoise-v2-prefinetune-calibration-cpu.log", "results/prefinetune_calibration/evaluation.log", "clean_calibration_log"),
    _spec("runs/denoise_v2/release_readback/20260710T074500Z/real_finetune_v2/real_finetune_result.json", "results/real_finetune/real_finetune_result.json", "real_finetune_report_not_promoted"),
    _spec("runs/denoise_v2/release_readback/20260710T074500Z/real_finetune_v2/vsos-denoise-v2-real-finetune.log", "results/real_finetune/training.log", "real_finetune_log_not_promoted"),
    _spec("runs/denoise_v2/release_readback/20260710T074500Z/real_finetune_v2/tilenaf_real_finetune.pt", "artifacts/not_promoted/real_finetune_rollback_safe_step0.pt", "rollback_safe_checkpoint_not_selected"),
    _spec("runs/denoise_v2/release_readback/20260710T074500Z/final_gate_cpu_v1/selected_final_gate_report.json", "results/final_gate/selected_final_gate_report.json", "one_shot_final_gate_report"),
    _spec("runs/denoise_v2/release_readback/20260710T074500Z/final_gate_cpu_v1/vsos-denoise-v2-one-shot-final-gate-cpu.log", "results/final_gate/evaluation.log", "one_shot_final_gate_log"),
    _spec("runs/denoise_v2/visual_qa/synthetic50k_calibration.png", "visual_qa/synthetic50k_calibration.png", "visual_qa_contact_sheet"),
    _spec("runs/denoise_v2/visual_qa/synthetic50k_calibration.json", "visual_qa/synthetic50k_calibration.json", "visual_qa_report"),
    _spec("runs/denoise_v2/release_readback/20260710T074500Z/prefinetune_cpu_v1_error/vsos-denoise-v2-prefinetune-calibration-cpu.log", "results/failed_attempts/prefinetune_cpu_v1_error.log", "archival_failure_log"),
    _spec("runs/denoise_v2/release_readback/20260710T074500Z/real_finetune_v1_error/vsos-denoise-v2-real-finetune.log", "results/failed_attempts/real_finetune_v1_error.log", "archival_failure_log"),
    _spec("kaggle_datasets/denoise_v2_code/denoise_v2_code.zip", "remote_payload/code/denoise_v2_code.zip", "kaggle_code_payload"),
    _spec("kaggle_datasets/denoise_v2_code/dataset-metadata.json", "remote_payload/code/dataset-metadata.json", "kaggle_dataset_metadata"),
    _spec("kaggle_datasets/denoise_v2_real_pairs/dataset-metadata.json", "remote_payload/real_pairs/dataset-metadata.json", "kaggle_dataset_metadata"),
    _spec("kaggle_datasets/denoise_v2_legacy_baseline/artifact-metadata.json", "remote_payload/legacy_baseline/artifact-metadata.json", "kaggle_artifact_metadata"),
    _spec("kaggle_datasets/denoise_v2_legacy_baseline/dataset-metadata.json", "remote_payload/legacy_baseline/dataset-metadata.json", "kaggle_dataset_metadata"),
)


CRITICAL_SHA256 = {
    "configs/denoise_splits_seed20260710.json": "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345",
    "configs/denoise_validation_quarantine_v1.json": "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215",
    "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "runs/denoise_v2/release/selected_model.json": "ce244ce8c9759be859262fd16560f8318814022883ec52cdc380ad490a924080",
    "runs/denoise_v2/release/final_audit.json": "bffe7eb6fde235fe3de69a3ee0f330655d0cd1eafbb30d2f5001ac6754be0748",
    "runs/denoise_v2/release_readback/20260710T074500Z/final_gate_cpu_v1/selected_final_gate_report.json": "afc5b311c3234048c5d28d1d5cb6d9745c4e4578b34de001bb2ae0fd86066264",
}


def _dynamic_specs() -> list[FileSpec]:
    specs: list[FileSpec] = []
    for source in sorted((PROJECT_ROOT / "src/puzzle_denoise_v2").glob("*.py")):
        specs.append(FileSpec(source, PurePosixPath("code/src/puzzle_denoise_v2") / source.name, "python_source"))
    for source in sorted((PROJECT_ROOT / "scripts").iterdir()):
        if source.is_file() and (source.suffix == ".py" or source.name == "doctor.sh"):
            specs.append(FileSpec(source, PurePosixPath("code/scripts") / source.name, "command_line_tool"))
    for source in sorted((PROJECT_ROOT / "tests").glob("test_*.py")):
        specs.append(FileSpec(source, PurePosixPath("code/tests") / source.name, "test"))
    job_names = (
        "denoise_v2_probe",
        "denoise_v2_torch26_probe",
        "denoise_v2_smoke",
        "denoise_v2_synth_50k",
        "denoise_v2_prefinetune_cpu",
        "denoise_v2_real_finetune",
        "denoise_v2_final_gate_cpu",
    )
    for job_name in job_names:
        job_dir = PROJECT_ROOT / "kaggle_jobs" / job_name
        for source in sorted(job_dir.iterdir()):
            if source.is_file() and (source.suffix == ".py" or source.name == "kernel-metadata.json"):
                specs.append(FileSpec(source, PurePosixPath("kaggle_jobs") / job_name / source.name, "kaggle_job"))
    return specs


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_specs(specs: Iterable[FileSpec]) -> list[FileSpec]:
    result = list(specs)
    archive_paths: set[PurePosixPath] = set()
    for spec in result:
        if not spec.source.is_file():
            raise FileNotFoundError(f"required bundle input is missing: {spec.source}")
        if spec.archive_path.is_absolute() or ".." in spec.archive_path.parts:
            raise ValueError(f"unsafe archive path: {spec.archive_path}")
        if spec.archive_path in archive_paths:
            raise ValueError(f"duplicate archive path: {spec.archive_path}")
        archive_paths.add(spec.archive_path)
    for source_relative, expected in CRITICAL_SHA256.items():
        actual = _sha256_file(PROJECT_ROOT / source_relative)
        if actual != expected:
            raise RuntimeError(
                f"critical artifact hash mismatch for {source_relative}: expected {expected}, got {actual}"
            )
    return sorted(result, key=lambda spec: spec.archive_path.as_posix())


README = """# Denoise V2 unified release bundle

This compact hand-off contains the selected synthetic-50k checkpoint,
leakage-safe split/provenance, source code, tests, Kaggle entrypoints, derived
real-pair supervision, baseline, decision-bearing reports/logs, visual QA and a
full-frame integration example.

Final frozen-gate result (350 untouched sources, 2800 pairs per panel):

- primary source-macro RGB SSIM: 0.81097747 vs legacy 0.77100406;
- primary delta: +0.03997341, 95% CI [+0.03810371, +0.04184709];
- sensitivity SSIM: 0.79937019 vs legacy 0.75684379;
- all six precommitted lower-bound checks passed.

`artifacts/model/selected_tilenaf_synth_50k.pt` is the only selected model.
`artifacts/not_promoted/real_finetune_rollback_safe_step0.pt` is audit evidence:
the fine-tune did not meet its precommitted +0.003 threshold. Unsafe/latest,
candidate and duplicate checkpoints are omitted. Raw `puzzle/` images, `.conda`
and unrelated submission/assembly outputs are also excluded. No assembly model
was trained.

Verify after extraction from the directory containing this README:

    shasum -a 256 -c SHA256SUMS

Recreate the environment from `reproducibility/environment.yml`, then run tests
from the extracted bundle root:

    PYTHONPATH=code:code/src python -m pytest -q code/tests

Example inference:

    PYTHONPATH=code/src python code/scripts/apply_denoise_v2.py \\
      --checkpoint artifacts/model/selected_tilenaf_synth_50k.pt \\
      --input /path/to/input.png \\
      --output /path/to/restored.png

See `docs/DENOISE_V2.md` for the protocol and `MANIFEST.json` for every file's
role, byte size, source-relative path and SHA256.
"""


def _zip_info(path: PurePosixPath, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo((BUNDLE_ROOT / path).as_posix(), date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = 0o755 if executable else 0o644
    info.external_attr = (mode & 0xFFFF) << 16
    return info


def _write_file(zf: zipfile.ZipFile, spec: FileSpec) -> None:
    executable = spec.archive_path.suffix in {".py", ".sh"} and "code/scripts" in spec.archive_path.as_posix()
    info = _zip_info(spec.archive_path, executable=executable)
    with spec.source.open("rb") as source, zf.open(info, "w", force_zip64=True) as destination:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            destination.write(chunk)


def _verify_archive(path: Path, expected_members: set[str]) -> None:
    """Verify member set, CRCs and every checksum using only archive contents."""

    with zipfile.ZipFile(path, "r") as zf:
        corrupt = zf.testzip()
        if corrupt is not None:
            raise RuntimeError(f"ZIP CRC validation failed at {corrupt}")
        if set(zf.namelist()) != expected_members:
            raise RuntimeError("ZIP member set differs from the package manifest")
        checksum_member = (BUNDLE_ROOT / "SHA256SUMS").as_posix()
        checksum_lines = zf.read(checksum_member).decode("utf-8").splitlines()
        checked: set[str] = set()
        for line in checksum_lines:
            expected_digest, relative_path = line.split("  ", maxsplit=1)
            member = (BUNDLE_ROOT / PurePosixPath(relative_path)).as_posix()
            if member in checked:
                raise RuntimeError(f"duplicate checksum member: {relative_path}")
            actual_digest = sha256(zf.read(member)).hexdigest()
            if actual_digest != expected_digest:
                raise RuntimeError(
                    f"internal SHA256 mismatch for {relative_path}: "
                    f"expected {expected_digest}, got {actual_digest}"
                )
            checked.add(member)
        if checked | {checksum_member} != expected_members:
            raise RuntimeError("SHA256SUMS does not cover every non-self archive member exactly once")


def build_bundle(output: Path) -> dict[str, object]:
    specs = _validate_specs((*STATIC_SPECS, *_dynamic_specs()))
    records: list[dict[str, object]] = []
    for spec in specs:
        records.append(
            {
                "path": spec.archive_path.as_posix(),
                "role": spec.role,
                "source_path": spec.source.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": spec.source.stat().st_size,
                "sha256": _sha256_file(spec.source),
            }
        )
    readme_bytes = README.encode("utf-8")
    records.append(
        {
            "path": "README.md",
            "role": "bundle_entrypoint",
            "source_path": None,
            "bytes": len(readme_bytes),
            "sha256": sha256(readme_bytes).hexdigest(),
        }
    )
    records.sort(key=lambda record: str(record["path"]))
    manifest = {
        "schema_version": 1,
        "kind": "denoise_v2_unified_release_bundle",
        "release_utc": RELEASE_UTC,
        "selected_checkpoint_sha256": CRITICAL_SHA256[
            "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
        ],
        "final_gate_passed": True,
        "tile_assembly_included": False,
        "raw_dataset_included": False,
        "files": records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    checksums = {str(record["path"]): str(record["sha256"]) for record in records}
    checksums["MANIFEST.json"] = sha256(manifest_bytes).hexdigest()
    checksum_bytes = "".join(
        f"{digest}  {path}\n" for path, digest in sorted(checksums.items())
    ).encode("utf-8")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as zf:
            for spec in specs:
                _write_file(zf, spec)
            zf.writestr(_zip_info(PurePosixPath("README.md")), readme_bytes)
            zf.writestr(_zip_info(PurePosixPath("MANIFEST.json")), manifest_bytes)
            zf.writestr(_zip_info(PurePosixPath("SHA256SUMS")), checksum_bytes)
        expected = {
            (BUNDLE_ROOT / spec.archive_path).as_posix() for spec in specs
        } | {
            (BUNDLE_ROOT / PurePosixPath(name)).as_posix()
            for name in ("README.md", "MANIFEST.json", "SHA256SUMS")
        }
        _verify_archive(temporary_path, expected)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "archive": str(output),
        "archive_sha256": _sha256_file(output),
        "archive_bytes": output.stat().st_size,
        "files": len(records) + 2,
        "bundle_root": BUNDLE_ROOT.as_posix(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs/denoise_v2" / f"{BUNDLE_STEM}.zip",
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_bundle(parse_args().output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
