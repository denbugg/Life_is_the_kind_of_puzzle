#!/usr/bin/env python3
"""Package the final tile-assembly submission, code, and evidence reproducibly.

The archive is deliberately compact.  It contains the final ``submission.zip``
as a nested, stored member; canonical source; the promoted model assets; job
definitions; and the latest authoritative summaries for every global-solver
branch.  Large candidate dumps and duplicate Kaggle payload/code trees are not
copied, but direct omitted artifacts are fingerprinted in ``MANIFEST.json``.

This command never opens raw puzzle inputs or targets.  It fully decodes every
PNG in the final nested submission and cross-checks names, sizes, SHA-256,
remote manifests, source hashes, model hashes, and archive CRCs before packing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile


ARCHIVE_TIMESTAMP = (2026, 7, 11, 0, 0, 0)
DEFAULT_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024

REQUIRED_FINAL_COMPANIONS = (
    "final_submission_manifest.json",
    "final_qap_submission_report.json",
    "final_qap_submission_run.json",
    "final_artifact_hashes.json",
    "final_run_artifact_hashes.json",
    "SHA256SUMS.txt",
)

ESSENTIAL_CHECKPOINTS = (
    "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/"
    "hbt_d320_denoised_rgb_sobel.pt",
)

CORE_SCRIPTS = (
    "scripts/build_assembly_submission.py",
    "scripts/package_final_assembly_bundle.py",
    "scripts/evaluate_assembly_baselines.py",
    "scripts/evaluate_real_assembly.py",
    "scripts/evaluate_selected_final_gate.py",
    "scripts/evaluate_fixed_layout_renderer.py",
    "scripts/train_context_reorg.py",
    "scripts/evaluate_context_reorg.py",
    "scripts/train_hyperedge_verifier.py",
    "scripts/evaluate_hyperedge_solver.py",
    "scripts/train_evaluate_dino_superblock.py",
)

JOB_DIRS = {
    "final_qap": "runs/assembly_v1/kaggle/final_qap_submission_job",
    "global_control": "runs/assembly_v1/kaggle/global_solvers_night_job",
    "qap_tuning": "runs/assembly_v1/kaggle/qap_tuning_night_job",
    "solver_rework_control": "runs/assembly_v1/kaggle/solver_rework_night_job",
    "line_cpsat": "runs/assembly_v1/kaggle/line_cpsat_gate_job",
    "context_reorg": "runs/assembly_v1/kaggle/context_reorg_gate_job",
    "mae_energy": "runs/assembly_v1/kaggle/mae_energy_gate_job",
    "mae_search": "runs/assembly_v1/kaggle/mae_search_gate_job",
    "hyperedge": "runs/assembly_v1/kaggle/hyperedge_gate_job",
    "dino_superblock": "runs/assembly_v1/kaggle/dino_superblock_probe_job",
    "lama_consistency": "runs/assembly_v1/kaggle/lama_consistency_gate_job",
}


@dataclass(frozen=True)
class EvidenceSpec:
    root: str
    require_analysis: bool = True
    require_json: bool = True
    require_log: bool = True


EVIDENCE = {
    "global_control": EvidenceSpec(
        "runs/assembly_v1/kaggle/global_solvers_night_output",
        require_analysis=False,
    ),
    "qap_tuning": EvidenceSpec(
        "runs/assembly_v1/kaggle/qap_tuning_night_output"
    ),
    "solver_rework_control": EvidenceSpec(
        "runs/assembly_v1/kaggle/solver_rework_night_job_download"
    ),
    "line_cpsat": EvidenceSpec(
        "runs/assembly_v1/kaggle/line_cpsat_gate_output"
    ),
    "context_reorg": EvidenceSpec(
        "runs/assembly_v1/kaggle/context_reorg_gate_output"
    ),
    "mae_energy": EvidenceSpec(
        "runs/assembly_v1/kaggle/mae_energy_gate_output"
    ),
    "mae_search": EvidenceSpec(
        "runs/assembly_v1/kaggle/mae_search_gate_output"
    ),
    "hyperedge": EvidenceSpec(
        "runs/assembly_v1/kaggle/hyperedge_gate_output"
    ),
    "dino_superblock": EvidenceSpec(
        "runs/assembly_v1/kaggle/dino_superblock_probe_output"
    ),
    # This branch was deliberately closed after three infrastructure attempts;
    # its authoritative ANALYSIS explains why no result JSON/log was produced.
    "lama_consistency": EvidenceSpec(
        "runs/assembly_v1/kaggle/lama_consistency_gate_output",
        require_json=False,
        require_log=False,
    ),
}

EXPLICIT_NESTED_EVIDENCE = {
    "dino_superblock": (
        "dino_superblock_code/reference/qap_l1w4_boundary_real16_manifest.json",
    ),
}

# Compact authoritative records for the pre-QAP restoration/local-scorer era.
# These directories predate the versioned vN evidence convention, so they are
# intentionally pinned file-by-file instead of routed through latest_version().
EARLY_EVIDENCE_FILES = (
    "runs/denoise_v2/release/selected_model.json",
    "runs/denoise_v2/denoise_v2_bundle_20260710/results/synthetic_50k/synth_50k_result.json",
    "runs/assembly_v1/FINAL_SOLVER_REPORT.md",
    "runs/assembly_v1/kaggle/l0_gpu_full/l0_gpu_full.json",
    "runs/assembly_v1/kaggle/l0_gpu_full/gpu_full_wrapper.json",
    "runs/assembly_v1/kaggle/l0_gpu_full/vsos-assembly-v1-l0-gpu-full.log",
    "runs/assembly_v1/kaggle/l1_gpu_full/l1_gpu_full.json",
    "runs/assembly_v1/kaggle/l1_gpu_full/gpu_full_wrapper.json",
    "runs/assembly_v1/kaggle/l1_gpu_full/vsos-assembly-v1-l1-gpu-full.log",
    "runs/assembly_v1/kaggle/l1v2_gpu_full/l1v2_gpu_full.json",
    "runs/assembly_v1/kaggle/l1v2_gpu_full/gpu_full_wrapper.json",
    "runs/assembly_v1/kaggle/l1v2_gpu_full/vsos-assembly-v1-l1v2-gpu-full.log",
    "runs/assembly_v1/kaggle/t0_gpu_full/t0_gpu_full.json",
    "runs/assembly_v1/kaggle/t0_gpu_full/gpu_full_wrapper.json",
    "runs/assembly_v1/kaggle/t0_gpu_full/vsos-assembly-v1-t0-gpu-full.log",
    "runs/assembly_v1/kaggle/x0_gpu_full/x0_gpu_full.json",
    "runs/assembly_v1/kaggle/x0_gpu_full/gpu_full_wrapper.json",
    "runs/assembly_v1/kaggle/x0_gpu_full/vsos-assembly-v1-x0-gpu-full.log",
    "runs/assembly_v1/kaggle/g0_global_matcher_gpu/g0_global_matcher_512x2.json",
    "runs/assembly_v1/kaggle/g0_global_matcher_gpu/vsos-assembly-v1-g0-global-matcher-gpu.log",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/gpu_full_wrapper.json",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_binary_edges.json",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_norm.json",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.json",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_sobel_only.json",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_raw_binary_edges.json",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_raw_rgb_sobel.json",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_raw_sobel_only.json",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/vsos-assembly-v1-edge2vec-gradient-gpu.log",
    "runs/assembly_v1/kaggle/seam_denoiser_gpu/gpu_full_wrapper.json",
    "runs/assembly_v1/kaggle/seam_denoiser_gpu/vsos-assembly-v1-seam-denoiser-gpu.log",
    "runs/assembly_v1/real_cal/real_cal_16_l1full.json",
    "runs/assembly_v1/real_cal/real_cal_16_l1full_t0full.json",
    "runs/assembly_v1/real_cal/real_cal_16_l1full_x0full.json",
    "runs/assembly_v1/real_cal/real_cal_16_l1full_x0full_t0full.json",
    "runs/assembly_v1/real_cal/real_cal_16_seamdenoise_l1full_t0full.json",
    "runs/assembly_v1/real_cal/real_cal_64_selectedfusion_seamrender.json",
)

EARLY_HASH_ONLY_FILES = (
    "runs/assembly_v1/real_cal/real_cal_64_l1full_x0full_t0full.json",
    "runs/assembly_v1/real_cal/real_cal_16_hbt_d320_denoised_rgb_sobel.json",
    "runs/assembly_v1/real_cal/real_cal_16_hbt_d320_denoised_rgb_norm.json",
    "runs/assembly_v1/real_cal/real_cal_64_selecteddenoise_classical.json",
    "runs/assembly_v1/kaggle/seam_denoiser_gpu/seam_denoiser_gpu.pt",
)

REPORT_MARKER_GROUPS = {
    "QAP": ("qap",),
    "solver-rework control": ("solver rework", "solver-rework", "lns"),
    "line/CP-SAT": ("cp-sat", "cpsat"),
    "context reorganization": (
        "context reorganization",
        "context-reorganization",
        "context reorg",
        "context-reorg",
    ),
    "MAE": ("mae",),
    "hyperedge": ("hyperedge",),
    "DINO": ("dino",),
    "LaMa": ("lama",),
}

JOB_ALLOWED_SUFFIXES = {".py", ".json", ".md", ".txt", ".yaml", ".yml"}
EVIDENCE_ALLOWED_SUFFIXES = {".json", ".md", ".log", ".txt", ".csv", ".tsv"}
SUBMISSION_COMPANION_SUFFIXES = {".json", ".md", ".log", ".txt"}


@dataclass(frozen=True)
class BundleEntry:
    source: Path
    archive_path: str
    role: str
    origin: str


@dataclass(frozen=True)
class GeneratedEntry:
    payload: bytes
    archive_path: str
    role: str
    origin: str = "generated"


@dataclass(frozen=True)
class OmittedFile:
    source: Path
    origin: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-dir",
        type=Path,
        required=True,
        help="downloaded final Kaggle output directory containing submission.zip",
    )
    parser.add_argument("--output", type=Path, required=True, help="final bundle ZIP")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="defaults to runs/assembly_v1/FINAL_SOLVER_REPORT.md",
    )
    parser.add_argument(
        "--research-note",
        type=Path,
        help="defaults to ASSEMBLY_RESEARCH.md",
    )
    parser.add_argument(
        "--essential-checkpoint",
        type=Path,
        action="append",
        help=(
            "repeat to replace the two default promoted checkpoint paths; "
            "every supplied path is required"
        ),
    )
    parser.add_argument(
        "--expected-submission-members",
        type=int,
        default=700,
        help="expected number of flat PNG members in the nested submission ZIP",
    )
    parser.add_argument(
        "--max-artifact-bytes",
        type=int,
        default=DEFAULT_MAX_ARTIFACT_BYTES,
        help="maximum direct evidence/config file size before hash-only omission",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"required {label} must not be a symlink: {path}")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"required {label} directory must not be a symlink: {path}")
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"required {label} directory is missing: {path}")
    return path


def semantic_origin(path: Path, project_root: Path, submission_dir: Path) -> str:
    path = path.resolve()
    try:
        return "project:" + path.relative_to(project_root).as_posix()
    except ValueError:
        pass
    try:
        return "submission:" + path.relative_to(submission_dir).as_posix()
    except ValueError:
        return "external:" + path.name


def validate_archive_path(value: str) -> str:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or value.endswith("/"):
        raise ValueError(f"unsafe archive path: {value!r}")
    if not candidate.parts:
        raise ValueError("empty archive path")
    return candidate.as_posix()


def add_entry(
    entries: dict[str, BundleEntry],
    *,
    source: Path,
    archive_path: str,
    role: str,
    project_root: Path,
    submission_dir: Path,
) -> None:
    source = require_file(source, role)
    archive_path = validate_archive_path(archive_path)
    if archive_path in entries:
        raise RuntimeError(f"duplicate bundle member: {archive_path}")
    entries[archive_path] = BundleEntry(
        source=source,
        archive_path=archive_path,
        role=role,
        origin=semantic_origin(source, project_root, submission_dir),
    )


def latest_version(root: Path, label: str) -> Path:
    root = require_dir(root, label)
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = re.fullmatch(r"v(\d+)", child.name)
        if match and child.is_dir() and not child.is_symlink():
            candidates.append((int(match.group(1)), child.resolve()))
    if not candidates:
        raise RuntimeError(f"no version directories found for {label}: {root}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def validate_submission_zip(path: Path, expected_count: int) -> dict[str, object]:
    if expected_count <= 0:
        raise ValueError("--expected-submission-members must be positive")
    path = require_file(path, "final submission.zip")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise RuntimeError(f"submission ZIP has a corrupt member: {corrupt}")
            infos = archive.infolist()
            payload_records: list[dict[str, object]] = []
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError(
                    "Pillow is required for final RGB/480x480 PNG validation"
                ) from exc
            for info in sorted(infos, key=lambda item: item.filename):
                payload = archive.read(info)
                with Image.open(BytesIO(payload)) as image:
                    image.load()
                    if image.format != "PNG":
                        raise RuntimeError(
                            f"submission member is not a PNG: {info.filename}"
                        )
                    if image.mode != "RGB" or image.size != (480, 480):
                        raise RuntimeError(
                            "invalid submission image "
                            f"{info.filename}: mode={image.mode}, size={image.size}"
                        )
                payload_records.append(
                    {
                        "name": info.filename,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"invalid final submission ZIP: {path}") from exc
    names = [info.filename for info in infos]
    if len(names) != expected_count:
        raise RuntimeError(
            f"submission member count is {len(names)}, expected {expected_count}"
        )
    if len(names) != len(set(names)):
        raise RuntimeError("submission ZIP contains duplicate names")
    invalid = [
        name
        for name in names
        if PurePosixPath(name).name != name
        or not name.lower().endswith(".png")
        or name in {"", ".", ".."}
    ]
    if invalid:
        raise RuntimeError(f"submission ZIP has non-flat/non-PNG members: {invalid[:5]}")
    if any(info.is_dir() for info in infos):
        raise RuntimeError("submission ZIP contains directory entries")
    return {
        "member_count": len(names),
        "first_member": min(names),
        "last_member": max(names),
        "validation": "CRC, flat unique names, per-member SHA-256, and full RGB/480x480 PIL decode",
        "member_records": payload_records,
    }


def validate_final_report(report: Path) -> None:
    text = report.read_text(encoding="utf-8").casefold()
    forbidden = ("todo_", "draft", "черновик")
    stale = [marker for marker in forbidden if marker in text]
    if stale:
        raise RuntimeError(
            "final report still contains draft/TODO markers: " + ", ".join(stale)
        )
    missing = [
        label
        for label, alternatives in REPORT_MARKER_GROUPS.items()
        if not any(marker.casefold() in text for marker in alternatives)
    ]
    if missing:
        raise RuntimeError(
            "final report is stale/incomplete; missing experiment sections: "
            + ", ".join(missing)
        )


def load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(require_file(path, label).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload


def require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError(f"invalid SHA-256 for {label}: {value!r}")
    return value


def artifact_index(payload: dict[str, object], label: str) -> dict[str, dict[str, object]]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(f"{label}.artifacts must be a non-empty list")
    result: dict[str, dict[str, object]] = {}
    for record in artifacts:
        if not isinstance(record, dict):
            raise RuntimeError(f"{label} contains a non-object artifact record")
        name = record.get("path")
        if not isinstance(name, str) or PurePosixPath(name).name != name:
            raise RuntimeError(f"unsafe artifact path in {label}: {name!r}")
        if name in result:
            raise RuntimeError(f"duplicate artifact path in {label}: {name}")
        require_digest(record.get("sha256"), f"{label}:{name}")
        if not isinstance(record.get("bytes"), int) or int(record["bytes"]) < 0:
            raise RuntimeError(f"invalid byte count in {label}:{name}")
        result[name] = record
    return result


def parse_sha256s(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(
        require_file(path, "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise RuntimeError(f"malformed SHA256SUMS.txt line {line_number}")
        digest, name = parts
        require_digest(digest, f"SHA256SUMS.txt line {line_number}")
        if PurePosixPath(name).name != name or name in result:
            raise RuntimeError(f"unsafe/duplicate SHA256SUMS entry: {name!r}")
        result[name] = digest
    if not result:
        raise RuntimeError("SHA256SUMS.txt is empty")
    return result


def verify_recorded_file(
    submission_dir: Path,
    name: str,
    record: dict[str, object],
    sums: dict[str, str],
) -> None:
    path = require_file(submission_dir / name, f"recorded final artifact {name}")
    size, digest = fingerprint(path)
    expected_digest = require_digest(record.get("sha256"), name)
    if size != record.get("bytes") or digest != expected_digest:
        raise RuntimeError(
            f"recorded artifact mismatch for {name}: "
            f"expected bytes/hash {record.get('bytes')}/{expected_digest}, "
            f"got {size}/{digest}"
        )
    if sums.get(name) != digest:
        raise RuntimeError(f"SHA256SUMS mismatch or missing entry for {name}")


def validate_final_provenance(
    *,
    project_root: Path,
    submission_dir: Path,
    submission: Path,
    promoted_assets: list[Path],
    submission_validation: dict[str, object],
) -> dict[str, object]:
    manifest_path = submission_dir / "final_submission_manifest.json"
    report_path = submission_dir / "final_qap_submission_report.json"
    run_path = submission_dir / "final_qap_submission_run.json"
    hashes_path = submission_dir / "final_artifact_hashes.json"
    run_hashes_path = submission_dir / "final_run_artifact_hashes.json"
    sums_path = submission_dir / "SHA256SUMS.txt"

    manifest = load_json_object(manifest_path, "final submission manifest")
    report = load_json_object(report_path, "deterministic final report")
    run = load_json_object(run_path, "operational final run report")
    deterministic_records = artifact_index(
        load_json_object(hashes_path, "deterministic artifact hashes"),
        "final_artifact_hashes",
    )
    operational_records = artifact_index(
        load_json_object(run_hashes_path, "operational artifact hashes"),
        "final_run_artifact_hashes",
    )
    overlap = sorted(set(deterministic_records) & set(operational_records))
    if overlap:
        raise RuntimeError(
            "artifact appears in both deterministic and operational hash manifests: "
            + ", ".join(overlap)
        )
    sums = parse_sha256s(sums_path)

    for name, record in {**deterministic_records, **operational_records}.items():
        verify_recorded_file(submission_dir, name, record, sums)
    for index_path in (hashes_path, run_hashes_path):
        _, digest = fingerprint(index_path)
        if sums.get(index_path.name) != digest:
            raise RuntimeError(
                f"SHA256SUMS mismatch or missing entry for {index_path.name}"
            )

    submission_size, submission_digest = fingerprint(submission)
    expected_submission_values = {
        "manifest": manifest.get("archive_sha256"),
        "deterministic report": (
            report.get("submission", {}).get("sha256")
            if isinstance(report.get("submission"), dict)
            else None
        ),
        "operational run report": (
            run.get("submission", {}).get("sha256")
            if isinstance(run.get("submission"), dict)
            else None
        ),
        "deterministic artifact hashes": deterministic_records.get(
            submission.name, {}
        ).get("sha256"),
        "SHA256SUMS": sums.get(submission.name),
    }
    for label, value in expected_submission_values.items():
        if require_digest(value, f"{label} submission") != submission_digest:
            raise RuntimeError(f"submission.zip SHA-256 disagrees with {label}")
    if manifest.get("archive_bytes") != submission_size:
        raise RuntimeError("submission.zip byte count disagrees with manifest")

    member_records = submission_validation.pop("member_records")
    manifest_members = manifest.get("members")
    if not isinstance(manifest_members, list):
        raise RuntimeError("final submission manifest lacks members list")
    if manifest_members != member_records:
        raise RuntimeError(
            "submission member names/bytes/SHA-256 disagree with final manifest"
        )
    if manifest.get("member_count") != len(member_records):
        raise RuntimeError("submission member count disagrees with final manifest")

    report_manifest = report.get("manifest")
    run_manifest = run.get("manifest")
    for label, value in (
        (
            "deterministic report",
            report_manifest.get("sha256") if isinstance(report_manifest, dict) else None,
        ),
        (
            "operational run report",
            run_manifest.get("sha256") if isinstance(run_manifest, dict) else None,
        ),
    ):
        _, actual = fingerprint(manifest_path)
        if require_digest(value, f"{label} manifest") != actual:
            raise RuntimeError(f"final manifest SHA-256 disagrees with {label}")

    local_assets = {path.name: require_file(path, "promoted asset") for path in promoted_assets}
    report_assets = report.get("assets")
    if not isinstance(report_assets, list) or not report_assets:
        raise RuntimeError("deterministic report lacks final solver assets")
    for record in report_assets:
        if not isinstance(record, dict) or not isinstance(record.get("filename"), str):
            raise RuntimeError("malformed final solver asset record")
        name = str(record["filename"])
        local = local_assets.get(name)
        if local is None:
            raise RuntimeError(f"remote final solver asset is not packaged: {name}")
        size, digest = fingerprint(local)
        if size != record.get("bytes") or digest != require_digest(
            record.get("sha256"), f"remote asset {name}"
        ):
            raise RuntimeError(f"remote/local promoted asset mismatch: {name}")

    report_code = report.get("code")
    if not isinstance(report_code, dict) or not isinstance(report_code.get("contract"), list):
        raise RuntimeError("deterministic report lacks final code contract")
    for record in report_code["contract"]:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError("malformed final code-contract record")
        relative = PurePosixPath(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe final code-contract path: {relative}")
        local = require_file(project_root / Path(*relative.parts), "final solver source")
        size, digest = fingerprint(local)
        if size != record.get("bytes") or digest != require_digest(
            record.get("sha256"), f"remote source {relative}"
        ):
            raise RuntimeError(f"remote/local final source mismatch: {relative}")

    return {
        **submission_validation,
        "submission_bytes": submission_size,
        "submission_sha256": submission_digest,
        "manifest_member_records_matched": True,
        "deterministic_and_operational_hash_manifests_matched": True,
        "remote_code_and_asset_contracts_matched_local_bundle": True,
    }


def collect_source(
    *,
    project_root: Path,
    submission_dir: Path,
    entries: dict[str, BundleEntry],
) -> None:
    for relative in ("AGENTS.md", "environment.yml", *CORE_SCRIPTS):
        source = project_root / relative
        add_entry(
            entries,
            source=source,
            archive_path="source/" + relative,
            role="canonical source/reproducibility file",
            project_root=project_root,
            submission_dir=submission_dir,
        )
    for package in ("puzzle_assembly", "puzzle_denoise_v2"):
        package_root = require_dir(project_root / "src" / package, package)
        files = sorted(package_root.glob("*.py"))
        if not files:
            raise RuntimeError(f"no Python source found in {package_root}")
        for source in files:
            add_entry(
                entries,
                source=source,
                archive_path=f"source/src/{package}/{source.name}",
                role=f"canonical {package} source",
                project_root=project_root,
                submission_dir=submission_dir,
            )


def collect_jobs(
    *,
    project_root: Path,
    submission_dir: Path,
    entries: dict[str, BundleEntry],
    omitted: list[OmittedFile],
    omitted_directories: list[dict[str, str]],
    max_bytes: int,
) -> None:
    for label, relative in JOB_DIRS.items():
        job_root = require_dir(project_root / relative, f"{label} job")
        require_file(job_root / "kernel-metadata.json", f"{label} kernel metadata")
        runners = sorted(job_root.glob("run_*.py"))
        if not runners:
            raise RuntimeError(f"{label} job has no run_*.py: {job_root}")
        for source in sorted(job_root.rglob("*")):
            if not source.is_file() or source.is_symlink():
                continue
            relative_job = source.relative_to(job_root)
            if "__pycache__" in relative_job.parts:
                continue
            if relative_job.parts[0] in {"src", "scripts"}:
                continue
            origin = semantic_origin(source, project_root, submission_dir)
            if source.suffix.lower() in {".zip", ".pt", ".pth", ".ckpt"}:
                omitted.append(
                    OmittedFile(
                        source,
                        origin,
                        "embedded job payload/checkpoint; canonical source or promoted asset "
                        "is stored once elsewhere",
                    )
                )
                continue
            if source.suffix.lower() not in JOB_ALLOWED_SUFFIXES:
                continue
            if source.stat().st_size > max_bytes:
                omitted.append(
                    OmittedFile(source, origin, "job file exceeds compact-artifact limit")
                )
                continue
            add_entry(
                entries,
                source=source,
                archive_path=f"jobs/{label}/{relative_job.as_posix()}",
                role=f"{label} job definition",
                project_root=project_root,
                submission_dir=submission_dir,
            )
        for duplicate in (job_root / "src", job_root / "scripts"):
            if duplicate.is_dir():
                omitted_directories.append(
                    {
                        "origin": semantic_origin(duplicate, project_root, submission_dir),
                        "reason": "duplicate mounted/staged code tree; canonical source stored once",
                    }
                )


def collect_evidence(
    *,
    project_root: Path,
    submission_dir: Path,
    entries: dict[str, BundleEntry],
    omitted: list[OmittedFile],
    omitted_directories: list[dict[str, str]],
    max_bytes: int,
) -> dict[str, str]:
    selected_versions: dict[str, str] = {}
    for label, spec in EVIDENCE.items():
        version = latest_version(project_root / spec.root, f"{label} evidence")
        selected_versions[label] = version.name
        direct_files = sorted(path for path in version.iterdir() if path.is_file())
        if spec.require_analysis and not (version / "ANALYSIS.md").is_file():
            raise RuntimeError(f"{label} latest evidence lacks ANALYSIS.md: {version}")
        if spec.require_json and not any(path.suffix.lower() == ".json" for path in direct_files):
            raise RuntimeError(f"{label} latest evidence has no direct JSON: {version}")
        if spec.require_log and not any(path.suffix.lower() == ".log" for path in direct_files):
            raise RuntimeError(f"{label} latest evidence has no direct log: {version}")
        included = 0
        for source in direct_files:
            origin = semantic_origin(source, project_root, submission_dir)
            suffix = source.suffix.lower()
            if suffix not in EVIDENCE_ALLOWED_SUFFIXES:
                omitted.append(
                    OmittedFile(
                        source,
                        origin,
                        "binary preview/checkpoint/payload excluded from compact evidence",
                    )
                )
                continue
            if source.stat().st_size > max_bytes:
                omitted.append(
                    OmittedFile(
                        source,
                        origin,
                        "authoritative raw/candidate dump exceeds compact-artifact limit; "
                        "fingerprint retained",
                    )
                )
                continue
            add_entry(
                entries,
                source=source,
                archive_path=f"evidence/{label}/{version.name}/{source.name}",
                role=f"{label} authoritative evidence",
                project_root=project_root,
                submission_dir=submission_dir,
            )
            included += 1
        for relative_name in EXPLICIT_NESTED_EVIDENCE.get(label, ()):
            relative = PurePosixPath(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe explicit evidence path: {relative_name}")
            source = require_file(
                version / Path(*relative.parts),
                f"{label} explicit nested evidence",
            )
            if source.stat().st_size > max_bytes:
                raise RuntimeError(
                    f"required explicit evidence exceeds compact-artifact limit: {source}"
                )
            add_entry(
                entries,
                source=source,
                archive_path=(
                    f"evidence/{label}/{version.name}/{relative.as_posix()}"
                ),
                role=f"{label} authoritative nested evidence",
                project_root=project_root,
                submission_dir=submission_dir,
            )
            included += 1
        if included == 0:
            raise RuntimeError(f"no compact {label} evidence selected from {version}")
        for child in sorted(path for path in version.iterdir() if path.is_dir()):
            omitted_directories.append(
                {
                    "origin": semantic_origin(child, project_root, submission_dir),
                    "reason": (
                        "remaining nested previews or duplicate mounted code/results omitted; "
                        "explicit authoritative nested files are stored separately"
                    ),
                }
            )
    return selected_versions


def collect_early_evidence(
    *,
    project_root: Path,
    submission_dir: Path,
    entries: dict[str, BundleEntry],
    omitted: list[OmittedFile],
    max_bytes: int,
) -> None:
    for relative_name in EARLY_EVIDENCE_FILES:
        relative = PurePosixPath(relative_name)
        source = require_file(
            project_root / Path(*relative.parts), "early compact evidence"
        )
        if source.stat().st_size > max_bytes:
            raise RuntimeError(
                f"pinned early evidence exceeds compact-artifact limit: {source}"
            )
        archive_relative = (
            "HISTORICAL_PRE_QAP_REPORT.md"
            if relative_name == "runs/assembly_v1/FINAL_SOLVER_REPORT.md"
            else relative.as_posix()
        )
        add_entry(
            entries,
            source=source,
            archive_path="evidence/early/" + archive_relative,
            role="authoritative compact pre-QAP evidence",
            project_root=project_root,
            submission_dir=submission_dir,
        )
    for relative_name in EARLY_HASH_ONLY_FILES:
        relative = PurePosixPath(relative_name)
        source = require_file(
            project_root / Path(*relative.parts), "early hash-only evidence"
        )
        omitted.append(
            OmittedFile(
                source,
                semantic_origin(source, project_root, submission_dir),
                "authoritative early raw report/non-promoted checkpoint exceeds compact scope; "
                "SHA-256 and byte size retained",
            )
        )


def collect_submission(
    *,
    project_root: Path,
    submission_dir: Path,
    entries: dict[str, BundleEntry],
    omitted: list[OmittedFile],
    max_bytes: int,
) -> Path:
    submission = require_file(submission_dir / "submission.zip", "final submission")
    add_entry(
        entries,
        source=submission,
        archive_path="submission/submission.zip",
        role="final 700-image nested submission",
        project_root=project_root,
        submission_dir=submission_dir,
    )
    required_companions = {
        require_file(submission_dir / filename, f"final companion {filename}")
        for filename in REQUIRED_FINAL_COMPANIONS
    }
    for source in sorted(path for path in submission_dir.iterdir() if path.is_file()):
        if source == submission:
            continue
        origin = semantic_origin(source, project_root, submission_dir)
        if source.suffix.lower() == ".zip":
            omitted.append(
                OmittedFile(source, origin, "preflight/shard ZIP; merged submission is canonical")
            )
            continue
        if source.suffix.lower() not in SUBMISSION_COMPANION_SUFFIXES:
            continue
        if source.stat().st_size > max_bytes:
            if source.resolve() in required_companions:
                raise RuntimeError(
                    f"required final companion exceeds compact-artifact limit: {source}"
                )
            omitted.append(
                OmittedFile(source, origin, "final companion exceeds compact-artifact limit")
            )
            continue
        add_entry(
            entries,
            source=source,
            archive_path="submission/metadata/" + source.name,
            role="final submission provenance",
            project_root=project_root,
            submission_dir=submission_dir,
        )
    return submission


def fingerprint(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"artifact changed while hashing: {path}")
    if size != before.st_size:
        raise RuntimeError(f"short read while hashing: {path}")
    return size, digest.hexdigest()


def zip_info(name: str, *, stored: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(validate_archive_path(name), ARCHIVE_TIMESTAMP)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    return info


def stream_entry(
    archive: zipfile.ZipFile, entry: BundleEntry
) -> dict[str, object]:
    before = entry.source.stat()
    stored = entry.source.suffix.lower() in {".zip", ".pt", ".pth", ".ckpt"}
    info = zip_info(entry.archive_path, stored=stored)
    info.file_size = before.st_size
    digest = hashlib.sha256()
    size = 0
    with entry.source.open("rb") as source, archive.open(
        info, "w", force_zip64=True
    ) as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            target.write(chunk)
    after = entry.source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"artifact changed while packaging: {entry.source}")
    if size != before.st_size:
        raise RuntimeError(f"short read while packaging: {entry.source}")
    return {
        "archive_path": entry.archive_path,
        "origin": entry.origin,
        "role": entry.role,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "zip_method": "stored" if stored else "deflated",
    }


def write_generated(
    archive: zipfile.ZipFile, entry: GeneratedEntry
) -> dict[str, object]:
    info = zip_info(entry.archive_path, stored=False)
    archive.writestr(info, entry.payload, compresslevel=9)
    return {
        "archive_path": entry.archive_path,
        "origin": entry.origin,
        "role": entry.role,
        "bytes": len(entry.payload),
        "sha256": hashlib.sha256(entry.payload).hexdigest(),
        "zip_method": "deflated",
    }


def bundle_readme() -> bytes:
    return (
        "# Final tile-assembly bundle\n\n"
        "`submission/submission.zip` is the canonical nested Kaggle submission.\n"
        "`report/` and `research/` contain the final narrative and research note; "
        "`source/` contains canonical code and the two promoted model assets; "
        "`jobs/` contains Kaggle job definitions; and `evidence/` contains compact "
        "authoritative experiment artifacts.\n\n"
        "`MANIFEST.json` records SHA-256, byte size, semantic origin, selected "
        "evidence versions, and exclusion rationale. Large raw candidate dumps are "
        "hash-only omissions. Puzzle data, PNG previews, duplicate mounted code "
        "trees, caches, and non-promoted checkpoints are intentionally absent.\n"
    ).encode("utf-8")


def package(args: argparse.Namespace) -> dict[str, object]:
    project_root = require_dir(args.project_root, "project root")
    submission_dir = require_dir(args.submission_dir, "submission")
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("--output must end in .zip")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    if args.max_artifact_bytes <= 0:
        raise ValueError("--max-artifact-bytes must be positive")

    report = require_file(
        args.report or project_root / "runs/assembly_v1/FINAL_SOLVER_REPORT.md",
        "final solver report",
    )
    validate_final_report(report)
    research = require_file(
        args.research_note or project_root / "ASSEMBLY_RESEARCH.md",
        "assembly research note",
    )

    entries: dict[str, BundleEntry] = {}
    omitted: list[OmittedFile] = []
    omitted_directories: list[dict[str, str]] = []
    collect_source(
        project_root=project_root,
        submission_dir=submission_dir,
        entries=entries,
    )
    add_entry(
        entries,
        source=report,
        archive_path="report/FINAL_SOLVER_REPORT.md",
        role="final experiment report",
        project_root=project_root,
        submission_dir=submission_dir,
    )
    add_entry(
        entries,
        source=research,
        archive_path="research/ASSEMBLY_RESEARCH.md",
        role="assembly research note",
        project_root=project_root,
        submission_dir=submission_dir,
    )

    checkpoint_values = args.essential_checkpoint or [
        project_root / relative for relative in ESSENTIAL_CHECKPOINTS
    ]
    promoted_assets: list[Path] = []
    for checkpoint in checkpoint_values:
        checkpoint = checkpoint if checkpoint.is_absolute() else project_root / checkpoint
        checkpoint = require_file(checkpoint, "essential promoted model asset")
        promoted_assets.append(checkpoint)
        add_entry(
            entries,
            source=checkpoint,
            archive_path="source/promoted_assets/" + checkpoint.name,
            role="essential promoted model asset",
            project_root=project_root,
            submission_dir=submission_dir,
        )

    collect_jobs(
        project_root=project_root,
        submission_dir=submission_dir,
        entries=entries,
        omitted=omitted,
        omitted_directories=omitted_directories,
        max_bytes=args.max_artifact_bytes,
    )
    selected_versions = collect_evidence(
        project_root=project_root,
        submission_dir=submission_dir,
        entries=entries,
        omitted=omitted,
        omitted_directories=omitted_directories,
        max_bytes=args.max_artifact_bytes,
    )
    collect_early_evidence(
        project_root=project_root,
        submission_dir=submission_dir,
        entries=entries,
        omitted=omitted,
        max_bytes=args.max_artifact_bytes,
    )
    submission = collect_submission(
        project_root=project_root,
        submission_dir=submission_dir,
        entries=entries,
        omitted=omitted,
        max_bytes=args.max_artifact_bytes,
    )
    submission_validation = validate_submission_zip(
        submission, args.expected_submission_members
    )
    submission_validation = validate_final_provenance(
        project_root=project_root,
        submission_dir=submission_dir,
        submission=submission,
        promoted_assets=promoted_assets,
        submission_validation=submission_validation,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    if output in {entry.source for entry in entries.values()}:
        raise RuntimeError("output path aliases a required input artifact")

    generated = GeneratedEntry(
        payload=bundle_readme(),
        archive_path="BUNDLE_README.md",
        role="bundle navigation",
    )
    records: list[dict[str, object]] = []
    omitted_records: list[dict[str, object]] = []
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            records.append(write_generated(archive, generated))
            for archive_path in sorted(entries):
                records.append(stream_entry(archive, entries[archive_path]))

            for item in sorted(omitted, key=lambda value: value.origin):
                size, digest = fingerprint(item.source)
                omitted_records.append(
                    {
                        "origin": item.origin,
                        "bytes": size,
                        "sha256": digest,
                        "reason": item.reason,
                    }
                )

            manifest = {
                "schema": "assembly_final_bundle_manifest_v1",
                "determinism": {
                    "member_order": "BUNDLE_README.md, sorted input paths, MANIFEST.json",
                    "member_timestamp": "2026-07-11T00:00:00",
                    "text_compression": "deflate level 9",
                    "already-compressed_assets": "stored",
                    "filesystem_metadata_preserved": False,
                },
                "submission_validation": submission_validation,
                "selected_evidence_versions": selected_versions,
                "max_compact_artifact_bytes": args.max_artifact_bytes,
                "entries": records,
                "exclusions": {
                    "policies": [
                        {
                            "pattern": "puzzle/** and extracted submission PNGs",
                            "reason": "raw/user puzzle data is never part of the evidence bundle",
                        },
                        {
                            "pattern": "**/__pycache__/**",
                            "reason": "runtime cache is non-source and platform-specific",
                        },
                        {
                            "pattern": "nested Kaggle src/scripts/code trees",
                            "reason": "canonical source is stored once under source/",
                        },
                        {
                            "pattern": "PNG previews/contact sheets",
                            "reason": "compact textual evidence is authoritative",
                        },
                        {
                            "pattern": "non-promoted checkpoints and raw candidate dumps",
                            "reason": "not required to run the final solver; direct omissions are fingerprinted",
                        },
                    ],
                    "omitted_files": omitted_records,
                    "omitted_directories": sorted(
                        omitted_directories,
                        key=lambda value: (value["origin"], value["reason"]),
                    ),
                },
                "manifest_note": (
                    "MANIFEST.json cannot self-hash. The package command prints the "
                    "SHA-256 of the completed outer ZIP."
                ),
            }
            manifest_payload = (
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            archive.writestr(
                zip_info("MANIFEST.json", stored=False),
                manifest_payload,
                compresslevel=9,
            )
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output appeared during packaging: {output}")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    output_bytes, output_sha256 = fingerprint(output)
    return {
        "output": str(output),
        "bytes": output_bytes,
        "sha256": output_sha256,
        "packaged_entries_excluding_manifest": len(records),
        "omitted_files_fingerprinted": len(omitted_records),
        "selected_evidence_versions": selected_versions,
    }


def main() -> int:
    args = parse_args()
    try:
        result = package(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
