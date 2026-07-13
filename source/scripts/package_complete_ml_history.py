#!/usr/bin/env python3
"""Build the final restoration+assembly history bundle deterministically.

The bundle contains canonical source/config/tests, compact decision-bearing
history, the best scored submission, promoted runtime assets, and the final
narrative. Large duplicate or non-promoted experiment artifacts are not copied;
they are fingerprinted in the manifest instead. Raw puzzle data and credentials
are never inspected or packaged.
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
import tempfile
import zipfile

from PIL import Image


ARCHIVE_TIMESTAMP = (2026, 7, 13, 0, 0, 0)
MAX_COMPACT_BYTES = 8 * 1024 * 1024
HISTORY_SUFFIXES = {".json", ".md", ".txt", ".csv", ".tsv", ".yaml", ".yml"}
SOURCE_SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml", ".toml"}
ROOT_DOCS = (
    "AGENTS.md",
    "environment.yml",
    "pyproject.toml",
    "ASSEMBLY_RESEARCH.md",
    "DENOISE_PIPELINE.md",
    "DENOISE_V2.md",
    "FINAL_EXPERIMENTS_REPORT.md",
    "TILE_ASSEMBLY_HANDOFF.md",
    "COMPLETE_ML_TASK_HISTORY.md",
)
SOURCE_ROOTS = ("src", "scripts", "configs", "tests", "kaggle_jobs")
HISTORY_ROOTS = ("runs/denoise_v2", "runs/assembly_v1")
JOB_DEFINITION_SUFFIXES = SOURCE_SUFFIXES | {".md", ".txt"}
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".idea",
    ".conda",
    ".cache",
    "dino_model_cache",
    "fixture_label",
    "site-packages",
}
EXCLUDED_PATH_FRAGMENTS = (
    "/torch/hub/",
    "/huggingface/",
    "/.kaggle/",
    "/puzzle/train/",
    "/puzzle/test/",
)
DENY_MEMBER_TOKENS = (
    "kaggle.json",
    ".kaggle",
    ".conda",
    "access_token",
    "credentials",
    "token",
    "secret",
    "password",
    "api_key",
    ".env",
    "puzzle/train",
    "puzzle/test",
    "dino_model_cache",
    "__pycache__",
)
BEST_SUBMISSION = "runs/assembly_v1/kaggle/luma_harmonized_submission_output/v1/submission.zip"
BEST_SUBMISSION_SHA256 = "099d1c5fe69cda8519a4f19750cb3a481ac87999c294a35e19691a849d4c6096"
PROMOTED_ASSETS = (
    "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt",
    "runs/assembly_v1/kaggle/seam_denoiser_gpu/seam_denoiser_gpu.pt",
)
EXPLICIT_COMPACT = (
    (
        "runs/assembly_v1/kaggle/masked_gap_gate_code_dataset/upload_v1/masked_gap_gate_code.zip",
        "masked_gap/masked_gap_gate_code.zip",
        "masked-gap protocol evidence",
    ),
    (
        "runs/assembly_v1/kaggle/masked_gap_stage1_output/v4_required_readback/training/masked_gap_gate.pt",
        "masked_gap/masked_gap_gate.pt",
        "masked-gap Stage 1 synchronized checkpoint",
    ),
    (
        "runs/denoise_v2/release/SHA256SUMS",
        "promoted_assets/denoiser_release_SHA256SUMS.txt",
        "promoted denoiser release checksums",
    ),
)
EXPLICIT_HASH_ONLY = (
    "runs/assembly_v1/kaggle/mae_search_gate_output/v3/mae_search_gate_report.json",
    "runs/assembly_v1/real_cal/real_cal_64_l1full_x0full_t0full.json",
    "runs/assembly_v1/real_cal/real_cal_16_hbt_d320_denoised_rgb_sobel.json",
    "runs/assembly_v1/real_cal/real_cal_16_hbt_d320_denoised_rgb_norm.json",
    "runs/assembly_v1/real_cal/real_cal_64_selecteddenoise_classical.json",
    "runs/assembly_v1/final_tile_assembly_bundle.zip",
    "runs/denoise_v2/denoise_v2_bundle_20260710.zip",
    "runs/assembly_v1/neural_upgrade_results_bundle_v1.zip",
    "runs/assembly_v1/kaggle/final_qap_submission_output/v1/submission.zip",
    "runs/assembly_v1/harmonized_submission/local_full700_v1/submission.zip",
    "submission.zip",
    "submission_harmonized_v1.zip",
    "runs/assembly_v1/spatial_prior/spatial_prior_512.joblib",
    "runs/assembly_v1/candidate_edge_verifier_v1/candidate_edge_verifier.joblib",
    "runs/assembly_v1/candidate_edge_verifier_v1_p80/candidate_edge_verifier.joblib",
    "runs/assembly_v1/full_union_tabular/v1/full_union_tabular.joblib",
    "runs/assembly_v1/kaggle/positional_diffusion_pilot_output/v2/positional_diffusion_pilot/positional_diffusion_latest.pt.previous",
    "runs/assembly_v1/kaggle/latent_edge_embedding_output/v3_complete/latent_edge_pilot/latent_edge_embedding.pt",
)
FINAL_REPORT_FORBIDDEN = (
    "Статус: черновик",
    "будет дополнен",
    "будут добавлены",
    "[PENDING]",
    "TODO",
)
SUBMISSION_NAME = re.compile(r"img_[0-9]{6}\.png", flags=re.ASCII)


@dataclass(frozen=True)
class Entry:
    source: Path
    archive_path: str
    role: str


def sha256_path(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    if size != before.st_size:
        raise RuntimeError(f"short read while hashing: {path}")
    return size, digest.hexdigest()


def safe_member(value: str) -> str:
    if "\\" in value or "\0" in value:
        raise ValueError(f"unsafe archive path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or value != path.as_posix()
        or ":" in path.parts[0]
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe archive path: {value!r}")
    lowered = path.as_posix().lower()
    if any(token in lowered for token in DENY_MEMBER_TOKENS):
        raise ValueError(f"forbidden archive path: {value!r}")
    return path.as_posix()


def excluded(path: Path, *, repo: Path, generated_outputs: set[Path]) -> bool:
    resolved = path.resolve()
    if resolved in generated_outputs:
        return True
    try:
        relative = resolved.relative_to(repo).as_posix()
    except ValueError as error:
        raise RuntimeError("source escapes repository") from error
    if any(part in EXCLUDED_PARTS for part in Path(relative).parts):
        return True
    marker = "/" + relative.lower().strip("/") + "/"
    return any(fragment in marker for fragment in EXCLUDED_PATH_FRAGMENTS)


def add(entries: dict[str, Entry], *, source: Path, archive_path: str, role: str) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"required regular file missing: {source}")
    member = safe_member(archive_path)
    prior = entries.get(member)
    item = Entry(source.resolve(), member, role)
    if prior is not None and prior.source != item.source:
        raise RuntimeError(f"duplicate archive member: {member}")
    entries[member] = item


def collect(repo: Path, generated_outputs: set[Path], goal_objective: Path) -> tuple[dict[str, Entry], list[dict]]:
    entries: dict[str, Entry] = {}
    references: list[dict] = []

    for relative in ROOT_DOCS:
        source = repo / relative
        add(entries, source=source, archive_path=f"project/{relative}", role="project documentation")
    add(entries, source=goal_objective, archive_path="project/goal-objective.md", role="original user objective")

    for root_name in SOURCE_ROOTS:
        root = repo / root_name
        if not root.is_dir():
            raise FileNotFoundError(root)
        for source in sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink()):
            if excluded(source, repo=repo, generated_outputs=generated_outputs):
                continue
            if source.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            relative = source.relative_to(repo).as_posix()
            add(entries, source=source, archive_path=f"source/{relative}", role="canonical source/config/test")

    # Kaggle wrappers under runs are executable source, not merely historical
    # reports. Keep the direct job definitions while excluding nested mounted
    # bundles/readbacks that duplicate the canonical repository snapshot.
    job_roots = (
        repo / "runs/assembly_v1/kaggle",
        repo / "runs/denoise_v2",
    )
    for job_root in job_roots:
        for job_dir in sorted(
            path for path in job_root.glob("*_job") if path.is_dir() and not path.is_symlink()
        ):
            for source in sorted(
                path for path in job_dir.iterdir() if path.is_file() and not path.is_symlink()
            ):
                if excluded(source, repo=repo, generated_outputs=generated_outputs):
                    continue
                if source.suffix.lower() not in JOB_DEFINITION_SUFFIXES:
                    continue
                relative = source.relative_to(repo).as_posix()
                add(
                    entries,
                    source=source,
                    archive_path=f"source/job_definitions/{relative}",
                    role="Kaggle job definition",
                )

    for root_name in HISTORY_ROOTS:
        root = repo / root_name
        if not root.is_dir():
            raise FileNotFoundError(root)
        for source in sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink()):
            if excluded(source, repo=repo, generated_outputs=generated_outputs):
                continue
            relative = source.relative_to(repo).as_posix()
            suffix = source.suffix.lower()
            if suffix in HISTORY_SUFFIXES:
                if source.stat().st_size <= MAX_COMPACT_BYTES:
                    add(entries, source=source, archive_path=f"history/{relative}", role="decision-bearing history")

    best = repo / BEST_SUBMISSION
    if sha256_path(best)[1] != BEST_SUBMISSION_SHA256:
        raise RuntimeError("best submission SHA-256 drift")
    add(entries, source=best, archive_path="submission/submission.zip", role="best user-scored submission")
    for relative in PROMOTED_ASSETS:
        source = repo / relative
        add(entries, source=source, archive_path=f"promoted_assets/{source.name}", role="promoted runtime asset")
    for relative, archive_path, role in EXPLICIT_COMPACT:
        source = repo / relative
        add(entries, source=source, archive_path=archive_path, role=role)

    # Hash-reference only explicitly selected, decision-relevant large artifacts.
    # A broad runs/** scan used to read more than 10 GB of previews and duplicate
    # archives; that is both unnecessary and too close to inspecting raw-like data.
    included_sources = {entry.source for entry in entries.values()}
    referenced_paths = {item["origin"] for item in references}
    for relative in EXPLICIT_HASH_ONLY:
        source = repo / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"explicit hash-only artifact missing: {source}")
        if excluded(source, repo=repo, generated_outputs=generated_outputs) or source.resolve() in included_sources:
            continue
        if relative in referenced_paths:
            continue
        size, digest = sha256_path(source)
        references.append({"origin": relative, "bytes": size, "sha256": digest, "reason": "selected large, duplicate, or non-promoted artifact"})
        referenced_paths.add(relative)
    return entries, sorted(references, key=lambda item: item["origin"])


def zip_info(member: str, *, stored: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(safe_member(member), date_time=ARCHIVE_TIMESTAMP)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    return info


def write_file(archive: zipfile.ZipFile, entry: Entry, *, repo: Path) -> dict:
    stored = entry.source.suffix.lower() in {".zip", ".pt", ".pth", ".ckpt", ".safetensors"}
    before = entry.source.stat()
    digest = hashlib.sha256()
    size = 0
    info = zip_info(entry.archive_path, stored=stored)
    info.file_size = before.st_size
    with entry.source.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            target.write(block)
    after = entry.source.stat()
    if size != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed during packaging: {entry.source}")
    try:
        logical_origin = entry.source.relative_to(repo).as_posix()
    except ValueError:
        logical_origin = "external/goal-objective.md"
    return {
        "archive_path": entry.archive_path,
        "origin": logical_origin,
        "role": entry.role,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "zip_method": "stored" if stored else "deflated",
    }


def validate_submission(path: Path) -> dict:
    size, digest = sha256_path(path)
    if digest != BEST_SUBMISSION_SHA256:
        raise RuntimeError("submission hash mismatch")
    names: list[str] = []
    pixel_digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("nested submission CRC failure")
        infos = sorted(archive.infolist(), key=lambda item: item.filename)
        if len(infos) != 700:
            raise RuntimeError(f"expected 700 submission members, got {len(infos)}")
        for info in infos:
            if info.is_dir() or "/" in info.filename or not SUBMISSION_NAME.fullmatch(info.filename):
                raise RuntimeError(f"invalid submission member: {info.filename!r}")
            payload = archive.read(info)
            with Image.open(BytesIO(payload)) as image:
                image.load()
                if image.mode != "RGB" or image.size != (480, 480) or image.format != "PNG":
                    raise RuntimeError(f"invalid submission image: {info.filename}")
                pixel_digest.update(info.filename.encode("ascii") + b"\0" + image.tobytes())
            names.append(info.filename)
    if len(set(names)) != 700:
        raise RuntimeError("duplicate submission names")
    return {
        "bytes": size,
        "sha256": digest,
        "members": 700,
        "all_root_png_rgb_480x480": True,
        "decoded_pixel_stream_sha256": pixel_digest.hexdigest(),
        "leaderboard_score": 0.218,
        "leaderboard_score_precision": "rounded_to_three_decimals_as_reported_by_user",
    }


def verify_outer_archive(path: Path) -> dict:
    """Re-read and hash every embedded member against the generated manifest."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("outer archive has duplicate members")
        if archive.testzip() is not None:
            raise RuntimeError("outer archive CRC failure")
        for info in infos:
            safe_member(info.filename)
            file_type = (info.external_attr >> 16) & 0o170000
            if info.is_dir() or file_type not in {0, 0o100000}:
                raise RuntimeError(f"non-regular archive member: {info.filename!r}")

        required = {"README.md", "MANIFEST.json", "SHA256SUMS.txt", "submission/submission.zip"}
        missing = required.difference(names)
        if missing:
            raise RuntimeError(f"outer archive missing required members: {sorted(missing)}")
        manifest_bytes = archive.read("MANIFEST.json")
        manifest = json.loads(manifest_bytes)
        records = manifest.get("entries")
        if not isinstance(records, list):
            raise RuntimeError("manifest entries are missing")
        record_by_name = {record.get("archive_path"): record for record in records}
        if len(record_by_name) != len(records) or None in record_by_name:
            raise RuntimeError("manifest has duplicate or invalid entry names")
        if set(record_by_name) != set(names).difference({"MANIFEST.json", "SHA256SUMS.txt"}):
            raise RuntimeError("manifest/member set mismatch")
        submission_record = record_by_name.get("submission/submission.zip", {})
        if submission_record.get("sha256") != BEST_SUBMISSION_SHA256:
            raise RuntimeError("embedded submission SHA-256 drift")
        if manifest.get("submission_validation", {}).get("sha256") != BEST_SUBMISSION_SHA256:
            raise RuntimeError("manifest submission validation drift")

        verified_bytes = 0
        for name, record in record_by_name.items():
            digest = hashlib.sha256()
            size = 0
            with archive.open(name) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            if size != record.get("bytes") or digest.hexdigest() != record.get("sha256"):
                raise RuntimeError(f"manifest hash mismatch: {name}")
            verified_bytes += size

        sums: dict[str, str] = {}
        for line in archive.read("SHA256SUMS.txt").decode("utf-8").splitlines():
            digest, separator, name = line.partition("  ")
            if not separator or name in sums or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise RuntimeError("invalid SHA256SUMS line")
            sums[name] = digest
        expected_sums = {name: record["sha256"] for name, record in record_by_name.items()}
        expected_sums["MANIFEST.json"] = hashlib.sha256(manifest_bytes).hexdigest()
        if sums != expected_sums:
            raise RuntimeError("SHA256SUMS content mismatch")
    return {
        "archive_members": len(names),
        "embedded_entries": len(records),
        "verified_uncompressed_bytes": verified_bytes,
        "outer_crc_ok": True,
        "safe_member_paths": True,
        "regular_members_only": True,
        "manifest_hashes_ok": True,
        "sha256sums_ok": True,
    }


def package(args: argparse.Namespace) -> dict:
    repo = args.repo.resolve()
    output = args.output.resolve()
    output_dir = output.parent
    report_path = output.with_suffix(".verification.json")
    try:
        output.relative_to(repo)
    except ValueError as error:
        raise ValueError("--output must stay inside the repository") from error
    if "puzzle" in output.relative_to(repo).parts:
        raise ValueError("--output cannot be inside protected puzzle data")
    protected_inputs = {
        (repo / relative).resolve()
        for relative in (
            *ROOT_DOCS,
            BEST_SUBMISSION,
            *PROMOTED_ASSETS,
            *(relative for relative, _archive_path, _role in EXPLICIT_COMPACT),
            *EXPLICIT_HASH_ONLY,
        )
    }
    if output in protected_inputs or report_path in protected_inputs:
        raise ValueError("--output or verification path collides with a protected input artifact")
    generated_outputs = {output, report_path}
    if output.suffix.lower() != ".zip":
        raise ValueError("--output must end in .zip")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"verification report exists; pass --overwrite: {report_path}")
    narrative = (repo / "COMPLETE_ML_TASK_HISTORY.md").read_text(encoding="utf-8")
    blockers = [token for token in FINAL_REPORT_FORBIDDEN if token in narrative]
    if blockers:
        raise RuntimeError(f"final narrative still contains pending markers: {blockers}")
    submission_validation = validate_submission(repo / BEST_SUBMISSION)
    entries, references = collect(repo, generated_outputs, args.goal_objective.resolve())
    required_members = {
        "project/COMPLETE_ML_TASK_HISTORY.md",
        "source/src/puzzle_assembly/masked_gap.py",
        "source/job_definitions/runs/assembly_v1/kaggle/masked_gap_stage1_job/run_stage1_train_prepare.py",
        "source/job_definitions/runs/assembly_v1/kaggle/masked_gap_phasea_job/run_phasea_authorize.py",
        "source/job_definitions/runs/assembly_v1/kaggle/masked_gap_phaseb_job/run_phaseb_isolated.py",
        "history/runs/assembly_v1/kaggle/luma_harmonized_submission_output/v1/leaderboard_observation.json",
        "promoted_assets/selected_tilenaf_synth_50k.pt",
        "promoted_assets/hbt_d320_denoised_rgb_sobel.pt",
        "promoted_assets/seam_denoiser_gpu.pt",
        "masked_gap/masked_gap_gate_code.zip",
        "masked_gap/masked_gap_gate.pt",
    }
    missing_members = required_members.difference(entries)
    if missing_members:
        raise RuntimeError(f"required delivery members not collected: {sorted(missing_members)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    readme = (
        "# Complete ML task delivery\n\n"
        "Start with `project/COMPLETE_ML_TASK_HISTORY.md`. The best scored archive is "
        "`submission/submission.zip`. `source/` is the canonical code snapshot, `history/` "
        "contains compact decision evidence, and `promoted_assets/` contains runtime weights.\n\n"
        "The older `history/.../RESULT_SUMMARY.md` may say the luma submission was unscored; "
        "the newer leaderboard observation and final narrative supersede it with the user-reported "
        "rounded leaderboard value 0.218. Large duplicate/non-promoted artifacts are hash-only in "
        "`MANIFEST.json`. Raw puzzle data and credentials are excluded.\n"
    ).encode("utf-8")
    records: list[dict] = []
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            archive.writestr(zip_info("README.md", stored=False), readme, compresslevel=9)
            records.append({"archive_path": "README.md", "origin": "generated", "role": "bundle navigation", "bytes": len(readme), "sha256": hashlib.sha256(readme).hexdigest(), "zip_method": "deflated"})
            for member in sorted(entries):
                records.append(write_file(archive, entries[member], repo=repo))
            manifest = {
                "schema": "complete_ml_task_history_bundle_v1",
                "created_local_date": "2026-07-13",
                "submission_validation": submission_validation,
                "entries": records,
                "hash_only_artifacts": references,
                "exclusion_policy": [
                    "raw puzzle train/test and sample submission",
                    "credentials and Kaggle token files",
                    "local environments, VCS/IDE metadata, caches and bytecode",
                    "external model/source caches",
                    "duplicate/non-promoted checkpoints and candidate tensors are hash-only",
                ],
                "determinism": {"timestamp": "2026-07-13T00:00:00", "member_order": "README, sorted inputs, MANIFEST, SHA256SUMS", "compressed_assets": "stored", "text": "deflate-9"},
            }
            manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            archive.writestr(zip_info("MANIFEST.json", stored=False), manifest_bytes, compresslevel=9)
            sums = {record["archive_path"]: record["sha256"] for record in records}
            sums["MANIFEST.json"] = hashlib.sha256(manifest_bytes).hexdigest()
            sums_bytes = "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())).encode("utf-8")
            archive.writestr(zip_info("SHA256SUMS.txt", stored=False), sums_bytes, compresslevel=9)
        archive_validation = verify_outer_archive(temporary)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    output_bytes, output_sha = sha256_path(output)
    report = {
        "kind": "complete_ml_task_delivery_verification_v1",
        "archive": str(output),
        "archive_bytes": output_bytes,
        "archive_sha256": output_sha,
        **archive_validation,
        "hash_only_artifacts": len(references),
        "submission_validation": submission_validation,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--goal-objective", type=Path, default=Path("/Users/rusyalain/.codex/attachments/4efb14c2-993b-457a-a126-8d3125200cee/goal-objective.md"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = package(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
