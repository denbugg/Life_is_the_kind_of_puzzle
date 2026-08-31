"""Deterministic, allowlist-only source snapshots for the final submission.

The submission workspace is intentionally not assumed to be committed to Git.
This module preserves every file named by the production runtime manifest plus
the small set of documentation and validation files needed to audit the final
artifact.  Data, generated outputs and the historical research repository are
never discovered recursively and therefore cannot enter the archive by accident.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from aiijc_puzzle.compliant_submission import (
    PROJECT_ROOT,
    RUNTIME_FILE_RELATIVE_PATHS,
    build_runtime_manifest,
)

SNAPSHOT_SCHEMA = "aiijc-puzzle-source-snapshot-v1"
EMBEDDED_MANIFEST_NAME = "SOURCE_SNAPSHOT_MANIFEST.json"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FORBIDDEN_TOP_LEVEL_PATHS = frozenset({"data", "outputs", ".git"})

# This is an explicit allowlist, not a directory walk.  Keep it focused on
# reproduction, compliance and validation of the frozen production artifact.
SUPPLEMENTAL_FILE_RELATIVE_PATHS = (
    ".python-version",
    "README.md",
    "pyproject.toml",
    "docs/AI Challenge.pdf",
    "docs/submission-compliance.md",
    "docs/experiments/frozen-final-evaluation.md",
    "docs/experiments/nlm-strength-manual-safety.md",
    "docs/experiments/postassembly-harmonizer.md",
    "src/aiijc_puzzle/__init__.py",
    "src/aiijc_puzzle/frozen_final_evaluator.py",
    "src/aiijc_puzzle/source_snapshot.py",
    "scripts/run_frozen_final_evaluation.py",
    "scripts/build_source_snapshot.py",
    "tests/test_compliant_submission.py",
    "tests/test_frozen_final_evaluator.py",
    "tests/test_source_snapshot.py",
    "uv.lock",
)

# The final narrative is written late in the workflow.  Include it if present,
# while recording its presence explicitly in the manifest.
OPTIONAL_FILE_RELATIVE_PATHS = ("docs/final-solution.md",)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_path(relative: str) -> PurePosixPath:
    """Return a normalized project-relative path or reject it fail-closed."""

    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe source-snapshot path: {relative!r}")
    if path.parts[0] in FORBIDDEN_TOP_LEVEL_PATHS:
        raise ValueError(f"forbidden source-snapshot path: {relative!r}")
    return path


def _read_regular_source(project_root: Path, relative: str) -> bytes:
    safe = _safe_relative_path(relative)
    root = project_root.resolve(strict=True)
    path = root.joinpath(*safe.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"snapshot source must be a regular non-symlink file: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"snapshot source escapes project root: {relative}")
    return resolved.read_bytes()


def selected_source_paths(project_root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    """Return the sorted, explicit file selection for the current workspace."""

    required = set(RUNTIME_FILE_RELATIVE_PATHS) | set(SUPPLEMENTAL_FILE_RELATIVE_PATHS)
    selected = set(required)
    for relative in OPTIONAL_FILE_RELATIVE_PATHS:
        safe = _safe_relative_path(relative)
        candidate = project_root.joinpath(*safe.parts)
        if candidate.exists():
            selected.add(relative)
    paths = tuple(sorted(selected))
    for relative in paths:
        _read_regular_source(project_root, relative)
    return paths


def _source_payload(
    project_root: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if project_root.resolve(strict=True) != PROJECT_ROOT.resolve(strict=True):
        raise ValueError("source snapshots may only be built from the production project root")
    runtime_manifest = build_runtime_manifest()
    runtime_paths = set(RUNTIME_FILE_RELATIVE_PATHS)
    payloads: dict[str, bytes] = {}
    records: dict[str, dict[str, Any]] = {}
    for relative in selected_source_paths(project_root):
        payload = _read_regular_source(project_root, relative)
        payloads[relative] = payload
        records[relative] = {
            "role": "production_runtime" if relative in runtime_paths else "audit_support",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    if runtime_manifest["files"] != {
        relative: records[relative]["sha256"] for relative in RUNTIME_FILE_RELATIVE_PATHS
    }:
        raise ValueError("snapshot files do not match the production runtime manifest")

    content: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "purpose": (
            "Exact source and audit-support snapshot for the frozen compliant "
            "AIJIC puzzle submission"
        ),
        "selection": {
            "mode": "explicit_allowlist_no_recursive_discovery",
            "optional_files_included": sorted(
                set(payloads).intersection(OPTIONAL_FILE_RELATIVE_PATHS)
            ),
            "forbidden_top_level_paths": sorted(FORBIDDEN_TOP_LEVEL_PATHS),
        },
        "files": records,
        "production_runtime_manifest": runtime_manifest,
    }
    content["source_files_digest_sha256"] = _canonical_json_sha256({"files": records})
    return payloads, content


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIMESTAMP)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_zip_bytes(payloads: Mapping[str, bytes], manifest_bytes: bytes) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(payloads):
            archive.writestr(_zip_info(name), payloads[name], compresslevel=9)
        archive.writestr(
            _zip_info(EMBEDDED_MANIFEST_NAME),
            manifest_bytes,
            compresslevel=9,
        )
    return buffer.getvalue()


def build_source_snapshot(
    *,
    archive_path: Path,
    manifest_path: Path,
    checksum_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build and then independently validate a deterministic source snapshot."""

    payloads, manifest = _source_payload(project_root)
    manifest_bytes = _canonical_json_bytes(manifest)
    archive_bytes = _build_zip_bytes(payloads, manifest_bytes)
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    _write_atomic(archive_path, archive_bytes)
    _write_atomic(manifest_path, manifest_bytes)
    _write_atomic(checksum_path, f"{archive_sha256}  {archive_path.name}\n".encode())

    report = validate_source_snapshot(
        archive_path=archive_path,
        manifest_path=manifest_path,
        checksum_path=checksum_path,
        project_root=project_root,
        compare_with_workspace=True,
    )
    return {"status": "PASS", **report}


def _load_manifest(payload: bytes) -> dict[str, Any]:
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source snapshot manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("source snapshot manifest has an unsupported schema")
    if _canonical_json_bytes(manifest) != payload:
        raise ValueError("source snapshot manifest is not canonical JSON")
    return manifest


def _validate_checksum(checksum_path: Path, archive_path: Path, actual_hash: str) -> None:
    line = checksum_path.read_text(encoding="ascii")
    expected = f"{actual_hash}  {archive_path.name}\n"
    if line != expected:
        raise ValueError("source snapshot checksum file does not match the archive")


def validate_source_snapshot(
    *,
    archive_path: Path,
    manifest_path: Path,
    checksum_path: Path,
    project_root: Path = PROJECT_ROOT,
    compare_with_workspace: bool = True,
) -> dict[str, Any]:
    """Verify archive structure, hashes, metadata and optionally current sources."""

    archive_bytes = archive_path.read_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    _validate_checksum(checksum_path, archive_path, archive_sha256)
    sidecar_bytes = manifest_path.read_bytes()
    manifest = _load_manifest(sidecar_bytes)

    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("source snapshot contains duplicate ZIP entries")
            expected_names = sorted(manifest["files"]) + [EMBEDDED_MANIFEST_NAME]
            if names != expected_names:
                raise ValueError("source snapshot entries are missing, extra or out of order")
            if archive.testzip() is not None:
                raise ValueError("source snapshot failed ZIP CRC validation")
            embedded = archive.read(EMBEDDED_MANIFEST_NAME)
            if embedded != sidecar_bytes:
                raise ValueError("embedded and sidecar source manifests differ")
            for info in infos:
                if (
                    info.date_time != FIXED_ZIP_TIMESTAMP
                    or info.create_system != 3
                    or info.compress_type != zipfile.ZIP_DEFLATED
                    or stat.S_IMODE(info.external_attr >> 16) != 0o644
                    or not stat.S_ISREG(info.external_attr >> 16)
                ):
                    raise ValueError(f"non-deterministic ZIP metadata for {info.filename}")
            for relative, record in manifest["files"].items():
                _safe_relative_path(relative)
                payload = archive.read(relative)
                if len(payload) != record["size_bytes"]:
                    raise ValueError(f"source snapshot size mismatch: {relative}")
                if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                    raise ValueError(f"source snapshot hash mismatch: {relative}")
                if compare_with_workspace:
                    current = _read_regular_source(project_root, relative)
                    if current != payload:
                        raise ValueError(f"workspace differs from source snapshot: {relative}")
    except zipfile.BadZipFile as exc:
        raise ValueError("source snapshot is not a valid ZIP archive") from exc

    records = manifest["files"]
    required_paths = set(RUNTIME_FILE_RELATIVE_PATHS) | set(SUPPLEMENTAL_FILE_RELATIVE_PATHS)
    allowed_paths = required_paths | set(OPTIONAL_FILE_RELATIVE_PATHS)
    record_paths = set(records)
    if not required_paths <= record_paths or not record_paths <= allowed_paths:
        raise ValueError("source snapshot manifest differs from the explicit allowlist")
    if manifest.get("source_files_digest_sha256") != _canonical_json_sha256({"files": records}):
        raise ValueError("source snapshot aggregate file digest differs")
    runtime_manifest = manifest.get("production_runtime_manifest", {})
    runtime_files = runtime_manifest.get("files")
    expected_runtime_files = {
        relative: records[relative]["sha256"] for relative in RUNTIME_FILE_RELATIVE_PATHS
    }
    if runtime_files != expected_runtime_files:
        raise ValueError("embedded production runtime file manifest differs")
    runtime_content = {
        "files": runtime_manifest.get("files"),
        "versions": runtime_manifest.get("versions"),
    }
    if runtime_manifest.get("digest_sha256") != _canonical_json_sha256(runtime_content):
        raise ValueError("embedded production runtime digest differs")
    forbidden = [
        relative
        for relative in records
        if PurePosixPath(relative).parts[0] in FORBIDDEN_TOP_LEVEL_PATHS
    ]
    if forbidden:
        raise ValueError(f"forbidden paths entered source snapshot: {forbidden}")

    return {
        "archive": str(archive_path.resolve()),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": len(archive_bytes),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "checksum": str(checksum_path.resolve()),
        "file_count": len(records),
        "source_files_digest_sha256": manifest["source_files_digest_sha256"],
        "runtime_manifest_sha256": manifest["production_runtime_manifest"]["digest_sha256"],
        "workspace_match": compare_with_workspace,
    }


def reproducibility_check(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Build the same snapshot twice in a temporary directory and compare bytes."""

    with tempfile.TemporaryDirectory(prefix="aiijc-source-snapshot-check-") as temporary:
        root = Path(temporary)
        archive_hashes: list[str] = []
        manifest_hashes: list[str] = []
        for index in (1, 2):
            archive = root / f"source-snapshot-{index}.zip"
            manifest = root / f"source-snapshot-{index}.json"
            checksum = root / f"source-snapshot-{index}.sha256"
            build_source_snapshot(
                archive_path=archive,
                manifest_path=manifest,
                checksum_path=checksum,
                project_root=project_root,
            )
            archive_hashes.append(hashlib.sha256(archive.read_bytes()).hexdigest())
            manifest_hashes.append(hashlib.sha256(manifest.read_bytes()).hexdigest())
        if len(set(archive_hashes)) != 1 or len(set(manifest_hashes)) != 1:
            raise ValueError("two source snapshot builds were not byte-for-byte reproducible")
        return {
            "status": "PASS",
            "archive_sha256": archive_hashes[0],
            "manifest_sha256": manifest_hashes[0],
        }
