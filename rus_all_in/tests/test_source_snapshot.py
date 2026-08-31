from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from aiijc_puzzle.compliant_submission import PROJECT_ROOT, RUNTIME_FILE_RELATIVE_PATHS
from aiijc_puzzle.source_snapshot import (
    EMBEDDED_MANIFEST_NAME,
    _safe_relative_path,
    build_source_snapshot,
    validate_source_snapshot,
)


def _build(tmp_path: Path, stem: str) -> tuple[Path, Path, Path]:
    archive = tmp_path / f"{stem}.zip"
    manifest = tmp_path / f"{stem}.json"
    checksum = tmp_path / f"{stem}.sha256"
    report = build_source_snapshot(
        archive_path=archive,
        manifest_path=manifest,
        checksum_path=checksum,
    )
    assert report["status"] == "PASS"
    return archive, manifest, checksum


def test_source_snapshot_is_reproducible_and_complete(tmp_path: Path) -> None:
    first = _build(tmp_path, "first")
    second = _build(tmp_path, "second")
    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()

    report = validate_source_snapshot(
        archive_path=first[0],
        manifest_path=first[1],
        checksum_path=first[2],
    )
    manifest = json.loads(first[1].read_text())
    assert report["file_count"] == len(manifest["files"])
    assert set(RUNTIME_FILE_RELATIVE_PATHS) <= set(manifest["files"])
    assert not any(name.startswith(("data/", "outputs/", ".git/")) for name in manifest["files"])
    with zipfile.ZipFile(first[0]) as archive:
        assert archive.namelist() == sorted(manifest["files"]) + [EMBEDDED_MANIFEST_NAME]
        assert archive.read(EMBEDDED_MANIFEST_NAME) == first[1].read_bytes()
        assert all(
            hashlib.sha256(archive.read(name)).hexdigest() == record["sha256"]
            for name, record in manifest["files"].items()
        )


@pytest.mark.parametrize(
    "relative",
    ["/absolute", "../escape", "docs/../escape", "data/raw/test.png", "outputs/a.zip"],
)
def test_source_snapshot_rejects_unsafe_or_forbidden_paths(relative: str) -> None:
    with pytest.raises(ValueError):
        _safe_relative_path(relative)


def test_source_snapshot_detects_workspace_drift(tmp_path: Path) -> None:
    archive, manifest, checksum = _build(tmp_path, "snapshot")
    synthetic_root = tmp_path / "workspace"
    synthetic_root.mkdir()
    with pytest.raises(ValueError, match="regular non-symlink|workspace differs"):
        validate_source_snapshot(
            archive_path=archive,
            manifest_path=manifest,
            checksum_path=checksum,
            project_root=synthetic_root,
            compare_with_workspace=True,
        )


def test_runtime_manifest_files_match_current_workspace(tmp_path: Path) -> None:
    archive, manifest_path, _ = _build(tmp_path, "runtime")
    manifest = json.loads(manifest_path.read_text())
    with zipfile.ZipFile(archive) as snapshot:
        for relative, expected_hash in manifest["production_runtime_manifest"]["files"].items():
            assert relative in RUNTIME_FILE_RELATIVE_PATHS
            assert hashlib.sha256(snapshot.read(relative)).hexdigest() == expected_hash
            workspace_hash = hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
            assert workspace_hash == expected_hash
