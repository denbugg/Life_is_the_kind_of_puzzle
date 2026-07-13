from __future__ import annotations

from io import BytesIO
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "runs/assembly_v1/kaggle/harmonized_submission_job/run_harmonized_submission.py"
)
BUNDLE_ARCHIVE = (
    REPO_ROOT
    / "runs/assembly_v1/kaggle/harmonized_submission_runtime_bundle_v1/"
    "harmonized_submission_runtime_v1.zip"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("harmonized_kaggle_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png(value: int) -> bytes:
    array = np.full((480, 480, 3), value, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def _archive(path: Path, names: list[str], timestamp: tuple[int, ...]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for index, name in enumerate(names):
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _png(30 + index), compresslevel=6)


def test_real_runtime_bundle_validates(tmp_path: Path) -> None:
    runner = _load_runner()
    with zipfile.ZipFile(BUNDLE_ARCHIVE) as archive:
        archive.extractall(tmp_path)
    manifest, members = runner.validate_bundle(tmp_path)
    assert manifest["member_count"] == 50
    assert len(members) == 50
    assert {
        record["sha256"]
        for record in members
        if record["path"].startswith("layouts/")
    } == runner.EXPECTED_LAYOUT_REPORT_SHA256


def test_runtime_bundle_rejects_extra_file(tmp_path: Path) -> None:
    runner = _load_runner()
    with zipfile.ZipFile(BUNDLE_ARCHIVE) as archive:
        archive.extractall(tmp_path)
    (tmp_path / "unexpected.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file tree drift"):
        runner.validate_bundle(tmp_path)


def test_merge_shards_is_deterministic_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "WORKING", tmp_path)
    monkeypatch.setattr(runner, "EXPECTED_TOTAL", 2)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _archive(first, ["img_a.png"], runner.ARCHIVE_TIMESTAMP)
    _archive(second, ["img_b.png"], runner.ARCHIVE_TIMESTAMP)
    destination, manifest = runner.merge_shards(
        test_names=["img_a.png", "img_b.png"],
        first={"output": str(first)},
        second={"output": str(second)},
    )
    assert destination == tmp_path / "submission.zip"
    assert manifest["member_count"] == 2
    assert manifest["archive"]["sha256"] == runner.sha256(destination)
    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist() == ["img_a.png", "img_b.png"]
        assert all(info.date_time == runner.ARCHIVE_TIMESTAMP for info in archive.infolist())
    persisted = json.loads(
        (tmp_path / "HARMONIZED_SUBMISSION_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert persisted == manifest


def test_merge_rejects_duplicate_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "WORKING", tmp_path)
    monkeypatch.setattr(runner, "EXPECTED_TOTAL", 2)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _archive(first, ["img_a.png"], runner.ARCHIVE_TIMESTAMP)
    _archive(second, ["img_a.png"], runner.ARCHIVE_TIMESTAMP)
    with pytest.raises(RuntimeError, match="duplicate shard member"):
        runner.merge_shards(
            test_names=["img_a.png", "img_b.png"],
            first={"output": str(first)},
            second={"output": str(second)},
        )

