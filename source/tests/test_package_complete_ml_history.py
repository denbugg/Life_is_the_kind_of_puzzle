from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/package_complete_ml_history.py"
SPEC = importlib.util.spec_from_file_location("package_complete_ml_history", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bundle
SPEC.loader.exec_module(bundle)


def test_masked_gap_reproducibility_payload_is_explicitly_packaged() -> None:
    mapped = {archive_path for _source, archive_path, _role in bundle.EXPLICIT_COMPACT}
    assert "masked_gap/masked_gap_gate_code.zip" in mapped
    assert "masked_gap/masked_gap_gate.pt" in mapped


@pytest.mark.parametrize(
    "name",
    [
        "/absolute.txt",
        "../escape.txt",
        "safe/../../escape.txt",
        "..\\escape.txt",
        "C:/escape.txt",
        "safe/name\0.txt",
        "source/.conda/state.json",
        "source/.kaggle/access_token",
        "history/puzzle/train/img.png",
        "history/dino_model_cache/model.bin",
    ],
)
def test_safe_member_rejects_unsafe_or_denied_names(name: str) -> None:
    with pytest.raises(ValueError):
        bundle.safe_member(name)


def test_safe_member_normalizes_valid_posix_name() -> None:
    assert bundle.safe_member("source/scripts/run.py") == "source/scripts/run.py"


def test_verify_outer_archive_hashes_every_manifest_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle.zip"
    payload = b"history\n"
    readme = b"readme\n"
    submission = b"nested-placeholder"
    submission_sha = hashlib.sha256(submission).hexdigest()
    monkeypatch.setattr(bundle, "BEST_SUBMISSION_SHA256", submission_sha)
    records = [
        {
            "archive_path": "README.md",
            "origin": "generated",
            "role": "navigation",
            "bytes": len(readme),
            "sha256": hashlib.sha256(readme).hexdigest(),
            "zip_method": "deflated",
        },
        {
            "archive_path": "project/history.md",
            "origin": "history.md",
            "role": "history",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "zip_method": "deflated",
        },
        {
            "archive_path": "submission/submission.zip",
            "origin": "submission.zip",
            "role": "submission",
            "bytes": len(submission),
            "sha256": submission_sha,
            "zip_method": "stored",
        },
    ]
    manifest = {"entries": records, "submission_validation": {"sha256": submission_sha}}
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    sums = {record["archive_path"]: record["sha256"] for record in records}
    sums["MANIFEST.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    sums_bytes = "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())).encode()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(bundle.zip_info("README.md", stored=False), readme)
        archive.writestr(bundle.zip_info("project/history.md", stored=False), payload)
        archive.writestr(bundle.zip_info("submission/submission.zip", stored=True), submission)
        archive.writestr(bundle.zip_info("MANIFEST.json", stored=False), manifest_bytes)
        archive.writestr(bundle.zip_info("SHA256SUMS.txt", stored=False), sums_bytes)

    report = bundle.verify_outer_archive(output)
    assert report["manifest_hashes_ok"] is True
    assert report["sha256sums_ok"] is True
    assert report["embedded_entries"] == 3


def test_verify_outer_archive_rejects_manifest_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle.zip"
    readme = b"readme\n"
    submission = b"nested-placeholder"
    submission_sha = hashlib.sha256(submission).hexdigest()
    monkeypatch.setattr(bundle, "BEST_SUBMISSION_SHA256", submission_sha)
    records = [
        {
            "archive_path": "README.md",
            "bytes": len(readme),
            "sha256": "0" * 64,
        },
        {
            "archive_path": "submission/submission.zip",
            "bytes": len(submission),
            "sha256": submission_sha,
        },
    ]
    manifest_bytes = (
        json.dumps(
            {"entries": records, "submission_validation": {"sha256": submission_sha}},
            sort_keys=True,
        )
        + "\n"
    ).encode()
    sums = {record["archive_path"]: record["sha256"] for record in records}
    sums["MANIFEST.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    sums_bytes = "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())).encode()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(bundle.zip_info("README.md", stored=False), readme)
        archive.writestr(bundle.zip_info("submission/submission.zip", stored=True), submission)
        archive.writestr(bundle.zip_info("MANIFEST.json", stored=False), manifest_bytes)
        archive.writestr(bundle.zip_info("SHA256SUMS.txt", stored=False), sums_bytes)

    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        bundle.verify_outer_archive(output)
