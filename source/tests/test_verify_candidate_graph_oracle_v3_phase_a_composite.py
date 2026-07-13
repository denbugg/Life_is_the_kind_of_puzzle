from __future__ import annotations

import copy
from pathlib import Path
import zipfile

import pytest

from scripts import verify_candidate_graph_oracle_v3_phase_a_composite as composite


def _recovered_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v3_recovered_launch_verification",
        "verified_utc": "2026-07-12T16:45:41Z",
        "status": "verified_complete",
        "protocol_instance_id": composite.INSTANCE,
        "kernel": {
            "slug": composite.KERNEL,
            "id": composite.KERNEL_ID,
            "version": 2,
            "source_sha256": composite.RUNNER_SHA256,
            "session_status": "KernelWorkerStatus.COMPLETE",
        },
        "datasets": {
            label: {"status": "ready", "current_version_number": 2}
            for label in composite.DATASETS
        },
        **composite.RECOVERY_HASHES,
        "launch_receipt_sha256": composite.LAUNCH_RECEIPT_SHA256,
        "raw_ref_before": f"/code/{composite.KERNEL}",
        "canonical_ref_after": composite.KERNEL,
        "remote_write_performed_by_recovery": False,
        "kernel_version_advance_from_reservation": 1,
        "label_access_claim_present": False,
        "label_fixture_opened_by_verifier": False,
        "safe_for_submission": False,
    }


def test_recovered_schema_accepts_exact_complete_evidence() -> None:
    composite.validate_recovered_verification(_recovered_payload())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("label_fixture_opened_by_verifier", True),
        ("label_access_claim_present", True),
        ("remote_write_performed_by_recovery", True),
        ("status", "verified_running"),
        ("raw_ref_before", composite.KERNEL),
    ],
)
def test_recovered_schema_rejects_unsafe_or_incomplete_evidence(
    key: str, value: object
) -> None:
    payload = copy.deepcopy(_recovered_payload())
    payload[key] = value
    with pytest.raises(RuntimeError):
        composite.validate_recovered_verification(payload)


def test_recovered_schema_rejects_extra_normal_attestation_claim() -> None:
    payload = _recovered_payload()
    payload["normal_schema_launch_attestation"] = True
    with pytest.raises(RuntimeError, match="schema drift"):
        composite.validate_recovered_verification(payload)


def _synthetic_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "snapshot"
    archive = tmp_path / "snapshot.zip"
    root.mkdir()
    with zipfile.ZipFile(archive, "w") as bundle:
        for index in range(38):
            relative = f"tree/member_{index:02d}.txt"
            data = f"payload-{index}".encode()
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            bundle.writestr(relative, data)
    return root, archive


def test_snapshot_exact_tree_rejects_extra_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, archive = _synthetic_snapshot(tmp_path)
    monkeypatch.setattr(composite, "SNAPSHOT_ARCHIVE_SHA256", composite._sha(archive))
    monkeypatch.setattr(composite, "SUPPLEMENT_SHA256S", {})
    # The production verifier member is required only after exact path closure;
    # use the lower-level member parser to establish the 38-member schema first.
    assert len(composite._snapshot_archive_members(archive)) == 38
    (root / "unexpected.pyc").write_bytes(b"not allowed")
    with pytest.raises(RuntimeError, match="path closure"):
        composite.verify_snapshot_exact(root, archive)


def test_snapshot_archive_rejects_traversal_member(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for index in range(37):
            bundle.writestr(f"safe/{index}.txt", b"safe")
        bundle.writestr("../escape.txt", b"unsafe")
    with pytest.raises(RuntimeError, match="unsafe snapshot archive member"):
        composite._snapshot_archive_members(archive)
