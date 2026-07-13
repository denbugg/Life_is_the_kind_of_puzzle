from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_candidate_graph_oracle_v4_pre_reservation_manifest as closure


REPO = Path(__file__).resolve().parents[1]


def _load_envelope(path: Path) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == closure._canonical_file_bytes(value)
    assert set(value) == {"payload", "payload_sha256"}
    assert value["payload_sha256"] == hashlib.sha256(
        closure._canonical_object_bytes(value["payload"])
    ).hexdigest()
    return value


def test_live_v4_pre_reservation_source_closure_is_complete_and_unpinned() -> None:
    result = closure.validate_source_closure(REPO)
    assert result["protocol_instance_id"] == closure.INSTANCE
    assert result["frozen_contract_sha256"] == closure.FROZEN_SHA256
    assert result["runtime_pin_state"] == {
        "code_pin_pair_count": 12,
        "code_sha256_values": "all_null",
        "fixture_pin_pair_count": 3,
        "fixture_sha256_values": "all_null",
    }
    assert result["reservation_binding"]["kernel_id"] == -1
    assert result["reservation_binding"]["reservation_receipt_sha256"] is None
    assert result["reservation_binding"]["pinnable"] is False
    assert result["remote_api_called"] is False
    assert result["label_paths_constructed"] is False
    assert result["code_pin_performed"] is False
    assert result["prep_claimed"] is False
    assert len(result["historical_v3_byte_stability"]) == 10


def test_source_manifest_covers_snapshot_reservation_and_local_utilities() -> None:
    result = closure.validate_source_closure(REPO)
    paths = {record["path"] for record in result["source_files"]}
    assert "scripts/reserve_candidate_graph_oracle_v4_kaggle.py" in paths
    assert "scripts/audit_candidate_graph_oracle_v4_launch_closure.py" in paths
    assert "scripts/build_candidate_graph_oracle_v4_fixtures.py" in paths
    assert "tests/test_reserve_candidate_graph_oracle_v4_kaggle.py" in paths
    assert (
        closure.RESERVATION_ROOT_RELATIVE + "/kernel/reservation_runner.py"
    ) in paths
    snapshot = [
        path
        for path in paths
        if "/candidate_graph_oracle_v4_source_snapshot/src/" in path
    ]
    assert len(snapshot) == 18


def test_build_evidence_writes_two_exclusive_self_hashed_files(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    mutation = tmp_path / "mutation.json"
    result = closure.build_evidence(
        repo_root=REPO,
        manifest_path=manifest,
        mutation_receipt_path=mutation,
    )
    manifest_envelope = _load_envelope(manifest)
    mutation_envelope = _load_envelope(mutation)
    assert result["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result["mutation_receipt_sha256"] == hashlib.sha256(
        mutation.read_bytes()
    ).hexdigest()
    assert mutation_envelope["payload"]["source_manifest"]["sha256"] == result[
        "manifest_sha256"
    ]
    assert mutation_envelope["payload"]["captured_config_sha256"] == (
        manifest_envelope["payload"]["config"]["sha256"]
    )
    with pytest.raises(FileExistsError):
        closure.build_evidence(
            repo_root=REPO,
            manifest_path=manifest,
            mutation_receipt_path=tmp_path / "second-mutation.json",
        )
