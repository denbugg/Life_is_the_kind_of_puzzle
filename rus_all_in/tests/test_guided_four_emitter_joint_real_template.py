from __future__ import annotations

import json
from pathlib import Path

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import names_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "configs/guided_four_emitter_joint_real_unsigned_template_v1.json"


def test_real_template_is_unsigned_inventory_blocked_and_hash_consistent() -> None:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert payload["status"] == "unsigned-template-blocked-no-source-disjoint-dev32"
    assert payload["real_protocol_signed"] is False
    assert not Path(f"{TEMPLATE}.sha256").exists()
    assert payload["source_protocol"]["dev_filenames"] == []
    assert payload["source_protocol"]["dev_source_count"] == 0
    assert payload["inventory_blocker"]["excluded_train_count"] == 5600
    assert payload["inventory_blocker"]["eligible_train_count"] == 0
    assert payload["proposed_training"]["authorised"] is False
    assert payload["proposed_training"]["legacy_endpoint"] is None
    assert payload["source_protocol"]["fit_digest"] == names_digest(
        payload["source_protocol"]["fit_filenames"]
    )
    assert payload["objective"]["threshold_or_fraction_sweep"] is False
    assert payload["fixed_model"]["candidate_width"] == 128
    assert payload["fixed_model"]["guided_auxiliary_dim"] == 7
    for section in (
        "implementation_artifacts",
        "observed_dependencies_requiring_hash_review_before_any_signature",
    ):
        for artifact in payload[section].values():
            path = PROJECT_ROOT / artifact["path"]
            assert path.is_file()
            assert sha256_file(path) == artifact["sha256"]
