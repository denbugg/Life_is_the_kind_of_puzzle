from __future__ import annotations

import json
from pathlib import Path

from aiijc_puzzle.protocol import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "configs/default_six_emitter_joint_real_unsigned_template_v1.json"


def test_default_six_real_template_is_unsigned_blocked_and_hash_consistent() -> None:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert payload["status"] == "unsigned-template-blocked-no-training-binding-or-fresh-dev"
    assert payload["real_protocol_signed"] is False
    assert not Path(f"{TEMPLATE}.sha256").exists()
    assert payload["fixed_model"]["emitter_order"] == [
        "raw",
        "adapter1600",
        "dinov2",
        "guided",
        "wiener",
        "haar_bayesshrink",
    ]
    assert payload["fixed_model"]["wavelet_sidecar_source_indices"] == [0, 1, 2, 3, 4, 6]
    assert payload["fixed_model"]["local_rank"]["enabled"] is False
    assert payload["fixed_model"]["supply_feature_dim"] == 12
    assert payload["fixed_model"]["parameter_counts"] == {
        "total": 41717,
        "trainable": 414,
    }
    assert payload["fixed_model"]["legacy_head_scope"] == "legacy_slot_present_only"
    assert payload["fixed_model"]["novel_tri_auxiliary"] == "forbidden_not_zero_imputed"
    assert payload["tri_v2_dev_prerequisite"]["gate_passed"] is True
    assert payload["tri_v2_dev_prerequisite"]["score_sha256"] == (
        "9548487b73481d5ec01963911a75c62d117ae634d7105df708edad1802be5274"
    )
    assert payload["proposed_training"]["authorised"] is False
    assert payload["proposed_evaluation"]["fresh_dev_available"] is False
    assert payload["legality"]["labels_opened_by_scaffolding"] is False
    assert payload["legality"]["weco_used"] is False

    for section in ("implementation_artifacts", "immutable_input_artifacts"):
        for artifact in payload[section].values():
            path = PROJECT_ROOT / artifact["path"]
            assert path.is_file()
            assert sha256_file(path) == artifact["sha256"]
