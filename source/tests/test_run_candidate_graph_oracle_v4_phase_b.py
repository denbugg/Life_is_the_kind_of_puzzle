from __future__ import annotations

import json
from pathlib import Path

from scripts import run_candidate_graph_oracle_v4_phase_b as phase_b


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/candidate_graph_oracle_ceiling_v4.json"


def test_phase_b_is_bound_to_v4_config_and_snapshot_closure() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert phase_b.EXPECTED_CONFIG == CONFIG
    known = config["frozen_contract"]["assets"]["known_code_sha256"]
    assert set(known) == set(phase_b.KNOWN_CODE_ALLOWLIST)
    assert "scripts/build_candidate_graph_oracle_v4_fixtures.py" in (
        phase_b.PINNED_REPO_READ_FILES
    )
    assert "scripts/build_candidate_graph_oracle_fixtures.py" not in (
        phase_b.PINNED_REPO_READ_FILES
    )
    assert "scripts/evaluate_candidate_graph_oracle_v4.py" in (
        phase_b.PINNED_REPO_READ_FILES
    )
    assert "scripts/verify_candidate_graph_oracle_v4_result.py" in (
        phase_b.PINNED_REPO_READ_FILES
    )


def test_phase_b_pre_reservation_config_is_not_runnable() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pins = config["runtime_pins"]
    assert any(value is None for key, value in pins.items() if key.endswith("_sha256"))
    assert not (
        REPO
        / config["runtime_pin_mutation_policy"]["transition_ledger_root"]
        / "PREP.json"
    ).exists()


def test_phase_b_static_allowlist_has_no_v3_runtime_identity() -> None:
    joined = "\n".join(phase_b.PINNED_REPO_READ_FILES)
    assert "candidate_graph_oracle_v3" not in joined
    assert "candidate_graph_oracle_ceiling_v3" not in joined
