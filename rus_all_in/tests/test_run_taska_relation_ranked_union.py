from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_taska_relation_ranked_union as runner  # noqa: E402


def test_signed_no_sweep_contract_and_artifacts_validate() -> None:
    contract, digest = runner._load_config(runner.DEFAULT_CONFIG)
    assert digest == "2f0beb7fc071f4aef673267fab348baaeccdb664048e2ac2154edf45fa2723a7"
    assert contract["candidate"]["threshold"] is None
    assert contract["candidate"]["top_k"] is None
    assert contract["no_threshold_topk_weight_parameter_or_model_sweep"] is True
    assert contract["formal_if_all_development_gates_pass"][
        "must_sign_before_generation_or_scoring"
    ] is True


def test_runner_cli_exposes_no_candidate_tuning() -> None:
    args = runner.parse_args([])
    assert args.config == runner.DEFAULT_CONFIG
    for option in ("--threshold", "--top-k", "--weight", "--arm", "--model"):
        with pytest.raises(SystemExit):
            runner.parse_args([option, "1"])


def test_one_case_target_free_replay_is_strict_and_all_edge(tmp_path: Path) -> None:
    output = tmp_path / "target-free-smoke"
    report = runner.run(
        runner.parse_args(
            ["--output-dir", str(output), "--target-free-smoke"]
        )
    )
    assert report["status"] == "target-free-smoke"
    with np.load(
        output / "local32/frozen-target-free-eval.npz", allow_pickle=False
    ) as archive:
        control = archive["case_0000__relation_truth_selector_layout"]
        candidate = archive["case_0000__relation_ranked_all_edge_union_layout"]
        probability = archive["case_0000__union_probability"]
    assert np.array_equal(np.sort(control), np.arange(576))
    assert np.array_equal(np.sort(candidate), np.arange(576))
    assert len(probability) > 1_104
    assert np.all(probability[:-1] >= probability[1:])


def test_fixed_local_report_stops_before_held_fresh_or_formal() -> None:
    root = PROJECT_ROOT / "outputs/taska-relation-ranked-union/fixed-v1"
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "stopped_after_local32_gate_failure"
    assert report["panels"]["local32"]["gate_passed"] is False
    assert report["panels"]["local32"]["candidate_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"] == -127.25
    assert report["formal_confirmation"]["eligible"] is False
    assert report["formal_confirmation"]["new_roster_generated_or_scored"] is False
    assert not (root / "held32").exists()
    assert not (root / "fresh32").exists()
