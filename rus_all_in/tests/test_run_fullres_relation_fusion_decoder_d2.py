from __future__ import annotations

import json
from pathlib import Path

from scripts.run_fullres_relation_fusion_decoder_d2 import (
    EXPECTED_CONFIG_SHA256,
    evaluate_d2_gate,
    load_d2_config,
    selected_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/fullres_relation_fusion_decoder_d2_preregistered_v1.json"
)


def _contract() -> dict:
    return json.loads(CONFIG_PATH.read_text())["d2_discovery_gate"]


def test_preregistered_config_and_source40_digests_are_immutable() -> None:
    config, digest = load_d2_config(CONFIG_PATH)
    manifest = json.loads(
        (PROJECT_ROOT / "data/interim/validation_manifest.json").read_text()
    )
    records, names = selected_records(config, manifest)
    assert digest == EXPECTED_CONFIG_SHA256
    assert len(records) == len(names) == 40
    assert [str(record["filename"]) for record in records] == list(names)


def test_d2_gate_accepts_exact_branch_at_registered_boundary() -> None:
    result = evaluate_d2_gate(
        mean_exact_delta=0.1,
        mean_adjacency_delta=-0.002,
        strict_permutation_count=40,
        contract=_contract(),
    )
    assert result["pass"]
    assert result["checks"]["exact_branch"]["pass"]
    assert not result["promotion_authorized"]
    assert not result["competition_test_authorized"]


def test_d2_gate_accepts_adjacency_branch_but_rejects_exact_regression() -> None:
    accepted = evaluate_d2_gate(
        mean_exact_delta=0.0,
        mean_adjacency_delta=0.0005,
        strict_permutation_count=40,
        contract=_contract(),
    )
    rejected = evaluate_d2_gate(
        mean_exact_delta=-0.001,
        mean_adjacency_delta=0.05,
        strict_permutation_count=40,
        contract=_contract(),
    )
    assert accepted["pass"]
    assert accepted["checks"]["adjacency_branch"]["pass"]
    assert not rejected["pass"]


def test_d2_gate_requires_all_strict_permutations_and_adjacency_bound() -> None:
    missing_permutation = evaluate_d2_gate(
        mean_exact_delta=5.0,
        mean_adjacency_delta=0.01,
        strict_permutation_count=39,
        contract=_contract(),
    )
    adjacency_loss = evaluate_d2_gate(
        mean_exact_delta=5.0,
        mean_adjacency_delta=-0.002001,
        strict_permutation_count=40,
        contract=_contract(),
    )
    assert not missing_permutation["pass"]
    assert not adjacency_loss["pass"]
