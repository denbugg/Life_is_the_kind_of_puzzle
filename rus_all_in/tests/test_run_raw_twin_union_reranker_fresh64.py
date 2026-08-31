from __future__ import annotations

from pathlib import Path

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import names_digest
from scripts.run_raw_twin_union_reranker_fresh64 import (
    COMPONENT_SELECTION,
    DEFAULT_CONFIG,
    EXPECTED_SOURCES,
    UNION_SELECTION,
    _ordered_roster_names,
    evaluate_gate,
    load_config,
    source_clustered_ci,
)


def test_fresh64_config_and_sidecar_are_frozen_before_access() -> None:
    config, digest = load_config(DEFAULT_CONFIG)
    assert digest == sha256_file(DEFAULT_CONFIG)
    assert config["registered_before_selected_target_access"] is True
    assert config["registered_before_dirty_prediction_generation"] is True
    names = config["selection"]["source_filenames"]
    assert len(names) == EXPECTED_SOURCES == len(set(names))
    assert names_digest(names) == config["selection"]["source_order_digest"]
    sidecar = Path(f"{DEFAULT_CONFIG}.sha256").read_text(encoding="utf-8").split()[0]
    assert sidecar == digest


def test_union_and_component_complete_rosters_are_excluded() -> None:
    config, _ = load_config(DEFAULT_CONFIG)
    selected = set(config["selection"]["source_filenames"])
    union = set(_ordered_roster_names(UNION_SELECTION))
    component = set(_ordered_roster_names(COMPONENT_SELECTION))
    assert len(union) == 280
    assert len(component) == 288
    assert selected.isdisjoint(union | component)
    assert config["selection"]["union_fit_eval_excluded"] == 280
    assert config["selection"]["component_fit_eval_excluded"] == 288


def test_source_clustered_ci_is_deterministic_and_contains_mean() -> None:
    values = np.linspace(-1.0, 2.0, 64)
    first = source_clustered_ci(values, seed=7, resamples=1000)
    second = source_clustered_ci(values, seed=7, resamples=1000)
    assert first == second
    assert first["mean"] == float(values.mean())
    assert first["ci95_lower"] < first["mean"] < first["ci95_upper"]


def test_submission_gate_uses_exact_adjacency_and_strictness_only() -> None:
    positive = {
        "exact_tiles_delta": {"mean": 0.25, "ci95_lower": -0.2},
        "adjacency_delta": {"mean": 1e-6},
    }
    passed = evaluate_gate(positive, strict_layouts=128)
    assert passed["pass"] is True
    assert passed["exact_ci_excludes_zero"] is False
    assert "confirmed" in passed["status"]
    assert evaluate_gate(positive, strict_layouts=127)["pass"] is False
    negative = {
        "exact_tiles_delta": {"mean": 0.249, "ci95_lower": 0.1},
        "adjacency_delta": {"mean": 0.1},
    }
    assert evaluate_gate(negative, strict_layouts=128)["pass"] is False
