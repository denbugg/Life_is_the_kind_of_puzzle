from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_component_relation_cyclic_fresh_gate.py"
    specification = importlib.util.spec_from_file_location(
        "run_component_relation_cyclic_fresh_gate_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _gate_contract() -> dict:
    return {
        "minimum_mean_exact_tiles_gain_per_board": 0.5,
        "minimum_source_cluster_bootstrap_ci95_lower_exact_gain_strictly_greater_than": 0.0,
        "minimum_adjacency_delta_fraction": -0.002,
        "strict_original_permutation_count_required": 128,
    }


def test_frozen_config_and_roster_reproduce_before_target_access() -> None:
    config, digest = runner.load_frozen_config(
        PROJECT_ROOT / "configs/component_relation_cyclic_fresh_gate_v1.json"
    )
    manifest = json.loads(
        (PROJECT_ROOT / "data/interim/validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    records, names = runner.validate_selection(
        config,
        manifest,
        {"excluded_filenames": set()},
    )
    assert digest == "7bfd475c5c65bf56d4627eaf718cdf06eb12ce6fdf9f939e13a682782c225b44"
    assert len(records) == len(names) == 64
    assert runner.filename_digest(names) == config["selection"]["source_order_digest"]
    assert config["arms"]["candidate"]["query_confidence_cap"] == 32
    assert config["arms"]["candidate"]["hard_edge_bonus_scale"] == 0.25
    assert config["arms"]["candidate"]["cyclic_border_weight"] == 5.0


def test_source_cluster_bootstrap_uses_two_draws_per_source() -> None:
    rows = []
    expected_source_deltas = []
    for source_index in range(64):
        first = float((source_index % 5) - 2)
        second = first + 2.0
        expected_source_deltas.append((first + second) / 2.0)
        for draw_index, delta in enumerate((first, second)):
            rows.append(
                {
                    "source_filename": f"img_{source_index:06d}.png",
                    "draw_index": draw_index,
                    "exact_delta_tiles": delta,
                }
            )
    result = runner.paired_source_cluster_bootstrap(rows, samples=2_000, seed=17)
    assert result["source_count"] == 64
    assert result["case_count"] == 128
    assert result["mean_delta_per_board"] == pytest.approx(
        sum(expected_source_deltas) / 64
    )

    with pytest.raises(ValueError, match="source64xdraw2"):
        runner.paired_source_cluster_bootstrap(rows[:-1], samples=20, seed=17)


def test_fresh_gate_requires_every_preregistered_condition() -> None:
    passing = {
        "mean_delta_per_board": 0.5,
        "source_cluster_bootstrap_ci95": [0.001, 1.0],
    }
    result = runner.evaluate_gate(
        passing,
        adjacency_delta=-0.002,
        strict_permutations=128,
        contract=_gate_contract(),
    )
    assert result["pass"]
    assert result["status"] == "pass-await-root-review"
    assert not result["competition_test_authorized"]

    zero_lower = dict(passing)
    zero_lower["source_cluster_bootstrap_ci95"] = [0.0, 1.0]
    assert not runner.evaluate_gate(
        zero_lower,
        adjacency_delta=-0.002,
        strict_permutations=128,
        contract=_gate_contract(),
    )["pass"]
    assert not runner.evaluate_gate(
        passing,
        adjacency_delta=-0.00201,
        strict_permutations=128,
        contract=_gate_contract(),
    )["pass"]
    assert not runner.evaluate_gate(
        passing,
        adjacency_delta=-0.002,
        strict_permutations=127,
        contract=_gate_contract(),
    )["pass"]
