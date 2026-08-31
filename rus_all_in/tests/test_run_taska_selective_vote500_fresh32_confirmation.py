from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_taska_selective_vote500_fresh32_confirmation.py"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_selective_vote500_fresh32_confirmation_test", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_cli_has_no_threshold_arm_budget_or_roster_tuning() -> None:
    args = runner.parse_args([])
    assert args.config == runner.DEFAULT_CONFIG
    assert args.output_dir == runner.DEFAULT_OUTPUT
    for option in ("--threshold", "--arm", "--max-swaps", "--seed", "--source"):
        with pytest.raises(SystemExit):
            runner.parse_args([option, "1"])


def test_signed_preregistration_reconstructs_new_disjoint_roster() -> None:
    config = runner._load_config(runner.DEFAULT_CONFIG)
    roster, _ = runner._validate_preregistration(runner.DEFAULT_CONFIG, config)
    assert roster == tuple(config["panel"]["source_filenames"])
    assert len(roster) == 16
    assert all(6_700 <= int(name[4:10]) <= 6_999 for name in roster)
    assert all(not (6_400 <= int(name[4:10]) <= 6_699) for name in roster)
    snapshot_path = PROJECT_ROOT / config["artifacts"]["exclusion_snapshot"]["path"]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    excluded = set(snapshot["explicit_source_union"]["source_filenames"])
    assert not set(roster) & excluded
    for field in ("tail192", "fullres_combo_confirmation"):
        dependency = PROJECT_ROOT / snapshot["required_signed_reservations"][field]["path"]
        payload = json.loads(dependency.read_text(encoding="utf-8"))
        assert set(payload["panel"]["source_filenames"]) <= excluded


def test_preregistered_solver_and_gate_are_fixed() -> None:
    config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    sidecar = Path(f"{runner.DEFAULT_CONFIG}.sha256")
    assert sidecar.read_text(encoding="utf-8").split()[0] == runner.sha256_file(
        runner.DEFAULT_CONFIG
    )
    assert config["protocol"]["config_and_sha_sidecar_created_before_inference_or_scoring"]
    assert config["candidate"]["entrypoint"].endswith("solve_selective_vote500")
    assert config["candidate"]["matcher_passes_per_case"] == 1
    assert config["candidate"]["matcher_vote_target"] == 500
    assert config["candidate"]["same_pass_current_vote_target"] == 350
    assert config["candidate"]["tail_max_swaps"] == 96
    assert not config["candidate"]["threshold_arm_or_budget_sweep"]
    assert runner.PAIR_GATE_MEAN == 2.0
    assert runner.PAIR_GATE_CI95_LOWER == 0.0


def test_edge_archive_round_trip_and_truth_has_all_bonds() -> None:
    prefix = "case_000"
    edges = (
        RawTailEdge(1, 2, "right"),
        RawTailEdge(3, 4, "down"),
    )
    arrays = runner._edge_arrays(prefix, "accepted_new", edges)
    assert runner._frozen_edges(arrays, prefix, "accepted_new") == set(edges)
    truth = runner._truth_edges(np.arange(576, dtype=np.int32))
    assert len(truth) == 1104


def test_source_cluster_bootstrap_and_freeze_order_are_fixed() -> None:
    sources = [f"source_{index:02d}" for index in range(16) for _ in range(2)]
    interval = runner._cluster_ci([2.0] * 32, sources, seed=123)
    assert interval["mean"] == 2.0
    assert interval["ci95_lower"] == 2.0
    assert interval["source_count"] == 16
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index("_freeze_target_free(") < source.index("_score_after_freeze(")
    assert '"reference_reconstructed_yet": False' in source
