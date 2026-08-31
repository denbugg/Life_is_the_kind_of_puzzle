from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_taska_selective_vote500.py"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_selective_vote500_test", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_cli_exposes_no_threshold_budget_or_arm_tuning() -> None:
    args = runner.parse_args([])
    assert args.output_dir == runner.DEFAULT_OUTPUT
    assert args.device == "mps"
    with pytest.raises(SystemExit):
        runner.parse_args(["--threshold", "1"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--vote-target", "450"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--max-swaps", "192"])


def test_fixed_protocol_and_frozen_input_hashes() -> None:
    assert runner.VOTE_TARGET == 500
    assert runner.SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD == 0.0
    assert runner.LOCAL_GATE == 0.0
    assert runner.HELD_GATE == 0.5
    assert runner.PAIR_DENOMINATOR == 1104
    assert set(runner.PANELS) == {"local32", "held32", "fresh32"}
    assert all(spec.case_count == 32 for spec in runner.PANELS.values())
    runner._require_inputs()


def test_smoke_panel_reads_only_one_frozen_target_free_row() -> None:
    parent = runner.PANELS["local32"]
    smoke = runner.PanelSpec(
        "smoke1", 1, parent.historical_archive, parent.historical_metadata
    )
    rows = runner._historical_rows(smoke)
    assert len(rows) == 1
    assert rows[0]["prefix"] == "case_0000"


def test_cluster_bootstrap_is_source_deterministic() -> None:
    sources = ["a", "a", "b", "b"]
    values = [2.0, 0.0, -1.0, 1.0]
    first = runner._cluster_ci(values, sources, seed=19)
    second = runner._cluster_ci(values, sources, seed=19)
    assert first == second
    assert first["mean"] == 0.5
    assert first["case_wins_ties_losses"] == {
        "wins": 2,
        "ties": 1,
        "losses": 1,
    }
