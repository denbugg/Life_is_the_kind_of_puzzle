from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT_ROOT
    / "scripts/run_taska_focal_gated_protected_tail_fresh16_confirmation.py"
)


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_focal_gated_protected_tail_fresh16_confirmation_test",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_cli_has_no_candidate_tuning_surface() -> None:
    args = runner.parse_args([])
    assert args.config == runner.DEFAULT_CONFIG
    assert args.output_dir == runner.DEFAULT_OUTPUT
    with pytest.raises(SystemExit):
        runner.parse_args(["--threshold", "1"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--max-swaps", "192"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--arm", "raw"])


def test_preregistration_reconstructs_exact_fixed_roster_and_exclusions() -> None:
    config = runner._load_config(runner.DEFAULT_CONFIG)
    roster = runner._validate_preregistration(config)
    assert len(roster) == 16
    assert roster == tuple(config["panel"]["source_filenames"])
    rosters = runner._registered_rosters(config)
    excluded = set().union(*(set(values) for values in rosters.values()))
    assert not set(roster) & excluded
    assert len(rosters["train256"]) == 256
    assert rosters["local32_96_128"] == rosters["train256"][96:128]
    assert rosters["focal_current_training224"] == (
        rosters["train256"][:96] + rosters["train256"][128:256]
    )
    assert len(rosters["fullres_terminal16"]) == 16


def test_preregistration_sidecar_is_frozen_before_scoring() -> None:
    sidecar = Path(f"{runner.DEFAULT_CONFIG}.sha256")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    assert tokens[0] == runner.sha256_file(runner.DEFAULT_CONFIG)
    payload = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert payload["protocol"][
        "preregistration_and_sha_sidecar_created_before_selected_reference_scoring"
    ]
    assert payload["protocol"]["one_panel_only"]
    assert not payload["protocol"]["threshold_budget_or_arm_search"]


def test_fixed_candidate_and_gate_constants() -> None:
    assert runner.FOCAL_PROTECTION_LOGIT_THRESHOLD == 0.0
    assert runner.TAIL_MAX_SWAPS == 96
    assert runner.TAIL_MINIMUM_GAIN == 1e-9
    assert runner.PAIR_DENOMINATOR == 1104
    assert runner.PAIR_GATE_MEAN == 0.5
    assert runner.PAIR_GATE_CI95_LOWER == -0.25
    assert runner.RAW_SOLVER_SHA256 == (
        "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
    )


def test_source_cluster_bootstrap_is_deterministic() -> None:
    sources = [f"source-{index:02d}" for index in range(16) for _ in range(2)]
    values = [1.0, 0.0] * 16
    first = runner._cluster_ci(values, sources, seed=17)
    second = runner._cluster_ci(values, sources, seed=17)
    assert first == second
    assert first["mean"] == 0.5
    assert first["source_wins_ties_losses"] == {
        "wins": 16,
        "ties": 0,
        "losses": 0,
    }
