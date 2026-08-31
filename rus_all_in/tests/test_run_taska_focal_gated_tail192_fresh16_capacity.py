from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_taska_focal_gated_tail192_fresh16_capacity.py"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_focal_gated_tail192_fresh16_capacity_test", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_cli_exposes_no_budget_threshold_or_arm_tuning() -> None:
    args = runner.parse_args([])
    assert args.config == runner.DEFAULT_CONFIG
    assert args.output_dir == runner.DEFAULT_OUTPUT
    for option in ("--max-swaps", "--threshold", "--arm", "--seed"):
        with pytest.raises(SystemExit):
            runner.parse_args([option, "1"])


def test_preregistration_reconstructs_roster_and_full_collision_audit() -> None:
    config = runner._load_config(runner.DEFAULT_CONFIG)
    roster = runner._validate_preregistration(config)
    assert roster == tuple(config["panel"]["source_filenames"])
    assert len(roster) == 16
    rosters = runner._registered_rosters(config)
    excluded = set().union(*(set(values) for values in rosters.values()))
    assert not set(roster) & excluded
    required = {
        "taska_train256",
        "taska_extension128",
        "taska_focal_train224",
        "taska_local32",
        "taska_held32",
        "taska_fresh32",
        "taska_fresh16_confirmation",
        "fullres_denoiser_train32",
        "fullres_denoiser_eval16",
        "active_fullres_union_fresh",
        "active_incidence_fresh",
        "active_focal_gate_fresh",
    }
    assert required <= set(rosters)


def test_config_sidecar_predates_scoring_and_candidate_is_fixed() -> None:
    sidecar = Path(f"{runner.DEFAULT_CONFIG}.sha256")
    assert sidecar.read_text(encoding="utf-8").split()[0] == runner.sha256_file(
        runner.DEFAULT_CONFIG
    )
    config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"][
        "config_and_sha_sidecar_created_before_candidate_code_and_scoring"
    ]
    assert config["protocol"]["collision_audit_fail_closed"]
    assert config["candidate"]["control_tail_max_swaps"] == 96
    assert config["candidate"]["candidate_tail_max_swaps"] == 192
    assert not config["candidate"]["budget_threshold_arm_sweep"]


def test_fixed_pair_primary_gate_and_raw_solver() -> None:
    assert runner.PAIR_GATE_MEAN == 0.5
    assert runner.PAIR_GATE_CI95_LOWER == -0.25
    assert runner.PAIR_DENOMINATOR == 1104
    assert runner.sha256_file(
        runner.PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
    ) == runner.RAW_SOLVER_SHA256


def test_freeze_occurs_before_reference_scoring_in_runner() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index("_freeze_target_free(") < source.index("_score_after_freeze(")
    assert "reference_reconstructed_yet\": False" in source
