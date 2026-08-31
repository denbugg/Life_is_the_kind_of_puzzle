from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_taska_focal_gated_protected_tail.py"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_focal_gated_protected_tail_test",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_cli_exposes_no_threshold_or_budget_tuning() -> None:
    args = runner.parse_args([])
    assert args.output_dir == runner.DEFAULT_OUTPUT
    assert args.targets == runner.DEFAULT_TARGETS
    with pytest.raises(SystemExit):
        runner.parse_args(["--threshold", "1.0"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--max-swaps", "192"])


def test_fixed_protocol_constants() -> None:
    assert runner.FOCAL_PROTECTION_LOGIT_THRESHOLD == 0.0
    assert runner.TAIL_SWAPS == 96
    assert runner.MINIMUM_GAIN == 1e-9
    assert runner.PAIR_DENOMINATOR == 1104
    assert runner.ARMS == (
        "control_all_edges_tail96",
        "focal_gated_tail96",
    )


def test_cluster_bootstrap_is_deterministic_and_reports_case_signs() -> None:
    values = [2.0, -1.0, 0.0, 3.0]
    sources = ["a", "a", "b", "b"]
    first = runner._cluster_ci(values, sources, seed=17)
    second = runner._cluster_ci(values, sources, seed=17)
    assert first == second
    assert first["mean"] == 1.0
    assert first["source_count"] == 2
    assert first["case_wins_ties_losses"] == {
        "wins": 2,
        "ties": 1,
        "losses": 1,
    }


def test_raw_solver_expected_hash_remains_frozen() -> None:
    path = runner.PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
    assert isinstance(path, Path)
    assert runner.EXPECTED_SHA256[path] == (
        "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
    )
