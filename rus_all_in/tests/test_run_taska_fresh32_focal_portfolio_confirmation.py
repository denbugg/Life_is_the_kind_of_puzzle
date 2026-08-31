from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_taska_fresh32_focal_portfolio_confirmation.py"
    specification = importlib.util.spec_from_file_location(
        "run_taska_fresh32_focal_portfolio_confirmation_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_fixed_candidate_roster_has_no_sweep_arm() -> None:
    assert runner.PORTFOLIO_ARMS == ("raw", "logistic", "focal", "nonlinear")
    assert (
        *runner.PORTFOLIO_ARMS,
        "portfolio",
        "portfolio_tail96",
    ) == runner.SCORED_ARMS
    assert runner.PRIMARY_ARMS == ("focal", "portfolio_tail96")
    assert runner.PAIR_DENOMINATOR == 1104


def test_layout_contract_is_strict() -> None:
    layout = runner._strict_layout(np.arange(576)[::-1])
    assert layout.dtype == np.int32
    assert np.array_equal(np.sort(layout), np.arange(576))


def test_source_wins_are_clustered_over_two_draws() -> None:
    result = runner._source_wins_ties_losses(
        [2.0, -1.0, 1.0, -1.0, -3.0, -2.0],
        ["a", "a", "b", "b", "c", "c"],
    )
    assert result == {"wins": 1, "ties": 1, "losses": 1}
