from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_taska_monotone_components.py"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_monotone_components_test",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_fixed_experiment_contract() -> None:
    assert runner.SEARCH_ROUNDS == 6
    assert runner.TAIL_MAX_SWAPS == 96
    assert runner.TAIL_MINIMUM_GAIN == 1e-9
    assert tuple(runner.PANEL_SPECS) == ("opened32", "held32", "fresh32")
    assert runner.PARENT_PANEL == {
        "opened32": "opened32",
        "held32": "held300",
        "fresh32": "fresh32",
    }
    assert runner.SCORED_ARMS == (
        "control_tail96",
        "monotone_portfolio",
        "monotone_tail96",
    )
    assert runner.EXPECTED_CONTROL_MEANS == {
        "opened32": (341.3125, 4.75),
        "held32": (337.5625, 3.0625),
        "fresh32": (346.0625, 1.15625),
    }


def test_scored_layout_metric_contract_is_1104_pairs() -> None:
    exact = np.arange(576, dtype=np.int32)
    metrics = runner.replay.parent._layout_metrics(exact, exact)
    assert metrics == {
        "satisfied_adjacent_pairs": 1104,
        "adjacency_recall": 1.0,
        "exact_tiles": 576,
        "strict_permutation": True,
    }


def test_runner_freezes_target_free_layouts_before_scoring() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "created_before_exact_reference_recreation" in source
    assert "unconditional_two_component_relocation_loop_omitted" in source
    assert source.index("_freeze_target_free(") < source.index("_score_after_freeze(")
