from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_taska_focal_feature_stacker.py"
    specification = importlib.util.spec_from_file_location(
        "run_taska_focal_feature_stacker_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _metric(pairs: int, exact: int) -> dict[str, float | int | bool]:
    return {
        "satisfied_adjacent_pairs": pairs,
        "adjacency_recall": pairs / runner.PAIR_DENOMINATOR,
        "exact_tiles": exact,
        "strict_permutation": True,
    }


def test_stage_summary_uses_five_minus_four_and_source_clusters(monkeypatch) -> None:
    monkeypatch.setattr(runner, "BOOTSTRAP_RESAMPLES", 128)
    rows = [
        {
            "source_filename": "a.png",
            "four_arm_choice": "raw",
            "five_arm_choice": "stacker",
            "metrics": {
                "stacker": _metric(10, 2),
                "four_arm_tail96": _metric(12, 1),
                "five_arm_tail96": _metric(15, 3),
            },
        },
        {
            "source_filename": "b.png",
            "four_arm_choice": "focal_top5",
            "five_arm_choice": "focal_top5",
            "metrics": {
                "stacker": _metric(20, 1),
                "four_arm_tail96": _metric(21, 4),
                "five_arm_tail96": _metric(20, 4),
            },
        },
    ]
    summary = runner._summarize(rows)
    assert summary["arms"]["five_arm_tail96"]["satisfied_adjacent_pairs"] == 17.5
    delta = summary["five_minus_four"]["satisfied_adjacent_pairs"]
    assert delta["mean"] == 1.0
    assert delta["source_count"] == 2
    assert delta["case_wins_ties_losses"] == {"wins": 1, "ties": 0, "losses": 1}
    assert summary["five_arm_choice_counts"] == {"stacker": 1, "focal_top5": 1}


def test_strict_layout_rejects_duplicates() -> None:
    layout = np.arange(runner.COUNT, dtype=np.int32)
    assert np.array_equal(runner._strict_layout(layout), layout)
    layout[-1] = layout[-2]
    with pytest.raises(ValueError, match="strict"):
        runner._strict_layout(layout)


def test_three_panel_report_disclaims_independent_inference() -> None:
    path = Path(
        "outputs/taska-focal-feature-stacker/train96-v1/"
        "three-panel-descriptive-report.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload["descriptive_equal_case_aggregate"]
    assert aggregate["case_count"] == 96
    assert aggregate["inferential_claim"] is False
    assert aggregate["five_minus_four"]["satisfied_adjacent_pairs_per_board"] == 0.21875
    assert payload["verdict"]["replace_current_pair_default"] is False
