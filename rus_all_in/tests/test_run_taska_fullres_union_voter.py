from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_taska_fullres_union_voter.py"
    specification = importlib.util.spec_from_file_location(
        "run_taska_fullres_union_voter_test", path
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


def test_summary_tracks_pairs_exact_and_candidate_supply(monkeypatch) -> None:
    monkeypatch.setattr(runner, "BOOTSTRAP_RESAMPLES", 128)
    rows = [
        {
            "source_filename": "a.png",
            "five_arm_choice": "fullres_union_focal",
            "metrics": {
                "fullres_union_focal": _metric(12, 2),
                "control_tail96": _metric(10, 1),
                "five_arm_tail96": _metric(13, 3),
            },
            "candidate_supply": {
                "current_true_edges": 300,
                "proposed_absent_edges": 10,
                "proposed_absent_true_edges": 4,
                "accepted_new_edges": 5,
                "accepted_new_true_edges": 3,
                "union_true_edges": 303,
            },
        },
        {
            "source_filename": "b.png",
            "five_arm_choice": "raw",
            "metrics": {
                "fullres_union_focal": _metric(9, 0),
                "control_tail96": _metric(11, 2),
                "five_arm_tail96": _metric(10, 2),
            },
            "candidate_supply": {
                "current_true_edges": 302,
                "proposed_absent_edges": 8,
                "proposed_absent_true_edges": 2,
                "accepted_new_edges": 4,
                "accepted_new_true_edges": 1,
                "union_true_edges": 303,
            },
        },
    ]
    summary = runner._summarize(rows)
    assert summary["arms"]["five_arm_tail96"]["satisfied_adjacent_pairs"] == 11.5
    assert summary["five_minus_control"]["satisfied_adjacent_pairs"]["mean"] == 1.0
    assert summary["candidate_supply_mean_per_board"]["accepted_new_edges"] == 4.5
    assert summary["candidate_supply"]["accepted_new_precision"] == 4 / 9
    assert summary["candidate_supply"]["union_recall"] == 303 / 1104


def test_declared_panels_are_local32_held32_fresh32() -> None:
    assert runner.PANELS["local32"].case_count == 32
    assert runner.PANELS["held32"].case_count == 32
    assert runner.PANELS["fresh32"].case_count == 32
