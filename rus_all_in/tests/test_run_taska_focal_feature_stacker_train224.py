from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_taska_focal_feature_stacker_train224.py"
    spec = importlib.util.spec_from_file_location("train224_stacker_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _metric(pairs: int, exact: int) -> dict[str, float | int | bool]:
    return {
        "satisfied_adjacent_pairs": pairs,
        "adjacency_recall": pairs / runner.baseline.PAIR_DENOMINATOR,
        "exact_tiles": exact,
        "strict_permutation": True,
    }


def test_summary_compares_train224_to_both_controls(monkeypatch) -> None:
    monkeypatch.setattr(runner.baseline, "BOOTSTRAP_RESAMPLES", 128)
    rows = [
        {
            "source_filename": "a.png",
            "four_arm_choice": "raw",
            "train96_five_arm_choice": "stacker",
            "train224_five_arm_choice": "stacker",
            "metrics": {
                "stacker": _metric(8, 2),
                "four_arm_tail96": _metric(10, 1),
                "train96_five_arm_tail96": _metric(11, 2),
                "train224_five_arm_tail96": _metric(12, 3),
            },
        },
        {
            "source_filename": "b.png",
            "four_arm_choice": "focal_top5",
            "train96_five_arm_choice": "focal_top5",
            "train224_five_arm_choice": "focal_top5",
            "metrics": {
                "stacker": _metric(12, 1),
                "four_arm_tail96": _metric(20, 4),
                "train96_five_arm_tail96": _metric(18, 4),
                "train224_five_arm_tail96": _metric(19, 5),
            },
        },
    ]
    summary = runner._summarize(rows)
    assert summary["deltas"]["train224_minus_four"][
        "satisfied_adjacent_pairs"
    ]["mean"] == 0.5
    assert summary["deltas"]["train224_minus_train96"][
        "satisfied_adjacent_pairs"
    ]["mean"] == 1.0
    assert summary["deltas"]["train224_minus_train96"]["exact_tiles"]["mean"] == 1.0


def test_fresh_gate_allows_pair_or_exact_signal() -> None:
    def held(pair_four: float, pair_train96: float, exact_train96: float):
        return {
            "summary": {
                "deltas": {
                    "train224_minus_four": {
                        "satisfied_adjacent_pairs": {"mean": pair_four}
                    },
                    "train224_minus_train96": {
                        "satisfied_adjacent_pairs": {"mean": pair_train96},
                        "exact_tiles": {"mean": exact_train96},
                    },
                }
            }
        }

    assert runner._fresh_gate(held(0.0, -3.0, -1.0))["passed"] is True
    assert runner._fresh_gate(held(-2.0, -0.75, 0.25))["passed"] is True
    assert runner._fresh_gate(held(-2.0, -1.25, 0.25))["passed"] is False


def test_fixed_train_indices_exclude_local32() -> None:
    assert len(runner.TRAIN256_INDICES) == runner.TRAIN224_COUNT
    assert not any(96 <= int(index) < 128 for index in runner.TRAIN256_INDICES)
