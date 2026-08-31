from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_taska_fullres_focal_gated_tail.py"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_fullres_focal_gated_tail_test", SCRIPT
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


def test_cli_has_no_threshold_or_budget_surface() -> None:
    args = runner.parse_args([])
    assert args.output_dir == runner.DEFAULT_OUTPUT
    with pytest.raises(SystemExit):
        runner.parse_args(["--threshold", "1"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--max-swaps", "192"])


def test_summary_reports_both_controls(monkeypatch) -> None:
    monkeypatch.setattr(runner, "BOOTSTRAP_RESAMPLES", 128)
    rows = [
        {
            "source_filename": "a.png",
            "metrics": {
                "four_arm_control_tail96": _metric(10, 1),
                "fullres_five_arm_tail96": _metric(13, 2),
                "combo_focal_gated_tail96": _metric(15, 3),
            },
        },
        {
            "source_filename": "b.png",
            "metrics": {
                "four_arm_control_tail96": _metric(11, 1),
                "fullres_five_arm_tail96": _metric(12, 1),
                "combo_focal_gated_tail96": _metric(11, 1),
            },
        },
    ]
    frozen = [
        {
            "five_arm_choice": "fullres_union_focal",
            "combo": {
                "current_edge_count": 400,
                "accepted_new_edge_count": 20,
                "focal_gate": {
                    "focal_kept_edge_count": 310,
                    "tail": {"protected_tile_count": 300, "accepted_swap_count": 96},
                },
            },
        },
        {
            "five_arm_choice": "raw",
            "combo": {
                "current_edge_count": 380,
                "accepted_new_edge_count": 10,
                "focal_gate": {
                    "focal_kept_edge_count": 290,
                    "tail": {"protected_tile_count": 310, "accepted_swap_count": 90},
                },
            },
        },
    ]
    summary = runner._summarize(rows, frozen)
    assert summary["combo_minus_fullres"]["satisfied_adjacent_pairs"]["mean"] == 0.5
    assert summary["combo_minus_four_arm"]["satisfied_adjacent_pairs"]["mean"] == 2.5
    assert summary["target_free_diagnostics"]["mean_focal_kept_edges"] == 300


def test_fixed_parent_report_hash_is_pinned() -> None:
    assert runner.FIXED_INPUTS[runner.FULLRES_REPORT] == (
        "d67a7ed7e2cd9e7c333052ab4db9d0b32e444980da83939f0e54e7f88c7195b8"
    )
