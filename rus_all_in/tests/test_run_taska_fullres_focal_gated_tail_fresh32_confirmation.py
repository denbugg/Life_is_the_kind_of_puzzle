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
    / "scripts/run_taska_fullres_focal_gated_tail_fresh32_confirmation.py"
)
REPORT = (
    PROJECT_ROOT
    / "outputs/taska-fullres-focal-gated-tail/fresh32-confirmation-v1/report.json"
)


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_fullres_focal_gated_tail_fresh32_confirmation_test",
        SCRIPT,
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


def _supply() -> dict[str, int | bool]:
    return {
        "current_candidate_count": 400,
        "current_true_edges": 280,
        "proposed_absent_count": 200,
        "proposed_true_edges": 40,
        "accepted_new_count": 30,
        "accepted_new_true_edges": 20,
        "union_candidate_count": 430,
        "union_true_edges": 300,
        "accepted_edges_are_current_absent": True,
    }


def test_cli_exposes_no_threshold_budget_orientation_or_panel_sweep() -> None:
    args = runner.parse_args([])
    assert args.output_dir == runner.DEFAULT_OUTPUT
    for forbidden in (
        ["--threshold", "1"],
        ["--max-swaps", "192"],
        ["--support", "2"],
        ["--orientation", "2"],
        ["--source-count", "8"],
    ):
        with pytest.raises(SystemExit):
            runner.parse_args(forbidden)


def test_signed_preregistration_reconstructs_the_registered_roster() -> None:
    config = runner._load_config(runner.DEFAULT_CONFIG)
    roster = runner._validate_preregistration(config)
    assert len(roster) == runner.SOURCE_COUNT == 16
    assert runner._names_digest(roster) == (
        "3120b719d7cbf496f5505e0459ecdf597a98637c841c8ef843eb11945adf6c1a"
    )
    assert runner._cases_digest(roster) == (
        "999667ad4ee95b35cc76537a7e1b99f3f3d96b2dc92c8c42eca80baeef0ac745"
    )


def test_summary_uses_source_cluster_bootstrap_and_fixed_gate(monkeypatch) -> None:
    monkeypatch.setattr(runner, "BOOTSTRAP_RESAMPLES", 128)
    rows = []
    for source in range(runner.SOURCE_COUNT):
        for draw in runner.DRAWS:
            rows.append(
                {
                    "source_filename": f"img_{source:06d}.png",
                    "draw_index": draw,
                    "four_arm_choice": "raw",
                    "five_arm_choice": "fullres_union_focal",
                    "candidate_supply": _supply(),
                    runner.ARMS[0]: _metric(100, 1),
                    runner.ARMS[1]: _metric(103, 2),
                    runner.ARMS[2]: _metric(108, 2),
                }
            )
    summary = runner._summarize(rows)
    primary = summary["comparisons"]["combo_minus_control"][
        "satisfied_adjacent_pairs"
    ]
    assert primary["mean"] == 8.0
    assert primary["source_count"] == 16
    assert primary["case_count"] == 32
    assert summary["confirmation_gate"]["passed"] is True
    assert summary["candidate_supply"]["accepted_new_precision"] == pytest.approx(
        2 / 3
    )


def test_completed_report_preserves_confirmation_and_legality_contract() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == runner.REPORT_SCHEMA
    assert report["status"] == "confirmed"
    assert report["metrics"]["confirmation_gate"]["passed"] is True
    assert report["legality"]["competition_test_accessed"] is False
    assert report["legality"]["postprocessing_used"] is False
    assert report["legality"]["restored_pixels_matcher_only"] is True
    assert len(report["rows"]) == runner.CASE_COUNT
    for row in report["rows"]:
        for arm in runner.ARMS:
            assert row[arm]["strict_permutation"] is True
