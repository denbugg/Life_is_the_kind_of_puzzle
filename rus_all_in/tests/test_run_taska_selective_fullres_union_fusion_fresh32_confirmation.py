from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT_ROOT
    / "scripts/run_taska_selective_fullres_union_fusion_fresh32_confirmation.py"
)
REPORT = (
    PROJECT_ROOT
    / "outputs/taska-selective-fullres-union-fusion/"
    "fresh32-formal-confirmation-v1/report.json"
)


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_selective_fullres_fusion_confirmation_test", SCRIPT
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


def _supply() -> dict[str, int]:
    return {
        "current_edge_count": 400,
        "current_true_edges": 280,
        "selective_accepted_new_count": 60,
        "selective_accepted_true_edges": 20,
        "fullres_accepted_new_count": 30,
        "fullres_accepted_true_edges": 15,
        "unique_fullres_accepted_count": 20,
        "unique_fullres_true_edges": 10,
        "combined_union_edge_count": 480,
        "combined_union_true_edges": 310,
    }


def test_cli_exposes_no_candidate_or_roster_sweep() -> None:
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


def test_signed_preregistration_reconstructs_disjoint_roster() -> None:
    config = runner._load_config(runner.DEFAULT_CONFIG)
    roster, _ = runner._validate_preregistration(runner.DEFAULT_CONFIG, config)
    assert len(roster) == runner.SOURCE_COUNT == 16
    assert runner._digest(roster) == (
        "46675bda2ae6280b7894793cb9c96c52de4824fb9f5e6a7544bccee921fdc848"
    )
    assert runner._cases_digest(roster) == (
        "5a6cea1273009339f616475cf73c963e199169b47d46b798dcf167a42ba621a5"
    )
    snapshot_path = PROJECT_ROOT / config["artifacts"]["exclusion_snapshot"]["path"]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    excluded = set(snapshot["explicit_source_union"]["source_filenames"])
    selective_path = (
        PROJECT_ROOT
        / config["artifacts"]["selective_target500_confirmation_reservation"][
            "path"
        ]
    )
    selective = json.loads(selective_path.read_text(encoding="utf-8"))
    excluded.update(selective["panel"]["source_filenames"])
    assert not set(roster) & excluded


def test_archive_edges_round_trip_and_truth_has_all_bonds() -> None:
    edges = (RawTailEdge(1, 2, "right"), RawTailEdge(3, 4, "down"))
    raw = runner._edge_arrays("unique_fullres", edges)
    archive = {f"case_000__{name}": value for name, value in raw.items()}
    assert runner._edges_from_archive(
        archive, "case_000", "unique_fullres"
    ) == edges
    assert len(runner._truth_edges(np.arange(576, dtype=np.int32))) == 1104


def test_summary_uses_source_cluster_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "BOOTSTRAP_RESAMPLES", 128)
    rows = []
    for source in range(runner.SOURCE_COUNT):
        for draw in runner.DRAWS:
            rows.append(
                {
                    "source_filename": f"img_{source:06d}.png",
                    "draw_index": draw,
                    "fusion_choice": "combined_union_focal",
                    "selective_choice": "selective_vote500_focal",
                    "candidate_supply": _supply(),
                    runner.CONTROL_ARM: _metric(100, 1),
                    runner.CANDIDATE_ARM: _metric(102, 2),
                }
            )
    summary = runner._summarize(rows)
    primary = summary["candidate_minus_control"]["satisfied_adjacent_pairs"]
    assert primary["mean"] == 2.0
    assert primary["ci95_lower"] == 2.0
    assert summary["confirmation_gate"]["passed"] is True
    assert summary["candidate_supply"]["unique_fullres_precision"] == 0.5


def test_completed_report_preserves_freeze_and_legality_contract() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == runner.REPORT_SCHEMA
    assert report["legality"]["competition_test_accessed"] is False
    assert report["legality"]["postprocessing_used"] is False
    assert report["legality"]["restored_pixels_matcher_only"] is True
    assert report["frozen_eval"]["contains_exact_references_or_labels"] is False
    assert len(report["rows"]) == runner.CASE_COUNT
    for row in report["rows"]:
        for arm in runner.ARMS:
            assert row[arm]["strict_permutation"] is True
