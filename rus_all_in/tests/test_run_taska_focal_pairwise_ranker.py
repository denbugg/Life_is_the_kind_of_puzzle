from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_pair_pipeline import RAW_TAIL_GLOBAL_SOLVER_SHA256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_taska_focal_pairwise_ranker.py"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_focal_pairwise_ranker_test",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


experiment = _load_runner()


def _metric(value: float) -> dict[str, float | bool]:
    return {
        "satisfied_adjacent_pairs": value,
        "adjacency_recall": value / experiment.PAIR_DENOMINATOR,
        "exact_tiles": value / 10,
        "strict_permutation": True,
    }


def test_fixed_gate_and_panel_contract() -> None:
    assert experiment.FRESH_HELD_PAIR_DELTA_GATE == 1.0
    assert experiment.TAIL_MAX_SWAPS == 96
    assert experiment.STAGE_ARMS == (
        "pairwise_ranker",
        "four_arm_tail96",
        "five_arm_tail96",
    )
    assert tuple(experiment.PANEL_SPECS) == ("local32", "held32", "fresh32")
    for spec in experiment.PANEL_SPECS.values():
        assert spec.metadata.is_file()
        assert spec.base_archive.is_file()
        assert spec.evidence_archive.is_file()
        assert spec.portfolio_archive.is_file()


def test_signed_pairwise_inputs_and_raw_solver_are_unchanged() -> None:
    experiment._require_frozen_inputs()
    solver = experiment.PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
    assert sha256_file(solver) == RAW_TAIL_GLOBAL_SOLVER_SHA256


def test_summary_reports_pair_recall_and_exact_deltas() -> None:
    rows = [
        {
            "source_filename": "a.png",
            "five_arm_choice": "pairwise_ranker",
            "metrics": {
                "pairwise_ranker": _metric(12),
                "four_arm_tail96": _metric(10),
                "five_arm_tail96": _metric(13),
            },
        },
        {
            "source_filename": "b.png",
            "five_arm_choice": "raw",
            "metrics": {
                "pairwise_ranker": _metric(8),
                "four_arm_tail96": _metric(11),
                "five_arm_tail96": _metric(11),
            },
        },
    ]
    summary = experiment._summarize(rows)
    assert summary["pair_denominator"] == 1104
    assert summary["arms"]["five_arm_tail96"]["satisfied_adjacent_pairs"] == 12
    delta = summary["five_minus_four"]
    assert delta["satisfied_adjacent_pairs"]["mean"] == 1.5
    assert np.isclose(delta["adjacency_recall"]["mean"], 1.5 / 1104)
    assert np.isclose(delta["exact_tiles"]["mean"], 0.15)
    assert delta["satisfied_adjacent_pairs"]["case_wins_ties_losses"] == {
        "wins": 1,
        "ties": 1,
        "losses": 0,
    }


def test_local_frozen_evidence_has_no_labels_and_is_aligned() -> None:
    spec = experiment.PANEL_SPECS["local32"]
    with np.load(spec.base_archive, allow_pickle=False) as archive:
        assert not any("label" in key or "reference" in key for key in archive.files)
        prefix = "case_0000"
        edges = experiment._edges(archive, prefix)
        assert len(edges) == archive[f"{prefix}__edge_features"].shape[0]
        assert archive[f"{prefix}__focal_logits"].shape == (len(edges),)
        assert archive[f"{prefix}__focal_features"].shape == (len(edges), 6)
        for arm in experiment.STAGE_ARMS[1:]:
            layout = experiment._strict_layout(archive[f"{prefix}__{arm}_layout"])
            np.testing.assert_array_equal(np.sort(layout), np.arange(576))
