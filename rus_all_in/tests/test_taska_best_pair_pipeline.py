from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aiijc_puzzle.taska_best_pair_pipeline import (
    CONFIRMATION_CONFIG_SHA256,
    CONFIRMATION_REPORT_SHA256,
    SELECTIVE_SOLVER_SHA256,
    TaskaBestPairPipelineResult,
    parse_args,
    solve_taska_best_pair_pipeline,
    verify_taska_best_pair_solver,
)
from aiijc_puzzle.taska_pair_pipeline import EXPECTED_ARTIFACT_SHA256

FIVE_COSTS = (
    ("raw", 5.0),
    ("logistic", 4.0),
    ("focal_top5", 3.0),
    ("nonlinear", 2.0),
    ("selective_vote500_focal", 1.0),
)


def test_confirmed_solver_source_is_byte_gated() -> None:
    assert verify_taska_best_pair_solver() == SELECTIVE_SOLVER_SHA256
    assert len(CONFIRMATION_CONFIG_SHA256) == 64
    assert len(CONFIRMATION_REPORT_SHA256) == 64


def test_result_is_strict_read_only_layout_and_has_receipt() -> None:
    result = TaskaBestPairPipelineResult(
        layout=np.arange(576, dtype=np.int64),
        selected_arm="selective_vote500_focal",
        costs=FIVE_COSTS,
        diagnostics={"accepted_new_edge_count": 7},
        artifact_sha256=EXPECTED_ARTIFACT_SHA256,
    )
    assert result.layout.dtype == np.int32
    assert not result.layout.flags.writeable
    assert result.as_dict()["confirmation_report_sha256"] == (CONFIRMATION_REPORT_SHA256)
    with pytest.raises(ValueError):
        result.layout[0] = 1


def test_adapter_returns_only_candidate_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    solved = SimpleNamespace(
        candidate_layout=np.arange(575, -1, -1, dtype=np.int32),
        candidate_choice="selective_vote500_focal",
        five_arm_costs=FIVE_COSTS,
        diagnostics=lambda: {"accepted_new_edge_count": 9},
    )
    calls: list[tuple[object, object, int]] = []

    def fake_solver(tiles: object, resources: object, *, focal_chunk_size: int) -> object:
        calls.append((tiles, resources, focal_chunk_size))
        return solved

    monkeypatch.setattr(
        "aiijc_puzzle.taska_best_pair_pipeline.selective_vote500.solve_selective_vote500",
        fake_solver,
    )
    resources = SimpleNamespace(artifact_sha256=EXPECTED_ARTIFACT_SHA256)
    tiles = np.zeros((576, 20, 20, 3), dtype=np.uint8)
    result = solve_taska_best_pair_pipeline(tiles, resources, focal_chunk_size=17)
    assert np.array_equal(result.layout, solved.candidate_layout)
    assert calls == [(tiles, resources, 17)]


def test_cli_is_layout_only_and_has_no_solver_tuning() -> None:
    args = parse_args(["tiles.npy", "--output-layout", "layout.npy"])
    assert str(args.tiles) == "tiles.npy"
    assert str(args.output_layout) == "layout.npy"
    for option in ("--vote-target", "--threshold", "--arm", "--tail-max-swaps"):
        with pytest.raises(SystemExit):
            parse_args(["tiles.npy", "--output-layout", "layout.npy", option, "1"])
