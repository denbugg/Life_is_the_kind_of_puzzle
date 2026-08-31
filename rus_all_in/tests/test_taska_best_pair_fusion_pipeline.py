from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aiijc_puzzle.taska_best_pair_fusion_pipeline import (
    CONFIRMATION_CONFIG_SHA256,
    CONFIRMATION_REPORT_SHA256,
    TaskaBestPairFusionPipelineResult,
    load_taska_best_pair_fusion_resources,
    parse_args,
    solve_taska_best_pair_fusion_pipeline,
    verify_taska_best_pair_fusion_solver,
)
from aiijc_puzzle.taska_pair_pipeline import EXPECTED_ARTIFACT_SHA256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_taska_protected_tail_fresh32_confirmation as synthetic  # noqa: E402

CONFIRMATION_ROOT = (
    PROJECT_ROOT
    / "outputs/taska-selective-fullres-union-fusion/"
    "fresh32-formal-confirmation-v1"
)
SIX_COSTS = (
    ("raw", 6.0),
    ("logistic", 5.0),
    ("focal_top5", 4.0),
    ("nonlinear", 3.0),
    ("selective_vote500_focal", 2.0),
    ("combined_union_focal", 1.0),
)


def test_confirmed_sources_models_and_reports_are_byte_gated() -> None:
    verified = dict(verify_taska_best_pair_fusion_solver())
    assert verified["confirmation_config"] == CONFIRMATION_CONFIG_SHA256
    assert verified["confirmation_report"] == CONFIRMATION_REPORT_SHA256
    assert verified["raw_solver"] == EXPECTED_ARTIFACT_SHA256[-1][1]
    assert set(verified) == {
        "selective_solver",
        "fusion_solver",
        "fullres_supply_solver",
        "raw_solver",
        "parent_report",
        "confirmation_config",
        "confirmation_report",
        "fullres_denoiser",
    }


def test_result_is_strict_read_only_layout_with_fusion_receipt() -> None:
    verified = verify_taska_best_pair_fusion_solver()
    result = TaskaBestPairFusionPipelineResult(
        layout=np.arange(576, dtype=np.int64),
        selected_arm="combined_union_focal",
        costs=SIX_COSTS,
        diagnostics={"unique_fullres_accepted_count": 7},
        pair_artifact_sha256=EXPECTED_ARTIFACT_SHA256,
        confirmed_sha256=verified,
    )
    assert result.layout.dtype == np.int32
    assert not result.layout.flags.writeable
    assert result.as_dict()["restored_pixels_matcher_only"] is True
    assert result.as_dict()["confirmation_report_sha256"] == (
        CONFIRMATION_REPORT_SHA256
    )
    with pytest.raises(ValueError):
        result.layout[0] = 1


def test_cli_is_layout_only_and_exposes_no_solver_tuning() -> None:
    args = parse_args(["tiles.npy", "--output-layout", "layout.npy"])
    assert str(args.output_layout) == "layout.npy"
    for option in (
        "--vote-target",
        "--threshold",
        "--support",
        "--arm",
        "--tail-max-swaps",
    ):
        with pytest.raises(SystemExit):
            parse_args(
                ["tiles.npy", "--output-layout", "layout.npy", option, "1"]
            )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="frozen replay was created by the preregistered MPS confirmation",
)
def test_frozen_case_zero_end_to_end_replay() -> None:
    metadata = json.loads(
        (CONFIRMATION_ROOT / "frozen-target-free-eval.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = metadata["rows"][0]
    manifest = json.loads(
        (PROJECT_ROOT / "data/interim/validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    records = {
        str(row["filename"]): row
        for rows in manifest["splits"].values()
        for row in rows
    }
    source = str(frozen["source_filename"])
    draw = int(frozen["draw_index"])
    cache = synthetic.CleanTileCache(
        PROJECT_ROOT / "data/raw/train/targets", maximum_boards=1
    )
    dirty = synthetic._dirty_case(cache, records[source], source, draw)
    assert synthetic._dirty_sha256(dirty.dirty_tiles) == frozen["dirty_sha256"]

    resources = load_taska_best_pair_fusion_resources(device="mps")
    result = solve_taska_best_pair_fusion_pipeline(dirty.dirty_tiles, resources)
    with np.load(
        CONFIRMATION_ROOT / "frozen-target-free-eval.npz", allow_pickle=False
    ) as archive:
        expected = archive[
            "case_000__selective_unique_fullres_fusion_focal_gated_tail96_layout"
        ]
    assert np.array_equal(result.layout, expected)
    assert result.selected_arm == frozen["choice"]
    assert result.diagnostics["mechanical_selective_control_replay_matches"] is True
