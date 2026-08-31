from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aiijc_puzzle.taska_pair_pipeline import EXPECTED_ARTIFACT_SHA256
from aiijc_puzzle.taska_relation_selector_pipeline import (
    CONFIRMATION_CONFIG_SHA256,
    CONFIRMATION_REPORT_SHA256,
    DEVELOPMENT_CONFIG_SHA256,
    DEVELOPMENT_REPORT_SHA256,
    MODEL_SHA256,
    TaskaRelationSelectorPipelineResult,
    load_taska_relation_selector_resources,
    parse_args,
    solve_taska_relation_selector_pipeline,
    verify_taska_relation_selector_solver,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_taska_protected_tail_fresh32_confirmation as synthetic  # noqa: E402

CONFIRMATION_ROOT = (
    PROJECT_ROOT / "outputs/taska-relation-truth-selector/formal-confirmation-v1"
)
SIX_SCORES = (
    ("raw", 1.0),
    ("logistic", 2.0),
    ("focal_top5", 3.0),
    ("nonlinear", 4.0),
    ("selective_vote500_focal", 5.0),
    ("combined_union_focal", 6.0),
)


def test_confirmed_parent_model_configs_reports_and_sources_are_byte_gated() -> None:
    verified = dict(verify_taska_relation_selector_solver())
    assert verified["development_config"] == DEVELOPMENT_CONFIG_SHA256
    assert verified["development_report"] == DEVELOPMENT_REPORT_SHA256
    assert verified["relation_model"] == MODEL_SHA256
    assert verified["confirmation_config"] == CONFIRMATION_CONFIG_SHA256
    assert verified["confirmation_report"] == CONFIRMATION_REPORT_SHA256
    assert {
        "parent_selective_solver",
        "parent_fusion_solver",
        "parent_fullres_supply_solver",
        "parent_raw_solver",
        "parent_parent_report",
        "parent_confirmation_config",
        "parent_confirmation_report",
        "parent_fullres_denoiser",
        "parent_pipeline",
        "relation_selector",
        "six_arm_preparer",
        "focal_tail",
        "layout_portfolio",
        "development_config",
        "development_report",
        "relation_model",
        "relation_model_freeze",
        "confirmation_config",
        "confirmation_report",
        "confirmation_archive",
        "confirmation_metadata",
        "confirmation_pre_score_freeze",
    } == set(verified)


def test_result_is_strict_read_only_layout_with_legal_receipt() -> None:
    result = TaskaRelationSelectorPipelineResult(
        layout=np.arange(576, dtype=np.int64),
        selected_arm="combined_union_focal",
        control_arm="nonlinear",
        expected_correct_scores=SIX_SCORES,
        parent_costs=tuple((name, -score) for name, score in SIX_SCORES),
        diagnostics={"relation_rows_per_arm": 1_104},
        pair_artifact_sha256=EXPECTED_ARTIFACT_SHA256,
        confirmed_sha256=verify_taska_relation_selector_solver(),
    )
    receipt = result.as_dict()
    assert result.layout.dtype == np.int32
    assert not result.layout.flags.writeable
    assert receipt["layout_only"] is True
    assert receipt["original_upright_tile_permutation"] is True
    assert receipt["restored_pixels_matcher_only"] is True
    assert receipt["denoised_output_pixels"] is False
    assert receipt["relation_model_sha256"] == MODEL_SHA256
    with pytest.raises(ValueError):
        result.layout[0] = 1


def test_cli_is_layout_only_and_exposes_no_model_or_solver_tuning() -> None:
    args = parse_args(["tiles.npy", "--output-layout", "layout.npy"])
    assert str(args.output_layout) == "layout.npy"
    for option in (
        "--model",
        "--threshold",
        "--top-k",
        "--arm",
        "--tail-max-swaps",
        "--output-image",
    ):
        with pytest.raises(SystemExit):
            parse_args(["tiles.npy", "--output-layout", "layout.npy", option, "1"])


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="frozen replay was created by the preregistered MPS confirmation",
)
def test_frozen_formal_case_zero_end_to_end_bitwise_replay() -> None:
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

    resources = load_taska_relation_selector_resources(device="mps")
    result = solve_taska_relation_selector_pipeline(dirty.dirty_tiles, resources)
    with np.load(
        CONFIRMATION_ROOT / "frozen-target-free-eval.npz", allow_pickle=False
    ) as archive:
        expected = archive["case_000__relation_truth_selector_layout"]
        expected_scores = archive["case_000__relation_expected_correct_scores"]
    assert np.array_equal(result.layout, expected)
    assert result.selected_arm == frozen["choice"]
    assert result.control_arm == frozen["control_choice"]
    assert np.array_equal(
        np.asarray([score for _, score in result.expected_correct_scores]),
        expected_scores,
    )
    assert result.diagnostics["parent"][
        "mechanical_selective_control_replay_matches"
    ] is True
