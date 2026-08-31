from __future__ import annotations

import copy
import json
import stat
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import aiijc_puzzle.frozen_final_evaluator as evaluator
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_COUNT


def _score_rows(count: int, *, raw: float, control: float, final: float) -> list[dict]:
    return [
        {
            "filename": f"board-{index}.png",
            "ssim": {
                evaluator.RAW_ARM: raw,
                evaluator.CONTROL_ARM: control,
                evaluator.FINAL_ARM: final,
            },
        }
        for index in range(count)
    ]


def test_v1_context_has_exact_preregistered_panels() -> None:
    calibration = evaluator.load_context("calibration")
    holdout = evaluator.load_context("holdout")

    assert len(calibration.records) == 48
    assert calibration.selection_digest == (
        "5b1a8dcd358c87191d1c0ced0253ec66f45566568e7126c76259ff13f9289bbf"
    )
    assert len(holdout.records) == 96
    assert holdout.selection_digest == (
        "989d5cc1d4428a48304d1ce4a37046bc43e258b174ffbc58913e3ae85f328c4f"
    )


def test_config_accepts_fresh_panel_but_rejects_pipeline_drift() -> None:
    context = evaluator.load_context("calibration")
    fresh = copy.deepcopy(context.config)
    fresh["single_use_holdout"].update(
        {
            "offset": 96,
            "count": 96,
            "filenames_sha256": (
                "a8d840c30a15419852bbd748b06d3985b390d069cbed8ed39964ac6f4cc8c175"
            ),
        }
    )
    evaluator.validate_frozen_config(fresh)

    fresh["pipeline"]["restoration"]["passes"] = 2
    with pytest.raises(ValueError, match="restoration semantics drifted"):
        evaluator.validate_frozen_config(fresh)


def test_fallback_config_resolves_reserved_fresh_holdout() -> None:
    path = Path("configs/frozen_submission_h20x1_fallback_v1.json")
    calibration = evaluator.load_context("calibration", config_path=path)
    holdout = evaluator.load_context("holdout", config_path=path)

    assert len(calibration.records) == 48
    assert len(holdout.records) == 96
    assert holdout.selection_digest == (
        "a8d840c30a15419852bbd748b06d3985b390d069cbed8ed39964ac6f4cc8c175"
    )
    assert evaluator.artifact_paths(calibration.config_sha256).root.name == (
        "7609987c9d9b817c48cc893d58f2a77fc37b8c1a2911574bed0013e01e38a042"
    )


def test_immutable_receipt_is_exclusive_and_read_only(tmp_path) -> None:
    receipt = tmp_path / "opened.json"
    digest = evaluator.create_immutable_receipt(receipt, {"schema": "test", "opened": True})

    assert len(digest) == 64
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o444
    with pytest.raises(RuntimeError, match="already opened"):
        evaluator.create_immutable_receipt(receipt, {"schema": "second"})


def test_paired_gates_are_recomputed_from_rows() -> None:
    context = evaluator.load_context("calibration")
    rows = _score_rows(48, raw=0.11, control=0.247, final=0.258)
    summary, comparisons = evaluator.aggregate_scores(rows)
    gate = evaluator.evaluate_gate("calibration", context.config, summary, comparisons)

    assert gate["checks"]["final_mean_ssim_min"]["passed"] is False
    assert gate["checks"]["gain_ci95_lower_min"]["passed"] is True
    assert gate["checks"]["wins_min"]["passed"] is True
    assert gate["all_passed"] is False


def test_run_freezes_every_prediction_before_target_scoring(monkeypatch, tmp_path) -> None:
    full_context = evaluator.load_context("calibration")
    records = tuple(full_context.records[:2])
    context = replace(
        full_context,
        records=records,
        selection_digest=evaluator.names_digest(records),
    )
    events: list[str] = []
    layout = np.arange(TILE_COUNT, dtype=np.int32)

    def fake_freeze(*args, **kwargs):
        events.append("freeze_all")
        boards = []
        for index, record in enumerate(records):
            predictions = {
                arm: np.full((IMAGE_SIZE, IMAGE_SIZE, 3), index + arm_index, dtype=np.uint8)
                for arm_index, arm in enumerate(evaluator.ARMS)
            }
            boards.append(
                evaluator.FrozenBoard(
                    record=record,
                    layout=layout,
                    audit={"passed": True},
                    predictions=predictions,
                    prediction_sha256={
                        arm: evaluator.array_digest(image) for arm, image in predictions.items()
                    },
                    layout_sha256="layout",
                    objective=0.0,
                    solver="fake",
                    harmonizer_diagnostics={},
                    runtime_seconds=0.0,
                )
            )
        return tuple(boards)

    def fake_score(*args, **kwargs):
        assert events == ["freeze_all"]
        events.append("score_targets")
        result = []
        for record in records:
            result.append(
                {
                    "filename": record["filename"],
                    "input_sha256": record["input_sha256"],
                    "target_sha256": record["target_sha256"],
                    "all_predictions_frozen_before_any_target_decode": True,
                    "tile_at_position": layout.tolist(),
                    "layout_sha256": "layout",
                    "permutation_audit": {"passed": True},
                    "prediction_sha256": {arm: "0" * 64 for arm in evaluator.ARMS},
                    "solver": "fake",
                    "objective": 0.0,
                    "harmonizer_diagnostics": {},
                    "ssim": {
                        evaluator.RAW_ARM: 0.11,
                        evaluator.CONTROL_ARM: 0.247,
                        evaluator.FINAL_ARM: 0.258,
                    },
                }
            )
        return result

    monkeypatch.setattr(evaluator, "freeze_all_predictions", fake_freeze)
    monkeypatch.setattr(evaluator, "score_frozen_predictions", fake_score)
    report_path = tmp_path / "report.json"
    commitment_path = tmp_path / "commitment.json"
    report = evaluator.run_evaluation(
        context,
        report_path=report_path,
        commitment_path=commitment_path,
    )

    assert events == ["freeze_all", "score_targets"]
    assert report["arms"] == list(evaluator.ARMS)
    assert report["preregistered_gate"]["all_passed"] is False
    assert json.loads(commitment_path.read_text())["contract"]["target_paths_opened"] is False


def test_holdout_needs_explicit_authorization_before_any_work(monkeypatch) -> None:
    context = evaluator.load_context("holdout")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("inference must not start")

    monkeypatch.setattr(evaluator, "freeze_all_predictions", forbidden)
    with pytest.raises(RuntimeError, match="explicit --allow-holdout"):
        evaluator.run_evaluation(context)
    assert called is False
