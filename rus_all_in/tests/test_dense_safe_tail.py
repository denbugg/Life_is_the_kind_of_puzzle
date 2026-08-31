from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aiijc_puzzle import dense_safe_tail as dense


def test_preregistered_ranked_rosters_and_exposure_are_exact() -> None:
    config = dense.load_config()
    primary = dense.panel_records("primary")
    confirmation = dense.panel_records("confirmation")

    assert dense.sha256_file(dense.CONFIG_PATH) == dense.CONFIG_SHA256
    assert len(primary) == len(confirmation) == 36
    assert primary[0]["filename"] == "img_003858.png"
    assert primary[-1]["filename"] == "img_002358.png"
    assert confirmation[0]["filename"] == "img_005968.png"
    assert confirmation[-1]["filename"] == "img_001556.png"
    assert set(row["filename"] for row in primary).isdisjoint(
        row["filename"] for row in confirmation
    )
    assert (
        dense.filename_nul_digest(primary)
        == config["protocol"]["primary"]["filename_nul_join_sha256"]
    )
    assert (
        dense.filename_nul_digest(confirmation)
        == config["protocol"]["confirmation_if_and_only_if_primary_passes"][
            "filename_nul_join_sha256"
        ]
    )

    exposure = dense.audit_historical_exposure()
    assert exposure["freshness_claim"] is False
    assert exposure["legacy_exact_matches_to_current_calibration"] == 700
    assert exposure["primary_historically_exposed"] == 36
    assert exposure["confirmation_historically_exposed"] == 36


def test_prepare_has_no_target_directory_argument() -> None:
    assert "targets_dir" not in inspect.signature(dense.prepare_panel).parameters


def test_generate_arms_uses_independent_h_values_and_exact_blends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_nlm(image: np.ndarray, h: int) -> SimpleNamespace:
        calls.append(h)
        return SimpleNamespace(image=np.full_like(image, h), seconds=float(h) / 1000)

    monkeypatch.setattr(dense, "apply_nlm_color", fake_nlm)
    image = np.zeros((dense.IMAGE_SIZE, dense.IMAGE_SIZE, 3), dtype=np.uint8)
    outputs, seconds = dense.generate_arms(image)

    assert calls == list(range(20, 30))
    assert tuple(outputs) == dense.ARMS
    assert set(seconds) == set(dense.ARMS)
    assert np.all(outputs["blend_h20_75_h28_25"] == 22)
    assert np.all(outputs["blend_h20_50_h28_50"] == 24)


def test_safety_metrics_separate_tile_interior_and_grid() -> None:
    x = np.arange(dense.IMAGE_SIZE, dtype=np.uint16)
    luminance = ((x % dense.TILE_SIZE) * 5)[None, :]
    image = np.repeat(luminance, dense.IMAGE_SIZE, axis=0)
    rgb = np.repeat(image[..., None], 3, axis=2).astype(np.uint8)
    metrics = dense.safety_metrics(rgb)

    assert metrics["within_tile_gradient"] > 0
    assert metrics["laplacian_energy"] > 0
    assert metrics["grid_ratio"] > 1


def test_paired_t_interval_and_gate_selection() -> None:
    low, high = dense.paired_t_interval([0.01] * 36)
    assert low == pytest.approx(0.01)
    assert high == pytest.approx(0.01)

    score_summary: dict[str, dict[str, object]] = {}
    safety_summary: dict[str, dict[str, object]] = {}
    for index, arm in enumerate(dense.ARMS):
        score_summary[arm] = {
            "mean_ssim": 0.25 if arm == dense.BASELINE_ARM else 0.271 + index * 1e-5,
            "paired_gain_vs_h20_ci95": [0.01, 0.02],
            "wins_ties_losses_vs_h20": [36, 0, 0],
        }
        safety_summary[arm] = {
            "within_tile_gradient_retention_vs_h20": {
                "mean": 0.9,
                "min": 0.8,
                "max": 0.95,
            },
            "laplacian_retention_vs_h20": {"mean": 0.85, "min": 0.75, "max": 0.9},
            "grid_ratio_relative_to_h20": {"mean": 1.01, "min": 0.99, "max": 1.05},
            "all_predictions_distinct_across_boards": True,
        }
    result = dense.evaluate_primary_gates(score_summary, safety_summary)
    assert result["all_passed"] is True
    assert result["winner"] == dense.ARMS[-1]


def test_confirmation_fails_closed_without_passing_primary(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="primary report"):
        dense._require_confirmation_authorized(tmp_path)

    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "report.json").write_text(
        json.dumps({"promotion": {"all_passed": False}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="promotion gate failed"):
        dense._require_confirmation_authorized(tmp_path)
