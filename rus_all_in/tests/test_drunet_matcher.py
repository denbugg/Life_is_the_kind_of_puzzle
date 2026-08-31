from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import aiijc_puzzle.drunet_matcher as matcher
from aiijc_puzzle.protocol import TILE_COUNT, TILE_SIZE

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts/run_drunet_matcher_train_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("drunet_matcher_train_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _tiles(value: int) -> np.ndarray:
    return np.full((TILE_COUNT, TILE_SIZE, TILE_SIZE, 3), value, dtype=np.uint8)


def _score(offset: float) -> np.ndarray:
    rows = np.arange(TILE_COUNT, dtype=np.float32)[:, None]
    columns = np.arange(TILE_COUNT, dtype=np.float32)[None, :]
    return offset + rows * 0.001 + columns * 0.002


@pytest.mark.parametrize("weight", matcher.FUSION_WEIGHTS)
def test_normalized_score_fusion_uses_exact_global_weight(weight: float) -> None:
    dirty = _score(1.0)
    drunet = _score(5.0)
    fused = matcher.normalized_score_fusion(dirty, drunet, drunet_weight=weight)
    np.testing.assert_allclose(
        fused,
        (1.0 - weight) * dirty + weight * drunet,
        rtol=0,
        atol=1e-6,
    )


def test_matcher_roster_uses_bilateral_for_dirty_and_drunet(monkeypatch) -> None:
    dirty_scores = (_score(1.0), _score(2.0))
    drunet_scores = (_score(3.0), _score(4.0))
    calls: list[int] = []

    def fake_scores(tiles: np.ndarray, *, views: tuple[str, ...]):
        assert views == ("bilateral",)
        calls.append(int(tiles[0, 0, 0, 0]))
        return {"bilateral": dirty_scores if calls[-1] == 10 else drunet_scores}

    monkeypatch.setattr(matcher, "directional_scores", fake_scores)
    roster = matcher.matcher_score_roster(_tiles(10), _tiles(20))
    assert tuple(roster) == matcher.ARM_NAMES
    assert calls == [10, 20]
    for weight, name in zip(matcher.FUSION_WEIGHTS, matcher.FUSION_NAMES, strict=True):
        np.testing.assert_allclose(
            roster[name][0],
            (1.0 - weight) * dirty_scores[0] + weight * drunet_scores[0],
            rtol=0,
            atol=1e-6,
        )


def test_train_config_hash_and_exact_selection_verification_rosters() -> None:
    config, records = runner.load_contract(runner.MANIFEST)
    assert runner.sha256_file(runner.CONFIG) == runner.CONFIG_SHA256
    assert len(records) == 16
    assert records[0]["filename"] == "img_005961.png"
    assert records[-1]["filename"] == "img_001637.png"
    assert runner.names_digest(records) == config["protocol"]["filenames_newline_sha256"]
    assert runner.roster_digest(records) == config["protocol"]["filename_input_roster_sha256"]
    assert (
        runner.names_digest(records[:8]) == config["protocol"]["selection_first_8_filenames_sha256"]
    )
    assert (
        runner.names_digest(records[8:])
        == config["protocol"]["verification_last_8_filenames_sha256"]
    )


def test_selection_excludes_pure_drunet_and_uses_frozen_tie_breaks() -> None:
    summary = {
        name: {
            "F_ssim": 0.20,
            "h28_ssim": 0.19,
            "adjacency": 0.10,
            "translation_aligned_placement": 0.05,
        }
        for name in matcher.FUSION_NAMES
    }
    summary[matcher.FUSION_NAMES[1]]["F_ssim"] = 0.21
    summary[matcher.PURE_DRUNET] = {
        "F_ssim": 0.99,
        "h28_ssim": 0.99,
        "adjacency": 0.99,
        "translation_aligned_placement": 0.99,
    }
    assert runner.select_fusion(summary) == matcher.FUSION_NAMES[1]


def test_verification_gate_names_match_immutable_config() -> None:
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    selected = matcher.FUSION_NAMES[0]
    observed = {
        "F_ssim_gain": 0.01,
        "F_ssim_gain_ci95": {"lower": 0.005},
        "F_ssim_wins_ties_losses": [8, 0, 0],
        "h28_ssim_gain": 0.01,
        "h28_ssim_wins_ties_losses": [8, 0, 0],
        "adjacency_gain": 0.01,
        "adjacency_wins_ties_losses": [8, 0, 0],
        "translation_aligned_placement_gain": 0.01,
        "direct_placement_gain": 0.0,
    }
    result = runner.verification_gate(
        selected,
        {selected: observed},
        config,
        all_audits_passed=True,
    )
    assert result["passed"] is True
    assert set(result["checks"]) == set(config["verification_gate_last_8"])
