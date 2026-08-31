from __future__ import annotations

import inspect

import numpy as np
import pytest

import aiijc_puzzle.global_population_layout as global_layout
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_COUNT


def test_population_hungarian_recovers_known_global_permutation() -> None:
    rng = np.random.default_rng(20260830)
    tile_at_slot = rng.permutation(TILE_COUNT).astype(np.int32)
    scores = np.full((TILE_COUNT, TILE_COUNT), -2.0, dtype=np.float32)
    scores[tile_at_slot, np.arange(TILE_COUNT)] = 3.0
    solved = global_layout.solve_population_hungarian(scores)
    assert np.array_equal(solved.layout, tile_at_slot)
    assert solved.objective == pytest.approx(3.0 * TILE_COUNT)
    assert solved.solver == "population_hungarian_train5600"


def test_population_hungarian_rejects_invalid_scores() -> None:
    with pytest.raises(ValueError, match="expected"):
        global_layout.solve_population_hungarian(np.zeros((2, 2), dtype=np.float32))
    scores = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    scores[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        global_layout.solve_population_hungarian(scores)


def test_roster_is_preregistered_and_predictor_cannot_accept_target() -> None:
    assert global_layout.FROZEN_ARMS == (
        "no_atlas_buddies96",
        "population_hungarian",
        "population_w0p25_buddies96",
        "population_w1p0_buddies96",
    )
    assert global_layout.STRONG_POPULATION_WEIGHTS == (0.25, 1.0)
    assert global_layout.NLM_H == 20
    assert "target" not in inspect.signature(global_layout.predict_frozen_roster).parameters


def test_frozen_tail_order_and_exactly_one_nlm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    calls: list[str] = []

    def fake_rgb_offsets(tiles: np.ndarray, _config: object) -> tuple[np.ndarray, dict]:
        calls.append("estimate_rgb")
        return np.zeros((TILE_COUNT, 3)), {"stage": "rgb"}

    def fake_apply_rgb(tiles: np.ndarray, _offsets: np.ndarray) -> np.ndarray:
        calls.append("apply_rgb")
        return tiles.copy()

    def fake_luma(tiles: np.ndarray, _config: object) -> tuple[np.ndarray, dict]:
        calls.append("estimate_luma")
        return np.ones(TILE_COUNT), {"stage": "luma"}

    def fake_apply_luma(tiles: np.ndarray, _gains: np.ndarray) -> np.ndarray:
        calls.append("apply_luma")
        return tiles.copy()

    def fake_nlm(image: np.ndarray, h: int) -> np.ndarray:
        calls.append(f"nlm_h{h}")
        return image.copy()

    monkeypatch.setattr(global_layout, "seam_graph_rgb_offsets", fake_rgb_offsets)
    monkeypatch.setattr(global_layout, "apply_rgb_offsets", fake_apply_rgb)
    monkeypatch.setattr(global_layout, "seam_graph_luminance_gains", fake_luma)
    monkeypatch.setattr(global_layout, "apply_luminance_gains", fake_apply_luma)
    monkeypatch.setattr(global_layout, "nlm_color", fake_nlm)

    restored, diagnostics = global_layout.restore_frozen_tail(raw)
    assert np.array_equal(restored, raw)
    assert calls == [
        "estimate_rgb",
        "apply_rgb",
        "estimate_luma",
        "apply_luma",
        "nlm_h20",
    ]
    assert diagnostics == {"rgb": {"stage": "rgb"}, "luminance": {"stage": "luma"}}
