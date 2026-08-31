from __future__ import annotations

import numpy as np

import aiijc_puzzle.dualnaf_matcher as matcher
from aiijc_puzzle.protocol import TILE_COUNT, TILE_SIZE


def _tiles(value: int = 0) -> np.ndarray:
    return np.full((TILE_COUNT, TILE_SIZE, TILE_SIZE, 3), value, dtype=np.uint8)


def _score(offset: float) -> np.ndarray:
    rows = np.arange(TILE_COUNT, dtype=np.float32)[:, None]
    columns = np.arange(TILE_COUNT, dtype=np.float32)[None, :]
    return rows * 0.001 + columns * 0.002 + offset


def test_normalized_score_mean_is_exact_fixed_half_mix() -> None:
    first = _score(1.0)
    second = _score(5.0)
    fused = matcher.normalized_score_mean(first, second)
    np.testing.assert_allclose(fused, 0.5 * first + 0.5 * second, rtol=0, atol=1e-6)


def test_matcher_score_roster_is_preregistered_and_fuses_baseline_with_raw(
    monkeypatch,
) -> None:
    baseline = (_score(1.0), _score(2.0))
    dual_raw = (_score(3.0), _score(4.0))
    dual_bilateral = (_score(5.0), _score(6.0))
    calls: list[tuple[str, ...]] = []

    def fake_scores(tiles: np.ndarray, *, views: tuple[str, ...]):
        calls.append(views)
        if views == ("bilateral",):
            return {"bilateral": baseline}
        assert views == ("raw", "bilateral")
        return {"raw": dual_raw, "bilateral": dual_bilateral, "mean": dual_raw}

    monkeypatch.setattr(matcher, "directional_scores", fake_scores)
    roster = matcher.matcher_score_roster(_tiles(10), _tiles(20))
    assert tuple(roster) == matcher.MATCHER_ROSTER
    assert calls == [("bilateral",), ("raw", "bilateral")]
    np.testing.assert_allclose(roster[matcher.FUSION][0], 0.5 * (baseline[0] + dual_raw[0]))
    np.testing.assert_allclose(roster[matcher.FUSION][1], 0.5 * (baseline[1] + dual_raw[1]))


def test_matcher_rejects_non_uint8_or_wrong_roster() -> None:
    with np.testing.assert_raises(TypeError):
        matcher.matcher_score_roster(_tiles().astype(np.float32), _tiles())
    with np.testing.assert_raises(ValueError):
        matcher.solve_matcher_roster({}, edge_budget=96)
