from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import aiijc_puzzle.drunet_goal_cycle2 as cycle
from aiijc_puzzle.pretrained_drunet_protected_stack import (
    ARM_COMBINED as FROZEN_STACK_ARM,
)
from aiijc_puzzle.protocol import split_tiles


def test_t60_blend_uses_h28_at_content_edge_and_h50_in_flat_area() -> None:
    mask_source = np.full((480, 480, 3), 60, dtype=np.uint8)
    mask_source[:, 240:] = 200
    safe = np.full_like(mask_source, 20)
    flat = np.full_like(mask_source, 180)
    output, diagnostics = cycle.blend_h28_safe_h50_flat_t60(
        mask_source,
        safe,
        flat,
    )
    assert output.shape == mask_source.shape
    assert output.dtype == np.uint8
    assert output[100, 239, 0] <= 22
    assert output[50, 50, 0] == 180
    assert diagnostics["sobel_threshold"] == 60.0
    assert 0.0 < diagnostics["binary_dilated_protected_fraction"] < 1.0


def test_tile_flatness_counts_spatially_constant_and_near_flat_tiles() -> None:
    board = np.full((480, 480, 3), 80, dtype=np.uint8)
    counts = cycle.tile_flatness_counts(board)
    assert counts == {
        "exact_spatially_constant_rgb_tiles": 576,
        "near_flat_tiles_global_std_lt_2": 576,
        "near_flat_tiles_global_std_lt_4": 576,
    }
    board[:20, :20] = np.arange(20, dtype=np.uint8)[None, :, None]
    changed = cycle.tile_flatness_counts(board)
    assert changed["exact_spatially_constant_rgb_tiles"] == 575
    assert changed["near_flat_tiles_global_std_lt_2"] == 575


@dataclass
class DummyDiagnostics:
    sigma: float

    def as_dict(self) -> dict[str, float]:
        return {"sigma": self.sigma}


def test_fixed_flow_uses_two_independent_tilewise_drunet_calls(monkeypatch) -> None:
    tiles = np.full((576, 20, 20, 3), 60, dtype=np.uint8)
    current_d = np.full((480, 480, 3), 110, dtype=np.uint8)
    calls: list[tuple[float, np.ndarray]] = []

    def fake_frozen(*args, **kwargs):
        del args, kwargs
        return {
            FROZEN_STACK_ARM: current_d,
        }, {"frozen": True}

    def fake_render(model, value, *, sigma_255, device, batch_size):
        del model, device
        assert batch_size == 144
        calls.append((float(sigma_255), value.copy()))
        output = np.full_like(value, 120 if sigma_255 == 50 else 80)
        return output, DummyDiagnostics(float(sigma_255))

    def fake_nlm(image: np.ndarray, h: int) -> np.ndarray:
        return np.full_like(image, int(image[0, 0, 0]) + h // 10)

    monkeypatch.setattr(cycle, "render_combined_arms", fake_frozen)
    monkeypatch.setattr(cycle, "render_drunet_tiles", fake_render)
    monkeypatch.setattr(cycle, "colored_nlm", fake_nlm)
    predictions, diagnostics = cycle.render_goal_cycle2_arms(
        torch.nn.Identity(),
        tiles,
        device=torch.device("cpu"),
    )
    assert tuple(predictions) == cycle.ARM_NAMES
    assert [item[0] for item in calls] == [50.0, 30.0]
    np.testing.assert_array_equal(calls[0][1], tiles)
    direct_h28 = np.full((480, 480, 3), 122, dtype=np.uint8)
    np.testing.assert_array_equal(calls[1][1], split_tiles(direct_h28))
    np.testing.assert_array_equal(predictions[cycle.REFERENCE_CURRENT_D], current_d)
    assert np.all(predictions[cycle.CANDIDATE_POST_H28] == 101)
    expected_combo = (
        predictions[cycle.CANDIDATE_SIGMA50].astype(np.uint16)
        + predictions[cycle.CANDIDATE_POST_H28].astype(np.uint16)
        + 1
    ) // 2
    np.testing.assert_array_equal(
        predictions[cycle.CANDIDATE_COMBINATION],
        expected_combo.astype(np.uint8),
    )
    assert diagnostics["drunet50"] == {"sigma": 50.0}
    assert diagnostics["post_h28_drunet30"] == {"sigma": 30.0}


def test_renderer_rejects_any_roster_other_than_all_576_tiles() -> None:
    with pytest.raises(ValueError, match="576"):
        cycle.render_goal_cycle2_arms(
            torch.nn.Identity(),
            np.zeros((575, 20, 20, 3), dtype=np.uint8),
            device=torch.device("cpu"),
        )
