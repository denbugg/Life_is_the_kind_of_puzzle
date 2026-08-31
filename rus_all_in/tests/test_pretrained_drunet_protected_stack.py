from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import aiijc_puzzle.pretrained_drunet_protected_stack as stack
from aiijc_puzzle.edge_protected_nlm import protected_masks


def test_t40_blend_uses_h20_mask_but_h28_and_h40_pixels() -> None:
    mask_source = np.full((480, 480, 3), 80, dtype=np.uint8)
    mask_source[:, 240:] = 180
    safe = np.full_like(mask_source, 20)
    flat = np.full_like(mask_source, 200)
    output, diagnostics = stack.blend_v1_t40_from_h20_mask(mask_source, safe, flat)
    binary, soft, fraction = protected_masks(mask_source, sobel_threshold=40.0)
    expected = np.rint(
        soft[..., None] * safe.astype(np.float32)
        + (1.0 - soft[..., None]) * flat.astype(np.float32)
    ).astype(np.uint8)
    np.testing.assert_array_equal(output, expected)
    assert diagnostics["binary_mask_sha256"] == stack.array_digest(binary)
    assert diagnostics["binary_dilated_protected_fraction"] == fraction
    assert output[100, 239, 0] <= 22
    assert output[50, 50, 0] == 200


@dataclass
class DummyDiagnostics:
    def as_dict(self) -> dict[str, int]:
        return {"dummy": 1}


def test_fixed_arm_flow_has_one_neural_canvas_and_independent_nlm(
    monkeypatch,
) -> None:
    tiles = np.full((576, 20, 20, 3), 60, dtype=np.uint8)
    calls: list[int] = []

    def fake_render(*args, **kwargs):
        del args, kwargs
        return np.full_like(tiles, 70), DummyDiagnostics()

    def fake_nlm(image: np.ndarray, h: int) -> np.ndarray:
        calls.append(h)
        return np.full_like(image, int(image[0, 0, 0]) + h // 10)

    monkeypatch.setattr(stack, "render_drunet_tiles", fake_render)
    monkeypatch.setattr(stack, "colored_nlm", fake_nlm)
    predictions, diagnostics = stack.render_combined_arms(
        torch.nn.Identity(),
        tiles,
        device=torch.device("cpu"),
    )
    assert tuple(predictions) == stack.ARM_NAMES
    assert calls == [20, 28, 20, 28, 40]
    assert np.all(predictions[stack.ARM_ORIGINAL_H20] == 62)
    assert np.all(predictions[stack.ARM_ORIGINAL_H28] == 62)
    assert np.all(predictions[stack.ARM_DRUNET_H28] == 72)
    assert diagnostics["drunet"] == {"dummy": 1}
