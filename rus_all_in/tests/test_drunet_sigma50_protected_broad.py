from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import aiijc_puzzle.drunet_sigma50_protected_broad as broad


@dataclass
class DummyDiagnostics:
    def as_dict(self) -> dict[str, int]:
        return {"tile_count": 576}


def test_renderer_is_one_tilewise_sigma50_call_and_three_independent_nlm_calls(
    monkeypatch,
) -> None:
    tiles = np.full((576, 20, 20, 3), 80, dtype=np.uint8)
    render_calls: list[tuple[float, int, np.ndarray]] = []
    nlm_calls: list[int] = []

    def fake_render(model, value, *, sigma_255, device, batch_size):
        del model, device
        render_calls.append((float(sigma_255), int(batch_size), value.copy()))
        return np.full_like(value, 100), DummyDiagnostics()

    def fake_nlm(image: np.ndarray, h: int) -> np.ndarray:
        nlm_calls.append(h)
        output = np.full_like(image, 100 + h // 2)
        output[:, 240:] += 30
        return output

    monkeypatch.setattr(broad, "render_drunet_tiles", fake_render)
    monkeypatch.setattr(broad, "colored_nlm", fake_nlm)
    reference, candidate, diagnostics = broad.render_sigma50_protected(
        torch.nn.Identity(),
        tiles,
        device=torch.device("cpu"),
    )
    assert len(render_calls) == 1
    assert render_calls[0][:2] == (50.0, 144)
    np.testing.assert_array_equal(render_calls[0][2], tiles)
    assert nlm_calls == [20, 28, 50]
    assert reference.shape == candidate.shape == (480, 480, 3)
    assert reference.dtype == candidate.dtype == np.uint8
    assert not np.array_equal(reference, candidate)
    assert diagnostics["drunet"] == {"tile_count": 576}
    assert set(diagnostics["structure"]) == {
        broad.REFERENCE_DRUNET50_H28,
        broad.CANDIDATE_DRUNET50_PROTECTED,
    }


def test_renderer_rejects_incomplete_tile_roster() -> None:
    with pytest.raises(ValueError, match="576"):
        broad.render_sigma50_protected(
            torch.nn.Identity(),
            np.zeros((575, 20, 20, 3), dtype=np.uint8),
            device=torch.device("cpu"),
        )
