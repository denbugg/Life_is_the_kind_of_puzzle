from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import puzzle_denoise_v2.legacy_baseline as legacy_baseline
from puzzle_denoise_v2.legacy_baseline import (
    TileRestorer,
    load_legacy_tile_restorer,
    predict_legacy_tiles_uint8,
    sha256_file,
)


class HalfStepModel(torch.nn.Module):
    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return torch.full_like(tiles, 0.5 / 255.0)


def test_implicit_cuda_device_is_canonicalized_to_current_index(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert legacy_baseline._canonical_device("cuda") == torch.device("cuda:0")
    assert legacy_baseline._canonical_device("cuda:1") == torch.device("cuda:1")


def _legacy_checkpoint(model: TileRestorer, width: int, depth: int) -> dict:
    return {
        "model_state": model.state_dict(),
        "width": width,
        "depth": depth,
        "grid": 24,
        "tile": 20,
        "args": {"width": width, "depth": depth},
        "history": [],
    }


def _save_checkpoint(path: Path, checkpoint: dict) -> str:
    torch.save(checkpoint, path)
    return sha256_file(path)


def test_strict_loader_accepts_exact_schema_and_returns_provenance(tmp_path) -> None:
    model = TileRestorer(width=4, depth=1)
    checkpoint_path = tmp_path / "legacy.pt"
    expected_sha256 = _save_checkpoint(checkpoint_path, _legacy_checkpoint(model, 4, 1))

    loaded, device, metadata = load_legacy_tile_restorer(
        checkpoint_path,
        expected_sha256=expected_sha256,
        expected_width=4,
        expected_depth=1,
        device="cpu",
    )

    assert device == torch.device("cpu")
    assert metadata["checkpoint_sha256"] == expected_sha256
    assert metadata["state_entries"] == len(model.state_dict())
    assert metadata["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert list(loaded.state_dict()) == list(model.state_dict())
    assert all(
        torch.equal(loaded.state_dict()[name], tensor)
        for name, tensor in model.state_dict().items()
    )


def test_strict_loader_rejects_sha_metadata_and_state_schema_mismatches(tmp_path) -> None:
    model = TileRestorer(width=4, depth=1)
    checkpoint_path = tmp_path / "legacy.pt"
    expected_sha256 = _save_checkpoint(checkpoint_path, _legacy_checkpoint(model, 4, 1))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_legacy_tile_restorer(
            checkpoint_path,
            expected_sha256="0" * 64,
            expected_width=4,
            expected_depth=1,
        )

    wrong_width = _legacy_checkpoint(model, 5, 1)
    wrong_width_sha = _save_checkpoint(checkpoint_path, wrong_width)
    with pytest.raises(ValueError, match="'width' mismatch"):
        load_legacy_tile_restorer(
            checkpoint_path,
            expected_sha256=wrong_width_sha,
            expected_width=4,
            expected_depth=1,
        )

    wrong_state = _legacy_checkpoint(model, 4, 1)
    wrong_state["model_state"] = wrong_state["model_state"].copy()
    wrong_state["model_state"].pop("tail.bias")
    wrong_state_sha = _save_checkpoint(checkpoint_path, wrong_state)
    with pytest.raises(ValueError, match="state schema mismatch"):
        load_legacy_tile_restorer(
            checkpoint_path,
            expected_sha256=wrong_state_sha,
            expected_width=4,
            expected_depth=1,
        )


def test_uint8_prediction_matches_legacy_identity_and_validates_inputs() -> None:
    model = TileRestorer(width=4, depth=1)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.train()
    tiles = np.random.default_rng(20260710).integers(
        0, 256, size=(3, 20, 20, 3), dtype=np.uint8
    )

    restored = predict_legacy_tiles_uint8(model, tiles, "cpu", batch_size=2)

    assert np.array_equal(restored, tiles)
    assert model.training is False
    with pytest.raises(TypeError, match="uint8"):
        predict_legacy_tiles_uint8(model, tiles.astype(np.float32), "cpu")
    with pytest.raises(ValueError, match="Nx20x20x3"):
        predict_legacy_tiles_uint8(model, tiles[:, :-1], "cpu")
    with pytest.raises(ValueError, match="batch_size"):
        predict_legacy_tiles_uint8(model, tiles, "cpu", batch_size=0)


def test_uint8_prediction_uses_legacy_half_up_conversion() -> None:
    tiles = np.zeros((1, 20, 20, 3), dtype=np.uint8)
    restored = predict_legacy_tiles_uint8(HalfStepModel(), tiles, "cpu")
    assert np.all(restored == 1)
