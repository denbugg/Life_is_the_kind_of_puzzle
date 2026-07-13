from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.latent_edge_embedding import (
    LatentSideEmbeddingNet,
    blend_topk_rank_residual,
    compatibility_from_outputs,
    load_latent_edge_checkpoint,
    save_latent_edge_checkpoint,
)
from puzzle_denoise_v2.model import TileNAFNet


def _tiny_model() -> LatentSideEmbeddingNet:
    return LatentSideEmbeddingNet(
        latent_channels=16,
        model_dim=32,
        embedding_dim=24,
        layers=1,
        heads=4,
        feedforward_dim=64,
        side_band=3,
        dropout=0.0,
    )


def test_tilenaf_feature_api_is_checkpoint_neutral_and_matches_forward() -> None:
    torch.manual_seed(3)
    model = TileNAFNet(width=16, encoder_blocks=(1, 1), middle_blocks=1, decoder_blocks=(1, 1))
    tiles = torch.rand(2, 3, 20, 20)
    ordinary = model(tiles)
    restored, features, auxiliary = model.forward_with_features(tiles, return_aux=True)
    torch.testing.assert_close(restored, ordinary)
    assert features.shape == (2, 16, 20, 20)
    assert auxiliary.shape == (2, 5)
    # The API adds no parameters or buffers to checkpoint state.
    clone = TileNAFNet(width=16, encoder_blocks=(1, 1), middle_blocks=1, decoder_blocks=(1, 1))
    clone.load_state_dict(model.state_dict(), strict=True)


def test_latent_encoder_preserves_four_directional_embedding_shapes() -> None:
    torch.manual_seed(5)
    model = _tiny_model().eval()
    raw = torch.randint(0, 256, (7, 3, 20, 20), dtype=torch.uint8)
    restored = torch.rand(7, 3, 20, 20)
    latent = torch.randn(7, 16, 20, 20)
    with torch.inference_mode():
        outputs = model(raw, restored, latent)
    for key in ("q_right", "k_left", "q_down", "k_up"):
        assert outputs[key].shape == (7, 24)
        torch.testing.assert_close(
            outputs[key].norm(dim=1), torch.ones(7), atol=1e-5, rtol=1e-5
        )
    assert outputs["outside_logits"].shape == (7, 4)


def test_latent_encoder_backpropagates_to_trainable_stems_not_frozen_inputs() -> None:
    torch.manual_seed(7)
    model = _tiny_model().train()
    raw = torch.rand(5, 3, 20, 20)
    restored = torch.rand(5, 3, 20, 20)
    latent = torch.rand(5, 16, 20, 20)
    outputs = model(raw, restored, latent)
    loss = sum(outputs[key].square().mean() for key in ("raw_q_right", "raw_k_left"))
    loss.backward()
    assert model.latent_stem[1].weight.grad is not None
    assert torch.count_nonzero(model.latent_stem[1].weight.grad) > 0
    assert model.visual_stem[0].weight.grad is not None


def test_compatibility_and_topk_residual_are_fail_closed() -> None:
    torch.manual_seed(11)
    model = _tiny_model().eval()
    with torch.inference_mode():
        outputs = model(
            torch.rand(576, 3, 20, 20),
            torch.rand(576, 3, 20, 20),
            torch.rand(576, 16, 20, 20),
        )
    learned = compatibility_from_outputs(outputs)
    rng = np.random.default_rng(13)
    right = rng.random((576, 576), dtype=np.float32)
    down = rng.random((576, 576), dtype=np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    base = CompatibilityMatrices("base", right, down)
    candidates = []
    for _ in range(2):
        values = np.empty((576, 8), dtype=np.int32)
        for query in range(576):
            values[query] = rng.choice(
                np.delete(np.arange(576, dtype=np.int32), query), size=8, replace=False
            )
        candidates.append(values)

    identity = blend_topk_rank_residual(base, learned, tuple(candidates), alpha=0.0)
    np.testing.assert_array_equal(identity.right, base.right)
    np.testing.assert_array_equal(identity.down, base.down)

    changed = blend_topk_rank_residual(base, learned, tuple(candidates), alpha=0.1)
    for direction_name, selected in zip(("right", "down"), candidates, strict=True):
        before = getattr(base, direction_name)
        after = getattr(changed, direction_name)
        mask = np.zeros((576, 576), dtype=bool)
        mask[np.arange(576)[:, None], selected] = True
        np.testing.assert_array_equal(after[~mask], before[~mask])
        assert np.max(np.abs(after[mask] - before[mask])) <= 0.050001
    with pytest.raises(ValueError, match="non-negative"):
        blend_topk_rank_residual(base, learned, tuple(candidates), alpha=-0.1)


def test_checkpoint_roundtrip_is_research_only(tmp_path: Path) -> None:
    model = _tiny_model()
    path = tmp_path / "latent_edge.pt"
    save_latent_edge_checkpoint(path, model, metadata={"experiment": "unit"})
    loaded, metadata = load_latent_edge_checkpoint(path)
    assert loaded.config() == model.config()
    assert metadata["experiment"] == "unit"
    assert metadata["safe_for_submission"] is False
    for expected, actual in zip(
        model.state_dict().values(), loaded.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_invalid_inputs_fail_closed() -> None:
    model = _tiny_model()
    raw = torch.rand(2, 3, 20, 20)
    with pytest.raises(ValueError, match="latent_features"):
        model(raw, raw, torch.rand(2, 15, 20, 20))
    with pytest.raises(ValueError, match="restored_tiles"):
        model(raw, torch.rand(3, 3, 20, 20), torch.rand(2, 16, 20, 20))
    # Floating TileNAF outputs have a fixed [0,1] contract: never divide an
    # entire batch by 255 because one residual overshot.
    normalized = model._unit_range(torch.as_tensor([0.5, 2.0], dtype=torch.float32))
    torch.testing.assert_close(normalized, torch.as_tensor([0.5, 1.0]))
    with pytest.raises(ValueError, match="halves"):
        LatentSideEmbeddingNet(model_dim=24, heads=4)
