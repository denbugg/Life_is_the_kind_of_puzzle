from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.dense_pair_residual import (
    DOWN,
    RIGHT,
    DensePairResidualScorer,
    dense_pair_residual_compatibility,
    load_dense_pair_residual_checkpoint,
    load_dense_pair_residual_checkpoint_payload,
    save_dense_pair_residual_checkpoint,
)


def _tiny_model(*, tile_size: int = 6) -> DensePairResidualScorer:
    return DensePairResidualScorer(
        tile_size=tile_size,
        encoder_width=8,
        encoder_depth=2,
        expansion=2,
        side_band=2,
        profile_bins=3,
        embedding_dim=8,
        relation_hidden=16,
        pair_hidden=8,
        dropout=0.0,
        max_residual=0.2,
        initial_gain_fraction=0.4,
    )


def _activate_residual_head(model: DensePairResidualScorer) -> None:
    final = model.relation_head[-1]
    assert isinstance(final, torch.nn.Linear)
    torch.nn.init.normal_(final.weight, mean=0.0, std=0.1)
    torch.nn.init.constant_(final.bias, 0.03)


def test_default_model_is_bounded_pilot_size_and_zero_initialized() -> None:
    model = DensePairResidualScorer()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 2_000_000 <= parameters <= 4_000_000
    assert 0.0 < float(model.bounded_gain.detach()) < model.max_residual
    final = model.relation_head[-1]
    assert isinstance(final, torch.nn.Linear)
    assert torch.count_nonzero(final.weight) == 0
    assert torch.count_nonzero(final.bias) == 0


def test_zero_init_is_exact_and_first_step_can_train_output_layer() -> None:
    torch.manual_seed(3)
    model = _tiny_model().train()
    raw = torch.randint(0, 256, (7, 3, 6, 6), dtype=torch.uint8)
    denoised = torch.randint(0, 256, (7, 3, 6, 6), dtype=torch.uint8)
    first = torch.as_tensor([0, 0, 2, 4, 6])
    second = torch.as_tensor([1, 6, 3, 1, 5])
    direction = torch.as_tensor([RIGHT, DOWN, RIGHT, DOWN, RIGHT])
    residual = model(raw, denoised, first, second, direction)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=0.0, rtol=0.0)
    residual.sum().backward()
    final = model.relation_head[-1]
    assert isinstance(final, torch.nn.Linear)
    assert final.weight.grad is not None
    assert torch.count_nonzero(final.weight.grad) > 0


def test_optional_denoised_view_has_explicit_availability_signal() -> None:
    torch.manual_seed(5)
    model = _tiny_model().eval()
    raw = torch.rand(5, 3, 6, 6)
    with torch.inference_mode():
        raw_only = model.encode_tiles(raw, None)
        restored_available = model.encode_tiles(raw, raw.clone())
    # RGB and difference are identical; only the availability plane differs.
    assert not torch.allclose(
        raw_only.global_embeddings,
        restored_available.global_embeddings,
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.parametrize("direction", [RIGHT, DOWN])
def test_chunked_dense_scores_match_selected_pairs_without_topk(direction: int) -> None:
    torch.manual_seed(7 + direction)
    model = _tiny_model().eval()
    _activate_residual_head(model)
    raw = torch.rand(9, 3, 6, 6)
    denoised = torch.rand(9, 3, 6, 6)
    with torch.inference_mode():
        bank = model.encode_tiles(raw, denoised)
        dense = model.score_dense(bank, direction, chunk_size=4)
        first = torch.arange(9).repeat_interleave(9)
        second = torch.arange(9).repeat(9)
        directions = torch.full((81,), direction)
        selected = model.forward_from_encoded(bank, first, second, directions)
    assert dense.shape == (9, 9)
    assert torch.isfinite(dense).all()
    torch.testing.assert_close(dense.reshape(-1), selected, atol=1e-7, rtol=1e-6)
    assert float(dense.abs().max()) <= float(model.bounded_gain.detach()) + 1e-7


def test_pair_head_is_ordered_directional_and_backpropagates_to_pixels() -> None:
    torch.manual_seed(11)
    model = _tiny_model().train()
    _activate_residual_head(model)
    raw = torch.rand(4, 3, 6, 6, requires_grad=True)
    first = torch.as_tensor([0, 1, 0, 1])
    second = torch.as_tensor([1, 0, 1, 0])
    directions = torch.as_tensor([RIGHT, RIGHT, DOWN, DOWN])
    residual = model(raw, None, first, second, directions)
    assert residual.shape == (4,)
    assert not torch.allclose(residual[0], residual[1])
    assert not torch.allclose(residual[0], residual[2])
    residual.sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert torch.count_nonzero(raw.grad) > 0


def test_apply_residual_is_the_exact_shared_train_inference_math() -> None:
    base = torch.as_tensor([0.1, 0.5, 0.9])
    residual = torch.as_tensor([-0.02, 0.0, 0.03])
    combined = DensePairResidualScorer.apply_residual(base, residual)
    torch.testing.assert_close(combined, base + residual)
    with pytest.raises(ValueError, match="identical shapes"):
        DensePairResidualScorer.apply_residual(base, residual[:2])


def test_full_inference_returns_compatibility_masks_self_and_encodes_once() -> None:
    torch.manual_seed(13)
    model = DensePairResidualScorer(
        tile_size=4,
        encoder_width=4,
        encoder_depth=1,
        expansion=1,
        side_band=1,
        profile_bins=2,
        embedding_dim=4,
        relation_hidden=4,
        pair_hidden=2,
        dropout=0.0,
        max_residual=0.1,
    )
    rng = np.random.default_rng(17)
    raw = rng.integers(0, 256, size=(576, 4, 4, 3), dtype=np.uint8)
    right = rng.random((576, 576), dtype=np.float32)
    down = rng.random((576, 576), dtype=np.float32)
    # Deliberately finite diagonals: inference itself must forbid self-pairs.
    base = CompatibilityMatrices("base", right.copy(), down.copy())
    telemetry: dict[str, object] = {}
    result = dense_pair_residual_compatibility(
        model,
        raw,
        base,
        chunk_size=192,
        telemetry=telemetry,
    )
    assert isinstance(result, CompatibilityMatrices)
    off_diagonal = ~np.eye(576, dtype=bool)
    np.testing.assert_array_equal(result.right[off_diagonal], right[off_diagonal])
    np.testing.assert_array_equal(result.down[off_diagonal], down[off_diagonal])
    assert np.isposinf(np.diag(result.right)).all()
    assert np.isposinf(np.diag(result.down)).all()
    assert telemetry["tile_encoder_passes"] == 1
    assert telemetry["scored_pairs_per_direction"] == 576 * 576
    assert telemetry["proposal_top_k"] is None


def test_checkpoint_roundtrip_preserves_config_weights_and_metadata(tmp_path: Path) -> None:
    torch.manual_seed(19)
    path = tmp_path / "dense_pair.pt"
    model = _tiny_model()
    _activate_residual_head(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    save_dense_pair_residual_checkpoint(
        path,
        model,
        metadata={"experiment": "unit"},
        optimizer_state=optimizer.state_dict(),
        training_state={"epoch": 2},
    )
    payload = load_dense_pair_residual_checkpoint_payload(path)
    loaded, metadata = load_dense_pair_residual_checkpoint(path)
    assert loaded.config() == model.config()
    assert metadata["experiment"] == "unit"
    assert metadata["safe_for_submission"] is False
    assert payload["training_state"] == {"epoch": 2}
    for expected, actual in zip(
        model.state_dict().values(), loaded.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_invalid_configuration_and_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="side_band"):
        DensePairResidualScorer(tile_size=4, side_band=5)
    model = _tiny_model()
    raw = torch.rand(3, 3, 5, 6)
    indices = torch.as_tensor([0])
    with pytest.raises(ValueError, match="raw_tiles"):
        model(raw, None, indices, indices, torch.as_tensor([RIGHT]))
    valid = torch.rand(3, 3, 6, 6)
    bank = model.encode_tiles(valid)
    with pytest.raises(ValueError, match="direction"):
        model.forward_from_encoded(
            bank, indices, indices, torch.as_tensor([2])
        )
    with pytest.raises(ValueError, match="chunk_size"):
        model.score_dense(bank, RIGHT, chunk_size=0)
