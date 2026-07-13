from __future__ import annotations

import numpy as np
import pytest
import torch

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.geometry import TILE_COUNT
from puzzle_assembly.masked_gap import (
    DOWN,
    GAP_START,
    GAP_STOP,
    RIGHT,
    MaskedGapGenerator,
    PairListwiseRanker,
    blend_with_w4,
    canonical_pair_canvas,
    charbonnier_loss,
    clean_gap_target,
    gap_baselines,
    generator_input,
    hard_negative_groups,
    listwise_pair_loss,
    listwise_view_loss,
    load_models,
    module_state_sha256,
    ranker_input,
    state_dict_payload,
)


def _tiles(batch: int = 2) -> torch.Tensor:
    values = torch.arange(batch * 3 * 20 * 20, dtype=torch.float32)
    return (values.reshape(batch, 3, 20, 20) % 251) / 250.0


def test_canonical_pair_rotates_down_counter_clockwise() -> None:
    first = _tiles()
    second = first.flip(-1)
    direction = torch.tensor([RIGHT, DOWN])
    canvas = canonical_pair_canvas(first, second, direction)
    assert canvas.shape == (2, 3, 20, 40)
    assert torch.equal(canvas[0, :, :, :20], first[0])
    assert torch.equal(canvas[1, :, :, :20], torch.rot90(first[1], 1, (-2, -1)))


def test_generator_input_masks_exact_four_columns_and_target_matches() -> None:
    raw = _tiles()
    denoised = raw * 0.8
    direction = torch.tensor([RIGHT, DOWN])
    values = generator_input(raw, raw.flip(-1), denoised, denoised.flip(-1), direction)
    assert values.shape == (2, 7, 20, 40)
    assert torch.count_nonzero(values[:, :6, :, GAP_START:GAP_STOP]) == 0
    assert torch.all(values[:, 6:, :, GAP_START:GAP_STOP] == 1)
    assert torch.count_nonzero(values[:, 6:, :, :GAP_START]) == 0
    target = clean_gap_target(raw, raw.flip(-1), direction)
    canvas = canonical_pair_canvas(raw, raw.flip(-1), direction)
    assert torch.equal(target, canvas[..., GAP_START:GAP_STOP])

    perturbed_raw = raw.clone()
    perturbed_denoised = denoised.clone()
    perturbed_raw[..., -2:] = 1.0 - perturbed_raw[..., -2:]
    perturbed_denoised[..., -2:] = 1.0 - perturbed_denoised[..., -2:]
    # For the first/right example, first-tile columns 18:20 are hidden.  Their
    # perturbation cannot leak into the generator input.
    changed = generator_input(
        perturbed_raw[:1], raw.flip(-1)[:1], perturbed_denoised[:1], denoised.flip(-1)[:1], direction[:1]
    )
    assert torch.equal(changed, values[:1])


def test_gap_baselines_copy_and_interpolation() -> None:
    canvas = torch.zeros((1, 3, 20, 40))
    canvas[..., GAP_START:GAP_STOP] = torch.tensor([0.1, 0.2, 0.8, 0.9])
    canvas[..., GAP_START - 1] = 0.0
    canvas[..., GAP_STOP] = 1.0
    baselines = gap_baselines(canvas)
    assert torch.allclose(
        baselines["copy"][0, 0, 0], torch.tensor([0.1, 0.2, 0.8, 0.9])
    )
    assert torch.allclose(
        baselines["interpolation"][0, 0, 0], torch.tensor([0.2, 0.4, 0.6, 0.8])
    )
    perturbed = canvas.clone()
    perturbed[..., GAP_START:GAP_STOP] = 1.0 - perturbed[..., GAP_START:GAP_STOP]
    changed = gap_baselines(perturbed)
    assert not torch.equal(changed["copy"], baselines["copy"])
    assert torch.equal(changed["interpolation"], baselines["interpolation"])


def test_models_have_frozen_shapes_and_equal_ranker_capacity() -> None:
    raw = _tiles()
    denoised = raw.clone()
    direction = torch.tensor([RIGHT, DOWN])
    generator = MaskedGapGenerator(width=16, blocks=1)
    gap = generator(generator_input(raw, raw, denoised, denoised, direction))
    assert gap.shape == (2, 3, 20, 4)
    inpaint_values = ranker_input(raw, raw, denoised, denoised, direction, gap)
    direct_values = ranker_input(raw, raw, denoised, denoised, direction, None)
    assert inpaint_values.shape == direct_values.shape == (2, 10, 20, 40)
    assert torch.count_nonzero(direct_values[:, 6:9]) == 0
    first = PairListwiseRanker(width=16, blocks=1)
    import copy

    second = copy.deepcopy(first)
    assert sum(p.numel() for p in first.parameters()) == sum(p.numel() for p in second.parameters())
    assert module_state_sha256(first) == module_state_sha256(second)
    assert first(inpaint_values).shape == second(direct_values).shape == (2,)


def test_losses_require_exact_1_plus_31_group() -> None:
    outgoing = torch.randn(3, 32, requires_grad=True)
    incoming = torch.randn(3, 32, requires_grad=True)
    loss, parts = listwise_pair_loss(outgoing, incoming)
    loss.backward()
    assert set(parts) == {"total", "outgoing_ce", "incoming_ce", "bce"}
    assert outgoing.grad is not None and incoming.grad is not None
    assert charbonnier_loss(torch.zeros(1), torch.ones(1)) > 0
    with pytest.raises(ValueError):
        listwise_pair_loss(outgoing[:, :31], incoming[:, :31])


def test_two_ddp_view_losses_exactly_equal_frozen_pair_objective() -> None:
    outgoing = torch.randn(4, 32)
    incoming = torch.randn(4, 32)
    pair, _ = listwise_pair_loss(outgoing, incoming)
    split = listwise_view_loss(outgoing) + listwise_view_loss(incoming)
    assert torch.allclose(pair, split, atol=1e-6, rtol=1e-6)


def test_hard_negative_groups_are_truth_first_and_stable() -> None:
    base = np.broadcast_to(np.arange(TILE_COUNT, dtype=np.float32), (TILE_COUNT, TILE_COUNT)).copy()
    np.fill_diagonal(base, np.inf)
    score = CompatibilityMatrices("w4", base.copy(), base.copy())
    permutation = np.arange(TILE_COUNT, dtype=np.int32)
    outgoing, incoming = hard_negative_groups(score, permutation)
    assert outgoing.first.shape == incoming.first.shape == (1104, 32)
    assert outgoing.second[0, 0] == 1
    assert incoming.first[0, 0] == 0
    assert np.all(outgoing.first[:, 0] != outgoing.second[:, 0])
    assert len(np.unique(outgoing.second[0])) == 32


def test_frozen_blend_and_strict_checkpoint_roundtrip() -> None:
    matrix = np.broadcast_to(np.arange(TILE_COUNT, dtype=np.float32), (TILE_COUNT, TILE_COUNT)).copy()
    np.fill_diagonal(matrix, np.inf)
    w4 = CompatibilityMatrices("w4", matrix.copy(), matrix.copy())
    learned = CompatibilityMatrices("learned", matrix[:, ::-1].copy(), matrix[:, ::-1].copy())
    np.fill_diagonal(learned.right, np.inf)
    np.fill_diagonal(learned.down, np.inf)
    blend = blend_with_w4(w4, learned)
    assert blend.name == "frozen_w4_masked_gap_equal_rank_blend"
    with pytest.raises(ValueError):
        blend_with_w4(w4, learned, learned_weight=0.5)

    generator = MaskedGapGenerator(width=16, blocks=1)
    inpaint = PairListwiseRanker(width=16, blocks=1)
    direct = PairListwiseRanker(width=16, blocks=1)
    payload = state_dict_payload(generator, inpaint, direct, metadata={"safe_for_submission": False})
    loaded = load_models(payload)
    assert loaded[3] == {"safe_for_submission": False}
    assert module_state_sha256(loaded[0]) == module_state_sha256(generator)
    unsafe = dict(payload)
    unsafe["metadata"] = {}
    with pytest.raises(RuntimeError, match="fail-closed"):
        load_models(unsafe)
    broken = dict(payload)
    broken["extra"] = 1
    with pytest.raises(RuntimeError):
        load_models(broken)
