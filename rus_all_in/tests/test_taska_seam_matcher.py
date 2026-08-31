from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

import aiijc_puzzle.taska_seam_matcher as seam
from aiijc_puzzle.taska_seam_matcher import (
    DEFAULT_CHECKPOINT_DIR,
    TASKA_CHECKPOINTS,
    TaskaCheckpointError,
    TaskaSeamConfig,
    analytic_view,
    calibrated_log_assignments,
    load_taska_checkpoint,
    match_taska_tiles,
    pessimistic_log_assignments,
)


class _ToyMatcher(nn.Module):
    """Fast asymmetric descriptor matcher for algebra/device tests."""

    def __init__(self, variant: float) -> None:
        super().__init__()
        self.variant = variant
        self.anchor = nn.Parameter(torch.tensor(variant), requires_grad=False)

    def right_down_logits(
        self,
        tiles_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = tiles_tensor.float() / 255.0
        mean = values.mean(dim=(2, 3))
        deviation = values.std(dim=(2, 3))
        features = torch.cat(
            [
                mean,
                deviation,
                mean[:, :1] * mean[:, 1:2],
                torch.ones_like(mean[:, :1]),
            ],
            dim=1,
        )
        right_source = features * torch.tensor(
            [1.0, 0.7, -0.4, 0.3, 0.8, -0.6, 0.5, 0.2],
            device=features.device,
        )
        right_target = features.roll(1, dims=1) * torch.tensor(
            [0.5, -0.8, 0.6, 1.1, 0.4, 0.9, -0.3, 0.7],
            device=features.device,
        )
        down_source = features.roll(2, dims=1) * torch.tensor(
            [-0.6, 0.4, 1.0, 0.7, -0.5, 0.8, 0.3, 0.9],
            device=features.device,
        )
        down_target = features * torch.tensor(
            [0.9, 0.2, -0.7, 0.5, 1.1, -0.4, 0.8, 0.3],
            device=features.device,
        )
        scale = 11.0 + self.anchor
        return scale * (right_source @ right_target.t()), scale * (
            down_source @ down_target.t()
        )


def _random_tiles(count: int, seed: int = 41) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(count, 20, 20, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    ("kind", "filename", "parameter_count", "descriptor_size", "golden_right"),
    (
        (
            "v3",
            "seam_embed_v3.pt",
            5_931_955,
            192,
            np.asarray(
                [[13.7783422470, 12.6001558304], [7.8665246964, 18.3144893646]]
            ),
        ),
        (
            "local",
            "seam_embed_local.pt",
            1_729_459,
            3_840,
            np.asarray(
                [[14.4985351562, 6.2586517334], [4.2543745041, 17.9590263367]]
            ),
        ),
    ),
)
def test_audited_checkpoint_strict_load_and_golden_cpu_inference(
    kind: str,
    filename: str,
    parameter_count: int,
    descriptor_size: int,
    golden_right: np.ndarray,
) -> None:
    checkpoint = DEFAULT_CHECKPOINT_DIR / filename
    if not checkpoint.is_file():
        pytest.skip(f"optional audited artifact is absent: {checkpoint}")
    model = load_taska_checkpoint(checkpoint, kind, device="cpu")
    inputs = (
        torch.arange(2 * 3 * 20 * 20, dtype=torch.float32).reshape(2, 3, 20, 20)
        % 256
    )

    with torch.inference_mode():
        outputs = model(inputs)
        right, down = model.right_down_logits(inputs)

    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
    assert model.checkpoint_spec == TASKA_CHECKPOINTS[kind]
    assert model.checkpoint_path == checkpoint.resolve()
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert [tuple(value.shape) for value in outputs[:4]] == [(2, descriptor_size)] * 4
    assert tuple(outputs[4][0].shape) == (2, 3, 3, 20)
    assert tuple(right.shape) == tuple(down.shape) == (2, 2)
    assert np.allclose(right.numpy(), golden_right, atol=2e-4, rtol=2e-5)
    assert torch.all(torch.diag(right) > -1e3)  # raw protocol is pre-mask


def test_bad_hash_is_rejected_before_pickle_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "seam_embed_v3.pt"
    checkpoint.write_bytes(b"not a trusted torch pickle")
    deserialised = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal deserialised
        deserialised = True
        raise AssertionError("torch.load must not run before the digest is accepted")

    monkeypatch.setattr(torch, "load", forbidden_load)

    with pytest.raises(TaskaCheckpointError, match="SHA-256 mismatch"):
        load_taska_checkpoint(checkpoint, "v3", device="cpu")
    assert not deserialised


def test_matching_hash_still_requires_exact_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "seam_embed_v3.pt"
    checkpoint.write_bytes(b"stand-in")
    spec = TASKA_CHECKPOINTS["v3"]
    bad_args = dict(spec.args)
    bad_args["head"] = "local"
    payload = {
        "model": {"pred.placeholder": torch.zeros(1)},
        "args": bad_args,
        "eval": {"R@1": 0.1, "R@20": 0.2, "twinR@1": 0.3},
        "step": spec.step,
    }
    monkeypatch.setattr(seam, "_file_sha256", lambda path: spec.sha256)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)

    with pytest.raises(TaskaCheckpointError, match="metadata does not match"):
        load_taska_checkpoint(checkpoint, "v3", device="cpu")


def test_analytic_views_are_the_exact_per_tile_opencv_filters() -> None:
    tiles = _random_tiles(3).astype(np.float32)

    median = analytic_view("median", tiles)
    bilateral = analytic_view("bilateral", tiles)

    expected_median = np.stack(
        [cv2.medianBlur(tile.astype(np.uint8), 3) for tile in tiles]
    ).astype(np.float32)
    expected_bilateral = np.stack(
        [cv2.bilateralFilter(tile.astype(np.uint8), 7, 50, 7) for tile in tiles]
    ).astype(np.float32)
    assert np.array_equal(median, expected_median)
    assert np.array_equal(bilateral, expected_bilateral)


def _reference_sink(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    result = logits
    for _ in range(iterations):
        result = result - torch.logsumexp(result, dim=1, keepdim=True)
        result = result - torch.logsumexp(result, dim=0, keepdim=True)
    return result


def test_two_model_fusion_is_the_pessimistic_minimum_not_an_average() -> None:
    tiles = _random_tiles(6)
    models = (_ToyMatcher(0.0), _ToyMatcher(2.0))
    iterations = 4
    actual_right, actual_down = pessimistic_log_assignments(
        models,
        tiles,
        device="cpu",
        rounds=0,
        sinkhorn_iterations=iterations,
    )
    tensor = torch.from_numpy(tiles.astype(np.float32)).permute(0, 3, 1, 2)
    per_model: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        for model in models:
            right, down = model.right_down_logits(tensor)
            right.fill_diagonal_(-1e4)
            down.fill_diagonal_(-1e4)
            per_model.append(
                (_reference_sink(right, iterations), _reference_sink(down, iterations))
            )
    expected_right = _reference_sink(
        torch.stack([pair[0] for pair in per_model]).amin(0), iterations
    )
    expected_down = _reference_sink(
        torch.stack([pair[1] for pair in per_model]).amin(0), iterations
    )

    assert np.allclose(actual_right, expected_right.numpy(), atol=1e-6)
    assert np.allclose(actual_down, expected_down.numpy(), atol=1e-6)


def test_full_frontend_is_equivariant_to_input_bag_relabeling() -> None:
    tiles = _random_tiles(8, seed=93)
    models = (_ToyMatcher(0.0), _ToyMatcher(1.7))
    config = TaskaSeamConfig(vote_target=0, votes=1)
    original = match_taska_tiles(
        tiles,
        models,
        config=config,
        device="cpu",
        require_verified=False,
    )
    order = np.asarray([5, 0, 7, 2, 6, 1, 4, 3])
    relabelled = match_taska_tiles(
        tiles[order],
        models,
        config=config,
        device="cpu",
        require_verified=False,
    )

    assert np.allclose(
        relabelled.right_log,
        original.right_log[np.ix_(order, order)],
        atol=2e-5,
    )
    assert np.allclose(
        relabelled.down_log,
        original.down_log[np.ix_(order, order)],
        atol=2e-5,
    )
    original_edges = {
        (edge.source, edge.target, edge.axis) for edge in original.candidate_edges
    }
    relabelled_edges_in_original_ids = {
        (int(order[edge.source]), int(order[edge.target]), edge.axis)
        for edge in relabelled.candidate_edges
    }
    assert relabelled_edges_in_original_ids == original_edges
    assert original.scorer_count == relabelled.scorer_count == 12
    assert original.candidate_edges
    assert not original.right_log.flags.writeable
    assert not original.cost_right.flags.writeable
    expected_cost = -original.right_log
    expected_cost -= expected_cost.min()
    np.fill_diagonal(expected_cost, 0.0)
    assert np.array_equal(original.cost_right, expected_cost)


def test_target_position_quad_path_is_explicitly_rejected() -> None:
    tiles = _random_tiles(4)
    models = (_ToyMatcher(0.0), _ToyMatcher(1.0))
    config = TaskaSeamConfig(
        views=("raw",),
        orientations=1,
        votes=1,
        vote_target=0,
        quad_weight=0.4,
    )

    with pytest.raises(ValueError, match="quad_weight must be zero"):
        match_taska_tiles(
            tiles,
            models,
            config=config,
            device="cpu",
            require_verified=False,
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_inference_uses_the_float32_non_cuda_path() -> None:
    tiles = _random_tiles(4)
    model = _ToyMatcher(0.5).to("mps")

    right, down = calibrated_log_assignments(
        model,
        tiles,
        device="mps",
        rounds=1,
        sinkhorn_iterations=3,
    )

    assert right.shape == down.shape == (4, 4)
    assert np.isfinite(right).all()
    assert np.isfinite(down).all()

