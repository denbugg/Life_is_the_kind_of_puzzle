from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import aiijc_puzzle.taska_focal_verifier as focal
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_verifier import (
    TASKA_FOCAL_VERIFIER_ARGS,
    TASKA_FOCAL_VERIFIER_PARAMETER_COUNT,
    TASKA_FOCAL_VERIFIER_SHA256,
    SeamVerifier,
    TaskaFocalCheckpointError,
    build_focal_seam_patches,
    extract_focal_edge_features,
    load_taska_focal_verifier,
    score_focal_edges,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOVERED_CHECKPOINT = PROJECT_ROOT / "artifacts/prior-taska/ckpt/verify_pair_best.pt"


def _random_tiles(grid: int, seed: int = 91) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(
        0,
        256,
        size=(grid * grid, 20, 20, 3),
        dtype=np.uint8,
    )


def _costs(grid: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = grid * grid
    return (
        rng.uniform(0.0, 20.0, size=(count, count)).astype(np.float32),
        rng.uniform(0.0, 20.0, size=(count, count)).astype(np.float32),
    )


def _audited_model() -> SeamVerifier:
    if not RECOVERED_CHECKPOINT.is_file():
        pytest.skip(f"optional recovered checkpoint is absent: {RECOVERED_CHECKPOINT}")
    return load_taska_focal_verifier(RECOVERED_CHECKPOINT, device="cpu")


def test_bad_size_or_hash_is_rejected_before_deserialisation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "verify_pair_best.pt"
    checkpoint.write_bytes(b"not a trusted torch checkpoint")
    deserialised = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal deserialised
        deserialised = True
        raise AssertionError("torch.load must not run before the artifact gate")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(TaskaFocalCheckpointError, match="size mismatch"):
        load_taska_focal_verifier(checkpoint)
    assert not deserialised


def test_matching_digest_still_requires_exact_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "verify_pair_best.pt"
    checkpoint.write_bytes(b"x" * focal.TASKA_FOCAL_VERIFIER_SIZE_BYTES)
    bad_args = dict(TASKA_FOCAL_VERIFIER_ARGS)
    bad_args["strip"] = 5
    payload = {"model": {}, "args": bad_args}
    monkeypatch.setattr(focal, "_file_sha256", lambda path: TASKA_FOCAL_VERIFIER_SHA256)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)

    with pytest.raises(TaskaFocalCheckpointError, match="metadata does not match"):
        load_taska_focal_verifier(checkpoint)


def test_audited_checkpoint_load_is_strict_frozen_and_golden() -> None:
    model = _audited_model()
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        TASKA_FOCAL_VERIFIER_PARAMETER_COUNT
    )
    assert model.checkpoint_path == RECOVERED_CHECKPOINT.resolve()
    assert model.checkpoint_sha256 == TASKA_FOCAL_VERIFIER_SHA256
    assert model.checkpoint_args == TASKA_FOCAL_VERIFIER_ARGS
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())

    patch = torch.arange(2 * 3 * 20 * 8, dtype=torch.float32).reshape(2, 3, 20, 8) % 256
    features = torch.tensor(
        [[-0.5, -1.0, 0.0, 1.0, -0.8, 2.0], [-1.2, -0.4, 3.0, 0.0, -0.7, 5.0]]
    )
    with torch.inference_mode():
        actual = model(patch, features).numpy()
    expected = np.asarray([-1.5981793404, -3.0297243595], dtype=np.float32)
    assert np.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_patch_builder_matches_historical_horizontal_and_vertical_join() -> None:
    grid = 3
    tiles = np.zeros((grid * grid, 20, 20, 3), dtype=np.uint8)
    for tile in range(len(tiles)):
        rows, columns = np.indices((20, 20))
        tiles[tile, :, :, 0] = tile
        tiles[tile, :, :, 1] = rows
        tiles[tile, :, :, 2] = columns
    edges = (
        RawTailEdge(1, 7, "right"),
        RawTailEdge(6, 2, "down"),
    )
    patches = build_focal_seam_patches(tiles, edges, grid=grid)

    expected_right = np.concatenate(
        [tiles[1, :, -4:], tiles[7, :, :4]],
        axis=1,
    ).transpose(2, 0, 1)
    expected_down = np.concatenate(
        [tiles[6, -4:, :], tiles[2, :4, :]],
        axis=0,
    ).transpose(1, 0, 2).transpose(2, 0, 1)
    assert patches.shape == (2, 3, 20, 8)
    assert np.array_equal(patches[0], expected_right.astype(np.float32))
    assert np.array_equal(patches[1], expected_down.astype(np.float32))


@pytest.mark.parametrize(
    ("mode", "top_k"),
    (("train_exact_top5", 5), ("historical_tip_top8", 8)),
)
def test_features_match_literal_historical_formula(mode: str, top_k: int) -> None:
    grid = 3
    right, down = _costs(grid)
    edges = (
        RawTailEdge(2, 6, "right"),
        RawTailEdge(4, 1, "down"),
    )
    actual = extract_focal_edge_features(right, down, edges, mode=mode, grid=grid)

    expected = []
    for edge, cost in zip(edges, (right, down), strict=True):
        compatibility = -cost.astype(np.float64)
        np.fill_diagonal(compatibility, -1e9)
        row_sorted = np.sort(compatibility, axis=1)[:, ::-1]
        score = compatibility[edge.source, edge.target]
        best = row_sorted[edge.source, 0]
        expected.append(
            [
                score / 10.0,
                score - best,
                float((compatibility[edge.source] > score).sum()),
                float(score == best),
                row_sorted[edge.source, :top_k].mean() / 10.0,
                row_sorted[edge.source, 0] - row_sorted[edge.source, top_k - 1],
            ]
        )
    assert np.array_equal(actual, np.asarray(expected, dtype=np.float32))


def test_top5_and_top8_differ_only_in_mean_and_spread_features() -> None:
    grid = 3
    right, down = _costs(grid, seed=187)
    edges = (RawTailEdge(0, 1, "right"), RawTailEdge(7, 3, "down"))
    top5 = extract_focal_edge_features(
        right,
        down,
        edges,
        mode="train_exact_top5",
        grid=grid,
    )
    top8 = extract_focal_edge_features(
        right,
        down,
        edges,
        mode="historical_tip_top8",
        grid=grid,
    )
    assert np.array_equal(top5[:, :4], top8[:, :4])
    assert np.any(top5[:, 4:] != top8[:, 4:])


def test_scoring_is_edge_aligned_and_equivariant_to_bag_relabeling() -> None:
    model = _audited_model()
    grid = 3
    count = grid * grid
    tiles = _random_tiles(grid)
    right, down = _costs(grid, seed=211)
    edges = (
        RawTailEdge(0, 4, "right"),
        RawTailEdge(8, 3, "down"),
        RawTailEdge(2, 7, "right"),
        RawTailEdge(5, 1, "down"),
    )
    original = score_focal_edges(
        model,
        tiles,
        right,
        down,
        edges,
        mode="train_exact_top5",
        grid=grid,
        chunk_size=2,
    )

    order = np.asarray([5, 0, 8, 2, 6, 1, 4, 3, 7])
    inverse = np.empty(count, dtype=np.int64)
    inverse[order] = np.arange(count)
    relabelled_edges = tuple(
        RawTailEdge(int(inverse[edge.source]), int(inverse[edge.target]), edge.axis)
        for edge in edges
    )
    relabelled = score_focal_edges(
        model,
        tiles[order],
        right[np.ix_(order, order)],
        down[np.ix_(order, order)],
        relabelled_edges,
        mode="train_exact_top5",
        grid=grid,
        chunk_size=3,
    )

    assert original.edges == edges
    assert relabelled.edges == relabelled_edges
    assert np.allclose(relabelled.features, original.features, atol=0.0, rtol=0.0)
    assert np.allclose(relabelled.logits, original.logits, atol=2e-5, rtol=2e-5)
    assert not original.logits.flags.writeable
    assert not original.features.flags.writeable


def test_raw_uint8_and_unique_edge_contracts_are_strict() -> None:
    tiles = _random_tiles(3)
    edge = RawTailEdge(0, 1, "right")
    with pytest.raises(ValueError, match="raw uint8"):
        build_focal_seam_patches(tiles.astype(np.float32), (edge,), grid=3)
    with pytest.raises(ValueError, match="duplicates"):
        build_focal_seam_patches(tiles, (edge, edge), grid=3)
