from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.tri_emitter_edge_verifier import (
    AUXILIARY_DIM,
    EMITTERS,
    TriEmitterEdgeVerifier,
    build_candidate_pool,
    compress_dino_boundary_tokens,
    fixed_dino_projection,
    ordered_raw_side_sequences,
    sparse_reciprocal_evidence,
)


def _scores(count: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    generator = np.random.default_rng(7)
    result = {}
    for emitter in EMITTERS:
        axes = []
        for _ in range(2):
            matrix = generator.normal(size=(count, count)).astype(np.float32)
            np.fill_diagonal(matrix, -100.0)
            axes.append(matrix)
        result[emitter] = (axes[0], axes[1])
    return result


def test_candidate_union_preserves_raw_and_is_label_free() -> None:
    pool = build_candidate_pool(_scores(16), top_k=5)
    assert pool.candidates.shape == (2, 16, 15)
    assert pool.auxiliary.shape == (2, 16, 15, AUXILIARY_DIM)
    assert len(pool.identity_digest) == 64
    for axis in range(2):
        for anchor in range(16):
            candidates = pool.candidates[axis, anchor, pool.valid[axis, anchor]]
            assert len(candidates) == len(set(candidates.tolist()))
            assert set(pool.emitter_topk[0, axis, anchor]).issubset(candidates)


def test_ordered_content_features_are_finite_and_deterministic() -> None:
    generator = np.random.default_rng(9)
    tiles = generator.integers(0, 256, size=(16, 20, 20, 3), dtype=np.uint8)
    raw = ordered_raw_side_sequences(tiles)
    assert raw.shape == (4, 16, 20, 6)
    tokens = generator.normal(size=(16, 7, 7, 24)).astype(np.float32)
    projection = fixed_dino_projection(24, output_dim=8, seed=11)
    dino = compress_dino_boundary_tokens(tokens, projection)
    assert dino.shape == (4, 16, 14, 8)
    assert np.isfinite(dino).all()
    np.testing.assert_array_equal(projection, fixed_dino_projection(24, output_dim=8, seed=11))


def test_zero_initialised_vectorized_model_replays_raw_baseline() -> None:
    generator = np.random.default_rng(12)
    pool = build_candidate_pool(_scores(16), top_k=5)
    model = TriEmitterEdgeVerifier(dino_dim=8, width=16, hidden=32).eval()
    raw_sides = torch.from_numpy(
        generator.normal(size=(4, 16, 20, 6)).astype(np.float32)
    )
    dino_sides = torch.from_numpy(
        generator.normal(size=(4, 16, 14, 8)).astype(np.float32)
    )
    axis = 0
    anchors = torch.arange(16)
    candidates = torch.from_numpy(pool.candidates[axis]).long()
    valid = torch.from_numpy(pool.valid[axis])
    auxiliary = torch.from_numpy(pool.auxiliary[axis])
    baseline = torch.from_numpy(pool.raw_baseline[axis])
    directions = torch.zeros(16, dtype=torch.long)
    with torch.inference_mode():
        logits, delta = model(
            raw_sides,
            dino_sides,
            anchors,
            candidates,
            valid,
            directions,
            auxiliary,
            baseline,
        )
    torch.testing.assert_close(delta, torch.zeros_like(delta))
    torch.testing.assert_close(logits[valid], baseline[valid])
    evidence = sparse_reciprocal_evidence(
        pool.candidates[axis], pool.valid[axis], logits.numpy()
    )
    assert evidence["target"].shape == (16,)
    assert evidence["reciprocal"].dtype == np.bool_
