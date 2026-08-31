from __future__ import annotations

import numpy as np
import pytest

import aiijc_puzzle.guided_fourth_emitter as module
from aiijc_puzzle.guided_fourth_emitter import (
    FOURTH_EMITTERS,
    GUIDED_AUXILIARY_DIM,
    extend_with_guided_emitter,
    fixed_guided_standalone_scores,
    pool_from_target_free_legacy_cache,
)
from aiijc_puzzle.tri_emitter_edge_verifier import (
    AUXILIARY_DIM,
    EMITTERS,
    TOP_K,
    CandidatePool,
    build_candidate_pool,
)


def _ranked_matrix(count: int, offset: int) -> np.ndarray:
    matrix = np.empty((count, count), dtype=np.float32)
    for source in range(count):
        order = [(source + offset + step) % count for step in range(count)]
        order = [target for target in order if target != source]
        matrix[source] = -count
        matrix[source, order] = -np.arange(len(order), dtype=np.float32)
        matrix[source, source] = -1e4
    return matrix


def _legacy_pool(count: int = 80) -> CandidatePool:
    shared = _ranked_matrix(count, 1)
    return build_candidate_pool(
        {emitter: (shared, shared.copy()) for emitter in EMITTERS}, top_k=TOP_K
    )


def test_fourth_emitter_keeps_legacy_slots_and_raw_top32() -> None:
    legacy = _legacy_pool()
    guided = _ranked_matrix(80, 33)
    extended = extend_with_guided_emitter(legacy, (guided, guided.copy()))

    assert FOURTH_EMITTERS[:3] == EMITTERS
    assert extended.candidates.shape == (2, 80, 4 * TOP_K)
    assert extended.guided_auxiliary.shape == (
        2,
        80,
        4 * TOP_K,
        GUIDED_AUXILIARY_DIM,
    )
    assert extended.legacy_auxiliary.shape[-1] == AUXILIARY_DIM
    np.testing.assert_array_equal(extended.candidates[..., : 3 * TOP_K], legacy.candidates)
    np.testing.assert_array_equal(extended.valid[..., : 3 * TOP_K], legacy.valid)
    np.testing.assert_array_equal(extended.emitter_topk[:3], legacy.emitter_topk)
    assert extended.legacy_identity_digest == legacy.identity_digest
    assert len(extended.identity_digest) == 64
    for axis in range(2):
        for source in range(80):
            row = extended.candidates[axis, source, extended.valid[axis, source]]
            assert len(row) == len(np.unique(row))
            assert set(legacy.emitter_topk[0, axis, source]).issubset(row)
            assert set(extended.emitter_topk[3, axis, source]).issubset(row)
            assert np.count_nonzero(extended.valid[axis, source, 3 * TOP_K :]) > 0


def test_target_free_cache_constructor_rejects_labels_and_digest_drift() -> None:
    legacy = _legacy_pool()
    arrays = {
        "candidates": legacy.candidates,
        "valid": legacy.valid,
        "auxiliary": legacy.auxiliary,
        "raw_baseline": legacy.raw_baseline,
        "emitter_topk": legacy.emitter_topk,
    }
    replay = pool_from_target_free_legacy_cache(arrays)
    assert replay.identity_digest == legacy.identity_digest
    with pytest.raises(ValueError, match="only the five target-free"):
        pool_from_target_free_legacy_cache({**arrays, "target_slots": np.zeros((2, 80))})

    invalid = CandidatePool(
        candidates=legacy.candidates,
        valid=legacy.valid,
        auxiliary=legacy.auxiliary,
        raw_baseline=legacy.raw_baseline,
        emitter_topk=legacy.emitter_topk,
        identity_digest="0" * 64,
    )
    with pytest.raises(ValueError, match="identity digest"):
        extend_with_guided_emitter(invalid, (_ranked_matrix(80, 33),) * 2)


def test_fixed_guided_standalone_is_exact_frozen_half_fusion_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiles = np.zeros((576, 20, 20, 3), dtype=np.uint8)
    control = (
        np.full((576, 576), -4.0, dtype=np.float32),
        np.full((576, 576), -8.0, dtype=np.float32),
    )
    fused = (
        np.full((576, 576), -3.0, dtype=np.float32),
        np.full((576, 576), -5.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        module,
        "directional_scores",
        lambda _tiles, *, views: {"bilateral": control},
    )
    monkeypatch.setattr(
        module,
        "guided_fused_directional_scores",
        lambda _tiles, _control: fused,
    )
    right, down = fixed_guided_standalone_scores(tiles)
    np.testing.assert_array_equal(right, 2.0 * fused[0] - control[0])
    np.testing.assert_array_equal(down, 2.0 * fused[1] - control[1])


def test_guided_score_shape_and_legacy_schema_fail_closed() -> None:
    legacy = _legacy_pool()
    with pytest.raises(ValueError, match="aligned, square and finite"):
        extend_with_guided_emitter(
            legacy,
            (np.zeros((2, 2), dtype=np.float32), np.zeros((80, 80), dtype=np.float32)),
        )
    bad = CandidatePool(
        candidates=legacy.candidates[:, :, :-1],
        valid=legacy.valid[:, :, :-1],
        auxiliary=legacy.auxiliary[:, :, :-1],
        raw_baseline=legacy.raw_baseline[:, :, :-1],
        emitter_topk=legacy.emitter_topk,
        identity_digest=legacy.identity_digest,
    )
    with pytest.raises(ValueError, match="fixed tri-emitter"):
        extend_with_guided_emitter(bad, (_ranked_matrix(80, 33),) * 2)
