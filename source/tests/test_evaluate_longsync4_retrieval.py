from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_longsync4_retrieval import (
    SparseHypothesisGraph,
    _canonical_measurement,
    build_sparse_hypothesis_graph,
    rerank_frozen_top2,
)
from puzzle_assembly.longsync_translation import LongSyncTranslationResult


def _prepared(
    direction: list[int], source: list[int], destination: list[int]
) -> SimpleNamespace:
    count = len(direction)
    return SimpleNamespace(
        graph=SimpleNamespace(
            direction=np.asarray(direction, dtype=np.uint8),
            source=np.asarray(source, dtype=np.int32),
            destination=np.asarray(destination, dtype=np.int32),
        ),
        labels=np.zeros(count, dtype=np.float32),
    )


def _result(corruption: list[float], supported: list[bool]) -> LongSyncTranslationResult:
    values = np.asarray(corruption, dtype=np.float64)
    support = np.asarray(supported, dtype=bool)
    return LongSyncTranslationResult(
        corruption=values,
        weights=np.exp(-values),
        support_counts=support.astype(np.int64),
        supported=support,
        alternate_paths=tuple(() for _ in values),
        corruption_history=values[None, :],
        beta_history=np.asarray([1.0]),
    )


def test_canonical_measurement_inverts_reverse_traversal() -> None:
    assert _canonical_measurement(2, 7, 0)[0] == (2, 7)
    np.testing.assert_array_equal(_canonical_measurement(2, 7, 0)[1], [1.0, 0.0])
    assert _canonical_measurement(7, 2, 0)[0] == (2, 7)
    np.testing.assert_array_equal(_canonical_measurement(7, 2, 0)[1], [-1.0, 0.0])
    np.testing.assert_array_equal(_canonical_measurement(2, 7, 1)[1], [0.0, 1.0])
    with pytest.raises(ValueError, match="self"):
        _canonical_measurement(2, 2, 0)
    with pytest.raises(ValueError, match="direction"):
        _canonical_measurement(2, 7, 3)


def test_sparse_graph_deduplicates_pair_by_probability_and_tracks_owner() -> None:
    prepared = _prepared(
        direction=[0, 0, 0, 0],
        source=[0, 0, 1, 1],
        destination=[1, 2, 0, 3],
    )
    probability = np.asarray([0.8, 0.7, 0.9, 0.6])

    sparse = build_sparse_hypothesis_graph(prepared, probability)

    np.testing.assert_array_equal(sparse.edges, [[0, 1], [0, 2], [1, 3]])
    # Reverse candidate row 2 owns pair (0,1), so its canonical displacement is -right.
    np.testing.assert_array_equal(sparse.displacements[0], [-1.0, 0.0])
    np.testing.assert_array_equal(sparse.owner_candidate_indices, [2, 1, 3])
    assert sparse.selected_candidates == 4
    assert sparse.deduplicated_candidates == 1
    assert sparse.query_top_indices == ((0, 1), (2, 3))


def test_sparse_graph_ties_use_smaller_candidate_row() -> None:
    prepared = _prepared(
        direction=[0, 0, 0, 0],
        source=[0, 0, 1, 1],
        destination=[1, 2, 0, 3],
    )
    sparse = build_sparse_hypothesis_graph(
        prepared, np.asarray([0.8, 0.7, 0.8, 0.6])
    )
    owner = dict(zip(map(tuple, sparse.edges.tolist()), sparse.owner_candidate_indices))
    assert owner[(0, 1)] == 0


def test_rerank_swaps_only_strict_supported_top2_preference() -> None:
    sparse = SparseHypothesisGraph(
        edges=np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        displacements=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        owner_candidate_indices=np.asarray([0, 1], dtype=np.int64),
        query_top_indices=((0, 1),),
        selected_candidates=2,
        deduplicated_candidates=0,
    )
    base = np.asarray([0.9, 0.7, 0.2])

    adjusted, counts = rerank_frozen_top2(
        base, sparse, _result([0.8, 0.1], [True, True])
    )

    np.testing.assert_array_equal(adjusted, [0.7, 0.9, 0.2])
    assert counts["eligible_groups"] == 1
    assert counts["swaps"] == 1
    np.testing.assert_array_equal(base, [0.9, 0.7, 0.2])


def test_rerank_preserves_base_on_tie_unsupported_or_dedup_drop() -> None:
    base = np.asarray([0.9, 0.7])
    sparse = SparseHypothesisGraph(
        edges=np.asarray([[0, 1], [0, 2]], dtype=np.int64),
        displacements=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        owner_candidate_indices=np.asarray([0, 1], dtype=np.int64),
        query_top_indices=((0, 1),),
        selected_candidates=2,
        deduplicated_candidates=0,
    )
    tied, tied_counts = rerank_frozen_top2(
        base, sparse, _result([0.1, 0.1], [True, True])
    )
    unsupported, unsupported_counts = rerank_frozen_top2(
        base, sparse, _result([0.8, 0.1], [True, False])
    )
    dropped = SparseHypothesisGraph(
        edges=np.asarray([[0, 1]], dtype=np.int64),
        displacements=np.asarray([[0.0, 1.0]]),
        owner_candidate_indices=np.asarray([0], dtype=np.int64),
        query_top_indices=((0, 1),),
        selected_candidates=2,
        deduplicated_candidates=1,
    )
    dedup, dedup_counts = rerank_frozen_top2(
        base, dropped, _result([0.8], [True])
    )

    np.testing.assert_array_equal(tied, base)
    np.testing.assert_array_equal(unsupported, base)
    np.testing.assert_array_equal(dedup, base)
    assert tied_counts["eligible_groups"] == 1 and tied_counts["swaps"] == 0
    assert unsupported_counts["unsupported_fallback_groups"] == 1
    assert dedup_counts["dedup_fallback_groups"] == 1


def test_frozen_top_k_and_probability_alignment_fail_closed() -> None:
    prepared = _prepared([0, 0], [0, 0], [1, 2])
    with pytest.raises(ValueError, match="top_k=2"):
        build_sparse_hypothesis_graph(prepared, np.asarray([0.9, 0.8]), top_k=3)
    with pytest.raises(ValueError, match="finite vector"):
        build_sparse_hypothesis_graph(prepared, np.asarray([0.9, np.nan]))
