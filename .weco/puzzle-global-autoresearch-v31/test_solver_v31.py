import numpy as np

import solver_v31 as s


def test_stable_union_preserves_priority_and_limit():
    got = s.stable_union([500, 3, 400], [3, 2, 1], limit=4)
    assert got.tolist() == [500, 3, 400, 2]


def test_assert_permutation():
    s.assert_permutation(np.arange(s.N))
    broken = np.arange(s.N)
    broken[-1] = 0
    try:
        s.assert_permutation(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicate tile was accepted")


def test_identity_has_perfect_adjacency():
    metrics = s.v30.placement_metrics(np.arange(s.N, dtype=np.int32))
    assert metrics["adjacency"] == 1.0
    assert metrics["translation_aligned_placement"] == 1.0


def test_mutual_rank_is_finite_and_diagonal_zero():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(s.N, s.N)).astype(np.float32)
    result = s.mutual_rank_matrix(matrix)
    assert result.shape == matrix.shape
    assert np.isfinite(result).all()
    assert np.array_equal(np.diag(result), np.zeros(s.N))
