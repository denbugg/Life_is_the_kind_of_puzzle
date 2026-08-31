from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.taska_equivariant_square import (
    DEFAULT_SQUARE_WEIGHT,
    SQUARE_ROUNDS,
    SQUARE_SHORTLIST,
    SQUARE_SUPPORT_K,
    SQUARE_TEMPERATURE,
    equivariant_square_rerank,
)


def _historical_all_rows_reference(
    right: np.ndarray,
    down: np.ndarray,
    *,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal historical calculation with only the two id masks removed."""

    def squares(
        axis: np.ndarray,
        cross: np.ndarray,
        rows: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        forward = np.argsort(-cross, axis=1)[:, :SQUARE_SUPPORT_K]
        backward = np.argsort(-cross, axis=0).T[:, :SQUARE_SUPPORT_K]
        output = np.zeros((len(rows), candidates.shape[1], 2))
        for row, source in enumerate(rows):
            targets = candidates[row]
            for side in (0, 1):
                if side == 0:
                    first, second = forward[source], forward[targets]
                    energy = (
                        cross[source, first][:, None, None]
                        + axis[first[:, None, None], second[None]]
                        + cross[targets[:, None], second][None]
                    )
                else:
                    first, second = backward[source], backward[targets]
                    energy = (
                        cross[first, source][:, None, None]
                        + axis[first[:, None, None], second[None]]
                        + cross[second, targets[:, None]][None]
                    )
                flattened = energy.transpose(1, 0, 2).reshape(len(targets), -1) / 3.0
                maximum = flattened.max(1, keepdims=True)
                output[row, :, side] = maximum[:, 0] + SQUARE_TEMPERATURE * np.log(
                    np.exp((flattened - maximum) / SQUARE_TEMPERATURE).sum(1)
                )
        return output

    def bonus(axis: np.ndarray, cross: np.ndarray) -> np.ndarray:
        rows = np.arange(len(axis))
        candidates = np.argsort(-axis, axis=1)[:, :SQUARE_SHORTLIST]
        square = squares(axis, cross, rows, candidates).sum(axis=2)
        square -= square.mean(axis=1, keepdims=True)
        addition = np.zeros_like(axis)
        np.put_along_axis(addition, candidates, weight * square, axis=1)
        return axis + addition

    horizontal = np.array(right, dtype=np.float64, copy=True)
    vertical = np.array(down, dtype=np.float64, copy=True)
    horizontal_diagonal = np.diag(horizontal).copy()
    vertical_diagonal = np.diag(vertical).copy()
    np.fill_diagonal(horizontal, -1e9)
    np.fill_diagonal(vertical, -1e9)
    for _ in range(SQUARE_ROUNDS):
        horizontal, vertical = bonus(horizontal, vertical), bonus(vertical, horizontal)
    np.fill_diagonal(horizontal, horizontal_diagonal)
    np.fill_diagonal(vertical, vertical_diagonal)
    return horizontal, vertical


def _scores(seed: int, count: int = 24) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Continuous jitter makes every relevant rank strict.
    right = rng.normal(size=(count, count)) + 1e-7 * np.arange(count)[None]
    down = rng.normal(size=(count, count)) + 2e-7 * np.arange(count)[None]
    return right, down


def test_matches_historical_centred_formula_when_ranks_are_strict() -> None:
    right, down = _scores(7)
    expected = _historical_all_rows_reference(right, down, weight=DEFAULT_SQUARE_WEIGHT)
    actual = equivariant_square_rerank(right, down)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("seed", [3, 19, 101])
def test_arbitrary_simultaneous_tile_relabelling_is_equivariant(seed: int) -> None:
    right, down = _scores(seed)
    permutation = np.random.default_rng(seed + 1_000).permutation(len(right))
    relabelled_right = right[np.ix_(permutation, permutation)]
    relabelled_down = down[np.ix_(permutation, permutation)]
    original = equivariant_square_rerank(right, down)
    relabelled = equivariant_square_rerank(relabelled_right, relabelled_down)
    np.testing.assert_allclose(
        relabelled[0],
        original[0][np.ix_(permutation, permutation)],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        relabelled[1],
        original[1][np.ix_(permutation, permutation)],
        rtol=1e-12,
        atol=1e-12,
    )


def test_cutoff_ties_use_equivariant_complete_tie_blocks() -> None:
    right, down = _scores(31)
    # Force ties across both nominal cutoffs while retaining heterogeneous
    # cross-edges, the case where id-based arbitrary tie truncation can leak.
    right = np.round(right, 0)
    down = np.round(down, 0)
    permutation = np.random.default_rng(32).permutation(len(right))
    original = equivariant_square_rerank(right, down)
    relabelled = equivariant_square_rerank(
        right[np.ix_(permutation, permutation)],
        down[np.ix_(permutation, permutation)],
    )
    np.testing.assert_allclose(
        relabelled[0],
        original[0][np.ix_(permutation, permutation)],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        relabelled[1],
        original[1][np.ix_(permutation, permutation)],
        rtol=1e-12,
        atol=1e-12,
    )


def test_every_source_row_is_eligible_and_inputs_are_not_mutated() -> None:
    right, down = _scores(43)
    before_right, before_down = right.copy(), down.copy()
    reranked_right, reranked_down = equivariant_square_rerank(right, down)
    np.testing.assert_array_equal(right, before_right)
    np.testing.assert_array_equal(down, before_down)
    np.testing.assert_array_equal(np.diag(reranked_right), np.diag(right))
    np.testing.assert_array_equal(np.diag(reranked_down), np.diag(down))
    # Row 23 was excluded as a presumed right/bottom boundary by the leaky
    # historical masks.  It is a normal input-bag row here and must change.
    assert np.any(reranked_right[23] != right[23])
    assert np.any(reranked_down[23] != down[23])


def test_zero_weight_is_exact_identity() -> None:
    right, down = _scores(47)
    actual_right, actual_down = equivariant_square_rerank(right, down, weight=0.0)
    np.testing.assert_array_equal(actual_right, right)
    np.testing.assert_array_equal(actual_down, down)


@pytest.mark.parametrize(
    ("right", "down", "message"),
    [
        (np.zeros(20), np.zeros((20, 20)), "square"),
        (np.zeros((20, 20)), np.zeros((21, 21)), "same square shape"),
        (np.zeros((19, 19)), np.zeros((19, 19)), "at least 20"),
        (
            np.pad(np.array([[np.nan]]), ((0, 19), (0, 19))),
            np.zeros((20, 20)),
            "finite",
        ),
    ],
)
def test_rejects_bad_shapes_and_nonfinite_values(
    right: np.ndarray,
    down: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        equivariant_square_rerank(right, down)


@pytest.mark.parametrize("weight", [np.nan, np.inf, -0.1, True, [0.4]])
def test_rejects_bad_weight(weight: object) -> None:
    right, down = _scores(53)
    with pytest.raises(ValueError, match="weight"):
        equivariant_square_rerank(right, down, weight=weight)  # type: ignore[arg-type]
