from __future__ import annotations

import numpy as np

from puzzle_denoise_v2.matching import (
    MatchingThresholds,
    assignment_diagnostics,
    calibration_report,
    match_tile_sets,
)


def test_two_descriptor_gate_recovers_an_exact_permutation() -> None:
    rng = np.random.default_rng(17)
    clean = rng.integers(0, 256, size=(32, 20, 20, 3), dtype=np.uint8)
    permutation = rng.permutation(len(clean)).astype(np.int32)
    result = match_tile_sets(clean[permutation], clean)

    assert np.array_equal(result.coarse.mapping, permutation)
    assert np.array_equal(result.structural.mapping, permutation)
    assert result.consensus.all()
    assert result.coarse.mutual_nn_cycle.all()
    assert result.structural.mutual_nn_cycle.all()
    assert result.selected.all()

    report = calibration_report([result], [permutation])
    assert report["stages"]["selected"]["coverage"] == 1.0
    assert report["stages"]["selected"]["precision"] == 1.0


def test_hungarian_assignment_is_not_mistaken_for_mutual_nn() -> None:
    # Hungarian must choose the cross assignment globally, but neither selected
    # edge is a bidirectional nearest-neighbour pair.
    cost = np.asarray([[0.0, 1.0], [0.1, 100.0]], dtype=np.float32)
    diagnostics = assignment_diagnostics(cost)

    assert diagnostics.mapping.tolist() == [1, 0]
    assert not diagnostics.mutual_nn_cycle.any()
    assert (diagnostics.min_margin < 0).all()


def test_tied_constant_tiles_do_not_enter_high_purity_gold() -> None:
    clean = np.full((4, 20, 20, 3), 128, dtype=np.uint8)
    result = match_tile_sets(
        clean.copy(),
        clean,
        MatchingThresholds(coarse_min_margin=1e-6, structural_min_margin=1e-6),
    )

    assert not result.selected.any()
    assert np.all(result.coarse.min_margin == 0)
    assert np.all(result.structural.min_margin == 0)

