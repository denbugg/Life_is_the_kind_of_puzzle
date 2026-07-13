import numpy as np

from puzzle_assembly.compatibility import CompatibilityMatrices
from scripts.evaluate_dual_lambdarank_qap_diagnostic import (
    DUAL_MISSING_NEUTRAL_COST,
    fuse_with_neutral_dual,
    gate,
    summarize,
)


def _record(index: int, delta: float) -> dict:
    return {
        "name": f"img_{index}.png",
        "baseline_ssim": 0.2,
        "candidate_ssim": 0.2 + delta,
    }


def test_gate_passes_only_material_actual_input_gain():
    summary = summarize([_record(index, 0.004) for index in range(8)])
    assert gate(summary)["passed"] is True


def test_gate_rejects_too_few_wins():
    records = [_record(index, 0.008 if index < 5 else -0.001) for index in range(8)]
    result = gate(summarize(records))
    assert result["checks"]["wins_ge_6"] is False
    assert result["passed"] is False


def test_gate_rejects_large_single_regression():
    records = [_record(index, 0.006 if index else -0.011) for index in range(8)]
    result = gate(summarize(records))
    assert result["checks"]["no_ssim_delta_below_minus_0_01"] is False


def test_neutral_missing_channel_is_tile_id_permutation_equivariant():
    rng = np.random.default_rng(4)
    # Use exactly unique row values so this test isolates the neutral missing
    # channel rather than rank_normalize's documented stable tie-break rule.
    values = np.arange(576, dtype=np.float32)
    first = np.stack([np.roll(values, row) for row in range(576)])
    second = np.stack([np.roll(values, 3 * row + 1) for row in range(576)])
    dual = np.full((576, 576), DUAL_MISSING_NEUTRAL_COST, dtype=np.float32)
    np.fill_diagonal(first, np.inf)
    np.fill_diagonal(second, np.inf)
    np.fill_diagonal(dual, np.inf)
    base = fuse_with_neutral_dual(
        CompatibilityMatrices("c1", first, first.copy()),
        CompatibilityMatrices("hbt", second, second.copy()),
        CompatibilityMatrices("dual", dual, dual.copy()),
    ).right
    permutation = rng.permutation(576)
    inverse = np.argsort(permutation)
    permuted = fuse_with_neutral_dual(
        CompatibilityMatrices("c1", first[np.ix_(permutation, permutation)], first[np.ix_(permutation, permutation)]),
        CompatibilityMatrices("hbt", second[np.ix_(permutation, permutation)], second[np.ix_(permutation, permutation)]),
        CompatibilityMatrices("dual", dual[np.ix_(permutation, permutation)], dual[np.ix_(permutation, permutation)]),
    ).right
    restored = permuted[np.ix_(inverse, inverse)]
    assert np.allclose(base, restored, equal_nan=True)
