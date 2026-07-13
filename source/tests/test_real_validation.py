from __future__ import annotations

import numpy as np
import pytest

from puzzle_denoise_v2.metrics import tile_metrics
from puzzle_denoise_v2.real_validation import (
    ConfidenceStratum,
    confidence_stratum_masks,
    evaluate_confidence_strata,
    evaluate_real_pairs,
    paired_source_bootstrap_delta,
)


def _targets(count: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).integers(40, 181, size=(count, 20, 20, 3), dtype=np.uint8)


def _offset(tiles: np.ndarray, amount: np.ndarray | int) -> np.ndarray:
    offsets = np.asarray(amount, dtype=np.int16)
    if offsets.ndim:
        offsets = offsets[:, None, None, None]
    return np.clip(tiles.astype(np.int16) + offsets, 0, 255).astype(np.uint8)


def test_real_pair_metrics_are_macro_averaged_over_sources() -> None:
    target = _targets(4)
    source_indices = np.asarray([5, 9, 9, 9], dtype=np.uint16)
    prediction = _offset(target, np.asarray([0, 30, 30, 30]))

    evaluation = evaluate_real_pairs(
        prediction,
        target,
        source_indices,
        source_count=10,
    )
    source_five = tile_metrics(prediction[:1], target[:1])
    source_nine = tile_metrics(prediction[1:], target[1:])

    assert evaluation.pair_count == 4
    assert evaluation.source_count == 2
    assert evaluation.source_ids == (5, 9)
    assert evaluation.macro_metrics["mae"] == pytest.approx(
        0.5 * (source_five["mae"] + source_nine["mae"])
    )
    assert evaluation.micro_metrics["mae"] == pytest.approx(22.5)
    assert evaluation.macro_metrics["mae"] == pytest.approx(15.0)


def test_real_pair_validation_rejects_ambiguous_arrays_and_bad_source_indices() -> None:
    target = _targets(3)
    prediction = target.copy()

    with pytest.raises(TypeError, match="uint8"):
        evaluate_real_pairs(prediction.astype(np.float32), target, np.asarray([0, 0, 1]))
    with pytest.raises(ValueError, match="shapes differ"):
        evaluate_real_pairs(prediction[:-1], target, np.asarray([0, 0, 1]))
    with pytest.raises(ValueError, match="shape"):
        evaluate_real_pairs(prediction, target, np.asarray([0, 1]))
    with pytest.raises(TypeError, match="integer dtype"):
        evaluate_real_pairs(prediction, target, np.asarray([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_real_pairs(prediction, target, np.asarray([0, -1, 1]))
    with pytest.raises(ValueError, match="outside source_count"):
        evaluate_real_pairs(prediction, target, np.asarray([0, 1, 2]), source_count=2)


def test_paired_source_bootstrap_delta_is_deterministic_and_source_macro() -> None:
    target = _targets(6, seed=11)
    source_indices = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint16)
    baseline = _offset(target, 20)
    candidate = _offset(target, np.asarray([5, 5, 10, 10, 15, 15]))

    first = paired_source_bootstrap_delta(
        candidate,
        baseline,
        target,
        source_indices,
        metric="mae",
        resamples=1000,
        seed=123,
    )
    second = paired_source_bootstrap_delta(
        candidate,
        baseline,
        target,
        source_indices,
        metric="mae",
        resamples=1000,
        seed=123,
    )

    assert first == second
    assert first.candidate_minus_baseline == pytest.approx(-10.0)
    assert first.lower <= first.candidate_minus_baseline <= first.upper
    assert first.source_count == 3
    assert first.seed == 123


def test_non_finite_source_delta_is_rejected() -> None:
    target = _targets(2, seed=13)
    with pytest.raises(ValueError, match="non-finite"):
        paired_source_bootstrap_delta(
            target,
            target,
            target,
            np.asarray([0, 1]),
            metric="psnr",
            resamples=10,
        )


def test_explicit_confidence_strata_use_half_open_bounds() -> None:
    target = _targets(6, seed=17)
    source_indices = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint16)
    confidence = np.asarray([0.45, 0.99, 1.0, 1.49, 1.5, 2.0], dtype=np.float32)
    strata = (
        ConfidenceStratum("selected", minimum=0.45, maximum=1.0),
        ConfidenceStratum("strict", minimum=1.0, maximum=1.5),
        ConfidenceStratum("very_strict", minimum=1.5),
    )

    masks = confidence_stratum_masks(confidence, strata)
    assert {name: np.flatnonzero(mask).tolist() for name, mask in masks.items()} == {
        "selected": [0, 1],
        "strict": [2, 3],
        "very_strict": [4, 5],
    }

    evaluations = evaluate_confidence_strata(
        target,
        target,
        source_indices,
        confidence,
        strata,
        source_count=3,
    )
    assert {name: result.pair_count for name, result in evaluations.items()} == {
        "selected": 2,
        "strict": 2,
        "very_strict": 2,
    }
    assert evaluations["strict"].source_ids == (1,)


def test_empty_or_misaligned_confidence_strata_are_rejected() -> None:
    target = _targets(2, seed=19)
    source_indices = np.asarray([0, 1])

    with pytest.raises(ValueError, match="shape"):
        evaluate_confidence_strata(
            target,
            target,
            source_indices,
            np.asarray([1.0]),
            (ConfidenceStratum("strict", minimum=1.0),),
        )
    with pytest.raises(ValueError, match="empty"):
        evaluate_confidence_strata(
            target,
            target,
            source_indices,
            np.asarray([0.5, 0.6]),
            (ConfidenceStratum("strict", minimum=1.0),),
        )

