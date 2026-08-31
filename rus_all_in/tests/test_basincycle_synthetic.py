from __future__ import annotations

import numpy as np
import pytest

from src.aiijc_puzzle.basincycle_synthetic import (
    apply_cycle,
    evidence_energy,
    is_strict_permutation,
    make_synthetic_case,
    propose_cycles,
    refine_layout,
    relabel_instance,
    transpose_instance,
)


def _search_kwargs(grid_size: int) -> dict[str, int | float]:
    return {
        "grid_size": grid_size,
        "top_k": 6,
        "candidate_cap": 4,
        "seed_count": grid_size * grid_size,
        "max_cycle_length": 3,
        "max_steps": 8,
        "minimum_delta": 1e-9,
    }


def _case(seed: int = 71):
    return make_synthetic_case(
        grid_size=4,
        seed=seed,
        true_edge_score=8.0,
        false_edge_sigma=0.1,
        true_edge_noise_sigma=0.05,
        distractor_probability=0.0,
        distractor_boost=0.0,
        corruption_cycle_count=2,
        corruption_cycle_length=3,
    )


def test_closed_cycle_is_strict_and_rejects_collisions() -> None:
    layout = np.arange(9)
    moved = apply_cycle(layout, (0, 4, 8))
    assert is_strict_permutation(moved)
    assert np.array_equal(moved[[0, 4, 8]], [4, 8, 0])
    with pytest.raises(ValueError, match="distinct"):
        apply_cycle(layout, (0, 4, 0))


def test_truth_is_absorbing_keep_and_corruption_is_repaired() -> None:
    case = _case()
    truth_trace = refine_layout(
        case.truth,
        case.right_scores,
        case.down_scores,
        **_search_kwargs(4),
    )
    control_trace = refine_layout(
        case.control,
        case.right_scores,
        case.down_scores,
        **_search_kwargs(4),
    )
    assert truth_trace.chose_keep
    assert np.array_equal(truth_trace.output, case.truth)
    assert np.array_equal(control_trace.output, case.truth)
    assert all(is_strict_permutation(layout) for layout in control_trace.layouts)
    assert np.all(np.diff(control_trace.energies) > 0)


def test_relabel_equivariance() -> None:
    case = _case(73)
    mapping = np.random.default_rng(9).permutation(case.truth.size)
    layout, right, down = relabel_instance(
        case.control,
        case.right_scores,
        case.down_scores,
        mapping,
    )
    original = refine_layout(
        case.control,
        case.right_scores,
        case.down_scores,
        **_search_kwargs(4),
    )
    relabeled = refine_layout(layout, right, down, **_search_kwargs(4))
    assert len(original.moves) == len(relabeled.moves)
    assert [move.positions for move in original.moves] == [
        move.positions for move in relabeled.moves
    ]
    assert np.array_equal(relabeled.output, mapping[original.output])


def test_transpose_energy_and_refinement_equivariance() -> None:
    case = _case(75)
    layout, right, down = transpose_instance(
        case.control,
        case.right_scores,
        case.down_scores,
        grid_size=4,
    )
    truth, _, _ = transpose_instance(
        case.truth,
        case.right_scores,
        case.down_scores,
        grid_size=4,
    )
    assert evidence_energy(
        layout,
        right,
        down,
        grid_size=4,
    ) == pytest.approx(
        evidence_energy(
            case.control,
            case.right_scores,
            case.down_scores,
            grid_size=4,
        )
    )
    transposed = refine_layout(layout, right, down, **_search_kwargs(4))
    assert np.array_equal(transposed.output, truth)


def test_protected_slot_never_enters_a_proposal_or_changes() -> None:
    case = _case(77)
    protected = frozenset({case.corruption_cycles[0][0]})
    proposals = propose_cycles(
        case.control,
        case.right_scores,
        case.down_scores,
        grid_size=4,
        top_k=6,
        candidate_cap=4,
        seed_count=16,
        max_cycle_length=3,
        protected_positions=protected,
    )
    assert proposals
    assert all(protected.isdisjoint(cycle) for cycle in proposals)
    trace = refine_layout(
        case.control,
        case.right_scores,
        case.down_scores,
        protected_positions=protected,
        **_search_kwargs(4),
    )
    protected_position = next(iter(protected))
    assert all(
        layout[protected_position] == case.control[protected_position]
        for layout in trace.layouts
    )
