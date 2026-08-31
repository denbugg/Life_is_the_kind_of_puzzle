from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import (
    RawTailEdge,
    RawTailGlobalConfig,
    solve_raw_tail_global,
)
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost
from aiijc_puzzle.taska_monotone_components import (
    MONOTONE_ARM_NAMES,
    MONOTONE_TAIL_MAX_SWAPS,
    solve_monotone_raw_tail_global,
    solve_taska_monotone_component_portfolio,
)


def _case() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[RawTailEdge, ...],
    dict[str, np.ndarray],
]:
    generator = np.random.default_rng(2_026_08_31)
    count = 16
    right = generator.uniform(0.0, 4.0, size=(count, count))
    down = generator.uniform(0.0, 4.0, size=(count, count))
    np.fill_diagonal(right, 100.0)
    np.fill_diagonal(down, 100.0)
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(2, 3, "right"),
        RawTailEdge(4, 5, "down"),
        RawTailEdge(6, 7, "down"),
        RawTailEdge(8, 9, "right"),
        RawTailEdge(10, 11, "down"),
        RawTailEdge(12, 13, "right"),
        RawTailEdge(14, 15, "down"),
    )
    priorities = {
        "logistic": generator.normal(size=len(edges)),
        "focal": generator.normal(size=len(edges)),
        "nonlinear": generator.normal(size=len(edges)),
    }
    return right, down, edges, priorities


def _config(*, rounds: int, seed: int = 0) -> RawTailGlobalConfig:
    return RawTailGlobalConfig(
        baseline_quantile=0.15,
        search_rounds=rounds,
        border_weight=0.0,
        random_seed=seed,
        component_cap=0,
        fill_rounds=1,
    )


def test_zero_rounds_exactly_replays_historical_initial_placement_and_fill() -> None:
    right, down, edges, _ = _case()
    historical = solve_raw_tail_global(
        right,
        down,
        edges,
        grid=4,
        config=_config(rounds=0),
    )
    monotone = solve_monotone_raw_tail_global(
        right,
        down,
        edges,
        grid=4,
        config=_config(rounds=0),
    )

    assert np.array_equal(monotone.solver.layout, historical.layout)
    assert monotone.solver.components == historical.components
    assert monotone.solver.decisions == historical.decisions
    assert monotone.solver.diagnostics == historical.diagnostics
    assert monotone.placement.moved_components_per_round == ()
    assert monotone.placement.pair_relocation_attempts == 0


def test_fixed_four_arm_portfolio_is_strict_deterministic_and_seed_zero() -> None:
    right, down, edges, priorities = _case()
    first = solve_taska_monotone_component_portfolio(
        right,
        down,
        edges,
        priorities,
        grid=4,
        solver_config=_config(rounds=6, seed=91),
    )
    repeated = solve_taska_monotone_component_portfolio(
        right,
        down,
        edges,
        priorities,
        grid=4,
        solver_config=_config(rounds=6, seed=-17),
    )

    assert tuple(name for name, _ in first.layouts) == MONOTONE_ARM_NAMES
    assert tuple(name for name, _ in first.placement_traces) == MONOTONE_ARM_NAMES
    assert all(trace.pair_relocation_attempts == 0 for _, trace in first.placement_traces)
    for (_, layout), (_, replayed) in zip(first.layouts, repeated.layouts, strict=True):
        assert np.array_equal(np.sort(layout), np.arange(16))
        assert np.array_equal(layout, replayed)
        assert not layout.flags.writeable
    assert np.array_equal(first.selection.layout, repeated.selection.layout)
    assert np.array_equal(first.polish.layout, repeated.polish.layout)
    assert first.polish.diagnostics.accepted_swap_count <= MONOTONE_TAIL_MAX_SWAPS


def test_portfolio_selection_uses_original_all_bond_cost() -> None:
    right, down, edges, priorities = _case()
    result = solve_taska_monotone_component_portfolio(
        right,
        down,
        edges,
        priorities,
        grid=4,
        solver_config=_config(rounds=6),
    )
    costs = dict(result.selection.total_costs)
    assert result.selection.choice == min(costs, key=costs.__getitem__)
    assert costs[result.selection.choice] == pytest.approx(
        total_taska_adjacent_seam_cost(
            dict(result.layouts)[result.selection.choice],
            right,
            down,
            grid=4,
        )
    )


def test_priority_arm_contract_is_fixed() -> None:
    right, down, edges, priorities = _case()
    reordered = {
        "focal": priorities["focal"],
        "logistic": priorities["logistic"],
        "nonlinear": priorities["nonlinear"],
    }
    with pytest.raises(ValueError, match="in that order"):
        solve_taska_monotone_component_portfolio(
            right,
            down,
            edges,
            reordered,
            grid=4,
            solver_config=_config(rounds=6),
        )
