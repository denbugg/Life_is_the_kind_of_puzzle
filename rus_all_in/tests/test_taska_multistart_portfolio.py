from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, RawTailGlobalConfig
from aiijc_puzzle.taska_layout_portfolio import total_taska_adjacent_seam_cost
from aiijc_puzzle.taska_multistart_portfolio import (
    TASKA_MULTISTART_SEEDS,
    TASKA_MULTISTART_TAIL_SWAPS,
    solve_taska_multistart_portfolio,
)


def _case() -> tuple[np.ndarray, np.ndarray, tuple[RawTailEdge, ...], dict[str, np.ndarray]]:
    generator = np.random.default_rng(719)
    count = 9
    right = generator.uniform(0.0, 4.0, size=(count, count))
    down = generator.uniform(0.0, 4.0, size=(count, count))
    np.fill_diagonal(right, 100.0)
    np.fill_diagonal(down, 100.0)
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(1, 2, "right"),
        RawTailEdge(0, 3, "down"),
        RawTailEdge(3, 4, "right"),
    )
    priorities = {
        "logistic": np.asarray([0.7, 0.9, 0.3, 0.2]),
        "focal": np.asarray([0.2, 0.4, 0.8, 0.6]),
        "nonlinear": np.asarray([0.5, 0.1, 0.6, 0.9]),
    }
    return right, down, edges, priorities


def _config(seed: int) -> RawTailGlobalConfig:
    return RawTailGlobalConfig(
        baseline_quantile=0.15,
        search_rounds=2,
        border_weight=0.0,
        random_seed=seed,
        component_cap=0,
        fill_rounds=1,
    )


def test_fixed_multistart_is_strict_and_ignores_caller_seed() -> None:
    right, down, edges, priorities = _case()
    first = solve_taska_multistart_portfolio(
        right,
        down,
        edges,
        priorities,
        grid=3,
        solver_config=_config(81),
    )
    second = solve_taska_multistart_portfolio(
        right,
        down,
        edges,
        priorities,
        grid=3,
        solver_config=_config(-13),
    )

    expected_names = tuple(
        f"{arm}_seed{seed}"
        for seed in TASKA_MULTISTART_SEEDS
        for arm in ("raw", "logistic", "focal", "nonlinear")
    )
    assert tuple(name for name, _ in first.layouts) == expected_names
    assert TASKA_MULTISTART_TAIL_SWAPS == 96
    for (_, layout), (_, repeated) in zip(first.layouts, second.layouts, strict=True):
        assert np.array_equal(np.sort(layout), np.arange(9))
        assert np.array_equal(layout, repeated)
        assert not layout.flags.writeable
    assert np.array_equal(first.selection.layout, second.selection.layout)
    assert np.array_equal(first.polish.layout, second.polish.layout)
    assert np.array_equal(np.sort(first.polish.layout), np.arange(9))


def test_selection_is_minimum_original_all_bond_cost() -> None:
    right, down, edges, priorities = _case()
    result = solve_taska_multistart_portfolio(
        right,
        down,
        edges,
        priorities,
        grid=3,
        solver_config=_config(0),
    )
    costs = dict(result.selection.total_costs)
    assert result.selection.choice == min(costs, key=costs.__getitem__)
    selected = dict(result.layouts)[result.selection.choice]
    assert costs[result.selection.choice] == pytest.approx(
        total_taska_adjacent_seam_cost(selected, right, down, grid=3)
    )
    assert result.polish.diagnostics.accepted_swap_count <= TASKA_MULTISTART_TAIL_SWAPS


def test_priority_arm_contract_is_fixed() -> None:
    right, down, edges, priorities = _case()
    reordered = {
        "focal": priorities["focal"],
        "logistic": priorities["logistic"],
        "nonlinear": priorities["nonlinear"],
    }
    with pytest.raises(ValueError, match="in that order"):
        solve_taska_multistart_portfolio(
            right,
            down,
            edges,
            reordered,
            grid=3,
            solver_config=_config(0),
        )

