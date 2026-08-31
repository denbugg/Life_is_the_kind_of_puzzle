from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import aiijc_puzzle.union_component_pose_packer as pose_module
from aiijc_puzzle.socket_decoder import (
    PartialAxisMatching,
    SocketEdge,
    TranslationComponentBuild,
)
from aiijc_puzzle.union_component_pose_packer import (
    ComponentPosePackerConfig,
    ComponentPoseProblem,
    InterComponentPoseFactor,
    NonWrappingRigidComponent,
    audit_nonwrapping_layout,
    build_union_component_pose_problem,
    ordinary_feasible_origins,
    pack_union_component_poses,
    solve_component_pose_exact_cover,
)


def _perfect_assignment(layout: np.ndarray, *, grid: int, axis: str) -> np.ndarray:
    count = grid * grid
    board = np.asarray(layout).reshape(grid, grid)
    value = np.full((count + 1, count + 1), -20.0, dtype=np.float64)
    value[count, count] = -1e4
    if axis == "right":
        for row in range(grid):
            value[count, board[row, 0]] = 0.0
            value[board[row, -1], count] = 0.0
            for column in range(grid - 1):
                value[board[row, column], board[row, column + 1]] = 0.0
    elif axis == "down":
        for column in range(grid):
            value[count, board[0, column]] = 0.0
            value[board[-1, column], count] = 0.0
            for row in range(grid - 1):
                value[board[row, column], board[row + 1, column]] = 0.0
    else:
        raise ValueError(axis)
    return value


def _singleton_problem(
    anchor: np.ndarray,
    factors: tuple[InterComponentPoseFactor, ...],
    *,
    grid: int,
) -> ComponentPoseProblem:
    fragments = tuple(
        NonWrappingRigidComponent((tile,), (0,), (0,))
        for tile in range(grid * grid)
    )
    positions = np.empty(grid * grid, dtype=np.int32)
    positions[anchor] = np.arange(grid * grid, dtype=np.int32)
    return ComponentPoseProblem(
        grid=grid,
        fragments=fragments,
        factors=factors,
        anchor_layout=anchor,
        anchor_origins=positions,
        raw_seed_component_count=len(fragments),
        split_seed_component_count=0,
        seed_status_counts=(),
        selected_factor_edges_right=0,
        selected_factor_edges_down=0,
    )


def _grid2_config() -> ComponentPosePackerConfig:
    return ComponentPosePackerConfig(
        seed_edge_budget_per_axis=1,
        factor_edge_cap_per_axis=2,
        lexicographic_scale=5,
        milp_time_limit_seconds=5.0,
        milp_relative_gap=0.0,
    )


def test_ordinary_feasible_origins_exclude_every_wrapping_pose() -> None:
    fragment = NonWrappingRigidComponent(
        tiles=(0, 1, 2),
        relative_rows=(0, 0, 1),
        relative_columns=(0, 1, 0),
    )
    assert ordinary_feasible_origins(fragment, grid=3) == (0, 1, 3, 4)


def test_ordinary_audit_rejects_a_cyclically_wrapped_component() -> None:
    fragments = (
        NonWrappingRigidComponent((0, 1), (0, 0), (0, 1)),
        NonWrappingRigidComponent((2,), (0,), (0,)),
        NonWrappingRigidComponent((3,), (0,), (0,)),
    )
    audit = audit_nonwrapping_layout(np.asarray([1, 2, 3, 0]), fragments, grid=2)
    assert audit.strict_permutation
    assert not audit.rigidity_preserved
    assert audit.component_origins[0] is None


def test_split_rebuilds_component_ids_and_canonical_factor_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    right_edges = (
        SocketEdge(1, 0, 0, 1, 4.0, "right"),
        SocketEdge(2, 3, 0, 1, 3.0, "right"),
    )
    down_edges = (
        SocketEdge(3, 1, 1, 0, 2.0, "down"),
        SocketEdge(0, 2, 1, 0, 1.0, "down"),
    )
    matching_right = PartialAxisMatching(right_edges, (), ())
    matching_down = PartialAxisMatching(down_edges, (), ())
    component_build = TranslationComponentBuild(
        right_matching=matching_right,
        down_matching=matching_down,
        component_edges=(),
        decisions=(),
        components=(
            {0: (0, 0), 1: (0, 1)},
            {2: (0, 0)},
            {3: (0, 0)},
        ),
        status_counts={
            "added": 1,
            "consistent": 0,
            "contradiction": 0,
            "collision": 0,
            "span": 0,
        },
    )
    monkeypatch.setattr(
        pose_module,
        "build_translation_components",
        lambda *args, **kwargs: component_build,
    )
    priorities = {
        "right": np.zeros((4, 4), dtype=np.float64),
        "down": np.zeros((4, 4), dtype=np.float64),
    }
    priorities["right"][1, 0] = 9.0
    priorities["right"][2, 3] = 8.0
    priorities["down"][3, 1] = 7.0
    priorities["down"][0, 2] = 6.0
    anchor = np.asarray([1, 0, 2, 3], dtype=np.int32)
    problem = build_union_component_pose_problem(
        np.empty((0, 0)),
        np.empty((0, 0)),
        priorities,
        anchor,
        grid=2,
        config=_grid2_config(),
    )

    assert problem.split_seed_component_count == 1
    assert tuple(fragment.tiles for fragment in problem.fragments) == (
        (0,),
        (1,),
        (2,),
        (3,),
    )
    reversed_factor = next(
        factor
        for factor in problem.factors
        if (factor.first_component, factor.second_component) == (0, 1)
    )
    assert (reversed_factor.delta_row, reversed_factor.delta_column) == (0, -1)
    assert reversed_factor.weight == 2  # filtered rank 1 => cap + 1 - rank
    assert reversed_factor.evidence[0].rank == 1
    assert problem.selected_factor_edges_right == 2
    assert problem.selected_factor_edges_down == 2


def test_planted_factors_move_a_wrong_anchor_to_the_unique_ordinary_square() -> None:
    anchor = np.asarray([1, 0, 2, 3], dtype=np.int32)
    factors = (
        InterComponentPoseFactor(0, 1, 0, 1, 1),
        InterComponentPoseFactor(0, 2, 1, 0, 1),
        InterComponentPoseFactor(1, 3, 1, 0, 1),
        InterComponentPoseFactor(2, 3, 0, 1, 1),
    )
    result = solve_component_pose_exact_cover(
        _singleton_problem(anchor, factors, grid=2),
        config=_grid2_config(),
    )
    assert not result.diagnostics.used_fallback
    assert result.diagnostics.milp_status == 0
    assert np.array_equal(result.layout, np.arange(4))
    assert result.diagnostics.candidate_satisfied_factor_weight == 4
    assert result.diagnostics.anchor_satisfied_factor_weight == 1
    assert result.diagnostics.candidate_objective > result.diagnostics.anchor_objective
    assert result.audit.strict_permutation
    assert result.audit.rigidity_preserved


def test_nonoptimal_or_missing_incumbent_fails_closed_to_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = np.asarray([1, 0, 2, 3], dtype=np.int32)
    problem = _singleton_problem(
        anchor,
        (InterComponentPoseFactor(0, 1, 0, 1, 1),),
        grid=2,
    )
    monkeypatch.setattr(
        pose_module,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(
            status=1,
            message="Time limit reached",
            x=None,
            mip_gap=None,
        ),
    )
    result = solve_component_pose_exact_cover(problem, config=_grid2_config())
    assert result.diagnostics.used_fallback
    assert result.diagnostics.fallback_reason == "milp-nonoptimal-status-1"
    assert np.array_equal(result.layout, anchor)


def test_integral_but_infeasible_incumbent_is_independently_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = np.asarray([1, 0, 2, 3], dtype=np.int32)
    problem = _singleton_problem(anchor, (), grid=2)

    def zero_incumbent(c, **kwargs):
        return SimpleNamespace(
            status=0,
            message="fabricated",
            x=np.zeros_like(c),
            mip_gap=0.0,
        )

    monkeypatch.setattr(pose_module, "milp", zero_incumbent)
    result = solve_component_pose_exact_cover(problem, config=_grid2_config())
    assert result.diagnostics.used_fallback
    assert result.diagnostics.fallback_reason == "milp-component-cardinality-failure"
    assert np.array_equal(result.layout, anchor)


def test_end_to_end_perfect_board_stays_strict_and_target_free() -> None:
    grid = 3
    anchor = np.random.default_rng(93).permutation(grid * grid).astype(np.int32)
    right = _perfect_assignment(anchor, grid=grid, axis="right")
    down = _perfect_assignment(anchor, grid=grid, axis="down")
    priorities = {
        "right": right[: grid * grid, : grid * grid],
        "down": down[: grid * grid, : grid * grid],
    }
    config = ComponentPosePackerConfig(
        seed_edge_budget_per_axis=1,
        factor_edge_cap_per_axis=1,
        lexicographic_scale=10,
        milp_time_limit_seconds=5.0,
        milp_relative_gap=0.0,
    )
    result = pack_union_component_poses(
        right,
        down,
        priorities,
        anchor,
        grid=grid,
        config=config,
    )
    assert not result.diagnostics.used_fallback
    assert np.array_equal(result.layout, anchor)
    assert result.audit.strict_permutation
    assert result.audit.rigidity_preserved
    assert len(result.factors) > 0

    parameters = set(inspect.signature(pack_union_component_poses).parameters)
    assert parameters.isdisjoint(
        {"target", "reference", "clean_tiles", "source_filename", "input_tile_to_position"}
    )


def test_config_pins_the_grid24_lexicographic_constant() -> None:
    config = ComponentPosePackerConfig()
    config.validate(grid=24)
    assert config.lexicographic_scale == 577
    with pytest.raises(ValueError, match="exceed the tile count"):
        ComponentPosePackerConfig(lexicographic_scale=576).validate(grid=24)


@pytest.mark.parametrize(
    "invalid_layout, message",
    [
        (np.asarray([0.0, 1.0, 2.0, 2.75]), "exact integers"),
        (np.asarray([0, 1, 2, 2**40], dtype=np.int64), r"values must be in \[0, 3\]"),
    ],
)
def test_anchor_validation_rejects_before_any_narrowing_cast(
    invalid_layout: np.ndarray,
    message: str,
) -> None:
    fragments = tuple(
        NonWrappingRigidComponent((tile,), (0,), (0,)) for tile in range(4)
    )
    with pytest.raises(ValueError, match=message):
        ComponentPoseProblem(
            grid=2,
            fragments=fragments,
            factors=(),
            anchor_layout=invalid_layout,
            anchor_origins=np.arange(4),
            raw_seed_component_count=4,
            split_seed_component_count=0,
            seed_status_counts=(),
            selected_factor_edges_right=0,
            selected_factor_edges_down=0,
        )
    audit = audit_nonwrapping_layout(invalid_layout, fragments, grid=2)
    assert not audit.strict_permutation
    assert not audit.rigidity_preserved


def test_resource_guard_fails_closed_before_optimizer_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = np.asarray([1, 0, 2, 3], dtype=np.int32)
    problem = _singleton_problem(anchor, (), grid=2)
    monkeypatch.setattr(
        pose_module,
        "milp",
        lambda *args, **kwargs: pytest.fail("resource guard must run before milp"),
    )
    config = replace(_grid2_config(), max_placement_variables=3)
    result = solve_component_pose_exact_cover(problem, config=config)
    assert result.diagnostics.used_fallback
    assert result.diagnostics.fallback_reason is not None
    assert result.diagnostics.fallback_reason.startswith("model-resource-limit:")
    assert np.array_equal(result.layout, anchor)
    assert result.diagnostics.placement_variable_count == 0


def test_unexpected_model_build_error_also_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = np.asarray([1, 0, 2, 3], dtype=np.int32)
    problem = _singleton_problem(anchor, (), grid=2)

    def fail_build(*args, **kwargs):
        raise MemoryError("synthetic allocation failure")

    monkeypatch.setattr(pose_module, "_build_milp_model", fail_build)
    result = solve_component_pose_exact_cover(problem, config=_grid2_config())
    assert result.diagnostics.used_fallback
    assert result.diagnostics.fallback_reason is not None
    assert result.diagnostics.fallback_reason.startswith("model-build-error:MemoryError:")
    assert np.array_equal(result.layout, anchor)
