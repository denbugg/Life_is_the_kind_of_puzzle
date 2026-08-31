from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from aiijc_puzzle.union_hard_edge_priority import FEATURE_NAMES, UnionHardEdgeBoard

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_learned_membership_rank_delta_composition_opened64.py"
    specification = importlib.util.spec_from_file_location(
        "run_learned_membership_rank_delta_composition_opened64_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _board() -> UnionHardEdgeBoard:
    per_axis = runner.HARD_EDGES_PER_AXIS
    source_axis = np.arange(per_axis, dtype=np.int32)
    target_axis = (source_axis + 1) % runner.COUNT
    source = np.concatenate((source_axis, source_axis))
    target = np.concatenate((target_axis, target_axis))
    axis = np.repeat(np.arange(2, dtype=np.int64), per_axis)
    base_axis = np.linspace(2.0, 1.0, per_axis, dtype=np.float32)
    base = np.concatenate((base_axis, base_axis))
    values = np.zeros((runner.HARD_EDGE_COUNT, len(FEATURE_NAMES)), dtype=np.float32)
    delta_index = FEATURE_NAMES.index(runner.DIRECT_DELTA_FEATURE)
    learned_quality_index = FEATURE_NAMES.index(runner.DIRECT_LEARNED_QUALITY_FEATURE)
    present_index = FEATURE_NAMES.index(runner.DIRECT_PRESENT_FEATURE)
    values[:, present_index] = 1.0
    values[200, delta_index] = 1.0
    values[per_axis + 300, delta_index] = 1.0
    values[200, learned_quality_index] = 1.0
    values[per_axis + 300, learned_quality_index] = 1.0
    return UnionHardEdgeBoard(
        values=torch.from_numpy(values),
        base_priority=torch.from_numpy(base),
        priority_scale=torch.ones(runner.HARD_EDGE_COUNT),
        axis=torch.from_numpy(axis),
        source=source,
        target=target,
        grid=runner.GRID,
        edge_budget_per_axis=runner.DECODER_EDGE_BUDGET,
        direct_matches_per_axis=(per_axis, per_axis),
        fullres_supported_per_axis=(0, 0),
    )


def _gate_metrics(*, exact: float, adjacency: float, fixed: float) -> dict:
    return {
        "membership_rank_composition_vs_rank_delta_transfer": {
            "exact_tiles_delta": {"mean": exact},
            "adjacency_delta": {"mean": adjacency},
            "fixed_top288_correct_delta": {"mean": fixed},
        }
    }


def test_pinned_learned_artifact_hashes_match_final_pilot() -> None:
    runner._validate_pinned_learned_artifacts(
        runner.LEARNED_PILOT_CONFIG,
        runner.LEARNED_OUTPUT,
    )


def test_embedded_rank_delta_preserves_union_multiset_and_changes_cutoff() -> None:
    board = _board()
    result = runner._rank_delta_priority_from_board(board)

    for axis_index, name in ((0, "right"), (1, "down")):
        selected = result.axis == axis_index
        assert np.array_equal(
            np.sort(result.scores[selected]),
            np.sort(result.base_priority[selected]),
        )
        assert np.array_equal(
            result.component_edge_priority[name][result.source[selected], result.target[selected]],
            result.scores[selected],
        )
    assert result.changed_membership_per_axis == (1, 1)
    assert result.matched_per_axis == (
        runner.HARD_EDGES_PER_AXIS,
        runner.HARD_EDGES_PER_AXIS,
    )


def test_embedded_rank_delta_rejects_delta_without_direct_identity() -> None:
    board = _board()
    values = board.values.clone()
    values[0, FEATURE_NAMES.index(runner.DIRECT_PRESENT_FEATURE)] = 0.0
    bad = UnionHardEdgeBoard(
        values=values,
        base_priority=board.base_priority,
        priority_scale=board.priority_scale,
        axis=board.axis,
        source=board.source,
        target=board.target,
        grid=board.grid,
        edge_budget_per_axis=board.edge_budget_per_axis,
        direct_matches_per_axis=board.direct_matches_per_axis,
        fullres_supported_per_axis=board.fullres_supported_per_axis,
    )
    values[0, FEATURE_NAMES.index(runner.DIRECT_DELTA_FEATURE)] = 0.25

    with pytest.raises(ValueError, match="embedded rank-delta contract"):
        runner._rank_delta_priority_from_board(bad)


def test_learned_scores_are_aligned_by_explicit_edge_identity() -> None:
    board = _board()
    learned = torch.arange(runner.HARD_EDGE_COUNT, dtype=torch.float32)
    order = np.concatenate(
        (
            np.arange(runner.HARD_EDGES_PER_AXIS - 1, -1, -1),
            np.arange(
                runner.HARD_EDGE_COUNT - 1,
                runner.HARD_EDGES_PER_AXIS - 1,
                -1,
            ),
        )
    )
    axis = np.asarray(board.axis.numpy(), dtype=np.int8)[order]
    observed = runner._aligned_learned_scores(
        board,
        learned,
        board.source[order],
        board.target[order],
        axis,
    )

    assert np.array_equal(observed, np.arange(runner.HARD_EDGE_COUNT)[order])


def test_fixed_top288_scores_exactly_144_edges_per_axis() -> None:
    reference = np.arange(runner.COUNT, dtype=np.int32)
    archive: dict[str, np.ndarray] = {}
    prefix = "case_0000"
    for axis in (0, 1):
        source = np.arange(runner.HARD_EDGES_PER_AXIS, dtype=np.int32)
        target = source + (1 if axis == 0 else runner.GRID)
        priority = np.arange(runner.HARD_EDGES_PER_AXIS, 0, -1, dtype=np.float64)
        archive[f"{prefix}__axis_{axis}_source"] = source
        archive[f"{prefix}__axis_{axis}_target"] = target
        archive[f"{prefix}__axis_{axis}_union_v2_priority"] = priority

    observed = runner._fixed_top288_correct(
        archive,
        prefix,
        reference,
        arm="union_v2",
    )

    # Six right edges cross a row boundary among positions 0..143; every
    # selected down edge is correct.
    assert observed == (144 - 6) + 144


def test_gate_accepts_nonnegative_exact_and_fixed_but_requires_positive_adjacency() -> None:
    passing = runner.evaluate_gate(
        _gate_metrics(exact=0.0, adjacency=1e-9, fixed=0.0),
        strict_layouts=4 * runner.EXPECTED_SOURCES,
        case_count=runner.EXPECTED_SOURCES,
    )
    zero_adjacency = runner.evaluate_gate(
        _gate_metrics(exact=0.0, adjacency=0.0, fixed=0.0),
        strict_layouts=4 * runner.EXPECTED_SOURCES,
        case_count=runner.EXPECTED_SOURCES,
    )

    assert passing["pass"] is True
    assert zero_adjacency["pass"] is False
    assert zero_adjacency["checks"]["adjacency_strictly_positive_vs_rank_delta"]["pass"] is False


@pytest.mark.parametrize(
    ("exact", "adjacency", "fixed", "strict", "cases", "failed_check"),
    (
        (-0.01, 0.01, 0.0, 256, 64, "exact_nonnegative_vs_rank_delta"),
        (0.0, 0.01, -0.01, 256, 64, "fixed_top288_nonnegative_vs_rank_delta"),
        (0.0, 0.01, 0.0, 255, 64, "all_four_arms_strict"),
        (0.0, 0.01, 0.0, 4, 1, "complete_opened_fresh64_roster"),
    ),
)
def test_gate_fails_closed(
    exact: float,
    adjacency: float,
    fixed: float,
    strict: int,
    cases: int,
    failed_check: str,
) -> None:
    result = runner.evaluate_gate(
        _gate_metrics(exact=exact, adjacency=adjacency, fixed=fixed),
        strict_layouts=strict,
        case_count=cases,
    )

    assert result["pass"] is False
    assert result["checks"][failed_check]["pass"] is False


def test_parse_args_defaults_to_mps_and_separate_freeze_mode() -> None:
    args = runner.parse_args(["freeze"])

    assert args.mode == "freeze"
    assert args.device == "mps"
    assert args.limit == runner.EXPECTED_SOURCES
    assert args.allow_nondeterministic_mps is False


def test_score_requires_prior_freeze(tmp_path: Path) -> None:
    args = SimpleNamespace(
        output_dir=tmp_path,
        rank_config=runner.RANK_CONFIG,
        learned_output=runner.LEARNED_OUTPUT,
    )

    with pytest.raises(FileNotFoundError, match="complete prior target-free freeze"):
        runner._validate_freeze(args, runner._paths(tmp_path))
