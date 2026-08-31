from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_learned_no_qap_opened64.py"
    specification = importlib.util.spec_from_file_location(
        "run_learned_no_qap_opened64_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _true_hard_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = runner.GRID
    right_source = np.asarray(
        [row * grid + column for row in range(grid) for column in range(grid - 1)],
        dtype=np.int32,
    )
    down_source = np.asarray(
        [row * grid + column for row in range(grid - 1) for column in range(grid)],
        dtype=np.int32,
    )
    source = np.concatenate((right_source, down_source))
    target = np.concatenate((right_source + 1, down_source + grid))
    axis = np.repeat(np.arange(2, dtype=np.int8), runner.HARD_EDGES_PER_AXIS)
    return source, target, axis


def _gate_metrics(
    *,
    exact: float = 0.0,
    adjacency: float = 1e-6,
    pairs: float = 1.0,
    rank_exact: float = 0.0,
) -> dict:
    return {
        "learned_no_qap_vs_learned_standard": {
            "exact_tiles_delta": {"mean": exact},
            "adjacency_delta": {"mean": adjacency},
            "satisfied_pairs_delta": {"mean": pairs},
        },
        "learned_no_qap_vs_rank_delta_transfer": {
            "exact_tiles_delta": {"mean": rank_exact},
        },
    }


def test_cli_defaults_to_frozen_mps_engineering_replay() -> None:
    args = runner.parse_args(["freeze"])

    assert args.device == "mps"
    assert args.allow_nondeterministic_mps is False
    assert args.limit == runner.EXPECTED_SOURCES
    assert args.output_dir == runner.DEFAULT_OUTPUT


def test_decoder_contract_is_exactly_qap24_edge144_cyclic5() -> None:
    runner._validate_decoder_contract()

    assert runner.STANDARD_SWAP_STEPS == 24
    assert runner.DECODER_EDGE_BUDGET == 144
    assert runner.CYCLIC_BORDER_WEIGHT == 5.0


def test_no_qap_decoder_pins_only_swap_steps_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}
    identity = np.arange(runner.COUNT, dtype=np.int32)

    def fake_decode(
        right: np.ndarray,
        down: np.ndarray,
        *,
        grid: int,
        config: object,
        component_edge_priority: object,
    ) -> SimpleNamespace:
        observed["grid"] = grid
        observed["config"] = config
        observed["priority"] = component_edge_priority
        return SimpleNamespace(layout=identity)

    def fake_cyclic(
        layout: np.ndarray,
        right: np.ndarray,
        down: np.ndarray,
        *,
        grid: int,
        config: object,
    ) -> SimpleNamespace:
        observed["cyclic_grid"] = grid
        observed["cyclic_config"] = config
        return SimpleNamespace(layout=layout)

    monkeypatch.setattr(runner, "decode_socket_assignments", fake_decode)
    monkeypatch.setattr(runner, "select_global_cyclic_translation", fake_cyclic)
    priority = {
        "right": np.zeros((runner.COUNT, runner.COUNT)),
        "down": np.zeros((runner.COUNT, runner.COUNT)),
    }

    result = runner._decode_learned_no_qap(
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        component_edge_priority=priority,
    )

    config = observed["config"]
    cyclic = observed["cyclic_config"]
    assert np.array_equal(result, identity)
    assert config.component_edge_budget_per_axis == runner.DECODER_EDGE_BUDGET
    assert config.swap_edge_budget_per_axis == runner.DECODER_EDGE_BUDGET
    assert config.max_swap_steps == 0
    assert cyclic.border_weight == runner.CYCLIC_BORDER_WEIGHT
    assert observed["priority"] is priority


def test_own_top288_satisfied_counts_selected_relations_without_target() -> None:
    source, target, axis = _true_hard_edges()
    base_axis = np.arange(
        runner.HARD_EDGES_PER_AXIS,
        0,
        -1,
        dtype=np.float64,
    )
    priority = np.concatenate((base_axis, base_axis))

    observed = runner._own_top288_satisfied(
        source,
        target,
        axis,
        priority,
        priority,
        np.arange(runner.COUNT, dtype=np.int32),
    )

    assert observed == 2 * runner.DECODER_EDGE_BUDGET


def test_fixed_top288_uses_frozen_priority_and_reference() -> None:
    source, target, _ = _true_hard_edges()
    base_axis = np.arange(
        runner.HARD_EDGES_PER_AXIS,
        0,
        -1,
        dtype=np.float64,
    )
    archive: dict[str, np.ndarray] = {}
    prefix = "case_0000"
    for axis_index in (0, 1):
        selected = slice(
            axis_index * runner.HARD_EDGES_PER_AXIS,
            (axis_index + 1) * runner.HARD_EDGES_PER_AXIS,
        )
        archive[f"{prefix}__axis_{axis_index}_source"] = source[selected]
        archive[f"{prefix}__axis_{axis_index}_target"] = target[selected]
        archive[f"{prefix}__axis_{axis_index}_union_v2_priority"] = base_axis
        archive[f"{prefix}__axis_{axis_index}_learned_no_qap_priority"] = base_axis

    observed = runner._fixed_top288_correct(
        archive,
        prefix,
        np.arange(runner.COUNT, dtype=np.int32),
        arm="learned_no_qap",
    )

    assert observed == 2 * runner.DECODER_EDGE_BUDGET


def test_satisfied_pairs_requires_an_exact_1104_edge_fraction() -> None:
    assert runner._satisfied_pairs(138 / runner.HARD_EDGE_COUNT) == 138
    with pytest.raises(ValueError, match="satisfied-pair fraction"):
        runner._satisfied_pairs(0.123456789)


def test_gate_requires_pair_adjacency_exact_rank_and_complete_strictness() -> None:
    passing = runner.evaluate_gate(
        _gate_metrics(),
        strict_layouts=4 * runner.EXPECTED_SOURCES,
        case_count=runner.EXPECTED_SOURCES,
    )
    zero_pairs = runner.evaluate_gate(
        _gate_metrics(pairs=0.0),
        strict_layouts=4 * runner.EXPECTED_SOURCES,
        case_count=runner.EXPECTED_SOURCES,
    )
    below_rank = runner.evaluate_gate(
        _gate_metrics(rank_exact=-1e-9),
        strict_layouts=4 * runner.EXPECTED_SOURCES,
        case_count=runner.EXPECTED_SOURCES,
    )
    incomplete = runner.evaluate_gate(
        _gate_metrics(),
        strict_layouts=4,
        case_count=1,
    )

    assert passing["pass"] is True
    assert zero_pairs["pass"] is False
    assert below_rank["pass"] is False
    assert incomplete["pass"] is False


def test_runtime_commitment_pins_runner_and_upstream_dependency() -> None:
    records = runner._runtime_input_records(
        rank_config=runner.RANK_CONFIG,
        learned_config=runner.LEARNED_CONFIG,
        learned_output=runner.LEARNED_OUTPUT,
        manifest=runner.DEFAULT_MANIFEST,
    )

    assert "runner" in records
    assert "upstream_composition_runner" in records
    assert "socket_decoder" in records
    assert all(len(record["sha256"]) == 64 for record in records.values())


def test_score_requires_a_prior_target_free_freeze(tmp_path: Path) -> None:
    args = runner.parse_args(["score", "--output-dir", str(tmp_path / "missing")])

    with pytest.raises(FileNotFoundError, match="prior target-free freeze"):
        runner.score(args)
