from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_fullres_fusion_union_priority_opened40.py"
    specification = importlib.util.spec_from_file_location(
        "run_fullres_fusion_union_priority_opened40_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()
COUNT = runner.COUNT
DECODER_EDGE_BUDGET = runner.DECODER_EDGE_BUDGET
FIXED_TOP_EDGE_COUNT = runner.FIXED_TOP_EDGE_COUNT
GRID = runner.GRID
_edge_is_correct = runner._edge_is_correct
_fixed_top288_correct = runner._fixed_top288_correct
_prepare_output_paths = runner._prepare_output_paths
_select_device = runner._select_device
_strict_layout = runner._strict_layout
_validate_limit = runner._validate_limit


def _true_edges(axis: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(COUNT, dtype=np.int32)
    if axis == 0:
        valid = positions % GRID != GRID - 1
        return positions[valid], positions[valid] + 1
    valid = positions < COUNT - GRID
    return positions[valid], positions[valid] + GRID


def test_fixed_top288_scores_frozen_priorities_without_redecoding() -> None:
    prefix = "case_0000"
    archive: dict[str, np.ndarray] = {}
    for axis in (0, 1):
        source, target = _true_edges(axis)
        distractor_count = GRID * (GRID - 1) - len(source)
        distractor_source = np.arange(distractor_count, dtype=np.int32)
        distractor_target = (distractor_source + 2) % COUNT
        source = np.concatenate((source, distractor_source))
        target = np.concatenate((target, distractor_target))
        priority = np.linspace(2.0, 0.0, len(source), dtype=np.float64)
        archive[f"{prefix}__axis_{axis}_source"] = source
        archive[f"{prefix}__axis_{axis}_target"] = target
        archive[f"{prefix}__axis_{axis}_baseline_priority"] = priority
        archive[f"{prefix}__axis_{axis}_treatment_priority"] = priority
    observed = _fixed_top288_correct(
        archive,
        prefix,
        np.arange(COUNT, dtype=np.int32),
        arm="baseline",
    )
    assert observed == FIXED_TOP_EDGE_COUNT == 2 * DECODER_EDGE_BUDGET


def test_edge_truth_is_noncyclic_and_rejects_unknown_axis() -> None:
    reference = np.arange(COUNT, dtype=np.int32)
    horizontal = _edge_is_correct(
        np.asarray([0, GRID - 1], dtype=np.int32),
        np.asarray([1, GRID], dtype=np.int32),
        axis=0,
        reference=reference,
    )
    vertical = _edge_is_correct(
        np.asarray([0, COUNT - GRID], dtype=np.int32),
        np.asarray([GRID, 0], dtype=np.int32),
        axis=1,
        reference=reference,
    )
    assert horizontal.tolist() == [True, False]
    assert vertical.tolist() == [True, False]
    with pytest.raises(ValueError, match="axis"):
        _edge_is_correct(
            np.asarray([0]),
            np.asarray([1]),
            axis=2,
            reference=reference,
        )


def test_limit_and_strict_layout_contracts() -> None:
    assert _validate_limit(1) == 1
    assert _validate_limit(40) == 40
    for invalid in (True, 0, 41):
        with pytest.raises(ValueError, match="limit"):
            _validate_limit(invalid)  # type: ignore[arg-type]
    assert np.array_equal(_strict_layout(np.arange(COUNT)), np.arange(COUNT))
    duplicate = np.arange(COUNT, dtype=np.int32)
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="strict"):
        _strict_layout(duplicate)


def test_output_contract_refuses_any_existing_artifact(tmp_path) -> None:
    paths = _prepare_output_paths(tmp_path / "new")
    assert all(path.parent == (tmp_path / "new").resolve() for path in paths)
    paths[0].touch()
    with pytest.raises(FileExistsError, match="overwrite"):
        _prepare_output_paths(tmp_path / "new")


def test_device_contract_requires_explicit_mps_nondeterminism(monkeypatch) -> None:
    assert _select_device("cpu", allow_nondeterministic_mps=False).type == "cpu"
    with pytest.raises(ValueError, match="requires MPS"):
        _select_device("cpu", allow_nondeterministic_mps=True)
    with pytest.raises(ValueError, match="requires --allow"):
        _select_device("mps", allow_nondeterministic_mps=False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        _select_device("mps", allow_nondeterministic_mps=True)


def test_cli_defaults_to_mps_and_full_opened_panel(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["runner"])
    args: Namespace = runner.parse_args()
    assert args.device == "mps"
    assert args.limit == 40
    assert not args.allow_nondeterministic_mps
