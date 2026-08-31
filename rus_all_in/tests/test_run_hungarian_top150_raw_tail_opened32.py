from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runner() -> ModuleType:
    scripts = PROJECT_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "run_hungarian_top150_raw_tail_opened32.py"
    specification = importlib.util.spec_from_file_location(
        "run_hungarian_top150_raw_tail_opened32_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_hungarian_top_edges_keeps_highest_assigned_nonself_edges() -> None:
    scores = np.asarray(
        [
            [-100.0, 9.0, 1.0],
            [2.0, -100.0, 8.0],
            [7.0, 3.0, -100.0],
        ]
    )

    edges = runner._hungarian_top_edges(scores, axis="right", keep=2)

    assert [(edge.source, edge.target, edge.axis) for edge in edges] == [
        (0, 1, "right"),
        (1, 2, "right"),
    ]


def test_hungarian_top_edges_rejects_invalid_keep() -> None:
    with pytest.raises(ValueError, match="keep"):
        runner._hungarian_top_edges(np.eye(3), axis="down", keep=4)


def test_strict_layout_accepts_only_full_permutation() -> None:
    layout = runner._strict_layout(np.arange(runner.COUNT, dtype=np.int32))
    assert layout.dtype == np.int64
    broken = np.arange(runner.COUNT, dtype=np.int64)
    broken[-1] = 0
    with pytest.raises(ValueError, match="strict"):
        runner._strict_layout(broken)
