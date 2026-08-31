from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from aiijc_puzzle.protocol import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runner() -> ModuleType:
    path = PROJECT_ROOT / "scripts/run_direct_hard_edge_priority.py"
    specification = importlib.util.spec_from_file_location(
        "run_direct_hard_edge_priority_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_recursive_filename_exclusion_is_fail_closed() -> None:
    payload = {
        "selection": {
            "fit_source_filenames": ["nested/a.png"],
            "evaluation_source_filenames": ["b.png"],
        },
        "nested": [{"holdout_filenames": ("c.png",)}],
        "ignored_source_filename": "not-collected.png",
    }
    assert runner._collect_filename_lists(payload) == {"a.png", "b.png", "c.png"}
    with pytest.raises(ValueError, match="duplicate"):
        runner._collect_filename_lists({"bad_filenames": ["a.png", "a.png"]})
    with pytest.raises(ValueError, match="non-empty"):
        runner._collect_filename_lists({"bad_filenames": [""]})


def test_actual_panel_collector_ignores_broad_forbidden_registry() -> None:
    payload = {
        "selection": {
            "evaluation_source_filenames": ["a.png", "b.png"],
            "forbidden_filenames": ["not-opened.png"],
        },
        "rows": [{"source_filename": "c.png"}],
    }
    assert runner._collect_actual_roster_filenames(payload) == {
        "a.png",
        "b.png",
        "c.png",
    }


def test_d1_gate_uses_predeclared_or_rule_without_promotion() -> None:
    raw = {"correct_selected_edges": 70.0, "selected_edge_precision": 0.70}
    by_correct = runner.evaluate_d1_gate(
        raw,
        {"correct_selected_edges": 71.0, "selected_edge_precision": 0.701},
    )
    assert by_correct["pass"]
    assert not by_correct["promotion_authorized"]
    by_precision = runner.evaluate_d1_gate(
        raw,
        {"correct_selected_edges": 70.2, "selected_edge_precision": 0.71},
    )
    assert by_precision["pass"]
    failed = runner.evaluate_d1_gate(
        raw,
        {"correct_selected_edges": 70.99, "selected_edge_precision": 0.7099},
    )
    assert not failed["pass"]
    assert failed["status"] == "stop"


def test_preregistered_config_is_hash_locked_before_d1_access() -> None:
    path = PROJECT_ROOT / "configs/direct_hard_edge_board_priority_preregistered_v1.json"
    config, digest = runner.load_frozen_config(path)
    assert digest == "11ba187b5a739a54193e6f869f443a9bcd04d1559c641c8d3b3ffd0151f514fb"
    assert digest == sha256_file(path)
    assert len(config["selection"]["fit_source_filenames"]) == 256
    assert len(config["selection"]["d1_source_filenames"]) == 32
    assert not config["competition_test_opened"]
    assert not config["d1_gate"]["promotion_authorized"]
