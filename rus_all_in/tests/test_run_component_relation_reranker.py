from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runner() -> ModuleType:
    path = PROJECT_ROOT / "scripts/run_component_relation_reranker.py"
    specification = importlib.util.spec_from_file_location(
        "run_component_relation_reranker_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def test_recursive_filename_collector_fails_closed() -> None:
    payload = {
        "selection": {
            "fit_filenames": ["nested/a.png"],
            "local_eval_filenames": ["b.png"],
        },
        "history": [{"future_panel_filenames": ("c.png",)}],
        "ignored": {"source_filename": "not-collected.png"},
    }
    assert runner.collect_filename_lists(payload) == {"a.png", "b.png", "c.png"}
    with pytest.raises(ValueError, match="duplicate"):
        runner.collect_filename_lists({"bad_filenames": ["a.png", "a.png"]})
    with pytest.raises(ValueError, match="non-empty"):
        runner.collect_filename_lists({"bad_filenames": [""]})


def _metrics(*, r1: float, raw_r1: float, correct: float, raw_correct: float) -> dict:
    return {
        "oracle_query_count": 512,
        "candidate_supply_coverage": 0.40,
        "learned": {
            "r1": r1,
            "r5": 0.70,
            "high_confidence": {
                "top32": {"correct_per_board": correct, "precision": correct / 32}
            },
        },
        "raw_socket_component_baseline": {
            "r1": raw_r1,
            "r5": 0.70,
            "high_confidence": {
                "top32": {
                    "correct_per_board": raw_correct,
                    "precision": raw_correct / 32,
                }
            },
        },
    }


def test_local_gate_requires_material_raw_socket_improvement() -> None:
    passed = runner.evaluate_gate(
        _metrics(r1=0.43, raw_r1=0.39, correct=14.0, raw_correct=12.0)
    )
    assert passed["pass"]
    assert passed["status"] == "pass-await-root-review"
    assert not passed["quality_panel_authorized"]

    failed = runner.evaluate_gate(
        _metrics(r1=0.415, raw_r1=0.39, correct=14.0, raw_correct=12.0)
    )
    assert not failed["pass"]
    assert not failed["checks"]["pair_translation_r1_gain"]["pass"]

