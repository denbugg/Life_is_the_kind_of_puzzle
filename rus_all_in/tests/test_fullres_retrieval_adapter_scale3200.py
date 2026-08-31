from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str, filename: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


runner = _load(
    "fullres_retrieval_adapter_scale3200_test",
    "run_fullres_retrieval_adapter_scale3200.py",
)


def _metrics() -> dict:
    return {
        "retrieval": {
            "raw_d64_ot": {
                "pooled_total": 1000,
                "pooled_r1": 0.20,
                "pooled_r5": 0.40,
            },
            "adapter_step1600": {
                "pooled_r1": 0.203,
                "pooled_r5": 0.405,
                "pooled_r32": 0.70,
                "right_r1": 0.20,
                "right_r5": 0.40,
                "down_r1": 0.206,
                "down_r5": 0.41,
            },
            "adapter_step3200": {
                "pooled_r1": 0.206,
                "pooled_r5": 0.409,
                "pooled_r32": 0.704,
                "right_r1": 0.202,
                "right_r5": 0.404,
                "down_r1": 0.21,
                "down_r5": 0.414,
            },
        },
        "supply": {
            "adapter_step1600": {
                "pooled_union_coverage": 0.74,
                "axes": {
                    "right": {"union_coverage": 0.75},
                    "down": {"union_coverage": 0.73},
                },
            },
            "adapter_step3200": {
                "pooled_union_coverage": 0.745,
                "axes": {
                    "right": {"union_coverage": 0.753},
                    "down": {"union_coverage": 0.737},
                },
            },
        },
        "reciprocal": {
            "matched_vs_raw": {
                "adapter_step3200": {
                    "precision_gain": 0.004,
                    "matched_coverage": 0.45,
                }
            }
        },
        "checkpoint_matched_reciprocal": {
            "step3200_minus_step1600_precision": 0.003
        },
    }


def test_scaling_and_local_gate_pass_positive_fixed_slope() -> None:
    metrics = _metrics()
    scaling = runner._scaling(metrics)
    assert abs(scaling["retrieval"]["pooled_r5"] - 0.004) < 1e-12
    assert abs(scaling["raw_union_top32"]["pooled_coverage"] - 0.005) < 1e-12
    assert runner._gate(metrics, scaling, terminal=False)[
        "terminal_open_gate_passed"
    ]


def test_local_gate_fails_when_matched_union_slope_reverses() -> None:
    metrics = deepcopy(_metrics())
    metrics["supply"]["adapter_step3200"]["pooled_union_coverage"] = 0.739
    scaling = runner._scaling(metrics)
    assert not runner._gate(metrics, scaling, terminal=False)[
        "terminal_open_gate_passed"
    ]


def test_terminal_gate_does_not_require_local_scaling_payload() -> None:
    metrics = _metrics()
    gate = runner._gate(metrics, {}, terminal=True)
    assert gate["transfer_passed"]
    assert gate["matched_step1600_to_step3200_r5_gain"] is None
