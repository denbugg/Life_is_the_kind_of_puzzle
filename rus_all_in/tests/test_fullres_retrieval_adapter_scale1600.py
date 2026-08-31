from __future__ import annotations

import importlib.util
import sys
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


pilot = _load("fullres_retrieval_adapter_pilot_test", "run_fullres_retrieval_adapter.py")
scale = _load(
    "fullres_retrieval_adapter_scale1600_test",
    "run_fullres_retrieval_adapter_scale1600.py",
)


def test_extended_stream_preserves_the_signed_first_400_specs() -> None:
    old = pilot._training_specs()
    new = scale._training_specs()
    assert len(old) == 400
    assert len(new) == 1600
    assert [tuple(vars(item).values()) for item in new[:400]] == [
        tuple(vars(item).values()) for item in old
    ]


def _metrics(*, r1: float, r5: float, precision_gain: float, coverage: float) -> dict:
    return {
        "retrieval": {
            "raw_d64_ot": {"pooled_r1": 0.20, "pooled_r5": 0.40},
            "adapter_step1600": {"pooled_r1": r1, "pooled_r5": r5},
        },
        "reciprocal": {
            "matched_vs_raw": {
                "adapter_step1600": {
                    "precision_gain": precision_gain,
                    "matched_coverage": coverage,
                }
            }
        },
    }


def test_local_gate_requires_all_three_signed_signal_families() -> None:
    passing = _metrics(r1=0.206, r5=0.40, precision_gain=0.002, coverage=0.03)
    assert scale._local_gate(passing)["terminal_open_gate_passed"] is True

    for failing in (
        _metrics(r1=0.2049, r5=0.40, precision_gain=0.002, coverage=0.03),
        _metrics(r1=0.206, r5=0.3999, precision_gain=0.002, coverage=0.03),
        _metrics(r1=0.206, r5=0.40, precision_gain=0.0019, coverage=0.03),
        _metrics(r1=0.206, r5=0.40, precision_gain=0.002, coverage=0.029),
    ):
        assert scale._local_gate(failing)["terminal_open_gate_passed"] is False


def test_terminal_gate_is_nonnegative_transfer_not_local_threshold_reuse() -> None:
    nonnegative = _metrics(r1=0.20, r5=0.40, precision_gain=0.0, coverage=0.03)
    assert scale._terminal_gate(nonnegative)["transfer_passed"] is True
    negative = _metrics(r1=0.1999, r5=0.40, precision_gain=0.0, coverage=0.03)
    assert scale._terminal_gate(negative)["transfer_passed"] is False
