from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


WRAPPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "runs/assembly_v1/kaggle/longsync4_retrieval_job/run_longsync4_retrieval.py"
)
SPEC = importlib.util.spec_from_file_location("run_longsync4_retrieval", WRAPPER_PATH)
assert SPEC is not None and SPEC.loader is not None
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def _report() -> dict:
    records = [
        {"panel": panel}
        for panel in ("primary_kornia", "independent_libjpeg")
        for _ in range(8)
    ]
    return {
        "kind": "longsync4_translation_hgb_top2_retrieval_diagnostic",
        "safe_for_submission": False,
        "protocol": {
            "split": "edge_development",
            "source_offset": 316,
            "source_count": 8,
            "source_names_sha256": wrapper.EXPECTED_SOURCE_NAMES_SHA256,
            "top_k": 2,
            "iterations": 10,
            "parameter_sweeps": 0,
            "assembly_targets_opened": False,
            "whole_source_disjoint_from_hgb_fit_calibration": True,
        },
        "records": records,
        "gate": {"decision": "stop_no_retrieval_signal"},
    }


def test_verify_report_accepts_exact_frozen_contract() -> None:
    wrapper.verify_report(_report())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_offset", 315),
        ("source_names_sha256", "0" * 64),
        ("top_k", 8),
        ("assembly_targets_opened", True),
    ],
)
def test_verify_report_rejects_protocol_drift(field: str, value: object) -> None:
    report = _report()
    report["protocol"][field] = value
    with pytest.raises(RuntimeError, match="protocol drift"):
        wrapper.verify_report(report)


def test_verify_report_rejects_record_count_or_submission_flag() -> None:
    report = _report()
    report["records"] = report["records"][:-1]
    with pytest.raises(RuntimeError, match="16"):
        wrapper.verify_report(report)

    report = _report()
    report["safe_for_submission"] = True
    with pytest.raises(RuntimeError, match="submission"):
        wrapper.verify_report(report)
