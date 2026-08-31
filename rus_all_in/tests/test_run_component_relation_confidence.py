from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runner() -> ModuleType:
    path = PROJECT_ROOT / "scripts/run_component_relation_confidence.py"
    specification = importlib.util.spec_from_file_location(
        "run_component_relation_confidence_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _gate_contract() -> dict:
    return {
        "minimum_learned_pair_translation_r1_gain_over_raw": 0.02,
        "minimum_learned_pair_translation_r5_gain_over_raw": 0.02,
        "top32_either": {
            "minimum_correct_attachments_per_board_gain_over_raw": 0.25,
            "minimum_matched_precision_gain_over_raw": 0.01,
        },
        "minimum_top144_correct_attachments_per_board_gain_over_raw": 0.0,
    }


def _metrics(*, r1_gain: float, r5_gain: float, top32_gain: float) -> tuple[dict, dict]:
    relation = {
        "learned": {"r1": 0.30 + r1_gain, "r5": 0.70 + r5_gain},
        "raw_socket_component_baseline": {"r1": 0.30, "r5": 0.70},
    }
    confidence = {
        "calibrated": {
            "high_confidence": {
                "top32": {
                    "correct_per_board": 5.0 + top32_gain,
                    "precision": (5.0 + top32_gain) / 32,
                },
                "top144": {"correct_per_board": 18.0, "precision": 0.125},
            }
        },
        "raw_socket_component_baseline": {
            "high_confidence": {
                "top32": {"correct_per_board": 5.0, "precision": 5.0 / 32},
                "top144": {"correct_per_board": 18.0, "precision": 0.125},
            }
        },
    }
    return relation, confidence


def test_revised_discovery_gate_uses_top32_either_but_keeps_relation_signal() -> None:
    relation, confidence = _metrics(r1_gain=0.03, r5_gain=0.03, top32_gain=0.3)
    passed = runner.evaluate_confirm_gate(
        relation,
        confidence,
        ranking_unchanged=True,
        gate_contract=_gate_contract(),
    )
    assert passed["pass"]
    assert passed["decoder40_authorized"]
    assert not passed["promotion_authorized"]

    weak_relation, confidence = _metrics(
        r1_gain=0.01,
        r5_gain=0.03,
        top32_gain=0.4,
    )
    failed = runner.evaluate_confirm_gate(
        weak_relation,
        confidence,
        ranking_unchanged=True,
        gate_contract=_gate_contract(),
    )
    assert not failed["pass"]


def test_recursive_lineage_exclusion_and_preregistration_hash() -> None:
    assert runner.collect_filename_lists(
        {"outer": [{"future_confirm_filenames": ["x/a.png", "b.png"]}]}
    ) == {"a.png", "b.png"}
    with pytest.raises(ValueError, match="duplicate"):
        runner.collect_filename_lists({"bad_filenames": ["a.png", "a.png"]})
    prereg, digest = runner.load_preregistration(
        PROJECT_ROOT / "configs/component_relation_confidence_preregistered_v1_1.json"
    )
    assert digest.startswith("a029377e")
    assert prereg["policy_revision"]["recorded_before_confirm24_target_access"]


def test_frozen_forward_can_omit_all_exact_label_attachment(monkeypatch) -> None:
    case = runner.PreparedCase(
        case_id="fresh-case",
        source_filename="img_000001.png",
        dirty_tiles=np.zeros((576, 20, 20, 3), dtype=np.uint8),
        input_tile_to_position=np.arange(576, dtype=np.int32),
    )
    socket_output = SimpleNamespace(
        right_log_assignment=torch.empty(0),
        down_log_assignment=torch.empty(0),
    )
    monkeypatch.setattr(
        runner,
        "extract_frozen_socket_context",
        lambda *_args, **_kwargs: (torch.zeros((1, 576, 64)), socket_output),
    )
    monkeypatch.setattr(runner, "rebuild_decoder_components", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        runner,
        "component_descriptors_from_decoder",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        runner,
        "build_component_relation_candidates",
        lambda *_args, **_kwargs: (),
    )

    def labels_must_not_run(*_args, **_kwargs):
        raise AssertionError("exact target attachment ran in prediction-only mode")

    monkeypatch.setattr(runner, "component_relation_targets", labels_must_not_run)

    class Head:
        def __call__(self, *_args, **_kwargs):
            return torch.empty(0)

    output = runner.frozen_case_forward(
        case,
        socket=type("Socket", (), {"model": object()})(),
        head=Head(),
        device=torch.device("cpu"),
        attach_exact_labels=False,
    )
    assert output.labels == ()
    assert output.oracle_relations == frozenset()
    assert output.profiles == ()
    assert "exact_labels_after_freeze" not in output.runtime_seconds
