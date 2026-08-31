from __future__ import annotations

import json
from pathlib import Path

from scripts.run_fullres_relation_fusion import _gate

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _metrics(
    *,
    supply_gain: float,
    r1_gain: float,
    r5_gain: float,
    correct_gain: float,
    precision_gain: float,
) -> dict:
    relation_top32 = {"correct_per_board": 4.0, "precision": 0.125}
    fusion_top32 = {
        "correct_per_board": 4.0 + correct_gain,
        "precision": 0.125 + precision_gain,
    }
    return {
        "raw_candidate_supply_coverage": 0.30,
        "union_candidate_supply_coverage": 0.30 + supply_gain,
        "methods": {
            "frozen_relation": {
                "r1": 0.20,
                "r5": 0.60,
                "high_confidence": {"top32": relation_top32},
            },
            "fusion": {
                "r1": 0.20 + r1_gain,
                "r5": 0.60 + r5_gain,
                "high_confidence": {"top32": fusion_top32},
            },
        },
    }


def test_sensitive_gate_accepts_ranking_or_confidence_but_never_decoder() -> None:
    prereg = json.loads(
        (PROJECT_ROOT / "configs/fullres_relation_fusion_preregistered_v1.json").read_text()
    )
    ranking = _gate(
        _metrics(
            supply_gain=0.02,
            r1_gain=0.003,
            r5_gain=0.0,
            correct_gain=0.0,
            precision_gain=0.0,
        ),
        prereg,
    )
    confidence = _gate(
        _metrics(
            supply_gain=0.02,
            r1_gain=0.0,
            r5_gain=0.0,
            correct_gain=0.25,
            precision_gain=0.0,
        ),
        prereg,
    )
    assert ranking["discovery_pass"]
    assert confidence["discovery_pass"]
    assert not ranking["decoder_authorized"]
    assert not confidence["promotion_authorized"]


def test_supply_gain_is_mandatory_even_with_positive_ranking() -> None:
    prereg = json.loads(
        (PROJECT_ROOT / "configs/fullres_relation_fusion_preregistered_v1.json").read_text()
    )
    result = _gate(
        _metrics(
            supply_gain=0.0,
            r1_gain=0.05,
            r5_gain=0.05,
            correct_gain=2.0,
            precision_gain=0.05,
        ),
        prereg,
    )
    assert not result["discovery_pass"]
    assert not result["supply"]["pass"]
