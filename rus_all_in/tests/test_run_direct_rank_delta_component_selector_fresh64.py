from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts import run_direct_rank_delta_component_selector_fresh64 as runner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT_ROOT / "outputs/union-hard-edge-priority/roster-audit-v1.json"
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"


def test_canonical_audit_excludes_base_union_and_future_pilot() -> None:
    _, digest, excluded, pilot = runner.load_roster_audit(AUDIT)
    assert digest == runner.AUDIT_SHA256
    assert len(excluded) == runner.AUDIT_EXCLUDED_TRAIN_COUNT
    assert len(pilot) == runner.PILOT_SOURCE_COUNT
    assert not excluded.intersection(pilot)
    assert len(excluded | set(pilot)) == 3_144


def test_selection_mode_freezes_hash_locked_disjoint_config_without_targets(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "selector-fresh64.json"
    args = Namespace(
        mode="selection",
        config=config_path,
        audit=AUDIT,
        manifest=MANIFEST,
        targets=tmp_path / "must-not-be-read",
        output_dir=tmp_path / "must-not-be-written",
    )
    runner.freeze_selection(args)
    config, digest = runner.load_confirmation_config(config_path)
    assert len(digest) == 64
    assert not args.targets.exists()
    assert not args.output_dir.exists()
    assert len(config["selection"]["source_filenames"]) == runner.EXPECTED_SOURCES
    assert config["selection"]["combined_excluded_train_count"] == 3_144
    assert config["selection"]["selected_exclusion_overlap"] == []
    assert config["legality"]["competition_test_opened"] is False


def _comparison(exact: float, adjacency: float) -> dict[str, dict[str, float]]:
    return {
        "exact_tiles_delta": {"mean": exact},
        "adjacency_delta": {"mean": adjacency},
    }


def test_confirmation_gate_requires_both_exact_comparators_and_safety() -> None:
    metrics = {
        "component_selector_vs_union_v2": _comparison(0.25, 0.0),
        "component_selector_vs_rank_delta_transfer": _comparison(0.10, -1.0),
    }
    passed = runner.evaluate_confirmation_gate(metrics, strict_layouts=3 * 64)
    assert passed["pass"]
    assert not passed["promotion_automatic"]
    assert not runner.evaluate_confirmation_gate(metrics, strict_layouts=3 * 64 - 1)["pass"]

    metrics["component_selector_vs_union_v2"] = _comparison(0.249, 0.0)
    assert not runner.evaluate_confirmation_gate(metrics, strict_layouts=3 * 64)["pass"]
    metrics["component_selector_vs_union_v2"] = _comparison(0.25, -1e-12)
    assert not runner.evaluate_confirmation_gate(metrics, strict_layouts=3 * 64)["pass"]


def test_selection_counts_are_board_level() -> None:
    rows = [
        {
            "selection": {
                "selected_arm": "rank_delta_transfer",
                "reason": "more_consistent_redundant_constraints",
            }
        },
        {
            "selection": {
                "selected_arm": "rank_delta_transfer",
                "reason": "consistent_tie_larger_component",
            }
        },
        {
            "selection": {
                "selected_arm": "union_v2",
                "reason": "union_conservative_fallback",
            }
        },
    ]
    selections, reasons = runner._selection_counts(rows)
    assert selections == {"union_v2": 1, "rank_delta_transfer": 2}
    assert reasons == {
        "more_consistent_redundant_constraints": 1,
        "consistent_tie_larger_component": 1,
        "union_conservative_fallback": 1,
    }
