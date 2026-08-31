#!/usr/bin/env python3
"""One no-tuning fresh32 confirmation of the held exact-only stacker signal."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_edge_calibrator import TaskaEdgeCalibrator
from aiijc_puzzle.taska_focal_feature_stacker import TaskaFocalFeatureStacker
from aiijc_puzzle.taska_nonlinear_calibrator import TaskaNonlinearCalibrator

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_focal_feature_stacker as parent
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune
    import run_taska_focal_feature_stacker as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/taska-focal-feature-stacker/train96-v1"
PARENT_REPORT = OUTPUT_ROOT / "report.json"
STACKER_ARTIFACT = OUTPUT_ROOT / "stacker.npz"
REPORT_PATH = OUTPUT_ROOT / "fresh-exact-confirmation-report.json"
PARENT_REPORT_SHA256 = "cb2fca61e2f715abe096d9e2d10b4951825628c402adbaad6103aae7a054a485"
STACKER_SHA256 = "adad56de9245ec999741a0e0966c2767992ba362b6fd731a12885588bd13ae4f"


def _record(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(path.resolve()),
    }


def main() -> None:
    if sha256_file(PARENT_REPORT) != PARENT_REPORT_SHA256:
        raise ValueError("parent stacker report changed")
    if sha256_file(STACKER_ARTIFACT) != STACKER_SHA256:
        raise ValueError("fixed logistic stacker artifact changed")
    parent._require_frozen_inputs()
    prior = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    held = prior["held32"]["summary"]["five_minus_four"]
    if held["satisfied_adjacent_pairs"]["mean"] >= 0:
        raise ValueError("override requires the preregistered held pair gate to fail")
    if held["exact_tiles"]["ci95_lower"] <= 0:
        raise ValueError("override requires a positive held exact CI lower bound")

    stacker = TaskaFocalFeatureStacker.load_npz(STACKER_ARTIFACT)
    logistic = TaskaEdgeCalibrator.load_npz(
        PROJECT_ROOT / "outputs/taska-edge-calibrator/train256-v1/calibrator.npz"
    )
    nonlinear = TaskaNonlinearCalibrator.load_npz(
        PROJECT_ROOT / "outputs/taska-nonlinear-calibrator/train256-v1/calibrator.npz"
    )
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(parent.DEFAULT_TARGETS.resolve(), maximum_boards=2)
    started = perf_counter()
    result = parent._run_cached_panel(
        stage="fresh32-exact-override",
        output_dir=OUTPUT_ROOT,
        stacker=stacker,
        stacker_path=STACKER_ARTIFACT,
        logistic=logistic,
        nonlinear=nonlinear,
        lookup=lookup,
        cache=cache,
        parent_archive_path=parent.FRESH_PARENT_ARCHIVE,
        parent_metadata_path=parent.FRESH_PARENT_METADATA,
        focal_archive_path=parent.FRESH_LEADER_ARCHIVE,
        focal_metadata_path=parent.FRESH_LEADER_METADATA,
        leader_archive_path=parent.FRESH_LEADER_ARCHIVE,
    )
    report = {
        "schema": "aiijc-taska-focal-feature-stacker-fresh-exact-confirmation-v1",
        "status": "complete",
        "override_reason": {
            "preregistered_pair_promotion_failed": True,
            "held_exact_ci_triggered_one_confirmation": True,
            "no_tuning_or_parameter_change": True,
            "pair_production_integration_authorized": False,
        },
        "fresh32": result,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "parent_report": _record(PARENT_REPORT),
            "stacker": _record(STACKER_ARTIFACT),
            "wrapper": _record(Path(__file__)),
        },
        "legality": prior["legality"],
    }
    parent._write_json(REPORT_PATH, report)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
