#!/usr/bin/env python3
"""Run the fail-closed frozen h20x1 calibration or single-use holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aiijc_puzzle.frozen_final_evaluator import (
    DEFAULT_INPUTS_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TARGETS_DIR,
    FROZEN_CONFIG_PATH,
    artifact_paths,
    load_context,
    run_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("calibration", "holdout"),
        default="calibration",
        help="calibration is repeatable; holdout is single-use and gate-protected",
    )
    parser.add_argument("--config", type=Path, default=FROZEN_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_DIR)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="calibration may use a custom report path; holdout path is fixed",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=None,
        help="defaults to the config-addressed path; holdout does not permit redirection",
    )
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="explicitly authorize the single target-opening transition after gates pass",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="required acknowledgement for either evaluation mode",
    )
    return parser.parse_args()


def emit_progress(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("refusing evaluation without --run")
    context = load_context(args.mode, config_path=args.config, manifest_path=args.manifest)
    paths = artifact_paths(context.config_sha256)
    if args.mode == "calibration":
        output = args.output or paths.calibration_report
        commitment = paths.calibration_commitment
    else:
        output = args.output or paths.holdout_report
        commitment = paths.holdout_commitment
    report = run_evaluation(
        context,
        inputs_dir=args.inputs,
        targets_dir=args.targets,
        report_path=output,
        commitment_path=commitment,
        allow_holdout=args.allow_holdout,
        calibration_report_path=args.calibration_report,
        progress=emit_progress,
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "status": report["status"],
                "mode": report["mode"],
                "final": report["summary"]["rgb_luma_then_colored_nlm_h20x1_final"],
                "gate": report["preregistered_gate"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
