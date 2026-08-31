#!/usr/bin/env python3
"""CLI for the frozen alpha=.125 DualNAF plus h28 stacking experiment."""

from __future__ import annotations

import argparse
import json

from aiijc_puzzle.dualnaf_stack import (
    CONFIG_PATH,
    CONFIG_SHA256,
    OUTPUT_ROOT,
    audit_historical_exposure,
    panel_records,
    prepare_panel,
    record_manual_review,
    score_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "audit",
            "prepare-primary",
            "score-primary",
            "review-primary",
            "prepare-confirmation",
            "score-confirmation",
            "review-confirmation",
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--verdict", choices=("PASS", "FAIL"))
    parser.add_argument("--reason")
    arguments = parser.parse_args()

    if arguments.command == "audit":
        payload = {
            "config": str(CONFIG_PATH),
            "config_sha256": CONFIG_SHA256,
            "output_root": str(OUTPUT_ROOT),
            "primary": {
                "count": len(panel_records("primary")),
                "first": panel_records("primary")[0]["filename"],
                "last": panel_records("primary")[-1]["filename"],
            },
            "confirmation": {
                "count": len(panel_records("confirmation")),
                "first": panel_records("confirmation")[0]["filename"],
                "last": panel_records("confirmation")[-1]["filename"],
            },
            "historical_exposure": audit_historical_exposure(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    panel = "confirmation" if "confirmation" in arguments.command else "primary"
    if arguments.command.startswith("prepare"):
        path = prepare_panel(panel, device_name=arguments.device)
    elif arguments.command.startswith("score"):
        path = score_panel(panel)
    else:
        if arguments.verdict is None or arguments.reason is None:
            raise SystemExit("review commands require --verdict and --reason")
        path = record_manual_review(panel, arguments.verdict, arguments.reason)
    print(path)


if __name__ == "__main__":
    main()
