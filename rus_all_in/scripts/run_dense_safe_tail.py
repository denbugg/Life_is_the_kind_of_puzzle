#!/usr/bin/env python3
"""CLI for the preregistered dense legal single-pass NLM screen."""

from __future__ import annotations

import argparse
import json

from aiijc_puzzle.dense_safe_tail import (
    CONFIG_PATH,
    CONFIG_SHA256,
    OUTPUT_ROOT,
    audit_historical_exposure,
    panel_records,
    prepare_panel,
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
            "prepare-confirmation",
            "score-confirmation",
        ),
    )
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
    path = prepare_panel(panel) if arguments.command.startswith("prepare") else score_panel(panel)
    print(path)


if __name__ == "__main__":
    main()
