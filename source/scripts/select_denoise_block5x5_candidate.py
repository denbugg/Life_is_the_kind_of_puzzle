#!/usr/bin/env python3
"""Select one 5x5 candidate on development only, before frozen-gate access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from puzzle_denoise_v2.block5x5 import load_protocol, sha256_file


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidate_record(path: Path, protocol_sha256: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("kind") != "tile_naf_block5x5_finetune":
        raise ValueError(f"unsupported candidate checkpoint {path}")
    if checkpoint.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"candidate protocol hash mismatch: {path}")
    if checkpoint.get("init_checkpoint_sha256") != "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734":
        raise ValueError(f"candidate initialization mismatch: {path}")
    development = checkpoint.get("best_development")
    if not isinstance(development, dict):
        raise ValueError(f"candidate lacks best development record: {path}")
    delta = development.get("deltas_vs_initial")
    safety = development.get("safety")
    panels = development.get("panels")
    if not all(isinstance(value, dict) for value in (delta, safety, panels)):
        raise ValueError(f"candidate development schema mismatch: {path}")
    checks = {
        "balanced_ordered_image_ssim_delta_at_least_0_002": float(
            delta["ordered_image_ssim"]
        )
        >= 0.002,
        "balanced_tile_ssim_delta_nonnegative": float(delta["tile_ssim"]) >= 0.0,
        "boundary_mae_growth_at_most_0_2pct": bool(
            safety["boundary_mae_growth_at_most_0_2pct"]
        ),
        "gradient_mae_growth_at_most_0_2pct": bool(
            safety["gradient_mae_growth_at_most_0_2pct"]
        ),
        "both_panel_ordered_image_ssim_positive": bool(
            safety["both_panel_ordered_ssim_positive"]
        ),
    }
    return {
        "variant": checkpoint["variant"],
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "step": int(checkpoint["best_step"]),
        "development": development,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    load_protocol(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    candidates = [candidate_record(Path(value), protocol_sha256) for value in args.candidate]
    variants = [record["variant"] for record in candidates]
    if len(variants) != len(set(variants)):
        raise ValueError("candidate variants must be unique")
    eligible = [record for record in candidates if record["eligible"]]
    eligible.sort(
        key=lambda record: (
            -float(record["development"]["deltas_vs_initial"]["ordered_image_ssim"]),
            -float(record["development"]["deltas_vs_initial"]["tile_ssim"]),
            str(record["variant"]),
        )
    )
    selected = eligible[0] if eligible else None
    payload = {
        "schema_version": 1,
        "kind": "denoise_block5x5_development_selection",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha256,
        "frozen_gate_accessed": False,
        "candidates": candidates,
        "decision": "open_frozen_gate" if selected else "stop_no_development_signal",
        "selected_variant": selected["variant"] if selected else None,
        "selected_checkpoint": selected["path"] if selected else None,
        "selected_checkpoint_sha256": selected["sha256"] if selected else None,
    }
    output = Path(args.output)
    atomic_json(output, payload)
    print(json.dumps({"event": "block5x5_selection_complete", **payload}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
