#!/usr/bin/env python3
"""Fail-closed Kaggle staging entrypoint for the stopped scorer adaptation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


WORKING = Path("/kaggle/working")
EXPECTED_PROTOCOL_SHA256 = "48824d5e1cd0426d7fc85c145704065c45b6e83e4f744c0c13f7c20f9782e9e5"
UPSTREAM_SELECTION_SHA256 = "932276b13e4ae4f0c09ba384cbff0cac9e7c49ab3b1b6b25f2dce5647c342a0c"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    # This artifact intentionally exits before importing torch or probing a GPU.
    # A future selected denoiser must use a newly pinned protocol/job version.
    payload = {
        "schema_version": 1,
        "kind": "solver_denoiser_adaptation_kaggle_no_launch",
        "status": "stopped_before_gpu_use",
        "decision": "stop_upstream_no_selected_denoiser_checkpoint",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "upstream_selection_sha256": UPSTREAM_SELECTION_SHA256,
        "gpu_probed": False,
        "training_launched": False,
        "message": "The 5x5 denoiser produced no selected checkpoint; scorer adaptation is not eligible.",
    }
    payload["canonical_payload_sha256_without_this_field"] = canonical_sha256(payload)
    output = WORKING / "solver_denoiser_adaptation_NO_LAUNCH.json"
    atomic_json(output, payload)
    print(json.dumps({"event": "solver_adaptation_no_launch", **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
