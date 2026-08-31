#!/usr/bin/env python3
"""Run exactly one frozen, synthetic-only BasinCycle mechanism gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from aiijc_puzzle.basincycle_synthetic import evaluate_synthetic_gate  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = _load_json(config_path)
    if config.get("scope") != "synthetic-mechanism-only":
        raise ValueError("only a synthetic-mechanism-only configuration is accepted")
    if config.get("frozen") is not True:
        raise ValueError("configuration must explicitly be frozen")

    for label, artifact in config["implementation"].items():
        path = project_root / artifact["path"]
        observed = _sha256(path)
        if observed != artifact["sha256"]:
            raise ValueError(
                f"{label} implementation hash mismatch: {observed} != {artifact['sha256']}"
            )

    output_path = project_root / config["output"]["report_path"]
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite the one-shot report: {output_path}")

    result = evaluate_synthetic_gate(config)
    result["schema_version"] = "aiijc-basincycle-synthetic-gate-report-v1"
    result["config_path"] = str(config_path.relative_to(project_root))
    result["config_sha256"] = _sha256(config_path)
    result["implementation"] = config["implementation"]
    result["boundary"] = {
        "organizer_data_read": False,
        "images_read": False,
        "learned_model_used": False,
        "dev_or_test_opened": False,
        "promotion_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output_path), **result["summary"], "status": result["status"]}))


if __name__ == "__main__":
    main()
