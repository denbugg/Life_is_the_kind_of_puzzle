#!/usr/bin/env python3
"""Pin a staged scorer-adaptation protocol after its launch interlocks clear."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from puzzle_assembly.denoiser_adaptation import (
    canonical_sha256,
    names_sha256,
    sha256_file,
    validate_protocol_safety,
)
from puzzle_assembly.protocol import source_names_for_split


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template", default="configs/solver_denoiser_adaptation_v1.json"
    )
    parser.add_argument("--new-denoiser", required=True)
    parser.add_argument("--expected-new-denoiser-sha256", required=True)
    parser.add_argument(
        "--oracle-verdict", choices=("pass", "fail", "inconclusive"), required=True
    )
    parser.add_argument(
        "--root-launch-signal",
        required=True,
        help="must equal the predeclared literal ROOT_AUTHORIZED",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _verify_asset(record: dict, label: str) -> dict:
    path = _resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise ValueError(f"{label} SHA mismatch: {actual}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def _verify_splits(config: dict) -> dict:
    manifest = _resolve(config["authoritative_inputs"]["manifest"]["path"])
    quarantine = _resolve(config["authoritative_inputs"]["quarantine"]["path"])
    records = {}
    for key in ("scorer_training", "exact_selection"):
        spec = config["source_partitions"][key]
        names = source_names_for_split(
            spec["split"], manifest_path=manifest, quarantine_path=quarantine
        )[spec["offset"] : spec["offset"] + spec["count"]]
        if len(names) != spec["count"]:
            raise ValueError(f"{key} source slice is short")
        actual = names_sha256(names)
        if actual != spec["names_sha256"]:
            raise ValueError(f"{key} names SHA mismatch: {actual}")
        if key == "exact_selection" and names != spec["names"]:
            raise ValueError("exact_selection names drifted from the pinned list")
        records[key] = {
            "split": spec["split"],
            "offset": spec["offset"],
            "count": len(names),
            "names_sha256": actual,
        }
    if set(config["source_partitions"]["exact_selection"]["names"]) & set(
        source_names_for_split(
            config["source_partitions"]["scorer_training"]["split"],
            manifest_path=manifest,
            quarantine_path=quarantine,
        )[
            config["source_partitions"]["scorer_training"]["offset"] :
            config["source_partitions"]["scorer_training"]["offset"]
            + config["source_partitions"]["scorer_training"]["count"]
        ]
    ):
        raise ValueError("scorer training and exact selection overlap")
    return records


def prepare(args: argparse.Namespace) -> tuple[dict, dict]:
    if args.root_launch_signal != "ROOT_AUTHORIZED":
        raise ValueError("root launch interlock is not authorized")
    template_path = _resolve(args.template)
    template = _load_json(template_path)
    validate_protocol_safety(template)
    upstream = template.get("upstream_new_denoiser_result", {})
    if upstream.get("permits_adaptation") is not True:
        artifact = _resolve(upstream.get("artifact", ""))
        if not artifact.is_file():
            raise ValueError("upstream adaptation interlock is closed and its evidence is missing")
        actual_upstream_sha = sha256_file(artifact)
        if actual_upstream_sha != upstream.get("artifact_sha256"):
            raise ValueError("upstream no-launch evidence SHA mismatch")
        raise ValueError(
            "upstream denoiser selected no checkpoint; solver adaptation must not launch"
        )
    if len(args.expected_new_denoiser_sha256) != 64:
        raise ValueError("expected new-denoiser SHA256 must contain 64 hex characters")
    new_denoiser = _resolve(args.new_denoiser)
    if not new_denoiser.is_file():
        raise FileNotFoundError(f"missing new denoiser: {new_denoiser}")
    actual_new_sha = sha256_file(new_denoiser)
    if actual_new_sha != args.expected_new_denoiser_sha256.lower():
        raise ValueError(f"new denoiser SHA mismatch: {actual_new_sha}")

    assets = {
        key: _verify_asset(template["authoritative_inputs"][key], key)
        for key in ("manifest", "quarantine", "old_denoiser", "production_hbt")
    }
    assets["new_denoiser"] = {
        "path": str(new_denoiser),
        "sha256": actual_new_sha,
        "bytes": new_denoiser.stat().st_size,
    }
    splits = _verify_splits(template)

    runtime = deepcopy(template)
    runtime["status"] = "runtime_pinned_ready_for_explicit_kaggle_push"
    runtime["authoritative_inputs"]["new_denoiser"].update(
        {"path": str(new_denoiser), "sha256": actual_new_sha}
    )
    runtime["launch_interlock"].update(
        {
            "gpu_training_authorized_now": True,
            "kaggle_push_authorized_now": True,
            "candidate_graph_oracle_verdict": args.oracle_verdict,
            "root_launch_signal": args.root_launch_signal,
        }
    )
    runtime["runtime_pin"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "assets": assets,
        "splits": splits,
    }
    validate_protocol_safety(runtime)
    runtime_sha = canonical_sha256(runtime)
    runtime["runtime_pin"]["canonical_config_sha256_without_this_field"] = runtime_sha
    receipt = {
        "schema_version": 1,
        "kind": "solver_denoiser_adaptation_runtime_pin_receipt",
        "safe_for_kaggle_packaging": True,
        "oracle_verdict": args.oracle_verdict,
        "runtime_config_canonical_sha256": canonical_sha256(runtime),
        "new_denoiser_sha256": actual_new_sha,
        "assets": assets,
        "splits": splits,
    }
    return runtime, receipt


def main() -> None:
    args = parse_args()
    output = _resolve(args.output)
    receipt_path = _resolve(args.receipt)
    if output.exists() or receipt_path.exists():
        raise SystemExit("output or receipt already exists; choose fresh paths")
    runtime, receipt = prepare(args)
    _atomic_json(output, runtime)
    _atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "event": "solver_denoiser_adaptation_runtime_pinned",
                "output": str(output),
                "receipt": str(receipt_path),
                "runtime_config_sha256": sha256_file(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
