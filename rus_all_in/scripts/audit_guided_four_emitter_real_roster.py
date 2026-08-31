#!/usr/bin/env python3
"""Audit whether a fresh source-disjoint DEV32 still exists for four emitters."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aiijc_puzzle.four_emitter_real_roster import (
    deterministic_fresh_roster,
    minimal_inventory_blocker,
    names_digest,
    recursive_json_roster_inventory,
)
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
PRIOR_SNAPSHOT = (
    PROJECT_ROOT / "configs/pair_safe_cyclic_origin_fit_audit_exclusions_v1.json"
)
FINAL_SEVEN_CONFIG = PROJECT_ROOT / "configs/pair_safe_cyclic_origin_fit_audit_v1.json"
FINAL_SEVEN_REPORT = PROJECT_ROOT / "outputs/pair-safe-cyclic-origin-fit-audit/v1/report.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/guided-fourth-emitter/real-protocol-roster-audit-v1/report.json"
)
REPORT_SCHEMA = "aiijc-guided-four-emitter-real-roster-audit-v1"
SNAPSHOT_SCHEMA = "aiijc-pair-safe-cyclic-origin-fit-audit-exclusions-v1"
SELECTION_NAMESPACE = "aiijc-guided-four-emitter-real-dev32-v1"
SELECTION_SEED = 20260919
DEV_COUNT = 32


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved)}


def _load_signed_json(path: Path, *, schema: str | None = None) -> Mapping[str, Any]:
    sidecar = Path(f"{path}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"signed JSON is missing: {path}")
    digest = sha256_file(path)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError(f"signed JSON sidecar mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or (
        schema is not None and payload.get("schema") != schema
    ):
        raise RuntimeError(f"signed JSON schema mismatch: {path}")
    return payload


def _manifest_train(path: Path) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise RuntimeError("organizer manifest protocol digest mismatch")
    train = tuple(sorted(str(row["filename"]) for row in manifest["splits"]["train"]))
    if len(train) != 5600 or len(set(train)) != len(train):
        raise RuntimeError("organizer-train inventory changed")
    return manifest, train


def build_report(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    del output_path
    _, train = _manifest_train(manifest_path)
    snapshot = _load_signed_json(PRIOR_SNAPSHOT, schema=SNAPSHOT_SCHEMA)
    prior_union = tuple(snapshot["prior_source_union"]["source_filenames"])
    if (
        prior_union != tuple(sorted(set(prior_union)))
        or names_digest(prior_union) != snapshot["prior_source_union"]["digest"]
    ):
        raise RuntimeError("signed prior source union changed")
    eligible_before = tuple(snapshot["organizer_train_audit"]["eligible_train_filenames"])
    prior_train = tuple(sorted(set(train) & set(prior_union)))
    if len(prior_train) != 5593 or len(eligible_before) != 7:
        raise RuntimeError("signed prior complement no longer proves a 5593+7 partition")
    if set(prior_train) & set(eligible_before) or set(prior_train) | set(
        eligible_before
    ) != set(train):
        raise RuntimeError("signed prior complement does not exactly partition organizer train")

    final_config = _load_signed_json(FINAL_SEVEN_CONFIG)
    final_sources = tuple(final_config["panel"]["source_filenames"])
    final_report = json.loads(FINAL_SEVEN_REPORT.read_text(encoding="utf-8"))
    report_sources = tuple(final_report["panel"]["source_filenames"])
    row_sources = {str(row["source_filename"]) for row in final_report["rows"]}
    if (
        set(final_sources) != set(eligible_before)
        or report_sources != final_sources
        or row_sources != set(final_sources)
        or len(final_report["rows"]) != 56
    ):
        raise RuntimeError("the signed seven-source complement was not fully scored")

    reserved = (DEFAULT_OUTPUT,)
    artifacts, current_union, inventory_digest = recursive_json_roster_inventory(
        PROJECT_ROOT, reserved=reserved
    )
    roster, eligible_now = deterministic_fresh_roster(
        train,
        current_union,
        count=DEV_COUNT,
        namespace=SELECTION_NAMESPACE,
        seed=SELECTION_SEED,
    )
    blocker = minimal_inventory_blocker(train, prior_train, final_sources)
    current_train_union = tuple(sorted(set(train) & set(current_union)))
    if (
        not blocker["complete_inventory_blocker"]
        or current_train_union != train
        or eligible_now
        or roster
    ):
        raise RuntimeError("recursive inventory unexpectedly leaves a fresh DEV source")

    return {
        "schema": REPORT_SCHEMA,
        "status": "blocked-no-source-disjoint-organizer-train-dev32",
        "scope": {
            "metadata_only": True,
            "pixels_loaded": False,
            "labels_or_exact_references_loaded": False,
            "models_or_predictions_run": False,
            "dev_local_terminal_test_or_submission_accessed": False,
        },
        "semantic_rule": (
            "recursively union every explicit source-panel *_filenames roster and "
            "every row-wise source_filename from configs/**/*.json and outputs/**/*.json; "
            "the organizer manifest universe is not scanned as an experiment artifact"
        ),
        "recursive_inventory": {
            "artifact_count": len(artifacts),
            "artifact_inventory_digest": inventory_digest,
            "source_union_count_all_manifest_splits": len(current_union),
            "source_union_digest": names_digest(current_union),
            "organizer_train_count": len(train),
            "organizer_train_digest": names_digest(train),
            "excluded_train_count": len(current_train_union),
            "eligible_train_count": len(eligible_now),
            "eligible_train_filenames": list(eligible_now),
            "artifacts": [item.__dict__ for item in artifacts],
        },
        "minimal_inventory_blocker": {
            **blocker,
            "signed_prior_snapshot": _record(PRIOR_SNAPSHOT),
            "signed_prior_snapshot_excluded_train_count": 5593,
            "signed_prior_snapshot_complement": list(eligible_before),
            "final_complement_config": _record(FINAL_SEVEN_CONFIG),
            "final_complement_report": _record(FINAL_SEVEN_REPORT),
            "final_complement_case_count": len(final_report["rows"]),
            "proof": (
                "The earlier signed recursive snapshot excluded 5593 organizer-train "
                "sources and left exactly seven. Its later signed FIT audit used and "
                "scored all seven, so 5593 + 7 covers all 5600 train sources."
            ),
        },
        "proposed_dev32": {
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_seed": SELECTION_SEED,
            "requested_source_count": DEV_COUNT,
            "source_filenames": list(roster),
            "status": "not-created-inventory-blocked",
            "blocked_reason": (
                "zero organizer-train sources remain outside all recursively declared "
                "matcher, solver, fit and scored-panel rosters"
            ),
        },
        "decision": {
            "real_protocol_may_be_signed": False,
            "unsigned_implementation_template_allowed": True,
            "required_external_change": (
                "new organizer-labelled sources or an explicitly authorised reuse policy "
                "whose non-fresh claim and role are signed before any scoring"
            ),
        },
        "artifacts": {
            "manifest": _record(manifest_path),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/four_emitter_real_roster.py"
            ),
            "runner": _record(Path(__file__)),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args.manifest.resolve(), args.output.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "artifact_count": report["recursive_inventory"]["artifact_count"],
                "excluded_train_count": report["recursive_inventory"][
                    "excluded_train_count"
                ],
                "eligible_train_count": report["recursive_inventory"][
                    "eligible_train_count"
                ],
                "report": _record(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
