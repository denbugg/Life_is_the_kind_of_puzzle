#!/usr/bin/env python3
"""Derive a v3 launch receipt from the immutable SDK raw-response journal.

This recovery performs no remote write.  It accepts exactly one observed SDK
alias: ``/code/{owner}/{slug}`` for the already pinned canonical kernel ref.
Every other public response field must match the frozen launch expectation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from kaggle.api.kaggle_api_extended import KaggleApi

from scripts import push_candidate_graph_oracle_phase_a as launch


EXPECTED_INTENT_SHA256 = "610d2085d7aae2edc3d5680f92a9185301b0f0b7ae6cecdf35fb05f320ca15a6"
EXPECTED_RAW_SHA256 = "78846f0df32df680b18e3e9e2299da8ba6d209f854ad7afc492d92fa5208b2b2"
NORMALIZATION_NAME = "01b_push.ref_normalization.json"
NORMALIZATION_KIND = "candidate_graph_oracle_kaggle_raw_ref_normalization"
DERIVED_RESPONSE_KIND = "candidate_graph_oracle_kaggle_push_response_derived_from_raw"


def _canonical_object_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(launch._canonical_object_bytes(payload)).hexdigest()


def _expected_public_fields() -> dict[str, Any]:
    return {
        "ref": f"/code/{launch.EXPECTED_KERNEL_SLUG}",
        "url": (
            "https://www.kaggle.com/code/pasha883/"
            "vsos-candidate-graph-oracle-v3-phase-a-t4x2"
        ),
        "version_number": launch.EXPECTED_KERNEL_VERSION,
        "error": "",
        "invalid_tags": [],
        "invalid_dataset_sources": [],
        "invalid_competition_sources": [],
        "invalid_kernel_sources": [],
        "invalid_model_sources": [],
        "kernel_id": launch.EXPECTED_KERNEL_ID,
    }


def recover(
    *,
    job_dir: Path,
    state_dir: Path,
    receipt_path: Path,
    api: KaggleApi | None = None,
) -> dict[str, Any]:
    job_dir = job_dir.expanduser().resolve(strict=True)
    state_dir = state_dir.expanduser().resolve(strict=True)
    receipt_path = receipt_path.expanduser().absolute()
    intent_path = state_dir / launch.INTENT_NAME
    raw_path = state_dir / launch.RAW_RESPONSE_NAME
    normalization_path = state_dir / NORMALIZATION_NAME
    response_path = state_dir / launch.RESPONSE_NAME
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RuntimeError("derived launch receipt path must be fresh")
    for fresh in (normalization_path, response_path):
        if fresh.exists() or fresh.is_symlink():
            raise RuntimeError(f"derived launch state path must be fresh: {fresh.name}")

    metadata_path = job_dir / "kernel-metadata.json"
    metadata = launch._load_json(metadata_path)
    runner_path = job_dir / str(metadata.get("code_file"))
    metadata_sha = launch._sha256(metadata_path)
    runner_sha = launch._sha256(runner_path)
    launcher_sha = launch._sha256(Path(launch.__file__).resolve())
    parser_path = Path(__file__).resolve(strict=True)
    parser_sha = launch._sha256(parser_path)

    kaggle = KaggleApi() if api is None else api
    if api is None:
        kaggle.authenticate()
    datasets_before = launch._dataset_versions(kaggle)
    intent, intent_sha = launch._load_canonical(
        intent_path,
        expected_kind="candidate_graph_oracle_kaggle_launch_intent",
    )
    if intent_sha != EXPECTED_INTENT_SHA256:
        raise RuntimeError("immutable launch-intent SHA drift")
    launch._validate_intent(
        intent,
        metadata_sha256=metadata_sha,
        runner_sha256=runner_sha,
        launcher_sha256=launcher_sha,
        dataset_versions=datasets_before,
    )
    raw, raw_sha = launch._load_canonical(
        raw_path,
        expected_kind=launch.RAW_RESPONSE_KIND,
        expected_schema_version=launch.RAW_RESPONSE_SCHEMA_VERSION,
    )
    launch._validate_raw_response_payload(raw)
    if raw_sha != EXPECTED_RAW_SHA256:
        raise RuntimeError("immutable raw SDK response SHA drift")
    if raw.get("public_fields") != _expected_public_fields():
        raise RuntimeError("raw SDK fields differ from the one allowed alias case")

    normalization = {
        "schema_version": 1,
        "kind": NORMALIZATION_KIND,
        "created_utc": launch._utc_now(),
        "protocol_instance_id": launch.EXPECTED_PROTOCOL_INSTANCE_ID,
        "launch_intent_file": launch.INTENT_NAME,
        "launch_intent_sha256": intent_sha,
        "raw_response_file": launch.RAW_RESPONSE_NAME,
        "raw_response_sha256": raw_sha,
        "recovery_parser_path": "scripts/recover_candidate_graph_oracle_v3_launch_from_raw.py",
        "recovery_parser_sha256": parser_sha,
        "normalization_rule": "remove_exact_leading_/code/_from_ref_only",
        "before": {"ref": f"/code/{launch.EXPECTED_KERNEL_SLUG}"},
        "after": {"ref": launch.EXPECTED_KERNEL_SLUG},
        "all_non_ref_public_fields_must_match_frozen_expectation": True,
        "remote_write_performed": False,
        "safe_for_submission": False,
    }
    normalization_sha = launch._write_exclusive(normalization_path, normalization)

    fields = deepcopy(raw["public_fields"])
    fields["ref"] = launch.EXPECTED_KERNEL_SLUG
    derived = {
        "schema_version": 3,
        "kind": DERIVED_RESPONSE_KIND,
        "ref": fields["ref"],
        "kernel_id": fields["kernel_id"],
        "version_number": fields["version_number"],
        "url": fields["url"],
        "error": fields["error"],
        "invalid_dataset_sources": fields["invalid_dataset_sources"],
        "invalid_competition_sources": fields["invalid_competition_sources"],
        "invalid_kernel_sources": fields["invalid_kernel_sources"],
        "invalid_model_sources": fields["invalid_model_sources"],
        "raw_response_file": launch.RAW_RESPONSE_NAME,
        "raw_response_sha256": raw_sha,
        "derived_from_raw": True,
        "normalization_receipt_file": NORMALIZATION_NAME,
        "normalization_receipt_sha256": normalization_sha,
        "recovery_parser_sha256": parser_sha,
        "recorded_utc": launch._utc_now(),
    }
    if (
        derived["ref"] != launch.EXPECTED_KERNEL_SLUG
        or derived["kernel_id"] != launch.EXPECTED_KERNEL_ID
        or derived["version_number"] != launch.EXPECTED_KERNEL_VERSION
        or derived["error"] not in (None, "")
        or any(
            derived[key]
            for key in (
                "invalid_dataset_sources",
                "invalid_competition_sources",
                "invalid_kernel_sources",
                "invalid_model_sources",
            )
        )
    ):
        raise RuntimeError("derived response violates frozen expectation")
    response_sha = launch._write_exclusive(response_path, derived)

    current = launch._readback_with_retry(
        kaggle,
        runner_sha256=runner_sha,
        title=str(metadata["title"]),
        attempts=10,
    )
    datasets_after = launch._dataset_versions(kaggle)
    if datasets_after != datasets_before:
        raise RuntimeError("dataset versions changed during raw recovery")
    if (
        launch._sha256(intent_path) != intent_sha
        or launch._sha256(raw_path) != raw_sha
        or launch._sha256(normalization_path) != normalization_sha
        or launch._sha256(response_path) != response_sha
    ):
        raise RuntimeError("launch recovery journal changed before receipt commit")

    receipt = {
        "schema_version": 3,
        "kind": "candidate_graph_oracle_kaggle_launch_receipt_derived_from_raw",
        "created_utc": launch._utc_now(),
        "protocol_instance_id": launch.EXPECTED_PROTOCOL_INSTANCE_ID,
        "kernel": {
            "slug": launch.EXPECTED_KERNEL_SLUG,
            "kernel_id": launch.EXPECTED_KERNEL_ID,
            "version": launch.EXPECTED_KERNEL_VERSION,
            "url": derived["url"],
        },
        "dataset_versions_before_recovery": datasets_before,
        "dataset_versions_after_recovery": datasets_after,
        "local_kernel_metadata_sha256": metadata_sha,
        "local_runner_sha256": runner_sha,
        "local_launcher_sha256": launcher_sha,
        "recovery_parser_path": "scripts/recover_candidate_graph_oracle_v3_launch_from_raw.py",
        "recovery_parser_sha256": parser_sha,
        "launch_journal": {
            "intent_file": launch.INTENT_NAME,
            "intent_sha256": intent_sha,
            "raw_push_response_file": launch.RAW_RESPONSE_NAME,
            "raw_push_response_sha256": raw_sha,
            "normalization_receipt_file": NORMALIZATION_NAME,
            "normalization_receipt_sha256": normalization_sha,
            "derived_push_response_file": launch.RESPONSE_NAME,
            "derived_push_response_sha256": response_sha,
        },
        "launch_intent": intent,
        "raw_push_response": raw,
        "normalization_receipt": normalization,
        "derived_push_response": derived,
        "server_readback": current,
        "gpu_and_machine_metadata_authority": (
            "executed_phase_a_wrapper_hardware_not_normalized_get_kernel_metadata"
        ),
        "push_performed_in_this_process": False,
        "remote_write_performed_by_recovery": False,
        "response_provenance": "derived_from_immutable_raw_sdk_response",
        "safe_for_submission": False,
    }
    receipt_sha = launch._exclusive_receipt(receipt_path, receipt)
    return {
        "status": "launch_receipt_recovered_from_raw_without_remote_write",
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "intent_sha256": intent_sha,
        "raw_response_sha256": raw_sha,
        "normalization_receipt_sha256": normalization_sha,
        "derived_response_sha256": response_sha,
        "recovery_parser_sha256": parser_sha,
        "remote_write_performed": False,
        "safe_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            recover(
                job_dir=args.job_dir,
                state_dir=args.state_dir,
                receipt_path=args.receipt,
            ),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
