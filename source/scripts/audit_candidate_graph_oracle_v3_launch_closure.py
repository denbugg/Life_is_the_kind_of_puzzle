#!/usr/bin/env python3
"""Fail-closed, label-blind audit of the v3 Phase-A launch closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCE = "4f3da49d17e8adba46b1359d2cc81a19"
FROZEN_SHA256 = "5e7b8c1515c0d216e995b711cabbc59d5508518d688b80172c8a1bbe3e362ba4"
CONFIG_SHA256 = "4dcce283cab00f276ae29ccfb2f82a41edc2d6b7c2e23507341701da2da2c1aa"
KERNEL_SLUG = "pasha883/vsos-candidate-graph-oracle-v3-phase-a-t4x2"
KERNEL_ID = 126846203
DATASETS = {
    "code": "pasha883/vsos-candidate-graph-oracle-v3-code",
    "input": "pasha883/vsos-candidate-graph-oracle-v3-inputs",
    "runtime": "pasha883/vsos-candidate-graph-oracle-v3-runtime",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_file_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _canonical_object_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _read_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"not a one-link regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _load(path: Path) -> tuple[dict[str, Any], bytes, str]:
    raw = _read_regular(path)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload, raw, hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _hash(path: Path) -> str:
    return hashlib.sha256(_read_regular(path)).hexdigest()


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_file_bytes(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return hashlib.sha256(encoded).hexdigest()


def audit(*, output: Path) -> dict[str, Any]:
    config_path = REPO_ROOT / "configs/candidate_graph_oracle_ceiling_v3.json"
    config, config_raw, config_sha = _load(config_path)
    _require(config_sha == CONFIG_SHA256, "fully pinned config SHA drift")
    _require(
        config_raw == (json.dumps(config, ensure_ascii=True, indent=2) + "\n").encode(),
        "config is not canonical repository JSON",
    )
    _require(config.get("protocol_instance_id") == INSTANCE, "config instance drift")
    frozen = config.get("frozen_contract")
    _require(isinstance(frozen, dict), "missing frozen contract")
    frozen_sha = hashlib.sha256(_canonical_object_bytes(frozen)).hexdigest()
    _require(
        frozen_sha == FROZEN_SHA256 == config.get("frozen_contract_sha256"),
        "frozen contract SHA drift",
    )
    _require(
        frozen.get("protocol_instance", {}).get("exact_value") == INSTANCE,
        "frozen instance drift",
    )
    pins = config.get("runtime_pins")
    policy = config.get("runtime_pin_mutation_policy")
    _require(isinstance(pins, dict) and isinstance(policy, dict), "runtime pins missing")
    _require(
        policy.get("transition_ledger_root")
        == f"runs/assembly_v1/protocol_ledgers/candidate_graph_oracle/{INSTANCE}",
        "ledger root drift",
    )
    verified_pin_hashes: dict[str, str] = {}
    for pair in policy["code_pin_fields"]:
        path_field = pair["path_field"]
        sha_field = pair["sha256_field"]
        expected = pins.get(sha_field)
        _require(isinstance(expected, str) and SHA_RE.fullmatch(expected) is not None, f"unpopulated {sha_field}")
        actual = _hash(REPO_ROOT / pins[path_field])
        _require(actual == expected, f"runtime pin drift: {sha_field}")
        verified_pin_hashes[sha_field] = actual

    fixture_root = REPO_ROOT / "runs/assembly_v1/candidate_graph_oracle_fixtures_v3"
    input_manifest_path = fixture_root / "fixture_input/fixture_input_manifest.json"
    lock_path = fixture_root / "fixture_control/fixture_lock.json"
    input_manifest, _, input_sha = _load(input_manifest_path)
    lock, _, lock_sha = _load(lock_path)
    _require(input_sha == pins["fixture_input_manifest_sha256"], "input pin drift")
    _require(lock_sha == pins["fixture_lock_sha256"], "lock pin drift")
    for name, payload in (("input manifest", input_manifest), ("fixture lock", lock)):
        _require(payload.get("protocol_instance_id") == INSTANCE, f"{name} instance drift")
        _require(payload.get("frozen_contract_sha256") == FROZEN_SHA256, f"{name} frozen drift")
        _require(payload.get("record_count") == 64, f"{name} record count drift")
        for field, digest in verified_pin_hashes.items():
            _require(payload.get(field) == digest, f"{name} code binding drift: {field}")
    _require(
        lock.get("fixture_input_manifest_sha256") == input_sha
        and lock.get("fixture_label_manifest_sha256")
        == pins["fixture_label_manifest_sha256"],
        "fixture lock crosslink drift",
    )
    _require(lock.get("phase_a_may_receive_label_root") is False, "label root allowed")
    _require(lock.get("phase_a_may_receive_master_secret") is False, "secret allowed")

    ledger_root = REPO_ROOT / policy["transition_ledger_root"]
    expected_ledger = {"PREP.json", "SEALED.json", "PHASE_A.json", "runtime_pin_transitions"}
    _require({item.name for item in ledger_root.iterdir()} == expected_ledger, "ledger closure drift")
    _require(not (ledger_root / "LABEL_ACCESS.json").exists(), "LABEL_ACCESS already exists")
    lifecycle_hashes: dict[str, str] = {}
    for state in ("PREP", "SEALED", "PHASE_A"):
        payload, raw, digest = _load(ledger_root / f"{state}.json")
        _require(raw == _canonical_file_bytes(payload), f"{state} noncanonical")
        _require(payload.get("protocol_instance_id") == INSTANCE, f"{state} instance drift")
        _require(payload.get("frozen_contract_sha256") == FROZEN_SHA256, f"{state} frozen drift")
        if state != "PREP":
            _require(payload.get("config_sha256_or_null") == CONFIG_SHA256, f"{state} config drift")
        lifecycle_hashes[state] = digest

    receipt_path = REPO_ROOT / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_bundles/CANDIDATE_GRAPH_ORACLE_KAGGLE_BUNDLE_BUILD_RECEIPT.json"
    envelope, receipt_raw, receipt_sha = _load(receipt_path)
    _require(receipt_raw == _canonical_file_bytes(envelope), "bundle receipt noncanonical")
    _require(set(envelope) == {"payload", "payload_sha256"}, "bundle envelope drift")
    bundle = envelope["payload"]
    _require(
        envelope["payload_sha256"] == hashlib.sha256(_canonical_object_bytes(bundle)).hexdigest(),
        "bundle self hash drift",
    )
    _require(bundle.get("protocol_instance_id") == INSTANCE, "bundle instance drift")
    _require(bundle.get("frozen_contract_sha256") == FROZEN_SHA256, "bundle frozen drift")
    _require(bundle.get("fully_pinned_config_sha256") == CONFIG_SHA256, "bundle config drift")
    _require(bundle.get("lifecycle_terminal_state") == "PHASE_A", "bundle lifecycle drift")
    _require(bundle.get("input_payload_decoded") is False, "input payload was decoded")
    archives: dict[str, str] = {}
    for label, slug in DATASETS.items():
        descriptor = bundle["datasets"][label]
        _require(descriptor.get("slug") == slug, f"bundle dataset slug drift: {label}")
        _require(descriptor.get("expected_version") == 2, f"bundle dataset version drift: {label}")
        _require(descriptor.get("must_remain_private") is True, f"dataset privacy drift: {label}")
        archive = descriptor["archive"]
        archive_path = receipt_path.parent / archive["path"]
        actual = _hash(archive_path)
        _require(actual == archive["sha256"], f"archive hash drift: {label}")
        archives[label] = actual

    metadata_path = REPO_ROOT / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_job/kernel-metadata.json"
    metadata, _, metadata_sha = _load(metadata_path)
    _require(metadata_sha == pins["phase_a_kernel_metadata_sha256"], "kernel metadata pin drift")
    _require(
        metadata.get("id") == KERNEL_SLUG
        and metadata.get("id_no") == KERNEL_ID
        and metadata.get("is_private") is True
        and metadata.get("enable_gpu") is True
        and metadata.get("enable_internet") is False
        and metadata.get("machine_shape") == "NvidiaTeslaT4",
        "kernel metadata identity/hardware drift",
    )
    expected_sources = [f"{slug}/2" for slug in DATASETS.values()]
    _require(metadata.get("dataset_sources") == expected_sources, "kernel dataset source drift")
    expectation = metadata.get("oracle_launch_expectation")
    _require(
        expectation
        == {
            "kernel_id": KERNEL_ID,
            "kernel_slug": KERNEL_SLUG,
            "kernel_version": 2,
            "dataset_versions": {
                label: {"slug": slug, "version": 2}
                for label, slug in DATASETS.items()
            },
        },
        "kernel launch expectation drift",
    )

    reservation_path = REPO_ROOT / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_reservations/RESERVATION_RECEIPT.json"
    reservation, _, reservation_sha = _load(reservation_path)
    _require(reservation.get("protocol_instance_id") == INSTANCE, "reservation instance drift")
    _require(reservation["kernel"]["kernel_id"] == KERNEL_ID, "reservation kernel id drift")
    _require(reservation["kernel"]["reserved_version"] == 1, "reservation version drift")
    for label, slug in DATASETS.items():
        _require(
            reservation["datasets"][label]["slug"] == slug
            and reservation["datasets"][label]["reserved_version"] == 1,
            f"reservation dataset drift: {label}",
        )

    payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v3_prelaunch_closure_audit",
        "status": "ready_for_dataset_v2_upload_then_single_phase_a_push",
        "protocol_instance_id": INSTANCE,
        "frozen_contract_sha256": FROZEN_SHA256,
        "fully_pinned_config_sha256": CONFIG_SHA256,
        "lifecycle_terminal_state": "PHASE_A",
        "lifecycle_sha256": lifecycle_hashes,
        "bundle_receipt_sha256": receipt_sha,
        "dataset_archive_sha256": archives,
        "reservation_receipt_sha256": reservation_sha,
        "kernel": {
            "slug": KERNEL_SLUG,
            "id": KERNEL_ID,
            "current_required_prelaunch_version": 1,
            "single_push_intended_version": 2,
        },
        "datasets": {
            label: {
                "slug": slug,
                "current_required_preupload_version": 1,
                "uploaded_intended_version": 2,
            }
            for label, slug in DATASETS.items()
        },
        "label_access_claim_present": False,
        "label_manifest_opened_by_audit": False,
        "label_records_opened_by_audit": False,
        "safe_for_submission": False,
    }
    audit_sha = _write_exclusive(output, payload)
    return {"audit_path": str(output), "audit_sha256": audit_sha, **payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(output=args.output.absolute()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
