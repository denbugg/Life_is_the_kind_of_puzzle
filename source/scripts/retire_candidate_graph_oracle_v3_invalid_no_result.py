#!/usr/bin/env python3
"""Emit the fail-closed, input-only v3 INVALID_NO_RESULT disposition."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = "4f3da49d17e8adba46b1359d2cc81a19"
CONFIG_SHA = "4dcce283cab00f276ae29ccfb2f82a41edc2d6b7c2e23507341701da2da2c1aa"
FROZEN_SHA = "5e7b8c1515c0d216e995b711cabbc59d5508518d688b80172c8a1bbe3e362ba4"
PRODUCER_SHA = "7723d18b86d1181954117a2c813da0cb45948ccd415f47c2d2dce6575e8a3377"
VERIFIER_SHA = "f0df97c42e0354b37ec626828a81347c526d3d580fff5dcdc6fb4e1c068af4d8"
EXPECTED = {
    "phase_a_manifest": "ee9d801458b22be066d21ec296836346c137a495b9f71295c378c7492599c7f1",
    "wrapper": "43b16b21d866d68142f380626832266a5ba43f3195bdcdf5bb119f2fdcfedcf1",
    "launch_receipt": "6973ba816ffc5991aca3c12f9e5f1a8d26083fc31b52f4c94f724573f09c5ef4",
    "recovered_launch_verification": "09df81f1733eafa8c38bb7cc6a201167badb1ea50a76e4db2ec45abe2e120941",
    "prelaunch_bound_repository_closure": "410afa26c48ad42fcfb15cf881b15aaf6f84d12032c93744f229c1a107b5d278",
    "first_composite_incident": "3ce6bb1a30adfa4764a7de2a565d72a8a673a0433bfee371318eb74d72531218",
    "second_composite_incident": "7a9109ff6568d0c3f5648aa808edbe616cacd9ab1a27d6820400dda3996db870",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_object(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_file(payload: Mapping[str, Any]) -> bytes:
    return _canonical_object(payload) + b"\n"


def _read(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, f"unsafe file: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _sha(path: Path) -> str:
    return hashlib.sha256(_read(path)).hexdigest()


def _load(path: Path, expected_sha: str) -> dict[str, Any]:
    raw = _read(path)
    _require(hashlib.sha256(raw).hexdigest() == expected_sha, f"SHA drift: {path}")
    payload = json.loads(raw.decode("utf-8"))
    _require(isinstance(payload, dict), f"not a JSON object: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = _canonical_file(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
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


def main() -> None:
    readback = ROOT / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_readback"
    job = ROOT / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_job"
    ledger = ROOT / f"runs/assembly_v1/protocol_ledgers/candidate_graph_oracle/{INSTANCE}"
    paths = {
        "phase_a_manifest": readback
        / "candidate_graph_oracle_v3_phase_a/finalized/FROZEN_CANDIDATE_GRAPH_MANIFEST.json",
        "wrapper": readback / "candidate_graph_oracle_v3_phase_a_wrapper.json",
        "launch_receipt": job / "CANDIDATE_GRAPH_ORACLE_V3_KAGGLE_LAUNCH_RECEIPT.json",
        "recovered_launch_verification": job / "V3_RECOVERED_LAUNCH_VERIFICATION_COMPLETE.json",
        "prelaunch_bound_repository_closure": readback
        / "PRELAUNCH_BOUND_VERIFIER_REPOSITORY_CLOSURE.json",
        "first_composite_incident": readback / "FIRST_COMPOSITE_ATTEMPT_INCIDENT.md",
        "second_composite_incident": readback / "SECOND_COMPOSITE_ATTEMPT_INCIDENT.md",
    }
    for label, path in paths.items():
        _require(_sha(path) == EXPECTED[label], f"retirement evidence drift: {label}")
    _require(
        not (readback / "CANDIDATE_GRAPH_ORACLE_V3_PHASE_A_COMPOSITE_VERIFICATION.json").exists(),
        "accepted composite output unexpectedly exists",
    )
    _require(not (ledger / "LABEL_ACCESS.json").exists(), "LABEL_ACCESS exists")
    _require(
        {item.name for item in ledger.iterdir()}
        == {"PREP.json", "SEALED.json", "PHASE_A.json", "runtime_pin_transitions"},
        "v3 lifecycle tree drift",
    )
    manifest = _load(paths["phase_a_manifest"], EXPECTED["phase_a_manifest"])
    records = manifest.get("records")
    _require(isinstance(records, list) and len(records) == 64, "manifest record-count drift")
    _require(
        all(
            isinstance(record, dict)
            and isinstance(record.get("derivation_diagnostics"), dict)
            and set(record["derivation_diagnostics"])
            == {"hbt_outside_logits", "qap", "softcycle"}
            for record in records
        ),
        "producer schema mismatch is not universal 64/64",
    )
    evidence = {
        label: {"path": str(path.relative_to(ROOT)), "sha256": EXPECTED[label]}
        for label, path in paths.items()
    }
    payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_protocol_invalid_no_result",
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "protocol_instance_id": INSTANCE,
        "config_sha256": CONFIG_SHA,
        "frozen_contract_sha256": FROZEN_SHA,
        "disposition": "INVALID_NO_RESULT",
        "reason_code": "FROZEN_PRODUCER_VERIFIER_DERIVATION_DIAGNOSTICS_SCHEMA_MISMATCH",
        "reason": {
            "producer_sha256": PRODUCER_SHA,
            "producer_key": "hbt_outside_logits",
            "verifier_sha256": VERIFIER_SHA,
            "verifier_required_key": "hbt",
            "affected_phase_a_records": 64,
            "total_phase_a_records": 64,
        },
        "evidence": evidence,
        "phase_a_input_only_artifacts_preserved": True,
        "accepted_aggregate_metrics": None,
        "continuation_gate_passed": False,
        "phase_b_authorized": False,
        "label_access_lifecycle_claimed": False,
        "label_paths_constructed": False,
        "label_files_opened": False,
        "protocol_instance_consumed": True,
        "rerun_allowed": False,
        "safe_for_submission": False,
    }
    envelope = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical_object(payload)).hexdigest(),
    }
    output = readback / "CANDIDATE_GRAPH_ORACLE_V3_INVALID_NO_RESULT.json"
    _require(not output.exists(), "retirement output already exists")
    output_sha = _write(output, envelope)
    print(
        json.dumps(
            {
                "status": "retired_invalid_no_result",
                "output": str(output),
                "output_sha256": output_sha,
                "payload_sha256": envelope["payload_sha256"],
                "label_access_claimed_or_opened": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
