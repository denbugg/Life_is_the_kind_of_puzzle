#!/usr/bin/env python3
"""Independently verify the v3 raw-derived launch receipt and live server state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = "4f3da49d17e8adba46b1359d2cc81a19"
KERNEL = "pasha883/vsos-candidate-graph-oracle-v3-phase-a-t4x2"
KERNEL_ID = 126846203
DATASETS = {
    "code": "pasha883/vsos-candidate-graph-oracle-v3-code",
    "input": "pasha883/vsos-candidate-graph-oracle-v3-inputs",
    "runtime": "pasha883/vsos-candidate-graph-oracle-v3-runtime",
}
EXPECTED = {
    "config": "4dcce283cab00f276ae29ccfb2f82a41edc2d6b7c2e23507341701da2da2c1aa",
    "launcher": "cb0e308fbc309de4e96f684ad405f2cf23d40a7bf2ab675afea374a7a0fff243",
    "runner": "4dd0497701131d450aae57614e3b8a33ae75ff080e04fcf5037f3728b827ccc9",
    "metadata": "bc9260285acb285579ef3088aa1cdc31795a8796e1d54e72fdb93db84fc3b2d9",
    "intent": "610d2085d7aae2edc3d5680f92a9185301b0f0b7ae6cecdf35fb05f320ca15a6",
    "raw": "78846f0df32df680b18e3e9e2299da8ba6d209f854ad7afc492d92fa5208b2b2",
    "parser": "e137c2533e706a3d6a67febcb3b9854a85d1be64520dec652e7e68939bdfedbc",
    "normalization": "b3369f0e3f5b6d68fdc14f1ffb1d15199c895f0644bd13b8fa3e693f40249ce7",
    "derived": "0ed668dec3f5a67e74612a11a3b3c0e90f3b7fd52547a7e75bb9866e7ef1afd6",
    "receipt": "6973ba816ffc5991aca3c12f9e5f1a8d26083fc31b52f4c94f724573f09c5ef4",
}


def _canonical_file(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _canonical_object(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _read(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"not a one-link regular file: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _sha(path: Path) -> str:
    return hashlib.sha256(_read(path)).hexdigest()


def _load(path: Path, expected_sha: str) -> dict[str, Any]:
    raw = _read(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError(f"SHA drift: {path}")
    payload = json.loads(raw.decode())
    if not isinstance(payload, dict) or raw != _canonical_file(payload):
        raise RuntimeError(f"noncanonical JSON object: {path}")
    return payload


def _require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def _write(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_file(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
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


def verify(*, output: Path) -> dict[str, Any]:
    job = ROOT / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_job"
    state = job / "candidate_graph_oracle_v3_launch_state"
    expected_state_files = {
        "00_launch.intent.json",
        "01_push.raw_response.json",
        "01b_push.ref_normalization.json",
        "02_push.response.json",
    }
    _require({item.name for item in state.iterdir()} == expected_state_files, "state tree drift")
    _require(_sha(ROOT / "configs/candidate_graph_oracle_ceiling_v3.json") == EXPECTED["config"], "config drift")
    _require(_sha(ROOT / "scripts/push_candidate_graph_oracle_phase_a.py") == EXPECTED["launcher"], "launcher drift")
    _require(_sha(ROOT / "scripts/recover_candidate_graph_oracle_v3_launch_from_raw.py") == EXPECTED["parser"], "parser drift")
    _require(_sha(job / "run_phase_a.py") == EXPECTED["runner"], "runner drift")
    _require(_sha(job / "kernel-metadata.json") == EXPECTED["metadata"], "metadata drift")

    intent = _load(state / "00_launch.intent.json", EXPECTED["intent"])
    raw = _load(state / "01_push.raw_response.json", EXPECTED["raw"])
    normalization = _load(
        state / "01b_push.ref_normalization.json", EXPECTED["normalization"]
    )
    derived = _load(state / "02_push.response.json", EXPECTED["derived"])
    envelope = _load(
        job / "CANDIDATE_GRAPH_ORACLE_V3_KAGGLE_LAUNCH_RECEIPT.json",
        EXPECTED["receipt"],
    )
    _require(set(envelope) == {"payload", "payload_sha256"}, "receipt envelope drift")
    receipt = envelope["payload"]
    _require(
        envelope["payload_sha256"] == hashlib.sha256(_canonical_object(receipt)).hexdigest(),
        "receipt self hash drift",
    )
    _require(intent.get("protocol_instance_id") == INSTANCE, "intent instance drift")
    raw_fields = raw.get("public_fields")
    _require(isinstance(raw_fields, dict), "raw fields missing")
    _require(raw_fields.get("ref") == f"/code/{KERNEL}", "unexpected raw ref")
    _require(
        raw_fields.get("kernel_id") == KERNEL_ID
        and raw_fields.get("version_number") == 2
        and raw_fields.get("error") == ""
        and all(
            raw_fields.get(key) == []
            for key in (
                "invalid_tags",
                "invalid_dataset_sources",
                "invalid_competition_sources",
                "invalid_kernel_sources",
                "invalid_model_sources",
            )
        ),
        "raw non-ref fields drift",
    )
    _require(
        normalization.get("raw_response_sha256") == EXPECTED["raw"]
        and normalization.get("recovery_parser_sha256") == EXPECTED["parser"]
        and normalization.get("before") == {"ref": f"/code/{KERNEL}"}
        and normalization.get("after") == {"ref": KERNEL}
        and normalization.get("remote_write_performed") is False,
        "normalization receipt drift",
    )
    _require(
        derived.get("kind")
        == "candidate_graph_oracle_kaggle_push_response_derived_from_raw"
        and derived.get("derived_from_raw") is True
        and derived.get("ref") == KERNEL
        and derived.get("kernel_id") == KERNEL_ID
        and derived.get("version_number") == 2
        and derived.get("raw_response_sha256") == EXPECTED["raw"]
        and derived.get("normalization_receipt_sha256") == EXPECTED["normalization"]
        and derived.get("recovery_parser_sha256") == EXPECTED["parser"],
        "derived response drift",
    )
    journal = receipt.get("launch_journal")
    _require(isinstance(journal, dict), "receipt journal missing")
    _require(
        journal.get("intent_sha256") == EXPECTED["intent"]
        and journal.get("raw_push_response_sha256") == EXPECTED["raw"]
        and journal.get("normalization_receipt_sha256") == EXPECTED["normalization"]
        and journal.get("derived_push_response_sha256") == EXPECTED["derived"],
        "receipt journal hash drift",
    )
    _require(
        receipt.get("protocol_instance_id") == INSTANCE
        and receipt.get("response_provenance")
        == "derived_from_immutable_raw_sdk_response"
        and receipt.get("push_performed_in_this_process") is False
        and receipt.get("remote_write_performed_by_recovery") is False,
        "receipt provenance drift",
    )

    ledger = ROOT / f"runs/assembly_v1/protocol_ledgers/candidate_graph_oracle/{INSTANCE}"
    _require(not (ledger / "LABEL_ACCESS.json").exists(), "LABEL_ACCESS exists")
    api = KaggleApi()
    api.authenticate()
    dataset_state: dict[str, Any] = {}
    for label, slug in DATASETS.items():
        observed = json.loads(
            api.dataset_status(slug, format="json(status,current_version_number)")
        )
        _require(
            observed == {"status": "ready", "current_version_number": 2},
            f"remote dataset drift: {label}",
        )
        dataset_state[label] = observed
    owner, slug = KERNEL.split("/", 1)
    request = ApiGetKernelRequest()
    request.user_name = owner
    request.kernel_slug = slug
    with api.build_kaggle_client() as client:
        current = client.kernels.kernels_api_client.get_kernel(request)
    metadata = current.metadata
    _require(metadata is not None and current.blob is not None, "remote kernel readback incomplete")
    source_sha = hashlib.sha256(current.blob.source.encode()).hexdigest()
    _require(
        int(metadata.id) == KERNEL_ID
        and str(metadata.ref) == KERNEL
        and int(metadata.current_version_number) == 2
        and list(metadata.dataset_data_sources or []) == list(DATASETS.values())
        and source_sha == EXPECTED["runner"],
        "remote kernel identity/version/source drift",
    )
    session = api.kernels_status(KERNEL)
    session_status = str(session.status)
    _require(
        session_status in {"KernelWorkerStatus.RUNNING", "KernelWorkerStatus.COMPLETE"},
        f"unexpected remote session status: {session_status}",
    )
    _require(session.failure_message in (None, ""), "remote session reports failure")

    payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v3_recovered_launch_verification",
        "verified_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "verified_running" if session_status.endswith("RUNNING") else "verified_complete",
        "protocol_instance_id": INSTANCE,
        "kernel": {
            "slug": KERNEL,
            "id": KERNEL_ID,
            "version": 2,
            "source_sha256": source_sha,
            "session_status": session_status,
        },
        "datasets": dataset_state,
        "intent_sha256": EXPECTED["intent"],
        "raw_response_sha256": EXPECTED["raw"],
        "normalization_receipt_sha256": EXPECTED["normalization"],
        "derived_response_sha256": EXPECTED["derived"],
        "launch_receipt_sha256": EXPECTED["receipt"],
        "recovery_parser_sha256": EXPECTED["parser"],
        "raw_ref_before": f"/code/{KERNEL}",
        "canonical_ref_after": KERNEL,
        "remote_write_performed_by_recovery": False,
        "kernel_version_advance_from_reservation": 1,
        "label_access_claim_present": False,
        "label_fixture_opened_by_verifier": False,
        "safe_for_submission": False,
    }
    verification_sha = _write(output, payload)
    return {"verification_path": str(output), "verification_sha256": verification_sha, **payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(output=args.output.absolute()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
