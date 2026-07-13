#!/usr/bin/env python3
"""Verify v3 Phase A with the frozen input-only verifier plus recovered launch evidence.

The frozen production verifier predates the one-field Kaggle SDK ``/code/``
alias observed by the actual launch.  This driver therefore runs its production
input/graph/render path unchanged, deliberately omits only its normal-schema
launch-receipt branch, and binds that result to the separately verified
schema-v3 raw-response recovery.  It never accepts or constructs a label path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping
import zipfile


INSTANCE = "4f3da49d17e8adba46b1359d2cc81a19"
CONFIG_SHA256 = "4dcce283cab00f276ae29ccfb2f82a41edc2d6b7c2e23507341701da2da2c1aa"
FROZEN_CONTRACT_SHA256 = "5e7b8c1515c0d216e995b711cabbc59d5508518d688b80172c8a1bbe3e362ba4"
INPUT_MANIFEST_SHA256 = "6de4502908ccdbb74c262d63495792cc844f0faceb09f567ffcc8bd8dee9f444"
PHASE_A_MANIFEST_SHA256 = "ee9d801458b22be066d21ec296836346c137a495b9f71295c378c7492599c7f1"
SHARD_SHA256S = (
    "c9105b68f1601a19c5d823021efd77337b0a61d747d5ff6711f044c860e89dbb",
    "61e9290ab85476a8dc752e6c1f0554bcad6186f34c36740022e301548780e4ee",
)
SNAPSHOT_ARCHIVE_SHA256 = "aa8f050d4f01f6308236f51de414be868b367481de0e68228d2e7220952deab5"
FROZEN_VERIFIER_SHA256 = "f0df97c42e0354b37ec626828a81347c526d3d580fff5dcdc6fb4e1c068af4d8"
RECOVERED_VERIFIER_SHA256 = "43b115cab9bdb7c60f46a80ed0667fa17ae9f1a3afb429dd6d1b0ca39d697026"
RECOVERED_COMPLETE_SHA256 = "09df81f1733eafa8c38bb7cc6a201167badb1ea50a76e4db2ec45abe2e120941"
LAUNCH_RECEIPT_SHA256 = "6973ba816ffc5991aca3c12f9e5f1a8d26083fc31b52f4c94f724573f09c5ef4"
WRAPPER_SHA256 = "43b16b21d866d68142f380626832266a5ba43f3195bdcdf5bb119f2fdcfedcf1"
RUNNER_SHA256 = "4dd0497701131d450aae57614e3b8a33ae75ff080e04fcf5037f3728b827ccc9"
KERNEL = "pasha883/vsos-candidate-graph-oracle-v3-phase-a-t4x2"
KERNEL_ID = 126846203
RECOVERY_HASHES = {
    "intent_sha256": "610d2085d7aae2edc3d5680f92a9185301b0f0b7ae6cecdf35fb05f320ca15a6",
    "raw_response_sha256": "78846f0df32df680b18e3e9e2299da8ba6d209f854ad7afc492d92fa5208b2b2",
    "normalization_receipt_sha256": "b3369f0e3f5b6d68fdc14f1ffb1d15199c895f0644bd13b8fa3e693f40249ce7",
    "derived_response_sha256": "0ed668dec3f5a67e74612a11a3b3c0e90f3b7fd52547a7e75bb9866e7ef1afd6",
    "recovery_parser_sha256": "e137c2533e706a3d6a67febcb3b9854a85d1be64520dec652e7e68939bdfedbc",
}
DATASETS = {
    "code": "pasha883/vsos-candidate-graph-oracle-v3-code",
    "input": "pasha883/vsos-candidate-graph-oracle-v3-inputs",
    "runtime": "pasha883/vsos-candidate-graph-oracle-v3-runtime",
}
SUPPLEMENT_SHA256S = {
    "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt": (
        "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"
    ),
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt": (
        "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787"
    ),
    "configs/denoise_splits_seed20260710.json": (
        "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345"
    ),
    "configs/denoise_validation_quarantine_v1.json": (
        "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215"
    ),
    "configs/assembly_audit_exclusion_v1.json": (
        "772e89ad4f633d2050f8ad3806cd24bffed132bcd8914951b7b8edff3f608ab6"
    ),
    "configs/qap_weight_confirmation_v1.json": (
        "30732463fb200bdff8f909ef06be6cb6c4e7859692e01c9d33c5d55175ffe262"
    ),
    "runs/assembly_v1/kaggle/qap_weight_confirmation_output/v3_raw/"
    "qap_weight_confirmation_report.json": (
        "229f3751b85f26f9066c7fae0ed055f5a308354db1a408f0bcee88e8fa5189e7"
    ),
}
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_object(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_file(payload: Mapping[str, Any]) -> bytes:
    return _canonical_object(payload) + b"\n"


def _read_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")
        _require(info.st_nlink == 1, f"file must have exactly one link: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _sha(path: Path) -> str:
    return hashlib.sha256(_read_regular(path)).hexdigest()


def _load_canonical(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = _read_regular(path)
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, f"SHA drift: {path}")
    payload = json.loads(raw.decode("utf-8"))
    _require(isinstance(payload, dict), f"expected JSON object: {path}")
    _require(raw == _canonical_file(payload), f"noncanonical JSON: {path}")
    return payload


def _absolute(path: Path, *, label: str) -> Path:
    _require(path.is_absolute(), f"{label} must be an absolute path")
    return path.resolve(strict=True)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    _require(path.is_absolute(), "output must be an absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_file(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            _require(written > 0, "short composite-receipt write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_archive_members(archive: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            relative = PurePosixPath(info.filename)
            _require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and info.filename == relative.as_posix(),
                f"unsafe snapshot archive member: {info.filename}",
            )
            _require(info.filename not in members, "duplicate snapshot archive member")
            file_type = (info.external_attr >> 16) & 0o170000
            _require(
                file_type in (0, stat.S_IFREG),
                f"non-regular snapshot archive member: {info.filename}",
            )
            members[info.filename] = hashlib.sha256(bundle.read(info)).hexdigest()
    _require(len(members) == 38, "snapshot archive member-count drift")
    return members


def verify_snapshot_exact(snapshot_root: Path, archive: Path) -> dict[str, Any]:
    _require(_sha(archive) == SNAPSHOT_ARCHIVE_SHA256, "snapshot archive SHA drift")
    archive_members = _snapshot_archive_members(archive)
    _require(
        not set(archive_members).intersection(SUPPLEMENT_SHA256S),
        "snapshot archive/supplement path overlap",
    )
    expected = {**archive_members, **SUPPLEMENT_SHA256S}
    observed_paths: set[str] = set()
    observed_hashes: dict[str, str] = {}
    for candidate in snapshot_root.rglob("*"):
        if candidate.is_dir():
            _require(not candidate.is_symlink(), f"snapshot directory symlink: {candidate}")
            continue
        relative = candidate.relative_to(snapshot_root).as_posix()
        observed_paths.add(relative)
        observed_hashes[relative] = _sha(candidate)
    _require(observed_paths == set(expected), "snapshot extracted-tree path closure drift")
    _require(observed_hashes == expected, "snapshot extracted-tree byte drift")
    tree_sha = hashlib.sha256(_canonical_object(observed_hashes)).hexdigest()
    return {
        "archive_sha256": SNAPSHOT_ARCHIVE_SHA256,
        "archive_member_count": len(archive_members),
        "supplement_count": len(SUPPLEMENT_SHA256S),
        "extracted_file_count": len(expected),
        "supplement_sha256s": dict(sorted(SUPPLEMENT_SHA256S.items())),
        "extracted_tree_sha256": tree_sha,
        "frozen_verifier_sha256": observed_hashes[
            "scripts/verify_candidate_graph_oracle_result.py"
        ],
    }


def _load_frozen_verifier(snapshot_root: Path) -> Any:
    path = snapshot_root / "scripts/verify_candidate_graph_oracle_result.py"
    _require(_sha(path) == FROZEN_VERIFIER_SHA256, "frozen verifier SHA drift")
    module_name = "_candidate_graph_oracle_v3_frozen_verifier_f0df"
    spec = importlib.util.spec_from_file_location(module_name, path)
    _require(spec is not None and spec.loader is not None, "cannot load frozen verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _verify_wrapper(
    verifier: Any,
    *,
    context: Any,
    phase_a: Any,
    input_evidence: Any,
    wrapper_path: Path,
) -> dict[str, Any]:
    raw, _ = verifier._secure_absolute_file(wrapper_path)
    _require(verifier._sha256_bytes(raw) == WRAPPER_SHA256, "wrapper SHA drift")
    wrapper = verifier._require_object(
        verifier._parse_json(
            raw, label="Phase-A Kaggle wrapper", canonical_file=True
        ),
        label="Phase-A Kaggle wrapper",
    )
    keys = {
        "schema_version",
        "kind",
        "status",
        "safe_for_submission",
        "kernel_slug",
        "config_sha256",
        "runner_sha256",
        "kernel_metadata_sha256",
        "launch_expectation",
        "evaluator_sha256",
        "tests_sha256",
        "fixture_builder_tests_sha256",
        "environment_lock_sha256",
        "input_manifest_sha256",
        "runtime_assets",
        "dataset_mounts",
        "exact_code_mount_sha256",
        "hardware",
        "shards",
        "finalized_phase_a_manifest",
        "finalized_phase_a_manifest_sha256",
        "seconds",
    }
    verifier._require_exact_keys(wrapper, keys, label="Phase-A Kaggle wrapper")
    metadata = verifier._pinned_kernel_metadata(context)
    launch = metadata["oracle_launch_expectation"]
    pins = context.config["runtime_pins"]
    expected = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_phase_a_kaggle_wrapper",
        "status": "phase_a_complete_pending_local_verification",
        "safe_for_submission": False,
        "kernel_slug": launch["kernel_slug"],
        "config_sha256": CONFIG_SHA256,
        "runner_sha256": pins["phase_a_runner_sha256"],
        "kernel_metadata_sha256": pins["phase_a_kernel_metadata_sha256"],
        "launch_expectation": launch,
        "evaluator_sha256": pins["evaluator_sha256"],
        "tests_sha256": pins["tests_sha256"],
        "fixture_builder_tests_sha256": pins["fixture_builder_tests_sha256"],
        "environment_lock_sha256": pins["environment_lock_sha256"],
        "input_manifest_sha256": input_evidence.manifest_sha256,
        "runtime_assets": {
            "denoiser_sha256": context.config["frozen_contract"]["assets"]["denoiser"]["sha256"],
            "hbt_sha256": context.config["frozen_contract"]["assets"]["hbt"]["sha256"],
        },
        "exact_code_mount_sha256": verifier._expected_phase_a_code_mount(context),
        "finalized_phase_a_manifest": "finalized/FROZEN_CANDIDATE_GRAPH_MANIFEST.json",
        "finalized_phase_a_manifest_sha256": phase_a.envelope_sha256,
    }
    for key, value in expected.items():
        _require(wrapper.get(key) == value, f"wrapper crosslink drift: {key}")
    _require(
        wrapper["shards"]
        == [
            {"rank": index, "manifest_sha256": digest}
            for index, digest in enumerate(SHARD_SHA256S)
        ],
        "wrapper shard anchors drift",
    )
    mounts = verifier._require_object(wrapper["dataset_mounts"], label="dataset_mounts")
    verifier._require_exact_keys(mounts, set(DATASETS), label="dataset_mounts")
    for label, slug in DATASETS.items():
        mount = verifier._require_object(mounts[label], label=f"dataset_mounts.{label}")
        verifier._require_exact_keys(
            mount, {"slug", "version", "path"}, label=f"dataset_mounts.{label}"
        )
        _require(
            mount["slug"] == slug
            and mount["version"] == 2
            and PurePosixPath(mount["path"])
            == PurePosixPath("/kaggle/input") / slug.split("/", 1)[1],
            f"wrapper dataset mount drift: {label}",
        )
    verifier._verify_phase_a_hardware(context, wrapper["hardware"])
    _require(
        verifier._require_finite_float(wrapper["seconds"], label="wrapper.seconds")
        >= 0.0,
        "wrapper duration is negative",
    )
    return {
        "sha256": WRAPPER_SHA256,
        "kernel_slug": wrapper["kernel_slug"],
        "runner_sha256": wrapper["runner_sha256"],
        "hardware_device_count": len(wrapper["hardware"]["devices"]),
        "seconds": wrapper["seconds"],
    }


def validate_recovered_verification(payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "verified_utc",
        "status",
        "protocol_instance_id",
        "kernel",
        "datasets",
        *RECOVERY_HASHES.keys(),
        "launch_receipt_sha256",
        "raw_ref_before",
        "canonical_ref_after",
        "remote_write_performed_by_recovery",
        "kernel_version_advance_from_reservation",
        "label_access_claim_present",
        "label_fixture_opened_by_verifier",
        "safe_for_submission",
    }
    _require(set(payload) == expected_keys, "recovered verification schema drift")
    _require(
        payload["schema_version"] == 1
        and payload["kind"] == "candidate_graph_oracle_v3_recovered_launch_verification"
        and payload["status"] == "verified_complete"
        and payload["protocol_instance_id"] == INSTANCE,
        "recovered verification header drift",
    )
    _require(
        isinstance(payload["verified_utc"], str)
        and UTC_RE.fullmatch(payload["verified_utc"]) is not None,
        "recovered verification UTC drift",
    )
    _require(
        payload["kernel"]
        == {
            "slug": KERNEL,
            "id": KERNEL_ID,
            "version": 2,
            "source_sha256": RUNNER_SHA256,
            "session_status": "KernelWorkerStatus.COMPLETE",
        },
        "recovered verification kernel drift",
    )
    _require(
        payload["datasets"]
        == {
            label: {"status": "ready", "current_version_number": 2}
            for label in DATASETS
        },
        "recovered verification dataset drift",
    )
    for key, digest in RECOVERY_HASHES.items():
        _require(payload[key] == digest, f"recovered verification hash drift: {key}")
    _require(
        payload["launch_receipt_sha256"] == LAUNCH_RECEIPT_SHA256
        and payload["raw_ref_before"] == f"/code/{KERNEL}"
        and payload["canonical_ref_after"] == KERNEL
        and payload["remote_write_performed_by_recovery"] is False
        and payload["kernel_version_advance_from_reservation"] == 1
        and payload["label_access_claim_present"] is False
        and payload["label_fixture_opened_by_verifier"] is False
        and payload["safe_for_submission"] is False,
        "recovered verification provenance/safety drift",
    )


def _verify_recovered_evidence(
    *,
    recovered_verification_path: Path,
    recovered_verifier_path: Path,
    launch_receipt_path: Path,
) -> dict[str, Any]:
    _require(
        _sha(recovered_verifier_path) == RECOVERED_VERIFIER_SHA256,
        "recovered-launch verifier SHA drift",
    )
    payload = _load_canonical(recovered_verification_path, RECOVERED_COMPLETE_SHA256)
    validate_recovered_verification(payload)
    envelope = _load_canonical(launch_receipt_path, LAUNCH_RECEIPT_SHA256)
    _require(set(envelope) == {"payload", "payload_sha256"}, "launch receipt envelope drift")
    receipt = envelope["payload"]
    _require(isinstance(receipt, dict), "launch receipt payload missing")
    _require(
        envelope["payload_sha256"] == hashlib.sha256(_canonical_object(receipt)).hexdigest(),
        "launch receipt payload self-hash drift",
    )
    _require(
        receipt.get("schema_version") == 3
        and receipt.get("kind")
        == "candidate_graph_oracle_kaggle_launch_receipt_derived_from_raw"
        and receipt.get("protocol_instance_id") == INSTANCE
        and receipt.get("response_provenance")
        == "derived_from_immutable_raw_sdk_response"
        and receipt.get("remote_write_performed_by_recovery") is False
        and receipt.get("safe_for_submission") is False,
        "launch receipt schema-v3 provenance drift",
    )
    return {
        "verifier_sha256": RECOVERED_VERIFIER_SHA256,
        "verification_sha256": RECOVERED_COMPLETE_SHA256,
        "launch_receipt_sha256": LAUNCH_RECEIPT_SHA256,
        "status": payload["status"],
        "raw_ref_before": payload["raw_ref_before"],
        "canonical_ref_after": payload["canonical_ref_after"],
        "remote_write_performed_by_recovery": False,
    }


def _ledger_is_phase_a_only(ledger_root: Path) -> dict[str, Any]:
    top_files = {item.name for item in ledger_root.iterdir() if item.is_file()}
    _require("LABEL_ACCESS.json" not in top_files, "LABEL_ACCESS exists")
    _require(top_files == {"PREP.json", "SEALED.json", "PHASE_A.json"}, "ledger state drift")
    return {
        "terminal_state": "PHASE_A",
        "label_access_present": False,
        "phase_a_lifecycle_sha256": _sha(ledger_root / "PHASE_A.json"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: _absolute(getattr(args, name), label=name)
        for name in (
            "snapshot_root",
            "snapshot_archive",
            "fixture_input_root",
            "phase_a_root",
            "wrapper",
            "launch_receipt",
            "recovered_verification",
            "recovered_verifier",
            "ledger_root",
        )
    }
    _require(args.output.is_absolute(), "output must be an absolute path")
    _require(not args.output.exists(), "output already exists")
    forbidden = ("fixture_label", "master_secret", "/labels/", "/targets/")
    for name in ("fixture_input_root", "phase_a_root", "wrapper"):
        lowered = paths[name].as_posix().lower()
        _require(not any(token in lowered for token in forbidden), f"forbidden path: {name}")

    snapshot_before = verify_snapshot_exact(paths["snapshot_root"], paths["snapshot_archive"])
    verifier = _load_frozen_verifier(paths["snapshot_root"])
    config_path = paths["snapshot_root"] / "configs/candidate_graph_oracle_ceiling_v3.json"
    context = verifier._load_protocol(
        config_path,
        expected_config_sha256=CONFIG_SHA256,
        allow_unpinned_verifier=False,
    )
    _require(
        context.config["frozen_contract_sha256"] == FROZEN_CONTRACT_SHA256,
        "frozen contract drift after production load",
    )
    input_evidence = verifier.verify_input_fixture(
        context,
        fixture_root=paths["fixture_input_root"],
        expected_manifest_sha256=INPUT_MANIFEST_SHA256,
    )
    phase_a = verifier.verify_phase_a(
        context,
        phase_a_root=paths["phase_a_root"],
        expected_envelope_sha256=PHASE_A_MANIFEST_SHA256,
        shard_anchors=SHARD_SHA256S,
        input_evidence=input_evidence,
    )
    _require(phase_a.kaggle_attestation is None, "normal-schema attestation was unexpectedly set")
    wrapper = _verify_wrapper(
        verifier,
        context=context,
        phase_a=phase_a,
        input_evidence=input_evidence,
        wrapper_path=paths["wrapper"],
    )
    recovered = _verify_recovered_evidence(
        recovered_verification_path=paths["recovered_verification"],
        recovered_verifier_path=paths["recovered_verifier"],
        launch_receipt_path=paths["launch_receipt"],
    )
    ledger = _ledger_is_phase_a_only(paths["ledger_root"])
    verifier._post_phase_a_rehash(
        context,
        input_evidence=input_evidence,
        phase_a=phase_a,
        allow_unpinned_verifier=False,
    )
    _require(_sha(paths["wrapper"]) == WRAPPER_SHA256, "wrapper changed during verification")
    _require(
        _sha(paths["launch_receipt"]) == LAUNCH_RECEIPT_SHA256,
        "launch receipt changed during verification",
    )
    _require(
        _sha(paths["recovered_verification"]) == RECOVERED_COMPLETE_SHA256,
        "recovered verification changed during composite verification",
    )
    snapshot_after = verify_snapshot_exact(paths["snapshot_root"], paths["snapshot_archive"])
    _require(snapshot_after == snapshot_before, "snapshot changed during composite verification")

    render_count = sum(len(record.manifest["renders"]) for record in phase_a.records.values())
    _require(len(phase_a.records) == 64 and render_count == 192, "Phase-A count drift")
    payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v3_phase_a_composite_verification",
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": "verified_input_only_composite",
        "protocol_instance_id": INSTANCE,
        "config_sha256": CONFIG_SHA256,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "snapshot": snapshot_after,
        "frozen_phase_a_verification": {
            "production_static_binding": True,
            "allow_unpinned_verifier": False,
            "normal_schema_launch_attestation_branch_executed": False,
            "normal_schema_launch_attestation_omission_reason": (
                "actual_sdk_ref_used_exact_/code/_alias_and_is_bound_by_separate_schema_v3_recovery"
            ),
            "phase_a_manifest_sha256": phase_a.envelope_sha256,
            "phase_a_shard_sha256s": list(phase_a.shard_anchors),
            "input_manifest_sha256": input_evidence.manifest_sha256,
            "record_count": len(phase_a.records),
            "graph_artifact_count": len(phase_a.records),
            "render_count": render_count,
            "every_array_descriptor_verified": True,
            "every_candidate_union_reconstructed": True,
            "every_render_pixel_reconstructed": True,
            "post_verification_rehash_passed": True,
        },
        "wrapper": wrapper,
        "recovered_launch_verification": recovered,
        "ledger": ledger,
        "label_paths_constructed": False,
        "label_files_opened": False,
        "label_access_performed": False,
        "safe_for_submission": False,
    }
    envelope = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical_object(payload)).hexdigest(),
    }
    output_sha = _write_exclusive(args.output, envelope)
    return {
        "status": payload["status"],
        "output": str(args.output),
        "output_sha256": output_sha,
        "payload_sha256": envelope["payload_sha256"],
        "records": 64,
        "graphs": 64,
        "renders": 192,
        "labels_constructed_or_opened": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--snapshot-archive", type=Path, required=True)
    parser.add_argument("--fixture-input-root", type=Path, required=True)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--recovered-verification", type=Path, required=True)
    parser.add_argument("--recovered-verifier", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
