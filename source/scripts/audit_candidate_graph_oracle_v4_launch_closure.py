#!/usr/bin/env python3
"""Label-blind, pre-reservation-aware audit of the v4 launch closure.

The utility deliberately treats a reserved Kaggle identity, populated runtime
pins, and complete local evidence as separate states.  A valid draft is useful
evidence, but it is never reported as launchable while ``id_no == -1`` or any
runtime SHA pin is null.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_RELATIVE = "configs/candidate_graph_oracle_ceiling_v4.json"
INSTANCE = "6c0fe4e8524ce39d830d9a5bee118d8b"
FROZEN_CONTRACT_SHA256 = (
    "2070c1b4ff0a3ff42c5ffdd6d611c214c02dbb99b2c51a88f38a862bb1f8a05c"
)
KERNEL_SLUG = "pasha883/vsos-candidate-graph-oracle-v4-phase-a-t4x2"
KERNEL_VERSION = 2
KERNEL_METADATA_RELATIVE = (
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_job/"
    "kernel-metadata.json"
)
RESERVATION_RECEIPT_RELATIVE = (
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_reservations/"
    "RESERVATION_RECEIPT.json"
)
RESERVATION_RUNNER_RELATIVE = (
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_reservations/kernel/"
    "reservation_runner.py"
)
RESERVATION_RUNNER_SHA256 = (
    "adf4d61a528f91ce5a4c282b0f3999f8bcdbe8c18d5429a98210dc6b991ab460"
)
RESERVATION_ORCHESTRATOR_SHA256 = (
    "e546b84010de31021f9ab15b67de4c945cf7e26d52cab66996f4dea4bcd09ef6"
)
BUNDLE_RECEIPT_RELATIVE = (
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_bundles/"
    "CANDIDATE_GRAPH_ORACLE_KAGGLE_BUNDLE_BUILD_RECEIPT.json"
)
FIXTURE_ROOT_RELATIVE = (
    "runs/assembly_v1/"
    "candidate_graph_oracle_fixtures_v4_6c0fe4e8524ce39d830d9a5bee118d8b"
)
DATASETS = {
    "code": "pasha883/vsos-candidate-graph-oracle-v4-code",
    "input": "pasha883/vsos-candidate-graph-oracle-v4-inputs",
    "runtime": "pasha883/vsos-candidate-graph-oracle-v4-runtime",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INSTANCE_RE = re.compile(r"^[0-9a-f]{32}$")

CODE_PIN_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("evaluator_path", "evaluator_sha256", "scripts/evaluate_candidate_graph_oracle_v4.py"),
    ("tests_path", "tests_sha256", "tests/test_candidate_graph_oracle_v4.py"),
    (
        "fixture_builder_path",
        "fixture_builder_sha256",
        "scripts/build_candidate_graph_oracle_v4_fixtures.py",
    ),
    (
        "fixture_builder_tests_path",
        "fixture_builder_tests_sha256",
        "tests/test_build_candidate_graph_oracle_v4_fixtures.py",
    ),
    (
        "pin_finalizer_path",
        "pin_finalizer_sha256",
        "scripts/finalize_candidate_graph_oracle_v4_protocol.py",
    ),
    (
        "lifecycle_tool_path",
        "lifecycle_tool_sha256",
        "scripts/update_candidate_graph_oracle_v4_ledger.py",
    ),
    (
        "result_verifier_path",
        "result_verifier_sha256",
        "scripts/verify_candidate_graph_oracle_v4_result.py",
    ),
    (
        "environment_lock_path",
        "environment_lock_sha256",
        "configs/candidate_graph_oracle_environment_lock_v1.json",
    ),
    (
        "phase_a_runner_path",
        "phase_a_runner_sha256",
        "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_job/run_phase_a.py",
    ),
    (
        "phase_a_kernel_metadata_path",
        "phase_a_kernel_metadata_sha256",
        KERNEL_METADATA_RELATIVE,
    ),
    (
        "phase_a_launcher_path",
        "phase_a_launcher_sha256",
        "scripts/push_candidate_graph_oracle_v4_phase_a.py",
    ),
    (
        "phase_b_runner_path",
        "phase_b_runner_sha256",
        "scripts/run_candidate_graph_oracle_v4_phase_b.py",
    ),
)
FIXTURE_PIN_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "fixture_input_manifest_relative_path",
        "fixture_input_manifest_sha256",
        "fixture_input/fixture_input_manifest.json",
    ),
    (
        "fixture_label_manifest_relative_path",
        "fixture_label_manifest_sha256",
        "fixture_label/fixture_label_manifest.json",
    ),
    (
        "fixture_lock_relative_path",
        "fixture_lock_sha256",
        "fixture_control/fixture_lock.json",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        _require(
            stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            f"not a one-link regular file: {path}",
        )
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
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid UTF-8 JSON: {path}") from error
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value, raw, hashlib.sha256(raw).hexdigest()


def _canonical_object(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_file(value: Mapping[str, Any]) -> bytes:
    return _canonical_object(value) + b"\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(_read_regular(path)).hexdigest()


def _relative(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} must be a relative path")
    _require("\x00" not in value and "\\" not in value, f"{label} has a forbidden separator")
    pure = PurePosixPath(value)
    _require(
        not pure.is_absolute()
        and pure.parts
        and all(part not in ("", ".", "..") for part in pure.parts)
        and pure.as_posix() == value,
        f"{label} is not a canonical relative POSIX path",
    )
    return value


def _classify(values: Sequence[Any], *, label: str) -> str:
    nulls = [value is None for value in values]
    if all(nulls):
        return "null"
    _require(not any(nulls), f"partial {label} pin transition is forbidden")
    _require(
        all(isinstance(value, str) and SHA_RE.fullmatch(value) for value in values),
        f"malformed {label} SHA pin",
    )
    return "pinned"


def _validate_config(
    repo_root: Path, config_path: Path
) -> tuple[dict[str, Any], str, str, str, dict[str, str]]:
    config, raw, config_sha = _load(config_path)
    _require(
        raw == (json.dumps(config, ensure_ascii=True, indent=2) + "\n").encode("utf-8"),
        "config is not canonical repository JSON",
    )
    _require(
        config.get("schema_version") == 1
        and config.get("kind") == "candidate_graph_oracle_ceiling",
        "wrong v4 protocol schema",
    )
    _require(config.get("safe_for_submission") is False, "protocol became submission-safe")
    instance = config.get("protocol_instance_id")
    _require(
        instance == INSTANCE and INSTANCE_RE.fullmatch(instance) is not None,
        "v4 protocol instance drift",
    )
    frozen = config.get("frozen_contract")
    _require(isinstance(frozen, dict), "missing frozen contract")
    frozen_sha = hashlib.sha256(_canonical_object(frozen)).hexdigest()
    _require(
        config.get("frozen_contract_sha256")
        == frozen_sha
        == FROZEN_CONTRACT_SHA256,
        "declared frozen contract SHA does not match canonical frozen contract",
    )
    _require(
        frozen.get("protocol_instance", {}).get("exact_value") == INSTANCE,
        "frozen protocol instance drift",
    )
    pins = config.get("runtime_pins")
    policy = config.get("runtime_pin_mutation_policy")
    _require(isinstance(pins, dict) and isinstance(policy, dict), "runtime pin closure missing")
    _require(
        policy.get("transition_ledger_root")
        == f"runs/assembly_v1/protocol_ledgers/candidate_graph_oracle/{INSTANCE}",
        "transition ledger root drift",
    )
    expected_code_policy = [
        {"path_field": path_field, "sha256_field": sha_field}
        for path_field, sha_field, _ in CODE_PIN_PAIRS
    ]
    expected_fixture_policy = [
        {"path_field": path_field, "sha256_field": sha_field}
        for path_field, sha_field, _ in FIXTURE_PIN_PAIRS
    ]
    _require(policy.get("code_pin_fields") == expected_code_policy, "code pin policy drift")
    _require(
        policy.get("fixture_pin_fields") == expected_fixture_policy,
        "fixture pin policy drift",
    )
    for path_field, _, expected_path in (*CODE_PIN_PAIRS, *FIXTURE_PIN_PAIRS):
        _require(pins.get(path_field) == expected_path, f"runtime path drift: {path_field}")
        _relative(expected_path, label=path_field)

    code_state = _classify(
        [pins.get(sha_field) for _, sha_field, _ in CODE_PIN_PAIRS], label="code"
    )
    fixture_state = _classify(
        [pins.get(sha_field) for _, sha_field, _ in FIXTURE_PIN_PAIRS], label="fixture"
    )
    _require(
        not (code_state == "null" and fixture_state == "pinned"),
        "fixture pins precede code pins",
    )
    verified: dict[str, str] = {}
    if code_state == "pinned":
        for path_field, sha_field, _ in CODE_PIN_PAIRS:
            actual = _sha(repo_root / pins[path_field])
            _require(actual == pins[sha_field], f"runtime pin drift: {sha_field}")
            verified[sha_field] = actual
    return config, config_sha, frozen_sha, code_state, {
        "fixture_state": fixture_state,
        **verified,
    }


def _validate_kernel_metadata(
    repo_root: Path, config: Mapping[str, Any], code_state: str
) -> tuple[dict[str, Any], int, str]:
    pins = config["runtime_pins"]
    metadata_path = repo_root / pins["phase_a_kernel_metadata_path"]
    metadata, _, metadata_sha = _load(metadata_path)
    if code_state == "pinned":
        _require(
            metadata_sha == pins["phase_a_kernel_metadata_sha256"],
            "kernel metadata SHA pin drift",
        )
    kernel_id = metadata.get("id_no")
    _require(
        isinstance(kernel_id, int)
        and not isinstance(kernel_id, bool)
        and (kernel_id == -1 or kernel_id > 0),
        "kernel id_no must be exactly -1 before reservation or a positive integer",
    )
    _require(
        metadata.get("id") == KERNEL_SLUG
        and metadata.get("is_private") is True
        and metadata.get("enable_gpu") is True
        and metadata.get("enable_internet") is False
        and metadata.get("machine_shape") == "NvidiaTeslaT4",
        "kernel identity/privacy/hardware drift",
    )
    expected_sources = [f"{slug}/{KERNEL_VERSION}" for slug in DATASETS.values()]
    _require(metadata.get("dataset_sources") == expected_sources, "kernel dataset source drift")
    expectation = metadata.get("oracle_launch_expectation")
    _require(isinstance(expectation, dict), "kernel launch expectation missing")
    _require(
        expectation.get("kernel_id") == kernel_id
        and expectation.get("kernel_slug") == KERNEL_SLUG
        and expectation.get("kernel_version") == KERNEL_VERSION,
        "kernel launch identity drift",
    )
    reservation_sha = metadata.get("reservation_receipt_sha256")
    expectation_reservation_sha = expectation.get("reservation_receipt_sha256")
    if kernel_id == -1:
        _require(
            reservation_sha is None and expectation_reservation_sha is None,
            "pre-reservation metadata contains a reservation receipt binding",
        )
    else:
        _require(
            isinstance(reservation_sha, str)
            and SHA_RE.fullmatch(reservation_sha) is not None
            and expectation_reservation_sha == reservation_sha,
            "reserved kernel metadata lacks its receipt binding",
        )
    _require(
        expectation.get("dataset_versions")
        == {
            label: {"slug": slug, "version": KERNEL_VERSION}
            for label, slug in DATASETS.items()
        },
        "kernel launch dataset versions drift",
    )
    return metadata, kernel_id, metadata_sha


def _literal_assignments(path: Path) -> dict[str, Any]:
    raw = _read_regular(path)
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise RuntimeError(f"invalid Python source: {path}") from error
    values: dict[str, Any] = {}
    for node in tree.body:
        name: str | None = None
        expression: ast.expr | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            expression = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            expression = node.value
        if name is None or expression is None:
            continue
        try:
            values[name] = ast.literal_eval(expression)
        except (TypeError, ValueError):
            continue
    return values


def _validate_source_reservation_bindings(
    repo_root: Path, config: Mapping[str, Any], kernel_id: int, receipt_sha: str | None
) -> None:
    pins = config["runtime_pins"]
    runner = _literal_assignments(repo_root / pins["phase_a_runner_path"])
    launcher = _literal_assignments(repo_root / pins["phase_a_launcher_path"])
    _require(runner.get("KERNEL_ID") == kernel_id, "Phase-A runner kernel id drift")
    _require(
        launcher.get("EXPECTED_KERNEL_ID") == kernel_id,
        "Phase-A launcher kernel id drift",
    )
    _require(
        runner.get("RESERVATION_RECEIPT_SHA256") == receipt_sha
        and launcher.get("RESERVATION_RECEIPT_SHA256") == receipt_sha,
        "runner/launcher reservation receipt binding drift",
    )


def _validate_reservation(
    repo_root: Path, kernel_id: int, metadata: Mapping[str, Any]
) -> str | None:
    path = repo_root / RESERVATION_RECEIPT_RELATIVE
    if kernel_id == -1:
        _require(not path.exists(), "reservation receipt exists while kernel id is -1")
        return None
    envelope, raw, digest = _load(path)
    _require(raw == _canonical_file(envelope), "reservation receipt is noncanonical")
    _require(
        set(envelope) == {"payload", "payload_sha256"}
        and isinstance(envelope.get("payload"), dict),
        "reservation receipt envelope drift",
    )
    reservation = envelope["payload"]
    _require(
        set(reservation)
        == {
            "schema_version",
            "kind",
            "created_utc",
            "protocol_instance_id",
            "reservation_orchestrator_sha256",
            "local_validation",
            "datasets",
            "kernel",
            "contains_fixture_pixels",
            "gpu_requested",
            "dataset_v2_uploaded",
            "phase_a_push_performed",
            "safe_for_submission",
        },
        "reservation receipt top-level schema drift",
    )
    _require(
        envelope.get("payload_sha256")
        == hashlib.sha256(_canonical_object(reservation)).hexdigest(),
        "reservation receipt self-hash drift",
    )
    _require(
        metadata.get("reservation_receipt_sha256") == digest
        and metadata.get("oracle_launch_expectation", {}).get(
            "reservation_receipt_sha256"
        )
        == digest,
        "reservation receipt is not bound by kernel metadata",
    )
    local_validation = reservation.get("local_validation")
    _require(
        reservation.get("schema_version") == 1
        and reservation.get("kind")
        == "candidate_graph_oracle_v4_kaggle_reservation_receipt"
        and reservation.get("protocol_instance_id") == INSTANCE
        and reservation.get("contains_fixture_pixels") is False
        and reservation.get("gpu_requested") is False
        and reservation.get("dataset_v2_uploaded") is False
        and reservation.get("phase_a_push_performed") is False
        and reservation.get("safe_for_submission") is False,
        "reservation protocol instance drift",
    )
    _require(
        reservation.get("reservation_orchestrator_sha256")
        == RESERVATION_ORCHESTRATOR_SHA256
        and isinstance(local_validation, dict)
        and local_validation.get("schema_version") == 1
        and local_validation.get("kind")
        == "candidate_graph_oracle_v4_local_reservation_validation"
        and local_validation.get("protocol_instance_id") == INSTANCE
        and local_validation.get("reservation_orchestrator_sha256")
        == RESERVATION_ORCHESTRATOR_SHA256
        and local_validation.get("contains_fixture_pixels") is False
        and local_validation.get("gpu_requested") is False
        and local_validation.get("safe_for_submission") is False,
        "reservation orchestrator/local-validation binding drift",
    )
    kernel = reservation.get("kernel")
    _require(
        isinstance(kernel, dict)
        and kernel.get("kernel_id") == kernel_id
        and kernel.get("slug") == KERNEL_SLUG
        and kernel.get("reserved_version") == 1,
        "reservation kernel binding drift",
    )
    _require(
        kernel.get("is_private") is True
        and kernel.get("enable_gpu") is False
        and kernel.get("enable_tpu") is False
        and kernel.get("enable_internet") is False
        and kernel.get("dataset_sources") == []
        and kernel.get("kernel_sources") == []
        and kernel.get("competition_sources") == []
        and kernel.get("model_sources") == []
        and str(kernel.get("status", "")).lower() == "complete"
        and kernel.get("reservation_runner_sha256") == RESERVATION_RUNNER_SHA256,
        "reservation kernel isolation drift",
    )
    _require(
        _sha(repo_root / RESERVATION_RUNNER_RELATIVE) == RESERVATION_RUNNER_SHA256,
        "reservation runner source SHA drift",
    )
    datasets = reservation.get("datasets")
    _require(isinstance(datasets, dict), "reservation dataset closure missing")
    for label, slug in DATASETS.items():
        item = datasets.get(label)
        _require(
            isinstance(item, dict)
            and item.get("slug") == slug
            and item.get("reserved_version") == 1,
            f"reservation dataset drift: {label}",
        )
        _require(
            item.get("is_private") is True
            and str(item.get("status", "")).lower() == "ready",
            f"reservation dataset privacy/status drift: {label}",
        )
    return digest


def _validate_final_local_evidence(
    repo_root: Path,
    config: Mapping[str, Any],
    config_sha: str,
    frozen_sha: str,
    kernel_id: int,
) -> dict[str, Any]:
    pins = config["runtime_pins"]
    fixture_root = repo_root / FIXTURE_ROOT_RELATIVE
    input_manifest, _, input_sha = _load(
        fixture_root / pins["fixture_input_manifest_relative_path"]
    )
    lock, _, lock_sha = _load(fixture_root / pins["fixture_lock_relative_path"])
    _require(input_sha == pins["fixture_input_manifest_sha256"], "input fixture pin drift")
    _require(lock_sha == pins["fixture_lock_sha256"], "fixture lock pin drift")
    for label, value in (("input manifest", input_manifest), ("fixture lock", lock)):
        _require(value.get("protocol_instance_id") == INSTANCE, f"{label} instance drift")
        _require(value.get("frozen_contract_sha256") == frozen_sha, f"{label} frozen drift")
        _require(value.get("record_count") == 64, f"{label} record count drift")
        for _, sha_field, _ in CODE_PIN_PAIRS:
            _require(value.get(sha_field) == pins[sha_field], f"{label} code binding drift")
    _require(
        lock.get("fixture_input_manifest_sha256") == input_sha
        and lock.get("fixture_label_manifest_sha256")
        == pins["fixture_label_manifest_sha256"],
        "fixture lock crosslink drift",
    )
    _require(lock.get("phase_a_may_receive_label_root") is False, "Phase A may receive label root")
    _require(lock.get("phase_a_may_receive_master_secret") is False, "Phase A may receive secret")

    ledger_root = repo_root / config["runtime_pin_mutation_policy"]["transition_ledger_root"]
    _require(ledger_root.is_dir(), "lifecycle ledger root missing")
    names = {item.name for item in ledger_root.iterdir()}
    _require("LABEL_ACCESS.json" not in names, "LABEL_ACCESS already exists")
    _require(
        {"PREP.json", "SEALED.json", "PHASE_A.json"}.issubset(names),
        "lifecycle ledger is incomplete",
    )
    lifecycle: dict[str, str] = {}
    predecessor: str | None = None
    for state in ("PREP", "SEALED", "PHASE_A"):
        value, raw, digest = _load(ledger_root / f"{state}.json")
        _require(raw == _canonical_file(value), f"{state} ledger marker is noncanonical")
        _require(
            value.get("protocol_instance_id") == INSTANCE
            and value.get("frozen_contract_sha256") == frozen_sha
            and value.get("state") == state,
            f"{state} ledger binding drift",
        )
        _require(value.get("predecessor_sha256") == predecessor, f"{state} predecessor drift")
        if state != "PREP":
            _require(value.get("config_sha256_or_null") == config_sha, f"{state} config drift")
        predecessor = digest
        lifecycle[state] = digest

    receipt_path = repo_root / BUNDLE_RECEIPT_RELATIVE
    envelope, raw, receipt_sha = _load(receipt_path)
    _require(raw == _canonical_file(envelope), "bundle receipt is noncanonical")
    _require(set(envelope) == {"payload", "payload_sha256"}, "bundle receipt envelope drift")
    bundle = envelope.get("payload")
    _require(isinstance(bundle, dict), "bundle payload missing")
    _require(
        envelope.get("payload_sha256")
        == hashlib.sha256(_canonical_object(bundle)).hexdigest(),
        "bundle receipt self-hash drift",
    )
    _require(
        bundle.get("protocol_instance_id") == INSTANCE
        and bundle.get("frozen_contract_sha256") == frozen_sha
        and bundle.get("fully_pinned_config_sha256") == config_sha
        and bundle.get("lifecycle_terminal_state") == "PHASE_A"
        and bundle.get("input_payload_decoded") is False,
        "bundle receipt protocol binding drift",
    )
    archives: dict[str, str] = {}
    for label, slug in DATASETS.items():
        descriptor = bundle.get("datasets", {}).get(label)
        _require(
            isinstance(descriptor, dict)
            and descriptor.get("slug") == slug
            and descriptor.get("expected_version") == KERNEL_VERSION
            and descriptor.get("must_remain_private") is True,
            f"bundle dataset drift: {label}",
        )
        archive = descriptor.get("archive")
        _require(isinstance(archive, dict), f"bundle archive missing: {label}")
        relative = _relative(archive.get("path"), label=f"{label} archive path")
        actual = _sha(receipt_path.parent / relative)
        _require(actual == archive.get("sha256"), f"bundle archive SHA drift: {label}")
        archives[label] = actual
    return {
        "lifecycle_sha256": lifecycle,
        "bundle_receipt_sha256": receipt_sha,
        "dataset_archive_sha256": archives,
        "kernel_id": kernel_id,
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    encoded = _canonical_file(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            _require(written > 0, "short audit write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return hashlib.sha256(encoded).hexdigest()


def audit(
    *,
    repo_root: Path = REPO_ROOT,
    config_path: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Audit local state without contacting Kaggle or opening label files."""

    repo_root = repo_root.expanduser().resolve(strict=True)
    config_path = config_path or (repo_root / CONFIG_RELATIVE)
    config_path = config_path.expanduser().resolve(strict=True)
    config, config_sha, frozen_sha, code_state, details = _validate_config(
        repo_root, config_path
    )
    fixture_state = details.pop("fixture_state")
    metadata, kernel_id, metadata_sha = _validate_kernel_metadata(
        repo_root, config, code_state
    )
    reservation_sha = _validate_reservation(repo_root, kernel_id, metadata)
    _validate_source_reservation_bindings(
        repo_root, config, kernel_id, reservation_sha
    )

    blockers: list[str] = []
    if kernel_id == -1:
        blockers.append("kernel_identity_not_reserved")
    if code_state == "null":
        blockers.append("code_pins_are_null")
    if fixture_state == "null":
        blockers.append("fixture_pins_are_null")
    evidence: dict[str, Any] = {}
    if not blockers:
        evidence = _validate_final_local_evidence(
            repo_root, config, config_sha, frozen_sha, kernel_id
        )

    if kernel_id == -1:
        stage = "pre_reservation_draft"
    elif code_state == "null":
        stage = "reserved_unpinned"
    elif fixture_state == "null":
        stage = "code_pinned_pre_fixture"
    else:
        stage = "fully_pinned_prelaunch"
    launch_ready = not blockers
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v4_launch_closure_audit",
        "status": "ready_for_single_phase_a_launch" if launch_ready else "not_launchable",
        "stage": stage,
        "launch_ready": launch_ready,
        "blockers": blockers,
        "protocol_instance_id": INSTANCE,
        "frozen_contract_sha256": frozen_sha,
        "config_sha256": config_sha,
        "code_pin_state": code_state,
        "fixture_pin_state": fixture_state,
        "kernel": {
            "slug": KERNEL_SLUG,
            "id": kernel_id,
            "intended_version": KERNEL_VERSION,
            "metadata_sha256": metadata_sha,
        },
        "reservation_receipt_sha256": reservation_sha,
        "verified_code_pin_sha256": details,
        **evidence,
        "remote_api_called": False,
        "label_paths_constructed": False,
        "label_manifest_opened": False,
        "label_records_opened": False,
        "safe_for_submission": False,
    }
    if output is not None:
        output = output.expanduser().absolute()
        payload["audit_path"] = str(output)
        payload["audit_sha256"] = _write_exclusive(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(repo_root=args.repo_root, config_path=args.config, output=args.output)
    print(json.dumps(result, sort_keys=True, indent=2))
    if not result["launch_ready"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
