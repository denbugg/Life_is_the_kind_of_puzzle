#!/usr/bin/env python3
"""Irreversibly pin candidate-graph oracle runtime artifacts.

The protocol has exactly two legal mutations: all code/runtime SHA-256 pins are
filled together, then all fixture-manifest/lock pins are filled together.  The
caller must supply the exact current whole-config SHA-256 out of band.  Every
transition is preceded by an append-only intent and followed by an append-only
completion record, so rerunning after a crash can finish (but never alter) the
same transition.

This module intentionally uses only the Python standard library and never opens
image records or target data.
"""

from __future__ import annotations

import argparse
import ast
import copy
import errno
import hashlib
import importlib.util
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/candidate_graph_oracle_ceiling_v4.json"

SCHEMA_VERSION = 1
PROTOCOL_KIND = "candidate_graph_oracle_ceiling"
EXPECTED_PROTOCOL_INSTANCE_ID = "6c0fe4e8524ce39d830d9a5bee118d8b"
EXPECTED_FROZEN_CONTRACT_SHA256 = (
    "2070c1b4ff0a3ff42c5ffdd6d611c214c02dbb99b2c51a88f38a862bb1f8a05c"
)
RESERVATION_RECEIPT_RELATIVE = (
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_reservations/"
    "RESERVATION_RECEIPT.json"
)
EXPECTED_RESERVATION_RUNNER_SHA256 = (
    "adf4d61a528f91ce5a4c282b0f3999f8bcdbe8c18d5429a98210dc6b991ab460"
)
EXPECTED_KERNEL_SLUG = "pasha883/vsos-candidate-graph-oracle-v4-phase-a-t4x2"
EXPECTED_DATASET_SLUGS = {
    "code": "pasha883/vsos-candidate-graph-oracle-v4-code",
    "input": "pasha883/vsos-candidate-graph-oracle-v4-inputs",
    "runtime": "pasha883/vsos-candidate-graph-oracle-v4-runtime",
}
RESERVATION_ORCHESTRATOR_RELATIVE = (
    "scripts/reserve_candidate_graph_oracle_v4_kaggle.py"
)
EXPECTED_RESERVATION_ORCHESTRATOR_SHA256 = (
    "e546b84010de31021f9ab15b67de4c945cf7e26d52cab66996f4dea4bcd09ef6"
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INSTANCE_RE = re.compile(r"^[0-9a-f]{32}$")

STAGES = ("code", "fixtures")
STAGE_INDEX = {"code": 0, "fixtures": 1}
INTENT_KIND = "candidate_graph_oracle_runtime_pin_transition_intent"
COMPLETION_KIND = "candidate_graph_oracle_runtime_pin_transition_completion"
TRANSITION_DIR_NAME = "runtime_pin_transitions"

EXPECTED_CODE_PIN_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "evaluator_path",
        "evaluator_sha256",
        "scripts/evaluate_candidate_graph_oracle_v4.py",
    ),
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
        "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_job/kernel-metadata.json",
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

EXPECTED_FIXTURE_PIN_PAIRS: tuple[tuple[str, str, str], ...] = (
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
EXPECTED_PREP_MARKER_PATH = "fixture_control/FIXTURE_PIXEL_ACCESS_STARTED.json"

EXPECTED_TOP_LEVEL_KEYS = (
    "schema_version",
    "kind",
    "status",
    "created_utc",
    "protocol_instance_id",
    "decision_basis",
    "frozen_contract",
    "frozen_contract_sha256",
    "runtime_pins",
    "runtime_pin_mutation_policy",
    "safe_for_submission",
)
EXPECTED_TOP_LEVEL_IMMUTABLE_FIELDS = (
    "schema_version",
    "kind",
    "status",
    "created_utc",
    "protocol_instance_id",
    "decision_basis",
    "frozen_contract",
    "frozen_contract_sha256",
    "runtime_pin_mutation_policy",
    "safe_for_submission",
)
EXPECTED_POLICY_KEYS = (
    "frozen_contract_is_immutable",
    "protocol_instance_id_is_immutable",
    "runtime_pin_paths_are_immutable",
    "top_level_immutable_fields",
    "runtime_pins_schema_and_key_order_are_immutable",
    "only_null_sha256_values_may_transition_once_to_lowercase_64_hex",
    "transition_ledger_root",
    "code_pin_fields",
    "fixture_pin_fields",
    "pre_fixture_transition",
    "post_fixture_transition",
    "partial_pin_transition_forbidden",
    "all_code_test_runner_kernel_metadata_and_environment_hashes_must_be_pinned_before_fixture_pixel_access",
    "fixture_manifest_and_lock_hashes_may_be_pinned_once_after_fixture_preparation_but_before_phase_a",
    "every_pin_transition_requires_recomputing_and_recording_the_whole_config_file_sha256",
    "no_pin_may_change_after_phase_a_starts",
    "evaluator_hardcode_policy",
    "fixture_binding_policy",
    "phase_a_final_config_policy",
)


def _runtime_pin_key_order() -> tuple[str, ...]:
    keys: list[str] = []
    for path_field, sha_field, _ in EXPECTED_CODE_PIN_PAIRS:
        keys.extend((path_field, sha_field))
    # The fixture config deliberately stores SHA before relative path.
    for path_field, sha_field, _ in EXPECTED_FIXTURE_PIN_PAIRS:
        keys.extend((sha_field, path_field))
    keys.extend(
        (
            "fixture_prep_marker_relative_path",
            "phase_a_must_not_start_until_all_non_path_values_are_non_null",
        )
    )
    return tuple(keys)


EXPECTED_RUNTIME_PIN_KEYS = _runtime_pin_key_order()
INTENT_KEYS = (
    "schema_version",
    "kind",
    "stage",
    "stage_index",
    "protocol_instance_id",
    "frozen_contract_sha256",
    "config_relative_path",
    "previous_config_sha256",
    "intended_config_sha256",
    "pin_sha256_values",
    "created_utc",
)
COMPLETION_KEYS = (
    "schema_version",
    "kind",
    "stage",
    "stage_index",
    "protocol_instance_id",
    "frozen_contract_sha256",
    "config_relative_path",
    "previous_config_sha256",
    "final_config_sha256",
    "pin_sha256_values",
    "intent_sha256",
    "completed_utc",
)


@dataclass(frozen=True)
class FileSnapshot:
    relative_path: str
    data: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    @property
    def identity(self) -> tuple[int, int]:
        return (self.device, self.inode)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _canonical_object_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_config_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonical repository JSON while preserving immutable object key order."""

    return (json.dumps(payload, ensure_ascii=True, indent=2) + "\n").encode("utf-8")


def _canonical_ledger_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8"), object_pairs_hook=_reject_duplicate_object_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid UTF-8 JSON in {label}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {label}")
    return payload


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be exactly 64 lowercase hexadecimal characters")
    return value


def _literal_assignment(snapshot: FileSnapshot, name: str) -> Any:
    try:
        tree = ast.parse(snapshot.data.decode("utf-8"), filename=snapshot.relative_path)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise RuntimeError(f"invalid Python source in {snapshot.relative_path}") from error
    values: list[Any] = []
    for node in tree.body:
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                value_node = node.value
        if value_node is not None:
            try:
                value = ast.literal_eval(value_node)
            except (ValueError, TypeError) as error:
                raise RuntimeError(f"{name} is not a literal value") from error
            values.append(value)
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {name} assignment")
    return values[0]


def _literal_integer_assignment(snapshot: FileSnapshot, name: str) -> int:
    value = _literal_assignment(snapshot, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} is not a literal integer")
    return value


def _assert_reservation_bound_before_code_pin(
    repo_fd: int,
    config: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Reject code finalization until a durable v4 reservation readback exists."""

    pins = config["runtime_pins"]
    metadata_snapshot = _snapshot_regular_file(
        repo_fd,
        str(pins["phase_a_kernel_metadata_path"]),
        label="v4 Phase-A kernel metadata",
    )
    runner_snapshot = _snapshot_regular_file(
        repo_fd,
        str(pins["phase_a_runner_path"]),
        label="v4 Phase-A runner",
    )
    launcher_snapshot = _snapshot_regular_file(
        repo_fd,
        str(pins["phase_a_launcher_path"]),
        label="v4 Phase-A launcher",
    )
    metadata = _decode_json_object(
        metadata_snapshot.data, label="v4 Phase-A kernel metadata"
    )
    metadata_id = metadata.get("id_no")
    runner_id = _literal_integer_assignment(runner_snapshot, "KERNEL_ID")
    launcher_id = _literal_integer_assignment(launcher_snapshot, "EXPECTED_KERNEL_ID")
    if (
        isinstance(metadata_id, bool)
        or not isinstance(metadata_id, int)
        or metadata_id <= 0
        or runner_id != metadata_id
        or launcher_id != metadata_id
    ):
        raise RuntimeError(
            "v4 Kaggle kernel reservation id is unresolved or inconsistent; "
            "code pinning is forbidden"
        )
    expectation = metadata.get("oracle_launch_expectation")
    if (
        metadata.get("id") != EXPECTED_KERNEL_SLUG
        or not isinstance(expectation, dict)
        or expectation.get("kernel_id") != metadata_id
        or expectation.get("kernel_slug") != EXPECTED_KERNEL_SLUG
    ):
        raise RuntimeError("v4 Phase-A metadata reservation binding drift")

    receipt_snapshot = _snapshot_regular_file(
        repo_fd,
        RESERVATION_RECEIPT_RELATIVE,
        label="v4 Kaggle reservation receipt",
    )
    orchestrator_snapshot = _snapshot_regular_file(
        repo_fd,
        RESERVATION_ORCHESTRATOR_RELATIVE,
        label="v4 reservation orchestrator",
    )
    if orchestrator_snapshot.sha256 != EXPECTED_RESERVATION_ORCHESTRATOR_SHA256:
        raise RuntimeError("v4 reservation orchestrator source drift")
    orchestrator_path = repo_root / RESERVATION_ORCHESTRATOR_RELATIVE
    spec = importlib.util.spec_from_file_location(
        "_candidate_graph_oracle_v4_exact_reservation_validator_a3a6ea",
        orchestrator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the hash-bound v4 reservation validator")
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)
    exact_envelope = validator._validate_existing_receipt(
        repo_root / RESERVATION_RECEIPT_RELATIVE
    )
    envelope = _decode_json_object(
        receipt_snapshot.data, label="v4 Kaggle reservation receipt"
    )
    if (
        set(envelope) != {"payload", "payload_sha256"}
        or not isinstance(envelope.get("payload"), dict)
        or receipt_snapshot.data != _canonical_ledger_bytes(envelope)
        or envelope.get("payload_sha256")
        != _canonical_object_sha256(envelope["payload"])
    ):
        raise RuntimeError("v4 Kaggle reservation receipt envelope drift")
    if exact_envelope != envelope:
        raise RuntimeError("hash-bound reservation validator projection drift")
    verifier_snapshot = _snapshot_regular_file(
        repo_fd,
        str(pins["result_verifier_path"]),
        label="v4 result verifier",
    )
    receipt_sha256 = receipt_snapshot.sha256
    runner_receipt_sha256 = _literal_assignment(
        runner_snapshot, "RESERVATION_RECEIPT_SHA256"
    )
    launcher_receipt_sha256 = _literal_assignment(
        launcher_snapshot, "RESERVATION_RECEIPT_SHA256"
    )
    verifier_receipt_sha256 = _literal_assignment(
        verifier_snapshot, "EXPECTED_RESERVATION_RECEIPT_SHA256"
    )
    if (
        _require_sha(receipt_sha256, label="v4 reservation receipt SHA-256")
        != metadata.get("reservation_receipt_sha256")
        or expectation.get("reservation_receipt_sha256") != receipt_sha256
        or runner_receipt_sha256 != receipt_sha256
        or launcher_receipt_sha256 != receipt_sha256
        or verifier_receipt_sha256 != receipt_sha256
    ):
        raise RuntimeError("v4 reservation receipt SHA-256 is not transitively bound")
    payload = envelope["payload"]
    kernel = payload.get("kernel")
    datasets = payload.get("datasets")
    local_validation = payload.get("local_validation")
    if (
        set(payload)
        != {
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
        }
        or payload.get("schema_version") != 1
        or payload.get("kind")
        != "candidate_graph_oracle_v4_kaggle_reservation_receipt"
        or payload.get("protocol_instance_id") != EXPECTED_PROTOCOL_INSTANCE_ID
        or payload.get("contains_fixture_pixels") is not False
        or payload.get("gpu_requested") is not False
        or payload.get("dataset_v2_uploaded") is not False
        or payload.get("phase_a_push_performed") is not False
        or payload.get("safe_for_submission") is not False
        or payload.get("reservation_orchestrator_sha256")
        != EXPECTED_RESERVATION_ORCHESTRATOR_SHA256
        or not isinstance(local_validation, dict)
        or local_validation.get("schema_version") != 1
        or local_validation.get("kind")
        != "candidate_graph_oracle_v4_local_reservation_validation"
        or local_validation.get("protocol_instance_id")
        != EXPECTED_PROTOCOL_INSTANCE_ID
        or local_validation.get("reservation_orchestrator_sha256")
        != EXPECTED_RESERVATION_ORCHESTRATOR_SHA256
        or local_validation.get("contains_fixture_pixels") is not False
        or local_validation.get("gpu_requested") is not False
        or local_validation.get("safe_for_submission") is not False
        or not isinstance(kernel, dict)
        or not isinstance(datasets, dict)
        or set(datasets) != set(EXPECTED_DATASET_SLUGS)
    ):
        raise RuntimeError("v4 Kaggle reservation receipt identity drift")
    if (
        kernel.get("slug") != EXPECTED_KERNEL_SLUG
        or kernel.get("kernel_id") != metadata_id
        or kernel.get("reserved_version") != 1
        or kernel.get("is_private") is not True
        or kernel.get("enable_gpu") is not False
        or kernel.get("enable_tpu") is not False
        or kernel.get("enable_internet") is not False
        or kernel.get("dataset_sources") != []
        or kernel.get("kernel_sources") != []
        or kernel.get("competition_sources") != []
        or kernel.get("model_sources") != []
        or str(kernel.get("status", "")).lower() != "complete"
        or kernel.get("reservation_runner_sha256")
        != EXPECTED_RESERVATION_RUNNER_SHA256
    ):
        raise RuntimeError("v4 Kaggle kernel reservation readback drift")
    for role, slug in EXPECTED_DATASET_SLUGS.items():
        record = datasets.get(role)
        if (
            not isinstance(record, dict)
            or record.get("slug") != slug
            or record.get("reserved_version") != 1
            or record.get("is_private") is not True
            or str(record.get("status", "")).lower() != "ready"
        ):
            raise RuntimeError(f"v4 Kaggle dataset reservation drift: {role}")


def _relative_parts(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty relative POSIX path")
    if "\x00" in value or "\\" in value:
        raise RuntimeError(f"{label} contains a forbidden separator or NUL")
    pure = PurePosixPath(value)
    parts = pure.parts
    if pure.is_absolute() or not parts:
        raise RuntimeError(f"{label} must be relative")
    if any(part in ("", ".", "..") for part in parts):
        raise RuntimeError(f"{label} is not normalized")
    if pure.as_posix() != value:
        raise RuntimeError(f"{label} is not canonical POSIX form")
    return tuple(parts)


def _lexical_relative(root: Path, path: Path, *, label: str) -> str:
    root_absolute = Path(os.path.abspath(os.fspath(root.expanduser())))
    path_absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        relative = path_absolute.relative_to(root_absolute).as_posix()
    except ValueError as error:
        raise RuntimeError(f"{label} is not anchored below the declared root") from error
    _relative_parts(relative, label=label)
    return relative


def _open_absolute_directory(path: Path, *, label: str) -> int:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    if not absolute.is_absolute():
        raise RuntimeError(f"{label} must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                raise RuntimeError(f"{label} has a missing or symlinked component: {part}") from error
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"{label} is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_directory(root_fd: int, parts: Sequence[str], *, label: str) -> int:
    descriptor = os.dup(root_fd)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                raise RuntimeError(f"{label} has a missing or symlinked directory: {part}") from error
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_parent(root_fd: int, relative: str, *, label: str) -> tuple[int, str]:
    parts = _relative_parts(relative, label=label)
    return _open_relative_directory(root_fd, parts[:-1], label=label), parts[-1]


def _snapshot_regular_file(root_fd: int, relative: str, *, label: str) -> FileSnapshot:
    parent_fd, name = _open_parent(root_fd, relative, label=label)
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise RuntimeError(f"{label} is missing, inaccessible, or a symlink: {relative}") from error
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {relative}")
        if before.st_nlink != 1:
            raise RuntimeError(f"{label} must have st_nlink == 1: {relative}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        stable_fields = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        after_fields = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        if stable_fields != after_fields:
            raise RuntimeError(f"{label} changed while being hashed: {relative}")
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (entry.st_dev, entry.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"{label} directory entry changed while being hashed: {relative}")
        data = b"".join(chunks)
        if len(data) != before.st_size:
            raise RuntimeError(f"{label} size changed while being hashed: {relative}")
        return FileSnapshot(
            relative_path=relative,
            data=data,
            sha256=digest.hexdigest(),
            device=before.st_dev,
            inode=before.st_ino,
            size=before.st_size,
            mode=before.st_mode,
            mtime_ns=before.st_mtime_ns,
            ctime_ns=before.st_ctime_ns,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _assert_unique_files(snapshots: Sequence[FileSnapshot], *, config: FileSnapshot) -> None:
    paths: set[str] = set()
    identities: set[tuple[int, int]] = {config.identity}
    for snapshot in snapshots:
        if snapshot.relative_path in paths:
            raise RuntimeError(f"duplicate pinned path: {snapshot.relative_path}")
        if snapshot.identity in identities:
            raise RuntimeError(
                f"pinned file aliases the config or another pinned file: {snapshot.relative_path}"
            )
        paths.add(snapshot.relative_path)
        identities.add(snapshot.identity)


def _assert_fixture_root_separation(bundle_fd: int) -> None:
    identities: set[tuple[int, int]] = set()
    for name in ("fixture_input", "fixture_label", "fixture_control"):
        descriptor = _open_relative_directory(
            bundle_fd, (name,), label=f"{name} root"
        )
        try:
            info = os.fstat(descriptor)
            identity = (info.st_dev, info.st_ino)
            if identity in identities:
                raise RuntimeError("fixture input/label/control roots alias each other")
            identities.add(identity)
        finally:
            os.close(descriptor)


def _pair_objects(
    pairs: Sequence[tuple[str, str, str]],
) -> list[dict[str, str]]:
    return [
        {"path_field": path_field, "sha256_field": sha_field}
        for path_field, sha_field, _ in pairs
    ]


def _validate_protocol_schema(
    config: Mapping[str, Any],
    *,
    expected_protocol_instance_id: str,
    expected_frozen_contract_sha256: str,
) -> None:
    if tuple(config.keys()) != EXPECTED_TOP_LEVEL_KEYS:
        raise RuntimeError("top-level protocol schema/key order drift or extra key")
    if config.get("schema_version") != SCHEMA_VERSION or config.get("kind") != PROTOCOL_KIND:
        raise RuntimeError("wrong candidate-graph oracle protocol")
    instance = config.get("protocol_instance_id")
    if (
        not isinstance(instance, str)
        or INSTANCE_RE.fullmatch(instance) is None
        or instance != expected_protocol_instance_id
    ):
        raise RuntimeError("protocol_instance_id is not the immutable expected value")
    contract = config.get("frozen_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("frozen_contract must be an object")
    actual_contract_hash = _canonical_object_sha256(contract)
    if actual_contract_hash != expected_frozen_contract_sha256:
        raise RuntimeError("frozen_contract differs from the immutable expected contract")
    if config.get("frozen_contract_sha256") != expected_frozen_contract_sha256:
        raise RuntimeError("frozen_contract_sha256 mismatch")
    if contract.get("protocol_instance_id") not in (None, instance):
        raise RuntimeError("frozen/top-level protocol_instance_id mismatch")
    if config.get("safe_for_submission") is not False:
        raise RuntimeError("protocol must remain unsafe for submission")

    pins = config.get("runtime_pins")
    if not isinstance(pins, dict) or tuple(pins.keys()) != EXPECTED_RUNTIME_PIN_KEYS:
        raise RuntimeError("runtime_pins schema/key order drift or extra key")
    for path_field, _, expected_path in (*EXPECTED_CODE_PIN_PAIRS, *EXPECTED_FIXTURE_PIN_PAIRS):
        if pins.get(path_field) != expected_path:
            raise RuntimeError(f"immutable runtime path drift: {path_field}")
        _relative_parts(pins[path_field], label=path_field)
    if pins.get("fixture_prep_marker_relative_path") != EXPECTED_PREP_MARKER_PATH:
        raise RuntimeError("immutable fixture prep marker path drift")
    if pins.get("phase_a_must_not_start_until_all_non_path_values_are_non_null") is not True:
        raise RuntimeError("phase-A non-null pin guard drift")

    policy = config.get("runtime_pin_mutation_policy")
    if not isinstance(policy, dict) or tuple(policy.keys()) != EXPECTED_POLICY_KEYS:
        raise RuntimeError("runtime pin mutation policy schema/key order drift or extra key")
    required_true = (
        "frozen_contract_is_immutable",
        "protocol_instance_id_is_immutable",
        "runtime_pin_paths_are_immutable",
        "runtime_pins_schema_and_key_order_are_immutable",
        "only_null_sha256_values_may_transition_once_to_lowercase_64_hex",
        "partial_pin_transition_forbidden",
        "all_code_test_runner_kernel_metadata_and_environment_hashes_must_be_pinned_before_fixture_pixel_access",
        "fixture_manifest_and_lock_hashes_may_be_pinned_once_after_fixture_preparation_but_before_phase_a",
        "every_pin_transition_requires_recomputing_and_recording_the_whole_config_file_sha256",
        "no_pin_may_change_after_phase_a_starts",
    )
    if any(policy.get(field) is not True for field in required_true):
        raise RuntimeError("required immutable runtime mutation guard is not true")
    if tuple(policy.get("top_level_immutable_fields", ())) != EXPECTED_TOP_LEVEL_IMMUTABLE_FIELDS:
        raise RuntimeError("top-level immutable field closure drift")
    if policy.get("code_pin_fields") != _pair_objects(EXPECTED_CODE_PIN_PAIRS):
        raise RuntimeError("code pin field closure/order drift")
    if policy.get("fixture_pin_fields") != _pair_objects(EXPECTED_FIXTURE_PIN_PAIRS):
        raise RuntimeError("fixture pin field closure/order drift")
    expected_ledger = (
        "runs/assembly_v1/protocol_ledgers/candidate_graph_oracle/"
        + expected_protocol_instance_id
    )
    if policy.get("transition_ledger_root") != expected_ledger:
        raise RuntimeError("transition ledger path drift")
    _relative_parts(expected_ledger, label="transition_ledger_root")

    for _, sha_field, _ in (*EXPECTED_CODE_PIN_PAIRS, *EXPECTED_FIXTURE_PIN_PAIRS):
        value = pins.get(sha_field)
        if value is not None:
            _require_sha(value, label=sha_field)


def _pin_state(config: Mapping[str, Any]) -> tuple[str, str]:
    pins = config["runtime_pins"]
    code = [pins[sha_field] for _, sha_field, _ in EXPECTED_CODE_PIN_PAIRS]
    fixtures = [pins[sha_field] for _, sha_field, _ in EXPECTED_FIXTURE_PIN_PAIRS]

    def classify(values: Sequence[Any], label: str) -> str:
        nulls = [value is None for value in values]
        if all(nulls):
            return "null"
        if any(nulls):
            raise RuntimeError(f"partial {label} pin transition is forbidden")
        for index, value in enumerate(values):
            _require_sha(value, label=f"{label} pin {index}")
        return "pinned"

    code_state = classify(code, "code")
    fixture_state = classify(fixtures, "fixture")
    if code_state == "null" and fixture_state == "pinned":
        raise RuntimeError("fixture pins may not precede code pins")
    return code_state, fixture_state


def _mkdirs_anchored(root_fd: int, relative: str, *, label: str) -> int:
    parts = _relative_parts(relative, label=label)
    descriptor = os.dup(root_fd)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts:
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                raise RuntimeError(f"{label} contains a non-directory or symlink: {part}") from error
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _exclusive_ledger_json(directory_fd: int, name: str, payload: Mapping[str, Any]) -> str:
    if "/" in name or name in ("", ".", ".."):
        raise RuntimeError("invalid transition ledger filename")
    encoded = _canonical_ledger_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("new transition ledger entry is not an unlinked regular file")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise RuntimeError("short write to transition ledger")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)
    return hashlib.sha256(encoded).hexdigest()


def _ledger_snapshot(directory_fd: int, name: str) -> FileSnapshot | None:
    try:
        return _snapshot_regular_file(directory_fd, name, label="transition ledger entry")
    except RuntimeError as error:
        cause = error.__cause__
        if isinstance(cause, OSError) and cause.errno == errno.ENOENT:
            return None
        raise


def _load_existing_ledger_entry(
    directory_fd: int,
    name: str,
    *,
    expected_keys: Sequence[str],
) -> tuple[dict[str, Any], FileSnapshot] | None:
    snapshot = _ledger_snapshot(directory_fd, name)
    if snapshot is None:
        return None
    payload = _decode_json_object(snapshot.data, label=name)
    if set(payload.keys()) != set(expected_keys) or len(payload) != len(expected_keys):
        raise RuntimeError(f"transition ledger schema drift or extra key: {name}")
    if snapshot.data != _canonical_ledger_bytes(payload):
        raise RuntimeError(f"transition ledger is not canonical: {name}")
    return payload, snapshot


def _stage_names(stage: str) -> tuple[str, str]:
    prefix = f"{STAGE_INDEX[stage]:02d}_{stage}_pins"
    return f"{prefix}.intent.json", f"{prefix}.complete.json"


def _assert_transition_directory_entries(directory_fd: int, *, stage: str) -> None:
    names = set(os.listdir(directory_fd))
    code_names = set(_stage_names("code"))
    fixture_names = set(_stage_names("fixtures"))
    allowed = code_names if stage == "code" else code_names | fixture_names
    extra = names - allowed
    if extra:
        raise RuntimeError(f"unexpected transition ledger entry: {sorted(extra)}")


def _validate_common_ledger_fields(
    payload: Mapping[str, Any],
    *,
    kind: str,
    stage: str,
    instance: str,
    frozen_hash: str,
    config_relative: str,
) -> None:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != kind
        or payload.get("stage") != stage
        or payload.get("stage_index") != STAGE_INDEX[stage]
        or payload.get("protocol_instance_id") != instance
        or payload.get("frozen_contract_sha256") != frozen_hash
        or payload.get("config_relative_path") != config_relative
    ):
        raise RuntimeError("transition ledger immutable fields mismatch")


def _validate_pin_value_map(
    value: Any, *, expected_fields: Sequence[str]
) -> dict[str, str]:
    # Ledger JSON is canonicalized with ``sort_keys=True``, so nested object
    # order is lexicographic on disk.  The closed key set, not insertion order,
    # is the machine-readable invariant here.
    if (
        not isinstance(value, dict)
        or set(value.keys()) != set(expected_fields)
        or len(value) != len(expected_fields)
    ):
        raise RuntimeError("transition pin map schema/order drift or extra key")
    result: dict[str, str] = {}
    for field in expected_fields:
        result[field] = _require_sha(value[field], label=f"ledger {field}")
    return result


def _write_completion(
    transition_fd: int, name: str, payload: Mapping[str, Any]
) -> str:
    return _exclusive_ledger_json(transition_fd, name, payload)


def _atomic_replace_config(
    repo_fd: int,
    config_relative: str,
    *,
    expected_before: FileSnapshot,
    final_bytes: bytes,
) -> FileSnapshot:
    parent_fd, name = _open_parent(repo_fd, config_relative, label="protocol config")
    temporary = f".{name}.runtime-pin-{secrets.token_hex(12)}.tmp"
    descriptor = -1
    created = False
    try:
        current = _snapshot_regular_file(repo_fd, config_relative, label="protocol config")
        if current.sha256 != expected_before.sha256 or current.identity != expected_before.identity:
            raise RuntimeError("protocol config changed before atomic replacement")
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            stat.S_IMODE(expected_before.mode),
            dir_fd=parent_fd,
        )
        created = True
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("temporary config is not an unlinked regular file")
        offset = 0
        while offset < len(final_bytes):
            written = os.write(descriptor, final_bytes[offset:])
            if written <= 0:
                raise RuntimeError("short write to temporary protocol config")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1

        current_again = _snapshot_regular_file(repo_fd, config_relative, label="protocol config")
        if (
            current_again.sha256 != expected_before.sha256
            or current_again.identity != expected_before.identity
        ):
            raise RuntimeError("protocol config changed during transition")
        os.replace(
            temporary,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        created = False
        os.fsync(parent_fd)
        final = _snapshot_regular_file(repo_fd, config_relative, label="final protocol config")
        if final.data != final_bytes:
            raise RuntimeError("atomic protocol config replacement did not preserve canonical bytes")
        return final
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _hash_code_pins(
    repo_fd: int,
    config: Mapping[str, Any],
    *,
    config_snapshot: FileSnapshot,
) -> tuple[dict[str, str], list[FileSnapshot]]:
    pins = config["runtime_pins"]
    values: dict[str, str] = {}
    snapshots: list[FileSnapshot] = []
    for path_field, sha_field, expected_path in EXPECTED_CODE_PIN_PAIRS:
        if pins[path_field] != expected_path:
            raise RuntimeError(f"code path drift: {path_field}")
        snapshot = _snapshot_regular_file(
            repo_fd, expected_path, label=f"code pin {sha_field}"
        )
        values[sha_field] = snapshot.sha256
        snapshots.append(snapshot)
    environment_snapshot = next(
        snapshot
        for snapshot in snapshots
        if snapshot.relative_path
        == "configs/candidate_graph_oracle_environment_lock_v1.json"
    )
    _validate_environment_lock_against_frozen_contract(
        repo_fd=repo_fd,
        config=config,
        snapshot=environment_snapshot,
    )
    _assert_unique_files(snapshots, config=config_snapshot)
    return values, snapshots


def _validate_environment_lock_against_frozen_contract(
    *,
    repo_fd: int,
    config: Mapping[str, Any],
    snapshot: FileSnapshot,
) -> None:
    """Prevent pinning an environment lock that contradicts the frozen contract."""

    runtime = config.get("frozen_contract", {}).get("runtime_environment")
    # Minimal synthetic protocols used by adversarial finalizer tests may omit
    # the ML runtime contract entirely.  The production protocol contains it,
    # and then this check is mandatory and exact.
    if runtime is None:
        return
    if not isinstance(runtime, dict):
        raise RuntimeError("frozen runtime_environment is malformed")
    lock = _decode_json_object(snapshot.data, label="environment lock")
    if (
        lock.get("schema_version") != SCHEMA_VERSION
        or lock.get("kind") != "candidate_graph_oracle_environment_lock"
    ):
        raise RuntimeError("environment lock identity drift")

    package_names = (
        "numpy",
        "opencv",
        "pillow",
        "kornia",
        "scikit_image",
        "scipy",
        "torch",
    )
    frozen_local = runtime.get("fixture_preparation_and_phase_b")
    locked_local = lock.get("fixture_preparation_and_phase_b")
    if not isinstance(frozen_local, dict) or not isinstance(locked_local, dict):
        raise RuntimeError("local environment contract is missing")
    expected_local_packages = {
        name: frozen_local.get(name) for name in package_names
    }
    if (
        locked_local.get("python") != frozen_local.get("python")
        or locked_local.get("packages") != expected_local_packages
        or frozen_local.get("environment")
        not in str(locked_local.get("execution", ""))
        or locked_local.get("exact_match_required_before_fixture_pixel_access")
        is not True
        or locked_local.get(
            "phase_b_runs_in_a_fresh_local_process_with_the_same_exact_environment"
        )
        is not True
    ):
        raise RuntimeError("environment lock contradicts frozen local runtime")

    frozen_kaggle = runtime.get("kaggle_phase_a")
    locked_kaggle = lock.get("kaggle_phase_a")
    if not isinstance(frozen_kaggle, dict) or not isinstance(locked_kaggle, dict):
        raise RuntimeError("Kaggle environment contract is missing")
    expected_kaggle_packages = {
        name: frozen_kaggle.get(name) for name in package_names
    }
    expected_devices = [
        {"index": index, "name": "Tesla T4", "capability": [7, 5]}
        for index in range(2)
    ]
    if (
        locked_kaggle.get("python") != frozen_kaggle.get("python")
        or locked_kaggle.get("packages") != expected_kaggle_packages
        or locked_kaggle.get("cuda_runtime")
        != frozen_kaggle.get("cuda_runtime")
        or locked_kaggle.get("device_count") != 2
        or locked_kaggle.get("devices") != expected_devices
        or frozen_kaggle.get("device") != "2x Tesla T4 sm_75"
        or locked_kaggle.get("real_tensor_probe_required_on_each_device") is not True
        or locked_kaggle.get("phase_a_exact_library_match_required") is not True
    ):
        raise RuntimeError("environment lock contradicts frozen Kaggle runtime")

    reference = locked_kaggle.get("reference_evidence")
    if isinstance(reference, dict):
        relative = reference.get("path")
        expected_sha = reference.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise RuntimeError("environment reference evidence is malformed")
        evidence = _snapshot_regular_file(
            repo_fd, relative, label="environment reference evidence"
        )
        if evidence.sha256 != expected_sha:
            raise RuntimeError("environment reference evidence SHA256 mismatch")


def _forbid_whole_config_binding(
    value: Any,
    *,
    forbidden_hashes: set[str],
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if "config" in lowered and ("sha" in lowered or "hash" in lowered):
                raise RuntimeError(
                    f"fixture artifact binds a whole-config hash field: {'.'.join((*path, key))}"
                )
            _forbid_whole_config_binding(
                child, forbidden_hashes=forbidden_hashes, path=(*path, key)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbid_whole_config_binding(
                child, forbidden_hashes=forbidden_hashes, path=(*path, str(index))
            )
    elif isinstance(value, str) and value in forbidden_hashes:
        raise RuntimeError("fixture artifact embeds the whole-config SHA-256")


def _validate_fixture_bindings(
    *,
    config: Mapping[str, Any],
    input_payload: Mapping[str, Any],
    label_payload: Mapping[str, Any],
    lock_payload: Mapping[str, Any],
    input_sha256: str,
    label_sha256: str,
    marker_sha256: str,
    previous_config_sha256: str,
    intended_config_sha256: str,
) -> None:
    expected_kinds = (
        "candidate_graph_oracle_fixture_inputs",
        "candidate_graph_oracle_fixture_labels",
        "candidate_graph_oracle_fixture_lock",
    )
    for payload, kind, label in zip(
        (input_payload, label_payload, lock_payload),
        expected_kinds,
        ("input manifest", "label manifest", "fixture lock"),
        strict=True,
    ):
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != kind:
            raise RuntimeError(f"wrong {label} schema or kind")

    contract = config["frozen_contract"]
    preparation = contract.get("fixture_preparation")
    if not isinstance(preparation, dict):
        raise RuntimeError("frozen fixture preparation contract is missing")
    expected_common_fields = ["protocol_instance_id", "frozen_contract_sha256"] + [
        sha_field for _, sha_field, _ in EXPECTED_CODE_PIN_PAIRS
    ]
    if preparation.get("exact_common_manifest_binding_field_names") != expected_common_fields:
        raise RuntimeError("common fixture binding field closure/order drift")
    crosslinks = preparation.get("exact_crosslink_field_names")
    expected_crosslinks = {
        "label_to_input": "fixture_input_manifest_sha256",
        "lock_to_input": "fixture_input_manifest_sha256",
        "lock_to_label": "fixture_label_manifest_sha256",
        "lock_to_prep_marker": "prep_marker_sha256",
    }
    if crosslinks != expected_crosslinks:
        raise RuntimeError("fixture crosslink field schema drift")

    pins = config["runtime_pins"]
    common_values = {
        "protocol_instance_id": config["protocol_instance_id"],
        "frozen_contract_sha256": config["frozen_contract_sha256"],
        **{sha_field: pins[sha_field] for _, sha_field, _ in EXPECTED_CODE_PIN_PAIRS},
    }
    for payload, label in (
        (input_payload, "input manifest"),
        (label_payload, "label manifest"),
        (lock_payload, "fixture lock"),
    ):
        for field in expected_common_fields:
            if payload.get(field) != common_values[field]:
                raise RuntimeError(f"{label} common binding mismatch: {field}")

    input_field = expected_crosslinks["label_to_input"]
    label_field = expected_crosslinks["lock_to_label"]
    marker_field = expected_crosslinks["lock_to_prep_marker"]
    if any(field in input_payload for field in (input_field, label_field, marker_field)):
        raise RuntimeError("input manifest violates one-way fixture crosslink order")
    if label_payload.get(input_field) != input_sha256:
        raise RuntimeError("label manifest does not bind the exact input manifest")
    if label_field in label_payload or marker_field in label_payload:
        raise RuntimeError("label manifest violates one-way fixture crosslink order")
    if (
        lock_payload.get(input_field) != input_sha256
        or lock_payload.get(label_field) != label_sha256
        or lock_payload.get(marker_field) != marker_sha256
    ):
        raise RuntimeError("fixture lock crosslink mismatch")

    forbidden = {previous_config_sha256, intended_config_sha256}
    for payload in (input_payload, label_payload, lock_payload):
        _forbid_whole_config_binding(payload, forbidden_hashes=forbidden)


def _hash_fixture_pins(
    bundle_fd: int,
    config: Mapping[str, Any],
    *,
    config_snapshot: FileSnapshot,
    previous_config_sha256: str,
    intended_config_sha256_placeholder: str | None = None,
) -> tuple[dict[str, str], list[FileSnapshot], dict[str, dict[str, Any]], FileSnapshot]:
    pins = config["runtime_pins"]
    snapshots: list[FileSnapshot] = []
    payloads: dict[str, dict[str, Any]] = {}
    field_to_label = {
        "fixture_input_manifest_sha256": "input",
        "fixture_label_manifest_sha256": "label",
        "fixture_lock_sha256": "lock",
    }
    values: dict[str, str] = {}
    for path_field, sha_field, expected_path in EXPECTED_FIXTURE_PIN_PAIRS:
        if pins[path_field] != expected_path:
            raise RuntimeError(f"fixture path drift: {path_field}")
        snapshot = _snapshot_regular_file(
            bundle_fd, expected_path, label=f"fixture pin {sha_field}"
        )
        snapshots.append(snapshot)
        values[sha_field] = snapshot.sha256
        payloads[field_to_label[sha_field]] = _decode_json_object(
            snapshot.data, label=expected_path
        )
    marker = _snapshot_regular_file(
        bundle_fd,
        EXPECTED_PREP_MARKER_PATH,
        label="fixture preparation marker",
    )
    _assert_unique_files([*snapshots, marker], config=config_snapshot)

    # The final config hash depends on these three hashes.  Compute it before
    # checking that none of the fixture JSON objects embeds that final hash.
    candidate = copy.deepcopy(config)
    for field, digest in values.items():
        candidate["runtime_pins"][field] = digest
    intended = hashlib.sha256(_canonical_config_bytes(candidate)).hexdigest()
    if intended_config_sha256_placeholder is not None and intended != intended_config_sha256_placeholder:
        raise RuntimeError("fixture transition intended config hash drift")
    _validate_fixture_bindings(
        config=config,
        input_payload=payloads["input"],
        label_payload=payloads["label"],
        lock_payload=payloads["lock"],
        input_sha256=values["fixture_input_manifest_sha256"],
        label_sha256=values["fixture_label_manifest_sha256"],
        marker_sha256=marker.sha256,
        previous_config_sha256=previous_config_sha256,
        intended_config_sha256=intended,
    )
    return values, snapshots, payloads, marker


def _rehash_expected(
    root_fd: int,
    path_and_hashes: Sequence[tuple[str, str]],
    *,
    label: str,
) -> None:
    for relative, expected in path_and_hashes:
        actual = _snapshot_regular_file(root_fd, relative, label=label).sha256
        if actual != expected:
            raise RuntimeError(f"{label} changed after transition intent: {relative}")


def _validate_existing_intent(
    payload: Mapping[str, Any],
    *,
    stage: str,
    config_relative: str,
    instance: str,
    frozen_hash: str,
    pin_fields: Sequence[str],
) -> dict[str, str]:
    _validate_common_ledger_fields(
        payload,
        kind=INTENT_KIND,
        stage=stage,
        instance=instance,
        frozen_hash=frozen_hash,
        config_relative=config_relative,
    )
    _require_sha(payload.get("previous_config_sha256"), label="intent previous config hash")
    _require_sha(payload.get("intended_config_sha256"), label="intent intended config hash")
    return _validate_pin_value_map(payload.get("pin_sha256_values"), expected_fields=pin_fields)


def _validate_existing_completion(
    payload: Mapping[str, Any],
    *,
    stage: str,
    config_relative: str,
    instance: str,
    frozen_hash: str,
    pin_fields: Sequence[str],
    intent: Mapping[str, Any],
    intent_sha256: str,
) -> None:
    _validate_common_ledger_fields(
        payload,
        kind=COMPLETION_KIND,
        stage=stage,
        instance=instance,
        frozen_hash=frozen_hash,
        config_relative=config_relative,
    )
    pins = _validate_pin_value_map(payload.get("pin_sha256_values"), expected_fields=pin_fields)
    if (
        payload.get("previous_config_sha256") != intent.get("previous_config_sha256")
        or payload.get("final_config_sha256") != intent.get("intended_config_sha256")
        or pins != intent.get("pin_sha256_values")
        or payload.get("intent_sha256") != intent_sha256
    ):
        raise RuntimeError("transition completion does not match its immutable intent")


def finalize_runtime_pins(
    *,
    config_path: Path,
    expected_config_sha256: str,
    stage: str,
    fixture_bundle_root: Path | None = None,
    repo_root: Path = REPO_ROOT,
    expected_protocol_instance_id: str = EXPECTED_PROTOCOL_INSTANCE_ID,
    expected_frozen_contract_sha256: str = EXPECTED_FROZEN_CONTRACT_SHA256,
) -> dict[str, Any]:
    """Finalize one exact pin stage or recover the same interrupted stage."""

    if stage not in STAGES:
        raise RuntimeError(f"stage must be one of {STAGES}")
    expected_config_sha256 = _require_sha(
        expected_config_sha256, label="out-of-band expected whole-config SHA-256"
    )
    _require_sha(expected_frozen_contract_sha256, label="expected frozen contract hash")
    if INSTANCE_RE.fullmatch(expected_protocol_instance_id) is None:
        raise RuntimeError("expected protocol instance id is malformed")
    if stage == "code" and fixture_bundle_root is not None:
        raise RuntimeError("code stage must not receive or inspect a fixture bundle root")
    if stage == "fixtures" and fixture_bundle_root is None:
        raise RuntimeError("fixture stage requires --fixture-bundle-root")

    repo_fd = _open_absolute_directory(repo_root, label="repository root")
    bundle_fd: int | None = None
    transition_fd: int | None = None
    try:
        config_relative = _lexical_relative(
            repo_root, config_path, label="protocol config"
        )
        config_snapshot = _snapshot_regular_file(
            repo_fd, config_relative, label="protocol config"
        )
        if config_snapshot.sha256 != expected_config_sha256:
            raise RuntimeError(
                "whole-config SHA-256 does not match the exact out-of-band expectation"
            )
        config = _decode_json_object(config_snapshot.data, label="protocol config")
        _validate_protocol_schema(
            config,
            expected_protocol_instance_id=expected_protocol_instance_id,
            expected_frozen_contract_sha256=expected_frozen_contract_sha256,
        )
        code_state, fixture_state = _pin_state(config)
        if stage == "code" and code_state == "null":
            _assert_reservation_bound_before_code_pin(
                repo_fd, config, repo_root=repo_root
            )
        policy = config["runtime_pin_mutation_policy"]
        ledger_fd = _mkdirs_anchored(
            repo_fd, policy["transition_ledger_root"], label="transition ledger root"
        )
        try:
            transition_fd = _mkdirs_anchored(
                ledger_fd, TRANSITION_DIR_NAME, label="runtime pin transition ledger"
            )
        finally:
            os.close(ledger_fd)
        _assert_transition_directory_entries(transition_fd, stage=stage)

        intent_name, completion_name = _stage_names(stage)
        intent_existing = _load_existing_ledger_entry(
            transition_fd, intent_name, expected_keys=INTENT_KEYS
        )
        completion_existing = _load_existing_ledger_entry(
            transition_fd, completion_name, expected_keys=COMPLETION_KEYS
        )
        if completion_existing is not None and intent_existing is None:
            raise RuntimeError("transition completion exists without an intent")

        if stage == "fixtures":
            code_intent_name, code_completion_name = _stage_names("code")
            code_intent = _load_existing_ledger_entry(
                transition_fd, code_intent_name, expected_keys=INTENT_KEYS
            )
            code_completion = _load_existing_ledger_entry(
                transition_fd, code_completion_name, expected_keys=COMPLETION_KEYS
            )
            if code_intent is None or code_completion is None:
                raise RuntimeError("fixture stage requires a completed code-pin transition")
            code_intent_payload, code_intent_snapshot = code_intent
            _validate_existing_intent(
                code_intent_payload,
                stage="code",
                config_relative=config_relative,
                instance=expected_protocol_instance_id,
                frozen_hash=expected_frozen_contract_sha256,
                pin_fields=[sha for _, sha, _ in EXPECTED_CODE_PIN_PAIRS],
            )
            _validate_existing_completion(
                code_completion[0],
                stage="code",
                config_relative=config_relative,
                instance=expected_protocol_instance_id,
                frozen_hash=expected_frozen_contract_sha256,
                pin_fields=[sha for _, sha, _ in EXPECTED_CODE_PIN_PAIRS],
                intent=code_intent_payload,
                intent_sha256=code_intent_snapshot.sha256,
            )

        pin_fields = [
            sha
            for _, sha, _ in (
                EXPECTED_CODE_PIN_PAIRS if stage == "code" else EXPECTED_FIXTURE_PIN_PAIRS
            )
        ]

        intended_from_ledger: str | None = None
        prior_from_ledger: str | None = None
        ledger_pin_values: dict[str, str] | None = None
        intent_sha256: str | None = None
        if intent_existing is not None:
            intent_payload, intent_snapshot = intent_existing
            ledger_pin_values = _validate_existing_intent(
                intent_payload,
                stage=stage,
                config_relative=config_relative,
                instance=expected_protocol_instance_id,
                frozen_hash=expected_frozen_contract_sha256,
                pin_fields=pin_fields,
            )
            prior_from_ledger = str(intent_payload["previous_config_sha256"])
            intended_from_ledger = str(intent_payload["intended_config_sha256"])
            intent_sha256 = intent_snapshot.sha256
            if config_snapshot.sha256 not in (prior_from_ledger, intended_from_ledger):
                raise RuntimeError("current config matches neither side of the existing intent")

        if stage == "code":
            if fixture_state != "null":
                raise RuntimeError("code stage cannot run after fixture pins exist")
            pin_values, pin_snapshots = _hash_code_pins(
                repo_fd, config, config_snapshot=config_snapshot
            )
            root_fd_for_rehash = repo_fd
            path_and_hashes = [
                (path, pin_values[sha]) for _, sha, path in EXPECTED_CODE_PIN_PAIRS
            ]
        else:
            if code_state != "pinned":
                raise RuntimeError("fixture stage requires all code pins")
            # Re-verify every code pin before trusting fixture provenance.
            actual_code, code_snapshots = _hash_code_pins(
                repo_fd, config, config_snapshot=config_snapshot
            )
            for field, digest in actual_code.items():
                if config["runtime_pins"][field] != digest:
                    raise RuntimeError(f"existing code pin changed before fixture pinning: {field}")
            bundle_fd = _open_absolute_directory(
                fixture_bundle_root, label="fixture bundle root"  # type: ignore[arg-type]
            )
            _assert_fixture_root_separation(bundle_fd)
            pin_values, pin_snapshots, _, marker = _hash_fixture_pins(
                bundle_fd,
                config,
                config_snapshot=config_snapshot,
                previous_config_sha256=(
                    prior_from_ledger or config_snapshot.sha256
                ),
                intended_config_sha256_placeholder=intended_from_ledger,
            )
            _assert_unique_files(
                [*code_snapshots, *pin_snapshots, marker], config=config_snapshot
            )
            root_fd_for_rehash = bundle_fd
            path_and_hashes = [
                (path, pin_values[sha]) for _, sha, path in EXPECTED_FIXTURE_PIN_PAIRS
            ] + [(EXPECTED_PREP_MARKER_PATH, marker.sha256)]

        if ledger_pin_values is not None and pin_values != ledger_pin_values:
            raise RuntimeError("pinned artifact hashes differ from the append-only intent")

        current_pins = config["runtime_pins"]
        stage_state = code_state if stage == "code" else fixture_state
        if stage_state == "pinned":
            if intent_existing is None:
                raise RuntimeError("pre-existing pins have no append-only transition intent")
            if intended_from_ledger != config_snapshot.sha256:
                raise RuntimeError("pinned config hash does not match transition intent")
            for field, digest in pin_values.items():
                if current_pins[field] != digest:
                    raise RuntimeError(f"existing SHA pin changed: {field}")
            if completion_existing is not None:
                assert intent_sha256 is not None
                _validate_existing_completion(
                    completion_existing[0],
                    stage=stage,
                    config_relative=config_relative,
                    instance=expected_protocol_instance_id,
                    frozen_hash=expected_frozen_contract_sha256,
                    pin_fields=pin_fields,
                    intent=intent_existing[0],
                    intent_sha256=intent_sha256,
                )
                return {
                    "status": "already_completed",
                    "stage": stage,
                    "protocol_instance_id": expected_protocol_instance_id,
                    "previous_config_sha256": prior_from_ledger,
                    "final_config_sha256": config_snapshot.sha256,
                    "pin_sha256_values": pin_values,
                    "safe_for_submission": False,
                }
            final_snapshot = config_snapshot
        else:
            if completion_existing is not None:
                raise RuntimeError("completion exists while stage pins are still null")
            candidate = copy.deepcopy(config)
            for field, digest in pin_values.items():
                if candidate["runtime_pins"][field] is not None:
                    raise RuntimeError(f"refusing to repin existing SHA: {field}")
                candidate["runtime_pins"][field] = digest
            _validate_protocol_schema(
                candidate,
                expected_protocol_instance_id=expected_protocol_instance_id,
                expected_frozen_contract_sha256=expected_frozen_contract_sha256,
            )
            candidate_code_state, candidate_fixture_state = _pin_state(candidate)
            if stage == "code" and (candidate_code_state, candidate_fixture_state) != (
                "pinned",
                "null",
            ):
                raise RuntimeError("code transition mutated the wrong pin closure")
            if stage == "fixtures" and (
                candidate_code_state,
                candidate_fixture_state,
            ) != ("pinned", "pinned"):
                raise RuntimeError("fixture transition mutated the wrong pin closure")
            final_bytes = _canonical_config_bytes(candidate)
            final_sha256 = hashlib.sha256(final_bytes).hexdigest()

            if intent_existing is None:
                intent_payload = {
                    "schema_version": SCHEMA_VERSION,
                    "kind": INTENT_KIND,
                    "stage": stage,
                    "stage_index": STAGE_INDEX[stage],
                    "protocol_instance_id": expected_protocol_instance_id,
                    "frozen_contract_sha256": expected_frozen_contract_sha256,
                    "config_relative_path": config_relative,
                    "previous_config_sha256": config_snapshot.sha256,
                    "intended_config_sha256": final_sha256,
                    "pin_sha256_values": pin_values,
                    "created_utc": _utc_now(),
                }
                intent_sha256 = _exclusive_ledger_json(
                    transition_fd, intent_name, intent_payload
                )
                prior_from_ledger = config_snapshot.sha256
                intended_from_ledger = final_sha256
                intent_existing = (
                    intent_payload,
                    FileSnapshot(
                        relative_path=intent_name,
                        data=_canonical_ledger_bytes(intent_payload),
                        sha256=intent_sha256,
                        device=-1,
                        inode=-1,
                        size=len(_canonical_ledger_bytes(intent_payload)),
                        mode=stat.S_IFREG | 0o600,
                        mtime_ns=-1,
                        ctime_ns=-1,
                    ),
                )
            else:
                if (
                    prior_from_ledger != config_snapshot.sha256
                    or intended_from_ledger != final_sha256
                ):
                    raise RuntimeError("recovered transition candidate differs from its intent")

            _rehash_expected(
                root_fd_for_rehash,
                path_and_hashes,
                label=f"{stage} pinned artifact",
            )
            final_snapshot = _atomic_replace_config(
                repo_fd,
                config_relative,
                expected_before=config_snapshot,
                final_bytes=final_bytes,
            )
            if final_snapshot.sha256 != intended_from_ledger:
                raise RuntimeError("final whole-config hash differs from transition intent")
            # A mutation in the narrow gap between the pre-commit rehash and
            # config replacement leaves the transition intentionally
            # incomplete and therefore unusable.
            _rehash_expected(
                root_fd_for_rehash,
                path_and_hashes,
                label=f"post-commit {stage} pinned artifact",
            )

        assert intent_existing is not None and intent_sha256 is not None
        completion_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": COMPLETION_KIND,
            "stage": stage,
            "stage_index": STAGE_INDEX[stage],
            "protocol_instance_id": expected_protocol_instance_id,
            "frozen_contract_sha256": expected_frozen_contract_sha256,
            "config_relative_path": config_relative,
            "previous_config_sha256": prior_from_ledger,
            "final_config_sha256": final_snapshot.sha256,
            "pin_sha256_values": pin_values,
            "intent_sha256": intent_sha256,
            "completed_utc": _utc_now(),
        }
        _write_completion(transition_fd, completion_name, completion_payload)
        return {
            "status": "completed" if stage_state == "null" else "recovered_completion",
            "stage": stage,
            "protocol_instance_id": expected_protocol_instance_id,
            "previous_config_sha256": prior_from_ledger,
            "final_config_sha256": final_snapshot.sha256,
            "pin_sha256_values": pin_values,
            "safe_for_submission": False,
        }
    finally:
        if transition_fd is not None:
            os.close(transition_fd)
        if bundle_fd is not None:
            os.close(bundle_fd)
        os.close(repo_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--fixture-bundle-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = finalize_runtime_pins(
        config_path=args.config,
        expected_config_sha256=args.expected_config_sha256,
        stage=args.stage,
        fixture_bundle_root=args.fixture_bundle_root,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
