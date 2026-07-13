#!/usr/bin/env python3
"""Build deterministic, input-only Kaggle dataset-v2 payload archives.

This is deliberately a local packaging step, not an uploader.  It accepts the
already separated input fixture root directly and has no bundle-root or hidden
fixture-root argument.  Input NPZ files are copied as opaque bytes; they are
never decoded.

The three ZIP member trees are the trees that Kaggle must expose after the
archives are extracted for dataset version 2.  ``dataset-metadata.json`` files
are external sidecars because Kaggle consumes them as upload metadata and the
Phase-A runner rejects them as mounted payload files.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile


SCHEMA_VERSION = 1
PROTOCOL_KIND = "candidate_graph_oracle_ceiling"
CONFIG_MEMBER = "configs/candidate_graph_oracle_ceiling_v3.json"
INPUT_MANIFEST_NAME = "fixture_input_manifest.json"
TRANSITION_DIRECTORY = "runtime_pin_transitions"
LIFECYCLE_MEMBERS = (
    "lifecycle/PREP.json",
    "lifecycle/SEALED.json",
    "lifecycle/PHASE_A.json",
    "lifecycle/runtime_pin_transitions/00_code_pins.intent.json",
    "lifecycle/runtime_pin_transitions/00_code_pins.complete.json",
    "lifecycle/runtime_pin_transitions/01_fixtures_pins.intent.json",
    "lifecycle/runtime_pin_transitions/01_fixtures_pins.complete.json",
)
TRANSITION_NAMES = (
    "00_code_pins.intent.json",
    "00_code_pins.complete.json",
    "01_fixtures_pins.intent.json",
    "01_fixtures_pins.complete.json",
)
LIFECYCLE_STATES = ("PREP", "SEALED", "PHASE_A")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
OPAQUE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
RECEIPT_NAME = "CANDIDATE_GRAPH_ORACLE_KAGGLE_BUNDLE_BUILD_RECEIPT.json"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    slug: str
    title: str
    archive_name: str


DATASETS = (
    DatasetSpec(
        key="code",
        slug="pasha883/vsos-candidate-graph-oracle-v3-code",
        title="VSOS Candidate Graph Oracle V3 Code",
        archive_name="candidate_graph_oracle_v3_code_v2.zip",
    ),
    DatasetSpec(
        key="input",
        slug="pasha883/vsos-candidate-graph-oracle-v3-inputs",
        title="VSOS Candidate Graph Oracle V3 Inputs",
        archive_name="candidate_graph_oracle_v3_inputs_v2.zip",
    ),
    DatasetSpec(
        key="runtime",
        slug="pasha883/vsos-candidate-graph-oracle-v3-runtime",
        title="VSOS Candidate Graph Oracle V3 Runtime",
        archive_name="candidate_graph_oracle_v3_runtime_v2.zip",
    ),
)
DATASET_BY_KEY = {item.key: item for item in DATASETS}


@dataclass(frozen=True)
class FileSnapshot:
    source: Path
    member: str
    sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    @property
    def identity(self) -> tuple[int, int]:
        return (self.device, self.inode)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _canonical_object_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid UTF-8 JSON: {label}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {label}")
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a populated lowercase SHA-256")
    return value


def _relative_member(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise RuntimeError(f"invalid relative POSIX path for {label}")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise RuntimeError(f"non-canonical relative POSIX path for {label}")
    return value


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _assert_directory_anchor(path: Path, *, label: str) -> Path:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(f"missing directory component for {label}") from error
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"symlinked directory component for {label}")
    if not absolute.is_dir():
        raise RuntimeError(f"{label} is not a directory")
    return absolute


def _safe_child_file(root: Path, relative: str, *, label: str) -> Path:
    parts = PurePosixPath(_relative_member(relative, label=label)).parts
    current = root
    for part in parts[:-1]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(f"missing directory in {label}") from error
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"non-directory or symlink in {label}")
    path = current / parts[-1]
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"missing file for {label}") from error
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"non-regular or symlinked file for {label}")
    return path


def _snapshot_regular(path: Path, *, member: str, collect: bool = False) -> tuple[FileSnapshot, bytes | None]:
    member = _relative_member(member, label="archive member")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if collect else None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"source must be a regular file with nlink==1: {member}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        ):
            raise RuntimeError(f"source changed while hashing: {member}")
    finally:
        os.close(descriptor)
    snapshot = FileSnapshot(
        source=path,
        member=member,
        sha256=digest.hexdigest(),
        size=before.st_size,
        device=before.st_dev,
        inode=before.st_ino,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
    )
    return snapshot, None if chunks is None else b"".join(chunks)


def _json_snapshot(path: Path, *, member: str) -> tuple[FileSnapshot, dict[str, Any], bytes]:
    snapshot, data = _snapshot_regular(path, member=member, collect=True)
    assert data is not None
    if len(data) > 32 * 1024 * 1024:
        raise RuntimeError(f"JSON control file is unexpectedly large: {member}")
    return snapshot, _decode_json_object(data, label=member), data


def _assert_unique_members(snapshots: Iterable[FileSnapshot], *, label: str) -> list[FileSnapshot]:
    result = sorted(snapshots, key=lambda item: item.member)
    members: set[str] = set()
    identities: set[tuple[int, int]] = set()
    folded: set[str] = set()
    for item in result:
        if item.member in members or item.member.casefold() in folded:
            raise RuntimeError(f"duplicate or case-fold-colliding {label} member")
        if item.identity in identities:
            raise RuntimeError(f"hardlink/alias reused by {label} members")
        members.add(item.member)
        folded.add(item.member.casefold())
        identities.add(item.identity)
    return result


def _assert_exact_tree(root: Path, *, files: set[str]) -> None:
    expected_directories = {
        parent.as_posix()
        for relative in files
        for parent in list(PurePosixPath(relative).parents)[:-1]
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    root_device = root.stat().st_dev
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"input-only tree contains symlink: {relative}")
        if info.st_dev != root_device:
            raise RuntimeError(f"input-only tree crosses device boundary: {relative}")
        if stat.S_ISDIR(info.st_mode):
            actual_directories.add(relative)
        elif stat.S_ISREG(info.st_mode):
            if info.st_nlink != 1:
                raise RuntimeError(f"input-only tree contains hardlink: {relative}")
            actual_files.add(relative)
        else:
            raise RuntimeError(f"input-only tree contains special entry: {relative}")
    if actual_files != files or actual_directories != expected_directories:
        raise RuntimeError(
            "input-only exact tree drift: "
            f"missing_files={sorted(files - actual_files)} "
            f"extra_files={sorted(actual_files - files)} "
            f"missing_dirs={sorted(expected_directories - actual_directories)} "
            f"extra_dirs={sorted(actual_directories - expected_directories)}"
        )


def _validate_protocol(config: Mapping[str, Any], *, config_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any], str]:
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("kind") != PROTOCOL_KIND
        or config.get("safe_for_submission") is not False
    ):
        raise RuntimeError("protocol identity or safety state drift")
    frozen = config.get("frozen_contract")
    pins = config.get("runtime_pins")
    policy = config.get("runtime_pin_mutation_policy")
    if not isinstance(frozen, dict) or not isinstance(pins, dict) or not isinstance(policy, dict):
        raise RuntimeError("protocol closure is missing")
    frozen_sha = _require_sha(config.get("frozen_contract_sha256"), label="frozen contract hash")
    if _sha256_bytes(_canonical_object_bytes(frozen)) != frozen_sha:
        raise RuntimeError("frozen contract hash mismatch")
    if config_bytes != (json.dumps(config, ensure_ascii=True, indent=2) + "\n").encode("utf-8"):
        raise RuntimeError("protocol config is not canonical repository JSON")
    code_pairs = policy.get("code_pin_fields")
    fixture_pairs = policy.get("fixture_pin_fields")
    if not isinstance(code_pairs, list) or not code_pairs or not isinstance(fixture_pairs, list) or not fixture_pairs:
        raise RuntimeError("runtime pin policy is incomplete")
    for group, label in ((code_pairs, "code"), (fixture_pairs, "fixture")):
        for pair in group:
            if not isinstance(pair, dict) or set(pair) != {"path_field", "sha256_field"}:
                raise RuntimeError(f"malformed {label} pin pair")
            path_field = pair["path_field"]
            sha_field = pair["sha256_field"]
            if path_field not in pins or sha_field not in pins:
                raise RuntimeError(f"missing {label} runtime pin")
            _relative_member(pins[path_field], label=f"runtime pin {path_field}")
            _require_sha(pins[sha_field], label=f"runtime pin {sha_field}")
    for key, value in pins.items():
        if key.endswith("_sha256"):
            _require_sha(value, label=f"runtime pin {key}")
    if pins.get("phase_a_must_not_start_until_all_non_path_values_are_non_null") is not True:
        raise RuntimeError("Phase-A full-pin guard drift")
    return pins, policy, frozen_sha


def _load_protocol(repo_root: Path, config_path: Path) -> tuple[FileSnapshot, dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    repo_root = _assert_directory_anchor(repo_root, label="repository root")
    config_path = _lexical_absolute(config_path)
    try:
        config_relative = config_path.relative_to(repo_root).as_posix()
    except ValueError as error:
        raise RuntimeError("protocol config is outside repository root") from error
    if config_relative != CONFIG_MEMBER:
        raise RuntimeError("protocol config relative path drift")
    config_source = _safe_child_file(repo_root, config_relative, label="protocol config")
    snapshot, config, raw = _json_snapshot(config_source, member=CONFIG_MEMBER)
    pins, policy, frozen_sha = _validate_protocol(config, config_bytes=raw)
    return snapshot, config, pins, policy, frozen_sha, snapshot.sha256


def _load_lifecycle(
    *,
    repo_root: Path,
    ledger_root: Path,
    config: Mapping[str, Any],
    pins: Mapping[str, Any],
    policy: Mapping[str, Any],
    frozen_sha: str,
    config_sha: str,
) -> list[FileSnapshot]:
    expected_relative = _relative_member(
        policy.get("transition_ledger_root"), label="transition ledger root"
    )
    expected_root = _lexical_absolute(repo_root / PurePosixPath(expected_relative))
    ledger_root = _assert_directory_anchor(ledger_root, label="lifecycle ledger root")
    if ledger_root != expected_root:
        raise RuntimeError("lifecycle ledger root differs from frozen policy")
    if set(path.name for path in ledger_root.iterdir()) != {
        "PREP.json", "SEALED.json", "PHASE_A.json", TRANSITION_DIRECTORY
    }:
        raise RuntimeError("lifecycle ledger must terminate exactly at PHASE_A")
    transition_root = _assert_directory_anchor(
        ledger_root / TRANSITION_DIRECTORY, label="transition receipt root"
    )
    if set(path.name for path in transition_root.iterdir()) != set(TRANSITION_NAMES):
        raise RuntimeError("transition receipt tree is not the exact four-file closure")

    snapshots: list[FileSnapshot] = []
    lifecycle_payloads: dict[str, dict[str, Any]] = {}
    lifecycle_hashes: dict[str, str] = {}
    previous: str | None = None
    lifecycle_keys = {
        "schema_version", "kind", "protocol_instance_id", "state",
        "frozen_contract_sha256", "config_sha256_or_null", "predecessor_sha256",
    }
    instance = config.get("protocol_instance_id")
    for state in LIFECYCLE_STATES:
        source = _safe_child_file(ledger_root, f"{state}.json", label=f"{state} lifecycle")
        member = f"lifecycle/{state}.json"
        snapshot, payload, raw = _json_snapshot(source, member=member)
        if raw != _canonical_json_bytes(payload) or set(payload) != lifecycle_keys:
            raise RuntimeError(f"non-canonical lifecycle schema: {state}")
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("kind") != "candidate_graph_oracle_lifecycle"
            or payload.get("protocol_instance_id") != instance
            or payload.get("state") != state
            or payload.get("frozen_contract_sha256") != frozen_sha
            or payload.get("predecessor_sha256") != previous
        ):
            raise RuntimeError(f"lifecycle chain drift: {state}")
        binding = _require_sha(payload.get("config_sha256_or_null"), label=f"{state} config binding")
        if state in ("SEALED", "PHASE_A") and binding != config_sha:
            raise RuntimeError(f"{state} does not bind the fully pinned config")
        lifecycle_payloads[state] = payload
        lifecycle_hashes[state] = snapshot.sha256
        previous = snapshot.sha256
        snapshots.append(snapshot)

    prep_config_sha = lifecycle_payloads["PREP"]["config_sha256_or_null"]
    intent_keys = {
        "schema_version", "kind", "stage", "stage_index", "protocol_instance_id",
        "frozen_contract_sha256", "config_relative_path", "previous_config_sha256",
        "intended_config_sha256", "pin_sha256_values", "created_utc",
    }
    completion_keys = {
        "schema_version", "kind", "stage", "stage_index", "protocol_instance_id",
        "frozen_contract_sha256", "config_relative_path", "previous_config_sha256",
        "final_config_sha256", "pin_sha256_values", "intent_sha256", "completed_utc",
    }
    previous_final: str | None = None
    for stage, index, prefix, pair_key, expected_final in (
        ("code", 0, "00_code_pins", "code_pin_fields", prep_config_sha),
        ("fixtures", 1, "01_fixtures_pins", "fixture_pin_fields", config_sha),
    ):
        loaded: dict[str, tuple[FileSnapshot, dict[str, Any], bytes]] = {}
        for suffix in ("intent", "complete"):
            name = f"{prefix}.{suffix}.json"
            source = _safe_child_file(transition_root, name, label=f"{stage} {suffix} receipt")
            member = f"lifecycle/{TRANSITION_DIRECTORY}/{name}"
            loaded[suffix] = _json_snapshot(source, member=member)
        intent_snapshot, intent, intent_raw = loaded["intent"]
        completion_snapshot, completion, completion_raw = loaded["complete"]
        if (
            set(intent) != intent_keys
            or set(completion) != completion_keys
            or intent_raw != _canonical_json_bytes(intent)
            or completion_raw != _canonical_json_bytes(completion)
        ):
            raise RuntimeError(f"{stage} transition receipt schema drift")
        common = {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "stage_index": index,
            "protocol_instance_id": instance,
            "frozen_contract_sha256": frozen_sha,
            "config_relative_path": CONFIG_MEMBER,
        }
        if intent.get("kind") != "candidate_graph_oracle_runtime_pin_transition_intent" or completion.get("kind") != "candidate_graph_oracle_runtime_pin_transition_completion":
            raise RuntimeError(f"{stage} transition receipt identity drift")
        if any(intent.get(key) != value or completion.get(key) != value for key, value in common.items()):
            raise RuntimeError(f"{stage} transition receipt binding drift")
        pairs = policy[pair_key]
        fields = [pair["sha256_field"] for pair in pairs]
        expected_values = {field: pins[field] for field in fields}
        if intent.get("pin_sha256_values") != expected_values or completion.get("pin_sha256_values") != expected_values:
            raise RuntimeError(f"{stage} transition pin map drift")
        if (
            completion.get("previous_config_sha256") != intent.get("previous_config_sha256")
            or completion.get("final_config_sha256") != intent.get("intended_config_sha256")
            or completion.get("intent_sha256") != intent_snapshot.sha256
            or completion.get("final_config_sha256") != expected_final
            or (previous_final is not None and intent.get("previous_config_sha256") != previous_final)
        ):
            raise RuntimeError(f"{stage} transition receipt chain mismatch")
        previous_final = completion["final_config_sha256"]
        snapshots.extend((intent_snapshot, completion_snapshot))
    return _assert_unique_members(snapshots, label="lifecycle")


def _module_literal_assignments(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    result: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        try:
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "Path" and len(node.value.args) == 1:
                result[name] = ast.literal_eval(node.value.args[0])
            else:
                result[name] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return result


def _assert_runner_mount_contract(snapshot: FileSnapshot) -> None:
    raw = snapshot.source.read_bytes()
    if _sha256_bytes(raw) != snapshot.sha256:
        raise RuntimeError("Phase-A runner changed before static contract check")
    source = raw.decode("utf-8")
    assignments = _module_literal_assignments(source)
    expected = {
        "OWNER": "pasha883",
        "CODE_SLUG": DATASET_BY_KEY["code"].slug.split("/", 1)[1],
        "INPUT_SLUG": DATASET_BY_KEY["input"].slug.split("/", 1)[1],
        "RUNTIME_SLUG": DATASET_BY_KEY["runtime"].slug.split("/", 1)[1],
        "CONFIG_RELATIVE": CONFIG_MEMBER,
    }
    if any(assignments.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Phase-A runner dataset/config constants differ from v2 packaging")
    tree = ast.parse(source)
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_assert_exact_code_mount"),
        None,
    )
    if function is None:
        raise RuntimeError("Phase-A runner lacks _assert_exact_code_mount")
    lifecycle_literal: set[str] | None = None
    expected_files_expression: str | None = None
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "lifecycle_files":
                try:
                    lifecycle_literal = set(ast.literal_eval(node.value))
                except (ValueError, TypeError):
                    pass
            elif node.targets[0].id == "expected_files":
                expected_files_expression = ast.unparse(node.value)
    expected_lifecycle = set(LIFECYCLE_MEMBERS)
    if lifecycle_literal != expected_lifecycle:
        raise RuntimeError("Phase-A runner lifecycle allowlist differs from packager")
    if expected_files_expression != "set(expected_hashes) | lifecycle_files":
        raise RuntimeError("Phase-A runner code-mount closure formula differs from packager")


def _code_snapshots(
    *,
    repo_root: Path,
    config_snapshot: FileSnapshot,
    config: Mapping[str, Any],
    pins: Mapping[str, Any],
    policy: Mapping[str, Any],
    lifecycle: Iterable[FileSnapshot],
) -> list[FileSnapshot]:
    snapshots = [config_snapshot, *lifecycle]
    expected_hashes: dict[str, str] = {CONFIG_MEMBER: config_snapshot.sha256}
    for pair in policy["code_pin_fields"]:
        relative = _relative_member(pins[pair["path_field"]], label=pair["path_field"])
        digest = _require_sha(pins[pair["sha256_field"]], label=pair["sha256_field"])
        prior = expected_hashes.setdefault(relative, digest)
        if prior != digest:
            raise RuntimeError("conflicting code hash for duplicate path")
    known = config.get("frozen_contract", {}).get("assets", {}).get("known_code_sha256")
    if not isinstance(known, dict) or not known:
        raise RuntimeError("frozen known-code closure is missing")
    for relative_value, digest_value in known.items():
        relative = _relative_member(relative_value, label="known code path")
        digest = _require_sha(digest_value, label=f"known code hash {relative}")
        prior = expected_hashes.setdefault(relative, digest)
        if prior != digest:
            raise RuntimeError("known-code path conflicts with runtime code pin")
    for relative, expected in sorted(expected_hashes.items()):
        if relative == CONFIG_MEMBER:
            continue
        source = _safe_child_file(repo_root, relative, label=f"code source {relative}")
        snapshot, _ = _snapshot_regular(source, member=relative)
        if snapshot.sha256 != expected:
            raise RuntimeError(f"pinned code hash mismatch: {relative}")
        snapshots.append(snapshot)
    result = _assert_unique_members(snapshots, label="code")
    expected_members = set(expected_hashes) | set(LIFECYCLE_MEMBERS)
    if {item.member for item in result} != expected_members:
        raise RuntimeError("code archive tree differs from Phase-A exact mount closure")
    runner_relative = pins["phase_a_runner_path"]
    runner = next(item for item in result if item.member == runner_relative)
    _assert_runner_mount_contract(runner)
    return result


def _reject_forbidden_input_metadata(value: Any, *, location: str = "root") -> None:
    forbidden = ("source", "panel", "target", "label", "secret", "shuffle", "permutation")
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(token in str(key).lower() for token in forbidden):
                raise RuntimeError(f"forbidden input-only metadata key at {location}")
            _reject_forbidden_input_metadata(nested, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_input_metadata(nested, location=f"{location}[{index}]")


def _input_snapshots(
    *,
    fixture_input_root: Path,
    config: Mapping[str, Any],
    pins: Mapping[str, Any],
    policy: Mapping[str, Any],
    frozen_sha: str,
) -> list[FileSnapshot]:
    root = _assert_directory_anchor(fixture_input_root, label="fixture input root")
    pinned_relative = _relative_member(
        pins.get("fixture_input_manifest_relative_path"), label="input manifest runtime pin"
    )
    if pinned_relative != f"fixture_input/{INPUT_MANIFEST_NAME}":
        raise RuntimeError("input manifest runtime path drift")
    manifest_source = _safe_child_file(root, INPUT_MANIFEST_NAME, label="input manifest")
    manifest_snapshot, manifest, raw = _json_snapshot(
        manifest_source, member=INPUT_MANIFEST_NAME
    )
    if manifest_snapshot.sha256 != pins.get("fixture_input_manifest_sha256"):
        raise RuntimeError("input manifest hash differs from runtime pin")
    if raw != _canonical_json_bytes(manifest):
        raise RuntimeError("input manifest is not canonical JSON")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != "candidate_graph_oracle_fixture_inputs"
        or manifest.get("protocol_instance_id") != config.get("protocol_instance_id")
        or manifest.get("frozen_contract_sha256") != frozen_sha
        or manifest.get("canonical_record_order") != "ascending opaque_id"
        or manifest.get("allowed_record_metadata") != ["opaque_id", "artifact", "arrays"]
    ):
        raise RuntimeError("input manifest protocol/schema binding drift")
    for pair in policy["code_pin_fields"]:
        sha_field = pair["sha256_field"]
        if manifest.get(sha_field) != pins.get(sha_field):
            raise RuntimeError(f"input manifest code provenance drift: {sha_field}")
    _reject_forbidden_input_metadata(manifest)
    records = manifest.get("records")
    expected_count = config.get("frozen_contract", {}).get("source_selection", {}).get("total_fixture_records")
    if not isinstance(records, list) or manifest.get("record_count") != expected_count or len(records) != expected_count:
        raise RuntimeError("input manifest record coverage drift")
    snapshots = [manifest_snapshot]
    expected_files = {INPUT_MANIFEST_NAME}
    opaque_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"opaque_id", "artifact", "arrays"}:
            raise RuntimeError(f"input record schema drift at {index}")
        opaque_id = record.get("opaque_id")
        if not isinstance(opaque_id, str) or OPAQUE_ID_RE.fullmatch(opaque_id) is None:
            raise RuntimeError("invalid opaque input id")
        opaque_ids.append(opaque_id)
        artifact = record.get("artifact")
        if not isinstance(artifact, dict) or set(artifact) != {"path", "bytes", "sha256"}:
            raise RuntimeError("invalid input artifact descriptor")
        relative = _relative_member(artifact.get("path"), label="input artifact path")
        if relative != f"records/{opaque_id}.npz":
            raise RuntimeError("input artifact path is not opaque-id canonical")
        lowered = relative.lower()
        if any(token in lowered for token in ("label", "target", "secret", "truth")):
            raise RuntimeError("forbidden input-only artifact path")
        source = _safe_child_file(root, relative, label="input artifact")
        snapshot, _ = _snapshot_regular(source, member=relative)
        if snapshot.sha256 != _require_sha(artifact.get("sha256"), label="input artifact hash") or snapshot.size != artifact.get("bytes"):
            raise RuntimeError("input artifact bytes/hash mismatch")
        expected_files.add(relative)
        snapshots.append(snapshot)
    if opaque_ids != sorted(opaque_ids) or len(set(opaque_ids)) != expected_count:
        raise RuntimeError("opaque input ids are not a unique canonical order")
    if manifest.get("opaque_ids_sha256") != _sha256_bytes("\n".join(opaque_ids).encode("ascii")):
        raise RuntimeError("opaque id-list hash mismatch")
    _assert_exact_tree(root, files=expected_files)
    return _assert_unique_members(snapshots, label="input")


def _runtime_snapshots(*, repo_root: Path, config: Mapping[str, Any]) -> list[FileSnapshot]:
    assets = config.get("frozen_contract", {}).get("assets")
    if not isinstance(assets, dict):
        raise RuntimeError("runtime asset closure is missing")
    snapshots: list[FileSnapshot] = []
    for key in ("denoiser", "hbt"):
        asset = assets.get(key)
        if not isinstance(asset, dict):
            raise RuntimeError(f"missing runtime asset: {key}")
        relative = _relative_member(asset.get("path"), label=f"{key} checkpoint path")
        expected = _require_sha(asset.get("sha256"), label=f"{key} checkpoint hash")
        source = _safe_child_file(repo_root, relative, label=f"{key} checkpoint")
        member = PurePosixPath(relative).name
        snapshot, _ = _snapshot_regular(source, member=member)
        if snapshot.sha256 != expected:
            raise RuntimeError(f"runtime checkpoint hash mismatch: {key}")
        snapshots.append(snapshot)
    result = _assert_unique_members(snapshots, label="runtime")
    if len(result) != 2 or any("/" in item.member for item in result):
        raise RuntimeError("runtime archive must contain exactly two direct children")
    return result


def _zip_info(member: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(member, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.extra = b""
    info.comment = b""
    return info


def _write_member(archive: zipfile.ZipFile, snapshot: FileSnapshot) -> None:
    descriptor = os.open(snapshot.source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (snapshot.device, snapshot.inode, snapshot.size, snapshot.mtime_ns, snapshot.ctime_ns)
        ):
            raise RuntimeError(f"archive source changed before copy: {snapshot.member}")
        with archive.open(_zip_info(snapshot.member), "w", force_zip64=True) as output:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                output.write(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            != (snapshot.device, snapshot.inode, snapshot.size, snapshot.mtime_ns, snapshot.ctime_ns)
            or digest.hexdigest() != snapshot.sha256
        ):
            raise RuntimeError(f"archive source changed during copy: {snapshot.member}")
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> tuple[str, int]:
    snapshot, _ = _snapshot_regular(path, member=path.name)
    return snapshot.sha256, snapshot.size


def _build_archive(path: Path, snapshots: Sequence[FileSnapshot]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"archive output is not fresh: {path.name}")
    with zipfile.ZipFile(path, "x", allowZip64=True) as archive:
        for snapshot in snapshots:
            _write_member(archive, snapshot)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    expected = {snapshot.member: snapshot for snapshot in snapshots}
    with zipfile.ZipFile(path, "r") as archive:
        if archive.testzip() is not None or archive.namelist() != sorted(expected):
            raise RuntimeError("deterministic ZIP member order/CRC mismatch")
        for info in archive.infolist():
            snapshot = expected[info.filename]
            if (
                info.date_time != ZIP_TIMESTAMP
                or info.compress_type != zipfile.ZIP_STORED
                or info.file_size != snapshot.size
                or ((info.external_attr >> 16) & 0o777) != 0o644
                or _sha256_bytes(archive.read(info.filename)) != snapshot.sha256
            ):
                raise RuntimeError(f"ZIP member verification failed: {info.filename}")
    archive_sha, archive_size = _sha256_file(path)
    return {
        "path": path.name,
        "sha256": archive_sha,
        "bytes": archive_size,
        "compression": "ZIP_STORED",
        "timestamp": "1980-01-01T00:00:00",
        "members": [
            {"path": item.member, "sha256": item.sha256, "bytes": item.size}
            for item in snapshots
        ],
    }


def _dataset_metadata(spec: DatasetSpec) -> dict[str, Any]:
    return {
        "id": spec.slug,
        "isPrivate": True,
        "licenses": [{"name": "other"}],
        "title": spec.title,
    }


def _write_exclusive(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise RuntimeError("short write to package output")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_bundles(
    *,
    repo_root: Path,
    config_path: Path,
    lifecycle_ledger_root: Path,
    fixture_input_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build all three byte-reproducible v2 payload archives and receipt."""

    repo_root = _assert_directory_anchor(repo_root, label="repository root")
    output_root = _lexical_absolute(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("output root must be fresh")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    config_snapshot, config, pins, policy, frozen_sha, config_sha = _load_protocol(
        repo_root, config_path
    )
    lifecycle = _load_lifecycle(
        repo_root=repo_root,
        ledger_root=lifecycle_ledger_root,
        config=config,
        pins=pins,
        policy=policy,
        frozen_sha=frozen_sha,
        config_sha=config_sha,
    )
    code = _code_snapshots(
        repo_root=repo_root,
        config_snapshot=config_snapshot,
        config=config,
        pins=pins,
        policy=policy,
        lifecycle=lifecycle,
    )
    opaque_input = _input_snapshots(
        fixture_input_root=fixture_input_root,
        config=config,
        pins=pins,
        policy=policy,
        frozen_sha=frozen_sha,
    )
    runtime = _runtime_snapshots(repo_root=repo_root, config=config)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.build-", dir=output_root.parent)
    )
    try:
        archives_dir = staging / "archives"
        metadata_dir = staging / "dataset_metadata"
        archives_dir.mkdir()
        metadata_dir.mkdir()
        payloads = {"code": code, "input": opaque_input, "runtime": runtime}
        datasets: dict[str, Any] = {}
        for spec in DATASETS:
            archive = _build_archive(archives_dir / spec.archive_name, payloads[spec.key])
            sidecar_dir = metadata_dir / spec.key
            sidecar_dir.mkdir()
            metadata_path = sidecar_dir / "dataset-metadata.json"
            metadata_bytes = _canonical_json_bytes(_dataset_metadata(spec))
            _write_exclusive(metadata_path, metadata_bytes)
            metadata_sha, metadata_size = _sha256_file(metadata_path)
            datasets[spec.key] = {
                "slug": spec.slug,
                "expected_version": 2,
                "must_remain_private": True,
                "archive": {**archive, "path": f"archives/{archive['path']}"},
                "dataset_metadata": {
                    "path": f"dataset_metadata/{spec.key}/dataset-metadata.json",
                    "sha256": metadata_sha,
                    "bytes": metadata_size,
                    "excluded_from_mounted_payload": True,
                },
            }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "candidate_graph_oracle_kaggle_dataset_v2_build_receipt",
            "protocol_instance_id": config.get("protocol_instance_id"),
            "frozen_contract_sha256": frozen_sha,
            "fully_pinned_config_sha256": config_sha,
            "lifecycle_terminal_state": "PHASE_A",
            "lifecycle_member_sha256": {
                item.member: item.sha256 for item in lifecycle
            },
            "datasets": datasets,
            "upload_performed": False,
            "input_payload_decoded": False,
            "safe_for_submission": False,
        }
        envelope = {
            "payload": payload,
            "payload_sha256": _sha256_bytes(_canonical_object_bytes(payload)),
        }
        receipt_path = staging / RECEIPT_NAME
        _write_exclusive(receipt_path, _canonical_json_bytes(envelope))
        receipt_sha, receipt_size = _sha256_file(receipt_path)
        os.rename(staging, output_root)
        parent_fd = os.open(output_root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return {
            "status": "candidate_graph_oracle_dataset_v2_bundles_built",
            "output_root": os.fspath(output_root),
            "receipt": RECEIPT_NAME,
            "receipt_sha256": receipt_sha,
            "receipt_bytes": receipt_size,
            "config_sha256": config_sha,
            "dataset_archive_sha256": {
                key: value["archive"]["sha256"] for key, value in datasets.items()
            },
            "upload_performed": False,
            "safe_for_submission": False,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lifecycle-ledger-root", type=Path, required=True)
    parser.add_argument("--fixture-input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_bundles(
        repo_root=args.repo_root,
        config_path=args.config,
        lifecycle_ledger_root=args.lifecycle_ledger_root,
        fixture_input_root=args.fixture_input_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
