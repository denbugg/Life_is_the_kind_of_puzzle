#!/usr/bin/env python3
"""Validate or materialize the v4 source/config verifier closure.

Pre-reservation validation accepts only an all-null code-pin state and never
creates output.  Materialization requires every code pin to be populated and
verified.  Fixture and label paths are outside this utility's read authority.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence


CONFIG_RELATIVE = "configs/candidate_graph_oracle_ceiling_v4.json"
INSTANCE = "6c0fe4e8524ce39d830d9a5bee118d8b"
FROZEN_CONTRACT_SHA256 = (
    "2070c1b4ff0a3ff42c5ffdd6d611c214c02dbb99b2c51a88f38a862bb1f8a05c"
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INSTANCE_RE = re.compile(r"^[0-9a-f]{32}$")
FORBIDDEN_PATH_TOKENS = ("label", "target", "secret")

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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_regular(path: Path) -> tuple[bytes, os.stat_result]:
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
    return b"".join(chunks), info


def _load_json(path: Path) -> tuple[dict[str, Any], bytes, str]:
    raw, _ = _read_regular(path)
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


def _relative(value: Any, *, label: str, forbid_sensitive: bool = True) -> str:
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
    if forbid_sensitive:
        _require(
            not any(token in value.lower() for token in FORBIDDEN_PATH_TOKENS),
            f"{label} is outside the source-only authority",
        )
    return value


def _source_path(source_root: Path, relative: str, *, label: str) -> Path:
    """Reject symlinks in every component before opening a source file."""

    parts = PurePosixPath(relative).parts
    current = source_root
    for part in parts[:-1]:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError as error:
            raise RuntimeError(f"missing source directory for {label}: {current}") from error
        _require(
            stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            f"symlink or non-directory in source path for {label}: {current}",
        )
    return source_root / relative


def _classify_code_pins(pins: Mapping[str, Any]) -> str:
    values = [pins.get(sha_field) for _, sha_field, _ in CODE_PIN_PAIRS]
    nulls = [value is None for value in values]
    if all(nulls):
        return "null"
    _require(not any(nulls), "partial code-pin transition is forbidden")
    _require(
        all(isinstance(value, str) and SHA_RE.fullmatch(value) for value in values),
        "malformed code SHA pin",
    )
    return "pinned"


def _source_closure(
    source_root: Path, config_relative: str
) -> tuple[dict[str, Any], str, str, dict[str, dict[str, Any]], str]:
    config_relative = _relative(
        config_relative, label="config path", forbid_sensitive=True
    )
    _require(config_relative == CONFIG_RELATIVE, "v4 config path drift")
    config_path = _source_path(source_root, config_relative, label="config")
    config, config_raw, config_sha = _load_json(config_path)
    _require(
        config_raw
        == (json.dumps(config, ensure_ascii=True, indent=2) + "\n").encode("utf-8"),
        "config is not canonical repository JSON",
    )
    _require(
        config.get("schema_version") == 1
        and config.get("kind") == "candidate_graph_oracle_ceiling",
        "wrong v4 protocol schema",
    )
    instance = config.get("protocol_instance_id")
    _require(
        instance == INSTANCE and INSTANCE_RE.fullmatch(instance) is not None,
        "v4 protocol instance drift",
    )
    _require(config.get("safe_for_submission") is False, "protocol became submission-safe")
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
    expected_policy = [
        {"path_field": path_field, "sha256_field": sha_field}
        for path_field, sha_field, _ in CODE_PIN_PAIRS
    ]
    _require(policy.get("code_pin_fields") == expected_policy, "code pin policy drift")
    pin_state = _classify_code_pins(pins)

    closure: dict[str, dict[str, Any]] = {}
    identities: dict[tuple[int, int], str] = {}

    def bind(relative: str, *, expected_sha: str | None, role: str) -> None:
        relative = _relative(relative, label=f"{role} path", forbid_sensitive=True)
        raw, info = _read_regular(
            _source_path(source_root, relative, label=role)
        )
        identity = (info.st_dev, info.st_ino)
        _require(identity not in identities, f"source hardlink alias: {relative}")
        identities[identity] = relative
        actual_sha = hashlib.sha256(raw).hexdigest()
        if expected_sha is not None:
            _require(
                isinstance(expected_sha, str)
                and SHA_RE.fullmatch(expected_sha) is not None
                and actual_sha == expected_sha,
                f"source SHA drift: {relative}",
            )
        closure[relative] = {
            "role": role,
            "sha256": actual_sha,
            "bytes": len(raw),
        }

    for path_field, sha_field, expected_path in CODE_PIN_PAIRS:
        _require(pins.get(path_field) == expected_path, f"runtime path drift: {path_field}")
        bind(
            expected_path,
            expected_sha=pins.get(sha_field) if pin_state == "pinned" else None,
            role=sha_field,
        )

    known_code = frozen.get("assets", {}).get("known_code_sha256")
    _require(isinstance(known_code, dict) and known_code, "known-code closure missing")
    for relative, expected_sha in sorted(known_code.items()):
        if relative in closure:
            _require(
                closure[relative]["sha256"] == expected_sha,
                f"known-code overlap SHA drift: {relative}",
            )
            continue
        bind(relative, expected_sha=expected_sha, role="known_code_sha256")

    # The copied config is part of the closure, but it is already open and may
    # not be passed through bind() because duplicate-inode detection is strict.
    closure[config_relative] = {
        "role": "protocol_config",
        "sha256": config_sha,
        "bytes": len(config_raw),
    }
    return config, config_sha, frozen_sha, dict(sorted(closure.items())), pin_state


def validate_source_closure(
    source_root: Path, *, config_relative: str = CONFIG_RELATIVE
) -> dict[str, Any]:
    """Validate source/config state without writing or touching fixture paths."""

    source_root = source_root.expanduser().resolve(strict=True)
    _, config_sha, frozen_sha, closure, pin_state = _source_closure(
        source_root, config_relative
    )
    return {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v4_source_config_closure_validation",
        "status": (
            "valid_pre_reservation_source_closure"
            if pin_state == "null"
            else "valid_pinned_source_closure"
        ),
        "protocol_instance_id": INSTANCE,
        "frozen_contract_sha256": frozen_sha,
        "config_sha256": config_sha,
        "code_pin_state": pin_state,
        "source_files": closure,
        "source_file_count": len(closure),
        "files_written": 0,
        "fixture_paths_constructed": False,
        "fixture_files_opened": False,
        "label_paths_constructed": False,
        "label_files_opened": False,
        "safe_for_submission": False,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, f"short write: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _tree_files(root: Path) -> set[str]:
    result: set[str] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            _require(not child.is_symlink(), f"symlink directory in materialized tree: {child}")
        for name in files:
            child = directory_path / name
            _require(not child.is_symlink(), f"symlink file in materialized tree: {child}")
            result.add(child.relative_to(root).as_posix())
    return result


def materialize(
    *,
    source_root: Path,
    destination: Path,
    receipt: Path,
    config_relative: str = CONFIG_RELATIVE,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve(strict=True)
    destination = destination.expanduser().absolute()
    receipt = receipt.expanduser().absolute()
    _require(not destination.exists(), "destination already exists")
    _require(not receipt.exists(), "receipt already exists")
    _, config_sha, frozen_sha, closure, pin_state = _source_closure(
        source_root, config_relative
    )
    _require(pin_state == "pinned", "materialization requires all code pins populated")

    destination.mkdir(parents=True, exist_ok=False)
    for relative, descriptor in closure.items():
        raw, _ = _read_regular(
            _source_path(source_root, relative, label=str(descriptor["role"]))
        )
        _require(
            hashlib.sha256(raw).hexdigest() == descriptor["sha256"],
            f"source changed during materialization: {relative}",
        )
        _write_exclusive(destination / relative, raw)
    _require(
        _tree_files(destination) == set(closure),
        "materialized repository file closure drift",
    )
    for relative, descriptor in closure.items():
        raw, _ = _read_regular(destination / relative)
        _require(
            hashlib.sha256(raw).hexdigest() == descriptor["sha256"],
            f"materialized SHA drift: {relative}",
        )

    payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v4_bound_verifier_source_closure",
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "protocol_instance_id": INSTANCE,
        "frozen_contract_sha256": frozen_sha,
        "config_sha256": config_sha,
        "source_root": str(source_root),
        "bound_repository": str(destination),
        "files": closure,
        "file_count": len(closure),
        "current_source_modules_copied": True,
        "fixture_paths_constructed": False,
        "fixture_files_opened": False,
        "label_paths_constructed": False,
        "label_files_opened": False,
        "safe_for_submission": False,
    }
    envelope = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical_object(payload)).hexdigest(),
    }
    encoded = _canonical_file(envelope)
    _write_exclusive(receipt, encoded)
    return {
        "status": "materialized_and_verified_v4_source_closure",
        "bound_repository": str(destination),
        "closure_receipt": str(receipt),
        "closure_receipt_sha256": hashlib.sha256(encoded).hexdigest(),
        "total_files": len(closure),
        "label_paths_constructed_or_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config-relative", default=CONFIG_RELATIVE)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.validate_only:
        _require(
            args.destination is None and args.receipt is None,
            "validate-only forbids destination and receipt",
        )
        result = validate_source_closure(
            args.source_root, config_relative=args.config_relative
        )
    else:
        _require(
            args.destination is not None and args.receipt is not None,
            "materialization requires destination and receipt",
        )
        result = materialize(
            source_root=args.source_root,
            destination=args.destination,
            receipt=args.receipt,
            config_relative=args.config_relative,
        )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
