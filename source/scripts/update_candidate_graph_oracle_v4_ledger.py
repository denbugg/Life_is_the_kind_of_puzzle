#!/usr/bin/env python3
"""Advance the irreversible candidate-graph oracle v4 lifecycle ledger.

The four files are exact, append-only O_EXCL claims stored outside every
fixture, job, and output root. Their schema is frozen by the v4 protocol
configuration supplied on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
PROTOCOL_KIND = "candidate_graph_oracle_ceiling"
LEDGER_KIND = "candidate_graph_oracle_lifecycle"
TRANSITION_INTENT_KIND = "candidate_graph_oracle_runtime_pin_transition_intent"
TRANSITION_COMPLETION_KIND = (
    "candidate_graph_oracle_runtime_pin_transition_completion"
)
TRANSITION_DIRECTORY = "runtime_pin_transitions"
STATES = ("PREP", "SEALED", "PHASE_A", "LABEL_ACCESS")
STATE_INDEX = {state: index for index, state in enumerate(STATES)}
INSTANCE_RE = re.compile(r"^[0-9a-f]{32}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
EXACT_PAYLOAD_KEYS = {
    "schema_version",
    "kind",
    "protocol_instance_id",
    "state",
    "frozen_contract_sha256",
    "config_sha256_or_null",
    "predecessor_sha256",
}
TRANSITION_INTENT_KEYS = {
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
}
TRANSITION_COMPLETION_KEYS = {
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
}


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _canonical_object_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _secure_bytes(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"not an unlinked regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_secure_bytes(path)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_secure_bytes(path).decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _load_canonical_transition_json(
    path: Path, *, expected_keys: set[str]
) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise RuntimeError(f"transition receipt may not be a symlink: {path.name}")
    raw = _secure_bytes(path)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RuntimeError(f"transition receipt schema drift: {path.name}")
    if raw != _canonical_bytes(payload):
        raise RuntimeError(f"transition receipt is not canonical: {path.name}")
    return payload, hashlib.sha256(raw).hexdigest()


def _transition_pin_fields(
    protocol: Mapping[str, Any], *, stage: str
) -> tuple[str, ...]:
    policy = protocol.get("runtime_pin_mutation_policy")
    if not isinstance(policy, dict):
        raise RuntimeError("runtime pin policy is missing")
    policy_field = "code_pin_fields" if stage == "code" else "fixture_pin_fields"
    pairs = policy.get(policy_field)
    if not isinstance(pairs, list) or not pairs:
        raise RuntimeError(f"{stage} transition pin closure is missing")
    fields: list[str] = []
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"path_field", "sha256_field"}:
            raise RuntimeError(f"{stage} transition pin pair schema drift")
        field = pair.get("sha256_field")
        if not isinstance(field, str) or not field:
            raise RuntimeError(f"{stage} transition SHA field is invalid")
        fields.append(field)
    if len(fields) != len(set(fields)):
        raise RuntimeError(f"duplicate {stage} transition SHA field")
    return tuple(fields)


def _validate_transition_stage(
    *,
    transition_root: Path,
    protocol: Mapping[str, Any],
    stage: str,
    stage_index: int,
    instance: str,
    frozen_hash: str,
    config_relative: str,
) -> tuple[dict[str, Any], str]:
    prefix = f"{stage_index:02d}_{stage}_pins"
    intent_path = transition_root / f"{prefix}.intent.json"
    completion_path = transition_root / f"{prefix}.complete.json"
    if not intent_path.is_file() or not completion_path.is_file():
        raise RuntimeError(f"{stage} pin transition completion is missing")
    intent, intent_sha = _load_canonical_transition_json(
        intent_path, expected_keys=TRANSITION_INTENT_KEYS
    )
    completion, _ = _load_canonical_transition_json(
        completion_path, expected_keys=TRANSITION_COMPLETION_KEYS
    )
    common = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "stage_index": stage_index,
        "protocol_instance_id": instance,
        "frozen_contract_sha256": frozen_hash,
        "config_relative_path": config_relative,
    }
    for payload, kind, label in (
        (intent, TRANSITION_INTENT_KIND, "intent"),
        (completion, TRANSITION_COMPLETION_KIND, "completion"),
    ):
        if payload.get("kind") != kind:
            raise RuntimeError(f"{stage} transition {label} kind mismatch")
        for key, value in common.items():
            if payload.get(key) != value:
                raise RuntimeError(f"{stage} transition {label} mismatch: {key}")
    pin_fields = _transition_pin_fields(protocol, stage=stage)
    pins = protocol.get("runtime_pins")
    if not isinstance(pins, dict):
        raise RuntimeError("runtime pins are missing")
    expected_pin_values = {field: pins.get(field) for field in pin_fields}
    for field, value in expected_pin_values.items():
        if not isinstance(value, str) or not SHA_RE.fullmatch(value):
            raise RuntimeError(f"current {stage} pin is malformed: {field}")
    for payload, label in ((intent, "intent"), (completion, "completion")):
        values = payload.get("pin_sha256_values")
        if not isinstance(values, dict) or set(values) != set(pin_fields):
            raise RuntimeError(f"{stage} transition {label} pin closure mismatch")
        if values != expected_pin_values:
            raise RuntimeError(f"{stage} transition {label} pin values mismatch")
    previous = intent.get("previous_config_sha256")
    intended = intent.get("intended_config_sha256")
    if not isinstance(previous, str) or not SHA_RE.fullmatch(previous):
        raise RuntimeError(f"{stage} transition previous config hash is invalid")
    if not isinstance(intended, str) or not SHA_RE.fullmatch(intended):
        raise RuntimeError(f"{stage} transition intended config hash is invalid")
    if (
        completion.get("previous_config_sha256") != previous
        or completion.get("final_config_sha256") != intended
        or completion.get("intent_sha256") != intent_sha
    ):
        raise RuntimeError(f"{stage} transition completion does not match intent")
    return completion, intended


def _verify_transition_receipts(
    *,
    protocol: Mapping[str, Any],
    config_path: Path,
    ledger_root: Path,
    instance: str,
    frozen_hash: str,
    config_hash: str,
    fixtures_pinned: bool,
) -> None:
    transition_root = ledger_root / TRANSITION_DIRECTORY
    if transition_root.is_symlink() or not transition_root.is_dir():
        raise RuntimeError("runtime pin transition ledger is missing or invalid")
    try:
        config_relative = config_path.resolve(strict=True).relative_to(
            config_path.resolve(strict=True).parent.parent
        ).as_posix()
    except ValueError as error:
        raise RuntimeError("protocol config is not contained by its repository") from error

    code_names = {
        "00_code_pins.intent.json",
        "00_code_pins.complete.json",
    }
    fixture_names = {
        "01_fixtures_pins.intent.json",
        "01_fixtures_pins.complete.json",
    }
    actual_names = {path.name for path in transition_root.iterdir()}
    expected_names = code_names | (fixture_names if fixtures_pinned else set())
    if actual_names != expected_names:
        raise RuntimeError(
            "runtime pin transition ledger closure mismatch: "
            f"expected {sorted(expected_names)}, got {sorted(actual_names)}"
        )
    code_completion, code_final = _validate_transition_stage(
        transition_root=transition_root,
        protocol=protocol,
        stage="code",
        stage_index=0,
        instance=instance,
        frozen_hash=frozen_hash,
        config_relative=config_relative,
    )
    if not fixtures_pinned:
        if code_final != config_hash:
            raise RuntimeError(
                "current pre-fixture config SHA256 differs from code-pin completion"
            )
        return
    fixture_completion, fixture_final = _validate_transition_stage(
        transition_root=transition_root,
        protocol=protocol,
        stage="fixtures",
        stage_index=1,
        instance=instance,
        frozen_hash=frozen_hash,
        config_relative=config_relative,
    )
    if fixture_completion.get("previous_config_sha256") != code_completion.get(
        "final_config_sha256"
    ):
        raise RuntimeError("fixture transition is not chained to code completion")
    if fixture_final != config_hash:
        raise RuntimeError(
            "current final config SHA256 differs from fixture-pin completion"
        )


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    if set(payload) != EXACT_PAYLOAD_KEYS:
        raise RuntimeError("lifecycle payload schema drift")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        encoded = _canonical_bytes(payload)
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("lifecycle claim fstat validation failed")
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _load_protocol(config_path: Path) -> tuple[dict[str, Any], str, str, str]:
    config_path = config_path.expanduser().resolve(strict=True)
    if config_path.is_symlink():
        raise RuntimeError("protocol config may not be a symlink")
    raw = _secure_bytes(config_path)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("protocol must be a JSON object")
    if payload.get("kind") != PROTOCOL_KIND or payload.get("schema_version") != 1:
        raise RuntimeError("wrong candidate-graph protocol")
    contract = payload.get("frozen_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("protocol has no frozen_contract")
    frozen_hash = _canonical_object_sha256(contract)
    if frozen_hash != payload.get("frozen_contract_sha256"):
        raise RuntimeError("frozen contract hash mismatch")
    instance = payload.get("protocol_instance_id")
    if not isinstance(instance, str) or not INSTANCE_RE.fullmatch(instance):
        raise RuntimeError("protocol_instance_id must be 32 lowercase hex")
    return payload, instance, frozen_hash, hashlib.sha256(raw).hexdigest()


def _pin_state(protocol: Mapping[str, Any]) -> tuple[bool, bool]:
    pins = protocol.get("runtime_pins")
    policy = protocol.get("runtime_pin_mutation_policy")
    if not isinstance(pins, dict) or not isinstance(policy, dict):
        raise RuntimeError("runtime pin policy is missing")
    code_pairs = policy.get("code_pin_fields")
    fixture_pairs = policy.get("fixture_pin_fields")
    if not isinstance(code_pairs, list) or not isinstance(fixture_pairs, list):
        raise RuntimeError("runtime pin field lists are missing")

    def all_pinned(pairs: list[dict[str, Any]]) -> bool:
        states: list[bool] = []
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != {"path_field", "sha256_field"}:
                raise RuntimeError("runtime pin pair schema drift")
            path_field = pair["path_field"]
            sha_field = pair["sha256_field"]
            if not isinstance(pins.get(path_field), str) or not pins[path_field]:
                raise RuntimeError(f"runtime pin path is invalid: {path_field}")
            value = pins.get(sha_field)
            if value is None:
                states.append(False)
            elif not isinstance(value, str) or not SHA_RE.fullmatch(value):
                raise RuntimeError(f"runtime SHA pin is malformed: {sha_field}")
            else:
                states.append(True)
        if any(states) and not all(states):
            raise RuntimeError("partial runtime pin transition is forbidden")
        return bool(states) and all(states)

    return all_pinned(code_pairs), all_pinned(fixture_pairs)


def _allowed_ledger_entries(root: Path) -> set[str]:
    allowed = {f"{state}.json" for state in STATES}
    # The pin-finalizer owns this explicitly separate append-only subdirectory.
    allowed.add(TRANSITION_DIRECTORY)
    return allowed


def _existing_states(root: Path, *, instance: str, frozen_hash: str) -> list[str]:
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("lifecycle ledger root is invalid")
    names = {path.name for path in root.iterdir()}
    extras = names - _allowed_ledger_entries(root)
    if extras:
        raise RuntimeError(f"unrecognized lifecycle ledger entries: {sorted(extras)}")
    found: list[str] = []
    previous_hash: str | None = None
    for state in STATES:
        path = root / f"{state}.json"
        if not path.exists():
            break
        if path.is_symlink():
            raise RuntimeError("lifecycle state may not be a symlink")
        raw = _secure_bytes(path)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != EXACT_PAYLOAD_KEYS:
            raise RuntimeError(f"lifecycle payload schema drift: {state}")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "kind": LEDGER_KIND,
            "protocol_instance_id": instance,
            "state": state,
            "frozen_contract_sha256": frozen_hash,
            "predecessor_sha256": previous_hash,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"lifecycle chain mismatch: {state}.{key}")
        config_hash = payload.get("config_sha256_or_null")
        if not isinstance(config_hash, str) or not SHA_RE.fullmatch(config_hash):
            raise RuntimeError(f"lifecycle config hash is invalid: {state}")
        if raw != _canonical_bytes(payload):
            raise RuntimeError(f"lifecycle JSON is not canonical: {state}")
        found.append(state)
        previous_hash = hashlib.sha256(raw).hexdigest()
    for state in STATES[len(found) :]:
        if (root / f"{state}.json").exists():
            raise RuntimeError("lifecycle states are not a strict prefix")
    return found


def advance_state(
    *,
    config_path: Path,
    ledger_root: Path,
    state: str,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    if state not in STATE_INDEX:
        raise RuntimeError(f"state must be one of {STATES}")
    protocol, instance, frozen_hash, config_hash = _load_protocol(config_path)
    if expected_config_sha256 is not None and expected_config_sha256 != config_hash:
        raise RuntimeError("out-of-band whole-config SHA256 mismatch")
    code_pinned, fixtures_pinned = _pin_state(protocol)
    if not code_pinned:
        raise RuntimeError(f"{state} requires every code/environment/runner pin")
    if state == "PREP" and fixtures_pinned:
        raise RuntimeError("PREP requires fixture SHA pins to remain null")
    if state != "PREP" and not fixtures_pinned:
        raise RuntimeError(f"{state} requires every fixture SHA pin")

    policy = protocol["runtime_pin_mutation_policy"]
    configured_root = Path(str(policy["transition_ledger_root"]))
    repository = Path(config_path).expanduser().resolve(strict=True).parent.parent
    expected_root = (repository / configured_root).absolute()
    supplied_root = ledger_root.expanduser().absolute()
    if supplied_root != expected_root:
        raise RuntimeError("lifecycle ledger root differs from immutable protocol path")
    if supplied_root.is_symlink():
        raise RuntimeError("lifecycle ledger root may not be a symlink")
    if not supplied_root.is_dir():
        raise RuntimeError("lifecycle ledger root must be created by pin finalization")
    os.chmod(supplied_root, 0o700)
    _fsync_directory(supplied_root)

    _verify_transition_receipts(
        protocol=protocol,
        config_path=config_path,
        ledger_root=supplied_root,
        instance=instance,
        frozen_hash=frozen_hash,
        config_hash=config_hash,
        fixtures_pinned=fixtures_pinned,
    )

    existing = _existing_states(
        supplied_root, instance=instance, frozen_hash=frozen_hash
    )
    expected_existing = list(STATES[: STATE_INDEX[state]])
    if existing != expected_existing:
        if state == "PREP" and existing:
            raise RuntimeError("protocol instance has already been consumed")
        raise RuntimeError(
            f"cannot claim {state}; lifecycle prefix is {existing}, "
            f"expected {expected_existing}"
        )
    predecessor = (
        _sha256_file(supplied_root / f"{existing[-1]}.json") if existing else None
    )
    if state in {"PHASE_A", "LABEL_ACCESS"}:
        sealed = _load_json(supplied_root / "SEALED.json")
        if sealed.get("config_sha256_or_null") != config_hash:
            raise RuntimeError("final config changed after SEALED")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": LEDGER_KIND,
        "protocol_instance_id": instance,
        "state": state,
        "frozen_contract_sha256": frozen_hash,
        "config_sha256_or_null": config_hash,
        "predecessor_sha256": predecessor,
    }
    path = supplied_root / f"{state}.json"
    _exclusive_json(path, payload)
    return {
        "status": "claimed",
        "state": state,
        "protocol_instance_id": instance,
        "state_path": str(path),
        "state_sha256": _sha256_file(path),
        "config_sha256": config_hash,
        "safe_for_submission": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--state", choices=STATES, required=True)
    parser.add_argument("--expected-config-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = advance_state(
        config_path=args.config,
        ledger_root=args.ledger_root,
        state=args.state,
        expected_config_sha256=args.expected_config_sha256,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

