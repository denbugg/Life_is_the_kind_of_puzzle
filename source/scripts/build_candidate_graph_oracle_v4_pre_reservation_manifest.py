#!/usr/bin/env python3
"""Capture the complete local-only v4 source closure before Kaggle reservation.

This program is deliberately standard-library-only and has no Kaggle adapter.
It verifies the unresolved reservation placeholders, null runtime pins, frozen
source snapshot, fresh namespaces, reservation templates, and historical v3
byte stability, then writes two exclusive self-hashed evidence files.  It does
not reserve identities, finalize pins, create PREP, or construct fixture-label
paths.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCE = "6c0fe4e8524ce39d830d9a5bee118d8b"
FROZEN_SHA256 = "2070c1b4ff0a3ff42c5ffdd6d611c214c02dbb99b2c51a88f38a862bb1f8a05c"
CONFIG_RELATIVE = "configs/candidate_graph_oracle_ceiling_v4.json"
RESERVATION_ROOT_RELATIVE = (
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_reservations"
)
RESERVATION_RECEIPT_RELATIVE = RESERVATION_ROOT_RELATIVE + "/RESERVATION_RECEIPT.json"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_preflight"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "V4_PRE_RESERVATION_SOURCE_MANIFEST.json"
DEFAULT_MUTATION_RECEIPT = (
    DEFAULT_OUTPUT_ROOT / "V4_PRE_RESERVATION_MUTATION_RECEIPT.json"
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

HISTORICAL_V3_SHA256 = {
    "scripts/push_candidate_graph_oracle_phase_a.py": "cb0e308fbc309de4e96f684ad405f2cf23d40a7bf2ab675afea374a7a0fff243",
    "scripts/verify_candidate_graph_oracle_result.py": "f0df97c42e0354b37ec626828a81347c526d3d580fff5dcdc6fb4e1c068af4d8",
    "scripts/evaluate_candidate_graph_oracle.py": "7723d18b86d1181954117a2c813da0cb45948ccd415f47c2d2dce6575e8a3377",
    "scripts/finalize_candidate_graph_oracle_protocol.py": "7cfc95c894732c987c8ed51bf3943bcdff9afa9a1de5b24b42ed3965e76729f6",
    "scripts/run_candidate_graph_oracle_phase_b.py": "fe9cbd73da14c972a77ff19d0da7427f536801c4f23d089002262d51324fc952",
    "configs/candidate_graph_oracle_ceiling_v3.json": "4dcce283cab00f276ae29ccfb2f82a41edc2d6b7c2e23507341701da2da2c1aa",
    "tests/test_candidate_graph_oracle.py": "a6c526a01217f96049f6a007b39146afb0dc55baf8a79fd5de284ce4fcd4ff37",
    "scripts/build_candidate_graph_oracle_kaggle_bundles.py": "65ff9e1548ed36d4241275199f7b1bd4dfd3c21a547306895beb2ccb9603b3f5",
}

V3_RETIREMENT_EVIDENCE = {
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_readback/CANDIDATE_GRAPH_ORACLE_V3_INVALID_NO_RESULT.json": "bc8154087f9a24eadbc6ffe795f0e2113e63fe93d18fa27546ad9b2764725251",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_readback/V3_DIAGNOSTIC_SCHEMA_FIX_FULL_VERIFICATION.json": "a922404b3f3846add61694901df72f78b477bffea9e1740226aef4b2429dc540",
}

EXTRA_SOURCE_FILES = (
    "scripts/build_candidate_graph_oracle_v4_fixtures.py",
    "scripts/build_candidate_graph_oracle_v4_kaggle_bundles.py",
    "scripts/audit_candidate_graph_oracle_v4_launch_closure.py",
    "scripts/download_candidate_graph_oracle_v4_phase_a_files.py",
    "scripts/materialize_candidate_graph_oracle_v4_bound_verifier_repo.py",
    "scripts/reserve_candidate_graph_oracle_v4_kaggle.py",
    "scripts/build_candidate_graph_oracle_v4_pre_reservation_manifest.py",
    "tests/test_candidate_graph_oracle_v4_contract.py",
    "tests/test_build_candidate_graph_oracle_v4_kaggle_bundles.py",
    "tests/test_finalize_candidate_graph_oracle_v4_protocol.py",
    "tests/test_run_candidate_graph_oracle_v4_phase_b.py",
    "tests/test_candidate_graph_oracle_v4_utilities.py",
    "tests/test_reserve_candidate_graph_oracle_v4_kaggle.py",
    "tests/test_build_candidate_graph_oracle_v4_pre_reservation_manifest.py",
    "tests/test_push_candidate_graph_oracle_phase_a.py",
    "tests/test_verify_candidate_graph_oracle_result.py",
)

RESERVATION_TEMPLATE_FILES = (
    RESERVATION_ROOT_RELATIVE + "/code/RESERVED_VERSION_1.txt",
    RESERVATION_ROOT_RELATIVE + "/code/dataset-metadata.json",
    RESERVATION_ROOT_RELATIVE + "/input/RESERVED_VERSION_1.txt",
    RESERVATION_ROOT_RELATIVE + "/input/dataset-metadata.json",
    RESERVATION_ROOT_RELATIVE + "/runtime/RESERVED_VERSION_1.txt",
    RESERVATION_ROOT_RELATIVE + "/runtime/dataset-metadata.json",
    RESERVATION_ROOT_RELATIVE + "/kernel/kernel-metadata.json",
    RESERVATION_ROOT_RELATIVE + "/kernel/reservation_runner.py",
    RESERVATION_ROOT_RELATIVE + "/journal/.gitkeep",
)

PLANNED_FRESH_ROOTS = {
    "fixture_bundle": (
        "runs/assembly_v1/candidate_graph_oracle_fixtures_v4_" + INSTANCE
    ),
    "phase_a_readback": (
        "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_readback"
    ),
    "phase_b_output": (
        "runs/assembly_v1/candidate_graph_oracle_v4_phase_b_output"
    ),
    "dataset_v2_bundles": (
        "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_bundles"
    ),
    "phase_a_launch_state": (
        "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_launch_state"
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _canonical_object_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_file_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_object_bytes(value) + b"\n"


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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


def _sha(path: Path) -> str:
    return hashlib.sha256(_read_regular(path)).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value, raw


def _relative(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise RuntimeError(f"non-canonical repository path: {value!r}")
    return value


def _descriptor(root: Path, relative: str) -> dict[str, Any]:
    relative = _relative(relative)
    path = root / relative
    raw = _read_regular(path)
    return {
        "path": relative,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _literal(path: Path, name: str) -> Any:
    tree = ast.parse(_read_regular(path).decode("utf-8"), filename=str(path))
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
            values.append(ast.literal_eval(value_node))
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one literal {name} in {path}")
    return values[0]


def _assert_absent_or_empty_directory(path: Path, *, label: str) -> str:
    if path.is_symlink():
        raise RuntimeError(f"{label} may not be a symlink")
    if not path.exists():
        return "absent"
    if not path.is_dir() or any(path.iterdir()):
        raise RuntimeError(f"{label} must be absent or empty before reservation")
    return "empty"


def _validate_reservation_templates(root: Path) -> dict[str, Any]:
    reservation_root = root / RESERVATION_ROOT_RELATIVE
    if not reservation_root.is_dir() or reservation_root.is_symlink():
        raise RuntimeError("v4 reservation root is missing or symlinked")
    expected_dirs = {"code", "input", "runtime", "kernel", "journal"}
    if {path.name for path in reservation_root.iterdir()} != expected_dirs:
        raise RuntimeError("v4 reservation root exact directory closure drift")
    expected_slugs = {
        "code": "pasha883/vsos-candidate-graph-oracle-v4-code",
        "input": "pasha883/vsos-candidate-graph-oracle-v4-inputs",
        "runtime": "pasha883/vsos-candidate-graph-oracle-v4-runtime",
    }
    datasets: dict[str, Any] = {}
    for role, slug in expected_slugs.items():
        directory = reservation_root / role
        if {path.name for path in directory.iterdir()} != {
            "RESERVED_VERSION_1.txt",
            "dataset-metadata.json",
        }:
            raise RuntimeError(f"reservation dataset exact tree drift: {role}")
        metadata, _ = _load_json(directory / "dataset-metadata.json")
        marker = _read_regular(directory / "RESERVED_VERSION_1.txt")
        if (
            metadata.get("id") != slug
            or metadata.get("isPrivate") is not True
            or b"contains_fixture_pixels=false" not in marker
            or f"role={role}".encode("ascii") not in marker
            or INSTANCE.encode("ascii") not in marker
        ):
            raise RuntimeError(f"reservation dataset metadata drift: {role}")
        datasets[role] = {
            "slug": slug,
            "metadata_sha256": _sha(directory / "dataset-metadata.json"),
            "marker_sha256": hashlib.sha256(marker).hexdigest(),
        }
    kernel = reservation_root / "kernel"
    if {path.name for path in kernel.iterdir()} != {
        "kernel-metadata.json",
        "reservation_runner.py",
    }:
        raise RuntimeError("reservation kernel exact tree drift")
    metadata, _ = _load_json(kernel / "kernel-metadata.json")
    expected_kernel_slug = "pasha883/vsos-candidate-graph-oracle-v4-phase-a-t4x2"
    if (
        metadata.get("id") != expected_kernel_slug
        or metadata.get("code_file") != "reservation_runner.py"
        or metadata.get("is_private") is not True
        or metadata.get("enable_gpu") is not False
        or metadata.get("enable_tpu") is not False
        or metadata.get("enable_internet") is not False
        or metadata.get("dataset_sources") != []
        or metadata.get("kernel_sources") != []
        or metadata.get("competition_sources") != []
        or metadata.get("model_sources") != []
    ):
        raise RuntimeError("reservation kernel metadata drift")
    journal = reservation_root / "journal"
    if {path.name for path in journal.iterdir()} != {".gitkeep"}:
        raise RuntimeError("pre-reservation journal must contain only .gitkeep")
    return {
        "root": RESERVATION_ROOT_RELATIVE,
        "datasets": datasets,
        "kernel": {
            "slug": expected_kernel_slug,
            "metadata_sha256": _sha(kernel / "kernel-metadata.json"),
            "reservation_runner_sha256": _sha(kernel / "reservation_runner.py"),
        },
        "journal_state": "placeholder_only",
    }


def validate_source_closure(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    config_path = root / CONFIG_RELATIVE
    config, config_raw = _load_json(config_path)
    expected_pretty = (json.dumps(config, ensure_ascii=True, indent=2) + "\n").encode()
    if config_raw != expected_pretty:
        raise RuntimeError("v4 config is not canonical repository JSON")
    frozen = config.get("frozen_contract")
    if not isinstance(frozen, dict):
        raise RuntimeError("v4 frozen contract is missing")
    actual_frozen = hashlib.sha256(_canonical_object_bytes(frozen)).hexdigest()
    if (
        config.get("protocol_instance_id") != INSTANCE
        or frozen.get("protocol_instance", {}).get("exact_value") != INSTANCE
        or config.get("frozen_contract_sha256") != FROZEN_SHA256
        or actual_frozen != FROZEN_SHA256
        or config.get("status") != "local_pre_reservation_source_closure_no_claims"
        or config.get("safe_for_submission") is not False
    ):
        raise RuntimeError("v4 config identity/frozen contract drift")
    pins = config.get("runtime_pins")
    policy = config.get("runtime_pin_mutation_policy")
    if not isinstance(pins, dict) or not isinstance(policy, dict):
        raise RuntimeError("v4 runtime pin closure is missing")
    code_pairs = policy.get("code_pin_fields")
    fixture_pairs = policy.get("fixture_pin_fields")
    if not isinstance(code_pairs, list) or len(code_pairs) != 12:
        raise RuntimeError("v4 code pin field count must remain exactly 12")
    if not isinstance(fixture_pairs, list) or len(fixture_pairs) != 3:
        raise RuntimeError("v4 fixture pin field count must remain exactly 3")
    code_paths: list[str] = []
    for pair in code_pairs:
        if not isinstance(pair, dict) or set(pair) != {"path_field", "sha256_field"}:
            raise RuntimeError("v4 code pin pair schema drift")
        if pins.get(pair["sha256_field"]) is not None:
            raise RuntimeError("v4 code SHA pins must all remain null before reservation")
        code_paths.append(_relative(str(pins[pair["path_field"]])))
    for pair in fixture_pairs:
        if not isinstance(pair, dict) or set(pair) != {"path_field", "sha256_field"}:
            raise RuntimeError("v4 fixture pin pair schema drift")
        if pins.get(pair["sha256_field"]) is not None:
            raise RuntimeError("v4 fixture SHA pins must remain null")
    if pins.get("fixture_builder_path") != "scripts/build_candidate_graph_oracle_v4_fixtures.py":
        raise RuntimeError("v4 fixture builder path drift")

    known = frozen.get("assets", {}).get("known_code_sha256")
    if not isinstance(known, dict) or len(known) != 18:
        raise RuntimeError("v4 frozen source snapshot closure drift")
    for relative, digest in known.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RuntimeError("v4 known-code descriptor drift")
        if _sha(root / _relative(relative)) != digest:
            raise RuntimeError(f"v4 frozen source byte drift: {relative}")

    metadata_path = root / pins["phase_a_kernel_metadata_path"]
    metadata, _ = _load_json(metadata_path)
    expectation = metadata.get("oracle_launch_expectation")
    if (
        metadata.get("id_no") != -1
        or metadata.get("reservation_receipt_sha256") is not None
        or not isinstance(expectation, dict)
        or expectation.get("kernel_id") != -1
        or expectation.get("reservation_receipt_sha256") is not None
    ):
        raise RuntimeError("v4 Phase-A metadata is not an unresolved reservation template")
    if _literal(root / pins["phase_a_runner_path"], "KERNEL_ID") != -1:
        raise RuntimeError("v4 Phase-A runner kernel id is already bound")
    if _literal(root / pins["phase_a_runner_path"], "RESERVATION_RECEIPT_SHA256") is not None:
        raise RuntimeError("v4 Phase-A runner receipt hash is already bound")
    if _literal(root / pins["phase_a_launcher_path"], "EXPECTED_KERNEL_ID") != -1:
        raise RuntimeError("v4 launcher kernel id is already bound")
    if _literal(root / pins["phase_a_launcher_path"], "RESERVATION_RECEIPT_SHA256") is not None:
        raise RuntimeError("v4 launcher receipt hash is already bound")
    if _literal(
        root / pins["result_verifier_path"],
        "EXPECTED_RESERVATION_RECEIPT_SHA256",
    ) is not None:
        raise RuntimeError("v4 verifier receipt hash is already bound")
    if (root / RESERVATION_RECEIPT_RELATIVE).exists():
        raise RuntimeError("v4 reservation receipt already exists")

    reservation = _validate_reservation_templates(root)
    ledger_relative = _relative(str(policy["transition_ledger_root"]))
    ledger_state = _assert_absent_or_empty_directory(
        root / ledger_relative, label="v4 transition ledger"
    )
    fresh_states = {
        name: _assert_absent_or_empty_directory(root / relative, label=name)
        for name, relative in PLANNED_FRESH_ROOTS.items()
    }

    historical: dict[str, dict[str, Any]] = {}
    for relative, expected in {**HISTORICAL_V3_SHA256, **V3_RETIREMENT_EVIDENCE}.items():
        descriptor = _descriptor(root, relative)
        if descriptor["sha256"] != expected:
            raise RuntimeError(f"historical v3 evidence byte drift: {relative}")
        historical[relative] = descriptor

    source_paths = set(code_paths) | set(known) | set(EXTRA_SOURCE_FILES) | set(
        RESERVATION_TEMPLATE_FILES
    )
    source_paths.add(CONFIG_RELATIVE)
    source_files = [_descriptor(root, relative) for relative in sorted(source_paths)]
    if len({record["path"] for record in source_files}) != len(source_files):
        raise RuntimeError("duplicate v4 source closure path")
    config_descriptor = _descriptor(root, CONFIG_RELATIVE)
    return {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v4_pre_reservation_source_manifest",
        "created_utc": _utc_now(),
        "stage": "pre_reservation_source_closure",
        "status": "complete_local_closure_not_pinnable",
        "protocol_instance_id": INSTANCE,
        "frozen_contract_sha256": FROZEN_SHA256,
        "config": config_descriptor,
        "runtime_pin_state": {
            "code_pin_pair_count": 12,
            "code_sha256_values": "all_null",
            "fixture_pin_pair_count": 3,
            "fixture_sha256_values": "all_null",
        },
        "reservation_templates": reservation,
        "reservation_binding": {
            "kernel_id": -1,
            "reservation_receipt_sha256": None,
            "receipt_path": RESERVATION_RECEIPT_RELATIVE,
            "pinnable": False,
        },
        "fresh_namespaces": {
            "transition_ledger": {"path": ledger_relative, "state": ledger_state},
            "planned_roots": {
                name: {"path": PLANNED_FRESH_ROOTS[name], "state": fresh_states[name]}
                for name in sorted(PLANNED_FRESH_ROOTS)
            },
        },
        "source_files": source_files,
        "source_file_count": len(source_files),
        "historical_v3_byte_stability": historical,
        "v3_retirement_disposition": "INVALID_NO_RESULT",
        "blocking_fields": ["kernel_id", "reservation_receipt_sha256"],
        "remaining_sequence": [
            "execute and verify private v1 reservations once",
            "bind positive kernel id and reservation receipt SHA into metadata runner launcher verifier",
            "rerun local tests and source-closure audit",
            "finalize the twelve code SHA pins in one transaction",
            "only then create PREP and fixture pixels under separate authorization",
        ],
        "remote_api_called": False,
        "label_paths_constructed": False,
        "label_files_opened": False,
        "fixture_pixels_built": False,
        "code_pin_performed": False,
        "prep_claimed": False,
        "safe_for_submission": False,
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_file_bytes(value)
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
            if written <= 0:
                raise RuntimeError(f"short write: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return hashlib.sha256(encoded).hexdigest()


def build_evidence(
    *,
    repo_root: Path = REPO_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST,
    mutation_receipt_path: Path = DEFAULT_MUTATION_RECEIPT,
) -> dict[str, Any]:
    if manifest_path.absolute() == mutation_receipt_path.absolute():
        raise RuntimeError("manifest and mutation receipt paths must differ")
    payload = validate_source_closure(repo_root)
    envelope = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical_object_bytes(payload)).hexdigest(),
    }
    manifest_sha = _write_exclusive(manifest_path.absolute(), envelope)
    mutation = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v4_pre_reservation_mutation_receipt",
        "created_utc": _utc_now(),
        "protocol_instance_id": INSTANCE,
        "frozen_contract_sha256": FROZEN_SHA256,
        "source_manifest": {
            "path": str(manifest_path.absolute()),
            "sha256": manifest_sha,
            "payload_sha256": envelope["payload_sha256"],
        },
        "captured_config_sha256": payload["config"]["sha256"],
        "captured_source_file_count": payload["source_file_count"],
        "historical_v3_preserved": True,
        "historical_v3_sha256": {
            path: descriptor["sha256"]
            for path, descriptor in payload["historical_v3_byte_stability"].items()
        },
        "mutations_completed": [
            "fresh v4 protocol id config paths slugs and source snapshot",
            "versioned evaluator verifier launcher finalizer Phase-B runner fixture wrapper and job templates",
            "pixel-free reservation templates and crash-safe orchestrator",
            "versioned bundle audit download materializer utilities and tests",
        ],
        "mutations_explicitly_not_performed": [
            "Kaggle dataset or kernel reservation",
            "positive kernel id or reservation receipt binding",
            "runtime code pin finalization",
            "PREP lifecycle claim",
            "fixture or label construction",
            "Phase-A or Phase-B execution",
        ],
        "remote_api_called": False,
        "label_paths_constructed": False,
        "safe_for_submission": False,
    }
    mutation_envelope = {
        "payload": mutation,
        "payload_sha256": hashlib.sha256(_canonical_object_bytes(mutation)).hexdigest(),
    }
    mutation_sha = _write_exclusive(
        mutation_receipt_path.absolute(), mutation_envelope
    )
    return {
        "status": "v4_pre_reservation_source_closure_recorded",
        "manifest_path": str(manifest_path.absolute()),
        "manifest_sha256": manifest_sha,
        "mutation_receipt_path": str(mutation_receipt_path.absolute()),
        "mutation_receipt_sha256": mutation_sha,
        "source_file_count": payload["source_file_count"],
        "remote_api_called": False,
        "safe_for_submission": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--mutation-receipt", type=Path, default=DEFAULT_MUTATION_RECEIPT
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            build_evidence(
                repo_root=args.repo_root,
                manifest_path=args.manifest,
                mutation_receipt_path=args.mutation_receipt,
            ),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
