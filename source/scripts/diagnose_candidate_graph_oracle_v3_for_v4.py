#!/usr/bin/env python3
"""Input-only v3 postmortem used to preflight a fresh oracle v4.

This program can never validate, promote, or resume v3.  It runs the exact
frozen verifier in an isolated child process and changes one in-memory schema
expectation only: the producer's ``hbt_outside_logits`` diagnostics key is
accepted where the verifier expected ``hbt``.  The purpose is to expose any
later producer/verifier mismatch before a new protocol instance is created.

No label-side path is accepted, constructed, resolved, listed, or opened.
Every input is hashed before and after the child process.  The report is
explicitly diagnostic-only and never authorizes LABEL_ACCESS or Phase B.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCE = "4f3da49d17e8adba46b1359d2cc81a19"
CONFIG_SHA256 = "4dcce283cab00f276ae29ccfb2f82a41edc2d6b7c2e23507341701da2da2c1aa"
FROZEN_VERIFIER_SHA256 = "f0df97c42e0354b37ec626828a81347c526d3d580fff5dcdc6fb4e1c068af4d8"
FROZEN_EVALUATOR_SHA256 = "7723d18b86d1181954117a2c813da0cb45948ccd415f47c2d2dce6575e8a3377"
COMPOSITE_DRIVER_SHA256 = "d5a528fcdd0ebfc2a2cd6939a9561af8ad1592a6a33f321753413fdc0b0b6ca5"
PHASE_A_MANIFEST_SHA256 = "ee9d801458b22be066d21ec296836346c137a495b9f71295c378c7492599c7f1"
INPUT_MANIFEST_SHA256 = "6de4502908ccdbb74c262d63495792cc844f0faceb09f567ffcc8bd8dee9f444"
SHARD_SHA256S = (
    "c9105b68f1601a19c5d823021efd77337b0a61d747d5ff6711f044c860e89dbb",
    "61e9290ab85476a8dc752e6c1f0554bcad6186f34c36740022e301548780e4ee",
)
FORBIDDEN_PATH_TOKENS = (
    "fixture_label",
    "master_secret",
    "label_manifest",
    "/labels/",
    "/targets/",
    "target_access",
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_object(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_file(payload: Mapping[str, Any]) -> bytes:
    return _canonical_object(payload) + b"\n"


def normalize_exact_kaggle_kernel_ref(raw_ref: Any, expected_slug: str) -> str:
    """Executable v4 launcher contract for the one observed SDK alias.

    The immutable raw journal keeps ``raw_ref`` unchanged.  Only the semantic
    projection is normalized, and only these two exact spellings are allowed.
    """

    if not isinstance(raw_ref, str) or not isinstance(expected_slug, str):
        raise RuntimeError("Kaggle kernel ref must be an exact string")
    if raw_ref == expected_slug or raw_ref == f"/code/{expected_slug}":
        return expected_slug
    raise RuntimeError("Kaggle kernel ref is neither canonical nor exact /code alias")


def _guard_path(value: Path, *, label: str, must_exist: bool = True) -> Path:
    """Reject forbidden namespaces lexically and again after resolution."""

    lexical = value.expanduser().absolute()
    lowered = f"/{lexical.as_posix().lower().strip('/')}/"
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"forbidden namespace in {label}")
    resolved = lexical.resolve(strict=must_exist)
    lowered_resolved = f"/{resolved.as_posix().lower().strip('/')}/"
    if any(token in lowered_resolved for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"forbidden resolved namespace in {label}")
    return resolved


def _regular_file_sha256(path: Path) -> tuple[str, int, tuple[int, int]]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"input is not a one-link regular file: {path}")
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest(), int(info.st_size), (int(info.st_dev), int(info.st_ino))


def _tree_snapshot(root: Path, *, label: str) -> dict[str, Any]:
    """Hash an exact safe tree without following directory links."""

    root = _guard_path(root, label=label)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"{label} must be a non-symlink directory")
    files: dict[str, dict[str, Any]] = {}
    directories: list[str] = []
    identities: set[tuple[int, int]] = set()
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(dirnames):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            _guard_path(candidate, label=f"{label}.{relative}")
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError(f"unsafe directory entry in {label}: {relative}")
            directories.append(relative)
        for name in sorted(filenames):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            _guard_path(candidate, label=f"{label}.{relative}")
            digest, size, identity = _regular_file_sha256(candidate)
            if identity in identities:
                raise RuntimeError(f"hardlink/inode alias in {label}: {relative}")
            identities.add(identity)
            files[relative] = {"sha256": digest, "bytes": size}
    files = dict(sorted(files.items()))
    directories = sorted(directories)
    return {
        "root": str(root),
        "file_count": len(files),
        "directory_count": len(directories),
        "tree_sha256": hashlib.sha256(
            _canonical_object({"directories": directories, "files": files})
        ).hexdigest(),
        "files": files,
        "directories": directories,
    }


def _file_snapshot(path: Path, *, label: str) -> dict[str, Any]:
    path = _guard_path(path, label=label)
    digest, size, _ = _regular_file_sha256(path)
    return {"path": str(path), "sha256": digest, "bytes": size}


def _all_snapshots(args: argparse.Namespace) -> dict[str, Any]:
    trees = {
        "snapshot_root": _tree_snapshot(args.snapshot_root, label="snapshot_root"),
        "fixture_input_root": _tree_snapshot(
            args.fixture_input_root, label="fixture_input_root"
        ),
        "phase_a_root": _tree_snapshot(args.phase_a_root, label="phase_a_root"),
        "ledger_root": _tree_snapshot(args.ledger_root, label="ledger_root"),
    }
    files = {
        "snapshot_archive": _file_snapshot(
            args.snapshot_archive, label="snapshot_archive"
        ),
        "wrapper": _file_snapshot(args.wrapper, label="wrapper"),
        "launch_receipt": _file_snapshot(
            args.launch_receipt, label="launch_receipt"
        ),
        "recovered_verification": _file_snapshot(
            args.recovered_verification, label="recovered_verification"
        ),
        "recovered_verifier": _file_snapshot(
            args.recovered_verifier, label="recovered_verifier"
        ),
        "diagnostic_driver": _file_snapshot(Path(__file__), label="diagnostic_driver"),
        "composite_driver": _file_snapshot(
            REPO_ROOT / "scripts/verify_candidate_graph_oracle_v3_phase_a_composite.py",
            label="composite_driver",
        ),
    }
    return {"trees": trees, "files": files}


def _load_module(path: Path, *, module_name: str, expected_sha256: str) -> Any:
    digest, _, _ = _regular_file_sha256(path)
    if digest != expected_sha256:
        raise RuntimeError(f"pinned module SHA drift: {path.name}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    origin = Path(module.__file__).resolve(strict=True)
    if origin != path.resolve(strict=True):
        raise RuntimeError(f"module origin drift: {module_name}")
    return module


def _assert_bound_import_origins(snapshot_root: Path) -> dict[str, str]:
    allowed = (snapshot_root / "src").resolve(strict=True)
    observed: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == "puzzle_assembly" or name.startswith("puzzle_assembly.")) and not (
            name == "puzzle_denoise_v2" or name.startswith("puzzle_denoise_v2.")
        ):
            continue
        origin_value = getattr(module, "__file__", None)
        if not isinstance(origin_value, str):
            raise RuntimeError(f"bound module lacks origin: {name}")
        origin = Path(origin_value).resolve(strict=True)
        try:
            origin.relative_to(allowed)
        except ValueError as error:
            raise RuntimeError(f"module escaped bound snapshot: {name} -> {origin}") from error
        observed[name] = str(origin)
    if not observed:
        raise RuntimeError("no bound puzzle modules were imported")
    return observed


def _validate_diagnostics(records: Iterable[Any]) -> dict[str, Any]:
    count = 0
    hbt_hashes: set[str] = set()
    for index, phase_record in enumerate(records):
        diagnostics = phase_record.manifest.get("derivation_diagnostics")
        if not isinstance(diagnostics, dict) or set(diagnostics) != {
            "hbt_outside_logits",
            "softcycle",
            "qap",
        }:
            raise RuntimeError(f"aligned diagnostics schema drift at record {index}")
        hbt = diagnostics["hbt_outside_logits"]
        if (
            not isinstance(hbt, dict)
            or set(hbt) != {"dtype", "shape", "c_order_sha256"}
            or hbt.get("dtype") != "float32"
            or hbt.get("shape") != [576, 4]
            or not isinstance(hbt.get("c_order_sha256"), str)
            or HEX64.fullmatch(hbt["c_order_sha256"]) is None
        ):
            raise RuntimeError(f"HBT diagnostics descriptor drift at record {index}")
        hbt_hashes.add(hbt["c_order_sha256"])
        softcycle = diagnostics["softcycle"]
        if (
            not isinstance(softcycle, dict)
            or set(softcycle) != {"accepted_edges", "component_sizes"}
            or type(softcycle.get("accepted_edges")) is not int
            or softcycle["accepted_edges"] < 0
            or not isinstance(softcycle.get("component_sizes"), list)
            or not softcycle["component_sizes"]
            or any(type(value) is not int or value < 1 for value in softcycle["component_sizes"])
            or sum(softcycle["component_sizes"]) != 576
        ):
            raise RuntimeError(f"softcycle diagnostics drift at record {index}")
        qap = diagnostics["qap"]
        if not isinstance(qap, dict) or set(qap) != {"qap_w1", "qap_w4"}:
            raise RuntimeError(f"QAP diagnostics coverage drift at record {index}")
        for key in ("qap_w1", "qap_w4"):
            value = qap[key]
            if not isinstance(value, dict) or set(value) != {
                "objective",
                "relaxed_objective",
                "restart",
                "iterations",
                "converged",
            }:
                raise RuntimeError(f"QAP diagnostics schema drift: {index}.{key}")
            if (
                type(value["restart"]) is not int
                or type(value["iterations"]) is not int
                or type(value["converged"]) is not bool
                or type(value["objective"]) not in (int, float)
                or type(value["relaxed_objective"]) not in (int, float)
            ):
                raise RuntimeError(f"QAP diagnostics type drift: {index}.{key}")
        count += 1
    if count != 64:
        raise RuntimeError("diagnostics record count drift")
    return {
        "record_count": count,
        "schema": ["hbt_outside_logits", "qap", "softcycle"],
        "unique_hbt_diagnostic_hashes": len(hbt_hashes),
    }


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    snapshot_root = _guard_path(args.snapshot_root, label="snapshot_root")
    safe_paths = (
        args.snapshot_archive,
        args.fixture_input_root,
        args.phase_a_root,
        args.wrapper,
        args.launch_receipt,
        args.recovered_verification,
        args.recovered_verifier,
        args.ledger_root,
    )
    for index, value in enumerate(safe_paths):
        _guard_path(value, label=f"worker_input_{index}")

    # Bound code wins every project import.  Isolated mode prevents ambient
    # PYTHONPATH/user-site injection; origin assertions below catch reuse.
    sys.path.insert(0, str(snapshot_root / "src"))
    sys.path.insert(1, str(snapshot_root))
    verifier_path = snapshot_root / "scripts/verify_candidate_graph_oracle_result.py"
    evaluator_path = snapshot_root / "scripts/evaluate_candidate_graph_oracle.py"
    verifier = _load_module(
        verifier_path,
        module_name="_oracle_v3_diagnostic_frozen_verifier_f0df",
        expected_sha256=FROZEN_VERIFIER_SHA256,
    )
    evaluator = _load_module(
        evaluator_path,
        module_name="_oracle_v3_diagnostic_frozen_evaluator_7723",
        expected_sha256=FROZEN_EVALUATOR_SHA256,
    )
    composite = _load_module(
        REPO_ROOT / "scripts/verify_candidate_graph_oracle_v3_phase_a_composite.py",
        module_name="_oracle_v3_diagnostic_composite_d5a5",
        expected_sha256=COMPOSITE_DRIVER_SHA256,
    )
    snapshot_closure = composite.verify_snapshot_exact(
        snapshot_root, args.snapshot_archive
    )

    original_exact_keys = verifier._require_exact_keys

    def aligned_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
        if label == "phase_a.derivation_diagnostics":
            if set(expected) != {"hbt", "softcycle", "qap"}:
                raise RuntimeError("unexpected frozen diagnostics expectation")
            expected = {"hbt_outside_logits", "softcycle", "qap"}
        original_exact_keys(value, expected, label=label)

    verifier._require_exact_keys = aligned_exact_keys
    config_path = snapshot_root / "configs/candidate_graph_oracle_ceiling_v3.json"
    context = verifier._load_protocol(
        config_path,
        expected_config_sha256=CONFIG_SHA256,
        allow_unpinned_verifier=False,
    )
    input_evidence = verifier.verify_input_fixture(
        context,
        fixture_root=args.fixture_input_root,
        expected_manifest_sha256=INPUT_MANIFEST_SHA256,
    )
    phase_a = verifier.verify_phase_a(
        context,
        phase_a_root=args.phase_a_root,
        expected_envelope_sha256=PHASE_A_MANIFEST_SHA256,
        shard_anchors=SHARD_SHA256S,
        input_evidence=input_evidence,
    )
    diagnostics = _validate_diagnostics(phase_a.records.values())
    wrapper = composite._verify_wrapper(
        verifier,
        context=context,
        phase_a=phase_a,
        input_evidence=input_evidence,
        wrapper_path=args.wrapper,
    )
    recovered_launch = composite._verify_recovered_evidence(
        recovered_verification_path=args.recovered_verification,
        recovered_verifier_path=args.recovered_verifier,
        launch_receipt_path=args.launch_receipt,
    )
    ledger_root, lifecycle_hashes = evaluator._verify_lifecycle_chain(
        str(args.ledger_root),
        protocol_instance_id=INSTANCE,
        config_sha256=CONFIG_SHA256,
        required_last_state="PHASE_A",
        protocol=context.config,
    )
    if lifecycle_hashes.get("PHASE_A") != phase_a.payload.get(
        "phase_a_lifecycle_sha256"
    ):
        raise RuntimeError("Phase-A manifest does not bind strict lifecycle hash")
    verifier._post_phase_a_rehash(
        context,
        input_evidence=input_evidence,
        phase_a=phase_a,
        allow_unpinned_verifier=False,
    )
    module_origins = _assert_bound_import_origins(snapshot_root)
    render_count = sum(
        len(record.manifest["renders"]) for record in phase_a.records.values()
    )
    if len(phase_a.records) != 64 or render_count != 192:
        raise RuntimeError("Phase-A graph/render count drift")
    return {
        "status": "diagnostic_full_input_only_verification_passed",
        "diagnostic_schema_override": {
            "frozen_expected": ["hbt", "qap", "softcycle"],
            "producer_actual": ["hbt_outside_logits", "qap", "softcycle"],
            "scope": "one in-memory exact-key expectation only",
        },
        "records": len(phase_a.records),
        "graph_artifacts": len(phase_a.records),
        "renders": render_count,
        "diagnostics": diagnostics,
        "snapshot_closure": snapshot_closure,
        "wrapper": wrapper,
        "recovered_launch": recovered_launch,
        "phase_a_manifest_sha256": phase_a.envelope_sha256,
        "input_manifest_sha256": input_evidence.manifest_sha256,
        "strict_lifecycle": {
            "root": str(ledger_root),
            "terminal_state": "PHASE_A",
            "hashes": lifecycle_hashes,
            "transition_receipts_verified": True,
            "label_access_present": False,
        },
        "module_origins": module_origins,
        "frozen_verifier_sha256": FROZEN_VERIFIER_SHA256,
        "frozen_evaluator_sha256": FROZEN_EVALUATOR_SHA256,
        "composite_driver_sha256": COMPOSITE_DRIVER_SHA256,
        "label_paths_constructed_or_opened": False,
        "accepted_v3_result": False,
        "phase_b_authorized": False,
    }


def _worker_command(args: argparse.Namespace) -> list[str]:
    result = [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"]
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
    ):
        result.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("diagnostic output must be fresh")
    _guard_path(args.output, label="output", must_exist=False)
    before = _all_snapshots(args)
    completed = subprocess.run(
        _worker_command(args),
        cwd=REPO_ROOT,
        env={"PATH": os.environ.get("PATH", "")},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    after = _all_snapshots(args)
    if before != after:
        raise RuntimeError("one or more read inputs changed during diagnostic verification")
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated diagnostic worker failed:\n"
            + completed.stderr[-12000:]
            + completed.stdout[-4000:]
        )
    worker_payload = json.loads(completed.stdout)
    if not isinstance(worker_payload, dict):
        raise RuntimeError("isolated diagnostic worker emitted a non-object")
    payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_v3_diagnostic_only_v4_preflight",
        "created_utc": _utc_now(),
        "status": "V3_REMAINS_INVALID_DIAGNOSTIC_PAYLOAD_OTHERWISE_CONSISTENT",
        "protocol_instance_id": INSTANCE,
        "worker": worker_payload,
        "input_snapshot_sha256": hashlib.sha256(_canonical_object(before)).hexdigest(),
        "pre_post_rehash_equal": True,
        "remote_reads_performed": False,
        "remote_writes_performed": False,
        "label_paths_constructed_or_opened": False,
        "label_access_claimed": False,
        "accepted_v3_result": False,
        "phase_b_authorized": False,
        "safe_for_submission": False,
    }
    envelope = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical_object(payload)).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        args.output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        encoded = _canonical_file(envelope)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise RuntimeError("short diagnostic report write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return {
        "status": payload["status"],
        "output": str(args.output),
        "output_sha256": hashlib.sha256(_canonical_file(envelope)).hexdigest(),
        "records": worker_payload["records"],
        "graphs": worker_payload["graph_artifacts"],
        "renders": worker_payload["renders"],
        "accepted_v3_result": False,
        "phase_b_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--snapshot-archive", type=Path, required=True)
    parser.add_argument("--fixture-input-root", type=Path, required=True)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--recovered-verification", type=Path, required=True)
    parser.add_argument("--recovered-verifier", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.worker and args.output is None:
        parser.error("--output is required outside worker mode")
    return args


def main() -> None:
    args = parse_args()
    result = _worker(args) if args.worker else run(args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
