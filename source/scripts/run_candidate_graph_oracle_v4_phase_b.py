#!/usr/bin/env python3
"""Run candidate-graph-oracle Phase B in a fail-closed macOS sandbox.

The unsandboxed parent parses only opaque command-line strings, creates the
fresh output directory, builds a static default-deny profile, and immediately
self-reexecs through ``/usr/bin/sandbox-exec``.  In particular, it does not
open or parse the protocol config.  The sandbox profile contains the fixture
bundle root as one opaque read root; it never constructs or names a label
subpath.  Label-relative paths remain evaluator-only state and are first read
after the evaluator's durable ``TARGET_ACCESS_STARTED`` marker.

The evaluator owns the exact Phase-B output tree.  Runner provenance is emitted
as a canonical authenticated JSON envelope on stdout, rather than adding files
that would invalidate the independently verified evaluator tree.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import platform
import shutil
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
EXPECTED_PYTHON = REPO_ROOT / ".conda/bin/python"
EXPECTED_PYTHON_REAL = REPO_ROOT / ".conda/bin/python3.11"
EXPECTED_CONFIG = REPO_ROOT / "configs/candidate_graph_oracle_ceiling_v4.json"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
SYSTEM_EXECUTABLES = (Path("/usr/bin/uname"),)
SANDBOXED_ENV = "CANDIDATE_GRAPH_ORACLE_PHASE_B_SANDBOXED"
SANDBOX_PROFILE_SHA_ENV = "CANDIDATE_GRAPH_ORACLE_PHASE_B_PROFILE_SHA256"
SANDBOX_PROFILE_TEXT_ENV = "CANDIDATE_GRAPH_ORACLE_PHASE_B_PROFILE_TEXT"
REPORT_NAME = "candidate_graph_oracle_ceiling_report.json"

# These are the only repository payloads the wrapper makes readable.  The list
# is static so the parent can construct the sandbox before opening the config.
# The evaluator later authenticates every protocol-declared file against its
# frozen hash.  Directory ancestors are granted as exact literals only.
PINNED_REPO_READ_FILES = (
    "configs/candidate_graph_oracle_ceiling_v4.json",
    "configs/candidate_graph_oracle_environment_lock_v1.json",
    "configs/denoise_splits_seed20260710.json",
    "configs/denoise_validation_quarantine_v1.json",
    "configs/assembly_audit_exclusion_v1.json",
    "scripts/evaluate_candidate_graph_oracle_v4.py",
    "tests/test_candidate_graph_oracle_v4.py",
    "scripts/build_candidate_graph_oracle_v4_fixtures.py",
    "tests/test_build_candidate_graph_oracle_v4_fixtures.py",
    "scripts/finalize_candidate_graph_oracle_v4_protocol.py",
    "tests/test_finalize_candidate_graph_oracle_v4_protocol.py",
    "scripts/update_candidate_graph_oracle_v4_ledger.py",
    "scripts/verify_candidate_graph_oracle_v4_result.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_job/run_phase_a.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_job/kernel-metadata.json",
    "scripts/push_candidate_graph_oracle_v4_phase_a.py",
    "scripts/run_candidate_graph_oracle_v4_phase_b.py",
    "tests/test_run_candidate_graph_oracle_v4_phase_b.py",
    "scripts/build_candidate_graph_oracle_v4_kaggle_bundles.py",
    "tests/test_build_candidate_graph_oracle_v4_kaggle_bundles.py",
    "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt",
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/__init__.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/compatibility.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/components.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/geometry.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/learned.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/metrics.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/panels.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/protocol.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/qap.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_assembly/solvers.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_denoise_v2/__init__.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_denoise_v2/degradation.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_denoise_v2/inference.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_denoise_v2/losses.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_denoise_v2/metrics.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_denoise_v2/model.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_denoise_v2/tiles.py",
    "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src/puzzle_denoise_v2/training.py",
)

KNOWN_CODE_ALLOWLIST = frozenset(
    value
    for value in PINNED_REPO_READ_FILES
    if "/candidate_graph_oracle_v4_source_snapshot/src/" in value
)

# Runtime roots needed by dyld, CPython, native scientific packages, locale and
# timezone lookup.  None is writable, and network operations remain denied.
SYSTEM_READ_ROOTS = (
    "/System",
    "/usr",
    "/Library",
    "/bin",
    "/sbin",
    "/dev",
    "/private/etc",
    "/private/var/db",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_object_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_line(payload: Any) -> bytes:
    return _canonical_object_bytes(payload) + b"\n"


def _read_regular_file(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"not a single-link regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _sha256(path: Path) -> str:
    return _sha256_bytes(_read_regular_file(path))


def _load_json_object(path: Path) -> dict[str, Any]:
    raw = _read_regular_file(path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _lexical_absolute(value: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeError("sandbox path must be one non-empty NUL-free string")
    return Path(os.path.abspath(os.path.expanduser(value)))


def _profile_canonical(path: Path) -> Path:
    """Return the kernel-visible spelling even when an ancestor is a symlink."""

    absolute = _lexical_absolute(os.fspath(path))
    if os.path.lexists(absolute):
        return Path(os.path.realpath(absolute))
    return Path(os.path.realpath(absolute.parent)) / absolute.name


def _sbpl_quote(value: str | Path) -> str:
    text = os.fspath(value)
    if "\x00" in text or "\n" in text or "\r" in text:
        raise RuntimeError("sandbox path contains a forbidden control character")
    # SBPL accepts the same backslash escapes needed for quotes and slashes in
    # ordinary macOS paths.  Keep non-ASCII characters literal.
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _literal_ancestors(path: Path) -> set[Path]:
    absolute = _lexical_absolute(os.fspath(path))
    result: set[Path] = {Path("/")}
    current = absolute
    while True:
        result.add(current)
        if current == current.parent:
            break
        current = current.parent
    return result


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_profile_topology(args: argparse.Namespace) -> None:
    config = _profile_canonical(_lexical_absolute(args.config))
    if config != _profile_canonical(EXPECTED_CONFIG):
        raise RuntimeError("Phase B accepts only the frozen repository config path")

    phase_a = _profile_canonical(_lexical_absolute(args.phase_a_dir))
    bundle = _profile_canonical(_lexical_absolute(args.fixture_bundle_root))
    lifecycle = _profile_canonical(_lexical_absolute(args.lifecycle_ledger))
    output = _profile_canonical(_lexical_absolute(args.output))
    roots = {
        "phase_a": phase_a,
        "fixture_bundle": bundle,
        "lifecycle": lifecycle,
        "output": output,
    }
    values = list(roots.items())
    for index, (first_name, first_path) in enumerate(values):
        for second_name, second_path in values[index + 1 :]:
            if _paths_overlap(first_path, second_path):
                raise RuntimeError(
                    f"sandbox roots overlap: {first_name}/{second_name}"
                )

    forbidden = _profile_canonical(REPO_ROOT / "puzzle")
    for name, root in roots.items():
        if _paths_overlap(root, forbidden):
            raise RuntimeError(f"sandbox root overlaps forbidden puzzle data: {name}")

    # Input paths are safe to construct before the target marker and must be
    # exact descendants of the one opaque bundle.  No private descendant name
    # is known or joined here.
    fixture_root = _profile_canonical(_lexical_absolute(args.fixture_root))
    fixture_manifest = _profile_canonical(
        _lexical_absolute(args.fixture_manifest)
    )
    expected_fixture_root = bundle / "fixture_input"
    expected_fixture_manifest = expected_fixture_root / "fixture_input_manifest.json"
    if fixture_root != expected_fixture_root:
        raise RuntimeError("fixture input root is not the exact bundle child")
    if fixture_manifest != expected_fixture_manifest:
        raise RuntimeError("fixture input manifest is not the exact bundle artifact")


def _profile_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path], Path]:
    _validate_profile_topology(args)
    lexical_dynamic_paths = {
        _lexical_absolute(args.config),
        _lexical_absolute(args.phase_a_dir),
        _lexical_absolute(args.fixture_bundle_root),
        _lexical_absolute(args.lifecycle_ledger),
        _lexical_absolute(args.output),
    }
    config_path = _profile_canonical(_lexical_absolute(args.config))
    phase_a_root = _profile_canonical(_lexical_absolute(args.phase_a_dir))
    bundle_root = _profile_canonical(_lexical_absolute(args.fixture_bundle_root))
    lifecycle_root = _profile_canonical(_lexical_absolute(args.lifecycle_ledger))
    output_root = _profile_canonical(_lexical_absolute(args.output))

    read_files = {
        _profile_canonical(REPO_ROOT / relative)
        for relative in PINNED_REPO_READ_FILES
    }
    read_files.add(config_path)
    read_subpaths = {
        _profile_canonical(EXPECTED_PYTHON.parent.parent),
        phase_a_root,
        bundle_root,
        lifecycle_root,
        output_root,
        *(_profile_canonical(Path(value)) for value in SYSTEM_READ_ROOTS),
    }
    ancestors: set[Path] = set()
    for path in read_files | read_subpaths | {output_root} | lexical_dynamic_paths:
        ancestors.update(_literal_ancestors(path))
    # Exact ancestors permit path traversal and dyld startup without granting
    # recursive access to the repository (notably puzzle/train/targets).
    read_literals = sorted(read_files | ancestors, key=os.fspath)
    return read_literals, sorted(read_subpaths, key=os.fspath), output_root


def build_sandbox_profile(args: argparse.Namespace) -> str:
    read_literals, read_subpaths, output_root = _profile_paths(args)
    literal_rules = " ".join(
        f"(literal {_sbpl_quote(path)})" for path in read_literals
    )
    subpath_rules = " ".join(
        f"(subpath {_sbpl_quote(path)})" for path in read_subpaths
    )
    process_exec_rules = " ".join(
        f"(literal {_sbpl_quote(path)})"
        for path in (EXPECTED_PYTHON, EXPECTED_PYTHON_REAL, *SYSTEM_EXECUTABLES)
    )
    output_rules = " ".join(
        (
            f"(literal {_sbpl_quote(output_root)})",
            f"(subpath {_sbpl_quote(output_root)})",
            '(literal "/dev/null")',
        )
    )
    pytest_metadata_root = _profile_canonical(REPO_ROOT / "tests")
    profile = "\n".join(
        (
            "(version 1)",
            "(deny default)",
            "(deny network*)",
            f"(allow process-exec {process_exec_rules})",
            "(allow process-fork)",
            "(allow process-info*)",
            "(allow signal (target self))",
            "(allow sysctl-read)",
            "(allow mach-lookup)",
            "(allow ipc-posix*)",
            "(allow file-read-metadata "
            f"(literal {_sbpl_quote(pytest_metadata_root)}) "
            f"(subpath {_sbpl_quote(pytest_metadata_root)}))",
            f"(allow file-read* {literal_rules} {subpath_rules})",
            f"(allow file-write* {output_rules})",
        )
    ) + "\n"
    # The profile is generated solely from the opaque bundle root; this
    # function deliberately has no knowledge of evaluator-private descendants.
    return profile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--phase-a-dir", required=True)
    parser.add_argument("--phase-a-envelope-sha256", required=True)
    parser.add_argument("--fixture-manifest", required=True)
    parser.add_argument("--fixture-manifest-sha256", required=True)
    parser.add_argument("--fixture-root", required=True)
    parser.add_argument("--fixture-bundle-root", required=True)
    parser.add_argument("--lifecycle-ledger", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _assert_expected_python() -> None:
    executable = Path(os.path.realpath(sys.executable))
    expected = Path(os.path.realpath(EXPECTED_PYTHON))
    if executable != expected:
        raise RuntimeError(f"Phase B must run with {EXPECTED_PYTHON}")


def _create_fresh_output_root(output_root: Path) -> None:
    parent = output_root.parent
    parent_descriptor = os.open(
        parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.mkdir(output_root.name, mode=0o700, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except FileExistsError as error:
        raise RuntimeError("Phase-B output root must be fresh") from error
    finally:
        os.close(parent_descriptor)


def _reexec_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--config",
        args.config,
        "--config-sha256",
        args.config_sha256,
        "--phase-a-dir",
        args.phase_a_dir,
        "--phase-a-envelope-sha256",
        args.phase_a_envelope_sha256,
        "--fixture-manifest",
        args.fixture_manifest,
        "--fixture-manifest-sha256",
        args.fixture_manifest_sha256,
        "--fixture-root",
        args.fixture_root,
        "--fixture-bundle-root",
        args.fixture_bundle_root,
        "--lifecycle-ledger",
        args.lifecycle_ledger,
        "--output",
        args.output,
    ]


def _self_reexec_in_sandbox(args: argparse.Namespace) -> None:
    _assert_expected_python()
    if not SANDBOX_EXEC.is_file():
        raise RuntimeError(f"required sandbox backend is missing: {SANDBOX_EXEC}")
    profile = build_sandbox_profile(args)
    profile_sha256 = _sha256_bytes(profile.encode("utf-8"))
    _, _, output_root = _profile_paths(args)
    _create_fresh_output_root(output_root)
    environment = dict(os.environ)
    environment[SANDBOXED_ENV] = "1"
    environment[SANDBOX_PROFILE_SHA_ENV] = profile_sha256
    environment[SANDBOX_PROFILE_TEXT_ENV] = profile
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    command = [
        os.fspath(SANDBOX_EXEC),
        "-p",
        profile,
        os.fspath(EXPECTED_PYTHON),
        os.fspath(Path(__file__).absolute()),
        *_reexec_arguments(args),
    ]
    os.execve(os.fspath(SANDBOX_EXEC), command, environment)
    raise AssertionError("os.execve unexpectedly returned")


def _denied_errno(error: OSError, *, label: str) -> dict[str, Any]:
    if error.errno not in {errno.EACCES, errno.EPERM}:
        raise RuntimeError(f"sandbox probe {label} failed for a non-denial reason") from error
    return {
        "label": label,
        "denied": True,
        "errno": int(error.errno),
        "errno_name": errno.errorcode.get(error.errno, "UNKNOWN"),
    }


def _probe_read_denied(path: Path, *, label: str) -> dict[str, Any]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        return _denied_errno(error, label=label)
    else:
        os.close(descriptor)
        raise RuntimeError(f"sandbox unexpectedly exposed {label}")


def _probe_phase_a_write_denied(phase_a_root: Path) -> dict[str, Any]:
    probe = phase_a_root / f".phase_b_sandbox_write_probe_{os.getpid()}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            probe,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as error:
        return _denied_errno(error, label="phase_a_write")
    else:
        os.close(descriptor)
        descriptor = None
        # This branch means the sandbox was bypassed or miscompiled.  Remove
        # the zero-byte sentinel before failing; no Phase-A bytes were opened.
        os.unlink(probe)
        raise RuntimeError("sandbox unexpectedly permitted a Phase-A write")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _probe_network_denied() -> dict[str, Any]:
    try:
        handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError as error:
        result = _denied_errno(error, label="network_outbound")
        result["denied_at"] = "socket_create"
        return result
    try:
        try:
            handle.connect(("127.0.0.1", 9))
        except OSError as error:
            result = _denied_errno(error, label="network_outbound")
            result["denied_at"] = "connect"
            return result
        raise RuntimeError("sandbox unexpectedly permitted an outbound socket")
    finally:
        handle.close()


def _sandbox_attestation(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get(SANDBOXED_ENV) != "1":
        raise RuntimeError("Phase-B child lacks sandbox reexec marker")
    profile = build_sandbox_profile(args)
    profile_sha256 = _sha256_bytes(profile.encode("utf-8"))
    inherited_profile_sha256 = os.environ.get(SANDBOX_PROFILE_SHA_ENV)
    inherited_profile = os.environ.get(SANDBOX_PROFILE_TEXT_ENV)
    if inherited_profile_sha256 != profile_sha256 or inherited_profile != profile:
        raise RuntimeError("sandbox profile/reexec hash mismatch")
    os.environ.pop(SANDBOX_PROFILE_TEXT_ENV, None)
    os.environ.pop(SANDBOX_PROFILE_SHA_ENV, None)
    os.environ.pop(SANDBOXED_ENV, None)

    config_path = _lexical_absolute(args.config)
    config_sha256 = _sha256(config_path)
    if config_sha256 != args.config_sha256:
        raise RuntimeError("out-of-band final config SHA256 mismatch")
    output_root = _lexical_absolute(args.output)
    if not output_root.is_dir() or output_root.is_symlink() or any(output_root.iterdir()):
        raise RuntimeError("fresh sandbox output root invariant failed")

    output_probe = output_root / f".sandbox_output_probe_{os.getpid()}"
    descriptor = os.open(
        output_probe,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    os.close(descriptor)
    os.unlink(output_probe)

    denial_probes = [
        _probe_read_denied(REPO_ROOT / "puzzle/train", label="repo_puzzle_train_read"),
        _probe_read_denied(
            REPO_ROOT / "puzzle/train/targets", label="repo_puzzle_train_targets_read"
        ),
        _probe_phase_a_write_denied(_lexical_absolute(args.phase_a_dir)),
        _probe_network_denied(),
    ]
    return {
        "backend": os.fspath(SANDBOX_EXEC),
        "profile_sha256": profile_sha256,
        "default_deny": True,
        "network_policy": "deny network*",
        "config_readable_and_sha256_verified": True,
        "fresh_output_write_probe": True,
        "denial_probes": denial_probes,
    }


def _assert_local_environment(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    # Heavy imports are deliberately child-only: no project or config-related
    # module is imported before sandbox reexec.
    import cv2
    import kornia
    import numpy as np
    import scipy
    import skimage
    import torch
    from PIL import __version__ as pillow_version

    pins = config["runtime_pins"]
    lock_path = (config_path.parent.parent / pins["environment_lock_path"]).absolute()
    if _sha256(lock_path) != pins["environment_lock_sha256"]:
        raise RuntimeError("environment-lock runtime pin mismatch")
    lock = _load_json_object(lock_path)
    expected = lock.get("fixture_preparation_and_phase_b")
    if not isinstance(expected, dict):
        raise RuntimeError("environment lock lacks local Phase-B contract")
    actual = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": pillow_version,
            "kornia": kornia.__version__,
            "scikit_image": skimage.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
    }
    for field in ("platform", "python", "packages"):
        if expected.get(field) != actual[field]:
            raise RuntimeError(f"local Phase-B environment mismatch: {field}")
    return {"lock_sha256": _sha256(lock_path), **actual}


def _run_capture(
    command: list[str], *, environment: Mapping[str, str] | None = None
) -> tuple[str, str]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=None if environment is None else dict(environment),
    )
    output = completed.stdout
    if completed.returncode != 0:
        tail = output[-4000:]
        raise RuntimeError(
            f"command failed with code {completed.returncode}; captured tail:\n{tail}"
        )
    return output, _sha256_bytes(output.encode("utf-8"))


def _verified_report_envelope(
    report_path: Path,
) -> tuple[dict[str, Any], bytes, str]:
    report_raw = _read_regular_file(report_path)
    try:
        report_envelope = json.loads(report_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Phase-B report is not valid JSON") from error
    if (
        not isinstance(report_envelope, dict)
        or set(report_envelope) != {"payload", "payload_sha256"}
        or not isinstance(report_envelope.get("payload"), dict)
        or report_raw != _canonical_line(report_envelope)
    ):
        raise RuntimeError("Phase-B report envelope is malformed or non-canonical")
    report = report_envelope["payload"]
    computed_payload_sha256 = _sha256_bytes(_canonical_object_bytes(report))
    if report_envelope["payload_sha256"] != computed_payload_sha256:
        raise RuntimeError("Phase-B report payload_sha256 mismatch")
    return report, report_raw, computed_payload_sha256


def _filesystem_bindings(
    args: argparse.Namespace, output_root: Path
) -> dict[str, str]:
    return {
        "phase_a_root": os.fspath(_lexical_absolute(args.phase_a_dir)),
        "phase_a_artifact_envelope_sha256": args.phase_a_envelope_sha256,
        "fixture_bundle_root": os.fspath(_lexical_absolute(args.fixture_bundle_root)),
        "fixture_input_root": os.fspath(_lexical_absolute(args.fixture_root)),
        "fixture_input_manifest": os.fspath(_lexical_absolute(args.fixture_manifest)),
        "fixture_input_manifest_sha256": args.fixture_manifest_sha256,
        "lifecycle_ledger_root": os.fspath(_lexical_absolute(args.lifecycle_ledger)),
        "output_root": os.fspath(_lexical_absolute(os.fspath(output_root))),
    }


def _child_main(args: argparse.Namespace) -> None:
    _assert_expected_python()
    sandbox = _sandbox_attestation(args)

    config_path = _lexical_absolute(args.config)
    config = _load_json_object(config_path)
    if config.get("kind") != "candidate_graph_oracle_ceiling":
        raise RuntimeError("wrong Phase-B protocol")
    pins = config.get("runtime_pins")
    if not isinstance(pins, dict) or any(
        value is None for key, value in pins.items() if key.endswith("_sha256")
    ):
        raise RuntimeError("Phase B refuses null runtime pins")
    frozen_known_code = config.get("frozen_contract", {}).get("assets", {}).get(
        "known_code_sha256"
    )
    if not isinstance(frozen_known_code, dict) or set(frozen_known_code) != set(
        KNOWN_CODE_ALLOWLIST
    ):
        raise RuntimeError("sandbox code allowlist differs from frozen known-code closure")
    if args.fixture_manifest_sha256 != pins.get("fixture_input_manifest_sha256"):
        raise RuntimeError("Phase-B input manifest anchor differs from runtime pin")

    runner_path = Path(__file__).absolute()
    if _sha256(runner_path) != pins.get("phase_b_runner_sha256"):
        raise RuntimeError("Phase-B runner runtime pin mismatch")
    evaluator_path = REPO_ROOT / str(pins["evaluator_path"])
    tests_path = REPO_ROOT / str(pins["tests_path"])
    if _sha256(evaluator_path) != pins["evaluator_sha256"]:
        raise RuntimeError("evaluator runtime pin mismatch")
    if _sha256(tests_path) != pins["tests_sha256"]:
        raise RuntimeError("evaluator-test runtime pin mismatch")
    environment = _assert_local_environment(config, config_path)

    output_root = _lexical_absolute(args.output)
    runner_scratch = output_root / ".runner_scratch"
    pytest_tmp = runner_scratch / "pytest"
    runner_scratch.mkdir(mode=0o700)
    pytest_config = runner_scratch / "pytest.ini"
    pytest_config_descriptor = os.open(
        pytest_config,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        pytest_config_bytes = b"[pytest]\n"
        if os.write(pytest_config_descriptor, pytest_config_bytes) != len(
            pytest_config_bytes
        ):
            raise RuntimeError("short write while creating isolated pytest config")
        os.fsync(pytest_config_descriptor)
    finally:
        os.close(pytest_config_descriptor)
    preflight_environment = dict(os.environ)
    for key in ("TMPDIR", "TMP", "TEMP"):
        preflight_environment[key] = os.fspath(runner_scratch)
    try:
        preflight_output, preflight_output_sha256 = _run_capture(
            [
                os.fspath(EXPECTED_PYTHON),
                "-m",
                "pytest",
                "-q",
                "-c",
                os.fspath(pytest_config),
                "--rootdir",
                os.fspath(REPO_ROOT),
                "-p",
                "no:cacheprovider",
                "--basetemp",
                os.fspath(pytest_tmp),
                os.fspath(tests_path),
            ],
            environment=preflight_environment,
        )
    finally:
        shutil.rmtree(runner_scratch, ignore_errors=True)
    if any(output_root.iterdir()):
        raise RuntimeError("preflight polluted the exact evaluator output tree")

    command = [
        os.fspath(EXPECTED_PYTHON),
        os.fspath(evaluator_path),
        "--action",
        "phase-b",
        "--config",
        os.fspath(config_path),
        "--config-sha256",
        args.config_sha256,
        "--phase-a-dir",
        args.phase_a_dir,
        "--phase-a-envelope-sha256",
        args.phase_a_envelope_sha256,
        "--fixture-manifest",
        args.fixture_manifest,
        "--fixture-manifest-sha256",
        args.fixture_manifest_sha256,
        "--fixture-root",
        args.fixture_root,
        "--fixture-bundle-root",
        args.fixture_bundle_root,
        "--lifecycle-ledger",
        args.lifecycle_ledger,
        "--output",
        args.output,
        "--device",
        "cpu",
    ]
    evaluator_output, evaluator_output_sha256 = _run_capture(command)

    report_path = output_root / REPORT_NAME
    if not report_path.is_file():
        raise RuntimeError("Phase-B report is missing")
    report, report_raw, computed_payload_sha256 = _verified_report_envelope(
        report_path
    )

    runner_payload = {
        "schema_version": 2,
        "kind": "candidate_graph_oracle_phase_b_runner_attestation",
        "status": report.get("status", "unknown"),
        "safe_for_submission": False,
        "process_id": os.getpid(),
        "config_sha256": args.config_sha256,
        "runner_sha256": _sha256(runner_path),
        "evaluator_sha256": _sha256(evaluator_path),
        "tests_sha256": _sha256(tests_path),
        "environment": environment,
        "sandbox": sandbox,
        "phase_a_envelope_sha256": args.phase_a_envelope_sha256,
        "fixture_input_manifest_sha256": args.fixture_manifest_sha256,
        "filesystem_bindings": _filesystem_bindings(args, output_root),
        "report_path": REPORT_NAME,
        "report_sha256": _sha256_bytes(report_raw),
        "report_payload_sha256": computed_payload_sha256,
        "preflight_output_sha256": preflight_output_sha256,
        "preflight_output_bytes": len(preflight_output.encode("utf-8")),
        "evaluator_output_sha256": evaluator_output_sha256,
        "evaluator_output_bytes": len(evaluator_output.encode("utf-8")),
        "evaluator_output_tree_mutated_by_runner": False,
    }
    runner_envelope = {
        "payload": runner_payload,
        "payload_sha256": _sha256_bytes(_canonical_object_bytes(runner_payload)),
    }
    sys.stdout.buffer.write(_canonical_line(runner_envelope))
    sys.stdout.buffer.flush()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if os.environ.get(SANDBOXED_ENV) != "1":
        _self_reexec_in_sandbox(args)
    _child_main(args)


if __name__ == "__main__":
    main()
