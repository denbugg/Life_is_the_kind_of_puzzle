#!/usr/bin/env python3
"""Kaggle 2xT4 runner for isolated candidate-graph oracle Phase A."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import kornia
import numpy as np
import scipy
import skimage
import torch
from PIL import __version__ as pillow_version


OWNER = "pasha883"
CODE_SLUG = "vsos-candidate-graph-oracle-v2-code"
INPUT_SLUG = "vsos-candidate-graph-oracle-v2-inputs"
RUNTIME_SLUG = "vsos-candidate-graph-oracle-v2-runtime"
CONFIG_RELATIVE = Path("configs/candidate_graph_oracle_ceiling_v2.json")
ENV_RELATIVE = Path("configs/candidate_graph_oracle_environment_lock_v1.json")
EVALUATOR_RELATIVE = Path("scripts/evaluate_candidate_graph_oracle.py")
TEST_RELATIVE = Path("tests/test_candidate_graph_oracle.py")
INPUT_MANIFEST_NAME = "fixture_input_manifest.json"
SHARD_MANIFEST_NAME = "FROZEN_CANDIDATE_GRAPH_SHARD_MANIFEST.json"
FINAL_MANIFEST_NAME = "FROZEN_CANDIDATE_GRAPH_MANIFEST.json"
OUTPUT_ROOT = Path("/kaggle/working/candidate_graph_oracle_v2_phase_a")
WRAPPER_PATH = Path("/kaggle/working/candidate_graph_oracle_v2_phase_a_wrapper.json")


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"not an unlinked regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        encoded = _canonical_bytes(payload)
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _dataset_root(slug: str) -> Path:
    candidates = [
        Path(f"/kaggle/input/datasets/{OWNER}/{slug}"),
        Path(f"/kaggle/input/{slug}"),
    ]
    found = [path.resolve() for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one mounted dataset root for {slug}: {found}")
    if found[0].is_symlink():
        raise RuntimeError(f"dataset root may not be a symlink: {slug}")
    return found[0]


def _find_unique(root: Path, name: str, expected_sha256: str | None = None) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file() and not path.is_symlink()]
    if expected_sha256 is not None:
        matches = [path for path in matches if _sha256(path) == expected_sha256]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {matches}")
    return matches[0]


def _assert_exact_code_mount(
    root: Path, config: Mapping[str, Any]
) -> dict[str, str]:
    """Reject every unpinned file, shadow package, link, and mount escape."""

    pins = config.get("runtime_pins")
    policy = config.get("runtime_pin_mutation_policy")
    frozen = config.get("frozen_contract")
    if not isinstance(pins, dict) or not isinstance(policy, dict) or not isinstance(frozen, dict):
        raise RuntimeError("code mount protocol closure is missing")
    pairs = policy.get("code_pin_fields")
    known_code = frozen.get("assets", {}).get("known_code_sha256")
    if not isinstance(pairs, list) or not isinstance(known_code, dict):
        raise RuntimeError("code mount hash closure is missing")
    expected_hashes: dict[str, str] = {
        CONFIG_RELATIVE.as_posix(): _sha256(root / CONFIG_RELATIVE),
    }
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"path_field", "sha256_field"}:
            raise RuntimeError("code pin pair schema drift")
        relative = pins.get(pair["path_field"])
        digest = pins.get(pair["sha256_field"])
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RuntimeError("null code mount runtime pin")
        expected_hashes[relative] = digest
    for relative, digest in known_code.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RuntimeError("known-code hash closure drift")
        expected_hashes[relative] = digest

    lifecycle_files = {
        "lifecycle/PREP.json",
        "lifecycle/SEALED.json",
        "lifecycle/PHASE_A.json",
        "lifecycle/runtime_pin_transitions/00_code_pins.intent.json",
        "lifecycle/runtime_pin_transitions/00_code_pins.complete.json",
        "lifecycle/runtime_pin_transitions/01_fixtures_pins.intent.json",
        "lifecycle/runtime_pin_transitions/01_fixtures_pins.complete.json",
    }
    expected_files = set(expected_hashes) | lifecycle_files
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    root_device = root.stat().st_dev
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"code mount contains a symlink: {relative}")
        info = path.stat()
        if info.st_dev != root_device:
            raise RuntimeError(f"code mount crosses a device boundary: {relative}")
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            if info.st_nlink != 1:
                raise RuntimeError(f"code mount contains a hardlink: {relative}")
            actual_files.add(relative)
        else:
            raise RuntimeError(f"code mount contains a special entry: {relative}")
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in list(Path(relative).parents)[:-1]
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError(
            "code mount exact tree drift: "
            f"missing_files={sorted(expected_files - actual_files)} "
            f"extra_files={sorted(actual_files - expected_files)} "
            f"missing_dirs={sorted(expected_directories - actual_directories)} "
            f"extra_dirs={sorted(actual_directories - expected_directories)}"
        )
    for relative, expected in expected_hashes.items():
        if _sha256(root / relative) != expected:
            raise RuntimeError(f"code mount pinned hash mismatch: {relative}")
    return expected_hashes


def _hardware_and_environment(lock_path: Path) -> dict[str, Any]:
    lock = _load_json(lock_path)
    expected = lock.get("kaggle_phase_a")
    if not isinstance(expected, dict):
        raise RuntimeError("environment lock lacks Kaggle Phase-A contract")
    devices = []
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Phase A requires exactly two CUDA devices")
    for index in range(2):
        device = torch.device("cuda", index)
        left = torch.arange(4096, device=device, dtype=torch.float32).reshape(64, 64)
        value = float((left @ left.T).sum().item())
        info = {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "tensor_probe": value,
        }
        if info["name"] != "Tesla T4" or info["capability"] != [7, 5]:
            raise RuntimeError(f"unexpected Phase-A GPU: {info}")
        devices.append(info)
    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "pillow": pillow_version,
        "opencv": cv2.__version__,
        "kornia": kornia.__version__,
        "devices": devices,
    }
    if actual["python"] != expected["python"]:
        raise RuntimeError("Kaggle Phase-A environment mismatch: python")
    if actual["cuda_runtime"] != expected["cuda_runtime"]:
        raise RuntimeError("Kaggle Phase-A environment mismatch: cuda_runtime")
    for key in (
        "torch",
        "numpy",
        "scipy",
        "scikit_image",
        "pillow",
        "opencv",
        "kornia",
    ):
        if actual[key] != expected["packages"][key]:
            raise RuntimeError(f"Kaggle Phase-A environment mismatch: {key}")
    return actual


def _run_checked(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=command_cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")


def _spawn(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[subprocess.Popen[str], Any]:
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=command_cwd,
        env=env,
        text=True,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, handle


started = time.time()
code_root = _dataset_root(CODE_SLUG)
input_root = _dataset_root(INPUT_SLUG)
runtime_root = _dataset_root(RUNTIME_SLUG)
command_cwd = code_root
base_env = os.environ.copy()
base_env.update(
    {
        "PYTHONPATH": os.pathsep.join(
            [str(code_root / "src"), str(code_root), base_env.get("PYTHONPATH", "")]
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_ADDOPTS": "-p no:cacheprovider",
    }
)

config_path = code_root / CONFIG_RELATIVE
environment_path = code_root / ENV_RELATIVE
evaluator_path = code_root / EVALUATOR_RELATIVE
tests_path = code_root / TEST_RELATIVE
for required in (config_path, environment_path, evaluator_path, tests_path):
    if not required.is_file() or required.is_symlink():
        raise RuntimeError(f"missing pinned code artifact: {required}")
config = _load_json(config_path)
pins = config["runtime_pins"]
fixture_builder_tests_path = code_root / pins["fixture_builder_tests_path"]
if not fixture_builder_tests_path.is_file() or fixture_builder_tests_path.is_symlink():
    raise RuntimeError("missing pinned builder-to-evaluator integration tests")
config_sha256 = _sha256(config_path)
code_mount_hashes = _assert_exact_code_mount(code_root, config)
kernel_metadata_path = code_root / pins["phase_a_kernel_metadata_path"]
kernel_metadata = _load_json(kernel_metadata_path)
expected_versioned_sources = [
    f"{OWNER}/{CODE_SLUG}/2",
    f"{OWNER}/{INPUT_SLUG}/2",
    f"{OWNER}/{RUNTIME_SLUG}/2",
]
expected_launch = {
    "kernel_id": 126840275,
    "kernel_slug": f"{OWNER}/vsos-candidate-graph-oracle-v2-phase-a-t4x2",
    "kernel_version": 2,
    "dataset_versions": {
        "code": {"slug": f"{OWNER}/{CODE_SLUG}", "version": 2},
        "input": {"slug": f"{OWNER}/{INPUT_SLUG}", "version": 2},
        "runtime": {"slug": f"{OWNER}/{RUNTIME_SLUG}", "version": 2},
    },
}
if (
    kernel_metadata.get("id") != expected_launch["kernel_slug"]
    or kernel_metadata.get("id_no") != expected_launch["kernel_id"]
    or kernel_metadata.get("is_private") is not True
    or kernel_metadata.get("enable_gpu") is not True
    or kernel_metadata.get("enable_internet") is not False
    or kernel_metadata.get("machine_shape") != "NvidiaTeslaT4"
    or kernel_metadata.get("dataset_sources") != expected_versioned_sources
    or kernel_metadata.get("oracle_launch_expectation") != expected_launch
):
    raise RuntimeError("Kaggle kernel launch expectation drift")
for mounted_root, allow_model_binaries in ((code_root, False), (runtime_root, True)):
    for mounted_path in mounted_root.rglob("*"):
        lowered = mounted_path.name.lower()
        if any(token in lowered for token in ("fixture_label", "target", "master_secret")):
            raise RuntimeError(f"forbidden Phase-A mount entry: {mounted_path}")
        if (
            mounted_path.is_file()
            and not allow_model_binaries
            and mounted_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npz", ".pt", ".pth", ".bin"}
        ):
            raise RuntimeError(f"unexpected data/binary artifact in code mount: {mounted_path}")
if _sha256(Path(__file__).resolve()) != pins["phase_a_runner_sha256"]:
    raise RuntimeError("Phase-A runner runtime pin mismatch")
for path, field in (
    (environment_path, "environment_lock_sha256"),
    (evaluator_path, "evaluator_sha256"),
    (tests_path, "tests_sha256"),
    (fixture_builder_tests_path, "fixture_builder_tests_sha256"),
):
    if _sha256(path) != pins[field]:
        raise RuntimeError(f"Phase-A runtime pin mismatch: {field}")

hardware = _hardware_and_environment(environment_path)
input_manifest = input_root / INPUT_MANIFEST_NAME
if not input_manifest.is_file() or _sha256(input_manifest) != pins["fixture_input_manifest_sha256"]:
    raise RuntimeError("input fixture manifest runtime pin mismatch")
if {path.name for path in input_root.iterdir()} != {INPUT_MANIFEST_NAME, "records"}:
    raise RuntimeError("input-only dataset root has unlisted entries")
for path in input_root.rglob("*"):
    lowered = path.name.lower()
    if any(token in lowered for token in ("label", "target", "secret", "truth")):
        raise RuntimeError(f"forbidden input-only mount path: {path}")

denoiser = _find_unique(
    runtime_root,
    Path(config["frozen_contract"]["assets"]["denoiser"]["path"]).name,
    config["frozen_contract"]["assets"]["denoiser"]["sha256"],
)
hbt = _find_unique(
    runtime_root,
    Path(config["frozen_contract"]["assets"]["hbt"]["path"]).name,
    config["frozen_contract"]["assets"]["hbt"]["sha256"],
)
if denoiser.parent != runtime_root or hbt.parent != runtime_root:
    raise RuntimeError("runtime checkpoints must be direct mount children")
runtime_entries = {path.name for path in runtime_root.iterdir()}
if runtime_entries != {denoiser.name, hbt.name}:
    raise RuntimeError(
        f"runtime dataset exact tree drift: {sorted(runtime_entries)}"
    )
lifecycle_root = code_root / "lifecycle"
if not lifecycle_root.is_dir():
    raise RuntimeError("read-only lifecycle snapshot is missing from code dataset")

if OUTPUT_ROOT.exists():
    raise RuntimeError("Phase-A output root must be fresh")
OUTPUT_ROOT.mkdir(parents=True)
logs_root = OUTPUT_ROOT / "logs"
logs_root.mkdir()
# The pinned builder-to-evaluator e2e suite is run locally before code pin.  It
# deliberately creates synthetic label-side fixtures, so Phase A only executes
# the input-only evaluator suite; the real mounted manifest is then parsed by
# the evaluator before any model work.
_run_checked(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(tests_path),
    ],
    logs_root / "tests.log",
    env=base_env,
)

processes: list[tuple[subprocess.Popen[str], Any]] = []
shard_dirs: list[Path] = []
for rank in range(2):
    shard_dir = OUTPUT_ROOT / f"shard_{rank}"
    shard_dirs.append(shard_dir)
    command = [
        sys.executable,
        str(evaluator_path),
        "--action",
        "phase-a",
        "--config",
        str(config_path),
        "--config-sha256",
        config_sha256,
        "--fixture-manifest",
        str(input_manifest),
        "--fixture-manifest-sha256",
        pins["fixture_input_manifest_sha256"],
        "--fixture-root",
        str(input_root),
        "--phase-a-dir",
        str(shard_dir),
        "--rank",
        str(rank),
        "--world-size",
        "2",
        "--lifecycle-ledger",
        str(lifecycle_root),
        "--denoiser",
        str(denoiser),
        "--hbt-checkpoint",
        str(hbt),
        "--device",
        "cuda:0",
    ]
    process_env = base_env.copy()
    process_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(rank),
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
    )
    processes.append(
        _spawn(command, logs_root / f"phase_a_rank{rank}.log", process_env)
    )

failures = []
for rank, (process, handle) in enumerate(processes):
    return_code = process.wait()
    handle.close()
    if return_code != 0:
        failures.append((rank, return_code))
if failures:
    raise RuntimeError(f"Phase-A shard failures: {failures}")

shard_manifests = [directory / SHARD_MANIFEST_NAME for directory in shard_dirs]
shard_hashes = [_sha256(path) for path in shard_manifests]
finalized_dir = OUTPUT_ROOT / "finalized"
finalize_command = [
    sys.executable,
    str(evaluator_path),
    "--action",
    "finalize-phase-a",
    "--config",
    str(config_path),
    "--config-sha256",
    config_sha256,
    "--phase-a-dirs",
    *[str(path) for path in shard_dirs],
    "--phase-a-envelope-sha256s",
    *shard_hashes,
    "--finalized-phase-a-dir",
    str(finalized_dir),
    "--lifecycle-ledger",
    str(lifecycle_root),
    "--denoiser",
    str(denoiser),
    "--hbt-checkpoint",
    str(hbt),
]
_run_checked(finalize_command, logs_root / "finalize.log", env=base_env)
final_manifest = finalized_dir / FINAL_MANIFEST_NAME
if not final_manifest.is_file():
    raise RuntimeError("finalized Phase-A manifest is missing")

wrapper = {
    "schema_version": 1,
    "kind": "candidate_graph_oracle_phase_a_kaggle_wrapper",
    "status": "phase_a_complete_pending_local_verification",
    "safe_for_submission": False,
    "kernel_slug": "pasha883/vsos-candidate-graph-oracle-v2-phase-a-t4x2",
    "config_sha256": config_sha256,
    "runner_sha256": _sha256(Path(__file__).resolve()),
    "kernel_metadata_sha256": _sha256(
        kernel_metadata_path
    ),
    "launch_expectation": expected_launch,
    "evaluator_sha256": _sha256(evaluator_path),
    "tests_sha256": _sha256(tests_path),
    "fixture_builder_tests_sha256": _sha256(fixture_builder_tests_path),
    "environment_lock_sha256": _sha256(environment_path),
    "input_manifest_sha256": _sha256(input_manifest),
    "runtime_assets": {
        "denoiser_sha256": _sha256(denoiser),
        "hbt_sha256": _sha256(hbt),
    },
    "dataset_mounts": {
        "code": {"slug": f"{OWNER}/{CODE_SLUG}", "version": 2, "path": str(code_root)},
        "input": {"slug": f"{OWNER}/{INPUT_SLUG}", "version": 2, "path": str(input_root)},
        "runtime": {"slug": f"{OWNER}/{RUNTIME_SLUG}", "version": 2, "path": str(runtime_root)},
    },
    "exact_code_mount_sha256": code_mount_hashes,
    "hardware": hardware,
    "shards": [
        {"rank": rank, "manifest_sha256": shard_hashes[rank]}
        for rank in range(2)
    ],
    "finalized_phase_a_manifest": str(final_manifest.relative_to(OUTPUT_ROOT)),
    "finalized_phase_a_manifest_sha256": _sha256(final_manifest),
    "seconds": time.time() - started,
}
_atomic_json(WRAPPER_PATH, wrapper)
print(json.dumps(wrapper, sort_keys=True, indent=2))
