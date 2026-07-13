#!/usr/bin/env python3
"""Fail-closed T4x2 runner for the fixed QAP-weight confirmation gate.

The evaluator is deliberately not launched with DDP.  Two isolated GPU
processes produce disjoint input-only Phase-A shards.  They receive a data
root in which ``train/targets`` does not exist.  After both shard trees are
frozen and hashed, one canonical GPU-0 process performs Phase B and is the
only process that receives the real data root containing targets.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from typing import Any
import zipfile

import numpy as np
from PIL import Image


INPUT = Path("/kaggle/input/datasets/pasha883")
WORKING = Path("/kaggle/working")
PUZZLE_INPUT = INPUT / "vsos-ai-initiative-pazzle"
RUNTIME_INPUT = INPUT / "vsos-assembly-v1-runtime"
BASE_INPUT = INPUT / "vsos-solver-rework-night-code"
OVERLAY_INPUT = INPUT / "vsos-qap-weight-confirmation-code"

STAGING = WORKING / ".qap_weight_confirmation_staging"
WRAPPER = WORKING / "qap_weight_confirmation_wrapper.json"
SUMS = WORKING / "SHA256SUMS.txt"
REPORT_NAME = "qap_weight_confirmation_report.json"
GLOBAL_MANIFEST_NAME = "FROZEN_INPUT_ONLY_MANIFEST.json"
SHARD_MANIFEST_NAME = "FROZEN_INPUT_ONLY_SHARD_MANIFEST.json"
TARGET_MARKER_NAME = "TARGET_ACCESS_STARTED.json"
PHASE_A_ARCHIVE_NAME = "qap_weight_confirmation_phase_a_frozen.zip"
ARCHIVE_TIMESTAMP = (2026, 7, 11, 0, 0, 0)
EXPECTED_KERNEL_METADATA_SHA256 = (
    "0fbbea001b9640f98a9e7938990cefafd657df566bf4d7c1cbee0264b9cc8d87"
)
KERNEL_ID = "pasha883/vsos-fixed-qap-weight-confirmation-t4x2"
DATASET_SOURCES = [
    "pasha883/vsos-ai-initiative-pazzle",
    "pasha883/vsos-assembly-v1-runtime",
    "pasha883/vsos-solver-rework-night-code",
    "pasha883/vsos-qap-weight-confirmation-code",
]

EXPECTED_BASE_ARCHIVE_SHA256 = (
    "a980c158fb349fbc8619e39eb829acdc675e7332d1ec3995c08f38eb49f45d0c"
)
EXPECTED_OVERLAY_ARCHIVE_SHA256 = (
    "46dd531516e691c7c4ff425ba3cc73af40d910e0d8b0ce606b543523d0e68e2b"
)

# These placeholders make an accidentally pushed scaffold fail before it can
# read a single image.  Replace every PENDING value after the overlay files
# are final and locally tested.
EXPECTED_OVERLAY_HASHES = {
    "scripts/evaluate_qap_weight_confirmation.py": (
        "4083d11146f62a91d007a553cfb1ae0ec943141e7a0b3a4639fac3d2f1d9559a"
    ),
    "configs/qap_weight_confirmation_v1.json": (
        "30732463fb200bdff8f909ef06be6cb6c4e7859692e01c9d33c5d55175ffe262"
    ),
    "tests/test_qap_weight_confirmation.py": (
        "f2962e8de01639272acc272e8260a27373f1ccce30b5163228d4ba67bd14e3d4"
    ),
}

# Hash the full imported base package rather than relying on the archive name.
# This is intentionally broader than the evaluator's direct imports so that a
# lazy import cannot silently escape the executable provenance closure.
EXPECTED_BASE_HASHES = {
    "src/puzzle_assembly/__init__.py": "09e051b7555471aafca03cd666d789f033aca47f1c82f6e2af9c0cce50afe9d5",
    "src/puzzle_assembly/compatibility.py": "aff2149b161c4fded4e5d91fbea49a8a62967886148d3ad374467331e0416a9f",
    "src/puzzle_assembly/components.py": "53fcc7c4fd23956db884ee45060e47f8e94a931c16e497e426d67549621bd367",
    "src/puzzle_assembly/cpsat.py": "b368a884c886273156a0fee3ef00a1e4e6e7766daa5ea09531983162e8199abf",
    "src/puzzle_assembly/geometry.py": "1e16bec6fb98a33060558d5d28062334d9114b12424733ef103a40393ef1ba86",
    "src/puzzle_assembly/learned.py": "9e3dba673aa85eaab5698dbeb63b3d94f88e3ea92b5e5979bde4b0273642697b",
    "src/puzzle_assembly/line_seam.py": "56c3065fb36427a96c3fbddda515fc28f49dcfd8e0b3a5a721dd8fd28603305d",
    "src/puzzle_assembly/metrics.py": "84857ef92c382cc0964c21bfec67c13308014a1674aebf8686b17514784dae69",
    "src/puzzle_assembly/panels.py": "783356628517e3a23b8703672bca604c3d879c875f5b5f35f87182425500280f",
    "src/puzzle_assembly/particle.py": "a4aebf628bbdf91fd64fecaaf2b32c7720bb244a911ead4cbab5c29611406fb3",
    "src/puzzle_assembly/protocol.py": "b711ad6d28a2fe60329e3e8236e58adbfbceea8ca4c8bf85e9a057e7619e24f4",
    "src/puzzle_assembly/qap.py": "b8a5e1da67387fd04effd979270ca16925aceab23d37083eda108c6e3e349c32",
    "src/puzzle_assembly/solvers.py": "23f9e32200748349d0da8558b7b44053a758e1c1eb306d8f31ce59feae03fe8e",
    "src/puzzle_assembly/spatial_prior.py": "01901fb3ea8ed584b8ebdbe286973beaa3acd5a74d6fbec83783d6a67912de3a",
    "src/puzzle_denoise_v2/__init__.py": "30849e0f937ba4a50e85ce2eee0d2b930db06fbcc0b7dff84547e121ef2f30b7",
    "src/puzzle_denoise_v2/degradation.py": "7e314081c143a1c7846a9777eaea8716092a85595f856769efd3704a2c583a75",
    "src/puzzle_denoise_v2/final_gate_audit.py": "12f44b5adf808c86bad805a285f2d22fcab36e7c6203a5df11fbb40583ee6f92",
    "src/puzzle_denoise_v2/inference.py": "20767cc26270cfde7472cf33a0247b1ea6d96e5b5c8ff5d705b785ae710dd6da",
    "src/puzzle_denoise_v2/legacy_baseline.py": "0b82f48d1846332462912995c6c13d9433822a8e7bec08e8b92d7f873760cfac",
    "src/puzzle_denoise_v2/losses.py": "56776289cd51e49a28ce54bc4762d144d87c7efbf6d4ca56668fc3b019dbbf34",
    "src/puzzle_denoise_v2/matching.py": "90535d7090c211f78d7d0fb16cc289bff26ae0fd3307dfeeb648924d37f0c0ec",
    "src/puzzle_denoise_v2/metrics.py": "e8275fb096276a63b7114be1a74b24009dc2143dddf299ce5eaceac401a27d36",
    "src/puzzle_denoise_v2/model.py": "37db32fb83ece0f122757bdbec19ffc6a17c5e5e00ef92a26328247d95c55d11",
    "src/puzzle_denoise_v2/prefinetune_benchmark.py": "4e79cbb7b96a56aac6cee77093b11c71a6980e32c1901e154cb4161bce9f487e",
    "src/puzzle_denoise_v2/real_pairs.py": "c0f8db24fd2c503a60e95c42d901643901618494da94a9270c6a26a32dd0b180",
    "src/puzzle_denoise_v2/real_training.py": "4a09201a714534d3624f7ac26c8d68b30a824d93a47c7af1df350264bd4f2fad",
    "src/puzzle_denoise_v2/real_validation.py": "6f0ca30fc5c0489223048f2ffcf7088a110e20c854f038d58877f2bf76cba64f",
    "src/puzzle_denoise_v2/tiles.py": "21270e283e50ea0b155ef194de889222fb0c4f6954437eb1526342c006eefaa7",
    "src/puzzle_denoise_v2/training.py": "6719ee6a62434cd8a00fafb92b28f6a10941cdbf5c83573fc6556b33e5eba56e",
    "src/puzzle_denoise_v2/visual_qa.py": "927709178a42a434b010a50a5cac3d9cd44652453afd784dbb4afe9712139356",
    "configs/denoise_splits_seed20260710.json": "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345",
    "configs/denoise_validation_quarantine_v1.json": "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215",
}

EXPECTED_ASSET_HASHES = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}

ALLOWED_STATUS = {
    "promotion_gate_passed",
    "confirmed_small_gain_no_promotion",
    "promotion_gate_failed",
}
REQUIRED_OUTPUTS = (
    REPORT_NAME,
    GLOBAL_MANIFEST_NAME,
    TARGET_MARKER_NAME,
    PHASE_A_ARCHIVE_NAME,
    "qap_weight_confirmation_tests.log",
    "qap_weight_phase_a_gpu0.log",
    "qap_weight_phase_a_gpu1.log",
    "qap_weight_finalize_phase_a.log",
    "qap_weight_phase_b_gpu0.log",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def names_sha256(names: list[str]) -> str:
    return sha256_bytes("\n".join(names).encode("utf-8"))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode(
        "utf-8"
    )
    atomic_bytes(path, encoded)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def load_exact_envelope(path: Path, expected_file_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular canonical envelope: {path}")
    file_sha256 = sha256(path)
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
        raise RuntimeError(f"canonical envelope SHA256 anchor mismatch: {path}")
    envelope = load_json(path)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise RuntimeError(f"non-canonical envelope keys: {path}")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"canonical envelope payload is not an object: {path}")
    if envelope.get("payload_sha256") != sha256_bytes(canonical_bytes(payload)):
        raise RuntimeError(f"canonical envelope payload SHA256 mismatch: {path}")
    if path.read_bytes() != canonical_bytes(envelope) + b"\n":
        raise RuntimeError(f"non-canonical envelope encoding: {path}")
    return payload, file_sha256


def exactly_one(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {values}")
    return values[0]


def require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing exact Kaggle {label} mount: {path}")
    return path


def require_final_hashes() -> None:
    pending = {
        relative: digest
        for relative, digest in EXPECTED_OVERLAY_HASHES.items()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
    }
    if pending:
        raise RuntimeError(f"runner contains unpinned overlay SHA256 values: {pending}")
    if (
        len(EXPECTED_OVERLAY_ARCHIVE_SHA256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in EXPECTED_OVERLAY_ARCHIVE_SHA256
        )
    ):
        raise RuntimeError("runner contains an unpinned overlay archive SHA256")


def exact_hashes(
    root: Path, expected: dict[str, str], label: str
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative, expected_digest in sorted(expected.items()):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{label} missing regular file {relative}")
        actual = sha256(path)
        if actual != expected_digest:
            raise RuntimeError(
                f"{label} hash mismatch for {relative}: {actual} != {expected_digest}"
            )
        records[relative] = {"sha256": actual, "bytes": path.stat().st_size}
    return records


def safe_extract(archive: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            name = member.filename
            mode = (member.external_attr >> 16) & 0o170000
            if name.startswith("/") or ".." in Path(name).parts or mode == 0o120000:
                raise RuntimeError(f"unsafe archive member: {name}")
        handle.extractall(destination)
    return destination


def find_base_source_root() -> tuple[Path, dict[str, Any]]:
    mount = require_directory(BASE_INPUT, "solver base")
    roots = {
        path.parents[2]
        for path in mount.glob("**/src/puzzle_assembly/qap.py")
        if path.is_file()
    }
    if len(roots) == 1:
        root = roots.pop()
        return root, {"mode": "direct", "source_root": str(root)}
    if roots:
        raise RuntimeError(f"ambiguous direct solver roots: {sorted(roots)}")
    archive = exactly_one(list(mount.glob("**/solver_rework_code.zip")), "solver archive")
    actual = sha256(archive)
    if actual != EXPECTED_BASE_ARCHIVE_SHA256:
        raise RuntimeError(
            f"solver archive hash mismatch: {actual} != {EXPECTED_BASE_ARCHIVE_SHA256}"
        )
    extracted = safe_extract(archive, STAGING / "base_archive")
    roots = {
        path.parents[2]
        for path in extracted.glob("**/src/puzzle_assembly/qap.py")
        if path.is_file()
    }
    if len(roots) != 1:
        raise RuntimeError(f"ambiguous archived solver roots: {sorted(roots)}")
    root = roots.pop()
    return root, {"mode": "archive", "archive_sha256": actual}


def stage_code() -> tuple[Path, dict[str, Any]]:
    source_root, base_mode = find_base_source_root()
    exact_hashes(source_root, EXPECTED_BASE_HASHES, "read-only base")
    code_root = STAGING / "code"
    if code_root.exists():
        shutil.rmtree(code_root)
    code_root.mkdir(parents=True)
    for directory in ("src", "configs"):
        source = source_root / directory
        if not source.is_dir():
            raise FileNotFoundError(f"base code lacks {directory}/")
        shutil.copytree(source, code_root / directory)
    base_records = exact_hashes(code_root, EXPECTED_BASE_HASHES, "writable base copy")

    overlay_mount = require_directory(OVERLAY_INPUT, "QAP confirmation overlay")
    direct_evaluator = overlay_mount / "scripts" / "evaluate_qap_weight_confirmation.py"
    archive = overlay_mount / "qap_weight_confirmation_code.zip"
    if direct_evaluator.is_file() and not archive.exists():
        # Kaggle expands uploaded ZIP contents into dataset files.  The exact
        # per-file allowlist and SHA closure below remains authoritative.
        overlay_root = overlay_mount.resolve()
        overlay_source = {"mode": "direct_dataset_files"}
    elif archive.is_file() and not direct_evaluator.exists():
        archive_sha256 = sha256(archive)
        if archive_sha256 != EXPECTED_OVERLAY_ARCHIVE_SHA256:
            raise RuntimeError(
                "QAP confirmation overlay archive SHA256 mismatch: "
                f"{archive_sha256} != {EXPECTED_OVERLAY_ARCHIVE_SHA256}"
            )
        overlay_root = safe_extract(archive, STAGING / "overlay_archive")
        overlay_source = {
            "mode": "pinned_archive",
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
        }
    else:
        raise RuntimeError(
            "QAP confirmation overlay must expose exactly one direct tree or pinned archive"
        )
    files: set[str] = set()
    for path in overlay_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"overlay contains symlink: {path}")
        if path.is_file():
            files.add(path.relative_to(overlay_root).as_posix())
    expected_files = set(EXPECTED_OVERLAY_HASHES)
    if files != expected_files:
        raise RuntimeError(
            "overlay allowlist mismatch: "
            f"missing={sorted(expected_files - files)}, extra={sorted(files - expected_files)}"
        )
    overlay_records = exact_hashes(
        overlay_root, EXPECTED_OVERLAY_HASHES, "read-only overlay"
    )
    for relative in EXPECTED_OVERLAY_HASHES:
        source = overlay_root / relative
        destination = code_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    staged_overlay = exact_hashes(code_root, EXPECTED_OVERLAY_HASHES, "staged overlay")
    return code_root, {
        "base": base_mode,
        "base_hashes": base_records,
        "overlay_root": str(overlay_root),
        "overlay_source": overlay_source,
        "overlay_hashes": overlay_records,
        "staged_overlay_hashes": staged_overlay,
    }


def find_data_root() -> Path:
    mount = require_directory(PUZZLE_INPUT, "puzzle data")
    inputs = mount / "train" / "inputs"
    if not inputs.is_dir() or len(list(inputs.glob("*.png"))) != 7000:
        raise RuntimeError(f"exact puzzle mount does not contain 7000 inputs: {inputs}")
    return mount


def make_input_only_data_root(data_root: Path) -> Path:
    root = STAGING / "input_only_data"
    if root.exists() or root.is_symlink():
        if root.is_dir() and not root.is_symlink():
            shutil.rmtree(root)
        else:
            root.unlink()
    (root / "train").mkdir(parents=True)
    os.symlink(data_root / "train" / "inputs", root / "train" / "inputs")
    if sorted(path.name for path in (root / "train").iterdir()) != ["inputs"]:
        raise RuntimeError("input-only Phase-A root contains an unexpected train entry")
    return root


def find_asset(filename: str) -> Path:
    mount = require_directory(RUNTIME_INPUT, "assembly runtime")
    asset = exactly_one(
        [path for path in mount.glob(f"**/{filename}") if path.is_file()], filename
    )
    actual = sha256(asset)
    expected = EXPECTED_ASSET_HASHES[filename]
    if actual != expected:
        raise RuntimeError(f"asset hash mismatch for {filename}: {actual} != {expected}")
    return asset


def stage_configured_asset_aliases(
    *,
    code_root: Path,
    config: dict[str, Any],
    actual_paths: dict[str, Path],
) -> dict[str, str]:
    """Make evaluator self-tests resolve the pinned config paths offline."""
    aliases: dict[str, str] = {}
    for label, actual in actual_paths.items():
        configured = Path(str(config["assets"][label]))
        if configured.is_absolute() or ".." in configured.parts:
            raise RuntimeError(f"configured {label} path is not a safe repo-relative path")
        destination = code_root / configured
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if not destination.is_file() or sha256(destination) != sha256(actual):
                raise RuntimeError(f"configured {label} alias collides with different bytes")
        else:
            os.symlink(actual.resolve(), destination)
        aliases[label] = str(destination)
    return aliases


def sanitized_environment() -> dict[str, str]:
    blocked_fragments = (
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "CREDENTIAL",
        "PRIVATE_KEY",
        "API_KEY",
        "AUTHORIZATION",
    )
    return {
        key: value
        for key, value in os.environ.items()
        if not any(fragment in key.upper() for fragment in blocked_fragments)
    }


def gpu_preflight() -> dict[str, Any]:
    code = r'''import json, sys, numpy, scipy, skimage, PIL, torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
count = torch.cuda.device_count()
if count != 2:
    raise RuntimeError(f"expected exactly two visible GPUs, found {count}")
devices = []
for index in range(count):
    name = torch.cuda.get_device_name(index)
    if "T4" not in name.upper():
        raise RuntimeError(f"GPU {index} is not a T4: {name}")
    device = torch.device("cuda", index)
    left = torch.arange(4096, device=device, dtype=torch.float32).reshape(64, 64)
    product = left @ left.T
    devices.append({
        "index": index,
        "name": name,
        "capability": list(torch.cuda.get_device_capability(index)),
        "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
        "tensor_probe": float(product.mean().item()),
    })
print(json.dumps({
    "python": sys.version,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "libraries": {
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "pillow": PIL.__version__,
    },
    "device_count": count,
    "devices": devices,
}))'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env=sanitized_environment(),
    )
    payload = json.loads(completed.stdout)
    payload["nvidia_smi"] = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=sanitized_environment(),
    ).stdout.strip().splitlines()
    if len(payload["nvidia_smi"]) != 2:
        raise RuntimeError("nvidia-smi and torch disagree on the two-GPU contract")
    return payload


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    label: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    returncode: int | None = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        log.flush()
        os.fsync(log.fileno())
    return {
        "label": label,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "seconds": time.perf_counter() - started,
        "log": log_path.name,
        "log_sha256": sha256(log_path),
    }


def require_success(record: dict[str, Any], log_path: Path) -> None:
    if record.get("returncode") == 0 and record.get("timed_out") is False:
        return
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
    raise RuntimeError(
        f"{record['label']} failed: returncode={record.get('returncode')}, "
        f"timed_out={record.get('timed_out')}\n" + "\n".join(tail)
    )


def tree_sha256(root: Path, *, ignore_names: set[str] | None = None) -> str:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"expected regular tree: {root}")
    ignored = ignore_names or set()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in ignored:
            continue
        if path.is_symlink():
            raise RuntimeError(f"frozen Phase-A tree contains symlink: {relative}")
        if path.is_file():
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(str(path.stat().st_size).encode("ascii") + b"\0")
            digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def freeze_tree(root: Path, *, keep_root_writable: bool = False) -> str:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"cannot freeze symlinked output: {path}")
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o755 if keep_root_writable else 0o555)
    return tree_sha256(root)


def write_deterministic_phase_a_archive(
    archive_path: Path, roots: list[tuple[str, Path]]
) -> dict[str, Any]:
    """Persist every frozen layout/render byte with deterministic metadata."""
    temporary = archive_path.with_name(f".{archive_path.name}.tmp-{os.getpid()}")
    member_records: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            strict_timestamps=True,
        ) as archive:
            for prefix, root in roots:
                for path in sorted(
                    (value for value in root.rglob("*") if value.is_file()),
                    key=lambda value: value.relative_to(root).as_posix(),
                ):
                    if path.is_symlink():
                        raise RuntimeError(f"cannot archive symlinked Phase-A artifact: {path}")
                    relative = f"{prefix}/{path.relative_to(root).as_posix()}"
                    info = zipfile.ZipInfo(relative, date_time=ARCHIVE_TIMESTAMP)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = 0o100444 << 16
                    with path.open("rb") as source, archive.open(info, "w") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    member_records.append(
                        {
                            "path": relative,
                            "source_path": str(path.resolve()),
                            "bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                    )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, archive_path)
        _fsync_directory(archive_path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if names != [record["path"] for record in member_records] or len(names) != len(set(names)):
            raise RuntimeError("Phase-A archive member order/uniqueness mismatch")
        for record in member_records:
            payload = archive.read(record["path"])
            if len(payload) != record["bytes"] or sha256_bytes(payload) != record["sha256"]:
                raise RuntimeError(f"Phase-A archive readback mismatch: {record['path']}")
    return {
        "path": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": sha256(archive_path),
        "member_count": len(member_records),
        "members": member_records,
    }


def validate_phase_a_manifests(
    phase_dirs: list[Path], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    manifests: list[dict[str, Any]] = []
    anchors: list[str] = []
    indexed_names: dict[int, str] = {}
    for shard_index, directory in enumerate(phase_dirs):
        marker = directory / TARGET_MARKER_NAME
        if marker.exists():
            raise RuntimeError(f"Phase-A shard {shard_index} wrote target-access marker")
        path = directory / SHARD_MANIFEST_NAME
        if not path.is_file():
            raise RuntimeError(f"Phase-A shard {shard_index} lacks frozen manifest")
        payload, anchor = load_exact_envelope(path)
        if payload.get("kind") != "qap_weight_confirmation_phase_a_shard":
            raise RuntimeError(f"Phase-A shard {shard_index} has wrong kind")
        if payload.get("rank") != shard_index or payload.get("world_size") != 2:
            raise RuntimeError(f"Phase-A shard {shard_index} identity mismatch")
        if (
            payload.get("target_paths_constructed") is not False
            or payload.get("target_files_opened") is not False
        ):
            raise RuntimeError(f"Phase-A shard {shard_index} does not attest zero target reads")
        sources = payload.get("records")
        if not isinstance(sources, list) or len(sources) != 32:
            raise RuntimeError(f"Phase-A shard {shard_index} must contain 32 sources")
        for source in sources:
            if not isinstance(source, dict):
                raise RuntimeError("Phase-A source records must be objects")
            source_index = source.get("source_index")
            name = source.get("name")
            if (
                not isinstance(source_index, int)
                or source_index < 0
                or source_index >= 64
                or source_index % 2 != shard_index
                or not isinstance(name, str)
            ):
                raise RuntimeError("invalid or wrongly sharded canonical source index")
            if source_index in indexed_names:
                raise RuntimeError(f"duplicate canonical source index {source_index}")
            indexed_names[source_index] = name
        manifests.append(payload)
        anchors.append(anchor)
    expected = config["original_real_confirmation"]
    expected_indices = list(range(int(expected["count"])))
    if sorted(indexed_names) != expected_indices:
        raise RuntimeError("Phase-A shards have a canonical source-index gap or extra index")
    canonical_names = [indexed_names[index] for index in expected_indices]
    if len(set(canonical_names)) != len(canonical_names):
        raise RuntimeError("Phase-A shards contain duplicate source names")
    if names_sha256(canonical_names) != expected["names_sha256"]:
        raise RuntimeError("canonical Phase-A names hash disagrees with frozen config")
    return manifests, canonical_names, anchors


def validate_finalized_manifest(
    finalized_dir: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    evaluator: Path,
    asset_records: dict[str, dict[str, str]],
    phase_dirs: list[Path],
    phase_payloads: list[dict[str, Any]],
    canonical_names: list[str],
    shard_envelopes: list[str],
) -> tuple[dict[str, Any], str]:
    path = finalized_dir / GLOBAL_MANIFEST_NAME
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("finalize-phase-a did not produce a canonical manifest")
    payload, envelope_sha256 = load_exact_envelope(path)
    if payload.get("kind") != "qap_weight_confirmation_finalized_phase_a":
        raise RuntimeError("canonical Phase-A manifest has wrong kind")
    required = {
        "schema_version": 1,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "code_path": str(evaluator.resolve()),
        "code_sha256": sha256(evaluator),
        "assets": asset_records,
        "split": "assembly_incremental_gate[128:192]",
        "source_names": canonical_names,
        "source_names_sha256": config["original_real_confirmation"]["names_sha256"],
        "source_count": 64,
        "artifact_root": "artifacts",
        "common_solver": config["common_solver"],
        "baseline": config["baseline"],
        "candidate": config["candidate"],
    }
    for key, expected_value in required.items():
        if payload.get(key) != expected_value:
            raise RuntimeError(f"canonical Phase-A binding mismatch: {key}")
    if (
        payload.get("target_paths_constructed") is not False
        or payload.get("target_files_opened") is not False
    ):
        raise RuntimeError("canonical Phase-A manifest does not attest zero target reads")
    shards = payload.get("shards")
    if not isinstance(shards, list) or [item.get("manifest_sha256") for item in shards] != shard_envelopes:
        raise RuntimeError("canonical Phase-A manifest changed shard-envelope ordering")
    for rank, shard in enumerate(shards):
        shard_path = (phase_dirs[rank] / SHARD_MANIFEST_NAME).resolve()
        artifact_snapshot: dict[str, str] = {}
        for record in phase_payloads[rank]["records"]:
            for variant in record["variants"].values():
                for path_key in ("layout_path", "render_path"):
                    relative = Path(str(variant[path_key]))
                    relative_key = relative.as_posix()
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or relative_key in artifact_snapshot
                    ):
                        raise RuntimeError("invalid or duplicate relative shard artifact key")
                    artifact_path = (phase_dirs[rank] / relative).resolve()
                    artifact_snapshot[relative_key] = sha256(artifact_path)
        expected_shard = {
            "rank": rank,
            "world_size": 2,
            "manifest_path": str(shard_path),
            "manifest_sha256": shard_envelopes[rank],
            "payload_sha256": sha256_bytes(canonical_bytes(phase_payloads[rank])),
            "artifact_snapshot_sha256": sha256_bytes(
                canonical_bytes(artifact_snapshot)
            ),
        }
        for key, expected_value in expected_shard.items():
            if shard.get(key) != expected_value:
                raise RuntimeError(f"canonical Phase-A shard binding mismatch: {rank}/{key}")
    sources = payload.get("records")
    if not isinstance(sources, list) or len(sources) != 64:
        raise RuntimeError("canonical Phase-A manifest must contain 64 records")
    indices = [record.get("source_index") for record in sources if isinstance(record, dict)]
    names = [record.get("name") for record in sources if isinstance(record, dict)]
    if indices != list(range(64)) or names != canonical_names:
        raise RuntimeError("canonical Phase-A manifest is not in authoritative source order")
    expected = config["original_real_confirmation"]
    if names_sha256(names) != expected["names_sha256"]:
        raise RuntimeError("canonical Phase-A manifest names hash mismatch")
    expected_records = sorted(
        [record for shard in phase_payloads for record in shard["records"]],
        key=lambda record: int(record["source_index"]),
    )
    if sources != expected_records:
        raise RuntimeError("canonical Phase-A records differ from frozen shard records")
    if payload.get("final_audit_opened") is not False or payload.get(
        "confirmation_audit_opened"
    ) is not False:
        raise RuntimeError("canonical Phase-A manifest opened a sealed audit")
    return payload, envelope_sha256


def validate_finalized_artifacts(
    *,
    finalized_dir: Path,
    payload: dict[str, Any],
    phase_dirs: list[Path],
) -> None:
    if payload.get("artifact_root") != "artifacts":
        raise RuntimeError("finalized Phase-A artifact root is not relative")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise RuntimeError("finalized Phase-A artifact records are missing")
    for record in records:
        source_index = int(record["source_index"])
        shard_dir = phase_dirs[source_index % 2]
        for key in ("baseline", "candidate"):
            variant = record["variants"][key]
            for path_key, hash_key in (
                ("layout_path", "layout_sha256"),
                ("render_path", "render_sha256"),
            ):
                relative = Path(str(variant[path_key]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("finalized Phase-A artifact path escaped")
                finalized_path = (finalized_dir / relative).resolve()
                shard_path = (shard_dir / relative).resolve()
                expected_parent = (finalized_dir / "artifacts").resolve()
                if finalized_path.parent != expected_parent:
                    raise RuntimeError("finalized Phase-A artifact is not contained")
                if not finalized_path.is_file() or finalized_path.is_symlink():
                    raise RuntimeError("finalized Phase-A artifact is not a regular copied file")
                expected_sha256 = str(variant[hash_key])
                if (
                    sha256(finalized_path) != expected_sha256
                    or sha256(shard_path) != expected_sha256
                ):
                    raise RuntimeError("finalized Phase-A artifact copy differs from frozen shard")


def validate_relocated_finalized_tree(
    finalized_dir: Path, *, expected_manifest_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    """Verify a finalized tree using no paths or bytes outside that tree."""
    root = finalized_dir.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("relocated finalized Phase-A root is not a regular directory")
    top_level = sorted(path.name for path in root.iterdir())
    if top_level != [GLOBAL_MANIFEST_NAME, "artifacts"]:
        raise RuntimeError(f"relocated finalized Phase-A tree has extra entries: {top_level}")
    payload, manifest_sha256 = load_exact_envelope(
        root / GLOBAL_MANIFEST_NAME, expected_manifest_sha256
    )
    if payload.get("artifact_root") != "artifacts":
        raise RuntimeError("relocated finalized manifest has a non-relative artifact root")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise RuntimeError("relocated finalized manifest must contain 64 records")
    expected_files: set[str] = set()
    names: list[str] = []
    for source_index, record in enumerate(records):
        if record.get("source_index") != source_index:
            raise RuntimeError("relocated finalized source-index order mismatch")
        name = str(record.get("name"))
        names.append(name)
        if record.get("input_path") != f"train/inputs/{name}":
            raise RuntimeError("relocated finalized input path is not canonical relative")
        stem = Path(name).stem
        variants = record.get("variants")
        if not isinstance(variants, dict) or set(variants) != {"baseline", "candidate"}:
            raise RuntimeError("relocated finalized variants mismatch")
        for key in ("baseline", "candidate"):
            variant = variants[key]
            specifications = (
                (
                    "layout_path",
                    "layout_sha256",
                    f"artifacts/{stem}.{key}.layout.npy",
                ),
                ("render_path", "render_sha256", f"artifacts/{stem}.{key}.png"),
            )
            for path_key, hash_key, expected_relative in specifications:
                relative = Path(str(variant.get(path_key)))
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or relative.as_posix() != expected_relative
                    or expected_relative in expected_files
                ):
                    raise RuntimeError("relocated finalized artifact path is invalid")
                path = (root / relative).resolve()
                if path.parent != (root / "artifacts").resolve() or not path.is_file() or path.is_symlink():
                    raise RuntimeError("relocated finalized artifact escaped containment")
                if sha256(path) != variant.get(hash_key):
                    raise RuntimeError("relocated finalized artifact hash mismatch")
                expected_files.add(expected_relative)
            layout_payload = (root / str(variant["layout_path"])).read_bytes()
            layout = np.load(BytesIO(layout_payload), allow_pickle=False)
            if (
                layout.shape != (576,)
                or len(np.unique(layout)) != 576
                or int(layout.min()) != 0
                or int(layout.max()) != 575
                or sha256_bytes(np.asarray(layout, dtype=np.int32).tobytes())
                != variant.get("layout_value_sha256")
            ):
                raise RuntimeError("relocated finalized layout value mismatch")
            with Image.open(root / str(variant["render_path"])) as image:
                render = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if render.shape != (480, 480, 3):
                raise RuntimeError("relocated finalized render shape mismatch")
    if len(set(names)) != 64 or names_sha256(names) != payload.get("source_names_sha256"):
        raise RuntimeError("relocated finalized source-name binding mismatch")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in (root / "artifacts").rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError("relocated finalized artifact allowlist mismatch")
    return payload, manifest_sha256


def filename_qap_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little") + 7001


def validate_phase_a_artifacts(
    *,
    payloads: list[dict[str, Any]],
    phase_dirs: list[Path],
    config: dict[str, Any],
    config_path: Path,
    evaluator: Path,
    data_root: Path,
    asset_records: dict[str, dict[str, str]],
) -> None:
    input_root = (data_root / "train" / "inputs").resolve()
    expected_code_sha256 = sha256(evaluator)
    for rank, (payload, phase_dir) in enumerate(zip(payloads, phase_dirs, strict=True)):
        required = {
            "schema_version": 1,
            "kind": "qap_weight_confirmation_phase_a_shard",
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256(config_path),
            "code_path": str(evaluator.resolve()),
            "code_sha256": expected_code_sha256,
            "assets": asset_records,
            "split": "assembly_incremental_gate[128:192]",
            "source_names_sha256": config["original_real_confirmation"]["names_sha256"],
            "source_count_total": 64,
            "rank": rank,
            "world_size": 2,
            "assigned_source_indices": list(range(rank, 64, 2)),
            "artifact_root": "artifacts",
            "common_solver": config["common_solver"],
            "baseline": config["baseline"],
            "candidate": config["candidate"],
            "target_paths_constructed": False,
            "target_files_opened": False,
        }
        for key, expected_value in required.items():
            if payload.get(key) != expected_value:
                raise RuntimeError(f"Phase-A shard {rank} binding mismatch: {key}")
        records = payload.get("records")
        if not isinstance(records, list) or len(records) != 32:
            raise RuntimeError(f"Phase-A shard {rank} record count mismatch")
        for record in records:
            source_index = int(record["source_index"])
            name = str(record["name"])
            expected_input = (input_root / name).resolve()
            if expected_input.parent != input_root or record.get("input_path") != f"train/inputs/{name}":
                raise RuntimeError(f"Phase-A input path mismatch: {name}")
            if record.get("input_sha256") != sha256(expected_input):
                raise RuntimeError(f"Phase-A input bytes changed: {name}")
            expected_seed = filename_qap_seed(name)
            if record.get("qap_seed") != expected_seed:
                raise RuntimeError(f"Phase-A QAP seed mismatch: {name}")
            variants = record.get("variants")
            if not isinstance(variants, dict) or set(variants) != {"baseline", "candidate"}:
                raise RuntimeError(f"Phase-A variants mismatch: {name}")
            initial_hashes: set[str] = set()
            for key in ("baseline", "candidate"):
                variant = variants[key]
                spec = config[key]
                for field in ("label", "score", "hbt_weight"):
                    if variant.get(field) != spec[field]:
                        raise RuntimeError(f"Phase-A {key} {field} drift: {name}")
                stem = Path(name).stem
                artifact_root = (phase_dir / "artifacts").resolve()
                layout_relative = Path(str(variant.get("layout_path")))
                render_relative = Path(str(variant.get("render_path")))
                if (
                    layout_relative.is_absolute()
                    or render_relative.is_absolute()
                    or ".." in layout_relative.parts
                    or ".." in render_relative.parts
                ):
                    raise RuntimeError(f"Phase-A artifact path is not relative: {name}/{key}")
                layout_path = (phase_dir / layout_relative).resolve()
                render_path = (phase_dir / render_relative).resolve()
                if (
                    layout_path.parent != artifact_root
                    or layout_relative.as_posix() != f"artifacts/{stem}.{key}.layout.npy"
                    or layout_path.name != f"{stem}.{key}.layout.npy"
                    or render_path.parent != artifact_root
                    or render_relative.as_posix() != f"artifacts/{stem}.{key}.png"
                    or render_path.name != f"{stem}.{key}.png"
                ):
                    raise RuntimeError(f"Phase-A artifact path escaped shard: {name}/{key}")
                layout_payload = layout_path.read_bytes()
                render_payload = render_path.read_bytes()
                if sha256_bytes(layout_payload) != variant.get("layout_sha256"):
                    raise RuntimeError(f"Phase-A layout hash mismatch: {name}/{key}")
                if sha256_bytes(render_payload) != variant.get("render_sha256"):
                    raise RuntimeError(f"Phase-A render hash mismatch: {name}/{key}")
                layout = np.load(BytesIO(layout_payload), allow_pickle=False)
                if (
                    layout.shape != (576,)
                    or len(np.unique(layout)) != 576
                    or int(layout.min()) != 0
                    or int(layout.max()) != 575
                ):
                    raise RuntimeError(f"Phase-A invalid layout: {name}/{key}")
                value_hash = sha256_bytes(
                    np.asarray(layout, dtype=np.int32).tobytes()
                )
                if value_hash != variant.get("layout_value_sha256"):
                    raise RuntimeError(f"Phase-A layout value hash mismatch: {name}/{key}")
                with Image.open(BytesIO(render_payload)) as image:
                    render = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if render.shape != (480, 480, 3):
                    raise RuntimeError(f"Phase-A invalid render: {name}/{key}")
                if variant.get("valid_permutation") is not True:
                    raise RuntimeError(f"Phase-A validity flag missing: {name}/{key}")
                if variant.get("qap_seed") != expected_seed:
                    raise RuntimeError(f"Phase-A variant QAP seed drift: {name}/{key}")
                initial_hashes.add(str(variant.get("initial_layout_value_sha256")))
            if initial_hashes != {str(record.get("initial_layout_value_sha256"))}:
                raise RuntimeError(f"Phase-A variants do not share one initializer: {name}")


def close(value: Any, expected: float, *, tolerance: float = 1e-12) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and abs(numeric - expected) <= tolerance


def validate_final_report(
    report_path: Path,
    *,
    config_path: Path,
    evaluator: Path,
    asset_records: dict[str, dict[str, str]],
    finalized_manifest_path: Path,
    target_marker_path: Path,
    combined_names: list[str],
) -> dict[str, Any]:
    report = load_json(report_path)
    config = load_json(config_path)
    if report_path.read_bytes() != canonical_bytes(report) + b"\n":
        raise RuntimeError("QAP confirmation report is not canonical JSON")
    if report.get("schema_version") != 1 or report.get("kind") != "qap_weight_confirmation_report":
        raise RuntimeError("unexpected QAP confirmation report schema")
    status = report.get("status")
    if status not in ALLOWED_STATUS:
        raise RuntimeError(f"unexpected QAP confirmation status: {status}")
    if report.get("safe_for_submission") is not False:
        raise RuntimeError("confirmation report is not fail-closed for submission")
    if report.get("config") != {
        "path": str(config_path.resolve()),
        "sha256": sha256(config_path),
    }:
        raise RuntimeError("report config SHA256 mismatch")
    expected = config["original_real_confirmation"]
    expected_code = {"path": str(evaluator.resolve()), "sha256": sha256(evaluator)}
    if report.get("code") != expected_code:
        raise RuntimeError("report evaluator code binding mismatch")
    if report.get("split") != "assembly_incremental_gate[128:192]":
        raise RuntimeError("report split disagrees with frozen protocol")
    if report.get("source_names") != combined_names:
        raise RuntimeError("report source ordering differs from frozen Phase-A ordering")
    if report.get("source_names_sha256") != expected["names_sha256"] or names_sha256(
        combined_names
    ) != expected["names_sha256"]:
        raise RuntimeError("report source-name hash is not reproducible")
    expected_protocol_report = {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "split": "assembly_incremental_gate",
        "offset": 128,
        "count": 64,
        "source_names_sha256": expected["names_sha256"],
    }
    if report.get("protocol") != expected_protocol_report:
        raise RuntimeError("report protocol alias is inconsistent")
    if report.get("assets") != asset_records:
        raise RuntimeError("report asset binding mismatch")
    if report.get("baseline") != config["baseline"] or report.get("candidate") != config["candidate"]:
        raise RuntimeError("baseline/candidate identity drifted from frozen config")
    if report.get("common_solver") != config["common_solver"]:
        raise RuntimeError("common solver parameters drifted from frozen config")
    if report.get("solver") != {
        "common": config["common_solver"],
        "baseline": config["baseline"],
        "candidate": config["candidate"],
    }:
        raise RuntimeError("report solver alias is inconsistent")

    finalized_payload, finalized_sha256 = load_exact_envelope(finalized_manifest_path)
    finalized_payload_sha256 = sha256_bytes(canonical_bytes(finalized_payload))
    phase_a = report.get("phase_a")
    if not isinstance(phase_a, dict) or phase_a.get("manifest") != str(
        finalized_manifest_path.resolve()
    ):
        raise RuntimeError("report lacks the canonical finalized Phase-A manifest")
    if phase_a.get("manifest_sha256") != finalized_sha256 or phase_a.get(
        "payload_sha256"
    ) != finalized_payload_sha256:
        raise RuntimeError("report finalized Phase-A envelope binding mismatch")
    if phase_a.get("integrity_before_sha256") != phase_a.get("integrity_after_sha256"):
        raise RuntimeError("report Phase-A integrity changed during target scoring")
    for key in ("integrity_before_sha256", "integrity_after_sha256"):
        value = phase_a.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError("report lacks valid Phase-A integrity hashes")
    if (
        phase_a.get("source_count") != 64
        or phase_a.get("source_names_sha256") != expected["names_sha256"]
        or phase_a.get("shards") != finalized_payload.get("shards")
    ):
        raise RuntimeError("report Phase-A alias metadata is inconsistent")

    marker_payload, marker_sha256 = load_exact_envelope(target_marker_path)
    expected_marker_payload = {
        "schema_version": 1,
        "kind": "qap_weight_confirmation_target_access_event",
        "phase_a_manifest_sha256": finalized_sha256,
        "phase_a_payload_sha256": finalized_payload_sha256,
        "config_sha256": sha256(config_path),
        "code_sha256": sha256(evaluator),
        "source_names_sha256": expected["names_sha256"],
        "target_access_started": True,
        "target_files_may_have_been_opened": True,
    }
    if marker_payload != expected_marker_payload:
        raise RuntimeError("durable target marker payload mismatch")
    expected_target_access = {
        "marker": str(target_marker_path.resolve()),
        "marker_sha256": marker_sha256,
        "marker_payload_sha256": sha256_bytes(canonical_bytes(marker_payload)),
        "marker_preceded_first_target_path_construction": True,
        "target_access_count": 64,
    }
    target_access = report.get("target_access")
    if not isinstance(target_access, dict) or target_access != expected_target_access:
        raise RuntimeError("report target-access provenance mismatch")
    if report.get("phase_b") != {
        **expected_target_access,
        "integrity_before_sha256": phase_a["integrity_before_sha256"],
        "integrity_after_sha256": phase_a["integrity_after_sha256"],
        "post_score_rehash_matched": True,
    }:
        raise RuntimeError("report Phase-B alias is inconsistent")
    sealed = report.get("sealed_sets")
    if sealed != {
        "final_audit_opened": False,
        "confirmation_audit_opened": False,
        "must_remain_unopened": True,
    }:
        raise RuntimeError("report opened or ambiguously represented a sealed audit")
    if report.get("metric") != expected["metric"]:
        raise RuntimeError("report metric/bootstrap contract drifted")
    if report.get("post_phase_b_mutation_policy") != expected[
        "post_phase_b_mutation_policy"
    ]:
        raise RuntimeError("report post-gate mutation policy drifted")

    records = report.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise RuntimeError("report must retain 64 per-source score records")
    if [record.get("source_index") for record in records] != list(range(64)) or [
        record.get("name") for record in records
    ] != combined_names:
        raise RuntimeError("report score records are not in canonical source order")
    manifest_records = finalized_payload.get("records")
    if not isinstance(manifest_records, list) or len(manifest_records) != 64:
        raise RuntimeError("finalized Phase-A manifest record count mismatch")
    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    numeric_deltas: list[float] = []
    for record, frozen in zip(records, manifest_records, strict=True):
        baseline = float(record.get("baseline_ssim"))
        candidate = float(record.get("candidate_ssim"))
        delta = float(record.get("delta_ssim"))
        if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in (baseline, candidate)):
            raise RuntimeError("report contains invalid per-source SSIM")
        if not close(delta, candidate - baseline, tolerance=2e-15):
            raise RuntimeError("report per-source delta is inconsistent")
        if record.get("baseline_layout_sha256") != frozen["variants"]["baseline"].get(
            "layout_value_sha256"
        ) or record.get("candidate_layout_sha256") != frozen["variants"]["candidate"].get(
            "layout_value_sha256"
        ):
            raise RuntimeError("report layout hashes differ from frozen Phase A")
        baseline_scores.append(baseline)
        candidate_scores.append(candidate)
        numeric_deltas.append(delta)

    metrics = report.get("aggregate")
    gate = report.get("gate")
    if not isinstance(metrics, dict) or not isinstance(gate, dict):
        raise RuntimeError("report lacks paired metrics/gate")
    if report.get("paired_metrics") != metrics:
        raise RuntimeError("report paired-metrics alias is inconsistent")
    if metrics.get("source_count") != 64 or metrics.get(
        "bootstrap_unit"
    ) != "paired_whole_source_delta_candidate_minus_baseline":
        raise RuntimeError("aggregate source/bootstrap unit mismatch")
    mean_delta = sum(numeric_deltas) / len(numeric_deltas)
    if not close(metrics.get("mean_baseline_ssim"), float(np.mean(baseline_scores))):
        raise RuntimeError("reported mean baseline SSIM is inconsistent")
    if not close(metrics.get("mean_candidate_ssim"), float(np.mean(candidate_scores))):
        raise RuntimeError("reported mean candidate SSIM is inconsistent")
    if not close(metrics.get("mean_ssim_delta"), mean_delta):
        raise RuntimeError("reported mean SSIM delta is inconsistent with per-source deltas")
    if not close(metrics.get("median_ssim_delta"), float(np.median(numeric_deltas))):
        raise RuntimeError("reported median SSIM delta is inconsistent")
    tie_tolerance = float(expected["metric"]["tie_tolerance"])
    wins = sum(value > tie_tolerance for value in numeric_deltas)
    losses = sum(value < -tie_tolerance for value in numeric_deltas)
    ties = len(numeric_deltas) - wins - losses
    large_regressions = sum(value < -0.01 for value in numeric_deltas)
    for actual, recomputed, label in (
        (metrics.get("wins"), wins, "wins"),
        (metrics.get("losses"), losses, "losses"),
        (metrics.get("ties"), ties, "ties"),
    ):
        if actual != recomputed:
            raise RuntimeError(f"reported {label} is inconsistent with paired deltas")
    if metrics.get("large_regressions") != large_regressions:
        raise RuntimeError("reported large-regression count is inconsistent")
    frozen_valid_permutations = sum(
        frozen["variants"]["baseline"].get("valid_permutation") is True
        and frozen["variants"]["candidate"].get("valid_permutation") is True
        for frozen in manifest_records
    )
    if (
        metrics.get("valid_permutation_count") != frozen_valid_permutations
        or frozen_valid_permutations != len(numeric_deltas)
    ):
        raise RuntimeError("all baseline and candidate layouts must be valid permutations")
    interval = metrics.get("bootstrap_95_ci")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in interval)
        or float(interval[0]) > float(interval[1])
    ):
        raise RuntimeError("invalid bootstrap confidence interval")
    metric_contract = expected["metric"]
    bootstrap_rng = np.random.default_rng(int(metric_contract["bootstrap_seed"]))
    bootstrap_indices = bootstrap_rng.integers(
        0,
        len(numeric_deltas),
        size=(int(metric_contract["bootstrap_resamples"]), len(numeric_deltas)),
    )
    bootstrap_means = np.asarray(numeric_deltas, dtype=np.float64)[
        bootstrap_indices
    ].mean(axis=1)
    recomputed_interval = np.quantile(
        bootstrap_means,
        [float(value) for value in metric_contract["bootstrap_quantiles"]],
    ).tolist()
    if not all(close(actual, expected_value) for actual, expected_value in zip(interval, recomputed_interval)):
        raise RuntimeError("reported bootstrap interval is inconsistent with fixed resampling")

    thresholds = expected["gate"]
    checks = {
        "mean_ssim_delta_ge_0.005": mean_delta
        >= float(thresholds["mean_ssim_delta_min"]),
        "bootstrap_95_lower_gt_0": float(interval[0])
        > float(thresholds["ssim_bootstrap_95_lower_gt"]),
        "wins_ge_40": wins >= int(thresholds["wins_min"]),
        "large_regressions_le_6": large_regressions
        <= int(thresholds["large_regressions_max"]),
        "valid_permutation_count_eq_64": metrics.get("valid_permutation_count")
        == int(thresholds["valid_permutation_count"]),
    }
    if gate.get("checks") != checks:
        raise RuntimeError("reported gate checks do not match recomputed checks")
    if gate.get("logic") != "all_of" or thresholds.get("logic") != "all_of":
        raise RuntimeError("promotion gate must use the frozen all_of logic")
    passed = all(checks.values())
    if gate.get("passed") is not passed:
        raise RuntimeError("reported gate decision is inconsistent")
    positive_ci = float(interval[0]) > 0.0
    expected_status = (
        "promotion_gate_passed"
        if passed
        else "confirmed_small_gain_no_promotion"
        if positive_ci
        and mean_delta > 0.0
        and not checks["mean_ssim_delta_ge_0.005"]
        else "promotion_gate_failed"
    )
    if status != expected_status:
        raise RuntimeError(f"report status {status!r} should be {expected_status!r}")
    if report.get("eligible_for_final_audit") is not (status == "promotion_gate_passed"):
        raise RuntimeError("final-audit eligibility is inconsistent with gate status")
    return {
        "path": report_path.name,
        "sha256": sha256(report_path),
        "status": status,
        "mean_ssim_delta": mean_delta,
        "bootstrap_95_ci": interval,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "gate_passed": passed,
        "eligible_for_final_audit": status == "promotion_gate_passed",
        "safe_for_submission": False,
    }


def copy_required_outputs(
    result_dir: Path, finalized_dir: Path, log_dir: Path
) -> list[Path]:
    sources = {
        REPORT_NAME: result_dir / REPORT_NAME,
        GLOBAL_MANIFEST_NAME: finalized_dir / GLOBAL_MANIFEST_NAME,
        TARGET_MARKER_NAME: finalized_dir / TARGET_MARKER_NAME,
        PHASE_A_ARCHIVE_NAME: STAGING / PHASE_A_ARCHIVE_NAME,
        "qap_weight_confirmation_tests.log": log_dir / "qap_weight_confirmation_tests.log",
        "qap_weight_phase_a_gpu0.log": log_dir / "qap_weight_phase_a_gpu0.log",
        "qap_weight_phase_a_gpu1.log": log_dir / "qap_weight_phase_a_gpu1.log",
        "qap_weight_finalize_phase_a.log": log_dir / "qap_weight_finalize_phase_a.log",
        "qap_weight_phase_b_gpu0.log": log_dir / "qap_weight_phase_b_gpu0.log",
    }
    copied: list[Path] = []
    for name in REQUIRED_OUTPUTS:
        source = sources[name]
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"required output missing: {source}")
        destination = WORKING / name
        atomic_copy(source, destination)
        copied.append(destination)
    return copied


def write_sha256s(paths: list[Path]) -> None:
    unique = {path.name: path for path in paths}
    payload = "".join(
        f"{sha256(path)}  {name}\n" for name, path in sorted(unique.items())
    ).encode("utf-8")
    atomic_bytes(SUMS, payload)


def common_evaluator_args(
    *,
    config: Path,
    denoiser: Path,
    hbt: Path,
    manifest: Path,
    quarantine: Path,
) -> list[str]:
    return [
        "--config",
        str(config),
        "--denoiser",
        str(denoiser),
        "--hbt-checkpoint",
        str(hbt),
        "--manifest",
        str(manifest),
        "--quarantine",
        str(quarantine),
    ]


def main() -> None:
    wrapper: dict[str, Any] = {
        "schema_version": 1,
        "kind": "qap_weight_confirmation_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "eligible_for_final_audit": False,
        "started_unix": time.time(),
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "kernel": {
            "id": KERNEL_ID,
            "metadata_sha256": EXPECTED_KERNEL_METADATA_SHA256,
            "dataset_sources": DATASET_SOURCES,
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4",
        },
        "steps": [],
    }
    atomic_json(WRAPPER, wrapper)
    try:
        require_final_hashes()
        if STAGING.exists():
            shutil.rmtree(STAGING)
        STAGING.mkdir(parents=True)
        log_dir = STAGING / "logs"
        result_dir = STAGING / "result"
        result_dir.mkdir(parents=True)

        code_root, code_provenance = stage_code()
        data_root = find_data_root()
        input_only_root = make_input_only_data_root(data_root)
        denoiser = find_asset("selected_tilenaf_synth_50k.pt")
        hbt = find_asset("hbt_d320_denoised_rgb_sobel.pt")
        config = code_root / "configs" / "qap_weight_confirmation_v1.json"
        manifest = code_root / "configs" / "denoise_splits_seed20260710.json"
        quarantine = code_root / "configs" / "denoise_validation_quarantine_v1.json"
        evaluator = code_root / "scripts" / "evaluate_qap_weight_confirmation.py"
        config_payload = load_json(config)
        asset_records = {
            "denoiser": {
                "path": str(denoiser.resolve()),
                "sha256": sha256(denoiser),
                "configured_path": str(config_payload["assets"]["denoiser"]),
            },
            "hbt": {
                "path": str(hbt.resolve()),
                "sha256": sha256(hbt),
                "configured_path": str(config_payload["assets"]["hbt"]),
            },
            "manifest": {
                "path": str(manifest.resolve()),
                "sha256": sha256(manifest),
                "configured_path": str(config_payload["assets"]["manifest"]),
            },
            "quarantine": {
                "path": str(quarantine.resolve()),
                "sha256": sha256(quarantine),
                "configured_path": str(config_payload["assets"]["quarantine"]),
            },
        }
        portable_asset_records = {
            label: {
                "sha256": record["sha256"],
                "configured_path": record["configured_path"],
            }
            for label, record in asset_records.items()
        }
        configured_asset_aliases = stage_configured_asset_aliases(
            code_root=code_root,
            config=config_payload,
            actual_paths={
                "denoiser": denoiser,
                "hbt": hbt,
                "manifest": manifest,
                "quarantine": quarantine,
            },
        )
        hardware = gpu_preflight()
        wrapper.update(
            {
                "status": "staged",
                "hardware": hardware,
                "code": code_provenance,
                "config": {"sha256": sha256(config), "payload": config_payload},
                "assets": {
                    "denoiser": {"sha256": sha256(denoiser), "bytes": denoiser.stat().st_size},
                    "hbt": {"sha256": sha256(hbt), "bytes": hbt.stat().st_size},
                    "manifest": {"sha256": sha256(manifest), "bytes": manifest.stat().st_size},
                    "quarantine": {
                        "sha256": sha256(quarantine),
                        "bytes": quarantine.stat().st_size,
                    },
                },
                "configured_asset_aliases": configured_asset_aliases,
                "anti_leakage": {
                    "phase_a_data_root": "input-only staging root",
                    "phase_a_train_entries": sorted(
                        path.name for path in (input_only_root / "train").iterdir()
                    ),
                    "phase_b_process_count": 1,
                    "phase_b_cuda_visible_devices": "0",
                },
            }
        )
        if wrapper["anti_leakage"]["phase_a_train_entries"] != ["inputs"]:
            raise RuntimeError("Phase-A data root exposes an unexpected train entry")
        atomic_json(WRAPPER, wrapper)

        base_env = sanitized_environment()
        base_env["PYTHONPATH"] = str(code_root / "src")
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            base_env[variable] = "1"
        common = common_evaluator_args(
            config=config,
            denoiser=denoiser,
            hbt=hbt,
            manifest=manifest,
            quarantine=quarantine,
        )

        tests_log = log_dir / "qap_weight_confirmation_tests.log"
        tests = run_logged(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_qap_weight_confirmation.py",
            ],
            cwd=code_root,
            env=base_env,
            log_path=tests_log,
            label="focused QAP-confirmation tests",
            timeout_seconds=1200,
        )
        wrapper["steps"].append(tests)
        atomic_json(WRAPPER, wrapper)
        require_success(tests, tests_log)

        phase_dirs = [STAGING / "phase_a_shard0", STAGING / "phase_a_shard1"]

        def run_phase_a(shard_index: int) -> tuple[dict[str, Any], Path]:
            directory = phase_dirs[shard_index]
            log_path = log_dir / f"qap_weight_phase_a_gpu{shard_index}.log"
            env = dict(base_env)
            env["CUDA_VISIBLE_DEVICES"] = str(shard_index)
            record = run_logged(
                [
                    sys.executable,
                    str(evaluator),
                    "--action",
                    "phase-a",
                    "--rank",
                    str(shard_index),
                    "--world-size",
                    "2",
                    "--device",
                    "cuda:0",
                    "--phase-a-dir",
                    str(directory),
                    "--data-root",
                    str(input_only_root),
                    *common,
                ],
                cwd=code_root,
                env=env,
                log_path=log_path,
                label=f"input-only Phase A shard {shard_index} on physical GPU {shard_index}",
                timeout_seconds=10800,
            )
            require_success(record, log_path)
            return record, log_path

        phase_records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(run_phase_a, index): index for index in range(2)}
            for future in as_completed(futures):
                record, _ = future.result()
                phase_records.append(record)
        phase_records.sort(key=lambda item: item["label"])
        wrapper["steps"].extend(phase_records)
        phase_manifests, combined_names, evaluator_envelopes = validate_phase_a_manifests(
            phase_dirs, config_payload
        )
        validate_phase_a_artifacts(
            payloads=phase_manifests,
            phase_dirs=phase_dirs,
            config=config_payload,
            config_path=config,
            evaluator=evaluator,
            data_root=data_root,
            asset_records=portable_asset_records,
        )
        shard_trees_before_finalize = [freeze_tree(directory) for directory in phase_dirs]

        finalized_dir = STAGING / "finalized_phase_a"
        finalize_log = log_dir / "qap_weight_finalize_phase_a.log"
        finalize_env = dict(base_env)
        finalize_env["CUDA_VISIBLE_DEVICES"] = ""
        finalize_record = run_logged(
            [
                sys.executable,
                str(evaluator),
                "--action",
                "finalize-phase-a",
                "--phase-a-dirs",
                *(str(directory) for directory in phase_dirs),
                "--phase-a-envelope-sha256s",
                *evaluator_envelopes,
                "--finalized-phase-a-dir",
                str(finalized_dir),
                *common,
            ],
            cwd=code_root,
            env=finalize_env,
            log_path=finalize_log,
            label="canonical CPU-only Phase-A finalization",
            timeout_seconds=1800,
        )
        wrapper["steps"].append(finalize_record)
        atomic_json(WRAPPER, wrapper)
        require_success(finalize_record, finalize_log)
        if [tree_sha256(directory) for directory in phase_dirs] != shard_trees_before_finalize:
            raise RuntimeError("finalize-phase-a mutated a frozen shard tree")
        finalized_manifest, finalized_envelope = validate_finalized_manifest(
            finalized_dir,
            config=config_payload,
            config_path=config,
            evaluator=evaluator,
            asset_records=portable_asset_records,
            phase_dirs=phase_dirs,
            phase_payloads=phase_manifests,
            canonical_names=combined_names,
            shard_envelopes=evaluator_envelopes,
        )
        validate_finalized_artifacts(
            finalized_dir=finalized_dir,
            payload=finalized_manifest,
            phase_dirs=phase_dirs,
        )
        finalized_tree_before_phase_b = freeze_tree(
            finalized_dir, keep_root_writable=True
        )
        archive_provenance_root = STAGING / "archive_shard_manifests"
        archive_shard_roots: list[tuple[str, Path]] = []
        for rank, phase_dir in enumerate(phase_dirs):
            shard_root = archive_provenance_root / f"shard{rank}"
            shard_root.mkdir(parents=True)
            atomic_copy(
                phase_dir / SHARD_MANIFEST_NAME,
                shard_root / SHARD_MANIFEST_NAME,
            )
            archive_shard_roots.append((f"shard{rank}", shard_root))
        phase_a_archive = write_deterministic_phase_a_archive(
            STAGING / PHASE_A_ARCHIVE_NAME,
            [("finalized", finalized_dir), *archive_shard_roots],
        )
        archive_readback = safe_extract(
            STAGING / PHASE_A_ARCHIVE_NAME, STAGING / "phase_a_archive_readback"
        )
        relocated_payload, relocated_manifest_sha256 = validate_relocated_finalized_tree(
            archive_readback / "finalized",
            expected_manifest_sha256=sha256(finalized_dir / GLOBAL_MANIFEST_NAME),
        )
        if relocated_payload != finalized_manifest:
            raise RuntimeError("relocated archive manifest payload changed")
        wrapper["phase_a"] = {
            "process_count": 2,
            "physical_gpus": [0, 1],
            "source_count": len(combined_names),
            "names_sha256": names_sha256(combined_names),
            "shard_envelope_sha256s": evaluator_envelopes,
            "shard_runner_tree_sha256s": shard_trees_before_finalize,
            "finalized_envelope_sha256": finalized_envelope,
            "finalized_manifest_sha256": sha256(finalized_dir / GLOBAL_MANIFEST_NAME),
            "finalized_runner_tree_sha256": finalized_tree_before_phase_b,
            "frozen_archive": phase_a_archive,
            "relocation_readback": {
                "validated_from_extracted_tree_only": True,
                "manifest_sha256": relocated_manifest_sha256,
                "source_count": len(relocated_payload["records"]),
                "shard_manifests_included": 2,
            },
            "target_paths_or_pixels_read": False,
        }
        atomic_json(WRAPPER, wrapper)

        phase_b_log = log_dir / "qap_weight_phase_b_gpu0.log"
        phase_b_env = dict(base_env)
        phase_b_env["CUDA_VISIBLE_DEVICES"] = "0"
        phase_b = run_logged(
            [
                sys.executable,
                str(evaluator),
                "--action",
                "phase-b",
                "--finalized-phase-a-dir",
                str(finalized_dir),
                "--phase-a-envelope-sha256",
                finalized_envelope,
                "--device",
                "cuda:0",
                "--output",
                str(result_dir / REPORT_NAME),
                "--data-root",
                str(data_root),
                *common,
            ],
            cwd=code_root,
            env=phase_b_env,
            log_path=phase_b_log,
            label="canonical target-attaching Phase B on physical GPU 0",
            timeout_seconds=3600,
        )
        wrapper["steps"].append(phase_b)
        atomic_json(WRAPPER, wrapper)
        require_success(phase_b, phase_b_log)
        shard_trees_after = [tree_sha256(directory) for directory in phase_dirs]
        if shard_trees_after != shard_trees_before_finalize:
            raise RuntimeError("Phase B mutated a frozen Phase-A shard tree")
        finalized_tree_after_phase_b = tree_sha256(
            finalized_dir, ignore_names={TARGET_MARKER_NAME}
        )
        if finalized_tree_after_phase_b != finalized_tree_before_phase_b:
            raise RuntimeError("Phase B mutated the finalized Phase-A prediction envelope")
        if sha256(STAGING / PHASE_A_ARCHIVE_NAME) != phase_a_archive["sha256"]:
            raise RuntimeError("Phase B mutated the independently persisted Phase-A archive")

        report_path = result_dir / REPORT_NAME
        global_manifest = finalized_dir / GLOBAL_MANIFEST_NAME
        target_marker = finalized_dir / TARGET_MARKER_NAME
        if not global_manifest.is_file() or not target_marker.is_file():
            raise RuntimeError("Phase B lacks global frozen manifest or durable target marker")
        report_summary = validate_final_report(
            report_path,
            config_path=config,
            evaluator=evaluator,
            asset_records=asset_records,
            finalized_manifest_path=global_manifest,
            target_marker_path=target_marker,
            combined_names=combined_names,
        )
        outputs = copy_required_outputs(result_dir, finalized_dir, log_dir)
        wrapper.update(
            {
                "status": "complete",
                "completed_unix": time.time(),
                "phase_a_shard_runner_tree_sha256s_after": shard_trees_after,
                "phase_a_finalized_runner_tree_sha256_after": finalized_tree_after_phase_b,
                "report": report_summary,
                "safe_for_submission": False,
                "eligible_for_final_audit": report_summary["eligible_for_final_audit"],
            }
        )
        atomic_json(WRAPPER, wrapper)
        write_sha256s([*outputs, WRAPPER])
        print(json.dumps(wrapper, indent=2, sort_keys=True, default=str), flush=True)
    except BaseException as error:
        wrapper.update(
            {
                "status": "failed",
                "failed_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "safe_for_submission": False,
                "eligible_for_final_audit": False,
            }
        )
        atomic_json(WRAPPER, wrapper)
        failure_paths = [WRAPPER]
        if STAGING.is_dir():
            for name in (
                "qap_weight_confirmation_tests.log",
                "qap_weight_phase_a_gpu0.log",
                "qap_weight_phase_a_gpu1.log",
                "qap_weight_finalize_phase_a.log",
                "qap_weight_phase_b_gpu0.log",
            ):
                source = STAGING / "logs" / name
                if source.is_file():
                    destination = WORKING / name
                    atomic_copy(source, destination)
                    failure_paths.append(destination)
        write_sha256s(failure_paths)
        raise
    finally:
        if STAGING.exists():
            # Phase-A trees were chmod read-only; restore owner write bits so
            # cleanup succeeds and Kaggle exports only the audited artifacts.
            for path in sorted(STAGING.rglob("*"), reverse=True):
                if path.is_dir() and not path.is_symlink():
                    path.chmod(0o755)
                elif path.is_file() and not path.is_symlink():
                    path.chmod(0o644)
            shutil.rmtree(STAGING)


if __name__ == "__main__":
    main()
