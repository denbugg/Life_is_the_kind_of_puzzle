"""Durable M144 DCT-Where four-arm trainer and evaluator.

This module is intentionally specific to the M144 capability gate.  It does
not reuse the generic E26 stage runner and it never reads the test split.  The
scientific experiment is fixed as follows:

* cache frozen dirty-tile embeddings from the authenticated paired-alignment
  checkpoint;
* train ``dct_full``, ``dct_blind``, ``rgb_full`` and ``rgb_blind`` for 2,500
  paired steps on the same stateless FIT minibatch schedule;
* atomically checkpoint all four arms every 100 steps;
* evaluate CAL and open DEV only when the predeclared CAL gate passes; and
* leave numeric per-board evidence, receipts, status and a resumable checkpoint
  below an E:-drive work root.

The runner owns orchestration, IO and optimization only.  DCT/RGB rendering,
the set model, SSIM and bootstrap statistics live in :mod:`m144_dct_where`.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

import m144_dct_where as core
from eval_paired_alignment import (
    PairedAlignment,
    rank_of_diagonal,
    symmetric_info_nce,
)
from distort import distort_frags
from imgio import load, to_frags


RUN_SCHEMA = "pazzle-m144-dct-where-run-contract-v2"
STATUS_SCHEMA = "pazzle-m144-dct-where-status-v1"
CACHE_PROGRESS_SCHEMA = "pazzle-m144-dct-where-cache-progress-v1"
CACHE_SCHEMA = "pazzle-m144-dct-where-embedding-cache-v2"
TARGET_RECEIPT_SCHEMA = "pazzle-m144-dct-where-target-receipt-v1"
SWAP_WHITENING_SCHEMA = "pazzle-m144-dct-where-swap-whitening-v1"
SWAP_RECEIPT_SCHEMA = "pazzle-m144-dct-where-swap-receipt-v1"
ORACLE_RECEIPT_SCHEMA = "pazzle-m144-dct-where-oracle-receipt-v1"
CAPACITY_RECEIPT_SCHEMA = "pazzle-m144-dct-where-capacity-receipt-v1"
ENCODER_CHECKPOINT_SCHEMA = "pazzle-m144-fit-encoder-checkpoint-v1"
ENCODER_CHECKPOINT_RECEIPT_SCHEMA = "pazzle-m144-fit-encoder-checkpoint-receipt-v1"
ENCODER_GATE_RECEIPT_SCHEMA = "pazzle-m144-fit-encoder-cal-gate-receipt-v1"
REPRESENTATION_SCHEMA = "pazzle-m144-representation-contract-v1"
REPRESENTATION_RECEIPT_SCHEMA = "pazzle-m144-representation-contract-receipt-v1"
TERMINAL_RECEIPT_SCHEMA = "pazzle-m144-dct-where-terminal-receipt-v1"
VERIFICATION_INVOCATION_SCHEMA = "pazzle-m144-dct-where-verification-invocation-v1"
LOCK_SCHEMA = "pazzle-m144-dct-where-lock-v1"
CHECKPOINT_SCHEMA = "pazzle-m144-dct-where-checkpoint-v1"
CHECKPOINT_RECEIPT_SCHEMA = "pazzle-m144-dct-where-checkpoint-receipt-v1"
RAW_RECEIPT_SCHEMA = "pazzle-m144-dct-where-raw-receipt-v1"
REPORT_SCHEMA = "pazzle-m144-dct-where-report-v2"

PINNED_SPLIT_SHA256 = "a858a194ceab9976b72069aef6c46481734ce15594f67ae6818b4d7bfe30231a"
PINNED_SOURCE_SHA256 = "fa142c5f9c4fa17671b60d72b9acedff0eafcad4e77afac2b17a9649adfbfbd9"
PINNED_PRIOR_PAIRED_SHA256 = "a93405fc0e5cc129e8008bd3875957b0683e0dad3671f360a197b806d45fb554"

DEFAULT_WORK_ROOT = Path(r"E:\pazzle_work\m144_dct_where_v1")
DEFAULT_DATA_ROOT = Path(r"E:\pazzle_data")
DEFAULT_SPLIT = Path(
    r"E:\pazzle_work\pazzle_fixed_orientation_20260813"
    r"\PGA1_set_slot\source_disjoint_split_v1.json"
)
DEFAULT_SOURCE_MANIFEST = Path(r"E:\pazzle_work\rank96_e11_v4\source_groups_v4.json")

FIT_COUNT = 5_360
CAL_COUNT = 670
DEV_COUNT = 670
RESERVE_COUNT = 300
TRAIN_STEPS = 2_500
ENCODER_SEED = 144_011
ENCODER_STEPS = 1_500
ENCODER_BOARD_BATCH = 4
ENCODER_TILES_PER_BOARD = 192
ENCODER_CHECKPOINT_EVERY = 100
ENCODER_LEARNING_RATE = 3.0e-4
ENCODER_WEIGHT_DECAY = 1.0e-4
ENCODER_BETAS = (0.9, 0.999)
ENCODER_EPS = 1.0e-8
BATCH_SIZE = 8
CHECKPOINT_EVERY = 100
CACHE_CHUNK = 8
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
ADAM_BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
ORACLE_GAIN_MIN = 0.040
PALETTE_DIM = 60
PALETTE_QUANTILES = np.linspace(0.0, 1.0, 13, dtype=np.float64)
MIN_GPU_TOTAL_BYTES = 15 << 29  # 7.5 GiB
MIN_GPU_FREE_BYTES = 4 << 30
MAX_GPU_ALLOCATED_BYTES = 3 << 30
MAX_WORK_ROOT_BYTES = 6 << 30
MAX_WALL_SECONDS = 8 * 60 * 60
ARM_NAMES = ("dct_full", "dct_blind", "rgb_full", "rgb_blind")
RAW_ID_KEYS = (
    "board_id",
    "source_group_id",
    "swap_cycle_id",
)
RAW_SSIM_KEYS = (
    "flat_ssim",
    "target_oracle_dct_ssim",
    "dct_full_ssim",
    "dct_blind_ssim",
    "dct_swapped_ssim",
    "rgb8_full_ssim",
    "rgb8_blind_ssim",
)
RAW_PREDICTION_KEYS = (
    "flat_rgb",
    "dct_full_coeff",
    "dct_blind_coeff",
    "dct_swapped_coeff",
    "rgb8_full_residual",
    "rgb8_blind_residual",
)
RAW_KEYS = RAW_ID_KEYS + RAW_SSIM_KEYS + RAW_PREDICTION_KEYS
ID_KEYS = RAW_ID_KEYS
SSIM_KEYS = RAW_SSIM_KEYS
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_NAME = re.compile(r"^img_(\d{6})\.png$")
SOURCE_CLOSURE = (
    "autoresearch-runs/pazzle-mgc-restoration-20260818/M144_DCT_WHERE_PLAN.md",
    "launch_m144_dct_where.ps1",
    "src/config.py",
    "src/distort.py",
    "src/eval_paired_alignment.py",
    "src/imgio.py",
    "src/m144_dct_where.py",
    "src/run_m144_dct_where.py",
    "src/verify_m144_dct_where.py",
    "tests/test_m144_dct_where.py",
    "tests/test_run_m144_dct_where.py",
    "tests/test_verify_m144_dct_where.py",
)
PROCESS_STARTED_MONOTONIC = time.monotonic()
CUMULATIVE_ACTIVE_BASE = 0.0


class ContractError(RuntimeError):
    """An immutable input, cache or recovery artifact failed validation."""


class ScientificReject(RuntimeError):
    """The authenticated experiment reached a predeclared KILL gate."""


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(value))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, canonical_json(value))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def path_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise ContractError(f"required file is missing: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def verify_path_record(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if set(record) != {"path", "bytes", "sha256"}:
        raise ContractError(f"{label} path record has wrong keys")
    actual = path_record(Path(str(record["path"])))
    expected = {"path": str(Path(str(record["path"])).resolve()),
                "bytes": int(record["bytes"]), "sha256": str(record["sha256"])}
    if actual != expected:
        raise ContractError(f"{label} hash/size drift: {actual['path']}")
    return actual


def _require_drive_e(path: Path, label: str, *, allow_non_e: bool = False) -> Path:
    resolved = Path(path).resolve()
    if not allow_non_e and resolved.drive.upper() != "E:":
        raise ContractError(f"{label} must be on E:, got {resolved}")
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _board_id(name: str) -> int:
    match = IMAGE_NAME.fullmatch(name)
    if match is None:
        raise ContractError(f"invalid board filename: {name!r}")
    return int(match.group(1))


def _names_digest(names: Sequence[str]) -> str:
    return sha256_bytes("\n".join(names).encode("ascii"))


@dataclass(frozen=True)
class SplitData:
    names: dict[str, tuple[str, ...]]
    group_for_name: dict[str, str]
    group_id_for_name: dict[str, int]


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def contract(self) -> Path:
        return self.root / "contract.json"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def receipts(self) -> Path:
        return self.root / "receipts"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def report(self) -> Path:
        return self.artifacts / "m144_report.json"

    @property
    def lock(self) -> Path:
        return self.root / "run.lock"

    def ensure(self) -> None:
        for path in (self.root, self.cache, self.checkpoints, self.receipts, self.artifacts):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CacheFiles:
    partition: str
    embeddings: Path
    flat: Path
    palette: Path
    manifest: Path
    progress: Path
    partial_embeddings: Path
    partial_flat: Path
    partial_palette: Path


def cache_files(paths: RunPaths, partition: str) -> CacheFiles:
    key = partition.lower()
    if key not in {"fit", "cal", "dev"}:
        raise ContractError(f"unsupported cache partition: {partition}")
    return CacheFiles(
        key,
        paths.cache / f"{key}_dirty_embeddings.f16.npy",
        paths.cache / f"{key}_flat_rgb.f32.npy",
        paths.cache / f"{key}_dirty_feature60.f64.npy",
        paths.cache / f"{key}_cache.json",
        paths.cache / f"{key}_progress.json",
        paths.cache / f".{key}_dirty_embeddings.partial.npy",
        paths.cache / f".{key}_flat_rgb.partial.npy",
        paths.cache / f".{key}_dirty_feature60.partial.npy",
    )


def validate_split_payload(
    split_payload: Mapping[str, Any], source_payload: Mapping[str, Any]
) -> SplitData:
    expected = {"fit": FIT_COUNT, "cal": CAL_COUNT, "dev": DEV_COUNT, "reserve": RESERVE_COUNT}
    splits = split_payload.get("splits")
    counts = split_payload.get("counts")
    if not isinstance(splits, Mapping) or not isinstance(counts, Mapping):
        raise ContractError("split manifest lacks counts/splits")
    normalized: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for label, count in expected.items():
        rows = splits.get(label)
        if not isinstance(rows, list) or len(rows) != count or int(counts.get(label, -1)) != count:
            raise ContractError(f"split {label} must contain exactly {count} names")
        names = tuple(str(name) for name in rows)
        if len(set(names)) != len(names) or any(name in seen for name in names):
            raise ContractError(f"split {label} overlaps or repeats names")
        for name in names:
            _board_id(name)
        normalized[label] = names
        seen.update(names)
    if len(seen) != FIT_COUNT + CAL_COUNT + DEV_COUNT + RESERVE_COUNT:
        raise ContractError("split does not cover exactly 7,000 unique boards")

    files = source_payload.get("files")
    groups = source_payload.get("groups")
    if not isinstance(files, Mapping) or not isinstance(groups, Mapping) or set(files) != seen:
        raise ContractError("source manifest does not exactly cover split names")
    group_for_name: dict[str, str] = {}
    for name in seen:
        row = files.get(name)
        if not isinstance(row, Mapping) or not isinstance(row.get("source_group"), str):
            raise ContractError(f"source manifest lacks source group for {name}")
        group_for_name[name] = str(row["source_group"])
    groups_by_split = {
        label: {group_for_name[name] for name in normalized[label]}
        for label in ("fit", "cal", "dev", "reserve")
    }
    labels = tuple(groups_by_split)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                raise ContractError(
                    f"source groups overlap across {left}/{right}: {sorted(overlap)[:3]}"
                )
    unique_groups = sorted(set(group_for_name.values()))
    group_index = {group: index for index, group in enumerate(unique_groups)}
    return SplitData(
        names=normalized,
        group_for_name=group_for_name,
        group_id_for_name={name: group_index[group] for name, group in group_for_name.items()},
    )


def load_split_data(split_path: Path, source_path: Path) -> SplitData:
    split_record = path_record(split_path)
    source_record = path_record(source_path)
    if split_record["sha256"] != PINNED_SPLIT_SHA256:
        raise ContractError("split manifest SHA-256 does not match M144 freeze")
    if source_record["sha256"] != PINNED_SOURCE_SHA256:
        raise ContractError("source manifest SHA-256 does not match M144 freeze")
    return validate_split_payload(read_json(split_path), read_json(source_path))


def runtime_record() -> dict[str, Any]:
    import scipy
    import skimage

    cuda_available = bool(torch.cuda.is_available())
    gpu: dict[str, Any] | None = None
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": str(props.name),
            "total_memory": int(props.total_memory),
            "capability": [int(props.major), int(props.minor)],
        }
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "scipy": str(scipy.__version__),
        "skimage": str(skimage.__version__),
        "cuda_available": cuda_available,
        "torch_cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "gpu": gpu,
        "determinism": {
            "algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
    }


def configure_deterministic_runtime() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_contract(args: argparse.Namespace, paths: RunPaths, split: SplitData) -> dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    source_files: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_CLOSURE:
        source_files[relative] = path_record(repo_root / Path(relative))
    config = {
        "arms": list(ARM_NAMES),
        "steps": int(args.steps),
        "batch": int(args.batch),
        "checkpoint_every": int(args.checkpoint_every),
        "learning_rate": LEARNING_RATE,
        "adam_betas": list(ADAM_BETAS),
        "weight_decay": WEIGHT_DECAY,
        "one_cycle_pct_start": 0.05,
        "grad_clip": GRAD_CLIP,
        "bootstrap_seed": int(core.BOOTSTRAP_SEED),
        "amp": bool(args.amp),
        "cache_dtype": "float16",
        "cache_chunk": int(args.cache_chunk),
        "dirty_feature": {
            "schema": "m144-dirty-feature60-v1",
            "tile_value_dtype": "float64",
            "tile_value_scale": "uint8/255",
            "columns": ["mean_R", "mean_G", "mean_B", "centered_rgb_rms"],
            "quantiles": PALETTE_QUANTILES.tolist(),
            "quantile_method": "linear",
            "tail": ["population_mean", "population_std_ddof0"],
        },
        "swap": {
            "whitening_center": "FIT_population_mean_float64",
            "whitening_scale": "maximum(FIT_population_std_ddof0,1e-6)",
            "cost": "squared_euclidean_sum_float64",
            "assignment": "scipy.optimize.linear_sum_assignment_no_jitter",
            "forbidden": ["self", "same_source_group"],
            "cycles": "arbitrary_canonical_dense_by_min_board_id",
        },
        "resource_caps": {
            "min_gpu_total_bytes": MIN_GPU_TOTAL_BYTES,
            "min_gpu_free_pre_stage_bytes": MIN_GPU_FREE_BYTES,
            "max_gpu_allocated_bytes": MAX_GPU_ALLOCATED_BYTES,
            "max_work_root_bytes": MAX_WORK_ROOT_BYTES,
            "max_wall_seconds": MAX_WALL_SECONDS,
        },
        "encoder": encoder_recipe(),
        "no_d4": True,
    }
    if config["steps"] != TRAIN_STEPS or config["batch"] != BATCH_SIZE:
        raise ContractError("production M144 requires exactly 2,500 steps and batch 8")
    if config["checkpoint_every"] != CHECKPOINT_EVERY:
        raise ContractError("production M144 checkpoint interval must be 100")
    if config["cache_chunk"] < 1:
        raise ContractError("cache chunk must be positive")
    input_root = data_root / "train" / "inputs"
    target_root = data_root / "train" / "targets"
    for root, label in ((input_root, "train input root"), (target_root, "train target root")):
        if not root.is_dir():
            raise ContractError(f"{label} missing: {root}")
    body = {
        "schema": RUN_SCHEMA,
        "work_root": str(paths.root.resolve()),
        "data": {"root": str(data_root), "inputs": str(input_root), "targets": str(target_root)},
        "split_manifest": path_record(Path(args.split)),
        "source_manifest": path_record(Path(args.source_manifest)),
        "prior_evidence": {
            "kind": "leaky_paired_alignment_checkpoint",
            "sha256": PINNED_PRIOR_PAIRED_SHA256,
            "loaded": False,
            "overlap": {"fit": [5_133, 5_360], "cal": [639, 670],
                        "dev": [638, 670], "reserve": [290, 300]},
            "use": "human_prior_only_not_runtime_input",
        },
        "source_files": source_files,
        "runtime": runtime_record(),
        "partitions": {
            label: {"count": len(split.names[label]), "names_sha256": _names_digest(split.names[label])}
            for label in ("fit", "cal", "dev")
        },
        "reserve": {"count": len(split.names["reserve"]),
                    "names_sha256": _names_digest(split.names["reserve"]), "accessed": False},
        "config": config,
    }
    return {**body, "contract_sha256": canonical_digest(body)}


def freeze_or_verify_contract(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(contract)
    if path.exists():
        current = read_json(path)
        if current != normalized:
            raise ContractError("existing M144 contract differs from current invocation")
        return current
    atomic_json(path, normalized)
    if read_json(path) != normalized:
        raise ContractError("M144 contract did not round-trip")
    return normalized


def _tree_file_bytes(root: Path) -> int:
    total = 0
    for directory, _, files in os.walk(root):
        for filename in files:
            try:
                total += int((Path(directory) / filename).stat().st_size)
            except FileNotFoundError:
                continue
    return total


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_run_lock(paths: RunPaths) -> dict[str, Any]:
    """Acquire the single-writer lock, archiving only a provably dead owner."""
    global CUMULATIVE_ACTIVE_BASE
    paths.ensure()
    previous_cumulative = 0.0
    if paths.status.exists():
        previous = read_json(paths.status)
        previous_cumulative = float(
            previous.get("resources", {}).get("cumulative_active_seconds", 0.0)
        )
    if paths.lock.exists():
        stale = read_json(paths.lock)
        if stale.get("schema") != LOCK_SCHEMA:
            raise ContractError("existing run lock has an unknown schema")
        owner = int(stale.get("pid", -1))
        if str(stale.get("host")) != platform.node() or _pid_alive(owner):
            raise ContractError(f"M144 work root is already locked by pid {owner}")
        started = float(stale.get("started_unix", 0.0))
        heartbeat = float(stale.get("heartbeat_unix", started))
        stale_cumulative = float(stale.get("cumulative_at_start", 0.0)) + max(
            0.0, heartbeat - started
        )
        previous_cumulative = max(previous_cumulative, stale_cumulative)
        archive = paths.receipts / f"stale_lock_{uuid.uuid4().hex}.json"
        atomic_json(archive, {"schema": "pazzle-m144-stale-lock-v1", "lock": stale})
        paths.lock.unlink()
    CUMULATIVE_ACTIVE_BASE = previous_cumulative
    now = time.time()
    lock = {
        "schema": LOCK_SCHEMA,
        "pid": os.getpid(),
        "host": platform.node(),
        "started_unix": now,
        "heartbeat_unix": now,
        "cumulative_at_start": previous_cumulative,
        "argv": [str(value) for value in sys.argv],
    }
    payload = canonical_json(lock)
    try:
        with paths.lock.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ContractError("M144 work root was locked concurrently") from exc
    return lock


def heartbeat_run_lock(paths: RunPaths) -> None:
    if not paths.lock.exists():
        return
    lock = read_json(paths.lock)
    if lock.get("schema") != LOCK_SCHEMA or int(lock.get("pid", -1)) != os.getpid():
        raise ContractError("current process no longer owns the M144 run lock")
    lock["heartbeat_unix"] = time.time()
    atomic_json(paths.lock, lock)


def release_run_lock(paths: RunPaths) -> None:
    if not paths.lock.exists():
        return
    lock = read_json(paths.lock)
    if lock.get("schema") != LOCK_SCHEMA or int(lock.get("pid", -1)) != os.getpid():
        raise ContractError("refusing to release another process's M144 run lock")
    paths.lock.unlink()


def resource_snapshot(paths: RunPaths) -> dict[str, Any]:
    disk = shutil.disk_usage(paths.root)
    cuda = bool(torch.cuda.is_available())
    gpu_free, gpu_total = torch.cuda.mem_get_info() if cuda else (0, 0)
    process_elapsed = float(time.monotonic() - PROCESS_STARTED_MONOTONIC)
    return {
        "free_disk_bytes": int(disk.free),
        "work_root_bytes": _tree_file_bytes(paths.root),
        "process_elapsed_seconds": process_elapsed,
        "cumulative_active_seconds": float(CUMULATIVE_ACTIVE_BASE + process_elapsed),
        "cuda": cuda,
        "gpu_free_bytes": int(gpu_free),
        "gpu_total_bytes": int(gpu_total),
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated()) if cuda else 0,
        "gpu_reserved_bytes": int(torch.cuda.memory_reserved()) if cuda else 0,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if cuda else 0,
    }


def enforce_resource_caps(paths: RunPaths, *, pre_stage: bool = False) -> dict[str, Any]:
    evidence = resource_snapshot(paths)
    if evidence["work_root_bytes"] > MAX_WORK_ROOT_BYTES:
        raise ContractError("M144 work-root footprint exceeded its frozen 6 GiB cap")
    if evidence["gpu_peak_allocated_bytes"] > MAX_GPU_ALLOCATED_BYTES:
        raise ContractError("M144 GPU peak allocation exceeded its frozen 3 GiB cap")
    if evidence["cumulative_active_seconds"] > MAX_WALL_SECONDS:
        raise ContractError("M144 cumulative active wall time exceeded its frozen 8 hour cap")
    if pre_stage:
        if not evidence["cuda"] or evidence["gpu_total_bytes"] < MIN_GPU_TOTAL_BYTES:
            raise ContractError("M144 requires a CUDA device with at least 7.5 GiB total memory")
        if evidence["gpu_free_bytes"] < MIN_GPU_FREE_BYTES:
            raise ContractError("M144 requires at least 4 GiB free GPU memory before each stage")
    return evidence


def write_status(paths: RunPaths, contract_sha256: str, *, state: str, step: int,
                 message: str, checkpoint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    heartbeat_run_lock(paths)
    resources = resource_snapshot(paths)
    value = {
        "schema": STATUS_SCHEMA,
        "state": state,
        "step": int(step),
        "total_steps": TRAIN_STEPS,
        "contract_sha256": contract_sha256,
        "checkpoint": dict(checkpoint) if checkpoint is not None else None,
        "message": message,
        "resources": resources,
        "updated_unix": time.time(),
        "resume_command": [sys.executable, "-B", str(Path(__file__).resolve()), "run",
                           "--work-root", str(paths.root.resolve())],
    }
    atomic_json(paths.status, value)
    return value


def stateless_batch_indices(count: int, batch: int, step: int, seed: int) -> np.ndarray:
    """Return the exact FIT indices for a 1-based step without mutable RNG state."""
    if count < 1 or batch < 1 or step < 1:
        raise ValueError("count, batch and step must be positive")
    begin = (step - 1) * batch
    result: list[int] = []
    cursor = begin
    while len(result) < batch:
        epoch, offset = divmod(cursor, count)
        rng = np.random.default_rng(int(seed) + 1_000_003 * epoch)
        order = rng.permutation(count)
        take = min(batch - len(result), count - offset)
        result.extend(int(value) for value in order[offset : offset + take])
        cursor += take
    return np.asarray(result, dtype=np.int64)


def dirty_feature60(fragments: np.ndarray) -> np.ndarray:
    """Return the frozen 60-D board palette descriptor from dirty upright tiles.

    The only input is the stored dirty board.  Pixels are converted to
    ``float64 / 255``; each tile contributes RGB channel means and one RMS over
    all RGB residuals after subtracting those three means.  Thirteen linear
    quantiles per column plus population mean/std per column form 60 values.
    """
    raw = np.asarray(fragments)
    if raw.shape != (576, 20, 20, 3) or raw.dtype != np.uint8:
        raise ContractError(f"dirty tiles must be uint8[576,20,20,3], got {raw.shape}/{raw.dtype}")
    tiles = raw.astype(np.float64) / np.float64(255.0)
    means = tiles.mean(axis=(1, 2), dtype=np.float64)
    centered = tiles - means[:, None, None, :]
    rms = np.sqrt(np.mean(centered * centered, axis=(1, 2, 3), dtype=np.float64))
    per_tile = np.concatenate((means, rms[:, None]), axis=1)
    quantiles = np.quantile(
        per_tile, PALETTE_QUANTILES, axis=0, method="linear"
    ).T.reshape(-1)
    population_mean = per_tile.mean(axis=0, dtype=np.float64)
    population_std = per_tile.std(axis=0, ddof=0, dtype=np.float64)
    feature = np.concatenate((quantiles, population_mean, population_std)).astype(
        np.float64, copy=False
    )
    if feature.shape != (PALETTE_DIM,) or not np.isfinite(feature).all():
        raise ContractError("dirty palette descriptor is not finite float64[60]")
    return feature


def canonical_cycle_ids(donor: np.ndarray, board_ids: np.ndarray) -> np.ndarray:
    """Label arbitrary permutation cycles by increasing minimum board ID."""
    permutation = np.asarray(donor)
    identifiers = np.asarray(board_ids)
    count = len(identifiers)
    if permutation.shape != (count,) or permutation.dtype != np.int64:
        raise ContractError("donor must be int64[N]")
    if identifiers.shape != (count,) or identifiers.dtype != np.int64:
        raise ContractError("board_ids must be int64[N]")
    if not np.all(np.diff(identifiers) > 0):
        raise ContractError("cycle board IDs must be strictly increasing")
    if set(permutation.tolist()) != set(range(count)):
        raise ContractError("donor assignment is not a permutation")
    if np.any(permutation == np.arange(count, dtype=np.int64)):
        raise ContractError("donor assignment contains a fixed point")

    cycle = np.full(count, -1, dtype=np.int64)
    next_id = 0
    for start in range(count):
        if cycle[start] >= 0:
            continue
        members: list[int] = []
        cursor = start
        local: set[int] = set()
        while cursor not in local:
            if cycle[cursor] >= 0:
                raise ContractError("donor walk entered an already labelled cycle")
            local.add(cursor)
            members.append(cursor)
            cursor = int(permutation[cursor])
        if cursor != start or len(members) < 2:
            raise ContractError("donor permutation has a malformed cycle")
        cycle[np.asarray(members, dtype=np.int64)] = next_id
        next_id += 1
    if (cycle < 0).any():
        raise ContractError("not every board received a swap cycle")
    return cycle


def solve_swap_assignment(
    features: np.ndarray,
    source_group_ids: np.ndarray,
    board_ids: np.ndarray,
    mean60: np.ndarray,
    scale60: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the frozen palette-matched min-cost derangement."""
    from scipy.optimize import linear_sum_assignment

    feature = np.asarray(features, dtype=np.float64)
    groups = np.asarray(source_group_ids)
    identifiers = np.asarray(board_ids)
    mean = np.asarray(mean60, dtype=np.float64)
    scale = np.asarray(scale60, dtype=np.float64)
    count = feature.shape[0]
    if feature.shape != (count, PALETTE_DIM) or not np.isfinite(feature).all():
        raise ContractError("swap features must be finite float64[N,60]")
    if groups.shape != (count,) or groups.dtype != np.int64:
        raise ContractError("swap source groups must be int64[N]")
    if identifiers.shape != (count,) or identifiers.dtype != np.int64:
        raise ContractError("swap board IDs must be int64[N]")
    if mean.shape != (PALETTE_DIM,) or scale.shape != (PALETTE_DIM,):
        raise ContractError("FIT whitening statistics must be float64[60]")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale < 1.0e-6):
        raise ContractError("FIT whitening statistics are invalid")
    if not np.all(np.diff(identifiers) > 0):
        raise ContractError("swap rows/columns must use sorted canonical board IDs")

    whitened = (feature - mean) / scale
    cost = np.empty((count, count), dtype=np.float64)
    for anchor in range(count):
        delta = whitened - whitened[anchor]
        cost[anchor] = np.sum(delta * delta, axis=1, dtype=np.float64)
    forbidden = (groups[:, None] == groups[None, :])
    forbidden[np.arange(count), np.arange(count)] = True
    if np.any(np.all(forbidden, axis=1)) or np.any(np.all(forbidden, axis=0)):
        raise ContractError("source-group constraints make swap assignment infeasible")
    cost[forbidden] = np.inf
    rows, columns = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(count)):
        raise ContractError("SciPy returned non-canonical assignment rows")
    donor = columns.astype(np.int64, copy=False)
    if not np.isfinite(cost[rows, donor]).all():
        raise ContractError("swap assignment selected a forbidden edge")
    if len(np.unique(donor)) != count or np.any(donor == rows):
        raise ContractError("swap assignment is not a fixed-point-free bijection")
    if np.any(groups[donor] == groups):
        raise ContractError("swap assignment collided with a source group")
    return donor, canonical_cycle_ids(donor, identifiers)


def _array_sha(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def encoder_recipe() -> dict[str, Any]:
    from verify_m144_dct_where import ENCODER_RECIPE

    return copy.deepcopy(ENCODER_RECIPE)


@dataclass
class EncoderState:
    model: PairedAlignment
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
    scaler: torch.amp.GradScaler


def create_fit_encoder(device: torch.device, *, amp: bool) -> EncoderState:
    torch.manual_seed(ENCODER_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(ENCODER_SEED)
    model = PairedAlignment(embed_dim=128).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ENCODER_LEARNING_RATE,
        weight_decay=ENCODER_WEIGHT_DECAY, betas=ENCODER_BETAS, eps=ENCODER_EPS,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ENCODER_STEPS, eta_min=0.0
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    return EncoderState(model, optimizer, scheduler, scaler)


def _encoder_checkpoint_payload(
    state: EncoderState, *, step: int, contract_sha256: str, loss: float,
) -> dict[str, Any]:
    return {
        "schema": ENCODER_CHECKPOINT_SCHEMA,
        "contract_sha256": contract_sha256,
        "step": int(step),
        "embed_dim": 128,
        "recipe": encoder_recipe(),
        "loss_definition": "symmetric_InfoNCE_float32",
        "loss": float(loss),
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
        "scaler": state.scaler.state_dict(),
    }


def _safe_encoder_payload(path: Path, contract_sha256: str, step: int) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ContractError(f"cannot load FIT-only encoder checkpoint safely: {exc}") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != ENCODER_CHECKPOINT_SCHEMA
        or payload.get("contract_sha256") != contract_sha256
        or int(payload.get("step", -1)) != step
        or int(payload.get("embed_dim", -1)) != 128
        or payload.get("recipe") != encoder_recipe()
        or payload.get("loss_definition") != "symmetric_InfoNCE_float32"
        or not isinstance(payload.get("model"), Mapping)
    ):
        raise ContractError(f"FIT-only encoder checkpoint identity mismatch at step {step}")
    return payload


def _verified_encoder_receipts(paths: RunPaths, contract_sha256: str) -> list[dict[str, Any]]:
    checkpoints: dict[int, Path] = {}
    receipts: dict[int, Path] = {}
    for path in paths.checkpoints.glob("encoder_step_*.pt"):
        match = re.fullmatch(r"encoder_step_(\d{7})\.pt", path.name)
        if match is None:
            raise ContractError(f"unexpected encoder checkpoint filename: {path.name}")
        checkpoints[int(match.group(1))] = path
    for path in paths.receipts.glob("encoder_checkpoint_step_*.json"):
        match = re.fullmatch(r"encoder_checkpoint_step_(\d{7})\.json", path.name)
        if match is None:
            raise ContractError(f"unexpected encoder receipt filename: {path.name}")
        receipts[int(match.group(1))] = path
    steps = sorted(set(checkpoints) | set(receipts))
    if steps and steps != list(range(ENCODER_CHECKPOINT_EVERY, steps[-1] + 1,
                                     ENCODER_CHECKPOINT_EVERY)):
        raise ContractError("encoder checkpoints are not a contiguous 100-step prefix")
    result: list[dict[str, Any]] = []
    for step in steps:
        checkpoint = checkpoints.get(step)
        if checkpoint is None:
            raise ContractError(f"encoder receipt step {step} lost its checkpoint")
        payload = _safe_encoder_payload(checkpoint, contract_sha256, step)
        expected = {
            "schema": ENCODER_CHECKPOINT_RECEIPT_SCHEMA,
            "contract_sha256": contract_sha256,
            "step": step,
            "checkpoint": path_record(checkpoint),
            "loss_definition": "symmetric_InfoNCE_float32",
            "loss": float(payload["loss"]),
        }
        receipt_path = receipts.get(step)
        if receipt_path is None:
            receipt_path = paths.receipts / f"encoder_checkpoint_step_{step:07d}.json"
            atomic_json(receipt_path, expected)
        elif read_json(receipt_path) != expected:
            raise ContractError(f"encoder checkpoint receipt drift at step {step}")
        result.append(expected)
    return result


def save_encoder_checkpoint(
    paths: RunPaths, state: EncoderState, *, step: int, contract_sha256: str, loss: float,
) -> dict[str, Any]:
    checkpoint = paths.checkpoints / f"encoder_step_{step:07d}.pt"
    receipt = paths.receipts / f"encoder_checkpoint_step_{step:07d}.json"
    if checkpoint.exists() or receipt.exists():
        raise ContractError(f"encoder checkpoint step {step} already exists")
    _atomic_torch(
        checkpoint,
        _encoder_checkpoint_payload(
            state, step=step, contract_sha256=contract_sha256, loss=loss
        ),
    )
    return _verified_encoder_receipts(paths, contract_sha256)[-1]


def load_latest_encoder_checkpoint(
    paths: RunPaths, state: EncoderState, contract_sha256: str,
) -> tuple[int, dict[str, Any] | None]:
    receipts = _verified_encoder_receipts(paths, contract_sha256)
    if not receipts:
        return 0, None
    receipt = receipts[-1]
    payload = _safe_encoder_payload(
        Path(str(receipt["checkpoint"]["path"])), contract_sha256, int(receipt["step"])
    )
    state.model.load_state_dict(payload["model"], strict=True)
    state.optimizer.load_state_dict(payload["optimizer"])
    optimizer_state_to_device(state.optimizer, next(state.model.parameters()).device)
    state.scheduler.load_state_dict(payload["scheduler"])
    state.scaler.load_state_dict(payload["scaler"])
    return int(receipt["step"]), receipt


def _tiles_to_device(tiles: np.ndarray, device: torch.device) -> Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2)
        .float().div_(255.0).to(device)
    )


def train_fit_only_encoder(
    *, paths: RunPaths, contract: Mapping[str, Any], split: SplitData,
    device: torch.device, amp: bool,
) -> tuple[PairedAlignment, dict[str, Any]]:
    contract_sha = str(contract["contract_sha256"])
    names = tuple(sorted(split.names["fit"], key=_board_id))
    if len(names) != FIT_COUNT:
        raise ContractError("FIT-only encoder requires exactly 5,360 FIT boards")
    state = create_fit_encoder(device, amp=amp)
    start, latest = load_latest_encoder_checkpoint(paths, state, contract_sha)
    target_root = Path(str(contract["data"]["targets"]))
    write_status(
        paths, contract_sha, state="encoder_training", step=start,
        message=f"FIT-only scratch encoder step {start}/{ENCODER_STEPS}; CAL/DEV sealed",
        checkpoint=latest,
    )
    losses: list[float] = []
    for step in range(start + 1, ENCODER_STEPS + 1):
        indices = stateless_batch_indices(
            FIT_COUNT, ENCODER_BOARD_BATCH, step, ENCODER_SEED
        )
        dirty_rows: list[np.ndarray] = []
        clean_rows: list[np.ndarray] = []
        for slot, raw_index in enumerate(indices):
            name = names[int(raw_index)]
            board_id = _board_id(name)
            clean_all = to_frags(load(str(target_root / name)))
            choice_rng = np.random.default_rng(
                [ENCODER_SEED, 1, step, slot, board_id, 0]
            )
            selected = choice_rng.choice(
                576, size=ENCODER_TILES_PER_BOARD, replace=False
            )
            clean = np.ascontiguousarray(clean_all[selected])
            corruption_rng = np.random.default_rng(
                [ENCODER_SEED, 1, step, slot, board_id, 1]
            )
            dirty = distort_frags(clean, corruption_rng)
            clean_rows.append(clean)
            dirty_rows.append(dirty)
        dirty_tensor = _tiles_to_device(np.concatenate(dirty_rows), device)
        clean_tensor = _tiles_to_device(np.concatenate(clean_rows), device)
        state.model.train()
        state.optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast("cuda", dtype=torch.float16)
            if amp and device.type == "cuda" else nullcontext()
        )
        with context:
            dirty_embed, clean_embed = state.model(dirty_tensor, clean_tensor)
        loss = symmetric_info_nce(
            dirty_embed.float(), clean_embed.float(), state.model.scale().float()
        )
        if not torch.isfinite(loss):
            raise ContractError(f"FIT-only encoder loss is non-finite at step {step}")
        state.scaler.scale(loss).backward()
        state.scaler.unscale_(state.optimizer)
        nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
        state.scaler.step(state.optimizer)
        state.scaler.update()
        state.scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if step % ENCODER_CHECKPOINT_EVERY == 0:
            enforce_resource_caps(paths)
            mean_loss = float(np.mean(losses))
            latest = save_encoder_checkpoint(
                paths, state, step=step, contract_sha256=contract_sha, loss=mean_loss
            )
            losses = []
            write_status(
                paths, contract_sha, state="encoder_training", step=step,
                message=f"FIT-only scratch encoder step {step}/{ENCODER_STEPS}; DEV sealed",
                checkpoint=latest,
            )
            print(json.dumps({"encoder_step": step, "loss": mean_loss}), flush=True)
    receipts = _verified_encoder_receipts(paths, contract_sha)
    if not receipts or int(receipts[-1]["step"]) != ENCODER_STEPS:
        raise ContractError("FIT-only encoder lacks fixed final step-1500 checkpoint")
    state.model.eval()
    return state.model, receipts[-1]


ENCODER_RANK_KEYS = (
    "board_id", "source_group_id", "dirty_to_clean_rank", "clean_to_dirty_rank"
)


def validate_encoder_rank_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != set(ENCODER_RANK_KEYS):
        raise ContractError("encoder CAL rank artifact has wrong keys")
    for key in ("board_id", "source_group_id"):
        value = np.asarray(arrays[key])
        if value.shape != (CAL_COUNT,) or value.dtype != np.int64:
            raise ContractError(f"encoder CAL {key} must be int64[670]")
    if not np.all(np.diff(np.asarray(arrays["board_id"])) > 0):
        raise ContractError("encoder CAL board_id must be strictly increasing")
    for key in ("dirty_to_clean_rank", "clean_to_dirty_rank"):
        value = np.asarray(arrays[key])
        if value.shape != (CAL_COUNT, 576) or value.dtype != np.uint16:
            raise ContractError(f"encoder CAL {key} must be uint16[670,576]")
        if (value < 1).any() or (value > 576).any():
            raise ContractError(f"encoder CAL {key} contains an invalid rank")


def _rank_summary(rank: np.ndarray) -> dict[str, float]:
    value = np.asarray(rank)
    return {
        "micro_r1": float(np.mean(value <= 1)),
        "micro_r5": float(np.mean(value <= 5)),
        "micro_r20": float(np.mean(value <= 20)),
        "macro_r1": float(np.mean(np.mean(value <= 1, axis=1))),
        "macro_r5": float(np.mean(np.mean(value <= 5, axis=1))),
        "macro_r20": float(np.mean(np.mean(value <= 20, axis=1))),
        "median_rank": float(np.median(value)),
        "mrr": float(np.mean(1.0 / value.astype(np.float64))),
    }


def encoder_gate_summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    validate_encoder_rank_arrays(arrays)
    return {
        "n_boards": CAL_COUNT,
        "tiles_per_board": 576,
        "n_tile_queries": CAL_COUNT * 576,
        "dirty_to_clean": _rank_summary(arrays["dirty_to_clean_rank"]),
        "clean_to_dirty": _rank_summary(arrays["clean_to_dirty_rank"]),
    }


def encoder_gate_map(metrics: Mapping[str, Any]) -> dict[str, Any]:
    values = metrics["dirty_to_clean"]
    thresholds = {"dirty_to_clean_r1": ("micro_r1", 0.20),
                  "dirty_to_clean_r5": ("micro_r5", 0.45),
                  "dirty_to_clean_r20": ("micro_r20", 0.70)}
    checks: dict[str, Any] = {}
    for name, (metric_name, threshold) in thresholds.items():
        observed = float(values[metric_name])
        checks[name] = {"observed": observed, "operator": ">=",
                        "threshold": threshold, "passed": bool(observed >= threshold)}
    return {"checks": checks, "passed": bool(all(row["passed"] for row in checks.values()))}


@torch.inference_mode()
def evaluate_fit_encoder_cal(
    *, model: PairedAlignment, contract: Mapping[str, Any], split: SplitData,
    device: torch.device, amp: bool,
) -> dict[str, np.ndarray]:
    names = tuple(sorted(split.names["cal"], key=_board_id))
    arrays: dict[str, np.ndarray] = {
        "board_id": np.asarray([_board_id(name) for name in names], dtype=np.int64),
        "source_group_id": np.asarray(
            [split.group_id_for_name[name] for name in names], dtype=np.int64
        ),
        "dirty_to_clean_rank": np.empty((CAL_COUNT, 576), dtype=np.uint16),
        "clean_to_dirty_rank": np.empty((CAL_COUNT, 576), dtype=np.uint16),
    }
    target_root = Path(str(contract["data"]["targets"]))
    model.eval()
    for index, name in enumerate(names):
        board_id = _board_id(name)
        clean = to_frags(load(str(target_root / name)))
        dirty = distort_frags(
            clean, np.random.default_rng([ENCODER_SEED, 2, board_id])
        )
        clean_tensor = _tiles_to_device(clean, device)
        dirty_tensor = _tiles_to_device(dirty, device)
        context = (
            torch.autocast("cuda", dtype=torch.float16)
            if amp and device.type == "cuda" else nullcontext()
        )
        with context:
            dirty_embed, clean_embed = model(dirty_tensor, clean_tensor)
        dirty_embed = dirty_embed.float()
        clean_embed = clean_embed.float()
        arrays["dirty_to_clean_rank"][index] = rank_of_diagonal(
            dirty_embed @ clean_embed.t()
        ).cpu().numpy().astype(np.uint16)
        arrays["clean_to_dirty_rank"][index] = rank_of_diagonal(
            clean_embed @ dirty_embed.t()
        ).cpu().numpy().astype(np.uint16)
        if (index + 1) % 10 == 0 or index + 1 == CAL_COUNT:
            print(f"M144 ENCODER CAL {index + 1}/{CAL_COUNT}", flush=True)
    validate_encoder_rank_arrays(arrays)
    return arrays


def write_or_load_encoder_gate(
    *, paths: RunPaths, contract: Mapping[str, Any], final_checkpoint: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    path = paths.artifacts / "encoder_cal_ranks.npz"
    if path.exists():
        candidate = _load_exact_npz(path, ENCODER_RANK_KEYS)
    elif arrays is not None:
        candidate = {key: np.asarray(arrays[key]) for key in ENCODER_RANK_KEYS}
    else:
        raise ContractError("encoder CAL rank evidence is absent")
    metrics = encoder_gate_summary(candidate)
    gate = encoder_gate_map(metrics)
    record, loaded = _write_or_recover_npz(
        path=path,
        receipt_path=paths.receipts / "encoder_cal_gate.json",
        schema=ENCODER_GATE_RECEIPT_SCHEMA,
        contract_sha256=str(contract["contract_sha256"]),
        arrays=arrays,
        keys=ENCODER_RANK_KEYS,
        validate=validate_encoder_rank_arrays,
        receipt_extra={
            "checkpoint": dict(final_checkpoint),
            "cal_target_receipt": path_record(paths.receipts / "cal_targets.json"),
            "recipe": encoder_recipe(),
            "runtime": copy.deepcopy(contract["runtime"]),
            "metrics": metrics,
            "gate": gate,
        },
        max_bytes=4 << 20,
    )
    return record, loaded, metrics, gate


def write_or_load_representation_contract(
    *, paths: RunPaths, contract: Mapping[str, Any], checkpoint: Mapping[str, Any],
    rank_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    body = {
        "schema": REPRESENTATION_SCHEMA,
        "run_contract_sha256": str(contract["contract_sha256"]),
        "encoder_checkpoint": dict(checkpoint),
        "encoder_cal_ranks": dict(rank_record),
        "encoder_cal_gate_receipt": path_record(paths.receipts / "encoder_cal_gate.json"),
        "fit_target_receipt": path_record(paths.receipts / "fit_targets.json"),
        "cal_target_receipt": path_record(paths.receipts / "cal_targets.json"),
        "recipe": encoder_recipe(),
        "runtime": copy.deepcopy(contract["runtime"]),
        "source_files": {
            relative: str(contract["source_files"][relative]["sha256"])
            for relative in SOURCE_CLOSURE
        },
    }
    value = {**body, "representation_sha256": canonical_digest(body)}
    path = paths.artifacts / "representation_contract.json"
    if path.exists():
        if read_json(path) != value:
            raise ContractError("representation contract is create-once and drifted")
    else:
        atomic_json(path, value)
    record = path_record(path)
    receipt = {
        "schema": REPRESENTATION_RECEIPT_SCHEMA,
        "contract_sha256": str(contract["contract_sha256"]),
        "artifact": record,
        "representation_sha256": value["representation_sha256"],
    }
    receipt_path = paths.receipts / "representation_contract.json"
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise ContractError("representation contract receipt drift")
    else:
        atomic_json(receipt_path, receipt)
    return record, value


def load_representation_contract(
    paths: RunPaths, contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = paths.artifacts / "representation_contract.json"
    value = read_json(path)
    required = {
        "schema", "run_contract_sha256", "encoder_checkpoint", "encoder_cal_ranks",
        "encoder_cal_gate_receipt", "fit_target_receipt", "cal_target_receipt",
        "recipe", "runtime", "source_files", "representation_sha256",
    }
    if set(value) != required or value.get("schema") != REPRESENTATION_SCHEMA:
        raise ContractError("representation contract schema drift")
    body = dict(value)
    claimed = str(body.pop("representation_sha256", ""))
    if value.get("run_contract_sha256") != contract_sha256 or claimed != canonical_digest(body):
        raise ContractError("representation contract identity/self-digest mismatch")
    if value.get("recipe") != encoder_recipe():
        raise ContractError("representation encoder recipe drift")
    for key in (
        "encoder_checkpoint", "encoder_cal_ranks", "encoder_cal_gate_receipt",
        "fit_target_receipt", "cal_target_receipt",
    ):
        verify_path_record(value[key], label=f"representation {key}")
    receipt = {
        "schema": REPRESENTATION_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha256,
        "artifact": path_record(path),
        "representation_sha256": claimed,
    }
    if read_json(paths.receipts / "representation_contract.json") != receipt:
        raise ContractError("representation contract receipt mismatch")
    return receipt["artifact"], value


def _load_paired_encoder(
    checkpoint: Path, device: torch.device, *, contract_sha256: str,
) -> PairedAlignment:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ContractError(f"cannot load paired checkpoint safely: {exc}") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != ENCODER_CHECKPOINT_SCHEMA
        or payload.get("contract_sha256") != contract_sha256
        or int(payload.get("step", -1)) != ENCODER_STEPS
        or not isinstance(payload.get("model"), Mapping)
    ):
        raise ContractError("paired checkpoint lacks model state")
    embed_dim = int(payload.get("embed_dim", 0))
    if embed_dim != 128:
        raise ContractError(f"M144 requires paired embedding_dim=128, got {embed_dim}")
    model = PairedAlignment(embed_dim=embed_dim)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _encode_dirty_board(model: PairedAlignment, image_path: Path, device: torch.device,
                        amp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = load(str(image_path))
    if image.shape != (480, 480, 3) or image.dtype != np.uint8:
        raise ContractError(f"invalid dirty board {image_path}: {image.shape} {image.dtype}")
    fragments = to_frags(image)
    palette = dirty_feature60(fragments)
    tiles = (
        torch.from_numpy(np.ascontiguousarray(fragments)).permute(0, 3, 1, 2)
        .float().div_(255.0).to(device)
    )
    context = torch.autocast("cuda", dtype=torch.float16) if amp and device.type == "cuda" else nullcontext()
    with torch.inference_mode(), context:
        embedding = model.dirty_encoder(tiles)
    flat = core.flat_rgb_from_tiles(tiles.unsqueeze(0)).squeeze(0)
    return (
        embedding.detach().float().cpu().numpy().astype(np.float16),
        flat.detach().float().cpu().numpy().astype(np.float32),
        palette,
    )


def _verify_cache_manifest(files: CacheFiles, contract_sha256: str,
                           names: Sequence[str]) -> dict[str, Any]:
    manifest = read_json(files.manifest)
    required = {"schema", "partition", "contract_sha256", "count", "names_sha256",
                "embedding_dim", "representation_contract", "representation_sha256",
                "embeddings", "flat", "palette", "boards"}
    if set(manifest) != required or manifest.get("schema") != CACHE_SCHEMA:
        raise ContractError(f"invalid {files.partition} cache manifest schema")
    if manifest["partition"] != files.partition or manifest["contract_sha256"] != contract_sha256:
        raise ContractError(f"{files.partition} cache identity mismatch")
    if int(manifest["count"]) != len(names) or manifest["names_sha256"] != _names_digest(names):
        raise ContractError(f"{files.partition} cache name contract mismatch")
    root = files.manifest.parent.parent
    representation_path = root / "artifacts" / "representation_contract.json"
    representation_record = verify_path_record(
        manifest["representation_contract"], label=f"{files.partition} representation contract"
    )
    if representation_record != path_record(representation_path):
        raise ContractError(f"{files.partition} cache binds the wrong representation contract")
    representation = read_json(representation_path)
    if manifest["representation_sha256"] != representation.get("representation_sha256"):
        raise ContractError(f"{files.partition} cache representation digest mismatch")
    verify_path_record(manifest["embeddings"], label=f"{files.partition} embeddings")
    verify_path_record(manifest["flat"], label=f"{files.partition} flat")
    verify_path_record(manifest["palette"], label=f"{files.partition} dirty feature60")
    emb = np.load(files.embeddings, mmap_mode="r", allow_pickle=False)
    flat = np.load(files.flat, mmap_mode="r", allow_pickle=False)
    palette = np.load(files.palette, mmap_mode="r", allow_pickle=False)
    if emb.shape != (len(names), 576, 128) or emb.dtype != np.float16:
        raise ContractError(f"{files.partition} embedding cache has wrong shape/dtype")
    if flat.shape != (len(names), 3) or flat.dtype != np.float32:
        raise ContractError(f"{files.partition} flat cache has wrong shape/dtype")
    if palette.shape != (len(names), PALETTE_DIM) or palette.dtype != np.float64:
        raise ContractError(f"{files.partition} palette cache has wrong shape/dtype")
    if not np.isfinite(palette).all():
        raise ContractError(f"{files.partition} palette cache contains non-finite values")
    boards = manifest["boards"]
    if not isinstance(boards, list) or [row.get("name") for row in boards] != list(names):
        raise ContractError(f"{files.partition} cache board order mismatch")
    for row in boards:
        if set(row) != {"name", "input_bytes", "input_sha256"}:
            raise ContractError(f"{files.partition} cache board ledger is not dirty-only")
        if not HEX64.fullmatch(str(row["input_sha256"])) or int(row["input_bytes"]) < 1:
            raise ContractError(f"{files.partition} cache board ledger is invalid")
    return manifest


def _cache_progress_prefix(
    progress: Mapping[str, Any], names: Sequence[str], emb_map: np.ndarray,
    flat_map: np.ndarray, palette_map: np.ndarray, partition: str,
) -> int:
    chunks = progress.get("chunks")
    boards = progress.get("boards")
    if not isinstance(chunks, list) or not isinstance(boards, list):
        raise ContractError(f"{partition} progress chunks/boards invalid")
    completed = 0
    for chunk in chunks:
        if not isinstance(chunk, Mapping) or int(chunk.get("start", -1)) != completed:
            raise ContractError(f"{partition} progress is not a contiguous prefix")
        if set(chunk) != {
            "start", "stop", "embeddings_sha256", "flat_sha256", "palette_sha256"
        }:
            raise ContractError(f"{partition} progress chunk schema drift")
        stop = int(chunk.get("stop", -1))
        if stop <= completed or stop > len(names):
            raise ContractError(f"{partition} progress has invalid chunk boundary")
        if _array_sha(emb_map[completed:stop]) != chunk["embeddings_sha256"]:
            raise ContractError(f"{partition} partial embedding chunk drift")
        if _array_sha(flat_map[completed:stop]) != chunk["flat_sha256"]:
            raise ContractError(f"{partition} partial flat chunk drift")
        if _array_sha(palette_map[completed:stop]) != chunk["palette_sha256"]:
            raise ContractError(f"{partition} partial palette chunk drift")
        completed = stop
    if len(boards) != completed or [row.get("name") for row in boards] != list(names[:completed]):
        raise ContractError(f"{partition} progress board ledger mismatch")
    return completed


def _write_cache_manifest(
    files: CacheFiles, progress: Mapping[str, Any], names: Sequence[str],
    contract_sha: str,
) -> dict[str, Any]:
    representation_record, representation = load_representation_contract(
        paths=RunPaths(files.manifest.parent.parent), contract_sha256=contract_sha
    )
    manifest = {
        "schema": CACHE_SCHEMA,
        "partition": files.partition,
        "contract_sha256": contract_sha,
        "count": len(names),
        "names_sha256": _names_digest(names),
        "embedding_dim": 128,
        "representation_contract": representation_record,
        "representation_sha256": representation["representation_sha256"],
        "embeddings": path_record(files.embeddings),
        "flat": path_record(files.flat),
        "palette": path_record(files.palette),
        "boards": list(progress["boards"]),
    }
    atomic_json(files.manifest, manifest)
    return _verify_cache_manifest(files, contract_sha, names)


def build_embedding_cache(
    *, paths: RunPaths, partition: str, names: Sequence[str], contract: Mapping[str, Any],
    device: torch.device, amp: bool, chunk_size: int,
) -> dict[str, Any]:
    """Build/authenticate a resumable dirty-only NPY cache for one partition."""
    files = cache_files(paths, partition)
    contract_sha = str(contract["contract_sha256"])
    if files.manifest.exists():
        return _verify_cache_manifest(files, contract_sha, names)

    input_root = Path(str(contract["data"]["inputs"]))
    count = len(names)
    shape = (count, 576, 128)
    flat_shape = (count, 3)
    palette_shape = (count, PALETTE_DIM)
    progress: dict[str, Any]
    if files.progress.exists():
        progress = read_json(files.progress)
        if (
            progress.get("schema") != CACHE_PROGRESS_SCHEMA
            or progress.get("partition") != partition
            or progress.get("contract_sha256") != contract_sha
            or progress.get("count") != count
            or progress.get("names_sha256") != _names_digest(names)
        ):
            raise ContractError(f"{partition} cache progress identity mismatch")
        available = (
            files.embeddings if files.embeddings.exists() else files.partial_embeddings,
            files.flat if files.flat.exists() else files.partial_flat,
            files.palette if files.palette.exists() else files.partial_palette,
        )
        if not all(path.exists() for path in available):
            raise ContractError(f"{partition} cache progress lost an array")
        emb_map = np.lib.format.open_memmap(available[0], mode="r+", dtype=np.float16, shape=shape)
        flat_map = np.lib.format.open_memmap(available[1], mode="r+", dtype=np.float32, shape=flat_shape)
        palette_map = np.lib.format.open_memmap(
            available[2], mode="r+", dtype=np.float64, shape=palette_shape
        )
    else:
        if any(path.exists() for path in (
            files.partial_embeddings, files.partial_flat, files.partial_palette,
            files.embeddings, files.flat, files.palette,
        )):
            raise ContractError(f"orphan {partition} cache arrays without authenticated progress")
        files.embeddings.parent.mkdir(parents=True, exist_ok=True)
        emb_map = np.lib.format.open_memmap(files.partial_embeddings, mode="w+", dtype=np.float16, shape=shape)
        flat_map = np.lib.format.open_memmap(files.partial_flat, mode="w+", dtype=np.float32, shape=flat_shape)
        palette_map = np.lib.format.open_memmap(
            files.partial_palette, mode="w+", dtype=np.float64, shape=palette_shape
        )
        progress = {
            "schema": CACHE_PROGRESS_SCHEMA,
            "partition": partition,
            "contract_sha256": contract_sha,
            "count": count,
            "names_sha256": _names_digest(names),
            "chunks": [],
            "boards": [],
        }
        atomic_json(files.progress, progress)

    chunks = progress["chunks"]
    boards = progress["boards"]
    completed = _cache_progress_prefix(
        progress, names, emb_map, flat_map, palette_map, partition
    )
    if completed == count:
        del emb_map, flat_map, palette_map
        for partial, final in (
            (files.partial_embeddings, files.embeddings),
            (files.partial_flat, files.flat),
            (files.partial_palette, files.palette),
        ):
            if partial.exists() and not final.exists():
                os.replace(partial, final)
            elif partial.exists() and final.exists():
                raise ContractError(f"duplicate complete {partition} cache array")
        return _write_cache_manifest(files, progress, names, contract_sha)
    if any(path.exists() for path in (files.embeddings, files.flat, files.palette)):
        raise ContractError(f"incomplete {partition} progress has prematurely finalized arrays")

    _, representation = load_representation_contract(paths, contract_sha)
    paired = _load_paired_encoder(
        Path(str(representation["encoder_checkpoint"]["path"])),
        device,
        contract_sha256=contract_sha,
    )
    try:
        for start in range(completed, count, chunk_size):
            stop = min(count, start + chunk_size)
            chunk_embedding = np.empty((stop - start, 576, 128), dtype=np.float16)
            chunk_flat = np.empty((stop - start, 3), dtype=np.float32)
            chunk_palette = np.empty((stop - start, PALETTE_DIM), dtype=np.float64)
            new_boards: list[dict[str, Any]] = []
            for local, name in enumerate(names[start:stop]):
                input_path = input_root / name
                input_record = path_record(input_path)
                embedding, flat, palette = _encode_dirty_board(paired, input_path, device, amp)
                chunk_embedding[local] = embedding
                chunk_flat[local] = flat
                chunk_palette[local] = palette
                new_boards.append({
                    "name": name,
                    "input_bytes": input_record["bytes"],
                    "input_sha256": input_record["sha256"],
                })
            emb_map[start:stop] = chunk_embedding
            flat_map[start:stop] = chunk_flat
            palette_map[start:stop] = chunk_palette
            emb_map.flush(); flat_map.flush(); palette_map.flush()
            chunks.append({
                "start": start,
                "stop": stop,
                "embeddings_sha256": _array_sha(emb_map[start:stop]),
                "flat_sha256": _array_sha(flat_map[start:stop]),
                "palette_sha256": _array_sha(palette_map[start:stop]),
            })
            boards.extend(new_boards)
            atomic_json(files.progress, progress)
            write_status(paths, contract_sha, state="caching", step=0,
                         message=f"cached {partition} {stop}/{count}")
    finally:
        del paired
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del emb_map; del flat_map; del palette_map
    os.replace(files.partial_embeddings, files.embeddings)
    os.replace(files.partial_flat, files.flat)
    os.replace(files.partial_palette, files.palette)
    return _write_cache_manifest(files, progress, names, contract_sha)


def load_cache_arrays(paths: RunPaths, partition: str, names: Sequence[str],
                      contract_sha256: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = cache_files(paths, partition)
    _verify_cache_manifest(files, contract_sha256, names)
    return (
        np.load(files.embeddings, mmap_mode="r", allow_pickle=False),
        np.load(files.flat, mmap_mode="r", allow_pickle=False),
        np.load(files.palette, mmap_mode="r", allow_pickle=False),
    )


@dataclass
class Arm:
    name: str
    kind: str
    blind: bool
    model: core.M144WhereModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.OneCycleLR
    scaler: torch.amp.GradScaler


def authenticate_partition_targets(
    *, paths: RunPaths, partition: str, names: Sequence[str],
    contract: Mapping[str, Any], target_sha_for_name: Mapping[str, str],
) -> dict[str, Any]:
    """Open and authenticate clean labels only at the stage that is allowed."""
    if partition not in {"fit", "cal", "dev"}:
        raise ContractError(f"unsupported target partition: {partition}")
    receipt_path = paths.receipts / f"{partition}_targets.json"
    contract_sha = str(contract["contract_sha256"])
    target_root = Path(str(contract["data"]["targets"]))
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        required = {
            "schema", "partition", "contract_sha256", "count", "names_sha256", "targets"
        }
        if set(receipt) != required or receipt.get("schema") != TARGET_RECEIPT_SCHEMA:
            raise ContractError(f"invalid {partition} target receipt")
        if (
            receipt["partition"] != partition
            or receipt["contract_sha256"] != contract_sha
            or int(receipt["count"]) != len(names)
            or receipt["names_sha256"] != _names_digest(names)
        ):
            raise ContractError(f"{partition} target receipt identity mismatch")
        rows = receipt["targets"]
        if not isinstance(rows, list) or [row.get("name") for row in rows] != list(names):
            raise ContractError(f"{partition} target receipt order mismatch")
        for row in rows:
            if set(row) != {"name", "bytes", "sha256"}:
                raise ContractError(f"{partition} target receipt row schema drift")
            actual = path_record(target_root / str(row["name"]))
            if actual["bytes"] != int(row["bytes"]) or actual["sha256"] != row["sha256"]:
                raise ContractError(f"{partition} target bytes drift: {row['name']}")
        return receipt

    rows: list[dict[str, Any]] = []
    for name in names:
        record = path_record(target_root / name)
        expected = str(target_sha_for_name.get(name, "")).lower()
        if not HEX64.fullmatch(expected) or record["sha256"] != expected:
            raise ContractError(f"target bytes drift from source manifest: {name}")
        rows.append({"name": name, "bytes": record["bytes"], "sha256": record["sha256"]})
    receipt = {
        "schema": TARGET_RECEIPT_SCHEMA,
        "partition": partition,
        "contract_sha256": contract_sha,
        "count": len(names),
        "names_sha256": _names_digest(names),
        "targets": rows,
    }
    atomic_json(receipt_path, receipt)
    if read_json(receipt_path) != receipt:
        raise ContractError(f"{partition} target receipt did not round-trip")
    return receipt


def create_arms(device: torch.device, *, amp: bool) -> dict[str, Arm]:
    def pair(output_dim: int, seed: int) -> tuple[core.M144WhereModel, core.M144WhereModel]:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        full = core.M144WhereModel(output_dim=output_dim).to(device)
        blind = copy.deepcopy(full).to(device)
        return full, blind

    dct_full, dct_blind = pair(core.DCT_OUTPUT_DIM, int(core.BOOTSTRAP_SEED) + 11)
    rgb_full, rgb_blind = pair(core.RGB_OUTPUT_DIM, int(core.BOOTSTRAP_SEED) + 29)
    specs = {
        "dct_full": ("dct", False, dct_full),
        "dct_blind": ("dct", True, dct_blind),
        "rgb_full": ("rgb", False, rgb_full),
        "rgb_blind": ("rgb", True, rgb_blind),
    }
    arms: dict[str, Arm] = {}
    for name in ARM_NAMES:
        kind, blind, model = specs[name]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE, total_steps=TRAIN_STEPS, pct_start=0.05
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
        arms[name] = Arm(name, kind, blind, model, optimizer, scheduler, scaler)
    return arms


def _load_targets(target_root: Path, names: Sequence[str], device: torch.device) -> Tensor:
    rows = _load_targets_uint8(target_root, names)
    return torch.from_numpy(rows).permute(0, 3, 1, 2).float().div_(255.0).to(device)


def _load_targets_uint8(target_root: Path, names: Sequence[str]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for name in names:
        image = load(str(target_root / name))
        if image.shape != (480, 480, 3) or image.dtype != np.uint8:
            raise ContractError(f"invalid clean target {name}: {image.shape} {image.dtype}")
        rows.append(image)
    return np.stack(rows).astype(np.uint8, copy=False)


def quantize_render_uint8(rendered: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(rendered, Tensor):
        value = rendered.detach().cpu().numpy()
    else:
        value = np.asarray(rendered)
    if value.ndim != 4 or value.shape[1] != 3:
        raise ContractError("rendered images must have BCHW RGB shape")
    return np.clip(np.rint(value.astype(np.float32, copy=False) * np.float32(255.0)), 0, 255).astype(
        np.uint8
    )


def official_uint8_ssim(rendered: Tensor | np.ndarray, target_uint8: np.ndarray) -> np.ndarray:
    predicted = quantize_render_uint8(rendered)
    target = np.asarray(target_uint8)
    if target.ndim != 4 or target.shape[-1] != 3 or target.dtype != np.uint8:
        raise ContractError("official SSIM targets must be BHWC uint8")
    target_bchw = np.moveaxis(target, -1, 1)
    if predicted.shape != target_bchw.shape:
        raise ContractError("official SSIM render/target shapes differ")
    scores = core.skimage_ssim_parity(
        predicted,
        target_bchw,
        data_range=255.0,
        win_size=7,
        use_sample_covariance=True,
    )
    result = np.asarray(scores, dtype=np.float64)
    if result.shape != (target.shape[0],) or not np.isfinite(result).all():
        raise ContractError("official uint8 SSIM returned invalid values")
    return result


def _render_arm(kind: str, prediction: Tensor, flat: Tensor, size: tuple[int, int]) -> Tensor:
    if kind == "dct":
        return core.render_dct_residual(prediction.float(), flat.float(), size=size)
    if kind == "rgb":
        return core.render_rgb_residual(prediction.float(), flat.float(), size=size)
    raise AssertionError(kind)


def train_one_arm(arm: Arm, embeddings: Tensor, flat: Tensor, target: Tensor, *, amp: bool) -> float:
    arm.model.train()
    arm.optimizer.zero_grad(set_to_none=True)
    context = torch.autocast("cuda", dtype=torch.float16) if amp and embeddings.device.type == "cuda" else nullcontext()
    with context:
        prediction = arm.model(embeddings, flat, blind=arm.blind)
    rendered = _render_arm(arm.kind, prediction.float(), flat.float(), tuple(target.shape[-2:]))
    score = core.uniform_ssim(rendered.float(), target.float())
    loss = 1.0 - score.mean()
    arm.scaler.scale(loss).backward()
    arm.scaler.unscale_(arm.optimizer)
    nn.utils.clip_grad_norm_(arm.model.parameters(), GRAD_CLIP)
    arm.scaler.step(arm.optimizer)
    arm.scaler.update()
    arm.scheduler.step()
    return float(loss.detach().cpu())


def run_capacity_smoke(
    *, paths: RunPaths, contract: Mapping[str, Any], device: torch.device, amp: bool,
) -> dict[str, Any]:
    """Exercise all four resident arms at the exact B8/576/480 production shape."""
    receipt_path = paths.receipts / "capacity_smoke.json"
    contract_sha = str(contract["contract_sha256"])
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        representation_record, representation = load_representation_contract(paths, contract_sha)
        if (
            receipt.get("schema") != CAPACITY_RECEIPT_SCHEMA
            or receipt.get("contract_sha256") != contract_sha
            or receipt.get("passed") is not True
            or int(receipt.get("peak_allocated_bytes", MAX_GPU_ALLOCATED_BYTES + 1))
            > MAX_GPU_ALLOCATED_BYTES
            or receipt.get("representation_contract") != representation_record
            or receipt.get("encoder_checkpoint") != representation["encoder_checkpoint"]
        ):
            raise ContractError("capacity smoke receipt is invalid")
        return receipt

    before = enforce_resource_caps(paths, pre_stage=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device="cpu").manual_seed(int(core.BOOTSTRAP_SEED) + 404)
    embeddings = torch.randn(
        BATCH_SIZE, 576, 128, generator=generator, dtype=torch.float32
    ).to(device)
    flat = torch.rand(BATCH_SIZE, 3, generator=generator, dtype=torch.float32).to(device)
    target = torch.rand(
        BATCH_SIZE, 3, 480, 480, generator=generator, dtype=torch.float32
    ).to(device)
    arms = create_arms(device, amp=amp)
    started = time.perf_counter()
    losses = {
        name: train_one_arm(arms[name], embeddings, flat, target, amp=amp)
        for name in ARM_NAMES
    }
    if not all(math.isfinite(value) for value in losses.values()):
        raise ContractError("capacity smoke produced a non-finite loss")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = float(time.perf_counter() - started)
    del arms, embeddings, flat, target
    torch.cuda.empty_cache()

    representation_record, representation = load_representation_contract(paths, contract_sha)
    paired = _load_paired_encoder(
        Path(str(representation["encoder_checkpoint"]["path"])), device,
        contract_sha256=contract_sha,
    )
    tiles = torch.rand(576, 3, 20, 20, generator=generator, dtype=torch.float32).to(device)
    encoder_context = (
        torch.autocast("cuda", dtype=torch.float16)
        if amp and device.type == "cuda" else nullcontext()
    )
    encoder_seconds: list[float] = []
    with torch.inference_mode(), encoder_context:
        for _ in range(2):
            paired.dirty_encoder(tiles)
        torch.cuda.synchronize(device)
        for _ in range(5):
            encoder_started = time.perf_counter()
            paired.dirty_encoder(tiles)
            torch.cuda.synchronize(device)
            encoder_seconds.append(float(time.perf_counter() - encoder_started))
    del paired, tiles
    torch.cuda.empty_cache()
    peak = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    if peak > MAX_GPU_ALLOCATED_BYTES:
        raise ContractError(
            f"capacity smoke peak {peak} exceeded frozen cap {MAX_GPU_ALLOCATED_BYTES}"
        )
    receipt = {
        "schema": CAPACITY_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha,
        "passed": True,
        "synthetic": True,
        "shapes": {
            "embeddings": [BATCH_SIZE, 576, 128],
            "flat": [BATCH_SIZE, 3],
            "target": [BATCH_SIZE, 3, 480, 480],
        },
        "arms": list(ARM_NAMES),
        "amp_model_only": bool(amp),
        "render_ssim_fp32": True,
        "loss_definition": "1-mean_uniform_ssim_float_proxy",
        "losses": losses,
        "elapsed_seconds": elapsed,
        "paired_dirty_encoder": {
            "synthetic": True,
            "shape": [576, 3, 20, 20],
            "warmup_runs": 2,
            "timed_runs": 5,
            "seconds": encoder_seconds,
            "median_seconds": float(np.median(encoder_seconds)),
            "mean_seconds": float(np.mean(encoder_seconds)),
            "encoder_checkpoint": dict(representation["encoder_checkpoint"]),
        },
        "representation_contract": representation_record,
        "encoder_checkpoint": dict(representation["encoder_checkpoint"]),
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
        "max_gpu_allocated_bytes": MAX_GPU_ALLOCATED_BYTES,
        "resources_before": before,
    }
    atomic_json(receipt_path, receipt)
    enforce_resource_caps(paths)
    return receipt


def _checkpoint_payload(arms: Mapping[str, Arm], *, step: int,
                        contract_sha256: str, losses: Mapping[str, float]) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "contract_sha256": contract_sha256,
        "step": int(step),
        "loss_definition": "1-mean_uniform_ssim_float_proxy",
        "losses": dict(losses),
        "arms": {
            name: {
                "model": arm.model.state_dict(),
                "optimizer": arm.optimizer.state_dict(),
                "scheduler": arm.scheduler.state_dict(),
                "scaler": arm.scaler.state_dict(),
                "kind": arm.kind,
                "blind": arm.blind,
            }
            for name, arm in arms.items()
        },
    }


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(payload), stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(paths: RunPaths, arms: Mapping[str, Arm], *, step: int,
                    contract_sha256: str, losses: Mapping[str, float]) -> dict[str, Any]:
    checkpoint = paths.checkpoints / f"step_{step:07d}.pt"
    receipt_path = paths.receipts / f"checkpoint_step_{step:07d}.json"
    if checkpoint.exists() or receipt_path.exists():
        raise ContractError(f"checkpoint step {step} already exists without resume skip")
    _atomic_torch(checkpoint, _checkpoint_payload(
        arms, step=step, contract_sha256=contract_sha256, losses=losses
    ))
    receipt = {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha256,
        "step": int(step),
        "loss_definition": "1-mean_uniform_ssim_float_proxy",
        "checkpoint": path_record(checkpoint),
        "losses": dict(losses),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _safe_checkpoint_payload(path: Path, contract_sha256: str, step: int) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ContractError(f"cannot load recovery checkpoint safely: {exc}") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("contract_sha256") != contract_sha256
        or int(payload.get("step", -1)) != step
        or payload.get("loss_definition") != "1-mean_uniform_ssim_float_proxy"
        or not isinstance(payload.get("losses"), Mapping)
        or not isinstance(payload.get("arms"), Mapping)
        or set(payload["arms"]) != set(ARM_NAMES)
    ):
        raise ContractError(f"recovery checkpoint identity mismatch at step {step}")
    return payload


def _verified_checkpoint_receipts(paths: RunPaths, contract_sha256: str) -> list[dict[str, Any]]:
    checkpoint_for_step: dict[int, Path] = {}
    receipt_for_step: dict[int, Path] = {}
    for path in paths.checkpoints.glob("step_*.pt"):
        match = re.fullmatch(r"step_(\d{7})\.pt", path.name)
        if match is None:
            raise ContractError(f"unexpected checkpoint filename: {path.name}")
        checkpoint_for_step[int(match.group(1))] = path
    for path in paths.receipts.glob("checkpoint_step_*.json"):
        match = re.fullmatch(r"checkpoint_step_(\d{7})\.json", path.name)
        if match is None:
            raise ContractError(f"unexpected checkpoint receipt filename: {path.name}")
        receipt_for_step[int(match.group(1))] = path
    all_steps = sorted(set(checkpoint_for_step) | set(receipt_for_step))
    for step in all_steps:
        if step < CHECKPOINT_EVERY or step > TRAIN_STEPS or step % CHECKPOINT_EVERY:
            raise ContractError(f"invalid checkpoint step {step}")
    if all_steps:
        expected = list(range(CHECKPOINT_EVERY, all_steps[-1] + 1, CHECKPOINT_EVERY))
        if all_steps != expected:
            raise ContractError("checkpoint artifacts are not a contiguous 100-step prefix")

    receipts: list[dict[str, Any]] = []
    for step in all_steps:
        checkpoint = checkpoint_for_step.get(step)
        receipt_path = receipt_for_step.get(step)
        if checkpoint is None:
            raise ContractError(f"checkpoint receipt step {step} lost its checkpoint")
        payload = _safe_checkpoint_payload(checkpoint, contract_sha256, step)
        if receipt_path is None:
            # Recover the only legitimate crash window: durable checkpoint was
            # renamed, but its small authenticated receipt was not yet created.
            receipt = {
                "schema": CHECKPOINT_RECEIPT_SCHEMA,
                "contract_sha256": contract_sha256,
                "step": step,
                "loss_definition": "1-mean_uniform_ssim_float_proxy",
                "checkpoint": path_record(checkpoint),
                "losses": {name: float(payload["losses"][name]) for name in ARM_NAMES},
            }
            atomic_json(paths.receipts / f"checkpoint_step_{step:07d}.json", receipt)
        else:
            receipt = read_json(receipt_path)
            if (
                set(receipt) != {
                    "schema", "contract_sha256", "step", "loss_definition",
                    "checkpoint", "losses",
                }
                or receipt.get("schema") != CHECKPOINT_RECEIPT_SCHEMA
                or receipt.get("contract_sha256") != contract_sha256
                or int(receipt.get("step", -1)) != step
                or receipt.get("loss_definition") != "1-mean_uniform_ssim_float_proxy"
            ):
                raise ContractError(f"checkpoint receipt identity mismatch: {receipt_path}")
            verify_path_record(receipt["checkpoint"], label=f"checkpoint step {step}")
            if {name: float(payload["losses"][name]) for name in ARM_NAMES} != {
                name: float(receipt["losses"][name]) for name in ARM_NAMES
            }:
                raise ContractError(f"checkpoint loss receipt drift at step {step}")
        receipts.append(receipt)
    return receipts


def _value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_value_to_device(item, device) for item in value)
    return value


def optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = _value_to_device(state, device)


def load_latest_checkpoint(paths: RunPaths, arms: Mapping[str, Arm],
                           contract_sha256: str) -> tuple[int, dict[str, Any] | None]:
    receipts = _verified_checkpoint_receipts(paths, contract_sha256)
    if not receipts:
        return 0, None
    receipt = receipts[-1]
    checkpoint_path = Path(str(receipt["checkpoint"]["path"]))
    payload = _safe_checkpoint_payload(checkpoint_path, contract_sha256, int(receipt["step"]))
    states = payload.get("arms")
    if not isinstance(states, Mapping) or set(states) != set(ARM_NAMES):
        raise ContractError("recovery checkpoint arm set mismatch")
    for name, arm in arms.items():
        row = states[name]
        if row.get("kind") != arm.kind or bool(row.get("blind")) != arm.blind:
            raise ContractError(f"recovery checkpoint arm contract mismatch: {name}")
        arm.model.load_state_dict(row["model"], strict=True)
        arm.optimizer.load_state_dict(row["optimizer"])
        optimizer_state_to_device(arm.optimizer, next(arm.model.parameters()).device)
        arm.scheduler.load_state_dict(row["scheduler"])
        arm.scaler.load_state_dict(row["scaler"])
    return int(receipt["step"]), receipt


def train_arms(
    *, paths: RunPaths, contract: Mapping[str, Any], split: SplitData,
    device: torch.device, amp: bool,
) -> dict[str, Arm]:
    names = tuple(sorted(split.names["fit"], key=_board_id))
    embeddings_np, flat_np, _ = load_cache_arrays(
        paths, "fit", names, str(contract["contract_sha256"])
    )
    enforce_resource_caps(paths, pre_stage=True)
    torch.cuda.reset_peak_memory_stats(device)
    arms = create_arms(device, amp=amp)
    start, receipt = load_latest_checkpoint(paths, arms, str(contract["contract_sha256"]))
    if start >= TRAIN_STEPS:
        return arms
    target_root = Path(str(contract["data"]["targets"]))
    windows: dict[str, list[float]] = {name: [] for name in ARM_NAMES}
    write_status(paths, str(contract["contract_sha256"]), state="training", step=start,
                 message=f"resuming four paired arms from step {start}", checkpoint=receipt)
    for step in range(start + 1, TRAIN_STEPS + 1):
        indices = stateless_batch_indices(len(names), BATCH_SIZE, step, int(core.BOOTSTRAP_SEED))
        batch_names = [names[int(index)] for index in indices]
        embedding = torch.from_numpy(np.asarray(embeddings_np[indices]).astype(np.float32)).to(device)
        flat = torch.from_numpy(np.asarray(flat_np[indices]).astype(np.float32)).to(device)
        target = _load_targets(target_root, batch_names, device)
        for name in ARM_NAMES:
            windows[name].append(train_one_arm(arms[name], embedding, flat, target, amp=amp))
        if step % CHECKPOINT_EVERY == 0:
            enforce_resource_caps(paths)
            mean_losses = {name: float(np.mean(windows[name])) for name in ARM_NAMES}
            receipt = save_checkpoint(
                paths, arms, step=step, contract_sha256=str(contract["contract_sha256"]),
                losses=mean_losses,
            )
            windows = {name: [] for name in ARM_NAMES}
            write_status(paths, str(contract["contract_sha256"]), state="training", step=step,
                         message=f"completed paired step {step}/{TRAIN_STEPS}", checkpoint=receipt)
            print(json.dumps({"step": step, "total": TRAIN_STEPS, "losses": mean_losses}), flush=True)
    enforce_resource_caps(paths)
    return arms


def validate_raw_arrays(arrays: Mapping[str, np.ndarray], *, count: int) -> None:
    if set(arrays) != set(RAW_KEYS):
        raise ContractError(f"raw arrays need exact keys {RAW_KEYS}")
    for key in ID_KEYS:
        value = np.asarray(arrays[key])
        if value.shape != (count,) or value.dtype != np.int64:
            raise ContractError(f"{key} must be int64[{count}]")
    for key in SSIM_KEYS:
        value = np.asarray(arrays[key])
        if value.shape != (count,) or value.dtype != np.float64:
            raise ContractError(f"{key} must be float64[{count}]")
        if not np.isfinite(value).all() or (value < -1.0).any() or (value > 1.0).any():
            raise ContractError(f"{key} contains invalid SSIM values")
    prediction_shapes = {
        "flat_rgb": (count, 3),
        "dct_full_coeff": (count, core.DCT_OUTPUT_DIM),
        "dct_blind_coeff": (count, core.DCT_OUTPUT_DIM),
        "dct_swapped_coeff": (count, core.DCT_OUTPUT_DIM),
        "rgb8_full_residual": (count, core.RGB_OUTPUT_DIM),
        "rgb8_blind_residual": (count, core.RGB_OUTPUT_DIM),
    }
    for key, shape in prediction_shapes.items():
        value = np.asarray(arrays[key])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ContractError(f"{key} must be finite float32{shape}")
    flat_rgb = np.asarray(arrays["flat_rgb"])
    if (flat_rgb < 0.0).any() or (flat_rgb > 1.0).any():
        raise ContractError("flat_rgb must lie in [0,1]")
    if not np.all(np.diff(np.asarray(arrays["board_id"])) > 0):
        raise ContractError("board_id must be strictly increasing")
    cycle = np.asarray(arrays["swap_cycle_id"])
    if (cycle < 0).any():
        raise ContractError("swap_cycle_id cannot be negative")
    unique, counts = np.unique(cycle, return_counts=True)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise ContractError("swap_cycle_id must be dense and canonical")
    if (counts < 2).any():
        raise ContractError("swap_cycle_id cannot contain fixed-point components")
    if len(unique) < 64:
        raise ContractError("swap_cycle_id needs at least 64 bootstrap components")


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_exact_npz(path: Path, keys: Sequence[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(keys):
                raise ContractError(f"{path.name} has wrong NPZ keys")
            return {key: np.array(archive[key], copy=True) for key in keys}
    except (OSError, ValueError, KeyError) as exc:
        raise ContractError(f"cannot load numeric artifact {path}: {exc}") from exc


def _write_or_recover_npz(
    *, path: Path, receipt_path: Path, schema: str, contract_sha256: str,
    arrays: Mapping[str, np.ndarray] | None, keys: Sequence[str],
    validate: Any, receipt_extra: Mapping[str, Any], max_bytes: int = 32 << 20,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Create/recover an NPZ+receipt pair across either atomic crash window."""
    if receipt_path.exists() and not path.exists():
        raise ContractError(f"receipt exists but artifact is missing: {path}")
    if path.exists():
        loaded = _load_exact_npz(path, keys)
        validate(loaded)
        record = path_record(path)
        if record["bytes"] > max_bytes:
            raise ContractError(f"numeric artifact exceeds {max_bytes} bytes: {path}")
        expected_receipt = {
            "schema": schema,
            "contract_sha256": contract_sha256,
            "artifact": record,
            **dict(receipt_extra),
        }
        if receipt_path.exists():
            if read_json(receipt_path) != expected_receipt:
                raise ContractError(f"numeric artifact receipt drift: {receipt_path}")
        else:
            atomic_json(receipt_path, expected_receipt)
        return record, loaded
    if arrays is None:
        raise ContractError(f"numeric artifact is absent and no recovery arrays were supplied: {path}")
    normalized = {key: np.asarray(arrays[key]) for key in keys}
    if set(arrays) != set(keys):
        raise ContractError(f"creation arrays for {path.name} have wrong keys")
    validate(normalized)
    _atomic_npz(path, normalized)
    return _write_or_recover_npz(
        path=path, receipt_path=receipt_path, schema=schema,
        contract_sha256=contract_sha256, arrays=None, keys=keys,
        validate=validate, receipt_extra=receipt_extra, max_bytes=max_bytes,
    )


ORACLE_KEYS = (
    "board_id", "source_group_id", "flat_ssim", "target_oracle_dct_ssim",
    "flat_rgb", "oracle_coeff",
)
ORACLE_CAL_SHARED_KEYS = (
    "board_id", "source_group_id", "flat_ssim", "target_oracle_dct_ssim", "flat_rgb"
)
FIT_PALETTE_KEYS = ("fit_board_id", "dirty_feature60", "mean60", "scale60")
SWAP_KEYS = (
    "board_id", "source_group_id", "donor_board_id", "donor_source_group_id",
    "swap_cycle_id", "dirty_feature60",
)


def validate_oracle_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != set(ORACLE_KEYS):
        raise ContractError("oracle evidence has wrong keys")
    for key in ("board_id", "source_group_id"):
        if np.asarray(arrays[key]).shape != (CAL_COUNT,) or np.asarray(arrays[key]).dtype != np.int64:
            raise ContractError(f"oracle {key} must be int64[670]")
    if not np.all(np.diff(np.asarray(arrays["board_id"])) > 0):
        raise ContractError("oracle board_id must be strictly increasing")
    for key in ("flat_ssim", "target_oracle_dct_ssim"):
        value = np.asarray(arrays[key])
        if value.shape != (CAL_COUNT,) or value.dtype != np.float64:
            raise ContractError(f"oracle {key} must be float64[670]")
        if not np.isfinite(value).all() or (value < -1.0).any() or (value > 1.0).any():
            raise ContractError(f"oracle {key} contains invalid SSIM")
    for key, shape in (
        ("flat_rgb", (CAL_COUNT, 3)),
        ("oracle_coeff", (CAL_COUNT, core.DCT_OUTPUT_DIM)),
    ):
        value = np.asarray(arrays[key])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ContractError(f"oracle {key} must be finite float32{shape}")
    if (np.asarray(arrays["flat_rgb"]) < 0.0).any() or (
        np.asarray(arrays["flat_rgb"]) > 1.0
    ).any():
        raise ContractError("oracle flat_rgb must lie in [0,1]")


def oracle_summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    validate_oracle_arrays(arrays)
    flat = np.asarray(arrays["flat_ssim"], dtype=np.float64)
    oracle = np.asarray(arrays["target_oracle_dct_ssim"], dtype=np.float64)
    return {
        "n_boards": CAL_COUNT,
        "means": {"flat": float(flat.mean()), "target_oracle_dct": float(oracle.mean())},
        "gains": {"target_oracle_dct": float((oracle - flat).mean())},
    }


def oracle_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    observed = float(metrics["gains"]["target_oracle_dct"])
    check = {
        "observed": observed,
        "operator": ">=",
        "threshold": ORACLE_GAIN_MIN,
        "passed": bool(observed >= ORACLE_GAIN_MIN),
    }
    return {"checks": {"oracle_gain": check}, "passed": bool(check["passed"])}


@torch.inference_mode()
def evaluate_cal_oracle(
    *, paths: RunPaths, contract: Mapping[str, Any], split: SplitData,
    device: torch.device, batch_size: int,
) -> dict[str, np.ndarray]:
    names = tuple(sorted(split.names["cal"], key=_board_id))
    _, flat_np, _ = load_cache_arrays(paths, "cal", names, str(contract["contract_sha256"]))
    arrays: dict[str, np.ndarray] = {
        "board_id": np.asarray([_board_id(name) for name in names], dtype=np.int64),
        "source_group_id": np.asarray(
            [split.group_id_for_name[name] for name in names], dtype=np.int64
        ),
        "flat_ssim": np.empty(CAL_COUNT, dtype=np.float64),
        "target_oracle_dct_ssim": np.empty(CAL_COUNT, dtype=np.float64),
        "flat_rgb": np.empty((CAL_COUNT, 3), dtype=np.float32),
        "oracle_coeff": np.empty((CAL_COUNT, core.DCT_OUTPUT_DIM), dtype=np.float32),
    }
    target_root = Path(str(contract["data"]["targets"]))
    for start in range(0, CAL_COUNT, batch_size):
        stop = min(CAL_COUNT, start + batch_size)
        target_uint8 = _load_targets_uint8(target_root, names[start:stop])
        target = torch.from_numpy(target_uint8).permute(0, 3, 1, 2).float().div_(255.0)
        flat = torch.from_numpy(np.asarray(flat_np[start:stop]).astype(np.float32))
        flat_image = flat[:, :, None, None].expand_as(target)
        coeff = core.encode_dct_residual(target.float(), flat.float())
        rendered = core.render_dct_residual(coeff.float(), flat.float(), size=(480, 480))
        arrays["flat_ssim"][start:stop] = official_uint8_ssim(flat_image, target_uint8)
        arrays["target_oracle_dct_ssim"][start:stop] = official_uint8_ssim(
            rendered, target_uint8
        )
        arrays["flat_rgb"][start:stop] = flat.numpy().astype(np.float32)
        arrays["oracle_coeff"][start:stop] = coeff.reshape(
            stop - start, core.DCT_OUTPUT_DIM
        ).cpu().numpy().astype(np.float32)
        print(f"M144 ORACLE CAL {stop}/{CAL_COUNT}", flush=True)
    validate_oracle_arrays(arrays)
    return arrays


def write_or_load_oracle(
    *, paths: RunPaths, contract: Mapping[str, Any], arrays: Mapping[str, np.ndarray] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    path = paths.artifacts / "cal_oracle_pretrain.npz"
    # Metrics/gate are deterministic functions of the numeric artifact; place
    # them in the receipt so an orphan file can be authenticated and recovered.
    if path.exists():
        candidate = _load_exact_npz(path, ORACLE_KEYS)
    elif arrays is not None:
        candidate = {key: np.asarray(arrays[key]) for key in ORACLE_KEYS}
    else:
        raise ContractError("CAL oracle evidence is absent")
    metrics = oracle_summary(candidate)
    gate = oracle_gate(metrics)
    record, loaded = _write_or_recover_npz(
        path=path,
        receipt_path=paths.receipts / "cal_oracle_pretrain.json",
        schema=ORACLE_RECEIPT_SCHEMA,
        contract_sha256=str(contract["contract_sha256"]),
        arrays=arrays,
        keys=ORACLE_KEYS,
        validate=validate_oracle_arrays,
        receipt_extra={
            "metrics": metrics,
            "gate": gate,
            "cal_cache_manifest": path_record(cache_files(paths, "cal").manifest),
            "cal_target_receipt": path_record(paths.receipts / "cal_targets.json"),
        },
        max_bytes=1 << 20,
    )
    return record, loaded, metrics, gate


def validate_fit_palette_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != set(FIT_PALETTE_KEYS):
        raise ContractError("FIT palette evidence has wrong keys")
    board = np.asarray(arrays["fit_board_id"])
    feature = np.asarray(arrays["dirty_feature60"])
    mean = np.asarray(arrays["mean60"])
    scale = np.asarray(arrays["scale60"])
    if board.shape != (FIT_COUNT,) or board.dtype != np.int64 or not np.all(np.diff(board) > 0):
        raise ContractError("fit_board_id must be increasing int64[5360]")
    if feature.shape != (FIT_COUNT, PALETTE_DIM) or feature.dtype != np.float64:
        raise ContractError("FIT dirty_feature60 must be float64[5360,60]")
    if mean.shape != (PALETTE_DIM,) or mean.dtype != np.float64:
        raise ContractError("FIT mean60 must be float64[60]")
    if scale.shape != (PALETTE_DIM,) or scale.dtype != np.float64:
        raise ContractError("FIT scale60 must be float64[60]")
    if not all(np.isfinite(value).all() for value in (feature, mean, scale)):
        raise ContractError("FIT palette evidence is non-finite")
    expected_mean = feature.mean(axis=0, dtype=np.float64)
    expected_scale = np.maximum(feature.std(axis=0, ddof=0, dtype=np.float64), 1.0e-6)
    if not np.array_equal(mean, expected_mean) or not np.array_equal(scale, expected_scale):
        raise ContractError("FIT whitening statistics do not exactly match dirty features")


def build_or_load_fit_palette(
    *, paths: RunPaths, contract: Mapping[str, Any], split: SplitData,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    names = tuple(sorted(split.names["fit"], key=_board_id))
    _, _, feature_np = load_cache_arrays(paths, "fit", names, str(contract["contract_sha256"]))
    feature = np.asarray(feature_np, dtype=np.float64)
    arrays = {
        "fit_board_id": np.asarray([_board_id(name) for name in names], dtype=np.int64),
        "dirty_feature60": feature,
        "mean60": feature.mean(axis=0, dtype=np.float64),
        "scale60": np.maximum(feature.std(axis=0, ddof=0, dtype=np.float64), 1.0e-6),
    }
    record, loaded = _write_or_recover_npz(
        path=paths.artifacts / "fit_palette_whitening.npz",
        receipt_path=paths.receipts / "fit_palette_whitening.json",
        schema=SWAP_WHITENING_SCHEMA,
        contract_sha256=str(contract["contract_sha256"]),
        arrays=arrays,
        keys=FIT_PALETTE_KEYS,
        validate=validate_fit_palette_arrays,
        receipt_extra={
            "fit_cache_manifest": path_record(cache_files(paths, "fit").manifest),
            "algorithm": "FIT_population_mean_std_ddof0_scale_max_1e-6_float64",
        },
        max_bytes=8 << 20,
    )
    return record, loaded


def validate_swap_arrays(arrays: Mapping[str, np.ndarray], *, count: int = CAL_COUNT) -> None:
    if set(arrays) != set(SWAP_KEYS):
        raise ContractError("swap evidence has wrong keys")
    for key in SWAP_KEYS[:-1]:
        value = np.asarray(arrays[key])
        if value.shape != (count,) or value.dtype != np.int64:
            raise ContractError(f"swap {key} must be int64[{count}]")
    feature = np.asarray(arrays["dirty_feature60"])
    if feature.shape != (count, PALETTE_DIM) or feature.dtype != np.float64:
        raise ContractError(f"swap dirty_feature60 must be float64[{count},60]")
    if not np.isfinite(feature).all():
        raise ContractError("swap dirty_feature60 is non-finite")
    board = np.asarray(arrays["board_id"])
    donor_board = np.asarray(arrays["donor_board_id"])
    groups = np.asarray(arrays["source_group_id"])
    donor_groups = np.asarray(arrays["donor_source_group_id"])
    if not np.all(np.diff(board) > 0) or set(donor_board.tolist()) != set(board.tolist()):
        raise ContractError("swap donor_board_id is not a permutation of sorted board_id")
    if np.any(board == donor_board) or np.any(groups == donor_groups):
        raise ContractError("swap evidence contains a fixed/group collision")
    donor_index = np.searchsorted(board, donor_board)
    if (
        np.any(donor_index >= count)
        or not np.array_equal(board[donor_index], donor_board)
        or not np.array_equal(groups[donor_index], donor_groups)
    ):
        raise ContractError("swap donor source-group evidence does not match donor IDs")
    cycle = np.asarray(arrays["swap_cycle_id"])
    unique, sizes = np.unique(cycle, return_counts=True)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)) or np.any(sizes < 2):
        raise ContractError("swap cycles must be dense arbitrary components of size >=2")
    if not np.array_equal(cycle, canonical_cycle_ids(donor_index.astype(np.int64), board)):
        raise ContractError("swap_cycle_id does not canonically label donor cycles")


def swap_cycle_stats(cycle_ids: np.ndarray) -> dict[str, Any]:
    cycle = np.asarray(cycle_ids)
    if cycle.ndim != 1 or cycle.dtype != np.int64 or (cycle < 0).any():
        raise ContractError("cycle statistics require non-negative int64[N]")
    unique, lengths = np.unique(cycle, return_counts=True)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)) or len(unique) < 1:
        raise ContractError("cycle statistics require dense IDs")
    sizes, frequencies = np.unique(lengths, return_counts=True)
    return {
        "count": int(len(lengths)),
        "min_size": int(lengths.min()),
        "max_size": int(lengths.max()),
        "mean_size": float(len(cycle) / len(lengths)),
        "median_size": float(np.median(lengths)),
        "size_histogram": [
            [int(size), int(frequency)]
            for size, frequency in zip(sizes, frequencies, strict=True)
        ],
    }


def build_or_load_swap(
    *, paths: RunPaths, partition: str, contract: Mapping[str, Any], split: SplitData,
    fit_palette_record: Mapping[str, Any], fit_palette: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import scipy

    names = tuple(sorted(split.names[partition], key=_board_id))
    _, _, feature_np = load_cache_arrays(
        paths, partition, names, str(contract["contract_sha256"])
    )
    feature = np.asarray(feature_np, dtype=np.float64)
    board = np.asarray([_board_id(name) for name in names], dtype=np.int64)
    groups = np.asarray([split.group_id_for_name[name] for name in names], dtype=np.int64)
    donor, cycle = solve_swap_assignment(
        feature, groups, board,
        np.asarray(fit_palette["mean60"]), np.asarray(fit_palette["scale60"]),
    )
    arrays = {
        "board_id": board,
        "source_group_id": groups,
        "donor_board_id": board[donor],
        "donor_source_group_id": groups[donor],
        "swap_cycle_id": cycle,
        "dirty_feature60": feature,
    }
    cycle_stats = swap_cycle_stats(cycle)
    receipt_extra = {
        "partition": partition,
        "scipy_version": str(scipy.__version__),
        "algorithm": "LSA_squared_euclidean_sum_float64_no_jitter_forbid_self_source_v1",
        "fit_palette": dict(fit_palette_record),
        "partition_cache_manifest": path_record(cache_files(paths, partition).manifest),
        "cycle_stats": cycle_stats,
    }
    record, loaded = _write_or_recover_npz(
        path=paths.artifacts / f"swap_{partition}.npz",
        receipt_path=paths.receipts / f"swap_{partition}.json",
        schema=SWAP_RECEIPT_SCHEMA,
        contract_sha256=str(contract["contract_sha256"]),
        arrays=arrays,
        keys=SWAP_KEYS,
        validate=lambda value: validate_swap_arrays(value, count=len(names)),
        receipt_extra=receipt_extra,
        max_bytes=2 << 20,
    )
    # Recompute assignment even for a recovered artifact: this authenticates
    # SciPy/version-dependent donor bytes before any swapped metric is opened.
    donor_index = np.searchsorted(board, loaded["donor_board_id"])
    expected_donor, expected_cycle = solve_swap_assignment(
        loaded["dirty_feature60"], loaded["source_group_id"], loaded["board_id"],
        fit_palette["mean60"], fit_palette["scale60"],
    )
    if not np.array_equal(donor_index, expected_donor) or not np.array_equal(
        loaded["swap_cycle_id"], expected_cycle
    ):
        raise ContractError(f"{partition} committed donor bytes differ from frozen LSA")
    loaded_stats = swap_cycle_stats(np.asarray(loaded["swap_cycle_id"]))
    if loaded_stats != cycle_stats:
        raise ContractError(f"{partition} committed cycle statistics drift")
    if int(loaded_stats["count"]) < 64:
        raise ContractError(
            f"{partition} swap has only {loaded_stats['count']} cycles; minimum is 64"
        )
    return record, loaded


@torch.inference_mode()
def evaluate_partition(
    *, paths: RunPaths, partition: str, contract: Mapping[str, Any], split: SplitData,
    arms: Mapping[str, Arm], swap_arrays: Mapping[str, np.ndarray],
    device: torch.device, amp: bool, batch_size: int,
) -> dict[str, np.ndarray]:
    names = tuple(sorted(split.names[partition], key=_board_id))
    count = len(names)
    if count != 670:
        raise ContractError(f"{partition} evaluation requires exactly 670 boards")
    embeddings_np, flat_np, _ = load_cache_arrays(
        paths, partition, names, str(contract["contract_sha256"])
    )
    validate_swap_arrays(swap_arrays, count=count)
    board_ids = np.asarray([_board_id(name) for name in names], dtype=np.int64)
    if not np.array_equal(np.asarray(swap_arrays["board_id"]), board_ids):
        raise ContractError(f"{partition} swap artifact does not match evaluation board order")
    donor = np.searchsorted(board_ids, np.asarray(swap_arrays["donor_board_id"])).astype(
        np.int64, copy=False
    )
    arrays: dict[str, np.ndarray] = {
        "board_id": board_ids,
        "source_group_id": np.asarray(
            [split.group_id_for_name[name] for name in names], dtype=np.int64
        ),
        "swap_cycle_id": np.asarray(swap_arrays["swap_cycle_id"], dtype=np.int64),
    }
    for key in SSIM_KEYS:
        arrays[key] = np.empty(count, dtype=np.float64)
    arrays["flat_rgb"] = np.empty((count, 3), dtype=np.float32)
    for key in ("dct_full_coeff", "dct_blind_coeff", "dct_swapped_coeff"):
        arrays[key] = np.empty((count, core.DCT_OUTPUT_DIM), dtype=np.float32)
    for key in ("rgb8_full_residual", "rgb8_blind_residual"):
        arrays[key] = np.empty((count, core.RGB_OUTPUT_DIM), dtype=np.float32)
    target_root = Path(str(contract["data"]["targets"]))
    for arm in arms.values():
        arm.model.eval()
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        index = np.arange(start, stop, dtype=np.int64)
        target_uint8 = _load_targets_uint8(target_root, names[start:stop])
        target_cpu = torch.from_numpy(target_uint8).permute(0, 3, 1, 2).float().div_(255.0)
        embedding = torch.from_numpy(np.asarray(embeddings_np[index]).astype(np.float32)).to(device)
        swapped_embedding = torch.from_numpy(
            np.asarray(embeddings_np[donor[index]]).astype(np.float32)
        ).to(device)
        flat_cpu = torch.from_numpy(np.asarray(flat_np[index]).astype(np.float32))
        flat_gpu = flat_cpu.to(device)
        flat_image = flat_cpu[:, :, None, None].expand_as(target_cpu)
        oracle_coeff = core.encode_dct_residual(target_cpu.float(), flat_cpu.float())
        oracle_image = core.render_dct_residual(
            oracle_coeff.float(), flat_cpu.float(), size=tuple(target_cpu.shape[-2:])
        )
        arrays["flat_ssim"][start:stop] = official_uint8_ssim(flat_image, target_uint8)
        arrays["target_oracle_dct_ssim"][start:stop] = official_uint8_ssim(
            oracle_image, target_uint8
        )
        arrays["flat_rgb"][start:stop] = flat_cpu.numpy().astype(np.float32)

        context = torch.autocast("cuda", dtype=torch.float16) if amp and device.type == "cuda" else nullcontext()
        with context:
            dct_full = arms["dct_full"].model(embedding, flat_gpu, blind=False)
            dct_blind = arms["dct_blind"].model(embedding, flat_gpu, blind=True)
            dct_swapped = arms["dct_full"].model(swapped_embedding, flat_gpu, blind=False)
            rgb_full = arms["rgb_full"].model(embedding, flat_gpu, blind=False)
            rgb_blind = arms["rgb_blind"].model(embedding, flat_gpu, blind=True)
        dct_full_cpu = dct_full.float().cpu()
        dct_blind_cpu = dct_blind.float().cpu()
        dct_swapped_cpu = dct_swapped.float().cpu()
        rgb_full_cpu = rgb_full.float().cpu()
        rgb_blind_cpu = rgb_blind.float().cpu()
        predictions = {
            "dct_full_ssim": ("dct", dct_full_cpu),
            "dct_blind_ssim": ("dct", dct_blind_cpu),
            "dct_swapped_ssim": ("dct", dct_swapped_cpu),
            "rgb8_full_ssim": ("rgb", rgb_full_cpu),
            "rgb8_blind_ssim": ("rgb", rgb_blind_cpu),
        }
        arrays["dct_full_coeff"][start:stop] = dct_full_cpu.numpy().astype(np.float32)
        arrays["dct_blind_coeff"][start:stop] = dct_blind_cpu.numpy().astype(np.float32)
        arrays["dct_swapped_coeff"][start:stop] = dct_swapped_cpu.numpy().astype(np.float32)
        arrays["rgb8_full_residual"][start:stop] = rgb_full_cpu.numpy().astype(np.float32)
        arrays["rgb8_blind_residual"][start:stop] = rgb_blind_cpu.numpy().astype(np.float32)
        for key, (kind, prediction) in predictions.items():
            rendered = _render_arm(
                kind, prediction.float(), flat_cpu.float(), tuple(target_cpu.shape[-2:])
            )
            arrays[key][start:stop] = official_uint8_ssim(rendered, target_uint8)
        print(f"M144 {partition.upper()} {stop}/{count}", flush=True)
    validate_raw_arrays(arrays, count=count)
    return arrays


def write_raw_artifact(paths: RunPaths, partition: str,
                       arrays: Mapping[str, np.ndarray], contract_sha256: str,
                       dependencies: Mapping[str, Any]) -> dict[str, Any]:
    filename = "m144_cal_raw.npz" if partition == "cal" else "m144_dev_raw.npz"
    path = paths.artifacts / filename
    receipt_path = paths.receipts / f"{partition}_raw.json"
    count = CAL_COUNT if partition == "cal" else DEV_COUNT
    record, _ = _write_or_recover_npz(
        path=path,
        receipt_path=receipt_path,
        schema=RAW_RECEIPT_SCHEMA,
        contract_sha256=contract_sha256,
        arrays=arrays,
        keys=RAW_KEYS,
        validate=lambda value: validate_raw_arrays(value, count=count),
        receipt_extra={"partition": partition, "dependencies": dict(dependencies)},
        max_bytes=4 << 20,
    )
    return record


def load_raw_artifact(record: Mapping[str, Any], *, count: int) -> dict[str, np.ndarray]:
    verified = verify_path_record(record, label="raw artifact")
    with np.load(verified["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_raw_arrays(arrays, count=count)
    return arrays


def metric_summary(arrays: Mapping[str, np.ndarray], *, alpha: float) -> dict[str, Any]:
    return core.summarize_arm_metrics(
        flat_ssim=np.asarray(arrays["flat_ssim"], dtype=np.float64),
        target_oracle_dct_ssim=np.asarray(
            arrays["target_oracle_dct_ssim"], dtype=np.float64
        ),
        dct_full_ssim=np.asarray(arrays["dct_full_ssim"], dtype=np.float64),
        dct_blind_ssim=np.asarray(arrays["dct_blind_ssim"], dtype=np.float64),
        dct_swapped_ssim=np.asarray(arrays["dct_swapped_ssim"], dtype=np.float64),
        rgb8_full_ssim=np.asarray(arrays["rgb8_full_ssim"], dtype=np.float64),
        rgb8_blind_ssim=np.asarray(arrays["rgb8_blind_ssim"], dtype=np.float64),
        source_groups=np.asarray(arrays["source_group_id"], dtype=np.int64),
        swap_groups=np.asarray(arrays["swap_cycle_id"], dtype=np.int64),
        bootstrap_samples=int(core.BOOTSTRAP_SAMPLES),
        bootstrap_seed=int(core.BOOTSTRAP_SEED),
        alpha=alpha,
    )


def gate_map(metrics: Mapping[str, Any], partition: str) -> dict[str, Any]:
    if partition == "CAL":
        return core.evaluate_cal_gates(metrics)
    elif partition == "DEV":
        return core.evaluate_dev_gates(metrics)
    else:
        raise ValueError(partition)


def protocol_record() -> dict[str, Any]:
    # Wire compatibility is exact; the verifier still recomputes every numeric
    # decision independently and does not import this runner or the core.
    from verify_m144_dct_where import REPORT_PROTOCOL

    return copy.deepcopy(REPORT_PROTOCOL)


def report_contract(paths: RunPaths, contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_contract_sha256": str(contract["contract_sha256"]),
        "split_manifest": dict(contract["split_manifest"]),
        "source_manifest": dict(contract["source_manifest"]),
        "prior_evidence": copy.deepcopy(contract["prior_evidence"]),
        "source_files": {
            relative: str(contract["source_files"][relative]["sha256"])
            for relative in SOURCE_CLOSURE
        },
        "runtime": copy.deepcopy(contract["runtime"]),
        "cal_count": CAL_COUNT,
        "dev_count": DEV_COUNT,
    }


def _maybe_receipt(path: Path, present: bool) -> dict[str, Any] | None:
    if not present:
        if path.exists():
            raise ContractError(f"sealed stage unexpectedly has receipt {path}")
        return None
    return path_record(path)


def report_receipts(
    paths: RunPaths, *, encoder_passed: bool, oracle_opened: bool,
    learned_cal: bool, opened_dev: bool,
) -> dict[str, Any]:
    return {
        "encoder_fit_targets": _maybe_receipt(paths.receipts / "fit_targets.json", True),
        "encoder_cal_targets": _maybe_receipt(paths.receipts / "cal_targets.json", True),
        "encoder_final_checkpoint": _maybe_receipt(
            paths.receipts / f"encoder_checkpoint_step_{ENCODER_STEPS:07d}.json", True
        ),
        "encoder_cal_gate": _maybe_receipt(paths.receipts / "encoder_cal_gate.json", True),
        "representation_contract": _maybe_receipt(
            paths.receipts / "representation_contract.json", encoder_passed
        ),
        "capacity": _maybe_receipt(paths.receipts / "capacity_smoke.json", oracle_opened),
        "fit_cache": _maybe_receipt(cache_files(paths, "fit").manifest, oracle_opened),
        "cal_cache": _maybe_receipt(cache_files(paths, "cal").manifest, oracle_opened),
        "oracle_cal": _maybe_receipt(paths.receipts / "cal_oracle_pretrain.json", oracle_opened),
        "fit_palette": _maybe_receipt(
            paths.receipts / "fit_palette_whitening.json", oracle_opened
        ),
        "swap_cal": _maybe_receipt(paths.receipts / "swap_cal.json", learned_cal),
        "final_checkpoint": _maybe_receipt(
            paths.receipts / f"checkpoint_step_{TRAIN_STEPS:07d}.json", learned_cal
        ),
        "raw_cal": _maybe_receipt(paths.receipts / "cal_raw.json", learned_cal),
        "dev_targets": _maybe_receipt(paths.receipts / "dev_targets.json", opened_dev),
        "dev_cache": _maybe_receipt(cache_files(paths, "dev").manifest, opened_dev),
        "swap_dev": _maybe_receipt(paths.receipts / "swap_dev.json", opened_dev),
        "raw_dev": _maybe_receipt(paths.receipts / "dev_raw.json", opened_dev),
    }


def make_report(
    *, paths: RunPaths, contract: Mapping[str, Any], status: str, stage: str,
    decision: str, reason: str,
    encoder_checkpoint: Mapping[str, Any], encoder_rank_record: Mapping[str, Any],
    representation_record: Mapping[str, Any] | None,
    oracle_record: Mapping[str, Any] | None,
    fit_palette_record: Mapping[str, Any] | None,
    swap_cal_record: Mapping[str, Any] | None,
    cal_record: Mapping[str, Any] | None,
    final_checkpoint: Mapping[str, Any] | None,
    swap_dev_record: Mapping[str, Any] | None,
    dev_record: Mapping[str, Any] | None,
    encoder_metrics: Mapping[str, Any], encoder_gates: Mapping[str, Any],
    oracle_metrics: Mapping[str, Any] | None, oracle_gates: Mapping[str, Any] | None,
    cal_metrics: Mapping[str, Any] | None, cal_gates: Mapping[str, Any] | None,
    dev_metrics: Mapping[str, Any] | None, dev_gates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    learned_cal = cal_record is not None
    opened_dev = dev_record is not None
    encoder_passed = representation_record is not None
    oracle_opened = oracle_record is not None
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "stage": stage,
        "decision": decision,
        "reason": reason,
        "protocol": protocol_record(),
        "contract": report_contract(paths, contract),
        "encoder_final_checkpoint": dict(encoder_checkpoint),
        "encoder_cal_ranks_npz": dict(encoder_rank_record),
        "representation_contract": (
            dict(representation_record) if representation_record is not None else None
        ),
        "oracle_cal_npz": dict(oracle_record) if oracle_record is not None else None,
        "fit_palette_npz": (
            dict(fit_palette_record) if fit_palette_record is not None else None
        ),
        "swap_cal_npz": dict(swap_cal_record) if swap_cal_record is not None else None,
        "raw_cal_npz": dict(cal_record) if cal_record is not None else None,
        "swap_dev_npz": dict(swap_dev_record) if swap_dev_record is not None else None,
        "raw_dev_npz": dict(dev_record) if dev_record is not None else None,
        "final_checkpoint": dict(final_checkpoint) if final_checkpoint is not None else None,
        "metrics": {
            "ENCODER_CAL": dict(encoder_metrics),
            "ORACLE_CAL": dict(oracle_metrics) if oracle_metrics is not None else None,
            "CAL": dict(cal_metrics) if cal_metrics is not None else None,
            "DEV": dict(dev_metrics) if dev_metrics is not None else None,
        },
        "gates": {
            "ENCODER_CAL": dict(encoder_gates),
            "ORACLE_CAL": dict(oracle_gates) if oracle_gates is not None else None,
            "CAL": dict(cal_gates) if cal_gates is not None else None,
            "DEV": dict(dev_gates) if dev_gates is not None else None,
        },
        "receipts": report_receipts(
            paths, encoder_passed=encoder_passed, oracle_opened=oracle_opened,
            learned_cal=learned_cal, opened_dev=opened_dev,
        ),
        "prohibitions": ["no_test_access", "no_submission", "diagnostic_only"],
    }


def write_terminal_report(
    *, paths: RunPaths, contract_sha256: str, report: Mapping[str, Any], exit_code: int,
) -> dict[str, Any]:
    payload = canonical_json(report)
    if paths.report.exists():
        if paths.report.read_bytes() != payload:
            raise ContractError("terminal report is create-once and differs from prior bytes")
    else:
        _atomic_bytes(paths.report, payload)
    record = path_record(paths.report)
    receipt = {
        "schema": TERMINAL_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha256,
        "status": str(report["status"]),
        "stage": str(report["stage"]),
        "decision": str(report["decision"]),
        "exit_code": int(exit_code),
        "report": record,
    }
    receipt_path = paths.receipts / "terminal_report.json"
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise ContractError("terminal report receipt drift")
    else:
        atomic_json(receipt_path, receipt)
    return receipt


def invoke_independent_verifier(
    *, paths: RunPaths, contract: Mapping[str, Any],
) -> dict[str, Any]:
    invocation_path = paths.receipts / "verification_invocation.json"
    output_path = paths.artifacts / "m144_verification.json"
    verifier_path = Path(__file__).resolve().with_name("verify_m144_dct_where.py")
    report_record = path_record(paths.report)
    verifier_record = path_record(verifier_path)
    if invocation_path.exists():
        receipt = read_json(invocation_path)
        if (
            receipt.get("schema") != VERIFICATION_INVOCATION_SCHEMA
            or receipt.get("contract_sha256") != contract["contract_sha256"]
            or receipt.get("report") != report_record
            or receipt.get("verifier") != verifier_record
            or int(receipt.get("exit_code", -1)) != 0
        ):
            raise ContractError("independent verifier invocation receipt drift")
        verify_path_record(receipt["output"], label="independent verification")
        verify_path_record(receipt["stdout"], label="verifier stdout")
        verify_path_record(receipt["stderr"], label="verifier stderr")
        verification = read_json(output_path)
        if verification.get("valid") is not True:
            raise ContractError("independent verification is not valid")
        return receipt

    command = [
        sys.executable,
        "-B",
        str(verifier_path),
        "--work-root", str(paths.root.resolve()),
        "--contract", str(paths.contract.resolve()),
        "--report", str(paths.report.resolve()),
        "--output", str(output_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        check=False,
        timeout=60 * 60,
    )
    stdout_path = paths.receipts / "verifier.stdout.txt"
    stderr_path = paths.receipts / "verifier.stderr.txt"
    _atomic_bytes(stdout_path, bytes(completed.stdout))
    _atomic_bytes(stderr_path, bytes(completed.stderr))
    receipt = {
        "schema": VERIFICATION_INVOCATION_SCHEMA,
        "contract_sha256": str(contract["contract_sha256"]),
        "report": report_record,
        "verifier": verifier_record,
        "command": command,
        "exit_code": int(completed.returncode),
        "stdout": path_record(stdout_path),
        "stderr": path_record(stderr_path),
        "output": path_record(output_path) if output_path.exists() else None,
    }
    atomic_json(invocation_path, receipt)
    if completed.returncode != 0 or receipt["output"] is None:
        raise ContractError(
            "independent M144 verifier rejected the terminal report; see verifier.stderr.txt"
        )
    verification = read_json(output_path)
    if verification.get("valid") is not True:
        raise ContractError("independent verifier output is not valid")
    return receipt


def authenticated_terminal_return(
    *, paths: RunPaths, contract: Mapping[str, Any],
) -> int | None:
    receipt_path = paths.receipts / "terminal_report.json"
    if not paths.report.exists() and not receipt_path.exists():
        return None
    if receipt_path.exists() and not paths.report.exists():
        raise ContractError("terminal receipt exists but report is missing")
    report = read_json(paths.report)
    status = str(report.get("status", ""))
    expected_exit = 0 if status == "dev_pass" else 20 if status in {
        "encoder_reject", "oracle_reject", "cal_reject", "dev_reject"
    } else -1
    if expected_exit < 0:
        raise ContractError("existing terminal report has an unknown status")
    terminal = write_terminal_report(
        paths=paths,
        contract_sha256=str(contract["contract_sha256"]),
        report=report,
        exit_code=expected_exit,
    )
    invoke_independent_verifier(paths=paths, contract=contract)
    write_status(
        paths,
        str(contract["contract_sha256"]),
        state=status,
        step=(ENCODER_STEPS if status == "encoder_reject" else
              TRAIN_STEPS if status != "oracle_reject" else 0),
        message=f"authenticated terminal rerun: {report['decision']}",
        checkpoint=terminal,
    )
    return expected_exit


def _finalize_decision(
    *, paths: RunPaths, contract: Mapping[str, Any], report: Mapping[str, Any],
    exit_code: int, step: int,
) -> int:
    contract_sha = str(contract["contract_sha256"])
    terminal = write_terminal_report(
        paths=paths, contract_sha256=contract_sha, report=report, exit_code=exit_code
    )
    invoke_independent_verifier(paths=paths, contract=contract)
    write_status(
        paths,
        contract_sha,
        state=str(report["status"]),
        step=step,
        message=f"M144 terminal decision: {report['decision']}",
        checkpoint=terminal,
    )
    return exit_code


def _load_or_evaluate_raw(
    *, paths: RunPaths, partition: str, contract: Mapping[str, Any], split: SplitData,
    arms: Mapping[str, Arm], swap_arrays: Mapping[str, np.ndarray],
    dependencies: Mapping[str, Any], device: torch.device, amp: bool, eval_batch: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    filename = "m144_cal_raw.npz" if partition == "cal" else "m144_dev_raw.npz"
    artifact_path = paths.artifacts / filename
    count = CAL_COUNT if partition == "cal" else DEV_COUNT
    if artifact_path.exists():
        arrays = _load_exact_npz(artifact_path, RAW_KEYS)
        validate_raw_arrays(arrays, count=count)
    else:
        arrays = evaluate_partition(
            paths=paths,
            partition=partition,
            contract=contract,
            split=split,
            arms=arms,
            swap_arrays=swap_arrays,
            device=device,
            amp=amp,
            batch_size=eval_batch,
        )
    record = write_raw_artifact(
        paths, partition, arrays, str(contract["contract_sha256"]), dependencies
    )
    return record, arrays


def _target_sha_map(source_manifest: Path) -> dict[str, str]:
    payload = read_json(source_manifest)
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ContractError("source manifest lacks files mapping")
    result: dict[str, str] = {}
    for name, row in files.items():
        if not isinstance(row, Mapping) or not HEX64.fullmatch(str(row.get("sha256", ""))):
            raise ContractError(f"source manifest target hash invalid: {name}")
        result[str(name)] = str(row["sha256"])
    return result


def prepare(args: argparse.Namespace, *, allow_non_e: bool = False) -> tuple[RunPaths, SplitData, dict[str, Any]]:
    paths = RunPaths(_require_drive_e(Path(args.work_root), "M144 work root", allow_non_e=allow_non_e))
    _require_drive_e(Path(args.data_root), "M144 data root", allow_non_e=allow_non_e)
    _require_drive_e(Path(args.split), "M144 split", allow_non_e=allow_non_e)
    _require_drive_e(Path(args.source_manifest), "M144 source manifest", allow_non_e=allow_non_e)
    paths.ensure()
    split = load_split_data(Path(args.split), Path(args.source_manifest))
    contract = build_contract(args, paths, split)
    contract = freeze_or_verify_contract(paths.contract, contract)
    return paths, split, contract


def require_cuda(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ContractError("M144 production cache/training requires CUDA")
    return device


def run_cache(args: argparse.Namespace, paths: RunPaths, split: SplitData,
              contract: Mapping[str, Any]) -> int | None:
    device = require_cuda(args.device)
    enforce_resource_caps(paths, pre_stage=True)
    target_sha = _target_sha_map(Path(args.source_manifest))
    fit_names = tuple(sorted(split.names["fit"], key=_board_id))
    cal_names = tuple(sorted(split.names["cal"], key=_board_id))
    authenticate_partition_targets(
        paths=paths, partition="fit", names=fit_names, contract=contract,
        target_sha_for_name=target_sha,
    )
    authenticate_partition_targets(
        paths=paths, partition="cal", names=cal_names, contract=contract,
        target_sha_for_name=target_sha,
    )
    encoder, encoder_final_receipt = train_fit_only_encoder(
        paths=paths, contract=contract, split=split, device=device, amp=bool(args.amp)
    )
    encoder_checkpoint = dict(encoder_final_receipt["checkpoint"])
    if (paths.artifacts / "encoder_cal_ranks.npz").exists():
        encoder_rank_record, _, encoder_metrics, encoder_gates = write_or_load_encoder_gate(
            paths=paths, contract=contract, final_checkpoint=encoder_checkpoint,
            arrays=None,
        )
    else:
        encoder_arrays = evaluate_fit_encoder_cal(
            model=encoder, contract=contract, split=split, device=device, amp=bool(args.amp)
        )
        encoder_rank_record, _, encoder_metrics, encoder_gates = write_or_load_encoder_gate(
            paths=paths, contract=contract, final_checkpoint=encoder_checkpoint,
            arrays=encoder_arrays,
        )
    del encoder
    torch.cuda.empty_cache()
    if not encoder_gates["passed"]:
        report = make_report(
            paths=paths, contract=contract,
            status="encoder_reject", stage="encoder_cal_gate",
            decision="KILL_DCT_WHERE", reason="encoder_generalization_gate_failed",
            encoder_checkpoint=encoder_checkpoint,
            encoder_rank_record=encoder_rank_record,
            representation_record=None, oracle_record=None, fit_palette_record=None,
            swap_cal_record=None, cal_record=None, final_checkpoint=None,
            swap_dev_record=None, dev_record=None,
            encoder_metrics=encoder_metrics, encoder_gates=encoder_gates,
            oracle_metrics=None, oracle_gates=None, cal_metrics=None, cal_gates=None,
            dev_metrics=None, dev_gates=None,
        )
        return _finalize_decision(
            paths=paths, contract=contract, report=report, exit_code=20,
            step=ENCODER_STEPS,
        )
    write_or_load_representation_contract(
        paths=paths, contract=contract, checkpoint=encoder_checkpoint,
        rank_record=encoder_rank_record,
    )
    run_capacity_smoke(paths=paths, contract=contract, device=device, amp=bool(args.amp))
    for partition in ("fit", "cal"):
        names = tuple(sorted(split.names[partition], key=_board_id))
        build_embedding_cache(
            paths=paths, partition=partition, names=names, contract=contract,
            device=device, amp=bool(args.amp), chunk_size=int(args.cache_chunk),
        )
        enforce_resource_caps(paths)
    build_or_load_fit_palette(paths=paths, contract=contract, split=split)
    write_status(paths, str(contract["contract_sha256"]), state="pre_oracle_ready", step=0,
                 message="capacity, dirty FIT/CAL caches and FIT palette authenticated; DEV sealed")
    return None


def command_train(args: argparse.Namespace, paths: RunPaths, split: SplitData,
                  contract: Mapping[str, Any]) -> int:
    device = require_cuda(args.device)
    contract_sha = str(contract["contract_sha256"])
    # The encoder stage runs in `command_cache`, so its records have to be
    # reloaded here before any report can be written.  Every make_report call
    # in this function needs them and none of them used to supply them, which
    # made all three exits -- oracle reject, CAL reject and the successful DEV
    # path -- raise TypeError after the evidence had already been computed
    # (M147: the CAL reject fired for real after 670 boards of evaluation).
    representation_record, representation_value = load_representation_contract(
        paths, contract_sha
    )
    encoder_checkpoint = dict(representation_value["encoder_checkpoint"])
    encoder_rank_record, _, encoder_metrics, encoder_gates = write_or_load_encoder_gate(
        paths=paths, contract=contract, final_checkpoint=encoder_checkpoint,
        arrays=None,
    )
    for partition in ("fit", "cal"):
        names = tuple(sorted(split.names[partition], key=_board_id))
        _verify_cache_manifest(cache_files(paths, partition), contract_sha, names)
    run_capacity_smoke(paths=paths, contract=contract, device=device, amp=bool(args.amp))
    fit_palette_record, fit_palette = build_or_load_fit_palette(
        paths=paths, contract=contract, split=split
    )
    target_sha = _target_sha_map(Path(args.source_manifest))
    cal_names = tuple(sorted(split.names["cal"], key=_board_id))
    authenticate_partition_targets(
        paths=paths, partition="cal", names=cal_names, contract=contract,
        target_sha_for_name=target_sha,
    )
    write_status(paths, contract_sha, state="oracle_pretrain", step=0,
                 message="evaluating CAL DCT32 target-oracle capacity; DEV sealed")
    if (paths.artifacts / "cal_oracle_pretrain.npz").exists():
        oracle_record, oracle_arrays, oracle_metrics, oracle_gates = write_or_load_oracle(
            paths=paths, contract=contract, arrays=None
        )
    else:
        oracle_arrays = evaluate_cal_oracle(
            paths=paths, contract=contract, split=split, device=device,
            batch_size=int(args.eval_batch),
        )
        oracle_record, oracle_arrays, oracle_metrics, oracle_gates = write_or_load_oracle(
            paths=paths, contract=contract, arrays=oracle_arrays
        )
    if not oracle_gates["passed"]:
        report = make_report(
            paths=paths, contract=contract,
            status="oracle_reject", stage="oracle_pretrain", decision="KILL_DCT_WHERE",
            reason="oracle_capacity_gate_failed",
            encoder_checkpoint=encoder_checkpoint,
            encoder_rank_record=encoder_rank_record,
            representation_record=representation_record,
            encoder_metrics=encoder_metrics, encoder_gates=encoder_gates,
            oracle_record=oracle_record, fit_palette_record=fit_palette_record,
            swap_cal_record=None, cal_record=None, final_checkpoint=None,
            swap_dev_record=None, dev_record=None,
            oracle_metrics=oracle_metrics, oracle_gates=oracle_gates,
            cal_metrics=None, cal_gates=None, dev_metrics=None, dev_gates=None,
        )
        return _finalize_decision(
            paths=paths, contract=contract, report=report, exit_code=20, step=0
        )

    swap_cal_record, swap_cal = build_or_load_swap(
        paths=paths, partition="cal", contract=contract, split=split,
        fit_palette_record=fit_palette_record, fit_palette=fit_palette,
    )
    fit_names = tuple(sorted(split.names["fit"], key=_board_id))
    authenticate_partition_targets(
        paths=paths, partition="fit", names=fit_names, contract=contract,
        target_sha_for_name=target_sha,
    )
    enforce_resource_caps(paths, pre_stage=True)
    arms = train_arms(
        paths=paths, contract=contract, split=split, device=device, amp=bool(args.amp)
    )
    checkpoints = _verified_checkpoint_receipts(paths, contract_sha)
    if not checkpoints or int(checkpoints[-1]["step"]) != TRAIN_STEPS:
        raise ContractError("four-arm training did not commit the final step-2500 checkpoint")
    final_receipt = checkpoints[-1]
    final_checkpoint = dict(final_receipt["checkpoint"])

    write_status(paths, contract_sha, state="evaluating_cal", step=TRAIN_STEPS,
                 message="evaluating authenticated CAL boards; DEV sealed",
                 checkpoint=final_receipt)
    cal_dependencies = {
        "swap": swap_cal_record,
        "checkpoint": final_checkpoint,
        "cache_manifest": path_record(cache_files(paths, "cal").manifest),
        "target_receipt": path_record(paths.receipts / "cal_targets.json"),
        "oracle": oracle_record,
    }
    cal_record, cal_arrays = _load_or_evaluate_raw(
        paths=paths, partition="cal", contract=contract, split=split, arms=arms,
        swap_arrays=swap_cal, dependencies=cal_dependencies, device=device,
        amp=bool(args.amp), eval_batch=int(args.eval_batch),
    )
    for key in ORACLE_CAL_SHARED_KEYS:
        if not np.array_equal(np.asarray(cal_arrays[key]), np.asarray(oracle_arrays[key])):
            raise ContractError(f"learned CAL changed frozen pretrain oracle evidence: {key}")
    cal_metrics = metric_summary(cal_arrays, alpha=0.10)
    cal_gates = gate_map(cal_metrics, "CAL")
    if not cal_gates["passed"]:
        report = make_report(
            paths=paths, contract=contract,
            status="cal_reject", stage="cal", decision="KILL_DCT_WHERE",
            reason="cal_opening_gate_failed",
            encoder_checkpoint=encoder_checkpoint,
            encoder_rank_record=encoder_rank_record,
            representation_record=representation_record,
            encoder_metrics=encoder_metrics, encoder_gates=encoder_gates,
            oracle_record=oracle_record, fit_palette_record=fit_palette_record,
            swap_cal_record=swap_cal_record, cal_record=cal_record,
            final_checkpoint=final_checkpoint, swap_dev_record=None, dev_record=None,
            oracle_metrics=oracle_metrics, oracle_gates=oracle_gates,
            cal_metrics=cal_metrics, cal_gates=cal_gates,
            dev_metrics=None, dev_gates=None,
        )
        return _finalize_decision(
            paths=paths, contract=contract, report=report, exit_code=20,
            step=TRAIN_STEPS,
        )

    # CAL alone opens the DEV capability.  Nothing below this branch can run
    # on an oracle/CAL reject path.
    enforce_resource_caps(paths, pre_stage=True)
    dev_names = tuple(sorted(split.names["dev"], key=_board_id))
    build_embedding_cache(
        paths=paths, partition="dev", names=dev_names, contract=contract,
        device=device, amp=bool(args.amp), chunk_size=int(args.cache_chunk),
    )
    swap_dev_record, swap_dev = build_or_load_swap(
        paths=paths, partition="dev", contract=contract, split=split,
        fit_palette_record=fit_palette_record, fit_palette=fit_palette,
    )
    authenticate_partition_targets(
        paths=paths, partition="dev", names=dev_names, contract=contract,
        target_sha_for_name=target_sha,
    )
    write_status(paths, contract_sha, state="evaluating_dev", step=TRAIN_STEPS,
                 message="CAL passed; evaluating the one-shot authenticated DEV gate",
                 checkpoint=final_receipt)
    dev_dependencies = {
        "swap": swap_dev_record,
        "checkpoint": final_checkpoint,
        "cache_manifest": path_record(cache_files(paths, "dev").manifest),
        "target_receipt": path_record(paths.receipts / "dev_targets.json"),
        "fit_palette": fit_palette_record,
    }
    dev_record, dev_arrays = _load_or_evaluate_raw(
        paths=paths, partition="dev", contract=contract, split=split, arms=arms,
        swap_arrays=swap_dev, dependencies=dev_dependencies, device=device,
        amp=bool(args.amp), eval_batch=int(args.eval_batch),
    )
    dev_metrics = metric_summary(dev_arrays, alpha=0.05)
    dev_gates = gate_map(dev_metrics, "DEV")
    if dev_gates["passed"]:
        status, decision, exit_code = "dev_pass", "PROMOTE_DCT_WHERE", 0
        reason = "dev_gates_passed"
    else:
        status, decision, exit_code = "dev_reject", "KILL_DCT_WHERE", 20
        reason = "dev_confirmation_gate_failed"
    report = make_report(
        paths=paths, contract=contract, status=status, stage="dev", decision=decision,
        reason=reason,
        encoder_checkpoint=encoder_checkpoint,
        encoder_rank_record=encoder_rank_record,
        representation_record=representation_record,
        encoder_metrics=encoder_metrics, encoder_gates=encoder_gates,
        oracle_record=oracle_record, fit_palette_record=fit_palette_record,
        swap_cal_record=swap_cal_record, cal_record=cal_record,
        final_checkpoint=final_checkpoint, swap_dev_record=swap_dev_record,
        dev_record=dev_record, oracle_metrics=oracle_metrics, oracle_gates=oracle_gates,
        cal_metrics=cal_metrics, cal_gates=cal_gates,
        dev_metrics=dev_metrics, dev_gates=dev_gates,
    )
    return _finalize_decision(
        paths=paths, contract=contract, report=report, exit_code=exit_code,
        step=TRAIN_STEPS,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "cache", "train", "run", "status"))
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--cache-chunk", type=int, default=CACHE_CHUNK)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.eval_batch < 1:
        parser.error("--eval-batch must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "status":
        path = Path(args.work_root) / "status.json"
        if not path.is_file():
            print(json.dumps({"state": "not_started", "status": str(path.resolve())}))
            return 0
        print(json.dumps(read_json(path), ensure_ascii=False, indent=2))
        return 0
    paths: RunPaths | None = None
    contract: Mapping[str, Any] | None = None
    lock_acquired = False
    result = 1
    try:
        paths = RunPaths(_require_drive_e(Path(args.work_root), "M144 work root"))
        paths.ensure()
        acquire_run_lock(paths)
        lock_acquired = True
        configure_deterministic_runtime()
        paths, split, contract = prepare(args)
        if args.command in {"cache", "train", "run"}:
            terminal = authenticated_terminal_return(paths=paths, contract=contract)
            if terminal is not None:
                result = terminal
                return result
        if args.command == "preflight":
            device = require_cuda(args.device)
            enforce_resource_caps(paths, pre_stage=True)
            write_status(paths, str(contract["contract_sha256"]), state="preflight_complete",
                         step=0, message="M144 immutable inputs and protocol authenticated")
            print(json.dumps({"contract": str(paths.contract),
                              "contract_sha256": contract["contract_sha256"]}, indent=2))
            result = 0
            return result
        if args.command in {"cache", "run"}:
            cache_result = run_cache(args, paths, split, contract)
            if cache_result is not None:
                result = cache_result
                return result
            if args.command == "cache":
                result = 0
                return result
        result = command_train(args, paths, split, contract)
        return result
    except ScientificReject as exc:
        print(f"M144 scientific reject: {exc}", file=sys.stderr, flush=True)
        result = 20
        return result
    except Exception as exc:
        try:
            if paths is not None and lock_acquired:
                write_status(
                    paths,
                    str(contract["contract_sha256"]) if contract is not None else "unknown",
                    state="failed",
                    step=0,
                    message=f"{type(exc).__name__}: {exc}",
                )
        except Exception:
            pass
        print(f"M144 failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        result = 1
        return result
    finally:
        if paths is not None and lock_acquired:
            try:
                release_run_lock(paths)
            except Exception as exc:
                print(f"M144 lock release failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
