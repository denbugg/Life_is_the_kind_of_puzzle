#!/usr/bin/env python3
"""Train and development-gate the task-specific ViT-Sinkhorn pilot.

The default pilot is intentionally bounded but large enough to test signal:
256 whole synthetic sources for three epochs, an eight-source selection panel,
and a disjoint eight-source holdout panel.  Launching with
``torchrun --nproc_per_node=2`` uses both T4s through DDP.  Synthetic training
and both development panels run without any QAP prior: a truth-derived
"imperfect prior" would make the experiment circular.  A prior may be used
only on partial real pseudo-gold examples and only from an input-only NPZ whose
metadata explicitly records ``targets_opened=false``.

This script never reads competition test targets and never promotes from test
or audit metrics.  Even a passed holdout gate only approves a later genuine
QAP/real-layout comparison; every checkpoint remains unsafe for submission.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_assembly.geometry import GRID, TILE, TILE_COUNT
from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.vit_sinkhorn import (
    ViTSinkhorn,
    ViTSinkhornConfig,
    make_synthetic_smoke_batch,
    permutation_metrics_from_logits,
    vit_sinkhorn_losses,
)
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import split_tiles_numpy


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"
CHECKPOINT_NAME = "vit_sinkhorn_checkpoint.pt"
REPORT_NAME = "vit_sinkhorn_report.json"
HASHES_NAME = "SHA256SUMS.txt"
MODEL_SOURCE = REPO_ROOT / "src/puzzle_assembly/vit_sinkhorn.py"
SCRIPT_SOURCE = Path(__file__).resolve()
IMPORTED_CODE_SOURCES = (
    REPO_ROOT / "src/puzzle_assembly/geometry.py",
    REPO_ROOT / "src/puzzle_assembly/metrics.py",
    REPO_ROOT / "src/puzzle_assembly/panels.py",
    REPO_ROOT / "src/puzzle_assembly/protocol.py",
    REPO_ROOT / "src/puzzle_assembly/compatibility.py",
    REPO_ROOT / "src/puzzle_assembly/components.py",
    REPO_ROOT / "src/puzzle_assembly/solvers.py",
    REPO_ROOT / "src/puzzle_denoise_v2/degradation.py",
    REPO_ROOT / "src/puzzle_denoise_v2/inference.py",
    REPO_ROOT / "src/puzzle_denoise_v2/training.py",
    REPO_ROOT / "src/puzzle_denoise_v2/losses.py",
    REPO_ROOT / "src/puzzle_denoise_v2/metrics.py",
    REPO_ROOT / "src/puzzle_denoise_v2/model.py",
    REPO_ROOT / "src/puzzle_denoise_v2/tiles.py",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--disable-denoiser", action="store_true")
    parser.add_argument(
        "--real-gold",
        default="",
        help="Legacy flag name for a validated partial real pseudo-gold NPZ",
    )
    parser.add_argument("--real-gold-source-limit", type=int, default=64)
    parser.add_argument("--real-gold-probability", type=float, default=0.25)
    parser.add_argument("--qap-priors", default="")
    parser.add_argument(
        "--qap-prior-probability",
        type=float,
        default=0.0,
        help="Enable only with an input-only --qap-priors asset; never affects synthetic examples",
    )
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-sources", type=int, default=256)
    parser.add_argument("--dev-offset", type=int, default=0)
    parser.add_argument("--dev-sources", type=int, default=8)
    parser.add_argument("--holdout-sources", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "fp16", "bf16", "none"), default="auto")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--edge-channels", type=int, default=32)
    parser.add_argument("--edge-dim", type=int, default=64)
    parser.add_argument("--edge-band", type=int, default=4)
    parser.add_argument("--edge-bins", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--qap-prior-dropout", type=float, default=0.35)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--sinkhorn-temperature", type=float, default=0.10)
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=1024.0,
        help=(
            "Initial fp16 GradScaler scale. The default is intentionally below "
            "PyTorch's 65536 because the 576x576 Sinkhorn objective can overflow "
            "the first scaled backward on T4 before dynamic scaling adapts."
        ),
    )
    parser.add_argument(
        "--max-consecutive-amp-skips",
        type=int,
        default=8,
        help="Fail closed after this many consecutive non-finite fp16 updates.",
    )
    parser.add_argument("--assignment-weight", type=float, default=1.0)
    parser.add_argument("--directional-contrast-weight", type=float, default=0.20)
    parser.add_argument("--neighbor-consistency-weight", type=float, default=0.05)
    parser.add_argument("--contrast-temperature", type=float, default=0.07)
    parser.add_argument("--consistency-topk", type=int, default=16)
    parser.add_argument("--gate-min-position-accuracy", type=float, default=0.01)
    parser.add_argument("--gate-min-combined-adjacency", type=float, default=0.02)
    parser.add_argument(
        "--gate-min-classical-manhattan-reduction", type=float, default=0.02
    )
    parser.add_argument(
        "--gate-min-ssim-delta-vs-classical", type=float, default=0.003
    )
    parser.add_argument("--denoise-batch-size", type=int, default=576)
    return parser


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    distributed: bool

    @property
    def primary(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class PseudoGold:
    target_tile_to_position: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class PreparedExample:
    source_name: str
    supervision: str
    raw_tiles: np.ndarray
    restored_tiles: np.ndarray
    target_tile_to_position: np.ndarray
    confidence: np.ndarray
    qap_tile_to_position: np.ndarray | None
    qap_confidence: np.ndarray | None
    clean_target: np.ndarray | None
    panel: str | None
    panel_seed: int | None
    curriculum_severity: float | None


def _init_runtime(device_request: str) -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process pilot requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        return Runtime(torch.device("cuda", local_rank), rank, local_rank, world_size, True)
    if device_request == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_request)
    return Runtime(device, rank, local_rank, world_size, False)


def _cleanup_runtime(runtime: Runtime | None) -> None:
    if runtime is not None and runtime.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _barrier(runtime: Runtime) -> None:
    if runtime.distributed:
        torch.distributed.barrier()


def _print(runtime: Runtime, payload: Mapping[str, Any]) -> None:
    if runtime.primary:
        print(json.dumps(dict(payload), sort_keys=True), flush=True)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _names_sha256(names: list[str] | tuple[str, ...] | set[str]) -> str:
    payload = "\n".join(sorted(names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _named_files_sha256(paths: list[Path], *, root: Path) -> str:
    """Hash an exact named file set, including names and contents."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            name = str(path.relative_to(root))
        except ValueError:
            name = str(path.resolve())
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _code_provenance() -> dict[str, Any]:
    paths = (MODEL_SOURCE, SCRIPT_SOURCE, *IMPORTED_CODE_SOURCES)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"imported source files are missing: {missing}")
    records = {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in paths
    }
    return {
        "files": records,
        "combined_sha256": hashlib.sha256(
            "\n".join(f"{name}\0{records[name]}" for name in sorted(records)).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def _to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _capture_training_state(
    model: ViTSinkhorn,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    *,
    selected_epoch: int | None,
    rng_state: Any | None = None,
) -> dict[str, Any]:
    """Capture a resume-complete, CPU-portable training snapshot."""

    return {
        "model_state": _cpu_state(model),
        "optimizer_state": _to_cpu_tree(optimizer.state_dict()),
        "scaler_state": {} if scaler is None else _to_cpu_tree(scaler.state_dict()),
        "rng_state": _capture_rng_state() if rng_state is None else _to_cpu_tree(rng_state),
        "selected_epoch": selected_epoch,
    }


def _gather_rank_rng_states(runtime: Runtime) -> list[dict[str, Any]] | None:
    local = _capture_rng_state()
    if not runtime.distributed:
        return [local]
    gathered: list[dict[str, Any] | None] | None = (
        [None] * runtime.world_size if runtime.primary else None
    )
    torch.distributed.gather_object(local, gathered, dst=0)
    if not runtime.primary:
        return None
    assert gathered is not None and all(state is not None for state in gathered)
    return [state for state in gathered if state is not None]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _write_hashes(output_dir: Path, paths: list[Path]) -> Path:
    records = [f"{_sha256(path)}  {path.name}" for path in sorted(paths)]
    target = output_dir / HASHES_NAME
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    temporary.write_text("\n".join(records) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def _preflight_output(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        output_dir / CHECKPOINT_NAME,
        output_dir / REPORT_NAME,
        output_dir / HASHES_NAME,
    ]
    existing = [str(path) for path in artifacts if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output artifacts exist; pass --overwrite: {existing}")
    if args.overwrite:
        for path in artifacts:
            path.unlink(missing_ok=True)
    return output_dir


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (GRID * TILE, GRID * TILE, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _device_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    if values.shape != (TILE_COUNT, TILE, TILE, 3) or values.dtype != np.uint8:
        raise ValueError("tiles must be uint8 576x20x20x3")
    return torch.from_numpy(
        np.ascontiguousarray(values.transpose(0, 3, 1, 2))
    ).to(device=device, dtype=torch.float32).div_(255.0).unsqueeze(0)


def _fallback_restore(raw_tiles: np.ndarray) -> np.ndarray:
    values = torch.from_numpy(
        np.ascontiguousarray(raw_tiles.transpose(0, 3, 1, 2))
    ).float().div_(255.0)
    smoothed = F.avg_pool2d(values, kernel_size=3, stride=1, padding=1)
    restored = (0.75 * values + 0.25 * smoothed).clamp(0.0, 1.0)
    return (
        restored.mul(255.0)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .numpy()
    )


def _restore(
    raw_tiles: np.ndarray,
    *,
    restorer: nn.Module | None,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if restorer is None:
        return _fallback_restore(raw_tiles)
    return restore_tiles_uint8(restorer, raw_tiles, device, batch_size=batch_size)


def _load_pseudo_gold(
    path: Path,
    *,
    allowed_train_names: set[str],
    forbidden_dev_names: set[str],
    manifest_path: Path,
    data_root: Path,
) -> tuple[dict[str, PseudoGold], dict[str, Any]]:
    required = {
        "source_names",
        "source_index",
        "input_slot",
        "clean_tile_index",
        "joint_confidence",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"real-gold archive is missing {sorted(missing)}")
        source_names = [str(value) for value in archive["source_names"]]
        if len(source_names) != len(set(source_names)):
            raise ValueError("real-gold source_names contain duplicates")
        source_index = archive["source_index"].astype(np.int64, copy=False)
        input_slot = archive["input_slot"].astype(np.int64, copy=False)
        target_position = archive["clean_tile_index"].astype(np.int64, copy=False)
        raw_confidence = archive["joint_confidence"].astype(np.float32, copy=False)
        if not (
            source_index.shape
            == input_slot.shape
            == target_position.shape
            == raw_confidence.shape
        ):
            raise ValueError("real-gold pair arrays have inconsistent shapes")
        if not np.isfinite(raw_confidence).all() or np.any(raw_confidence < 0):
            raise ValueError("pseudo-gold joint_confidence must be finite and non-negative")
        if len(source_index) and (
            source_index.min() < 0 or source_index.max() >= len(source_names)
        ):
            raise ValueError("real-gold source_index is out of range")
        if len(input_slot) and (
            input_slot.min() < 0
            or input_slot.max() >= TILE_COUNT
            or target_position.min() < 0
            or target_position.max() >= TILE_COUNT
        ):
            raise ValueError("real-gold tile indices are out of range")
        if "meta" not in archive.files:
            raise ValueError("pseudo-gold archive must contain fail-closed meta")
        metadata: dict[str, Any] = json.loads(str(archive["meta"].item()))

    required_metadata = {
        "schema_version": 1,
        "kind": "high_purity_real_tile_pairs",
        "split": "train",
        "old_q90_used_as_ground_truth": False,
        "source_name_encoding": "source_names[source_index]",
    }
    mismatched = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in required_metadata.items()
        if type(metadata.get(key)) is not type(expected)
        or metadata.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"pseudo-gold metadata mismatch: {mismatched}")
    for key in (
        "manifest_sha256",
        "test_overlap_excluded",
        "source_count",
        "selected_pairs",
        "joint_confidence_definition",
        "selection_rule",
    ):
        if key not in metadata:
            raise ValueError(f"pseudo-gold metadata is missing {key}")
    for key in ("test_overlap_excluded", "source_count", "selected_pairs"):
        if type(metadata[key]) is not int or metadata[key] < 0:
            raise ValueError(f"pseudo-gold metadata {key} must be a non-negative integer")
    for key in ("joint_confidence_definition", "selection_rule"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"pseudo-gold metadata {key} must be a non-empty string")
    if metadata["manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("pseudo-gold manifest hash does not match active manifest")
    if int(metadata["source_count"]) != len(source_names):
        raise ValueError("pseudo-gold metadata source_count is inconsistent")
    if int(metadata["selected_pairs"]) != len(source_index):
        raise ValueError("pseudo-gold metadata selected_pairs is inconsistent")
    test_names = {
        candidate.name for candidate in (data_root / "test").glob("*.png")
    }
    if int(metadata["test_overlap_excluded"]) != len(test_names):
        raise ValueError(
            "pseudo-gold test_overlap_excluded does not match active test set"
        )
    overlap = sorted(set(source_names) & test_names)
    if overlap:
        raise ValueError(f"pseudo-gold source names overlap test: {overlap[:5]}")

    forbidden_in_archive = sorted(set(source_names) & forbidden_dev_names)
    # An archive may cover the whole original train split.  We never materialize
    # labels for edge_development sources, and record their exclusion explicitly.
    usable_indices = [
        index
        for index, name in enumerate(source_names)
        if name in allowed_train_names and name not in forbidden_dev_names
    ]
    gold: dict[str, PseudoGold] = {}
    for index in usable_indices:
        name = source_names[index]
        rows = np.flatnonzero(source_index == index)
        targets = np.full(TILE_COUNT, -1, dtype=np.int64)
        confidence = np.zeros(TILE_COUNT, dtype=np.float32)
        slots = input_slot[rows]
        positions = target_position[rows]
        if len(np.unique(slots)) != len(slots) or len(np.unique(positions)) != len(positions):
            raise ValueError(f"real-gold mapping is not one-to-one for {name}")
        targets[slots] = positions
        # The archive defines a normalized confidence relative to per-image
        # descriptor margins.  Use it directly as a calibrated pseudo-label
        # weight, clipped only to the loss contract [0,1].
        confidence[slots] = np.clip(raw_confidence[rows], 0.0, 1.0)
        gold[name] = PseudoGold(targets, confidence)
    if not gold:
        raise ValueError("real-gold archive has no sources in edge_train")
    return gold, {
        "path": str(path),
        "sha256": _sha256(path),
        "archive_source_count": len(source_names),
        "usable_edge_train_source_count": len(gold),
        "excluded_edge_development_source_count": len(forbidden_in_archive),
        "excluded_edge_development_names_sha256": _names_sha256(
            forbidden_in_archive
        ),
        "metadata": metadata,
        "label_kind": "partial_real_pseudo_gold_not_ground_truth",
        "confidence_transform": "clip(joint_confidence,0,1)",
        "active_manifest_sha256": _sha256(manifest_path),
        "active_test_name_count": len(test_names),
        "verified_test_name_overlap_count": 0,
    }


def _load_qap_priors(path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if "source_names" not in archive.files:
            raise ValueError("QAP prior archive must contain source_names")
        names = [str(value) for value in archive["source_names"]]
        if len(names) != len(set(names)):
            raise ValueError("QAP prior source_names contain duplicates")
        if "tile_to_position" in archive.files:
            positions = archive["tile_to_position"].astype(np.int64, copy=False)
        elif "position_to_tile" in archive.files:
            layouts = archive["position_to_tile"].astype(np.int64, copy=False)
            if layouts.shape != (len(names), TILE_COUNT):
                raise ValueError("position_to_tile must have shape Mx576")
            positions = np.empty_like(layouts)
            for index, layout in enumerate(layouts):
                if not np.array_equal(np.sort(layout), np.arange(TILE_COUNT)):
                    raise ValueError(f"invalid QAP permutation for {names[index]}")
                positions[index, layout] = np.arange(TILE_COUNT)
        else:
            raise ValueError("QAP prior archive needs tile_to_position or position_to_tile")
        if positions.shape != (len(names), TILE_COUNT):
            raise ValueError("QAP positions must have shape Mx576")
        if "confidence" in archive.files:
            confidence = archive["confidence"].astype(np.float32, copy=False)
            if confidence.shape != positions.shape:
                raise ValueError("QAP confidence must match positions")
            if not np.isfinite(confidence).all():
                raise ValueError("QAP confidence contains non-finite values")
            confidence = np.clip(confidence, 0.0, 1.0)
        else:
            confidence = np.ones_like(positions, dtype=np.float32)
        if "meta" not in archive.files:
            raise ValueError(
                "QAP prior archive must contain meta with targets_opened=false"
            )
        metadata: dict[str, Any] = json.loads(str(archive["meta"].item()))
        if metadata.get("targets_opened") is not False:
            raise ValueError(
                "QAP archive meta must explicitly record targets_opened=false"
            )
    table: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, name in enumerate(names):
        row = positions[index]
        if not np.array_equal(np.sort(row), np.arange(TILE_COUNT)):
            raise ValueError(f"QAP prior is not a permutation for {name}")
        table[name] = (row.copy(), confidence[index].copy())
    return table, {
        "path": str(path),
        "sha256": _sha256(path),
        "source_count": len(names),
        "metadata": metadata,
        "schema": "source_names + tile_to_position|position_to_tile + optional confidence",
    }


def _curriculum_severity(epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return 1.0
    fraction = epoch / float(epochs - 1)
    return float(0.30 + 0.70 * fraction)


def _prepare_synthetic(
    name: str,
    *,
    args: argparse.Namespace,
    epoch: int,
    stage: str,
    restorer: nn.Module | None,
    device: torch.device,
) -> PreparedExample:
    target_path = Path(args.data_root) / "train" / "targets" / name
    clean_target = _read_rgb(target_path)
    # Training gets a fresh per-tile draw of brightness, contrast, noise, blur,
    # and JPEG parameters for every source and epoch.  The slow, independent
    # Pillow/libjpeg renderer is reserved for the one-shot holdout so a 2xT4
    # pilot stays bounded while still testing backend transfer.
    panel_name = "independent_libjpeg" if stage == "holdout" else "primary_kornia"
    seed_epoch = 0 if stage in {"selection", "holdout"} else epoch
    panel_seed = per_source_seed(
        args.seed, f"vit-sinkhorn-{stage}-{panel_name}", name, seed_epoch
    )
    panel = make_exact_panel(clean_target, panel=panel_name, seed=panel_seed)
    severity = (
        1.0
        if stage in {"selection", "holdout"}
        else _curriculum_severity(epoch, args.epochs)
    )
    clean_slots = panel.clean_target_tiles[panel.slot_to_target]
    raw = np.clip(
        np.rint(
            (1.0 - severity) * clean_slots.astype(np.float32)
            + severity * panel.slot_tiles.astype(np.float32)
        ),
        0,
        255,
    ).astype(np.uint8)
    restored = _restore(
        raw,
        restorer=restorer,
        device=device,
        batch_size=args.denoise_batch_size,
    )
    return PreparedExample(
        source_name=name,
        supervision="synthetic_known_permutation",
        raw_tiles=np.ascontiguousarray(raw),
        restored_tiles=np.ascontiguousarray(restored),
        target_tile_to_position=panel.slot_to_target.astype(np.int64, copy=False),
        confidence=np.ones(TILE_COUNT, dtype=np.float32),
        qap_tile_to_position=None,
        qap_confidence=None,
        clean_target=clean_target,
        panel=panel_name,
        panel_seed=panel_seed,
        curriculum_severity=severity,
    )


def _prepare_partial_real(
    name: str,
    gold: PseudoGold,
    *,
    args: argparse.Namespace,
    restorer: nn.Module | None,
    device: torch.device,
    qap_priors: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> PreparedExample:
    raw_image = _read_rgb(Path(args.data_root) / "train" / "inputs" / name)
    raw = np.ascontiguousarray(split_tiles_numpy(raw_image))
    restored = _restore(
        raw,
        restorer=restorer,
        device=device,
        batch_size=args.denoise_batch_size,
    )
    qap = qap_priors.get(name)
    return PreparedExample(
        source_name=name,
        supervision="partial_real_pseudo_gold",
        raw_tiles=raw,
        restored_tiles=np.ascontiguousarray(restored),
        target_tile_to_position=gold.target_tile_to_position,
        confidence=gold.confidence,
        qap_tile_to_position=None if qap is None else qap[0],
        qap_confidence=None if qap is None else qap[1],
        clean_target=None,
        panel=None,
        panel_seed=None,
        curriculum_severity=None,
    )


def _example_tensors(
    example: PreparedExample,
    *,
    device: torch.device,
    include_prior: bool,
) -> dict[str, torch.Tensor | None]:
    return {
        "raw": _device_tensor(example.raw_tiles, device),
        "restored": _device_tensor(example.restored_tiles, device),
        "targets": torch.from_numpy(example.target_tile_to_position).to(
            device=device, dtype=torch.long
        ).unsqueeze(0),
        "confidence": torch.from_numpy(example.confidence).to(
            device=device, dtype=torch.float32
        ).unsqueeze(0),
        "qap": (
            None
            if not include_prior or example.qap_tile_to_position is None
            else torch.from_numpy(example.qap_tile_to_position).to(
                device=device, dtype=torch.long
            ).unsqueeze(0)
        ),
        "qap_confidence": (
            None
            if not include_prior or example.qap_confidence is None
            else torch.from_numpy(example.qap_confidence).to(
                device=device, dtype=torch.float32
            ).unsqueeze(0)
        ),
    }


def _amp_settings(args: argparse.Namespace, runtime: Runtime) -> tuple[bool, torch.dtype, str]:
    if args.amp == "none" or runtime.device.type != "cuda":
        return False, torch.float32, "none"
    choice = "fp16" if args.amp == "auto" else args.amp
    dtype = torch.float16 if choice == "fp16" else torch.bfloat16
    return True, dtype, choice


def _autocast(runtime: Runtime, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=runtime.device.type, dtype=dtype, enabled=True)


def _model_config(args: argparse.Namespace) -> ViTSinkhornConfig:
    return ViTSinkhornConfig(
        grid_size=GRID,
        tile_size=TILE,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        feedforward_dim=args.feedforward_dim,
        cnn_channels=args.cnn_channels,
        edge_channels=args.edge_channels,
        edge_dim=args.edge_dim,
        edge_band=args.edge_band,
        edge_bins=args.edge_bins,
        dropout=args.dropout,
        qap_prior_dropout=args.qap_prior_dropout,
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_temperature=args.sinkhorn_temperature,
        activation_checkpointing=args.activation_checkpointing,
    )


def _complete_source_split(
    args: argparse.Namespace,
    *,
    gold: Mapping[str, PseudoGold],
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    all_train = source_names_for_split(
        "edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine
    )
    all_dev = source_names_for_split(
        "edge_development", manifest_path=args.manifest, quarantine_path=args.quarantine
    )
    if set(all_train) & set(all_dev):
        raise RuntimeError("edge_train and edge_development overlap")
    if len(all_train) != len(set(all_train)) or len(all_dev) != len(set(all_dev)):
        raise RuntimeError("source partitions contain duplicates")
    if min(args.train_sources, args.dev_sources, args.holdout_sources) <= 0:
        raise ValueError(
            "train-sources, dev-sources, and holdout-sources must be positive"
        )
    if min(args.train_offset, args.dev_offset) < 0:
        raise ValueError("source offsets must be non-negative")

    base_train = all_train[
        args.train_offset : args.train_offset + args.train_sources
    ]
    if len(base_train) != args.train_sources:
        raise ValueError("requested train source slice exceeds edge_train")
    # Ensure the bounded pilot actually exercises partial supervision when an
    # archive is supplied, without changing the requested total source count.
    gold_candidates = [name for name in all_train if name in gold]
    gold_limit = min(
        args.real_gold_source_limit,
        args.train_sources,
        len(gold_candidates),
    )
    prioritized = gold_candidates[:gold_limit]
    train_names = prioritized + [name for name in base_train if name not in prioritized]
    if len(train_names) < args.train_sources:
        train_names.extend(
            name
            for name in all_train
            if name not in set(train_names)
        )
    train_names = train_names[: args.train_sources]
    dev_names = all_dev[args.dev_offset : args.dev_offset + args.dev_sources]
    if len(dev_names) != args.dev_sources:
        raise ValueError("requested development source slice exceeds edge_development")
    holdout_start = args.dev_offset + args.dev_sources
    holdout_names = all_dev[
        holdout_start : holdout_start + args.holdout_sources
    ]
    if len(holdout_names) != args.holdout_sources:
        raise ValueError("requested holdout slice exceeds edge_development")
    if set(train_names) & (set(dev_names) | set(holdout_names)):
        raise RuntimeError("selected train and development sources overlap")
    if set(dev_names) & set(holdout_names):
        raise RuntimeError("selection and holdout development sources overlap")

    test_names = {
        path.name for path in (Path(args.data_root) / "test").glob("*.png")
    }
    leaked = (set(train_names) | set(dev_names) | set(holdout_names)) & test_names
    if leaked:
        raise RuntimeError(
            f"selected whole-source split overlaps test filenames: {sorted(leaked)[:5]}"
        )
    missing: list[str] = []
    for name in train_names + dev_names + holdout_names:
        if not (Path(args.data_root) / "train" / "targets" / name).is_file():
            missing.append(f"target:{name}")
    for name in train_names:
        if name in gold and not (Path(args.data_root) / "train" / "inputs" / name).is_file():
            missing.append(f"input:{name}")
    if missing:
        raise FileNotFoundError(f"selected source files are missing: {missing[:8]}")
    return train_names, dev_names, holdout_names, {
        "policy": "whole source only; no tile-level source mixing",
        "train_partition": "edge_train",
        "development_partition": "edge_development",
        "evaluation_scope": "fixed selection development then one-shot independent holdout",
        "audit_opened": False,
        "test_targets_opened": False,
        "selected_train_count": len(train_names),
        "selected_development_count": len(dev_names),
        "selected_holdout_count": len(holdout_names),
        "selected_train_names": train_names,
        "selected_development_names": dev_names,
        "selected_holdout_names": holdout_names,
        "selected_train_names_sha256": _names_sha256(train_names),
        "selected_development_names_sha256": _names_sha256(dev_names),
        "selected_holdout_names_sha256": _names_sha256(holdout_names),
        "selected_overlap_count": len(
            set(train_names) & (set(dev_names) | set(holdout_names))
        ),
        "selection_holdout_overlap_count": len(set(dev_names) & set(holdout_names)),
        "selected_test_filename_overlap_count": len(leaked),
        "pseudo_gold_sources_selected": sum(name in gold for name in train_names),
    }


def _epoch_indices(count: int, runtime: Runtime, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(count).tolist()
    padded_count = int(math.ceil(count / runtime.world_size) * runtime.world_size)
    if padded_count > count:
        order.extend(order[: padded_count - count])
    local = order[runtime.rank : padded_count : runtime.world_size]
    if len(local) * runtime.world_size != padded_count:
        raise RuntimeError("distributed source sharding produced unequal step counts")
    return [int(value) for value in local]


def _aggregate_dev(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    numeric_keys: set[str] = set()
    for record in records:
        numeric_keys.update(
            name
            for name, value in record[key].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
    return {
        name: float(np.mean([float(record[key][name]) for record in records]))
        for name in sorted(numeric_keys)
        if all(name in record[key] for record in records)
    }


@torch.inference_mode()
def _evaluate_development(
    model: ViTSinkhorn,
    names: list[str],
    *,
    args: argparse.Namespace,
    runtime: Runtime,
    restorer: nn.Module | None,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    stage: str,
) -> dict[str, Any]:
    if stage not in {"selection", "holdout"}:
        raise ValueError("development stage must be selection or holdout")
    model.eval()
    records: list[dict[str, Any]] = []
    for name in names:
        example = _prepare_synthetic(
            name,
            args=args,
            epoch=0,
            stage=stage,
            restorer=restorer,
            device=runtime.device,
        )
        if example.qap_tile_to_position is not None or example.qap_confidence is not None:
            raise RuntimeError("synthetic development must not contain a QAP prior")
        tensors = _example_tensors(example, device=runtime.device, include_prior=False)
        with _autocast(runtime, amp_enabled, amp_dtype):
            prediction = model(tensors["raw"], tensors["restored"])
        target = example.target_tile_to_position
        model_metrics = permutation_metrics_from_logits(
            prediction.logits[0], target, grid_size=GRID
        )
        model_layout = np.asarray(
            model_metrics.pop("position_to_tile"), dtype=np.int32
        )
        values = example.restored_tiles.astype(np.float32) / 255.0
        strip = 4
        right_query = values[:, :, -strip:, :].reshape(TILE_COUNT, -1)
        right_key = values[:, :, :strip, :].reshape(TILE_COUNT, -1)
        down_query = values[:, -strip:, :, :].reshape(TILE_COUNT, -1)
        down_key = values[:, :strip, :, :].reshape(TILE_COUNT, -1)
        right = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.float32)
        down = np.empty_like(right)
        for start in range(0, TILE_COUNT, 64):
            stop = min(start + 64, TILE_COUNT)
            right[start:stop] = np.mean(
                np.abs(right_query[start:stop, None] - right_key[None]),
                axis=2,
                dtype=np.float32,
            )
            down[start:stop] = np.mean(
                np.abs(down_query[start:stop, None] - down_key[None]),
                axis=2,
                dtype=np.float32,
            )
        np.fill_diagonal(right, np.inf)
        np.fill_diagonal(down, np.inf)
        compatibility = CompatibilityMatrices("input_only_rgb_l1_w4", right, down)
        classical_layout = soft_cycle_component_solver(
            compatibility,
            top_k=8,
            keep_per_tile=1,
            proposal_keep_fraction=0.5,
            loop_weight=1.0,
            reciprocal_weight=0.35,
        ).position_to_slot.astype(np.int32, copy=False)
        classical_metrics = layout_metrics(classical_layout, target)
        model_metrics.update(
            predicted_image_metrics(
                model_layout, example.restored_tiles, example.clean_target
            )
        )
        classical_metrics.update(
            predicted_image_metrics(
                classical_layout, example.restored_tiles, example.clean_target
            )
        )
        records.append(
            {
                "source_name": name,
                "panel": example.panel,
                "panel_seed": example.panel_seed,
                "target_permutation_sha256": _array_sha256(target),
                "model_without_prior": model_metrics,
                "input_only_classical_baseline": classical_metrics,
                "model_layout_sha256": _array_sha256(model_layout),
                "classical_layout_sha256": _array_sha256(classical_layout),
            }
        )
    aggregate = {
        "model_without_prior": _aggregate_dev(records, "model_without_prior"),
        "input_only_classical_baseline": _aggregate_dev(
            records, "input_only_classical_baseline"
        ),
    }
    primary = aggregate["model_without_prior"]
    baseline = aggregate["input_only_classical_baseline"]
    manhattan_reduction = (
        (baseline["mean_manhattan"] - primary["mean_manhattan"])
        / baseline["mean_manhattan"]
        if baseline["mean_manhattan"] > 0
        else 0.0
    )
    ssim_delta = (
        primary["predicted_layout_ssim"] - baseline["predicted_layout_ssim"]
    )
    gates = {
        "position_accuracy": {
            "value": primary["position_accuracy"],
            "minimum": args.gate_min_position_accuracy,
            "passed": primary["position_accuracy"] >= args.gate_min_position_accuracy,
        },
        "combined_adjacency": {
            "value": primary["combined_adjacency"],
            "minimum": args.gate_min_combined_adjacency,
            "passed": primary["combined_adjacency"]
            >= args.gate_min_combined_adjacency,
        },
        "classical_manhattan_reduction": {
            "value": manhattan_reduction,
            "minimum": args.gate_min_classical_manhattan_reduction,
            "passed": manhattan_reduction
            >= args.gate_min_classical_manhattan_reduction,
        },
        "ssim_delta_vs_classical": {
            "value": ssim_delta,
            "minimum": args.gate_min_ssim_delta_vs_classical,
            "passed": ssim_delta >= args.gate_min_ssim_delta_vs_classical,
        },
    }
    return {
        "epoch": epoch,
        "stage": stage,
        "scope": "edge_development synthetic known-permutation panels only",
        "source_count": len(names),
        "panel": (
            "primary_kornia" if stage == "selection" else "independent_libjpeg"
        ),
        "prior_policy": {
            "qap_prior_used": False,
            "truth_derived_prior_used": False,
            "qap_delta_claimed": False,
            "later_required_comparison": "genuine input-only QAP asset on fixed real layouts",
        },
        "baseline": "input-only restored RGB L1 w4 softcycle k8; fixed before metrics",
        "aggregate": aggregate,
        "derived": {
            "classical_manhattan_reduction": manhattan_reduction,
            "ssim_delta_vs_classical": ssim_delta,
        },
        "gates": gates,
        "gate_passed": all(record["passed"] for record in gates.values()),
        "per_source": records,
    }


def _cpu_state(model: ViTSinkhorn) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _write_success_artifacts(
    output_dir: Path,
    *,
    checkpoint_payload: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, str]:
    checkpoint_path = output_dir / CHECKPOINT_NAME
    report_path = output_dir / REPORT_NAME
    state = checkpoint_payload["model_state"]
    checkpoint_payload["model_state_sha256"] = _state_dict_sha256(state)
    _atomic_torch_save(checkpoint_path, checkpoint_payload)
    report["checkpoint"] = {
        "path": checkpoint_path.name,
        "sha256": _sha256(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "model_state_sha256": checkpoint_payload["model_state_sha256"],
    }
    _atomic_json(report_path, report)
    hashes_path = _write_hashes(output_dir, [checkpoint_path, report_path])
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "hashes": str(hashes_path),
        "hashes_sha256": _sha256(hashes_path),
    }


def _run_smoke(
    args: argparse.Namespace,
    runtime: Runtime,
    output_dir: Path,
) -> None:
    if runtime.world_size != 1:
        raise RuntimeError("synthetic smoke is intentionally single-process")
    config = ViTSinkhornConfig(
        grid_size=4,
        tile_size=20,
        d_model=64,
        num_layers=2,
        num_heads=4,
        feedforward_dim=128,
        cnn_channels=16,
        edge_channels=8,
        edge_dim=16,
        edge_band=3,
        edge_bins=5,
        dropout=0.0,
        qap_prior_dropout=0.25,
        sinkhorn_iterations=12,
        sinkhorn_temperature=0.20,
        activation_checkpointing=False,
    )
    model = ViTSinkhorn(config).to(runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch = {
        name: value.to(runtime.device)
        for name, value in make_synthetic_smoke_batch(
            grid_size=4, tile_size=20, batch_size=1, seed=args.seed
        ).items()
    }
    history: list[dict[str, float]] = []
    for step in range(args.smoke_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["raw_tiles"], batch["restored_tiles"])
        losses = vit_sinkhorn_losses(
            output,
            batch["target_tile_to_position"],
            grid_size=4,
            consistency_topk=4,
        )
        if not torch.isfinite(losses["total"]):
            raise RuntimeError("synthetic smoke loss is non-finite")
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("synthetic smoke gradient norm is non-finite")
        optimizer.step()
        history.append(
            {
                name: float(value.detach().cpu())
                for name, value in losses.items()
            }
        )
    model.eval()
    with torch.inference_mode():
        output = model(batch["raw_tiles"], batch["restored_tiles"])
    probabilities = output.log_assignment.exp()
    row_error = float((probabilities.sum(2) - 1.0).abs().max().cpu())
    column_error = float((probabilities.sum(1) - 1.0).abs().max().cpu())
    metrics = permutation_metrics_from_logits(
        output.logits[0], batch["target_tile_to_position"][0], grid_size=4
    )
    layout = np.asarray(metrics.pop("position_to_tile"), dtype=np.int32)
    if not np.array_equal(np.sort(layout), np.arange(16)):
        raise RuntimeError("synthetic smoke Hungarian output is not a permutation")
    report = {
        "schema_version": 1,
        "kind": "vit_sinkhorn_synthetic_smoke",
        "status": "synthetic_smoke_passed",
        "safe_for_submission": False,
        "development_gate_eligible": False,
        "reason": "small-grid infrastructure smoke only",
        "seed": args.seed,
        "steps": args.smoke_steps,
        "model_config": config.to_dict(),
        "loss_history": history,
        "sinkhorn_max_row_error": row_error,
        "sinkhorn_max_column_error": column_error,
        "hungarian_metrics": metrics,
        "hungarian_layout_sha256": _array_sha256(layout),
        "prior_policy": "no QAP prior; infrastructure-only synthetic batch",
        "source_code": _code_provenance(),
        "runtime": {
            "device": str(runtime.device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    resume_state = _capture_training_state(
        model, optimizer, None, selected_epoch=None
    )
    checkpoint = {
        "schema_version": 1,
        "kind": "vit_sinkhorn_synthetic_smoke_checkpoint",
        "safe_for_submission": False,
        "model_config": config.to_dict(),
        **resume_state,
        "seed": args.seed,
        "source_code": report["source_code"],
    }
    artifacts = _write_success_artifacts(
        output_dir,
        checkpoint_payload=checkpoint,
        report=report,
    )
    _print(runtime, {"event": "synthetic_smoke_complete", **artifacts})


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.epochs, args.denoise_batch_size, args.smoke_steps) <= 0:
        raise ValueError("epochs, smoke-steps, and denoise-batch-size must be positive")
    probability_fields = {
        "real_gold_probability": args.real_gold_probability,
        "qap_prior_probability": args.qap_prior_probability,
    }
    if any(not 0.0 <= value <= 1.0 for value in probability_fields.values()):
        raise ValueError(f"probability arguments must be in [0,1]: {probability_fields}")
    if args.qap_prior_probability > 0 and not args.qap_priors:
        raise ValueError(
            "positive qap-prior-probability requires an explicit --qap-priors asset"
        )
    if min(args.learning_rate, args.grad_clip, args.contrast_temperature) <= 0:
        raise ValueError("learning-rate, grad-clip, and contrast-temperature must be positive")
    if args.real_gold_source_limit < 0 or args.consistency_topk <= 0:
        raise ValueError("real-gold-source-limit must be non-negative and topk positive")
    gate_fields = {
        "gate_min_position_accuracy": args.gate_min_position_accuracy,
        "gate_min_combined_adjacency": args.gate_min_combined_adjacency,
        "gate_min_classical_manhattan_reduction": (
            args.gate_min_classical_manhattan_reduction
        ),
        "gate_min_ssim_delta_vs_classical": args.gate_min_ssim_delta_vs_classical,
    }
    if any(not math.isfinite(value) or value <= 0 for value in gate_fields.values()):
        raise ValueError(f"all development gates must be finite and positive: {gate_fields}")


def _run_pilot(
    args: argparse.Namespace,
    runtime: Runtime,
    output_dir: Path,
) -> None:
    config = _model_config(args)
    config.validate()
    all_train = set(
        source_names_for_split(
            "edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine
        )
    )
    all_dev = set(
        source_names_for_split(
            "edge_development", manifest_path=args.manifest, quarantine_path=args.quarantine
        )
    )
    gold: dict[str, PseudoGold] = {}
    gold_provenance: dict[str, Any] | None = None
    if args.real_gold:
        gold, gold_provenance = _load_pseudo_gold(
            Path(args.real_gold),
            allowed_train_names=all_train,
            forbidden_dev_names=all_dev,
            manifest_path=Path(args.manifest),
            data_root=Path(args.data_root),
        )
    qap_priors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    qap_provenance: dict[str, Any] | None = None
    if args.qap_priors:
        qap_priors, qap_provenance = _load_qap_priors(Path(args.qap_priors))
    train_names, dev_names, holdout_names, split_audit = _complete_source_split(
        args, gold=gold
    )
    qap_dev_overlap = sorted(
        set(qap_priors) & (set(dev_names) | set(holdout_names))
    )
    split_audit["qap_archive_development_names_ignored_count"] = len(qap_dev_overlap)
    split_audit["qap_archive_development_names_ignored_sha256"] = _names_sha256(
        qap_dev_overlap
    )
    data_root = Path(args.data_root)
    data_provenance = {
        "manifest_path": str(Path(args.manifest)),
        "manifest_sha256": _sha256(Path(args.manifest)),
        "quarantine_path": str(Path(args.quarantine)),
        "quarantine_sha256": _sha256(Path(args.quarantine)),
        "train_target_files_sha256": _named_files_sha256(
            [data_root / "train" / "targets" / name for name in train_names],
            root=data_root,
        ),
        "selection_target_files_sha256": _named_files_sha256(
            [data_root / "train" / "targets" / name for name in dev_names],
            root=data_root,
        ),
        "holdout_target_files_sha256": _named_files_sha256(
            [data_root / "train" / "targets" / name for name in holdout_names],
            root=data_root,
        ),
        "pseudo_gold_input_files_sha256": _named_files_sha256(
            [data_root / "train" / "inputs" / name for name in train_names if name in gold],
            root=data_root,
        )
        if any(name in gold for name in train_names)
        else None,
    }

    restorer: nn.Module | None = None
    restorer_metadata: dict[str, Any]
    if args.disable_denoiser:
        restorer_metadata = {
            "backend": "fixed_3x3_lowpass_fallback",
            "warning": "not the promoted restoration model",
        }
        data_provenance["denoiser_checkpoint_sha256"] = None
    else:
        checkpoint_path = Path(args.denoiser)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"denoiser checkpoint is missing: {checkpoint_path}")
        restorer, restorer_device, restorer_metadata = load_restorer(
            checkpoint_path, device=str(runtime.device)
        )
        if restorer_device != runtime.device:
            raise RuntimeError(
                f"restorer device mismatch: {restorer_device} != {runtime.device}"
            )
        for parameter in restorer.parameters():
            parameter.requires_grad_(False)
        data_provenance["denoiser_checkpoint_sha256"] = _sha256(checkpoint_path)

    amp_enabled, amp_dtype, amp_label = _amp_settings(args, runtime)
    model = ViTSinkhorn(config).to(runtime.device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if runtime.distributed:
        wrapped: nn.Module = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            find_unused_parameters=True,
        )
    else:
        wrapped = model
    optimizer = torch.optim.AdamW(
        wrapped.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0:
        raise ValueError("--amp-init-scale must be finite and positive")
    if args.max_consecutive_amp_skips < 0:
        raise ValueError("--max-consecutive-amp-skips must be non-negative")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16,
        init_scale=args.amp_init_scale,
    )
    _print(
        runtime,
        {
            "event": "pilot_start",
            "world_size": runtime.world_size,
            "device": str(runtime.device),
            "amp": amp_label,
            "trainable_parameters": trainable_parameters,
            "train_sources": len(train_names),
            "dev_sources": len(dev_names),
            "holdout_sources": len(holdout_names),
            "partial_pseudo_gold_sources": sum(name in gold for name in train_names),
        },
    )
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_development: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_scaler_state: dict[str, Any] | None = None
    best_rng_state: dict[str, Any] | None = None
    best_epoch: int | None = None
    best_score = -float("inf")
    supervision_counts = {
        "synthetic_known_permutation": 0,
        "partial_real_pseudo_gold": 0,
    }
    amp_skipped_steps = 0
    consecutive_amp_skips = 0

    for epoch in range(args.epochs):
        wrapped.train()
        local_indices = _epoch_indices(
            len(train_names),
            runtime,
            per_source_seed(args.seed, "vit-sinkhorn-epoch", str(epoch)),
        )
        local_sums = np.zeros(5, dtype=np.float64)
        for step, source_index in enumerate(local_indices):
            name = train_names[source_index]
            decision_rng = np.random.default_rng(
                per_source_seed(args.seed, "vit-sinkhorn-real-choice", name, epoch)
            )
            use_real = (
                name in gold
                and decision_rng.random() < args.real_gold_probability
            )
            if use_real:
                example = _prepare_partial_real(
                    name,
                    gold[name],
                    args=args,
                    restorer=restorer,
                    device=runtime.device,
                    qap_priors=qap_priors,
                )
            else:
                example = _prepare_synthetic(
                    name,
                    args=args,
                    epoch=epoch,
                    stage="train",
                    restorer=restorer,
                    device=runtime.device,
                )
            include_prior = (
                example.qap_tile_to_position is not None
                and decision_rng.random() < args.qap_prior_probability
            )
            if include_prior and example.supervision != "partial_real_pseudo_gold":
                raise RuntimeError(
                    "QAP priors are permitted only on partial real pseudo-gold"
                )
            tensors = _example_tensors(
                example, device=runtime.device, include_prior=include_prior
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(runtime, amp_enabled, amp_dtype):
                output = wrapped(
                    tensors["raw"],
                    tensors["restored"],
                    qap_tile_to_position=tensors["qap"],
                    qap_confidence=tensors["qap_confidence"],
                )
                losses = vit_sinkhorn_losses(
                    output,
                    tensors["targets"],
                    confidence=tensors["confidence"],
                    grid_size=GRID,
                    assignment_weight=args.assignment_weight,
                    directional_contrast_weight=args.directional_contrast_weight,
                    neighbor_consistency_weight=args.neighbor_consistency_weight,
                    contrast_temperature=args.contrast_temperature,
                    consistency_topk=args.consistency_topk,
                )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(
                    f"non-finite training loss for epoch={epoch} source={name}"
                )
            if scaler.is_enabled():
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    wrapped.parameters(), args.grad_clip
                )
                finite_flag = torch.tensor(
                    int(bool(torch.isfinite(gradient_norm).item())),
                    device=runtime.device,
                    dtype=torch.int32,
                )
                if runtime.distributed:
                    torch.distributed.all_reduce(
                        finite_flag, op=torch.distributed.ReduceOp.MIN
                    )
                gradients_finite = bool(finite_flag.item())
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                skipped = (not gradients_finite) or scale_after < scale_before
                if skipped:
                    amp_skipped_steps += 1
                    consecutive_amp_skips += 1
                    optimizer.zero_grad(set_to_none=True)
                    _print(
                        runtime,
                        {
                            "event": "amp_update_skipped",
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "source": name,
                            "scale_before": scale_before,
                            "scale_after": scale_after,
                            "consecutive_skips": consecutive_amp_skips,
                        },
                    )
                    if consecutive_amp_skips > args.max_consecutive_amp_skips:
                        raise RuntimeError(
                            "fp16 gradients remained non-finite after "
                            f"{consecutive_amp_skips} consecutive GradScaler skips"
                        )
                    continue
                consecutive_amp_skips = 0
            else:
                losses["total"].backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    wrapped.parameters(), args.grad_clip
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("non-finite gradient norm")
                optimizer.step()
            local_sums += np.asarray(
                [
                    float(losses["total"].detach().cpu()),
                    float(losses["assignment"].detach().cpu()),
                    float(losses["directional_contrast"].detach().cpu()),
                    float(losses["neighbor_consistency"].detach().cpu()),
                    1.0,
                ]
            )
            supervision_counts[example.supervision] += 1
            if runtime.primary and step % 4 == 0:
                _print(
                    runtime,
                    {
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "local_steps": len(local_indices),
                        "source": name,
                        "supervision": example.supervision,
                        "prior_used": include_prior,
                        "total_loss": float(losses["total"].detach().cpu()),
                    },
                )
        reduced = torch.tensor(local_sums, device=runtime.device, dtype=torch.float64)
        if runtime.distributed:
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
        totals = reduced.cpu().numpy()
        epoch_record: dict[str, Any] = {
            "epoch": epoch + 1,
            "curriculum_severity": _curriculum_severity(epoch, args.epochs),
            "train": {
                "steps_all_ranks": int(totals[4]),
                "total_loss": float(totals[0] / totals[4]),
                "assignment_loss": float(totals[1] / totals[4]),
                "directional_contrast_loss": float(totals[2] / totals[4]),
                "neighbor_consistency_loss": float(totals[3] / totals[4]),
            },
        }
        epoch_rank_rng_states = _gather_rank_rng_states(runtime)
        _barrier(runtime)
        if runtime.primary:
            development = _evaluate_development(
                model,
                dev_names,
                args=args,
                runtime=runtime,
                restorer=restorer,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                epoch=epoch + 1,
                stage="selection",
            )
            epoch_record["selection_development"] = development
            score = development["aggregate"]["model_without_prior"][
                "predicted_layout_ssim"
            ]
            if score > best_score:
                best_score = score
                best_development = development
                snapshot = _capture_training_state(
                    model,
                    optimizer,
                    scaler,
                    selected_epoch=epoch + 1,
                    rng_state={
                        "capture_point": "after selected training epoch before development evaluation",
                        "per_rank": epoch_rank_rng_states,
                    },
                )
                best_state = snapshot["model_state"]
                best_optimizer_state = snapshot["optimizer_state"]
                best_scaler_state = snapshot["scaler_state"]
                best_rng_state = snapshot["rng_state"]
                best_epoch = snapshot["selected_epoch"]
            _print(
                runtime,
                {
                    "event": "development_epoch",
                    "epoch": epoch + 1,
                    "score": score,
                    "gate_passed": development["gate_passed"],
                    "position_accuracy": development["aggregate"]["model_without_prior"][
                        "position_accuracy"
                    ],
                    "ssim": score,
                    "ssim_delta_vs_classical": development["derived"][
                        "ssim_delta_vs_classical"
                    ],
                },
            )
        history.append(epoch_record)
        _barrier(runtime)

    if not runtime.primary:
        return
    if (
        best_state is None
        or best_development is None
        or best_optimizer_state is None
        or best_scaler_state is None
        or best_rng_state is None
        or best_epoch is None
    ):
        raise RuntimeError("no development checkpoint was selected")
    model.load_state_dict(best_state)
    holdout_development = _evaluate_development(
        model,
        holdout_names,
        args=args,
        runtime=runtime,
        restorer=restorer,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        epoch=best_epoch,
        stage="holdout",
    )
    selection_gate_passed = bool(best_development["gate_passed"])
    holdout_gate_passed = bool(holdout_development["gate_passed"])
    gate_passed = selection_gate_passed and holdout_gate_passed
    status = "holdout_gate_passed" if gate_passed else "holdout_gate_failed"
    safe_for_submission = False
    source_code = _code_provenance()
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "vit_sinkhorn_bounded_development_pilot",
        "status": status,
        "safe_for_submission": safe_for_submission,
        "promotion_policy": (
            "selection plus independent holdout gates can justify a later genuine "
            "QAP and fixed real-layout evaluation only; never a submission"
        ),
        "seed": args.seed,
        "selected_epoch": best_epoch,
        "checkpoint_resume_state": {
            "optimizer_state_included": True,
            "scaler_state_included": True,
            "python_numpy_torch_rng_states_included_for_every_rank": True,
            "state_corresponds_to_selected_epoch": True,
        },
        "amp_update_audit": {
            "initial_scale": args.amp_init_scale,
            "skipped_steps_rank0": amp_skipped_steps,
            "max_consecutive_skips": args.max_consecutive_amp_skips,
            "final_scale": float(scaler.get_scale()) if scaler.is_enabled() else None,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "model_config": config.to_dict(),
        "trainable_parameters": trainable_parameters,
        "training_config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "overwrite"}
        },
        "curriculum": {
            "known_permutation_panels": ["primary_kornia", "independent_libjpeg"],
            "training_panel_policy": (
                "primary_kornia with fresh per-tile brightness, contrast, noise, "
                "blur, and JPEG draws for every source and epoch"
            ),
            "selection_panel": "primary_kornia",
            "holdout_panel": "independent_libjpeg on disjoint whole sources",
            "severity_start": _curriculum_severity(0, args.epochs),
            "severity_end": _curriculum_severity(args.epochs - 1, args.epochs),
            "synthetic_qap_prior": "disabled; no truth-derived prior is constructed",
            "partial_real_pseudo_gold_probability": args.real_gold_probability,
        },
        "supervision_counts_rank0": supervision_counts,
        "split_audit": split_audit,
        "data_provenance": data_provenance,
        "source_code": source_code,
        "restorer": restorer_metadata,
        "pseudo_gold": gold_provenance,
        "qap_priors": qap_provenance,
        "runtime": {
            "world_size": runtime.world_size,
            "device": str(runtime.device),
            "amp": amp_label,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        },
        "epoch_history": history,
        "selected_selection_development": best_development,
        "independent_holdout_development": holdout_development,
        "selection_gate_passed": selection_gate_passed,
        "holdout_gate_passed": holdout_gate_passed,
        "limitations": [
            "selection and holdout use exact synthetic panels, not hidden test labels",
            "partial real pseudo-gold is target-assisted input-to-clean matching on the training split and is not ground truth",
            "no genuine QAP asset is evaluated here and no QAP delta is claimed",
            "a passed holdout gate still requires genuine QAP and fixed real-layout evaluation",
            "checkpoint is always marked safe_for_submission=false in this pilot",
        ],
    }
    checkpoint = {
        "schema_version": 1,
        "kind": "vit_sinkhorn_development_checkpoint",
        "status": status,
        "safe_for_submission": safe_for_submission,
        "development_gate_passed": gate_passed,
        "selection_gate_passed": selection_gate_passed,
        "holdout_gate_passed": holdout_gate_passed,
        "selected_epoch": best_epoch,
        "model_config": config.to_dict(),
        "model_state": best_state,
        "optimizer_state": best_optimizer_state,
        "scaler_state": best_scaler_state,
        "rng_state": best_rng_state,
        "selected_selection_development": best_development,
        "independent_holdout_development": holdout_development,
        "split_audit": split_audit,
        "data_provenance": data_provenance,
        "pseudo_gold": gold_provenance,
        "qap_priors": qap_provenance,
        "source_code": source_code,
        "seed": args.seed,
    }
    artifacts = _write_success_artifacts(
        output_dir,
        checkpoint_payload=checkpoint,
        report=report,
    )
    _print(runtime, {"event": "pilot_complete", "status": status, **artifacts})


def _write_failure_report(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    runtime: Runtime | None,
    error: BaseException,
) -> None:
    report_path = output_dir / REPORT_NAME
    if report_path.exists():
        return
    payload = {
        "schema_version": 1,
        "kind": "vit_sinkhorn_failure_report",
        "status": "failed",
        "safe_for_submission": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "seed": getattr(args, "seed", None),
        "runtime": None
        if runtime is None
        else {
            "rank": runtime.rank,
            "world_size": runtime.world_size,
            "device": str(runtime.device),
        },
        "source_code": {
            "model_sha256": _sha256(MODEL_SOURCE) if MODEL_SOURCE.is_file() else None,
            "script_sha256": _sha256(SCRIPT_SOURCE),
        },
    }
    _atomic_json(report_path, payload)
    _write_hashes(output_dir, [report_path])


def main() -> None:
    args = parse_args()
    runtime: Runtime | None = None
    output_dir = Path(args.output_dir)
    try:
        _validate_args(args)
        output_dir = _preflight_output(args)
        runtime = _init_runtime(args.device)
        random.seed(args.seed + runtime.rank)
        np.random.seed((args.seed + runtime.rank) % (2**32))
        torch.manual_seed(args.seed + runtime.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + runtime.rank)
        if args.synthetic_smoke:
            _run_smoke(args, runtime, output_dir)
        else:
            _run_pilot(args, runtime, output_dir)
    except BaseException as error:
        if runtime is None or runtime.primary:
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                _write_failure_report(
                    output_dir,
                    args=args,
                    runtime=runtime,
                    error=error,
                )
            except Exception as report_error:
                print(
                    f"unable to write fail-closed report: {report_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    finally:
        _cleanup_runtime(runtime)


if __name__ == "__main__":
    main()
