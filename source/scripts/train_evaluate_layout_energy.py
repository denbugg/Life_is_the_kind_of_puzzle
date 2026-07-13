#!/usr/bin/env python3
"""Train and gate a raw-only layout plausibility Transformer.

The model sees proposed grids made from dirty 20x20 tiles.  Every positive and
its hard negatives share the exact same corruption realization; only tile
positions differ, so JPEG/noise/brightness artifacts cannot leak the label.

Default training is a serious but bounded 2xT4 signal pilot: 512 whole sources,
four epochs, four hard negatives per source, a fixed primary selection set and
a disjoint independent-libjpeg holdout.  It never opens test targets and every
artifact is marked ``safe_for_submission=false``.  A passed gate only permits a
later fixed real-QAP refinement experiment.
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
from torch.nn.parallel import DistributedDataParallel


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_assembly.geometry import GRID, TILE, TILE_COUNT, inverse_permutation
from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.layout_energy_transformer import (
    NEGATIVE_FAMILIES,
    LayoutEnergyConfig,
    LayoutEnergyTransformer,
    NegativeLayout,
    classical_seam_energy,
    iterative_refine_layout,
    layout_energy_losses,
    make_negative_layout,
    score_candidate_layouts,
)
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split


DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"
CHECKPOINT_NAME = "layout_energy_checkpoint.pt"
REPORT_NAME = "layout_energy_report.json"
HASHES_NAME = "SHA256SUMS.txt"
RESUME_NAME = "layout_energy_resume_epoch.pt"
MODEL_SOURCE = REPO_ROOT / "src/puzzle_assembly/layout_energy_transformer.py"
SCRIPT_SOURCE = Path(__file__).resolve()
IMPORTED_SOURCES = (
    REPO_ROOT / "src/puzzle_assembly/__init__.py",
    REPO_ROOT / "src/puzzle_assembly/geometry.py",
    REPO_ROOT / "src/puzzle_assembly/metrics.py",
    REPO_ROOT / "src/puzzle_assembly/compatibility.py",
    REPO_ROOT / "src/puzzle_assembly/components.py",
    REPO_ROOT / "src/puzzle_assembly/solvers.py",
    REPO_ROOT / "src/puzzle_assembly/panels.py",
    REPO_ROOT / "src/puzzle_assembly/protocol.py",
    REPO_ROOT / "src/puzzle_denoise_v2/__init__.py",
    REPO_ROOT / "src/puzzle_denoise_v2/degradation.py",
    REPO_ROOT / "src/puzzle_denoise_v2/model.py",
    REPO_ROOT / "src/puzzle_denoise_v2/tiles.py",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-sources", type=int, default=512)
    parser.add_argument("--selection-offset", type=int, default=0)
    parser.add_argument("--selection-sources", type=int, default=16)
    parser.add_argument("--holdout-sources", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--negatives-per-source", type=int, default=4)
    parser.add_argument("--eval-negatives", type=int, default=8)
    parser.add_argument("--eval-replicas", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "fp16", "bf16", "none"), default="auto")
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
    parser.add_argument("--amp-growth-interval", type=int, default=2000)
    parser.add_argument("--max-consecutive-amp-skips", type=int, default=8)
    parser.add_argument("--max-total-amp-skips", type=int, default=32)
    parser.add_argument(
        "--require-t4",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=0,
        help="Test/operations hook: write an epoch-boundary resume and stop; 0 runs all epochs",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--local-layers", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--global-layers", type=int, default=2)
    parser.add_argument("--global-tokens", type=int, default=6)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--edge-dim", type=int, default=48)
    parser.add_argument("--edge-band", type=int, default=3)
    parser.add_argument("--move-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=0.20)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=0.5)
    parser.add_argument("--local-error-weight", type=float, default=0.5)
    parser.add_argument("--move-weight", type=float, default=0.25)
    parser.add_argument("--move-matching-weight", type=float, default=0.10)
    parser.add_argument("--graded-monotonic-weight", type=float, default=0.35)
    parser.add_argument("--energy-regularization", type=float, default=1e-4)
    parser.add_argument("--repair-steps", type=int, default=6)
    parser.add_argument("--repair-beam-width", type=int, default=3)
    parser.add_argument("--repair-hot-positions", type=int, default=32)
    parser.add_argument("--repair-proposals", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=6)
    parser.add_argument("--gate-min-ranking-accuracy", type=float, default=0.65)
    parser.add_argument("--gate-min-delta-vs-classical", type=float, default=0.03)
    parser.add_argument("--gate-min-delta-vs-random", type=float, default=0.08)
    parser.add_argument("--gate-min-energy-margin", type=float, default=0.10)
    parser.add_argument("--gate-min-local-auc", type=float, default=0.65)
    parser.add_argument("--gate-min-graded-monotonic-accuracy", type=float, default=0.65)
    parser.add_argument("--gate-min-relative-repair-error-reduction", type=float, default=0.25)
    parser.add_argument("--gate-min-repair-adjacency-delta", type=float, default=0.01)
    parser.add_argument("--gate-min-control-win-fraction", type=float, default=0.60)
    return parser


def parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


class Runtime:
    def __init__(
        self,
        device: torch.device,
        rank: int,
        local_rank: int,
        world_size: int,
        distributed: bool,
    ) -> None:
        self.device = device
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.distributed = distributed

    @property
    def primary(self) -> bool:
        return self.rank == 0


def _init_runtime(device_request: str) -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP layout-energy pilot requires CUDA/NCCL")
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


def _cleanup(runtime: Runtime | None) -> None:
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


def _names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _tile_multiset_sha256(tiles: np.ndarray) -> str:
    values = np.asarray(tiles)
    if values.ndim != 4:
        raise ValueError("tile multiset must be an NxHxWxC array")
    tile_hashes = sorted(
        hashlib.sha256(np.ascontiguousarray(tile).tobytes()).hexdigest()
        for tile in values
    )
    return hashlib.sha256("\n".join(tile_hashes).encode("ascii")).hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _named_file_set_sha256(paths: list[Path], root: Path) -> str:
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
    paths = (MODEL_SOURCE, SCRIPT_SOURCE, *IMPORTED_SOURCES)
    records = {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in paths
        if path.is_file()
    }
    if len(records) != len(paths):
        raise FileNotFoundError("one or more imported source files are missing")
    combined = hashlib.sha256(
        "\n".join(f"{name}\0{records[name]}" for name in sorted(records)).encode("utf-8")
    ).hexdigest()
    return {"files": records, "combined_sha256": combined}


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resume_contract(
    args: argparse.Namespace,
    *,
    split_audit: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    source_code: Mapping[str, Any],
) -> dict[str, Any]:
    excluded = {
        "output_dir",
        "overwrite",
        "resume_checkpoint",
        "synthetic_smoke",
        "smoke_steps",
        "stop_after_epoch",
    }
    training_args = {
        key: value for key, value in vars(args).items() if key not in excluded
    }
    contract = {
        "schema_version": 1,
        "training_args": training_args,
        "training_args_sha256": _json_sha256(training_args),
        "split_sha256": _json_sha256(dict(split_audit)),
        "data_sha256": _json_sha256(dict(data_provenance)),
        "source_code_combined_sha256": source_code["combined_sha256"],
    }
    contract["contract_sha256"] = _json_sha256(contract)
    return contract


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


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _gather_rng(runtime: Runtime) -> list[dict[str, Any]] | None:
    local = _rng_state()
    if not runtime.distributed:
        return [local]
    gathered: list[dict[str, Any] | None] | None = (
        [None] * runtime.world_size if runtime.primary else None
    )
    torch.distributed.gather_object(local, gathered, dst=0)
    if not runtime.primary:
        return None
    assert gathered is not None and all(item is not None for item in gathered)
    return [item for item in gathered if item is not None]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _write_hashes(output_dir: Path, paths: list[Path]) -> Path:
    target = output_dir / HASHES_NAME
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in sorted(paths)) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _preflight_output(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = [
        output / CHECKPOINT_NAME,
        output / REPORT_NAME,
        output / HASHES_NAME,
        output / RESUME_NAME,
    ]
    existing = [str(path) for path in artifacts if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"artifacts already exist; pass --overwrite: {existing}")
    if args.overwrite:
        if args.resume_checkpoint:
            resume_source = Path(args.resume_checkpoint).resolve()
            if any(path.resolve() == resume_source for path in artifacts):
                raise ValueError("refusing to overwrite the resume checkpoint being loaded")
        for path in artifacts:
            path.unlink(missing_ok=True)
    return output


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (GRID * TILE, GRID * TILE, 3):
        raise ValueError(f"unexpected image shape {values.shape} for {path}")
    return values


def _split_sources(args: argparse.Namespace) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    train_all = source_names_for_split(
        "edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine
    )
    development_all = source_names_for_split(
        "edge_development", manifest_path=args.manifest, quarantine_path=args.quarantine
    )
    if set(train_all) & set(development_all):
        raise RuntimeError("edge_train and edge_development overlap")
    train = train_all[args.train_offset : args.train_offset + args.train_sources]
    selection = development_all[
        args.selection_offset : args.selection_offset + args.selection_sources
    ]
    holdout_start = args.selection_offset + args.selection_sources
    holdout = development_all[holdout_start : holdout_start + args.holdout_sources]
    if len(train) != args.train_sources or len(selection) != args.selection_sources or len(holdout) != args.holdout_sources:
        raise ValueError("requested source slice exceeds authoritative partitions")
    if set(train) & (set(selection) | set(holdout)) or set(selection) & set(holdout):
        raise RuntimeError("train/selection/holdout overlap")
    data_root = Path(args.data_root)
    test_names = {path.name for path in (data_root / "test").glob("*.png")}
    overlap = (set(train) | set(selection) | set(holdout)) & test_names
    if overlap:
        raise RuntimeError(f"selected source names overlap test: {sorted(overlap)[:5]}")
    missing = [
        name
        for name in train + selection + holdout
        if not (data_root / "train" / "targets" / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing selected targets: {missing[:8]}")
    audit = {
        "policy": "whole-source train, fixed selection, disjoint one-shot holdout",
        "train_partition": "edge_train",
        "development_partition": "edge_development",
        "audit_opened": False,
        "test_targets_opened": False,
        "train_names": train,
        "selection_names": selection,
        "holdout_names": holdout,
        "train_names_sha256": _names_sha256(train),
        "selection_names_sha256": _names_sha256(selection),
        "holdout_names_sha256": _names_sha256(holdout),
        "train_selection_holdout_overlap_count": 0,
        "test_name_overlap_count": 0,
    }
    return train, selection, holdout, audit


def _tile_features(tiles: np.ndarray) -> np.ndarray:
    values = tiles.astype(np.float32) / 255.0
    pooled = values.reshape(TILE_COUNT, 4, 5, 4, 5, 3).mean(axis=(2, 4))
    moments = np.concatenate(
        [values.mean(axis=(1, 2)), values.std(axis=(1, 2))], axis=1
    )
    return np.concatenate([pooled.reshape(TILE_COUNT, -1), moments], axis=1)


def _raw_border_l1_w2(tiles: np.ndarray, *, chunk_size: int = 64) -> CompatibilityMatrices:
    if tiles.shape != (TILE_COUNT, TILE, TILE, 3) or tiles.dtype != np.uint8:
        raise ValueError("raw seam solver expects uint8 576x20x20x3 tiles")
    values = tiles.astype(np.float32) / 255.0
    strip = 2
    right_query = values[:, :, -strip:, :].reshape(TILE_COUNT, -1)
    right_key = values[:, :, :strip, :].reshape(TILE_COUNT, -1)
    down_query = values[:, -strip:, :, :].reshape(TILE_COUNT, -1)
    down_key = values[:, :strip, :, :].reshape(TILE_COUNT, -1)
    right = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    down = np.empty_like(right)
    for start in range(0, TILE_COUNT, chunk_size):
        stop = min(start + chunk_size, TILE_COUNT)
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
    return CompatibilityMatrices("raw_rgb_l1_w2", right, down)


@dataclass(frozen=True)
class PreparedPanel:
    source_name: str
    raw_tiles: np.ndarray
    slot_to_target: np.ndarray
    correct_position_to_slot: np.ndarray
    first_pass_position_to_slot: np.ndarray
    first_pass_position_to_target: np.ndarray
    metadata: dict[str, Any]


def _panel_for_source(
    name: str,
    *,
    args: argparse.Namespace,
    epoch: int,
    stage: str,
    evaluation_replica: int = 0,
) -> PreparedPanel:
    if stage not in {"train", "selection", "holdout"}:
        raise ValueError("stage must be train, selection, or holdout")
    panel_name = "independent_libjpeg" if stage == "holdout" else "primary_kornia"
    replica = epoch if stage == "train" else evaluation_replica
    seed = per_source_seed(
        args.seed, f"layout-energy-{stage}-{panel_name}", name, replica
    )
    clean = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
    panel = make_exact_panel(clean, panel=panel_name, seed=seed)
    raw_tiles = np.ascontiguousarray(panel.slot_tiles)
    correct = inverse_permutation(panel.slot_to_target).astype(np.int32, copy=False)
    compatibility = _raw_border_l1_w2(raw_tiles)
    first_pass = soft_cycle_component_solver(
        compatibility,
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        loop_weight=1.0,
        reciprocal_weight=0.35,
    ).position_to_slot.astype(np.int32, copy=False)
    first_pass_target = panel.slot_to_target[first_pass].astype(np.int32, copy=False)
    metadata = {
        "source_name": name,
        "panel": panel_name,
        "panel_seed": seed,
        "evaluation_replica": evaluation_replica if stage != "train" else None,
        "corrupted_tile_multiset_sha256": _tile_multiset_sha256(raw_tiles),
        "correct_position_to_slot_sha256": _array_sha256(correct),
        "raw_seam_first_pass_position_to_slot_sha256": _array_sha256(first_pass),
        "raw_seam_first_pass_position_to_target_sha256": _array_sha256(first_pass_target),
        "raw_seam_compatibility_sha256": _array_sha256(
            np.stack([compatibility.right, compatibility.down])
        ),
        "raw_seam_solver": {
            "compatibility": "input-only raw RGB border L1 strip=2",
            "solver": "soft_cycle_component_solver",
            "top_k": 8,
            "keep_per_tile": 1,
            "proposal_keep_fraction": 0.5,
            "loop_weight": 1.0,
            "reciprocal_weight": 0.35,
            "targets_opened": False,
        },
    }
    return PreparedPanel(
        source_name=name,
        raw_tiles=raw_tiles,
        slot_to_target=panel.slot_to_target.astype(np.int32, copy=False),
        correct_position_to_slot=correct,
        first_pass_position_to_slot=first_pass,
        first_pass_position_to_target=first_pass_target,
        metadata=metadata,
    )


def curriculum_families(epoch: int, epochs: int) -> tuple[tuple[str, ...], float]:
    if epochs <= 0 or not 0 <= epoch < epochs:
        raise ValueError("epoch must be in [0,epochs)")
    fraction = epoch / max(epochs - 1, 1)
    if fraction < 0.34:
        families = ("row_column", "block_swap", "component_translation", "segment")
        severity = 0.22
    elif fraction < 0.67:
        families = (
            "component_translation",
            "segment",
            "block_swap",
            "sparse_swap",
            "solver_like_sparse",
        )
        severity = 0.10
    else:
        families = (
            "solver_like_sparse",
            "similar_swap",
            "sparse_swap",
            "segment",
            "mixture",
        )
        severity = 0.04
    return families, severity


def _candidate_set(
    panel: PreparedPanel,
    *,
    source_name: str,
    args: argparse.Namespace,
    epoch: int,
    stage: str,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    if count <= 0:
        raise ValueError("negative count must be positive")
    if stage == "train":
        families, severity = curriculum_families(epoch, args.epochs)
    else:
        families = NEGATIVE_FAMILIES
        severity = 0.05
    correct_tiles = panel.raw_tiles[panel.correct_position_to_slot]
    features = _tile_features(correct_tiles)
    semantic_layouts = [np.arange(TILE_COUNT, dtype=np.int32)]
    errors = [np.zeros(TILE_COUNT, dtype=np.float32)]
    moves = [np.zeros((TILE_COUNT, 2), dtype=np.float32)]
    severities = [0.0]
    labels = ["positive_correct"]
    solver_semantic = panel.first_pass_position_to_target.copy()

    def supervision(layout: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        expected = np.arange(TILE_COUNT, dtype=np.int32)
        positions = expected
        scale = float(GRID - 1)
        error = (layout != expected).astype(np.float32)
        move = np.stack(
            [
                (layout // GRID - positions // GRID) / scale,
                (layout % GRID - positions % GRID) / scale,
            ],
            axis=1,
        ).astype(np.float32)
        stats = _layout_stats(layout)
        severity_value = (
            0.40 * stats["error_fraction"]
            + 0.40 * (1.0 - stats["combined_adjacency"])
            + 0.20 * min(stats["mean_manhattan"] / (2.0 * GRID), 1.0)
        )
        return error, move, float(severity_value)

    def append_semantic(layout: np.ndarray, label: str) -> None:
        if len(semantic_layouts) >= count + 1:
            return
        if np.array_equal(layout, np.arange(TILE_COUNT)):
            return
        if any(np.array_equal(layout, existing) for existing in semantic_layouts):
            return
        error, move, severity_value = supervision(layout)
        semantic_layouts.append(layout.astype(np.int32, copy=True))
        errors.append(error)
        moves.append(move)
        severities.append(severity_value)
        labels.append(label)

    append_semantic(solver_semantic, "raw_seam_first_pass")
    residual = solver_semantic.copy()
    repair_rng = np.random.default_rng(
        per_source_seed(
            args.seed, f"layout-energy-residual-chain-{stage}", source_name, epoch
        )
    )
    repair_number = 0
    while len(semantic_layouts) < min(count + 1, 4):
        wrong = np.flatnonzero(residual != np.arange(TILE_COUNT))
        if len(wrong) == 0:
            break
        current_severity = supervision(residual)[2]
        best_candidate: np.ndarray | None = None
        best_candidate_severity = current_severity
        candidates_to_try = repair_rng.choice(
            wrong, size=min(64, len(wrong)), replace=False
        )
        for value in candidates_to_try:
            position = int(value)
            source_position = int(np.flatnonzero(residual == position)[0])
            candidate = residual.copy()
            candidate[position], candidate[source_position] = (
                candidate[source_position],
                candidate[position],
            )
            candidate_severity = supervision(candidate)[2]
            if candidate_severity < best_candidate_severity - 1e-9:
                best_candidate = candidate
                best_candidate_severity = candidate_severity
        if best_candidate is None:
            break
        residual = best_candidate
        repair_number += 1
        append_semantic(
            residual,
            f"raw_seam_residual_repair_{repair_number}",
        )
    family_offset = per_source_seed(
        args.seed, f"layout-energy-family-offset-{stage}", source_name, epoch
    ) % len(families)
    synthetic_index = 0
    while len(semantic_layouts) < count + 1:
        family = families[(family_offset + synthetic_index) % len(families)]
        rng = np.random.default_rng(
            per_source_seed(
                args.seed,
                f"layout-energy-negative-{stage}-{family}",
                source_name,
                epoch * 1000 + synthetic_index,
            )
        )
        negative = make_negative_layout(
            grid_size=GRID,
            family=family,
            rng=rng,
            tile_features=features,
            severity=severity,
        )
        append_semantic(negative.position_to_tile, family)
        synthetic_index += 1
        if synthetic_index > count * 8:
            raise RuntimeError("unable to create unique candidate layouts")
    semantic_array = np.stack(semantic_layouts)
    layouts_array = np.stack(
        [panel.correct_position_to_slot[layout] for layout in semantic_array]
    )
    # Explicit invariant: every candidate is an index permutation of the exact
    # same corrupted tiles.  No re-corruption occurs after this point.
    if not all(np.array_equal(np.sort(row), np.arange(TILE_COUNT)) for row in layouts_array):
        raise RuntimeError("candidate generation did not preserve the tile multiset")
    return (
        layouts_array,
        np.stack(errors),
        np.stack(moves),
        np.asarray(severities, dtype=np.float32),
        labels,
        semantic_array,
    )


def _tile_tensor(tiles: np.ndarray, device: torch.device) -> torch.Tensor:
    if tiles.shape != (TILE_COUNT, TILE, TILE, 3) or tiles.dtype != np.uint8:
        raise ValueError("tiles must be uint8 576x20x20x3")
    return torch.from_numpy(
        np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))
    ).to(device=device, dtype=torch.float32).div_(255.0).unsqueeze(0)


def _model_config(args: argparse.Namespace) -> LayoutEnergyConfig:
    return LayoutEnergyConfig(
        grid_size=GRID,
        tile_size=TILE,
        d_model=args.d_model,
        num_heads=args.heads,
        local_layers=args.local_layers,
        window_size=args.window_size,
        global_layers=args.global_layers,
        global_tokens=args.global_tokens,
        feedforward_dim=args.feedforward_dim,
        cnn_channels=args.cnn_channels,
        edge_dim=args.edge_dim,
        edge_band=args.edge_band,
        move_dim=args.move_dim,
        dropout=args.dropout,
    )


def _amp(args: argparse.Namespace, runtime: Runtime) -> tuple[bool, torch.dtype, str]:
    if args.amp == "none" or runtime.device.type != "cuda":
        return False, torch.float32, "none"
    label = "fp16" if args.amp == "auto" else args.amp
    return True, torch.float16 if label == "fp16" else torch.bfloat16, label


def _autocast(runtime: Runtime, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=runtime.device.type, dtype=dtype)


def _synchronized_all_finite(runtime: Runtime, local_finite: bool) -> bool:
    flag = torch.tensor(
        1 if local_finite else 0,
        device=runtime.device,
        dtype=torch.int32,
    )
    if runtime.distributed:
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def _bounded_amp_skip_update(
    *,
    scale_before: float,
    consecutive_skips: int,
    total_skips: int,
    max_consecutive: int,
    max_total: int,
) -> tuple[float, int, int]:
    new_consecutive = consecutive_skips + 1
    new_total = total_skips + 1
    if new_consecutive > max_consecutive or new_total > max_total:
        raise RuntimeError(
            "bounded AMP skip budget exhausted: "
            f"consecutive={new_consecutive} total={new_total}"
        )
    return max(float(scale_before) * 0.5, 1.0), new_consecutive, new_total


def _rank_hardware_probe(runtime: Runtime, *, require_t4: bool) -> dict[str, Any]:
    if runtime.device.type != "cuda":
        if require_t4:
            raise RuntimeError("full pilot requires CUDA T4 devices")
        return {
            "rank": runtime.rank,
            "device": str(runtime.device),
            "cuda": False,
        }
    index = runtime.device.index or 0
    name = torch.cuda.get_device_name(index)
    capability = tuple(torch.cuda.get_device_capability(index))
    if require_t4 and ("T4" not in name.upper() or capability != (7, 5)):
        raise RuntimeError(
            f"rank {runtime.rank} requires T4 sm_75, got {name} capability={capability}"
        )
    generator = torch.Generator(device=runtime.device).manual_seed(20260711 + runtime.rank)
    left = torch.randn(
        512, 512, device=runtime.device, dtype=torch.float16, generator=generator
    )
    right = torch.randn(
        512, 512, device=runtime.device, dtype=torch.float16, generator=generator
    )
    product = left @ right
    finite = bool(torch.isfinite(product).all())
    if not finite:
        raise RuntimeError(f"rank {runtime.rank} failed fp16 matmul preflight")
    torch.cuda.synchronize(runtime.device)
    result = {
        "rank": runtime.rank,
        "local_rank": runtime.local_rank,
        "device": str(runtime.device),
        "name": name,
        "capability": list(capability),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "arch_list": torch.cuda.get_arch_list(),
        "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
        "fp16_matmul_shape": [512, 512, 512],
        "fp16_matmul_mean": float(product.float().mean().item()),
        "fp16_matmul_finite": finite,
    }
    del left, right, product
    torch.cuda.empty_cache()
    return result


def _gather_objects(runtime: Runtime, value: Any) -> list[Any] | None:
    if not runtime.distributed:
        return [value]
    gathered: list[Any] | None = [None] * runtime.world_size if runtime.primary else None
    torch.distributed.gather_object(value, gathered, dst=0)
    return gathered if runtime.primary else None


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _layout_stats(layout: np.ndarray, grid: int = GRID) -> dict[str, float]:
    values = np.asarray(layout, dtype=np.int32)
    expected = np.arange(grid * grid, dtype=np.int32)
    tiled = values.reshape(grid, grid)
    right = (tiled[:, :-1] % grid != grid - 1) & (tiled[:, 1:] == tiled[:, :-1] + 1)
    down = tiled[1:, :] == tiled[:-1, :] + grid
    displacement = np.abs(values // grid - expected // grid) + np.abs(
        values % grid - expected % grid
    )
    return {
        "error_fraction": float(np.mean(values != expected)),
        "combined_adjacency": float(0.5 * (right.mean() + down.mean())),
        "mean_manhattan": float(displacement.mean()),
    }


@torch.inference_mode()
def _equal_budget_control(
    model: LayoutEnergyTransformer,
    panel: PreparedPanel,
    initial_position_to_slot: np.ndarray,
    *,
    method: str,
    pair_schedule: list[list[tuple[int, int]]],
    runtime: Runtime,
    args: argparse.Namespace,
    amp_dtype: torch.dtype | None,
    seed: int,
) -> dict[str, Any]:
    if method not in {"learned_energy", "classical_seam", "random_energy", "no_op"}:
        raise ValueError(f"unknown equal-budget control {method}")
    current = initial_position_to_slot.copy()
    initial_semantic = panel.slot_to_target[current]
    before = _layout_stats(initial_semantic)
    candidates_scored = 0
    for step, pairs in enumerate(pair_schedule):
        candidates = [current.copy()]
        for first, second in pairs:
            candidate = current.copy()
            candidate[first], candidate[second] = candidate[second], candidate[first]
            candidates.append(candidate)
        layouts = np.stack(candidates)
        candidates_scored += len(layouts)
        if method == "no_op":
            chosen = 0
        elif method == "learned_energy":
            values = score_candidate_layouts(
                model,
                panel.raw_tiles,
                layouts,
                device=runtime.device,
                batch_size=args.score_batch_size,
                autocast_dtype=amp_dtype,
            ).energies
            chosen = int(np.argmin(values))
        elif method == "classical_seam":
            ordered = np.stack([panel.raw_tiles[layout] for layout in layouts])
            chosen = int(np.argmin(classical_seam_energy(ordered)))
        else:
            rng = np.random.default_rng(seed + step)
            chosen = int(np.argmin(rng.standard_normal(len(layouts))))
        current = layouts[chosen].copy()
    after = _layout_stats(panel.slot_to_target[current])
    absolute = before["error_fraction"] - after["error_fraction"]
    relative = absolute / before["error_fraction"] if before["error_fraction"] > 0 else 1.0
    return {
        "method": method,
        "position_to_slot_sha256": _array_sha256(current),
        "candidates_scored": candidates_scored,
        "before": before,
        "after": after,
        "absolute_error_fraction_reduction": absolute,
        "relative_error_reduction": relative,
        "adjacency_delta": after["combined_adjacency"] - before["combined_adjacency"],
    }


def _bootstrap_ci(values: np.ndarray, *, seed: int, draws: int = 2000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


@torch.inference_mode()
def _evaluate(
    model: LayoutEnergyTransformer,
    names: list[str],
    *,
    args: argparse.Namespace,
    runtime: Runtime,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    stage: str,
) -> dict[str, Any] | None:
    if stage not in {"selection", "holdout"}:
        raise ValueError("evaluation stage must be selection or holdout")
    model.eval()
    tasks = [
        (name, replica)
        for name in names
        for replica in range(args.eval_replicas)
    ]
    local_tasks = tasks[runtime.rank :: runtime.world_size]
    local_records: list[dict[str, Any]] = []
    for name, replica in local_tasks:
        panel = _panel_for_source(
            name,
            args=args,
            epoch=0,
            stage=stage,
            evaluation_replica=replica,
        )
        layouts, errors, moves, severities, families, semantic_layouts = _candidate_set(
            panel,
            source_name=name,
            args=args,
            epoch=replica,
            stage=stage,
            count=args.eval_negatives,
        )
        base = _tile_tensor(panel.raw_tiles, runtime.device)
        layout_tensor = torch.from_numpy(layouts).to(runtime.device).long().unsqueeze(0)
        with _autocast(runtime, amp_enabled, amp_dtype):
            output = model(base, candidate_layouts=layout_tensor)
        energies = output.energy.float().cpu().numpy()
        probabilities = output.local_error_logits.float().sigmoid().cpu().numpy()
        learned_accuracy = float(np.mean(energies[0] < energies[1:]))
        energy_margin = float(np.mean(energies[1:] - energies[0]))
        ranking_probability = 1.0 / (
            1.0 + np.exp(-np.clip(energies[1:] - energies[0], -30.0, 30.0))
        )
        uncalibrated_brier = float(np.mean((1.0 - ranking_probability) ** 2))
        ordered = np.stack([panel.raw_tiles[layout] for layout in layouts])
        classical = np.asarray(classical_seam_energy(ordered), dtype=np.float64)
        classical_accuracy = float(np.mean(classical[0] < classical[1:]))
        random_rng = np.random.default_rng(
            per_source_seed(args.seed, f"layout-energy-random-{stage}", name, replica)
        )
        random_energy = random_rng.standard_normal(len(layouts))
        random_accuracy = float(np.mean(random_energy[0] < random_energy[1:]))
        local_auc = _binary_auc(errors[1:], probabilities[1:])
        severity_difference = severities[:, None] - severities[None, :]
        graded_pairs = severity_difference > 1e-6
        energy_difference = energies[:, None] - energies[None, :]
        graded_accuracy = float(np.mean(energy_difference[graded_pairs] > 0))

        initial = panel.first_pass_position_to_slot
        before = _layout_stats(panel.first_pass_position_to_target)
        initial_error_count = int(round(before["error_fraction"] * TILE_COUNT))
        maximum_positions_per_step = 2 * max(1, args.repair_hot_positions // 2)
        theoretical_max_relative = (
            min(
                1.0,
                args.repair_steps * maximum_positions_per_step / initial_error_count,
            )
            if initial_error_count > 0
            else 1.0
        )
        if args.gate_min_relative_repair_error_reduction > theoretical_max_relative + 1e-12:
            raise ValueError(
                "repair gate is theoretically infeasible for raw-seam first pass: "
                f"required={args.gate_min_relative_repair_error_reduction} "
                f"max={theoretical_max_relative} errors={initial_error_count}"
            )
        refinement = iterative_refine_layout(
            model,
            panel.raw_tiles,
            initial,
            device=runtime.device,
            steps=args.repair_steps,
            beam_width=args.repair_beam_width,
            hot_positions=args.repair_hot_positions,
            proposals_per_layout=args.repair_proposals,
            score_batch_size=args.score_batch_size,
            autocast_dtype=amp_dtype if amp_enabled else None,
        )
        after_semantic = panel.slot_to_target[refinement.position_to_slot]
        after = _layout_stats(after_semantic)
        absolute_reduction = before["error_fraction"] - after["error_fraction"]
        relative_reduction = (
            absolute_reduction / before["error_fraction"]
            if before["error_fraction"] > 0
            else 1.0
        )

        schedule: list[list[tuple[int, int]]] = []
        schedule_rng = np.random.default_rng(
            per_source_seed(args.seed, f"layout-energy-control-{stage}", name, replica)
        )
        for _ in range(args.repair_steps):
            pairs: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            while len(pairs) < args.repair_proposals:
                first, second = sorted(
                    int(value)
                    for value in schedule_rng.choice(TILE_COUNT, size=2, replace=False)
                )
                if (first, second) not in seen:
                    seen.add((first, second))
                    pairs.append((first, second))
            schedule.append(pairs)
        controls = {
            method: _equal_budget_control(
                model,
                panel,
                initial,
                method=method,
                pair_schedule=schedule,
                runtime=runtime,
                args=args,
                amp_dtype=amp_dtype if amp_enabled else None,
                seed=per_source_seed(
                    args.seed, f"layout-energy-control-score-{stage}-{method}", name, replica
                ),
            )
            for method in ("learned_energy", "classical_seam", "random_energy", "no_op")
        }
        best_nonlearned = max(
            controls[method]["relative_error_reduction"]
            for method in ("classical_seam", "random_energy", "no_op")
        )
        local_records.append(
            {
                "source_name": name,
                "evaluation_replica": replica,
                **panel.metadata,
                "families": families,
                "candidate_layouts_sha256": _array_sha256(layouts),
                "candidate_semantic_layouts_sha256": _array_sha256(semantic_layouts),
                "candidate_severities": severities.tolist(),
                "learned_ranking_accuracy": learned_accuracy,
                "classical_ranking_accuracy": classical_accuracy,
                "random_ranking_accuracy": random_accuracy,
                "mean_negative_minus_positive_energy": energy_margin,
                "uncalibrated_pairwise_brier_diagnostic": uncalibrated_brier,
                "local_error_auc": local_auc,
                "graded_monotonic_accuracy": graded_accuracy,
                "_local_labels": errors[1:].reshape(-1),
                "_local_scores": probabilities[1:].reshape(-1),
                "repair": {
                    "initial_family": "genuine_input_only_raw_seam_first_pass",
                    "initial_layout_sha256": _array_sha256(initial),
                    "final_layout_sha256": _array_sha256(refinement.position_to_slot),
                    "initial_energy": refinement.initial_energy,
                    "final_energy": refinement.final_energy,
                    "before": before,
                    "after": after,
                    "absolute_error_fraction_reduction": absolute_reduction,
                    "relative_error_reduction": relative_reduction,
                    "adjacency_delta": after["combined_adjacency"] - before["combined_adjacency"],
                    "manhattan_reduction": before["mean_manhattan"] - after["mean_manhattan"],
                    "steps": [vars(item) for item in refinement.steps],
                    "theoretical_max_relative_error_reduction": theoretical_max_relative,
                },
                "equal_budget_controls": controls,
                "equal_budget_learned_minus_best_control_relative_reduction": (
                    controls["learned_energy"]["relative_error_reduction"]
                    - best_nonlearned
                ),
            }
        )
    if runtime.distributed:
        gathered: list[list[dict[str, Any]] | None] | None = (
            [None] * runtime.world_size if runtime.primary else None
        )
        torch.distributed.gather_object(local_records, gathered, dst=0)
        if not runtime.primary:
            return None
        assert gathered is not None and all(value is not None for value in gathered)
        records = [record for group in gathered if group is not None for record in group]
    else:
        records = local_records
    records.sort(key=lambda record: (record["source_name"], record["evaluation_replica"]))
    all_local_labels = [record.pop("_local_labels") for record in records]
    all_local_scores = [record.pop("_local_scores") for record in records]
    aggregate = {
        key: float(np.mean([record[key] for record in records]))
        for key in (
            "learned_ranking_accuracy",
            "classical_ranking_accuracy",
            "random_ranking_accuracy",
            "mean_negative_minus_positive_energy",
            "uncalibrated_pairwise_brier_diagnostic",
            "local_error_auc",
            "graded_monotonic_accuracy",
        )
    }
    aggregate["pooled_local_error_auc"] = _binary_auc(
        np.concatenate(all_local_labels), np.concatenate(all_local_scores)
    )
    aggregate["repair_relative_error_reduction"] = float(
        np.mean([record["repair"]["relative_error_reduction"] for record in records])
    )
    aggregate["repair_adjacency_delta"] = float(
        np.mean([record["repair"]["adjacency_delta"] for record in records])
    )
    aggregate["repair_manhattan_reduction"] = float(
        np.mean([record["repair"]["manhattan_reduction"] for record in records])
    )
    source_control_differences = np.asarray(
        [
            np.mean(
                [
                    record["equal_budget_learned_minus_best_control_relative_reduction"]
                    for record in records
                    if record["source_name"] == name
                ]
            )
            for name in names
        ],
        dtype=np.float64,
    )
    control_ci = _bootstrap_ci(
        source_control_differences,
        seed=per_source_seed(args.seed, f"layout-energy-control-ci-{stage}", "all"),
    )
    control_win_fraction = float(np.mean(source_control_differences > 0))
    aggregate["equal_budget_control_win_fraction"] = control_win_fraction
    aggregate["equal_budget_learned_minus_best_control"] = control_ci
    ranking_required = max(
        args.gate_min_ranking_accuracy,
        aggregate["classical_ranking_accuracy"]
        + args.gate_min_delta_vs_classical
        * (1.0 - aggregate["classical_ranking_accuracy"]),
        aggregate["random_ranking_accuracy"]
        + args.gate_min_delta_vs_random
        * (1.0 - aggregate["random_ranking_accuracy"]),
    )
    gates = {
        "ranking_vs_classical_and_random": {
            "value": aggregate["learned_ranking_accuracy"],
            "minimum": ranking_required,
            "passed": aggregate["learned_ranking_accuracy"] >= ranking_required,
            "classical_ceiling_aware": True,
        },
        "positive_energy_margin": {
            "value": aggregate["mean_negative_minus_positive_energy"],
            "minimum": args.gate_min_energy_margin,
            "passed": aggregate["mean_negative_minus_positive_energy"] >= args.gate_min_energy_margin,
        },
        "local_error_auc": {
            "value": aggregate["pooled_local_error_auc"],
            "minimum": args.gate_min_local_auc,
            "passed": aggregate["pooled_local_error_auc"] >= args.gate_min_local_auc,
        },
        "graded_wrong_vs_less_wrong_ordering": {
            "value": aggregate["graded_monotonic_accuracy"],
            "minimum": args.gate_min_graded_monotonic_accuracy,
            "passed": aggregate["graded_monotonic_accuracy"]
            >= args.gate_min_graded_monotonic_accuracy,
        },
        "genuine_raw_seam_target_free_relative_repair": {
            "value": aggregate["repair_relative_error_reduction"],
            "minimum": args.gate_min_relative_repair_error_reduction,
            "passed": aggregate["repair_relative_error_reduction"]
            >= args.gate_min_relative_repair_error_reduction,
        },
        "equal_budget_control_source_wins": {
            "value": control_win_fraction,
            "minimum": args.gate_min_control_win_fraction,
            "passed": control_win_fraction >= args.gate_min_control_win_fraction,
        },
        "equal_budget_control_paired_ci": {
            "value": control_ci["lower_95"],
            "minimum_exclusive": 0.0,
            "passed": control_ci["lower_95"] > 0.0,
        },
        "actual_target_free_repair_adjacency_delta": {
            "value": aggregate["repair_adjacency_delta"],
            "minimum": args.gate_min_repair_adjacency_delta,
            "passed": aggregate["repair_adjacency_delta"]
            >= args.gate_min_repair_adjacency_delta,
        },
    }
    return {
        "stage": stage,
        "epoch": epoch,
        "scope": "fixed synthetic whole-source evaluation; no test targets",
        "panel": "primary_kornia" if stage == "selection" else "independent_libjpeg",
        "source_count": len(names),
        "replicas_per_source": args.eval_replicas,
        "record_count": len(records),
        "candidate_policy": "positive first; identical raw corruption for all negatives",
        "random_baseline_policy": "deterministic iid Gaussian energies independent of labels",
        "classical_baseline": "raw RGB boundary L1 seam mean",
        "first_pass": "input-only raw RGB border-L1 w2 plus soft_cycle_component_solver",
        "pairwise_brier_note": "diagnostic only; no separate calibration split or fitted temperature",
        "aggregate": aggregate,
        "gates": gates,
        "gate_passed": all(value["passed"] for value in gates.values()),
        "per_source": records,
    }


def _epoch_indices(count: int, runtime: Runtime, seed: int) -> list[int]:
    order = np.random.default_rng(seed).permutation(count).tolist()
    padded = int(math.ceil(count / runtime.world_size) * runtime.world_size)
    if padded > count:
        order.extend(order[: padded - count])
    return [int(value) for value in order[runtime.rank : padded : runtime.world_size]]


def _cpu_state(model: LayoutEnergyTransformer) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _validate_args(args: argparse.Namespace) -> None:
    positive_counts = {
        "train_sources": args.train_sources,
        "selection_sources": args.selection_sources,
        "holdout_sources": args.holdout_sources,
        "epochs": args.epochs,
        "negatives_per_source": args.negatives_per_source,
        "eval_negatives": args.eval_negatives,
        "eval_replicas": args.eval_replicas,
        "smoke_steps": args.smoke_steps,
        "repair_steps": args.repair_steps,
        "repair_beam_width": args.repair_beam_width,
        "repair_hot_positions": args.repair_hot_positions,
        "repair_proposals": args.repair_proposals,
        "score_batch_size": args.score_batch_size,
        "amp_growth_interval": args.amp_growth_interval,
        "max_consecutive_amp_skips": args.max_consecutive_amp_skips,
        "max_total_amp_skips": args.max_total_amp_skips,
    }
    if any(type(value) is not int or value <= 0 for value in positive_counts.values()):
        raise ValueError(f"positive integer counts required: {positive_counts}")
    if min(args.train_offset, args.selection_offset) < 0:
        raise ValueError("source offsets must be non-negative")
    if args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs:
        raise ValueError("stop-after-epoch must be 0 or within configured epochs")
    positive_scalars = {
        "learning_rate": args.learning_rate,
        "grad_clip": args.grad_clip,
        "ranking_margin": args.ranking_margin,
        "gate_min_ranking_accuracy": args.gate_min_ranking_accuracy,
        "gate_min_delta_vs_classical": args.gate_min_delta_vs_classical,
        "gate_min_delta_vs_random": args.gate_min_delta_vs_random,
        "gate_min_energy_margin": args.gate_min_energy_margin,
        "gate_min_local_auc": args.gate_min_local_auc,
        "gate_min_graded_monotonic_accuracy": args.gate_min_graded_monotonic_accuracy,
        "gate_min_relative_repair_error_reduction": (
            args.gate_min_relative_repair_error_reduction
        ),
        "gate_min_repair_adjacency_delta": args.gate_min_repair_adjacency_delta,
        "gate_min_control_win_fraction": args.gate_min_control_win_fraction,
        "amp_init_scale": args.amp_init_scale,
    }
    if any(not math.isfinite(value) or value <= 0 for value in positive_scalars.values()):
        raise ValueError(f"positive finite scalar arguments required: {positive_scalars}")
    for name in (
        "gate_min_ranking_accuracy",
        "gate_min_local_auc",
        "gate_min_graded_monotonic_accuracy",
        "gate_min_relative_repair_error_reduction",
        "gate_min_control_win_fraction",
    ):
        if getattr(args, name) > 1.0:
            raise ValueError(f"{name} cannot exceed 1")
    theoretical_worst_case = min(
        1.0,
        args.repair_steps
        * 2
        * max(1, args.repair_hot_positions // 2)
        / TILE_COUNT,
    )
    if args.gate_min_relative_repair_error_reduction > theoretical_worst_case:
        raise ValueError(
            "relative repair gate exceeds the theoretical batched-swap maximum "
            f"for a fully wrong layout: gate={args.gate_min_relative_repair_error_reduction} "
            f"maximum={theoretical_worst_case}"
        )
    if args.repair_steps < 6:
        raise ValueError("review protocol requires at least 6 repair steps")
    if args.max_total_amp_skips < args.max_consecutive_amp_skips:
        raise ValueError("max-total-amp-skips must be >= max-consecutive-amp-skips")
    for name in (
        "ranking_weight",
        "listwise_weight",
        "local_error_weight",
        "move_weight",
        "move_matching_weight",
        "graded_monotonic_weight",
        "energy_regularization",
        "weight_decay",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")


def _save_artifacts(
    output_dir: Path,
    *,
    checkpoint: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, str]:
    checkpoint["model_state_sha256"] = _state_sha256(checkpoint["model_state"])
    checkpoint_path = output_dir / CHECKPOINT_NAME
    report_path = output_dir / REPORT_NAME
    _atomic_torch(checkpoint_path, checkpoint)
    report["checkpoint"] = {
        "path": checkpoint_path.name,
        "sha256": _sha256(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "model_state_sha256": checkpoint["model_state_sha256"],
    }
    _atomic_json(report_path, report)
    hash_inputs = [checkpoint_path, report_path]
    resume_path = output_dir / RESUME_NAME
    if resume_path.is_file():
        hash_inputs.append(resume_path)
    hashes = _write_hashes(output_dir, hash_inputs)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "hashes": str(hashes),
    }


def _run_smoke(args: argparse.Namespace, runtime: Runtime, output_dir: Path) -> None:
    if runtime.world_size != 1:
        raise RuntimeError("small-grid smoke is single-process")
    config = LayoutEnergyConfig(
        grid_size=4,
        tile_size=8,
        d_model=48,
        num_heads=4,
        local_layers=2,
        window_size=2,
        global_layers=1,
        global_tokens=2,
        feedforward_dim=96,
        cnn_channels=12,
        edge_dim=8,
        edge_band=2,
        move_dim=12,
        dropout=0.0,
    )
    model = LayoutEnergyTransformer(config).to(runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(args.seed)
    base = torch.rand(1, config.tile_count, 3, config.tile_size, config.tile_size, generator=generator).to(runtime.device)
    rng = np.random.default_rng(args.seed)
    negative = make_negative_layout(grid_size=4, family="block_swap", rng=rng, severity=0.2)
    layouts = np.stack([np.arange(16, dtype=np.int32), negative.position_to_tile])
    errors = np.stack([np.zeros(16, np.float32), negative.error_mask])
    moves = np.stack([np.zeros((16, 2), np.float32), negative.move_targets])
    history: list[dict[str, float]] = []
    for _ in range(args.smoke_steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(base, torch.from_numpy(layouts).to(runtime.device).long().unsqueeze(0))
        losses = layout_energy_losses(
            output,
            torch.from_numpy(errors).to(runtime.device),
            torch.from_numpy(moves).to(runtime.device),
            candidates_per_source=2,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({name: float(value.detach().cpu()) for name, value in losses.items()})
    report = {
        "schema_version": 1,
        "kind": "layout_energy_small_grid_smoke",
        "status": "smoke_passed",
        "safe_for_submission": False,
        "development_gate_eligible": False,
        "model_config": config.to_dict(),
        "loss_history": history,
        "source_code": _code_provenance(),
        "runtime": {"device": str(runtime.device), "torch": torch.__version__},
    }
    checkpoint = {
        "schema_version": 1,
        "kind": "layout_energy_smoke_checkpoint",
        "safe_for_submission": False,
        "model_config": config.to_dict(),
        "model_state": _cpu_state(model),
        "optimizer_state": _to_cpu_tree(optimizer.state_dict()),
        "scaler_state": {},
        "rng_state": [_rng_state()],
        "selected_epoch": None,
        "source_code": report["source_code"],
    }
    artifacts = _save_artifacts(output_dir, checkpoint=checkpoint, report=report)
    _print(runtime, {"event": "layout_energy_smoke_complete", **artifacts})


def _run_pilot(args: argparse.Namespace, runtime: Runtime, output_dir: Path) -> None:
    # Filesystem-heavy preflight and hashing are primary-only after DDP init.
    preflight: dict[str, Any] | None = None
    if runtime.primary:
        train_names, selection_names, holdout_names, split_audit = _split_sources(args)
        data_root = Path(args.data_root)
        data_provenance = {
            "manifest_path": args.manifest,
            "manifest_sha256": _sha256(args.manifest),
            "quarantine_path": args.quarantine,
            "quarantine_sha256": _sha256(args.quarantine),
            "train_targets_sha256": _named_file_set_sha256(
                [data_root / "train" / "targets" / name for name in train_names], data_root
            ),
            "selection_targets_sha256": _named_file_set_sha256(
                [data_root / "train" / "targets" / name for name in selection_names], data_root
            ),
            "holdout_targets_sha256": _named_file_set_sha256(
                [data_root / "train" / "targets" / name for name in holdout_names], data_root
            ),
        }
        source_code = _code_provenance()
        resume_contract = _resume_contract(
            args,
            split_audit=split_audit,
            data_provenance=data_provenance,
            source_code=source_code,
        )
        preflight = {
            "train_names": train_names,
            "selection_names": selection_names,
            "holdout_names": holdout_names,
            "split_audit": split_audit,
            "data_provenance": data_provenance,
            "source_code": source_code,
            "resume_contract": resume_contract,
        }
    if runtime.distributed:
        payload = [preflight]
        torch.distributed.broadcast_object_list(payload, src=0)
        preflight = payload[0]
    if preflight is None:
        raise RuntimeError("primary preflight did not broadcast")
    train_names = preflight["train_names"]
    selection_names = preflight["selection_names"]
    holdout_names = preflight["holdout_names"]
    split_audit = preflight["split_audit"]
    data_provenance = preflight["data_provenance"]
    source_code = preflight["source_code"]
    resume_contract = preflight["resume_contract"]

    config = _model_config(args)
    config.validate()
    amp_enabled, amp_dtype, amp_label = _amp(args, runtime)
    local_hardware = _rank_hardware_probe(runtime, require_t4=args.require_t4)
    hardware_by_rank = _gather_objects(runtime, local_hardware)
    model = LayoutEnergyTransformer(config).to(runtime.device)
    resume_payload: dict[str, Any] | None = None
    if args.resume_checkpoint:
        resume_payload = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if resume_payload.get("kind") != "layout_energy_epoch_boundary_resume":
            raise ValueError("resume checkpoint has wrong kind")
        saved_contract = resume_payload.get("resume_contract", {})
        if saved_contract.get("contract_sha256") != resume_contract["contract_sha256"]:
            raise ValueError("resume contract mismatch for args/split/code/data hashes")
        model.load_state_dict(resume_payload["current_model_state"])
    parameter_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    wrapped: nn.Module
    if runtime.distributed:
        wrapped = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            find_unused_parameters=False,
        )
    else:
        wrapped = model
    optimizer = torch.optim.AdamW(
        wrapped.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
        init_scale=args.amp_init_scale,
        growth_interval=args.amp_growth_interval,
    )
    start_epoch = 0
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer: dict[str, Any] | None = None
    best_scaler: dict[str, Any] | None = None
    best_rng: list[dict[str, Any]] | None = None
    best_selection: dict[str, Any] | None = None
    best_epoch: int | None = None
    total_amp_skips = 0
    consecutive_amp_skips = 0
    amp_skip_events: list[dict[str, Any]] = []
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["current_optimizer_state"])
        scaler.load_state_dict(resume_payload["current_scaler_state"])
        start_epoch = int(resume_payload["next_epoch"])
        if not 0 <= start_epoch < args.epochs:
            raise ValueError(f"resume next_epoch {start_epoch} is outside configured epochs")
        rank_states = resume_payload["rng_state_per_rank"]
        if len(rank_states) != runtime.world_size:
            raise ValueError("resume RNG rank count differs from current world size")
        _restore_rng_state(rank_states[runtime.rank])
        history = copy.deepcopy(resume_payload.get("history", []))
        best_score = float(resume_payload.get("best_score", -float("inf")))
        best_state = resume_payload.get("best_model_state")
        best_optimizer = resume_payload.get("best_optimizer_state")
        best_scaler = resume_payload.get("best_scaler_state")
        best_rng = resume_payload.get("best_rng_state_per_rank")
        best_selection = resume_payload.get("best_selection")
        best_epoch = resume_payload.get("best_epoch")
        total_amp_skips = int(resume_payload.get("total_amp_skips", 0))
        consecutive_amp_skips = int(resume_payload.get("consecutive_amp_skips", 0))
        amp_skip_events = copy.deepcopy(resume_payload.get("amp_skip_events", []))
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    _print(
        runtime,
        {
            "event": "layout_energy_pilot_start",
            "world_size": runtime.world_size,
            "device": str(runtime.device),
            "amp": amp_label,
            "parameters": parameter_count,
            "train_sources": len(train_names),
            "selection_sources": len(selection_names),
            "holdout_sources": len(holdout_names),
            "candidates_per_source": args.negatives_per_source + 1,
            "start_epoch": start_epoch,
            "resume_contract_sha256": resume_contract["contract_sha256"],
            "amp_init_scale": args.amp_init_scale,
        },
    )
    started = time.perf_counter()
    epoch_telemetry: list[dict[str, Any]] = [
        {"epoch": record.get("epoch"), "ranks": record.get("rank_telemetry")}
        for record in history
    ]
    end_epoch = args.epochs if args.stop_after_epoch == 0 else args.stop_after_epoch
    if end_epoch <= start_epoch:
        raise ValueError(
            f"stop-after-epoch={end_epoch} does not advance resume start_epoch={start_epoch}"
        )
    for epoch in range(start_epoch, end_epoch):
        epoch_started = time.perf_counter()
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        wrapped.train()
        indices = _epoch_indices(
            len(train_names),
            runtime,
            per_source_seed(args.seed, "layout-energy-epoch-order", str(epoch)),
        )
        local_sums = np.zeros(9, dtype=np.float64)
        families, severity = curriculum_families(epoch, args.epochs)
        for step, source_index in enumerate(indices):
            name = train_names[source_index]
            panel = _panel_for_source(name, args=args, epoch=epoch, stage="train")
            layouts, errors, moves, severities, labels, _ = _candidate_set(
                panel,
                source_name=name,
                args=args,
                epoch=epoch,
                stage="train",
                count=args.negatives_per_source,
            )
            base = _tile_tensor(panel.raw_tiles, runtime.device)
            layout_tensor = torch.from_numpy(layouts).to(runtime.device).long().unsqueeze(0)
            error_tensor = torch.from_numpy(errors).to(runtime.device)
            move_tensor = torch.from_numpy(moves).to(runtime.device)
            severity_tensor = torch.from_numpy(severities).to(runtime.device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(runtime, amp_enabled, amp_dtype):
                output = wrapped(base, candidate_layouts=layout_tensor)
                losses = layout_energy_losses(
                    output,
                    error_tensor,
                    move_tensor,
                    candidates_per_source=args.negatives_per_source + 1,
                    severity_targets=severity_tensor,
                    ranking_margin=args.ranking_margin,
                    ranking_weight=args.ranking_weight,
                    listwise_weight=args.listwise_weight,
                    local_error_weight=args.local_error_weight,
                    move_weight=args.move_weight,
                    move_matching_weight=args.move_matching_weight,
                    graded_monotonic_weight=args.graded_monotonic_weight,
                    energy_regularization=args.energy_regularization,
                )
            loss_finite = _synchronized_all_finite(
                runtime, bool(torch.isfinite(losses["total"]).item())
            )
            if not loss_finite:
                raise RuntimeError(f"synchronized non-finite loss at epoch={epoch} step={step}")
            optimized = 1.0
            if scaler.is_enabled():
                scale_before = float(scaler.get_scale())
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), args.grad_clip)
                gradients_finite = _synchronized_all_finite(
                    runtime, bool(torch.isfinite(norm).item())
                )
                if gradients_finite:
                    scaler.step(optimizer)
                    scaler.update()
                    consecutive_amp_skips = 0
                else:
                    optimized = 0.0
                    optimizer.zero_grad(set_to_none=True)
                    new_scale, consecutive_amp_skips, total_amp_skips = (
                        _bounded_amp_skip_update(
                            scale_before=scale_before,
                            consecutive_skips=consecutive_amp_skips,
                            total_skips=total_amp_skips,
                            max_consecutive=args.max_consecutive_amp_skips,
                            max_total=args.max_total_amp_skips,
                        )
                    )
                    scaler.update(new_scale=new_scale)
                    if runtime.primary:
                        amp_skip_events.append(
                            {
                                "epoch": epoch + 1,
                                "step": step + 1,
                                "scale_before": scale_before,
                                "scale_after": new_scale,
                                "reason": "DDP-synchronized non-finite gradient norm",
                            }
                        )
            else:
                losses["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), args.grad_clip)
                if not _synchronized_all_finite(runtime, bool(torch.isfinite(norm).item())):
                    raise RuntimeError("synchronized non-finite FP32/BF16 gradient norm")
                optimizer.step()
            local_sums += np.asarray(
                [
                    float(losses["total"].detach().cpu()),
                    float(losses["ranking"].detach().cpu()),
                    float(losses["listwise"].detach().cpu()),
                    float(losses["local_error"].detach().cpu()),
                    float(losses["move"].detach().cpu()),
                    float(losses["move_matching"].detach().cpu()),
                    float(losses["graded_monotonic"].detach().cpu()),
                    1.0,
                    optimized,
                ]
            )
            if runtime.primary and step % 8 == 0:
                _print(
                    runtime,
                    {
                        "event": "layout_energy_train_step",
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "local_steps": len(indices),
                        "source": name,
                        "negative_families": labels[1:],
                        "total_loss": float(losses["total"].detach().cpu()),
                    },
                )
        reduced = torch.tensor(local_sums, device=runtime.device, dtype=torch.float64)
        if runtime.distributed:
            torch.distributed.all_reduce(reduced)
        totals = reduced.cpu().numpy()
        rank_rng = _gather_rng(runtime)
        local_epoch_telemetry = {
            "rank": runtime.rank,
            "epoch": epoch + 1,
            "seconds": time.perf_counter() - epoch_started,
            "amp_scale": float(scaler.get_scale()) if scaler.is_enabled() else None,
            "total_amp_skips": total_amp_skips,
            "consecutive_amp_skips": consecutive_amp_skips,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(runtime.device))
            if runtime.device.type == "cuda"
            else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(runtime.device))
            if runtime.device.type == "cuda"
            else None,
        }
        gathered_epoch_telemetry = _gather_objects(runtime, local_epoch_telemetry)
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "curriculum_families": families,
            "curriculum_severity": severity,
            "train": {
                "attempted_steps_all_ranks": int(totals[7]),
                "optimized_steps_all_ranks": int(totals[8]),
                "total_loss": float(totals[0] / totals[7]),
                "ranking_loss": float(totals[1] / totals[7]),
                "listwise_loss": float(totals[2] / totals[7]),
                "local_error_loss": float(totals[3] / totals[7]),
                "move_loss": float(totals[4] / totals[7]),
                "move_matching_loss": float(totals[5] / totals[7]),
                "graded_monotonic_loss": float(totals[6] / totals[7]),
            },
            "rank_telemetry": gathered_epoch_telemetry,
        }
        selection = _evaluate(
            model,
            selection_names,
            args=args,
            runtime=runtime,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            epoch=epoch + 1,
            stage="selection",
        )
        is_best = False
        if runtime.primary:
            assert selection is not None
            record["selection"] = selection
            aggregate = selection["aggregate"]
            score = (
                aggregate["learned_ranking_accuracy"]
                + 0.35 * aggregate["pooled_local_error_auc"]
                + 0.50 * aggregate["repair_relative_error_reduction"]
                + 0.25 * aggregate["repair_adjacency_delta"]
            )
            if score > best_score:
                best_score = score
                best_selection = selection
                is_best = True
            _print(
                runtime,
                {
                    "event": "layout_energy_selection_epoch",
                    "epoch": epoch + 1,
                    "selection_score": score,
                    "ranking_accuracy": aggregate["learned_ranking_accuracy"],
                    "local_auc": aggregate["pooled_local_error_auc"],
                    "repair_relative_error_reduction": aggregate[
                        "repair_relative_error_reduction"
                    ],
                    "gate_passed": selection["gate_passed"],
                },
            )
        best_flag = torch.tensor(
            1 if is_best else 0, device=runtime.device, dtype=torch.int32
        )
        if runtime.distributed:
            torch.distributed.broadcast(best_flag, src=0)
        if bool(best_flag.item()):
            best_state = _cpu_state(model)
            best_optimizer = _to_cpu_tree(optimizer.state_dict())
            best_scaler = _to_cpu_tree(scaler.state_dict())
            best_epoch = epoch + 1
            if runtime.primary:
                best_rng = rank_rng
        if runtime.primary:
            history.append(record)
            epoch_telemetry.append(
                {"epoch": epoch + 1, "ranks": gathered_epoch_telemetry}
            )
            resume_payload_to_save = {
                "schema_version": 1,
                "kind": "layout_energy_epoch_boundary_resume",
                "safe_for_submission": False,
                "resume_contract": resume_contract,
                "next_epoch": epoch + 1,
                "current_model_state": _cpu_state(model),
                "current_optimizer_state": _to_cpu_tree(optimizer.state_dict()),
                "current_scaler_state": _to_cpu_tree(scaler.state_dict()),
                "rng_state_per_rank": rank_rng,
                "history": history,
                "best_score": best_score,
                "best_model_state": best_state,
                "best_optimizer_state": best_optimizer,
                "best_scaler_state": best_scaler,
                "best_rng_state_per_rank": best_rng,
                "best_selection": best_selection,
                "best_epoch": best_epoch,
                "total_amp_skips": total_amp_skips,
                "consecutive_amp_skips": consecutive_amp_skips,
                "amp_skip_events": amp_skip_events,
            }
            _atomic_torch(output_dir / RESUME_NAME, resume_payload_to_save)
        _barrier(runtime)

    if end_epoch < args.epochs:
        if runtime.primary:
            resume_path = output_dir / RESUME_NAME
            pause_report = {
                "schema_version": 1,
                "kind": "raw_layout_energy_epoch_boundary_pause",
                "status": "paused_for_resume",
                "safe_for_submission": False,
                "completed_epochs": end_epoch,
                "configured_epochs": args.epochs,
                "resume_checkpoint": resume_path.name,
                "resume_checkpoint_sha256": _sha256(resume_path),
                "resume_contract": resume_contract,
                "amp_skip_events": amp_skip_events,
                "hardware_preflight_by_rank": hardware_by_rank,
                "history": history,
            }
            report_path = output_dir / REPORT_NAME
            _atomic_json(report_path, pause_report)
            _write_hashes(output_dir, [report_path, resume_path])
            _print(
                runtime,
                {
                    "event": "layout_energy_paused_for_resume",
                    "completed_epochs": end_epoch,
                    "resume_checkpoint": str(resume_path),
                    "resume_checkpoint_sha256": _sha256(resume_path),
                },
            )
        _barrier(runtime)
        return

    if best_state is None or best_epoch is None:
        raise RuntimeError("no selection checkpoint was captured")
    model.load_state_dict(best_state)
    holdout = _evaluate(
        model,
        holdout_names,
        args=args,
        runtime=runtime,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        epoch=best_epoch,
        stage="holdout",
    )
    final_rank_telemetry = _gather_objects(
        runtime,
        {
            "rank": runtime.rank,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(runtime.device))
            if runtime.device.type == "cuda"
            else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(runtime.device))
            if runtime.device.type == "cuda"
            else None,
            "final_amp_scale": float(scaler.get_scale()) if scaler.is_enabled() else None,
            "total_amp_skips": total_amp_skips,
        },
    )
    if not runtime.primary:
        _barrier(runtime)
        return
    assert (
        best_selection is not None
        and holdout is not None
        and best_optimizer is not None
        and best_scaler is not None
        and best_rng is not None
    )
    _barrier(runtime)
    selection_passed = bool(best_selection["gate_passed"])
    holdout_passed = bool(holdout["gate_passed"])
    gate_passed = selection_passed and holdout_passed
    status = "holdout_gate_passed" if gate_passed else "holdout_gate_failed"
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(runtime.device))
        if runtime.device.type == "cuda"
        else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(runtime.device))
        if runtime.device.type == "cuda"
        else None
    )
    runtime_report = {
        "world_size": runtime.world_size,
        "device": str(runtime.device),
        "amp": amp_label,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
        "peak_cuda_memory_allocated_bytes_rank0": peak_allocated,
        "peak_cuda_memory_reserved_bytes_rank0": peak_reserved,
        "candidate_tile_encoding_reuse": True,
        "dense_global_tile_attention": False,
        "hardware_preflight_by_rank": hardware_by_rank,
        "epoch_telemetry": epoch_telemetry,
        "final_telemetry_by_rank": final_rank_telemetry,
        "amp_skip_audit": {
            "init_scale": args.amp_init_scale,
            "growth_interval": args.amp_growth_interval,
            "max_consecutive_skips": args.max_consecutive_amp_skips,
            "max_total_skips": args.max_total_amp_skips,
            "total_skips": total_amp_skips,
            "events": amp_skip_events,
            "ddp_synchronized": True,
        },
        "analytical_memory_lower_bounds_bytes": {
            "fp32_parameters": int(parameter_count * 4),
            "fp16_candidate_tokens_per_rank": int(
                (args.negatives_per_source + 1) * TILE_COUNT * config.d_model * 2
            ),
            "fp32_single_raw_tile_set": int(TILE_COUNT * 3 * TILE * TILE * 4),
            "fp16_window_attention_logits_one_layer": int(
                (args.negatives_per_source + 1)
                * (GRID // config.window_size) ** 2
                * config.num_heads
                * (config.window_size**2) ** 2
                * 2
            ),
        },
        "memory_note": "lower bounds exclude gradients, optimizer states, saved activations, allocator workspace, and DDP buffers",
    }
    report = {
        "schema_version": 1,
        "kind": "raw_layout_energy_transformer_bounded_pilot",
        "status": status,
        "safe_for_submission": False,
        "promotion_policy": (
            "selection and independent holdout may approve only a later fixed real-layout "
            "target-free refinement gate; never direct submission"
        ),
        "seed": args.seed,
        "selected_epoch": best_epoch,
        "elapsed_seconds": time.perf_counter() - started,
        "model_config": config.to_dict(),
        "trainable_parameters": parameter_count,
        "training_config": {
            key: value for key, value in vars(args).items() if key not in {"output_dir", "overwrite"}
        },
        "training_design": {
            "raw_only": True,
            "denoiser_used": False,
            "candidate_tile_encoding_reused": True,
            "positive_negative_corruption_realization_shared": True,
            "corruption_axes": ["brightness", "contrast", "noise", "blur", "JPEG"],
            "negative_families": list(NEGATIVE_FAMILIES),
            "curriculum": "genuine raw-seam first-pass residual chains plus large-to-sparse synthetic errors",
            "genuine_first_pass": "raw RGB border-L1 w2 plus soft_cycle_component_solver",
        },
        "split_audit": split_audit,
        "data_provenance": data_provenance,
        "source_code": source_code,
        "runtime": runtime_report,
        "epoch_history": history,
        "selected_selection": best_selection,
        "independent_holdout": holdout,
        "selection_gate_passed": selection_passed,
        "holdout_gate_passed": holdout_passed,
        "checkpoint_resume_state": {
            "optimizer": True,
            "scaler": True,
            "rng_every_rank": True,
            "selected_epoch_bound": True,
            "actual_epoch_boundary_resume_path": str(output_dir / RESUME_NAME),
            "resume_contract_sha256": resume_contract["contract_sha256"],
        },
        "limitations": [
            "trained and gated on synthetic exact layouts, not hidden test labels",
            "independent per-tile corruption may still differ from real competition corruption",
            "energy-guided swaps are local and cannot guarantee escape from large layout basins",
            "raw-seam first-pass is genuine input-only but does not validate HBT/QAP error distribution",
            "a frozen real16 HBT/QAP residual-repair gate remains mandatory",
            "a second training seed remains mandatory before promotion",
            "checkpoint is always safe_for_submission=false",
        ],
    }
    checkpoint = {
        "schema_version": 1,
        "kind": "raw_layout_energy_transformer_checkpoint",
        "status": status,
        "safe_for_submission": False,
        "development_gate_passed": gate_passed,
        "selection_gate_passed": selection_passed,
        "holdout_gate_passed": holdout_passed,
        "selected_epoch": best_epoch,
        "model_config": config.to_dict(),
        "model_state": best_state,
        "optimizer_state": best_optimizer,
        "scaler_state": best_scaler,
        "rng_state_per_rank": best_rng,
        "split_audit": split_audit,
        "data_provenance": data_provenance,
        "source_code": source_code,
        "resume_contract": resume_contract,
        "selected_selection": best_selection,
        "independent_holdout": holdout,
        "seed": args.seed,
    }
    resume_path = output_dir / RESUME_NAME
    report["epoch_boundary_resume"] = {
        "path": resume_path.name,
        "sha256": _sha256(resume_path),
        "bytes": resume_path.stat().st_size,
    }
    artifacts = _save_artifacts(output_dir, checkpoint=checkpoint, report=report)
    _print(runtime, {"event": "layout_energy_pilot_complete", "status": status, **artifacts})


def _failure_report(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    runtime: Runtime | None,
    error: BaseException,
) -> None:
    path = output_dir / REPORT_NAME
    if path.exists():
        return
    payload = {
        "schema_version": 1,
        "kind": "layout_energy_failure_report",
        "status": "failed",
        "safe_for_submission": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "seed": getattr(args, "seed", None),
        "runtime": None
        if runtime is None
        else {"rank": runtime.rank, "world_size": runtime.world_size, "device": str(runtime.device)},
        "source_code": {
            "model_sha256": _sha256(MODEL_SOURCE) if MODEL_SOURCE.is_file() else None,
            "script_sha256": _sha256(SCRIPT_SOURCE),
        },
    }
    _atomic_json(path, payload)
    _write_hashes(output_dir, [path])


def main() -> None:
    args = parse_args()
    runtime: Runtime | None = None
    output_dir = Path(args.output_dir)
    try:
        _validate_args(args)
        runtime = _init_runtime(args.device)
        if runtime.primary:
            output_dir = _preflight_output(args)
        _barrier(runtime)
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
                _failure_report(output_dir, args=args, runtime=runtime, error=error)
            except Exception as report_error:
                print(f"failed to write failure report: {report_error}", file=sys.stderr)
        raise
    finally:
        _cleanup(runtime)


if __name__ == "__main__":
    main()
