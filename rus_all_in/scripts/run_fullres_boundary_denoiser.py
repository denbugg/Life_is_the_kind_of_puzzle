#!/usr/bin/env python3
"""Train and locally gate a no-downsampling 20x20 matcher-view denoiser."""

from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.fullres_boundary_denoiser import (
    FullResolutionBoundaryDenoiser,
    FullResolutionDenoiserConfig,
    boundary_denoising_loss,
    model_config_dict,
    restore_matcher_view,
)
from aiijc_puzzle.protocol import IMAGE_SIZE, compute_protocol_digest, sha256_file, split_tiles
from aiijc_puzzle.restoration_r6 import distort_tiles
from aiijc_puzzle.restored_border_ranker import restored_descriptor_scores
from aiijc_puzzle.socket_pasha_matched import row_rank_percentiles
from aiijc_puzzle.socket_sorter_production import LoadedSocketCheckpoint, load_socket_checkpoint
from aiijc_puzzle.synthetic_socket_evaluation import (
    ExactSyntheticReference,
    SyntheticSocketInput,
    exact_local_retrieval_metrics,
    freeze_topk_candidates,
    load_checkpoint_with_lineage,
    make_exact_synthetic_case,
    names_digest,
    select_source_disjoint_train_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_SOCKET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
SELECTION_NAMESPACE = "aiijc-fullres-boundary-denoiser-v1"
MAX_TRAIN_SOURCES = 192
MAX_EVAL_SOURCES = 24
MAX_STEPS = 400
GRID = 24
COUNT = GRID * GRID
LOCAL_KS = (1, 5, 32)
TOP_K_SUPPLY = 32
DISCOVERY_DIRECTIONAL_SUPPLY_GAIN = 0.01
DISCOVERY_OTHER_DIRECTION_MAX_LOSS = 0.005
DISCOVERY_R1_DELTA = 0.0025
DISCOVERY_R5_DELTA = 0.0
DISCOVERY_PRECISION_DELTA = 0.01
STRONG_SUPPLY_DELTA = 0.01
STRONG_R1_DELTA = 0.01
STRONG_R5_DELTA = 0.0
STRONG_PRECISION_DELTA = 0.03
MIN_RECIPROCAL_COVERAGE = 0.03
PRIMARY_CANDIDATES = (
    "restored_d64_ot",
    "raw_restored_d64_rank50",
    "raw_restored_descriptor_rank50",
)


@dataclass(frozen=True)
class CleanBoard:
    filename: str
    target_sha256: str
    tiles: np.ndarray


@dataclass(frozen=True)
class TrainBatchSpec:
    source_index: int
    tile_indices: np.ndarray
    corruption_seed: int


@dataclass(frozen=True)
class FrozenPrediction:
    case_id: str
    source_filename: str
    candidates: dict[str, dict[str, np.ndarray]]
    reciprocal: dict[str, dict[str, dict[str, np.ndarray]]]
    supply: dict[str, dict[str, np.ndarray]]
    runtime_seconds: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--socket-checkpoint", type=Path, default=DEFAULT_SOCKET_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sources", type=int, default=192)
    parser.add_argument("--eval-sources", type=int, default=24)
    parser.add_argument("--terminal-sources", type=int, default=24)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--tile-batch", type=int, default=256)
    parser.add_argument("--inference-batch", type=int, default=576)
    parser.add_argument("--prefetch-workers", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--border-width", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--benchmark-batch", type=int, default=64)
    parser.add_argument("--benchmark-repeats", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--exclude-report", type=Path, action="append", default=[])
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.train_sources <= MAX_TRAIN_SOURCES:
        raise ValueError(f"train-sources must be in [1, {MAX_TRAIN_SOURCES}]")
    if not 1 <= args.eval_sources <= MAX_EVAL_SOURCES:
        raise ValueError(f"eval-sources must be in [1, {MAX_EVAL_SOURCES}]")
    if not 1 <= args.terminal_sources <= MAX_EVAL_SOURCES:
        raise ValueError(f"terminal-sources must be in [1, {MAX_EVAL_SOURCES}]")
    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    positive = (
        args.tile_batch,
        args.inference_batch,
        args.prefetch_workers,
        args.width,
        args.blocks,
        args.border_width,
        args.benchmark_batch,
        args.benchmark_repeats,
        args.log_every,
    )
    if any(isinstance(value, bool) or value <= 0 for value in positive):
        raise ValueError("batch, architecture, prefetch and logging values must be positive")
    if args.tile_batch > COUNT or args.benchmark_batch > COUNT:
        raise ValueError("tile and benchmark batches cannot exceed 576")
    if args.prefetch_workers > 4:
        raise ValueError("prefetch-workers cannot exceed four")
    if not 1 <= args.border_width <= 10:
        raise ValueError("border-width must be in [1, 10]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be finite and non-negative")
    if args.allow_nondeterministic_mps and args.device == "cpu":
        raise ValueError("allow-nondeterministic-mps is incompatible with --device cpu")
    if not args.exclude_report:
        raise ValueError("at least one prior report is required for recursive source exclusion")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def collect_declared_filenames(value: Any, *, parent_key: str = "") -> set[str]:
    """Collect every recursively declared ``*_filename(s)`` PNG lineage field."""

    names: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            names.update(collect_declared_filenames(child, parent_key=str(key)))
    elif isinstance(value, list):
        if parent_key.endswith("filenames") and all(isinstance(item, str) for item in value):
            names.update(Path(item).name for item in value if item.endswith(".png"))
        else:
            for child in value:
                names.update(collect_declared_filenames(child, parent_key=parent_key))
    elif isinstance(value, str) and parent_key.endswith("filename") and value.endswith(".png"):
        names.add(Path(value).name)
    return names


def _exclusion_lineage(
    reports: list[Path],
    socket_names: tuple[str, ...],
) -> tuple[set[str], list[dict[str, Any]]]:
    excluded = set(socket_names)
    records: list[dict[str, Any]] = []
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        found = collect_declared_filenames(payload)
        if not found:
            raise ValueError(f"exclude report contains no declared filenames: {path}")
        excluded.update(found)
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "declared_filename_count": len(found),
            }
        )
    return excluded, records


def validate_source_split(
    train_names: tuple[str, ...],
    eval_names: tuple[str, ...],
    terminal_names: tuple[str, ...],
    excluded: set[str],
) -> None:
    """Fail closed unless fit, local evaluation and unopened terminal pools are disjoint."""

    groups = (set(train_names), set(eval_names), set(terminal_names))
    if any(len(group) != len(names) for group, names in zip(
        groups,
        (train_names, eval_names, terminal_names),
        strict=True,
    )):
        raise ValueError("source split contains duplicate filenames")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("train, local evaluation and terminal pools must be disjoint")
    if set.union(*groups) & excluded:
        raise ValueError("source split overlaps an excluded checkpoint/report lineage")


def _prepare_boards(records: tuple[Any, ...], targets: Path) -> list[CleanBoard]:
    boards: list[CleanBoard] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        path = targets / filename
        observed = sha256_file(path)
        if observed != record.get("target_sha256"):
            raise ValueError(f"manifest target hash mismatch: {filename}")
        boards.append(CleanBoard(filename, observed, split_tiles(_load_rgb(path)).copy()))
        if index == 1 or index % 64 == 0 or index == len(records):
            print(f"prepared source {index}/{len(records)} {filename}", flush=True)
    return boards


def _tensor(tiles: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(tiles))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def _synchronise(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark_devices(
    clean_tiles: np.ndarray,
    dirty_tiles: np.ndarray,
    *,
    config: FullResolutionDenoiserConfig,
    border_width: int,
    repeats: int,
) -> dict[str, dict[str, float | int | str]]:
    """Measure identical forward/loss/backward steps on available CPU and MPS."""

    results: dict[str, dict[str, float | int | str]] = {}
    names = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for name in names:
        device = torch.device(name)
        model = FullResolutionBoundaryDenoiser(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        clean = _tensor(clean_tiles, device)
        dirty = _tensor(dirty_tiles, device)
        timings: list[float] = []
        for iteration in range(repeats + 1):
            started = perf_counter()
            prediction = model(dirty)
            loss, _ = boundary_denoising_loss(
                prediction,
                clean,
                dirty,
                border_width=border_width,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            _synchronise(device)
            if iteration:
                timings.append(perf_counter() - started)
        results[name] = {
            "mean_seconds_per_step": float(np.mean(timings)),
            "tiles_per_second": float(len(clean_tiles) / np.mean(timings)),
            "batch": len(clean_tiles),
            "repeats": repeats,
        }
        del model, optimizer, clean, dirty
        if device.type == "mps":
            torch.mps.empty_cache()
    return results


def choose_device(requested: str, benchmark: dict[str, dict[str, Any]]) -> torch.device:
    if requested == "auto":
        chosen = min(
            benchmark,
            key=lambda name: float(benchmark[name]["mean_seconds_per_step"]),
        )
    else:
        chosen = requested
    if chosen == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(chosen)


def _training_specs(args: argparse.Namespace, board_count: int) -> list[TrainBatchSpec]:
    generator = np.random.default_rng(args.seed + 11)
    specs: list[TrainBatchSpec] = []
    for step in range(args.steps):
        source_index = int(generator.integers(board_count))
        tile_indices = np.ascontiguousarray(
            generator.choice(COUNT, size=args.tile_batch, replace=False),
            dtype=np.int32,
        )
        specs.append(
            TrainBatchSpec(
                source_index,
                tile_indices,
                args.seed + 1_000_003 * (step + 1),
            )
        )
    return specs


def _materialise_train_batch(
    board: CleanBoard,
    spec: TrainBatchSpec,
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.ascontiguousarray(board.tiles[spec.tile_indices])
    dirty = distort_tiles(clean, np.random.default_rng(spec.corruption_seed))
    return dirty, clean


def train_model(
    model: FullResolutionBoundaryDenoiser,
    boards: list[CleanBoard],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.steps,
        eta_min=args.learning_rate * 0.05,
    )
    specs = _training_specs(args, len(boards))
    history: list[dict[str, Any]] = []
    started = perf_counter()
    prefetch_wait_seconds = 0.0
    model.train()
    with ThreadPoolExecutor(max_workers=args.prefetch_workers) as executor:
        futures: dict[int, Future[tuple[np.ndarray, np.ndarray]]] = {}
        submit_index = 0
        window = max(2, args.prefetch_workers * 2)
        while submit_index < min(args.steps, window):
            spec = specs[submit_index]
            futures[submit_index] = executor.submit(
                _materialise_train_batch,
                boards[spec.source_index],
                spec,
            )
            submit_index += 1
        for step, spec in enumerate(specs):
            wait_started = perf_counter()
            dirty_array, clean_array = futures.pop(step).result()
            prefetch_wait_seconds += perf_counter() - wait_started
            if submit_index < args.steps:
                next_spec = specs[submit_index]
                futures[submit_index] = executor.submit(
                    _materialise_train_batch,
                    boards[next_spec.source_index],
                    next_spec,
                )
                submit_index += 1

            dirty = _tensor(dirty_array, device)
            clean = _tensor(clean_array, device)
            prediction = model(dirty)
            loss, terms = boundary_denoising_loss(
                prediction,
                clean,
                dirty,
                border_width=args.border_width,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            scheduler.step()
            record: dict[str, Any] = {
                "step": step + 1,
                "source_filename": boards[spec.source_index].filename,
                "corruption_seed": spec.corruption_seed,
                "loss": float(loss.detach()),
                "border": float(terms["border"].detach()),
                "gradient": float(terms["gradient"].detach()),
                "shape": float(terms["shape"].detach()),
                "identity": float(terms["identity"].detach()),
                "grad_norm": grad_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(record)
            if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
                recent = history[-min(args.log_every, len(history)) :]
                print(
                    json.dumps(
                        {
                            "event": "train",
                            "step": step + 1,
                            "loss": float(np.mean([row["loss"] for row in recent])),
                            "border": float(np.mean([row["border"] for row in recent])),
                            "gradient": float(
                                np.mean([row["gradient"] for row in recent])
                            ),
                            "elapsed_seconds": perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    _synchronise(device)
    return history, {
        "training": perf_counter() - started,
        "prefetch_wait": prefetch_wait_seconds,
        "prefetch_workers": args.prefetch_workers,
    }


@torch.inference_mode()
def _socket_scores(
    socket: LoadedSocketCheckpoint,
    tiles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    output = socket.model(_tensor(tiles, device).unsqueeze(0), grid=GRID)
    normaliser = math.log(float(COUNT + GRID))
    return (
        np.ascontiguousarray(
            output.right_log_assignment[0, :COUNT, :COUNT].float().cpu().numpy()
            + normaliser,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            output.down_log_assignment[0, :COUNT, :COUNT].float().cpu().numpy()
            + normaliser,
            dtype=np.float32,
        ),
    )


def _rank_fusion(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        0.5 * row_rank_percentiles(first) + 0.5 * row_rank_percentiles(second),
        dtype=np.float32,
    )


def _reciprocal_evidence(scores: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(scores, dtype=np.float32).copy()
    count = len(value)
    if value.shape != (count, count) or not np.isfinite(value).all():
        raise ValueError("reciprocal scores must be one finite square matrix")
    value[np.arange(count), np.arange(count)] = -np.inf
    row_order = np.argsort(-value, axis=1, kind="stable")[:, :2]
    column_order = np.argsort(-value, axis=0, kind="stable")[:2]
    target = row_order[:, 0]
    row_margin = value[np.arange(count), row_order[:, 0]] - value[
        np.arange(count), row_order[:, 1]
    ]
    column_margin = value[column_order[0], np.arange(count)] - value[
        column_order[1], np.arange(count)
    ]
    reciprocal = column_order[0, target] == np.arange(count)
    confidence = np.minimum(row_margin, column_margin[target])
    return {
        "target": np.ascontiguousarray(target, dtype=np.int32),
        "reciprocal": np.ascontiguousarray(reciprocal),
        "confidence": np.ascontiguousarray(confidence, dtype=np.float32),
    }


@torch.inference_mode()
def freeze_prediction(
    model: FullResolutionBoundaryDenoiser,
    socket: LoadedSocketCheckpoint,
    synthetic_input: SyntheticSocketInput,
    *,
    device: torch.device,
    inference_batch: int,
) -> FrozenPrediction:
    raw_started = perf_counter()
    raw_right, raw_down = _socket_scores(socket, synthetic_input.tiles, device=device)
    raw_seconds = perf_counter() - raw_started
    restore_started = perf_counter()
    restored_tiles = restore_matcher_view(
        model,
        synthetic_input.tiles,
        device=device,
        batch_size=inference_batch,
    )
    restore_seconds = perf_counter() - restore_started
    restored_socket_started = perf_counter()
    restored_right, restored_down = _socket_scores(socket, restored_tiles, device=device)
    restored_socket_seconds = perf_counter() - restored_socket_started
    descriptor_started = perf_counter()
    descriptor_right = restored_descriptor_scores(restored_tiles, direction=0)
    descriptor_down = restored_descriptor_scores(restored_tiles, direction=1)
    descriptor_seconds = perf_counter() - descriptor_started
    variants = {
        "raw_d64_ot": (raw_right, raw_down),
        "restored_d64_ot": (restored_right, restored_down),
        "raw_restored_d64_rank50": (
            _rank_fusion(raw_right, restored_right),
            _rank_fusion(raw_down, restored_down),
        ),
        "raw_restored_descriptor_rank50": (
            _rank_fusion(raw_right, descriptor_right),
            _rank_fusion(raw_down, descriptor_down),
        ),
    }
    supply_scores = {
        "raw_d64_ot": (raw_right, raw_down),
        "restored_d64_ot": (restored_right, restored_down),
        "restored_descriptor": (descriptor_right, descriptor_down),
    }
    return FrozenPrediction(
        case_id=synthetic_input.case_id,
        source_filename=synthetic_input.source_filename,
        candidates={
            name: {
                "right": freeze_topk_candidates(right, max_k=max(LOCAL_KS)),
                "down": freeze_topk_candidates(down, max_k=max(LOCAL_KS)),
            }
            for name, (right, down) in variants.items()
        },
        reciprocal={
            name: {
                "right": _reciprocal_evidence(right),
                "down": _reciprocal_evidence(down),
            }
            for name, (right, down) in variants.items()
        },
        supply={
            name: {
                "right": freeze_topk_candidates(right, max_k=TOP_K_SUPPLY),
                "down": freeze_topk_candidates(down, max_k=TOP_K_SUPPLY),
            }
            for name, (right, down) in supply_scores.items()
        },
        runtime_seconds={
            "raw_d64": raw_seconds,
            "fullres_restore": restore_seconds,
            "restored_d64": restored_socket_seconds,
            "restored_descriptor": descriptor_seconds,
        },
    )


def _write_frozen(
    predictions: list[FrozenPrediction],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        prefix = f"case_{index:04d}"
        for variant, axes in prediction.candidates.items():
            for axis, candidates in axes.items():
                arrays[f"{prefix}__candidate__{variant}__{axis}"] = candidates
        for variant, axes in prediction.reciprocal.items():
            for axis, evidence in axes.items():
                for field, value in evidence.items():
                    arrays[f"{prefix}__reciprocal__{variant}__{axis}__{field}"] = value
        for variant, axes in prediction.supply.items():
            for axis, candidates in axes.items():
                arrays[f"{prefix}__supply__{variant}__{axis}"] = candidates
        cases.append(
            {
                "prefix": prefix,
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "variants": sorted(prediction.candidates),
                "runtime_seconds": prediction.runtime_seconds,
            }
        )
    array_path = output_dir / "frozen_local_predictions.npz"
    np.savez_compressed(array_path, **arrays)
    metadata_path = output_dir / "frozen_local_predictions.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "aiijc-fullres-boundary-local-predictions-v1",
                "contains_clean_pixels": False,
                "contains_restored_pixels": False,
                "contains_original_pixels": False,
                "contains_exact_references": False,
                "contains_layouts": False,
                "matcher_view_only": True,
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return array_path, metadata_path


def _truth_by_anchor(reference: np.ndarray, *, axis: str) -> np.ndarray:
    positions = np.arange(len(reference))
    valid = positions % GRID != GRID - 1 if axis == "right" else positions < COUNT - GRID
    delta = 1 if axis == "right" else GRID
    truth = np.full(COUNT, -1, dtype=np.int32)
    truth[reference[positions[valid]]] = reference[positions[valid] + delta]
    return truth


def _aggregate_supply(
    predictions: list[FrozenPrediction],
    references: dict[str, ExactSyntheticReference],
) -> dict[str, dict[str, float | int | dict[str, Any]]]:
    variants = ("restored_d64_ot", "restored_descriptor")
    totals: dict[str, dict[str, Any]] = {
        variant: {
            axis: {"total": 0, "raw_hits": 0, "union_hits": 0}
            for axis in ("right", "down")
        }
        for variant in variants
    }
    for prediction in predictions:
        reference = references[prediction.case_id].tile_at_position
        for axis in ("right", "down"):
            truth = _truth_by_anchor(reference, axis=axis)
            anchors = np.flatnonzero(truth >= 0)
            raw = prediction.supply["raw_d64_ot"][axis]
            for variant in variants:
                auxiliary = prediction.supply[variant][axis]
                raw_hit = np.any(raw[anchors] == truth[anchors, None], axis=1)
                auxiliary_hit = np.any(
                    auxiliary[anchors] == truth[anchors, None],
                    axis=1,
                )
                values = totals[variant][axis]
                values["total"] += len(anchors)
                values["raw_hits"] += int(raw_hit.sum())
                values["union_hits"] += int(np.count_nonzero(raw_hit | auxiliary_hit))
    output: dict[str, dict[str, float | int | dict[str, Any]]] = {}
    for variant, axes in totals.items():
        pooled_total = sum(int(values["total"]) for values in axes.values())
        pooled_raw = sum(int(values["raw_hits"]) for values in axes.values())
        pooled_union = sum(int(values["union_hits"]) for values in axes.values())
        axis_metrics = {}
        for axis, values in axes.items():
            denominator = int(values["total"])
            raw_coverage = int(values["raw_hits"]) / denominator
            union_coverage = int(values["union_hits"]) / denominator
            axis_metrics[axis] = {
                **values,
                "raw_coverage": raw_coverage,
                "union_coverage": union_coverage,
                "coverage_gain": union_coverage - raw_coverage,
            }
        output[variant] = {
            "axes": axis_metrics,
            "pooled_total": pooled_total,
            "pooled_raw_hits": pooled_raw,
            "pooled_union_hits": pooled_union,
            "pooled_raw_coverage": pooled_raw / pooled_total,
            "pooled_union_coverage": pooled_union / pooled_total,
            "pooled_coverage_gain": (pooled_union - pooled_raw) / pooled_total,
        }
    return output


def _aggregate_local(
    predictions: list[FrozenPrediction],
    references: dict[str, ExactSyntheticReference],
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, Any]]]:
    variants = tuple(predictions[0].candidates)
    totals: dict[str, dict[str, float | int]] = {}
    reciprocal_rows: dict[str, list[tuple[float, bool]]] = {name: [] for name in variants}
    valid_query_count = 0
    for prediction in predictions:
        reference = references[prediction.case_id].tile_at_position
        for name in variants:
            metrics = exact_local_retrieval_metrics(
                prediction.candidates[name]["right"],
                prediction.candidates[name]["down"],
                reference,
                ks=LOCAL_KS,
            )
            aggregate = totals.setdefault(
                name,
                {"pooled_total": 0, **{f"pooled_hits_at_{k}": 0 for k in LOCAL_KS}},
            )
            aggregate["pooled_total"] = int(aggregate["pooled_total"]) + int(
                metrics["pooled_total"]
            )
            for k in LOCAL_KS:
                key = f"pooled_hits_at_{k}"
                aggregate[key] = int(aggregate[key]) + int(metrics[key])
        for axis in ("right", "down"):
            truth = _truth_by_anchor(reference, axis=axis)
            valid = truth >= 0
            valid_query_count += int(valid.sum())
            for name in variants:
                evidence = prediction.reciprocal[name][axis]
                admitted = valid & evidence["reciprocal"]
                correct = evidence["target"] == truth
                reciprocal_rows[name].extend(
                    (float(confidence), bool(ok))
                    for confidence, ok in zip(
                        evidence["confidence"][admitted],
                        correct[admitted],
                        strict=True,
                    )
                )
    for metrics in totals.values():
        denominator = int(metrics["pooled_total"])
        for k in LOCAL_KS:
            metrics[f"pooled_r{k}"] = int(metrics[f"pooled_hits_at_{k}"]) / denominator

    native: dict[str, dict[str, Any]] = {}
    for name, rows in reciprocal_rows.items():
        correct = sum(int(ok) for _, ok in rows)
        native[name] = {
            "reciprocal_queries": len(rows),
            "coverage": len(rows) / valid_query_count,
            "precision": correct / len(rows) if rows else 0.0,
        }

    def precision_at(rows: list[tuple[float, bool]], limit: int) -> float:
        ordered = sorted(rows, key=lambda item: -item[0])[:limit]
        return sum(int(ok) for _, ok in ordered) / limit if limit else 0.0

    control_rows = reciprocal_rows["raw_d64_ot"]
    matched: dict[str, dict[str, Any]] = {}
    for name in PRIMARY_CANDIDATES:
        candidate_rows = reciprocal_rows[name]
        count = min(len(control_rows), len(candidate_rows))
        control_precision = precision_at(control_rows, count)
        candidate_precision = precision_at(candidate_rows, count)
        matched[name] = {
            "matched_query_count": count,
            "matched_coverage": count / valid_query_count,
            "candidate_precision": candidate_precision,
            "raw_d64_ot_precision": control_precision,
            "precision_gain": candidate_precision - control_precision,
        }
    return totals, {"native": native, "matched_vs_raw_d64_ot": matched}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.allow_nondeterministic_mps:
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "report.json").exists():
        raise FileExistsError(f"refusing to overwrite completed experiment: {output_dir}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest protocol digest is invalid")

    _, socket_lineage = load_checkpoint_with_lineage(
        args.socket_checkpoint,
        project_root=PROJECT_ROOT,
    )
    excluded, exclusion_reports = _exclusion_lineage(
        args.exclude_report,
        socket_lineage.filenames,
    )
    total_sources = args.train_sources + args.eval_sources + args.terminal_sources
    records = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=tuple(sorted(excluded)),
        limit=total_sources,
        seed=args.seed,
        namespace=SELECTION_NAMESPACE,
    )
    train_records = records[: args.train_sources]
    eval_records = records[args.train_sources : args.train_sources + args.eval_sources]
    terminal_records = records[args.train_sources + args.eval_sources :]
    train_names = tuple(str(record["filename"]) for record in train_records)
    eval_names = tuple(str(record["filename"]) for record in eval_records)
    terminal_names = tuple(str(record["filename"]) for record in terminal_records)
    validate_source_split(train_names, eval_names, terminal_names, excluded)

    preregistration = {
        "schema": "aiijc-fullres-boundary-local-gate-preregistration-v1",
        "created_before_training_or_eval_target_access": True,
        "candidate_roster": list(PRIMARY_CANDIDATES),
        "baseline": "frozen raw d64 partial-OT, unchanged raw tiles",
        "supply_arms": ["restored_d64_ot", "restored_descriptor"],
        "gate": {
            "discovery_continuation": {
                "ranking": (
                    f"pooled R1 gain >= {DISCOVERY_R1_DELTA} and pooled R5 gain >= "
                    f"{DISCOVERY_R5_DELTA}"
                ),
                "precision": (
                    f"matched reciprocal precision gain >= {DISCOVERY_PRECISION_DELTA} "
                    f"at coverage >= {MIN_RECIPROCAL_COVERAGE}"
                ),
                "supply": (
                    "directional raw-top32 union gain >= "
                    f"{DISCOVERY_DIRECTIONAL_SUPPLY_GAIN} on either axis, with the other "
                    f"axis no worse than -{DISCOVERY_OTHER_DIRECTION_MAX_LOSS}"
                ),
                "overall": "ranking OR precision OR supply",
            },
            "strong_decoder_promotion": {
                "supply": f"pooled raw-top32 union coverage gain >= {STRONG_SUPPLY_DELTA}",
                "ranking": (
                    f"pooled R1 gain >= {STRONG_R1_DELTA} and pooled R5 gain >= "
                    f"{STRONG_R5_DELTA}"
                ),
                "precision": (
                    f"matched reciprocal precision gain >= {STRONG_PRECISION_DELTA} "
                    f"at coverage >= {MIN_RECIPROCAL_COVERAGE}"
                ),
                "overall": "supply AND (ranking OR precision)",
            },
        },
        "selection": {
            "namespace": SELECTION_NAMESPACE,
            "train_filenames": list(train_names),
            "train_digest": names_digest(train_names),
            "eval_filenames": list(eval_names),
            "eval_digest": names_digest(eval_names),
            "terminal_filenames": list(terminal_names),
            "terminal_digest": names_digest(terminal_names),
            "terminal_target_files_opened": False,
        },
        "global_decoder_forbidden_in_this_runner": True,
    }
    preregistration_path = output_dir / "preregistered-local-gate.json"
    _write_json(preregistration_path, preregistration)

    train_boards = _prepare_boards(train_records, args.targets)
    benchmark_clean = np.ascontiguousarray(train_boards[0].tiles[: args.benchmark_batch])
    benchmark_dirty = distort_tiles(
        benchmark_clean,
        np.random.default_rng(args.seed + 991),
    )
    config = FullResolutionDenoiserConfig(width=args.width, blocks=args.blocks)
    benchmark = benchmark_devices(
        benchmark_clean,
        benchmark_dirty,
        config=config,
        border_width=args.border_width,
        repeats=args.benchmark_repeats,
    )
    device = choose_device(args.device, benchmark)
    print(
        json.dumps(
            {"event": "device_benchmark", "benchmark": benchmark, "chosen": str(device)}
        ),
        flush=True,
    )
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    observed_socket_digest = names_digest(socket_lineage.filenames, sort_names=True)
    if (
        len(socket_lineage.filenames) != socket.lineage.exposed_count
        or observed_socket_digest != socket.lineage.exposed_digest
    ):
        raise ValueError("strict Socket checkpoint and recursive lineage disagree")

    model = FullResolutionBoundaryDenoiser(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    history, runtime = train_model(model, train_boards, args, device)
    model.eval()
    checkpoint_path = output_dir / "fullres_boundary_denoiser.pt"
    exposed_names = tuple(
        sorted(set(train_names) | set(eval_names) | set(terminal_names) | excluded)
    )
    contract = {
        "architecture": "fullres-20x20-naf-boundary-denoiser-v1",
        "model_config": model_config_dict(model),
        "parameter_count": parameter_count,
        "spatial_downsampling": False,
        "feature_resolution_every_block": [20, 20],
        "input_views": ["raw_rgb", "per_tile_channel_standardised_rgb"],
        "objective": (
            "clean border-strip Charbonnier + boundary finite-difference preservation + "
            "normalised boundary shape + weak full-tile identity residual"
        ),
        "raw_tile_replacement": False,
        "matcher_view_only": True,
        "global_decoder": False,
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "contract": contract,
            "selection": {
                "train_filenames": list(train_names),
                "train_digest": names_digest(train_names),
                "eval_filenames": list(eval_names),
                "eval_digest": names_digest(eval_names),
                "terminal_filenames": list(terminal_names),
                "terminal_digest": names_digest(terminal_names),
                "lineage_train_filenames": list(train_names),
                "lineage_train_digest": names_digest(train_names, sort_names=True),
                "lineage_exposed_filenames": list(exposed_names),
                "lineage_exposed_digest": names_digest(exposed_names, sort_names=True),
            },
            "training_history": history,
        },
        checkpoint_path,
    )

    eval_boards = _prepare_boards(eval_records, args.targets)
    inputs: list[SyntheticSocketInput] = []
    references: dict[str, ExactSyntheticReference] = {}
    for board in eval_boards:
        synthetic_input, reference = make_exact_synthetic_case(
            board.tiles,
            source_filename=board.filename,
            draw_index=0,
            seed=args.seed,
        )
        inputs.append(synthetic_input)
        references[reference.case_id] = reference
    predictions = []
    for index, item in enumerate(inputs, start=1):
        predictions.append(
            freeze_prediction(
                model,
                socket,
                item,
                device=device,
                inference_batch=args.inference_batch,
            )
        )
        print(f"froze local prediction {index}/{len(inputs)} {item.source_filename}", flush=True)
    frozen_array, frozen_metadata = _write_frozen(predictions, output_dir=output_dir)
    local, reciprocal = _aggregate_local(predictions, references)
    supply = _aggregate_supply(predictions, references)

    candidate_to_supply = {
        "restored_d64_ot": "restored_d64_ot",
        "raw_restored_d64_rank50": "restored_d64_ot",
        "raw_restored_descriptor_rank50": "restored_descriptor",
    }
    baseline = local["raw_d64_ot"]
    candidate_gate: dict[str, dict[str, Any]] = {}
    for candidate in PRIMARY_CANDIDATES:
        supply_name = candidate_to_supply[candidate]
        r1_gain = float(local[candidate]["pooled_r1"]) - float(baseline["pooled_r1"])
        r5_gain = float(local[candidate]["pooled_r5"]) - float(baseline["pooled_r5"])
        precision = reciprocal["matched_vs_raw_d64_ot"][candidate]
        axis_gains = {
            axis: float(values["coverage_gain"])
            for axis, values in supply[supply_name]["axes"].items()
        }
        discovery_supply_passed = any(
            axis_gains[axis] >= DISCOVERY_DIRECTIONAL_SUPPLY_GAIN
            and axis_gains["down" if axis == "right" else "right"]
            >= -DISCOVERY_OTHER_DIRECTION_MAX_LOSS
            for axis in ("right", "down")
        )
        discovery_ranking_passed = (
            r1_gain >= DISCOVERY_R1_DELTA and r5_gain >= DISCOVERY_R5_DELTA
        )
        discovery_precision_passed = (
            float(precision["precision_gain"]) >= DISCOVERY_PRECISION_DELTA
            and float(precision["matched_coverage"]) >= MIN_RECIPROCAL_COVERAGE
        )
        strong_supply_passed = (
            float(supply[supply_name]["pooled_coverage_gain"]) >= STRONG_SUPPLY_DELTA
        )
        strong_ranking_passed = (
            r1_gain >= STRONG_R1_DELTA and r5_gain >= STRONG_R5_DELTA
        )
        strong_precision_passed = (
            float(precision["precision_gain"]) >= STRONG_PRECISION_DELTA
            and float(precision["matched_coverage"]) >= MIN_RECIPROCAL_COVERAGE
        )
        candidate_gate[candidate] = {
            "supply_arm": supply_name,
            "directional_supply_gains": axis_gains,
            "r1_gain": r1_gain,
            "r5_gain": r5_gain,
            "precision_gain": precision["precision_gain"],
            "matched_coverage": precision["matched_coverage"],
            "discovery": {
                "supply_passed": discovery_supply_passed,
                "ranking_passed": discovery_ranking_passed,
                "precision_passed": discovery_precision_passed,
                "continuation_passed": (
                    discovery_supply_passed
                    or discovery_ranking_passed
                    or discovery_precision_passed
                ),
            },
            "strong": {
                "supply_passed": strong_supply_passed,
                "ranking_passed": strong_ranking_passed,
                "precision_passed": strong_precision_passed,
                "decoder_promotion_passed": strong_supply_passed
                and (strong_ranking_passed or strong_precision_passed),
            },
        }
    discovery_continuation_passed = any(
        bool(values["discovery"]["continuation_passed"])
        for values in candidate_gate.values()
    )
    strong_decoder_promotion_passed = any(
        bool(values["strong"]["decoder_promotion_passed"])
        for values in candidate_gate.values()
    )
    report = {
        "experiment": "fullres-boundary-denoiser-local-pilot-v1",
        "status": (
            "strong-local-gate-passed-separate-global-diagnostic-eligible"
            if strong_decoder_promotion_passed
            else "discovery-positive-preserve-for-context-fusion-no-decoder"
            if discovery_continuation_passed
            else "discovery-gate-failed-stop-no-global-decoder"
        ),
        "contract": contract,
        "configuration": {
            key: [str(path) for path in value]
            if key == "exclude_report"
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        }
        | {"device_resolved": str(device)},
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "manifest_train_targets_only": True,
            "known_exact_synthetic_shuffle": True,
            "predictions_frozen_before_exact_reference_scoring": True,
            "raw_d64_baseline_unchanged": True,
            "original_raw_tiles_are_only_legal_output_material": True,
            "terminal_target_files_opened": False,
            "competition_test_opened": False,
            "global_decoder_run": False,
            "global_decoder_policy": "forbidden in this local-only runner",
            "preregistered_gate_path": str(preregistration_path),
            "preregistered_gate_sha256": sha256_file(preregistration_path),
        },
        "selection": {
            "train_filenames": list(train_names),
            "train_digest": names_digest(train_names),
            "eval_filenames": list(eval_names),
            "eval_digest": names_digest(eval_names),
            "terminal_filenames": list(terminal_names),
            "terminal_digest": names_digest(terminal_names),
            "excluded_reports": exclusion_reports,
            "excluded_filename_count": len(excluded),
            "excluded_filename_digest": names_digest(tuple(sorted(excluded))),
        },
        "resources": {
            "device_benchmark": benchmark,
            "device_resolved": str(device),
            "parameter_count": parameter_count,
            **runtime,
            "inference_seconds_sum": {
                name: float(sum(item.runtime_seconds[name] for item in predictions))
                for name in predictions[0].runtime_seconds
            },
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "socket_checkpoint": str(socket.path),
            "socket_checkpoint_sha256": socket.sha256,
            "frozen_predictions": str(frozen_array),
            "frozen_predictions_sha256": sha256_file(frozen_array),
            "frozen_metadata": str(frozen_metadata),
            "frozen_metadata_sha256": sha256_file(frozen_metadata),
        },
        "training_history": history,
        "local": local,
        "supply": supply,
        "reciprocal": reciprocal,
        "gate": {
            "candidate_results": candidate_gate,
            "discovery_continuation_passed": discovery_continuation_passed,
            "strong_decoder_promotion_passed": strong_decoder_promotion_passed,
        },
        "verdict": (
            "No decoder or exact/global placement panel was run. Passing this local gate "
            "would only authorise one separately frozen diagnostic."
        ),
    }
    report_path = output_dir / "report.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "local": local,
                "supply": supply,
                "gate": report["gate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
