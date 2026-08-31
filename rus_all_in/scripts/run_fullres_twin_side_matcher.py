#!/usr/bin/env python3
"""Capacity-check, train and locally gate the full-resolution twin side matcher."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.fullres_twin_side_matcher import (
    FullResolutionTwinSideMatcher,
    dual_corruption_retrieval_loss,
    twin_right_down_scores,
)
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.restoration_r6 import distort_tiles, split_square_tiles
from aiijc_puzzle.socket_sorter_production import LoadedSocketCheckpoint, load_socket_checkpoint
from aiijc_puzzle.synthetic_socket_evaluation import (
    ExactSyntheticReference,
    SyntheticSocketInput,
    exact_local_retrieval_metrics,
    freeze_topk_candidates,
    load_checkpoint_with_lineage,
    make_exact_synthetic_case,
    names_digest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/fullres_twin_side_matcher_preregistered_v1.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_SOCKET = (
    PROJECT_ROOT / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
GRID = 24
COUNT = GRID * GRID
LOCAL_KS = (1, 5, 32)
FIT_SOURCES = 256
EVAL_SOURCES = 24
MAX_STEPS = 600
EVALUATION_KEYWORDS = (
    "eval",
    "evaluation",
    "local",
    "confirm",
    "decoder",
    "terminal",
    "selected",
    "source",
    "panel",
    "calibration",
)
NON_PANEL_KEYWORDS = ("fit", "train", "lineage", "pool", "library", "excluded")


@dataclass(frozen=True)
class CleanBoard:
    filename: str
    target_sha256: str
    tiles: np.ndarray


@dataclass(frozen=True)
class TrainSpec:
    source_index: int
    first_seed: int
    second_seed: int
    permutation_seed: int


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
    parser.add_argument("stage", choices=("capacity", "benchmark", "pilot"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--socket-checkpoint", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--capacity-steps", type=int, default=160)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--prefetch-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20320917)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    if not 1 <= args.capacity_steps <= 300:
        raise ValueError("capacity-steps must be in [1, 300]")
    if not 1 <= args.prefetch_workers <= 4:
        raise ValueError("prefetch-workers must be in [1, 4]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be finite and non-negative")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    if args.device == "cpu" and args.allow_nondeterministic_mps:
        raise ValueError("MPS acknowledgment is incompatible with --device cpu")
    if args.device == "mps" and not args.allow_nondeterministic_mps:
        raise ValueError("MPS requires --allow-nondeterministic-mps")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "aiijc-fullres-twin-side-matcher-preregistered-v1":
        raise ValueError("unexpected preregistration schema")
    if payload["architecture"]["pixel_prediction_head"] is not False:
        raise ValueError("preregistered model must not contain a pixel prediction head")
    return payload


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _synchronise(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _tensor(tiles: np.ndarray, device: torch.device, *, batch: bool = True) -> torch.Tensor:
    value = (
        torch.from_numpy(np.ascontiguousarray(tiles))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    return value.unsqueeze(0) if batch else value


def _procedural_capacity_tiles() -> np.ndarray:
    size = 80
    y, x = np.mgrid[:size, :size].astype(np.float32)
    red = 127.5 + 60 * np.sin(x / 5.3) + 45 * np.cos((x + y) / 11.0)
    green = 127.5 + 55 * np.sin(y / 6.1) + 40 * np.cos((x - y) / 9.0)
    blue = 127.5 + 50 * np.sin((2 * x + y) / 13.0) + 35 * np.cos(y / 4.7)
    image = np.clip(np.stack((red, green, blue), axis=2), 0, 255).astype(np.uint8)
    return split_square_tiles(image)


def _two_view_case(
    clean: np.ndarray,
    *,
    first_seed: int,
    second_seed: int,
    permutation_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = distort_tiles(clean, np.random.default_rng(first_seed))
    second = distort_tiles(clean, np.random.default_rng(second_seed))
    permutation = np.random.default_rng(permutation_seed).permutation(len(clean))
    layout = np.argsort(permutation).astype(np.int64)
    return (
        np.ascontiguousarray(first[permutation]),
        np.ascontiguousarray(second[permutation]),
        np.ascontiguousarray(layout),
    )


def _train_step(
    model: FullResolutionTwinSideMatcher,
    optimizer: torch.optim.Optimizer,
    first: np.ndarray,
    second: np.ndarray,
    layout: np.ndarray,
    *,
    grid: int,
    device: torch.device,
) -> dict[str, float]:
    first_output = model(_tensor(first, device))
    second_output = model(_tensor(second, device))
    target = torch.from_numpy(layout).unsqueeze(0).to(device)
    loss, terms = dual_corruption_retrieval_loss(
        model,
        first_output,
        second_output,
        target,
        grid=grid,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        key: float(value.detach().cpu())
        for key, value in terms.items()
    } | {"grad_norm": float(grad_norm.detach().cpu())}


def run_capacity(args: argparse.Namespace, config: dict[str, Any]) -> None:
    device = _resolve_device(args.device)
    if device.type == "mps" and not args.allow_nondeterministic_mps:
        raise ValueError("capacity MPS run requires explicit nondeterminism acknowledgment")
    model = FullResolutionTwinSideMatcher().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-3,
        weight_decay=args.weight_decay,
    )
    clean = _procedural_capacity_tiles()
    permutation_seed = args.seed + 30
    history: list[dict[str, float | int]] = []
    started = perf_counter()
    model.train()
    for step in range(args.capacity_steps):
        first, second, layout = _two_view_case(
            clean,
            first_seed=args.seed + 2 * step + 1,
            second_seed=args.seed + 2 * step + 2,
            permutation_seed=permutation_seed,
        )
        record = _train_step(
            model,
            optimizer,
            first,
            second,
            layout,
            grid=4,
            device=device,
        )
        record["step"] = step + 1
        history.append(record)
        if step == 0 or (step + 1) % args.log_every == 0:
            print(json.dumps({"event": "capacity", **record}), flush=True)
    _synchronise(device)
    model.eval()
    first, second, layout = _two_view_case(
        clean,
        first_seed=args.seed + 9001,
        second_seed=args.seed + 9002,
        permutation_seed=permutation_seed,
    )
    with torch.inference_mode():
        output = model(_tensor(first, device))
    candidates = output.scores[0, (1, 3)].float().cpu().numpy()
    reference = np.asarray(layout, dtype=np.int32)
    metrics = exact_local_retrieval_metrics(
        freeze_topk_candidates(candidates[0], max_k=5),
        freeze_topk_candidates(candidates[1], max_k=5),
        reference,
        ks=(1, 5),
    )
    initial_loss = float(np.mean([row["loss"] for row in history[:10]]))
    final_loss = float(np.mean([row["loss"] for row in history[-10:]]))
    passed = bool(
        final_loss < 0.8 * initial_loss
        and float(metrics["pooled_r1"]) >= 0.25
        and float(metrics["pooled_r5"]) >= 0.70
    )
    report = {
        "schema": "aiijc-fullres-twin-capacity-v1",
        "status": "pass" if passed else "fail-stop",
        "preregistration": str(args.config),
        "preregistration_sha256": sha256_file(args.config),
        "device": str(device),
        "mps_bitwise_reproducibility_claimed": device.type != "mps",
        "grid": 4,
        "steps": args.capacity_steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "metrics": metrics,
        "gate": {
            "loss_ratio_max": 0.8,
            "minimum_r1": 0.25,
            "minimum_r5": 0.70,
            "passed": passed,
        },
        "runtime_seconds": perf_counter() - started,
        "history": history,
        "config_contract": config["architecture"],
    }
    _atomic_json(args.output_dir / "capacity-report.json", report)
    print(
        json.dumps({"event": "capacity_complete", "passed": passed, "metrics": metrics}),
        flush=True,
    )
    if not passed:
        raise RuntimeError("4x4 capacity gate failed")


def _benchmark_one_device(
    name: str,
    first: np.ndarray,
    second: np.ndarray,
    layout: np.ndarray,
) -> dict[str, Any]:
    device = torch.device(name)
    torch.manual_seed(41)
    model = FullResolutionTwinSideMatcher().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
    timings: list[float] = []
    for iteration in range(2):
        started = perf_counter()
        diagnostics = _train_step(
            model,
            optimizer,
            first,
            second,
            layout,
            grid=24,
            device=device,
        )
        _synchronise(device)
        if iteration:
            timings.append(perf_counter() - started)
    del model, optimizer
    if name == "mps":
        torch.mps.empty_cache()
    return {
        "seconds_per_full576_dual_view_step": float(np.mean(timings)),
        "boards_per_second": float(1.0 / np.mean(timings)),
        "diagnostics": diagnostics,
        "timed_repeats": len(timings),
    }


def run_benchmark(args: argparse.Namespace) -> None:
    clean = np.tile(_procedural_capacity_tiles(), (36, 1, 1, 1))
    first, second, layout = _two_view_case(
        clean,
        first_seed=args.seed + 7001,
        second_seed=args.seed + 7002,
        permutation_seed=args.seed + 7003,
    )
    names = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    results = {}
    for name in names:
        started = perf_counter()
        results[name] = _benchmark_one_device(name, first, second, layout)
        results[name]["wall_seconds_including_warmup"] = perf_counter() - started
        print(
            json.dumps({"event": "benchmark_device", "device": name, **results[name]}),
            flush=True,
        )
    chosen = min(
        results,
        key=lambda name: float(results[name]["seconds_per_full576_dual_view_step"]),
    )
    report = {
        "schema": "aiijc-fullres-twin-device-benchmark-v1",
        "preregistration_sha256": sha256_file(args.config),
        "workload": "full 576 tiles, two views, four within/cross retrieval losses, backward+AdamW",
        "results": results,
        "chosen_device": chosen,
    }
    _atomic_json(args.output_dir / "device-benchmark.json", report)
    print(json.dumps({"event": "benchmark_complete", "chosen": chosen}), flush=True)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB {IMAGE_SIZE}x{IMAGE_SIZE}: {path}")
        return np.asarray(image, dtype=np.uint8)


def _prepare_boards(records: tuple[dict[str, Any], ...], targets: Path) -> list[CleanBoard]:
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


def _json_panel_names(value: Any, *, key_path: tuple[str, ...] = ()) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            names.update(_json_panel_names(child, key_path=(*key_path, str(key).lower())))
        return names
    if isinstance(value, list):
        key = ".".join(key_path)
        if any(token in key for token in NON_PANEL_KEYWORDS):
            return names
        if any(token in key for token in EVALUATION_KEYWORDS):
            names.update(
                Path(item).name
                for item in value
                if isinstance(item, str) and item.endswith(".png")
            )
        else:
            for child in value:
                names.update(_json_panel_names(child, key_path=key_path))
        return names
    if isinstance(value, str) and value.endswith(".png"):
        key = ".".join(key_path)
        if not any(token in key for token in NON_PANEL_KEYWORDS) and any(
            token in key for token in EVALUATION_KEYWORDS
        ):
            names.add(Path(value).name)
    return names


def _evaluation_exclusion_registry(
    socket_lineage: tuple[str, ...],
    *,
    output_dir: Path,
) -> tuple[set[str], list[dict[str, Any]]]:
    excluded = set(socket_lineage)
    records: list[dict[str, Any]] = []
    roots = (PROJECT_ROOT / "configs", PROJECT_ROOT / "outputs")
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            if output_dir.resolve() in path.resolve().parents:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            names = _json_panel_names(payload)
            if names:
                excluded.update(names)
                records.append(
                    {
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "sha256": sha256_file(path),
                        "panel_filename_count": len(names),
                    }
                )
    return excluded, records


def _select_rosters(
    manifest: dict[str, Any],
    excluded: set[str],
    *,
    seed: int,
    namespace: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    ranked = select_manifest_records(
        manifest,
        "train",
        limit=len(manifest["splits"]["train"]),
        seed=seed,
        namespace=namespace,
    )
    eligible = tuple(record for record in ranked if record["filename"] not in excluded)
    required = FIT_SOURCES + EVAL_SOURCES
    if len(eligible) < required:
        raise ValueError(f"only {len(eligible)} eligible train sources remain, need {required}")
    fit = tuple(dict(record) for record in eligible[:FIT_SOURCES])
    evaluation = tuple(dict(record) for record in eligible[FIT_SOURCES:required])
    return fit, evaluation


def _training_specs(steps: int, *, seed: int) -> list[TrainSpec]:
    rng = np.random.default_rng(seed + 101)
    return [
        TrainSpec(
            source_index=int(rng.integers(FIT_SOURCES)),
            first_seed=int(rng.integers(0, np.iinfo(np.int64).max)),
            second_seed=int(rng.integers(0, np.iinfo(np.int64).max)),
            permutation_seed=int(rng.integers(0, np.iinfo(np.int64).max)),
        )
        for _ in range(steps)
    ]


def _materialise_training(
    board: CleanBoard,
    spec: TrainSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _two_view_case(
        board.tiles,
        first_seed=spec.first_seed,
        second_seed=spec.second_seed,
        permutation_seed=spec.permutation_seed,
    )


def train_model(
    model: FullResolutionTwinSideMatcher,
    boards: list[CleanBoard],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    specs = _training_specs(args.steps, seed=args.seed)
    history: list[dict[str, Any]] = []
    started = perf_counter()
    prefetch_wait = 0.0
    model.train()
    with ThreadPoolExecutor(max_workers=args.prefetch_workers) as executor:
        futures: dict[int, Future[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        submit = 0
        window = max(2, 2 * args.prefetch_workers)
        while submit < min(window, args.steps):
            spec = specs[submit]
            futures[submit] = executor.submit(
                _materialise_training,
                boards[spec.source_index],
                spec,
            )
            submit += 1
        for step, spec in enumerate(specs):
            wait_started = perf_counter()
            first, second, layout = futures.pop(step).result()
            prefetch_wait += perf_counter() - wait_started
            if submit < args.steps:
                next_spec = specs[submit]
                futures[submit] = executor.submit(
                    _materialise_training,
                    boards[next_spec.source_index],
                    next_spec,
                )
                submit += 1
            diagnostics = _train_step(
                model,
                optimizer,
                first,
                second,
                layout,
                grid=GRID,
                device=device,
            )
            scheduler.step()
            record: dict[str, Any] = {
                "step": step + 1,
                "source_filename": boards[spec.source_index].filename,
                "first_corruption_seed": spec.first_seed,
                "second_corruption_seed": spec.second_seed,
                "permutation_seed": spec.permutation_seed,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **diagnostics,
            }
            history.append(record)
            if step == 0 or (step + 1) % args.log_every == 0:
                recent = history[-args.log_every :]
                print(
                    json.dumps(
                        {
                            "event": "train",
                            "step": step + 1,
                            "loss": float(np.mean([row["loss"] for row in recent])),
                            "r1": float(np.mean([row["r1"] for row in recent])),
                            "r5": float(np.mean([row["r5"] for row in recent])),
                            "elapsed_seconds": perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    _synchronise(device)
    return history, {
        "training_seconds": perf_counter() - started,
        "prefetch_wait_seconds": prefetch_wait,
        "prefetch_workers": args.prefetch_workers,
    }


@torch.inference_mode()
def _socket_raw_scores(
    socket: LoadedSocketCheckpoint,
    tiles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    output = socket.model(_tensor(tiles, device), grid=GRID)
    return (
        np.ascontiguousarray(output.right_raw[0].float().cpu().numpy()),
        np.ascontiguousarray(output.down_raw[0].float().cpu().numpy()),
    )


def _reciprocal_evidence(scores: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(scores, dtype=np.float32).copy()
    count = len(value)
    if value.shape != (count, count) or not np.isfinite(value).all():
        raise ValueError("scores must be one finite square matrix")
    np.fill_diagonal(value, -np.inf)
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
    model: FullResolutionTwinSideMatcher,
    socket: LoadedSocketCheckpoint,
    item: SyntheticSocketInput,
    *,
    device: torch.device,
) -> FrozenPrediction:
    socket_started = perf_counter()
    raw_right, raw_down = _socket_raw_scores(socket, item.tiles, device=device)
    socket_seconds = perf_counter() - socket_started
    twin_started = perf_counter()
    twin_right, twin_down = twin_right_down_scores(model, item.tiles, device=device)
    twin_seconds = perf_counter() - twin_started
    variants = {
        "socket_d64_raw": (raw_right, raw_down),
        "fullres_twin": (twin_right, twin_down),
    }
    return FrozenPrediction(
        case_id=item.case_id,
        source_filename=item.source_filename,
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
                "right": freeze_topk_candidates(right, max_k=32),
                "down": freeze_topk_candidates(down, max_k=32),
            }
            for name, (right, down) in variants.items()
        },
        runtime_seconds={"socket_d64_raw": socket_seconds, "fullres_twin": twin_seconds},
    )


def _write_frozen(predictions: list[FrozenPrediction], output_dir: Path) -> tuple[Path, Path]:
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        prefix = f"case_{index:04d}"
        for family, variants in (
            ("candidate", prediction.candidates),
            ("supply", prediction.supply),
        ):
            for variant, axes in variants.items():
                for axis, value in axes.items():
                    arrays[f"{prefix}__{family}__{variant}__{axis}"] = value
        for variant, axes in prediction.reciprocal.items():
            for axis, evidence in axes.items():
                for field, value in evidence.items():
                    arrays[f"{prefix}__reciprocal__{variant}__{axis}__{field}"] = value
        cases.append(
            {
                "prefix": prefix,
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "runtime_seconds": prediction.runtime_seconds,
            }
        )
    array_path = output_dir / "frozen-local-predictions.npz"
    np.savez_compressed(array_path, **arrays)
    metadata_path = output_dir / "frozen-local-predictions.json"
    _atomic_json(
        metadata_path,
        {
            "schema": "aiijc-fullres-twin-frozen-local-v1",
            "contains_exact_references": False,
            "contains_clean_or_generated_pixels": False,
            "contains_layouts": False,
            "matcher_scores_only": True,
            "cases": cases,
        },
    )
    return array_path, metadata_path


def _truth_by_anchor(reference: np.ndarray, *, axis: str) -> np.ndarray:
    positions = np.arange(COUNT)
    valid = positions % GRID != GRID - 1 if axis == "right" else positions < COUNT - GRID
    delta = 1 if axis == "right" else GRID
    truth = np.full(COUNT, -1, dtype=np.int32)
    truth[reference[positions[valid]]] = reference[positions[valid] + delta]
    return truth


def _aggregate_metrics(
    predictions: list[FrozenPrediction],
    references: dict[str, ExactSyntheticReference],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    variants = ("socket_d64_raw", "fullres_twin")
    local = {
        name: {
            **{f"{axis}_total": 0 for axis in ("right", "down", "pooled")},
            **{
                f"{axis}_hits_at_{k}": 0
                for axis in ("right", "down", "pooled")
                for k in LOCAL_KS
            },
        }
        for name in variants
    }
    reciprocal_rows: dict[str, list[tuple[float, bool]]] = {name: [] for name in variants}
    supply = {
        axis: {"total": 0, "raw_hits": 0, "union_hits": 0}
        for axis in ("right", "down")
    }
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
            for axis in ("right", "down", "pooled"):
                local[name][f"{axis}_total"] += int(metrics[f"{axis}_total"])
                for k in LOCAL_KS:
                    local[name][f"{axis}_hits_at_{k}"] += int(
                        metrics[f"{axis}_hits_at_{k}"]
                    )
        for axis in ("right", "down"):
            truth = _truth_by_anchor(reference, axis=axis)
            valid = truth >= 0
            anchors = np.flatnonzero(valid)
            raw = prediction.supply["socket_d64_raw"][axis]
            twin = prediction.supply["fullres_twin"][axis]
            raw_hit = np.any(raw[anchors] == truth[anchors, None], axis=1)
            twin_hit = np.any(twin[anchors] == truth[anchors, None], axis=1)
            supply[axis]["total"] += len(anchors)
            supply[axis]["raw_hits"] += int(raw_hit.sum())
            supply[axis]["union_hits"] += int(np.count_nonzero(raw_hit | twin_hit))
            valid_query_count += len(anchors)
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
    for metrics in local.values():
        for axis in ("right", "down", "pooled"):
            denominator = metrics[f"{axis}_total"]
            for k in LOCAL_KS:
                metrics[f"{axis}_r{k}"] = (
                    metrics[f"{axis}_hits_at_{k}"] / denominator
                )
    pooled_total = sum(values["total"] for values in supply.values())
    pooled_raw = sum(values["raw_hits"] for values in supply.values())
    pooled_union = sum(values["union_hits"] for values in supply.values())
    supply_output: dict[str, Any] = {
        "pooled_total": pooled_total,
        "pooled_raw_top32_coverage": pooled_raw / pooled_total,
        "pooled_union_top32_coverage": pooled_union / pooled_total,
        "pooled_coverage_gain": (pooled_union - pooled_raw) / pooled_total,
        "axes": {},
    }
    for axis, values in supply.items():
        raw_coverage = values["raw_hits"] / values["total"]
        union_coverage = values["union_hits"] / values["total"]
        supply_output["axes"][axis] = values | {
            "raw_top32_coverage": raw_coverage,
            "union_top32_coverage": union_coverage,
            "coverage_gain": union_coverage - raw_coverage,
        }

    def precision_at(rows: list[tuple[float, bool]], limit: int) -> float:
        selected = sorted(rows, key=lambda item: -item[0])[:limit]
        return sum(int(ok) for _, ok in selected) / limit if limit else 0.0

    native = {}
    for name, rows in reciprocal_rows.items():
        native[name] = {
            "queries": len(rows),
            "coverage": len(rows) / valid_query_count,
            "precision": sum(int(ok) for _, ok in rows) / len(rows) if rows else 0.0,
        }
    control = reciprocal_rows["socket_d64_raw"]
    candidate = reciprocal_rows["fullres_twin"]
    matched_count = min(len(control), len(candidate))
    candidate_precision = precision_at(candidate, matched_count)
    control_precision = precision_at(control, matched_count)
    reciprocal = {
        "native": native,
        "matched_vs_socket_d64_raw": {
            "query_count": matched_count,
            "coverage": matched_count / valid_query_count,
            "candidate_precision": candidate_precision,
            "socket_d64_raw_precision": control_precision,
            "precision_gain": candidate_precision - control_precision,
        },
    }
    return local, supply_output, reciprocal


def run_pilot(args: argparse.Namespace, config: dict[str, Any]) -> None:
    capacity_path = args.output_dir / "capacity-report.json"
    benchmark_path = args.output_dir / "device-benchmark.json"
    capacity_passed = (
        capacity_path.is_file()
        and json.loads(capacity_path.read_text())["gate"]["passed"] is True
    )
    if not capacity_passed:
        raise RuntimeError("passing capacity-report.json is required before pilot")
    if not benchmark_path.is_file():
        raise RuntimeError("device-benchmark.json is required before pilot")
    benchmark = json.loads(benchmark_path.read_text())
    device_name = benchmark["chosen_device"] if args.device == "auto" else args.device
    device = _resolve_device(device_name)
    if device.type == "mps" and not args.allow_nondeterministic_mps:
        raise ValueError("pilot MPS run requires explicit nondeterminism acknowledgment")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest is invalid")
    socket_payload, socket_lineage = load_checkpoint_with_lineage(
        args.socket_checkpoint,
        project_root=PROJECT_ROOT,
    )
    del socket_payload
    excluded, exclusion_records = _evaluation_exclusion_registry(
        socket_lineage.filenames,
        output_dir=args.output_dir,
    )
    fit_records, eval_records = _select_rosters(
        manifest,
        excluded,
        seed=args.seed,
        namespace=config["selection"]["namespace"],
    )
    fit_names = tuple(str(record["filename"]) for record in fit_records)
    eval_names = tuple(str(record["filename"]) for record in eval_records)
    if set(fit_names) & set(eval_names) or set(eval_names) & excluded:
        raise RuntimeError("fit/evaluation source-disjoint invariant failed")
    commitment = {
        "schema": "aiijc-fullres-twin-selection-commitment-v1",
        "status": "frozen-before-selected-target-access",
        "preregistration_path": _project_relative(args.config),
        "preregistration_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_digest": manifest["protocol_digest"],
        "manifest_split": "train",
        "namespace": config["selection"]["namespace"],
        "seed": args.seed,
        "socket_checkpoint_sha256": sha256_file(args.socket_checkpoint),
        "socket_recursive_lineage_count": len(socket_lineage.filenames),
        "excluded_filename_count": len(excluded),
        "excluded_filename_digest": names_digest(tuple(sorted(excluded)), sort_names=True),
        "exclusion_registry": exclusion_records,
        "fit_filenames": list(fit_names),
        "fit_order_digest": names_digest(fit_names),
        "evaluation_filenames": list(eval_names),
        "evaluation_order_digest": names_digest(eval_names),
        "fit_evaluation_overlap": [],
        "evaluation_exclusion_overlap": [],
        "holdout_and_competition_test_opened": False,
    }
    commitment_path = args.output_dir / "selection-commitment.json"
    _atomic_json(commitment_path, commitment)
    print(
        json.dumps(
            {
                "event": "selection_frozen",
                "commitment": str(commitment_path),
                "sha256": sha256_file(commitment_path),
                "fit_digest": commitment["fit_order_digest"],
                "evaluation_digest": commitment["evaluation_order_digest"],
            }
        ),
        flush=True,
    )

    fit_boards = _prepare_boards(fit_records, args.targets)
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    model = FullResolutionTwinSideMatcher().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count >= 250_000:
        raise RuntimeError("model exceeds preregistered 250k parameter budget")
    history, training_runtime = train_model(model, fit_boards, args, device)
    model.eval()
    checkpoint_path = args.output_dir / "fullres-twin-side-matcher.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "contract": {
                "architecture": "fullres-ordered-twin-side-matcher-v1",
                "dimension": 48,
                "field_blocks": 4,
                "sequence_blocks": 2,
                "raw_skip_gain": 0.35,
                "parameter_count": parameter_count,
                "feature_resolution": [20, 20],
                "spatial_downsampling": False,
                "ordered_side_positions": 20,
                "pixel_prediction_head": False,
                "matcher_only": True,
            },
            "selection": {
                "train_filenames": list(fit_names),
                "train_digest": names_digest(fit_names),
                "evaluation_filenames": list(eval_names),
                "evaluation_digest": names_digest(eval_names),
            },
            "training_history": history,
        },
        checkpoint_path,
    )

    eval_boards = _prepare_boards(eval_records, args.targets)
    inputs: list[SyntheticSocketInput] = []
    references: dict[str, ExactSyntheticReference] = {}
    for board in eval_boards:
        item, reference = make_exact_synthetic_case(
            board.tiles,
            source_filename=board.filename,
            draw_index=0,
            seed=args.seed,
        )
        inputs.append(item)
        references[reference.case_id] = reference
    predictions: list[FrozenPrediction] = []
    for index, item in enumerate(inputs, start=1):
        predictions.append(freeze_prediction(model, socket, item, device=device))
        print(f"froze prediction {index}/{len(inputs)} {item.source_filename}", flush=True)
    frozen_array, frozen_metadata = _write_frozen(predictions, args.output_dir)
    local, supply, reciprocal = _aggregate_metrics(predictions, references)
    baseline = local["socket_d64_raw"]
    candidate = local["fullres_twin"]
    r1_gain = float(candidate["pooled_r1"]) - float(baseline["pooled_r1"])
    r5_gain = float(candidate["pooled_r5"]) - float(baseline["pooled_r5"])
    matched = reciprocal["matched_vs_socket_d64_raw"]
    ranking_passed = r1_gain >= 0.0025 and r5_gain >= 0.0
    supply_passed = (
        float(supply["pooled_coverage_gain"]) >= 0.01
        and float(matched["precision_gain"]) >= 0.01
        and float(matched["coverage"]) >= 0.03
    )
    gate_passed = ranking_passed or supply_passed
    report = {
        "schema": "aiijc-fullres-twin-side-matcher-pilot-v1",
        "status": "low-d1-discovery-pass" if gate_passed else "fail-stop-no-decoder",
        "preregistration": {
                "path": _project_relative(args.config),
            "sha256": sha256_file(args.config),
        },
        "selection_commitment": {
            "path": _project_relative(commitment_path),
            "sha256": sha256_file(commitment_path),
        },
        "device": str(device),
        "mps_nondeterminism_acknowledged": bool(
            device.type == "mps" and args.allow_nondeterministic_mps
        ),
        "mps_bitwise_reproducibility_claimed": device.type != "mps",
        "parameter_count": parameter_count,
        "steps": args.steps,
        "fit_sources": FIT_SOURCES,
        "evaluation_sources": EVAL_SOURCES,
        "training_runtime": training_runtime,
        "checkpoint": {
            "path": _project_relative(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "frozen_predictions": {
            "array_path": _project_relative(frozen_array),
            "array_sha256": sha256_file(frozen_array),
            "metadata_path": _project_relative(frozen_metadata),
            "metadata_sha256": sha256_file(frozen_metadata),
            "contains_exact_references_or_pixels": False,
        },
        "metrics": {
            "local": local,
            "top32_union_supply": supply,
            "reciprocal": reciprocal,
        },
        "gate": {
            "r1_gain": r1_gain,
            "r5_gain": r5_gain,
            "ranking_arm": {
                "minimum_r1_gain": 0.0025,
                "minimum_r5_gain": 0.0,
                "passed": ranking_passed,
            },
            "supply_arm": {
                "minimum_union_coverage_gain": 0.01,
                "minimum_matched_precision_gain": 0.01,
                "minimum_matched_coverage": 0.03,
                "passed": supply_passed,
            },
            "pass_logic": "ranking_arm OR supply_arm",
            "passed": gate_passed,
            "decoder_authorized": False,
        },
        "legality": {
            "organizer_train_only": True,
            "target_at_inference": False,
            "rgb_reconstruction_objective": False,
            "restored_or_generated_pixels_emitted": False,
            "global_decoder_run": False,
            "holdout_or_competition_test_opened": False,
        },
    }
    report_path = args.output_dir / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "pilot_complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "gate": report["gate"],
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = _load_config(args.config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.allow_nondeterministic_mps:
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "capacity":
        run_capacity(args, config)
    elif args.stage == "benchmark":
        run_benchmark(args)
    else:
        run_pilot(args, config)


if __name__ == "__main__":
    main()
