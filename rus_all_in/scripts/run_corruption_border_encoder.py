#!/usr/bin/env python3
"""Train and locally gate the historical E13 corruption-aware border encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.corruption_border_encoder import (
    CORRUPTION_MODES,
    CorruptionAwareBorderEncoder,
    corrupt_e13_tiles,
    corruption_aware_training_loss,
    e13_curriculum_severity,
    e13_right_down_scores,
)
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_pasha_matched import row_rank_percentiles
from aiijc_puzzle.socket_sorter_production import (
    LoadedSocketCheckpoint,
    choose_deterministic_device,
    load_socket_checkpoint,
)
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
SELECTION_NAMESPACE = "aiijc-e13-corruption-border-encoder-v1"
MAX_TRAIN_SOURCES = 256
MAX_STEPS = 400
LOCAL_KS = (1, 5)
R1_GATE_GAIN = 0.02
MATCHED_PRECISION_GATE_GAIN = 0.05
MIN_MATCHED_RECIPROCAL_COVERAGE = 0.03


@dataclass(frozen=True)
class CleanBoard:
    filename: str
    target_sha256: str
    tiles: np.ndarray


@dataclass(frozen=True)
class FrozenLocalPrediction:
    case_id: str
    source_filename: str
    candidates: dict[str, dict[str, np.ndarray]]
    reciprocal: dict[str, dict[str, dict[str, np.ndarray]]]
    inference_seconds: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--socket-checkpoint", type=Path, default=DEFAULT_SOCKET_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sources", type=int, default=256)
    parser.add_argument("--eval-sources", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--dimension", type=int, default=96)
    parser.add_argument("--border", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--allow-nondeterministic-mps",
        action="store_true",
        help=(
            "explicitly permit MPS backward operations that PyTorch cannot execute "
            "deterministically; source/corruption/shuffle selection remains seeded"
        ),
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--exclude-report",
        type=Path,
        action="append",
        default=[],
        help="prior exact/model report whose complete declared filename lineage is excluded",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.train_sources <= MAX_TRAIN_SOURCES:
        raise ValueError(f"train-sources must be in [1, {MAX_TRAIN_SOURCES}]")
    if not 1 <= args.eval_sources <= 16:
        raise ValueError("eval-sources must be in [1, 16]")
    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    if not 2 <= args.grid <= 24:
        raise ValueError("grid must be in [2, 24]")
    if args.dimension <= 0 or not 1 <= args.border <= 10 or args.log_every <= 0:
        raise ValueError("dimension/border/log-every are out of range")
    for name in ("learning_rate", "weight_decay"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and non-negative")
    if args.learning_rate == 0:
        raise ValueError("learning-rate must be positive")
    if args.allow_nondeterministic_mps and args.device != "mps":
        raise ValueError("allow-nondeterministic-mps is valid only with --device mps")
    if not args.exclude_report:
        raise ValueError("at least one prior exact/model exclude-report is required")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def _collect_declared_filenames(value: Any, *, parent_key: str = "") -> set[str]:
    """Collect all filename-list lineage fields, including fit/confirm variants."""

    names: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            names.update(_collect_declared_filenames(child, parent_key=str(key)))
    elif isinstance(value, list):
        if parent_key.endswith("filenames") and all(
            isinstance(item, str) for item in value
        ):
            names.update(Path(item).name for item in value if item.endswith(".png"))
        else:
            for child in value:
                names.update(_collect_declared_filenames(child, parent_key=parent_key))
    elif (
        isinstance(value, str)
        and parent_key.endswith("filename")
        and value.endswith(".png")
    ):
        names.add(Path(value).name)
    return names


def _exclusion_lineage(
    paths: list[Path],
    socket_lineage_names: tuple[str, ...],
) -> tuple[set[str], list[dict[str, Any]]]:
    names = set(socket_lineage_names)
    reports: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        found = _collect_declared_filenames(payload)
        if not found:
            raise ValueError(f"exclude-report exposes no declared filename lineage: {path}")
        names.update(found)
        reports.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "declared_filename_count": len(found),
            }
        )
    return names, reports


def _crop_tiles(tiles: np.ndarray, *, grid: int, filename: str, seed: int) -> np.ndarray:
    if grid == 24:
        return np.ascontiguousarray(tiles)
    digest = hashlib.sha256(f"{filename}\0{seed}\0e13-crop".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    row = int(generator.integers(0, 24 - grid + 1))
    column = int(generator.integers(0, 24 - grid + 1))
    board = tiles.reshape(24, 24, 20, 20, 3)
    crop = board[row : row + grid, column : column + grid]
    return np.ascontiguousarray(crop.reshape(-1, 20, 20, 3))


def _prepare_boards(
    records: tuple[Any, ...],
    *,
    targets: Path,
    grid: int,
    seed: int,
) -> list[CleanBoard]:
    boards: list[CleanBoard] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        path = targets / filename
        digest = sha256_file(path)
        if digest != record.get("target_sha256"):
            raise ValueError(f"manifest target hash mismatch: {filename}")
        tiles = _crop_tiles(split_tiles(_load_rgb(path)), grid=grid, filename=filename, seed=seed)
        boards.append(CleanBoard(filename, digest, tiles))
        if index == 1 or index % 64 == 0 or index == len(records):
            print(f"prepared source {index}/{len(records)} {filename}", flush=True)
    return boards


def _tensor(tiles: np.ndarray, *, device: torch.device) -> torch.Tensor:
    value = np.asarray(tiles)
    return torch.from_numpy(np.ascontiguousarray(value)).permute(0, 3, 1, 2).to(device)


def train_model(
    model: CorruptionAwareBorderEncoder,
    boards: list[CleanBoard],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.steps,
        eta_min=args.learning_rate * 0.08,
    )
    source_generator = np.random.default_rng(args.seed + 1)
    history: list[dict[str, Any]] = []
    started = perf_counter()
    model.train()
    for step in range(args.steps):
        board = boards[int(source_generator.integers(len(boards)))]
        severity = e13_curriculum_severity(step, args.steps)
        mode = CORRUPTION_MODES[step % len(CORRUPTION_MODES)]
        corruption_seed = args.seed + 1_000_003 * (step + 1)
        corrupt = corrupt_e13_tiles(
            board.tiles,
            np.random.default_rng(corruption_seed),
            severity=severity,
            mode=mode,
        )
        clean_sides = model(_tensor(board.tiles, device=device).unsqueeze(0))
        corrupt_sides = model(_tensor(corrupt, device=device).unsqueeze(0))
        loss, diagnostics = corruption_aware_training_loss(
            clean_sides,
            corrupt_sides,
            grid=args.grid,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        scheduler.step()
        record: dict[str, Any] = {
            "step": step + 1,
            "source_filename": board.filename,
            "severity": severity,
            "mode": mode,
            "corruption_seed": corruption_seed,
            **diagnostics,
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
                        "corrupt_r1": float(
                            np.mean([row["corrupt_r1"] for row in recent])
                        ),
                        "corrupt_r5": float(
                            np.mean([row["corrupt_r5"] for row in recent])
                        ),
                        "elapsed_seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return history, perf_counter() - started


def _socket_scores(
    socket: LoadedSocketCheckpoint,
    tiles: np.ndarray,
    *,
    grid: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tensor = _tensor(tiles.astype(np.float32) / 255.0, device=device)
    output = socket.model(tensor.unsqueeze(0), grid=grid)
    count = grid * grid
    normaliser = math.log(float(count + grid))
    return (
        output.right_raw[0].float().cpu().numpy(),
        output.down_raw[0].float().cpu().numpy(),
        output.right_log_assignment[0, :count, :count].float().cpu().numpy()
        + normaliser,
        output.down_log_assignment[0, :count, :count].float().cpu().numpy()
        + normaliser,
    )


def _rank_fusion(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
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
    model: CorruptionAwareBorderEncoder,
    socket: LoadedSocketCheckpoint,
    synthetic_input: SyntheticSocketInput,
    *,
    grid: int,
    device: torch.device,
) -> FrozenLocalPrediction:
    started = perf_counter()
    e13_right, e13_down = e13_right_down_scores(
        model,
        synthetic_input.tiles,
        device=device,
    )
    e13_seconds = perf_counter() - started
    socket_started = perf_counter()
    socket_raw_right, socket_raw_down, socket_ot_right, socket_ot_down = _socket_scores(
        socket,
        synthetic_input.tiles,
        grid=grid,
        device=device,
    )
    socket_seconds = perf_counter() - socket_started
    variants = {
        "e13_raw": (e13_right, e13_down),
        "socket_d64_raw": (socket_raw_right, socket_raw_down),
        "socket_d64_ot": (socket_ot_right, socket_ot_down),
        "e13_socket_ot_rank50": (
            _rank_fusion(e13_right, socket_ot_right),
            _rank_fusion(e13_down, socket_ot_down),
        ),
    }
    return FrozenLocalPrediction(
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
        inference_seconds={"e13": e13_seconds, "socket_d64": socket_seconds},
    )


def _write_frozen(
    predictions: list[FrozenLocalPrediction],
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
        cases.append(
            {
                "prefix": prefix,
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "variants": sorted(prediction.candidates),
                "inference_seconds": prediction.inference_seconds,
            }
        )
    array_path = output_dir / "frozen_local_predictions.npz"
    np.savez_compressed(array_path, **arrays)
    metadata_path = output_dir / "frozen_local_predictions.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "aiijc-e13-local-predictions-v1",
                "contains_clean_pixels": False,
                "contains_exact_references": False,
                "contains_layouts": False,
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
    count = len(reference)
    grid = round(math.sqrt(count))
    positions = np.arange(count)
    valid = positions % grid != grid - 1 if axis == "right" else positions < count - grid
    delta = 1 if axis == "right" else grid
    truth = np.full(count, -1, dtype=np.int32)
    truth[reference[positions[valid]]] = reference[positions[valid] + delta]
    return truth


def _aggregate_local(
    predictions: list[FrozenLocalPrediction],
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
                {
                    "pooled_total": 0,
                    "pooled_hits_at_1": 0,
                    "pooled_hits_at_5": 0,
                },
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
    matched: dict[str, dict[str, Any]] = {}
    control_rows = reciprocal_rows["socket_d64_ot"]
    for name in ("e13_raw", "e13_socket_ot_rank50"):
        candidate_rows = reciprocal_rows[name]
        count = min(len(control_rows), len(candidate_rows))

        def precision_at(rows: list[tuple[float, bool]], limit: int) -> float:
            ordered = sorted(rows, key=lambda item: -item[0])[:limit]
            return sum(int(ok) for _, ok in ordered) / limit if limit else 0.0

        control_precision = precision_at(control_rows, count)
        candidate_precision = precision_at(candidate_rows, count)
        matched[name] = {
            "matched_query_count": count,
            "matched_coverage": count / valid_query_count,
            "candidate_precision": candidate_precision,
            "socket_d64_ot_precision": control_precision,
            "precision_gain": candidate_precision - control_precision,
        }
    return totals, {"native": native, "matched_vs_socket_d64_ot": matched}


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_deterministic_device(args.device)
    if device.type == "mps" and args.allow_nondeterministic_mps:
        # PyTorch 2.9 has no deterministic MPS backward for the indexed
        # reductions used by cross-entropy and batch-hard triplet loss.  Keep
        # deterministic mode explicit but warn instead of aborting; this is a
        # one-shot pilot, not checkpoint selection over repeated MPS runs.
        torch.use_deterministic_algorithms(True, warn_only=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest protocol digest is invalid")
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    _, socket_lineage = load_checkpoint_with_lineage(
        args.socket_checkpoint,
        project_root=PROJECT_ROOT,
    )
    observed_socket_digest = names_digest(socket_lineage.filenames, sort_names=True)
    if (
        len(socket_lineage.filenames) != socket.lineage.exposed_count
        or observed_socket_digest != socket.lineage.exposed_digest
    ):
        raise ValueError("strict Socket checkpoint and recursive lineage disagree")
    excluded, exclusion_reports = _exclusion_lineage(
        args.exclude_report,
        socket_lineage.filenames,
    )
    records = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=tuple(sorted(excluded)),
        limit=args.train_sources + args.eval_sources,
        seed=args.seed,
        namespace=SELECTION_NAMESPACE,
    )
    train_records = records[: args.train_sources]
    eval_records = records[args.train_sources :]
    train_names = tuple(str(record["filename"]) for record in train_records)
    eval_names = tuple(str(record["filename"]) for record in eval_records)
    if set(train_names) & set(eval_names) or (set(train_names) | set(eval_names)) & excluded:
        raise RuntimeError("source-disjoint selection invariant failed")

    train_boards = _prepare_boards(
        train_records,
        targets=args.targets,
        grid=args.grid,
        seed=args.seed,
    )
    model = CorruptionAwareBorderEncoder(
        dimension=args.dimension,
        border=args.border,
    ).to(device)
    history, training_seconds = train_model(model, train_boards, args, device)
    model.eval()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "corruption_border_encoder.pt"
    exposed_names = tuple(sorted(set(train_names) | set(eval_names) | excluded))
    contract = {
        "architecture": "historical-e13-corruption-aware-border-encoder-v1",
        "historical_commits": ["a605814", "c0c3fec"],
        "grid": args.grid,
        "tile_size": 20,
        "border": args.border,
        "dimension": args.dimension,
        "temperature": 0.08,
        "triplet_margin": 0.12,
        "training_objective": (
            "corrupt full-candidate InfoNCE + .25 batch-hard triplet + "
            ".20 clean retrieval + .10 clean/corrupt cosine consistency"
        ),
        "whole_tile_or_board_context": False,
        "raw_tile_replacement": False,
        "matcher_only": True,
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "contract": contract,
            "selection": {
                "namespace": SELECTION_NAMESPACE,
                "train_filenames": list(train_names),
                "train_digest": names_digest(train_names),
                "lineage_train_filenames": sorted(train_names),
                "lineage_train_digest": names_digest(train_names, sort_names=True),
                "lineage_exposed_filenames": list(exposed_names),
                "lineage_exposed_digest": names_digest(exposed_names, sort_names=True),
            },
            "training_history": history,
        },
        checkpoint_path,
    )

    eval_boards = _prepare_boards(
        eval_records,
        targets=args.targets,
        grid=args.grid,
        seed=args.seed,
    )
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
    predictions = [
        freeze_prediction(
            model,
            socket,
            item,
            grid=args.grid,
            device=device,
        )
        for item in inputs
    ]
    frozen_array, frozen_metadata = _write_frozen(predictions, output_dir=output_dir)
    local, reciprocal = _aggregate_local(predictions, references)

    candidate_names = ("e13_raw", "e13_socket_ot_rank50")
    r1_gains = {
        name: float(local[name]["pooled_r1"]) - float(local["socket_d64_ot"]["pooled_r1"])
        for name in candidate_names
    }
    precision_gains = {
        name: float(reciprocal["matched_vs_socket_d64_ot"][name]["precision_gain"])
        for name in candidate_names
    }
    r1_gate = max(r1_gains.values()) >= R1_GATE_GAIN
    precision_gate = any(
        precision_gains[name] >= MATCHED_PRECISION_GATE_GAIN
        and reciprocal["matched_vs_socket_d64_ot"][name]["matched_coverage"]
        >= MIN_MATCHED_RECIPROCAL_COVERAGE
        for name in candidate_names
    )
    local_gate_passed = r1_gate or precision_gate
    report = {
        "experiment": "historical-e13-corruption-border-encoder-pilot-v1",
        "status": (
            "local-gate-passed-global-eligible"
            if local_gate_passed
            else "local-gate-failed-stop-no-global-decoder"
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
            "known_synthetic_shuffle": True,
            "eval_draws_per_source": 1,
            "all_candidates_per_query": args.grid * args.grid,
            "predictions_frozen_before_exact_reference_scoring": True,
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "torch_deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "socket_checkpoint_and_prior_exact_lineages_excluded": True,
            "excluded_filename_count": len(excluded),
            "competition_test_opened": False,
            "global_decoder_run": False,
            "global_decoder_policy": "forbidden unless the fixed local gate passes",
            "local_gate": {
                "r1": f"best candidate pooled R1 >= frozen d64 OT + {R1_GATE_GAIN}",
                "reciprocal": (
                    "candidate matched-coverage reciprocal precision >= d64 OT + "
                    f"{MATCHED_PRECISION_GATE_GAIN}, coverage >= "
                    f"{MIN_MATCHED_RECIPROCAL_COVERAGE}"
                ),
            },
        },
        "selection": {
            "train_filenames": list(train_names),
            "train_digest": names_digest(train_names),
            "eval_filenames": list(eval_names),
            "eval_digest": names_digest(eval_names),
            "excluded_reports": exclusion_reports,
            "excluded_filename_digest": names_digest(tuple(sorted(excluded))),
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "socket_checkpoint_sha256": socket.sha256,
            "frozen_predictions": str(frozen_array),
            "frozen_predictions_sha256": sha256_file(frozen_array),
            "frozen_metadata": str(frozen_metadata),
            "frozen_metadata_sha256": sha256_file(frozen_metadata),
        },
        "runtime_seconds": {
            "training": training_seconds,
            "e13_inference_sum": float(
                sum(item.inference_seconds["e13"] for item in predictions)
            ),
            "socket_d64_inference_sum": float(
                sum(item.inference_seconds["socket_d64"] for item in predictions)
            ),
        },
        "training_history": history,
        "local": local,
        "reciprocal": reciprocal,
        "gate": {
            "r1_gains": r1_gains,
            "precision_gains": precision_gains,
            "r1_gate_passed": r1_gate,
            "precision_gate_passed": precision_gate,
            "local_gate_passed": local_gate_passed,
        },
        "verdict": (
            "No global decoder was run in this local-only invocation. A passing gate only "
            "makes one separately frozen global diagnostic eligible; it does not promote E13."
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "local": local,
                "reciprocal": reciprocal,
                "gate": report["gate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
