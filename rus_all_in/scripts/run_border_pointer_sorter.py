#!/usr/bin/env python3
"""Bounded exact-shuffle experiment for the full-resolution BorderPointer sorter.

The pilot trains only on clean organizer-train targets outside the complete
frozen d64 checkpoint lineage.  Evaluation sources are disjoint from both
lineages.  Candidate and matched d64 decoder144 layouts are written to a
label-free artifact before exact inverse-shuffle references are opened.
"""

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

from aiijc_puzzle.border_pointer_sorter import BorderPointerSorter, border_pointer_loss
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import LoadedSocketCheckpoint, load_socket_checkpoint
from aiijc_puzzle.synthetic_socket_evaluation import (
    ExactSyntheticReference,
    SyntheticSocketInput,
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
DEFAULT_PREREGISTRATION = PROJECT_ROOT / "configs/border_pointer_preregistered_v1.json"
SELECTION_NAMESPACE = "aiijc-border-pointer-v1"
MAX_STEPS = 400
MAX_PARAMETERS = 10_000_000
FULL_GRID = 24


@dataclass(frozen=True)
class CleanBoard:
    filename: str
    target_sha256: str
    tiles: np.ndarray


@dataclass(frozen=True)
class TrainingCase:
    synthetic_input: SyntheticSocketInput
    reference: ExactSyntheticReference


@dataclass(frozen=True)
class FrozenPrediction:
    case_id: str
    source_filename: str
    draw_index: int
    corrupted_tiles_sha256: str
    candidate_layout: np.ndarray
    baseline_layout: np.ndarray
    baseline_decoder_report: dict[str, Any]
    runtime_seconds: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--socket-checkpoint", type=Path, default=DEFAULT_SOCKET_CHECKPOINT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("capacity", "benchmark", "pilot"), default="pilot")
    parser.add_argument("--train-sources", type=int, default=128)
    parser.add_argument("--eval-sources", type=int, default=16)
    parser.add_argument("--train-draws", type=int, default=1)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--feature-width", type=int, default=48)
    parser.add_argument("--feature-blocks", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--board-layers", type=int, default=4)
    parser.add_argument("--pointer-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--adjacency-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--exclude-report",
        type=Path,
        action="append",
        default=[],
        help="report/config whose explicitly declared PNG source filenames stay untouched",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.train_sources,
        args.train_draws,
        args.steps,
        args.feature_width,
        args.feature_blocks,
        args.dimension,
        args.heads,
        args.board_layers,
        args.pointer_layers,
        args.log_every,
    )
    if any(isinstance(value, bool) or value <= 0 for value in positive):
        raise ValueError("training, architecture, and logging arguments must be positive")
    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    if args.eval_sources < 0:
        raise ValueError("eval-sources must be non-negative")
    if args.dimension % args.heads:
        raise ValueError("dimension must be divisible by heads")
    for name in ("learning_rate", "weight_decay", "adjacency_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and non-negative")
    if args.learning_rate == 0:
        raise ValueError("learning-rate must be positive")
    if args.mode == "pilot":
        if not 128 <= args.train_sources <= 256:
            raise ValueError("pilot train-sources must be in [128, 256]")
        if args.eval_sources != 16 or args.train_draws != 1:
            raise ValueError("pilot is frozen to eval-sources=16 and train-draws=1")
    if args.mode == "capacity" and args.train_sources != 1:
        raise ValueError("capacity mode requires train-sources=1")
    if args.mode == "benchmark" and args.steps != 1:
        raise ValueError("benchmark mode requires steps=1")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def collect_declared_filenames(value: Any, *, parent_key: str = "") -> set[str]:
    """Collect explicit PNG filename rosters from an earlier report or config."""

    names: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            names.update(collect_declared_filenames(child, parent_key=str(key)))
    elif isinstance(value, list):
        if "filename" in parent_key:
            names.update(
                str(item)
                for item in value
                if isinstance(item, str) and item.lower().endswith(".png")
            )
        for item in value:
            if isinstance(item, (dict, list)):
                names.update(collect_declared_filenames(item))
    elif (
        isinstance(value, str)
        and "filename" in parent_key
        and value.lower().endswith(".png")
    ):
        names.add(value)
    return names


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def _crop_grid(tiles: np.ndarray, *, grid: int, filename: str, seed: int) -> np.ndarray:
    if grid == FULL_GRID:
        return np.ascontiguousarray(tiles)
    if not 2 <= grid < FULL_GRID:
        raise ValueError("experimental grid must be in [2, 24]")
    digest = hashlib.sha256(f"{filename}\0{seed}\0capacity-crop".encode()).digest()
    limit = FULL_GRID - grid + 1
    row = int.from_bytes(digest[:4], "little") % limit
    column = int.from_bytes(digest[4:8], "little") % limit
    board = tiles.reshape(FULL_GRID, FULL_GRID, 20, 20, 3)
    crop = board[row : row + grid, column : column + grid]
    return np.ascontiguousarray(crop.reshape(-1, 20, 20, 3))


def load_clean_boards(
    records: tuple[Any, ...],
    *,
    targets: Path,
    grid: int,
    seed: int,
) -> list[CleanBoard]:
    boards: list[CleanBoard] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        target_path = targets / filename
        observed = sha256_file(target_path)
        if observed != record.get("target_sha256"):
            raise ValueError(f"manifest target hash mismatch for {filename}")
        tiles = _crop_grid(
            split_tiles(_load_rgb(target_path)),
            grid=grid,
            filename=filename,
            seed=seed,
        )
        boards.append(CleanBoard(filename, observed, tiles))
        if index == 1 or index % 32 == 0 or index == len(records):
            print(f"loaded clean source {index}/{len(records)} {filename}", flush=True)
    return boards


def build_training_cases(
    boards: list[CleanBoard],
    *,
    draws: int,
    seed: int,
) -> list[TrainingCase]:
    cases: list[TrainingCase] = []
    for board in boards:
        for draw in range(draws):
            synthetic_input, reference = make_exact_synthetic_case(
                board.tiles,
                source_filename=board.filename,
                draw_index=draw,
                seed=seed,
            )
            cases.append(TrainingCase(synthetic_input, reference))
    return cases


def _tensor_tiles(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).permute(0, 3, 1, 2).unsqueeze(0).to(device)


def _tensor_layout(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value.astype(np.int64, copy=False)).unsqueeze(0).to(device)


def train_model(
    model: BorderPointerSorter,
    cases: list[TrainingCase],
    *,
    args: argparse.Namespace,
    device: torch.device,
    grid: int,
) -> tuple[list[dict[str, float]], float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.learning_rate * 0.1,
    )
    generator = np.random.default_rng(args.seed + 17)
    history: list[dict[str, float]] = []
    started = perf_counter()
    model.train()
    for step in range(args.steps):
        case = cases[int(generator.integers(len(cases)))]
        tiles = _tensor_tiles(case.synthetic_input.tiles, device)
        target = _tensor_layout(case.reference.tile_at_position, device)
        output = model(tiles, teacher_layout=target, grid=grid)
        loss, diagnostics = border_pointer_loss(
            output,
            target,
            grid=grid,
            adjacency_weight=args.adjacency_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        optimizer.step()
        scheduler.step()
        record = {
            "step": float(step + 1),
            **diagnostics,
            "grad_norm": grad_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": perf_counter() - started,
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
                        "pointer_nll": float(
                            np.mean([row["pointer_nll"] for row in recent])
                        ),
                        "teacher_pointer_accuracy": float(
                            np.mean([row["teacher_pointer_accuracy"] for row in recent])
                        ),
                        "right_r1": float(np.mean([row["right_r1"] for row in recent])),
                        "down_r1": float(np.mean([row["down_r1"] for row in recent])),
                        "elapsed_seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return history, perf_counter() - started


def _strict_layout(value: np.ndarray, *, count: int, name: str) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (count,) or not np.array_equal(np.sort(layout), np.arange(count)):
        raise RuntimeError(f"{name} is not a strict permutation")
    return np.ascontiguousarray(layout)


@torch.no_grad()
def freeze_predictions(
    model: BorderPointerSorter,
    socket: LoadedSocketCheckpoint,
    cases: list[TrainingCase],
    *,
    device: torch.device,
    grid: int,
) -> list[FrozenPrediction]:
    model.eval()
    count = grid * grid
    budget = min(144, count - grid)
    predictions: list[FrozenPrediction] = []
    for index, case in enumerate(cases, start=1):
        tiles = _tensor_tiles(case.synthetic_input.tiles, device)
        started = perf_counter()
        candidate = model.decode(tiles, grid=grid)[0].cpu().numpy()
        candidate_seconds = perf_counter() - started
        started = perf_counter()
        socket_output = socket.model(tiles, grid=grid)
        baseline = decode_socket_assignments(
            socket_output.right_log_assignment,
            socket_output.down_log_assignment,
            grid=grid,
            config=SocketDecoderConfig(
                component_edge_budget_per_axis=budget,
                swap_edge_budget_per_axis=budget,
                max_swap_steps=24,
            ),
        )
        baseline_seconds = perf_counter() - started
        predictions.append(
            FrozenPrediction(
                case_id=case.synthetic_input.case_id,
                source_filename=case.synthetic_input.source_filename,
                draw_index=case.synthetic_input.draw_index,
                corrupted_tiles_sha256=hashlib.sha256(
                    np.ascontiguousarray(case.synthetic_input.tiles).tobytes()
                ).hexdigest(),
                candidate_layout=_strict_layout(candidate, count=count, name="candidate"),
                baseline_layout=_strict_layout(
                    baseline.layout,
                    count=count,
                    name="baseline decoder",
                ),
                baseline_decoder_report=baseline.report(),
                runtime_seconds={
                    "candidate": candidate_seconds,
                    "baseline": baseline_seconds,
                },
            )
        )
        print(
            f"froze prediction {index}/{len(cases)} {case.synthetic_input.case_id}",
            flush=True,
        )
    return predictions


def write_frozen_predictions(
    predictions: list[FrozenPrediction],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        prefix = f"case_{index:04d}"
        arrays[f"{prefix}__border_pointer"] = prediction.candidate_layout
        arrays[f"{prefix}__socket_decoder144"] = prediction.baseline_layout
        cases.append(
            {
                "array_prefix": prefix,
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "draw_index": prediction.draw_index,
                "corrupted_tiles_sha256": prediction.corrupted_tiles_sha256,
                "candidate_layout_sha256": hashlib.sha256(
                    prediction.candidate_layout.astype("<i4").tobytes()
                ).hexdigest(),
                "baseline_layout_sha256": hashlib.sha256(
                    prediction.baseline_layout.astype("<i4").tobytes()
                ).hexdigest(),
                "baseline_decoder_report": prediction.baseline_decoder_report,
                "runtime_seconds": prediction.runtime_seconds,
            }
        )
    arrays_path = output_dir / "frozen_predictions.npz"
    np.savez_compressed(arrays_path, **arrays)
    metadata_path = output_dir / "frozen_predictions.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "aiijc-border-pointer-frozen-predictions-v1",
                "contains_exact_references": False,
                "contains_clean_pixels": False,
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return arrays_path, metadata_path


def _numeric_mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


@torch.no_grad()
def score_frozen_predictions(
    model: BorderPointerSorter,
    predictions: list[FrozenPrediction],
    cases: list[TrainingCase],
    *,
    device: torch.device,
    grid: int,
    adjacency_weight: float,
) -> dict[str, Any]:
    by_id = {case.reference.case_id: case for case in cases}
    if set(by_id) != {prediction.case_id for prediction in predictions}:
        raise ValueError("prediction/reference case identifiers differ")
    boards: list[dict[str, Any]] = []
    model.eval()
    for prediction in predictions:
        case = by_id[prediction.case_id]
        reference = case.reference.tile_at_position
        candidate = evaluate_layout(
            prediction.candidate_layout,
            reference,
            reference_is_exact=True,
        ).as_dict()
        baseline = evaluate_layout(
            prediction.baseline_layout,
            reference,
            reference_is_exact=True,
        ).as_dict()
        tiles = _tensor_tiles(case.synthetic_input.tiles, device)
        target = _tensor_layout(reference, device)
        output = model(tiles, teacher_layout=target, grid=grid)
        _, nll = border_pointer_loss(
            output,
            target,
            grid=grid,
            adjacency_weight=adjacency_weight,
        )
        boards.append(
            {
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "draw_index": prediction.draw_index,
                "candidate": candidate,
                "baseline": baseline,
                "teacher_forced_diagnostic": nll,
            }
        )
    candidate_mean = _numeric_mean([row["candidate"] for row in boards])
    baseline_mean = _numeric_mean([row["baseline"] for row in boards])
    nll_mean = _numeric_mean([row["teacher_forced_diagnostic"] for row in boards])
    keys = (
        "correct_tile_count",
        "direct_placement",
        "correct_row_count",
        "row_accuracy",
        "correct_column_count",
        "column_accuracy",
        "adjacency_correct",
        "adjacency",
    )
    delta = {key: candidate_mean[key] - baseline_mean[key] for key in keys}
    exact_positive = delta["correct_tile_count"] > 0
    exact_flat = abs(delta["correct_tile_count"]) < 1e-12
    positional_signal = (
        delta["correct_row_count"] > 0 or delta["correct_column_count"] > 0
    )
    descriptive_pass = exact_positive or (
        exact_flat and delta["adjacency"] >= -0.02 and positional_signal
    )
    return {
        "reference": "exact inverse deterministic shuffle opened after frozen artifact",
        "case_count": len(boards),
        "candidate_mean": candidate_mean,
        "baseline_mean": baseline_mean,
        "candidate_delta_vs_baseline": delta,
        "teacher_forced_diagnostic_mean": nll_mean,
        "strict_permutation_count": len(predictions),
        "descriptive_discovery_gate": {
            "pass": descriptive_pass,
            "exact_positive": exact_positive,
            "exact_flat": exact_flat,
            "row_or_column_positive": positional_signal,
            "adjacency_loss_at_most_2pp": delta["adjacency"] >= -0.02,
            "predicted_prefix_r1_arm_evaluated": False,
            "promotion_authorized": False,
        },
        "boards": boards,
    }


def _capacity_score(
    model: BorderPointerSorter,
    case: TrainingCase,
    *,
    device: torch.device,
    grid: int,
    adjacency_weight: float,
) -> dict[str, Any]:
    model.eval()
    tiles = _tensor_tiles(case.synthetic_input.tiles, device)
    target = _tensor_layout(case.reference.tile_at_position, device)
    with torch.no_grad():
        layout = model.decode(tiles, grid=grid)[0].cpu().numpy()
        output = model(tiles, teacher_layout=target, grid=grid)
        _, nll = border_pointer_loss(
            output,
            target,
            grid=grid,
            adjacency_weight=adjacency_weight,
        )
    metrics = evaluate_layout(
        layout,
        case.reference.tile_at_position,
        reference_is_exact=True,
    ).as_dict()
    return {
        "same_training_case_mechanical_check_only": True,
        "global": metrics,
        "teacher_forced": nll,
        "strict_permutation": bool(
            np.array_equal(np.sort(layout), np.arange(grid * grid))
        ),
        "pass": metrics["correct_tile_count"] == grid * grid,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    grid = 4 if args.mode == "capacity" else FULL_GRID
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    preregistration_hash = sha256_file(args.preregistration)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest is invalid")
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    external_exclusions: set[str] = set()
    exclusion_audit: list[dict[str, Any]] = []
    for path in args.exclude_report:
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = collect_declared_filenames(payload)
        external_exclusions.update(names)
        exclusion_audit.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "declared_filename_count": len(names),
                "declared_filename_digest": names_digest(sorted(names), sort_names=True),
            }
        )
    requested_eval = 0 if args.skip_evaluation or args.mode != "pilot" else args.eval_sources
    records = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=tuple(
            sorted(set(socket.lineage.exposed_filenames) | external_exclusions)
        ),
        limit=args.train_sources + requested_eval,
        seed=args.seed,
        namespace=SELECTION_NAMESPACE,
    )
    train_records = tuple(records[: args.train_sources])
    eval_records = tuple(records[args.train_sources :])
    train_names = tuple(str(record["filename"]) for record in train_records)
    eval_names = tuple(str(record["filename"]) for record in eval_records)
    if set(train_names) & set(eval_names) or set(eval_names) & set(
        socket.lineage.exposed_filenames
    ):
        raise RuntimeError("pointer evaluation is not source-disjoint")
    if (set(train_names) | set(eval_names)) & external_exclusions:
        raise RuntimeError("pointer selection overlaps an explicit external exclusion")
    selection_path = output_dir / "selection_commitment.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema": "aiijc-border-pointer-selection-v1",
                "seed": args.seed,
                "namespace": SELECTION_NAMESPACE,
                "socket_lineage_digest": socket.lineage.exposed_digest,
                "external_excluded_filenames": sorted(external_exclusions),
                "external_excluded_digest": names_digest(
                    sorted(external_exclusions), sort_names=True
                ),
                "fit_source_filenames": list(train_names),
                "fit_source_digest": names_digest(train_names),
                "evaluation_source_filenames": list(eval_names),
                "evaluation_source_digest": names_digest(eval_names),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"selection committed: {selection_path}", flush=True)

    clean_train = load_clean_boards(
        train_records,
        targets=args.targets.resolve(),
        grid=grid,
        seed=args.seed,
    )
    train_cases = build_training_cases(clean_train, draws=args.train_draws, seed=args.seed)
    model = BorderPointerSorter(
        socket_backbone=socket.model,
        feature_width=args.feature_width,
        feature_blocks=args.feature_blocks,
        dimension=args.dimension,
        heads=args.heads,
        board_layers=args.board_layers,
        pointer_layers=args.pointer_layers,
        max_grid=FULL_GRID,
        freeze_socket=True,
    ).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters > MAX_PARAMETERS:
        raise ValueError(
            f"trainable parameter cap exceeded: {trainable_parameters} > {MAX_PARAMETERS}"
        )
    history, training_seconds = train_model(
        model,
        train_cases,
        args=args,
        device=device,
        grid=grid,
    )
    checkpoint_path = output_dir / "border_pointer_sorter.pt"
    torch.save(
        {
            "schema": "aiijc-border-pointer-checkpoint-v1",
            "state_dict": model.state_dict(),
            "architecture": {
                "feature_width": args.feature_width,
                "feature_blocks": args.feature_blocks,
                "dimension": args.dimension,
                "heads": args.heads,
                "board_layers": args.board_layers,
                "pointer_layers": args.pointer_layers,
                "max_grid": FULL_GRID,
                "input_index_embedding": False,
                "strict_masked_pointer": True,
                "original_upright_tile_identities_only": True,
            },
            "socket_checkpoint_sha256": socket.sha256,
            "selection": {
                "train_filenames": list(train_names),
                "train_digest": names_digest(train_names),
                "lineage_exposed_filenames": sorted(
                    set(socket.lineage.exposed_filenames) | set(train_names)
                ),
                "lineage_exposed_digest": names_digest(
                    sorted(set(socket.lineage.exposed_filenames) | set(train_names)),
                    sort_names=True,
                ),
            },
        },
        checkpoint_path,
    )

    capacity = None
    evaluation = None
    frozen_artifact = None
    if args.mode == "capacity":
        capacity = _capacity_score(
            model,
            train_cases[0],
            device=device,
            grid=grid,
            adjacency_weight=args.adjacency_weight,
        )
    elif eval_records:
        clean_eval = load_clean_boards(
            eval_records,
            targets=args.targets.resolve(),
            grid=grid,
            seed=args.seed,
        )
        eval_cases = build_training_cases(clean_eval, draws=1, seed=args.seed + 100_000)
        predictions = freeze_predictions(
            model,
            socket,
            eval_cases,
            device=device,
            grid=grid,
        )
        arrays_path, metadata_path = write_frozen_predictions(
            predictions,
            output_dir=output_dir,
        )
        frozen_artifact = {
            "arrays_path": str(arrays_path),
            "arrays_sha256": sha256_file(arrays_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
        }
        print(f"frozen artifact committed: {arrays_path}", flush=True)
        evaluation = score_frozen_predictions(
            model,
            predictions,
            eval_cases,
            device=device,
            grid=grid,
            adjacency_weight=args.adjacency_weight,
        )

    report = {
        "experiment": "border-pointer-24-bounded-v1",
        "status": "capacity" if args.mode == "capacity" else "bounded-development",
        "mode": args.mode,
        "device": str(device),
        "grid": grid,
        "preregistration": {
            "path": str(args.preregistration.resolve()),
            "sha256": preregistration_hash,
            "payload": preregistration,
        },
        "socket_checkpoint": {
            "path": str(socket.path),
            "sha256": socket.sha256,
            "contract": socket.contract,
            "lineage": socket.lineage.as_dict(),
        },
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "manifest_split": "train",
            "clean_targets_only": True,
            "train_inputs_opened": False,
            "calibration_holdout_or_test_opened": False,
            "exact_known_shuffle": True,
            "challenge_like_corruption": "aiijc_puzzle.restoration_r6.distort_tiles",
            "prediction_output": "strict permutation of original upright tile identities",
            "input_index_embedding": False,
            "dirty_only_predictions_frozen_before_reference_scoring": evaluation is not None,
            "fresh_source64_draw2_opened": False,
        },
        "selection": {
            "selection_commitment_path": str(selection_path),
            "selection_commitment_sha256": sha256_file(selection_path),
            "fit_source_filenames": list(train_names),
            "fit_source_digest": names_digest(train_names),
            "evaluation_source_filenames": list(eval_names),
            "evaluation_source_digest": names_digest(eval_names),
            "source_disjoint": not bool(set(train_names) & set(eval_names)),
            "external_exclusion_audit": exclusion_audit,
            "external_exclusion_overlap": sorted(
                (set(train_names) | set(eval_names)) & external_exclusions
            ),
        },
        "architecture": {
            "feature_width": args.feature_width,
            "feature_blocks": args.feature_blocks,
            "dimension": args.dimension,
            "heads": args.heads,
            "board_layers": args.board_layers,
            "pointer_layers": args.pointer_layers,
            "trainable_parameters": trainable_parameters,
            "total_parameters_including_frozen_socket": total_parameters,
            "twenty_by_twenty_lattice_preserved": True,
            "ordered_perimeter_positions": 76,
            "conditional_left_up_evidence": True,
            "distance_to_border_unary": True,
        },
        "training": {
            "steps": args.steps,
            "fit_sources": args.train_sources,
            "draws_per_source": args.train_draws,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "adjacency_weight": args.adjacency_weight,
            "runtime_seconds": training_seconds,
            "history": history,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "capacity": capacity,
        "frozen_predictions": frozen_artifact,
        "evaluation": evaluation,
        "promotion_authorized": False,
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
                "capacity": capacity,
                "evaluation": evaluation,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
