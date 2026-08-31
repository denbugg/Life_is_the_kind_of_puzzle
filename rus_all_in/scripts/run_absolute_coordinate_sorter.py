#!/usr/bin/env python3
"""Train/evaluate a direct exact-coordinate sorter on source-disjoint puzzles.

Only clean images from the manifest's train split are used to make synthetic
challenge-like inputs with exact known shuffles.  A warm-started SocketMatcher
is a frozen dirty-visible backbone; the new head never sees organizer targets,
recovered layouts, calibration, holdout, test, or shuffled-index embeddings.
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
from torch.nn import functional as F

from aiijc_puzzle.absolute_coordinate_sorter import (
    AbsoluteCoordinateSorter,
    component_translation_loss,
    coordinate_sorting_loss,
    decode_coordinate_logits,
    train_consistent_component_unary,
    truth_consistent_component_targets,
)
from aiijc_puzzle.component_anchor_diagnostic import rebuild_decoder_components
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    collect_declared_source_filenames,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    decode_socket_assignments,
)
from aiijc_puzzle.socket_matcher import (
    BORDER_HEAD_EMBEDDING_V2,
    BORDER_HEAD_SCORE_STATS_V3,
    SocketMatcher,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
SELECTION_NAMESPACE = "aiijc-absolute-coordinate-sorter-v1"
GRID = 24
TILE_COUNT = GRID * GRID


@dataclass(frozen=True)
class CleanBoard:
    filename: str
    target_sha256: str
    tiles: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--socket-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--eval-limit", type=int, default=16)
    parser.add_argument("--eval-draws", type=int, default=1)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--head-dimension", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--sinkhorn-iterations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--assignment-weight", type=float, default=0.5)
    parser.add_argument(
        "--component-translation-weight",
        type=float,
        default=0.0,
        help="exact feasible-shift CE weight over truth-consistent predicted components",
    )
    parser.add_argument("--component-edge-budget", type=int, default=144)
    parser.add_argument(
        "--component-unary-weight",
        type=float,
        default=0.10,
        help="fixed train-consistent coordinate unary weight in a separate decoder arm",
    )
    parser.add_argument(
        "--historical-per-tile-zscore-comparator",
        action="store_true",
        help="report the old non-train-consistent per-tile z-score arm as diagnostic only",
    )
    parser.add_argument(
        "--cyclic-border5",
        action="store_true",
        help="also apply the independently confirmed cyclic-border5 tail to both socket arms",
    )
    parser.add_argument(
        "--axis-unary-diagnostics",
        action="store_true",
        help="evaluate row-only and column-only coordinate unary arms",
    )
    parser.add_argument(
        "--axis-unary-weight",
        dest="axis_unary_weights",
        action="append",
        type=float,
        default=[],
        help="bounded row-only/column-only diagnostic weight; may be repeated",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--exclude-report",
        type=Path,
        action="append",
        default=[],
        help="prior report whose train/eval/source panels are excluded before head training",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="training-only smoke/capacity run; does not open an exact evaluation source",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def names_digest(records: tuple[Any, ...]) -> str:
    names = "\n".join(str(record["filename"]) for record in records)
    return hashlib.sha256(names.encode()).hexdigest()


def load_socket_backbone(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[SocketMatcher, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = payload.get("contract", {})
    architecture = contract.get("architecture")
    border_version = {
        "board-conditioned-partial-socket-matcher-v1": BORDER_HEAD_EMBEDDING_V2,
        "board-conditioned-partial-socket-matcher-v2": BORDER_HEAD_EMBEDDING_V2,
        "board-conditioned-partial-socket-matcher-v3": BORDER_HEAD_SCORE_STATS_V3,
    }.get(architecture)
    if border_version is None:
        raise ValueError(f"unsupported socket checkpoint architecture: {architecture!r}")
    if contract.get("synthetic_grid") != GRID:
        raise ValueError("socket checkpoint must have been trained with full 24x24 synthetic grids")
    if contract.get("input_index_position_embedding") is not False:
        raise ValueError("socket checkpoint does not prove absence of shuffled-index embeddings")
    backbone = SocketMatcher(
        dimension=int(contract["dimension"]),
        heads=int(contract["heads"]),
        board_layers=int(contract["board_layers"]),
        socket_layers=int(contract["socket_layers"]),
        sinkhorn_iterations=int(contract["sinkhorn_iterations"]),
        border_head_version=border_version,
    )
    backbone.load_state_dict(payload["state_dict"])
    return backbone.to(device), payload


def select_source_disjoint_records(
    manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    train_limit: int,
    eval_limit: int,
    additional_forbidden: set[str] | None = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...], set[str]]:
    records = select_manifest_records(
        manifest,
        "train",
        limit=len(manifest["splits"]["train"]),
        namespace=SELECTION_NAMESPACE,
    )
    selection = checkpoint.get("selection", {})
    ancestral_train = selection.get(
        "lineage_train_filenames",
        selection.get("train_filenames", []),
    )
    ancestral_exposed = selection.get("lineage_exposed_filenames", ancestral_train)
    if not isinstance(ancestral_train, list) or not all(
        isinstance(name, str) for name in ancestral_train
    ):
        raise ValueError("socket checkpoint training lineage is malformed")
    if not isinstance(ancestral_exposed, list) or not all(
        isinstance(name, str) for name in ancestral_exposed
    ):
        raise ValueError("socket checkpoint exposure lineage is malformed")
    additional_forbidden = set() if additional_forbidden is None else additional_forbidden
    available = tuple(
        record for record in records if str(record["filename"]) not in additional_forbidden
    )
    train = tuple(available[:train_limit])
    forbidden = (
        set(ancestral_exposed)
        | additional_forbidden
        | {str(record["filename"]) for record in train}
    )
    evaluation = tuple(
        record for record in available[train_limit:] if str(record["filename"]) not in forbidden
    )[:eval_limit]
    if len(train) != train_limit or len(evaluation) != eval_limit:
        raise ValueError("could not form complete source-disjoint fit/evaluation panels")
    train_names = {str(record["filename"]) for record in train}
    eval_names = {str(record["filename"]) for record in evaluation}
    if train_names & eval_names or set(ancestral_exposed) & eval_names:
        raise RuntimeError("evaluation overlaps current or ancestral model exposure")
    return train, evaluation, set(ancestral_train)


def prepare_clean_boards(records: tuple[Any, ...], targets: Path) -> list[CleanBoard]:
    boards: list[CleanBoard] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        tiles = split_tiles(load_rgb(targets / filename))
        boards.append(CleanBoard(filename, str(record["target_sha256"]), tiles))
        if index == 1 or index % 64 == 0 or index == len(records):
            print(f"prepared clean source {index}/{len(records)} {filename}", flush=True)
    return boards


def _uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(shape, device=device).uniform_(low, high)


def challenge_augment(clean: torch.Tensor) -> torch.Tensor:
    """Match the currently declared SocketMatcher synthetic corruption contract."""

    count = len(clean)
    gray = 0.299 * clean[:, :1] + 0.587 * clean[:, 1:2] + 0.114 * clean[:, 2:3]
    pivot = gray.mean(dim=(1, 2, 3), keepdim=True)
    scale = _uniform((count, 1, 1, 1), 0.70, 1.30, device=clean.device)
    offset = _uniform((count, 1, 1, 1), -30 / 255, 30 / 255, device=clean.device)
    value = scale * (clean - pivot) + pivot + offset
    sigma = _uniform((count, 1, 1, 1), 40 / 255, 55 / 255, device=clean.device)
    value = value + sigma * torch.randn_like(value)
    kernel = value.new_tensor([0.25, 0.5, 0.25])
    horizontal = kernel.reshape(1, 1, 1, 3).expand(3, 1, 1, 3)
    vertical = kernel.reshape(1, 1, 3, 1).expand(3, 1, 3, 1)
    value = F.conv2d(F.pad(value, (1, 1, 0, 0), mode="reflect"), horizontal, groups=3)
    value = F.conv2d(F.pad(value, (0, 0, 1, 1), mode="reflect"), vertical, groups=3)
    levels = _uniform((count, 1, 1, 1), 40.0, 72.0, device=clean.device)
    return (torch.round(value.clamp(0, 1) * levels) / levels).clamp(0.0, 1.0)


def synthetic_example(
    board: CleanBoard,
    *,
    generator: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    clean = torch.from_numpy(board.tiles.astype(np.float32)).permute(0, 3, 1, 2).to(device)
    corrupted = challenge_augment(clean / 255.0)
    permutation = generator.permutation(TILE_COUNT).astype(np.int64)
    shuffled = corrupted[torch.from_numpy(permutation).to(device)]
    # Model rows follow shuffled input tiles: input tile i came from literal
    # row-major position permutation[i].
    target = torch.from_numpy(permutation.copy()).to(device)
    reference_tile_at_position = np.argsort(permutation).astype(np.int32)
    return shuffled.unsqueeze(0), target.unsqueeze(0), reference_tile_at_position


def train_model(
    model: AbsoluteCoordinateSorter,
    boards: list[CleanBoard],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        max(args.steps, 1),
        eta_min=args.learning_rate * 0.08,
    )
    generator = np.random.default_rng(args.seed + 1)
    history: list[dict[str, float]] = []
    started = perf_counter()
    model.train()
    for step in range(args.steps):
        board = boards[int(generator.integers(len(boards)))]
        tiles, target, _ = synthetic_example(board, generator=generator, device=device)
        output = model(tiles)
        coordinate_loss, diagnostics = coordinate_sorting_loss(
            output,
            target,
            grid=GRID,
            assignment_weight=args.assignment_weight,
        )
        if args.component_translation_weight > 0:
            component_build = rebuild_decoder_components(
                output.socket_output.right_log_assignment,
                output.socket_output.down_log_assignment,
                grid=GRID,
                edge_budget_per_axis=args.component_edge_budget,
            )
            component_targets = truth_consistent_component_targets(
                component_build.components,
                target,
                grid=GRID,
            )
            translation_loss, translation_diagnostics = component_translation_loss(
                output.slot_logits,
                component_targets,
            )
        else:
            translation_loss = output.slot_logits.sum() * 0.0
            translation_diagnostics = {
                "component_translation_nll": 0.0,
                "component_translation_uniform_nll": 0.0,
                "component_translation_nll_minus_uniform": 0.0,
                "component_translation_nll_ratio_to_uniform": 0.0,
                "component_translation_shift_top1_accuracy": 0.0,
                "component_translation_shift_chance_accuracy": 0.0,
                "supervised_component_count": 0.0,
                "supervised_component_tiles": 0.0,
                "maximum_supervised_component_size": 0.0,
                "mean_supervised_component_size": 0.0,
            }
        loss = coordinate_loss + args.component_translation_weight * translation_loss
        diagnostics["coordinate_loss"] = diagnostics["loss"]
        diagnostics["loss"] = float(loss.detach())
        diagnostics.update(translation_diagnostics)
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
                        "row_accuracy": float(
                            np.mean([row["row_argmax_accuracy"] for row in recent])
                        ),
                        "column_accuracy": float(
                            np.mean([row["column_argmax_accuracy"] for row in recent])
                        ),
                        "slot_accuracy": float(
                            np.mean([row["slot_argmax_accuracy"] for row in recent])
                        ),
                        "component_nll": float(
                            np.mean([row["component_translation_nll"] for row in recent])
                        ),
                        "components": float(
                            np.mean([row["supervised_component_count"] for row in recent])
                        ),
                        "component_tiles": float(
                            np.mean([row["supervised_component_tiles"] for row in recent])
                        ),
                        "elapsed_seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return history, perf_counter() - started


def _zscore_rows_historical(value: np.ndarray) -> np.ndarray:
    centred = value - value.mean(axis=1, keepdims=True)
    return centred / np.maximum(centred.std(axis=1, keepdims=True), 1e-6)


def _weight_tag(value: float) -> str:
    return f"{value:.6g}".replace(".", "p")


def _numeric_mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


@torch.no_grad()
def evaluate_exact(
    model: AbsoluteCoordinateSorter,
    records: tuple[Any, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], float]:
    model.eval()
    boards: list[dict[str, Any]] = []
    axis_weights = list(dict.fromkeys(float(value) for value in args.axis_unary_weights))
    if args.axis_unary_diagnostics and args.component_unary_weight not in axis_weights:
        axis_weights.append(float(args.component_unary_weight))
    started = perf_counter()
    for source_index, record in enumerate(records):
        board = prepare_clean_boards((record,), args.targets)[0]
        for draw in range(args.eval_draws):
            case_seed = args.seed + 100_000 + source_index * args.eval_draws + draw
            np_generator = np.random.default_rng(case_seed)
            torch.manual_seed(case_seed)
            tiles, target, reference = synthetic_example(
                board,
                generator=np_generator,
                device=device,
            )
            output = model(tiles)
            slot_logits = output.slot_logits[0].float().cpu().numpy()
            coordinate_layout = decode_coordinate_logits(slot_logits)
            decoder_config = SocketDecoderConfig(
                component_edge_budget_per_axis=144,
                swap_edge_budget_per_axis=144,
                max_swap_steps=24,
            )
            socket = decode_socket_assignments(
                output.socket_output.right_log_assignment,
                output.socket_output.down_log_assignment,
                grid=GRID,
                config=decoder_config,
            )
            variants: dict[str, np.ndarray] = {
                "coordinate_hungarian": coordinate_layout,
                "socket_ot_decoder144": socket.layout,
            }
            decoder_reports: dict[str, Any] = {"socket_ot_decoder144": socket.report()}
            axis_results: dict[str, Any] = {}
            if args.component_unary_weight > 0:
                anchored = decode_socket_assignments(
                    output.socket_output.right_log_assignment,
                    output.socket_output.down_log_assignment,
                    grid=GRID,
                    config=SocketDecoderConfig(
                        component_edge_budget_per_axis=144,
                        swap_edge_budget_per_axis=144,
                        max_swap_steps=24,
                        component_shift_unary_weight=args.component_unary_weight,
                    ),
                    component_shift_unary=train_consistent_component_unary(slot_logits),
                )
                primary_unary_name = (
                    "socket_ot_decoder144_coordinate_unary_train_consistent"
                )
                variants[primary_unary_name] = anchored.layout
                decoder_reports[primary_unary_name] = anchored.report()
                if args.historical_per_tile_zscore_comparator:
                    historical = decode_socket_assignments(
                        output.socket_output.right_log_assignment,
                        output.socket_output.down_log_assignment,
                        grid=GRID,
                        config=SocketDecoderConfig(
                            component_edge_budget_per_axis=144,
                            swap_edge_budget_per_axis=144,
                            max_swap_steps=24,
                            component_shift_unary_weight=args.component_unary_weight,
                        ),
                        component_shift_unary=_zscore_rows_historical(slot_logits),
                    )
                    historical_name = (
                        "socket_ot_decoder144_coordinate_unary_"
                        "per_tile_zscore_historical"
                    )
                    variants[historical_name] = historical.layout
                    decoder_reports[historical_name] = historical.report()
            if axis_weights:
                cells = np.arange(TILE_COUNT)
                row_unary = output.row_logits[0].float().cpu().numpy()[:, cells // GRID]
                column_unary = output.column_logits[0].float().cpu().numpy()[
                    :, cells % GRID
                ]
                for axis_name, axis_unary in (
                    ("row", row_unary),
                    ("column", column_unary),
                ):
                    for axis_weight in axis_weights:
                        axis_result = decode_socket_assignments(
                            output.socket_output.right_log_assignment,
                            output.socket_output.down_log_assignment,
                            grid=GRID,
                            config=SocketDecoderConfig(
                                component_edge_budget_per_axis=144,
                                swap_edge_budget_per_axis=144,
                                max_swap_steps=24,
                                component_shift_unary_weight=axis_weight,
                            ),
                            component_shift_unary=train_consistent_component_unary(
                                axis_unary
                            ),
                        )
                        name = (
                            f"socket_ot_decoder144_coordinate_{axis_name}_unary_"
                            f"train_consistent_w{_weight_tag(axis_weight)}"
                        )
                        variants[name] = axis_result.layout
                        decoder_reports[name] = axis_result.report()
                        axis_results[name] = axis_result

            if args.cyclic_border5:
                base_cyclic = select_global_cyclic_translation(
                    socket.layout,
                    output.socket_output.right_log_assignment,
                    output.socket_output.down_log_assignment,
                    grid=GRID,
                    config=CyclicTranslationConfig(border_weight=5.0),
                )
                variants["socket_ot_decoder144_cyclic_border5"] = base_cyclic.layout
                decoder_reports["socket_ot_decoder144_cyclic_border5"] = base_cyclic.report()
                if args.component_unary_weight > 0:
                    anchored_cyclic = select_global_cyclic_translation(
                        anchored.layout,
                        output.socket_output.right_log_assignment,
                        output.socket_output.down_log_assignment,
                        grid=GRID,
                        config=CyclicTranslationConfig(border_weight=5.0),
                    )
                    variants[
                        "socket_ot_decoder144_coordinate_unary_"
                        "train_consistent_cyclic_border5"
                    ] = anchored_cyclic.layout
                    decoder_reports[
                        "socket_ot_decoder144_coordinate_unary_"
                        "train_consistent_cyclic_border5"
                    ] = anchored_cyclic.report()
                for axis_name, axis_result in axis_results.items():
                    axis_cyclic = select_global_cyclic_translation(
                        axis_result.layout,
                        output.socket_output.right_log_assignment,
                        output.socket_output.down_log_assignment,
                        grid=GRID,
                        config=CyclicTranslationConfig(border_weight=5.0),
                    )
                    cyclic_name = f"{axis_name}_cyclic_border5"
                    variants[cyclic_name] = axis_cyclic.layout
                    decoder_reports[cyclic_name] = axis_cyclic.report()

            target_numpy = target[0].cpu().numpy()
            component_build = rebuild_decoder_components(
                output.socket_output.right_log_assignment,
                output.socket_output.down_log_assignment,
                grid=GRID,
                edge_budget_per_axis=144,
            )
            component_targets = truth_consistent_component_targets(
                component_build.components,
                target,
                grid=GRID,
            )
            _, component_diagnostics = component_translation_loss(
                output.slot_logits,
                component_targets,
            )
            row_target = target_numpy // GRID
            column_target = target_numpy % GRID
            classifier = {
                "row_argmax_correct": int(
                    np.count_nonzero(output.row_logits[0].argmax(1).cpu().numpy() == row_target)
                ),
                "row_argmax_accuracy": float(
                    np.mean(output.row_logits[0].argmax(1).cpu().numpy() == row_target)
                ),
                "column_argmax_correct": int(
                    np.count_nonzero(
                        output.column_logits[0].argmax(1).cpu().numpy() == column_target
                    )
                ),
                "column_argmax_accuracy": float(
                    np.mean(output.column_logits[0].argmax(1).cpu().numpy() == column_target)
                ),
                "slot_argmax_correct": int(
                    np.count_nonzero(output.slot_logits[0].argmax(1).cpu().numpy() == target_numpy)
                ),
                "slot_argmax_accuracy": float(
                    np.mean(output.slot_logits[0].argmax(1).cpu().numpy() == target_numpy)
                ),
            }
            board_metrics = {
                name: evaluate_layout(layout, reference, reference_is_exact=True).as_dict()
                for name, layout in variants.items()
            }
            strict_permutation = {
                name: bool(
                    layout.shape == (TILE_COUNT,)
                    and np.array_equal(np.sort(layout), np.arange(TILE_COUNT))
                )
                for name, layout in variants.items()
            }
            if not all(strict_permutation.values()):
                raise RuntimeError("an exact-evaluation decoder returned a non-permutation")
            case_id = hashlib.sha256(
                f"{board.filename}\0{draw}\0{case_seed}".encode()
            ).hexdigest()[:16]
            boards.append(
                {
                    "case_id": f"absolute-coordinate-{case_id}",
                    "source_filename": board.filename,
                    "target_sha256": board.target_sha256,
                    "draw_index": draw,
                    "case_seed": case_seed,
                    "classifier": classifier,
                    "component_translation": component_diagnostics,
                    "global": board_metrics,
                    "strict_permutation": strict_permutation,
                    "decoder_reports": decoder_reports,
                    "coordinate_layout_sha256": hashlib.sha256(
                        coordinate_layout.astype("<i4").tobytes()
                    ).hexdigest(),
                }
            )
            print(
                f"evaluated exact {len(boards)}/{len(records) * args.eval_draws} "
                f"{board.filename} draw={draw}",
                flush=True,
            )
    classifier_mean = _numeric_mean([board["classifier"] for board in boards])
    component_translation_mean = _numeric_mean(
        [board["component_translation"] for board in boards]
    )
    variant_names = list(boards[0]["global"])
    global_mean = {
        name: _numeric_mean([board["global"][name] for board in boards])
        for name in variant_names
    }
    totals = {
        name: {
            "correct_tile_count_total": int(
                sum(board["global"][name]["correct_tile_count"] for board in boards)
            ),
            "correct_row_count_total": int(
                sum(board["global"][name]["correct_row_count"] for board in boards)
            ),
            "correct_column_count_total": int(
                sum(board["global"][name]["correct_column_count"] for board in boards)
            ),
        }
        for name in variant_names
    }
    chance = {
        "correct_tile_count_per_board": 1.0,
        "direct_placement": 1.0 / TILE_COUNT,
        "correct_row_count_per_board": float(GRID),
        "row_accuracy": 1.0 / GRID,
        "correct_column_count_per_board": float(GRID),
        "column_accuracy": 1.0 / GRID,
    }
    return {
        "reference": "exact known synthetic shuffle; no target-assisted layout recovery",
        "case_count": len(boards),
        "classifier_mean": classifier_mean,
        "component_translation_mean": component_translation_mean,
        "global_mean": global_mean,
        "global_totals": totals,
        "theoretical_random_bijection": chance,
        "boards": boards,
    }, perf_counter() - started


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.train_limit,
        args.eval_limit,
        args.eval_draws,
        args.steps,
        args.head_dimension,
        args.heads,
        args.set_layers,
        args.sinkhorn_iterations,
        args.log_every,
        args.component_edge_budget,
    )
    if min(positive) <= 0:
        raise ValueError("limits, dimensions, layers and steps must be positive")
    if args.head_dimension % args.heads:
        raise ValueError("head-dimension must be divisible by heads")
    if not 1 <= args.component_edge_budget <= TILE_COUNT - GRID:
        raise ValueError(f"component-edge-budget must be in [1, {TILE_COUNT - GRID}]")
    for name in (
        "learning_rate",
        "weight_decay",
        "assignment_weight",
        "component_translation_weight",
        "component_unary_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and non-negative")
    if any(not math.isfinite(value) or value <= 0 for value in args.axis_unary_weights):
        raise ValueError("axis-unary-weight values must be finite and positive")
    if len(set(args.axis_unary_weights)) != len(args.axis_unary_weights):
        raise ValueError("axis-unary-weight values must be unique")
    if args.learning_rate == 0:
        raise ValueError("learning-rate must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    backbone, socket_checkpoint = load_socket_backbone(args.socket_checkpoint, device)
    additional_forbidden: set[str] = set()
    exclude_audit: list[dict[str, Any]] = []
    for path in args.exclude_report:
        declared = collect_declared_source_filenames(
            json.loads(path.read_text(encoding="utf-8"))
        )
        additional_forbidden.update(declared)
        exclude_audit.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "declared_source_count": len(declared),
            }
        )
    train_records, eval_records, ancestral_train = select_source_disjoint_records(
        manifest,
        socket_checkpoint,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        additional_forbidden=additional_forbidden,
    )
    train_boards = prepare_clean_boards(train_records, args.targets)
    model = AbsoluteCoordinateSorter(
        backbone,
        grid=GRID,
        head_dimension=args.head_dimension,
        heads=args.heads,
        set_layers=args.set_layers,
        sinkhorn_iterations=args.sinkhorn_iterations,
        freeze_backbone=True,
    ).to(device)
    history, training_seconds = train_model(model, train_boards, args, device)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "absolute_coordinate_sorter.pt"
    current_train_names = {str(record["filename"]) for record in train_records}
    if current_train_names & additional_forbidden:
        raise RuntimeError("training panel overlaps a declared exclude report")
    eval_names = (
        set()
        if args.skip_evaluation
        else {str(record["filename"]) for record in eval_records}
    )
    lineage_train = ancestral_train | current_train_names
    old_exposed = set(
        socket_checkpoint.get("selection", {}).get(
            "lineage_exposed_filenames",
            socket_checkpoint.get("selection", {}).get("train_filenames", []),
        )
    )
    lineage_exposed = old_exposed | additional_forbidden | current_train_names | eval_names
    contract = {
        "architecture": "socket-backed-absolute-coordinate-sorter-v1",
        "grid": GRID,
        "head_dimension": args.head_dimension,
        "heads": args.heads,
        "set_layers": args.set_layers,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "frozen_socket_backbone": True,
        "input_index_position_embedding": False,
        "output_coordinate_queries": True,
        "training_target": "exact input-tile-to-literal-slot permutation",
        "strict_decoder": "square Hungarian tile-to-slot projection",
        "component_translation_training": args.component_translation_weight > 0,
        "component_translation_weight": args.component_translation_weight,
        "component_edge_budget_per_axis": args.component_edge_budget,
        "component_target_policy": (
            "predicted decoder144 components with >=2 tiles; supervise exact feasible shift "
            "only when all synthetic-truth relative coordinates agree"
        ),
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "contract": contract,
            "socket_checkpoint": {
                "path": str(args.socket_checkpoint.resolve()),
                "sha256": sha256_file(args.socket_checkpoint),
                "contract": socket_checkpoint["contract"],
            },
            "selection": {
                "namespace": SELECTION_NAMESPACE,
                "train_filenames": [record["filename"] for record in train_records],
                "train_digest": names_digest(train_records),
                "lineage_train_filenames": sorted(lineage_train),
                "lineage_exposed_filenames": sorted(lineage_exposed),
            },
            "training_history": history,
        },
        checkpoint_path,
    )
    if args.skip_evaluation:
        evaluation, evaluation_seconds = None, 0.0
    else:
        evaluation, evaluation_seconds = evaluate_exact(model, eval_records, args, device)
    report = {
        "experiment": contract["architecture"],
        "status": "pilot-research-candidate-not-production",
        "hypothesis": (
            "direct board-conditioned coordinate supervision supplies the absolute gauge "
            "that a local-neighbour decoder lacks"
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
            "selection_namespace": SELECTION_NAMESPACE,
            "manifest_train_split_only": True,
            "coordinate_head_training_uses_clean_sources_only": True,
            "challenge_like_corruption_applied_before_model_input": True,
            "train_eval_source_disjoint": True,
            "warmstart_exposure_lineage_eval_source_disjoint": True,
            "calibration_opened": False,
            "holdout_opened": False,
            "competition_test_opened": False,
            "exact_truth_source": "known synthetic shuffle",
            "additional_prior_report_sources_excluded": len(additional_forbidden),
            "declared_exclude_report_audit": exclude_audit,
            "evaluation_skipped": args.skip_evaluation,
        },
        "selection": {
            "train_filenames": [record["filename"] for record in train_records],
            "train_digest": names_digest(train_records),
            "eval_filenames": (
                [] if args.skip_evaluation else [record["filename"] for record in eval_records]
            ),
            "eval_digest": None if args.skip_evaluation else names_digest(eval_records),
            "lineage_train_count": len(lineage_train),
            "lineage_exposed_count_after_evaluation": len(lineage_exposed),
        },
        "model": {
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "socket_checkpoint_sha256": sha256_file(args.socket_checkpoint),
        },
        "runtime_seconds": {
            "training": training_seconds,
            "exact_evaluation": evaluation_seconds,
        },
        "training_history": history,
        "evaluation": evaluation,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "classifier": None if evaluation is None else evaluation["classifier_mean"],
                "global": None if evaluation is None else evaluation["global_mean"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
