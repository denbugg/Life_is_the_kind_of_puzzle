#!/usr/bin/env python3
"""Bounded transpose-equivariant continuation of the frozen coordinate head.

This experiment reuses an already-opened exact synthetic panel for development.
It never opens the competition test, calibration, holdout, or a fresh exact
panel.  Transposition is a model-only view; every decoded layout remains a
strict permutation of the original upright input tiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from run_absolute_coordinate_sorter import (
    DEFAULT_MANIFEST,
    DEFAULT_TARGETS,
    GRID,
    TILE_COUNT,
    CleanBoard,
    load_socket_backbone,
    prepare_clean_boards,
    synthetic_example,
)
from torch.nn import functional as F

from aiijc_puzzle.absolute_coordinate_sorter import (
    AbsoluteCoordinateSorter,
    coordinate_sorting_loss,
    decode_coordinate_logits,
    train_consistent_component_unary,
)
from aiijc_puzzle.coordinate_transpose import (
    FusedCoordinateLogits,
    collect_transpose_coordinate_views,
    fuse_transpose_coordinate_views,
    symmetric_axis_consistency_loss,
    transpose_positions,
)
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import (
    collect_declared_source_filenames,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "absolute-coordinate-sorter"
    / "component-translation-scale-d64-head32-train2048-s1600"
    / "absolute_coordinate_sorter.pt"
)
DEFAULT_REPLAY = (
    PROJECT_ROOT
    / "outputs"
    / "absolute-coordinate-sorter"
    / "component-translation-scale-confirm-source64-draw2"
    / "report.json"
)
SELECTION_NAMESPACE = "aiijc-coordinate-transpose-continuation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reuse-panel-report", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--report-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=192)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--assignment-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--component-unary-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--skip-development-replay",
        action="store_true",
        help="train and save only; useful for a smoke or capacity check",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.train_limit <= 512:
        raise ValueError("train-limit must be in [1, 512]")
    if not 1 <= args.steps <= 400:
        raise ValueError("steps must be in [1, 400]")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    for name in (
        "learning_rate",
        "weight_decay",
        "assignment_weight",
        "consistency_weight",
        "component_unary_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and non-negative")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")


def _names_digest(names: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def load_coordinate_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[AbsoluteCoordinateSorter, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = checkpoint.get("contract", {})
    if contract.get("architecture") != "socket-backed-absolute-coordinate-sorter-v1":
        raise ValueError("unsupported absolute coordinate checkpoint architecture")
    if contract.get("input_index_position_embedding") is not False:
        raise ValueError("coordinate checkpoint does not prove input-index equivariance")
    socket_metadata = checkpoint.get("socket_checkpoint", {})
    socket_path = Path(str(socket_metadata.get("path", "")))
    if not socket_path.is_file():
        raise FileNotFoundError(f"socket checkpoint is unavailable: {socket_path}")
    if sha256_file(socket_path) != socket_metadata.get("sha256"):
        raise ValueError("socket checkpoint digest differs from coordinate lineage")
    backbone, _ = load_socket_backbone(socket_path, device)
    model = AbsoluteCoordinateSorter(
        backbone,
        grid=int(contract["grid"]),
        head_dimension=int(contract["head_dimension"]),
        heads=int(contract["heads"]),
        set_layers=int(contract["set_layers"]),
        sinkhorn_iterations=int(contract["sinkhorn_iterations"]),
        freeze_backbone=True,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint


def collect_recursive_report_exposure(
    report_root: Path,
) -> tuple[set[str], dict[str, Any]]:
    """Collect every recursively declared source panel under one report root."""

    if not report_root.is_dir():
        raise FileNotFoundError(f"report root is unavailable: {report_root}")
    names: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in sorted(report_root.rglob("report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        declared = collect_declared_source_filenames(payload)
        names.update(declared)
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "declared_source_count": len(declared),
            }
        )
    digest_payload = "\n".join(
        f"{row['path']}\0{row['sha256']}\0{row['declared_source_count']}" for row in rows
    )
    return names, {
        "report_root": str(report_root.resolve()),
        "report_count": len(rows),
        "distinct_declared_source_count": len(names),
        "report_inventory_sha256": hashlib.sha256(digest_payload.encode()).hexdigest(),
        "reports": rows,
    }


def select_train_and_replay_records(
    manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    replay_report: dict[str, Any],
    recursive_report_names: set[str],
    *,
    train_limit: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...], set[str]]:
    """Select continuation sources after excluding all ancestral/opened panels."""

    selection = checkpoint.get("selection", {})
    checkpoint_exposed = selection.get(
        "lineage_exposed_filenames",
        selection.get("train_filenames", []),
    )
    if not isinstance(checkpoint_exposed, list) or not all(
        isinstance(name, str) for name in checkpoint_exposed
    ):
        raise ValueError("checkpoint exposure lineage is malformed")
    replay_names = replay_report.get("selection", {}).get("eval_filenames")
    if not isinstance(replay_names, list) or not all(
        isinstance(name, str) for name in replay_names
    ):
        raise ValueError("reuse-panel report has malformed eval_filenames")
    ranked = select_manifest_records(
        manifest,
        "train",
        limit=len(manifest["splits"]["train"]),
        namespace=SELECTION_NAMESPACE,
    )
    by_name = {str(record["filename"]): record for record in ranked}
    if any(name not in by_name for name in replay_names):
        raise ValueError("reuse-panel report contains a source outside the train split")
    forbidden = set(checkpoint_exposed) | recursive_report_names | set(replay_names)
    train = tuple(
        record for record in ranked if str(record["filename"]) not in forbidden
    )[:train_limit]
    if len(train) != train_limit:
        raise ValueError(
            f"only {len(train)} recursively unexposed train sources remain; "
            f"requested {train_limit}"
        )
    train_names = {str(record["filename"]) for record in train}
    if train_names & forbidden:
        raise RuntimeError("continuation training overlaps recursive exposure lineage")
    replay = tuple(by_name[name] for name in replay_names)
    return train, replay, forbidden


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in keys
    }


def train_continuation(
    model: AbsoluteCoordinateSorter,
    boards: list[CleanBoard],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], float]:
    """Dual-view supervised continuation with a small symmetric-KL penalty."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.steps,
        eta_min=args.learning_rate * 0.08,
    )
    generator = np.random.default_rng(args.seed + 1)
    history: list[dict[str, float]] = []
    started = perf_counter()
    model.train()
    for step in range(args.steps):
        board = boards[int(generator.integers(len(boards)))]
        # CPU corruption keeps the synthetic input contract independent of the
        # selected accelerator; only model work is moved to the device.
        tiles, target, _ = synthetic_example(
            board,
            generator=generator,
            device=torch.device("cpu"),
        )
        tiles = tiles.to(device)
        target = target.to(device)
        views = collect_transpose_coordinate_views(model, tiles)
        original_loss, original_diagnostics = coordinate_sorting_loss(
            views.original,
            target,
            grid=GRID,
            assignment_weight=args.assignment_weight,
        )
        transposed_target = transpose_positions(target, grid=GRID)
        transposed_loss, transposed_diagnostics = coordinate_sorting_loss(
            views.transposed,
            transposed_target,
            grid=GRID,
            assignment_weight=args.assignment_weight,
        )
        consistency = symmetric_axis_consistency_loss(views)
        supervised = 0.5 * (original_loss + transposed_loss)
        loss = supervised + args.consistency_weight * consistency
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        optimizer.step()
        scheduler.step()
        history.append(
            {
                "step": float(step + 1),
                "loss": float(loss.detach()),
                "supervised_loss": float(supervised.detach()),
                "consistency_loss": float(consistency.detach()),
                "original_row_accuracy": original_diagnostics[
                    "row_argmax_accuracy"
                ],
                "original_column_accuracy": original_diagnostics[
                    "column_argmax_accuracy"
                ],
                "transposed_row_accuracy": transposed_diagnostics[
                    "row_argmax_accuracy"
                ],
                "transposed_column_accuracy": transposed_diagnostics[
                    "column_argmax_accuracy"
                ],
                "grad_norm": grad_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            recent = history[-min(args.log_every, len(history)) :]
            recent_mean = _mean(recent)
            recent_mean.pop("step", None)
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step + 1,
                        **recent_mean,
                        "elapsed_seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return history, perf_counter() - started


def _original_coordinate_logits(output: Any) -> FusedCoordinateLogits:
    row_logits = F.log_softmax(output.row_logits, dim=-1)
    column_logits = F.log_softmax(output.column_logits, dim=-1)
    cells = torch.arange(TILE_COUNT, device=row_logits.device)
    slot_logits = row_logits[:, :, cells // GRID] + column_logits[:, :, cells % GRID]
    return FusedCoordinateLogits(row_logits, column_logits, slot_logits)


def _classifier_metrics(
    logits: FusedCoordinateLogits,
    target: np.ndarray,
) -> dict[str, float]:
    rows = target // GRID
    columns = target % GRID
    predicted_rows = logits.row_logits[0].argmax(1).cpu().numpy()
    predicted_columns = logits.column_logits[0].argmax(1).cpu().numpy()
    predicted_slots = logits.slot_logits[0].argmax(1).cpu().numpy()
    return {
        "row_argmax_correct": float(np.count_nonzero(predicted_rows == rows)),
        "column_argmax_correct": float(np.count_nonzero(predicted_columns == columns)),
        "slot_argmax_correct": float(np.count_nonzero(predicted_slots == target)),
    }


@torch.no_grad()
def evaluate_reused_panel(
    model: AbsoluteCoordinateSorter,
    records: tuple[Any, ...],
    replay_report: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    *,
    stage: str,
) -> tuple[dict[str, Any], float]:
    model.eval()
    replay_configuration = replay_report.get("configuration", {})
    draws = int(replay_configuration.get("eval_draws", 0))
    replay_seed = int(replay_configuration.get("seed", -1))
    if draws <= 0 or replay_seed < 0:
        raise ValueError("reuse-panel report has malformed draw/seed configuration")
    boards: list[dict[str, Any]] = []
    started = perf_counter()
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=144,
        swap_edge_budget_per_axis=144,
        max_swap_steps=24,
        component_shift_unary_weight=args.component_unary_weight,
    )
    for source_index, record in enumerate(records):
        board = prepare_clean_boards((record,), args.targets)[0]
        for draw in range(draws):
            case_seed = replay_seed + 100_000 + source_index * draws + draw
            np_generator = np.random.default_rng(case_seed)
            torch.manual_seed(case_seed)
            tiles, target, reference = synthetic_example(
                board,
                generator=np_generator,
                device=torch.device("cpu"),
            )
            tiles = tiles.to(device)
            target = target.to(device)
            views = collect_transpose_coordinate_views(model, tiles)
            logits = {
                "original": _original_coordinate_logits(views.original),
                "transpose_symmetric": fuse_transpose_coordinate_views(
                    views,
                    grid=GRID,
                    mode="symmetric",
                ),
                "transpose_row_teacher": fuse_transpose_coordinate_views(
                    views,
                    grid=GRID,
                    mode="row-teacher",
                ),
            }
            socket = decode_socket_assignments(
                views.original.socket_output.right_log_assignment,
                views.original.socket_output.down_log_assignment,
                grid=GRID,
                config=SocketDecoderConfig(
                    component_edge_budget_per_axis=144,
                    swap_edge_budget_per_axis=144,
                    max_swap_steps=24,
                ),
            )
            layouts: dict[str, np.ndarray] = {"socket_ot_decoder144": socket.layout}
            classifier = {
                name: _classifier_metrics(value, target[0].cpu().numpy())
                for name, value in logits.items()
            }
            for name, value in logits.items():
                layouts[f"coordinate_{name}"] = decode_coordinate_logits(
                    value.slot_logits
                )
                anchored = decode_socket_assignments(
                    views.original.socket_output.right_log_assignment,
                    views.original.socket_output.down_log_assignment,
                    grid=GRID,
                    config=decoder_config,
                    component_shift_unary=train_consistent_component_unary(
                        value.slot_logits[0].float().cpu().numpy()
                    ),
                )
                layouts[f"socket_unary_{name}"] = anchored.layout
            strict = {
                name: bool(
                    layout.shape == (TILE_COUNT,)
                    and np.array_equal(np.sort(layout), np.arange(TILE_COUNT))
                )
                for name, layout in layouts.items()
            }
            if not all(strict.values()):
                raise RuntimeError("transpose development decoder returned a non-permutation")
            case_id = hashlib.sha256(
                f"{board.filename}\0{draw}\0{case_seed}".encode()
            ).hexdigest()[:16]
            boards.append(
                {
                    "case_id": f"absolute-coordinate-{case_id}",
                    "source_filename": board.filename,
                    "draw_index": draw,
                    "case_seed": case_seed,
                    "classifier": classifier,
                    "global": {
                        name: evaluate_layout(
                            layout,
                            reference,
                            reference_is_exact=True,
                        ).as_dict()
                        for name, layout in layouts.items()
                    },
                    "strict_permutation": strict,
                    "layout_sha256": {
                        name: hashlib.sha256(
                            layout.astype("<i4").tobytes()
                        ).hexdigest()
                        for name, layout in layouts.items()
                    },
                }
            )
            print(
                f"evaluated {stage} reused exact {len(boards)}/{len(records) * draws} "
                f"{board.filename} draw={draw}",
                flush=True,
            )
    classifier_names = tuple(boards[0]["classifier"])
    global_names = tuple(boards[0]["global"])
    return {
        "stage": stage,
        "case_count": len(boards),
        "source_count": len(records),
        "draws_per_source": draws,
        "classifier_mean": {
            name: _mean([board["classifier"][name] for board in boards])
            for name in classifier_names
        },
        "global_mean": {
            name: _mean([board["global"][name] for board in boards])
            for name in global_names
        },
        "boards": boards,
    }, perf_counter() - started


def _paired_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_path: tuple[str, ...],
    candidate_path: tuple[str, ...],
    metric: str,
    seed: int,
    samples: int = 100_000,
) -> dict[str, Any]:
    candidate_by_id = {board["case_id"]: board for board in candidate["boards"]}
    grouped: dict[str, list[float]] = defaultdict(list)
    case_deltas: list[float] = []
    for board in baseline["boards"]:
        other = candidate_by_id.get(board["case_id"])
        if other is None:
            raise ValueError("candidate stage does not reproduce every baseline case")
        first: Any = board
        second: Any = other
        for key in baseline_path:
            first = first[key]
        for key in candidate_path:
            second = second[key]
        delta = float(second[metric]) - float(first[metric])
        case_deltas.append(delta)
        grouped[str(board["source_filename"])].append(delta)
    source_deltas = np.asarray(
        [np.mean(values) for values in grouped.values()],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(source_deltas),
        size=(samples, len(source_deltas)),
    )
    bootstrap = source_deltas[indices].mean(axis=1)
    return {
        "mean_delta_per_board": float(np.mean(case_deltas)),
        "source_cluster_bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "case_wins": int(np.count_nonzero(np.asarray(case_deltas) > 0)),
        "case_ties": int(np.count_nonzero(np.asarray(case_deltas) == 0)),
        "case_losses": int(np.count_nonzero(np.asarray(case_deltas) < 0)),
    }


def select_development_candidate(
    frozen: dict[str, Any],
    continued: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Apply the predeclared column-first selection without looking at exact delta."""

    candidates = (
        ("frozen_transpose_symmetric", frozen, "transpose_symmetric"),
        ("frozen_transpose_row_teacher", frozen, "transpose_row_teacher"),
        ("continued_original", continued, "original"),
        ("continued_transpose_symmetric", continued, "transpose_symmetric"),
        ("continued_transpose_row_teacher", continued, "transpose_row_teacher"),
    )
    rows: list[dict[str, Any]] = []
    for index, (name, stage, variant) in enumerate(candidates):
        column_delta = _paired_delta(
            frozen,
            stage,
            baseline_path=("classifier", "original"),
            candidate_path=("classifier", variant),
            metric="column_argmax_correct",
            seed=seed + index * 10,
        )
        row_delta = _paired_delta(
            frozen,
            stage,
            baseline_path=("classifier", "original"),
            candidate_path=("classifier", variant),
            metric="row_argmax_correct",
            seed=seed + index * 10 + 1,
        )
        adjacency_delta = _paired_delta(
            frozen,
            stage,
            baseline_path=("global", "socket_unary_original"),
            candidate_path=("global", f"socket_unary_{variant}"),
            metric="adjacency",
            seed=seed + index * 10 + 2,
        )
        descriptive_exact_delta = _paired_delta(
            frozen,
            stage,
            baseline_path=("global", "socket_unary_original"),
            candidate_path=("global", f"socket_unary_{variant}"),
            metric="correct_tile_count",
            seed=seed + index * 10 + 3,
        )
        strict = all(
            board["strict_permutation"][f"socket_unary_{variant}"]
            for board in stage["boards"]
        )
        adjacency_loss_pp = -100.0 * adjacency_delta["mean_delta_per_board"]
        eligible = bool(
            column_delta["mean_delta_per_board"] >= 2.0
            and column_delta["source_cluster_bootstrap_ci95"][0] > 0.0
            and row_delta["mean_delta_per_board"] >= -1.0
            and adjacency_loss_pp <= 0.10
            and strict
        )
        rows.append(
            {
                "candidate": name,
                "stage": stage["stage"],
                "variant": variant,
                "column_delta": column_delta,
                "row_delta": row_delta,
                "adjacency_delta": adjacency_delta,
                "adjacency_loss_percentage_points": adjacency_loss_pp,
                "descriptive_exact_delta_not_used_for_selection": (
                    descriptive_exact_delta
                ),
                "strict_permutation": strict,
                "eligible": eligible,
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["column_delta"]["mean_delta_per_board"],
            row["candidate"],
        ),
    )
    return {
        "selection_metric": "maximum mean classifier column gain",
        "exact_metrics_used_for_selection": False,
        "requirements": {
            "mean_column_gain_per_board_at_least": 2.0,
            "source_clustered_column_ci95_lower_strictly_above": 0.0,
            "mean_row_loss_per_board_at_most": 1.0,
            "decoder_adjacency_loss_percentage_points_at_most": 0.10,
            "strict_original_tile_permutation": True,
        },
        "selected": selected,
        "passed": bool(selected["eligible"]),
        "candidates": rows,
        "fresh_gate_if_passed": {
            "action": "freeze selected variant and open one new source-disjoint panel",
            "matched_baseline": "socket_ot_decoder144",
            "paired_frozen_comparator": "frozen socket_unary_original",
            "requirements": {
                "candidate_minus_matched_socket_exact_tiles_per_board_at_least": 0.5,
                "matched_socket_exact_ci95_lower_strictly_above": 0.0,
                "candidate_minus_frozen_original_column_tiles_per_board_at_least": 2.0,
                "frozen_original_column_ci95_lower_strictly_above": 0.0,
                "candidate_minus_frozen_original_row_tiles_per_board_at_least": -1.0,
                "matched_socket_adjacency_loss_percentage_points_at_most": 0.2,
                "strict_original_upright_tile_permutation": True,
            },
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    replay_report = json.loads(args.reuse_panel_report.read_text(encoding="utf-8"))
    model, checkpoint = load_coordinate_model(args.checkpoint, device)
    replay_checkpoint_sha = replay_report.get("checkpoint", {}).get("sha256")
    if replay_checkpoint_sha != sha256_file(args.checkpoint):
        raise ValueError("reuse-panel report was not opened with this exact checkpoint")
    recursive_names, recursive_audit = collect_recursive_report_exposure(
        args.report_root
    )
    train_records, replay_records, forbidden = select_train_and_replay_records(
        manifest,
        checkpoint,
        replay_report,
        recursive_names,
        train_limit=args.train_limit,
    )
    replay_names = [str(record["filename"]) for record in replay_records]
    if replay_names != replay_report.get("selection", {}).get("eval_filenames"):
        raise RuntimeError("reuse-panel order differs from the opened report")
    if _names_digest(replay_names) != replay_report.get("selection", {}).get(
        "eval_digest"
    ):
        raise RuntimeError("reuse-panel source digest differs from the opened report")
    train_boards = prepare_clean_boards(train_records, args.targets)
    state_keys_before = tuple(model.state_dict())
    if args.skip_development_replay:
        frozen_evaluation, frozen_seconds = None, 0.0
    else:
        frozen_evaluation, frozen_seconds = evaluate_reused_panel(
            model,
            replay_records,
            replay_report,
            args,
            device,
            stage="frozen",
        )
    history, training_seconds = train_continuation(
        model,
        train_boards,
        args,
        device,
    )
    if tuple(model.state_dict()) != state_keys_before:
        raise RuntimeError("transpose continuation changed coordinate state-dict keys")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("frozen Socket backbone became trainable")
    if args.skip_development_replay:
        continued_evaluation, continued_seconds = None, 0.0
        selection = None
    else:
        continued_evaluation, continued_seconds = evaluate_reused_panel(
            model,
            replay_records,
            replay_report,
            args,
            device,
            stage="continued",
        )
        selection = select_development_candidate(
            frozen_evaluation,
            continued_evaluation,
            seed=args.seed + 900_000,
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "absolute_coordinate_sorter_transpose_continued.pt"
    prior_selection = checkpoint.get("selection", {})
    lineage_train = set(
        prior_selection.get(
            "lineage_train_filenames",
            prior_selection.get("train_filenames", []),
        )
    ) | {str(record["filename"]) for record in train_records}
    lineage_exposed = forbidden | {str(record["filename"]) for record in train_records}
    lineage_exposed.update(replay_names)
    continued_checkpoint = {
        "state_dict": model.state_dict(),
        "contract": checkpoint["contract"],
        "socket_checkpoint": checkpoint["socket_checkpoint"],
        "selection": {
            "namespace": SELECTION_NAMESPACE,
            "train_filenames": [record["filename"] for record in train_records],
            "train_digest": _names_digest(
                [str(record["filename"]) for record in train_records]
            ),
            "lineage_train_filenames": sorted(lineage_train),
            "lineage_exposed_filenames": sorted(lineage_exposed),
        },
        "continuation_contract": {
            "architecture": "state-dict-neutral-transpose-equivariant-continuation-v1",
            "base_checkpoint_sha256": sha256_file(args.checkpoint),
            "socket_backbone_frozen": True,
            "trainable_scope": "existing absolute coordinate head only",
            "dual_view_supervision": True,
            "transpose_axis_consistency_weight": args.consistency_weight,
            "input_index_position_embedding": False,
            "output_tile_geometry": "original upright 20x20 tiles only",
        },
        "continuation_training_history": history,
    }
    torch.save(continued_checkpoint, checkpoint_path)
    report = {
        "experiment": "absolute-coordinate-transpose-equivariant-continuation-v1",
        "status": (
            "development-gate-passed-await-fresh-panel"
            if selection is not None and selection["passed"]
            else "development-gate-failed-no-fresh-panel"
            if selection is not None
            else "training-only-no-development-panel"
        ),
        "hypothesis": (
            "the established absolute-row signal can supply the near-chance column axis "
            "when a consistent whole-board transpose swaps row and column semantics"
        ),
        "novelty_audit": {
            "verdict": "genuinely distinct from archived absolute-coordinate runs",
            "not_repeated": [
                "P14d symmetric relative-edge topology",
                "I20 upright-rotation audit",
                "P35 tile-only row/column regression",
                "capacity-only absolute head sweeps",
            ],
            "new_element": (
                "dual original/transpose absolute supervision plus mapped-axis TTA and "
                "symmetric consistency on the existing frozen coordinate state dict"
            ),
        },
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        | {"device_resolved": str(device)},
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "manifest_train_split_only": True,
            "recursive_report_exposure_excluded": True,
            "recursive_report_audit": recursive_audit,
            "distinct_forbidden_source_count": len(forbidden),
            "reuse_panel_only": not args.skip_development_replay,
            "reuse_panel_report_sha256": sha256_file(args.reuse_panel_report),
            "reuse_panel_checkpoint_sha256_verified": True,
            "fresh_exact_panel_opened": False,
            "calibration_opened": False,
            "holdout_opened": False,
            "competition_test_opened": False,
            "transpose_is_model_view_only": True,
            "original_upright_tiles_preserved_for_every_decoder": True,
        },
        "selection": {
            "train_filenames": [record["filename"] for record in train_records],
            "train_digest": _names_digest(
                [str(record["filename"]) for record in train_records]
            ),
            "reuse_eval_filenames": [] if args.skip_development_replay else replay_names,
            "reuse_eval_digest": (
                None if args.skip_development_replay else _names_digest(replay_names)
            ),
            "lineage_train_count": len(lineage_train),
            "lineage_exposed_count": len(lineage_exposed),
        },
        "model": {
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "state_dict_keys_unchanged": True,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "base_checkpoint_sha256": sha256_file(args.checkpoint),
        },
        "runtime_seconds": {
            "frozen_replay": frozen_seconds,
            "training": training_seconds,
            "continued_replay": continued_seconds,
        },
        "training_history": history,
        "development": {
            "frozen": frozen_evaluation,
            "continued": continued_evaluation,
            "predeclared_selection": selection,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "selection": selection,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
