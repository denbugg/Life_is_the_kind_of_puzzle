#!/usr/bin/env python3
"""Train and calibrate a dirty-only pairwise edge ranker and bijective solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.edge_ranker import (
    EdgeBoard,
    PairwiseEdgeRanker,
    attach_target_labels,
    build_inference_board,
    edge_listwise_loss,
    exact_edge_counts,
    pack_rows,
    prepare_tile_channels,
    score_board,
    unpack_logits,
)
from aiijc_puzzle.legacy_upgrade import layout_digest, solve_buddies, validate_layout
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    IMAGE_SIZE,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_VIEWS = ("raw", "tile_z", "bilateral", "gray")
GATE_THRESHOLDS = {
    "all_pooled_r1_delta": 0.005,
    "trusted_query_pooled_r1_delta": 0.01,
    "per_direction_r1_delta": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--eval-limit", type=int, default=4)
    parser.add_argument("--eval-offset", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rows-per-board", type=int, default=128)
    parser.add_argument("--batch-rows", type=int, default=24)
    parser.add_argument("--pair-batch", type=int, default=1024)
    parser.add_argument("--candidate-k", type=int, default=5)
    parser.add_argument("--view-mode", choices=("raw", "dual"), default="dual")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--checkpoint-in", type=Path)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest_names(records: tuple[dict[str, Any], ...] | tuple[Any, ...]) -> str:
    payload = "\n".join(str(record["filename"]) for record in records).encode()
    return hashlib.sha256(payload).hexdigest()


def choose_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def code_hashes() -> dict[str, str]:
    paths = {
        "edge_ranker": PROJECT_ROOT / "src" / "aiijc_puzzle" / "edge_ranker.py",
        "candidate_supply": PROJECT_ROOT / "src" / "aiijc_puzzle" / "candidate_supply.py",
        "legacy_upgrade": PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        "protocol": PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py",
        "runner": Path(__file__),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def checkpoint_contract(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    train_records: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        "architecture": "joint-seam-context-cnn-v1",
        "views": list(DEFAULT_VIEWS),
        "candidate_k": args.candidate_k,
        "view_mode": args.view_mode,
        "feature_dim": len(DEFAULT_VIEWS) * 3,
        "width": args.width,
        "hidden": args.hidden,
        "label_policy": "exact recovered neighbour; trusted-query training only",
        "teacher_policy": "trusted candidate clean symmetric extrapolation listwise CE",
        "teacher_weight": args.teacher_weight,
        "protocol_digest": compute_protocol_digest(manifest),
        "selector_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selector_seed": EXPERIMENT_SUBSET_SEED,
        "train_limit": len(train_records),
        "train_selection_digest": digest_names(train_records),
        "train_filenames": [record["filename"] for record in train_records],
        "semantic_code_sha256": code_hashes(),
    }


def save_checkpoint(
    path: Path,
    model: PairwiseEdgeRanker,
    contract: dict[str, Any],
    training: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "contract": contract,
            "training_configuration": training,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    train_records: tuple[Any, ...],
    device: torch.device,
) -> tuple[PairwiseEdgeRanker, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = checkpoint_contract(args, manifest, train_records)
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no contract")
    for field in (
        "architecture",
        "views",
        "candidate_k",
        "view_mode",
        "feature_dim",
        "width",
        "hidden",
        "label_policy",
        "teacher_policy",
        "teacher_weight",
        "protocol_digest",
        "selector_namespace",
        "selector_seed",
        "train_limit",
        "train_selection_digest",
        "train_filenames",
    ):
        if contract.get(field) != expected[field]:
            raise ValueError(
                f"checkpoint contract mismatch for {field}: "
                f"{contract.get(field)!r} != {expected[field]!r}"
            )
    model = PairwiseEdgeRanker(
        feature_dim=contract["feature_dim"],
        view_mode=contract["view_mode"],
        width=contract["width"],
        hidden=contract["hidden"],
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, payload


def build_training_boards(
    records: tuple[Any, ...], args: argparse.Namespace
) -> tuple[list[EdgeBoard], float]:
    started = perf_counter()
    boards: list[EdgeBoard] = []
    for index, record in enumerate(records, start=1):
        name = record["filename"]
        dirty = split_tiles(load_rgb(args.train_inputs / name))
        clean = split_tiles(load_rgb(args.targets / name))
        inference = build_inference_board(
            dirty, filename=name, views=DEFAULT_VIEWS, candidate_k=args.candidate_k
        )
        boards.append(attach_target_labels(inference, clean))
        print(f"prepared train {index}/{len(records)} {name}", flush=True)
    return boards, perf_counter() - started


def train_model(
    model: PairwiseEdgeRanker,
    boards: list[EdgeBoard],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []
    started = perf_counter()
    for epoch in range(args.epochs):
        model.train()
        totals: defaultdict[str, float] = defaultdict(float)
        steps = 0
        for board_index in generator.permutation(len(boards)):
            board = boards[int(board_index)]
            eligible = [row for row in board.rows if row.trusted_query and row.exact_candidate >= 0]
            if len(eligible) > args.rows_per_board:
                selected = generator.choice(len(eligible), size=args.rows_per_board, replace=False)
                eligible = [eligible[int(index)] for index in sorted(selected)]
            channels = prepare_tile_channels(board.tiles, view_mode=model.view_mode).to(device)
            for start in range(0, len(eligible), args.batch_rows):
                rows = eligible[start : start + args.batch_rows]
                packed = pack_rows(rows, device=device)
                predicted, _ = model(
                    channels,
                    packed["anchors"],
                    packed["candidates"],
                    packed["directions"],
                    packed["features"],
                    packed["baseline"],
                )
                logits = unpack_logits(predicted, packed)
                loss, parts = edge_listwise_loss(logits, packed, teacher_weight=args.teacher_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                for key, value in parts.items():
                    totals[key] += value
                steps += 1
            del channels
        summary = {key: value / max(steps, 1) for key, value in totals.items()}
        summary["epoch"] = float(epoch + 1)
        summary["steps"] = float(steps)
        history.append(summary)
        print(f"epoch {epoch + 1}/{args.epochs}: {summary}", flush=True)
    return history, perf_counter() - started


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def layout_metrics(layout: np.ndarray, recovered: Any) -> dict[str, float]:
    truth = recovered.dirty_at_position
    position_of_dirty = recovered.position_of_dirty
    predicted_positions = np.empty_like(layout)
    predicted_positions[layout] = np.arange(len(layout))
    shifts: dict[tuple[int, int], int] = {}
    for tile, predicted in enumerate(predicted_positions):
        true = int(position_of_dirty[tile])
        predicted_row, predicted_column = divmod(int(predicted), 24)
        true_row, true_column = divmod(true, 24)
        shift = (true_row - predicted_row, true_column - predicted_column)
        shifts[shift] = shifts.get(shift, 0) + 1
    grid = layout.reshape(24, 24)
    left = position_of_dirty[grid[:, :-1]]
    right = position_of_dirty[grid[:, 1:]]
    top = position_of_dirty[grid[:-1]]
    bottom = position_of_dirty[grid[1:]]
    right_accuracy = np.mean((right - left == 1) & (right // 24 == left // 24))
    down_accuracy = np.mean(bottom - top == 24)
    return {
        "direct_placement": float(np.mean(layout == truth)),
        "translation_aligned_placement": float(max(shifts.values()) / len(layout)),
        "right_adjacency": float(right_accuracy),
        "down_adjacency": float(down_accuracy),
        "adjacency": float(0.5 * (right_accuracy + down_accuracy)),
    }


def merge_edge_counts(groups: list[list[dict[str, int | str]]]) -> dict[str, Any]:
    totals: defaultdict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])
    for group in groups:
        for record in group:
            key = (str(record["scope"]), str(record["direction"]), int(record["k"]))
            totals[key][0] += int(record["edges"])
            totals[key][1] += int(record["hits"])
    result: dict[str, Any] = {}
    for (scope, direction, k), (edges, hits) in sorted(totals.items()):
        result[f"{scope}.{direction}.r{k}"] = {
            "edges": edges,
            "hits": hits,
            "recall": hits / edges if edges else 0.0,
        }
    for scope in ("all", "trusted_query"):
        for k in (1, 5):
            right = result[f"{scope}.right.r{k}"]
            down = result[f"{scope}.down.r{k}"]
            edges = right["edges"] + down["edges"]
            hits = right["hits"] + down["hits"]
            result[f"{scope}.pooled.r{k}"] = {
                "edges": edges,
                "hits": hits,
                "recall": hits / edges,
            }
    return result


def promotion_gate(baseline: dict[str, Any], learned: dict[str, Any]) -> dict[str, Any]:
    conditions: list[dict[str, Any]] = []
    keys = [
        ("all.pooled.r1", GATE_THRESHOLDS["all_pooled_r1_delta"]),
        (
            "trusted_query.pooled.r1",
            GATE_THRESHOLDS["trusted_query_pooled_r1_delta"],
        ),
    ]
    for scope in ("all", "trusted_query"):
        for direction in ("right", "down"):
            keys.append(
                (
                    f"{scope}.{direction}.r1",
                    GATE_THRESHOLDS["per_direction_r1_delta"],
                )
            )
    for key, threshold in keys:
        delta = learned[key]["recall"] - baseline[key]["recall"]
        conditions.append(
            {
                "metric": key,
                "delta": delta,
                "minimum": threshold,
                "passed": delta >= threshold,
            }
        )
    return {"passed": all(item["passed"] for item in conditions), "conditions": conditions}


def freeze_predictions(
    model: PairwiseEdgeRanker,
    records: tuple[Any, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    """Freeze all input-only scores, layouts, and images before target access."""

    started = perf_counter()
    frozen: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        name = record["filename"]
        input_image = load_rgb(args.train_inputs / name)
        tiles = split_tiles(input_image)
        board_started = perf_counter()
        board = build_inference_board(
            tiles, filename=name, views=DEFAULT_VIEWS, candidate_k=args.candidate_k
        )
        learned_right, learned_down, delta = score_board(
            model, board, device=device, pair_batch=args.pair_batch
        )
        baseline_layout_result = solve_buddies(
            board.right_baseline, board.down_baseline, max_edges=96
        )
        learned_layout_result = solve_buddies(learned_right, learned_down, max_edges=96)
        baseline_layout = validate_layout(baseline_layout_result.layout)
        learned_layout = validate_layout(learned_layout_result.layout)
        baseline_raw = assemble_tiles(tiles[baseline_layout])
        learned_raw = assemble_tiles(tiles[learned_layout])
        baseline_nlm = apply_nlm_color(baseline_raw, h=9).image
        learned_nlm = apply_nlm_color(learned_raw, h=9).image
        frozen.append(
            {
                "filename": name,
                "board": board,
                "input_image": input_image,
                "baseline_right": board.right_baseline,
                "baseline_down": board.down_baseline,
                "learned_right": learned_right,
                "learned_down": learned_down,
                "baseline_layout": baseline_layout,
                "learned_layout": learned_layout,
                "baseline_raw": baseline_raw,
                "learned_raw": learned_raw,
                "baseline_nlm": baseline_nlm,
                "learned_nlm": learned_nlm,
                "inference": {
                    "delta": delta,
                    "baseline_layout_sha256": layout_digest(baseline_layout),
                    "learned_layout_sha256": layout_digest(learned_layout),
                    "baseline_raw_sha256": image_digest(baseline_raw),
                    "learned_raw_sha256": image_digest(learned_raw),
                    "baseline_nlm_sha256": image_digest(baseline_nlm),
                    "learned_nlm_sha256": image_digest(learned_nlm),
                    "baseline_unique_tiles": int(len(np.unique(baseline_layout))),
                    "learned_unique_tiles": int(len(np.unique(learned_layout))),
                    "runtime_seconds": perf_counter() - board_started,
                },
            }
        )
        print(f"froze eval {index}/{len(records)} {name}", flush=True)
    return frozen, perf_counter() - started


def evaluate_frozen(
    frozen: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open targets only after every inference artifact has been frozen."""

    baseline_counts: list[list[dict[str, int | str]]] = []
    learned_counts: list[list[dict[str, int | str]]] = []
    labelled_items: list[tuple[dict[str, Any], np.ndarray, EdgeBoard]] = []
    for item in frozen:
        target = load_rgb(args.targets / item["filename"])
        clean_tiles = split_tiles(target)
        labelled = attach_target_labels(item["board"], clean_tiles)
        baseline_counts.append(
            exact_edge_counts(labelled, item["baseline_right"], item["baseline_down"])
        )
        learned_counts.append(
            exact_edge_counts(labelled, item["learned_right"], item["learned_down"])
        )
        labelled_items.append((item, target, labelled))
    baseline_local = merge_edge_counts(baseline_counts)
    learned_local = merge_edge_counts(learned_counts)
    gate = promotion_gate(baseline_local, learned_local)
    local = {"baseline_bilateral": baseline_local, "learned": learned_local, "gate": gate}

    full: dict[str, Any] = {
        "reported": bool(gate["passed"]),
        "reason": "local edge gate passed" if gate["passed"] else "local edge gate failed",
        "boards": [],
    }
    if not gate["passed"]:
        return local, full
    for item, target, labelled in labelled_items:
        recovered = recover_layout(labelled.tiles, split_tiles(target))
        board_record: dict[str, Any] = {"filename": item["filename"]}
        for variant in ("baseline", "learned"):
            raw = item[f"{variant}_raw"]
            nlm = item[f"{variant}_nlm"]
            layout = item[f"{variant}_layout"]
            board_record[variant] = {
                "raw_ssim": contest_ssim(target, raw),
                "nlm9_ssim": contest_ssim(target, nlm),
                **layout_metrics(layout, recovered),
            }
        full["boards"].append(board_record)
    for variant in ("baseline", "learned"):
        fields = list(full["boards"][0][variant])
        full[variant] = {
            field: float(np.mean([board[variant][field] for board in full["boards"]]))
            for field in fields
        }
    full["delta"] = {
        field: full["learned"][field] - full["baseline"][field] for field in full["baseline"]
    }
    return local, full


def main() -> None:
    args = parse_args()
    if (
        min(
            args.train_limit,
            args.eval_limit,
            args.epochs,
            args.rows_per_board,
            args.batch_rows,
            args.pair_batch,
        )
        <= 0
    ):
        raise ValueError("all limits, epochs, and batch sizes must be positive")
    if args.eval_offset < 0:
        raise ValueError("eval-offset must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_records = select_manifest_records(manifest, "train", limit=args.train_limit)
    eval_panel = select_manifest_records(
        manifest, "calibration", limit=args.eval_offset + args.eval_limit
    )
    eval_records = tuple(eval_panel[args.eval_offset :])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_configuration = {
        "epochs": args.epochs,
        "rows_per_board": args.rows_per_board,
        "batch_rows": args.batch_rows,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "teacher_weight": args.teacher_weight,
        "seed": args.seed,
        "device": str(device),
    }
    contract = checkpoint_contract(args, manifest, train_records)
    if args.checkpoint_in:
        model, checkpoint_payload = load_checkpoint(
            args.checkpoint_in,
            args=args,
            manifest=manifest,
            train_records=train_records,
            device=device,
        )
        training_history = checkpoint_payload.get("training_history", [])
        preparation_seconds = 0.0
        training_seconds = 0.0
        checkpoint_training_configuration = checkpoint_payload.get("training_configuration", {})
    else:
        boards, preparation_seconds = build_training_boards(train_records, args)
        model = PairwiseEdgeRanker(
            feature_dim=len(DEFAULT_VIEWS) * 3,
            view_mode=args.view_mode,
            width=args.width,
            hidden=args.hidden,
        ).to(device)
        training_history, training_seconds = train_model(model, boards, args, device)
        checkpoint_training_configuration = training_configuration
        checkpoint = output_dir / "edge_ranker.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "contract": contract,
                "training_configuration": training_configuration,
                "training_history": training_history,
            },
            checkpoint,
        )

    frozen, inference_seconds = freeze_predictions(model, eval_records, args, device)
    target_access_started = perf_counter()
    local, full = evaluate_frozen(frozen, args)
    evaluation_seconds = perf_counter() - target_access_started
    report = {
        "experiment": "joint-dirty-pairwise-edge-ranker-v1",
        "verdict": (
            "eligible-for-scale-or-decoder-comparison"
            if local["gate"]["passed"]
            else "local-edge-gate-failed"
        ),
        "leakage_boundary": {
            "prediction_inputs": "dirty tiles and deterministic analytic scores only",
            "targets": (
                "training labels; calibration metrics opened after all layouts/images frozen"
            ),
            "all_predictions_frozen_before_target_access": True,
            "holdout_opened": False,
            "test_opened": False,
        },
        "compliance": {
            "decoder": "solve_buddies(max_edges=96)",
            "input_tiles": 576,
            "required_unique_tiles": 576,
            "restoration": "frozen coloured NLM h=9 after strict tile assembly",
            "constant_or_template_substitution": False,
        },
        "contract": contract,
        "checkpoint_training_configuration": checkpoint_training_configuration,
        "current_run_configuration": vars(args) | {"device_resolved": str(device)},
        "selection": {
            "train_filenames": [record["filename"] for record in train_records],
            "train_digest": digest_names(train_records),
            "calibration_offset": args.eval_offset,
            "calibration_filenames": [record["filename"] for record in eval_records],
            "calibration_digest": digest_names(eval_records),
        },
        "runtime_seconds": {
            "training_board_preparation": preparation_seconds,
            "training": training_seconds,
            "inference_freeze": inference_seconds,
            "target_assisted_evaluation": evaluation_seconds,
        },
        "training_history": training_history,
        "local_edge_metrics": local,
        "full_board_metrics": full,
        "frozen_inference": [
            {"filename": item["filename"], **item["inference"]} for item in frozen
        ],
    }
    # Path objects are useful in argparse but not in JSON.
    report["current_run_configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in report["current_run_configuration"].items()
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(json.dumps({"report": str(report_path), "gate": local["gate"]}, indent=2))


if __name__ == "__main__":
    main()
