#!/usr/bin/env python3
"""Train and evaluate a full-tile listwise verifier on frozen manifest splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from aiijc_puzzle.candidate_supply import DEFAULT_VIEWS, split_tiles
from aiijc_puzzle.content_verifier import (
    CandidateBoard,
    CandidateRow,
    ContentListwiseVerifier,
    build_candidate_board,
    multi_positive_listwise_loss,
    summarize_choices,
    summarize_oracle,
)
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
INPUTS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
TARGETS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "content-verifier" / "calibration.json"
CHECKPOINT_SCHEMA_VERSION = 2


def _architecture_contract(feature_dim: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "feature_dim": feature_dim,
        "dim": args.dim,
        "heads": args.heads,
        "pair_layers": args.pair_layers,
        "list_layers": args.list_layers,
        "spatial_position": "learned-shared-5x5",
    }


def _label_policy(threshold: float) -> dict[str, Any]:
    return {
        "name": "confidence-aware-content-multipositive-v1",
        "positive_definition": "clean RGB tile RMSE <= threshold",
        "positive_rmse_threshold": float(threshold),
        "candidate_filter": "recovered candidate mapping margin >= board median",
        "exact_exception": "exact neighbour stays positive on a trusted row",
        "training_row_filter": "anchor and exact-neighbour mapping margins >= board median",
        "evaluation_only": "clean target, recovered mapping, margins, RMSE labels",
    }


def _semantic_code_hashes() -> dict[str, str]:
    return {
        "candidate_supply.py": sha256_file(
            PROJECT_ROOT / "src" / "aiijc_puzzle" / "candidate_supply.py"
        ),
        "content_verifier.py": sha256_file(
            PROJECT_ROOT / "src" / "aiijc_puzzle" / "content_verifier.py"
        ),
        "protocol.py": sha256_file(PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py"),
        "run_content_verifier.py": sha256_file(Path(__file__).resolve()),
    }


def _checkpoint_contract(
    training_configuration: dict[str, Any], architecture: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": architecture,
        "protocol_digest": training_configuration["protocol_digest"],
        "subset_namespace": training_configuration["subset_namespace"],
        "subset_seed": training_configuration["subset_seed"],
        "ordered_views": training_configuration["views"],
        "candidate_policy": {
            "name": "union-top-k-per-emitter",
            "candidate_k_per_emitter": training_configuration["candidate_k_per_emitter"],
        },
        "label_policy": _label_policy(training_configuration["train_positive_rmse"]),
        "train_selection_digest": training_configuration["train_selection_digest"],
        "training_code_sha256": _semantic_code_hashes(),
        "bound_runtime_code_sha256": _semantic_code_hashes(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=12)
    parser.add_argument(
        "--eval-offset",
        type=int,
        default=0,
        help="skip this many records in the shared deterministic evaluation subset",
    )
    parser.add_argument("--eval-split", choices=("calibration", "holdout"), default="calibration")
    parser.add_argument("--candidate-k", type=int, default=5)
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rows-per-board", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--pair-layers", type=int, default=1)
    parser.add_argument("--list-layers", type=int, default=1)
    parser.add_argument("--train-positive-rmse", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        help="evaluate an existing checkpoint without preparing or using train boards",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument(
        "--bind-checkpoint-report",
        type=Path,
        help="bind a legacy checkpoint to its report and write a contracted .pt output",
    )
    return parser


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError(f"validation manifest digest mismatch: {path}")
    return manifest


def _bind_checkpoint(
    checkpoint_path: Path,
    report_path: Path,
    output_path: Path,
    manifest: dict[str, Any],
) -> None:
    """Attach a strict semantic contract to this session's pre-contract checkpoint."""
    checkpoint_path = checkpoint_path.resolve()
    report_path = report_path.resolve()
    output_path = output_path.resolve()
    if output_path == checkpoint_path:
        raise ValueError("bound checkpoint output must differ from the legacy checkpoint")
    if output_path.suffix != ".pt":
        raise ValueError("bound checkpoint output must have a .pt suffix")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["artifacts"]["checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError("legacy checkpoint hash does not match its report")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training_configuration = checkpoint.get("configuration")
    if training_configuration != report.get("configuration"):
        raise ValueError("legacy checkpoint and report training configurations differ")
    if training_configuration["protocol_digest"] != manifest["protocol_digest"]:
        raise ValueError("legacy checkpoint protocol does not match the current manifest")
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("legacy checkpoint has no architecture metadata")
    contract = _checkpoint_contract(training_configuration, architecture)
    contract["training_code_sha256"] = {
        "content_verifier.py": report["artifacts"]["code_sha256"],
        "candidate_supply.py": "not-recorded-by-legacy-training-report",
        "protocol.py": "not-recorded-by-legacy-training-report",
        "run_content_verifier.py": "not-recorded-by-legacy-training-report",
    }
    contract["legacy_binding"] = {
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_report_sha256": sha256_file(report_path),
        "training_report_content_verifier_sha256": report["artifacts"]["code_sha256"],
    }
    bound = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": checkpoint["model"],
        "architecture": architecture,
        "contract": contract,
        "training_configuration": training_configuration,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(bound, temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "bound_checkpoint": str(output_path),
                "sha256": sha256_file(output_path),
                "contract": contract,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate all inference and label semantics before opening eval data."""
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint is missing the current semantic schema")
    training_configuration = checkpoint.get("training_configuration")
    contract = checkpoint.get("contract")
    if not isinstance(training_configuration, dict) or not isinstance(contract, dict):
        raise ValueError("checkpoint is missing training configuration or semantic contract")
    expected_architecture = _architecture_contract(len(args.views) * 3, args)
    try:
        train_records = [
            dict(record)
            for record in select_manifest_records(
                manifest,
                "train",
                limit=int(training_configuration["train_limit"]),
            )
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint has an invalid stored train selection") from error
    stored_train_digest = _selection_digest(train_records)
    stored_train_filenames = [record["filename"] for record in train_records]
    checks = {
        "architecture": contract.get("architecture") == expected_architecture,
        "checkpoint_architecture": checkpoint.get("architecture") == expected_architecture,
        "protocol_digest": contract.get("protocol_digest") == manifest["protocol_digest"],
        "subset_namespace": contract.get("subset_namespace") == EXPERIMENT_SUBSET_NAMESPACE,
        "subset_seed": contract.get("subset_seed") == EXPERIMENT_SUBSET_SEED,
        "ordered_views": contract.get("ordered_views") == args.views,
        "stored_ordered_views": training_configuration.get("views") == args.views,
        "candidate_policy": contract.get("candidate_policy")
        == {
            "name": "union-top-k-per-emitter",
            "candidate_k_per_emitter": args.candidate_k,
        },
        "label_policy": contract.get("label_policy") == _label_policy(args.train_positive_rmse),
        "stored_candidate_k": training_configuration.get("candidate_k_per_emitter")
        == args.candidate_k,
        "stored_label_threshold": training_configuration.get("train_positive_rmse")
        == args.train_positive_rmse,
        "stored_architecture": all(
            training_configuration.get(name) == value
            for name, value in expected_architecture.items()
            if name not in {"feature_dim", "spatial_position"}
        ),
        "train_selection_digest": contract.get("train_selection_digest")
        == training_configuration.get("train_selection_digest")
        == stored_train_digest,
        "train_filenames": training_configuration.get("train_filenames") == stored_train_filenames,
        "stored_protocol_digest": training_configuration.get("protocol_digest")
        == manifest["protocol_digest"],
        "stored_subset": training_configuration.get("subset_namespace")
        == EXPERIMENT_SUBSET_NAMESPACE
        and training_configuration.get("subset_seed") == EXPERIMENT_SUBSET_SEED,
        "bound_runtime_code_sha256": contract.get("bound_runtime_code_sha256")
        == _semantic_code_hashes(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"checkpoint semantic contract mismatch: {failed}")
    return training_configuration


def _selection_digest(records: list[dict[str, str]]) -> str:
    return hashlib.sha256("\n".join(record["filename"] for record in records).encode()).hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image[:, :, ::-1]


def _prepare_boards(
    records: list[dict[str, str]], views: list[str], k: int
) -> list[CandidateBoard]:
    boards: list[CandidateBoard] = []
    for index, record in enumerate(records, start=1):
        name = record["filename"]
        input_path = INPUTS_DIR / name
        target_path = TARGETS_DIR / name
        if sha256_file(input_path) != record["input_sha256"]:
            raise ValueError(f"input hash mismatch: {name}")
        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"target hash mismatch: {name}")
        started = time.perf_counter()
        board = build_candidate_board(
            split_tiles(_load_rgb(input_path)),
            split_tiles(_load_rgb(target_path)),
            filename=name,
            views=views,
            candidate_k=k,
        )
        boards.append(board)
        print(
            f"prepare {index:03d}/{len(records):03d} {name} "
            f"rows={len(board.rows)} {time.perf_counter() - started:.2f}s",
            flush=True,
        )
    return boards


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _collate(
    refs: list[tuple[CandidateBoard, CandidateRow]],
    device: torch.device,
    positive_rmse: float,
) -> dict[str, Any]:
    batch = len(refs)
    count = max(len(row.candidates) for _, row in refs)
    height, width = refs[0][0].tiles.shape[1:3]
    feature_dim = refs[0][1].features.shape[1]
    anchors = np.empty((batch, height, width, 3), dtype=np.uint8)
    candidates = np.zeros((batch, count, height, width, 3), dtype=np.uint8)
    features = np.zeros((batch, count, feature_dim), dtype=np.float32)
    valid = np.zeros((batch, count), dtype=bool)
    positives = np.zeros((batch, count), dtype=bool)
    directions = np.empty(batch, dtype=np.int64)
    for index, (board, row) in enumerate(refs):
        size = len(row.candidates)
        anchors[index] = board.tiles[row.anchor]
        candidates[index, :size] = board.tiles[row.candidates]
        features[index, :size] = row.features
        valid[index, :size] = True
        positives[index, :size] = row.training_positives(positive_rmse)
        directions[index] = row.direction
    return {
        "anchors": torch.from_numpy(anchors).permute(0, 3, 1, 2).to(device),
        "candidates": torch.from_numpy(candidates).permute(0, 1, 4, 2, 3).to(device),
        "features": torch.from_numpy(features).to(device),
        "valid": torch.from_numpy(valid).to(device),
        "positives": torch.from_numpy(positives).to(device),
        "directions": torch.from_numpy(directions).to(device),
    }


def _forward(model: ContentListwiseVerifier, batch: dict[str, Any]) -> torch.Tensor:
    return model(
        batch["anchors"],
        batch["candidates"],
        batch["features"],
        batch["valid"],
        batch["directions"],
    )


def _epoch_refs(
    boards: list[CandidateBoard], rows_per_board: int, threshold: float, seed: int
) -> list[tuple[CandidateBoard, CandidateRow]]:
    rng = np.random.default_rng(seed)
    refs: list[tuple[CandidateBoard, CandidateRow]] = []
    for board in boards:
        eligible = [
            row
            for row in board.rows
            if row.trusted and bool(row.training_positives(threshold).any())
        ]
        if len(eligible) > rows_per_board:
            chosen = rng.choice(len(eligible), size=rows_per_board, replace=False)
            eligible = [eligible[int(index)] for index in chosen]
        refs.extend((board, row) for row in eligible)
    rng.shuffle(refs)
    return refs


def _train(
    model: ContentListwiseVerifier,
    boards: list[CandidateBoard],
    *,
    device: torch.device,
    epochs: int,
    rows_per_board: int,
    batch_size: int,
    positive_rmse: float,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[list[dict[str, float | int]], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float | int]] = []
    eligible_total = sum(
        row.trusted and bool(row.training_positives(positive_rmse).any())
        for board in boards
        for row in board.rows
    )
    for epoch in range(epochs):
        model.train()
        refs = _epoch_refs(boards, rows_per_board, positive_rmse, seed + epoch)
        losses: list[float] = []
        started = time.perf_counter()
        for start in range(0, len(refs), batch_size):
            batch = _collate(refs[start : start + batch_size], device, positive_rmse)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch)
            loss = multi_positive_listwise_loss(logits, batch["positives"], batch["valid"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch + 1,
            "rows": len(refs),
            "mean_loss": float(np.mean(losses)),
            "seconds": elapsed,
        }
        history.append(row)
        print(
            f"epoch {epoch + 1:02d}/{epochs:02d} rows={len(refs)} "
            f"loss={row['mean_loss']:.5f} {elapsed:.1f}s",
            flush=True,
        )
    return history, int(eligible_total)


def _predict(
    model: ContentListwiseVerifier,
    boards: list[CandidateBoard],
    device: torch.device,
    batch_size: int,
) -> tuple[list[CandidateRow], list[int]]:
    refs = [(board, row) for board in boards for row in board.rows]
    choices: list[int] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(refs), batch_size):
            batch = _collate(refs[start : start + batch_size], device, 20.0)
            choices.extend(_forward(model, batch).argmax(dim=1).cpu().tolist())
    return [row for _, row in refs], choices


def _metrics(
    rows: list[CandidateRow], choices: list[int], views: list[str]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for scope in ("all", "trusted_query", "trusted"):
        methods: dict[str, Any] = {
            "candidate_pool_oracle": summarize_oracle(rows, scope=scope),
            "classical_ensemble": summarize_choices(
                rows, [row.ensemble_choice for row in rows], scope=scope
            ),
            "content_verifier": summarize_choices(rows, choices, scope=scope),
        }
        for index, view in enumerate(views):
            methods[f"classical_{view}"] = summarize_choices(
                rows, [row.baseline_choices[index] for row in rows], scope=scope
            )
        classical = {
            name: value for name, value in methods.items() if name.startswith("classical_")
        }
        strongest_exact = max(classical, key=lambda name: classical[name]["exact"])
        strongest_content20 = max(classical, key=lambda name: classical[name]["content_rmse_le_20"])
        methods["strongest_classical"] = {
            "by_exact": strongest_exact,
            "by_content_rmse_le_20": strongest_content20,
        }
        output[scope] = methods
    return output


def _metrics_by_direction(
    rows: list[CandidateRow], choices: list[int], views: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for direction, axis in (("right", 0), ("down", 1)):
        selected = [
            (row, choice)
            for row, choice in zip(rows, choices, strict=True)
            if row.direction == axis
        ]
        output[direction] = _metrics(
            [row for row, _ in selected],
            [choice for _, choice in selected],
            views,
        )
    return output


def _paired_board_deltas(
    boards: list[CandidateBoard], choices: list[int]
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Board-level verifier deltas against fixed classical comparators."""
    offsets = np.cumsum([0, *[len(board.rows) for board in boards]])
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for scope in ("all", "trusted_query", "trusted"):
        comparisons: dict[str, dict[str, float | int]] = {}
        for baseline_name in ("classical_ensemble", "classical_bilateral"):
            metric_deltas: dict[str, list[float]] = {
                "exact": [],
                "content_rmse_le_10": [],
                "content_rmse_le_20": [],
            }
            for index, board in enumerate(boards):
                board_choices = choices[offsets[index] : offsets[index + 1]]
                verifier = summarize_choices(board.rows, board_choices, scope=scope)
                if baseline_name == "classical_ensemble":
                    baseline_choices = [row.ensemble_choice for row in board.rows]
                else:
                    bilateral_index = board.views.index("bilateral")
                    baseline_choices = [row.baseline_choices[bilateral_index] for row in board.rows]
                baseline = summarize_choices(board.rows, baseline_choices, scope=scope)
                for metric in metric_deltas:
                    metric_deltas[metric].append(float(verifier[metric]) - float(baseline[metric]))
            for metric, values in metric_deltas.items():
                array = np.asarray(values, dtype=np.float64)
                mean = float(array.mean())
                standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
                comparisons[f"verifier_minus_{baseline_name}_{metric}"] = {
                    "boards": len(array),
                    "mean": mean,
                    "standard_error": standard_error,
                    "normal_lower_95": mean - 1.96 * standard_error,
                    "wins": int(np.sum(array > 0)),
                    "ties": int(np.sum(array == 0)),
                    "losses": int(np.sum(array < 0)),
                }
        output[scope] = comparisons
    return output


def _promotion_gate(
    metrics: dict[str, dict[str, Any]],
    metrics_by_direction: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Evaluate the final audited pooled and direction safety gates."""
    thresholds = {
        "all_exact": 0.005,
        "all_content_rmse_le_20": 0.0,
        "trusted_exact": 0.01,
        "trusted_content_rmse_le_20": 0.01,
        "right_all_exact": 0.0,
        "right_all_content_rmse_le_20": 0.0,
        "down_all_exact": 0.0,
        "down_all_content_rmse_le_20": 0.0,
    }
    comparators: dict[str, str] = {}
    deltas: dict[str, float] = {}
    checks: dict[str, bool] = {}
    for scope in ("all", "trusted"):
        verifier = metrics[scope]["content_verifier"]
        for metric in ("exact", "content_rmse_le_20"):
            name = f"{scope}_{metric}"
            comparator_key = "by_exact" if metric == "exact" else "by_content_rmse_le_20"
            comparators[name] = metrics[scope]["strongest_classical"][comparator_key]
            baseline = metrics[scope][comparators[name]]
            deltas[name] = float(verifier[metric]) - float(baseline[metric])
            checks[name] = deltas[name] >= thresholds[name]
    for direction in ("right", "down"):
        direction_metrics = metrics_by_direction[direction]["all"]
        verifier = direction_metrics["content_verifier"]
        for metric in ("exact", "content_rmse_le_20"):
            name = f"{direction}_all_{metric}"
            comparator_key = "by_exact" if metric == "exact" else "by_content_rmse_le_20"
            comparators[name] = direction_metrics["strongest_classical"][comparator_key]
            baseline = direction_metrics[comparators[name]]
            deltas[name] = float(verifier[metric]) - float(baseline[metric])
            checks[name] = deltas[name] >= thresholds[name]
    return {
        "comparators": comparators,
        "thresholds": thresholds,
        "deltas": deltas,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = _parser().parse_args()
    if args.epochs < 1 or args.rows_per_board < 1 or args.batch_size < 1:
        raise ValueError("epochs, rows-per-board, and batch-size must be positive")
    if args.eval_offset < 0:
        raise ValueError("eval-offset must be non-negative")
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    if args.bind_checkpoint_report is not None:
        if args.checkpoint_in is None:
            raise ValueError("--bind-checkpoint-report requires --checkpoint-in")
        _bind_checkpoint(
            args.checkpoint_in,
            args.bind_checkpoint_report,
            args.output,
            manifest,
        )
        return
    train_records = [
        dict(record)
        for record in select_manifest_records(manifest, "train", limit=args.train_limit)
    ]
    eval_panel = [
        dict(record)
        for record in select_manifest_records(
            manifest,
            args.eval_split,
            limit=args.eval_offset + args.eval_limit,
        )
    ]
    eval_records = eval_panel[args.eval_offset :]
    configuration = {
        "manifest_path": str(manifest_path),
        "protocol_digest": manifest["protocol_digest"],
        "subset_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "subset_seed": EXPERIMENT_SUBSET_SEED,
        "train_split": "train",
        "eval_split": args.eval_split,
        "train_limit": args.train_limit,
        "eval_limit": args.eval_limit,
        "eval_offset": args.eval_offset,
        "train_selection_digest": _selection_digest(train_records),
        "eval_selection_digest": _selection_digest(eval_records),
        "train_filenames": [record["filename"] for record in train_records],
        "eval_filenames": [record["filename"] for record in eval_records],
        "candidate_k_per_emitter": args.candidate_k,
        "views": args.views,
        "epochs": args.epochs,
        "rows_per_board": args.rows_per_board,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dim": args.dim,
        "heads": args.heads,
        "pair_layers": args.pair_layers,
        "list_layers": args.list_layers,
        "train_positive_rmse": args.train_positive_rmse,
        "seed": args.seed,
        "requested_device": args.device,
        "checkpoint_in": str(args.checkpoint_in.resolve()) if args.checkpoint_in else None,
    }
    if args.dry_run:
        print(json.dumps(configuration, ensure_ascii=False, indent=2))
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    checkpoint: dict[str, Any] | None = None
    checkpoint_training_configuration = configuration
    if args.checkpoint_in is not None:
        checkpoint = torch.load(args.checkpoint_in.resolve(), map_location="cpu", weights_only=True)
        checkpoint_training_configuration = _validate_checkpoint(checkpoint, args, manifest)
    started = time.perf_counter()
    train_boards: list[CandidateBoard] = []
    train_prepare_seconds = 0.0
    if args.checkpoint_in is None:
        print(f"device={device}; preparing train boards", flush=True)
        train_started = time.perf_counter()
        train_boards = _prepare_boards(train_records, args.views, args.candidate_k)
        train_prepare_seconds = time.perf_counter() - train_started
    else:
        print(f"device={device}; evaluating checkpoint {args.checkpoint_in.resolve()}", flush=True)
    print("preparing held-out evaluation boards", flush=True)
    eval_started = time.perf_counter()
    eval_boards = _prepare_boards(eval_records, args.views, args.candidate_k)
    eval_prepare_seconds = time.perf_counter() - eval_started

    feature_dim = len(args.views) * 3
    model = ContentListwiseVerifier(
        feature_dim=feature_dim,
        dim=args.dim,
        heads=args.heads,
        pair_layers=args.pair_layers,
        list_layers=args.list_layers,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"model parameters={parameter_count:,}", flush=True)
    if args.checkpoint_in is None:
        history, eligible_total = _train(
            model,
            train_boards,
            device=device,
            epochs=args.epochs,
            rows_per_board=args.rows_per_board,
            batch_size=args.batch_size,
            positive_rmse=args.train_positive_rmse,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
    else:
        assert checkpoint is not None
        model.load_state_dict(checkpoint["model"])
        history, eligible_total = [], 0

    eval_started = time.perf_counter()
    rows, choices = _predict(model, eval_boards, device, args.batch_size)
    eval_seconds = time.perf_counter() - eval_started
    metrics = _metrics(rows, choices, args.views)
    metrics_by_direction = _metrics_by_direction(rows, choices, args.views)
    paired_board_deltas = _paired_board_deltas(eval_boards, choices)
    promotion_gate = _promotion_gate(metrics, metrics_by_direction)
    output_path = args.output.resolve()
    if args.checkpoint_in is None:
        checkpoint_path = output_path.with_suffix(".pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model": state,
                "architecture": _architecture_contract(feature_dim, args),
                "contract": _checkpoint_contract(
                    configuration, _architecture_contract(feature_dim, args)
                ),
                "training_configuration": configuration,
            },
            checkpoint_path,
        )
    else:
        checkpoint_path = args.checkpoint_in.resolve()
    payload = {
        "schema_version": 1,
        "experiment": "dirty-full-tile-content-listwise-verifier",
        "status": "completed",
        "inference_inputs": [
            "corrupted shuffled tile pixels",
            "dirty-only classical union-top-k candidates and costs",
        ],
        "evaluation_only_inputs": [
            "clean train targets",
            "target-assisted Hungarian labels and mapping-confidence subset",
        ],
        "architecture": {
            "description": (
                "joint position-aware full-tile pair attention followed by shortlist attention"
            ),
            "parameters": parameter_count,
            "feature_dim": feature_dim,
            "spatial_position": "learned-shared-5x5",
        },
        "run_configuration": configuration,
        "checkpoint_training_configuration": checkpoint_training_configuration,
        "training": {
            "mode": "trained" if args.checkpoint_in is None else "pretrained-checkpoint",
            "eligible_trusted_rows": eligible_total,
            "history": history,
        },
        "metrics": metrics,
        "metrics_by_direction": metrics_by_direction,
        "promotion_gate": promotion_gate,
        "paired_board_deltas": paired_board_deltas,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "train_prepare_seconds": train_prepare_seconds,
            "eval_prepare_seconds": eval_prepare_seconds,
            "eval_seconds": eval_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "code_sha256": sha256_file(
                PROJECT_ROOT / "src" / "aiijc_puzzle" / "content_verifier.py"
            ),
        },
    }
    _write_json_atomic(payload, output_path)
    print(json.dumps(metrics["trusted"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps({"promotion_gate": promotion_gate}, ensure_ascii=False, indent=2), flush=True)
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
