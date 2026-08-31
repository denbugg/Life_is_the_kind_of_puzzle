#!/usr/bin/env python3
"""Train and gate a raw-preserving restored BorderRanker on exact synthetic boards."""

from __future__ import annotations

import argparse
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

from aiijc_puzzle.pretrained_tile_denoiser import (
    load_drunet_color,
    render_drunet_tiles,
)
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.restored_border_ranker import (
    CandidateUnion,
    RestoredBorderRanker,
    build_candidate_union,
    pad_candidate_rows,
    restored_descriptor_scores,
    unpack_candidate_logits,
)
from aiijc_puzzle.socket_sorter_production import (
    LoadedSocketCheckpoint,
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
DEFAULT_DRUNET_CHECKPOINT = (
    PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f/drunet_color.pth"
)
EXPECTED_DRUNET_SHA256 = "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4"
SELECTION_NAMESPACE = "aiijc-restored-border-ranker-oof-v1"
MAX_TRAIN_SOURCES = 256
MAX_STEPS = 400
GRID = 24
COUNT = GRID * GRID
LOCAL_KS = (1, 5)
TOP_K_SUPPLY = 32
SUPPLY_DELTA_GATE = 0.03
R1_DELTA_GATE = 0.005
R5_DELTA_GATE = 0.0
PRECISION_DELTA_GATE = 0.05
MIN_RECIPROCAL_COVERAGE = 0.03


@dataclass(frozen=True)
class CleanBoard:
    filename: str
    target_sha256: str
    tiles: np.ndarray


@dataclass(frozen=True)
class BoardView:
    restored_tiles: np.ndarray
    raw_scores: tuple[np.ndarray, np.ndarray]
    descriptor_scores: tuple[np.ndarray, np.ndarray]
    unions: tuple[CandidateUnion, CandidateUnion]
    runtime_seconds: dict[str, float]


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
    parser.add_argument("--drunet-checkpoint", type=Path, default=DEFAULT_DRUNET_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sources", type=int, default=256)
    parser.add_argument("--eval-sources", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--rows-per-step", type=int, default=32)
    parser.add_argument("--pair-batch", type=int, default=2048)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--sigma", type=float, default=40.0)
    parser.add_argument("--drunet-batch", type=int, default=144)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--exclude-report", type=Path, action="append", default=[])
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.train_sources <= MAX_TRAIN_SOURCES:
        raise ValueError(f"train-sources must be in [1, {MAX_TRAIN_SOURCES}]")
    if not 1 <= args.eval_sources <= 16:
        raise ValueError("eval-sources must be in [1, 16]")
    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    if min(args.rows_per_step, args.pair_batch, args.base, args.drunet_batch, args.log_every) <= 0:
        raise ValueError("batch, width and logging values must be positive")
    if args.base % 8:
        raise ValueError("base must be divisible by eight")
    if not 0.0 <= args.sigma <= 50.0:
        raise ValueError("sigma must be in [0, 50]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be finite and non-negative")
    if args.allow_nondeterministic_mps and args.device != "mps":
        raise ValueError("allow-nondeterministic-mps requires --device mps")
    if not args.exclude_report:
        raise ValueError("at least one prior report must be supplied for lineage exclusion")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def _collect_declared_filenames(value: Any, *, parent_key: str = "") -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            names.update(_collect_declared_filenames(child, parent_key=str(key)))
    elif isinstance(value, list):
        if parent_key.endswith("filenames") and all(isinstance(item, str) for item in value):
            names.update(Path(item).name for item in value if item.endswith(".png"))
        else:
            for child in value:
                names.update(_collect_declared_filenames(child, parent_key=parent_key))
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
        found = _collect_declared_filenames(payload)
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


@torch.inference_mode()
def _socket_scores(
    socket: LoadedSocketCheckpoint,
    tiles: np.ndarray,
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


def build_board_view(
    socket: LoadedSocketCheckpoint,
    drunet: torch.nn.Module,
    dirty_tiles: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> BoardView:
    started = perf_counter()
    raw_scores = _socket_scores(socket, dirty_tiles, device)
    socket_seconds = perf_counter() - started
    restored, drunet_diagnostics = render_drunet_tiles(
        drunet,
        dirty_tiles,
        sigma_255=args.sigma,
        device=device,
        batch_size=args.drunet_batch,
    )
    supply_started = perf_counter()
    descriptor_scores = (
        restored_descriptor_scores(restored, direction=0),
        restored_descriptor_scores(restored, direction=1),
    )
    unions = (
        build_candidate_union(raw_scores[0], descriptor_scores[0], topk=TOP_K_SUPPLY),
        build_candidate_union(raw_scores[1], descriptor_scores[1], topk=TOP_K_SUPPLY),
    )
    return BoardView(
        restored_tiles=restored,
        raw_scores=raw_scores,
        descriptor_scores=descriptor_scores,
        unions=unions,
        runtime_seconds={
            "socket_d64": socket_seconds,
            "drunet": drunet_diagnostics.runtime_seconds,
            "candidate_supply": perf_counter() - supply_started,
        },
    )


def _truth_by_anchor(reference: np.ndarray, *, direction: int) -> np.ndarray:
    positions = np.arange(len(reference))
    valid = positions % GRID != GRID - 1 if direction == 0 else positions < COUNT - GRID
    delta = 1 if direction == 0 else GRID
    truth = np.full(COUNT, -1, dtype=np.int32)
    truth[reference[positions[valid]]] = reference[positions[valid] + delta]
    return truth


def _labelled_rows(
    view: BoardView,
    reference: ExactSyntheticReference,
) -> list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, int]]:
    rows: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for direction, union in enumerate(view.unions):
        truth = _truth_by_anchor(reference.tile_at_position, direction=direction)
        for anchor, candidates in enumerate(union.rows):
            if truth[anchor] < 0:
                continue
            match = np.flatnonzero(candidates == truth[anchor])
            if len(match):
                rows.append(
                    (
                        anchor,
                        direction,
                        candidates,
                        union.scalar_features[anchor],
                        union.baseline_scores[anchor],
                        int(match[0]),
                    )
                )
    return rows


def _model_flat_scores(
    model: RestoredBorderRanker,
    restored_tensor: torch.Tensor,
    packed: dict[str, torch.Tensor],
    *,
    pair_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    count = len(packed["anchors"])
    for start in range(0, count, pair_batch):
        stop = min(start + pair_batch, count)
        score, residual = model(
            restored_tensor,
            packed["anchors"][start:stop],
            packed["candidates"][start:stop],
            packed["directions"][start:stop],
            packed["features"][start:stop],
            packed["baseline"][start:stop],
        )
        scores.append(score)
        residuals.append(residual)
    return torch.cat(scores), torch.cat(residuals)


def train_model(
    model: RestoredBorderRanker,
    socket: LoadedSocketCheckpoint,
    drunet: torch.nn.Module,
    boards: list[CleanBoard],
    *,
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
    generator = np.random.default_rng(args.seed + 1)
    history: list[dict[str, Any]] = []
    started = perf_counter()
    for step in range(args.steps):
        board = boards[int(generator.integers(len(boards)))]
        synthetic_input, reference = make_exact_synthetic_case(
            board.tiles,
            source_filename=board.filename,
            draw_index=step,
            seed=args.seed,
        )
        view = build_board_view(socket, drunet, synthetic_input.tiles, args=args, device=device)
        eligible = _labelled_rows(view, reference)
        if not eligible:
            raise RuntimeError("candidate union contains no true training neighbours")
        if len(eligible) > args.rows_per_step:
            selected = generator.choice(len(eligible), size=args.rows_per_step, replace=False)
            eligible = [eligible[int(index)] for index in sorted(selected)]
        packed = pad_candidate_rows(eligible, device=device)
        restored_tensor = _tensor(view.restored_tiles, device)
        model.train()
        flat, residual = _model_flat_scores(
            model,
            restored_tensor,
            packed,
            pair_batch=args.pair_batch,
        )
        logits = unpack_candidate_logits(flat, packed)
        loss = F.cross_entropy(logits, packed["targets"])
        prediction = logits.argmax(dim=1)
        accuracy = float((prediction == packed["targets"]).float().mean().detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0))
        optimizer.step()
        scheduler.step()
        record = {
            "step": step + 1,
            "source_filename": board.filename,
            "loss": float(loss.detach()),
            "row_top1": accuracy,
            "eligible_union_rows": len(_labelled_rows(view, reference)),
            "trained_rows": len(eligible),
            "trained_pairs": len(flat),
            "residual_mean": float(residual.detach().mean()),
            "residual_std": float(residual.detach().std()),
            "grad_norm": grad_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"runtime_{key}": value for key, value in view.runtime_seconds.items()},
        }
        history.append(record)
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            recent = history[-min(args.log_every, len(history)) :]
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step + 1,
                        "loss": float(np.mean([item["loss"] for item in recent])),
                        "row_top1": float(
                            np.mean([item["row_top1"] for item in recent])
                        ),
                        "seconds_per_step": (perf_counter() - started) / (step + 1),
                    }
                ),
                flush=True,
            )
    return history, perf_counter() - started


@torch.inference_mode()
def _rank_union(
    model: RestoredBorderRanker,
    view: BoardView,
    *,
    device: torch.device,
    pair_batch: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    packed_rows: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for direction, union in enumerate(view.unions):
        for anchor, candidates in enumerate(union.rows):
            packed_rows.append(
                (
                    anchor,
                    direction,
                    candidates,
                    union.scalar_features[anchor],
                    union.baseline_scores[anchor],
                    0,
                )
            )
    packed = pad_candidate_rows(packed_rows, device=device)
    restored_tensor = _tensor(view.restored_tiles, device)
    model.eval()
    flat, residual = _model_flat_scores(
        model,
        restored_tensor,
        packed,
        pair_batch=pair_batch,
    )
    values = flat.float().cpu().numpy()
    right = np.full((COUNT, COUNT), -1e4, dtype=np.float32)
    down = np.full((COUNT, COUNT), -1e4, dtype=np.float32)
    offset = 0
    for anchor, direction, candidates, _, _, _ in packed_rows:
        matrix = right if direction == 0 else down
        matrix[anchor, candidates] = values[offset : offset + len(candidates)]
        offset += len(candidates)
    np.fill_diagonal(right, -1e4)
    np.fill_diagonal(down, -1e4)
    return right, down, {
        "pairs": len(values),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_abs_max": float(residual.abs().max()),
    }


def _reciprocal_evidence(scores: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(scores, dtype=np.float32).copy()
    if value.shape != (COUNT, COUNT) or not np.isfinite(value).all():
        raise ValueError("reciprocal scores must be one finite full-board matrix")
    np.fill_diagonal(value, -np.inf)
    row_order = np.argsort(-value, axis=1, kind="stable")[:, :2]
    column_order = np.argsort(-value, axis=0, kind="stable")[:2]
    target = row_order[:, 0]
    row_margin = value[np.arange(COUNT), row_order[:, 0]] - value[
        np.arange(COUNT), row_order[:, 1]
    ]
    column_margin = value[column_order[0], np.arange(COUNT)] - value[
        column_order[1], np.arange(COUNT)
    ]
    reciprocal = column_order[0, target] == np.arange(COUNT)
    return {
        "target": np.ascontiguousarray(target, dtype=np.int32),
        "reciprocal": np.ascontiguousarray(reciprocal),
        "confidence": np.ascontiguousarray(
            np.minimum(row_margin, column_margin[target]), dtype=np.float32
        ),
    }


def freeze_prediction(
    model: RestoredBorderRanker,
    socket: LoadedSocketCheckpoint,
    drunet: torch.nn.Module,
    synthetic_input: SyntheticSocketInput,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> FrozenPrediction:
    view = build_board_view(socket, drunet, synthetic_input.tiles, args=args, device=device)
    started = perf_counter()
    learned_right, learned_down, learned_diagnostics = _rank_union(
        model,
        view,
        device=device,
        pair_batch=args.pair_batch,
    )
    ranker_seconds = perf_counter() - started
    variants = {
        "socket_d64_ot": view.raw_scores,
        "restored_cross_ranker": (learned_right, learned_down),
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
            axis: {
                "raw_top32": union.raw_topk,
                "union_padded": _pad_union(union.rows),
            }
            for axis, union in zip(("right", "down"), view.unions, strict=True)
        },
        runtime_seconds=view.runtime_seconds
        | {"cross_ranker": ranker_seconds, **learned_diagnostics},
    )


def _pad_union(rows: tuple[np.ndarray, ...]) -> np.ndarray:
    width = max(map(len, rows))
    output = np.full((len(rows), width), -1, dtype=np.int32)
    for row, candidates in enumerate(rows):
        output[row, : len(candidates)] = candidates
    return output


def _write_frozen(
    predictions: list[FrozenPrediction],
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
        for axis, supply in prediction.supply.items():
            for field, value in supply.items():
                arrays[f"{prefix}__supply__{axis}__{field}"] = value
        cases.append(
            {
                "prefix": prefix,
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "runtime_seconds": prediction.runtime_seconds,
            }
        )
    array_path = output_dir / "frozen_local_predictions.npz"
    np.savez_compressed(array_path, **arrays)
    metadata_path = output_dir / "frozen_local_predictions.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "aiijc-restored-border-ranker-local-predictions-v1",
                "contains_clean_pixels": False,
                "contains_restored_pixels": False,
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


def _aggregate_metrics(
    predictions: list[FrozenPrediction],
    references: dict[str, ExactSyntheticReference],
) -> tuple[dict[str, dict[str, float | int]], dict[str, Any], dict[str, Any]]:
    variants = tuple(predictions[0].candidates)
    local: dict[str, dict[str, float | int]] = {}
    reciprocal_rows: dict[str, list[tuple[float, bool]]] = {name: [] for name in variants}
    supply_counts = {
        axis: {"raw_hits": 0, "union_hits": 0, "total": 0}
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
            total = local.setdefault(
                name,
                {"pooled_total": 0, "pooled_hits_at_1": 0, "pooled_hits_at_5": 0},
            )
            total["pooled_total"] = int(total["pooled_total"]) + int(metrics["pooled_total"])
            for k in LOCAL_KS:
                key = f"pooled_hits_at_{k}"
                total[key] = int(total[key]) + int(metrics[key])
        for direction, axis in enumerate(("right", "down")):
            truth = _truth_by_anchor(reference, direction=direction)
            valid = truth >= 0
            anchors = np.flatnonzero(valid)
            raw = prediction.supply[axis]["raw_top32"]
            union = prediction.supply[axis]["union_padded"]
            supply_counts[axis]["raw_hits"] += int(
                sum(truth[anchor] in raw[anchor] for anchor in anchors)
            )
            supply_counts[axis]["union_hits"] += int(
                sum(truth[anchor] in union[anchor] for anchor in anchors)
            )
            supply_counts[axis]["total"] += len(anchors)
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
        denominator = int(metrics["pooled_total"])
        for k in LOCAL_KS:
            metrics[f"pooled_r{k}"] = int(metrics[f"pooled_hits_at_{k}"]) / denominator
    supply: dict[str, Any] = {}
    for axis, counts in supply_counts.items():
        raw_coverage = counts["raw_hits"] / counts["total"]
        union_coverage = counts["union_hits"] / counts["total"]
        supply[axis] = counts | {
            "raw_top32_coverage": raw_coverage,
            "union_coverage": union_coverage,
            "coverage_delta": union_coverage - raw_coverage,
        }
    native: dict[str, Any] = {}
    for name, rows in reciprocal_rows.items():
        correct = sum(int(ok) for _, ok in rows)
        native[name] = {
            "queries": len(rows),
            "coverage": len(rows) / valid_query_count,
            "precision": correct / len(rows) if rows else 0.0,
        }
    control = reciprocal_rows["socket_d64_ot"]
    candidate = reciprocal_rows["restored_cross_ranker"]
    count = min(len(control), len(candidate))

    def precision_at(rows: list[tuple[float, bool]], limit: int) -> float:
        selected = sorted(rows, key=lambda item: -item[0])[:limit]
        return sum(int(ok) for _, ok in selected) / limit if limit else 0.0

    matched = {
        "query_count": count,
        "coverage": count / valid_query_count,
        "candidate_precision": precision_at(candidate, count),
        "socket_d64_ot_precision": precision_at(control, count),
    }
    matched["precision_gain"] = (
        matched["candidate_precision"] - matched["socket_d64_ot_precision"]
    )
    return local, supply, {"native": native, "matched_vs_socket_d64_ot": matched}


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if args.device == "mps" and args.allow_nondeterministic_mps:
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest protocol digest is invalid")
    if sha256_file(args.drunet_checkpoint) != EXPECTED_DRUNET_SHA256:
        raise ValueError("official DRUNet checkpoint digest changed")
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    _, socket_lineage = load_checkpoint_with_lineage(
        args.socket_checkpoint,
        project_root=PROJECT_ROOT,
    )
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
    train_boards = _prepare_boards(train_records, args.targets)
    drunet = load_drunet_color(args.drunet_checkpoint, device)
    for parameter in drunet.parameters():
        parameter.requires_grad_(False)
    model = RestoredBorderRanker(base=args.base).to(device)
    history, training_seconds = train_model(
        model,
        socket,
        drunet,
        train_boards,
        args=args,
        device=device,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "architecture": "raw-d64-plus-restored-border-residual-ranker-v1",
        "historical_e20_commit": "a877065944",
        "historical_exact_binaries_available": False,
        "restored_view": "official external DRUNet colour sigma40 independent per tile",
        "raw_supply": "frozen d64 partial-OT top32",
        "restored_supply": "normalised DRUNet border descriptor top32",
        "candidate_union": "self-excluded target-blind set union",
        "ranker": "seven-channel restored seam CNN plus eight raw/restored rank features",
        "ranker_output": "learned residual over raw d64 row-standardised score",
        "topk_each": TOP_K_SUPPLY,
        "base": args.base,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_rendered_pixels_in_output": False,
        "original_tiles_only_if_later_decoded": True,
    }
    checkpoint_path = output_dir / "restored_border_ranker.pt"
    exposed = tuple(sorted(set(excluded) | set(train_names) | set(eval_names)))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "contract": contract,
            "selection": {
                "train_filenames": list(train_names),
                "train_digest": names_digest(train_names),
                "lineage_train_filenames": sorted(train_names),
                "lineage_train_digest": names_digest(train_names, sort_names=True),
                "lineage_exposed_filenames": list(exposed),
                "lineage_exposed_digest": names_digest(exposed, sort_names=True),
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
    predictions = [
        freeze_prediction(model, socket, drunet, item, args=args, device=device)
        for item in inputs
    ]
    frozen_array, frozen_metadata = _write_frozen(predictions, output_dir)
    local, supply, reciprocal = _aggregate_metrics(predictions, references)
    r1_gain = float(local["restored_cross_ranker"]["pooled_r1"]) - float(
        local["socket_d64_ot"]["pooled_r1"]
    )
    r5_gain = float(local["restored_cross_ranker"]["pooled_r5"]) - float(
        local["socket_d64_ot"]["pooled_r5"]
    )
    precision_gain = float(reciprocal["matched_vs_socket_d64_ot"]["precision_gain"])
    supply_passed = all(
        float(supply[axis]["coverage_delta"]) >= SUPPLY_DELTA_GATE
        for axis in ("right", "down")
    )
    ranked_passed = r1_gain >= R1_DELTA_GATE and r5_gain >= R5_DELTA_GATE
    precision_passed = (
        precision_gain >= PRECISION_DELTA_GATE
        and reciprocal["matched_vs_socket_d64_ot"]["coverage"]
        >= MIN_RECIPROCAL_COVERAGE
    )
    local_gate_passed = supply_passed and (ranked_passed or precision_passed)
    report = {
        "experiment": "restored-border-ranker-source-disjoint-oof-pilot-v1",
        "status": (
            "local-gate-passed-decoder-eligible"
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
            "ranker_train_eval_source_disjoint": True,
            "socket_checkpoint_and_prior_report_lineages_excluded": True,
            "predictions_frozen_before_exact_reference_scoring": True,
            "restored_pixels_used_for_matcher_only": True,
            "raw_original_tiles_reserved_for_any_future_output": True,
            "competition_test_opened": False,
            "calibration_opened": False,
            "holdout_opened": False,
            "global_decoder_run": False,
            "global_decoder_policy": "forbidden unless predeclared local gate passes",
            "local_gate": {
                "supply": f"both directional union coverage deltas >= {SUPPLY_DELTA_GATE}",
                "ranking": f"pooled R1 delta >= {R1_DELTA_GATE} and R5 delta >= {R5_DELTA_GATE}",
                "precision_alternative": (
                    f"matched reciprocal precision delta >= {PRECISION_DELTA_GATE} at "
                    f"coverage >= {MIN_RECIPROCAL_COVERAGE}"
                ),
                "combined": "supply AND (ranking OR precision alternative)",
            },
        },
        "historical_audit": {
            "source_commit": "a877065944",
            "missing_expected_artifacts": {
                "real_fragment_restorer_best.pt": (
                    "6fcc7de2cf8063b4f2f45d4b96b8999d5eb9c29a071ff2c0031d2703c70d6695"
                ),
                "restored_border_ranker_best.pt": (
                    "8eb7b7e106c0333b9a099f88894eac7b1081555643d3828e479aaf4e56137be1"
                ),
                "restored_sidecar_69387927_bytes": (
                    "65c04742aeaa1fb51934fd70951052a46443f09dd60c798b484f66aca29e5cab"
                ),
            },
            "historical_ranker_was_invoked_in_layout": False,
            "historical_training_stem_overlap_verifiable": False,
            "historical_candidate32_val_r1": 0.3868359375,
            "historical_candidate32_val_r5": 0.688515625,
        },
        "selection": {
            "train_filenames": list(train_names),
            "train_digest": names_digest(train_names),
            "eval_filenames": list(eval_names),
            "eval_digest": names_digest(eval_names),
            "excluded_filename_count": len(excluded),
            "excluded_filename_digest": names_digest(tuple(sorted(excluded))),
            "excluded_reports": exclusion_reports,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "socket_checkpoint_sha256": socket.sha256,
            "drunet_checkpoint_sha256": EXPECTED_DRUNET_SHA256,
            "frozen_predictions": str(frozen_array),
            "frozen_predictions_sha256": sha256_file(frozen_array),
            "frozen_metadata": str(frozen_metadata),
            "frozen_metadata_sha256": sha256_file(frozen_metadata),
        },
        "runtime_seconds": {
            "training": training_seconds,
            "mean_training_step": training_seconds / args.steps,
            "eval_components_mean": {
                key: float(np.mean([item.runtime_seconds[key] for item in predictions]))
                for key in predictions[0].runtime_seconds
            },
        },
        "training_history": history,
        "local": local,
        "candidate_supply": supply,
        "reciprocal": reciprocal,
        "gate": {
            "supply_passed": supply_passed,
            "r1_gain": r1_gain,
            "r5_gain": r5_gain,
            "ranked_passed": ranked_passed,
            "matched_precision_gain": precision_gain,
            "precision_passed": precision_passed,
            "local_gate_passed": local_gate_passed,
        },
        "verdict": (
            "meaningful local gate passed; global decoder may be evaluated separately"
            if local_gate_passed
            else "stop: no global decoder, no production/default change"
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
                "device": str(device),
                "runtime_seconds": report["runtime_seconds"],
                "local": local,
                "candidate_supply": supply,
                "reciprocal": reciprocal,
                "gate": report["gate"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
