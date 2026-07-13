#!/usr/bin/env python3
"""Train and gate a query-only visual continuation scorer.

ContinuationNet-0 predicts the clean four-pixel strip immediately to the right
of a corrupted query tile.  Downward relations are rotated counter-clockwise
before the model is called, so one network covers both puzzle directions.  The
neighbour tile is never an input to the network during training; at retrieval
time every one of the other 575 denoised tiles is compared with the predicted
strip.

This is a bounded, leakage-safe development experiment.  Calibration A may
select an epoch and a pre-declared frozen-w4 blend weight.  Calibration B stays
unopened until that choice is frozen.  V4 and all audit paths are absent by
construction.  Even a passing result is not directly safe for submission.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
for value in (SCRIPT_ROOT, REPO_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.continuation_net import (
    ContinuationNet0,
    save_continuation_net0_checkpoint,
)
from puzzle_assembly.geometry import (
    GRID,
    TILE_COUNT,
    inverse_permutation,
    validate_permutation,
)
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics, retrieval_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8


PANELS = ("primary_kornia", "independent_libjpeg")
ALPHAS = (0.1, 0.25, 0.5, 1.0)
RIGHT = 0
DOWN = 1
EPSILON = 1e-3
QAP_ITERATIONS = 25
QAP_RESTARTS = 2


@dataclass(frozen=True)
class EvaluationRecord:
    name: str
    panel: str
    seed: int
    raw_tiles: np.ndarray
    denoised_tiles: np.ndarray
    clean_target: np.ndarray
    slot_to_target: np.ndarray
    hbt: CompatibilityMatrices
    w4: CompatibilityMatrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument(
        "--denoiser", default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
    )
    parser.add_argument(
        "--embedding-checkpoint",
        default=(
            "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/"
            "hbt_d320_denoised_rgb_sobel.pt"
        ),
    )
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument(
        "--quarantine", default="configs/denoise_validation_quarantine_v1.json"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--train-offset", type=int, default=4096)
    parser.add_argument("--train-sources", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples-per-source", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--score-query-batch-size", type=int, default=192)
    parser.add_argument("--score-pair-query-chunk", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--calibration-a-offset", type=int, default=376)
    parser.add_argument("--calibration-b-offset", type=int, default=380)
    parser.add_argument("--calibration-sources", type=int, default=4)
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected RGB shape: {path}: {values.shape}")
    return values


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def hardware_probe(device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        output["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    return output


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def tensor_tiles(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    ).div_(255.0)


def canonicalize_down(values: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    output = values.clone()
    down = directions == DOWN
    if bool(down.any()):
        output[down] = torch.rot90(output[down], k=1, dims=(-2, -1))
    return output


def true_pairs() -> np.ndarray:
    values: list[tuple[int, int, int]] = []
    for position in range(TILE_COUNT):
        row, column = divmod(position, GRID)
        if column + 1 < GRID:
            values.append((position, position + 1, RIGHT))
        if row + 1 < GRID:
            values.append((position, position + GRID, DOWN))
    output = np.asarray(values, dtype=np.int32)
    if output.shape != (1104, 3):
        raise RuntimeError(f"unexpected true-pair table: {output.shape}")
    return output


def model_outputs(model: nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(inputs)
    if not isinstance(output, dict) or set(output) != {"continuation", "reconstruction"}:
        raise RuntimeError("ContinuationNet0 output schema mismatch")
    continuation = output["continuation"]
    reconstruction = output["reconstruction"]
    expected_strip = (len(inputs), 3, 20, 4)
    expected_query = (len(inputs), 3, 20, 20)
    if tuple(continuation.shape) != expected_strip or tuple(reconstruction.shape) != expected_query:
        raise RuntimeError(
            f"ContinuationNet0 output shape mismatch: "
            f"{tuple(continuation.shape)}, {tuple(reconstruction.shape)}"
        )
    return continuation, reconstruction


def charbonnier(values: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(values.square() + EPSILON**2)


def training_loss(
    predicted_strip: torch.Tensor,
    reconstructed_query: torch.Tensor,
    target_strip: torch.Tensor,
    clean_query: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    column_weights = predicted_strip.new_tensor([1.0, 0.5, 0.25, 0.125]).view(1, 1, 1, 4)
    continuation = (charbonnier(predicted_strip - target_strip) * column_weights).mean()
    continuation_grad_y = charbonnier(
        torch.diff(predicted_strip, dim=-2) - torch.diff(target_strip, dim=-2)
    ).mean()
    continuation_grad_x = charbonnier(
        torch.diff(predicted_strip, dim=-1) - torch.diff(target_strip, dim=-1)
    ).mean()
    query = charbonnier(reconstructed_query - clean_query).mean()
    query_grad_y = charbonnier(
        torch.diff(reconstructed_query, dim=-2) - torch.diff(clean_query, dim=-2)
    ).mean()
    query_grad_x = charbonnier(
        torch.diff(reconstructed_query, dim=-1) - torch.diff(clean_query, dim=-1)
    ).mean()
    query_gradients = query_grad_y + query_grad_x
    total = (
        continuation
        + 0.5 * continuation_grad_y
        + 0.25 * continuation_grad_x
        + 0.25 * query
        + 0.1 * query_gradients
    )
    parts = {
        "total": float(total.detach().cpu()),
        "continuation_charbonnier": float(continuation.detach().cpu()),
        "continuation_grad_y": float(continuation_grad_y.detach().cpu()),
        "continuation_grad_x": float(continuation_grad_x.detach().cpu()),
        "query_charbonnier": float(query.detach().cpu()),
        "query_gradients": float(query_gradients.detach().cpu()),
    }
    return total, parts


def prepare_training_tensors(
    name: str,
    panel_name: str,
    panel_seed: int,
    sample_seed: int,
    *,
    args: argparse.Namespace,
    restorer: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    clean_target = read_rgb(Path(args.data_root) / "train/targets" / name)
    panel = make_exact_panel(clean_target, panel=panel_name, seed=panel_seed)
    denoised = restore_tiles_uint8(
        restorer,
        panel.slot_tiles,
        device,
        batch_size=args.denoise_batch_size,
    )
    pairs = true_pairs()
    rng = np.random.default_rng(sample_seed)
    selected = rng.choice(len(pairs), size=args.samples_per_source, replace=False)
    sampled = pairs[selected]
    position_to_slot = inverse_permutation(panel.slot_to_target)
    query_slots = position_to_slot[sampled[:, 0]]
    clean_query = panel.clean_target_tiles[sampled[:, 0]]
    clean_neighbour = panel.clean_target_tiles[sampled[:, 1]]
    directions = torch.from_numpy(sampled[:, 2].astype(np.int64)).to(device)
    raw_query = canonicalize_down(tensor_tiles(panel.slot_tiles[query_slots], device), directions)
    denoised_query = canonicalize_down(tensor_tiles(denoised[query_slots], device), directions)
    clean_query_tensor = canonicalize_down(tensor_tiles(clean_query, device), directions)
    clean_neighbour_tensor = canonicalize_down(tensor_tiles(clean_neighbour, device), directions)
    target_strip = clean_neighbour_tensor[..., :4]
    return raw_query, denoised_query, clean_query_tensor, target_strip, sampled


def train_one_epoch(
    core_model: ContinuationNet0,
    forward_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_names: list[str],
    epoch: int,
    *,
    args: argparse.Namespace,
    restorer: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    forward_model.train()
    totals: dict[str, float] = {}
    optimizer_steps = 0
    sampled_edges = 0
    panel_counts = {panel: 0 for panel in PANELS}
    source_records = []
    amp_enabled = device.type == "cuda" and not args.no_amp
    for source_index, name in enumerate(train_names):
        panel = PANELS[(epoch + source_index) % len(PANELS)]
        panel_counts[panel] += 1
        panel_seed = per_source_seed(
            args.seed, f"continuation-net0-train-{panel}", name, epoch
        )
        sample_seed = per_source_seed(args.seed, "continuation-net0-sampling", name, epoch)
        raw, denoised, clean_query, target_strip, sampled = prepare_training_tensors(
            name,
            panel,
            panel_seed,
            sample_seed,
            args=args,
            restorer=restorer,
            device=device,
        )
        order_rng = np.random.default_rng(
            per_source_seed(args.seed, "continuation-net0-batch-order", name, epoch)
        )
        order = order_rng.permutation(len(sampled))
        source_loss = []
        for start in range(0, len(order), args.batch_size):
            indices = torch.as_tensor(order[start : start + args.batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predicted_strip, reconstructed_query = model_outputs(
                    forward_model, torch.cat([raw[indices], denoised[indices]], dim=1)
                )
                loss, parts = training_loss(
                    predicted_strip,
                    reconstructed_query,
                    target_strip[indices],
                    clean_query[indices],
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(core_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            sampled_edges += len(indices)
            source_loss.append(parts["total"])
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
        source_records.append(
            {
                "name": name,
                "panel": panel,
                "panel_seed": int(panel_seed),
                "sample_seed": int(sample_seed),
                "sampled_edges": int(len(sampled)),
                "right_edges": int(np.sum(sampled[:, 2] == RIGHT)),
                "down_edges": int(np.sum(sampled[:, 2] == DOWN)),
                "mean_loss": float(np.mean(source_loss)),
            }
        )
        del raw, denoised, clean_query, target_strip
        print(
            json.dumps(
                {
                    "stage": "train",
                    "epoch": epoch,
                    "done": source_index + 1,
                    "total": len(train_names),
                    "panel": panel,
                }
            ),
            flush=True,
        )
    if sampled_edges != len(train_names) * args.samples_per_source:
        raise RuntimeError("training sample accounting drift")
    return {
        "epoch": epoch,
        "optimizer_steps": optimizer_steps,
        "sampled_edges": sampled_edges,
        "panel_counts": panel_counts,
        "mean_losses": {key: value / optimizer_steps for key, value in totals.items()},
        "sources": source_records,
    }


def prepare_evaluation_record(
    name: str,
    panel_name: str,
    split_name: str,
    *,
    args: argparse.Namespace,
    restorer: nn.Module,
    embedding: nn.Module,
    device: torch.device,
) -> EvaluationRecord:
    seed = per_source_seed(
        args.seed, f"continuation-net0-{split_name}-{panel_name}", name, 0
    )
    clean_target = read_rgb(Path(args.data_root) / "train/targets" / name)
    panel = make_exact_panel(clean_target, panel=panel_name, seed=seed)
    denoised = restore_tiles_uint8(
        restorer,
        panel.slot_tiles,
        device,
        batch_size=args.denoise_batch_size,
    )
    hbt, _ = learned_compatibility(
        embedding, denoised, device=device, name="frozen_hbt"
    )
    classical_bank = build_classical_score_bank(
        denoised, prefix="denoised", chunk_size=64
    )
    classical_names = [
        score_name
        for score_name in sorted(classical_bank)
        if score_name.startswith("denoised_") and not score_name.endswith("_c2")
    ]
    c1 = fuse_ranked_scores(classical_bank, names=classical_names, name="frozen_c1")
    w4 = fuse_ranked_scores(
        {"c1": c1, "hbt": hbt},
        names=["c1", "hbt"],
        weights={"hbt": 4.0},
        name="frozen_w4",
    )
    return EvaluationRecord(
        name=name,
        panel=panel_name,
        seed=int(seed),
        raw_tiles=panel.slot_tiles,
        denoised_tiles=denoised,
        clean_target=clean_target,
        slot_to_target=panel.slot_to_target,
        hbt=hbt,
        w4=w4,
    )


@torch.inference_mode()
def predict_query_strips(
    model: nn.Module,
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    direction: int,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    directions = torch.full((TILE_COUNT,), direction, device=device, dtype=torch.long)
    raw = canonicalize_down(tensor_tiles(raw_tiles, device), directions)
    denoised = canonicalize_down(tensor_tiles(denoised_tiles, device), directions)
    strips = []
    model.eval()
    for start in range(0, TILE_COUNT, batch_size):
        inputs = torch.cat([raw[start : start + batch_size], denoised[start : start + batch_size]], dim=1)
        continuation, _ = model_outputs(model, inputs)
        strips.append(continuation.float())
    return torch.cat(strips, dim=0)


@torch.inference_mode()
def strip_cost_matrix(
    predicted: torch.Tensor,
    candidate_tiles: np.ndarray,
    direction: int,
    *,
    device: torch.device,
    query_chunk: int,
) -> np.ndarray:
    directions = torch.full((TILE_COUNT,), direction, device=device, dtype=torch.long)
    canonical_candidates = canonicalize_down(tensor_tiles(candidate_tiles, device), directions)
    boundary = canonical_candidates[..., :4]
    boundary_grad_y = torch.diff(boundary, dim=-2)
    boundary_centered = boundary - boundary.mean(dim=(-2, -1), keepdim=True)
    output = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    for start in range(0, TILE_COUNT, query_chunk):
        stop = min(start + query_chunk, TILE_COUNT)
        query = predicted[start:stop]
        pixel = charbonnier(query[:, None] - boundary[None]).mean(dim=(2, 3, 4))
        tangent = charbonnier(
            torch.diff(query, dim=-2)[:, None] - boundary_grad_y[None]
        ).mean(dim=(2, 3, 4))
        centered_query = query - query.mean(dim=(-2, -1), keepdim=True)
        centered = charbonnier(
            centered_query[:, None] - boundary_centered[None]
        ).mean(dim=(2, 3, 4))
        output[start:stop] = (pixel + 0.5 * tangent + 0.25 * centered).cpu().numpy()
    np.fill_diagonal(output, np.inf)
    return output


def continuation_compatibility(
    model: nn.Module,
    record: EvaluationRecord,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> CompatibilityMatrices:
    matrices = []
    for direction in (RIGHT, DOWN):
        predicted = predict_query_strips(
            model,
            record.raw_tiles,
            record.denoised_tiles,
            direction,
            device=device,
            batch_size=args.score_query_batch_size,
        )
        matrices.append(
            strip_cost_matrix(
                predicted,
                record.denoised_tiles,
                direction,
                device=device,
                query_chunk=args.score_pair_query_chunk,
            )
        )
    return CompatibilityMatrices("continuation_only", matrices[0], matrices[1])


def row_percentile_cost(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError(f"unexpected compatibility shape: {values.shape}")
    safe = values.copy()
    np.fill_diagonal(safe, np.inf)
    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(TILE_COUNT, dtype=np.int32), order.shape),
        axis=1,
    )
    output = ranks.astype(np.float32) / float(TILE_COUNT - 2)
    np.fill_diagonal(output, np.inf)
    return output


def blend_compatibility(
    baseline: CompatibilityMatrices,
    continuation: CompatibilityMatrices,
    alpha: float,
) -> CompatibilityMatrices:
    matrices = []
    for side in ("right", "down"):
        baseline_cost = np.asarray(getattr(baseline, side), dtype=np.float32)
        continuation_rank = row_percentile_cost(getattr(continuation, side))
        combined = (baseline_cost + float(alpha) * continuation_rank) / (1.0 + float(alpha))
        np.fill_diagonal(combined, np.inf)
        matrices.append(combined.astype(np.float32))
    return CompatibilityMatrices(f"w4_plus_{alpha:g}_continuation", matrices[0], matrices[1])


def compact_retrieval(values: CompatibilityMatrices, truth: np.ndarray) -> dict[str, float]:
    metrics = retrieval_metrics(values, truth, ks=(1, 5))["combined"]
    return {
        "recall_at_1": float(metrics["recall_at_1"]),
        "recall_at_5": float(metrics["recall_at_5"]),
        "mrr": float(metrics["mrr"]),
        "conditional_mrr": float(metrics["mrr"]),
        "median_rank": float(metrics["median_rank"]),
        "q90_rank": float(metrics["q90_rank"]),
        "queries": int(metrics["queries"]),
    }


def evaluate_split(
    model: nn.Module,
    names: list[str],
    split_name: str,
    *,
    args: argparse.Namespace,
    restorer: nn.Module,
    embedding: nn.Module,
    device: torch.device,
    keep_scores: bool = False,
    alphas: tuple[float, ...] = ALPHAS,
) -> tuple[list[dict[str, Any]], list[tuple[EvaluationRecord, dict[str, CompatibilityMatrices]]]]:
    records: list[dict[str, Any]] = []
    score_records: list[tuple[EvaluationRecord, dict[str, CompatibilityMatrices]]] = []
    for source_index, name in enumerate(names):
        for panel in PANELS:
            record = prepare_evaluation_record(
                name,
                panel,
                split_name,
                args=args,
                restorer=restorer,
                embedding=embedding,
                device=device,
            )
            continuation = continuation_compatibility(model, record, args=args, device=device)
            scores = {
                "hbt": record.hbt,
                "w4": record.w4,
                "continuation_only": continuation,
                **{
                    f"blend_alpha_{alpha:g}": blend_compatibility(record.w4, continuation, alpha)
                    for alpha in alphas
                },
            }
            metrics = {
                method: compact_retrieval(score, record.slot_to_target)
                for method, score in scores.items()
            }
            records.append(
                {
                    "name": name,
                    "panel": panel,
                    "seed": record.seed,
                    "candidate_policy": "all 575 non-self destinations for every right/down query",
                    "metrics": metrics,
                }
            )
            if keep_scores:
                score_records.append((record, scores))
        print(
            json.dumps(
                {"stage": split_name, "done": source_index + 1, "total": len(names)}
            ),
            flush=True,
        )
    return records, score_records


def aggregate_retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    methods = tuple(records[0]["metrics"])

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            method: {
                metric: float(np.mean([record["metrics"][method][metric] for record in selected]))
                for metric in ("recall_at_1", "recall_at_5", "mrr", "conditional_mrr")
            }
            for method in methods
        }

    return {
        "macro": summarize(records),
        "panels": {
            panel: summarize([record for record in records if record["panel"] == panel])
            for panel in PANELS
        },
    }


def calibration_a_decision(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 8:
        raise RuntimeError(f"calibration A must contain exactly 8 source-panel records, got {len(records)}")
    summary = aggregate_retrieval(records)
    macro = summary["macro"]
    selected_alpha = max(
        ALPHAS,
        key=lambda alpha: (macro[f"blend_alpha_{alpha:g}"]["mrr"], -alpha),
    )
    selected_method = f"blend_alpha_{selected_alpha:g}"
    panel_mrr_deltas = {
        panel: summary["panels"][panel][selected_method]["mrr"]
        - summary["panels"][panel]["w4"]["mrr"]
        for panel in PANELS
    }
    conditions = {
        "continuation_only_conditional_mrr_ge_0.16": (
            macro["continuation_only"]["conditional_mrr"] >= 0.16
        ),
        "continuation_only_recall_at_5_ge_0.25": (
            macro["continuation_only"]["recall_at_5"] >= 0.25
        ),
        "selected_blend_mrr_delta_ge_0.010": (
            macro[selected_method]["mrr"] - macro["w4"]["mrr"] >= 0.010
        ),
        "both_panels_mrr_nonnegative": all(value >= 0.0 for value in panel_mrr_deltas.values()),
    }
    return {
        "summary": summary,
        "selected_alpha": float(selected_alpha),
        "selected_method": selected_method,
        "selected_mrr_delta": float(
            macro[selected_method]["mrr"] - macro["w4"]["mrr"]
        ),
        "panel_mrr_deltas": panel_mrr_deltas,
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def calibration_b_decision(records: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    if len(records) != 8:
        raise RuntimeError(f"calibration B must contain exactly 8 source-panel records, got {len(records)}")
    summary = aggregate_retrieval(records)
    method = f"blend_alpha_{alpha:g}"
    deltas = {
        metric: summary["macro"][method][metric] - summary["macro"]["w4"][metric]
        for metric in ("mrr", "recall_at_1", "recall_at_5")
    }
    panel_deltas = {
        panel: {
            metric: summary["panels"][panel][method][metric]
            - summary["panels"][panel]["w4"][metric]
            for metric in ("mrr", "recall_at_1", "recall_at_5")
        }
        for panel in PANELS
    }
    wins = sum(
        record["metrics"][method]["mrr"] > record["metrics"]["w4"]["mrr"]
        for record in records
    )
    conditions = {
        "mrr_delta_ge_0.015": deltas["mrr"] >= 0.015,
        "recall_at_1_delta_ge_0.010": deltas["recall_at_1"] >= 0.010,
        "recall_at_5_delta_ge_0.020": deltas["recall_at_5"] >= 0.020,
        "both_panels_nonnegative_all_metrics": all(
            value >= 0.0 for panel in panel_deltas.values() for value in panel.values()
        ),
        "record_mrr_wins_ge_6_of_8": wins >= 6,
    }
    return {
        "summary": summary,
        "selected_alpha": float(alpha),
        "selected_method": method,
        "deltas": deltas,
        "panel_deltas": panel_deltas,
        "record_mrr_wins": int(wins),
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def qap_record(
    record: EvaluationRecord,
    scores: dict[str, CompatibilityMatrices],
    alpha: float,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_method = f"blend_alpha_{alpha:g}"
    initial_result = soft_cycle_component_solver(
        record.hbt,
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        loop_weight=1.0,
        reciprocal_weight=0.35,
    )
    initial = validate_permutation(initial_result.position_to_slot)
    qap_seed = per_source_seed(
        args.seed, "continuation-net0-calibration-b-qap", record.name, record.panel
    )

    def run(score: CompatibilityMatrices) -> dict[str, Any]:
        result = directional_qap(
            score,
            initial=initial.copy(),
            iterations=QAP_ITERATIONS,
            restarts=QAP_RESTARTS,
            seed=int(qap_seed),
            boundary_weight=0.05,
            initial_weight=0.75,
            noisy_components=3,
            noise_scale=1.0,
            refine_swaps=8,
            refine_weak_cells=32,
        )
        layout = validate_permutation(result.position_to_slot)
        geometry = layout_metrics(layout, record.slot_to_target)
        image = predicted_image_metrics(layout, record.denoised_tiles, record.clean_target)
        return {
            "valid_permutation": bool(geometry["valid_permutation"]),
            "combined_adjacency": float(geometry["combined_adjacency"]),
            "ssim": float(image["predicted_layout_ssim"]),
            "objective": float(result.objective),
            "restart": int(result.restart),
            "iterations": int(result.iterations),
        }

    baseline = run(scores["w4"])
    candidate = run(scores[selected_method])
    return {
        "name": record.name,
        "panel": record.panel,
        "seed": record.seed,
        "qap_seed": int(qap_seed),
        "iterations": QAP_ITERATIONS,
        "baseline_w4": baseline,
        "candidate": candidate,
        "delta_adjacency": candidate["combined_adjacency"] - baseline["combined_adjacency"],
        "delta_ssim": candidate["ssim"] - baseline["ssim"],
    }


def qap_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "records": len(records),
        "mean_baseline_adjacency": float(
            np.mean([record["baseline_w4"]["combined_adjacency"] for record in records])
        ),
        "mean_candidate_adjacency": float(
            np.mean([record["candidate"]["combined_adjacency"] for record in records])
        ),
        "mean_adjacency_delta": float(np.mean([record["delta_adjacency"] for record in records])),
        "mean_baseline_ssim": float(np.mean([record["baseline_w4"]["ssim"] for record in records])),
        "mean_candidate_ssim": float(np.mean([record["candidate"]["ssim"] for record in records])),
        "mean_ssim_delta": float(np.mean([record["delta_ssim"] for record in records])),
        "adjacency_wins": int(sum(record["delta_adjacency"] > 0 for record in records)),
        "ssim_wins": int(sum(record["delta_ssim"] > 0 for record in records)),
        "all_valid_permutations": all(
            record["baseline_w4"]["valid_permutation"]
            and record["candidate"]["valid_permutation"]
            for record in records
        ),
        "panels": {
            panel: {
                "records": sum(record["panel"] == panel for record in records),
                "mean_adjacency_delta": float(
                    np.mean(
                        [record["delta_adjacency"] for record in records if record["panel"] == panel]
                    )
                ),
                "mean_ssim_delta": float(
                    np.mean([record["delta_ssim"] for record in records if record["panel"] == panel])
                ),
            }
            for panel in PANELS
        },
    }
    conditions = {
        "all_valid_permutations": bool(summary["all_valid_permutations"]),
        "each_panel_mean_ssim_delta_ge_0.002": all(
            values["mean_ssim_delta"] >= 0.002 for values in summary["panels"].values()
        ),
        "overall_ssim_wins_ge_5_of_8": summary["ssim_wins"] >= 5,
        "mean_adjacency_delta_nonnegative": summary["mean_adjacency_delta"] >= 0.0,
        "each_panel_mean_adjacency_delta_nonnegative": all(
            values["mean_adjacency_delta"] >= 0.0 for values in summary["panels"].values()
        ),
    }
    summary["conditions"] = conditions
    summary["passed"] = all(conditions.values())
    return summary


def write_report(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if min(
        args.train_sources,
        args.epochs,
        args.samples_per_source,
        args.batch_size,
        args.calibration_sources,
    ) <= 0:
        raise SystemExit("source, epoch, sample, and batch counts must be positive")
    if args.samples_per_source > 1104:
        raise SystemExit("samples-per-source cannot exceed the 1104 directed truths")
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit("output root is not empty; pass --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "continuation_net0_report.json"
    started = time.time()
    set_determinism(args.seed)
    device = resolve_device(args.device)
    restorer, restored_device, denoiser_metadata = load_restorer(
        args.denoiser, device=str(device), state="ema"
    )
    if restored_device != device:
        raise RuntimeError(f"denoiser device drift: {restored_device} != {device}")
    embedding, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    for frozen in (restorer, embedding):
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)

    train_pool = source_names_for_split(
        "edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine
    )
    development = source_names_for_split(
        "edge_development", manifest_path=args.manifest, quarantine_path=args.quarantine
    )
    train_names = train_pool[args.train_offset : args.train_offset + args.train_sources]
    calibration_a_names = development[
        args.calibration_a_offset : args.calibration_a_offset + args.calibration_sources
    ]
    calibration_b_names = development[
        args.calibration_b_offset : args.calibration_b_offset + args.calibration_sources
    ]
    if (
        len(train_names) != args.train_sources
        or len(calibration_a_names) != args.calibration_sources
        or len(calibration_b_names) != args.calibration_sources
    ):
        raise RuntimeError("requested source slice is unavailable")
    source_sets = [set(train_names), set(calibration_a_names), set(calibration_b_names)]
    if any(source_sets[i] & source_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("train/calibration whole-source overlap")

    core_model = ContinuationNet0().to(device)
    forward_model: nn.Module = core_model
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        forward_model = nn.DataParallel(core_model)
    optimizer = torch.optim.AdamW(
        core_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not args.no_amp
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "continuation_net0_leakage_safe_retrieval_gate",
        "status": "training",
        "safe_for_submission": False,
        "args": vars(args),
        "hardware": hardware_probe(device),
        "model_config": ContinuationNet0.config(),
        "protocol": {
            "candidate_policy": "dense all-575 non-self candidates per right/down query",
            "down_canonicalization": "torch.rot90(k=1,dims=(-2,-1)) on raw query, denoised query, clean query, and clean neighbour",
            "training_panel_schedule": "PANELS[(epoch + source_index) % 2]",
            "calibration_a_role": "predeclared checkpoint/alpha selection and early-stop screen",
            "calibration_b_role": "source-disjoint frozen retrieval gate",
            "v4_paths_constructed_or_opened": False,
            "qap_policy": "matched production-budget 25x2 plus refine 8/32 gate only after calibration-B retrieval pass",
        },
        "split": {
            "train_names": train_names,
            "train_names_sha256": names_sha256(train_names),
            "calibration_a_names": calibration_a_names,
            "calibration_a_names_sha256": names_sha256(calibration_a_names),
            "calibration_b_names": calibration_b_names,
            "calibration_b_names_sha256": names_sha256(calibration_b_names),
            "pairwise_whole_source_disjoint": True,
        },
        "artifacts": {
            "manifest": {"path": args.manifest, "sha256": sha256(args.manifest)},
            "quarantine": {"path": args.quarantine, "sha256": sha256(args.quarantine)},
            "denoiser": {"path": args.denoiser, "sha256": sha256(args.denoiser)},
            "embedding": {
                "path": args.embedding_checkpoint,
                "sha256": sha256(args.embedding_checkpoint),
            },
            "trainer": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
            "model_source": {
                "path": str(REPO_ROOT / "src/puzzle_assembly/continuation_net.py"),
                "sha256": sha256(REPO_ROOT / "src/puzzle_assembly/continuation_net.py"),
            },
        },
        "denoiser_metadata": denoiser_metadata,
        "embedding_metadata": embedding_metadata,
        "epochs": [],
        "calibration_a": [],
    }

    selected_alpha: float | None = None
    selected_checkpoint: Path | None = None
    for epoch in range(args.epochs):
        epoch_record = train_one_epoch(
            core_model,
            forward_model,
            optimizer,
            scaler,
            train_names,
            epoch,
            args=args,
            restorer=restorer,
            device=device,
        )
        checkpoint = output_root / f"continuation_net0_epoch_{epoch:02d}.pt"
        checkpoint_metadata = {
            "schema_version": 1,
            "kind": "continuation_net0_training_checkpoint",
            "epoch": epoch,
            "seed": args.seed,
            "train_names": train_names,
            "train_names_sha256": names_sha256(train_names),
            "denoiser_sha256": sha256(args.denoiser),
            "embedding_sha256": sha256(args.embedding_checkpoint),
            "trainer_sha256": sha256(__file__),
            "model_source_sha256": sha256(
                REPO_ROOT / "src/puzzle_assembly/continuation_net.py"
            ),
        }
        save_continuation_net0_checkpoint(
            checkpoint,
            core_model,
            metadata=checkpoint_metadata,
            optimizer_state=optimizer.state_dict(),
            training_state={
                "epoch": epoch,
                "scaler": scaler.state_dict(),
                "epoch_record": epoch_record,
            },
        )
        epoch_record["checkpoint"] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
        report["epochs"].append(epoch_record)

        a_records, _ = evaluate_split(
            forward_model,
            calibration_a_names,
            "calibration-a",
            args=args,
            restorer=restorer,
            embedding=embedding,
            device=device,
        )
        a_decision = calibration_a_decision(a_records)
        report["calibration_a"].append(
            {"epoch": epoch, "checkpoint": epoch_record["checkpoint"], "records": a_records, **a_decision}
        )
        write_report(report_path, report)
        print(
            json.dumps(
                {
                    "stage": "calibration-a-decision",
                    "epoch": epoch,
                    "passed": a_decision["passed"],
                    "selected_alpha": a_decision["selected_alpha"],
                    "conditions": a_decision["conditions"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if a_decision["passed"]:
            selected_alpha = float(a_decision["selected_alpha"])
            selected_checkpoint = checkpoint
            break

    if selected_alpha is None or selected_checkpoint is None:
        report["status"] = "stop_calibration_a_no_signal"
        report["seconds"] = time.time() - started
        write_report(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path)}), flush=True)
        return

    report["freeze"] = {
        "epoch": int(report["calibration_a"][-1]["epoch"]),
        "alpha": selected_alpha,
        "checkpoint": {"path": str(selected_checkpoint), "sha256": sha256(selected_checkpoint)},
        "freeze_basis": "source-macro MRR on calibration A under predeclared early-stop conditions",
    }
    report["status"] = "checkpoint_alpha_frozen_before_calibration_b"
    write_report(report_path, report)

    b_records, b_scores = evaluate_split(
        forward_model,
        calibration_b_names,
        "calibration-b",
        args=args,
        restorer=restorer,
        embedding=embedding,
        device=device,
        keep_scores=True,
        alphas=(selected_alpha,),
    )
    b_decision = calibration_b_decision(b_records, selected_alpha)
    report["calibration_b"] = {"records": b_records, **b_decision}
    if not b_decision["passed"]:
        report["status"] = "stop_calibration_b_retrieval_gate_failed"
        report["seconds"] = time.time() - started
        write_report(report_path, report)
        print(json.dumps({"status": report["status"], "gate": b_decision}), flush=True)
        return

    qap_records = [
        qap_record(record, scores, selected_alpha, args=args) for record, scores in b_scores
    ]
    qap_gate_summary = qap_summary(qap_records)
    report["qap"] = {
        "policy": {
            "iterations": QAP_ITERATIONS,
            "restarts": QAP_RESTARTS,
            "refine_swaps": 8,
            "refine_weak_cells": 32,
            "initial": "frozen HBT soft-cycle layout",
            "run_condition": "calibration-B retrieval gate passed",
        },
        "summary": qap_gate_summary,
        "records": qap_records,
    }
    report["status"] = (
        "pass_retrieval_and_qap_transfer_gate"
        if qap_gate_summary["passed"]
        else "stop_qap_no_transfer"
    )
    report["seconds"] = time.time() - started
    write_report(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "report_sha256": sha256(report_path),
                "freeze": report["freeze"],
                "calibration_b_gate": b_decision["conditions"],
                "qap_summary": report["qap"]["summary"],
                "safe_for_submission": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
