#!/usr/bin/env python3
"""Train a binary pixel verifier on the full C1/HBT candidate union."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from puzzle_assembly.binary_edge_verifier import (
    BinaryEdgeVerifierNet,
    save_binary_edge_verifier,
)
from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import ProposedEdge, grow_components_with_edges
from puzzle_assembly.geometry import TILE_COUNT, true_neighbour_slots
from puzzle_assembly.learned import (
    learned_compatibility,
    load_embedding_checkpoint,
    seam_pair_patches,
)
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8


SCORE_NAMES = ("c1", "hbt", "w1", "w4")
ORIGIN_BITS = (1, 2, 4, 8)


@dataclass(frozen=True)
class CandidateGraph:
    direction: np.ndarray
    source: np.ndarray
    destination: np.ndarray
    origin_mask: np.ndarray


@dataclass(frozen=True)
class PreparedSource:
    name: str
    panel: str
    seed: int
    raw_tiles: np.ndarray
    denoised_tiles: np.ndarray
    truth: np.ndarray
    scores: dict[str, CompatibilityMatrices]
    graph: CandidateGraph
    features: np.ndarray
    labels: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument(
        "--denoiser", default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
    )
    parser.add_argument(
        "--embedding-checkpoint",
        default="runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt",
    )
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument(
        "--quarantine", default="configs/denoise_validation_quarantine_v1.json"
    )
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-sources", type=int, default=256)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-sources", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--side-band", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape: {path}: {values.shape}")
    return values


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_probe(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        result["devices"] = []
        for index in range(torch.cuda.device_count()):
            value = torch.randn(64, 64, device=f"cuda:{index}")
            result["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "tensor_op": float((value @ value).mean().cpu()),
                }
            )
        probe = subprocess.run(
            ["nvidia-smi"], capture_output=True, check=False, text=True
        )
        result["nvidia_smi"] = probe.stdout
    return result


def build_scores(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    embedding_model: torch.nn.Module,
    *,
    device: torch.device,
) -> dict[str, CompatibilityMatrices]:
    bank = build_classical_score_bank(denoised_tiles, prefix="denoised", chunk_size=64)
    classical_names = [
        name
        for name in sorted(bank)
        if name.startswith("denoised_") and not name.endswith("_c2")
    ]
    c1 = fuse_ranked_scores(bank, names=classical_names, name="c1")
    hbt, _ = learned_compatibility(
        embedding_model, denoised_tiles, device=device, name="hbt"
    )
    local = {"c1": c1, "hbt": hbt}
    w1 = fuse_ranked_scores(local, names=["c1", "hbt"], name="w1")
    w4 = fuse_ranked_scores(
        local, names=["c1", "hbt"], weights={"hbt": 4.0}, name="w4"
    )
    return {"c1": c1, "hbt": hbt, "w1": w1, "w4": w4}


def build_candidate_graph(scores: dict[str, CompatibilityMatrices]) -> CandidateGraph:
    directions, sources, destinations, masks = [], [], [], []
    for direction, side in ((0, "right"), (1, "down")):
        lookup: dict[tuple[int, int], int] = {}
        for score_name, outgoing_bit, incoming_bit in (
            ("c1", 1, 4),
            ("hbt", 2, 8),
        ):
            matrix = np.asarray(getattr(scores[score_name], side))
            safe = matrix.copy()
            np.fill_diagonal(safe, np.inf)
            outgoing = np.argpartition(safe, 31, axis=1)[:, :32]
            for source in range(TILE_COUNT):
                for destination in outgoing[source].tolist():
                    key = (source, int(destination))
                    lookup[key] = lookup.get(key, 0) | outgoing_bit
            incoming = np.argpartition(safe, 7, axis=0)[:8, :]
            for destination in range(TILE_COUNT):
                for source in incoming[:, destination].tolist():
                    key = (int(source), destination)
                    lookup[key] = lookup.get(key, 0) | incoming_bit
        for (source, destination), mask in sorted(lookup.items()):
            directions.append(direction)
            sources.append(source)
            destinations.append(destination)
            masks.append(mask)
    return CandidateGraph(
        direction=np.asarray(directions, dtype=np.uint8),
        source=np.asarray(sources, dtype=np.int32),
        destination=np.asarray(destinations, dtype=np.int32),
        origin_mask=np.asarray(masks, dtype=np.uint8),
    )


def rank_robust(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(matrix, dtype=np.float64)
    row_order = np.argsort(values, axis=1, kind="stable")
    row_rank = np.empty_like(row_order)
    np.put_along_axis(
        row_rank,
        row_order,
        np.broadcast_to(np.arange(TILE_COUNT), row_order.shape),
        axis=1,
    )
    column_order = np.argsort(values, axis=0, kind="stable")
    column_rank = np.empty_like(column_order)
    np.put_along_axis(
        column_rank,
        column_order,
        np.broadcast_to(np.arange(TILE_COUNT)[:, None], column_order.shape),
        axis=0,
    )
    finite = values.copy()
    np.fill_diagonal(finite, np.nan)
    row_median = np.nanmedian(finite, axis=1)
    row_scale = np.nanpercentile(finite, 75, axis=1) - np.nanpercentile(
        finite, 25, axis=1
    )
    column_median = np.nanmedian(finite, axis=0)
    column_scale = np.nanpercentile(finite, 75, axis=0) - np.nanpercentile(
        finite, 25, axis=0
    )
    row_z = np.clip(
        (values - row_median[:, None]) / np.maximum(row_scale[:, None], 1e-8),
        -8.0,
        8.0,
    ) / 8.0
    column_z = np.clip(
        (values - column_median[None, :])
        / np.maximum(column_scale[None, :], 1e-8),
        -8.0,
        8.0,
    ) / 8.0
    return (
        row_rank.astype(np.float32) / (TILE_COUNT - 1),
        column_rank.astype(np.float32) / (TILE_COUNT - 1),
        row_z.astype(np.float32),
        column_z.astype(np.float32),
    )


def feature_names() -> list[str]:
    names = []
    for score in SCORE_NAMES:
        names.extend(
            [
                f"{score}_row_rank",
                f"{score}_column_rank",
                f"{score}_row_robust",
                f"{score}_column_robust",
            ]
        )
    names.extend(
        [
            "rank_min",
            "rank_mean",
            "rank_std",
            "origin_c1_out32",
            "origin_hbt_out32",
            "origin_c1_in8",
            "origin_hbt_in8",
            "origin_popcount",
            "direction_down",
        ]
    )
    return names


def candidate_features(
    scores: dict[str, CompatibilityMatrices], graph: CandidateGraph
) -> np.ndarray:
    output = np.empty((len(graph.direction), len(feature_names())), dtype=np.float32)
    for direction, side in ((0, "right"), (1, "down")):
        indices = np.flatnonzero(graph.direction == direction)
        source, destination = graph.source[indices], graph.destination[indices]
        parts, row_ranks = [], []
        for name in SCORE_NAMES:
            rank_values = rank_robust(getattr(scores[name], side))
            parts.extend([value[source, destination] for value in rank_values])
            row_ranks.append(rank_values[0][source, destination])
        rank_stack = np.stack(row_ranks, axis=1)
        parts.extend(
            [rank_stack.min(axis=1), rank_stack.mean(axis=1), rank_stack.std(axis=1)]
        )
        mask = graph.origin_mask[indices]
        parts.extend(
            [((mask & bit) != 0).astype(np.float32) for bit in ORIGIN_BITS]
        )
        parts.append(
            np.asarray([int(value).bit_count() / 4.0 for value in mask], dtype=np.float32)
        )
        parts.append(np.full(len(indices), float(direction), dtype=np.float32))
        output[indices] = np.stack(parts, axis=1)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("non-finite candidate features")
    return output


def candidate_labels(graph: CandidateGraph, truth: np.ndarray) -> np.ndarray:
    right, down = true_neighbour_slots(truth)
    return np.where(
        graph.direction == 0,
        right[graph.source] == graph.destination,
        down[graph.source] == graph.destination,
    ).astype(np.float32)


def prepare_source(
    name: str,
    panel: str,
    seed: int,
    *,
    args: argparse.Namespace,
    restorer: torch.nn.Module,
    embedding_model: torch.nn.Module,
    device: torch.device,
) -> PreparedSource:
    target = read_rgb(Path(args.data_root) / "train/targets" / name)
    exact = make_exact_panel(target, panel=panel, seed=seed)
    denoised = restore_tiles_uint8(
        restorer,
        exact.slot_tiles,
        device,
        batch_size=args.denoise_batch_size,
    )
    scores = build_scores(exact.slot_tiles, denoised, embedding_model, device=device)
    graph = build_candidate_graph(scores)
    features = candidate_features(scores, graph)
    labels = candidate_labels(graph, exact.slot_to_target)
    return PreparedSource(
        name=name,
        panel=panel,
        seed=seed,
        raw_tiles=exact.slot_tiles,
        denoised_tiles=denoised,
        truth=exact.slot_to_target,
        scores=scores,
        graph=graph,
        features=features,
        labels=labels,
    )


def selected_training_indices(
    source: PreparedSource, *, negative_ratio: int, rng: np.random.Generator
) -> np.ndarray:
    positive = np.flatnonzero(source.labels > 0.5)
    negative = np.flatnonzero(source.labels < 0.5)
    if len(positive) == 0:
        raise RuntimeError("candidate graph contains no positive edges")
    hard_score = source.features[negative, 16]
    hard_pool_size = min(len(negative), len(positive) * negative_ratio * 3)
    hard_pool = negative[np.argsort(hard_score, kind="stable")[:hard_pool_size]]
    take = min(len(hard_pool), len(positive) * negative_ratio)
    selected_negative = rng.choice(hard_pool, size=take, replace=False)
    selected = np.concatenate([positive, selected_negative])
    rng.shuffle(selected)
    return selected.astype(np.int64)


def tile_tensor(tiles: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    )


def batch_inputs(
    source: PreparedSource,
    indices: np.ndarray,
    *,
    device: torch.device,
    side_band: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    graph = source.graph
    first = torch.as_tensor(graph.source[indices], device=device, dtype=torch.long)
    second = torch.as_tensor(graph.destination[indices], device=device, dtype=torch.long)
    direction = torch.as_tensor(graph.direction[indices], device=device, dtype=torch.long)
    raw = seam_pair_patches(
        tile_tensor(source.raw_tiles, device),
        first,
        second,
        direction,
        side_band=side_band,
    )
    denoised = seam_pair_patches(
        tile_tensor(source.denoised_tiles, device),
        first,
        second,
        direction,
        side_band=side_band,
    )
    tabular = torch.as_tensor(source.features[indices], device=device)
    labels = torch.as_tensor(source.labels[indices], device=device)
    return raw, denoised, tabular, labels


def binary_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.uint8)
    probability = np.asarray(probability, dtype=np.float64)
    order = np.argsort(-probability, kind="stable")
    ordered = labels[order]
    positives = int(labels.sum())
    negatives = len(labels) - positives
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(labels) + 1)
    average_precision = float(precision[ordered == 1].mean())
    ascending = np.argsort(probability, kind="stable")
    ranks = np.empty(len(labels), dtype=np.float64)
    ranks[ascending] = np.arange(1, len(labels) + 1)
    auc = float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )
    return {
        "edges": int(len(labels)),
        "positives": positives,
        "positive_rate": float(labels.mean()),
        "average_precision": average_precision,
        "roc_auc": auc,
    }


def precision_frontier(
    labels: np.ndarray, probability: np.ndarray, *, target_precision: float
) -> dict[str, Any]:
    order = np.argsort(-probability, kind="stable")
    ordered = labels[order].astype(np.int64)
    cumulative = np.cumsum(ordered)
    count = np.arange(1, len(order) + 1)
    precision = cumulative / count
    eligible = np.flatnonzero(precision >= target_precision)
    chosen = int(eligible[np.argmax(cumulative[eligible])]) if len(eligible) else 0
    fixed = {}
    for selected in (128, 256, 512, 1024, 2048, 4096, 8192):
        if selected > len(order):
            continue
        index = selected - 1
        fixed[str(selected)] = {
            "threshold": float(probability[order[index]]),
            "precision": float(precision[index]),
            "true_edges": int(cumulative[index]),
        }
    return {
        "target_precision": target_precision,
        "threshold": float(probability[order[chosen]]),
        "selected_edges": int(chosen + 1),
        "true_edges": int(cumulative[chosen]),
        "precision": float(precision[chosen]),
        "recall": float(cumulative[chosen] / labels.sum()),
        "target_achieved": bool(len(eligible)),
        "fixed_prefixes": fixed,
    }


@torch.inference_mode()
def score_prepared(
    model: torch.nn.Module,
    source: PreparedSource,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    raw_tiles = tile_tensor(source.raw_tiles, device)
    denoised_tiles = tile_tensor(source.denoised_tiles, device)
    outputs = []
    for start in range(0, len(source.labels), args.score_batch_size):
        indices = np.arange(start, min(start + args.score_batch_size, len(source.labels)))
        first = torch.as_tensor(source.graph.source[indices], device=device, dtype=torch.long)
        second = torch.as_tensor(
            source.graph.destination[indices], device=device, dtype=torch.long
        )
        direction = torch.as_tensor(
            source.graph.direction[indices], device=device, dtype=torch.long
        )
        raw = seam_pair_patches(
            raw_tiles, first, second, direction, side_band=args.side_band
        )
        denoised = seam_pair_patches(
            denoised_tiles, first, second, direction, side_band=args.side_band
        )
        tabular = torch.as_tensor(source.features[indices], device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(raw, denoised, tabular)
        outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(outputs)


def component_metrics(
    source: PreparedSource, probability: np.ndarray, threshold: float
) -> dict[str, Any]:
    indices = np.flatnonzero(probability >= threshold)
    indices = indices[np.argsort(-probability[indices], kind="stable")]
    proposals = [
        ProposedEdge(
            first=int(source.graph.source[index]),
            second=int(source.graph.destination[index]),
            dx=1 if int(source.graph.direction[index]) == 0 else 0,
            dy=0 if int(source.graph.direction[index]) == 0 else 1,
            cost=float(1.0 - probability[index]),
            margin=float(probability[index]),
            reciprocal=False,
            in_loop=int(source.graph.origin_mask[index]).bit_count() >= 2,
        )
        for index in indices.tolist()
    ]
    components, accepted = grow_components_with_edges(proposals)
    lookup = {
        (
            int(source.graph.direction[index]),
            int(source.graph.source[index]),
            int(source.graph.destination[index]),
        ): bool(source.labels[index])
        for index in range(len(source.labels))
    }
    accepted_true = sum(
        lookup[(0 if edge.dx else 1, edge.first, edge.second)] for edge in accepted
    )
    return {
        "selected_edges": len(proposals),
        "accepted_edges": len(accepted),
        "accepted_precision": float(accepted_true / max(1, len(accepted))),
        "largest_component": int(max(len(component) for component in components)),
        "non_singleton_tile_fraction": float(
            sum(len(component) for component in components if len(component) > 1)
            / TILE_COUNT
        ),
    }


def main() -> None:
    args = parse_args()
    if min(args.train_sources, args.val_sources, args.epochs) <= 0:
        raise SystemExit("source counts and epochs must be positive")
    output = Path(args.output)
    report_path = Path(args.report) if args.report else output.with_suffix(".json")
    if (output.exists() or report_path.exists()) and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    probe = hardware_probe(device)
    restorer, restored_device, denoiser_metadata = load_restorer(
        args.denoiser, device=str(device)
    )
    if restored_device != device:
        raise RuntimeError("denoiser device drift")
    embedding_model, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    restorer.eval()
    embedding_model.eval()
    for frozen in (restorer, embedding_model):
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
    names = feature_names()
    core_model = BinaryEdgeVerifierNet(
        tabular_dim=len(names),
        channels=args.channels,
        side_band=args.side_band,
        dropout=args.dropout,
    ).to(device)
    forward_model: torch.nn.Module = core_model
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        forward_model = torch.nn.DataParallel(core_model)
    optimizer = torch.optim.AdamW(
        core_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_names = source_names_for_split(
        "edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.train_offset : args.train_offset + args.train_sources]
    val_names = source_names_for_split(
        "edge_development", manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.val_offset : args.val_offset + args.val_sources]
    if set(train_names) & set(val_names):
        raise RuntimeError("whole-source train/validation overlap")
    history = []
    best_state: dict[str, torch.Tensor] | None = None
    best_average_precision = -1.0
    started = time.time()
    panels = ("primary_kornia", "independent_libjpeg")
    for epoch in range(args.epochs):
        forward_model.train()
        train_loss, train_correct, train_examples, train_positives = 0.0, 0, 0, 0
        epoch_started = time.time()
        for source_index, name in enumerate(train_names):
            panel = panels[(epoch + source_index) % len(panels)]
            seed = per_source_seed(
                args.seed, f"binary-edge-verifier-train-{panel}", name, epoch
            )
            prepared = prepare_source(
                name,
                panel,
                seed,
                args=args,
                restorer=restorer,
                embedding_model=embedding_model,
                device=device,
            )
            selected = selected_training_indices(
                prepared, negative_ratio=args.negative_ratio, rng=rng
            )
            for start in range(0, len(selected), args.batch_size):
                batch_indices = selected[start : start + args.batch_size]
                raw, denoised, tabular, labels = batch_inputs(
                    prepared,
                    batch_indices,
                    device=device,
                    side_band=args.side_band,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = forward_model(raw, denoised, tabular)
                    loss = F.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        pos_weight=labels.new_tensor(float(args.negative_ratio)),
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(core_model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                count = len(labels)
                train_loss += float(loss.detach().cpu()) * count
                train_correct += int(
                    ((logits.detach() >= 0) == (labels >= 0.5)).sum().cpu()
                )
                train_examples += count
                train_positives += int(labels.sum().cpu())
            print(
                json.dumps(
                    {
                        "stage": "train",
                        "epoch": epoch + 1,
                        "done": source_index + 1,
                        "total": len(train_names),
                        "panel": panel,
                        "candidate_edges": len(prepared.labels),
                        "candidate_positives": int(prepared.labels.sum()),
                    }
                ),
                flush=True,
            )

        validation_records = []
        prepared_validation: list[PreparedSource] = []
        all_labels, all_probability = [], []
        for source_index, name in enumerate(val_names):
            for panel in panels:
                seed = per_source_seed(
                    args.seed, f"binary-edge-verifier-val-{panel}", name, 0
                )
                prepared = prepare_source(
                    name,
                    panel,
                    seed,
                    args=args,
                    restorer=restorer,
                    embedding_model=embedding_model,
                    device=device,
                )
                probability = score_prepared(
                    forward_model, prepared, args=args, device=device
                )
                metrics = binary_metrics(prepared.labels, probability)
                metrics.update(
                    {
                        "name": name,
                        "panel": panel,
                        "seed": seed,
                        "candidate_recall": float(prepared.labels.sum() / 1104.0),
                    }
                )
                validation_records.append(metrics)
                prepared_validation.append(prepared)
                all_labels.append(prepared.labels)
                all_probability.append(probability)
                print(
                    json.dumps(
                        {
                            "stage": "validation",
                            "epoch": epoch + 1,
                            "done": len(validation_records),
                            "total": len(val_names) * len(panels),
                        }
                    ),
                    flush=True,
                )
        pooled_labels = np.concatenate(all_labels)
        pooled_probability = np.concatenate(all_probability)
        pooled_metrics = binary_metrics(pooled_labels, pooled_probability)
        frontier = precision_frontier(
            pooled_labels,
            pooled_probability,
            target_precision=args.target_precision,
        )
        threshold = float(frontier["threshold"])
        component_records = []
        for prepared, probability in zip(
            prepared_validation, all_probability, strict=True
        ):
            values = component_metrics(prepared, probability, threshold)
            values.update({"name": prepared.name, "panel": prepared.panel})
            component_records.append(values)
        epoch_record = {
            "epoch": epoch + 1,
            "train": {
                "loss": float(train_loss / train_examples),
                "accuracy": float(train_correct / train_examples),
                "examples": train_examples,
                "positives": train_positives,
                "seconds": time.time() - epoch_started,
            },
            "validation": pooled_metrics,
            "precision_frontier": frontier,
            "mean_component_accepted_precision": float(
                np.mean([value["accepted_precision"] for value in component_records])
            ),
            "mean_largest_component": float(
                np.mean([value["largest_component"] for value in component_records])
            ),
            "mean_non_singleton_tile_fraction": float(
                np.mean([value["non_singleton_tile_fraction"] for value in component_records])
            ),
            "validation_records": validation_records,
            "component_records": component_records,
        }
        history.append(epoch_record)
        if pooled_metrics["average_precision"] > best_average_precision:
            best_average_precision = pooled_metrics["average_precision"]
            best_state = copy.deepcopy(core_model.state_dict())
        print(json.dumps({"epoch_result": epoch_record}, sort_keys=True), flush=True)

    best = max(history, key=lambda value: value["validation"]["average_precision"])
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint state")
    core_model.load_state_dict(best_state, strict=True)
    metadata = {
        "seed": args.seed,
        "train_names": train_names,
        "val_names": val_names,
        "best_epoch": best["epoch"],
        "best_validation": best["validation"],
        "best_precision_frontier": best["precision_frontier"],
        "denoiser_sha256": sha256(args.denoiser),
        "embedding_sha256": sha256(args.embedding_checkpoint),
    }
    save_binary_edge_verifier(
        output,
        core_model,
        feature_names=names,
        metadata=metadata,
    )
    report = {
        "schema_version": 1,
        "kind": "binary_edge_verifier_training",
        "status": "continue" if (
            best["validation"]["average_precision"] >= 0.20
            and best["precision_frontier"]["precision"] >= 0.80
            and best["precision_frontier"]["true_edges"]
            >= 50 * len(val_names) * len(panels)
        ) else "stop_or_redesign",
        "args": vars(args),
        "hardware": probe,
        "model_config": core_model.config(),
        "feature_names": names,
        "train_names": train_names,
        "val_names": val_names,
        "whole_source_disjoint": not bool(set(train_names) & set(val_names)),
        "denoiser_metadata": denoiser_metadata,
        "embedding_metadata": embedding_metadata,
        "history": history,
        "best_epoch": best["epoch"],
        "best_validation": best["validation"],
        "best_precision_frontier": best["precision_frontier"],
        "checkpoint": str(output),
        "seconds": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "checkpoint": str(output),
                "checkpoint_sha256": sha256(output),
                "report": str(report_path),
                "report_sha256": sha256(report_path),
                "best_validation": best["validation"],
                "best_precision_frontier": best["precision_frontier"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
