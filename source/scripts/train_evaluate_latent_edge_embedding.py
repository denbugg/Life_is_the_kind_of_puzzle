#!/usr/bin/env python3
"""Train and gate TileNAF-latent side embeddings on two exact panels.

This is a bounded Stage-1 experiment.  The learned model sees frozen TileNAF
decoder features together with raw/restored RGB, trains against all 575 valid
neighbours, and may alter only a frozen C1/HBT/W4 candidate union.  QAP and
submission generation are intentionally absent: they open only after the
retrieval gate passes on both source-disjoint panels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.geometry import TILE_COUNT, true_neighbour_slots
from puzzle_assembly.latent_edge_embedding import (
    LatentSideEmbeddingNet,
    blend_topk_rank_residual,
    compatibility_from_outputs,
    save_latent_edge_checkpoint,
)
from puzzle_assembly.learned import (
    candidate_union,
    direction_labels,
    embedding_hard_triplet_loss,
    embedding_retrieval_metrics,
    learned_compatibility,
    load_embedding_checkpoint,
)
from puzzle_assembly.metrics import retrieval_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_HBT = (
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/"
    "hbt_d320_denoised_rgb_sobel.pt"
)
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"
REPORT_NAME = "latent_edge_embedding_report.json"
CHECKPOINT_NAME = "latent_edge_embedding.pt"


@dataclass(frozen=True)
class Runtime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--hbt-checkpoint", default=DEFAULT_HBT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--train-offset", type=int, default=4096)
    parser.add_argument("--train-sources", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--panels", default="primary_kornia,independent_libjpeg")
    parser.add_argument("--selection-split", default="assembly_incremental_gate")
    parser.add_argument("--selection-offset", type=int, default=192)
    parser.add_argument("--selection-sources", type=int, default=16)
    parser.add_argument("--holdout-offset", type=int, default=208)
    parser.add_argument("--holdout-sources", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=384)
    parser.add_argument("--side-band", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--cross-entropy-weight", type=float, default=0.5)
    parser.add_argument("--embedding-l2-weight", type=float, default=1e-4)
    parser.add_argument("--outside-weight", type=float, default=0.2)
    parser.add_argument("--candidate-loss-weight", type=float, default=1.0)
    parser.add_argument("--full-loss-weight", type=float, default=0.25)
    parser.add_argument("--train-candidate-k", type=int, default=64)
    parser.add_argument("--denoise-batch-size", type=int, default=192)
    parser.add_argument("--classical-chunk-size", type=int, default=64)
    parser.add_argument("--candidate-top-k", type=int, default=32)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--alphas", default="0,0.025,0.05,0.1,0.2")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.train_sources = 2
        args.epochs = 1
        args.selection_sources = 1
        args.holdout_sources = 1
        args.model_dim = 32
        args.embedding_dim = 32
        args.layers = 1
        args.heads = 4
        args.feedforward_dim = 64
        args.denoise_batch_size = 64
        args.candidate_top_k = 4
        args.candidate_cap = 8
        args.train_candidate_k = 8
    return args


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _init_runtime(seed: int) -> Runtime:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    resolved_seed = seed + 1009 * rank
    random.seed(resolved_seed)
    np.random.seed(resolved_seed % (2**32 - 1))
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
    return Runtime(rank, world_size, local_rank, device)


def _barrier(runtime: Runtime) -> None:
    if runtime.world_size > 1:
        dist.barrier()


def _gather(values: Any, runtime: Runtime) -> list[Any]:
    if runtime.world_size == 1:
        return [values]
    output: list[Any] = [None] * runtime.world_size
    dist.all_gather_object(output, values)
    return output


def _primary_action(runtime: Runtime, action: Any) -> None:
    """Run a filesystem action on rank zero and broadcast any failure."""

    error = None
    if runtime.primary:
        try:
            action()
        except Exception as exc:  # propagated identically before any barrier
            error = f"{type(exc).__name__}: {exc}"
    if runtime.world_size > 1:
        values = [error]
        dist.broadcast_object_list(values, src=0)
        error = values[0]
    if error is not None:
        raise RuntimeError(f"rank-zero action failed: {error}")


def _hardware_probe(runtime: Runtime) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rank": runtime.rank,
        "world_size": runtime.world_size,
        "device": str(runtime.device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if runtime.device.type == "cuda":
        probe = torch.arange(4096, device=runtime.device, dtype=torch.float32)
        result.update(
            {
                "gpu": torch.cuda.get_device_name(runtime.device),
                "capability": list(torch.cuda.get_device_capability(runtime.device)),
                "total_memory": int(
                    torch.cuda.get_device_properties(runtime.device).total_memory
                ),
                "tensor_probe": float(probe.sin().square().mean().item()),
            }
        )
        if runtime.primary:
            result["nvidia_smi"] = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip().splitlines()
    return result


def _panels(args: argparse.Namespace) -> list[str]:
    return [value.strip() for value in args.panels.split(",") if value.strip()]


def _alphas(args: argparse.Namespace) -> list[float]:
    values = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("alphas must be finite non-negative values")
    if 0.0 not in values:
        raise ValueError("alphas must include exact identity alpha=0")
    return sorted(set(values))


def _validate_args(args: argparse.Namespace, runtime: Runtime) -> None:
    panels = _panels(args)
    if panels != ["primary_kornia", "independent_libjpeg"]:
        raise ValueError("the exact two-panel order is required")
    _alphas(args)
    if args.train_offset < 4096:
        raise ValueError("train-offset must avoid HBT edge_train[0:4096]")
    if args.train_sources <= 0 or args.train_sources % runtime.world_size:
        raise ValueError("train-sources must be positive and divisible by world size")
    if min(args.epochs, args.selection_sources, args.holdout_sources) <= 0:
        raise ValueError("epoch and evaluation source counts must be positive")
    if args.smoke:
        if args.selection_split != "edge_development":
            raise ValueError("smoke must use previously exposed edge_development sources")
    else:
        if args.selection_split != "assembly_incremental_gate":
            raise ValueError("full pilot requires assembly_incremental_gate")
        if args.selection_offset < 192:
            raise ValueError("selection must avoid assembly_incremental_gate[0:192]")
    if args.holdout_offset < args.selection_offset + args.selection_sources:
        raise ValueError("holdout must be source-disjoint and follow selection")
    if args.candidate_cap < args.candidate_top_k:
        raise ValueError("candidate-cap must be at least candidate-top-k")
    if args.train_candidate_k < 2 or args.train_candidate_k >= TILE_COUNT:
        raise ValueError("train-candidate-k must be in [2,575]")
    if args.candidate_loss_weight <= 0.0 or args.full_loss_weight < 0.0:
        raise ValueError("candidate loss weight must be positive and full loss non-negative")
    if not args.smoke:
        if runtime.world_size != 2 or runtime.device.type != "cuda":
            raise RuntimeError("full pilot requires torchrun on exactly two GPUs")
        devices = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if len(devices) != 2 or not all("T4" in value.upper() for value in devices):
            raise RuntimeError(f"full pilot requires exactly two Tesla T4s, got {devices}")


@torch.no_grad()
def _extract_tilenaf_views(
    restorer: nn.Module,
    raw_tiles: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    raw_tiles = np.asarray(raw_tiles)
    if raw_tiles.shape != (TILE_COUNT, 20, 20, 3) or raw_tiles.dtype != np.uint8:
        raise ValueError("raw tiles must be uint8 576x20x20x3")
    if not hasattr(restorer, "forward_with_features"):
        raise TypeError("restorer does not expose forward_with_features")
    raw = torch.from_numpy(np.ascontiguousarray(raw_tiles.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    ).div_(255.0)
    restored_parts: list[torch.Tensor] = []
    latent_parts: list[torch.Tensor] = []
    restorer.eval()
    for start in range(0, len(raw), batch_size):
        restored, latent = restorer.forward_with_features(raw[start : start + batch_size])
        restored_parts.append(restored.detach())
        latent_parts.append(latent.detach())
    restored = torch.cat(restored_parts)
    latent = torch.cat(latent_parts)
    restored_uint8 = (
        restored.float()
        .cpu()
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .byte()
        .permute(0, 2, 3, 1)
        .numpy()
    )
    return raw, restored, latent, restored_uint8


def _prepare_source(
    name: str,
    panel_name: str,
    *,
    stage: str,
    replica: int,
    args: argparse.Namespace,
    runtime: Runtime,
    restorer: nn.Module,
    hbt_model: nn.Module,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    Any,
    int,
    tuple[np.ndarray, np.ndarray],
]:
    clean = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
    seed = per_source_seed(args.seed, f"latent-edge-{stage}-{panel_name}", name, replica)
    exact = make_exact_panel(clean, panel=panel_name, seed=seed)
    raw, restored, latent, restored_uint8 = _extract_tilenaf_views(
        restorer,
        exact.slot_tiles,
        device=runtime.device,
        batch_size=args.denoise_batch_size,
    )
    hbt, _ = learned_compatibility(
        hbt_model, restored_uint8, device=runtime.device, name="train_frozen_hbt"
    )
    train_candidates = candidate_union(
        {hbt.name: hbt},
        names=[hbt.name],
        per_score_top_k=args.train_candidate_k,
        cap=args.train_candidate_k,
    )
    return (
        raw,
        restored,
        latent,
        restored_uint8,
        direction_labels(exact.slot_to_target),
        seed,
        train_candidates,
    )


def _model(args: argparse.Namespace) -> LatentSideEmbeddingNet:
    return LatentSideEmbeddingNet(
        latent_channels=48,
        model_dim=args.model_dim,
        embedding_dim=args.embedding_dim,
        layers=args.layers,
        heads=args.heads,
        feedforward_dim=args.feedforward_dim,
        side_band=args.side_band,
        dropout=args.dropout,
        temperature=args.temperature,
    )


def candidate_aligned_loss(
    outputs: dict[str, torch.Tensor],
    labels: Any,
    candidates: tuple[np.ndarray, np.ndarray],
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Listwise CE and conditional retrieval inside frozen HBT proposals."""

    device = outputs["q_right"].device
    directional_losses = []
    covered_count = 0
    total_count = 0
    ranks = []
    for query_name, key_name, queries_array, targets_array, selected in (
        (
            "q_right",
            "k_left",
            labels.right_queries,
            labels.right_targets,
            candidates[0],
        ),
        (
            "q_down",
            "k_up",
            labels.down_queries,
            labels.down_targets,
            candidates[1],
        ),
    ):
        queries = torch.as_tensor(queries_array, device=device, dtype=torch.long)
        targets = torch.as_tensor(targets_array, device=device, dtype=torch.long)
        proposal = torch.as_tensor(selected[queries_array], device=device, dtype=torch.long)
        matches = proposal == targets[:, None]
        covered = matches.any(dim=1)
        total_count += len(queries)
        covered_count += int(covered.sum().item())
        if not bool(covered.any()):
            continue
        covered_queries = queries[covered]
        covered_proposals = proposal[covered]
        target_position = matches[covered].float().argmax(dim=1)
        query_values = outputs[query_name][covered_queries]
        key_values = outputs[key_name][covered_proposals]
        logits = torch.einsum("qd,qkd->qk", query_values, key_values) / temperature
        directional_losses.append(torch.nn.functional.cross_entropy(logits, target_position))
        order = logits.argsort(dim=1, descending=True)
        rank = (order == target_position[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        ranks.append(rank)
    if not directional_losses or not ranks:
        raise RuntimeError("frozen training proposals covered no true neighbours")
    loss = torch.stack(directional_losses).mean()
    rank = torch.cat(ranks).float()
    return loss, {
        "candidate_loss": float(loss.detach().cpu()),
        "candidate_coverage": covered_count / max(total_count, 1),
        "candidate_recall_at_1": float((rank <= 1).float().mean().cpu()),
        "candidate_recall_at_5": float((rank <= 5).float().mean().cpu()),
        "candidate_mrr": float((1.0 / rank).mean().cpu()),
    }


def _train_source(
    forward_model: nn.Module,
    core_model: LatentSideEmbeddingNet,
    source: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
        Any,
        int,
        tuple[np.ndarray, np.ndarray],
    ],
    *,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    runtime: Runtime,
) -> dict[str, float]:
    raw, restored, latent, _, labels, _, candidates = source
    forward_model.train()
    optimizer.zero_grad(set_to_none=True)
    amp = runtime.device.type == "cuda" and not args.no_amp
    with torch.autocast(device_type=runtime.device.type, dtype=torch.float16, enabled=amp):
        outputs = forward_model(raw, restored, latent)
        full_loss, metrics = embedding_hard_triplet_loss(
            outputs,
            labels,
            temperature=core_model.temperature,
            margin=args.triplet_margin,
            cross_entropy_weight=args.cross_entropy_weight,
            embedding_l2_weight=args.embedding_l2_weight,
            outside_weight=args.outside_weight,
        )
        candidate_loss, candidate_metrics = candidate_aligned_loss(
            outputs,
            labels,
            candidates,
            temperature=core_model.temperature,
        )
        loss = (
            args.candidate_loss_weight * candidate_loss
            + args.full_loss_weight * full_loss
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(core_model.parameters(), args.grad_clip)
    if not bool(torch.isfinite(grad_norm).item()):
        raise RuntimeError("non-finite latent-edge gradient")
    scaler.step(optimizer)
    scaler.update()
    retrieval = embedding_retrieval_metrics(
        outputs, labels, temperature=core_model.temperature
    )
    return {
        **metrics,
        **retrieval,
        **candidate_metrics,
        "full_loss": metrics["loss"],
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
    }


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        raise ValueError("cannot average empty records")
    keys = set.intersection(*(set(record) for record in records))
    return {
        key: float(np.mean([float(record[key]) for record in records]))
        for key in sorted(keys)
    }


def _frozen_scores(
    restored_uint8: np.ndarray,
    *,
    hbt_model: nn.Module,
    runtime: Runtime,
    classical_chunk_size: int,
) -> tuple[CompatibilityMatrices, CompatibilityMatrices, CompatibilityMatrices]:
    hbt, _ = learned_compatibility(
        hbt_model, restored_uint8, device=runtime.device, name="denoised_hbt"
    )
    bank = build_classical_score_bank(
        restored_uint8, prefix="denoised", chunk_size=classical_chunk_size
    )
    c1_names = [
        name
        for name in sorted(bank)
        if name.startswith("denoised_") and not name.endswith("_c2")
    ]
    c1 = fuse_ranked_scores(bank, names=c1_names, name="denoised_C1")
    w4 = fuse_ranked_scores(
        {c1.name: c1, hbt.name: hbt},
        names=[c1.name, hbt.name],
        weights={hbt.name: 4.0},
        name="frozen_C1_HBTw4",
    )
    return hbt, c1, w4


def _candidate_coverage(
    candidates: tuple[np.ndarray, np.ndarray], slot_to_target: np.ndarray
) -> float:
    truth = true_neighbour_slots(slot_to_target)
    hits = []
    for selected, targets in zip(candidates, truth, strict=True):
        queries = np.flatnonzero(targets >= 0)
        hits.extend(
            int(targets[query] in selected[query]) for query in queries.tolist()
        )
    return float(np.mean(hits))


def conditional_candidate_metrics(
    compatibility: CompatibilityMatrices,
    candidates: tuple[np.ndarray, np.ndarray],
    slot_to_target: np.ndarray,
) -> dict[str, float]:
    """Retrieval among frozen proposals, conditional on truth being present."""

    truth = true_neighbour_slots(slot_to_target)
    ranks = []
    covered_count = 0
    total_count = 0
    for direction_name, selected, targets in zip(
        ("right", "down"), candidates, truth, strict=True
    ):
        matrix = getattr(compatibility, direction_name)
        queries = np.flatnonzero(targets >= 0)
        total_count += len(queries)
        for query in queries.tolist():
            proposal = selected[query]
            match = np.flatnonzero(proposal == targets[query])
            if len(match) == 0:
                continue
            covered_count += 1
            order = np.argsort(matrix[query, proposal], kind="stable")
            ranks.append(int(np.flatnonzero(order == match[0])[0]) + 1)
    if not ranks:
        raise RuntimeError("candidate union covered no true neighbours")
    rank = np.asarray(ranks, dtype=np.float64)
    return {
        "coverage": covered_count / max(total_count, 1),
        "recall_at_1": float(np.mean(rank <= 1)),
        "recall_at_5": float(np.mean(rank <= 5)),
        "recall_at_32": float(np.mean(rank <= 32)),
        "mrr": float(np.mean(1.0 / rank)),
        "covered_queries": float(covered_count),
        "total_queries": float(total_count),
    }


@torch.no_grad()
def _evaluate_record(
    name: str,
    panel_name: str,
    *,
    stage: str,
    args: argparse.Namespace,
    runtime: Runtime,
    restorer: nn.Module,
    hbt_model: nn.Module,
    model: LatentSideEmbeddingNet,
) -> dict[str, Any]:
    clean = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
    seed = per_source_seed(args.seed, f"latent-edge-{stage}-{panel_name}", name, 0)
    exact = make_exact_panel(clean, panel=panel_name, seed=seed)
    raw, restored, latent, restored_uint8 = _extract_tilenaf_views(
        restorer,
        exact.slot_tiles,
        device=runtime.device,
        batch_size=args.denoise_batch_size,
    )
    model.eval()
    outputs = model(raw, restored, latent)
    learned = compatibility_from_outputs(outputs)
    hbt, c1, w4 = _frozen_scores(
        restored_uint8,
        hbt_model=hbt_model,
        runtime=runtime,
        classical_chunk_size=args.classical_chunk_size,
    )
    bank = {score.name: score for score in (c1, hbt, w4)}
    candidates = candidate_union(
        bank,
        names=[c1.name, hbt.name, w4.name],
        per_score_top_k=args.candidate_top_k,
        cap=args.candidate_cap,
    )
    scores: dict[str, dict[str, float]] = {
        "hbt": retrieval_metrics(hbt, exact.slot_to_target)["combined"],
        "w4": retrieval_metrics(w4, exact.slot_to_target)["combined"],
        "learned": retrieval_metrics(learned, exact.slot_to_target)["combined"],
    }
    for alpha in _alphas(args):
        candidate = blend_topk_rank_residual(
            w4,
            learned,
            candidates,
            alpha=alpha,
            name=f"w4_latent_alpha_{alpha:g}",
        )
        scores[f"alpha_{alpha:g}"] = retrieval_metrics(
            candidate, exact.slot_to_target
        )["combined"]
    conditional = {
        "w4": conditional_candidate_metrics(w4, candidates, exact.slot_to_target),
        "learned": conditional_candidate_metrics(
            learned, candidates, exact.slot_to_target
        ),
    }
    return {
        "name": name,
        "panel": panel_name,
        "seed": seed,
        "candidate_coverage": _candidate_coverage(candidates, exact.slot_to_target),
        "scores": scores,
        "conditional": conditional,
    }


def _paired_stats(values: list[float], *, seed_label: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("paired deltas must be a finite non-empty vector")
    seed = int.from_bytes(hashlib.sha256(seed_label.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(5000, len(array)))
    bootstrap = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "bootstrap_95_lower": float(np.quantile(bootstrap, 0.025)),
        "wins": float(np.sum(array > 0.0)),
        "win_fraction": float(np.mean(array > 0.0)),
        "worst": float(array.min()),
        "count": float(len(array)),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact source-panel records without mixing panels."""

    result: dict[str, Any] = {}
    panels = sorted({str(record["panel"]) for record in records})
    for panel in panels:
        selected = [record for record in records if record["panel"] == panel]
        score_names = sorted(selected[0]["scores"])
        panel_result = {
            "source_count": len(selected),
            "candidate_coverage": float(
                np.mean([record["candidate_coverage"] for record in selected])
            ),
            "scores": {
                score: _mean([record["scores"][score] for record in selected])
                for score in score_names
            },
        }
        paired = {}
        for score in score_names:
            if not score.startswith("alpha_"):
                continue
            paired[score] = {}
            for metric in ("recall_at_1", "mrr"):
                deltas = []
                for record in selected:
                    baseline = max(
                        float(record["scores"]["hbt"][metric]),
                        float(record["scores"]["w4"][metric]),
                    )
                    deltas.append(float(record["scores"][score][metric]) - baseline)
                paired[score][metric] = _paired_stats(
                    deltas, seed_label=f"{panel}:{score}:{metric}"
                )
        panel_result["paired"] = paired
        if "conditional" in selected[0]:
            conditional_names = sorted(selected[0]["conditional"])
            panel_result["conditional"] = {
                score: _mean([record["conditional"][score] for record in selected])
                for score in conditional_names
            }
        result[panel] = panel_result
    return result


def alpha_assessment(
    aggregate: dict[str, Any], alpha: float
) -> dict[str, Any]:
    """Compare one alpha to the strongest frozen HBT/W4 metric by metric."""

    checks: dict[str, dict[str, Any]] = {}
    for panel, panel_result in sorted(aggregate.items()):
        candidate = panel_result["scores"][f"alpha_{alpha:g}"]
        hbt = panel_result["scores"]["hbt"]
        w4 = panel_result["scores"]["w4"]
        lower_is_better = {"median_rank", "q90_rank"}
        comparator = {
            key: (
                min(float(hbt[key]), float(w4[key]))
                if key in lower_is_better
                else max(float(hbt[key]), float(w4[key]))
            )
            for key in candidate
        }
        delta = {}
        for key in candidate:
            if key == "queries":
                continue
            if key in lower_is_better:
                delta[key] = comparator[key] - float(candidate[key])
            else:
                delta[key] = float(candidate[key]) - comparator[key]
        panel_checks = {
            "recall_at_1_delta_ge_0.010": delta["recall_at_1"] >= 0.010,
            "mrr_delta_ge_0.008": delta["mrr"] >= 0.008,
            "recall_at_5_delta_ge_minus_0.003": delta["recall_at_5"] >= -0.003,
            "recall_at_32_delta_ge_minus_0.005": delta["recall_at_32"] >= -0.005,
            "candidate_coverage_ge_0.68": panel_result["candidate_coverage"] >= 0.68,
            "recall_at_1_bootstrap_lower_gt_0": panel_result["paired"][
                f"alpha_{alpha:g}"
            ]["recall_at_1"]["bootstrap_95_lower"]
            > 0.0,
            "mrr_bootstrap_lower_gt_0": panel_result["paired"][f"alpha_{alpha:g}"][
                "mrr"
            ]["bootstrap_95_lower"]
            > 0.0,
            "recall_at_1_win_fraction_ge_0.625": panel_result["paired"][
                f"alpha_{alpha:g}"
            ]["recall_at_1"]["win_fraction"]
            >= 0.625,
            "recall_at_1_worst_ge_minus_0.02": panel_result["paired"][
                f"alpha_{alpha:g}"
            ]["recall_at_1"]["worst"]
            >= -0.02,
        }
        checks[panel] = {
            "candidate": candidate,
            "comparator": comparator,
            "delta": delta,
            "paired": panel_result["paired"][f"alpha_{alpha:g}"],
            "checks": panel_checks,
            "passed": all(panel_checks.values()),
        }
    passed = len(checks) == 2 and all(value["passed"] for value in checks.values())
    objective = min(value["delta"]["recall_at_1"] for value in checks.values())
    objective += 0.5 * min(value["delta"]["mrr"] for value in checks.values())
    return {
        "alpha": alpha,
        "panels": checks,
        "objective": float(objective),
        "passed": passed,
    }


def choose_alpha(aggregate: dict[str, Any], alphas: list[float]) -> dict[str, Any]:
    assessments = [alpha_assessment(aggregate, alpha) for alpha in alphas]
    passing = [item for item in assessments if item["passed"]]
    pool = passing if passing else assessments
    selected = max(pool, key=lambda item: (item["objective"], -item["alpha"]))
    return {
        "selected_alpha": selected["alpha"],
        "passed": bool(selected["passed"]),
        "selected": selected,
        "assessments": assessments,
    }


def _evaluate_split(
    names: list[str],
    *,
    stage: str,
    args: argparse.Namespace,
    runtime: Runtime,
    restorer: nn.Module,
    hbt_model: nn.Module,
    model: LatentSideEmbeddingNet,
) -> list[dict[str, Any]]:
    local_names = names[runtime.rank :: runtime.world_size]
    records = []
    for name in local_names:
        for panel_name in _panels(args):
            record = _evaluate_record(
                name,
                panel_name,
                stage=stage,
                args=args,
                runtime=runtime,
                restorer=restorer,
                hbt_model=hbt_model,
                model=model,
            )
            records.append(record)
            print(
                json.dumps(
                    {
                        "event": "latent_edge_eval_source",
                        "stage": stage,
                        "rank": runtime.rank,
                        "name": name,
                        "panel": panel_name,
                        "learned_r1": record["scores"]["learned"]["recall_at_1"],
                        "w4_r1": record["scores"]["w4"]["recall_at_1"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    gathered = _gather(records, runtime)
    return [record for shard in gathered for record in shard]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runtime = _init_runtime(args.seed)
    try:
        _validate_args(args, runtime)
        output_dir = Path(args.output_dir)
        report_path = output_dir / REPORT_NAME
        checkpoint_path = output_dir / CHECKPOINT_NAME
        def prepare_output() -> None:
            if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
                raise RuntimeError(f"output is not empty; pass --overwrite: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

        _primary_action(runtime, prepare_output)

        train_all = source_names_for_split(
            "edge_train",
            manifest_path=args.manifest,
            quarantine_path=args.quarantine,
        )
        selection_all = source_names_for_split(
            args.selection_split,
            manifest_path=args.manifest,
            quarantine_path=args.quarantine,
        )
        train_names = train_all[args.train_offset : args.train_offset + args.train_sources]
        selection_names = selection_all[
            args.selection_offset : args.selection_offset + args.selection_sources
        ]
        holdout_names = selection_all[
            args.holdout_offset : args.holdout_offset + args.holdout_sources
        ]
        if (
            len(train_names) != args.train_sources
            or len(selection_names) != args.selection_sources
            or len(holdout_names) != args.holdout_sources
        ):
            raise RuntimeError("requested source slice extends beyond its split")
        if set(selection_names) & set(holdout_names):
            raise RuntimeError("selection and holdout sources overlap")
        if set(train_names) & (set(selection_names) | set(holdout_names)):
            raise RuntimeError("training sources overlap selection or holdout")

        restorer, _, denoiser_metadata = load_restorer(
            args.denoiser,
            device=str(runtime.device),
            state="ema",
        )
        for parameter in restorer.parameters():
            parameter.requires_grad_(False)
        hbt_model, hbt_metadata = load_embedding_checkpoint(
            args.hbt_checkpoint, device=runtime.device
        )
        hbt_model.eval()
        for parameter in hbt_model.parameters():
            parameter.requires_grad_(False)

        model = _model(args).to(runtime.device)
        forward_model: nn.Module = model
        if runtime.world_size > 1:
            forward_model = DistributedDataParallel(
                model,
                device_ids=[runtime.local_rank],
                output_device=runtime.local_rank,
                broadcast_buffers=False,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        amp = runtime.device.type == "cuda" and not args.no_amp
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        hardware = _gather(_hardware_probe(runtime), runtime)
        started = time.perf_counter()
        history = []
        local_train_names = train_names[runtime.rank :: runtime.world_size]
        for epoch in range(args.epochs):
            epoch_records = []
            for index, name in enumerate(local_train_names):
                panel_name = _panels(args)[
                    (index + epoch + runtime.rank) % len(_panels(args))
                ]
                source = _prepare_source(
                    name,
                    panel_name,
                    stage="train",
                    replica=epoch,
                    args=args,
                    runtime=runtime,
                    restorer=restorer,
                    hbt_model=hbt_model,
                )
                metrics = _train_source(
                    forward_model,
                    model,
                    source,
                    args=args,
                    optimizer=optimizer,
                    scaler=scaler,
                    runtime=runtime,
                )
                epoch_records.append(metrics)
                print(
                    json.dumps(
                        {
                            "event": "latent_edge_train_source",
                            "rank": runtime.rank,
                            "epoch": epoch + 1,
                            "index": index + 1,
                            "count": len(local_train_names),
                            "name": name,
                            "panel": panel_name,
                            "loss": metrics["loss"],
                            "recall_at_1": metrics["recall_at_1"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            gathered_epoch = _gather(epoch_records, runtime)
            if runtime.primary:
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train": _mean([item for shard in gathered_epoch for item in shard]),
                        "seconds": time.perf_counter() - started,
                    }
                )
            _barrier(runtime)

        metadata = {
            "experiment": "tilenaf_latent_edge_stage1",
            "safe_for_submission": False,
            "seed": args.seed,
            "train_partition": f"edge_train[{args.train_offset}:{args.train_offset + len(train_names)}]",
            "train_names_sha256": _names_sha256(train_names),
            "panels": _panels(args),
            "epochs": args.epochs,
            "denoiser_sha256": _sha256(args.denoiser),
            "hbt_sha256": _sha256(args.hbt_checkpoint),
        }
        _primary_action(
            runtime,
            lambda: save_latent_edge_checkpoint(
                checkpoint_path, model, metadata={**metadata, "stage": "trained"}
            ),
        )

        selection_records = _evaluate_split(
            selection_names,
            stage="selection",
            args=args,
            runtime=runtime,
            restorer=restorer,
            hbt_model=hbt_model,
            model=model,
        )
        selection_aggregate = aggregate_records(selection_records)
        selection_gate = choose_alpha(selection_aggregate, _alphas(args))
        holdout_records: list[dict[str, Any]] = []
        holdout_aggregate = None
        holdout_gate = None
        if selection_gate["passed"]:
            holdout_records = _evaluate_split(
                holdout_names,
                stage="holdout",
                args=args,
                runtime=runtime,
                restorer=restorer,
                hbt_model=hbt_model,
                model=model,
            )
            holdout_aggregate = aggregate_records(holdout_records)
            holdout_gate = alpha_assessment(
                holdout_aggregate, float(selection_gate["selected_alpha"])
            )
        status = (
            "retrieval_gate_passed_qap_not_run"
            if holdout_gate is not None and holdout_gate["passed"]
            else (
                "stop_holdout_retrieval"
                if holdout_gate is not None
                else "stop_selection_retrieval"
            )
        )
        final_metadata = {
            **metadata,
            "stage": "retrieval_gated",
            "status": status,
            "selected_alpha": float(selection_gate["selected_alpha"]),
            "selection_passed": bool(selection_gate["passed"]),
            "holdout_passed": bool(holdout_gate and holdout_gate["passed"]),
        }
        _primary_action(
            runtime,
            lambda: save_latent_edge_checkpoint(
                checkpoint_path, model, metadata=final_metadata
            ),
        )

        def write_report() -> None:
            report = {
                "schema_version": 1,
                "kind": "tilenaf_latent_edge_stage1_report",
                "status": status,
                "safe_for_submission": False,
                "qap_run": False,
                "args": vars(args),
                "hardware": hardware,
                "model_config": model.config(),
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "denoiser_metadata": denoiser_metadata,
                "hbt_metadata": hbt_metadata,
                "partitions": {
                    "train": {
                        "slice": f"edge_train[{args.train_offset}:{args.train_offset + len(train_names)}]",
                        "names_sha256": _names_sha256(train_names),
                    },
                    "selection": {
                        "slice": f"{args.selection_split}[{args.selection_offset}:{args.selection_offset + len(selection_names)}]",
                        "names_sha256": _names_sha256(selection_names),
                    },
                    "holdout": {
                        "slice": f"{args.selection_split}[{args.holdout_offset}:{args.holdout_offset + len(holdout_names)}]",
                        "names_sha256": _names_sha256(holdout_names),
                        "opened": bool(selection_gate["passed"]),
                    },
                },
                "history": history,
                "selection": {
                    "aggregate": selection_aggregate,
                    "gate": selection_gate,
                    "records": selection_records,
                },
                "holdout": (
                    {
                        "aggregate": holdout_aggregate,
                        "gate": holdout_gate,
                        "records": holdout_records,
                    }
                    if holdout_gate is not None
                    else None
                ),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "seconds": time.perf_counter() - started,
            }
            _atomic_json(report_path, report)
            print(
                json.dumps(
                    {
                        "event": "latent_edge_complete",
                        "status": status,
                        "selected_alpha": selection_gate["selected_alpha"],
                        "report": str(report_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        _primary_action(runtime, write_report)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
