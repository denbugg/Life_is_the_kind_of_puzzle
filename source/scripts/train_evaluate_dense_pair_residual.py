#!/usr/bin/env python3
"""Train and gate a dense all-pairs residual tile-neighbour scorer.

This is a bounded research pilot, not a submission builder.  For each sampled
query tile and direction the training objective contains the true successor
and every one of the other 575 slots.  The symmetric incoming objective also
compares the true predecessor with all 575 alternatives, directly penalizing
many-to-one conflicts.  A compact CNN embeds the raw and
denoised tile views once; the learned model then predicts a bounded *cost
residual* over a frozen HBT+classical compatibility matrix.  At initialization
the resulting solver is therefore exactly the frozen baseline.

The global decision is deliberately left to the existing QAP solver.  Row-wise
top-1 scores are useful retrieval diagnostics, but greedy ``pick top-2 after a
conflict`` cannot enforce a consistent 24x24 one-to-one layout.

Recommended Kaggle pilot invocation::

    torchrun --standalone --nproc_per_node=2 \
      scripts/train_evaluate_dense_pair_residual.py \
      --action pilot --output-dir /kaggle/working/dense_pair_residual

Training uses a late HBT-clean ``edge_train`` slice.  Exact selection then uses
``edge_development`` and a source-disjoint ``assembly_cal`` transfer slice.
Only after both pass is an original-input assembly gate opened via an
input-only freeze followed by immutable target attachment.  The true audit and
confirmation slices remain sealed until a later multiseed candidate freeze.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
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
from torch.nn import functional as F

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.geometry import GRID, TILE_COUNT, true_neighbour_slots, validate_permutation
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics, retrieval_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import merge_tiles_numpy, split_tiles_numpy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_HBT = (
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/"
    "hbt_d320_denoised_rgb_sobel.pt"
)
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"
DEFAULT_AUDIT_EXCLUSION = "configs/assembly_audit_exclusion_v1.json"
REPORT_NAME = "dense_pair_residual_report.json"
BEST_CHECKPOINT = "dense_pair_residual_best.pt"
LATEST_CHECKPOINT = "dense_pair_residual_latest.pt"
HASHES_NAME = "SHA256SUMS.txt"
RIGHT = 0
DOWN = 1


@dataclass(frozen=True)
class Runtime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class PreparedSource:
    name: str
    panel: str
    replica: int
    seed: int
    raw: np.ndarray
    denoised: np.ndarray
    clean: np.ndarray
    slot_to_target: np.ndarray
    seed_score: CompatibilityMatrices
    base: CompatibilityMatrices


@dataclass(frozen=True)
class PreparedRealSource:
    """Original competition-style input prepared without reading its target."""

    name: str
    raw: np.ndarray
    denoised: np.ndarray
    seed_score: CompatibilityMatrices
    base: CompatibilityMatrices


def _dense_api() -> Any:
    """Late import is the explicit adapter point to the separately owned model."""

    from puzzle_assembly import dense_pair_residual as api

    required = (
        "DensePairResidualScorer",
        "dense_pair_residual_compatibility",
        "load_dense_pair_residual_checkpoint",
        "load_dense_pair_residual_checkpoint_payload",
        "save_dense_pair_residual_checkpoint",
    )
    missing = [name for name in required if not hasattr(api, name)]
    if missing:
        raise RuntimeError(f"dense-pair model API is incomplete: {missing}")
    return api


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("pilot", "train", "evaluate"), default="pilot")
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--hbt-checkpoint", default=DEFAULT_HBT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--audit-exclusion", default=DEFAULT_AUDIT_EXCLUSION)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260711)

    # Bounded 2xT4 pilot.  Each source contributes one optimizer step per rank.
    parser.add_argument("--train-offset", type=int, default=4096)
    parser.add_argument("--train-sources", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--queries-per-source", type=int, default=48)
    parser.add_argument("--panels", default="primary_kornia,independent_libjpeg")
    parser.add_argument("--selection-offset", type=int, default=96)
    parser.add_argument("--selection-sources", type=int, default=32)
    parser.add_argument("--holdout-offset", type=int, default=112)
    parser.add_argument("--holdout-sources", type=int, default=16)
    parser.add_argument("--real-gate-offset", type=int, default=128)
    parser.add_argument("--real-gate-sources", type=int, default=64)
    parser.add_argument("--final-audit-offset", type=int, default=0)
    parser.add_argument("--final-audit-sources", type=int, default=64)
    parser.add_argument("--confirmation-offset", type=int, default=64)
    parser.add_argument("--confirmation-sources", type=int, default=64)
    parser.add_argument("--evaluation-replicas", type=int, default=1)
    parser.add_argument("--quick-sources", type=int, default=32)

    parser.add_argument("--channels", type=int, default=160)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--pair-hidden-dim", type=int, default=384)
    parser.add_argument("--side-band", type=int, default=6)
    parser.add_argument("--bounded-gain", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--warmup-fraction", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cost-temperature", type=float, default=0.10)
    parser.add_argument("--residual-l2", type=float, default=0.01)
    parser.add_argument("--column-loss-weight", type=float, default=0.50)
    parser.add_argument("--margin-loss-weight", type=float, default=0.15)
    parser.add_argument("--margin-cost", type=float, default=0.03)
    parser.add_argument("--pair-chunk-size", type=int, default=8192)
    parser.add_argument("--dense-chunk-size", type=int, default=64)
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--classical-chunk-size", type=int, default=64)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-amp-skips", type=int, default=3)

    # Mild post-panel augmentation; the exact corruption engines remain the
    # primary domain randomization.  Raw and denoised views are both retained.
    parser.add_argument("--extra-noise-sigma", type=float, default=8.0)
    parser.add_argument("--extra-noise-probability", type=float, default=0.40)
    parser.add_argument("--affine-probability", type=float, default=0.50)
    parser.add_argument("--blur-probability", type=float, default=0.20)
    parser.add_argument("--quantize-probability", type=float, default=0.25)
    parser.add_argument("--view-dropout", type=float, default=0.05)

    parser.add_argument("--qap-iterations", type=int, default=25)
    parser.add_argument("--qap-restarts", type=int, default=2)
    parser.add_argument("--qap-refine-swaps", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-model-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.train_offset = 4096
        args.train_sources = 2
        args.epochs = 1
        args.queries_per_source = 48 if args.full_model_smoke else 4
        args.selection_sources = 1
        args.holdout_sources = 1
        args.real_gate_sources = 1
        args.final_audit_sources = 1
        args.confirmation_sources = 1
        args.quick_sources = 1
        args.evaluation_replicas = 1
        if not args.full_model_smoke:
            args.channels = 32
            args.embedding_dim = 32
            args.pair_hidden_dim = 64
        if not args.full_model_smoke:
            args.pair_chunk_size = 1024
            args.dense_chunk_size = 8
        args.qap_iterations = 1
        args.qap_restarts = 1
        args.qap_refine_swaps = 0
    return args


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _init_runtime(seed: int) -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed dense-pair training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    resolved = seed + 1009 * rank
    random.seed(resolved)
    np.random.seed(resolved % (2**32 - 1))
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    return Runtime(rank, world_size, local_rank, device)


def _barrier(runtime: Runtime) -> None:
    if runtime.world_size > 1:
        dist.barrier()


def _broadcast_model(model: nn.Module, runtime: Runtime) -> None:
    """Make rank zero authoritative even though augmentation RNG is rank-local."""

    if runtime.world_size == 1:
        return
    for tensor in model.state_dict().values():
        dist.broadcast(tensor, src=0)


def _all_gather(value: Any, runtime: Runtime) -> list[Any]:
    if runtime.world_size == 1:
        return [value]
    output: list[Any] = [None] * runtime.world_size
    dist.all_gather_object(output, value)
    return output


def _hardware_probe(runtime: Runtime) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rank": runtime.rank,
        "world_size": runtime.world_size,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(runtime.device),
        "cuda_available": torch.cuda.is_available(),
    }
    if runtime.device.type == "cuda":
        probe = torch.arange(64, device=runtime.device, dtype=torch.float32)
        result.update(
            {
                "gpu": torch.cuda.get_device_name(runtime.device),
                "capability": list(torch.cuda.get_device_capability(runtime.device)),
                "total_memory": int(torch.cuda.get_device_properties(runtime.device).total_memory),
                "tensor_probe": float((probe.square().mean()).item()),
            }
        )
        try:
            result["nvidia_smi"] = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            result["nvidia_smi"] = ["unavailable"]
    return result


def _model(args: argparse.Namespace) -> nn.Module:
    cls = _dense_api().DensePairResidualScorer
    return cls(
        encoder_width=args.channels,
        encoder_depth=8,
        expansion=4,
        embedding_dim=args.embedding_dim,
        relation_hidden=args.pair_hidden_dim,
        pair_hidden=max(args.embedding_dim, args.pair_hidden_dim // 2),
        side_band=args.side_band,
        profile_bins=10,
        max_residual=args.bounded_gain,
        initial_gain_fraction=0.5,
        dropout=args.dropout,
    )


def _bounded_t4_preflight(model: nn.Module, args: argparse.Namespace, runtime: Runtime) -> dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    pairs = 2 * args.queries_per_source * TILE_COUNT
    # Conservative accounting: fp32 weights/grad/master+Adam, cached tile
    # activations, pair MLP activations, input banks, and 2x safety margin.
    parameter_bytes = parameters * 16
    activation_bytes = pairs * max(args.pair_hidden_dim, args.embedding_dim * 5) * 2 * 8
    bank_bytes = TILE_COUNT * 6 * 20 * 20 * 4 * 4
    estimate = 2 * (parameter_bytes + activation_bytes + bank_bytes)
    total = (
        int(torch.cuda.get_device_properties(runtime.device).total_memory)
        if runtime.device.type == "cuda"
        else 0
    )
    if runtime.device.type == "cuda" and estimate > int(total * 0.82):
        raise RuntimeError(
            f"estimated dense-pair peak {estimate / 2**30:.2f} GiB exceeds bounded GPU envelope"
        )
    return {
        "parameters": parameters,
        "all_candidates_per_query": TILE_COUNT - 1,
        "pairs_per_source_step": pairs,
        "estimated_peak_bytes": estimate,
        "device_total_bytes": total,
        "estimated_fraction": float(estimate / total) if total else None,
        "bounded": True,
    }


def _validate_args(args: argparse.Namespace, runtime: Runtime) -> None:
    panels = [value.strip() for value in args.panels.split(",") if value.strip()]
    if args.action == "evaluate" and args.resume_checkpoint:
        raise ValueError("--resume-checkpoint is only valid for pilot/train")
    if args.action != "evaluate" and args.checkpoint:
        raise ValueError("--checkpoint is evaluate-only; use --resume-checkpoint for training")
    if not panels or any(value not in {"primary_kornia", "independent_libjpeg"} for value in panels):
        raise ValueError("panels must contain primary_kornia and/or independent_libjpeg")
    if args.train_offset < 0:
        raise ValueError("train-offset must be non-negative")
    if args.train_sources <= 0 or args.train_sources % runtime.world_size:
        raise ValueError("train-sources must be positive and divisible by world size")
    if args.queries_per_source <= 0 or args.queries_per_source % 2:
        raise ValueError("queries-per-source must be a positive even number")
    if args.cost_temperature <= 0.0:
        raise ValueError("cost-temperature must be positive")
    if min(args.residual_l2, args.column_loss_weight, args.margin_loss_weight, args.margin_cost) < 0.0:
        raise ValueError("loss weights and margin must be non-negative")
    if min(
        args.selection_sources,
        args.holdout_sources,
        args.real_gate_sources,
        args.final_audit_sources,
        args.confirmation_sources,
        args.quick_sources,
    ) <= 0:
        raise ValueError("evaluation source counts must be positive")
    if args.train_offset < 4096:
        raise ValueError("train-offset must avoid frozen/continued HBT edge_train[0:4096]")
    if args.selection_offset < 96:
        raise ValueError("selection-offset must avoid reused edge_development[0:96]")
    if args.holdout_offset < 112:
        raise ValueError("holdout-offset must avoid reused assembly_cal[0:112]")
    if args.real_gate_offset < 128:
        raise ValueError("real-gate-offset must avoid prior assembly_incremental_gate[0:128]")
    if args.final_audit_offset < 0:
        raise ValueError("final-audit-offset must be non-negative")
    if args.confirmation_offset < 64:
        raise ValueError("confirmation-offset must preserve final_audit[0:64]")
    if args.action != "evaluate" and not args.smoke:
        if panels != ["primary_kornia", "independent_libjpeg"]:
            raise RuntimeError("pilot requires the exact two-panel order")
        if args.qap_iterations != 25 or args.qap_restarts != 2:
            raise RuntimeError("pilot requires the frozen 25x2 QAP budget")
    if args.action != "evaluate" and not args.smoke:
        if runtime.device.type != "cuda" or runtime.world_size != 2:
            raise RuntimeError("non-smoke training is precommitted to torchrun on exactly 2 GPUs")
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if len(names) != 2 or not all("T4" in name.upper() for name in names):
            raise RuntimeError(f"pilot requires exactly two Tesla T4 devices, got {names}")


def _rank_cost(matrix: np.ndarray) -> np.ndarray:
    """Convert any finite directional cost to a stable row-rank cost in [0,1]."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError("compatibility matrix must be 576x576")
    order = np.argsort(values, axis=1, kind="stable")
    rank = np.empty_like(order, dtype=np.int32)
    np.put_along_axis(
        rank,
        order,
        np.broadcast_to(np.arange(TILE_COUNT, dtype=np.int32), order.shape),
        axis=1,
    )
    result = rank.astype(np.float32) / float(TILE_COUNT - 1)
    np.fill_diagonal(result, np.inf)
    return result


def _frozen_base(
    raw: np.ndarray,
    denoised: np.ndarray,
    *,
    hbt_model: nn.Module,
    runtime: Runtime,
    chunk_size: int,
) -> tuple[CompatibilityMatrices, CompatibilityMatrices]:
    """Return the promoted HBT seed score and frozen C1+HBTw4 QAP cost."""

    hbt, _ = learned_compatibility(
        hbt_model, denoised, device=runtime.device, name="denoised_hbt"
    )
    # The promoted C1 branch is denoised-only.  Avoid computing an unused raw
    # expert bank for every training source.
    bank = build_classical_score_bank(
        denoised, prefix="denoised", chunk_size=chunk_size
    )
    c1_names = [
        name
        for name in sorted(bank)
        if name.startswith("denoised_") and not name.endswith("_c2")
    ]
    c1 = fuse_ranked_scores(bank, names=c1_names, name="denoised_C1")
    fused = fuse_ranked_scores(
        {c1.name: c1, hbt.name: hbt},
        names=[c1.name, hbt.name],
        weights={hbt.name: 4.0},
        name="frozen_C1_HBTw4",
    )
    # ``fuse_ranked_scores`` is already scale-stable and produces the exact
    # established C1+HBTw4 cost.  Do not rank it a second time: zero residual
    # must reproduce the current solver input bit-for-bit.
    return (
        CompatibilityMatrices(
            hbt.name,
            np.asarray(hbt.right, dtype=np.float32).copy(),
            np.asarray(hbt.down, dtype=np.float32).copy(),
        ),
        CompatibilityMatrices(
            fused.name,
            np.asarray(fused.right, dtype=np.float32).copy(),
            np.asarray(fused.down, dtype=np.float32).copy(),
        ),
    )


def _prepare_source(
    name: str,
    panel_name: str,
    replica: int,
    *,
    stage: str,
    args: argparse.Namespace,
    runtime: Runtime,
    restorer: nn.Module,
    hbt_model: nn.Module,
) -> PreparedSource:
    clean = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
    seed = per_source_seed(args.seed, f"dense-pair-{stage}-{panel_name}", name, replica)
    exact = make_exact_panel(clean, panel=panel_name, seed=seed)
    denoised = restore_tiles_uint8(
        restorer,
        exact.slot_tiles,
        runtime.device,
        batch_size=args.denoise_batch_size,
    )
    seed_score, base = _frozen_base(
        exact.slot_tiles,
        denoised,
        hbt_model=hbt_model,
        runtime=runtime,
        chunk_size=args.classical_chunk_size,
    )
    return PreparedSource(
        name=name,
        panel=panel_name,
        replica=replica,
        seed=seed,
        raw=exact.slot_tiles,
        denoised=denoised,
        clean=clean,
        slot_to_target=exact.slot_to_target,
        seed_score=seed_score,
        base=base,
    )


def _prepare_real_source(
    name: str,
    *,
    args: argparse.Namespace,
    runtime: Runtime,
    restorer: nn.Module,
    hbt_model: nn.Module,
) -> PreparedRealSource:
    """Prepare an original shuffled input without touching ``train/targets``."""

    shuffled = _read_rgb(Path(args.data_root) / "train" / "inputs" / name)
    raw = split_tiles_numpy(shuffled)
    denoised = restore_tiles_uint8(
        restorer,
        raw,
        runtime.device,
        batch_size=args.denoise_batch_size,
    )
    seed_score, base = _frozen_base(
        raw,
        denoised,
        hbt_model=hbt_model,
        runtime=runtime,
        chunk_size=args.classical_chunk_size,
    )
    return PreparedRealSource(
        name=name,
        raw=raw,
        denoised=denoised,
        seed_score=seed_score,
        base=base,
    )


def dense_targets(slot_to_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact successor slot for each query; -1 marks right/bottom boundaries."""

    right, down = true_neighbour_slots(slot_to_target)
    if int(np.sum(right >= 0)) != GRID * (GRID - 1):
        raise RuntimeError("right target count is not 552")
    if int(np.sum(down >= 0)) != GRID * (GRID - 1):
        raise RuntimeError("down target count is not 552")
    return right.astype(np.int32), down.astype(np.int32)


def sample_queries(
    slot_to_target: np.ndarray,
    *,
    total: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced query indices for right/down dense 576-way retrieval rows."""

    if total <= 0 or total % 2:
        raise ValueError("total query count must be positive and even")
    targets = dense_targets(slot_to_target)
    output = []
    for direction in (RIGHT, DOWN):
        eligible = np.flatnonzero(targets[direction] >= 0)
        take = total // 2
        output.append(rng.choice(eligible, size=take, replace=take > len(eligible)).astype(np.int32))
    return output[0], output[1]


def _to_tile_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    ).div_(255.0)


def _augment_views(
    raw: torch.Tensor,
    denoised: torch.Tensor,
    *,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Task-faithful mild augmentation after an exact primary/libjpeg panel."""

    raw = raw.clone()
    denoised = denoised.clone()
    count = len(raw)
    affine_mask = torch.rand((count, 1, 1, 1), device=raw.device, generator=generator) < args.affine_probability
    scale = 0.90 + 0.20 * torch.rand((count, 1, 1, 1), device=raw.device, generator=generator)
    offset = (torch.rand((count, 1, 1, 1), device=raw.device, generator=generator) - 0.5) * (16.0 / 255.0)
    raw = torch.where(affine_mask, (raw - 0.5) * scale + 0.5 + offset, raw)
    denoised = torch.where(affine_mask, (denoised - 0.5) * scale + 0.5 + offset, denoised)

    noise_mask = torch.rand((count, 1, 1, 1), device=raw.device, generator=generator) < args.extra_noise_probability
    noise = torch.randn(raw.shape, device=raw.device, generator=generator) * (args.extra_noise_sigma / 255.0)
    raw = raw + noise_mask * noise
    denoised = denoised + noise_mask * noise * 0.25

    blur_mask = torch.rand((count, 1, 1, 1), device=raw.device, generator=generator) < args.blur_probability
    blurred = F.avg_pool2d(F.pad(raw, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    raw = torch.where(blur_mask, 0.45 * raw + 0.55 * blurred, raw)

    quantize_mask = torch.rand((count, 1, 1, 1), device=raw.device, generator=generator) < args.quantize_probability
    step = (3.0 + 9.0 * torch.rand((count, 1, 1, 1), device=raw.device, generator=generator)) / 255.0
    raw = torch.where(quantize_mask, torch.round(raw / step) * step, raw)

    dropout = torch.rand((count, 1, 1, 1), device=raw.device, generator=generator)
    raw = torch.where(dropout < args.view_dropout, raw.new_tensor(0.5), raw)
    denoised = torch.where(
        (dropout >= args.view_dropout) & (dropout < 2.0 * args.view_dropout),
        denoised.new_tensor(0.5),
        denoised,
    )
    return raw.clamp(0.0, 1.0), denoised.clamp(0.0, 1.0)


def _score_query_rows(
    model: nn.Module,
    bank: Any,
    queries: torch.Tensor,
    direction: int,
    *,
    pair_chunk_size: int,
) -> torch.Tensor:
    count = int(queries.numel())
    first = queries[:, None].expand(count, TILE_COUNT).reshape(-1)
    second = torch.arange(TILE_COUNT, device=queries.device)[None, :].expand(count, -1).reshape(-1)
    directions = torch.full_like(first, direction)
    residuals = []
    for start in range(0, len(first), pair_chunk_size):
        stop = min(start + pair_chunk_size, len(first))
        residuals.append(
            model.forward_from_encoded(
                bank,
                first[start:stop],
                second[start:stop],
                directions[start:stop],
            )
        )
    return torch.cat(residuals).reshape(count, TILE_COUNT)


def _score_incoming_columns(
    model: nn.Module,
    bank: Any,
    seconds: torch.Tensor,
    direction: int,
    *,
    pair_chunk_size: int,
) -> torch.Tensor:
    """Score every possible predecessor for each selected successor tile."""

    count = int(seconds.numel())
    first = torch.arange(TILE_COUNT, device=seconds.device)[None, :].expand(count, -1).reshape(-1)
    second = seconds[:, None].expand(count, TILE_COUNT).reshape(-1)
    directions = torch.full_like(first, direction)
    residuals = []
    for start in range(0, len(first), pair_chunk_size):
        stop = min(start + pair_chunk_size, len(first))
        residuals.append(
            model.forward_from_encoded(
                bank,
                first[start:stop],
                second[start:stop],
                directions[start:stop],
            )
        )
    return torch.cat(residuals).reshape(count, TILE_COUNT)


def _hard_negative_margin(
    final_cost: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    positive = final_cost.gather(1, target[:, None]).squeeze(1)
    negatives = final_cost.clone()
    negatives.scatter_(1, target[:, None], torch.inf)
    hardest = negatives.min(dim=1).values
    return F.relu(positive + margin - hardest).mean()


def _sync_gradients(model: nn.Module, runtime: Runtime) -> None:
    if runtime.world_size == 1:
        return
    parameters = list(model.parameters())
    local_present = torch.as_tensor(
        [parameter.grad is not None for parameter in parameters],
        device=runtime.device,
        dtype=torch.int32,
    )
    minimum = local_present.clone()
    maximum = local_present.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not torch.equal(minimum, maximum):
        raise RuntimeError("gradient presence differs across ranks")
    active = [parameter for parameter, present in zip(parameters, minimum.tolist(), strict=True) if present]
    if not active:
        return
    sizes = [parameter.grad.numel() for parameter in active]
    flat = torch.cat([parameter.grad.reshape(-1) for parameter in active])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(runtime.world_size)
    offset = 0
    for parameter, size in zip(active, sizes, strict=True):
        parameter.grad.copy_(flat[offset : offset + size].view_as(parameter.grad))
        offset += size


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, int(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _train_source_step(
    model: nn.Module,
    source: PreparedSource,
    *,
    args: argparse.Namespace,
    runtime: Runtime,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
    rng: np.random.Generator,
) -> dict[str, float]:
    raw = _to_tile_tensor(source.raw, runtime.device)
    denoised = _to_tile_tensor(source.denoised, runtime.device)
    raw, denoised = _augment_views(raw, denoised, args=args, generator=generator)
    sampled = sample_queries(source.slot_to_target, total=args.queries_per_source, rng=rng)
    exact = dense_targets(source.slot_to_target)
    amp = runtime.device.type == "cuda" and not args.no_amp
    optimizer.zero_grad(set_to_none=True)
    residual_parts: list[torch.Tensor] = []
    row_losses: list[torch.Tensor] = []
    column_losses: list[torch.Tensor] = []
    margin_losses: list[torch.Tensor] = []
    hits: list[torch.Tensor] = []
    incoming_hits: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.autocast(device_type=runtime.device.type, dtype=torch.float16, enabled=amp):
        bank = model.encode_tiles(raw, denoised)
        for direction in (RIGHT, DOWN):
            query = torch.as_tensor(sampled[direction], device=runtime.device, dtype=torch.long)
            residual = _score_query_rows(
                model,
                bank,
                query,
                direction,
                pair_chunk_size=args.pair_chunk_size,
            )
            base_matrix = source.base.right if direction == RIGHT else source.base.down
            base = torch.from_numpy(base_matrix[sampled[direction]].copy()).to(
                runtime.device, dtype=torch.float32
            )
            # Self pairs are impossible.  Keep a finite large cost for CE.
            base = torch.where(torch.isfinite(base), base, base.new_tensor(1.0e4))
            final_cost = model.apply_residual(base, residual.float())
            logits = -final_cost / args.cost_temperature
            target = torch.as_tensor(
                exact[direction][sampled[direction]], device=runtime.device, dtype=torch.long
            )
            row_losses.append(F.cross_entropy(logits, target))
            margin_losses.append(
                _hard_negative_margin(final_cost, target, margin=args.margin_cost)
            )
            residual_parts.append(residual.float())
            hits.append((logits.argmax(dim=1) == target).float().mean())

            incoming_residual = _score_incoming_columns(
                model,
                bank,
                target,
                direction,
                pair_chunk_size=args.pair_chunk_size,
            )
            incoming_base = torch.from_numpy(
                base_matrix[:, exact[direction][sampled[direction]]].T.copy()
            ).to(runtime.device, dtype=torch.float32)
            incoming_base = torch.where(
                torch.isfinite(incoming_base), incoming_base, incoming_base.new_tensor(1.0e4)
            )
            incoming_cost = model.apply_residual(incoming_base, incoming_residual.float())
            incoming_logits = -incoming_cost / args.cost_temperature
            column_losses.append(F.cross_entropy(incoming_logits, query))
            margin_losses.append(
                _hard_negative_margin(incoming_cost, query, margin=args.margin_cost)
            )
            residual_parts.append(incoming_residual.float())
            incoming_hits.append((incoming_logits.argmax(dim=1) == query).float().mean())
        loss = torch.stack(row_losses).mean()
        loss = loss + args.column_loss_weight * torch.stack(column_losses).mean()
        loss = loss + args.margin_loss_weight * torch.stack(margin_losses).mean()
        residual_l2 = torch.cat([part.reshape(-1) for part in residual_parts]).square().mean()
        loss = loss + args.residual_l2 * residual_l2
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    _sync_gradients(model, runtime)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    finite = torch.tensor(
        int(bool(torch.isfinite(grad_norm).item())), device=runtime.device, dtype=torch.int32
    )
    if runtime.world_size > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    skipped = not bool(finite.item())
    if skipped:
        optimizer.zero_grad(set_to_none=True)
        scaler.update(max(float(scaler.get_scale()) / 2.0, 1.0))
    else:
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    return {
        "loss": float(loss.detach()),
        "residual_l2": float(residual_l2.detach()),
        "sampled_recall_at_1": float(torch.stack(hits).mean()),
        "sampled_incoming_recall_at_1": float(torch.stack(incoming_hits).mean()),
        "grad_norm": float(grad_norm.detach()) if torch.isfinite(grad_norm) else float("inf"),
        "skipped": float(skipped),
        "pairs": float(2 * args.queries_per_source * (TILE_COUNT - 1)),
        "seconds": float(time.perf_counter() - started),
    }


def top1_conflict_metrics(score: CompatibilityMatrices, slot_to_target: np.ndarray) -> dict[str, float]:
    """Target-free row-top1 collision and directed two-cycle diagnostics."""

    truths = dense_targets(slot_to_target)
    records = []
    for direction, matrix in ((RIGHT, score.right), (DOWN, score.down)):
        valid = np.flatnonzero(truths[direction] >= 0)
        choice = np.argmin(matrix, axis=1)
        selected = choice[valid]
        collision_excess = len(selected) - len(np.unique(selected))
        cycles = sum(int(choice[int(choice[q])] == q) for q in valid if int(choice[q]) != q) / 2.0
        records.append((collision_excess / len(valid), cycles / len(valid)))
    return {
        "top1_collision_excess_rate": float(np.mean([value[0] for value in records])),
        "directed_two_cycle_rate": float(np.mean([value[1] for value in records])),
    }


@torch.inference_mode()
def _evaluate_retrieval_source(
    model: nn.Module,
    source: PreparedSource,
    *,
    args: argparse.Namespace,
    runtime: Runtime,
) -> tuple[dict[str, Any], CompatibilityMatrices]:
    api = _dense_api()
    neural = api.dense_pair_residual_compatibility(
        model,
        source.raw,
        source.base,
        denoised_tiles=source.denoised,
        device=runtime.device,
        chunk_size=args.dense_chunk_size,
        name="dense_pair_residual_C1_HBTw4",
    )
    base_metrics = retrieval_metrics(source.base, source.slot_to_target)["combined"]
    neural_metrics = retrieval_metrics(neural, source.slot_to_target)["combined"]
    base_conflict = top1_conflict_metrics(source.base, source.slot_to_target)
    neural_conflict = top1_conflict_metrics(neural, source.slot_to_target)
    record = {
        "name": source.name,
        "panel": source.panel,
        "replica": source.replica,
        "seed": source.seed,
        "base": base_metrics,
        "neural": neural_metrics,
        "delta": {
            "recall_at_1": neural_metrics["recall_at_1"] - base_metrics["recall_at_1"],
            "recall_at_5": neural_metrics["recall_at_5"] - base_metrics["recall_at_5"],
            "recall_at_32": neural_metrics["recall_at_32"] - base_metrics["recall_at_32"],
            "mrr": neural_metrics["mrr"] - base_metrics["mrr"],
            "top1_collision_excess_rate": (
                neural_conflict["top1_collision_excess_rate"]
                - base_conflict["top1_collision_excess_rate"]
            ),
            "directed_two_cycle_rate": (
                neural_conflict["directed_two_cycle_rate"]
                - base_conflict["directed_two_cycle_rate"]
            ),
        },
        "base_conflicts": base_conflict,
        "neural_conflicts": neural_conflict,
    }
    return record, neural


def _component_seed(score: CompatibilityMatrices) -> np.ndarray:
    return soft_cycle_component_solver(
        score,
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        loop_weight=1.0,
        reciprocal_weight=0.35,
    ).position_to_slot


def _qap_layout(
    score: CompatibilityMatrices,
    *,
    initial: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    return directional_qap(
        score,
        initial=initial,
        iterations=args.qap_iterations,
        restarts=args.qap_restarts,
        seed=seed,
        boundary_weight=0.05,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=args.qap_refine_swaps,
        refine_weak_cells=32,
    ).position_to_slot


def _evaluate_qap_source(
    source: PreparedSource,
    neural: CompatibilityMatrices,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    # Reproduce the promoted solver exactly: soft-cycle is seeded from frozen
    # HBT alone, while both candidates optimize their C1+HBTw4-style QAP cost
    # from the same fixed layout.  The filename seed is also identical to the
    # submission builder, so the learned residual cannot win through a changed
    # initializer or RNG draw.
    initial = _component_seed(source.seed_score)
    filename_seed = int.from_bytes(
        hashlib.sha256(source.name.encode("utf-8")).digest()[:4], "little"
    )
    qap_seed = filename_seed + 7001
    base_layout = _qap_layout(source.base, initial=initial, seed=qap_seed, args=args)
    neural_layout = _qap_layout(neural, initial=initial, seed=qap_seed, args=args)
    base_layout_metrics = layout_metrics(base_layout, source.slot_to_target)
    neural_layout_metrics = layout_metrics(neural_layout, source.slot_to_target)
    base_image = predicted_image_metrics(base_layout, source.denoised, source.clean)
    neural_image = predicted_image_metrics(neural_layout, source.denoised, source.clean)
    return {
        "name": source.name,
        "panel": source.panel,
        "replica": source.replica,
        "seed": source.seed,
        "qap_seed": qap_seed,
        "initial_layout_sha256": hashlib.sha256(
            np.asarray(initial, dtype=np.int32).tobytes()
        ).hexdigest(),
        "base": {"layout": base_layout_metrics, "image": base_image},
        "neural": {"layout": neural_layout_metrics, "image": neural_image},
        "delta": {
            "ssim": (
                neural_image["predicted_layout_ssim"] - base_image["predicted_layout_ssim"]
            ),
            "adjacency": (
                neural_layout_metrics["combined_adjacency"]
                - base_layout_metrics["combined_adjacency"]
            ),
            "position_accuracy": (
                neural_layout_metrics["position_accuracy"]
                - base_layout_metrics["position_accuracy"]
            ),
        },
    }


def _atomic_png(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.shape != (480, 480, 3) or values.dtype != np.uint8:
        raise ValueError("frozen render must be uint8 RGB 480x480")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.png")
    try:
        Image.fromarray(values, mode="RGB").save(
            temporary, format="PNG", compress_level=6
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_rgb_hashed(path: Path, expected_sha256: str) -> np.ndarray:
    """Hash and decode the exact same immutable byte string."""

    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"frozen PNG hash mismatch: {path}")
    with Image.open(BytesIO(payload)) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected frozen image shape for {path}: {values.shape}")
    return values


def _frozen_image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    if prediction.shape != (480, 480, 3) or target.shape != prediction.shape:
        raise ValueError("real-gate images must be RGB 480x480")
    return {
        "ssim": float(
            structural_similarity(target, prediction, channel_axis=2, data_range=255)
        ),
        "psnr": float(peak_signal_noise_ratio(target, prediction, data_range=255)),
        "mae": float(
            np.mean(
                np.abs(prediction.astype(np.float32) - target.astype(np.float32))
            )
        ),
    }


def _real_gate_aggregate(records: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    deltas = [float(record["delta_ssim"]) for record in records]
    interval = bootstrap_mean_ci(deltas, seed=seed)
    return {
        "source_count": len(records),
        "bootstrap_unit": "whole_original_source",
        "mean_base_ssim": float(np.mean([record["base"]["ssim"] for record in records])),
        "mean_candidate_ssim": float(
            np.mean([record["candidate"]["ssim"] for record in records])
        ),
        "mean_delta_ssim": float(np.mean(deltas)),
        "median_delta_ssim": float(np.median(deltas)),
        "win_rate": float(np.mean(np.asarray(deltas) > 0.0)),
        "bootstrap_95_delta_ssim": list(interval),
    }


def real_input_gate(aggregate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "mean_real_ssim_delta_ge_0.005": aggregate["mean_delta_ssim"] >= 0.005,
        "bootstrap_real_ssim_lower_gt_0": aggregate["bootstrap_95_delta_ssim"][0] > 0.0,
        "real_ssim_win_rate_ge_0.60": aggregate["win_rate"] >= 0.60,
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def _require_exact_frozen_envelope(
    loaded: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    if loaded != expected:
        raise RuntimeError("real-gate phase-A manifest changed before target access")
    payload = loaded.get("payload")
    digest = loaded.get("payload_sha256")
    if not isinstance(payload, dict) or digest != _canonical_json_sha256(payload):
        raise RuntimeError("real-gate phase-A manifest envelope hash is invalid")
    return payload


@torch.inference_mode()
def _evaluate_real_input_gate(
    model: nn.Module,
    names: list[str],
    *,
    split_label: str,
    output_dir: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
    runtime: Runtime,
    restorer: nn.Module,
    hbt_model: nn.Module,
) -> dict[str, Any]:
    """Freeze input-only renders, then attach targets in a separate phase."""

    checkpoint_path = checkpoint_path.resolve()
    checkpoint_sha256 = _sha256(checkpoint_path)
    frozen_dir = (output_dir / "frozen_real_predictions" / split_label).resolve()
    if runtime.primary:
        if frozen_dir.exists() and any(frozen_dir.iterdir()):
            raise FileExistsError(f"real-gate freeze directory is non-empty: {frozen_dir}")
        frozen_dir.mkdir(parents=True, exist_ok=True)
    _barrier(runtime)

    local_manifest: list[dict[str, Any]] = []
    for name in names[runtime.rank :: runtime.world_size]:
        source = _prepare_real_source(
            name,
            args=args,
            runtime=runtime,
            restorer=restorer,
            hbt_model=hbt_model,
        )
        candidate = _dense_api().dense_pair_residual_compatibility(
            model,
            source.raw,
            source.base,
            denoised_tiles=source.denoised,
            device=runtime.device,
            chunk_size=args.dense_chunk_size,
            name="dense_pair_residual_real_input",
        )
        initial = _component_seed(source.seed_score)
        filename_seed = int.from_bytes(
            hashlib.sha256(name.encode("utf-8")).digest()[:4], "little"
        )
        qap_seed = filename_seed + 7001
        base_layout = _qap_layout(
            source.base, initial=initial, seed=qap_seed, args=args
        )
        candidate_layout = _qap_layout(
            candidate, initial=initial, seed=qap_seed, args=args
        )
        base_render = merge_tiles_numpy(source.denoised[base_layout])
        candidate_render = merge_tiles_numpy(source.denoised[candidate_layout])
        base_path = frozen_dir / f"{Path(name).stem}.base.png"
        candidate_path = frozen_dir / f"{Path(name).stem}.candidate.png"
        _atomic_png(base_path, base_render)
        _atomic_png(candidate_path, candidate_render)
        local_manifest.append(
            {
                "name": name,
                "input_pixel_sha256": hashlib.sha256(
                    source.raw.tobytes()
                ).hexdigest(),
                "base_layout": base_layout.tolist(),
                "base_layout_sha256": hashlib.sha256(
                    np.asarray(base_layout, dtype=np.int32).tobytes()
                ).hexdigest(),
                "candidate_layout": candidate_layout.tolist(),
                "candidate_layout_sha256": hashlib.sha256(
                    np.asarray(candidate_layout, dtype=np.int32).tobytes()
                ).hexdigest(),
                "base_render": str(base_path),
                "base_render_sha256": _sha256(base_path),
                "candidate_render": str(candidate_path),
                "candidate_render_sha256": _sha256(candidate_path),
                "qap_seed": qap_seed,
            }
        )

    gathered = [
        item for rank_items in _all_gather(local_manifest, runtime) for item in rank_items
    ]
    if len(gathered) != len(names) or {item["name"] for item in gathered} != set(names):
        raise RuntimeError("real-gate input-only phase lost or duplicated sources")
    order = {name: index for index, name in enumerate(names)}
    gathered.sort(key=lambda item: order[str(item["name"])])
    if [str(record["name"]) for record in gathered] != names:
        raise RuntimeError("real-gate manifest order differs from frozen name order")
    for record in gathered:
        name = str(record["name"])
        stem = Path(name).stem
        for label, expected_name in (
            ("base_render", f"{stem}.base.png"),
            ("candidate_render", f"{stem}.candidate.png"),
        ):
            path = Path(str(record[label])).resolve()
            if path.parent != frozen_dir or path.name != expected_name:
                raise RuntimeError(f"frozen render escaped its directory: {path}")
        for label in ("base", "candidate"):
            layout = validate_permutation(
                np.asarray(record[f"{label}_layout"], dtype=np.int32),
                name=f"{label}_layout",
            )
            digest = hashlib.sha256(layout.tobytes()).hexdigest()
            if digest != record[f"{label}_layout_sha256"]:
                raise RuntimeError(f"{label} layout hash mismatch before freeze: {name}")
    manifest_path = frozen_dir / "FROZEN_INPUT_ONLY_MANIFEST.json"
    manifest_payload = {
        "schema_version": 1,
        "kind": "dense_pair_input_only_frozen_predictions",
        "split": split_label,
        "source_names": names,
        "source_names_sha256": _names_sha256(names),
        "candidate_checkpoint_path": str(checkpoint_path),
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "target_files_opened": False,
        "records": gathered,
    }
    expected_envelope = {
        "payload": manifest_payload,
        "payload_sha256": _canonical_json_sha256(manifest_payload),
    }
    if runtime.primary:
        _atomic_json(manifest_path, expected_envelope)
    _barrier(runtime)

    loaded_envelope = _load_json(manifest_path)
    frozen_manifest = _require_exact_frozen_envelope(
        loaded_envelope, expected_envelope
    )
    if frozen_manifest.get("target_files_opened") is not False:
        raise RuntimeError("real-gate phase-A manifest is not target-sealed")
    if frozen_manifest.get("candidate_checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("real-gate checkpoint changed after input-only freeze")
    if frozen_manifest.get("source_names_sha256") != _names_sha256(names):
        raise RuntimeError("real-gate name list changed after input-only freeze")
    frozen_records = frozen_manifest.get("records")
    if not isinstance(frozen_records, list) or len(frozen_records) != len(names):
        raise RuntimeError("real-gate frozen manifest is incomplete")
    by_name = {str(record["name"]): record for record in frozen_records}
    if set(by_name) != set(names) or len(by_name) != len(names):
        raise RuntimeError("real-gate manifest names are not unique and exact")
    anchored_manifest_sha256 = _sha256(manifest_path)
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("candidate checkpoint changed before target access")

    target_event_path = frozen_dir / "TARGET_ACCESS_STARTED.json"
    expected_target_event = {
        "schema_version": 1,
        "kind": "dense_pair_target_access_event",
        "split": split_label,
        "phase_a_manifest_sha256": anchored_manifest_sha256,
        "phase_a_payload_sha256": expected_envelope["payload_sha256"],
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "source_names_sha256": _names_sha256(names),
        "target_access_started": True,
        "target_files_may_have_been_opened": True,
    }
    if runtime.primary:
        _atomic_json(target_event_path, expected_target_event)
    _barrier(runtime)
    if _load_json(target_event_path) != expected_target_event:
        raise RuntimeError("durable target-access event is missing or changed")

    # Phase B starts here.  No target path is constructed before every render,
    # layout, name list, and checkpoint hash above has been frozen and verified.
    local_scores: list[dict[str, Any]] = []
    for name in names[runtime.rank :: runtime.world_size]:
        frozen = by_name[name]
        base_path = Path(str(frozen["base_render"]))
        candidate_path = Path(str(frozen["candidate_render"]))
        base_render = _read_rgb_hashed(base_path, frozen["base_render_sha256"])
        candidate_render = _read_rgb_hashed(
            candidate_path, frozen["candidate_render_sha256"]
        )
        target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
        base_metrics = _frozen_image_metrics(base_render, target)
        candidate_metrics = _frozen_image_metrics(candidate_render, target)
        local_scores.append(
            {
                "name": name,
                "base": base_metrics,
                "candidate": candidate_metrics,
                "delta_ssim": candidate_metrics["ssim"] - base_metrics["ssim"],
                "base_layout_sha256": frozen["base_layout_sha256"],
                "candidate_layout_sha256": frozen["candidate_layout_sha256"],
            }
        )
    records = [
        item for rank_items in _all_gather(local_scores, runtime) for item in rank_items
    ]
    if (
        len(records) != len(names)
        or len({str(record["name"]) for record in records}) != len(names)
        or {str(record["name"]) for record in records} != set(names)
    ):
        raise RuntimeError("real-gate target attach lost or duplicated sources")
    records.sort(key=lambda item: order[str(item["name"])])
    if _sha256(manifest_path) != anchored_manifest_sha256:
        raise RuntimeError("phase-A manifest changed during target attachment")
    if _load_json(target_event_path) != expected_target_event:
        raise RuntimeError("target-access event changed during scoring")
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("candidate checkpoint changed during real-gate scoring")
    aggregate = _real_gate_aggregate(records, seed=args.seed + 503)
    return {
        "split": split_label,
        "source_names": names,
        "source_names_sha256": _names_sha256(names),
        "phase_a_manifest": str(manifest_path),
        "phase_a_manifest_sha256": anchored_manifest_sha256,
        "phase_a_payload_sha256": expected_envelope["payload_sha256"],
        "target_access_event": str(target_event_path),
        "target_access_event_sha256": _sha256(target_event_path),
        "target_opened_after_predictions_frozen": True,
        "records": records,
        "aggregate": aggregate,
        "gate": real_input_gate(aggregate),
    }


def bootstrap_mean_ci(values: list[float], *, seed: int, samples: int = 4000) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _aggregate(records: list[dict[str, Any]], *, kind: str, seed: int) -> dict[str, Any]:
    metrics = (
        ("recall_at_1", "recall_at_5", "recall_at_32", "mrr", "top1_collision_excess_rate", "directed_two_cycle_rate")
        if kind == "retrieval"
        else ("ssim", "adjacency", "position_accuracy")
    )
    source_names = sorted({str(record["name"]) for record in records})
    aggregate: dict[str, Any] = {
        "count": len(records),
        "source_count": len(source_names),
        "bootstrap_unit": "whole_source_mean_across_panels_and_replicas",
        "panels": {},
    }
    for metric in metrics:
        values = [float(record["delta"][metric]) for record in records]
        source_values = [
            float(
                np.mean(
                    [
                        record["delta"][metric]
                        for record in records
                        if str(record["name"]) == name
                    ]
                )
            )
            for name in source_names
        ]
        aggregate[f"mean_delta_{metric}"] = float(np.mean(values))
        aggregate[f"bootstrap_95_delta_{metric}"] = list(
            bootstrap_mean_ci(source_values, seed=seed + len(metric))
        )
    for panel in sorted({str(record["panel"]) for record in records}):
        selected = [record for record in records if record["panel"] == panel]
        aggregate["panels"][panel] = {
            f"mean_delta_{metric}": float(np.mean([record["delta"][metric] for record in selected]))
            for metric in metrics
        }
    return aggregate


def retrieval_gate(aggregate: dict[str, Any]) -> dict[str, Any]:
    panel_positive = all(
        values["mean_delta_recall_at_1"] > 0.0
        for values in aggregate["panels"].values()
    )
    checks = {
        "mean_recall_at_1_delta_ge_0.01": aggregate["mean_delta_recall_at_1"] >= 0.01,
        "mean_mrr_delta_ge_0.01": aggregate["mean_delta_mrr"] >= 0.01,
        "mean_recall_at_32_delta_ge_minus_0.005": aggregate["mean_delta_recall_at_32"] >= -0.005,
        "bootstrap_recall_at_1_lower_gt_0": aggregate["bootstrap_95_delta_recall_at_1"][0] > 0.0,
        "every_panel_recall_at_1_positive": panel_positive,
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def qap_gate(aggregate: dict[str, Any]) -> dict[str, Any]:
    panel_positive = all(
        values["mean_delta_ssim"] > 0.0 for values in aggregate["panels"].values()
    )
    checks = {
        "mean_qap_ssim_delta_ge_0.005": aggregate["mean_delta_ssim"] >= 0.005,
        "mean_qap_adjacency_delta_ge_0.01": aggregate["mean_delta_adjacency"] >= 0.01,
        "bootstrap_qap_ssim_lower_gt_0": aggregate["bootstrap_95_delta_ssim"][0] > 0.0,
        "every_panel_qap_ssim_positive": panel_positive,
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def _evaluate_split(
    model: nn.Module,
    names: list[str],
    *,
    split_label: str,
    run_qap: bool,
    args: argparse.Namespace,
    runtime: Runtime,
    restorer: nn.Module,
    hbt_model: nn.Module,
) -> dict[str, Any]:
    panels = [value.strip() for value in args.panels.split(",") if value.strip()]
    tasks = [
        (name, panel, replica)
        for name in names
        for panel in panels
        for replica in range(args.evaluation_replicas)
    ]
    retrieval_records: list[dict[str, Any]] = []
    qap_records: list[dict[str, Any]] = []
    for name, panel, replica in tasks[runtime.rank :: runtime.world_size]:
        prepared = _prepare_source(
            name,
            panel,
            replica,
            stage=split_label,
            args=args,
            runtime=runtime,
            restorer=restorer,
            hbt_model=hbt_model,
        )
        retrieval, neural = _evaluate_retrieval_source(
            model, prepared, args=args, runtime=runtime
        )
        retrieval_records.append(retrieval)
        if run_qap:
            qap_records.append(_evaluate_qap_source(prepared, neural, args=args))
    retrieval_records = [
        item for rank_items in _all_gather(retrieval_records, runtime) for item in rank_items
    ]
    qap_records = [item for rank_items in _all_gather(qap_records, runtime) for item in rank_items]
    if len(retrieval_records) != len(tasks):
        raise RuntimeError("distributed retrieval evaluation lost or duplicated tasks")
    if run_qap and len(qap_records) != len(tasks):
        raise RuntimeError("distributed QAP evaluation lost or duplicated tasks")
    retrieval_aggregate = _aggregate(
        retrieval_records, kind="retrieval", seed=args.seed + 301
    )
    result: dict[str, Any] = {
        "split": split_label,
        "names": names,
        "names_sha256": _names_sha256(names),
        "retrieval": {"records": retrieval_records, "aggregate": retrieval_aggregate},
        "retrieval_gate": retrieval_gate(retrieval_aggregate),
        "synthetic_target_files_opened": True,
        "qap_metrics_computed": bool(run_qap),
    }
    if run_qap:
        qap_aggregate = _aggregate(qap_records, kind="qap", seed=args.seed + 401)
        result["qap"] = {"records": qap_records, "aggregate": qap_aggregate}
        result["qap_gate"] = qap_gate(qap_aggregate)
    return result


def _quick_metrics(
    model: nn.Module,
    names: list[str],
    *,
    args: argparse.Namespace,
    runtime: Runtime,
    restorer: nn.Module,
    hbt_model: nn.Module,
) -> dict[str, Any]:
    panels = [value.strip() for value in args.panels.split(",") if value.strip()]
    tasks = [(name, panel) for name in names for panel in panels]
    records: list[dict[str, Any]] = []
    for name, panel in tasks[runtime.rank :: runtime.world_size]:
        source = _prepare_source(
            name,
            panel,
            0,
            stage="quick-selection",
            args=args,
            runtime=runtime,
            restorer=restorer,
            hbt_model=hbt_model,
        )
        record, _ = _evaluate_retrieval_source(model, source, args=args, runtime=runtime)
        records.append(record)
    merged = [value for rank_values in _all_gather(records, runtime) for value in rank_values]
    if len(merged) != len(tasks):
        raise RuntimeError("quick validation distributed count mismatch")
    return _aggregate(merged, kind="retrieval", seed=args.seed + 211)


def _train(
    model: nn.Module,
    train_names: list[str],
    quick_names: list[str],
    *,
    args: argparse.Namespace,
    runtime: Runtime,
    restorer: nn.Module,
    hbt_model: nn.Module,
    output_dir: Path,
    provenance: dict[str, Any],
    resume_payload: dict[str, Any] | None = None,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    api = _dense_api()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    local_sources = len(train_names) // runtime.world_size
    scheduler = _scheduler(
        optimizer,
        total_steps=args.epochs * local_sources,
        warmup_fraction=args.warmup_fraction,
    )
    amp = runtime.device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0, growth_interval=1000)
    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(args.seed + 65537 * runtime.rank)
    panels = [value.strip() for value in args.panels.split(",") if value.strip()]
    # Epoch zero is a real candidate: the zero-initialized residual reproduces
    # the frozen baseline exactly and must beat any degraded learned epoch.
    best_score = 0.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    best_path = output_dir / BEST_CHECKPOINT
    latest_path = output_dir / LATEST_CHECKPOINT
    total_skips = 0
    start_epoch = 0
    early_stop_reason: str | None = None
    if resume_payload is not None:
        training_state = resume_payload.get("training_state")
        optimizer_state = resume_payload.get("optimizer_state")
        if not isinstance(training_state, dict) or not isinstance(optimizer_state, dict):
            raise RuntimeError("resume checkpoint lacks exact optimizer/training state")
        if training_state.get("capture_point") != "epoch_boundary":
            raise RuntimeError("resume checkpoint is not at an epoch boundary")
        if training_state.get("terminal") is True or training_state.get(
            "early_stop_reason"
        ):
            raise RuntimeError("resume checkpoint is terminal and must not continue")
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(training_state["scheduler_state"])
        scaler.load_state_dict(training_state["scaler_state"])
        start_epoch = int(training_state["next_epoch_index"])
        total_skips = int(training_state["amp_skips"])
        history = list(training_state["history"])
        best_score = float(training_state["best_score"])
        best_epoch = int(training_state["best_epoch"])
        rng_states = training_state.get("rng_states")
        if not isinstance(rng_states, list) or len(rng_states) != runtime.world_size:
            raise RuntimeError("resume checkpoint lacks per-rank RNG states")
        rank_rng = rng_states[runtime.rank]
        if runtime.primary:
            best_state = training_state.get("best_model_state")
            if not isinstance(best_state, dict):
                raise RuntimeError("resume checkpoint lacks best-model state")
            best_model = api.DensePairResidualScorer(**model.config())
            best_model.load_state_dict(best_state, strict=True)
            api.save_dense_pair_residual_checkpoint(
                best_path,
                best_model,
                metadata={
                    **provenance,
                    "training_history": history,
                    "best_epoch": best_epoch,
                    "selection_metric": "quick two-panel delta recall@1 + delta MRR over frozen C1+HBTw4",
                    "resumed_from_epoch_boundary": start_epoch,
                    "safe_for_submission": False,
                },
            )
        # Constructing the temporary best-model copy consumes CPU RNG.  Restore
        # every per-rank stream only after that bookkeeping so resumed dropout
        # and augmentation are bit-for-bit aligned with uninterrupted training.
        generator.set_state(rank_rng["augmentation_generator"])
        random.setstate(rank_rng["python_random"])
        np.random.set_state(rank_rng["numpy_random"])
        torch.set_rng_state(rank_rng["torch_cpu"])
        if runtime.device.type == "cuda":
            torch.cuda.set_rng_state(rank_rng["torch_cuda"], runtime.device)
        if start_epoch >= args.epochs:
            raise RuntimeError("resume checkpoint already completed requested epochs")
    elif runtime.primary:
        api.save_dense_pair_residual_checkpoint(
            best_path,
            model,
            metadata={
                **provenance,
                "training_history": [],
                "best_epoch": 0,
                "selection_metric": "quick two-panel delta recall@1 + delta MRR over frozen C1+HBTw4",
                "epoch0_contract": "exact zero residual; frozen base retained bitwise off diagonal",
                "safe_for_submission": False,
            },
        )
    _barrier(runtime)
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    for epoch in range(start_epoch, args.epochs):
        order = list(train_names)
        np.random.default_rng(args.seed + epoch * 104729).shuffle(order)
        rank_names = order[runtime.rank :: runtime.world_size]
        records = []
        started = time.perf_counter()
        model.train()
        for source_index, name in enumerate(rank_names):
            panel = panels[(source_index + epoch + runtime.rank) % len(panels)]
            source = _prepare_source(
                name,
                panel,
                epoch,
                stage="train",
                args=args,
                runtime=runtime,
                restorer=restorer,
                hbt_model=hbt_model,
            )
            record = _train_source_step(
                model,
                source,
                args=args,
                runtime=runtime,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                generator=generator,
                rng=np.random.default_rng(source.seed),
            )
            records.append(record)
            total_skips += int(record["skipped"])
            if total_skips > args.max_amp_skips:
                raise RuntimeError("bounded AMP recovery budget exhausted; fail closed")
            if runtime.primary and (source_index + 1) % 16 == 0:
                print(
                    json.dumps(
                        {
                            "event": "dense_pair_train_progress",
                            "epoch": epoch + 1,
                            "source": source_index + 1,
                            "per_rank": len(rank_names),
                            "loss": record["loss"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        rank_summary = {
            key: float(np.mean([record[key] for record in records]))
            for key in (
                "loss",
                "residual_l2",
                "sampled_recall_at_1",
                "sampled_incoming_recall_at_1",
                "pairs",
                "seconds",
            )
        }
        summaries = _all_gather(rank_summary, runtime)
        summary = {
            key: float(np.mean([record[key] for record in summaries])) for key in rank_summary
        }
        model.eval()
        quick = _quick_metrics(
            model,
            quick_names,
            args=args,
            runtime=runtime,
            restorer=restorer,
            hbt_model=hbt_model,
        )
        record = {
            "epoch": epoch + 1,
            "train": summary,
            "quick_selection": quick,
            "seconds": time.perf_counter() - started,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "amp_skips": total_skips,
        }
        history.append(record)
        quick_score = float(
            quick["mean_delta_recall_at_1"] + quick["mean_delta_mrr"]
        )
        if (
            epoch == 0
            and args.epochs > 1
            and float(quick["mean_delta_recall_at_1"]) <= 0.0
            and float(quick["mean_delta_mrr"]) <= 0.0
        ):
            early_stop_reason = "epoch1_recall_at_1_and_mrr_nonpositive"
        local_rng_state = {
            "augmentation_generator": generator.get_state().cpu(),
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_cuda": (
                torch.cuda.get_rng_state(runtime.device).cpu()
                if runtime.device.type == "cuda"
                else None
            ),
        }
        rng_states = _all_gather(local_rng_state, runtime)
        if runtime.primary:
            metadata = {
                **provenance,
                "training_history": history,
                "safe_for_submission": False,
            }
            if quick_score > best_score:
                best_score = quick_score
                best_epoch = epoch + 1
                api.save_dense_pair_residual_checkpoint(
                    best_path,
                    model,
                    metadata={
                        **metadata,
                        "best_epoch": best_epoch,
                        "selection_metric": "quick two-panel delta recall@1 + delta MRR over frozen C1+HBTw4",
                    },
                )
            best_payload = api.load_dense_pair_residual_checkpoint_payload(best_path)
            api.save_dense_pair_residual_checkpoint(
                latest_path,
                model,
                metadata={**metadata, "latest_completed_epoch": epoch + 1},
                optimizer_state=optimizer.state_dict(),
                training_state={
                    "capture_point": "epoch_boundary",
                    "completed_epoch_index": epoch,
                    "next_epoch_index": epoch + 1,
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "amp_skips": total_skips,
                    "history": history,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "best_model_state": best_payload["model_state"],
                    "rng_states": rng_states,
                    "early_stop_reason": early_stop_reason,
                    "terminal": bool(
                        early_stop_reason is not None or epoch + 1 >= args.epochs
                    ),
                },
            )
            print(json.dumps({"event": "dense_pair_epoch", **record}, sort_keys=True), flush=True)
        _barrier(runtime)
        if early_stop_reason is not None:
            if runtime.primary:
                print(
                    json.dumps(
                        {
                            "event": "dense_pair_early_stop",
                            "reason": early_stop_reason,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            break
    telemetry = {
        "history": history,
        "amp_skips": total_skips,
        "early_stop_reason": early_stop_reason,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": _sha256(best_path) if best_path.is_file() else None,
        "latest_checkpoint": str(latest_path),
        "latest_checkpoint_sha256": _sha256(latest_path) if latest_path.is_file() else None,
        "peak_cuda_by_rank": _all_gather(
            {
                "rank": runtime.rank,
                "allocated": int(torch.cuda.max_memory_allocated(runtime.device))
                if runtime.device.type == "cuda"
                else 0,
                "reserved": int(torch.cuda.max_memory_reserved(runtime.device))
                if runtime.device.type == "cuda"
                else 0,
            },
            runtime,
        ),
    }
    return best_path, history, telemetry


def _scientific_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Canonical knobs that must not change between training and evaluation."""

    payload = {
        "seed": int(args.seed),
        "panels": [value.strip() for value in args.panels.split(",") if value.strip()],
        "evaluation_replicas": int(args.evaluation_replicas),
        "model": {
            "channels": int(args.channels),
            "embedding_dim": int(args.embedding_dim),
            "pair_hidden_dim": int(args.pair_hidden_dim),
            "side_band": int(args.side_band),
            "bounded_gain": float(args.bounded_gain),
            "dropout": float(args.dropout),
        },
        "training": {
            "epochs": int(args.epochs),
            "queries_per_source": int(args.queries_per_source),
            "quick_sources": int(args.quick_sources),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_fraction": float(args.warmup_fraction),
            "grad_clip": float(args.grad_clip),
            "cost_temperature": float(args.cost_temperature),
            "residual_l2": float(args.residual_l2),
            "column_loss_weight": float(args.column_loss_weight),
            "margin_loss_weight": float(args.margin_loss_weight),
            "margin_cost": float(args.margin_cost),
            "no_amp": bool(args.no_amp),
            "max_amp_skips": int(args.max_amp_skips),
            "pair_chunk_size": int(args.pair_chunk_size),
            "dense_chunk_size": int(args.dense_chunk_size),
            "denoise_batch_size": int(args.denoise_batch_size),
            "classical_chunk_size": int(args.classical_chunk_size),
        },
        "corruption": {
            "extra_noise_sigma": float(args.extra_noise_sigma),
            "extra_noise_probability": float(args.extra_noise_probability),
            "affine_probability": float(args.affine_probability),
            "blur_probability": float(args.blur_probability),
            "quantize_probability": float(args.quantize_probability),
            "view_dropout": float(args.view_dropout),
        },
        "qap": {
            "iterations": int(args.qap_iterations),
            "restarts": int(args.qap_restarts),
            "refine_swaps": int(args.qap_refine_swaps),
            "boundary_weight": 0.05,
            "initial_weight": 0.75,
            "noisy_components": 3,
            "noise_scale": 1.0,
            "seed_formula": "sha256(filename)[:4]_little + 7001",
            "initial_layout": "soft_cycle_frozen_HBT_top8_keep1_fraction0.5_loop1_reciprocal0.35",
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "config": payload,
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def _provenance(
    args: argparse.Namespace,
    *,
    train_names: list[str],
    selection_names: list[str],
    holdout_names: list[str],
    real_gate_names: list[str],
    final_audit_names: list[str],
    confirmation_names: list[str],
    denoiser_metadata: dict[str, Any],
    hbt_metadata: dict[str, Any],
    hardware: list[dict[str, Any]],
) -> dict[str, Any]:
    named_partitions = {
        "train": set(train_names),
        "selection": set(selection_names),
        "holdout": set(holdout_names),
        "real_gate": set(real_gate_names),
        "final_audit": set(final_audit_names),
        "confirmation": set(confirmation_names),
    }
    labels = list(named_partitions)
    for index, first in enumerate(labels):
        for second in labels[index + 1 :]:
            if named_partitions[first] & named_partitions[second]:
                raise RuntimeError(f"source partitions overlap: {first} and {second}")
    hbt_train = set(map(str, hbt_metadata.get("train_names", [])))
    evaluation = set().union(
        named_partitions["selection"],
        named_partitions["holdout"],
        named_partitions["real_gate"],
        named_partitions["final_audit"],
        named_partitions["confirmation"],
    )
    if hbt_train & evaluation:
        raise RuntimeError("evaluation overlaps frozen HBT training sources")
    hbt_validation = set(
        map(
            str,
            hbt_metadata.get(
                "val_names", hbt_metadata.get("validation_names", [])
            ),
        )
    )
    if hbt_validation & evaluation:
        raise RuntimeError("evaluation overlaps frozen HBT validation sources")
    manifest = _load_json(args.manifest)
    manifest_train = set(map(str, manifest.get("splits", {}).get("train", [])))
    promotion_names = named_partitions["holdout"] | named_partitions["real_gate"] | named_partitions["final_audit"] | named_partitions["confirmation"]
    if promotion_names & manifest_train:
        raise RuntimeError("promotion gate overlaps denoiser training sources")
    audit_exclusion = _load_json(args.audit_exclusion)
    exposed_audit = set(map(str, audit_exclusion.get("excluded_names", [])))
    if len(exposed_audit) != 32:
        raise RuntimeError("audit exclusion ledger must contain exactly 32 names")
    if exposed_audit & (
        named_partitions["final_audit"] | named_partitions["confirmation"]
    ):
        raise RuntimeError("final audit overlaps the known-exposure ledger")
    return {
        "kind": "dense_all_pairs_residual_pilot",
        "safe_for_submission": False,
        "seed": args.seed,
        "scientific_protocol": _scientific_protocol(args),
        "all_negatives_contract": "each sampled outgoing row and incoming column scores all 576 slots; self is masked, hence 575 valid alternatives in both orientations",
        "base_contract": "exact frozen fuse_ranked_scores denoised C1 + HBT weight 4; learned output is bounded cost residual",
        "train_partition": f"edge_train[{args.train_offset}:{args.train_offset + len(train_names)}]",
        "train_names": train_names,
        "train_names_sha256": _names_sha256(train_names),
        "selection_partition": f"edge_development[{args.selection_offset}:{args.selection_offset + len(selection_names)}]",
        "selection_names": selection_names,
        "selection_names_sha256": _names_sha256(selection_names),
        "quick_selection_names": selection_names[: min(args.quick_sources, len(selection_names))],
        "quick_selection_names_sha256": _names_sha256(
            selection_names[: min(args.quick_sources, len(selection_names))]
        ),
        "holdout_partition": f"assembly_cal[{args.holdout_offset}:{args.holdout_offset + len(holdout_names)}]",
        "holdout_names": holdout_names,
        "holdout_names_sha256": _names_sha256(holdout_names),
        "real_gate_partition": f"assembly_incremental_gate[{args.real_gate_offset}:{args.real_gate_offset + len(real_gate_names)}]",
        "real_gate_names": real_gate_names,
        "real_gate_names_sha256": _names_sha256(real_gate_names),
        "final_audit_partition": f"assembly_final_audit[{args.final_audit_offset}:{args.final_audit_offset + len(final_audit_names)}]",
        "final_audit_names": final_audit_names,
        "final_audit_names_sha256": _names_sha256(final_audit_names),
        "confirmation_partition": f"assembly_final_audit[{args.confirmation_offset}:{args.confirmation_offset + len(confirmation_names)}]",
        "confirmation_names": confirmation_names,
        "confirmation_names_sha256": _names_sha256(confirmation_names),
        "exposure_scope": {
            "selection": "manifest-train source unseen by HBT checkpoint but seen by TileNAF; scorer development only",
            "holdout": "manifest-val TileNAF calibration-exposed; synthetic transfer only",
            "real_gate": "original real input; assembly-fresh but TileNAF one-shot gate target was previously opened",
            "final_and_confirmation": "audit sources excluded from all known target runs and unopened by this pilot",
        },
        "known_reused_prefixes_avoided": {
            "edge_development_0_96": True,
            "assembly_cal_0_112": True,
            "assembly_incremental_gate_0_128": True,
            "random_audit_exclusion_ledger_applied": True,
        },
        "manifest": {"path": args.manifest, "sha256": _sha256(args.manifest)},
        "quarantine": {"path": args.quarantine, "sha256": _sha256(args.quarantine)},
        "audit_exclusion": {
            "path": args.audit_exclusion,
            "sha256": _sha256(args.audit_exclusion),
            "excluded_names_sha256": _names_sha256(sorted(exposed_audit)),
        },
        "denoiser": denoiser_metadata,
        "hbt": {**hbt_metadata, "checkpoint": args.hbt_checkpoint, "checkpoint_sha256": _sha256(args.hbt_checkpoint)},
        "hardware": hardware,
        "gate_contract": {
            "selection_order": [
                "cheap_synthetic_selection_retrieval",
                "synthetic_transfer_holdout_retrieval_QAP",
                "frozen_original_real_input_QAP_SSIM",
            ],
            "pilot_stop": "true audit remains sealed; passing candidate must be retrained with the formal multiseed protocol before audit opens",
            "retrieval": {
                "mean_recall_at_1_delta": 0.01,
                "mean_mrr_delta": 0.01,
                "recall_at_32_floor": -0.005,
                "bootstrap_lower": 0.0,
                "every_panel_positive": True,
            },
            "qap": {
                "mean_ssim_delta": 0.005,
                "mean_adjacency_delta": 0.01,
                "bootstrap_lower": 0.0,
                "every_panel_positive": True,
            },
        },
    }


def _validate_checkpoint_provenance(
    checkpoint_metadata: dict[str, Any], active: dict[str, Any]
) -> None:
    """Reject evaluation under a different frozen base or source protocol."""

    exact_keys = (
        "kind",
        "base_contract",
        "train_names_sha256",
        "selection_names_sha256",
        "quick_selection_names_sha256",
        "holdout_names_sha256",
        "real_gate_names_sha256",
        "final_audit_names_sha256",
        "confirmation_names_sha256",
    )
    mismatched = [
        key for key in exact_keys if checkpoint_metadata.get(key) != active.get(key)
    ]
    for nested in ("manifest", "quarantine", "audit_exclusion"):
        if checkpoint_metadata.get(nested, {}).get("sha256") != active.get(nested, {}).get("sha256"):
            mismatched.append(f"{nested}.sha256")
    if checkpoint_metadata.get("scientific_protocol", {}).get("sha256") != active.get(
        "scientific_protocol", {}
    ).get("sha256"):
        mismatched.append("scientific_protocol.sha256")
    if checkpoint_metadata.get("hbt", {}).get("checkpoint_sha256") != active.get("hbt", {}).get(
        "checkpoint_sha256"
    ):
        mismatched.append("hbt.checkpoint_sha256")
    if checkpoint_metadata.get("denoiser", {}).get("checkpoint_sha256") != active.get(
        "denoiser", {}
    ).get("checkpoint_sha256"):
        mismatched.append("denoiser.checkpoint_sha256")
    if mismatched:
        raise RuntimeError(f"checkpoint provenance differs from active protocol: {mismatched}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runtime = _init_runtime(args.seed)
    try:
        _validate_args(args, runtime)
        output_dir = Path(args.output_dir)
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"non-empty output directory: {output_dir}")
        if runtime.primary:
            output_dir.mkdir(parents=True, exist_ok=True)
        _barrier(runtime)

        edge_train = source_names_for_split(
            "edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine
        )
        edge_development = source_names_for_split(
            "edge_development",
            manifest_path=args.manifest,
            quarantine_path=args.quarantine,
        )
        assembly_cal = source_names_for_split(
            "assembly_cal",
            manifest_path=args.manifest,
            quarantine_path=args.quarantine,
        )
        incremental = source_names_for_split(
            "assembly_incremental_gate",
            manifest_path=args.manifest,
            quarantine_path=args.quarantine,
        )
        final_audit_pool = source_names_for_split(
            "assembly_final_audit",
            manifest_path=args.manifest,
            quarantine_path=args.quarantine,
            audit_exclusion_path=args.audit_exclusion,
        )
        train_names = edge_train[
            args.train_offset : args.train_offset + args.train_sources
        ]
        selection_names = edge_development[
            args.selection_offset : args.selection_offset + args.selection_sources
        ]
        holdout_names = assembly_cal[
            args.holdout_offset : args.holdout_offset + args.holdout_sources
        ]
        real_gate_names = incremental[
            args.real_gate_offset : args.real_gate_offset + args.real_gate_sources
        ]
        final_audit_names = final_audit_pool[
            args.final_audit_offset : args.final_audit_offset + args.final_audit_sources
        ]
        confirmation_names = final_audit_pool[
            args.confirmation_offset : args.confirmation_offset + args.confirmation_sources
        ]
        requested = {
            "train": (train_names, args.train_sources),
            "selection": (selection_names, args.selection_sources),
            "holdout": (holdout_names, args.holdout_sources),
            "real_gate": (real_gate_names, args.real_gate_sources),
            "final_audit": (final_audit_names, args.final_audit_sources),
            "confirmation": (confirmation_names, args.confirmation_sources),
        }
        incomplete = [
            label for label, (names, count) in requested.items() if len(names) != count
        ]
        if incomplete:
            raise ValueError(f"requested slices exceed authoritative splits: {incomplete}")
        quick_names = selection_names[: min(args.quick_sources, len(selection_names))]

        restorer, _, denoiser_metadata = load_restorer(
            args.denoiser, device=str(runtime.device), state="ema"
        )
        hbt_model, hbt_metadata = load_embedding_checkpoint(
            args.hbt_checkpoint, device=runtime.device
        )
        hbt_model.eval().requires_grad_(False)
        restorer.eval().requires_grad_(False)
        hardware = _all_gather(_hardware_probe(runtime), runtime)
        provenance = _provenance(
            args,
            train_names=train_names,
            selection_names=selection_names,
            holdout_names=holdout_names,
            real_gate_names=real_gate_names,
            final_audit_names=final_audit_names,
            confirmation_names=confirmation_names,
            denoiser_metadata=denoiser_metadata,
            hbt_metadata=hbt_metadata,
            hardware=hardware,
        )

        api = _dense_api()
        training: dict[str, Any] | None = None
        if args.action in {"pilot", "train"}:
            resume_payload = None
            if args.resume_checkpoint:
                resume_payload = api.load_dense_pair_residual_checkpoint_payload(
                    args.resume_checkpoint
                )
                resume_metadata = resume_payload["metadata"]
                if resume_metadata.get("safe_for_submission") is not False:
                    raise RuntimeError("resume checkpoint is not fail-closed")
                _validate_checkpoint_provenance(resume_metadata, provenance)
                expected_config = _model(args).config()
                if resume_payload["model_config"] != expected_config:
                    raise RuntimeError("resume checkpoint model config differs from active pilot")
                model = api.DensePairResidualScorer(**resume_payload["model_config"])
                model.load_state_dict(resume_payload["model_state"], strict=True)
                model.to(runtime.device)
            else:
                model = _model(args).to(runtime.device)
            _broadcast_model(model, runtime)
            preflight = _bounded_t4_preflight(model, args, runtime)
            best_path, history, telemetry = _train(
                model,
                train_names,
                quick_names,
                args=args,
                runtime=runtime,
                restorer=restorer,
                hbt_model=hbt_model,
                output_dir=output_dir,
                provenance=provenance,
                resume_payload=resume_payload,
            )
            if runtime.device.type == "cuda":
                unsafe = [
                    record
                    for record in telemetry["peak_cuda_by_rank"]
                    if int(record["reserved"]) > int(preflight["device_total_bytes"] * 0.90)
                ]
                if unsafe:
                    raise RuntimeError(
                        f"measured CUDA reserve exceeded 90% safety envelope: {unsafe}"
                    )
            training = {"history": history, "telemetry": telemetry, "preflight": preflight}
            _barrier(runtime)
            model, checkpoint_metadata = api.load_dense_pair_residual_checkpoint(
                best_path, device=runtime.device
            )
            candidate_checkpoint_path = best_path
        else:
            if not args.checkpoint:
                raise ValueError("--checkpoint is required for evaluate")
            model, checkpoint_metadata = api.load_dense_pair_residual_checkpoint(
                args.checkpoint, device=runtime.device
            )
            candidate_checkpoint_path = Path(args.checkpoint)
            preflight = _bounded_t4_preflight(model, args, runtime)
        model.to(runtime.device).eval()
        if checkpoint_metadata.get("safe_for_submission") is not False:
            raise RuntimeError("research checkpoint must be explicitly safe_for_submission=false")
        _validate_checkpoint_provenance(checkpoint_metadata, provenance)
        candidate_checkpoint_sha256 = _sha256(candidate_checkpoint_path)

        selection = None
        holdout = None
        real_gate = None
        if args.action == "train":
            status = "trained_not_gated"
        else:
            # Stage 1: exact synthetic retrieval on a scorer-clean development
            # slice.  No QAP or real target is opened if this signal is absent.
            selection = _evaluate_split(
                model,
                selection_names,
                split_label="cheap_selection_edge_development",
                run_qap=False,
                args=args,
                runtime=runtime,
                restorer=restorer,
                hbt_model=hbt_model,
            )
            if not selection["retrieval_gate"]["passed"]:
                status = "stop_cheap_selection_retrieval"
            else:
                # Stage 2: source-disjoint synthetic transfer.  QAP is opened
                # only after retrieval passes on this exact same fixed slice.
                holdout = _evaluate_split(
                    model,
                    holdout_names,
                    split_label="synthetic_transfer_assembly_cal",
                    run_qap=False,
                    args=args,
                    runtime=runtime,
                    restorer=restorer,
                    hbt_model=hbt_model,
                )

                if holdout["retrieval_gate"]["passed"]:
                    holdout = _evaluate_split(
                        model,
                        holdout_names,
                        split_label="synthetic_transfer_assembly_cal",
                        run_qap=True,
                        args=args,
                        runtime=runtime,
                        restorer=restorer,
                        hbt_model=hbt_model,
                    )
                if not holdout["retrieval_gate"]["passed"]:
                    status = "stop_synthetic_transfer_retrieval"
                elif not holdout.get("qap_gate", {}).get("passed", False):
                    status = "stop_synthetic_transfer_qap"
                else:
                    # Stage 3: original shuffled/corrupted inputs.  Phase A
                    # freezes layouts/renders without target access; Phase B
                    # verifies hashes before attaching targets for real SSIM.
                    real_gate = _evaluate_real_input_gate(
                        model,
                        real_gate_names,
                        split_label="frozen_original_real_input_gate",
                        output_dir=output_dir,
                        checkpoint_path=candidate_checkpoint_path,
                        args=args,
                        runtime=runtime,
                        restorer=restorer,
                        hbt_model=hbt_model,
                    )
                    status = (
                        "continue_candidate_only"
                        if real_gate["gate"]["passed"]
                        else "stop_original_real_input_gate"
                    )
        report = {
            "schema_version": 1,
            "kind": "dense_all_pairs_residual_pilot_report",
            "status": status,
            "safe_for_submission": False,
            "provenance": provenance,
            "model_config": model.config(),
            "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
            "checkpoint_metadata": checkpoint_metadata,
            "training": training,
            "selection": selection,
            "holdout": holdout,
            "real_gate": real_gate,
            "gate_opened": {
                "synthetic_transfer": holdout is not None,
                "original_real_input": real_gate is not None,
                "true_final_audit": False,
                "true_confirmation": False,
            },
            "audit_policy": "assembly_final_audit[0:128] (audit minus the random 32-name exposure ledger) remains sealed until a candidate passes this pilot and the formal multiseed freeze",
            "preflight": preflight,
        }
        if runtime.primary:
            report_path = output_dir / REPORT_NAME
            _atomic_json(report_path, report)
            artifacts = [report_path]
            for name in (BEST_CHECKPOINT, LATEST_CHECKPOINT):
                path = output_dir / name
                if path.is_file():
                    artifacts.append(path)
            hashes_path = output_dir / HASHES_NAME
            hashes_path.write_text(
                "".join(f"{_sha256(path)}  {path.name}\n" for path in artifacts),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "event": "dense_pair_complete",
                        "status": status,
                        "report": str(report_path),
                        "report_sha256": _sha256(report_path),
                        "safe_for_submission": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        _barrier(runtime)
    finally:
        if runtime.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
