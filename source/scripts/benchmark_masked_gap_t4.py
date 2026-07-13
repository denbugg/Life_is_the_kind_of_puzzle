#!/usr/bin/env python3
"""Target-free DDP T4x2 AMP capacity benchmark for the masked-gap CNN pipeline.

This program intentionally has no dataset, checkpoint, image, target, or label
input.  It benchmarks real convolutional forward/backward work on deterministic
finite synthetic 20x40 tensors, projects the frozen end-to-end neural workload,
and writes a fail-closed capacity selection report.  It never trains a model for
use and every report is marked ``safe_for_submission: false``.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import traceback
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


REPORT_KIND = "masked_gap_t4x2_amp_ddp_capacity_selection_v2"
SEED = 20260713
GRID = 24
TILE = 20
PAIR_WIDTH = 40
GAP_START = 18
GAP_STOP = 22
TRAIN_SOURCES = 96
PANELS = 2
EPOCHS = 2
ADJACENCY_GROUPS = 2 * GRID * (GRID - 1)  # 1,104 right/down true pairs.
GROUP_CANDIDATES = 32
GROUP_VIEWS = 2  # outgoing and incoming.
FINAL_SOURCES = 8
DEVELOPMENT_SOURCES = 4
FINAL_DIRECTIONS = 2
FINAL_TILES = GRID * GRID
FINAL_NON_SELF_CANDIDATES = FINAL_TILES - 1
SAFETY_FACTOR = 1.35
MAX_PROJECTED_SECONDS = 5.5 * 60.0 * 60.0
MAX_PEAK_BYTES_PER_GPU = 13_500_000_000  # 13.5 decimal GB, conservatively.
SOURCE_PREPARATION_RESERVE_SECONDS = 60 * 60

GENERATOR_BATCH = 128
RANKER_GROUP_BATCH = 4
DENSE_PAIR_BATCH = 512
WARMUP_STEPS = 5
GENERATOR_REPEATS = 20
RANKER_REPEATS = 20
DENSE_REPEATS = 40
CAPACITY_PROCESS_TIMEOUT_SECONDS = 15 * 60


@dataclass(frozen=True)
class Capacity:
    width: int
    generator_blocks: int
    ranker_blocks: int

    @property
    def key(self) -> str:
        return f"w{self.width}_g{self.generator_blocks}_r{self.ranker_blocks}"


class NoFeasibleCapacity(RuntimeError):
    """Raised after an atomic, evidence-bearing no-capacity report is sealed."""


# Ordered largest to smallest before any data or benchmark result is opened.
CAPACITIES = (
    Capacity(64, 6, 5),
    Capacity(48, 4, 4),
    Capacity(32, 3, 3),
    Capacity(24, 2, 2),
    Capacity(16, 2, 2),
)


def frozen_workload() -> dict[str, int]:
    """Return exact precommitted example counts used for time projection."""

    generator_train = (
        TRAIN_SOURCES * ADJACENCY_GROUPS * PANELS * EPOCHS
    )
    ranker_train_per_arm = (
        TRAIN_SOURCES
        * ADJACENCY_GROUPS
        * GROUP_VIEWS
        * GROUP_CANDIDATES
        * PANELS
        * EPOCHS
    )
    final_per_model = (
        FINAL_SOURCES
        * PANELS
        * FINAL_DIRECTIONS
        * FINAL_TILES
        * FINAL_NON_SELF_CANDIDATES
    )
    development_pass_per_model = (
        DEVELOPMENT_SOURCES
        * PANELS
        * FINAL_DIRECTIONS
        * FINAL_TILES
        * FINAL_NON_SELF_CANDIDATES
    )
    checkpoint_selection_per_model = development_pass_per_model * EPOCHS
    calibration_b_per_model = development_pass_per_model
    all_dense_per_model = (
        checkpoint_selection_per_model + calibration_b_per_model + final_per_model
    )
    generator_source_panel_preparations = TRAIN_SOURCES * PANELS * EPOCHS
    ranker_source_panel_preparations = TRAIN_SOURCES * PANELS * EPOCHS
    checkpoint_selection_source_panel_preparations = (
        DEVELOPMENT_SOURCES * PANELS * EPOCHS
    )
    calibration_b_source_panel_preparations = DEVELOPMENT_SOURCES * PANELS
    final_source_panel_preparations = FINAL_SOURCES * PANELS
    all_source_panel_preparations = (
        generator_source_panel_preparations
        + ranker_source_panel_preparations
        + checkpoint_selection_source_panel_preparations
        + calibration_b_source_panel_preparations
        + final_source_panel_preparations
    )
    all_w4_source_panel_preparations = (
        ranker_source_panel_preparations
        + checkpoint_selection_source_panel_preparations
        + calibration_b_source_panel_preparations
        + final_source_panel_preparations
    )
    return {
        "generator_train_true_pairs": generator_train,
        "ranker_train_pair_candidates_per_arm": ranker_train_per_arm,
        "ranker_train_pair_candidates_two_arms": 2 * ranker_train_per_arm,
        "development_dense_pairs_per_model_per_pass": development_pass_per_model,
        "checkpoint_selection_dense_pairs_per_model_two_epochs": checkpoint_selection_per_model,
        "calibration_b_dense_pairs_per_model": calibration_b_per_model,
        "final_dense_pairs_per_model": final_per_model,
        "all_dense_pairs_per_model": all_dense_per_model,
        "all_dense_component_forwards_generator_plus_two_rankers": 3 * all_dense_per_model,
        "generator_source_panel_preparations_tilenaf": generator_source_panel_preparations,
        "ranker_source_panel_preparations_tilenaf_plus_w4": ranker_source_panel_preparations,
        "checkpoint_selection_source_panel_preparations_tilenaf_plus_w4": checkpoint_selection_source_panel_preparations,
        "calibration_b_source_panel_preparations_tilenaf_plus_w4": calibration_b_source_panel_preparations,
        "final_source_panel_preparations_tilenaf_plus_w4": final_source_panel_preparations,
        "all_source_panel_preparations_tilenaf": all_source_panel_preparations,
        "all_source_panel_preparations_w4": all_w4_source_panel_preparations,
    }


def frozen_contract() -> dict[str, Any]:
    return {
        "capacities_largest_first": [asdict(value) for value in CAPACITIES],
        "workload": frozen_workload(),
        "workload_formula": {
            "generator_train_true_pairs": "96*1104*2_panels*2_epochs",
            "ranker_train_pair_candidates_per_arm": (
                "96*1104_groups*2_outgoing_incoming*32_candidates*2_panels*2_epochs"
            ),
            "ranker_arms": 2,
            "development_dense_pairs_per_model_per_pass": (
                "4*2_panels*2_directions*576*575"
            ),
            "checkpoint_selection_dense_pairs_per_model": (
                "development_pass*2_ranker_epochs"
            ),
            "calibration_b_dense_pairs_per_model": "development_pass*1",
            "final_dense_pairs_per_model": "8*2_panels*2_directions*576*575",
            "final_models": "generator+inpaint_ranker+direct_ranker",
            "all_dense_pairs_per_model": (
                "checkpoint_selection_two_epochs+calibration_b+final_holdout"
            ),
            "all_source_panel_preparations_tilenaf": (
                "96*2_panels*2_epochs_generator + 96*2_panels*2_epochs_ranker + "
                "4*2_panels*2_checkpoint_epochs + 4*2_calibration_b + 8*2_final"
            ),
            "all_source_panel_preparations_w4": (
                "ranker_train_preparations + checkpoint_selection + calibration_b + final"
            ),
        },
        "batches": {
            "ddp_generator_per_gpu": GENERATOR_BATCH,
            "ddp_ranker_groups_per_gpu": RANKER_GROUP_BATCH,
            "ddp_ranker_pairs_per_arm_per_gpu": RANKER_GROUP_BATCH * GROUP_VIEWS * GROUP_CANDIDATES,
            "ddp_dense_pairs_per_gpu": DENSE_PAIR_BATCH,
        },
        "optimizers": {
            "generator": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
            "inpaint_ranker": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
            "direct_ranker": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
            "ranker_grad_scaler": "one shared CUDA GradScaler",
            "ranker_microsteps": (
                "view0 no_sync backward plus view1 synchronized backward"
            ),
            "ranker_update": (
                "separate unscale, max_norm=1.0 clip, and step per ranker"
            ),
        },
        "timing": {
            "warmup_steps": WARMUP_STEPS,
            "generator_repeats": GENERATOR_REPEATS,
            "ranker_repeats": RANKER_REPEATS,
            "dense_repeats": DENSE_REPEATS,
            "cuda_synchronize": True,
            "two_processes_one_per_gpu": True,
            "fresh_process_pair_per_capacity": True,
            "measured_ddp_all_reduce_during_training": True,
            "ddp_gradient_buckets_in_peak_memory": True,
            "effective_2gpu_rate": "2*minimum(per_rank_rate); never sum unequal rates",
            "final_decision": "largest DDP-feasible capacity in precommitted order",
            "data_parallel_route": "not executed by protocol v2",
            "amp_dtype": "float16",
        },
        "selection": {
            "safety_factor": SAFETY_FACTOR,
            "max_projected_seconds": MAX_PROJECTED_SECONDS,
            "max_projected_hours": MAX_PROJECTED_SECONDS / 3600.0,
            "max_peak_bytes_per_gpu": MAX_PEAK_BYTES_PER_GPU,
            "memory_unit": "decimal GB",
            "fixed_source_preparation_reserve_seconds_before_safety": SOURCE_PREPARATION_RESERVE_SECONDS,
            "source_preparation_reserve_scope": (
                "TileNAF over 808 source-panels and w4 over 424; fixed one-hour reserve "
                "is added before the global 1.35 safety factor"
            ),
            "rule": "largest precommitted capacity meeting both inclusive thresholds",
        },
    }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _positive_rate(measurement: dict[str, Any], name: str) -> float:
    value = float(measurement[name])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def inclusive_feasibility(
    projected_seconds_with_safety: float, peak_reserved_bytes: int
) -> tuple[bool, bool, bool]:
    """Apply the precommitted inclusive time and memory boundaries."""

    seconds = float(projected_seconds_with_safety)
    peak = int(peak_reserved_bytes)
    if not math.isfinite(seconds) or seconds < 0.0 or peak < 0:
        raise ValueError("projection and peak memory must be finite/non-negative")
    time_ok = seconds <= MAX_PROJECTED_SECONDS
    memory_ok = peak <= MAX_PEAK_BYTES_PER_GPU
    return time_ok, memory_ok, bool(time_ok and memory_ok)


def project_candidate(measurement: dict[str, Any]) -> dict[str, Any]:
    """Project one capacity from concurrent two-GPU measured throughputs."""

    status = measurement.get("status")
    if status == "oom":
        return {
            **measurement,
            "projection_components_seconds_before_safety": None,
            "projected_seconds_before_safety": None,
            "projected_seconds_with_1p35_safety": None,
            "projected_hours_with_1p35_safety": None,
            "max_peak_reserved_bytes": None,
            "time_threshold_inclusive_pass": False,
            "memory_threshold_inclusive_pass": False,
            "feasible": False,
            "rejection_reason": "isolated_capacity_out_of_memory",
        }
    if status != "complete":
        raise ValueError(f"unexpected capacity measurement status: {status!r}")
    workload = frozen_workload()
    rates = measurement["throughput_2gpu"]
    generator_rate = _positive_rate(rates, "generator_train_pairs_per_second")
    ranker_rate = _positive_rate(
        rates, "joint_ranker_train_pairs_per_arm_per_second"
    )
    dense_rate = _positive_rate(
        rates, "dense_pipeline_pairs_per_second"
    )
    peaks = [int(value) for value in measurement["peak_reserved_bytes_per_gpu"]]
    if len(peaks) != 2 or any(value < 0 for value in peaks):
        raise ValueError("exactly two non-negative per-GPU peak values are required")

    components = {
        "generator_train_seconds": (
            workload["generator_train_true_pairs"] / generator_rate
        ),
        # Both arms execute inside every timed ranker step.  Therefore the
        # per-arm example count is divided by the joint-two-arm throughput.
        "two_rankers_train_seconds": (
            workload["ranker_train_pair_candidates_per_arm"] / ranker_rate
        ),
        # Every timed dense pair runs the frozen generator and both rankers.
        # This covers checkpoint selection on development A after both epochs,
        # the sealed development-B pass, and the final eight-source gate.
        "all_dense_generator_plus_two_rankers_seconds": (
            workload["all_dense_pairs_per_model"] / dense_rate
        ),
        "fixed_source_preparation_reserve_seconds": float(
            SOURCE_PREPARATION_RESERVE_SECONDS
        ),
    }
    raw_seconds = float(sum(components.values()))
    projected_seconds = raw_seconds * SAFETY_FACTOR
    max_peak = max(peaks)
    time_ok, memory_ok, feasible = inclusive_feasibility(
        projected_seconds, max_peak
    )
    return {
        **measurement,
        "projection_components_seconds_before_safety": components,
        "projected_seconds_before_safety": raw_seconds,
        "projected_seconds_with_1p35_safety": projected_seconds,
        "projected_hours_with_1p35_safety": projected_seconds / 3600.0,
        "max_peak_reserved_bytes": max_peak,
        "time_threshold_inclusive_pass": time_ok,
        "memory_threshold_inclusive_pass": memory_ok,
        "feasible": feasible,
    }


def select_largest_feasible(
    measurements: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Project all fixed capacities and select the first feasible one."""

    expected = [capacity.key for capacity in CAPACITIES]
    actual = [str(value.get("capacity_key")) for value in measurements]
    if actual != expected:
        raise ValueError(f"capacity order/config drift: expected {expected}, got {actual}")
    projected = [project_candidate(value) for value in measurements]
    selected = next((value for value in projected if value["feasible"]), None)
    return selected, projected


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.silu(values + self.block(values))


class BenchmarkGenerator(nn.Module):
    """Exact compute shape of the masked-gap 7ch -> 3x20x4 generator."""

    def __init__(self, width: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(7, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.body = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.head = nn.Conv2d(width, 3, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        full = torch.sigmoid(self.head(self.body(self.stem(values))))
        return full[..., GAP_START:GAP_STOP]


class BenchmarkRanker(nn.Module):
    """Exact compute shape of either equal-capacity 10ch pair scorer arm."""

    def __init__(self, width: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(10, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.body = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.body(self.stem(values)))).squeeze(1)


def _ranker_view_loss(logits: torch.Tensor) -> torch.Tensor:
    shaped = logits.reshape(RANKER_GROUP_BATCH, GROUP_CANDIDATES)
    target = torch.zeros(RANKER_GROUP_BATCH, dtype=torch.long, device=logits.device)
    labels = torch.zeros_like(shaped)
    labels[:, 0] = 1.0
    # Two calls (outgoing and incoming) sum to CE_out + CE_in +
    # 0.25 * 0.5 * (BCE_out + BCE_in), exactly the scientific objective.
    return F.cross_entropy(shaped, target) + 0.125 * F.binary_cross_entropy_with_logits(
        shaped, labels
    )


def _make_scaler() -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _timed_stage(
    operation: Callable[[], None],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, dict[str, int]]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        operation()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    dist.barrier()
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("invalid CUDA timing")
    return elapsed, {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _capacity_benchmark(
    capacity: Capacity, rank: int, device: torch.device
) -> dict[str, Any]:
    torch.manual_seed(SEED + rank * 1009 + capacity.width)
    torch.cuda.manual_seed(SEED + rank * 1009 + capacity.width)
    generator = BenchmarkGenerator(capacity.width, capacity.generator_blocks).to(device)
    inpaint_ranker = BenchmarkRanker(capacity.width, capacity.ranker_blocks).to(device)
    direct_ranker = BenchmarkRanker(capacity.width, capacity.ranker_blocks).to(device)
    # DDP is created before memory sampling; its reducer/gradient buckets are
    # materialized by warmup backward passes and remain included in every peak.
    generator_ddp = DDP(
        generator, device_ids=[rank], output_device=rank, broadcast_buffers=False
    )
    inpaint_ranker_ddp = DDP(
        inpaint_ranker, device_ids=[rank], output_device=rank, broadcast_buffers=False
    )
    direct_ranker_ddp = DDP(
        direct_ranker, device_ids=[rank], output_device=rank, broadcast_buffers=False
    )

    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=3e-4, weight_decay=1e-4)
    inpaint_optimizer = torch.optim.AdamW(
        inpaint_ranker.parameters(), lr=3e-4, weight_decay=1e-4
    )
    direct_optimizer = torch.optim.AdamW(
        direct_ranker.parameters(), lr=3e-4, weight_decay=1e-4
    )
    generator_scaler = _make_scaler()
    ranker_scaler = _make_scaler()

    gen_input = torch.rand(
        GENERATOR_BATCH, 7, TILE, PAIR_WIDTH, device=device, dtype=torch.float32
    )
    gen_target = torch.rand(
        GENERATOR_BATCH, 3, TILE, GAP_STOP - GAP_START,
        device=device,
        dtype=torch.float32,
    )
    rank_pairs_per_view = RANKER_GROUP_BATCH * GROUP_CANDIDATES
    rank_pairs = GROUP_VIEWS * rank_pairs_per_view
    rank_generator_input = torch.rand(
        GROUP_VIEWS, rank_pairs_per_view, 7, TILE, PAIR_WIDTH,
        device=device, dtype=torch.float32
    )
    rank_visible = torch.rand(
        GROUP_VIEWS, rank_pairs_per_view, 6, TILE, PAIR_WIDTH,
        device=device, dtype=torch.float32
    )
    rank_mask = torch.zeros(
        GROUP_VIEWS, rank_pairs_per_view, 1, TILE, PAIR_WIDTH,
        device=device, dtype=torch.float32
    )
    rank_mask[..., GAP_START:GAP_STOP] = 1.0
    direct_gap = torch.zeros(
        GROUP_VIEWS, rank_pairs_per_view, 3, TILE, PAIR_WIDTH,
        device=device, dtype=torch.float32
    )
    direct_input = torch.cat((rank_visible, direct_gap, rank_mask), dim=2)

    dense_generator_input = torch.rand(
        DENSE_PAIR_BATCH, 7, TILE, PAIR_WIDTH, device=device, dtype=torch.float32
    )
    dense_visible = torch.rand(
        DENSE_PAIR_BATCH, 6, TILE, PAIR_WIDTH, device=device, dtype=torch.float32
    )
    dense_mask = torch.zeros(
        DENSE_PAIR_BATCH, 1, TILE, PAIR_WIDTH, device=device, dtype=torch.float32
    )
    dense_mask[..., GAP_START:GAP_STOP] = 1.0
    dense_direct_gap = torch.zeros(
        DENSE_PAIR_BATCH, 3, TILE, PAIR_WIDTH, device=device, dtype=torch.float32
    )
    dense_direct_input = torch.cat(
        (dense_visible, dense_direct_gap, dense_mask), dim=1
    )

    generator.train()

    def generator_step() -> None:
        generator_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = generator_ddp(gen_input)
            loss = torch.sqrt((prediction - gen_target).square() + 1e-6).mean()
        generator_scaler.scale(loss).backward()
        generator_scaler.unscale_(generator_optimizer)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        generator_scaler.step(generator_optimizer)
        generator_scaler.update()

    generator_elapsed, generator_memory = _timed_stage(
        generator_step,
        device=device,
        warmup=WARMUP_STEPS,
        repeats=GENERATOR_REPEATS,
    )

    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    inpaint_ranker.train()
    direct_ranker.train()

    def ranker_step() -> None:
        inpaint_optimizer.zero_grad(set_to_none=True)
        direct_optimizer.zero_grad(set_to_none=True)
        for view in range(GROUP_VIEWS):
            inpaint_sync = inpaint_ranker_ddp.no_sync() if view == 0 else nullcontext()
            direct_sync = direct_ranker_ddp.no_sync() if view == 0 else nullcontext()
            with inpaint_sync, direct_sync:
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    gap = generator(rank_generator_input[view])
                    padded_gap = F.pad(
                        gap, (GAP_START, PAIR_WIDTH - GAP_STOP, 0, 0)
                    )
                    inpaint_input = torch.cat(
                        (rank_visible[view], padded_gap, rank_mask[view]), dim=1
                    )
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = _ranker_view_loss(inpaint_ranker_ddp(inpaint_input))
                    loss = loss + _ranker_view_loss(
                        direct_ranker_ddp(direct_input[view])
                    )
                ranker_scaler.scale(loss).backward()
        ranker_scaler.unscale_(inpaint_optimizer)
        ranker_scaler.unscale_(direct_optimizer)
        torch.nn.utils.clip_grad_norm_(inpaint_ranker.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(direct_ranker.parameters(), 1.0)
        ranker_scaler.step(inpaint_optimizer)
        ranker_scaler.step(direct_optimizer)
        ranker_scaler.update()

    ranker_elapsed, ranker_memory = _timed_stage(
        ranker_step,
        device=device,
        warmup=WARMUP_STEPS,
        repeats=RANKER_REPEATS,
    )

    inpaint_ranker.eval()
    direct_ranker.eval()

    def dense_step() -> None:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            gap = generator(dense_generator_input)
            padded_gap = F.pad(gap, (GAP_START, PAIR_WIDTH - GAP_STOP, 0, 0))
            inpaint_input = torch.cat((dense_visible, padded_gap, dense_mask), dim=1)
            inpaint_ranker(inpaint_input)
            direct_ranker(dense_direct_input)

    dense_elapsed, dense_memory = _timed_stage(
        dense_step,
        device=device,
        warmup=WARMUP_STEPS,
        repeats=DENSE_REPEATS,
    )

    losses_are_finite = all(
        math.isfinite(float(value))
        for value in (
            gen_input.mean(),
            gen_target.mean(),
            direct_input.mean(),
            dense_direct_input.mean(),
        )
    )
    if not losses_are_finite:
        raise RuntimeError("synthetic tensor finite check failed")

    stage_memory = {
        "generator_train": generator_memory,
        "joint_two_rankers_train": ranker_memory,
        "dense_generator_plus_two_rankers": dense_memory,
    }
    peak_reserved = max(value["peak_reserved_bytes"] for value in stage_memory.values())
    peak_allocated = max(value["peak_allocated_bytes"] for value in stage_memory.values())
    return {
        "capacity_key": capacity.key,
        "capacity": asdict(capacity),
        "rank": rank,
        "parameter_counts": {
            "generator": sum(value.numel() for value in generator.parameters()),
            "ranker_per_arm": sum(value.numel() for value in inpaint_ranker.parameters()),
            "pipeline": sum(value.numel() for value in generator.parameters())
            + sum(value.numel() for value in inpaint_ranker.parameters())
            + sum(value.numel() for value in direct_ranker.parameters()),
        },
        "elapsed_seconds": {
            "generator_train": generator_elapsed,
            "joint_two_rankers_train": ranker_elapsed,
            "dense_generator_plus_two_rankers": dense_elapsed,
        },
        "throughput": {
            "generator_train_pairs_per_second": (
                GENERATOR_BATCH * GENERATOR_REPEATS / generator_elapsed
            ),
            "joint_ranker_train_pairs_per_arm_per_second": (
                rank_pairs * RANKER_REPEATS / ranker_elapsed
            ),
            "dense_pipeline_pairs_per_second": (
                DENSE_PAIR_BATCH * DENSE_REPEATS / dense_elapsed
            ),
        },
        "stage_memory": stage_memory,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_bytes": peak_allocated,
        "synthetic_tensor_shape_hw": [TILE, PAIR_WIDTH],
        "synthetic_tensors_finite": losses_are_finite,
        "measured_ddp_all_reduce": True,
        "ddp_buckets_in_peak_memory": True,
        "allocator_cleared_before_capacity": True,
        "fresh_process_pair_per_capacity": True,
    }


def _worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    scratch: str,
    capacity_index: int,
) -> None:
    output = Path(scratch) / f"rank_{rank}.json"
    device_record: dict[str, Any] | None = None
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        name = torch.cuda.get_device_name(rank)
        capability = tuple(torch.cuda.get_device_capability(rank))
        if "T4" not in name.upper() or capability != (7, 5):
            raise RuntimeError(
                f"GPU {rank} is not the frozen Tesla T4 target: {name}, {capability}"
            )
        probe = torch.randn(128, 128, device=device)
        tensor_op = float((probe @ probe).mean().cpu())
        if not math.isfinite(tensor_op):
            raise RuntimeError("real CUDA tensor op was non-finite")
        device_record = {
            "index": rank,
            "name": name,
            "capability": list(capability),
            "actual_tensor_op": tensor_op,
        }
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{rendezvous}",
            rank=rank,
            world_size=world_size,
        )
        capacity = CAPACITIES[capacity_index]
        measurement = _capacity_benchmark(capacity, rank, device)
        dist.barrier()
        atomic_json(
            output,
            {
                "status": "complete",
                "rank": rank,
                "device": device_record,
                "capacity_key": capacity.key,
                "measurement": measurement,
            },
        )
    except torch.cuda.OutOfMemoryError as error:
        atomic_json(
            output,
            {
                "status": "oom",
                "rank": rank,
                "device": device_record,
                "capacity_key": CAPACITIES[capacity_index].key,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise
    except Exception as error:
        atomic_json(
            output,
            {
                "status": "failed",
                "rank": rank,
                "device": device_record,
                "capacity_key": CAPACITIES[capacity_index].key,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def aggregate_capacity(
    capacity: Capacity, rank_reports: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate one isolated process pair, conservatively using the slow rank."""

    if len(rank_reports) != 2 or [value["rank"] for value in rank_reports] != [0, 1]:
        raise RuntimeError("exactly rank 0 and rank 1 reports are required")
    if any(report.get("capacity_key") != capacity.key for report in rank_reports):
        raise RuntimeError("per-rank capacity identity drift")
    statuses = [str(report.get("status")) for report in rank_reports]
    devices = [report.get("device") for report in rank_reports]
    if "oom" in statuses:
        return {
            "capacity_key": capacity.key,
            "capacity": asdict(capacity),
            "status": "oom",
            "isolated_fresh_process_pair": True,
            "devices": devices,
            "rank_reports": rank_reports,
        }
    if statuses != ["complete", "complete"]:
        raise RuntimeError(f"capacity {capacity.key} failed: {rank_reports}")
    rows = [report["measurement"] for report in rank_reports]
    if any(row["capacity_key"] != capacity.key for row in rows):
        raise RuntimeError("per-rank measurement capacity drift")
    parameter_counts = rows[0]["parameter_counts"]
    if rows[1]["parameter_counts"] != parameter_counts:
        raise RuntimeError("per-rank model parameter count drift")
    rate_keys = (
        "generator_train_pairs_per_second",
        "joint_ranker_train_pairs_per_arm_per_second",
        "dense_pipeline_pairs_per_second",
    )
    per_rank_rates = {
        key: [float(row["throughput"][key]) for row in rows] for key in rate_keys
    }
    # Each GPU processes an equal source shard.  Completion is gated by the
    # slower rank, so never sum unequal optimistic rates.  Training rates have
    # already paid the actual DDP all-reduce cost inside every timed backward.
    effective_rates = {
        key: 2.0 * min(values) for key, values in per_rank_rates.items()
    }
    imbalance = {
        key: max(values) / min(values) for key, values in per_rank_rates.items()
    }
    return {
        "capacity_key": capacity.key,
        "capacity": asdict(capacity),
        "status": "complete",
        "parameter_counts": parameter_counts,
        "throughput_2gpu": effective_rates,
        "throughput_aggregation": "2*minimum_per_rank_rate",
        "ddp_all_reduce_cost_measured_in_training_rates": True,
        "per_rank_throughput": per_rank_rates,
        "rank_imbalance_ratio": imbalance,
        "peak_reserved_bytes_per_gpu": [
            int(row["peak_reserved_bytes"]) for row in rows
        ],
        "peak_allocated_bytes_per_gpu": [
            int(row["peak_allocated_bytes"]) for row in rows
        ],
        "ddp_buckets_in_peak_memory": all(
            row.get("ddp_buckets_in_peak_memory") is True for row in rows
        ),
        "isolated_fresh_process_pair": True,
        "allocator_cleared_before_capacity": all(
            row.get("allocator_cleared_before_capacity") is True for row in rows
        ),
        "devices": devices,
        "per_rank": rows,
    }


def _run_capacity_pair(
    capacity_index: int, scratch_root: Path
) -> dict[str, Any]:
    capacity = CAPACITIES[capacity_index]
    scratch = scratch_root / f"capacity_{capacity_index}_{capacity.key}"
    scratch.mkdir(parents=True, exist_ok=False)
    rendezvous = str(scratch / "nccl_rendezvous")
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_worker,
            args=(rank, 2, rendezvous, str(scratch), capacity_index),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + CAPACITY_PROCESS_TIMEOUT_SECONDS
    timed_out = False
    peer_failed = False
    oom_observed = False
    while any(process.is_alive() for process in processes):
        for rank in range(2):
            report_path = scratch / f"rank_{rank}.json"
            if report_path.is_file():
                try:
                    value = json.loads(report_path.read_text(encoding="utf-8"))
                    oom_observed = oom_observed or value.get("status") == "oom"
                except (OSError, json.JSONDecodeError):
                    pass
        peer_failed = any(
            process.exitcode not in (None, 0) for process in processes
        )
        if oom_observed or peer_failed:
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(0.25)
    if timed_out or oom_observed or peer_failed:
        for process in processes:
            if process.is_alive():
                process.terminate()
    for process in processes:
        process.join(timeout=30.0)

    reports: list[dict[str, Any] | None] = []
    for rank in range(2):
        report_path = scratch / f"rank_{rank}.json"
        reports.append(
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else None
        )
    observed = [value for value in reports if value is not None]
    # A peer may be blocked in NCCL when the other rank OOMs.  The OOM report
    # is sufficient to reject only this isolated capacity and continue smaller.
    if any(value.get("status") == "oom" for value in observed):
        normalized = []
        for rank, value in enumerate(reports):
            normalized.append(
                value
                if value is not None
                else {
                    "status": "oom_peer_terminated",
                    "rank": rank,
                    "device": None,
                    "capacity_key": capacity.key,
                }
            )
        # Normalize peer termination to OOM for pure aggregation semantics.
        normalized = [
            {**value, "status": "oom"}
            if value.get("status") == "oom_peer_terminated"
            else value
            for value in normalized
        ]
        return aggregate_capacity(capacity, normalized)
    if timed_out:
        raise RuntimeError(f"capacity {capacity.key} timed out without an OOM report")
    if any(value is None for value in reports):
        raise RuntimeError(f"capacity {capacity.key} exited without both rank reports")
    return aggregate_capacity(capacity, [value for value in reports if value is not None])


def _nvidia_smi() -> dict[str, Any]:
    completed = subprocess.run(
        ["nvidia-smi"], capture_output=True, check=False, text=True
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_benchmark(output: str | Path) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("frozen benchmark requires exactly two visible CUDA GPUs")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    capabilities = [
        list(torch.cuda.get_device_capability(index)) for index in range(2)
    ]
    if any("T4" not in name.upper() for name in names) or any(
        value != [7, 5] for value in capabilities
    ):
        raise RuntimeError(
            f"frozen benchmark requires two Tesla T4 sm_75 GPUs: {names}, {capabilities}"
        )

    contract = frozen_contract()
    nvidia_smi_before = _nvidia_smi()
    if nvidia_smi_before["returncode"] != 0:
        raise RuntimeError(f"nvidia-smi preflight failed: {nvidia_smi_before}")
    with tempfile.TemporaryDirectory(prefix="masked_gap_t4_benchmark_") as scratch:
        scratch_root = Path(scratch)
        measurements = [
            _run_capacity_pair(index, scratch_root)
            for index in range(len(CAPACITIES))
        ]
        selected, projections = select_largest_feasible(measurements)
    device_evidence = next(
        (
            value.get("devices")
            for value in measurements
            if isinstance(value.get("devices"), list)
            and len(value["devices"]) == 2
            and all(device is not None for device in value["devices"])
        ),
        None,
    )
    if device_evidence is None:
        raise RuntimeError("no complete two-rank device evidence was recorded")
    source_path = Path(__file__).resolve()
    report = {
        "kind": REPORT_KIND,
        "status": "complete" if selected is not None else "aborted_no_feasible_capacity",
        "safe_for_submission": False,
        "launches_scientific_training": False,
        "synthetic_optimizer_steps": True,
        "weights_discarded": True,
        "synthetic_only": True,
        "scientific_images_labels_targets_opened": False,
        "benchmark_source_sha256": sha256(source_path),
        "contract_sha256": canonical_json_sha256(contract),
        "contract": contract,
        "hardware": {
            "required": "exactly 2x Tesla T4 sm_75",
            "devices": device_evidence,
            "nvidia_smi_before": nvidia_smi_before,
            "nvidia_smi_after": _nvidia_smi(),
        },
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "candidates": projections,
        "selected_capacity": None
        if selected is None
        else {
            "capacity_key": selected["capacity_key"],
            "capacity": selected["capacity"],
            "projected_seconds_with_1p35_safety": selected[
                "projected_seconds_with_1p35_safety"
            ],
            "projected_hours_with_1p35_safety": selected[
                "projected_hours_with_1p35_safety"
            ],
            "max_peak_reserved_bytes": selected["max_peak_reserved_bytes"],
            "execution_route": "DDP_T4x2_AMP_v2",
        },
        "selection_is_engineering_only": True,
        "scientific_hypothesis_or_threshold_changed": False,
    }
    atomic_json(output, report)
    if selected is None:
        raise NoFeasibleCapacity(
            "no capacity meets the frozen time and memory thresholds"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="/kaggle/working/masked_gap_t4_ddp_selection_v2.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        output.unlink()
    try:
        report = run_benchmark(output)
    except Exception as error:
        # A no-capacity decision already has a sealed report containing all
        # candidate measurements.  Preserve it while still exiting non-zero.
        if not output.is_file():
            source_path = Path(__file__).resolve()
            failure = {
                "kind": REPORT_KIND,
                "status": "failed",
                "safe_for_submission": False,
                "launches_scientific_training": False,
                "synthetic_optimizer_steps": True,
                "weights_discarded": True,
                "synthetic_only": True,
                "scientific_images_labels_targets_opened": False,
                "benchmark_source_sha256": sha256(source_path),
                "contract_sha256": canonical_json_sha256(frozen_contract()),
                "contract": frozen_contract(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            atomic_json(output, failure)
        raise
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
