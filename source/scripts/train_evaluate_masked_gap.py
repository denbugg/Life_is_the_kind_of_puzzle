#!/usr/bin/env python3
"""Train and leakage-gate the two-sided masked-gap pair scorer.

The CLI deliberately uses physically separate commands.  ``prepare`` writes
input-only and label-only fixtures to different directories.  ``phase-a`` can
accept only an input manifest plus a frozen checkpoint and produces dense
all-575 scores without opening targets.  ``authorize`` hash-binds those frozen
scores.  A separate ``phase-b`` process is the first process allowed to open
the label manifest and compute reconstruction/retrieval metrics.

The shared upstream TileNAF restorer is frozen but may have been exposed to
``edge_development`` source images.  Consequently this gate measures
incremental downstream signal under a shared-frozen upstream representation;
it is not claimed to be fully source-unseen absolute validation.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


REPO_ROOT = Path(__file__).resolve().parents[1]
for value in (REPO_ROOT, REPO_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.geometry import GRID, TILE_COUNT, inverse_permutation
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.masked_gap import (
    DOWN,
    RIGHT,
    MaskedGapGenerator,
    PairGroups,
    PairListwiseRanker,
    blend_with_w4,
    canonical_pair_canvas,
    charbonnier_loss,
    clean_gap_target,
    gap_baselines,
    generator_input,
    hard_negative_groups,
    listwise_view_loss,
    load_models,
    module_state_sha256,
    ranker_input,
    state_dict_payload,
)
from puzzle_assembly.metrics import retrieval_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8


PANELS = ("primary_kornia", "independent_libjpeg")
MASTER_SEED = 20260713
FROZEN_SPLITS = {
    "train": ("edge_train", 4096, 96, "8cb529285a13c710d483531121819ebc340f22cfdae16182742d1dfdcf6039cf"),
    "calibration_a": ("edge_development", 384, 4, "b9cd5752c2745e67f52f3b30aae03416a78a5640d282a337cca1ad2ec13ff07c"),
    "calibration_b": ("edge_development", 388, 4, "1f6fb483add0d647d30f1584f23488f25920eeb3192c782c5075d735cf43e492"),
    "holdout": ("edge_development", 392, 8, "a0e3c535212bb73a736bc6a0763d500c37b92a49bdd141a1699ee5b6304ef3f8"),
}
INPUT_MANIFEST_KIND = "masked_gap_input_manifest_v1"
LABEL_MANIFEST_KIND = "masked_gap_label_manifest_v1"
SECRET_SEED_MAPPING_KIND = "masked_gap_secret_panel_seed_mapping_v1"
PHASE_A_KIND = "masked_gap_phase_a_report_v1"
AUTH_KIND = "masked_gap_global_phase_b_authorization_v1"
PHASE_B_KIND = "masked_gap_phase_b_report_v1"
INPUT_KEYS = {"raw_tiles", "denoised_tiles", "w4_right", "w4_down"}
LABEL_KEYS = {"slot_to_target", "clean_slot_tiles"}
EXTERNAL_BENCHMARK_KIND = "masked_gap_t4x2_amp_ddp_capacity_selection_v2"
TRAIN_EPOCHS = 2
GENERATOR_BATCH = 128
RANKER_GROUP_BATCH = 4
DENSE_PAIR_BATCH = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
TRUE_GROUPS = 2 * GRID * (GRID - 1)
EXPECTED_BENCHMARK_SOURCE_SHA256 = "2ee1f73992df440c90949e71edca9aa5e5a7289b5811486852a25ab79def07c5"
EXPECTED_BENCHMARK_BUNDLE_SHA256 = "5237d7f033122248029c4f01277af17022306a4a2ab3b33d35241b32b060e1f4"
EXPECTED_BENCHMARK_CONTRACT_SHA256 = "3b396bb6fd8a2945e6cd43fc82a9b36a92d4eba7fe3de89babed88b416bc2be6"
EXPECTED_CAPACITY_REPORT_SHA256 = "4fe36a4cf8fd637b519aa18da8a5b1c6ca762458fbebba3e33b226ebe3d09843"
EXPECTED_CAPACITY_WRAPPER_REPORT_SHA256 = "b46d30c4486f0d2b2502f01993a181ac002bbdcc7c121ec30e60857a4fa2bb04"
EXPECTED_SELECTED_CAPACITY = {
    "capacity_key": "w32_g3_r3",
    "capacity": {"width": 32, "generator_blocks": 3, "ranker_blocks": 3},
    "projected_seconds_with_1p35_safety": 12591.709799280843,
    "projected_hours_with_1p35_safety": 3.4976971664669008,
    "max_peak_reserved_bytes": 1031798784,
    "execution_route": "DDP_T4x2_AMP_v2",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def frozen_scientific_config(
    *,
    width: int,
    generator_blocks: int,
    ranker_blocks: int,
    capacity_selection_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "canonical_pair_shape": [3, 20, 40],
        "masked_central_columns": [18, 19, 20, 21],
        "generator_input": "masked raw RGB + masked denoised RGB + binary mask",
        "generator": {"width": int(width), "blocks": int(generator_blocks)},
        "ranker": {"width": int(width), "blocks": int(ranker_blocks), "input_channels": 10},
        "direct_control": "same architecture and initialization; predicted-gap channels zero",
        "hard_negative_miner": "single production w4=C1+HBT(weight4), stable top31",
        "ranker_group": "1 positive + 31 negatives",
        "ranker_loss": "outgoing CE + incoming CE + 0.25 BCE",
        "candidate": "equal row-rank w4 + masked-gap inpaint learned cost",
        "direct_control_score": "equal row-rank w4 + equal-capacity direct learned cost",
        "dense_candidates_per_query": 575,
        "training_workload": {
            "train_sources": 96,
            "train_sources_per_rank": 48,
            "panels_per_source_per_epoch": 2,
            "epochs": TRAIN_EPOCHS,
            "generator_true_pairs_per_source_panel_epoch": TRUE_GROUPS,
            "ranker_groups_per_source_panel_epoch": TRUE_GROUPS,
            "ranker_candidates_per_group": 32,
            "ranker_views": ["outgoing", "incoming"],
            "subsampling": False,
        },
        "batches": {
            "generator_per_rank": GENERATOR_BATCH,
            "ranker_groups_per_rank": RANKER_GROUP_BATCH,
            "dense_pairs_per_rank": DENSE_PAIR_BATCH,
        },
        "precision": {
            "training_autocast": "float16",
            "grad_scaler": True,
            "execution": "torch DistributedDataParallel",
            "world_size": 2,
            "backend": "nccl",
            "allreduce_every_optimizer_step": True,
            "separate_ranker_optimizers": True,
            "ranker_microsteps": "view0 no_sync backward plus view1 synchronized backward",
            "ranker_update": "separate unscale, max_norm=1.0 clip, and step per ranker",
        },
        "qap": False,
    }
    if capacity_selection_binding is not None:
        config["external_capacity_selection"] = dict(capacity_selection_binding)
    return config


@dataclass(frozen=True)
class Prepared:
    name: str
    panel: str
    seed: int
    raw: np.ndarray
    denoised: np.ndarray
    clean_slots: np.ndarray
    slot_to_target: np.ndarray
    w4: CompatibilityMatrices | None


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def matrix_fingerprint(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix)
    off_diagonal = ~np.eye(TILE_COUNT, dtype=bool)
    if values.shape != (TILE_COUNT, TILE_COUNT) or values.dtype != np.float32:
        raise RuntimeError("score matrix must be float32 576x576")
    if not np.all(np.isposinf(np.diag(values))) or not np.all(np.isfinite(values[off_diagonal])):
        raise RuntimeError("score matrix requires +inf diagonal and finite off-diagonal")
    return {
        "shape": [TILE_COUNT, TILE_COUNT],
        "dtype": "float32",
        "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
        "positive_inf_diagonal": True,
        "finite_off_diagonal": True,
    }


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_secret_panel_seeds(
    path: str | Path,
    *,
    split: str,
    names: list[str],
) -> dict[tuple[str, str], int]:
    """Load the complete label-only uint64 seed map for a sealed gate split."""

    payload = read_json(Path(path).resolve(strict=True))
    if set(payload) != {"kind", "split", "records"}:
        raise RuntimeError("secret panel seed mapping schema mismatch")
    if payload.get("kind") != SECRET_SEED_MAPPING_KIND or payload.get("split") != split:
        raise RuntimeError("secret panel seed mapping identity mismatch")
    records = payload.get("records")
    expected = [(name, panel) for name in names for panel in PANELS]
    if not isinstance(records, list) or len(records) != len(expected):
        raise RuntimeError("secret panel seed mapping record count mismatch")
    actual: list[tuple[str, str]] = []
    seeds: list[int] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"name", "panel", "seed"}:
            raise RuntimeError("secret panel seed record schema mismatch")
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise RuntimeError("secret panel seed must be a uint64 integer")
        actual.append((str(record.get("name")), str(record.get("panel"))))
        seeds.append(seed)
    if actual != expected:
        raise RuntimeError("secret panel seed mapping record set/order mismatch")
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("secret panel seeds must be unique")
    return {identity: seed for identity, seed in zip(actual, seeds, strict=True)}


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if value.shape != (480, 480, 3):
        raise RuntimeError(f"unexpected image shape: {path}: {value.shape}")
    return value


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    active: bool

    @property
    def primary(self) -> bool:
        return self.rank == 0


def shard_indices(length: int, rank: int, world_size: int) -> list[int]:
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid distributed rank/world size")
    return list(range(rank, length, world_size))


def merge_indexed_shards(shards: list[list[dict[str, Any]]], total: int) -> list[dict[str, Any]]:
    merged = [record for shard in shards for record in shard]
    if len(merged) != total or sorted(int(record["index"]) for record in merged) != list(range(total)):
        raise RuntimeError("distributed record shard coverage is not exact")
    return sorted(merged, key=lambda record: int(record["index"]))


def distributed_context(requested_device: str, *, required: bool) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1:
        if required:
            raise RuntimeError("this command requires torchrun with exactly two processes")
        return DistributedContext(0, 0, 1, resolve_device(requested_device), False)
    if world_size != 2 or rank not in (0, 1) or local_rank not in (0, 1):
        raise RuntimeError("scientific distributed execution requires world_size=2 and local ranks 0,1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("scientific distributed execution requires exactly two visible CUDA GPUs")
    for index in range(2):
        if (
            "T4" not in torch.cuda.get_device_name(index).upper()
            or list(torch.cuda.get_device_capability(index)) != [7, 5]
        ):
            raise RuntimeError("scientific distributed execution requires exactly 2x Tesla T4 sm75")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    return DistributedContext(rank, local_rank, world_size, torch.device(f"cuda:{local_rank}"), True)


def ddp_model(model: nn.Module, context: DistributedContext) -> nn.Module:
    if not context.active:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
    )


def gather_objects(value: Any, context: DistributedContext) -> list[Any]:
    if not context.active:
        return [value]
    gathered: list[Any] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def distributed_mean(total: float, count: int, context: DistributedContext) -> float:
    values = torch.tensor([total, float(count)], dtype=torch.float64, device=context.device)
    if context.active:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if values[1].item() <= 0:
        raise RuntimeError("cannot average an empty distributed metric")
    return float(values[0].item() / values[1].item())


def make_cuda_grad_scaler() -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:  # pragma: no cover - older compatible PyTorch fallback.
        return torch.cuda.amp.GradScaler(enabled=True)  # type: ignore[return-value]


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


def frozen_names(
    key: str, *, manifest: str | Path, quarantine: str | Path
) -> list[str]:
    split, offset, count, expected_hash = FROZEN_SPLITS[key]
    names = source_names_for_split(
        split, manifest_path=manifest, quarantine_path=quarantine
    )[offset : offset + count]
    if len(names) != count or names_sha256(names) != expected_hash:
        raise RuntimeError(f"frozen {key} source list/hash drift")
    if any(Path(name).name != name or not name.endswith(".png") for name in names):
        raise RuntimeError(f"frozen {key} contains a non-basename PNG")
    return names


def protocol_audit(
    *,
    manifest: str | Path,
    quarantine: str | Path,
    embedding_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    groups = {
        key: frozen_names(key, manifest=manifest, quarantine=quarantine)
        for key in FROZEN_SPLITS
    }
    for first, first_names in groups.items():
        for second, second_names in groups.items():
            if first < second and set(first_names) & set(second_names):
                raise RuntimeError(f"source exposure overlap: {first} vs {second}")
    manifest_payload = read_json(manifest)
    quarantine_payload = read_json(quarantine)
    audit_names = {str(name) for name in manifest_payload.get("splits", {}).get("audit", [])}
    quarantine_names = {str(name) for name in quarantine_payload.get("quarantine_names", [])}
    assembly_names = set(source_names_for_split(
        "assembly_cal", manifest_path=manifest, quarantine_path=quarantine
    )) | set(source_names_for_split(
        "assembly_incremental_gate", manifest_path=manifest, quarantine_path=quarantine
    ))
    all_frozen = set().union(*(set(names) for names in groups.values()))
    exposure_intersections = {
        "quarantine": sorted(all_frozen & quarantine_names),
        "audit": sorted(all_frozen & audit_names),
        "assembly_cal_or_incremental": sorted(all_frozen & assembly_names),
    }
    resolved_embedding = Path(
        embedding_checkpoint
        or REPO_ROOT / "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt"
    ).resolve(strict=True)
    embedding_payload = torch.load(resolved_embedding, map_location="cpu", weights_only=False)
    embedding_metadata = embedding_payload.get("metadata", {})
    hbt_names = {
        str(name)
        for key in ("train_names", "val_names")
        for name in embedding_metadata.get(key, [])
    }
    exposure_intersections["frozen_hbt_train_or_val"] = sorted(all_frozen & hbt_names)
    if any(exposure_intersections.values()):
        raise RuntimeError(f"frozen split exposure audit failed: {exposure_intersections}")
    return {
        "exact_basename_audit": True,
        "all_splits_pairwise_disjoint": True,
        "all_112_vs_quarantine_audit_assembly_hbt_intersections_empty": True,
        "exposure_intersections": exposure_intersections,
        "hbt_checkpoint_sha256": sha256(resolved_embedding),
        "hbt_exposure_names": len(hbt_names),
        "splits": {
            key: {"count": len(names), "names": names, "names_sha256": names_sha256(names)}
            for key, names in groups.items()
        },
        "upstream_exposure_disclosure": (
            "The frozen shared TileNAF restorer may have seen edge_development sources. "
            "This is incremental downstream validation, not fully source-unseen absolute validation."
        ),
        "forbidden_historical_artifacts": ["spatial-prior", "l1-real-pseudo"],
    }


def tensor_bank(values: np.ndarray, device: torch.device) -> torch.Tensor:
    if values.shape != (TILE_COUNT, 20, 20, 3) or values.dtype != np.uint8:
        raise ValueError("tile bank must be uint8 576x20x20x3")
    return torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    ).div_(255.0)


def build_w4(
    denoised: np.ndarray,
    *,
    embedding: nn.Module,
    device: torch.device,
) -> CompatibilityMatrices:
    hbt, _ = learned_compatibility(embedding, denoised, device=device, name="frozen_hbt")
    bank = build_classical_score_bank(denoised, prefix="denoised", chunk_size=64)
    names = [name for name in sorted(bank) if not name.endswith("_c2")]
    c1 = fuse_ranked_scores(bank, names=names, name="frozen_c1")
    return fuse_ranked_scores(
        {"c1": c1, "hbt": hbt},
        names=["c1", "hbt"],
        weights={"hbt": 4.0},
        name="frozen_w4",
    )


def prepare_source(
    name: str,
    panel_name: str,
    split_key: str,
    *,
    data_root: str | Path,
    restorer: nn.Module,
    embedding: nn.Module,
    device: torch.device,
    denoise_batch_size: int,
    seed: int,
    panel_seed: int | None = None,
    require_w4: bool = True,
) -> Prepared:
    sealed_gate_split = split_key in {"calibration_b", "holdout"}
    if sealed_gate_split and panel_seed is None:
        raise RuntimeError("sealed gate source preparation requires an explicit secret panel seed")
    if not sealed_gate_split and panel_seed is not None:
        raise RuntimeError("secret panel seeds are reserved for sealed gate fixtures")
    if panel_seed is None:
        panel_seed = per_source_seed(seed, f"masked-gap-{split_key}-{panel_name}", name)
    clean = read_rgb(Path(data_root) / "train" / "targets" / name)
    panel = make_exact_panel(clean, panel=panel_name, seed=panel_seed)
    denoised = restore_tiles_uint8(restorer, panel.slot_tiles, device, batch_size=denoise_batch_size)
    w4 = build_w4(denoised, embedding=embedding, device=device) if require_w4 else None
    clean_slots = np.ascontiguousarray(panel.clean_target_tiles[panel.slot_to_target])
    return Prepared(
        name=name,
        panel=panel_name,
        seed=int(panel_seed),
        raw=panel.slot_tiles,
        denoised=denoised,
        clean_slots=clean_slots,
        slot_to_target=panel.slot_to_target,
        w4=w4,
    )


def true_pairs(slot_to_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_to_slot = inverse_permutation(slot_to_target)
    first: list[int] = []
    second: list[int] = []
    direction: list[int] = []
    for position in range(TILE_COUNT):
        row, column = divmod(position, GRID)
        if column + 1 < GRID:
            first.append(int(position_to_slot[position]))
            second.append(int(position_to_slot[position + 1]))
            direction.append(RIGHT)
        if row + 1 < GRID:
            first.append(int(position_to_slot[position]))
            second.append(int(position_to_slot[position + GRID]))
            direction.append(DOWN)
    return (
        np.asarray(first, dtype=np.int32),
        np.asarray(second, dtype=np.int32),
        np.asarray(direction, dtype=np.int64),
    )


def _pair_tensors(
    raw: torch.Tensor,
    denoised: torch.Tensor,
    first: np.ndarray,
    second: np.ndarray,
    direction: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_t = torch.as_tensor(first, device=raw.device, dtype=torch.long)
    second_t = torch.as_tensor(second, device=raw.device, dtype=torch.long)
    direction_t = torch.as_tensor(direction, device=raw.device, dtype=torch.long)
    return raw[first_t], raw[second_t], denoised[first_t], denoised[second_t], direction_t


def _ranker_logits(
    ranker: nn.Module,
    generator: nn.Module,
    raw: torch.Tensor,
    denoised: torch.Tensor,
    first: np.ndarray,
    second: np.ndarray,
    direction: np.ndarray,
    *,
    inpaint: bool,
) -> torch.Tensor:
    values = _pair_tensors(raw, denoised, first, second, direction)
    predicted = None
    if inpaint:
        with torch.no_grad():
            predicted = generator(generator_input(*values))
    return ranker(ranker_input(*values, predicted))


def _group_logits(
    ranker: nn.Module,
    generator: nn.Module,
    raw: torch.Tensor,
    denoised: torch.Tensor,
    groups: PairGroups,
    indices: np.ndarray,
    *,
    inpaint: bool,
) -> torch.Tensor:
    first = groups.first[indices]
    second = groups.second[indices]
    direction = np.repeat(groups.direction[indices, None], first.shape[1], axis=1)
    logits = _ranker_logits(
        ranker,
        generator,
        raw,
        denoised,
        first.reshape(-1),
        second.reshape(-1),
        direction.reshape(-1),
        inpaint=inpaint,
    )
    return logits.reshape(len(indices), -1)


@torch.inference_mode()
def dense_scores(
    generator: nn.Module,
    inpaint_ranker: nn.Module,
    direct_ranker: nn.Module,
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    *,
    device: torch.device,
    query_chunk: int,
    pair_chunk: int,
    amp: bool = True,
) -> tuple[CompatibilityMatrices, CompatibilityMatrices]:
    """Score all 575 non-self candidates per direction without labels."""

    generator.eval()
    inpaint_ranker.eval()
    direct_ranker.eval()
    raw = tensor_bank(raw_tiles, device)
    denoised = tensor_bank(denoised_tiles, device)
    outputs: dict[str, list[np.ndarray]] = {"inpaint": [], "direct": []}
    all_candidates = np.arange(TILE_COUNT, dtype=np.int32)
    for direction_value in (RIGHT, DOWN):
        inpaint_matrix = np.full((TILE_COUNT, TILE_COUNT), np.inf, dtype=np.float32)
        direct_matrix = np.full_like(inpaint_matrix, np.inf)
        for query_start in range(0, TILE_COUNT, query_chunk):
            queries = np.arange(query_start, min(query_start + query_chunk, TILE_COUNT), dtype=np.int32)
            first = np.repeat(queries, TILE_COUNT - 1)
            second = np.concatenate([
                all_candidates[all_candidates != query]
                for query in queries
            ])
            direction = np.full(len(first), direction_value, dtype=np.int64)
            inpaint_parts: list[np.ndarray] = []
            direct_parts: list[np.ndarray] = []
            for start in range(0, len(first), pair_chunk):
                selected = slice(start, start + pair_chunk)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=bool(amp and device.type == "cuda"),
                ):
                    inpaint_logits = _ranker_logits(
                            inpaint_ranker, generator, raw, denoised,
                            first[selected], second[selected], direction[selected], inpaint=True,
                        )
                    direct_logits = _ranker_logits(
                            direct_ranker, generator, raw, denoised,
                            first[selected], second[selected], direction[selected], inpaint=False,
                        )
                inpaint_parts.append((-inpaint_logits).float().cpu().numpy())
                direct_parts.append((-direct_logits).float().cpu().numpy())
            inpaint_rows = np.concatenate(inpaint_parts).reshape(len(queries), TILE_COUNT - 1)
            direct_rows = np.concatenate(direct_parts).reshape(len(queries), TILE_COUNT - 1)
            for row, query in enumerate(queries):
                nonself = all_candidates != query
                inpaint_matrix[query, nonself] = inpaint_rows[row]
                direct_matrix[query, nonself] = direct_rows[row]
        inpaint_matrix = np.asarray(inpaint_matrix, dtype=np.float32)
        direct_matrix = np.asarray(direct_matrix, dtype=np.float32)
        for name, matrix in (("inpaint", inpaint_matrix), ("direct", direct_matrix)):
            off_diagonal = ~np.eye(TILE_COUNT, dtype=bool)
            if matrix.shape != (TILE_COUNT, TILE_COUNT) or matrix.dtype != np.float32:
                raise RuntimeError(f"dense {name} matrix shape/dtype drift")
            if not np.all(np.isposinf(np.diag(matrix))) or not np.all(np.isfinite(matrix[off_diagonal])):
                raise RuntimeError(f"dense {name} matrix must have +inf diagonal and finite off-diagonal")
        outputs["inpaint"].append(inpaint_matrix)
        outputs["direct"].append(direct_matrix)
    return (
        CompatibilityMatrices("masked_gap_inpaint", *outputs["inpaint"]),
        CompatibilityMatrices("direct_no_inpaint", *outputs["direct"]),
    )


def compact_metrics(score: CompatibilityMatrices, truth: np.ndarray) -> dict[str, float]:
    values = retrieval_metrics(score, truth, ks=(1, 5))["combined"]
    return {
        "mrr": float(values["mrr"]),
        "recall_at_1": float(values["recall_at_1"]),
        "recall_at_5": float(values["recall_at_5"]),
    }


def _generator_reconstruction(
    generator: nn.Module,
    prepared: Prepared,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    first, second, direction = true_pairs(prepared.slot_to_target)
    raw = tensor_bank(prepared.raw, device)
    denoised = tensor_bank(prepared.denoised, device)
    clean = tensor_bank(prepared.clean_slots, device)
    totals = {
        "generator_charbonnier": 0.0,
        "copy_charbonnier": 0.0,
        "interpolation_charbonnier": 0.0,
        "generator_mae": 0.0,
        "copy_mae": 0.0,
        "interpolation_mae": 0.0,
    }
    count = 0
    generator.eval()
    with torch.inference_mode():
        for start in range(0, len(first), batch_size):
            selected = slice(start, start + batch_size)
            values = _pair_tensors(raw, denoised, first[selected], second[selected], direction[selected])
            predicted = generator(generator_input(*values))
            clean_values = _pair_tensors(clean, clean, first[selected], second[selected], direction[selected])
            target = clean_gap_target(clean_values[0], clean_values[1], clean_values[4])
            canvas = canonical_pair_canvas(values[2], values[3], values[4])
            baselines = gap_baselines(canvas)
            batch_count = len(predicted)
            totals["generator_charbonnier"] += float(charbonnier_loss(predicted, target)) * batch_count
            totals["generator_mae"] += float(torch.mean(torch.abs(predicted - target))) * batch_count
            for key in ("copy", "interpolation"):
                totals[f"{key}_charbonnier"] += float(charbonnier_loss(baselines[key], target)) * batch_count
                totals[f"{key}_mae"] += float(torch.mean(torch.abs(baselines[key] - target))) * batch_count
            count += batch_count
    return {key: value / count for key, value in totals.items()}


def validate_external_capacity_selection(
    report_path: str | Path,
    wrapper_path: str | Path,
    *,
    expected_report_sha256: str,
    expected_wrapper_sha256: str,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Validate the standalone T4x2 selection before any scientific image opens."""

    report_path = Path(report_path).resolve(strict=True)
    wrapper_path = Path(wrapper_path).resolve(strict=True)
    for label, expected, path in (
        ("capacity report", expected_report_sha256, report_path),
        ("capacity wrapper", expected_wrapper_sha256, wrapper_path),
    ):
        if not isinstance(expected, str) or len(expected) != 64 or expected.startswith("__"):
            raise RuntimeError(f"{label} expected SHA256 is unresolved")
        if sha256(path) != expected:
            raise RuntimeError(f"{label} file SHA256 mismatch")
    if (
        expected_report_sha256 != EXPECTED_CAPACITY_REPORT_SHA256
        or expected_wrapper_sha256 != EXPECTED_CAPACITY_WRAPPER_REPORT_SHA256
    ):
        raise RuntimeError("external capacity report/wrapper are not the frozen approved files")
    if EXPECTED_BENCHMARK_SOURCE_SHA256.startswith("__") or EXPECTED_BENCHMARK_CONTRACT_SHA256.startswith("__"):
        raise RuntimeError("external benchmark source/contract hashes are not pinned")

    report = read_json(report_path)
    required_false = (
        "safe_for_submission",
        "launches_scientific_training",
        "scientific_images_labels_targets_opened",
    )
    if report.get("kind") != EXTERNAL_BENCHMARK_KIND or report.get("status") != "complete":
        raise RuntimeError("external capacity report kind/status mismatch")
    if any(report.get(key) is not False for key in required_false):
        raise RuntimeError("external capacity report fail-closed flags mismatch")
    if (
        report.get("synthetic_only") is not True
        or report.get("synthetic_optimizer_steps") is not True
        or report.get("weights_discarded") is not True
        or report.get("selection_is_engineering_only") is not True
        or report.get("scientific_hypothesis_or_threshold_changed") is not False
        or report.get("benchmark_source_sha256") != EXPECTED_BENCHMARK_SOURCE_SHA256
        or report.get("contract_sha256") != EXPECTED_BENCHMARK_CONTRACT_SHA256
        or canonical_hash(report.get("contract")) != EXPECTED_BENCHMARK_CONTRACT_SHA256
    ):
        raise RuntimeError("external capacity source/contract/synthetic proof mismatch")
    contract = report["contract"]
    workload = contract.get("workload", {})
    batches = contract.get("batches", {})
    if (
        workload.get("generator_train_true_pairs") != 96 * TRUE_GROUPS * 2 * TRAIN_EPOCHS
        or workload.get("ranker_train_pair_candidates_per_arm")
        != 96 * TRUE_GROUPS * 2 * 32 * 2 * TRAIN_EPOCHS
        or workload.get("all_source_panel_preparations_tilenaf") != 808
        or workload.get("all_source_panel_preparations_w4") != 424
        or batches != {
            "ddp_generator_per_gpu": GENERATOR_BATCH,
            "ddp_ranker_groups_per_gpu": RANKER_GROUP_BATCH,
            "ddp_ranker_pairs_per_arm_per_gpu": 256,
            "ddp_dense_pairs_per_gpu": DENSE_PAIR_BATCH,
        }
        or contract.get("optimizers") != {
            "generator": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
            "inpaint_ranker": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
            "direct_ranker": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
            "ranker_grad_scaler": "one shared CUDA GradScaler",
            "ranker_microsteps": "view0 no_sync backward plus view1 synchronized backward",
            "ranker_update": "separate unscale, max_norm=1.0 clip, and step per ranker",
        }
        or contract.get("timing", {}).get("amp_dtype") != "float16"
        or contract.get("timing", {}).get("two_processes_one_per_gpu") is not True
        or contract.get("timing", {}).get("measured_ddp_all_reduce_during_training") is not True
        or contract.get("timing", {}).get("ddp_gradient_buckets_in_peak_memory") is not True
        or contract.get("timing", {}).get("data_parallel_route") != "not executed by protocol v2"
    ):
        raise RuntimeError("external capacity workload/batch contract mismatch")
    hardware = report.get("hardware", {}).get("devices")
    if not isinstance(hardware, list) or len(hardware) != 2 or any(
        device.get("index") != index
        or "T4" not in str(device.get("name", "")).upper()
        or device.get("capability") != [7, 5]
        for index, device in enumerate(hardware)
    ):
        raise RuntimeError("external capacity report lacks exact 2xT4 sm75 evidence")

    selected = report.get("selected_capacity")
    if not isinstance(selected, dict) or not isinstance(selected.get("capacity"), dict):
        raise RuntimeError("external capacity report has no selected capacity")
    if selected != EXPECTED_SELECTED_CAPACITY:
        raise RuntimeError("external capacity selected fields differ from the approved DDP measurement")
    capacity = selected["capacity"]
    expected_capacity_keys = {"width", "generator_blocks", "ranker_blocks"}
    if set(capacity) != expected_capacity_keys or any(
        not isinstance(capacity[key], int) or capacity[key] <= 0 for key in expected_capacity_keys
    ):
        raise RuntimeError("selected capacity schema mismatch")
    expected_key = f"w{capacity['width']}_g{capacity['generator_blocks']}_r{capacity['ranker_blocks']}"
    if selected.get("capacity_key") != expected_key:
        raise RuntimeError("selected capacity key/config mismatch")
    if capacity != {"width": 32, "generator_blocks": 3, "ranker_blocks": 3}:
        raise RuntimeError("DDP v2 frozen capacity selection must be w32_g3_r3")
    candidates = report.get("candidates")
    expected_capacities = [
        {"width": 64, "generator_blocks": 6, "ranker_blocks": 5},
        {"width": 48, "generator_blocks": 4, "ranker_blocks": 4},
        {"width": 32, "generator_blocks": 3, "ranker_blocks": 3},
        {"width": 24, "generator_blocks": 2, "ranker_blocks": 2},
        {"width": 16, "generator_blocks": 2, "ranker_blocks": 2},
    ]
    if contract.get("capacities_largest_first") != expected_capacities:
        raise RuntimeError("external capacity list/order contract drift")
    if not isinstance(candidates, list) or len(candidates) != len(expected_capacities):
        raise RuntimeError("external capacity candidate ledger is missing")
    if [candidate.get("capacity") for candidate in candidates] != expected_capacities:
        raise RuntimeError("external capacity candidate order/config drift")
    feasible = [value for value in candidates if value.get("feasible") is True]
    if not feasible or selected.get("capacity_key") != feasible[0].get("capacity_key"):
        raise RuntimeError("selected capacity is not the largest feasible precommitted capacity")
    for candidate in candidates:
        if candidate.get("status") == "oom":
            if candidate.get("feasible") is not False:
                raise RuntimeError("OOM DDP capacity was not rejected")
            continue
        if (
            candidate.get("status") != "complete"
            or candidate.get("throughput_aggregation") != "2*minimum_per_rank_rate"
            or candidate.get("ddp_all_reduce_cost_measured_in_training_rates") is not True
            or candidate.get("ddp_buckets_in_peak_memory") is not True
            or candidate.get("isolated_fresh_process_pair") is not True
            or candidate.get("allocator_cleared_before_capacity") is not True
        ):
            raise RuntimeError("DDP candidate measurement proof mismatch")
    if (
        selected.get("execution_route") != "DDP_T4x2_AMP_v2"
        or not math.isfinite(float(selected.get("projected_hours_with_1p35_safety")))
        or float(selected["projected_hours_with_1p35_safety"]) > 5.5
        or int(selected.get("max_peak_reserved_bytes")) > 13_500_000_000
    ):
        raise RuntimeError("selected DDP capacity exceeds the frozen feasibility gate")
    wrapper = read_json(wrapper_path)
    if (
        wrapper.get("kind") != "masked_gap_t4x2_ddp_benchmark_wrapper_v2"
        or wrapper.get("status") != "complete"
        or wrapper.get("safe_for_submission") is not False
        or wrapper.get("launches_scientific_training") is not False
        or wrapper.get("synthetic_optimizer_steps") is not True
        or wrapper.get("weights_discarded") is not True
        or wrapper.get("synthetic_only") is not True
        or wrapper.get("scientific_images_labels_targets_opened") is not False
        or wrapper.get("selection_report_sha256") != expected_report_sha256
        or wrapper.get("code_bundle_sha256") != EXPECTED_BENCHMARK_BUNDLE_SHA256
        or wrapper.get("benchmark_source_sha256") != EXPECTED_BENCHMARK_SOURCE_SHA256
        or wrapper.get("selected_capacity") != selected
    ):
        raise RuntimeError("external capacity wrapper binding mismatch")
    binding = {
        "report_sha256": expected_report_sha256,
        "wrapper_report_sha256": expected_wrapper_sha256,
        "benchmark_source_sha256": EXPECTED_BENCHMARK_SOURCE_SHA256,
        "benchmark_bundle_sha256": EXPECTED_BENCHMARK_BUNDLE_SHA256,
        "contract_sha256": EXPECTED_BENCHMARK_CONTRACT_SHA256,
        "selected_capacity_key": expected_key,
        "ddp_selection_evidence_sha256": canonical_hash({
            "candidates": candidates,
            "selected_capacity": selected,
        }),
    }
    return {key: int(capacity[key]) for key in expected_capacity_keys}, binding


def train_command(args: argparse.Namespace) -> None:
    set_determinism(args.seed)
    capacity, capacity_binding = validate_external_capacity_selection(
        args.capacity_report,
        args.capacity_wrapper_report,
        expected_report_sha256=args.capacity_report_sha256,
        expected_wrapper_sha256=args.capacity_wrapper_report_sha256,
    )
    scientific_config = frozen_scientific_config(
        width=capacity["width"],
        generator_blocks=capacity["generator_blocks"],
        ranker_blocks=capacity["ranker_blocks"],
        capacity_selection_binding=capacity_binding,
    )
    audit = protocol_audit(
        manifest=args.manifest,
        quarantine=args.quarantine,
        embedding_checkpoint=args.embedding_checkpoint,
    )
    train_names = audit["splits"]["train"]["names"]
    cal_names = audit["splits"]["calibration_a"]["names"]
    context = distributed_context(args.device, required=True)
    device = context.device
    local_train_names = [train_names[index] for index in shard_indices(len(train_names), context.rank, context.world_size)]
    local_cal_names = [cal_names[index] for index in shard_indices(len(cal_names), context.rank, context.world_size)]
    if len(local_train_names) != 48 or len(local_cal_names) != 2:
        raise RuntimeError("exact two-rank source sharding drift")
    restorer_core, _, restorer_metadata = load_restorer(args.denoiser, device=str(device))
    embedding_core, embedding_metadata = load_embedding_checkpoint(args.embedding_checkpoint, device=device)
    restorer = restorer_core
    embedding = embedding_core
    generator_core = MaskedGapGenerator(
        width=capacity["width"], blocks=capacity["generator_blocks"]
    ).to(device)
    inpaint_core = PairListwiseRanker(
        width=capacity["width"], blocks=capacity["ranker_blocks"]
    ).to(device)
    direct_core = copy.deepcopy(inpaint_core).to(device)
    generator = ddp_model(generator_core, context)
    inpaint = ddp_model(inpaint_core, context)
    direct = ddp_model(direct_core, context)
    ranker_initial_hash = module_state_sha256(inpaint_core)
    if module_state_sha256(direct_core) != ranker_initial_hash:
        raise RuntimeError("direct/inpaint initial states are not byte-identical")

    generator_optimizer = torch.optim.AdamW(
        generator_core.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator_scaler = make_cuda_grad_scaler()
    generator_epochs: list[dict[str, Any]] = []
    generator_steps = 0
    for epoch in range(TRAIN_EPOCHS):
        losses: list[float] = []
        generator.train()
        for name in local_train_names:
            for panel_name in PANELS:
                prepared = prepare_source(
                    name, panel_name, "train", data_root=args.data_root, restorer=restorer,
                    embedding=embedding, device=device, denoise_batch_size=args.denoise_batch_size,
                    seed=args.seed, require_w4=False,
                )
                first, second, direction = true_pairs(prepared.slot_to_target)
                if len(first) != TRUE_GROUPS:
                    raise RuntimeError("generator true-pair workload drift")
                selected = np.arange(TRUE_GROUPS, dtype=np.int32)
                raw = tensor_bank(prepared.raw, device)
                denoised = tensor_bank(prepared.denoised, device)
                clean = tensor_bank(prepared.clean_slots, device)
                for start in range(0, TRUE_GROUPS, GENERATOR_BATCH):
                    index = selected[start : start + GENERATOR_BATCH]
                    values = _pair_tensors(raw, denoised, first[index], second[index], direction[index])
                    clean_values = _pair_tensors(clean, clean, first[index], second[index], direction[index])
                    target = clean_gap_target(clean_values[0], clean_values[1], clean_values[4])
                    generator_optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = charbonnier_loss(generator(generator_input(*values)), target)
                    generator_scaler.scale(loss).backward()
                    generator_scaler.unscale_(generator_optimizer)
                    torch.nn.utils.clip_grad_norm_(generator_core.parameters(), 1.0)
                    generator_scaler.step(generator_optimizer)
                    generator_scaler.update()
                    generator_steps += 1
                    losses.append(float(loss.detach()))
        mean_generator_loss = distributed_mean(float(sum(losses)), len(losses), context)
        generator_epochs.append({
            "epoch": epoch + 1,
            "mean_charbonnier": mean_generator_loss,
            "source_panel_records": len(train_names) * len(PANELS),
            "source_panel_records_per_rank": len(local_train_names) * len(PANELS),
            "true_pairs_per_source_panel": TRUE_GROUPS,
            "optimizer_steps_cumulative_per_rank": generator_steps,
            "amp_dtype": "float16",
            "grad_scaler": True,
        })

    generator.eval()
    for parameter in generator_core.parameters():
        parameter.requires_grad_(False)
    # The frozen generator no longer participates in gradient reduction during
    # ranker training; each rank uses its synchronized local module directly.
    generator = generator_core
    frozen_generator_hash = module_state_sha256(generator_core)
    inpaint_optimizer = torch.optim.AdamW(
        inpaint_core.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    direct_optimizer = torch.optim.AdamW(
        direct_core.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    ranker_scaler = make_cuda_grad_scaler()
    ranker_epochs: list[dict[str, Any]] = []
    group_ledger = hashlib.sha256()
    inpaint_steps = 0
    direct_steps = 0
    best_score = -math.inf
    best_states: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None = None
    for epoch in range(TRAIN_EPOCHS):
        losses: list[float] = []
        inpaint.train()
        direct.train()
        for name in local_train_names:
            for panel_name in PANELS:
                prepared = prepare_source(
                    name, panel_name, "train", data_root=args.data_root, restorer=restorer,
                    embedding=embedding, device=device, denoise_batch_size=args.denoise_batch_size,
                    seed=args.seed,
                )
                if prepared.w4 is None:
                    raise RuntimeError("ranker training requires frozen w4")
                outgoing, incoming = hard_negative_groups(prepared.w4, prepared.slot_to_target)
                if len(outgoing.first) != TRUE_GROUPS or len(incoming.first) != TRUE_GROUPS:
                    raise RuntimeError("ranker full-group workload drift")
                selected = np.arange(TRUE_GROUPS, dtype=np.int32)
                group_ledger.update(f"{epoch}:{name}:{panel_name}".encode("utf-8"))
                group_ledger.update(np.ascontiguousarray(selected, dtype=np.int32).tobytes())
                group_ledger.update(np.ascontiguousarray(outgoing.first).tobytes())
                group_ledger.update(np.ascontiguousarray(outgoing.second).tobytes())
                group_ledger.update(np.ascontiguousarray(incoming.first).tobytes())
                group_ledger.update(np.ascontiguousarray(incoming.second).tobytes())
                raw = tensor_bank(prepared.raw, device)
                denoised = tensor_bank(prepared.denoised, device)
                for start in range(0, TRUE_GROUPS, RANKER_GROUP_BATCH):
                    index = selected[start : start + RANKER_GROUP_BATCH]
                    inpaint_optimizer.zero_grad(set_to_none=True)
                    direct_optimizer.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    for view_index, groups in enumerate((outgoing, incoming)):
                        inpaint_sync = inpaint.no_sync() if context.active and view_index == 0 else nullcontext()
                        direct_sync = direct.no_sync() if context.active and view_index == 0 else nullcontext()
                        with inpaint_sync, direct_sync:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                view_loss = listwise_view_loss(
                                    _group_logits(
                                        inpaint, generator, raw, denoised, groups, index, inpaint=True
                                    )
                                ) + listwise_view_loss(
                                    _group_logits(
                                        direct, generator, raw, denoised, groups, index, inpaint=False
                                    )
                                )
                            ranker_scaler.scale(view_loss).backward()
                            step_loss += float(view_loss.detach())
                    ranker_scaler.unscale_(inpaint_optimizer)
                    ranker_scaler.unscale_(direct_optimizer)
                    torch.nn.utils.clip_grad_norm_(inpaint_core.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(direct_core.parameters(), 1.0)
                    ranker_scaler.step(inpaint_optimizer)
                    ranker_scaler.step(direct_optimizer)
                    ranker_scaler.update()
                    inpaint_steps += 1
                    direct_steps += 1
                    losses.append(step_loss)

        # Calibration A is the only labelled model-selection surface.  The
        # equal w4 blend and all thresholds were frozen before it is opened.
        calibration_records: list[dict[str, Any]] = []
        inpaint.eval()
        direct.eval()
        for name in local_cal_names:
            for panel_name in PANELS:
                prepared = prepare_source(
                    name, panel_name, "calibration_a", data_root=args.data_root,
                    restorer=restorer, embedding=embedding, device=device,
                    denoise_batch_size=args.denoise_batch_size, seed=args.seed,
                )
                inpaint_score, direct_score = dense_scores(
                    generator, inpaint, direct, prepared.raw, prepared.denoised,
                    device=device, query_chunk=4, pair_chunk=DENSE_PAIR_BATCH,
                )
                if prepared.w4 is None:
                    raise RuntimeError("calibration A requires frozen w4")
                blend = blend_with_w4(prepared.w4, inpaint_score)
                calibration_records.append({
                    "index": cal_names.index(name) * len(PANELS) + PANELS.index(panel_name),
                    "name": name,
                    "panel": panel_name,
                    "metrics": {
                        "w4": compact_metrics(prepared.w4, prepared.slot_to_target),
                        "direct": compact_metrics(direct_score, prepared.slot_to_target),
                        "inpaint": compact_metrics(inpaint_score, prepared.slot_to_target),
                        "blend": compact_metrics(blend, prepared.slot_to_target),
                    },
                    "reconstruction": _generator_reconstruction(
                        generator, prepared, device=device, batch_size=GENERATOR_BATCH
                    ),
                })
        calibration_records = merge_indexed_shards(
            gather_objects(calibration_records, context), len(cal_names) * len(PANELS)
        )
        selection_value = (
            float(np.mean([record["metrics"]["blend"]["mrr"] for record in calibration_records]))
            if context.primary
            else 0.0
        )
        selection_tensor = torch.tensor([selection_value], dtype=torch.float64, device=device)
        if context.active:
            dist.broadcast(selection_tensor, src=0)
        selection_score = float(selection_tensor.item())
        mean_ranker_loss = distributed_mean(float(sum(losses)), len(losses), context)
        ranker_epochs.append({
            "epoch": epoch + 1,
            "mean_training_loss": mean_ranker_loss,
            "source_panel_records": len(train_names) * len(PANELS),
            "source_panel_records_per_rank": len(local_train_names) * len(PANELS),
            "groups_per_source_panel": TRUE_GROUPS,
            "optimizer_steps_cumulative_per_arm_per_rank": inpaint_steps,
            "amp_dtype": "float16",
            "grad_scaler": True,
            "calibration_a_blend_mrr": selection_score,
            "calibration_a_records": [
                {key: value for key, value in record.items() if key != "index"}
                for record in calibration_records
            ],
        })
        if selection_score > best_score:
            best_score = selection_score
            best_states = (copy.deepcopy(inpaint_core.state_dict()), copy.deepcopy(direct_core.state_dict()))

    if best_states is None:
        raise RuntimeError("no ranker epoch was selected")
    expected_generator_steps = TRAIN_EPOCHS * len(local_train_names) * len(PANELS) * math.ceil(TRUE_GROUPS / GENERATOR_BATCH)
    expected_ranker_steps = TRAIN_EPOCHS * len(local_train_names) * len(PANELS) * (TRUE_GROUPS // RANKER_GROUP_BATCH)
    if generator_steps != expected_generator_steps:
        raise RuntimeError("full generator optimizer-step workload drift")
    if module_state_sha256(generator_core) != frozen_generator_hash:
        raise RuntimeError("frozen generator changed during ranker training")
    if inpaint_steps != direct_steps or inpaint_steps != expected_ranker_steps:
        raise RuntimeError("direct/inpaint optimizer step ledgers differ")
    inpaint_core.load_state_dict(best_states[0], strict=True)
    direct_core.load_state_dict(best_states[1], strict=True)
    per_rank_final_model_state_sha256 = gather_objects({
        "generator": module_state_sha256(generator_core),
        "inpaint_ranker": module_state_sha256(inpaint_core),
        "direct_ranker": module_state_sha256(direct_core),
    }, context)
    if any(
        value != per_rank_final_model_state_sha256[0]
        for value in per_rank_final_model_state_sha256[1:]
    ):
        raise RuntimeError("DDP final model states differ between ranks")
    synchronized_model_state_sha256 = per_rank_final_model_state_sha256[0]
    rank_group_ledgers = gather_objects(group_ledger.hexdigest(), context)
    shared_group_ledger_sha256 = canonical_hash({
        "world_size": context.world_size,
        "per_rank_sha256": rank_group_ledgers,
    })
    output = Path(args.output_dir)
    if context.primary:
        output.mkdir(parents=True, exist_ok=True)
    if context.active:
        dist.barrier()
    checkpoint = output / "masked_gap_gate.pt"
    payload = state_dict_payload(
        generator_core, inpaint_core, direct_core,
        metadata={
            "safe_for_submission": False,
            "seed": args.seed,
            "selected_calibration_a_blend_mrr": best_score,
            "protocol": audit,
            "restorer": restorer_metadata,
            "embedding": embedding_metadata,
            "shared_upstream_exposure": True,
            "capacity_selection_binding": capacity_binding,
            "capacity_report_sha256": capacity_binding["report_sha256"],
            "capacity_wrapper_report_sha256": capacity_binding["wrapper_report_sha256"],
            "scientific_config": scientific_config,
            "scientific_config_sha256": canonical_hash(scientific_config),
            "ranker_initial_state_sha256": ranker_initial_hash,
            "shared_group_ledger_sha256": shared_group_ledger_sha256,
            "per_rank_group_ledger_sha256": rank_group_ledgers,
            "per_rank_final_model_state_sha256": per_rank_final_model_state_sha256,
            "synchronized_model_state_sha256": synchronized_model_state_sha256,
            "inpaint_optimizer_steps": inpaint_steps,
            "direct_optimizer_steps": direct_steps,
            "optimizer": {
                "generator": "separate AdamW",
                "inpaint_ranker": "separate AdamW",
                "direct_ranker": "separate AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            "training_precision": {"autocast": "float16", "grad_scaler": True},
            "generator_optimizer_steps": generator_steps,
            "scheduler": None,
            "generator_frozen_sha256_before_ranker": frozen_generator_hash,
            "generator_frozen_sha256_after_ranker": module_state_sha256(generator_core),
            "distributed_execution": {
                "kind": "torch_DistributedDataParallel",
                "backend": "nccl",
                "world_size": 2,
                "sources_per_rank": 48,
                "allreduce_every_optimizer_step": True,
                "generator_batch_per_rank": GENERATOR_BATCH,
                "ranker_groups_per_rank": RANKER_GROUP_BATCH,
            },
        },
    )
    if context.primary:
        temporary = checkpoint.with_name(checkpoint.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
    if context.active:
        dist.barrier()
    report = {
        "kind": "masked_gap_training_report_v1",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "protocol": audit,
        "generator_epochs": generator_epochs,
        "ranker_epochs": ranker_epochs,
        "selected_calibration_a_blend_mrr": best_score,
        "capacity_selection_binding": capacity_binding,
        "ranker_training_ledger": {
            "byte_identical_initial_state_sha256": ranker_initial_hash,
            "shared_group_ledger_sha256": shared_group_ledger_sha256,
            "per_rank_group_ledger_sha256": rank_group_ledgers,
            "per_rank_final_model_state_sha256": per_rank_final_model_state_sha256,
            "synchronized_model_state_sha256": synchronized_model_state_sha256,
            "inpaint_optimizer_steps": inpaint_steps,
            "direct_optimizer_steps": direct_steps,
            "optimizer": {
                "generator": "separate AdamW",
                "inpaint_ranker": "separate AdamW",
                "direct_ranker": "separate AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            "training_precision": {"autocast": "float16", "grad_scaler": True},
            "generator_optimizer_steps": generator_steps,
            "scheduler": None,
            "generator_sha256_before_ranker": frozen_generator_hash,
            "generator_sha256_after_ranker": module_state_sha256(generator_core),
        },
        "scientific_contract": {
            "canonical_pair": "20x40",
            "masked_central_columns": 4,
            "ranker_group": "1 positive + 31 frozen-w4 hard negatives",
            "hard_negative_miner": "single production w4 = C1 + HBT(weight 4); no HBT/w4 union",
            "ranker_loss": "outgoing CE + incoming CE + 0.25 BCE",
            "direct_control_equal_capacity": True,
            "blend": "equal row-rank w4 + masked-gap inpaint",
            "qap_run": False,
        },
    }
    if context.primary:
        atomic_json(output / "training_report.json", report)
    if context.active:
        dist.barrier()


def prepare_command(args: argparse.Namespace) -> None:
    if args.split not in {"calibration_b", "holdout"}:
        raise ValueError("prepare split must be calibration_b or holdout")
    audit = protocol_audit(
        manifest=args.manifest,
        quarantine=args.quarantine,
        embedding_checkpoint=args.embedding_checkpoint,
    )
    names = audit["splits"][args.split]["names"]
    secret_seed_mapping_path = Path(args.secret_seed_mapping).resolve(strict=True)
    try:
        secret_seed_mapping_path.relative_to(Path(args.label_dir).resolve())
    except ValueError as error:
        raise RuntimeError("secret panel seed mapping must live inside the label-only tree") from error
    secret_panel_seeds = load_secret_panel_seeds(
        secret_seed_mapping_path,
        split=args.split,
        names=names,
    )
    context = distributed_context(args.device, required=getattr(args, "require_ddp", False))
    device = context.device
    restorer_core, _, restorer_metadata = load_restorer(args.denoiser, device=str(device))
    embedding_core, embedding_metadata = load_embedding_checkpoint(args.embedding_checkpoint, device=device)
    restorer = restorer_core
    embedding = embedding_core
    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    input_records: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []
    for name_index in shard_indices(len(names), context.rank, context.world_size):
        name = names[name_index]
        for panel_name in PANELS:
            prepared = prepare_source(
                name, panel_name, args.split, data_root=args.data_root, restorer=restorer,
                embedding=embedding, device=device, denoise_batch_size=args.denoise_batch_size,
                seed=args.seed, panel_seed=secret_panel_seeds[(name, panel_name)],
            )
            stem = f"{Path(name).stem}__{panel_name}"
            input_path = input_dir / f"{stem}.input.npz"
            label_path = label_dir / f"{stem}.labels.npz"
            if not args.overwrite and (input_path.exists() or label_path.exists()):
                raise FileExistsError(f"fixture already exists: {stem}")
            atomic_npz(
                input_path,
                raw_tiles=prepared.raw,
                denoised_tiles=prepared.denoised,
                w4_right=prepared.w4.right,
                w4_down=prepared.w4.down,
            )
            atomic_npz(
                label_path,
                slot_to_target=prepared.slot_to_target,
                clean_slot_tiles=prepared.clean_slots,
            )
            input_records.append({
                "index": name_index * len(PANELS) + PANELS.index(panel_name),
                "name": name, "panel": panel_name,
                "file": input_path.name, "sha256": sha256(input_path),
            })
            label_records.append({
                "index": name_index * len(PANELS) + PANELS.index(panel_name),
                "name": name, "panel": panel_name, "seed": prepared.seed,
                "file": label_path.name, "sha256": sha256(label_path),
            })
    if context.active:
        dist.barrier()
    input_records = merge_indexed_shards(
        gather_objects(input_records, context), len(names) * len(PANELS)
    )
    label_records = merge_indexed_shards(
        gather_objects(label_records, context), len(names) * len(PANELS)
    )
    input_records = [
        {key: value for key, value in record.items() if key != "index"}
        for record in input_records
    ]
    label_records = [
        {key: value for key, value in record.items() if key != "index"}
        for record in label_records
    ]
    input_manifest = {
        "kind": INPUT_MANIFEST_KIND, "split": args.split, "names": names,
        "names_sha256": names_sha256(names), "panels": list(PANELS),
        "records": input_records,
        "allowed_npz_keys": sorted(INPUT_KEYS),
        "target_or_label_fields_attached": False,
        "panel_seed_attached": False,
        "panel_seed_derivation_available": False,
        "upstream_assets": {"restorer": restorer_metadata, "embedding": embedding_metadata},
        "upstream_asset_sha256": {
            "restorer": sha256(Path(args.denoiser).resolve(strict=True)),
            "embedding": sha256(Path(args.embedding_checkpoint).resolve(strict=True)),
        },
    }
    label_manifest = {
        "kind": LABEL_MANIFEST_KIND, "split": args.split, "names": names,
        "names_sha256": names_sha256(names), "panels": list(PANELS),
        "records": label_records, "allowed_npz_keys": sorted(LABEL_KEYS),
    }
    if context.primary:
        atomic_json(input_dir / "input_manifest.json", input_manifest)
        atomic_json(label_dir / "label_manifest.json", label_manifest)
    if context.active:
        dist.barrier()


def _verify_manifest_records(manifest_path: Path, expected_kind: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    if manifest.get("kind") != expected_kind:
        raise RuntimeError("fixture manifest kind mismatch")
    split = manifest.get("split")
    if split not in {"calibration_b", "holdout"}:
        raise RuntimeError("fixture split mismatch")
    frozen = FROZEN_SPLITS[str(split)]
    names = manifest.get("names")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) for name in names)
        or names_sha256(names) != manifest.get("names_sha256")
        or manifest.get("names_sha256") != frozen[3]
        or len(names) != frozen[2]
    ):
        raise RuntimeError("fixture frozen names/hash mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != frozen[2] * len(PANELS):
        raise RuntimeError("fixture record count mismatch")
    expected_order = [(name, panel) for name in manifest["names"] for panel in PANELS]
    actual_order = [(record.get("name"), record.get("panel")) for record in records]
    if actual_order != expected_order:
        raise RuntimeError("fixture record order mismatch")
    expected_record_keys = (
        {"name", "panel", "file", "sha256"}
        if expected_kind == INPUT_MANIFEST_KIND
        else {"name", "panel", "seed", "file", "sha256"}
    )
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise RuntimeError("fixture record schema mismatch")
        path = manifest_path.parent / str(record["file"])
        if sha256(path) != record.get("sha256"):
            raise RuntimeError(f"fixture hash mismatch: {path.name}")
    return manifest, records


def phase_a_command(args: argparse.Namespace) -> None:
    manifest_path = Path(args.input_manifest).resolve(strict=True)
    manifest, records = _verify_manifest_records(manifest_path, INPUT_MANIFEST_KIND)
    if (
        manifest.get("allowed_npz_keys") != sorted(INPUT_KEYS)
        or manifest.get("target_or_label_fields_attached") is not False
        or manifest.get("panel_seed_attached") is not False
        or manifest.get("panel_seed_derivation_available") is not False
    ):
        raise RuntimeError("input-only manifest contract mismatch")
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    generator_core, inpaint_core, direct_core, metadata = load_models(payload)
    if metadata.get("safe_for_submission") is not False:
        raise RuntimeError("masked-gap checkpoint fail-closed metadata mismatch")
    context = distributed_context(args.device, required=getattr(args, "require_ddp", False))
    device = context.device
    generator_core.to(device).eval()
    inpaint_core.to(device).eval()
    direct_core.to(device).eval()
    generator_hash = module_state_sha256(generator_core)
    if metadata.get("generator_frozen_sha256_after_ranker") != generator_hash:
        raise RuntimeError("checkpoint generator hash does not match frozen training ledger")
    if metadata.get("scientific_config_sha256") != canonical_hash(metadata.get("scientific_config")):
        raise RuntimeError("checkpoint scientific config hash mismatch")
    training_ledger = {
        key: metadata.get(key)
        for key in (
            "ranker_initial_state_sha256", "shared_group_ledger_sha256",
            "inpaint_optimizer_steps", "direct_optimizer_steps", "optimizer", "scheduler",
            "generator_optimizer_steps", "training_precision", "capacity_selection_binding",
            "per_rank_group_ledger_sha256", "distributed_execution",
            "per_rank_final_model_state_sha256", "synchronized_model_state_sha256",
            "generator_frozen_sha256_before_ranker", "generator_frozen_sha256_after_ranker",
        )
    }
    if (
        not isinstance(training_ledger["ranker_initial_state_sha256"], str)
        or len(training_ledger["ranker_initial_state_sha256"]) != 64
        or not isinstance(training_ledger["shared_group_ledger_sha256"], str)
        or len(training_ledger["shared_group_ledger_sha256"]) != 64
        or training_ledger["inpaint_optimizer_steps"] != training_ledger["direct_optimizer_steps"]
        or not isinstance(training_ledger["inpaint_optimizer_steps"], int)
        or training_ledger["inpaint_optimizer_steps"] <= 0
        or not isinstance(training_ledger["generator_optimizer_steps"], int)
        or training_ledger["generator_optimizer_steps"] <= 0
        or training_ledger["training_precision"] != {"autocast": "float16", "grad_scaler": True}
        or not isinstance(training_ledger["capacity_selection_binding"], dict)
        or not isinstance(training_ledger["per_rank_group_ledger_sha256"], list)
        or len(training_ledger["per_rank_group_ledger_sha256"]) != 2
        or not isinstance(training_ledger["per_rank_final_model_state_sha256"], list)
        or len(training_ledger["per_rank_final_model_state_sha256"]) != 2
        or not isinstance(training_ledger["synchronized_model_state_sha256"], dict)
        or any(
            value != training_ledger["synchronized_model_state_sha256"]
            for value in training_ledger["per_rank_final_model_state_sha256"]
        )
        or training_ledger["synchronized_model_state_sha256"].get("generator") != generator_hash
        or training_ledger["distributed_execution"] != {
            "kind": "torch_DistributedDataParallel",
            "backend": "nccl",
            "world_size": 2,
            "sources_per_rank": 48,
            "allreduce_every_optimizer_step": True,
            "generator_batch_per_rank": GENERATOR_BATCH,
            "ranker_groups_per_rank": RANKER_GROUP_BATCH,
        }
        or training_ledger["scheduler"] is not None
        or training_ledger["generator_frozen_sha256_before_ranker"] != generator_hash
        or training_ledger["generator_frozen_sha256_after_ranker"] != generator_hash
    ):
        raise RuntimeError("checkpoint direct/inpaint training ledger is not fail-closed")
    generator = generator_core
    inpaint = inpaint_core
    direct = direct_core
    arrays: dict[str, np.ndarray] = {}
    matrix_manifest: dict[str, dict[str, Any]] = {}
    record_reports: list[dict[str, Any]] = []
    for index in shard_indices(len(records), context.rank, context.world_size):
        record = records[index]
        path = manifest_path.parent / record["file"]
        with np.load(path, allow_pickle=False) as fixture:
            if set(fixture.files) != INPUT_KEYS:
                raise RuntimeError("Phase A input fixture contains non-input fields")
            raw = fixture["raw_tiles"]
            denoised = fixture["denoised_tiles"]
            w4 = CompatibilityMatrices("frozen_w4", fixture["w4_right"], fixture["w4_down"])
        inpaint_score, direct_score = dense_scores(
            generator, inpaint, direct, raw, denoised, device=device,
            query_chunk=4, pair_chunk=DENSE_PAIR_BATCH,
        )
        candidate = blend_with_w4(
            w4, inpaint_score, name="frozen_w4_masked_gap_candidate"
        )
        direct_control = blend_with_w4(
            w4, direct_score, name="frozen_w4_direct_control"
        )
        prefix = f"r{index:02d}"
        for method, score in (
            ("w4", w4),
            ("direct_only", direct_score),
            ("inpaint_only", inpaint_score),
            ("direct_control", direct_control),
            ("candidate", candidate),
        ):
            arrays[f"{prefix}_{method}_right"] = score.right
            arrays[f"{prefix}_{method}_down"] = score.down
            matrix_manifest[f"{prefix}_{method}_right"] = matrix_fingerprint(score.right)
            matrix_manifest[f"{prefix}_{method}_down"] = matrix_fingerprint(score.down)
        record_reports.append({
            "index": index, "name": record["name"], "panel": record["panel"],
            "input_sha256": record["sha256"],
        })
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if context.active:
        shard_artifact = output_dir / f"phase_a_scores.rank{context.rank}.npz"
        atomic_npz(shard_artifact, **arrays)
        shard_metadata = {
            "rank": context.rank,
            "artifact": shard_artifact.name,
            "artifact_sha256": sha256(shard_artifact),
            "matrix_manifest": matrix_manifest,
            "records": record_reports,
        }
        shards = gather_objects(shard_metadata, context)
        dist.barrier()
        if not context.primary:
            dist.barrier()
            return
        if sorted(int(shard.get("rank", -1)) for shard in shards) != [0, 1]:
            raise RuntimeError("Phase A rank shard identities are not exact")
        arrays = {}
        matrix_manifest = {}
        record_reports = merge_indexed_shards(
            [shard["records"] for shard in shards], len(records)
        )
        for shard in shards:
            shard_path = output_dir / str(shard["artifact"])
            if sha256(shard_path) != shard["artifact_sha256"]:
                raise RuntimeError("Phase A rank shard artifact hash mismatch")
            shard_manifest = shard["matrix_manifest"]
            with np.load(shard_path, allow_pickle=False) as values:
                if set(values.files) != set(shard_manifest):
                    raise RuntimeError("Phase A rank shard key mismatch")
                for key in values.files:
                    if key in arrays or key in matrix_manifest:
                        raise RuntimeError(f"duplicate Phase A rank shard matrix: {key}")
                    value = values[key]
                    if matrix_fingerprint(value) != shard_manifest[key]:
                        raise RuntimeError(f"Phase A rank shard fingerprint mismatch: {key}")
                    arrays[key] = value
            matrix_manifest.update(shard_manifest)
        if len(arrays) != len(records) * 10 or set(arrays) != set(matrix_manifest):
            raise RuntimeError("Phase A rank shard matrix coverage is not exact")
    artifact = output_dir / "phase_a_scores.npz"
    atomic_npz(artifact, **arrays)
    report = {
        "kind": PHASE_A_KIND,
        "split": manifest["split"],
        "input_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "artifact": artifact.name,
        "artifact_sha256": sha256(artifact),
        "records": record_reports,
        "target_metrics_opened": False,
        "labels_or_targets_loaded": False,
        "dense_candidates_per_query": TILE_COUNT - 1,
        "qap_run": False,
        "source_names_sha256": manifest["names_sha256"],
        "panels": list(PANELS),
        "upstream_asset_sha256": manifest.get("upstream_asset_sha256"),
        "evaluator_code_sha256": sha256(Path(__file__)),
        "core_code_sha256": sha256(REPO_ROOT / "src/puzzle_assembly/masked_gap.py"),
        "scientific_config": metadata.get("scientific_config"),
        "scientific_config_sha256": metadata.get("scientific_config_sha256"),
        "generator_state_sha256": generator_hash,
        "training_ledger": training_ledger,
        "training_ledger_sha256": canonical_hash(training_ledger),
        "matrix_manifest": matrix_manifest,
    }
    atomic_json(output_dir / "phase_a_report.json", report)
    if context.active:
        dist.barrier()


def authorize_command(args: argparse.Namespace) -> None:
    report_path = Path(args.phase_a_report).resolve(strict=True)
    artifact_path = Path(args.phase_a_artifact).resolve(strict=True)
    report = read_json(report_path)
    if report.get("kind") != PHASE_A_KIND or report.get("target_metrics_opened") is not False:
        raise RuntimeError("Phase A report is not sealed target-blind output")
    if report.get("labels_or_targets_loaded") is not False or report.get("qap_run") is not False:
        raise RuntimeError("Phase A separation contract failed")
    if report.get("artifact_sha256") != sha256(artifact_path):
        raise RuntimeError("Phase A artifact hash mismatch")
    if report.get("evaluator_code_sha256") != sha256(Path(__file__)):
        raise RuntimeError("Phase A evaluator code hash mismatch")
    if report.get("core_code_sha256") != sha256(REPO_ROOT / "src/puzzle_assembly/masked_gap.py"):
        raise RuntimeError("Phase A core code hash mismatch")
    if report.get("panels") != list(PANELS):
        raise RuntimeError("Phase A panel contract mismatch")
    matrix_manifest = report.get("matrix_manifest")
    if not isinstance(matrix_manifest, dict) or len(matrix_manifest) != len(report["records"]) * 10:
        raise RuntimeError("Phase A matrix manifest count mismatch")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        if set(artifact.files) != set(matrix_manifest):
            raise RuntimeError("Phase A artifact matrix key set mismatch")
        for key, expected_fingerprint in matrix_manifest.items():
            if matrix_fingerprint(artifact[key]) != expected_fingerprint:
                raise RuntimeError(f"Phase A matrix fingerprint mismatch: {key}")
    required_bound_fields = (
        "source_names_sha256", "upstream_asset_sha256", "scientific_config_sha256",
        "generator_state_sha256", "checkpoint_sha256", "input_manifest_sha256",
        "training_ledger_sha256",
    )
    if any(report.get(key) in (None, "", {}) for key in required_bound_fields):
        raise RuntimeError("Phase A report is missing a hash-bound protocol field")
    authorization = {
        "kind": AUTH_KIND,
        "split": report["split"],
        "phase_a_report_sha256": sha256(report_path),
        "phase_a_artifact_sha256": sha256(artifact_path),
        "input_manifest_sha256": report["input_manifest_sha256"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "authorized_record_count": len(report["records"]),
        "phase_b_authorized": True,
        "source_names_sha256": report["source_names_sha256"],
        "panels": report["panels"],
        "upstream_asset_sha256": report["upstream_asset_sha256"],
        "evaluator_code_sha256": report["evaluator_code_sha256"],
        "core_code_sha256": report["core_code_sha256"],
        "scientific_config_sha256": report["scientific_config_sha256"],
        "generator_state_sha256": report["generator_state_sha256"],
        "training_ledger_sha256": report["training_ledger_sha256"],
        "matrix_manifest_sha256": canonical_hash(matrix_manifest),
    }
    atomic_json(args.output, authorization)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    methods = ("w4", "direct_only", "inpaint_only", "direct_control", "candidate")
    metrics = ("mrr", "recall_at_1", "recall_at_5")

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            method: {
                metric: float(np.mean([record["metrics"][method][metric] for record in selected]))
                for metric in metrics
            }
            for method in methods
        }

    return {
        "macro": summarize(records),
        "panels": {panel: summarize([record for record in records if record["panel"] == panel]) for panel in PANELS},
    }


def gate_decision(records: list[dict[str, Any]], *, final_holdout: bool) -> dict[str, Any]:
    expected_records = 16 if final_holdout else 8
    if len(records) != expected_records:
        raise RuntimeError(f"the frozen gate requires exactly {expected_records} source-panel records")
    summary = _aggregate(records)
    macro = summary["macro"]
    blend_delta = {
        metric: macro["candidate"][metric] - macro["w4"][metric]
        for metric in ("mrr", "recall_at_1", "recall_at_5")
    }
    panel_blend_delta = {
        panel: {
            metric: summary["panels"][panel]["candidate"][metric] - summary["panels"][panel]["w4"][metric]
            for metric in blend_delta
        }
        for panel in PANELS
    }
    direct_delta = macro["candidate"]["mrr"] - macro["direct_control"]["mrr"]
    panel_direct_delta = {
        panel: summary["panels"][panel]["candidate"]["mrr"] - summary["panels"][panel]["direct_control"]["mrr"]
        for panel in PANELS
    }
    reconstruction = {
        panel: {
            key: float(np.mean([record["reconstruction"][key] for record in records if record["panel"] == panel]))
            for key in (
                "generator_charbonnier", "copy_charbonnier", "interpolation_charbonnier",
                "generator_mae", "copy_mae", "interpolation_mae",
            )
        }
        for panel in PANELS
    }
    names = sorted({record["name"] for record in records})
    expected_sources = 8 if final_holdout else 4
    if len(names) != expected_sources or any(
        sorted(record["panel"] for record in records if record["name"] == name) != sorted(PANELS)
        for name in names
    ):
        raise RuntimeError("each frozen source must have exactly the two declared panels")
    source_wins = int(sum(
        np.mean([
            record["metrics"]["candidate"]["mrr"]
            for record in records if record["name"] == name
        ])
        > np.mean([
            record["metrics"]["w4"]["mrr"]
            for record in records if record["name"] == name
        ])
        for name in names
    ))
    conditions = {
        "reconstruction_charbonnier_and_mae_5pct_better_each_control_each_panel": all(
            values[f"generator_{metric}"] <= 0.95 * values[f"{baseline}_{metric}"] + 1e-12
            for values in reconstruction.values()
            for metric in ("charbonnier", "mae")
            for baseline in ("copy", "interpolation")
        ),
        "candidate_mrr_ge_equal_rank_direct_control_plus_0.005": direct_delta + 1e-12 >= 0.005,
        "blend_vs_w4_mrr_delta_ge_0.015": blend_delta["mrr"] + 1e-12 >= 0.015,
        "blend_vs_w4_recall_at_1_delta_ge_0.010": blend_delta["recall_at_1"] + 1e-12 >= 0.010,
        "blend_vs_w4_recall_at_5_delta_ge_0.020": blend_delta["recall_at_5"] + 1e-12 >= 0.020,
        "all_retrieval_deltas_nonnegative_per_panel": (
            all(value >= -1e-12 for panel in panel_blend_delta.values() for value in panel.values())
        ),
    }
    if final_holdout:
        conditions["source_mean_over_two_panels_mrr_wins_ge_6_of_8"] = source_wins >= 6
    return {
        "summary": summary,
        "reconstruction": reconstruction,
        "candidate_vs_equal_rank_direct_control_mrr_delta": direct_delta,
        "panel_candidate_vs_equal_rank_direct_control_mrr_delta": panel_direct_delta,
        "blend_vs_w4_deltas": blend_delta,
        "panel_blend_vs_w4_deltas": panel_blend_delta,
        "source_mean_over_two_panels_mrr_wins": int(source_wins),
        "final_holdout": bool(final_holdout),
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def phase_b_command(args: argparse.Namespace) -> None:
    input_manifest_path = Path(args.input_manifest).resolve(strict=True)
    # Do not resolve/stat/open the label manifest path before every input-only
    # authorization, artifact, checkpoint, code, and source binding passes.
    unresolved_label_manifest_path = Path(args.label_manifest)
    phase_a_report_path = Path(args.phase_a_report).resolve(strict=True)
    artifact_path = Path(args.phase_a_artifact).resolve(strict=True)
    authorization_path = Path(args.authorization).resolve(strict=True)
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    input_manifest, input_records = _verify_manifest_records(input_manifest_path, INPUT_MANIFEST_KIND)
    report = read_json(phase_a_report_path)
    authorization = read_json(authorization_path)
    if authorization.get("kind") != AUTH_KIND or authorization.get("phase_b_authorized") is not True:
        raise RuntimeError("missing global Phase B authorization")
    if (
        report.get("kind") != PHASE_A_KIND
        or report.get("split") != input_manifest.get("split")
        or report.get("target_metrics_opened") is not False
        or report.get("labels_or_targets_loaded") is not False
        or report.get("qap_run") is not False
    ):
        raise RuntimeError("Phase A report is not sealed target-blind output")
    expected = {
        "phase_a_report_sha256": sha256(phase_a_report_path),
        "phase_a_artifact_sha256": sha256(artifact_path),
        "input_manifest_sha256": sha256(input_manifest_path),
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise RuntimeError("global Phase B authorization hash mismatch")
    if report.get("artifact_sha256") != expected["phase_a_artifact_sha256"]:
        raise RuntimeError("Phase A report/artifact mismatch")
    if (
        report.get("input_manifest_sha256") != expected["input_manifest_sha256"]
        or report.get("checkpoint_sha256") != expected["checkpoint_sha256"]
        or len(report.get("records", [])) != len(input_records)
    ):
        raise RuntimeError("Phase A report input/checkpoint/record binding mismatch")
    bound_checks = {
        "source_names_sha256": input_manifest["names_sha256"],
        "panels": list(PANELS),
        "upstream_asset_sha256": input_manifest.get("upstream_asset_sha256"),
        "evaluator_code_sha256": sha256(Path(__file__)),
        "core_code_sha256": sha256(REPO_ROOT / "src/puzzle_assembly/masked_gap.py"),
        "matrix_manifest_sha256": canonical_hash(report.get("matrix_manifest")),
        "training_ledger_sha256": report.get("training_ledger_sha256"),
    }
    if any(authorization.get(key) != value for key, value in bound_checks.items()):
        raise RuntimeError("authorization protocol/code/matrix binding mismatch")
    if authorization.get("authorized_record_count") != len(input_records):
        raise RuntimeError("authorization record count mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    generator_core, _, _, checkpoint_metadata = load_models(payload)
    context = distributed_context(args.device, required=getattr(args, "require_ddp", False))
    device = context.device
    generator_core.to(device).eval()
    if module_state_sha256(generator_core) != authorization.get("generator_state_sha256"):
        raise RuntimeError("authorization generator state hash mismatch")
    if checkpoint_metadata.get("scientific_config_sha256") != authorization.get("scientific_config_sha256"):
        raise RuntimeError("authorization scientific config hash mismatch")
    generator = generator_core
    with np.load(artifact_path, allow_pickle=False) as scores:
        matrix_manifest = report.get("matrix_manifest")
        if not isinstance(matrix_manifest, dict) or set(scores.files) != set(matrix_manifest):
            raise RuntimeError("Phase A artifact matrix key binding mismatch")
        for key in scores.files:
            if matrix_fingerprint(scores[key]) != matrix_manifest[key]:
                raise RuntimeError(f"Phase A artifact matrix fingerprint mismatch: {key}")
        score_arrays = {key: scores[key] for key in scores.files}
    calibration_b_report_sha256: str | None = None
    if input_manifest["split"] == "calibration_b":
        if args.calibration_b_report is not None:
            raise RuntimeError("calibration-B Phase B must not consume a prior calibration-B report")
    else:
        if args.calibration_b_report is None:
            raise RuntimeError("holdout labels remain sealed until a passing calibration-B report is supplied")
        calibration_b_path = Path(args.calibration_b_report).resolve(strict=True)
        calibration_b_report = read_json(calibration_b_path)
        if (
            calibration_b_report.get("kind") != PHASE_B_KIND
            or calibration_b_report.get("split") != "calibration_b"
            or calibration_b_report.get("decision", {}).get("passed") is not True
        ):
            raise RuntimeError("holdout requires a passing frozen calibration-B gate")
        calibration_b_report_sha256 = sha256(calibration_b_path)

    # This is the first label-manifest access in the function.
    label_manifest_path = unresolved_label_manifest_path.resolve(strict=True)
    label_manifest, label_records = _verify_manifest_records(label_manifest_path, LABEL_MANIFEST_KIND)
    if label_manifest.get("allowed_npz_keys") != sorted(LABEL_KEYS):
        raise RuntimeError("label manifest contract mismatch")
    if input_manifest["split"] != label_manifest["split"] or input_manifest["names_sha256"] != label_manifest["names_sha256"]:
        raise RuntimeError("input/label fixture split mismatch")
    label_seeds = [record.get("seed") for record in label_records]
    if (
        any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64 for seed in label_seeds)
        or len(set(label_seeds)) != len(label_seeds)
    ):
        raise RuntimeError("label manifest secret panel seeds are not unique uint64 values")
    records: list[dict[str, Any]] = []
    for index in shard_indices(len(input_records), context.rank, context.world_size):
        input_record = input_records[index]
        label_record = label_records[index]
        if (input_record["name"], input_record["panel"]) != (
            label_record["name"], label_record["panel"]
        ):
            raise RuntimeError("input/label record identity mismatch")
        input_path = input_manifest_path.parent / input_record["file"]
        label_path = label_manifest_path.parent / label_record["file"]
        with np.load(input_path, allow_pickle=False) as fixture:
            if set(fixture.files) != INPUT_KEYS:
                raise RuntimeError("input fixture contract drift in Phase B")
            raw = fixture["raw_tiles"]
            denoised = fixture["denoised_tiles"]
            w4 = CompatibilityMatrices("w4", fixture["w4_right"], fixture["w4_down"])
        with np.load(label_path, allow_pickle=False) as labels:
            if set(labels.files) != LABEL_KEYS:
                raise RuntimeError("label fixture contract drift")
            truth = labels["slot_to_target"]
            clean_slots = labels["clean_slot_tiles"]
        prefix = f"r{index:02d}"
        methods = {
            "w4": w4,
            **{
                method: CompatibilityMatrices(
                    method,
                    score_arrays[f"{prefix}_{method}_right"],
                    score_arrays[f"{prefix}_{method}_down"],
                )
                for method in ("direct_only", "inpaint_only", "direct_control", "candidate")
            },
        }
        prepared = Prepared(
            name=input_record["name"], panel=input_record["panel"], seed=label_record["seed"],
            raw=raw, denoised=denoised, clean_slots=clean_slots, slot_to_target=truth, w4=w4,
        )
        records.append({
            "index": index,
            "name": prepared.name,
            "panel": prepared.panel,
            "metrics": {method: compact_metrics(score, truth) for method, score in methods.items()},
            "reconstruction": _generator_reconstruction(
                generator, prepared, device=device, batch_size=GENERATOR_BATCH
            ),
        })
    records = merge_indexed_shards(
        gather_objects(records, context), len(input_records)
    )
    records = [
        {key: value for key, value in record.items() if key != "index"}
        for record in records
    ]
    if context.active and not context.primary:
        dist.barrier()
        return
    if input_manifest["split"] == "calibration_b":
        decision = gate_decision(records, final_holdout=False)
    else:
        decision = gate_decision(records, final_holdout=True)
    output = {
        "kind": PHASE_B_KIND,
        "split": input_manifest["split"],
        "authorization_sha256": sha256(authorization_path),
        "input_manifest_sha256": sha256(input_manifest_path),
        "label_manifest_sha256": sha256(label_manifest_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "phase_a_report_sha256": sha256(phase_a_report_path),
        "phase_a_artifact_sha256": sha256(artifact_path),
        "records": records,
        "decision": decision,
        "qap_run": False,
        "calibration_b_report_sha256": calibration_b_report_sha256,
        "upstream_exposure_disclosure": (
            "Shared frozen TileNAF may have seen edge_development; results are incremental downstream validation."
        ),
    }
    atomic_json(args.output, output)
    if context.active:
        dist.barrier()


def common_assets(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt")
    parser.add_argument(
        "--embedding-checkpoint",
        default="runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt",
    )
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--quarantine", default="configs/denoise_validation_quarantine_v1.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-protocol")
    audit.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    audit.add_argument("--quarantine", default="configs/denoise_validation_quarantine_v1.json")
    audit.add_argument(
        "--embedding-checkpoint",
        default="runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt",
    )

    train = subparsers.add_parser("train")
    common_assets(train)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--capacity-report", required=True)
    train.add_argument("--capacity-wrapper-report", required=True)
    train.add_argument("--capacity-report-sha256", required=True)
    train.add_argument("--capacity-wrapper-report-sha256", required=True)

    prepare = subparsers.add_parser("prepare")
    common_assets(prepare)
    prepare.add_argument("--split", required=True, choices=("calibration_b", "holdout"))
    prepare.add_argument("--input-dir", required=True)
    prepare.add_argument("--label-dir", required=True)
    prepare.add_argument("--secret-seed-mapping", required=True)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--require-ddp", action="store_true")

    phase_a = subparsers.add_parser("phase-a")
    phase_a.add_argument("--input-manifest", required=True)
    phase_a.add_argument("--checkpoint", required=True)
    phase_a.add_argument("--output-dir", required=True)
    phase_a.add_argument("--device", default="cuda")
    phase_a.add_argument("--require-ddp", action="store_true")

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--phase-a-report", required=True)
    authorize.add_argument("--phase-a-artifact", required=True)
    authorize.add_argument("--output", required=True)

    phase_b = subparsers.add_parser("phase-b")
    phase_b.add_argument("--input-manifest", required=True)
    phase_b.add_argument("--label-manifest", required=True)
    phase_b.add_argument("--phase-a-report", required=True)
    phase_b.add_argument("--phase-a-artifact", required=True)
    phase_b.add_argument("--authorization", required=True)
    phase_b.add_argument("--checkpoint", required=True)
    phase_b.add_argument("--output", required=True)
    phase_b.add_argument("--calibration-b-report")
    phase_b.add_argument("--device", default="cuda")
    phase_b.add_argument("--require-ddp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit-protocol":
        print(json.dumps(protocol_audit(
            manifest=args.manifest,
            quarantine=args.quarantine,
            embedding_checkpoint=args.embedding_checkpoint,
        ), indent=2))
    elif args.command == "train":
        train_command(args)
    elif args.command == "prepare":
        prepare_command(args)
    elif args.command == "phase-a":
        phase_a_command(args)
    elif args.command == "authorize":
        authorize_command(args)
    elif args.command == "phase-b":
        phase_b_command(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
