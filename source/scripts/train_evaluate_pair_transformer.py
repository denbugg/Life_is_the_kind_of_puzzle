#!/usr/bin/env python3
"""Train and gate the sparse full-tile pair Transformer on Kaggle 2xT4.

This is a research pilot, never a submission builder.  It deliberately uses a
coarse-to-hard-negative pipeline: HBT proposes a small candidate graph, while a
large joint Transformer sees complete raw+denoised tile pairs and explicit
touching bands.  Evaluation is source-disjoint, uses two corruption engines
and repeat seeds, and compares both the high-adjacency pure-HBT path and the
higher-SSIM promoted C1/HBT-w4 path.

Recommended Kaggle invocation (two GPUs):

  torchrun --standalone --nproc_per_node=2 \
    scripts/train_evaluate_pair_transformer.py --action pilot --output-dir /kaggle/working/pair_v1

The bounded defaults use 512 whole training sources, three epochs, fp16, DDP,
and gradient checkpointing.  Increase to 1024 sources only after the pilot
passes the strict continuation gate.
"""

from __future__ import annotations

import argparse
import copy
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
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
    prediction_compatibility,
)
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.geometry import GRID, TILE_COUNT, true_neighbour_slots
from puzzle_assembly.learned import (
    DirectionLabels,
    direction_labels,
    learned_compatibility,
    load_embedding_checkpoint,
)
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics, retrieval_metrics
from puzzle_assembly.pair_transformer import (
    DOWN,
    RIGHT,
    PairCandidates,
    PairTransformerScorer,
    fit_binary_temperature,
    load_pair_transformer_checkpoint,
    load_pair_transformer_checkpoint_payload,
    multistage_candidates,
    pair_transformer_compatibility,
    save_pair_transformer_checkpoint,
    score_pairs,
)
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import split_tiles_numpy


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_HBT = (
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/"
    "hbt_d320_denoised_rgb_sobel.pt"
)
DEFAULT_PSEUDO = "runs/denoise_v2/real_gold_train_512.npz"
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"
LATEST_CHECKPOINT = "pair_transformer_latest.pt"
HASHES_NAME = "SHA256SUMS.txt"
PROMOTED_QAP_ITERATIONS = 25
PROMOTED_QAP_RESTARTS = 2
CODE_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parents[1] / "src/puzzle_assembly/__init__.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_assembly/pair_transformer.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_assembly/compatibility.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_assembly/components.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_assembly/geometry.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_assembly/learned.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_assembly/metrics.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_assembly/panels.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_assembly/protocol.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_assembly/qap.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_assembly/solvers.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_denoise_v2/__init__.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_denoise_v2/degradation.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_denoise_v2/inference.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_denoise_v2/losses.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_denoise_v2/metrics.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_denoise_v2/model.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_denoise_v2/tiles.py",
    Path(__file__).resolve().parents[1] / "src/puzzle_denoise_v2/training.py",
)
RESUME_ARGUMENTS = (
    "data_root",
    "denoiser",
    "hbt_checkpoint",
    "pseudo_gold",
    "manifest",
    "quarantine",
    "seed",
    "train_sources",
    "epochs",
    "quick_val_sources",
    "queries_per_source",
    "negatives",
    "groups_per_step",
    "hbt_negative_fraction",
    "visual_negative_fraction",
    "pseudo_confidence",
    "pseudo_every",
    "pseudo_weight",
    "model_dim",
    "layers",
    "heads",
    "feedforward_dim",
    "cnn_channels",
    "patch_grid",
    "side_band",
    "band_tokens",
    "dropout",
    "no_gradient_checkpointing",
    "learning_rate",
    "weight_decay",
    "warmup_fraction",
    "grad_clip",
    "bce_weight",
    "no_amp",
    "amp_init_scale",
    "max_amp_skips",
    "affine_probability",
    "extra_noise_probability",
    "extra_noise_sigma",
    "extrapolation_probability",
    "extrapolation_sigma",
    "blur_probability",
    "jpeg_probability",
    "erosion_probability",
    "max_erosion",
    "view_dropout",
    "panels",
    "denoise_batch_size",
    "candidate_top_k",
    "pair_batch_size",
    "neural_blend",
)


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
class PositiveEdges:
    first: np.ndarray
    second: np.ndarray
    direction: np.ndarray
    weight: np.ndarray

    def __post_init__(self) -> None:
        arrays = [np.asarray(value) for value in (self.first, self.second, self.direction, self.weight)]
        if any(value.ndim != 1 for value in arrays) or len({len(value) for value in arrays}) != 1:
            raise ValueError("positive-edge arrays must be equally sized vectors")
        if not len(self.first):
            raise ValueError("positive-edge collection is empty")


@dataclass(frozen=True)
class TrainingGroups:
    first: np.ndarray
    second: np.ndarray
    direction: np.ndarray
    weight: np.ndarray

    @property
    def group_count(self) -> int:
        return int(self.first.shape[0])

    @property
    def group_size(self) -> int:
        return int(self.first.shape[1])


@dataclass(frozen=True)
class PseudoSource:
    name: str
    slots: np.ndarray
    positions: np.ndarray
    confidence: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["pilot", "train", "evaluate"], default="pilot")
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--hbt-checkpoint", default=DEFAULT_HBT)
    parser.add_argument("--pseudo-gold", default=DEFAULT_PSEUDO)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Exact epoch-boundary checkpoint with optimizer/scaler/scheduler/RNG state",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260711)

    # Bounded pilot: meaningful but small enough for one Kaggle 2xT4 session.
    parser.add_argument("--train-sources", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--quick-val-sources", type=int, default=2)
    parser.add_argument("--calibration-sources", type=int, default=4)
    parser.add_argument("--validation-sources", type=int, default=8)
    parser.add_argument("--validation-replicas", type=int, default=2)
    parser.add_argument("--solver-sources", type=int, default=4)
    parser.add_argument("--panels", default="primary_kornia,independent_libjpeg")
    parser.add_argument("--queries-per-source", type=int, default=48)
    parser.add_argument("--negatives", type=int, default=31)
    parser.add_argument("--groups-per-step", type=int, default=4)
    parser.add_argument("--hbt-negative-fraction", type=float, default=0.65)
    parser.add_argument("--visual-negative-fraction", type=float, default=0.25)
    parser.add_argument("--pseudo-confidence", type=float, default=1.5)
    parser.add_argument("--pseudo-every", type=int, default=4)
    parser.add_argument("--pseudo-weight", type=float, default=0.20)

    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=2048)
    parser.add_argument("--cnn-channels", type=int, default=128)
    parser.add_argument("--patch-grid", type=int, default=5)
    parser.add_argument("--side-band", type=int, default=6)
    parser.add_argument("--band-tokens", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.25)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
    parser.add_argument("--max-amp-skips", type=int, default=4)

    # Task-faithful augmentation.  Panels already contain sigma 40-55, blur,
    # JPEG and affine corruption; these ranges add interpolation/extrapolation.
    parser.add_argument("--affine-probability", type=float, default=0.80)
    parser.add_argument("--extra-noise-probability", type=float, default=0.60)
    parser.add_argument("--extra-noise-sigma", type=float, default=12.0)
    parser.add_argument("--extrapolation-probability", type=float, default=0.15)
    parser.add_argument("--extrapolation-sigma", type=float, default=28.0)
    parser.add_argument("--blur-probability", type=float, default=0.30)
    parser.add_argument("--jpeg-probability", type=float, default=0.35)
    parser.add_argument("--erosion-probability", type=float, default=0.20)
    parser.add_argument("--max-erosion", type=int, default=3)
    parser.add_argument("--view-dropout", type=float, default=0.10)

    parser.add_argument("--candidate-top-k", type=int, default=48)
    parser.add_argument("--candidate-reverse-top-k", type=int, default=8)
    parser.add_argument("--pair-batch-size", type=int, default=512)
    parser.add_argument("--neural-blend", type=float, default=0.75)
    parser.add_argument("--iterative-passes", type=int, default=2)
    parser.add_argument("--qap-iterations", type=int, default=12)
    parser.add_argument("--qap-restarts", type=int, default=1)
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.train_sources = 2
        args.epochs = 1
        args.quick_val_sources = 1
        args.calibration_sources = 1
        args.validation_sources = 1
        args.validation_replicas = 1
        args.solver_sources = 1
        args.queries_per_source = 4
        args.negatives = 3
        args.groups_per_step = 2
        args.model_dim = 64
        args.layers = 2
        args.heads = 4
        args.feedforward_dim = 128
        args.cnn_channels = 32
        args.patch_grid = 3
        args.band_tokens = 4
        args.candidate_top_k = 4
        args.candidate_reverse_top_k = 0
        args.pair_batch_size = 32
        args.qap_iterations = 1
        args.iterative_passes = 1
    return args


def _init_runtime(seed: int) -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    resolved_seed = seed + rank
    random.seed(resolved_seed)
    np.random.seed(resolved_seed % (2**32 - 1))
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
    return Runtime(rank, world_size, local_rank, device)


def _barrier(runtime: Runtime) -> None:
    if runtime.world_size > 1:
        dist.barrier()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path.parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _write_hashes(output_dir: Path, paths: list[Path]) -> Path:
    target = output_dir / HASHES_NAME
    _atomic_write_text(
        target,
        "".join(f"{_sha256(path)}  {path.name}\n" for path in paths),
    )
    return target


def _names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"RNG state is missing {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu())
    cuda_states = list(state["torch_cuda"])
    if torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG count differs from visible devices")
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(value, dtype=torch.uint8).cpu() for value in cuda_states]
        )
    elif cuda_states:
        raise ValueError("checkpoint contains CUDA RNG but CUDA is unavailable")


def _all_gather_objects(value: Any, runtime: Runtime) -> list[Any]:
    if runtime.world_size == 1:
        return [value]
    gathered: list[Any] = [None] * runtime.world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _all_ranks_finite(value: torch.Tensor, runtime: Runtime) -> bool:
    flag = torch.tensor(
        int(bool(torch.isfinite(value.detach()).all())),
        dtype=torch.int32,
        device=runtime.device,
    )
    if runtime.world_size > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _runtime_resume_contract(runtime: Runtime, *, amp: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rank": runtime.rank,
        "device_type": runtime.device.type,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "amp": bool(amp),
        "amp_dtype": "torch.float16" if amp else "torch.float32",
    }
    if runtime.device.type == "cuda":
        record.update(
            {
                "gpu": torch.cuda.get_device_name(runtime.device),
                "capability": list(torch.cuda.get_device_capability(runtime.device)),
                "total_memory": int(
                    torch.cuda.get_device_properties(runtime.device).total_memory
                ),
            }
        )
    return record


def _current_code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)): _sha256(path) for path in CODE_PATHS}


def _filename_qap_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little") + 7001


def _upstream_disjoint_audit(
    *,
    quick_names: list[str],
    calibration_names: list[str],
    validation_names: list[str],
    assembly_cal: list[str],
    assembly_incremental_gate: list[str],
    manifest_path: str | Path,
    quarantine_path: str | Path,
    denoiser_metadata: dict[str, Any],
    hbt_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Prove evaluation sources were unseen by every frozen upstream learner."""

    manifest = _load_json(manifest_path)
    quarantine = _load_json(quarantine_path)
    manifest_hash = _sha256(manifest_path)
    quarantine_hash = _sha256(quarantine_path)
    if denoiser_metadata.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("denoiser manifest provenance differs from active manifest")
    if hbt_metadata.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("HBT manifest provenance differs from active manifest")
    if hbt_metadata.get("quarantine_sha256") != quarantine_hash:
        raise RuntimeError("HBT quarantine provenance differs from active quarantine")
    quarantine_manifest = quarantine.get("manifest", {})
    if not isinstance(quarantine_manifest, dict) or quarantine_manifest.get("sha256") != manifest_hash:
        raise RuntimeError("quarantine file does not attest the active manifest")

    if quick_names != assembly_cal[: len(quick_names)]:
        raise RuntimeError("quick validation is not the authoritative assembly_cal prefix")
    expected_calibration = assembly_cal[
        len(quick_names) : len(quick_names) + len(calibration_names)
    ]
    if calibration_names != expected_calibration:
        raise RuntimeError("calibration is not the authoritative disjoint assembly_cal slice")
    if validation_names != assembly_incremental_gate[: len(validation_names)]:
        raise RuntimeError("holdout is not the authoritative assembly_incremental_gate prefix")

    splits = manifest.get("splits", {})
    if not isinstance(splits, dict):
        raise RuntimeError("manifest lacks split provenance")
    upstream_sets = {
        "denoiser_train": set(map(str, splits.get("train", []))),
        "denoiser_validation": set(
            map(str, quarantine.get("synthetic_validation_names", []))
        ),
        "denoiser_quarantine": set(map(str, quarantine.get("quarantine_names", []))),
        "denoiser_legacy_train": set(
            map(str, quarantine.get("legacy_train_seen_names", []))
        ),
        "denoiser_legacy_validation": set(
            map(str, quarantine.get("legacy_validation_seen_names", []))
        ),
        "hbt_train": set(map(str, hbt_metadata.get("train_names", []))),
    }
    if len(upstream_sets["denoiser_train"]) != 4900:
        raise RuntimeError("denoiser train provenance must contain 4900 sources")
    if len(upstream_sets["denoiser_validation"]) != 24:
        raise RuntimeError("denoiser validation provenance must contain 24 sources")
    if not upstream_sets["hbt_train"]:
        raise RuntimeError("HBT checkpoint does not record training source names")
    manifest_validation = set(map(str, splits.get("val", [])))
    if not upstream_sets["denoiser_validation"] <= manifest_validation:
        raise RuntimeError("denoiser validation names are not in manifest val")
    if not upstream_sets["denoiser_validation"] <= upstream_sets["denoiser_quarantine"]:
        raise RuntimeError("denoiser validation names are not quarantined")
    if not upstream_sets["hbt_train"] <= upstream_sets["denoiser_train"]:
        raise RuntimeError("HBT train provenance is not a subset of manifest train")

    evaluation = set(quick_names) | set(calibration_names) | set(validation_names)
    overlap_counts = {
        label: len(evaluation & upstream) for label, upstream in upstream_sets.items()
    }
    if any(overlap_counts.values()):
        raise RuntimeError(f"evaluation overlaps frozen upstream training: {overlap_counts}")
    if (set(quick_names) | set(calibration_names)) & set(validation_names):
        raise RuntimeError("assembly_cal and incremental holdout overlap")
    return {
        "quick_partition": "assembly_cal",
        "calibration_partition": "assembly_cal",
        "holdout_partition": "assembly_incremental_gate",
        "quick_names_sha256": _names_sha256(quick_names),
        "calibration_names_sha256": _names_sha256(calibration_names),
        "holdout_names_sha256": _names_sha256(validation_names),
        "upstream_source_counts": {
            label: len(values) for label, values in upstream_sets.items()
        },
        "overlap_counts": overlap_counts,
        "all_upstream_overlaps_zero": True,
        "manifest_sha256": manifest_hash,
        "quarantine_sha256": quarantine_hash,
        "denoiser_training_data_sha256": denoiser_metadata.get("training_data_sha256"),
        "denoiser_validation_data_sha256": denoiser_metadata.get("validation_data_sha256"),
        "hbt_train_names_sha256": _names_sha256(sorted(upstream_sets["hbt_train"])),
    }


def _bounded_t4_preflight(
    model: PairTransformerScorer,
    args: argparse.Namespace,
    runtime: Runtime,
) -> dict[str, Any]:
    tokens = 1 + 2 * args.patch_grid**2 + 2 * args.band_tokens
    micro_pairs = args.groups_per_step * (args.negatives + 1)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    # Conservative lower-bound style envelope: fp32 master/grad + Adam moments,
    # checkpointed token/attention activations, and both CNN endpoint streams.
    parameter_bytes = parameters * 16
    token_bytes = (
        micro_pairs
        * args.layers
        * (
            24 * tokens * args.model_dim
            + 4 * args.heads * tokens * tokens
        )
        * 2
    )
    cnn_bytes = micro_pairs * 2 * args.cnn_channels * 20 * 20 * 12
    estimated = int(parameter_bytes + token_bytes + cnn_bytes)
    t4_usable = int(15 * 1024**3 * 0.72)
    passed = estimated <= t4_usable and micro_pairs <= 512
    record: dict[str, Any] = {
        "tokens_per_pair": int(tokens),
        "pairs_per_microstep": int(micro_pairs),
        "parameter_count": int(parameters),
        "estimated_training_envelope_bytes": estimated,
        "t4_72pct_budget_bytes": t4_usable,
        "fits_bounded_t4_envelope": passed,
    }
    if runtime.device.type == "cuda":
        free, total = torch.cuda.mem_get_info(runtime.device)
        record.update(
            {
                "actual_free_bytes_before_training": int(free),
                "actual_total_bytes": int(total),
                "actual_gpu_name": torch.cuda.get_device_name(runtime.device),
                "actual_headroom_passed": estimated <= int(free * 0.85),
            }
        )
        passed = passed and bool(record["actual_headroom_passed"])
    record["passed"] = bool(passed)
    if not passed:
        raise RuntimeError(f"bounded T4 preflight failed: {record}")
    return record


def _real_cuda_microstep_preflight(
    model: PairTransformerScorer,
    args: argparse.Namespace,
    runtime: Runtime,
) -> dict[str, Any]:
    """Execute one non-updating default-size forward/backward on each CUDA rank."""

    if runtime.device.type != "cuda":
        return {"executed": False, "reason": "CUDA unavailable"}
    rng_state = _capture_rng_state()
    was_training = model.training
    pairs = args.groups_per_step * (args.negatives + 1)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.perf_counter()
    try:
        first = torch.rand(pairs, 6, 20, 20, device=runtime.device)
        second = torch.rand_like(first)
        directions = torch.arange(pairs, device=runtime.device) % 2
        amp = not args.no_amp
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp
        ):
            output = model(first, second, directions)
            loss = output["logits"].square().mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("T4 preflight produced non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError("T4 preflight produced non-finite gradients")
        torch.cuda.synchronize(runtime.device)
        return {
            "executed": True,
            "pairs": pairs,
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(runtime.device)
            ),
            "peak_cuda_reserved_bytes": int(
                torch.cuda.max_memory_reserved(runtime.device)
            ),
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        _restore_rng_state(rng_state)


def _build_resume_contract(
    *,
    args: argparse.Namespace,
    model: PairTransformerScorer,
    train_names: list[str],
    quick_names: list[str],
    pseudo_names: list[str],
    provenance: dict[str, Any],
    runtime_contracts: list[Any],
) -> dict[str, Any]:
    return {
        "model_config": model.config(),
        "trajectory_arguments": {name: getattr(args, name) for name in RESUME_ARGUMENTS},
        "train_names_sha256": _names_sha256(train_names),
        "quick_names_sha256": _names_sha256(quick_names),
        "pseudo_names_sha256": _names_sha256(pseudo_names),
        "code_sha256": _current_code_hashes(),
        "asset_sha256": {
            "denoiser": provenance["assets"]["denoiser_sha256"],
            "hbt": provenance["assets"]["hbt_sha256"],
            "pseudo": provenance["assets"]["pseudo"].get("sha256"),
            "manifest": provenance["code"]["manifest_sha256"],
            "quarantine": provenance["code"]["quarantine_sha256"],
        },
        "runtime_contracts_by_rank": runtime_contracts,
    }


def _validate_resume_payload(
    payload: dict[str, Any],
    *,
    expected_contract: dict[str, Any],
    runtime: Runtime,
) -> dict[str, Any]:
    required = {
        "optimizer_state",
        "scaler_state",
        "scheduler_state",
        "training_state",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"resume checkpoint is missing {sorted(missing)}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("resume_contract") != expected_contract:
        raise ValueError("resume checkpoint config/hash contract differs from this launch")
    state = payload["training_state"]
    if not isinstance(state, dict):
        raise ValueError("resume training_state must be a dictionary")
    if int(state.get("world_size", -1)) != runtime.world_size:
        raise ValueError("resume world_size differs from this launch")
    cursor = state.get("cursor")
    if not isinstance(cursor, dict) or cursor.get("source_index") != 0:
        raise ValueError("resume checkpoint is not at an epoch boundary")
    if cursor.get("pseudo_cursor") != 0 or cursor.get("capture_point") != "epoch_boundary":
        raise ValueError("resume cursor is not an exact epoch boundary")
    rng_states = state.get("rng_states_by_rank")
    generator_states = state.get("generator_states_by_rank")
    if not isinstance(rng_states, list) or len(rng_states) != runtime.world_size:
        raise ValueError("resume lacks one RNG state per rank")
    if not isinstance(generator_states, list) or len(generator_states) != runtime.world_size:
        raise ValueError("resume lacks one augmentation Generator state per rank")
    return state


def _hardware_probe(runtime: Runtime) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rank": runtime.rank,
        "world_size": runtime.world_size,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(runtime.device),
    }
    if runtime.device.type == "cuda":
        result.update(
            {
                "gpu": torch.cuda.get_device_name(runtime.device),
                "capability": list(torch.cuda.get_device_capability(runtime.device)),
                "total_memory": int(torch.cuda.get_device_properties(runtime.device).total_memory),
                "tensor_probe": float((torch.ones(8, device=runtime.device) * 2.0).sum()),
            }
        )
        try:
            result["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            result["nvidia_smi"] = ["unavailable"]
    return result


def _model(args: argparse.Namespace) -> PairTransformerScorer:
    return PairTransformerScorer(
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        feedforward_dim=args.feedforward_dim,
        cnn_channels=args.cnn_channels,
        patch_grid=args.patch_grid,
        side_band=args.side_band,
        band_tokens=args.band_tokens,
        dropout=args.dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )


def _exact_edges(labels: DirectionLabels) -> PositiveEdges:
    first = np.concatenate([labels.right_queries, labels.down_queries]).astype(np.int32)
    second = np.concatenate([labels.right_targets, labels.down_targets]).astype(np.int32)
    directions = np.concatenate(
        [
            np.full(len(labels.right_queries), RIGHT, dtype=np.int8),
            np.full(len(labels.down_queries), DOWN, dtype=np.int8),
        ]
    )
    return PositiveEdges(first, second, directions, np.ones(len(first), dtype=np.float32))


def _pseudo_edges(source: PseudoSource) -> PositiveEdges | None:
    mapping = {int(position): (int(slot), float(confidence)) for slot, position, confidence in zip(source.slots, source.positions, source.confidence, strict=True)}
    first: list[int] = []
    second: list[int] = []
    directions: list[int] = []
    weights: list[float] = []
    for position, (slot, confidence) in mapping.items():
        for direction, neighbour in ((RIGHT, position + 1), (DOWN, position + GRID)):
            if direction == RIGHT and position % GRID == GRID - 1:
                continue
            if direction == DOWN and position >= TILE_COUNT - GRID:
                continue
            if neighbour in mapping:
                target, target_confidence = mapping[neighbour]
                first.append(slot)
                second.append(target)
                directions.append(direction)
                weights.append(min(confidence, target_confidence))
    if not first:
        return None
    weight_array = np.asarray(weights, dtype=np.float32)
    weight_array /= max(float(np.median(weight_array)), 1e-6)
    weight_array = np.clip(weight_array, 0.5, 2.0)
    return PositiveEdges(
        np.asarray(first, dtype=np.int32),
        np.asarray(second, dtype=np.int32),
        np.asarray(directions, dtype=np.int8),
        weight_array,
    )


def _load_pseudo_sources(
    path: Path,
    *,
    allowed_names: set[str],
    confidence_threshold: float,
) -> tuple[list[PseudoSource], dict[str, Any]]:
    if not path.is_file():
        return [], {"available": False, "path": str(path)}
    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["meta"]))
        names = artifact["source_names"].astype(str)
        source_index = artifact["source_index"].astype(np.int64)
        selected = artifact["joint_confidence"].astype(np.float32) >= confidence_threshold
        output: list[PseudoSource] = []
        for index, name in enumerate(names.tolist()):
            if name not in allowed_names:
                continue
            mask = selected & (source_index == index)
            if int(mask.sum()) < 8:
                continue
            candidate = PseudoSource(
                name=name,
                slots=artifact["input_slot"][mask].astype(np.int32),
                positions=artifact["clean_tile_index"][mask].astype(np.int32),
                confidence=artifact["joint_confidence"][mask].astype(np.float32),
            )
            edges = _pseudo_edges(candidate)
            if edges is not None and set(edges.direction.tolist()) == {RIGHT, DOWN}:
                output.append(candidate)
    return output, {
        "available": True,
        "path": str(path),
        "sha256": _sha256(path),
        "kind": metadata.get("kind"),
        "confidence_threshold": confidence_threshold,
        "eligible_source_count": len(output),
        "whole_source_filter": "edge_train only; edge_development excluded",
    }


def _mine_groups(
    positives: PositiveEdges,
    hbt: CompatibilityMatrices,
    visual: CompatibilityMatrices,
    *,
    rng: np.random.Generator,
    queries: int,
    negatives: int,
    hbt_fraction: float,
    visual_fraction: float,
) -> TrainingGroups:
    selected_indices: list[int] = []
    direction_quotas = {RIGHT: queries // 2, DOWN: queries - queries // 2}
    for direction in (RIGHT, DOWN):
        eligible = np.flatnonzero(positives.direction == direction)
        if not len(eligible):
            raise ValueError("positive source has no edges in one direction")
        take = direction_quotas[direction]
        selected_indices.extend(
            rng.choice(eligible, size=take, replace=len(eligible) < take).tolist()
        )
    groups_first: list[list[int]] = []
    groups_second: list[list[int]] = []
    groups_direction: list[list[int]] = []
    weights: list[float] = []
    hbt_count = int(round(negatives * hbt_fraction))
    visual_count = int(round(negatives * visual_fraction))
    for edge_index in selected_indices:
        first = int(positives.first[edge_index])
        target = int(positives.second[edge_index])
        direction = int(positives.direction[edge_index])
        hbt_matrix = hbt.right if direction == RIGHT else hbt.down
        visual_matrix = visual.right if direction == RIGHT else visual.down
        chosen: list[int] = []
        seen = {first, target}

        def add_from(order: np.ndarray, limit: int) -> None:
            for candidate in order.tolist():
                candidate = int(candidate)
                if candidate not in seen:
                    seen.add(candidate)
                    chosen.append(candidate)
                if len(chosen) >= limit:
                    break

        add_from(np.argsort(hbt_matrix[first], kind="stable"), hbt_count)
        add_from(np.argsort(visual_matrix[first], kind="stable"), hbt_count + visual_count)
        if len(chosen) < negatives:
            add_from(rng.permutation(TILE_COUNT), negatives)
        chosen = chosen[:negatives]
        if len(chosen) != negatives:
            raise RuntimeError("failed to fill hard-negative group")
        seconds = [target, *chosen]
        groups_first.append([first] * len(seconds))
        groups_second.append(seconds)
        groups_direction.append([direction] * len(seconds))
        weights.append(float(positives.weight[edge_index]))
    return TrainingGroups(
        np.asarray(groups_first, dtype=np.int32),
        np.asarray(groups_second, dtype=np.int32),
        np.asarray(groups_direction, dtype=np.int8),
        np.asarray(weights, dtype=np.float32),
    )


def _random(
    shape: tuple[int, ...],
    values: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.rand(shape, device=values.device, dtype=values.dtype, generator=generator)


def _augment_views(
    values: torch.Tensor,
    directions: torch.Tensor,
    *,
    endpoint: str,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> torch.Tensor:
    """GPU augmentation after real primary/libjpeg corruption and denoising."""

    values = values.float().div(255.0) if values.detach().amax() > 1.5 else values.float()
    values = values.clone()
    batch = len(values)
    affine_mask = _random((batch, 1, 1, 1), values, generator) < args.affine_probability
    extrapolate = _random((batch, 1, 1, 1), values, generator) < args.extrapolation_probability
    scale_width = torch.where(extrapolate, values.new_tensor(0.35), values.new_tensor(0.20))
    offset_width = torch.where(extrapolate, values.new_tensor(30.0 / 255.0), values.new_tensor(15.0 / 255.0))
    scale = 1.0 + (2.0 * _random((batch, 1, 1, 1), values, generator) - 1.0) * scale_width
    offset = (2.0 * _random((batch, 1, 1, 1), values, generator) - 1.0) * offset_width
    transformed = (values - 0.5) * scale + 0.5 + offset
    values = torch.where(affine_mask, transformed, values)

    noise_mask = _random((batch, 1, 1, 1), values, generator) < args.extra_noise_probability
    sigma = torch.where(
        extrapolate,
        values.new_tensor(args.extrapolation_sigma / 255.0),
        values.new_tensor(args.extra_noise_sigma / 255.0),
    )
    noise = torch.randn(values.shape, device=values.device, dtype=values.dtype, generator=generator)
    channel_scale = values.new_tensor([1.0, 1.0, 1.0, 0.30, 0.30, 0.30]).view(1, 6, 1, 1)
    values = values + noise_mask * sigma * channel_scale * noise

    blur_mask = _random((batch, 1, 1, 1), values, generator) < args.blur_probability
    blurred = F.avg_pool2d(F.pad(values, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    blur_mix = 0.35 + 0.50 * _random((batch, 1, 1, 1), values, generator)
    values = torch.where(blur_mask, (1.0 - blur_mix) * values + blur_mix * blurred, values)

    # True JPEG is present in independent_libjpeg panels.  Random quantization
    # adds a cheap quality extrapolation between true codec examples.
    jpeg_mask = _random((batch, 1, 1, 1), values, generator) < args.jpeg_probability
    quant_step = (2.0 + 14.0 * _random((batch, 1, 1, 1), values, generator)) / 255.0
    quantized = torch.round(values / quant_step) * quant_step
    values = torch.where(jpeg_mask, quantized, values)

    if args.max_erosion > 0 and args.erosion_probability > 0:
        erosion_mask = _random((batch,), values, generator) < args.erosion_probability
        widths = torch.randint(
            1,
            args.max_erosion + 1,
            (batch,),
            device=values.device,
            generator=generator,
        )
        for index in torch.nonzero(erosion_mask, as_tuple=False).flatten().tolist():
            width = int(widths[index])
            direction = int(directions[index])
            if endpoint == "first" and direction == RIGHT:
                values[index, :, :, -width:] = values[index, :, :, -width - 1 : -width]
            elif endpoint == "second" and direction == RIGHT:
                values[index, :, :, :width] = values[index, :, :, width : width + 1]
            elif endpoint == "first" and direction == DOWN:
                values[index, :, -width:, :] = values[index, :, -width - 1 : -width, :]
            else:
                values[index, :, :width, :] = values[index, :, width : width + 1, :]

    dropout_draw = _random((batch, 1, 1, 1), values, generator)
    raw_drop = dropout_draw < args.view_dropout
    denoised_drop = (dropout_draw >= args.view_dropout) & (dropout_draw < 2 * args.view_dropout)
    values[:, :3] = torch.where(raw_drop, values.new_tensor(0.5), values[:, :3])
    values[:, 3:] = torch.where(denoised_drop, values.new_tensor(0.5), values[:, 3:])
    return values.clamp(0.0, 1.0)


def _augment_training_group_batch(
    bank: torch.Tensor,
    groups: TrainingGroups,
    start: int,
    stop: int,
    *,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_first = groups.first[start:stop]
    group_direction = groups.direction[start:stop]
    if not np.all(group_first == group_first[:, :1]):
        raise RuntimeError("ranking group repeats different query tiles")
    if not np.all(group_direction == group_direction[:, :1]):
        raise RuntimeError("ranking group mixes directions")
    query_index = torch.as_tensor(group_first[:, 0], device=bank.device)
    query_direction = torch.as_tensor(group_direction[:, 0], device=bank.device)
    second_index = torch.as_tensor(
        groups.second[start:stop].reshape(-1), device=bank.device
    )
    directions = torch.as_tensor(group_direction.reshape(-1), device=bank.device)
    first_once = _augment_views(
        bank[query_index],
        query_direction,
        endpoint="first",
        args=args,
        generator=generator,
    )
    first = (
        first_once[:, None]
        .expand(-1, groups.group_size, -1, -1, -1)
        .reshape(-1, *first_once.shape[1:])
    )
    second = _augment_views(
        bank[second_index],
        directions,
        endpoint="second",
        args=args,
        generator=generator,
    )
    return first, second, directions


def _source_banks(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    *,
    hbt_model: nn.Module,
    runtime: Runtime,
) -> tuple[CompatibilityMatrices, CompatibilityMatrices]:
    hbt, _ = learned_compatibility(
        hbt_model, denoised_tiles, device=runtime.device, name="denoised_hbt"
    )
    visual = prediction_compatibility(raw_tiles, prefix="raw")
    return hbt, visual


def _train_groups(
    forward_model: nn.Module,
    groups: TrainingGroups,
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    *,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    runtime: Runtime,
    generator: torch.Generator,
    source_weight: float,
    amp_skip_state: dict[str, int],
) -> dict[str, float]:
    views = np.concatenate([raw_tiles, denoised_tiles], axis=3)
    bank = torch.from_numpy(np.ascontiguousarray(views.transpose(0, 3, 1, 2))).to(
        runtime.device, dtype=torch.float32
    )
    losses: list[float] = []
    hits: list[float] = []
    completed_steps = 0
    attempted_steps = 0
    skipped_steps = 0
    amp = runtime.device.type == "cuda" and not args.no_amp
    forward_model.train()
    started = time.perf_counter()
    for start in range(0, groups.group_count, args.groups_per_step):
        stop = min(start + args.groups_per_step, groups.group_count)
        # One stochastic query view per ranking group, then exact replication.
        # Independently augmenting the same query for every candidate creates a
        # shortcut where candidates are ranked partly by query-corruption noise.
        first, second, directions = _augment_training_group_batch(
            bank,
            groups,
            start,
            stop,
            args=args,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=runtime.device.type, dtype=torch.float16, enabled=amp):
            output = forward_model(first, second, directions)
            logits = output["logits"].reshape(stop - start, groups.group_size)
            truth = torch.zeros(stop - start, dtype=torch.long, device=runtime.device)
            cross_entropy = F.cross_entropy(logits, truth, reduction="none")
            binary_truth = torch.zeros_like(logits)
            binary_truth[:, 0] = 1.0
            positive_weight = logits.new_tensor(float(groups.group_size - 1))
            binary = F.binary_cross_entropy_with_logits(
                logits, binary_truth, pos_weight=positive_weight, reduction="none"
            ).mean(dim=1)
            weights = torch.as_tensor(groups.weight[start:stop], device=runtime.device)
            loss = source_weight * torch.mean(weights * (cross_entropy + args.bce_weight * binary))
        if not _all_ranks_finite(loss, runtime):
            raise RuntimeError("non-finite pair-transformer loss across DDP ranks")
        losses.append(float(loss.detach().cpu()))
        hits.append(float((logits.argmax(dim=1) == 0).float().mean().detach().cpu()))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            forward_model.parameters(), args.grad_clip
        )
        gradients_finite = _all_ranks_finite(gradient_norm, runtime)
        if not gradients_finite and not scaler.is_enabled():
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("non-finite gradients with AMP disabled")
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        attempted_steps += 1
        skipped = bool(scaler.is_enabled() and scale_after < scale_before)
        if runtime.world_size > 1:
            skip_min = torch.tensor(int(skipped), device=runtime.device)
            skip_max = skip_min.clone()
            dist.all_reduce(skip_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(skip_max, op=dist.ReduceOp.MAX)
            if int(skip_min.item()) != int(skip_max.item()):
                raise RuntimeError("GradScaler skip decision differs across DDP ranks")
            skipped = bool(skip_min.item())
        if skipped:
            skipped_steps += 1
            amp_skip_state["count"] = int(amp_skip_state.get("count", 0)) + 1
            optimizer.zero_grad(set_to_none=True)
            if amp_skip_state["count"] > args.max_amp_skips:
                raise RuntimeError(
                    f"AMP skipped more than {args.max_amp_skips} bounded updates"
                )
            if runtime.primary:
                print(
                    json.dumps(
                        {
                            "event": "pair_amp_update_skipped",
                            "count": amp_skip_state["count"],
                            "scale_before": scale_before,
                            "scale_after": scale_after,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            continue
        if not gradients_finite:
            raise RuntimeError("non-finite gradients were not skipped by GradScaler")
        scheduler.step()
        completed_steps += 1
    elapsed = time.perf_counter() - started
    pair_count = groups.group_count * groups.group_size
    return {
        "loss": float(np.mean(losses)),
        "forced_group_recall_at_1": float(np.mean(hits)),
        "steps": float(completed_steps),
        "attempted_steps": float(attempted_steps),
        "skipped_steps": float(skipped_steps),
        "pairs": float(pair_count),
        "seconds": float(elapsed),
        "pairs_per_second": float(pair_count / max(elapsed, 1.0e-9)),
    }


def _c1_w1_and_w4(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    hbt: CompatibilityMatrices,
    *,
    chunk_size: int,
) -> tuple[CompatibilityMatrices, CompatibilityMatrices, CompatibilityMatrices]:
    bank = build_classical_score_bank(raw_tiles, prefix="raw", chunk_size=chunk_size)
    bank.update(build_classical_score_bank(denoised_tiles, prefix="denoised", chunk_size=chunk_size))
    names = [name for name in sorted(bank) if name.startswith("denoised_") and not name.endswith("_c2")]
    c1 = fuse_ranked_scores(bank, names=names, name="denoised_C1")
    fusion_bank = {c1.name: c1, hbt.name: hbt}
    w1 = fuse_ranked_scores(
        fusion_bank,
        names=[c1.name, hbt.name],
        weights={hbt.name: 1.0},
        name="denoised_C1_HBTw1",
    )
    w4 = fuse_ranked_scores(
        fusion_bank,
        names=[c1.name, hbt.name],
        weights={hbt.name: 4.0},
        name="denoised_C1_HBTw4",
    )
    return c1, w1, w4


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
    initial: np.ndarray,
    *,
    seed: int,
    args: argparse.Namespace,
    iterations: int | None = None,
    restarts: int | None = None,
) -> np.ndarray:
    return directional_qap(
        score,
        initial=initial,
        iterations=args.qap_iterations if iterations is None else iterations,
        restarts=args.qap_restarts if restarts is None else restarts,
        seed=seed,
        boundary_weight=0.05,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=8,
        refine_weak_cells=32,
    ).position_to_slot


def _equal_budget_no_neural_control(
    score: CompatibilityMatrices,
    initial: np.ndarray,
    *,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """Run the exact same post-shared QAP stage count without neural scores."""

    current = np.asarray(initial, dtype=np.int32).copy()
    for _ in range(args.iterative_passes):
        current = _qap_layout(score, current, seed=seed, args=args)
    return current


def _truth_for_candidates(candidates: PairCandidates, slot_to_target: np.ndarray) -> np.ndarray:
    right, down = true_neighbour_slots(slot_to_target)
    truth = np.where(candidates.direction == RIGHT, right[candidates.first], down[candidates.first])
    return (candidates.second == truth).astype(np.float32)


def _prepare_exact(
    name: str,
    panel_name: str,
    replica: int,
    *,
    args: argparse.Namespace,
    restorer: nn.Module,
    hbt_model: nn.Module,
    runtime: Runtime,
    stage: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CompatibilityMatrices, CompatibilityMatrices, int]:
    target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
    seed = per_source_seed(args.seed, f"pair-transformer-{stage}-{panel_name}", name, replica)
    panel = make_exact_panel(target, panel=panel_name, seed=seed)
    denoised = restore_tiles_uint8(restorer, panel.slot_tiles, runtime.device, batch_size=args.denoise_batch_size)
    hbt, visual = _source_banks(panel.slot_tiles, denoised, hbt_model=hbt_model, runtime=runtime)
    return panel.slot_tiles, denoised, panel.slot_to_target, hbt, visual, seed


def _quick_validation(
    model: PairTransformerScorer,
    names: list[str],
    *,
    args: argparse.Namespace,
    restorer: nn.Module,
    hbt_model: nn.Module,
    runtime: Runtime,
) -> dict[str, float]:
    records: list[dict[str, float]] = []
    for name in names[runtime.rank :: runtime.world_size]:
        raw, denoised, permutation, hbt, _, _ = _prepare_exact(
            name,
            "primary_kornia",
            0,
            args=args,
            restorer=restorer,
            hbt_model=hbt_model,
            runtime=runtime,
            stage="quick-validation",
        )
        result = pair_transformer_compatibility(
            model,
            raw,
            denoised,
            hbt,
            device=runtime.device,
            top_k=min(args.candidate_top_k, 32),
            reverse_top_k=0,
            batch_size=args.pair_batch_size,
            blend=args.neural_blend,
        )
        baseline = retrieval_metrics(hbt, permutation)["combined"]
        neural = retrieval_metrics(result.compatibility, permutation)["combined"]
        records.append(
            {
                "hbt_recall_at_1": baseline["recall_at_1"],
                "neural_recall_at_1": neural["recall_at_1"],
                "delta_recall_at_1": neural["recall_at_1"] - baseline["recall_at_1"],
                "neural_recall_at_32": neural["recall_at_32"],
            }
        )
    gathered = _all_gather_objects(records, runtime)
    merged = [record for rank_records in gathered for record in rank_records]
    if len(merged) != len(names):
        raise RuntimeError("distributed quick validation lost or duplicated sources")
    return {
        key: float(np.mean([record[key] for record in merged])) for key in merged[0]
    }


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
        progress = (step - warmup) / float(max(total_steps - warmup, 1))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(set.intersection(*(set(record) for record in records)))
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def _train(
    model: PairTransformerScorer,
    train_names: list[str],
    quick_names: list[str],
    pseudo_sources: list[PseudoSource],
    *,
    args: argparse.Namespace,
    restorer: nn.Module,
    hbt_model: nn.Module,
    runtime: Runtime,
    output_dir: Path,
    provenance: dict[str, Any],
    resume_payload: dict[str, Any] | None = None,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    if len(train_names) % runtime.world_size:
        raise RuntimeError(
            "train source count must be divisible by world_size for exact resume"
        )
    rank_names = train_names[runtime.rank :: runtime.world_size]
    forward_model: nn.Module = model
    if runtime.world_size > 1:
        forward_model = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_source = math.ceil(args.queries_per_source / args.groups_per_step)
    pseudo_steps = (
        len(rank_names) // max(args.pseudo_every, 1) * steps_per_source
        if pseudo_sources and args.pseudo_every > 0
        else 0
    )
    steps_per_epoch = len(rank_names) * steps_per_source + pseudo_steps
    total_steps = args.epochs * (len(rank_names) * steps_per_source + pseudo_steps)
    scheduler = _scheduler(optimizer, total_steps=total_steps, warmup_fraction=args.warmup_fraction)
    amp = runtime.device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp,
        init_scale=args.amp_init_scale,
        growth_interval=2000,
    )
    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(args.seed + 1009 * runtime.rank)
    best_delta = -float("inf")
    best_path = output_dir / "pair_transformer_best.pt"
    latest_path = output_dir / LATEST_CHECKPOINT
    history: list[dict[str, Any]] = []
    start_epoch = 0
    optimizer_steps = 0
    attempted_steps = 0
    amp_skip_state = {"count": 0}
    resumed_from: str | None = None
    resume_used_previous_fallback = False
    if resume_payload is not None:
        state = _validate_resume_payload(
            resume_payload,
            expected_contract=provenance["resume_contract"],
            runtime=runtime,
        )
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scaler.load_state_dict(resume_payload["scaler_state"])
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        cursor = state["cursor"]
        start_epoch = int(cursor.get("next_epoch", -1))
        if not 0 <= start_epoch <= args.epochs:
            raise ValueError("resume next_epoch is outside configured training")
        if int(cursor.get("completed_epoch", -2)) != start_epoch - 1:
            raise ValueError("resume completed/next epoch markers disagree")
        optimizer_steps = int(state.get("optimizer_steps", -1))
        attempted_steps = int(state.get("attempted_steps", -1))
        amp_skip_state["count"] = int(state.get("amp_skips", -1))
        if min(optimizer_steps, attempted_steps, amp_skip_state["count"]) < 0:
            raise ValueError("resume counters must be non-negative")
        if attempted_steps != optimizer_steps + amp_skip_state["count"]:
            raise ValueError("resume optimizer/skip counters disagree")
        expected_attempted = start_epoch * steps_per_epoch
        if attempted_steps != expected_attempted:
            raise ValueError("resume cursor is inconsistent with attempted step count")
        history = copy.deepcopy(state.get("history", []))
        if len(history) != start_epoch:
            raise ValueError("resume history length differs from epoch cursor")
        best_delta = float(state.get("best_delta", -float("inf")))
        expected_best_hash = state.get("best_checkpoint_sha256")
        if start_epoch > 0:
            if not best_path.is_file() or _sha256(best_path) != expected_best_hash:
                raise ValueError("resume best-checkpoint artifact is missing or changed")
        rng_states = state["rng_states_by_rank"]
        generator_states = state["generator_states_by_rank"]
        _restore_rng_state(rng_states[runtime.rank])
        generator.set_state(
            torch.as_tensor(generator_states[runtime.rank], dtype=torch.uint8).cpu()
        )
        resumed_from = str(
            resume_payload.get("loaded_checkpoint_path", args.resume_checkpoint)
        )
        resume_used_previous_fallback = bool(
            resume_payload.get("used_previous_fallback", False)
        )

    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    panels = [value.strip() for value in args.panels.split(",") if value.strip()]
    total_pairs_local = 0
    total_model_seconds_local = 0.0
    for epoch in range(start_epoch, args.epochs):
        epoch_rng = np.random.default_rng(args.seed + 104729 * epoch)
        shuffled = list(train_names)
        epoch_rng.shuffle(shuffled)
        rank_epoch_names = shuffled[runtime.rank :: runtime.world_size]
        train_records: list[dict[str, float]] = []
        pseudo_cursor = 0
        started = time.perf_counter()
        for source_index, name in enumerate(rank_epoch_names):
            panel_name = panels[(source_index + epoch + runtime.rank) % len(panels)]
            raw, denoised, permutation, hbt, visual, panel_seed = _prepare_exact(
                name,
                panel_name,
                epoch,
                args=args,
                restorer=restorer,
                hbt_model=hbt_model,
                runtime=runtime,
                stage="train",
            )
            groups = _mine_groups(
                _exact_edges(direction_labels(permutation)),
                hbt,
                visual,
                rng=np.random.default_rng(panel_seed),
                queries=args.queries_per_source,
                negatives=args.negatives,
                hbt_fraction=args.hbt_negative_fraction,
                visual_fraction=args.visual_negative_fraction,
            )
            metrics = _train_groups(
                forward_model,
                groups,
                raw,
                denoised,
                args=args,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                runtime=runtime,
                generator=generator,
                source_weight=1.0,
                amp_skip_state=amp_skip_state,
            )
            train_records.append(metrics)
            optimizer_steps += int(metrics["steps"])
            attempted_steps += int(metrics["attempted_steps"])
            total_pairs_local += int(metrics["pairs"])
            total_model_seconds_local += float(metrics["seconds"])

            if pseudo_sources and args.pseudo_every > 0 and (source_index + 1) % args.pseudo_every == 0:
                pseudo = pseudo_sources[(pseudo_cursor * runtime.world_size + runtime.rank) % len(pseudo_sources)]
                pseudo_cursor += 1
                positives = _pseudo_edges(pseudo)
                if positives is not None:
                    input_image = _read_rgb(Path(args.data_root) / "train" / "inputs" / pseudo.name)
                    pseudo_raw = split_tiles_numpy(input_image)
                    pseudo_denoised = restore_tiles_uint8(
                        restorer, pseudo_raw, runtime.device, batch_size=args.denoise_batch_size
                    )
                    pseudo_hbt, pseudo_visual = _source_banks(
                        pseudo_raw, pseudo_denoised, hbt_model=hbt_model, runtime=runtime
                    )
                    pseudo_groups = _mine_groups(
                        positives,
                        pseudo_hbt,
                        pseudo_visual,
                        rng=np.random.default_rng(per_source_seed(args.seed, "pair-pseudo", pseudo.name, epoch)),
                        queries=args.queries_per_source,
                        negatives=args.negatives,
                        hbt_fraction=args.hbt_negative_fraction,
                        visual_fraction=args.visual_negative_fraction,
                    )
                    pseudo_metrics = _train_groups(
                        forward_model,
                        pseudo_groups,
                        pseudo_raw,
                        pseudo_denoised,
                        args=args,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        runtime=runtime,
                        generator=generator,
                        source_weight=args.pseudo_weight,
                        amp_skip_state=amp_skip_state,
                    )
                    optimizer_steps += int(pseudo_metrics["steps"])
                    attempted_steps += int(pseudo_metrics["attempted_steps"])
                    total_pairs_local += int(pseudo_metrics["pairs"])
                    total_model_seconds_local += float(pseudo_metrics["seconds"])
                    pseudo_metrics = {f"pseudo_{key}": value for key, value in pseudo_metrics.items()}
                    train_records.append(pseudo_metrics)
            if runtime.primary and (source_index + 1) % 16 == 0:
                print(json.dumps({"event": "pair_train_progress", "epoch": epoch + 1, "source": source_index + 1, "per_rank": len(rank_epoch_names), "loss": metrics["loss"]}, sort_keys=True), flush=True)

        local_loss_values = [record["loss"] for record in train_records if "loss" in record]
        local_stats = torch.tensor(
            [float(np.sum(local_loss_values)), float(len(local_loss_values))],
            device=runtime.device,
            dtype=torch.float64,
        )
        if runtime.world_size > 1:
            dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
        train_loss = float(local_stats[0] / local_stats[1].clamp_min(1.0))
        epoch_pair_values = [
            record["pairs"] for record in train_records if "pairs" in record
        ] + [
            record["pseudo_pairs"] for record in train_records if "pseudo_pairs" in record
        ]
        epoch_model_seconds = [
            record["seconds"] for record in train_records if "seconds" in record
        ] + [
            record["pseudo_seconds"] for record in train_records if "pseudo_seconds" in record
        ]
        throughput = torch.tensor(
            [float(np.sum(epoch_pair_values)), float(np.sum(epoch_model_seconds))],
            device=runtime.device,
            dtype=torch.float64,
        )
        if runtime.world_size > 1:
            pair_total = throughput[:1].clone()
            seconds_max = throughput[1:].clone()
            dist.all_reduce(pair_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(seconds_max, op=dist.ReduceOp.MAX)
            throughput = torch.cat([pair_total, seconds_max])

        quick = _quick_validation(
            model,
            quick_names,
            args=args,
            restorer=restorer,
            hbt_model=hbt_model,
            runtime=runtime,
        )
        epoch_payload: list[Any] = [None]
        if runtime.primary:
            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "quick_validation": quick,
                "seconds": time.perf_counter() - started,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_steps": optimizer_steps,
                "attempted_steps": attempted_steps,
                "amp_skips": amp_skip_state["count"],
                "training_pairs": int(throughput[0].item()),
                "model_seconds_slowest_rank": float(throughput[1].item()),
                "training_pairs_per_second": float(
                    throughput[0].item() / max(throughput[1].item(), 1.0e-9)
                ),
            }
            if quick["delta_recall_at_1"] > best_delta:
                best_delta = quick["delta_recall_at_1"]
                save_pair_transformer_checkpoint(
                    best_path,
                    model,
                    metadata={
                        **provenance,
                        "training_history": [*history, record],
                        "best_epoch": epoch + 1,
                        "selection_metric": "quick exact primary delta R1 over HBT",
                        "safe_for_submission": False,
                    },
                )
            epoch_payload[0] = {
                "record": record,
                "best_delta": best_delta,
                "best_checkpoint_sha256": _sha256(best_path),
            }
            print(json.dumps({"event": "pair_epoch", **record}, sort_keys=True), flush=True)
        if runtime.world_size > 1:
            dist.broadcast_object_list(epoch_payload, src=0)
        epoch_shared = epoch_payload[0]
        if not isinstance(epoch_shared, dict):
            raise RuntimeError("failed to broadcast epoch checkpoint state")
        history.append(copy.deepcopy(epoch_shared["record"]))
        best_delta = float(epoch_shared["best_delta"])

        rng_states = _all_gather_objects(_capture_rng_state(), runtime)
        generator_states = _all_gather_objects(generator.get_state().cpu(), runtime)
        training_state = {
            "world_size": runtime.world_size,
            "cursor": {
                "completed_epoch": epoch,
                "next_epoch": epoch + 1,
                "source_index": 0,
                "pseudo_cursor": 0,
                "capture_point": "epoch_boundary",
            },
            "optimizer_steps": optimizer_steps,
            "attempted_steps": attempted_steps,
            "amp_skips": amp_skip_state["count"],
            "rng_states_by_rank": rng_states,
            "generator_states_by_rank": generator_states,
            "history": history,
            "best_delta": best_delta,
            "best_checkpoint_sha256": epoch_shared["best_checkpoint_sha256"],
        }
        if runtime.primary:
            save_pair_transformer_checkpoint(
                latest_path,
                model,
                metadata={
                    **provenance,
                    "training_history": history,
                    "latest_completed_epoch": epoch + 1,
                    "safe_for_submission": False,
                },
                optimizer_state=_to_cpu_tree(optimizer.state_dict()),
                scaler_state=_to_cpu_tree(scaler.state_dict()),
                scheduler_state=_to_cpu_tree(scheduler.state_dict()),
                training_state=_to_cpu_tree(training_state),
                preserve_previous=True,
            )
        _barrier(runtime)
    if runtime.primary and not best_path.is_file():
        raise RuntimeError("training did not produce a checkpoint")
    local_peak = {
        "rank": runtime.rank,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(runtime.device))
        if runtime.device.type == "cuda"
        else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(runtime.device))
        if runtime.device.type == "cuda"
        else 0,
        "training_pairs": int(total_pairs_local),
        "model_seconds": float(total_model_seconds_local),
        "pairs_per_second": float(
            total_pairs_local / max(total_model_seconds_local, 1.0e-9)
        ),
    }
    telemetry = {
        "by_rank": _all_gather_objects(local_peak, runtime),
        "resumed_from": resumed_from,
        "resume_used_previous_fallback": resume_used_previous_fallback,
        "start_epoch": start_epoch,
        "optimizer_steps": optimizer_steps,
        "attempted_steps": attempted_steps,
        "amp_skips": amp_skip_state["count"],
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": _sha256(best_path)
        if best_path.is_file()
        else None,
        "latest_checkpoint": str(latest_path),
        "latest_checkpoint_sha256": _sha256(latest_path)
        if latest_path.is_file()
        else None,
    }
    return best_path, history, telemetry


def _calibrate(
    model: PairTransformerScorer,
    names: list[str],
    *,
    args: argparse.Namespace,
    restorer: nn.Module,
    hbt_model: nn.Module,
    runtime: Runtime,
) -> dict[str, Any]:
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for panel_name in [value.strip() for value in args.panels.split(",") if value.strip()]:
        for name in names:
            raw, denoised, permutation, hbt, _, _ = _prepare_exact(
                name,
                panel_name,
                0,
                args=args,
                restorer=restorer,
                hbt_model=hbt_model,
                runtime=runtime,
                stage="calibration",
            )
            candidates = multistage_candidates(
                hbt,
                top_k=args.candidate_top_k,
                reverse_top_k=args.candidate_reverse_top_k,
            )
            logits, _, _ = score_pairs(
                model,
                raw,
                denoised,
                candidates,
                device=runtime.device,
                batch_size=args.pair_batch_size,
            )
            logits_parts.append(logits)
            label_parts.append(_truth_for_candidates(candidates, permutation))
    logits_array = np.concatenate(logits_parts)
    labels_array = np.concatenate(label_parts)
    temperature, bias, metrics = fit_binary_temperature(logits_array, labels_array)
    model.set_calibration(temperature, bias)
    return {
        "temperature": temperature,
        "bias": bias,
        "metrics": metrics,
        "candidate_pairs": int(len(logits_array)),
        "positive_rate": float(labels_array.mean()),
        "source_names": names,
    }


def _evaluate(
    model: PairTransformerScorer,
    names: list[str],
    *,
    args: argparse.Namespace,
    restorer: nn.Module,
    hbt_model: nn.Module,
    runtime: Runtime,
) -> dict[str, Any]:
    panels = [value.strip() for value in args.panels.split(",") if value.strip()]
    retrieval_records: list[dict[str, Any]] = []
    solver_records: list[dict[str, Any]] = []
    for panel_name in panels:
        for replica in range(args.validation_replicas):
            for source_index, name in enumerate(names):
                raw, denoised, permutation, hbt, _, panel_seed = _prepare_exact(
                    name,
                    panel_name,
                    replica,
                    args=args,
                    restorer=restorer,
                    hbt_model=hbt_model,
                    runtime=runtime,
                    stage="holdout",
                )
                hbt_seed = _component_seed(hbt)
                c1, w1, w4 = _c1_w1_and_w4(
                    raw, denoised, hbt, chunk_size=args.chunk_size
                )
                w4_seed = _component_seed(w4)
                neural = pair_transformer_compatibility(
                    model,
                    raw,
                    denoised,
                    hbt,
                    device=runtime.device,
                    top_k=args.candidate_top_k,
                    reverse_top_k=args.candidate_reverse_top_k,
                    layouts=[hbt_seed, w4_seed],
                    batch_size=args.pair_batch_size,
                    blend=args.neural_blend,
                    name="pair_transformer_pass1",
                )
                hbt_metrics = retrieval_metrics(hbt, permutation)["combined"]
                neural_metrics = retrieval_metrics(neural.compatibility, permutation)["combined"]
                neural_seed = _component_seed(neural.compatibility)
                retrieval_records.append(
                    {
                        "panel": panel_name,
                        "replica": replica,
                        "source": name,
                        "hbt_recall_at_1": hbt_metrics["recall_at_1"],
                        "hbt_recall_at_32": hbt_metrics["recall_at_32"],
                        "neural_recall_at_1": neural_metrics["recall_at_1"],
                        "neural_recall_at_32": neural_metrics["recall_at_32"],
                        "delta_recall_at_1": neural_metrics["recall_at_1"] - hbt_metrics["recall_at_1"],
                        "delta_recall_at_32": neural_metrics["recall_at_32"] - hbt_metrics["recall_at_32"],
                        "hbt_softcycle_adjacency": layout_metrics(hbt_seed, permutation)["combined_adjacency"],
                        "neural_softcycle_adjacency": layout_metrics(neural_seed, permutation)["combined_adjacency"],
                        "delta_softcycle_adjacency": layout_metrics(neural_seed, permutation)["combined_adjacency"] - layout_metrics(hbt_seed, permutation)["combined_adjacency"],
                        "candidate_recall": float(
                            np.sum(_truth_for_candidates(neural.candidates, permutation))
                            / float(2 * GRID * (GRID - 1))
                        ),
                        "mean_confidence": neural.diagnostics["mean_confidence"],
                        "candidate_pairs": int(len(neural.candidates)),
                        "pair_transformer_seconds": float(
                            neural.diagnostics["inference_telemetry"][
                                "pair_transformer_seconds"
                            ]
                        ),
                        "pairs_per_second": float(
                            neural.diagnostics["inference_telemetry"]["pairs_per_second"]
                        ),
                    }
                )

                if source_index < args.solver_sources:
                    qap_seed = int(panel_seed % (2**31 - 1))
                    hbt_qap = _qap_layout(hbt, hbt_seed, seed=qap_seed, args=args)
                    w4_qap = _qap_layout(w4, w4_seed, seed=qap_seed, args=args)
                    equal_budget_control = _equal_budget_no_neural_control(
                        w4,
                        w4_qap,
                        seed=qap_seed,
                        args=args,
                    )
                    promoted_w4_i25 = _qap_layout(
                        w4,
                        hbt_seed,
                        seed=_filename_qap_seed(name),
                        args=args,
                        iterations=PROMOTED_QAP_ITERATIONS,
                        restarts=PROMOTED_QAP_RESTARTS,
                    )
                    promoted_w1_i25 = _qap_layout(
                        w1,
                        hbt_seed,
                        seed=_filename_qap_seed(name),
                        args=args,
                        iterations=PROMOTED_QAP_ITERATIONS,
                        restarts=PROMOTED_QAP_RESTARTS,
                    )
                    current_layout = _qap_layout(
                        fuse_ranked_scores(
                            {c1.name: c1, neural.compatibility.name: neural.compatibility},
                            names=[c1.name, neural.compatibility.name],
                            weights={neural.compatibility.name: 4.0},
                            name="C1_pairw4_pass1",
                        ),
                        w4_qap,
                        seed=qap_seed,
                        args=args,
                    )
                    final_neural = neural
                    inference_telemetry = [
                        dict(neural.diagnostics["inference_telemetry"])
                    ]
                    for pass_index in range(1, args.iterative_passes):
                        final_neural = pair_transformer_compatibility(
                            model,
                            raw,
                            denoised,
                            hbt,
                            device=runtime.device,
                            top_k=args.candidate_top_k,
                            reverse_top_k=args.candidate_reverse_top_k,
                            layouts=[hbt_qap, w4_qap, current_layout],
                            batch_size=args.pair_batch_size,
                            blend=args.neural_blend,
                            name=f"pair_transformer_pass{pass_index + 1}",
                        )
                        stage_score = fuse_ranked_scores(
                            {c1.name: c1, final_neural.compatibility.name: final_neural.compatibility},
                            names=[c1.name, final_neural.compatibility.name],
                            weights={final_neural.compatibility.name: 4.0},
                            name=f"C1_pairw4_pass{pass_index + 1}",
                        )
                        current_layout = _qap_layout(
                            stage_score, current_layout, seed=qap_seed, args=args
                        )
                        inference_telemetry.append(
                            dict(final_neural.diagnostics["inference_telemetry"])
                        )
                    hbt_layout = layout_metrics(hbt_qap, permutation)
                    w4_layout = layout_metrics(w4_qap, permutation)
                    equal_layout = layout_metrics(equal_budget_control, permutation)
                    promoted_w4_layout = layout_metrics(
                        promoted_w4_i25, permutation
                    )
                    promoted_w1_layout = layout_metrics(
                        promoted_w1_i25, permutation
                    )
                    neural_layout = layout_metrics(current_layout, permutation)
                    # Every input-only layout is frozen before the target is opened.
                    target_image = _read_rgb(
                        Path(args.data_root) / "train" / "targets" / name
                    )
                    hbt_image = predicted_image_metrics(hbt_qap, denoised, target_image)
                    w4_image = predicted_image_metrics(w4_qap, denoised, target_image)
                    equal_image = predicted_image_metrics(
                        equal_budget_control, denoised, target_image
                    )
                    promoted_w4_image = predicted_image_metrics(
                        promoted_w4_i25, denoised, target_image
                    )
                    promoted_w1_image = predicted_image_metrics(
                        promoted_w1_i25, denoised, target_image
                    )
                    neural_image = predicted_image_metrics(
                        current_layout, denoised, target_image
                    )
                    baseline_adjacency = max(
                        hbt_layout["combined_adjacency"],
                        w4_layout["combined_adjacency"],
                        equal_layout["combined_adjacency"],
                        promoted_w1_layout["combined_adjacency"],
                        promoted_w4_layout["combined_adjacency"],
                    )
                    baseline_ssim = max(
                        hbt_image["predicted_layout_ssim"],
                        w4_image["predicted_layout_ssim"],
                        equal_image["predicted_layout_ssim"],
                        promoted_w1_image["predicted_layout_ssim"],
                        promoted_w4_image["predicted_layout_ssim"],
                    )
                    scored_pairs = sum(
                        int(value["candidate_pairs"]) for value in inference_telemetry
                    )
                    scored_seconds = sum(
                        float(value["pair_transformer_seconds"])
                        for value in inference_telemetry
                    )
                    solver_records.append(
                        {
                            "panel": panel_name,
                            "replica": replica,
                            "source": name,
                            "hbt_qap_adjacency": hbt_layout["combined_adjacency"],
                            "w4_qap_adjacency": w4_layout["combined_adjacency"],
                            "equal_budget_control_adjacency": equal_layout[
                                "combined_adjacency"
                            ],
                            "promoted_w4_i25_adjacency": promoted_w4_layout[
                                "combined_adjacency"
                            ],
                            "promoted_w1_i25_adjacency": promoted_w1_layout[
                                "combined_adjacency"
                            ],
                            "no_neural_envelope_adjacency": baseline_adjacency,
                            "neural_qap_adjacency": neural_layout["combined_adjacency"],
                            "delta_adjacency_vs_hbt": neural_layout["combined_adjacency"] - hbt_layout["combined_adjacency"],
                            "delta_adjacency_vs_no_neural_envelope": neural_layout[
                                "combined_adjacency"
                            ]
                            - baseline_adjacency,
                            "hbt_qap_ssim": hbt_image["predicted_layout_ssim"],
                            "w4_qap_ssim": w4_image["predicted_layout_ssim"],
                            "equal_budget_control_ssim": equal_image[
                                "predicted_layout_ssim"
                            ],
                            "promoted_w4_i25_ssim": promoted_w4_image[
                                "predicted_layout_ssim"
                            ],
                            "promoted_w1_i25_ssim": promoted_w1_image[
                                "predicted_layout_ssim"
                            ],
                            "no_neural_envelope_ssim": baseline_ssim,
                            "neural_qap_ssim": neural_image["predicted_layout_ssim"],
                            "delta_ssim_vs_w4": neural_image["predicted_layout_ssim"] - w4_image["predicted_layout_ssim"],
                            "delta_ssim_vs_hbt": neural_image["predicted_layout_ssim"] - hbt_image["predicted_layout_ssim"],
                            "delta_ssim_vs_no_neural_envelope": neural_image[
                                "predicted_layout_ssim"
                            ]
                            - baseline_ssim,
                            "iterative_passes": args.iterative_passes,
                            "shared_initial_w4_qap_calls": 1,
                            "neural_post_shared_qap_calls": args.iterative_passes,
                            "control_post_shared_qap_calls": args.iterative_passes,
                            "equal_qap_schedule": True,
                            "promoted_qap_iterations": PROMOTED_QAP_ITERATIONS,
                            "promoted_qap_restarts": PROMOTED_QAP_RESTARTS,
                            "promoted_qap_seed": _filename_qap_seed(name),
                            "candidate_pairs": scored_pairs,
                            "pair_transformer_seconds": scored_seconds,
                            "pairs_per_second": float(
                                scored_pairs / max(scored_seconds, 1.0e-9)
                            ),
                            "peak_cuda_allocated_bytes": max(
                                int(value["peak_cuda_allocated_bytes"])
                                for value in inference_telemetry
                            ),
                        }
                    )
                print(json.dumps({"event": "pair_holdout", "panel": panel_name, "replica": replica, "source": source_index + 1, "count": len(names)}, sort_keys=True), flush=True)

    retrieval_aggregate = _mean([{key: value for key, value in record.items() if isinstance(value, (int, float))} for record in retrieval_records])
    solver_aggregate = _mean([{key: value for key, value in record.items() if isinstance(value, (int, float))} for record in solver_records])
    cell_summaries: dict[str, dict[str, float]] = {}
    for panel_name in panels:
        for replica in range(args.validation_replicas):
            key = f"{panel_name}:replica{replica}"
            retrieval_cell = [record for record in retrieval_records if record["panel"] == panel_name and record["replica"] == replica]
            solver_cell = [record for record in solver_records if record["panel"] == panel_name and record["replica"] == replica]
            combined = [
                {key: value for key, value in record.items() if isinstance(value, (int, float))}
                for record in retrieval_cell
            ]
            summary = _mean(combined)
            if solver_cell:
                summary.update({f"solver_{name}": value for name, value in _mean([{key: value for key, value in record.items() if isinstance(value, (int, float))} for record in solver_cell]).items()})
            cell_summaries[key] = summary
    gates = {
        "aggregate_recall_at_1_delta_ge_0.02": retrieval_aggregate["delta_recall_at_1"] >= 0.02,
        "aggregate_recall_at_32_delta_ge_minus_0.005": retrieval_aggregate["delta_recall_at_32"] >= -0.005,
        "aggregate_softcycle_adjacency_delta_ge_0.01": retrieval_aggregate["delta_softcycle_adjacency"] >= 0.01,
        "aggregate_qap_adjacency_delta_vs_no_neural_envelope_ge_0.01": solver_aggregate["delta_adjacency_vs_no_neural_envelope"] >= 0.01,
        "aggregate_qap_ssim_delta_vs_no_neural_envelope_ge_0.005": solver_aggregate["delta_ssim_vs_no_neural_envelope"] >= 0.005,
        "every_panel_replica_positive_r1": all(value["delta_recall_at_1"] > 0.0 for value in cell_summaries.values()),
        "every_panel_replica_positive_qap_ssim_vs_no_neural_envelope": all(value.get("solver_delta_ssim_vs_no_neural_envelope", -1.0) > 0.0 for value in cell_summaries.values()),
    }
    return {
        "source_names": names,
        "panels": panels,
        "replicas": args.validation_replicas,
        "retrieval_records": retrieval_records,
        "solver_records": solver_records,
        "retrieval_aggregate": retrieval_aggregate,
        "solver_aggregate": solver_aggregate,
        "solver_control_contract": {
            "equal_budget_control": "same shared w4 seed plus identical short-QAP stage count as neural",
            "strongest_known_promoted_comparator": "qap_w1_b0.05_i25, restarts=2, HBT softcycle init, authoritative filename seed",
            "additional_promoted_comparator": "qap_w4_b0.05_i25 under the same i25/r2/seed contract",
            "gate_baseline": "per-source max of hbt short, w4 short, equal-budget control, promoted w1 i25, and promoted w4 i25",
        },
        "panel_replica_summaries": cell_summaries,
        "continuation_gates": gates,
        "continue_to_1024_source_two_seed_run": bool(all(gates.values())),
    }


def _validate_args(args: argparse.Namespace, runtime: Runtime) -> None:
    if args.action in {"pilot", "train"} and runtime.world_size > 1 and args.train_sources < runtime.world_size:
        raise SystemExit("train source count is smaller than DDP world size")
    if min(args.epochs, args.queries_per_source, args.negatives, args.groups_per_step) <= 0:
        raise SystemExit("training counts must be positive")
    if args.negatives < 3 or args.queries_per_source < 2:
        raise SystemExit("pilot requires at least three negatives and two queries")
    if args.hbt_negative_fraction + args.visual_negative_fraction > 1.0:
        raise SystemExit("hard-negative fractions sum above one")
    if args.action == "evaluate" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for evaluate")
    if args.resume_checkpoint and args.action not in {"pilot", "train"}:
        raise SystemExit("--resume-checkpoint is valid only for pilot/train")
    if args.resume_checkpoint and args.checkpoint:
        raise SystemExit("--checkpoint and --resume-checkpoint are mutually exclusive")
    if args.validation_sources <= 0 or args.calibration_sources <= 0:
        raise SystemExit("calibration/validation counts must be positive")
    if args.solver_sources <= 0 or args.solver_sources > args.validation_sources:
        raise SystemExit("solver sources must lie in [1, validation_sources]")
    if args.train_sources % runtime.world_size:
        raise SystemExit("train sources must be divisible by DDP world size")
    if args.iterative_passes <= 0 or args.qap_iterations <= 0 or args.qap_restarts <= 0:
        raise SystemExit("iterative/QAP counts must be positive")
    if not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0:
        raise SystemExit("AMP init scale must be finite and positive")
    if args.max_amp_skips < 0:
        raise SystemExit("max AMP skips must be non-negative")


def main() -> None:
    args = parse_args()
    runtime = _init_runtime(args.seed)
    _validate_args(args, runtime)
    output_dir = Path(args.output_dir)
    if runtime.primary:
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and args.action != "evaluate":
            if args.resume_checkpoint:
                allowed = {
                    "pair_transformer_best.pt",
                    LATEST_CHECKPOINT,
                    f"{LATEST_CHECKPOINT}.previous",
                }
                unexpected = {path.name for path in output_dir.iterdir()} - allowed
                if unexpected:
                    raise SystemExit(
                        f"resume output contains unrelated artifacts {sorted(unexpected)}"
                    )
            else:
                raise SystemExit(f"output directory is non-empty; pass --overwrite: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    _barrier(runtime)
    hardware = _hardware_probe(runtime)
    hardware_by_rank = _all_gather_objects(hardware, runtime)
    print(json.dumps({"event": "pair_hardware", **hardware}, sort_keys=True), flush=True)

    train_names = source_names_for_split("edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine)[: args.train_sources]
    assembly_cal = source_names_for_split(
        "assembly_cal", manifest_path=args.manifest, quarantine_path=args.quarantine
    )
    assembly_incremental_gate = source_names_for_split(
        "assembly_incremental_gate",
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )
    quick_names = assembly_cal[: args.quick_val_sources]
    calibration_start = args.quick_val_sources
    calibration_names = assembly_cal[
        calibration_start : calibration_start + args.calibration_sources
    ]
    validation_names = assembly_incremental_gate[: args.validation_sources]
    if (
        len(train_names) != args.train_sources
        or len(quick_names) != args.quick_val_sources
        or len(calibration_names) != args.calibration_sources
        or len(validation_names) != args.validation_sources
    ):
        raise RuntimeError("requested source slice exceeds its authoritative partition")
    partitions = [set(train_names), set(quick_names), set(calibration_names), set(validation_names)]
    if any(partitions[i] & partitions[j] for i in range(len(partitions)) for j in range(i + 1, len(partitions))):
        raise RuntimeError("whole-source partitions overlap")

    restorer, device, denoiser_metadata = load_restorer(args.denoiser, device=str(runtime.device), state="ema")
    if device != runtime.device:
        runtime = Runtime(runtime.rank, runtime.world_size, runtime.local_rank, device)
    hbt_model, hbt_metadata = load_embedding_checkpoint(args.hbt_checkpoint, device=runtime.device)
    for frozen in (restorer, hbt_model):
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
    upstream_audit = _upstream_disjoint_audit(
        quick_names=quick_names,
        calibration_names=calibration_names,
        validation_names=validation_names,
        assembly_cal=assembly_cal,
        assembly_incremental_gate=assembly_incremental_gate,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
        denoiser_metadata=denoiser_metadata,
        hbt_metadata=hbt_metadata,
    )
    pseudo_sources, pseudo_metadata = _load_pseudo_sources(
        Path(args.pseudo_gold),
        allowed_names=set(train_names),
        confidence_threshold=args.pseudo_confidence,
    )
    provenance = {
        "schema_version": 1,
        "kind": "pair_transformer_training_provenance",
        "args": vars(args),
        "seed": args.seed,
        "code": {
            "trainer": str(Path(__file__).resolve()),
            "trainer_sha256": _sha256(Path(__file__).resolve()),
            "model": str((Path(__file__).resolve().parents[1] / "src/puzzle_assembly/pair_transformer.py")),
            "model_sha256": _sha256(Path(__file__).resolve().parents[1] / "src/puzzle_assembly/pair_transformer.py"),
            "manifest_sha256": _sha256(args.manifest),
            "quarantine_sha256": _sha256(args.quarantine),
            "transitive_code_sha256": _current_code_hashes(),
        },
        "whole_source_splits": {
            "train": train_names,
            "quick_validation": quick_names,
            "calibration": calibration_names,
            "holdout": validation_names,
            "pairwise_disjoint": True,
            "partition_contract": {
                "quick_and_calibration": "assembly_cal",
                "holdout": "assembly_incremental_gate",
            },
        },
        "assets": {
            "denoiser": args.denoiser,
            "denoiser_sha256": _sha256(args.denoiser),
            "denoiser_metadata": denoiser_metadata,
            "hbt": args.hbt_checkpoint,
            "hbt_sha256": _sha256(args.hbt_checkpoint),
            "hbt_metadata": hbt_metadata,
            "pseudo": pseudo_metadata,
        },
        "hardware_by_rank": hardware_by_rank,
        "augmentation_contract": {
            "base": "primary_kornia and true independent_libjpeg task panels",
            "brightness_contrast": True,
            "gaussian_noise_sigma_interpolation_and_extrapolation": [args.extra_noise_sigma, args.extrapolation_sigma],
            "blur": True,
            "jpeg": "true panel codec plus stochastic quantization extrapolation",
            "edge_erosion": [args.erosion_probability, args.max_erosion],
            "raw_denoised_view_dropout": args.view_dropout,
        },
        "anti_leakage": {
            "split_by_whole_source": True,
            "real_pseudo_filtered_to_edge_train": True,
            "pseudo_is_partial_not_full_permutation": True,
            "holdout_not_used_for_checkpoint_selection": True,
            "upstream_disjoint_audit": upstream_audit,
            "safe_for_submission": False,
        },
        "runtime_plan_2xt4": {
            "default_512x3": "approximately 1.5-3.5 hours including exact gate",
            "full_1024x6_two_seeds": "approximately 7-12 GPU-hours total",
            "pair_passes_per_image": f"about 2*576*({args.candidate_top_k}+{args.candidate_reverse_top_k}), not dense 2*576^2",
            "inference_tile_encoder_cache": "one shared 576-tile CNN encoding per scoring pass",
        },
        "prior_exact_evidence": {
            "pure_hbt_qap_adjacency_delta_vs_w4": 0.05820,
            "pure_hbt_qap_ssim_delta_vs_w4": -0.00366,
            "implication": "HBT remains the proposal ceiling, but promotion requires neural edge precision plus SSIM and absolute-structure gains",
        },
        "prior_failed_route_guard": {
            "old_route": "one-epoch L1 embedding fine-tune on 512 real pseudo sources",
            "observed_train_recall_at_1": 0.503,
            "observed_exact_validation_before": 0.213,
            "observed_exact_validation_after": 0.194,
            "design_change": "real pseudo edges are only 0.20-weight anchors mixed every fourth synthetic source; the model is a full-pair cross-attention transformer and checkpoint selection stays on disjoint exact synthetic sources",
        },
        "layout_energy_handoff": {
            "exported_feature": "PairTransformerScorer.forward pair_embedding (2*model_dim) for every rescored edge",
            "candidate_api": "multistage_candidates merges HBT top-k, reverse top-k, and all edges realized by one or more QAP layouts",
            "qap_like_negative_recipe": "start from first-pass QAP, then sample tile swaps, row/column strips, block swaps, and locally plausible wrong HBT edges; aggregate pair_embedding/probability/confidence into a later whole-layout energy ViT",
            "implemented_here": False,
            "reason": "kept pair scorer scoped; whole-layout raw-only energy is an independent model family",
        },
        "known_risks": [
            "HBT top-k candidate recall is an upper bound for unseen true edges",
            "partial real pseudo-gold can retain matching bias despite confidence filtering",
            "pairwise compatibility alone may not recover absolute grid position",
            "a passing pilot still requires an independent training seed and real16 freeze-before-target gate",
            "safe_for_submission remains false until those later gates pass",
        ],
    }

    training_telemetry: dict[str, Any] | None = None
    if args.action == "evaluate":
        model, checkpoint_metadata = load_pair_transformer_checkpoint(args.checkpoint, device=runtime.device)
        provenance["loaded_checkpoint"] = {"path": args.checkpoint, "sha256": _sha256(args.checkpoint), "metadata": checkpoint_metadata}
        history: list[dict[str, Any]] = []
        checkpoint_path = Path(args.checkpoint)
    else:
        model = _model(args).to(runtime.device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        provenance["model_parameter_count"] = parameter_count
        if not args.smoke and not 15_000_000 <= parameter_count <= 60_000_000:
            raise RuntimeError(f"default serious model unexpectedly has {parameter_count:,} parameters")
        preflight = _bounded_t4_preflight(model, args, runtime)
        preflight["real_cuda_microstep"] = _real_cuda_microstep_preflight(
            model, args, runtime
        )
        provenance["bounded_t4_preflight_by_rank"] = _all_gather_objects(
            preflight, runtime
        )
        amp = runtime.device.type == "cuda" and not args.no_amp
        runtime_contracts = _all_gather_objects(
            _runtime_resume_contract(runtime, amp=amp), runtime
        )
        provenance["resume_contract"] = _build_resume_contract(
            args=args,
            model=model,
            train_names=train_names,
            quick_names=quick_names,
            pseudo_names=[source.name for source in pseudo_sources],
            provenance=provenance,
            runtime_contracts=runtime_contracts,
        )
        resume_payload: dict[str, Any] | None = None
        if args.resume_checkpoint:
            resume_payload = load_pair_transformer_checkpoint_payload(
                args.resume_checkpoint,
                require_training_state=True,
            )
            if resume_payload["model_config"] != model.config():
                raise ValueError("resume checkpoint model configuration differs from CLI")
            model.load_state_dict(resume_payload["model_state"], strict=True)
        checkpoint_path, history, training_telemetry = _train(
            model,
            train_names,
            quick_names,
            pseudo_sources,
            args=args,
            restorer=restorer,
            hbt_model=hbt_model,
            runtime=runtime,
            output_dir=output_dir,
            provenance=provenance,
            resume_payload=resume_payload,
        )
        provenance["actual_training_telemetry"] = training_telemetry

    # End every distributed collective before the long calibration/solver gate.
    # Rank zero then evaluates as a true single process; other ranks exit instead
    # of waiting in an NCCL barrier until the default process-group timeout.
    _barrier(runtime)
    if dist.is_initialized():
        dist.destroy_process_group()
    if not runtime.primary:
        return
    runtime = Runtime(rank=0, world_size=1, local_rank=0, device=runtime.device)

    if args.action != "evaluate":
        model, checkpoint_metadata = load_pair_transformer_checkpoint(checkpoint_path, device=runtime.device)
    else:
        checkpoint_metadata = provenance["loaded_checkpoint"]["metadata"]
    calibration = _calibrate(
        model,
        calibration_names,
        args=args,
        restorer=restorer,
        hbt_model=hbt_model,
        runtime=runtime,
    )
    calibrated_path = output_dir / "pair_transformer_calibrated.pt"
    save_pair_transformer_checkpoint(
        calibrated_path,
        model,
        metadata={
            **checkpoint_metadata,
            "calibration": calibration,
            "safe_for_submission": False,
        },
    )
    evaluation = _evaluate(
        model,
        validation_names,
        args=args,
        restorer=restorer,
        hbt_model=hbt_model,
        runtime=runtime,
    )
    report = {
        "schema_version": 1,
        "kind": "pair_transformer_2xt4_pilot",
        "status": "continue" if evaluation["continue_to_1024_source_two_seed_run"] else "stop_or_redesign",
        "safe_for_submission": False,
        "provenance": provenance,
        "training_history": history,
        "training_telemetry": training_telemetry,
        "best_checkpoint": None
        if training_telemetry is None
        else training_telemetry.get("best_checkpoint"),
        "best_checkpoint_sha256": None
        if training_telemetry is None
        else training_telemetry.get("best_checkpoint_sha256"),
        "checkpoint": str(calibrated_path),
        "checkpoint_sha256": _sha256(calibrated_path),
        "calibration": calibration,
        "evaluation": evaluation,
        "next_step_if_passed": "repeat 1024-source training with two independent seeds, then freeze real16 layouts before opening targets",
        "next_step_if_failed": "inspect candidate ceiling and pair calibration; pivot to global layout-energy ViT if pair R1 rises without solver SSIM",
    }
    report_path = output_dir / "pair_transformer_report.json"
    _atomic_write_text(
        report_path,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    hash_paths = [report_path, calibrated_path]
    best_path = output_dir / "pair_transformer_best.pt"
    latest_path = output_dir / LATEST_CHECKPOINT
    hash_paths.extend(path for path in (best_path, latest_path) if path.is_file())
    hashes_path = _write_hashes(output_dir, hash_paths)
    print(json.dumps({"event": "pair_pilot_complete", "status": report["status"], "safe_for_submission": False, "report": str(report_path), "report_sha256": _sha256(report_path), "checkpoint": str(calibrated_path), "checkpoint_sha256": report["checkpoint_sha256"], "hashes": str(hashes_path), "hashes_sha256": _sha256(hashes_path), "gates": evaluation["continuation_gates"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
