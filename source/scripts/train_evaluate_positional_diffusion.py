#!/usr/bin/env python3
"""Train and gate a bounded 2xT4 positional-diffusion signal pilot.

The implementation is task-specific but follows the primary Positional
Diffusion design (arXiv:2303.11120): continuous two-dimensional positions,
linear Gaussian diffusion, an attention/GNN reverse model, x0 prediction, and
deterministic DDIM.  Corruption coverage also includes the edge/content erosion
families studied in the corrupted-puzzle benchmark (arXiv:2507.07828).

Important experimental contract
--------------------------------
* Training and development are split by whole source image.
* Raw and denoised tiles are both model inputs.
* HBT supplies only input-derived relative graph evidence.  It never supplies
  labels, target confidence, or a target-selected candidate.
* The reverse-process start is a frozen input-only layout selected once for an
  experiment (soft-cycle by default, w4-QAP as an explicit ablation), with no
  truth-derived confidence.  Training and evaluation use the same family.
* Development uses the same sources under both primary_kornia and
  independent_libjpeg corruption engines on the clean
  assembly_incremental_gate split.  The candidate must beat the per-source
  envelope of frozen equal-budget w1-QAP, w4-QAP, and pure-HBT-QAP on both
  adjacency and restored-layout SSIM.  A zero delta can never pass.
* A pass is evidence for a later bounded signal experiment, never submission
  approval.  Competition test targets are never read.

Recommended Kaggle command (two T4s)::

    torchrun --standalone --nproc_per_node=2 \
      scripts/train_evaluate_positional_diffusion.py \
      --mode pilot --output-dir /kaggle/working/posdiff_pilot
"""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_assembly.compatibility import build_classical_score_bank, fuse_ranked_scores
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.geometry import GRID, TILE, TILE_COUNT
from puzzle_assembly.learned import (
    learned_compatibility,
    load_embedding_checkpoint,
)
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.positional_diffusion import (
    GaussianPositionDiffusion,
    PositionalDiffusionConfig,
    PositionalDiffusionNet,
    compatibility_to_relative_graph,
    estimate_peak_memory_bytes,
    layout_to_tile_positions,
    load_positional_diffusion_checkpoint_payload,
    model_parameter_count,
    normalized_grid_positions,
    save_positional_diffusion_checkpoint,
)
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_HBT = (
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/"
    "hbt_d320_denoised_rgb_sobel.pt"
)
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"
CHECKPOINT_NAME = "positional_diffusion.pt"
LATEST_NAME = "positional_diffusion_latest.pt"
REPORT_NAME = "positional_diffusion_report.json"
HASHES_NAME = "SHA256SUMS.txt"
CODE_PATHS = (
    REPO_ROOT / "src/puzzle_assembly/positional_diffusion.py",
    Path(__file__).resolve(),
    REPO_ROOT / "src/puzzle_assembly/compatibility.py",
    REPO_ROOT / "src/puzzle_assembly/components.py",
    REPO_ROOT / "src/puzzle_assembly/geometry.py",
    REPO_ROOT / "src/puzzle_assembly/learned.py",
    REPO_ROOT / "src/puzzle_assembly/metrics.py",
    REPO_ROOT / "src/puzzle_assembly/panels.py",
    REPO_ROOT / "src/puzzle_assembly/protocol.py",
    REPO_ROOT / "src/puzzle_assembly/qap.py",
    REPO_ROOT / "src/puzzle_assembly/solvers.py",
    REPO_ROOT / "src/puzzle_denoise_v2/degradation.py",
    REPO_ROOT / "src/puzzle_denoise_v2/inference.py",
    REPO_ROOT / "src/puzzle_denoise_v2/losses.py",
    REPO_ROOT / "src/puzzle_denoise_v2/metrics.py",
    REPO_ROOT / "src/puzzle_denoise_v2/model.py",
    REPO_ROOT / "src/puzzle_denoise_v2/tiles.py",
    REPO_ROOT / "src/puzzle_denoise_v2/training.py",
)
RESUME_TRAINING_ARGUMENTS = (
    "train_offset",
    "train_sources",
    "epochs",
    "gradient_accumulation",
    "max_optimizer_steps",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "structure_weight",
    "baseline_condition_dropout",
    "warm_start_layout",
    "graph_top_k",
    "graph_temperature",
    "denoise_batch_size",
    "qap_iterations",
    "qap_restarts",
    "qap_boundary_weight",
    "qap_refine_swaps",
    "amp",
    "amp_init_scale",
    "amp_max_consecutive_skips",
    "amp_max_total_skips",
    "data_root",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "train", "evaluate", "pilot"), default="pilot")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Epoch-boundary training checkpoint; restores optimizer/scaler and every rank RNG",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--hbt-checkpoint", default=DEFAULT_HBT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("auto", "fp16", "bf16", "none"), default="fp16")
    parser.add_argument("--amp-init-scale", type=float, default=4096.0)
    parser.add_argument("--amp-max-consecutive-skips", type=int, default=3)
    parser.add_argument("--amp-max-total-skips", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260711)

    # A bounded but non-trivial 2xT4 pilot: microbatch one per GPU, global
    # effective batch eight, and at most 192 synchronized optimizer updates.
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-sources", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-optimizer-steps", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--structure-weight", type=float, default=0.20)
    parser.add_argument("--baseline-condition-dropout", type=float, default=0.25)
    parser.add_argument(
        "--warm-start-layout",
        choices=("softcycle", "w4-qap"),
        default="softcycle",
        help="Use the same frozen input-only layout family for training and evaluation",
    )

    parser.add_argument("--model-dim", type=int, default=384)
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--feedforward-dim", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--diffusion-steps", type=int, default=300)
    parser.add_argument("--sampling-steps", type=int, default=30)
    parser.add_argument("--tile-encode-chunk", type=int, default=192)
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--graph-top-k", type=int, default=16)
    parser.add_argument("--graph-temperature", type=float, default=0.35)
    parser.add_argument("--denoise-batch-size", type=int, default=576)

    parser.add_argument("--dev-offset", type=int, default=0)
    parser.add_argument(
        "--dev-split",
        choices=("assembly_incremental_gate", "assembly_cal"),
        default="assembly_incremental_gate",
    )
    parser.add_argument("--dev-sources", type=int, default=8)
    parser.add_argument("--dev-replicas", type=int, default=2)
    parser.add_argument("--gate-bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--gate-bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--qap-iterations", type=int, default=25)
    parser.add_argument("--qap-restarts", type=int, default=2)
    parser.add_argument("--qap-boundary-weight", type=float, default=0.05)
    parser.add_argument("--qap-refine-swaps", type=int, default=8)
    parser.add_argument("--gate-min-adjacency-gain", type=float, default=0.002)
    parser.add_argument("--gate-min-ssim-gain", type=float, default=0.001)
    parser.add_argument("--gate-min-positive-source-fraction", type=float, default=0.50)
    parser.add_argument("--log-every", type=int, default=8)
    return parser


def parse_args() -> argparse.Namespace:
    args = _parser().parse_args()
    positive_ints = (
        "train_sources",
        "epochs",
        "gradient_accumulation",
        "max_optimizer_steps",
        "model_dim",
        "cnn_channels",
        "layers",
        "heads",
        "feedforward_dim",
        "diffusion_steps",
        "sampling_steps",
        "tile_encode_chunk",
        "graph_top_k",
        "denoise_batch_size",
        "dev_sources",
        "dev_replicas",
        "gate_bootstrap_resamples",
        "amp_max_consecutive_skips",
        "amp_max_total_skips",
        "qap_iterations",
        "qap_restarts",
        "log_every",
    )
    for name in positive_ints:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.sampling_steps > args.diffusion_steps:
        raise ValueError("sampling steps cannot exceed diffusion steps")
    if min(args.train_offset, args.dev_offset, args.qap_refine_swaps) < 0:
        raise ValueError("source offsets and qap-refine-swaps must be non-negative")
    if not 1 <= args.graph_top_k < TILE_COUNT:
        raise ValueError("graph top-k must be in [1,575]")
    if args.graph_temperature <= 0:
        raise ValueError("graph temperature must be positive")
    if not 0.0 <= args.baseline_condition_dropout < 1.0:
        raise ValueError("baseline condition dropout must be in [0,1)")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
        or not math.isfinite(args.grad_clip)
        or args.grad_clip <= 0
        or not math.isfinite(args.structure_weight)
        or args.structure_weight < 0
    ):
        raise ValueError("optimizer/loss arguments are invalid")
    if args.gate_min_adjacency_gain <= 0 or args.gate_min_ssim_gain <= 0:
        raise ValueError("strict gate deltas must be greater than zero")
    if not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0:
        raise ValueError("--amp-init-scale must be finite and positive")
    if not 0.0 < args.gate_bootstrap_confidence < 1.0:
        raise ValueError("--gate-bootstrap-confidence must be in (0,1)")
    if not 0.0 < args.gate_min_positive_source_fraction <= 1.0:
        raise ValueError("positive-source fraction must be in (0,1]")
    if args.mode in {"evaluate", "pilot"} and args.dev_sources < 8:
        raise ValueError("evaluation and pilot modes require at least 8 development sources")
    return args


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    distributed: bool

    @property
    def primary(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class InputOnlyEvidence:
    graph: np.ndarray
    hbt_score: Any
    soft_layout: np.ndarray
    w1_score: Any | None = None
    w1_qap_layout: np.ndarray | None = None
    w4_score: Any | None = None
    w4_qap_layout: np.ndarray | None = None
    hbt_qap_layout: np.ndarray | None = None
    diagnostics: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TrainingExample:
    source_name: str
    panel: str
    panel_seed: int
    raw_tiles: np.ndarray
    restored_tiles: np.ndarray
    graph: np.ndarray
    baseline_kind: str
    baseline_positions: torch.Tensor
    target_positions: torch.Tensor


def _init_runtime(requested: str) -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP pilot requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        return Runtime(torch.device("cuda", local_rank), rank, local_rank, world_size, True)
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)
    return Runtime(device, rank, local_rank, world_size, False)


def _cleanup(runtime: Runtime | None) -> None:
    if runtime is not None and runtime.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _barrier(runtime: Runtime) -> None:
    if runtime.distributed:
        torch.distributed.barrier()


def _print(runtime: Runtime, payload: Mapping[str, Any]) -> None:
    if runtime.primary:
        print(json.dumps(dict(payload), sort_keys=True), flush=True)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _dataset_slice_sha256(data_root: Path, names: list[str]) -> str:
    """Hash the ordered source names and actual training image bytes."""

    digest = hashlib.sha256()
    for name in names:
        path = data_root / "train" / "targets" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _current_code_hashes() -> dict[str, str]:
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in CODE_PATHS}


def _seed_everything(seed: int, rank: int) -> None:
    """Seed this process before any trainable module is constructed."""

    resolved = int(seed) + int(rank)
    random.seed(resolved)
    np.random.seed(resolved % (2**32 - 1))
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)


def _configure_determinism() -> dict[str, Any]:
    """Configure and report trajectory-affecting deterministic backend flags."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return _determinism_contract()


def _determinism_contract() -> dict[str, Any]:
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


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
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
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
            raise ValueError(
                "checkpoint CUDA RNG state count does not match visible CUDA devices"
            )
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(value, dtype=torch.uint8).cpu() for value in cuda_states]
        )
    elif cuda_states:
        raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")


def _gather_rank_rng_states(runtime: Runtime) -> list[dict[str, Any]]:
    local = _capture_rng_state()
    if not runtime.distributed:
        return [local]
    gathered: list[Any] = [None] * runtime.world_size
    torch.distributed.all_gather_object(gathered, local)
    if any(not isinstance(value, dict) for value in gathered):
        raise RuntimeError("failed to gather every DDP rank RNG state")
    return [dict(value) for value in gathered]


def _runtime_resume_contract(
    runtime: Runtime,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "rank": runtime.rank,
        "device_type": runtime.device.type,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp_enabled": bool(amp_enabled),
        "amp_dtype": str(amp_dtype),
        "determinism": _determinism_contract(),
    }
    if runtime.device.type == "cuda":
        contract.update(
            {
                "device_name": torch.cuda.get_device_name(runtime.device),
                "device_capability": list(torch.cuda.get_device_capability(runtime.device)),
            }
        )
    return contract


def _gather_runtime_resume_contracts(
    runtime: Runtime,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    local = _runtime_resume_contract(
        runtime,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    if not runtime.distributed:
        return [local]
    gathered: list[Any] = [None] * runtime.world_size
    torch.distributed.all_gather_object(gathered, local)
    if any(not isinstance(value, dict) for value in gathered):
        raise RuntimeError("failed to gather every DDP rank runtime contract")
    return [dict(value) for value in gathered]


def _filename_qap_seed(name: str) -> int:
    """Authoritative submission seed: filename SHA256 first4 little-endian + 7001."""

    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little") + 7001


def _select_warm_layout(evidence: InputOnlyEvidence, kind: str) -> np.ndarray:
    if kind == "softcycle":
        layout = evidence.soft_layout
    elif kind == "w4-qap":
        if evidence.w4_qap_layout is None:
            raise ValueError("w4-qap warm start requires QAP evidence")
        layout = evidence.w4_qap_layout
    else:
        raise ValueError("warm-start layout must be softcycle or w4-qap")
    values = np.asarray(layout, dtype=np.int32)
    if values.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(values), np.arange(TILE_COUNT, dtype=np.int32)
    ):
        raise ValueError(f"invalid {kind} warm-start permutation")
    return values.copy()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (GRID * TILE, GRID * TILE, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _tiles_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    if values.shape != (TILE_COUNT, TILE, TILE, 3) or values.dtype != np.uint8:
        raise ValueError("expected uint8 576x20x20x3 tiles")
    return torch.from_numpy(
        np.ascontiguousarray(values.transpose(0, 3, 1, 2))
    ).to(device=device, dtype=torch.float32).div_(255.0).unsqueeze(0)


def _config(args: argparse.Namespace) -> PositionalDiffusionConfig:
    return PositionalDiffusionConfig(
        model_dim=args.model_dim,
        cnn_channels=args.cnn_channels,
        layers=args.layers,
        heads=args.heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        diffusion_steps=args.diffusion_steps,
        tile_encode_chunk=args.tile_encode_chunk,
        activation_checkpointing=args.activation_checkpointing,
    )


def _amp_settings(args: argparse.Namespace, runtime: Runtime) -> tuple[bool, torch.dtype]:
    if runtime.device.type != "cuda" or args.amp == "none":
        return False, torch.float32
    if args.amp == "bf16":
        return True, torch.bfloat16
    if args.amp == "auto":
        major, _ = torch.cuda.get_device_capability(runtime.device)
        return True, torch.bfloat16 if major >= 8 else torch.float16
    return True, torch.float16


def _autocast(runtime: Runtime, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=runtime.device.type, dtype=dtype)


def _hardware_record(runtime: Runtime) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rank": runtime.rank,
        "device": str(runtime.device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if runtime.device.type == "cuda":
        properties = torch.cuda.get_device_properties(runtime.device)
        record.update(
            {
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(runtime.device)),
                "total_memory_bytes": int(properties.total_memory),
                "cuda_version": torch.version.cuda,
            }
        )
        # A reported GPU is insufficient; execute a real operation.
        probe = torch.randn(128, 128, device=runtime.device, dtype=torch.float16)
        record["fp16_matmul_checksum"] = float((probe @ probe).float().mean().cpu())
    return record


def _all_hardware(runtime: Runtime) -> list[dict[str, Any]]:
    local = _hardware_record(runtime)
    if not runtime.distributed:
        return [local]
    gathered: list[Any] = [None] * runtime.world_size
    torch.distributed.all_gather_object(gathered, local)
    return [dict(value) for value in gathered]


def _nvidia_smi() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _strong_corrupt_tiles(
    tiles: np.ndarray,
    *,
    rng: np.random.Generator,
    severity: float,
) -> np.ndarray:
    """Input-only task corruption plus benchmark-style edge/content erosion."""

    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0,1]")
    values = tiles.astype(np.float32) / 255.0
    count = len(values)
    contrast = rng.uniform(1.0 - 0.22 * severity, 1.0 + 0.22 * severity, count)
    brightness = rng.uniform(-0.10 * severity, 0.10 * severity, count)
    gray = values.mean(axis=3, keepdims=True)
    saturation = rng.uniform(1.0 - 0.25 * severity, 1.0 + 0.25 * severity, count)
    values = gray + saturation[:, None, None, None] * (values - gray)
    tile_mean = values.mean(axis=(1, 2), keepdims=True)
    values = tile_mean + contrast[:, None, None, None] * (values - tile_mean)
    values += brightness[:, None, None, None]
    noise_sigma = rng.uniform(0.0, 0.055 * severity, count)
    values += rng.normal(size=values.shape).astype(np.float32) * noise_sigma[:, None, None, None]
    output = np.clip(values * 255.0, 0, 255).astype(np.uint8)

    blur_mask = rng.random(count) < (0.08 + 0.20 * severity)
    for index in np.flatnonzero(blur_mask):
        sigma = float(rng.uniform(0.25, 1.2 * max(severity, 0.1)))
        output[index] = cv2.GaussianBlur(output[index], (3, 3), sigmaX=sigma)

    # The exact independent_libjpeg engine already JPEG-corrupts every tile.
    # A small additional subset avoids making Python JPEG encoding the runtime bottleneck.
    jpeg_mask = rng.random(count) < (0.01 + 0.035 * severity)
    for index in np.flatnonzero(jpeg_mask):
        buffer = io.BytesIO()
        Image.fromarray(output[index]).save(
            buffer,
            format="JPEG",
            quality=int(rng.integers(25, 86)),
            subsampling=2,
        )
        buffer.seek(0)
        with Image.open(buffer) as image:
            output[index] = np.asarray(image.convert("RGB"), dtype=np.uint8)

    # arXiv:2507.07828 replaces the outer two rows/columns of selected pieces.
    eroded = rng.random(count) < (0.04 + 0.22 * severity)
    donor_pixels = output.reshape(-1, 3)
    for index in np.flatnonzero(eroded):
        side = int(rng.integers(4))
        if side < 2:
            donor = donor_pixels[rng.integers(len(donor_pixels), size=(2, TILE))]
            if side == 0:
                output[index, :2, :, :] = donor
            else:
                output[index, -2:, :, :] = donor
        else:
            donor = donor_pixels[rng.integers(len(donor_pixels), size=(TILE, 2))]
            if side == 2:
                output[index, :, :2, :] = donor
            else:
                output[index, :, -2:, :] = donor

    # Paint-flaking/content loss is independent per piece and never uses the target.
    flake_probability = 0.002 + 0.035 * severity
    flake_mask = rng.random((count, TILE, TILE, 1)) < flake_probability
    fill = rng.integers(0, 256, size=(count, 1, 1, 3), dtype=np.uint8)
    output = np.where(flake_mask, fill, output).astype(np.uint8)
    return np.ascontiguousarray(output)


def _augment_restorer_view(
    restored: np.ndarray,
    raw: np.ndarray,
    *,
    rng: np.random.Generator,
    severity: float,
) -> np.ndarray:
    """Simulate imperfect denoiser behaviour without access to clean pixels."""

    restored_f = restored.astype(np.float32)
    raw_f = raw.astype(np.float32)
    blend = rng.uniform(0.0, 0.30 * severity, len(restored))[:, None, None, None]
    values = restored_f * (1.0 - blend) + raw_f * blend
    gain = rng.uniform(1.0 - 0.07 * severity, 1.0 + 0.07 * severity, len(restored))
    offset = rng.uniform(-7.0 * severity, 7.0 * severity, len(restored))
    values = values * gain[:, None, None, None] + offset[:, None, None, None]
    return np.ascontiguousarray(np.clip(values, 0, 255).round().astype(np.uint8))


@torch.inference_mode()
def _input_only_evidence(
    restored_tiles: np.ndarray,
    *,
    hbt_model: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    source_name: str,
    qap_mode: str,
) -> InputOnlyEvidence:
    """Freeze relative graph/layouts without accepting any target argument."""

    if qap_mode not in {"none", "w4", "comparators"}:
        raise ValueError("qap_mode must be none, w4, or comparators")

    started = time.perf_counter()
    hbt, _ = learned_compatibility(
        hbt_model,
        restored_tiles,
        device=device,
        name="input_only_hbt",
    )
    graph = compatibility_to_relative_graph(
        hbt.right,
        hbt.down,
        top_k=args.graph_top_k,
        temperature=args.graph_temperature,
    )
    soft = soft_cycle_component_solver(
        hbt,
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        reciprocal_weight=0.35,
        loop_weight=1.0,
    )
    diagnostics: dict[str, Any] = {
        "soft_accepted_edges": int(soft.accepted_edges),
        "soft_proposed_edges": int(soft.proposed_edges),
        "soft_component_sizes": [int(value) for value in soft.component_sizes],
        "graph_top_k": int(args.graph_top_k),
        "targets_opened": False,
    }
    if qap_mode == "none":
        diagnostics["seconds"] = time.perf_counter() - started
        return InputOnlyEvidence(graph, hbt, soft.position_to_slot.copy(), diagnostics=diagnostics)

    bank = build_classical_score_bank(restored_tiles, prefix="denoised", chunk_size=64)
    c1_names = [
        name for name in sorted(bank)
        if name.startswith("denoised_") and not name.endswith("_c2")
    ]
    c1 = fuse_ranked_scores(bank, names=c1_names, name="denoised_C1_equal_rank")
    bank[c1.name] = c1
    bank[hbt.name] = hbt
    w1 = fuse_ranked_scores(
        bank,
        names=[c1.name, hbt.name],
        name="input_only_C1_HBTw1",
    )
    w4 = fuse_ranked_scores(
        bank,
        names=[c1.name, hbt.name],
        weights={hbt.name: 4.0},
        name="input_only_C1_HBTw4",
    )
    seed = _filename_qap_seed(source_name)
    common = dict(
        initial=soft.position_to_slot,
        iterations=args.qap_iterations,
        restarts=args.qap_restarts,
        seed=seed,
        boundary_weight=args.qap_boundary_weight,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=args.qap_refine_swaps,
    )
    w4_result = directional_qap(w4, **common)
    w1_result = directional_qap(w1, **common) if qap_mode == "comparators" else None
    hbt_result = directional_qap(hbt, **common) if qap_mode == "comparators" else None
    diagnostics.update(
        {
            "seed": seed,
            "seed_formula": "filename_sha256_first4_le + 7001",
            "w4_objective": float(w4_result.objective),
            "w4_restart": int(w4_result.restart),
            "qap_iterations": int(args.qap_iterations),
            "qap_restarts": int(args.qap_restarts),
            "qap_boundary_weight": float(args.qap_boundary_weight),
            "qap_mode": qap_mode,
            "seconds": time.perf_counter() - started,
        }
    )
    if w1_result is not None and hbt_result is not None:
        diagnostics.update(
            {
                "w1_objective": float(w1_result.objective),
                "w1_restart": int(w1_result.restart),
                "hbt_objective": float(hbt_result.objective),
                "hbt_restart": int(hbt_result.restart),
            }
        )
    return InputOnlyEvidence(
        graph=graph,
        hbt_score=hbt,
        soft_layout=soft.position_to_slot.copy(),
        w1_score=w1 if w1_result is not None else None,
        w1_qap_layout=w1_result.position_to_slot.copy() if w1_result is not None else None,
        w4_score=w4,
        w4_qap_layout=w4_result.position_to_slot.copy(),
        hbt_qap_layout=hbt_result.position_to_slot.copy() if hbt_result is not None else None,
        diagnostics=diagnostics,
    )


def _restore(
    restorer: nn.Module,
    raw_tiles: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    return restore_tiles_uint8(restorer, raw_tiles, device, batch_size=batch_size)


def _curriculum(epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return 1.0
    return float(0.35 + 0.65 * epoch / float(epochs - 1))


def _prepare_training_example(
    name: str,
    *,
    epoch: int,
    args: argparse.Namespace,
    data_root: Path,
    restorer: nn.Module,
    hbt_model: nn.Module,
    device: torch.device,
) -> TrainingExample:
    target = _read_rgb(data_root / "train" / "targets" / name)
    panel_selector = per_source_seed(args.seed, "posdiff:engine", name, epoch) % 2
    panel_name = "primary_kornia" if panel_selector == 0 else "independent_libjpeg"
    panel_seed = per_source_seed(args.seed, f"posdiff:train:{panel_name}", name, epoch)
    panel = make_exact_panel(target, panel=panel_name, seed=panel_seed)
    severity = _curriculum(epoch, args.epochs)
    rng = np.random.default_rng(
        per_source_seed(args.seed, "posdiff:augmentation", name, epoch)
    )
    raw = _strong_corrupt_tiles(panel.slot_tiles, rng=rng, severity=severity)
    restored = _restore(
        restorer, raw, device=device, batch_size=args.denoise_batch_size
    )
    restored = _augment_restorer_view(restored, raw, rng=rng, severity=severity)
    evidence = _input_only_evidence(
        restored,
        hbt_model=hbt_model,
        device=device,
        args=args,
        source_name=name,
        qap_mode="w4" if args.warm_start_layout == "w4-qap" else "none",
    )
    warm_layout = _select_warm_layout(evidence, args.warm_start_layout)
    baseline = layout_to_tile_positions(warm_layout, GRID, GRID)
    grid = normalized_grid_positions(GRID, GRID)
    target_positions = grid[torch.from_numpy(panel.slot_to_target.astype(np.int64))]
    return TrainingExample(
        source_name=name,
        panel=panel_name,
        panel_seed=panel_seed,
        raw_tiles=raw,
        restored_tiles=restored,
        graph=evidence.graph,
        baseline_kind=args.warm_start_layout,
        baseline_positions=baseline,
        target_positions=target_positions,
    )


def _example_tensors(example: TrainingExample, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "raw": _tiles_tensor(example.raw_tiles, device),
        "restored": _tiles_tensor(example.restored_tiles, device),
        "graph": torch.from_numpy(example.graph).to(device=device, dtype=torch.float32).unsqueeze(0),
        "baseline": example.baseline_positions.to(device=device, dtype=torch.float32).unsqueeze(0),
        "target": example.target_positions.to(device=device, dtype=torch.float32).unsqueeze(0),
    }


def _reduce_epoch(values: dict[str, float], runtime: Runtime) -> dict[str, float]:
    names = sorted(values)
    tensor = torch.tensor([values[name] for name in names], device=runtime.device, dtype=torch.float64)
    if runtime.distributed:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return {name: float(tensor[index].cpu()) for index, name in enumerate(names)}


def _all_ranks_finite(value: torch.Tensor, runtime: Runtime) -> bool:
    flag = torch.tensor(
        1 if bool(torch.isfinite(value.detach()).all()) else 0,
        device=runtime.device,
        dtype=torch.int32,
    )
    if runtime.distributed:
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def _all_ranks_same_int(value: int, runtime: Runtime) -> bool:
    minimum = torch.tensor(int(value), device=runtime.device, dtype=torch.int64)
    maximum = minimum.clone()
    if runtime.distributed:
        torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    return int(minimum.item()) == int(maximum.item())


def _all_ranks_same_float(value: float, runtime: Runtime) -> bool:
    minimum = torch.tensor(float(value), device=runtime.device, dtype=torch.float64)
    maximum = minimum.clone()
    if runtime.distributed:
        torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    return float(minimum.item()) == float(maximum.item())


def _bounded_skip_state(
    *,
    total_skips: int,
    consecutive_skips: int,
    skipped: bool,
    max_total: int,
    max_consecutive: int,
) -> tuple[int, int, bool]:
    """Pure counter transition used by the AMP overflow recovery path."""

    if skipped:
        total_skips += 1
        consecutive_skips += 1
    else:
        consecutive_skips = 0
    exceeded = total_skips > max_total or consecutive_skips > max_consecutive
    return total_skips, consecutive_skips, exceeded


def _checkpoint_metadata(
    *,
    args: argparse.Namespace,
    train_names: list[str],
    epoch: int,
    optimizer_steps: int,
    restorer_metadata: Mapping[str, Any],
    hbt_metadata: Mapping[str, Any],
    train_data_sha256: str | None = None,
    attempted_optimizer_steps: int | None = None,
    skipped_optimizer_updates: int = 0,
) -> dict[str, Any]:
    manifest = (
        (REPO_ROOT / args.manifest).resolve()
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest)
    )
    quarantine = (
        (REPO_ROOT / args.quarantine).resolve()
        if not Path(args.quarantine).is_absolute()
        else Path(args.quarantine)
    )
    attempted = int(optimizer_steps if attempted_optimizer_steps is None else attempted_optimizer_steps)
    return {
        "schema_version": 1,
        "experiment": "bounded_positional_diffusion_signal_pilot",
        "epoch": int(epoch),
        "optimizer_steps": int(optimizer_steps),
        "attempted_optimizer_steps": attempted,
        "successful_optimizer_steps": int(optimizer_steps),
        "skipped_optimizer_updates": int(skipped_optimizer_updates),
        "seed": int(args.seed),
        "train_split": "edge_train whole-source",
        "train_source_count": len(train_names),
        "train_source_names": list(train_names),
        "train_source_names_sha256": _names_sha256(train_names),
        "train_data_sha256": train_data_sha256,
        "training_arguments": dict(vars(args)),
        "corruption_engines": ["primary_kornia", "independent_libjpeg"],
        "additional_augmentation": "tone/noise/blur/jpeg/edge erosion/content flaking",
        "relative_graph": "input-only HBT rank graph; no truth confidence",
        "warm_start_layout": args.warm_start_layout,
        "warm_start_alignment": "identical layout family in train and evaluation",
        "training_supervision": "synthetic exact permutation from clean train source",
        "development_targets_opened_during_training": False,
        "competition_test_targets_opened": False,
        "denoiser": dict(restorer_metadata),
        "hbt": dict(hbt_metadata),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "quarantine": str(quarantine),
        "quarantine_sha256": _sha256(quarantine),
        "code_sha256": _current_code_hashes(),
        "determinism": _determinism_contract(),
        "primary_sources": [
            "https://arxiv.org/abs/2303.11120",
            "https://github.com/IIT-PAVIS/Positional_Diffusion",
            "https://arxiv.org/abs/2507.07828",
        ],
        "safe_for_submission": False,
        "submission_ready": False,
    }


def _validate_resume_contract(
    payload: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    train_names: list[str],
    restorer_metadata: Mapping[str, Any],
    hbt_metadata: Mapping[str, Any],
    runtime_contracts: list[dict[str, Any]],
    train_data_sha256: str | None = None,
) -> None:
    """Reject any trajectory-affecting drift before loading training state."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("resume checkpoint metadata must be a mapping")
    saved_arguments = metadata.get("training_arguments")
    if not isinstance(saved_arguments, Mapping):
        raise ValueError("resume checkpoint lacks recorded training arguments")

    mismatches: list[str] = []
    for name in RESUME_TRAINING_ARGUMENTS:
        saved = saved_arguments.get(name, object())
        current = getattr(args, name)
        if saved != current:
            mismatches.append(f"argument {name}: checkpoint={saved!r}, current={current!r}")

    expected_names_hash = _names_sha256(train_names)
    if metadata.get("train_source_names_sha256") != expected_names_hash:
        mismatches.append("whole-source training slice hash")
    if list(metadata.get("train_source_names", [])) != list(train_names):
        mismatches.append("whole-source training slice order")
    if int(metadata.get("seed", -1)) != int(args.seed):
        mismatches.append("seed")
    if metadata.get("train_data_sha256") != train_data_sha256:
        mismatches.append("training image byte hash")
    if metadata.get("determinism") != _determinism_contract():
        mismatches.append("deterministic backend flags")

    manifest = (
        (REPO_ROOT / args.manifest).resolve()
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest)
    )
    quarantine = (
        (REPO_ROOT / args.quarantine).resolve()
        if not Path(args.quarantine).is_absolute()
        else Path(args.quarantine)
    )
    if metadata.get("manifest_sha256") != _sha256(manifest):
        mismatches.append("manifest content hash")
    if metadata.get("quarantine_sha256") != _sha256(quarantine):
        mismatches.append("quarantine content hash")
    if metadata.get("code_sha256") != _current_code_hashes():
        mismatches.append("transitive code hashes")

    for label, current_metadata in (
        ("denoiser", restorer_metadata),
        ("hbt", hbt_metadata),
    ):
        saved_metadata = metadata.get(label)
        saved_hash = (
            saved_metadata.get("checkpoint_sha256")
            if isinstance(saved_metadata, Mapping)
            else None
        )
        if saved_hash is None or saved_hash != current_metadata.get("checkpoint_sha256"):
            mismatches.append(f"{label} checkpoint hash")

    training_state = payload.get("training_state")
    if not isinstance(training_state, Mapping):
        mismatches.append("training_state")
    elif training_state.get("runtime_contracts_by_rank") != runtime_contracts:
        mismatches.append("per-rank runtime/AMP contract")

    if mismatches:
        raise ValueError(
            "resume checkpoint is not an exact continuation: " + "; ".join(mismatches)
        )


def _upstream_exposure_audit(
    dev_names: list[str],
    *,
    manifest: Path,
    quarantine: Path,
    hbt_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed if development sources were exposed to frozen upstream models."""

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    quarantine_payload = json.loads(quarantine.read_text(encoding="utf-8"))
    splits = manifest_payload.get("splits", {})
    denoiser_exposure = set(map(str, splits.get("train", [])))
    denoiser_exposure.update(map(str, quarantine_payload.get("quarantine_names", [])))
    hbt_exposure = set(map(str, hbt_metadata.get("train_names", [])))
    hbt_exposure.update(map(str, hbt_metadata.get("val_names", [])))
    requested = set(dev_names)
    denoiser_overlap = sorted(requested & denoiser_exposure)
    hbt_overlap = sorted(requested & hbt_exposure)
    if denoiser_overlap or hbt_overlap:
        raise RuntimeError(
            "development has frozen-upstream exposure: "
            f"denoiser={denoiser_overlap}, hbt={hbt_overlap}"
        )
    return {
        "development_source_count": len(dev_names),
        "denoiser_exposure_definition": "manifest train plus versioned denoise quarantine",
        "hbt_exposure_definition": "checkpoint metadata train_names plus val_names",
        "denoiser_exposure_source_count": len(denoiser_exposure),
        "hbt_exposure_source_count": len(hbt_exposure),
        "denoiser_overlap_count": 0,
        "hbt_overlap_count": 0,
        "zero_upstream_exposure_asserted": True,
    }


def _validate_evaluation_contract(
    payload: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    config: PositionalDiffusionConfig,
    restorer_metadata: Mapping[str, Any],
    hbt_metadata: Mapping[str, Any],
    manifest: Path,
    quarantine: Path,
) -> dict[str, Any]:
    """Validate a standalone checkpoint against all inference-affecting assets."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("evaluation checkpoint metadata must be a mapping")
    saved_arguments = metadata.get("training_arguments")
    if not isinstance(saved_arguments, Mapping):
        raise ValueError("evaluation checkpoint lacks training arguments")
    mismatches: list[str] = []
    if payload.get("safe_for_submission") is not False:
        mismatches.append("top-level safe_for_submission=false")
    if PositionalDiffusionConfig(**payload.get("model_config", {})) != config:
        mismatches.append("model configuration")
    if int(metadata.get("seed", -1)) != int(args.seed):
        mismatches.append("seed")
    for name in (
        "warm_start_layout",
        "sampling_steps",
        "amp",
        "graph_top_k",
        "graph_temperature",
        "qap_iterations",
        "qap_restarts",
        "qap_boundary_weight",
        "qap_refine_swaps",
    ):
        if saved_arguments.get(name, object()) != getattr(args, name):
            mismatches.append(f"argument {name}")
    if metadata.get("warm_start_layout") != args.warm_start_layout:
        mismatches.append("warm-start family")
    if metadata.get("manifest_sha256") != _sha256(manifest):
        mismatches.append("manifest hash")
    if metadata.get("quarantine_sha256") != _sha256(quarantine):
        mismatches.append("quarantine hash")
    if metadata.get("code_sha256") != _current_code_hashes():
        mismatches.append("transitive code hashes")
    if metadata.get("determinism") != _determinism_contract():
        mismatches.append("deterministic backend flags")
    for label, current in (("denoiser", restorer_metadata), ("hbt", hbt_metadata)):
        saved = metadata.get(label)
        if (
            not isinstance(saved, Mapping)
            or saved.get("checkpoint_sha256") != current.get("checkpoint_sha256")
        ):
            mismatches.append(f"{label} checkpoint hash")
    if mismatches:
        raise ValueError(
            "evaluation checkpoint contract mismatch: " + "; ".join(mismatches)
        )
    return {
        "validated": True,
        "seed": int(args.seed),
        "warm_start_layout": args.warm_start_layout,
        "assets_match": True,
        "graph_and_qap_arguments_match": True,
        "code_hashes_match": True,
        "loaded_checkpoint_path": payload.get("loaded_checkpoint_path"),
        "used_previous_fallback": bool(payload.get("used_previous_fallback", False)),
        "safe_for_submission": False,
    }


def _train(
    model: PositionalDiffusionNet,
    diffusion: GaussianPositionDiffusion,
    *,
    args: argparse.Namespace,
    runtime: Runtime,
    train_names: list[str],
    data_root: Path,
    output_dir: Path,
    restorer: nn.Module,
    hbt_model: nn.Module,
    restorer_metadata: Mapping[str, Any],
    hbt_metadata: Mapping[str, Any],
    train_data_sha256: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    resume_payload: Mapping[str, Any] | None = None,
) -> tuple[PositionalDiffusionNet, dict[str, Any]]:
    if len(train_names) % runtime.world_size:
        raise ValueError("train source count must be divisible by DDP world size")
    local_source_count = len(train_names) // runtime.world_size
    if local_source_count % args.gradient_accumulation:
        raise ValueError(
            "per-rank source count must be divisible by gradient accumulation "
            "for exact epoch-boundary resume"
        )
    updates_per_epoch = local_source_count // args.gradient_accumulation
    total_planned_updates = args.epochs * updates_per_epoch
    if (
        args.max_optimizer_steps < total_planned_updates
        and args.max_optimizer_steps % updates_per_epoch
    ):
        raise ValueError(
            "max optimizer steps must stop on an epoch boundary for exact resume"
        )
    model.to(runtime.device)
    diffusion.to(runtime.device)
    train_model: nn.Module = model
    if runtime.distributed:
        train_model = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
        init_scale=args.amp_init_scale,
    )
    runtime_contracts = _gather_runtime_resume_contracts(
        runtime,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    start_epoch = 0
    attempted_optimizer_steps = 0
    successful_optimizer_steps = 0
    skipped_updates = 0
    consecutive_skipped_updates = 0
    epoch_records: list[dict[str, Any]] = []
    resumed_from: str | None = None
    resume_used_previous_fallback = False
    if resume_payload is not None:
        _validate_resume_contract(
            resume_payload,
            args=args,
            train_names=train_names,
            restorer_metadata=restorer_metadata,
            hbt_metadata=hbt_metadata,
            runtime_contracts=runtime_contracts,
            train_data_sha256=train_data_sha256,
        )
        missing = {
            "optimizer_state",
            "scaler_state",
            "training_state",
        } - set(resume_payload)
        if missing:
            raise ValueError(f"resume checkpoint is missing {sorted(missing)}")
        training_state = resume_payload["training_state"]
        if not isinstance(training_state, Mapping):
            raise ValueError("resume training_state must be a mapping")
        if int(training_state.get("world_size", -1)) != runtime.world_size:
            raise ValueError("resume checkpoint world_size does not match this launch")
        if int(training_state.get("gradient_accumulation", -1)) != args.gradient_accumulation:
            raise ValueError("resume gradient accumulation does not match checkpoint")
        rng_states = training_state.get("rng_states_by_rank")
        if not isinstance(rng_states, (list, tuple)) or len(rng_states) != runtime.world_size:
            raise ValueError("resume checkpoint lacks one RNG state per rank")
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scaler.load_state_dict(resume_payload["scaler_state"])
        start_epoch = int(training_state.get("next_epoch", -1))
        attempted_optimizer_steps = int(training_state.get("attempted_optimizer_steps", -1))
        successful_optimizer_steps = int(training_state.get("successful_optimizer_steps", -1))
        skipped_updates = int(training_state.get("skipped_optimizer_updates", -1))
        consecutive_skipped_updates = int(
            training_state.get("consecutive_skipped_optimizer_updates", -1)
        )
        saved_history = training_state.get("epoch_history")
        if not isinstance(saved_history, list):
            raise ValueError("resume checkpoint lacks cumulative epoch history")
        epoch_records = [dict(value) for value in saved_history]
        if (
            not 0 <= start_epoch <= args.epochs
            or min(
                attempted_optimizer_steps,
                successful_optimizer_steps,
                skipped_updates,
                consecutive_skipped_updates,
            )
            < 0
        ):
            raise ValueError("resume epoch/optimizer step is out of range")
        if attempted_optimizer_steps != start_epoch * updates_per_epoch:
            raise ValueError(
                "resume attempted optimizer cursor is inconsistent with the epoch boundary"
            )
        if successful_optimizer_steps + skipped_updates != attempted_optimizer_steps:
            raise ValueError("resume attempted/successful/skipped counters are inconsistent")
        if len(epoch_records) != start_epoch:
            raise ValueError("resume cumulative history is inconsistent with next_epoch")
        if int(training_state.get("completed_epoch", -2)) != start_epoch - 1:
            raise ValueError("resume completed/next epoch markers are inconsistent")
        if training_state.get("capture_point") not in {
            "epoch boundary after optimizer update and before checkpoint save",
            "final epoch boundary before final checkpoint save",
        }:
            raise ValueError("resume checkpoint was not captured at a supported boundary")
        resume_metadata = resume_payload["metadata"]
        if (
            int(resume_metadata.get("epoch", -2)) != start_epoch - 1
            or int(resume_metadata.get("successful_optimizer_steps", -1))
            != successful_optimizer_steps
            or int(resume_metadata.get("attempted_optimizer_steps", -1))
            != attempted_optimizer_steps
            or int(resume_metadata.get("skipped_optimizer_updates", -1))
            != skipped_updates
        ):
            raise ValueError("resume metadata disagrees with training_state")
        _restore_rng_state(rng_states[runtime.rank])
        resumed_from = str(
            resume_payload.get("loaded_checkpoint_path", args.resume_checkpoint)
        )
        resume_used_previous_fallback = bool(
            resume_payload.get("used_previous_fallback", False)
        )

    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)

    started = time.perf_counter()
    last_completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        if attempted_optimizer_steps >= args.max_optimizer_steps:
            break
        ordering = list(train_names)
        random.Random(args.seed + 10_007 * epoch).shuffle(ordering)
        local_names = ordering[runtime.rank :: runtime.world_size]
        train_model.train()
        epoch_totals = {
            "loss": 0.0,
            "position_loss": 0.0,
            "structure_loss": 0.0,
            "bounds_loss": 0.0,
            "microexamples": 0.0,
            "primary_examples": 0.0,
            "independent_examples": 0.0,
            "softcycle_warm_examples": 0.0,
            "w4_qap_warm_examples": 0.0,
        }
        optimizer.zero_grad(set_to_none=True)
        for group_start in range(0, len(local_names), args.gradient_accumulation):
            if attempted_optimizer_steps >= args.max_optimizer_steps:
                break
            group = local_names[group_start : group_start + args.gradient_accumulation]
            for micro_index, name in enumerate(group):
                example = _prepare_training_example(
                    name,
                    epoch=epoch,
                    args=args,
                    data_root=data_root,
                    restorer=restorer,
                    hbt_model=hbt_model,
                    device=runtime.device,
                )
                tensors = _example_tensors(example, runtime.device)
                dropout_seed = per_source_seed(
                    args.seed, "posdiff:baseline-dropout", name, epoch
                )
                if np.random.default_rng(dropout_seed).random() < args.baseline_condition_dropout:
                    tensors["baseline"].zero_()
                final_micro = micro_index + 1 == len(group)
                sync_context = nullcontext()
                if runtime.distributed and not final_micro:
                    sync_context = train_model.no_sync()  # type: ignore[union-attr]
                with sync_context:
                    with _autocast(runtime, amp_enabled, amp_dtype):
                        losses = diffusion.training_loss(
                            train_model,  # type: ignore[arg-type]
                            tensors["raw"],
                            tensors["restored"],
                            tensors["target"],
                            rows=GRID,
                            columns=GRID,
                            relative_graph=tensors["graph"],
                            baseline_positions=tensors["baseline"],
                            structure_weight=args.structure_weight,
                        )
                        scaled_loss = losses["loss"] / float(len(group))
                    loss_values = torch.stack(
                        [
                            losses["loss"].detach().float(),
                            losses["position_loss"].detach().float(),
                            losses["structure_loss"].detach().float(),
                            losses["bounds_loss"].detach().float(),
                        ]
                    )
                    if not _all_ranks_finite(loss_values, runtime):
                        raise RuntimeError(
                            f"non-finite loss across DDP ranks at epoch={epoch} source={name}"
                        )
                    scaler.scale(scaled_loss).backward()
                epoch_totals["loss"] += float(losses["loss"].detach().float().cpu())
                epoch_totals["position_loss"] += float(losses["position_loss"].cpu())
                epoch_totals["structure_loss"] += float(losses["structure_loss"].cpu())
                epoch_totals["bounds_loss"] += float(losses["bounds_loss"].cpu())
                epoch_totals["microexamples"] += 1.0
                epoch_totals[
                    "primary_examples" if example.panel == "primary_kornia" else "independent_examples"
                ] += 1.0
                epoch_totals[
                    "w4_qap_warm_examples"
                    if example.baseline_kind == "w4-qap"
                    else "softcycle_warm_examples"
                ] += 1.0
            scaler.unscale_(optimizer)
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                train_model.parameters(), args.grad_clip
            )
            gradients_finite = _all_ranks_finite(gradient_norm_tensor, runtime)
            if not gradients_finite and not scaler.is_enabled():
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    f"non-finite gradient norm across DDP ranks at epoch={epoch} "
                    f"attempted_optimizer_step={attempted_optimizer_steps + 1}"
                )
            gradient_norm = (
                float(gradient_norm_tensor.detach().cpu()) if gradients_finite else None
            )
            scale_before = float(scaler.get_scale())
            if not _all_ranks_same_float(scale_before, runtime):
                raise RuntimeError("GradScaler scale diverged across DDP ranks before update")
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if not _all_ranks_same_float(scale_after, runtime):
                raise RuntimeError("GradScaler scale diverged across DDP ranks after update")
            optimizer.zero_grad(set_to_none=True)
            attempted_optimizer_steps += 1
            update_skipped = bool(scaler.is_enabled() and scale_after < scale_before)
            if scaler.is_enabled() and not gradients_finite and not update_skipped:
                raise RuntimeError("GradScaler failed to skip a non-finite synchronized update")
            if not _all_ranks_same_int(int(update_skipped), runtime):
                raise RuntimeError("GradScaler skip decision diverged across DDP ranks")
            skipped_updates, consecutive_skipped_updates, skip_limit_exceeded = (
                _bounded_skip_state(
                    total_skips=skipped_updates,
                    consecutive_skips=consecutive_skipped_updates,
                    skipped=update_skipped,
                    max_total=args.amp_max_total_skips,
                    max_consecutive=args.amp_max_consecutive_skips,
                )
            )
            if update_skipped:
                _print(
                    runtime,
                    {
                        "event": "amp_update_skipped",
                        "epoch": epoch,
                        "attempted_optimizer_step": attempted_optimizer_steps,
                        "successful_optimizer_steps": successful_optimizer_steps,
                        "total_skipped_optimizer_updates": skipped_updates,
                        "consecutive_skipped_optimizer_updates": consecutive_skipped_updates,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                    },
                )
                if skip_limit_exceeded:
                    raise RuntimeError(
                        "GradScaler overflow recovery limit exceeded: "
                        f"total={skipped_updates}/{args.amp_max_total_skips}, "
                        f"consecutive={consecutive_skipped_updates}/"
                        f"{args.amp_max_consecutive_skips}"
                    )
                continue
            successful_optimizer_steps += 1
            if attempted_optimizer_steps % args.log_every == 0:
                _print(
                    runtime,
                    {
                        "event": "train_step",
                        "epoch": epoch,
                        "attempted_optimizer_step": attempted_optimizer_steps,
                        "successful_optimizer_steps": successful_optimizer_steps,
                        "skipped_optimizer_updates": skipped_updates,
                        "rank0_latest_loss": float(losses["loss"].detach().float().cpu()),
                        "gradient_norm": gradient_norm,
                    },
                )

        reduced = _reduce_epoch(epoch_totals, runtime)
        count = max(reduced.pop("microexamples"), 1.0)
        record = {
            "epoch": epoch,
            "attempted_optimizer_steps": attempted_optimizer_steps,
            "successful_optimizer_steps": successful_optimizer_steps,
            "skipped_optimizer_updates": skipped_updates,
            "consecutive_skipped_optimizer_updates": consecutive_skipped_updates,
            "mean_loss": reduced.pop("loss") / count,
            "mean_position_loss": reduced.pop("position_loss") / count,
            "mean_structure_loss": reduced.pop("structure_loss") / count,
            "mean_bounds_loss": reduced.pop("bounds_loss") / count,
            "global_examples": int(count),
            "primary_examples": int(reduced["primary_examples"]),
            "independent_examples": int(reduced["independent_examples"]),
            "softcycle_warm_examples": int(reduced["softcycle_warm_examples"]),
            "w4_qap_warm_examples": int(reduced["w4_qap_warm_examples"]),
        }
        last_completed_epoch = epoch
        epoch_records.append(record)
        _print(runtime, {"event": "epoch_complete", **record})
        rng_states = _gather_rank_rng_states(runtime)
        training_state = {
            "world_size": runtime.world_size,
            "gradient_accumulation": args.gradient_accumulation,
            "completed_epoch": epoch,
            "next_epoch": epoch + 1,
            "attempted_optimizer_steps": attempted_optimizer_steps,
            "successful_optimizer_steps": successful_optimizer_steps,
            "skipped_optimizer_updates": skipped_updates,
            "consecutive_skipped_optimizer_updates": consecutive_skipped_updates,
            "epoch_history": copy.deepcopy(epoch_records),
            "rng_states_by_rank": rng_states,
            "runtime_contracts_by_rank": runtime_contracts,
            "capture_point": "epoch boundary after optimizer update and before checkpoint save",
        }
        if runtime.primary:
            metadata = _checkpoint_metadata(
                args=args,
                train_names=train_names,
                epoch=epoch,
                optimizer_steps=successful_optimizer_steps,
                attempted_optimizer_steps=attempted_optimizer_steps,
                skipped_optimizer_updates=skipped_updates,
                restorer_metadata=restorer_metadata,
                hbt_metadata=hbt_metadata,
                train_data_sha256=train_data_sha256,
            )
            save_positional_diffusion_checkpoint(
                output_dir / LATEST_NAME,
                model,
                metadata=metadata,
                optimizer_state=_to_cpu_tree(optimizer.state_dict()),
                scaler_state=_to_cpu_tree(scaler.state_dict()),
                training_state=_to_cpu_tree(training_state),
                preserve_previous=True,
            )
        _barrier(runtime)

    final_rng_states = _gather_rank_rng_states(runtime)
    final_training_state = {
        "world_size": runtime.world_size,
        "gradient_accumulation": args.gradient_accumulation,
        "completed_epoch": last_completed_epoch,
        "next_epoch": max(start_epoch, last_completed_epoch + 1),
        "attempted_optimizer_steps": attempted_optimizer_steps,
        "successful_optimizer_steps": successful_optimizer_steps,
        "skipped_optimizer_updates": skipped_updates,
        "consecutive_skipped_optimizer_updates": consecutive_skipped_updates,
        "epoch_history": copy.deepcopy(epoch_records),
        "rng_states_by_rank": final_rng_states,
        "runtime_contracts_by_rank": runtime_contracts,
        "capture_point": "final epoch boundary before final checkpoint save",
    }
    if runtime.primary:
        metadata = _checkpoint_metadata(
            args=args,
            train_names=train_names,
            epoch=last_completed_epoch,
            optimizer_steps=successful_optimizer_steps,
            attempted_optimizer_steps=attempted_optimizer_steps,
            skipped_optimizer_updates=skipped_updates,
            restorer_metadata=restorer_metadata,
            hbt_metadata=hbt_metadata,
            train_data_sha256=train_data_sha256,
        )
        save_positional_diffusion_checkpoint(
            output_dir / CHECKPOINT_NAME,
            model,
            metadata=metadata,
            optimizer_state=_to_cpu_tree(optimizer.state_dict()),
            scaler_state=_to_cpu_tree(scaler.state_dict()),
            training_state=_to_cpu_tree(final_training_state),
        )
    _barrier(runtime)
    peak = int(torch.cuda.max_memory_allocated(runtime.device)) if runtime.device.type == "cuda" else 0
    peaks = [peak]
    if runtime.distributed:
        gathered: list[Any] = [None] * runtime.world_size
        torch.distributed.all_gather_object(gathered, peak)
        peaks = [int(value) for value in gathered]
    return model, {
        "attempted_optimizer_steps": attempted_optimizer_steps,
        "successful_optimizer_steps": successful_optimizer_steps,
        "epochs": epoch_records,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes_by_rank": peaks,
        "global_effective_batch": runtime.world_size * args.gradient_accumulation,
        "start_epoch": start_epoch,
        "resumed_from": resumed_from,
        "resume_used_previous_fallback": resume_used_previous_fallback,
        "skipped_optimizer_updates": skipped_updates,
        "consecutive_skipped_optimizer_updates": consecutive_skipped_updates,
        "resume_cursor": {
            "next_epoch": max(start_epoch, last_completed_epoch + 1),
            "attempted_optimizer_steps": attempted_optimizer_steps,
        },
        "train_data_sha256": train_data_sha256,
        "determinism": _determinism_contract(),
    }


def _numeric_mean(records: list[dict[str, Any]], section: str) -> dict[str, float]:
    keys = sorted(
        key
        for key, value in records[0][section].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    return {
        key: float(np.mean([float(record[section][key]) for record in records]))
        for key in keys
    }


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires at least two finite source-level values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "confidence": float(confidence),
        "resamples": int(resamples),
        "unit": "whole source after averaging corruption replicas",
    }


def _source_level_deltas(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_source.setdefault(str(record["source"]), []).append(record)
    adjacency = np.asarray(
        [
            np.mean([item["paired_delta"]["combined_adjacency"] for item in items])
            for _, items in sorted(by_source.items())
        ],
        dtype=np.float64,
    )
    ssim = np.asarray(
        [
            np.mean([item["paired_delta"]["predicted_layout_ssim"] for item in items])
            for _, items in sorted(by_source.items())
        ],
        dtype=np.float64,
    )
    return adjacency, ssim


@torch.inference_mode()
def _evaluate(
    model: PositionalDiffusionNet,
    diffusion: GaussianPositionDiffusion,
    *,
    args: argparse.Namespace,
    runtime: Runtime,
    dev_names: list[str],
    data_root: Path,
    restorer: nn.Module,
    hbt_model: nn.Module,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    panel_order = {"primary_kornia": 0, "independent_libjpeg": 1}
    cells = [
        (panel_name, replica, name)
        for panel_name in panel_order
        for replica in range(args.dev_replicas)
        for name in dev_names
    ]
    local_records: list[dict[str, Any]] = []
    for panel_name, replica, name in cells[runtime.rank :: runtime.world_size]:
        w1_comparator_id = (
            f"qap_w1_b{args.qap_boundary_weight:g}_i{args.qap_iterations}_"
            f"r{args.qap_restarts}_filename_seed"
        )
        clean_target = _read_rgb(data_root / "train" / "targets" / name)
        panel_seed = per_source_seed(
            args.seed, f"posdiff:development:{panel_name}", name, replica
        )
        panel = make_exact_panel(clean_target, panel=panel_name, seed=panel_seed)
        raw = panel.slot_tiles
        restored = _restore(
            restorer,
            raw,
            device=runtime.device,
            batch_size=args.denoise_batch_size,
        )
        evidence = _input_only_evidence(
            restored,
            hbt_model=hbt_model,
            device=runtime.device,
            args=args,
            source_name=name,
            qap_mode="comparators",
        )
        if (
            evidence.w1_qap_layout is None
            or evidence.w4_qap_layout is None
            or evidence.hbt_qap_layout is None
        ):
            raise RuntimeError("comparator QAP layouts were not produced")
        warm_layout = _select_warm_layout(evidence, args.warm_start_layout)
        baseline_positions = layout_to_tile_positions(
            warm_layout,
            GRID,
            GRID,
            device=runtime.device,
        ).unsqueeze(0)
        graph = torch.from_numpy(evidence.graph).to(
            device=runtime.device, dtype=torch.float32
        ).unsqueeze(0)
        with _autocast(runtime, amp_enabled, amp_dtype):
            sample = diffusion.ddim_sample(
                model,
                _tiles_tensor(raw, runtime.device),
                _tiles_tensor(restored, runtime.device),
                rows=GRID,
                columns=GRID,
                relative_graph=graph,
                baseline_positions=baseline_positions,
                sampling_steps=args.sampling_steps,
                initialization="input_layout",
                seed=per_source_seed(
                    args.seed, f"posdiff:ddim:{panel_name}", name, replica
                )
                % (2**63 - 1),
            )
        candidate_layout = sample.projections[0].position_to_tile.copy()
        candidate = layout_metrics(candidate_layout, panel.slot_to_target)
        w1 = layout_metrics(evidence.w1_qap_layout, panel.slot_to_target)
        w4 = layout_metrics(evidence.w4_qap_layout, panel.slot_to_target)
        pure_hbt = layout_metrics(evidence.hbt_qap_layout, panel.slot_to_target)
        candidate.update(predicted_image_metrics(candidate_layout, restored, clean_target))
        w1.update(predicted_image_metrics(evidence.w1_qap_layout, restored, clean_target))
        w4.update(predicted_image_metrics(evidence.w4_qap_layout, restored, clean_target))
        pure_hbt.update(
            predicted_image_metrics(evidence.hbt_qap_layout, restored, clean_target)
        )
        envelope_adjacency = max(
            float(w1["combined_adjacency"]),
            float(w4["combined_adjacency"]),
            float(pure_hbt["combined_adjacency"]),
        )
        envelope_ssim = max(
            float(w1["predicted_layout_ssim"]),
            float(w4["predicted_layout_ssim"]),
            float(pure_hbt["predicted_layout_ssim"]),
        )
        record = {
            "source": name,
            "panel": panel_name,
            "replica": int(replica),
            "panel_seed": int(panel_seed),
            "target_permutation_sha256": _array_sha256(panel.slot_to_target),
            "candidate": candidate,
            "qap_w1_baseline": w1,
            "qap_w1_comparator_id": w1_comparator_id,
            "w4_qap_baseline": w4,
            "pure_hbt_qap_baseline": pure_hbt,
            "baseline_envelope": {
                "combined_adjacency": envelope_adjacency,
                "predicted_layout_ssim": envelope_ssim,
            },
            "paired_delta": {
                "combined_adjacency": float(candidate["combined_adjacency"])
                - envelope_adjacency,
                "predicted_layout_ssim": float(candidate["predicted_layout_ssim"])
                - envelope_ssim,
            },
            "candidate_layout_sha256": _array_sha256(candidate_layout),
            "w1_layout_sha256": _array_sha256(evidence.w1_qap_layout),
            "w4_layout_sha256": _array_sha256(evidence.w4_qap_layout),
            "hbt_layout_sha256": _array_sha256(evidence.hbt_qap_layout),
            "warm_start_kind": args.warm_start_layout,
            "warm_start_layout_sha256": _array_sha256(warm_layout),
            "hungarian_squared_cost": sample.projections[0].squared_assignment_cost,
            "input_only_diagnostics": dict(evidence.diagnostics or {}),
            "truth_derived_confidence_used": False,
            "target_selected_candidate_used": False,
        }
        local_records.append(record)
        _print(
            runtime,
            {
                "event": "development_source",
                "source": name,
                "panel": panel_name,
                "replica": replica,
                "adjacency_delta": record["paired_delta"]["combined_adjacency"],
                "ssim_delta": record["paired_delta"]["predicted_layout_ssim"],
            },
        )

    if runtime.distributed:
        gathered: list[Any] = [None] * runtime.world_size
        torch.distributed.all_gather_object(gathered, local_records)
        all_records = [record for rank_records in gathered for record in rank_records]
    else:
        all_records = local_records
    all_records.sort(
        key=lambda record: (
            panel_order[str(record["panel"])],
            int(record["replica"]),
            str(record["source"]),
        )
    )
    if len(all_records) != len(cells):
        raise RuntimeError("distributed development gather lost or duplicated cells")

    panels: dict[str, Any] = {}
    for panel_name in panel_order:
        records = [record for record in all_records if record["panel"] == panel_name]
        adjacency_deltas, ssim_deltas = _source_level_deltas(records)
        mean_adjacency = float(adjacency_deltas.mean())
        mean_ssim = float(ssim_deltas.mean())
        positive_fraction = float(np.mean((adjacency_deltas > 0.0) & (ssim_deltas > 0.0)))
        adjacency_ci = _bootstrap_mean_ci(
            adjacency_deltas,
            seed=per_source_seed(args.seed, "posdiff:bootstrap-adjacency", panel_name),
            resamples=args.gate_bootstrap_resamples,
            confidence=args.gate_bootstrap_confidence,
        )
        ssim_ci = _bootstrap_mean_ci(
            ssim_deltas,
            seed=per_source_seed(args.seed, "posdiff:bootstrap-ssim", panel_name),
            resamples=args.gate_bootstrap_resamples,
            confidence=args.gate_bootstrap_confidence,
        )
        gates = {
            "adjacency_gain_vs_per_source_best_baseline": {
                "value": mean_adjacency,
                "minimum": args.gate_min_adjacency_gain,
                "passed": mean_adjacency >= args.gate_min_adjacency_gain,
            },
            "ssim_gain_vs_per_source_best_baseline": {
                "value": mean_ssim,
                "minimum": args.gate_min_ssim_gain,
                "passed": mean_ssim >= args.gate_min_ssim_gain,
            },
            "joint_positive_source_fraction": {
                "value": positive_fraction,
                "minimum": args.gate_min_positive_source_fraction,
                "passed": positive_fraction >= args.gate_min_positive_source_fraction,
            },
            "bootstrap_lower_adjacency_positive": {
                "value": adjacency_ci["lower"],
                "minimum": 0.0,
                "passed": adjacency_ci["lower"] > 0.0,
            },
            "bootstrap_lower_ssim_positive": {
                "value": ssim_ci["lower"],
                "minimum": 0.0,
                "passed": ssim_ci["lower"] > 0.0,
            },
        }
        panels[panel_name] = {
            "source_count": len(dev_names),
            "replicas_per_source": args.dev_replicas,
            "cell_count": len(records),
            "candidate_mean": _numeric_mean(records, "candidate"),
            "qap_w1_mean": _numeric_mean(records, "qap_w1_baseline"),
            "qap_w1_comparator_id": records[0]["qap_w1_comparator_id"],
            "w4_qap_mean": _numeric_mean(records, "w4_qap_baseline"),
            "pure_hbt_qap_mean": _numeric_mean(records, "pure_hbt_qap_baseline"),
            "mean_paired_delta_vs_envelope": {
                "combined_adjacency": mean_adjacency,
                "predicted_layout_ssim": mean_ssim,
            },
            "source_bootstrap_ci": {
                "combined_adjacency": adjacency_ci,
                "predicted_layout_ssim": ssim_ci,
            },
            "gates": gates,
            "gate_passed": all(value["passed"] for value in gates.values()),
            "per_source": records,
        }

    macro_adjacency_values, macro_ssim_values = _source_level_deltas(all_records)
    macro_adjacency = float(macro_adjacency_values.mean())
    macro_ssim = float(macro_ssim_values.mean())
    macro_gate = {
        "adjacency_gain": {
            "value": macro_adjacency,
            "minimum": args.gate_min_adjacency_gain,
            "passed": macro_adjacency >= args.gate_min_adjacency_gain,
        },
        "ssim_gain": {
            "value": macro_ssim,
            "minimum": args.gate_min_ssim_gain,
            "passed": macro_ssim >= args.gate_min_ssim_gain,
        },
    }
    panel_pass = all(value["gate_passed"] for value in panels.values())
    gate_passed = panel_pass and all(value["passed"] for value in macro_gate.values())
    return {
        "scope": f"{args.dev_split} whole-source paired exact panels",
        "source_names": dev_names,
        "source_names_sha256": _names_sha256(dev_names),
        "baseline_contract": (
            "per-cell metric envelope of equal-budget filename-seeded "
            f"C1+HBTw1 QAP, C1+HBTw4 QAP, and pure-HBT QAP; "
            f"i{args.qap_iterations}/r{args.qap_restarts}/"
            f"b{args.qap_boundary_weight:g}; layouts frozen before targets are scored"
        ),
        "required_default_comparator": "qap_w1_b0.05_i25_r2_filename_seed",
        "candidate_contract": (
            f"single deterministic {args.sampling_steps}-step DDIM path, input-only "
            f"HBT graph, {args.warm_start_layout} coordinate warm start shared with "
            "training, exact Hungarian; no target selector"
        ),
        "panels": panels,
        "macro_delta_vs_envelope": {
            "combined_adjacency": macro_adjacency,
            "predicted_layout_ssim": macro_ssim,
        },
        "macro_gates": macro_gate,
        "development_gate_passed": gate_passed,
        "assessment": (
            "bounded positive signal only; still not submission-ready"
            if gate_passed
            else "no trustworthy cross-corruption signal; stop/pivot before larger training"
        ),
        "safe_for_submission": False,
        "submission_ready": False,
    }


def _prepare_output(args: argparse.Namespace, runtime: Runtime) -> Path:
    output = Path(args.output_dir).expanduser().resolve()
    error: str | None = None
    if runtime.primary:
        output.mkdir(parents=True, exist_ok=True)
        if args.mode == "evaluate" or args.resume_checkpoint:
            known = [output / REPORT_NAME]
        else:
            known = [output / CHECKPOINT_NAME, output / LATEST_NAME, output / REPORT_NAME]
        if any(path.exists() for path in known) and not args.overwrite:
            error = f"known artifacts already exist in {output}; pass --overwrite explicitly"
    if runtime.distributed:
        payload: list[Any] = [error]
        torch.distributed.broadcast_object_list(payload, src=0)
        error = payload[0]
    if error is not None:
        raise FileExistsError(error)
    _barrier(runtime)
    return output


def _write_hashes(output: Path) -> None:
    paths = [
        path
        for path in (
            output / CHECKPOINT_NAME,
            output / LATEST_NAME,
            output / f"{LATEST_NAME}.previous",
            output / REPORT_NAME,
        )
        if path.exists()
    ]
    lines = [f"{_sha256(path)}  {path.name}" for path in paths]
    (output / HASHES_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    runtime: Runtime | None = None
    try:
        determinism = _configure_determinism()
        runtime = _init_runtime(args.device)
        _seed_everything(args.seed, runtime.rank)
        config = _config(args)
        model = PositionalDiffusionNet(config)
        resume_payload: Mapping[str, Any] | None = None
        evaluation_payload: Mapping[str, Any] | None = None
        if args.resume_checkpoint:
            if args.mode not in {"train", "pilot"}:
                raise ValueError("--resume-checkpoint is valid only in train or pilot mode")
            resume_path = Path(args.resume_checkpoint).expanduser().resolve()
            resume_payload = load_positional_diffusion_checkpoint_payload(resume_path)
            saved_config = PositionalDiffusionConfig(**resume_payload["model_config"])
            if saved_config != config:
                raise ValueError("resume model configuration does not match CLI configuration")
            metadata = resume_payload.get("metadata", {})
            if int(metadata.get("seed", -1)) != args.seed:
                raise ValueError("resume checkpoint seed does not match --seed")
            model.load_state_dict(resume_payload["model_state"], strict=True)
        elif args.mode == "evaluate":
            checkpoint_path = (
                Path(args.checkpoint).expanduser().resolve()
                if args.checkpoint
                else Path(args.output_dir).expanduser().resolve() / CHECKPOINT_NAME
            )
            evaluation_payload = load_positional_diffusion_checkpoint_payload(checkpoint_path)
            saved_config = PositionalDiffusionConfig(**evaluation_payload["model_config"])
            if saved_config != config:
                raise ValueError("evaluation model configuration does not match CLI configuration")
            model.load_state_dict(evaluation_payload["model_state"], strict=True)
        memory = estimate_peak_memory_bytes(
            config,
            batch_size=1,
            tile_count=TILE_COUNT,
            bytes_per_value=2,
            training=True,
        )
        dry_record = {
            "event": "dry_run_contract",
            "model_config": asdict(config),
            "parameters": model_parameter_count(model),
            "memory_estimate_bytes_per_gpu_microbatch1": memory,
            "recommended_launcher": "torchrun --standalone --nproc_per_node=2",
            "primary_adaptation": {
                "continuous_positions": True,
                "attention_gnn_reverse": True,
                "predict_x0": True,
                "linear_diffusion_steps": config.diffusion_steps,
                "deterministic_ddim_steps": args.sampling_steps,
                "reference_greedy_projection_replaced_by_hungarian": True,
                "raw_plus_denoised_encoder": True,
                "input_only_hbt_relative_graph": True,
            },
            "ready_for_bounded_kaggle_signal_pilot": True,
            "submission_ready": False,
            "safe_for_submission": False,
            "determinism": determinism,
        }
        if args.mode == "dry-run":
            _print(runtime, dry_record)
            return 0

        output = _prepare_output(args, runtime)
        manifest = (REPO_ROOT / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
        quarantine = (REPO_ROOT / args.quarantine).resolve() if not Path(args.quarantine).is_absolute() else Path(args.quarantine)
        data_root = (REPO_ROOT / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root)
        edge_train = source_names_for_split(
            "edge_train", manifest_path=manifest, quarantine_path=quarantine
        )
        development_pool = source_names_for_split(
            args.dev_split, manifest_path=manifest, quarantine_path=quarantine
        )
        train_names = edge_train[args.train_offset : args.train_offset + args.train_sources]
        dev_names = development_pool[args.dev_offset : args.dev_offset + args.dev_sources]
        if len(train_names) != args.train_sources or len(dev_names) != args.dev_sources:
            raise ValueError("requested source slice exceeds authoritative split")
        if set(train_names) & set(dev_names):
            raise RuntimeError("whole-source train/development leakage detected")
        for name in set(train_names if args.mode in {"train", "pilot"} else []) | set(dev_names if args.mode in {"evaluate", "pilot"} else []):
            path = data_root / "train" / "targets" / name
            if not path.is_file():
                raise FileNotFoundError(path)

        hardware = _all_hardware(runtime)
        amp_enabled, amp_dtype = _amp_settings(args, runtime)
        denoiser_path = (REPO_ROOT / args.denoiser).resolve() if not Path(args.denoiser).is_absolute() else Path(args.denoiser)
        hbt_path = (REPO_ROOT / args.hbt_checkpoint).resolve() if not Path(args.hbt_checkpoint).is_absolute() else Path(args.hbt_checkpoint)
        restorer, restored_device, restorer_metadata = load_restorer(
            denoiser_path, device=str(runtime.device)
        )
        if restored_device != runtime.device:
            raise RuntimeError(f"restorer resolved to {restored_device}, expected {runtime.device}")
        hbt_model, hbt_metadata = load_embedding_checkpoint(hbt_path, device=runtime.device)
        restorer.requires_grad_(False).eval()
        hbt_model.requires_grad_(False).eval()
        hbt_metadata = {
            **hbt_metadata,
            "checkpoint": str(hbt_path),
            "checkpoint_sha256": _sha256(hbt_path),
        }
        upstream_exposure = _upstream_exposure_audit(
            dev_names,
            manifest=manifest,
            quarantine=quarantine,
            hbt_metadata=hbt_metadata,
        )
        train_data_sha256 = (
            _dataset_slice_sha256(data_root, train_names)
            if args.mode in {"train", "pilot"}
            else None
        )
        evaluation_contract: dict[str, Any] | None = None
        if evaluation_payload is not None:
            evaluation_contract = _validate_evaluation_contract(
                evaluation_payload,
                args=args,
                config=config,
                restorer_metadata=restorer_metadata,
                hbt_metadata=hbt_metadata,
                manifest=manifest,
                quarantine=quarantine,
            )

        training_report: dict[str, Any] | None = None
        if args.mode in {"train", "pilot"}:
            model, training_report = _train(
                model,
                GaussianPositionDiffusion(config),
                args=args,
                runtime=runtime,
                train_names=train_names,
                data_root=data_root,
                output_dir=output,
                restorer=restorer,
                hbt_model=hbt_model,
                restorer_metadata=restorer_metadata,
                hbt_metadata=hbt_metadata,
                train_data_sha256=str(train_data_sha256),
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                resume_payload=resume_payload,
            )
        elif args.mode == "evaluate":
            model.to(runtime.device)
            memory = estimate_peak_memory_bytes(
                config,
                batch_size=1,
                tile_count=TILE_COUNT,
                bytes_per_value=2,
                training=False,
            )

        evaluation_report: dict[str, Any] | None = None
        if args.mode in {"evaluate", "pilot"}:
            evaluation_report = _evaluate(
                model,
                GaussianPositionDiffusion(config).to(runtime.device),
                args=args,
                runtime=runtime,
                dev_names=dev_names,
                data_root=data_root,
                restorer=restorer,
                hbt_model=hbt_model,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

        if runtime.primary:
            report = {
                "schema_version": 1,
                "kind": "positional_diffusion_bounded_signal_report",
                "created_unix": time.time(),
                "mode": args.mode,
                "arguments": vars(args),
                "model_config": asdict(config),
                "model_parameters": model_parameter_count(model),
                "memory_estimate": memory,
                "runtime": {
                    "world_size": runtime.world_size,
                    "amp_enabled": amp_enabled,
                    "amp_dtype": str(amp_dtype),
                    "determinism": determinism,
                    "hardware_by_rank": hardware,
                    "nvidia_smi": _nvidia_smi(),
                },
                "split_provenance": {
                    "manifest": str(manifest),
                    "manifest_sha256": _sha256(manifest),
                    "quarantine": str(quarantine),
                    "quarantine_sha256": _sha256(quarantine),
                    "train_source_names_sha256": _names_sha256(train_names),
                    "train_data_sha256": train_data_sha256,
                    "development_source_names": dev_names,
                    "development_source_names_sha256": _names_sha256(dev_names),
                    "whole_source_disjoint": True,
                    "development_split": args.dev_split,
                    "upstream_exposure_audit": upstream_exposure,
                },
                "evaluation_checkpoint_contract": evaluation_contract,
                "training": training_report,
                "development": evaluation_report,
                "method_status": (
                    "ready_for_bounded_kaggle_signal_pilot"
                    if args.mode == "train"
                    else (evaluation_report or {}).get("assessment", "dry contract only")
                ),
                "safe_for_submission": False,
                "submission_ready": False,
                "primary_sources": dry_record["primary_adaptation"],
            }
            (output / REPORT_NAME).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _write_hashes(output)
            print(json.dumps({"event": "complete", "report": str(output / REPORT_NAME), "submission_ready": False}), flush=True)
        return 0
    finally:
        _cleanup(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
