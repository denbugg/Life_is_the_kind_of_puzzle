"""Independent, fail-closed verifier for the frozen M144 DCT-Where run.

This module deliberately does not import ``m144_dct_where`` or the runner.
It authenticates the frozen run contract and numeric artifacts, independently
recomputes the dirty-palette Hungarian controls, clustered bootstraps, metrics,
gates and terminal decision, then writes a create-once verification receipt.

Production paths are E:-only.  ``require_e_drive=False`` is a narrow test hook
for synthetic, data-free unit tests; the command-line interface never exposes
that relaxation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


REPORT_SCHEMA = "pazzle-m144-dct-where-report-v2"
VERIFICATION_SCHEMA = "pazzle-m144-dct-where-verification-v1"
ORACLE_STAGE = "oracle_pretrain"
CAL_STAGE = "cal"
DEV_STAGE = "dev"

BOOTSTRAP_SEED = 144_032
BOOTSTRAP_SAMPLES = 10_000
CAL_CONFIDENCE = 0.90
DEV_CONFIDENCE = 0.95
FIT_COUNT = 5_360
PARTITION_COUNT = 670
PALETTE_DIM = 60
PALETTE_QUANTILES = np.linspace(0.0, 1.0, 13, dtype=np.float64)
PALETTE_EPSILON = 1.0e-6

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_NUMERIC_NPZ_BYTES = 16 * 1024 * 1024
MAX_NPZ_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
MAX_GPU_ALLOCATED_BYTES = 3 * 1024 * 1024 * 1024
MIN_GPU_TOTAL_BYTES = int(7.5 * 1024 * 1024 * 1024)
MIN_GPU_FREE_BYTES = 4 * 1024 * 1024 * 1024
MAX_WORK_ROOT_BYTES = 6 * 1024 * 1024 * 1024
MAX_ACTIVE_WALL_SECONDS = 8 * 60 * 60

PINNED_SOURCE_SHA256 = (
    "fa142c5f9c4fa17671b60d72b9acedff0eafcad4e77afac2b17a9649adfbfbd9"
)
PINNED_SPLIT_SHA256 = (
    "a858a194ceab9976b72069aef6c46481734ce15594f67ae6818b4d7bfe30231a"
)
LEGACY_PAIRED_SHA256_EVIDENCE_ONLY = (
    "a93405fc0e5cc129e8008bd3875957b0683e0dad3671f360a197b806d45fb554"
)
PRIOR_EVIDENCE = {
    "kind": "leaky_paired_alignment_checkpoint",
    "sha256": LEGACY_PAIRED_SHA256_EVIDENCE_ONLY,
    "loaded": False,
    "overlap": {
        "fit": [5133, 5360],
        "cal": [639, 670],
        "dev": [638, 670],
        "reserve": [290, 300],
    },
    "use": "human_prior_only_not_runtime_input",
}

SOURCE_CLOSURE = (
    "autoresearch-runs/pazzle-mgc-restoration-20260818/M144_DCT_WHERE_PLAN.md",
    "launch_m144_dct_where.ps1",
    "src/config.py",
    "src/distort.py",
    "src/eval_paired_alignment.py",
    "src/imgio.py",
    "src/m144_dct_where.py",
    "src/run_m144_dct_where.py",
    "src/verify_m144_dct_where.py",
    "tests/test_m144_dct_where.py",
    "tests/test_run_m144_dct_where.py",
    "tests/test_verify_m144_dct_where.py",
)

RAW_ID_KEYS = ("board_id", "source_group_id", "swap_cycle_id")
RAW_SSIM_KEYS = (
    "flat_ssim",
    "target_oracle_dct_ssim",
    "dct_full_ssim",
    "dct_blind_ssim",
    "dct_swapped_ssim",
    "rgb8_full_ssim",
    "rgb8_blind_ssim",
)
RAW_PREDICTION_KEYS = (
    "flat_rgb",
    "dct_full_coeff",
    "dct_blind_coeff",
    "dct_swapped_coeff",
    "rgb8_full_residual",
    "rgb8_blind_residual",
)
RAW_KEYS = RAW_ID_KEYS + RAW_SSIM_KEYS + RAW_PREDICTION_KEYS
ORACLE_ID_KEYS = ("board_id", "source_group_id")
ORACLE_SSIM_KEYS = ("flat_ssim", "target_oracle_dct_ssim")
ORACLE_PREDICTION_KEYS = ("flat_rgb", "oracle_coeff")
ORACLE_KEYS = ORACLE_ID_KEYS + ORACLE_SSIM_KEYS + ORACLE_PREDICTION_KEYS
FIT_PALETTE_KEYS = ("fit_board_id", "dirty_feature60", "mean60", "scale60")
SWAP_ID_KEYS = (
    "board_id",
    "source_group_id",
    "donor_board_id",
    "donor_source_group_id",
    "swap_cycle_id",
)
SWAP_KEYS = SWAP_ID_KEYS + ("dirty_feature60",)
ENCODER_RANK_KEYS = (
    "board_id",
    "source_group_id",
    "dirty_to_clean_rank",
    "clean_to_dirty_rank",
)

REPORT_KEYS = {
    "schema",
    "status",
    "stage",
    "decision",
    "reason",
    "protocol",
    "contract",
    "encoder_final_checkpoint",
    "encoder_cal_ranks_npz",
    "representation_contract",
    "oracle_cal_npz",
    "fit_palette_npz",
    "swap_cal_npz",
    "raw_cal_npz",
    "swap_dev_npz",
    "raw_dev_npz",
    "final_checkpoint",
    "metrics",
    "gates",
    "receipts",
    "prohibitions",
}
REPORT_CONTRACT_KEYS = {
    "run_contract_sha256",
    "split_manifest",
    "source_manifest",
    "source_files",
    "runtime",
    "prior_evidence",
    "cal_count",
    "dev_count",
}
PATH_RECORD_KEYS = {"path", "bytes", "sha256"}
RECEIPT_KEYS = {
    "encoder_fit_targets",
    "encoder_cal_targets",
    "encoder_final_checkpoint",
    "encoder_cal_gate",
    "representation_contract",
    "capacity",
    "fit_cache",
    "cal_cache",
    "oracle_cal",
    "fit_palette",
    "swap_cal",
    "final_checkpoint",
    "raw_cal",
    "dev_targets",
    "dev_cache",
    "swap_dev",
    "raw_dev",
}
PROHIBITIONS = ["no_test_access", "no_submission", "diagnostic_only"]

CACHE_SCHEMA = "pazzle-m144-dct-where-embedding-cache-v1"
TARGET_RECEIPT_SCHEMA = "pazzle-m144-dct-where-target-receipt-v1"
CAPACITY_RECEIPT_SCHEMA = "pazzle-m144-dct-where-capacity-receipt-v1"
ORACLE_RECEIPT_SCHEMA = "pazzle-m144-dct-where-oracle-receipt-v1"
FIT_PALETTE_RECEIPT_SCHEMA = "pazzle-m144-dct-where-swap-whitening-v1"
SWAP_RECEIPT_SCHEMA = "pazzle-m144-dct-where-swap-receipt-v1"
CHECKPOINT_RECEIPT_SCHEMA = "pazzle-m144-dct-where-checkpoint-receipt-v1"
RAW_RECEIPT_SCHEMA = "pazzle-m144-dct-where-raw-receipt-v1"
ENCODER_CHECKPOINT_RECEIPT_SCHEMA = "pazzle-m144-fit-encoder-checkpoint-receipt-v1"
ENCODER_GATE_RECEIPT_SCHEMA = "pazzle-m144-fit-encoder-cal-gate-receipt-v1"
REPRESENTATION_CONTRACT_SCHEMA = "pazzle-m144-representation-contract-v1"
REPRESENTATION_RECEIPT_SCHEMA = "pazzle-m144-representation-contract-receipt-v1"

CAL_THRESHOLDS = {
    "oracle_gain": 0.040,
    "full_gain": 0.008,
    "full_blind": 0.003,
    "full_swapped": 0.002,
    "representation_delta": 0.001,
}
DEV_THRESHOLDS = {
    "full_gain": 0.012,
    "full_blind": 0.005,
    "full_swapped": 0.003,
    "representation_delta": 0.003,
    "full_blind_win": 0.60,
    "full_swapped_win": 0.60,
}

ENCODER_RECIPE: dict[str, Any] = {
    "version": "m144-fit-only-paired-encoder-v1",
    "architecture": {"class": "PairedAlignment", "embed_dim": 128, "init": "scratch"},
    "seed": 144011,
    "fit": {
        "boards": FIT_COUNT,
        "steps": 1500,
        "board_batch": 4,
        "tiles_per_board": 192,
    },
    "board_schedule": "stateless_batch_indices(FIT,4,step,seed=144011)",
    "rng": {
        "tile_choice": (
            "default_rng([144011,1,step,slot,board_id,0]);"
            "choice(576,192,replace=False)"
        ),
        "corruption": (
            "default_rng([144011,1,step,slot,board_id,1]);"
            "distort_selected_clean"
        ),
        "cal": "default_rng([144011,2,board_id]);distort_all_576_clean",
    },
    "distortion": "distort.distort_frags challenge chain",
    "objective": "symmetric_InfoNCE_float32",
    "optimizer": {
        "name": "AdamW",
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
    },
    "scheduler": {"name": "CosineAnnealingLR", "T_max": 1500, "eta_min": 0.0},
    "amp": "model_forward_only_with_GradScaler",
    "grad_clip": 1.0,
    "checkpoint_every": 100,
    "selection": "fixed_final_step_1500_no_periodic_eval_no_best_selection",
    "determinism": {
        "torch.use_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    },
    "cal": {
        "boards": 670,
        "tiles_per_board": 576,
        "rank": "1+count_strictly_greater_similarity",
        "gate": {
            "dirty_to_clean_r1": 0.20,
            "dirty_to_clean_r5": 0.45,
            "dirty_to_clean_r20": 0.70,
        },
    },
}

# The runner copies this object verbatim into the report.  It is intentionally
# explicit so neither a changed hard-negative recipe nor a changed control can
# silently inherit the M144 name.
REPORT_PROTOCOL: dict[str, Any] = {
    "version": "m144-dct-where-freeze-20260820-v1",
    "representation": {
        "upright_tiles_only": True,
        "dct": {
            "channels": 3,
            "side": 16,
            "coefficients_per_channel": 32,
            "target": "low_frequency_residual_above_dirty_flat",
        },
        "rgb8_control": {
            "channels": 3,
            "side": 8,
            "target": "rgb_residual_above_same_dirty_flat",
        },
    },
    "encoder": ENCODER_RECIPE,
    "dirty_feature60": {
        "schema": "m144-dirty-feature60-v1",
        "source": "dirty_tiles_only",
        "value_conversion": "uint8_to_float64_div255",
        "tile_columns": ["mean_R", "mean_G", "mean_B", "centered_rgb_rms"],
        "centered_rgb_rms": (
            "sqrt(mean_chw((tile[t,c,h,w]-mean_hw(tile[t,c]))**2))"
        ),
        "quantiles": PALETTE_QUANTILES.tolist(),
        "quantile_method": "linear",
        "ordering": (
            "dimension_major_13_quantiles_R_G_B_rms_then_4_population_means_"
            "then_4_population_std_ddof0"
        ),
        "dimension": PALETTE_DIM,
    },
    "swap_control": {
        "fit_center": "FIT_population_mean_float64",
        "fit_scale": "maximum(FIT_population_std_ddof0,1e-6)",
        "cost": "squared_euclidean_sum_float64",
        "assignment": (
            "scipy.optimize.linear_sum_assignment canonical float64 "
            "squared-Euclidean; diagonal/same-group forbidden; no jitter"
        ),
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
        "forbidden": ["self", "same_source_group"],
        "donors": "within_partition_bijection",
        "cycles": "arbitrary_canonical_dense_by_smallest_unvisited_board_index",
        "minimum_cycles_per_670_partition": 64,
    },
    "bootstrap": {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "quantile_method": "linear",
        "cal_confidence": CAL_CONFIDENCE,
        "dev_confidence": DEV_CONFIDENCE,
        "seed_offsets": {
            "full_minus_blind": 0,
            "full_minus_swapped": 1,
            "representation_delta": 2,
        },
        "cluster_units": {
            "full_minus_blind": "source_group_id",
            "full_minus_swapped": "swap_cycle_id",
            "representation_delta": "source_group_id",
        },
    },
    "formulas": {
        "gain": "mean(arm_ssim-flat_ssim)",
        "full_minus_blind": "dct_full_ssim-dct_blind_ssim",
        "full_minus_swapped": "dct_full_ssim-dct_swapped_ssim",
        "representation_delta": (
            "(dct_full_ssim-dct_blind_ssim)-"
            "(rgb8_full_ssim-rgb8_blind_ssim)"
        ),
        "paired_win": "mean(candidate_ssim>control_ssim)",
    },
    "scientific_metric": {
        "renderer": "independent_CPU_float32",
        "prediction_quantization": "clip(rint(rendered_float*255),0,255).astype(uint8)",
        "target": "original_uint8",
        "implementation": "skimage.metrics.structural_similarity",
        "channel_axis": 2,
        "data_range": 255,
        "win_size": 7,
        "gaussian_weights": False,
        "use_sample_covariance": True,
        "stored_score_atol": 1.0e-12,
        "oracle_coeff_recompute_atol": 2.0e-5,
        "oracle_coeff_recompute_rtol": 2.0e-5,
        "training_loss": "differentiable_float_uniform_window7_proxy_only",
    },
    "gates": {
        "ORACLE_CAL": {"oracle_gain": 0.040},
        "CAL": {
            "confidence": CAL_CONFIDENCE,
            "thresholds": CAL_THRESHOLDS,
            "required_positive_lower": ["full_blind"],
        },
        "DEV": {
            "confidence": DEV_CONFIDENCE,
            "thresholds": DEV_THRESHOLDS,
            "required_positive_lower": [
                "full_blind",
                "full_swapped",
                "representation_delta",
            ],
        },
    },
    "opening_order": [
        "encoder_fit_train",
        "encoder_cal_gate",
        "representation_contract_if_encoder_pass",
        "oracle_pretrain",
        "train",
        "cal",
        "dev_if_cal_pass",
    ],
    "resource_caps": {
        "max_gpu_allocated_bytes": MAX_GPU_ALLOCATED_BYTES,
        "min_gpu_total_bytes": MIN_GPU_TOTAL_BYTES,
        "min_gpu_free_before_stage_bytes": MIN_GPU_FREE_BYTES,
        "max_work_root_bytes": MAX_WORK_ROOT_BYTES,
        "max_cumulative_active_seconds": MAX_ACTIVE_WALL_SECONDS,
    },
}


class VerificationError(RuntimeError):
    """The run contract, evidence, report, or receipt is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise VerificationError(f"value is not canonical JSON: {error}") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise VerificationError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _duplicate_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON constant forbidden: {value}")


def load_canonical_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise VerificationError(f"cannot read {label}: {error}") from error
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise VerificationError(
            f"{label} size must be 1..{MAX_JSON_BYTES} bytes, got {len(raw)}"
        )
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_guard,
            parse_constant=_reject_constant,
        )
    except VerificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid UTF-8 JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{label} root must be an object")
    if raw != canonical_json_bytes(value):
        raise VerificationError(f"{label} must be compact sorted canonical JSON")
    return value, raw


def _exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be an object")
    if set(value) != keys:
        raise VerificationError(
            f"{label} keys mismatch; missing={sorted(keys-set(value))}, "
            f"extra={sorted(set(value)-keys)}"
        )
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise VerificationError(f"{label} must be a string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerificationError(f"{label} must be an integer")
    return value


def _digest(value: Any, label: str) -> str:
    text = _string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise VerificationError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _assert_e(path: Path, label: str) -> None:
    if path.drive.upper() != "E:":
        raise VerificationError(f"{label} must be on E:, got {path}")


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _assert_equivalent(expected: Any, claimed: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(claimed, dict) or set(expected) != set(claimed):
            raise VerificationError(f"{label} object shape mismatch")
        for key in expected:
            _assert_equivalent(expected[key], claimed[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(claimed, list) or len(expected) != len(claimed):
            raise VerificationError(f"{label} list mismatch")
        for index, (left, right) in enumerate(zip(expected, claimed)):
            _assert_equivalent(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, bool):
        if type(claimed) is not bool or expected is not claimed:
            raise VerificationError(f"{label} mismatch")
        return
    if isinstance(expected, int):
        if type(claimed) is not int or expected != claimed:
            raise VerificationError(f"{label} mismatch")
        return
    if isinstance(expected, float):
        if isinstance(claimed, bool) or not isinstance(claimed, (int, float)):
            raise VerificationError(f"{label} must be numeric")
        observed = float(claimed)
        if not math.isfinite(observed) or not math.isclose(
            expected, observed, rel_tol=1.0e-12, abs_tol=1.0e-14
        ):
            raise VerificationError(
                f"{label} mismatch: recomputed={expected:.17g}, claimed={observed:.17g}"
            )
        return
    if type(expected) is not type(claimed) or expected != claimed:
        raise VerificationError(f"{label} mismatch: {expected!r} != {claimed!r}")


def current_runtime_record() -> dict[str, Any]:
    import skimage
    import torch

    cuda_available = bool(torch.cuda.is_available())
    gpu: dict[str, Any] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": str(properties.name),
            "total_memory": int(properties.total_memory),
            "capability": [int(properties.major), int(properties.minor)],
        }
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "scipy": str(scipy.__version__),
        "skimage": str(skimage.__version__),
        "cuda_available": cuda_available,
        "torch_cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu": gpu,
        "determinism": {
            "algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
    }


def _verify_record(
    raw: Any,
    *,
    label: str,
    require_e_drive: bool,
    expected_parent: Path | None = None,
    expected_name: str | None = None,
    max_bytes: int = MAX_NUMERIC_NPZ_BYTES,
) -> tuple[Path, dict[str, Any]]:
    record = _exact_keys(raw, PATH_RECORD_KEYS, label)
    raw_path = Path(_string(record["path"], f"{label}.path"))
    if not raw_path.is_absolute():
        raise VerificationError(f"{label}.path must be absolute")
    try:
        path = raw_path.resolve(strict=True)
    except OSError as error:
        raise VerificationError(f"{label} is missing: {raw_path}") from error
    if not path.is_file() or path.is_symlink():
        raise VerificationError(f"{label} must be a regular non-symlink file")
    if require_e_drive:
        _assert_e(path, label)
    if expected_parent is not None and path.parent != expected_parent.resolve():
        raise VerificationError(f"{label} is outside its frozen directory")
    if expected_name is not None and path.name != expected_name:
        raise VerificationError(f"{label} must be named {expected_name}")
    claimed_size = _integer(record["bytes"], f"{label}.bytes")
    actual_size = int(path.stat().st_size)
    if claimed_size != actual_size or not 0 < actual_size <= max_bytes:
        raise VerificationError(f"{label} byte receipt/limit mismatch")
    claimed_sha = _digest(record["sha256"], f"{label}.sha256")
    if claimed_sha != sha256_file(path):
        raise VerificationError(f"{label} SHA-256 mismatch")
    return path, dict(record)


def _load_reported_receipt(
    raw_record: Any,
    *,
    root: Path,
    relative_path: str,
    schema: str,
    contract_sha256: str,
    require_e_drive: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = (root / relative_path).resolve()
    path, record = _verify_record(
        raw_record,
        label=f"receipt[{relative_path}]",
        require_e_drive=require_e_drive,
        expected_parent=expected.parent,
        expected_name=expected.name,
        max_bytes=MAX_JSON_BYTES,
    )
    payload, _ = load_canonical_json(path, label=f"receipt {relative_path}")
    if payload.get("schema") != schema:
        raise VerificationError(f"receipt {relative_path} schema mismatch")
    if payload.get("contract_sha256") != contract_sha256:
        raise VerificationError(f"receipt {relative_path} contract mismatch")
    return payload, record


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise VerificationError(f"{label} must be finite")
    return result


def validate_capacity_receipt(payload: Mapping[str, Any]) -> None:
    if payload.get("passed") is not True or payload.get("synthetic") is not True:
        raise VerificationError("capacity smoke must be passing and data-free")
    if payload.get("arms") != ["dct_full", "dct_blind", "rgb_full", "rgb_blind"]:
        raise VerificationError("capacity smoke did not exercise all four arms")
    if payload.get("shapes") != {
        "embeddings": [8, 576, 128],
        "flat": [8, 3],
        "target": [8, 3, 480, 480],
    }:
        raise VerificationError("capacity smoke did not use the production B8 shape")
    if payload.get("render_ssim_fp32") is not True:
        raise VerificationError("capacity smoke must render and score in FP32")
    peak = _integer(payload.get("peak_allocated_bytes"), "capacity.peak_allocated_bytes")
    if peak < 0 or peak > MAX_GPU_ALLOCATED_BYTES:
        raise VerificationError("capacity smoke exceeded the 3 GiB allocation cap")
    if payload.get("max_gpu_allocated_bytes") != MAX_GPU_ALLOCATED_BYTES:
        raise VerificationError("capacity receipt allocation threshold drift")
    losses = payload.get("losses")
    if not isinstance(losses, dict) or set(losses) != {
        "dct_full", "dct_blind", "rgb_full", "rgb_blind"
    }:
        raise VerificationError("capacity smoke loss map mismatch")
    for name, value in losses.items():
        _finite_number(value, f"capacity.losses.{name}")
    resources = payload.get("resources_before")
    if not isinstance(resources, dict):
        raise VerificationError("capacity receipt lacks pre-stage resources")
    if resources.get("cuda") is not True:
        raise VerificationError("capacity smoke was not run on CUDA")
    if _integer(resources.get("gpu_total_bytes"), "capacity.gpu_total_bytes") < MIN_GPU_TOTAL_BYTES:
        raise VerificationError("capacity GPU has less than 7.5 GiB total memory")
    if _integer(resources.get("gpu_free_bytes"), "capacity.gpu_free_bytes") < MIN_GPU_FREE_BYTES:
        raise VerificationError("capacity pre-stage free GPU memory is below 4 GiB")
    if _integer(resources.get("work_root_bytes"), "capacity.work_root_bytes") > MAX_WORK_ROOT_BYTES:
        raise VerificationError("capacity work-root footprint exceeds 6 GiB")
    if _finite_number(
        resources.get("cumulative_active_seconds"), "capacity.cumulative_active_seconds"
    ) > MAX_ACTIVE_WALL_SECONDS:
        raise VerificationError("capacity active wall exceeds 8 hours")


def _close_memmap(array: np.ndarray) -> None:
    backing = getattr(array, "_mmap", None)
    if backing is not None:
        backing.close()


def validate_cache_receipt(
    payload: Mapping[str, Any],
    *,
    root: Path,
    partition: str,
    count: int,
    require_e_drive: bool,
) -> dict[str, np.ndarray]:
    required = {
        "schema", "partition", "contract_sha256", "count", "names_sha256",
        "embedding_dim", "embeddings", "flat", "palette", "boards",
    }
    _exact_keys(payload, required, f"{partition} cache receipt")
    if payload["partition"] != partition or payload["count"] != count:
        raise VerificationError(f"{partition} cache identity/count mismatch")
    if payload["embedding_dim"] != 128:
        raise VerificationError(f"{partition} cache embedding dimension drift")
    _digest(payload["names_sha256"], f"{partition}.names_sha256")
    cache_root = root / "cache"
    embeddings_path, _ = _verify_record(
        payload["embeddings"],
        label=f"{partition}.embeddings",
        require_e_drive=require_e_drive,
        expected_parent=cache_root,
        expected_name=f"{partition}_dirty_embeddings.f16.npy",
        max_bytes=MAX_CHECKPOINT_BYTES,
    )
    flat_path, _ = _verify_record(
        payload["flat"],
        label=f"{partition}.flat",
        require_e_drive=require_e_drive,
        expected_parent=cache_root,
        expected_name=f"{partition}_flat_rgb.f32.npy",
        max_bytes=MAX_CHECKPOINT_BYTES,
    )
    palette_path, _ = _verify_record(
        payload["palette"],
        label=f"{partition}.palette",
        require_e_drive=require_e_drive,
        expected_parent=cache_root,
        expected_name=f"{partition}_dirty_feature60.f64.npy",
        max_bytes=MAX_CHECKPOINT_BYTES,
    )
    try:
        embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
        flat = np.load(flat_path, mmap_mode="r", allow_pickle=False)
        palette = np.load(palette_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise VerificationError(f"cannot load {partition} dirty-only cache: {error}") from error
    try:
        if embeddings.shape != (count, 576, 128) or embeddings.dtype != np.float16:
            raise VerificationError(f"{partition} embedding cache shape/dtype mismatch")
        if flat.shape != (count, 3) or flat.dtype != np.float32:
            raise VerificationError(f"{partition} flat cache shape/dtype mismatch")
        if palette.shape != (count, PALETTE_DIM) or palette.dtype != np.float64:
            raise VerificationError(f"{partition} palette cache shape/dtype mismatch")
        flat_copy = np.asarray(flat).copy()
        palette_copy = np.asarray(palette).copy()
    finally:
        _close_memmap(embeddings)
        _close_memmap(flat)
        _close_memmap(palette)
    _validate_feature_rows(palette_copy, count, f"{partition} cache palette")
    boards = payload["boards"]
    if not isinstance(boards, list) or len(boards) != count:
        raise VerificationError(f"{partition} cache board ledger length mismatch")
    names: list[str] = []
    board_ids = np.empty(count, dtype=np.int64)
    image_pattern = re.compile(r"^img_(\d{6})\.png$")
    for index, row in enumerate(boards):
        _exact_keys(row, {"name", "input_bytes", "input_sha256"}, f"{partition}.boards[{index}]")
        name = _string(row["name"], f"{partition}.boards[{index}].name")
        match = image_pattern.fullmatch(name)
        if match is None:
            raise VerificationError(f"{partition} cache has invalid board name")
        if _integer(row["input_bytes"], f"{partition}.input_bytes") <= 0:
            raise VerificationError(f"{partition} cache input_bytes must be positive")
        _digest(row["input_sha256"], f"{partition}.input_sha256")
        names.append(name)
        board_ids[index] = int(match.group(1))
    if len(set(names)) != count or not np.all(np.diff(board_ids) > 0):
        raise VerificationError(f"{partition} dirty-only cache order is not canonical")
    expected_names_digest = sha256_bytes("\n".join(names).encode("ascii"))
    if payload["names_sha256"] != expected_names_digest:
        raise VerificationError(f"{partition} cache names digest mismatch")
    return {"board_id": board_ids, "flat_rgb": flat_copy, "dirty_feature60": palette_copy}


def validate_linked_receipt(
    payload: Mapping[str, Any],
    *,
    nested_key: str,
    artifact_record_value: Mapping[str, Any],
    label: str,
) -> None:
    if nested_key not in payload:
        raise VerificationError(f"{label} receipt lacks {nested_key}")
    _assert_equivalent(artifact_record_value, payload[nested_key], f"{label}.{nested_key}")


def validate_target_receipt_record(
    raw_record: Any,
    *,
    root: Path,
    partition: str,
    count: int,
    expected_board_ids: np.ndarray,
    target_root: Path,
    contract_sha256: str,
    require_e_drive: bool,
) -> dict[str, Any]:
    payload, record = _load_reported_receipt(
        raw_record,
        root=root,
        relative_path=f"receipts/{partition}_targets.json",
        schema=TARGET_RECEIPT_SCHEMA,
        contract_sha256=contract_sha256,
        require_e_drive=require_e_drive,
    )
    required = {
        "schema", "partition", "contract_sha256", "count", "names_sha256", "targets"
    }
    _exact_keys(payload, required, f"{partition} target receipt")
    if payload["partition"] != partition or payload["count"] != count:
        raise VerificationError(f"{partition} target receipt identity mismatch")
    targets = payload["targets"]
    if not isinstance(targets, list) or len(targets) != count:
        raise VerificationError(f"{partition} target receipt ledger length mismatch")
    names: list[str] = []
    identifiers = np.empty(count, dtype=np.int64)
    pattern = re.compile(r"^img_(\d{6})\.png$")
    for index, row in enumerate(targets):
        _exact_keys(row, {"name", "bytes", "sha256"}, f"{partition}.targets[{index}]")
        name = _string(row["name"], f"{partition}.targets[{index}].name")
        match = pattern.fullmatch(name)
        if match is None:
            raise VerificationError(f"{partition} target receipt has invalid name")
        if _integer(row["bytes"], f"{partition}.targets[{index}].bytes") <= 0:
            raise VerificationError(f"{partition} target receipt has empty target")
        _digest(row["sha256"], f"{partition}.targets[{index}].sha256")
        target_path = (target_root / name).resolve(strict=True)
        if target_path.parent != target_root.resolve() or not target_path.is_file():
            raise VerificationError(f"{partition} target escapes/misses the target root")
        actual = path_record(target_path)
        if actual["bytes"] != row["bytes"] or actual["sha256"] != row["sha256"]:
            raise VerificationError(f"{partition} target bytes/hash drift for {name}")
        names.append(name)
        identifiers[index] = int(match.group(1))
    if not np.array_equal(identifiers, expected_board_ids):
        raise VerificationError(f"{partition} target receipt board order mismatch")
    if payload["names_sha256"] != sha256_bytes("\n".join(names).encode("ascii")):
        raise VerificationError(f"{partition} target receipt names digest mismatch")
    return {"record": record, "names": names}


def _source_digest_map(raw: Any, repo_root: Path) -> dict[str, str]:
    if not isinstance(raw, dict) or set(raw) != set(SOURCE_CLOSURE):
        raise VerificationError("run contract source_files closure mismatch")
    result: dict[str, str] = {}
    for relative in SOURCE_CLOSURE:
        entry = raw[relative]
        if isinstance(entry, dict):
            source_path, record = _verify_record(
                entry,
                label=f"source_files[{relative}]",
                require_e_drive=False,
                expected_parent=(repo_root / relative).resolve().parent,
                expected_name=Path(relative).name,
                max_bytes=MAX_JSON_BYTES * 8,
            )
            if source_path != (repo_root / relative).resolve():
                raise VerificationError(f"source file path drift: {relative}")
            result[relative] = record["sha256"]
        else:
            claimed = _digest(entry, f"source_files[{relative}]")
            source_path = (repo_root / relative).resolve(strict=True)
            actual = sha256_file(source_path)
            if claimed != actual:
                raise VerificationError(f"source file SHA-256 drift: {relative}")
            result[relative] = claimed
    return result


def validate_run_contract(
    contract_path: Path,
    *,
    work_root: Path,
    repo_root: Path,
    require_e_drive: bool,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    contract, raw = load_canonical_json(contract_path, label="run contract")
    if contract.get("schema") != "pazzle-m144-dct-where-run-contract-v2":
        raise VerificationError("unsupported run contract schema")
    if not isinstance(contract.get("contract_sha256"), str):
        raise VerificationError("run contract lacks contract_sha256")
    body = dict(contract)
    claimed_contract_sha = _digest(body.pop("contract_sha256"), "contract_sha256")
    if claimed_contract_sha != sha256_bytes(canonical_json_bytes(body)):
        raise VerificationError("run contract self-digest mismatch")
    if Path(str(contract.get("work_root", ""))).resolve() != work_root:
        raise VerificationError("run contract work_root mismatch")
    if contract_path.resolve() != work_root / "contract.json":
        raise VerificationError("run contract must be work_root/contract.json")
    if require_e_drive:
        _assert_e(contract_path, "run contract")

    split_path, split_record = _verify_record(
        contract.get("split_manifest"),
        label="split_manifest",
        require_e_drive=require_e_drive,
        max_bytes=MAX_JSON_BYTES * 8,
    )
    source_path, source_record = _verify_record(
        contract.get("source_manifest"),
        label="source_manifest",
        require_e_drive=require_e_drive,
        max_bytes=MAX_JSON_BYTES * 32,
    )
    if split_record["sha256"] != PINNED_SPLIT_SHA256:
        raise VerificationError("split manifest is not the pinned source-disjoint split")
    if source_record["sha256"] != PINNED_SOURCE_SHA256:
        raise VerificationError("source manifest is not pinned")
    if "paired_checkpoint" in contract:
        raise VerificationError(
            "legacy paired checkpoint is forbidden; M144 encoder must be FIT-only scratch"
        )
    if require_e_drive and (split_path.drive.upper() != "E:" or source_path.drive.upper() != "E:"):
        raise VerificationError("pinned manifests must remain on E:")

    source_files = _source_digest_map(contract.get("source_files"), repo_root)
    runtime = contract.get("runtime")
    _assert_equivalent(current_runtime_record(), runtime, "runtime")
    partitions = contract.get("partitions")
    if not isinstance(partitions, dict):
        raise VerificationError("run contract lacks partitions")
    for label, count in (("fit", FIT_COUNT), ("cal", PARTITION_COUNT), ("dev", PARTITION_COUNT)):
        row = partitions.get(label)
        if not isinstance(row, dict) or row.get("count") != count:
            raise VerificationError(f"run contract {label} count mismatch")
    reserve = contract.get("reserve")
    if not isinstance(reserve, dict) or reserve.get("accessed") is not False:
        raise VerificationError("reserve/test access is not sealed")
    config = contract.get("config")
    if not isinstance(config, dict):
        raise VerificationError("run contract lacks config")
    encoder_config = config.get("encoder")
    _assert_equivalent(
        ENCODER_RECIPE, encoder_config, "run contract.config.encoder"
    )
    _assert_equivalent(PRIOR_EVIDENCE, contract.get("prior_evidence"), "prior_evidence")

    report_contract = {
        "run_contract_sha256": claimed_contract_sha,
        "split_manifest": split_record,
        "source_manifest": source_record,
        "source_files": source_files,
        "runtime": runtime,
        "prior_evidence": PRIOR_EVIDENCE,
        "cal_count": PARTITION_COUNT,
        "dev_count": PARTITION_COUNT,
    }
    return contract, raw, report_contract


def load_pinned_partition_identity(
    split_path: Path, source_path: Path
) -> dict[str, dict[str, Any]]:
    try:
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot parse pinned split/source manifests: {error}") from error
    splits = split_payload.get("splits")
    files = source_payload.get("files")
    if not isinstance(splits, dict) or not isinstance(files, dict):
        raise VerificationError("pinned manifests lack splits/files")
    expected_counts = {
        "fit": FIT_COUNT,
        "cal": PARTITION_COUNT,
        "dev": PARTITION_COUNT,
        "reserve": 300,
    }
    all_names: list[str] = []
    raw_names: dict[str, list[str]] = {}
    for partition, count in expected_counts.items():
        values = splits.get(partition)
        if not isinstance(values, list) or len(values) != count:
            raise VerificationError(f"pinned {partition} split count mismatch")
        names = [str(value) for value in values]
        if len(set(names)) != count:
            raise VerificationError(f"pinned {partition} split repeats a board")
        raw_names[partition] = names
        all_names.extend(names)
    if len(set(all_names)) != sum(expected_counts.values()) or set(files) != set(all_names):
        raise VerificationError("pinned partitions overlap or source manifest coverage drifts")
    group_for_name: dict[str, str] = {}
    for name in all_names:
        row = files.get(name)
        if not isinstance(row, dict) or not isinstance(row.get("source_group"), str):
            raise VerificationError(f"source manifest lacks source_group for {name}")
        group_for_name[name] = row["source_group"]
    unique_groups = sorted(set(group_for_name.values()))
    group_id = {name: index for index, name in enumerate(unique_groups)}
    identities: dict[str, dict[str, Any]] = {}
    partition_groups: dict[str, set[str]] = {}
    pattern = re.compile(r"^img_(\d{6})\.png$")
    for partition, names in raw_names.items():
        try:
            ordered = sorted(
                names,
                key=lambda name: int(pattern.fullmatch(name).group(1)),  # type: ignore[union-attr]
            )
        except AttributeError as error:
            raise VerificationError(f"invalid board name in {partition} split") from error
        boards = np.asarray(
            [int(pattern.fullmatch(name).group(1)) for name in ordered], dtype=np.int64  # type: ignore[union-attr]
        )
        source_ids = np.asarray(
            [group_id[group_for_name[name]] for name in ordered], dtype=np.int64
        )
        identities[partition] = {
            "names": ordered,
            "board_id": boards,
            "source_group_id": source_ids,
        }
        partition_groups[partition] = {group_for_name[name] for name in ordered}
    labels = list(partition_groups)
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            if partition_groups[left] & partition_groups[right]:
                raise VerificationError(f"source-group contamination between {left} and {right}")
    return identities


def _validate_npz_members(path: Path, keys: Sequence[str]) -> None:
    expected = {f"{key}.npy" for key in keys}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise VerificationError(f"invalid NPZ {path.name}: {error}") from error
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or set(names) != expected:
        raise VerificationError(f"{path.name} has duplicate/unknown/missing members")
    if sum(int(info.file_size) for info in infos) > MAX_NPZ_UNCOMPRESSED_BYTES:
        raise VerificationError(f"{path.name} expands beyond the numeric evidence cap")


def _load_npz(path: Path, keys: Sequence[str]) -> dict[str, np.ndarray]:
    _validate_npz_members(path, keys)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(keys) or len(archive.files) != len(keys):
                raise VerificationError(f"{path.name} array key mismatch")
            return {key: np.array(archive[key], copy=True) for key in keys}
    except VerificationError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise VerificationError(f"cannot load numeric NPZ {path.name}: {error}") from error


def _validate_ids(array: np.ndarray, count: int, label: str, *, sorted_ids: bool = False) -> None:
    if array.dtype != np.dtype(np.int64) or array.shape != (count,):
        raise VerificationError(f"{label} must be exactly int64[{count}]")
    if np.any(array < 0):
        raise VerificationError(f"{label} contains a negative ID")
    if sorted_ids and not np.all(np.diff(array) > 0):
        raise VerificationError(f"{label} must be strictly increasing")


def _validate_ssim(array: np.ndarray, count: int, label: str) -> None:
    if array.dtype != np.dtype(np.float64) or array.shape != (count,):
        raise VerificationError(f"{label} must be exactly float64[{count}]")
    if not np.isfinite(array).all() or np.any(array < -1.0) or np.any(array > 1.0):
        raise VerificationError(f"{label} contains invalid SSIM values")


def load_oracle_npz(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, ORACLE_KEYS)
    _validate_ids(arrays["board_id"], PARTITION_COUNT, "oracle.board_id", sorted_ids=True)
    _validate_ids(arrays["source_group_id"], PARTITION_COUNT, "oracle.source_group_id")
    if np.unique(arrays["source_group_id"]).size < 2:
        raise VerificationError("oracle evidence needs at least two source groups")
    for key in ORACLE_SSIM_KEYS:
        _validate_ssim(arrays[key], PARTITION_COUNT, f"oracle.{key}")
    if arrays["flat_rgb"].dtype != np.float32 or arrays["flat_rgb"].shape != (
        PARTITION_COUNT, 3
    ):
        raise VerificationError("oracle.flat_rgb must be exactly float32[670,3]")
    if arrays["oracle_coeff"].dtype != np.float32 or arrays["oracle_coeff"].shape != (
        PARTITION_COUNT, 96
    ):
        raise VerificationError("oracle.oracle_coeff must be exactly float32[670,96]")
    if not np.isfinite(arrays["flat_rgb"]).all() or not np.isfinite(
        arrays["oracle_coeff"]
    ).all():
        raise VerificationError("oracle prediction evidence contains non-finite values")
    if np.any(arrays["flat_rgb"] < 0.0) or np.any(arrays["flat_rgb"] > 1.0):
        raise VerificationError("oracle.flat_rgb must stay in [0,1]")
    return arrays


def load_raw_npz(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, RAW_KEYS)
    for key in RAW_ID_KEYS:
        _validate_ids(
            arrays[key], PARTITION_COUNT, f"{path.name}.{key}", sorted_ids=(key == "board_id")
        )
    if np.unique(arrays["source_group_id"]).size < 2:
        raise VerificationError(f"{path.name} needs at least two source groups")
    for key in RAW_SSIM_KEYS:
        _validate_ssim(arrays[key], PARTITION_COUNT, f"{path.name}.{key}")
    prediction_shapes = {
        "flat_rgb": (PARTITION_COUNT, 3),
        "dct_full_coeff": (PARTITION_COUNT, 96),
        "dct_blind_coeff": (PARTITION_COUNT, 96),
        "dct_swapped_coeff": (PARTITION_COUNT, 96),
        "rgb8_full_residual": (PARTITION_COUNT, 192),
        "rgb8_blind_residual": (PARTITION_COUNT, 192),
    }
    for key, shape in prediction_shapes.items():
        value = arrays[key]
        if value.dtype != np.float32 or value.shape != shape:
            raise VerificationError(f"{path.name}.{key} must be exactly float32{shape}")
        if not np.isfinite(value).all():
            raise VerificationError(f"{path.name}.{key} contains non-finite values")
    if np.any(arrays["flat_rgb"] < 0.0) or np.any(arrays["flat_rgb"] > 1.0):
        raise VerificationError(f"{path.name}.flat_rgb must stay in [0,1]")
    return arrays


def load_encoder_rank_npz(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, ENCODER_RANK_KEYS)
    _validate_ids(
        arrays["board_id"], PARTITION_COUNT, "encoder.board_id", sorted_ids=True
    )
    _validate_ids(
        arrays["source_group_id"], PARTITION_COUNT, "encoder.source_group_id"
    )
    if np.unique(arrays["source_group_id"]).size < 2:
        raise VerificationError("encoder CAL ranks need at least two source groups")
    for key in ("dirty_to_clean_rank", "clean_to_dirty_rank"):
        value = arrays[key]
        if value.dtype != np.uint16 or value.shape != (PARTITION_COUNT, 576):
            raise VerificationError(f"encoder.{key} must be uint16[670,576]")
        if np.any(value < 1) or np.any(value > 576):
            raise VerificationError(f"encoder.{key} must contain 1-based ranks in [1,576]")
    return arrays


def summarize_encoder_ranks(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    def direction(value: np.ndarray) -> dict[str, float]:
        matrix = np.asarray(value, dtype=np.float64)
        ranks = matrix.reshape(-1)
        return {
            "micro_r1": float(np.mean(ranks <= 1.0)),
            "micro_r5": float(np.mean(ranks <= 5.0)),
            "micro_r20": float(np.mean(ranks <= 20.0)),
            "macro_r1": float(np.mean(np.mean(matrix <= 1.0, axis=1))),
            "macro_r5": float(np.mean(np.mean(matrix <= 5.0, axis=1))),
            "macro_r20": float(np.mean(np.mean(matrix <= 20.0, axis=1))),
            "median_rank": float(np.median(ranks)),
            "mrr": float(np.mean(1.0 / ranks)),
        }

    return {
        "n_boards": PARTITION_COUNT,
        "tiles_per_board": 576,
        "n_tile_queries": PARTITION_COUNT * 576,
        "dirty_to_clean": direction(arrays["dirty_to_clean_rank"]),
        "clean_to_dirty": direction(arrays["clean_to_dirty_rank"]),
    }


def evaluate_encoder_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    dirty = summary["dirty_to_clean"]
    thresholds = {"r1": 0.20, "r5": 0.45, "r20": 0.70}
    checks: dict[str, Any] = {}
    for name, threshold in thresholds.items():
        observed = float(dirty[f"micro_{name}"])
        checks[f"dirty_to_clean_{name}"] = {
            "observed": observed,
            "operator": ">=",
            "threshold": threshold,
            "passed": bool(observed >= threshold),
        }
    return {
        "checks": checks,
        "passed": bool(all(record["passed"] for record in checks.values())),
    }


def _zigzag32() -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    side = 16
    for diagonal in range(2 * side - 1):
        low = max(0, diagonal - side + 1)
        high = min(side - 1, diagonal)
        rows = range(high, low - 1, -1) if diagonal % 2 == 0 else range(low, high + 1)
        for row in rows:
            result.append((row, diagonal - row))
            if len(result) == 32:
                return tuple(result)
    raise AssertionError("failed to build DCT32 zigzag")


def _torch_dct_matrix(device: Any) -> Any:
    import torch

    size = 16
    samples = torch.arange(size, device=device, dtype=torch.float32) + 0.5
    frequencies = torch.arange(size, device=device, dtype=torch.float32).unsqueeze(1)
    matrix = torch.cos((torch.pi / float(size)) * frequencies * samples)
    scale = torch.full(
        (size,), (2.0 / float(size)) ** 0.5, device=device, dtype=torch.float32
    )
    scale[0] = (1.0 / float(size)) ** 0.5
    return matrix * scale.unsqueeze(1)


def _encode_oracle_coefficients(target: Any, flat: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    residual = functional.interpolate(
        target - flat[:, :, None, None], size=(16, 16), mode="area"
    )
    matrix = _torch_dct_matrix(target.device)
    spectrum = torch.matmul(torch.matmul(matrix, residual), matrix.transpose(0, 1))
    zigzag = _zigzag32()
    rows = torch.tensor([row for row, _ in zigzag], dtype=torch.long, device=target.device)
    columns = torch.tensor(
        [column for _, column in zigzag], dtype=torch.long, device=target.device
    )
    return spectrum[:, :, rows, columns].reshape(target.shape[0], 96)


def _torch_bicubic_matrix(source: int, target: int, device: Any) -> Any:
    """Rebuild the frozen bicubic resize weights, independently of the core.

    The rendering contract is a pair of matrix multiplications against weights
    read off ``interpolate(..., align_corners=False)`` one-hot basis vectors,
    NOT a direct call to the resize kernel.  Calling the kernel here instead
    was worth up to 2.4e-07 per pixel, which survives the uint8 quantisation on
    roughly a hundred pixels per board and therefore breaks the 1e-12 tolerance
    that ``_recompute_stored_scores`` applies to the SSIM it recomputes.  This
    reimplements the contract rather than importing it, so the verifier stays
    an independent second implementation.
    """

    import torch
    import torch.nn.functional as functional

    basis = torch.eye(source, dtype=torch.float32).reshape(source, 1, source, 1)
    with torch.no_grad():
        resized = functional.interpolate(
            basis, size=(target, 1), mode="bicubic", align_corners=False
        )
    return resized[:, 0, :, 0].transpose(0, 1).contiguous().to(device)


def _fixed_bicubic_to_480(value: Any) -> Any:
    import torch

    source = int(value.shape[-1])
    if source == 480:
        return value
    matrix = _torch_bicubic_matrix(source, 480, value.device)
    return torch.matmul(torch.matmul(matrix, value), matrix.transpose(0, 1))


def _render_dct_coefficients(coefficients: Any, flat: Any) -> Any:
    import torch

    batch = coefficients.shape[0]
    retained = coefficients.reshape(batch, 3, 32)
    spectrum = torch.zeros((batch, 3, 16, 16), dtype=torch.float32)
    zigzag = _zigzag32()
    rows = torch.tensor([row for row, _ in zigzag], dtype=torch.long)
    columns = torch.tensor([column for _, column in zigzag], dtype=torch.long)
    spectrum[:, :, rows, columns] = retained
    matrix = _torch_dct_matrix(spectrum.device)
    residual = torch.matmul(
        torch.matmul(matrix.transpose(0, 1), spectrum), matrix
    )
    residual = _fixed_bicubic_to_480(residual)
    return (residual + flat[:, :, None, None]).clamp(0.0, 1.0)


def _render_rgb8_residual(residual: Any, flat: Any) -> Any:
    field = residual.reshape(residual.shape[0], 3, 8, 8)
    field = _fixed_bicubic_to_480(field)
    return (field + flat[:, :, None, None]).clamp(0.0, 1.0)


def _load_target_uint8(path: Path) -> np.ndarray:
    from PIL import Image

    try:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise VerificationError(f"target is not original RGB: {path}")
            array = np.asarray(image, dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise VerificationError(f"cannot decode target {path}: {error}") from error
    if array.shape != (480, 480, 3):
        raise VerificationError(f"target must be uint8 RGB 480x480: {path}")
    return array


def _quantize_prediction(image_chw: np.ndarray) -> np.ndarray:
    image_hwc = np.moveaxis(image_chw, 0, -1)
    return np.clip(np.rint(image_hwc * np.float32(255.0)), 0.0, 255.0).astype(
        np.uint8
    )


def _official_uint8_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            prediction,
            target,
            channel_axis=2,
            data_range=255,
            win_size=7,
            gaussian_weights=False,
            use_sample_covariance=True,
        )
    )


def verify_prediction_evidence(
    *,
    flat_rgb: np.ndarray,
    target_paths: Sequence[Path],
    stored_scores: Mapping[str, np.ndarray],
    dct_predictions: Mapping[str, np.ndarray] | None = None,
    rgb_predictions: Mapping[str, np.ndarray] | None = None,
    oracle_coeff_reference: np.ndarray | None = None,
    batch_size: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render prediction tensors on CPU and recompute official uint8 SSIM."""

    import torch

    dct_predictions = {} if dct_predictions is None else dict(dct_predictions)
    rgb_predictions = {} if rgb_predictions is None else dict(rgb_predictions)
    expected_score_keys = {"flat_ssim", "target_oracle_dct_ssim"}
    expected_score_keys.update(dct_predictions)
    expected_score_keys.update(rgb_predictions)
    if set(stored_scores) != expected_score_keys:
        raise VerificationError("stored score keys do not match prediction evidence")
    count = len(target_paths)
    if count != PARTITION_COUNT or flat_rgb.shape != (count, 3):
        raise VerificationError("prediction evidence/targets are not a 670-board partition")
    recomputed = {
        key: np.empty(count, dtype=np.float64) for key in sorted(expected_score_keys)
    }
    coefficient_max_abs = 0.0
    coefficient_mean_numerator = 0.0
    coefficient_elements = 0
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        targets_uint8 = np.stack(
            [_load_target_uint8(path) for path in target_paths[start:stop]], axis=0
        )
        targets = torch.from_numpy(targets_uint8).permute(0, 3, 1, 2).float().div_(255.0)
        flat = torch.from_numpy(np.asarray(flat_rgb[start:stop], dtype=np.float32))
        oracle_coeff = _encode_oracle_coefficients(targets, flat)
        if oracle_coeff_reference is not None:
            reference = torch.from_numpy(
                np.asarray(oracle_coeff_reference[start:stop], dtype=np.float32)
            )
            difference = (oracle_coeff - reference).abs().numpy()
            coefficient_max_abs = max(coefficient_max_abs, float(difference.max(initial=0.0)))
            coefficient_mean_numerator += float(difference.sum(dtype=np.float64))
            coefficient_elements += int(difference.size)
            if not np.allclose(
                oracle_coeff.numpy(), reference.numpy(), rtol=2.0e-5, atol=2.0e-5
            ):
                raise VerificationError("stored oracle_coeff differs from independent DCT32 encoding")
        rendered: dict[str, np.ndarray] = {
            "flat_ssim": flat[:, :, None, None]
            .expand(-1, -1, 480, 480)
            .contiguous()
            .numpy(),
            "target_oracle_dct_ssim": _render_dct_coefficients(
                oracle_coeff, flat
            ).numpy(),
        }
        for score_key, coefficient_values in dct_predictions.items():
            coefficients = torch.from_numpy(
                np.asarray(coefficient_values[start:stop], dtype=np.float32)
            )
            rendered[score_key] = _render_dct_coefficients(coefficients, flat).numpy()
        for score_key, residual_values in rgb_predictions.items():
            residual = torch.from_numpy(
                np.asarray(residual_values[start:stop], dtype=np.float32)
            )
            rendered[score_key] = _render_rgb8_residual(residual, flat).numpy()
        for local_index in range(stop - start):
            target = targets_uint8[local_index]
            for score_key, images in rendered.items():
                recomputed[score_key][start + local_index] = _official_uint8_ssim(
                    _quantize_prediction(images[local_index]), target
                )

    errors: dict[str, Any] = {}
    for key in sorted(expected_score_keys):
        claimed = np.asarray(stored_scores[key], dtype=np.float64)
        difference = np.abs(recomputed[key] - claimed)
        maximum = float(difference.max(initial=0.0))
        mean = float(difference.mean())
        errors[key] = {"max_abs": maximum, "mean_abs": mean, "atol": 1.0e-12}
        if maximum > 1.0e-12:
            raise VerificationError(
                f"{key} differs from independent official uint8 SSIM: max_abs={maximum}"
            )
    diagnostic: dict[str, Any] = {"stored_score_error": errors}
    if oracle_coeff_reference is not None:
        diagnostic["oracle_coeff_error"] = {
            "max_abs": coefficient_max_abs,
            "mean_abs": (
                coefficient_mean_numerator / coefficient_elements
                if coefficient_elements
                else 0.0
            ),
            "atol": 2.0e-5,
            "rtol": 2.0e-5,
        }
    return recomputed, diagnostic


def _validate_feature_rows(features: np.ndarray, count: int, label: str) -> None:
    if features.dtype != np.dtype(np.float64) or features.shape != (count, PALETTE_DIM):
        raise VerificationError(f"{label} must be exactly float64[{count},60]")
    if not np.isfinite(features).all():
        raise VerificationError(f"{label} contains non-finite values")
    quantile_blocks = features[:, :52].reshape(count, 4, 13)
    if np.any(np.diff(quantile_blocks, axis=2) < 0.0):
        raise VerificationError(f"{label} quantile blocks are not nondecreasing")
    means = features[:, 52:56]
    standard_deviations = features[:, 56:60]
    if np.any(means < quantile_blocks[:, :, 0]) or np.any(
        means > quantile_blocks[:, :, -1]
    ):
        raise VerificationError(f"{label} tail means fall outside quantile extrema")
    if np.any(quantile_blocks < 0.0) or np.any(quantile_blocks > 1.0):
        raise VerificationError(f"{label} dirty-tile statistics must remain in [0,1]")
    if np.any(means < 0.0) or np.any(means > 1.0) or np.any(standard_deviations < 0.0):
        raise VerificationError(f"{label} has impossible mean/std coordinates")


def load_fit_palette_npz(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, FIT_PALETTE_KEYS)
    _validate_ids(arrays["fit_board_id"], FIT_COUNT, "fit_board_id", sorted_ids=True)
    _validate_feature_rows(arrays["dirty_feature60"], FIT_COUNT, "fit.dirty_feature60")
    for key in ("mean60", "scale60"):
        value = arrays[key]
        if value.dtype != np.dtype(np.float64) or value.shape != (PALETTE_DIM,):
            raise VerificationError(f"fit.{key} must be exactly float64[60]")
        if not np.isfinite(value).all():
            raise VerificationError(f"fit.{key} contains non-finite values")
    features = arrays["dirty_feature60"]
    expected_mean = np.mean(features, axis=0, dtype=np.float64)
    expected_scale = np.maximum(
        np.std(features, axis=0, dtype=np.float64, ddof=0), PALETTE_EPSILON
    )
    if not np.allclose(arrays["mean60"], expected_mean, rtol=0.0, atol=1.0e-14):
        raise VerificationError("fit mean60 is not the float64 FIT population mean")
    if not np.allclose(arrays["scale60"], expected_scale, rtol=0.0, atol=1.0e-14):
        raise VerificationError("fit scale60 is not max(FIT population std,1e-6)")
    if np.any(arrays["scale60"] < PALETTE_EPSILON):
        raise VerificationError("fit scale60 violates the 1e-6 floor")
    return arrays


def load_swap_npz(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, SWAP_KEYS)
    for key in SWAP_ID_KEYS:
        _validate_ids(
            arrays[key], PARTITION_COUNT, f"{path.name}.{key}", sorted_ids=(key == "board_id")
        )
    _validate_feature_rows(arrays["dirty_feature60"], PARTITION_COUNT, f"{path.name}.feature")
    return arrays


def canonical_cycle_ids(board_ids: np.ndarray, donor_board_ids: np.ndarray) -> np.ndarray:
    """Return dense permutation-cycle IDs in smallest-unvisited-board order."""

    count = int(board_ids.size)
    if donor_board_ids.shape != board_ids.shape or np.unique(donor_board_ids).size != count:
        raise VerificationError("swap donors must be a within-partition bijection")
    index_for_board = {int(board): index for index, board in enumerate(board_ids)}
    if set(int(value) for value in donor_board_ids) != set(index_for_board):
        raise VerificationError("swap donors leave the partition")
    donor_index = np.asarray(
        [index_for_board[int(value)] for value in donor_board_ids], dtype=np.int64
    )
    if np.any(donor_index == np.arange(count, dtype=np.int64)):
        raise VerificationError("swap assignment contains a fixed point")
    result = np.full(count, -1, dtype=np.int64)
    visited = np.zeros(count, dtype=bool)
    cycle_identifier = 0
    for start in range(count):
        if visited[start]:
            continue
        members: list[int] = []
        current = start
        while not visited[current]:
            visited[current] = True
            members.append(current)
            current = int(donor_index[current])
        if current != start or len(members) < 2:
            raise VerificationError("donor mapping is not a disjoint derangement cycle cover")
        result[np.asarray(members, dtype=np.int64)] = cycle_identifier
        cycle_identifier += 1
    if np.any(result < 0):
        raise VerificationError("failed to assign canonical swap cycles")
    return result


def recompute_palette_assignment(
    features: np.ndarray,
    source_group_ids: np.ndarray,
    mean60: np.ndarray,
    scale60: np.ndarray,
) -> np.ndarray:
    """Re-run the frozen no-jitter SciPy Hungarian donor assignment."""

    whitened = (features - mean60[None, :]) / scale60[None, :]
    count = whitened.shape[0]
    cost = np.empty((count, count), dtype=np.float64)
    for row in range(count):
        difference = whitened - whitened[row]
        cost[row] = np.sum(difference * difference, axis=1, dtype=np.float64)
    forbidden = source_group_ids[:, None] == source_group_ids[None, :]
    forbidden[np.arange(count), np.arange(count)] = True
    if np.any(np.all(forbidden, axis=1)) or np.any(np.all(forbidden, axis=0)):
        raise VerificationError("palette assignment has no group-safe perfect matching")
    cost[forbidden] = np.inf
    try:
        rows, columns = linear_sum_assignment(cost)
    except ValueError as error:
        raise VerificationError(f"palette Hungarian assignment failed: {error}") from error
    if not np.array_equal(rows, np.arange(count, dtype=rows.dtype)):
        raise VerificationError("SciPy returned non-canonical row order")
    return columns.astype(np.int64, copy=False)


def verify_swap_semantics(
    swap: Mapping[str, np.ndarray],
    fit: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    board_ids = swap["board_id"]
    source_ids = swap["source_group_id"]
    donor_ids = swap["donor_board_id"]
    donor_source_ids = swap["donor_source_group_id"]
    index_for_board = {int(board): index for index, board in enumerate(board_ids)}
    if set(int(value) for value in donor_ids) != set(index_for_board):
        raise VerificationError("swap donor_board_id is not a partition bijection")
    donor_indices = np.asarray(
        [index_for_board[int(value)] for value in donor_ids], dtype=np.int64
    )
    expected_donor_sources = source_ids[donor_indices]
    if not np.array_equal(donor_source_ids, expected_donor_sources):
        raise VerificationError("donor_source_group_id does not match donor_board_id")
    if np.any(source_ids == donor_source_ids):
        raise VerificationError("swap assignment contains a same-source-group donor")
    expected_indices = recompute_palette_assignment(
        swap["dirty_feature60"], source_ids, fit["mean60"], fit["scale60"]
    )
    if not np.array_equal(donor_indices, expected_indices):
        raise VerificationError("committed donors differ from frozen no-jitter Hungarian result")
    expected_cycles = canonical_cycle_ids(board_ids, donor_ids)
    if not np.array_equal(swap["swap_cycle_id"], expected_cycles):
        raise VerificationError("swap_cycle_id is not the canonical donor cycle decomposition")
    _, lengths = np.unique(expected_cycles, return_counts=True)
    if lengths.size < 64:
        raise VerificationError(
            f"swap cycle bootstrap is under-clustered: {lengths.size} < 64 cycles"
        )
    sizes, counts = np.unique(lengths, return_counts=True)
    return {
        "count": int(lengths.size),
        "min_size": int(lengths.min()),
        "max_size": int(lengths.max()),
        "mean_size": float(PARTITION_COUNT / int(lengths.size)),
        "median_size": float(np.median(lengths)),
        "size_histogram": [
            [int(size), int(count)] for size, count in zip(sizes, counts)
        ],
    }


def _group_ids(groups: np.ndarray) -> tuple[np.ndarray, int]:
    mapping: dict[int, int] = {}
    identifiers = np.empty(groups.size, dtype=np.int64)
    for index, raw in enumerate(groups):
        identifiers[index] = mapping.setdefault(int(raw), len(mapping))
    return identifiers, len(mapping)


def one_sided_cluster_bootstrap_lower(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    confidence: float,
    seed: int,
) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (PARTITION_COUNT,) or groups.shape != vector.shape:
        raise VerificationError("bootstrap inputs must be aligned 670-vectors")
    group_ids, group_count = _group_ids(groups)
    if group_count < 2:
        raise VerificationError("cluster bootstrap needs at least two groups")
    sums = np.bincount(group_ids, weights=vector, minlength=group_count)
    sizes = np.bincount(group_ids, minlength=group_count).astype(np.float64)
    rng = np.random.default_rng(seed)
    replicates = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 1_024):
        stop = min(BOOTSTRAP_SAMPLES, start + 1_024)
        selected = rng.integers(0, group_count, size=(stop - start, group_count))
        replicates[start:stop] = sums[selected].sum(axis=1) / sizes[selected].sum(axis=1)
    return float(np.quantile(replicates, 1.0 - confidence, method="linear"))


def summarize_oracle(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    flat = arrays["flat_ssim"]
    oracle = arrays["target_oracle_dct_ssim"]
    return {
        "n_boards": PARTITION_COUNT,
        "means": {
            "flat": float(flat.mean()),
            "target_oracle_dct": float(oracle.mean()),
        },
        "gains": {"target_oracle_dct": float((oracle - flat).mean())},
    }


def evaluate_oracle_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    observed = float(summary["gains"]["target_oracle_dct"])
    passed = bool(observed >= CAL_THRESHOLDS["oracle_gain"])
    return {
        "checks": {
            "oracle_gain": {
                "observed": observed,
                "operator": ">=",
                "threshold": CAL_THRESHOLDS["oracle_gain"],
                "passed": passed,
            }
        },
        "passed": passed,
    }


def summarize_arrays(arrays: Mapping[str, np.ndarray], *, confidence: float) -> dict[str, Any]:
    raw = {
        "flat": arrays["flat_ssim"],
        "dct_full": arrays["dct_full_ssim"],
        "dct_blind": arrays["dct_blind_ssim"],
        "dct_swapped": arrays["dct_swapped_ssim"],
        "rgb8_full": arrays["rgb8_full_ssim"],
        "rgb8_blind": arrays["rgb8_blind_ssim"],
        "target_oracle_dct": arrays["target_oracle_dct_ssim"],
    }
    means = {name: float(values.mean()) for name, values in raw.items()}
    gains = {
        name: float((values - raw["flat"]).mean())
        for name, values in raw.items()
        if name != "flat"
    }
    full_blind = raw["dct_full"] - raw["dct_blind"]
    full_swapped = raw["dct_full"] - raw["dct_swapped"]
    representation = full_blind - (raw["rgb8_full"] - raw["rgb8_blind"])

    def contrast(
        values: np.ndarray,
        groups: np.ndarray,
        seed_offset: int,
        include_wins: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mean": float(values.mean()),
            "lower": one_sided_cluster_bootstrap_lower(
                values,
                groups,
                confidence=confidence,
                seed=BOOTSTRAP_SEED + seed_offset,
            ),
            "confidence": float(confidence),
        }
        if include_wins:
            result["win_fraction"] = float(np.mean(values > 0.0))
        return result

    return {
        "n_boards": PARTITION_COUNT,
        "n_source_groups": int(np.unique(arrays["source_group_id"]).size),
        "n_swap_cycles": int(np.unique(arrays["swap_cycle_id"]).size),
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": float(confidence),
        },
        "means": means,
        "gains": gains,
        "contrasts": {
            "full_minus_blind": contrast(
                full_blind, arrays["source_group_id"], 0, True
            ),
            "full_minus_swapped": contrast(
                full_swapped, arrays["swap_cycle_id"], 1, True
            ),
            "representation_delta": contrast(
                representation, arrays["source_group_id"], 2, False
            ),
        },
    }


def _gate_record(
    observed: float,
    threshold: float,
    *,
    lower: float | None = None,
    require_positive_lower: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "observed": float(observed),
        "threshold": float(threshold),
        "passed": bool(observed >= threshold),
    }
    if lower is not None:
        record.update(
            lower=float(lower), lower_threshold=0.0, lower_passed=bool(lower > 0.0)
        )
        if require_positive_lower:
            record["passed"] = bool(record["passed"] and lower > 0.0)
    return record


def evaluate_cal_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    gains, contrasts = summary["gains"], summary["contrasts"]
    checks = {
        "oracle_gain": _gate_record(
            gains["target_oracle_dct"], CAL_THRESHOLDS["oracle_gain"]
        ),
        "full_gain": _gate_record(gains["dct_full"], CAL_THRESHOLDS["full_gain"]),
        "full_blind": _gate_record(
            contrasts["full_minus_blind"]["mean"],
            CAL_THRESHOLDS["full_blind"],
            lower=contrasts["full_minus_blind"]["lower"],
            require_positive_lower=True,
        ),
        "full_swapped": _gate_record(
            contrasts["full_minus_swapped"]["mean"],
            CAL_THRESHOLDS["full_swapped"],
            lower=contrasts["full_minus_swapped"]["lower"],
        ),
        "representation_delta": _gate_record(
            contrasts["representation_delta"]["mean"],
            CAL_THRESHOLDS["representation_delta"],
            lower=contrasts["representation_delta"]["lower"],
        ),
    }
    return {
        "confidence": CAL_CONFIDENCE,
        "checks": checks,
        "passed": bool(all(row["passed"] for row in checks.values())),
    }


def evaluate_dev_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    gains, contrasts = summary["gains"], summary["contrasts"]
    checks = {
        "full_gain": _gate_record(gains["dct_full"], DEV_THRESHOLDS["full_gain"]),
        "full_blind": _gate_record(
            contrasts["full_minus_blind"]["mean"],
            DEV_THRESHOLDS["full_blind"],
            lower=contrasts["full_minus_blind"]["lower"],
            require_positive_lower=True,
        ),
        "full_swapped": _gate_record(
            contrasts["full_minus_swapped"]["mean"],
            DEV_THRESHOLDS["full_swapped"],
            lower=contrasts["full_minus_swapped"]["lower"],
            require_positive_lower=True,
        ),
        "representation_delta": _gate_record(
            contrasts["representation_delta"]["mean"],
            DEV_THRESHOLDS["representation_delta"],
            lower=contrasts["representation_delta"]["lower"],
            require_positive_lower=True,
        ),
        "full_blind_win": _gate_record(
            contrasts["full_minus_blind"]["win_fraction"],
            DEV_THRESHOLDS["full_blind_win"],
        ),
        "full_swapped_win": _gate_record(
            contrasts["full_minus_swapped"]["win_fraction"],
            DEV_THRESHOLDS["full_swapped_win"],
        ),
    }
    return {
        "confidence": DEV_CONFIDENCE,
        "checks": checks,
        "passed": bool(all(row["passed"] for row in checks.values())),
    }


def _assert_partition_alignment(
    raw: Mapping[str, np.ndarray], swap: Mapping[str, np.ndarray], label: str
) -> None:
    for key in ("board_id", "source_group_id", "swap_cycle_id"):
        if not np.array_equal(raw[key], swap[key]):
            raise VerificationError(f"{label} raw/swap {key} mismatch")


def _assert_disjoint(
    left: np.ndarray, right: np.ndarray, label: str
) -> None:
    if np.intersect1d(left, right).size:
        raise VerificationError(f"split contamination: {label} overlap")


def _assert_dev_sealed(work_root: Path) -> None:
    candidates: list[Path] = []
    for directory in (work_root / "cache", work_root / "artifacts", work_root / "receipts"):
        if directory.is_dir():
            candidates.extend(
                path for path in directory.iterdir() if path.is_file() and "dev" in path.name.lower()
            )
    if candidates:
        raise VerificationError(
            "DEV must remain sealed after a failed opening gate; found "
            + ", ".join(str(path) for path in sorted(candidates))
        )


def _assert_oracle_reject_sealed(work_root: Path) -> None:
    checkpoint_root = work_root / "checkpoints"
    if checkpoint_root.is_dir() and any(path.is_file() for path in checkpoint_root.rglob("*")):
        raise VerificationError("oracle reject must not leave model/optimizer checkpoints")
    for name in (
        "swap_cal.npz",
        "m144_cal_raw.npz",
        "swap_dev.npz",
        "m144_dev_raw.npz",
    ):
        if (work_root / "artifacts" / name).exists():
            raise VerificationError(f"oracle reject must not leave {name}")
    receipt_root = work_root / "receipts"
    if receipt_root.is_dir():
        forbidden = [
            path
            for path in receipt_root.iterdir()
            if path.is_file()
            and (
                path.name.startswith("checkpoint_step_")
                or path.name in {"swap_cal.json", "cal_raw.json", "fit_targets.json"}
            )
        ]
        if forbidden:
            raise VerificationError("oracle reject left learned CAL/checkpoint receipts")
    _assert_dev_sealed(work_root)


def _terminal(cal_passed: bool, dev_passed: bool | None) -> tuple[str, str, str]:
    if not cal_passed:
        return "cal_reject", CAL_STAGE, "KILL_DCT_WHERE"
    if dev_passed is None:
        raise VerificationError("CAL pass must open exactly one DEV evaluation")
    if dev_passed:
        return "dev_pass", DEV_STAGE, "PROMOTE_DCT_WHERE"
    return "dev_reject", DEV_STAGE, "KILL_DCT_WHERE"


def verify_report(
    *,
    work_root: str | Path,
    contract_path: str | Path,
    report_path: str | Path,
    require_e_drive: bool = True,
) -> dict[str, Any]:
    root = Path(work_root).resolve()
    contract_file = Path(contract_path).resolve()
    report_file = Path(report_path).resolve()
    if require_e_drive:
        _assert_e(root, "work root")
    if report_file != root / "artifacts" / "m144_report.json":
        raise VerificationError("report must be work_root/artifacts/m144_report.json")
    if not _inside(report_file, root):
        raise VerificationError("report escapes work root")
    repo_root = Path(__file__).resolve().parents[1]
    run_contract, contract_bytes, expected_report_contract = validate_run_contract(
        contract_file,
        work_root=root,
        repo_root=repo_root,
        require_e_drive=require_e_drive,
    )
    report, report_bytes = load_canonical_json(report_file, label="terminal report")
    _exact_keys(report, REPORT_KEYS, "report")
    if report["schema"] != REPORT_SCHEMA:
        raise VerificationError("unsupported terminal report schema")
    _assert_equivalent(REPORT_PROTOCOL, report["protocol"], "protocol")
    _exact_keys(report["contract"], REPORT_CONTRACT_KEYS, "report.contract")
    _assert_equivalent(expected_report_contract, report["contract"], "report.contract")
    _assert_equivalent(PROHIBITIONS, report["prohibitions"], "prohibitions")
    data_contract = run_contract.get("data")
    if not isinstance(data_contract, dict) or "targets" not in data_contract:
        raise VerificationError("run contract lacks the clean train target root")
    target_root = Path(str(data_contract["targets"])).resolve(strict=True)
    if not target_root.is_dir() or (require_e_drive and target_root.drive.upper() != "E:"):
        raise VerificationError("clean target root must be an existing E: directory")

    reported_receipts = _exact_keys(report["receipts"], RECEIPT_KEYS, "receipts")
    always_required = ("capacity", "fit_cache", "cal_cache", "oracle_cal", "fit_palette")
    for key in always_required:
        if reported_receipts[key] is None:
            raise VerificationError(f"pre-oracle evidence requires receipt {key}")
    contract_sha = expected_report_contract["run_contract_sha256"]
    receipt_records: dict[str, Any] = {key: None for key in RECEIPT_KEYS}

    capacity_payload, receipt_records["capacity"] = _load_reported_receipt(
        reported_receipts["capacity"],
        root=root,
        relative_path="receipts/capacity_smoke.json",
        schema=CAPACITY_RECEIPT_SCHEMA,
        contract_sha256=contract_sha,
        require_e_drive=require_e_drive,
    )
    validate_capacity_receipt(capacity_payload)
    fit_cache_payload, receipt_records["fit_cache"] = _load_reported_receipt(
        reported_receipts["fit_cache"],
        root=root,
        relative_path="cache/fit_cache.json",
        schema=CACHE_SCHEMA,
        contract_sha256=contract_sha,
        require_e_drive=require_e_drive,
    )
    cal_cache_payload, receipt_records["cal_cache"] = _load_reported_receipt(
        reported_receipts["cal_cache"],
        root=root,
        relative_path="cache/cal_cache.json",
        schema=CACHE_SCHEMA,
        contract_sha256=contract_sha,
        require_e_drive=require_e_drive,
    )
    fit_cache = validate_cache_receipt(
        fit_cache_payload,
        root=root,
        partition="fit",
        count=FIT_COUNT,
        require_e_drive=require_e_drive,
    )
    cal_cache = validate_cache_receipt(
        cal_cache_payload,
        root=root,
        partition="cal",
        count=PARTITION_COUNT,
        require_e_drive=require_e_drive,
    )

    artifacts = root / "artifacts"
    oracle_path, oracle_record = _verify_record(
        report["oracle_cal_npz"],
        label="oracle_cal_npz",
        require_e_drive=require_e_drive,
        expected_parent=artifacts,
        expected_name="cal_oracle_pretrain.npz",
    )
    fit_path, fit_record = _verify_record(
        report["fit_palette_npz"],
        label="fit_palette_npz",
        require_e_drive=require_e_drive,
        expected_parent=artifacts,
        expected_name="fit_palette_whitening.npz",
    )
    oracle_receipt_payload, receipt_records["oracle_cal"] = _load_reported_receipt(
        reported_receipts["oracle_cal"],
        root=root,
        relative_path="receipts/cal_oracle_pretrain.json",
        schema=ORACLE_RECEIPT_SCHEMA,
        contract_sha256=contract_sha,
        require_e_drive=require_e_drive,
    )
    fit_receipt_payload, receipt_records["fit_palette"] = _load_reported_receipt(
        reported_receipts["fit_palette"],
        root=root,
        relative_path="receipts/fit_palette_whitening.json",
        schema=FIT_PALETTE_RECEIPT_SCHEMA,
        contract_sha256=contract_sha,
        require_e_drive=require_e_drive,
    )
    validate_linked_receipt(
        oracle_receipt_payload,
        nested_key="artifact",
        artifact_record_value=oracle_record,
        label="oracle_cal",
    )
    validate_linked_receipt(
        fit_receipt_payload,
        nested_key="artifact",
        artifact_record_value=fit_record,
        label="fit_palette",
    )
    if "cal_cache_manifest" in oracle_receipt_payload:
        _assert_equivalent(
            receipt_records["cal_cache"],
            oracle_receipt_payload["cal_cache_manifest"],
            "oracle_cal.cal_cache_manifest",
        )
    if fit_receipt_payload.get("algorithm") != (
        "FIT_population_mean_std_ddof0_scale_max_1e-6_float64"
    ):
        raise VerificationError("FIT palette receipt algorithm drift")
    if "fit_cache_manifest" in fit_receipt_payload:
        _assert_equivalent(
            receipt_records["fit_cache"],
            fit_receipt_payload["fit_cache_manifest"],
            "fit_palette.fit_cache_manifest",
        )

    oracle_arrays = load_oracle_npz(oracle_path)
    fit = load_fit_palette_npz(fit_path)
    if not np.array_equal(fit_cache["board_id"], fit["fit_board_id"]):
        raise VerificationError("FIT cache/palette board order mismatch")
    if not np.array_equal(fit_cache["dirty_feature60"], fit["dirty_feature60"]):
        raise VerificationError("FIT whitening features differ from dirty-only FIT cache")
    if not np.array_equal(cal_cache["board_id"], oracle_arrays["board_id"]):
        raise VerificationError("CAL oracle board order differs from dirty-only CAL cache")
    if not np.array_equal(cal_cache["flat_rgb"], oracle_arrays["flat_rgb"]):
        raise VerificationError("CAL oracle flat_rgb differs from dirty-only CAL cache")
    _assert_disjoint(fit["fit_board_id"], oracle_arrays["board_id"], "FIT/CAL board_id")
    if "cal_target_receipt" not in oracle_receipt_payload:
        raise VerificationError("oracle receipt lacks authenticated CAL targets")
    cal_target_info = validate_target_receipt_record(
        oracle_receipt_payload["cal_target_receipt"],
        root=root,
        partition="cal",
        count=PARTITION_COUNT,
        expected_board_ids=oracle_arrays["board_id"],
        target_root=target_root,
        contract_sha256=contract_sha,
        require_e_drive=require_e_drive,
    )
    cal_target_paths = [target_root / name for name in cal_target_info["names"]]
    oracle_official, oracle_prediction_diagnostic = verify_prediction_evidence(
        flat_rgb=oracle_arrays["flat_rgb"],
        target_paths=cal_target_paths,
        stored_scores={key: oracle_arrays[key] for key in ORACLE_SSIM_KEYS},
        oracle_coeff_reference=oracle_arrays["oracle_coeff"],
    )
    oracle_metrics = summarize_oracle(oracle_arrays)
    oracle_gates = evaluate_oracle_gate(oracle_metrics)
    oracle_official_metrics = summarize_oracle(oracle_official)
    oracle_official_gates = evaluate_oracle_gate(oracle_official_metrics)
    if oracle_official_gates["passed"] != oracle_gates["passed"]:
        raise VerificationError("official oracle gate disagrees with stored oracle gate")
    if "metrics" in oracle_receipt_payload:
        _assert_equivalent(oracle_metrics, oracle_receipt_payload["metrics"], "oracle receipt metrics")
    if "gate" in oracle_receipt_payload:
        _assert_equivalent(oracle_gates, oracle_receipt_payload["gate"], "oracle receipt gate")

    expected_metrics: dict[str, Any] = {
        "ORACLE_CAL": oracle_metrics,
        "CAL": None,
        "DEV": None,
    }
    expected_gates: dict[str, Any] = {
        "ORACLE_CAL": oracle_gates,
        "CAL": None,
        "DEV": None,
    }
    official_metrics: dict[str, Any] = {
        "ORACLE_CAL": oracle_official_metrics,
        "CAL": None,
        "DEV": None,
    }
    official_gates: dict[str, Any] = {
        "ORACLE_CAL": oracle_official_gates,
        "CAL": None,
        "DEV": None,
    }
    prediction_diagnostics: dict[str, Any] = {
        "ORACLE_CAL": oracle_prediction_diagnostic,
        "CAL": None,
        "DEV": None,
    }
    swap_cycle_stats: dict[str, Any] = {"CAL": None, "DEV": None}
    artifact_receipts: dict[str, Any] = {
        "oracle_cal_npz": oracle_record,
        "fit_palette_npz": fit_record,
        "swap_cal_npz": None,
        "raw_cal_npz": None,
        "swap_dev_npz": None,
        "raw_dev_npz": None,
        "final_checkpoint": None,
    }

    if not oracle_gates["passed"]:
        for key in (
            "swap_cal_npz",
            "raw_cal_npz",
            "swap_dev_npz",
            "raw_dev_npz",
            "final_checkpoint",
        ):
            if report[key] is not None:
                raise VerificationError(f"oracle reject requires {key}=null")
        for key in ("swap_cal", "final_checkpoint", "raw_cal", "dev_cache", "swap_dev", "raw_dev"):
            if reported_receipts[key] is not None:
                raise VerificationError(f"oracle reject requires receipt {key}=null")
        status, stage, decision = "oracle_reject", ORACLE_STAGE, "KILL_DCT_WHERE"
        _assert_oracle_reject_sealed(root)
    else:
        for key in ("swap_cal_npz", "raw_cal_npz", "final_checkpoint"):
            if report[key] is None:
                raise VerificationError(f"oracle pass requires {key}")
        for key in ("swap_cal", "final_checkpoint", "raw_cal"):
            if reported_receipts[key] is None:
                raise VerificationError(f"oracle pass requires receipt {key}")
        swap_cal_path, swap_cal_record = _verify_record(
            report["swap_cal_npz"],
            label="swap_cal_npz",
            require_e_drive=require_e_drive,
            expected_parent=artifacts,
            expected_name="swap_cal.npz",
        )
        raw_cal_path, raw_cal_record = _verify_record(
            report["raw_cal_npz"],
            label="raw_cal_npz",
            require_e_drive=require_e_drive,
            expected_parent=artifacts,
            expected_name="m144_cal_raw.npz",
        )
        checkpoint_path, checkpoint_record = _verify_record(
            report["final_checkpoint"],
            label="final_checkpoint",
            require_e_drive=require_e_drive,
            expected_parent=root / "checkpoints",
            expected_name="step_0002500.pt",
            max_bytes=MAX_CHECKPOINT_BYTES,
        )
        if not _inside(checkpoint_path, root):
            raise VerificationError("final checkpoint escapes work root")
        swap_cal_receipt, receipt_records["swap_cal"] = _load_reported_receipt(
            reported_receipts["swap_cal"],
            root=root,
            relative_path="receipts/swap_cal.json",
            schema=SWAP_RECEIPT_SCHEMA,
            contract_sha256=contract_sha,
            require_e_drive=require_e_drive,
        )
        checkpoint_receipt, receipt_records["final_checkpoint"] = _load_reported_receipt(
            reported_receipts["final_checkpoint"],
            root=root,
            relative_path="receipts/checkpoint_step_0002500.json",
            schema=CHECKPOINT_RECEIPT_SCHEMA,
            contract_sha256=contract_sha,
            require_e_drive=require_e_drive,
        )
        raw_cal_receipt, receipt_records["raw_cal"] = _load_reported_receipt(
            reported_receipts["raw_cal"],
            root=root,
            relative_path="receipts/cal_raw.json",
            schema=RAW_RECEIPT_SCHEMA,
            contract_sha256=contract_sha,
            require_e_drive=require_e_drive,
        )
        validate_linked_receipt(
            swap_cal_receipt, nested_key="artifact",
            artifact_record_value=swap_cal_record, label="swap_cal",
        )
        validate_linked_receipt(
            checkpoint_receipt, nested_key="checkpoint",
            artifact_record_value=checkpoint_record, label="final_checkpoint",
        )
        validate_linked_receipt(
            raw_cal_receipt, nested_key="artifact",
            artifact_record_value=raw_cal_record, label="raw_cal",
        )
        if raw_cal_receipt.get("partition") != "cal":
            raise VerificationError("CAL raw receipt partition mismatch")
        cal_dependencies = raw_cal_receipt.get("dependencies")
        _exact_keys(
            cal_dependencies,
            {"swap", "checkpoint", "cache_manifest", "target_receipt", "oracle"},
            "raw_cal.dependencies",
        )
        for dependency_name, expected_dependency in (
            ("swap", swap_cal_record),
            ("checkpoint", checkpoint_record),
            ("cache_manifest", receipt_records["cal_cache"]),
            ("target_receipt", cal_target_info["record"]),
            ("oracle", oracle_record),
        ):
            _assert_equivalent(
                expected_dependency,
                cal_dependencies[dependency_name],
                f"raw_cal.dependencies.{dependency_name}",
            )
        if checkpoint_receipt.get("step") != 2_500:
            raise VerificationError("final checkpoint receipt is not step 2500")
        if checkpoint_receipt.get("loss_definition") != "1-mean_uniform_ssim_float_proxy":
            raise VerificationError("checkpoint training-loss definition drift")
        if swap_cal_receipt.get("partition") != "cal":
            raise VerificationError("CAL swap receipt partition mismatch")
        if swap_cal_receipt.get("scipy_version") != str(scipy.__version__):
            raise VerificationError("CAL swap receipt SciPy version drift")
        if swap_cal_receipt.get("algorithm") != (
            "LSA_squared_euclidean_sum_float64_no_jitter_forbid_self_source_v1"
        ):
            raise VerificationError("CAL swap receipt algorithm drift")
        if "fit_palette" in swap_cal_receipt:
            _assert_equivalent(fit_record, swap_cal_receipt["fit_palette"], "swap_cal.fit_palette")
        if "partition_cache_manifest" in swap_cal_receipt:
            _assert_equivalent(
                receipt_records["cal_cache"],
                swap_cal_receipt["partition_cache_manifest"],
                "swap_cal.partition_cache_manifest",
            )
        fit_target_record = path_record(root / "receipts" / "fit_targets.json")
        validate_target_receipt_record(
            fit_target_record,
            root=root,
            partition="fit",
            count=FIT_COUNT,
            expected_board_ids=fit["fit_board_id"],
            target_root=target_root,
            contract_sha256=contract_sha,
            require_e_drive=require_e_drive,
        )
        swap_cal = load_swap_npz(swap_cal_path)
        raw_cal = load_raw_npz(raw_cal_path)
        cal_cycle_stats = verify_swap_semantics(swap_cal, fit)
        swap_cycle_stats["CAL"] = cal_cycle_stats
        _assert_equivalent(
            cal_cycle_stats, swap_cal_receipt.get("cycle_stats"), "swap_cal.cycle_stats"
        )
        _assert_partition_alignment(raw_cal, swap_cal, "CAL")
        if not np.array_equal(cal_cache["dirty_feature60"], swap_cal["dirty_feature60"]):
            raise VerificationError("CAL swap features differ from dirty-only CAL cache")
        for key in ORACLE_ID_KEYS + ORACLE_SSIM_KEYS + ("flat_rgb",):
            if not np.array_equal(oracle_arrays[key], raw_cal[key]):
                raise VerificationError(f"pretrain oracle and learned CAL differ at {key}")

        cal_official_scores, cal_prediction_diagnostic = verify_prediction_evidence(
            flat_rgb=raw_cal["flat_rgb"],
            target_paths=cal_target_paths,
            stored_scores={key: raw_cal[key] for key in RAW_SSIM_KEYS},
            dct_predictions={
                "dct_full_ssim": raw_cal["dct_full_coeff"],
                "dct_blind_ssim": raw_cal["dct_blind_coeff"],
                "dct_swapped_ssim": raw_cal["dct_swapped_coeff"],
            },
            rgb_predictions={
                "rgb8_full_ssim": raw_cal["rgb8_full_residual"],
                "rgb8_blind_ssim": raw_cal["rgb8_blind_residual"],
            },
            oracle_coeff_reference=oracle_arrays["oracle_coeff"],
        )

        cal_metrics = summarize_arrays(raw_cal, confidence=CAL_CONFIDENCE)
        cal_gates = evaluate_cal_gates(cal_metrics)
        cal_official_arrays = dict(raw_cal)
        cal_official_arrays.update(cal_official_scores)
        cal_official_metrics = summarize_arrays(
            cal_official_arrays, confidence=CAL_CONFIDENCE
        )
        cal_official_gates = evaluate_cal_gates(cal_official_metrics)
        if cal_official_gates["passed"] != cal_gates["passed"]:
            raise VerificationError("official CAL gate disagrees with stored CAL gate")
        expected_metrics["CAL"] = cal_metrics
        expected_gates["CAL"] = cal_gates
        official_metrics["CAL"] = cal_official_metrics
        official_gates["CAL"] = cal_official_gates
        prediction_diagnostics["CAL"] = cal_prediction_diagnostic
        artifact_receipts.update(
            fit_palette_npz=fit_record,
            swap_cal_npz=swap_cal_record,
            raw_cal_npz=raw_cal_record,
            final_checkpoint=checkpoint_record,
        )

        if not cal_gates["passed"]:
            if report["swap_dev_npz"] is not None or report["raw_dev_npz"] is not None:
                raise VerificationError("CAL reject requires all DEV records null")
            for key in ("dev_cache", "swap_dev", "raw_dev"):
                if reported_receipts[key] is not None:
                    raise VerificationError(f"CAL reject requires receipt {key}=null")
            status, stage, decision = _terminal(False, None)
            _assert_dev_sealed(root)
        else:
            if report["swap_dev_npz"] is None or report["raw_dev_npz"] is None:
                raise VerificationError("CAL pass requires authenticated DEV swap/raw artifacts")
            for key in ("dev_cache", "swap_dev", "raw_dev"):
                if reported_receipts[key] is None:
                    raise VerificationError(f"CAL pass requires receipt {key}")
            dev_cache_payload, receipt_records["dev_cache"] = _load_reported_receipt(
                reported_receipts["dev_cache"],
                root=root,
                relative_path="cache/dev_cache.json",
                schema=CACHE_SCHEMA,
                contract_sha256=contract_sha,
                require_e_drive=require_e_drive,
            )
            dev_cache = validate_cache_receipt(
                dev_cache_payload,
                root=root,
                partition="dev",
                count=PARTITION_COUNT,
                require_e_drive=require_e_drive,
            )
            swap_dev_path, swap_dev_record = _verify_record(
                report["swap_dev_npz"],
                label="swap_dev_npz",
                require_e_drive=require_e_drive,
                expected_parent=artifacts,
                expected_name="swap_dev.npz",
            )
            raw_dev_path, raw_dev_record = _verify_record(
                report["raw_dev_npz"],
                label="raw_dev_npz",
                require_e_drive=require_e_drive,
                expected_parent=artifacts,
                expected_name="m144_dev_raw.npz",
            )
            swap_dev_receipt, receipt_records["swap_dev"] = _load_reported_receipt(
                reported_receipts["swap_dev"],
                root=root,
                relative_path="receipts/swap_dev.json",
                schema=SWAP_RECEIPT_SCHEMA,
                contract_sha256=contract_sha,
                require_e_drive=require_e_drive,
            )
            raw_dev_receipt, receipt_records["raw_dev"] = _load_reported_receipt(
                reported_receipts["raw_dev"],
                root=root,
                relative_path="receipts/dev_raw.json",
                schema=RAW_RECEIPT_SCHEMA,
                contract_sha256=contract_sha,
                require_e_drive=require_e_drive,
            )
            validate_linked_receipt(
                swap_dev_receipt, nested_key="artifact",
                artifact_record_value=swap_dev_record, label="swap_dev",
            )
            validate_linked_receipt(
                raw_dev_receipt, nested_key="artifact",
                artifact_record_value=raw_dev_record, label="raw_dev",
            )
            if swap_dev_receipt.get("partition") != "dev":
                raise VerificationError("DEV swap receipt partition mismatch")
            if swap_dev_receipt.get("scipy_version") != str(scipy.__version__) or (
                swap_dev_receipt.get("algorithm")
                != "LSA_squared_euclidean_sum_float64_no_jitter_forbid_self_source_v1"
            ):
                raise VerificationError("DEV swap receipt algorithm/version drift")
            if "fit_palette" in swap_dev_receipt:
                _assert_equivalent(fit_record, swap_dev_receipt["fit_palette"], "swap_dev.fit_palette")
            if "partition_cache_manifest" in swap_dev_receipt:
                _assert_equivalent(
                    receipt_records["dev_cache"],
                    swap_dev_receipt["partition_cache_manifest"],
                    "swap_dev.partition_cache_manifest",
                )
            swap_dev = load_swap_npz(swap_dev_path)
            raw_dev = load_raw_npz(raw_dev_path)
            dev_dependencies = raw_dev_receipt.get("dependencies")
            _exact_keys(
                dev_dependencies,
                {"swap", "checkpoint", "cache_manifest", "target_receipt", "fit_palette"},
                "raw_dev.dependencies",
            )
            dev_target_info = validate_target_receipt_record(
                dev_dependencies["target_receipt"],
                root=root,
                partition="dev",
                count=PARTITION_COUNT,
                expected_board_ids=raw_dev["board_id"],
                target_root=target_root,
                contract_sha256=contract_sha,
                require_e_drive=require_e_drive,
            )
            for dependency_name, expected_dependency in (
                ("swap", swap_dev_record),
                ("checkpoint", checkpoint_record),
                ("cache_manifest", receipt_records["dev_cache"]),
                ("target_receipt", dev_target_info["record"]),
                ("fit_palette", fit_record),
            ):
                _assert_equivalent(
                    expected_dependency,
                    dev_dependencies[dependency_name],
                    f"raw_dev.dependencies.{dependency_name}",
                )
            if raw_dev_receipt.get("partition") != "dev":
                raise VerificationError("DEV raw receipt partition mismatch")
            dev_cycle_stats = verify_swap_semantics(swap_dev, fit)
            swap_cycle_stats["DEV"] = dev_cycle_stats
            _assert_equivalent(
                dev_cycle_stats, swap_dev_receipt.get("cycle_stats"), "swap_dev.cycle_stats"
            )
            _assert_partition_alignment(raw_dev, swap_dev, "DEV")
            if not np.array_equal(dev_cache["board_id"], swap_dev["board_id"]):
                raise VerificationError("DEV cache/swap board order mismatch")
            if not np.array_equal(dev_cache["dirty_feature60"], swap_dev["dirty_feature60"]):
                raise VerificationError("DEV swap features differ from dirty-only DEV cache")
            if not np.array_equal(dev_cache["flat_rgb"], raw_dev["flat_rgb"]):
                raise VerificationError("DEV raw flat_rgb differs from dirty-only DEV cache")
            _assert_disjoint(fit["fit_board_id"], raw_dev["board_id"], "FIT/DEV board_id")
            _assert_disjoint(raw_cal["board_id"], raw_dev["board_id"], "CAL/DEV board_id")
            _assert_disjoint(
                raw_cal["source_group_id"], raw_dev["source_group_id"],
                "CAL/DEV source_group_id",
            )
            dev_target_paths = [target_root / name for name in dev_target_info["names"]]
            dev_official_scores, dev_prediction_diagnostic = verify_prediction_evidence(
                flat_rgb=raw_dev["flat_rgb"],
                target_paths=dev_target_paths,
                stored_scores={key: raw_dev[key] for key in RAW_SSIM_KEYS},
                dct_predictions={
                    "dct_full_ssim": raw_dev["dct_full_coeff"],
                    "dct_blind_ssim": raw_dev["dct_blind_coeff"],
                    "dct_swapped_ssim": raw_dev["dct_swapped_coeff"],
                },
                rgb_predictions={
                    "rgb8_full_ssim": raw_dev["rgb8_full_residual"],
                    "rgb8_blind_ssim": raw_dev["rgb8_blind_residual"],
                },
            )
            dev_metrics = summarize_arrays(raw_dev, confidence=DEV_CONFIDENCE)
            dev_gates = evaluate_dev_gates(dev_metrics)
            dev_official_arrays = dict(raw_dev)
            dev_official_arrays.update(dev_official_scores)
            dev_official_metrics = summarize_arrays(
                dev_official_arrays, confidence=DEV_CONFIDENCE
            )
            dev_official_gates = evaluate_dev_gates(dev_official_metrics)
            if dev_official_gates["passed"] != dev_gates["passed"]:
                raise VerificationError("official DEV gate disagrees with stored DEV gate")
            expected_metrics["DEV"] = dev_metrics
            expected_gates["DEV"] = dev_gates
            official_metrics["DEV"] = dev_official_metrics
            official_gates["DEV"] = dev_official_gates
            prediction_diagnostics["DEV"] = dev_prediction_diagnostic
            artifact_receipts.update(
                swap_dev_npz=swap_dev_record, raw_dev_npz=raw_dev_record
            )
            status, stage, decision = _terminal(True, bool(dev_gates["passed"]))

    if report["status"] != status or report["stage"] != stage or report["decision"] != decision:
        raise VerificationError(
            f"terminal route mismatch; expected {status}/{stage}/{decision}"
        )
    _assert_equivalent(expected_metrics, report["metrics"], "metrics")
    _assert_equivalent(expected_gates, report["gates"], "gates")
    _assert_equivalent(receipt_records, report["receipts"], "receipts")

    verifier_record = path_record(Path(__file__))
    receipt = {
        "schema": VERIFICATION_SCHEMA,
        "valid": True,
        "status": status,
        "stage": stage,
        "decision": decision,
        "report": {
            "path": str(report_file),
            "bytes": len(report_bytes),
            "sha256": sha256_bytes(report_bytes),
        },
        "contract": {
            "path": str(contract_file),
            "bytes": len(contract_bytes),
            "sha256": sha256_bytes(contract_bytes),
            "contract_sha256": expected_report_contract["run_contract_sha256"],
        },
        "verifier": verifier_record,
        "source_files": expected_report_contract["source_files"],
        "artifacts": artifact_receipts,
        "receipts": receipt_records,
        "gates": expected_gates,
        "official_metrics": official_metrics,
        "official_gates": official_gates,
        "prediction_diagnostics": prediction_diagnostics,
        "swap_cycle_stats": swap_cycle_stats,
    }
    canonical_json_bytes(receipt)
    return receipt


def write_create_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            current = path.read_bytes()
        except OSError as error:
            raise VerificationError(f"cannot read prior verification: {error}") from error
        if current != payload:
            raise VerificationError("prior verification exists and is not byte-exact")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.work_root.resolve()
        output = args.output.resolve()
        if output != root / "artifacts" / "m144_verification.json":
            raise VerificationError(
                "output must be work_root/artifacts/m144_verification.json"
            )
        _assert_e(output, "verification output")
        receipt = verify_report(
            work_root=root,
            contract_path=args.contract,
            report_path=args.report,
            require_e_drive=True,
        )
        write_create_once(output, receipt)
    except (VerificationError, OSError) as error:
        print(f"M144 verification invalid: {error}", file=sys.stderr)
        return 1
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
