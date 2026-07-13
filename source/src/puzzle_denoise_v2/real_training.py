"""Conservative real-pair fine-tuning with synthetic anchoring and rollback gates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from .degradation import SyntheticTileDegrader
from .legacy_baseline import load_legacy_tile_restorer, predict_legacy_tiles_uint8
from .losses import LossWeights, RestorationLoss
from .real_pairs import RealPairBatch, RealPairSampler, RealPairTable
from .real_validation import evaluate_real_pairs, paired_source_bootstrap_delta
from .training import (
    CleanTileStore,
    atomic_torch_save,
    build_model,
    choose_device,
    evaluate_model,
    load_manifest,
    make_fixed_validation_plan,
    random_dihedral,
    render_fixed_validation,
    resolved_device_fingerprint,
    runtime_versions,
    seed_everything,
    source_code_fingerprint,
    update_ema,
)


@dataclass(frozen=True)
class FineTuneConfig:
    data_root: str
    manifest: str
    train_pairs: str
    val_pairs: str
    init_checkpoint: str
    legacy_checkpoint: str
    quarantine_artifact: str
    output: str
    expected_manifest_sha256: str
    expected_train_pairs_sha256: str
    expected_val_pairs_sha256: str
    expected_init_checkpoint_sha256: str
    expected_legacy_checkpoint_sha256: str
    expected_quarantine_sha256: str
    expected_training_pixels_sha256: str
    expected_validation_pixels_sha256: str
    expected_opencv_version: str
    gate_source_count: int
    model: str = "tile-naf"
    steps: int = 4000
    batch_size: int = 256
    pairs_per_real_source: int = 32
    synthetic_train_images: int = 512
    train_min_confidence: float = 1.0
    val_sensitivity_confidence: float = 1.0
    val_primary_confidence: float = 1.5
    val_pairs_per_source: int = 8
    peak_learning_rate: float = 1e-5
    encoder_lr_scale: float = 0.5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    early_real_period: int = 8
    late_real_period: int = 4
    schedule_switch_step: int = 500
    eval_interval: int = 500
    log_interval: int = 100
    bootstrap_resamples: int = 5000
    no_gain_patience: int = 3
    no_gain_min_delta: float = 1e-4
    max_seconds: float | None = None
    seed: int = 20260710
    device: str = "auto"


def validate_fine_tune_config(config: FineTuneConfig) -> None:
    integer_positive = (
        "steps",
        "batch_size",
        "pairs_per_real_source",
        "synthetic_train_images",
        "val_pairs_per_source",
        "warmup_steps",
        "early_real_period",
        "late_real_period",
        "schedule_switch_step",
        "eval_interval",
        "log_interval",
        "bootstrap_resamples",
        "no_gain_patience",
        "gate_source_count",
    )
    for field in integer_positive:
        if getattr(config, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if config.warmup_steps > config.steps:
        raise ValueError("warmup_steps cannot exceed steps")
    if not 0 < config.encoder_lr_scale <= 1:
        raise ValueError("encoder_lr_scale must be in (0, 1]")
    if not 0 < config.min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in (0, 1]")
    if config.peak_learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if not 0 <= config.ema_decay < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if not 0 <= config.train_min_confidence <= config.val_primary_confidence:
        raise ValueError("confidence thresholds are inconsistent")
    if not config.train_min_confidence <= config.val_sensitivity_confidence <= config.val_primary_confidence:
        raise ValueError("sensitivity confidence must lie between train and primary thresholds")
    if config.gate_source_count != 350:
        raise ValueError("gate_source_count must equal the pinned frozen-gate size 350")
    if not math.isfinite(config.no_gain_min_delta) or config.no_gain_min_delta < 0:
        raise ValueError("no_gain_min_delta must be finite and non-negative")
    if config.max_seconds is not None and (
        not math.isfinite(config.max_seconds) or config.max_seconds <= 0
    ):
        raise ValueError("max_seconds must be finite and positive when provided")
    for field in (
        "expected_manifest_sha256",
        "expected_train_pairs_sha256",
        "expected_val_pairs_sha256",
        "expected_init_checkpoint_sha256",
        "expected_legacy_checkpoint_sha256",
        "expected_quarantine_sha256",
        "expected_training_pixels_sha256",
        "expected_validation_pixels_sha256",
    ):
        value = getattr(config, field)
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{field} must be a lowercase SHA256 digest")
    if re.fullmatch(r"\d+(?:\.\d+){1,3}", config.expected_opencv_version) is None:
        raise ValueError("expected_opencv_version must be a dotted numeric version")


def deterministic_source_split(
    source_names: tuple[str, ...],
    eligible_source_indices: np.ndarray,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split validation sources reproducibly without depending on array order."""
    indices = np.asarray(eligible_source_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 2 or len(np.unique(indices)) != len(indices):
        raise ValueError("eligible_source_indices must contain at least two unique indices")
    if indices.min() < 0 or indices.max() >= len(source_names):
        raise ValueError("eligible_source_indices contains an out-of-range source")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")

    ranked = sorted(
        indices.tolist(),
        key=lambda index: hashlib.sha256(
            f"{seed}:{source_names[index]}".encode("utf-8")
        ).digest()
        + source_names[index].encode("utf-8"),
    )
    cut = min(max(int(round(len(ranked) * calibration_fraction)), 1), len(ranked) - 1)
    calibration = np.asarray(sorted(ranked[:cut]), dtype=np.int64)
    gate = np.asarray(sorted(ranked[cut:]), dtype=np.int64)
    return calibration, gate


def deterministic_contamination_aware_split(
    source_names: tuple[str, ...],
    eligible_source_indices: np.ndarray,
    quarantine_names: tuple[str, ...],
    gate_source_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exclude quarantined names, then deterministically reserve an exact gate."""
    if len(source_names) != len(set(source_names)):
        raise ValueError("source_names must be unique")
    indices = np.asarray(eligible_source_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 2 or len(np.unique(indices)) != len(indices):
        raise ValueError("eligible_source_indices must contain at least two unique indices")
    if indices.min() < 0 or indices.max() >= len(source_names):
        raise ValueError("eligible_source_indices contains an out-of-range source")
    quarantine = tuple(quarantine_names)
    if not quarantine or tuple(sorted(quarantine)) != quarantine or len(set(quarantine)) != len(quarantine):
        raise ValueError("quarantine_names must be a non-empty sorted unique tuple")
    missing_quarantine = sorted(set(quarantine) - set(source_names))
    if missing_quarantine:
        raise ValueError(f"quarantine names are absent from source_names: {missing_quarantine[:5]}")
    if isinstance(gate_source_count, bool) or not isinstance(
        gate_source_count, (int, np.integer)
    ):
        raise TypeError("gate_source_count must be an integer")

    quarantine_set = set(quarantine)
    clean_indices = [
        int(index) for index in indices if source_names[int(index)] not in quarantine_set
    ]
    if not 0 < gate_source_count < len(clean_indices):
        raise ValueError("gate_source_count must leave a non-empty calibration set")
    ranked = sorted(
        clean_indices,
        key=lambda index: hashlib.sha256(
            f"{seed}:{source_names[index]}".encode("utf-8")
        ).digest()
        + source_names[index].encode("utf-8"),
    )
    calibration_count = len(ranked) - int(gate_source_count)
    calibration = np.asarray(sorted(ranked[:calibration_count]), dtype=np.int64)
    gate = np.asarray(sorted(ranked[calibration_count:]), dtype=np.int64)
    return calibration, gate


def source_name_list_sha256(names: list[str] | tuple[str, ...]) -> str:
    payload = json.dumps(list(names), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_validation_quarantine(
    path: str | Path,
    expected_sha256: str,
    *,
    manifest_sha256: str,
    manifest_validation_names: list[str],
    expected_legacy_checkpoint_sha256: str,
    expected_synthetic_validation_names: list[str],
    gate_source_count: int,
    seed: int,
) -> tuple[dict, str]:
    """Load and strictly validate the pinned legacy/synthetic contamination set."""
    artifact_path = Path(path)
    payload_bytes = artifact_path.read_bytes()
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("validation-quarantine SHA256 does not match the pinned digest")
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("validation-quarantine artifact is not valid JSON") from error
    if type(payload) is not dict:
        raise ValueError("validation-quarantine artifact must contain a plain object")
    required_top_level = {
        "schema_version",
        "kind",
        "manifest",
        "maps_artifact",
        "legacy_checkpoint",
        "counts",
        "legacy_train_seen_names",
        "legacy_validation_seen_names",
        "name_sha256",
        "quarantine_names",
        "synthetic_validation_names",
        "policy",
    }
    if set(payload) != required_top_level:
        raise ValueError("validation-quarantine artifact has an unexpected top-level schema")
    if payload["schema_version"] != 1 or payload["kind"] != "denoise_validation_quarantine":
        raise ValueError("unsupported validation-quarantine schema or kind")

    manifest_record = payload["manifest"]
    if type(manifest_record) is not dict or set(manifest_record) != {
        "path",
        "sha256",
        "validation_source_count",
    }:
        raise ValueError("validation-quarantine manifest record is malformed")
    if manifest_record["sha256"] != manifest_sha256:
        raise ValueError("validation-quarantine manifest SHA256 mismatch")
    if manifest_record["validation_source_count"] != len(manifest_validation_names):
        raise ValueError("validation-quarantine manifest source count mismatch")

    maps_record = payload["maps_artifact"]
    expected_maps_meta_keys = {
        "count",
        "data_root",
        "grid",
        "mean_cost",
        "mean_margin",
        "median_cost",
        "seconds",
        "tile",
    }
    if type(maps_record) is not dict or set(maps_record) != {
        "path",
        "sha256",
        "names_count",
        "metadata",
    }:
        raise ValueError("validation-quarantine maps record is malformed")
    if re.fullmatch(r"[0-9a-f]{64}", maps_record["sha256"]) is None:
        raise ValueError("validation-quarantine maps SHA256 is malformed")
    maps_metadata = maps_record["metadata"]
    if type(maps_metadata) is not dict or set(maps_metadata) != expected_maps_meta_keys:
        raise ValueError("validation-quarantine maps metadata is malformed")
    if (
        maps_record["names_count"] != 1024
        or maps_metadata["count"] != 1024
        or maps_metadata["grid"] != 24
        or maps_metadata["tile"] != 20
    ):
        raise ValueError("validation-quarantine maps geometry/count mismatch")
    for key in ("mean_cost", "mean_margin", "median_cost", "seconds"):
        if not isinstance(maps_metadata[key], (int, float)) or not math.isfinite(
            maps_metadata[key]
        ):
            raise ValueError(f"validation-quarantine maps metadata {key} is not finite")

    legacy_record = payload["legacy_checkpoint"]
    if type(legacy_record) is not dict or set(legacy_record) != {
        "path",
        "sha256",
        "architecture",
        "args_summary",
    }:
        raise ValueError("validation-quarantine legacy checkpoint record is malformed")
    if legacy_record["sha256"] != expected_legacy_checkpoint_sha256:
        raise ValueError("validation-quarantine legacy checkpoint SHA256 mismatch")
    if legacy_record["architecture"] != {"width": 64, "depth": 8, "grid": 24, "tile": 20}:
        raise ValueError("validation-quarantine legacy architecture mismatch")
    args_summary = legacy_record["args_summary"]
    required_args = {
        "command",
        "data_root",
        "maps",
        "train_images",
        "val_images",
        "epochs",
        "seed",
        "width",
        "depth",
        "cost_quantile",
        "max_pairs",
        "shuffle_images",
    }
    if type(args_summary) is not dict or set(args_summary) != required_args:
        raise ValueError("validation-quarantine legacy args summary is malformed")
    if (
        args_summary["command"] != "train"
        or args_summary["maps"] != maps_record["path"]
        or args_summary["train_images"] != 960
        or args_summary["val_images"] != 64
        or args_summary["shuffle_images"] is not False
        or args_summary["width"] != 64
        or args_summary["depth"] != 8
    ):
        raise ValueError("validation-quarantine legacy partition/model mismatch")

    def validated_names(key: str, expected_count: int) -> tuple[str, ...]:
        values = payload[key]
        if (
            type(values) is not list
            or any(type(value) is not str for value in values)
            or values != sorted(values)
            or len(values) != len(set(values))
            or len(values) != expected_count
        ):
            raise ValueError(f"validation-quarantine {key} must be sorted and unique")
        return tuple(values)

    counts = payload["counts"]
    expected_counts = {
        "quarantine": 93,
        "legacy_train_seen": 87,
        "legacy_validation_seen": 6,
        "synthetic_validation_seen": 24,
        "eligible_after_quarantine": 607,
        "calibration": 607 - gate_source_count,
        "frozen_gate": gate_source_count,
    }
    if type(counts) is not dict or counts != expected_counts:
        raise ValueError("validation-quarantine counts do not match the pinned protocol")
    legacy_train_names = validated_names("legacy_train_seen_names", 87)
    legacy_validation_names = validated_names("legacy_validation_seen_names", 6)
    quarantine_names = validated_names("quarantine_names", 93)
    synthetic_validation_names = validated_names("synthetic_validation_names", 24)
    if set(legacy_train_names) & set(legacy_validation_names):
        raise ValueError("legacy train/validation quarantine sets overlap")
    if set(quarantine_names) != set(legacy_train_names) | set(legacy_validation_names):
        raise ValueError("quarantine names are not the legacy-seen union")
    manifest_validation_set = set(manifest_validation_names)
    if not set(quarantine_names) <= manifest_validation_set:
        raise ValueError("quarantine contains names outside current manifest validation")
    if list(synthetic_validation_names) != expected_synthetic_validation_names:
        raise ValueError("synthetic validation names do not match the initial checkpoint protocol")
    if not set(synthetic_validation_names) <= set(quarantine_names):
        raise ValueError("synthetic validation names are not fully quarantined")

    manifest_names_tuple = tuple(manifest_validation_names)
    expected_calibration, expected_gate = deterministic_contamination_aware_split(
        manifest_names_tuple,
        np.arange(len(manifest_names_tuple), dtype=np.int64),
        quarantine_names,
        gate_source_count,
        seed,
    )
    expected_name_hashes = {
        "legacy_train_seen": source_name_list_sha256(legacy_train_names),
        "legacy_validation_seen": source_name_list_sha256(legacy_validation_names),
        "quarantine": source_name_list_sha256(quarantine_names),
        "synthetic_validation": source_name_list_sha256(synthetic_validation_names),
        "eligible_after_quarantine": source_name_list_sha256(
            sorted(manifest_validation_set - set(quarantine_names))
        ),
        "calibration": source_name_list_sha256(
            sorted(manifest_names_tuple[int(index)] for index in expected_calibration)
        ),
        "frozen_gate": source_name_list_sha256(
            sorted(manifest_names_tuple[int(index)] for index in expected_gate)
        ),
    }
    if payload["name_sha256"] != expected_name_hashes:
        raise ValueError("validation-quarantine source-name hashes do not match its contents")

    policy = payload["policy"]
    if type(policy) is not dict or type(policy.get("deterministic_split")) is not dict:
        raise ValueError("validation-quarantine policy is malformed")
    deterministic_policy = policy["deterministic_split"]
    if (
        deterministic_policy.get("seed") != seed
        or deterministic_policy.get("ranking")
        != "ascending SHA256 of seed, colon, and source name; source name is the collision tie-breaker"
        or deterministic_policy.get("calibration")
        != f"first {607 - gate_source_count} ranked eligible sources"
        or deterministic_policy.get("frozen_gate")
        != f"remaining {gate_source_count} ranked eligible sources"
    ):
        raise ValueError("validation-quarantine deterministic split policy mismatch")
    required_disjoint_sets = {
        "quarantine_names",
        "synthetic_validation_names",
        "legacy_train_seen_names",
        "legacy_validation_seen_names",
    }
    if set(policy.get("frozen_gate_must_be_disjoint_from", [])) != required_disjoint_sets:
        raise ValueError("validation-quarantine frozen-gate policy is incomplete")
    return payload, actual_sha256


def fine_tune_code_fingerprint() -> str:
    package_dir = Path(__file__).resolve().parent
    names = (
        "__init__.py",
        "degradation.py",
        "legacy_baseline.py",
        "losses.py",
        "metrics.py",
        "model.py",
        "real_pairs.py",
        "real_training.py",
        "real_validation.py",
        "tiles.py",
        "training.py",
    )
    digest = hashlib.sha256()
    for name in names:
        path = package_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fine_tune_pixel_fingerprints(
    data_root: str | Path,
    synthetic_target_names: list[str] | tuple[str, ...],
    train_source_names: tuple[str, ...],
    validation_source_names: tuple[str, ...],
) -> dict[str, str]:
    """Hash every decoded RGB image that can affect fine-tuning or promotion."""
    root = Path(data_root)
    train_dir = root / "train"

    def update_image(digest, label: str, path: Path) -> None:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (480, 480, 3):
            raise ValueError(f"expected RGB 480x480 image at {path}, got {rgb.shape}")
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(rgb.tobytes())

    training_digest = hashlib.sha256()
    for name in sorted(synthetic_target_names):
        update_image(training_digest, "synthetic_target", train_dir / "targets" / name)
    for name in sorted(train_source_names):
        update_image(training_digest, "real_input", train_dir / "inputs" / name)
        update_image(training_digest, "real_target", train_dir / "targets" / name)

    validation_digest = hashlib.sha256()
    for name in sorted(validation_source_names):
        update_image(validation_digest, "real_input", train_dir / "inputs" / name)
        update_image(validation_digest, "real_target", train_dir / "targets" / name)
    return {
        "training_pixels_sha256": training_digest.hexdigest(),
        "validation_pixels_sha256": validation_digest.hexdigest(),
    }


def is_real_batch_step(step: int, config: FineTuneConfig) -> bool:
    period = config.early_real_period if step <= config.schedule_switch_step else config.late_real_period
    return step % period == 0


def learning_rate_scale(step: int, config: FineTuneConfig) -> float:
    if not 1 <= step <= config.steps:
        raise ValueError(f"step {step} outside [1, {config.steps}]")
    if step <= config.warmup_steps:
        return step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine


def paired_dihedral(
    corrupt: torch.Tensor,
    clean: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if corrupt.shape != clean.shape or corrupt.ndim != 4:
        raise ValueError("corrupt and clean must be matching BCHW tensors")
    transforms = rng.integers(0, 8, size=len(corrupt))
    corrupt_output = torch.empty_like(corrupt)
    clean_output = torch.empty_like(clean)
    for transform in range(8):
        positions = np.flatnonzero(transforms == transform)
        if len(positions) == 0:
            continue
        indices = torch.as_tensor(positions, device=corrupt.device)
        corrupt_part = torch.rot90(corrupt[indices], transform % 4, dims=(-2, -1))
        clean_part = torch.rot90(clean[indices], transform % 4, dims=(-2, -1))
        if transform >= 4:
            corrupt_part = torch.flip(corrupt_part, dims=(-1,))
            clean_part = torch.flip(clean_part, dims=(-1,))
        corrupt_output[indices] = corrupt_part
        clean_output[indices] = clean_part
    return corrupt_output, clean_output


def _uint8_tiles(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach()
        .cpu()
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .byte()
        .permute(0, 2, 3, 1)
        .numpy()
    )


def _classical_tiles_uint8(tiles: np.ndarray) -> np.ndarray:
    """Run the fixed pre-v2 classical NLM baseline on independent RGB tiles."""
    array = np.asarray(tiles)
    if array.ndim != 4 or array.shape[1:] != (20, 20, 3) or array.dtype != np.uint8:
        raise ValueError("classical baseline expects uint8 Nx20x20x3 tiles")
    restored = np.empty_like(array)
    for index, tile in enumerate(array):
        bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 7.0, 7.0, 5, 11)
        restored[index] = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    return restored


@torch.no_grad()
def _predict_uint8(
    model: torch.nn.Module,
    tiles: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions = []
    for start in range(0, len(tiles), batch_size):
        prediction = model(tiles[start : start + batch_size].to(device))
        predictions.append(_uint8_tiles(prediction))
    return np.concatenate(predictions)


def _real_panel_metrics(
    prediction: np.ndarray,
    panel: RealPairBatch,
    source_count: int,
) -> dict:
    evaluation = evaluate_real_pairs(
        prediction,
        _uint8_tiles(panel.clean),
        panel.source_index.numpy(),
        source_count=source_count,
    )
    return {
        "pairs": evaluation.pair_count,
        "sources": evaluation.source_count,
        "micro": evaluation.micro_metrics,
        "macro": evaluation.macro_metrics,
    }


def assess_candidate(
    baseline_synthetic: dict[str, float],
    candidate_synthetic: dict[str, float],
    baseline_primary: dict[str, float],
    candidate_primary: dict[str, float],
    baseline_sensitivity: dict[str, float],
    candidate_sensitivity: dict[str, float],
    raw_primary: dict[str, float],
    raw_sensitivity: dict[str, float],
    classical_primary: dict[str, float],
    classical_sensitivity: dict[str, float],
    legacy_primary: dict[str, float],
    legacy_sensitivity: dict[str, float],
    bootstrap_lower: float,
    raw_bootstrap_lower: float,
    classical_bootstrap_lower: float,
    legacy_bootstrap_lower: float,
) -> dict:
    primary_delta = candidate_primary["tile_ssim"] - baseline_primary["tile_ssim"]
    sensitivity_delta = candidate_sensitivity["tile_ssim"] - baseline_sensitivity["tile_ssim"]
    checks = {
        "real_primary_delta_at_least_0_003": primary_delta >= 0.003,
        "real_primary_bootstrap_lower_positive": bootstrap_lower > 0.0,
        "real_sensitivity_delta_positive": sensitivity_delta > 0.0,
        "real_primary_not_worse_than_raw": (
            candidate_primary["tile_ssim"] >= raw_primary["tile_ssim"]
        ),
        "real_primary_raw_bootstrap_lower_positive": raw_bootstrap_lower > 0.0,
        "real_sensitivity_not_worse_than_raw": (
            candidate_sensitivity["tile_ssim"] >= raw_sensitivity["tile_ssim"]
        ),
        "real_primary_not_worse_than_classical": (
            candidate_primary["tile_ssim"] >= classical_primary["tile_ssim"]
        ),
        "real_primary_classical_bootstrap_lower_positive": classical_bootstrap_lower > 0.0,
        "real_sensitivity_not_worse_than_classical": (
            candidate_sensitivity["tile_ssim"] >= classical_sensitivity["tile_ssim"]
        ),
        "real_primary_not_worse_than_legacy": (
            candidate_primary["tile_ssim"] >= legacy_primary["tile_ssim"]
        ),
        "real_primary_legacy_bootstrap_lower_positive": legacy_bootstrap_lower > 0.0,
        "real_sensitivity_not_worse_than_legacy": (
            candidate_sensitivity["tile_ssim"] >= legacy_sensitivity["tile_ssim"]
        ),
        "synthetic_ssim_drop_at_most_0_002": (
            candidate_synthetic["tile_ssim"] >= baseline_synthetic["tile_ssim"] - 0.002
        ),
        "synthetic_psnr_drop_at_most_0_10": (
            candidate_synthetic["psnr"] >= baseline_synthetic["psnr"] - 0.10
        ),
        "synthetic_boundary_mae_growth_at_most_1pct": (
            candidate_synthetic["boundary_mae"] <= baseline_synthetic["boundary_mae"] * 1.01
        ),
    }
    for channel in "rgb":
        key = f"signed_bias_{channel}"
        checks[f"synthetic_abs_{key}_growth_at_most_0_5"] = (
            abs(candidate_synthetic[key]) <= abs(baseline_synthetic[key]) + 0.5
        )
    synthetic_check_names = [name for name in checks if name.startswith("synthetic_")]
    return {
        "eligible": all(checks.values()),
        "synthetic_safe": all(checks[name] for name in synthetic_check_names),
        "primary_ssim_delta": primary_delta,
        "sensitivity_ssim_delta": sensitivity_delta,
        "checks": checks,
    }


def _model_snapshot_metrics(
    model: torch.nn.Module,
    synthetic_corrupt: np.ndarray,
    synthetic_clean: np.ndarray,
    primary_panel: RealPairBatch,
    sensitivity_panel: RealPairBatch,
    val_source_count: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    synthetic = evaluate_model(
        model,
        synthetic_corrupt,
        synthetic_clean,
        device,
        batch_size,
        complete_images=len(synthetic_clean) % 576 == 0,
    )
    primary_prediction = _predict_uint8(model, primary_panel.corrupt, device, batch_size)
    sensitivity_prediction = _predict_uint8(model, sensitivity_panel.corrupt, device, batch_size)
    return (
        {
            "synthetic": synthetic,
            "real_primary": _real_panel_metrics(primary_prediction, primary_panel, val_source_count),
            "real_sensitivity": _real_panel_metrics(
                sensitivity_prediction, sensitivity_panel, val_source_count
            ),
        },
        primary_prediction,
        sensitivity_prediction,
    )


def fine_tune(config: FineTuneConfig) -> dict:
    validate_fine_tune_config(config)
    if cv2.__version__ != config.expected_opencv_version:
        raise RuntimeError(
            "OpenCV version does not match the pinned classical-baseline runtime: "
            f"expected {config.expected_opencv_version}, got {cv2.__version__}"
        )
    started = time.perf_counter()
    seed_everything(config.seed)
    device = choose_device(config.device)
    root = Path(config.data_root)
    manifest_path = Path(config.manifest)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha256 != config.expected_manifest_sha256:
        raise ValueError("manifest SHA256 does not match the pinned expected digest")
    target_dir = root / "train" / "targets"
    synthetic_names = manifest["splits"]["train"][: config.synthetic_train_images]
    rng = np.random.default_rng(config.seed)

    train_table = RealPairTable.load(
        config.train_pairs,
        manifest_path=manifest_path,
        data_root=root,
        expected_split="train",
        min_confidence=config.train_min_confidence,
    )
    primary_table = RealPairTable.load(
        config.val_pairs,
        manifest_path=manifest_path,
        data_root=root,
        expected_split="val",
        min_confidence=config.val_primary_confidence,
    )
    sensitivity_table = RealPairTable.load(
        config.val_pairs,
        manifest_path=manifest_path,
        data_root=root,
        expected_split="val",
        min_confidence=config.val_sensitivity_confidence,
    )
    if train_table.npz_sha256 != config.expected_train_pairs_sha256:
        raise ValueError("train-pair SHA256 does not match the pinned expected digest")
    if primary_table.npz_sha256 != config.expected_val_pairs_sha256:
        raise ValueError("validation-pair SHA256 does not match the pinned expected digest")
    if sensitivity_table.npz_sha256 != primary_table.npz_sha256:
        raise ValueError("primary and sensitivity tables must come from the same pinned NPZ")
    if primary_table.source_names != sensitivity_table.source_names:
        raise ValueError("primary and sensitivity tables disagree on validation source names")
    pixel_fingerprints = fine_tune_pixel_fingerprints(
        root,
        synthetic_names,
        train_table.source_names,
        primary_table.source_names,
    )
    if pixel_fingerprints["training_pixels_sha256"] != config.expected_training_pixels_sha256:
        raise ValueError("training decoded-pixel SHA256 does not match the pinned expected digest")
    if (
        pixel_fingerprints["validation_pixels_sha256"]
        != config.expected_validation_pixels_sha256
    ):
        raise ValueError("validation decoded-pixel SHA256 does not match the pinned expected digest")
    synthetic_store = CleanTileStore(target_dir, synthetic_names)

    init_path = Path(config.init_checkpoint)
    init_sha256 = hashlib.sha256(init_path.read_bytes()).hexdigest()
    if init_sha256 != config.expected_init_checkpoint_sha256:
        raise ValueError("initial checkpoint SHA256 does not match the pinned expected digest")
    init_checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    if init_checkpoint.get("schema_version") != 2:
        raise ValueError("initial checkpoint must use the synthetic schema_version=2 format")
    if init_checkpoint.get("model_name") != config.model:
        raise ValueError("initial checkpoint model does not match fine-tune model")
    if init_checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("initial checkpoint manifest hash mismatch")
    saved_source_sha = init_checkpoint.get("source_code_sha256")
    if saved_source_sha != source_code_fingerprint():
        raise ValueError("initial checkpoint was produced by different restoration source code")
    required_init_fields = (
        "ema_state",
        "config",
        "training_data_sha256",
        "validation_data_sha256",
        "runtime_versions",
        "resolved_device_fingerprint",
    )
    missing_init_fields = [field for field in required_init_fields if field not in init_checkpoint]
    if missing_init_fields:
        raise ValueError(f"initial checkpoint is missing provenance/state: {missing_init_fields}")
    initial_config = init_checkpoint["config"]
    initial_validation_count = int(initial_config.get("val_images", 24))
    synthetic_checkpoint_validation_names = manifest["splits"]["val"][:initial_validation_count]

    legacy_sha256 = hashlib.sha256(Path(config.legacy_checkpoint).read_bytes()).hexdigest()
    if legacy_sha256 != config.expected_legacy_checkpoint_sha256:
        raise ValueError("legacy checkpoint SHA256 does not match the pinned expected digest")
    quarantine, quarantine_sha256 = load_validation_quarantine(
        config.quarantine_artifact,
        config.expected_quarantine_sha256,
        manifest_sha256=manifest_sha256,
        manifest_validation_names=manifest["splits"]["val"],
        expected_legacy_checkpoint_sha256=legacy_sha256,
        expected_synthetic_validation_names=synthetic_checkpoint_validation_names,
        gate_source_count=config.gate_source_count,
        seed=config.seed,
    )
    manifest_validation_set = set(manifest["splits"]["val"])
    table_validation_set = set(primary_table.source_names)
    if table_validation_set != manifest_validation_set:
        raise ValueError("validation pair table must contain every current manifest validation source")
    eligible_sources = np.intersect1d(
        primary_table.active_source_indices,
        sensitivity_table.active_source_indices,
        assume_unique=True,
    )
    eligible_names = {primary_table.source_names[int(index)] for index in eligible_sources}
    if eligible_names != manifest_validation_set:
        raise ValueError("confidence filtering must leave all 700 validation sources active")
    quarantine_names = tuple(quarantine["quarantine_names"])
    calibration_sources, gate_sources = deterministic_contamination_aware_split(
        primary_table.source_names,
        eligible_sources,
        quarantine_names,
        config.gate_source_count,
        config.seed,
    )
    if len(calibration_sources) != 257 or len(gate_sources) != 350:
        raise RuntimeError("contamination-aware split did not produce pinned 257/350 counts")
    calibration_source_names = sorted(
        primary_table.source_names[int(index)] for index in calibration_sources
    )
    gate_source_names = sorted(primary_table.source_names[int(index)] for index in gate_sources)
    clean_eligible_source_names = sorted(set(calibration_source_names) | set(gate_source_names))
    split_name_sha256 = {
        "quarantine": source_name_list_sha256(quarantine_names),
        "eligible_after_quarantine": source_name_list_sha256(clean_eligible_source_names),
        "calibration": source_name_list_sha256(calibration_source_names),
        "frozen_gate": source_name_list_sha256(gate_source_names),
    }
    for key, actual in split_name_sha256.items():
        if actual != quarantine["name_sha256"][key]:
            raise RuntimeError(f"contamination-aware {key} source-name hash mismatch")
    fine_tune_synthetic_validation_names = calibration_source_names[:initial_validation_count]
    if (
        set(calibration_source_names) & set(quarantine_names)
        or set(gate_source_names) & set(quarantine_names)
        or set(fine_tune_synthetic_validation_names) & set(quarantine_names)
    ):
        raise RuntimeError("quarantined source leaked into a fine-tune metric panel")
    if (
        set(calibration_source_names) & set(gate_source_names)
        or (set(quarantine_names) | set(clean_eligible_source_names))
        != manifest_validation_set
    ):
        raise RuntimeError("quarantine/calibration/gate do not partition manifest validation")
    real_sampler = RealPairSampler(train_table, seed=config.seed, cache_size=16)
    calibration_primary_panel = RealPairSampler(
        primary_table, seed=config.seed, cache_size=16
    ).materialize_validation(
        source_indices=calibration_sources,
        pairs_per_source=config.val_pairs_per_source,
        seed=config.seed,
    )
    calibration_sensitivity_panel = RealPairSampler(
        sensitivity_table, seed=config.seed + 1, cache_size=16
    ).materialize_validation(
        source_indices=calibration_sources,
        pairs_per_source=config.val_pairs_per_source,
        seed=config.seed + 1,
    )

    model = build_model(config.model).to(device)
    model.load_state_dict(init_checkpoint["ema_state"])
    ema = copy.deepcopy(model).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    rollback_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    validation_names = fine_tune_synthetic_validation_names
    degrader = SyntheticTileDegrader(
        variant_weights=tuple(initial_config.get("variant_weights", (1.0, 0.0, 0.0)))
    )
    synthetic_plan = make_fixed_validation_plan(
        target_dir,
        validation_names,
        int(initial_config.get("val_tiles_per_image", 576)),
        int(initial_config.get("seed", config.seed)) + 1,
        degrader,
    )
    synthetic_corrupt = render_fixed_validation(
        synthetic_plan,
        degrader,
        batch_size=min(config.batch_size * 2, 512),
        codec="kornia",
    )
    synthetic_clean = synthetic_plan.clean

    calibration_baseline, calibration_primary_prediction, calibration_sensitivity_prediction = (
        _model_snapshot_metrics(
            ema,
            synthetic_corrupt,
            synthetic_clean,
            calibration_primary_panel,
            calibration_sensitivity_panel,
            primary_table.source_count,
            device,
            min(config.batch_size * 2, 512),
        )
    )
    calibration_primary_raw_prediction = _uint8_tiles(calibration_primary_panel.corrupt)
    calibration_sensitivity_raw_prediction = _uint8_tiles(calibration_sensitivity_panel.corrupt)
    calibration_baseline["real_primary_raw"] = _real_panel_metrics(
        calibration_primary_raw_prediction,
        calibration_primary_panel,
        primary_table.source_count,
    )
    calibration_baseline["real_sensitivity_raw"] = _real_panel_metrics(
        calibration_sensitivity_raw_prediction,
        calibration_sensitivity_panel,
        sensitivity_table.source_count,
    )
    calibration_primary_classical_prediction = _classical_tiles_uint8(
        calibration_primary_raw_prediction
    )
    calibration_sensitivity_classical_prediction = _classical_tiles_uint8(
        calibration_sensitivity_raw_prediction
    )
    calibration_baseline["real_primary_classical"] = _real_panel_metrics(
        calibration_primary_classical_prediction,
        calibration_primary_panel,
        primary_table.source_count,
    )
    calibration_baseline["real_sensitivity_classical"] = _real_panel_metrics(
        calibration_sensitivity_classical_prediction,
        calibration_sensitivity_panel,
        sensitivity_table.source_count,
    )
    legacy_model, legacy_device, legacy_metadata = load_legacy_tile_restorer(
        config.legacy_checkpoint,
        expected_sha256=config.expected_legacy_checkpoint_sha256,
        device=device,
    )
    calibration_primary_legacy_prediction = predict_legacy_tiles_uint8(
        legacy_model,
        calibration_primary_raw_prediction,
        legacy_device,
        min(config.batch_size * 2, 512),
    )
    calibration_sensitivity_legacy_prediction = predict_legacy_tiles_uint8(
        legacy_model,
        calibration_sensitivity_raw_prediction,
        legacy_device,
        min(config.batch_size * 2, 512),
    )
    calibration_baseline["real_primary_legacy"] = _real_panel_metrics(
        calibration_primary_legacy_prediction,
        calibration_primary_panel,
        primary_table.source_count,
    )
    calibration_baseline["real_sensitivity_legacy"] = _real_panel_metrics(
        calibration_sensitivity_legacy_prediction,
        calibration_sensitivity_panel,
        sensitivity_table.source_count,
    )
    del legacy_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def selected_pair_count(table: RealPairTable, sources: np.ndarray) -> int:
        return sum(
            min(len(table.source_rows(int(source_index))), config.val_pairs_per_source)
            for source_index in sources
        )

    source_index_by_name = {
        name: index for index, name in enumerate(primary_table.source_names)
    }
    quarantine_source_indices = sorted(
        source_index_by_name[name] for name in quarantine_names
    )
    source_split = {
        "protocol": "contamination_aware_quarantine_v1",
        "quarantine_artifact": str(Path(config.quarantine_artifact)),
        "quarantine_artifact_sha256": quarantine_sha256,
        "quarantine_excluded_from_metrics": True,
        "manifest_validation_source_count": len(manifest_validation_set),
        "quarantine_source_count": len(quarantine_names),
        "clean_eligible_source_count": len(clean_eligible_source_names),
        "calibration_source_count": len(calibration_sources),
        "gate_source_count": len(gate_sources),
        "legacy_train_seen_source_count": len(quarantine["legacy_train_seen_names"]),
        "legacy_validation_seen_source_count": len(
            quarantine["legacy_validation_seen_names"]
        ),
        "synthetic_checkpoint_validation_source_count": len(
            synthetic_checkpoint_validation_names
        ),
        "fine_tune_synthetic_validation_source_count": len(
            fine_tune_synthetic_validation_names
        ),
        "quarantine_source_indices": quarantine_source_indices,
        "quarantine_source_names": list(quarantine_names),
        "calibration_source_indices": calibration_sources.tolist(),
        "gate_source_indices": gate_sources.tolist(),
        "clean_eligible_source_names": clean_eligible_source_names,
        "calibration_source_names": calibration_source_names,
        "gate_source_names": gate_source_names,
        "legacy_train_seen_names": quarantine["legacy_train_seen_names"],
        "legacy_validation_seen_names": quarantine["legacy_validation_seen_names"],
        "synthetic_checkpoint_validation_names": synthetic_checkpoint_validation_names,
        "fine_tune_synthetic_validation_names": fine_tune_synthetic_validation_names,
        "source_name_sha256": {
            **quarantine["name_sha256"],
            "fine_tune_synthetic_validation": source_name_list_sha256(
                fine_tune_synthetic_validation_names
            ),
        },
        "calibration_primary_pairs": len(calibration_primary_panel),
        "calibration_sensitivity_pairs": len(calibration_sensitivity_panel),
        "calibration_primary_sensitivity_pair_overlap": int(
            len(
                np.intersect1d(
                    calibration_primary_panel.pair_row.numpy(),
                    calibration_sensitivity_panel.pair_row.numpy(),
                    assume_unique=True,
                )
            )
        ),
        "calibration_sensitivity_is_independent": False,
        "gate_primary_pairs": selected_pair_count(primary_table, gate_sources),
        "gate_sensitivity_pairs": selected_pair_count(sensitivity_table, gate_sources),
        "gate_sensitivity_is_independent": False,
    }
    baseline = {"calibration": calibration_baseline}

    encoder_parameters = list(model.degradation_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    restoration_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": restoration_parameters, "lr": config.peak_learning_rate},
            {"params": encoder_parameters, "lr": config.peak_learning_rate * config.encoder_lr_scale},
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.99),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = RestorationLoss(LossWeights(ssim=0.10))
    output = Path(config.output)
    latest_output = output.with_name(output.stem + "_unsafe_latest" + output.suffix)
    candidate_output = output.with_name(output.stem + "_calibration_candidate" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_real_ssim = -math.inf
    best_step: int | None = None
    best_observed_ssim = calibration_baseline["real_primary"]["macro"]["tile_ssim"]
    evaluations_without_gain = 0
    consecutive_synthetic_failures = 0
    stopped_early = False
    stopped_reason: str | None = None
    running: dict[str, float] = {}
    running_batches = 0
    versions = runtime_versions()
    code_sha256 = fine_tune_code_fingerprint()
    device_fingerprint = resolved_device_fingerprint(device)

    rollback_checkpoint = {
        "schema_version": 1,
        "kind": "conservative_real_pair_fine_tune_rollback",
        "promotion_status": "rollback_safe",
        "safe_for_inference": True,
        "model_name": config.model,
        "model_state": rollback_state,
        "ema_state": rollback_state,
        "step": 0,
        "config": asdict(config),
        "manifest_sha256": manifest_sha256,
        "validation_quarantine_sha256": quarantine_sha256,
        "maps_1024_sha256": quarantine["maps_artifact"]["sha256"],
        "fine_tune_code_sha256": code_sha256,
        "init_checkpoint_sha256": init_sha256,
        "legacy_checkpoint_sha256": legacy_metadata["checkpoint_sha256"],
        "legacy_baseline_metadata": legacy_metadata,
        "train_pairs_sha256": train_table.npz_sha256,
        "val_pairs_sha256": primary_table.npz_sha256,
        **pixel_fingerprints,
        "runtime_versions": versions,
        "resolved_device_fingerprint": device_fingerprint,
        "source_split": source_split,
        "baseline": baseline,
        "history": history,
        "rolled_back": True,
        "reason": "safe synthetic EMA written before fine-tune updates",
    }
    atomic_torch_save(rollback_checkpoint, output)
    diagnostic_placeholder = {
        **rollback_checkpoint,
        "kind": "conservative_real_pair_fine_tune",
        "promotion_status": "diagnostic_placeholder",
        "safe_for_inference": False,
        "rolled_back": False,
        "best_step": None,
        "best_real_ssim": None,
        "reason": "invalidates diagnostic artifacts left by an earlier run",
    }
    atomic_torch_save(diagnostic_placeholder, latest_output)
    atomic_torch_save(
        {**diagnostic_placeholder, "promotion_status": "no_calibration_candidate"},
        candidate_output,
    )

    print(
        json.dumps(
            {
                "event": "real_fine_tune_start",
                "config": asdict(config),
                "device": str(device),
                "device_fingerprint": device_fingerprint,
                "runtime_versions": versions,
                "fine_tune_code_sha256": code_sha256,
                "init_checkpoint_sha256": init_sha256,
                "validation_quarantine_sha256": quarantine_sha256,
                "maps_1024_sha256": quarantine["maps_artifact"]["sha256"],
                "legacy_checkpoint_sha256": legacy_metadata["checkpoint_sha256"],
                "legacy_baseline_metadata": legacy_metadata,
                "train_pairs_sha256": train_table.npz_sha256,
                "val_pairs_sha256": primary_table.npz_sha256,
                **pixel_fingerprints,
                "source_split": source_split,
                "baseline": baseline,
                "crash_safe_rollback": str(output),
                "unsafe_latest": str(latest_output),
                "calibration_candidate": str(candidate_output),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    last_step = 0
    for step in range(1, config.steps + 1):
        if config.max_seconds is not None and time.perf_counter() - started >= config.max_seconds:
            stopped_early = True
            stopped_reason = "max_seconds reached before the next update"
            print(
                json.dumps(
                    {"event": "real_fine_tune_early_stop", "step": last_step, "reason": stopped_reason},
                    sort_keys=True,
                ),
                flush=True,
            )
            break
        last_step = step
        scale = learning_rate_scale(step, config)
        optimizer.param_groups[0]["lr"] = config.peak_learning_rate * scale
        optimizer.param_groups[1]["lr"] = (
            config.peak_learning_rate * config.encoder_lr_scale * scale
        )
        real_batch = is_real_batch_step(step, config)
        if real_batch:
            batch = real_sampler.sample_grouped(
                config.batch_size,
                config.pairs_per_real_source,
            )
            corrupt, clean = paired_dihedral(batch.corrupt, batch.clean, rng)
            degradation_target = None
        else:
            clean = synthetic_store.sample(config.batch_size, rng).to(device)
            clean = random_dihedral(clean, rng)
            with torch.no_grad():
                corrupt, degradation_parameters = degrader(clean)
            degradation_target = degradation_parameters.normalized()

        corrupt = corrupt.to(device)
        clean = clean.to(device)
        if degradation_target is not None:
            degradation_target = degradation_target.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction, parameter_prediction = model(corrupt, return_aux=True)
            if real_batch:
                loss, components = criterion(prediction, clean)
            else:
                loss, components = criterion(
                    prediction,
                    clean,
                    parameter_prediction,
                    degradation_target,
                )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema, model, config.ema_decay)

        for key, value in components.items():
            running[key] = running.get(key, 0.0) + float(value.cpu())
        running["real_batches"] = running.get("real_batches", 0.0) + float(real_batch)
        running_batches += 1

        if step % config.log_interval == 0 or step == 1:
            print(
                json.dumps(
                    {
                        "event": "real_fine_tune_step",
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "seconds": time.perf_counter() - started,
                        **{key: value / running_batches for key, value in running.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            running = {}
            running_batches = 0

        if step % config.eval_interval != 0 and step != config.steps:
            continue

        raw_metrics, _, _ = _model_snapshot_metrics(
            model,
            synthetic_corrupt,
            synthetic_clean,
            calibration_primary_panel,
            calibration_sensitivity_panel,
            primary_table.source_count,
            device,
            min(config.batch_size * 2, 512),
        )
        ema_metrics, primary_prediction, sensitivity_prediction = _model_snapshot_metrics(
            ema,
            synthetic_corrupt,
            synthetic_clean,
            calibration_primary_panel,
            calibration_sensitivity_panel,
            primary_table.source_count,
            device,
            min(config.batch_size * 2, 512),
        )
        bootstrap = paired_source_bootstrap_delta(
            primary_prediction,
            calibration_primary_prediction,
            _uint8_tiles(calibration_primary_panel.clean),
            calibration_primary_panel.source_index.numpy(),
            metric="tile_ssim",
            source_count=primary_table.source_count,
            resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        raw_bootstrap = paired_source_bootstrap_delta(
            primary_prediction,
            calibration_primary_raw_prediction,
            _uint8_tiles(calibration_primary_panel.clean),
            calibration_primary_panel.source_index.numpy(),
            metric="tile_ssim",
            source_count=primary_table.source_count,
            resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        classical_bootstrap = paired_source_bootstrap_delta(
            primary_prediction,
            calibration_primary_classical_prediction,
            _uint8_tiles(calibration_primary_panel.clean),
            calibration_primary_panel.source_index.numpy(),
            metric="tile_ssim",
            source_count=primary_table.source_count,
            resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        legacy_bootstrap = paired_source_bootstrap_delta(
            primary_prediction,
            calibration_primary_legacy_prediction,
            _uint8_tiles(calibration_primary_panel.clean),
            calibration_primary_panel.source_index.numpy(),
            metric="tile_ssim",
            source_count=primary_table.source_count,
            resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        assessment = assess_candidate(
            calibration_baseline["synthetic"],
            ema_metrics["synthetic"],
            calibration_baseline["real_primary"]["macro"],
            ema_metrics["real_primary"]["macro"],
            calibration_baseline["real_sensitivity"]["macro"],
            ema_metrics["real_sensitivity"]["macro"],
            calibration_baseline["real_primary_raw"]["macro"],
            calibration_baseline["real_sensitivity_raw"]["macro"],
            calibration_baseline["real_primary_classical"]["macro"],
            calibration_baseline["real_sensitivity_classical"]["macro"],
            calibration_baseline["real_primary_legacy"]["macro"],
            calibration_baseline["real_sensitivity_legacy"]["macro"],
            bootstrap.lower,
            raw_bootstrap.lower,
            classical_bootstrap.lower,
            legacy_bootstrap.lower,
        )
        record = {
            "panel": "calibration",
            "step": step,
            "raw": raw_metrics,
            "ema": ema_metrics,
            "primary_bootstrap": asdict(bootstrap),
            "primary_vs_raw_bootstrap": asdict(raw_bootstrap),
            "primary_vs_classical_bootstrap": asdict(classical_bootstrap),
            "primary_vs_legacy_bootstrap": asdict(legacy_bootstrap),
            "assessment": assessment,
        }
        history.append(record)
        print(json.dumps({"event": "real_fine_tune_validation", **record}, sort_keys=True), flush=True)

        if assessment["synthetic_safe"]:
            consecutive_synthetic_failures = 0
        else:
            consecutive_synthetic_failures += 1

        candidate_real_ssim = ema_metrics["real_primary"]["macro"]["tile_ssim"]
        if candidate_real_ssim >= best_observed_ssim + config.no_gain_min_delta:
            best_observed_ssim = candidate_real_ssim
            evaluations_without_gain = 0
        else:
            evaluations_without_gain += 1
        improved = assessment["eligible"] and candidate_real_ssim > best_real_ssim
        if improved:
            best_real_ssim = candidate_real_ssim
            best_step = step

        checkpoint = {
            "schema_version": 1,
            "kind": "conservative_real_pair_fine_tune",
            "promotion_status": "diagnostic_unvalidated",
            "safe_for_inference": False,
            "model_name": config.model,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "step": step,
            "config": asdict(config),
            "manifest_sha256": manifest_sha256,
            "validation_quarantine_sha256": quarantine_sha256,
            "maps_1024_sha256": quarantine["maps_artifact"]["sha256"],
            "fine_tune_code_sha256": code_sha256,
            "init_checkpoint_sha256": init_sha256,
            "legacy_checkpoint_sha256": legacy_metadata["checkpoint_sha256"],
            "legacy_baseline_metadata": legacy_metadata,
            "train_pairs_sha256": train_table.npz_sha256,
            "val_pairs_sha256": primary_table.npz_sha256,
            **pixel_fingerprints,
            "runtime_versions": versions,
            "resolved_device_fingerprint": device_fingerprint,
            "source_split": source_split,
            "baseline": baseline,
            "history": history,
            "latest_validation": record,
            "best_real_ssim": best_real_ssim,
            "best_step": best_step,
        }
        atomic_torch_save(checkpoint, latest_output)

        if improved:
            checkpoint["promotion_status"] = "calibration_candidate"
            checkpoint["calibration_validation"] = record
            atomic_torch_save(checkpoint, candidate_output)
            print(
                json.dumps(
                    {
                        "event": "real_fine_tune_calibration_candidate_saved",
                        "path": str(candidate_output),
                        "step": step,
                        "real_primary_macro_ssim": best_real_ssim,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if consecutive_synthetic_failures >= 2:
            stopped_early = True
            stopped_reason = "two consecutive synthetic safety violations"
            print(
                json.dumps(
                    {
                        "event": "real_fine_tune_early_stop",
                        "step": step,
                        "reason": stopped_reason,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break
        if evaluations_without_gain >= config.no_gain_patience:
            stopped_early = True
            stopped_reason = (
                f"no calibration SSIM gain >= {config.no_gain_min_delta} for "
                f"{config.no_gain_patience} evaluations"
            )
            print(
                json.dumps(
                    {"event": "real_fine_tune_early_stop", "step": step, "reason": stopped_reason},
                    sort_keys=True,
                ),
                flush=True,
            )
            break

    gate_record: dict | None = None
    rolled_back = True
    promoted_step: int | None = None
    if best_step is not None:
        candidate_checkpoint = torch.load(candidate_output, map_location="cpu", weights_only=False)

        # The gate panel is not materialized or evaluated until calibration has
        # selected one immutable candidate.  Everything below is a single
        # terminal look at the source-disjoint gate.
        gate_primary_panel = RealPairSampler(
            primary_table, seed=config.seed + 2, cache_size=16
        ).materialize_validation(
            source_indices=gate_sources,
            pairs_per_source=config.val_pairs_per_source,
            seed=config.seed + 2,
        )
        gate_sensitivity_panel = RealPairSampler(
            sensitivity_table, seed=config.seed + 3, cache_size=16
        ).materialize_validation(
            source_indices=gate_sources,
            pairs_per_source=config.val_pairs_per_source,
            seed=config.seed + 3,
        )
        if (
            len(gate_primary_panel) != source_split["gate_primary_pairs"]
            or len(gate_sensitivity_panel) != source_split["gate_sensitivity_pairs"]
        ):
            raise RuntimeError("materialized gate pair count differs from the sealed source split")
        source_split["gate_primary_sensitivity_pair_overlap"] = int(
            len(
                np.intersect1d(
                    gate_primary_panel.pair_row.numpy(),
                    gate_sensitivity_panel.pair_row.numpy(),
                    assume_unique=True,
                )
            )
        )

        ema.load_state_dict(rollback_state)
        gate_baseline, gate_primary_prediction, gate_sensitivity_prediction = (
            _model_snapshot_metrics(
                ema,
                synthetic_corrupt,
                synthetic_clean,
                gate_primary_panel,
                gate_sensitivity_panel,
                primary_table.source_count,
                device,
                min(config.batch_size * 2, 512),
            )
        )
        gate_primary_raw_prediction = _uint8_tiles(gate_primary_panel.corrupt)
        gate_sensitivity_raw_prediction = _uint8_tiles(gate_sensitivity_panel.corrupt)
        gate_baseline["real_primary_raw"] = _real_panel_metrics(
            gate_primary_raw_prediction,
            gate_primary_panel,
            primary_table.source_count,
        )
        gate_baseline["real_sensitivity_raw"] = _real_panel_metrics(
            gate_sensitivity_raw_prediction,
            gate_sensitivity_panel,
            sensitivity_table.source_count,
        )
        gate_primary_classical_prediction = _classical_tiles_uint8(
            gate_primary_raw_prediction
        )
        gate_sensitivity_classical_prediction = _classical_tiles_uint8(
            gate_sensitivity_raw_prediction
        )
        gate_baseline["real_primary_classical"] = _real_panel_metrics(
            gate_primary_classical_prediction,
            gate_primary_panel,
            primary_table.source_count,
        )
        gate_baseline["real_sensitivity_classical"] = _real_panel_metrics(
            gate_sensitivity_classical_prediction,
            gate_sensitivity_panel,
            sensitivity_table.source_count,
        )
        legacy_model, legacy_device, gate_legacy_metadata = load_legacy_tile_restorer(
            config.legacy_checkpoint,
            expected_sha256=config.expected_legacy_checkpoint_sha256,
            device=device,
        )
        if gate_legacy_metadata["checkpoint_sha256"] != legacy_metadata["checkpoint_sha256"]:
            raise RuntimeError("legacy checkpoint changed between calibration and frozen gate")
        gate_primary_legacy_prediction = predict_legacy_tiles_uint8(
            legacy_model,
            gate_primary_raw_prediction,
            legacy_device,
            min(config.batch_size * 2, 512),
        )
        gate_sensitivity_legacy_prediction = predict_legacy_tiles_uint8(
            legacy_model,
            gate_sensitivity_raw_prediction,
            legacy_device,
            min(config.batch_size * 2, 512),
        )
        gate_baseline["real_primary_legacy"] = _real_panel_metrics(
            gate_primary_legacy_prediction,
            gate_primary_panel,
            primary_table.source_count,
        )
        gate_baseline["real_sensitivity_legacy"] = _real_panel_metrics(
            gate_sensitivity_legacy_prediction,
            gate_sensitivity_panel,
            sensitivity_table.source_count,
        )
        del legacy_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        baseline["gate"] = gate_baseline

        ema.load_state_dict(candidate_checkpoint["ema_state"])
        gate_metrics, gate_candidate_primary, gate_candidate_sensitivity = _model_snapshot_metrics(
            ema,
            synthetic_corrupt,
            synthetic_clean,
            gate_primary_panel,
            gate_sensitivity_panel,
            primary_table.source_count,
            device,
            min(config.batch_size * 2, 512),
        )
        gate_bootstrap = paired_source_bootstrap_delta(
            gate_candidate_primary,
            gate_primary_prediction,
            _uint8_tiles(gate_primary_panel.clean),
            gate_primary_panel.source_index.numpy(),
            metric="tile_ssim",
            source_count=primary_table.source_count,
            resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        gate_raw_bootstrap = paired_source_bootstrap_delta(
            gate_candidate_primary,
            gate_primary_raw_prediction,
            _uint8_tiles(gate_primary_panel.clean),
            gate_primary_panel.source_index.numpy(),
            metric="tile_ssim",
            source_count=primary_table.source_count,
            resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        gate_classical_bootstrap = paired_source_bootstrap_delta(
            gate_candidate_primary,
            gate_primary_classical_prediction,
            _uint8_tiles(gate_primary_panel.clean),
            gate_primary_panel.source_index.numpy(),
            metric="tile_ssim",
            source_count=primary_table.source_count,
            resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        gate_legacy_bootstrap = paired_source_bootstrap_delta(
            gate_candidate_primary,
            gate_primary_legacy_prediction,
            _uint8_tiles(gate_primary_panel.clean),
            gate_primary_panel.source_index.numpy(),
            metric="tile_ssim",
            source_count=primary_table.source_count,
            resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        gate_assessment = assess_candidate(
            gate_baseline["synthetic"],
            gate_metrics["synthetic"],
            gate_baseline["real_primary"]["macro"],
            gate_metrics["real_primary"]["macro"],
            gate_baseline["real_sensitivity"]["macro"],
            gate_metrics["real_sensitivity"]["macro"],
            gate_baseline["real_primary_raw"]["macro"],
            gate_baseline["real_sensitivity_raw"]["macro"],
            gate_baseline["real_primary_classical"]["macro"],
            gate_baseline["real_sensitivity_classical"]["macro"],
            gate_baseline["real_primary_legacy"]["macro"],
            gate_baseline["real_sensitivity_legacy"]["macro"],
            gate_bootstrap.lower,
            gate_raw_bootstrap.lower,
            gate_classical_bootstrap.lower,
            gate_legacy_bootstrap.lower,
        )
        gate_record = {
            "panel": "frozen_gate",
            "selected_step": best_step,
            "baseline": gate_baseline,
            "ema": gate_metrics,
            "primary_bootstrap": asdict(gate_bootstrap),
            "primary_vs_raw_bootstrap": asdict(gate_raw_bootstrap),
            "primary_vs_classical_bootstrap": asdict(gate_classical_bootstrap),
            "primary_vs_legacy_bootstrap": asdict(gate_legacy_bootstrap),
            "assessment": gate_assessment,
        }
        print(json.dumps({"event": "real_fine_tune_frozen_gate", **gate_record}, sort_keys=True), flush=True)
        if gate_assessment["eligible"]:
            promoted_checkpoint = dict(candidate_checkpoint)
            promoted_checkpoint.update(
                {
                    "promotion_status": "promoted",
                    "safe_for_inference": True,
                    "rolled_back": False,
                    "model_state": candidate_checkpoint["ema_state"],
                    "best_validation": candidate_checkpoint["calibration_validation"],
                    "gate_validation": gate_record,
                    "baseline": baseline,
                    "source_split": source_split,
                    "history": history,
                }
            )
            atomic_torch_save(promoted_checkpoint, output)
            rolled_back = False
            promoted_step = best_step

    if rolled_back:
        rollback_checkpoint.update(
            {
                "history": history,
                "best_step": best_step,
                "gate_validation": gate_record,
                "reason": (
                    "no calibration checkpoint satisfied every promotion gate"
                    if best_step is None
                    else "best calibration checkpoint failed the frozen source-disjoint gate"
                ),
            }
        )
        atomic_torch_save(rollback_checkpoint, output)

    return {
        "output": str(output),
        "latest_output": str(latest_output),
        "candidate_output": str(candidate_output),
        "best_step": promoted_step,
        "calibration_best_step": best_step,
        "best_real_ssim": None if best_step is None else best_real_ssim,
        "rolled_back": rolled_back,
        "stopped_early": stopped_early,
        "stopped_reason": stopped_reason,
        "source_split": source_split,
        "baseline": baseline,
        "history": history,
        "gate_validation": gate_record,
        "seconds": time.perf_counter() - started,
    }
