"""CPU-only calibration benchmark for deciding whether real-pair fine-tuning is justified.

This module excludes the pinned 93-source contamination quarantine, then
evaluates only the deterministic 257-source clean calibration partition.  It
computes the sealed 350-source gate names so the split can be audited, but no
gate tile is passed to a model or metric.  The all-700 decoded-pixel hash is an
integrity check only.  The output is a diagnostic; it cannot start training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping

import cv2
import numpy as np
import torch

from .inference import load_restorer, restore_tiles_uint8
from .legacy_baseline import (
    load_legacy_tile_restorer,
    predict_legacy_tiles_uint8,
    sha256_file,
)
from .real_pairs import RealPairBatch, RealPairSampler, RealPairTable
from .real_training import (
    deterministic_contamination_aware_split,
    fine_tune_pixel_fingerprints,
    load_validation_quarantine,
    source_name_list_sha256,
)
from .real_validation import evaluate_real_pairs, paired_source_bootstrap_delta
from .training import load_manifest, runtime_versions, source_code_fingerprint


PROTOCOL_SEED = 20260710
EXPECTED_VALIDATION_SOURCES = 700
QUARANTINE_SOURCE_COUNT = 93
CLEAN_ELIGIBLE_SOURCE_COUNT = 607
CALIBRATION_SOURCE_COUNT = 257
SEALED_GATE_SOURCE_COUNT = 350
PAIRS_PER_SOURCE = 8
PRIMARY_CONFIDENCE = 1.5
SENSITIVITY_CONFIDENCE = 1.0
MAX_LEGACY_SSIM_DEFICIT = 0.01
NLM_PARAMETERS = {
    "h": 7.0,
    "h_color": 7.0,
    "template_window_size": 5,
    "search_window_size": 11,
}

# Every project source file that can affect panel selection, model loading,
# prediction, baselines, metrics, or the bounded diagnostic is pinned together.
PREFINETUNE_BENCHMARK_CODE_FILES = (
    "__init__.py",
    "degradation.py",
    "inference.py",
    "legacy_baseline.py",
    "losses.py",
    "metrics.py",
    "model.py",
    "prefinetune_benchmark.py",
    "real_pairs.py",
    "real_training.py",
    "real_validation.py",
    "tiles.py",
    "training.py",
)


@dataclass(frozen=True)
class PreFineTuneBenchmarkConfig:
    data_root: str
    manifest: str
    val_pairs: str
    init_checkpoint: str
    legacy_checkpoint: str
    quarantine_artifact: str
    output: str
    expected_manifest_sha256: str
    expected_val_pairs_sha256: str
    expected_init_checkpoint_sha256: str
    expected_legacy_checkpoint_sha256: str
    expected_quarantine_sha256: str
    expected_validation_pixels_sha256: str
    expected_code_sha256: str
    expected_opencv_version: str
    gate_source_count: int = SEALED_GATE_SOURCE_COUNT
    batch_size: int = 128
    bootstrap_resamples: int = 5000
    torch_threads: int = 4
    max_legacy_ssim_deficit: float = MAX_LEGACY_SSIM_DEFICIT


def _require_sha256(name: str, value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def validate_prefinetune_config(config: PreFineTuneBenchmarkConfig) -> None:
    for name in (
        "expected_manifest_sha256",
        "expected_val_pairs_sha256",
        "expected_init_checkpoint_sha256",
        "expected_legacy_checkpoint_sha256",
        "expected_quarantine_sha256",
        "expected_validation_pixels_sha256",
        "expected_code_sha256",
    ):
        _require_sha256(name, getattr(config, name))
    if re.fullmatch(r"\d+(?:\.\d+){1,3}", config.expected_opencv_version) is None:
        raise ValueError("expected_opencv_version must be a dotted numeric version")
    for name in ("batch_size", "bootstrap_resamples", "torch_threads", "gate_source_count"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.bootstrap_resamples < 1000:
        raise ValueError("bootstrap_resamples must be at least 1000")
    if config.torch_threads > 16:
        raise ValueError("torch_threads is intentionally bounded at 16")
    if config.gate_source_count != SEALED_GATE_SOURCE_COUNT:
        raise ValueError(f"gate_source_count must be exactly {SEALED_GATE_SOURCE_COUNT}")
    if (
        not math.isfinite(config.max_legacy_ssim_deficit)
        or not 0.0 <= config.max_legacy_ssim_deficit <= 0.02
    ):
        raise ValueError("max_legacy_ssim_deficit must be finite and in [0, 0.02]")


def prefinetune_benchmark_code_fingerprint(
    package_dir: str | Path | None = None,
) -> str:
    """Hash the exact transitive project code used by this benchmark."""
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in PREFINETUNE_BENCHMARK_CODE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"benchmark code file is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def select_frozen_calibration_sources(
    source_names: tuple[str, ...],
    primary_active_sources: np.ndarray,
    sensitivity_active_sources: np.ndarray,
    quarantine_names: tuple[str, ...],
    *,
    gate_source_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map names to the frozen 93/257/350 quarantine-aware partition."""
    if len(source_names) != EXPECTED_VALIDATION_SOURCES:
        raise ValueError(
            f"expected {EXPECTED_VALIDATION_SOURCES} validation sources, got {len(source_names)}"
        )
    if len(set(source_names)) != len(source_names):
        raise ValueError("validation source names contain duplicates")
    if (
        len(quarantine_names) != QUARANTINE_SOURCE_COUNT
        or tuple(sorted(quarantine_names)) != quarantine_names
        or len(set(quarantine_names)) != len(quarantine_names)
    ):
        raise ValueError("quarantine_names must contain exactly 93 sorted unique names")
    if not set(quarantine_names) <= set(source_names):
        raise ValueError("quarantine_names contains a name outside validation")
    if gate_source_count != SEALED_GATE_SOURCE_COUNT:
        raise ValueError(f"gate_source_count must be exactly {SEALED_GATE_SOURCE_COUNT}")
    eligible = np.intersect1d(
        np.asarray(primary_active_sources, dtype=np.int64),
        np.asarray(sensitivity_active_sources, dtype=np.int64),
        assume_unique=False,
    )
    expected = np.arange(EXPECTED_VALIDATION_SOURCES, dtype=np.int64)
    if not np.array_equal(eligible, expected):
        missing = np.setdiff1d(expected, eligible).tolist()
        raise ValueError(
            "primary and sensitivity panels must cover every validation source; "
            f"missing={missing[:10]}"
        )
    calibration, gate = deterministic_contamination_aware_split(
        source_names,
        eligible,
        quarantine_names,
        gate_source_count,
        PROTOCOL_SEED,
    )
    if len(calibration) != CALIBRATION_SOURCE_COUNT or len(gate) != SEALED_GATE_SOURCE_COUNT:
        raise RuntimeError("quarantine-aware source split is not exactly 257/350")
    quarantine_set = set(quarantine_names)
    calibration_names = {source_names[int(index)] for index in calibration}
    gate_names = {source_names[int(index)] for index in gate}
    if calibration_names & quarantine_set or gate_names & quarantine_set:
        raise RuntimeError("quarantined source leaked into a clean metric partition")
    if calibration_names & gate_names:
        raise RuntimeError("calibration and sealed gate overlap")
    if calibration_names | gate_names | quarantine_set != set(source_names):
        raise RuntimeError("quarantine/calibration/gate do not partition validation")
    return calibration, gate


def classical_nlm_tiles_uint8(tiles: np.ndarray) -> np.ndarray:
    """Apply the fixed OpenCV colored-NLM control independently to each tile."""
    array = np.asarray(tiles)
    if array.ndim != 4 or array.shape[1:] != (20, 20, 3):
        raise ValueError(f"expected Nx20x20x3 tiles, got {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"expected uint8 tiles, got {array.dtype}")
    if len(array) == 0:
        raise ValueError("tile array must not be empty")
    restored = np.empty_like(array)
    for index, tile in enumerate(array):
        bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            NLM_PARAMETERS["h"],
            NLM_PARAMETERS["h_color"],
            NLM_PARAMETERS["template_window_size"],
            NLM_PARAMETERS["search_window_size"],
        )
        restored[index] = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    return restored


def _batch_uint8(tensor: torch.Tensor) -> np.ndarray:
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


def _canonical_array_bytes(array: np.ndarray, dtype: str) -> bytes:
    return np.ascontiguousarray(np.asarray(array, dtype=np.dtype(dtype))).tobytes()


def _panel_hashes(panel: RealPairBatch, source_names: tuple[str, ...]) -> dict[str, str]:
    source_indices = panel.source_index.numpy()
    selection = hashlib.sha256()
    for source_index in source_indices:
        encoded = source_names[int(source_index)].encode("utf-8")
        selection.update(len(encoded).to_bytes(2, "little"))
        selection.update(encoded)
    for tensor in (
        panel.source_index,
        panel.input_slot,
        panel.clean_tile_index,
        panel.pair_row,
    ):
        selection.update(_canonical_array_bytes(tensor.numpy(), "<i8"))
    selection.update(_canonical_array_bytes(panel.confidence.numpy(), "<f4"))

    pixels = hashlib.sha256()
    pixels.update(_batch_uint8(panel.corrupt).tobytes())
    pixels.update(_batch_uint8(panel.clean).tobytes())
    return {
        "selection_sha256": selection.hexdigest(),
        "decoded_panel_pixels_sha256": pixels.hexdigest(),
    }


def _validate_panel(
    panel: RealPairBatch,
    calibration_sources: np.ndarray,
    source_names: tuple[str, ...],
    table: RealPairTable,
) -> dict:
    expected_pairs = CALIBRATION_SOURCE_COUNT * PAIRS_PER_SOURCE
    if len(panel) != expected_pairs:
        raise ValueError(f"expected exactly {expected_pairs} panel pairs, got {len(panel)}")
    source_indices = panel.source_index.numpy()
    if not np.array_equal(np.unique(source_indices), calibration_sources):
        raise ValueError("materialized panel contains a non-calibration source")
    counts = np.bincount(source_indices, minlength=len(source_names))[calibration_sources]
    if not np.all(counts == PAIRS_PER_SOURCE):
        raise ValueError("every calibration source must contribute exactly eight pairs")
    rows = panel.pair_row.numpy()
    if len(np.unique(rows)) != len(rows):
        raise ValueError("materialized panel contains duplicate pair rows")
    active_pairs = int(
        sum(len(table.source_rows(int(source_index))) for source_index in calibration_sources)
    )
    tile_universe = CALIBRATION_SOURCE_COUNT * 24 * 24
    return {
        "source_count": CALIBRATION_SOURCE_COUNT,
        "pair_count": len(panel),
        "pairs_per_source": PAIRS_PER_SOURCE,
        "confidence_floor": table.min_confidence,
        "evaluated_tile_coverage": len(panel) / tile_universe,
        "active_pair_count_on_calibration_sources": active_pairs,
        "active_pair_coverage_on_calibration_sources": active_pairs / tile_universe,
        "evaluated_fraction_of_active_pairs": len(panel) / active_pairs,
        **_panel_hashes(panel, source_names),
    }


def _evaluation_dict(
    prediction: np.ndarray,
    target: np.ndarray,
    source_indices: np.ndarray,
) -> dict:
    evaluation = evaluate_real_pairs(
        prediction,
        target,
        source_indices,
        source_count=EXPECTED_VALIDATION_SOURCES,
    )
    if evaluation.source_count != CALIBRATION_SOURCE_COUNT:
        raise RuntimeError("metric evaluation did not cover exactly 257 clean calibration sources")
    return {
        "pair_count": evaluation.pair_count,
        "source_count": evaluation.source_count,
        "micro": evaluation.micro_metrics,
        "source_macro": evaluation.macro_metrics,
    }


def _bootstrap_dict(
    candidate: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    source_indices: np.ndarray,
    resamples: int,
) -> dict:
    return asdict(
        paired_source_bootstrap_delta(
            candidate,
            baseline,
            target,
            source_indices,
            metric="tile_ssim",
            source_count=EXPECTED_VALIDATION_SOURCES,
            resamples=resamples,
            seed=PROTOCOL_SEED,
        )
    )


def assess_prefinetune_diagnostic(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    bootstraps: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    max_legacy_ssim_deficit: float = MAX_LEGACY_SSIM_DEFICIT,
) -> dict:
    """Return a bounded go/hold diagnostic without authorizing or running training.

    The synthetic EMA must beat raw and NLM with a positive lower bootstrap
    bound on both confidence panels.  It may trail the legacy network by at
    most the configured macro-SSIM spending bound, and the paired source
    bootstrap lower bound must remain above the same non-inferiority margin.
    This is a headroom/quota decision only.  It is not a promotion decision:
    any fine-tuned checkpoint must later satisfy the stricter frozen gate and
    beat the legacy model.
    """
    if not math.isfinite(max_legacy_ssim_deficit) or max_legacy_ssim_deficit < 0:
        raise ValueError("max_legacy_ssim_deficit must be finite and non-negative")

    checks: dict[str, bool] = {}
    deltas: dict[str, dict[str, float]] = {}
    for panel_name in ("primary", "sensitivity"):
        panel_metrics = metrics[panel_name]
        panel_bootstrap = bootstraps[panel_name]
        candidate_ssim = float(panel_metrics["synthetic_ema"]["source_macro"]["tile_ssim"])
        deltas[panel_name] = {}
        for baseline_name in ("raw", "opencv_nlm", "legacy_q90"):
            baseline_ssim = float(
                panel_metrics[baseline_name]["source_macro"]["tile_ssim"]
            )
            delta = candidate_ssim - baseline_ssim
            deltas[panel_name][f"candidate_minus_{baseline_name}"] = delta
            bootstrap = panel_bootstrap[f"candidate_minus_{baseline_name}"]
            reported_delta = float(bootstrap["candidate_minus_baseline"])
            if not math.isclose(delta, reported_delta, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("metric and bootstrap candidate deltas disagree")
            lower = float(bootstrap["lower"])
            upper = float(bootstrap["upper"])
            if not all(math.isfinite(value) for value in (reported_delta, lower, upper)):
                raise ValueError("bootstrap summary contains a non-finite value")
            if lower > reported_delta or reported_delta > upper:
                raise ValueError("bootstrap interval does not contain its reported delta")

        for baseline_name in ("raw", "opencv_nlm"):
            lower = float(
                panel_bootstrap[f"candidate_minus_{baseline_name}"]["lower"]
            )
            checks[f"{panel_name}_beats_{baseline_name}_bootstrap_lower_positive"] = lower > 0.0

        legacy_delta = deltas[panel_name]["candidate_minus_legacy_q90"]
        legacy_lower = float(panel_bootstrap["candidate_minus_legacy_q90"]["lower"])
        checks[f"{panel_name}_legacy_deficit_at_most_{max_legacy_ssim_deficit:.3f}"] = (
            legacy_delta >= -max_legacy_ssim_deficit
        )
        checks[
            f"{panel_name}_legacy_noninferiority_ci_lower_at_least_minus_"
            f"{max_legacy_ssim_deficit:.3f}"
        ] = legacy_lower >= -max_legacy_ssim_deficit

    failed = [name for name, passed in checks.items() if not passed]
    proceed = not failed
    return {
        "decision": (
            "proceed_to_bounded_real_pair_finetune"
            if proceed
            else "hold_do_not_start_real_pair_finetune"
        ),
        "proceed_to_finetune": proceed,
        "diagnostic_only": True,
        "launches_training": False,
        "decision_kind": "fine_tune_spending_and_headroom",
        "not_a_model_promotion_decision": True,
        "max_legacy_ssim_deficit": max_legacy_ssim_deficit,
        "checks": checks,
        "failed_checks": failed,
        "source_macro_ssim_deltas": deltas,
        "scope": (
            "257 clean calibration sources only; 93 quarantined sources excluded; "
            "sealed 350-source gate integrity-hashed only and never evaluated"
        ),
    }


def _atomic_json_dump(payload: dict, output: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, output)


def run_prefinetune_benchmark(config: PreFineTuneBenchmarkConfig) -> dict:
    """Run the frozen benchmark on CPU and write one provenance-rich JSON report."""
    validate_prefinetune_config(config)
    if cv2.__version__ != config.expected_opencv_version:
        raise RuntimeError(
            "OpenCV version mismatch for fixed NLM baseline: "
            f"expected {config.expected_opencv_version}, got {cv2.__version__}"
        )
    torch.set_num_threads(config.torch_threads)

    code_sha256 = prefinetune_benchmark_code_fingerprint()
    if code_sha256 != config.expected_code_sha256:
        raise ValueError("benchmark code SHA256 does not match the pinned digest")

    manifest_path = Path(config.manifest)
    val_pairs_path = Path(config.val_pairs)
    init_path = Path(config.init_checkpoint)
    legacy_path = Path(config.legacy_checkpoint)
    quarantine_path = Path(config.quarantine_artifact)
    manifest_sha256 = sha256_file(manifest_path)
    val_pairs_sha256 = sha256_file(val_pairs_path)
    init_sha256 = sha256_file(init_path)
    legacy_sha256 = sha256_file(legacy_path)
    quarantine_sha256 = sha256_file(quarantine_path)
    expected_inputs = {
        "manifest": (manifest_sha256, config.expected_manifest_sha256),
        "val_pairs": (val_pairs_sha256, config.expected_val_pairs_sha256),
        "init_checkpoint": (init_sha256, config.expected_init_checkpoint_sha256),
        "legacy_checkpoint": (legacy_sha256, config.expected_legacy_checkpoint_sha256),
        "validation_quarantine": (
            quarantine_sha256,
            config.expected_quarantine_sha256,
        ),
    }
    mismatches = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in expected_inputs.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"pinned input SHA256 mismatch: {mismatches}")

    manifest = load_manifest(manifest_path)
    root = Path(config.data_root)
    init_checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    if not isinstance(init_checkpoint, dict):
        raise ValueError("initial checkpoint must contain a dictionary")
    if init_checkpoint.get("schema_version") != 2:
        raise ValueError("initial checkpoint must use synthetic schema_version=2")
    if init_checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("initial checkpoint manifest SHA256 mismatch")
    if init_checkpoint.get("source_code_sha256") != source_code_fingerprint():
        raise ValueError("initial checkpoint synthetic source-code SHA256 mismatch")
    initial_config = init_checkpoint.get("config")
    if not isinstance(initial_config, dict):
        raise ValueError("initial checkpoint config is missing or malformed")
    initial_validation_count = int(initial_config.get("val_images", 24))
    if not 1 <= initial_validation_count <= EXPECTED_VALIDATION_SOURCES:
        raise ValueError("initial checkpoint validation-image count is invalid")
    synthetic_validation_names = manifest["splits"]["val"][:initial_validation_count]
    quarantine, loaded_quarantine_sha256 = load_validation_quarantine(
        quarantine_path,
        config.expected_quarantine_sha256,
        manifest_sha256=manifest_sha256,
        manifest_validation_names=manifest["splits"]["val"],
        expected_legacy_checkpoint_sha256=legacy_sha256,
        expected_synthetic_validation_names=synthetic_validation_names,
        gate_source_count=config.gate_source_count,
        seed=PROTOCOL_SEED,
    )
    if loaded_quarantine_sha256 != quarantine_sha256:
        raise RuntimeError("quarantine artifact changed during strict loading")

    primary_table = RealPairTable.load(
        val_pairs_path,
        manifest_path=manifest_path,
        data_root=root,
        expected_split="val",
        min_confidence=PRIMARY_CONFIDENCE,
    )
    sensitivity_table = RealPairTable.load(
        val_pairs_path,
        manifest_path=manifest_path,
        data_root=root,
        expected_split="val",
        min_confidence=SENSITIVITY_CONFIDENCE,
    )
    if primary_table.npz_sha256 != val_pairs_sha256:
        raise ValueError("validated pair table digest changed during loading")
    if primary_table.source_names != sensitivity_table.source_names:
        raise ValueError("primary and sensitivity pair tables disagree on source names")
    if set(primary_table.source_names) != set(manifest["splits"]["val"]):
        raise ValueError("real-pair artifact must contain exactly the manifest validation sources")

    # This deliberately decodes all 700 validation input/target images only to
    # verify the pinned integrity digest.  No resulting image or tile is
    # retained, selected, passed to a model, or passed to a metric.
    validation_pixels_sha256 = fine_tune_pixel_fingerprints(
        root,
        (),
        (),
        primary_table.source_names,
    )["validation_pixels_sha256"]
    if validation_pixels_sha256 != config.expected_validation_pixels_sha256:
        raise ValueError("full validation decoded-pixel SHA256 does not match the pinned digest")

    calibration_sources, sealed_gate_sources = select_frozen_calibration_sources(
        primary_table.source_names,
        primary_table.active_source_indices,
        sensitivity_table.active_source_indices,
        tuple(quarantine["quarantine_names"]),
        gate_source_count=config.gate_source_count,
    )
    quarantine_names = tuple(quarantine["quarantine_names"])
    calibration_names = tuple(
        sorted(primary_table.source_names[int(index)] for index in calibration_sources)
    )
    sealed_gate_names = tuple(
        sorted(primary_table.source_names[int(index)] for index in sealed_gate_sources)
    )
    clean_eligible_names = tuple(sorted(set(calibration_names) | set(sealed_gate_names)))
    split_name_sha256 = {
        "quarantine": source_name_list_sha256(quarantine_names),
        "eligible_after_quarantine": source_name_list_sha256(clean_eligible_names),
        "calibration": source_name_list_sha256(calibration_names),
        "frozen_gate": source_name_list_sha256(sealed_gate_names),
    }
    for name, actual in split_name_sha256.items():
        expected = quarantine["name_sha256"][name]
        if actual != expected:
            raise RuntimeError(
                f"quarantine-aware source-name hash mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )
    source_index_by_name = {
        name: index for index, name in enumerate(primary_table.source_names)
    }
    quarantine_source_indices = sorted(
        source_index_by_name[name] for name in quarantine_names
    )
    if (
        set(quarantine_source_indices) & set(calibration_sources.tolist())
        or set(quarantine_source_indices) & set(sealed_gate_sources.tolist())
    ):
        raise RuntimeError("quarantine leaked through name-to-index mapping")
    # Only calibration_sources are passed to a sampler.  sealed_gate_sources
    # remain indices/names and are never materialized below.
    primary_panel = RealPairSampler(
        primary_table, seed=PROTOCOL_SEED, cache_size=16
    ).materialize_validation(
        source_indices=calibration_sources,
        pairs_per_source=PAIRS_PER_SOURCE,
        seed=PROTOCOL_SEED,
    )
    sensitivity_panel = RealPairSampler(
        sensitivity_table, seed=PROTOCOL_SEED + 1, cache_size=16
    ).materialize_validation(
        source_indices=calibration_sources,
        pairs_per_source=PAIRS_PER_SOURCE,
        seed=PROTOCOL_SEED + 1,
    )
    panel_summary = {
        "primary": _validate_panel(
            primary_panel, calibration_sources, primary_table.source_names, primary_table
        ),
        "sensitivity": _validate_panel(
            sensitivity_panel,
            calibration_sources,
            sensitivity_table.source_names,
            sensitivity_table,
        ),
    }
    primary_rows = primary_panel.pair_row.numpy()
    sensitivity_rows = sensitivity_panel.pair_row.numpy()
    overlap = int(len(np.intersect1d(primary_rows, sensitivity_rows, assume_unique=True)))
    panel_summary["primary_sensitivity_pair_overlap"] = {
        "pair_count": overlap,
        "fraction_of_primary": overlap / len(primary_panel),
        "fraction_of_sensitivity": overlap / len(sensitivity_panel),
    }

    del init_checkpoint

    candidate_model, candidate_device, candidate_metadata = load_restorer(
        init_path,
        device="cpu",
        state="ema",
        allow_unpromoted=False,
    )
    if candidate_device.type != "cpu" or any(
        parameter.device.type != "cpu" for parameter in candidate_model.parameters()
    ):
        raise RuntimeError("synthetic EMA benchmark must remain CPU-only")
    required_candidate_metadata = {
        "checkpoint_sha256": init_sha256,
        "schema_version": 2,
        "manifest_sha256": manifest_sha256,
        "source_code_sha256": source_code_fingerprint(),
        "model_name": "tile-naf",
        "state": "ema",
        "device": "cpu",
    }
    for name, expected in required_candidate_metadata.items():
        if candidate_metadata.get(name) != expected:
            raise ValueError(
                f"synthetic checkpoint metadata {name!r} mismatch: "
                f"expected {expected!r}, got {candidate_metadata.get(name)!r}"
            )

    legacy_model, legacy_device, legacy_metadata = load_legacy_tile_restorer(
        legacy_path,
        expected_sha256=config.expected_legacy_checkpoint_sha256,
        device="cpu",
    )
    if legacy_device.type != "cpu":
        raise RuntimeError("legacy benchmark must remain CPU-only")

    metrics: dict[str, dict] = {}
    bootstraps: dict[str, dict] = {}
    for panel_name, panel in (
        ("primary", primary_panel),
        ("sensitivity", sensitivity_panel),
    ):
        raw = _batch_uint8(panel.corrupt)
        target = _batch_uint8(panel.clean)
        source_indices = panel.source_index.numpy()
        candidate = restore_tiles_uint8(
            candidate_model,
            raw,
            candidate_device,
            batch_size=config.batch_size,
        )
        opencv_nlm = classical_nlm_tiles_uint8(raw)
        legacy = predict_legacy_tiles_uint8(
            legacy_model,
            raw,
            legacy_device,
            batch_size=config.batch_size,
        )
        predictions = {
            "raw": raw,
            "opencv_nlm": opencv_nlm,
            "legacy_q90": legacy,
            "synthetic_ema": candidate,
        }
        metrics[panel_name] = {
            name: _evaluation_dict(prediction, target, source_indices)
            for name, prediction in predictions.items()
        }
        bootstraps[panel_name] = {
            f"candidate_minus_{baseline_name}": _bootstrap_dict(
                candidate,
                predictions[baseline_name],
                target,
                source_indices,
                config.bootstrap_resamples,
            )
            for baseline_name in ("raw", "opencv_nlm", "legacy_q90")
        }

    diagnostic = assess_prefinetune_diagnostic(
        metrics,
        bootstraps,
        max_legacy_ssim_deficit=config.max_legacy_ssim_deficit,
    )
    report = {
        "schema_version": 1,
        "kind": "cpu_prefinetune_quarantine_aware_calibration_benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "seed": PROTOCOL_SEED,
            "validation_source_count": EXPECTED_VALIDATION_SOURCES,
            "quarantine_source_count": QUARANTINE_SOURCE_COUNT,
            "clean_eligible_source_count": CLEAN_ELIGIBLE_SOURCE_COUNT,
            "calibration_source_count": CALIBRATION_SOURCE_COUNT,
            "sealed_gate_source_count": SEALED_GATE_SOURCE_COUNT,
            "pairs_per_source": PAIRS_PER_SOURCE,
            "primary_confidence": PRIMARY_CONFIDENCE,
            "sensitivity_confidence": SENSITIVITY_CONFIDENCE,
            "bootstrap_resamples": config.bootstrap_resamples,
            "bootstrap_unit": "source image",
            "metric_aggregation": "equal-weight source macro",
            "device": "cpu",
            "all_700_validation_pixels_decoded_for_integrity_hash": True,
            "gate_integrity_hashed_only": True,
            "sealed_gate_metric_panel_materialized": False,
            "sealed_gate_tiles_passed_to_model_or_metrics": False,
            "quarantine_tiles_passed_to_model_or_metrics": False,
            "training_launched": False,
        },
        "inputs": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "val_pairs": str(val_pairs_path.resolve()),
            "val_pairs_sha256": val_pairs_sha256,
            "init_checkpoint": str(init_path.resolve()),
            "init_checkpoint_sha256": init_sha256,
            "legacy_checkpoint": str(legacy_path.resolve()),
            "legacy_checkpoint_sha256": legacy_sha256,
            "validation_quarantine": str(quarantine_path.resolve()),
            "validation_quarantine_sha256": quarantine_sha256,
            "full_validation_decoded_pixels_sha256": validation_pixels_sha256,
            "benchmark_code_sha256": code_sha256,
        },
        "runtime": {
            **runtime_versions(),
            "opencv": cv2.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
        },
        "source_split": {
            "mapping": "source name to real-pair-table source index",
            "source_index_by_name": source_index_by_name,
            "quarantine_source_count": len(quarantine_names),
            "quarantine_source_indices": quarantine_source_indices,
            "quarantine_source_names": list(quarantine_names),
            "quarantine_source_names_sha256": split_name_sha256["quarantine"],
            "clean_eligible_source_count": len(clean_eligible_names),
            "clean_eligible_source_names_sha256": split_name_sha256[
                "eligible_after_quarantine"
            ],
            "calibration_source_indices": calibration_sources.tolist(),
            "calibration_source_names": list(calibration_names),
            "calibration_source_names_sha256": split_name_sha256["calibration"],
            "sealed_gate_source_count": len(sealed_gate_names),
            "sealed_gate_source_names": list(sealed_gate_names),
            "sealed_gate_source_names_sha256": split_name_sha256["frozen_gate"],
            "quarantine_excluded_from_calibration_and_gate": True,
            "three_way_disjoint": True,
            "complete_93_257_350_partition": True,
        },
        "panels": panel_summary,
        "baselines": {
            "raw": {"kind": "identity_copy"},
            "opencv_nlm": {
                "kind": "cv2.fastNlMeansDenoisingColored_per_tile",
                "opencv_version": cv2.__version__,
                "parameters": NLM_PARAMETERS,
            },
            "legacy_q90": {
                **legacy_metadata,
                "metric_scope": "257 clean calibration sources only",
                "quarantined_legacy_seen_sources_excluded": True,
            },
            "synthetic_ema": candidate_metadata,
        },
        "metrics": metrics,
        "paired_source_bootstrap": bootstraps,
        "diagnostic": diagnostic,
    }
    _atomic_json_dump(report, Path(config.output))
    return report


__all__ = [
    "CALIBRATION_SOURCE_COUNT",
    "CLEAN_ELIGIBLE_SOURCE_COUNT",
    "MAX_LEGACY_SSIM_DEFICIT",
    "NLM_PARAMETERS",
    "PAIRS_PER_SOURCE",
    "PREFINETUNE_BENCHMARK_CODE_FILES",
    "PRIMARY_CONFIDENCE",
    "PROTOCOL_SEED",
    "QUARANTINE_SOURCE_COUNT",
    "PreFineTuneBenchmarkConfig",
    "SENSITIVITY_CONFIDENCE",
    "SEALED_GATE_SOURCE_COUNT",
    "assess_prefinetune_diagnostic",
    "classical_nlm_tiles_uint8",
    "prefinetune_benchmark_code_fingerprint",
    "run_prefinetune_benchmark",
    "select_frozen_calibration_sources",
    "validate_prefinetune_config",
]
