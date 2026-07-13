"""One-shot CPU audit of a model frozen before the sealed real-pair gate opens."""

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
from .legacy_baseline import load_legacy_tile_restorer, predict_legacy_tiles_uint8
from .prefinetune_benchmark import (
    _batch_uint8,
    _panel_hashes,
    classical_nlm_tiles_uint8,
    select_frozen_calibration_sources,
)
from .real_pairs import RealPairBatch, RealPairSampler, RealPairTable
from .real_training import load_validation_quarantine, source_name_list_sha256
from .real_validation import evaluate_real_pairs, paired_source_bootstrap_delta
from .training import load_manifest, runtime_versions, source_code_fingerprint


PROTOCOL_SEED = 20260710
PRIMARY_PANEL_SEED = 20260712
SENSITIVITY_PANEL_SEED = 20260713
VALIDATION_SOURCE_COUNT = 700
QUARANTINE_SOURCE_COUNT = 93
CALIBRATION_SOURCE_COUNT = 257
GATE_SOURCE_COUNT = 350
PAIRS_PER_SOURCE = 8
PRIMARY_CONFIDENCE = 1.5
SENSITIVITY_CONFIDENCE = 1.0
SELECTED_CHECKPOINT_LOGICAL_PATH = (
    "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
)

FINAL_GATE_CODE_FILES = (
    "__init__.py",
    "degradation.py",
    "final_gate_audit.py",
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
class FinalGateAuditConfig:
    data_root: str
    manifest: str
    val_pairs: str
    checkpoint: str
    legacy_checkpoint: str
    quarantine_artifact: str
    selection_manifest: str
    output: str
    expected_manifest_sha256: str
    expected_val_pairs_sha256: str
    expected_checkpoint_sha256: str
    expected_legacy_checkpoint_sha256: str
    expected_quarantine_sha256: str
    expected_selection_manifest_sha256: str
    expected_code_sha256: str
    expected_opencv_version: str
    batch_size: int = 128
    bootstrap_resamples: int = 5000
    torch_threads: int = 4


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def validate_final_gate_config(config: FinalGateAuditConfig) -> None:
    for name in (
        "expected_manifest_sha256",
        "expected_val_pairs_sha256",
        "expected_checkpoint_sha256",
        "expected_legacy_checkpoint_sha256",
        "expected_quarantine_sha256",
        "expected_selection_manifest_sha256",
        "expected_code_sha256",
    ):
        _require_sha256(name, getattr(config, name))
    if re.fullmatch(r"\d+(?:\.\d+){1,3}", config.expected_opencv_version) is None:
        raise ValueError("expected_opencv_version must be dotted numeric")
    for name in ("batch_size", "bootstrap_resamples", "torch_threads"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.bootstrap_resamples != 5000:
        raise ValueError("bootstrap_resamples must remain exactly 5000")
    if config.torch_threads > 16:
        raise ValueError("torch_threads must not exceed 16")


def final_gate_code_fingerprint(package_dir: str | Path | None = None) -> str:
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in FINAL_GATE_CODE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"final-gate code file is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_frozen_selection(config: FinalGateAuditConfig) -> tuple[dict, str]:
    path = Path(config.selection_manifest).resolve(strict=True)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != config.expected_selection_manifest_sha256:
        raise ValueError("selection manifest SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("kind") != "denoise_v2_selected_model":
        raise ValueError("selection manifest schema/kind mismatch")
    try:
        selection_created = datetime.fromisoformat(str(payload["created_utc"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("selection manifest created_utc is invalid") from error
    if selection_created.tzinfo is None or selection_created >= datetime.now(timezone.utc):
        raise ValueError("selection manifest must predate the final-gate audit")
    selected = payload.get("selected")
    fine_tune = payload.get("bounded_real_pair_fine_tune")
    policy = payload.get("policy")
    if not all(isinstance(value, dict) for value in (selected, fine_tune, policy)):
        raise ValueError("selection manifest is missing decision sections")
    expected_selected = {
        "checkpoint_sha256": config.expected_checkpoint_sha256,
        "model_name": "tile-naf",
        "state": "ema",
        "step": 50000,
        "tile_order_preserved": True,
    }
    mismatches = {
        name: {"actual": selected.get(name), "expected": expected}
        for name, expected in expected_selected.items()
        if selected.get(name) != expected
    }
    if selected.get("checkpoint") != SELECTED_CHECKPOINT_LOGICAL_PATH:
        mismatches["checkpoint"] = {
            "actual": selected.get("checkpoint"),
            "expected": SELECTED_CHECKPOINT_LOGICAL_PATH,
        }
    if fine_tune.get("frozen_gate_opened") is not False:
        mismatches["bounded_real_pair_fine_tune.frozen_gate_opened"] = {
            "actual": fine_tune.get("frozen_gate_opened"),
            "expected": False,
        }
    if fine_tune.get("rolled_back") is not True:
        mismatches["bounded_real_pair_fine_tune.rolled_back"] = {
            "actual": fine_tune.get("rolled_back"),
            "expected": True,
        }
    if policy.get("promotion_threshold_relaxed_after_results") is not False:
        mismatches["policy.promotion_threshold_relaxed_after_results"] = {
            "actual": policy.get("promotion_threshold_relaxed_after_results"),
            "expected": False,
        }
    if mismatches:
        raise ValueError(f"frozen selection contract mismatch: {mismatches}")
    return payload, actual_sha256


def _validate_gate_panel(
    panel: RealPairBatch,
    gate_sources: np.ndarray,
    source_names: tuple[str, ...],
    table: RealPairTable,
) -> dict:
    expected_pairs = GATE_SOURCE_COUNT * PAIRS_PER_SOURCE
    if len(panel) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} final-gate pairs, got {len(panel)}")
    indices = panel.source_index.numpy()
    if not np.array_equal(np.unique(indices), gate_sources):
        raise ValueError("final-gate panel contains a non-gate source")
    counts = np.bincount(indices, minlength=len(source_names))[gate_sources]
    if not np.all(counts == PAIRS_PER_SOURCE):
        raise ValueError("every frozen-gate source must contribute exactly eight pairs")
    rows = panel.pair_row.numpy()
    if len(np.unique(rows)) != len(rows):
        raise ValueError("final-gate panel contains duplicate pair rows")
    active_pairs = int(sum(len(table.source_rows(int(index))) for index in gate_sources))
    return {
        "source_count": GATE_SOURCE_COUNT,
        "pair_count": len(panel),
        "pairs_per_source": PAIRS_PER_SOURCE,
        "confidence_floor": table.min_confidence,
        "active_pair_count_on_gate_sources": active_pairs,
        "evaluated_fraction_of_active_pairs": len(panel) / active_pairs,
        **_panel_hashes(panel, source_names),
    }


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _evaluation(prediction: np.ndarray, target: np.ndarray, indices: np.ndarray) -> dict:
    result = evaluate_real_pairs(
        prediction,
        target,
        indices,
        source_count=VALIDATION_SOURCE_COUNT,
    )
    if result.source_count != GATE_SOURCE_COUNT:
        raise RuntimeError("final-gate evaluation did not cover exactly 350 sources")
    return {
        "pair_count": result.pair_count,
        "source_count": result.source_count,
        "micro": result.micro_metrics,
        "source_macro": result.macro_metrics,
    }


def _bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict:
    return asdict(
        paired_source_bootstrap_delta(
            candidate,
            baseline,
            target,
            indices,
            metric="tile_ssim",
            source_count=VALIDATION_SOURCE_COUNT,
            resamples=resamples,
            seed=seed,
        )
    )


def assess_final_gate(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    bootstraps: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict:
    checks: dict[str, bool] = {}
    deltas: dict[str, dict[str, float]] = {}
    for panel in ("primary", "sensitivity"):
        candidate = float(metrics[panel]["selected_ema"]["source_macro"]["tile_ssim"])
        deltas[panel] = {}
        for baseline in ("raw", "opencv_nlm", "legacy_q90"):
            baseline_value = float(metrics[panel][baseline]["source_macro"]["tile_ssim"])
            delta = candidate - baseline_value
            summary = bootstraps[panel][f"selected_minus_{baseline}"]
            reported = float(summary["candidate_minus_baseline"])
            lower = float(summary["lower"])
            upper = float(summary["upper"])
            if not all(math.isfinite(value) for value in (delta, reported, lower, upper)):
                raise ValueError("final-gate metric/bootstrap contains a non-finite value")
            if not math.isclose(delta, reported, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("final-gate metric/bootstrap deltas disagree")
            if lower > reported or reported > upper:
                raise ValueError("final-gate bootstrap interval excludes its delta")
            deltas[panel][f"selected_minus_{baseline}"] = delta
            checks[f"{panel}_beats_{baseline}_bootstrap_lower_positive"] = lower > 0.0
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "decision": "pass" if not failed else "fail",
        "passes_final_gate": not failed,
        "checks": checks,
        "failed_checks": failed,
        "source_macro_ssim_deltas": deltas,
        "model_choice_was_frozen_before_gate": True,
        "model_choice_changed_after_gate": False,
        "training_or_tuning_launched": False,
        "scope": "350 frozen-gate sources only; 257 calibration and 93 quarantine sources excluded",
    }


def _atomic_json_dump(payload: dict, output: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, output)


def run_final_gate_audit(config: FinalGateAuditConfig) -> dict:
    validate_final_gate_config(config)
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise RuntimeError("final gate requires a CPU-only PyTorch runtime")
    if cv2.__version__ != config.expected_opencv_version:
        raise RuntimeError("OpenCV version mismatch for fixed NLM baseline")
    torch.set_num_threads(config.torch_threads)
    code_sha256 = final_gate_code_fingerprint()
    if code_sha256 != config.expected_code_sha256:
        raise ValueError("final-gate code SHA256 mismatch")
    output = Path(config.output).expanduser()
    data_root = Path(config.data_root).expanduser().resolve(strict=True)
    output_resolved = output.resolve(strict=False)
    try:
        output_resolved.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError("final-gate output must stay outside the puzzle data root")
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"final-gate output already exists or is a symlink: {output}")

    selection, selection_sha256 = _load_frozen_selection(config)
    manifest_path = Path(config.manifest).resolve(strict=True)
    val_pairs_path = Path(config.val_pairs).resolve(strict=True)
    checkpoint_path = Path(config.checkpoint).resolve(strict=True)
    legacy_path = Path(config.legacy_checkpoint).resolve(strict=True)
    quarantine_path = Path(config.quarantine_artifact).resolve(strict=True)
    actual_inputs = {
        "manifest": sha256_file(manifest_path),
        "validation_pairs": sha256_file(val_pairs_path),
        "selected_checkpoint": sha256_file(checkpoint_path),
        "legacy_checkpoint": sha256_file(legacy_path),
        "validation_quarantine": sha256_file(quarantine_path),
        "selection_manifest": selection_sha256,
    }
    expected_inputs = {
        "manifest": config.expected_manifest_sha256,
        "validation_pairs": config.expected_val_pairs_sha256,
        "selected_checkpoint": config.expected_checkpoint_sha256,
        "legacy_checkpoint": config.expected_legacy_checkpoint_sha256,
        "validation_quarantine": config.expected_quarantine_sha256,
        "selection_manifest": config.expected_selection_manifest_sha256,
    }
    mismatches = {
        name: {"actual": actual_inputs[name], "expected": expected}
        for name, expected in expected_inputs.items()
        if actual_inputs[name] != expected
    }
    if mismatches:
        raise ValueError(f"final-gate pinned input mismatch: {mismatches}")

    manifest = load_manifest(manifest_path)
    validation_names = manifest["splits"]["val"]
    if len(validation_names) != VALIDATION_SOURCE_COUNT:
        raise ValueError("manifest validation split must contain exactly 700 sources")
    quarantine, loaded_quarantine_sha256 = load_validation_quarantine(
        quarantine_path,
        config.expected_quarantine_sha256,
        manifest_sha256=actual_inputs["manifest"],
        manifest_validation_names=validation_names,
        expected_legacy_checkpoint_sha256=config.expected_legacy_checkpoint_sha256,
        expected_synthetic_validation_names=validation_names[:24],
        gate_source_count=GATE_SOURCE_COUNT,
        seed=PROTOCOL_SEED,
    )
    if loaded_quarantine_sha256 != actual_inputs["validation_quarantine"]:
        raise RuntimeError("quarantine artifact changed during strict loading")
    primary_table = RealPairTable.load(
        val_pairs_path,
        manifest_path=manifest_path,
        data_root=data_root,
        expected_split="val",
        min_confidence=PRIMARY_CONFIDENCE,
    )
    sensitivity_table = RealPairTable.load(
        val_pairs_path,
        manifest_path=manifest_path,
        data_root=data_root,
        expected_split="val",
        min_confidence=SENSITIVITY_CONFIDENCE,
    )
    if primary_table.npz_sha256 != actual_inputs["validation_pairs"]:
        raise RuntimeError("validation-pair artifact changed during loading")
    if primary_table.source_names != sensitivity_table.source_names:
        raise ValueError("primary and sensitivity source names differ")
    calibration_sources, gate_sources = select_frozen_calibration_sources(
        primary_table.source_names,
        primary_table.active_source_indices,
        sensitivity_table.active_source_indices,
        tuple(quarantine["quarantine_names"]),
        gate_source_count=GATE_SOURCE_COUNT,
    )
    gate_names = tuple(sorted(primary_table.source_names[int(index)] for index in gate_sources))
    calibration_names = tuple(
        sorted(primary_table.source_names[int(index)] for index in calibration_sources)
    )
    if source_name_list_sha256(gate_names) != quarantine["name_sha256"]["frozen_gate"]:
        raise RuntimeError("frozen-gate source-name hash mismatch")
    if source_name_list_sha256(calibration_names) != quarantine["name_sha256"]["calibration"]:
        raise RuntimeError("calibration source-name hash mismatch")

    primary_panel = RealPairSampler(
        primary_table, seed=PRIMARY_PANEL_SEED, cache_size=16
    ).materialize_validation(
        source_indices=gate_sources,
        pairs_per_source=PAIRS_PER_SOURCE,
        seed=PRIMARY_PANEL_SEED,
    )
    sensitivity_panel = RealPairSampler(
        sensitivity_table, seed=SENSITIVITY_PANEL_SEED, cache_size=16
    ).materialize_validation(
        source_indices=gate_sources,
        pairs_per_source=PAIRS_PER_SOURCE,
        seed=SENSITIVITY_PANEL_SEED,
    )
    panels = {
        "primary": _validate_gate_panel(
            primary_panel, gate_sources, primary_table.source_names, primary_table
        ),
        "sensitivity": _validate_gate_panel(
            sensitivity_panel, gate_sources, sensitivity_table.source_names, sensitivity_table
        ),
    }
    overlap = int(
        len(
            np.intersect1d(
                primary_panel.pair_row.numpy(),
                sensitivity_panel.pair_row.numpy(),
                assume_unique=True,
            )
        )
    )
    panels["primary_sensitivity_pair_overlap"] = {
        "pair_count": overlap,
        "fraction_of_primary": overlap / len(primary_panel),
        "fraction_of_sensitivity": overlap / len(sensitivity_panel),
    }

    selected_model, selected_device, selected_metadata = load_restorer(
        checkpoint_path,
        device="cpu",
        state="ema",
        allow_unpromoted=False,
    )
    if selected_device.type != "cpu" or selected_metadata["checkpoint_sha256"] != actual_inputs[
        "selected_checkpoint"
    ]:
        raise RuntimeError("selected checkpoint did not load on the pinned CPU path")
    expected_selected_metadata = {
        "model_name": "tile-naf",
        "state": "ema",
        "step": 50000,
        "schema_version": 2,
        "manifest_sha256": actual_inputs["manifest"],
        "source_code_sha256": source_code_fingerprint(),
    }
    selected_metadata_mismatches = {
        name: {"actual": selected_metadata.get(name), "expected": expected}
        for name, expected in expected_selected_metadata.items()
        if selected_metadata.get(name) != expected
    }
    if selected_metadata_mismatches:
        raise RuntimeError(
            f"selected checkpoint metadata mismatch: {selected_metadata_mismatches}"
        )
    legacy_model, legacy_device, legacy_metadata = load_legacy_tile_restorer(
        legacy_path,
        expected_sha256=config.expected_legacy_checkpoint_sha256,
        device="cpu",
    )

    metrics: dict[str, dict] = {}
    bootstraps: dict[str, dict] = {}
    prediction_hashes: dict[str, dict] = {}
    for panel_name, panel, bootstrap_seed in (
        ("primary", primary_panel, PRIMARY_PANEL_SEED),
        ("sensitivity", sensitivity_panel, SENSITIVITY_PANEL_SEED),
    ):
        raw = _batch_uint8(panel.corrupt)
        target = _batch_uint8(panel.clean)
        indices = panel.source_index.numpy()
        selected_prediction = restore_tiles_uint8(
            selected_model,
            raw,
            selected_device,
            batch_size=config.batch_size,
        )
        nlm_prediction = classical_nlm_tiles_uint8(raw)
        legacy_prediction = predict_legacy_tiles_uint8(
            legacy_model,
            raw,
            legacy_device,
            batch_size=config.batch_size,
        )
        predictions = {
            "raw": raw,
            "opencv_nlm": nlm_prediction,
            "legacy_q90": legacy_prediction,
            "selected_ema": selected_prediction,
        }
        metrics[panel_name] = {
            name: _evaluation(prediction, target, indices)
            for name, prediction in predictions.items()
        }
        bootstraps[panel_name] = {
            f"selected_minus_{name}": _bootstrap(
                selected_prediction,
                prediction,
                target,
                indices,
                seed=bootstrap_seed,
                resamples=config.bootstrap_resamples,
            )
            for name, prediction in predictions.items()
            if name != "selected_ema"
        }
        prediction_hashes[panel_name] = {
            name: _array_sha256(prediction) for name, prediction in predictions.items()
        }
        prediction_hashes[panel_name]["target"] = _array_sha256(target)

    assessment = assess_final_gate(metrics, bootstraps)
    report = {
        "schema_version": 1,
        "kind": "selected_denoiser_one_shot_frozen_gate_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "selection_manifest_was_frozen_before_gate": True,
            "selection_manifest_created_utc": selection["created_utc"],
            "training_or_tuning_launched": False,
            "model_choice_changed_after_gate": False,
            "validation_source_count": VALIDATION_SOURCE_COUNT,
            "quarantine_source_count": QUARANTINE_SOURCE_COUNT,
            "calibration_source_count": CALIBRATION_SOURCE_COUNT,
            "frozen_gate_source_count": GATE_SOURCE_COUNT,
            "pairs_per_source": PAIRS_PER_SOURCE,
            "primary_confidence": PRIMARY_CONFIDENCE,
            "sensitivity_confidence": SENSITIVITY_CONFIDENCE,
            "bootstrap_resamples": config.bootstrap_resamples,
            "bootstrap_unit": "source image",
            "metric_aggregation": "equal-weight source macro",
            "device": "cpu",
            "calibration_pixels_materialized": False,
            "quarantine_pixels_materialized": False,
            "frozen_gate_opened_once": True,
        },
        "inputs": {
            **actual_inputs,
            "final_gate_code_sha256": code_sha256,
            "selected_checkpoint_metadata": {
                name: selected_metadata.get(name)
                for name in (
                    "checkpoint_sha256",
                    "model_name",
                    "state",
                    "step",
                    "schema_version",
                    "manifest_sha256",
                    "source_code_sha256",
                )
            },
            "legacy_checkpoint_metadata": legacy_metadata,
        },
        "runtime": {
            **runtime_versions(),
            "opencv": cv2.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
        },
        "source_split": {
            "quarantine_source_names_sha256": quarantine["name_sha256"]["quarantine"],
            "calibration_source_names_sha256": quarantine["name_sha256"]["calibration"],
            "frozen_gate_source_names_sha256": quarantine["name_sha256"]["frozen_gate"],
            "three_way_disjoint": True,
            "complete_93_257_350_partition": True,
        },
        "panels": panels,
        "prediction_hashes": prediction_hashes,
        "metrics": metrics,
        "paired_source_bootstrap": bootstraps,
        "assessment": assessment,
    }
    _atomic_json_dump(report, output)
    return report


__all__ = [
    "FinalGateAuditConfig",
    "assess_final_gate",
    "final_gate_code_fingerprint",
    "run_final_gate_audit",
    "validate_final_gate_config",
]
