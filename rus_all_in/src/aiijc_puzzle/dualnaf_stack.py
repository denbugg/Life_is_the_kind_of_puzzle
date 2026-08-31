"""Fail-closed evaluation of one frozen DualNAF/NLM stacking candidate.

All calibration records used here are historically target-exposed.  This module
therefore makes no freshness claim; its phase separation only guarantees that
the current experiment commits all four target-free predictions before it
decodes targets.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.dense_safe_tail import array_sha256, paired_t_interval, safety_metrics
from aiijc_puzzle.frozen_final_evaluator import load_rgb_verified
from aiijc_puzzle.legacy_upgrade import directional_scores, layout_digest, solve_buddies
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.postassembly_harmonizer import (
    DEFAULT_LUMINANCE_GAIN_CONFIG,
    DEFAULT_SEAM_GRAPH_CONFIG,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    IMAGE_SIZE,
    assemble_tiles,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.restoration_r6 import TileAwareDualNAFNet
from aiijc_puzzle.tilewise_renderer import render_tiles_independently

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "dualnaf_stack_reused_calibration_preregistered_v1.json"
CONFIG_SHA256 = "c4cf677227da645021e4d06874a29aae4c22db82d77c1e3e8af3825f20d0405a"
MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
INPUTS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
TARGETS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "restoration-r6" / "compliant-r6-medium-train256-step2000-h10.pt"
)
CHECKPOINT_SHA256 = "331322460c8af87e5d4760b075726979f0574a23209889c1e95b6b90f2eac1a9"
LEGACY_ALL_CALIBRATION_REPORT = (
    PROJECT_ROOT / "outputs" / "legacy-upgrade" / "calibration700-champion" / "report.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dualnaf-stack" / CONFIG_SHA256

ARM_A = "A_alpha0_h20_baseline"
ARM_B = "B_alpha0_h28_denoise_control"
ARM_C = "C_alpha0p125_h20_bridge"
ARM_D = "D_alpha0p125_h28_candidate"
ARMS = (ARM_A, ARM_B, ARM_C, ARM_D)
SUPPORT_IMAGES = (
    "raw",
    "dualnaf_same_index",
    "blend_alpha0p125",
    "harmonized_alpha0",
    "harmonized_alpha0p125",
)
PANEL_KEYS = {
    "primary": "primary",
    "confirmation": "confirmation_if_and_only_if_all_primary_gates_pass",
}
COMMITMENT_SCHEMA = "aiijc-dualnaf-stack-prediction-commitment-v1"
REPORT_SCHEMA = "aiijc-dualnaf-stack-reused-calibration-report-v1"
BOOTSTRAP_REPLICATES = 20_000


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _filename_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = b"\0".join(str(record["filename"]).encode("utf-8") for record in records)
    return hashlib.sha256(payload).hexdigest()


def _write_bytes_exclusive(path: Path, contents: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
    *,
    readonly: bool = False,
) -> None:
    contents = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    _write_bytes_exclusive(path, contents, mode=0o444 if readonly else 0o644)
    if readonly:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _png_bytes(rgb: np.ndarray) -> bytes:
    array = np.asarray(rgb)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or array.dtype != np.uint8:
        raise ValueError(f"invalid RGB image: {array.shape} {array.dtype}")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def load_config() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("DualNAF stack preregistration hash mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status_at_freeze") != (
        "preregistered_before_this_experiment_decodes_any_target_pixels"
    ):
        raise ValueError("preregistration status drifted")
    if config["protocol"]["historical_exposure_audit"]["freshness_claim"] is not False:
        raise ValueError("historical exposure may not be hidden")
    if tuple(arm["name"] for arm in config["arms"]) != ARMS:
        raise ValueError("frozen arm roster drifted")
    if config["arms"][-1].get("only_promotable_candidate") is not True:
        raise ValueError("D must remain the only candidate")
    return config


def _load_manifest() -> dict[str, Any]:
    config = load_config()
    if sha256_file(MANIFEST_PATH) != config["protocol"]["manifest_sha256"]:
        raise ValueError("manifest file hash mismatch")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != (
        "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
    ):
        raise ValueError("manifest protocol drifted")
    return manifest


def panel_records(panel: str) -> tuple[Mapping[str, Any], ...]:
    if panel not in PANEL_KEYS:
        raise ValueError("panel must be primary or confirmation")
    config = load_config()
    section = config["protocol"][PANEL_KEYS[panel]]
    ranked = select_manifest_records(
        _load_manifest(),
        "calibration",
        limit=int(section["offset"]) + int(section["count"]),
        seed=EXPERIMENT_SUBSET_SEED,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
    )
    records = tuple(ranked[int(section["offset"]) :])
    if len(records) != section["count"]:
        raise ValueError("panel count drifted")
    if _filename_digest(records) != section["filename_nul_join_sha256"]:
        raise ValueError("panel digest drifted")
    return records


def audit_historical_exposure() -> dict[str, Any]:
    primary = panel_records("primary")
    confirmation = panel_records("confirmation")
    if {row["filename"] for row in primary} & {row["filename"] for row in confirmation}:
        raise ValueError("primary and confirmation overlap")
    legacy = json.loads(LEGACY_ALL_CALIBRATION_REPORT.read_text(encoding="utf-8"))
    legacy_records = {
        str(row["filename"]): (str(row["input_sha256"]), str(row["target_sha256"]))
        for row in legacy["per_board"]
    }

    def exact_count(records: Sequence[Mapping[str, Any]]) -> int:
        return sum(
            legacy_records.get(str(row["filename"]))
            == (str(row["input_sha256"]), str(row["target_sha256"]))
            for row in records
        )

    all_calibration = _load_manifest()["splits"]["calibration"]
    result = {
        "freshness_claim": False,
        "primary_confirmation_overlap": 0,
        "legacy_report_sha256": sha256_file(LEGACY_ALL_CALIBRATION_REPORT),
        "legacy_exact_matches_to_current_calibration": exact_count(all_calibration),
        "primary_historically_exposed": exact_count(primary),
        "confirmation_historically_exposed": exact_count(confirmation),
    }
    if result["legacy_exact_matches_to_current_calibration"] != 700:
        raise ValueError("legacy calibration-700 exposure proof drifted")
    if result["primary_historically_exposed"] != 32:
        raise ValueError("primary historical exposure proof drifted")
    if result["confirmation_historically_exposed"] != 32:
        raise ValueError("confirmation historical exposure proof drifted")
    return result


def select_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be auto, cpu or mps")
    name = requested
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def load_model(device: torch.device) -> tuple[TileAwareDualNAFNet, dict[str, Any]]:
    if sha256_file(CHECKPOINT_PATH) != CHECKPOINT_SHA256:
        raise ValueError("frozen DualNAF checkpoint hash mismatch")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model_configuration = checkpoint.get("model_configuration")
    training = checkpoint.get("training_configuration")
    if not isinstance(model_configuration, Mapping) or not isinstance(training, Mapping):
        raise ValueError("checkpoint configuration missing")
    expected_model = {"architecture": "dual_naf", "base": 24, "depth": 3, "blocks": 2}
    if dict(model_configuration) != expected_model:
        raise ValueError("checkpoint model configuration drifted")
    if training.get("protocol_digest") != _load_manifest()["protocol_digest"]:
        raise ValueError("checkpoint protocol drifted")
    if training.get("nlm_h") != 10:
        raise ValueError("checkpoint conditioning strength drifted")
    model = TileAwareDualNAFNet(base=24, depth=3, blocks=2).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, {
        "model_configuration": dict(model_configuration),
        "training_protocol_digest": str(training["protocol_digest"]),
        "training_nlm_h": int(training["nlm_h"]),
    }


def blend_tiles_alpha0125(original: np.ndarray, rendered: np.ndarray) -> np.ndarray:
    source = np.asarray(original)
    prediction = np.asarray(rendered)
    if source.shape != (576, 20, 20, 3) or source.dtype != np.uint8:
        raise ValueError("original tile roster is malformed")
    if prediction.shape != source.shape or prediction.dtype != np.uint8:
        raise ValueError("rendered tile roster is malformed")
    return np.clip(
        np.rint(0.875 * source.astype(np.float64) + 0.125 * prediction.astype(np.float64)),
        0,
        255,
    ).astype(np.uint8)


def harmonize_tiles(ordered_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(
        ordered_tiles,
        DEFAULT_SEAM_GRAPH_CONFIG,
    )
    rgb_tiles = apply_rgb_offsets(ordered_tiles, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(
        rgb_tiles,
        DEFAULT_LUMINANCE_GAIN_CONFIG,
    )
    return assemble_tiles(apply_luminance_gains(rgb_tiles, gains)), {
        "rgb_seam_offsets": rgb_diagnostics,
        "bounded_luminance_gains": luma_diagnostics,
    }


def infer_four_arms(
    dirty: np.ndarray,
    model: TileAwareDualNAFNet,
    device: torch.device,
) -> dict[str, Any]:
    started = perf_counter()
    input_tiles = split_tiles(dirty)
    right, down = directional_scores(input_tiles, views=("bilateral",))["bilateral"]
    solved = solve_buddies(right, down, max_edges=96)
    layout = np.asarray(solved.layout, dtype=np.int32)
    ordered_raw = np.ascontiguousarray(input_tiles[layout])
    raw = assemble_tiles(ordered_raw)
    audit = audit_raw_permutation(
        dirty,
        raw,
        layout,
        restoration_applied_after_audit=True,
    )
    if not audit.passed:
        raise RuntimeError(f"strict permutation audit failed: {audit.as_dict()}")

    rendered_tiles, render_diagnostics = render_tiles_independently(
        model,
        ordered_raw,
        device,
        nlm_h=10,
        batch_size=144,
    )
    blended_tiles = blend_tiles_alpha0125(ordered_raw, rendered_tiles)
    harmonized_alpha0, raw_harmonizer = harmonize_tiles(ordered_raw)
    harmonized_alpha0125, blend_harmonizer = harmonize_tiles(blended_tiles)

    h0_20 = apply_nlm_color(harmonized_alpha0, h=20)
    h0_28 = apply_nlm_color(harmonized_alpha0, h=28)
    h125_20 = apply_nlm_color(harmonized_alpha0125, h=20)
    h125_28 = apply_nlm_color(harmonized_alpha0125, h=28)
    arms = {
        ARM_A: h0_20.image,
        ARM_B: h0_28.image,
        ARM_C: h125_20.image,
        ARM_D: h125_28.image,
    }
    support = {
        "raw": raw,
        "dualnaf_same_index": assemble_tiles(rendered_tiles),
        "blend_alpha0p125": assemble_tiles(blended_tiles),
        "harmonized_alpha0": harmonized_alpha0,
        "harmonized_alpha0p125": harmonized_alpha0125,
    }
    if tuple(arms) != ARMS or tuple(support) != SUPPORT_IMAGES:
        raise RuntimeError("frozen output roster drifted")
    return {
        "layout": layout,
        "layout_sha256": layout_digest(layout),
        "audit": audit.as_dict(),
        "support": support,
        "arms": arms,
        "solver": str(solved.solver),
        "objective": float(solved.objective),
        "renderer_diagnostics": render_diagnostics.as_dict(),
        "harmonizer_diagnostics": {
            "alpha0": raw_harmonizer,
            "alpha0p125": blend_harmonizer,
        },
        "nlm_seconds": {
            ARM_A: float(h0_20.seconds),
            ARM_B: float(h0_28.seconds),
            ARM_C: float(h125_20.seconds),
            ARM_D: float(h125_28.seconds),
        },
        "total_seconds": perf_counter() - started,
    }


def _summarize_safety(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        metric_values: dict[str, np.ndarray] = {}
        for metric, output_name in (
            ("within_tile_gradient", "within_tile_gradient_retention_vs_A"),
            ("laplacian_energy", "laplacian_retention_vs_A"),
            ("grid_ratio", "grid_ratio_relative_to_A"),
        ):
            values = np.asarray(
                [row["safety"][arm][metric] / row["safety"][ARM_A][metric] for row in rows],
                dtype=np.float64,
            )
            metric_values[output_name] = values
        hashes = [str(row["artifact_sha256"][arm]["array"]) for row in rows]
        result[arm] = {
            name: {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for name, values in metric_values.items()
        }
        result[arm]["distinct_prediction_hashes"] = len(set(hashes))
        result[arm]["all_predictions_distinct_across_boards"] = len(set(hashes)) == len(hashes)
    return result


def _source_hashes() -> dict[str, str]:
    paths = (
        CONFIG_PATH,
        MANIFEST_PATH,
        CHECKPOINT_PATH,
        PROJECT_ROOT / "configs" / "postassembly_rgb_offset_v1.json",
        PROJECT_ROOT / "configs" / "postassembly_luminance_gain_v1.json",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "dualnaf_stack.py",
        PROJECT_ROOT / "scripts" / "run_dualnaf_stack.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "dense_safe_tail.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "tilewise_renderer.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "restoration_r6.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "postassembly_harmonizer.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _panel_root(panel: str, output_root: Path = OUTPUT_ROOT) -> Path:
    if panel not in PANEL_KEYS:
        raise ValueError("panel must be primary or confirmation")
    return output_root / panel


def _require_confirmation_authorized(output_root: Path) -> None:
    primary_root = _panel_root("primary", output_root)
    report_path = primary_root / "report.json"
    review_path = primary_root / "manual-review.json"
    if not report_path.is_file() or not review_path.is_file():
        raise RuntimeError("confirmation requires primary report and manual review")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if report.get("numeric_gate", {}).get("all_passed") is not True:
        raise RuntimeError("confirmation forbidden because primary numeric gate failed")
    if review.get("overall_verdict") != "PASS":
        raise RuntimeError("confirmation forbidden because primary manual review did not pass")


def prepare_panel(
    panel: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    inputs_dir: Path = INPUTS_DIR,
    device_name: str = "auto",
) -> Path:
    config = load_config()
    exposure = audit_historical_exposure()
    if panel == "confirmation":
        _require_confirmation_authorized(output_root)
    records = panel_records(panel)
    root = _panel_root(panel, output_root)
    if root.exists():
        raise FileExistsError(f"panel output already exists: {root}")
    artifacts_root = root / "artifacts"
    artifacts_root.mkdir(parents=True)
    device = select_device(device_name)
    model, checkpoint_metadata = load_model(device)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        filename = str(record["filename"])
        dirty = load_rgb_verified(inputs_dir / filename, str(record["input_sha256"]))
        inference = infer_four_arms(dirty, model, device)
        board_root = artifacts_root / Path(filename).stem
        board_root.mkdir()
        images = {**inference["support"], **inference["arms"]}
        artifact_paths: dict[str, str] = {}
        artifact_sha256: dict[str, Any] = {}
        for name, image in images.items():
            relative = Path("artifacts") / Path(filename).stem / f"{name}.png"
            path = root / relative
            _write_bytes_exclusive(path, _png_bytes(image))
            artifact_paths[name] = relative.as_posix()
            artifact_sha256[name] = {
                "file": sha256_file(path),
                "array": array_sha256(image),
            }
        layout_relative = Path("artifacts") / Path(filename).stem / "layout.npy"
        layout_path = root / layout_relative
        _write_bytes_exclusive(layout_path, _npy_bytes(inference["layout"]))
        rows.append(
            {
                "index": index,
                "filename": filename,
                "input_sha256": str(record["input_sha256"]),
                "target_sha256_committed_but_not_opened": str(record["target_sha256"]),
                "layout_sha256": inference["layout_sha256"],
                "layout_path": layout_relative.as_posix(),
                "layout_file_sha256": sha256_file(layout_path),
                "artifact_paths": artifact_paths,
                "artifact_sha256": artifact_sha256,
                "permutation_audit": inference["audit"],
                "solver": inference["solver"],
                "objective": inference["objective"],
                "renderer_diagnostics": inference["renderer_diagnostics"],
                "harmonizer_diagnostics": inference["harmonizer_diagnostics"],
                "nlm_seconds": inference["nlm_seconds"],
                "total_seconds": inference["total_seconds"],
                "safety": {
                    name: safety_metrics(image) for name, image in inference["arms"].items()
                },
            }
        )
        print(
            json.dumps(
                {
                    "phase": "prepare",
                    "panel": panel,
                    "completed": index + 1,
                    "total": len(records),
                    "filename": filename,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    commitment = {
        "schema": COMMITMENT_SCHEMA,
        "panel": panel,
        "config_sha256": CONFIG_SHA256,
        "scientific_scope": config["scientific_scope"],
        "count": len(records),
        "filename_nul_join_sha256": _filename_digest(records),
        "arms": list(ARMS),
        "only_promotable_candidate": ARM_D,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_metadata": checkpoint_metadata,
        "device": str(device),
        "historical_exposure_audit": exposure,
        "source_sha256": _source_hashes(),
        "target_access_contract": {
            "target_directory_was_not_an_argument_to_prepare_panel": True,
            "target_paths_opened_by_this_phase": False,
            "all_support_and_arm_predictions_frozen": True,
            "all_permutation_audits_passed": all(
                row["permutation_audit"]["passed"] is True for row in rows
            ),
        },
        "target_free_safety_summary": _summarize_safety(rows),
        "aggregate_artifact_manifest_sha256": hashlib.sha256(
            _canonical_json({"per_board": rows})
        ).hexdigest(),
        "per_board": rows,
    }
    commitment_path = root / "prediction-commitment.json"
    _write_json_exclusive(commitment_path, commitment, readonly=True)
    return commitment_path


def paired_bootstrap(values: Sequence[float]) -> tuple[float, float]:
    differences = np.asarray(values, dtype=np.float64)
    if differences.ndim != 1 or len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("paired bootstrap requires at least two finite values")
    rng = np.random.default_rng(EXPERIMENT_SUBSET_SEED)
    chunks: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        count = min(remaining, 4096)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        chunks.append(differences[indices].mean(axis=1))
        remaining -= count
    low, high = np.quantile(np.concatenate(chunks), (0.025, 0.975))
    return float(low), float(high)


def _load_artifact_rgb(root: Path, row: Mapping[str, Any], name: str) -> np.ndarray:
    path = root / row["artifact_paths"][name]
    if sha256_file(path) != row["artifact_sha256"][name]["file"]:
        raise ValueError(f"artifact file hash drifted: {path}")
    with Image.open(path) as image:
        image.load()
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array_sha256(value) != row["artifact_sha256"][name]["array"]:
        raise ValueError(f"artifact array hash drifted: {path}")
    return value


def _score_summary(scores: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        values = np.asarray(scores[arm], dtype=np.float64)
        result[arm] = {
            "mean_ssim": float(values.mean()),
            "std_ssim": float(values.std()),
            "min_ssim": float(values.min()),
            "max_ssim": float(values.max()),
            "count": len(values),
        }
    for baseline in (ARM_A, ARM_B):
        differences = np.asarray(scores[ARM_D], dtype=np.float64) - np.asarray(
            scores[baseline], dtype=np.float64
        )
        result[f"{ARM_D}__minus__{baseline}"] = {
            "mean_gain": float(differences.mean()),
            "paired_t_ci95": list(paired_t_interval(differences)),
            "paired_bootstrap_ci95": list(paired_bootstrap(differences)),
            "wins_ties_losses": [
                int(np.sum(differences > 0)),
                int(np.sum(differences == 0)),
                int(np.sum(differences < 0)),
            ],
            "count": len(differences),
        }
    return result


def evaluate_numeric_gate(
    score_summary: Mapping[str, Any],
    safety_summary: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = load_config()["primary_numeric_gate_for_D_only"]
    comparison_a = score_summary[f"{ARM_D}__minus__{ARM_A}"]
    comparison_b = score_summary[f"{ARM_D}__minus__{ARM_B}"]
    safety = safety_summary[ARM_D]
    checks = {
        "mean_rgb_ssim_min": score_summary[ARM_D]["mean_ssim"] >= thresholds["mean_rgb_ssim_min"],
        "paired_t_gain_vs_A_ci95_lower_strictly_greater_than": comparison_a["paired_t_ci95"][0]
        > thresholds["paired_t_gain_vs_A_ci95_lower_strictly_greater_than"],
        "paired_bootstrap_gain_vs_A_ci95_lower_strictly_greater_than": comparison_a[
            "paired_bootstrap_ci95"
        ][0]
        > thresholds["paired_bootstrap_gain_vs_A_ci95_lower_strictly_greater_than"],
        "paired_t_gain_vs_B_ci95_lower_strictly_greater_than": comparison_b["paired_t_ci95"][0]
        > thresholds["paired_t_gain_vs_B_ci95_lower_strictly_greater_than"],
        "paired_bootstrap_gain_vs_B_ci95_lower_strictly_greater_than": comparison_b[
            "paired_bootstrap_ci95"
        ][0]
        > thresholds["paired_bootstrap_gain_vs_B_ci95_lower_strictly_greater_than"],
        "wins_vs_A_min": comparison_a["wins_ties_losses"][0] >= thresholds["wins_vs_A_min"],
        "wins_vs_B_min": comparison_b["wins_ties_losses"][0] >= thresholds["wins_vs_B_min"],
        "mean_within_tile_gradient_retention_vs_A_min": safety[
            "within_tile_gradient_retention_vs_A"
        ]["mean"]
        >= thresholds["mean_within_tile_gradient_retention_vs_A_min"],
        "minimum_board_within_tile_gradient_retention_vs_A_min": safety[
            "within_tile_gradient_retention_vs_A"
        ]["min"]
        >= thresholds["minimum_board_within_tile_gradient_retention_vs_A_min"],
        "mean_laplacian_retention_vs_A_min": safety["laplacian_retention_vs_A"]["mean"]
        >= thresholds["mean_laplacian_retention_vs_A_min"],
        "minimum_board_laplacian_retention_vs_A_min": safety["laplacian_retention_vs_A"]["min"]
        >= thresholds["minimum_board_laplacian_retention_vs_A_min"],
        "mean_grid_ratio_relative_to_A_max": safety["grid_ratio_relative_to_A"]["mean"]
        <= thresholds["mean_grid_ratio_relative_to_A_max"],
        "maximum_board_grid_ratio_relative_to_A_max": safety["grid_ratio_relative_to_A"]["max"]
        <= thresholds["maximum_board_grid_ratio_relative_to_A_max"],
        "all_D_predictions_distinct_across_boards": safety["all_predictions_distinct_across_boards"]
        is thresholds["all_D_predictions_distinct_across_boards"],
    }
    return {
        "candidate": ARM_D,
        "thresholds": thresholds,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def _contact_sheet(
    root: Path,
    commitment: Mapping[str, Any],
    targets: Mapping[str, np.ndarray],
    *,
    center_zoom: bool,
) -> Path:
    indices = (0, 10, 21, 31)
    columns = ("target", ARM_A, ARM_B, ARM_C, ARM_D)
    cell = 180
    label_height = 24
    sheet = Image.new("RGB", (len(columns) * cell, len(indices) * (cell + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for output_row, board_index in enumerate(indices):
        row = commitment["per_board"][board_index]
        filename = str(row["filename"])
        for column_index, name in enumerate(columns):
            image = targets[filename] if name == "target" else _load_artifact_rgb(root, row, name)
            if center_zoom:
                image = image[120:360, 120:360]
            thumbnail = Image.fromarray(image, mode="RGB").resize(
                (cell, cell),
                Image.Resampling.LANCZOS,
            )
            x = column_index * cell
            y = output_row * (cell + label_height)
            sheet.paste(thumbnail, (x, y))
            draw.text((x + 2, y + cell + 2), f"{filename[:-4]} {name}", fill="black", font=font)
    suffix = "center-zoom" if center_zoom else "full"
    path = root / f"manual-review-{suffix}.png"
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG", optimize=False)
    _write_bytes_exclusive(path, buffer.getvalue())
    return path


def score_panel(
    panel: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    targets_dir: Path = TARGETS_DIR,
) -> Path:
    root = _panel_root(panel, output_root)
    commitment_path = root / "prediction-commitment.json"
    receipt_path = root / "TARGETS_OPENED.receipt.json"
    report_path = root / "report.json"
    if receipt_path.exists() or report_path.exists():
        raise RuntimeError(f"panel scoring is single-use: {root}")
    if panel == "confirmation":
        _require_confirmation_authorized(output_root)
    commitment_bytes = commitment_path.read_bytes()
    commitment = json.loads(commitment_bytes)
    if commitment.get("schema") != COMMITMENT_SCHEMA:
        raise ValueError("commitment schema drifted")
    if commitment.get("panel") != panel or commitment.get("config_sha256") != CONFIG_SHA256:
        raise ValueError("commitment identity drifted")
    if commitment.get("source_sha256") != _source_hashes():
        raise ValueError("source changed after prediction commitment")
    records = panel_records(panel)
    if [row["filename"] for row in records] != [row["filename"] for row in commitment["per_board"]]:
        raise ValueError("committed roster drifted")
    for row in commitment["per_board"]:
        layout_path = root / row["layout_path"]
        if sha256_file(layout_path) != row["layout_file_sha256"]:
            raise ValueError(f"layout artifact drifted: {layout_path}")
        for name in (*SUPPORT_IMAGES, *ARMS):
            _load_artifact_rgb(root, row, name)

    _write_json_exclusive(
        receipt_path,
        {
            "schema": "aiijc-dualnaf-stack-target-open-receipt-v1",
            "panel": panel,
            "config_sha256": CONFIG_SHA256,
            "commitment_file_sha256": hashlib.sha256(commitment_bytes).hexdigest(),
            "historically_exposed_before_this_experiment": True,
            "meaning": "single-use transition for this experiment; not a freshness claim",
        },
        readonly=True,
    )

    scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    scored_rows: list[dict[str, Any]] = []
    target_cache: dict[str, np.ndarray] = {}
    for index, (record, row) in enumerate(zip(records, commitment["per_board"], strict=True)):
        filename = str(record["filename"])
        target = load_rgb_verified(targets_dir / filename, str(record["target_sha256"]))
        target_cache[filename] = target
        board_scores: dict[str, float] = {}
        for arm in ARMS:
            value = contest_ssim(target, _load_artifact_rgb(root, row, arm))
            scores[arm].append(value)
            board_scores[arm] = value
        scored_rows.append({"filename": filename, "ssim": board_scores})
        print(
            json.dumps(
                {
                    "phase": "score",
                    "panel": panel,
                    "completed": index + 1,
                    "total": len(records),
                    "filename": filename,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = _score_summary(scores)
    safety = commitment["target_free_safety_summary"]
    gate = evaluate_numeric_gate(summary, safety)
    full_sheet = _contact_sheet(root, commitment, target_cache, center_zoom=False)
    zoom_sheet = _contact_sheet(root, commitment, target_cache, center_zoom=True)
    report = {
        "schema": REPORT_SCHEMA,
        "panel": panel,
        "scientific_scope": load_config()["scientific_scope"],
        "config_sha256": CONFIG_SHA256,
        "count": len(records),
        "filename_nul_join_sha256": _filename_digest(records),
        "historical_exposure_audit": commitment["historical_exposure_audit"],
        "prediction_commitment_sha256": sha256_file(commitment_path),
        "target_open_receipt_sha256": sha256_file(receipt_path),
        "score_summary": summary,
        "target_free_safety_summary": safety,
        "numeric_gate": gate,
        "manual_review": {
            "status": "PENDING_EXPLICIT_REVIEW",
            "required_before_confirmation": True,
            "boards_by_panel_index": [0, 10, 21, 31],
            "full_sheet": {
                "path": str(full_sheet.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(full_sheet),
            },
            "center_zoom_sheet": {
                "path": str(zoom_sheet.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(zoom_sheet),
            },
        },
        "per_board": scored_rows,
    }
    _write_json_exclusive(report_path, report, readonly=True)
    return report_path


def record_manual_review(
    panel: str,
    verdict: str,
    reason: str,
    *,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    if verdict not in {"PASS", "FAIL"}:
        raise ValueError("manual verdict must be PASS or FAIL")
    if not reason.strip():
        raise ValueError("manual review reason must be non-empty")
    root = _panel_root(panel, output_root)
    report_path = root / "report.json"
    if not report_path.is_file():
        raise RuntimeError("manual review requires a scored report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = load_config()["manual_review_gate"]
    payload = {
        "schema": "aiijc-dualnaf-stack-manual-review-v1",
        "panel": panel,
        "config_sha256": CONFIG_SHA256,
        "report_sha256": sha256_file(report_path),
        "boards_by_panel_index": protocol["boards_by_panel_index"],
        "views": protocol["views"],
        "comparison": protocol["comparison"],
        "criteria": protocol["must_pass_all"],
        "overall_verdict": verdict,
        "reason": reason.strip(),
        "numeric_gate_passed": report["numeric_gate"]["all_passed"],
        "confirmation_authorized": bool(
            verdict == "PASS" and report["numeric_gate"]["all_passed"] is True
        ),
    }
    path = root / "manual-review.json"
    _write_json_exclusive(path, payload, readonly=True)
    return path
