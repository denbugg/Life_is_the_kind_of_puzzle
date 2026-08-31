"""Preregistered reused-calibration screen for legal dense single-pass NLM tails.

This module deliberately separates target-free prediction commitment from target
decoding.  The selected calibration panels are *not* fresh: an older legacy
report already evaluated all 700 records.  The separation still prevents this
experiment from choosing per-board outputs after looking at its targets.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import t as student_t

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
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
    TILE_SIZE,
    assemble_tiles,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "dense_safe_tail_reused_calibration_preregistered_v2.json"
CONFIG_SHA256 = "6e1ed5840bf77f9ce5ef7f3a83cdfb84232fa34813f700bda11a56eae2b8fa3c"
MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
INPUTS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
TARGETS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
LEGACY_ALL_CALIBRATION_REPORT = (
    PROJECT_ROOT / "outputs" / "legacy-upgrade" / "calibration700-champion" / "report.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dense-safe-tail" / CONFIG_SHA256

BASELINE_ARM = "nlm_h20x1_baseline"
NLM_ARMS = tuple(f"nlm_h{strength}x1" for strength in range(21, 30))
BLEND_ARMS = ("blend_h20_75_h28_25", "blend_h20_50_h28_50")
ARMS = (BASELINE_ARM, *NLM_ARMS, *BLEND_ARMS)
PANEL_KEYS = {
    "primary": "primary",
    "confirmation": "confirmation_if_and_only_if_primary_passes",
}
REPORT_SCHEMA = "aiijc-dense-safe-tail-reused-calibration-report-v1"
COMMITMENT_SCHEMA = "aiijc-dense-safe-tail-prediction-commitment-v1"


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Serialize a mapping deterministically for content addressing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def array_sha256(value: np.ndarray) -> str:
    """Hash a contiguous array including a small dtype/shape domain separator."""

    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{','.join(map(str, array.shape))}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def filename_nul_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash UTF-8 filenames joined by NUL bytes without a terminal NUL."""

    payload = b"\0".join(str(record["filename"]).encode("utf-8") for record in records)
    return hashlib.sha256(payload).hexdigest()


def _write_bytes_exclusive(path: Path, contents: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, mode)
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
        raise ValueError(f"invalid RGB image: shape={array.shape}, dtype={array.dtype}")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def load_config() -> dict[str, Any]:
    """Load the exact read-only preregistration and reject any byte drift."""

    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("dense-safe-tail preregistration hash mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status_at_freeze") != (
        "preregistered_before_this_experiment_decodes_any_target_pixels"
    ):
        raise ValueError("preregistration status drifted")
    if config.get("scientific_scope") != (
        "reused calibration; not fresh, not holdout, and not an unbiased generalization estimate"
    ):
        raise ValueError("scientific scope drifted")
    if config["protocol"]["historical_exposure_audit"]["freshness_claim"] is not False:
        raise ValueError("historical target exposure must not be hidden")
    declared_arms: list[str] = [config["arms"][0]["name"]]
    declared_arms.extend(
        config["arms"][1]["name_pattern"].format(h=value) for value in config["arms"][1]["h_values"]
    )
    declared_arms.extend(arm["name"] for arm in config["arms"][2:])
    if tuple(declared_arms) != ARMS:
        raise ValueError("arm roster drifted")
    return config


def _load_manifest() -> dict[str, Any]:
    config = load_config()
    if sha256_file(MANIFEST_PATH) != config["protocol"]["manifest_sha256"]:
        raise ValueError("validation manifest hash mismatch")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != (
        "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
    ):
        raise ValueError("validation manifest protocol drifted")
    return manifest


def panel_records(panel: str) -> tuple[Mapping[str, Any], ...]:
    """Return the exact direct manifest slice named in the preregistration."""

    if panel not in PANEL_KEYS:
        raise ValueError("panel must be primary or confirmation")
    config = load_config()
    protocol = config["protocol"]
    section = protocol[PANEL_KEYS[panel]]
    selector = protocol["selector"]
    if selector["namespace"] != EXPERIMENT_SUBSET_NAMESPACE:
        raise ValueError("selector namespace drifted")
    if selector["seed"] != EXPERIMENT_SUBSET_SEED:
        raise ValueError("selector seed drifted")
    ranked_prefix = select_manifest_records(
        _load_manifest(),
        "calibration",
        limit=int(section["offset"]) + int(section["count"]),
        seed=EXPERIMENT_SUBSET_SEED,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
    )
    records = tuple(ranked_prefix[int(section["offset"]) :])
    if len(records) != section["count"]:
        raise ValueError("panel count drifted")
    if filename_nul_digest(records) != section["filename_nul_join_sha256"]:
        raise ValueError("panel filename digest drifted")
    return records


def audit_historical_exposure() -> dict[str, Any]:
    """Prove the panels are mutually disjoint but historically target-exposed."""

    primary = panel_records("primary")
    confirmation = panel_records("confirmation")
    primary_names = {str(record["filename"]) for record in primary}
    confirmation_names = {str(record["filename"]) for record in confirmation}
    if primary_names & confirmation_names:
        raise ValueError("primary and confirmation panels overlap")

    legacy = json.loads(LEGACY_ALL_CALIBRATION_REPORT.read_text(encoding="utf-8"))
    legacy_records = {
        str(row["filename"]): (
            str(row["input_sha256"]),
            str(row["target_sha256"]),
        )
        for row in legacy["per_board"]
    }
    current_calibration = _load_manifest()["splits"]["calibration"]
    exact_current_matches = sum(
        legacy_records.get(str(record["filename"]))
        == (str(record["input_sha256"]), str(record["target_sha256"]))
        for record in current_calibration
    )

    def exposed_count(records: Sequence[Mapping[str, Any]]) -> int:
        return sum(
            legacy_records.get(str(record["filename"]))
            == (str(record["input_sha256"]), str(record["target_sha256"]))
            for record in records
        )

    audit = {
        "freshness_claim": False,
        "primary_confirmation_overlap": 0,
        "legacy_report_sha256": sha256_file(LEGACY_ALL_CALIBRATION_REPORT),
        "legacy_exact_matches_to_current_calibration": exact_current_matches,
        "primary_historically_exposed": exposed_count(primary),
        "confirmation_historically_exposed": exposed_count(confirmation),
        "primary_count": len(primary),
        "confirmation_count": len(confirmation),
    }
    if exact_current_matches != 700:
        raise ValueError("legacy calibration exposure evidence is incomplete")
    if audit["primary_historically_exposed"] != len(primary):
        raise ValueError("primary exposure count drifted")
    if audit["confirmation_historically_exposed"] != len(confirmation):
        raise ValueError("confirmation exposure count drifted")
    return audit


def _infer_frozen_canvases(dirty: np.ndarray) -> dict[str, Any]:
    """Run the fixed no-atlas layout and target-blind harmonizers."""

    started = perf_counter()
    input_tiles = split_tiles(dirty)
    right, down = directional_scores(input_tiles, views=("bilateral",))["bilateral"]
    solved = solve_buddies(right, down, max_edges=96)
    layout = np.asarray(solved.layout, dtype=np.int32)
    raw = assemble_tiles(input_tiles[layout])
    audit = audit_raw_permutation(
        dirty,
        raw,
        layout,
        restoration_applied_after_audit=True,
    )
    if not audit.passed:
        raise RuntimeError(f"strict permutation audit failed: {audit.as_dict()}")

    ordered = split_tiles(raw)
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(
        ordered,
        DEFAULT_SEAM_GRAPH_CONFIG,
    )
    rgb_tiles = apply_rgb_offsets(ordered, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(
        rgb_tiles,
        DEFAULT_LUMINANCE_GAIN_CONFIG,
    )
    harmonized = assemble_tiles(apply_luminance_gains(rgb_tiles, gains))
    return {
        "layout": layout,
        "raw": raw,
        "harmonized": harmonized,
        "audit": audit.as_dict(),
        "layout_sha256": layout_digest(layout),
        "objective": float(solved.objective),
        "solver": str(solved.solver),
        "harmonizer_diagnostics": {
            "rgb_seam_offsets": rgb_diagnostics,
            "bounded_luminance_gains": luma_diagnostics,
        },
        "seconds": perf_counter() - started,
    }


def generate_arms(harmonized: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Generate the exact fixed arm roster without sequential filtering."""

    outputs: dict[str, np.ndarray] = {}
    seconds: dict[str, float] = {}
    for strength in range(20, 30):
        result = apply_nlm_color(harmonized, h=strength)
        name = BASELINE_ARM if strength == 20 else f"nlm_h{strength}x1"
        outputs[name] = result.image
        seconds[name] = float(result.seconds)

    h20 = outputs[BASELINE_ARM].astype(np.float64)
    h28 = outputs["nlm_h28x1"].astype(np.float64)
    blend_specs = {
        "blend_h20_75_h28_25": (0.75, 0.25),
        "blend_h20_50_h28_50": (0.50, 0.50),
    }
    for name, (weight20, weight28) in blend_specs.items():
        started = perf_counter()
        outputs[name] = np.clip(np.rint(weight20 * h20 + weight28 * h28), 0, 255).astype(np.uint8)
        seconds[name] = perf_counter() - started
    if tuple(outputs) != ARMS:
        raise RuntimeError("generated arm order drifted")
    return outputs, seconds


def safety_metrics(rgb: np.ndarray) -> dict[str, float]:
    """Compute the three preregistered target-free detail/grid diagnostics."""

    image = np.asarray(rgb)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError("safety metrics require uint8 RGB 480x480")
    luminance = (
        0.299 * image[..., 0].astype(np.float64)
        + 0.587 * image[..., 1].astype(np.float64)
        + 0.114 * image[..., 2].astype(np.float64)
    )
    horizontal = np.abs(np.diff(luminance, axis=1))
    vertical = np.abs(np.diff(luminance, axis=0))
    h_positions = np.arange(1, IMAGE_SIZE)
    v_positions = np.arange(1, IMAGE_SIZE)
    h_interior = horizontal[:, h_positions % TILE_SIZE != 0]
    v_interior = vertical[v_positions % TILE_SIZE != 0, :]
    h_grid = horizontal[:, h_positions % TILE_SIZE == 0]
    v_grid = vertical[v_positions % TILE_SIZE == 0, :]
    within = float((h_interior.sum() + v_interior.sum()) / (h_interior.size + v_interior.size))
    grid = float((h_grid.sum() + v_grid.sum()) / (h_grid.size + v_grid.size))
    laplacian = float(np.mean(np.abs(cv2.Laplacian(luminance, cv2.CV_64F, ksize=3))))
    return {
        "within_tile_gradient": within,
        "laplacian_energy": laplacian,
        "grid_ratio": grid / max(within, 1e-12),
    }


def summarize_safety(per_board: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate target-free safety ratios relative to the h20 arm."""

    summary: dict[str, Any] = {}
    for arm in ARMS:
        ratios = {
            "within_tile_gradient_retention_vs_h20": np.asarray(
                [
                    row["safety"][arm]["within_tile_gradient"]
                    / row["safety"][BASELINE_ARM]["within_tile_gradient"]
                    for row in per_board
                ],
                dtype=np.float64,
            ),
            "laplacian_retention_vs_h20": np.asarray(
                [
                    row["safety"][arm]["laplacian_energy"]
                    / row["safety"][BASELINE_ARM]["laplacian_energy"]
                    for row in per_board
                ],
                dtype=np.float64,
            ),
            "grid_ratio_relative_to_h20": np.asarray(
                [
                    row["safety"][arm]["grid_ratio"] / row["safety"][BASELINE_ARM]["grid_ratio"]
                    for row in per_board
                ],
                dtype=np.float64,
            ),
        }
        hashes = [str(row["artifact_sha256"][arm]["array"]) for row in per_board]
        summary[arm] = {
            name: {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for name, values in ratios.items()
        }
        summary[arm]["distinct_prediction_hashes"] = len(set(hashes))
        summary[arm]["all_predictions_distinct_across_boards"] = len(set(hashes)) == len(hashes)
    return summary


def _source_hashes() -> dict[str, str]:
    paths = (
        CONFIG_PATH,
        MANIFEST_PATH,
        PROJECT_ROOT / "configs" / "postassembly_rgb_offset_v1.json",
        PROJECT_ROOT / "configs" / "postassembly_luminance_gain_v1.json",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "dense_safe_tail.py",
        PROJECT_ROOT / "scripts" / "run_dense_safe_tail.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "postassembly_harmonizer.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "pixel_tails.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _panel_root(panel: str, output_root: Path = OUTPUT_ROOT) -> Path:
    if panel not in PANEL_KEYS:
        raise ValueError("panel must be primary or confirmation")
    return output_root / panel


def _require_confirmation_authorized(output_root: Path) -> str:
    report_path = _panel_root("primary", output_root) / "report.json"
    if not report_path.is_file():
        raise RuntimeError("confirmation forbidden before a primary report exists")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("promotion", {}).get("all_passed") is not True:
        raise RuntimeError("confirmation forbidden because primary promotion gate failed")
    winner = report["promotion"].get("winner")
    if winner not in ARMS or winner == BASELINE_ARM:
        raise RuntimeError("primary report has no valid frozen winner")
    return str(winner)


def prepare_panel(
    panel: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    inputs_dir: Path = INPUTS_DIR,
) -> Path:
    """Freeze all predictions and commitment without accepting a target path."""

    config = load_config()
    exposure = audit_historical_exposure()
    frozen_winner = None
    if panel == "confirmation":
        frozen_winner = _require_confirmation_authorized(output_root)
    records = panel_records(panel)
    root = _panel_root(panel, output_root)
    commitment_path = root / "prediction-commitment.json"
    if root.exists():
        raise FileExistsError(f"panel output already exists: {root}")
    artifacts_root = root / "artifacts"
    artifacts_root.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        filename = str(record["filename"])
        dirty = load_rgb_verified(inputs_dir / filename, str(record["input_sha256"]))
        inferred = _infer_frozen_canvases(dirty)
        predictions, arm_seconds = generate_arms(inferred["harmonized"])
        board_root = artifacts_root / Path(filename).stem
        board_root.mkdir()

        arrays: dict[str, np.ndarray] = {
            "raw": inferred["raw"],
            "harmonized": inferred["harmonized"],
            **predictions,
        }
        artifact_sha256: dict[str, Any] = {}
        artifact_paths: dict[str, str] = {}
        for name, array in arrays.items():
            relative = Path("artifacts") / Path(filename).stem / f"{name}.png"
            path = root / relative
            _write_bytes_exclusive(path, _png_bytes(array))
            artifact_paths[name] = relative.as_posix()
            artifact_sha256[name] = {
                "file": sha256_file(path),
                "array": array_sha256(array),
            }

        layout_relative = Path("artifacts") / Path(filename).stem / "layout.npy"
        layout_path = root / layout_relative
        _write_bytes_exclusive(layout_path, _npy_bytes(inferred["layout"]))
        rows.append(
            {
                "index": index,
                "filename": filename,
                "input_sha256": str(record["input_sha256"]),
                "target_sha256_committed_but_not_opened": str(record["target_sha256"]),
                "layout_sha256": inferred["layout_sha256"],
                "layout_file_sha256": sha256_file(layout_path),
                "layout_path": layout_relative.as_posix(),
                "artifact_paths": artifact_paths,
                "artifact_sha256": artifact_sha256,
                "permutation_audit": inferred["audit"],
                "objective": inferred["objective"],
                "solver": inferred["solver"],
                "harmonizer_diagnostics": inferred["harmonizer_diagnostics"],
                "runtime_seconds": {
                    "layout_and_harmonization": inferred["seconds"],
                    "arms": arm_seconds,
                },
                "safety": {name: safety_metrics(value) for name, value in predictions.items()},
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

    safety = summarize_safety(rows)
    aggregate = hashlib.sha256(canonical_json({"per_board": rows})).hexdigest()
    commitment = {
        "schema": COMMITMENT_SCHEMA,
        "config": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "config_sha256": CONFIG_SHA256,
        "panel": panel,
        "count": len(records),
        "filename_nul_sha256": filename_nul_digest(records),
        "arms": list(ARMS),
        "frozen_primary_winner_for_confirmation": frozen_winner,
        "historical_exposure_audit": exposure,
        "scientific_scope": config["scientific_scope"],
        "source_sha256": _source_hashes(),
        "target_access_contract": {
            "target_directory_was_not_an_argument_to_prepare_panel": True,
            "target_paths_opened_by_this_phase": False,
            "all_layouts_raw_harmonized_and_arm_predictions_frozen": True,
            "all_permutation_audits_passed": all(
                row["permutation_audit"]["passed"] is True for row in rows
            ),
        },
        "target_free_safety_summary": safety,
        "aggregate_artifact_manifest_sha256": aggregate,
        "per_board": rows,
    }
    _write_json_exclusive(commitment_path, commitment, readonly=True)
    return commitment_path


def paired_t_interval(values: Sequence[float]) -> tuple[float, float]:
    """Return the exact preregistered two-sided 95% paired t interval."""

    differences = np.asarray(values, dtype=np.float64)
    if differences.ndim != 1 or len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("paired t interval requires at least two finite values")
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    critical = float(student_t.ppf(0.975, df=len(differences) - 1))
    return mean - critical * standard_error, mean + critical * standard_error


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
    baseline = np.asarray(scores[BASELINE_ARM], dtype=np.float64)
    summary: dict[str, Any] = {}
    for arm in ARMS:
        values = np.asarray(scores[arm], dtype=np.float64)
        difference = values - baseline
        summary[arm] = {
            "mean_ssim": float(values.mean()),
            "std_ssim": float(values.std()),
            "min_ssim": float(values.min()),
            "max_ssim": float(values.max()),
            "mean_gain_vs_h20": float(difference.mean()),
            "paired_gain_vs_h20_ci95": list(paired_t_interval(difference)),
            "wins_ties_losses_vs_h20": [
                int(np.sum(difference > 0)),
                int(np.sum(difference == 0)),
                int(np.sum(difference < 0)),
            ],
            "count": len(values),
        }
    return summary


def evaluate_primary_gates(
    score_summary: Mapping[str, Any],
    safety_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every fixed gate and select by the preregistered rule."""

    thresholds = load_config()["primary_promotion_gate"]
    candidates: dict[str, Any] = {}
    for arm in ARMS[1:]:
        scores = score_summary[arm]
        safety = safety_summary[arm]
        checks = {
            "mean_rgb_ssim_min": scores["mean_ssim"] >= thresholds["mean_rgb_ssim_min"],
            "paired_gain_vs_h20_ci95_lower_strictly_greater_than": scores[
                "paired_gain_vs_h20_ci95"
            ][0]
            > thresholds["paired_gain_vs_h20_ci95_lower_strictly_greater_than"],
            "wins_vs_h20_min": scores["wins_ties_losses_vs_h20"][0]
            >= thresholds["wins_vs_h20_min"],
            "mean_within_tile_gradient_retention_vs_h20_min": safety[
                "within_tile_gradient_retention_vs_h20"
            ]["mean"]
            >= thresholds["mean_within_tile_gradient_retention_vs_h20_min"],
            "minimum_board_within_tile_gradient_retention_vs_h20_min": safety[
                "within_tile_gradient_retention_vs_h20"
            ]["min"]
            >= thresholds["minimum_board_within_tile_gradient_retention_vs_h20_min"],
            "mean_laplacian_retention_vs_h20_min": safety["laplacian_retention_vs_h20"]["mean"]
            >= thresholds["mean_laplacian_retention_vs_h20_min"],
            "minimum_board_laplacian_retention_vs_h20_min": safety["laplacian_retention_vs_h20"][
                "min"
            ]
            >= thresholds["minimum_board_laplacian_retention_vs_h20_min"],
            "mean_grid_ratio_relative_to_h20_max": safety["grid_ratio_relative_to_h20"]["mean"]
            <= thresholds["mean_grid_ratio_relative_to_h20_max"],
            "maximum_board_grid_ratio_relative_to_h20_max": safety["grid_ratio_relative_to_h20"][
                "max"
            ]
            <= thresholds["maximum_board_grid_ratio_relative_to_h20_max"],
            "all_predictions_distinct_across_boards": safety[
                "all_predictions_distinct_across_boards"
            ]
            is thresholds["all_predictions_distinct_across_boards"],
        }
        candidates[arm] = {"checks": checks, "all_passed": all(checks.values())}

    passers = [arm for arm in ARMS[1:] if candidates[arm]["all_passed"]]
    effective_h = {
        **{f"nlm_h{strength}x1": float(strength) for strength in range(21, 30)},
        "blend_h20_75_h28_25": 22.0,
        "blend_h20_50_h28_50": 24.0,
    }
    winner = None
    if passers:
        winner = sorted(
            passers,
            key=lambda arm: (-float(score_summary[arm]["mean_ssim"]), effective_h[arm], arm),
        )[0]
    return {
        "thresholds": thresholds,
        "candidates": candidates,
        "winner": winner,
        "all_passed": winner is not None,
    }


def _make_contact_sheet(
    panel: str,
    root: Path,
    commitment: Mapping[str, Any],
    targets: Mapping[str, np.ndarray],
) -> Path:
    board_indices = (0, 12, 24, 35)
    columns = ("target", "harmonized", *ARMS)
    cell = 128
    label_height = 24
    sheet = Image.new(
        "RGB",
        (len(columns) * cell, len(board_indices) * (cell + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for output_row, board_index in enumerate(board_indices):
        row = commitment["per_board"][board_index]
        filename = str(row["filename"])
        for column_index, name in enumerate(columns):
            image = targets[filename] if name == "target" else _load_artifact_rgb(root, row, name)
            thumbnail = Image.fromarray(image, mode="RGB").resize(
                (cell, cell),
                Image.Resampling.LANCZOS,
            )
            x = column_index * cell
            y = output_row * (cell + label_height)
            sheet.paste(thumbnail, (x, y))
            draw.text((x + 2, y + cell + 2), f"{filename[:-4]} {name}", fill="black", font=font)
    path = root / "manual-safety-contact-sheet.png"
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
    """Verify commitment, burn a receipt, then decode and score targets once."""

    root = _panel_root(panel, output_root)
    commitment_path = root / "prediction-commitment.json"
    report_path = root / "report.json"
    receipt_path = root / "TARGETS_OPENED.receipt.json"
    if report_path.exists() or receipt_path.exists():
        raise RuntimeError(f"panel scoring is single-use: {root}")
    commitment_bytes = commitment_path.read_bytes()
    commitment = json.loads(commitment_bytes)
    if commitment.get("schema") != COMMITMENT_SCHEMA:
        raise ValueError("commitment schema mismatch")
    if commitment.get("config_sha256") != CONFIG_SHA256 or commitment.get("panel") != panel:
        raise ValueError("commitment identity mismatch")
    if commitment.get("source_sha256") != _source_hashes():
        raise ValueError("source changed after prediction commitment")
    records = panel_records(panel)
    if [record["filename"] for record in records] != [
        row["filename"] for row in commitment["per_board"]
    ]:
        raise ValueError("committed panel roster drifted")
    if panel == "confirmation":
        frozen_winner = _require_confirmation_authorized(output_root)
        if commitment.get("frozen_primary_winner_for_confirmation") != frozen_winner:
            raise ValueError("confirmation winner drifted")

    # Verify every frozen output before permitting the phase transition.
    for row in commitment["per_board"]:
        layout_path = root / row["layout_path"]
        if sha256_file(layout_path) != row["layout_file_sha256"]:
            raise ValueError(f"layout artifact drifted: {layout_path}")
        for name in ("raw", "harmonized", *ARMS):
            _load_artifact_rgb(root, row, name)

    receipt = {
        "schema": "aiijc-dense-safe-tail-target-open-receipt-v1",
        "panel": panel,
        "config_sha256": CONFIG_SHA256,
        "commitment_file_sha256": hashlib.sha256(commitment_bytes).hexdigest(),
        "historically_exposed_before_this_experiment": True,
        "meaning": "single-use transition for this experiment; not a freshness claim",
    }
    _write_json_exclusive(receipt_path, receipt, readonly=True)

    scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    scored_rows: list[dict[str, Any]] = []
    target_cache: dict[str, np.ndarray] = {}
    for index, (record, row) in enumerate(zip(records, commitment["per_board"], strict=True)):
        filename = str(record["filename"])
        target = load_rgb_verified(targets_dir / filename, str(record["target_sha256"]))
        target_cache[filename] = target
        board_scores: dict[str, float] = {}
        for arm in ARMS:
            prediction = _load_artifact_rgb(root, row, arm)
            value = contest_ssim(target, prediction)
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
    promotion = evaluate_primary_gates(summary, safety)
    if panel == "confirmation":
        frozen_winner = str(commitment["frozen_primary_winner_for_confirmation"])
        winner_result = promotion["candidates"][frozen_winner]
        promotion = {
            **promotion,
            "winner": frozen_winner if winner_result["all_passed"] else None,
            "all_passed": winner_result["all_passed"],
            "selection_was_frozen_on_primary": True,
        }
    sheet_path = _make_contact_sheet(panel, root, commitment, target_cache)
    report = {
        "schema": REPORT_SCHEMA,
        "panel": panel,
        "scientific_scope": load_config()["scientific_scope"],
        "config_sha256": CONFIG_SHA256,
        "count": len(records),
        "filename_nul_sha256": filename_nul_digest(records),
        "historical_exposure_audit": commitment["historical_exposure_audit"],
        "prediction_commitment_sha256": sha256_file(commitment_path),
        "target_open_receipt_sha256": sha256_file(receipt_path),
        "score_summary": summary,
        "target_free_safety_summary": safety,
        "promotion": promotion,
        "manual_safety_contact_sheet": {
            "path": str(sheet_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(sheet_path),
            "boards": [0, 12, 24, 35],
            "columns": ["target", "harmonized", *ARMS],
        },
        "per_board": scored_rows,
    }
    _write_json_exclusive(report_path, report, readonly=True)
    return report_path
