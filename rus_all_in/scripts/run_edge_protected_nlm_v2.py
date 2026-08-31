#!/usr/bin/env python3
"""Run the preregistered h28-safe / h40-flat protected-NLM v2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.edge_protected_nlm import (
    SEARCH_WINDOW,
    TEMPLATE_WINDOW,
    colored_nlm,
    image_digest,
)
from aiijc_puzzle.edge_protected_nlm_v2 import SOBEL_THRESHOLD, blend_h28safe_h40flat
from aiijc_puzzle.frozen_final_evaluator import _validate_method_configs, load_rgb_verified
from aiijc_puzzle.legacy_upgrade import directional_scores, layout_digest, solve_buddies
from aiijc_puzzle.nlm_luma_chroma import (
    paired_t_interval,
    safety_summary,
    structure_diagnostics,
)
from aiijc_puzzle.postassembly_harmonizer import (
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    assemble_tiles,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT / "configs/edge_protected_nlm_h28safe_reused_calibration_preregistered_v2.json"
)
CONFIG_SHA256 = "fcb48204015d240d400aec4e5e9d95f0564a5781e305b5b59cf23452da5ede0d"
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
TARGETS = PROJECT_ROOT / "data/raw/train/targets"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/edge-protected-nlm-v2"
PRIMARY_ROOT = OUTPUT_ROOT / "primary-calibration-offset60-count60"
CONFIRMATION_ROOT = OUTPUT_ROOT / "confirmation-calibration-offset0-count60"

CONTROL = "A_nlm_h20_control"
STRONG_CONTROL = "B_nlm_h28_strong_control"
CANDIDATE = "F_h28safe_flat_h40_t40"
PREDICTION_NAMES = (CONTROL, STRONG_CONTROL, CANDIDATE)
ZOOM_INDICES = (0, 19, 39, 59)
SOURCE_FILES = (
    "scripts/run_edge_protected_nlm_v2.py",
    "src/aiijc_puzzle/edge_protected_nlm_v2.py",
    "src/aiijc_puzzle/edge_protected_nlm.py",
    "src/aiijc_puzzle/frozen_final_evaluator.py",
    "src/aiijc_puzzle/nlm_luma_chroma.py",
    "src/aiijc_puzzle/legacy_upgrade.py",
    "src/aiijc_puzzle/postassembly_harmonizer.py",
    "src/aiijc_puzzle/compliant_atlas_decoder.py",
    "src/aiijc_puzzle/protocol.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primary", "confirmation"), default="primary")
    parser.add_argument("--phase", choices=("prepare", "score"), required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--targets", type=Path, default=TARGETS)
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any], *, readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if readonly:
            os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)


def names_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode()
    ).hexdigest()


def roster_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(f"{record['filename']}\0{record['input_sha256']}" for record in records).encode()
    ).hexdigest()


def canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{','.join(map(str, array.shape))}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)


def expected_arms() -> list[dict[str, Any]]:
    return [
        {
            "name": CONTROL,
            "kind": "global_single_pass",
            "h": 20,
            "h_color": 20,
            "role": "baseline_and_candidate_mask_source",
        },
        {
            "name": STRONG_CONTROL,
            "kind": "global_single_pass",
            "h": 28,
            "h_color": 28,
            "role": "strong_control_and_candidate_safe_pixel_source",
        },
        {
            "name": CANDIDATE,
            "kind": "edge_protected_blend",
            "mask_source_h": 20,
            "safe_h": 28,
            "safe_h_color": 28,
            "aggressive_h": 40,
            "aggressive_h_color": 40,
            "sobel_threshold": 40.0,
            "role": "only_candidate",
        },
    ]


def validate_config_agreement(config: Mapping[str, Any]) -> None:
    protocol = config["protocol"]
    if protocol["selector_namespace"] != EXPERIMENT_SUBSET_NAMESPACE:
        raise RuntimeError("selector namespace differs from preregistration")
    if protocol["selector_seed"] != EXPERIMENT_SUBSET_SEED:
        raise RuntimeError("selector seed differs from preregistration")
    if protocol["split"] != "calibration":
        raise RuntimeError("only reused calibration is allowed")
    if (protocol["primary"]["offset"], protocol["primary"]["count"]) != (60, 60):
        raise RuntimeError("primary panel bounds drifted")
    confirmation = protocol["confirmation_if_and_only_if_primary_passes"]
    if (confirmation["offset"], confirmation["count"]) != (0, 60):
        raise RuntimeError("confirmation panel bounds drifted")
    if protocol["primary_confirmation_filename_overlap"] != 0:
        raise RuntimeError("panel overlap contract drifted")
    history = protocol["historical_exposure"]
    if history["freshness_claim"] is not False:
        raise RuntimeError("freshness claim is forbidden")
    if not (
        history["all_primary_records_previously_exposed"]
        and history["all_confirmation_records_previously_exposed"]
    ):
        raise RuntimeError("historical exposure acknowledgement drifted")
    if config["arms"] != expected_arms():
        raise RuntimeError("executable arms differ from immutable config")
    fixed = config["fixed_pipeline"]
    if fixed["nlm_template_window_size"] != TEMPLATE_WINDOW:
        raise RuntimeError("NLM template window drifted")
    if fixed["nlm_search_window_size"] != SEARCH_WINDOW:
        raise RuntimeError("NLM search window drifted")
    mask = config["mask_algorithm"]
    if mask["sobel_threshold"] != SOBEL_THRESHOLD:
        raise RuntimeError("mask threshold drifted")
    if mask["target_or_filename_dependency"] is not False:
        raise RuntimeError("mask independence contract drifted")
    if config["paired_ci"]["confidence"] != 0.95:
        raise RuntimeError("paired CI confidence drifted")
    if tuple(config["manual_review"]["fixed_zoom_board_indices"]) != ZOOM_INDICES:
        raise RuntimeError("manual zoom roster drifted")


def select_section(
    manifest: Mapping[str, Any],
    section: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    offset, count = int(section["offset"]), int(section["count"])
    ranked = select_manifest_records(
        manifest,
        "calibration",
        limit=offset + count,
        seed=EXPERIMENT_SUBSET_SEED,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
    )
    records = tuple(ranked[offset : offset + count])
    if len(records) != count:
        raise RuntimeError("panel count drifted")
    if names_digest(records) != section["filenames_newline_sha256"]:
        raise RuntimeError("filename roster drifted")
    if roster_digest(records) != section["filename_input_roster_sha256"]:
        raise RuntimeError("filename/input roster drifted")
    return records


def load_contract(
    manifest_path: Path,
    mode: str,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    if mode not in {"primary", "confirmation"}:
        raise ValueError(f"unsupported mode: {mode}")
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise RuntimeError("v2 preregistration hash drifted")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config_agreement(config)
    if sha256_file(manifest_path) != config["protocol"]["manifest_sha256"]:
        raise RuntimeError("manifest file hash drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != config["protocol"]["protocol_digest"]:
        raise RuntimeError("manifest protocol digest drifted")
    primary = select_section(manifest, config["protocol"]["primary"])
    confirmation = select_section(
        manifest,
        config["protocol"]["confirmation_if_and_only_if_primary_passes"],
    )
    if {str(row["filename"]) for row in primary} & {str(row["filename"]) for row in confirmation}:
        raise RuntimeError("primary and confirmation panels overlap")
    return config, primary if mode == "primary" else confirmation


def source_hashes() -> dict[str, str]:
    files = (str(CONFIG.relative_to(PROJECT_ROOT)), *SOURCE_FILES)
    return {name: sha256_file(PROJECT_ROOT / name) for name in files}


def freeze_predictions(
    records: Sequence[Mapping[str, Any]],
    *,
    inputs_dir: Path,
) -> list[dict[str, Any]]:
    rgb_config, luma_config, method_hashes = _validate_method_configs()
    frozen: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        started = perf_counter()
        filename = str(record["filename"])
        dirty = load_rgb_verified(inputs_dir / filename, str(record["input_sha256"]))
        tiles = split_tiles(dirty)
        right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
        solved = solve_buddies(right, down, max_edges=96)
        layout = np.asarray(solved.layout, dtype=np.int32)
        ordered = np.ascontiguousarray(tiles[layout])
        raw = assemble_tiles(ordered)
        audit = audit_raw_permutation(
            dirty,
            raw,
            layout,
            restoration_applied_after_audit=True,
        )
        if not audit.passed:
            raise RuntimeError(f"strict permutation audit failed for {filename}")
        offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered, rgb_config)
        rgb_tiles = apply_rgb_offsets(ordered, offsets)
        gains, luma_diagnostics = seam_graph_luminance_gains(rgb_tiles, luma_config)
        harmonized = assemble_tiles(apply_luminance_gains(rgb_tiles, gains))

        h20 = colored_nlm(harmonized, 20)
        h28 = colored_nlm(harmonized, 28)
        h40 = colored_nlm(harmonized, 40)
        candidate, dilated_mask, soft_mask, mask_diagnostics = blend_h28safe_h40flat(
            h20,
            h28,
            h40,
        )
        predictions = {CONTROL: h20, STRONG_CONTROL: h28, CANDIDATE: candidate}
        hashes = {name: image_digest(value) for name, value in predictions.items()}
        frozen.append(
            {
                "record": dict(record),
                "dirty": dirty,
                "layout": layout,
                "layout_sha256": layout_digest(layout),
                "raw": raw,
                "harmonized": harmonized,
                "independent_nlm": {"h20": h20, "h28": h28, "h40": h40},
                "dilated_mask": dilated_mask,
                "soft_mask": soft_mask,
                "predictions": predictions,
                "prediction_sha256": hashes,
                "structure_diagnostics": {
                    name: structure_diagnostics(value) for name, value in predictions.items()
                },
                "mask_diagnostics": mask_diagnostics,
                "audit": audit.as_dict(),
                "objective": float(solved.objective),
                "solver": solved.solver,
                "harmonizer_diagnostics": {
                    "rgb": rgb_diagnostics,
                    "luma": luma_diagnostics,
                    "method_config_sha256": method_hashes,
                },
                "runtime_seconds": perf_counter() - started,
            }
        )
        print(f"froze {index}/{len(records)} {filename}", flush=True)
    return frozen


def target_free_safety(frozen: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = [row["structure_diagnostics"][CONTROL] for row in frozen]
    candidate = [row["structure_diagnostics"][CANDIDATE] for row in frozen]
    summary = safety_summary(candidate, baseline)
    protected = np.asarray(
        [row["mask_diagnostics"]["binary_dilated_protected_fraction"] for row in frozen]
    )
    clipped_increase = np.asarray(
        [
            row["structure_diagnostics"][CANDIDATE]["clipped_fraction"]
            - row["structure_diagnostics"][CONTROL]["clipped_fraction"]
            for row in frozen
        ]
    )
    return {
        **summary,
        "mean_protected_pixel_fraction": float(protected.mean()),
        "minimum_protected_pixel_fraction": float(protected.min()),
        "maximum_protected_pixel_fraction": float(protected.max()),
        "maximum_clipped_fraction_increase": float(clipped_increase.max()),
        "distinct_from_A_and_B_on_every_board": all(
            row["prediction_sha256"][CANDIDATE] != row["prediction_sha256"][CONTROL]
            and row["prediction_sha256"][CANDIDATE] != row["prediction_sha256"][STRONG_CONTROL]
            for row in frozen
        ),
    }


def persist_frozen_artifacts(
    frozen: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    artifact_root = output_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=False)
    metadata: list[dict[str, Any]] = []
    for row in frozen:
        filename = str(row["record"]["filename"])
        arrays: dict[str, np.ndarray] = {
            "dirty": row["dirty"],
            "layout": row["layout"],
            "raw": row["raw"],
            "harmonized": row["harmonized"],
            "dilated_mask_h20_t40": row["dilated_mask"],
            "soft_mask_h20_t40": row["soft_mask"],
        }
        arrays.update(
            {f"independent_nlm__{name}": value for name, value in row["independent_nlm"].items()}
        )
        arrays.update({f"prediction__{name}": value for name, value in row["predictions"].items()})
        relative = Path("artifacts") / f"{Path(filename).stem}.npz"
        path = output_root / relative
        write_npz_exclusive(path, arrays)
        metadata.append(
            {
                "path": relative.as_posix(),
                "file_sha256": sha256_file(path),
                "array_sha256": {name: array_digest(value) for name, value in arrays.items()},
            }
        )
    return metadata


def build_commitment(
    frozen: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    records: Sequence[Mapping[str, Any]],
    safety: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "aiijc-edge-protected-nlm-h28safe-prediction-commitment-v2",
        "mode": mode,
        "config": str(CONFIG),
        "config_sha256": CONFIG_SHA256,
        "source_sha256": source_hashes(),
        "filenames": [record["filename"] for record in records],
        "filenames_newline_sha256": names_digest(records),
        "filename_input_roster_sha256": roster_digest(records),
        "prediction_names": list(PREDICTION_NAMES),
        "candidate": expected_arms()[2],
        "target_free_safety": dict(safety),
        "contract": {
            "target_paths_opened": False,
            "all_predictions_frozen_before_target_access": True,
            "all_raw_permutation_audits_passed": all(row["audit"]["passed"] for row in frozen),
            "all_576_original_upright_tiles_used_once": True,
            "corresponding_input_only": True,
            "no_geometry_change_or_substitution": True,
            "freshness_claim": False,
            "post_freeze_development_did_not_change_candidate": True,
        },
        "per_board": [
            {
                "filename": row["record"]["filename"],
                "input_sha256": row["record"]["input_sha256"],
                "layout_sha256": row["layout_sha256"],
                "tile_at_position": row["layout"].tolist(),
                "raw_sha256": image_digest(row["raw"]),
                "harmonized_sha256": image_digest(row["harmonized"]),
                "prediction_sha256": row["prediction_sha256"],
                "structure_diagnostics": row["structure_diagnostics"],
                "mask_diagnostics": row["mask_diagnostics"],
                "audit": row["audit"],
                "objective": row["objective"],
                "solver": row["solver"],
                "runtime_seconds": row["runtime_seconds"],
                "artifact": dict(artifact),
            }
            for row, artifact in zip(frozen, artifacts, strict=True)
        ],
    }
    payload["commitment_sha256"] = canonical_digest(payload)
    return payload


def reload_committed_predictions(
    commitment: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    if commitment.get("source_sha256") != source_hashes():
        raise RuntimeError("source changed after prediction commitment")
    payload = dict(commitment)
    claimed_digest = payload.pop("commitment_sha256", None)
    if claimed_digest != canonical_digest(payload):
        raise RuntimeError("commitment payload digest mismatch")
    if commitment.get("filenames_newline_sha256") != names_digest(records):
        raise RuntimeError("commitment filename roster drifted")
    if commitment.get("filename_input_roster_sha256") != roster_digest(records):
        raise RuntimeError("commitment filename/input roster drifted")
    if commitment.get("prediction_names") != list(PREDICTION_NAMES):
        raise RuntimeError("commitment prediction roster drifted")
    boards = commitment.get("per_board")
    if not isinstance(boards, list) or len(boards) != len(records):
        raise RuntimeError("commitment board roster malformed")
    reloaded: list[dict[str, Any]] = []
    for record, board in zip(records, boards, strict=True):
        if board.get("filename") != record["filename"]:
            raise RuntimeError("commitment filename order drifted")
        if board.get("input_sha256") != record["input_sha256"]:
            raise RuntimeError("commitment input hash roster drifted")
        artifact = board.get("artifact")
        if not isinstance(artifact, Mapping):
            raise RuntimeError("commitment artifact metadata missing")
        path = output_root / str(artifact["path"])
        if sha256_file(path) != artifact["file_sha256"]:
            raise RuntimeError(f"committed artifact file changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(artifact["array_sha256"]):
                raise RuntimeError(f"committed artifact array roster changed: {path}")
            arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        for name, value in arrays.items():
            if array_digest(value) != artifact["array_sha256"][name]:
                raise RuntimeError(f"committed artifact array changed: {path}:{name}")
        predictions = {
            name.removeprefix("prediction__"): value
            for name, value in arrays.items()
            if name.startswith("prediction__")
        }
        if set(predictions) != set(PREDICTION_NAMES):
            raise RuntimeError("committed prediction roster changed")
        for name, expected in board["prediction_sha256"].items():
            if image_digest(predictions[name]) != expected:
                raise RuntimeError(f"prediction hash mismatch: {record['filename']}:{name}")
        reloaded.append(
            {"record": dict(record), "dirty": arrays["dirty"], "predictions": predictions}
        )
    return reloaded


def numeric_gate(
    candidate_scores: Sequence[float],
    control_scores: Sequence[float],
    strong_control_scores: Sequence[float],
    safety: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    control = np.asarray(control_scores, dtype=np.float64)
    strong = np.asarray(strong_control_scores, dtype=np.float64)
    difference_a = candidate - control
    difference_b = candidate - strong
    interval_a = paired_t_interval(difference_a)
    interval_b = paired_t_interval(difference_b)
    observed = {
        "mean_rgb_ssim": float(candidate.mean()),
        "mean_gain_vs_A": float(difference_a.mean()),
        "mean_gain_vs_B": float(difference_b.mean()),
        "paired_gain_vs_A_ci95": interval_a,
        "paired_gain_vs_B_ci95": interval_b,
        "wins_vs_A": int(np.sum(difference_a > 0)),
        "ties_vs_A": int(np.sum(difference_a == 0)),
        "losses_vs_A": int(np.sum(difference_a < 0)),
        "wins_vs_B": int(np.sum(difference_b > 0)),
        "ties_vs_B": int(np.sum(difference_b == 0)),
        "losses_vs_B": int(np.sum(difference_b < 0)),
        **dict(safety),
    }
    mean_range = thresholds["mean_protected_pixel_fraction_range"]
    board_range = thresholds["every_board_protected_pixel_fraction_range"]
    checks = {
        "candidate_mean_rgb_ssim_min": observed["mean_rgb_ssim"]
        >= thresholds["candidate_mean_rgb_ssim_min"],
        "candidate_paired_gain_vs_A_ci95_lower_strictly_greater_than": interval_a["lower"]
        > thresholds["candidate_paired_gain_vs_A_ci95_lower_strictly_greater_than"],
        "candidate_paired_gain_vs_B_ci95_lower_strictly_greater_than": interval_b["lower"]
        > thresholds["candidate_paired_gain_vs_B_ci95_lower_strictly_greater_than"],
        "candidate_wins_vs_A_min": observed["wins_vs_A"] >= thresholds["candidate_wins_vs_A_min"],
        "candidate_wins_vs_B_min": observed["wins_vs_B"] >= thresholds["candidate_wins_vs_B_min"],
        "mean_within_tile_luminance_gradient_retention_vs_A_min": observed[
            "mean_luminance_gradient_retention"
        ]
        >= thresholds["mean_within_tile_luminance_gradient_retention_vs_A_min"],
        "minimum_board_within_tile_luminance_gradient_retention_vs_A_min": observed[
            "minimum_luminance_gradient_retention"
        ]
        >= thresholds["minimum_board_within_tile_luminance_gradient_retention_vs_A_min"],
        "mean_within_tile_chroma_gradient_retention_vs_A_min": observed[
            "mean_chroma_gradient_retention"
        ]
        >= thresholds["mean_within_tile_chroma_gradient_retention_vs_A_min"],
        "minimum_board_within_tile_chroma_gradient_retention_vs_A_min": observed[
            "minimum_chroma_gradient_retention"
        ]
        >= thresholds["minimum_board_within_tile_chroma_gradient_retention_vs_A_min"],
        "mean_luminance_laplacian_retention_vs_A_min": observed["mean_laplacian_retention"]
        >= thresholds["mean_luminance_laplacian_retention_vs_A_min"],
        "minimum_board_luminance_laplacian_retention_vs_A_min": observed[
            "minimum_laplacian_retention"
        ]
        >= thresholds["minimum_board_luminance_laplacian_retention_vs_A_min"],
        "mean_grid_ratio_relative_to_A_max": observed["mean_grid_ratio_relative_to_baseline"]
        <= thresholds["mean_grid_ratio_relative_to_A_max"],
        "maximum_board_grid_ratio_relative_to_A_max": observed[
            "maximum_grid_ratio_relative_to_baseline"
        ]
        <= thresholds["maximum_board_grid_ratio_relative_to_A_max"],
        "mean_protected_pixel_fraction_range": mean_range[0]
        <= observed["mean_protected_pixel_fraction"]
        <= mean_range[1],
        "every_board_protected_pixel_fraction_range": board_range[0]
        <= observed["minimum_protected_pixel_fraction"]
        and observed["maximum_protected_pixel_fraction"] <= board_range[1],
        "maximum_clipped_fraction_increase_vs_A": observed["maximum_clipped_fraction_increase"]
        <= thresholds["maximum_clipped_fraction_increase_vs_A"],
        "all_predictions_distinct_from_A_and_B_on_every_board": bool(
            observed["distinct_from_A_and_B_on_every_board"]
        ),
    }
    expected_numeric = set(thresholds) - {"manual_severe_new_artifacts_allowed"}
    if set(checks) != expected_numeric:
        raise RuntimeError("numeric gate fields differ from immutable config")
    return {"observed": observed, "checks": checks, "all_passed": all(checks.values())}


def _paste_resized(canvas: Image.Image, array: np.ndarray, x: int, y: int, size: int) -> None:
    image = Image.fromarray(array).resize((size, size), Image.Resampling.LANCZOS)
    canvas.paste(image, (x, y))


def overview_sheet(
    frozen: Sequence[Mapping[str, Any]],
    targets: Mapping[str, np.ndarray],
    output: Path,
) -> None:
    thumb, label_width, header = 96, 145, 38
    columns = ("dirty", "target", CONTROL, STRONG_CONTROL, CANDIDATE)
    canvas = Image.new(
        "RGB", (label_width + thumb * len(columns), header + thumb * len(frozen)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(columns):
        draw.text((label_width + column * thumb + 3, 8), label, fill="black")
    for row_index, row in enumerate(frozen):
        filename = str(row["record"]["filename"])
        y = header + row_index * thumb
        draw.text((3, y + 4), filename, fill="black")
        images = (
            row["dirty"],
            targets[filename],
            row["predictions"][CONTROL],
            row["predictions"][STRONG_CONTROL],
            row["predictions"][CANDIDATE],
        )
        for column, array in enumerate(images):
            _paste_resized(canvas, array, label_width + column * thumb, y, thumb)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def manual_full_canvas_pages(
    frozen: Sequence[Mapping[str, Any]],
    targets: Mapping[str, np.ndarray],
    output_root: Path,
) -> list[Path]:
    thumb, label_width, header, rows_per_page = 240, 145, 38, 10
    columns = ("target", CONTROL, STRONG_CONTROL, CANDIDATE)
    paths: list[Path] = []
    for page_start in range(0, len(frozen), rows_per_page):
        page_rows = frozen[page_start : page_start + rows_per_page]
        canvas = Image.new(
            "RGB",
            (label_width + thumb * len(columns), header + thumb * len(page_rows)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(columns):
            draw.text((label_width + column * thumb + 3, 8), label, fill="black")
        for local_index, row in enumerate(page_rows):
            filename = str(row["record"]["filename"])
            y = header + local_index * thumb
            draw.text((3, y + 4), f"{page_start + local_index}: {filename}", fill="black")
            images = (
                targets[filename],
                row["predictions"][CONTROL],
                row["predictions"][STRONG_CONTROL],
                row["predictions"][CANDIDATE],
            )
            for column, array in enumerate(images):
                _paste_resized(canvas, array, label_width + column * thumb, y, thumb)
        path = output_root / f"manual-full-canvas-page-{page_start // rows_per_page + 1:02d}.png"
        canvas.save(path)
        paths.append(path)
    return paths


def zoom_sheet(
    frozen: Sequence[Mapping[str, Any]],
    targets: Mapping[str, np.ndarray],
    output: Path,
) -> None:
    thumb, label_width, header = 360, 145, 38
    columns = ("target", CONTROL, STRONG_CONTROL, CANDIDATE)
    canvas = Image.new(
        "RGB",
        (label_width + thumb * len(columns), header + thumb * len(ZOOM_INDICES)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(columns):
        draw.text((label_width + column * thumb + 3, 8), label, fill="black")
    for output_row, board_index in enumerate(ZOOM_INDICES):
        row = frozen[board_index]
        filename = str(row["record"]["filename"])
        y = header + output_row * thumb
        draw.text((3, y + 4), f"{board_index}: {filename}", fill="black")
        images = (
            targets[filename],
            row["predictions"][CONTROL],
            row["predictions"][STRONG_CONTROL],
            row["predictions"][CANDIDATE],
        )
        for column, array in enumerate(images):
            crop = array[120:360, 120:360]
            _paste_resized(canvas, crop, label_width + column * thumb, y, thumb)
    canvas.save(output)


def score_after_commitment(
    frozen: Sequence[Mapping[str, Any]],
    *,
    targets_dir: Path,
    safety: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    scores = {name: [] for name in PREDICTION_NAMES}
    targets: dict[str, np.ndarray] = {}
    boards: list[dict[str, Any]] = []
    for index, row in enumerate(frozen, start=1):
        record = row["record"]
        filename = str(record["filename"])
        target = load_rgb_verified(targets_dir / filename, str(record["target_sha256"]))
        targets[filename] = target
        board_scores = {
            name: contest_ssim(target, row["predictions"][name]) for name in PREDICTION_NAMES
        }
        for name, value in board_scores.items():
            scores[name].append(value)
        boards.append({"filename": filename, "ssim": board_scores})
        print(f"scored {index}/{len(frozen)} {filename}", flush=True)

    gate = numeric_gate(
        scores[CANDIDATE],
        scores[CONTROL],
        scores[STRONG_CONTROL],
        safety,
        thresholds,
    )
    overview = output_root / "contact-sheet-overview.png"
    zoom = output_root / "contact-sheet-fixed-zooms.png"
    overview_sheet(frozen, targets, overview)
    pages = manual_full_canvas_pages(frozen, targets, output_root)
    zoom_sheet(frozen, targets, zoom)
    return {
        "means": {name: float(np.mean(values)) for name, values in scores.items()},
        "gate": gate,
        "numeric_gate_passed": gate["all_passed"],
        "provisional_winner": CANDIDATE if gate["all_passed"] else None,
        "manual_review_status": "required" if gate["all_passed"] else "not_authorized",
        "boards": boards,
        "contact_sheets": {
            "overview": {"path": str(overview.resolve()), "sha256": sha256_file(overview)},
            "all_60_full_canvas_pages": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in pages
            ],
            "fixed_zoom_indices_0_19_39_59": {
                "path": str(zoom.resolve()),
                "sha256": sha256_file(zoom),
            },
        },
    }


def authorized_confirmation() -> None:
    report_path = PRIMARY_ROOT / "report.json"
    review_path = PRIMARY_ROOT / "manual-review.json"
    commitment_path = PRIMARY_ROOT / "prediction-commitment.json"
    if not report_path.is_file() or not review_path.is_file() or not commitment_path.is_file():
        raise RuntimeError("confirmation requires primary report, commitment and manual review")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if report.get("config_sha256") != CONFIG_SHA256:
        raise RuntimeError("primary report config hash drifted")
    evaluation = report.get("evaluation", {})
    if evaluation.get("numeric_gate_passed") is not True:
        raise RuntimeError("primary numeric gate failed; confirmation is forbidden")
    if evaluation.get("provisional_winner") != CANDIDATE:
        raise RuntimeError("primary winner differs from frozen F")
    if review.get("passed") is not True or review.get("winner") != CANDIDATE:
        raise RuntimeError("explicit root manual PASS for frozen F is required")
    if review.get("severe_new_artifacts_count") != 0:
        raise RuntimeError("manual severe-artifact gate failed")
    if review.get("primary_report_sha256") != sha256_file(report_path):
        raise RuntimeError("manual review is not bound to the primary report")
    commitment_hash = sha256_file(commitment_path)
    if review.get("prediction_commitment_file_sha256") != commitment_hash:
        raise RuntimeError("manual review is not bound to the prediction commitment")
    if report["prediction_contract"]["commitment_file_sha256"] != commitment_hash:
        raise RuntimeError("primary report commitment hash drifted")
    for label, expected in evaluation["contact_sheets"].items():
        reviewed = review.get("contact_sheets", {}).get(label)
        if reviewed != expected:
            raise RuntimeError(f"manual review is not bound to sheet roster: {label}")
        if isinstance(expected, list):
            for item in expected:
                if sha256_file(Path(item["path"])) != item["sha256"]:
                    raise RuntimeError("primary manual sheet changed")
        elif sha256_file(Path(expected["path"])) != expected["sha256"]:
            raise RuntimeError("primary manual sheet changed")
    _, records = load_contract(MANIFEST, "primary")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    reload_committed_predictions(commitment, records, PRIMARY_ROOT)


def main() -> None:
    args = parse_args()
    config, records = load_contract(args.manifest.resolve(), args.mode)
    if args.mode == "primary":
        output_root = PRIMARY_ROOT
    else:
        authorized_confirmation()
        output_root = CONFIRMATION_ROOT
    commitment_path = output_root / "prediction-commitment.json"
    report_path = output_root / "report.json"

    if args.phase == "prepare":
        if output_root.exists():
            raise RuntimeError(f"refusing to overwrite frozen experiment directory: {output_root}")
        output_root.mkdir(parents=True)
        started = perf_counter()
        frozen = freeze_predictions(records, inputs_dir=args.inputs.resolve())
        safety = target_free_safety(frozen)
        artifacts = persist_frozen_artifacts(frozen, output_root)
        commitment = build_commitment(
            frozen,
            mode=args.mode,
            records=records,
            safety=safety,
            artifacts=artifacts,
        )
        commitment["prediction_freeze_seconds"] = perf_counter() - started
        commitment.pop("commitment_sha256")
        commitment["commitment_sha256"] = canonical_digest(commitment)
        atomic_json(commitment_path, commitment, readonly=True)
        reloaded = reload_committed_predictions(commitment, records, output_root)
        if len(reloaded) != len(records):
            raise RuntimeError("commitment readback count failed")
        print(
            json.dumps(
                {
                    "phase": "prepare",
                    "commitment": str(commitment_path),
                    "commitment_file_sha256": sha256_file(commitment_path),
                    "target_paths_opened": False,
                },
                indent=2,
            )
        )
        return

    if not commitment_path.is_file():
        raise RuntimeError("score phase requires an existing prediction commitment")
    receipt_path = output_root / "TARGETS_OPENED.receipt.json"
    if report_path.exists() or receipt_path.exists():
        raise RuntimeError("score phase is single-use and has already started")
    commitment_bytes = commitment_path.read_bytes()
    commitment = json.loads(commitment_bytes)
    if commitment.get("mode") != args.mode or commitment.get("config_sha256") != CONFIG_SHA256:
        raise RuntimeError("commitment identity drifted")
    frozen = reload_committed_predictions(commitment, records, output_root)
    commitment_file_sha256 = hashlib.sha256(commitment_bytes).hexdigest()
    atomic_json(
        receipt_path,
        {
            "schema": "aiijc-edge-protected-nlm-h28safe-target-open-receipt-v2",
            "mode": args.mode,
            "config_sha256": CONFIG_SHA256,
            "commitment_file_sha256": commitment_file_sha256,
            "historically_exposed_before_this_experiment": True,
            "meaning": "single-use transition for v2; not a freshness claim",
        },
        readonly=True,
    )
    started = perf_counter()
    evaluation = score_after_commitment(
        frozen,
        targets_dir=args.targets.resolve(),
        safety=commitment["target_free_safety"],
        thresholds=config["primary_promotion_gate"],
        output_root=output_root,
    )
    if sha256_file(commitment_path) != commitment_file_sha256:
        raise RuntimeError("commitment changed after target access")
    report = {
        "schema": "aiijc-edge-protected-nlm-h28safe-evaluation-v2",
        "status": "scored_after_verified_prediction_commitment",
        "mode": args.mode,
        "config": str(CONFIG),
        "config_sha256": CONFIG_SHA256,
        "historical_target_exposure": True,
        "freshness_claim": False,
        "post_freeze_development_change": False,
        "selection": {
            "filenames": [record["filename"] for record in records],
            "filenames_newline_sha256": names_digest(records),
            "filename_input_roster_sha256": roster_digest(records),
        },
        "prediction_contract": {
            "all_predictions_frozen_before_target_access": True,
            "all_persisted_arrays_verified_before_target_access": True,
            "commitment_path": str(commitment_path.resolve()),
            "commitment_file_sha256": commitment_file_sha256,
            "commitment_payload_sha256": commitment["commitment_sha256"],
            "target_open_receipt_sha256": sha256_file(receipt_path),
            "holdout_access": False,
            "test_access": False,
        },
        "runtime_seconds": {
            "prediction_freeze": commitment["prediction_freeze_seconds"],
            "target_scoring_and_sheets": perf_counter() - started,
        },
        "evaluation": evaluation,
    }
    atomic_json(report_path, report, readonly=True)
    print(
        json.dumps(
            {
                "phase": "score",
                "report": str(report_path),
                "means": evaluation["means"],
                "numeric_gate_passed": evaluation["numeric_gate_passed"],
                "provisional_winner": evaluation["provisional_winner"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
