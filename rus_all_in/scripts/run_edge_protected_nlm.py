#!/usr/bin/env python3
"""Evaluate the preregistered edge-protected flat-region NLM tail."""

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
    ProtectedArm,
    blend_protected,
    colored_nlm,
    image_digest,
    protected_masks,
)
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
CONFIG = PROJECT_ROOT / "configs/edge_protected_nlm_reused_calibration_preregistered_v1.json"
CONFIG_SHA256 = "63713f0da78940daae3738626cbb33e6d76f9b35daf1f6689842b5e40956c62b"
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
TARGETS = PROJECT_ROOT / "data/raw/train/targets"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/edge-protected-nlm"
CONTROL = "A_nlm_h20_baseline"
REFERENCE = "B_nlm_h28_safe_reference"
PROTECTED_ARMS = (
    ProtectedArm("C_flat_h35_t30", 35, 30.0),
    ProtectedArm("D_flat_h40_t30", 40, 30.0),
    ProtectedArm("E_flat_h40_t40", 40, 40.0),
)
PRIMARY_ROOT = OUTPUT_ROOT / "primary-calibration-offset120-count24"
CONFIRMATION_ROOT = OUTPUT_ROOT / "confirmation-calibration-offset144-count24"
SOURCE_FILES = (
    "scripts/run_edge_protected_nlm.py",
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


def load_contract(
    manifest_path: Path,
    mode: str,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise RuntimeError("edge-protected preregistration hash drifted")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config_agreement(config)
    if sha256_file(manifest_path) != config["protocol"]["manifest_sha256"]:
        raise RuntimeError("manifest file hash drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != config["protocol"]["protocol_digest"]:
        raise RuntimeError("manifest protocol digest drifted")
    if mode not in {"primary", "confirmation"}:
        raise ValueError(f"unsupported mode: {mode}")
    section_name = "primary" if mode == "primary" else "confirmation_if_and_only_if_primary_passes"
    section = config["protocol"][section_name]
    offset, count = int(section["offset"]), int(section["count"])
    records = tuple(
        select_manifest_records(
            manifest,
            "calibration",
            limit=offset + count,
            seed=EXPERIMENT_SUBSET_SEED,
            namespace=EXPERIMENT_SUBSET_NAMESPACE,
        )[offset:]
    )
    if len(records) != count:
        raise RuntimeError("panel count drifted")
    if names_digest(records) != section["filenames_newline_sha256"]:
        raise RuntimeError("filename roster drifted")
    if roster_digest(records) != section["filename_input_roster_sha256"]:
        raise RuntimeError("filename/input roster drifted")
    other_name = "confirmation_if_and_only_if_primary_passes" if mode == "primary" else "primary"
    other = config["protocol"][other_name]
    other_records = tuple(
        select_manifest_records(
            manifest,
            "calibration",
            limit=int(other["offset"]) + int(other["count"]),
            seed=EXPERIMENT_SUBSET_SEED,
            namespace=EXPERIMENT_SUBSET_NAMESPACE,
        )[int(other["offset"]) :]
    )
    if names_digest(other_records) != other["filenames_newline_sha256"]:
        raise RuntimeError("paired panel filename roster drifted")
    if roster_digest(other_records) != other["filename_input_roster_sha256"]:
        raise RuntimeError("paired panel filename/input roster drifted")
    overlap = {str(record["filename"]) for record in records} & {
        str(record["filename"]) for record in other_records
    }
    if len(overlap) != config["protocol"]["primary_confirmation_filename_overlap"]:
        raise RuntimeError("primary/confirmation overlap drifted")
    return config, records


def validate_config_agreement(config: Mapping[str, Any]) -> None:
    """Fail closed when executable constants differ from the immutable preregistration."""

    protocol = config["protocol"]
    if protocol["selector_namespace"] != EXPERIMENT_SUBSET_NAMESPACE:
        raise RuntimeError("selector namespace differs from preregistration")
    if protocol["selector_seed"] != EXPERIMENT_SUBSET_SEED:
        raise RuntimeError("selector seed differs from preregistration")
    if protocol["split"] != "calibration":
        raise RuntimeError("only the preregistered reused calibration split is allowed")
    if (protocol["primary"]["offset"], protocol["primary"]["count"]) != (120, 24):
        raise RuntimeError("primary panel bounds differ from executable output contract")
    confirmation = protocol["confirmation_if_and_only_if_primary_passes"]
    if (confirmation["offset"], confirmation["count"]) != (144, 24):
        raise RuntimeError("confirmation panel bounds differ from executable output contract")
    history = protocol["historical_exposure"]
    if history["freshness_claim"] is not False:
        raise RuntimeError("freshness claim is forbidden for reused calibration")
    if not (
        history["all_primary_records_previously_exposed"]
        and history["all_confirmation_records_previously_exposed"]
    ):
        raise RuntimeError("historical target exposure acknowledgement drifted")

    expected_arms = [
        {
            "name": CONTROL,
            "kind": "global_single_pass",
            "h": 20,
            "h_color": 20,
            "role": "control",
        },
        {
            "name": REFERENCE,
            "kind": "global_single_pass",
            "h": 28,
            "h_color": 28,
            "role": "diagnostic",
        },
        *[
            {
                "name": arm.name,
                "kind": "edge_protected_blend",
                "aggressive_h": arm.aggressive_h,
                "aggressive_h_color": arm.aggressive_h,
                "sobel_threshold": arm.sobel_threshold,
                "role": "candidate",
            }
            for arm in PROTECTED_ARMS
        ],
    ]
    if config["arms"] != expected_arms:
        raise RuntimeError("executable arm roster differs from preregistration")
    fixed = config["fixed_pipeline"]
    if fixed["nlm_template_window_size"] != TEMPLATE_WINDOW:
        raise RuntimeError("NLM template window differs from preregistration")
    if fixed["nlm_search_window_size"] != SEARCH_WINDOW:
        raise RuntimeError("NLM search window differs from preregistration")
    if config["mask_algorithm"]["target_or_filename_dependency"] is not False:
        raise RuntimeError("mask target/filename independence contract drifted")
    if config["paired_ci"]["confidence"] != 0.95:
        raise RuntimeError("paired CI confidence differs from executable gate")
    if config["manual_review"]["fixed_zoom_board_indices"] != [0, 7, 15, 23]:
        raise RuntimeError("manual-review zoom roster differs from executable sheet")


def source_hashes() -> dict[str, str]:
    files = (str(CONFIG.relative_to(PROJECT_ROOT)), *SOURCE_FILES)
    return {name: sha256_file(PROJECT_ROOT / name) for name in files}


def authorized_winner() -> ProtectedArm:
    report_path = PRIMARY_ROOT / "report.json"
    review_path = PRIMARY_ROOT / "manual-review.json"
    if not report_path.is_file() or not review_path.is_file():
        raise RuntimeError("confirmation requires primary report and explicit manual review")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if report.get("config_sha256") != CONFIG_SHA256:
        raise RuntimeError("primary report config hash drifted")
    evaluation = report.get("evaluation", {})
    if evaluation.get("numeric_gate_passed") is not True or review.get("passed") is not True:
        raise RuntimeError("primary numeric/manual gate failed; confirmation is forbidden")
    winner_name = evaluation.get("provisional_winner")
    winner = next((arm for arm in PROTECTED_ARMS if arm.name == winner_name), None)
    if winner is None or review.get("winner") != winner.name:
        raise RuntimeError("manual review winner differs from the frozen numeric winner")
    commitment = Path(str(report["prediction_contract"]["commitment_path"]))
    if sha256_file(commitment) != report["prediction_contract"]["commitment_file_sha256"]:
        raise RuntimeError("primary commitment changed")
    return winner


def freeze_predictions(
    records: Sequence[Mapping[str, Any]],
    *,
    inputs_dir: Path,
    protected_arms: Sequence[ProtectedArm],
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
        safe = colored_nlm(harmonized, 20)
        reference = colored_nlm(harmonized, 28)
        aggressive = {
            h: colored_nlm(harmonized, h)
            for h in sorted({arm.aggressive_h for arm in protected_arms})
        }
        predictions: dict[str, np.ndarray] = {CONTROL: safe, REFERENCE: reference}
        mask_diagnostics: dict[str, dict[str, float]] = {}
        dilated_masks: dict[str, np.ndarray] = {}
        soft_masks: dict[str, np.ndarray] = {}
        for arm in protected_arms:
            dilated, soft, protected_fraction = protected_masks(
                safe,
                sobel_threshold=arm.sobel_threshold,
            )
            prediction, diagnostics = blend_protected(
                safe,
                aggressive[arm.aggressive_h],
                sobel_threshold=arm.sobel_threshold,
            )
            if not np.isclose(
                protected_fraction,
                diagnostics["binary_dilated_protected_fraction"],
            ):
                raise RuntimeError("mask diagnostics do not reproduce")
            predictions[arm.name] = prediction
            mask_diagnostics[arm.name] = diagnostics
            dilated_masks[arm.name] = np.ascontiguousarray(dilated)
            soft_masks[arm.name] = np.ascontiguousarray(soft)
        hashes = {name: image_digest(value) for name, value in predictions.items()}
        structures = {name: structure_diagnostics(value) for name, value in predictions.items()}
        frozen.append(
            {
                "record": dict(record),
                "dirty": dirty,
                "layout": layout,
                "layout_sha256": layout_digest(layout),
                "raw": raw,
                "harmonized": harmonized,
                "independent_nlm": {
                    "h20": safe,
                    "h28": reference,
                    **{f"h{h}": image for h, image in aggressive.items()},
                },
                "predictions": predictions,
                "dilated_masks": dilated_masks,
                "soft_masks": soft_masks,
                "prediction_sha256": hashes,
                "structure_diagnostics": structures,
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


def target_free_safety(
    frozen: Sequence[Mapping[str, Any]],
    protected_arms: Sequence[ProtectedArm],
) -> dict[str, Any]:
    baseline = [row["structure_diagnostics"][CONTROL] for row in frozen]
    output: dict[str, Any] = {}
    for arm in protected_arms:
        candidate = [row["structure_diagnostics"][arm.name] for row in frozen]
        summary = safety_summary(candidate, baseline)
        protected = np.asarray(
            [
                row["mask_diagnostics"][arm.name]["binary_dilated_protected_fraction"]
                for row in frozen
            ]
        )
        clipped_increase = np.asarray(
            [
                row["structure_diagnostics"][arm.name]["clipped_fraction"]
                - row["structure_diagnostics"][CONTROL]["clipped_fraction"]
                for row in frozen
            ]
        )
        output[arm.name] = {
            **summary,
            "mean_protected_pixel_fraction": float(protected.mean()),
            "minimum_protected_pixel_fraction": float(protected.min()),
            "maximum_protected_pixel_fraction": float(protected.max()),
            "maximum_clipped_fraction_increase": float(clipped_increase.max()),
            "distinct_from_A_on_every_board": all(
                row["prediction_sha256"][arm.name] != row["prediction_sha256"][CONTROL]
                for row in frozen
            ),
        }
    return output


def persist_frozen_artifacts(
    frozen: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    """Persist every preregistered array before target access."""

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
        }
        arrays.update(
            {f"independent_nlm__{name}": value for name, value in row["independent_nlm"].items()}
        )
        arrays.update({f"prediction__{name}": value for name, value in row["predictions"].items()})
        arrays.update(
            {f"dilated_mask__{name}": value for name, value in row["dilated_masks"].items()}
        )
        arrays.update({f"soft_mask__{name}": value for name, value in row["soft_masks"].items()})
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


def reload_committed_predictions(
    commitment: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    """Verify every persisted array and reload scoring inputs from disk."""

    if commitment.get("source_sha256") != source_hashes():
        raise RuntimeError("source changed after prediction commitment")
    payload = dict(commitment)
    claimed_digest = payload.pop("commitment_sha256", None)
    if claimed_digest != canonical_digest(payload):
        raise RuntimeError("commitment payload digest mismatch")
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
            names = set(archive.files)
            if names != set(artifact["array_sha256"]):
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
        if set(predictions) != {CONTROL, REFERENCE, *[arm.name for arm in PROTECTED_ARMS]}:
            # Confirmation commitments contain only one protected winner.
            expected_protected = {arm["name"] for arm in commitment["protected_arms"]}
            if set(predictions) != {CONTROL, REFERENCE, *expected_protected}:
                raise RuntimeError("committed prediction roster changed")
        for name, expected in board["prediction_sha256"].items():
            if image_digest(predictions[name]) != expected:
                raise RuntimeError(f"prediction hash mismatch: {record['filename']}:{name}")
        reloaded.append(
            {
                "record": dict(record),
                "dirty": arrays["dirty"],
                "predictions": predictions,
            }
        )
    return reloaded


def build_commitment(
    frozen: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    records: Sequence[Mapping[str, Any]],
    protected_arms: Sequence[ProtectedArm],
    safety: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "aiijc-edge-protected-nlm-prediction-commitment-v1",
        "mode": mode,
        "config": str(CONFIG),
        "config_sha256": CONFIG_SHA256,
        "source_sha256": source_hashes(),
        "filenames": [record["filename"] for record in records],
        "filenames_newline_sha256": names_digest(records),
        "filename_input_roster_sha256": roster_digest(records),
        "protected_arms": [arm.__dict__ for arm in protected_arms],
        "target_free_safety": safety,
        "contract": {
            "target_paths_opened": False,
            "all_predictions_frozen_before_target_access": True,
            "all_raw_permutation_audits_passed": all(row["audit"]["passed"] for row in frozen),
            "all_576_original_upright_tiles_used_once": True,
            "corresponding_input_only": True,
            "no_geometry_change_or_substitution": True,
            "freshness_claim": False,
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


def numeric_gate(
    scores: Sequence[float],
    baseline: Sequence[float],
    safety: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    control = np.asarray(baseline, dtype=np.float64)
    difference = values - control
    interval = paired_t_interval(difference)
    observed = {
        "mean_rgb_ssim": float(values.mean()),
        "mean_gain_vs_A": float(difference.mean()),
        "paired_gain_vs_A_ci95": interval,
        "wins_vs_A": int(np.sum(difference > 0)),
        "ties_vs_A": int(np.sum(difference == 0)),
        "losses_vs_A": int(np.sum(difference < 0)),
        **dict(safety),
    }
    mean_range = thresholds["mean_protected_pixel_fraction_range"]
    board_range = thresholds["every_board_protected_pixel_fraction_range"]
    checks = {
        "winner_mean_rgb_ssim_min": observed["mean_rgb_ssim"]
        >= thresholds["winner_mean_rgb_ssim_min"],
        "winner_paired_gain_vs_A_ci95_lower_strictly_greater_than": interval["lower"]
        > thresholds["winner_paired_gain_vs_A_ci95_lower_strictly_greater_than"],
        "winner_wins_vs_A_min": observed["wins_vs_A"] >= thresholds["winner_wins_vs_A_min"],
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
        "all_predictions_distinct_from_A_on_every_board": bool(
            observed["distinct_from_A_on_every_board"]
        ),
    }
    expected_numeric_fields = set(thresholds) - {"manual_severe_new_artifacts_allowed"}
    if set(checks) != expected_numeric_fields:
        raise RuntimeError("numeric gate fields differ from the immutable preregistration")
    return {"observed": observed, "checks": checks, "all_passed": all(checks.values())}


def contact_sheet(
    frozen: Sequence[Mapping[str, Any]],
    targets: Mapping[str, np.ndarray],
    winner: str,
    output: Path,
) -> None:
    thumb, label_width, header = 120, 145, 38
    columns = ("dirty", "target", CONTROL, REFERENCE, winner)
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
            row["predictions"][REFERENCE],
            row["predictions"][winner],
        )
        for column, array in enumerate(images):
            image = Image.fromarray(array).resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(image, (label_width + column * thumb, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def zoom_contact_sheet(
    frozen: Sequence[Mapping[str, Any]],
    targets: Mapping[str, np.ndarray],
    winner: str,
    output: Path,
) -> None:
    indices = (0, 7, 15, 23)
    thumb, label_width, header = 220, 145, 38
    columns = ("target", CONTROL, REFERENCE, winner)
    canvas = Image.new(
        "RGB",
        (label_width + thumb * len(columns), header + thumb * len(indices)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(columns):
        draw.text((label_width + column * thumb + 3, 8), label, fill="black")
    for output_row, board_index in enumerate(indices):
        row = frozen[board_index]
        filename = str(row["record"]["filename"])
        y = header + output_row * thumb
        draw.text((3, y + 4), filename, fill="black")
        images = (
            targets[filename],
            row["predictions"][CONTROL],
            row["predictions"][REFERENCE],
            row["predictions"][winner],
        )
        for column, array in enumerate(images):
            crop = array[120:360, 120:360]
            image = Image.fromarray(crop).resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(image, (label_width + column * thumb, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def score_after_commitment(
    frozen: Sequence[Mapping[str, Any]],
    *,
    targets_dir: Path,
    protected_arms: Sequence[ProtectedArm],
    safety: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    sheet_path: Path,
) -> dict[str, Any]:
    arm_names = (CONTROL, REFERENCE, *(arm.name for arm in protected_arms))
    scores = {name: [] for name in arm_names}
    targets: dict[str, np.ndarray] = {}
    boards: list[dict[str, Any]] = []
    for index, row in enumerate(frozen, start=1):
        record = row["record"]
        filename = str(record["filename"])
        target = load_rgb_verified(targets_dir / filename, str(record["target_sha256"]))
        targets[filename] = target
        board_scores = {name: contest_ssim(target, row["predictions"][name]) for name in arm_names}
        for name, value in board_scores.items():
            scores[name].append(value)
        boards.append({"filename": filename, "ssim": board_scores})
        print(f"scored {index}/{len(frozen)} {filename}", flush=True)
    gates = {
        arm.name: numeric_gate(scores[arm.name], scores[CONTROL], safety[arm.name], thresholds)
        for arm in protected_arms
    }
    passing = [arm for arm in protected_arms if gates[arm.name]["all_passed"]]
    ranking = sorted(
        protected_arms,
        key=lambda arm: (
            -float(np.mean(scores[arm.name])),
            arm.aggressive_h,
            arm.sobel_threshold,
            arm.name,
        ),
    )
    winner = next((arm for arm in ranking if arm in passing), None)
    display = winner or ranking[0]
    contact_sheet(frozen, targets, display.name, sheet_path)
    zoom_path = sheet_path.with_name("contact-sheet-fixed-zooms.png")
    zoom_contact_sheet(frozen, targets, display.name, zoom_path)
    return {
        "means": {name: float(np.mean(value)) for name, value in scores.items()},
        "gates": gates,
        "numeric_passing_arms": [arm.name for arm in passing],
        "provisional_winner": winner.name if winner else None,
        "diagnostic_best_mean_arm": ranking[0].name,
        "numeric_gate_passed": winner is not None,
        "manual_review_status": "required" if winner else "not_authorized_after_numeric_fail",
        "boards": boards,
        "contact_sheets": {
            "all_24_full_canvas_pairs": {
                "path": str(sheet_path.resolve()),
                "sha256": sha256_file(sheet_path),
            },
            "fixed_zoom_indices_0_7_15_23": {
                "path": str(zoom_path.resolve()),
                "sha256": sha256_file(zoom_path),
            },
        },
    }


def main() -> None:
    args = parse_args()
    config, records = load_contract(args.manifest.resolve(), args.mode)
    if args.mode == "primary":
        protected_arms = PROTECTED_ARMS
        output_root = PRIMARY_ROOT
    else:
        protected_arms = (authorized_winner(),)
        output_root = CONFIRMATION_ROOT
    commitment_path = output_root / "prediction-commitment.json"
    report_path = output_root / "report.json"
    if args.phase == "prepare":
        if output_root.exists():
            raise RuntimeError(f"refusing to overwrite frozen experiment directory: {output_root}")
        output_root.mkdir(parents=True)
        started = perf_counter()
        frozen = freeze_predictions(
            records,
            inputs_dir=args.inputs.resolve(),
            protected_arms=protected_arms,
        )
        safety = target_free_safety(frozen, protected_arms)
        artifacts = persist_frozen_artifacts(frozen, output_root)
        commitment = build_commitment(
            frozen,
            mode=args.mode,
            records=records,
            protected_arms=protected_arms,
            safety=safety,
            artifacts=artifacts,
        )
        commitment["prediction_freeze_seconds"] = perf_counter() - started
        # Recompute after adding the final target-free runtime field.
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
    if commitment.get("filenames_newline_sha256") != names_digest(records):
        raise RuntimeError("commitment roster drifted")
    if commitment.get("filename_input_roster_sha256") != roster_digest(records):
        raise RuntimeError("commitment filename/input roster drifted")
    frozen = reload_committed_predictions(commitment, records, output_root)
    commitment_file_sha256 = hashlib.sha256(commitment_bytes).hexdigest()
    atomic_json(
        receipt_path,
        {
            "schema": "aiijc-edge-protected-nlm-target-open-receipt-v1",
            "mode": args.mode,
            "config_sha256": CONFIG_SHA256,
            "commitment_file_sha256": commitment_file_sha256,
            "historically_exposed_before_this_experiment": True,
            "meaning": "single-use transition for this experiment; not a freshness claim",
        },
        readonly=True,
    )
    started = perf_counter()
    evaluation = score_after_commitment(
        frozen,
        targets_dir=args.targets.resolve(),
        protected_arms=protected_arms,
        safety=commitment["target_free_safety"],
        thresholds=config["primary_promotion_gate"],
        sheet_path=output_root / "contact-sheet.png",
    )
    if sha256_file(commitment_path) != commitment_file_sha256:
        raise RuntimeError("commitment changed after target access")
    report = {
        "schema": "aiijc-edge-protected-nlm-evaluation-v1",
        "status": "scored_after_verified_prediction_commitment",
        "mode": args.mode,
        "config": str(CONFIG),
        "config_sha256": CONFIG_SHA256,
        "historical_target_exposure": True,
        "freshness_claim": False,
        "selection": {
            "filenames": [record["filename"] for record in records],
            "filenames_newline_sha256": names_digest(records),
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
            "target_scoring": perf_counter() - started,
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
