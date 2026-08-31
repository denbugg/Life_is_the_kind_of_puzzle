#!/usr/bin/env python3
"""Run the frozen TRAIN-only DRUNet40 matcher selection/verification diagnostic."""

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
import torch

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.drunet_matcher import (
    ARM_NAMES,
    BASELINE,
    FUSION_NAMES,
    FUSION_WEIGHTS,
    matcher_score_roster,
    solve_matcher_roster,
)
from aiijc_puzzle.edge_protected_nlm import colored_nlm, image_digest
from aiijc_puzzle.edge_protected_nlm_v2 import blend_h28safe_h40flat
from aiijc_puzzle.frozen_final_evaluator import _validate_method_configs, load_rgb_verified
from aiijc_puzzle.legacy_upgrade import layout_digest
from aiijc_puzzle.nlm_luma_chroma import paired_t_interval
from aiijc_puzzle.postassembly_harmonizer import (
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.pretrained_tile_denoiser import load_drunet_color, render_drunet_tiles
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    GRID_SIZE,
    assemble_tiles,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/drunet_matcher_train_diagnostic_preregistered_v1.json"
CONFIG_SHA256 = "781c177c391b30e0a261c12ffd5723928d6f031eb468815279c4df4392cef62e"
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
TARGETS = PROJECT_ROOT / "data/raw/train/targets"
ASSET_ROOT = PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f"
CHECKPOINT = ASSET_ROOT / "drunet_color.pth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/drunet-matcher/train-offset512-count16"
COMMITMENT_PATH = OUTPUT_ROOT / "prediction-commitment.json"
REPORT_PATH = OUTPUT_ROOT / "report.json"
RECEIPT_PATH = OUTPUT_ROOT / "TARGETS_OPENED.receipt.json"
SELECTION_DECISION_PATH = OUTPUT_ROOT / "selection-decision.json"
VERIFICATION_RECEIPT_PATH = OUTPUT_ROOT / "VERIFICATION_TARGETS_OPENED.receipt.json"

SIGMA = 40.0
BATCH_SIZE = 144
EDGE_BUDGET = 96
SELECTION_COUNT = 8
VERIFICATION_COUNT = 8
TAIL_H28 = "h28"
TAIL_F = "F_h28safe_flat_h40_t40"
TAIL_NAMES = (TAIL_H28, TAIL_F)
EXPECTED_ASSET_SHA256 = {
    "LICENSE": "448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5",
    "drunet_color.pth": "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4",
    "models/basicblock.py": "48406db8867394ac5ae233ebeec7711ac10acfc3a6bbf0072c33aa77d659b6fd",
    "models/network_unet.py": "8043b6350f1589d5f08892e3be0b4d12c5a502058014285107b7360696d12bf5",
}
EXPECTED_PARAMETER_COUNT = 32_640_960
SOURCE_FILES = (
    "scripts/run_drunet_matcher_train_diagnostic.py",
    "src/aiijc_puzzle/drunet_matcher.py",
    "src/aiijc_puzzle/pretrained_tile_denoiser.py",
    "src/aiijc_puzzle/edge_protected_nlm.py",
    "src/aiijc_puzzle/edge_protected_nlm_v2.py",
    "src/aiijc_puzzle/frozen_final_evaluator.py",
    "src/aiijc_puzzle/candidate_supply.py",
    "src/aiijc_puzzle/legacy_upgrade.py",
    "src/aiijc_puzzle/postassembly_harmonizer.py",
    "src/aiijc_puzzle/compliant_atlas_decoder.py",
    "src/aiijc_puzzle/nlm_luma_chroma.py",
    "src/aiijc_puzzle/protocol.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "score"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
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


def choose_device(requested: str) -> torch.device:
    name = requested
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def verify_assets() -> dict[str, str]:
    observed = {relative: sha256_file(ASSET_ROOT / relative) for relative in EXPECTED_ASSET_SHA256}
    if observed != EXPECTED_ASSET_SHA256:
        raise RuntimeError("official KAIR asset hashes drifted")
    return observed


def validate_config(config: Mapping[str, Any]) -> None:
    protocol = config["protocol"]
    expected_protocol = {
        "selector_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selector_seed": EXPERIMENT_SUBSET_SEED,
        "split": "train",
        "offset": 512,
        "count": 16,
    }
    for field, expected in expected_protocol.items():
        if protocol[field] != expected:
            raise RuntimeError(f"config protocol field drifted: {field}")
    model = config["official_model"]
    if model["sigma_255"] != SIGMA or model["batch_size"] != BATCH_SIZE:
        raise RuntimeError("DRUNet runtime differs from config")
    matcher = config["matcher"]
    if tuple(matcher["fusion_weights"]) != FUSION_WEIGHTS:
        raise RuntimeError("fusion weight roster differs from config")
    if matcher["pure_drunet_weight"] != 1.0:
        raise RuntimeError("pure DRUNet diagnostic weight drifted")
    gate = config["verification_gate_last_8"]
    expected_gate = {
        "selected_fusion_mean_F_ssim_gain_vs_dirty_bilateral_strictly_positive",
        "selected_fusion_F_ssim_paired_t_ci95_lower_strictly_positive",
        "selected_fusion_F_ssim_wins_vs_dirty_bilateral_min",
        "selected_fusion_mean_h28_ssim_gain_vs_dirty_bilateral_strictly_positive",
        "selected_fusion_h28_ssim_wins_vs_dirty_bilateral_min",
        "selected_fusion_mean_exact_adjacency_gain_vs_dirty_bilateral_strictly_positive",
        "selected_fusion_exact_adjacency_wins_vs_dirty_bilateral_min",
        "selected_fusion_translation_aligned_placement_gain_vs_dirty_bilateral_strictly_positive",
        "selected_fusion_direct_placement_gain_vs_dirty_bilateral_nonnegative",
        "all_original_tile_permutation_audits_passed",
    }
    if set(gate) != expected_gate:
        raise RuntimeError("verification gate field roster drifted")


def load_contract(
    manifest_path: Path,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise RuntimeError("train diagnostic config hash drifted")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    protocol = config["protocol"]
    if sha256_file(manifest_path) != protocol["manifest_sha256"]:
        raise RuntimeError("manifest hash drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != protocol["protocol_digest"]:
        raise RuntimeError("manifest protocol digest drifted")
    offset, count = int(protocol["offset"]), int(protocol["count"])
    records = tuple(
        select_manifest_records(
            manifest,
            "train",
            limit=offset + count,
            seed=EXPERIMENT_SUBSET_SEED,
            namespace=EXPERIMENT_SUBSET_NAMESPACE,
        )[offset : offset + count]
    )
    if len(records) != count:
        raise RuntimeError("train roster count drifted")
    if names_digest(records) != protocol["filenames_newline_sha256"]:
        raise RuntimeError("train filename roster drifted")
    if roster_digest(records) != protocol["filename_input_roster_sha256"]:
        raise RuntimeError("train filename/input roster drifted")
    if names_digest(records[:SELECTION_COUNT]) != protocol["selection_first_8_filenames_sha256"]:
        raise RuntimeError("selection roster drifted")
    if names_digest(records[SELECTION_COUNT:]) != protocol["verification_last_8_filenames_sha256"]:
        raise RuntimeError("verification roster drifted")
    return config, records


def source_hashes() -> dict[str, str]:
    files = (str(CONFIG.relative_to(PROJECT_ROOT)), *SOURCE_FILES)
    return {name: sha256_file(PROJECT_ROOT / name) for name in files}


def infer_board(
    dirty: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    original_tiles = split_tiles(dirty)
    drunet_tiles, render_diagnostics = render_drunet_tiles(
        model,
        original_tiles,
        sigma_255=SIGMA,
        device=device,
        batch_size=BATCH_SIZE,
    )
    scores = matcher_score_roster(original_tiles, drunet_tiles)
    solved = solve_matcher_roster(scores, edge_budget=EDGE_BUDGET)
    rgb_config, luma_config, method_hashes = _validate_method_configs()
    variants: dict[str, dict[str, Any]] = {}
    for name in ARM_NAMES:
        result = solved[name]
        layout = np.asarray(result.layout, dtype=np.int32)
        ordered = np.ascontiguousarray(original_tiles[layout])
        raw = assemble_tiles(ordered)
        audit = audit_raw_permutation(
            dirty,
            raw,
            layout,
            restoration_applied_after_audit=True,
        )
        if not audit.passed:
            raise RuntimeError(f"strict original-tile audit failed for {name}")
        offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered, rgb_config)
        rgb_tiles = apply_rgb_offsets(ordered, offsets)
        gains, luma_diagnostics = seam_graph_luminance_gains(rgb_tiles, luma_config)
        harmonized = assemble_tiles(apply_luminance_gains(rgb_tiles, gains))
        h20 = colored_nlm(harmonized, 20)
        h28 = colored_nlm(harmonized, 28)
        h40 = colored_nlm(harmonized, 40)
        candidate, _, _, mask_diagnostics = blend_h28safe_h40flat(h20, h28, h40)
        variants[name] = {
            "layout": layout,
            "raw": raw,
            "harmonized": harmonized,
            "predictions": {TAIL_H28: h28, TAIL_F: candidate},
            "audit": audit.as_dict(),
            "layout_sha256": layout_digest(layout),
            "raw_sha256": image_digest(raw),
            "harmonized_sha256": image_digest(harmonized),
            "prediction_sha256": {
                TAIL_H28: image_digest(h28),
                TAIL_F: image_digest(candidate),
            },
            "objective": float(result.objective),
            "solver": result.solver,
            "mask_diagnostics": mask_diagnostics,
            "harmonizer_diagnostics": {
                "rgb": rgb_diagnostics,
                "luma": luma_diagnostics,
                "method_config_sha256": method_hashes,
            },
        }
    return {
        "dirty": dirty,
        "drunet_tiles": drunet_tiles,
        "render_diagnostics": render_diagnostics.as_dict(),
        "variants": variants,
    }


def freeze_predictions(
    records: Sequence[Mapping[str, Any]],
    *,
    inputs_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    model = load_drunet_color(CHECKPOINT, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("DRUNet architecture parameter count drifted")
    frozen: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        started = perf_counter()
        filename = str(record["filename"])
        dirty = load_rgb_verified(inputs_dir / filename, str(record["input_sha256"]))
        inference = infer_board(dirty, model, device)
        inference["record"] = dict(record)
        inference["runtime_seconds"] = perf_counter() - started
        frozen.append(inference)
        print(f"froze {index}/{len(records)} {filename}", flush=True)
    return frozen


def persist_artifacts(
    frozen: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    artifact_root = output_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=False)
    metadata: list[dict[str, Any]] = []
    for row in frozen:
        arrays: dict[str, np.ndarray] = {
            "dirty": row["dirty"],
            "drunet_matcher_tiles_only": row["drunet_tiles"],
        }
        for arm in ARM_NAMES:
            variant = row["variants"][arm]
            arrays[f"layout__{arm}"] = variant["layout"]
            for tail in TAIL_NAMES:
                arrays[f"prediction__{arm}__{tail}"] = variant["predictions"][tail]
        filename = str(row["record"]["filename"])
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
    records: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    assets: Mapping[str, str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "aiijc-drunet40-matcher-train-commitment-v1",
        "config": str(CONFIG),
        "config_sha256": CONFIG_SHA256,
        "source_sha256": source_hashes(),
        "asset_sha256": dict(assets),
        "device": str(device),
        "sigma_255": SIGMA,
        "batch_size": BATCH_SIZE,
        "arm_names": list(ARM_NAMES),
        "tail_names": list(TAIL_NAMES),
        "filenames": [record["filename"] for record in records],
        "filenames_newline_sha256": names_digest(records),
        "filename_input_roster_sha256": roster_digest(records),
        "contract": {
            "target_paths_opened": False,
            "all_layouts_and_predictions_frozen_before_target_access": True,
            "drunet_pixels_used_for_matcher_scores_only": True,
            "all_output_pixels_from_original_dirty_tiles": True,
            "all_raw_permutation_audits_passed": all(
                variant["audit"]["passed"] for row in frozen for variant in row["variants"].values()
            ),
            "freshness_claim": False,
            "calibration_access": False,
        },
        "per_board": [
            {
                "filename": row["record"]["filename"],
                "input_sha256": row["record"]["input_sha256"],
                "render_diagnostics": row["render_diagnostics"],
                "runtime_seconds": row["runtime_seconds"],
                "variants": {
                    arm: {
                        "layout_sha256": row["variants"][arm]["layout_sha256"],
                        "tile_at_position": row["variants"][arm]["layout"].tolist(),
                        "raw_sha256": row["variants"][arm]["raw_sha256"],
                        "harmonized_sha256": row["variants"][arm]["harmonized_sha256"],
                        "prediction_sha256": row["variants"][arm]["prediction_sha256"],
                        "audit": row["variants"][arm]["audit"],
                        "objective": row["variants"][arm]["objective"],
                        "solver": row["variants"][arm]["solver"],
                        "mask_diagnostics": row["variants"][arm]["mask_diagnostics"],
                    }
                    for arm in ARM_NAMES
                },
                "artifact": dict(artifact),
            }
            for row, artifact in zip(frozen, artifacts, strict=True)
        ],
    }
    payload["commitment_sha256"] = canonical_digest(payload)
    return payload


def reload_commitment(
    commitment: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if commitment.get("source_sha256") != source_hashes():
        raise RuntimeError("source changed after commitment")
    if commitment.get("asset_sha256") != verify_assets():
        raise RuntimeError("official DRUNet assets changed after commitment")
    payload = dict(commitment)
    claimed = payload.pop("commitment_sha256", None)
    if claimed != canonical_digest(payload):
        raise RuntimeError("commitment payload digest mismatch")
    if commitment.get("filenames_newline_sha256") != names_digest(records):
        raise RuntimeError("commitment filename roster drifted")
    if commitment.get("filename_input_roster_sha256") != roster_digest(records):
        raise RuntimeError("commitment filename/input roster drifted")
    if commitment.get("arm_names") != list(ARM_NAMES):
        raise RuntimeError("commitment arm roster drifted")
    boards = commitment.get("per_board")
    if not isinstance(boards, list) or len(boards) != len(records):
        raise RuntimeError("commitment board roster malformed")
    output: list[dict[str, Any]] = []
    for record, board in zip(records, boards, strict=True):
        if board.get("filename") != record["filename"]:
            raise RuntimeError("commitment filename order drifted")
        if board.get("input_sha256") != record["input_sha256"]:
            raise RuntimeError("commitment input hash drifted")
        artifact = board["artifact"]
        path = OUTPUT_ROOT / str(artifact["path"])
        if sha256_file(path) != artifact["file_sha256"]:
            raise RuntimeError("committed artifact file changed")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(artifact["array_sha256"]):
                raise RuntimeError("committed artifact array roster changed")
            arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        for name, value in arrays.items():
            if array_digest(value) != artifact["array_sha256"][name]:
                raise RuntimeError(f"committed artifact array changed: {name}")
        variants: dict[str, dict[str, Any]] = {}
        for arm in ARM_NAMES:
            layout = arrays[f"layout__{arm}"]
            predictions = {tail: arrays[f"prediction__{arm}__{tail}"] for tail in TAIL_NAMES}
            if layout_digest(layout) != board["variants"][arm]["layout_sha256"]:
                raise RuntimeError("committed layout hash mismatch")
            for tail in TAIL_NAMES:
                if (
                    image_digest(predictions[tail])
                    != board["variants"][arm]["prediction_sha256"][tail]
                ):
                    raise RuntimeError("committed prediction hash mismatch")
            variants[arm] = {"layout": layout, "predictions": predictions}
        output.append({"record": dict(record), "dirty": arrays["dirty"], "variants": variants})
    return output


def layout_metrics(layout: np.ndarray, recovered: Any) -> dict[str, float]:
    truth = recovered.dirty_at_position
    position_of_dirty = recovered.position_of_dirty
    predicted_positions = np.empty_like(layout)
    predicted_positions[layout] = np.arange(len(layout))
    shifts: dict[tuple[int, int], int] = {}
    for tile, predicted in enumerate(predicted_positions):
        true = int(position_of_dirty[tile])
        predicted_row, predicted_column = divmod(int(predicted), GRID_SIZE)
        true_row, true_column = divmod(true, GRID_SIZE)
        shift = (true_row - predicted_row, true_column - predicted_column)
        shifts[shift] = shifts.get(shift, 0) + 1
    grid = layout.reshape(GRID_SIZE, GRID_SIZE)
    left = position_of_dirty[grid[:, :-1]]
    right = position_of_dirty[grid[:, 1:]]
    top = position_of_dirty[grid[:-1]]
    bottom = position_of_dirty[grid[1:]]
    right_accuracy = np.mean((right - left == 1) & (right // GRID_SIZE == left // GRID_SIZE))
    down_accuracy = np.mean(bottom - top == GRID_SIZE)
    return {
        "direct_placement": float(np.mean(layout == truth)),
        "translation_aligned_placement": float(max(shifts.values()) / len(layout)),
        "right_adjacency": float(right_accuracy),
        "down_adjacency": float(down_accuracy),
        "adjacency": float(0.5 * (right_accuracy + down_accuracy)),
        "label_mapping_mean_margin": float(recovered.margin_at_position.mean()),
    }


def safe_t_interval(values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if np.all(vector == vector[0]):
        return {
            "mean": float(vector[0]),
            "lower": float(vector[0]),
            "upper": float(vector[0]),
            "confidence": 0.95,
            "degrees_of_freedom": len(vector) - 1,
        }
    return paired_t_interval(vector)


def summarize_panel(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "direct_placement",
        "translation_aligned_placement",
        "right_adjacency",
        "down_adjacency",
        "adjacency",
        "h28_ssim",
        "F_ssim",
    )
    output: dict[str, Any] = {}
    baseline_values = {
        metric: np.asarray([row["variants"][BASELINE][metric] for row in rows])
        for metric in metrics
    }
    for arm in ARM_NAMES:
        values = {
            metric: np.asarray([row["variants"][arm][metric] for row in rows]) for metric in metrics
        }
        summary: dict[str, Any] = {metric: float(value.mean()) for metric, value in values.items()}
        summary["comparison_baseline"] = BASELINE
        for metric in metrics:
            difference = values[metric] - baseline_values[metric]
            summary[f"{metric}_gain"] = float(difference.mean())
            summary[f"{metric}_gain_ci95"] = safe_t_interval(difference)
            summary[f"{metric}_wins_ties_losses"] = [
                int(np.sum(difference > 0)),
                int(np.sum(difference == 0)),
                int(np.sum(difference < 0)),
            ]
        output[arm] = summary
    return output


def select_fusion(selection_summary: Mapping[str, Mapping[str, Any]]) -> str:
    weight_by_name = dict(zip(FUSION_NAMES, FUSION_WEIGHTS, strict=True))
    return min(
        FUSION_NAMES,
        key=lambda name: (
            -selection_summary[name]["F_ssim"],
            -selection_summary[name]["h28_ssim"],
            -selection_summary[name]["adjacency"],
            -selection_summary[name]["translation_aligned_placement"],
            weight_by_name[name],
        ),
    )


def verification_gate(
    selected: str,
    verification_summary: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    all_audits_passed: bool,
) -> dict[str, Any]:
    observed = verification_summary[selected]
    thresholds = config["verification_gate_last_8"]
    translation_gain = observed["translation_aligned_placement_gain"]
    translation_key = (
        "selected_fusion_translation_aligned_placement_gain_vs_dirty_bilateral_strictly_positive"
    )
    checks = {
        "selected_fusion_mean_F_ssim_gain_vs_dirty_bilateral_strictly_positive": observed[
            "F_ssim_gain"
        ]
        > 0,
        "selected_fusion_F_ssim_paired_t_ci95_lower_strictly_positive": observed[
            "F_ssim_gain_ci95"
        ]["lower"]
        > 0,
        "selected_fusion_F_ssim_wins_vs_dirty_bilateral_min": observed["F_ssim_wins_ties_losses"][0]
        >= thresholds["selected_fusion_F_ssim_wins_vs_dirty_bilateral_min"],
        "selected_fusion_mean_h28_ssim_gain_vs_dirty_bilateral_strictly_positive": observed[
            "h28_ssim_gain"
        ]
        > 0,
        "selected_fusion_h28_ssim_wins_vs_dirty_bilateral_min": observed[
            "h28_ssim_wins_ties_losses"
        ][0]
        >= thresholds["selected_fusion_h28_ssim_wins_vs_dirty_bilateral_min"],
        "selected_fusion_mean_exact_adjacency_gain_vs_dirty_bilateral_strictly_positive": observed[
            "adjacency_gain"
        ]
        > 0,
        "selected_fusion_exact_adjacency_wins_vs_dirty_bilateral_min": observed[
            "adjacency_wins_ties_losses"
        ][0]
        >= thresholds["selected_fusion_exact_adjacency_wins_vs_dirty_bilateral_min"],
        translation_key: translation_gain > 0,
        "selected_fusion_direct_placement_gain_vs_dirty_bilateral_nonnegative": observed[
            "direct_placement_gain"
        ]
        >= 0,
        "all_original_tile_permutation_audits_passed": all_audits_passed,
    }
    if set(checks) != set(thresholds):
        raise RuntimeError("verification check names differ from immutable config")
    return {"selected_fusion": selected, "checks": checks, "passed": all(checks.values())}


def score_committed(
    frozen: Sequence[Mapping[str, Any]],
    *,
    targets_dir: Path,
    config: Mapping[str, Any],
    commitment: Mapping[str, Any],
    commitment_file_sha256: str,
) -> dict[str, Any]:
    def evaluate_subset(
        subset: Sequence[Mapping[str, Any]],
        *,
        role: str,
        global_start: int,
    ) -> list[dict[str, Any]]:
        evaluated: list[dict[str, Any]] = []
        for local_index, row in enumerate(subset):
            record = row["record"]
            filename = str(record["filename"])
            target = load_rgb_verified(
                targets_dir / filename,
                str(record["target_sha256"]),
            )
            recovered = recover_layout(split_tiles(row["dirty"]), split_tiles(target))
            variants: dict[str, Any] = {}
            for arm in ARM_NAMES:
                variant = row["variants"][arm]
                metrics = layout_metrics(variant["layout"], recovered)
                variants[arm] = {
                    **metrics,
                    "h28_ssim": contest_ssim(target, variant["predictions"][TAIL_H28]),
                    "F_ssim": contest_ssim(target, variant["predictions"][TAIL_F]),
                }
            evaluated.append({"filename": filename, "panel_role": role, "variants": variants})
            global_index = global_start + local_index + 1
            print(f"scored {global_index}/{len(frozen)} {filename}", flush=True)
        return evaluated

    selection_rows = evaluate_subset(
        frozen[:SELECTION_COUNT],
        role="selection",
        global_start=0,
    )
    selection_summary = summarize_panel(selection_rows)
    selected = select_fusion(selection_summary)
    selection_decision = {
        "schema": "aiijc-drunet40-matcher-train-selection-decision-v1",
        "config_sha256": CONFIG_SHA256,
        "commitment_file_sha256": commitment_file_sha256,
        "commitment_payload_sha256": commitment["commitment_sha256"],
        "selection_filenames": [row["filename"] for row in selection_rows],
        "selection_filenames_sha256": names_digest(
            [row["record"] for row in frozen[:SELECTION_COUNT]]
        ),
        "selectable_fusions": list(FUSION_NAMES),
        "pure_drunet_selectable": False,
        "selected_fusion": selected,
        "selection_summary_first_8": selection_summary,
        "verification_target_paths_opened": False,
    }
    atomic_json(SELECTION_DECISION_PATH, selection_decision, readonly=True)
    atomic_json(
        VERIFICATION_RECEIPT_PATH,
        {
            "schema": "aiijc-drunet40-matcher-train-verification-target-receipt-v1",
            "config_sha256": CONFIG_SHA256,
            "commitment_file_sha256": commitment_file_sha256,
            "selection_decision_sha256": sha256_file(SELECTION_DECISION_PATH),
            "selected_fusion": selected,
            "meaning": "verification targets open only after immutable first-8 selection",
        },
        readonly=True,
    )
    verification_rows = evaluate_subset(
        frozen[SELECTION_COUNT:],
        role="verification",
        global_start=SELECTION_COUNT,
    )
    verification_summary = summarize_panel(verification_rows)
    all_audits_passed = bool(commitment["contract"]["all_raw_permutation_audits_passed"])
    gate = verification_gate(
        selected,
        verification_summary,
        config,
        all_audits_passed=all_audits_passed,
    )
    return {
        "selection_summary_first_8": selection_summary,
        "selected_fusion": selected,
        "pure_drunet_selectable": False,
        "verification_summary_last_8": verification_summary,
        "verification_gate": gate,
        "calibration_authorized": gate["passed"],
        "selection_decision": {
            "path": str(SELECTION_DECISION_PATH.resolve()),
            "sha256": sha256_file(SELECTION_DECISION_PATH),
        },
        "verification_target_receipt": {
            "path": str(VERIFICATION_RECEIPT_PATH.resolve()),
            "sha256": sha256_file(VERIFICATION_RECEIPT_PATH),
        },
        "rows": [*selection_rows, *verification_rows],
    }


def main() -> None:
    args = parse_args()
    config, records = load_contract(args.manifest.resolve())
    if args.phase == "prepare":
        if OUTPUT_ROOT.exists():
            raise RuntimeError(f"refusing to overwrite train diagnostic: {OUTPUT_ROOT}")
        OUTPUT_ROOT.mkdir(parents=True)
        assets = verify_assets()
        device = choose_device(args.device)
        started = perf_counter()
        frozen = freeze_predictions(
            records,
            inputs_dir=args.inputs.resolve(),
            device=device,
        )
        artifacts = persist_artifacts(frozen, OUTPUT_ROOT)
        commitment = build_commitment(
            frozen,
            records,
            artifacts,
            device=device,
            assets=assets,
        )
        commitment["prediction_freeze_seconds"] = perf_counter() - started
        commitment.pop("commitment_sha256")
        commitment["commitment_sha256"] = canonical_digest(commitment)
        atomic_json(COMMITMENT_PATH, commitment, readonly=True)
        if len(reload_commitment(commitment, records)) != len(records):
            raise RuntimeError("commitment readback failed")
        print(
            json.dumps(
                {
                    "phase": "prepare",
                    "commitment": str(COMMITMENT_PATH),
                    "commitment_file_sha256": sha256_file(COMMITMENT_PATH),
                    "target_paths_opened": False,
                },
                indent=2,
            )
        )
        return

    if not COMMITMENT_PATH.is_file():
        raise RuntimeError("score phase requires prediction commitment")
    if any(
        path.exists()
        for path in (
            REPORT_PATH,
            RECEIPT_PATH,
            SELECTION_DECISION_PATH,
            VERIFICATION_RECEIPT_PATH,
        )
    ):
        raise RuntimeError("score phase is single-use and already started")
    commitment_bytes = COMMITMENT_PATH.read_bytes()
    commitment = json.loads(commitment_bytes)
    frozen = reload_commitment(commitment, records)
    commitment_file_sha256 = hashlib.sha256(commitment_bytes).hexdigest()
    atomic_json(
        RECEIPT_PATH,
        {
            "schema": "aiijc-drunet40-matcher-train-target-receipt-v1",
            "config_sha256": CONFIG_SHA256,
            "commitment_file_sha256": commitment_file_sha256,
            "historically_exposed_before_this_experiment": True,
            "calibration_access": False,
        },
        readonly=True,
    )
    started = perf_counter()
    evaluation = score_committed(
        frozen,
        targets_dir=args.targets.resolve(),
        config=config,
        commitment=commitment,
        commitment_file_sha256=commitment_file_sha256,
    )
    if sha256_file(COMMITMENT_PATH) != commitment_file_sha256:
        raise RuntimeError("commitment changed after target access")
    report = {
        "schema": "aiijc-drunet40-matcher-train-report-v1",
        "status": "selection_and_disjoint_verification_scored_after_commitment",
        "config": str(CONFIG),
        "config_sha256": CONFIG_SHA256,
        "historical_target_exposure": True,
        "freshness_claim": False,
        "selection": {
            "filenames": [record["filename"] for record in records[:SELECTION_COUNT]],
            "filenames_sha256": names_digest(records[:SELECTION_COUNT]),
        },
        "verification": {
            "filenames": [record["filename"] for record in records[SELECTION_COUNT:]],
            "filenames_sha256": names_digest(records[SELECTION_COUNT:]),
        },
        "prediction_contract": {
            "all_layouts_and_predictions_frozen_before_target_access": True,
            "drunet_pixels_used_for_matcher_scores_only": True,
            "all_output_pixels_from_original_dirty_tiles": True,
            "commitment_file_sha256": commitment_file_sha256,
            "commitment_payload_sha256": commitment["commitment_sha256"],
            "target_receipt_sha256": sha256_file(RECEIPT_PATH),
            "selection_decision_sha256": sha256_file(SELECTION_DECISION_PATH),
            "verification_target_receipt_sha256": sha256_file(VERIFICATION_RECEIPT_PATH),
            "calibration_access": False,
            "holdout_access": False,
            "test_access": False,
        },
        "runtime_seconds": {
            "prediction_freeze": commitment["prediction_freeze_seconds"],
            "target_evaluation": perf_counter() - started,
        },
        "evaluation": evaluation,
    }
    atomic_json(REPORT_PATH, report, readonly=True)
    print(
        json.dumps(
            {
                "phase": "score",
                "report": str(REPORT_PATH),
                "selected_fusion": evaluation["selected_fusion"],
                "verification_gate_passed": evaluation["verification_gate"]["passed"],
                "calibration_authorized": evaluation["calibration_authorized"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
