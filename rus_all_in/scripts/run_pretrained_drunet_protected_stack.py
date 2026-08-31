#!/usr/bin/env python3
"""Train sanity, freeze, and score the fixed DRUNet40 protected-NLM stack."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.dualnaf_bounded_residual import paired_bootstrap_ci
from aiijc_puzzle.legacy_upgrade import (
    directional_scores,
    layout_digest,
    solve_buddies,
)
from aiijc_puzzle.nlm_luma_chroma import safety_summary
from aiijc_puzzle.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.pretrained_drunet_protected_stack import (
    ARM_COMBINED,
    ARM_DRUNET_H28,
    ARM_NAMES,
    ARM_ORIGINAL_H28,
    DRUNET_SIGMA,
    MODEL_BATCH_SIZE,
    SOBEL_THRESHOLD,
    image_digest,
    render_combined_arms,
)
from aiijc_puzzle.pretrained_tile_denoiser import load_drunet_color
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    IMAGE_SIZE,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    PROJECT_ROOT / "configs/pretrained_drunet_protected_stack_preregistered_v1.json"
)
PREREGISTRATION_SHA256 = "6e6db6d4becb22a5fb70a9ce20474c6350f7e55ff391d65d763699938958e5a8"
OVERLAP_LEDGER = (
    PROJECT_ROOT / "configs/pretrained_drunet_protected_stack_overlap_ledger_v1.json"
)
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
TARGETS = PROJECT_ROOT / "data/raw/train/targets"
ASSET_ROOT = PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f"
CHECKPOINT = ASSET_ROOT / "drunet_color.pth"
OUTPUT_PARENT = PROJECT_ROOT / "outputs/pretrained-drunet-protected-stack"
TRAIN_REPORT = OUTPUT_PARENT / "train-sanity-offset512-count16/report.json"
STAGE_ROOTS = {
    "primary": OUTPUT_PARENT / "primary-calibration-offset264-count120",
    "confirmation": OUTPUT_PARENT / "confirmation-calibration-offset408-count120",
}
COMMITMENT_RECEIPTS = {
    stage: Path(f"{root}.commitment-receipt.json") for stage, root in STAGE_ROOTS.items()
}
PRIMARY_REPORT = STAGE_ROOTS["primary"] / "report.json"
PRIMARY_MANUAL_REVIEW = STAGE_ROOTS["primary"] / "manual-review.json"
EXPECTED_MANIFEST_SHA256 = "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
EXPECTED_PROTOCOL_DIGEST = "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
EXPECTED_ASSET_SHA256 = {
    "LICENSE": "448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5",
    "drunet_color.pth": "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4",
    "models/basicblock.py": "48406db8867394ac5ae233ebeec7711ac10acfc3a6bbf0072c33aa77d659b6fd",
    "models/network_unet.py": "8043b6350f1589d5f08892e3be0b4d12c5a502058014285107b7360696d12bf5",
}
MODEL_PARAMETER_COUNT = 32_640_960
EDGE_BUDGET = 96
TRAIN_OFFSET = 512
TRAIN_COUNT = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("train-sanity", "freeze", "score"),
        required=True,
    )
    parser.add_argument("--stage", choices=tuple(STAGE_ROOTS), default="primary")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def require_readonly(path: Path) -> None:
    if path.stat().st_mode & WRITE_BITS:
        raise PermissionError(f"integrity-bound artifact is writable: {path}")


def write_bytes_exclusive_readonly(path: Path, payload: bytes) -> str:
    """Create one immutable artifact without any overwrite-capable operation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o444)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    require_readonly(path)
    return hashlib.sha256(payload).hexdigest()


def write_json_exclusive_readonly(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return write_bytes_exclusive_readonly(path, encoded)


def write_png_exclusive_readonly(path: Path, image: np.ndarray) -> str:
    if image.dtype != np.uint8 or image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("frozen prediction must be uint8 RGB 480x480")
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG", compress_level=6)
    return write_bytes_exclusive_readonly(path, buffer.getvalue())


def load_rgb_verified(path: Path, expected_hash: str) -> np.ndarray:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def names_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode("utf-8")
    ).hexdigest()


def input_roster_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(
            f"{record['filename']} {record['input_sha256']}" for record in records
        ).encode("utf-8")
    ).hexdigest()


def verify_assets() -> dict[str, str]:
    observed = {
        relative: sha256_file(ASSET_ROOT / relative) for relative in EXPECTED_ASSET_SHA256
    }
    if observed != EXPECTED_ASSET_SHA256:
        raise ValueError(f"official KAIR assets changed: {observed}")
    license_text = (ASSET_ROOT / "LICENSE").read_text(encoding="utf-8")
    if "MIT License" not in license_text or "Copyright (c) 2019 Kai Zhang" not in license_text:
        raise ValueError("official KAIR MIT license text changed")
    return observed


def load_manifest() -> dict[str, Any]:
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("validation manifest file changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest self-digest is invalid")
    if manifest.get("protocol_digest") != EXPECTED_PROTOCOL_DIGEST:
        raise ValueError("validation protocol changed")
    return manifest


def select_records(
    manifest: Mapping[str, Any],
    split: str,
    offset: int,
    count: int,
) -> tuple[Mapping[str, Any], ...]:
    records = tuple(
        select_manifest_records(
            manifest,
            split,
            limit=offset + count,
            seed=EXPERIMENT_SUBSET_SEED,
            namespace=EXPERIMENT_SUBSET_NAMESPACE,
        )[offset:]
    )
    if len(records) != count:
        raise RuntimeError("selected record count drifted")
    return records


def load_context(stage: str) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    sidecar = Path(f"{PREREGISTRATION}.sha256")
    for integrity_path in (PREREGISTRATION, sidecar, OVERLAP_LEDGER):
        require_readonly(integrity_path)
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("combined stack preregistration changed")
    if sidecar.read_text(encoding="utf-8").split()[0] != PREREGISTRATION_SHA256:
        raise ValueError("combined stack preregistration sidecar changed")
    config = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if tuple(config["arm_names"]) != ARM_NAMES:
        raise ValueError("runtime arm roster differs from preregistration")
    if sha256_file(OVERLAP_LEDGER) != config["data"]["overlap_ledger_sha256"]:
        raise ValueError("historical overlap ledger changed")
    manifest = load_manifest()
    data = config["data"]
    offset = int(data[f"{stage}_offset"])
    count = int(data[f"{stage}_count"])
    records = select_records(manifest, "calibration", offset, count)
    if names_digest(records) != data[f"{stage}_filenames_sha256"]:
        raise ValueError("selected filename roster differs from preregistration")
    if input_roster_digest(records) != data[f"{stage}_input_roster_sha256"]:
        raise ValueError("selected input roster differs from preregistration")
    return config, records


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/aiijc_puzzle/pretrained_drunet_protected_stack.py",
        PROJECT_ROOT / "src/aiijc_puzzle/pretrained_tile_denoiser.py",
        PROJECT_ROOT / "src/aiijc_puzzle/edge_protected_nlm.py",
        PROJECT_ROOT / "src/aiijc_puzzle/nlm_luma_chroma.py",
        PROJECT_ROOT / "src/aiijc_puzzle/legacy_upgrade.py",
        PROJECT_ROOT / "src/aiijc_puzzle/postassembly_harmonizer.py",
        PROJECT_ROOT / "src/aiijc_puzzle/protocol.py",
        PROJECT_ROOT / "src/aiijc_puzzle/compliant_atlas_decoder.py",
        PROJECT_ROOT / "src/aiijc_puzzle/dualnaf_bounded_residual.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def apply_rgb_luma(ordered_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered_tiles, SeamGraphConfig())
    rgb_tiles = apply_rgb_offsets(ordered_tiles, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(rgb_tiles, LuminanceGainConfig())
    return apply_luminance_gains(rgb_tiles, gains), {
        "rgb_seam_offsets": rgb_diagnostics,
        "bounded_luminance_gains": luma_diagnostics,
    }


def infer_board(
    dirty: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    input_tiles = split_tiles(dirty)
    right, down = directional_scores(input_tiles, views=("bilateral",))["bilateral"]
    solved = solve_buddies(right, down, max_edges=EDGE_BUDGET)
    layout = np.asarray(solved.layout, dtype=np.int32)
    ordered = np.ascontiguousarray(input_tiles[layout])
    raw = assemble_tiles(ordered)
    audit = audit_raw_permutation(
        dirty, raw, layout, restoration_applied_after_audit=True
    )
    if not audit.passed:
        raise RuntimeError("strict raw permutation audit failed")
    harmonized_tiles, harmonizer = apply_rgb_luma(ordered)
    predictions, render_diagnostics = render_combined_arms(
        model,
        harmonized_tiles,
        device=device,
    )
    hashes = {name: image_digest(image) for name, image in predictions.items()}
    if len(set(hashes.values())) != len(ARM_NAMES):
        raise RuntimeError("the four fixed predictions were not distinct")
    return {
        "layout": layout,
        "raw": raw,
        "audit": audit.as_dict(),
        "solver": solved.solver,
        "objective": float(solved.objective),
        "harmonizer": harmonizer,
        "predictions": predictions,
        "pixel_sha256": hashes,
        "render_diagnostics": render_diagnostics,
    }


def board_safety(result: Mapping[str, Any]) -> dict[str, Any]:
    predictions = result["predictions"]
    structures = result["render_diagnostics"]["structure"]
    baseline = predictions[ARM_ORIGINAL_H28]
    contender = predictions[ARM_COMBINED]
    baseline_float = baseline.astype(np.float64)
    contender_float = contender.astype(np.float64)
    shift = contender_float.mean(axis=(0, 1)) - baseline_float.mean(axis=(0, 1))
    return {
        "baseline_structure": structures[ARM_ORIGINAL_H28],
        "candidate_structure": structures[ARM_COMBINED],
        "protected_fraction": result["render_diagnostics"]["mask"][
            "binary_dilated_protected_fraction"
        ],
        "maximum_abs_rgb_mean_shift_vs_original_h28": float(np.max(np.abs(shift))),
        "global_rgb_std_ratio_vs_original_h28": float(
            contender_float.std() / baseline_float.std()
        ),
        "mean_abs_pixel_change_vs_original_h28": float(
            np.abs(contender.astype(np.int16) - baseline.astype(np.int16)).mean()
        ),
        "clipped_fraction_increase_vs_original_h28": float(
            structures[ARM_COMBINED]["clipped_fraction"]
            - structures[ARM_ORIGINAL_H28]["clipped_fraction"]
        ),
    }


def comparison(rows: Sequence[Mapping[str, Any]], baseline: str) -> dict[str, Any]:
    candidate = np.asarray([row["ssim"][ARM_COMBINED] for row in rows])
    control = np.asarray([row["ssim"][baseline] for row in rows])
    difference = candidate - control
    ci = paired_bootstrap_ci(difference)
    return {
        "candidate_mean_ssim": float(candidate.mean()),
        "baseline_mean_ssim": float(control.mean()),
        "mean_delta": float(difference.mean()),
        "paired_bootstrap_ci95": list(ci),
        "wins_ties_losses": [
            int(np.sum(difference > 0)),
            int(np.sum(difference == 0)),
            int(np.sum(difference < 0)),
        ],
        "board_deltas": difference.tolist(),
    }


def train_sanity(device: torch.device) -> None:
    if TRAIN_REPORT.exists():
        raise FileExistsError(f"refusing to overwrite train sanity report: {TRAIN_REPORT}")
    manifest = load_manifest()
    records = select_records(manifest, "train", TRAIN_OFFSET, TRAIN_COUNT)
    assets = verify_assets()
    model = load_drunet_color(CHECKPOINT, device)
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = load_rgb_verified(INPUTS / filename, record["input_sha256"])
        result = infer_board(dirty, model, device)
        target = load_rgb_verified(TARGETS / filename, record["target_sha256"])
        scores = {
            name: contest_ssim(target, result["predictions"][name]) for name in ARM_NAMES
        }
        safety = board_safety(result)
        safety_rows.append(safety)
        rows.append(
            {
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "target_sha256": record["target_sha256"],
                "layout_sha256": layout_digest(result["layout"]),
                "raw_permutation_audit": result["audit"],
                "ssim": scores,
                "safety": safety,
                "mask": result["render_diagnostics"]["mask"],
            }
        )
        print(
            json.dumps(
                {
                    "phase": "fixed_train_sanity",
                    "done": index,
                    "total": len(records),
                    "filename": filename,
                    "B": scores[ARM_ORIGINAL_H28],
                    "C": scores[ARM_DRUNET_H28],
                    "D": scores[ARM_COMBINED],
                }
            ),
            flush=True,
        )
    report = {
        "schema": "aiijc-pretrained-drunet-protected-stack-train-sanity-v1",
        "status": "fixed_composition_sanity_only_no_parameter_selection",
        "split": "train",
        "offset": TRAIN_OFFSET,
        "count": TRAIN_COUNT,
        "selection_digest": names_digest(records),
        "fixed_parameters_not_tunable_from_this_report": {
            "drunet_sigma": DRUNET_SIGMA,
            "nlm_strengths": [20, 28, 40],
            "sobel_threshold": SOBEL_THRESHOLD,
        },
        "mean_ssim": {
            name: float(np.mean([row["ssim"][name] for row in rows])) for name in ARM_NAMES
        },
        "comparisons": {
            baseline: comparison(rows, baseline)
            for baseline in (ARM_ORIGINAL_H28, ARM_DRUNET_H28)
        },
        "safety_vs_original_h28": aggregate_safety(safety_rows),
        "assets_sha256": assets,
        "source_sha256_at_sanity_time": source_hashes(),
        "runtime_seconds": perf_counter() - started,
        "calibration_targets_accessed": False,
        "holdout_access": False,
        "competition_test_access": False,
        "rows": rows,
    }
    atomic_json(TRAIN_REPORT, report)
    print(
        json.dumps(
            {
                "report": str(TRAIN_REPORT),
                "report_sha256": sha256_file(TRAIN_REPORT),
                "mean_ssim": report["mean_ssim"],
                "comparisons": report["comparisons"],
                "safety": report["safety_vs_original_h28"],
            },
            indent=2,
        )
    )


def aggregate_safety(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    structure = safety_summary(
        [row["candidate_structure"] for row in rows],
        [row["baseline_structure"] for row in rows],
    )
    protected = np.asarray([row["protected_fraction"] for row in rows])
    color_shift = np.asarray(
        [row["maximum_abs_rgb_mean_shift_vs_original_h28"] for row in rows]
    )
    spread = np.asarray([row["global_rgb_std_ratio_vs_original_h28"] for row in rows])
    pixel = np.asarray([row["mean_abs_pixel_change_vs_original_h28"] for row in rows])
    clipping = np.asarray(
        [row["clipped_fraction_increase_vs_original_h28"] for row in rows]
    )
    return {
        **structure,
        "protected_fraction_mean_min_max": [
            float(protected.mean()),
            float(protected.min()),
            float(protected.max()),
        ],
        "maximum_rgb_mean_shift": float(color_shift.max()),
        "global_rgb_std_ratio_mean_min_max": [
            float(spread.mean()),
            float(spread.min()),
            float(spread.max()),
        ],
        "mean_abs_pixel_change_mean_max": [float(pixel.mean()), float(pixel.max())],
        "maximum_clipping_increase": float(clipping.max()),
    }


def verify_commitment_receipt(stage: str, commitment_path: Path) -> dict[str, Any]:
    receipt_path = COMMITMENT_RECEIPTS[stage]
    if not receipt_path.is_file():
        raise FileNotFoundError("external target-blind commitment receipt is missing")
    require_readonly(commitment_path)
    require_readonly(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "aiijc-pretrained-drunet-protected-stack-receipt-v1",
        "status": "committed_before_any_target_decode_in_this_stage",
        "stage": stage,
        "commitment_relative_path": str(commitment_path.relative_to(PROJECT_ROOT)),
        "commitment_sha256": sha256_file(commitment_path),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "overlap_ledger_sha256": sha256_file(OVERLAP_LEDGER),
        "targets_decoded_before_receipt": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"commitment receipt binding changed: {key}")
    return receipt


def require_confirmation_authorized() -> None:
    primary_commitment = STAGE_ROOTS["primary"] / "prediction-commitment.json"
    if not all(
        path.is_file()
        for path in (PRIMARY_REPORT, PRIMARY_MANUAL_REVIEW, primary_commitment)
    ):
        raise RuntimeError("primary report, commitment, and root manual review are required")
    require_readonly(PRIMARY_REPORT)
    require_readonly(PRIMARY_MANUAL_REVIEW)
    receipt = verify_commitment_receipt("primary", primary_commitment)
    report_sha256 = sha256_file(PRIMARY_REPORT)
    commitment_sha256 = sha256_file(primary_commitment)
    receipt_sha256 = sha256_file(COMMITMENT_RECEIPTS["primary"])
    report = json.loads(PRIMARY_REPORT.read_text(encoding="utf-8"))
    review = json.loads(PRIMARY_MANUAL_REVIEW.read_text(encoding="utf-8"))
    if (
        report.get("stage") != "primary"
        or report.get("count") != 120
        or report.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or report.get("commitment_sha256") != commitment_sha256
        or report.get("commitment_receipt_sha256") != receipt_sha256
        or report.get("frozen_prediction_roster_sha256")
        != receipt.get("frozen_prediction_roster_sha256")
        or report.get("quantitative_pass") is not True
        or report.get("selected_passing_winner") != ARM_COMBINED
    ):
        raise RuntimeError("combined primary candidate did not pass bound quantitative gates")
    required_review_bindings = {
        "schema": "aiijc-pretrained-drunet-protected-stack-root-review-v1",
        "reviewer": "root",
        "reviewed_arm": ARM_COMBINED,
        "reviewed_board_count": 120,
        "reviewed_all_full_canvas_triplets": True,
        "severe_artifacts": 0,
        "material_face_text_or_object_loss": False,
        "mask_halo_or_boundary_damage": False,
        "passed": True,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "primary_report_sha256": report_sha256,
        "primary_commitment_sha256": commitment_sha256,
        "primary_commitment_receipt_sha256": receipt_sha256,
        "frozen_prediction_roster_sha256": receipt[
            "frozen_prediction_roster_sha256"
        ],
        "manual_review_sheet_roster_sha256": report[
            "manual_review_sheet_roster_sha256"
        ],
    }
    for key, value in required_review_bindings.items():
        if review.get(key) != value:
            raise RuntimeError(f"root manual preservation gate binding failed: {key}")


def freeze(stage: str, device: torch.device) -> None:
    config, records = load_context(stage)
    if stage == "confirmation":
        require_confirmation_authorized()
    root = STAGE_ROOTS[stage]
    receipt_path = COMMITMENT_RECEIPTS[stage]
    if root.exists() or receipt_path.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen experiment or receipt: {root}"
        )
    canonical_device = str(config["runtime_reproducibility"]["canonical_prediction_device"])
    if device.type != canonical_device:
        raise RuntimeError(
            f"frozen predictions are preregistered on {canonical_device}, got {device.type}"
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("preregistered MPS device is unavailable")
    assets = verify_assets()
    model = load_drunet_color(CHECKPOINT, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != MODEL_PARAMETER_COUNT:
        raise ValueError("DRUNet architecture parameter count changed")
    root.mkdir(parents=True)
    started = perf_counter()
    boards: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = load_rgb_verified(INPUTS / filename, record["input_sha256"])
        result = infer_board(dirty, model, device)
        board_directory = root / "predictions" / Path(filename).stem
        prediction_records: dict[str, Any] = {}
        for name, prediction in result["predictions"].items():
            output_path = board_directory / f"{name}.png"
            prediction_records[name] = {
                "relative_path": str(output_path.relative_to(root)),
                "png_sha256": write_png_exclusive_readonly(output_path, prediction),
                "pixel_sha256": result["pixel_sha256"][name],
                "structure": result["render_diagnostics"]["structure"][name],
            }
        boards.append(
            {
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "layout_sha256": layout_digest(result["layout"]),
                "raw_sha256": image_digest(result["raw"]),
                "raw_permutation_audit": result["audit"],
                "solver": result["solver"],
                "objective": result["objective"],
                "all_predictions_distinct": True,
                "harmonizer": result["harmonizer"],
                "render_diagnostics": result["render_diagnostics"],
                "candidate_safety_vs_original_h28": board_safety(result),
                "predictions": prediction_records,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "target_blind_freeze",
                    "stage": stage,
                    "done": index,
                    "total": len(records),
                    "filename": filename,
                }
            ),
            flush=True,
        )
    prediction_roster_sha256 = hashlib.sha256(
        "\n".join(
            f"{board['filename']} "
            + " ".join(
                board["predictions"][name]["pixel_sha256"] for name in ARM_NAMES
            )
            for board in boards
        ).encode("utf-8")
    ).hexdigest()
    commitment = {
        "schema": "aiijc-pretrained-drunet-protected-stack-commitment-v1",
        "status": "frozen_before_any_target_decode_in_this_stage",
        "stage": stage,
        "split": "calibration",
        "offset": config["data"][f"{stage}_offset"],
        "count": len(records),
        "selection_digest": names_digest(records),
        "historical_target_exposure": config["data"]["historical_target_exposure"],
        "targets_decoded_during_freeze": False,
        "holdout_access": False,
        "competition_test_access": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "overlap_ledger_sha256": sha256_file(OVERLAP_LEDGER),
        "manifest_sha256": sha256_file(MANIFEST),
        "arm_names": list(ARM_NAMES),
        "fixed_composition": config["fixed_composition"],
        "model": {
            "architecture": "KAIR colour DRUNet UNetRes",
            "parameter_count": parameter_count,
            "sigma_255": DRUNET_SIGMA,
            "batch_size": MODEL_BATCH_SIZE,
            "checkpoint_sha256": assets["drunet_color.pth"],
            "strict_state_dict_load": True,
        },
        "license_and_vendor": {
            "official_repository": "https://github.com/cszn/KAIR",
            "official_commit": "fc1732f4a4514e42ce15e5b3a1e18c828af47a1e",
            "license": "MIT",
            "asset_sha256": assets,
            "submission_contains_png_predictions_only": True,
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "geometry_contract": config["geometry_contract"],
        "source_sha256": source_hashes(),
        "frozen_prediction_roster_sha256": prediction_roster_sha256,
        "all_boards_raw_permutation_pass": all(
            board["raw_permutation_audit"]["passed"] for board in boards
        ),
        "all_boards_predictions_distinct": all(
            board["all_predictions_distinct"] for board in boards
        ),
        "runtime_seconds": perf_counter() - started,
        "boards": boards,
    }
    path = root / "prediction-commitment.json"
    commitment_sha256 = write_json_exclusive_readonly(path, commitment)
    prediction_root = root / "predictions"
    for directory in sorted(
        (item for item in prediction_root.rglob("*") if item.is_dir()),
        reverse=True,
    ):
        os.chmod(directory, 0o555)
    os.chmod(prediction_root, 0o555)
    receipt = {
        "schema": "aiijc-pretrained-drunet-protected-stack-receipt-v1",
        "status": "committed_before_any_target_decode_in_this_stage",
        "stage": stage,
        "commitment_relative_path": str(path.relative_to(PROJECT_ROOT)),
        "commitment_sha256": commitment_sha256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "overlap_ledger_sha256": sha256_file(OVERLAP_LEDGER),
        "selection_digest": names_digest(records),
        "input_roster_sha256": input_roster_digest(records),
        "frozen_prediction_roster_sha256": prediction_roster_sha256,
        "source_sha256": commitment["source_sha256"],
        "targets_decoded_before_receipt": False,
        "holdout_access": False,
        "competition_test_access": False,
    }
    receipt_sha256 = write_json_exclusive_readonly(receipt_path, receipt)
    print(
        json.dumps(
            {
                "commitment": str(path),
                "commitment_sha256": commitment_sha256,
                "external_receipt": str(receipt_path),
                "external_receipt_sha256": receipt_sha256,
                "prediction_roster_sha256": prediction_roster_sha256,
            },
            indent=2,
        )
    )


def load_frozen(stage: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = STAGE_ROOTS[stage]
    path = root / "prediction-commitment.json"
    if not path.is_file():
        raise FileNotFoundError("target-blind prediction commitment is missing")
    receipt = verify_commitment_receipt(stage, path)
    commitment = json.loads(path.read_text(encoding="utf-8"))
    if commitment.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("frozen commitment preregistration binding changed")
    if commitment.get("source_sha256") != source_hashes():
        raise ValueError("source changed after target-blind commitment")
    if commitment.get("stage") != stage:
        raise ValueError("frozen commitment stage binding changed")
    if (
        commitment.get("frozen_prediction_roster_sha256")
        != receipt.get("frozen_prediction_roster_sha256")
    ):
        raise ValueError("frozen prediction roster differs from external receipt")
    if commitment.get("license_and_vendor", {}).get("asset_sha256") != verify_assets():
        raise ValueError("checkpoint or vendor assets changed after commitment")
    frozen: list[dict[str, Any]] = []
    for board in commitment["boards"]:
        predictions: dict[str, np.ndarray] = {}
        for name in ARM_NAMES:
            record = board["predictions"][name]
            png = root / record["relative_path"]
            require_readonly(png)
            if sha256_file(png) != record["png_sha256"]:
                raise ValueError(f"frozen PNG changed: {png}")
            image = load_rgb_verified(png, record["png_sha256"])
            if image_digest(image) != record["pixel_sha256"]:
                raise ValueError(f"frozen pixels changed: {png}")
            predictions[name] = image
        frozen.append({"board": board, "predictions": predictions})
    return commitment, frozen


def safety_gate(commitment: Mapping[str, Any]) -> dict[str, Any]:
    rows = [board["candidate_safety_vs_original_h28"] for board in commitment["boards"]]
    summary = aggregate_safety(rows)
    protected = summary["protected_fraction_mean_min_max"]
    spread = summary["global_rgb_std_ratio_mean_min_max"]
    checks = {
        "luma_gradient_mean_at_least_0_90": (
            summary["mean_luminance_gradient_retention"] >= 0.90
        ),
        "luma_gradient_min_at_least_0_80": (
            summary["minimum_luminance_gradient_retention"] >= 0.80
        ),
        "chroma_gradient_mean_at_least_0_80": (
            summary["mean_chroma_gradient_retention"] >= 0.80
        ),
        "chroma_gradient_min_at_least_0_60": (
            summary["minimum_chroma_gradient_retention"] >= 0.60
        ),
        "laplacian_mean_at_least_0_90": summary["mean_laplacian_retention"] >= 0.90,
        "laplacian_min_at_least_0_80": (
            summary["minimum_laplacian_retention"] >= 0.80
        ),
        "grid_ratio_mean_at_most_1_08": (
            summary["mean_grid_ratio_relative_to_baseline"] <= 1.08
        ),
        "grid_ratio_max_at_most_1_15": (
            summary["maximum_grid_ratio_relative_to_baseline"] <= 1.15
        ),
        "protected_fraction_mean_between_0_40_and_0_75": 0.40 <= protected[0] <= 0.75,
        "protected_fraction_every_board_between_0_30_and_0_85": (
            protected[1] >= 0.30 and protected[2] <= 0.85
        ),
        "rgb_mean_shift_at_most_3": summary["maximum_rgb_mean_shift"] <= 3.0,
        "rgb_std_ratio_every_board_between_0_90_and_1_05": (
            spread[1] >= 0.90 and spread[2] <= 1.05
        ),
        "mean_abs_pixel_change_max_at_most_8": (
            summary["mean_abs_pixel_change_mean_max"][1] <= 8.0
        ),
        "clipping_increase_at_most_0_01": summary["maximum_clipping_increase"] <= 0.01,
        "all_raw_permutation_audits_pass": bool(
            commitment["all_boards_raw_permutation_pass"]
        ),
        "all_four_predictions_distinct": bool(
            commitment["all_boards_predictions_distinct"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


def atomic_audit_png(path: Path, image: np.ndarray) -> None:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG", compress_level=6)
    write_bytes_exclusive_readonly(path, buffer.getvalue())


def make_manual_sheets(
    root: Path, frozen: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    directory = root / "root-manual-review-sheets"
    directory.mkdir(exist_ok=True)
    records: list[dict[str, str]] = []
    for page_start in range(0, len(frozen), 2):
        page = Image.new("RGB", (1460, 1046), "white")
        draw = ImageDraw.Draw(page)
        draw.text(
            (8, 8),
            f"left {ARM_ORIGINAL_H28} | center {ARM_DRUNET_H28} | right {ARM_COMBINED}",
            fill="black",
        )
        for local_index, item in enumerate(frozen[page_start : page_start + 2]):
            y = 34 + local_index * 505
            draw.text((8, y), str(item["board"]["filename"]), fill="black")
            for column, name in enumerate(
                (ARM_ORIGINAL_H28, ARM_DRUNET_H28, ARM_COMBINED)
            ):
                page.paste(Image.fromarray(item["predictions"][name]), (8 + 484 * column, y + 18))
        path = directory / f"page-{page_start // 2 + 1:03d}.png"
        atomic_audit_png(path, np.asarray(page, dtype=np.uint8))
        records.append(
            {
                "relative_path": str(path.relative_to(root)),
                "png_sha256": sha256_file(path),
            }
        )
    os.chmod(directory, 0o555)
    return records


def score(stage: str) -> None:
    config, records = load_context(stage)
    if stage == "confirmation":
        require_confirmation_authorized()
    root = STAGE_ROOTS[stage]
    report_path = root / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {report_path}")
    commitment, frozen = load_frozen(stage)
    if names_digest(records) != commitment["selection_digest"]:
        raise ValueError("frozen selection differs from selected records")
    rows: list[dict[str, Any]] = []
    for record, item in zip(records, frozen, strict=True):
        if record["filename"] != item["board"]["filename"]:
            raise ValueError("frozen board order differs from selected records")
        target = load_rgb_verified(TARGETS / record["filename"], record["target_sha256"])
        rows.append(
            {
                "filename": record["filename"],
                "ssim": {
                    name: contest_ssim(target, item["predictions"][name])
                    for name in ARM_NAMES
                },
            }
        )
    comparisons = {
        baseline: comparison(rows, baseline)
        for baseline in (ARM_ORIGINAL_H28, ARM_DRUNET_H28)
    }
    candidate_mean = float(np.mean([row["ssim"][ARM_COMBINED] for row in rows]))
    safety = safety_gate(commitment)
    quantitative_checks = {
        "D_mean_ssim_at_least_0_27": candidate_mean >= 0.27,
        "D_ci_lower_positive_and_wins_at_least_90_vs_original_h28": (
            comparisons[ARM_ORIGINAL_H28]["paired_bootstrap_ci95"][0] > 0
            and comparisons[ARM_ORIGINAL_H28]["wins_ties_losses"][0] >= 90
        ),
        "D_ci_lower_positive_and_wins_at_least_90_vs_C": (
            comparisons[ARM_DRUNET_H28]["paired_bootstrap_ci95"][0] > 0
            and comparisons[ARM_DRUNET_H28]["wins_ties_losses"][0] >= 90
        ),
        "strict_h28_safe_gate_pass": safety["passed"],
    }
    quantitative_pass = all(quantitative_checks.values())
    sheets = make_manual_sheets(root, frozen) if quantitative_pass else []
    sheet_roster_sha256 = hashlib.sha256(
        "\n".join(
            f"{sheet['relative_path']} {sheet['png_sha256']}" for sheet in sheets
        ).encode("utf-8")
    ).hexdigest()
    commitment_path = root / "prediction-commitment.json"
    receipt_path = COMMITMENT_RECEIPTS[stage]
    report = {
        "schema": "aiijc-pretrained-drunet-protected-stack-report-v1",
        "status": "scored_from_frozen_predictions",
        "stage": stage,
        "split": "calibration",
        "offset": config["data"][f"{stage}_offset"],
        "count": len(rows),
        "historical_target_exposure": config["data"]["historical_target_exposure"],
        "commitment_sha256": sha256_file(commitment_path),
        "commitment_receipt_sha256": sha256_file(receipt_path),
        "frozen_prediction_roster_sha256": commitment[
            "frozen_prediction_roster_sha256"
        ],
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "mean_ssim": {
            name: float(np.mean([row["ssim"][name] for row in rows])) for name in ARM_NAMES
        },
        "comparisons": comparisons,
        "target_free_safety": safety,
        "quantitative_checks": quantitative_checks,
        "quantitative_pass": quantitative_pass,
        "selected_passing_winner": ARM_COMBINED if quantitative_pass else None,
        "manual_root_review_required_before_confirmation": True,
        "manual_root_review_completed": False,
        "root_manual_review_sheets": sheets,
        "manual_review_sheet_roster_sha256": sheet_roster_sha256,
        "root_manual_review_required_fields": {
            "reviewed_board_count": len(rows),
            "reviewed_all_full_canvas_triplets": True,
            "severe_artifacts": 0,
            "material_face_text_or_object_loss": False,
            "mask_halo_or_boundary_damage": False,
            "bind_report_commitment_receipt_preregistration_prediction_and_sheet_hashes": True,
        },
        "confirmation_authorized": False,
        "holdout_access": False,
        "competition_test_access": False,
        "rows": rows,
    }
    report_sha256 = write_json_exclusive_readonly(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": report_sha256,
                "mean_ssim": report["mean_ssim"],
                "comparisons": comparisons,
                "safety": safety,
                "quantitative_pass": quantitative_pass,
                "manual_sheets": sheets,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("Refusing to run without --run")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.phase == "train-sanity":
        train_sanity(device)
    elif args.phase == "freeze":
        freeze(args.stage, device)
    else:
        score(args.stage)


if __name__ == "__main__":
    main()
