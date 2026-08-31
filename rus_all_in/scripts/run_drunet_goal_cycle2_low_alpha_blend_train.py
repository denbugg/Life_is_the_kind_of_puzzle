#!/usr/bin/env python3
"""Freeze and score the bounded cycle2 B/C low-alpha train-only rescue."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aiijc_puzzle.drunet_goal_cycle2 import tile_flatness_counts
from aiijc_puzzle.nlm_luma_chroma import safety_summary, structure_diagnostics
from aiijc_puzzle.protocol import IMAGE_SIZE, contest_ssim, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "configs/drunet_goal_cycle2_low_alpha_blend_train_preregistered_v1.json"
)
CONFIG_SHA256 = "f19f743659bceaf4b7d237ce581ce4162a4c4d3b6a042cf2551f02f8af84a85e"
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
PARENT_ROOT = PROJECT_ROOT / "outputs/drunet-goal-cycle2-v2/train-offset528-count16"
PARENT_COMMITMENT = PARENT_ROOT / "prediction-commitment.json"
PARENT_RECEIPT = Path(f"{PARENT_ROOT}.commitment-receipt.json")
TARGETS = PROJECT_ROOT / "data/raw/train/targets"
OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/drunet-goal-cycle2-low-alpha-blend-v1/train-offset528-count16"
)
COMMITMENT_PATH = OUTPUT_ROOT / "prediction-commitment.json"
SAFETY_PATH = OUTPUT_ROOT / "target-free-safety-decision.json"
RECEIPT_PATH = Path(f"{OUTPUT_ROOT}.commitment-receipt.json")
SELECTION_PATH = OUTPUT_ROOT / "selection-decision.json"
REPORT_PATH = OUTPUT_ROOT / "report.json"

PARENT_COMMITMENT_SHA256 = (
    "d8eb8314a6a281ad62371bd884267247376fa3d2ee09b71840742dfec4177120"
)
PARENT_RECEIPT_SHA256 = (
    "4ccb02425cd9538606e54d17cc08127af4a3a9eaff37657a68d61a73ba5ddbc1"
)
MANIFEST_SHA256 = "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
PROTOCOL_DIGEST = "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
PARENT_ROSTER_SHA256 = "2c449f87c7444da1334fd6cb1d14c4ff9d4c139ec368a87f175821bd430dfcb4"

BASELINE = "B_drunet50_protected_h28_h50_t60"
SOURCE_C = "C_drunet50_h28_tilewise_drunet30_alpha_0_5"
CANDIDATES = (
    ("E_B90_C10", 9, 1, 10),
    ("F_B80_C20", 8, 2, 10),
    ("G_B70_C30", 7, 3, 10),
)
SELECTION_COUNT = 8
VERIFICATION_COUNT = 8
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "score"), required=True)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def require_readonly(path: Path) -> None:
    if path.stat().st_mode & WRITE_BITS:
        raise PermissionError(f"integrity-bound artifact is writable: {path}")


def write_bytes_exclusive_readonly(path: Path, payload: bytes) -> str:
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


def write_json_exclusive_readonly(path: Path, value: Mapping[str, Any]) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return write_bytes_exclusive_readonly(path, payload)


def write_png_exclusive_readonly(path: Path, image: np.ndarray) -> str:
    validate_image(image)
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG", compress_level=6)
    return write_bytes_exclusive_readonly(path, buffer.getvalue())


def validate_image(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError(f"expected strict uint8 RGB 480x480, got {value.dtype} {value.shape}")
    return np.ascontiguousarray(value)


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(validate_image(image).tobytes()).hexdigest()


def load_rgb_verified(path: Path, expected_sha256: str) -> np.ndarray:
    require_readonly(path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def blend_half_up(
    baseline: np.ndarray,
    source_c: np.ndarray,
    baseline_numerator: int,
    source_c_numerator: int,
    denominator: int,
) -> np.ndarray:
    left = validate_image(baseline)
    right = validate_image(source_c)
    if (
        baseline_numerator <= 0
        or source_c_numerator <= 0
        or baseline_numerator + source_c_numerator != denominator
        or denominator != 10
    ):
        raise ValueError("blend must be one of the preregistered positive tenths")
    numerator = (
        baseline_numerator * left.astype(np.uint16)
        + source_c_numerator * right.astype(np.uint16)
        + denominator // 2
    )
    return np.ascontiguousarray((numerator // denominator).astype(np.uint8))


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/aiijc_puzzle/drunet_goal_cycle2.py",
        PROJECT_ROOT / "src/aiijc_puzzle/nlm_luma_chroma.py",
        PROJECT_ROOT / "src/aiijc_puzzle/protocol.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def load_bound_context() -> tuple[dict[str, Any], dict[str, Any]]:
    config_sidecar = Path(f"{CONFIG}.sha256")
    for path in (CONFIG, config_sidecar, PARENT_COMMITMENT, PARENT_RECEIPT):
        require_readonly(path)
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("low-alpha preregistration changed")
    if config_sidecar.read_text(encoding="utf-8").split()[0] != CONFIG_SHA256:
        raise ValueError("low-alpha preregistration sidecar changed")
    if sha256_file(PARENT_COMMITMENT) != PARENT_COMMITMENT_SHA256:
        raise ValueError("parent cycle2 commitment changed")
    if sha256_file(PARENT_RECEIPT) != PARENT_RECEIPT_SHA256:
        raise ValueError("parent cycle2 receipt changed")
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("validation manifest changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_COMMITMENT.read_text(encoding="utf-8"))
    if parent.get("frozen_prediction_roster_sha256") != PARENT_ROSTER_SHA256:
        raise ValueError("parent frozen prediction roster changed")
    if parent.get("split") != "train" or parent.get("offset") != 528:
        raise ValueError("parent panel changed")
    if parent.get("count") != 16 or len(parent.get("boards", ())) != 16:
        raise ValueError("parent board count changed")
    if parent.get("selection_count") != 8 or parent.get("verification_count") != 8:
        raise ValueError("parent panel halves changed")
    configured_candidates = tuple(
        (
            row["name"],
            row["B_numerator"],
            row["C_numerator"],
            row["denominator"],
        )
        for row in config["candidate_roster"]
    )
    if configured_candidates != CANDIDATES:
        raise ValueError("candidate roster changed")
    return config, parent


def flatness_summary(
    candidate_rows: Sequence[Mapping[str, int]],
    baseline_rows: Sequence[Mapping[str, int]],
) -> dict[str, Any]:
    exact_delta = np.asarray(
        [
            right["exact_spatially_constant_rgb_tiles"]
            - left["exact_spatially_constant_rgb_tiles"]
            for left, right in zip(baseline_rows, candidate_rows, strict=True)
        ],
        dtype=np.int64,
    )
    nearflat_delta = np.asarray(
        [
            right["near_flat_tiles_global_std_lt_2"]
            - left["near_flat_tiles_global_std_lt_2"]
            for left, right in zip(baseline_rows, candidate_rows, strict=True)
        ],
        dtype=np.int64,
    )
    return {
        "baseline_exact_constant_total": int(
            sum(row["exact_spatially_constant_rgb_tiles"] for row in baseline_rows)
        ),
        "candidate_exact_constant_total": int(
            sum(row["exact_spatially_constant_rgb_tiles"] for row in candidate_rows)
        ),
        "exact_constant_total_increase": int(exact_delta.sum()),
        "exact_constant_board_deltas": exact_delta.tolist(),
        "near_flat_std_lt_2_delta_mean": float(nearflat_delta.mean()),
        "near_flat_std_lt_2_delta_max": int(nearflat_delta.max()),
        "near_flat_std_lt_2_board_deltas": nearflat_delta.tolist(),
    }


def evaluate_safety(
    config: Mapping[str, Any],
    boards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    thresholds = config["target_free_safety_vs_B"]
    baseline_structure = [row["structure"][BASELINE] for row in boards]
    baseline_flatness = [row["tile_flatness"][BASELINE] for row in boards]
    candidates: dict[str, Any] = {}
    for name, _, _, _ in CANDIDATES:
        structure = safety_summary(
            [row["structure"][name] for row in boards],
            baseline_structure,
        )
        flatness = flatness_summary(
            [row["tile_flatness"][name] for row in boards],
            baseline_flatness,
        )
        checks = {
            "luma_mean": structure["mean_luminance_gradient_retention"]
            >= thresholds["mean_luminance_gradient_retention_at_least"],
            "luma_min": structure["minimum_luminance_gradient_retention"]
            >= thresholds["minimum_board_luminance_gradient_retention_at_least"],
            "chroma_mean": structure["mean_chroma_gradient_retention"]
            >= thresholds["mean_chroma_gradient_retention_at_least"],
            "chroma_min": structure["minimum_chroma_gradient_retention"]
            >= thresholds["minimum_board_chroma_gradient_retention_at_least"],
            "laplacian_mean": structure["mean_laplacian_retention"]
            >= thresholds["mean_laplacian_retention_at_least"],
            "laplacian_min": structure["minimum_laplacian_retention"]
            >= thresholds["minimum_board_laplacian_retention_at_least"],
            "grid_mean": structure["mean_grid_ratio_relative_to_baseline"]
            <= thresholds["mean_grid_ratio_at_most"],
            "grid_max": structure["maximum_grid_ratio_relative_to_baseline"]
            <= thresholds["maximum_board_grid_ratio_at_most"],
            "exact_constant": flatness["exact_constant_total_increase"]
            <= thresholds["total_exact_spatially_constant_tile_increase_at_most"],
            "nearflat_mean": flatness["near_flat_std_lt_2_delta_mean"]
            <= thresholds["mean_near_flat_std_lt_2_tile_count_increase_at_most"],
            "nearflat_max": flatness["near_flat_std_lt_2_delta_max"]
            <= thresholds["maximum_board_near_flat_std_lt_2_tile_count_increase_at_most"],
        }
        candidates[name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "structure_vs_B": structure,
            "flatness_vs_B": flatness,
        }
    eligible = [name for name, _, _, _ in CANDIDATES if candidates[name]["passed"]]
    return {
        "schema": "aiijc-drunet-goal-cycle2-low-alpha-target-free-safety-v1",
        "status": (
            "eligible_arms_frozen_before_target_scoring"
            if eligible
            else "no_safe_arm_stop_before_target_scoring"
        ),
        "panel": "all 16 reused train records; no target pixels accessed",
        "thresholds": thresholds,
        "candidates": candidates,
        "eligible_candidates_in_preregistered_order": eligible,
        "target_scoring_authorized": bool(eligible),
        "targets_accessed": False,
    }


def freeze() -> None:
    config, parent = load_bound_context()
    if OUTPUT_ROOT.exists() or RECEIPT_PATH.exists():
        raise FileExistsError("refusing to overwrite frozen low-alpha experiment")
    OUTPUT_ROOT.mkdir(parents=True)
    boards: list[dict[str, Any]] = []
    for item in parent["boards"]:
        filename = str(item["filename"])
        parent_predictions = item["predictions"]
        baseline_record = parent_predictions[BASELINE]
        source_c_record = parent_predictions[SOURCE_C]
        baseline_path = PARENT_ROOT / baseline_record["relative_path"]
        source_c_path = PARENT_ROOT / source_c_record["relative_path"]
        baseline = load_rgb_verified(baseline_path, baseline_record["png_sha256"])
        source_c = load_rgb_verified(source_c_path, source_c_record["png_sha256"])
        if image_digest(baseline) != baseline_record["pixel_sha256"]:
            raise ValueError(f"parent B pixel digest changed: {filename}")
        if image_digest(source_c) != source_c_record["pixel_sha256"]:
            raise ValueError(f"parent C pixel digest changed: {filename}")
        predictions: dict[str, np.ndarray] = {}
        prediction_records: dict[str, Any] = {}
        for name, b_weight, c_weight, denominator in CANDIDATES:
            prediction = blend_half_up(baseline, source_c, b_weight, c_weight, denominator)
            path = OUTPUT_ROOT / "predictions" / Path(filename).stem / f"{name}.png"
            predictions[name] = prediction
            prediction_records[name] = {
                "relative_path": str(path.relative_to(OUTPUT_ROOT)),
                "png_sha256": write_png_exclusive_readonly(path, prediction),
                "pixel_sha256": image_digest(prediction),
                "B_numerator": b_weight,
                "C_numerator": c_weight,
                "denominator": denominator,
            }
        structure = {BASELINE: structure_diagnostics(baseline)}
        structure.update(
            {name: structure_diagnostics(image) for name, image in predictions.items()}
        )
        tile_flatness = {BASELINE: tile_flatness_counts(baseline)}
        tile_flatness.update(
            {name: tile_flatness_counts(image) for name, image in predictions.items()}
        )
        boards.append(
            {
                "filename": filename,
                "input_sha256": item["input_sha256"],
                "parent_B": baseline_record,
                "parent_C": source_c_record,
                "predictions": prediction_records,
                "structure": structure,
                "tile_flatness": tile_flatness,
            }
        )
    roster_sha256 = hashlib.sha256(
        "\n".join(
            f"{row['filename']} "
            + " ".join(row["predictions"][name]["pixel_sha256"] for name, _, _, _ in CANDIDATES)
            for row in boards
        ).encode("utf-8")
    ).hexdigest()
    commitment = {
        "schema": "aiijc-drunet-goal-cycle2-low-alpha-train-commitment-v1",
        "status": "three_blend_roster_frozen_before_any_low_alpha_target_scoring",
        "historical_exposure": config["historical_exposure"],
        "split": "train",
        "offset": 528,
        "count": 16,
        "selection_count": SELECTION_COUNT,
        "verification_count": VERIFICATION_COUNT,
        "preregistration_sha256": CONFIG_SHA256,
        "parent_commitment_sha256": PARENT_COMMITMENT_SHA256,
        "parent_receipt_sha256": PARENT_RECEIPT_SHA256,
        "parent_prediction_roster_sha256": PARENT_ROSTER_SHA256,
        "candidate_names": [name for name, _, _, _ in CANDIDATES],
        "formula": config["pixel_formula"],
        "new_neural_inference": False,
        "targets_accessed_during_freeze": False,
        "source_sha256": source_hashes(),
        "frozen_prediction_roster_sha256": roster_sha256,
        "boards": boards,
    }
    commitment_sha256 = write_json_exclusive_readonly(COMMITMENT_PATH, commitment)
    safety = evaluate_safety(config, boards)
    safety["preregistration_sha256"] = CONFIG_SHA256
    safety["commitment_sha256"] = commitment_sha256
    safety["frozen_prediction_roster_sha256"] = roster_sha256
    safety_sha256 = write_json_exclusive_readonly(SAFETY_PATH, safety)
    for directory in sorted(
        (path for path in (OUTPUT_ROOT / "predictions").rglob("*") if path.is_dir()),
        reverse=True,
    ):
        os.chmod(directory, 0o555)
    os.chmod(OUTPUT_ROOT / "predictions", 0o555)
    receipt = {
        "schema": "aiijc-drunet-goal-cycle2-low-alpha-train-receipt-v1",
        "status": "config_predictions_and_safety_frozen_before_target_scoring",
        "preregistration_sha256": CONFIG_SHA256,
        "commitment_sha256": commitment_sha256,
        "target_free_safety_decision_sha256": safety_sha256,
        "frozen_prediction_roster_sha256": roster_sha256,
        "source_sha256": commitment["source_sha256"],
        "targets_accessed_before_receipt": False,
    }
    receipt_sha256 = write_json_exclusive_readonly(RECEIPT_PATH, receipt)
    print(
        json.dumps(
            {
                "commitment_sha256": commitment_sha256,
                "safety_sha256": safety_sha256,
                "receipt_sha256": receipt_sha256,
                "eligible_candidates": safety["eligible_candidates_in_preregistered_order"],
                "target_scoring_authorized": safety["target_scoring_authorized"],
                "safety": safety["candidates"],
            },
            indent=2,
        )
    )


def load_frozen() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path in (COMMITMENT_PATH, SAFETY_PATH, RECEIPT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen artifact: {path}")
        require_readonly(path)
    commitment = json.loads(COMMITMENT_PATH.read_text(encoding="utf-8"))
    safety = json.loads(SAFETY_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    expected = {
        "preregistration_sha256": CONFIG_SHA256,
        "commitment_sha256": sha256_file(COMMITMENT_PATH),
        "target_free_safety_decision_sha256": sha256_file(SAFETY_PATH),
        "frozen_prediction_roster_sha256": commitment["frozen_prediction_roster_sha256"],
        "source_sha256": source_hashes(),
        "targets_accessed_before_receipt": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"frozen receipt binding changed: {key}")
    frozen: dict[str, Any] = {}
    for board in commitment["boards"]:
        images: dict[str, np.ndarray] = {}
        baseline_record = board["parent_B"]
        baseline_path = PARENT_ROOT / baseline_record["relative_path"]
        baseline = load_rgb_verified(baseline_path, baseline_record["png_sha256"])
        if image_digest(baseline) != baseline_record["pixel_sha256"]:
            raise ValueError(f"parent B pixel digest changed: {board['filename']}")
        images[BASELINE] = baseline
        for name, _, _, _ in CANDIDATES:
            record = board["predictions"][name]
            path = OUTPUT_ROOT / record["relative_path"]
            image = load_rgb_verified(path, record["png_sha256"])
            if image_digest(image) != record["pixel_sha256"]:
                raise ValueError(f"frozen blend pixel digest changed: {path}")
            images[name] = image
        frozen[board["filename"]] = images
    return commitment, safety, frozen


def load_target_hashes() -> dict[str, str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != PROTOCOL_DIGEST:
        raise ValueError("protocol digest changed")
    return {
        str(row["filename"]): str(row["target_sha256"])
        for row in manifest["splits"]["train"]
    }


def comparison(rows: Sequence[Mapping[str, Any]], candidate: str) -> dict[str, Any]:
    contender = np.asarray([row["ssim"][candidate] for row in rows], dtype=np.float64)
    baseline = np.asarray([row["ssim"][BASELINE] for row in rows], dtype=np.float64)
    delta = contender - baseline
    return {
        "candidate": candidate,
        "candidate_mean_ssim": float(contender.mean()),
        "baseline_B_mean_ssim": float(baseline.mean()),
        "mean_delta_vs_B": float(delta.mean()),
        "wins_ties_losses_vs_B": [
            int(np.sum(delta > 0)),
            int(np.sum(delta == 0)),
            int(np.sum(delta < 0)),
        ],
        "board_deltas_vs_B": delta.tolist(),
    }


def score() -> None:
    config, parent = load_bound_context()
    if SELECTION_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("refusing to overwrite low-alpha score artifacts")
    commitment, safety, frozen = load_frozen()
    eligible = tuple(safety["eligible_candidates_in_preregistered_order"])
    if not eligible or not safety.get("target_scoring_authorized"):
        print(
            json.dumps(
                {
                    "status": "stopped_before_target_scoring",
                    "reason": "no preregistered arm passed every target-free safety condition",
                    "eligible_candidates": [],
                    "targets_accessed": False,
                    "line_decision": "reject_entire_low_alpha_blend_line",
                },
                indent=2,
            )
        )
        return
    target_hashes = load_target_hashes()
    boards = commitment["boards"]
    selection_rows: list[dict[str, Any]] = []
    for board in boards[:SELECTION_COUNT]:
        filename = str(board["filename"])
        target = load_rgb_verified(TARGETS / filename, target_hashes[filename])
        images = frozen[filename]
        selection_rows.append(
            {
                "filename": filename,
                "ssim": {
                    name: contest_ssim(target, images[name])
                    for name in (BASELINE, *eligible)
                },
            }
        )
    means = {
        name: float(np.mean([row["ssim"][name] for row in selection_rows]))
        for name in eligible
    }
    tie_order = tuple(config["selection_rule"]["exact_tie_order"])
    selected = max(eligible, key=lambda name: (means[name], -tie_order.index(name)))
    selection_vs_b = comparison(selection_rows, selected)
    selection_decision = {
        "schema": "aiijc-drunet-goal-cycle2-low-alpha-selection-decision-v1",
        "status": "selected_on_first_eight_before_last_eight_target_decode_in_this_run",
        "historical_exposure": config["historical_exposure"],
        "preregistration_sha256": CONFIG_SHA256,
        "commitment_sha256": sha256_file(COMMITMENT_PATH),
        "receipt_sha256": sha256_file(RECEIPT_PATH),
        "safety_decision_sha256": sha256_file(SAFETY_PATH),
        "eligible_candidates": list(eligible),
        "selection_means": means,
        "exact_tie_order": list(tie_order),
        "selected_candidate": selected,
        "selection_vs_B": selection_vs_b,
        "last_eight_targets_decoded_before_selection_decision": False,
        "selection_rows": selection_rows,
    }
    selection_sha256 = write_json_exclusive_readonly(SELECTION_PATH, selection_decision)
    verification_rows: list[dict[str, Any]] = []
    for board in boards[SELECTION_COUNT:]:
        filename = str(board["filename"])
        target = load_rgb_verified(TARGETS / filename, target_hashes[filename])
        images = frozen[filename]
        verification_rows.append(
            {
                "filename": filename,
                "ssim": {
                    BASELINE: contest_ssim(target, images[BASELINE]),
                    selected: contest_ssim(target, images[selected]),
                },
            }
        )
    verification_vs_b = comparison(verification_rows, selected)
    thresholds = config["acceptance_gate_vs_B"]
    checks = {
        "selection_mean_delta_at_least_0_0003": selection_vs_b["mean_delta_vs_B"]
        >= thresholds["selection_first8_mean_delta_at_least"],
        "selection_wins_at_least_6_of_8": selection_vs_b["wins_ties_losses_vs_B"][0]
        >= thresholds["selection_first8_wins_at_least"],
        "verification_mean_delta_at_least_0_0003": verification_vs_b["mean_delta_vs_B"]
        >= thresholds["verification_last8_mean_delta_at_least"],
        "verification_wins_at_least_6_of_8": verification_vs_b["wins_ties_losses_vs_B"][0]
        >= thresholds["verification_last8_wins_at_least"],
    }
    accepted = all(checks.values())
    report = {
        "schema": "aiijc-drunet-goal-cycle2-low-alpha-train-report-v1",
        "status": "adaptive_reused_train_only_rescue_screen",
        "historical_exposure": config["historical_exposure"],
        "split": "train",
        "offset": 528,
        "count": 16,
        "preregistration_sha256": CONFIG_SHA256,
        "commitment_sha256": sha256_file(COMMITMENT_PATH),
        "receipt_sha256": sha256_file(RECEIPT_PATH),
        "safety_decision_sha256": sha256_file(SAFETY_PATH),
        "selection_decision_sha256": selection_sha256,
        "selected_candidate": selected,
        "selection_vs_B": selection_vs_b,
        "verification_vs_B": verification_vs_b,
        "acceptance_checks": checks,
        "accepted": accepted,
        "line_decision": (
            "bounded_reused_train_rescue_pass_only_no_generalization_claim"
            if accepted
            else "reject_entire_low_alpha_blend_line"
        ),
        "C_remains_rejected": True,
        "D_remains_rejected": True,
        "new_inference": False,
        "calibration_access": False,
        "holdout_access": False,
        "competition_test_access": False,
        "production_changed": False,
        "selection_rows": selection_rows,
        "verification_rows": verification_rows,
    }
    report_sha256 = write_json_exclusive_readonly(REPORT_PATH, report)
    print(
        json.dumps(
            {
                "report_sha256": report_sha256,
                "selected_candidate": selected,
                "selection_vs_B": selection_vs_b,
                "verification_vs_B": verification_vs_b,
                "checks": checks,
                "accepted": accepted,
                "line_decision": report["line_decision"],
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("Refusing to run without --run")
    if args.phase == "freeze":
        freeze()
    else:
        score()


if __name__ == "__main__":
    main()
