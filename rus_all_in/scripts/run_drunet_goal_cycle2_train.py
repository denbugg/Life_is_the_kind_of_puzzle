#!/usr/bin/env python3
"""Freeze and score the preregistered train-only DRUNet goal-cycle-2 roster."""

from __future__ import annotations

import argparse
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
from PIL import Image

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.drunet_goal_cycle2 import (
    ARM_NAMES,
    CANDIDATE_COMBINATION,
    CANDIDATE_POST_H28,
    CANDIDATE_SIGMA50,
    DIRECT_SIGMA,
    MODEL_BATCH_SIZE,
    POST_H28_SIGMA,
    REFERENCE_CURRENT_D,
    SOBEL_THRESHOLD,
    render_goal_cycle2_arms,
)
from aiijc_puzzle.legacy_upgrade import directional_scores, layout_digest, solve_buddies
from aiijc_puzzle.nlm_luma_chroma import safety_summary
from aiijc_puzzle.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
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
CONFIG = PROJECT_ROOT / "configs/drunet_goal_cycle2_train_preregistered_v2.json"
CONFIG_SHA256 = "b163d1060c2ce8a88890a7f671971f2603564d360bdf7f88940968d0665a00db"
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
TARGETS = PROJECT_ROOT / "data/raw/train/targets"
ASSET_ROOT = PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f"
CHECKPOINT = ASSET_ROOT / "drunet_color.pth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/drunet-goal-cycle2-v2/train-offset528-count16"
COMMITMENT_PATH = OUTPUT_ROOT / "prediction-commitment.json"
RECEIPT_PATH = Path(f"{OUTPUT_ROOT}.commitment-receipt.json")
REPORT_PATH = OUTPUT_ROOT / "report.json"
SELECTION_DECISION_PATH = OUTPUT_ROOT / "selection-decision.json"

EXPECTED_MANIFEST_SHA256 = "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
EXPECTED_PROTOCOL_DIGEST = "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
EXPECTED_ASSET_SHA256 = {
    "LICENSE": "448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5",
    "drunet_color.pth": "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4",
    "models/basicblock.py": "48406db8867394ac5ae233ebeec7711ac10acfc3a6bbf0072c33aa77d659b6fd",
    "models/network_unet.py": "8043b6350f1589d5f08892e3be0b4d12c5a502058014285107b7360696d12bf5",
}
TRAIN_OFFSET = 528
SELECTION_COUNT = 8
VERIFICATION_COUNT = 8
TRAIN_COUNT = SELECTION_COUNT + VERIFICATION_COUNT
EDGE_BUDGET = 96
MODEL_PARAMETER_COUNT = 32_640_960
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "score"), required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
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
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError("prediction must be strict uint8 RGB 480x480")
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG", compress_level=6)
    return write_bytes_exclusive_readonly(path, buffer.getvalue())


def load_rgb_verified(path: Path, expected_sha256: str) -> np.ndarray:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size
            != (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def names_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode("utf-8")
    ).hexdigest()


def input_roster_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(f"{record['filename']} {record['input_sha256']}" for record in records).encode(
            "utf-8"
        )
    ).hexdigest()


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/aiijc_puzzle/drunet_goal_cycle2.py",
        PROJECT_ROOT / "src/aiijc_puzzle/pretrained_drunet_protected_stack.py",
        PROJECT_ROOT / "src/aiijc_puzzle/pretrained_tile_denoiser.py",
        PROJECT_ROOT / "src/aiijc_puzzle/edge_protected_nlm.py",
        PROJECT_ROOT / "src/aiijc_puzzle/nlm_luma_chroma.py",
        PROJECT_ROOT / "src/aiijc_puzzle/legacy_upgrade.py",
        PROJECT_ROOT / "src/aiijc_puzzle/postassembly_harmonizer.py",
        PROJECT_ROOT / "src/aiijc_puzzle/compliant_atlas_decoder.py",
        PROJECT_ROOT / "src/aiijc_puzzle/protocol.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def verify_assets() -> dict[str, str]:
    observed = {relative: sha256_file(ASSET_ROOT / relative) for relative in EXPECTED_ASSET_SHA256}
    if observed != EXPECTED_ASSET_SHA256:
        raise ValueError("official KAIR assets changed")
    return observed


def load_context() -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    sidecar = Path(f"{CONFIG}.sha256")
    for path in (CONFIG, sidecar):
        require_readonly(path)
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("train preregistration changed")
    if sidecar.read_text(encoding="utf-8").split()[0] != CONFIG_SHA256:
        raise ValueError("train preregistration sidecar changed")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest file changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest self-digest is invalid")
    if manifest.get("protocol_digest") != EXPECTED_PROTOCOL_DIGEST:
        raise ValueError("protocol changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    records = tuple(
        select_manifest_records(
            manifest,
            "train",
            limit=TRAIN_OFFSET + TRAIN_COUNT,
            seed=EXPERIMENT_SUBSET_SEED,
            namespace=EXPERIMENT_SUBSET_NAMESPACE,
        )[TRAIN_OFFSET:]
    )
    if len(records) != TRAIN_COUNT:
        raise RuntimeError("train roster count drifted")
    expected = config["train_panel"]
    if names_digest(records) != expected["filenames_sha256"]:
        raise ValueError("train filename roster changed")
    if input_roster_digest(records) != expected["input_roster_sha256"]:
        raise ValueError("train input roster changed")
    if names_digest(records[:SELECTION_COUNT]) != expected["selection_filenames_sha256"]:
        raise ValueError("selection roster changed")
    if input_roster_digest(records[:SELECTION_COUNT]) != expected["selection_input_roster_sha256"]:
        raise ValueError("selection input roster changed")
    if names_digest(records[SELECTION_COUNT:]) != expected["verification_filenames_sha256"]:
        raise ValueError("verification roster changed")
    if (
        input_roster_digest(records[SELECTION_COUNT:])
        != expected["verification_input_roster_sha256"]
    ):
        raise ValueError("verification input roster changed")
    if {str(record["filename"]) for record in records[:SELECTION_COUNT]} & {
        str(record["filename"]) for record in records[SELECTION_COUNT:]
    }:
        raise ValueError("selection and verification record rosters overlap")
    if tuple(config["arm_names"]) != ARM_NAMES:
        raise ValueError("arm roster differs from preregistration")
    fixed = config["fixed_algorithm"]
    expected_constants = {
        "edge_budget": EDGE_BUDGET,
        "direct_drunet_sigma": DIRECT_SIGMA,
        "post_h28_drunet_sigma": POST_H28_SIGMA,
        "sigma50_mask_sobel_threshold": SOBEL_THRESHOLD,
        "model_batch_size": MODEL_BATCH_SIZE,
    }
    for key, value in expected_constants.items():
        if fixed[key] != value:
            raise ValueError(f"algorithm constant changed: {key}")
    return config, records


def apply_rgb_luma(ordered_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered_tiles, SeamGraphConfig())
    rgb_tiles = apply_rgb_offsets(ordered_tiles, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(
        rgb_tiles,
        LuminanceGainConfig(),
    )
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
        dirty,
        raw,
        layout,
        restoration_applied_after_audit=True,
    )
    if not audit.passed:
        raise RuntimeError("strict raw permutation audit failed")
    harmonized_tiles, harmonizer = apply_rgb_luma(ordered)
    predictions, diagnostics = render_goal_cycle2_arms(
        model,
        harmonized_tiles,
        device=device,
    )
    pixel_hashes = {name: image_digest(image) for name, image in predictions.items()}
    if len(set(pixel_hashes.values())) != len(ARM_NAMES):
        raise RuntimeError("goal-cycle-2 predictions are not all distinct")
    return {
        "layout": layout,
        "raw": raw,
        "audit": audit.as_dict(),
        "solver": solved.solver,
        "objective": float(solved.objective),
        "harmonizer": harmonizer,
        "predictions": predictions,
        "pixel_sha256": pixel_hashes,
        "diagnostics": diagnostics,
    }


def freeze(device: torch.device) -> None:
    config, records = load_context()
    if OUTPUT_ROOT.exists() or RECEIPT_PATH.exists():
        raise FileExistsError("refusing to overwrite frozen train experiment")
    canonical_device = str(config["runtime"]["canonical_prediction_device"])
    if device.type != canonical_device:
        raise RuntimeError(f"canonical train freeze requires {canonical_device}")
    assets = verify_assets()
    model = load_drunet_color(CHECKPOINT, device)
    if sum(parameter.numel() for parameter in model.parameters()) != MODEL_PARAMETER_COUNT:
        raise ValueError("DRUNet parameter count changed")
    OUTPUT_ROOT.mkdir(parents=True)
    started = perf_counter()
    boards: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = load_rgb_verified(INPUTS / filename, str(record["input_sha256"]))
        result = infer_board(dirty, model, device)
        board_root = OUTPUT_ROOT / "predictions" / Path(filename).stem
        prediction_records: dict[str, Any] = {}
        for name, prediction in result["predictions"].items():
            path = board_root / f"{name}.png"
            prediction_records[name] = {
                "relative_path": str(path.relative_to(OUTPUT_ROOT)),
                "png_sha256": write_png_exclusive_readonly(path, prediction),
                "pixel_sha256": result["pixel_sha256"][name],
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
                "harmonizer": result["harmonizer"],
                "all_predictions_distinct": True,
                "diagnostics": result["diagnostics"],
                "predictions": prediction_records,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "target_blind_train_freeze",
                    "done": index,
                    "total": len(records),
                    "filename": filename,
                }
            ),
            flush=True,
        )
    roster_sha256 = hashlib.sha256(
        "\n".join(
            f"{board['filename']} "
            + " ".join(board["predictions"][name]["pixel_sha256"] for name in ARM_NAMES)
            for board in boards
        ).encode("utf-8")
    ).hexdigest()
    commitment = {
        "schema": "aiijc-drunet-goal-cycle2-train-commitment-v2",
        "status": "four_arm_roster_frozen_before_any_further_train_target_decode",
        "split": "train",
        "offset": TRAIN_OFFSET,
        "count": TRAIN_COUNT,
        "selection_count": SELECTION_COUNT,
        "verification_count": VERIFICATION_COUNT,
        "historical_exposure": config["historical_exposure"],
        "targets_decoded_during_freeze": False,
        "preregistration_sha256": CONFIG_SHA256,
        "selection_digest": names_digest(records),
        "input_roster_sha256": input_roster_digest(records),
        "arm_names": list(ARM_NAMES),
        "fixed_algorithm": config["fixed_algorithm"],
        "geometry_contract": config["geometry_contract"],
        "source_sha256": source_hashes(),
        "asset_sha256": assets,
        "runtime": {"device": str(device), "torch": torch.__version__},
        "frozen_prediction_roster_sha256": roster_sha256,
        "all_raw_permutation_audits_pass": all(
            board["raw_permutation_audit"]["passed"] for board in boards
        ),
        "all_predictions_distinct": all(board["all_predictions_distinct"] for board in boards),
        "runtime_seconds": perf_counter() - started,
        "boards": boards,
    }
    commitment_sha256 = write_json_exclusive_readonly(COMMITMENT_PATH, commitment)
    for directory in sorted(
        (path for path in (OUTPUT_ROOT / "predictions").rglob("*") if path.is_dir()),
        reverse=True,
    ):
        os.chmod(directory, 0o555)
    os.chmod(OUTPUT_ROOT / "predictions", 0o555)
    receipt = {
        "schema": "aiijc-drunet-goal-cycle2-train-receipt-v2",
        "status": "committed_before_any_further_train_target_decode",
        "commitment_sha256": commitment_sha256,
        "preregistration_sha256": CONFIG_SHA256,
        "selection_digest": names_digest(records),
        "input_roster_sha256": input_roster_digest(records),
        "frozen_prediction_roster_sha256": roster_sha256,
        "source_sha256": commitment["source_sha256"],
        "targets_decoded_before_receipt": False,
    }
    receipt_sha256 = write_json_exclusive_readonly(RECEIPT_PATH, receipt)
    print(
        json.dumps(
            {
                "commitment": str(COMMITMENT_PATH),
                "commitment_sha256": commitment_sha256,
                "receipt": str(RECEIPT_PATH),
                "receipt_sha256": receipt_sha256,
                "prediction_roster_sha256": roster_sha256,
            },
            indent=2,
        )
    )


def load_frozen() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not COMMITMENT_PATH.is_file() or not RECEIPT_PATH.is_file():
        raise FileNotFoundError("target-blind train commitment/receipt is missing")
    require_readonly(COMMITMENT_PATH)
    require_readonly(RECEIPT_PATH)
    commitment = json.loads(COMMITMENT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    expected_receipt = {
        "commitment_sha256": sha256_file(COMMITMENT_PATH),
        "preregistration_sha256": CONFIG_SHA256,
        "source_sha256": source_hashes(),
        "targets_decoded_before_receipt": False,
    }
    for key, value in expected_receipt.items():
        if receipt.get(key) != value:
            raise ValueError(f"train receipt binding changed: {key}")
    if commitment.get("source_sha256") != source_hashes():
        raise ValueError("source changed after train freeze")
    if commitment.get("asset_sha256") != verify_assets():
        raise ValueError("assets changed after train freeze")
    frozen: list[dict[str, Any]] = []
    for board in commitment["boards"]:
        predictions: dict[str, np.ndarray] = {}
        for name in ARM_NAMES:
            record = board["predictions"][name]
            path = OUTPUT_ROOT / record["relative_path"]
            require_readonly(path)
            if sha256_file(path) != record["png_sha256"]:
                raise ValueError(f"frozen PNG changed: {path}")
            image = load_rgb_verified(path, record["png_sha256"])
            if image_digest(image) != record["pixel_sha256"]:
                raise ValueError(f"frozen pixels changed: {path}")
            predictions[name] = image
        frozen.append({"board": board, "predictions": predictions})
    return commitment, frozen


def comparison(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    baseline: str,
) -> dict[str, Any]:
    contender = np.asarray([row["ssim"][candidate] for row in rows])
    control = np.asarray([row["ssim"][baseline] for row in rows])
    difference = contender - control
    return {
        "candidate_mean_ssim": float(contender.mean()),
        "baseline_mean_ssim": float(control.mean()),
        "mean_delta": float(difference.mean()),
        "wins_ties_losses": [
            int(np.sum(difference > 0)),
            int(np.sum(difference == 0)),
            int(np.sum(difference < 0)),
        ],
        "board_deltas": difference.tolist(),
    }


def select_candidate(selection_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base_candidates = (CANDIDATE_SIGMA50, CANDIDATE_POST_H28)
    means = {
        name: float(np.mean([row["ssim"][name] for row in selection_rows])) for name in ARM_NAMES
    }
    best_base = max(base_candidates, key=lambda name: (means[name], -ARM_NAMES.index(name)))
    combo_comparison = comparison(
        selection_rows,
        CANDIDATE_COMBINATION,
        best_base,
    )
    combo_eligible = bool(
        means[CANDIDATE_COMBINATION] >= means[best_base] + 0.0002
        and combo_comparison["wins_ties_losses"][0] >= 5
    )
    eligible = (*base_candidates, *((CANDIDATE_COMBINATION,) if combo_eligible else ()))
    winner = max(eligible, key=lambda name: (means[name], -ARM_NAMES.index(name)))
    return {
        "selection_means": means,
        "best_base_candidate": best_base,
        "combination_rule": {
            "formula": "exact uint8 half-up 0.5*B + 0.5*C",
            "required_margin_over_best_base": 0.0002,
            "required_wins_over_best_base": 5,
            "comparison": combo_comparison,
            "eligible": combo_eligible,
        },
        "selected_candidate": winner,
    }


def preservation_summary(
    commitment: Mapping[str, Any],
    candidate: str,
) -> dict[str, Any]:
    boards = commitment["boards"]
    baseline_structure = [
        board["diagnostics"]["structure"][REFERENCE_CURRENT_D] for board in boards
    ]
    candidate_structure = [board["diagnostics"]["structure"][candidate] for board in boards]
    structure = safety_summary(candidate_structure, baseline_structure)
    baseline_flat = [board["diagnostics"]["tile_flatness"][REFERENCE_CURRENT_D] for board in boards]
    candidate_flat = [board["diagnostics"]["tile_flatness"][candidate] for board in boards]
    exact_delta = np.asarray(
        [
            right["exact_spatially_constant_rgb_tiles"] - left["exact_spatially_constant_rgb_tiles"]
            for left, right in zip(baseline_flat, candidate_flat, strict=True)
        ]
    )
    near2_delta = np.asarray(
        [
            right["near_flat_tiles_global_std_lt_2"] - left["near_flat_tiles_global_std_lt_2"]
            for left, right in zip(baseline_flat, candidate_flat, strict=True)
        ]
    )
    flatness = {
        "reference_exact_constant_total": int(
            sum(row["exact_spatially_constant_rgb_tiles"] for row in baseline_flat)
        ),
        "candidate_exact_constant_total": int(
            sum(row["exact_spatially_constant_rgb_tiles"] for row in candidate_flat)
        ),
        "exact_constant_delta_mean_max": [
            float(exact_delta.mean()),
            int(exact_delta.max()),
        ],
        "reference_near_flat_std_lt_2_mean_max": [
            float(np.mean([row["near_flat_tiles_global_std_lt_2"] for row in baseline_flat])),
            int(max(row["near_flat_tiles_global_std_lt_2"] for row in baseline_flat)),
        ],
        "candidate_near_flat_std_lt_2_mean_max": [
            float(np.mean([row["near_flat_tiles_global_std_lt_2"] for row in candidate_flat])),
            int(max(row["near_flat_tiles_global_std_lt_2"] for row in candidate_flat)),
        ],
        "near_flat_std_lt_2_delta_mean_max": [
            float(near2_delta.mean()),
            int(near2_delta.max()),
        ],
    }
    checks = {
        "mean_luma_gradient_at_least_current_D": (
            structure["mean_luminance_gradient_retention"] >= 1.0
        ),
        "minimum_board_luma_gradient_ratio_at_least_0_95": (
            structure["minimum_luminance_gradient_retention"] >= 0.95
        ),
        "mean_chroma_gradient_at_least_current_D": (
            structure["mean_chroma_gradient_retention"] >= 1.0
        ),
        "minimum_board_chroma_gradient_ratio_at_least_0_95": (
            structure["minimum_chroma_gradient_retention"] >= 0.95
        ),
        "mean_laplacian_at_least_current_D": (structure["mean_laplacian_retention"] >= 1.0),
        "minimum_board_laplacian_ratio_at_least_0_95": (
            structure["minimum_laplacian_retention"] >= 0.95
        ),
        "mean_grid_ratio_at_most_1_02": (structure["mean_grid_ratio_relative_to_baseline"] <= 1.02),
        "maximum_board_grid_ratio_at_most_1_10": (
            structure["maximum_grid_ratio_relative_to_baseline"] <= 1.10
        ),
        "no_increase_in_total_exact_constant_tiles": (
            flatness["candidate_exact_constant_total"] <= flatness["reference_exact_constant_total"]
        ),
        "near_flat_mean_increase_at_most_2_tiles": near2_delta.mean() <= 2.0,
        "near_flat_maximum_board_increase_at_most_6_tiles": near2_delta.max() <= 6,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "structure_vs_current_D": structure,
        "tile_flatness_vs_current_D": flatness,
    }


def score() -> None:
    config, records = load_context()
    if REPORT_PATH.exists() or SELECTION_DECISION_PATH.exists():
        raise FileExistsError("refusing to overwrite train score artifacts")
    commitment, frozen = load_frozen()
    if names_digest(records) != commitment["selection_digest"]:
        raise ValueError("frozen train selection differs from preregistration")

    selection_rows: list[dict[str, Any]] = []
    for record, item in zip(
        records[:SELECTION_COUNT],
        frozen[:SELECTION_COUNT],
        strict=True,
    ):
        if record["filename"] != item["board"]["filename"]:
            raise ValueError("frozen train board order changed")
        target = load_rgb_verified(
            TARGETS / str(record["filename"]),
            str(record["target_sha256"]),
        )
        selection_rows.append(
            {
                "filename": record["filename"],
                "ssim": {
                    name: contest_ssim(target, item["predictions"][name]) for name in ARM_NAMES
                },
            }
        )
    selection = select_candidate(selection_rows)
    candidate = str(selection["selected_candidate"])
    selection_decision = {
        "schema": "aiijc-drunet-goal-cycle2-selection-decision-v2",
        "status": "candidate_selected_before_verification_target_decode_in_this_run",
        "historical_exposure": config["historical_exposure"],
        "preregistration_sha256": CONFIG_SHA256,
        "commitment_sha256": sha256_file(COMMITMENT_PATH),
        "receipt_sha256": sha256_file(RECEIPT_PATH),
        "frozen_prediction_roster_sha256": commitment["frozen_prediction_roster_sha256"],
        "selection_filenames_sha256": names_digest(records[:SELECTION_COUNT]),
        "verification_targets_decoded_before_decision": False,
        "selection_rule_result": selection,
        "selection_rows": selection_rows,
    }
    selection_decision_sha256 = write_json_exclusive_readonly(
        SELECTION_DECISION_PATH,
        selection_decision,
    )

    verification_rows: list[dict[str, Any]] = []
    for record, item in zip(
        records[SELECTION_COUNT:],
        frozen[SELECTION_COUNT:],
        strict=True,
    ):
        if record["filename"] != item["board"]["filename"]:
            raise ValueError("frozen train verification board order changed")
        target = load_rgb_verified(
            TARGETS / str(record["filename"]),
            str(record["target_sha256"]),
        )
        verification_rows.append(
            {
                "filename": record["filename"],
                "ssim": {
                    name: contest_ssim(target, item["predictions"][name]) for name in ARM_NAMES
                },
            }
        )
    rows = [*selection_rows, *verification_rows]
    selection_vs_reference = comparison(
        selection_rows,
        candidate,
        REFERENCE_CURRENT_D,
    )
    verification_vs_reference = comparison(
        verification_rows,
        candidate,
        REFERENCE_CURRENT_D,
    )
    preservation = preservation_summary(commitment, candidate)
    gate_checks = {
        "selection_mean_delta_positive_and_wins_at_least_6_of_8": (
            selection_vs_reference["mean_delta"] > 0
            and selection_vs_reference["wins_ties_losses"][0] >= 6
        ),
        "verification_mean_delta_positive_and_wins_at_least_6_of_8": (
            verification_vs_reference["mean_delta"] > 0
            and verification_vs_reference["wins_ties_losses"][0] >= 6
        ),
        "preservation_at_least_current_D_gate": preservation["passed"],
    }
    report = {
        "schema": "aiijc-drunet-goal-cycle2-train-report-v2",
        "status": "formal_reproduction_on_previously_decoded_train_records",
        "split": "train",
        "offset": TRAIN_OFFSET,
        "count": TRAIN_COUNT,
        "selection_count": SELECTION_COUNT,
        "verification_count": VERIFICATION_COUNT,
        "historical_exposure": config["historical_exposure"],
        "preregistration_sha256": CONFIG_SHA256,
        "commitment_sha256": sha256_file(COMMITMENT_PATH),
        "receipt_sha256": sha256_file(RECEIPT_PATH),
        "selection_decision_sha256": selection_decision_sha256,
        "frozen_prediction_roster_sha256": commitment["frozen_prediction_roster_sha256"],
        "mean_ssim_all_16_descriptive": {
            name: float(np.mean([row["ssim"][name] for row in rows])) for name in ARM_NAMES
        },
        "selection_rule_result": selection,
        "selection_vs_current_D": selection_vs_reference,
        "verification_vs_current_D": verification_vs_reference,
        "preservation": preservation,
        "train_gate_checks": gate_checks,
        "train_gate_pass": all(gate_checks.values()),
        "calibration_authorized": all(gate_checks.values()),
        "calibration_targets_accessed": False,
        "holdout_access": False,
        "competition_test_access": False,
        "rows": rows,
    }
    report_sha256 = write_json_exclusive_readonly(REPORT_PATH, report)
    print(
        json.dumps(
            {
                "report": str(REPORT_PATH),
                "report_sha256": report_sha256,
                "selected_candidate": candidate,
                "means": report["mean_ssim_all_16_descriptive"],
                "selection_vs_current_D": selection_vs_reference,
                "verification_vs_current_D": verification_vs_reference,
                "preservation": preservation,
                "train_gate_pass": report["train_gate_pass"],
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
        raise RuntimeError("MPS requested but unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.phase == "freeze":
        freeze(device)
    else:
        score()


if __name__ == "__main__":
    main()
