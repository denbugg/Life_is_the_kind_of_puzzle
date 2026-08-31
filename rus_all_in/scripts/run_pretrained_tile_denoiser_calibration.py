#!/usr/bin/env python3
"""Freeze then score the preregistered legal pretrained DRUNet tile tail."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
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
    atomic_write_png,
    directional_scores,
    layout_digest,
    solve_buddies,
)
from aiijc_puzzle.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.pretrained_tile_denoiser import (
    ARM_DRUNET,
    ARM_H20,
    ARM_H28,
    ARM_NAMES,
    board_safety_diagnostics,
    candidate_safety_ratios,
    load_drunet_color,
    render_frozen_drunet_arms,
)
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
PREREGISTRATION = PROJECT_ROOT / "configs/pretrained_drunet_tile_tail_preregistered_v1.json"
PREREGISTRATION_SHA256 = "43f62344d9b2302323780a92edadc2b5122d2ab97d6830c9a900cda504df3fdb"
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
TARGETS = PROJECT_ROOT / "data/raw/train/targets"
ASSET_ROOT = PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f"
CHECKPOINT = ASSET_ROOT / "drunet_color.pth"
OUTPUT_PARENT = PROJECT_ROOT / "outputs/pretrained-tile-denoiser"
STAGE_ROOTS = {
    "primary": OUTPUT_PARENT / "primary-calibration-offset384-count24",
    "confirmation": OUTPUT_PARENT / "confirmation-calibration-offset600-count24",
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
EDGE_BUDGET = 96
MODEL_BATCH_SIZE = 144
MODEL_SIGMA = 40
MODEL_PARAMETER_COUNT = 32_640_960


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "score"), required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_ROOTS), default="primary")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
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


def image_digest(image: np.ndarray) -> str:
    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError("pixel digest requires one strict uint8 RGB prediction")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


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


def load_context(stage: str) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("pretrained DRUNet preregistration changed")
    sidecar = Path(f"{PREREGISTRATION}.sha256")
    sidecar_digest = sidecar.read_text(encoding="utf-8").split()[0]
    if sidecar_digest != PREREGISTRATION_SHA256:
        raise ValueError("pretrained DRUNet preregistration sidecar changed")
    config = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if tuple(config["arm_names"]) != ARM_NAMES:
        raise ValueError("runtime arm roster differs from preregistration")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("validation manifest file changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest self-digest is invalid")
    if manifest.get("protocol_digest") != EXPECTED_PROTOCOL_DIGEST:
        raise ValueError("validation protocol differs from preregistration")
    data = config["data"]
    offset = int(data[f"{stage}_offset"])
    count = int(data[f"{stage}_count"])
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
        raise RuntimeError("selected calibration record count drifted")
    if names_digest(records) != data[f"{stage}_filenames_sha256"]:
        raise ValueError("selected filename roster differs from preregistration")
    if input_roster_digest(records) != data[f"{stage}_input_roster_sha256"]:
        raise ValueError("selected input roster differs from preregistration")
    return config, records


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/aiijc_puzzle/pretrained_tile_denoiser.py",
        PROJECT_ROOT / "src/aiijc_puzzle/legacy_upgrade.py",
        PROJECT_ROOT / "src/aiijc_puzzle/postassembly_harmonizer.py",
        PROJECT_ROOT / "src/aiijc_puzzle/pixel_tails.py",
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


def require_confirmation_authorized() -> None:
    if not PRIMARY_REPORT.is_file() or not PRIMARY_MANUAL_REVIEW.is_file():
        raise RuntimeError("primary report and root manual review are required")
    report = json.loads(PRIMARY_REPORT.read_text(encoding="utf-8"))
    manual = json.loads(PRIMARY_MANUAL_REVIEW.read_text(encoding="utf-8"))
    if report.get("selected_passing_winner") != ARM_DRUNET:
        raise RuntimeError("primary DRUNet candidate did not pass every quantitative gate")
    if (
        manual.get("reviewer") != "root"
        or manual.get("reviewed_arm") != ARM_DRUNET
        or manual.get("severe_artifacts") != 0
        or not manual.get("passed")
    ):
        raise RuntimeError("root manual preservation gate has not passed")


def freeze(stage: str, device: torch.device) -> None:
    config, records = load_context(stage)
    if stage == "confirmation":
        require_confirmation_authorized()
    root = STAGE_ROOTS[stage]
    if root.exists():
        raise FileExistsError(f"refusing to overwrite frozen experiment: {root}")
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
        dirty = load_rgb_verified(INPUTS / filename, str(record["input_sha256"]))
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
            raise RuntimeError(f"strict raw permutation audit failed for {filename}")
        harmonized_tiles, harmonizer = apply_rgb_luma(ordered)
        predictions, render_diagnostics = render_frozen_drunet_arms(
            model,
            harmonized_tiles,
            device=device,
            batch_size=MODEL_BATCH_SIZE,
        )
        pixel_hashes = {name: image_digest(image) for name, image in predictions.items()}
        all_distinct = len(set(pixel_hashes.values())) == len(ARM_NAMES)
        if not all_distinct:
            raise RuntimeError(f"requested predictions were not distinct for {filename}")
        board_directory = root / "predictions" / Path(filename).stem
        prediction_records: dict[str, Any] = {}
        for name, prediction in predictions.items():
            output_path = board_directory / f"{name}.png"
            prediction_records[name] = {
                "relative_path": str(output_path.relative_to(root)),
                "png_sha256": atomic_write_png(output_path, prediction),
                "pixel_sha256": pixel_hashes[name],
                "safety": board_safety_diagnostics(prediction),
            }
        boards.append(
            {
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "layout_sha256": layout_digest(layout),
                "raw_sha256": image_digest(raw),
                "raw_permutation_audit": audit.as_dict(),
                "solver": solved.solver,
                "objective": float(solved.objective),
                "all_predictions_distinct": all_distinct,
                "harmonizer": harmonizer,
                "render_diagnostics": render_diagnostics,
                "candidate_safety_vs_h28": candidate_safety_ratios(
                    predictions[ARM_H28], predictions[ARM_DRUNET]
                ),
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
                    "distinct": all_distinct,
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
        "schema": "aiijc-pretrained-drunet-tile-tail-commitment-v1",
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
        "manifest_sha256": sha256_file(MANIFEST),
        "arm_names": list(ARM_NAMES),
        "model": {
            "architecture": "KAIR colour DRUNet UNetRes",
            "parameter_count": parameter_count,
            "sigma_255": MODEL_SIGMA,
            "batch_size": MODEL_BATCH_SIZE,
            "tile_input": "20x20 RGB plus constant noise map",
            "padding": "same-tile reflection right=4 bottom=4 to 24x24",
            "crop": "top-left exact 20x20 after network",
            "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": assets["drunet_color.pth"],
            "strict_state_dict_load": True,
        },
        "license_and_vendor": {
            "official_repository": "https://github.com/cszn/KAIR",
            "official_commit": "fc1732f4a4514e42ce15e5b3a1e18c828af47a1e",
            "checkpoint_url": (
                "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth"
            ),
            "license": "MIT",
            "asset_sha256": assets,
            "submission_contains_model_code_or_checkpoint": False,
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
    atomic_json(path, commitment)
    print(
        json.dumps(
            {
                "commitment": str(path),
                "commitment_sha256": sha256_file(path),
                "prediction_roster_sha256": prediction_roster_sha256,
            },
            indent=2,
        )
    )


def load_frozen(stage: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = STAGE_ROOTS[stage]
    commitment_path = root / "prediction-commitment.json"
    if not commitment_path.is_file():
        raise FileNotFoundError("target-blind prediction commitment is missing")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    if commitment.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("frozen commitment preregistration binding changed")
    if commitment.get("source_sha256") != source_hashes():
        raise ValueError("source changed after target-blind commitment")
    if commitment.get("license_and_vendor", {}).get("asset_sha256") != verify_assets():
        raise ValueError("checkpoint or vendor assets changed after commitment")
    frozen: list[dict[str, Any]] = []
    for board in commitment["boards"]:
        predictions: dict[str, np.ndarray] = {}
        for name in ARM_NAMES:
            record = board["predictions"][name]
            path = root / record["relative_path"]
            if sha256_file(path) != record["png_sha256"]:
                raise ValueError(f"frozen PNG changed: {path}")
            image = load_rgb_verified(path, record["png_sha256"])
            if image_digest(image) != record["pixel_sha256"]:
                raise ValueError(f"frozen pixels changed: {path}")
            predictions[name] = image
        frozen.append({"board": board, "predictions": predictions})
    return commitment, frozen


def comparison_summary(rows: Sequence[Mapping[str, Any]], baseline: str) -> dict[str, Any]:
    candidate = np.asarray([row["ssim"][ARM_DRUNET] for row in rows], dtype=np.float64)
    control = np.asarray([row["ssim"][baseline] for row in rows], dtype=np.float64)
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
        "checks": {
            "ci95_lower_strictly_positive": ci[0] > 0,
            "wins_at_least_18_of_24": int(np.sum(difference > 0)) >= 18,
        },
    }


def safety_gate(commitment: Mapping[str, Any]) -> dict[str, Any]:
    rows = [board["candidate_safety_vs_h28"] for board in commitment["boards"]]
    gradient = np.asarray([row["gradient_ratio_vs_h28"] for row in rows])
    laplacian = np.asarray([row["laplacian_ratio_vs_h28"] for row in rows])
    grid = np.asarray([row["grid_seam_ratio_vs_h28"] for row in rows])
    spread = np.asarray([row["global_rgb_std_ratio_vs_h28"] for row in rows])
    rgb_shift = np.asarray([row["maximum_abs_rgb_mean_shift_vs_h28"] for row in rows])
    pixel_change = np.asarray([row["mean_abs_pixel_change_vs_h28"] for row in rows])
    clipping = np.asarray([row["clipped_fraction_increase_vs_h28"] for row in rows])
    checks = {
        "gradient_ratio_mean_at_least_0_85": float(gradient.mean()) >= 0.85,
        "gradient_ratio_min_at_least_0_80": float(gradient.min()) >= 0.80,
        "laplacian_ratio_mean_at_least_0_85": float(laplacian.mean()) >= 0.85,
        "laplacian_ratio_min_at_least_0_80": float(laplacian.min()) >= 0.80,
        "grid_seam_ratio_every_board_between_0_75_and_1_05": bool(
            np.all((grid >= 0.75) & (grid <= 1.05))
        ),
        "global_rgb_std_ratio_every_board_between_0_95_and_1_05": bool(
            np.all((spread >= 0.95) & (spread <= 1.05))
        ),
        "maximum_rgb_mean_shift_no_more_than_2": float(rgb_shift.max()) <= 2.0,
        "maximum_mean_abs_pixel_change_no_more_than_4": (
            float(pixel_change.max()) <= 4.0
        ),
        "maximum_clipping_increase_no_more_than_0_005": float(clipping.max()) <= 0.005,
        "all_raw_permutation_audits_pass": bool(
            commitment["all_boards_raw_permutation_pass"]
        ),
        "all_predictions_distinct": bool(commitment["all_boards_predictions_distinct"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": {
            "gradient_ratio_mean_min_max": [
                float(gradient.mean()),
                float(gradient.min()),
                float(gradient.max()),
            ],
            "laplacian_ratio_mean_min_max": [
                float(laplacian.mean()),
                float(laplacian.min()),
                float(laplacian.max()),
            ],
            "grid_seam_ratio_mean_min_max": [
                float(grid.mean()),
                float(grid.min()),
                float(grid.max()),
            ],
            "global_rgb_std_ratio_mean_min_max": [
                float(spread.mean()),
                float(spread.min()),
                float(spread.max()),
            ],
            "rgb_mean_shift_max": float(rgb_shift.max()),
            "mean_abs_pixel_change_mean_max": [
                float(pixel_change.mean()),
                float(pixel_change.max()),
            ],
            "clipping_increase_max": float(clipping.max()),
        },
    }


def atomic_audit_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            Image.fromarray(image, mode="RGB").save(stream, format="PNG", compress_level=6)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def make_contact_sheets(root: Path, frozen: Sequence[Mapping[str, Any]]) -> list[str]:
    directory = root / "root-manual-review-sheets"
    directory.mkdir(exist_ok=True)
    paths: list[str] = []
    for page_start in range(0, len(frozen), 4):
        page = Image.new("RGB", (980, 2056), "white")
        draw = ImageDraw.Draw(page)
        draw.text((8, 8), f"left: {ARM_H28} | right: {ARM_DRUNET}", fill="black")
        for local_index, item in enumerate(frozen[page_start : page_start + 4]):
            y = 34 + local_index * 505
            draw.text((8, y), str(item["board"]["filename"]), fill="black")
            page.paste(Image.fromarray(item["predictions"][ARM_H28]), (8, y + 18))
            page.paste(Image.fromarray(item["predictions"][ARM_DRUNET]), (492, y + 18))
        path = directory / f"page-{page_start // 4 + 1}.png"
        atomic_audit_png(path, np.asarray(page, dtype=np.uint8))
        paths.append(str(path.relative_to(root)))
    return paths


def score(stage: str) -> None:
    config, records = load_context(stage)
    if stage == "confirmation":
        require_confirmation_authorized()
    root = STAGE_ROOTS[stage]
    report_path = root / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite score report: {report_path}")
    commitment, frozen = load_frozen(stage)
    if names_digest(records) != commitment["selection_digest"]:
        raise ValueError("frozen record selection changed")
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
        baseline: comparison_summary(rows, baseline) for baseline in (ARM_H20, ARM_H28)
    }
    candidate_mean = float(np.mean([row["ssim"][ARM_DRUNET] for row in rows]))
    safety = safety_gate(commitment)
    quantitative_checks = {
        "candidate_mean_ssim_at_least_0_27": candidate_mean >= 0.27,
        "paired_ci_and_wins_pass_vs_h20": all(
            comparisons[ARM_H20]["checks"].values()
        ),
        "paired_ci_and_wins_pass_vs_h28": all(
            comparisons[ARM_H28]["checks"].values()
        ),
        "target_free_safety_gate_pass": safety["passed"],
    }
    quantitative_pass = all(quantitative_checks.values())
    sheets = make_contact_sheets(root, frozen) if quantitative_pass else []
    report = {
        "schema": "aiijc-pretrained-drunet-tile-tail-report-v1",
        "stage": stage,
        "status": "scored_from_frozen_predictions",
        "split": "calibration",
        "offset": config["data"][f"{stage}_offset"],
        "count": len(rows),
        "historical_target_exposure": config["data"]["historical_target_exposure"],
        "commitment_sha256": sha256_file(root / "prediction-commitment.json"),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "mean_ssim": {
            name: float(np.mean([row["ssim"][name] for row in rows])) for name in ARM_NAMES
        },
        "comparisons": comparisons,
        "target_free_safety": safety,
        "quantitative_checks": quantitative_checks,
        "quantitative_pass": quantitative_pass,
        "selected_passing_winner": ARM_DRUNET if quantitative_pass else None,
        "manual_root_review_required_before_confirmation": True,
        "manual_root_review_completed": False,
        "root_manual_review_sheets": sheets,
        "confirmation_authorized": False,
        "holdout_access": False,
        "competition_test_access": False,
        "rows": rows,
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
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
    if args.phase == "freeze":
        freeze(args.stage, device)
    else:
        score(args.stage)


if __name__ == "__main__":
    main()
