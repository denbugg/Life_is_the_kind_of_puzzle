#!/usr/bin/env python3
"""Freeze and score the single preregistered DRUNet symmetric-halo arm."""

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
from aiijc_puzzle.drunet_symmetric_halo import (
    ARM_NAMES,
    BASELINE_B,
    HALO_BATCH_SIZE,
    PADDED_TILE_SIZE,
    SYMMETRIC_HALO,
    SYMMETRIC_HALO_B,
    render_symmetric_halo_arms,
)
from aiijc_puzzle.dualnaf_bounded_residual import paired_bootstrap_ci
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
CONFIG = PROJECT_ROOT / "configs/drunet_symmetric_halo_train512_preregistered_v1.json"
CONFIG_SHA256 = "662270187b1a93d85a7423ad7be52959a0df289bf2f9fafa277cfc693654dc09"
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
TARGETS = PROJECT_ROOT / "data/raw/train/targets"
ASSET_ROOT = PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f"
CHECKPOINT = ASSET_ROOT / "drunet_color.pth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/drunet-symmetric-halo-v1/train-offset512-count16"
COMMITMENT_PATH = OUTPUT_ROOT / "prediction-commitment.json"
RECEIPT_PATH = Path(f"{OUTPUT_ROOT}.commitment-receipt.json")
REPORT_PATH = OUTPUT_ROOT / "report.json"

EXPECTED_MANIFEST_SHA256 = "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
EXPECTED_PROTOCOL_DIGEST = "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
EXPECTED_ASSET_SHA256 = {
    "LICENSE": "448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5",
    "drunet_color.pth": "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4",
    "models/basicblock.py": "48406db8867394ac5ae233ebeec7711ac10acfc3a6bbf0072c33aa77d659b6fd",
    "models/network_unet.py": "8043b6350f1589d5f08892e3be0b4d12c5a502058014285107b7360696d12bf5",
}
TRAIN_OFFSET = 512
TRAIN_COUNT = 16
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
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
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
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
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
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def verify_assets() -> dict[str, str]:
    observed = {relative: sha256_file(ASSET_ROOT / relative) for relative in EXPECTED_ASSET_SHA256}
    if observed != EXPECTED_ASSET_SHA256:
        raise ValueError("official KAIR assets changed")
    return observed


def verify_bound_sources(config: Mapping[str, Any]) -> dict[str, str]:
    expected = dict(config["bound_source_sha256"])
    observed = {relative: sha256_file(PROJECT_ROOT / relative) for relative in expected}
    if observed != expected:
        raise ValueError("a preregistered source dependency changed")
    return observed


def source_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(Path(__file__).resolve().relative_to(PROJECT_ROOT)): sha256_file(
            Path(__file__).resolve()
        ),
        **verify_bound_sources(config),
    }


def verify_cycle2_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["cycle2_binding"]
    path_fields = (
        ("preregistration_path", "preregistration_sha256"),
        ("commitment_path", "commitment_sha256"),
        ("receipt_path", "receipt_sha256"),
        ("selection_decision_path", "selection_decision_sha256"),
        ("report_path", "report_sha256"),
    )
    observed: dict[str, str] = {}
    for path_key, digest_key in path_fields:
        path = PROJECT_ROOT / str(binding[path_key])
        require_readonly(path)
        digest = sha256_file(path)
        if digest != binding[digest_key]:
            raise ValueError(f"bound cycle2 artifact changed: {path_key}")
        observed[str(path.relative_to(PROJECT_ROOT))] = digest

    report = json.loads((PROJECT_ROOT / binding["report_path"]).read_text(encoding="utf-8"))
    decision = json.loads(
        (PROJECT_ROOT / binding["selection_decision_path"]).read_text(encoding="utf-8")
    )
    commitment = json.loads(
        (PROJECT_ROOT / binding["commitment_path"]).read_text(encoding="utf-8")
    )
    if report.get("train_gate_pass") is not False:
        raise ValueError("cycle2 formal train gate disposition changed")
    if report.get("calibration_authorized") is not False:
        raise ValueError("cycle2 unexpectedly authorized calibration")
    if (
        report.get("holdout_access") is not False
        or report.get("competition_test_access") is not False
    ):
        raise ValueError("cycle2 access contract changed")
    if decision["selection_rule_result"]["selected_candidate"] != "D_half_B_half_C":
        raise ValueError("cycle2 metric selection changed")
    preservation = report["preservation_by_arm_vs_current_D"]
    if preservation["C_drunet50_h28_tilewise_drunet30_alpha_0_5"]["passed"] is not False:
        raise ValueError("cycle2 C safety disposition changed")
    if preservation["D_half_B_half_C"]["passed"] is not False:
        raise ValueError("cycle2 D safety disposition changed")
    for section in ("selection_comparisons_vs_current_D", "verification_comparisons_vs_current_D"):
        if report[section][BASELINE_B]["wins_ties_losses"] != [8, 0, 0]:
            raise ValueError("cycle2 B no longer wins every paired board")
    if set(str(board["filename"]) for board in commitment["boards"]) != set(
        str(row["filename"]) for row in report["rows"]
    ):
        raise ValueError("cycle2 report and commitment board rosters differ")
    return {
        "artifact_sha256": observed,
        "filenames": [board["filename"] for board in commitment["boards"]],
    }


def load_context() -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    sidecar = Path(f"{CONFIG}.sha256")
    for path in (CONFIG, sidecar):
        require_readonly(path)
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("symmetric-halo preregistration changed")
    if sidecar.read_text(encoding="utf-8").split()[0] != CONFIG_SHA256:
        raise ValueError("symmetric-halo preregistration sidecar changed")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest file changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest self-digest is invalid")
    if manifest.get("protocol_digest") != EXPECTED_PROTOCOL_DIGEST:
        raise ValueError("protocol changed")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if tuple(config["arm_names"]) != ARM_NAMES:
        raise ValueError("arm roster differs from preregistration")
    fixed = config["fixed_algorithm"]
    expected_constants = {
        "edge_budget": EDGE_BUDGET,
        "drunet_sigma": 50.0,
        "baseline_model_batch_size": 144,
        "candidate_model_batch_size": HALO_BATCH_SIZE,
    }
    for key, value in expected_constants.items():
        if fixed[key] != value:
            raise ValueError(f"algorithm constant changed: {key}")
    if SYMMETRIC_HALO != 6 or PADDED_TILE_SIZE != 32:
        raise ValueError("compiled symmetric-halo geometry changed")
    verify_bound_sources(config)
    cycle2 = verify_cycle2_binding(config)

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
    expected_panel = config["train_panel"]
    if expected_panel["split"] != "train" or expected_panel["offset"] != TRAIN_OFFSET:
        raise ValueError("train split or offset changed")
    if expected_panel["count"] != TRAIN_COUNT:
        raise ValueError("train count changed")
    if names_digest(records) != expected_panel["filenames_sha256"]:
        raise ValueError("train filename roster changed")
    if input_roster_digest(records) != expected_panel["input_roster_sha256"]:
        raise ValueError("train input roster changed")
    if {str(record["filename"]) for record in records} & set(cycle2["filenames"]):
        raise ValueError("train512 panel overlaps bound cycle2 train528 panel")
    return config, records


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
    audit = audit_raw_permutation(dirty, raw, layout, restoration_applied_after_audit=True)
    if not audit.passed:
        raise RuntimeError("strict raw permutation audit failed")
    harmonized_tiles, harmonizer = apply_rgb_luma(ordered)
    predictions, diagnostics = render_symmetric_halo_arms(
        model,
        harmonized_tiles,
        device=device,
    )
    pixel_hashes = {name: image_digest(image) for name, image in predictions.items()}
    if len(set(pixel_hashes.values())) != len(ARM_NAMES):
        raise RuntimeError("baseline and symmetric-halo predictions are not distinct")
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
        raise FileExistsError("refusing to overwrite frozen symmetric-halo experiment")
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
                    "phase": "target_blind_train512_freeze",
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
        "schema": "aiijc-drunet-symmetric-halo-train512-commitment-v1",
        "status": "both_B_geometry_rosters_frozen_before_any_train512_target_decode",
        "split": "train",
        "offset": TRAIN_OFFSET,
        "count": TRAIN_COUNT,
        "historical_exposure": config["historical_exposure"],
        "targets_decoded_during_freeze": False,
        "preregistration_sha256": CONFIG_SHA256,
        "selection_digest": names_digest(records),
        "input_roster_sha256": input_roster_digest(records),
        "arm_names": list(ARM_NAMES),
        "fixed_algorithm": config["fixed_algorithm"],
        "geometry_contract": config["geometry_contract"],
        "cycle2_binding": verify_cycle2_binding(config)["artifact_sha256"],
        "source_sha256": source_hashes(config),
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
        "schema": "aiijc-drunet-symmetric-halo-train512-receipt-v1",
        "status": "committed_before_any_train512_target_decode",
        "commitment_sha256": commitment_sha256,
        "preregistration_sha256": CONFIG_SHA256,
        "selection_digest": names_digest(records),
        "input_roster_sha256": input_roster_digest(records),
        "frozen_prediction_roster_sha256": roster_sha256,
        "source_sha256": commitment["source_sha256"],
        "cycle2_binding": commitment["cycle2_binding"],
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


def load_frozen(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not COMMITMENT_PATH.is_file() or not RECEIPT_PATH.is_file():
        raise FileNotFoundError("target-blind symmetric-halo commitment/receipt is missing")
    require_readonly(COMMITMENT_PATH)
    require_readonly(RECEIPT_PATH)
    commitment = json.loads(COMMITMENT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    expected_receipt = {
        "commitment_sha256": sha256_file(COMMITMENT_PATH),
        "preregistration_sha256": CONFIG_SHA256,
        "source_sha256": source_hashes(config),
        "cycle2_binding": verify_cycle2_binding(config)["artifact_sha256"],
        "targets_decoded_before_receipt": False,
    }
    for key, value in expected_receipt.items():
        if receipt.get(key) != value:
            raise ValueError(f"symmetric-halo receipt binding changed: {key}")
    if commitment.get("source_sha256") != source_hashes(config):
        raise ValueError("source changed after symmetric-halo freeze")
    if commitment.get("asset_sha256") != verify_assets():
        raise ValueError("assets changed after symmetric-halo freeze")
    if commitment.get("targets_decoded_during_freeze") is not False:
        raise ValueError("commitment target-access assertion changed")

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


def preservation_summary(
    commitment: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    boards = commitment["boards"]
    baseline_structure = [board["diagnostics"]["structure"][BASELINE_B] for board in boards]
    candidate_structure = [
        board["diagnostics"]["structure"][SYMMETRIC_HALO_B] for board in boards
    ]
    structure = safety_summary(candidate_structure, baseline_structure)
    baseline_flat = [board["diagnostics"]["tile_flatness"][BASELINE_B] for board in boards]
    candidate_flat = [
        board["diagnostics"]["tile_flatness"][SYMMETRIC_HALO_B] for board in boards
    ]
    exact_delta = np.asarray(
        [
            right["exact_spatially_constant_rgb_tiles"]
            - left["exact_spatially_constant_rgb_tiles"]
            for left, right in zip(baseline_flat, candidate_flat, strict=True)
        ]
    )
    near2_delta = np.asarray(
        [
            right["near_flat_tiles_global_std_lt_2"] - left["near_flat_tiles_global_std_lt_2"]
            for left, right in zip(baseline_flat, candidate_flat, strict=True)
        ]
    )
    clipping_delta = np.asarray(
        [
            right["clipped_fraction"] - left["clipped_fraction"]
            for left, right in zip(baseline_structure, candidate_structure, strict=True)
        ]
    )
    thresholds = config["train_gate"]["target_free_safety_vs_baseline"]
    checks = {
        "mean_luma_gradient_ratio": bool(
            structure["mean_luminance_gradient_retention"]
            >= thresholds["mean_luma_gradient_ratio_at_least"]
        ),
        "minimum_board_luma_gradient_ratio": bool(
            structure["minimum_luminance_gradient_retention"]
            >= thresholds["minimum_board_luma_gradient_ratio_at_least"]
        ),
        "mean_chroma_gradient_ratio": bool(
            structure["mean_chroma_gradient_retention"]
            >= thresholds["mean_chroma_gradient_ratio_at_least"]
        ),
        "minimum_board_chroma_gradient_ratio": bool(
            structure["minimum_chroma_gradient_retention"]
            >= thresholds["minimum_board_chroma_gradient_ratio_at_least"]
        ),
        "mean_laplacian_ratio": bool(
            structure["mean_laplacian_retention"]
            >= thresholds["mean_laplacian_ratio_at_least"]
        ),
        "minimum_board_laplacian_ratio": bool(
            structure["minimum_laplacian_retention"]
            >= thresholds["minimum_board_laplacian_ratio_at_least"]
        ),
        "mean_grid_ratio": bool(
            structure["mean_grid_ratio_relative_to_baseline"]
            <= thresholds["mean_grid_ratio_at_most"]
        ),
        "maximum_board_grid_ratio": bool(
            structure["maximum_grid_ratio_relative_to_baseline"]
            <= thresholds["maximum_board_grid_ratio_at_most"]
        ),
        "total_exact_constant_tile_increase": bool(
            exact_delta.sum()
            <= thresholds["total_exact_spatially_constant_tile_increase_at_most"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "structure_vs_baseline_B": structure,
        "exact_constant_tile_delta_total_mean_max": [
            int(exact_delta.sum()),
            float(exact_delta.mean()),
            int(exact_delta.max()),
        ],
        "near_flat_std_lt_2_delta_mean_max": [float(near2_delta.mean()), int(near2_delta.max())],
        "clipped_fraction_delta_mean_max": [
            float(clipping_delta.mean()),
            float(clipping_delta.max()),
        ],
    }


def score() -> None:
    config, records = load_context()
    if REPORT_PATH.exists():
        raise FileExistsError("refusing to overwrite symmetric-halo train report")
    commitment, frozen = load_frozen(config)
    if names_digest(records) != commitment["selection_digest"]:
        raise ValueError("frozen train512 selection differs from preregistration")
    if len(frozen) != TRAIN_COUNT:
        raise ValueError("frozen prediction board count changed")

    rows: list[dict[str, Any]] = []
    for record, item in zip(records, frozen, strict=True):
        if record["filename"] != item["board"]["filename"]:
            raise ValueError("frozen train512 board order changed")
        target = load_rgb_verified(TARGETS / str(record["filename"]), str(record["target_sha256"]))
        rows.append(
            {
                "filename": record["filename"],
                "ssim": {
                    name: contest_ssim(target, item["predictions"][name]) for name in ARM_NAMES
                },
            }
        )

    baseline = np.asarray([row["ssim"][BASELINE_B] for row in rows], dtype=np.float64)
    candidate = np.asarray([row["ssim"][SYMMETRIC_HALO_B] for row in rows], dtype=np.float64)
    difference = candidate - baseline
    ci = paired_bootstrap_ci(difference)
    comparison = {
        "candidate_mean_ssim": float(candidate.mean()),
        "baseline_mean_ssim": float(baseline.mean()),
        "mean_delta": float(difference.mean()),
        "paired_bootstrap_ci95": list(ci),
        "wins_ties_losses": [
            int(np.sum(difference > 0)),
            int(np.sum(difference == 0)),
            int(np.sum(difference < 0)),
        ],
        "board_deltas": difference.tolist(),
    }
    preservation = preservation_summary(commitment, config)
    thresholds = config["train_gate"]
    gate_checks = {
        "mean_delta_at_least_0_001": bool(
            comparison["mean_delta"] >= thresholds["candidate_mean_ssim_delta_vs_baseline_at_least"]
        ),
        "paired_bootstrap_lower_strictly_above_zero": bool(
            ci[0] > thresholds["paired_bootstrap_ci95_lower_strictly_above"]
        ),
        "wins_at_least_12_of_16": bool(
            comparison["wins_ties_losses"][0] >= thresholds["wins_vs_baseline_at_least"]
        ),
        "target_free_safety": bool(preservation["passed"]),
    }
    report = {
        "schema": "aiijc-drunet-symmetric-halo-train512-report-v1",
        "status": "reused_train512_one_arm_geometry_ablation",
        "split": "train",
        "offset": TRAIN_OFFSET,
        "count": TRAIN_COUNT,
        "historical_exposure": config["historical_exposure"],
        "preregistration_sha256": CONFIG_SHA256,
        "commitment_sha256": sha256_file(COMMITMENT_PATH),
        "receipt_sha256": sha256_file(RECEIPT_PATH),
        "frozen_prediction_roster_sha256": commitment["frozen_prediction_roster_sha256"],
        "comparison": comparison,
        "preservation": preservation,
        "train_gate_checks": gate_checks,
        "train_gate_pass": all(gate_checks.values()),
        "manual_target_content_inspection": False,
        "calibration_authorized": False,
        "calibration_targets_accessed": False,
        "holdout_access": False,
        "competition_test_access": False,
        "production_change_authorized": False,
        "rows": rows,
    }
    report_sha256 = write_json_exclusive_readonly(REPORT_PATH, report)
    print(
        json.dumps(
            {
                "report": str(REPORT_PATH),
                "report_sha256": report_sha256,
                "comparison": comparison,
                "preservation": preservation,
                "train_gate_pass": report["train_gate_pass"],
                "further_data_access_authorized": False,
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
