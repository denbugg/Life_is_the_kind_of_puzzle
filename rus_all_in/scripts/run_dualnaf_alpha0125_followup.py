#!/usr/bin/env python3
"""Freeze then score the single-alpha bounded DualNAF follow-up."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.dualnaf_bounded_residual import (
    apply_frozen_tail,
    blend_same_index_tiles,
    image_digest,
    image_structure_diagnostics,
    paired_bootstrap_ci,
)
from aiijc_puzzle.legacy_upgrade import (
    atomic_write_png,
    directional_scores,
    layout_digest,
    solve_buddies,
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
from aiijc_puzzle.restoration_r6 import TileAwareDualNAFNet
from aiijc_puzzle.tilewise_renderer import render_tiles_independently

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = PROJECT_ROOT / "configs/dualnaf_alpha0125_followup_preregistered_v1.json"
PREREGISTRATION_SHA256 = "a11b246fc14b3c04b1e995892f66b87a895f73142d47f8afc886ec3a1c99e162"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs/restoration-r6/compliant-r6-medium-train256-step2000-h10.pt"
)
OUTPUT_PARENT = PROJECT_ROOT / "outputs/dualnaf-alpha0125-followup"
STAGE_ROOTS = {
    "primary": OUTPUT_PARENT / "primary-calibration-offset540-count48",
    "confirmation": OUTPUT_PARENT / "confirmation-calibration-offset588-count48",
}
PRIMARY_REPORT = STAGE_ROOTS["primary"] / "report.json"
PRIMARY_MANUAL_REVIEW = STAGE_ROOTS["primary"] / "manual-review.json"
ARM_ALPHAS = {
    "baseline_alpha_0": 0.0,
    "dualnaf_residual_alpha_0_125": 0.125,
}
CONTROL_ARM = "baseline_alpha_0"
CANDIDATE_ARM = "dualnaf_residual_alpha_0_125"
EXPECTED_MANIFEST_SHA256 = "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
EXPECTED_PROTOCOL_DIGEST = "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
EXPECTED_CHECKPOINT_SHA256 = "331322460c8af87e5d4760b075726979f0574a23209889c1e95b6b90f2eac1a9"
EDGE_BUDGET = 96
RENDERER_CONDITIONING_H = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "score"), required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_ROOTS), default="primary")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
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


def load_rgb_verified(path: Path, expected_hash: str) -> np.ndarray:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def choose_device(requested: str) -> torch.device:
    name = requested
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def load_context(
    manifest_path: Path, stage: str
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("follow-up preregistration hash changed")
    config = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    expected_arms = [
        {"name": CONTROL_ARM, "alpha": 0.0, "role": "control"},
        {"name": CANDIDATE_ARM, "alpha": 0.125, "role": "single_frozen_candidate"},
    ]
    if config.get("arms") != expected_arms:
        raise ValueError("follow-up arm roster differs from preregistration")
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("validation manifest file changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        raise RuntimeError("selected record count drifted")
    if names_digest(records) != data[f"{stage}_filenames_sha256"]:
        raise ValueError("selected filename roster differs from preregistration")
    if input_roster_digest(records) != data[f"{stage}_input_roster_sha256"]:
        raise ValueError("selected input roster differs from preregistration")
    return config, records


def load_model(path: Path, device: torch.device) -> tuple[TileAwareDualNAFNet, dict[str, Any]]:
    if sha256_file(path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen DualNAF checkpoint changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model_configuration = checkpoint.get("model_configuration")
    training = checkpoint.get("training_configuration")
    if not isinstance(model_configuration, Mapping) or not isinstance(training, Mapping):
        raise ValueError("checkpoint configuration is missing")
    if model_configuration.get("architecture") != "dual_naf":
        raise ValueError("checkpoint architecture differs from preregistration")
    if training.get("protocol_digest") != EXPECTED_PROTOCOL_DIGEST:
        raise ValueError("checkpoint protocol differs from preregistration")
    if training.get("nlm_h") != RENDERER_CONDITIONING_H:
        raise ValueError("checkpoint tile conditioning differs from preregistration")
    model = TileAwareDualNAFNet(
        base=int(model_configuration["base"]),
        depth=int(model_configuration["depth"]),
        blocks=int(model_configuration["blocks"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/aiijc_puzzle/dualnaf_bounded_residual.py",
        PROJECT_ROOT / "src/aiijc_puzzle/tilewise_renderer.py",
        PROJECT_ROOT / "src/aiijc_puzzle/restoration_r6.py",
        PROJECT_ROOT / "src/aiijc_puzzle/postassembly_harmonizer.py",
        PROJECT_ROOT / "src/aiijc_puzzle/pixel_tails.py",
        PROJECT_ROOT / "src/aiijc_puzzle/legacy_upgrade.py",
        PROJECT_ROOT / "src/aiijc_puzzle/protocol.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def require_confirmation_authorized() -> None:
    if not PRIMARY_REPORT.is_file() or not PRIMARY_MANUAL_REVIEW.is_file():
        raise RuntimeError("primary report and manual review are required before confirmation")
    report = json.loads(PRIMARY_REPORT.read_text(encoding="utf-8"))
    manual = json.loads(PRIMARY_MANUAL_REVIEW.read_text(encoding="utf-8"))
    if not report.get("quantitative_gate", {}).get("passed") or not manual.get("passed"):
        raise RuntimeError("follow-up primary gate failed; confirmation access is forbidden")


def freeze(args: argparse.Namespace) -> None:
    config, records = load_context(args.manifest, args.stage)
    if args.stage == "confirmation":
        require_confirmation_authorized()
    root = STAGE_ROOTS[args.stage]
    if root.exists():
        raise FileExistsError(f"refusing to overwrite frozen experiment directory: {root}")
    root.mkdir(parents=True)
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    started = perf_counter()
    boards: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = load_rgb_verified(args.inputs / filename, str(record["input_sha256"]))
        input_tiles = split_tiles(dirty)
        right, down = directional_scores(input_tiles, views=("bilateral",))["bilateral"]
        solved = solve_buddies(right, down, max_edges=EDGE_BUDGET)
        layout = np.asarray(solved.layout, dtype=np.int32)
        ordered_original = np.ascontiguousarray(input_tiles[layout])
        raw = assemble_tiles(ordered_original)
        audit = audit_raw_permutation(
            dirty, raw, layout, restoration_applied_after_audit=True
        )
        if not audit.passed:
            raise RuntimeError(f"strict raw permutation audit failed for {filename}")
        rendered_unordered, renderer_diagnostics = render_tiles_independently(
            model,
            input_tiles,
            device,
            nlm_h=RENDERER_CONDITIONING_H,
            batch_size=args.batch_size,
        )
        ordered_rendered = np.ascontiguousarray(rendered_unordered[layout])
        candidate_tiles = blend_same_index_tiles(
            ordered_original, ordered_rendered, ARM_ALPHAS[CANDIDATE_ARM]
        )
        baseline, baseline_tail = apply_frozen_tail(ordered_original)
        candidate, candidate_tail = apply_frozen_tail(candidate_tiles)
        predictions = {CONTROL_ARM: baseline, CANDIDATE_ARM: candidate}
        blend_delta = np.abs(
            candidate_tiles.astype(np.int16) - ordered_original.astype(np.int16)
        )
        arm_diagnostics = {
            CONTROL_ARM: {"alpha": 0.0, "tail": baseline_tail},
            CANDIDATE_ARM: {
                "alpha": 0.125,
                "blend_mean_abs_change": float(blend_delta.mean()),
                "blend_q99_abs_change": float(np.quantile(blend_delta, 0.99)),
                "blend_maximum_abs_change": int(blend_delta.max()),
                "tail": candidate_tail,
            },
        }
        prediction_records: dict[str, Any] = {}
        board_directory = root / "predictions" / Path(filename).stem
        for name, prediction in predictions.items():
            output_path = board_directory / f"{name}.png"
            file_hash = atomic_write_png(output_path, prediction)
            prediction_records[name] = {
                "relative_path": str(output_path.relative_to(root)),
                "png_sha256": file_hash,
                "pixel_sha256": image_digest(prediction),
                "structure": image_structure_diagnostics(prediction),
            }
        boards.append(
            {
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "layout_sha256": layout_digest(layout),
                "raw_sha256": image_digest(raw),
                "permutation_audit": audit.as_dict(),
                "solver": solved.solver,
                "objective": float(solved.objective),
                "renderer_diagnostics": renderer_diagnostics.as_dict(),
                "arm_diagnostics": arm_diagnostics,
                "predictions": prediction_records,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "target_blind_freeze",
                    "stage": args.stage,
                    "done": index,
                    "total": len(records),
                    "filename": filename,
                }
            ),
            flush=True,
        )
    roster_digest = hashlib.sha256(
        "\n".join(
            f"{board['filename']} "
            + " ".join(
                board["predictions"][name]["pixel_sha256"] for name in ARM_ALPHAS
            )
            for board in boards
        ).encode("utf-8")
    ).hexdigest()
    commitment = {
        "schema": "aiijc-dualnaf-alpha0125-followup-prediction-commitment-v1",
        "status": "frozen_before_any_target_decode_in_this_stage",
        "stage": args.stage,
        "split": "calibration",
        "offset": int(config["data"][f"{args.stage}_offset"]),
        "count": len(records),
        "selection_digest": names_digest(records),
        "historical_target_exposure": config["data"]["historical_exposure"],
        "targets_decoded_during_freeze": False,
        "holdout_access": False,
        "competition_test_access": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "manifest_sha256": sha256_file(args.manifest),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_model_configuration": checkpoint["model_configuration"],
        "arm_names": list(ARM_ALPHAS),
        "frozen_prediction_roster_sha256": roster_digest,
        "source_sha256": source_hashes(),
        "runtime_seconds": perf_counter() - started,
        "boards": boards,
    }
    commitment_path = root / "prediction-commitment.json"
    atomic_json(commitment_path, commitment)
    print(
        json.dumps(
            {
                "commitment": str(commitment_path.resolve()),
                "commitment_sha256": sha256_file(commitment_path),
                "frozen_prediction_roster_sha256": roster_digest,
            },
            indent=2,
        ),
        flush=True,
    )


def load_frozen_predictions(
    root: Path, commitment: Mapping[str, Any]
) -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    for board in commitment["boards"]:
        predictions: dict[str, np.ndarray] = {}
        if tuple(board["predictions"]) != tuple(ARM_ALPHAS):
            raise ValueError("frozen arm roster differs from preregistration")
        for name, prediction_record in board["predictions"].items():
            path = root / prediction_record["relative_path"]
            if sha256_file(path) != prediction_record["png_sha256"]:
                raise ValueError(f"frozen PNG changed: {path}")
            image = load_rgb_verified(path, str(prediction_record["png_sha256"]))
            if image_digest(image) != prediction_record["pixel_sha256"]:
                raise ValueError(f"frozen pixels changed: {path}")
            predictions[name] = image
        frozen.append({"board": board, "predictions": predictions})
    return frozen


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


def make_contact_sheets(
    root: Path, frozen: Sequence[Mapping[str, Any]]
) -> list[str]:
    directory = root / "manual-review-sheets"
    directory.mkdir(exist_ok=True)
    paths: list[str] = []
    for page_start in range(0, len(frozen), 6):
        canvas = Image.new("RGB", (516, 1638), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"left: {CONTROL_ARM} | right: {CANDIDATE_ARM}", fill="black")
        for local_index, item in enumerate(frozen[page_start : page_start + 6]):
            y = 34 + local_index * 267
            draw.text((8, y), str(item["board"]["filename"]), fill="black")
            baseline = Image.fromarray(item["predictions"][CONTROL_ARM]).resize(
                (240, 240), Image.Resampling.NEAREST
            )
            candidate = Image.fromarray(item["predictions"][CANDIDATE_ARM]).resize(
                (240, 240), Image.Resampling.NEAREST
            )
            canvas.paste(baseline, (8, y + 18))
            canvas.paste(candidate, (268, y + 18))
        path = directory / f"page-{page_start // 6 + 1}.png"
        atomic_audit_png(path, np.asarray(canvas, dtype=np.uint8))
        paths.append(str(path.relative_to(root)))
    return paths


def score(args: argparse.Namespace) -> None:
    config, records = load_context(args.manifest, args.stage)
    root = STAGE_ROOTS[args.stage]
    report_path = root / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite score report: {report_path}")
    commitment_path = root / "prediction-commitment.json"
    if not commitment_path.is_file():
        raise FileNotFoundError("prediction commitment must exist before target access")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    if commitment.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("commitment preregistration hash mismatch")
    if commitment.get("selection_digest") != names_digest(records):
        raise ValueError("commitment record roster mismatch")
    frozen = load_frozen_predictions(root, commitment)
    if len(frozen) != len(records):
        raise ValueError("commitment board count mismatch")

    rows: list[dict[str, Any]] = []
    for item, record in zip(frozen, records, strict=True):
        if item["board"]["filename"] != record["filename"]:
            raise ValueError("commitment and selected record order differ")
        target = load_rgb_verified(
            args.targets / str(record["filename"]), str(record["target_sha256"])
        )
        rows.append(
            {
                "filename": record["filename"],
                "ssim": {
                    name: contest_ssim(target, prediction)
                    for name, prediction in item["predictions"].items()
                },
            }
        )
    baseline_scores = np.asarray([row["ssim"][CONTROL_ARM] for row in rows])
    candidate_scores = np.asarray([row["ssim"][CANDIDATE_ARM] for row in rows])
    differences = candidate_scores - baseline_scores
    gain_ci = paired_bootstrap_ci(differences)
    wins = int(np.sum(differences > 0))
    gradient_ratios: list[float] = []
    clipped_increases: list[float] = []
    for item in frozen:
        baseline_structure = item["board"]["predictions"][CONTROL_ARM]["structure"]
        candidate_structure = item["board"]["predictions"][CANDIDATE_ARM]["structure"]
        gradient_ratios.append(
            float(candidate_structure["gradient_energy"] / baseline_structure["gradient_energy"])
        )
        clipped_increases.append(
            float(candidate_structure["clipped_fraction"] - baseline_structure["clipped_fraction"])
        )
    checks = {
        "candidate_mean_final_ssim_at_least_0_27": float(candidate_scores.mean()) >= 0.27,
        "paired_ci95_lower_vs_baseline_strictly_positive": gain_ci[0] > 0,
        "wins_at_least_32_of_48": wins >= 32,
        "all_gradient_energy_ratios_in_0_75_to_1_25": all(
            0.75 <= value <= 1.25 for value in gradient_ratios
        ),
        "maximum_clipped_fraction_increase_at_most_0_01": max(clipped_increases) <= 0.01,
    }
    sheets = make_contact_sheets(root, frozen)
    report = {
        "schema": "aiijc-dualnaf-alpha0125-followup-score-v1",
        "status": "scored_after_verified_prediction_commitment",
        "stage": args.stage,
        "split": "calibration",
        "offset": int(config["data"][f"{args.stage}_offset"]),
        "count": len(rows),
        "selection_digest": names_digest(records),
        "historical_target_exposure": config["data"]["historical_exposure"],
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "prediction_commitment_sha256": sha256_file(commitment_path),
        "all_prediction_files_verified_before_first_target_decode": True,
        "holdout_access": False,
        "competition_test_access": False,
        "baseline": {"name": CONTROL_ARM, "mean_ssim": float(baseline_scores.mean())},
        "candidate": {
            "name": CANDIDATE_ARM,
            "alpha": 0.125,
            "mean_ssim": float(candidate_scores.mean()),
            "mean_gain_vs_baseline": float(differences.mean()),
            "gain_ci95": list(gain_ci),
            "wins_ties_losses": [
                wins,
                int(np.sum(differences == 0)),
                int(np.sum(differences < 0)),
            ],
        },
        "target_free_preservation": {
            "gradient_energy_ratio_min_mean_max": [
                float(np.min(gradient_ratios)),
                float(np.mean(gradient_ratios)),
                float(np.max(gradient_ratios)),
            ],
            "clipped_fraction_increase_min_mean_max": [
                float(np.min(clipped_increases)),
                float(np.mean(clipped_increases)),
                float(np.max(clipped_increases)),
            ],
            "manual_contact_sheets": sheets,
            "manual_review_status": "pending independent visual inspection",
        },
        "quantitative_gate": {
            "checks": checks,
            "passed": all(checks.values()),
            "confirmation_authorized_only_after_separate_manual_review_pass": True,
        },
        "per_board": rows,
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "baseline": report["baseline"],
                "candidate": report["candidate"],
                "quantitative_gate": report["quantitative_gate"],
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("dry-run only; pass --run to execute the requested phase")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.phase == "freeze":
        freeze(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
