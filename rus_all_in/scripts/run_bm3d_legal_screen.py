#!/usr/bin/env python3
"""Freeze then score the preregistered legal BM3D restoration screen."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from importlib.metadata import distribution, version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from aiijc_puzzle.bm3d_screen import (
    ARM_NAMES,
    CANDIDATE_ARMS,
    CONTROL_ARM,
    all_predictions_distinct,
    image_digest,
    render_arms,
    structure_diagnostics,
)
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
PREREGISTRATION = PROJECT_ROOT / "configs/bm3d_legal_screen_preregistered_v1.json"
PREREGISTRATION_SHA256 = "934506c22420aba4aabe7d3c0ba786482a7c3d5ca700fa3866c6a55c44ea15d4"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_INPUTS = PROJECT_ROOT / "data/raw/train/inputs"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
OUTPUT_PARENT = PROJECT_ROOT / "outputs/bm3d-legal-screen"
STAGE_ROOTS = {
    "primary": OUTPUT_PARENT / "primary-calibration-offset276-count24",
    "confirmation": OUTPUT_PARENT / "confirmation-calibration-offset252-count24",
}
PRIMARY_REPORT = STAGE_ROOTS["primary"] / "report.json"
PRIMARY_MANUAL_REVIEW = STAGE_ROOTS["primary"] / "manual-review.json"
EXPECTED_MANIFEST_SHA256 = "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
EXPECTED_PROTOCOL_DIGEST = "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
EXPECTED_VERSIONS = {
    "bm3d": "4.0.3",
    "bm4d": "4.2.5",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "PyWavelets": "1.9.0",
}
EDGE_BUDGET = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "score"), required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_ROOTS), default="primary")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
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


def package_runtime() -> dict[str, Any]:
    observed = {name: version(name) for name in EXPECTED_VERSIONS}
    if observed != EXPECTED_VERSIONS:
        raise ValueError(f"ephemeral dependency versions drifted: {observed}")

    records: dict[str, Any] = {}
    for package in ("bm3d", "bm4d"):
        dist = distribution(package)
        files = tuple(dist.files or ())
        record_file = next(
            (item for item in files if str(item).endswith(".dist-info/RECORD")), None
        )
        license_file = next(
            (item for item in files if str(item).endswith(".dist-info/LICENSE")), None
        )
        metadata_file = next(
            (item for item in files if str(item).endswith(".dist-info/METADATA")), None
        )
        if record_file is None or license_file is None or metadata_file is None:
            raise FileNotFoundError(f"{package} distribution provenance files are incomplete")
        record_path = Path(dist.locate_file(record_file))
        license_path = Path(dist.locate_file(license_file))
        metadata_path = Path(dist.locate_file(metadata_file))
        license_text = license_path.read_text(encoding="utf-8")
        if "non-commercial scope only" not in license_text:
            raise ValueError(f"{package} installed license differs from expected research terms")
        records[package] = {
            "version": observed[package],
            "record_sha256": sha256_file(record_path),
            "license_sha256": sha256_file(license_path),
            "metadata_sha256": sha256_file(metadata_path),
            "license_noncommercial_research_only": True,
        }
    return {
        "versions": observed,
        "distributions": records,
        "submission_contains_package_code_or_binary": False,
        "submission_contains_png_predictions_only": True,
    }


def load_context(
    manifest_path: Path, stage: str
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("BM3D preregistration hash changed")
    config = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if [arm["name"] for arm in config.get("arms", [])] != list(ARM_NAMES):
        raise ValueError("runtime arm roster differs from preregistration")
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


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/aiijc_puzzle/bm3d_screen.py",
        PROJECT_ROOT / "src/aiijc_puzzle/dualnaf_bounded_residual.py",
        PROJECT_ROOT / "src/aiijc_puzzle/postassembly_harmonizer.py",
        PROJECT_ROOT / "src/aiijc_puzzle/pixel_tails.py",
        PROJECT_ROOT / "src/aiijc_puzzle/legacy_upgrade.py",
        PROJECT_ROOT / "src/aiijc_puzzle/protocol.py",
        PROJECT_ROOT / "src/aiijc_puzzle/compliant_atlas_decoder.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def apply_rgb_luma(ordered_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered_tiles, SeamGraphConfig())
    rgb_tiles = apply_rgb_offsets(ordered_tiles, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(
        rgb_tiles, LuminanceGainConfig()
    )
    harmonized = assemble_tiles(apply_luminance_gains(rgb_tiles, gains))
    return harmonized, {
        "rgb_seam_offsets": rgb_diagnostics,
        "bounded_luminance_gains": luma_diagnostics,
    }


def require_confirmation_authorized() -> str:
    if not PRIMARY_REPORT.is_file() or not PRIMARY_MANUAL_REVIEW.is_file():
        raise RuntimeError("primary report and manual review are required before confirmation")
    report = json.loads(PRIMARY_REPORT.read_text(encoding="utf-8"))
    manual = json.loads(PRIMARY_MANUAL_REVIEW.read_text(encoding="utf-8"))
    winner = report.get("selected_passing_winner")
    if winner not in CANDIDATE_ARMS:
        raise RuntimeError("no primary candidate passed every quantitative gate")
    if not manual.get("passed") or manual.get("reviewed_arm") != winner:
        raise RuntimeError("primary winner did not pass the separate manual artifact gate")
    return str(winner)


def freeze(args: argparse.Namespace) -> None:
    config, records = load_context(args.manifest, args.stage)
    runtime = package_runtime()
    winner = require_confirmation_authorized() if args.stage == "confirmation" else None
    arm_names = (CONTROL_ARM, winner) if winner is not None else ARM_NAMES
    root = STAGE_ROOTS[args.stage]
    if root.exists():
        raise FileExistsError(f"refusing to overwrite frozen experiment directory: {root}")
    root.mkdir(parents=True)
    started = perf_counter()
    boards: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = load_rgb_verified(args.inputs / filename, str(record["input_sha256"]))
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
        harmonized, harmonizer_diagnostics = apply_rgb_luma(ordered)
        predictions, render_diagnostics = render_arms(harmonized, arm_names)
        distinct = all_predictions_distinct(predictions)
        prediction_records: dict[str, Any] = {}
        board_directory = root / "predictions" / Path(filename).stem
        for name, prediction in predictions.items():
            output_path = board_directory / f"{name}.png"
            file_hash = atomic_write_png(output_path, prediction)
            prediction_records[name] = {
                "relative_path": str(output_path.relative_to(root)),
                "png_sha256": file_hash,
                "pixel_sha256": image_digest(prediction),
                "structure": structure_diagnostics(prediction),
            }
        boards.append(
            {
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "layout_sha256": layout_digest(layout),
                "raw_sha256": image_digest(raw),
                "harmonized_sha256": image_digest(harmonized),
                "permutation_audit": audit.as_dict(),
                "solver": solved.solver,
                "objective": float(solved.objective),
                "all_requested_predictions_distinct": distinct,
                "harmonizer_diagnostics": harmonizer_diagnostics,
                "render_diagnostics": render_diagnostics,
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
                    "all_distinct": distinct,
                }
            ),
            flush=True,
        )
    roster_digest = hashlib.sha256(
        "\n".join(
            f"{board['filename']} "
            + " ".join(
                board["predictions"][name]["pixel_sha256"]
                for name in board["predictions"]
            )
            for board in boards
        ).encode("utf-8")
    ).hexdigest()
    commitment = {
        "schema": "aiijc-bm3d-legal-screen-prediction-commitment-v1",
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
        "arm_names": list(arm_names),
        "confirmation_winner": winner,
        "all_boards_requested_predictions_distinct": all(
            board["all_requested_predictions_distinct"] for board in boards
        ),
        "frozen_prediction_roster_sha256": roster_digest,
        "dependency_runtime": runtime,
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
                "all_distinct": commitment["all_boards_requested_predictions_distinct"],
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
    root: Path,
    frozen: Sequence[Mapping[str, Any]],
    candidate: str,
) -> list[str]:
    directory = root / "manual-review-sheets"
    directory.mkdir(exist_ok=True)
    paths: list[str] = []
    for page_start in range(0, len(frozen), 6):
        canvas = Image.new("RGB", (516, 1638), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"left: {CONTROL_ARM} | right: {candidate}", fill="black")
        for local_index, item in enumerate(frozen[page_start : page_start + 6]):
            y = 34 + local_index * 267
            draw.text((8, y), str(item["board"]["filename"]), fill="black")
            baseline = Image.fromarray(item["predictions"][CONTROL_ARM]).resize(
                (240, 240), Image.Resampling.NEAREST
            )
            contender = Image.fromarray(item["predictions"][candidate]).resize(
                (240, 240), Image.Resampling.NEAREST
            )
            canvas.paste(baseline, (8, y + 18))
            canvas.paste(contender, (268, y + 18))
        path = directory / f"page-{page_start // 6 + 1}.png"
        atomic_audit_png(path, np.asarray(canvas, dtype=np.uint8))
        paths.append(str(path.relative_to(root)))
    return paths


def candidate_summary(
    rows: Sequence[Mapping[str, Any]],
    frozen: Sequence[Mapping[str, Any]],
    name: str,
) -> dict[str, Any]:
    baseline_scores = np.asarray([row["ssim"][CONTROL_ARM] for row in rows])
    scores = np.asarray([row["ssim"][name] for row in rows])
    difference = scores - baseline_scores
    gradient_ratios: list[float] = []
    laplacian_ratios: list[float] = []
    clipped_increases: list[float] = []
    for item in frozen:
        baseline = item["board"]["predictions"][CONTROL_ARM]["structure"]
        contender = item["board"]["predictions"][name]["structure"]
        gradient_ratios.append(
            float(
                contender["within_tile_luma_gradient_mean_abs"]
                / baseline["within_tile_luma_gradient_mean_abs"]
            )
        )
        laplacian_ratios.append(
            float(
                contender["within_tile_luma_laplacian_mean_abs"]
                / baseline["within_tile_luma_laplacian_mean_abs"]
            )
        )
        clipped_increases.append(
            float(contender["clipped_fraction"] - baseline["clipped_fraction"])
        )
    ci = paired_bootstrap_ci(difference)
    distinct = all(
        item["board"]["all_requested_predictions_distinct"] for item in frozen
    )
    checks = {
        "mean_ssim_at_least_0_27": float(scores.mean()) >= 0.27,
        "paired_ci95_lower_vs_A_strictly_positive": ci[0] > 0,
        "wins_vs_A_at_least_18_of_24": int(np.sum(difference > 0)) >= 18,
        "gradient_ratio_mean_at_least_0_80": float(np.mean(gradient_ratios)) >= 0.80,
        "gradient_ratio_board_min_at_least_0_70": float(np.min(gradient_ratios)) >= 0.70,
        "laplacian_ratio_mean_at_least_0_75": float(np.mean(laplacian_ratios)) >= 0.75,
        "laplacian_ratio_board_min_at_least_0_65": float(np.min(laplacian_ratios)) >= 0.65,
        "clipped_fraction_increase_board_max_at_most_0_01": (
            float(np.max(clipped_increases)) <= 0.01
        ),
        "all_required_predictions_distinct_on_every_board": distinct,
    }
    return {
        "mean_ssim": float(scores.mean()),
        "mean_gain_vs_A": float(difference.mean()),
        "gain_ci95": list(ci),
        "wins_ties_losses": [
            int(np.sum(difference > 0)),
            int(np.sum(difference == 0)),
            int(np.sum(difference < 0)),
        ],
        "structure": {
            "gradient_ratio_min_mean_max": [
                float(np.min(gradient_ratios)),
                float(np.mean(gradient_ratios)),
                float(np.max(gradient_ratios)),
            ],
            "laplacian_ratio_min_mean_max": [
                float(np.min(laplacian_ratios)),
                float(np.mean(laplacian_ratios)),
                float(np.max(laplacian_ratios)),
            ],
            "clipped_fraction_increase_min_mean_max": [
                float(np.min(clipped_increases)),
                float(np.mean(clipped_increases)),
                float(np.max(clipped_increases)),
            ],
        },
        "quantitative_checks": checks,
        "quantitative_passed": all(checks.values()),
    }


def score(args: argparse.Namespace) -> None:
    config, records = load_context(args.manifest, args.stage)
    package_runtime()
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
    arm_names = tuple(commitment["arm_names"])
    summaries = {
        name: candidate_summary(
            rows,
            frozen,
            name,
        )
        for name in arm_names
        if name != CONTROL_ARM
    }
    if args.stage == "primary":
        passing = [name for name in CANDIDATE_ARMS if summaries[name]["quantitative_passed"]]
        winner = max(
            passing,
            key=lambda name: (summaries[name]["mean_ssim"], -CANDIDATE_ARMS.index(name)),
            default=None,
        )
        diagnostic_best = max(CANDIDATE_ARMS, key=lambda name: summaries[name]["mean_ssim"])
    else:
        candidates = tuple(name for name in arm_names if name != CONTROL_ARM)
        if len(candidates) != 1:
            raise ValueError("confirmation must contain A and exactly one frozen winner")
        diagnostic_best = candidates[0]
        passing = [diagnostic_best] if summaries[diagnostic_best]["quantitative_passed"] else []
        winner = diagnostic_best if passing else None
    sheet_candidate = winner or diagnostic_best
    sheets = make_contact_sheets(root, frozen, sheet_candidate)
    baseline_scores = np.asarray([row["ssim"][CONTROL_ARM] for row in rows])
    report = {
        "schema": "aiijc-bm3d-legal-screen-score-v1",
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
        "summaries_vs_A": summaries,
        "quantitatively_passing_candidates": passing,
        "selected_passing_winner": winner,
        "manual_review_candidate": sheet_candidate,
        "manual_review_sheets": sheets,
        "manual_gate_status": "pending separate review artifact",
        "confirmation_authorized": False,
        "per_board": rows,
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "baseline": report["baseline"],
                "summaries_vs_A": summaries,
                "quantitatively_passing_candidates": passing,
                "selected_passing_winner": winner,
                "manual_review_candidate": sheet_candidate,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("dry-run only; pass --run to execute the requested phase")
    if args.phase == "freeze":
        freeze(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
