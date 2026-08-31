#!/usr/bin/env python3
"""Run the preregistered decoupled luma/chroma NLM screen in two phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.frozen_final_evaluator import _validate_method_configs, load_rgb_verified
from aiijc_puzzle.legacy_upgrade import (
    directional_scores,
    layout_digest,
    solve_buddies,
    validate_layout,
)
from aiijc_puzzle.nlm_luma_chroma import (
    NLMArm,
    apply_nlm_luma_chroma,
    image_digest,
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
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "nlm_luma_chroma_reused_calibration_preregistered_v1.json"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_INPUTS = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "nlm-luma-chroma"
EXPECTED_CONFIG_SHA256 = "38151503c12a39b3f5be7f2a19ad7d939796d33dc43416609810da47ef901108"
CONTROL_ARM = "nlm_h20_hc20_baseline"
SOURCE_FILES = (
    "scripts/run_nlm_luma_chroma.py",
    "src/aiijc_puzzle/nlm_luma_chroma.py",
    "src/aiijc_puzzle/legacy_upgrade.py",
    "src/aiijc_puzzle/postassembly_harmonizer.py",
    "src/aiijc_puzzle/compliant_atlas_decoder.py",
    "src/aiijc_puzzle/protocol.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primary", "confirmation"), default="primary")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
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
    finally:
        temporary.unlink(missing_ok=True)


def names_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode("utf-8")
    ).hexdigest()


def roster_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(
            f"{record['filename']}\0{record['input_sha256']}" for record in records
        ).encode("utf-8")
    ).hexdigest()


def canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_contract(
    config_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], tuple[NLMArm, ...]]:
    config_path = config_path.resolve()
    manifest_path = manifest_path.resolve()
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("preregistered NLM config hash drifted")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != config["protocol"]["manifest_sha256"]:
        raise RuntimeError("manifest file hash drifted")
    if manifest.get("protocol_digest") != config["protocol"]["protocol_digest"]:
        raise RuntimeError("manifest protocol digest drifted")
    if config["protocol"]["selector_namespace"] != EXPERIMENT_SUBSET_NAMESPACE:
        raise RuntimeError("selector namespace drifted")
    if config["protocol"]["selector_seed"] != EXPERIMENT_SUBSET_SEED:
        raise RuntimeError("selector seed drifted")
    arms = tuple(
        NLMArm(
            str(row["name"]),
            int(row["h"]),
            int(row["h_color"]),
            str(row["role"]),
        )
        for row in config["arms"]
    )
    if len({arm.name for arm in arms}) != len(arms):
        raise RuntimeError("duplicate arm in preregistration")
    controls = tuple(arm.name for arm in arms if arm.role == "control")
    if controls != (CONTROL_ARM,):
        raise RuntimeError("control arm roster drifted")
    return config, manifest, arms


def select_records(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    mode: str,
) -> tuple[Mapping[str, Any], ...]:
    section_name = "primary" if mode == "primary" else "confirmation_if_and_only_if_primary_passes"
    section = config["protocol"][section_name]
    offset = int(section["offset"])
    count = int(section["count"])
    ranked = select_manifest_records(
        manifest,
        str(config["protocol"]["split"]),
        limit=offset + count,
        seed=int(config["protocol"]["selector_seed"]),
        namespace=str(config["protocol"]["selector_namespace"]),
    )
    records = tuple(ranked[offset:])
    if len(records) != count:
        raise RuntimeError("selected panel count drifted")
    if names_digest(records) != section["filenames_newline_sha256"]:
        raise RuntimeError("selected filename digest drifted")
    if roster_digest(records) != section["filename_input_roster_sha256"]:
        raise RuntimeError("selected filename/input roster digest drifted")
    return records


def source_hashes(config_path: Path) -> dict[str, str]:
    files = (str(config_path.resolve().relative_to(PROJECT_ROOT)), *SOURCE_FILES)
    return {name: sha256_file(PROJECT_ROOT / name) for name in files}


def harmonized_canvas(ordered_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    rgb_config, luma_config, method_hashes = _validate_method_configs()
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered_tiles, rgb_config)
    rgb_tiles = apply_rgb_offsets(ordered_tiles, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(rgb_tiles, luma_config)
    harmonized = assemble_tiles(apply_luminance_gains(rgb_tiles, gains))
    return harmonized, {
        "rgb_seam_offsets": rgb_diagnostics,
        "bounded_luminance_gains": luma_diagnostics,
        "method_config_sha256": method_hashes,
    }


def freeze_predictions(
    records: Sequence[Mapping[str, Any]],
    *,
    inputs_dir: Path,
    arms: Sequence[NLMArm],
) -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        started = perf_counter()
        filename = str(record["filename"])
        dirty = load_rgb_verified(inputs_dir / filename, str(record["input_sha256"]))
        tiles = split_tiles(dirty)
        right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
        solved = solve_buddies(right, down, max_edges=96)
        layout = validate_layout(solved.layout)
        ordered = np.ascontiguousarray(tiles[layout])
        raw = assemble_tiles(ordered)
        audit = audit_raw_permutation(
            dirty,
            raw,
            layout,
            restoration_applied_after_audit=True,
        )
        if not audit.passed:
            raise RuntimeError(f"strict permutation audit failed for {filename}: {audit.as_dict()}")
        harmonized, tail_diagnostics = harmonized_canvas(ordered)
        predictions: dict[str, np.ndarray] = {}
        prediction_sha256: dict[str, str] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        for arm in arms:
            prediction = apply_nlm_luma_chroma(harmonized, h=arm.h, h_color=arm.h_color)
            prediction.flags.writeable = False
            predictions[arm.name] = prediction
            prediction_sha256[arm.name] = image_digest(prediction)
            diagnostics[arm.name] = structure_diagnostics(prediction)
        frozen.append(
            {
                "record": dict(record),
                "dirty": dirty,
                "raw": raw,
                "harmonized": harmonized,
                "layout": layout,
                "layout_sha256": layout_digest(layout),
                "audit": audit.as_dict(),
                "objective": float(solved.objective),
                "solver": solved.solver,
                "tail_diagnostics": tail_diagnostics,
                "predictions": predictions,
                "prediction_sha256": prediction_sha256,
                "structure_diagnostics": diagnostics,
                "runtime_seconds": perf_counter() - started,
            }
        )
        print(f"froze {index}/{len(records)} {filename}", flush=True)
    return frozen


def target_free_arm_safety(
    frozen: Sequence[Mapping[str, Any]],
    arms: Sequence[NLMArm],
) -> dict[str, Any]:
    baseline = [row["structure_diagnostics"][CONTROL_ARM] for row in frozen]
    summary: dict[str, Any] = {}
    for arm in arms:
        candidate = [row["structure_diagnostics"][arm.name] for row in frozen]
        summary[arm.name] = (
            {
                "mean_luminance_gradient_retention": 1.0,
                "minimum_luminance_gradient_retention": 1.0,
                "mean_chroma_gradient_retention": 1.0,
                "minimum_chroma_gradient_retention": 1.0,
                "mean_laplacian_retention": 1.0,
                "minimum_laplacian_retention": 1.0,
                "mean_grid_ratio_relative_to_baseline": 1.0,
                "maximum_grid_ratio_relative_to_baseline": 1.0,
            }
            if arm.name == CONTROL_ARM
            else safety_summary(candidate, baseline)
        )
        summary[arm.name]["distinct_from_baseline_on_every_board"] = all(
            row["prediction_sha256"][arm.name] != row["prediction_sha256"][CONTROL_ARM]
            for row in frozen
        ) if arm.name != CONTROL_ARM else True
    return summary


def build_commitment(
    frozen: Sequence[Mapping[str, Any]],
    *,
    config_path: Path,
    mode: str,
    records: Sequence[Mapping[str, Any]],
    arms: Sequence[NLMArm],
    safety: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "aiijc-nlm-luma-chroma-prediction-commitment-v1",
        "mode": mode,
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes(config_path),
        "selector_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selector_seed": EXPERIMENT_SUBSET_SEED,
        "split": "calibration",
        "filenames": [record["filename"] for record in records],
        "filenames_newline_sha256": names_digest(records),
        "arms": [
            {"name": arm.name, "h": arm.h, "h_color": arm.h_color, "role": arm.role}
            for arm in arms
        ],
        "contract": {
            "all_predictions_frozen_before_target_access": True,
            "target_paths_opened": False,
            "all_raw_permutation_audits_passed": all(row["audit"]["passed"] for row in frozen),
            "corresponding_input_only": True,
            "all_576_original_upright_tiles_used_once": True,
            "single_nlm_pass": True,
            "historically_exposed_reused_calibration": True,
            "freshness_claim": False,
        },
        "target_free_safety": safety,
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
                "audit": row["audit"],
                "objective": row["objective"],
                "solver": row["solver"],
                "runtime_seconds": row["runtime_seconds"],
            }
            for row in frozen
        ],
    }
    payload["commitment_sha256"] = canonical_digest(payload)
    return payload


def arm_gate(
    scores: Sequence[float],
    baseline_scores: Sequence[float],
    safety: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    difference = values - baseline
    interval = paired_t_interval(difference)
    observed = {
        "mean_rgb_ssim": float(values.mean()),
        "mean_gain_vs_baseline": float(difference.mean()),
        "paired_gain_ci95": interval,
        "wins_vs_baseline": int(np.sum(difference > 0)),
        "ties_vs_baseline": int(np.sum(difference == 0)),
        "losses_vs_baseline": int(np.sum(difference < 0)),
        **dict(safety),
    }
    checks = {
        "mean_rgb_ssim_min": observed["mean_rgb_ssim"]
        >= float(thresholds["winner_mean_rgb_ssim_min"]),
        "paired_gain_ci95_lower_positive": interval["lower"] > float(
            thresholds["winner_paired_gain_vs_baseline_ci95_lower_strictly_greater_than"]
        ),
        "wins_min": observed["wins_vs_baseline"]
        >= int(thresholds["winner_wins_vs_baseline_min"]),
        "mean_luminance_gradient_retention_min": observed[
            "mean_luminance_gradient_retention"
        ]
        >= float(thresholds["mean_luminance_gradient_retention_min"]),
        "minimum_luminance_gradient_retention_min": observed[
            "minimum_luminance_gradient_retention"
        ]
        >= float(thresholds["minimum_board_luminance_gradient_retention_min"]),
        "mean_chroma_gradient_retention_min": observed["mean_chroma_gradient_retention"]
        >= float(thresholds["mean_chroma_gradient_retention_min"]),
        "minimum_chroma_gradient_retention_min": observed[
            "minimum_chroma_gradient_retention"
        ]
        >= float(thresholds["minimum_board_chroma_gradient_retention_min"]),
        "mean_laplacian_retention_min": observed["mean_laplacian_retention"]
        >= float(thresholds["mean_laplacian_retention_min"]),
        "minimum_laplacian_retention_min": observed["minimum_laplacian_retention"]
        >= float(thresholds["minimum_board_laplacian_retention_min"]),
        "mean_grid_ratio_relative_to_baseline_max": observed[
            "mean_grid_ratio_relative_to_baseline"
        ]
        <= float(thresholds["mean_grid_ratio_relative_to_baseline_max"]),
        "maximum_grid_ratio_relative_to_baseline_max": observed[
            "maximum_grid_ratio_relative_to_baseline"
        ]
        <= float(thresholds["maximum_board_grid_ratio_relative_to_baseline_max"]),
        "all_predictions_distinct_across_boards": bool(
            observed["distinct_from_baseline_on_every_board"]
        ),
    }
    return {"observed": observed, "checks": checks, "all_passed": all(checks.values())}


def make_contact_sheet(
    frozen: Sequence[Mapping[str, Any]],
    targets: Mapping[str, np.ndarray],
    candidate_name: str,
    output: Path,
) -> None:
    thumb = 120
    label_width = 145
    header = 38
    columns = ("dirty input", "clean target", "h20/hc20", candidate_name)
    canvas = Image.new(
        "RGB",
        (label_width + thumb * len(columns), header + thumb * len(frozen)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(columns):
        draw.text((label_width + column * thumb + 4, 8), label, fill="black")
    for row_index, row in enumerate(frozen):
        filename = str(row["record"]["filename"])
        y = header + row_index * thumb
        draw.text((4, y + 5), filename, fill="black")
        images = (
            row["dirty"],
            targets[filename],
            row["predictions"][CONTROL_ARM],
            row["predictions"][candidate_name],
        )
        for column, array in enumerate(images):
            thumbnail = Image.fromarray(array).resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(thumbnail, (label_width + column * thumb, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def score_after_commitment(
    frozen: Sequence[Mapping[str, Any]],
    *,
    targets_dir: Path,
    arms: Sequence[NLMArm],
    safety: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    commitment_sha256: str,
    sheet_path: Path,
) -> dict[str, Any]:
    scores: dict[str, list[float]] = {arm.name: [] for arm in arms}
    targets: dict[str, np.ndarray] = {}
    boards: list[dict[str, Any]] = []
    for index, row in enumerate(frozen, start=1):
        record = row["record"]
        filename = str(record["filename"])
        target = load_rgb_verified(targets_dir / filename, str(record["target_sha256"]))
        targets[filename] = target
        board_scores = {
            arm.name: contest_ssim(target, row["predictions"][arm.name]) for arm in arms
        }
        for name, value in board_scores.items():
            scores[name].append(value)
        boards.append({"filename": filename, "ssim": board_scores})
        print(f"scored {index}/{len(frozen)} {filename}", flush=True)

    baseline = scores[CONTROL_ARM]
    gates = {
        arm.name: arm_gate(scores[arm.name], baseline, safety[arm.name], thresholds)
        for arm in arms
        if arm.role == "candidate"
    }
    passed = [arm for arm in arms if arm.role == "candidate" and gates[arm.name]["all_passed"]]
    ranking = sorted(
        (arm for arm in arms if arm.role == "candidate"),
        key=lambda arm: (-float(np.mean(scores[arm.name])), arm.h, arm.h_color, arm.name),
    )
    winner = next((arm for arm in ranking if arm in passed), None)
    display = winner or ranking[0]
    make_contact_sheet(frozen, targets, display.name, sheet_path)
    return {
        "commitment_sha256": commitment_sha256,
        "target_access_started_only_after_commitment": True,
        "board_count": len(frozen),
        "means": {name: float(np.mean(values)) for name, values in scores.items()},
        "gates": gates,
        "passing_arms": [arm.name for arm in passed],
        "winner": winner.name if winner is not None else None,
        "diagnostic_best_mean_arm": ranking[0].name,
        "all_passed": winner is not None,
        "boards": boards,
        "contact_sheet": str(sheet_path.resolve()),
    }


def primary_report_path(output_root: Path) -> Path:
    return output_root.resolve() / "primary-calibration-offset468-count36" / "report.json"


def confirmation_winner(
    output_root: Path,
    *,
    config_hash: str,
    available_arms: Sequence[NLMArm],
) -> NLMArm:
    path = primary_report_path(output_root)
    if not path.is_file():
        raise RuntimeError("confirmation requires the completed primary report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("config_sha256") != config_hash:
        raise RuntimeError("primary report belongs to a different frozen config")
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("all_passed") is not True:
        raise RuntimeError("primary gate failed; confirmation access is forbidden")
    winner_name = evaluation.get("winner")
    candidate = next((arm for arm in available_arms if arm.name == winner_name), None)
    if candidate is None or candidate.role != "candidate":
        raise RuntimeError("primary winner is absent from preregistered candidates")
    commitment_path = Path(str(report["prediction_contract"]["commitment_path"]))
    if sha256_file(commitment_path) != report["prediction_contract"]["commitment_file_sha256"]:
        raise RuntimeError("primary prediction commitment changed")
    return candidate


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config, manifest, all_arms = load_contract(config_path, args.manifest)
    records = select_records(config, manifest, args.mode)
    if args.mode == "primary":
        arms = all_arms
        section = "primary-calibration-offset468-count36"
    else:
        winner = confirmation_winner(
            args.output_root,
            config_hash=sha256_file(config_path),
            available_arms=all_arms,
        )
        arms = (next(arm for arm in all_arms if arm.name == CONTROL_ARM), winner)
        section = "confirmation-calibration-offset504-count36"

    output_dir = args.output_root.resolve() / section
    report_path = output_dir / "report.json"
    commitment_path = output_dir / "prediction-commitment.json"
    if report_path.exists() or commitment_path.exists():
        raise RuntimeError(f"refusing to overwrite existing experiment artifact in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    frozen = freeze_predictions(records, inputs_dir=args.inputs.resolve(), arms=arms)
    safety = target_free_arm_safety(frozen, arms)
    commitment = build_commitment(
        frozen,
        config_path=config_path,
        mode=args.mode,
        records=records,
        arms=arms,
        safety=safety,
    )
    atomic_json(commitment_path, commitment)
    commitment_file_sha256 = sha256_file(commitment_path)
    readback = json.loads(commitment_path.read_text(encoding="utf-8"))
    if readback.get("commitment_sha256") != commitment["commitment_sha256"]:
        raise RuntimeError("prediction commitment readback failed")
    freeze_seconds = perf_counter() - started

    evaluation = score_after_commitment(
        frozen,
        targets_dir=args.targets.resolve(),
        arms=arms,
        safety=safety,
        thresholds=config["primary_promotion_gate"],
        commitment_sha256=commitment["commitment_sha256"],
        sheet_path=output_dir / "contact-sheet.png",
    )
    if sha256_file(commitment_path) != commitment_file_sha256:
        raise RuntimeError("prediction commitment changed after target access")
    report = {
        "schema": "aiijc-nlm-luma-chroma-evaluation-v1",
        "status": "completed_gate_passed" if evaluation["all_passed"] else "completed_gate_failed",
        "mode": args.mode,
        "verdict": (
            "primary-passed-run-frozen-confirmation"
            if args.mode == "primary" and evaluation["all_passed"]
            else "confirmation-passed-candidate-for-manual-audit"
            if args.mode == "confirmation" and evaluation["all_passed"]
            else "gate-failed-do-not-promote"
        ),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "selection": {
            "split": "calibration",
            "selector_namespace": EXPERIMENT_SUBSET_NAMESPACE,
            "selector_seed": EXPERIMENT_SUBSET_SEED,
            "filenames": [record["filename"] for record in records],
            "filenames_newline_sha256": names_digest(records),
            "reused_historically_target_exposed": True,
            "freshness_claim": False,
        },
        "prediction_contract": {
            "all_predictions_frozen_before_target_access": True,
            "commitment_path": str(commitment_path),
            "commitment_file_sha256": commitment_file_sha256,
            "commitment_payload_sha256": commitment["commitment_sha256"],
            "holdout_access": False,
            "test_access": False,
        },
        "runtime_seconds": {
            "prediction_freeze_and_commitment": freeze_seconds,
            "total": perf_counter() - started,
        },
        "evaluation": evaluation,
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": report["status"],
                "means": evaluation["means"],
                "winner": evaluation["winner"],
                "passing_arms": evaluation["passing_arms"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
