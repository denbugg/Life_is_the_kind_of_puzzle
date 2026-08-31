#!/usr/bin/env python3
"""Pilot frozen tile-wise DualNAF as an edge matcher, never an output renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.dualnaf_matcher import (
    BASELINE,
    MATCHER_ROSTER,
    PRIMARY_VARIANT,
    matcher_score_roster,
    solve_matcher_roster,
)
from aiijc_puzzle.legacy_upgrade import layout_digest
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
    GRID_SIZE,
    IMAGE_SIZE,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.restoration_r6 import TileAwareDualNAFNet, nlm_color
from aiijc_puzzle.tilewise_renderer import render_tiles_independently

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs/restoration-r6/compliant-r6-medium-train256-step2000-h10.pt"
)
STAGE_SPECIFICATIONS = {
    "pilot": {"offset": 192, "count": 12},
    "confirm": {"offset": 204, "count": 24},
}
EDGE_BUDGET = 96
RENDERER_CONDITIONING_H = 10
TAIL_NLM_H = 20
TAIL_NLM_PASSES = 1
BOOTSTRAP_REPLICATES = 20_000
MINIMUM_PRIMARY_SSIM_GAIN = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stage", choices=tuple(STAGE_SPECIFICATIONS), default="pilot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_rgb_verified(path: Path, expected_hash: str) -> np.ndarray:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(path) as image:
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


def image_digest(value: np.ndarray) -> str:
    image = np.asarray(value)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError(f"invalid prediction: {image.dtype} {image.shape}")
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def names_digest(records: tuple[Mapping[str, Any], ...]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode()
    ).hexdigest()


def choose_device(requested: str) -> torch.device:
    device_name = requested
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(device_name)


def load_model(
    path: Path,
    device: torch.device,
    protocol_digest: str,
) -> tuple[TileAwareDualNAFNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model_configuration = checkpoint.get("model_configuration")
    training = checkpoint.get("training_configuration")
    if not isinstance(model_configuration, dict) or not isinstance(training, dict):
        raise ValueError("checkpoint configuration is missing")
    if model_configuration.get("architecture") != "dual_naf":
        raise ValueError("only the frozen dual_naf checkpoint is supported")
    if training.get("protocol_digest") != protocol_digest:
        raise ValueError("checkpoint protocol differs from the manifest")
    if training.get("nlm_h") != RENDERER_CONDITIONING_H:
        raise ValueError("checkpoint NLM conditioning differs from the frozen experiment")
    model = TileAwareDualNAFNet(
        base=int(model_configuration["base"]),
        depth=int(model_configuration["depth"]),
        blocks=int(model_configuration["blocks"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def apply_rgb_luma(ordered_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered_tiles, SeamGraphConfig())
    rgb = apply_rgb_offsets(ordered_tiles, offsets)
    gains, luminance_diagnostics = seam_graph_luminance_gains(rgb, LuminanceGainConfig())
    return apply_luminance_gains(rgb, gains), {
        "rgb": rgb_diagnostics,
        "luminance": luminance_diagnostics,
    }


def infer_board(
    dirty: np.ndarray,
    model: TileAwareDualNAFNet,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Freeze all score-derived layouts and original-pixel outputs without a target."""

    original_tiles = split_tiles(dirty)
    rendered_tiles, renderer_diagnostics = render_tiles_independently(
        model,
        original_tiles,
        device,
        nlm_h=RENDERER_CONDITIONING_H,
        batch_size=batch_size,
    )
    scores = matcher_score_roster(original_tiles, rendered_tiles)
    layouts = solve_matcher_roster(scores, edge_budget=EDGE_BUDGET)
    variants: dict[str, dict[str, Any]] = {}
    for name in MATCHER_ROSTER:
        solved = layouts[name].result
        ordered_original = np.ascontiguousarray(original_tiles[solved.layout])
        raw = assemble_tiles(ordered_original)
        audit = audit_raw_permutation(
            dirty,
            raw,
            solved.layout,
            restoration_applied_after_audit=True,
        )
        if not audit.passed:
            raise RuntimeError(f"strict original-tile audit failed for {name}")
        harmonized_tiles, harmonizer_diagnostics = apply_rgb_luma(ordered_original)
        harmonized = assemble_tiles(harmonized_tiles)
        tail = nlm_color(harmonized, h=TAIL_NLM_H)
        variants[name] = {
            "layout": solved.layout,
            "raw": raw,
            "harmonized": harmonized,
            "tail": tail,
            "inference": {
                "layout_sha256": layout_digest(solved.layout),
                "raw_sha256": image_digest(raw),
                "harmonized_sha256": image_digest(harmonized),
                "tail_sha256": image_digest(tail),
                "permutation_audit": audit.as_dict(),
                "solver": solved.solver,
                "solver_seconds": solved.runtime_seconds,
                "objective": solved.objective,
                "output_pixels_source": "original dirty tiles only",
                "dualnaf_pixels_rendered_into_output": False,
                "harmonizer_diagnostics": harmonizer_diagnostics,
            },
        }
    return {
        "original_tiles": original_tiles,
        "renderer_diagnostics": renderer_diagnostics.as_dict(),
        "variants": variants,
    }


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


def paired_bootstrap(values: np.ndarray) -> tuple[float, float]:
    difference = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(EXPERIMENT_SUBSET_SEED)
    pieces: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        count = min(remaining, 4_096)
        indices = rng.integers(0, len(difference), size=(count, len(difference)))
        pieces.append(difference[indices].mean(axis=1))
        remaining -= count
    return tuple(float(value) for value in np.quantile(np.concatenate(pieces), (0.025, 0.975)))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "raw_ssim",
        "harmonized_ssim",
        "tail_ssim",
        "direct_placement",
        "translation_aligned_placement",
        "right_adjacency",
        "down_adjacency",
        "adjacency",
    )
    result: dict[str, Any] = {}
    for variant in MATCHER_ROSTER:
        values = {
            metric: np.asarray([row["variants"][variant][metric] for row in rows])
            for metric in metrics
        }
        baseline = {
            metric: np.asarray([row["variants"][BASELINE][metric] for row in rows])
            for metric in metrics
        }
        summary: dict[str, Any] = {
            metric: float(metric_values.mean()) for metric, metric_values in values.items()
        }
        summary["comparison_baseline"] = BASELINE
        for metric in metrics:
            difference = values[metric] - baseline[metric]
            summary[f"{metric}_gain"] = float(difference.mean())
            summary[f"{metric}_gain_ci95"] = list(paired_bootstrap(difference))
            summary[f"{metric}_wins_ties_losses"] = [
                int(np.sum(difference > 0)),
                int(np.sum(difference == 0)),
                int(np.sum(difference < 0)),
            ]
        result[variant] = summary
    return result


def primary_gate(
    summary: dict[str, Any], board_count: int, all_audits_passed: bool
) -> dict[str, Any]:
    primary = summary[PRIMARY_VARIANT]
    conditions = {
        "tail_ssim_gain_at_least_0p002": (primary["tail_ssim_gain"] >= MINIMUM_PRIMARY_SSIM_GAIN),
        "tail_ssim_paired_ci_lower_positive": primary["tail_ssim_gain_ci95"][0] > 0,
        "tail_ssim_wins_at_least_two_thirds": (
            primary["tail_ssim_wins_ties_losses"][0] >= int(np.ceil(2 * board_count / 3))
        ),
        "adjacency_gain_positive": primary["adjacency_gain"] > 0,
        "right_adjacency_gain_nonnegative": primary["right_adjacency_gain"] >= 0,
        "down_adjacency_gain_nonnegative": primary["down_adjacency_gain"] >= 0,
        "translation_aligned_placement_gain_nonnegative": (
            primary["translation_aligned_placement_gain"] >= 0
        ),
        "all_original_tile_permutation_audits_passed": all_audits_passed,
    }
    return {
        "primary_variant": PRIMARY_VARIANT,
        "comparison_baseline": BASELINE,
        "minimum_mean_tail_ssim_gain": MINIMUM_PRIMARY_SSIM_GAIN,
        "conditions": conditions,
        "passed": bool(all(conditions.values())),
    }


def frozen_manifest_payload(
    records: tuple[Mapping[str, Any], ...], frozen: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": "aiijc-dualnaf-matcher-frozen-inference-v1",
        "prediction_boundary": "all dirty-only predictions written before any target decode",
        "renderer_pixels_in_outputs": False,
        "boards": [
            {
                "filename": record["filename"],
                "input_sha256": record["input_sha256"],
                "variants": {name: item["variants"][name]["inference"] for name in MATCHER_ROSTER},
            }
            for record, item in zip(records, frozen, strict=True)
        ],
    }


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/aiijc_puzzle/dualnaf_matcher.py",
        PROJECT_ROOT / "src/aiijc_puzzle/tilewise_renderer.py",
        PROJECT_ROOT / "src/aiijc_puzzle/legacy_upgrade.py",
        PROJECT_ROOT / "src/aiijc_puzzle/postassembly_harmonizer.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("refusing evaluation without --run")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    specification = STAGE_SPECIFICATIONS[args.stage]
    offset, count = specification["offset"], specification["count"]
    output = args.output or (
        PROJECT_ROOT
        / "outputs/dualnaf-matcher"
        / f"{args.stage}-calibration-offset{offset}-count{count}.json"
    )
    frozen_output = output.with_name(f"{output.stem}.frozen.json")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest digest mismatch")
    prefix = select_manifest_records(
        manifest,
        "calibration",
        limit=offset + count,
        seed=EXPERIMENT_SUBSET_SEED,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
    )
    records = tuple(prefix[offset:])
    device = choose_device(args.device)
    model, checkpoint = load_model(
        args.checkpoint,
        device,
        str(manifest["protocol_digest"]),
    )

    started = perf_counter()
    frozen: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = load_rgb_verified(args.inputs / filename, str(record["input_sha256"]))
        inference = infer_board(
            dirty,
            model,
            device,
            batch_size=args.batch_size,
        )
        frozen.append(inference)
        print(
            json.dumps(
                {
                    "phase": "target_blind_freeze",
                    "done": index,
                    "total": count,
                    "filename": filename,
                    "elapsed_seconds": perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    atomic_json(frozen_output, frozen_manifest_payload(records, frozen))
    frozen_digest = sha256_file(frozen_output)

    rows: list[dict[str, Any]] = []
    for record, item in zip(records, frozen, strict=True):
        filename = str(record["filename"])
        target = load_rgb_verified(args.targets / filename, str(record["target_sha256"]))
        recovered = recover_layout(item["original_tiles"], split_tiles(target))
        variants: dict[str, Any] = {}
        for name in MATCHER_ROSTER:
            value = item["variants"][name]
            variants[name] = {
                "raw_ssim": contest_ssim(target, value["raw"]),
                "harmonized_ssim": contest_ssim(target, value["harmonized"]),
                "tail_ssim": contest_ssim(target, value["tail"]),
                **layout_metrics(value["layout"], recovered),
            }
        rows.append(
            {
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "target_sha256": record["target_sha256"],
                "all_predictions_frozen_before_any_target_decode": True,
                "renderer_diagnostics": item["renderer_diagnostics"],
                "variants": variants,
            }
        )
    summary = aggregate(rows)
    all_audits_passed = all(
        item["variants"][name]["inference"]["permutation_audit"]["passed"]
        for item in frozen
        for name in MATCHER_ROSTER
    )
    gate = primary_gate(summary, len(rows), all_audits_passed)
    report = {
        "schema": "aiijc-dualnaf-matcher-v1",
        "status": "completed",
        "stage": args.stage,
        "split": "calibration",
        "offset": offset,
        "count": count,
        "protocol_digest": manifest["protocol_digest"],
        "selection_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selection_seed": EXPERIMENT_SUBSET_SEED,
        "selection_digest": names_digest(records),
        "prediction_contract": {
            "inference_target_access": False,
            "all_predictions_frozen_before_any_target_decode": True,
            "frozen_manifest": str(frozen_output.resolve()),
            "frozen_manifest_sha256": frozen_digest,
            "holdout_access": False,
            "test_access": False,
        },
        "preregistration": {
            "baseline": BASELINE,
            "primary_variant": PRIMARY_VARIANT,
            "diagnostic_variants": [
                name for name in MATCHER_ROSTER if name not in {BASELINE, PRIMARY_VARIANT}
            ],
            "roster": list(MATCHER_ROSTER),
            "gate_defined_before_target_access": True,
            "confirm_only_if_primary_gate_passes": True,
            "confirm_panel": STAGE_SPECIFICATIONS["confirm"],
        },
        "configuration": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_model_configuration": checkpoint["model_configuration"],
            "checkpoint_training_configuration": checkpoint["training_configuration"],
            "device": str(device),
            "batch_size": args.batch_size,
            "renderer": "independent upright 20x20 tiles, used for scores only",
            "renderer_conditioning_nlm_h": RENDERER_CONDITIONING_H,
            "score": "E14 MGC+one-pixel SSD converted to row log-probabilities",
            "fusion": "fixed 0.5 dirty-bilateral + 0.5 DualNAF-raw normalized scores",
            "decoder": f"strict no-atlas buddies max_edges={EDGE_BUDGET}",
            "render": "original dirty tiles selected exactly once by the frozen layout",
            "harmonizer": "RGB seam offsets then bounded luminance gains",
            "tail": f"full-canvas colored NLM h={TAIL_NLM_H} exactly {TAIL_NLM_PASSES} pass",
        },
        "compliance": {
            "all_pre_restoration_permutation_audits_passed": all_audits_passed,
            "all_576_original_dirty_tiles_used_exactly_once": all_audits_passed,
            "dualnaf_pixels_rendered_into_output": False,
            "dualnaf_used_only_for_directional_scores": True,
            "tile_rotation_or_spatial_warp": False,
            "template_render_or_tile_substitution": False,
            "harmonizer_and_tail_target_blind": True,
        },
        "primary_gate": gate,
        "summary": summary,
        "per_board": rows,
        "runtime_seconds": perf_counter() - started,
        "source_sha256": source_hashes(),
    }
    atomic_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "frozen_output": str(frozen_output.resolve()),
                "primary_gate": gate,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
