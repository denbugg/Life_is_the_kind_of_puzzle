#!/usr/bin/env python3
"""Pilot the frozen DualNAF as an independent per-tile faithful renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.legacy_upgrade import directional_scores, layout_digest, solve_buddies
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
from aiijc_puzzle.restoration_r6 import TileAwareDualNAFNet, nlm_color
from aiijc_puzzle.tilewise_renderer import render_tiles_independently

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs/restoration-r6/compliant-r6-medium-train256-step2000-h10.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/tilewise-dualnaf/no-atlas-calibration-offset96-count12-h10.json"
)
DEFAULT_OFFSET = 96
DEFAULT_COUNT = 12
DEFAULT_PASSES = (5, 10, 20)
DEFAULT_NLM_H = 10
EDGE_BUDGET = 96
BOOTSTRAP_REPLICATES = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--nlm-h", type=int, default=DEFAULT_NLM_H)
    parser.add_argument("--passes", type=int, nargs="+", default=list(DEFAULT_PASSES))
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def load_rgb_verified(path: Path, expected_hash: str) -> np.ndarray:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def image_digest(value: np.ndarray) -> str:
    image = np.asarray(value)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError(f"invalid prediction: {image.dtype} {image.shape}")
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def names_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode()
    ).hexdigest()


def prediction_names(passes_roster: tuple[int, ...]) -> tuple[str, ...]:
    names = ["raw", "raw_rgb_luma", "dualnaf_tiles", "dualnaf_tiles_rgb_luma"]
    for passes in passes_roster:
        names.extend(
            (
                f"raw_rgb_luma_then_nlm_{passes}x",
                f"dualnaf_tiles_rgb_luma_then_nlm_{passes}x",
            )
        )
    return tuple(names)


def apply_rgb_luma(ordered_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered_tiles, SeamGraphConfig())
    rgb = apply_rgb_offsets(ordered_tiles, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(rgb, LuminanceGainConfig())
    return apply_luminance_gains(rgb, gains), {
        "rgb": rgb_diagnostics,
        "luminance": luma_diagnostics,
    }


def load_model(
    path: Path, device: torch.device, nlm_h: int, protocol_digest: str
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
    if training.get("nlm_h") != nlm_h:
        raise ValueError("checkpoint conditioning NLM strength differs from evaluation")
    model = TileAwareDualNAFNet(
        base=int(model_configuration["base"]),
        depth=int(model_configuration["depth"]),
        blocks=int(model_configuration["blocks"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def infer_and_render(
    dirty: np.ndarray,
    model: TileAwareDualNAFNet,
    device: torch.device,
    *,
    nlm_h: int,
    passes_roster: tuple[int, ...],
    batch_size: int,
) -> dict[str, Any]:
    """Freeze the no-atlas strict layout and every target-blind pixel variant."""

    input_tiles = split_tiles(dirty)
    right, down = directional_scores(input_tiles, views=("bilateral",))["bilateral"]
    solved = solve_buddies(right, down, max_edges=EDGE_BUDGET)
    ordered_raw = np.ascontiguousarray(input_tiles[solved.layout])
    raw = assemble_tiles(ordered_raw)
    audit = audit_raw_permutation(dirty, raw, solved.layout, restoration_applied_after_audit=True)
    if not audit.passed:
        raise RuntimeError(f"strict pre-restoration permutation audit failed: {audit.as_dict()}")

    ordered_rendered, render_diagnostics = render_tiles_independently(
        model,
        ordered_raw,
        device,
        nlm_h=nlm_h,
        batch_size=batch_size,
    )
    raw_rgb_luma_tiles, raw_harmonizer = apply_rgb_luma(ordered_raw)
    rendered_rgb_luma_tiles, rendered_harmonizer = apply_rgb_luma(ordered_rendered)
    dualnaf_tiles = assemble_tiles(ordered_rendered)
    raw_rgb_luma = assemble_tiles(raw_rgb_luma_tiles)
    dualnaf_tiles_rgb_luma = assemble_tiles(rendered_rgb_luma_tiles)
    predictions: dict[str, np.ndarray] = {
        "raw": raw,
        "raw_rgb_luma": raw_rgb_luma,
        "dualnaf_tiles": dualnaf_tiles,
        "dualnaf_tiles_rgb_luma": dualnaf_tiles_rgb_luma,
    }
    iterative_raw = raw_rgb_luma
    iterative_rendered = dualnaf_tiles_rgb_luma
    for pass_count in range(1, max(passes_roster) + 1):
        iterative_raw = nlm_color(iterative_raw, nlm_h)
        iterative_rendered = nlm_color(iterative_rendered, nlm_h)
        if pass_count in passes_roster:
            predictions[f"raw_rgb_luma_then_nlm_{pass_count}x"] = iterative_raw
            predictions[f"dualnaf_tiles_rgb_luma_then_nlm_{pass_count}x"] = iterative_rendered
    roster = prediction_names(passes_roster)
    if set(predictions) != set(roster):
        raise RuntimeError(f"prediction roster mismatch: {set(predictions) ^ set(roster)}")
    predictions = {name: predictions[name] for name in roster}
    return {
        "layout": solved.layout,
        "layout_sha256": layout_digest(solved.layout),
        "audit": audit.as_dict(),
        "predictions": predictions,
        "prediction_sha256": {
            name: image_digest(prediction) for name, prediction in predictions.items()
        },
        "tilewise_render_diagnostics": render_diagnostics.as_dict(),
        "harmonizer_diagnostics": {
            "raw_tiles": raw_harmonizer,
            "dualnaf_tiles": rendered_harmonizer,
        },
        "solver": solved.solver,
        "objective": solved.objective,
    }


def paired_bootstrap(values: np.ndarray) -> tuple[float, float]:
    difference = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(EXPERIMENT_SUBSET_SEED)
    samples: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        count = min(remaining, 4_096)
        indices = rng.integers(0, len(difference), size=(count, len(difference)))
        samples.append(difference[indices].mean(axis=1))
        remaining -= count
    return tuple(float(value) for value in np.quantile(np.concatenate(samples), (0.025, 0.975)))


def variant_baseline(variant: str) -> str:
    if variant == "dualnaf_tiles":
        return "raw"
    if variant == "dualnaf_tiles_rgb_luma":
        return "raw_rgb_luma"
    if variant.startswith("dualnaf_tiles_rgb_luma_then_nlm_"):
        return variant.replace("dualnaf_tiles", "raw", 1)
    return variant


def aggregate(rows: list[dict[str, Any]], passes_roster: tuple[int, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in prediction_names(passes_roster):
        baseline = variant_baseline(variant)
        scores = np.asarray([row["ssim"][variant] for row in rows])
        baseline_scores = np.asarray([row["ssim"][baseline] for row in rows])
        difference = scores - baseline_scores
        result[variant] = {
            "mean_ssim": float(scores.mean()),
            "comparison_baseline": baseline,
            "mean_gain": float(difference.mean()),
            "gain_ci95": list(paired_bootstrap(difference)),
            "wins_ties_losses": [
                int(np.sum(difference > 0)),
                int(np.sum(difference == 0)),
                int(np.sum(difference < 0)),
            ],
        }
    return result


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def select_device(requested: str) -> torch.device:
    device_name = requested
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(device_name)


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("refusing evaluation without --run")
    passes_roster = tuple(sorted(set(args.passes)))
    if (
        args.offset < 0
        or args.count <= 0
        or args.nlm_h <= 0
        or args.batch_size <= 0
        or not passes_roster
        or passes_roster[0] <= 0
    ):
        raise ValueError("offset/count/nlm-h/batch-size/passes are invalid")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest digest mismatch")
    selected = select_manifest_records(
        manifest,
        "calibration",
        limit=args.offset + args.count,
        seed=EXPERIMENT_SUBSET_SEED,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
    )
    records = [dict(record) for record in selected[args.offset :]]
    device = select_device(args.device)
    model, checkpoint = load_model(
        args.checkpoint, device, args.nlm_h, str(manifest["protocol_digest"])
    )

    started = perf_counter()
    frozen_boards: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = load_rgb_verified(args.inputs / filename, str(record["input_sha256"]))
        inference = infer_and_render(
            dirty,
            model,
            device,
            nlm_h=args.nlm_h,
            passes_roster=passes_roster,
            batch_size=args.batch_size,
        )
        frozen_boards.append({"record": record, "inference": inference})
        print(
            json.dumps(
                {
                    "phase": "target_blind_render",
                    "done": index,
                    "total": len(records),
                    "filename": filename,
                }
            ),
            flush=True,
        )
    frozen_prediction_digest = hashlib.sha256(
        "\n".join(
            " ".join(board["inference"]["prediction_sha256"].values()) for board in frozen_boards
        ).encode()
    ).hexdigest()

    rows: list[dict[str, Any]] = []
    for board in frozen_boards:
        record = board["record"]
        inference = board["inference"]
        filename = str(record["filename"])
        target = load_rgb_verified(args.targets / filename, str(record["target_sha256"]))
        rows.append(
            {
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "target_sha256": record["target_sha256"],
                "all_predictions_frozen_before_any_target_decode": True,
                "layout_sha256": inference["layout_sha256"],
                "permutation_audit": inference["audit"],
                "prediction_sha256": inference["prediction_sha256"],
                "tilewise_render_diagnostics": inference["tilewise_render_diagnostics"],
                "harmonizer_diagnostics": inference["harmonizer_diagnostics"],
                "ssim": {
                    name: contest_ssim(target, prediction)
                    for name, prediction in inference["predictions"].items()
                },
            }
        )

    summary = aggregate(rows, passes_roster)
    tail_names = [f"dualnaf_tiles_rgb_luma_then_nlm_{passes}x" for passes in passes_roster]
    best_tail = max(tail_names, key=lambda name: summary[name]["mean_ssim"])
    best_tail_summary = summary[best_tail]
    mean_pixel_change = float(
        np.mean([row["tilewise_render_diagnostics"]["mean_abs_change"] for row in rows])
    )
    mean_model_residual = float(
        np.mean(
            [
                row["tilewise_render_diagnostics"]["residual_from_conditioning_mean_abs"]
                for row in rows
            ]
        )
    )
    pilot_gate = {
        "primary_variant": best_tail,
        "downstream_mean_gain_positive": best_tail_summary["mean_gain"] > 0,
        "downstream_ci_lower_positive": best_tail_summary["gain_ci95"][0] > 0,
        "wins_at_least_two_thirds": bool(
            best_tail_summary["wins_ties_losses"][0] >= np.ceil(2 * len(rows) / 3)
        ),
        "geometry_faithful_by_construction": True,
        "mean_abs_pixel_change": mean_pixel_change,
        "mean_abs_model_residual_from_tile_nlm": mean_model_residual,
    }
    pilot_gate["passed"] = bool(
        pilot_gate["downstream_mean_gain_positive"]
        and pilot_gate["downstream_ci_lower_positive"]
        and pilot_gate["wins_at_least_two_thirds"]
        and pilot_gate["geometry_faithful_by_construction"]
    )
    source_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/aiijc_puzzle/tilewise_renderer.py",
        PROJECT_ROOT / "src/aiijc_puzzle/restoration_r6.py",
        PROJECT_ROOT / "src/aiijc_puzzle/postassembly_harmonizer.py",
        PROJECT_ROOT / "src/aiijc_puzzle/legacy_upgrade.py",
    )
    report = {
        "schema": "aiijc-tilewise-dualnaf-harmonizer-pilot-v1",
        "status": "completed",
        "split": "calibration",
        "offset": args.offset,
        "count": len(records),
        "protocol_digest": manifest["protocol_digest"],
        "selection_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selection_seed": EXPERIMENT_SUBSET_SEED,
        "selection_digest": names_digest(records),
        "prediction_contract": {
            "inference_target_access": False,
            "all_predictions_frozen_before_any_target_decode": True,
            "frozen_prediction_digest": frozen_prediction_digest,
            "holdout_access": False,
            "test_access": False,
        },
        "configuration": {
            "layout": "bilateral true no-atlas solve_buddies96",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_model_configuration": checkpoint["model_configuration"],
            "checkpoint_training_configuration": checkpoint["training_configuration"],
            "renderer": (
                "576 upright 20x20 tiles independently; per-tile NLM conditioning; "
                "batched only along tile identity"
            ),
            "device": str(device),
            "batch_size": args.batch_size,
            "nlm_h": args.nlm_h,
            "passes": list(passes_roster),
            "variants": list(prediction_names(passes_roster)),
        },
        "compliance": {
            "all_576_tiles_audited_before_restoration": True,
            "all_pre_restoration_permutation_audits_passed": all(
                row["permutation_audit"]["passed"] for row in rows
            ),
            "one_renderer_output_per_input_tile_same_index": True,
            "cross_tile_pixels_or_context": False,
            "tile_rotation_or_spatial_warp": False,
            "template_render_or_tile_substitution": False,
            "harmonizer_target_blind": True,
        },
        "pilot_gate": pilot_gate,
        "best_dualnaf_tail": best_tail,
        "best_dualnaf_tail_summary": best_tail_summary,
        "summary": summary,
        "runtime_seconds": perf_counter() - started,
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in source_paths
        },
        "per_board": rows,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "best_dualnaf_tail": best_tail,
                "best_dualnaf_tail_summary": best_tail_summary,
                "pilot_gate": pilot_gate,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
