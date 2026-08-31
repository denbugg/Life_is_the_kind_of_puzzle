#!/usr/bin/env python3
"""Evaluate the exact historical harmonizers after the strict atlas decoder."""

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
from PIL import Image

from aiijc_puzzle.compliant_atlas_decoder import (
    PRODUCTION_ATLAS_WEIGHT,
    PRODUCTION_EDGE_BUDGET,
    audit_raw_permutation,
    population_position_scores,
    solve_buddies_with_position,
)
from aiijc_puzzle.legacy_upgrade import directional_scores, layout_digest, solve_buddies
from aiijc_puzzle.low_frequency_prior import FrozenLowFrequencyPrior
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
from aiijc_puzzle.restoration_r6 import nlm_color

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_ATLAS = PROJECT_ROOT / "artifacts" / "low-frequency-prior" / "train5600-v1.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "postassembly-harmonizer" / "calibration24.json"
RGB_CONFIG_PATH = PROJECT_ROOT / "configs" / "postassembly_rgb_offset_v1.json"
LUMA_CONFIG_PATH = PROJECT_ROOT / "configs" / "postassembly_luminance_gain_v1.json"

DEFAULT_PASSES = (1, 5, 10, 20)
DEFAULT_NLM_H = 10
DEFAULT_OFFSET = 48
DEFAULT_COUNT = 24
BOOTSTRAP_REPLICATES = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--layout",
        choices=("atlas", "no-atlas"),
        default="atlas",
        help="atlas uses w=0.03; no-atlas calls the true legacy solve_buddies path",
    )
    parser.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--nlm-h", type=int, default=DEFAULT_NLM_H)
    parser.add_argument("--passes", type=int, nargs="+", default=list(DEFAULT_PASSES))
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


def prediction_digest(image: np.ndarray) -> str:
    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError(f"invalid prediction {value.dtype} {value.shape}")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def names_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode()
    ).hexdigest()


def variant_names(passes_roster: tuple[int, ...]) -> tuple[str, ...]:
    names: list[str] = ["raw", "rgb", "rgb_luma"]
    for passes in passes_roster:
        names.extend(
            (
                f"nlm_{passes}x",
                f"rgb_then_nlm_{passes}x",
                f"rgb_luma_then_nlm_{passes}x",
                f"nlm_{passes}x_then_rgb",
                f"nlm_{passes}x_then_rgb_luma",
            )
        )
    return tuple(names)


def harmonize_tiles(
    tiles: np.ndarray,
    *,
    include_luminance: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb_offsets, rgb_diagnostics = seam_graph_rgb_offsets(tiles, SeamGraphConfig())
    rgb = apply_rgb_offsets(tiles, rgb_offsets)
    diagnostics: dict[str, Any] = {"rgb": rgb_diagnostics}
    if not include_luminance:
        return rgb, diagnostics
    gains, luminance_diagnostics = seam_graph_luminance_gains(rgb, LuminanceGainConfig())
    diagnostics["luminance"] = luminance_diagnostics
    return apply_luminance_gains(rgb, gains), diagnostics


def infer_and_render(
    dirty: np.ndarray,
    atlas: FrozenLowFrequencyPrior | None,
    *,
    layout_mode: str,
    passes_roster: tuple[int, ...],
    nlm_h: int,
) -> dict[str, Any]:
    """Build the entire target-blind render roster; there is no target argument."""

    input_tiles = split_tiles(dirty)
    right, down = directional_scores(input_tiles, views=("bilateral",))["bilateral"]
    if layout_mode == "atlas":
        if atlas is None:
            raise ValueError("atlas layout requires a loaded train-only atlas")
        position = population_position_scores(input_tiles, atlas.generic_tile_template)
        solved = solve_buddies_with_position(
            right,
            down,
            position,
            position_weight=PRODUCTION_ATLAS_WEIGHT,
            max_edges=PRODUCTION_EDGE_BUDGET,
        )
    elif layout_mode == "no-atlas":
        if atlas is not None:
            raise ValueError("true no-atlas layout must not receive an atlas")
        solved = solve_buddies(right, down, max_edges=PRODUCTION_EDGE_BUDGET)
    else:
        raise ValueError(f"unknown layout mode: {layout_mode}")
    raw = assemble_tiles(input_tiles[solved.layout])
    audit = audit_raw_permutation(
        dirty,
        raw,
        solved.layout,
        restoration_applied_after_audit=True,
    )
    if not audit.passed:
        raise RuntimeError(f"strict pre-restoration permutation audit failed: {audit.as_dict()}")
    ordered = split_tiles(raw)
    rgb_tiles, rgb_diagnostics = harmonize_tiles(ordered, include_luminance=False)
    rgb_luma_tiles, rgb_luma_diagnostics = harmonize_tiles(ordered, include_luminance=True)
    rgb = assemble_tiles(rgb_tiles)
    rgb_luma = assemble_tiles(rgb_luma_tiles)
    predictions: dict[str, np.ndarray] = {
        "raw": raw,
        "rgb": rgb,
        "rgb_luma": rgb_luma,
    }
    diagnostics: dict[str, Any] = {
        "rgb": rgb_diagnostics,
        "rgb_luma": rgb_luma_diagnostics,
        "nlm_then_harmonize": {},
    }

    iterative_raw = raw
    iterative_rgb = rgb
    iterative_rgb_luma = rgb_luma
    selected_nlm: dict[int, np.ndarray] = {}
    for pass_count in range(1, max(passes_roster) + 1):
        iterative_raw = nlm_color(iterative_raw, nlm_h)
        iterative_rgb = nlm_color(iterative_rgb, nlm_h)
        iterative_rgb_luma = nlm_color(iterative_rgb_luma, nlm_h)
        if pass_count not in passes_roster:
            continue
        selected_nlm[pass_count] = iterative_raw
        predictions[f"nlm_{pass_count}x"] = iterative_raw
        predictions[f"rgb_then_nlm_{pass_count}x"] = iterative_rgb
        predictions[f"rgb_luma_then_nlm_{pass_count}x"] = iterative_rgb_luma

    for pass_count in passes_roster:
        nlm_tiles = split_tiles(selected_nlm[pass_count])
        nlm_rgb_tiles, nlm_rgb_diagnostics = harmonize_tiles(nlm_tiles, include_luminance=False)
        nlm_rgb_luma_tiles, nlm_rgb_luma_diagnostics = harmonize_tiles(
            nlm_tiles, include_luminance=True
        )
        predictions[f"nlm_{pass_count}x_then_rgb"] = assemble_tiles(nlm_rgb_tiles)
        predictions[f"nlm_{pass_count}x_then_rgb_luma"] = assemble_tiles(nlm_rgb_luma_tiles)
        diagnostics["nlm_then_harmonize"][str(pass_count)] = {
            "rgb": nlm_rgb_diagnostics,
            "rgb_luma": nlm_rgb_luma_diagnostics,
        }

    # Construction order above is chosen for efficient iterative NLM.  Reorder
    # to a frozen semantic roster before hashing or evaluation.
    roster = variant_names(passes_roster)
    if set(predictions) != set(roster):
        raise RuntimeError(f"prediction roster mismatch: {set(predictions) ^ set(roster)}")
    predictions = {name: predictions[name] for name in roster}
    return {
        "layout": solved.layout,
        "layout_sha256": layout_digest(solved.layout),
        "audit": audit.as_dict(),
        "predictions": predictions,
        "prediction_sha256": {
            name: prediction_digest(prediction) for name, prediction in predictions.items()
        },
        "diagnostics": diagnostics,
        "solver": solved.solver,
        "objective": solved.objective,
    }


def paired_bootstrap(values: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(EXPERIMENT_SUBSET_SEED)
    samples: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        count = min(remaining, 4_096)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        samples.append(differences[indices].mean(axis=1))
        remaining -= count
    return tuple(float(value) for value in np.quantile(np.concatenate(samples), (0.025, 0.975)))


def comparison_baseline(variant: str, passes_roster: tuple[int, ...]) -> str:
    if variant in {"raw", "rgb", "rgb_luma"}:
        return "raw"
    pass_count = next(passes for passes in passes_roster if f"_{passes}x" in variant)
    return f"nlm_{pass_count}x"


def aggregate(rows: list[dict[str, Any]], passes_roster: tuple[int, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in variant_names(passes_roster):
        baseline = comparison_baseline(variant, passes_roster)
        scores = np.asarray([row["ssim"][variant] for row in rows])
        baseline_scores = np.asarray([row["ssim"][baseline] for row in rows])
        difference = scores - baseline_scores
        result[variant] = {
            "mean_ssim": float(scores.mean()),
            "std_ssim": float(scores.std()),
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


def order_comparisons(rows: list[dict[str, Any]], passes_roster: tuple[int, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pass_count in passes_roster:
        for method in ("rgb", "rgb_luma"):
            before = f"{method}_then_nlm_{pass_count}x"
            after = f"nlm_{pass_count}x_then_{method}"
            difference = np.asarray([row["ssim"][before] - row["ssim"][after] for row in rows])
            result[f"{method}__{pass_count}x__harmonize_then_nlm_minus_reverse"] = {
                "harmonize_then_nlm": before,
                "nlm_then_harmonize": after,
                "mean_difference": float(difference.mean()),
                "ci95": list(paired_bootstrap(difference)),
                "wins_ties_losses": [
                    int(np.sum(difference > 0)),
                    int(np.sum(difference == 0)),
                    int(np.sum(difference < 0)),
                ],
                "preferred_order": (
                    "harmonize_then_nlm" if difference.mean() > 0 else "nlm_then_harmonize"
                ),
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


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "postassembly_harmonizer.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "compliant_atlas_decoder.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "restoration_r6.py",
        RGB_CONFIG_PATH,
        LUMA_CONFIG_PATH,
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("refusing evaluation without --run")
    if args.offset < 0 or args.count <= 0:
        raise ValueError("offset must be non-negative and count must be positive")
    if args.nlm_h <= 0:
        raise ValueError("nlm-h must be positive")
    passes_roster = tuple(sorted(set(args.passes)))
    if not passes_roster or passes_roster[0] <= 0:
        raise ValueError("passes must contain positive integers")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest digest mismatch")
    panel = select_manifest_records(
        manifest,
        "calibration",
        limit=args.offset + args.count,
        seed=EXPERIMENT_SUBSET_SEED,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
    )
    records = [dict(record) for record in panel[args.offset :]]
    atlas: FrozenLowFrequencyPrior | None = None
    if args.layout == "atlas":
        atlas = FrozenLowFrequencyPrior.load(args.atlas)
        expected_atlas = {
            "protocol_digest": manifest["protocol_digest"],
            "train_records": 5_600,
            "target_contract": "manifest train split only",
        }
        for key, value in expected_atlas.items():
            if atlas.metadata.get(key) != value:
                raise ValueError(f"atlas provenance mismatch for {key}")

    started = perf_counter()
    frozen_boards: list[dict[str, Any]] = []
    # Phase 1: freeze every prediction on all 24 boards before any target path
    # is opened.  This is stronger than merely freezing per board.
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = load_rgb_verified(args.inputs / filename, str(record["input_sha256"]))
        inference = infer_and_render(
            dirty,
            atlas,
            layout_mode=args.layout,
            passes_roster=passes_roster,
            nlm_h=args.nlm_h,
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

    scored_rows: list[dict[str, Any]] = []
    for index, board in enumerate(frozen_boards, start=1):
        record = board["record"]
        inference = board["inference"]
        filename = str(record["filename"])
        target = load_rgb_verified(args.targets / filename, str(record["target_sha256"]))
        scores = {
            name: contest_ssim(target, prediction)
            for name, prediction in inference["predictions"].items()
        }
        scored_rows.append(
            {
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "target_sha256": record["target_sha256"],
                "all_predictions_frozen_before_any_target_decode": True,
                "layout_sha256": inference["layout_sha256"],
                "permutation_audit": inference["audit"],
                "prediction_sha256": inference["prediction_sha256"],
                "harmonizer_diagnostics": inference["diagnostics"],
                "ssim": scores,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "posthoc_target_score",
                    "done": index,
                    "total": len(records),
                    "filename": filename,
                    "best_ssim": max(scores.values()),
                }
            ),
            flush=True,
        )

    summary = aggregate(scored_rows, passes_roster)
    champion = max(summary, key=lambda name: summary[name]["mean_ssim"])
    report = {
        "schema": "aiijc-postassembly-harmonizer-atlas-calibration-v1",
        "status": "completed",
        "split": "calibration",
        "offset": args.offset,
        "count": args.count,
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
            "layout": (
                "bilateral atlas_w0.03 + buddies96"
                if args.layout == "atlas"
                else "bilateral true no-atlas solve_buddies96"
            ),
            "atlas": str(args.atlas.resolve()) if atlas is not None else None,
            "atlas_sha256": sha256_file(args.atlas) if atlas is not None else None,
            "rgb_config": json.loads(RGB_CONFIG_PATH.read_text(encoding="utf-8")),
            "luminance_config": json.loads(LUMA_CONFIG_PATH.read_text(encoding="utf-8")),
            "nlm": {
                "implementation": "proper RGB<->BGR OpenCV colored NLM",
                "h": args.nlm_h,
                "template_window": 7,
                "search_window": 21,
                "passes": list(passes_roster),
            },
            "variants": list(variant_names(passes_roster)),
        },
        "compliance": {
            "all_576_tiles_audited_before_restoration": True,
            "all_pre_restoration_permutation_audits_passed": all(
                row["permutation_audit"]["passed"] for row in scored_rows
            ),
            "raw_assembly_pixel_preserving": True,
            "harmonizers_target_blind": True,
            "no_template_render_or_tile_substitution": True,
            "no_spatial_warp": True,
        },
        "champion": champion,
        "champion_summary": summary[champion],
        "summary": summary,
        "order_comparisons": order_comparisons(scored_rows, passes_roster),
        "runtime_seconds": perf_counter() - started,
        "source_sha256": source_hashes(),
        "per_board": scored_rows,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "champion": champion,
                "champion_summary": summary[champion],
                "order_comparisons": report["order_comparisons"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
