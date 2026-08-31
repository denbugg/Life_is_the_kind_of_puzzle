#!/usr/bin/env python3
"""Evaluate weak population unaries inside strict tile-preserving decoders."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.compliant_atlas_decoder import (
    ATLAS_WEIGHTS,
    audit_raw_permutation,
    population_position_scores,
    solve_buddies_with_position,
)
from aiijc_puzzle.legacy_upgrade import (
    border_position_scores,
    directional_scores,
    layout_digest,
    solve_buddies,
    solve_relaxation,
)
from aiijc_puzzle.low_frequency_prior import FrozenLowFrequencyPrior
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_ATLAS = PROJECT_ROOT / "artifacts" / "low-frequency-prior" / "train5600-v1.npz"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "compliant-atlas-decoder" / "calibration48.json"
)
BASELINES = {
    "buddies96": "bilateral_buddies96",
    "relax": "bilateral_relax",
    "relax_border": "bilateral_relax_border",
}
NLM_H = 9
BOOTSTRAP_SAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def load_rgb_verified(path: Path, expected_hash: str) -> np.ndarray:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"content hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError(f"validation manifest digest mismatch: {path}")
    return manifest


def selection_digest(records: tuple[Any, ...]) -> str:
    names = [str(record["filename"]) for record in records]
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def prediction_hash(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def build_layouts(
    input_image: np.ndarray,
    atlas: FrozenLowFrequencyPrior,
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Infer strict layout variants; this API deliberately has no target."""

    tiles = split_tiles(input_image)
    score_started = perf_counter()
    right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
    score_seconds = perf_counter() - score_started
    unary_started = perf_counter()
    population = population_position_scores(tiles, atlas.generic_tile_template)
    border = border_position_scores(right, down)
    unary_seconds = perf_counter() - unary_started

    layouts: dict[str, Any] = {
        "bilateral_buddies96": solve_buddies(right, down, max_edges=96),
        "bilateral_relax": solve_relaxation(right, down),
        "bilateral_relax_border": solve_relaxation(right, down, position=border),
    }
    for weight in ATLAS_WEIGHTS:
        suffix = str(weight).replace(".", "p")
        layouts[f"bilateral_buddies96_atlas_w{suffix}"] = solve_buddies_with_position(
            right,
            down,
            population,
            position_weight=weight,
            max_edges=96,
        )
        layouts[f"bilateral_relax_atlas_w{suffix}"] = solve_relaxation(
            right,
            down,
            position=population,
            position_weight=weight,
        )
    layouts["bilateral_relax_border_plus_atlas"] = solve_relaxation(
        right,
        down,
        position=border + 0.5 * population,
        position_weight=0.11,
    )

    result: dict[str, dict[str, Any]] = {}
    for name, layout_result in layouts.items():
        raw = assemble_tiles(tiles[layout_result.layout])
        # This call precedes restoration. A failed audit aborts before NLM.
        audit = audit_raw_permutation(
            input_image,
            raw,
            layout_result.layout,
            restoration_applied_after_audit=True,
        )
        if not audit.passed:
            raise RuntimeError(f"permutation compliance failed for {name}: {audit.as_dict()}")
        restored = apply_nlm_color(raw, h=NLM_H)
        result[name] = {
            "layout": layout_result.layout,
            "raw": raw,
            "restored": restored.image,
            "layout_sha256": layout_digest(layout_result.layout),
            "raw_sha256": prediction_hash(raw),
            "restored_sha256": prediction_hash(restored.image),
            "permutation_audit": audit.as_dict(),
            "solver": layout_result.solver,
            "objective": layout_result.objective,
            "solve_seconds": layout_result.runtime_seconds,
            "restoration_seconds": restored.seconds,
            "restoration_order": "strict raw permutation audit, then full-canvas colored NLM",
        }
    return result, {"score_seconds": score_seconds, "unary_seconds": unary_seconds}


def posthoc_layout_metrics(
    layout: np.ndarray,
    recovered: Any,
) -> dict[str, float]:
    """Evaluate a previously frozen layout using approximate train labels."""

    truth = recovered.dirty_at_position
    position_of_dirty = recovered.position_of_dirty
    predicted_positions = np.empty_like(layout)
    predicted_positions[layout] = np.arange(len(layout))
    shifts: dict[tuple[int, int], int] = {}
    for tile, predicted in enumerate(predicted_positions):
        true = int(position_of_dirty[tile])
        predicted_row, predicted_column = divmod(int(predicted), 24)
        true_row, true_column = divmod(true, 24)
        shift = (true_row - predicted_row, true_column - predicted_column)
        shifts[shift] = shifts.get(shift, 0) + 1
    grid = layout.reshape(24, 24)
    left = position_of_dirty[grid[:, :-1]]
    right = position_of_dirty[grid[:, 1:]]
    top = position_of_dirty[grid[:-1]]
    bottom = position_of_dirty[grid[1:]]
    right_accuracy = np.mean((right - left == 1) & (right // 24 == left // 24))
    down_accuracy = np.mean(bottom - top == 24)
    return {
        "direct_placement": float(np.mean(layout == truth)),
        "translation_aligned_placement": float(max(shifts.values()) / len(layout)),
        "right_adjacency": float(right_accuracy),
        "down_adjacency": float(down_accuracy),
        "adjacency": float(0.5 * (right_accuracy + down_accuracy)),
        "label_mapping_mean_margin": float(recovered.margin_at_position.mean()),
    }


def paired_bootstrap_ci(differences: np.ndarray) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(EXPERIMENT_SUBSET_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), (0.025, 0.975))
    return float(low), float(high)


def comparison_baseline(name: str) -> str:
    if "buddies96_atlas" in name:
        return BASELINES["buddies96"]
    if name == "bilateral_relax_border_plus_atlas":
        return BASELINES["relax_border"]
    if "relax_atlas" in name:
        return BASELINES["relax"]
    return name


def aggregate(per_board: list[dict[str, Any]]) -> dict[str, Any]:
    names = tuple(per_board[0]["variants"])
    result: dict[str, Any] = {}
    metric_names = (
        "raw_ssim",
        "restored_ssim",
        "direct_placement",
        "translation_aligned_placement",
        "adjacency",
        "right_adjacency",
        "down_adjacency",
    )
    for name in names:
        summary = {
            metric: float(np.mean([row["variants"][name][metric] for row in per_board]))
            for metric in metric_names
        }
        baseline_name = comparison_baseline(name)
        summary["comparison_baseline"] = baseline_name
        if baseline_name == name:
            summary["restored_ssim_gain"] = 0.0
            summary["restored_ssim_gain_ci95"] = [0.0, 0.0]
            summary["adjacency_gain"] = 0.0
            summary["adjacency_gain_ci95"] = [0.0, 0.0]
        else:
            ssim_difference = np.asarray(
                [
                    row["variants"][name]["restored_ssim"]
                    - row["variants"][baseline_name]["restored_ssim"]
                    for row in per_board
                ]
            )
            adjacency_difference = np.asarray(
                [
                    row["variants"][name]["adjacency"]
                    - row["variants"][baseline_name]["adjacency"]
                    for row in per_board
                ]
            )
            summary["restored_ssim_gain"] = float(ssim_difference.mean())
            summary["restored_ssim_gain_ci95"] = list(paired_bootstrap_ci(ssim_difference))
            summary["adjacency_gain"] = float(adjacency_difference.mean())
            summary["adjacency_gain_ci95"] = list(paired_bootstrap_ci(adjacency_difference))
        result[name] = summary
    return result


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "compliant_atlas_decoder.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "low_frequency_prior.py",
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("Refusing evaluation without --run")
    if args.limit != 48:
        raise ValueError("the frozen comparison requires --limit 48")
    manifest = load_manifest(args.manifest)
    records = select_manifest_records(
        manifest,
        "calibration",
        limit=args.limit,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
        seed=EXPERIMENT_SUBSET_SEED,
    )
    atlas = FrozenLowFrequencyPrior.load(args.atlas)
    expected_metadata = {
        "protocol_digest": manifest["protocol_digest"],
        "train_records": 5_600,
        "target_contract": "manifest train split only",
    }
    for key, expected in expected_metadata.items():
        if atlas.metadata.get(key) != expected:
            raise ValueError(f"atlas metadata mismatch for {key}: {atlas.metadata.get(key)!r}")

    started = perf_counter()
    per_board: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        name = str(record["filename"])
        dirty = load_rgb_verified(args.inputs / name, str(record["input_sha256"]))

        # All inference, raw permutation audits, restoration and hashes are
        # complete before this calibration target is opened.
        frozen, shared_runtime = build_layouts(dirty, atlas)
        target = load_rgb_verified(args.targets / name, str(record["target_sha256"]))
        recovered = recover_layout(split_tiles(dirty), split_tiles(target))

        variants: dict[str, Any] = {}
        for variant_name, inference in frozen.items():
            layout_metrics = posthoc_layout_metrics(inference["layout"], recovered)
            variants[variant_name] = {
                "raw_ssim": contest_ssim(target, inference["raw"]),
                "restored_ssim": contest_ssim(target, inference["restored"]),
                **layout_metrics,
                "layout_sha256": inference["layout_sha256"],
                "raw_sha256": inference["raw_sha256"],
                "restored_sha256": inference["restored_sha256"],
                "permutation_audit": inference["permutation_audit"],
                "solver": inference["solver"],
                "objective": inference["objective"],
                "solve_seconds": inference["solve_seconds"],
                "restoration_seconds": inference["restoration_seconds"],
                "restoration_order": inference["restoration_order"],
            }
        per_board.append(
            {
                "filename": name,
                "predictions_frozen_before_target_decode": True,
                "shared_runtime": shared_runtime,
                "variants": variants,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "calibration",
                    "done": index,
                    "total": len(records),
                    "buddies96_nlm": variants["bilateral_buddies96"]["restored_ssim"],
                }
            ),
            flush=True,
        )

    summary = aggregate(per_board)
    new_names = [name for name in summary if comparison_baseline(name) != name]
    selected = max(new_names, key=lambda name: summary[name]["restored_ssim"])
    compliance_passed = all(
        row["variants"][name]["permutation_audit"]["passed"]
        for row in per_board
        for name in row["variants"]
    )
    result = {
        "schema_version": 1,
        "experiment": "compliant-population-atlas-unary-v1",
        "status": "calibration_complete",
        "split": "calibration",
        "holdout_opened": False,
        "protocol_digest": manifest["protocol_digest"],
        "selection_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selection_seed": EXPERIMENT_SUBSET_SEED,
        "selection_digest": selection_digest(records),
        "count": len(records),
        "atlas": {
            "path": str(args.atlas.resolve()),
            "sha256": sha256_file(args.atlas),
            "metadata": atlas.metadata,
            "use": "weak tile-to-position unary only; never rendered into output pixels",
        },
        "compliance": {
            "requires_24x24_layout": True,
            "requires_all_576_dirty_tiles_exactly_once": True,
            "raw_pixels_may_not_be_substituted_or_modified": True,
            "full_canvas_restoration": f"colored NLM h={NLM_H}, after raw audit",
            "all_permutation_audits_passed": compliance_passed,
            "target_used_at_inference": False,
            "prediction_and_audit_frozen_before_target_decode": True,
            "population_atlas_rendered_as_output": False,
        },
        "roster": {
            "score": "bilateral E14 MGC+SSD",
            "atlas_weights": list(ATLAS_WEIGHTS),
            "variants": list(per_board[0]["variants"]),
        },
        "selected_new_variant": selected,
        "summary": summary,
        "source_hashes": source_hashes(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "runtime_seconds": perf_counter() - started,
        "per_board": per_board,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_new_variant": selected,
                "selected": summary[selected],
                "baseline_buddies96": summary[BASELINES["buddies96"]],
                "compliance_passed": compliance_passed,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
