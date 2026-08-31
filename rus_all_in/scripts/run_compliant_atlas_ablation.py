#!/usr/bin/env python3
"""Fresh-panel edge-budget and proper RGB NLM ablation for the compliant atlas decoder."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.compliant_atlas_decoder import (
    audit_raw_permutation,
    population_position_scores,
    solve_buddies_with_position,
)
from aiijc_puzzle.legacy_upgrade import directional_scores, layout_digest
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
    PROJECT_ROOT / "outputs" / "compliant-atlas-decoder" / "fresh-calibration24-ablation.json"
)
FRESH_NAMESPACE = "compliant-atlas-tail-edge-ablation-v1"
EXCLUDED_PRIOR_COUNT = 48
EVAL_COUNT = 24
ATLAS_WEIGHT = 0.06
EDGE_BUDGETS = (96, 256)
NLM_STRENGTHS = (9, 10)
BOOTSTRAP_SAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def records_digest(records: tuple[Any, ...]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode()
    ).hexdigest()


def fresh_records(manifest: dict[str, Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    excluded = select_manifest_records(
        manifest,
        "calibration",
        limit=EXCLUDED_PRIOR_COUNT,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
        seed=EXPERIMENT_SUBSET_SEED,
    )
    excluded_names = {str(record["filename"]) for record in excluded}
    candidates = [
        record
        for record in manifest["splits"]["calibration"]
        if str(record["filename"]) not in excluded_names
    ]
    prefix = f"{FRESH_NAMESPACE}\0{EXPERIMENT_SUBSET_SEED}\0".encode()
    ranked = sorted(
        candidates,
        key=lambda record: (
            hashlib.sha256(prefix + str(record["filename"]).encode()).digest(),
            str(record["filename"]),
        ),
    )
    selected = tuple(ranked[:EVAL_COUNT])
    if excluded_names & {str(record["filename"]) for record in selected}:
        raise RuntimeError("fresh ablation overlaps the prior calibration-48 panel")
    return tuple(excluded), selected


def prediction_hash(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def build_predictions(
    dirty: np.ndarray,
    atlas: FrozenLowFrequencyPrior,
) -> dict[str, dict[str, Any]]:
    """Build the four fixed arms without a target argument."""

    tiles = split_tiles(dirty)
    right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
    position = population_position_scores(tiles, atlas.generic_tile_template)
    result: dict[str, dict[str, Any]] = {}
    for budget in EDGE_BUDGETS:
        solved = solve_buddies_with_position(
            right,
            down,
            position,
            position_weight=ATLAS_WEIGHT,
            max_edges=budget,
        )
        raw = assemble_tiles(tiles[solved.layout])
        audit = audit_raw_permutation(
            dirty,
            raw,
            solved.layout,
            restoration_applied_after_audit=True,
        )
        if not audit.passed:
            raise RuntimeError(f"permutation audit failed for buddies{budget}")
        for h in NLM_STRENGTHS:
            restored = apply_nlm_color(raw, h=h)
            name = f"atlas_w0p06_buddies{budget}_proper_rgb_nlm_h{h}"
            result[name] = {
                "layout": solved.layout,
                "raw": raw,
                "restored": restored.image,
                "layout_sha256": layout_digest(solved.layout),
                "raw_sha256": prediction_hash(raw),
                "restored_sha256": prediction_hash(restored.image),
                "permutation_audit": audit.as_dict(),
                "restoration_seconds": restored.seconds,
                "restoration_order": "strict raw permutation audit, then proper RGB NLM",
            }
    if len(result) != len(EDGE_BUDGETS) * len(NLM_STRENGTHS):
        raise RuntimeError("frozen ablation roster mismatch")
    return result


def layout_metrics(layout: np.ndarray, recovered: Any) -> dict[str, float]:
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
    }


def paired(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(EXPERIMENT_SUBSET_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    low, high = np.quantile(array[indices].mean(axis=1), (0.025, 0.975))
    return {
        "mean": float(array.mean()),
        "bootstrap_ci95": [float(low), float(high)],
        "wins": int(np.sum(array > 0)),
        "ties": int(np.sum(array == 0)),
        "losses": int(np.sum(array < 0)),
    }


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("Refusing evaluation without --run")
    manifest = load_manifest(args.manifest)
    excluded, records = fresh_records(manifest)
    atlas = FrozenLowFrequencyPrior.load(args.atlas)
    if atlas.metadata.get("protocol_digest") != manifest["protocol_digest"]:
        raise ValueError("atlas/manifest protocol mismatch")
    if atlas.metadata.get("train_records") != 5_600:
        raise ValueError("fresh ablation requires the full train atlas")

    started = perf_counter()
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        name = str(record["filename"])
        dirty = load_rgb_verified(args.inputs / name, str(record["input_sha256"]))
        predictions = build_predictions(dirty, atlas)
        target = load_rgb_verified(args.targets / name, str(record["target_sha256"]))
        recovered = recover_layout(split_tiles(dirty), split_tiles(target))
        variants: dict[str, Any] = {}
        metric_cache: dict[str, dict[str, float]] = {}
        for variant, inference in predictions.items():
            layout_key = inference["layout_sha256"]
            if layout_key not in metric_cache:
                metric_cache[layout_key] = layout_metrics(inference["layout"], recovered)
            variants[variant] = {
                "raw_ssim": contest_ssim(target, inference["raw"]),
                "restored_ssim": contest_ssim(target, inference["restored"]),
                **metric_cache[layout_key],
                "layout_sha256": layout_key,
                "raw_sha256": inference["raw_sha256"],
                "restored_sha256": inference["restored_sha256"],
                "permutation_audit": inference["permutation_audit"],
                "restoration_seconds": inference["restoration_seconds"],
                "restoration_order": inference["restoration_order"],
            }
        rows.append(
            {
                "filename": name,
                "predictions_frozen_before_target_decode": True,
                "variants": variants,
            }
        )
        print(json.dumps({"done": index, "total": len(records)}), flush=True)

    names = tuple(rows[0]["variants"])
    summary = {
        name: {
            metric: float(np.mean([row["variants"][name][metric] for row in rows]))
            for metric in (
                "raw_ssim",
                "restored_ssim",
                "direct_placement",
                "translation_aligned_placement",
                "right_adjacency",
                "down_adjacency",
                "adjacency",
            )
        }
        for name in names
    }
    def arm(budget: int, h: int) -> str:
        return f"atlas_w0p06_buddies{budget}_proper_rgb_nlm_h{h}"

    comparisons = {
        f"h10_minus_h9_buddies{budget}": paired(
            np.asarray(
                [
                    row["variants"][arm(budget, 10)]["restored_ssim"]
                    - row["variants"][arm(budget, 9)]["restored_ssim"]
                    for row in rows
                ]
            )
        )
        for budget in EDGE_BUDGETS
    }
    comparisons.update(
        {
            f"buddies256_minus_96_h{h}": paired(
                np.asarray(
                    [
                        row["variants"][arm(256, h)]["restored_ssim"]
                        - row["variants"][arm(96, h)]["restored_ssim"]
                        for row in rows
                    ]
                )
            )
            for h in NLM_STRENGTHS
        }
    )
    selected = max(names, key=lambda name: summary[name]["restored_ssim"])
    compliance = all(
        row["variants"][name]["permutation_audit"]["passed"]
        for row in rows
        for name in names
    )
    result = {
        "schema_version": 1,
        "experiment": "compliant-atlas-edge-tail-ablation-v1",
        "status": "calibration_complete",
        "split": "calibration_fresh_excluding_prior48",
        "holdout_opened": False,
        "protocol_digest": manifest["protocol_digest"],
        "fresh_namespace": FRESH_NAMESPACE,
        "seed": EXPERIMENT_SUBSET_SEED,
        "excluded_prior48_digest": records_digest(excluded),
        "selection_digest": records_digest(records),
        "selection_count": len(records),
        "selection_disjoint_from_prior48": True,
        "frozen_roster": {
            "atlas_weight": ATLAS_WEIGHT,
            "edge_budgets": list(EDGE_BUDGETS),
            "proper_rgb_nlm_h": list(NLM_STRENGTHS),
            "arms": list(names),
        },
        "atlas_sha256": sha256_file(args.atlas),
        "compliance": {
            "all_permutation_audits_passed": compliance,
            "all_576_tiles_exactly_once_before_restoration": True,
            "raw_input_pixels_preserved": True,
            "population_atlas_used_only_as_unary": True,
            "blur_or_template_output_used": False,
            "target_used_at_inference": False,
            "predictions_frozen_before_target_decode": True,
        },
        "selected_variant": selected,
        "summary": summary,
        "paired_comparisons": comparisons,
        "runtime_seconds": perf_counter() - started,
        "per_board": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": selected,
                "selected_summary": summary[selected],
                "comparisons": comparisons,
                "compliance": compliance,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
