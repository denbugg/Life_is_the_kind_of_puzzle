#!/usr/bin/env python3
"""Build high-purity real tile pairs or calibrate the matching gate.

Examples:

  python scripts/build_real_gold.py build --output runs/denoise_v2/real_gold_train.npz
  python scripts/build_real_gold.py build --split val --limit 2 --output /tmp/real_gold_smoke.npz
  python scripts/build_real_gold.py calibrate --split audit --limit 4 --repeats 2

The build output stores indices and matching diagnostics only; pixel arrays are
never duplicated into the NPZ.  The calibrate mode creates synthetic corrupted
tiles with a known permutation and reports gate precision/coverage without
using any pseudo-map as ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_denoise_v2.degradation import SyntheticTileDegrader
from puzzle_denoise_v2.matching import (
    MatchingThresholds,
    calibration_report,
    match_tile_sets,
    summarize_match,
)
from puzzle_denoise_v2.tiles import GRID, split_tiles_numpy


def _read_tiles(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return split_tiles_numpy(rgb)


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_splits = {"train", "val", "audit"}
    if set(payload.get("splits", {})) != required_splits:
        raise ValueError(f"manifest must contain exactly {sorted(required_splits)}")
    if not payload.get("policy", {}).get("exclude_all_test_filename_overlaps", False):
        raise ValueError("manifest does not enforce test-filename exclusion")
    excluded = set(payload.get("excluded_test_overlap", []))
    for split, names in payload["splits"].items():
        leaked = excluded & set(names)
        if leaked:
            raise ValueError(f"manifest split {split} contains {len(leaked)} excluded names")
    return payload


def _select_names(
    payload: dict,
    data_root: Path,
    split: str,
    seed: int,
    limit: int,
) -> list[str]:
    if split not in payload["splits"]:
        raise ValueError(f"unknown split {split}")
    names = list(payload["splits"][split])
    excluded = set(payload["excluded_test_overlap"])
    actual_test_names = {path.name for path in (data_root / "test").glob("*.png")}
    leaked = (set(names) & excluded) | (set(names) & actual_test_names)
    if leaked:
        raise ValueError(f"selected split contains {len(leaked)} test-overlap names")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    names = [names[int(index)] for index in order]
    if limit:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        names = names[:limit]
    if not names:
        raise ValueError("selected source list is empty")

    missing = []
    for name in names:
        if not (data_root / "train" / "inputs" / name).is_file():
            missing.append(f"input:{name}")
        if not (data_root / "train" / "targets" / name).is_file():
            missing.append(f"target:{name}")
    if missing:
        raise FileNotFoundError(f"missing selected data files: {missing[:5]}")
    return names


def _thresholds(args: argparse.Namespace) -> MatchingThresholds:
    return MatchingThresholds(
        coarse_min_margin=args.coarse_min_margin,
        structural_min_margin=args.structural_min_margin,
        joint_min_confidence=args.joint_min_confidence,
    )


def _coverage_quantiles(counts: np.ndarray) -> dict[str, float]:
    coverage = counts.astype(np.float64) / float(GRID * GRID)
    levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return {
        f"q{int(level * 100):02d}": float(value)
        for level, value in zip(levels, np.quantile(coverage, levels), strict=True)
    }


def _concat(parts: list[np.ndarray], dtype) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def build_gold(args: argparse.Namespace) -> dict:
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest)
    payload = _load_manifest(manifest_path)
    names = _select_names(payload, data_root, args.split, args.seed, args.limit)
    thresholds = _thresholds(args)
    if len(names) > np.iinfo(np.uint16).max:
        raise ValueError("compact uint16 source_index cannot encode this many sources")

    pair_parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "source_index",
            "input_slot",
            "clean_tile_index",
            "coarse_cost",
            "structural_cost",
            "coarse_row_margin",
            "coarse_column_margin",
            "structural_row_margin",
            "structural_column_margin",
            "joint_confidence",
            "consensus",
            "coarse_mutual_cycle",
            "structural_mutual_cycle",
        )
    }
    source_consensus = np.zeros(len(names), dtype=np.uint16)
    source_both_mutual = np.zeros(len(names), dtype=np.uint16)
    source_selected = np.zeros(len(names), dtype=np.uint16)

    iterator = tqdm(enumerate(names), total=len(names), desc=f"matching {args.split}")
    for source_index, name in iterator:
        input_tiles = _read_tiles(data_root / "train" / "inputs" / name)
        clean_tiles = _read_tiles(data_root / "train" / "targets" / name)
        result = match_tile_sets(input_tiles, clean_tiles, thresholds)
        selected_slots = np.flatnonzero(result.selected)
        both_mutual = result.coarse.mutual_nn_cycle & result.structural.mutual_nn_cycle
        source_consensus[source_index] = int(result.consensus.sum())
        source_both_mutual[source_index] = int(both_mutual.sum())
        source_selected[source_index] = len(selected_slots)

        pair_parts["source_index"].append(np.full(len(selected_slots), source_index))
        pair_parts["input_slot"].append(selected_slots)
        pair_parts["clean_tile_index"].append(result.coarse.mapping[selected_slots])
        pair_parts["coarse_cost"].append(result.coarse.assigned_cost[selected_slots])
        pair_parts["structural_cost"].append(result.structural.assigned_cost[selected_slots])
        pair_parts["coarse_row_margin"].append(result.coarse.row_margin[selected_slots])
        pair_parts["coarse_column_margin"].append(result.coarse.column_margin[selected_slots])
        pair_parts["structural_row_margin"].append(result.structural.row_margin[selected_slots])
        pair_parts["structural_column_margin"].append(result.structural.column_margin[selected_slots])
        pair_parts["joint_confidence"].append(result.joint_confidence[selected_slots])
        pair_parts["consensus"].append(result.consensus[selected_slots])
        pair_parts["coarse_mutual_cycle"].append(result.coarse.mutual_nn_cycle[selected_slots])
        pair_parts["structural_mutual_cycle"].append(result.structural.mutual_nn_cycle[selected_slots])
        if args.verbose:
            print(json.dumps({"event": "source_match", "name": name, **summarize_match(result)}, sort_keys=True))

    selected_pairs = int(source_selected.sum())
    total_tiles = len(names) * GRID * GRID
    metadata = {
        "schema_version": 1,
        "kind": "high_purity_real_tile_pairs",
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_seed": payload.get("seed"),
        "selection_seed": args.seed,
        "split": args.split,
        "limit": args.limit,
        "source_count": len(names),
        "total_tiles": total_tiles,
        "selected_pairs": selected_pairs,
        "selected_coverage": selected_pairs / total_tiles,
        "source_selected_coverage_quantiles": _coverage_quantiles(source_selected),
        "thresholds": thresholds.to_dict(),
        "test_overlap_excluded": len(payload["excluded_test_overlap"]),
        "source_name_encoding": "source_names[source_index]",
        "descriptors": {
            "coarse": "5x5 pooled normalized RGB plus weak absolute colour",
            "structural": "multi-scale cosine RGB/luminance/gradient ensemble",
        },
        "selection_rule": (
            "coarse and structural Hungarian assignments agree; each assignment is a "
            "bidirectional mutual-nearest-neighbour cycle; both row and column margins "
            "and normalized joint confidence meet the recorded thresholds"
        ),
        "joint_confidence_definition": (
            "minimum of each descriptor's bidirectional assigned margin divided by "
            "that image's median positive descriptor margin"
        ),
        "old_q90_used_as_ground_truth": False,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        meta=np.asarray(json.dumps(metadata, sort_keys=True)),
        source_names=np.asarray(names),
        source_index=_concat(pair_parts["source_index"], np.uint16),
        input_slot=_concat(pair_parts["input_slot"], np.uint16),
        clean_tile_index=_concat(pair_parts["clean_tile_index"], np.uint16),
        coarse_cost=_concat(pair_parts["coarse_cost"], np.float32),
        structural_cost=_concat(pair_parts["structural_cost"], np.float32),
        coarse_row_margin=_concat(pair_parts["coarse_row_margin"], np.float32),
        coarse_column_margin=_concat(pair_parts["coarse_column_margin"], np.float32),
        structural_row_margin=_concat(pair_parts["structural_row_margin"], np.float32),
        structural_column_margin=_concat(pair_parts["structural_column_margin"], np.float32),
        joint_confidence=_concat(pair_parts["joint_confidence"], np.float32),
        consensus=_concat(pair_parts["consensus"], np.uint8),
        coarse_mutual_cycle=_concat(pair_parts["coarse_mutual_cycle"], np.uint8),
        structural_mutual_cycle=_concat(pair_parts["structural_mutual_cycle"], np.uint8),
        source_consensus_count=source_consensus,
        source_both_mutual_count=source_both_mutual,
        source_selected_count=source_selected,
    )
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    result = {"event": "real_gold_written", "output": str(output), "sha256": sha256, **metadata}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@torch.no_grad()
def calibrate(args: argparse.Namespace) -> dict:
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest)
    payload = _load_manifest(manifest_path)
    names = _select_names(payload, data_root, args.split, args.seed, args.limit)
    thresholds = _thresholds(args)
    degrader = SyntheticTileDegrader()
    numpy_rng = np.random.default_rng(args.seed)
    torch_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    results = []
    true_mappings = []
    iterator = tqdm(names, desc=f"calibrating {args.split}")
    for name in iterator:
        clean_tiles = _read_tiles(data_root / "train" / "targets" / name)
        for repeat in range(args.repeats):
            true_mapping = numpy_rng.permutation(GRID * GRID).astype(np.int32)
            shuffled_clean = clean_tiles[true_mapping]
            tensor = torch.from_numpy(
                np.ascontiguousarray(shuffled_clean.transpose(0, 3, 1, 2))
            ).float().div_(255.0)
            corrupted, _ = degrader(tensor, generator=torch_generator)
            corrupted_tiles = np.clip(
                np.rint(corrupted.numpy().transpose(0, 2, 3, 1) * 255.0),
                0,
                255,
            ).astype(np.uint8)
            result = match_tile_sets(corrupted_tiles, clean_tiles, thresholds)
            results.append(result)
            true_mappings.append(true_mapping)
            if args.verbose:
                correct = result.coarse.mapping == true_mapping
                print(
                    json.dumps(
                        {
                            "event": "calibration_example",
                            "name": name,
                            "repeat": repeat,
                            "coarse_accuracy": float(correct.mean()),
                            **summarize_match(result),
                        },
                        sort_keys=True,
                    )
                )

    report = calibration_report(results, true_mappings)
    report.update(
        {
            "event": "synthetic_matching_calibration",
            "data_root": str(data_root),
            "manifest": str(manifest_path),
            "split": args.split,
            "source_count": len(names),
            "repeats": args.repeats,
            "seed": args.seed,
            "degradation": "SyntheticTileDegrader primary noise-before-blur variant",
            "ground_truth": "known synthetic input-slot to clean-tile permutation",
            "old_q90_used_as_ground_truth": False,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def _add_common(parser: argparse.ArgumentParser, *, split: str, limit: int) -> None:
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--split", choices=["train", "val", "audit"], default=split)
    parser.add_argument("--limit", type=int, default=limit, help="deterministic source-image limit; 0 means full split")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--coarse-min-margin", type=float, default=1e-6)
    parser.add_argument("--structural-min-margin", type=float, default=1e-6)
    parser.add_argument(
        "--joint-min-confidence",
        type=float,
        default=0.45,
        help="optional calibrated normalized-confidence floor; inspect calibrate sweep first",
    )
    parser.add_argument("--verbose", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="write compact high-purity real-pair NPZ")
    _add_common(build_parser, split="train", limit=0)
    build_parser.add_argument("--output", required=True)
    build_parser.set_defaults(func=build_gold)

    calibration_parser = subparsers.add_parser(
        "calibrate",
        help="measure matching precision/coverage on known synthetic permutations",
    )
    _add_common(calibration_parser, split="audit", limit=4)
    calibration_parser.add_argument("--repeats", type=int, default=1)
    calibration_parser.set_defaults(func=calibrate)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if getattr(arguments, "repeats", 1) <= 0:
        raise SystemExit("--repeats must be positive")
    arguments.func(arguments)
