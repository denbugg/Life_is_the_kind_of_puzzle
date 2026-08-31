#!/usr/bin/env python3
"""Run clean-oracle and recovered-dirty content-substitution SSIM experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aiijc_puzzle.content_substitution import (
    aggregate_dirty_alignments,
    aggregate_evaluations,
    build_assignments,
    contest_rgb_ssim,
    evaluate_variants,
    extract_tiles,
    pairwise_tile_rmse,
    recover_dirty_tile_alignment,
    select_target_paths,
)
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        help="matching shuffled train inputs; enables the realistic raw-dirty proxy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="frozen validation manifest; use with --split for official experiment panels",
    )
    parser.add_argument("--split", choices=("calibration", "holdout"))
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, default=420_048)
    parser.add_argument("--assignment-seed", type=int, default=420)
    parser.add_argument("--pool-start", type=int, default=0)
    parser.add_argument("--pool-stop", type=int)
    parser.add_argument("--nearest-k", type=int, nargs="+", default=(3, 10))
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--tile-size", type=int, default=20)
    parser.add_argument(
        "--apply-nlm-h9",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the frozen winning full-frame colored NLM tail to dirty renders",
    )
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="save every rendered variant as PNG",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    expected_shape = (args.grid_size * args.tile_size,) * 2 + (3,)
    manifest = None
    selected_records: tuple[dict[str, Any], ...] = ()
    if (args.manifest is None) != (args.split is None):
        raise ValueError("--manifest and --split must be provided together")
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected_digest = compute_protocol_digest(manifest)
        if manifest.get("protocol_digest") != expected_digest:
            raise ValueError(f"validation manifest digest mismatch: {args.manifest}")
        selected_records = tuple(
            dict(record)
            for record in select_manifest_records(manifest, args.split, limit=args.count)
        )
        selected_paths = [args.targets_dir / record["filename"] for record in selected_records]
    else:
        selected_paths = select_target_paths(
            args.targets_dir,
            count=args.count,
            seed=args.selection_seed,
            pool_start=args.pool_start,
            pool_stop=args.pool_stop,
        )
    selected_record_by_name = {record["filename"]: record for record in selected_records}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    board_results: list[dict[str, Any]] = []
    rmse_samples: dict[str, list[np.ndarray]] = {}
    dirty_alignments = []
    experiment_started = time.perf_counter()

    for board_number, path in enumerate(selected_paths, start=1):
        board_started = time.perf_counter()
        record = selected_record_by_name.get(path.name)
        if record is not None and sha256_file(path) != record["target_sha256"]:
            raise ValueError(f"target hash does not match frozen manifest: {path.name}")
        with Image.open(path) as source:
            target = np.asarray(source.convert("RGB"), dtype=np.uint8)
        if target.shape != expected_shape:
            raise ValueError(f"{path}: expected shape {expected_shape}, got {target.shape}")

        tiles = extract_tiles(target, grid_size=args.grid_size, tile_size=args.tile_size)
        cost_started = time.perf_counter()
        costs = pairwise_tile_rmse(tiles)
        cost_seconds = time.perf_counter() - cost_started
        assignments = build_assignments(
            costs,
            seed=args.assignment_seed,
            board_key=path.name,
            nearest_ks=args.nearest_k,
        )

        evaluation_started = time.perf_counter()
        clean_evaluations = evaluate_variants(
            target,
            costs,
            assignments,
            grid_size=args.grid_size,
        )
        clean_evaluation_seconds = time.perf_counter() - evaluation_started

        clean_variants: dict[str, dict[str, Any]] = {}
        for name, evaluation in clean_evaluations.items():
            clean_variants[name] = evaluation.metrics
            rmse_samples.setdefault(name, []).append(evaluation.selected_rmse)
            if args.save_images:
                image_path = args.output_dir / "images" / path.stem / "clean_oracle" / f"{name}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(evaluation.rendered, mode="RGB").save(image_path)

        dirty_alignment = None
        raw_dirty_variants: dict[str, dict[str, Any]] | None = None
        dirty_alignment_seconds = 0.0
        dirty_evaluation_seconds = 0.0
        nlm_evaluation_seconds = 0.0
        nlm_dirty_variants: dict[str, dict[str, Any]] | None = None
        if args.inputs_dir is not None:
            input_path = args.inputs_dir / path.name
            if not input_path.is_file():
                raise FileNotFoundError(f"missing matching train input: {input_path}")
            if record is not None and sha256_file(input_path) != record["input_sha256"]:
                raise ValueError(f"input hash does not match frozen manifest: {path.name}")
            with Image.open(input_path) as source:
                input_image = np.asarray(source.convert("RGB"), dtype=np.uint8)
            if input_image.shape != expected_shape:
                raise ValueError(
                    f"{input_path}: expected shape {expected_shape}, got {input_image.shape}"
                )

            alignment_started = time.perf_counter()
            dirty_alignment = recover_dirty_tile_alignment(
                input_image,
                target,
                grid_size=args.grid_size,
                tile_size=args.tile_size,
            )
            dirty_alignment_seconds = time.perf_counter() - alignment_started
            dirty_alignments.append(dirty_alignment)
            dirty_evaluation_started = time.perf_counter()
            dirty_evaluations = evaluate_variants(
                target,
                costs,
                assignments,
                grid_size=args.grid_size,
                source_tiles=dirty_alignment.aligned_tiles,
            )
            dirty_evaluation_seconds = time.perf_counter() - dirty_evaluation_started
            raw_dirty_variants = {
                name: evaluation.metrics for name, evaluation in dirty_evaluations.items()
            }
            if args.apply_nlm_h9:
                nlm_evaluation_started = time.perf_counter()
                nlm_dirty_variants = {}
                for name, evaluation in dirty_evaluations.items():
                    filtered = apply_nlm_color(evaluation.rendered, h=9)
                    nlm_metrics = dict(evaluation.metrics)
                    nlm_metrics["ssim"] = contest_rgb_ssim(target, filtered.image)
                    nlm_metrics["tail_runtime_seconds"] = filtered.seconds
                    nlm_dirty_variants[name] = nlm_metrics
                    if args.save_images:
                        image_path = (
                            args.output_dir
                            / "images"
                            / path.stem
                            / "nlm_h9_dirty_proxy"
                            / f"{name}.png"
                        )
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(filtered.image, mode="RGB").save(image_path)
                nlm_evaluation_seconds = time.perf_counter() - nlm_evaluation_started
            if args.save_images:
                for name, evaluation in dirty_evaluations.items():
                    image_path = (
                        args.output_dir / "images" / path.stem / "raw_dirty_proxy" / f"{name}.png"
                    )
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(evaluation.rendered, mode="RGB").save(image_path)

        board_seconds = time.perf_counter() - board_started
        board_result: dict[str, Any] = {
            "board": path.name,
            "pairwise_cost_seconds": cost_seconds,
            "clean_evaluation_seconds": clean_evaluation_seconds,
            "dirty_alignment_seconds": dirty_alignment_seconds,
            "dirty_evaluation_seconds": dirty_evaluation_seconds,
            "nlm_evaluation_seconds": nlm_evaluation_seconds,
            "runtime_seconds": board_seconds,
            "clean_oracle_variants": clean_variants,
        }
        if dirty_alignment is not None and raw_dirty_variants is not None:
            board_result["dirty_alignment"] = dirty_alignment.metrics
            board_result["raw_dirty_proxy_variants"] = raw_dirty_variants
        if nlm_dirty_variants is not None:
            board_result["nlm_h9_dirty_proxy_variants"] = nlm_dirty_variants
        board_results.append(board_result)
        nearest_ssim = clean_variants["nearest_other"]["ssim"]
        bijective_ssim = clean_variants["bijective_derangement"]["ssim"]
        dirty_message = ""
        if raw_dirty_variants is not None:
            dirty_message = (
                f" dirty_identity={raw_dirty_variants['identity']['ssim']:.6f}"
                f" dirty_bijective={raw_dirty_variants['bijective_derangement']['ssim']:.6f}"
            )
        if nlm_dirty_variants is not None:
            dirty_message += (
                f" nlm_identity={nlm_dirty_variants['identity']['ssim']:.6f}"
                f" nlm_bijective={nlm_dirty_variants['bijective_derangement']['ssim']:.6f}"
            )
        print(
            f"[{board_number:02d}/{len(selected_paths):02d}] {path.name} "
            f"clean_nearest={nearest_ssim:.6f} clean_bijective={bijective_ssim:.6f}"
            f"{dirty_message} "
            f"time={board_seconds:.2f}s",
            flush=True,
        )

    total_seconds = time.perf_counter() - experiment_started
    pool_stop = args.pool_stop
    if pool_stop is None:
        pool_stop = len(sorted(args.targets_dir.glob("*.png")))
    selected_board_names = [path.name for path in selected_paths]
    selection_digest = hashlib.sha256("\n".join(selected_board_names).encode()).hexdigest()
    configuration = {
        "schema_version": 2,
        "targets_dir": str(args.targets_dir.resolve()),
        "inputs_dir": None if args.inputs_dir is None else str(args.inputs_dir.resolve()),
        "target_source": (
            "clean train targets select content substitutes and score outputs; "
            "no test files or historical labels"
        ),
        "dirty_proxy_source": (
            None
            if args.inputs_dir is None
            else "matching shuffled train inputs aligned by target-assisted 5x5 descriptors"
        ),
        "count": args.count,
        "manifest_path": None if args.manifest is None else str(args.manifest.resolve()),
        "protocol_digest": None if manifest is None else manifest["protocol_digest"],
        "split": args.split,
        "selection": (
            "SHA-256 rank of filename within the sorted positional pool"
            if manifest is None
            else "shared protocol.select_manifest_records panel"
        ),
        "selection_seed": (args.selection_seed if manifest is None else EXPERIMENT_SUBSET_SEED),
        "selection_namespace": None if manifest is None else EXPERIMENT_SUBSET_NAMESPACE,
        "selection_digest": selection_digest,
        "assignment_seed": args.assignment_seed,
        "pool_start": args.pool_start if manifest is None else None,
        "pool_stop": pool_stop if manifest is None else None,
        "nearest_k": list(args.nearest_k),
        "grid_size": args.grid_size,
        "tile_size": args.tile_size,
        "apply_nlm_h9": args.apply_nlm_h9,
        "save_images": args.save_images,
        "selected_boards": selected_board_names,
        "metric": ("skimage.metrics.structural_similarity(channel_axis=2, data_range=255)"),
        "cost": "full 20x20x3 clean-tile RGB RMSE via exact-range float64 Gram matrix",
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "scikit_image": version("scikit-image"),
        },
    }
    aggregate = {
        "configuration": configuration,
        "runtime_seconds": total_seconds,
        "runtime_seconds_per_board": total_seconds / len(board_results),
        "clean_oracle_variants": aggregate_evaluations(
            [board["clean_oracle_variants"] for board in board_results], rmse_samples
        ),
    }
    if args.inputs_dir is not None:
        aggregate["raw_dirty_proxy_variants"] = aggregate_evaluations(
            [board["raw_dirty_proxy_variants"] for board in board_results], rmse_samples
        )
        aggregate["dirty_alignment"] = aggregate_dirty_alignments(dirty_alignments)
    if args.inputs_dir is not None and args.apply_nlm_h9:
        aggregate["nlm_h9_dirty_proxy_variants"] = aggregate_evaluations(
            [board["nlm_h9_dirty_proxy_variants"] for board in board_results], rmse_samples
        )
    _write_json(args.output_dir / "per_board.json", board_results)
    _write_json(args.output_dir / "aggregate.json", aggregate)
    print(f"wrote {args.output_dir / 'per_board.json'}")
    print(f"wrote {args.output_dir / 'aggregate.json'}")
    print(f"total runtime: {total_seconds:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
