#!/usr/bin/env python3
"""Train-split-only diagnostics for conservative k16 mutual-edge fusion.

This script is deliberately separate from the preregistered calibration runner.
It uses records 256:280 of the manifest's ``train`` split, which were not used
to fit the frozen 256-board checkpoint.  The target is opened only after every
input-only arm prediction for the board has been constructed.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from run_edge_ranker_k16_tail import (
    CHECKPOINT,
    INPUTS,
    MANIFEST,
    TARGETS,
    VIEWS,
    choose_device,
    harmonized_tail,
    load_checkpoint,
    load_verified_rgb,
)

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.edge_ranker import build_inference_board, score_board
from aiijc_puzzle.edge_ranker_conservative_fusion import FusionArm, apply_conservative_fusion
from aiijc_puzzle.edge_ranker_final_tail import layout_metrics, names_digest
from aiijc_puzzle.frozen_final_evaluator import _validate_method_configs, atomic_json
from aiijc_puzzle.legacy_upgrade import layout_digest, solve_buddies
from aiijc_puzzle.protocol import assemble_tiles, contest_ssim, select_manifest_records, split_tiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "edge-ranker"
    / "conservative-fusion-train-diagnostic-256-280"
    / "report.json"
)
TRAIN_OFFSET = 256
TRAIN_COUNT = 24

# Exploratory arms are intentionally broad.  At most five will be copied into
# the immutable calibration preregistration after this train-only report exists.
ARMS = (
    FusionArm("cap04-v0-c000", 4, 0, 0.0),
    FusionArm("cap08-v0-c000", 8, 0, 0.0),
    FusionArm("cap16-v0-c000", 16, 0, 0.0),
    FusionArm("cap32-v0-c000", 32, 0, 0.0),
    FusionArm("cap08-v2-c000", 8, 2, 0.0),
    FusionArm("cap16-v2-c000", 16, 2, 0.0),
    FusionArm("cap32-v2-c000", 32, 2, 0.0),
    FusionArm("cap08-v3-c000", 8, 3, 0.0),
    FusionArm("cap16-v3-c000", 16, 3, 0.0),
    FusionArm("cap08-v0-c050", 8, 0, 0.5),
    FusionArm("cap16-v0-c050", 16, 0, 0.5),
    FusionArm("cap16-v2-c050", 16, 2, 0.5),
)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ranked = select_manifest_records(
        manifest,
        "train",
        limit=TRAIN_OFFSET + TRAIN_COUNT,
    )
    records = tuple(ranked[TRAIN_OFFSET:])
    if len(records) != TRAIN_COUNT:
        raise RuntimeError("train diagnostic panel is incomplete")
    device = choose_device("mps")
    model, _ = load_checkpoint(CHECKPOINT, manifest=manifest, device=device)
    rgb_config, luma_config, _ = _validate_method_configs()
    board_reports = []
    started = perf_counter()
    for board_index, record in enumerate(records, start=1):
        dirty = load_verified_rgb(INPUTS / str(record["filename"]), str(record["input_sha256"]))
        tiles = split_tiles(dirty)
        board = build_inference_board(
            tiles,
            filename=str(record["filename"]),
            views=VIEWS,
            candidate_k=16,
        )
        learned_right, learned_down, delta = score_board(
            model,
            board,
            device=device,
            pair_batch=1024,
        )
        score_arms = {
            "baseline": (board.right_baseline, board.down_baseline, {"selected_count": 0})
        }
        for arm in ARMS:
            score_arms[arm.name] = apply_conservative_fusion(
                board,
                learned_right,
                learned_down,
                arm,
            )
        predictions = {}
        tail_cache = {}
        for name, (right, down, diagnostics) in score_arms.items():
            solved = solve_buddies(right, down, max_edges=96)
            digest = layout_digest(solved.layout)
            if digest not in tail_cache:
                ordered = np.ascontiguousarray(tiles[solved.layout])
                raw = assemble_tiles(ordered)
                tail = harmonized_tail(ordered, rgb_config, luma_config)
                tail_cache[digest] = (raw, tail["final"])
            raw, final = tail_cache[digest]
            predictions[name] = {
                "layout": solved.layout,
                "layout_sha256": digest,
                "raw": raw,
                "final": final,
                "diagnostics": diagnostics,
            }
        # No target path is constructed until every variant above is frozen.
        target = load_verified_rgb(TARGETS / str(record["filename"]), str(record["target_sha256"]))
        recovered = recover_layout(board.tiles, split_tiles(target))
        variants = {}
        for name, prediction in predictions.items():
            variants[name] = {
                **layout_metrics(prediction["layout"], recovered),
                "raw_ssim": contest_ssim(target, prediction["raw"]),
                "final_ssim": contest_ssim(target, prediction["final"]),
                "layout_sha256": prediction["layout_sha256"],
                "selected_count": int(prediction["diagnostics"]["selected_count"]),
            }
        board_reports.append(
            {
                "filename": record["filename"],
                "score_delta": delta,
                "unique_layouts": len(tail_cache),
                "variants": variants,
            }
        )
        print(f"diagnosed {board_index}/{len(records)} {record['filename']}", flush=True)

    metric_names = (
        "adjacency",
        "translation_aligned_placement",
        "raw_ssim",
        "final_ssim",
        "selected_count",
    )
    means = {
        name: {
            metric: float(np.mean([board["variants"][name][metric] for board in board_reports]))
            for metric in metric_names
        }
        for name in ("baseline", *(arm.name for arm in ARMS))
    }
    baseline = means["baseline"]
    deltas = {
        name: {metric: means[name][metric] - baseline[metric] for metric in metric_names[:-1]}
        for name in means
        if name != "baseline"
    }
    report = {
        "schema": "aiijc-edge-ranker-conservative-fusion-train-diagnostic-v1",
        "split": "train",
        "offset": TRAIN_OFFSET,
        "count": TRAIN_COUNT,
        "filenames": [record["filename"] for record in records],
        "filenames_sha256": names_digest(records),
        "checkpoint_fit_records": "train[0:256]",
        "diagnostic_records": "train[256:280]",
        "calibration_or_holdout_targets_accessed": False,
        "target_access_policy": "per-board after every input-only arm prediction is frozen",
        "arms": [arm.__dict__ for arm in ARMS],
        "means": means,
        "deltas_vs_baseline": deltas,
        "boards": board_reports,
        "runtime_seconds": perf_counter() - started,
    }
    atomic_json(OUTPUT, report)
    print(json.dumps({"means": means, "deltas": deltas}, indent=2), flush=True)


if __name__ == "__main__":
    main()
