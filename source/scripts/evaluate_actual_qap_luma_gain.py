#!/usr/bin/env python3
"""Evaluate frozen bounded luminance gain on actual qap_w4 RGB harmonization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, SCRIPT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    blend_tiles_uint8,
    image_quality_metrics,
    paired_bootstrap_ci,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import split_tiles_numpy
from train_binary_edge_verifier import prepare_source, read_rgb
from evaluate_hgb_component_sync import baseline_layout


PANELS = ("primary_kornia", "independent_libjpeg")
RGB_CONFIG = SeamGraphConfig(
    extrapolation_band=3,
    confidence_scale=12.0,
    confidence_floor=0.05,
    ridge=0.2,
    huber_delta=4.0,
    irls_steps=4,
    max_abs_offset=12.0,
)
GAIN_CONFIG = LuminanceGainConfig(
    extrapolation_band=3,
    confidence_scale=0.08,
    confidence_floor=0.05,
    ridge=0.5,
    huber_delta=0.025,
    irls_steps=4,
    max_fractional_gain=0.04,
    luminance_floor=12.0,
    luminance_ceiling=243.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--selected-denoiser", required=True)
    parser.add_argument("--seam-denoiser", required=True)
    parser.add_argument("--embedding-checkpoint", required=True)
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--quarantine", default="configs/denoise_validation_quarantine_v1.json")
    parser.add_argument("--split", default="assembly_cal")
    parser.add_argument("--source-offset", type=int, default=32)
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--baseline-iterations", type=int, default=25)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def summarize(records: list[dict]) -> dict:
    panels = {}
    for panel in PANELS:
        selected = [record for record in records if record["panel"] == panel]
        ssim_delta = np.asarray([
            record["gain"]["ssim"] - record["rgb"]["ssim"] for record in selected
        ])
        seam_delta = np.asarray([
            record["gain"]["target_referenced_seam_error"]
            - record["rgb"]["target_referenced_seam_error"]
            for record in selected
        ])
        panels[panel] = {
            "mean_ssim_delta": float(ssim_delta.mean()),
            "paired_bootstrap_95_ci": list(
                paired_bootstrap_ci(ssim_delta, seed=20260722, resamples=20000)
            ),
            "mean_target_referenced_seam_error_delta": float(seam_delta.mean()),
            "wins_ties_losses": [
                int(np.count_nonzero(ssim_delta > 0)),
                int(np.count_nonzero(ssim_delta == 0)),
                int(np.count_nonzero(ssim_delta < 0)),
            ],
            "worst_ssim_delta": float(ssim_delta.min()),
        }
    source_names = sorted({record["name"] for record in records})
    source_delta = np.asarray([
        np.mean([
            record["gain"]["ssim"] - record["rgb"]["ssim"]
            for record in records if record["name"] == name
        ])
        for name in source_names
    ])
    source_ci = paired_bootstrap_ci(source_delta, seed=20260723, resamples=20000)
    return {
        "panels": panels,
        "source_macro_mean_ssim_delta": float(source_delta.mean()),
        "source_macro_paired_bootstrap_95_ci": list(source_ci),
        "source_macro_wins": int(np.count_nonzero(source_delta > 0)),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_model, device, selected_metadata = load_restorer(
        args.selected_denoiser, device=args.device, state="ema"
    )
    seam_model, seam_device, seam_metadata = load_restorer(
        args.seam_denoiser, device=str(device), state="ema"
    )
    if seam_device != device:
        raise RuntimeError("selected and seam models resolved to different devices")
    embedding, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    source_names = source_names_for_split(
        args.split, manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.source_offset : args.source_offset + args.sources]
    records = []
    started = time.time()
    for source_index, name in enumerate(source_names):
        clean = read_rgb(Path(args.data_root) / "train/targets" / name)
        target_tiles = split_tiles_numpy(clean)
        for panel in PANELS:
            panel_seed = per_source_seed(args.seed, f"actual-qap-luma-{panel}", name, 0)
            prepared = prepare_source(
                name,
                panel,
                panel_seed,
                args=args,
                restorer=selected_model,
                embedding_model=embedding,
                device=device,
            )
            seam_tiles = restore_tiles_uint8(
                seam_model, prepared.raw_tiles, seam_device, batch_size=args.batch_size
            )
            qap_seed = per_source_seed(args.seed, f"actual-qap-luma-qap-{panel}", name, 0)
            layout = baseline_layout(
                prepared, qap_seed=qap_seed, iterations=args.baseline_iterations
            )
            selected_ordered = np.ascontiguousarray(prepared.denoised_tiles[layout])
            seam_ordered = np.ascontiguousarray(seam_tiles[layout])
            blend = blend_tiles_uint8(selected_ordered, seam_ordered, auxiliary_weight=0.5)
            offsets, rgb_diagnostics = seam_graph_rgb_offsets(blend, RGB_CONFIG)
            rgb = apply_rgb_offsets(blend, offsets)
            gains, gain_diagnostics = seam_graph_luminance_gains(rgb, GAIN_CONFIG)
            gain = apply_luminance_gains(rgb, gains)
            records.append({
                "name": name,
                "panel": panel,
                "rgb": image_quality_metrics(rgb, target_tiles),
                "gain": image_quality_metrics(gain, target_tiles),
                "rgb_diagnostics": rgb_diagnostics,
                "gain_diagnostics": gain_diagnostics,
            })
        print(json.dumps({"stage": "actual_qap_luma", "done": source_index + 1, "total": len(source_names)}), flush=True)
    summary = summarize(records)
    eligible = (
        summary["source_macro_mean_ssim_delta"] >= 0.001
        and summary["source_macro_paired_bootstrap_95_ci"][0] > 0.0
        and summary["source_macro_wins"] >= 10
        and all(
            panel["mean_ssim_delta"] >= 0.001
            and panel["paired_bootstrap_95_ci"][0] > 0.0
            and panel["mean_target_referenced_seam_error_delta"] <= 0.0
            for panel in summary["panels"].values()
        )
    )
    payload = {
        "schema_version": 1,
        "kind": "actual_qap_w4_bounded_luminance_gain_calibration",
        "split": args.split,
        "source_offset": args.source_offset,
        "source_names": source_names,
        "records": records,
        "summary": summary,
        "gate": "source/panels delta>=.001, bootstrap lower>0, >=10/16 source wins, seam error nonregression",
        "eligible": bool(eligible),
        "selected": "rgb_plus_bounded_luminance_gain" if eligible else None,
        "model_metadata": {
            "selected": selected_metadata,
            "seam": seam_metadata,
            "embedding": embedding_metadata,
        },
        "seconds": time.time() - started,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": payload["selected"], "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
