#!/usr/bin/env python3
"""Calibrate robust translation synchronization plus Hungarian grid rounding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, SCRIPT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.geometry import GRID, TILE_COUNT, validate_permutation
from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer
from train_binary_edge_verifier import PreparedSource, prepare_source, read_rgb
from evaluate_hgb_component_sync import baseline_layout


PANELS = ("primary_kornia", "independent_libjpeg")
IDEAL_STD = float(np.sqrt((GRID * GRID - 1.0) / 12.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tabular-report", required=True)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", required=True)
    parser.add_argument("--embedding-checkpoint", required=True)
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--quarantine", default="configs/denoise_validation_quarantine_v1.json")
    parser.add_argument("--split", default="assembly_cal")
    parser.add_argument("--source-offset", type=int, default=48)
    parser.add_argument("--sources", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--baseline-iterations", type=int, default=25)
    parser.add_argument("--irls-rounds", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def initial_edge_weight(prepared: PreparedSource, probability: np.ndarray, threshold85: float, boost: float) -> np.ndarray:
    c1_rank = prepared.features[:, 0].astype(np.float64) * (TILE_COUNT - 1)
    hbt_rank = prepared.features[:, 4].astype(np.float64) * (TILE_COUNT - 1)
    rank_weight = np.exp(-hbt_rank / 4.0) + 0.35 * np.exp(-c1_rank / 4.0)
    confidence = np.maximum(0.0, (probability - threshold85) / max(1.0 - threshold85, 1e-12))
    return rank_weight * (1.0 + boost * confidence)


def select_edges(prepared: PreparedSource, weights: np.ndarray, top_k: int) -> np.ndarray:
    graph = prepared.graph
    chosen = []
    for direction in (0, 1):
        for source in range(TILE_COUNT):
            indices = np.flatnonzero((graph.direction == direction) & (graph.source == source))
            order = indices[np.argsort(-weights[indices], kind="stable")]
            chosen.extend(order[:top_k].tolist())
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def solve_coordinates(
    prepared: PreparedSource,
    base_weight: np.ndarray,
    selected: np.ndarray,
    *,
    cutoff: float,
    rounds: int,
    ridge: float,
) -> tuple[np.ndarray, dict]:
    graph = prepared.graph
    source = graph.source[selected].astype(np.int32)
    destination = graph.destination[selected].astype(np.int32)
    direction = graph.direction[selected].astype(np.int32)
    count = len(selected)
    rows = np.repeat(np.arange(count, dtype=np.int32), 2)
    columns = np.empty(count * 2, dtype=np.int32)
    columns[0::2] = source
    columns[1::2] = destination
    data = np.tile(np.asarray([-1.0, 1.0]), count)
    incidence = coo_matrix((data, (rows, columns)), shape=(count, TILE_COUNT)).tocsr()
    displacement = np.stack(
        [(direction == 0).astype(np.float64), (direction == 1).astype(np.float64)],
        axis=1,
    )
    weights = np.maximum(base_weight[selected].astype(np.float64), 1e-12)
    coordinates = np.zeros((TILE_COUNT, 2), dtype=np.float64)
    residual = np.full(count, np.inf, dtype=np.float64)
    for _ in range(rounds):
        weighted = incidence.multiply(weights[:, None])
        laplacian = incidence.T @ weighted + ridge * eye(TILE_COUNT, format="csr")
        rhs = incidence.T @ (weights[:, None] * displacement)
        for axis in (0, 1):
            coordinates[:, axis] = spsolve(laplacian, rhs[:, axis])
        coordinates -= coordinates.mean(axis=0, keepdims=True)
        error = incidence @ coordinates - displacement
        residual = np.sqrt(np.sum(error * error, axis=1))
        weights = base_weight[selected] / (1.0 + np.square(residual / cutoff))
        weights = np.maximum(weights, 1e-12)
    scaled = coordinates.copy()
    for axis in (0, 1):
        std = float(scaled[:, axis].std())
        if not np.isfinite(std) or std < 1e-8:
            raise RuntimeError("coordinate synchronization collapsed")
        scaled[:, axis] = scaled[:, axis] * (IDEAL_STD / std) + (GRID - 1) / 2.0
    cells = np.stack(
        np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="xy"), axis=-1
    ).reshape(TILE_COUNT, 2)
    cost = np.square(scaled[:, None, :] - cells[None, :, :]).sum(axis=2)
    tile_indices, cell_indices = linear_sum_assignment(cost)
    layout = np.empty(TILE_COUNT, dtype=np.int32)
    layout[cell_indices] = tile_indices.astype(np.int32)
    layout = validate_permutation(layout, name="rts_ot_layout")
    return layout, {
        "selected_edges": count,
        "coordinate_std": coordinates.std(axis=0).tolist(),
        "scaled_min": scaled.min(axis=0).tolist(),
        "scaled_max": scaled.max(axis=0).tolist(),
        "residual_mean": float(residual.mean()),
        "residual_median": float(np.median(residual)),
        "final_weight_mean": float(weights.mean()),
        "assignment_cost_mean": float(cost[tile_indices, cell_indices].mean()),
    }


def score_layout(layout: np.ndarray, prepared: PreparedSource, clean: np.ndarray) -> dict[str, float]:
    geometry = layout_metrics(layout, prepared.truth)
    image = predicted_image_metrics(layout, prepared.denoised_tiles, clean)
    return {
        "ssim": float(image["predicted_layout_ssim"]),
        "adjacency": float(geometry["combined_adjacency"]),
        "largest_correct_component": float(geometry["largest_correct_component"]),
        "within_one_manhattan": float(geometry["within_one_manhattan"]),
    }


def summarize(records: list[dict], candidate: str) -> dict:
    panels = {}
    for panel in PANELS:
        selected = [record for record in records if record["panel"] == panel]
        ssim = np.asarray([record["candidates"][candidate]["ssim"] - record["baseline"]["ssim"] for record in selected])
        adjacency = np.asarray([record["candidates"][candidate]["adjacency"] - record["baseline"]["adjacency"] for record in selected])
        panels[panel] = {
            "mean_ssim_delta": float(ssim.mean()),
            "mean_adjacency_delta": float(adjacency.mean()),
            "ssim_wins": int(np.count_nonzero(ssim > 0)),
            "worst_ssim_delta": float(ssim.min()),
        }
    names = sorted({record["name"] for record in records})
    source_delta = np.asarray([
        np.mean([record["candidates"][candidate]["ssim"] - record["baseline"]["ssim"] for record in records if record["name"] == name])
        for name in names
    ])
    return {
        "panels": panels,
        "source_macro_mean_ssim_delta": float(source_delta.mean()),
        "source_macro_wins": int(np.count_nonzero(source_delta > 0)),
        "worst_panel_mean_ssim_delta": min(value["mean_ssim_delta"] for value in panels.values()),
        "worst_panel_mean_adjacency_delta": min(value["mean_adjacency_delta"] for value in panels.values()),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)["model"]
    report = json.loads(Path(args.tabular_report).read_text())
    threshold85 = float(report["calibration"]["frontiers"]["0.85"]["threshold"])
    restorer, device, _ = load_restorer(args.denoiser, device=args.device)
    embedding, _ = load_embedding_checkpoint(args.embedding_checkpoint, device=device)
    names = source_names_for_split(args.split, manifest_path=args.manifest, quarantine_path=args.quarantine)[args.source_offset : args.source_offset + args.sources]
    configs = [
        (f"k{top_k}_c{cutoff:g}_a{boost:g}", top_k, cutoff, boost)
        for top_k in (8, 16, 32)
        for cutoff in (0.75, 1.5)
        for boost in (0.5, 1.0, 2.0)
    ]
    records = []
    started = time.time()
    for source_index, name in enumerate(names):
        clean = read_rgb(Path(args.data_root) / "train/targets" / name)
        for panel in PANELS:
            panel_seed = per_source_seed(args.seed, f"rts-ot-{panel}", name, 0)
            prepared = prepare_source(name, panel, panel_seed, args=args, restorer=restorer, embedding_model=embedding, device=device)
            probability = model.predict_proba(prepared.features)[:, 1]
            qap_seed = per_source_seed(args.seed, f"rts-ot-qap-{panel}", name, 0)
            baseline = baseline_layout(prepared, qap_seed=qap_seed, iterations=args.baseline_iterations)
            record = {
                "name": name,
                "panel": panel,
                "baseline": score_layout(baseline, prepared, clean),
                "candidates": {},
                "diagnostics": {},
            }
            for config, top_k, cutoff, boost in configs:
                weights = initial_edge_weight(prepared, probability, threshold85, boost)
                selected = select_edges(prepared, weights, top_k)
                layout, diagnostics = solve_coordinates(
                    prepared, weights, selected, cutoff=cutoff,
                    rounds=args.irls_rounds, ridge=args.ridge,
                )
                record["candidates"][config] = score_layout(layout, prepared, clean)
                record["diagnostics"][config] = diagnostics
            records.append(record)
        print(json.dumps({"stage": "rts_ot", "done": source_index + 1, "total": len(names)}), flush=True)
    summaries = {name: summarize(records, name) for name, *_ in configs}
    eligible = [
        name for name, *_ in configs
        if summaries[name]["source_macro_mean_ssim_delta"] >= 0.003
        and summaries[name]["source_macro_wins"] >= 6
        and summaries[name]["worst_panel_mean_ssim_delta"] >= 0.001
        and summaries[name]["worst_panel_mean_adjacency_delta"] >= 0.02
        and all(panel["worst_ssim_delta"] >= -0.02 for panel in summaries[name]["panels"].values())
    ]
    by_name = {name: {"top_k": top_k, "cutoff": cutoff, "boost": boost} for name, top_k, cutoff, boost in configs}
    selected = min(
        eligible,
        key=lambda name: (
            -summaries[name]["worst_panel_mean_ssim_delta"],
            by_name[name]["top_k"], by_name[name]["boost"], -by_name[name]["cutoff"], name,
        ),
        default=None,
    )
    payload = {
        "schema_version": 1,
        "kind": "robust_translation_synchronization_ot_calibration",
        "split": args.split,
        "source_offset": args.source_offset,
        "source_names": names,
        "threshold85": threshold85,
        "configs": by_name,
        "records": records,
        "summaries": summaries,
        "gate": "source macro >=.003, both panels >=.001, >=6/8 source wins, adjacency >=.02, no record below -.02",
        "eligible": eligible,
        "selected": selected,
        "selected_summary": summaries.get(selected),
        "seconds": time.time() - started,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selected, "selected_summary": summaries.get(selected)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
