#!/usr/bin/env python3
"""Calibrate a sparse HGB trust-region repair on top of frozen qap_w4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, SCRIPT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.geometry import GRID, TILE_COUNT, validate_permutation
from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.solvers import placement_unary
from puzzle_denoise_v2.inference import load_restorer
from train_binary_edge_verifier import PreparedSource, prepare_source, read_rgb
from evaluate_hgb_component_sync import baseline_layout, make_residual


PANELS = ("primary_kornia", "independent_libjpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="runs/assembly_v1/full_union_tabular/v1/full_union_tabular.joblib",
    )
    parser.add_argument(
        "--tabular-report",
        default="runs/assembly_v1/full_union_tabular/v1/report.json",
    )
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument(
        "--denoiser", default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
    )
    parser.add_argument(
        "--embedding-checkpoint",
        default="runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt",
    )
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument(
        "--quarantine", default="configs/denoise_validation_quarantine_v1.json"
    )
    parser.add_argument("--split", default="assembly_cal")
    parser.add_argument("--source-offset", type=int, default=0)
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--baseline-iterations", type=int, default=25)
    parser.add_argument("--trust-radius", type=float, default=0.002)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def realized_mask(layout: np.ndarray, direction: np.ndarray, source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    inverse = np.empty(TILE_COUNT, dtype=np.int32)
    inverse[layout] = np.arange(TILE_COUNT, dtype=np.int32)
    first = inverse[source]
    second = inverse[destination]
    horizontal = direction == 0
    expected = np.where(horizontal, first + 1, first + GRID)
    valid = np.where(horizontal, (first // GRID) == (second // GRID), True)
    return valid & (second == expected)


def w4_energy(layout: np.ndarray, prepared: PreparedSource, unary: np.ndarray) -> float:
    grid = np.asarray(layout, dtype=np.int32).reshape(GRID, GRID)
    value = prepared.scores["w4"].right[grid[:, :-1], grid[:, 1:]].sum(dtype=np.float64)
    value += prepared.scores["w4"].down[grid[:-1, :], grid[1:, :]].sum(dtype=np.float64)
    positions = np.arange(TILE_COUNT, dtype=np.int32)
    value += 0.05 * unary[positions, layout].sum(dtype=np.float64)
    return float(value / (2 * GRID * (GRID - 1)))


def hgb_bonus(
    layout: np.ndarray,
    probability: np.ndarray,
    prepared: PreparedSource,
    threshold: float,
) -> tuple[float, int]:
    chosen = probability >= threshold
    if not np.any(chosen):
        return 0.0, 0
    realized = realized_mask(
        layout,
        prepared.graph.direction[chosen],
        prepared.graph.source[chosen],
        prepared.graph.destination[chosen],
    )
    scaled = (probability[chosen] - threshold) / max(1.0 - threshold, 1e-12)
    return float(scaled[realized].sum(dtype=np.float64) / 1104.0), int(realized.sum())


def proposed_swaps(
    layout: np.ndarray,
    probability: np.ndarray,
    prepared: PreparedSource,
    threshold: float,
) -> list[tuple[int, int, int, int, int]]:
    inverse = np.empty(TILE_COUNT, dtype=np.int32)
    inverse[layout] = np.arange(TILE_COUNT, dtype=np.int32)
    candidates = np.flatnonzero(probability >= threshold)
    candidates = candidates[np.argsort(-probability[candidates], kind="stable")]
    output: set[tuple[int, int, int, int, int]] = set()
    for index in candidates.tolist():
        direction = int(prepared.graph.direction[index])
        source = int(prepared.graph.source[index])
        destination = int(prepared.graph.destination[index])
        source_pos = int(inverse[source])
        destination_pos = int(inverse[destination])
        if realized_mask(
            layout,
            np.asarray([direction]),
            np.asarray([source]),
            np.asarray([destination]),
        )[0]:
            continue
        if direction == 0:
            after = source_pos + 1 if source_pos % GRID < GRID - 1 else None
            before = destination_pos - 1 if destination_pos % GRID > 0 else None
        else:
            after = source_pos + GRID if source_pos // GRID < GRID - 1 else None
            before = destination_pos - GRID if destination_pos // GRID > 0 else None
        if after is not None and after != destination_pos:
            output.add((destination_pos, int(after), source, destination, index))
        if before is not None and before != source_pos:
            output.add((source_pos, int(before), source, destination, index))
    return sorted(output, key=lambda value: (value[2], value[3], value[0], value[1], value[4]))


def repair_layout(
    baseline: np.ndarray,
    probability: np.ndarray,
    prepared: PreparedSource,
    *,
    threshold: float,
    protected_threshold: float,
    weight: float,
    budget: int,
    trust_radius: float,
) -> tuple[np.ndarray, dict]:
    layout = validate_permutation(baseline, name="baseline").copy()
    unary = placement_unary(prepared.scores["w4"])
    baseline_energy = w4_energy(layout, prepared, unary)
    bonus, realized = hgb_bonus(layout, probability, prepared, threshold)
    protected_bonus, protected_realized = hgb_bonus(
        layout, probability, prepared, protected_threshold
    )
    accepted = []
    for _ in range(budget):
        current_objective = -w4_energy(layout, prepared, unary) + weight * bonus
        best = None
        for first, second, source, destination, edge_index in proposed_swaps(
            layout, probability, prepared, threshold
        ):
            candidate = layout.copy()
            candidate[first], candidate[second] = candidate[second], candidate[first]
            energy = w4_energy(candidate, prepared, unary)
            if energy - baseline_energy > trust_radius + 1e-12:
                continue
            candidate_bonus, candidate_realized = hgb_bonus(
                candidate, probability, prepared, threshold
            )
            if candidate_bonus <= bonus + 1e-12:
                continue
            candidate_protected_bonus, candidate_protected_realized = hgb_bonus(
                candidate, probability, prepared, protected_threshold
            )
            if candidate_protected_realized < protected_realized:
                continue
            objective = -energy + weight * candidate_bonus
            if objective <= current_objective + 1e-12:
                continue
            key = (
                -objective,
                energy,
                -candidate_bonus,
                source,
                destination,
                first,
                second,
                edge_index,
            )
            if best is None or key < best[0]:
                best = (
                    key,
                    candidate,
                    energy,
                    candidate_bonus,
                    candidate_realized,
                    candidate_protected_bonus,
                    candidate_protected_realized,
                    (first, second, source, destination, edge_index),
                )
        if best is None:
            break
        (
            _, layout, energy, bonus, realized,
            protected_bonus, protected_realized, move,
        ) = best
        accepted.append(
            {
                "positions": [int(move[0]), int(move[1])],
                "edge": [int(move[2]), int(move[3])],
                "edge_index": int(move[4]),
                "energy": float(energy),
                "bonus": float(bonus),
            }
        )
    final_energy = w4_energy(layout, prepared, unary)
    return layout, {
        "swaps": len(accepted),
        "accepted": accepted,
        "baseline_energy": baseline_energy,
        "final_energy": final_energy,
        "energy_delta": final_energy - baseline_energy,
        "final_bonus": bonus,
        "final_realized": realized,
        "final_protected_bonus": protected_bonus,
        "final_protected_realized": protected_realized,
        "changed_tiles": int(np.count_nonzero(layout != baseline)),
    }


def score_layout(layout: np.ndarray, prepared: PreparedSource, clean: np.ndarray) -> dict[str, float]:
    geometry = layout_metrics(layout, prepared.truth)
    image = predicted_image_metrics(layout, prepared.denoised_tiles, clean)
    return {
        "adjacency": float(geometry["combined_adjacency"]),
        "ssim": float(image["predicted_layout_ssim"]),
    }


def summarize(records: list[dict], candidate: str) -> dict:
    panels = {}
    for panel in PANELS:
        selected = [record for record in records if record["panel"] == panel]
        ssim_delta = np.asarray([
            record["candidates"][candidate]["ssim"] - record["baseline"]["ssim"]
            for record in selected
        ])
        adjacency_delta = np.asarray([
            record["candidates"][candidate]["adjacency"] - record["baseline"]["adjacency"]
            for record in selected
        ])
        panels[panel] = {
            "mean_ssim_delta": float(ssim_delta.mean()),
            "mean_adjacency_delta": float(adjacency_delta.mean()),
            "ssim_wins": int(np.count_nonzero(ssim_delta > 0)),
            "worst_ssim_delta": float(ssim_delta.min()),
        }
    source_names = sorted({record["name"] for record in records})
    source_delta = []
    for name in source_names:
        source_delta.append(float(np.mean([
            record["candidates"][candidate]["ssim"] - record["baseline"]["ssim"]
            for record in records if record["name"] == name
        ])))
    values = np.asarray(source_delta)
    return {
        "panels": panels,
        "source_macro_mean_ssim_delta": float(values.mean()),
        "source_macro_wins": int(np.count_nonzero(values > 0)),
        "source_regressions_below_minus_0_005": int(np.count_nonzero(values < -0.005)),
        "worst_panel_mean_ssim_delta": min(
            panel["mean_ssim_delta"] for panel in panels.values()
        ),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)["model"]
    report = json.loads(Path(args.tabular_report).read_text())
    thresholds = {
        int(target * 100): float(report["calibration"]["frontiers"][str(target)]["threshold"])
        for target in (0.75, 0.80, 0.85)
    }
    variants = [
        (f"p{target}_l{weight:g}_b{budget}", threshold, weight, budget)
        for target, threshold in thresholds.items()
        for weight in (0.5, 1.0, 2.0)
        for budget in (1, 2, 4)
    ]
    restorer, device, _ = load_restorer(args.denoiser, device=args.device)
    embedding, _ = load_embedding_checkpoint(args.embedding_checkpoint, device=device)
    source_names = source_names_for_split(
        args.split, manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.source_offset : args.source_offset + args.sources]
    records = []
    started = time.time()
    for source_index, name in enumerate(source_names):
        clean = read_rgb(Path(args.data_root) / "train/targets" / name)
        for panel in PANELS:
            panel_seed = per_source_seed(args.seed, f"hgb-trust-repair-{panel}", name, 0)
            prepared = prepare_source(
                name, panel, panel_seed, args=args, restorer=restorer,
                embedding_model=embedding, device=device,
            )
            probability, _ = make_residual(model, prepared, hgb_weight=0.25)
            qap_seed = per_source_seed(args.seed, f"hgb-trust-repair-qap-{panel}", name, 0)
            baseline = baseline_layout(
                prepared, qap_seed=qap_seed, iterations=args.baseline_iterations
            )
            record = {
                "name": name,
                "panel": panel,
                "baseline": score_layout(baseline, prepared, clean),
                "candidates": {},
                "diagnostics": {},
            }
            for variant, threshold, weight, budget in variants:
                layout, diagnostics = repair_layout(
                    baseline, probability, prepared,
                    threshold=threshold,
                    protected_threshold=thresholds[85],
                    weight=weight,
                    budget=budget,
                    trust_radius=args.trust_radius,
                )
                record["candidates"][variant] = score_layout(layout, prepared, clean)
                record["diagnostics"][variant] = diagnostics
            records.append(record)
        print(json.dumps({"stage": "calibration", "done": source_index + 1, "total": len(source_names)}), flush=True)
    summaries = {variant: summarize(records, variant) for variant, *_ in variants}
    eligible = [
        variant for variant, *_ in variants
        if summaries[variant]["source_macro_mean_ssim_delta"] >= 0.0005
        and summaries[variant]["source_macro_wins"] >= 9
        and summaries[variant]["source_regressions_below_minus_0_005"] <= 1
        and all(
            panel["mean_ssim_delta"] >= 0.0
            for panel in summaries[variant]["panels"].values()
        )
    ]
    config_by_name = {
        name: {"threshold": threshold, "weight": weight, "budget": budget}
        for name, threshold, weight, budget in variants
    }
    selected = min(
        eligible,
        key=lambda name: (
            -summaries[name]["worst_panel_mean_ssim_delta"],
            config_by_name[name]["budget"],
            -config_by_name[name]["threshold"],
            config_by_name[name]["weight"],
            name,
        ),
        default=None,
    )
    payload = {
        "schema_version": 1,
        "kind": "hgb_trust_region_repair_calibration",
        "split": args.split,
        "source_offset": args.source_offset,
        "source_names": source_names,
        "panels": list(PANELS),
        "thresholds": thresholds,
        "trust_radius": args.trust_radius,
        "variants": config_by_name,
        "records": records,
        "summaries": summaries,
        "selection_rule": "eligible source-macro>=.0005, both panels>=0, wins>=9/16, <=1 regression below -.005; maximize worst panel",
        "eligible": eligible,
        "selected": selected,
        "selected_summary": summaries.get(selected),
        "seconds": time.time() - started,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selected, "selected_summary": summaries.get(selected)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
