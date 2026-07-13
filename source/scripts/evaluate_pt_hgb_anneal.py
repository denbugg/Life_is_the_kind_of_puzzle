#!/usr/bin/env python3
"""Screen parallel-tempered HGB-rewarded annealing from frozen qap_w4."""

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

from puzzle_assembly.anneal_refine import (
    _calibrate_temperature,
    _choose_move,
    _edge_sum,
    _incremental_delta,
    _local_costs,
    _polish_swaps,
)
from puzzle_assembly.geometry import TILE_COUNT, validate_permutation
from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.solvers import placement_unary
from puzzle_denoise_v2.inference import load_restorer
from train_binary_edge_verifier import PreparedSource, prepare_source, read_rgb
from evaluate_hgb_component_sync import baseline_layout


PANELS = ("primary_kornia", "independent_libjpeg")
MOVE_NAMES = ("swap", "segment", "block", "band")
MOVE_CUMULATIVE = np.cumsum(np.asarray([0.45, 0.20, 0.25, 0.10]))
BLOCK_SHAPES = ((1, 2), (2, 1), (1, 4), (4, 1), (2, 2), (4, 4))


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
    parser.add_argument("--source-offset", type=int, default=56)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--baseline-iterations", type=int, default=25)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--exchange-interval", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reward_matrices(
    prepared: PreparedSource,
    probability: np.ndarray,
    *,
    threshold: float,
    clip_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    threshold_logit = np.log(threshold) - np.log1p(-threshold)
    values = np.clip(probability, 1e-6, 1.0 - 1e-6)
    llr = np.log(values) - np.log1p(-values) - threshold_logit
    reward = np.clip(np.maximum(llr, 0.0), 0.0, clip_value)
    right = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float64)
    down = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float64)
    for direction in (0, 1):
        indices = np.flatnonzero(prepared.graph.direction == direction)
        matrix = right if direction == 0 else down
        for index in indices.tolist():
            first = int(prepared.graph.source[index])
            second = int(prepared.graph.destination[index])
            matrix[first, second] = max(matrix[first, second], float(reward[index]))
    return right, down


def augmented_arrays(
    prepared: PreparedSource,
    probability: np.ndarray,
    *,
    threshold: float,
    alpha: float,
    clip_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reward_right, reward_down = reward_matrices(
        prepared, probability, threshold=threshold, clip_value=clip_value
    )
    right = np.asarray(prepared.scores["w4"].right, dtype=np.float64).copy()
    down = np.asarray(prepared.scores["w4"].down, dtype=np.float64).copy()
    right -= alpha * reward_right
    down -= alpha * reward_down
    unary = placement_unary(prepared.scores["w4"]).astype(np.float64, copy=False)
    return right, down, unary


def full_energy(layout: np.ndarray, right: np.ndarray, down: np.ndarray, unary: np.ndarray) -> float:
    return float(
        _edge_sum(layout, right, down)
        + 0.05 * unary[np.arange(TILE_COUNT, dtype=np.int32), layout].sum(dtype=np.float64)
    )


def parallel_tempered(
    initial: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    unary: np.ndarray,
    *,
    seed: int,
    replicas: int,
    steps: int,
    exchange_interval: int,
) -> tuple[np.ndarray, dict]:
    initial = validate_permutation(initial, name="pt_initial").copy()
    calibration_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xCA1]))
    start_temperature = _calibrate_temperature(
        initial,
        calibration_rng,
        right=right,
        down=down,
        unary=unary,
        boundary_weight=0.05,
        edge_scale=1.0,
        samples=128,
    )
    temperatures = start_temperature * np.geomspace(0.02, 4.0, replicas)
    rngs = [np.random.default_rng(np.random.SeedSequence([seed, index, 0xA11E])) for index in range(replicas)]
    layouts = [initial.copy() for _ in range(replicas)]
    all_positions = np.arange(TILE_COUNT, dtype=np.int32)
    for replica in range(1, replicas):
        perturbations = min(64, 2 ** replica)
        for _ in range(perturbations):
            move = _choose_move(
                layouts[replica], rngs[replica], all_positions,
                weak_bias=0.0, max_segment=16, block_shapes=BLOCK_SHAPES,
                move_names=MOVE_NAMES, cumulative_weights=MOVE_CUMULATIVE,
            )
            layouts[replica] = move.candidate
    energies = np.asarray([full_energy(layout, right, down, unary) for layout in layouts])
    best = initial.copy()
    best_energy = full_energy(initial, right, down, unary)
    accepted = np.zeros(replicas, dtype=np.int64)
    proposed = np.zeros(replicas, dtype=np.int64)
    exchanges = 0
    weak_pools = [all_positions[:64] for _ in range(replicas)]
    for step in range(steps):
        for replica in range(replicas):
            if step % 64 == 0:
                local = _local_costs(
                    layouts[replica], right=right, down=down, unary=unary,
                    boundary_weight=0.05,
                )
                weak_pools[replica] = np.argsort(-local, kind="stable")[:64]
            move = _choose_move(
                layouts[replica], rngs[replica], weak_pools[replica],
                weak_bias=0.7, max_segment=16, block_shapes=BLOCK_SHAPES,
                move_names=MOVE_NAMES, cumulative_weights=MOVE_CUMULATIVE,
            )
            delta, _ = _incremental_delta(
                layouts[replica], move.candidate, move.changed_positions,
                augmented_right=right, augmented_down=down, unary=unary,
                boundary_weight=0.05,
            )
            proposed[replica] += 1
            if delta <= 0.0 or rngs[replica].random() < np.exp(-delta / max(temperatures[replica], 1e-12)):
                layouts[replica] = move.candidate
                energies[replica] += delta
                accepted[replica] += 1
                if energies[replica] < best_energy - 1e-10:
                    best = layouts[replica].copy()
                    best_energy = float(energies[replica])
        if (step + 1) % exchange_interval == 0:
            parity = ((step + 1) // exchange_interval) % 2
            for first in range(parity, replicas - 1, 2):
                second = first + 1
                log_accept = (
                    (1.0 / temperatures[first] - 1.0 / temperatures[second])
                    * (energies[first] - energies[second])
                )
                if log_accept >= 0.0 or rngs[first].random() < np.exp(log_accept):
                    layouts[first], layouts[second] = layouts[second], layouts[first]
                    energies[first], energies[second] = energies[second], energies[first]
                    exchanges += 1
        if (step + 1) % 2048 == 0:
            for replica in range(replicas):
                audited = full_energy(layouts[replica], right, down, unary)
                if abs(audited - energies[replica]) > 1e-7 * max(1.0, abs(audited)):
                    raise RuntimeError("parallel-tempering incremental energy drift")
                energies[replica] = audited
    polished, polished_moves = _polish_swaps(
        best, right=right, down=down, unary=unary, boundary_weight=0.05,
        weak_cells=48, moves=8,
    )
    polished_energy = full_energy(polished, right, down, unary)
    if polished_energy <= best_energy + 1e-8:
        best, best_energy = polished, polished_energy
    return validate_permutation(best, name="pt_layout"), {
        "initial_energy": full_energy(initial, right, down, unary),
        "best_energy": float(best_energy),
        "energy_delta": float(best_energy - full_energy(initial, right, down, unary)),
        "temperatures": temperatures.tolist(),
        "accepted": accepted.tolist(),
        "proposed": proposed.tolist(),
        "exchange_accepts": exchanges,
        "polished_swaps": polished_moves,
        "changed_tiles": int(np.count_nonzero(best != initial)),
    }


def score_layout(layout: np.ndarray, prepared: PreparedSource, clean: np.ndarray) -> dict[str, float]:
    geometry = layout_metrics(layout, prepared.truth)
    image = predicted_image_metrics(layout, prepared.denoised_tiles, clean)
    return {
        "ssim": float(image["predicted_layout_ssim"]),
        "adjacency": float(geometry["combined_adjacency"]),
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
    tabular = json.loads(Path(args.tabular_report).read_text())
    threshold85 = float(tabular["calibration"]["frontiers"]["0.85"]["threshold"])
    restorer, device, _ = load_restorer(args.denoiser, device=args.device)
    embedding, _ = load_embedding_checkpoint(args.embedding_checkpoint, device=device)
    names = source_names_for_split(args.split, manifest_path=args.manifest, quarantine_path=args.quarantine)[args.source_offset : args.source_offset + args.sources]
    configs = [(f"a{alpha:g}_l{clip_value:g}", alpha, clip_value) for alpha in (0.1, 0.25, 0.5) for clip_value in (0.5, 1.0)]
    records = []
    started = time.time()
    for source_index, name in enumerate(names):
        clean = read_rgb(Path(args.data_root) / "train/targets" / name)
        for panel in PANELS:
            panel_seed = per_source_seed(args.seed, f"pt-hgb-{panel}", name, 0)
            prepared = prepare_source(name, panel, panel_seed, args=args, restorer=restorer, embedding_model=embedding, device=device)
            probability = model.predict_proba(prepared.features)[:, 1]
            qap_seed = per_source_seed(args.seed, f"pt-hgb-qap-{panel}", name, 0)
            baseline = baseline_layout(prepared, qap_seed=qap_seed, iterations=args.baseline_iterations)
            record = {"name": name, "panel": panel, "baseline": score_layout(baseline, prepared, clean), "candidates": {}, "diagnostics": {}}
            for config, alpha, clip_value in configs:
                right, down, unary = augmented_arrays(
                    prepared, probability, threshold=threshold85,
                    alpha=alpha, clip_value=clip_value,
                )
                anneal_seed = per_source_seed(args.seed, f"pt-hgb-{config}-{panel}", name, 0)
                layout, diagnostics = parallel_tempered(
                    baseline, right, down, unary, seed=anneal_seed,
                    replicas=args.replicas, steps=args.steps,
                    exchange_interval=args.exchange_interval,
                )
                record["candidates"][config] = score_layout(layout, prepared, clean)
                record["diagnostics"][config] = diagnostics
            records.append(record)
        print(json.dumps({"stage": "pt_hgb", "done": source_index + 1, "total": len(names)}), flush=True)
    summaries = {name: summarize(records, name) for name, *_ in configs}
    eligible = [
        name for name, *_ in configs
        if summaries[name]["source_macro_mean_ssim_delta"] >= 0.001
        and summaries[name]["source_macro_wins"] >= 3
        and summaries[name]["worst_panel_mean_ssim_delta"] >= 0.0
        and summaries[name]["worst_panel_mean_adjacency_delta"] >= 0.0
    ]
    selected = min(eligible, key=lambda name: (-summaries[name]["worst_panel_mean_ssim_delta"], name), default=None)
    payload = {
        "schema_version": 1,
        "kind": "parallel_tempered_hgb_anneal_screen",
        "split": args.split,
        "source_offset": args.source_offset,
        "source_names": names,
        "threshold85": threshold85,
        "configs": {name: {"alpha": alpha, "clip": clip_value} for name, alpha, clip_value in configs},
        "records": records,
        "summaries": summaries,
        "gate": "4-source screen: macro>=.001, both panels and adjacency nonnegative, >=3/4 source wins",
        "eligible": eligible,
        "selected": selected,
        "selected_summary": summaries.get(selected),
        "seconds": time.time() - started,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selected, "selected_summary": summaries.get(selected)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
