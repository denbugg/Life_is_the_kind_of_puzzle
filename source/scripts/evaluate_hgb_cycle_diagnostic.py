#!/usr/bin/env python3
"""Test whether soft 2x2-cycle support improves frozen HGB edge ranking."""

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

from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer
from train_binary_edge_verifier import PreparedSource, binary_metrics, prepare_source


PANELS = ("primary_kornia", "independent_libjpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/assembly_v1/full_union_tabular/v1/full_union_tabular.joblib")
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt")
    parser.add_argument("--embedding-checkpoint", default="runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt")
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--quarantine", default="configs/denoise_validation_quarantine_v1.json")
    parser.add_argument("--split", default="assembly_cal")
    parser.add_argument("--source-offset", type=int, default=16)
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(values) - np.log1p(-values)


def adjacency_lists(prepared: PreparedSource, score: np.ndarray, top_k: int):
    graph = prepared.graph
    outgoing = [[[] for _ in range(576)] for _ in range(2)]
    incoming = [[[] for _ in range(576)] for _ in range(2)]
    lookup = [{}, {}]
    for index in range(len(score)):
        direction = int(graph.direction[index])
        source = int(graph.source[index])
        destination = int(graph.destination[index])
        value = float(score[index])
        key = (source, destination)
        if key not in lookup[direction] or value > lookup[direction][key]:
            lookup[direction][key] = value
    for direction in (0, 1):
        for (source, destination), value in lookup[direction].items():
            outgoing[direction][source].append((destination, value))
            incoming[direction][destination].append((source, value))
        for tile in range(576):
            outgoing[direction][tile] = sorted(
                outgoing[direction][tile], key=lambda item: (-item[1], item[0])
            )[:top_k]
            incoming[direction][tile] = sorted(
                incoming[direction][tile], key=lambda item: (-item[1], item[0])
            )[:top_k]
    return outgoing, incoming, lookup


def best_square_support(
    first_side: list[tuple[int, float]],
    second_side: list[tuple[int, float]],
    closing: dict[tuple[int, int], float],
) -> tuple[float, int]:
    best = 0.0
    count = 0
    for first, first_score in first_side:
        for second, second_score in second_side:
            closing_score = closing.get((first, second))
            if closing_score is None:
                continue
            count += 1
            best = max(best, min(first_score, second_score, closing_score))
    return best, count


def cycle_support(prepared: PreparedSource, probability: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    outgoing, incoming, lookup = adjacency_lists(prepared, probability, top_k)
    support = np.zeros(len(probability), dtype=np.float32)
    counts = np.zeros(len(probability), dtype=np.int16)
    graph = prepared.graph
    for index in range(len(probability)):
        direction = int(graph.direction[index])
        source = int(graph.source[index])
        destination = int(graph.destination[index])
        if direction == 0:
            forward = best_square_support(
                outgoing[1][source], outgoing[1][destination], lookup[0]
            )
            reverse = best_square_support(
                incoming[1][source], incoming[1][destination], lookup[0]
            )
        else:
            forward = best_square_support(
                outgoing[0][source], outgoing[0][destination], lookup[1]
            )
            reverse = best_square_support(
                incoming[0][source], incoming[0][destination], lookup[1]
            )
        support[index] = max(forward[0], reverse[0])
        counts[index] = min(forward[1] + reverse[1], np.iinfo(np.int16).max)
    return support, counts


def retrieval_metrics(prepared: PreparedSource, score: np.ndarray) -> dict[str, float]:
    graph = prepared.graph
    hits = {1: 0, 5: 0, 32: 0}
    reciprocal_rank = 0.0
    groups = 0
    for direction in (0, 1):
        for source in range(576):
            indices = np.flatnonzero((graph.direction == direction) & (graph.source == source))
            if len(indices) == 0:
                continue
            order = indices[np.argsort(-score[indices], kind="stable")]
            positive = np.flatnonzero(prepared.labels[order] > 0.5)
            rank = int(positive[0]) + 1 if len(positive) else None
            groups += 1
            if rank is not None:
                reciprocal_rank += 1.0 / rank
                for cutoff in hits:
                    hits[cutoff] += int(rank <= cutoff)
    return {
        "groups": groups,
        "r1": hits[1] / groups,
        "r5": hits[5] / groups,
        "r32": hits[32] / groups,
        "mrr": reciprocal_rank / groups,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)["model"]
    restorer, device, _ = load_restorer(args.denoiser, device=args.device)
    embedding, _ = load_embedding_checkpoint(args.embedding_checkpoint, device=device)
    source_names = source_names_for_split(
        args.split, manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.source_offset : args.source_offset + args.sources]
    configs = [(f"k{top_k}_l{weight:g}", top_k, weight) for top_k in (8, 16) for weight in (1.0, 2.0, 4.0)]
    records = []
    started = time.time()
    for source_index, name in enumerate(source_names):
        for panel in PANELS:
            panel_seed = per_source_seed(args.seed, f"hgb-cycle-diagnostic-{panel}", name, 0)
            prepared = prepare_source(
                name, panel, panel_seed, args=args, restorer=restorer,
                embedding_model=embedding, device=device,
            )
            probability = model.predict_proba(prepared.features)[:, 1]
            record = {
                "name": name,
                "panel": panel,
                "base": {
                    "binary": binary_metrics(prepared.labels, probability),
                    "retrieval": retrieval_metrics(prepared, probability),
                },
                "configs": {},
            }
            supports = {}
            for top_k in (8, 16):
                support, counts = cycle_support(prepared, probability, top_k)
                supports[top_k] = support
                positive = prepared.labels > 0.5
                record[f"support_k{top_k}"] = {
                    "positive_mean": float(support[positive].mean()),
                    "negative_mean": float(support[~positive].mean()),
                    "positive_nonzero": float(np.mean(support[positive] > 0)),
                    "negative_nonzero": float(np.mean(support[~positive] > 0)),
                    "positive_cycle_count_mean": float(counts[positive].mean()),
                    "negative_cycle_count_mean": float(counts[~positive].mean()),
                }
            base_logit = clipped_logit(probability)
            for config, top_k, weight in configs:
                score = base_logit + weight * supports[top_k]
                record["configs"][config] = {
                    "binary": binary_metrics(prepared.labels, score),
                    "retrieval": retrieval_metrics(prepared, score),
                }
            records.append(record)
        print(json.dumps({"stage": "cycle_diagnostic", "done": source_index + 1, "total": len(source_names)}), flush=True)
    summaries = {}
    for config, *_ in configs:
        panels = {}
        for panel in PANELS:
            selected = [record for record in records if record["panel"] == panel]
            ap_delta = np.asarray([
                record["configs"][config]["binary"]["average_precision"] - record["base"]["binary"]["average_precision"]
                for record in selected
            ])
            r1_delta = np.asarray([
                record["configs"][config]["retrieval"]["r1"] - record["base"]["retrieval"]["r1"]
                for record in selected
            ])
            mrr_delta = np.asarray([
                record["configs"][config]["retrieval"]["mrr"] - record["base"]["retrieval"]["mrr"]
                for record in selected
            ])
            panels[panel] = {
                "mean_ap_delta": float(ap_delta.mean()),
                "mean_r1_delta": float(r1_delta.mean()),
                "mean_mrr_delta": float(mrr_delta.mean()),
                "ap_wins": int(np.count_nonzero(ap_delta > 0)),
            }
        summaries[config] = {
            "panels": panels,
            "worst_panel_ap_delta": min(value["mean_ap_delta"] for value in panels.values()),
            "worst_panel_r1_delta": min(value["mean_r1_delta"] for value in panels.values()),
        }
    eligible = [
        config for config, *_ in configs
        if summaries[config]["worst_panel_ap_delta"] >= 0.005
        and summaries[config]["worst_panel_r1_delta"] >= 0.0
    ]
    selected = min(
        eligible,
        key=lambda config: (-summaries[config]["worst_panel_ap_delta"], config),
        default=None,
    )
    payload = {
        "schema_version": 1,
        "kind": "hgb_soft_4cycle_edge_diagnostic",
        "split": args.split,
        "source_offset": args.source_offset,
        "source_names": source_names,
        "records": records,
        "configs": {name: {"top_k": top_k, "weight": weight} for name, top_k, weight in configs},
        "summaries": summaries,
        "gate": "both panels AP delta >= .005 and R1 delta >= 0",
        "eligible": eligible,
        "selected": selected,
        "selected_summary": summaries.get(selected),
        "seconds": time.time() - started,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selected, "selected_summary": summaries.get(selected)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
