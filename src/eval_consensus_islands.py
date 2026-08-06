"""Held-out gate for label-free growing-consensus jigsaw islands."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from candidate_rank import neighbor_targets, score_candidate_rows
from canvas_data import CanvasDataset
from config import SEED, WORK_ROOT
from consensus_islands import (
    ConsensusAssembler,
    edge_metrics,
    graph_from_scores,
    island_metrics,
    select_edges,
)
from eval_test_time_adaptation import _all_rows, _load_ranker
from imgio import train_val_split
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


def main() -> None:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranker",
        default=str(workspace / "artifacts/candidate_rank/rank_v2w64_best.pt"),
    )
    parser.add_argument(
        "--affinity-ckpt",
        default=str(workspace / "artifacts/macro_affinity/affinity_r1_1200_best.pt"),
    )
    parser.add_argument(
        "--affinity-ckpt2",
        default=str(workspace / "artifacts/macro_affinity/affinity_r3_1000_best.pt"),
    )
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--pair-batch", type=int, default=4096)
    parser.add_argument("--quantiles", default="0.50,0.70,0.85,0.93")
    parser.add_argument("--max-directed-edges", type=int, default=512)
    parser.add_argument(
        "--dual-consensus",
        action="store_true",
        help="require independent primary/secondary affinity candidate graphs to agree",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "consensus_islands_gate.json",
    )
    args = parser.parse_args()
    quantiles = [float(value) for value in args.quantiles.split(",")]
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ranker = _load_ranker(args.ranker, device)
    affinity, _, _ = load_frozen_affinity(args.affinity_ckpt, device)
    secondary = None
    if args.affinity_ckpt2:
        secondary, _, _ = load_frozen_affinity(args.affinity_ckpt2, device)
    _, validation_names = train_val_split()
    dataset = CanvasDataset(
        validation_names[: args.images],
        real_prob=0.0,
        seed=args.seed + 90_000,
    )
    rows_by_quantile: dict[str, list[dict[str, float]]] = {
        str(value): [] for value in quantiles
    }
    raw_graph_rows: list[dict[str, float]] = []

    for image_index in range(args.images):
        sample = dataset[image_index]
        tiles = sample["tiles"].unsqueeze(0).to(device)
        permutation = sample["perm"].numpy()
        perm_device = sample["perm"].unsqueeze(0).to(device).long()
        candidates, valid = mine_affinity_candidates(
            affinity,
            tiles,
            candidate_k=args.candidate_k,
            device=device,
            affinity_secondary=None if args.dual_consensus else secondary,
        )
        all_rows = _all_rows(candidates, valid)
        with torch.inference_mode():
            scores = score_candidate_rows(
                ranker,
                tiles,
                candidates,
                valid,
                all_rows,
                pair_batch=args.pair_batch,
            )
            graph = graph_from_scores(candidates, valid, scores)
            if args.dual_consensus:
                if secondary is None:
                    raise ValueError("--dual-consensus requires --affinity-ckpt2")
                candidates2, valid2 = mine_affinity_candidates(
                    secondary,
                    tiles,
                    candidate_k=args.candidate_k,
                    device=device,
                    affinity_secondary=None,
                )
                rows2 = _all_rows(candidates2, valid2)
                scores2 = score_candidate_rows(
                    ranker,
                    tiles,
                    candidates2,
                    valid2,
                    rows2,
                    pair_batch=args.pair_batch,
                )
                graph2 = graph_from_scores(candidates2, valid2, scores2)
                agrees = graph.predicted.eq(graph2.predicted)
                # Both directed decisions and both independent reverse decisions
                # must agree. The weaker of the two margins controls selection.
                graph = type(graph)(
                    predicted=graph.predicted,
                    margins=torch.minimum(graph.margins, graph2.margins),
                    mutual=graph.mutual & graph2.mutual & agrees,
                    loop=graph.loop & graph2.loop & agrees,
                )
            targets, _ = neighbor_targets(perm_device)
        raw_graph_rows.append(
            {
                "mutual_directed": float(graph.mutual.sum()),
                "loop_directed": float(graph.loop.sum()),
            }
        )
        for quantile in quantiles:
            edges = select_edges(
                graph,
                confidence_quantile=quantile,
                max_directed_edges=args.max_directed_edges,
            )
            assembler = ConsensusAssembler(len(permutation))
            assembler.add_edges(edges)
            metrics = {
                **edge_metrics(edges, targets),
                **island_metrics(
                    assembler,
                    permutation=permutation,
                    grid_side=24,
                ),
                "image_index": float(image_index),
                "confidence_threshold": edges.threshold,
            }
            rows_by_quantile[str(quantile)].append(metrics)
            print(json.dumps({"image": image_index, "quantile": quantile, **metrics}), flush=True)

    summary: dict[str, dict[str, float]] = {}
    for quantile, rows in rows_by_quantile.items():
        keys = [key for key in rows[0] if key not in ("image_index", "confidence_threshold")]
        summary[quantile] = {
            key: float(np.mean([row[key] for row in rows])) for key in keys
        }
    # Useful islands must be both reliable and materially larger than pairs.
    thresholds = {
        "exact_edge_precision": 0.90,
        "pure_nontrivial_tile_coverage": 0.15,
        "largest_pure_component": 8.0,
    }
    # Prefer operating points satisfying the most safety/utility checks; only
    # then maximize coverage. This prevents an impure low-threshold graph from
    # hiding a smaller but actionable high-precision consensus regime.
    best_key = max(
        summary,
        key=lambda key: (
            sum(summary[key][metric] >= threshold for metric, threshold in thresholds.items()),
            summary[key]["exact_edge_precision"] >= thresholds["exact_edge_precision"],
            summary[key]["pure_nontrivial_tile_coverage"],
            summary[key]["largest_pure_component"],
        ),
    )
    best = summary[best_key]
    checks = {key: best[key] >= value for key, value in thresholds.items()}
    report = {
        "experiment": "label_free_consensus_islands",
        "status": "pass" if all(checks.values()) else "fail",
        "best_quantile": best_key,
        "best": best,
        "thresholds": thresholds,
        "checks": checks,
        "summary": summary,
        "raw_graph_mean": {
            key: float(np.mean([row[key] for row in raw_graph_rows]))
            for key in raw_graph_rows[0]
        },
        "images": args.images,
        "dual_consensus": args.dual_consensus,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate": report, "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
