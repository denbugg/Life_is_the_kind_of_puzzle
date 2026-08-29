"""Select one board from several frozen seed basins using the solver objective."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import solver_v31 as s


def seed_candidates(scene, matrices, heads, unary_weight, device, seed):
    right, down = matrices
    unary = s.v30.unary_from_heads(heads, matrices, device)
    portfolio = s.v30.candidate_portfolio(right, down, seed + scene)
    rows = []
    for index, (name, board) in enumerate(portfolio.items()):
        refined, score = s.v30.lns_refine(
            board, right, down, unary, unary_weight, seed + scene + index * 97)
        rows.append((score, seed, name, refined))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "final"), default="validation")
    parser.add_argument("--seeds", default="350826,360826,380826")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    scenes = s.VALID_SCENES if args.split == "validation" else s.FINAL_SCENES
    device = torch.device("cuda")
    reranker, heads, unary_weight = s.load_models(device, "old")
    rows = []
    started = time.perf_counter()
    for scene in scenes:
        matrices = s.v30.load_eval(scene, reranker, device)
        candidates = []
        for seed in seeds:
            candidates.extend(seed_candidates(scene, matrices, heads, unary_weight, device, seed))
        score, seed, name, board = max(candidates, key=lambda row: row[0])
        oracle = max(candidates, key=lambda row: s.v30.placement_metrics(row[3])["adjacency"])
        metrics = s.v30.placement_metrics(board)
        oracle_metrics = s.v30.placement_metrics(oracle[3])
        row = {"scene": scene, "seed": seed, "selected": name,
               "objective": float(score), "oracle_seed": oracle[1],
               "oracle_method": oracle[2], "oracle_adjacency": oracle_metrics["adjacency"],
               **metrics}
        rows.append(row)
        print(json.dumps({"event": "scene", "seconds": time.perf_counter() - started,
                          **row}), flush=True)
    aggregate = {key: float(np.mean([row[key] for row in rows]))
                 for key in ("adjacency", "translation_aligned_placement", "direct_placement",
                             "oracle_adjacency")}
    aggregate["composite"] = aggregate["adjacency"] + .25 * aggregate["translation_aligned_placement"]
    report = {"split": args.split, "seeds": seeds, "aggregate": aggregate, "rows": rows}
    (s.OUT / args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "complete", **aggregate}), flush=True)


if __name__ == "__main__":
    main()

