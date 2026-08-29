"""Validation-only global seed sweep for the frozen fused V30 solver."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import solver_v31 as s


def solve_seed(scene, matrices, heads, unary_weight, device, seed):
    right, down = matrices
    unary = s.v30.unary_from_heads(heads, matrices, device)
    portfolio = s.v30.candidate_portfolio(right, down, seed + scene)
    boards = {}
    objectives = {}
    for index, (name, board) in enumerate(portfolio.items()):
        boards[name], objectives[name] = s.v30.lns_refine(
            board, right, down, unary, unary_weight, seed + scene + index * 97)
    selected = max(boards, key=objectives.get)
    return s.v30.placement_metrics(boards[selected]), selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    device = torch.device("cuda")
    reranker, heads, unary_weight = s.load_models(device, "old")
    bundles = {scene: s.v30.load_eval(scene, reranker, device) for scene in s.VALID_SCENES}
    results = []
    started = time.perf_counter()
    for seed in seeds:
        rows = []
        for scene in s.VALID_SCENES:
            metrics, selected = solve_seed(scene, bundles[scene], heads, unary_weight,
                                           device, seed)
            rows.append({"scene": scene, "selected": selected, **metrics})
        aggregate = {key: float(np.mean([row[key] for row in rows]))
                     for key in ("adjacency", "translation_aligned_placement", "direct_placement")}
        aggregate["composite"] = aggregate["adjacency"] + .25 * aggregate["translation_aligned_placement"]
        result = {"seed": seed, "aggregate": aggregate, "rows": rows}
        results.append(result)
        print(json.dumps({"event": "seed", "seconds": time.perf_counter() - started,
                          **result}), flush=True)
    report = {"results": results, "winner": max(results, key=lambda row: row["aggregate"]["adjacency"])}
    (s.OUT / args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "complete", "winner": report["winner"]}), flush=True)


if __name__ == "__main__":
    main()

