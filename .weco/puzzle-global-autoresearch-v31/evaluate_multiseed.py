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
    visualization = None
    maps = np.load(s.v30.v25.MAP_FILE)["inv"] if args.split == "final" else None
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
        if args.split == "final" and scene == 6989:
            tiles = s.v30.v25.load_raw_target_order(scene, maps).permute(0, 2, 3, 1).mul(255).byte().numpy()
            target = s.v30.v25.v10.load_rgb(
                s.v30.v25.RAW_INPUTS.parent / "targets" / f"img_{scene:06d}.png")
            montage = np.hstack((
                s.v30.labelled(s.v30.render_board(tiles, board), "V31 fused 3-seed global solver"),
                s.v30.labelled(target, "Clean target (reference)")))
            visualization = s.OUT / f"assembly_v31_scene_{scene}.png"
            s.v30.cv2.imwrite(str(visualization), s.v30.cv2.cvtColor(
                montage, s.v30.cv2.COLOR_RGB2BGR))
        print(json.dumps({"event": "scene", "seconds": time.perf_counter() - started,
                          **row}), flush=True)
    aggregate = {key: float(np.mean([row[key] for row in rows]))
                 for key in ("adjacency", "translation_aligned_placement", "direct_placement",
                             "oracle_adjacency")}
    aggregate["composite"] = aggregate["adjacency"] + .25 * aggregate["translation_aligned_placement"]
    report = {"split": args.split, "seeds": seeds, "aggregate": aggregate,
              "rows": rows, "visualization": str(visualization) if visualization else None}
    (s.OUT / args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "complete", **aggregate}), flush=True)


if __name__ == "__main__":
    main()
