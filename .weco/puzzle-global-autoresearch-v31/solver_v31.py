"""V31 structural global-solver experiments.

The module deliberately imports frozen V30 feature/model code and changes only
global objective and search. Target layouts are used for reporting, never for
candidate generation or selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

ROOT = Path("/home/kva/pazzle_global_autoresearch_v31")
V30_ROOT = Path("/home/kva/pazzle_edge_unary_lns_v30")
sys.path.insert(0, str(V30_ROOT))
import train_solver_v30 as v30

SIDE = 24
N = SIDE * SIDE
SEED = 310826
VALID_SCENES = tuple(range(6981, 6989))
FINAL_SCENES = (6732, 6733, 6734, 6735) + tuple(range(6989, 7000))
OUT = ROOT / "outputs"


def log(**values):
    print(json.dumps(values, sort_keys=True), flush=True)


def assert_permutation(board):
    board = np.asarray(board)
    if board.shape != (N,) or not np.array_equal(np.sort(board), np.arange(N)):
        raise AssertionError("board must contain every tile exactly once")


def stable_union(*groups, limit=None):
    """Deduplicate without np.unique's destructive sorting/truncation bias."""
    seen = set()
    values = []
    for group in groups:
        for value in np.asarray(group).reshape(-1):
            item = int(value)
            if item not in seen:
                seen.add(item)
                values.append(item)
                if limit is not None and len(values) == limit:
                    return np.asarray(values, np.int32)
    return np.asarray(values, np.int32)


def mutual_rank_matrix(matrix):
    safe = np.asarray(matrix, np.float32).copy()
    np.fill_diagonal(safe, -np.inf)
    row_order = np.argsort(-safe, axis=1, kind="stable")
    col_order = np.argsort(-safe, axis=0, kind="stable")
    row_rank = np.empty_like(row_order)
    col_rank = np.empty_like(col_order)
    row_rank[np.arange(N)[:, None], row_order] = np.arange(N)[None, :]
    col_rank[col_order, np.arange(N)[None, :]] = np.arange(N)[:, None]
    row_conf = (N - 1 - np.minimum(row_rank, N - 1)) / (N - 1)
    col_conf = (N - 1 - np.minimum(col_rank, N - 1)) / (N - 1)
    result = np.sqrt(np.maximum(0.0, row_conf * col_conf)).astype(np.float32)
    np.fill_diagonal(result, 0.0)
    return result


def structural_matrices(right, down):
    return mutual_rank_matrix(right), mutual_rank_matrix(down)


def edge_and_loops(board, right, down):
    grid = np.asarray(board).reshape(SIDE, SIDE)
    horizontal = right[grid[:, :-1], grid[:, 1:]]
    vertical = down[grid[:-1], grid[1:]]
    loops = np.minimum.reduce((horizontal[:-1], horizontal[1:],
                               vertical[:, :-1], vertical[:, 1:]))
    return horizontal, vertical, loops


def objective(board, right, down, unary, unary_weight, loop_weight):
    horizontal, vertical, loops = edge_and_loops(board, right, down)
    pair = float(horizontal.sum() + vertical.sum())
    unary_score = float(unary[np.asarray(board), np.arange(N)].sum())
    return pair + unary_weight * unary_score + loop_weight * float(loops.sum())


def local_quality(board, right, down, unary, unary_weight, loop_weight):
    grid = np.asarray(board).reshape(SIDE, SIDE)
    horizontal, vertical, loops = edge_and_loops(board, right, down)
    local = np.zeros((SIDE, SIDE), np.float32)
    local[:, :-1] += horizontal
    local[:, 1:] += horizontal
    local[:-1] += vertical
    local[1:] += vertical
    if loop_weight:
        weighted = loop_weight * loops
        local[:-1, :-1] += weighted
        local[:-1, 1:] += weighted
        local[1:, :-1] += weighted
        local[1:, 1:] += weighted
    local.reshape(-1)[:] += unary_weight * unary[np.asarray(board), np.arange(N)]
    return local.reshape(-1)


def destroy_cells(board, right, down, unary, unary_weight, loop_weight,
                  rng, width, operator):
    quality = local_quality(board, right, down, unary, unary_weight, loop_weight)
    if operator == "worst":
        # Gumbel perturbation prevents the same deterministic basin each round.
        scale = max(1e-4, float(np.std(quality)) * 0.15)
        priority = quality + rng.gumbel(0.0, scale, N)
        return np.argpartition(priority, width)[:width].astype(np.int32)
    if operator == "rectangle":
        height = max(2, int(round(math.sqrt(width))))
        width_cells = max(2, int(math.ceil(width / height)))
        row = int(rng.integers(max(1, SIDE - height + 1)))
        col = int(rng.integers(max(1, SIDE - width_cells + 1)))
        region = [r * SIDE + c for r in range(row, min(SIDE, row + height))
                  for c in range(col, min(SIDE, col + width_cells))]
        remainder = np.argsort(quality, kind="stable")
        return stable_union(region, remainder, limit=width)
    if operator == "related":
        pivot = int(np.argmin(quality + rng.gumbel(0.0, max(1e-4, np.std(quality) * .1), N)))
        pr, pc = divmod(pivot, SIDE)
        cells = np.arange(N)
        rr, cc = cells // SIDE, cells % SIDE
        distance = np.abs(rr - pr) + np.abs(cc - pc)
        priority = distance + .15 * (quality - quality.min()) / (np.std(quality) + 1e-6)
        return np.argsort(priority, kind="stable")[:width].astype(np.int32)
    if operator == "strip":
        if rng.random() < .5:
            row = int(rng.integers(SIDE))
            region = np.arange(row * SIDE, (row + 1) * SIDE)
        else:
            col = int(rng.integers(SIDE))
            region = np.arange(col, N, SIDE)
        return stable_union(region, np.argsort(quality, kind="stable"), limit=width)
    raise ValueError(operator)


def hungarian_scores(board, tiles, cells, right, down, unary, unary_weight):
    scores = v30.global_solver._cell_scores(board, tiles, cells, right, down, SIDE)
    scores += unary_weight * unary[np.ix_(tiles, cells)]
    return scores


def iterative_repair(board, cells, right, down, unary, unary_weight,
                     loop_weight, passes=3):
    current = np.asarray(board).copy()
    best_score = objective(current, right, down, unary, unary_weight, loop_weight)
    tiles = current[cells].copy()
    for _ in range(passes):
        scores = hungarian_scores(current, tiles, cells, right, down, unary, unary_weight)
        tile_rows, cell_cols = linear_sum_assignment(-scores.astype(np.float64))
        candidate = current.copy()
        candidate[cells[cell_cols]] = tiles[tile_rows]
        candidate_score = objective(candidate, right, down, unary, unary_weight, loop_weight)
        if candidate_score > best_score + 1e-7:
            current, best_score = candidate, candidate_score
        else:
            break
    assert_permutation(current)
    return current, best_score


def exact_two_opt(board, cells, right, down, unary, unary_weight, loop_weight,
                  rng, proposals=96):
    current = np.asarray(board).copy()
    best = objective(current, right, down, unary, unary_weight, loop_weight)
    for _ in range(proposals):
        a, b = rng.choice(cells, 2, replace=False)
        candidate = current.copy()
        candidate[a], candidate[b] = candidate[b], candidate[a]
        score = objective(candidate, right, down, unary, unary_weight, loop_weight)
        if score > best + 1e-7:
            current, best = candidate, score
    assert_permutation(current)
    return current, best


def refine(board, raw_right, raw_down, unary, unary_weight, seed,
           rounds=24, widths=(32, 64, 96), loop_weight=.5,
           operators=("worst", "related", "rectangle", "strip"), two_opt=64):
    right, down = structural_matrices(raw_right, raw_down)
    current = np.asarray(board).copy()
    best = objective(current, right, down, unary, unary_weight, loop_weight)
    rng = np.random.default_rng(seed)
    stagnation = 0
    op_reward = {name: 1.0 for name in operators}
    for iteration in range(rounds):
        width_index = min(len(widths) - 1, stagnation // max(1, rounds // len(widths)))
        width = int(widths[width_index])
        weights = np.asarray([op_reward[name] for name in operators], np.float64)
        weights /= weights.sum()
        operator = str(rng.choice(operators, p=weights))
        cells = destroy_cells(current, right, down, unary, unary_weight,
                              loop_weight, rng, width, operator)
        candidate, score = iterative_repair(current, cells, right, down, unary,
                                             unary_weight, loop_weight)
        candidate, score = exact_two_opt(candidate, cells, right, down, unary,
                                         unary_weight, loop_weight, rng, two_opt)
        gain = score - best
        op_reward[operator] = .85 * op_reward[operator] + .15 * (1.0 + max(0.0, gain))
        if gain > 1e-7:
            current, best, stagnation = candidate, score, 0
        else:
            stagnation += 1
    assert_permutation(current)
    return current, best


def load_models(device):
    reranker_state = torch.load(v30.V27_ROOT / "outputs/set_reranker_best.pt",
                                map_location=device, weights_only=True)
    reranker = v30.v27.SetReranker().to(device)
    reranker.load_state_dict(reranker_state["model"])
    reranker.eval()
    state = torch.load(V30_ROOT / "outputs/solver_v30.pt", map_location=device,
                       weights_only=True)
    heads = v30.DirectionalCoordinateGNN(width=state["head_width"],
                                         steps=state["head_steps"]).to(device)
    heads.load_state_dict(state["heads"])
    heads.eval()
    return reranker, heads, float(state["unary_weight"])


def solve_scene(scene, matrices, heads, unary_weight, device, config):
    right, down = matrices
    unary = v30.unary_from_heads(heads, matrices, device)
    portfolio = v30.candidate_portfolio(right, down, SEED + scene)
    boards = {}
    scores = {}
    started = time.perf_counter()
    for index, (name, board) in enumerate(portfolio.items()):
        boards[name], scores[name] = refine(
            board, right, down, unary, unary_weight,
            SEED + scene * 101 + index * 977, **config)
    selected = max(scores, key=scores.get)
    oracle = max(boards, key=lambda key: v30.placement_metrics(boards[key])["adjacency"])
    return {
        "board": boards[selected], "selected": selected,
        "selected_score": scores[selected], "oracle": oracle,
        "oracle_metrics": v30.placement_metrics(boards[oracle]),
        "metrics": v30.placement_metrics(boards[selected]),
        "seconds": time.perf_counter() - started,
    }


def aggregate(rows):
    keys = ("adjacency", "aligned_placement", "composite")
    result = {key: float(np.mean([row["metrics"][key] for row in rows])) for key in keys}
    result["oracle_adjacency"] = float(np.mean([row["oracle_metrics"]["adjacency"] for row in rows]))
    result["seconds"] = float(sum(row["seconds"] for row in rows))
    result["coverage"] = 1.0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("smoke", "validation", "final"), default="smoke")
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--loop-weight", type=float, default=.5)
    parser.add_argument("--output", default="report.json")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    reranker, heads, unary_weight = load_models(device)
    scenes = (6981,) if args.split == "smoke" else (VALID_SCENES if args.split == "validation" else FINAL_SCENES)
    config = {"rounds": args.rounds, "loop_weight": args.loop_weight}
    rows = []
    for scene in scenes:
        matrices = v30.load_v27(scene, reranker, device) if scene in VALID_SCENES else v30.load_eval(scene, reranker, device)
        row = solve_scene(scene, matrices, heads, unary_weight, device, config)
        row["scene"] = scene
        assert_permutation(row.pop("board"))
        rows.append(row)
        log(event="scene", **row)
    report = {"split": args.split, "config": config, "aggregate": aggregate(rows), "scenes": rows}
    path = OUT / args.output
    path.write_text(json.dumps(report, indent=2))
    log(event="complete", path=str(path), **report["aggregate"])


if __name__ == "__main__":
    main()

