"""Generate real solver boards, near misses, and paired spatial tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/kva/pazzle_global_autoresearch_v32_noise")
V31_ROOT = Path("/home/kva/pazzle_global_autoresearch_v31")
sys.path[:0] = [str(ROOT), str(V31_ROOT)]
import spatial_critic_v32 as spatial
import solver_v31 as solver

SCENES = tuple(range(6700, 6728)) + tuple(range(6957, 6989))
SCORE_CACHE = ROOT / "noisy_score_cache"
OUT = ROOT / "spatial_cache"


@torch.inference_mode()
def head_views(heads, matrices, device):
    node, neighbours, weights = solver.v30.graph_inputs(matrices)
    row, col, border = heads(torch.from_numpy(node).to(device),
                             torch.from_numpy(neighbours).to(device),
                             torch.from_numpy(weights).to(device))
    row = F.log_softmax(row, 1).cpu().numpy().astype(np.float32)
    col = F.log_softmax(col, 1).cpu().numpy().astype(np.float32)
    border = border.cpu().numpy().astype(np.float32)
    cells = np.arange(spatial.N); rr, cc = cells // spatial.SIDE, cells % spatial.SIDE
    unary = row[:, rr] + col[:, cc]
    targets = np.stack((rr == 0, rr == spatial.SIDE - 1, cc == 0, cc == spatial.SIDE - 1), 1).astype(np.float32)
    unary += .25 * (border[:, None] * (2 * targets[None] - 1)).sum(2)
    unary = (unary - unary.mean(1, keepdims=True)) / (unary.std(1, keepdims=True) + 1e-6)
    return unary.astype(np.float32), row, col, border


def load_scores(scene, view):
    with np.load(SCORE_CACHE / f"scene_{scene:06d}_{view}.npz", allow_pickle=False) as data:
        scores = data["scores"].astype(np.float32)
    return scores[0], scores[1]


def unique_boards(rows):
    result, seen = [], set()
    for name, board, synthetic in rows:
        key = np.asarray(board, np.int16).tobytes()
        if key not in seen:
            seen.add(key); result.append((name, np.asarray(board, np.int16), synthetic))
    return result


def scene_boards(scene, clean_matrices, noise_matrices, unary_clean, unary_noise, unary_weight):
    rows = []
    for source_name, matrices, unary, offset in (
        ("clean", clean_matrices, unary_clean, 0), ("noise", noise_matrices, unary_noise, 500_000)):
        portfolio = solver.v30.candidate_portfolio(*matrices, solver.SEED + scene + offset)
        for index, (name, board) in enumerate(portfolio.items()):
            v30_board, _ = solver.v30.lns_refine(board, *matrices, unary, unary_weight,
                                                  solver.SEED + scene + offset + index * 97,
                                                  rounds=12, width=64)
            rows.append((f"{source_name}_v30_{name}", v30_board, False))
            v31_board, _ = solver.refine(v30_board, *matrices, unary, unary_weight,
                                          solver.SEED + scene * 101 + offset + index * 977,
                                          rounds=12, widths=(32, 64), loop_weight=.25, two_opt=32)
            rows.append((f"{source_name}_v31_{name}", v31_board, False))
    rows = unique_boards(rows)
    # Dense local supervision around real basins; capped below real-board count.
    rng = np.random.default_rng(32_082_600 + scene)
    real = list(rows)
    for index in range(min(len(real), 10)):
        board = real[index % len(real)][1].copy()
        swaps = (1, 2, 4, 8)[index % 4]
        chosen = rng.choice(spatial.N, 2 * swaps, replace=False)
        for left, right in chosen.reshape(-1, 2):
            board[left], board[right] = board[right], board[left]
        rows.append((f"near_swap{swaps}_{index}", board, True))
    return unique_boards(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=len(SCENES))
    args = parser.parse_args()
    device = torch.device("cuda")
    _reranker, heads, unary_weight = solver.load_models(device, "old")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for ordinal, scene in enumerate(SCENES[args.start:args.stop], args.start + 1):
        path = OUT / f"scene_{scene:06d}.npz"
        required = [SCORE_CACHE / f"scene_{scene:06d}_{view}.npz" for view in ("clean", "noise_0", "noise_1")]
        if not all(item.exists() for item in required):
            print(json.dumps({"event": "waiting_for_scores", "scene": scene}), flush=True); break
        if path.exists():
            print(json.dumps({"event": "spatial_cache", "scene": scene, "cached": True}), flush=True); continue
        clean, noise0, noise1 = (load_scores(scene, view) for view in ("clean", "noise_0", "noise_1"))
        clean_heads = head_views(heads, clean, device)
        noise0_heads = head_views(heads, noise0, device)
        noise1_heads = head_views(heads, noise1, device)
        boards = scene_boards(scene, clean, noise0, clean_heads[0], noise0_heads[0], unary_weight)
        x_clean = []; x_noise = [[], []]; local = []; global_y = []
        for _name, board, _synthetic in boards:
            solver.assert_permutation(board)
            x_clean.append(spatial.board_tensor(board, *clean, *clean_heads))
            x_noise[0].append(spatial.board_tensor(board, *noise0, *noise0_heads))
            x_noise[1].append(spatial.board_tensor(board, *noise1, *noise1_heads))
            right_y, down_y, cell_y, adjacency = spatial.board_targets(board)
            local.append(np.stack((right_y, down_y, cell_y))); global_y.append(adjacency)
        names = np.asarray([row[0] for row in boards]); synthetic = np.asarray([row[2] for row in boards])
        real = np.flatnonzero(~synthetic)
        objective = [solver.objective(boards[i][1], *clean, clean_heads[0], unary_weight, .25) for i in real]
        baseline_index = int(real[int(np.argmax(objective))])
        np.savez_compressed(path, x_clean=np.asarray(x_clean, np.float16),
                            x_noise=np.asarray(x_noise, np.float16).transpose(1, 0, 2, 3, 4),
                            local=np.asarray(local, np.float16), global_y=np.asarray(global_y, np.float32),
                            names=names, synthetic=synthetic, baseline_index=baseline_index, scene=scene)
        print(json.dumps({"event": "spatial_cache", "scene": scene, "boards": len(boards),
                          "real": int((~synthetic).sum()), "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
