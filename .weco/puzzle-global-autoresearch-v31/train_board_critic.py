"""Generate fused-domain solver candidates and train an OOF pairwise board critic."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

import solver_v31 as s

TRAIN = tuple(range(6700, 6728)) + tuple(range(6957, 6981))
VALID = tuple(range(6981, 6989))
CACHE = s.ROOT / "critic_cache"
QUANTILES = np.asarray((0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1.0))


def summarize(values):
    x = np.asarray(values, np.float64).reshape(-1)
    return np.concatenate((np.quantile(x, QUANTILES), (x.mean(), x.std())))


def board_features(board, raw_right, raw_down, unary):
    right = s.v30.global_solver._normalise(raw_right)
    down = s.v30.global_solver._normalise(raw_down)
    rank_right, rank_down = s.structural_matrices(raw_right, raw_down)
    raw_h, raw_v, raw_loops = s.edge_and_loops(board, right, down)
    rank_h, rank_v, rank_loops = s.edge_and_loops(board, rank_right, rank_down)
    selected_unary = unary[np.asarray(board), np.arange(s.N)]
    local = s.local_quality(board, right, down, unary, .5, 0.0)
    threshold_counts = np.asarray([
        np.mean(np.concatenate((rank_h.reshape(-1), rank_v.reshape(-1))) >= threshold)
        for threshold in (.90, .95, .98, .99, .995)
    ])
    return np.concatenate((summarize(raw_h), summarize(raw_v), summarize(raw_loops),
                           summarize(rank_h), summarize(rank_v), summarize(rank_loops),
                           summarize(selected_unary), summarize(local), threshold_counts)).astype(np.float32)


def scene_candidates(scene, matrices, heads, unary_weight, device):
    path = CACHE / f"scene_{scene:06d}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return data["features"], data["labels"], data["names"].astype(str)
    right, down = matrices
    unary = s.v30.unary_from_heads(heads, matrices, device)
    portfolio = s.v30.candidate_portfolio(right, down, s.SEED + scene)
    boards = {}
    objectives = {}
    for index, (name, board) in enumerate(portfolio.items()):
        v30_board, v30_objective = s.v30.lns_refine(
            board, right, down, unary, unary_weight, s.SEED + scene + index * 97)
        boards[f"v30_{name}"] = v30_board
        objectives[f"v30_{name}"] = v30_objective
        for loop_weight in (0.0, .25):
            key = f"v31_l{loop_weight:g}_{name}"
            boards[key], objectives[key] = s.refine(
                v30_board, right, down, unary, unary_weight,
                s.SEED + scene * 101 + index * 977 + int(loop_weight * 10000),
                rounds=24, loop_weight=loop_weight)
    names = np.asarray(list(boards))
    features = np.stack([board_features(boards[name], right, down, unary) for name in names])
    labels = np.asarray([s.v30.placement_metrics(boards[name])["adjacency"] for name in names],
                        np.float32)
    baseline_name = max((name for name in names if name.startswith("v30_")),
                        key=lambda name: objectives[name])
    names = np.concatenate((names, np.asarray(["baseline_marker"])))
    features = np.concatenate((features, features[[list(boards).index(baseline_name)]]))
    labels = np.concatenate((labels, labels[[list(boards).index(baseline_name)]]))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, labels=labels, names=names)
    return features, labels, names


def pairwise_rows(scene_data, scenes):
    rows = []
    targets = []
    for scene in scenes:
        features, labels, names = scene_data[scene]
        keep = names != "baseline_marker"
        features, labels = features[keep], labels[keep]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                delta = float(labels[i] - labels[j])
                if abs(delta) < 1e-9:
                    continue
                rows.extend((features[i] - features[j], features[j] - features[i]))
                targets.extend((delta, -delta))
    return np.asarray(rows, np.float64), np.asarray(targets, np.float64)


def fit_ridge(scene_data, scenes, regularization):
    x, y = pairwise_rows(scene_data, scenes)
    scale = x.std(0) + 1e-6
    z = x / scale
    coef = np.linalg.solve(z.T @ z + regularization * np.eye(z.shape[1]), z.T @ y)
    return coef / scale


def select_metrics(scene_data, scenes, coef):
    selected = []
    baseline = []
    oracle = []
    rows = []
    for scene in scenes:
        features, labels, names = scene_data[scene]
        candidate = names != "baseline_marker"
        scores = features[candidate] @ coef
        picked = int(np.flatnonzero(candidate)[int(np.argmax(scores))])
        base = int(np.flatnonzero(names == "baseline_marker")[0])
        best = int(np.flatnonzero(candidate)[int(np.argmax(labels[candidate]))])
        selected.append(labels[picked]); baseline.append(labels[base]); oracle.append(labels[best])
        rows.append({"scene": scene, "selected": str(names[picked]),
                     "selected_adjacency": float(labels[picked]),
                     "baseline_adjacency": float(labels[base]),
                     "oracle_adjacency": float(labels[best])})
    return {"selected": float(np.mean(selected)), "baseline": float(np.mean(baseline)),
            "oracle": float(np.mean(oracle)), "rows": rows}


def main():
    device = torch.device("cuda")
    reranker, heads, unary_weight = s.load_models(device, "fused")
    scene_data = {}
    started = time.perf_counter()
    for index, scene in enumerate(TRAIN + VALID, 1):
        matrices = s.v30.load_eval(scene, reranker, device)
        scene_data[scene] = scene_candidates(scene, matrices, heads, unary_weight, device)
        print(json.dumps({"event": "candidates", "scene": scene, "index": index,
                          "of": len(TRAIN + VALID), "seconds": time.perf_counter() - started}), flush=True)
    folds = tuple(tuple(TRAIN[index::4]) for index in range(4))
    trials = []
    for regularization in (1e-4, 1e-3, 1e-2, .1, 1.0, 10.0, 100.0):
        values = []
        for heldout in folds:
            fit_scenes = tuple(scene for scene in TRAIN if scene not in heldout)
            coef = fit_ridge(scene_data, fit_scenes, regularization)
            values.append(select_metrics(scene_data, heldout, coef)["selected"])
        trials.append({"regularization": regularization, "oof_adjacency": float(np.mean(values))})
    selected_regularization = max(trials, key=lambda row: row["oof_adjacency"])["regularization"]
    coef = fit_ridge(scene_data, TRAIN, selected_regularization)
    validation = select_metrics(scene_data, VALID, coef)
    np.savez(s.OUT / "board_critic_v31.npz", coef=coef,
             regularization=selected_regularization)
    report = {"trials": trials, "selected_regularization": selected_regularization,
              "validation": validation, "feature_count": int(len(coef)),
              "seconds": time.perf_counter() - started}
    (s.OUT / "board_critic_v31.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "complete", **report}), flush=True)


if __name__ == "__main__":
    main()

