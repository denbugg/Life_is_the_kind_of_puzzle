"""Re-rank every frozen seam candidate with scene and reverse-row evidence.

The existing edge-confidence model can only accept or reject the ranker's
already selected top-1 edge.  This experiment instead builds one feature row
for every candidate in the frozen K=64 affinity union.  The important new
features are the candidate's rank in the reverse physical direction and
whole-scene normalized tile statistics.

Training and model selection are split by complete images.  The final report
also evaluates a disjoint external image range and runs the ordinary buddies
solver with the calibrated candidate probabilities.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lightgbm import LGBMRanker
from sklearn.ensemble import HistGradientBoostingClassifier

from config import GRID, NFRAG, WORK_ROOT
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import solve_buddies_from_scores


INVERSE = np.asarray((1, 0, 3, 2), dtype=np.int64)
DELTAS = np.asarray(((-1, 0), (1, 0), (0, -1), (0, 1)), dtype=np.int64)


@dataclass
class Graph:
    image: int
    permutation: np.ndarray
    raw: np.ndarray
    logp: np.ndarray
    percentile: np.ndarray
    valid: np.ndarray
    stats: np.ndarray
    scene_mean: np.ndarray
    scene_std: np.ndarray


def _log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max())
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum())


def load_graph(path: Path) -> Graph:
    stored = np.load(path)
    candidate_ids = stored["candidate_ids"].astype(np.int64)
    candidate_scores = stored["candidate_scores"].reshape(NFRAG, 4, -1)
    raw = np.full((4, NFRAG, NFRAG), -np.inf, dtype=np.float32)
    for anchor in range(NFRAG):
        for direction in range(4):
            values = candidate_scores[anchor, direction]
            finite = np.isfinite(values)
            np.maximum.at(
                raw[direction, anchor],
                candidate_ids[anchor, finite],
                values[finite],
            )
    for direction in range(4):
        np.fill_diagonal(raw[direction], -np.inf)

    valid = np.isfinite(raw)
    logp = np.full_like(raw, -20.0)
    percentile = np.zeros_like(raw)
    for direction in range(4):
        for anchor in range(NFRAG):
            mask = valid[direction, anchor]
            values = raw[direction, anchor, mask]
            if not len(values):
                continue
            logp[direction, anchor, mask] = _log_softmax(values)
            order = np.argsort(np.argsort(values))
            percentile[direction, anchor, mask] = (
                order.astype(np.float32) / max(1, len(values) - 1)
            )

    # Every full cache has four rows per anchor.  The source statistics are
    # consequently recoverable without storing the original tile tensor.
    features = stored["features"]
    anchors = stored["anchors"].astype(np.int64)
    stats = np.zeros((NFRAG, 7), dtype=np.float32)
    stats[anchors] = features[:, 17:24]
    return Graph(
        image=int(path.stem.split("_")[1]),
        permutation=stored["permutation"].astype(np.int64),
        raw=raw,
        logp=logp,
        percentile=percentile,
        valid=valid,
        stats=stats,
        scene_mean=stats.mean(axis=0),
        scene_std=stats.std(axis=0),
    )


def true_target(graph: Graph, anchor: int, direction: int) -> int:
    cell = int(graph.permutation[anchor])
    row, col = divmod(cell, GRID)
    dr, dc = DELTAS[direction]
    target_row, target_col = row + int(dr), col + int(dc)
    if not (0 <= target_row < GRID and 0 <= target_col < GRID):
        return -1
    inverse = np.empty(NFRAG, dtype=np.int64)
    inverse[graph.permutation] = np.arange(NFRAG)
    return int(inverse[target_row * GRID + target_col])


def edge_features(
    graph: Graph,
    anchor: int,
    direction: int,
    candidates: np.ndarray,
) -> np.ndarray:
    reverse = int(INVERSE[direction])
    direct_mask = graph.valid[direction, anchor]
    count = max(1, int(direct_mask.sum()))
    direct_raw = graph.raw[direction, anchor, candidates]
    # These moments must be computed from the complete frozen row.  Computing
    # them from the selected hard-negative subset during fitting creates a
    # severe train/evaluation distribution mismatch.
    direct_all = graph.raw[direction, anchor, direct_mask]
    direct_mean = float(direct_all.mean())
    direct_std = max(float(direct_all.std()), 1.0e-4)
    reverse_valid = graph.valid[reverse, candidates, anchor]
    reverse_logp = graph.logp[reverse, candidates, anchor]
    reverse_percentile = graph.percentile[reverse, candidates, anchor]
    reverse_z = np.full(len(candidates), -4.0, dtype=np.float32)
    for index, candidate in enumerate(candidates):
        if not reverse_valid[index]:
            continue
        reverse_mask = graph.valid[reverse, candidate]
        reverse_all = graph.raw[reverse, candidate, reverse_mask]
        reverse_mean = float(reverse_all.mean())
        reverse_std = max(float(reverse_all.std()), 1.0e-4)
        reverse_z[index] = (
            graph.raw[reverse, candidate, anchor] - reverse_mean
        ) / reverse_std
    source = np.repeat(graph.stats[anchor][None], len(candidates), axis=0)
    target = graph.stats[candidates]
    direction_one_hot = np.zeros((len(candidates), 4), dtype=np.float32)
    direction_one_hot[:, direction] = 1.0
    scene_mean = np.repeat(graph.scene_mean[None], len(candidates), axis=0)
    scene_std = np.repeat(graph.scene_std[None], len(candidates), axis=0)
    columns = (
        graph.logp[direction, anchor, candidates, None],
        ((direct_raw - direct_mean) / direct_std)[:, None],
        graph.percentile[direction, anchor, candidates, None],
        (graph.percentile[direction, anchor, candidates] >= 1.0 - 0.5 / count)[:, None],
        reverse_valid[:, None].astype(np.float32),
        reverse_logp[:, None],
        reverse_percentile[:, None],
        reverse_z[:, None],
        (
            reverse_valid
            & (reverse_percentile >= 1.0 - 1.0e-7)
        )[:, None].astype(np.float32),
        direction_one_hot,
        source,
        target,
        np.abs(source - target),
        scene_mean,
        scene_std,
    )
    return np.concatenate(columns, axis=1).astype(np.float32)


def training_rows(
    graphs: list[Graph],
    *,
    hard_negatives: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    image_ids: list[np.ndarray] = []
    groups: list[int] = []
    for graph in graphs:
        inverse = np.empty(NFRAG, dtype=np.int64)
        inverse[graph.permutation] = np.arange(NFRAG)
        for anchor in range(NFRAG):
            cell = int(graph.permutation[anchor])
            row, col = divmod(cell, GRID)
            for direction, (dr, dc) in enumerate(DELTAS):
                rr, cc = row + int(dr), col + int(dc)
                if not (0 <= rr < GRID and 0 <= cc < GRID):
                    continue
                target = int(inverse[rr * GRID + cc])
                if not graph.valid[direction, anchor, target]:
                    continue
                candidates = np.flatnonzero(graph.valid[direction, anchor])
                negative = candidates[candidates != target]
                order = np.argsort(-graph.raw[direction, anchor, negative])
                negative = negative[order[:hard_negatives]]
                selected = np.concatenate((np.asarray([target]), negative))
                one = edge_features(graph, anchor, direction, selected)
                y = np.zeros(len(selected), dtype=np.uint8)
                y[0] = 1
                features.append(one)
                labels.append(y)
                image_ids.append(np.full(len(selected), graph.image, dtype=np.int64))
                groups.append(len(selected))
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(image_ids),
        np.asarray(groups, dtype=np.int32),
    )


def model_scores(model: object, features: np.ndarray) -> np.ndarray:
    """Return a continuous higher-is-better score for either gate model."""
    if isinstance(model, HistGradientBoostingClassifier):
        probability = model.predict_proba(features)[:, 1]
        return np.log(
            np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
            / np.clip(1.0 - probability, 1.0e-6, 1.0)
        )
    return np.asarray(model.predict(features), dtype=np.float32)


def evaluate_ranker(
    model: object,
    graphs: list[Graph],
) -> dict[str, float]:
    totals = {
        "physical_rows": 0,
        "covered": 0,
        "base_r1": 0,
        "base_r5": 0,
        "cal_r1": 0,
        "cal_r5": 0,
    }
    for graph in graphs:
        inverse = np.empty(NFRAG, dtype=np.int64)
        inverse[graph.permutation] = np.arange(NFRAG)
        for anchor in range(NFRAG):
            cell = int(graph.permutation[anchor])
            row, col = divmod(cell, GRID)
            for direction, (dr, dc) in enumerate(DELTAS):
                rr, cc = row + int(dr), col + int(dc)
                if not (0 <= rr < GRID and 0 <= cc < GRID):
                    continue
                totals["physical_rows"] += 1
                target = int(inverse[rr * GRID + cc])
                if not graph.valid[direction, anchor, target]:
                    continue
                totals["covered"] += 1
                candidates = np.flatnonzero(graph.valid[direction, anchor])
                target_slot = int(np.flatnonzero(candidates == target)[0])
                base = graph.raw[direction, anchor, candidates]
                calibrated = model_scores(
                    model, edge_features(graph, anchor, direction, candidates)
                )
                base_rank = 1 + int(np.sum(base > base[target_slot]))
                cal_rank = 1 + int(np.sum(calibrated > calibrated[target_slot]))
                totals["base_r1"] += base_rank <= 1
                totals["base_r5"] += base_rank <= 5
                totals["cal_r1"] += cal_rank <= 1
                totals["cal_r5"] += cal_rank <= 5
    physical = max(1, totals["physical_rows"])
    covered = max(1, totals["covered"])
    return {
        "candidate_recall": totals["covered"] / physical,
        "base_conditional_r1": totals["base_r1"] / covered,
        "base_conditional_r5": totals["base_r5"] / covered,
        "calibrated_conditional_r1": totals["cal_r1"] / covered,
        "calibrated_conditional_r5": totals["cal_r5"] / covered,
        "base_all_true_r1": totals["base_r1"] / physical,
        "calibrated_all_true_r1": totals["cal_r1"] / physical,
        "physical_rows": float(totals["physical_rows"]),
    }


def calibrated_rd(
    model: object,
    graph: Graph,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = np.zeros((4, NFRAG, NFRAG), dtype=np.float32)
    for anchor in range(NFRAG):
        for direction in range(4):
            candidates = np.flatnonzero(graph.valid[direction, anchor])
            if not len(candidates):
                continue
            score = model_scores(
                model, edge_features(graph, anchor, direction, candidates)
            )
            probability = np.exp(score - score.max())
            probability /= max(float(probability.sum()), 1.0e-8)
            matrices[direction, anchor, candidates] = probability
    right = 0.5 * (matrices[3] + matrices[2].T)
    down = 0.5 * (matrices[1] + matrices[0].T)
    np.fill_diagonal(right, 0.0)
    np.fill_diagonal(down, 0.0)
    return right, down


def solver_metrics(
    model: object,
    graphs: list[Graph],
    budgets: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    rows = {budget: [] for budget in budgets}
    for graph in graphs:
        right, down = calibrated_rd(model, graph)
        truth = np.argsort(graph.permutation)
        for budget in budgets:
            placement, value = solve_buddies_from_scores(
                right, down, max_edges=budget, repair_passes=0
            )
            place, _ = placement_accuracy(placement, truth)
            neighbour, horizontal, vertical = neighbour_accuracy(placement, truth)
            rows[budget].append((place, neighbour, horizontal, vertical, value))
    return {
        str(budget): {
            "placement": float(np.mean([row[0] for row in values])),
            "neighbour": float(np.mean([row[1] for row in values])),
            "right": float(np.mean([row[2] for row in values])),
            "down": float(np.mean([row[3] for row in values])),
            "objective": float(np.mean([row[4] for row in values])),
        }
        for budget, values in rows.items()
    }


def _paths(cache_dir: Path, text: str) -> list[Path]:
    result = []
    for part in text.split(","):
        image = int(part.strip())
        path = cache_dir / f"image_{image:04d}_k64.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--fit-images", default="10,11,12,13,14,15,16,17")
    parser.add_argument("--validation-images", default="18,19,20,21")
    parser.add_argument("--external-images", default="50,51,52,53,54,55")
    parser.add_argument("--hard-negatives", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--model-type", choices=("lambdarank", "binary"), default="lambdarank")
    parser.add_argument("--budgets", default="128,256,384,512")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "candidate_calibrator_gate.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "candidate_calibrator.pkl",
    )
    args = parser.parse_args()
    if args.hard_negatives < 1 or args.max_iter < 1:
        parser.error("--hard-negatives and --max-iter must be positive")

    fit = [load_graph(path) for path in _paths(args.cache_dir, args.fit_images)]
    validation = [
        load_graph(path) for path in _paths(args.cache_dir, args.validation_images)
    ]
    external = [
        load_graph(path) for path in _paths(args.cache_dir, args.external_images)
    ]
    x, y, image_ids, groups = training_rows(fit, hard_negatives=args.hard_negatives)
    positive = max(1, int(y.sum()))
    negative = max(1, int((1 - y).sum()))
    sample_weight = np.where(y > 0, negative / positive, 1.0)
    if args.model_type == "lambdarank":
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            learning_rate=args.learning_rate,
            n_estimators=args.max_iter,
            num_leaves=args.max_leaf_nodes,
            min_child_samples=40,
            reg_lambda=1.0,
            verbosity=-1,
            random_state=1234,
            n_jobs=-1,
        )
        model.fit(x, y, group=groups)
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=1234,
        )
        model.fit(x, y, sample_weight=sample_weight)
    validation_metrics = evaluate_ranker(model, validation)
    external_metrics = evaluate_ranker(model, external)
    budgets = tuple(int(value) for value in args.budgets.split(","))
    solver = solver_metrics(model, external, budgets)
    best_budget = max(
        solver,
        key=lambda key: (solver[key]["neighbour"], solver[key]["placement"]),
    )
    checks = {
        "validation_r1_delta": (
            validation_metrics["calibrated_conditional_r1"]
            - validation_metrics["base_conditional_r1"]
            >= 0.03
        ),
        "external_r1_delta": (
            external_metrics["calibrated_conditional_r1"]
            - external_metrics["base_conditional_r1"]
            >= 0.02
        ),
        "external_neighbour": solver[best_budget]["neighbour"] >= 0.18,
    }
    report = {
        "experiment": "all_candidate_scene_reverse_calibrator",
        "status": "pass" if all(checks.values()) else "fail",
        "config": vars(args) | {"cache_dir": str(args.cache_dir), "report": str(args.report),
                               "checkpoint": str(args.checkpoint)},
        "training": {
            "rows": int(len(y)),
            "positives": positive,
            "images": sorted(np.unique(image_ids).tolist()),
            "features": int(x.shape[1]),
        },
        "validation": validation_metrics,
        "external": external_metrics,
        "solver": solver,
        "best_budget": best_budget,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.checkpoint.open("wb") as handle:
        pickle.dump({"model": model, "report": report}, handle)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
