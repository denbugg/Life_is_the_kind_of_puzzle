"""Factorize the puzzle into 24 directed paths, then order those paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import GRID, NFRAG, WORK_ROOT
from eval_genetic_solver import board_fitness, load_graph
from placement_metrics import neighbour_accuracy, placement_accuracy


def equal_path_forest(scores: np.ndarray, path_count: int, path_length: int) -> list[np.ndarray]:
    """Maximum-edge greedy directed forest with exact equal final capacities."""
    count = scores.shape[0]
    if scores.shape != (count, count) or count != path_count * path_length:
        raise ValueError("score shape and requested path geometry disagree")
    paths: dict[int, list[int]] = {index: [index] for index in range(count)}
    component = np.arange(count, dtype=np.int64)
    head = np.arange(count, dtype=np.int64)
    tail = np.arange(count, dtype=np.int64)
    values = scores.copy()
    np.fill_diagonal(values, -np.inf)
    order = np.argsort(-values, axis=None)
    for flat in order:
        source, target = divmod(int(flat), count)
        first, second = int(component[source]), int(component[target])
        if first == second or source != tail[first] or target != head[second]:
            continue
        if len(paths[first]) + len(paths[second]) > path_length:
            continue
        moved = paths.pop(second)
        paths[first].extend(moved)
        for node in moved:
            component[node] = first
        tail[first] = tail[second]
        if len(paths) == path_count:
            break
    result = [np.asarray(path, dtype=np.int64) for path in paths.values()]
    if len(result) != path_count or any(len(path) != path_length for path in result):
        return assignment_cycle_paths(scores, path_count, path_length)
    return result


def assignment_cycle_paths(
    scores: np.ndarray, path_count: int, path_length: int
) -> list[np.ndarray]:
    """Hungarian successor cover, greedily stitch its cycles, then cut equally."""
    count = scores.shape[0]
    values = scores.copy()
    np.fill_diagonal(values, -1.0e9)
    source, target = linear_sum_assignment(-values)
    successor = np.empty(count, dtype=np.int64)
    successor[source] = target
    unseen = set(range(count))
    cycles: list[list[int]] = []
    while unseen:
        start = next(iter(unseen))
        cycle, node = [], start
        while node in unseen:
            unseen.remove(node)
            cycle.append(node)
            node = int(successor[node])
        cycles.append(cycle)
    cycles.sort(key=len, reverse=True)
    sequence = cycles.pop(0)
    while cycles:
        tail = sequence[-1]
        best_cycle = best_index = -1
        best_score = -np.inf
        for cycle_index, cycle in enumerate(cycles):
            candidate_scores = scores[tail, cycle]
            local = int(np.argmax(candidate_scores))
            if float(candidate_scores[local]) > best_score:
                best_score = float(candidate_scores[local])
                best_cycle, best_index = cycle_index, local
        cycle = cycles.pop(best_cycle)
        cycle = cycle[best_index:] + cycle[:best_index]
        sequence.extend(cycle)
    if len(sequence) != count:
        raise AssertionError("cycle stitching lost tiles")
    return [
        np.asarray(sequence[start : start + path_length], dtype=np.int64)
        for start in range(0, count, path_length)
    ]


def rows_then_vertical(right: np.ndarray, down: np.ndarray) -> np.ndarray:
    rows = equal_path_forest(right, GRID, GRID)
    row_score = np.empty((GRID, GRID), dtype=np.float32)
    for first in range(GRID):
        for second in range(GRID):
            row_score[first, second] = down[rows[first], rows[second]].sum()
    order = equal_path_forest(row_score, 1, GRID)[0]
    return np.concatenate([rows[int(index)] for index in order])


def columns_then_horizontal(right: np.ndarray, down: np.ndarray) -> np.ndarray:
    columns = equal_path_forest(down, GRID, GRID)
    column_score = np.empty((GRID, GRID), dtype=np.float32)
    for first in range(GRID):
        for second in range(GRID):
            column_score[first, second] = right[columns[first], columns[second]].sum()
    order = equal_path_forest(column_score, 1, GRID)[0]
    board = np.stack([columns[int(index)] for index in order], axis=1)
    return board.reshape(-1)


def solve(right: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, str, float]:
    candidates = {
        "rows": rows_then_vertical(right, down),
        "columns": columns_then_horizontal(right, down),
    }
    scored = {
        key: board_fitness(place, right, down)
        for key, place in candidates.items()
    }
    selected = max(scored, key=scored.get)
    return candidates[selected], selected, scored[selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--images", default="50,51,52,53,54,55")
    parser.add_argument(
        "--report", type=Path,
        default=Path(WORK_ROOT) / "gates" / "path_factorized_solver_gate.json",
    )
    args = parser.parse_args()
    rows = []
    for value in args.images.split(","):
        image = int(value)
        right, down, truth = load_graph(
            args.cache_dir / f"image_{image:04d}_k64.npz"
        )
        placement, selected, objective = solve(right, down)
        metrics = {
            "image": image,
            "selected": selected,
            "objective": objective,
            "placement": placement_accuracy(placement, truth)[0],
            "neighbour": neighbour_accuracy(placement, truth)[0],
            "right": neighbour_accuracy(placement, truth)[1],
            "down": neighbour_accuracy(placement, truth)[2],
        }
        rows.append(metrics)
        print(json.dumps(metrics), flush=True)
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("objective", "placement", "neighbour", "right", "down")
    }
    report = {
        "experiment": "equal_directed_path_factorization",
        "images": rows,
        "summary": summary,
        "thresholds": {"neighbour": 0.20, "placement": 0.01},
        "passed": summary["neighbour"] >= 0.20 and summary["placement"] >= 0.01,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
