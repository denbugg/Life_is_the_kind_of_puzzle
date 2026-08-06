"""Synchronous multi-neighbour Hungarian repair of a buddies draft board."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import GRID, NFRAG, SEED, WORK_ROOT
from eval_genetic_solver import load_graph
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import solve_buddies_from_scores


def context_matrix(
    place: np.ndarray,
    log_right: np.ndarray,
    log_down: np.ndarray,
) -> np.ndarray:
    """Score every tile in every slot against all currently adjacent tiles."""
    board = place.reshape(GRID, GRID)
    scores = np.zeros((NFRAG, NFRAG), dtype=np.float32)
    for row in range(GRID):
        for col in range(GRID):
            slot = row * GRID + col
            if col > 0:
                scores[slot] += log_right[int(board[row, col - 1])]
            if col + 1 < GRID:
                scores[slot] += log_right[:, int(board[row, col + 1])]
            if row > 0:
                scores[slot] += log_down[int(board[row - 1, col])]
            if row + 1 < GRID:
                scores[slot] += log_down[:, int(board[row + 1, col])]
    return scores


def repair(
    place: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    *,
    active_fraction: float,
    identity_bonus: float,
    iterations: int,
    log_floor: float,
) -> np.ndarray:
    result = place.copy()
    log_right = np.log(np.maximum(right, np.exp(log_floor))).astype(np.float32)
    log_down = np.log(np.maximum(down, np.exp(log_floor))).astype(np.float32)
    active_count = min(NFRAG, max(2, int(round(active_fraction * NFRAG))))
    for _ in range(iterations):
        scores = context_matrix(result, log_right, log_down)
        current = scores[np.arange(NFRAG), result]
        active = np.argsort(current)[:active_count]
        available = result[active].copy()
        local = scores[np.ix_(active, available)].copy()
        local[np.arange(active_count), np.arange(active_count)] += identity_bonus
        rows, columns = linear_sum_assignment(-local)
        updated = result.copy()
        updated[active[rows]] = available[columns]
        if np.array_equal(updated, result):
            break
        result = updated
    return result


def evaluate(
    graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    active_fraction: float,
    identity_bonus: float,
    iterations: int,
    log_floor: float,
    budget: int,
) -> dict[str, float]:
    rows = []
    for right, down, truth, initial in graphs:
        refined = repair(
            initial, right, down,
            active_fraction=active_fraction,
            identity_bonus=identity_bonus,
            iterations=iterations,
            log_floor=log_floor,
        )
        rows.append(
            {
                "initial_placement": placement_accuracy(initial, truth)[0],
                "initial_neighbour": neighbour_accuracy(initial, truth)[0],
                "refined_placement": placement_accuracy(refined, truth)[0],
                "refined_neighbour": neighbour_accuracy(refined, truth)[0],
                "changed": float(np.mean(refined != initial)),
            }
        )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def _load(cache_dir: Path, images: str, budget: int):
    output = []
    for value in images.split(","):
        right, down, truth = load_graph(
            cache_dir / f"image_{int(value):04d}_k64.npz"
        )
        initial, _ = solve_buddies_from_scores(
            right, down, max_edges=budget, repair_passes=0
        )
        output.append((right, down, truth, initial))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--validation-images", default="18,19,20,21")
    parser.add_argument("--external-images", default="50,51,52,53,54,55")
    parser.add_argument("--fractions", default="0.1,0.2,0.35,0.5,1")
    parser.add_argument("--bonuses", default="0,1,2,4,8,12,16")
    parser.add_argument("--iteration-values", default="1,2,3")
    parser.add_argument("--floors", default="-8,-12,-20")
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument(
        "--report", type=Path,
        default=Path(WORK_ROOT) / "gates" / "context_hungarian_repair_gate.json",
    )
    args = parser.parse_args()
    validation = _load(args.cache_dir, args.validation_images, args.budget)
    external = _load(args.cache_dir, args.external_images, args.budget)
    fractions = [float(value) for value in args.fractions.split(",")]
    bonuses = [float(value) for value in args.bonuses.split(",")]
    iteration_values = [int(value) for value in args.iteration_values.split(",")]
    floors = [float(value) for value in args.floors.split(",")]
    rows: dict[str, dict[str, float]] = {}
    for floor in floors:
        for iterations in iteration_values:
            for fraction in fractions:
                for bonus in bonuses:
                    key = f"floor{floor:g}:i{iterations}:f{fraction:g}:b{bonus:g}"
                    metrics = evaluate(
                        validation,
                        active_fraction=fraction,
                        identity_bonus=bonus,
                        iterations=iterations,
                        log_floor=floor,
                        budget=args.budget,
                    )
                    rows[key] = metrics
                    print(
                        key + " " + " ".join(f"{name}={value:.4f}" for name, value in metrics.items()),
                        flush=True,
                    )
    selected = max(
        rows,
        key=lambda key: (
            rows[key]["refined_neighbour"],
            rows[key]["refined_placement"],
        ),
    )
    parts = selected.split(":")
    floor = float(parts[0][5:])
    iterations = int(parts[1][1:])
    fraction = float(parts[2][1:])
    bonus = float(parts[3][1:])
    external_metrics = evaluate(
        external,
        active_fraction=fraction,
        identity_bonus=bonus,
        iterations=iterations,
        log_floor=floor,
        budget=args.budget,
    )
    delta = {
        "placement": (
            external_metrics["refined_placement"]
            - external_metrics["initial_placement"]
        ),
        "neighbour": (
            external_metrics["refined_neighbour"]
            - external_metrics["initial_neighbour"]
        ),
    }
    report = {
        "experiment": "synchronous_multi_context_hungarian_repair",
        "selected": selected,
        "validation": rows[selected],
        "external": external_metrics,
        "delta": delta,
        "thresholds": {"neighbour": 0.01, "placement": 0.005},
        "passed": delta["neighbour"] >= 0.01 and delta["placement"] >= 0.005,
        "all_validation": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
