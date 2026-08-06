"""Genetic kernel-growing gate for the candidate-ranker compatibility graph.

The existing buddies solver commits to one sparse component packing.  This
evaluator instead evolves a population of complete boards.  Crossover preserves
tile relations shared by two parents first, then grows through high-scoring
ranker edges.  Selection is entirely label-free; the known permutation is read
only after a generation has been produced for diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import GRID, NFRAG, SEED, WORK_ROOT
from eval_seeded_qap import dense_rd
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import (
    _fill_board,
    _place_components,
    _place_components_randomized,
    build_buddies_components,
)

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
OPPOSITE = (DOWN, UP, RIGHT, LEFT)


def board_fitness(place: np.ndarray, right: np.ndarray, down: np.ndarray) -> float:
    board = place.reshape(GRID, GRID)
    return float(
        right[board[:, :-1], board[:, 1:]].sum()
        + down[board[:-1, :], board[1:, :]].sum()
    )


def relations(place: np.ndarray) -> np.ndarray:
    """``out[direction,tile]`` is that tile's neighbour in one complete board."""
    board = place.reshape(GRID, GRID)
    out = np.full((4, NFRAG), -1, np.int64)
    out[RIGHT, board[:, :-1]] = board[:, 1:]
    out[LEFT, board[:, 1:]] = board[:, :-1]
    out[DOWN, board[:-1, :]] = board[1:, :]
    out[UP, board[1:, :]] = board[:-1, :]
    return out


class Compatibility:
    def __init__(self, right: np.ndarray, down: np.ndarray, top_k: int = 8) -> None:
        self.right = np.asarray(right, np.float32)
        self.down = np.asarray(down, np.float32)
        self.top_k = int(top_k)
        self.row_right = np.argsort(-self.right, axis=1)[:, :top_k]
        self.col_right = np.argsort(-self.right, axis=0)[:top_k].T
        self.row_down = np.argsort(-self.down, axis=1)[:, :top_k]
        self.col_down = np.argsort(-self.down, axis=0)[:top_k].T

    def score(self, anchor: int, candidate: int, direction: int) -> float:
        if direction == RIGHT:
            return float(self.right[anchor, candidate])
        if direction == LEFT:
            return float(self.right[candidate, anchor])
        if direction == DOWN:
            return float(self.down[anchor, candidate])
        return float(self.down[candidate, anchor])

    def candidates(self, anchor: int, direction: int) -> np.ndarray:
        if direction == RIGHT:
            return self.row_right[anchor]
        if direction == LEFT:
            return self.col_right[anchor]
        if direction == DOWN:
            return self.row_down[anchor]
        return self.col_down[anchor]


def _neighbours(cell: int):
    row, col = divmod(cell, GRID)
    for direction, (dr, dc) in enumerate(DELTAS):
        nr, nc = row + dr, col + dc
        if 0 <= nr < GRID and 0 <= nc < GRID:
            yield direction, nr * GRID + nc


def crossover(
    first: np.ndarray,
    second: np.ndarray,
    compatibility: Compatibility,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fixed-frame kernel growth with shared-parent relations as the first tier."""
    parents = (relations(first), relations(second))
    board = np.full(NFRAG, -1, np.int64)
    used = np.zeros(NFRAG, dtype=bool)

    # Copy a small seed patch from either parent.  It supplies more than one
    # reliable frontier relation without forcing the rest of that parent's frame.
    source = first if rng.random() < 0.5 else second
    row = int(rng.integers(GRID - 1))
    col = int(rng.integers(GRID - 1))
    seed_cells = np.array(
        [row * GRID + col, row * GRID + col + 1, (row + 1) * GRID + col, (row + 1) * GRID + col + 1]
    )
    seed_tiles = source[seed_cells]
    board[seed_cells] = seed_tiles
    used[seed_tiles] = True
    frontier: set[int] = set()
    for cell in seed_cells:
        frontier.update(other for _, other in _neighbours(int(cell)) if board[other] < 0)

    while frontier:
        # Most constrained cell first; ties are randomized to keep population diversity.
        counts = np.array(
            [sum(board[other] >= 0 for _, other in _neighbours(cell)) for cell in frontier],
            np.int64,
        )
        cells = np.fromiter(frontier, dtype=np.int64)
        tied = cells[counts == counts.max()]
        cell = int(tied[int(rng.integers(len(tied)))])
        contacts = [
            (OPPOSITE[direction], int(board[other]))
            for direction, other in _neighbours(cell)
            if board[other] >= 0
        ]
        pool: set[int] = set()
        votes: dict[int, int] = {}
        agreements: dict[int, int] = {}
        for direction_from_anchor, anchor in contacts:
            suggestions = [int(parent[direction_from_anchor, anchor]) for parent in parents]
            for suggestion in suggestions:
                if suggestion >= 0 and not used[suggestion]:
                    pool.add(suggestion)
                    votes[suggestion] = votes.get(suggestion, 0) + 1
            if suggestions[0] == suggestions[1] and suggestions[0] >= 0 and not used[suggestions[0]]:
                agreements[suggestions[0]] = agreements.get(suggestions[0], 0) + 1
            for suggestion in compatibility.candidates(anchor, direction_from_anchor):
                if not used[int(suggestion)]:
                    pool.add(int(suggestion))
        if not pool:
            pool.update(np.flatnonzero(~used).tolist())

        best_tile = -1
        best_key: tuple[float, ...] | None = None
        for tile in pool:
            contact_scores = [
                compatibility.score(anchor, tile, direction) for direction, anchor in contacts
            ]
            key = (
                float(agreements.get(tile, 0)),
                float(votes.get(tile, 0)),
                float(sum(contact_scores)),
                float(max(contact_scores)),
                float(rng.random()) * 1.0e-9,
            )
            if best_key is None or key > best_key:
                best_key, best_tile = key, tile
        board[cell] = best_tile
        used[best_tile] = True
        frontier.remove(cell)
        frontier.update(other for _, other in _neighbours(cell) if board[other] < 0)

    if np.unique(board).size != NFRAG:
        raise AssertionError("crossover did not produce a permutation")
    return board


def initial_population(
    components: list[dict[int, tuple[int, int]]],
    right: np.ndarray,
    down: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    population: list[np.ndarray] = []
    board, used = _place_components(components, right, down)
    population.append(_fill_board(board, used, right, down))
    for _ in range(size - 1):
        board, used = _place_components_randomized(
            components, right, down, rng,
            temperature=float(rng.uniform(0.01, 0.20)),
            order_jitter=float(rng.uniform(0.10, 0.80)),
        )
        population.append(_fill_board(board, used, right, down))
    return population


def evolve(
    right: np.ndarray,
    down: np.ndarray,
    *,
    population_size: int,
    generations: int,
    elite: int,
    max_edges: int,
    seed: int,
    truth: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    components = build_buddies_components(right, down, max_edges=max_edges)
    population = initial_population(components, right, down, population_size, rng)
    compatibility = Compatibility(right, down)
    history: list[dict[str, float]] = []
    for generation in range(generations + 1):
        scores = np.asarray([board_fitness(item, right, down) for item in population])
        order = np.argsort(-scores)
        population = [population[int(index)] for index in order]
        row: dict[str, float] = {
            "generation": float(generation),
            "fitness": float(scores[order[0]]),
            "unique": float(len({item.tobytes() for item in population})),
        }
        if truth is not None:
            row["placement"] = placement_accuracy(population[0], truth)[0]
            row["neighbour"] = neighbour_accuracy(population[0], truth)[0]
            row["population_best_neighbour"] = max(
                neighbour_accuracy(item, truth)[0] for item in population
            )
        history.append(row)
        print(json.dumps(row), flush=True)
        if generation == generations:
            break
        next_population = [item.copy() for item in population[:elite]]
        ranks = np.arange(population_size, 0, -1, dtype=np.float64)
        probability = ranks / ranks.sum()
        attempts = 0
        known = {item.tobytes() for item in next_population}
        while len(next_population) < population_size and attempts < population_size * 8:
            parent_ids = rng.choice(population_size, size=2, replace=False, p=probability)
            child = crossover(
                population[int(parent_ids[0])],
                population[int(parent_ids[1])],
                compatibility,
                rng,
            )
            attempts += 1
            key = child.tobytes()
            if key not in known:
                known.add(key)
                next_population.append(child)
        while len(next_population) < population_size:
            next_population.append(population[int(rng.integers(population_size))].copy())
        population = next_population
    return population[0], history


def load_graph(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stored = np.load(path)
    candidates = torch.from_numpy(stored["candidate_ids"]).long()
    scores = (
        torch.from_numpy(stored["candidate_scores"]).float()
        .reshape(NFRAG, 4, -1).permute(1, 0, 2).contiguous()
    )
    right, down = dense_rd(candidates, scores)
    truth = np.argsort(stored["permutation"].astype(np.int64))
    return right.numpy(), down.numpy(), truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--images", default="50")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--max-edges", type=int, default=384)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--report", type=Path,
        default=Path(WORK_ROOT) / "gates" / "genetic_solver_gate.json",
    )
    args = parser.parse_args()
    if not 1 <= args.elite < args.population:
        parser.error("--elite must be in [1, population)")
    rows = []
    for image in (int(value) for value in args.images.split(",")):
        right, down, truth = load_graph(args.cache_dir / f"image_{image:04d}_k64.npz")
        best, history = evolve(
            right, down, population_size=args.population, generations=args.generations,
            elite=args.elite, max_edges=args.max_edges, seed=args.seed + image, truth=truth,
        )
        rows.append(
            {
                "image": image,
                "placement": placement_accuracy(best, truth)[0],
                "neighbour": neighbour_accuracy(best, truth)[0],
                "initial_neighbour": history[0]["neighbour"],
                "population_oracle_initial": history[0]["population_best_neighbour"],
                "history": history,
            }
        )
    summary = {
        "placement": float(np.mean([row["placement"] for row in rows])),
        "neighbour": float(np.mean([row["neighbour"] for row in rows])),
        "initial_neighbour": float(np.mean([row["initial_neighbour"] for row in rows])),
    }
    report = {
        "experiment": "genetic_kernel_growing_solver",
        "images": rows,
        "summary": summary,
        "thresholds": {"neighbour": 0.20, "delta": 0.02},
        "passed": (
            summary["neighbour"] >= 0.20
            and summary["neighbour"] - summary["initial_neighbour"] >= 0.02
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
