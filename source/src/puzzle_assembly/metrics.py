"""Exact synthetic retrieval, layout, and image metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from puzzle_denoise_v2.tiles import merge_tiles_numpy

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE_COUNT, target_positions, true_neighbour_slots, validate_permutation


def _direction_retrieval(matrix: np.ndarray, truth: np.ndarray, ks: Iterable[int]) -> dict[str, float]:
    queries = np.flatnonzero(truth >= 0)
    if len(queries) != GRID * (GRID - 1):
        raise ValueError(f"expected 552 interior queries, got {len(queries)}")
    ranks = np.empty(len(queries), dtype=np.int32)
    for output_index, query in enumerate(queries.tolist()):
        order = np.argsort(matrix[query], kind="stable")
        rank = np.flatnonzero(order == truth[query])
        if len(rank) != 1:
            raise ValueError("true neighbour is missing from candidate ordering")
        ranks[output_index] = int(rank[0]) + 1
    metrics = {f"recall_at_{k}": float(np.mean(ranks <= k)) for k in ks}
    metrics.update(
        {
            "mrr": float(np.mean(1.0 / ranks)),
            "median_rank": float(np.median(ranks)),
            "q90_rank": float(np.quantile(ranks, 0.9)),
            "queries": int(len(queries)),
        }
    )
    return metrics


def retrieval_metrics(
    compatibility: CompatibilityMatrices,
    slot_to_target: np.ndarray,
    *,
    ks: tuple[int, ...] = (1, 5, 10, 20, 32),
) -> dict[str, dict[str, float]]:
    right_truth, down_truth = true_neighbour_slots(slot_to_target)
    right = _direction_retrieval(compatibility.right, right_truth, ks)
    down = _direction_retrieval(compatibility.down, down_truth, ks)
    combined = {
        key: float((right[key] + down[key]) / 2.0)
        for key in right
        if key != "queries"
    }
    combined["queries"] = int(right["queries"] + down["queries"])
    return {"right": right, "down": down, "combined": combined}


def _largest_correct_component(target_grid: np.ndarray) -> int:
    parent = np.arange(TILE_COUNT, dtype=np.int32)
    size = np.ones(TILE_COUNT, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if size[first_root] < size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size[second_root]

    for row in range(GRID):
        for column in range(GRID):
            position = row * GRID + column
            target = int(target_grid[row, column])
            if column + 1 < GRID:
                right_target = int(target_grid[row, column + 1])
                if target % GRID != GRID - 1 and right_target == target + 1:
                    union(position, position + 1)
            if row + 1 < GRID:
                down_target = int(target_grid[row + 1, column])
                if down_target == target + GRID:
                    union(position, position + GRID)
    roots = np.asarray([find(index) for index in range(TILE_COUNT)], dtype=np.int32)
    return int(np.bincount(roots, minlength=TILE_COUNT).max())


def layout_metrics(position_to_slot: np.ndarray, slot_to_target: np.ndarray) -> dict[str, float | bool]:
    position_to_slot = validate_permutation(position_to_slot, name="position_to_slot")
    slot_to_target = validate_permutation(slot_to_target, name="slot_to_target")
    placed_targets = target_positions(position_to_slot, slot_to_target)
    expected = np.arange(TILE_COUNT, dtype=np.int32)
    displacement = np.abs(placed_targets // GRID - expected // GRID) + np.abs(
        placed_targets % GRID - expected % GRID
    )
    target_grid = placed_targets.reshape(GRID, GRID)
    left = target_grid[:, :-1]
    right = target_grid[:, 1:]
    top = target_grid[:-1, :]
    bottom = target_grid[1:, :]
    right_correct = (left % GRID != GRID - 1) & (right == left + 1)
    down_correct = bottom == top + GRID

    boundary_positions = np.unique(
        np.concatenate(
            [
                np.arange(GRID),
                np.arange((GRID - 1) * GRID, TILE_COUNT),
                np.arange(0, TILE_COUNT, GRID),
                np.arange(GRID - 1, TILE_COUNT, GRID),
            ]
        )
    )
    corners = np.asarray([0, GRID - 1, (GRID - 1) * GRID, TILE_COUNT - 1])
    return {
        "valid_permutation": True,
        "position_accuracy": float(np.mean(placed_targets == expected)),
        "row_accuracy": float(np.mean(placed_targets // GRID == expected // GRID)),
        "column_accuracy": float(np.mean(placed_targets % GRID == expected % GRID)),
        "mean_manhattan": float(np.mean(displacement)),
        "median_manhattan": float(np.median(displacement)),
        "q90_manhattan": float(np.quantile(displacement, 0.9)),
        "within_one_manhattan": float(np.mean(displacement <= 1)),
        "right_adjacency": float(np.mean(right_correct)),
        "down_adjacency": float(np.mean(down_correct)),
        "combined_adjacency": float((np.mean(right_correct) + np.mean(down_correct)) / 2.0),
        "exact_solved": bool(np.array_equal(placed_targets, expected)),
        "boundary_position_accuracy": float(
            np.mean(placed_targets[boundary_positions] == expected[boundary_positions])
        ),
        "corner_position_accuracy": float(np.mean(placed_targets[corners] == expected[corners])),
        "largest_correct_component": _largest_correct_component(target_grid),
    }


def predicted_image_metrics(
    position_to_slot: np.ndarray,
    slot_tiles: np.ndarray,
    clean_target: np.ndarray,
) -> dict[str, float]:
    position_to_slot = validate_permutation(position_to_slot, name="position_to_slot")
    slot_tiles = np.asarray(slot_tiles)
    clean_target = np.asarray(clean_target)
    if slot_tiles.shape != (TILE_COUNT, 20, 20, 3) or slot_tiles.dtype != np.uint8:
        raise ValueError("slot_tiles must be uint8 576x20x20x3")
    if clean_target.shape != (480, 480, 3) or clean_target.dtype != np.uint8:
        raise ValueError("clean_target must be uint8 480x480x3")
    predicted = merge_tiles_numpy(slot_tiles[position_to_slot])
    return {
        "predicted_layout_ssim": float(
            structural_similarity(clean_target, predicted, channel_axis=2, data_range=255)
        ),
        "psnr": float(peak_signal_noise_ratio(clean_target, predicted, data_range=255)),
        "mae": float(np.mean(np.abs(predicted.astype(np.float32) - clean_target.astype(np.float32)))),
    }
