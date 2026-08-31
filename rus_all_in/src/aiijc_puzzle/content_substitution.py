"""Content-substitution probes for the puzzle's RGB SSIM metric.

The experiment deliberately operates on clean target images.  It replaces every
20x20 target tile with another tile from the same image and measures how much of
the official image-level SSIM survives.  This separates tolerance to visually
equivalent content from puzzle-reconstruction or restoration quality.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity

UInt8Image = NDArray[np.uint8]
FloatArray = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]
IntVector = NDArray[np.int64]


@dataclass(frozen=True)
class VariantEvaluation:
    """One substitution assignment and its measured output."""

    assignment: IntVector
    rendered: UInt8Image
    selected_rmse: FloatArray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class DirtyTileAlignment:
    """A target-position view of shuffled dirty input tiles and diagnostics."""

    aligned_tiles: UInt8Image
    target_to_input: IntVector
    descriptor_correlation: FloatArray
    aligned_rgb_rmse: FloatArray
    metrics: dict[str, Any]


def extract_tiles(
    image: UInt8Image,
    *,
    grid_size: int = 24,
    tile_size: int = 20,
) -> UInt8Image:
    """Extract row-major tiles from an RGB image without copying pixel values."""
    expected_shape = (grid_size * tile_size, grid_size * tile_size, 3)
    if image.shape != expected_shape:
        raise ValueError(f"expected image shape {expected_shape}, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {image.dtype}")

    return (
        image.reshape(grid_size, tile_size, grid_size, tile_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid_size * grid_size, tile_size, tile_size, 3)
    )


def render_tiles(
    tiles: UInt8Image,
    *,
    grid_size: int = 24,
) -> UInt8Image:
    """Render row-major RGB tiles back into a full image."""
    expected_tiles = grid_size * grid_size
    if tiles.ndim != 4 or tiles.shape[0] != expected_tiles or tiles.shape[-1] != 3:
        raise ValueError(f"expected ({expected_tiles}, tile, tile, 3) tiles, got {tiles.shape}")
    if tiles.shape[1] != tiles.shape[2]:
        raise ValueError(f"tiles must be square, got {tiles.shape[1:3]}")

    tile_size = tiles.shape[1]
    return (
        tiles.reshape(grid_size, grid_size, tile_size, tile_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid_size * tile_size, grid_size * tile_size, 3)
    )


def pairwise_tile_rmse(tiles: UInt8Image) -> FloatMatrix:
    """Compute the exact full-RGB RMSE between every pair of tiles.

    The Gram-matrix identity avoids materialising a
    ``(576, 576, 20, 20, 3)`` difference tensor.  At the contest dimensions the
    largest temporary is only the 576x576 distance matrix.
    """
    if tiles.ndim != 4 or tiles.shape[-1] != 3:
        raise ValueError(f"expected (n, height, width, 3) tiles, got {tiles.shape}")

    # Float64 is intentional: with bright near-identical tiles, float32 Gram
    # subtraction can lose squared distances smaller than roughly one ulp of a
    # ~1e8 norm.  Integer-valued RGB dot products are still exact in float64 at
    # these dimensions, while the 576x1200 multiplication remains inexpensive.
    flat = np.ascontiguousarray(tiles.reshape(tiles.shape[0], -1), dtype=np.float64)
    squared_norms = np.einsum("ij,ij->i", flat, flat)
    squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (flat @ flat.T)
    # Guard against generic floating-point round-off and enforce exact symmetry.
    squared_distances = (squared_distances + squared_distances.T) * 0.5
    np.maximum(squared_distances, 0.0, out=squared_distances)
    np.fill_diagonal(squared_distances, 0.0)
    rmse = np.sqrt(squared_distances / flat.shape[1])
    np.fill_diagonal(rmse, 0.0)
    return rmse


def _normalized_block_descriptors(tiles: UInt8Image, *, descriptor_grid: int) -> FloatMatrix:
    """Build degradation-robust per-tile descriptors used by the legacy aligner."""
    if tiles.ndim != 4 or tiles.shape[1] != tiles.shape[2] or tiles.shape[-1] != 3:
        raise ValueError(f"expected square RGB tiles, got {tiles.shape}")
    tile_size = tiles.shape[1]
    if descriptor_grid < 1 or tile_size % descriptor_grid:
        raise ValueError(f"descriptor_grid {descriptor_grid} must divide tile size {tile_size}")

    block_size = tile_size // descriptor_grid
    descriptors = (
        tiles.astype(np.float64)
        .reshape(
            tiles.shape[0],
            descriptor_grid,
            block_size,
            descriptor_grid,
            block_size,
            3,
        )
        .mean(axis=(2, 4))
        .reshape(tiles.shape[0], -1)
    )
    descriptor_mean = descriptors.mean(axis=1, keepdims=True)
    descriptor_std = descriptors.std(axis=1, keepdims=True)
    return (descriptors - descriptor_mean) / (descriptor_std + 1e-6)


def recover_dirty_tile_alignment(
    input_image: UInt8Image,
    target_image: UInt8Image,
    *,
    grid_size: int = 24,
    tile_size: int = 20,
    descriptor_grid: int = 5,
) -> DirtyTileAlignment:
    """Align shuffled dirty train tiles to target positions with a robust proxy.

    This reproduces the established train-pair recovery method: normalized 5x5
    block descriptors followed by a global Hungarian assignment.  The dataset
    does not expose permutation labels, so its descriptor statistics are
    diagnostics rather than a claim of exact alignment accuracy.
    """
    input_tiles = extract_tiles(input_image, grid_size=grid_size, tile_size=tile_size)
    target_tiles = extract_tiles(target_image, grid_size=grid_size, tile_size=tile_size)
    input_descriptors = _normalized_block_descriptors(input_tiles, descriptor_grid=descriptor_grid)
    target_descriptors = _normalized_block_descriptors(
        target_tiles, descriptor_grid=descriptor_grid
    )
    input_norms = np.einsum("ij,ij->i", input_descriptors, input_descriptors)
    target_norms = np.einsum("ij,ij->i", target_descriptors, target_descriptors)
    descriptor_costs = (
        input_norms[:, None]
        + target_norms[None, :]
        - 2.0 * (input_descriptors @ target_descriptors.T)
    )
    input_indices, target_indices = linear_sum_assignment(descriptor_costs)
    expected_indices = np.arange(input_tiles.shape[0], dtype=np.int64)
    if not np.array_equal(input_indices, expected_indices):
        raise RuntimeError("Hungarian aligner returned unexpected input row ordering")

    target_to_input = np.empty(input_tiles.shape[0], dtype=np.int64)
    target_to_input[target_indices] = input_indices
    aligned_tiles = input_tiles[target_to_input]
    descriptor_correlation = np.mean(
        input_descriptors[target_to_input] * target_descriptors,
        axis=1,
    )
    delta = aligned_tiles.astype(np.float64) - target_tiles.astype(np.float64)
    aligned_rgb_rmse = np.sqrt(np.mean(np.square(delta), axis=(1, 2, 3)))
    input_to_target = np.empty(input_tiles.shape[0], dtype=np.int64)
    input_to_target[input_indices] = target_indices
    rowwise_best = np.argmin(descriptor_costs, axis=1)
    assignment_digest = hashlib.sha256(target_to_input.astype("<i4").tobytes()).hexdigest()
    metrics = {
        "method": "normalized 5x5 block descriptor plus Hungarian assignment",
        "permutation_labels_available": False,
        "assigned_is_rowwise_best_fraction": float(np.mean(input_to_target == rowwise_best)),
        "descriptor_correlation_mean": float(np.mean(descriptor_correlation)),
        "descriptor_correlation_quantiles": _quantiles(descriptor_correlation),
        "aligned_rgb_rmse_mean": float(np.mean(aligned_rgb_rmse)),
        "aligned_rgb_rmse_quantiles": _quantiles(aligned_rgb_rmse),
        "target_to_input_sha256": assignment_digest,
    }
    return DirtyTileAlignment(
        aligned_tiles=aligned_tiles,
        target_to_input=target_to_input,
        descriptor_correlation=descriptor_correlation,
        aligned_rgb_rmse=aligned_rgb_rmse,
        metrics=metrics,
    )


def _derived_rng(seed: int, board_key: str, variant: str) -> np.random.Generator:
    payload = f"{seed}\0{board_key}\0{variant}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived_seed)


def _without_diagonal(costs: FloatMatrix) -> FloatMatrix:
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError(f"expected square costs, got {costs.shape}")
    if costs.shape[0] < 2:
        raise ValueError("a derangement needs at least two tiles")
    result = costs.copy()
    np.fill_diagonal(result, np.inf)
    return result


def build_assignments(
    costs: FloatMatrix,
    *,
    seed: int,
    board_key: str,
    nearest_ks: Iterable[int] = (3, 10),
) -> dict[str, IntVector]:
    """Build deterministic identity, independent, and bijective substitutions.

    Every non-identity variant excludes its tile's own index.  The nearest and
    random-k assignments choose independently and may reuse source tiles.  The
    Hungarian assignment is a minimum-total-RMSE bijection with the diagonal
    forbidden.
    """
    excluded = _without_diagonal(costs)
    n_tiles = costs.shape[0]
    rows = np.arange(n_tiles, dtype=np.int64)
    assignments: dict[str, IntVector] = {
        "identity": rows.copy(),
        "nearest_other": np.argmin(excluded, axis=1).astype(np.int64, copy=False),
    }

    row_indices, column_indices = linear_sum_assignment(excluded)
    if not np.array_equal(row_indices, rows):
        raise RuntimeError("Hungarian solver returned unexpected row ordering")
    assignments["bijective_derangement"] = column_indices.astype(np.int64, copy=False)

    random_other_rng = _derived_rng(seed, board_key, "random_other")
    random_other = random_other_rng.integers(0, n_tiles - 1, size=n_tiles, dtype=np.int64)
    random_other += random_other >= rows
    assignments["random_other"] = random_other

    requested_ks = tuple(dict.fromkeys(int(value) for value in nearest_ks))
    if any(value < 1 or value >= n_tiles for value in requested_ks):
        raise ValueError(f"nearest_ks must be in [1, {n_tiles - 1}], got {requested_ks}")
    if requested_ks:
        max_k = max(requested_ks)
        # Stable sorting gives deterministic tile-index tie breaking.
        nearest = np.argsort(excluded, axis=1, kind="stable")[:, :max_k]
        for k in requested_ks:
            variant = f"random_k{k}_nearest"
            rng = _derived_rng(seed, board_key, variant)
            selected_rank = rng.integers(0, k, size=n_tiles)
            assignments[variant] = nearest[rows, selected_rank].astype(np.int64, copy=False)

    for name, assignment in assignments.items():
        if assignment.shape != (n_tiles,):
            raise RuntimeError(f"{name} returned invalid shape {assignment.shape}")
        if name != "identity" and np.any(assignment == rows):
            raise RuntimeError(f"{name} violated the forbidden diagonal")
    return assignments


def contest_rgb_ssim(target: UInt8Image, prediction: UInt8Image) -> float:
    """Compute the contest's full-image RGB SSIM exactly as specified."""
    if target.shape != prediction.shape:
        raise ValueError(f"shape mismatch: target {target.shape}, prediction {prediction.shape}")
    return float(structural_similarity(target, prediction, channel_axis=2, data_range=255))


def _quantiles(values: NDArray[np.floating[Any]]) -> dict[str, float]:
    probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    quantiles = np.quantile(values, probabilities)
    return {
        f"q{round(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, quantiles, strict=True)
    }


def evaluate_variants(
    target: UInt8Image,
    costs: FloatMatrix,
    assignments: Mapping[str, IntVector],
    *,
    grid_size: int = 24,
    source_tiles: UInt8Image | None = None,
) -> dict[str, VariantEvaluation]:
    """Render and score all supplied substitutions for one target board."""
    tiles = extract_tiles(
        target,
        grid_size=grid_size,
        tile_size=target.shape[0] // grid_size,
    )
    n_tiles = tiles.shape[0]
    if costs.shape != (n_tiles, n_tiles):
        raise ValueError(f"cost matrix has shape {costs.shape}, expected {(n_tiles, n_tiles)}")
    if source_tiles is None:
        source_tiles = tiles
    if source_tiles.shape != tiles.shape or source_tiles.dtype != np.uint8:
        raise ValueError(
            f"source tiles must have shape {tiles.shape} and dtype uint8, "
            f"got {source_tiles.shape} and {source_tiles.dtype}"
        )
    rows = np.arange(n_tiles, dtype=np.int64)
    results: dict[str, VariantEvaluation] = {}

    for name, assignment in assignments.items():
        if assignment.shape != (n_tiles,):
            raise ValueError(
                f"{name} assignment has shape {assignment.shape}, expected {(n_tiles,)}"
            )
        if np.any((assignment < 0) | (assignment >= n_tiles)):
            raise ValueError(f"{name} assignment contains an out-of-range tile index")

        rendered = render_tiles(source_tiles[assignment], grid_size=grid_size)
        selected_rmse = costs[rows, assignment]
        unique_sources = int(np.unique(assignment).size)
        assignment_digest = hashlib.sha256(assignment.astype("<i4").tobytes()).hexdigest()
        metrics = {
            "ssim": contest_rgb_ssim(target, rendered),
            "exact_placement_count": int(np.count_nonzero(assignment == rows)),
            "exact_placement_fraction": float(np.mean(assignment == rows)),
            "unique_source_tiles": unique_sources,
            "duplicate_use_count": n_tiles - unique_sources,
            "selected_rmse_mean": float(np.mean(selected_rmse)),
            "selected_rmse_quantiles": _quantiles(selected_rmse),
            "assignment_sha256": assignment_digest,
        }
        results[name] = VariantEvaluation(
            assignment=assignment,
            rendered=rendered,
            selected_rmse=selected_rmse,
            metrics=metrics,
        )
    return results


def aggregate_evaluations(
    board_results: Sequence[Mapping[str, Mapping[str, Any]]],
    selected_rmse: Mapping[str, Sequence[FloatArray]],
) -> dict[str, dict[str, Any]]:
    """Aggregate board metrics and exact selected-tile RMSE distributions."""
    if not board_results:
        raise ValueError("cannot aggregate an empty experiment")

    variant_names = tuple(board_results[0])
    aggregate: dict[str, dict[str, Any]] = {}
    for variant in variant_names:
        metrics = [board[variant] for board in board_results]
        ssim_values = np.asarray([item["ssim"] for item in metrics], dtype=np.float64)
        placement_values = np.asarray(
            [item["exact_placement_fraction"] for item in metrics], dtype=np.float64
        )
        duplicate_values = np.asarray(
            [item["duplicate_use_count"] for item in metrics], dtype=np.float64
        )
        all_rmse = np.concatenate(selected_rmse[variant])
        variant_aggregate = {
            "board_count": len(metrics),
            "ssim_mean": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
            "ssim_min": float(np.min(ssim_values)),
            "ssim_max": float(np.max(ssim_values)),
            "exact_placement_fraction_mean": float(np.mean(placement_values)),
            "duplicate_use_count_mean": float(np.mean(duplicate_values)),
            "duplicate_use_count_min": int(np.min(duplicate_values)),
            "duplicate_use_count_max": int(np.max(duplicate_values)),
            "selected_rmse_mean": float(np.mean(all_rmse)),
            "selected_rmse_quantiles": _quantiles(all_rmse),
        }
        if all("tail_runtime_seconds" in item for item in metrics):
            tail_runtime = np.asarray(
                [item["tail_runtime_seconds"] for item in metrics], dtype=np.float64
            )
            variant_aggregate["tail_runtime_seconds_mean"] = float(np.mean(tail_runtime))
            variant_aggregate["tail_runtime_seconds_total"] = float(np.sum(tail_runtime))
        aggregate[variant] = variant_aggregate
    return aggregate


def aggregate_dirty_alignments(
    alignments: Sequence[DirtyTileAlignment],
) -> dict[str, Any]:
    """Aggregate alignment diagnostics without pretending labels are available."""
    if not alignments:
        raise ValueError("cannot aggregate an empty alignment list")
    correlations = np.concatenate([item.descriptor_correlation for item in alignments])
    rgb_rmse = np.concatenate([item.aligned_rgb_rmse for item in alignments])
    rowwise_best = np.asarray(
        [item.metrics["assigned_is_rowwise_best_fraction"] for item in alignments],
        dtype=np.float64,
    )
    return {
        "board_count": len(alignments),
        "permutation_labels_available": False,
        "assigned_is_rowwise_best_fraction_mean": float(np.mean(rowwise_best)),
        "descriptor_correlation_mean": float(np.mean(correlations)),
        "descriptor_correlation_quantiles": _quantiles(correlations),
        "aligned_rgb_rmse_mean": float(np.mean(rgb_rmse)),
        "aligned_rgb_rmse_quantiles": _quantiles(rgb_rmse),
    }


def select_target_paths(
    targets_dir: Path,
    *,
    count: int,
    seed: int,
    pool_start: int = 0,
    pool_stop: int | None = None,
) -> list[Path]:
    """Select a stable hash-ranked sample from a sorted target-file slice."""
    all_paths = sorted(targets_dir.glob("*.png"))
    stop = len(all_paths) if pool_stop is None else pool_stop
    pool = all_paths[pool_start:stop]
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if count > len(pool):
        raise ValueError(f"requested {count} boards from a pool of {len(pool)}")

    def rank(path: Path) -> tuple[bytes, str]:
        digest = hashlib.sha256(f"{seed}\0{path.name}".encode()).digest()
        return digest, path.name

    return sorted(pool, key=rank)[:count]
