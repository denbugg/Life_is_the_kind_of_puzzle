"""Target-assisted diagnostics for neighbour candidate supply.

The functions in this module deliberately separate two concerns:

* candidate emitters see only the corrupted, shuffled tiles;
* the clean target is used afterwards to recover labels and score the emitted
  candidates.

That makes the resulting recall numbers suitable for deciding whether a more
expensive pair verifier has enough useful candidates to work with.  It does not
make the recovered labels or the clean target legal inference-time inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

GRID = 24
TILE_SIZE = 20
N_TILES = GRID * GRID

DEFAULT_K = (1, 5, 20, 32)
DEFAULT_RMSE_THRESHOLDS = (10.0, 20.0, 30.0)
DEFAULT_VIEWS = ("raw", "tile_z", "bilateral", "gray")

_DUMMY_GRADIENTS = np.asarray(
    [
        [0, 0, 0],
        [1, 1, 1],
        [-1, -1, -1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
    dtype=np.float32,
)


def split_tiles(image: np.ndarray, *, grid: int = GRID) -> np.ndarray:
    """Split an ``H x W x 3`` board into row-major square tiles."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got {image.shape}")
    if image.shape[0] != image.shape[1] or image.shape[0] % grid:
        raise ValueError(f"image shape {image.shape} is incompatible with grid={grid}")
    size = image.shape[0] // grid
    return (
        image.reshape(grid, size, grid, size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid * grid, size, size, 3)
    )


def blur3(tiles: np.ndarray) -> np.ndarray:
    """Apply the generator's separable 3x3 Gaussian to each tile."""
    x = np.asarray(tiles, dtype=np.float32)
    xp = np.pad(x, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="reflect")
    x = 0.25 * xp[:, :-2] + 0.5 * xp[:, 1:-1] + 0.25 * xp[:, 2:]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="reflect")
    return 0.25 * xp[:, :, :-2] + 0.5 * xp[:, :, 1:-1] + 0.25 * xp[:, :, 2:]


def normalised_descriptors(tiles: np.ndarray) -> np.ndarray:
    """Remove each tile's scalar brightness and contrast."""
    flat = np.asarray(tiles, dtype=np.float32).reshape(len(tiles), -1)
    return (flat - flat.mean(axis=1, keepdims=True)) / (flat.std(axis=1, keepdims=True) + 1e-6)


@dataclass(frozen=True)
class RecoveredLayout:
    """Target-assisted dirty-tile labels for one training board."""

    dirty_at_position: np.ndarray
    margin_at_position: np.ndarray

    @property
    def position_of_dirty(self) -> np.ndarray:
        result = np.empty(len(self.dirty_at_position), dtype=np.int64)
        result[self.dirty_at_position] = np.arange(len(result))
        return result


def recover_layout(dirty: np.ndarray, clean: np.ndarray) -> RecoveredLayout:
    """Recover a one-to-one target position for every corrupted tile.

    This is the label recovery used by the historical restoration pipeline:
    full-tile normalized descriptors, the known generator blur on the clean
    side, and a Hungarian assignment.  The margin is diagnostic confidence,
    not an inference feature.
    """
    if dirty.shape != clean.shape or dirty.ndim != 4:
        raise ValueError(
            f"dirty and clean tiles must have equal 4-D shapes: {dirty.shape}, {clean.shape}"
        )
    di = normalised_descriptors(dirty)
    target = normalised_descriptors(blur3(clean))
    cost = (
        np.square(di).sum(axis=1)[:, None]
        + np.square(target).sum(axis=1)[None, :]
        - 2.0 * di @ target.T
    )
    rows, columns = linear_sum_assignment(cost)
    dirty_at_position = np.empty(len(dirty), dtype=np.int64)
    dirty_at_position[columns] = rows

    chosen = cost[rows, columns]
    alternatives = cost.copy()
    alternatives[rows, columns] = np.inf
    next_best = alternatives.min(axis=1)
    # Compare the actual globally assigned column with that row's best
    # alternative.  The historical top1/top2-row gap was optimistic whenever
    # Hungarian selected a non-row-best column.
    margin_by_dirty = (next_best - chosen) / (np.abs(chosen) + 1e-6)
    margin_at_position = np.empty(len(dirty), dtype=np.float32)
    margin_at_position[columns] = margin_by_dirty[rows]
    return RecoveredLayout(dirty_at_position, margin_at_position)


def analytic_views(
    tiles: np.ndarray, names: Sequence[str] = DEFAULT_VIEWS
) -> dict[str, np.ndarray]:
    """Build diverse, inference-visible views for classical edge emitters."""
    source = np.clip(tiles, 0, 255).astype(np.uint8)
    result: dict[str, np.ndarray] = {}
    for name in names:
        if name == "raw":
            view = source.astype(np.float32)
        elif name == "tile_z":
            flat = normalised_descriptors(source)
            view = (32.0 * flat + 128.0).reshape(source.shape)
        elif name == "bilateral":
            view = np.stack([cv2.bilateralFilter(tile, 5, 25, 5) for tile in source]).astype(
                np.float32
            )
        elif name == "median":
            view = np.stack([cv2.medianBlur(tile, 3) for tile in source]).astype(np.float32)
        elif name == "gray":
            gray = np.stack([cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY) for tile in source])
            view = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
        else:
            raise ValueError(f"unknown analytic view: {name}")
        result[name] = view
    return result


def _mahalanobis_gradient_cost(
    source_boundary: np.ndarray,
    source_inner: np.ndarray,
    target_boundary: np.ndarray,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Asymmetric Mahalanobis Gradient Compatibility for all tile pairs."""
    source_boundary = np.asarray(source_boundary, dtype=np.float32)
    source_inner = np.asarray(source_inner, dtype=np.float32)
    target_boundary = np.asarray(target_boundary, dtype=np.float32)
    n = len(source_boundary)
    gradients = source_boundary - source_inner
    means = gradients.mean(axis=1)
    dummy = np.broadcast_to(_DUMMY_GRADIENTS, (n, *_DUMMY_GRADIENTS.shape))
    samples = np.concatenate((gradients, dummy), axis=1).astype(np.float64)
    centered = samples - samples.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True)
    covariance /= samples.shape[1] - 1
    precisions = np.linalg.inv(covariance).astype(np.float32)

    costs = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        residual = (
            target_boundary[None, :, :, :]
            - source_boundary[start:stop, None, :, :]
            - means[start:stop, None, None, :]
        )
        costs[start:stop] = np.einsum(
            "btkc,bcd,btkd->bt",
            residual,
            precisions[start:stop],
            residual,
            optimize=True,
        )
    return costs


def _ssd_cost(
    source_boundary: np.ndarray,
    target_boundary: np.ndarray,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    n = len(source_boundary)
    source_boundary = np.asarray(source_boundary, dtype=np.float32)
    target_boundary = np.asarray(target_boundary, dtype=np.float32)
    costs = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        residual = source_boundary[start:stop, None] - target_boundary[None]
        costs[start:stop] = np.einsum("btkc,btkc->bt", residual, residual, optimize=True)
    return costs


def _row_robust(cost: np.ndarray) -> np.ndarray:
    """Put pair costs on a per-anchor median/MAD scale."""
    cost = np.asarray(cost, dtype=np.float32)
    n = len(cost)
    off_diagonal = cost[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    median = np.median(off_diagonal, axis=1, keepdims=True)
    mad = np.median(np.abs(off_diagonal - median), axis=1, keepdims=True)
    scaled = (cost - median) / np.maximum(mad, 1e-6)
    np.fill_diagonal(scaled, np.inf)
    return scaled


def classical_costs(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed 50/50 MGC+one-pixel-SSD right/down dissimilarities."""
    pixels = np.asarray(tiles, dtype=np.float32)
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"expected N x H x W x 3 tiles, got {pixels.shape}")
    left, left_inner = pixels[:, :, 0], pixels[:, :, 1]
    right, right_inner = pixels[:, :, -1], pixels[:, :, -2]
    top, top_inner = pixels[:, 0], pixels[:, 1]
    bottom, bottom_inner = pixels[:, -1], pixels[:, -2]

    right_mgc = _mahalanobis_gradient_cost(right, right_inner, left)
    right_mgc += _mahalanobis_gradient_cost(left, left_inner, right).T
    down_mgc = _mahalanobis_gradient_cost(bottom, bottom_inner, top)
    down_mgc += _mahalanobis_gradient_cost(top, top_inner, bottom).T
    right_ssd = _ssd_cost(right, left)
    down_ssd = _ssd_cost(bottom, top)
    return (
        0.5 * (_row_robust(right_mgc) + _row_robust(right_ssd)),
        0.5 * (_row_robust(down_mgc) + _row_robust(down_ssd)),
    )


def top_candidates(cost: np.ndarray, k: int) -> np.ndarray:
    """Return each row's best ``k`` candidates, ordered by ascending cost."""
    cost = np.asarray(cost)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError(f"expected a square cost matrix, got {cost.shape}")
    if not 1 <= k < len(cost):
        raise ValueError(f"k must be in [1, {len(cost) - 1}], got {k}")
    pool = np.argpartition(cost, kth=k - 1, axis=1)[:, :k]
    values = np.take_along_axis(cost, pool, axis=1)
    order = np.argsort(values, axis=1)
    return np.take_along_axis(pool, order, axis=1)


def _candidate_rmse(
    clean_flat: np.ndarray,
    true_position: int,
    candidate_positions: np.ndarray,
) -> np.ndarray:
    delta = clean_flat[candidate_positions] - clean_flat[true_position]
    return np.sqrt(np.mean(np.square(delta), axis=1))


def _empty_record(emitter: str, scope: str, direction: str, k: int) -> dict[str, object]:
    return {
        "emitter": emitter,
        "scope": scope,
        "direction": direction,
        "k": k,
        "edge_count": 0,
        "candidate_count_sum": 0,
        "content_candidate_count_sum": 0,
        "exact_hits": 0,
        "best_rmse_sum": 0.0,
        "rmse_hits": {},
    }


def _update_record(
    record: dict[str, object],
    *,
    candidates: np.ndarray,
    true_dirty: int,
    content_candidate_rmse: np.ndarray,
    thresholds: Sequence[float],
) -> None:
    record["edge_count"] = int(record["edge_count"]) + 1
    record["candidate_count_sum"] = int(record["candidate_count_sum"]) + len(candidates)
    record["content_candidate_count_sum"] = int(record["content_candidate_count_sum"]) + len(
        content_candidate_rmse
    )
    record["exact_hits"] = int(record["exact_hits"]) + int(np.any(candidates == true_dirty))
    # 255 is the maximum possible uint8 RGB RMSE and makes an edge with no
    # independently trusted candidate an explicit miss rather than dropping it.
    best_rmse = float(content_candidate_rmse.min()) if len(content_candidate_rmse) else 255.0
    record["best_rmse_sum"] = float(record["best_rmse_sum"]) + best_rmse
    hits = record["rmse_hits"]
    assert isinstance(hits, dict)
    for threshold in thresholds:
        key = f"le_{threshold:g}"
        hits[key] = int(hits.get(key, 0)) + int(best_rmse <= threshold)


def evaluate_board(
    dirty: np.ndarray,
    clean: np.ndarray,
    *,
    views: Sequence[str] = DEFAULT_VIEWS,
    ks: Sequence[int] = DEFAULT_K,
    rmse_thresholds: Sequence[float] = DEFAULT_RMSE_THRESHOLDS,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Evaluate exact and content-aware candidate recall for one train board.

    The ``union`` emitter is the all-emitter oracle pool: at each ``k`` it is
    the union of the first ``k`` candidates from every individual view.  Its
    average candidate count is reported, so its higher recall is not confused
    with a fixed-budget comparison.
    """
    dirty = np.asarray(dirty)
    clean = np.asarray(clean)
    if dirty.shape != clean.shape:
        raise ValueError(f"dirty and clean shapes differ: {dirty.shape} vs {clean.shape}")
    n = len(dirty)
    grid = round(n**0.5)
    if grid * grid != n:
        raise ValueError(f"tile count must be a square, got {n}")
    if not ks or min(ks) < 1 or max(ks) >= n:
        raise ValueError(f"invalid ks={ks} for {n} tiles")

    recovered = recover_layout(dirty, clean)
    position_of_dirty = recovered.position_of_dirty
    clean_flat = clean.astype(np.float32).reshape(n, -1)
    confidence_cut = float(np.median(recovered.margin_at_position))

    max_k = max(ks)
    ranked: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, view in analytic_views(dirty, views).items():
        right_cost, down_cost = classical_costs(view)
        ranked[name] = (top_candidates(right_cost, max_k), top_candidates(down_cost, max_k))

    records: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for direction, delta, axis in (("right", 1, 0), ("down", grid, 1)):
        positions = [
            position
            for position in range(n)
            if (direction == "right" and position % grid != grid - 1)
            or (direction == "down" and position < n - grid)
        ]
        for position in positions:
            neighbour_position = position + delta
            anchor_dirty = int(recovered.dirty_at_position[position])
            true_dirty = int(recovered.dirty_at_position[neighbour_position])
            trusted = (
                recovered.margin_at_position[position] >= confidence_cut
                and recovered.margin_at_position[neighbour_position] >= confidence_cut
            )
            for k in ks:
                emitted: dict[str, np.ndarray] = {
                    name: rank_pair[axis][anchor_dirty, :k] for name, rank_pair in ranked.items()
                }
                emitted["union"] = np.unique(np.concatenate(list(emitted.values())))
                for emitter, candidates in emitted.items():
                    candidate_positions = position_of_dirty[candidates]
                    candidate_rmse = _candidate_rmse(
                        clean_flat, neighbour_position, candidate_positions
                    )
                    candidate_is_trusted = (
                        recovered.margin_at_position[candidate_positions] >= confidence_cut
                    )
                    for scope in ("all", "trusted_query", "trusted"):
                        if scope != "all" and not trusted:
                            continue
                        content_candidate_rmse = (
                            candidate_rmse[candidate_is_trusted]
                            if scope == "trusted"
                            else candidate_rmse
                        )
                        key = (emitter, scope, direction, k)
                        record = records.setdefault(key, _empty_record(*key))
                        _update_record(
                            record,
                            candidates=candidates,
                            true_dirty=true_dirty,
                            content_candidate_rmse=content_candidate_rmse,
                            thresholds=rmse_thresholds,
                        )

    mapping = {
        "median_margin": confidence_cut,
        "mean_margin": float(recovered.margin_at_position.mean()),
        "trusted_position_fraction": float(np.mean(recovered.margin_at_position >= confidence_cut)),
    }
    return list(records.values()), mapping


def merge_records(record_groups: Iterable[Iterable[dict[str, object]]]) -> list[dict[str, object]]:
    """Merge per-board sufficient statistics and add human-readable rates."""
    merged: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for records in record_groups:
        for incoming in records:
            key = (
                str(incoming["emitter"]),
                str(incoming["scope"]),
                str(incoming["direction"]),
                int(incoming["k"]),
            )
            target = merged.setdefault(key, _empty_record(*key))
            for field in (
                "edge_count",
                "candidate_count_sum",
                "content_candidate_count_sum",
                "exact_hits",
                "best_rmse_sum",
            ):
                target[field] = target[field] + incoming[field]  # type: ignore[operator]
            target_hits = target["rmse_hits"]
            incoming_hits = incoming["rmse_hits"]
            assert isinstance(target_hits, dict) and isinstance(incoming_hits, dict)
            for threshold, count in incoming_hits.items():
                target_hits[threshold] = int(target_hits.get(threshold, 0)) + int(count)

    output: list[dict[str, object]] = []
    for key in sorted(merged):
        record = merged[key]
        count = int(record["edge_count"])
        if not count:
            continue
        hits = record.pop("rmse_hits")
        assert isinstance(hits, dict)
        record["mean_candidates"] = float(record.pop("candidate_count_sum")) / count
        record["mean_content_candidates"] = float(record.pop("content_candidate_count_sum")) / count
        record["exact_recall"] = float(record.pop("exact_hits")) / count
        record["mean_best_rmse"] = float(record.pop("best_rmse_sum")) / count
        for threshold, hit_count in sorted(hits.items()):
            record[f"content_recall_rmse_{threshold}"] = int(hit_count) / count
        output.append(record)
    return output
