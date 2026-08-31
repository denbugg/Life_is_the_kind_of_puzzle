"""Scene-analog layout from a permutation-invariant reference library.

This module implements an inference-visible global layout signal.  A shuffled
board is represented by a permutation-invariant distribution of full-tile
appearance features.  A ridge bridge maps dirty-board signatures into the
clean-reference signature domain, and nearest clean training scenes act as
spatial templates.  Several templates vote through a single averaged cost
matrix before a one-to-one Hungarian projection.

The mechanism intentionally does not inspect target pixels at inference and it
does not use boundary-only compatibility scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge

from aiijc_puzzle.protocol import GRID_SIZE, TILE_COUNT, TILE_SIZE, assemble_tiles, split_tiles

ROLE_DIM = 14
SIGNATURE_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def tile_semantic_features(tiles: np.ndarray) -> np.ndarray:
    """Return color-role, texture and normalized-shape features per tile.

    The first ``ROLE_DIM`` columns describe broad photometric/texture roles.
    The remaining 4x4 normalized luminance thumbnail encodes full-tile shape,
    rather than only the one-pixel seam.
    """

    array = np.asarray(tiles)
    if array.ndim != 4 or array.shape[1:] != (TILE_SIZE, TILE_SIZE, 3):
        raise ValueError(f"expected N x {TILE_SIZE} x {TILE_SIZE} x 3 tiles, got {array.shape}")
    pixels = array.astype(np.float32) / 255.0
    means = pixels.mean(axis=(1, 2))
    stds = pixels.std(axis=(1, 2))
    gray = cv2.cvtColor(
        pixels.reshape(-1, TILE_SIZE, 3),
        cv2.COLOR_RGB2GRAY,
    ).reshape(len(pixels), TILE_SIZE, TILE_SIZE)
    gx = np.empty_like(gray)
    gy = np.empty_like(gray)
    gx[:, :, 1:-1] = 0.5 * (gray[:, :, 2:] - gray[:, :, :-2])
    gx[:, :, 0] = gray[:, :, 1] - gray[:, :, 0]
    gx[:, :, -1] = gray[:, :, -1] - gray[:, :, -2]
    gy[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gy[:, 0] = gray[:, 1] - gray[:, 0]
    gy[:, -1] = gray[:, -1] - gray[:, -2]
    magnitude = np.sqrt(np.square(gx) + np.square(gy))
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    orientation = []
    for lower in np.linspace(0.0, np.pi, 5)[:-1]:
        upper = lower + np.pi / 4.0
        orientation.append((magnitude * ((angle >= lower) & (angle < upper))).mean((1, 2)))
    orientation_array = np.stack(orientation, axis=1)
    role = np.concatenate(
        (
            means,
            stds,
            gray.mean((1, 2), keepdims=False)[:, None],
            gray.std((1, 2), keepdims=False)[:, None],
            magnitude.mean((1, 2), keepdims=False)[:, None],
            magnitude.std((1, 2), keepdims=False)[:, None],
            orientation_array,
        ),
        axis=1,
    )
    thumbnails = np.stack(
        [cv2.resize(tile, (4, 4), interpolation=cv2.INTER_AREA) for tile in gray]
    ).reshape(len(gray), -1)
    thumbnails -= thumbnails.mean(axis=1, keepdims=True)
    thumbnails /= thumbnails.std(axis=1, keepdims=True) + 1e-6
    return np.concatenate((role, thumbnails), axis=1).astype(np.float32)


def percentile_ranks(features: np.ndarray) -> np.ndarray:
    """Rank each feature within its board, preserving row permutation equivariance."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError(f"expected at least two feature rows, got {values.shape}")
    order = np.argsort(values, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    columns = np.arange(values.shape[1])[None, :]
    ranks[order, columns] = np.arange(len(values), dtype=np.float32)[:, None]
    return ranks / float(len(values) - 1)


def board_signature(features: np.ndarray) -> np.ndarray:
    """Build a permutation-invariant distribution signature for one board."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a feature matrix, got {values.shape}")
    quantiles = np.quantile(values, SIGNATURE_QUANTILES, axis=0)
    moments = np.stack((values.mean(axis=0), values.std(axis=0)), axis=0)
    return np.concatenate((quantiles, moments), axis=0).reshape(-1).astype(np.float32)


@dataclass(frozen=True)
class SignatureBridge:
    """Dirty-to-clean board-signature ridge map with frozen normalizers."""

    dirty_mean: np.ndarray
    dirty_scale: np.ndarray
    clean_mean: np.ndarray
    clean_scale: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    alpha: float

    def transform(self, dirty_signatures: np.ndarray) -> np.ndarray:
        values = np.asarray(dirty_signatures, dtype=np.float32)
        normalized = (values - self.dirty_mean) / self.dirty_scale
        predicted = normalized @ self.coefficient.T + self.intercept
        return predicted * self.clean_scale + self.clean_mean


def fit_signature_bridge(
    dirty_signatures: np.ndarray,
    clean_signatures: np.ndarray,
    *,
    alpha: float = 10.0,
) -> SignatureBridge:
    """Fit a fixed ridge bridge on paired train boards only."""

    dirty = np.asarray(dirty_signatures, dtype=np.float32)
    clean = np.asarray(clean_signatures, dtype=np.float32)
    if dirty.shape != clean.shape or dirty.ndim != 2:
        raise ValueError(
            f"dirty/clean signatures must have equal 2-D shape: {dirty.shape}, {clean.shape}"
        )
    if len(dirty) < 2:
        raise ValueError("at least two paired signatures are required")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    dirty_mean = dirty.mean(axis=0)
    clean_mean = clean.mean(axis=0)
    dirty_scale = np.maximum(dirty.std(axis=0), 1e-5)
    clean_scale = np.maximum(clean.std(axis=0), 1e-5)
    dirty_z = (dirty - dirty_mean) / dirty_scale
    clean_z = (clean - clean_mean) / clean_scale
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(dirty_z, clean_z)
    return SignatureBridge(
        dirty_mean=dirty_mean,
        dirty_scale=dirty_scale,
        clean_mean=clean_mean,
        clean_scale=clean_scale,
        coefficient=np.asarray(model.coef_, dtype=np.float32),
        intercept=np.asarray(model.intercept_, dtype=np.float32),
        alpha=float(alpha),
    )


def retrieve_analogs(
    predicted_clean_signature: np.ndarray,
    library_clean_signatures: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest clean reference indices and standardized squared distances."""

    query = np.asarray(predicted_clean_signature, dtype=np.float32).reshape(-1)
    library = np.asarray(library_clean_signatures, dtype=np.float32)
    if library.ndim != 2 or library.shape[1] != len(query):
        raise ValueError(f"incompatible query/library shapes: {query.shape}, {library.shape}")
    if not 1 <= k <= len(library):
        raise ValueError(f"k must be in [1, {len(library)}], got {k}")
    scale = np.maximum(library.std(axis=0), 1e-5)
    distance = np.mean(np.square((library - query) / scale), axis=1)
    indices = np.argpartition(distance, k - 1)[:k]
    indices = indices[np.argsort(distance[indices])]
    return indices.astype(np.int64), distance[indices].astype(np.float32)


def analog_position_cost(query_features: np.ndarray, template_features: np.ndarray) -> np.ndarray:
    """Cost from every query tile to every spatial slot of one clean analog."""

    query = np.asarray(query_features, dtype=np.float32)
    template = np.asarray(template_features, dtype=np.float32)
    if query.shape != template.shape or query.ndim != 2:
        raise ValueError(
            f"query/template features must have equal 2-D shape: {query.shape}, {template.shape}"
        )
    if query.shape[1] <= ROLE_DIM:
        raise ValueError("feature matrix has no normalized-shape columns")

    query_role = percentile_ranks(query[:, :ROLE_DIM])
    template_role = percentile_ranks(template[:, :ROLE_DIM])
    role_cost = np.mean(
        np.square(query_role[:, None, :] - template_role[None, :, :]),
        axis=2,
    )
    query_shape = query[:, ROLE_DIM:]
    template_shape = template[:, ROLE_DIM:]
    query_shape /= np.linalg.norm(query_shape, axis=1, keepdims=True) + 1e-6
    template_shape /= np.linalg.norm(template_shape, axis=1, keepdims=True) + 1e-6
    shape_cost = 1.0 - query_shape @ template_shape.T
    result = 0.75 * role_cost + 0.25 * shape_cost
    return result.astype(np.float32)


def robust_row_scale(cost: np.ndarray) -> np.ndarray:
    """Normalize template costs per query tile before cross-template voting."""

    values = np.asarray(cost, dtype=np.float32)
    median = np.median(values, axis=1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=1, keepdims=True)
    return (values - median) / np.maximum(mad, 1e-5)


def consensus_layout(
    query_features: np.ndarray,
    template_features: np.ndarray,
    template_distances: np.ndarray,
    *,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Average analog spatial costs and project once to a global bijection.

    Returns ``(slot_to_query, consensus_cost)``.  Template order is irrelevant.
    """

    templates = np.asarray(template_features, dtype=np.float32)
    distances = np.asarray(template_distances, dtype=np.float32).reshape(-1)
    if templates.ndim != 3 or templates.shape[0] != len(distances):
        raise ValueError(
            f"template features/distances disagree: {templates.shape}, {distances.shape}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = -(distances - distances.min()) / temperature
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    consensus = np.zeros((len(query_features), len(query_features)), dtype=np.float32)
    for weight, features in zip(weights, templates, strict=True):
        consensus += float(weight) * robust_row_scale(
            analog_position_cost(query_features, features)
        )
    query_indices, slots = linear_sum_assignment(consensus)
    slot_to_query = np.empty(len(query_features), dtype=np.int64)
    slot_to_query[slots] = query_indices
    return slot_to_query, consensus


def render_layout(input_image: np.ndarray, slot_to_query: np.ndarray) -> np.ndarray:
    """Render a predicted one-to-one layout from the original dirty tiles."""

    mapping = np.asarray(slot_to_query)
    if mapping.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(mapping), np.arange(TILE_COUNT)
    ):
        raise ValueError("slot_to_query must be a permutation of all 576 tile indices")
    return assemble_tiles(split_tiles(input_image)[mapping])


def generic_template_features(library_features: np.ndarray) -> np.ndarray:
    """Average clean spatial template used as a no-retrieval control."""

    library = np.asarray(library_features, dtype=np.float32)
    if library.ndim != 3 or library.shape[1] != GRID_SIZE * GRID_SIZE:
        raise ValueError(f"expected L x {TILE_COUNT} x D library, got {library.shape}")
    return library.mean(axis=0)
