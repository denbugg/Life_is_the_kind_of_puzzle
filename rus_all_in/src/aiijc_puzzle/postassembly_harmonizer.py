"""Target-blind analytic harmonizers for a fixed 24x24 tile layout.

This is a package-local port of the RGB-offset and bounded-luminance algorithms
from read-only historical branch ``origin/таска-говно`` at commit
``d6a82f82ceefa109ef706402712d03805bc9e880`` (source blob
``9d8d01c0f48d0e1473c1ff48285b06ab786a5dd8``).  The numerical algorithm and
frozen defaults are unchanged; only package imports were adapted.

The functions operate only on already ordered dirty tiles.  They never receive
a clean target, source identity, or layout label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from aiijc_puzzle.protocol import GRID_SIZE, TILE_COUNT, TILE_SIZE


@dataclass(frozen=True)
class SeamGraphConfig:
    """Frozen hyperparameters for the additive RGB seam-graph solve."""

    extrapolation_band: int = 3
    confidence_scale: float = 12.0
    confidence_floor: float = 0.05
    ridge: float = 0.20
    huber_delta: float = 4.0
    irls_steps: int = 4
    max_abs_offset: float = 12.0

    def validate(self) -> None:
        if not 2 <= self.extrapolation_band <= min(8, TILE_SIZE):
            raise ValueError("extrapolation_band must be in [2, 8]")
        if self.confidence_scale <= 0:
            raise ValueError("confidence_scale must be positive")
        if not 0 < self.confidence_floor <= 1:
            raise ValueError("confidence_floor must be in (0, 1]")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if self.irls_steps <= 0:
            raise ValueError("irls_steps must be positive")
        if self.max_abs_offset <= 0:
            raise ValueError("max_abs_offset must be positive")


@dataclass(frozen=True)
class LuminanceGainConfig:
    """Frozen hyperparameters for the separate bounded-gain experiment."""

    extrapolation_band: int = 3
    confidence_scale: float = 0.08
    confidence_floor: float = 0.05
    ridge: float = 0.50
    huber_delta: float = 0.025
    irls_steps: int = 4
    max_fractional_gain: float = 0.04
    luminance_floor: float = 12.0
    luminance_ceiling: float = 243.0

    def validate(self) -> None:
        if not 2 <= self.extrapolation_band <= min(8, TILE_SIZE):
            raise ValueError("extrapolation_band must be in [2, 8]")
        if self.confidence_scale <= 0:
            raise ValueError("confidence_scale must be positive")
        if not 0 < self.confidence_floor <= 1:
            raise ValueError("confidence_floor must be in (0, 1]")
        if self.ridge <= 0 or self.huber_delta <= 0 or self.irls_steps <= 0:
            raise ValueError("ridge, huber_delta and irls_steps must be positive")
        if not 0 < self.max_fractional_gain < 0.25:
            raise ValueError("max_fractional_gain must be in (0, 0.25)")
        if not 0 < self.luminance_floor < self.luminance_ceiling < 255:
            raise ValueError("invalid luminance validity interval")


DEFAULT_SEAM_GRAPH_CONFIG = SeamGraphConfig()
DEFAULT_LUMINANCE_GAIN_CONFIG = LuminanceGainConfig()


def _as_tiles(tiles: np.ndarray) -> np.ndarray:
    array = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if array.shape != expected:
        raise ValueError(f"expected {expected} ordered tiles, got {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"expected uint8 tiles, got {array.dtype}")
    return np.ascontiguousarray(array)


def _linear_boundary_projection(
    tile: np.ndarray,
    *,
    side: str,
    band: int,
) -> np.ndarray:
    """Project a band of pixel-centre values to the physical seam centre."""

    values = np.asarray(tile, dtype=np.float64)
    if values.shape != (TILE_SIZE, TILE_SIZE, 3):
        raise ValueError("tile must have shape 20x20x3")
    if side in {"left", "right"}:
        strip = (
            values[:, :band, :].transpose(1, 0, 2)
            if side == "left"
            else values[:, -band:, :].transpose(1, 0, 2)
        )
        coordinates = (
            np.arange(band, dtype=np.float64)
            if side == "left"
            else np.arange(TILE_SIZE - band, TILE_SIZE, dtype=np.float64)
        )
        boundary = -0.5 if side == "left" else TILE_SIZE - 0.5
    elif side in {"top", "bottom"}:
        strip = values[:band, :, :] if side == "top" else values[-band:, :, :]
        coordinates = (
            np.arange(band, dtype=np.float64)
            if side == "top"
            else np.arange(TILE_SIZE - band, TILE_SIZE, dtype=np.float64)
        )
        boundary = -0.5 if side == "top" else TILE_SIZE - 0.5
    else:
        raise ValueError("side must be left, right, top, or bottom")

    centred = coordinates - float(coordinates.mean())
    denominator = float(np.dot(centred, centred))
    slope = np.tensordot(centred, strip, axes=(0, 0)) / denominator
    intercept = strip.mean(axis=0)
    return intercept + (boundary - float(coordinates.mean())) * slope


def _grid_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first: list[int] = []
    second: list[int] = []
    direction: list[int] = []
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            tile = row * GRID_SIZE + column
            if column + 1 < GRID_SIZE:
                first.append(tile)
                second.append(tile + 1)
                direction.append(0)
            if row + 1 < GRID_SIZE:
                first.append(tile)
                second.append(tile + GRID_SIZE)
                direction.append(1)
    return (
        np.asarray(first, dtype=np.int32),
        np.asarray(second, dtype=np.int32),
        np.asarray(direction, dtype=np.int8),
    )


def _rgb_seam_equations(
    tiles: np.ndarray,
    config: SeamGraphConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first, second, direction = _grid_edges()
    desired = np.empty((len(first), 3), dtype=np.float64)
    confidence = np.empty(len(first), dtype=np.float64)
    for index, (left_or_top, right_or_bottom, axis) in enumerate(
        zip(first, second, direction, strict=True)
    ):
        if axis == 0:
            before = _linear_boundary_projection(
                tiles[left_or_top], side="right", band=config.extrapolation_band
            )
            after = _linear_boundary_projection(
                tiles[right_or_bottom], side="left", band=config.extrapolation_band
            )
        else:
            before = _linear_boundary_projection(
                tiles[left_or_top], side="bottom", band=config.extrapolation_band
            )
            after = _linear_boundary_projection(
                tiles[right_or_bottom], side="top", band=config.extrapolation_band
            )
        delta = before - after
        centre = np.median(delta, axis=0)
        desired[index] = centre
        residual = delta - centre[None, :]
        robust_spread = float(np.median(np.sqrt(np.mean(np.square(residual), axis=1))))
        confidence[index] = max(
            config.confidence_floor,
            1.0 / (1.0 + (robust_spread / config.confidence_scale) ** 2),
        )
    return first, second, desired, confidence


def _incidence(first: np.ndarray, second: np.ndarray) -> sparse.csr_matrix:
    rows = np.repeat(np.arange(len(first), dtype=np.int32), 2)
    columns = np.column_stack((first, second)).reshape(-1)
    data = np.tile(np.asarray([-1.0, 1.0], dtype=np.float64), len(first))
    return sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(first), TILE_COUNT), dtype=np.float64
    ).tocsr()


def _irls_graph_solve(
    first: np.ndarray,
    second: np.ndarray,
    desired: np.ndarray,
    confidence: np.ndarray,
    *,
    ridge: float,
    huber_delta: float,
    steps: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if desired.ndim == 1:
        desired = desired[:, None]
    incidence = _incidence(first, second)
    base_weight = np.asarray(confidence, dtype=np.float64)
    weight = base_weight.copy()
    solution = np.zeros((TILE_COUNT, desired.shape[1]), dtype=np.float64)
    residual_norm = np.zeros(len(first), dtype=np.float64)
    identity = sparse.identity(TILE_COUNT, dtype=np.float64, format="csr")
    for _ in range(steps):
        weighted = incidence.multiply(weight[:, None])
        normal = incidence.T @ weighted + ridge * identity
        right = incidence.T @ (weight[:, None] * desired)
        solution = np.asarray(spsolve(normal.tocsc(), right), dtype=np.float64)
        if solution.ndim == 1:
            solution = solution[:, None]
        solution -= np.median(solution, axis=0, keepdims=True)
        residual = incidence @ solution - desired
        residual_norm = np.sqrt(np.mean(np.square(residual), axis=1))
        huber = np.minimum(1.0, huber_delta / np.maximum(residual_norm, 1e-12))
        weight = base_weight * huber
    diagnostics = {
        "edge_count": float(len(first)),
        "confidence_mean": float(base_weight.mean()),
        "confidence_min": float(base_weight.min()),
        "final_weight_mean": float(weight.mean()),
        "residual_median": float(np.median(residual_norm)),
        "residual_q90": float(np.quantile(residual_norm, 0.90)),
    }
    return solution, diagnostics


def seam_graph_rgb_offsets(
    tiles: np.ndarray,
    config: SeamGraphConfig = DEFAULT_SEAM_GRAPH_CONFIG,
    *,
    placebo_seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate bounded per-tile additive RGB offsets."""

    ordered = _as_tiles(tiles)
    config.validate()
    first, second, desired, confidence = _rgb_seam_equations(ordered, config)
    placebo = placebo_seed is not None
    if placebo:
        rng = np.random.default_rng(placebo_seed)
        _, _, direction = _grid_edges()
        shuffled = np.arange(len(first))
        for axis in (0, 1):
            positions = np.flatnonzero(direction == axis)
            shuffled[positions] = rng.permutation(positions)
        desired = desired[shuffled]
        confidence = confidence[shuffled]
    offsets, solve = _irls_graph_solve(
        first,
        second,
        desired,
        confidence,
        ridge=config.ridge,
        huber_delta=config.huber_delta,
        steps=config.irls_steps,
    )
    offsets = np.clip(offsets, -config.max_abs_offset, config.max_abs_offset)
    diagnostics: dict[str, Any] = {
        **solve,
        "placebo": placebo,
        "placebo_seed": placebo_seed,
        "max_abs_offset": float(np.max(np.abs(offsets))),
        "mean_abs_offset": float(np.mean(np.abs(offsets))),
        "clipped_fraction": float(np.mean(np.abs(offsets) >= config.max_abs_offset - 1e-9)),
    }
    return offsets.astype(np.float32), diagnostics


def apply_rgb_offsets(tiles: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    ordered = _as_tiles(tiles)
    values = np.asarray(offsets, dtype=np.float64)
    if values.shape != (TILE_COUNT, 3) or not np.isfinite(values).all():
        raise ValueError("offsets must be finite with shape 576x3")
    corrected = ordered.astype(np.float64) + values[:, None, None, :]
    return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return np.tensordot(
        np.asarray(rgb, dtype=np.float64),
        np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64),
        axes=([-1], [0]),
    )


def seam_graph_luminance_gains(
    tiles: np.ndarray,
    config: LuminanceGainConfig = DEFAULT_LUMINANCE_GAIN_CONFIG,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a separate bounded multiplicative luminance correction."""

    ordered = _as_tiles(tiles)
    config.validate()
    first, second, direction = _grid_edges()
    desired = np.zeros(len(first), dtype=np.float64)
    confidence = np.full(len(first), config.confidence_floor, dtype=np.float64)
    valid_counts = np.zeros(len(first), dtype=np.int32)
    for index, (before_index, after_index, axis) in enumerate(
        zip(first, second, direction, strict=True)
    ):
        if axis == 0:
            before_rgb = _linear_boundary_projection(
                ordered[before_index], side="right", band=config.extrapolation_band
            )
            after_rgb = _linear_boundary_projection(
                ordered[after_index], side="left", band=config.extrapolation_band
            )
        else:
            before_rgb = _linear_boundary_projection(
                ordered[before_index], side="bottom", band=config.extrapolation_band
            )
            after_rgb = _linear_boundary_projection(
                ordered[after_index], side="top", band=config.extrapolation_band
            )
        before_y = _luminance(before_rgb)
        after_y = _luminance(after_rgb)
        valid = (
            (before_y >= config.luminance_floor)
            & (before_y <= config.luminance_ceiling)
            & (after_y >= config.luminance_floor)
            & (after_y <= config.luminance_ceiling)
        )
        valid_counts[index] = int(valid.sum())
        if valid_counts[index] < 4:
            continue
        values = np.log(np.maximum(before_y[valid], 1.0)) - np.log(np.maximum(after_y[valid], 1.0))
        centre = float(np.median(values))
        spread = float(np.median(np.abs(values - centre)))
        desired[index] = centre
        confidence[index] = max(
            config.confidence_floor,
            1.0 / (1.0 + (spread / config.confidence_scale) ** 2),
        )
    log_gain, solve = _irls_graph_solve(
        first,
        second,
        desired,
        confidence,
        ridge=config.ridge,
        huber_delta=config.huber_delta,
        steps=config.irls_steps,
    )
    bound_low = np.log(1.0 - config.max_fractional_gain)
    bound_high = np.log(1.0 + config.max_fractional_gain)
    log_gain = np.clip(log_gain[:, 0], bound_low, bound_high)
    gains = np.exp(log_gain)
    diagnostics: dict[str, Any] = {
        **solve,
        "valid_tangent_samples_mean": float(valid_counts.mean()),
        "gain_min": float(gains.min()),
        "gain_max": float(gains.max()),
        "mean_abs_fractional_gain": float(np.mean(np.abs(gains - 1.0))),
        "clipped_fraction": float(
            np.mean((log_gain <= bound_low + 1e-12) | (log_gain >= bound_high - 1e-12))
        ),
    }
    return gains.astype(np.float32), diagnostics


def apply_luminance_gains(tiles: np.ndarray, gains: np.ndarray) -> np.ndarray:
    ordered = _as_tiles(tiles)
    values = np.asarray(gains, dtype=np.float64)
    if values.shape != (TILE_COUNT,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("gains must be finite and positive with shape 576")
    corrected = ordered.astype(np.float64) * values[:, None, None, None]
    return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)


__all__ = [
    "LuminanceGainConfig",
    "SeamGraphConfig",
    "apply_luminance_gains",
    "apply_rgb_offsets",
    "seam_graph_luminance_gains",
    "seam_graph_rgb_offsets",
]
