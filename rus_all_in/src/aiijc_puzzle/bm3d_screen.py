"""Target-blind BM3D restoration arms and within-tile safety diagnostics."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.protocol import GRID_SIZE, IMAGE_SIZE, TILE_SIZE

ARM_NAMES = (
    "nlm_h20_baseline",
    "nlm_h28_reference",
    "bm3d_rgb_sigma_0_12",
    "bm3d_rgb_sigma_0_16",
    "bm3d_rgb_sigma_0_20",
    "bm3d_rgb_sigma_0_16_then_nlm_h10",
    "blend50_bm3d_sigma_0_16_nlm_h20",
)
CONTROL_ARM = ARM_NAMES[0]
REFERENCE_ARM = ARM_NAMES[1]
CANDIDATE_ARMS = ARM_NAMES[2:]

BM3D_SIGMA_BY_ARM = {
    "bm3d_rgb_sigma_0_12": 0.12,
    "bm3d_rgb_sigma_0_16": 0.16,
    "bm3d_rgb_sigma_0_20": 0.20,
}

BM3DCallable = Callable[..., np.ndarray]


def validate_image(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image with shape {expected}")
    return np.ascontiguousarray(value)


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(validate_image(image).tobytes()).hexdigest()


def apply_bm3d_rgb(
    image: np.ndarray,
    sigma: float,
    *,
    denoiser: BM3DCallable | None = None,
) -> np.ndarray:
    """Apply pinned two-stage BM3D RGB with the preregistered conversion."""

    source = validate_image(image)
    if sigma not in {0.12, 0.16, 0.20}:
        raise ValueError("sigma must be one of the three preregistered values")
    if denoiser is None:
        from bm3d import bm3d_rgb as denoiser  # type: ignore[import-not-found]

    restored = np.asarray(
        denoiser(
            source.astype(np.float64) / 255.0,
            sigma,
            profile="np",
            colorspace="opp",
        ),
        dtype=np.float64,
    )
    if restored.shape != source.shape or not np.isfinite(restored).all():
        raise RuntimeError("BM3D returned malformed or non-finite RGB output")
    return np.ascontiguousarray(
        np.rint(np.clip(restored, 0.0, 1.0) * 255.0).astype(np.uint8)
    )


def blend50_uint8(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the frozen half-up uint8 50/50 blend without overflow."""

    left = validate_image(first)
    right = validate_image(second)
    return np.ascontiguousarray(
        ((left.astype(np.uint16) + right.astype(np.uint16) + 1) // 2).astype(np.uint8)
    )


def render_arms(
    harmonized: np.ndarray,
    arm_names: Sequence[str] = ARM_NAMES,
    *,
    denoiser: BM3DCallable | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render an exact requested arm roster while reusing shared dependencies."""

    source = validate_image(harmonized)
    requested = tuple(arm_names)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("arm_names must be a non-empty unique roster")
    if any(name not in ARM_NAMES for name in requested):
        raise ValueError("arm_names contains a non-preregistered arm")

    cache: dict[str, np.ndarray] = {}
    runtime: dict[str, float] = {}

    def timed(name: str, function: Callable[[], np.ndarray]) -> np.ndarray:
        if name not in cache:
            started = perf_counter()
            cache[name] = validate_image(function())
            runtime[name] = perf_counter() - started
        return cache[name]

    if CONTROL_ARM in requested or "blend50_bm3d_sigma_0_16_nlm_h20" in requested:
        timed(CONTROL_ARM, lambda: apply_nlm_color(source, h=20).image)
    if REFERENCE_ARM in requested:
        timed(REFERENCE_ARM, lambda: apply_nlm_color(source, h=28).image)

    required_bm3d = {
        name for name in requested if name in BM3D_SIGMA_BY_ARM
    }
    if any(
        name in requested
        for name in (
            "bm3d_rgb_sigma_0_16_then_nlm_h10",
            "blend50_bm3d_sigma_0_16_nlm_h20",
        )
    ):
        required_bm3d.add("bm3d_rgb_sigma_0_16")
    for name in BM3D_SIGMA_BY_ARM:
        if name in required_bm3d:
            sigma = BM3D_SIGMA_BY_ARM[name]
            timed(name, lambda sigma=sigma: apply_bm3d_rgb(source, sigma, denoiser=denoiser))

    cascade = "bm3d_rgb_sigma_0_16_then_nlm_h10"
    if cascade in requested:
        timed(cascade, lambda: apply_nlm_color(cache["bm3d_rgb_sigma_0_16"], h=10).image)
    blend = "blend50_bm3d_sigma_0_16_nlm_h20"
    if blend in requested:
        timed(
            blend,
            lambda: blend50_uint8(cache["bm3d_rgb_sigma_0_16"], cache[CONTROL_ARM]),
        )

    predictions = {name: cache[name] for name in requested}
    return predictions, {
        "dependency_runtime_seconds": runtime,
        "requested_arm_names": list(requested),
        "bm3d_calls": len(required_bm3d),
    }


def _luma(image: np.ndarray) -> np.ndarray:
    value = validate_image(image).astype(np.float64)
    return 0.299 * value[..., 0] + 0.587 * value[..., 1] + 0.114 * value[..., 2]


def structure_diagnostics(image: np.ndarray) -> dict[str, float]:
    """Measure within-tile luma detail without counting 20px puzzle seams."""

    value = validate_image(image)
    luma = _luma(value)
    horizontal = np.abs(np.diff(luma, axis=1))
    vertical = np.abs(np.diff(luma, axis=0))
    horizontal_mask = np.arange(IMAGE_SIZE - 1) % TILE_SIZE != TILE_SIZE - 1
    vertical_mask = np.arange(IMAGE_SIZE - 1) % TILE_SIZE != TILE_SIZE - 1
    gradient = float(
        (horizontal[:, horizontal_mask].mean() + vertical[vertical_mask, :].mean()) / 2.0
    )

    tiles = (
        luma.reshape(GRID_SIZE, TILE_SIZE, GRID_SIZE, TILE_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(GRID_SIZE * GRID_SIZE, TILE_SIZE, TILE_SIZE)
    )
    laplacian = np.abs(
        -4.0 * tiles[:, 1:-1, 1:-1]
        + tiles[:, :-2, 1:-1]
        + tiles[:, 2:, 1:-1]
        + tiles[:, 1:-1, :-2]
        + tiles[:, 1:-1, 2:]
    )
    return {
        "within_tile_luma_gradient_mean_abs": gradient,
        "within_tile_luma_laplacian_mean_abs": float(laplacian.mean()),
        "clipped_fraction": float(np.mean((value == 0) | (value == 255))),
    }


def all_predictions_distinct(predictions: dict[str, np.ndarray]) -> bool:
    """Require every requested arm to have a distinct uncompressed pixel digest."""

    digests = [image_digest(predictions[name]) for name in predictions]
    return len(digests) == len(set(digests))
