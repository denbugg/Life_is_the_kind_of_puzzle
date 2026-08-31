"""Target-assisted fixed-layout diagnostics and classical pixel tails.

The puzzle's training targets make it possible to recover an approximate true
layout without using a learned solver.  That recovery is an *oracle diagnostic*:
targets are never available at test time.  Keeping this code separate from the
solver makes it useful for answering a narrower question: once layout is held
fixed, which inexpensive image-space post-process improves contest RGB SSIM?
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t
from skimage.metrics import structural_similarity

GRID_SIZE = 24
TILE_SIZE = 20
TILE_COUNT = GRID_SIZE * GRID_SIZE
IMAGE_SIZE = GRID_SIZE * TILE_SIZE

NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21
GRAY_MEAN_RANGE = 10.0
GRAY_STD_LIMIT = 25.0


@dataclass(frozen=True)
class LayoutRecovery:
    """One-to-one target-assisted mapping from shuffled input tiles to slots."""

    input_to_slot: np.ndarray
    slot_to_input: np.ndarray
    confidence_by_slot: np.ndarray
    row_margin_by_slot: np.ndarray
    mutual_nearest_by_slot: np.ndarray
    descriptor_bins: int

    def diagnostics(self) -> dict[str, float | int]:
        """Return JSON-ready confidence diagnostics for this assignment."""
        return {
            "descriptor_bins": self.descriptor_bins,
            "mean_confidence": float(self.confidence_by_slot.mean()),
            "median_confidence": float(np.median(self.confidence_by_slot)),
            "minimum_confidence": float(self.confidence_by_slot.min()),
            "mean_row_margin": float(self.row_margin_by_slot.mean()),
            "median_row_margin": float(np.median(self.row_margin_by_slot)),
            "positive_row_margin_fraction": float((self.row_margin_by_slot > 0).mean()),
            "mutual_nearest_fraction": float(self.mutual_nearest_by_slot.mean()),
        }


@dataclass(frozen=True)
class TimedImage:
    """A post-processed image and the wall time required to obtain it."""

    image: np.ndarray
    seconds: float
    deployable: bool = True


def validate_rgb(image: np.ndarray) -> np.ndarray:
    """Validate one contest-shaped RGB image without copying it."""
    image = np.asarray(image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, 3)
    if image.shape != expected:
        raise ValueError(f"expected RGB image with shape {expected}, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {image.dtype}")
    return image


def split_tiles(image: np.ndarray) -> np.ndarray:
    """Split a contest image into row-major ``(576, 20, 20, 3)`` tiles."""
    image = validate_rgb(image)
    return (
        image.reshape(GRID_SIZE, TILE_SIZE, GRID_SIZE, TILE_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    )


def assemble_tiles(tiles: np.ndarray, slot_to_input: np.ndarray | None = None) -> np.ndarray:
    """Assemble row-major tiles, optionally selecting an input tile for each slot."""
    tiles = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if tiles.shape != expected or tiles.dtype != np.uint8:
        raise ValueError(f"expected uint8 tiles with shape {expected}, got {tiles.shape}")
    if slot_to_input is not None:
        slot_to_input = np.asarray(slot_to_input, dtype=np.int64)
        if slot_to_input.shape != (TILE_COUNT,):
            raise ValueError(f"expected {TILE_COUNT} slot indices, got {slot_to_input.shape}")
        if not np.array_equal(np.sort(slot_to_input), np.arange(TILE_COUNT)):
            raise ValueError("slot_to_input must be a permutation")
        tiles = tiles[slot_to_input]
    return (
        tiles.reshape(GRID_SIZE, GRID_SIZE, TILE_SIZE, TILE_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    )


def normalized_grid_descriptors(tiles: np.ndarray, bins: int = 5) -> np.ndarray:
    """Build brightness/contrast-invariant block-mean descriptors.

    ``bins=5`` reproduces the robust descriptor used by the original recovery
    pipeline: each feature averages a 4x4 region before the whole RGB vector is
    standardized.  ``bins=20`` is retained as an independent, more
    noise-sensitive full-resolution reference for the bakeoff.
    """
    tiles = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if tiles.shape != expected:
        raise ValueError(f"expected tiles with shape {expected}, got {tiles.shape}")
    if bins <= 0 or TILE_SIZE % bins:
        raise ValueError(f"bins must be a positive divisor of {TILE_SIZE}, got {bins}")

    block = TILE_SIZE // bins
    descriptor = (
        tiles.astype(np.float32)
        .reshape(TILE_COUNT, bins, block, bins, block, 3)
        .mean(axis=(2, 4))
        .reshape(TILE_COUNT, -1)
    )
    descriptor -= descriptor.mean(axis=1, keepdims=True)
    descriptor /= descriptor.std(axis=1, keepdims=True) + 1e-6
    return descriptor


def recover_layout(
    input_image: np.ndarray,
    target_image: np.ndarray,
    *,
    descriptor_bins: int = 5,
) -> LayoutRecovery:
    """Recover a target-assisted bijection with normalized descriptors + Hungarian."""
    input_tiles = split_tiles(input_image)
    target_tiles = split_tiles(target_image)
    input_desc = normalized_grid_descriptors(input_tiles, bins=descriptor_bins)
    target_desc = normalized_grid_descriptors(target_tiles, bins=descriptor_bins)

    # All standardized vectors have the same norm, so Euclidean assignment and
    # maximum normalized correlation are equivalent.  Correlation is easier to
    # interpret and keeps diagnostics on a stable [-1, 1] scale.
    dimension = input_desc.shape[1]
    correlation = (input_desc @ target_desc.T) / dimension
    cost = 1.0 - correlation
    input_indices, target_slots = linear_sum_assignment(cost)

    input_to_slot = np.empty(TILE_COUNT, dtype=np.int32)
    input_to_slot[input_indices] = target_slots
    slot_to_input = np.empty(TILE_COUNT, dtype=np.int32)
    slot_to_input[target_slots] = input_indices

    chosen_cost = cost[input_indices, target_slots]
    alternative = cost.copy()
    alternative[input_indices, target_slots] = np.inf
    row_margin_input = alternative.min(axis=1) - chosen_cost
    confidence_input = correlation[input_indices, target_slots]
    mutual_input = (target_slots == cost.argmin(axis=1)) & (
        input_indices == cost.argmin(axis=0)[target_slots]
    )

    confidence_by_slot = np.empty(TILE_COUNT, dtype=np.float32)
    confidence_by_slot[target_slots] = confidence_input
    row_margin_by_slot = np.empty(TILE_COUNT, dtype=np.float32)
    row_margin_by_slot[target_slots] = row_margin_input
    mutual_nearest_by_slot = np.empty(TILE_COUNT, dtype=bool)
    mutual_nearest_by_slot[target_slots] = mutual_input
    return LayoutRecovery(
        input_to_slot=input_to_slot,
        slot_to_input=slot_to_input,
        confidence_by_slot=confidence_by_slot,
        row_margin_by_slot=row_margin_by_slot,
        mutual_nearest_by_slot=mutual_nearest_by_slot,
        descriptor_bins=descriptor_bins,
    )


def contest_ssim(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute the contest's full-image RGB SSIM exactly."""
    target = validate_rgb(target)
    prediction = validate_rgb(prediction)
    return float(structural_similarity(target, prediction, channel_axis=2, data_range=255))


def gray_cell_mask(image: np.ndarray) -> np.ndarray:
    """Frozen E18b low-variance, nearly-achromatic cell definition."""
    tiles = split_tiles(image)
    channel_means = tiles.mean(axis=(1, 2))
    overall_std = tiles.std(axis=(1, 2, 3))
    channel_range = np.ptp(channel_means, axis=1)
    return (channel_range < GRAY_MEAN_RANGE) & (overall_std < GRAY_STD_LIMIT)


def no_new_gray_guard(raw: np.ndarray, filtered: np.ndarray) -> tuple[np.ndarray, int]:
    """Revert cells classified gray only after filtering (the exact E18b guard)."""
    raw = validate_rgb(raw)
    filtered = validate_rgb(filtered)
    revert = gray_cell_mask(filtered) & ~gray_cell_mask(raw)
    guarded_tiles = split_tiles(filtered).copy()
    guarded_tiles[revert] = split_tiles(raw)[revert]
    return assemble_tiles(guarded_tiles), int(revert.sum())


def oracle_cell_fallback(
    raw: np.ndarray,
    filtered: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Target-leaking cellwise raw/filter chooser used only as a headroom diagnostic.

    This is not a strict upper bound on full-canvas SSIM because SSIM windows
    cross tile boundaries.  It is deliberately marked non-deployable by
    :func:`pixel_tail_variants`.
    """
    raw_tiles = split_tiles(raw)
    filtered_tiles = split_tiles(filtered).copy()
    target_tiles = split_tiles(target)
    reverted = np.zeros(TILE_COUNT, dtype=bool)
    for index in range(TILE_COUNT):
        raw_score = structural_similarity(
            target_tiles[index], raw_tiles[index], channel_axis=2, data_range=255
        )
        filtered_score = structural_similarity(
            target_tiles[index], filtered_tiles[index], channel_axis=2, data_range=255
        )
        if raw_score > filtered_score:
            filtered_tiles[index] = raw_tiles[index]
            reverted[index] = True
    return assemble_tiles(filtered_tiles), int(reverted.sum())


def _time_image(function: Any, *args: Any, deployable: bool = True) -> TimedImage:
    started = perf_counter()
    image = validate_rgb(function(*args))
    return TimedImage(image=image, seconds=perf_counter() - started, deployable=deployable)


def _nlm_color(rgb: np.ndarray, h: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filtered = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h,
        h,
        NLM_TEMPLATE_WINDOW,
        NLM_SEARCH_WINDOW,
    )
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)


def apply_nlm_color(image: np.ndarray, h: int = 9) -> TimedImage:
    """Apply only the winning colored-NLM tail and report transform wall time.

    This lightweight entrypoint avoids computing the full bakeoff roster when
    another experiment needs the frozen h=9 tail on many assembled variants.
    """
    image = validate_rgb(image)
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    return _time_image(_nlm_color, image, h)


def _nlm_gray(rgb: np.ndarray, h: int = 9) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    filtered = cv2.fastNlMeansDenoising(
        gray,
        None,
        h,
        NLM_TEMPLATE_WINDOW,
        NLM_SEARCH_WINDOW,
    )
    return cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)


def _nlm_luma(rgb: np.ndarray, h: int = 9) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.fastNlMeansDenoising(
        ycrcb[..., 0],
        None,
        h,
        NLM_TEMPLATE_WINDOW,
        NLM_SEARCH_WINDOW,
    )
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _bilateral(rgb: np.ndarray, sigma: int) -> np.ndarray:
    return cv2.bilateralFilter(rgb, 5, sigma, sigma)


def _gaussian(rgb: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)


def pixel_tail_variants(
    raw: np.ndarray,
    *,
    target_for_oracle: np.ndarray | None = None,
) -> tuple[dict[str, TimedImage], dict[str, int]]:
    """Apply the fixed classical tail roster without changing the layout.

    Colored NLM outputs are reused by their gray guards, while each guarded
    runtime includes both filtering and guard time to reflect standalone use.
    """
    raw = validate_rgb(raw)
    variants: dict[str, TimedImage] = {
        "raw": TimedImage(raw.copy(), 0.0),
    }
    audit: dict[str, int] = {"raw_gray_cells": int(gray_cell_mask(raw).sum())}

    for h in (3, 5, 7, 9):
        name = f"nlm_color_h{h}"
        variants[name] = _time_image(_nlm_color, raw, h)
        guard_started = perf_counter()
        guarded, reverted = no_new_gray_guard(raw, variants[name].image)
        guard_seconds = perf_counter() - guard_started
        variants[f"{name}_gray_guard"] = TimedImage(
            image=guarded,
            seconds=variants[name].seconds + guard_seconds,
        )
        audit[f"{name}_new_gray_reverted"] = reverted

    variants["nlm_gray_h9"] = _time_image(_nlm_gray, raw, 9)
    variants["nlm_luma_h9"] = _time_image(_nlm_luma, raw, 9)
    variants["bilateral_s15"] = _time_image(_bilateral, raw, 15)
    variants["bilateral_s25"] = _time_image(_bilateral, raw, 25)
    variants["gaussian_sigma0.5"] = _time_image(_gaussian, raw, 0.5)
    variants["gaussian_sigma1.0"] = _time_image(_gaussian, raw, 1.0)

    if target_for_oracle is not None:
        target_for_oracle = validate_rgb(target_for_oracle)
        oracle_started = perf_counter()
        oracle, reverted = oracle_cell_fallback(
            raw,
            variants["nlm_color_h9"].image,
            target_for_oracle,
        )
        variants["nlm_color_h9_oracle_cell_fallback"] = TimedImage(
            image=oracle,
            seconds=variants["nlm_color_h9"].seconds + perf_counter() - oracle_started,
            deployable=False,
        )
        audit["nlm_color_h9_oracle_cell_reverted"] = reverted

    return variants, audit


def summarize_variant_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-board variant metrics with deterministic four-fold robustness."""
    if not rows:
        raise ValueError("at least one result row is required")
    names = list(rows[0]["variants"])
    raw_scores = np.asarray([row["variants"]["raw"]["ssim"] for row in rows])
    summary: dict[str, dict[str, Any]] = {}
    for name in names:
        scores = np.asarray([row["variants"][name]["ssim"] for row in rows], np.float64)
        runtimes = np.asarray(
            [row["variants"][name]["runtime_seconds"] for row in rows], np.float64
        )
        gains = scores - raw_scores
        folds = [float(scores[offset::4].mean()) for offset in range(min(4, len(scores)))]
        robust = float(scores.mean() - 0.5 * np.std(folds))
        if len(gains) > 1:
            half_width = float(
                student_t.ppf(0.975, len(gains) - 1) * gains.std(ddof=1) / np.sqrt(len(gains))
            )
        else:
            half_width = 0.0
        summary[name] = {
            "deployable": bool(rows[0]["variants"][name]["deployable"]),
            "mean_ssim": float(scores.mean()),
            "robust_ssim": robust,
            "fold_ssim": folds,
            "mean_gain_vs_raw": float(gains.mean()),
            "mean_gain_ci95_low": float(gains.mean() - half_width),
            "mean_gain_ci95_high": float(gains.mean() + half_width),
            "wins_vs_raw": int((scores > raw_scores).sum()),
            "ties_vs_raw": int((scores == raw_scores).sum()),
            "losses_vs_raw": int((scores < raw_scores).sum()),
            "mean_runtime_seconds": float(runtimes.mean()),
            "total_runtime_seconds": float(runtimes.sum()),
        }
    return summary
