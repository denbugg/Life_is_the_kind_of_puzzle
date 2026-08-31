"""Fixed evidence-composed legal puzzle stack and its frozen promotion gate.

This module has no target loading, model loading, data discovery, or parameter
selection.  It combines already ordered original/rendered tiles into the four
preregistered arms and evaluates target-free safety summaries plus paired gate
vectors supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from aiijc_puzzle.dense_safe_tail import safety_metrics
from aiijc_puzzle.dualnaf_bounded_residual import blend_same_index_tiles
from aiijc_puzzle.edge_ranker_final_tail import paired_bootstrap_ci
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.postassembly_harmonizer import (
    DEFAULT_LUMINANCE_GAIN_CONFIG,
    DEFAULT_SEAM_GRAPH_CONFIG,
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import TILE_COUNT, TILE_SIZE, assemble_tiles

ARM_A = "A_bilateral_alpha0_h20"
ARM_B = "B_bilateral_alpha0_h28"
ARM_C = "C_fused_cap08_alpha0_h28"
ARM_D = "D_fused_cap08_dualnaf_alpha0125_h28"
ARMS = (ARM_A, ARM_B, ARM_C, ARM_D)
PROMOTABLE_ARM = ARM_D
DUALNAF_ALPHA = 0.125


def _validate_tiles(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if array.shape != expected or array.dtype != np.uint8:
        raise ValueError(f"expected uint8 tiles {expected}, got {array.dtype} {array.shape}")
    return np.ascontiguousarray(array)


def apply_tail(
    ordered_tiles: np.ndarray,
    *,
    h: int,
    rgb_config: SeamGraphConfig = DEFAULT_SEAM_GRAPH_CONFIG,
    luma_config: LuminanceGainConfig = DEFAULT_LUMINANCE_GAIN_CONFIG,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply exact RGB offsets, bounded luma, and one proper coloured NLM pass."""

    if h not in {20, 28}:
        raise ValueError("ultimate stack permits only the frozen h20 or h28 tail")
    ordered = _validate_tiles(ordered_tiles)
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered, rgb_config)
    rgb_tiles = apply_rgb_offsets(ordered, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(rgb_tiles, luma_config)
    harmonized = assemble_tiles(apply_luminance_gains(rgb_tiles, gains))
    final = apply_nlm_color(harmonized, h=h).image
    return (
        harmonized,
        final,
        {
            "h": h,
            "h_color": h,
            "template_window": 7,
            "search_window": 21,
            "rgb_seam_offsets": rgb_diagnostics,
            "bounded_luminance_gains": luma_diagnostics,
        },
    )


def render_arms(
    bilateral_original: np.ndarray,
    fused_original: np.ndarray,
    fused_rendered: np.ndarray,
    *,
    rgb_config: SeamGraphConfig = DEFAULT_SEAM_GRAPH_CONFIG,
    luma_config: LuminanceGainConfig = DEFAULT_LUMINANCE_GAIN_CONFIG,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render exactly the fixed four-arm roster without a sweep."""

    bilateral = _validate_tiles(bilateral_original)
    fused = _validate_tiles(fused_original)
    rendered = _validate_tiles(fused_rendered)
    dualnaf = blend_same_index_tiles(fused, rendered, DUALNAF_ALPHA)
    specifications = (
        (ARM_A, bilateral, 20),
        (ARM_B, bilateral, 28),
        (ARM_C, fused, 28),
        (ARM_D, dualnaf, 28),
    )
    predictions: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for name, tiles, strength in specifications:
        harmonized, final, tail = apply_tail(
            tiles,
            h=strength,
            rgb_config=rgb_config,
            luma_config=luma_config,
        )
        predictions[name] = final
        diagnostics[name] = {
            "tail": tail,
            "harmonized_sha_input_shape": list(harmonized.shape),
            "safety": safety_metrics(final),
        }
    delta = np.abs(dualnaf.astype(np.int16) - fused.astype(np.int16))
    diagnostics[ARM_D]["dualnaf_blend"] = {
        "alpha": DUALNAF_ALPHA,
        "mean_abs_change": float(delta.mean()),
        "q99_abs_change": float(np.quantile(delta, 0.99)),
        "maximum_abs_change": int(delta.max()),
    }
    if tuple(predictions) != ARMS:
        raise RuntimeError("ultimate arm roster drifted")
    return predictions, diagnostics


def safety_summary(
    baseline: Sequence[Mapping[str, float]],
    candidate: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Aggregate the frozen D/A detail and grid ratios across boards."""

    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError("safety vectors must have equal length >=2")
    gradient = np.asarray(
        [
            float(after["within_tile_gradient"]) / float(before["within_tile_gradient"])
            for before, after in zip(baseline, candidate, strict=True)
        ],
        dtype=np.float64,
    )
    laplacian = np.asarray(
        [
            float(after["laplacian_energy"]) / float(before["laplacian_energy"])
            for before, after in zip(baseline, candidate, strict=True)
        ],
        dtype=np.float64,
    )
    grid = np.asarray(
        [
            float(after["grid_ratio"]) / float(before["grid_ratio"])
            for before, after in zip(baseline, candidate, strict=True)
        ],
        dtype=np.float64,
    )
    if not all(np.isfinite(vector).all() for vector in (gradient, laplacian, grid)):
        raise ValueError("safety ratios must be finite")
    return {
        "within_tile_gradient_retention_mean": float(gradient.mean()),
        "within_tile_gradient_retention_min": float(gradient.min()),
        "laplacian_retention_mean": float(laplacian.mean()),
        "laplacian_retention_min": float(laplacian.min()),
        "grid_ratio_relative_mean": float(grid.mean()),
        "grid_ratio_relative_max": float(grid.max()),
    }


def quantitative_gate(
    score_a: Sequence[float],
    score_b: Sequence[float],
    score_d: Sequence[float],
    fused_adjacency_delta: Sequence[float],
    fused_translation_delta: Sequence[float],
    safety: Mapping[str, float],
    *,
    seed: int = 20260920,
    replicates: int = 20_000,
) -> dict[str, Any]:
    """Apply every frozen ultimate-stack quantitative promotion condition."""

    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    d = np.asarray(score_d, dtype=np.float64)
    adjacency = np.asarray(fused_adjacency_delta, dtype=np.float64)
    translation = np.asarray(fused_translation_delta, dtype=np.float64)
    lengths = {len(a), len(b), len(d), len(adjacency), len(translation)}
    if len(lengths) != 1 or next(iter(lengths), 0) < 2:
        raise ValueError("all gate vectors must have the same length >=2")
    if not all(np.isfinite(value).all() for value in (a, b, d, adjacency, translation)):
        raise ValueError("gate vectors must be finite")
    d_vs_a = d - a
    d_vs_b = d - b
    d_a_ci = paired_bootstrap_ci(d_vs_a, seed=seed, replicates=replicates)
    d_b_ci = paired_bootstrap_ci(d_vs_b, seed=seed + 1, replicates=replicates)
    adjacency_ci = paired_bootstrap_ci(adjacency, seed=seed + 2, replicates=replicates)
    observed = {
        "D_mean_final_ssim": float(d.mean()),
        "D_vs_A_ci95_lower": float(d_a_ci["ci95_lower"]),
        "D_vs_A_wins": int(np.sum(d_vs_a > 0)),
        "D_vs_B_ci95_lower": float(d_b_ci["ci95_lower"]),
        "D_vs_B_wins": int(np.sum(d_vs_b > 0)),
        "fused_adjacency_ci95_lower": float(adjacency_ci["ci95_lower"]),
        "fused_translation_delta_mean": float(translation.mean()),
        **{key: float(value) for key, value in safety.items()},
    }
    requirements = (
        ("D_mean_final_ssim", ">= 0.27", observed["D_mean_final_ssim"] >= 0.27),
        ("D_vs_A_ci95_lower", "> 0", observed["D_vs_A_ci95_lower"] > 0.0),
        ("D_vs_A_wins", ">= 18", observed["D_vs_A_wins"] >= 18),
        ("D_vs_B_ci95_lower", "> 0", observed["D_vs_B_ci95_lower"] > 0.0),
        ("D_vs_B_wins", ">= 15", observed["D_vs_B_wins"] >= 15),
        (
            "fused_adjacency_ci95_lower",
            ">= 0",
            observed["fused_adjacency_ci95_lower"] >= 0.0,
        ),
        (
            "fused_translation_delta_mean",
            ">= 0",
            observed["fused_translation_delta_mean"] >= 0.0,
        ),
        (
            "within_tile_gradient_retention_mean",
            ">= 0.80",
            observed["within_tile_gradient_retention_mean"] >= 0.80,
        ),
        (
            "within_tile_gradient_retention_min",
            ">= 0.70",
            observed["within_tile_gradient_retention_min"] >= 0.70,
        ),
        (
            "laplacian_retention_mean",
            ">= 0.72",
            observed["laplacian_retention_mean"] >= 0.72,
        ),
        (
            "laplacian_retention_min",
            ">= 0.60",
            observed["laplacian_retention_min"] >= 0.60,
        ),
        (
            "grid_ratio_relative_mean",
            "<= 1.05",
            observed["grid_ratio_relative_mean"] <= 1.05,
        ),
        (
            "grid_ratio_relative_max",
            "<= 1.12",
            observed["grid_ratio_relative_max"] <= 1.12,
        ),
    )
    conditions = [
        {
            "metric": metric,
            "observed": observed[metric],
            "required": required,
            "passed": bool(passed),
        }
        for metric, required, passed in requirements
    ]
    return {
        "passed": all(condition["passed"] for condition in conditions),
        "conditions": conditions,
        "D_vs_A": {
            **d_a_ci,
            "wins": observed["D_vs_A_wins"],
            "ties": int(np.sum(d_vs_a == 0)),
            "losses": int(np.sum(d_vs_a < 0)),
        },
        "D_vs_B": {
            **d_b_ci,
            "wins": observed["D_vs_B_wins"],
            "ties": int(np.sum(d_vs_b == 0)),
            "losses": int(np.sum(d_vs_b < 0)),
        },
        "fused_adjacency_delta": adjacency_ci,
        "fused_translation_delta_mean": observed["fused_translation_delta_mean"],
        "safety": dict(safety),
    }


__all__ = [
    "ARM_A",
    "ARM_B",
    "ARM_C",
    "ARM_D",
    "ARMS",
    "DUALNAF_ALPHA",
    "PROMOTABLE_ARM",
    "apply_tail",
    "quantitative_gate",
    "render_arms",
    "safety_summary",
]
