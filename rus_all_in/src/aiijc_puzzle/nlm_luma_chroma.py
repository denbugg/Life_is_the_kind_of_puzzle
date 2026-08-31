"""Single-pass colored NLM with independently frozen luma/chroma strengths.

The functions in this module are inference-only: they accept one RGB image and
never discover files, targets, or examples from another board.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import stats

from aiijc_puzzle.protocol import IMAGE_SIZE

NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21


@dataclass(frozen=True)
class NLMArm:
    """One preregistered OpenCV colored-NLM arm."""

    name: str
    h: int
    h_color: int
    role: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("arm name must be non-empty")
        if self.role not in {"control", "candidate"}:
            raise ValueError("arm role must be control or candidate")
        for label, value in (("h", self.h), ("h_color", self.h_color)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 30:
                raise ValueError(f"{label} must be an integer in [1, 29]")


def validate_rgb(image: np.ndarray) -> np.ndarray:
    """Validate one strict contest-shaped RGB image."""

    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 RGB {(IMAGE_SIZE, IMAGE_SIZE, 3)}, got {value.dtype} {value.shape}"
        )
    return np.ascontiguousarray(value)


def image_digest(image: np.ndarray) -> str:
    """Hash uncompressed RGB pixels."""

    return hashlib.sha256(validate_rgb(image).tobytes()).hexdigest()


def apply_nlm_luma_chroma(image: np.ndarray, *, h: int, h_color: int) -> np.ndarray:
    """Apply exactly one OpenCV colored-NLM pass with separate strengths."""

    arm = NLMArm("validation", h, h_color, "candidate")
    rgb = validate_rgb(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filtered = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        arm.h,
        arm.h_color,
        NLM_TEMPLATE_WINDOW,
        NLM_SEARCH_WINDOW,
    )
    return np.ascontiguousarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))


def _pair_means(values: np.ndarray) -> tuple[float, float]:
    horizontal = float(np.mean(values[:, :-1]))
    vertical = float(np.mean(values[:-1, :]))
    return horizontal, vertical


def structure_diagnostics(image: np.ndarray) -> dict[str, float]:
    """Measure target-free within-tile detail and grid discontinuities."""

    rgb = validate_rgb(image)
    value = rgb.astype(np.float64)
    luminance = 0.299 * value[..., 0] + 0.587 * value[..., 1] + 0.114 * value[..., 2]

    horizontal_y = np.abs(np.diff(luminance, axis=1))
    vertical_y = np.abs(np.diff(luminance, axis=0))
    horizontal_inside = (np.arange(IMAGE_SIZE - 1) + 1) % 20 != 0
    vertical_inside = (np.arange(IMAGE_SIZE - 1) + 1) % 20 != 0
    within_y = float(
        (horizontal_y[:, horizontal_inside].mean() + vertical_y[vertical_inside, :].mean())
        / 2.0
    )
    grid_y = float(
        (horizontal_y[:, ~horizontal_inside].mean() + vertical_y[~vertical_inside, :].mean())
        / 2.0
    )

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)[..., 1:]
    horizontal_ab = np.linalg.norm(np.diff(lab, axis=1), axis=2)
    vertical_ab = np.linalg.norm(np.diff(lab, axis=0), axis=2)
    within_ab = float(
        (
            horizontal_ab[:, horizontal_inside].mean()
            + vertical_ab[vertical_inside, :].mean()
        )
        / 2.0
    )
    laplacian = float(np.mean(np.abs(cv2.Laplacian(luminance, cv2.CV_64F, ksize=3))))
    return {
        "within_tile_luminance_gradient": within_y,
        "within_tile_chroma_gradient": within_ab,
        "luminance_laplacian_energy": laplacian,
        "grid_luminance_gradient": grid_y,
        "grid_ratio": grid_y / max(within_y, 1e-12),
        "clipped_fraction": float(np.mean((rgb == 0) | (rgb == 255))),
    }


def paired_t_interval(values: Sequence[float], *, confidence: float = 0.95) -> dict[str, float]:
    """Return the preregistered two-sided paired Student-t interval."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) < 2 or not np.isfinite(vector).all():
        raise ValueError("paired values must be a finite vector of length at least two")
    mean = float(vector.mean())
    standard_error = float(stats.sem(vector))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=len(vector) - 1))
    half_width = critical * standard_error
    return {
        "mean": mean,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "confidence": confidence,
        "degrees_of_freedom": len(vector) - 1,
    }


def safety_summary(
    candidate: Sequence[Mapping[str, float]],
    baseline: Sequence[Mapping[str, float]],
) -> dict[str, float | bool]:
    """Aggregate the exact target-free ratios used by the frozen gate."""

    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("candidate and baseline diagnostics must be non-empty and paired")

    def ratios(field: str) -> np.ndarray:
        numerator = np.asarray([row[field] for row in candidate], dtype=np.float64)
        denominator = np.asarray([row[field] for row in baseline], dtype=np.float64)
        if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
            raise ValueError(f"non-finite safety metric {field}")
        return numerator / np.maximum(denominator, 1e-12)

    luminance = ratios("within_tile_luminance_gradient")
    chroma = ratios("within_tile_chroma_gradient")
    laplacian = ratios("luminance_laplacian_energy")
    grid = ratios("grid_ratio")
    return {
        "mean_luminance_gradient_retention": float(luminance.mean()),
        "minimum_luminance_gradient_retention": float(luminance.min()),
        "mean_chroma_gradient_retention": float(chroma.mean()),
        "minimum_chroma_gradient_retention": float(chroma.min()),
        "mean_laplacian_retention": float(laplacian.mean()),
        "minimum_laplacian_retention": float(laplacian.min()),
        "mean_grid_ratio_relative_to_baseline": float(grid.mean()),
        "maximum_grid_ratio_relative_to_baseline": float(grid.max()),
    }


__all__ = [
    "NLMArm",
    "apply_nlm_luma_chroma",
    "image_digest",
    "paired_t_interval",
    "safety_summary",
    "structure_diagnostics",
    "validate_rgb",
]
