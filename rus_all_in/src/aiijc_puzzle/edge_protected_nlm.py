"""Target-blind flat-region NLM with protected content and tile edges."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from aiijc_puzzle.protocol import IMAGE_SIZE

TILE_SIZE = 20
TEMPLATE_WINDOW = 7
SEARCH_WINDOW = 21


@dataclass(frozen=True)
class ProtectedArm:
    name: str
    aggressive_h: int
    sobel_threshold: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("arm name must be non-empty")
        if isinstance(self.aggressive_h, bool) or not 30 <= self.aggressive_h <= 60:
            raise ValueError("aggressive_h must be an integer in [30, 60]")
        if not np.isfinite(self.sobel_threshold) or self.sobel_threshold <= 0:
            raise ValueError("sobel_threshold must be finite and positive")


def validate_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 RGB {(IMAGE_SIZE, IMAGE_SIZE, 3)}, got {value.dtype} {value.shape}"
        )
    return np.ascontiguousarray(value)


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(validate_rgb(image).tobytes()).hexdigest()


def colored_nlm(image: np.ndarray, h: int) -> np.ndarray:
    """Apply exactly one proper RGB/BGR OpenCV colored-NLM pass."""

    if isinstance(h, bool) or not isinstance(h, int) or not 0 < h <= 60:
        raise ValueError("h must be an integer in [1, 60]")
    rgb = validate_rgb(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filtered = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h,
        h,
        TEMPLATE_WINDOW,
        SEARCH_WINDOW,
    )
    return np.ascontiguousarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))


def protected_masks(
    safe_h20: np.ndarray,
    *,
    sobel_threshold: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the exact dilated binary mask, soft mask, and protected fraction."""

    if not np.isfinite(sobel_threshold) or sobel_threshold <= 0:
        raise ValueError("sobel_threshold must be finite and positive")
    safe = validate_rgb(safe_h20)
    gray = cv2.cvtColor(safe, cv2.COLOR_RGB2GRAY).astype(np.float32)
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(horizontal, vertical)
    grid = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    for coordinate in range(TILE_SIZE, IMAGE_SIZE, TILE_SIZE):
        grid[:, coordinate - 1 : coordinate + 1] = True
        grid[coordinate - 1 : coordinate + 1, :] = True
    binary = (magnitude >= sobel_threshold) | grid
    dilated = cv2.dilate(
        binary.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    soft = cv2.GaussianBlur(
        dilated.astype(np.float32),
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0,
    )
    return dilated, np.clip(soft, 0.0, 1.0), float(dilated.mean())


def protected_weight(safe_h20: np.ndarray, *, sobel_threshold: float) -> tuple[np.ndarray, float]:
    """Return the exact preregistered soft weight for the safe h20 image."""

    _, soft, protected_fraction = protected_masks(
        safe_h20,
        sobel_threshold=sobel_threshold,
    )
    return soft, protected_fraction


def blend_protected(
    safe_h20: np.ndarray,
    aggressive: np.ndarray,
    *,
    sobel_threshold: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Blend two independent single-pass outputs without changing geometry."""

    safe = validate_rgb(safe_h20)
    strong = validate_rgb(aggressive)
    weight, protected_fraction = protected_weight(safe, sobel_threshold=sobel_threshold)
    mixed = np.rint(
        weight[..., None] * safe.astype(np.float32)
        + (1.0 - weight[..., None]) * strong.astype(np.float32)
    )
    output = np.ascontiguousarray(mixed.clip(0, 255).astype(np.uint8))
    return output, {
        "binary_dilated_protected_fraction": protected_fraction,
        "mean_soft_safe_weight": float(weight.mean()),
        "minimum_soft_safe_weight": float(weight.min()),
        "maximum_soft_safe_weight": float(weight.max()),
    }


__all__ = [
    "ProtectedArm",
    "blend_protected",
    "colored_nlm",
    "image_digest",
    "protected_masks",
    "protected_weight",
    "validate_rgb",
]
