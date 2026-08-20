"""Target-free E18b full-image NLM polish with a deterministic no-gray guard."""
from __future__ import annotations

from typing import Any

import numpy as np

GRID = 24
TILE = 20
N = GRID * GRID
IMAGE_SIZE = GRID * TILE
NLM_H = 9
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(
            f"expected {(IMAGE_SIZE, IMAGE_SIZE, 3)} RGB image, got {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image, got {image.dtype}")
    return image


def gray_mask(image: np.ndarray) -> np.ndarray:
    """Frozen gray-cell definition shared with the independent archive audit."""
    image = _validate_rgb(image)
    tiles = (image.reshape(GRID, TILE, GRID, TILE, 3)
             .transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3))
    mean = tiles.mean((1, 2))
    std = tiles.std((1, 2, 3))
    return (mean.max(1) - mean.min(1) < 10) & (std < 25)


def gray_count(image: np.ndarray) -> int:
    return int(gray_mask(image).sum())


def nlm_h9(rgb: np.ndarray, cv2_module: Any | None = None) -> np.ndarray:
    """Apply the frozen full-image colored-NLM configuration."""
    rgb = _validate_rgb(rgb)
    if cv2_module is None:
        import cv2 as cv2_module
    bgr = cv2_module.cvtColor(rgb, cv2_module.COLOR_RGB2BGR)
    filtered = cv2_module.fastNlMeansDenoisingColored(
        bgr,
        None,
        NLM_H,
        NLM_H,
        NLM_TEMPLATE_WINDOW,
        NLM_SEARCH_WINDOW,
    )
    output = cv2_module.cvtColor(filtered, cv2_module.COLOR_BGR2RGB)
    return _validate_rgb(output)


def no_gray_guard(raw: np.ndarray, filtered: np.ndarray) -> tuple[np.ndarray, int]:
    """Revert only 20x20 cells that the polish newly classifies as gray."""
    raw = _validate_rgb(raw)
    filtered = _validate_rgb(filtered)
    raw_tiles = (raw.reshape(GRID, TILE, GRID, TILE, 3)
                 .transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3))
    filtered_tiles = (filtered.reshape(GRID, TILE, GRID, TILE, 3)
                      .transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)).copy()
    revert = gray_mask(filtered) & ~gray_mask(raw)
    filtered_tiles[revert] = raw_tiles[revert]
    guarded = (filtered_tiles.reshape(GRID, GRID, TILE, TILE, 3)
               .transpose(0, 2, 1, 3, 4).reshape(IMAGE_SIZE, IMAGE_SIZE, 3))
    return guarded, int(revert.sum())


def no_gray_nlm_h9(
    raw: np.ndarray,
    cv2_module: Any | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Run E18 NLM then the E18b guard, raising on any audit violation."""
    raw = _validate_rgb(raw)
    unguarded = nlm_h9(raw, cv2_module=cv2_module)
    guarded, reverted = no_gray_guard(raw, unguarded)
    raw_gray = gray_count(raw)
    unguarded_gray = gray_count(unguarded)
    guarded_gray = gray_count(guarded)
    if guarded_gray > raw_gray:
        raise AssertionError(
            f"E18b gray audit failed: raw={raw_gray}, guarded={guarded_gray}"
        )
    return guarded, {
        "raw_gray_count": raw_gray,
        "unguarded_gray_count": unguarded_gray,
        "guarded_gray_count": guarded_gray,
        "reverted_new_gray_cells": reverted,
    }


def polish_or_raw(
    raw: np.ndarray,
    *,
    enabled: bool = True,
    fallback_on_error: bool = True,
    cv2_module: Any | None = None,
) -> tuple[np.ndarray, bool, str | None, dict[str, int]]:
    """Return guarded E18b output, or the bit-identical raw image on failure."""
    raw = _validate_rgb(raw)
    if not enabled:
        return raw.copy(), False, "disabled", {
            "raw_gray_count": gray_count(raw),
            "unguarded_gray_count": gray_count(raw),
            "guarded_gray_count": gray_count(raw),
            "reverted_new_gray_cells": 0,
        }
    try:
        output, stats = no_gray_nlm_h9(raw, cv2_module=cv2_module)
        return output, True, None, stats
    except Exception as exc:
        if not fallback_on_error:
            raise
        raw_gray = gray_count(raw)
        return raw.copy(), False, f"{type(exc).__name__}: {exc}", {
            "raw_gray_count": raw_gray,
            "unguarded_gray_count": raw_gray,
            "guarded_gray_count": raw_gray,
            "reverted_new_gray_cells": 0,
        }
