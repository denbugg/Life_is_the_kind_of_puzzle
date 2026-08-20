"""Frozen E19 raw/NLM dual-view classical score construction."""
from __future__ import annotations

import cv2
import numpy as np

from e2_raw_fusion import ALPHA, classical_mgc_ssd_scores, fuse_scores

GRID, TILE, N = 24, 20, 576
NLM_H = 9
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21
CLASSICAL_RAW_WEIGHT = 0.5


def denoise_tiles_nlm(tiles: np.ndarray) -> np.ndarray:
    """Denoise each tile independently so shuffled-grid seams cannot leak in."""
    tiles = np.asarray(tiles)
    if tiles.shape != (N, TILE, TILE, 3) or tiles.dtype != np.uint8:
        raise ValueError(f"expected uint8 {(N, TILE, TILE, 3)} tiles, got {tiles.dtype} {tiles.shape}")
    output = np.empty_like(tiles)
    for index, tile in enumerate(tiles):
        bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        filtered = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            NLM_H,
            NLM_H,
            NLM_TEMPLATE_WINDOW,
            NLM_SEARCH_WINDOW,
        )
        output[index] = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    return output


def average_classical_logp(raw: np.ndarray, nlm: np.ndarray) -> np.ndarray:
    """Apply the frozen equal-weight average to two classical log-probability graphs."""
    raw = np.asarray(raw, np.float32)
    nlm = np.asarray(nlm, np.float32)
    if raw.shape != (N, N) or nlm.shape != (N, N):
        raise ValueError("classical directional matrices must be 576x576")
    averaged = CLASSICAL_RAW_WEIGHT * raw + (1.0 - CLASSICAL_RAW_WEIGHT) * nlm
    averaged = np.asarray(averaged, np.float32)
    np.fill_diagonal(averaged, -1e4)
    return averaged


def dual_view_classical_scores(
    raw_tiles: np.ndarray,
    raw_scores: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 50/50 raw and per-tile-NLM MGC+SSD log-probabilities."""
    raw_right, raw_down = raw_scores or classical_mgc_ssd_scores(raw_tiles)
    nlm_tiles = denoise_tiles_nlm(raw_tiles)
    nlm_right, nlm_down = classical_mgc_ssd_scores(nlm_tiles)
    return (
        average_classical_logp(raw_right, nlm_right),
        average_classical_logp(raw_down, nlm_down),
    )


def fused_directional_scores(
    raw_tiles: np.ndarray,
    learned_right: np.ndarray,
    learned_down: np.ndarray,
    raw_scores: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply unchanged E14 alpha=0.2 after constructing E19's dual classical view."""
    classical_right, classical_down = dual_view_classical_scores(raw_tiles, raw_scores)
    return (
        fuse_scores(learned_right, classical_right, alpha=ALPHA),
        fuse_scores(learned_down, classical_down, alpha=ALPHA),
    )
