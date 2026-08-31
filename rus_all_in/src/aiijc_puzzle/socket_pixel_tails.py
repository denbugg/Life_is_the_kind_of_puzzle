"""Legal post-layout pixel tails reusable by the Socket production runner.

These functions accept only an already assembled RGB canvas. They cannot see
tile identities, matcher scores, targets, filenames, or cross-board pixels, so
they can restore pixels only after the original-tile permutation was audited.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from aiijc_puzzle.pixel_tails import (
    NLM_SEARCH_WINDOW,
    NLM_TEMPLATE_WINDOW,
    apply_nlm_color,
)
from aiijc_puzzle.postassembly_harmonizer import (
    DEFAULT_LUMINANCE_GAIN_CONFIG,
    DEFAULT_SEAM_GRAPH_CONFIG,
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    RGB_CHANNELS,
    assemble_tiles,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RGB_CONFIG_PATH = PROJECT_ROOT / "configs/postassembly_rgb_offset_v1.json"
LUMA_CONFIG_PATH = PROJECT_ROOT / "configs/postassembly_luminance_gain_v1.json"
RGB_CONFIG_SHA256 = "4adfd9b614e8556b7de5c1f527d759d15d29c0f74e20aa26ff87900dd773ec9a"
LUMA_CONFIG_SHA256 = "7488cad2ae7cc75792d6ff0ff2ea0a38fa778979083ffd5c161c857b68fd550f"
HISTORICAL_ORIGIN = {
    "repository": "/Users/rusyalain/Documents/GitHub/pazzle_will_be_killed",
    "branch": "origin/таска-говно",
    "commit": "d6a82f82ceefa109ef706402712d03805bc9e880",
    "source_path": "source/src/puzzle_assembly/postassembly_harmonizer.py",
    "source_blob": "9d8d01c0f48d0e1473c1ff48285b06ab786a5dd8",
}
NLM_H = 20


def _load_method(
    path: Path,
    *,
    expected_sha256: str,
    expected_kind: str,
) -> dict[str, Any]:
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"historical harmonizer config hash drifted: {path}; "
            f"expected {expected_sha256}, got {observed_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("method"), dict):
        raise ValueError(f"invalid target-blind harmonizer config: {path}")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != expected_kind
        or payload.get("origin") != HISTORICAL_ORIGIN
        or payload.get("target_access") is not False
    ):
        raise ValueError(f"harmonizer config is not target-blind: {path}")
    return payload["method"]


def historical_rgb_luma_nlm_h20_contract() -> dict[str, Any]:
    """Validate and describe the checked-in historical legal tail."""

    expected_rgb = {
        **asdict(DEFAULT_SEAM_GRAPH_CONFIG),
        "global_gauge": "per-channel median offset equals zero",
    }
    expected_luma = {
        **asdict(DEFAULT_LUMINANCE_GAIN_CONFIG),
        "global_gauge": "median log gain equals zero",
    }
    if _load_method(
        RGB_CONFIG_PATH,
        expected_sha256=RGB_CONFIG_SHA256,
        expected_kind="ported_target_blind_postassembly_rgb_offset",
    ) != expected_rgb:
        raise ValueError("RGB seam-offset config no longer matches the implementation")
    if _load_method(
        LUMA_CONFIG_PATH,
        expected_sha256=LUMA_CONFIG_SHA256,
        expected_kind="ported_target_blind_postassembly_bounded_luminance_gain",
    ) != expected_luma:
        raise ValueError("luminance-gain config no longer matches the implementation")
    return {
        "name": "historical-rgb-luma-nlm-h20-once",
        "input": "audited post-layout RGB canvas",
        "layout_changed": False,
        "target_blind": True,
        "cross_board_pixels_used": False,
        "rgb_config_sha256": RGB_CONFIG_SHA256,
        "luma_config_sha256": LUMA_CONFIG_SHA256,
        "operations": [
            "additive_rgb_seam_offsets",
            "bounded_luminance_gains",
            "opencv_fast_nl_means_colored_once",
        ],
        "nlm": {
            "h": NLM_H,
            "h_color": NLM_H,
            "template_window_size": NLM_TEMPLATE_WINDOW,
            "search_window_size": NLM_SEARCH_WINDOW,
            "passes": 1,
        },
    }


def historical_rgb_luma_nlm_h20_once(raw: np.ndarray) -> np.ndarray:
    """Apply the historical legal restoration tail to one audited canvas."""

    value = np.asarray(raw)
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image {expected}, got {value.dtype} {value.shape}")
    ordered = split_tiles(value)
    offsets, _ = seam_graph_rgb_offsets(ordered, DEFAULT_SEAM_GRAPH_CONFIG)
    rgb_corrected = apply_rgb_offsets(ordered, offsets)
    gains, _ = seam_graph_luminance_gains(rgb_corrected, DEFAULT_LUMINANCE_GAIN_CONFIG)
    harmonized = assemble_tiles(apply_luminance_gains(rgb_corrected, gains))
    return np.ascontiguousarray(apply_nlm_color(harmonized, h=NLM_H).image)


__all__ = [
    "historical_rgb_luma_nlm_h20_contract",
    "historical_rgb_luma_nlm_h20_once",
]
