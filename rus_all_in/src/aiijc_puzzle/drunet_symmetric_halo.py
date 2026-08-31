"""Fixed symmetric-halo ablation for the safe goal-cycle-2 DRUNet50 arm.

The baseline is goal-cycle-2 arm B byte-for-byte: independent tilewise
DRUNet50 with the historical right/bottom-only 20x20 -> 24x24 reflection,
followed by the frozen protected h28/h50 t60 tail.  The sole candidate changes
only neural inference geometry: each tile is reflected by six pixels on all
four sides, inferred at 32x32, and center-cropped back to its exact 20x20
support before the same protected tail.

No neural call can observe another tile or board.  The strict layout,
harmonizer, sigma, NLM strengths, mask, and blend are supplied unchanged by the
existing legal pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from aiijc_puzzle.drunet_goal_cycle2 import (
    CANDIDATE_SIGMA50,
    DIRECT_SIGMA,
    MODEL_BATCH_SIZE,
    blend_h28_safe_h50_flat_t60,
    tile_flatness_counts,
)
from aiijc_puzzle.edge_protected_nlm import colored_nlm
from aiijc_puzzle.nlm_luma_chroma import structure_diagnostics
from aiijc_puzzle.pretrained_tile_denoiser import render_drunet_tiles
from aiijc_puzzle.protocol import TILE_COUNT, TILE_SIZE, assemble_tiles

BASELINE_B = CANDIDATE_SIGMA50
SYMMETRIC_HALO_B = "E_drunet50_symmetric_halo32_protected_h28_h50_t60"
ARM_NAMES = (BASELINE_B, SYMMETRIC_HALO_B)

SYMMETRIC_HALO = 6
PADDED_TILE_SIZE = TILE_SIZE + 2 * SYMMETRIC_HALO
HALO_BATCH_SIZE = 72


@dataclass(frozen=True)
class SymmetricHaloDiagnostics:
    """Target-free diagnostics for one complete independent-tile roster."""

    tile_count: int
    sigma_255: float
    batch_size: int
    halo_top: int
    halo_bottom: int
    halo_left: int
    halo_right: int
    padded_tile_size: int
    crop_start: int
    crop_stop: int
    runtime_seconds: float
    mean_abs_change: float
    q99_abs_change: float
    maximum_abs_change: int
    clipped_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 tiles {expected}, got {value.dtype} {value.shape}")
    return np.ascontiguousarray(value)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def render_drunet_tiles_symmetric_halo(
    model: nn.Module,
    tiles: np.ndarray,
    *,
    sigma_255: float,
    device: torch.device,
    batch_size: int = HALO_BATCH_SIZE,
) -> tuple[np.ndarray, SymmetricHaloDiagnostics]:
    """Denoise each tile at 32x32 and return its exact center 20x20 crop."""

    source = _validate_tiles(tiles)
    if not np.isfinite(sigma_255) or not 0.0 <= sigma_255 <= 50.0:
        raise ValueError("official DRUNet sigma must be finite and in [0, 50]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if PADDED_TILE_SIZE != 32:
        raise RuntimeError("symmetric-halo geometry drifted from frozen 32x32")

    outputs: list[np.ndarray] = []
    model.eval()
    _synchronize(device)
    started = perf_counter()
    for start in range(0, len(source), batch_size):
        batch = (
            torch.from_numpy(source[start : start + batch_size])
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float32)
            .div_(255.0)
        )
        padded = functional.pad(
            batch,
            (SYMMETRIC_HALO,) * 4,
            mode="reflect",
        )
        noise_map = torch.full(
            (len(batch), 1, PADDED_TILE_SIZE, PADDED_TILE_SIZE),
            float(sigma_255 / 255.0),
            dtype=padded.dtype,
            device=device,
        )
        prediction = model(torch.cat((padded, noise_map), dim=1))
        crop_stop = SYMMETRIC_HALO + TILE_SIZE
        prediction = prediction[
            ...,
            SYMMETRIC_HALO:crop_stop,
            SYMMETRIC_HALO:crop_stop,
        ].clamp_(0.0, 1.0)
        array = prediction.permute(0, 2, 3, 1).float().cpu().numpy()
        outputs.append(np.rint(array * 255.0).clip(0, 255).astype(np.uint8))
    _synchronize(device)
    runtime = perf_counter() - started

    restored = np.ascontiguousarray(np.concatenate(outputs, axis=0))
    if restored.shape != source.shape:
        raise RuntimeError("symmetric-halo DRUNet changed the tile roster shape")
    absolute = np.abs(restored.astype(np.int16) - source.astype(np.int16))
    diagnostics = SymmetricHaloDiagnostics(
        tile_count=len(source),
        sigma_255=float(sigma_255),
        batch_size=int(batch_size),
        halo_top=SYMMETRIC_HALO,
        halo_bottom=SYMMETRIC_HALO,
        halo_left=SYMMETRIC_HALO,
        halo_right=SYMMETRIC_HALO,
        padded_tile_size=PADDED_TILE_SIZE,
        crop_start=SYMMETRIC_HALO,
        crop_stop=SYMMETRIC_HALO + TILE_SIZE,
        runtime_seconds=runtime,
        mean_abs_change=float(absolute.mean()),
        q99_abs_change=float(np.quantile(absolute, 0.99)),
        maximum_abs_change=int(absolute.max()),
        clipped_fraction=float(np.mean((restored == 0) | (restored == 255))),
    )
    return restored, diagnostics


def _protected_tail(restored_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    canvas = assemble_tiles(_validate_tiles(restored_tiles))
    h20 = colored_nlm(canvas, 20)
    h28 = colored_nlm(canvas, 28)
    h50 = colored_nlm(canvas, 50)
    return blend_h28_safe_h50_flat_t60(h20, h28, h50)


def render_symmetric_halo_arms(
    model: nn.Module,
    harmonized_tiles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render frozen goal-cycle-2 B and its sole symmetric-halo ablation."""

    tiles = _validate_tiles(harmonized_tiles)
    baseline_tiles, baseline_neural = render_drunet_tiles(
        model,
        tiles,
        sigma_255=DIRECT_SIGMA,
        device=device,
        batch_size=MODEL_BATCH_SIZE,
    )
    baseline, baseline_mask = _protected_tail(baseline_tiles)

    halo_tiles, halo_neural = render_drunet_tiles_symmetric_halo(
        model,
        tiles,
        sigma_255=DIRECT_SIGMA,
        device=device,
        batch_size=HALO_BATCH_SIZE,
    )
    halo, halo_mask = _protected_tail(halo_tiles)

    predictions = {BASELINE_B: baseline, SYMMETRIC_HALO_B: halo}
    if tuple(predictions) != ARM_NAMES:
        raise RuntimeError("symmetric-halo arm order drifted")
    if any(
        value.shape != baseline.shape or value.dtype != np.uint8
        for value in predictions.values()
    ):
        raise RuntimeError("symmetric-halo renderer changed output geometry or dtype")
    if np.array_equal(baseline, halo):
        raise RuntimeError("symmetric-halo prediction is not distinct from baseline B")

    return predictions, {
        "baseline_neural": baseline_neural.as_dict(),
        "baseline_mask": baseline_mask,
        "symmetric_halo_neural": halo_neural.as_dict(),
        "symmetric_halo_mask": halo_mask,
        "fixed_difference": (
            "only same-tile DRUNet padding/crop: right-bottom 24x24/top-left crop "
            "versus four-sided halo6 32x32/center crop"
        ),
        "structure": {name: structure_diagnostics(image) for name, image in predictions.items()},
        "tile_flatness": {name: tile_flatness_counts(image) for name, image in predictions.items()},
    }


__all__ = [
    "ARM_NAMES",
    "BASELINE_B",
    "HALO_BATCH_SIZE",
    "PADDED_TILE_SIZE",
    "SYMMETRIC_HALO",
    "SYMMETRIC_HALO_B",
    "SymmetricHaloDiagnostics",
    "render_drunet_tiles_symmetric_halo",
    "render_symmetric_halo_arms",
]
