"""Geometry-faithful batched inference for independent upright puzzle tiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from aiijc_puzzle.protocol import TILE_COUNT, TILE_SIZE
from aiijc_puzzle.restoration_r6 import nlm_color


@dataclass(frozen=True)
class TilewiseRenderDiagnostics:
    """Target-free pixel-change diagnostics for a one-to-one tile render."""

    tile_count: int
    batch_size: int
    conditioning_mean_abs_change: float
    mean_abs_change: float
    residual_from_conditioning_mean_abs: float
    residual_from_conditioning_q99_abs: float
    q95_abs_change: float
    q99_abs_change: float
    maximum_abs_change: int
    unchanged_fraction: float
    clipped_fraction: float

    def as_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    expected = (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3)
    if value.shape != expected:
        raise ValueError(f"expected {expected} upright tiles, got {value.shape}")
    if value.dtype != np.uint8:
        raise TypeError(f"expected uint8 tiles, got {value.dtype}")
    return np.ascontiguousarray(value)


def _batch_tensor(tiles: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(tiles))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def _tensor_tiles(value: torch.Tensor) -> np.ndarray:
    if value.ndim != 4 or value.shape[1:] != (3, TILE_SIZE, TILE_SIZE):
        raise ValueError(f"expected Bx3x{TILE_SIZE}x{TILE_SIZE}, got {tuple(value.shape)}")
    array = value.detach().clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy()
    return np.rint(array * 255.0).clip(0, 255).astype(np.uint8)


@torch.inference_mode()
def render_tiles_independently(
    model: nn.Module,
    tiles: np.ndarray,
    device: torch.device,
    *,
    nlm_h: int = 10,
    batch_size: int = 144,
) -> tuple[np.ndarray, TilewiseRenderDiagnostics]:
    """Render 576 tiles without any cross-tile pixels, context, or resampling.

    OpenCV NLM conditioning is computed on each 20x20 tile independently. The
    network receives a batch of independent tiles; batch membership never
    changes spatial support and the output keeps the input tile index exactly.
    """

    source = _validate_tiles(tiles)
    if nlm_h <= 0 or batch_size <= 0:
        raise ValueError("nlm_h and batch_size must be positive")
    conditioning = np.stack([nlm_color(tile, nlm_h) for tile in source])
    rendered_chunks: list[np.ndarray] = []
    model.eval()
    for start in range(0, TILE_COUNT, batch_size):
        stop = min(start + batch_size, TILE_COUNT)
        prediction = model(
            _batch_tensor(source[start:stop], device),
            _batch_tensor(conditioning[start:stop], device),
        )
        rendered_chunks.append(_tensor_tiles(prediction))
    rendered = np.ascontiguousarray(np.concatenate(rendered_chunks, axis=0))
    if rendered.shape != source.shape:
        raise RuntimeError("tile-wise renderer changed the tile roster shape")

    absolute = np.abs(rendered.astype(np.int16) - source.astype(np.int16))
    conditioning_absolute = np.abs(conditioning.astype(np.int16) - source.astype(np.int16))
    residual_absolute = np.abs(rendered.astype(np.int16) - conditioning.astype(np.int16))
    diagnostics = TilewiseRenderDiagnostics(
        tile_count=TILE_COUNT,
        batch_size=batch_size,
        conditioning_mean_abs_change=float(conditioning_absolute.mean()),
        mean_abs_change=float(absolute.mean()),
        residual_from_conditioning_mean_abs=float(residual_absolute.mean()),
        residual_from_conditioning_q99_abs=float(np.quantile(residual_absolute, 0.99)),
        q95_abs_change=float(np.quantile(absolute, 0.95)),
        q99_abs_change=float(np.quantile(absolute, 0.99)),
        maximum_abs_change=int(absolute.max()),
        unchanged_fraction=float(np.mean(absolute == 0)),
        clipped_fraction=float(np.mean((rendered == 0) | (rendered == 255))),
    )
    return rendered, diagnostics
