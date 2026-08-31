"""Legal independent-tile inference for the official colour DRUNet checkpoint.

The architecture is the small subset of KAIR's MIT-licensed ``UNetRes`` needed
to load ``drunet_color.pth`` strictly.  Every network call sees one or more
independent upright tiles; padding is reflected from the same tile and cropped
back to the original 20x20 support.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.protocol import TILE_SIZE
from aiijc_puzzle.protocol import assemble_tiles as assemble_tiles
from aiijc_puzzle.protocol import split_tiles as split_tiles

ARM_H20 = "A_nlm_h20"
ARM_H28 = "B_nlm_h28"
ARM_DRUNET = "C_drunet_sigma40_then_nlm_h28"
ARM_NAMES = (ARM_H20, ARM_H28, ARM_DRUNET)


class DrunetResBlock(nn.Module):
    """KAIR-compatible residual block with checkpoint-identical key names."""

    def __init__(self, channels: int, *, bias: bool = False) -> None:
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.res(value)


def _residual_stage(channels: int, blocks: int, *, bias: bool) -> list[nn.Module]:
    return [DrunetResBlock(channels, bias=bias) for _ in range(blocks)]


class DrunetColor(nn.Module):
    """Exact colour DRUNet topology used by KAIR's official v1.0 weights."""

    def __init__(
        self,
        *,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        blocks: int = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or any(channel <= 0 for channel in channels):
            raise ValueError("DRUNet requires four positive channel widths")
        if blocks <= 0:
            raise ValueError("DRUNet blocks must be positive")
        c0, c1, c2, c3 = channels
        self.m_head = nn.Conv2d(4, c0, 3, padding=1, bias=bias)
        self.m_down1 = nn.Sequential(
            *_residual_stage(c0, blocks, bias=bias),
            nn.Conv2d(c0, c1, 2, stride=2, bias=bias),
        )
        self.m_down2 = nn.Sequential(
            *_residual_stage(c1, blocks, bias=bias),
            nn.Conv2d(c1, c2, 2, stride=2, bias=bias),
        )
        self.m_down3 = nn.Sequential(
            *_residual_stage(c2, blocks, bias=bias),
            nn.Conv2d(c2, c3, 2, stride=2, bias=bias),
        )
        self.m_body = nn.Sequential(*_residual_stage(c3, blocks, bias=bias))
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 2, stride=2, bias=bias),
            *_residual_stage(c2, blocks, bias=bias),
        )
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 2, stride=2, bias=bias),
            *_residual_stage(c1, blocks, bias=bias),
        )
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(c1, c0, 2, stride=2, bias=bias),
            *_residual_stage(c0, blocks, bias=bias),
        )
        self.m_tail = nn.Conv2d(c0, 3, 3, padding=1, bias=bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x1 = self.m_head(value)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        body = self.m_body(x4)
        body = self.m_up3(body + x4)
        body = self.m_up2(body + x3)
        body = self.m_up1(body + x2)
        return self.m_tail(body + x1)


@dataclass(frozen=True)
class DrunetRenderDiagnostics:
    """Target-free diagnostics for one complete tile roster."""

    tile_count: int
    sigma_255: float
    batch_size: int
    padding_bottom: int
    padding_right: int
    runtime_seconds: float
    mean_abs_change: float
    q99_abs_change: float
    maximum_abs_change: int
    clipped_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    """Hash a local source or checkpoint without decoding it."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_drunet_color(checkpoint: Path, device: torch.device) -> DrunetColor:
    """Strictly load the official bias-free colour DRUNet state dictionary."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("DRUNet checkpoint must be a non-empty state dictionary")
    model = DrunetColor()
    model.load_state_dict(payload, strict=True)
    return model.to(device).eval()


def _validate_tiles(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    if value.ndim != 4 or value.shape[1:] != (TILE_SIZE, TILE_SIZE, 3):
        raise ValueError(f"expected Nx{TILE_SIZE}x{TILE_SIZE}x3 tiles, got {value.shape}")
    if not len(value) or value.dtype != np.uint8:
        raise TypeError("tiles must be a non-empty uint8 roster")
    return np.ascontiguousarray(value)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def render_drunet_tiles(
    model: nn.Module,
    tiles: np.ndarray,
    *,
    sigma_255: float,
    device: torch.device,
    batch_size: int = 144,
) -> tuple[np.ndarray, DrunetRenderDiagnostics]:
    """Denoise upright tiles independently and crop exact 20x20 outputs."""

    source = _validate_tiles(tiles)
    if not np.isfinite(sigma_255) or not 0.0 <= sigma_255 <= 50.0:
        raise ValueError("official DRUNet sigma must be finite and in [0, 50]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pad_bottom = (-TILE_SIZE) % 8
    pad_right = (-TILE_SIZE) % 8
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
        padded = functional.pad(batch, (0, pad_right, 0, pad_bottom), mode="reflect")
        noise_map = torch.full(
            (len(batch), 1, padded.shape[-2], padded.shape[-1]),
            float(sigma_255 / 255.0),
            dtype=padded.dtype,
            device=device,
        )
        prediction = model(torch.cat((padded, noise_map), dim=1))
        prediction = prediction[..., :TILE_SIZE, :TILE_SIZE].clamp_(0.0, 1.0)
        array = prediction.permute(0, 2, 3, 1).float().cpu().numpy()
        outputs.append(np.rint(array * 255.0).clip(0, 255).astype(np.uint8))
    _synchronize(device)
    runtime = perf_counter() - started
    restored = np.ascontiguousarray(np.concatenate(outputs, axis=0))
    if restored.shape != source.shape:
        raise RuntimeError("DRUNet changed the tile roster shape")
    absolute = np.abs(restored.astype(np.int16) - source.astype(np.int16))
    diagnostics = DrunetRenderDiagnostics(
        tile_count=len(source),
        sigma_255=float(sigma_255),
        batch_size=int(batch_size),
        padding_bottom=pad_bottom,
        padding_right=pad_right,
        runtime_seconds=runtime,
        mean_abs_change=float(absolute.mean()),
        q99_abs_change=float(np.quantile(absolute, 0.99)),
        maximum_abs_change=int(absolute.max()),
        clipped_fraction=float(np.mean((restored == 0) | (restored == 255))),
    )
    return restored, diagnostics


def blend_uint8_fraction(
    baseline: np.ndarray,
    contender: np.ndarray,
    alpha: Fraction,
) -> np.ndarray:
    """Blend exact uint8 pixels with deterministic half-up integer rounding."""

    left = np.asarray(baseline)
    right = np.asarray(contender)
    if left.shape != right.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("blend inputs must be shape-matched uint8 arrays")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be strictly between zero and one")
    numerator = alpha.numerator
    denominator = alpha.denominator
    mixed = (
        left.astype(np.uint32) * (denominator - numerator)
        + right.astype(np.uint32) * numerator
        + denominator // 2
    ) // denominator
    return np.ascontiguousarray(mixed.astype(np.uint8))


def _validate_board(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    expected = (TILE_SIZE * 24, TILE_SIZE * 24, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 board {expected}, got {value.dtype} {value.shape}")
    return np.ascontiguousarray(value)


def render_frozen_drunet_arms(
    model: nn.Module,
    harmonized_tiles: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 144,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render the frozen h20, h28, and train-selected DRUNet sigma40 tail."""

    tiles = _validate_tiles(harmonized_tiles)
    if len(tiles) != 24 * 24:
        raise ValueError("frozen DRUNet screen requires all 576 ordered tiles")
    harmonized = assemble_tiles(tiles)
    h20 = apply_nlm_color(harmonized, h=20)
    h28 = apply_nlm_color(harmonized, h=28)
    restored_tiles, drunet = render_drunet_tiles(
        model,
        tiles,
        sigma_255=40,
        device=device,
        batch_size=batch_size,
    )
    neural_then_h28 = apply_nlm_color(assemble_tiles(restored_tiles), h=28)
    predictions = {
        ARM_H20: h20.image,
        ARM_H28: h28.image,
        ARM_DRUNET: neural_then_h28.image,
    }
    return predictions, {
        "nlm_runtime_seconds": {
            ARM_H20: h20.seconds,
            ARM_H28: h28.seconds,
            ARM_DRUNET: neural_then_h28.seconds,
        },
        "drunet": drunet.as_dict(),
    }


def board_safety_diagnostics(image: np.ndarray) -> dict[str, Any]:
    """Measure within-tile detail, grid seams, colour spread, and clipping."""

    value = _validate_board(image)
    floating = value.astype(np.float64)
    luma = (
        0.299 * floating[..., 0]
        + 0.587 * floating[..., 1]
        + 0.114 * floating[..., 2]
    )
    horizontal = np.abs(np.diff(luma, axis=1))
    vertical = np.abs(np.diff(luma, axis=0))
    inside = np.arange(value.shape[0] - 1) % TILE_SIZE != TILE_SIZE - 1
    gradient = float((horizontal[:, inside].mean() + vertical[inside, :].mean()) / 2.0)
    tiles = split_tiles(value)
    tile_luma = (
        0.299 * tiles[..., 0].astype(np.float64)
        + 0.587 * tiles[..., 1].astype(np.float64)
        + 0.114 * tiles[..., 2].astype(np.float64)
    )
    laplacian = np.abs(
        -4.0 * tile_luma[:, 1:-1, 1:-1]
        + tile_luma[:, :-2, 1:-1]
        + tile_luma[:, 2:, 1:-1]
        + tile_luma[:, 1:-1, :-2]
        + tile_luma[:, 1:-1, 2:]
    )
    horizontal_seams = np.abs(
        luma[:, TILE_SIZE - 1 : -1 : TILE_SIZE] - luma[:, TILE_SIZE::TILE_SIZE]
    )
    vertical_seams = np.abs(
        luma[TILE_SIZE - 1 : -1 : TILE_SIZE, :] - luma[TILE_SIZE::TILE_SIZE, :]
    )
    return {
        "within_tile_luma_gradient_mean_abs": gradient,
        "within_tile_luma_laplacian_mean_abs": float(laplacian.mean()),
        "grid_luma_seam_mean_abs": float(
            (horizontal_seams.mean() + vertical_seams.mean()) / 2.0
        ),
        "rgb_mean": floating.mean(axis=(0, 1)).tolist(),
        "rgb_std": floating.std(axis=(0, 1)).tolist(),
        "global_rgb_std": float(floating.std()),
        "clipped_fraction": float(np.mean((value == 0) | (value == 255))),
    }


def candidate_safety_ratios(
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    """Compare the neural candidate with h28 using target-free fixed diagnostics."""

    control = _validate_board(baseline)
    contender = _validate_board(candidate)
    left = board_safety_diagnostics(control)
    right = board_safety_diagnostics(contender)
    rgb_shift = np.asarray(right["rgb_mean"]) - np.asarray(left["rgb_mean"])
    return {
        "gradient_ratio_vs_h28": float(
            right["within_tile_luma_gradient_mean_abs"]
            / left["within_tile_luma_gradient_mean_abs"]
        ),
        "laplacian_ratio_vs_h28": float(
            right["within_tile_luma_laplacian_mean_abs"]
            / left["within_tile_luma_laplacian_mean_abs"]
        ),
        "grid_seam_ratio_vs_h28": float(
            right["grid_luma_seam_mean_abs"] / left["grid_luma_seam_mean_abs"]
        ),
        "global_rgb_std_ratio_vs_h28": float(
            right["global_rgb_std"] / left["global_rgb_std"]
        ),
        "rgb_mean_shift_vs_h28": rgb_shift.tolist(),
        "maximum_abs_rgb_mean_shift_vs_h28": float(np.max(np.abs(rgb_shift))),
        "mean_abs_pixel_change_vs_h28": float(
            np.abs(contender.astype(np.int16) - control.astype(np.int16)).mean()
        ),
        "clipped_fraction_increase_vs_h28": float(
            right["clipped_fraction"] - left["clipped_fraction"]
        ),
        "baseline": left,
        "candidate": right,
    }
