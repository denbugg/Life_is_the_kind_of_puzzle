"""Exact synthetic shuffled panels with known 576-tile permutations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from puzzle_denoise_v2.degradation import SyntheticTileDegrader, pillow_libjpeg_degrade
from puzzle_denoise_v2.tiles import split_tiles_numpy

from .geometry import TILE, TILE_COUNT, validate_permutation


@dataclass(frozen=True)
class ExactPanel:
    panel: str
    seed: int
    slot_tiles: np.ndarray
    clean_target_tiles: np.ndarray
    slot_to_target: np.ndarray

    def __post_init__(self) -> None:
        if self.slot_tiles.shape != (TILE_COUNT, TILE, TILE, 3) or self.slot_tiles.dtype != np.uint8:
            raise ValueError("slot_tiles must be uint8 576x20x20x3")
        if (
            self.clean_target_tiles.shape != (TILE_COUNT, TILE, TILE, 3)
            or self.clean_target_tiles.dtype != np.uint8
        ):
            raise ValueError("clean_target_tiles must be uint8 576x20x20x3")
        validate_permutation(self.slot_to_target, name="slot_to_target")


def _torch_tiles(clean_tiles: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(clean_tiles.transpose(0, 3, 1, 2))).float().div(255.0)


def make_exact_panel(clean_target: np.ndarray, *, panel: str, seed: int) -> ExactPanel:
    clean_target = np.asarray(clean_target)
    if clean_target.shape != (480, 480, 3) or clean_target.dtype != np.uint8:
        raise ValueError("clean_target must be RGB uint8 480x480x3")
    if panel not in {"clean_shuffle", "primary_kornia", "independent_libjpeg"}:
        raise ValueError("panel must be clean_shuffle, primary_kornia, or independent_libjpeg")
    clean_tiles = split_tiles_numpy(clean_target)
    rng = np.random.default_rng(seed)
    slot_to_target = rng.permutation(TILE_COUNT).astype(np.int32)

    if panel == "clean_shuffle":
        corrupted = clean_tiles.copy()
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        degrader = SyntheticTileDegrader(variant_weights=(1.0, 0.0, 0.0))
        clean_tensor = _torch_tiles(clean_tiles)
        params = degrader.sample_parameters(TILE_COUNT, torch.device("cpu"), generator)
        noise = torch.randn(clean_tensor.shape, generator=generator, dtype=torch.float32)
        if panel == "primary_kornia":
            output, _ = degrader(clean_tensor, generator=generator, params=params, noise=noise)
            corrupted = (
                output.mul(255.0)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(0, 2, 3, 1)
                .numpy()
            )
        else:
            corrupted = pillow_libjpeg_degrade(
                clean_tiles,
                params,
                noise.permute(0, 2, 3, 1).numpy(),
            )
    return ExactPanel(
        panel=panel,
        seed=seed,
        slot_tiles=np.ascontiguousarray(corrupted[slot_to_target]),
        clean_target_tiles=clean_tiles,
        slot_to_target=slot_to_target,
    )
