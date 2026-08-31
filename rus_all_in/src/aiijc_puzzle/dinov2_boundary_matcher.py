"""Frozen DINOv2 patch-token boundary descriptors for candidate retrieval.

The model is used only as an additional neighbour *nominator*.  It never
changes a tile, predicts an absolute position, or replaces the raw Socket
scores.  Every returned candidate is an original upright tile id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

MODEL_NAME = "vit_small_patch14_dinov2"
IMAGE_SIZE = 98
PATCH_GRID = 7
BAND_WIDTH = 2
TOP_K = 32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DINOv2BoundaryScores:
    """Dense directional similarities for one 576-tile board."""

    right: np.ndarray
    down: np.ndarray

    def __post_init__(self) -> None:
        right = np.ascontiguousarray(self.right, dtype=np.float32)
        down = np.ascontiguousarray(self.down, dtype=np.float32)
        if right.ndim != 2 or right.shape[0] != right.shape[1]:
            raise ValueError("right must be one square score matrix")
        if down.shape != right.shape or not np.isfinite(right).all() or not np.isfinite(down).all():
            raise ValueError("right/down scores must be aligned and finite")
        right.setflags(write=False)
        down.setflags(write=False)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "down", down)


def load_official_dinov2(checkpoint: Path, *, device: torch.device) -> torch.nn.Module:
    """Strict-load the retained official ViT-S/14 checkpoint through timm."""

    import timm

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("DINOv2 checkpoint must contain one state dictionary")
    payload = dict(payload)
    # The official training artifact contains a pretraining-only mask token;
    # timm's inference model intentionally has no corresponding parameter.
    payload.pop("mask_token", None)
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(payload, strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _normalised_tiles(tiles: np.ndarray, *, device: torch.device) -> torch.Tensor:
    values = np.asarray(tiles)
    if values.ndim != 4 or values.shape[1:] != (20, 20, 3) or values.dtype != np.uint8:
        raise ValueError("tiles must be uint8 N x 20 x 20 x 3")
    tensor = torch.from_numpy(np.ascontiguousarray(values)).to(
        device=device, dtype=torch.float32
    )
    tensor = tensor.permute(0, 3, 1, 2).div_(255.0)
    tensor = functional.interpolate(
        tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


@torch.inference_mode()
def extract_patch_tokens(
    model: torch.nn.Module,
    tiles: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return L2-normalised 7x7 DINO patch tokens for every dirty tile."""

    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    inputs = _normalised_tiles(tiles, device=device)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(inputs), batch_size):
        output = model.forward_features(inputs[start : start + batch_size])
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise RuntimeError("unexpected DINOv2 forward_features output")
        patch = output[:, 1:, :]
        if patch.shape[1] != PATCH_GRID * PATCH_GRID:
            raise RuntimeError("DINOv2 dynamic patch grid changed")
        chunks.append(functional.normalize(patch, dim=-1).cpu())
    result = torch.cat(chunks).reshape(len(inputs), PATCH_GRID, PATCH_GRID, -1)
    array = np.ascontiguousarray(result.numpy(), dtype=np.float32)
    if not np.isfinite(array).all():
        raise RuntimeError("DINOv2 patch tokens contain non-finite values")
    return array


def scores_from_patch_tokens(
    patch_tokens: Any,
    *,
    band_width: int = BAND_WIDTH,
) -> DINOv2BoundaryScores:
    """Compare corresponding ordered patch tokens across opposing sides."""

    values = np.asarray(patch_tokens, dtype=np.float32)
    if (
        values.ndim != 4
        or values.shape[1:3] != (PATCH_GRID, PATCH_GRID)
        or not np.isfinite(values).all()
    ):
        raise ValueError("patch_tokens must be finite N x 7 x 7 x D")
    if isinstance(band_width, bool) or not 1 <= band_width <= PATCH_GRID:
        raise ValueError("band_width must be in [1, 7]")
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("patch tokens must have nonzero norm")
    values = values / norm

    # Boundary-to-interior order is shared by both opposing sides.  Spatial
    # order along the seam is preserved; scores average corresponding token
    # cosine similarities instead of collapsing each side to one mean vector.
    right = values[:, :, -band_width:, :][:, :, ::-1, :].reshape(len(values), -1, values.shape[-1])
    left = values[:, :, :band_width, :].reshape(len(values), -1, values.shape[-1])
    bottom = values[:, -band_width:, :, :][:, ::-1, :, :].reshape(len(values), -1, values.shape[-1])
    top = values[:, :band_width, :, :].reshape(len(values), -1, values.shape[-1])
    horizontal = np.einsum("nld,mld->nm", right, left, optimize=True) / right.shape[1]
    vertical = np.einsum("nld,mld->nm", bottom, top, optimize=True) / bottom.shape[1]
    # DINO similarities are finite.  Use a finite diagonal sentinel so the
    # score container can remain fail-closed while top-k still excludes self.
    diagonal_floor = np.float32(-2.0)
    np.fill_diagonal(horizontal, diagonal_floor)
    np.fill_diagonal(vertical, diagonal_floor)
    return DINOv2BoundaryScores(horizontal, vertical)


def freeze_topk(scores: Any, *, k: int = TOP_K) -> np.ndarray:
    """Freeze stable descending candidate identities, excluding self."""

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not np.isfinite(values).all():
        raise ValueError("scores must be one finite square matrix")
    count = len(values)
    if isinstance(k, bool) or not 1 <= k < count:
        raise ValueError("k must be in [1, tile_count)")
    work = values.copy()
    np.fill_diagonal(work, -np.inf)
    result = np.argsort(-work, axis=1, kind="stable")[:, :k]
    return np.ascontiguousarray(result, dtype=np.int32)


def score_dirty_tiles(
    model: torch.nn.Module,
    tiles: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> DINOv2BoundaryScores:
    """End-to-end frozen DINOv2 side scoring for one dirty board."""

    tokens = extract_patch_tokens(model, tiles, device=device, batch_size=batch_size)
    return scores_from_patch_tokens(tokens)


__all__ = [
    "BAND_WIDTH",
    "DINOv2BoundaryScores",
    "IMAGE_SIZE",
    "MODEL_NAME",
    "PATCH_GRID",
    "TOP_K",
    "extract_patch_tokens",
    "freeze_topk",
    "load_official_dinov2",
    "score_dirty_tiles",
    "scores_from_patch_tokens",
]
