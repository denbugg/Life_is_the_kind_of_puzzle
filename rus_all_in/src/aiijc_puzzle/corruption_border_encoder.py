"""Historical E13 corruption-aware border encoder, materialised locally.

The architecture and objective follow historical commits ``a605814`` and
``c0c3fec``: canonical four-pixel side strips, one shared CNN, four linear
direction heads, full-board InfoNCE, batch-hard triplet loss and clean/corrupt
embedding consistency.  Unlike :mod:`socket_matcher`, this encoder has no
whole-tile or board-context path and no transport/dustbin layer.  It is an
auxiliary matcher only; raw tiles remain the sole material used for assembly.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.nn import functional as F

TILE_SIZE = 20
DEFAULT_BORDER = 4
DEFAULT_DIMENSION = 96
DEFAULT_TEMPERATURE = 0.08
DEFAULT_TRIPLET_MARGIN = 0.12
SIDE_NAMES = ("left", "right", "top", "bottom")
OPPOSITE_SIDE = (1, 0, 3, 2)
CORRUPTION_MODES = ("noise", "blur", "jpeg", "erosion", "combined")


@dataclass(frozen=True)
class BorderRetrievalOutput:
    """Four directional score matrices and scalar training diagnostics."""

    loss: torch.Tensor
    info_nce: torch.Tensor
    triplet: torch.Tensor
    r1: torch.Tensor
    r5: torch.Tensor


def e13_curriculum_severity(step: int, steps: int) -> float:
    """Return the historical linear 0.2->1 curriculum without FP overshoot."""

    if (
        isinstance(step, bool)
        or isinstance(steps, bool)
        or not isinstance(step, int)
        or not isinstance(steps, int)
        or steps <= 0
        or not 0 <= step < steps
    ):
        raise ValueError("step must be an integer in [0, steps), with steps positive")
    return min(1.0, 0.20 + 0.80 * step / max(1, steps - 1))


def _normalise_tiles(tiles: torch.Tensor) -> torch.Tensor:
    if tiles.ndim != 5 or tiles.shape[2:] != (3, TILE_SIZE, TILE_SIZE):
        raise ValueError(
            "tiles must have shape B x N x 3 x 20 x 20, "
            f"got {tuple(tiles.shape)}"
        )
    value = tiles.float()
    if bool((value.detach().amax() > 1.5).item()):
        value = value / 255.0
    if not torch.isfinite(value).all():
        raise ValueError("tiles contain non-finite values")
    return value.clamp(0.0, 1.0).mul(2.0).sub(1.0)


def canonical_borders(
    tiles: torch.Tensor,
    *,
    border: int = DEFAULT_BORDER,
) -> torch.Tensor:
    """Return ``B,N,4,C,border,20`` with edge-to-interior orientation."""

    value = _normalise_tiles(tiles)
    if isinstance(border, bool) or not isinstance(border, int) or not 1 <= border <= 10:
        raise ValueError("border must be an integer in [1, 10]")
    left = value[..., :, :border].transpose(-2, -1)
    right = value[..., :, -border:].flip(-1).transpose(-2, -1)
    top = value[..., :border, :]
    bottom = value[..., -border:, :].flip(-2)
    return torch.stack((left, right, top, bottom), dim=2)


class CorruptionAwareBorderEncoder(nn.Module):
    """Checkpoint-independent port of historical E13 ``BorderEncoder``."""

    def __init__(
        self,
        *,
        dimension: int = DEFAULT_DIMENSION,
        border: int = DEFAULT_BORDER,
    ) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not 1 <= border <= 10:
            raise ValueError("border must be in [1, 10]")
        self.dimension = dimension
        self.border = border
        self.shared = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=(1, 2), padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, dimension),
            nn.LayerNorm(dimension),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(dimension, dimension, bias=False) for _ in SIDE_NAMES]
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        batch, count = tiles.shape[:2]
        patches = canonical_borders(tiles, border=self.border).reshape(
            batch * count * 4,
            3,
            self.border,
            TILE_SIZE,
        )
        shared = self.shared(patches).reshape(batch, count, 4, self.dimension)
        sides = torch.stack(
            [head(shared[:, :, side]) for side, head in enumerate(self.heads)],
            dim=2,
        )
        return F.normalize(sides, dim=-1)


def direction_targets(
    *,
    grid: int,
    direction: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-major neighbour targets and physical-query mask."""

    if grid < 2:
        raise ValueError("grid must be at least two")
    if not 0 <= direction < 4:
        raise ValueError("direction must be in [0, 3]")
    count = grid * grid
    index = torch.arange(count, device=device)
    row, column = index // grid, index % grid
    if direction == 0:
        return index - 1, column > 0
    if direction == 1:
        return index + 1, column < grid - 1
    if direction == 2:
        return index - grid, row > 0
    return index + grid, row < grid - 1


def directional_score_matrices(sides: torch.Tensor) -> torch.Tensor:
    """Return cosine scores with self-pairs masked, shape ``B,4,N,N``."""

    if sides.ndim != 4 or sides.shape[2] != 4:
        raise ValueError(f"sides must have shape B x N x 4 x D, got {tuple(sides.shape)}")
    count = sides.shape[1]
    identity = torch.eye(count, dtype=torch.bool, device=sides.device)
    scores = []
    for direction, opposite in enumerate(OPPOSITE_SIDE):
        score = sides[:, :, direction] @ sides[:, :, opposite].transpose(1, 2)
        scores.append(score.masked_fill(identity.unsqueeze(0), -1e4))
    return torch.stack(scores, dim=1)


def border_retrieval_loss(
    sides: torch.Tensor,
    *,
    grid: int,
    temperature: float = DEFAULT_TEMPERATURE,
    triplet_margin: float = DEFAULT_TRIPLET_MARGIN,
) -> BorderRetrievalOutput:
    """Historical full-candidate InfoNCE plus batch-hard triplet objective."""

    if sides.shape[1] != grid * grid:
        raise ValueError("side count must equal grid**2")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(triplet_margin) or triplet_margin < 0:
        raise ValueError("triplet_margin must be finite and non-negative")
    scores = directional_score_matrices(sides)
    info_losses: list[torch.Tensor] = []
    triplet_losses: list[torch.Tensor] = []
    r1_values: list[torch.Tensor] = []
    r5_values: list[torch.Tensor] = []
    count = grid * grid
    batch = sides.shape[0]
    for direction in range(4):
        target, valid = direction_targets(
            grid=grid,
            direction=direction,
            device=sides.device,
        )
        raw = scores[:, direction, valid]
        target_batch = target[valid].unsqueeze(0).expand(batch, -1)
        info_losses.append(
            F.cross_entropy(
                (raw / temperature).reshape(-1, count),
                target_batch.reshape(-1),
            )
        )
        positive = raw.gather(2, target_batch.unsqueeze(2)).squeeze(2)
        negative_mask = torch.zeros_like(raw, dtype=torch.bool)
        negative_mask.scatter_(2, target_batch.unsqueeze(2), True)
        hardest = raw.masked_fill(negative_mask, -1e4).amax(2)
        triplet_losses.append(F.relu(hardest - positive + triplet_margin).mean())
        width = min(5, count)
        top = raw.topk(width, dim=2).indices
        r1_values.append((top[:, :, 0] == target_batch).float().mean())
        r5_values.append((top == target_batch.unsqueeze(2)).any(2).float().mean())
    info_nce = torch.stack(info_losses).mean()
    triplet = torch.stack(triplet_losses).mean()
    return BorderRetrievalOutput(
        loss=info_nce + 0.25 * triplet,
        info_nce=info_nce,
        triplet=triplet,
        r1=torch.stack(r1_values).mean(),
        r5=torch.stack(r5_values).mean(),
    )


def corruption_aware_training_loss(
    clean_sides: torch.Tensor,
    corrupt_sides: torch.Tensor,
    *,
    grid: int,
    temperature: float = DEFAULT_TEMPERATURE,
    triplet_margin: float = DEFAULT_TRIPLET_MARGIN,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact historical E13 weighted clean/corrupt/consistency loss."""

    if clean_sides.shape != corrupt_sides.shape:
        raise ValueError("clean and corrupt side embeddings must have equal shape")
    corrupt = border_retrieval_loss(
        corrupt_sides,
        grid=grid,
        temperature=temperature,
        triplet_margin=triplet_margin,
    )
    clean = border_retrieval_loss(
        clean_sides,
        grid=grid,
        temperature=temperature,
        triplet_margin=triplet_margin,
    )
    consistency = 1.0 - (clean_sides * corrupt_sides).sum(3).mean()
    loss = corrupt.loss + 0.20 * clean.loss + 0.10 * consistency
    return loss, {
        "loss": float(loss.detach()),
        "corrupt_retrieval_loss": float(corrupt.loss.detach()),
        "clean_retrieval_loss": float(clean.loss.detach()),
        "corrupt_info_nce": float(corrupt.info_nce.detach()),
        "corrupt_triplet": float(corrupt.triplet.detach()),
        "consistency": float(consistency.detach()),
        "corrupt_r1": float(corrupt.r1.detach()),
        "corrupt_r5": float(corrupt.r5.detach()),
    }


def _tiles_to_canvas(tiles: np.ndarray) -> np.ndarray:
    value = np.asarray(tiles)
    count = len(value)
    grid = round(math.sqrt(count))
    if (
        value.shape != (count, TILE_SIZE, TILE_SIZE, 3)
        or value.dtype != np.uint8
        or grid * grid != count
    ):
        raise ValueError("tiles must be one uint8 square grid of 20x20 RGB tiles")
    return (
        value.reshape(grid, grid, TILE_SIZE, TILE_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid * TILE_SIZE, grid * TILE_SIZE, 3)
    )


def _canvas_to_tiles(canvas: np.ndarray) -> np.ndarray:
    value = np.asarray(canvas, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3 or value.shape[0] != value.shape[1]:
        raise ValueError("canvas must be square RGB")
    if value.shape[0] % TILE_SIZE:
        raise ValueError("canvas side must be divisible by 20")
    grid = value.shape[0] // TILE_SIZE
    return (
        value.reshape(grid, TILE_SIZE, grid, TILE_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid * grid, TILE_SIZE, TILE_SIZE, 3)
    )


def _jpeg_roundtrip(canvas: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(canvas).save(
        buffer,
        format="JPEG",
        quality=int(quality),
        subsampling=2,
    )
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def corrupt_e13_tiles(
    clean_tiles: np.ndarray,
    rng: np.random.Generator,
    *,
    severity: float,
    mode: str,
) -> np.ndarray:
    """Port the historical noise/blur/JPEG/erosion curriculum exactly."""

    if not np.isfinite(severity) or not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    if mode not in CORRUPTION_MODES:
        raise ValueError(f"mode must be one of {CORRUPTION_MODES}")
    canvas = _tiles_to_canvas(clean_tiles)
    if mode in {"blur", "combined"}:
        radius = 0.25 + 1.10 * severity
        canvas = np.asarray(
            Image.fromarray(canvas).filter(ImageFilter.GaussianBlur(radius)),
            dtype=np.uint8,
        )
    if mode in {"jpeg", "combined"}:
        canvas = _jpeg_roundtrip(canvas, int(round(92 - 57 * severity)))
    tiles = _canvas_to_tiles(canvas).astype(np.float32) / 255.0
    count = len(tiles)
    scale = rng.uniform(
        1.0 - 0.22 * severity,
        1.0 + 0.22 * severity,
        (count, 1, 1, 3),
    )
    bias = rng.uniform(
        -0.10 * severity,
        0.10 * severity,
        (count, 1, 1, 3),
    )
    tiles = tiles * scale + bias
    if mode in {"noise", "combined"}:
        sigma = 0.012 + 0.060 * severity
        tiles += rng.normal(0.0, sigma, tiles.shape).astype(np.float32)
    if mode in {"erosion", "combined"}:
        width = max(1, min(3, int(math.ceil(3 * severity))))
        fill = tiles.mean((1, 2), keepdims=True)
        tiles[:, :width] = fill
        tiles[:, -width:] = fill
        tiles[:, :, :width] = fill
        tiles[:, :, -width:] = fill
    # Historical E13 kept the synthetic corruption in float32 instead of
    # introducing an extra uint8 quantisation before the encoder.
    return np.ascontiguousarray(np.clip(tiles, 0.0, 1.0).astype(np.float32))


@torch.inference_mode()
def e13_right_down_scores(
    model: CorruptionAwareBorderEncoder,
    tiles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dirty-only outgoing right/down cosine score matrices."""

    value = np.asarray(tiles)
    if value.ndim != 4 or value.shape[1:] != (20, 20, 3) or value.dtype != np.uint8:
        raise ValueError("tiles must be uint8 N x 20 x 20 x 3")
    tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 3, 1, 2)
    sides = model(tensor.unsqueeze(0).to(device))
    scores = directional_score_matrices(sides)[0].float().cpu().numpy()
    return np.ascontiguousarray(scores[1]), np.ascontiguousarray(scores[3])


__all__ = [
    "CORRUPTION_MODES",
    "DEFAULT_BORDER",
    "DEFAULT_DIMENSION",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TRIPLET_MARGIN",
    "BorderRetrievalOutput",
    "CorruptionAwareBorderEncoder",
    "canonical_borders",
    "corrupt_e13_tiles",
    "corruption_aware_training_loss",
    "direction_targets",
    "directional_score_matrices",
    "e13_curriculum_severity",
    "e13_right_down_scores",
]
