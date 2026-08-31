"""Full-resolution ordered-side matcher for 20x20 puzzle fragments.

This module is an Edge2Vec/TEN-inspired representation arm, not a renderer or
layout solver.  A shared stride-one field keeps one 48-D feature at every input
pixel.  Each physical side remains an ordered sequence of twenty tokens through
the final compatibility calculation; no side is pooled into a single vector.

The four directional heads form twin source/target encoders and include an
explicit raw-RGB/standardised-RGB skip beside the learned field.  Training uses
only exact neighbour identities from organizer-train synthetic shuffles, with
two independently corrupted views and a corruption-consistency term.  No
function predicts replacement pixels or assembles a canvas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TILE_SIZE = 20
SIDE_NAMES = ("left", "right", "top", "bottom")
OPPOSITE_SIDE = (1, 0, 3, 2)
DEFAULT_LOG_SCALE = math.log(8.0)


@dataclass(frozen=True)
class TwinSideOutput:
    """Full-resolution field, ordered side tokens and directional scores."""

    field: torch.Tensor
    sides: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class DirectionalRetrievalOutput:
    """Listwise retrieval objective and target-free training diagnostics."""

    loss: torch.Tensor
    cross_entropy: torch.Tensor
    hard_negative: torch.Tensor
    r1: torch.Tensor
    r5: torch.Tensor


def _normalise_input(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tiles.ndim != 5 or tiles.shape[2:] != (3, TILE_SIZE, TILE_SIZE):
        raise ValueError(
            "tiles must have shape B x N x 3 x 20 x 20, "
            f"got {tuple(tiles.shape)}"
        )
    value = tiles.float() if not torch.is_floating_point(tiles) else tiles
    if not torch.isfinite(value).all():
        raise ValueError("tiles contain non-finite values")
    if bool((value.detach().amax() > 1.5).item()):
        value = value / 255.0
    raw = value.clamp(0.0, 1.0)
    mean = raw.mean(dim=(-2, -1), keepdim=True)
    variance = (raw - mean).square().mean(dim=(-2, -1), keepdim=True)
    standard = ((raw - mean) * torch.rsqrt(variance + 1e-4)).clamp(-3.0, 3.0) / 3.0
    return raw, torch.cat((raw * 2.0 - 1.0, standard), dim=2)


class FullResolutionFieldBlock(nn.Module):
    """Depthwise residual block with no pooling, stride or spatial resize."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dimension)
        self.expand = nn.Conv2d(dimension, 2 * dimension, 1)
        self.depthwise = nn.Conv2d(
            2 * dimension,
            2 * dimension,
            3,
            padding=1,
            groups=2 * dimension,
        )
        self.project = nn.Conv2d(dimension, dimension, 1)
        self.scale = nn.Parameter(torch.full((1, dimension, 1, 1), 0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        branch = self.depthwise(self.expand(self.norm(value)))
        first, second = branch.chunk(2, dim=1)
        return value + self.scale * self.project(F.silu(first) * second)


class OrderedSequenceBlock(nn.Module):
    """Tangent-axis residual mixer that preserves all twenty side positions."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.expand = nn.Linear(dimension, 2 * dimension)
        self.depthwise = nn.Conv1d(
            2 * dimension,
            2 * dimension,
            3,
            padding=1,
            groups=2 * dimension,
        )
        self.project = nn.Linear(dimension, dimension)
        self.scale = nn.Parameter(torch.full((1, 1, dimension), 0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        branch = self.expand(self.norm(value)).transpose(1, 2)
        branch = self.depthwise(branch).transpose(1, 2)
        first, second = branch.chunk(2, dim=2)
        return value + self.scale * self.project(F.silu(first) * second)


def _ordered_sides(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return left/right top-to-bottom and top/bottom left-to-right sequences."""

    if value.ndim != 4 or value.shape[-2:] != (TILE_SIZE, TILE_SIZE):
        raise ValueError("value must have shape batch x channels x 20 x 20")
    return (
        value[..., :, 0].transpose(1, 2),
        value[..., :, -1].transpose(1, 2),
        value[..., 0, :].transpose(1, 2),
        value[..., -1, :].transpose(1, 2),
    )


class FullResolutionTwinSideMatcher(nn.Module):
    """20x20->20x20 field with ordered directional twin side embeddings."""

    def __init__(
        self,
        *,
        dimension: int = 48,
        field_blocks: int = 4,
        sequence_blocks: int = 2,
        raw_skip_gain: float = 0.35,
    ) -> None:
        super().__init__()
        if isinstance(dimension, bool) or dimension < 8:
            raise ValueError("dimension must be an integer >= 8")
        if isinstance(field_blocks, bool) or field_blocks < 1:
            raise ValueError("field_blocks must be positive")
        if isinstance(sequence_blocks, bool) or sequence_blocks < 1:
            raise ValueError("sequence_blocks must be positive")
        if not math.isfinite(raw_skip_gain) or raw_skip_gain <= 0:
            raise ValueError("raw_skip_gain must be finite and positive")
        self.dimension = int(dimension)
        self.field_blocks = int(field_blocks)
        self.sequence_blocks = int(sequence_blocks)
        self.raw_skip_gain = float(raw_skip_gain)
        self.intro = nn.Conv2d(6, dimension, 3, padding=1)
        self.field_body = nn.Sequential(
            *(FullResolutionFieldBlock(dimension) for _ in range(field_blocks))
        )
        self.position = nn.Parameter(torch.empty(1, TILE_SIZE, dimension))
        self.side_type = nn.Parameter(torch.empty(4, 1, dimension))
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.side_type, std=0.02)
        self.sequence_body = nn.Sequential(
            *(OrderedSequenceBlock(dimension) for _ in range(sequence_blocks))
        )
        self.field_heads = nn.ModuleList(
            [nn.Linear(dimension, dimension, bias=False) for _ in SIDE_NAMES]
        )
        self.raw_skip_heads = nn.ModuleList(
            [nn.Linear(6, dimension, bias=False) for _ in SIDE_NAMES]
        )
        self.horizontal_log_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.vertical_log_scale = nn.Parameter(torch.tensor(math.log(8.0)))

    def encode(self, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full field and B,N,4,20,D unit side tokens."""

        _, views = _normalise_input(tiles)
        batch, count = views.shape[:2]
        flat_views = views.reshape(batch * count, 6, TILE_SIZE, TILE_SIZE)
        field = self.field_body(self.intro(flat_views))
        field_sides = _ordered_sides(field)
        raw_sides = _ordered_sides(flat_views)
        encoded: list[torch.Tensor] = []
        for side, (field_side, raw_side) in enumerate(
            zip(field_sides, raw_sides, strict=True)
        ):
            sequence = self.sequence_body(
                field_side + self.position + self.side_type[side : side + 1]
            )
            token = self.field_heads[side](sequence)
            token = token + self.raw_skip_gain * self.raw_skip_heads[side](raw_side)
            encoded.append(F.normalize(token, dim=2))
        sides = torch.stack(encoded, dim=1).reshape(
            batch,
            count,
            4,
            TILE_SIZE,
            self.dimension,
        )
        return field.reshape(batch, count, self.dimension, TILE_SIZE, TILE_SIZE), sides

    def forward(self, tiles: torch.Tensor) -> TwinSideOutput:
        field, sides = self.encode(tiles)
        return TwinSideOutput(
            field=field,
            sides=sides,
            scores=directional_score_matrices(
                sides,
                horizontal_log_scale=self.horizontal_log_scale,
                vertical_log_scale=self.vertical_log_scale,
            ),
        )


def directional_score_matrices(
    source_sides: torch.Tensor,
    target_sides: torch.Tensor | None = None,
    *,
    horizontal_log_scale: torch.Tensor | float = DEFAULT_LOG_SCALE,
    vertical_log_scale: torch.Tensor | float = DEFAULT_LOG_SCALE,
) -> torch.Tensor:
    """Compare corresponding ordered positions, returning B,4,N,N scores."""

    if target_sides is None:
        target_sides = source_sides
    if (
        source_sides.ndim != 5
        or source_sides.shape[2] != 4
        or source_sides.shape[3] != TILE_SIZE
        or target_sides.shape != source_sides.shape
    ):
        raise ValueError("side tensors must have equal shape B x N x 4 x 20 x D")
    count = source_sides.shape[1]
    identity = torch.eye(count, dtype=torch.bool, device=source_sides.device).unsqueeze(0)
    output: list[torch.Tensor] = []
    for direction, opposite in enumerate(OPPOSITE_SIDE):
        log_scale = horizontal_log_scale if direction < 2 else vertical_log_scale
        if not isinstance(log_scale, torch.Tensor):
            log_scale = source_sides.new_tensor(log_scale)
        scale = log_scale.exp().clamp(1.0, 100.0)
        score = torch.einsum(
            "bnld,bmld->bnm",
            source_sides[:, :, direction],
            target_sides[:, :, opposite],
        )
        score = scale * score / float(TILE_SIZE)
        output.append(score.masked_fill(identity, -1e4))
    return torch.stack(output, dim=1)


def directional_neighbour_targets(
    tile_at_position: torch.Tensor,
    *,
    grid: int,
) -> torch.Tensor:
    """Return B,4,N exact tile targets, with -1 on physical borders."""

    layout = tile_at_position.long()
    count = grid * grid
    if grid < 2 or layout.ndim != 2 or layout.shape[1] != count:
        raise ValueError("tile_at_position must have shape B x grid**2 for grid >= 2")
    expected = torch.arange(count, device=layout.device).expand_as(layout)
    if not torch.equal(layout.sort(dim=1).values, expected):
        raise ValueError("each tile_at_position row must be a strict permutation")
    positions = torch.arange(count, device=layout.device)
    row = positions // grid
    column = positions % grid
    targets = torch.full(
        (layout.shape[0], 4, count),
        -1,
        dtype=torch.long,
        device=layout.device,
    )
    definitions = (
        (-1, column > 0),
        (1, column < grid - 1),
        (-grid, row > 0),
        (grid, row < grid - 1),
    )
    for direction, (delta, valid) in enumerate(definitions):
        source = layout[:, valid]
        target = layout[:, positions[valid] + delta]
        targets[:, direction].scatter_(1, source, target)
    return targets


def directional_listwise_loss(
    scores: torch.Tensor,
    tile_at_position: torch.Tensor,
    *,
    grid: int,
    hard_margin: float = 0.15,
    hard_weight: float = 0.25,
) -> DirectionalRetrievalOutput:
    """Full in-board InfoNCE/listwise CE plus explicit hardest-negative margin."""

    count = grid * grid
    if scores.ndim != 4 or scores.shape[1:] != (4, count, count):
        raise ValueError("scores must have shape B x 4 x grid**2 x grid**2")
    if not math.isfinite(hard_margin) or hard_margin < 0:
        raise ValueError("hard_margin must be finite and non-negative")
    if not math.isfinite(hard_weight) or hard_weight < 0:
        raise ValueError("hard_weight must be finite and non-negative")
    targets = directional_neighbour_targets(tile_at_position, grid=grid)
    cross_entropies: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    r1_values: list[torch.Tensor] = []
    r5_values: list[torch.Tensor] = []
    for direction in range(4):
        valid = targets[:, direction] >= 0
        logits = scores[:, direction][valid]
        truth = targets[:, direction][valid]
        cross_entropies.append(F.cross_entropy(logits, truth))
        positive = logits.gather(1, truth[:, None]).squeeze(1)
        negative_mask = torch.zeros_like(logits, dtype=torch.bool)
        negative_mask.scatter_(1, truth[:, None], True)
        hardest = logits.masked_fill(negative_mask, -1e4).amax(1)
        margins.append(F.relu(hardest - positive + hard_margin).mean())
        width = min(5, count)
        top = logits.topk(width, dim=1).indices
        r1_values.append((top[:, 0] == truth).float().mean())
        r5_values.append((top == truth[:, None]).any(1).float().mean())
    cross_entropy = torch.stack(cross_entropies).mean()
    hard_negative = torch.stack(margins).mean()
    return DirectionalRetrievalOutput(
        loss=cross_entropy + hard_weight * hard_negative,
        cross_entropy=cross_entropy,
        hard_negative=hard_negative,
        r1=torch.stack(r1_values).mean(),
        r5=torch.stack(r5_values).mean(),
    )


def dual_corruption_retrieval_loss(
    model: FullResolutionTwinSideMatcher,
    first: TwinSideOutput,
    second: TwinSideOutput,
    tile_at_position: torch.Tensor,
    *,
    grid: int,
    consistency_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train within and across two independent legal corruption views."""

    if first.sides.shape != second.sides.shape:
        raise ValueError("the two corruption views must have equal side shapes")
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("consistency_weight must be finite and non-negative")
    cross_first_second = directional_score_matrices(
        first.sides,
        second.sides,
        horizontal_log_scale=model.horizontal_log_scale,
        vertical_log_scale=model.vertical_log_scale,
    )
    cross_second_first = directional_score_matrices(
        second.sides,
        first.sides,
        horizontal_log_scale=model.horizontal_log_scale,
        vertical_log_scale=model.vertical_log_scale,
    )
    retrievals = (
        directional_listwise_loss(first.scores, tile_at_position, grid=grid),
        directional_listwise_loss(second.scores, tile_at_position, grid=grid),
        directional_listwise_loss(cross_first_second, tile_at_position, grid=grid),
        directional_listwise_loss(cross_second_first, tile_at_position, grid=grid),
    )
    retrieval = torch.stack([item.loss for item in retrievals]).mean()
    consistency = 1.0 - (first.sides * second.sides).sum(dim=-1).mean()
    total = retrieval + consistency_weight * consistency
    return total, {
        "loss": total,
        "retrieval": retrieval,
        "cross_entropy": torch.stack([item.cross_entropy for item in retrievals]).mean(),
        "hard_negative": torch.stack([item.hard_negative for item in retrievals]).mean(),
        "consistency": consistency,
        "r1": torch.stack([item.r1 for item in retrievals[:2]]).mean(),
        "r5": torch.stack([item.r5 for item in retrievals[:2]]).mean(),
    }


@torch.inference_mode()
def twin_right_down_scores(
    model: FullResolutionTwinSideMatcher,
    tiles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matcher-only right/down score matrices for one dirty board."""

    value = np.asarray(tiles)
    if value.ndim != 4 or value.shape[1:] != (20, 20, 3) or value.dtype != np.uint8:
        raise ValueError("tiles must be uint8 N x 20 x 20 x 3")
    tensor = (
        torch.from_numpy(np.ascontiguousarray(value))
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    scores = model(tensor).scores[0].float().cpu().numpy()
    return np.ascontiguousarray(scores[1]), np.ascontiguousarray(scores[3])


__all__ = [
    "DirectionalRetrievalOutput",
    "FullResolutionFieldBlock",
    "FullResolutionTwinSideMatcher",
    "OrderedSequenceBlock",
    "TwinSideOutput",
    "directional_listwise_loss",
    "directional_neighbour_targets",
    "directional_score_matrices",
    "dual_corruption_retrieval_loss",
    "twin_right_down_scores",
]
