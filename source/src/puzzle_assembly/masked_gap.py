"""Two-sided masked-gap generation and listwise pair ranking.

The module contains no dataset access and no target-dependent evaluation.  It
only defines the canonical 20x40 pair representation, the two frozen model
families, loss functions, deterministic HBT/w4 hard-negative groups, and dense
input-derived scoring.  Downward pairs are rotated counter-clockwise so the
proposed neighbour is always on the right.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE, TILE_COUNT, inverse_permutation, validate_permutation


RIGHT = 0
DOWN = 1
PAIR_WIDTH = 2 * TILE
GAP_WIDTH = 4
GAP_START = TILE - GAP_WIDTH // 2
GAP_STOP = TILE + GAP_WIDTH // 2
NEGATIVES = 31
CHECKPOINT_SCHEMA = 1
CHECKPOINT_KIND = "puzzle_masked_gap_gate_checkpoint"


def _validate_tiles(values: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.ndim != 4 or tuple(values.shape[1:]) != (3, TILE, TILE):
        raise ValueError(f"{name} must have shape (N,3,{TILE},{TILE})")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be finite floating point")
    return values


def canonicalize_tiles(values: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Rotate downward examples into the canonical neighbour-on-right frame."""

    values = _validate_tiles(values, name="values")
    if direction.ndim != 1 or len(direction) != len(values):
        raise ValueError("direction must be a vector aligned with values")
    if not bool(torch.all((direction == RIGHT) | (direction == DOWN))):
        raise ValueError("direction values must be RIGHT or DOWN")
    result = values.clone()
    selected = direction == DOWN
    if bool(selected.any()):
        result[selected] = torch.rot90(result[selected], k=1, dims=(-2, -1))
    return result


def canonical_pair_canvas(
    first: torch.Tensor,
    second: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    first = canonicalize_tiles(first, direction)
    second = canonicalize_tiles(second, direction)
    if first.shape != second.shape:
        raise ValueError("first and second tile batches must align")
    return torch.cat([first, second], dim=-1)


def pair_mask(batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if batch < 1:
        raise ValueError("batch must be positive")
    mask = torch.zeros((batch, 1, TILE, PAIR_WIDTH), device=device, dtype=dtype)
    mask[..., GAP_START:GAP_STOP] = 1.0
    return mask


def generator_input(
    raw_first: torch.Tensor,
    raw_second: torch.Tensor,
    denoised_first: torch.Tensor,
    denoised_second: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen seven-channel masked 20x40 generator input."""

    raw = canonical_pair_canvas(raw_first, raw_second, direction)
    denoised = canonical_pair_canvas(denoised_first, denoised_second, direction)
    if raw.shape != denoised.shape:
        raise ValueError("raw and denoised batches must align")
    mask = pair_mask(len(raw), device=raw.device, dtype=raw.dtype)
    visible = 1.0 - mask
    return torch.cat([raw * visible, denoised * visible, mask], dim=1)


def clean_gap_target(
    clean_first: torch.Tensor,
    clean_second: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    canvas = canonical_pair_canvas(clean_first, clean_second, direction)
    return canvas[..., GAP_START:GAP_STOP]


def gap_baselines(denoised_pair_canvas: torch.Tensor) -> dict[str, torch.Tensor]:
    """Original denoised-gap copy and visible-boundary interpolation controls."""

    if denoised_pair_canvas.ndim != 4 or tuple(denoised_pair_canvas.shape[1:]) != (
        3,
        TILE,
        PAIR_WIDTH,
    ):
        raise ValueError("denoised_pair_canvas must have shape (N,3,20,40)")
    left = denoised_pair_canvas[..., GAP_START - 1 : GAP_START]
    right = denoised_pair_canvas[..., GAP_STOP : GAP_STOP + 1]
    copy = denoised_pair_canvas[..., GAP_START:GAP_STOP].clone()
    weights = denoised_pair_canvas.new_tensor([0.2, 0.4, 0.6, 0.8]).view(1, 1, 1, 4)
    interpolation = left * (1.0 - weights) + right * weights
    return {"copy": copy, "interpolation": interpolation}


class _Residual(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.silu(values + self.block(values))


class MaskedGapGenerator(nn.Module):
    """Small full-height CNN that predicts the four hidden clean columns."""

    def __init__(self, width: int = 64, blocks: int = 6) -> None:
        super().__init__()
        if width <= 0 or width % 8 or blocks <= 0:
            raise ValueError("width must be positive/divisible by 8 and blocks positive")
        self.width = int(width)
        self.blocks = int(blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(7, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.body = nn.Sequential(*[_Residual(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 3, 1)

    def config(self) -> dict[str, Any]:
        return {"width": self.width, "blocks": self.blocks}

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or tuple(values.shape[1:]) != (7, TILE, PAIR_WIDTH):
            raise ValueError("generator input must have shape (N,7,20,40)")
        if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
            raise ValueError("generator input must be finite floating point")
        full = torch.sigmoid(self.head(self.body(self.stem(values))))
        return full[..., GAP_START:GAP_STOP]


def ranker_input(
    raw_first: torch.Tensor,
    raw_second: torch.Tensor,
    denoised_first: torch.Tensor,
    denoised_second: torch.Tensor,
    direction: torch.Tensor,
    predicted_gap: torch.Tensor | None,
) -> torch.Tensor:
    """Return equal-capacity 10-channel input for inpaint or direct control."""

    raw = canonical_pair_canvas(raw_first, raw_second, direction)
    denoised = canonical_pair_canvas(denoised_first, denoised_second, direction)
    mask = pair_mask(len(raw), device=raw.device, dtype=raw.dtype)
    inpaint = torch.zeros_like(raw)
    if predicted_gap is not None:
        if predicted_gap.shape != (len(raw), 3, TILE, GAP_WIDTH):
            raise ValueError("predicted_gap must have shape (N,3,20,4)")
        inpaint[..., GAP_START:GAP_STOP] = predicted_gap
    return torch.cat([raw, denoised, inpaint, mask], dim=1)


class PairListwiseRanker(nn.Module):
    """A compact, identical-capacity scorer for inpaint and direct arms."""

    def __init__(self, width: int = 64, blocks: int = 5) -> None:
        super().__init__()
        if width <= 0 or width % 8 or blocks <= 0:
            raise ValueError("width must be positive/divisible by 8 and blocks positive")
        self.width = int(width)
        self.blocks = int(blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(10, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.body = nn.Sequential(*[_Residual(width) for _ in range(blocks)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def config(self) -> dict[str, Any]:
        return {"width": self.width, "blocks": self.blocks, "input_channels": 10}

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or tuple(values.shape[1:]) != (10, TILE, PAIR_WIDTH):
            raise ValueError("ranker input must have shape (N,10,20,40)")
        if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
            raise ValueError("ranker input must be finite floating point")
        return self.head(self.pool(self.body(self.stem(values)))).squeeze(1)


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("prediction and target must be aligned and non-empty")
    return torch.sqrt((prediction - target).square() + float(epsilon) ** 2).mean()


def listwise_pair_loss(
    outgoing_logits: torch.Tensor,
    incoming_logits: torch.Tensor,
    *,
    bce_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Frozen outgoing CE + incoming CE + 0.25 binary CE objective."""

    if outgoing_logits.ndim != 2 or incoming_logits.shape != outgoing_logits.shape:
        raise ValueError("outgoing/incoming logits must be aligned (groups,candidates)")
    if outgoing_logits.shape[1] != NEGATIVES + 1:
        raise ValueError("each group must contain one positive and 31 negatives")
    target = torch.zeros(len(outgoing_logits), dtype=torch.long, device=outgoing_logits.device)
    outgoing_ce = F.cross_entropy(outgoing_logits, target)
    incoming_ce = F.cross_entropy(incoming_logits, target)
    labels = torch.zeros_like(outgoing_logits)
    labels[:, 0] = 1.0
    bce = 0.5 * (
        F.binary_cross_entropy_with_logits(outgoing_logits, labels)
        + F.binary_cross_entropy_with_logits(incoming_logits, labels)
    )
    total = outgoing_ce + incoming_ce + float(bce_weight) * bce
    return total, {
        "total": float(total.detach()),
        "outgoing_ce": float(outgoing_ce.detach()),
        "incoming_ce": float(incoming_ce.detach()),
        "bce": float(bce.detach()),
    }


def listwise_view_loss(
    logits: torch.Tensor,
    *,
    pair_bce_weight: float = 0.25,
) -> torch.Tensor:
    """One-view half of the frozen two-view listwise objective.

    Summing this function for outgoing and incoming logits is exactly
    ``CE_out + CE_in + pair_bce_weight * mean(BCE_out, BCE_in)``.  Keeping
    the views separate permits DDP ``no_sync`` on the first backward while
    preserving the precommitted mathematical objective.
    """

    if logits.ndim != 2 or logits.shape[1] != NEGATIVES + 1:
        raise ValueError("view logits must be aligned (groups,32)")
    target = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    labels = torch.zeros_like(logits)
    labels[:, 0] = 1.0
    return F.cross_entropy(logits, target) + 0.5 * float(pair_bce_weight) * F.binary_cross_entropy_with_logits(
        logits, labels
    )


@dataclass(frozen=True)
class PairGroups:
    first: np.ndarray
    second: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.first), NEGATIVES + 1)
        if self.first.shape != expected or self.second.shape != expected:
            raise ValueError("pair groups must be Gx32")
        if self.direction.shape != (len(self.first),):
            raise ValueError("group directions must align")


def _stable_candidates(cost: np.ndarray, forbidden: set[int], count: int) -> list[int]:
    order = np.argsort(np.asarray(cost), kind="stable")
    chosen = [int(value) for value in order if int(value) not in forbidden]
    if len(chosen) < count:
        raise RuntimeError("not enough hard-negative candidates")
    return chosen[:count]


def hard_negative_groups(
    w4: CompatibilityMatrices,
    slot_to_target: np.ndarray,
) -> tuple[PairGroups, PairGroups]:
    """Build true 1+31 groups using only the precommitted production w4 miner.

    The caller constructs w4 as equal-rank C1 plus HBT with HBT weight four.
    This is one score matrix and not an HBT/w4 candidate union.
    """

    slot_to_target = validate_permutation(slot_to_target, name="slot_to_target")
    position_to_slot = inverse_permutation(slot_to_target)
    outgoing_first: list[list[int]] = []
    outgoing_second: list[list[int]] = []
    incoming_first: list[list[int]] = []
    incoming_second: list[list[int]] = []
    directions: list[int] = []
    for direction, step, matrix in ((RIGHT, 1, w4.right), (DOWN, GRID, w4.down)):
        for position in range(TILE_COUNT):
            row, column = divmod(position, GRID)
            if (direction == RIGHT and column == GRID - 1) or (direction == DOWN and row == GRID - 1):
                continue
            source = int(position_to_slot[position])
            destination = int(position_to_slot[position + step])
            outgoing_negative = _stable_candidates(
                matrix[source], {source, destination}, NEGATIVES
            )
            incoming_negative = _stable_candidates(
                matrix[:, destination], {source, destination}, NEGATIVES
            )
            outgoing_first.append([source] * (NEGATIVES + 1))
            outgoing_second.append([destination, *outgoing_negative])
            incoming_first.append([source, *incoming_negative])
            incoming_second.append([destination] * (NEGATIVES + 1))
            directions.append(direction)
    expected_groups = 2 * GRID * (GRID - 1)
    if len(directions) != expected_groups:
        raise RuntimeError("true pair group count drift")
    direction_array = np.asarray(directions, dtype=np.int8)
    return (
        PairGroups(
            np.asarray(outgoing_first, dtype=np.int32),
            np.asarray(outgoing_second, dtype=np.int32),
            direction_array,
        ),
        PairGroups(
            np.asarray(incoming_first, dtype=np.int32),
            np.asarray(incoming_second, dtype=np.int32),
            direction_array.copy(),
        ),
    )


def blend_with_w4(
    w4: CompatibilityMatrices,
    learned_cost: CompatibilityMatrices,
    *,
    learned_weight: float = 1.0,
    name: str = "frozen_w4_masked_gap_equal_rank_blend",
) -> CompatibilityMatrices:
    """Frozen equal row-rank blend (w4 + learned) / 2 by default."""

    from .compatibility import fuse_ranked_scores

    if learned_weight != 1.0:
        raise ValueError("the frozen gate requires learned_weight=1.0")
    return fuse_ranked_scores(
        {"w4": w4, "learned": learned_cost},
        names=["w4", "learned"],
        weights={"w4": 1.0, "learned": 1.0},
        name=str(name),
    )


def state_dict_payload(
    generator: MaskedGapGenerator,
    inpaint_ranker: PairListwiseRanker,
    direct_ranker: PairListwiseRanker,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if inpaint_ranker.config() != direct_ranker.config():
        raise ValueError("inpaint and direct controls must have identical capacity")
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "kind": CHECKPOINT_KIND,
        "generator_config": generator.config(),
        "ranker_config": inpaint_ranker.config(),
        "generator_state": generator.state_dict(),
        "inpaint_ranker_state": inpaint_ranker.state_dict(),
        "direct_ranker_state": direct_ranker.state_dict(),
        "metadata": dict(metadata),
    }


def module_state_sha256(module: nn.Module) -> str:
    """Hash tensor names, dtypes, shapes, and exact CPU bytes deterministically."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_models(payload: dict[str, Any]) -> tuple[MaskedGapGenerator, PairListwiseRanker, PairListwiseRanker, dict[str, Any]]:
    expected = {
        "schema_version", "kind", "generator_config", "ranker_config",
        "generator_state", "inpaint_ranker_state", "direct_ranker_state", "metadata",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise RuntimeError("masked-gap checkpoint schema mismatch")
    if payload["schema_version"] != CHECKPOINT_SCHEMA or payload["kind"] != CHECKPOINT_KIND:
        raise RuntimeError("masked-gap checkpoint identity mismatch")
    generator = MaskedGapGenerator(**payload["generator_config"])
    ranker_config = dict(payload["ranker_config"])
    if ranker_config.pop("input_channels", None) != 10:
        raise RuntimeError("masked-gap ranker input contract mismatch")
    inpaint = PairListwiseRanker(**ranker_config)
    direct = PairListwiseRanker(**ranker_config)
    generator.load_state_dict(payload["generator_state"], strict=True)
    inpaint.load_state_dict(payload["inpaint_ranker_state"], strict=True)
    direct.load_state_dict(payload["direct_ranker_state"], strict=True)
    if not isinstance(payload["metadata"], dict):
        raise RuntimeError("masked-gap checkpoint metadata mismatch")
    if payload["metadata"].get("safe_for_submission") is not False:
        raise RuntimeError("masked-gap checkpoint must be fail-closed safe_for_submission=false")
    return generator, inpaint, direct, dict(payload["metadata"])
