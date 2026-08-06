"""Context-conditioned continuation ranking on a frozen affinity graph.

The preceding seam ranker scores a proposed direct relation in isolation.  This
module instead receives an **oracle-confirmed** oriented two-tile chain
``A -> B`` and ranks every frozen-affinity candidate ``C`` for the next step
``A -> B -> C`` in that same direction.  It is an intentionally narrow oracle
gate: the clean permutation is used only to construct labelled synthetic
chains, never as a model feature.

Every cardinal direction is rotated into one physical 20 x 60 layout
(``A | B | C`` left-to-right).  The compact CNN sees raw RGB plus an
independently exposure-normalised copy of each tile, so it can use global
three-tile image context without relying only on the B/C seam.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from config import FS, GRID, NFRAG


# Keep the cardinal convention compatible with the existing candidate-ranker:
# the integer describes the clean-grid displacement of the next tile.
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
NUM_DIRECTIONS = 4
DIRECTION_NAMES: tuple[str, ...] = ("up", "down", "left", "right")
_CANONICAL_ROTATIONS = (3, 1, 2, 0)  # UP/DOWN/LEFT/RIGHT -> torch.rot90 k


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(channels), maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _groups(channels)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.layers(x))


def _exposure_normalize(tile: Tensor) -> Tensor:
    """Normalize each tile independently while raw RGB remains available."""
    mean = tile.mean(dim=(-3, -2, -1), keepdim=True)
    rms = (tile - mean).square().mean(dim=(-3, -2, -1), keepdim=True).add(1.0e-5).sqrt()
    return ((tile - mean) / rms).clamp(-5.0, 5.0)


def _orient_to_canonical(tile: Tensor, directions: Tensor) -> Tensor:
    """Rotate tile rows so the requested physical direction points right."""
    oriented = torch.empty_like(tile)
    for direction, turns in enumerate(_CANONICAL_ROTATIONS):
        indices = torch.nonzero(directions.eq(direction), as_tuple=False).flatten()
        if indices.numel():
            oriented.index_copy_(
                0,
                indices,
                torch.rot90(tile.index_select(0, indices), turns, dims=(-2, -1)),
            )
    return oriented


def canonical_triple_layout(anchor: Tensor, middle: Tensor, candidate: Tensor, directions: Tensor) -> Tensor:
    """Pack ``A -> B -> C`` into a canonical six-channel 20 x 60 tensor.

    The returned channels are ``raw RGB`` followed by ``exposure-normalised
    RGB``.  For every direction, A/B/C are all rotated together, so the model
    never receives a direction id or absolute position as an input feature.
    """
    if anchor.ndim != 4 or anchor.shape[1] != 3:
        raise ValueError(f"anchor must have shape (pairs,3,H,W), got {tuple(anchor.shape)}")
    if middle.shape != anchor.shape or candidate.shape != anchor.shape:
        raise ValueError("middle and candidate must exactly match anchor shape")
    if anchor.shape[-2] != anchor.shape[-1]:
        raise ValueError("canonical rotations require square tiles")
    if directions.ndim != 1 or directions.shape[0] != anchor.shape[0]:
        raise ValueError("directions must contain one cardinal value per triple")
    if anchor.device != middle.device or anchor.device != candidate.device or anchor.device != directions.device:
        raise ValueError("all triple tensors and directions must live on one device")
    if not (torch.is_floating_point(anchor) and torch.is_floating_point(middle) and torch.is_floating_point(candidate)):
        raise TypeError("triple image tensors must be floating point")
    if torch.any(directions < 0) or torch.any(directions >= NUM_DIRECTIONS):
        raise ValueError("directions must lie in [0, 3]")

    direction = directions.long()
    a = _orient_to_canonical(anchor, direction)
    b = _orient_to_canonical(middle, direction)
    c = _orient_to_canonical(candidate, direction)
    raw = torch.cat((a, b, c), dim=-1)
    normalized = torch.cat((_exposure_normalize(a), _exposure_normalize(b), _exposure_normalize(c)), dim=-1)
    return torch.cat((raw, normalized), dim=1)


class ContextContinuationRanker(nn.Module):
    """Compact cross-encoder scoring a candidate extension of ``A -> B``.

    No clean-grid coordinate, original input position, candidate rank, or
    direction embedding enters the network.  Direction is removed solely by
    physical rotation before the CNN.
    """

    def __init__(
        self,
        *,
        tile_size: int = FS,
        width: int = 24,
        dropout: float = 0.10,
        context_band: int = 2,
    ) -> None:
        super().__init__()
        if tile_size < 8:
            raise ValueError("tile_size must be at least 8 after two downsampling stages")
        if width < 4:
            raise ValueError("width must be at least 4")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if context_band < 1:
            raise ValueError("context_band must be positive")

        middle = int(width) * 2
        final = int(width) * 3
        self.tile_size = int(tile_size)
        self.width = int(width)
        self.dropout = float(dropout)
        self.context_band = int(context_band)
        self.stem = nn.Sequential(
            nn.Conv2d(6, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
        )
        self.block1 = _ResidualBlock(width)
        self.down1 = nn.Sequential(
            nn.Conv2d(width, middle, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
        )
        self.block2 = _ResidualBlock(middle)
        self.down2 = nn.Sequential(
            nn.Conv2d(middle, final, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(final), final),
            nn.GELU(),
        )
        self.block3 = _ResidualBlock(final)
        # Full field + middle tile + A/B seam band + B/C seam band.  This is a
        # small context-aware summary, not an edge-strip-only model.
        representation = final * 12
        self.head = nn.Sequential(
            nn.LayerNorm(representation),
            nn.Linear(representation, final * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final * 2, 1),
        )

    def _check(self, anchor: Tensor, middle: Tensor, candidate: Tensor, directions: Tensor) -> None:
        expected = (3, self.tile_size, self.tile_size)
        if anchor.ndim != 4 or tuple(anchor.shape[1:]) != expected:
            raise ValueError(f"anchor must have shape (pairs,{expected}), got {tuple(anchor.shape)}")
        if middle.shape != anchor.shape or candidate.shape != anchor.shape:
            raise ValueError("middle and candidate must exactly match anchor")
        if directions.ndim != 1 or directions.shape[0] != anchor.shape[0]:
            raise ValueError("directions must have one row per triple")

    @staticmethod
    def _summary(features: Tensor) -> Tensor:
        flattened = features.flatten(start_dim=2)
        return torch.cat(
            (
                flattened.mean(dim=-1),
                flattened.var(dim=-1, unbiased=False).add(1.0e-6).sqrt(),
                flattened.amax(dim=-1),
            ),
            dim=-1,
        )

    def forward(self, anchor: Tensor, middle: Tensor, candidate: Tensor, directions: Tensor) -> Tensor:
        """Return one listwise logit per ordered ``A -> B -> C`` triple."""
        self._check(anchor, middle, candidate, directions)
        if anchor.shape[0] == 0:
            return anchor.new_empty((0,))
        layout = canonical_triple_layout(anchor, middle, candidate, directions)
        features = self.block1(self.stem(layout))
        features = self.block2(self.down1(features))
        features = self.block3(self.down2(features))
        width = features.shape[-1]
        first_boundary = width // 3
        second_boundary = (2 * width) // 3
        if first_boundary < 1 or second_boundary <= first_boundary:
            raise RuntimeError(f"context feature map is too narrow for triple layout: width={width}")
        band = min(self.context_band, first_boundary, width - second_boundary)
        ab = features[..., first_boundary - band : first_boundary + band]
        bc = features[..., second_boundary - band : second_boundary + band]
        middle_tile = features[..., first_boundary:second_boundary]
        representation = torch.cat(
            (self._summary(features), self._summary(middle_tile), self._summary(ab), self._summary(bc)), dim=-1
        )
        return self.head(representation).squeeze(-1)


def continuation_targets(perm: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return exact oracle chains ``(B, C, valid)`` for each anchor/direction.

    ``perm[b, input_tile]`` maps a shuffled input tile to its clean row-major
    cell.  A row is valid only when both B and C exist: e.g. a right-going row
    requires two cells to the right of A.  Invalid B/C entries are ``-1``.
    """
    if perm.ndim != 2 or perm.shape[1] != NFRAG:
        raise ValueError(f"perm must have shape (B,{NFRAG}), got {tuple(perm.shape)}")
    if torch.any(perm < 0) or torch.any(perm >= NFRAG):
        raise ValueError("perm contains cells outside the puzzle grid")
    cells = perm.long()
    rows = torch.div(cells, GRID, rounding_mode="floor")
    cols = torch.remainder(cells, GRID)
    exists = torch.stack(
        (rows.ge(2), rows.le(GRID - 3), cols.ge(2), cols.le(GRID - 3)), dim=-1
    )
    deltas = torch.tensor(((-GRID), GRID, -1, 1), device=perm.device, dtype=torch.long)
    middle_cells = cells.unsqueeze(-1) + deltas.view(1, 1, NUM_DIRECTIONS)
    target_cells = cells.unsqueeze(-1) + 2 * deltas.view(1, 1, NUM_DIRECTIONS)
    inverse = torch.empty_like(cells)
    input_ids = torch.arange(NFRAG, device=perm.device, dtype=torch.long).expand_as(cells)
    inverse.scatter_(1, cells, input_ids)
    middle = inverse.gather(1, middle_cells.clamp(0, NFRAG - 1).reshape(cells.shape[0], -1)).reshape_as(middle_cells)
    target = inverse.gather(1, target_cells.clamp(0, NFRAG - 1).reshape(cells.shape[0], -1)).reshape_as(target_cells)
    return middle.masked_fill(~exists, -1), target.masked_fill(~exists, -1), exists


def continuation_target_slots(
    candidates: Tensor,
    valid: Tensor,
    middles: Tensor,
    targets: Tensor,
    exists: Tensor,
) -> tuple[Tensor, Tensor]:
    """Find C's slot in B's frozen candidate row for every oracle A/B chain."""
    if candidates.ndim != 3 or candidates.shape[1] != NFRAG:
        raise ValueError(f"candidates must have shape (B,{NFRAG},K)")
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask aligned with candidates")
    expected = (*candidates.shape[:2], NUM_DIRECTIONS)
    if middles.shape != expected or targets.shape != expected or exists.shape != expected:
        raise ValueError("middles, targets, and exists must have shape (B,576,4)")
    if exists.dtype != torch.bool:
        raise ValueError("exists must be boolean")
    if torch.any(candidates < 0) or torch.any(candidates >= NFRAG):
        raise ValueError("candidate indices must lie in [0, 576)")

    batch, _, width = candidates.shape
    safe_middle = middles.clamp(0, NFRAG - 1).long()
    index = safe_middle.unsqueeze(-1).expand(batch, NFRAG, NUM_DIRECTIONS, width)
    rows = candidates.unsqueeze(2).expand(batch, NFRAG, NUM_DIRECTIONS, width).gather(1, index)
    row_valid = valid.unsqueeze(2).expand(batch, NFRAG, NUM_DIRECTIONS, width).gather(1, index)
    matches = row_valid & rows.eq(targets.unsqueeze(-1))
    counts = matches.sum(dim=-1)
    if torch.any(counts.gt(1)):
        raise ValueError("a valid B candidate row contains duplicate target C ids")
    slots = matches.long().argmax(dim=-1)
    return slots, exists & counts.eq(1)


@dataclass(frozen=True)
class ContinuationRows:
    """Complete frozen candidate rows for valid oracle A/B/C continuations."""

    image_ids: Tensor
    anchors: Tensor
    middles: Tensor
    directions: Tensor
    target_slots: Tensor
    target_indices: Tensor

    @property
    def count(self) -> int:
        return int(self.image_ids.numel())


def select_continuation_rows(
    middles: Tensor,
    targets: Tensor,
    target_slots: Tensor,
    available: Tensor,
    *,
    rows_per_image: int,
    random_sample: bool,
) -> ContinuationRows:
    """Choose direction-balanced available oracle continuation rows."""
    if rows_per_image < 1:
        raise ValueError("rows_per_image must be positive")
    if not (middles.shape == targets.shape == target_slots.shape == available.shape):
        raise ValueError("all chain tensors must share shape (B,576,4)")
    if middles.ndim != 3 or middles.shape[1:] != (NFRAG, NUM_DIRECTIONS):
        raise ValueError("expected chain tensors shaped (B,576,4)")
    if available.dtype != torch.bool:
        raise ValueError("available must be boolean")

    base, remainder = divmod(int(rows_per_image), NUM_DIRECTIONS)
    image_parts: list[Tensor] = []
    anchor_parts: list[Tensor] = []
    middle_parts: list[Tensor] = []
    direction_parts: list[Tensor] = []
    slot_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for image in range(middles.shape[0]):
        for direction in range(NUM_DIRECTIONS):
            requested = base + int(direction < remainder)
            population = torch.nonzero(available[image, :, direction], as_tuple=False).flatten()
            if not population.numel() or not requested:
                continue
            if random_sample:
                if requested <= population.numel():
                    picked = population[torch.randperm(population.numel(), device=population.device)[:requested]]
                else:
                    picked = population[torch.randint(population.numel(), (requested,), device=population.device)]
            else:
                positions = torch.arange(requested, device=population.device)
                picked = population[(positions * population.numel() // requested) % population.numel()]
            image_parts.append(torch.full_like(picked, image))
            anchor_parts.append(picked)
            middle_parts.append(middles[image, picked, direction])
            direction_parts.append(torch.full_like(picked, direction))
            slot_parts.append(target_slots[image, picked, direction])
            target_parts.append(targets[image, picked, direction])
    if not image_parts:
        empty = torch.empty(0, dtype=torch.long, device=middles.device)
        return ContinuationRows(empty, empty, empty, empty, empty, empty)
    return ContinuationRows(
        torch.cat(image_parts),
        torch.cat(anchor_parts),
        torch.cat(middle_parts),
        torch.cat(direction_parts),
        torch.cat(slot_parts),
        torch.cat(target_parts),
    )


def score_continuation_rows(
    model: ContextContinuationRanker,
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    rows: ContinuationRows,
    *,
    pair_batch: int,
    checkpoint_chunks: bool = False,
) -> Tensor:
    """Score every valid C in each selected full B candidate list.

    The scorer never expands to all 576 x 575 triples.  It consumes only the
    frozen hard list attached to the known middle tile B.  Invalid duplicate
    slots stay ``-inf`` and cannot affect listwise loss or ranking.
    """
    if pair_batch < 1:
        raise ValueError("pair_batch must be positive")
    if tiles.ndim != 5 or tuple(tiles.shape[1:]) != (NFRAG, 3, model.tile_size, model.tile_size):
        raise ValueError(f"tiles must have shape (B,{NFRAG},3,{model.tile_size},{model.tile_size})")
    if candidates.ndim != 3 or candidates.shape[:2] != tiles.shape[:2]:
        raise ValueError("candidates must align with tiles as (B,576,K)")
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean candidate mask")
    if rows.count == 0:
        return tiles.new_empty((0, candidates.shape[-1]))
    for value, label in (
        (rows.image_ids, "image_ids"),
        (rows.anchors, "anchors"),
        (rows.middles, "middles"),
        (rows.directions, "directions"),
        (rows.target_slots, "target_slots"),
        (rows.target_indices, "target_indices"),
    ):
        if value.ndim != 1 or value.shape[0] != rows.count:
            raise ValueError(f"rows.{label} must have one value per selected row")
    if torch.any(rows.image_ids < 0) or torch.any(rows.image_ids >= tiles.shape[0]):
        raise ValueError("row image ids lie outside the batch")
    if torch.any(rows.anchors < 0) or torch.any(rows.anchors >= NFRAG):
        raise ValueError("row anchors lie outside the tile bag")
    if torch.any(rows.middles < 0) or torch.any(rows.middles >= NFRAG):
        raise ValueError("row middles lie outside the tile bag")
    if torch.any(rows.directions < 0) or torch.any(rows.directions >= NUM_DIRECTIONS):
        raise ValueError("row directions are invalid")

    row_candidates = candidates[rows.image_ids, rows.middles]
    row_valid = valid[rows.image_ids, rows.middles]
    width = row_candidates.shape[-1]
    if torch.any(rows.target_slots < 0) or torch.any(rows.target_slots >= width):
        raise ValueError("row target slots lie outside candidate lists")
    if not bool(row_valid.gather(1, rows.target_slots[:, None]).all()):
        raise ValueError("every scored row needs a valid target C slot")

    flat_valid = row_valid.reshape(-1)
    image_flat = rows.image_ids[:, None].expand(-1, width).reshape(-1)[flat_valid]
    anchor_flat = rows.anchors[:, None].expand(-1, width).reshape(-1)[flat_valid]
    middle_flat = rows.middles[:, None].expand(-1, width).reshape(-1)[flat_valid]
    candidate_flat = row_candidates.reshape(-1)[flat_valid]
    direction_flat = rows.directions[:, None].expand(-1, width).reshape(-1)[flat_valid]
    use_checkpoint = checkpoint_chunks and torch.is_grad_enabled()
    chunks: list[Tensor] = []
    for start in range(0, image_flat.numel(), pair_batch):
        stop = min(start + pair_batch, image_flat.numel())
        anchor = tiles[image_flat[start:stop], anchor_flat[start:stop]]
        middle = tiles[image_flat[start:stop], middle_flat[start:stop]]
        candidate = tiles[image_flat[start:stop], candidate_flat[start:stop]]
        direction = direction_flat[start:stop]
        # Constant convolution shapes prevent expensive cudnn re-tuning on a
        # changing tail while the padded triples never enter the loss.
        count = stop - start
        if count < pair_batch:
            pad = pair_batch - count
            anchor = torch.cat((anchor, anchor[-1:].expand(pad, -1, -1, -1)), dim=0)
            middle = torch.cat((middle, middle[-1:].expand(pad, -1, -1, -1)), dim=0)
            candidate = torch.cat((candidate, candidate[-1:].expand(pad, -1, -1, -1)), dim=0)
            direction = torch.cat((direction, direction[-1:].expand(pad)), dim=0)
        if use_checkpoint:
            scores = checkpoint(model, anchor, middle, candidate, direction, use_reentrant=False)
        else:
            scores = model(anchor, middle, candidate, direction)
        chunks.append(scores[:count])
    flat_scores = torch.cat(chunks, dim=0)
    masked = flat_scores.new_full((rows.count * width,), -torch.inf)
    return masked.masked_scatter(flat_valid, flat_scores).reshape(rows.count, width)


def listwise_cross_entropy(scores: Tensor, target_slots: Tensor) -> Tensor:
    if scores.ndim != 2 or target_slots.ndim != 1 or scores.shape[0] != target_slots.shape[0]:
        raise ValueError("scores must be (rows,K) with one target slot per row")
    if not scores.shape[0]:
        raise ValueError("listwise loss requires at least one row")
    if torch.any(target_slots < 0) or torch.any(target_slots >= scores.shape[1]):
        raise ValueError("target slots lie outside score rows")
    target = scores.gather(1, target_slots.long()[:, None]).squeeze(1)
    if not bool(torch.isfinite(target).all()):
        raise ValueError("every listwise target must have a finite score")
    return F.cross_entropy(scores.float(), target_slots.long())


def continuation_rank_metric_sums(scores: Tensor, target_slots: Tensor) -> dict[str, float]:
    """Additive conditional-on-candidate oracle ranking diagnostics."""
    if scores.ndim != 2 or not scores.shape[0]:
        raise ValueError("scores must be a non-empty (rows,K) tensor")
    target = scores.gather(1, target_slots.long()[:, None]).squeeze(1)
    if not bool(torch.isfinite(target).all()):
        raise ValueError("metrics require finite target scores")
    rank = scores.gt(target[:, None]).sum(dim=1).add(1)
    return {
        "rows": float(rank.numel()),
        "target_r1": float(rank.le(1).sum()),
        "target_r5": float(rank.le(5).sum()),
        "target_mrr_sum": float(rank.float().reciprocal().sum()),
        "target_cross_entropy_sum": float(listwise_cross_entropy(scores, target_slots).detach() * rank.numel()),
        "target_rank_sum": float(rank.sum()),
    }


def finalize_continuation_metrics(sums: dict[str, float]) -> dict[str, float]:
    rows = float(sums.get("rows", 0.0))
    if rows <= 0.0:
        return {
            "continuation_target_r1": 0.0,
            "continuation_target_r5": 0.0,
            "continuation_target_mrr": 0.0,
            "continuation_target_cross_entropy": 0.0,
            "continuation_target_mean_rank": 0.0,
            "continuation_rank_rows": 0.0,
        }
    return {
        "continuation_target_r1": sums.get("target_r1", 0.0) / rows,
        "continuation_target_r5": sums.get("target_r5", 0.0) / rows,
        "continuation_target_mrr": sums.get("target_mrr_sum", 0.0) / rows,
        "continuation_target_cross_entropy": sums.get("target_cross_entropy_sum", 0.0) / rows,
        "continuation_target_mean_rank": sums.get("target_rank_sum", 0.0) / rows,
        "continuation_rank_rows": rows,
    }


def count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _perfect_candidate_graph(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Data-free direct-neighbour graph sufficient for continuation smoke."""
    perm = torch.arange(NFRAG, device=device, dtype=torch.long).unsqueeze(0)
    cells = perm[0]
    rows = torch.div(cells, GRID, rounding_mode="floor")
    cols = torch.remainder(cells, GRID)
    exists = torch.stack((rows.gt(0), rows.lt(GRID - 1), cols.gt(0), cols.lt(GRID - 1)), dim=-1)
    deltas = torch.tensor(((-GRID), GRID, -1, 1), dtype=torch.long, device=device)
    direct = (cells[:, None] + deltas[None]).clamp(0, NFRAG - 1)
    candidates = torch.empty((1, NFRAG, 5), dtype=torch.long, device=device)
    valid = torch.zeros_like(candidates, dtype=torch.bool)
    candidates[0, :, :4] = direct
    valid[0, :, :4] = exists
    candidates[0, :, 4] = torch.remainder(cells + NFRAG // 2, NFRAG)
    valid[0, :, 4] = True
    return perm, candidates, valid


def smoke(device: torch.device | str = "cpu") -> dict[str, object]:
    """Data-free shape, label, listwise-gradient, and rank-contract smoke."""
    device = torch.device(device)
    torch.manual_seed(2468)
    directions = torch.tensor((RIGHT, UP, LEFT, DOWN), device=device)
    anchor = torch.arange(directions.numel() * 3 * FS * FS, dtype=torch.float32, device=device).reshape(-1, 3, FS, FS)
    middle = anchor.add(10_000.0)
    candidate = anchor.add(20_000.0)
    layout = canonical_triple_layout(anchor, middle, candidate, directions)
    if tuple(layout.shape) != (directions.numel(), 6, FS, FS * 3):
        raise AssertionError(f"unexpected canonical triple layout {tuple(layout.shape)}")
    # Reference all-four-view construction guards the direction convention.
    rows = torch.arange(directions.numel(), device=device)
    views = lambda x: torch.stack(tuple(torch.rot90(x, k, dims=(-2, -1)) for k in _CANONICAL_ROTATIONS), dim=1)
    a, b, c = (views(x)[rows, directions] for x in (anchor, middle, candidate))
    reference = torch.cat((torch.cat((a, b, c), dim=-1), torch.cat((_exposure_normalize(a), _exposure_normalize(b), _exposure_normalize(c)), dim=-1)), dim=1)
    if not torch.equal(layout, reference):
        raise AssertionError("canonical triple layout differs from its four-view reference")

    perm, candidates, valid = _perfect_candidate_graph(device)
    middles, targets, exists = continuation_targets(perm)
    if int(exists.sum()) != 4 * (GRID - 2) * GRID:
        raise AssertionError("two-step chain border accounting is inconsistent")
    slots, available = continuation_target_slots(candidates, valid, middles, targets, exists)
    if not bool(available[exists].all()):
        raise AssertionError("perfect direct-neighbour graph missed an oracle continuation")
    selected = select_continuation_rows(middles, targets, slots, available, rows_per_image=16, random_sample=False)
    model = ContextContinuationRanker(width=8, dropout=0.0).to(device)
    tiles = torch.rand(1, NFRAG, 3, FS, FS, device=device)
    scores = score_continuation_rows(model, tiles, candidates, valid, selected, pair_batch=13, checkpoint_chunks=True)
    loss = listwise_cross_entropy(scores, selected.target_slots)
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("context model parameters received no gradient")
    perfect = torch.full_like(scores, -8.0)
    perfect.scatter_(1, selected.target_slots[:, None], 8.0)
    metrics = finalize_continuation_metrics(continuation_rank_metric_sums(perfect, selected.target_slots))
    if metrics["continuation_target_r1"] < 0.999 or metrics["continuation_target_mrr"] < 0.999:
        raise AssertionError(f"perfect continuation rank metric failed: {metrics}")
    return {
        "layout": tuple(layout.shape),
        "rows": selected.count,
        "sample_loss": float(loss.detach()),
        "perfect_target_r1": metrics["continuation_target_r1"],
        "parameters": count_params(model),
    }


__all__: Sequence[str] = (
    "UP", "DOWN", "LEFT", "RIGHT", "NUM_DIRECTIONS", "DIRECTION_NAMES",
    "canonical_triple_layout", "ContextContinuationRanker", "continuation_targets",
    "continuation_target_slots", "ContinuationRows", "select_continuation_rows",
    "score_continuation_rows", "listwise_cross_entropy", "continuation_rank_metric_sums",
    "finalize_continuation_metrics", "count_params", "smoke",
)


if __name__ == "__main__":
    print(smoke())
