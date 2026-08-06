"""Listwise seam ranking on a frozen hard candidate graph.

The affinity encoders are useful as a *recall* mechanism: they shrink a
576-tile bag to a short, hard list for each source tile.  They do not answer
which candidate is the physical neighbour in a particular direction.  This
module supplies that missing operation without turning the problem back into
a 576 x 576 all-pairs cross-encoder.

For a requested direction, an ordered source/target pair is rotated into one
canonical physical layout: source on the left, target on the right, and the
putative seam at the centre of a 20 x 40 image.  ``CandidateSeamRanker`` then
returns one scalar compatibility logit.  A training row scores the *entire*
frozen candidate list and uses a listwise cross-entropy target, never an
independent ``far`` class.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import FS, GRID, NFRAG


# Direction of the target tile relative to the source tile.  These integers
# are deliberately shared by data labels, canonical rotations, and metrics.
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
NUM_DIRECTIONS = 4
DIRECTION_NAMES: tuple[str, ...] = ("up", "down", "left", "right")
_INVERSE_DIRECTION = (DOWN, UP, RIGHT, LEFT)
# torch.rot90's k=1 is counter-clockwise.  Each mapping makes the requested
# source border the right border of the left tile in the canonical pair.
_CANONICAL_ROTATIONS = (3, 1, 2, 0)  # UP, DOWN, LEFT, RIGHT -> torch.rot90 k


def inverse_directions(directions: Tensor) -> Tensor:
    """Return the cardinal inverse for arbitrary-shaped direction tensors."""
    if torch.any(directions < 0) or torch.any(directions >= NUM_DIRECTIONS):
        raise ValueError("directions must lie in [0, 3]")
    table = torch.tensor(_INVERSE_DIRECTION, device=directions.device, dtype=torch.long)
    return table[directions.long()]


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(channels), maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    """A small residual block which keeps the seam at a known location."""

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
    """Per-tile normalization that retains raw RGB as a separate input."""
    mean = tile.mean(dim=(-3, -2, -1), keepdim=True)
    rms = (tile - mean).square().mean(dim=(-3, -2, -1), keepdim=True).add(1.0e-5).sqrt()
    return ((tile - mean) / rms).clamp(-5.0, 5.0)


def _orient_to_canonical(tile: Tensor, directions: Tensor) -> Tensor:
    """Rotate every tile exactly once into the source-left/target-right frame.

    The first implementation stacked all four ``rot90`` variants for *every*
    candidate pair and then gathered one.  Candidate scoring uses thousands of
    pairs at a time, so that multiplied both rotation work and activation
    memory by four.  Here each direction bucket is rotated once and copied
    back to its original row positions.  ``index_copy_`` preserves the order
    and propagates autograd through the selected rotation only.
    """
    oriented = torch.empty_like(tile)
    for direction, turns in enumerate(_CANONICAL_ROTATIONS):
        indices = torch.nonzero(directions.eq(direction), as_tuple=False).flatten()
        if indices.numel():
            rotated = torch.rot90(tile.index_select(0, indices), turns, dims=(-2, -1))
            oriented.index_copy_(0, indices, rotated)
    return oriented


def canonical_pair_layout(source: Tensor, target: Tensor, directions: Tensor) -> Tensor:
    """Pack directed tile pairs into a canonical horizontal seam layout.

    The output has six channels: raw RGB followed by independently
    exposure-normalized RGB.  The left half is always the source tile and the
    right half always the target.  For example, for ``UP`` both tiles rotate
    clockwise, taking the source's original top edge to the central right
    border and the target's original bottom edge to the central left border.
    """
    if source.ndim != 4 or tuple(source.shape[1:]) != (3, source.shape[-2], source.shape[-1]):
        raise ValueError(f"source must have shape (pairs,3,H,W), got {tuple(source.shape)}")
    if target.shape != source.shape:
        raise ValueError(f"target must match source shape, got {tuple(target.shape)} vs {tuple(source.shape)}")
    if source.shape[-2] != source.shape[-1]:
        raise ValueError("canonical rotations require square tiles")
    if directions.ndim != 1 or directions.shape[0] != source.shape[0]:
        raise ValueError("directions must have shape (pairs,)")
    if source.device != target.device or source.device != directions.device:
        raise ValueError("source, target, and directions must share a device")
    if not torch.is_floating_point(source) or not torch.is_floating_point(target):
        raise TypeError("source and target must be floating point tensors")
    if torch.any(directions < 0) or torch.any(directions >= NUM_DIRECTIONS):
        raise ValueError("directions must lie in [0, 3]")

    # Only rotate each physical pair in its requested orientation.  This is
    # deliberately a grouped operation rather than a four-view stack: a
    # 4,096-pair inference chunk now makes 4,096 rotations, not 16,384.
    direction = directions.long()
    oriented_source = _orient_to_canonical(source, direction)
    oriented_target = _orient_to_canonical(target, direction)
    raw = torch.cat((oriented_source, oriented_target), dim=-1)
    normalized = torch.cat((_exposure_normalize(oriented_source), _exposure_normalize(oriented_target)), dim=-1)
    return torch.cat((raw, normalized), dim=1)


class CandidateSeamRanker(nn.Module):
    """Compact cross-encoder that assigns one seam score to an oriented pair."""

    def __init__(
        self,
        *,
        tile_size: int = FS,
        width: int = 32,
        dropout: float = 0.10,
        seam_band: int = 6,
    ) -> None:
        super().__init__()
        if tile_size < 4:
            raise ValueError("tile_size must be at least 4")
        if width < 4:
            raise ValueError("width must be at least 4")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if seam_band < 2:
            raise ValueError("seam_band must be at least 2")

        hidden = int(width) * 2
        self.tile_size = int(tile_size)
        self.width = int(width)
        self.dropout = float(dropout)
        self.seam_band = int(seam_band)
        self.stem = nn.Sequential(
            nn.Conv2d(6, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
        )
        self.block1 = _ResidualBlock(width)
        # One downsample leaves a 10 x 20 feature map for 20px tiles, so the
        # centre seam still occupies several columns rather than one pixel.
        self.down = nn.Sequential(
            nn.Conv2d(width, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
        )
        self.block2 = _ResidualBlock(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 6),
            nn.Linear(hidden * 6, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def _check(self, source: Tensor, target: Tensor, directions: Tensor) -> None:
        expected = (3, self.tile_size, self.tile_size)
        if source.ndim != 4 or tuple(source.shape[1:]) != expected:
            raise ValueError(f"source must have shape (pairs,{expected}), got {tuple(source.shape)}")
        if target.shape != source.shape:
            raise ValueError("target must exactly match source shape")
        if directions.ndim != 1 or directions.shape[0] != source.shape[0]:
            raise ValueError("directions must have one value per pair")

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

    def forward(self, source: Tensor, target: Tensor, directions: Tensor) -> Tensor:
        """Return a scalar compatibility logit for each ordered candidate pair."""
        self._check(source, target, directions)
        if source.shape[0] == 0:
            return source.new_empty((0,))
        layout = canonical_pair_layout(source, target, directions)
        features = self.block1(self.stem(layout))
        features = self.block2(self.down(features))
        width = features.shape[-1]
        band = min(max(2, self.seam_band), width)
        start = (width - band) // 2
        seam_features = features[..., start : start + band]
        representation = torch.cat((self._summary(features), self._summary(seam_features)), dim=-1)
        return self.head(representation).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def neighbor_targets(perm: Tensor) -> tuple[Tensor, Tensor]:
    """Return exact synthetic target tile ids for U/D/L/R and board-valid mask.

    ``perm[b, input_tile]`` is the clean row-major cell occupied by that
    shuffled input tile.  The output target ids are back in *input-tile*
    coordinates, exactly matching a frozen candidate graph.
    """
    if perm.ndim != 2 or perm.shape[1] != NFRAG:
        raise ValueError(f"perm must have shape (B,{NFRAG}), got {tuple(perm.shape)}")
    if torch.any(perm < 0) or torch.any(perm >= NFRAG):
        raise ValueError("perm contains cells outside the puzzle grid")
    cells = perm.long()
    rows = torch.div(cells, GRID, rounding_mode="floor")
    cols = torch.remainder(cells, GRID)
    exists = torch.stack(
        (rows.gt(0), rows.lt(GRID - 1), cols.gt(0), cols.lt(GRID - 1)), dim=-1
    )
    deltas = torch.tensor(((-GRID), GRID, -1, 1), device=perm.device, dtype=torch.long)
    target_cells = cells.unsqueeze(-1) + deltas.view(1, 1, NUM_DIRECTIONS)
    # inverse[b, clean_cell] = input_tile.  A permutation check is intentionally
    # omitted here because this runs every train step; CanvasDataset guarantees it.
    inverse = torch.empty_like(cells)
    tile_ids = torch.arange(NFRAG, device=perm.device, dtype=torch.long).expand_as(cells)
    inverse.scatter_(1, cells, tile_ids)
    safe_cells = target_cells.clamp(0, NFRAG - 1)
    targets = inverse.gather(1, safe_cells.reshape(cells.shape[0], -1)).reshape_as(safe_cells)
    return targets.masked_fill(~exists, -1), exists


def candidate_target_slots(
    candidates: Tensor,
    valid: Tensor,
    targets: Tensor,
    exists: Tensor,
) -> tuple[Tensor, Tensor]:
    """Locate each exact U/D/L/R target inside a deduplicated candidate row."""
    if candidates.ndim != 3 or candidates.shape[1] != NFRAG:
        raise ValueError(f"candidates must have shape (B,{NFRAG},K)")
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask aligned with candidates")
    if targets.shape != (*candidates.shape[:2], NUM_DIRECTIONS):
        raise ValueError("targets must have shape (B,576,4)")
    if exists.shape != targets.shape or exists.dtype != torch.bool:
        raise ValueError("exists must be a boolean mask aligned with targets")
    if torch.any(candidates < 0) or torch.any(candidates >= NFRAG):
        raise ValueError("candidate ids must lie in [0, NFRAG)")
    matches = valid.unsqueeze(-2) & candidates.unsqueeze(-2).eq(targets.unsqueeze(-1))
    counts = matches.sum(dim=-1)
    if torch.any(counts.gt(1)):
        raise ValueError("a valid candidate list contains duplicate target ids")
    slots = matches.long().argmax(dim=-1)
    return slots, exists & counts.eq(1)


@dataclass(frozen=True)
class ListwiseRows:
    """A compact batch of anchor + direction rows with valid list targets."""

    image_ids: Tensor
    anchors: Tensor
    directions: Tensor
    target_slots: Tensor
    target_indices: Tensor

    @property
    def count(self) -> int:
        return int(self.image_ids.numel())


def select_listwise_rows(
    targets: Tensor,
    target_slots: Tensor,
    available: Tensor,
    *,
    rows_per_image: int,
    random_sample: bool,
) -> ListwiseRows:
    """Choose direction-balanced valid listwise rows from each puzzle bag."""
    if rows_per_image < 1:
        raise ValueError("rows_per_image must be positive")
    if targets.shape != target_slots.shape or targets.shape != available.shape:
        raise ValueError("targets, target_slots, and available must share shape (B,576,4)")
    if targets.ndim != 3 or targets.shape[1:] != (NFRAG, NUM_DIRECTIONS):
        raise ValueError("expected (B,576,4) listwise target tensors")
    if available.dtype != torch.bool:
        raise ValueError("available must be boolean")

    base, remainder = divmod(int(rows_per_image), NUM_DIRECTIONS)
    image_parts: list[Tensor] = []
    anchor_parts: list[Tensor] = []
    direction_parts: list[Tensor] = []
    slot_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for image in range(targets.shape[0]):
        for direction in range(NUM_DIRECTIONS):
            requested = base + int(direction < remainder)
            population = torch.nonzero(available[image, :, direction], as_tuple=False).flatten()
            if not population.numel():
                continue
            if random_sample:
                if requested <= population.numel():
                    picked = population[torch.randperm(population.numel(), device=population.device)[:requested]]
                else:
                    picked = population[
                        torch.randint(population.numel(), (requested,), device=population.device)
                    ]
            else:
                # Deterministic, roughly uniform subsampling is preferable for
                # held-out reports: it does not depend on training RNG state.
                indices = torch.arange(requested, device=population.device)
                picked = population[(indices * population.numel() // requested) % population.numel()]
            image_parts.append(torch.full_like(picked, image))
            anchor_parts.append(picked)
            direction_parts.append(torch.full_like(picked, direction))
            slot_parts.append(target_slots[image, picked, direction])
            target_parts.append(targets[image, picked, direction])
    if not image_parts:
        empty = torch.empty(0, dtype=torch.long, device=targets.device)
        return ListwiseRows(empty, empty, empty, empty, empty)
    return ListwiseRows(
        torch.cat(image_parts),
        torch.cat(anchor_parts),
        torch.cat(direction_parts),
        torch.cat(slot_parts),
        torch.cat(target_parts),
    )


def score_candidate_rows(
    model: CandidateSeamRanker,
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    rows: ListwiseRows,
    *,
    pair_batch: int,
    checkpoint_chunks: bool = False,
) -> Tensor:
    """Score each selected *full* hard candidate list in bounded pair chunks.

    This is the central anti-collapse contract: every sampled training row sees
    all valid candidates in its frozen list, so a false candidate is a negative
    only relative to the true candidate in that same list.  No all-pairs score
    tensor is materialized.

    ``checkpoint_chunks`` recomputes each chunk's activations during backward,
    capping training memory at one ``pair_batch`` chunk instead of the sum of
    all scored seams.  Without it, one 96-row step holds ~10 GB of activations
    and spills out of an 8 GB GPU into shared memory, which is far slower than
    the ~33% recompute cost.
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
        (rows.directions, "directions"),
        (rows.target_slots, "target_slots"),
    ):
        if value.ndim != 1 or value.shape[0] != rows.count:
            raise ValueError(f"rows.{label} must have one entry per row")
    if torch.any(rows.image_ids < 0) or torch.any(rows.image_ids >= tiles.shape[0]):
        raise ValueError("row image ids are outside the batch")
    if torch.any(rows.anchors < 0) or torch.any(rows.anchors >= NFRAG):
        raise ValueError("row anchors are outside the tile bag")
    if torch.any(rows.directions < 0) or torch.any(rows.directions >= NUM_DIRECTIONS):
        raise ValueError("row directions are invalid")

    row_candidates = candidates[rows.image_ids, rows.anchors]
    row_valid = valid[rows.image_ids, rows.anchors]
    width = row_candidates.shape[-1]
    if torch.any(rows.target_slots < 0) or torch.any(rows.target_slots >= width):
        raise ValueError("row target slots are outside the candidate list")
    if not bool(row_valid.gather(1, rows.target_slots[:, None]).all()):
        raise ValueError("each scored listwise row must have a valid target slot")

    flat_valid = row_valid.reshape(-1)
    if not bool(flat_valid.any()):
        raise RuntimeError("selected listwise rows contain no valid candidates")
    image_flat = rows.image_ids[:, None].expand(-1, width).reshape(-1)[flat_valid]
    anchor_flat = rows.anchors[:, None].expand(-1, width).reshape(-1)[flat_valid]
    target_flat = row_candidates.reshape(-1)[flat_valid]
    direction_flat = rows.directions[:, None].expand(-1, width).reshape(-1)[flat_valid]
    use_checkpoint = checkpoint_chunks and torch.is_grad_enabled()
    chunks: list[Tensor] = []
    for start in range(0, image_flat.numel(), pair_batch):
        stop = min(start + pair_batch, image_flat.numel())
        source = tiles[image_flat[start:stop], anchor_flat[start:stop]]
        target = tiles[image_flat[start:stop], target_flat[start:stop]]
        direction = direction_flat[start:stop]
        # Pad the tail chunk to a constant shape.  Valid-candidate counts vary
        # per step, and cudnn.benchmark re-autotunes every previously unseen
        # convolution shape; with a varying remainder that turned each step
        # into a multi-second re-benchmark.  The pad rows are sliced off, so
        # they never contribute to loss or metrics.
        count = stop - start
        if count < pair_batch:
            pad = pair_batch - count
            source = torch.cat((source, source[-1:].expand(pad, -1, -1, -1)), dim=0)
            target = torch.cat((target, target[-1:].expand(pad, -1, -1, -1)), dim=0)
            direction = torch.cat((direction, direction[-1:].expand(pad)), dim=0)
        if use_checkpoint:
            chunk = torch.utils.checkpoint.checkpoint(
                model, source, target, direction, use_reentrant=False
            )
        else:
            chunk = model(source, target, direction)
        chunks.append(chunk[:count])
    flat_scores = torch.cat(chunks, dim=0)
    # masked_scatter (rather than a detached indexed assignment) preserves
    # autograd from every valid candidate score back into the seam CNN.
    masked = flat_scores.new_full((rows.count * width,), -torch.inf)
    return masked.masked_scatter(flat_valid, flat_scores).reshape(rows.count, width)


def listwise_cross_entropy(scores: Tensor, target_slots: Tensor) -> Tensor:
    """Cross-entropy over a full candidate list; target must be finite."""
    if scores.ndim != 2 or target_slots.ndim != 1 or scores.shape[0] != target_slots.shape[0]:
        raise ValueError("scores must be (rows,K) and target_slots must be (rows,)")
    if scores.shape[0] == 0:
        raise ValueError("listwise loss needs at least one row")
    if torch.any(target_slots < 0) or torch.any(target_slots >= scores.shape[1]):
        raise ValueError("target slots are outside score rows")
    target_score = scores.gather(1, target_slots.long()[:, None]).squeeze(1)
    if not bool(torch.isfinite(target_score).all()):
        raise ValueError("listwise target must be a valid finite candidate score")
    return F.cross_entropy(scores.float(), target_slots.long())


def rank_metric_sums(scores: Tensor, target_slots: Tensor) -> dict[str, float]:
    """Additive listwise ranking diagnostics for held-out aggregation."""
    if scores.ndim != 2 or scores.shape[0] == 0:
        raise ValueError("scores must be a non-empty (rows,K) tensor")
    target = scores.gather(1, target_slots.long()[:, None]).squeeze(1)
    if not bool(torch.isfinite(target).all()):
        raise ValueError("metrics require finite list targets")
    # Continuous seam scores make ties extremely unlikely.  Counting strictly
    # larger scores gives the conventional one-based target rank.
    rank = scores.gt(target[:, None]).sum(dim=1).add(1)
    return {
        "rows": float(rank.numel()),
        "target_r1": float(rank.le(1).sum()),
        "target_r5": float(rank.le(5).sum()),
        "target_mrr_sum": float(rank.float().reciprocal().sum()),
        "target_cross_entropy_sum": float(listwise_cross_entropy(scores, target_slots).detach() * rank.numel()),
        "target_rank_sum": float(rank.sum()),
    }


def finalize_rank_metrics(sums: dict[str, float]) -> dict[str, float]:
    """Normalize additive listwise metrics, retaining explicit candidate scope."""
    rows = float(sums.get("rows", 0.0))
    if rows <= 0.0:
        return {
            "candidate_target_r1": 0.0,
            "candidate_target_r5": 0.0,
            "candidate_target_mrr": 0.0,
            "candidate_target_cross_entropy": 0.0,
            "candidate_target_mean_rank": 0.0,
            "candidate_rank_rows": 0.0,
        }
    return {
        "candidate_target_r1": sums.get("target_r1", 0.0) / rows,
        "candidate_target_r5": sums.get("target_r5", 0.0) / rows,
        "candidate_target_mrr": sums.get("target_mrr_sum", 0.0) / rows,
        "candidate_target_cross_entropy": sums.get("target_cross_entropy_sum", 0.0) / rows,
        "candidate_target_mean_rank": sums.get("target_rank_sum", 0.0) / rows,
        "candidate_rank_rows": rows,
    }


def smoke(device: torch.device | str = "cpu") -> dict[str, object]:
    """Data-free contract test for canonical orientation, ranking, and gradients."""
    device = torch.device(device)
    torch.manual_seed(4321)
    directions = torch.tensor(
        (RIGHT, UP, LEFT, DOWN, UP, RIGHT, DOWN, LEFT), device=device
    )
    source = torch.arange(
        directions.numel() * 3 * FS * FS, device=device, dtype=torch.float32
    ).reshape(directions.numel(), 3, FS, FS)
    target = source.add(10_000.0)
    layout = canonical_pair_layout(source, target, directions)
    if tuple(layout.shape) != (directions.numel(), 6, FS, FS * 2):
        raise AssertionError(f"unexpected canonical layout shape {tuple(layout.shape)}")
    # Compare against the former all-four-views formula.  The optimization is
    # allowed to reduce work and memory only; it may not change any row,
    # direction convention, or normalized channel.
    source_views = torch.stack(
        tuple(torch.rot90(source, turns, dims=(-2, -1)) for turns in _CANONICAL_ROTATIONS),
        dim=1,
    )
    target_views = torch.stack(
        tuple(torch.rot90(target, turns, dims=(-2, -1)) for turns in _CANONICAL_ROTATIONS),
        dim=1,
    )
    row = torch.arange(directions.numel(), device=device)
    reference_source = source_views[row, directions]
    reference_target = target_views[row, directions]
    reference = torch.cat(
        (
            torch.cat((reference_source, reference_target), dim=-1),
            torch.cat((_exposure_normalize(reference_source), _exposure_normalize(reference_target)), dim=-1),
        ),
        dim=1,
    )
    if not torch.equal(layout, reference):
        raise AssertionError("optimized canonical layout differs from the reference four-view formula")

    # A source gradient check also guards against an accidental detached
    # scatter/index path; model-parameter gradients flow through this layout.
    grad_source = torch.rand_like(source, requires_grad=True)
    grad_target = torch.rand_like(target, requires_grad=True)
    canonical_pair_layout(grad_source, grad_target, directions).square().mean().backward()
    if (
        grad_source.grad is None
        or grad_target.grad is None
        or not bool(torch.isfinite(grad_source.grad).all())
        or not bool(torch.isfinite(grad_target.grad).all())
    ):
        raise AssertionError("canonical layout lost a finite input gradient")
    model = CandidateSeamRanker(width=8, dropout=0.0).to(device)
    logits = model(source / source.max(), target / target.max(), directions)
    if tuple(logits.shape) != (directions.numel(),):
        raise AssertionError(f"unexpected seam logits shape {tuple(logits.shape)}")
    logits.square().mean().backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("model parameters received no gradient through canonical layout")
    toy_scores = torch.tensor(((0.1, 0.8, -0.2), (0.5, 0.4, 0.3)), device=device, requires_grad=True)
    toy_targets = torch.tensor((1, 0), device=device)
    loss = listwise_cross_entropy(toy_scores, toy_targets)
    loss.backward()
    metrics = finalize_rank_metrics(rank_metric_sums(toy_scores.detach(), toy_targets))
    if metrics["candidate_target_r1"] < 0.999 or metrics["candidate_target_mrr"] < 0.999:
        raise AssertionError(f"listwise ranking smoke failed: {metrics}")
    return {
        "layout": tuple(layout.shape),
        "logits": tuple(logits.shape),
        "parameters": count_params(model),
        "listwise_loss": float(loss.detach()),
        "target_r1": metrics["candidate_target_r1"],
    }


__all__: Sequence[str] = (
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "NUM_DIRECTIONS",
    "DIRECTION_NAMES",
    "inverse_directions",
    "canonical_pair_layout",
    "CandidateSeamRanker",
    "count_params",
    "neighbor_targets",
    "candidate_target_slots",
    "ListwiseRows",
    "select_listwise_rows",
    "score_candidate_rows",
    "listwise_cross_entropy",
    "rank_metric_sums",
    "finalize_rank_metrics",
    "smoke",
)


if __name__ == "__main__":
    print(smoke())
