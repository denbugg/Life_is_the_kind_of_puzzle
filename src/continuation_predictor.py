"""Generative-contrastive oracle gate for extending a correct tile chain.

This module intentionally does not score an A,B,C triple with a cross-encoder.
It uses an exact synthetic, noisy oriented A -> B chain to produce two outputs:

1. a clean RGB prediction for the next tile C;
2. a normalized query embedding that retrieves C from B's frozen affinity
   candidate list through dot products with a separate noisy-tile encoder.

The clean target is supervision only. It never enters the query/candidate
features. Likewise, cardinal direction is removed by rotating all tiles into
one canonical left-to-right orientation; the network receives no direction id,
absolute coordinate, input position, or candidate rank.

The resulting gate asks a genuinely different question from a seam
cross-encoder: can a short correct component predict enough image context to
retrieve its next continuation among semantic hard negatives?
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from config import FS, GRID, NFRAG


# Keep the convention shared by existing directional experiments. The integer
# describes the clean-grid displacement of B/C from the preceding tile, while
# rotations map it into a visual left-to-right chain.
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
NUM_DIRECTIONS = 4
DIRECTION_NAMES: tuple[str, ...] = ("up", "down", "left", "right")
_CANONICAL_ROTATIONS = (3, 1, 2, 0)


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(channels), maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    """Small GroupNorm residual block that remains stable at tiny batch sizes."""

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


def exposure_normalize(tile: Tensor) -> Tensor:
    """Independently normalize every RGB tile while raw RGB stays available."""
    if tile.ndim != 4 or tile.shape[1] != 3:
        raise ValueError(f"tile must have shape (rows,3,H,W), got {tuple(tile.shape)}")
    if not torch.is_floating_point(tile):
        raise TypeError(f"tile must be floating point, got {tile.dtype}")
    mean = tile.mean(dim=(-3, -2, -1), keepdim=True)
    rms = (tile - mean).square().mean(dim=(-3, -2, -1), keepdim=True).add(1.0e-5).sqrt()
    return ((tile - mean) / rms).clamp(-5.0, 5.0)


def orient_to_canonical(tile: Tensor, directions: Tensor) -> Tensor:
    """Rotate per-row square tiles so the requested continuation points right."""
    if tile.ndim != 4 or tile.shape[1] != 3:
        raise ValueError(f"tile must have shape (rows,3,H,W), got {tuple(tile.shape)}")
    if tile.shape[-2] != tile.shape[-1]:
        raise ValueError("canonical rotation needs square tiles")
    if directions.ndim != 1 or directions.shape[0] != tile.shape[0]:
        raise ValueError("directions must contain one cardinal value per tile")
    if tile.device != directions.device:
        raise ValueError("tiles and directions must be on the same device")
    if torch.any(directions < 0) or torch.any(directions >= NUM_DIRECTIONS):
        raise ValueError("directions must lie in [0, 3]")

    oriented = torch.empty_like(tile)
    direction = directions.long()
    for value, turns in enumerate(_CANONICAL_ROTATIONS):
        indices = torch.nonzero(direction.eq(value), as_tuple=False).flatten()
        if indices.numel():
            oriented.index_copy_(
                0,
                indices,
                torch.rot90(tile.index_select(0, indices), turns, dims=(-2, -1)),
            )
    return oriented


def canonical_pair_layout(anchor: Tensor, middle: Tensor, directions: Tensor) -> Tensor:
    """Pack an oriented A -> B chain into raw + normalized 20 x 40 input."""
    if anchor.ndim != 4 or anchor.shape[1] != 3:
        raise ValueError(f"anchor must have shape (rows,3,H,W), got {tuple(anchor.shape)}")
    if middle.shape != anchor.shape:
        raise ValueError("middle must exactly match anchor")
    if anchor.device != middle.device:
        raise ValueError("anchor and middle must be on the same device")

    a = orient_to_canonical(anchor, directions)
    b = orient_to_canonical(middle, directions)
    raw = torch.cat((a, b), dim=-1)
    normalized = torch.cat((exposure_normalize(a), exposure_normalize(b)), dim=-1)
    return torch.cat((raw, normalized), dim=1)


def canonical_candidate_layout(candidate: Tensor, directions: Tensor) -> Tensor:
    """Return raw + normalized candidate-only features in canonical orientation."""
    c = orient_to_canonical(candidate, directions)
    return torch.cat((c, exposure_normalize(c)), dim=1)


class _CandidateEncoder(nn.Module):
    """Noisy-candidate-only encoder; it has no access to A/B/query features."""

    def __init__(self, *, width: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        middle = width * 2
        final = width * 4
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
        self.head = nn.Sequential(
            nn.LayerNorm(final * 2),
            nn.Linear(final * 2, final * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.block1(self.stem(x))
        features = self.block2(self.down1(features))
        features = self.block3(self.down2(features))
        flattened = features.flatten(start_dim=2)
        pooled = torch.cat(
            (
                flattened.mean(dim=-1),
                flattened.var(dim=-1, unbiased=False).add(1.0e-6).sqrt(),
            ),
            dim=-1,
        )
        return self.head(pooled)


class ContinuationPredictor(nn.Module):
    """Predict a clean next tile and retrieve it through separate encoders.

    The context path sees only canonicalized noisy A/B pixels. The candidate
    path sees only canonicalized noisy C pixels. Their sole interaction is a
    normalized dot product in candidate_logits. This explicit separation
    prevents accidentally turning the gate back into a candidate cross-encoder.
    """

    def __init__(
        self,
        *,
        tile_size: int = FS,
        width: int = 16,
        embedding_dim: int = 64,
        dropout: float = 0.05,
        temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if tile_size < 8 or tile_size % 4:
            raise ValueError("tile_size must be divisible by four and at least eight")
        if width < 4 or embedding_dim < 8:
            raise ValueError("width must be >=4 and embedding_dim must be >=8")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        middle = int(width) * 2
        final = int(width) * 4
        self.tile_size = int(tile_size)
        self.width = int(width)
        self.embedding_dim = int(embedding_dim)
        self.dropout = float(dropout)
        self.initial_temperature = float(temperature)

        # Context encoder over the physical A|B chain. The spatial map is
        # retained for the decoder rather than reduced to a seam descriptor.
        self.context_stem = nn.Sequential(
            nn.Conv2d(6, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
        )
        self.context_block1 = _ResidualBlock(width)
        self.context_down1 = nn.Sequential(
            nn.Conv2d(width, middle, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
        )
        self.context_block2 = _ResidualBlock(middle)
        self.context_down2 = nn.Sequential(
            nn.Conv2d(middle, final, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(final), final),
            nn.GELU(),
        )
        self.context_block3 = _ResidualBlock(final)
        self.query_head = nn.Sequential(
            nn.LayerNorm(final * 2),
            nn.Linear(final * 2, final * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Candidate encoder is deliberately a distinct module. It is never
        # fed concatenated A/B/C pixels and therefore cannot inspect a seam.
        self.candidate_encoder = _CandidateEncoder(
            width=int(width), embedding_dim=int(embedding_dim), dropout=float(dropout)
        )

        # A 20x40 context becomes 5x10 after two strides. The learned 1x2
        # reduction retains directional A/B context and emits a 5x5 seed for
        # the clean 20x20 next-tile decoder.
        self.decode_reduce = nn.Sequential(
            nn.Conv2d(final, final, kernel_size=(1, 2), stride=(1, 2), bias=False),
            nn.GroupNorm(_groups(final), final),
            nn.GELU(),
            _ResidualBlock(final),
        )
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(final, middle, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
            _ResidualBlock(middle),
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(middle, width, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            _ResidualBlock(width),
        )
        self.decode_out = nn.Conv2d(width, 3, 3, padding=1)

        # A learnable scale is standard normalized-InfoNCE parametrization.
        # Clamping in candidate_logits prevents unstable extreme temperatures.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def _check_context(self, anchor: Tensor, middle: Tensor, directions: Tensor) -> None:
        expected = (3, self.tile_size, self.tile_size)
        if anchor.ndim != 4 or tuple(anchor.shape[1:]) != expected:
            raise ValueError(f"anchor must have shape (rows,{expected}), got {tuple(anchor.shape)}")
        if middle.shape != anchor.shape:
            raise ValueError("middle must exactly match anchor")
        if directions.ndim != 1 or directions.shape[0] != anchor.shape[0]:
            raise ValueError("directions must have one value per context row")
        if not torch.is_floating_point(anchor) or not torch.is_floating_point(middle):
            raise TypeError("context tiles must be floating point")

    def _check_candidate(self, candidate: Tensor, directions: Tensor) -> None:
        expected = (3, self.tile_size, self.tile_size)
        if candidate.ndim != 4 or tuple(candidate.shape[1:]) != expected:
            raise ValueError(f"candidate must have shape (rows,{expected}), got {tuple(candidate.shape)}")
        if directions.ndim != 1 or directions.shape[0] != candidate.shape[0]:
            raise ValueError("directions must have one value per candidate")
        if not torch.is_floating_point(candidate):
            raise TypeError("candidate tiles must be floating point")

    @staticmethod
    def _pool(features: Tensor) -> Tensor:
        flat = features.flatten(start_dim=2)
        return torch.cat(
            (
                flat.mean(dim=-1),
                flat.var(dim=-1, unbiased=False).add(1.0e-6).sqrt(),
            ),
            dim=-1,
        )

    def _context_features(self, anchor: Tensor, middle: Tensor, directions: Tensor) -> Tensor:
        layout = canonical_pair_layout(anchor, middle, directions)
        features = self.context_block1(self.context_stem(layout))
        features = self.context_block2(self.context_down1(features))
        return self.context_block3(self.context_down2(features))

    def encode_context(
        self,
        anchor: Tensor,
        middle: Tensor,
        directions: Tensor,
        *,
        predict_clean: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """Return normalized A/B query and, optionally, clean-next RGB prediction.

        Retrieval scoring requests only the query, so it does not pay decoder
        FLOPs or retain decoder activations for every candidate-list row.
        """
        self._check_context(anchor, middle, directions)
        if anchor.shape[0] == 0:
            empty_query = anchor.new_empty((0, self.embedding_dim))
            empty_clean = (
                anchor.new_empty((0, 3, self.tile_size, self.tile_size))
                if predict_clean
                else None
            )
            return empty_query, empty_clean
        features = self._context_features(anchor, middle, directions)
        query = F.normalize(self.query_head(self._pool(features)), p=2, dim=-1, eps=1.0e-6)
        if not predict_clean:
            return query, None
        decoded = self.decode_out(self.decode2(self.decode1(self.decode_reduce(features))))
        if tuple(decoded.shape[1:]) != (3, self.tile_size, self.tile_size):
            raise RuntimeError(
                "clean decoder changed spatial size: "
                f"got {tuple(decoded.shape[1:])}, expected {(3, self.tile_size, self.tile_size)}"
            )
        return query, torch.sigmoid(decoded)

    def predict_clean(self, anchor: Tensor, middle: Tensor, directions: Tensor) -> Tensor:
        """Predict canonical clean C from an exact noisy A/B chain."""
        _, prediction = self.encode_context(anchor, middle, directions, predict_clean=True)
        if prediction is None:
            raise RuntimeError("context decoder unexpectedly returned no clean prediction")
        return prediction

    def encode_query(self, anchor: Tensor, middle: Tensor, directions: Tensor) -> Tensor:
        """Return an A/B retrieval query without running the clean decoder."""
        return self.encode_context(anchor, middle, directions, predict_clean=False)[0]

    def encode_candidate(self, candidate: Tensor, directions: Tensor) -> Tensor:
        """Embed a noisy candidate independently of its query/context."""
        self._check_candidate(candidate, directions)
        if candidate.shape[0] == 0:
            return candidate.new_empty((0, self.embedding_dim))
        layout = canonical_candidate_layout(candidate, directions)
        return F.normalize(self.candidate_encoder(layout), p=2, dim=-1, eps=1.0e-6)

    def candidate_logits(self, queries: Tensor, candidate_embeddings: Tensor) -> Tensor:
        """Return one scaled dot-product score per aligned query/candidate row."""
        if queries.shape != candidate_embeddings.shape:
            raise ValueError(
                "queries and candidate_embeddings must have matching shape, got "
                f"{tuple(queries.shape)} vs {tuple(candidate_embeddings.shape)}"
            )
        if queries.ndim != 2 or queries.shape[1] != self.embedding_dim:
            raise ValueError(f"embeddings must be (rows,{self.embedding_dim})")
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (queries * candidate_embeddings).sum(dim=-1)

    def forward(self, anchor: Tensor, middle: Tensor, directions: Tensor) -> dict[str, Tensor]:
        """Convenience forward for a context-only generation/query pass."""
        query, clean = self.encode_context(anchor, middle, directions, predict_clean=True)
        if clean is None:
            raise RuntimeError("context decoder unexpectedly returned no clean prediction")
        return {"query": query, "clean_prediction": clean}


@dataclass(frozen=True)
class ChainRows:
    """Exact oracle A/B/C rows independent of frozen candidate availability."""

    image_ids: Tensor
    anchors: Tensor
    middles: Tensor
    directions: Tensor
    target_indices: Tensor

    @property
    def count(self) -> int:
        return int(self.image_ids.numel())


def _rows_count(rows: Any) -> int:
    image_ids = getattr(rows, "image_ids", None)
    if not isinstance(image_ids, Tensor) or image_ids.ndim != 1:
        raise ValueError("rows must expose one-dimensional Tensor image_ids")
    return int(image_ids.numel())


def _validate_rows(rows: Any, *, batch: int, require_targets: bool = True) -> int:
    count = _rows_count(rows)
    names = ("image_ids", "anchors", "middles", "directions")
    if require_targets:
        names += ("target_indices",)
    for name in names:
        value = getattr(rows, name, None)
        if not isinstance(value, Tensor) or value.ndim != 1 or value.shape[0] != count:
            raise ValueError(f"rows.{name} must be a one-dimensional Tensor of length {count}")
    if count == 0:
        return 0
    if torch.any(rows.image_ids < 0) or torch.any(rows.image_ids >= batch):
        raise ValueError("rows.image_ids lie outside the tile batch")
    for name in ("anchors", "middles", "target_indices"):
        if hasattr(rows, name):
            value = getattr(rows, name)
            if torch.any(value < 0) or torch.any(value >= NFRAG):
                raise ValueError(f"rows.{name} contains an out-of-range tile id")
    if torch.any(rows.directions < 0) or torch.any(rows.directions >= NUM_DIRECTIONS):
        raise ValueError("rows.directions contains an invalid cardinal value")
    return count


def select_oracle_chain_rows(
    middles: Tensor,
    targets: Tensor,
    exists: Tensor,
    *,
    rows_per_image: int | None,
    random_sample: bool,
) -> ChainRows:
    """Select direction-balanced exact continuation rows without graph filtering.

    rows_per_image=None or a non-positive value means all valid chains. That is
    useful for an unbiased held-out clean-prediction L1. Positive values sample
    equally across directions and use replacement only if requested.
    """
    expected = (NFRAG, NUM_DIRECTIONS)
    if middles.ndim != 3 or tuple(middles.shape[1:]) != expected:
        raise ValueError(f"middles must have shape (B,{NFRAG},4)")
    if targets.shape != middles.shape or exists.shape != middles.shape:
        raise ValueError("middles, targets, and exists must share shape (B,576,4)")
    if exists.dtype != torch.bool:
        raise TypeError("exists must be boolean")

    requested = None if rows_per_image is None or rows_per_image <= 0 else int(rows_per_image)
    if requested is not None and requested < NUM_DIRECTIONS:
        raise ValueError("positive rows_per_image must be at least four for direction balance")
    image_parts: list[Tensor] = []
    anchor_parts: list[Tensor] = []
    middle_parts: list[Tensor] = []
    direction_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    base, remainder = divmod(requested or 0, NUM_DIRECTIONS)
    for image in range(middles.shape[0]):
        for direction in range(NUM_DIRECTIONS):
            population = torch.nonzero(exists[image, :, direction], as_tuple=False).flatten()
            if not population.numel():
                continue
            if requested is None:
                picked = population
            else:
                number = base + int(direction < remainder)
                if number <= 0:
                    continue
                if random_sample:
                    if number <= population.numel():
                        picked = population[
                            torch.randperm(population.numel(), device=population.device)[:number]
                        ]
                    else:
                        picked = population[
                            torch.randint(population.numel(), (number,), device=population.device)
                        ]
                else:
                    positions = torch.arange(number, device=population.device)
                    picked = population[(positions * population.numel() // number) % population.numel()]
            image_parts.append(torch.full_like(picked, image))
            anchor_parts.append(picked)
            middle_parts.append(middles[image, picked, direction])
            direction_parts.append(torch.full_like(picked, direction))
            target_parts.append(targets[image, picked, direction])
    if not image_parts:
        empty = torch.empty(0, dtype=torch.long, device=middles.device)
        return ChainRows(empty, empty, empty, empty, empty)
    return ChainRows(
        torch.cat(image_parts),
        torch.cat(anchor_parts),
        torch.cat(middle_parts),
        torch.cat(direction_parts),
        torch.cat(target_parts),
    )


def _check_bag(model: ContinuationPredictor, tiles: Tensor) -> None:
    expected = (NFRAG, 3, model.tile_size, model.tile_size)
    if tiles.ndim != 5 or tuple(tiles.shape[1:]) != expected:
        raise ValueError(f"tiles must have shape (B,{expected}), got {tuple(tiles.shape)}")
    if not torch.is_floating_point(tiles):
        raise TypeError("tiles must be floating point")


def clean_targets_for_rows(clean: Tensor, perm: Tensor, rows: Any) -> Tensor:
    """Fetch canonical clean C targets from full clean train images.

    perm maps noisy input-tile ids to clean row-major cells. Thus clean
    supervision reaches the model only here, after the exact oracle C id has
    been constructed; no clean pixel or coordinate becomes a model input.
    """
    if clean.ndim != 4 or tuple(clean.shape[1:]) != (3, GRID * FS, GRID * FS):
        raise ValueError(f"clean must have shape (B,3,{GRID * FS},{GRID * FS})")
    if perm.ndim != 2 or perm.shape != (clean.shape[0], NFRAG):
        raise ValueError(f"perm must have shape (B,{NFRAG}) aligned with clean")
    count = _validate_rows(rows, batch=clean.shape[0])
    if count == 0:
        return clean.new_empty((0, 3, FS, FS))
    if clean.device != perm.device or clean.device != rows.image_ids.device:
        raise ValueError("clean, perm, and rows must live on one device")
    if torch.any(perm < 0) or torch.any(perm >= NFRAG):
        raise ValueError("perm contains a clean cell outside the puzzle")

    # (B,3,480,480) -> (B,576,3,20,20), clean row-major grid order.
    batch = clean.shape[0]
    clean_tiles = (
        clean.reshape(batch, 3, GRID, FS, GRID, FS)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, NFRAG, 3, FS, FS)
    )
    target_cells = perm[rows.image_ids.long(), rows.target_indices.long()]
    target = clean_tiles[rows.image_ids.long(), target_cells.long()]
    return orient_to_canonical(target, rows.directions.long())


def predict_clean_rows(
    model: ContinuationPredictor,
    tiles: Tensor,
    rows: Any,
    *,
    context_batch: int,
) -> Tensor:
    """Predict clean next tiles for exact chains in bounded context batches."""
    if context_batch < 1:
        raise ValueError("context_batch must be positive")
    _check_bag(model, tiles)
    count = _validate_rows(rows, batch=tiles.shape[0])
    if count == 0:
        return tiles.new_empty((0, 3, model.tile_size, model.tile_size))
    predictions: list[Tensor] = []
    for start in range(0, count, context_batch):
        stop = min(start + context_batch, count)
        image = rows.image_ids[start:stop].long()
        anchor = tiles[image, rows.anchors[start:stop].long()]
        middle = tiles[image, rows.middles[start:stop].long()]
        predictions.append(model.predict_clean(anchor, middle, rows.directions[start:stop].long()))
    return torch.cat(predictions, dim=0)


def score_candidate_rows(
    model: ContinuationPredictor,
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    rows: Any,
    *,
    candidate_batch: int,
    checkpoint_chunks: bool = False,
) -> Tensor:
    """Score each selected B candidate list by query/candidate dot products.

    Candidate rows are never concatenated with A/B pixels. The trainable work
    is query(A,B) dot key(C) over the frozen affinity list, with invalid
    duplicate slots kept at negative infinity for valid InfoNCE normalization.
    """
    if candidate_batch < 1:
        raise ValueError("candidate_batch must be positive")
    _check_bag(model, tiles)
    if candidates.ndim != 3 or candidates.shape[:2] != tiles.shape[:2]:
        raise ValueError("candidates must align with tiles as (B,576,K)")
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask aligned with candidates")
    if torch.any(candidates < 0) or torch.any(candidates >= NFRAG):
        raise ValueError("candidate ids must lie in [0, 576)")
    count = _validate_rows(rows, batch=tiles.shape[0])
    width = int(candidates.shape[-1])
    if count == 0:
        return tiles.new_empty((0, width))

    image = rows.image_ids.long()
    row_candidates = candidates[image, rows.middles.long()]
    row_valid = valid[image, rows.middles.long()]
    if not bool(row_valid.any(dim=-1).all()):
        raise ValueError("every selected B row needs at least one valid affinity candidate")

    anchor = tiles[image, rows.anchors.long()]
    middle = tiles[image, rows.middles.long()]
    queries = model.encode_query(anchor, middle, rows.directions.long())

    flat_valid = row_valid.reshape(-1)
    row_ids = (
        torch.arange(count, device=tiles.device)[:, None]
        .expand(count, width)
        .reshape(-1)[flat_valid]
    )
    candidate_ids = row_candidates.reshape(-1)[flat_valid].long()
    candidate_images = image[:, None].expand(count, width).reshape(-1)[flat_valid]
    candidate_directions = rows.directions.long()[:, None].expand(count, width).reshape(-1)[flat_valid]

    flat_scores: list[Tensor] = []
    use_checkpoint = checkpoint_chunks and torch.is_grad_enabled()
    for start in range(0, candidate_ids.numel(), candidate_batch):
        stop = min(start + candidate_batch, candidate_ids.numel())
        candidate = tiles[candidate_images[start:stop], candidate_ids[start:stop]]
        direction = candidate_directions[start:stop]
        if use_checkpoint:
            keys = checkpoint(model.encode_candidate, candidate, direction, use_reentrant=False)
        else:
            keys = model.encode_candidate(candidate, direction)
        flat_scores.append(model.candidate_logits(queries[row_ids[start:stop]], keys))
    compact_scores = torch.cat(flat_scores, dim=0)
    dense = compact_scores.new_full((count * width,), -torch.inf)
    return dense.masked_scatter(flat_valid, compact_scores).reshape(count, width)


def listwise_info_nce(scores: Tensor, target_slots: Tensor) -> Tensor:
    """Frozen-list InfoNCE: one positive C against B's full valid hard list."""
    if scores.ndim != 2 or target_slots.ndim != 1 or scores.shape[0] != target_slots.shape[0]:
        raise ValueError("scores must be (rows,K) with one target slot per row")
    if not scores.shape[0]:
        raise ValueError("listwise InfoNCE needs at least one row")
    if torch.any(target_slots < 0) or torch.any(target_slots >= scores.shape[1]):
        raise ValueError("target slots lie outside candidate rows")
    target = scores.gather(1, target_slots.long()[:, None]).squeeze(1)
    if not bool(torch.isfinite(target).all()):
        raise ValueError("every listwise target must have a finite score")
    return F.cross_entropy(scores.float(), target_slots.long())


def continuation_rank_metric_sums(scores: Tensor, target_slots: Tensor) -> dict[str, float]:
    """Additive R@1/R@5/MRR diagnostics conditional on candidate coverage."""
    if scores.ndim != 2 or not scores.shape[0]:
        raise ValueError("scores must be a non-empty (rows,K) tensor")
    target = scores.gather(1, target_slots.long()[:, None]).squeeze(1)
    if not bool(torch.isfinite(target).all()):
        raise ValueError("metrics need finite target scores")
    rank = scores.gt(target[:, None]).sum(dim=-1).add(1)
    return {
        "rows": float(rank.numel()),
        "target_r1": float(rank.le(1).sum()),
        "target_r5": float(rank.le(5).sum()),
        "target_mrr_sum": float(rank.float().reciprocal().sum()),
        "target_cross_entropy_sum": float(listwise_info_nce(scores, target_slots).detach() * rank.numel()),
        "target_rank_sum": float(rank.sum()),
    }


def finalize_continuation_metrics(sums: dict[str, float]) -> dict[str, float]:
    """Turn additive conditional rank statistics into named held-out metrics."""
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


def smoke(device: torch.device | str = "cpu") -> dict[str, object]:
    """Data-free CPU-safe shape, gradient, clean-target, and ranking contract."""
    device = torch.device(device)
    torch.manual_seed(2618)
    model = ContinuationPredictor(width=8, embedding_dim=16, dropout=0.0).to(device)
    directions = torch.tensor((RIGHT, UP, LEFT, DOWN), device=device)
    anchor = torch.rand(4, 3, FS, FS, device=device)
    middle = torch.rand_like(anchor)
    query, prediction = model.encode_context(anchor, middle, directions)
    key = model.encode_candidate(torch.rand_like(anchor), directions)
    if tuple(query.shape) != (4, 16) or tuple(prediction.shape) != (4, 3, FS, FS):
        raise AssertionError("context encoder returned unexpected shapes")
    if tuple(key.shape) != (4, 16):
        raise AssertionError("candidate encoder returned an unexpected shape")
    if not torch.allclose(query.norm(dim=-1), torch.ones(4, device=device), atol=3e-5, rtol=3e-5):
        raise AssertionError("context queries are not L2-normalized")
    if not torch.allclose(key.norm(dim=-1), torch.ones(4, device=device), atol=3e-5, rtol=3e-5):
        raise AssertionError("candidate keys are not L2-normalized")

    # Two exact target slots exercise sparse hard-list scoring without making
    # a 576x576 trainable candidate universe.
    tiles = torch.rand(1, NFRAG, 3, FS, FS, device=device)
    candidates = torch.arange(5, device=device).view(1, 1, 5).expand(1, NFRAG, -1).clone()
    valid = torch.ones_like(candidates, dtype=torch.bool)
    rows = ChainRows(
        image_ids=torch.zeros(2, dtype=torch.long, device=device),
        anchors=torch.tensor((0, 1), dtype=torch.long, device=device),
        middles=torch.tensor((1, 2), dtype=torch.long, device=device),
        directions=torch.tensor((RIGHT, DOWN), dtype=torch.long, device=device),
        target_indices=torch.tensor((2, 3), dtype=torch.long, device=device),
    )
    scores = score_candidate_rows(model, tiles, candidates, valid, rows, candidate_batch=7, checkpoint_chunks=True)
    slots = torch.tensor((2, 3), dtype=torch.long, device=device)
    rank_loss = listwise_info_nce(scores, slots)

    clean = torch.rand(1, 3, GRID * FS, GRID * FS, device=device)
    perm = torch.arange(NFRAG, device=device).unsqueeze(0)
    target = clean_targets_for_rows(clean, perm, rows)
    reconstruction = predict_clean_rows(model, tiles, rows, context_batch=3)
    loss = rank_loss + F.l1_loss(reconstruction, target)
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("continuation predictor received no gradients")

    perfect = torch.full_like(scores, -7.0)
    perfect.scatter_(1, slots[:, None], 7.0)
    metrics = finalize_continuation_metrics(continuation_rank_metric_sums(perfect, slots))
    if metrics["continuation_target_r1"] < 0.999 or metrics["continuation_target_mrr"] < 0.999:
        raise AssertionError(f"perfect rank metric failed: {metrics}")
    return {
        "pair_layout": tuple(canonical_pair_layout(anchor, middle, directions).shape),
        "query": tuple(query.shape),
        "clean_prediction": tuple(prediction.shape),
        "candidate_scores": tuple(scores.shape),
        "sample_loss": float(loss.detach()),
        "perfect_target_r1": metrics["continuation_target_r1"],
        "parameters": count_params(model),
    }


__all__: Sequence[str] = (
    "UP", "DOWN", "LEFT", "RIGHT", "NUM_DIRECTIONS", "DIRECTION_NAMES",
    "exposure_normalize", "orient_to_canonical", "canonical_pair_layout",
    "canonical_candidate_layout", "ContinuationPredictor", "ChainRows",
    "select_oracle_chain_rows", "clean_targets_for_rows", "predict_clean_rows",
    "score_candidate_rows", "listwise_info_nce", "continuation_rank_metric_sums",
    "finalize_continuation_metrics", "count_params", "smoke",
)


if __name__ == "__main__":
    print(smoke())
