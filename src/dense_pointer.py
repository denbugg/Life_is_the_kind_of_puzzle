"""Dense all-bag pointer for the oracle component-continuation gate.

The input is an *unordered* corrupted tile bag plus the identities of an
oracle-correct, oriented seed pair ``A -> B``.  The task is to point to the
tile immediately after ``B`` in the same physical direction.  This is not a
seam scorer: candidate tiles are encoded independently and the final score is
one dense dot-product pointer over every unused tile in the bag.

Direction is deliberately never represented by a learned token or an input
coordinate.  Instead, all tiles in an example are physically rotated so that
``A -> B`` always reads left-to-right before either the context encoder or the
shared candidate encoder sees pixels.  The only use of input tile indices is
to gather A/B and mask them from the pointer distribution; there are no
input-order or absolute-grid embeddings.
"""
from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import FS


# These values describe the clean-grid displacement of B from A.  They are
# consumed solely by ``canonicalize_bag``; the neural network sees no direction
# identifier after the physical rotation.
UP: Final[int] = 0
DOWN: Final[int] = 1
LEFT: Final[int] = 2
RIGHT: Final[int] = 3
NUM_DIRECTIONS: Final[int] = 4
DIRECTION_NAMES: Final[tuple[str, ...]] = ("up", "down", "left", "right")

# torch.rot90 turns counter-clockwise.  These turns map each physical A->B
# direction into visual left-to-right orientation.
_CANONICAL_TURNS: Final[tuple[int, ...]] = (3, 1, 2, 0)


def _groups(channels: int, maximum: int = 8) -> int:
    """Return a small GroupNorm group count that divides ``channels``."""
    for groups in range(min(int(channels), maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    """Small residual convolution block stable for tiny puzzle batches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _groups(channels)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.layers(x))


def exposure_normalize(tiles: Tensor) -> Tensor:
    """Normalize RGB exposure independently for every tile in a bag.

    Raw RGB remains present in the model input.  This second view merely makes
    the shared encoder less sensitive to the per-fragment brightness/contrast
    corruption imposed by the task generator.
    """
    if tiles.ndim != 5 or tiles.shape[2] != 3:
        raise ValueError(f"tiles must have shape (B,N,3,H,W), got {tuple(tiles.shape)}")
    if not torch.is_floating_point(tiles):
        raise TypeError(f"tiles must be floating point, got {tiles.dtype}")
    mean = tiles.mean(dim=(-3, -2, -1), keepdim=True)
    rms = (tiles - mean).square().mean(dim=(-3, -2, -1), keepdim=True).add(1.0e-5).sqrt()
    return ((tiles - mean) / rms).clamp(-5.0, 5.0)


def canonicalize_bag(tiles: Tensor, directions: Tensor) -> Tensor:
    """Rotate each complete bag so its oracle A->B direction points right.

    ``directions`` never becomes an embedding or a scalar feature.  Rotating
    every candidate with the seed pair is essential: a candidate's pixels then
    live in the same physical frame as the context while still being encoded
    independently of that context.
    """
    if tiles.ndim != 5 or tiles.shape[2] != 3:
        raise ValueError(f"tiles must have shape (B,N,3,H,W), got {tuple(tiles.shape)}")
    if tiles.shape[-2] != tiles.shape[-1]:
        raise ValueError("canonical rotation requires square tiles")
    if directions.ndim != 1 or directions.shape[0] != tiles.shape[0]:
        raise ValueError("directions must provide one cardinal value per bag")
    if directions.device != tiles.device:
        raise ValueError("directions and tiles must be on the same device")
    if torch.is_floating_point(directions) or directions.dtype == torch.bool:
        raise TypeError(f"directions must contain integer cardinal values, got {directions.dtype}")
    if torch.any(directions < 0) or torch.any(directions >= NUM_DIRECTIONS):
        raise ValueError("directions must lie in [0, 3]")

    result = torch.empty_like(tiles)
    directions = directions.long()
    for direction, turns in enumerate(_CANONICAL_TURNS):
        rows = torch.nonzero(directions.eq(direction), as_tuple=False).flatten()
        if rows.numel():
            result.index_copy_(0, rows, torch.rot90(tiles.index_select(0, rows), turns, dims=(-2, -1)))
    return result


def _tile_views(tiles: Tensor) -> Tensor:
    """Stack raw and independently normalized RGB views into six channels."""
    return torch.cat((tiles, exposure_normalize(tiles)), dim=2)


def _gather_tiles(tiles: Tensor, indices: Tensor) -> Tensor:
    """Gather one tile per bag without exposing index values as features."""
    batch, _, channels, height, width = tiles.shape
    expanded = indices.view(batch, 1, 1, 1, 1).expand(-1, 1, channels, height, width)
    return tiles.gather(1, expanded).squeeze(1)


def _gather_tokens(tokens: Tensor, indices: Tensor) -> Tensor:
    """Gather one embedding per bag without introducing position embeddings."""
    batch, _, width = tokens.shape
    expanded = indices.view(batch, 1, 1).expand(-1, 1, width)
    return tokens.gather(1, expanded).squeeze(1)


class _TileEncoder(nn.Module):
    """The shared, full-tile encoder used for every pointer candidate."""

    def __init__(self, *, width: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        middle = width * 2
        final = width * 4
        self.stem = nn.Sequential(
            nn.Conv2d(6, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
        )
        self.block1 = _ResidualBlock(width)
        self.down1 = nn.Sequential(
            nn.Conv2d(width, middle, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
        )
        self.block2 = _ResidualBlock(middle)
        self.down2 = nn.Sequential(
            nn.Conv2d(middle, final, kernel_size=3, stride=2, padding=1, bias=False),
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

    def forward(self, tile_views: Tensor) -> Tensor:
        """Encode flattened ``(rows,6,H,W)`` tile views into descriptors."""
        x = self.block1(self.stem(tile_views))
        x = self.block2(self.down1(x))
        x = self.block3(self.down2(x))
        flattened = x.flatten(start_dim=2)
        pooled = torch.cat(
            (
                flattened.mean(dim=-1),
                flattened.var(dim=-1, unbiased=False).add(1.0e-6).sqrt(),
            ),
            dim=-1,
        )
        return self.head(pooled)


class _PairContextEncoder(nn.Module):
    """Encode the canonical physical A|B component into a query seed."""

    def __init__(self, *, width: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        middle = width * 2
        final = width * 4
        self.stem = nn.Sequential(
            nn.Conv2d(6, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
        )
        self.block1 = _ResidualBlock(width)
        self.down1 = nn.Sequential(
            nn.Conv2d(width, middle, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
        )
        self.block2 = _ResidualBlock(middle)
        self.down2 = nn.Sequential(
            nn.Conv2d(middle, final, kernel_size=3, stride=2, padding=1, bias=False),
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

    def forward(self, context: Tensor) -> Tensor:
        """Encode raw+normalized canonical A|B pixels, shape ``(B,6,H,2W)``."""
        x = self.block1(self.stem(context))
        x = self.block2(self.down1(x))
        x = self.block3(self.down2(x))
        flattened = x.flatten(start_dim=2)
        pooled = torch.cat(
            (
                flattened.mean(dim=-1),
                flattened.var(dim=-1, unbiased=False).add(1.0e-6).sqrt(),
            ),
            dim=-1,
        )
        return self.head(pooled)


class DensePointerNet(nn.Module):
    """Point from an oracle A->B component to C among all unused bag tiles.

    Candidate scores are dense dot products between one set-conditioned query
    and independent shared-tile keys.  A/B are always excluded internally,
    giving exactly ``N - 2`` legal pointer destinations for an N-tile bag.

    Args:
        tile_size: Square tile side in pixels.  Production uses ``20``.
        width: Base CNN width.  ``24`` is intentionally compact for full bags.
        embedding_dim: Width of tile keys, context query, and bag summaries.
        code_patch: Side of optional clean target code; prediction has
            ``3 * code_patch * code_patch`` values in RGB [0,1].  The code is
            an output/loss target only, never an input.
        dropout: Projection-head dropout.
        temperature: Initial pointer softmax temperature.
    """

    def __init__(
        self,
        *,
        tile_size: int = FS,
        width: int = 24,
        embedding_dim: int = 128,
        code_patch: int = 4,
        dropout: float = 0.05,
        temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if tile_size < 8 or tile_size % 4:
            raise ValueError("tile_size must be divisible by four and at least eight")
        if width < 4:
            raise ValueError("width must be at least four")
        if embedding_dim < 8:
            raise ValueError("embedding_dim must be at least eight")
        if code_patch < 1:
            raise ValueError("code_patch must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.tile_size = int(tile_size)
        self.width = int(width)
        self.embedding_dim = int(embedding_dim)
        self.code_patch = int(code_patch)
        self.dropout = float(dropout)
        self.initial_temperature = float(temperature)
        self.code_dim = 3 * self.code_patch * self.code_patch

        self.tile_encoder = _TileEncoder(
            width=self.width, embedding_dim=self.embedding_dim, dropout=self.dropout
        )
        self.context_encoder = _PairContextEncoder(
            width=self.width, embedding_dim=self.embedding_dim, dropout=self.dropout
        )

        # The context combines visual A|B evidence with its two shared-tile
        # descriptors.  It has no input-order or clean-grid features.
        self.seed_fuse = nn.Sequential(
            nn.LayerNorm(3 * self.embedding_dim),
            nn.Linear(3 * self.embedding_dim, 2 * self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )

        # A single attention pool plus a mean pool is a compact permutation-
        # invariant bag conditioner.  It lets the component query use scene
        # context while preserving a pure, independently encoded pointer key.
        self.pool_query = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.pool_key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.pool_value = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.query_fuse = nn.Sequential(
            nn.LayerNorm(3 * self.embedding_dim),
            nn.Linear(3 * self.embedding_dim, 2 * self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )

        self.pointer_query = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.pointer_key = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.code_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.code_dim),
            nn.Sigmoid(),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / float(temperature))))

    def _check_inputs(
        self,
        tiles: Tensor,
        anchor_indices: Tensor,
        middle_indices: Tensor,
        directions: Tensor,
    ) -> None:
        expected_tail = (3, self.tile_size, self.tile_size)
        if tiles.ndim != 5 or tuple(tiles.shape[2:]) != expected_tail:
            raise ValueError(
                "tiles must have shape (B,N,3,H,W) with "
                f"tail {expected_tail}, got {tuple(tiles.shape)}"
            )
        if not torch.is_floating_point(tiles):
            raise TypeError(f"tiles must be floating point, got {tiles.dtype}")
        batch, count = tiles.shape[:2]
        if count < 3:
            raise ValueError("a dense pointer needs at least three tiles")
        for name, indices in (("anchor_indices", anchor_indices), ("middle_indices", middle_indices)):
            if indices.ndim != 1 or indices.shape[0] != batch:
                raise ValueError(f"{name} must have shape ({batch},), got {tuple(indices.shape)}")
            if indices.device != tiles.device:
                raise ValueError(f"{name} and tiles must be on the same device")
            if torch.is_floating_point(indices) or indices.dtype == torch.bool:
                raise TypeError(f"{name} must contain integer tile identities, got {indices.dtype}")
            if torch.any(indices < 0) or torch.any(indices >= count):
                raise ValueError(f"{name} contains an out-of-range tile index")
        if torch.any(anchor_indices.eq(middle_indices)):
            raise ValueError("anchor_indices and middle_indices must differ")
        if directions.ndim != 1 or directions.shape[0] != batch:
            raise ValueError(f"directions must have shape ({batch},), got {tuple(directions.shape)}")
        if directions.device != tiles.device:
            raise ValueError("directions and tiles must be on the same device")
        if torch.is_floating_point(directions) or directions.dtype == torch.bool:
            raise TypeError(f"directions must contain integer cardinal values, got {directions.dtype}")

    @staticmethod
    def _available_mask(anchor_indices: Tensor, middle_indices: Tensor, count: int) -> Tensor:
        """Make the all-bag candidate mask, excluding exactly the two seed tiles."""
        batch = anchor_indices.shape[0]
        valid = torch.ones(batch, count, dtype=torch.bool, device=anchor_indices.device)
        valid.scatter_(1, anchor_indices[:, None], False)
        valid.scatter_(1, middle_indices[:, None], False)
        return valid

    def encode_tiles(self, canonical_tiles: Tensor) -> Tensor:
        """Encode every candidate independently using the shared full-tile CNN."""
        if canonical_tiles.ndim != 5:
            raise ValueError("canonical_tiles must have shape (B,N,3,H,W)")
        batch, count, channels, height, width = canonical_tiles.shape
        views = _tile_views(canonical_tiles).reshape(batch * count, 6, height, width)
        encoded = self.tile_encoder(views)
        return encoded.reshape(batch, count, self.embedding_dim)

    def forward(
        self,
        tiles: Tensor,
        anchor_indices: Tensor,
        middle_indices: Tensor,
        directions: Tensor,
    ) -> dict[str, Tensor]:
        """Return masked dense pointer logits and optional clean-code prediction.

        ``tiles`` is the complete shuffled bag.  ``anchor_indices`` and
        ``middle_indices`` choose the supplied correct A->B component; neither
        index value is converted into a feature.  Every other tile is scored.
        """
        self._check_inputs(tiles, anchor_indices, middle_indices, directions)
        batch, count = tiles.shape[:2]
        anchor_indices = anchor_indices.long()
        middle_indices = middle_indices.long()
        canonical_tiles = canonicalize_bag(tiles, directions)
        tile_embeddings = self.encode_tiles(canonical_tiles)
        anchor_embedding = _gather_tokens(tile_embeddings, anchor_indices)
        middle_embedding = _gather_tokens(tile_embeddings, middle_indices)

        anchor_tile = _gather_tiles(canonical_tiles, anchor_indices)
        middle_tile = _gather_tiles(canonical_tiles, middle_indices)
        raw_context = torch.cat((anchor_tile, middle_tile), dim=-1)
        normalized_context = torch.cat(
            (
                exposure_normalize(anchor_tile.unsqueeze(1)).squeeze(1),
                exposure_normalize(middle_tile.unsqueeze(1)).squeeze(1),
            ),
            dim=-1,
        )
        context = torch.cat((raw_context, normalized_context), dim=1)
        pair_embedding = self.context_encoder(context)
        seed_query = self.seed_fuse(
            torch.cat((pair_embedding, anchor_embedding, middle_embedding), dim=-1)
        )

        valid = self._available_mask(anchor_indices, middle_indices, count)
        pool_logits = (
            self.pool_key(tile_embeddings) * self.pool_query(seed_query).unsqueeze(1)
        ).sum(dim=-1) / math.sqrt(self.embedding_dim)
        # ``-inf`` is mathematically natural but made fp16 GradScaler runs on
        # some WDDM/CUDA stacks fragile: a later reduction can propagate an
        # infinite intermediate even though the corresponding probability is
        # zero.  The finite dtype floor has the same practical softmax effect
        # here (there are always N-2 valid candidates), while keeping every
        # tensor observable by the trainer finite.
        pool_logits = pool_logits.masked_fill(~valid, torch.finfo(pool_logits.dtype).min)
        attention = torch.softmax(pool_logits, dim=-1)
        attention_summary = torch.bmm(attention.unsqueeze(1), self.pool_value(tile_embeddings)).squeeze(1)
        valid_count = valid.sum(dim=-1, keepdim=True).to(tile_embeddings.dtype)
        mean_summary = (tile_embeddings * valid.unsqueeze(-1)).sum(dim=1) / valid_count
        query = self.query_fuse(torch.cat((seed_query, attention_summary, mean_summary), dim=-1))

        query_key = F.normalize(self.pointer_query(query), dim=-1, eps=1.0e-6)
        candidate_keys = F.normalize(self.pointer_key(tile_embeddings), dim=-1, eps=1.0e-6)
        scale = self.logit_scale.exp().clamp(min=1.0, max=100.0)
        logits = (candidate_keys * query_key.unsqueeze(1)).sum(dim=-1) * scale
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

        return {
            "logits": logits,
            "query": query,
            "candidate_embeddings": tile_embeddings,
            "candidate_keys": candidate_keys,
            "attention": attention,
            "candidate_mask": valid,
            "code": self.code_head(query),
        }


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def smoke(device: torch.device | str = "cpu") -> dict[str, tuple[int, ...] | float]:
    """Exercise a tiny CPU-safe forward/backward pass without puzzle data."""
    device = torch.device(device)
    model = DensePointerNet(width=8, embedding_dim=32, code_patch=2, dropout=0.0).to(device)
    tiles = torch.rand(1, 9, 3, FS, FS, device=device)
    anchor = torch.tensor([1], device=device)
    middle = torch.tensor([3], device=device)
    direction = torch.tensor([RIGHT], device=device)
    target = torch.tensor([5], device=device)
    target_code = torch.rand(1, 12, device=device)
    output = model(tiles, anchor, middle, direction)
    if output["logits"].shape != (1, 9):
        raise AssertionError(f"unexpected pointer shape {tuple(output['logits'].shape)}")
    floor = torch.finfo(output["logits"].dtype).min
    if bool(output["candidate_mask"][0, anchor.item()]) or not bool(
        output["logits"][0, anchor.item()].eq(floor)
    ):
        raise AssertionError("anchor was not masked from dense pointer logits")
    if bool(output["candidate_mask"][0, middle.item()]) or not bool(
        output["logits"][0, middle.item()].eq(floor)
    ):
        raise AssertionError("middle was not masked from dense pointer logits")
    loss = F.cross_entropy(output["logits"], target) + 0.1 * F.l1_loss(output["code"], target_code)
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError("dense pointer smoke produced a non-finite loss")
    return {
        "logits": tuple(output["logits"].shape),
        "code": tuple(output["code"].shape),
        "loss": float(loss.detach().cpu()),
    }


__all__ = [
    "DOWN",
    "DIRECTION_NAMES",
    "DensePointerNet",
    "LEFT",
    "NUM_DIRECTIONS",
    "RIGHT",
    "UP",
    "canonicalize_bag",
    "count_params",
    "exposure_normalize",
    "smoke",
]


if __name__ == "__main__":
    print(smoke())
    print("parameters:", count_params(DensePointerNet()))
