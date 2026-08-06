"""Local frontier-inpainting pointer for a shuffled 24 x 24 tile bag.

The model deliberately has no absolute grid-position input.  A query consists
only of a relative 5 x 5 neighbourhood whose centre is an unknown tile.  Its
contextual centre token points to one of the independently encoded bag tiles,
so re-indexing the bag re-indexes the logits in exactly the same way.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


TILE_COUNT = 24 * 24
TILE_SIZE = 20


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _TileEncoder(nn.Module):
    """Shared dirty-tile encoder; no bag index can enter the descriptor."""

    def __init__(self, d: int) -> None:
        super().__init__()
        width = max(24, d // 5)
        middle = max(32, d // 2)
        self.features = nn.Sequential(
            nn.Conv2d(6, width, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            nn.Conv2d(width, middle, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
            nn.Conv2d(middle, d, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(d), d),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, tiles: Tensor) -> Tensor:
        # Keep raw colour while adding an exposure-normalized view.  Both are
        # computed per tile, hence this remains permutation equivariant.
        mean = tiles.mean(dim=(-3, -2, -1), keepdim=True)
        rms = (tiles - mean).square().mean(dim=(-3, -2, -1), keepdim=True)
        normalized = ((tiles - mean) / rms.add(1.0e-5).sqrt()).clamp(-5.0, 5.0)
        x = self.features(torch.cat((tiles, normalized), dim=1))
        flat = x.flatten(start_dim=2)
        stats = torch.cat(
            (flat.mean(dim=-1), flat.var(dim=-1, unbiased=False).add(1.0e-6).sqrt()),
            dim=-1,
        )
        return F.normalize(self.project(stats), dim=-1)


class FrontierPointer(nn.Module):
    """Predict the missing centre tile from a relative local frontier.

    ``context_indices`` and ``occupied`` both have shape ``(B,Q,window**2)``.
    An index of ``-1`` denotes an unknown/masked slot and centre must always be
    ``-1``.  ``available`` may be shared across queries (``B,N``) or supplied
    per query (``B,Q,N``).  ``value`` is an uncalibrated validity logit suitable
    for ``binary_cross_entropy_with_logits``.
    """

    def __init__(
        self,
        d: int = 160,
        window: int = 5,
        layers: int = 3,
        heads: int = 5,
    ) -> None:
        super().__init__()
        if d < 20 or d % heads:
            raise ValueError("d must be at least 20 and divisible by heads")
        if window < 3 or window % 2 == 0:
            raise ValueError("window must be an odd integer of at least three")
        if layers < 1 or heads < 1:
            raise ValueError("layers and heads must be positive")

        self.d = int(d)
        self.window = int(window)
        self.layers = int(layers)
        self.heads = int(heads)
        self.slots = self.window * self.window
        self.center = self.slots // 2

        self.tile_encoder = _TileEncoder(self.d)
        self.mask_token = nn.Parameter(torch.empty(self.d))
        self.relative_embedding = nn.Embedding(self.slots, self.d)
        self.occupied_embedding = nn.Embedding(2, self.d)
        encoder_layer = nn.TransformerEncoderLayer(
            self.d,
            self.heads,
            dim_feedforward=4 * self.d,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, self.layers, norm=nn.LayerNorm(self.d)
        )
        self.query_head = nn.Sequential(
            nn.LayerNorm(self.d),
            nn.Linear(self.d, self.d),
            nn.GELU(),
            nn.Linear(self.d, self.d),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.d), nn.Linear(self.d, self.d // 2), nn.GELU(), nn.Linear(self.d // 2, 1)
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.relative_embedding.weight, std=0.02)
        nn.init.normal_(self.occupied_embedding.weight, std=0.02)

    def encode_tiles(self, tiles: Tensor) -> Tensor:
        """Encode ``(B,576,3,20,20)`` into normalized pointer keys."""
        if tiles.ndim != 5 or tuple(tiles.shape[1:]) != (
            TILE_COUNT,
            3,
            TILE_SIZE,
            TILE_SIZE,
        ):
            raise ValueError(
                "tiles must have shape (B,576,3,20,20), "
                f"got {tuple(tiles.shape)}"
            )
        if not torch.is_floating_point(tiles):
            raise TypeError("tiles must be floating point")
        batch = tiles.shape[0]
        encoded = self.tile_encoder(tiles.reshape(-1, 3, TILE_SIZE, TILE_SIZE))
        return encoded.reshape(batch, TILE_COUNT, self.d)

    def score_from_embeddings(
        self,
        keys: Tensor,
        context_indices: Tensor,
        occupied: Tensor,
        available: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Score every bag tile for each relative frontier query."""
        if keys.ndim != 3 or keys.shape[-1] != self.d or not torch.is_floating_point(keys):
            raise ValueError(f"keys must be floating (B,N,{self.d}), got {tuple(keys.shape)}")
        batch, count, _ = keys.shape
        expected = (batch, context_indices.shape[1], self.slots) if context_indices.ndim == 3 else ()
        if context_indices.ndim != 3 or tuple(context_indices.shape) != expected:
            raise ValueError(
                f"context_indices must have shape (B,Q,{self.slots}), got {tuple(context_indices.shape)}"
            )
        if context_indices.dtype != torch.long:
            raise TypeError("context_indices must have dtype torch.long")
        if occupied.shape != context_indices.shape or occupied.dtype != torch.bool:
            raise ValueError("occupied must be bool with the same shape as context_indices")
        if keys.device != context_indices.device or keys.device != occupied.device:
            raise ValueError("keys, context_indices, and occupied must share a device")
        if torch.any(context_indices < -1) or torch.any(context_indices >= count):
            raise ValueError("context_indices contains an out-of-range tile index")
        if torch.any(context_indices[..., self.center] != -1):
            raise ValueError(f"context centre {self.center} must be -1")

        queries = context_indices.shape[1]
        safe = context_indices.clamp(min=0)
        batch_index = torch.arange(batch, device=keys.device)[:, None, None]
        gathered = keys[batch_index, safe]
        visible = occupied & context_indices.ge(0)
        masks = self.mask_token.view(1, 1, 1, self.d)
        tokens = torch.where(visible.unsqueeze(-1), gathered, masks)
        relative = self.relative_embedding.weight.view(1, 1, self.slots, self.d)
        tokens = tokens + relative + self.occupied_embedding(occupied.long())
        encoded = self.context_encoder(tokens.reshape(batch * queries, self.slots, self.d))
        centre = encoded[:, self.center].reshape(batch, queries, self.d)
        query = F.normalize(self.query_head(centre), dim=-1)
        normalized_keys = F.normalize(keys, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = torch.einsum("bqd,bnd->bqn", query, normalized_keys) * scale

        if available is not None:
            if available.dtype != torch.bool or available.device != keys.device:
                raise ValueError("available must be bool on the same device as keys")
            if available.shape == (batch, count):
                allowed = available[:, None, :].expand(batch, queries, count)
            elif available.shape == (batch, queries, count):
                allowed = available
            else:
                raise ValueError(
                    "available must have shape (B,N) or (B,Q,N), "
                    f"got {tuple(available.shape)}"
                )
            logits = logits.masked_fill(~allowed, -torch.inf)

        return {
            "keys": normalized_keys,
            "logits": logits,
            "value": self.value_head(centre).squeeze(-1),
            "query": query,
        }

    def forward(
        self,
        tiles: Tensor,
        context_indices: Tensor,
        occupied: Tensor,
        available: Tensor | None = None,
    ) -> dict[str, Tensor]:
        keys = self.encode_tiles(tiles)
        return self.score_from_embeddings(keys, context_indices, occupied, available)


def run_smoke() -> None:
    """Check shapes, gradients, masking, and exact bag permutation equivariance."""
    torch.manual_seed(17)
    model = FrontierPointer()
    model.eval()
    tiles = torch.rand(1, TILE_COUNT, 3, TILE_SIZE, TILE_SIZE, requires_grad=True)
    context = torch.full((1, 2, 25), -1, dtype=torch.long)
    positions = torch.tensor([1, 2, 3, 6, 7, 8, 11, 13, 16, 17, 18, 21, 22, 23])
    context[0, 0, positions] = torch.arange(positions.numel())
    context[0, 1, positions] = torch.arange(40, 40 + positions.numel())
    occupied = context.ge(0)
    available = torch.ones(1, 2, TILE_COUNT, dtype=torch.bool)
    available.scatter_(2, context.clamp_min(0), False)

    output = model(tiles, context, occupied, available)
    assert output["keys"].shape == (1, TILE_COUNT, 160)
    assert output["logits"].shape == (1, 2, TILE_COUNT)
    assert output["query"].shape == (1, 2, 160) and output["value"].shape == (1, 2)
    assert torch.isneginf(output["logits"][~available]).all()
    assert torch.isfinite(output["logits"][available]).all()
    assert torch.allclose(output["keys"].norm(dim=-1), torch.ones(1, TILE_COUNT), atol=1e-5)
    loss = output["logits"][available].square().mean() + output["value"].square().mean()
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert tiles.grad is not None and torch.isfinite(tiles.grad).all()
    assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)
    assert all(grad is None or torch.isfinite(grad).all() for grad in grads)

    with torch.no_grad():
        permutation = torch.randperm(TILE_COUNT)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(TILE_COUNT)
        remapped = torch.where(context.ge(0), inverse[context.clamp_min(0)], context)
        permuted = model(
            tiles.detach()[:, permutation], remapped, occupied, available[:, :, permutation]
        )
        assert torch.allclose(permuted["keys"], output["keys"].detach()[:, permutation], atol=2e-5)
        assert torch.allclose(permuted["query"], output["query"].detach(), atol=2e-5)
        assert torch.allclose(permuted["value"], output["value"].detach(), atol=2e-5)
        assert torch.allclose(
            permuted["logits"], output["logits"].detach()[:, :, permutation], atol=2e-5
        )
    print("frontier_pointer smoke: shapes=ok gradients=ok permutation_equivariance=ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run the self-contained contract test")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("nothing to do (pass --smoke)")
    run_smoke()


if __name__ == "__main__":
    main()
