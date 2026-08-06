"""Directional Siamese CNN for all-pairs 24x24 neighbour retrieval."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from config import FS, NFRAG
from eval_paired_alignment import TileEncoder


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class _EdgeHead(nn.Module):
    def __init__(self, channels: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(4 * channels),
            nn.Linear(4 * channels, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.normalize(self.net(value), dim=-1)


class DirectionalSiamese(nn.Module):
    """One shared tile CNN with four side-specific descriptor heads."""

    def __init__(self, channels: int = 128, embed_dim: int = 96) -> None:
        super().__init__()
        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        # Reuse the exact spatial backbone whose pooled form learned robust
        # dirty-to-clean tile identity.  The old pooling head is intentionally
        # omitted because directional boundary locations matter here.
        self.backbone = TileEncoder(channels).features
        self.heads = nn.ModuleList(
            [_EdgeHead(channels, embed_dim) for _ in range(4)]
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    @staticmethod
    def _normalize(tiles: Tensor) -> Tensor:
        mean = tiles.mean(dim=(-3, -2, -1), keepdim=True)
        rms = (tiles - mean).square().mean(dim=(-3, -2, -1), keepdim=True)
        return ((tiles - mean) / rms.add(1.0e-5).sqrt()).clamp(-5.0, 5.0)

    @staticmethod
    def _stats(value: Tensor) -> tuple[Tensor, Tensor]:
        flat = value.flatten(start_dim=2)
        return flat.mean(dim=-1), flat.var(dim=-1, unbiased=False).add(1.0e-6).sqrt()

    def embeddings(self, tiles: Tensor) -> Tensor:
        if tiles.ndim != 5 or tuple(tiles.shape[2:]) != (3, FS, FS):
            raise ValueError(f"tiles must be (B,N,3,{FS},{FS}), got {tuple(tiles.shape)}")
        batch, count = tiles.shape[:2]
        flat = tiles.reshape(batch * count, 3, FS, FS)
        feature = self.backbone(torch.cat((flat, self._normalize(flat)), dim=1))
        global_mean, global_std = self._stats(feature)
        sides = (
            feature[:, :, 0, :],
            feature[:, :, -1, :],
            feature[:, :, :, 0],
            feature[:, :, :, -1],
        )
        output = []
        for head, side in zip(self.heads, sides):
            side_mean, side_std = self._stats(side.unsqueeze(-2))
            descriptor = torch.cat(
                (global_mean, global_std, side_mean, side_std), dim=-1
            )
            output.append(head(descriptor).reshape(batch, count, self.embed_dim))
        return torch.stack(output, dim=1)

    def scores(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim != 4 or embeddings.shape[1] != 4:
            raise ValueError("embeddings must be (B,4,N,D)")
        opposite = (DOWN, UP, RIGHT, LEFT)
        rows = [
            embeddings[:, direction] @ embeddings[:, opposite[direction]].transpose(1, 2)
            for direction in range(4)
        ]
        scores = torch.stack(rows, dim=1) * self.logit_scale.exp().clamp(max=100.0)
        diagonal = torch.eye(
            scores.shape[-1], device=scores.device, dtype=torch.bool
        ).reshape(1, 1, scores.shape[-1], scores.shape[-1])
        return scores.masked_fill(diagonal, -1.0e4)

    def forward(self, tiles: Tensor) -> Tensor:
        return self.scores(self.embeddings(tiles))


def load_paired_backbone(model: DirectionalSiamese, checkpoint: dict) -> int:
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    prefix = "dirty_encoder.features."
    copied = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    model.backbone.load_state_dict(copied, strict=True)
    return len(copied)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    net = DirectionalSiamese(32, 24)
    value = net(torch.rand(1, NFRAG, 3, FS, FS))
    print(tuple(value.shape), count_parameters(net))
