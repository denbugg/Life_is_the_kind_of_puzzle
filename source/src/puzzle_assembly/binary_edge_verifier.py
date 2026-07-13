"""Binary pixel-and-rank verifier for sparse puzzle adjacency candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import TILE


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.norm = nn.GroupNorm(8, channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.pointwise(F.silu(self.norm(self.depthwise(values))))


class BinaryEdgeVerifierNet(nn.Module):
    """Verify a proposed right/down edge from raw+denoised seam strips.

    Each seam patch is canonicalized to left|right orientation before entering
    the network.  The tabular branch contains only input-derived candidate
    ranks, robust costs, direction, and origin-consensus bits.
    """

    def __init__(
        self,
        *,
        tabular_dim: int,
        channels: int = 64,
        side_band: int = 8,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if tabular_dim <= 0:
            raise ValueError("tabular_dim must be positive")
        if channels <= 0 or channels % 8:
            raise ValueError("channels must be positive and divisible by 8")
        if not 2 <= side_band <= 10:
            raise ValueError("side_band must be in [2, 10]")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.tabular_dim = int(tabular_dim)
        self.channels = int(channels)
        self.side_band = int(side_band)
        self.dropout = float(dropout)
        # Per image view: RGB, half-wise normalized RGB, dx, dy = 12.
        # Raw and denoised views are concatenated, hence 24 channels.
        self.encoder = nn.Sequential(
            nn.Conv2d(24, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            ResidualConvBlock(channels),
            ResidualConvBlock(channels),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.SiLU(),
            ResidualConvBlock(channels * 2),
            ResidualConvBlock(channels * 2),
            nn.Conv2d(channels * 2, channels * 3, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels * 3),
            nn.SiLU(),
            ResidualConvBlock(channels * 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.tabular = nn.Sequential(
            nn.LayerNorm(tabular_dim),
            nn.Linear(tabular_dim, channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 4, channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, 1),
        )

    @staticmethod
    def _view_features(patches: torch.Tensor) -> torch.Tensor:
        values = patches.float()
        if values.detach().amax() > 1.5:
            values = values / 255.0
        left, right = values.chunk(2, dim=3)
        normalized_parts = []
        for half in (left, right):
            mean = half.mean(dim=(2, 3), keepdim=True)
            std = half.std(dim=(2, 3), keepdim=True).clamp_min(1.0 / 255.0)
            normalized_parts.append(((half - mean) / std).clamp(-4.0, 4.0) / 4.0)
        normalized = torch.cat(normalized_parts, dim=3)
        dx = F.pad(values[:, :, :, 1:] - values[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(values[:, :, 1:, :] - values[:, :, :-1, :], (0, 0, 0, 1))
        return torch.cat([values, normalized, dx, dy], dim=1)

    def forward(
        self,
        raw_patches: torch.Tensor,
        denoised_patches: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        expected = (3, TILE, 2 * self.side_band)
        if raw_patches.ndim != 4 or tuple(raw_patches.shape[1:]) != expected:
            raise ValueError(f"raw_patches must have shape N{expected}")
        if denoised_patches.shape != raw_patches.shape:
            raise ValueError("raw and denoised patch shapes must match")
        if tabular.ndim != 2 or tabular.shape != (
            raw_patches.shape[0],
            self.tabular_dim,
        ):
            raise ValueError("tabular feature shape mismatch")
        pixels = torch.cat(
            [self._view_features(raw_patches), self._view_features(denoised_patches)],
            dim=1,
        )
        encoded = self.encoder(pixels)
        tabular_encoded = self.tabular(tabular.float())
        return self.head(torch.cat([encoded, tabular_encoded], dim=1)).squeeze(1)

    def config(self) -> dict[str, Any]:
        return {
            "tabular_dim": self.tabular_dim,
            "channels": self.channels,
            "side_band": self.side_band,
            "dropout": self.dropout,
        }


def save_binary_edge_verifier(
    path: str | Path,
    model: BinaryEdgeVerifierNet,
    *,
    feature_names: list[str],
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "puzzle_binary_edge_verifier",
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "feature_names": list(feature_names),
            "metadata": metadata,
        },
        path,
    )


def load_binary_edge_verifier(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[BinaryEdgeVerifierNet, list[str], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "puzzle_binary_edge_verifier"
    ):
        raise ValueError("unsupported binary edge verifier checkpoint")
    model = BinaryEdgeVerifierNet(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, list(payload["feature_names"]), dict(payload.get("metadata", {}))


__all__ = [
    "BinaryEdgeVerifierNet",
    "load_binary_edge_verifier",
    "save_binary_edge_verifier",
]
