"""Multi-task seam ranker with clean-structure reconstruction supervision.

The ranking path is a drop-in replacement for ``CandidateSeamRanker``.  During
training, ``forward_with_structure`` additionally predicts a canonical clean
luminance/gradient field for the dirty directed pair.  The auxiliary target is
available from synthetic clean images but is not needed at inference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from candidate_rank import (
    _ResidualBlock,
    _exposure_normalize,
    _groups,
    _orient_to_canonical,
    canonical_pair_layout,
)
from config import FS


def _luminance(rgb: Tensor) -> Tensor:
    weights = rgb.new_tensor((0.299, 0.587, 0.114)).reshape(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def _sobel(value: Tensor) -> tuple[Tensor, Tensor]:
    """Return scale-stable horizontal/vertical derivatives."""
    kernel_x = value.new_tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    ).reshape(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    return (
        F.conv2d(value, kernel_x, padding=1),
        F.conv2d(value, kernel_y, padding=1),
    )


def _pair_structure_from_halves(left: Tensor, right: Tensor) -> Tensor:
    """Build luma/dx/dy without differentiating across the artificial seam."""
    left_luma = _luminance(left)
    right_luma = _luminance(right)
    left_dx, left_dy = _sobel(left_luma)
    right_dx, right_dy = _sobel(right_luma)
    return torch.cat(
        (
            torch.cat((left_luma, right_luma), dim=-1),
            torch.cat((left_dx, right_dx), dim=-1),
            torch.cat((left_dy, right_dy), dim=-1),
        ),
        dim=1,
    )


def dirty_structure_channels(layout: Tensor, tile_size: int = FS) -> Tensor:
    """Extract robust structural channels from a canonical six-channel layout."""
    if layout.ndim != 4 or layout.shape[1] != 6:
        raise ValueError("layout must have shape (pairs,6,H,2H)")
    if tuple(layout.shape[-2:]) != (tile_size, tile_size * 2):
        raise ValueError(f"unexpected layout spatial shape {tuple(layout.shape[-2:])}")
    normalized = layout[:, 3:]
    return _pair_structure_from_halves(
        normalized[..., :tile_size],
        normalized[..., tile_size:],
    )


def clean_structure_target(
    source: Tensor,
    target: Tensor,
    directions: Tensor,
) -> Tensor:
    """Return the canonical clean luma/dx/dy auxiliary target."""
    if source.shape != target.shape or source.ndim != 4 or source.shape[1] != 3:
        raise ValueError("source and target must be matching (pairs,3,H,W) tensors")
    if directions.ndim != 1 or directions.shape[0] != source.shape[0]:
        raise ValueError("directions must contain one value per pair")
    left = _orient_to_canonical(source, directions.long())
    right = _orient_to_canonical(target, directions.long())
    # Clean RGB is already in [0,1].  Keep true luminance but normalize each
    # tile independently for derivatives, matching the invariance required by
    # independently corrupted observations.
    luma = torch.cat((_luminance(left), _luminance(right)), dim=-1)
    normalized_left = _exposure_normalize(left)
    normalized_right = _exposure_normalize(right)
    normalized_structure = _pair_structure_from_halves(
        normalized_left, normalized_right
    )
    return torch.cat((luma, normalized_structure[:, 1:]), dim=1)


class StructuralSeamRanker(nn.Module):
    """Rank a directed pair and reconstruct its clean structural field."""

    def __init__(
        self,
        *,
        tile_size: int = FS,
        width: int = 32,
        dropout: float = 0.10,
        seam_band: int = 6,
    ) -> None:
        super().__init__()
        if tile_size < 4 or width < 4 or seam_band < 2:
            raise ValueError("invalid tile_size, width, or seam_band")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.tile_size = int(tile_size)
        self.width = int(width)
        self.dropout = float(dropout)
        self.seam_band = int(seam_band)
        hidden = self.width * 2

        # Raw RGB + normalized RGB + normalized luma/dx/dy.
        self.stem = nn.Sequential(
            nn.Conv2d(9, self.width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(self.width), self.width),
            nn.GELU(),
        )
        self.block1 = _ResidualBlock(self.width)
        self.down = nn.Sequential(
            nn.Conv2d(self.width, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
        )
        self.block2 = _ResidualBlock(hidden)
        self.rank_head = nn.Sequential(
            nn.LayerNorm(hidden * 6),
            nn.Linear(hidden * 6, hidden * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden * 2, 1),
        )
        self.structure_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, self.width, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(self.width), self.width),
            nn.GELU(),
            _ResidualBlock(self.width),
            nn.Conv2d(self.width, 3, 3, padding=1),
        )

    @property
    def model_kwargs(self) -> dict[str, int | float]:
        return {
            "tile_size": self.tile_size,
            "width": self.width,
            "dropout": self.dropout,
            "seam_band": self.seam_band,
        }

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

    def _encode(
        self,
        source: Tensor,
        target: Tensor,
        directions: Tensor,
    ) -> Tensor:
        expected = (3, self.tile_size, self.tile_size)
        if source.ndim != 4 or tuple(source.shape[1:]) != expected:
            raise ValueError(f"source must have shape (pairs,{expected})")
        if target.shape != source.shape:
            raise ValueError("target must match source")
        if directions.ndim != 1 or directions.shape[0] != source.shape[0]:
            raise ValueError("directions must contain one value per pair")
        layout = canonical_pair_layout(source, target, directions)
        structural = dirty_structure_channels(layout, self.tile_size)
        features = self.block1(self.stem(torch.cat((layout, structural), dim=1)))
        return self.block2(self.down(features))

    def _rank(self, features: Tensor) -> Tensor:
        width = features.shape[-1]
        band = min(max(2, self.seam_band), width)
        start = (width - band) // 2
        seam = features[..., start : start + band]
        representation = torch.cat(
            (self._summary(features), self._summary(seam)), dim=-1
        )
        return self.rank_head(representation).squeeze(-1)

    def forward(self, source: Tensor, target: Tensor, directions: Tensor) -> Tensor:
        if source.shape[0] == 0:
            return source.new_empty((0,))
        return self._rank(self._encode(source, target, directions))

    def forward_with_structure(
        self,
        source: Tensor,
        target: Tensor,
        directions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        features = self._encode(source, target, directions)
        prediction = self.structure_decoder(features)
        expected = (self.tile_size, self.tile_size * 2)
        if tuple(prediction.shape[-2:]) != expected:
            prediction = F.interpolate(
                prediction,
                size=expected,
                mode="bilinear",
                align_corners=False,
            )
        return self._rank(features), prediction


def structural_reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Robust clean-luma and derivative loss with extra seam emphasis."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target structure must match")
    width = prediction.shape[-1]
    seam = width // 2
    full = F.smooth_l1_loss(prediction, target)
    seam_loss = F.smooth_l1_loss(
        prediction[..., seam - 4 : seam + 4],
        target[..., seam - 4 : seam + 4],
    )
    return full + seam_loss


def smoke(device: torch.device | str = "cpu") -> dict[str, object]:
    device = torch.device(device)
    torch.manual_seed(41)
    count = 8
    source = torch.rand(count, 3, FS, FS, device=device)
    target = torch.rand_like(source)
    directions = torch.arange(count, device=device) % 4
    model = StructuralSeamRanker(width=8, dropout=0.0).to(device)
    score, prediction = model.forward_with_structure(source, target, directions)
    clean_target = clean_structure_target(source, target, directions)
    loss = score.square().mean() + structural_reconstruction_loss(
        prediction, clean_target
    )
    loss.backward()
    if score.shape != (count,) or prediction.shape != (count, 3, FS, FS * 2):
        raise AssertionError("unexpected structural seam output shapes")
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise AssertionError("structural seam smoke produced non-finite gradients")
    return {
        "score_shape": tuple(score.shape),
        "structure_shape": tuple(prediction.shape),
        "loss": float(loss.detach()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


if __name__ == "__main__":
    print(f"structural seam smoke: {smoke('cuda' if torch.cuda.is_available() else 'cpu')}")
