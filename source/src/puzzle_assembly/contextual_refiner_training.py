"""Training losses for the bounded contextual post-assembly refiner.

Training targets are allowed here, but the deployable feature/model module is
kept target-blind in :mod:`puzzle_assembly.contextual_refiner`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from puzzle_denoise_v2.losses import charbonnier, skimage_like_ssim


def _image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        image[:, :, :, 1:] - image[:, :, :, :-1],
        image[:, :, 1:, :] - image[:, :, :-1, :],
    )


def _texture_masks(
    target: torch.Tensor,
    seam_mask: torch.Tensor,
    *,
    quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_x, target_y = _image_gradients(target)
    magnitude_x = target_x.square().mean(dim=1, keepdim=True).sqrt()
    magnitude_y = target_y.square().mean(dim=1, keepdim=True).sqrt()
    thresholds_x = torch.quantile(
        magnitude_x.detach().flatten(1), quantile, dim=1, keepdim=True
    )[:, :, None, None]
    thresholds_y = torch.quantile(
        magnitude_y.detach().flatten(1), quantile, dim=1, keepdim=True
    )[:, :, None, None]
    nonseam_x = 1.0 - torch.maximum(seam_mask[:, :, :, 1:], seam_mask[:, :, :, :-1])
    nonseam_y = 1.0 - torch.maximum(seam_mask[:, :, 1:, :], seam_mask[:, :, :-1, :])
    return (
        (magnitude_x >= thresholds_x).to(target.dtype) * nonseam_x,
        (magnitude_y >= thresholds_y).to(target.dtype) * nonseam_y,
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and values.shape[1] != 1:
        mask = mask.expand(-1, values.shape[1], -1, -1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


@dataclass(frozen=True)
class ContextualLossWeights:
    ssim: float = 0.15
    seam_extra: float = 0.50
    gradient: float = 0.10
    texture: float = 0.10
    residual: float = 0.02

    def validate(self) -> None:
        values = (
            self.ssim,
            self.seam_extra,
            self.gradient,
            self.texture,
            self.residual,
        )
        if min(values) < 0:
            raise ValueError("loss weights must be non-negative")


class ContextualRefinerLoss(nn.Module):
    """SSIM-aligned loss with explicit seam and texture protection."""

    def __init__(
        self,
        weights: ContextualLossWeights = ContextualLossWeights(),
        *,
        texture_quantile: float = 0.75,
    ) -> None:
        super().__init__()
        weights.validate()
        if not 0.5 <= texture_quantile < 1.0:
            raise ValueError("texture_quantile must be in [0.5, 1)")
        self.weights = weights
        self.texture_quantile = texture_quantile

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        identity_input: torch.Tensor,
        seam_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape or prediction.shape != identity_input.shape:
            raise ValueError("prediction, target and identity_input must have equal shapes")
        if prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError("images must be BCHW RGB")
        if seam_mask.shape != (prediction.shape[0], 1, *prediction.shape[-2:]):
            raise ValueError("seam_mask must be Bx1xHxW")
        if not all(
            torch.isfinite(value).all()
            for value in (prediction, target, identity_input, seam_mask)
        ):
            raise ValueError("loss inputs must be finite")

        error = charbonnier(prediction - target)
        pixel = error.mean()
        seam = _masked_mean(error, seam_mask)
        ssim = skimage_like_ssim(prediction, target)
        pred_x, pred_y = _image_gradients(prediction)
        true_x, true_y = _image_gradients(target)
        gradient = charbonnier(pred_x - true_x).mean() + charbonnier(
            pred_y - true_y
        ).mean()
        texture_x, texture_y = _texture_masks(
            target, seam_mask, quantile=self.texture_quantile
        )
        texture = _masked_mean(charbonnier(pred_x - true_x), texture_x) + _masked_mean(
            charbonnier(pred_y - true_y), texture_y
        )
        residual = (prediction - identity_input).abs().mean()
        total = (
            pixel
            + self.weights.ssim * (1.0 - ssim)
            + self.weights.seam_extra * seam
            + self.weights.gradient * gradient
            + self.weights.texture * texture
            + self.weights.residual * residual
        )
        return total, {
            "total": total.detach(),
            "pixel": pixel.detach(),
            "ssim": ssim.detach(),
            "seam": seam.detach(),
            "gradient": gradient.detach(),
            "texture": texture.detach(),
            "residual": residual.detach(),
        }


__all__ = ["ContextualLossWeights", "ContextualRefinerLoss"]
