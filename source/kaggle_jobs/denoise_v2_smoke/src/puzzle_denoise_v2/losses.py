"""Pixel-faithful losses aligned with RGB SSIM and tile-boundary quality."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def charbonnier(error: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(error.square() + epsilon * epsilon)


def skimage_like_ssim(prediction: torch.Tensor, target: torch.Tensor, window: int = 7) -> torch.Tensor:
    """Differentiable uniform-window SSIM matching skimage's key defaults."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have identical BCHW shapes")
    sample_correction = (window * window) / (window * window - 1.0)
    mean_x = F.avg_pool2d(prediction, window, stride=1)
    mean_y = F.avg_pool2d(target, window, stride=1)
    covariance_x = sample_correction * (F.avg_pool2d(prediction.square(), window, stride=1) - mean_x.square())
    covariance_y = sample_correction * (F.avg_pool2d(target.square(), window, stride=1) - mean_y.square())
    covariance_xy = sample_correction * (F.avg_pool2d(prediction * target, window, stride=1) - mean_x * mean_y)
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance_xy + c2)
    denominator = (mean_x.square() + mean_y.square() + c1) * (covariance_x + covariance_y + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean()


def boundary_mask(height: int, width: int, band: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
    mask[:, :, :band] = 1
    mask[:, :, -band:] = 1
    mask[:, :, :, :band] = 1
    mask[:, :, :, -band:] = 1
    return mask


def gradient_charbonnier(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    true_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    true_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    return charbonnier(pred_x - true_x).mean() + charbonnier(pred_y - true_y).mean()


@dataclass(frozen=True)
class LossWeights:
    ssim: float = 0.10
    gradient: float = 0.05
    color: float = 0.02
    degradation: float = 0.02
    boundary_extra: float = 0.50


class RestorationLoss(nn.Module):
    def __init__(self, weights: LossWeights = LossWeights(), boundary_band: int = 3) -> None:
        super().__init__()
        self.weights = weights
        self.boundary_band = boundary_band

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        degradation_prediction: torch.Tensor | None = None,
        degradation_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = boundary_mask(
            prediction.shape[-2],
            prediction.shape[-1],
            self.boundary_band,
            prediction.device,
            prediction.dtype,
        )
        pixel_weights = 1.0 + self.weights.boundary_extra * mask
        pixel_map = charbonnier(prediction - target)
        pixel = (pixel_map * pixel_weights).sum() / (pixel_weights.sum() * prediction.shape[0] * prediction.shape[1])
        ssim = skimage_like_ssim(prediction, target)
        gradient = gradient_charbonnier(prediction, target)

        pred_mean = prediction.mean(dim=(2, 3))
        true_mean = target.mean(dim=(2, 3))
        pred_std = prediction.std(dim=(2, 3), correction=0)
        true_std = target.std(dim=(2, 3), correction=0)
        color = F.smooth_l1_loss(pred_mean, true_mean) + F.smooth_l1_loss(pred_std, true_std)

        degradation = prediction.new_zeros(())
        if degradation_prediction is not None and degradation_target is not None:
            degradation = F.smooth_l1_loss(degradation_prediction, degradation_target)

        total = (
            pixel
            + self.weights.ssim * (1.0 - ssim)
            + self.weights.gradient * gradient
            + self.weights.color * color
            + self.weights.degradation * degradation
        )
        components = {
            "total": total.detach(),
            "pixel": pixel.detach(),
            "ssim": ssim.detach(),
            "gradient": gradient.detach(),
            "color": color.detach(),
            "degradation": degradation.detach(),
        }
        return total, components
