"""Batched synthetic corruption for exact clean-tile supervision.

The competition corruption is applied after cropping each 20x20 tile. This
module preserves that boundary condition and exposes operation-order variants
so the simulator can be calibrated rather than silently assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from kornia.enhance import jpeg_codec_differentiable


@dataclass(frozen=True)
class DegradationParameters:
    brightness: torch.Tensor
    contrast: torch.Tensor
    noise_sigma: torch.Tensor
    blur_sigma: torch.Tensor
    jpeg_quality: torch.Tensor
    variant: torch.Tensor

    def normalized(self) -> torch.Tensor:
        """Return stable auxiliary-regression targets."""
        return torch.stack(
            [
                self.brightness / 30.0,
                (self.contrast - 1.0) / 0.30,
                (self.noise_sigma - 47.5) / 7.5,
                (self.blur_sigma - 0.85) / 0.25,
                (self.jpeg_quality - 42.5) / 7.5,
            ],
            dim=1,
        )


def _quantize_uint8(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0


def _gaussian_kernel3(sigma: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.tensor([-1.0, 0.0, 1.0], device=sigma.device, dtype=dtype)
    kernel_1d = torch.exp(-(coords[None, :] ** 2) / (2.0 * sigma[:, None] ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum(dim=1, keepdim=True)
    return kernel_1d[:, :, None] * kernel_1d[:, None, :]


def _blur3_per_sample(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    kernel = _gaussian_kernel3(sigma, x.dtype)
    weight = kernel[:, None, :, :].repeat_interleave(channels, dim=0)
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    merged = padded.reshape(1, batch * channels, height + 2, width + 2)
    blurred = F.conv2d(merged, weight, groups=batch * channels)
    return blurred.reshape(batch, channels, height, width)


class SyntheticTileDegrader:
    """Generate exact synthetic pairs with configurable operation-order mixture.

    Variant 0 is the documented primary recipe:
    contrast -> additive brightness -> Gaussian noise -> 3x3 blur -> JPEG.
    Variant 1 swaps blur/noise, and variant 2 swaps brightness/contrast. Those
    variants are disabled by default until real-domain calibration justifies a
    mixture.
    """

    def __init__(
        self,
        brightness=(-30.0, 30.0),
        contrast=(0.70, 1.30),
        noise_sigma=(40.0, 55.0),
        blur_sigma=(0.75, 0.95),
        jpeg_quality=(35, 50),
        variant_weights=(1.0, 0.0, 0.0),
        quantize_stages: bool = True,
    ) -> None:
        self.brightness_range = brightness
        self.contrast_range = contrast
        self.noise_range = noise_sigma
        self.blur_range = blur_sigma
        self.jpeg_range = jpeg_quality
        weights = torch.as_tensor(variant_weights, dtype=torch.float32)
        if weights.ndim != 1 or len(weights) != 3 or float(weights.sum()) <= 0:
            raise ValueError("variant_weights must contain three non-negative values")
        self.variant_weights = weights / weights.sum()
        self.quantize_stages = quantize_stages

    @staticmethod
    def _uniform(
        batch: int,
        bounds: tuple[float, float],
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        low, high = bounds
        return torch.rand(batch, device=device, generator=generator) * (high - low) + low

    def sample_parameters(
        self,
        batch: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> DegradationParameters:
        qualities = torch.randint(
            int(self.jpeg_range[0]),
            int(self.jpeg_range[1]) + 1,
            (batch,),
            device=device,
            generator=generator,
        ).float()
        weights = self.variant_weights.to(device)
        variants = torch.multinomial(weights, batch, replacement=True, generator=generator)
        return DegradationParameters(
            brightness=self._uniform(batch, self.brightness_range, device, generator),
            contrast=self._uniform(batch, self.contrast_range, device, generator),
            noise_sigma=self._uniform(batch, self.noise_range, device, generator),
            blur_sigma=self._uniform(batch, self.blur_range, device, generator),
            jpeg_quality=qualities,
            variant=variants,
        )

    def _photometric(
        self,
        clean: torch.Tensor,
        params: DegradationParameters,
        brightness_first: bool,
    ) -> torch.Tensor:
        brightness = params.brightness[:, None, None, None] / 255.0
        contrast = params.contrast[:, None, None, None]
        if brightness_first:
            image = clean + brightness
            luminance = 0.299 * image[:, :1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
            center = luminance.mean(dim=(2, 3), keepdim=True)
            image = center + contrast * (image - center)
        else:
            luminance = 0.299 * clean[:, :1] + 0.587 * clean[:, 1:2] + 0.114 * clean[:, 2:3]
            center = luminance.mean(dim=(2, 3), keepdim=True)
            image = center + contrast * (clean - center)
            image = image + brightness
        return _quantize_uint8(image) if self.quantize_stages else image.clamp(0.0, 1.0)

    def _noise(
        self,
        image: torch.Tensor,
        params: DegradationParameters,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        noise = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
        image = image + noise * (params.noise_sigma[:, None, None, None] / 255.0)
        return _quantize_uint8(image) if self.quantize_stages else image.clamp(0.0, 1.0)

    def _blur(self, image: torch.Tensor, params: DegradationParameters) -> torch.Tensor:
        image = _blur3_per_sample(image, params.blur_sigma.to(image.dtype))
        return _quantize_uint8(image) if self.quantize_stages else image.clamp(0.0, 1.0)

    def __call__(
        self,
        clean: torch.Tensor,
        generator: torch.Generator | None = None,
        params: DegradationParameters | None = None,
    ) -> tuple[torch.Tensor, DegradationParameters]:
        if clean.ndim != 4 or clean.shape[1:] != (3, 20, 20):
            raise ValueError(f"expected Bx3x20x20, got {tuple(clean.shape)}")
        clean = clean.float().clamp(0.0, 1.0)
        params = params or self.sample_parameters(len(clean), clean.device, generator)
        output = torch.empty_like(clean)

        for variant in range(3):
            indices = torch.nonzero(params.variant == variant, as_tuple=False).flatten()
            if len(indices) == 0:
                continue
            sub = DegradationParameters(
                brightness=params.brightness[indices],
                contrast=params.contrast[indices],
                noise_sigma=params.noise_sigma[indices],
                blur_sigma=params.blur_sigma[indices],
                jpeg_quality=params.jpeg_quality[indices],
                variant=params.variant[indices],
            )
            image = self._photometric(clean[indices], sub, brightness_first=(variant == 2))
            if variant == 1:
                image = self._blur(image, sub)
                image = self._noise(image, sub, generator)
            else:
                image = self._noise(image, sub, generator)
                image = self._blur(image, sub)
            output[indices] = image

        output = jpeg_codec_differentiable(output, params.jpeg_quality.to(output.dtype))
        output = _quantize_uint8(output) if self.quantize_stages else output.clamp(0.0, 1.0)
        return output, params
