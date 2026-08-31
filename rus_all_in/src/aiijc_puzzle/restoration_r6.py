"""Broad post-layout restoration models and corruption-matched training tools.

The historical S1 submission used a small residual U-Net (R5) trained on only
two scenes, then applied coloured NLM.  This module keeps that exact model as a
control and adds a tile-aware dual-input NAF-style restorer.  Both operate only
after a layout has been chosen; neither can access clean targets at inference.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TILE_SIZE = 20
_GRAY = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)


def split_square_tiles(image: np.ndarray, tile_size: int = TILE_SIZE) -> np.ndarray:
    """Split a square RGB canvas into row-major tiles of ``tile_size``."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] != image.shape[1]:
        raise ValueError(f"expected a square RGB image, got {image.shape}")
    if image.shape[0] % tile_size:
        raise ValueError(f"image side {image.shape[0]} is not divisible by {tile_size}")
    grid = image.shape[0] // tile_size
    return (
        image.reshape(grid, tile_size, grid, tile_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid * grid, tile_size, tile_size, 3)
    )


def assemble_square_tiles(tiles: np.ndarray) -> np.ndarray:
    """Assemble a square row-major tile array into one RGB canvas."""
    tiles = np.asarray(tiles)
    grid = round(len(tiles) ** 0.5)
    if tiles.ndim != 4 or tiles.shape[1:] != (TILE_SIZE, TILE_SIZE, 3) or grid * grid != len(tiles):
        raise ValueError(f"expected a square N x 20 x 20 x 3 tile array, got {tiles.shape}")
    return (
        tiles.reshape(grid, grid, TILE_SIZE, TILE_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid * TILE_SIZE, grid * TILE_SIZE, 3)
    )


def _blur3(tiles: np.ndarray) -> np.ndarray:
    x = np.asarray(tiles, dtype=np.float32)
    padded = np.pad(x, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="reflect")
    x = 0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="reflect")
    return 0.25 * padded[:, :, :-2] + 0.5 * padded[:, :, 1:-1] + 0.25 * padded[:, :, 2:]


def distort_tiles(
    tiles: np.ndarray,
    rng: np.random.Generator,
    *,
    brightness: float = 30.0,
    contrast: tuple[float, float] = (0.70, 1.30),
    noise_sigma: tuple[float, float] = (40.0, 55.0),
    jpeg_quality: tuple[int, int] = (35, 50),
) -> np.ndarray:
    """Apply the reverse-engineered challenge corruption independently per tile."""
    tiles = np.asarray(tiles)
    if tiles.ndim != 4 or tiles.shape[-1] != 3:
        raise ValueError(f"expected N x H x W x 3 tiles, got {tiles.shape}")
    count = len(tiles)
    x = tiles.astype(np.float32)
    scale = rng.uniform(*contrast, size=(count, 1, 1, 1)).astype(np.float32)
    offset = rng.uniform(-brightness, brightness, size=(count, 1, 1, 1)).astype(np.float32)
    pivot = (x * _GRAY).sum(axis=-1, keepdims=True).mean(axis=(1, 2), keepdims=True)
    x = scale * (x - pivot) + pivot + offset
    sigma = rng.uniform(*noise_sigma, size=(count, 1, 1, 1)).astype(np.float32)
    x += rng.standard_normal(x.shape, dtype=np.float32) * sigma
    x = np.clip(_blur3(np.clip(x, 0, 255)), 0, 255).astype(np.uint8)

    qualities = rng.integers(jpeg_quality[0], jpeg_quality[1] + 1, size=count)
    output = np.empty_like(x)
    for index, quality in enumerate(qualities):
        ok, encoded = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(x[index, ..., ::-1]),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            output[index] = x[index]
            continue
        output[index] = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1]
    return output


def distort_canvas(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Corrupt a correctly ordered canvas while preserving tile positions."""
    return assemble_square_tiles(distort_tiles(split_square_tiles(image), rng))


def nlm_color(image: np.ndarray, h: int = 9) -> np.ndarray:
    """Apply the frozen coloured NLM tail to an arbitrary-sized RGB canvas."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image, got {image.dtype} {image.shape}")
    if h <= 0:
        raise ValueError("h must be positive")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    filtered = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)


def _gaussian_kernel(window: int, sigma: float, x: torch.Tensor) -> torch.Tensor:
    coordinates = torch.arange(window, device=x.device, dtype=x.dtype)
    coordinates -= (window - 1) / 2
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    return kernel / kernel.sum()


def _ssim_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window: int = 11,
    sigma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = prediction.shape[1]
    one_d = _gaussian_kernel(window, sigma, prediction)
    kernel = (one_d[:, None] * one_d[None]).expand(channels, 1, window, window)
    padding = window // 2

    def smooth(value: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(value, [padding] * 4, mode="reflect"), kernel, groups=channels)

    mean_x, mean_y = smooth(prediction), smooth(target)
    var_x = smooth(prediction.square()) - mean_x.square()
    var_y = smooth(target.square()) - mean_y.square()
    covariance = smooth(prediction * target) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    contrast = (2 * covariance + c2) / (var_x + var_y + c2)
    structure = (2 * mean_x * mean_y + c1) / (mean_x.square() + mean_y.square() + c1)
    ssim = structure * contrast
    return ssim.mean(dim=(1, 2, 3)), contrast.mean(dim=(1, 2, 3))


def multi_scale_ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
) -> torch.Tensor:
    """Numerically guarded differentiable MS-SSIM loss."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have equal B x C x H x W shapes")
    scale_weights = prediction.new_tensor(tuple(weights))
    contrasts: list[torch.Tensor] = []
    for level in range(len(scale_weights)):
        similarity, contrast = _ssim_components(prediction, target)
        if level < len(scale_weights) - 1:
            contrasts.append(contrast.clamp_min(1e-6))
            prediction = F.avg_pool2d(prediction, 2)
            target = F.avg_pool2d(target, 2)
    stacked = torch.stack(contrasts)
    value = torch.prod(stacked ** scale_weights[:-1, None], dim=0)
    value = value * similarity.clamp_min(1e-6) ** scale_weights[-1]
    return 1 - value.mean()


def restoration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.84,
) -> torch.Tensor:
    """Historical MS-SSIM + L1 objective used by R5."""
    return alpha * multi_scale_ssim_loss(prediction.float(), target.float()) + (
        1 - alpha
    ) * F.l1_loss(prediction.float(), target.float())


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.second = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.first(value))
        return self.activation(self.second(hidden) + self.skip(value))


class HistoricalRestoreNet(nn.Module):
    """Exact R5 residual U-Net architecture, retained as the retraining control."""

    def __init__(self, base: int = 32, depth: int = 4) -> None:
        super().__init__()
        channels = [base * 2**index for index in range(depth)]
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index in range(depth - 1):
            self.encoders.append(ConvBlock(channels[index], channels[index]))
            self.downsamples.append(nn.Conv2d(channels[index], channels[index + 1], 2, stride=2))
        self.middle = ConvBlock(channels[-1], channels[-1])
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index in range(depth - 1, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose2d(channels[index], channels[index - 1], 2, stride=2)
            )
            self.decoders.append(ConvBlock(channels[index - 1] * 2, channels[index - 1]))
        self.head = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, source: torch.Tensor, nlm: torch.Tensor | None = None) -> torch.Tensor:
        del nlm
        hidden = self.stem(source)
        skips: list[torch.Tensor] = []
        for encoder, downsample in zip(self.encoders, self.downsamples, strict=True):
            hidden = encoder(hidden)
            skips.append(hidden)
            hidden = downsample(hidden)
        hidden = self.middle(hidden)
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips), strict=True
        ):
            hidden = decoder(torch.cat((upsample(hidden), skip), dim=1))
        return (source + self.head(hidden)).clamp(0, 1)


class NAFBlock(nn.Module):
    """Small NAFNet-style image-restoration block."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        if hidden % 2:
            raise ValueError("expanded channel count must be even")
        self.norm1 = nn.GroupNorm(1, channels)
        self.project_in = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden // 2, hidden // 2, 1),
        )
        self.project_out = nn.Conv2d(hidden // 2, channels, 1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn_in = nn.Conv2d(channels, hidden, 1)
        self.ffn_out = nn.Conv2d(hidden // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def _gate(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return first * second

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(self.project_in(self.norm1(source)))
        hidden = self._gate(hidden)
        hidden = hidden * self.attention(hidden)
        source = source + self.beta * self.project_out(hidden)
        hidden = self._gate(self.ffn_in(self.norm2(source)))
        return source + self.gamma * self.ffn_out(hidden)


def tile_coordinate_channels(
    source: torch.Tensor,
    *,
    tile_size: int = TILE_SIZE,
) -> torch.Tensor:
    """Return periodic x/y positions within each tile in [-1, 1]."""
    height, width = source.shape[-2:]
    y = torch.arange(height, device=source.device, dtype=source.dtype).remainder(tile_size)
    x = torch.arange(width, device=source.device, dtype=source.dtype).remainder(tile_size)
    y = y / max(tile_size - 1, 1) * 2 - 1
    x = x / max(tile_size - 1, 1) * 2 - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx))[None].expand(source.shape[0], -1, -1, -1)


class TileAwareDualNAFNet(nn.Module):
    """Full-canvas NAF U-Net conditioned on raw, NLM and within-tile position."""

    def __init__(self, base: int = 24, depth: int = 3, blocks: int = 2) -> None:
        super().__init__()
        if depth < 2 or blocks < 1:
            raise ValueError("depth must be >=2 and blocks must be positive")
        channels = [base * 2**index for index in range(depth)]
        self.stem = nn.Conv2d(8, base, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index in range(depth - 1):
            self.encoders.append(nn.Sequential(*(NAFBlock(channels[index]) for _ in range(blocks))))
            self.downsamples.append(nn.Conv2d(channels[index], channels[index + 1], 2, stride=2))
        self.middle = nn.Sequential(*(NAFBlock(channels[-1]) for _ in range(blocks + 1)))
        self.upsamples = nn.ModuleList()
        self.reductions = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index in range(depth - 1, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose2d(channels[index], channels[index - 1], 2, stride=2)
            )
            self.reductions.append(nn.Conv2d(channels[index - 1] * 2, channels[index - 1], 1))
            self.decoders.append(
                nn.Sequential(*(NAFBlock(channels[index - 1]) for _ in range(blocks)))
            )
        self.head = nn.Conv2d(base, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, source: torch.Tensor, nlm: torch.Tensor | None = None) -> torch.Tensor:
        if nlm is None:
            raise ValueError("TileAwareDualNAFNet requires an NLM conditioning image")
        if source.shape != nlm.shape:
            raise ValueError("source and NLM tensors must have equal shapes")
        coordinates = tile_coordinate_channels(source)
        hidden = self.stem(torch.cat((source, nlm, coordinates), dim=1))
        skips: list[torch.Tensor] = []
        for encoder, downsample in zip(self.encoders, self.downsamples, strict=True):
            hidden = encoder(hidden)
            skips.append(hidden)
            hidden = downsample(hidden)
        hidden = self.middle(hidden)
        for upsample, reduction, decoder, skip in zip(
            self.upsamples,
            self.reductions,
            self.decoders,
            reversed(skips),
            strict=True,
        ):
            hidden = decoder(reduction(torch.cat((upsample(hidden), skip), dim=1)))
        return (nlm + self.head(hidden)).clamp(0, 1)


def image_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert uint8 RGB HWC to float BCHW on ``device``."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image, got {image.dtype} {image.shape}")
    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255)
    )


def tensor_image(value: torch.Tensor) -> np.ndarray:
    """Convert one float BCHW RGB tensor back to uint8 HWC."""
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
        raise ValueError(f"expected 1 x 3 x H x W tensor, got {tuple(value.shape)}")
    array = value.detach().clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    return np.rint(array * 255).clip(0, 255).astype(np.uint8)


@torch.inference_mode()
def restore_image(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    *,
    nlm_h: int = 9,
) -> np.ndarray:
    """Run one trained restorer on an arbitrary tile-aligned RGB canvas."""
    model.eval()
    conditioning = nlm_color(image, nlm_h)
    return tensor_image(model(image_tensor(image, device), image_tensor(conditioning, device)))
