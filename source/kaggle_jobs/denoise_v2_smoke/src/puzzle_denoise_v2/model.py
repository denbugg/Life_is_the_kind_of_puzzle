"""Small NAF-style restorers specialized for isolated 20x20 RGB tiles."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=1)
        return left * right


class DegradationEncoder(nn.Module):
    def __init__(self, code_dim: int = 32, parameter_dim: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1, groups=24, padding_mode="reflect"),
            nn.Conv2d(24, 48, 1),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, stride=2, padding=1, groups=48, padding_mode="reflect"),
            nn.Conv2d(48, 64, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_code = nn.Sequential(nn.Flatten(), nn.Linear(64, code_dim), nn.Tanh())
        self.to_parameters = nn.Linear(code_dim, parameter_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.to_code(self.features(x))
        return code, self.to_parameters(code)


class FiLMNAFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        code_dim: int,
        expansion: int = 2,
        ffn_expansion: int = 2,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        expanded = channels * expansion
        ffn_channels = channels * ffn_expansion
        if expanded % 2 or ffn_channels % 2:
            raise ValueError("expanded channel counts must be even")

        self.norm1 = LayerNorm2d(channels)
        self.film1 = nn.Linear(code_dim, channels * 2)
        self.in_conv = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(
            expanded,
            expanded,
            3,
            padding=dilation,
            dilation=dilation,
            groups=expanded,
            padding_mode="reflect",
        )
        self.gate1 = SimpleGate()
        gated = expanded // 2
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(gated, gated, 1))
        self.out_conv = nn.Conv2d(gated, channels, 1)
        self.dropout1 = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.film2 = nn.Linear(code_dim, channels * 2)
        self.ffn_in = nn.Conv2d(channels, ffn_channels, 1)
        self.gate2 = SimpleGate()
        self.ffn_out = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def _film(x: torch.Tensor, projection: nn.Linear, code: torch.Tensor) -> torch.Tensor:
        scale, bias = projection(code).chunk(2, dim=1)
        scale = 0.25 * torch.tanh(scale)[:, :, None, None]
        bias = 0.25 * torch.tanh(bias)[:, :, None, None]
        return x * (1.0 + scale) + bias

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        y = self._film(self.norm1(x), self.film1, code)
        y = self.gate1(self.depthwise(self.in_conv(y)))
        y = y * self.channel_attention(y)
        x = x + self.dropout1(self.out_conv(y)) * self.beta

        y = self._film(self.norm2(x), self.film2, code)
        y = self.ffn_out(self.gate2(self.ffn_in(y)))
        return x + self.dropout2(y) * self.gamma


class BlockStack(nn.Module):
    def __init__(self, channels: int, count: int, code_dim: int, dilations=(1,)) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [FiLMNAFBlock(channels, code_dim, dilation=dilations[index % len(dilations)]) for index in range(count)]
        )

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, code)
        return x


class FullResolutionTileNAF(nn.Module):
    """No-downsampling control model for border-preserving ablations."""

    def __init__(self, width: int = 48, blocks: int = 12, code_dim: int = 32) -> None:
        super().__init__()
        self.degradation_encoder = DegradationEncoder(code_dim)
        self.stem = nn.Conv2d(3, width, 3, padding=1, padding_mode="reflect")
        self.body = BlockStack(width, blocks, code_dim, dilations=(1, 2, 1, 3))
        self.tail = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        code, parameter_prediction = self.degradation_encoder(x)
        residual = self.tail(self.body(self.stem(x), code))
        restored = (x + residual).clamp(0.0, 1.0)
        return (restored, parameter_prediction) if return_aux else restored


class TileNAFNet(nn.Module):
    """Two-level 20->10->5 NAF-style tile restorer with FiLM conditioning."""

    def __init__(
        self,
        width: int = 48,
        encoder_blocks: tuple[int, int] = (2, 4),
        middle_blocks: int = 8,
        decoder_blocks: tuple[int, int] = (4, 2),
        code_dim: int = 32,
    ) -> None:
        super().__init__()
        self.degradation_encoder = DegradationEncoder(code_dim)
        self.stem = nn.Conv2d(3, width, 3, padding=1, padding_mode="reflect")

        self.encoder1 = BlockStack(width, encoder_blocks[0], code_dim)
        self.down1 = nn.Conv2d(width, width * 2, 2, stride=2)
        self.encoder2 = BlockStack(width * 2, encoder_blocks[1], code_dim)
        self.down2 = nn.Conv2d(width * 2, width * 4, 2, stride=2)

        self.middle = BlockStack(width * 4, middle_blocks, code_dim)

        self.up2 = nn.Sequential(nn.Conv2d(width * 4, width * 8, 1), nn.PixelShuffle(2))
        self.decoder2 = BlockStack(width * 2, decoder_blocks[0], code_dim)
        self.up1 = nn.Sequential(nn.Conv2d(width * 2, width * 4, 1), nn.PixelShuffle(2))
        self.decoder1 = BlockStack(width, decoder_blocks[1], code_dim)
        self.tail = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if tuple(x.shape[-2:]) != (20, 20):
            raise ValueError(f"TileNAFNet requires 20x20 tiles, got {tuple(x.shape[-2:])}")
        code, parameter_prediction = self.degradation_encoder(x)
        base = self.stem(x)
        skip1 = self.encoder1(base, code)
        skip2 = self.encoder2(self.down1(skip1), code)
        features = self.middle(self.down2(skip2), code)
        features = self.decoder2(self.up2(features) + skip2, code)
        features = self.decoder1(self.up1(features) + skip1, code)
        restored = (x + self.tail(features)).clamp(0.0, 1.0)
        return (restored, parameter_prediction) if return_aux else restored


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
