"""Latent image prior and unordered-bag initialiser for the next jigsaw gate.

The previous CanvasNet tried to amortise ``bag -> ordered canvas`` directly and
collapsed to a mean scene.  Here the ordered image is constrained to a compact
VAE manifold, then fitted per puzzle by differentiable optimal transport.  The
bag encoder is merely an optional initialiser, never the final canvas decoder.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for group in range(min(8, channels), 0, -1):
        if channels % group == 0:
            return group
    return 1


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        g = _groups(channels)
        self.net = nn.Sequential(
            nn.GroupNorm(g, channels), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(g, channels), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CanvasVAE(nn.Module):
    """A compact global-latent VAE for 96x96 (or other /16) clean canvases.

    A global latent is intentional: a very expressive spatial latent could fit
    any scrambled bag and would no longer impose an image-level layout prior.
    ``zdim=256`` is a useful initial trade-off for 7,000 narrow-domain photos.
    """

    def __init__(self, image_size: int = 96, zdim: int = 256, base: int = 32) -> None:
        super().__init__()
        if image_size <= 0 or image_size % 16:
            raise ValueError("image_size must be positive and divisible by 16")
        if zdim <= 0 or base <= 0:
            raise ValueError("zdim and base must be positive")
        self.image_size = int(image_size)
        self.zdim = int(zdim)
        self.base = int(base)
        side = image_size // 16
        self.side = side

        def down(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, cout, 4, stride=2, padding=1),
                                 nn.GroupNorm(_groups(cout), cout), nn.GELU(), _ResBlock(cout))
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.GroupNorm(_groups(base), base), nn.GELU(), _ResBlock(base),
            down(base, base * 2), down(base * 2, base * 4), down(base * 4, base * 8), down(base * 8, base * 8),
        )
        hidden = base * 8 * side * side
        self.to_mu = nn.Linear(hidden, zdim)
        self.to_logvar = nn.Linear(hidden, zdim)
        self.from_z = nn.Sequential(nn.Linear(zdim, hidden), nn.GELU())

        def up(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
                                 nn.GroupNorm(_groups(cout), cout), nn.GELU(), _ResBlock(cout))
        self.decoder = nn.Sequential(
            _ResBlock(base * 8),
            up(base * 8, base * 8), up(base * 8, base * 4), up(base * 4, base * 2), up(base * 2, base),
            nn.Conv2d(base, 3, 3, padding=1),
        )

    def encode(self, canvas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if canvas.ndim != 4 or tuple(canvas.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"expected (B,3,{self.image_size},{self.image_size}), got {tuple(canvas.shape)}")
        h = self.encoder(canvas).flatten(1)
        return self.to_mu(h), self.to_logvar(h).clamp(-12.0, 8.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.zdim:
            raise ValueError(f"expected (B,{self.zdim}) latent, got {tuple(z.shape)}")
        h = self.from_z(z).reshape(z.shape[0], self.base * 8, self.side, self.side)
        return torch.sigmoid(self.decoder(h))

    def forward(self, canvas: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(canvas)
        z = self.reparameterize(mu, logvar)
        return {"recon": self.decode(z), "mu": mu, "logvar": logvar, "z": z}


class _TileEncoder(nn.Module):
    def __init__(self, d: int = 128) -> None:
        super().__init__()
        width = max(32, d // 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.GroupNorm(_groups(width), width), nn.GELU(), _ResBlock(width),
            nn.Conv2d(width, d, 3, stride=2, padding=1), nn.GroupNorm(_groups(d), d), nn.GELU(), _ResBlock(d),
            nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GroupNorm(_groups(d), d), nn.GELU(), _ResBlock(d),
        )
        self.proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(d, d), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


class BagLatentEncoder(nn.Module):
    """Permutation-invariant latent initialiser; it intentionally emits no grid."""

    def __init__(self, zdim: int = 256, d: int = 128, tiles: int = 576) -> None:
        super().__init__()
        self.zdim, self.d, self.tiles = int(zdim), int(d), int(tiles)
        self.tile = _TileEncoder(d)
        self.post = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, zdim))

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        if tiles.ndim != 5 or tiles.shape[1] != self.tiles or tiles.shape[2] != 3:
            raise ValueError(f"expected (B,{self.tiles},3,H,W), got {tuple(tiles.shape)}")
        b, n, c, h, w = tiles.shape
        token = self.tile(tiles.reshape(b * n, c, h, w)).reshape(b, n, self.d)
        # Mean + standard deviation retain distributional scene evidence while
        # remaining exactly insensitive to the shuffled input order.
        pooled = torch.cat((token.mean(1), token.std(1, unbiased=False)), dim=-1)
        return self.post(pooled)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    vae = CanvasVAE()
    bag = BagLatentEncoder()
    x = torch.rand(2, 3, 96, 96)
    tiles = torch.rand(2, 576, 3, 20, 20)
    out = vae(x)
    print({k: tuple(v.shape) for k, v in out.items()}, count_params(vae))
    print(tuple(bag(tiles).shape), count_params(bag))
