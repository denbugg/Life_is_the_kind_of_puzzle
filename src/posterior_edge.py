"""Stochastic residual hypotheses around a frozen deterministic tile denoiser."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import FS


def _groups(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.gelu(value + self.net(value))


class PosteriorEdgeRestorer(nn.Module):
    """Generate a bounded clean-tile residual conditioned on a Gaussian latent."""

    def __init__(
        self,
        *,
        width: int = 32,
        blocks: int = 4,
        latent_dim: int = 8,
        max_residual: float = 0.20,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.blocks = int(blocks)
        self.latent_dim = int(latent_dim)
        self.max_residual = float(max_residual)
        # dirty RGB + per-tile normalized RGB + deterministic clean mean + latent
        self.stem = nn.Sequential(
            nn.Conv2d(9 + latent_dim, width, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
        )
        self.body = nn.Sequential(*[_ResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        # Start exactly at the deterministic denoiser. Latent specialization
        # appears only when best-of-K gradients separate the hypotheses.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def model_kwargs(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "blocks": self.blocks,
            "latent_dim": self.latent_dim,
            "max_residual": self.max_residual,
        }

    @staticmethod
    def normalize(dirty: Tensor) -> Tensor:
        mean = dirty.mean(dim=(-3, -2, -1), keepdim=True)
        std = dirty.std(dim=(-3, -2, -1), keepdim=True).clamp_min(1.0e-3)
        return ((dirty - mean) / std).clamp(-5.0, 5.0)

    def forward(self, dirty: Tensor, deterministic_mean: Tensor, latent: Tensor) -> Tensor:
        if dirty.ndim != 4 or tuple(dirty.shape[1:]) != (3, FS, FS):
            raise ValueError(f"dirty must have shape (B,3,{FS},{FS})")
        if deterministic_mean.shape != dirty.shape:
            raise ValueError("deterministic_mean must match dirty")
        if latent.shape != (dirty.shape[0], self.latent_dim):
            raise ValueError("latent shape does not match batch/latent_dim")
        z = latent[:, :, None, None].expand(-1, -1, FS, FS)
        features = torch.cat((dirty, self.normalize(dirty), deterministic_mean, z), dim=1)
        residual = torch.tanh(self.head(self.body(self.stem(features)))) * self.max_residual
        return (deterministic_mean + residual).clamp(0.0, 1.0)

    def sample(
        self,
        dirty: Tensor,
        deterministic_mean: Tensor,
        *,
        hypotheses: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if hypotheses < 1:
            raise ValueError("hypotheses must be positive")
        batch = dirty.shape[0]
        latent = torch.randn(
            hypotheses,
            batch,
            self.latent_dim,
            device=dirty.device,
            generator=generator,
        )
        dirty_k = dirty.unsqueeze(0).expand(hypotheses, -1, -1, -1, -1)
        mean_k = deterministic_mean.unsqueeze(0).expand_as(dirty_k)
        return self(
            dirty_k.reshape(-1, 3, FS, FS),
            mean_k.reshape(-1, 3, FS, FS),
            latent.reshape(-1, self.latent_dim),
        ).reshape(hypotheses, batch, 3, FS, FS)


def boundary_pixels(tiles: Tensor, band: int = 4) -> Tensor:
    """Flatten four boundary strips while avoiding corner duplication."""
    if tiles.shape[-2:] != (FS, FS) or band < 1 or band * 2 >= FS:
        raise ValueError("invalid tiles or boundary band")
    top = tiles[..., :band, :]
    bottom = tiles[..., -band:, :]
    left = tiles[..., band:-band, :band]
    right = tiles[..., band:-band, -band:]
    return torch.cat(
        (
            top.flatten(start_dim=-2),
            bottom.flatten(start_dim=-2),
            left.flatten(start_dim=-2),
            right.flatten(start_dim=-2),
        ),
        dim=-1,
    )


def best_of_k_edge_loss(
    hypotheses: Tensor,
    clean: Tensor,
    deterministic_mean: Tensor,
    *,
    band: int = 4,
    full_weight: float = 0.10,
    diversity_floor: float = 0.012,
    diversity_weight: float = 0.20,
    residual_weight: float = 0.02,
) -> tuple[Tensor, dict[str, float]]:
    """Winner-takes-most reconstruction plus a small anti-collapse constraint."""
    if hypotheses.ndim != 5 or hypotheses.shape[1:] != clean.shape:
        raise ValueError("hypotheses must have shape (K,B,3,H,W)")
    target_edge = boundary_pixels(clean, band).unsqueeze(0)
    predicted_edge = boundary_pixels(hypotheses, band)
    edge_error = (predicted_edge - target_edge).abs().mean(dim=(-1, -2))
    full_error = (hypotheses - clean.unsqueeze(0)).abs().mean(dim=(-1, -2, -3))
    reconstruction = (edge_error + full_weight * full_error).min(dim=0).values.mean()

    if hypotheses.shape[0] > 1:
        pairwise = (
            predicted_edge[:, None] - predicted_edge[None, :]
        ).abs().mean(dim=(-1, -2))
        mask = ~torch.eye(hypotheses.shape[0], dtype=torch.bool, device=hypotheses.device)
        diversity = pairwise[:, :, :].masked_select(mask[:, :, None]).mean()
        diversity_penalty = F.relu(hypotheses.new_tensor(diversity_floor) - diversity)
    else:
        diversity = hypotheses.new_zeros(())
        diversity_penalty = hypotheses.new_zeros(())
    residual_l2 = (hypotheses - deterministic_mean.unsqueeze(0)).square().mean()
    loss = (
        reconstruction
        + diversity_weight * diversity_penalty
        + residual_weight * residual_l2
    )
    return loss, {
        "loss": float(loss.detach()),
        "best_edge_reconstruction": float(reconstruction.detach()),
        "edge_diversity": float(diversity.detach()),
        "diversity_penalty": float(diversity_penalty.detach()),
        "residual_l2": float(residual_l2.detach()),
    }


def smoke_test(device: torch.device = torch.device("cpu")) -> dict[str, float]:
    torch.manual_seed(404)
    model = PosteriorEdgeRestorer(width=8, blocks=2, latent_dim=4).to(device)
    dirty = torch.rand(3, 3, FS, FS, device=device)
    mean = torch.rand_like(dirty)
    clean = torch.rand_like(dirty)
    output = model.sample(dirty, mean, hypotheses=4)
    if output.shape != (4, 3, 3, FS, FS):
        raise AssertionError("unexpected posterior sample shape")
    loss, parts = best_of_k_edge_loss(output, clean, mean)
    loss.backward()
    if not any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise AssertionError("posterior edge model received no finite gradient")
    if not torch.isfinite(output).all() or output.min() < 0 or output.max() > 1:
        raise AssertionError("posterior samples left [0,1]")
    return parts


if __name__ == "__main__":
    print(smoke_test(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
