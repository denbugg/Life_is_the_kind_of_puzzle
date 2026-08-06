"""Discrete denoising/refinement of anonymous balanced macro partitions."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


class _RefinementBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_mult: int) -> None:
        super().__init__()
        self.tile_norm = nn.LayerNorm(d_model)
        self.group_norm = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True
        )
        self.tile_ff_norm = nn.LayerNorm(d_model)
        self.tile_ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.group_self_norm = nn.LayerNorm(d_model)
        self.group_self = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True
        )
        self.group_ff_norm = nn.LayerNorm(d_model)
        self.group_ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, tiles: Tensor, groups: Tensor) -> tuple[Tensor, Tensor]:
        group_normalized = self.group_self_norm(groups)
        groups = groups + self.group_self(
            group_normalized,
            group_normalized,
            group_normalized,
            need_weights=False,
        )[0]
        groups = groups + self.group_ff(self.group_ff_norm(groups))
        tiles = tiles + self.cross(
            self.tile_norm(tiles),
            self.group_norm(groups),
            self.group_norm(groups),
            need_weights=False,
        )[0]
        tiles = tiles + self.tile_ff(self.tile_ff_norm(tiles))
        return tiles, groups


class BalancedPartitionRefiner(nn.Module):
    """Predict a cleaner 36-way assignment from embeddings and a noisy partition.

    Group IDs have no learned embeddings. A group is represented only by the
    symmetric mean of its current members; permuting group labels therefore
    permutes output columns exactly.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 128,
        d_model: int = 128,
        groups: int = 36,
        capacity: int = 16,
        layers: int = 3,
        heads: int = 4,
        ff_mult: int = 3,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.embed_dim = int(embed_dim)
        self.d_model = int(d_model)
        self.groups = int(groups)
        self.capacity = int(capacity)
        self.layers = int(layers)
        self.tile_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.current_group_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            _RefinementBlock(d_model, heads, ff_mult) for _ in range(layers)
        )
        self.tile_match = nn.Linear(d_model, d_model, bias=False)
        self.group_match = nn.Linear(d_model, d_model, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        # Residual-denoising prior: copying the current anonymous group is a
        # strong baseline. The network must supply evidence before moving a
        # tile, while exact-capacity decoding turns moves into balanced swaps.
        self.identity_logit = nn.Parameter(torch.tensor(4.0))

    @property
    def model_kwargs(self) -> dict[str, int]:
        return {
            "embed_dim": self.embed_dim,
            "d_model": self.d_model,
            "groups": self.groups,
            "capacity": self.capacity,
            "layers": self.layers,
        }

    def group_means(self, tiles: Tensor, assignment: Tensor) -> Tensor:
        membership = F.one_hot(assignment.long(), num_classes=self.groups).to(tiles.dtype)
        counts = membership.sum(dim=1).unsqueeze(-1).clamp_min(1.0)
        return (membership.transpose(1, 2) @ tiles) / counts

    def forward(
        self,
        embeddings: Tensor,
        assignment: Tensor,
        noise_level: Tensor,
    ) -> Tensor:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.embed_dim:
            raise ValueError("embeddings must have shape (B,N,embed_dim)")
        batch, count = embeddings.shape[:2]
        if assignment.shape != (batch, count):
            raise ValueError("assignment must align with embeddings")
        if noise_level.ndim == 0:
            noise_level = noise_level.expand(batch)
        if noise_level.shape != (batch,):
            raise ValueError("noise_level must be scalar or shape (B,)")
        if torch.any(assignment < 0) or torch.any(assignment >= self.groups):
            raise ValueError("assignment contains an invalid group")

        tiles = self.tile_projection(embeddings)
        groups = self.group_means(tiles, assignment)
        current = groups.gather(
            1, assignment[:, :, None].expand(-1, -1, self.d_model)
        )
        t = noise_level[:, None]
        time = self.time_projection(
            torch.cat((t, torch.sin(math.pi * t), torch.cos(math.pi * t)), dim=-1)
        )
        tiles = tiles + self.current_group_projection(current) + time[:, None, :]
        for block in self.blocks:
            tiles, groups = block(tiles, groups)
        tile_keys = F.normalize(self.tile_match(tiles), dim=-1)
        group_keys = F.normalize(self.group_match(groups), dim=-1)
        logits = (tile_keys @ group_keys.transpose(1, 2)) * self.logit_scale.exp().clamp(max=50.0)
        return logits + self.identity_logit.clamp(min=0.0, max=12.0) * F.one_hot(
            assignment, num_classes=self.groups
        ).to(logits.dtype)


def capacity_preserving_corruption(
    target: Tensor,
    corruption: Tensor,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Shuffle labels on a subset of tiles while preserving every group count."""
    if target.ndim != 2 or corruption.shape != (target.shape[0],):
        raise ValueError("invalid target/corruption shapes")
    output = target.clone()
    for image in range(target.shape[0]):
        selected_count = max(2, int(round(float(corruption[image]) * target.shape[1])))
        selected = torch.randperm(
            target.shape[1], device=target.device, generator=generator
        )[:selected_count]
        shuffled = selected[
            torch.randperm(selected_count, device=target.device, generator=generator)
        ]
        output[image, selected] = target[image, shuffled]
    actual = output.ne(target).float().mean(dim=1)
    return output, actual


def refinement_loss(
    logits: Tensor,
    target: Tensor,
    *,
    capacity_weight: float = 0.10,
) -> tuple[Tensor, dict[str, float]]:
    if logits.shape[:2] != target.shape:
        raise ValueError("logits and target do not align")
    ce = F.cross_entropy(logits.flatten(0, 1), target.flatten())
    expected = F.softmax(logits.float(), dim=-1).sum(dim=1)
    capacity = (
        (expected - logits.shape[1] / logits.shape[2])
        / (logits.shape[1] / logits.shape[2])
    ).square().mean()
    loss = ce + capacity_weight * capacity
    return loss, {
        "loss": float(loss.detach()),
        "cross_entropy": float(ce.detach()),
        "capacity_loss": float(capacity.detach()),
    }


def exact_capacity_decode(logits: Tensor, capacity: int) -> Tensor:
    """Hungarian assignment with exactly ``capacity`` tiles per group."""
    if logits.ndim != 3 or logits.shape[1] != logits.shape[2] * capacity:
        raise ValueError("logits do not match the requested balanced capacity")
    decoded: list[np.ndarray] = []
    for score in logits.detach().float().cpu().numpy():
        expanded = np.repeat(-score, capacity, axis=1)
        rows, virtual = linear_sum_assignment(expanded)
        assignment = np.empty(score.shape[0], dtype=np.int64)
        assignment[rows] = virtual // capacity
        decoded.append(assignment)
    return torch.from_numpy(np.stack(decoded)).to(logits.device)


@torch.inference_mode()
def iterative_refine(
    model: BalancedPartitionRefiner,
    embeddings: Tensor,
    initial: Tensor,
    *,
    noise_schedule: tuple[float, ...] = (0.75, 0.55, 0.35, 0.15),
    inertia: float = 0.25,
) -> list[Tensor]:
    assignment = initial
    outputs: list[Tensor] = []
    for noise in noise_schedule:
        level = embeddings.new_full((embeddings.shape[0],), noise)
        logits = model(embeddings, assignment, level)
        if inertia:
            logits = logits + inertia * F.one_hot(
                assignment, num_classes=model.groups
            ).to(logits.dtype)
        assignment = exact_capacity_decode(logits, model.capacity)
        outputs.append(assignment)
    return outputs


def smoke_test(device: torch.device = torch.device("cpu")) -> dict[str, float]:
    torch.manual_seed(612)
    groups, capacity, dim = 4, 3, 12
    model = BalancedPartitionRefiner(
        embed_dim=dim,
        d_model=24,
        groups=groups,
        capacity=capacity,
        layers=2,
        heads=4,
    ).to(device)
    embeddings = torch.randn(2, groups * capacity, dim, device=device)
    target = torch.arange(groups, device=device).repeat_interleave(capacity)[None].expand(2, -1)
    generator = torch.Generator(device=device).manual_seed(99)
    noisy, actual = capacity_preserving_corruption(
        target,
        torch.tensor([0.5, 0.8], device=device),
        generator=generator,
    )
    logits = model(embeddings, noisy, actual)
    loss, parts = refinement_loss(logits, target)
    loss.backward()
    decoded = exact_capacity_decode(logits, capacity)
    counts = F.one_hot(decoded, num_classes=groups).sum(dim=1)
    if not torch.equal(counts, torch.full_like(counts, capacity)):
        raise AssertionError("capacity decoder violated exact counts")

    tile_permutation = torch.randperm(groups * capacity, device=device)
    group_permutation = torch.randperm(groups, device=device)
    inverse_group = torch.argsort(group_permutation)
    model.eval()
    with torch.no_grad():
        reference = model(embeddings, noisy, actual)
        permuted_assignment = group_permutation[noisy[:, tile_permutation]]
        permuted = model(
            embeddings[:, tile_permutation], permuted_assignment, actual
        )
        restored = permuted[:, torch.argsort(tile_permutation)][:, :, group_permutation]
    error = (reference - restored).abs().max()
    if error > 3.0e-5:
        raise AssertionError(f"tile/group permutation equivariance failed: {float(error)}")
    return {**parts, "equivariance_max_abs": float(error)}


if __name__ == "__main__":
    print(smoke_test(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
