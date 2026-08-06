"""Permutation-equivariant coordinate flow for small jigsaw blocks.

The model receives an unordered set of independently degraded 20x20 tiles and
a noisy 2-D coordinate for every tile.  It predicts the velocity of the
straight probability path from Gaussian coordinates to the tile's true grid
coordinate.  There are deliberately no token indices or positional embeddings:
permuting tiles and their current coordinates must only permute the output.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from config import FS
from eval_paired_alignment import TileEncoder


def grid_coordinates(side: int, *, device: torch.device | None = None) -> Tensor:
    """Row-major square grid scaled to [-1, 1] in (x, y) order."""
    if side < 2:
        raise ValueError("side must be at least 2")
    axis = torch.linspace(-1.0, 1.0, side, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(side * side, 2)


class RelativeCoordinateFlow(nn.Module):
    """Set-conditioned velocity field over one coordinate per tile."""

    def __init__(
        self,
        *,
        side: int = 4,
        d_model: int = 128,
        layers: int = 4,
        heads: int = 4,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.side = int(side)
        self.n_tiles = self.side * self.side
        self.d_model = int(d_model)
        self.tile_encoder = TileEncoder(embed_dim=d_model)
        self.state_encoder = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_mult * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def encode_tiles(self, tiles: Tensor) -> Tensor:
        if tiles.ndim != 5 or tiles.shape[1] != self.n_tiles:
            raise ValueError(
                f"tiles must have shape (B,{self.n_tiles},3,{FS},{FS}), got {tuple(tiles.shape)}"
            )
        batch, count = tiles.shape[:2]
        return self.tile_encoder(tiles.reshape(batch * count, 3, FS, FS)).reshape(
            batch, count, self.d_model
        )

    def velocity_from_embeddings(self, embeddings: Tensor, coordinates: Tensor, time: Tensor) -> Tensor:
        batch, count, _ = embeddings.shape
        if coordinates.shape != (batch, count, 2):
            raise ValueError("coordinate shape does not match embeddings")
        if time.ndim == 0:
            time = time.expand(batch)
        if time.shape != (batch,):
            raise ValueError("time must be scalar or shape (B,)")
        t = time[:, None, None].expand(batch, count, 1)
        state = torch.cat((coordinates, t, torch.sin(math.pi * t), torch.cos(math.pi * t)), dim=-1)
        tokens = embeddings + self.state_encoder(state)
        return self.output(self.transformer(tokens))

    def forward(self, tiles: Tensor, coordinates: Tensor, time: Tensor) -> Tensor:
        return self.velocity_from_embeddings(self.encode_tiles(tiles), coordinates, time)


def flow_matching_loss(
    model: RelativeCoordinateFlow,
    tiles: Tensor,
    target_coordinates: Tensor,
    *,
    pair_weight: float = 0.25,
    grid_weight: float = 0.25,
) -> tuple[Tensor, dict[str, float]]:
    """Conditional flow matching plus endpoint geometry auxiliaries."""
    batch = tiles.shape[0]
    x0 = torch.randn_like(target_coordinates)
    time = torch.rand(batch, device=tiles.device).clamp_(0.01, 0.99)
    t = time[:, None, None]
    xt = (1.0 - t) * x0 + t * target_coordinates
    target_velocity = target_coordinates - x0
    velocity = model(tiles, xt, time)
    endpoint = xt + (1.0 - t) * velocity

    velocity_loss = F.mse_loss(velocity, target_velocity)
    pred_delta = endpoint[:, :, None, :] - endpoint[:, None, :, :]
    true_delta = target_coordinates[:, :, None, :] - target_coordinates[:, None, :, :]
    pair_loss = F.smooth_l1_loss(pred_delta, true_delta)

    slots = grid_coordinates(model.side, device=tiles.device)
    logits = -(endpoint[:, :, None, :] - slots[None, None, :, :]).square().sum(dim=-1) / 0.15
    labels = torch.cdist(target_coordinates, slots[None].expand(batch, -1, -1)).argmin(dim=-1)
    grid_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    loss = velocity_loss + pair_weight * pair_loss + grid_weight * grid_loss
    return loss, {
        "loss": float(loss.detach()),
        "velocity_loss": float(velocity_loss.detach()),
        "pair_loss": float(pair_loss.detach()),
        "grid_loss": float(grid_loss.detach()),
    }


@torch.inference_mode()
def integrate_flow(
    model: RelativeCoordinateFlow,
    tiles: Tensor,
    *,
    steps: int = 20,
    seed: int = 0,
) -> Tensor:
    """Euler integration from deterministic Gaussian coordinates."""
    if steps < 1:
        raise ValueError("steps must be positive")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=tiles.device).manual_seed(seed)
    x = torch.randn(
        tiles.shape[0], model.n_tiles, 2, generator=generator, device=tiles.device
    )
    embeddings = model.encode_tiles(tiles)
    dt = 1.0 / steps
    for step in range(steps):
        time = torch.full((tiles.shape[0],), (step + 0.5) * dt, device=tiles.device)
        x = x + dt * model.velocity_from_embeddings(embeddings, x, time)
        x.clamp_(-3.0, 3.0)
    if was_training:
        model.train()
    return x


def hungarian_slots(coordinates: Tensor, side: int) -> Tensor:
    """Map every token to one unique row-major grid slot."""
    slots = grid_coordinates(side, device=coordinates.device)
    costs = torch.cdist(coordinates.float(), slots).detach().cpu().numpy()
    assignments: list[np.ndarray] = []
    for cost in costs:
        rows, cols = linear_sum_assignment(cost)
        token_to_slot = np.empty(side * side, dtype=np.int64)
        token_to_slot[rows] = cols
        assignments.append(token_to_slot)
    return torch.from_numpy(np.stack(assignments)).to(coordinates.device)


def arrangement_metrics(predicted_slots: Tensor, true_slots: Tensor, side: int) -> dict[str, float]:
    """Exact placement and undirected true-neighbor recovery."""
    if predicted_slots.shape != true_slots.shape:
        raise ValueError("predicted and true slots must share shape")
    placement = predicted_slots.eq(true_slots).float().mean()
    neighbor_scores: list[float] = []
    for predicted, truth in zip(predicted_slots.detach().cpu(), true_slots.detach().cpu()):
        slot_to_token = torch.empty_like(predicted)
        slot_to_token[predicted] = torch.arange(predicted.numel())
        recovered = 0
        total = 2 * side * (side - 1)
        for row in range(side):
            for col in range(side):
                slot = row * side + col
                if col + 1 < side:
                    a = int(truth[slot_to_token[slot]])
                    b = int(truth[slot_to_token[slot + 1]])
                    recovered += int(abs(a - b) == 1 and a // side == b // side)
                if row + 1 < side:
                    a = int(truth[slot_to_token[slot]])
                    b = int(truth[slot_to_token[slot + side]])
                    recovered += int(abs(a - b) == side)
        neighbor_scores.append(recovered / total)
    return {
        "placement_accuracy": float(placement),
        "neighbor_accuracy": float(np.mean(neighbor_scores)),
        "perfect_puzzles": float(
            predicted_slots.eq(true_slots).all(dim=1).float().mean()
        ),
    }


def load_paired_dirty_encoder(model: RelativeCoordinateFlow, checkpoint: str) -> dict[str, Any]:
    """Initialize the tile encoder from the successful dirty-clean retrieval model."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"]
    prefix = "dirty_encoder."
    dirty_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    missing, unexpected = model.tile_encoder.load_state_dict(dirty_state, strict=True)
    return {
        "checkpoint": checkpoint,
        "source_step": int(payload.get("step", -1)),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def smoke_test(device: torch.device = torch.device("cpu")) -> dict[str, float]:
    torch.manual_seed(8128)
    model = RelativeCoordinateFlow(side=4, d_model=32, layers=2, heads=4).to(device)
    tiles = torch.rand(2, 16, 3, FS, FS, device=device)
    slots = grid_coordinates(4, device=device)
    permutation = torch.randperm(16, device=device)
    target = slots[permutation][None].expand(2, -1, -1)
    loss, parts = flow_matching_loss(model, tiles, target)
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError("non-finite flow loss")
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError("model received no finite gradient")

    model.eval()
    coordinates = torch.randn(2, 16, 2, device=device)
    time = torch.tensor([0.2, 0.8], device=device)
    with torch.no_grad():
        direct = model(tiles, coordinates, time)
        permuted = model(tiles[:, permutation], coordinates[:, permutation], time)
    equivariance_error = (direct[:, permutation] - permuted).abs().max()
    if equivariance_error > 2.0e-5:
        raise AssertionError(f"permutation equivariance failed: {float(equivariance_error)}")

    decoded = hungarian_slots(target, 4)
    true_slots = torch.cdist(target, slots[None].expand(2, -1, -1)).argmin(dim=-1)
    perfect = arrangement_metrics(decoded, true_slots, 4)
    if perfect["placement_accuracy"] != 1.0 or perfect["neighbor_accuracy"] != 1.0:
        raise AssertionError("Hungarian/metric perfect-case contract failed")
    return {**parts, "equivariance_max_abs": float(equivariance_error), **perfect}

