"""Full-board positional diffusion for the fixed-orientation 24x24 puzzle.

This is deliberately permutation equivariant: input tiles have no sequence
position embedding.  The only changing state is one noisy 2-D output coordinate
per tile.  A transformer lets every tile reason about every other tile, matching
the complete-graph continuous formulation used by Positional Diffusion.
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
    axis = torch.linspace(-1.0, 1.0, side, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(side * side, 2)


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """Standard sinusoidal embedding for integer diffusion timesteps."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class DiffusionSchedule(nn.Module):
    """Linear DDPM schedule, stored as buffers with gather helpers."""

    def __init__(self, steps: int = 300, beta_start: float = 1.0e-4, beta_end: float = 2.0e-2) -> None:
        super().__init__()
        beta = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.steps = int(steps)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    @staticmethod
    def _extract(values: Tensor, time: Tensor, ndim: int) -> Tensor:
        return values[time].reshape(time.shape[0], *((1,) * (ndim - 1)))

    def q_sample(self, clean: Tensor, time: Tensor, noise: Tensor) -> Tensor:
        alpha_bar = self._extract(self.alpha_bar, time, clean.ndim)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def predict_clean(self, noisy: Tensor, time: Tensor, predicted_noise: Tensor) -> Tensor:
        alpha_bar = self._extract(self.alpha_bar, time, noisy.ndim)
        return (noisy - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt().clamp_min(1.0e-5)


class PositionalDDPM(nn.Module):
    """Predict DDPM noise for an unordered full puzzle of fixed-orientation tiles."""

    def __init__(
        self,
        *,
        side: int = 24,
        tile_dim: int = 128,
        d_model: int = 192,
        layers: int = 4,
        heads: int = 6,
        diffusion_steps: int = 300,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.side = int(side)
        self.n_tiles = self.side * self.side
        self.tile_dim = int(tile_dim)
        self.d_model = int(d_model)
        # Keep the 3x3 backbone map.  The paired-retrieval encoder's original
        # mean/std pooling was excellent for tile identity but erased which
        # content touched each physical side -- fatal for assembly.
        self.tile_backbone = TileEncoder(embed_dim=tile_dim).features
        self.tile_project = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(tile_dim * 3 * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.coordinate_project = nn.Sequential(
            nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.time_project = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2)
        )
        edge_dim = max(32, d_model // 2)
        self.edge_queries = nn.ModuleList([nn.Linear(d_model, edge_dim) for _ in range(4)])
        self.edge_keys = nn.ModuleList([nn.Linear(d_model, edge_dim) for _ in range(4)])
        self.edge_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.schedule = DiffusionSchedule(diffusion_steps)

    def encode_tiles(self, tiles: Tensor) -> Tensor:
        if tiles.ndim != 5 or tiles.shape[1] != self.n_tiles or tuple(tiles.shape[2:]) != (3, FS, FS):
            raise ValueError(
                f"tiles must have shape (B,{self.n_tiles},3,{FS},{FS}), got {tuple(tiles.shape)}"
            )
        batch, count = tiles.shape[:2]
        flat = tiles.reshape(batch * count, 3, FS, FS)
        mean = flat.mean(dim=(-3, -2, -1), keepdim=True)
        rms = (flat - mean).square().mean(dim=(-3, -2, -1), keepdim=True)
        normalized = ((flat - mean) / rms.add(1.0e-5).sqrt()).clamp(-5.0, 5.0)
        spatial = self.tile_backbone(torch.cat((flat, normalized), dim=1))
        if tuple(spatial.shape[-2:]) != (3, 3):
            raise RuntimeError(f"unexpected tile backbone map {tuple(spatial.shape)}")
        return self.tile_project(spatial).reshape(batch, count, self.d_model)

    def predict_noise_from_embeddings(self, tile_features: Tensor, coordinates: Tensor, time: Tensor) -> Tensor:
        if coordinates.shape[:2] != tile_features.shape[:2] or coordinates.shape[-1] != 2:
            raise ValueError("coordinate and tile-feature shapes do not match")
        if time.ndim == 0:
            time = time.expand(tile_features.shape[0])
        time_features = self.time_project(timestep_embedding(time, self.d_model)).unsqueeze(1)
        tokens = tile_features + self.coordinate_project(coordinates) + time_features
        return self.output(self.transformer(tokens))

    def forward(self, tiles: Tensor, coordinates: Tensor, time: Tensor) -> Tensor:
        return self.predict_noise_from_embeddings(self.encode_tiles(tiles), coordinates, time)

    def directional_edge_scores(self, tile_features: Tensor) -> Tensor:
        """All-pairs U/D/L/R logits from the spatial tile representation."""
        rows: list[Tensor] = []
        scale = self.edge_logit_scale.exp().clamp(max=100.0)
        diagonal = torch.eye(self.n_tiles, device=tile_features.device, dtype=torch.bool)[None]
        for direction in range(4):
            query = F.normalize(self.edge_queries[direction](tile_features), dim=-1)
            key = F.normalize(self.edge_keys[direction](tile_features), dim=-1)
            logits = (query @ key.transpose(1, 2)) * scale
            rows.append(logits.masked_fill(diagonal, -1.0e4))
        return torch.stack(rows, dim=1)


def diffusion_loss(
    model: PositionalDDPM,
    tiles: Tensor,
    target_coordinates: Tensor,
    *,
    grid_weight: float = 0.0,
    edge_weight: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    batch = tiles.shape[0]
    time = torch.randint(model.schedule.steps, (batch,), device=tiles.device)
    noise = torch.randn_like(target_coordinates)
    noisy = model.schedule.q_sample(target_coordinates, time, noise)
    tile_features = model.encode_tiles(tiles)
    predicted_noise = model.predict_noise_from_embeddings(tile_features, noisy, time)
    noise_loss = F.mse_loss(predicted_noise, noise)
    loss = noise_loss
    grid_loss = torch.zeros((), device=tiles.device)
    if grid_weight:
        predicted_clean = model.schedule.predict_clean(noisy, time, predicted_noise).clamp(-2.0, 2.0)
        slots = grid_coordinates(model.side, device=tiles.device)
        logits = -torch.cdist(predicted_clean.float(), slots[None].expand(batch, -1, -1)).square() / 0.05
        labels = torch.cdist(target_coordinates.float(), slots[None].expand(batch, -1, -1)).argmin(dim=-1)
        grid_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss = loss + grid_weight * grid_loss
    edge_loss = torch.zeros((), device=tiles.device)
    edge_r1 = torch.zeros((), device=tiles.device)
    if edge_weight:
        # target_coordinates are exact grid points.  Recover tile->slot ids,
        # then build exact tile-index targets for U/D/L/R.
        slots = grid_coordinates(model.side, device=tiles.device)
        true_slots = torch.cdist(
            target_coordinates.float(), slots[None].expand(batch, -1, -1)
        ).argmin(dim=-1)
        inverse = torch.empty_like(true_slots)
        tile_ids = torch.arange(model.n_tiles, device=tiles.device).expand_as(true_slots)
        inverse.scatter_(1, true_slots, tile_ids)
        row = torch.div(true_slots, model.side, rounding_mode="floor")
        col = true_slots.remainder(model.side)
        valid_by_direction = (
            row.gt(0), row.lt(model.side - 1), col.gt(0), col.lt(model.side - 1)
        )
        deltas = (-model.side, model.side, -1, 1)
        losses: list[Tensor] = []
        correct = torch.zeros((), device=tiles.device)
        total = torch.zeros((), device=tiles.device)
        all_edge_logits = model.directional_edge_scores(tile_features)
        for direction, (valid, delta) in enumerate(zip(valid_by_direction, deltas)):
            target_cell = (true_slots + delta).clamp(0, model.n_tiles - 1)
            target_tile = inverse.gather(1, target_cell)
            logits = all_edge_logits[:, direction]
            losses.append(F.cross_entropy(logits[valid], target_tile[valid]))
            correct = correct + logits[valid].argmax(dim=-1).eq(target_tile[valid]).float().sum()
            total = total + valid.sum()
        edge_loss = torch.stack(losses).mean()
        edge_r1 = correct / total.clamp_min(1)
        loss = loss + edge_weight * edge_loss
    return loss, {
        "loss": float(loss.detach()),
        "noise_loss": float(noise_loss.detach()),
        "grid_loss": float(grid_loss.detach()),
        "edge_loss": float(edge_loss.detach()),
        "edge_r1": float(edge_r1.detach()),
        "mean_t": float(time.float().mean()),
    }


@torch.inference_mode()
def ddim_sample(
    model: PositionalDDPM,
    tiles: Tensor,
    *,
    sample_steps: int = 50,
    seed: int = 0,
) -> Tensor:
    """Deterministic eta=0 DDIM sampling on a strided training schedule."""
    if not 1 <= sample_steps <= model.schedule.steps:
        raise ValueError("sample_steps must be within the training schedule")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=tiles.device).manual_seed(seed)
    coordinates = torch.randn(
        tiles.shape[0], model.n_tiles, 2, device=tiles.device, generator=generator
    )
    tile_features = model.encode_tiles(tiles)
    times = torch.linspace(model.schedule.steps - 1, 0, sample_steps, device=tiles.device).round().long()
    times = torch.unique_consecutive(times)
    for index, scalar_time in enumerate(times):
        time = scalar_time.expand(tiles.shape[0])
        predicted_noise = model.predict_noise_from_embeddings(tile_features, coordinates, time)
        predicted_clean = model.schedule.predict_clean(coordinates, time, predicted_noise).clamp(-2.5, 2.5)
        if index + 1 == len(times):
            coordinates = predicted_clean
            break
        next_time = times[index + 1]
        next_alpha_bar = model.schedule.alpha_bar[next_time]
        coordinates = next_alpha_bar.sqrt() * predicted_clean + (1.0 - next_alpha_bar).sqrt() * predicted_noise
    if was_training:
        model.train()
    return coordinates


def hungarian_slots(coordinates: Tensor, side: int) -> Tensor:
    slots = grid_coordinates(side, device=coordinates.device)
    costs = torch.cdist(coordinates.float(), slots).cpu().numpy()
    results: list[np.ndarray] = []
    for cost in costs:
        rows, cols = linear_sum_assignment(cost)
        assignment = np.empty(side * side, dtype=np.int64)
        assignment[rows] = cols
        results.append(assignment)
    return torch.from_numpy(np.stack(results)).to(coordinates.device)


def arrangement_metrics(predicted_slots: Tensor, true_slots: Tensor, side: int) -> dict[str, float]:
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
                    a, b = int(truth[slot_to_token[slot]]), int(truth[slot_to_token[slot + 1]])
                    recovered += int(abs(a - b) == 1 and a // side == b // side)
                if row + 1 < side:
                    a, b = int(truth[slot_to_token[slot]]), int(truth[slot_to_token[slot + side]])
                    recovered += int(abs(a - b) == side)
        neighbor_scores.append(recovered / total)
    return {
        "placement_accuracy": float(placement),
        "neighbor_accuracy": float(np.mean(neighbor_scores)),
        "perfect_puzzles": float(predicted_slots.eq(true_slots).all(dim=1).float().mean()),
    }


def load_paired_dirty_encoder(model: PositionalDDPM, checkpoint: str) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"]
    prefix = "dirty_encoder.features."
    dirty_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    model.tile_backbone.load_state_dict(dirty_state, strict=True)
    return {"checkpoint": checkpoint, "source_step": int(payload.get("step", -1))}


def smoke_test(device: torch.device = torch.device("cpu")) -> dict[str, float]:
    torch.manual_seed(19)
    model = PositionalDDPM(side=4, tile_dim=32, d_model=48, layers=2, heads=4, diffusion_steps=20).to(device)
    tiles = torch.rand(2, 16, 3, FS, FS, device=device)
    permutation = torch.randperm(16, device=device)
    target = grid_coordinates(4, device=device)[permutation][None].expand(2, -1, -1)
    loss, parts = diffusion_loss(model, tiles, target, grid_weight=0.01, edge_weight=0.01)
    loss.backward()
    coordinates = torch.randn(2, 16, 2, device=device)
    time = torch.tensor([3, 11], device=device)
    model.eval()
    with torch.no_grad():
        direct = model(tiles, coordinates, time)
        permuted = model(tiles[:, permutation], coordinates[:, permutation], time)
    error = (direct[:, permutation] - permuted).abs().max()
    if error > 3.0e-5:
        raise AssertionError(f"permutation equivariance failed: {float(error)}")
    decoded = hungarian_slots(target, 4)
    truth = torch.cdist(target, grid_coordinates(4, device=device)[None].expand(2, -1, -1)).argmin(-1)
    perfect = arrangement_metrics(decoded, truth, 4)
    if perfect["placement_accuracy"] != 1.0 or perfect["neighbor_accuracy"] != 1.0:
        raise AssertionError("decoder metric perfect case failed")
    return {**parts, "equivariance_max_abs": float(error), **perfect}
