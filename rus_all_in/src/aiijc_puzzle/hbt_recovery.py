"""Exact recovery port of the historical HBT side-embedding matcher.

The architecture and full-board hard-triplet loss are ported from read-only
``origin/таска-говно`` at commit d6a82f82ceefa109ef706402712d03805bc9e880,
blob ``source/src/puzzle_assembly/learned.py``
``fa6209701c06667526bc609158874df96618dc47``.  Only package/protocol glue is
new.  Inference consumes dirty tiles only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import log_softmax
from torch import nn

from aiijc_puzzle.protocol import GRID_SIZE, TILE_COUNT, TILE_SIZE, split_tiles
from aiijc_puzzle.restoration_r6 import distort_tiles

HISTORICAL_COMMIT = "d6a82f82ceefa109ef706402712d03805bc9e880"
HISTORICAL_LEARNED_BLOB = "fa6209701c06667526bc609158874df96618dc47"
HISTORICAL_TRAIN_BLOB = "8277cb961e9bedd8e41e0b0bade2615f757b6db5"


@dataclass(frozen=True)
class DirectionLabels:
    right_queries: np.ndarray
    right_targets: np.ndarray
    down_queries: np.ndarray
    down_targets: np.ndarray
    outside: np.ndarray


@dataclass(frozen=True)
class SyntheticPanel:
    slot_tiles: np.ndarray
    slot_to_target: np.ndarray
    labels: DirectionLabels
    seed: int


def validate_permutation(value: np.ndarray, *, name: str = "permutation") -> np.ndarray:
    result = np.asarray(value, dtype=np.int32)
    if result.shape != (TILE_COUNT,) or not np.array_equal(np.sort(result), np.arange(TILE_COUNT)):
        raise ValueError(f"{name} must be a permutation of 0..{TILE_COUNT - 1}")
    return result


def direction_labels(slot_to_target: np.ndarray) -> DirectionLabels:
    """Port of the historical exact-neighbour label construction."""

    slot_to_target = validate_permutation(slot_to_target, name="slot_to_target")
    position_to_slot = np.empty(TILE_COUNT, dtype=np.int32)
    position_to_slot[slot_to_target] = np.arange(TILE_COUNT, dtype=np.int32)
    right_positions = np.asarray(
        [position for position in range(TILE_COUNT) if position % GRID_SIZE < GRID_SIZE - 1],
        dtype=np.int32,
    )
    down_positions = np.arange(TILE_COUNT - GRID_SIZE, dtype=np.int32)
    outside = np.zeros((TILE_COUNT, 4), dtype=np.float32)
    for slot, position in enumerate(slot_to_target.tolist()):
        row, column = divmod(position, GRID_SIZE)
        outside[slot] = (
            column == 0,
            column == GRID_SIZE - 1,
            row == 0,
            row == GRID_SIZE - 1,
        )
    return DirectionLabels(
        right_queries=position_to_slot[right_positions],
        right_targets=position_to_slot[right_positions + 1],
        down_queries=position_to_slot[down_positions],
        down_targets=position_to_slot[down_positions + GRID_SIZE],
        outside=outside,
    )


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(values)
        values = F.silu(self.norm(self.pointwise(values)))
        return values + residual


class SideEmbeddingNet(nn.Module):
    """Historical pooled four-side query/key encoder, unchanged mathematically."""

    def __init__(
        self,
        *,
        channels: int = 64,
        embedding_dim: int = 96,
        side_band: int = 4,
        tangent_bins: int = 10,
        temperature: float = 0.07,
        input_mode: str = "rgb_norm",
        edge_threshold: float = 0.12,
    ) -> None:
        super().__init__()
        if channels <= 0 or embedding_dim <= 0:
            raise ValueError("channels and embedding_dim must be positive")
        if channels % 8 != 0:
            raise ValueError("channels must be divisible by 8 for GroupNorm")
        if not 1 <= side_band <= TILE_SIZE:
            raise ValueError("side_band must be in [1, 20]")
        if tangent_bins <= 0:
            raise ValueError("tangent_bins must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        input_channels = {
            "rgb_norm": 6,
            "rgb_sobel": 9,
            "sobel_only": 3,
            "binary_edges": 1,
        }
        if input_mode not in input_channels:
            raise ValueError(f"unsupported input_mode: {input_mode}")
        if not 0 < edge_threshold < 2:
            raise ValueError("edge_threshold must be in (0, 2)")
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.side_band = side_band
        self.tangent_bins = tangent_bins
        self.temperature = float(temperature)
        self.input_mode = input_mode
        self.edge_threshold = float(edge_threshold)
        self.register_buffer(
            "sobel_x",
            torch.tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]).unsqueeze(0)
            / 8.0,
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]).unsqueeze(0)
            / 8.0,
            persistent=False,
        )
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels[input_mode], channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            _ResidualBlock(channels),
            _ResidualBlock(channels),
        )
        feature_dim = channels * tangent_bins
        self.query_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embedding_dim),
        )
        self.key_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embedding_dim),
        )
        self.outside_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 1),
        )

    def _side_feature(self, features: torch.Tensor, side: str) -> torch.Tensor:
        band = self.side_band
        if side == "left":
            line = features[:, :, :, :band].mean(dim=3)
        elif side == "right":
            line = features[:, :, :, -band:].mean(dim=3)
        elif side == "up":
            line = features[:, :, :band, :].mean(dim=2)
        elif side == "down":
            line = features[:, :, -band:, :].mean(dim=2)
        else:
            raise ValueError(f"unknown side: {side}")
        line = F.adaptive_avg_pool1d(line, self.tangent_bins)
        return line.flatten(1)

    def _input_features(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=(2, 3), keepdim=True)
        std = values.std(dim=(2, 3), keepdim=True).clamp_min(1.0 / 255.0)
        normalized = ((values - mean) / std).clamp(-4.0, 4.0) / 4.0
        if self.input_mode == "rgb_norm":
            return torch.cat([values, normalized], dim=1)
        luma = 0.299 * values[:, 0:1] + 0.587 * values[:, 1:2] + 0.114 * values[:, 2:3]
        padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")
        gradient_x = F.conv2d(padded, self.sobel_x)
        gradient_y = F.conv2d(padded, self.sobel_y)
        magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-8).clamp_max(1.0)
        if self.input_mode == "rgb_sobel":
            return torch.cat([values, normalized, gradient_x, gradient_y, magnitude], dim=1)
        if self.input_mode == "sobel_only":
            return torch.cat([gradient_x, gradient_y, magnitude], dim=1)
        return (magnitude >= self.edge_threshold).to(values.dtype)

    def forward(self, tiles: torch.Tensor) -> dict[str, torch.Tensor]:
        if tiles.ndim != 4 or tiles.shape[1:] != (3, TILE_SIZE, TILE_SIZE):
            raise ValueError(f"expected NCHW tiles with shape (*,3,20,20), got {tiles.shape}")
        values = tiles.float()
        if values.detach().amax() > 1.5:
            values = values / 255.0
        features = self.stem(self._input_features(values))
        sides = {
            side: self._side_feature(features, side) for side in ("left", "right", "up", "down")
        }
        outside = torch.cat(
            [self.outside_head(sides[side]) for side in ("left", "right", "up", "down")],
            dim=1,
        )
        raw_q_right = self.query_projection(sides["right"])
        raw_k_left = self.key_projection(sides["left"])
        raw_q_down = self.query_projection(sides["down"])
        raw_k_up = self.key_projection(sides["up"])
        return {
            "q_right": F.normalize(raw_q_right, dim=1),
            "k_left": F.normalize(raw_k_left, dim=1),
            "q_down": F.normalize(raw_q_down, dim=1),
            "k_up": F.normalize(raw_k_up, dim=1),
            "raw_q_right": raw_q_right,
            "raw_k_left": raw_k_left,
            "raw_q_down": raw_q_down,
            "raw_k_up": raw_k_up,
            "outside_logits": outside,
        }

    def config(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "embedding_dim": self.embedding_dim,
            "side_band": self.side_band,
            "tangent_bins": self.tangent_bins,
            "temperature": self.temperature,
            "input_mode": self.input_mode,
            "edge_threshold": self.edge_threshold,
        }


def _masked_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    query_slots: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    selected_query = query[query_slots]
    logits = selected_query @ key.T
    logits = logits / temperature
    rows = torch.arange(len(query_slots), device=logits.device)
    logits[rows, query_slots] = torch.finfo(logits.dtype).min
    return logits


def embedding_hard_triplet_loss(
    outputs: dict[str, torch.Tensor],
    labels: DirectionLabels,
    *,
    temperature: float,
    margin: float = 0.2,
    cross_entropy_weight: float = 0.25,
    embedding_l2_weight: float = 1e-4,
    outside_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact historical hardest-negative loss over all 575 alternatives."""

    if margin <= 0:
        raise ValueError("margin must be positive")
    if cross_entropy_weight < 0 or embedding_l2_weight < 0 or outside_weight < 0:
        raise ValueError("loss weights must be non-negative")
    device = outputs["q_right"].device
    directional: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for query_name, key_name, queries_array, targets_array in (
        ("q_right", "k_left", labels.right_queries, labels.right_targets),
        ("q_down", "k_up", labels.down_queries, labels.down_targets),
    ):
        queries = torch.as_tensor(queries_array, device=device, dtype=torch.long)
        targets = torch.as_tensor(targets_array, device=device, dtype=torch.long)
        logits = _masked_logits(
            outputs[query_name], outputs[key_name], queries, temperature=temperature
        )
        rows = torch.arange(len(queries), device=device)
        positive = logits[rows, targets]
        negative_logits = logits.clone()
        negative_logits[rows, targets] = torch.finfo(logits.dtype).min
        hardest_negative = negative_logits.max(dim=1).values
        triplet = F.relu(margin + hardest_negative - positive).mean()
        cross_entropy = F.cross_entropy(logits, targets)
        top1 = (logits.argmax(dim=1) == targets).float().mean()
        directional.append((triplet, cross_entropy, top1))

    triplet_loss = 0.5 * (directional[0][0] + directional[1][0])
    cross_entropy_loss = 0.5 * (directional[0][1] + directional[1][1])
    raw_names = ("raw_q_right", "raw_k_left", "raw_q_down", "raw_k_up")
    embedding_l2 = torch.stack([outputs[name].square().mean() for name in raw_names]).mean()
    outside_targets = torch.as_tensor(labels.outside, device=device)
    outside_loss = F.binary_cross_entropy_with_logits(outputs["outside_logits"], outside_targets)
    loss = (
        triplet_loss
        + cross_entropy_weight * cross_entropy_loss
        + embedding_l2_weight * embedding_l2
        + outside_weight * outside_loss
    )
    recall_at_1 = 0.5 * (directional[0][2] + directional[1][2])
    return loss, {
        "loss": float(loss.detach().cpu()),
        "triplet_loss": float(triplet_loss.detach().cpu()),
        "cross_entropy_loss": float(cross_entropy_loss.detach().cpu()),
        "embedding_l2": float(embedding_l2.detach().cpu()),
        "outside_loss": float(outside_loss.detach().cpu()),
        "recall_at_1": float(recall_at_1.detach().cpu()),
    }


def make_synthetic_panel(clean_target: np.ndarray, *, seed: int) -> SyntheticPanel:
    """Build one exact organizer-style corruption replica and shuffled labels."""

    clean_target = np.asarray(clean_target)
    if clean_target.shape != (480, 480, 3) or clean_target.dtype != np.uint8:
        raise ValueError("clean target must be uint8 RGB 480x480")
    clean_tiles = split_tiles(clean_target)
    rng = np.random.default_rng(seed)
    slot_to_target = rng.permutation(TILE_COUNT).astype(np.int32)
    corrupted = distort_tiles(clean_tiles, rng)
    slot_tiles = np.ascontiguousarray(corrupted[slot_to_target])
    return SyntheticPanel(
        slot_tiles=slot_tiles,
        slot_to_target=slot_to_target,
        labels=direction_labels(slot_to_target),
        seed=seed,
    )


def view_tiles(tiles: np.ndarray, *, view: str) -> np.ndarray:
    """Apply one inference-visible model view without changing tile order."""

    source = np.asarray(tiles, dtype=np.uint8)
    if source.shape != (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3):
        raise ValueError(f"unexpected tile shape {source.shape}")
    if view == "raw":
        return source
    if view == "bilateral":
        return np.stack([cv2.bilateralFilter(tile, 5, 25, 5) for tile in source])
    raise ValueError(f"unknown HBT view {view!r}")


def tiles_tensor(tiles: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    )


@torch.inference_mode()
def dense_scores(
    model: SideEmbeddingNet,
    tiles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-log-prob compatibility and outside logits from dirty tiles."""

    model.eval()
    outputs = model(tiles_tensor(tiles, device))
    right = (outputs["q_right"] @ outputs["k_left"].T).float().cpu().numpy()
    down = (outputs["q_down"] @ outputs["k_up"].T).float().cpu().numpy()
    np.fill_diagonal(right, -1e4)
    np.fill_diagonal(down, -1e4)
    return (
        log_softmax(right / model.temperature, axis=1).astype(np.float32),
        log_softmax(down / model.temperature, axis=1).astype(np.float32),
        outputs["outside_logits"].float().cpu().numpy(),
    )


def exact_retrieval_counts(
    right: np.ndarray,
    down: np.ndarray,
    labels: DirectionLabels,
    *,
    ks: tuple[int, ...] = (1, 5, 32),
) -> list[dict[str, int | str]]:
    """Compute exact counts from target-assisted labels after score freeze."""

    records: list[dict[str, int | str]] = []
    for name, matrix, queries, targets in (
        ("right", right, labels.right_queries, labels.right_targets),
        ("down", down, labels.down_queries, labels.down_targets),
    ):
        for k in ks:
            order = np.argpartition(matrix[queries], -k, axis=1)[:, -k:]
            hits = int(np.sum(np.any(order == targets[:, None], axis=1)))
            records.append({"direction": name, "k": k, "edges": len(queries), "hits": hits})
    return records
