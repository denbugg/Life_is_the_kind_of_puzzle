"""Dense learned residuals for all directional tile pairs.

The classical compatibility matrix remains the anchor.  Every tile is encoded
exactly once, then a small nonlinear relation head scores every ordered pair in
right/down chunks.  Unlike proposal-only re-rankers, this module has no top-k
ceiling: all ``N * N`` pairs receive a learned residual.

The final relation layer is initialized to exactly zero.  Consequently a fresh
model returns the supplied base compatibility unchanged (apart from enforcing
the forbidden self diagonal).  Learned changes are bounded by a trainable gain
whose magnitude can never exceed ``max_residual``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .compatibility import CompatibilityMatrices
from .geometry import TILE, TILE_COUNT


RIGHT = 0
DOWN = 1

LEFT_SIDE = 0
RIGHT_SIDE = 1
TOP_SIDE = 2
BOTTOM_SIDE = 3


@dataclass(frozen=True)
class EncodedDenseTileBank:
    """Features cached once for dense all-pairs scoring.

    ``side_embeddings`` retains one full-resolution-profile embedding for each
    physical side in ``left, right, top, bottom`` order.  Endpoint embeddings
    already fuse the side profile with whole-tile context, so scoring a pair
    does not repeat any convolutional work.
    """

    global_embeddings: torch.Tensor
    side_embeddings: torch.Tensor
    endpoint_embeddings: torch.Tensor

    def __post_init__(self) -> None:
        if self.global_embeddings.ndim != 2:
            raise ValueError("global_embeddings must have shape (N,D)")
        expected = (self.global_embeddings.shape[0], 4, self.global_embeddings.shape[1])
        if self.side_embeddings.shape != expected:
            raise ValueError(f"side_embeddings must have shape {expected}")
        if self.endpoint_embeddings.shape != expected:
            raise ValueError(f"endpoint_embeddings must have shape {expected}")
        devices = {
            self.global_embeddings.device,
            self.side_embeddings.device,
            self.endpoint_embeddings.device,
        }
        if len(devices) != 1:
            raise ValueError("encoded tile-bank tensors must share one device")

    def __len__(self) -> int:
        return int(self.global_embeddings.shape[0])


class _LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.permute(0, 2, 3, 1)
        values = self.norm(values)
        return values.permute(0, 3, 1, 2)


class _ConvNeXtBlock(nn.Module):
    """A full-resolution ConvNeXt-style residual block."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int,
        dropout: float,
        layer_scale_init: float = 1.0e-6,
    ) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=True,
        )
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, hidden)
        self.contract = nn.Linear(hidden, channels)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(
            torch.full((channels,), float(layer_scale_init))
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(values).permute(0, 2, 3, 1)
        values = self.norm(values)
        values = self.expand(values)
        values = F.gelu(values)
        values = self.dropout(values)
        values = self.contract(values)
        values = values * self.layer_scale
        values = self.dropout(values).permute(0, 3, 1, 2)
        return residual + values


class DensePairResidualScorer(nn.Module):
    """Encode tiles once and score every directional ordered pair.

    Inputs are raw RGB tiles and an optional restored RGB view.  When the
    restored view is absent, it is replaced with the raw view and an explicit
    availability channel is set to zero.  This permits view-dropout training
    and raw-only inference without changing the network shape.

    Defaults are intentionally a bounded 2--4M parameter pilot.  ``tile_size``
    and every width are configurable so unit tests can exercise the complete
    dense path on a small CPU grid.
    """

    def __init__(
        self,
        *,
        tile_size: int = TILE,
        encoder_width: int = 160,
        encoder_depth: int = 8,
        expansion: int = 4,
        side_band: int = 4,
        profile_bins: int = 10,
        embedding_dim: int = 192,
        relation_hidden: int = 384,
        pair_hidden: int = 192,
        dropout: float = 0.05,
        max_residual: float = 0.25,
        initial_gain_fraction: float = 0.5,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if tile_size < 2:
            raise ValueError("tile_size must be at least 2")
        if encoder_width < 4 or encoder_depth < 1 or expansion < 1:
            raise ValueError("encoder dimensions must be positive")
        if not 1 <= side_band <= tile_size:
            raise ValueError("side_band must lie in [1, tile_size]")
        if not 1 <= profile_bins <= tile_size:
            raise ValueError("profile_bins must lie in [1, tile_size]")
        if min(embedding_dim, relation_hidden, pair_hidden) < 2:
            raise ValueError("embedding and relation dimensions must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not math.isfinite(max_residual) or max_residual <= 0.0:
            raise ValueError("max_residual must be finite and positive")
        if not 0.0 < initial_gain_fraction < 1.0:
            raise ValueError("initial_gain_fraction must lie strictly in (0, 1)")

        self.tile_size = int(tile_size)
        self.encoder_width = int(encoder_width)
        self.encoder_depth = int(encoder_depth)
        self.expansion = int(expansion)
        self.side_band = int(side_band)
        self.profile_bins = int(profile_bins)
        self.embedding_dim = int(embedding_dim)
        self.relation_hidden = int(relation_hidden)
        self.pair_hidden = int(pair_hidden)
        self.dropout = float(dropout)
        self.max_residual = float(max_residual)
        self.initial_gain_fraction = float(initial_gain_fraction)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # raw RGB, optional restored RGB, restored-minus-raw RGB, availability
        input_channels = 10
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, encoder_width, 3, padding=1, bias=True),
            _LayerNorm2d(encoder_width),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            *[
                _ConvNeXtBlock(
                    encoder_width,
                    expansion=expansion,
                    dropout=dropout,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = _LayerNorm2d(encoder_width)

        self.global_projection = nn.Sequential(
            nn.LayerNorm(encoder_width),
            nn.Linear(encoder_width, embedding_dim),
            nn.GELU(),
        )
        self.side_projection = nn.Sequential(
            nn.LayerNorm(encoder_width * profile_bins),
            nn.Linear(encoder_width * profile_bins, embedding_dim),
            nn.GELU(),
        )
        self.endpoint_projection = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.GELU(),
        )

        self.endpoint_role = nn.Parameter(torch.zeros(2, embedding_dim))
        self.direction_embedding = nn.Parameter(torch.zeros(2, embedding_dim))

        relation_dim = 5 * embedding_dim
        self.relation_head = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, relation_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(relation_hidden, pair_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, 1),
        )
        gain_logit = math.log(initial_gain_fraction / (1.0 - initial_gain_fraction))
        self.gain_logit = nn.Parameter(torch.tensor(gain_logit, dtype=torch.float32))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.endpoint_role, std=0.02)
        nn.init.trunc_normal_(self.direction_embedding, std=0.02)
        # This is the safety contract: a fresh model cannot alter the base.
        final = self.relation_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def bounded_gain(self) -> torch.Tensor:
        """Trainable positive gain, strictly bounded by ``max_residual``."""

        return self.gain_logit.sigmoid() * self.max_residual

    @staticmethod
    def apply_residual(base_cost: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Canonical train/inference combination for lower-is-better costs."""

        if base_cost.shape != residual.shape:
            raise ValueError("base_cost and residual must have identical shapes")
        return base_cost + residual

    def config(self) -> dict[str, Any]:
        return {
            "tile_size": self.tile_size,
            "encoder_width": self.encoder_width,
            "encoder_depth": self.encoder_depth,
            "expansion": self.expansion,
            "side_band": self.side_band,
            "profile_bins": self.profile_bins,
            "embedding_dim": self.embedding_dim,
            "relation_hidden": self.relation_hidden,
            "pair_hidden": self.pair_hidden,
            "dropout": self.dropout,
            "max_residual": self.max_residual,
            "initial_gain_fraction": self.initial_gain_fraction,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    def _validate_views(
        self,
        raw_tiles: torch.Tensor,
        denoised_tiles: torch.Tensor | None,
    ) -> None:
        expected_tail = (3, self.tile_size, self.tile_size)
        if raw_tiles.ndim != 4 or tuple(raw_tiles.shape[1:]) != expected_tail:
            raise ValueError(f"raw_tiles must have shape (N,{expected_tail})")
        if len(raw_tiles) < 1:
            raise ValueError("at least one tile is required")
        if denoised_tiles is not None:
            if denoised_tiles.shape != raw_tiles.shape:
                raise ValueError("denoised_tiles must match raw_tiles")
            if denoised_tiles.device != raw_tiles.device:
                raise ValueError("raw and denoised tiles must share one device")

    @staticmethod
    def _unit_range(values: torch.Tensor) -> torch.Tensor:
        original_dtype = values.dtype
        values = values.float()
        if not (original_dtype.is_floating_point or original_dtype.is_complex):
            values = values / 255.0
        elif values.numel() and bool(values.detach().amax() > 1.5):
            values = values / 255.0
        return values.clamp(0.0, 1.0)

    def _input_features(
        self,
        raw_tiles: torch.Tensor,
        denoised_tiles: torch.Tensor | None,
    ) -> torch.Tensor:
        self._validate_views(raw_tiles, denoised_tiles)
        raw = self._unit_range(raw_tiles)
        if denoised_tiles is None:
            denoised = raw
            availability = raw.new_zeros((len(raw), 1, self.tile_size, self.tile_size))
        else:
            denoised = self._unit_range(denoised_tiles)
            availability = raw.new_ones((len(raw), 1, self.tile_size, self.tile_size))
        return torch.cat([raw, denoised, denoised - raw, availability], dim=1)

    def _side_profiles(self, features: torch.Tensor) -> torch.Tensor:
        band = self.side_band
        # Horizontal sides retain top-to-bottom order; vertical sides retain
        # left-to-right order.  The CNN itself has never been downsampled.
        left = features[:, :, :, :band].mean(dim=3)
        right = features[:, :, :, -band:].mean(dim=3)
        top = features[:, :, :band, :].mean(dim=2)
        bottom = features[:, :, -band:, :].mean(dim=2)
        profiles = torch.stack([left, right, top, bottom], dim=1)
        profiles = profiles.reshape(-1, self.encoder_width, self.tile_size)
        profiles = F.adaptive_avg_pool1d(profiles, self.profile_bins)
        profiles = profiles.reshape(len(features), 4, -1)
        return self.side_projection(profiles)

    def encode_tiles(
        self,
        raw_tiles: torch.Tensor,
        denoised_tiles: torch.Tensor | None = None,
    ) -> EncodedDenseTileBank:
        """Run the full-resolution encoder exactly once for a tile bank."""

        features = self.stem(self._input_features(raw_tiles, denoised_tiles))
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            for block in self.encoder:
                features = checkpoint(block, features, use_reentrant=False)
        else:
            features = self.encoder(features)
        features = self.encoder_norm(features)
        global_embeddings = self.global_projection(features.mean(dim=(2, 3)))
        side_embeddings = self._side_profiles(features)
        expanded_global = global_embeddings[:, None, :].expand(-1, 4, -1)
        endpoint_embeddings = self.endpoint_projection(
            torch.cat([side_embeddings, expanded_global], dim=-1)
        )
        return EncodedDenseTileBank(
            global_embeddings=global_embeddings,
            side_embeddings=side_embeddings,
            endpoint_embeddings=endpoint_embeddings,
        )

    @staticmethod
    def _validate_pair_indices(
        bank: EncodedDenseTileBank,
        first: torch.Tensor,
        second: torch.Tensor,
        direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if first.ndim != 1 or second.shape != first.shape or direction.shape != first.shape:
            raise ValueError("first, second, and direction must be equally sized vectors")
        device = bank.endpoint_embeddings.device
        first = first.to(device=device, dtype=torch.long)
        second = second.to(device=device, dtype=torch.long)
        direction = direction.to(device=device, dtype=torch.long)
        if len(first):
            if bool(torch.any((first < 0) | (first >= len(bank)))):
                raise ValueError("first index is outside the tile bank")
            if bool(torch.any((second < 0) | (second >= len(bank)))):
                raise ValueError("second index is outside the tile bank")
            if bool(torch.any((direction < RIGHT) | (direction > DOWN))):
                raise ValueError("direction must contain only RIGHT/DOWN")
        return first, second, direction

    def _residual_from_endpoints(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        if first.shape != second.shape or first.shape[-1] != self.embedding_dim:
            raise ValueError("endpoint feature shapes differ")
        relations = torch.cat(
            [first, second, first - second, (first - second).abs(), first * second],
            dim=-1,
        )
        unit_residual = torch.tanh(self.relation_head(relations).squeeze(-1))
        return unit_residual * self.bounded_gain

    def forward_from_encoded(
        self,
        bank: EncodedDenseTileBank,
        first: torch.Tensor,
        second: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """Score selected ordered pairs from a cached bank (lower is better)."""

        first, second, direction = self._validate_pair_indices(
            bank, first, second, direction
        )
        outgoing_side = direction.new_tensor([RIGHT_SIDE, BOTTOM_SIDE])[direction]
        incoming_side = direction.new_tensor([LEFT_SIDE, TOP_SIDE])[direction]
        direction_features = self.direction_embedding[direction]
        first_features = (
            bank.endpoint_embeddings[first, outgoing_side]
            + self.endpoint_role[0]
            + direction_features
        )
        second_features = (
            bank.endpoint_embeddings[second, incoming_side]
            + self.endpoint_role[1]
            + direction_features
        )
        return self._residual_from_endpoints(first_features, second_features)

    def forward(
        self,
        raw_tiles: torch.Tensor,
        denoised_tiles: torch.Tensor | None,
        first: torch.Tensor,
        second: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """DDP-safe training path: encode one bank, then score selected pairs."""

        bank = self.encode_tiles(raw_tiles, denoised_tiles)
        return self.forward_from_encoded(bank, first, second, direction)

    def score_dense(
        self,
        bank: EncodedDenseTileBank,
        direction: int,
        *,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """Score all ``N * N`` ordered pairs without a proposal/top-k ceiling."""

        if direction not in (RIGHT, DOWN):
            raise ValueError("direction must be RIGHT or DOWN")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        outgoing_side, incoming_side = (
            (RIGHT_SIDE, LEFT_SIDE) if direction == RIGHT else (BOTTOM_SIDE, TOP_SIDE)
        )
        direction_feature = self.direction_embedding[direction]
        outgoing = (
            bank.endpoint_embeddings[:, outgoing_side]
            + self.endpoint_role[0]
            + direction_feature
        )
        incoming = (
            bank.endpoint_embeddings[:, incoming_side]
            + self.endpoint_role[1]
            + direction_feature
        )
        rows: list[torch.Tensor] = []
        count = len(bank)
        for start in range(0, count, chunk_size):
            first = outgoing[start : start + chunk_size]
            row_count = len(first)
            first = first[:, None, :].expand(row_count, count, -1)
            second = incoming[None, :, :].expand(row_count, count, -1)
            rows.append(self._residual_from_endpoints(first, second))
        return torch.cat(rows, dim=0)


def _validate_numpy_tiles(
    values: np.ndarray,
    *,
    model: DensePairResidualScorer,
    name: str,
) -> np.ndarray:
    values = np.asarray(values)
    expected = (TILE_COUNT, model.tile_size, model.tile_size, 3)
    if values.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
    if values.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8, got {values.dtype}")
    return values


@torch.inference_mode()
def dense_pair_residual_compatibility(
    model: DensePairResidualScorer,
    raw_tiles: np.ndarray,
    base: CompatibilityMatrices,
    *,
    denoised_tiles: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    chunk_size: int = 64,
    name: str = "dense_pair_residual",
    telemetry: dict[str, Any] | None = None,
) -> CompatibilityMatrices:
    """Apply dense bounded residuals to every base right/down cost.

    The returned object is directly consumable by existing solvers.  Both
    matrices are fully scored and their diagonals are always ``+inf``.
    """

    raw = _validate_numpy_tiles(raw_tiles, model=model, name="raw_tiles")
    denoised = None
    if denoised_tiles is not None:
        denoised = _validate_numpy_tiles(
            denoised_tiles, model=model, name="denoised_tiles"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    resolved_device = torch.device(device)
    model.to(resolved_device)
    was_training = model.training
    model.eval()
    raw_tensor = torch.from_numpy(
        np.ascontiguousarray(raw.transpose(0, 3, 1, 2))
    ).to(resolved_device)
    denoised_tensor = None
    if denoised is not None:
        denoised_tensor = torch.from_numpy(
            np.ascontiguousarray(denoised.transpose(0, 3, 1, 2))
        ).to(resolved_device)

    amp_enabled = resolved_device.type == "cuda"
    with torch.autocast(
        device_type=resolved_device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        bank = model.encode_tiles(raw_tensor, denoised_tensor)
        right_residual = model.score_dense(bank, RIGHT, chunk_size=chunk_size)
        down_residual = model.score_dense(bank, DOWN, chunk_size=chunk_size)
        right_base = torch.as_tensor(
            np.asarray(base.right, dtype=np.float32), device=resolved_device
        )
        down_base = torch.as_tensor(
            np.asarray(base.down, dtype=np.float32), device=resolved_device
        )
        right = model.apply_residual(right_base, right_residual).float()
        down = model.apply_residual(down_base, down_residual).float()
        diagonal = torch.arange(TILE_COUNT, device=resolved_device)
        right[diagonal, diagonal] = torch.inf
        down[diagonal, diagonal] = torch.inf

    result = CompatibilityMatrices(
        name,
        right.cpu().numpy().astype(np.float32, copy=False),
        down.cpu().numpy().astype(np.float32, copy=False),
    )
    if was_training:
        model.train()
    if telemetry is not None:
        telemetry.update(
            {
                "tile_count": TILE_COUNT,
                "tile_encoder_passes": 1,
                "scored_pairs_per_direction": TILE_COUNT * TILE_COUNT,
                "proposal_top_k": None,
                "chunk_size": int(chunk_size),
                "denoised_view": denoised is not None,
                "bounded_gain": float(model.bounded_gain.detach().cpu()),
            }
        )
    return result


def save_dense_pair_residual_checkpoint(
    path: str | Path,
    model: DensePairResidualScorer,
    *,
    metadata: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Atomically save model configuration, weights, and optional train state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = dict(metadata or {})
    payload_metadata["safe_for_submission"] = False
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "puzzle_dense_pair_residual",
        "safe_for_submission": False,
        "model_config": model.config(),
        "model_state": model.state_dict(),
        "metadata": payload_metadata,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if training_state is not None:
        payload["training_state"] = training_state
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_dense_pair_residual_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("dense-pair checkpoint must contain a dictionary")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "puzzle_dense_pair_residual"
    ):
        raise ValueError("unsupported dense-pair checkpoint")
    if payload.get("safe_for_submission") is not False:
        raise ValueError("dense-pair research checkpoint must be fail-closed")
    config = payload.get("model_config")
    state = payload.get("model_state")
    metadata = payload.get("metadata")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("dense-pair checkpoint is incomplete")
    if not isinstance(metadata, dict) or metadata.get("safe_for_submission") is not False:
        raise ValueError("dense-pair checkpoint metadata is not fail-closed")
    try:
        validation_model = DensePairResidualScorer(**config)
        validation_model.load_state_dict(state, strict=True)
    except Exception as error:
        raise ValueError("dense-pair model payload is not strictly loadable") from error
    return payload


def load_dense_pair_residual_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[DensePairResidualScorer, dict[str, Any]]:
    payload = load_dense_pair_residual_checkpoint_payload(path)
    model = DensePairResidualScorer(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload["metadata"])


__all__ = [
    "DOWN",
    "RIGHT",
    "DensePairResidualScorer",
    "EncodedDenseTileBank",
    "dense_pair_residual_compatibility",
    "load_dense_pair_residual_checkpoint",
    "load_dense_pair_residual_checkpoint_payload",
    "save_dense_pair_residual_checkpoint",
]
