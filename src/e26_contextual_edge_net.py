"""Permutation-equivariant contextual edge network for the upright E26 puzzle.

The network consumes an unordered bag of RGB tiles.  Every tile is encoded from
both its raw pixels and an independently normalized copy, so the local boundary
branch can retain colour while remaining robust to the per-tile photometric
distortion.  Four self-attention blocks then provide image-level context without
ever introducing a tile position or input-order embedding.

Only right and down pair matrices are predicted.  Left and up are their exact
transposes, respectively, which makes the inverse-edge contract true by
construction instead of merely encouraging it with a loss.  Each directed row
also receives an explicit NONE class for border tiles.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
DIRECTION_COUNT = 4
DIRECTION_NAMES = ("up", "down", "left", "right")
CHECKPOINT_SCHEMA = "pazzle-e26-contextual-edge-net-v1"


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(channels), maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(frozen=True)
class ContextualEdgeConfig:
    """All architecture choices needed to reproduce an E26 model.

    The defaults keep a full 576-tile forward pass practical on an 8 GB GPU.
    ``encoder_chunk_size`` only chunks the independent CNN; all 576 contextual
    tokens still meet in every set-attention layer.
    """

    grid_height: int = 24
    grid_width: int = 24
    input_channels: int = 3
    cnn_width: int = 48
    d_model: int = 128
    local_dim: int = 96
    match_dim: int = 64
    transformer_layers: int = 4
    attention_heads: int = 4
    ff_multiplier: float = 2.0
    dropout: float = 0.05
    boundary_band: int = 2
    boundary_bins: int = 5
    reconstruction_samples: int = 10
    encoder_chunk_size: int = 144
    normalization_eps: float = 1.0e-5
    initial_logit_scale: float = 10.0
    diagonal_mask_value: float = -10_000.0

    @property
    def tile_count(self) -> int:
        return self.grid_height * self.grid_width

    def validate(self) -> None:
        integer_positive = {
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "input_channels": self.input_channels,
            "cnn_width": self.cnn_width,
            "d_model": self.d_model,
            "local_dim": self.local_dim,
            "match_dim": self.match_dim,
            "transformer_layers": self.transformer_layers,
            "attention_heads": self.attention_heads,
            "boundary_band": self.boundary_band,
            "boundary_bins": self.boundary_bins,
            "reconstruction_samples": self.reconstruction_samples,
            "encoder_chunk_size": self.encoder_chunk_size,
        }
        bad = [name for name, value in integer_positive.items() if int(value) <= 0]
        if bad:
            raise ValueError(f"positive integer configuration required for: {', '.join(bad)}")
        if self.input_channels != 3:
            raise ValueError("E26 currently requires RGB input_channels=3")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.ff_multiplier < 1.0:
            raise ValueError("ff_multiplier must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.normalization_eps <= 0.0 or self.initial_logit_scale <= 0.0:
            raise ValueError("normalization_eps and initial_logit_scale must be positive")
        if not math.isfinite(self.diagonal_mask_value) or self.diagonal_mask_value >= 0.0:
            raise ValueError("diagonal_mask_value must be a finite negative number")


class _ResidualConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(values + self.body(values))


class BoundaryAwareTileEncoder(nn.Module):
    """Encode local U/D/L/R traces and one global token per tile."""

    def __init__(self, config: ContextualEdgeConfig) -> None:
        super().__init__()
        self.config = config
        width = config.cnn_width
        middle = max(width, config.d_model // 2)
        # Six channels are exactly raw RGB plus independently normalized RGB.
        self.stem = nn.Sequential(
            nn.Conv2d(2 * config.input_channels, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            _ResidualConv(width),
        )
        self.global_body = nn.Sequential(
            nn.Conv2d(width, middle, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
            _ResidualConv(middle),
            nn.Conv2d(middle, config.d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(config.d_model), config.d_model),
            nn.GELU(),
            _ResidualConv(config.d_model),
        )
        side_input = width * config.boundary_bins
        self.side_projection = nn.Sequential(
            nn.LayerNorm(side_input),
            nn.Linear(side_input, config.local_dim),
            nn.GELU(),
            nn.LayerNorm(config.local_dim),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.side_summary = nn.Linear(config.local_dim, config.d_model, bias=False)

    def _six_channel_input(self, tiles: Tensor) -> Tensor:
        mean = tiles.mean(dim=(-2, -1), keepdim=True)
        variance = tiles.var(dim=(-2, -1), keepdim=True, unbiased=False)
        normalized = (tiles - mean) * torch.rsqrt(variance + self.config.normalization_eps)
        return torch.cat((tiles, normalized), dim=1)

    def _side_traces(self, features: Tensor) -> Tensor:
        _, _, height, width = features.shape
        band = self.config.boundary_band
        if height < band or width < band:
            raise ValueError(
                f"boundary_band={band} exceeds the encoded tile shape {(height, width)}"
            )
        # Tangential order is retained: left-to-right for U/D and top-to-bottom
        # for L/R.  Upright tiles therefore never require reversal or rotation.
        top = features[:, :, :band, :].mean(dim=2)
        bottom = features[:, :, -band:, :].mean(dim=2)
        left = features[:, :, :, :band].mean(dim=3)
        right = features[:, :, :, -band:].mean(dim=3)
        traces = []
        for trace in (top, bottom, left, right):
            pooled = F.adaptive_avg_pool1d(trace, self.config.boundary_bins)
            traces.append(pooled.flatten(1))
        return torch.stack(traces, dim=1)

    def forward(self, tiles: Tensor) -> tuple[Tensor, Tensor]:
        if tiles.ndim != 4 or tiles.shape[1] != self.config.input_channels:
            raise ValueError(
                "flat tiles must have shape (tiles, 3, height, width), "
                f"got {tuple(tiles.shape)}"
            )
        features = self.stem(self._six_channel_input(tiles))
        local_sides = self.side_projection(self._side_traces(features))
        body = self.global_body(features)
        global_token = self.global_projection(F.adaptive_avg_pool2d(body, 1).flatten(1))
        global_token = global_token + self.side_summary(local_sides.mean(dim=1))
        return global_token, local_sides


class _SetAttentionBlock(nn.Module):
    """Pre-norm set self-attention without positional information."""

    def __init__(self, config: ContextualEdgeConfig) -> None:
        super().__init__()
        hidden = max(config.d_model, int(round(config.d_model * config.ff_multiplier)))
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        values = self.norm1(tokens)
        tokens = tokens + self.attention(values, values, values, need_weights=False)[0]
        return tokens + self.feed_forward(self.norm2(tokens))


def pack_directional_logits(pair_logits: Tensor, none_logits: Tensor) -> Tensor:
    """Append NONE as class ``N`` to pair logits ``[B,4,N,N]``."""

    if pair_logits.ndim != 4 or pair_logits.shape[1] != DIRECTION_COUNT:
        raise ValueError("pair_logits must have shape (batch, 4, tiles, tiles)")
    if pair_logits.shape[-2] != pair_logits.shape[-1]:
        raise ValueError("pair_logits candidate axes must be square")
    expected_none = pair_logits.shape[:3]
    if tuple(none_logits.shape) != tuple(expected_none):
        raise ValueError(
            f"none_logits must have shape {tuple(expected_none)}, got {tuple(none_logits.shape)}"
        )
    return torch.cat((pair_logits, none_logits.unsqueeze(-1)), dim=-1)


class ContextualDirectionalEdgeNet(nn.Module):
    """Predict contextual U/D/L/R neighbour distributions plus NONE."""

    def __init__(self, config: ContextualEdgeConfig | None = None) -> None:
        super().__init__()
        self.config = config or ContextualEdgeConfig()
        self.config.validate()
        self.tile_encoder = BoundaryAwareTileEncoder(self.config)
        self.set_blocks = nn.ModuleList(
            _SetAttentionBlock(self.config) for _ in range(self.config.transformer_layers)
        )
        self.context_norm = nn.LayerNorm(self.config.d_model)
        self.local_to_match = nn.Linear(self.config.local_dim, self.config.match_dim)
        self.context_to_match = nn.Linear(
            self.config.d_model, DIRECTION_COUNT * self.config.match_dim
        )
        self.match_norms = nn.ModuleList(
            nn.LayerNorm(self.config.match_dim) for _ in range(DIRECTION_COUNT)
        )
        self.right_query = nn.Linear(self.config.match_dim, self.config.match_dim, bias=False)
        self.left_key = nn.Linear(self.config.match_dim, self.config.match_dim, bias=False)
        self.down_query = nn.Linear(self.config.match_dim, self.config.match_dim, bias=False)
        self.up_key = nn.Linear(self.config.match_dim, self.config.match_dim, bias=False)
        none_input = self.config.local_dim + self.config.d_model
        none_hidden = max(32, self.config.match_dim)
        self.none_heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(none_input),
                nn.Linear(none_input, none_hidden),
                nn.GELU(),
                nn.Linear(none_hidden, 1),
            )
            for _ in range(DIRECTION_COUNT)
        )
        reconstruction_input = self.config.local_dim + self.config.d_model
        reconstruction_hidden = max(64, self.config.local_dim)
        self.boundary_decoder = nn.Sequential(
            nn.LayerNorm(reconstruction_input),
            nn.Linear(reconstruction_input, reconstruction_hidden),
            nn.GELU(),
            nn.Linear(
                reconstruction_hidden,
                self.config.input_channels * self.config.reconstruction_samples,
            ),
        )
        self.logit_log_scale = nn.Parameter(
            torch.tensor(math.log(self.config.initial_logit_scale), dtype=torch.float32)
        )

    def _encode_in_chunks(self, flat_tiles: Tensor) -> tuple[Tensor, Tensor]:
        global_parts = []
        side_parts = []
        size = self.config.encoder_chunk_size
        for start in range(0, flat_tiles.shape[0], size):
            global_token, side_tokens = self.tile_encoder(flat_tiles[start : start + size])
            global_parts.append(global_token)
            side_parts.append(side_tokens)
        return torch.cat(global_parts, dim=0), torch.cat(side_parts, dim=0)

    @staticmethod
    def _unit(values: Tensor) -> Tensor:
        return F.normalize(values, dim=-1, eps=1.0e-6)

    def _pair_logits(self, side_match: Tensor) -> tuple[Tensor, Tensor]:
        right_source = self._unit(self.right_query(side_match[:, :, RIGHT]))
        left_target = self._unit(self.left_key(side_match[:, :, LEFT]))
        down_source = self._unit(self.down_query(side_match[:, :, DOWN]))
        up_target = self._unit(self.up_key(side_match[:, :, UP]))
        scale = self.logit_log_scale.exp().clamp(1.0, 50.0)
        right = scale * torch.einsum("bid,bjd->bij", right_source, left_target)
        down = scale * torch.einsum("bid,bjd->bij", down_source, up_target)
        count = right.shape[-1]
        diagonal = torch.eye(count, dtype=torch.bool, device=right.device).unsqueeze(0)
        right = right.masked_fill(diagonal, self.config.diagonal_mask_value)
        down = down.masked_fill(diagonal, self.config.diagonal_mask_value)
        # These are views of the already-masked matrices.  The equality is
        # bit-exact, including the diagonal and mixed-precision execution.
        left = right.transpose(-1, -2)
        up = down.transpose(-1, -2)
        return torch.stack((up, down, left, right), dim=1), scale

    def forward(self, tiles: Tensor) -> dict[str, Tensor]:
        if tiles.ndim != 5:
            raise ValueError(
                "tiles must have shape (batch, tiles, 3, height, width), "
                f"got {tuple(tiles.shape)}"
            )
        batch, count, channels, height, width = tiles.shape
        if count != self.config.tile_count:
            raise ValueError(
                f"grid needs {self.config.tile_count} tiles, got {count}"
            )
        if channels != self.config.input_channels or height < 4 or width < 4:
            raise ValueError("E26 expects RGB tiles at least 4x4 pixels")
        flat = tiles.reshape(batch * count, channels, height, width)
        global_flat, local_flat = self._encode_in_chunks(flat)
        context = global_flat.reshape(batch, count, self.config.d_model)
        local = local_flat.reshape(batch, count, DIRECTION_COUNT, self.config.local_dim)
        for block in self.set_blocks:
            context = block(context)
        context = self.context_norm(context)

        local_match = self.local_to_match(local)
        context_match = self.context_to_match(context).reshape(
            batch, count, DIRECTION_COUNT, self.config.match_dim
        )
        side_match = torch.stack(
            [
                self.match_norms[direction](
                    local_match[:, :, direction] + context_match[:, :, direction]
                )
                for direction in range(DIRECTION_COUNT)
            ],
            dim=2,
        )
        pair_logits, scale = self._pair_logits(side_match)

        expanded_context = context.unsqueeze(2).expand(-1, -1, DIRECTION_COUNT, -1)
        decoder_input = torch.cat((local, expanded_context), dim=-1)
        none_logits = torch.stack(
            [
                self.none_heads[direction](decoder_input[:, :, direction]).squeeze(-1)
                for direction in range(DIRECTION_COUNT)
            ],
            dim=1,
        )
        reconstruction = torch.sigmoid(self.boundary_decoder(decoder_input))
        reconstruction = reconstruction.reshape(
            batch,
            count,
            DIRECTION_COUNT,
            self.config.input_channels,
            self.config.reconstruction_samples,
        )
        return {
            "pair_logits": pair_logits,
            "none_logits": none_logits,
            "logits": pack_directional_logits(pair_logits, none_logits),
            "tile_tokens": context,
            "side_tokens": side_match,
            "boundary_reconstruction": reconstruction,
            "scale": scale,
        }


def directional_neighbour_labels(
    permutation: Tensor,
    grid_height: int = 24,
    grid_width: int = 24,
) -> Tensor:
    """Return labels ``[B,4,N]``; class ``N`` is NONE.

    ``permutation[b, input_tile]`` is the clean row-major cell from which that
    shuffled input tile came.  This is exactly the synthetic ``CanvasDataset``
    convention.
    """

    if permutation.ndim == 1:
        permutation = permutation.unsqueeze(0)
    if permutation.ndim != 2:
        raise ValueError("permutation must have shape (batch, tiles)")
    batch, count = permutation.shape
    expected = int(grid_height) * int(grid_width)
    if grid_height <= 0 or grid_width <= 0 or count != expected:
        raise ValueError(f"grid has {expected} cells but permutation has {count} tiles")
    permutation = permutation.long()
    canonical = torch.arange(count, device=permutation.device).expand(batch, -1)
    if not torch.equal(permutation.sort(dim=1).values, canonical):
        raise ValueError("each row must be an exact permutation of 0..N-1")
    inverse = torch.empty_like(permutation)
    inverse.scatter_(1, permutation, canonical)
    row = torch.div(permutation, grid_width, rounding_mode="floor")
    column = permutation.remainder(grid_width)
    target_positions = (
        permutation - grid_width,
        permutation + grid_width,
        permutation - 1,
        permutation + 1,
    )
    valid = (
        row > 0,
        row + 1 < grid_height,
        column > 0,
        column + 1 < grid_width,
    )
    labels = []
    for target, is_valid in zip(target_positions, valid):
        neighbour = inverse.gather(1, target.clamp(0, count - 1))
        labels.append(torch.where(is_valid, neighbour, torch.full_like(neighbour, count)))
    return torch.stack(labels, dim=1)


def listwise_directional_ce(
    logits: Tensor,
    labels: Tensor,
    *,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross entropy over all N possible neighbours and the explicit NONE."""

    if logits.ndim != 4 or logits.shape[1] != DIRECTION_COUNT:
        raise ValueError("logits must have shape (batch, 4, tiles, tiles+1)")
    batch, _, count, classes = logits.shape
    if classes != count + 1 or tuple(labels.shape) != (batch, DIRECTION_COUNT, count):
        raise ValueError("logit/label shapes do not implement the N-neighbours-plus-NONE contract")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) > count):
        raise ValueError("labels must be candidate indices 0..N, where N is NONE")
    flat_logits = logits.float().reshape(-1, classes)
    flat_labels = labels.long().reshape(-1)
    log_probability = F.log_softmax(flat_logits, dim=-1)
    nll = -log_probability.gather(1, flat_labels[:, None]).squeeze(1)
    smoothing = float(label_smoothing)
    if smoothing == 0.0:
        return nll.mean()
    # Standard F.cross_entropy smoothing would assign probability to the
    # impossible self-neighbour, whose logit is deliberately masked.  Smooth
    # uniformly over all *valid* alternatives (N-1 other tiles plus NONE).
    valid = torch.ones((count, classes), dtype=torch.bool, device=logits.device)
    valid[:, :count].fill_diagonal_(False)
    valid = valid.unsqueeze(0).unsqueeze(0).expand(batch, DIRECTION_COUNT, -1, -1)
    smooth = -(logits.float().log_softmax(dim=-1).masked_fill(~valid, 0.0).sum(dim=-1))
    smooth = smooth / valid.sum(dim=-1)
    return ((1.0 - smoothing) * nll.reshape_as(labels) + smoothing * smooth).mean()


def clean_boundary_targets(
    clean_tiles: Tensor,
    samples: int,
    *,
    band: int = 2,
) -> Tensor:
    """Pool clean tile boundaries to ``[B,N,4,3,samples]`` in U/D/L/R order."""

    if clean_tiles.ndim != 5 or clean_tiles.shape[2] != 3:
        raise ValueError("clean_tiles must have shape (batch, tiles, 3, height, width)")
    if samples <= 0 or band <= 0:
        raise ValueError("samples and band must be positive")
    _, _, _, height, width = clean_tiles.shape
    if height < band or width < band:
        raise ValueError("band exceeds clean tile dimensions")
    top = clean_tiles[:, :, :, :band, :].mean(dim=3)
    bottom = clean_tiles[:, :, :, -band:, :].mean(dim=3)
    left = clean_tiles[:, :, :, :, :band].mean(dim=4)
    right = clean_tiles[:, :, :, :, -band:].mean(dim=4)
    result = []
    for trace in (top, bottom, left, right):
        shape = trace.shape
        pooled = F.adaptive_avg_pool1d(trace.reshape(-1, shape[-2], shape[-1]), samples)
        result.append(pooled.reshape(shape[0], shape[1], shape[2], samples))
    return torch.stack(result, dim=2)


def boundary_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    gradient_weight: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Clean-boundary reconstruction with an auxiliary tangential gradient loss."""

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must share shape (batch, tiles, 4, 3, samples)")
    if gradient_weight < 0.0:
        raise ValueError("gradient_weight cannot be negative")
    pixels = F.smooth_l1_loss(prediction, target)
    if prediction.shape[-1] > 1:
        prediction_gradient = prediction[..., 1:] - prediction[..., :-1]
        target_gradient = target[..., 1:] - target[..., :-1]
        gradients = F.smooth_l1_loss(prediction_gradient, target_gradient)
    else:
        gradients = pixels.new_zeros(())
    total = pixels + float(gradient_weight) * gradients
    return total, {"boundary_pixels": pixels, "boundary_gradients": gradients}


def model_config_dict(model: ContextualDirectionalEdgeNet) -> dict[str, Any]:
    return asdict(model.config)


def model_from_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    strict: bool = True,
) -> ContextualDirectionalEdgeNet:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("not an E26 contextual edge checkpoint")
    raw_config = payload.get("model_config")
    state = payload.get("model")
    if not isinstance(raw_config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("E26 checkpoint is missing model_config or model state")
    model = ContextualDirectionalEdgeNet(ContextualEdgeConfig(**dict(raw_config)))
    model.load_state_dict(state, strict=strict)
    return model


__all__ = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "DIRECTION_COUNT",
    "DIRECTION_NAMES",
    "CHECKPOINT_SCHEMA",
    "ContextualEdgeConfig",
    "BoundaryAwareTileEncoder",
    "ContextualDirectionalEdgeNet",
    "pack_directional_logits",
    "directional_neighbour_labels",
    "listwise_directional_ce",
    "clean_boundary_targets",
    "boundary_reconstruction_loss",
    "model_config_dict",
    "model_from_checkpoint_payload",
]
