"""Board-conditioned partial matching for square puzzle-tile sockets.

The puzzle pieces have no jigsaw-shaped geometry: every piece is an upright
20x20 pixel square.  This module therefore treats the four *visual* sides of a
tile as sockets.  For each axis it jointly matches all outgoing sockets to all
incoming sockets with a SuperGlue-style contextual network and a partial
optimal-transport layer.

The transport marginals encode a useful hard fact about a ``g x g`` board:
exactly ``g * (g - 1)`` horizontal (and vertical) pairs exist, while exactly
``g`` outgoing and ``g`` incoming sockets touch the image border.  A single
dustbin with capacity ``g`` represents those unmatched sockets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

SIDE_NAMES = ("right", "left", "bottom", "top")
BORDER_HEAD_EMBEDDING_V2 = "embedding_v2"
BORDER_HEAD_SCORE_STATS_V3 = "score_stats_v3"
BORDER_HEAD_VERSIONS = (BORDER_HEAD_EMBEDDING_V2, BORDER_HEAD_SCORE_STATS_V3)
SCORE_STATISTIC_NAMES = (
    "top1",
    "top1_minus_top2",
    "log_mean_exp",
    "normalised_entropy",
    "mean",
    "standard_deviation",
)


@dataclass(frozen=True)
class SocketOutput:
    """Raw and globally balanced scores for the two directed axes."""

    right_raw: torch.Tensor
    down_raw: torch.Tensor
    right_log_assignment: torch.Tensor
    down_log_assignment: torch.Tensor
    right_out_border_logits: torch.Tensor
    left_in_border_logits: torch.Tensor
    bottom_out_border_logits: torch.Tensor
    top_in_border_logits: torch.Tensor


def _normalise_tiles(tiles: torch.Tensor) -> torch.Tensor:
    if tiles.ndim != 5 or tiles.shape[2:] != (3, 20, 20):
        raise ValueError(f"expected B x N x 3 x 20 x 20 tiles, got {tuple(tiles.shape)}")
    value = tiles.float()
    if bool((value.detach().amax() > 1.5).item()):
        value = value / 255.0
    return value.clamp(0.0, 1.0)


def robust_tile_views(tiles: torch.Tensor) -> torch.Tensor:
    """Return raw and brightness-invariant channels for each tile."""

    value = _normalise_tiles(tiles)
    batch, count = value.shape[:2]
    flat = value.reshape(batch * count, 3, 20, 20)
    mean = flat.mean(dim=(1, 2, 3), keepdim=True)
    std = flat.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    local = ((flat - mean) / std).clamp(-4.0, 4.0) / 4.0
    gray = 0.299 * flat[:, :1] + 0.587 * flat[:, 1:2] + 0.114 * flat[:, 2:3]
    smooth = F.avg_pool2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    high = gray - smooth
    views = torch.cat((flat, local, gray, smooth, high), dim=1)
    return views.reshape(batch, count, 9, 20, 20)


class _Residual1d(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        groups = math.gcd(channels, max(1, channels // 16))
        self.network = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


class BoundarySequenceEncoder(nn.Module):
    """Encode one side while retaining its 20-pixel tangential sequence."""

    def __init__(
        self,
        *,
        dimension: int = 64,
        widths: tuple[int, ...] = (2, 4, 8),
        heads: int = 4,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.dimension = dimension
        self.widths = widths
        feature_channels = 9 * 2 * len(widths)
        self.stem = nn.Sequential(
            nn.Conv1d(feature_channels, dimension, 3, padding=1),
            _Residual1d(dimension, 1),
            _Residual1d(dimension, 2),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=dimension * 3,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=1)
        self.tangent_position = nn.Parameter(torch.randn(1, 20, dimension) * 0.02)
        self.projection = nn.Sequential(
            nn.LayerNorm(2 * dimension),
            nn.Linear(2 * dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )

    def _features(self, views: torch.Tensor, side: str) -> torch.Tensor:
        features: list[torch.Tensor] = []
        for width in self.widths:
            if side == "right":
                strip = views[..., -width:]
            elif side == "left":
                strip = views[..., :width]
            elif side == "bottom":
                strip = views[..., -width:, :].transpose(-2, -1)
            elif side == "top":
                strip = views[..., :width, :].transpose(-2, -1)
            else:
                raise ValueError(f"unknown side {side!r}")
            features.extend((strip.mean(-1), strip.std(-1, unbiased=False)))
        return torch.cat(features, dim=1)

    def forward(self, views: torch.Tensor, side: str) -> torch.Tensor:
        sequence = self.stem(self._features(views, side)).transpose(1, 2)
        sequence = self.context(sequence + self.tangent_position)
        pooled = torch.cat((sequence.mean(1), sequence.amax(1)), dim=1)
        return self.projection(pooled)


class TileContextEncoder(nn.Module):
    """Permutation-equivariant whole-board context; shuffled indices have no embedding."""

    def __init__(self, dimension: int, heads: int, layers: int) -> None:
        super().__init__()
        groups = math.gcd(dimension, max(1, dimension // 16))
        half_groups = math.gcd(dimension // 2, max(1, (dimension // 2) // 16))
        self.tile = nn.Sequential(
            nn.Conv2d(9, dimension // 2, 3, padding=1),
            nn.GroupNorm(half_groups, dimension // 2),
            nn.GELU(),
            nn.Conv2d(dimension // 2, dimension, 3, stride=2, padding=1),
            nn.GroupNorm(groups, dimension),
            nn.GELU(),
            nn.Conv2d(dimension, dimension, 3, stride=2, padding=1),
            nn.GroupNorm(groups, dimension),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=dimension * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.board = nn.TransformerEncoder(layer, num_layers=layers)
        self.normalise = nn.LayerNorm(dimension)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        batch, count = views.shape[:2]
        tokens = self.tile(views.reshape(batch * count, 9, 20, 20)).reshape(batch, count, -1)
        return self.normalise(tokens + self.board(tokens))


class _AttentionUpdate(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.source_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=0.05,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension * 3),
            nn.GELU(),
            nn.Linear(dimension * 3, dimension),
        )

    def forward(self, source: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query = self.source_norm(source)
        key_value = self.memory_norm(memory)
        update, _ = self.attention(query, key_value, key_value, need_weights=False)
        value = source + update
        return value + self.feed_forward(value)


class SocketGNN(nn.Module):
    """Alternating self/cross attention over both socket sets of one axis."""

    def __init__(self, dimension: int, heads: int, layers: int) -> None:
        super().__init__()
        self.source_self = nn.ModuleList(
            [_AttentionUpdate(dimension, heads) for _ in range(layers)]
        )
        self.target_self = nn.ModuleList(
            [_AttentionUpdate(dimension, heads) for _ in range(layers)]
        )
        self.source_cross = nn.ModuleList(
            [_AttentionUpdate(dimension, heads) for _ in range(layers)]
        )
        self.target_cross = nn.ModuleList(
            [_AttentionUpdate(dimension, heads) for _ in range(layers)]
        )

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for source_self, target_self, source_cross, target_cross in zip(
            self.source_self,
            self.target_self,
            self.source_cross,
            self.target_cross,
            strict=True,
        ):
            source = source_self(source, source)
            target = target_self(target, target)
            old_source, old_target = source, target
            source = source_cross(old_source, old_target)
            target = target_cross(old_target, old_source)
        return F.normalize(source, dim=-1), F.normalize(target, dim=-1)


def log_sinkhorn_iterations(
    scores: torch.Tensor,
    log_row_mass: torch.Tensor,
    log_column_mass: torch.Tensor,
    *,
    iterations: int,
) -> torch.Tensor:
    """Differentiable matrix scaling in log space."""

    value = scores
    row_potential = torch.zeros_like(log_row_mass)
    column_potential = torch.zeros_like(log_column_mass)
    for _ in range(iterations):
        row_potential = log_row_mass - torch.logsumexp(value + column_potential.unsqueeze(1), dim=2)
        column_potential = log_column_mass - torch.logsumexp(
            value + row_potential.unsqueeze(2), dim=1
        )
    return value + row_potential.unsqueeze(2) + column_potential.unsqueeze(1)


def partial_log_optimal_transport(
    scores: torch.Tensor,
    source_bin_score: torch.Tensor,
    *,
    unmatched: int,
    iterations: int = 12,
    target_bin_score: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match two equal socket sets with an exact-capacity shared dustbin."""

    if scores.ndim != 3 or scores.shape[1] != scores.shape[2]:
        raise ValueError(f"expected B x N x N scores, got {tuple(scores.shape)}")
    if unmatched <= 0 or unmatched >= scores.shape[1]:
        raise ValueError("unmatched capacity must be between 1 and N-1")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    batch, count, _ = scores.shape

    def expand_bin(value: torch.Tensor, *, outgoing: bool) -> torch.Tensor:
        value = value.to(device=scores.device, dtype=scores.dtype)
        if value.ndim == 0:
            shape = (batch, count, 1) if outgoing else (batch, 1, count)
            return value.expand(shape)
        if value.shape != (batch, count):
            raise ValueError(
                f"per-socket bin score must have shape {(batch, count)}, got {tuple(value.shape)}"
            )
        return value.unsqueeze(2 if outgoing else 1)

    if target_bin_score is None:
        target_bin_score = source_bin_score
    bins_column = expand_bin(source_bin_score, outgoing=True)
    bins_row = expand_bin(target_bin_score, outgoing=False)
    forbidden = torch.full((batch, 1, 1), -1e4, dtype=scores.dtype, device=scores.device)
    augmented = torch.cat(
        (torch.cat((scores, bins_column), dim=2), torch.cat((bins_row, forbidden), dim=2)),
        dim=1,
    )

    total = float(count + unmatched)
    real_mass = -math.log(total)
    dustbin_mass = math.log(float(unmatched)) - math.log(total)
    log_row_mass = scores.new_full((batch, count + 1), real_mass)
    log_column_mass = scores.new_full((batch, count + 1), real_mass)
    log_row_mass[:, -1] = dustbin_mass
    log_column_mass[:, -1] = dustbin_mass
    return log_sinkhorn_iterations(
        augmented,
        log_row_mass,
        log_column_mass,
        iterations=iterations,
    )


def socket_score_statistics(scores: torch.Tensor, *, outgoing: bool) -> torch.Tensor:
    """Summarise each socket's real-partner score distribution.

    Border identity is often negative evidence: a socket reaches the frame
    because *none* of its candidate partners fits well.  A single socket
    embedding cannot express that fact.  These six cheap statistics retain the
    best score, best-vs-second margin, smooth maximum, entropy, mean and spread.
    They are standardised across sockets on each board, which keeps the signal
    relative to the known exact border cardinality.

    ``outgoing=True`` summarises matrix rows; ``False`` summarises columns.  No
    tile index or target position enters the calculation, and a simultaneous
    permutation of the two socket sets simply permutes the returned rows.
    """

    if scores.ndim != 3 or scores.shape[1] != scores.shape[2]:
        raise ValueError(f"expected B x N x N scores, got {tuple(scores.shape)}")
    count = scores.shape[1]
    if count < 3:
        raise ValueError("at least three sockets are required for top-two statistics")
    candidates = scores if outgoing else scores.transpose(1, 2)
    self_pair = torch.eye(count, dtype=torch.bool, device=scores.device).unsqueeze(0)
    masked = candidates.masked_fill(self_pair, -torch.inf)

    top_two = masked.topk(2, dim=2).values
    top1 = top_two[:, :, 0]
    margin = top1 - top_two[:, :, 1]
    log_mean_exp = torch.logsumexp(masked, dim=2) - math.log(float(count - 1))
    log_probability = F.log_softmax(masked, dim=2)
    probability = log_probability.exp()
    entropy_terms = probability * log_probability.masked_fill(self_pair, 0.0)
    entropy = -entropy_terms.sum(2) / math.log(float(count - 1))

    finite_candidates = candidates.masked_fill(self_pair, 0.0)
    mean = finite_candidates.sum(2) / float(count - 1)
    squared_residual = (finite_candidates - mean.unsqueeze(2)).square().masked_fill(
        self_pair, 0.0
    )
    standard_deviation = (squared_residual.sum(2) / float(count - 1) + 1e-6).sqrt()
    features = torch.stack(
        (top1, margin, log_mean_exp, entropy, mean, standard_deviation),
        dim=2,
    )

    # Board-relative normalisation is itself permutation equivariant.  It also
    # prevents the learned head from depending on the current cosine-logit
    # scale or the synthetic crop size.
    centre = features.mean(dim=1, keepdim=True)
    scale = (features.var(dim=1, unbiased=False, keepdim=True) + 1e-4).sqrt()
    return (features - centre) / scale


class SocketMatcher(nn.Module):
    """Contextual side matcher with horizontal and vertical partial OT heads."""

    def __init__(
        self,
        *,
        dimension: int = 64,
        heads: int = 4,
        board_layers: int = 1,
        socket_layers: int = 1,
        sinkhorn_iterations: int = 12,
        border_head_version: str = BORDER_HEAD_EMBEDDING_V2,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        if border_head_version not in BORDER_HEAD_VERSIONS:
            raise ValueError(
                f"border_head_version must be one of {BORDER_HEAD_VERSIONS}, "
                f"got {border_head_version!r}"
            )
        self.dimension = dimension
        self.heads = heads
        self.board_layers = board_layers
        self.socket_layers = socket_layers
        self.sinkhorn_iterations = sinkhorn_iterations
        self.border_head_version = border_head_version
        self.boundary = BoundarySequenceEncoder(dimension=dimension, heads=heads)
        self.tile_context = TileContextEncoder(dimension, heads, board_layers)
        self.side_context = nn.ModuleDict(
            {name: nn.Linear(dimension, dimension, bias=False) for name in SIDE_NAMES}
        )
        self.horizontal = SocketGNN(dimension, heads, socket_layers)
        self.vertical = SocketGNN(dimension, heads, socket_layers)
        self.horizontal_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.vertical_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.horizontal_bin = nn.Parameter(torch.tensor(0.0))
        self.vertical_bin = nn.Parameter(torch.tensor(0.0))
        self.border_heads = nn.ModuleDict({name: nn.Linear(dimension, 1) for name in SIDE_NAMES})
        for head in self.border_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if border_head_version == BORDER_HEAD_SCORE_STATS_V3:
            # Conditional construction deliberately leaves the default v2
            # state_dict unchanged.  Zero initialisation makes a v2 -> v3
            # warm-start exactly behaviour-preserving before further training.
            self.border_distribution_heads = nn.ModuleDict(
                {
                    name: nn.Linear(len(SCORE_STATISTIC_NAMES), 1, bias=False)
                    for name in SIDE_NAMES
                }
            )
            for head in self.border_distribution_heads.values():
                nn.init.zeros_(head.weight)

    def _side_embeddings(
        self, views: torch.Tensor, context: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch, count = views.shape[:2]
        flat = views.reshape(batch * count, 9, 20, 20)
        output: dict[str, torch.Tensor] = {}
        for side in SIDE_NAMES:
            boundary = self.boundary(flat, side).reshape(batch, count, self.dimension)
            output[side] = boundary + self.side_context[side](context)
        return output

    @staticmethod
    def _similarity(
        source: torch.Tensor,
        target: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> torch.Tensor:
        scale = log_scale.exp().clamp(1.0, 100.0)
        scores = scale * source @ target.transpose(1, 2)
        diagonal = torch.eye(scores.shape[1], dtype=torch.bool, device=scores.device)
        return scores.masked_fill(diagonal.unsqueeze(0), -1e4)

    def _border_logits(
        self,
        *,
        side: str,
        embedding: torch.Tensor,
        raw_scores: torch.Tensor,
        outgoing: bool,
        shared_bin: torch.Tensor,
    ) -> torch.Tensor:
        logits = shared_bin + self.border_heads[side](embedding).squeeze(2)
        if self.border_head_version == BORDER_HEAD_SCORE_STATS_V3:
            statistics = socket_score_statistics(raw_scores, outgoing=outgoing)
            logits = logits + self.border_distribution_heads[side](statistics).squeeze(2)
        return logits

    def forward(self, tiles: torch.Tensor, *, grid: int | None = None) -> SocketOutput:
        views = robust_tile_views(tiles)
        count = views.shape[1]
        inferred = round(math.sqrt(count))
        grid = inferred if grid is None else grid
        if grid * grid != count:
            raise ValueError(f"tile count {count} is not grid^2 for grid={grid}")
        context = self.tile_context(views)
        sides = self._side_embeddings(views, context)
        right_source, left_target = self.horizontal(sides["right"], sides["left"])
        down_source, top_target = self.vertical(sides["bottom"], sides["top"])
        right_raw = self._similarity(right_source, left_target, self.horizontal_scale)
        down_raw = self._similarity(down_source, top_target, self.vertical_scale)
        right_out_border = self._border_logits(
            side="right",
            embedding=right_source,
            raw_scores=right_raw,
            outgoing=True,
            shared_bin=self.horizontal_bin,
        )
        left_in_border = self._border_logits(
            side="left",
            embedding=left_target,
            raw_scores=right_raw,
            outgoing=False,
            shared_bin=self.horizontal_bin,
        )
        bottom_out_border = self._border_logits(
            side="bottom",
            embedding=down_source,
            raw_scores=down_raw,
            outgoing=True,
            shared_bin=self.vertical_bin,
        )
        top_in_border = self._border_logits(
            side="top",
            embedding=top_target,
            raw_scores=down_raw,
            outgoing=False,
            shared_bin=self.vertical_bin,
        )
        return SocketOutput(
            right_raw=right_raw,
            down_raw=down_raw,
            right_log_assignment=partial_log_optimal_transport(
                right_raw,
                right_out_border,
                unmatched=grid,
                iterations=self.sinkhorn_iterations,
                target_bin_score=left_in_border,
            ),
            down_log_assignment=partial_log_optimal_transport(
                down_raw,
                bottom_out_border,
                unmatched=grid,
                iterations=self.sinkhorn_iterations,
                target_bin_score=top_in_border,
            ),
            right_out_border_logits=right_out_border,
            left_in_border_logits=left_in_border,
            bottom_out_border_logits=bottom_out_border,
            top_in_border_logits=top_in_border,
        )


def socket_targets(tile_at_position: torch.Tensor, *, grid: int) -> dict[str, torch.Tensor]:
    """Build exact outgoing and incoming targets, including the shared dustbin."""

    layout = tile_at_position.long()
    if layout.ndim != 2 or layout.shape[1] != grid * grid:
        raise ValueError(f"expected B x {grid * grid} layouts, got {tuple(layout.shape)}")
    count = grid * grid
    batch = layout.shape[0]
    expected = torch.arange(count, device=layout.device).expand(batch, -1)
    if not torch.equal(layout.sort(dim=1).values, expected):
        raise ValueError("each tile_at_position row must be a complete permutation")
    position = torch.empty_like(layout)
    position.scatter_(1, layout, expected)
    row = position // grid
    column = position % grid

    right_out = torch.full_like(layout, count)
    right_in = torch.full_like(layout, count)
    down_out = torch.full_like(layout, count)
    down_in = torch.full_like(layout, count)
    source = layout[:, :, None]
    right_neighbour = torch.roll(layout, shifts=-1, dims=1)[:, :, None]
    down_neighbour = torch.roll(layout, shifts=-grid, dims=1)[:, :, None]
    right_out.scatter_(1, source.squeeze(2), right_neighbour.squeeze(2))
    down_out.scatter_(1, source.squeeze(2), down_neighbour.squeeze(2))
    right_out[column == grid - 1] = count
    down_out[row == grid - 1] = count

    right_source = torch.roll(layout, shifts=1, dims=1)
    down_source = torch.roll(layout, shifts=grid, dims=1)
    right_in.scatter_(1, layout, right_source)
    down_in.scatter_(1, layout, down_source)
    right_in[column == 0] = count
    down_in[row == 0] = count
    return {
        "right_out": right_out,
        "right_in": right_in,
        "down_out": down_out,
        "down_in": down_in,
    }


def socket_matching_loss(
    output: SocketOutput,
    tile_at_position: torch.Tensor,
    *,
    grid: int,
    trusted_position: torch.Tensor | None = None,
    border_weight: float = 0.25,
    raw_rank_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Bidirectional partial-OT and optional raw listwise socket losses.

    ``trusted_position`` masks noisy target-assisted real labels.  Synthetic
    examples with exact known permutations should omit it.  The optional raw
    ranking auxiliary applies row and column cross-entropy only to interior
    edges whose two endpoints are trusted.  On weakly labelled real boards it
    also removes untrusted tiles from each softmax candidate set, so uncertain
    pseudo-labels are not silently trained as negatives.
    """

    if not math.isfinite(border_weight) or border_weight < 0:
        raise ValueError("border_weight must be finite and non-negative")
    if not math.isfinite(raw_rank_weight) or raw_rank_weight < 0:
        raise ValueError("raw_rank_weight must be finite and non-negative")
    targets = socket_targets(tile_at_position, grid=grid)
    batch, count = tile_at_position.shape
    total_mass_log = math.log(float(count + grid))

    if trusted_position is None:
        trusted_tile = torch.ones_like(tile_at_position, dtype=torch.bool)
    else:
        trusted_position = trusted_position.bool()
        if trusted_position.shape != tile_at_position.shape:
            raise ValueError("trusted_position shape differs from tile_at_position")
        trusted_tile = torch.empty_like(trusted_position)
        trusted_tile.scatter_(1, tile_at_position, trusted_position)

    terms: list[torch.Tensor] = []
    raw_rank_terms: list[torch.Tensor] = []
    diagnostics: dict[str, float] = {}
    for name, raw, assignment, out_key, in_key in (
        (
            "right",
            output.right_raw,
            output.right_log_assignment,
            "right_out",
            "right_in",
        ),
        (
            "down",
            output.down_raw,
            output.down_log_assignment,
            "down_out",
            "down_in",
        ),
    ):
        outgoing_target = targets[out_key]
        incoming_source = targets[in_key]
        outgoing = assignment[:, :count].gather(2, outgoing_target.unsqueeze(2)).squeeze(2)
        incoming = assignment.gather(1, incoming_source.unsqueeze(1)).squeeze(1)[:, :count]

        outgoing_partner_trusted = torch.ones_like(trusted_tile)
        interior_out = outgoing_target < count
        outgoing_partner_trusted[interior_out] = trusted_tile.gather(
            1, outgoing_target.clamp_max(count - 1)
        )[interior_out]
        incoming_partner_trusted = torch.ones_like(trusted_tile)
        interior_in = incoming_source < count
        incoming_partner_trusted[interior_in] = trusted_tile.gather(
            1, incoming_source.clamp_max(count - 1)
        )[interior_in]
        outgoing_mask = trusted_tile & outgoing_partner_trusted
        incoming_mask = trusted_tile & incoming_partner_trusted
        selected = torch.cat((outgoing[outgoing_mask], incoming[incoming_mask]))
        if not len(selected):
            raise ValueError("trusted mask removed every supervised socket")
        axis_loss = -(selected + total_mass_log).mean()
        terms.append(axis_loss)
        diagnostics[f"{name}_nll"] = float(axis_loss.detach())
        diagnostics[f"{name}_supervised"] = float(len(selected))

        # The OT NLL above supervises the globally balanced assignment.  This
        # auxiliary sharpens the unnormalised pair scores directly: outgoing
        # sockets are rows and incoming sockets are columns.  Border sockets
        # have no real partner and are intentionally left to OT/border heads.
        outgoing_rank_mask = outgoing_mask & interior_out
        incoming_rank_mask = incoming_mask & interior_in
        trusted_candidates = trusted_tile.unsqueeze(1)
        outgoing_scores = raw.masked_fill(~trusted_candidates, -1e4)
        incoming_scores = raw.transpose(1, 2).masked_fill(~trusted_candidates, -1e4)
        outgoing_rank_nll = F.cross_entropy(
            outgoing_scores.reshape(batch * count, count),
            outgoing_target.clamp_max(count - 1).reshape(batch * count),
            reduction="none",
        ).reshape(batch, count)
        incoming_rank_nll = F.cross_entropy(
            incoming_scores.reshape(batch * count, count),
            incoming_source.clamp_max(count - 1).reshape(batch * count),
            reduction="none",
        ).reshape(batch, count)
        raw_selected = torch.cat(
            (
                outgoing_rank_nll[outgoing_rank_mask],
                incoming_rank_nll[incoming_rank_mask],
            )
        )
        diagnostics[f"{name}_raw_rank_supervised"] = float(len(raw_selected))
        if len(raw_selected):
            axis_raw_rank_loss = raw_selected.mean()
            raw_rank_terms.append(axis_raw_rank_loss)
            diagnostics[f"{name}_raw_rank_nll"] = float(axis_raw_rank_loss.detach())
        else:
            diagnostics[f"{name}_raw_rank_nll"] = 0.0
    matching_loss = torch.stack(terms).mean()
    raw_rank_loss = (
        torch.stack(raw_rank_terms).mean() if raw_rank_terms else matching_loss.new_zeros(())
    )

    border_terms: list[torch.Tensor] = []
    for logits, target_key in (
        (output.right_out_border_logits, "right_out"),
        (output.left_in_border_logits, "right_in"),
        (output.bottom_out_border_logits, "down_out"),
        (output.top_in_border_logits, "down_in"),
    ):
        border = targets[target_key] == count
        for batch_index in range(batch):
            valid = trusted_tile[batch_index]
            positive = border[batch_index] & valid
            if not bool(positive.any()):
                continue
            log_probability = F.log_softmax(logits[batch_index].masked_fill(~valid, -1e4), dim=0)
            border_terms.append(-log_probability[positive].mean())
    border_loss = torch.stack(border_terms).mean() if border_terms else matching_loss.new_zeros(())
    loss = matching_loss + border_weight * border_loss + raw_rank_weight * raw_rank_loss
    diagnostics["matching_nll"] = float(matching_loss.detach())
    diagnostics["border_nll"] = float(border_loss.detach())
    diagnostics["raw_rank_nll"] = float(raw_rank_loss.detach())
    diagnostics["border_weight"] = float(border_weight)
    diagnostics["raw_rank_weight"] = float(raw_rank_weight)
    diagnostics["loss"] = float(loss.detach())
    return loss, diagnostics


@torch.no_grad()
def socket_retrieval_metrics(
    output: SocketOutput,
    tile_at_position: torch.Tensor,
    *,
    grid: int,
    ks: tuple[int, ...] = (1, 5, 16, 32),
) -> dict[str, float]:
    """Exact interior-neighbour and border-unmatched diagnostics."""

    targets = socket_targets(tile_at_position, grid=grid)
    count = grid * grid
    result: dict[str, float] = {}
    border_scores: list[tuple[str, torch.Tensor, torch.Tensor, str]] = []
    for name, raw, assignment, key, incoming_key, out_logits, in_logits in (
        (
            "right",
            output.right_raw,
            output.right_log_assignment,
            "right_out",
            "right_in",
            output.right_out_border_logits,
            output.left_in_border_logits,
        ),
        (
            "down",
            output.down_raw,
            output.down_log_assignment,
            "down_out",
            "down_in",
            output.bottom_out_border_logits,
            output.top_in_border_logits,
        ),
    ):
        truth = targets[key]
        interior = truth < count
        for variant, scores in (("raw", raw), ("ot", assignment[:, :count, :count])):
            order = scores.argsort(dim=2, descending=True)
            for k in ks:
                hits = (order[:, :, :k] == truth.unsqueeze(2)).any(dim=2)
                result[f"{name}_{variant}_r{k}"] = float(hits[interior].float().mean())
        border_scores.extend(
            (
                (f"{name}_out", out_logits, assignment[:, :count, count], key),
                (f"{name}_in", in_logits, assignment[:, count, :count], incoming_key),
            )
        )

    for side_name, logits, ot_score, target_key in border_scores:
        truth = targets[target_key] == count
        for evidence_name, evidence in (("head", logits), ("ot", ot_score)):
            predicted = torch.zeros_like(truth)
            indices = evidence.topk(grid, dim=1).indices
            predicted.scatter_(1, indices, True)
            # Exactly ``grid`` sides are predicted and exactly ``grid`` are
            # true, so precision, recall and F1 are identical.
            result[f"{side_name}_{evidence_name}_border_top{grid}"] = float(
                (predicted & truth).sum() / truth.sum()
            )
    for suffix in tuple(result):
        if not suffix.startswith("right_"):
            continue
        down_key = "down_" + suffix.removeprefix("right_")
        if down_key in result:
            result["pooled_" + suffix.removeprefix("right_")] = 0.5 * (
                result[suffix] + result[down_key]
            )
    for evidence_name in ("head", "ot"):
        keys = (
            f"right_out_{evidence_name}_border_top{grid}",
            f"right_in_{evidence_name}_border_top{grid}",
            f"down_out_{evidence_name}_border_top{grid}",
            f"down_in_{evidence_name}_border_top{grid}",
        )
        result[f"pooled_four_side_{evidence_name}_border_top{grid}"] = float(
            sum(result[key] for key in keys) / len(keys)
        )
    return result


__all__ = [
    "BORDER_HEAD_EMBEDDING_V2",
    "BORDER_HEAD_SCORE_STATS_V3",
    "BORDER_HEAD_VERSIONS",
    "BoundarySequenceEncoder",
    "SCORE_STATISTIC_NAMES",
    "SocketMatcher",
    "SocketOutput",
    "partial_log_optimal_transport",
    "robust_tile_views",
    "socket_matching_loss",
    "socket_retrieval_metrics",
    "socket_score_statistics",
    "socket_targets",
]
