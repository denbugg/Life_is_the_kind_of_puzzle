"""Minimal faithful learned 6x6 gate for the BasinCycle hypothesis.

The module contains architecture and evaluation primitives only.  It has no
organizer-data loader, no training CLI, and no DEV/test path.  Its proposal
builder consumes only dirty-visible model scores and a current strict layout;
truth is accepted solely by the separate oracle/label functions after proposal
identities have been frozen.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from aiijc_puzzle.basincycle_synthetic import (
    apply_cycle,
    canonical_cycle,
    evidence_energy,
    exact_count,
    is_strict_permutation,
    propose_cycles,
    true_pair_count,
)

GRID_SIZE = 6
TILE_COUNT = GRID_SIZE * GRID_SIZE
TILE_SIZE = 20
SIDE_COUNT = 4
SIDE_WIDTH = 4
SIDE_NAMES = ("right", "left", "bottom", "top")
AXIS_NAMES = ("right", "down")
METRIC_NAMES = ("pair", "exact", "radius2")
QUANTILE_LEVELS = (0.10, 0.50, 0.90)


@dataclass(frozen=True)
class ProposalBank:
    """Padded target-free cycle identities with KEEP fixed at index zero."""

    positions: torch.Tensor
    lengths: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class StageBOutput:
    """All learned tensors and strict candidate layouts for one batch."""

    pair_logits: torch.Tensor
    boundary_prediction: torch.Tensor
    proposal_bank: ProposalBank
    candidate_layouts: torch.Tensor
    action_logits: torch.Tensor
    quantiles: torch.Tensor
    risk_logits: torch.Tensor


@dataclass(frozen=True)
class StageBLabels:
    """Reference-only labels attached after a target-free proposal freeze."""

    positive_actions: torch.Tensor
    metric_deltas: torch.Tensor
    loses_true_pair: torch.Tensor
    edge_targets: torch.Tensor
    clean_boundary_targets: torch.Tensor


@dataclass(frozen=True)
class OracleDiagnostic:
    """One state-level comparison of frozen proposals with all 2/3-cycles."""

    exhaustive_has_benefit: bool
    proposal_has_benefit: bool
    exhaustive_best_pair_delta: int
    proposal_best_pair_delta: int
    proposal_best_exact_delta: int
    proposal_count: int


class StrideOneResidual(nn.Module):
    """Depthwise-separable residual block that preserves every spatial sample."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm_depthwise = nn.GroupNorm(8, channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm_pointwise = nn.GroupNorm(8, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = functional.silu(self.norm_depthwise(self.depthwise(value)))
        value = self.norm_pointwise(self.pointwise(value))
        return functional.silu(value + residual)


def _strict_layouts(layouts: torch.Tensor, *, tile_count: int) -> bool:
    if layouts.ndim < 2 or layouts.shape[-1] != tile_count:
        return False
    expected = torch.arange(tile_count, device=layouts.device)
    observed = torch.sort(layouts, dim=-1).values
    return bool(torch.all(observed == expected).item())


def _hashable_cycle_sort_key(cycle: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    return len(cycle), cycle


def build_target_free_proposal_bank(
    pair_logits: torch.Tensor,
    layouts: torch.Tensor,
    *,
    grid_size: int,
    top_k: int,
    candidate_cap: int,
    seed_count: int,
    max_cycle_length: int,
    proposal_cap: int,
) -> ProposalBank:
    """Freeze short-cycle identities from visible logits, never from truth.

    Candidate enumeration is deliberately detached.  The selected cycle
    utility remains differentiable through current-state features and visible
    before/after edge scores, but no gradient or label can alter membership.
    """

    tile_count = grid_size * grid_size
    if pair_logits.shape != (len(layouts), 2, tile_count, tile_count):
        raise ValueError("pair_logits shape does not match layouts/grid")
    if layouts.shape != (len(layouts), tile_count) or not _strict_layouts(
        layouts,
        tile_count=tile_count,
    ):
        raise ValueError("layouts must be a batch of strict permutations")
    if proposal_cap < 2:
        raise ValueError("proposal_cap must reserve KEEP and at least one cycle")

    positions = torch.full(
        (len(layouts), proposal_cap, max_cycle_length),
        -1,
        dtype=torch.long,
        device=layouts.device,
    )
    lengths = torch.zeros(
        (len(layouts), proposal_cap),
        dtype=torch.long,
        device=layouts.device,
    )
    valid = torch.zeros(
        (len(layouts), proposal_cap),
        dtype=torch.bool,
        device=layouts.device,
    )
    valid[:, 0] = True  # The immutable identity/control path.

    score_values = pair_logits.detach().to(device="cpu", dtype=torch.float64).numpy()
    layout_values = layouts.detach().cpu().numpy()
    for batch_index, layout in enumerate(layout_values):
        right = score_values[batch_index, 0]
        down = score_values[batch_index, 1]
        proposals = propose_cycles(
            layout,
            right,
            down,
            grid_size=grid_size,
            top_k=top_k,
            candidate_cap=candidate_cap,
            seed_count=seed_count,
            max_cycle_length=max_cycle_length,
        )
        baseline = evidence_energy(layout, right, down, grid_size=grid_size)
        ranked: list[tuple[float, tuple[int, ...]]] = []
        for cycle in proposals:
            candidate = apply_cycle(layout, cycle)
            delta = evidence_energy(candidate, right, down, grid_size=grid_size) - baseline
            ranked.append((float(delta), cycle))
        ranked.sort(key=lambda item: (-item[0], *_hashable_cycle_sort_key(item[1])))
        for proposal_index, (_, cycle) in enumerate(ranked[: proposal_cap - 1], start=1):
            positions[batch_index, proposal_index, : len(cycle)] = torch.tensor(
                cycle,
                dtype=torch.long,
                device=layouts.device,
            )
            lengths[batch_index, proposal_index] = len(cycle)
            valid[batch_index, proposal_index] = True
    return ProposalBank(positions=positions, lengths=lengths, valid=valid)


def materialize_candidate_layouts(layouts: torch.Tensor, bank: ProposalBank) -> torch.Tensor:
    """Materialise KEEP/cycle results and assert the hard-permutation contract."""

    if layouts.ndim != 2 or not _strict_layouts(layouts, tile_count=layouts.shape[1]):
        raise ValueError("layouts must be strict permutations")
    batch_size, tile_count = layouts.shape
    if bank.positions.shape[:2] != bank.valid.shape or bank.lengths.shape != bank.valid.shape:
        raise ValueError("proposal bank tensors are misaligned")
    if bank.positions.shape[0] != batch_size:
        raise ValueError("proposal bank batch differs from layouts")
    if not torch.all(bank.valid[:, 0]) or not torch.all(bank.lengths[:, 0] == 0):
        raise ValueError("proposal zero must be valid KEEP")

    proposal_cap = bank.valid.shape[1]
    candidates = layouts[:, None, :].expand(-1, proposal_cap, -1).clone()
    for batch_index in range(batch_size):
        for proposal_index in range(1, proposal_cap):
            if not bool(bank.valid[batch_index, proposal_index]):
                continue
            length = int(bank.lengths[batch_index, proposal_index].item())
            cycle = bank.positions[batch_index, proposal_index, :length]
            if length < 2 or torch.any(cycle < 0) or len(torch.unique(cycle)) != length:
                raise ValueError("valid proposal is not a closed cycle")
            destination = cycle
            source = torch.roll(cycle, shifts=-1)
            candidates[batch_index, proposal_index, destination] = layouts[
                batch_index,
                source,
            ]
    if not _strict_layouts(candidates, tile_count=tile_count):
        raise AssertionError("candidate materialization violated strict permutation")
    if not torch.equal(candidates[:, 0], layouts):
        raise AssertionError("KEEP no longer exactly replays the control layout")
    return candidates


class BasinCycleStageB(nn.Module):
    """Small learned Stage-B refiner with no soft-assignment output path."""

    def __init__(
        self,
        *,
        grid_size: int = GRID_SIZE,
        feature_channels: int = 48,
        retrieval_dim: int = 64,
        state_dim: int = 96,
        encoder_blocks: int = 4,
        state_blocks: int = 3,
        proposal_top_k: int = 8,
        proposal_candidate_cap: int = 6,
        proposal_seed_count: int = 18,
        max_cycle_length: int = 3,
        proposal_cap: int = 256,
    ) -> None:
        super().__init__()
        if grid_size != 6:
            raise ValueError("the reviewed Stage-B architecture is fixed to 6x6")
        if feature_channels % 8 or state_dim % 8:
            raise ValueError("feature/state widths must be divisible by eight")
        self.grid_size = grid_size
        self.tile_count = grid_size * grid_size
        self.feature_channels = feature_channels
        self.retrieval_dim = retrieval_dim
        self.state_dim = state_dim
        self.proposal_top_k = proposal_top_k
        self.proposal_candidate_cap = proposal_candidate_cap
        self.proposal_seed_count = proposal_seed_count
        self.max_cycle_length = max_cycle_length
        self.proposal_cap = proposal_cap

        self.image_stem = nn.Sequential(
            nn.Conv2d(10, feature_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
        )
        self.image_blocks = nn.ModuleList(
            [StrideOneResidual(feature_channels) for _ in range(encoder_blocks)]
        )
        self.side_query = nn.Linear(feature_channels, retrieval_dim)
        self.side_key = nn.Linear(feature_channels, retrieval_dim)
        self.side_type = nn.Parameter(torch.zeros(SIDE_COUNT, retrieval_dim))
        self.boundary_head = nn.Linear(feature_channels, 6)
        self.tile_projection = nn.Linear(feature_channels, state_dim)

        self.state_stem = nn.Conv2d(state_dim + 8, state_dim, kernel_size=1)
        self.state_blocks = nn.ModuleList(
            [StrideOneResidual(state_dim) for _ in range(state_blocks)]
        )

        self.scalar_feature_count = 11
        action_input_dim = 5 * state_dim + self.scalar_feature_count
        self.action_head = nn.Sequential(
            nn.Linear(action_input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 11),
        )

        scharr_x = torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]
        ) / 16.0
        scharr_y = scharr_x.T.contiguous()
        laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("scharr_x", scharr_x[None, None], persistent=False)
        self.register_buffer("scharr_y", scharr_y[None, None], persistent=False)
        self.register_buffer("laplacian", laplacian[None, None], persistent=False)

    def trainable_parameter_count(self) -> int:
        """Return the exact trainable parameter count of this instance."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _preprocess(self, tiles: torch.Tensor) -> torch.Tensor:
        if tiles.ndim != 5 or tiles.shape[1:] != (self.tile_count, 3, 20, 20):
            raise ValueError("tiles must have shape B x 36 x 3 x 20 x 20")
        if not tiles.is_floating_point() or not torch.isfinite(tiles).all():
            raise ValueError("tiles must be finite floating point values")
        batch_size = len(tiles)
        raw = tiles.reshape(batch_size * self.tile_count, 3, 20, 20)
        mean = raw.mean(dim=(2, 3), keepdim=True)
        std = raw.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        standardized = (raw - mean) / std
        luma = (
            0.2126 * raw[:, 0:1]
            + 0.7152 * raw[:, 1:2]
            + 0.0722 * raw[:, 2:3]
        )
        padded = functional.pad(luma, (1, 1, 1, 1), mode="replicate")
        grad_x = functional.conv2d(padded, self.scharr_x)
        grad_y = functional.conv2d(padded, self.scharr_y)
        laplace = functional.conv2d(padded, self.laplacian)
        highpass = luma - functional.avg_pool2d(
            padded,
            kernel_size=3,
            stride=1,
        )
        return torch.cat((raw, standardized, grad_x, grad_y, laplace, highpass), dim=1)

    @staticmethod
    def _side_sequences(features: torch.Tensor) -> torch.Tensor:
        """Return B x N x four-sides x 20-tangent x C at full resolution."""

        values = features.reshape(-1, TILE_COUNT, features.shape[1], 20, 20)
        right = values[..., -SIDE_WIDTH:].mean(dim=-1).permute(0, 1, 3, 2)
        left = values[..., :SIDE_WIDTH].mean(dim=-1).permute(0, 1, 3, 2)
        bottom = values[..., -SIDE_WIDTH:, :].mean(dim=-2).permute(0, 1, 3, 2)
        top = values[..., :SIDE_WIDTH, :].mean(dim=-2).permute(0, 1, 3, 2)
        return torch.stack((right, left, bottom, top), dim=2)

    def _pair_logits(self, sides: torch.Tensor) -> torch.Tensor:
        side_type = self.side_type[None, None, :, None]
        query = functional.normalize(self.side_query(sides) + side_type, dim=-1)
        key = functional.normalize(self.side_key(sides) + side_type, dim=-1)
        scale = math.sqrt(self.retrieval_dim * sides.shape[-2])
        right = torch.einsum("bnld,bmld->bnm", query[:, :, 0], key[:, :, 1]) / scale
        down = torch.einsum("bnld,bmld->bnm", query[:, :, 2], key[:, :, 3]) / scale
        result = torch.stack((right, down), dim=1)
        diagonal = torch.eye(self.tile_count, dtype=torch.bool, device=result.device)
        return result.masked_fill(diagonal[None, None], -1e4)

    def _contact_features(self, pair_logits: torch.Tensor, layouts: torch.Tensor) -> torch.Tensor:
        batch_size = len(layouts)
        board = layouts.reshape(batch_size, self.grid_size, self.grid_size)
        scores = torch.zeros(
            batch_size,
            self.grid_size,
            self.grid_size,
            4,
            dtype=pair_logits.dtype,
            device=pair_logits.device,
        )
        valid = torch.zeros_like(scores)
        batch = torch.arange(batch_size, device=pair_logits.device)[:, None, None]

        horizontal = pair_logits[
            batch,
            0,
            board[:, :, :-1],
            board[:, :, 1:],
        ]
        scores[:, :, :-1, 0] = horizontal
        scores[:, :, 1:, 1] = horizontal
        valid[:, :, :-1, 0] = 1.0
        valid[:, :, 1:, 1] = 1.0

        vertical = pair_logits[
            batch,
            1,
            board[:, :-1, :],
            board[:, 1:, :],
        ]
        scores[:, :-1, :, 2] = vertical
        scores[:, 1:, :, 3] = vertical
        valid[:, :-1, :, 2] = 1.0
        valid[:, 1:, :, 3] = 1.0
        return torch.cat((torch.tanh(scores), valid), dim=-1).reshape(
            batch_size,
            self.tile_count,
            8,
        )

    def _state_context(
        self,
        tile_embeddings: torch.Tensor,
        pair_logits: torch.Tensor,
        layouts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index = layouts[..., None].expand(-1, -1, self.state_dim)
        slot_embeddings = torch.gather(tile_embeddings, dim=1, index=index)
        contacts = self._contact_features(pair_logits, layouts)
        state = torch.cat((slot_embeddings, contacts), dim=-1)
        state = state.reshape(len(layouts), self.grid_size, self.grid_size, -1).permute(0, 3, 1, 2)
        state = functional.silu(self.state_stem(state))
        for block in self.state_blocks:
            state = block(state)
        context = state.permute(0, 2, 3, 1).reshape(len(layouts), self.tile_count, self.state_dim)
        return context, slot_embeddings

    def _edge_vectors(self, pair_logits: torch.Tensor, layouts: torch.Tensor) -> torch.Tensor:
        """Return visible scores of all 60 grid contacts for B x P layouts."""

        if layouts.ndim != 3:
            raise ValueError("candidate layouts must have shape B x P x N")
        batch_size, proposal_cap, _ = layouts.shape
        board = layouts.reshape(batch_size, proposal_cap, self.grid_size, self.grid_size)
        batch = torch.arange(batch_size, device=layouts.device)[:, None, None, None]
        horizontal = pair_logits[
            batch,
            0,
            board[:, :, :, :-1],
            board[:, :, :, 1:],
        ].reshape(batch_size, proposal_cap, -1)
        vertical = pair_logits[
            batch,
            1,
            board[:, :, :-1, :],
            board[:, :, 1:, :],
        ].reshape(batch_size, proposal_cap, -1)
        return torch.cat((horizontal, vertical), dim=-1)

    def _action_features(
        self,
        context: torch.Tensor,
        slot_embeddings: torch.Tensor,
        tile_embeddings: torch.Tensor,
        pair_logits: torch.Tensor,
        candidates: torch.Tensor,
        bank: ProposalBank,
    ) -> torch.Tensor:
        batch_size, proposal_cap, _ = candidates.shape
        length = bank.positions.shape[-1]
        safe_positions = bank.positions.clamp_min(0)
        batch = torch.arange(batch_size, device=candidates.device)[:, None, None]
        proposal = torch.arange(proposal_cap, device=candidates.device)[None, :, None]
        position_mask = (
            torch.arange(length, device=candidates.device)[None, None, :]
            < bank.lengths[..., None]
        )
        divisor = position_mask.sum(dim=-1, keepdim=True).clamp_min(1)

        gathered_context = context[batch, safe_positions]
        context_mean = (gathered_context * position_mask[..., None]).sum(dim=2) / divisor
        context_max = gathered_context.masked_fill(
            ~position_mask[..., None],
            -torch.inf,
        ).max(dim=2).values
        context_max = torch.where(
            torch.isfinite(context_max),
            context_max,
            torch.zeros_like(context_max),
        )

        old_slots = slot_embeddings[batch, safe_positions]
        new_tile_ids = candidates[batch, proposal, safe_positions]
        new_tiles = tile_embeddings[batch, new_tile_ids]
        tile_delta = (new_tiles - old_slots) * position_mask[..., None]
        delta_mean = tile_delta.sum(dim=2) / divisor
        delta_max = tile_delta.abs().max(dim=2).values
        global_context = context.mean(dim=1)[:, None, :].expand(-1, proposal_cap, -1)

        edge_values = self._edge_vectors(pair_logits, candidates)
        base_edges = edge_values[:, :1]
        edge_delta = edge_values - base_edges
        control_layout = candidates[:, :1]
        base_board = control_layout.reshape(batch_size, 1, self.grid_size, self.grid_size)
        candidate_board = candidates.reshape(
            batch_size,
            proposal_cap,
            self.grid_size,
            self.grid_size,
        )
        changed_horizontal = (
            (candidate_board[:, :, :, :-1] != base_board[:, :, :, :-1])
            | (candidate_board[:, :, :, 1:] != base_board[:, :, :, 1:])
        ).reshape(batch_size, proposal_cap, -1)
        changed_vertical = (
            (candidate_board[:, :, :-1, :] != base_board[:, :, :-1, :])
            | (candidate_board[:, :, 1:, :] != base_board[:, :, 1:, :])
        ).reshape(batch_size, proposal_cap, -1)
        changed = torch.cat((changed_horizontal, changed_vertical), dim=-1)
        changed_count = changed.sum(dim=-1).clamp_min(1)
        changed_delta = edge_delta.masked_fill(~changed, 0.0)
        minimum = edge_delta.masked_fill(~changed, torch.inf).min(dim=-1).values
        maximum = edge_delta.masked_fill(~changed, -torch.inf).max(dim=-1).values
        minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        edge_count = edge_delta.shape[-1]
        scalars = torch.stack(
            (
                (bank.lengths == 0).to(edge_delta.dtype),
                bank.lengths.to(edge_delta.dtype) / self.max_cycle_length,
                edge_delta.sum(dim=-1) / edge_count,
                changed_delta.sum(dim=-1) / changed_count,
                minimum,
                maximum,
                changed_delta.clamp_max(0).sum(dim=-1) / edge_count,
                changed_delta.clamp_min(0).sum(dim=-1) / edge_count,
                ((edge_delta < 0) & changed).sum(dim=-1) / edge_count,
                ((edge_delta > 0) & changed).sum(dim=-1) / edge_count,
                changed.sum(dim=-1) / edge_count,
            ),
            dim=-1,
        )
        return torch.cat(
            (context_mean, context_max, delta_mean, delta_max, global_context, scalars),
            dim=-1,
        )

    def forward(self, tiles: torch.Tensor, layouts: torch.Tensor) -> StageBOutput:
        if layouts.shape != (len(tiles), self.tile_count) or not _strict_layouts(
            layouts,
            tile_count=self.tile_count,
        ):
            raise ValueError("layouts must be strict B x 36 permutations")
        preprocessed = self._preprocess(tiles)
        features = self.image_stem(preprocessed)
        for block in self.image_blocks:
            features = block(features)
        sides = self._side_sequences(features)
        pair_logits = self._pair_logits(sides)
        boundary_prediction = self.boundary_head(sides)
        pooled = features.mean(dim=(2, 3)).reshape(len(tiles), self.tile_count, -1)
        tile_embeddings = self.tile_projection(pooled)
        context, slot_embeddings = self._state_context(tile_embeddings, pair_logits, layouts)

        bank = build_target_free_proposal_bank(
            pair_logits,
            layouts,
            grid_size=self.grid_size,
            top_k=self.proposal_top_k,
            candidate_cap=self.proposal_candidate_cap,
            seed_count=self.proposal_seed_count,
            max_cycle_length=self.max_cycle_length,
            proposal_cap=self.proposal_cap,
        )
        candidates = materialize_candidate_layouts(layouts, bank)
        action_features = self._action_features(
            context,
            slot_embeddings,
            tile_embeddings,
            pair_logits,
            candidates,
            bank,
        )
        raw = self.action_head(action_features)
        action_logits = raw[..., 0].masked_fill(~bank.valid, -torch.inf)
        raw_quantiles = raw[..., 1:10].reshape(*raw.shape[:2], 3, 3)
        q10 = raw_quantiles[..., 0]
        q50 = q10 + functional.softplus(raw_quantiles[..., 1])
        q90 = q50 + functional.softplus(raw_quantiles[..., 2])
        quantiles = torch.stack((q10, q50, q90), dim=-1)
        risk_logits = raw[..., 10].masked_fill(~bank.valid, torch.inf)
        return StageBOutput(
            pair_logits=pair_logits,
            boundary_prediction=boundary_prediction,
            proposal_bank=bank,
            candidate_layouts=candidates,
            action_logits=action_logits,
            quantiles=quantiles,
            risk_logits=risk_logits,
        )


def select_hard_action(
    output: StageBOutput,
    *,
    minimum_pair_q10: float = 0.0,
    maximum_risk: float = 0.10,
    minimum_pair_q50_margin_over_keep: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the fixed conservative value rule and return strict layouts."""

    bank = output.proposal_bank
    batch_size = len(output.candidate_layouts)
    selected = torch.zeros(batch_size, dtype=torch.long, device=output.action_logits.device)
    for batch_index in range(batch_size):
        keep_pair = float(output.quantiles[batch_index, 0, 0, 1].item())
        eligible: list[int] = []
        for proposal_index in range(1, bank.valid.shape[1]):
            if not bool(bank.valid[batch_index, proposal_index]):
                continue
            pair_q10 = float(output.quantiles[batch_index, proposal_index, 0, 0].item())
            pair_q50 = float(output.quantiles[batch_index, proposal_index, 0, 1].item())
            risk = float(torch.sigmoid(output.risk_logits[batch_index, proposal_index]).item())
            if (
                pair_q10 >= minimum_pair_q10
                and risk <= maximum_risk
                and pair_q50 > keep_pair + minimum_pair_q50_margin_over_keep
            ):
                eligible.append(proposal_index)
        if eligible:
            selected[batch_index] = min(
                eligible,
                key=lambda index: (
                    -float(output.quantiles[batch_index, index, 0, 1].item()),
                    -float(output.quantiles[batch_index, index, 2, 1].item()),
                    -float(output.quantiles[batch_index, index, 1, 1].item()),
                    int(bank.lengths[batch_index, index].item()),
                    tuple(int(value) for value in bank.positions[batch_index, index].tolist()),
                ),
            )
    batch = torch.arange(batch_size, device=selected.device)
    layouts = output.candidate_layouts[batch, selected]
    if not _strict_layouts(layouts, tile_count=layouts.shape[-1]):
        raise AssertionError("selected Stage-B action is not a strict permutation")
    return selected, layouts


def stage_b_loss(
    output: StageBOutput,
    labels: StageBLabels,
    *,
    edge_weight: float = 0.25,
    restore_weight: float = 0.15,
    quantile_weight: float = 0.50,
    risk_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the fixed policy/edge/restore/quantile/risk Stage-B objective."""

    valid = output.proposal_bank.valid
    if labels.positive_actions.shape != valid.shape:
        raise ValueError("positive action mask shape differs from proposal bank")
    if torch.any(labels.positive_actions & ~valid):
        raise ValueError("a positive action points to padding")
    if torch.any(labels.positive_actions.sum(dim=1) == 0):
        raise ValueError("each state needs at least one basin-equivalent positive action")
    if labels.metric_deltas.shape != (*valid.shape, 3):
        raise ValueError("metric deltas must be B x P x three metrics")
    if labels.loses_true_pair.shape != valid.shape:
        raise ValueError("risk labels must match proposal bank")

    log_probabilities = functional.log_softmax(output.action_logits, dim=1)
    positive_log_mass = torch.logsumexp(
        log_probabilities.masked_fill(~labels.positive_actions, -torch.inf),
        dim=1,
    )
    policy = -positive_log_mass.mean()

    target = labels.metric_deltas[..., None]
    errors = target - output.quantiles
    levels = torch.tensor(QUANTILE_LEVELS, device=errors.device, dtype=errors.dtype)
    pinball = torch.maximum(levels * errors, (levels - 1.0) * errors)
    quantile = pinball[valid[..., None, None].expand_as(pinball)].mean()
    risk = functional.binary_cross_entropy_with_logits(
        output.risk_logits[valid],
        labels.loses_true_pair[valid].to(output.risk_logits.dtype),
    )

    if labels.edge_targets.shape != output.pair_logits.shape[:3]:
        raise ValueError("edge targets must have shape B x two axes x N")
    edge_rows = output.pair_logits.reshape(-1, output.pair_logits.shape[-1])
    edge_targets = labels.edge_targets.reshape(-1)
    edge = functional.cross_entropy(edge_rows, edge_targets, ignore_index=-1)

    if labels.clean_boundary_targets.shape != output.boundary_prediction.shape:
        raise ValueError("clean boundary targets differ from prediction shape")
    restore_error = output.boundary_prediction - labels.clean_boundary_targets
    restore = torch.sqrt(restore_error.square() + 1e-6).mean()

    total = (
        policy
        + edge_weight * edge
        + restore_weight * restore
        + quantile_weight * quantile
        + risk_weight * risk
    )
    return total, {
        "policy": policy,
        "edge": edge,
        "restore": restore,
        "quantile": quantile,
        "risk": risk,
    }


def radius_two_count(layout: np.ndarray, truth: np.ndarray, *, grid_size: int) -> int:
    """Count tiles within Manhattan radius two of their planted truth slots."""

    values = np.asarray(layout)
    reference = np.asarray(truth)
    if not is_strict_permutation(values) or not is_strict_permutation(reference):
        raise ValueError("layout and truth must be strict permutations")
    if values.shape != reference.shape or values.size != grid_size * grid_size:
        raise ValueError("layout/truth size differs from grid")
    truth_slot = np.empty(values.size, dtype=np.int64)
    truth_slot[reference] = np.arange(values.size)
    target = truth_slot[values]
    current = np.arange(values.size)
    distance = np.abs(target // grid_size - current // grid_size) + np.abs(
        target % grid_size - current % grid_size
    )
    return int(np.count_nonzero(distance <= 2))


def metric_deltas_for_bank(
    layouts: np.ndarray,
    truth: np.ndarray,
    positions: np.ndarray,
    lengths: np.ndarray,
    valid: np.ndarray,
    *,
    grid_size: int,
) -> np.ndarray:
    """Score a frozen bank after proposal identities can no longer change."""

    control = np.asarray(layouts)
    reference = np.asarray(truth)
    if not is_strict_permutation(control) or not is_strict_permutation(reference):
        raise ValueError("control and truth must be strict permutations")
    result = np.zeros((len(valid), 3), dtype=np.int16)
    baseline = np.array(
        (
            true_pair_count(control, reference, grid_size=grid_size),
            exact_count(control, reference),
            radius_two_count(control, reference, grid_size=grid_size),
        ),
        dtype=np.int16,
    )
    for index in range(1, len(valid)):
        if not bool(valid[index]):
            continue
        length = int(lengths[index])
        cycle = tuple(int(value) for value in positions[index, :length])
        candidate = apply_cycle(control, cycle)
        metrics = np.array(
            (
                true_pair_count(candidate, reference, grid_size=grid_size),
                exact_count(candidate, reference),
                radius_two_count(candidate, reference, grid_size=grid_size),
            ),
            dtype=np.int16,
        )
        result[index] = metrics - baseline
    return result


def exhaustive_short_cycles(tile_count: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate every unique 2-cycle and both orientations of every 3-cycle."""

    cycles: list[tuple[int, ...]] = []
    cycles.extend(tuple(pair) for pair in itertools.combinations(range(tile_count), 2))
    for triple in itertools.combinations(range(tile_count), 3):
        cycles.append(canonical_cycle(triple))
        cycles.append(canonical_cycle((triple[0], triple[2], triple[1])))
    return tuple(cycles)


def proposal_oracle_diagnostic(
    control: np.ndarray,
    truth: np.ndarray,
    bank_positions: np.ndarray,
    bank_lengths: np.ndarray,
    bank_valid: np.ndarray,
    *,
    grid_size: int,
) -> OracleDiagnostic:
    """Compare a frozen target-free bank against the complete 2/3-cycle oracle."""

    bank_deltas = metric_deltas_for_bank(
        control,
        truth,
        bank_positions,
        bank_lengths,
        bank_valid,
        grid_size=grid_size,
    )
    exhaustive_best = 0
    baseline_pairs = true_pair_count(control, truth, grid_size=grid_size)
    for cycle in exhaustive_short_cycles(len(control)):
        candidate = apply_cycle(control, cycle)
        delta = true_pair_count(candidate, truth, grid_size=grid_size) - baseline_pairs
        exhaustive_best = max(exhaustive_best, delta)
    proposal_best_index = int(np.argmax(bank_deltas[:, 0]))
    proposal_best_pair = int(bank_deltas[proposal_best_index, 0])
    return OracleDiagnostic(
        exhaustive_has_benefit=exhaustive_best > 0,
        proposal_has_benefit=proposal_best_pair > 0,
        exhaustive_best_pair_delta=exhaustive_best,
        proposal_best_pair_delta=proposal_best_pair,
        proposal_best_exact_delta=int(bank_deltas[proposal_best_index, 1]),
        proposal_count=int(np.count_nonzero(bank_valid)),
    )


def aggregate_oracle_diagnostics(
    diagnostics: list[OracleDiagnostic] | tuple[OracleDiagnostic, ...],
) -> dict[str, int | float | None]:
    """Aggregate recall only over states where a short-cycle fix exists.

    States with no beneficial exhaustive 2/3-cycle are reported but excluded
    from the coverage denominator.  This prevents trivially unfixable states
    from inflating proposal-oracle recall.
    """

    denominator = sum(item.exhaustive_has_benefit for item in diagnostics)
    covered = sum(
        item.exhaustive_has_benefit and item.proposal_has_benefit
        for item in diagnostics
    )
    return {
        "state_count": len(diagnostics),
        "beneficial_exhaustive_state_count": denominator,
        "beneficial_proposal_state_count": covered,
        "proposal_oracle_coverage": None if denominator == 0 else covered / denominator,
    }


def frozen_positive_action_mask(metric_deltas: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return all lexicographically best non-worsening actions, KEEP on failure."""

    deltas = np.asarray(metric_deltas)
    valid_values = np.asarray(valid, dtype=bool)
    if deltas.shape != (len(valid_values), 3) or not valid_values[0]:
        raise ValueError("metric deltas/valid mask are malformed or KEEP is absent")
    eligible = np.flatnonzero(valid_values & (deltas[:, 0] >= 0))
    if len(eligible) == 0:
        eligible = np.array([0])
    best_key = max(tuple(int(value) for value in deltas[index]) for index in eligible)
    positive = np.zeros_like(valid_values)
    for index in eligible:
        positive[index] = tuple(int(value) for value in deltas[index]) == best_key
    if not positive.any():
        positive[0] = True
    return positive


def model_static_ledger(model: BasinCycleStageB, *, batch_size: int) -> dict[str, Any]:
    """Return the reviewed analytic forward-MAC/activation ledger for 6x6."""

    n = model.tile_count
    pixels = TILE_SIZE * TILE_SIZE
    c = model.feature_channels
    r = model.retrieval_dim
    d = model.state_dim
    p = model.proposal_cap
    image_block_count = len(model.image_blocks)
    state_block_count = len(model.state_blocks)
    edge_count = 2 * model.grid_size * (model.grid_size - 1)
    per_board = {
        "image_stem": n * pixels * 10 * c * 9,
        "image_blocks": image_block_count * n * pixels * (c * 9 + c * c),
        "side_query_key": n * SIDE_COUNT * TILE_SIZE * c * r * 2,
        "pair_dot_products": 2 * n * n * TILE_SIZE * r,
        "boundary_head": n * SIDE_COUNT * TILE_SIZE * c * 6,
        "tile_projection": n * c * d,
        "state_stem": n * (d + 8) * d,
        "state_blocks": state_block_count * n * (d * 9 + d * d),
        "action_head": p * ((5 * d + model.scalar_feature_count) * 128 + 128 * 64 + 64 * 11),
    }
    total = sum(per_board.values())
    return {
        "trainable_parameters": model.trainable_parameter_count(),
        "forward_learned_macs_per_board": total,
        "forward_learned_macs_per_batch": total * batch_size,
        "per_board": per_board,
        "nonlearned_state_edge_values": p * edge_count,
        "pair_logit_values": 2 * n * n,
        "proposal_layout_values": p * n,
        "deep_feature_values_per_board": n * c * pixels,
        "scope": (
            "analytic multiply-accumulate count; excludes fixed preprocessing, "
            "normalization, activation, top-k CPU closure and backward"
        ),
    }
