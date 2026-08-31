"""Mechanical MPS all-masked reduction fix for BasinCycle Stage B v3.

The reviewed Stage-B action features use padded KEEP/cycle proposal banks.
KEEP and padding entries have empty position/changed-edge masks by design.  On
MPS, reducing an all-``-inf`` maximum or all-``+inf`` minimum can save an
invalid ``-1`` arg index and fail later in backward.  This adapter replaces
only those sentinels with the finite dtype extrema and explicitly maps empty
reductions to zero.  Non-empty values and gradients are unchanged.
"""

from __future__ import annotations

import torch

import aiijc_puzzle.basincycle_stage_b as stage_b
import aiijc_puzzle.basincycle_stage_b_runner as stage_b_runner
from aiijc_puzzle.basincycle_stage_b import ProposalBank
from aiijc_puzzle.basincycle_stage_b_mps_transfer_v2 import install_mps_transfer_v2

_ORIGINAL_RUNNER_MODEL = stage_b_runner.BasinCycleStageB


def finite_masked_max_or_zero(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    """Reduce a mask with a finite floor and return zero for an empty mask."""

    expanded_mask = mask.expand_as(values)
    floor = torch.finfo(values.dtype).min
    reduced = values.masked_fill(~expanded_mask, floor).max(dim=dim).values
    has_any = expanded_mask.any(dim=dim)
    return torch.where(has_any, reduced, torch.zeros_like(reduced))


def finite_masked_min_or_zero(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    """Reduce a mask with a finite ceiling and return zero for an empty mask."""

    expanded_mask = mask.expand_as(values)
    ceiling = torch.finfo(values.dtype).max
    reduced = values.masked_fill(~expanded_mask, ceiling).min(dim=dim).values
    has_any = expanded_mask.any(dim=dim)
    return torch.where(has_any, reduced, torch.zeros_like(reduced))


class BasinCycleStageBMPSReductionsV3(stage_b.BasinCycleStageB):
    """Byte-compatible Stage-B model with safe padded reductions on MPS."""

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
            torch.arange(length, device=candidates.device)[None, None, :] < bank.lengths[..., None]
        )
        divisor = position_mask.sum(dim=-1, keepdim=True).clamp_min(1)

        gathered_context = context[batch, safe_positions]
        context_mean = (gathered_context * position_mask[..., None]).sum(dim=2) / divisor
        context_max = finite_masked_max_or_zero(
            gathered_context,
            position_mask[..., None],
            dim=2,
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
        minimum = finite_masked_min_or_zero(edge_delta, changed, dim=-1)
        maximum = finite_masked_max_or_zero(edge_delta, changed, dim=-1)
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


def install_mps_reductions_v3() -> None:
    """Install the v2 transfer and v3 runner model override, idempotently."""

    install_mps_transfer_v2()
    current = stage_b_runner.BasinCycleStageB
    if current is BasinCycleStageBMPSReductionsV3:
        return
    if current is not _ORIGINAL_RUNNER_MODEL:
        raise RuntimeError("Stage-B runner model was already replaced")
    stage_b_runner.BasinCycleStageB = BasinCycleStageBMPSReductionsV3


__all__ = [
    "BasinCycleStageBMPSReductionsV3",
    "finite_masked_max_or_zero",
    "finite_masked_min_or_zero",
    "install_mps_reductions_v3",
]
