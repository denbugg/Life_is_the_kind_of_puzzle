"""Mechanical MPS transfer fix for the BasinCycle Stage-B proposal closure.

The signed Stage-B model detaches pair logits before deterministic CPU proposal
enumeration.  PyTorch 2.13 on MPS can corrupt a fused ``MPS float32 -> CPU
float64`` transfer.  This adapter preserves the original float32 values during
the device transfer; the unchanged v1 builder then performs its float64 cast
on CPU.
"""

from __future__ import annotations

import torch

import aiijc_puzzle.basincycle_stage_b as stage_b
from aiijc_puzzle.basincycle_stage_b import ProposalBank

_ORIGINAL_PROPOSAL_BUILDER = stage_b.build_target_free_proposal_bank


def stage_pair_logits_on_cpu(pair_logits: torch.Tensor) -> torch.Tensor:
    """Detach and copy to CPU without combining device and dtype conversion."""

    staged = pair_logits.detach().to(device="cpu")
    if staged.dtype != pair_logits.dtype:
        raise RuntimeError("staged proposal transfer changed pair-logit dtype")
    if not torch.isfinite(staged).all():
        raise ValueError("pair logits became non-finite during staged CPU transfer")
    return staged


def build_target_free_proposal_bank_mps_transfer_v2(
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
    """Run the unchanged target-free builder after a safe detached CPU copy."""

    staged_pair_logits = stage_pair_logits_on_cpu(pair_logits)
    return _ORIGINAL_PROPOSAL_BUILDER(
        staged_pair_logits,
        layouts,
        grid_size=grid_size,
        top_k=top_k,
        candidate_cap=candidate_cap,
        seed_count=seed_count,
        max_cycle_length=max_cycle_length,
        proposal_cap=proposal_cap,
    )


def install_mps_transfer_v2() -> None:
    """Install exactly one reviewed proposal-transfer override, idempotently."""

    current = stage_b.build_target_free_proposal_bank
    if current is build_target_free_proposal_bank_mps_transfer_v2:
        return
    if current is not _ORIGINAL_PROPOSAL_BUILDER:
        raise RuntimeError("Stage-B proposal builder was already replaced")
    stage_b.build_target_free_proposal_bank = (
        build_target_free_proposal_bank_mps_transfer_v2
    )


__all__ = [
    "build_target_free_proposal_bank_mps_transfer_v2",
    "install_mps_transfer_v2",
    "stage_pair_logits_on_cpu",
]
