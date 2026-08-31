from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

import aiijc_puzzle.basincycle_stage_b as stage_b
import aiijc_puzzle.basincycle_stage_b_runner as stage_b_runner
from aiijc_puzzle.basincycle_stage_b_mps_transfer_v2 import (
    build_target_free_proposal_bank_mps_transfer_v2,
    install_mps_transfer_v2,
    stage_pair_logits_on_cpu,
)
from aiijc_puzzle.basincycle_stage_b_runner import audit_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/basincycle_stage_b_6x6_preregistered_v1.json"
BINDING_PATH = PROJECT_ROOT / "configs/basincycle_stage_b_execution_binding_v2.json"


def _finite_pair_logits() -> torch.Tensor:
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(2, 2, 36, 36, generator=generator)
    diagonal = torch.eye(36, dtype=torch.bool)[None, None]
    return values.masked_fill(diagonal, -1.0e4)


def _layouts() -> torch.Tensor:
    return torch.stack((torch.arange(36), torch.arange(35, -1, -1)))


def _bank_signature(bank: stage_b.ProposalBank) -> tuple[np.ndarray, ...]:
    return tuple(
        value.detach().cpu().numpy()
        for value in (bank.positions, bank.lengths, bank.valid)
    )


def test_staged_cpu_transfer_preserves_finite_float32_values_exactly() -> None:
    pair_logits = _finite_pair_logits().requires_grad_(True)
    staged = stage_pair_logits_on_cpu(pair_logits)
    assert staged.device.type == "cpu"
    assert staged.dtype == torch.float32
    assert not staged.requires_grad
    assert torch.equal(staged, pair_logits.detach())
    promoted = staged.to(dtype=torch.float64)
    assert promoted.dtype == torch.float64
    assert torch.isfinite(promoted).all()


def test_v2_builder_matches_original_cpu_proposal_bank() -> None:
    pair_logits = _finite_pair_logits()
    layouts = _layouts()
    kwargs = {
        "grid_size": 6,
        "top_k": 6,
        "candidate_cap": 4,
        "seed_count": 8,
        "max_cycle_length": 3,
        "proposal_cap": 64,
    }
    expected = stage_b.build_target_free_proposal_bank(pair_logits, layouts, **kwargs)
    observed = build_target_free_proposal_bank_mps_transfer_v2(
        pair_logits,
        layouts,
        **kwargs,
    )
    for expected_value, observed_value in zip(
        _bank_signature(expected),
        _bank_signature(observed),
        strict=True,
    ):
        assert np.array_equal(expected_value, observed_value)


def test_install_is_exact_idempotent_override(monkeypatch: pytest.MonkeyPatch) -> None:
    original = stage_b.build_target_free_proposal_bank
    monkeypatch.setattr(stage_b, "build_target_free_proposal_bank", original)
    install_mps_transfer_v2()
    assert (
        stage_b.build_target_free_proposal_bank
        is build_target_free_proposal_bank_mps_transfer_v2
    )
    install_mps_transfer_v2()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_transfer_stages_float32_before_cpu_float64() -> None:
    pair_logits = _finite_pair_logits().to(device="mps")
    staged = stage_pair_logits_on_cpu(pair_logits)
    assert staged.device.type == "cpu"
    assert staged.dtype == torch.float32
    assert torch.isfinite(staged).all()
    promoted = staged.to(dtype=torch.float64)
    assert torch.isfinite(promoted).all()
    assert float(promoted.abs().max()) == 1.0e4


def test_v2_binding_preserves_scientific_config_and_failed_v1_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scientific_sha = "133587c2e0257c206b8d81009e7ba2addfb6bd48a167527c0e9771334df05b91"
    failed_log_sha = "fd13ba6b618683601d6f8cfc76302b3e5ddc06514e434dc14ed951488d49b717"
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == scientific_sha
    monkeypatch.setattr(
        stage_b_runner,
        "EXECUTION_BINDING_SCHEMA",
        "aiijc-basincycle-stage-b-execution-binding-v2",
    )
    audit = audit_protocol(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        binding_path=BINDING_PATH,
    )
    assert audit["scientific_config_sha256"] == scientific_sha
    assert audit["binding_hashes"]["failed_v1_log"] == failed_log_sha
    assert audit["organizer_pixels_opened"] is False
    assert audit["organizer_labels_opened"] is False
