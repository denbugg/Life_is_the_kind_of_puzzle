from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import aiijc_puzzle.basincycle_stage_b as stage_b
import aiijc_puzzle.basincycle_stage_b_runner as stage_b_runner
from aiijc_puzzle.basincycle_stage_b_mps_reductions_v3 import (
    BasinCycleStageBMPSReductionsV3,
    finite_masked_max_or_zero,
    finite_masked_min_or_zero,
    install_mps_reductions_v3,
)
from aiijc_puzzle.basincycle_stage_b_mps_transfer_v2 import (
    build_target_free_proposal_bank_mps_transfer_v2,
)
from aiijc_puzzle.basincycle_stage_b_runner import audit_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/basincycle_stage_b_6x6_preregistered_v1.json"
BINDING_PATH = PROJECT_ROOT / "configs/basincycle_stage_b_execution_binding_v3.json"
FAILED_V1_LOG = PROJECT_ROOT / "outputs/basincycle-stage-b-minimum-6x6-v1.fit.log"
FAILED_V2_LOG = PROJECT_ROOT / "outputs/basincycle-stage-b-minimum-6x6-v2.fit.log"


def _proposal_bank(device: torch.device) -> stage_b.ProposalBank:
    positions = torch.full((2, 4, 3), -1, dtype=torch.long, device=device)
    lengths = torch.tensor([[0, 2, 3, 0], [0, 2, 3, 0]], device=device)
    valid = torch.tensor(
        [[True, True, True, False], [True, True, True, False]],
        device=device,
    )
    positions[:, 1, :2] = torch.tensor([0, 1], device=device)
    positions[:, 2, :3] = torch.tensor([2, 3, 4], device=device)
    return stage_b.ProposalBank(positions=positions, lengths=lengths, valid=valid)


def _action_inputs(
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor, stage_b.ProposalBank]:
    generator = torch.Generator().manual_seed(20260931)
    layouts = torch.stack((torch.arange(36), torch.arange(35, -1, -1))).to(device)
    bank = _proposal_bank(device)
    candidates = stage_b.materialize_candidate_layouts(layouts, bank)
    raw = (
        torch.randn(2, 36, 96, generator=generator),
        torch.randn(2, 36, 96, generator=generator),
        torch.randn(2, 36, 96, generator=generator),
        torch.randn(2, 2, 36, 36, generator=generator),
    )
    inputs = [value.to(device).requires_grad_() for value in raw]
    return inputs, candidates, bank


def _action_features_and_gradients(
    model: stage_b.BasinCycleStageB,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    inputs, candidates, bank = _action_inputs(device)
    output = model._action_features(*inputs, candidates, bank)
    gradients = torch.autograd.grad(output.sum(), inputs)
    return output.detach(), tuple(value.detach() for value in gradients)


def test_finite_reducers_match_original_cpu_values_and_gradients() -> None:
    generator = torch.Generator().manual_seed(91)
    base = torch.randn(2, 256, 60, generator=generator)
    mask = torch.rand(2, 256, 60, generator=generator) > 0.8
    mask[:, 0] = False
    mask[:, 200:] = False

    original = base.clone().requires_grad_()
    old_minimum = original.masked_fill(~mask, torch.inf).min(dim=-1).values
    old_maximum = original.masked_fill(~mask, -torch.inf).max(dim=-1).values
    old_minimum = torch.where(
        torch.isfinite(old_minimum), old_minimum, torch.zeros_like(old_minimum)
    )
    old_maximum = torch.where(
        torch.isfinite(old_maximum), old_maximum, torch.zeros_like(old_maximum)
    )
    old_gradient = torch.autograd.grad(old_minimum.sum() + old_maximum.sum(), original)[0]

    revised = base.clone().requires_grad_()
    new_minimum = finite_masked_min_or_zero(revised, mask, dim=-1)
    new_maximum = finite_masked_max_or_zero(revised, mask, dim=-1)
    new_gradient = torch.autograd.grad(new_minimum.sum() + new_maximum.sum(), revised)[0]

    assert torch.equal(new_minimum, old_minimum)
    assert torch.equal(new_maximum, old_maximum)
    assert torch.equal(new_gradient, old_gradient)
    assert torch.count_nonzero(new_gradient[~mask.any(dim=-1)]) == 0


def test_v3_action_features_match_original_cpu_forward_and_gradient() -> None:
    original = stage_b.BasinCycleStageB()
    revised = BasinCycleStageBMPSReductionsV3()
    old_output, old_gradients = _action_features_and_gradients(original, torch.device("cpu"))
    new_output, new_gradients = _action_features_and_gradients(revised, torch.device("cpu"))

    assert torch.equal(new_output, old_output)
    for new_gradient, old_gradient in zip(new_gradients, old_gradients, strict=True):
        assert torch.equal(new_gradient, old_gradient)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_v3_action_features_backward_handles_keep_and_padding_on_mps() -> None:
    model = BasinCycleStageBMPSReductionsV3().to("mps")
    output, gradients = _action_features_and_gradients(model, torch.device("mps"))
    torch.mps.synchronize()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(value).all() for value in gradients)


def test_v3_model_keeps_parameter_and_state_dict_contract() -> None:
    original = stage_b.BasinCycleStageB()
    revised = BasinCycleStageBMPSReductionsV3()
    assert revised.trainable_parameter_count() == original.trainable_parameter_count()
    assert tuple(revised.state_dict()) == tuple(original.state_dict())
    for name, value in revised.state_dict().items():
        assert value.shape == original.state_dict()[name].shape


def test_install_composes_v2_transfer_and_v3_runner_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage_b,
        "build_target_free_proposal_bank",
        stage_b.build_target_free_proposal_bank,
    )
    monkeypatch.setattr(
        stage_b_runner,
        "BasinCycleStageB",
        stage_b.BasinCycleStageB,
    )
    install_mps_reductions_v3()
    assert (
        stage_b.build_target_free_proposal_bank is build_target_free_proposal_bank_mps_transfer_v2
    )
    assert stage_b_runner.BasinCycleStageB is BasinCycleStageBMPSReductionsV3
    install_mps_reductions_v3()


def test_v3_binding_preserves_science_and_binds_both_failed_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scientific_sha = "133587c2e0257c206b8d81009e7ba2addfb6bd48a167527c0e9771334df05b91"
    failed_v1_sha = "fd13ba6b618683601d6f8cfc76302b3e5ddc06514e434dc14ed951488d49b717"
    failed_v2_sha = "405e8ea216a7e437cbdde95b5bfa14febb487fab5b699939cc717f911fdb240a"
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == scientific_sha
    assert hashlib.sha256(FAILED_V1_LOG.read_bytes()).hexdigest() == failed_v1_sha
    assert hashlib.sha256(FAILED_V2_LOG.read_bytes()).hexdigest() == failed_v2_sha
    monkeypatch.setattr(
        stage_b_runner,
        "EXECUTION_BINDING_SCHEMA",
        "aiijc-basincycle-stage-b-execution-binding-v3",
    )
    audit = audit_protocol(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        binding_path=BINDING_PATH,
    )
    assert audit["scientific_config_sha256"] == scientific_sha
    assert audit["binding_hashes"]["failed_v1_log"] == failed_v1_sha
    assert audit["binding_hashes"]["failed_v2_log"] == failed_v2_sha
    assert audit["organizer_pixels_opened"] is False
    assert audit["organizer_labels_opened"] is False

    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    assert binding["failed_v2"]["optimizer_updates_completed"] == 0
    for path in binding["artifacts"].values():
        if isinstance(path, str) and path.startswith("outputs/"):
            assert "minimum-6x6-v3" in path
            assert not (PROJECT_ROOT / path).exists()
