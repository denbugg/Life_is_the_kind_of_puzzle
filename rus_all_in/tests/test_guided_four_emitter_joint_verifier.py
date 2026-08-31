from __future__ import annotations

import numpy as np
import pytest
import torch

from aiijc_puzzle.guided_four_emitter_joint_verifier import (
    EXTENDED_SLOT_WIDTH,
    LEGACY_SLOT_WIDTH,
    GuidedFourEmitterJointVerifier,
    build_target_free_four_emitter_case,
    four_emitter_joint_contract,
    transplant_legacy_joint_state,
)
from aiijc_puzzle.guided_fourth_emitter import extend_with_guided_emitter
from aiijc_puzzle.joint_reciprocal_tri_emitter_verifier import (
    JointReciprocalTriEmitterVerifier,
    exact_joint_targets,
    joint_assignment_loss,
)
from aiijc_puzzle.tri_emitter_edge_verifier import EMITTERS, TOP_K, build_candidate_pool


def _ranked_matrix(count: int, offset: int) -> np.ndarray:
    matrix = np.full((count, count), -count, dtype=np.float32)
    for source in range(count):
        order = [
            (source + offset + step) % count
            for step in range(count)
            if (source + offset + step) % count != source
        ]
        matrix[source, order] = -np.arange(len(order), dtype=np.float32)
        matrix[source, source] = -1e4
    return matrix


def _target_free_arrays(count: int = 40) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    old_scores = _ranked_matrix(count, 1)
    legacy_pool = build_candidate_pool(
        {emitter: (old_scores, old_scores.copy()) for emitter in EMITTERS},
        top_k=TOP_K,
    )
    guided_scores = _ranked_matrix(count, 33)
    guided_pool = extend_with_guided_emitter(
        legacy_pool, (guided_scores, guided_scores.copy())
    )
    generator = np.random.default_rng(8)
    legacy = {
        "raw_sides": generator.normal(size=(4, count, 5, 6)).astype(np.float32),
        "dino_sides": generator.normal(size=(4, count, 5, 16)).astype(np.float32),
        "candidates": legacy_pool.candidates,
        "valid": legacy_pool.valid,
        "auxiliary": legacy_pool.auxiliary,
        "raw_baseline": legacy_pool.raw_baseline,
        "emitter_topk": legacy_pool.emitter_topk,
    }
    sidecar = {
        "candidates": guided_pool.candidates,
        "valid": guided_pool.valid,
        "legacy_slot": guided_pool.legacy_slot,
        "guided_auxiliary": guided_pool.guided_auxiliary,
        "guided_baseline": guided_pool.guided_baseline,
        "emitter_topk": guided_pool.emitter_topk,
        "legacy_identity_digest_ascii": np.frombuffer(
            guided_pool.legacy_identity_digest.encode(), dtype=np.uint8
        ),
        "identity_digest_ascii": np.frombuffer(
            guided_pool.identity_digest.encode(), dtype=np.uint8
        ),
    }
    return legacy, sidecar


def _axis_tensors(case: object, axis: int) -> dict[str, torch.Tensor]:
    return {
        "raw_sides": torch.from_numpy(case.raw_sides),
        "dino_sides": torch.from_numpy(case.dino_sides),
        "candidates": torch.from_numpy(case.candidates[axis]).long(),
        "valid": torch.from_numpy(case.valid[axis]),
        "legacy_slot": torch.from_numpy(case.legacy_slot[axis]).long(),
        "legacy_auxiliary": torch.from_numpy(case.legacy_auxiliary[axis]),
        "legacy_raw_baseline": torch.from_numpy(case.legacy_raw_baseline[axis]),
        "guided_auxiliary": torch.from_numpy(case.guided_auxiliary[axis]),
        "guided_baseline": torch.from_numpy(case.guided_baseline[axis]),
    }


def test_target_free_consumer_preserves_legacy_slots_and_rejects_labels() -> None:
    legacy, sidecar = _target_free_arrays()
    case = build_target_free_four_emitter_case(legacy, sidecar)
    assert case.candidates.shape == (2, 40, EXTENDED_SLOT_WIDTH)
    np.testing.assert_array_equal(
        case.candidates[..., :LEGACY_SLOT_WIDTH], legacy["candidates"]
    )
    np.testing.assert_array_equal(
        case.legacy_auxiliary[..., :LEGACY_SLOT_WIDTH, :], legacy["auxiliary"]
    )
    np.testing.assert_array_equal(
        case.legacy_raw_baseline[..., :LEGACY_SLOT_WIDTH], legacy["raw_baseline"]
    )
    assert np.all(case.legacy_raw_baseline[..., LEGACY_SLOT_WIDTH:] == -1e4)
    with pytest.raises(ValueError, match="exactly seven target-free"):
        build_target_free_four_emitter_case(
            {**legacy, "target_slots": np.zeros((2, 40), dtype=np.int16)}, sidecar
        )


def test_zero_initialisation_replays_hybrid_baseline_and_has_finite_gradients() -> None:
    legacy, sidecar = _target_free_arrays()
    case = build_target_free_four_emitter_case(legacy, sidecar)
    tensors = _axis_tensors(case, 0)
    model = GuidedFourEmitterJointVerifier(width=4, hidden=8, guided_width=4)
    output = model(**tensors, direction=0)
    hybrid = torch.where(
        tensors["legacy_slot"] >= 0,
        tensors["legacy_raw_baseline"],
        tensors["guided_baseline"],
    )
    torch.testing.assert_close(output.edge_logits[tensors["valid"]], hybrid[tensors["valid"]])

    truth = torch.full((40,), -1, dtype=torch.long)
    used: set[int] = set()
    appended = 0
    for source in range(40):
        slots = torch.nonzero(
            tensors["valid"][source]
            & (torch.arange(EXTENDED_SLOT_WIDTH) >= LEGACY_SLOT_WIDTH),
            as_tuple=False,
        ).flatten()
        for slot in slots:
            target = int(tensors["candidates"][source, slot])
            if target not in used:
                truth[source] = target
                used.add(target)
                appended += 1
                break
    assert appended >= 10
    targets = exact_joint_targets(tensors["candidates"], tensors["valid"], truth)
    assert int((targets.row_slots >= LEGACY_SLOT_WIDTH).sum()) == appended
    loss = joint_assignment_loss(output, targets, tensors["valid"])
    assert torch.isfinite(loss.total)
    loss.total.backward()
    bad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
    ]
    assert bad == []


def test_legacy_joint_state_transplant_keeps_old_edge_logits_exact() -> None:
    torch.manual_seed(3)
    legacy_arrays, sidecar = _target_free_arrays()
    case = build_target_free_four_emitter_case(legacy_arrays, sidecar)
    tensors = _axis_tensors(case, 1)
    legacy_model = JointReciprocalTriEmitterVerifier(width=4, hidden=8)
    with torch.no_grad():
        legacy_model.edge_verifier.head[-1].weight.normal_(std=0.02)
        legacy_model.edge_verifier.head[-1].bias.fill_(0.1)
    new_model = GuidedFourEmitterJointVerifier(width=4, hidden=8, guided_width=4)
    result = transplant_legacy_joint_state(new_model, legacy_model.state_dict())
    assert result["legacy_key_count"] > 0 and result["new_guided_key_count"] > 0

    old_output = legacy_model(
        tensors["raw_sides"],
        tensors["dino_sides"],
        tensors["candidates"][:, :LEGACY_SLOT_WIDTH],
        tensors["valid"][:, :LEGACY_SLOT_WIDTH],
        tensors["legacy_auxiliary"][:, :LEGACY_SLOT_WIDTH],
        tensors["legacy_raw_baseline"][:, :LEGACY_SLOT_WIDTH],
        direction=1,
    )
    new_output = new_model(**tensors, direction=1)
    mask = tensors["valid"][:, :LEGACY_SLOT_WIDTH]
    torch.testing.assert_close(
        new_output.edge_logits[:, :LEGACY_SLOT_WIDTH][mask],
        old_output.edge_logits[mask],
    )
    assert torch.count_nonzero(new_model.guided_residual.network[-1].weight) == 0
    assert torch.count_nonzero(new_model.guided_residual.network[-1].bias) == 0


def test_contract_has_fixed_128_slots_none_and_no_threshold_sweep() -> None:
    contract = four_emitter_joint_contract(
        GuidedFourEmitterJointVerifier(width=4, hidden=8, guided_width=4)
    )
    assert contract["candidate_slots"]["total"] == 128
    assert contract["legacy_path"]["raw_baseline_retained"] is True
    assert contract["legacy_path"]["learned_relation_residual_retained"] is True
    assert contract["guided_path"]["auxiliary_dim"] == 7
    assert contract["joint_assignment"]["learned_row_none_per_axis"] is True
    assert contract["joint_assignment"]["learned_column_none_per_axis"] is True
    assert contract["deployment_head"]["threshold_sweep"] is False
    assert contract["real_protocol_signed"] is False
