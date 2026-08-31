from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from aiijc_puzzle.default_six_emitter_joint_verifier import (
    DEFAULT_WAVELET_SOURCE_INDICES,
    LOCAL_RANK_SOURCE_INDEX,
    SUPPLY_FEATURE_DIM,
    DefaultSixEmitterJointVerifier,
    build_target_free_default_six_case,
    default_six_contract,
    freeze_target_free_case_exclusive,
    load_frozen_target_free_case,
    parameter_counts,
    target_free_case_arrays,
    transplant_tri_v2_state,
)
from aiijc_puzzle.guided_fourth_emitter import (
    extend_with_guided_emitter,
    guided_fourth_pool_digest,
)
from aiijc_puzzle.joint_reciprocal_tri_emitter_verifier import (
    JointReciprocalTriEmitterVerifier,
    exact_joint_targets,
    joint_assignment_loss,
)
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.tri_emitter_edge_verifier import (
    EMITTERS,
    TOP_K,
    build_candidate_pool,
    candidate_pool_digest,
)


def _ranked_matrix(count: int, offset: int) -> np.ndarray:
    matrix = np.full((count, count), -1e4, dtype=np.float32)
    for source in range(count):
        order = [
            (source + offset + step) % count
            for step in range(count)
            if (source + offset + step) % count != source
        ]
        matrix[source, order] = -np.arange(len(order), dtype=np.float32)
    return matrix


def _standalone_topk(count: int, offset: int) -> np.ndarray:
    result = np.empty((2, count, TOP_K), dtype=np.int32)
    for axis in range(2):
        matrix = _ranked_matrix(count, offset + 3 * axis)
        work = matrix.copy()
        np.fill_diagonal(work, -np.inf)
        result[axis] = np.argsort(-work, axis=1, kind="stable")[:, :TOP_K]
    return result


def _add_many_to_one_collision(topk: np.ndarray) -> None:
    count = topk.shape[1]
    for axis in range(2):
        for source in range(count):
            target = axis if source != axis else axis + 1
            row = topk[axis, source]
            matches = np.flatnonzero(row == target)
            if len(matches):
                row[0], row[matches[0]] = row[matches[0]], row[0]
            else:
                row[-1] = target


def _target_free_inputs(
    count: int = 96,
    *,
    collisions: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    legacy_scores = {
        emitter: (
            _ranked_matrix(count, 1),
            _ranked_matrix(count, 4),
        )
        for emitter in EMITTERS
    }
    legacy_pool = build_candidate_pool(legacy_scores, top_k=TOP_K)
    guided_pool = extend_with_guided_emitter(
        legacy_pool,
        (_ranked_matrix(count, 33), _ranked_matrix(count, 36)),
    )
    generator = np.random.default_rng(20260924)
    legacy = {
        "raw_sides": generator.normal(size=(4, count, 20, 6)).astype(np.float16),
        "dino_sides": generator.normal(size=(4, count, 14, 16)).astype(np.float16),
        "candidates": legacy_pool.candidates,
        "valid": legacy_pool.valid,
        "auxiliary": legacy_pool.auxiliary.astype(np.float16),
        "raw_baseline": legacy_pool.raw_baseline.astype(np.float16),
        "emitter_topk": legacy_pool.emitter_topk,
    }
    guided = {
        "candidates": guided_pool.candidates,
        "valid": guided_pool.valid,
        "legacy_slot": guided_pool.legacy_slot,
        "guided_auxiliary": guided_pool.guided_auxiliary.astype(np.float16),
        "guided_baseline": guided_pool.guided_baseline.astype(np.float16),
        "emitter_topk": guided_pool.emitter_topk,
        "legacy_identity_digest_ascii": np.frombuffer(
            guided_pool.legacy_identity_digest.encode(), dtype=np.uint8
        ),
        "identity_digest_ascii": np.frombuffer(
            guided_pool.identity_digest.encode(), dtype=np.uint8
        ),
    }
    wiener = _standalone_topk(count, 65)
    if collisions:
        _add_many_to_one_collision(wiener)
    wavelet_topk = np.stack(
        (
            *guided_pool.emitter_topk,
            wiener,
            _standalone_topk(count, 49),
            _standalone_topk(count, 81),
        ),
        axis=0,
    ).astype(np.int32)
    return legacy, guided, {"emitter_topk": wavelet_topk}


def _axis_tensors(case: object, axis: int) -> dict[str, torch.Tensor]:
    return {
        "raw_sides": torch.from_numpy(case.raw_sides.astype(np.float32)),
        "dino_sides": torch.from_numpy(case.dino_sides.astype(np.float32)),
        "candidates": torch.from_numpy(case.candidates[axis]).long(),
        "valid": torch.from_numpy(case.valid[axis]),
        "legacy_slot": torch.from_numpy(case.legacy_slot[axis]).long(),
        "guided_slot": torch.from_numpy(case.guided_slot[axis]).long(),
        "legacy_auxiliary": torch.from_numpy(
            case.legacy_auxiliary[axis].astype(np.float32)
        ),
        "legacy_raw_baseline": torch.from_numpy(
            case.legacy_raw_baseline[axis].astype(np.float32)
        ),
        "guided_auxiliary": torch.from_numpy(
            case.guided_auxiliary[axis].astype(np.float32)
        ),
        "guided_baseline": torch.from_numpy(
            case.guided_baseline[axis].astype(np.float32)
        ),
        "supply_features": torch.from_numpy(
            case.supply_features[axis].astype(np.float32)
        ),
    }


def _refresh_digests(
    legacy: dict[str, np.ndarray], guided: dict[str, np.ndarray]
) -> None:
    legacy_digest = candidate_pool_digest(
        legacy["candidates"], legacy["valid"], legacy["emitter_topk"]
    )
    guided_digest = guided_fourth_pool_digest(
        guided["candidates"],
        guided["valid"],
        guided["legacy_slot"],
        guided["emitter_topk"],
    )
    guided["legacy_identity_digest_ascii"] = np.frombuffer(
        legacy_digest.encode(), dtype=np.uint8
    )
    guided["identity_digest_ascii"] = np.frombuffer(
        guided_digest.encode(), dtype=np.uint8
    )


def _relabel_inputs(
    legacy: dict[str, np.ndarray],
    guided: dict[str, np.ndarray],
    wavelet: dict[str, np.ndarray],
    order: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))

    new_legacy = {
        "raw_sides": legacy["raw_sides"][:, order].copy(),
        "dino_sides": legacy["dino_sides"][:, order].copy(),
        "candidates": legacy["candidates"][:, order].copy(),
        "valid": legacy["valid"][:, order].copy(),
        "auxiliary": legacy["auxiliary"][:, order].copy(),
        "raw_baseline": legacy["raw_baseline"][:, order].copy(),
        "emitter_topk": legacy["emitter_topk"][:, :, order].copy(),
    }
    mask = new_legacy["valid"]
    new_legacy["candidates"][mask] = inverse[new_legacy["candidates"][mask]]
    new_legacy["emitter_topk"] = inverse[new_legacy["emitter_topk"]].astype(
        np.int32
    )

    new_guided = {
        "candidates": guided["candidates"][:, order].copy(),
        "valid": guided["valid"][:, order].copy(),
        "legacy_slot": guided["legacy_slot"][:, order].copy(),
        "guided_auxiliary": guided["guided_auxiliary"][:, order].copy(),
        "guided_baseline": guided["guided_baseline"][:, order].copy(),
        "emitter_topk": guided["emitter_topk"][:, :, order].copy(),
        "legacy_identity_digest_ascii": np.empty(64, dtype=np.uint8),
        "identity_digest_ascii": np.empty(64, dtype=np.uint8),
    }
    mask = new_guided["valid"]
    new_guided["candidates"][mask] = inverse[new_guided["candidates"][mask]]
    new_guided["emitter_topk"] = inverse[new_guided["emitter_topk"]].astype(
        np.int32
    )
    _refresh_digests(new_legacy, new_guided)
    new_wavelet = {
        "emitter_topk": inverse[
            wavelet["emitter_topk"][:, :, order]
        ].astype(np.int32)
    }
    return new_legacy, new_guided, new_wavelet


def _transpose_inputs(
    legacy: dict[str, np.ndarray],
    guided: dict[str, np.ndarray],
    wavelet: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    axis_order = np.asarray([1, 0])
    side_order = np.asarray([2, 3, 0, 1])
    new_legacy = {
        "raw_sides": legacy["raw_sides"][side_order].copy(),
        "dino_sides": legacy["dino_sides"][side_order].copy(),
        "candidates": legacy["candidates"][axis_order].copy(),
        "valid": legacy["valid"][axis_order].copy(),
        "auxiliary": legacy["auxiliary"][axis_order].copy(),
        "raw_baseline": legacy["raw_baseline"][axis_order].copy(),
        "emitter_topk": legacy["emitter_topk"][:, axis_order].copy(),
    }
    new_guided = {
        "candidates": guided["candidates"][axis_order].copy(),
        "valid": guided["valid"][axis_order].copy(),
        "legacy_slot": guided["legacy_slot"][axis_order].copy(),
        "guided_auxiliary": guided["guided_auxiliary"][axis_order].copy(),
        "guided_baseline": guided["guided_baseline"][axis_order].copy(),
        "emitter_topk": guided["emitter_topk"][:, axis_order].copy(),
        "legacy_identity_digest_ascii": np.empty(64, dtype=np.uint8),
        "identity_digest_ascii": np.empty(64, dtype=np.uint8),
    }
    _refresh_digests(new_legacy, new_guided)
    return (
        new_legacy,
        new_guided,
        {"emitter_topk": wavelet["emitter_topk"][:, axis_order].copy()},
    )


def test_compact_composer_has_exact_rank_features_and_excludes_local_rank() -> None:
    legacy, guided, wavelet = _target_free_inputs()
    case = build_target_free_default_six_case(legacy, guided, wavelet)
    assert case.supply_features.shape[-1] == SUPPLY_FEATURE_DIM == 12
    assert case.candidates.shape[:2] == (2, 96)
    assert case.candidates.shape[-1] < 128
    assert np.any(case.valid & (case.guided_slot < 0))
    novel = case.valid & (case.guided_slot < 0)
    assert np.all(case.legacy_auxiliary[novel] == 0)
    assert np.all(case.guided_auxiliary[novel] == 0)
    assert np.all(case.legacy_raw_baseline[novel] == 0)
    assert np.all(case.guided_baseline[novel] == 0)
    assert DEFAULT_WAVELET_SOURCE_INDICES == (0, 1, 2, 3, 4, 6)
    assert LOCAL_RANK_SOURCE_INDEX == 5

    changed = wavelet["emitter_topk"].copy()
    changed[LOCAL_RANK_SOURCE_INDEX] = _standalone_topk(96, 17)
    without_local_rank = build_target_free_default_six_case(
        legacy, guided, {"emitter_topk": changed}
    )
    assert without_local_rank.identity_digest == case.identity_digest
    np.testing.assert_array_equal(without_local_rank.candidates, case.candidates)

    with pytest.raises(ValueError, match="exactly seven target-free"):
        build_target_free_default_six_case(
            {**legacy, "target_slots": np.zeros((2, 96), dtype=np.int16)},
            guided,
            wavelet,
        )


def test_freeze_load_is_exact_key_fail_closed_and_exclusive(tmp_path: Path) -> None:
    case = build_target_free_default_six_case(*_target_free_inputs())
    path = tmp_path / "case.npz"
    record = freeze_target_free_case_exclusive(path, case)
    assert record["identity_digest"] == case.identity_digest
    assert len(record["sha256"]) == 64
    loaded = load_frozen_target_free_case(
        path,
        expected_sha256=record["sha256"],
        expected_identity_digest=record["identity_digest"],
    )
    assert loaded.identity_digest == case.identity_digest
    np.testing.assert_array_equal(loaded.candidates, case.candidates)
    np.testing.assert_array_equal(loaded.supply_features, case.supply_features)
    with pytest.raises(FileExistsError):
        freeze_target_free_case_exclusive(path, case)

    arrays = target_free_case_arrays(case)
    arrays["target_slots"] = np.zeros((2, 96), dtype=np.int16)
    labelled = tmp_path / "labelled.npz"
    np.savez_compressed(labelled, **arrays)
    with pytest.raises(ValueError, match="archive keys changed"):
        load_frozen_target_free_case(
            labelled,
            expected_sha256=sha256_file(labelled),
            expected_identity_digest=case.identity_digest,
        )


def test_tri_replay_is_exact_and_legacy_head_never_indexes_novel_identity() -> None:
    legacy, guided, wavelet = _target_free_inputs()
    case = build_target_free_default_six_case(legacy, guided, wavelet)
    tensors = _axis_tensors(case, 0)
    torch.manual_seed(17)
    tri = JointReciprocalTriEmitterVerifier(width=4, hidden=8)
    with torch.no_grad():
        tri.edge_verifier.head[-1].weight.normal_(std=0.03)
        tri.edge_verifier.head[-1].bias.fill_(0.07)
    model = DefaultSixEmitterJointVerifier(width=4, hidden=8)
    transplant_tri_v2_state(model, tri.state_dict())

    with patch.object(
        model.edge_verifier,
        "forward",
        wraps=model.edge_verifier.forward,
    ) as legacy_forward:
        output = model(**tensors, direction=0)
    call = legacy_forward.call_args.args
    passed_candidates = call[3]
    passed_valid = call[4]
    anchors = call[2]
    expected_padding = anchors[:, None].expand_as(passed_candidates)
    assert torch.equal(passed_candidates[~passed_valid], expected_padding[~passed_valid])
    assert torch.equal(passed_valid, tensors["legacy_slot"] >= 0)

    old_output = tri(
        tensors["raw_sides"],
        tensors["dino_sides"],
        torch.from_numpy(legacy["candidates"][0]).long(),
        torch.from_numpy(legacy["valid"][0]),
        torch.from_numpy(legacy["auxiliary"][0].astype(np.float32)),
        torch.from_numpy(legacy["raw_baseline"][0].astype(np.float32)),
        direction=0,
    )
    for source in range(96):
        slots = torch.nonzero(tensors["legacy_slot"][source] >= 0).flatten()
        old_slots = tensors["legacy_slot"][source, slots]
        torch.testing.assert_close(
            output.edge_logits[source, slots],
            old_output.edge_logits[source, old_slots],
        )


def test_novel_rank_only_candidates_are_trainable_and_masking_is_finite() -> None:
    case = build_target_free_default_six_case(*_target_free_inputs(collisions=True))
    tensors = _axis_tensors(case, 1)
    model = DefaultSixEmitterJointVerifier(width=4, hidden=8)
    output = model(**tensors, direction=1)
    novel = tensors["valid"] & (tensors["guided_slot"] < 0)
    assert novel.sum() >= 20
    assert torch.isfinite(output.edge_logits[novel]).all()
    assert torch.all(output.edge_logits[~tensors["valid"]] == -1e4)
    assert torch.isneginf(output.joint_confidence[~tensors["valid"]]).all()

    truth = torch.full((96,), -1, dtype=torch.long)
    used: set[int] = set()
    for source in range(96):
        for slot in torch.nonzero(novel[source]).flatten():
            target = int(tensors["candidates"][source, slot])
            if target not in used:
                truth[source] = target
                used.add(target)
                break
    assert (truth >= 0).sum() >= 20
    targets = exact_joint_targets(tensors["candidates"], tensors["valid"], truth)
    loss = joint_assignment_loss(output, targets, tensors["valid"])
    loss.total.backward()
    assert all(parameter.grad is None for parameter in model.edge_verifier.parameters())
    bad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (parameter.grad is None or not torch.isfinite(parameter.grad).all())
    ]
    assert bad == []

    # The collision emitter nominates target zero from almost every source.
    target_one = (tensors["candidates"] == 1) & tensors["valid"]
    assert target_one.sum() >= 90
    assert output.dense_valid[:, 1].sum() >= 90


def test_composer_is_relabel_and_transpose_equivariant() -> None:
    legacy, guided, wavelet = _target_free_inputs()
    case = build_target_free_default_six_case(legacy, guided, wavelet)
    generator = np.random.default_rng(41)
    order = generator.permutation(96)
    relabelled = build_target_free_default_six_case(
        *_relabel_inputs(legacy, guided, wavelet, order)
    )
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    expected = case.candidates[:, order].copy()
    expected_mask = case.valid[:, order]
    expected[expected_mask] = inverse[expected[expected_mask]]
    np.testing.assert_array_equal(relabelled.candidates, expected)
    np.testing.assert_array_equal(relabelled.valid, expected_mask)
    np.testing.assert_array_equal(relabelled.legacy_slot, case.legacy_slot[:, order])
    np.testing.assert_array_equal(relabelled.guided_slot, case.guided_slot[:, order])
    np.testing.assert_array_equal(
        relabelled.supply_features, case.supply_features[:, order]
    )

    transposed = build_target_free_default_six_case(
        *_transpose_inputs(legacy, guided, wavelet)
    )
    np.testing.assert_array_equal(transposed.candidates, case.candidates[[1, 0]])
    np.testing.assert_array_equal(transposed.valid, case.valid[[1, 0]])
    np.testing.assert_array_equal(
        transposed.supply_features, case.supply_features[[1, 0]]
    )
    np.testing.assert_array_equal(
        transposed.raw_sides, case.raw_sides[[2, 3, 0, 1]]
    )


def test_default_model_contract_has_exact_414_trainable_parameters() -> None:
    model = DefaultSixEmitterJointVerifier()
    assert parameter_counts(model) == {"total": 41717, "trainable": 414}
    contract = default_six_contract(model)
    assert contract["parameter_counts"] == {"total": 41717, "trainable": 414}
    assert contract["local_rank"]["enabled"] is False
    assert contract["legacy_path"]["novel_identity_ever_indexed"] is False
    assert contract["wiener_haar_only_path"]["rank_membership_residual_only"] is True
    assert contract["supply_features"]["direct_score_fusion"] is False
