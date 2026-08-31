from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import aiijc_puzzle.basincycle_stage_b_runner as runner
from aiijc_puzzle.basincycle_stage_b import BasinCycleStageB
from aiijc_puzzle.basincycle_stage_b_protocol import StageBPlanRow
from aiijc_puzzle.basincycle_stage_b_runner import (
    FREEZE_RECEIPT_SCHEMA,
    TARGET_FREE_REFERENCE_SEMANTICS,
    VisibleCase,
    _freeze_array_inventory,
    atomic_save_npz,
    audit_protocol,
    clean_boundary_targets,
    corrupt_clean_tiles,
    directional_edge_targets,
    iter_prefetched_case_batches,
    labels_for_output,
    pair_loss_labels,
    procedural_control,
    score_frozen_predictions,
    sha256_arrays,
    source_clustered_mean_ci,
    truth_and_tile_order,
    validate_execution_acknowledgement,
    validate_freeze_bundle,
    validate_frozen_array_semantics,
)
from aiijc_puzzle.basincycle_synthetic import apply_cycle, is_strict_permutation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/basincycle_stage_b_6x6_preregistered_v1.json"
BINDING_PATH = PROJECT_ROOT / "configs/basincycle_stage_b_execution_binding_v1.json"


def _tiny_model() -> BasinCycleStageB:
    return BasinCycleStageB(
        feature_channels=16,
        retrieval_dim=16,
        state_dim=32,
        encoder_blocks=1,
        state_blocks=1,
        proposal_top_k=4,
        proposal_candidate_cap=2,
        proposal_seed_count=3,
        proposal_cap=16,
    )


def _target_free_arrays() -> dict[str, np.ndarray]:
    case_count = 64
    proposal_count = 256
    control = np.tile(np.arange(36, dtype=np.int16), (case_count, 1))
    positions = np.full((case_count, proposal_count, 3), -1, dtype=np.int16)
    positions[:, 1, :2] = (0, 1)
    lengths = np.zeros((case_count, proposal_count), dtype=np.int8)
    lengths[:, 1] = 2
    valid = np.zeros((case_count, proposal_count), dtype=bool)
    valid[:, :2] = True
    candidates = np.repeat(control[:, None, :], proposal_count, axis=1)
    candidates[:, 1, 0], candidates[:, 1, 1] = control[:, 1], control[:, 0]
    action_logits = np.full((case_count, proposal_count), -np.inf, dtype=np.float32)
    action_logits[:, :2] = 0.0
    risk_logits = np.full((case_count, proposal_count), np.inf, dtype=np.float32)
    risk_logits[:, :2] = 0.0
    return {
        "source_index": np.repeat(np.arange(32, dtype=np.int16), 2),
        "draw_index": np.tile(np.arange(2, dtype=np.int8), 32),
        "state_family_code": np.tile(np.arange(2, dtype=np.int8), 32),
        "control_layout": control,
        "proposal_positions": positions,
        "proposal_lengths": lengths,
        "proposal_valid": valid,
        "candidate_layouts": candidates,
        "pair_logits": np.zeros((case_count, 2, 36, 36), dtype=np.float32),
        "action_logits": action_logits,
        "quantiles": np.zeros((case_count, proposal_count, 3, 3), dtype=np.float32),
        "risk_logits": risk_logits,
        "selected_index": np.zeros(case_count, dtype=np.int16),
        "selected_layout": control.copy(),
    }


def _target_free_receipt(
    arrays: dict[str, np.ndarray],
    *,
    config_sha: str,
    binding_sha: str,
    bundle_sha: str,
) -> dict[str, object]:
    return {
        "schema": FREEZE_RECEIPT_SCHEMA,
        "config_sha256": config_sha,
        "execution_binding_sha256": binding_sha,
        "reference_opened": False,
        "reference_semantics": TARGET_FREE_REFERENCE_SEMANTICS,
        "synthetic_shuffle_truth_constructed_for_case_generation": True,
        "procedural_control_initialized_from_planted_truth": True,
        "derived_procedural_control_supplied_to_model": True,
        "synthetic_shuffle_truth_supplied_directly_to_model_proposals_or_selector": False,
        "synthetic_shuffle_truth_persisted_in_bundle": False,
        "clean_pixels_supplied_directly_to_model_proposals_or_selector": False,
        "clean_pixels_persisted_in_bundle": False,
        "evaluation_metric_or_oracle_attached_before_freeze": False,
        "all_controls_strict": True,
        "all_banks_keep_index0": True,
        "all_candidate_layouts_strict": True,
        "all_selected_outputs_strict": True,
        "selection_or_threshold_sweep_performed": False,
        "eval_case_count": 64,
        "model_sha256": "c" * 64,
        "prediction_roster_sha256": sha256_arrays(
            [(name, arrays[name]) for name in _freeze_array_inventory()]
        ),
        "proposal_identity_sha256": sha256_arrays(
            [
                ("proposal_positions", arrays["proposal_positions"]),
                ("proposal_lengths", arrays["proposal_lengths"]),
                ("proposal_valid", arrays["proposal_valid"]),
            ]
        ),
        "control_layout_sha256": sha256_arrays(
            [("control_layout", arrays["control_layout"])]
        ),
        "bundle": {"sha256": bundle_sha},
    }


@pytest.mark.parametrize(
    "recipe",
    (
        "gaussian_poisson",
        "gaussian_blur",
        "motion_blur",
        "jpeg_ringing",
        "scale_bias_chroma",
        "edge_erosion",
        "mixed_two_stage",
    ),
)
def test_bound_pixel_recipes_are_deterministic_finite_and_shape_preserving(
    recipe: str,
) -> None:
    clean = np.random.default_rng(3).random((36, 20, 20, 3), dtype=np.float32)
    first = corrupt_clean_tiles(clean, recipe=recipe, seed=19)
    second = corrupt_clean_tiles(clean, recipe=recipe, seed=19)
    assert first.shape == clean.shape
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    "recipe",
    (
        "short_tile_cycle",
        "congruent_patch_cycle",
        "wrong_edge_weld_cycle",
        "band_cyclic_roll",
        "whole_board_roll",
    ),
)
def test_bound_procedural_states_are_deterministic_strict_and_nonidentity(
    recipe: str,
) -> None:
    truth, _, _ = truth_and_tile_order(71)
    first = procedural_control(
        truth,
        recipe=recipe,
        severity=4,
        rng=np.random.default_rng(91),
    )
    second = procedural_control(
        truth,
        recipe=recipe,
        severity=4,
        rng=np.random.default_rng(91),
    )
    assert is_strict_permutation(first)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, truth)


def test_clean_boundary_target_binds_rgb_and_forward_tangent_difference() -> None:
    tiles = np.zeros((36, 20, 20, 3), dtype=np.float32)
    tiles[0, :, :, 0] = np.arange(20, dtype=np.float32)[:, None]
    tiles[0, :, :, 1] = np.arange(20, dtype=np.float32)[None, :]
    target = clean_boundary_targets(tiles)
    assert target.shape == (36, 4, 20, 6)
    assert np.array_equal(target[0, 0, :, 0], np.arange(20, dtype=np.float32))
    assert np.array_equal(target[0, 0, :, 3], np.r_[0.0, np.ones(19)])
    assert np.array_equal(target[0, 2, :, 1], np.arange(20, dtype=np.float32))
    assert np.array_equal(target[0, 2, :, 4], np.r_[0.0, np.ones(19)])


def test_directional_targets_and_pair_loss_are_reference_correct() -> None:
    truth = np.arange(36)
    edges = directional_edge_targets(truth)
    assert edges.shape == (2, 36)
    assert edges[0, 0] == 1 and edges[0, 5] == -1
    assert edges[1, 0] == 6 and edges[1, 30] == -1

    control = truth.copy()
    candidates = np.stack((control, apply_cycle(control, (0, 35))))
    valid = np.array([True, True])
    losses = pair_loss_labels(control, truth, candidates, valid)
    assert losses.tolist() == [False, True]


def test_fit_label_attachment_cannot_change_frozen_proposal_membership() -> None:
    torch.manual_seed(7)
    model = _tiny_model().eval()
    tiles = torch.rand(1, 36, 3, 20, 20)
    truth = np.arange(36, dtype=np.int64)
    control = apply_cycle(truth, (0, 1, 2))
    with torch.no_grad():
        output = model(tiles, torch.from_numpy(control)[None])
    positions = output.proposal_bank.positions.clone()
    lengths = output.proposal_bank.lengths.clone()
    valid = output.proposal_bank.valid.clone()
    case = VisibleCase(
        filename="synthetic.png",
        state_family="procedural",
        tiles=tiles[0],
        control=control,
        truth=truth,
        clean_tiles_by_identity=np.zeros((36, 20, 20, 3), dtype=np.float32),
    )
    labels = labels_for_output(output, (case,))
    assert torch.equal(output.proposal_bank.positions, positions)
    assert torch.equal(output.proposal_bank.lengths, lengths)
    assert torch.equal(output.proposal_bank.valid, valid)
    assert labels.positive_actions.shape == valid.shape
    assert labels.positive_actions.any(dim=1).all()
    assert not torch.any(labels.positive_actions & ~valid)


def test_prefetch_pipeline_preserves_frozen_row_order_and_strictness() -> None:
    rows = tuple(
        StageBPlanRow(
            phase="fit",
            step_or_source=index,
            batch_slot_or_draw=index % 2,
            source_filename="synthetic.png",
            crop_tile_row=index,
            crop_tile_col=index,
            state_family="procedural",
            state_recipe="short_tile_cycle",
            severity=1,
            state_seed=100 + index,
            pixel_recipe="gaussian_blur",
            pixel_seed=200 + index,
        )
        for index in range(5)
    )
    canvas = np.zeros((24, 24, 20, 20, 3), dtype=np.uint8)
    observed = list(
        iter_prefetched_case_batches(
            rows,
            batch_size=2,
            clean_tile_canvases={"synthetic.png": canvas},
            socket=object(),  # Procedural rows never call the Socket path.
            socket_device=torch.device("cpu"),
            workers=2,
            thread_name_prefix="test-basincycle-prefetch",
        )
    )
    assert [start for start, _ in observed] == [0, 2, 4]
    flattened = [case for _, batch in observed for case in batch]
    assert [case.filename for case in flattened] == ["synthetic.png"] * 5
    assert all(is_strict_permutation(case.control) for case in flattened)


def test_source_clustered_bootstrap_retains_two_draws_and_is_deterministic() -> None:
    sources = np.repeat(np.arange(32), 2)
    values = np.repeat(np.arange(32, dtype=np.float64), 2)
    first = source_clustered_mean_ci(values, sources, seed=11, resamples=2_000)
    second = source_clustered_mean_ci(values, sources, seed=11, resamples=2_000)
    assert first == second
    assert first["mean"] == 15.5
    assert first["ci95_lower"] < first["mean"] < first["ci95_upper"]
    with pytest.raises(ValueError, match="two fixed draws"):
        source_clustered_mean_ci(values[:-1], sources[:-1], seed=11, resamples=100)


def test_target_free_bundle_roundtrip_is_hash_bound_and_pickle_free(tmp_path: Path) -> None:
    arrays = _target_free_arrays()
    bundle = tmp_path / "frozen.npz"
    bundle_sha = atomic_save_npz(bundle, arrays)
    config_sha = "a" * 64
    binding_sha = "b" * 64
    receipt = _target_free_receipt(
        arrays,
        config_sha=config_sha,
        binding_sha=binding_sha,
        bundle_sha=bundle_sha,
    )
    observed = validate_freeze_bundle(
        bundle_path=bundle,
        receipt=receipt,
        config_sha256=config_sha,
        binding_sha256=binding_sha,
    )
    assert set(observed) == set(arrays)
    assert np.array_equal(observed["candidate_layouts"], arrays["candidate_layouts"])


def test_frozen_array_semantics_rejects_selected_layout_index_disagreement() -> None:
    arrays = _target_free_arrays()
    arrays["selected_layout"][0] = arrays["candidate_layouts"][0, 1]
    with pytest.raises(ValueError, match="selected layout differs"):
        validate_frozen_array_semantics(arrays)


def test_score_rejects_array_hash_drift_before_reconstructing_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _target_free_arrays()
    config_sha = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    binding_sha = "b" * 64
    receipt = _target_free_receipt(
        arrays,
        config_sha=config_sha,
        binding_sha=binding_sha,
        bundle_sha="d" * 64,
    )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    receipt["eval_source_digest"] = config["source_protocol"]["eval_digest"]
    receipt["eval_plan_digest"] = config["corruption_plan"]["eval_plan_digest"]
    arrays["pair_logits"][0, 0, 0, 1] = 1.0

    def fail_if_truth_is_reconstructed(seed: int) -> tuple[np.ndarray, np.ndarray, object]:
        raise AssertionError(f"truth reconstructed before freeze validation: {seed}")

    monkeypatch.setattr(runner, "truth_and_tile_order", fail_if_truth_is_reconstructed)
    with pytest.raises(ValueError, match="prediction roster digest mismatch"):
        score_frozen_predictions(
            arrays=arrays,
            receipt=receipt,
            config=config,
            config_sha256=config_sha,
            binding={},
            binding_sha256=binding_sha,
        )


def test_real_binding_audit_is_metadata_only_and_scientific_config_unchanged() -> None:
    expected = "133587c2e0257c206b8d81009e7ba2addfb6bd48a167527c0e9771334df05b91"
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == expected
    audit = audit_protocol(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        binding_path=BINDING_PATH,
    )
    assert audit["organizer_pixels_opened"] is False
    assert audit["organizer_labels_opened"] is False
    assert audit["protocol"]["roster_and_plans"]["pixels_or_labels_opened"] is False

    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    acknowledgement = binding["execution"]["required_review_acknowledgement"]
    validate_execution_acknowledgement(binding, acknowledgement)
    with pytest.raises(PermissionError, match="acknowledgement"):
        validate_execution_acknowledgement(binding, "not-reviewed")
