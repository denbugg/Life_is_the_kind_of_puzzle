from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import names_digest
from scripts import run_joint_reciprocal_tri_emitter_real as runner


def _cache_arrays(count: int = 4) -> dict[str, np.ndarray]:
    width = count - 1
    candidates = np.empty((2, count, width), dtype=np.int32)
    for axis in range(2):
        for source in range(count):
            candidates[axis, source] = [target for target in range(count) if target != source]
    valid = np.ones_like(candidates, dtype=bool)
    target_slots = np.full((2, count), -1, dtype=np.int16)
    target_slots[:, :-1] = 0
    return {
        "raw_sides": np.zeros((4, count, 20, 6), dtype=np.float16),
        "dino_sides": np.zeros((4, count, 14, 16), dtype=np.float16),
        "candidates": candidates,
        "valid": valid,
        "auxiliary": np.zeros((2, count, width, 19), dtype=np.float16),
        "raw_baseline": np.zeros((2, count, width), dtype=np.float16),
        "emitter_topk": np.stack((candidates, candidates, candidates)),
        "target_slots": target_slots,
    }


def _source_config() -> dict[str, object]:
    fit = ["fit_a.png", "fit_b.png"]
    dev = ["dev_a.png", "dev_b.png"]
    return {
        "source_protocol": {
            "fit_filenames": fit,
            "fit_digest": names_digest(fit),
            "dev_filenames": dev,
            "dev_digest": names_digest(dev),
            "opened_local16_owned_filenames": ["opened.png"],
            "terminal16_owned_filenames": ["terminal.png"],
            "source_audit_excluded_filenames": ["prior.png"],
        }
    }


def _gate_metrics() -> dict[str, object]:
    raw = {
        "right_r1": 0.20,
        "right_r5": 0.40,
        "down_r1": 0.20,
        "down_r5": 0.40,
        "pooled_r1": 0.20,
        "pooled_r5": 0.40,
    }
    learned = {
        "right_r1": 0.206,
        "right_r5": 0.401,
        "down_r1": 0.206,
        "down_r5": 0.401,
        "pooled_r1": 0.206,
        "pooled_r5": 0.401,
    }

    def heads(right: float, down: float) -> dict[str, object]:
        return {
            "right": {"precision": right, "coverage_complete": True},
            "down": {"precision": down, "coverage_complete": True},
            "pooled": {
                "precision": (right + down) / 2,
                "coverage_complete": True,
            },
        }

    return {
        "retrieval": {"raw_d64_ot": raw, "joint_reciprocal": learned},
        "fixed_5_percent_reciprocal_head": {
            "raw_d64_ot": heads(0.80, 0.80),
            "joint_reciprocal": heads(0.83, 0.83),
        },
        "union": {
            "identities_unchanged": True,
            "raw_top32_preserved": True,
            "coverage_nonregression": True,
        },
    }


def _gate_thresholds() -> dict[str, float]:
    return {
        "pooled_r1_gain_minimum": 0.005,
        "pooled_r5_gain_minimum": 0.0,
        "pooled_fixed_head_precision_gain_minimum": 0.02,
        "per_axis_gain_minimum": 0.0,
    }


def _write_freeze(panel: Path, config_sha: str) -> None:
    panel.mkdir()
    archive = panel / runner.FREEZE_ARCHIVE
    with archive.open("wb") as stream:
        np.savez_compressed(stream, sentinel=np.arange(3))
    metadata = panel / runner.FREEZE_METADATA
    metadata.write_text(
        json.dumps(
            {
                "schema": "aiijc-joint-reciprocal-target-free-dev-v1",
                "config_sha256": config_sha,
                "contains_exact_references_or_labels": False,
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    freeze = {
        "schema": "aiijc-joint-reciprocal-pre-score-freeze-v1",
        "created_before_exact_reference_scoring": True,
        "contains_exact_references_or_labels": False,
        "config_sha256": config_sha,
        "artifacts": {
            "archive": {"sha256": sha256_file(archive)},
            "metadata": {"sha256": sha256_file(metadata)},
        },
    }
    (panel / runner.PRE_SCORE_FREEZE).write_text(json.dumps(freeze), encoding="utf-8")


def test_fit_cache_schema_accepts_prior_shape_and_rejects_invalid_slot() -> None:
    arrays = _cache_arrays()
    runner.validate_fit_cache_arrays(arrays, expected_tile_count=4)
    arrays["valid"][0, 0, 0] = False
    with pytest.raises(RuntimeError, match="invalid candidate slot"):
        runner.validate_fit_cache_arrays(arrays, expected_tile_count=4)


def test_fit_cache_schema_rejects_key_dtype_and_shape_drift() -> None:
    arrays = _cache_arrays()
    arrays["unexpected"] = np.zeros(1)
    with pytest.raises(RuntimeError, match="keys changed"):
        runner.validate_fit_cache_arrays(arrays, expected_tile_count=4)
    arrays = _cache_arrays()
    arrays["valid"] = arrays["valid"].astype(np.uint8)
    with pytest.raises(RuntimeError, match="valid mask"):
        runner.validate_fit_cache_arrays(arrays, expected_tile_count=4)
    arrays = _cache_arrays()
    arrays["dino_sides"] = arrays["dino_sides"][:, :, :-1]
    with pytest.raises(RuntimeError, match="dino_sides shape"):
        runner.validate_fit_cache_arrays(arrays, expected_tile_count=4)


def test_source_rosters_are_disjoint_and_refuse_owned_dev_sources() -> None:
    rosters = runner.validate_source_rosters(_source_config())
    assert rosters["fit"] == ("fit_a.png", "fit_b.png")
    config = _source_config()
    source = config["source_protocol"]
    source["dev_filenames"] = ["opened.png"]
    source["dev_digest"] = names_digest(source["dev_filenames"])
    with pytest.raises(RuntimeError, match="opened-local/terminal/audited"):
        runner.validate_source_rosters(config)
    config = _source_config()
    source = config["source_protocol"]
    source["dev_filenames"] = ["fit_a.png"]
    source["dev_digest"] = names_digest(source["dev_filenames"])
    with pytest.raises(RuntimeError, match="FIT and DEV"):
        runner.validate_source_rosters(config)


def test_unsigned_template_is_explicitly_blocked() -> None:
    config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    runner._require_exact_contract(config)
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        runner._load_signed_config(runner.DEFAULT_CONFIG)


def test_audited_optimizer_update_contract_cannot_silently_shrink() -> None:
    config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["training"]["optimizer_updates"] = 192
    with pytest.raises(RuntimeError, match="optimizer update count"):
        runner._require_exact_contract(config)


def test_capacity_or_non_final_checkpoint_is_refused() -> None:
    valid = {
        "schema": runner.CHECKPOINT_SCHEMA,
        "capacity_only_not_reusable_for_real_fit": False,
        "training": {
            "from_scratch": True,
            "checkpoint_selection": "single-final-endpoint-no-selection",
        },
    }
    runner._validate_real_checkpoint_payload(valid, Path("fit/endpoint.pt"))
    capacity = {**valid, "capacity_only_not_reusable_for_real_fit": True}
    with pytest.raises(RuntimeError, match="capacity-only"):
        runner._validate_real_checkpoint_payload(capacity, Path("fit/endpoint.pt"))
    with pytest.raises(RuntimeError, match="capacity checkpoint"):
        runner._validate_real_checkpoint_payload(valid, Path("outputs/capacity4x4/model.pt"))
    selected = {**valid, "training": {**valid["training"], "checkpoint_selection": "best"}}
    with pytest.raises(RuntimeError, match="selection contract"):
        runner._validate_real_checkpoint_payload(selected, Path("fit/endpoint.pt"))


def test_audited_cache_schedule_has_exact_updates_and_is_deterministic() -> None:
    first = runner.fixed_cache_schedule(case_count=64, optimizer_updates=1752, seed=20260913)
    second = runner.fixed_cache_schedule(case_count=64, optimizer_updates=1752, seed=20260913)
    assert first == second
    assert len(first) == 1752
    assert set(first[:64]) == set(range(64))
    assert set(first[64:128]) == set(range(64))
    assert min(first) == 0 and max(first) == 63


def test_target_free_case_matches_dirty_input_without_reference_construction() -> None:
    clean = np.random.default_rng(4).integers(0, 256, size=(4, 20, 20, 3), dtype=np.uint8)
    target_free = runner.make_target_free_synthetic_case(
        clean, source_filename="synthetic.png", draw_index=0, seed=20260908
    )
    exact_input, _ = runner.make_exact_synthetic_case(
        clean, source_filename="synthetic.png", draw_index=0, seed=20260908
    )
    assert target_free.case_id == exact_input.case_id
    assert target_free.corruption_seed == exact_input.corruption_seed
    assert target_free.permutation_seed == exact_input.permutation_seed
    np.testing.assert_array_equal(target_free.tiles, exact_input.tiles)


def test_frozen_fit_head_contains_only_target_free_selected_arrays() -> None:
    arrays = _cache_arrays()
    for axis in range(2):
        for source in range(4):
            target = (source + 1) % 4
            slot = int(np.flatnonzero(arrays["candidates"][axis, source] == target)[0])
            arrays["raw_baseline"][axis, source, slot] = 10.0
    model = runner.JointReciprocalTriEmitterVerifier(dino_dim=16, width=8, hidden=16).eval()
    with torch.no_grad():
        model.row_none_logits.fill_(-10.0)
        model.column_none_logits.fill_(-10.0)
    frozen = runner._freeze_fit_head_case(model, arrays, device=torch.device("cpu"))
    assert set(frozen) == {
        "union_identity_digest_ascii",
        "selected_sources__right",
        "selected_targets__right",
        "selected_joint_confidences__right",
        "requested_count__right",
        "reciprocal_count__right",
        "selected_sources__down",
        "selected_targets__down",
        "selected_joint_confidences__down",
        "requested_count__down",
        "reciprocal_count__down",
    }
    assert not any("target_slot" in key or "truth" in key for key in frozen)
    for axis in ("right", "down"):
        sources = frozen[f"selected_sources__{axis}"]
        targets = frozen[f"selected_targets__{axis}"]
        confidence = frozen[f"selected_joint_confidences__{axis}"]
        assert int(frozen[f"requested_count__{axis}"]) == 1
        assert len(sources) == len(targets) == len(confidence) == 1
        assert len(np.unique(sources)) == len(sources)
        assert len(np.unique(targets)) == len(targets)


def test_labels_are_loaded_only_after_target_free_hash_verification(tmp_path: Path) -> None:
    panel = tmp_path / "dev"
    config_sha = "a" * 64
    _write_freeze(panel, config_sha)
    events: list[str] = []

    def load(verified: runner.VerifiedFreeze) -> str:
        assert verified.archive_sha256 == sha256_file(verified.archive)
        events.append("labels")
        return "reference"

    def score(verified: runner.VerifiedFreeze, reference: str) -> str:
        assert verified.metadata_sha256 == sha256_file(verified.metadata)
        assert reference == "reference"
        events.append("score")
        return "done"

    assert (
        runner.score_after_verified_freeze(panel, config_sha, reference_loader=load, scorer=score)
        == "done"
    )
    assert events == ["labels", "score"]
    with (panel / runner.FREEZE_ARCHIVE).open("ab") as stream:
        stream.write(b"tamper")
    events.clear()
    with pytest.raises(RuntimeError, match="changed after freeze"):
        runner.score_after_verified_freeze(panel, config_sha, reference_loader=load, scorer=score)
    assert events == []


def test_fixed_discovery_gate_passes_all_thresholds_together() -> None:
    gate = runner.joint_discovery_gate(_gate_metrics(), _gate_thresholds())
    assert gate["passed"] is True
    assert gate["each_axis_nonnegative"] is True
    assert gate["fixed_head_coverage_complete"] is True


def test_fixed_discovery_gate_refuses_axis_loss_or_incomplete_head() -> None:
    metrics = _gate_metrics()
    metrics["retrieval"]["joint_reciprocal"]["right_r1"] = 0.199
    # Keep the pooled aggregate above +0.5 pp: the independent axis gate must fail.
    assert runner.joint_discovery_gate(metrics, _gate_thresholds())["passed"] is False
    metrics = _gate_metrics()
    metrics["fixed_5_percent_reciprocal_head"]["joint_reciprocal"]["down"]["coverage_complete"] = (
        False
    )
    assert runner.joint_discovery_gate(metrics, _gate_thresholds())["passed"] is False
    metrics = _gate_metrics()
    metrics["union"]["raw_top32_preserved"] = False
    assert runner.joint_discovery_gate(metrics, _gate_thresholds())["passed"] is False
