from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from aiijc_puzzle.basincycle_stage_b_protocol import (
    BATCH_SIZE,
    EVAL_DRAWS,
    EVAL_SOURCE_COUNT,
    FIT_SOURCE_COUNT,
    FIT_UPDATES,
    PLAN_SEED,
    SCHEMA,
    SELECTION_NAMESPACE,
    SELECTION_SEED,
    SIGNED_STATUS,
    eval_plan,
    exclusion_digest,
    fit_plan,
    names_digest,
    plan_digest,
    relevant_exclusion_groups,
    require_target_free_freeze_receipt,
    select_stage_b_roster,
    validate_frozen_inputs,
    validate_frozen_roster_and_plans,
)
from aiijc_puzzle.protocol import compute_protocol_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = PROJECT_ROOT / "configs/basincycle_stage_b_6x6_preregistered_v1.json"


def _record(name: str) -> dict[str, str]:
    return {"filename": name, "input_sha256": "1" * 64, "target_sha256": "2" * 64}


def _metadata_fixture() -> tuple[dict, dict, dict, dict]:
    names = iter(f"source_{index:05d}.png" for index in range(2_000))
    train = [next(names) for _ in range(1_024)]
    opened_eval = [next(names) for _ in range(32)]
    active_fit = [next(names) for _ in range(256)]
    active_dev = [next(names) for _ in range(64)]
    protected = [next(names) for _ in range(16)]
    available = [next(names) for _ in range(300)]
    manifest = {"splits": {"train": [_record(name) for name in [
        *train,
        *opened_eval,
        *active_fit,
        *active_dev,
        *protected,
        *available,
    ]]}}
    manifest["protocol_digest"] = compute_protocol_digest(manifest)
    socket = {
        "selection": {
            "train_filenames": train,
            "train_digest": names_digest(train),
            "eval_filenames": opened_eval,
            "eval_digest": names_digest(opened_eval),
        }
    }
    active = {
        "source_protocol": {
            "fit_filenames": active_fit,
            "fit_digest": names_digest(active_fit),
            "reserved_dev_filenames": active_dev,
            "reserved_dev_digest": names_digest(active_dev),
        }
    }
    protected_audit = {
        "protected_terminal16": {
            "source_filenames": protected,
            "ordered_filenames_newline_sha256": names_digest(protected),
        }
    }
    return manifest, socket, active, protected_audit


def _signed_config() -> tuple[dict, tuple[dict, dict, dict, dict]]:
    fixture = _metadata_fixture()
    manifest, socket, active, protected = fixture
    groups = relevant_exclusion_groups(socket, active, protected)
    excluded = sorted(set().union(*(set(values) for values in groups.values())))
    fit, evaluation = select_stage_b_roster(manifest, excluded_filenames=excluded)
    fit_rows = fit_plan(fit)
    eval_rows = eval_plan(evaluation)
    config = {
        "schema": SCHEMA,
        "status": SIGNED_STATUS,
        "source_protocol": {
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_seed": SELECTION_SEED,
            "fit_source_count": FIT_SOURCE_COUNT,
            "fit_filenames": list(fit),
            "fit_digest": names_digest(fit),
            "eval_source_count": EVAL_SOURCE_COUNT,
            "eval_filenames": list(evaluation),
            "eval_digest": names_digest(evaluation),
            "exclusion_count": len(excluded),
            "exclusion_digest": exclusion_digest(groups),
        },
        "corruption_plan": {
            "plan_seed": PLAN_SEED,
            "fit_updates": FIT_UPDATES,
            "batch_size": BATCH_SIZE,
            "fit_row_count": len(fit_rows),
            "fit_plan_digest": plan_digest(fit_rows),
            "eval_draw_indices": list(EVAL_DRAWS),
            "eval_row_count": len(eval_rows),
            "eval_plan_digest": plan_digest(eval_rows),
            "solver_replay_fraction": 0.5,
            "procedural_fraction": 0.5,
        },
    }
    return config, fixture


def test_metadata_only_roster_and_complete_corruption_schedule_are_exact() -> None:
    config, fixture = _signed_config()
    audit = validate_frozen_roster_and_plans(config, *fixture)
    assert audit["pixels_or_labels_opened"] is False
    assert audit["excluded_count"] == 1_392
    assert len(config["source_protocol"]["fit_filenames"]) == 64
    assert len(config["source_protocol"]["eval_filenames"]) == 32
    assert not set(config["source_protocol"]["fit_filenames"]) & set(
        config["source_protocol"]["eval_filenames"]
    )
    assert len(fit_plan(config["source_protocol"]["fit_filenames"])) == 8_000
    assert len(eval_plan(config["source_protocol"]["eval_filenames"])) == 64


def test_roster_or_plan_drift_fails_closed() -> None:
    config, fixture = _signed_config()
    changed = copy.deepcopy(config)
    changed["source_protocol"]["fit_filenames"][0] = changed["source_protocol"][
        "eval_filenames"
    ][0]
    with pytest.raises(ValueError, match="deterministic selection"):
        validate_frozen_roster_and_plans(changed, *fixture)

    changed = copy.deepcopy(config)
    changed["corruption_plan"]["fit_plan_digest"] = "0" * 64
    with pytest.raises(ValueError, match="fit_plan_digest"):
        validate_frozen_roster_and_plans(changed, *fixture)


def test_reference_cannot_open_before_target_free_identity_receipt() -> None:
    receipt = {
        "schema": "aiijc-basincycle-stage-b-target-free-freeze-v1",
        "config_sha256": "a" * 64,
        "reference_opened": False,
        "all_controls_strict": True,
        "all_banks_keep_index0": True,
        "all_candidate_layouts_strict": True,
        "eval_case_count": 64,
        "model_sha256": "b" * 64,
        "prediction_roster_sha256": "c" * 64,
        "proposal_identity_sha256": "d" * 64,
        "control_layout_sha256": "e" * 64,
    }
    require_target_free_freeze_receipt(receipt, config_sha256="a" * 64)
    changed = dict(receipt, reference_opened=True)
    with pytest.raises(ValueError, match="reference_opened"):
        require_target_free_freeze_receipt(changed, config_sha256="a" * 64)


def test_real_signed_config_is_hash_bound_and_metadata_reconstructible() -> None:
    config = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    sidecar = REAL_CONFIG.with_suffix(".json.sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(REAL_CONFIG.read_bytes()).hexdigest() == sidecar
    assert len(validate_frozen_inputs(config, project_root=PROJECT_ROOT)) >= 10

    manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    socket = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["socket_parent_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    active = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["active_scale_config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    protected = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["protected_roster_audit"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    audit = validate_frozen_roster_and_plans(config, manifest, socket, active, protected)
    assert audit["pixels_or_labels_opened"] is False
