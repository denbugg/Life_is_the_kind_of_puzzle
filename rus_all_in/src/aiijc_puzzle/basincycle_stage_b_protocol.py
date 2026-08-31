"""Metadata-only roster and corruption freeze for BasinCycle Stage B.

Nothing in this module opens an image, target, model, prediction, DEV panel, or
competition artifact.  It deterministically reserves source-disjoint
organizer-train filenames and hashes every crop/state/corruption decision that
a later, separately authorised 6x6 run would consume.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aiijc_puzzle.protocol import compute_protocol_digest
from aiijc_puzzle.synthetic_socket_evaluation import (
    names_digest,
    select_source_disjoint_train_records,
)

SCHEMA = "aiijc-basincycle-stage-b-6x6-preregistered-v1"
SIGNED_STATUS = "signed-blocked-awaiting-review-no-data-access"
SELECTION_NAMESPACE = "aiijc-basincycle-stage-b-6x6-fit64-eval32-v1"
SELECTION_SEED = 20260921
FIT_SOURCE_COUNT = 64
EVAL_SOURCE_COUNT = 32
FIT_UPDATES = 2_000
BATCH_SIZE = 4
EVAL_DRAWS = (0, 1)
PLAN_SEED = 20260922
GRID_SIZE = 6
PARENT_GRID_SIZE = 24

PROCEDURAL_KINDS = (
    "short_tile_cycle",
    "congruent_patch_cycle",
    "wrong_edge_weld_cycle",
    "band_cyclic_roll",
    "whole_board_roll",
)
PROCEDURAL_WEIGHTS = np.asarray((0.30, 0.25, 0.20, 0.15, 0.10), dtype=np.float64)
SEVERITIES = np.asarray((1, 2, 4, 8), dtype=np.int64)
SEVERITY_WEIGHTS = np.asarray((0.35, 0.30, 0.20, 0.15), dtype=np.float64)
PIXEL_RECIPES = (
    "gaussian_poisson",
    "gaussian_blur",
    "motion_blur",
    "jpeg_ringing",
    "scale_bias_chroma",
    "edge_erosion",
    "mixed_two_stage",
)


@dataclass(frozen=True)
class StageBPlanRow:
    """One immutable source/crop/state/corruption decision."""

    phase: str
    step_or_source: int
    batch_slot_or_draw: int
    source_filename: str
    crop_tile_row: int
    crop_tile_col: int
    state_family: str
    state_recipe: str
    severity: int
    state_seed: int
    pixel_recipe: str
    pixel_seed: int


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_inputs(
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, str]:
    """Hash-check every implementation and metadata input before data access."""

    artifacts = config.get("frozen_inputs")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("Stage-B config has no frozen input inventory")
    observed: dict[str, str] = {}
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"frozen input {name} is not a mapping")
        path_value = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str):
            raise ValueError(f"frozen input {name} lacks path/hash")
        path = (project_root / path_value).resolve()
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input hash mismatch: {name}")
        observed[str(name)] = actual
    return observed


def plan_digest(rows: Sequence[StageBPlanRow]) -> str:
    """Hash ordered plan rows using canonical JSON."""

    return _canonical_digest([asdict(row) for row in rows])


def _validated_filenames(value: Any, *, field: str, count: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field} must be a list of non-empty filenames")
    result = tuple(value)
    if len(result) != count or len(set(result)) != count:
        raise ValueError(f"{field} must contain exactly {count} unique filenames")
    return result


def relevant_exclusion_groups(
    socket_report: Mapping[str, Any],
    active_scale_config: Mapping[str, Any],
    protected_roster_audit: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return exact source groups relevant to the frozen warm-start lineage."""

    selection = socket_report.get("selection")
    source_protocol = active_scale_config.get("source_protocol")
    protected_terminal = protected_roster_audit.get("protected_terminal16")
    if (
        not isinstance(selection, Mapping)
        or not isinstance(source_protocol, Mapping)
        or not isinstance(protected_terminal, Mapping)
    ):
        raise ValueError("parent reports do not expose the required source metadata")
    groups = {
        "socket_v2_train1024": _validated_filenames(
            selection.get("train_filenames"),
            field="socket train_filenames",
            count=1024,
        ),
        "socket_v2_opened_eval32": _validated_filenames(
            selection.get("eval_filenames"),
            field="socket eval_filenames",
            count=32,
        ),
        "active_joint_fit256": _validated_filenames(
            source_protocol.get("fit_filenames"),
            field="active joint fit_filenames",
            count=256,
        ),
        "active_joint_reserved_dev64": _validated_filenames(
            source_protocol.get("reserved_dev_filenames"),
            field="active joint reserved_dev_filenames",
            count=64,
        ),
        "protected_adapter3200_terminal16": _validated_filenames(
            protected_terminal.get("source_filenames"),
            field="protected adapter3200 terminal filenames",
            count=16,
        ),
    }
    expected_digests = {
        "socket_v2_train1024": selection.get("train_digest"),
        "socket_v2_opened_eval32": selection.get("eval_digest"),
        "active_joint_fit256": source_protocol.get("fit_digest"),
        "active_joint_reserved_dev64": source_protocol.get("reserved_dev_digest"),
        "protected_adapter3200_terminal16": protected_terminal.get(
            "ordered_filenames_newline_sha256"
        ),
    }
    for name, filenames in groups.items():
        if names_digest(filenames) != expected_digests[name]:
            raise ValueError(f"parent source digest mismatch for {name}")
    return groups


def select_stage_b_roster(
    manifest: Mapping[str, Any],
    *,
    excluded_filenames: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select the exact fit64/eval32 organizer-train roster from metadata."""

    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest protocol digest is invalid")
    selected = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=excluded_filenames,
        limit=FIT_SOURCE_COUNT + EVAL_SOURCE_COUNT,
        seed=SELECTION_SEED,
        namespace=SELECTION_NAMESPACE,
    )
    names = tuple(str(record["filename"]) for record in selected)
    return names[:FIT_SOURCE_COUNT], names[FIT_SOURCE_COUNT:]


def exclusion_digest(groups: Mapping[str, Sequence[str]]) -> str:
    """Hash named exclusion groups without collapsing their provenance."""

    payload = {
        name: {
            "count": len(filenames),
            "ordered_digest": names_digest(tuple(filenames)),
        }
        for name, filenames in sorted(groups.items())
    }
    return _canonical_digest(payload)


def _random_plan_row(
    rng: np.random.Generator,
    *,
    phase: str,
    step_or_source: int,
    batch_slot_or_draw: int,
    filename: str,
    state_family: str,
) -> StageBPlanRow:
    if state_family == "solver_replay":
        state_recipe = "frozen_socket_v2_grid6_decoder_control"
        severity = 0
    elif state_family == "procedural":
        state_recipe = str(rng.choice(PROCEDURAL_KINDS, p=PROCEDURAL_WEIGHTS))
        severity = int(rng.choice(SEVERITIES, p=SEVERITY_WEIGHTS))
    else:
        raise ValueError("unknown state family")
    return StageBPlanRow(
        phase=phase,
        step_or_source=step_or_source,
        batch_slot_or_draw=batch_slot_or_draw,
        source_filename=filename,
        crop_tile_row=int(rng.integers(0, PARENT_GRID_SIZE - GRID_SIZE + 1)),
        crop_tile_col=int(rng.integers(0, PARENT_GRID_SIZE - GRID_SIZE + 1)),
        state_family=state_family,
        state_recipe=state_recipe,
        severity=severity,
        state_seed=int(rng.integers(0, 2**31)),
        pixel_recipe=str(rng.choice(PIXEL_RECIPES)),
        pixel_seed=int(rng.integers(0, 2**31)),
    )


def fit_plan(fit_filenames: Sequence[str]) -> tuple[StageBPlanRow, ...]:
    """Return the complete 2,000-update x batch-four training schedule."""

    filenames = tuple(fit_filenames)
    if len(filenames) != FIT_SOURCE_COUNT or len(set(filenames)) != len(filenames):
        raise ValueError("fit roster must contain 64 unique sources")
    rng = np.random.default_rng(PLAN_SEED)
    total = FIT_UPDATES * BATCH_SIZE
    modes = np.array(["solver_replay"] * (total // 2) + ["procedural"] * (total // 2))
    rng.shuffle(modes)
    rows: list[StageBPlanRow] = []
    mode_index = 0
    for step in range(FIT_UPDATES):
        source_indices = rng.choice(FIT_SOURCE_COUNT, size=BATCH_SIZE, replace=False)
        for batch_slot, source_index in enumerate(source_indices):
            rows.append(
                _random_plan_row(
                    rng,
                    phase="fit",
                    step_or_source=step,
                    batch_slot_or_draw=batch_slot,
                    filename=filenames[int(source_index)],
                    state_family=str(modes[mode_index]),
                )
            )
            mode_index += 1
    return tuple(rows)


def eval_plan(eval_filenames: Sequence[str]) -> tuple[StageBPlanRow, ...]:
    """Return two fixed source-clustered draws per reserved eval source."""

    filenames = tuple(eval_filenames)
    if len(filenames) != EVAL_SOURCE_COUNT or len(set(filenames)) != len(filenames):
        raise ValueError("eval roster must contain 32 unique sources")
    rng = np.random.default_rng(PLAN_SEED + 1)
    rows: list[StageBPlanRow] = []
    for source_index, filename in enumerate(filenames):
        for draw_index in EVAL_DRAWS:
            rows.append(
                _random_plan_row(
                    rng,
                    phase="eval",
                    step_or_source=source_index,
                    batch_slot_or_draw=draw_index,
                    filename=filename,
                    state_family="solver_replay" if draw_index == 0 else "procedural",
                )
            )
    return tuple(rows)


def validate_frozen_roster_and_plans(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    socket_report: Mapping[str, Any],
    active_scale_config: Mapping[str, Any],
    protected_roster_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct all metadata decisions and fail closed on any drift."""

    if config.get("schema") != SCHEMA or config.get("status") != SIGNED_STATUS:
        raise ValueError("Stage-B config is not the reviewed signed blocked schema")
    source = config.get("source_protocol")
    plan = config.get("corruption_plan")
    if not isinstance(source, Mapping) or not isinstance(plan, Mapping):
        raise ValueError("Stage-B config lacks source/corruption protocol")
    groups = relevant_exclusion_groups(
        socket_report,
        active_scale_config,
        protected_roster_audit,
    )
    excluded = tuple(sorted(set().union(*(set(values) for values in groups.values()))))
    fit, evaluation = select_stage_b_roster(manifest, excluded_filenames=excluded)
    configured_fit = _validated_filenames(
        source.get("fit_filenames"),
        field="fit_filenames",
        count=FIT_SOURCE_COUNT,
    )
    configured_eval = _validated_filenames(
        source.get("eval_filenames"),
        field="eval_filenames",
        count=EVAL_SOURCE_COUNT,
    )
    expected_scalars = {
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_seed": SELECTION_SEED,
        "fit_source_count": FIT_SOURCE_COUNT,
        "eval_source_count": EVAL_SOURCE_COUNT,
        "exclusion_count": len(excluded),
        "exclusion_digest": exclusion_digest(groups),
        "fit_digest": names_digest(fit),
        "eval_digest": names_digest(evaluation),
    }
    for key, expected in expected_scalars.items():
        if source.get(key) != expected:
            raise ValueError(f"frozen source field changed: {key}")
    if configured_fit != fit or configured_eval != evaluation:
        raise ValueError("configured Stage-B roster differs from deterministic selection")
    if set(fit) & set(evaluation) or (set(fit) | set(evaluation)) & set(excluded):
        raise ValueError("Stage-B source-disjointness invariant failed")

    fit_rows = fit_plan(fit)
    eval_rows = eval_plan(evaluation)
    expected_plan = {
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
    }
    for key, expected in expected_plan.items():
        if plan.get(key) != expected:
            raise ValueError(f"frozen corruption-plan field changed: {key}")
    return {
        "excluded_count": len(excluded),
        "excluded_digest": exclusion_digest(groups),
        "fit_digest": names_digest(fit),
        "eval_digest": names_digest(evaluation),
        "fit_plan_digest": plan_digest(fit_rows),
        "eval_plan_digest": plan_digest(eval_rows),
        "pixels_or_labels_opened": False,
    }


def require_target_free_freeze_receipt(
    receipt: Mapping[str, Any],
    *,
    config_sha256: str,
) -> None:
    """Guard the future transition from prediction freeze to oracle scoring."""

    expected = {
        "schema": "aiijc-basincycle-stage-b-target-free-freeze-v1",
        "config_sha256": config_sha256,
        "reference_opened": False,
        "all_controls_strict": True,
        "all_banks_keep_index0": True,
        "all_candidate_layouts_strict": True,
        "eval_case_count": EVAL_SOURCE_COUNT * len(EVAL_DRAWS),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"target-free freeze receipt failed: {key}")
    for key in (
        "model_sha256",
        "prediction_roster_sha256",
        "proposal_identity_sha256",
        "control_layout_sha256",
    ):
        value = receipt.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"target-free freeze receipt lacks {key}")
