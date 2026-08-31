"""Metadata-only protocol scaffold for selector-aligned BasinCycle v2.

The module has no filesystem or image access.  It can deterministically split
an already supplied list of organizer-train filenames after a caller supplies
the complete required named exclusion inventory.  The checked-in JSON remains
unsigned, and this no-runner module deliberately refuses every execution
mapping rather than pretending to authenticate a future signature.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

UNSIGNED_SCHEMA = "aiijc-basincycle-selector-aligned-v2-unsigned-template"
UNSIGNED_STATUS = "unsigned-design-data-blocked"
SELECTION_NAMESPACE = "aiijc-basincycle-selector-aligned-v2-fit128-cal32-confirm32"
SELECTION_SEED = 20261001
PLAN_SEED = 20261002
FIT_SOURCE_COUNT = 128
CAL_SOURCE_COUNT = 32
CONFIRM_SOURCE_COUNT = 32
FIT_UPDATES = 3_000
BATCH_SIZE = 4
EVALUATION_DRAWS = (0, 1)
GRID_SIZE = 6
PARENT_GRID_SIZE = 24
REQUIRED_EXCLUSION_GROUP_COUNTS: dict[str, int | None] = {
    "stage_b_v1_fit64": 64,
    "stage_b_v1_opened_eval32": 32,
    "socket_v2_train1024": 1024,
    "socket_v2_opened_eval32": 32,
    "active_joint_fit256": 256,
    "active_joint_reserved_dev64": 64,
    "protected_adapter3200_terminal16": 16,
    "post_design_opened_or_reserved": None,
}

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
class SelectorAlignedRoster:
    """Three mutually source-disjoint organizer-train filename groups."""

    fit_filenames: tuple[str, ...]
    calibration_filenames: tuple[str, ...]
    confirmation_filenames: tuple[str, ...]


@dataclass(frozen=True)
class SelectorAlignedPlanRow:
    """One immutable source/crop/state/pixel decision."""

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


def canonical_digest(value: Any) -> str:
    """Hash JSON-compatible metadata using a single canonical encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def filenames_digest(filenames: Sequence[str]) -> str:
    """Hash an ordered filename sequence without reading any named file."""

    return hashlib.sha256("\n".join(filenames).encode("utf-8")).hexdigest()


def _validated_names(
    filenames: Sequence[str],
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(filenames, (str, bytes)):
        raise ValueError(f"{field} must be a filename sequence")
    values = tuple(filenames)
    if (not values and not allow_empty) or not all(
        isinstance(item, str) and item for item in values
    ):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate filenames")
    return values


def _selection_key(filename: str) -> tuple[bytes, str]:
    payload = f"{SELECTION_NAMESPACE}\0{SELECTION_SEED}\0{filename}".encode()
    return hashlib.sha256(payload).digest(), filename


def _validated_exclusion_provenance(
    organizer_train_filenames: Sequence[str],
    exclusion_groups: Mapping[str, Sequence[str]],
) -> tuple[set[str], dict[str, Any]]:
    source_names = _validated_names(organizer_train_filenames, field="organizer train")
    if not isinstance(exclusion_groups, Mapping):
        raise ValueError("exclusion groups must be a named mapping")
    required_names = set(REQUIRED_EXCLUSION_GROUP_COUNTS)
    observed_names = set(exclusion_groups)
    if observed_names != required_names:
        missing = sorted(required_names - observed_names)
        unexpected = sorted(observed_names - required_names)
        raise ValueError(
            f"exclusion-group inventory is incomplete: missing={missing}, unexpected={unexpected}"
        )

    manifest_set = set(source_names)
    union: set[str] = set()
    named_metadata: dict[str, Any] = {}
    for name, expected_count in REQUIRED_EXCLUSION_GROUP_COUNTS.items():
        values = _validated_names(
            exclusion_groups[name],
            field=f"exclusion group {name}",
            allow_empty=expected_count is None,
        )
        if expected_count is not None and len(values) != expected_count:
            raise ValueError(f"exclusion group {name} must contain {expected_count} filenames")
        outside_manifest = sorted(set(values) - manifest_set)
        if outside_manifest:
            raise ValueError(f"exclusion group {name} contains filenames outside organizer train")
        union.update(values)
        named_metadata[name] = {
            "count": len(values),
            "ordered_filenames_newline_sha256": filenames_digest(values),
        }
    sorted_union = tuple(sorted(union))
    provenance_payload = {
        "required_group_counts": REQUIRED_EXCLUSION_GROUP_COUNTS,
        "named_groups": named_metadata,
        "deduplicated_union_count": len(sorted_union),
        "deduplicated_union_sorted_newline_sha256": filenames_digest(sorted_union),
        "all_exclusions_belong_to_organizer_train": True,
    }
    metadata = {
        **provenance_payload,
        "provenance_sha256": canonical_digest(provenance_payload),
    }
    return union, metadata


def exclusion_provenance_metadata(
    organizer_train_filenames: Sequence[str],
    *,
    exclusion_groups: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Validate the complete named inventory and hash its exact deduplicated union."""

    _, metadata = _validated_exclusion_provenance(
        organizer_train_filenames,
        exclusion_groups,
    )
    return metadata


def select_source_disjoint_roster(
    organizer_train_filenames: Sequence[str],
    *,
    exclusion_groups: Mapping[str, Sequence[str]],
) -> SelectorAlignedRoster:
    """Select fit/calibration/confirmation together before any pixel access."""

    source_names = _validated_names(organizer_train_filenames, field="organizer train")
    excluded, _ = _validated_exclusion_provenance(source_names, exclusion_groups)
    eligible = sorted((name for name in source_names if name not in excluded), key=_selection_key)
    required = FIT_SOURCE_COUNT + CAL_SOURCE_COUNT + CONFIRM_SOURCE_COUNT
    if len(eligible) < required:
        raise ValueError(f"only {len(eligible)} eligible sources remain, requested {required}")
    selected = tuple(eligible[:required])
    fit_end = FIT_SOURCE_COUNT
    cal_end = fit_end + CAL_SOURCE_COUNT
    roster = SelectorAlignedRoster(
        fit_filenames=selected[:fit_end],
        calibration_filenames=selected[fit_end:cal_end],
        confirmation_filenames=selected[cal_end:],
    )
    validate_roster(roster, excluded_filenames=excluded)
    return roster


def validate_roster(
    roster: SelectorAlignedRoster,
    *,
    excluded_filenames: Sequence[str],
) -> None:
    """Fail if any phase overlaps another phase or the exclusion union."""

    fit = _validated_names(roster.fit_filenames, field="fit roster")
    calibration = _validated_names(roster.calibration_filenames, field="calibration roster")
    confirmation = _validated_names(roster.confirmation_filenames, field="confirmation roster")
    if len(fit) != FIT_SOURCE_COUNT:
        raise ValueError(f"fit roster must contain {FIT_SOURCE_COUNT} sources")
    if len(calibration) != CAL_SOURCE_COUNT:
        raise ValueError(f"calibration roster must contain {CAL_SOURCE_COUNT} sources")
    if len(confirmation) != CONFIRM_SOURCE_COUNT:
        raise ValueError(f"confirmation roster must contain {CONFIRM_SOURCE_COUNT} sources")
    if set(fit) & set(calibration) or set(fit) & set(confirmation):
        raise ValueError("fit roster overlaps an evaluation phase")
    if set(calibration) & set(confirmation):
        raise ValueError("calibration and confirmation rosters overlap")
    excluded = set(excluded_filenames)
    if (set(fit) | set(calibration) | set(confirmation)) & excluded:
        raise ValueError("selected roster overlaps the exclusion union")


def roster_metadata(roster: SelectorAlignedRoster) -> dict[str, Any]:
    """Return the exact counts/digests a future signed config must bind."""

    return {
        "fit": {
            "count": len(roster.fit_filenames),
            "ordered_filenames_newline_sha256": filenames_digest(roster.fit_filenames),
        },
        "calibration": {
            "count": len(roster.calibration_filenames),
            "ordered_filenames_newline_sha256": filenames_digest(roster.calibration_filenames),
        },
        "confirmation": {
            "count": len(roster.confirmation_filenames),
            "ordered_filenames_newline_sha256": filenames_digest(roster.confirmation_filenames),
        },
        "joint_count": (
            len(roster.fit_filenames)
            + len(roster.calibration_filenames)
            + len(roster.confirmation_filenames)
        ),
    }


def future_binding_metadata(
    organizer_train_filenames: Sequence[str],
    *,
    exclusion_groups: Mapping[str, Sequence[str]],
    roster: SelectorAlignedRoster,
) -> dict[str, Any]:
    """Build metadata a future runner must independently reproduce and bind.

    This function authenticates no file or execution transition.  It only
    makes the required manifest-name, exclusion, roster and plan hashes
    explicit for a future separately audited runner.
    """

    source_names = _validated_names(organizer_train_filenames, field="organizer train")
    expected_roster = select_source_disjoint_roster(
        source_names,
        exclusion_groups=exclusion_groups,
    )
    if roster != expected_roster:
        raise ValueError("roster does not match the deterministic joint selection")
    exclusion_metadata = exclusion_provenance_metadata(
        source_names,
        exclusion_groups=exclusion_groups,
    )
    fit_rows = fit_plan(roster.fit_filenames)
    calibration_rows = evaluation_plan(
        roster.calibration_filenames,
        phase="calibration",
    )
    confirmation_rows = evaluation_plan(
        roster.confirmation_filenames,
        phase="confirmation",
    )
    payload = {
        "organizer_train": {
            "count": len(source_names),
            "ordered_filenames_newline_sha256": filenames_digest(source_names),
        },
        "selection": {
            "namespace": SELECTION_NAMESPACE,
            "seed": SELECTION_SEED,
        },
        "exclusions": exclusion_metadata,
        "roster": roster_metadata(roster),
        "plans": {
            "fit": {"row_count": len(fit_rows), "sha256": plan_digest(fit_rows)},
            "calibration": {
                "row_count": len(calibration_rows),
                "sha256": plan_digest(calibration_rows),
            },
            "confirmation": {
                "row_count": len(confirmation_rows),
                "sha256": plan_digest(confirmation_rows),
            },
        },
    }
    return {**payload, "binding_metadata_sha256": canonical_digest(payload)}


def _random_plan_row(
    rng: np.random.Generator,
    *,
    phase: str,
    step_or_source: int,
    batch_slot_or_draw: int,
    filename: str,
    state_family: str,
) -> SelectorAlignedPlanRow:
    if state_family == "solver_replay":
        state_recipe = "frozen_socket_v2_grid6_decoder_control"
        severity = 0
    elif state_family == "procedural":
        state_recipe = str(rng.choice(PROCEDURAL_KINDS, p=PROCEDURAL_WEIGHTS))
        severity = int(rng.choice(SEVERITIES, p=SEVERITY_WEIGHTS))
    else:
        raise ValueError("unknown state family")
    return SelectorAlignedPlanRow(
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


def fit_plan(fit_filenames: Sequence[str]) -> tuple[SelectorAlignedPlanRow, ...]:
    """Build the final-endpoint-only 3,000 x four FIT schedule."""

    filenames = _validated_names(fit_filenames, field="fit roster")
    if len(filenames) != FIT_SOURCE_COUNT:
        raise ValueError(f"fit roster must contain {FIT_SOURCE_COUNT} sources")
    rng = np.random.default_rng(PLAN_SEED)
    total = FIT_UPDATES * BATCH_SIZE
    families = np.asarray(["solver_replay"] * (total // 2) + ["procedural"] * (total // 2))
    rng.shuffle(families)
    source_schedule: list[int] = []
    while len(source_schedule) < total:
        source_schedule.extend(int(index) for index in rng.permutation(FIT_SOURCE_COUNT))
    source_schedule = source_schedule[:total]
    rows: list[SelectorAlignedPlanRow] = []
    family_index = 0
    for step in range(FIT_UPDATES):
        source_indices = source_schedule[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        for batch_slot, source_index in enumerate(source_indices):
            rows.append(
                _random_plan_row(
                    rng,
                    phase="fit",
                    step_or_source=step,
                    batch_slot_or_draw=batch_slot,
                    filename=filenames[int(source_index)],
                    state_family=str(families[family_index]),
                )
            )
            family_index += 1
    return tuple(rows)


def evaluation_plan(
    filenames: Sequence[str],
    *,
    phase: str,
) -> tuple[SelectorAlignedPlanRow, ...]:
    """Build two source-clustered draws, one replay and one procedural."""

    if phase not in {"calibration", "confirmation"}:
        raise ValueError("evaluation phase must be calibration or confirmation")
    values = _validated_names(filenames, field=f"{phase} roster")
    expected_count = CAL_SOURCE_COUNT if phase == "calibration" else CONFIRM_SOURCE_COUNT
    if len(values) != expected_count:
        raise ValueError(f"{phase} roster must contain {expected_count} sources")
    phase_offset = 1 if phase == "calibration" else 2
    rng = np.random.default_rng(PLAN_SEED + phase_offset)
    rows: list[SelectorAlignedPlanRow] = []
    for source_index, filename in enumerate(values):
        for draw_index in EVALUATION_DRAWS:
            rows.append(
                _random_plan_row(
                    rng,
                    phase=phase,
                    step_or_source=source_index,
                    batch_slot_or_draw=draw_index,
                    filename=filename,
                    state_family=("solver_replay" if draw_index == 0 else "procedural"),
                )
            )
    return tuple(rows)


def plan_digest(rows: Sequence[SelectorAlignedPlanRow]) -> str:
    """Hash an ordered plan without reading pixels or labels."""

    return canonical_digest([asdict(row) for row in rows])


def require_signed_execution(config: Mapping[str, Any]) -> None:
    """Unconditionally deny execution in this no-runner unsigned scaffold."""

    del config
    raise PermissionError(
        "this no-runner scaffold cannot authenticate execution; "
        "a future separately audited runner must implement the firewall"
    )


__all__ = [
    "BATCH_SIZE",
    "CAL_SOURCE_COUNT",
    "CONFIRM_SOURCE_COUNT",
    "EVALUATION_DRAWS",
    "FIT_SOURCE_COUNT",
    "FIT_UPDATES",
    "PLAN_SEED",
    "REQUIRED_EXCLUSION_GROUP_COUNTS",
    "SELECTION_NAMESPACE",
    "SELECTION_SEED",
    "SelectorAlignedPlanRow",
    "SelectorAlignedRoster",
    "UNSIGNED_SCHEMA",
    "UNSIGNED_STATUS",
    "canonical_digest",
    "evaluation_plan",
    "exclusion_provenance_metadata",
    "filenames_digest",
    "fit_plan",
    "future_binding_metadata",
    "plan_digest",
    "require_signed_execution",
    "roster_metadata",
    "select_source_disjoint_roster",
    "validate_roster",
]
