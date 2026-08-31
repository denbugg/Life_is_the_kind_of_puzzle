from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiijc_puzzle.basincycle_selector_aligned_v2_protocol import (
    BATCH_SIZE,
    CAL_SOURCE_COUNT,
    CONFIRM_SOURCE_COUNT,
    FIT_SOURCE_COUNT,
    FIT_UPDATES,
    PLAN_SEED,
    REQUIRED_EXCLUSION_GROUP_COUNTS,
    SELECTION_NAMESPACE,
    SELECTION_SEED,
    SelectorAlignedRoster,
    evaluation_plan,
    exclusion_provenance_metadata,
    fit_plan,
    future_binding_metadata,
    plan_digest,
    require_signed_execution,
    roster_metadata,
    select_source_disjoint_roster,
    validate_roster,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/basincycle_selector_aligned_v2_unsigned.json"


def _names(count: int) -> tuple[str, ...]:
    return tuple(f"img_{index:06d}.png" for index in range(count))


def _exclusion_groups(names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, tuple[str, ...]] = {}
    offset = 0
    for name, count in REQUIRED_EXCLUSION_GROUP_COUNTS.items():
        if count is None:
            groups[name] = ()
        else:
            groups[name] = names[offset : offset + count]
            offset += count
    return groups


def test_roster_is_deterministic_joint_and_source_disjoint() -> None:
    all_names = _names(2400)
    groups = _exclusion_groups(all_names)
    excluded = set().union(*groups.values())
    first = select_source_disjoint_roster(all_names, exclusion_groups=groups)
    second = select_source_disjoint_roster(all_names, exclusion_groups=groups)
    assert first == second
    assert len(first.fit_filenames) == FIT_SOURCE_COUNT
    assert len(first.calibration_filenames) == CAL_SOURCE_COUNT
    assert len(first.confirmation_filenames) == CONFIRM_SOURCE_COUNT
    assert not set(first.fit_filenames) & set(first.calibration_filenames)
    assert not set(first.fit_filenames) & set(first.confirmation_filenames)
    assert not set(first.calibration_filenames) & set(first.confirmation_filenames)
    assert not (
        set(first.fit_filenames)
        | set(first.calibration_filenames)
        | set(first.confirmation_filenames)
    ) & set(excluded)
    metadata = roster_metadata(first)
    assert metadata["joint_count"] == FIT_SOURCE_COUNT + CAL_SOURCE_COUNT + CONFIRM_SOURCE_COUNT
    assert len(metadata["fit"]["ordered_filenames_newline_sha256"]) == 64
    assert SELECTION_NAMESPACE.endswith("fit128-cal32-confirm32")
    assert SELECTION_SEED == 20261001
    exclusion_metadata = exclusion_provenance_metadata(
        all_names,
        exclusion_groups=groups,
    )
    assert exclusion_metadata["deduplicated_union_count"] == len(excluded)
    assert exclusion_metadata["all_exclusions_belong_to_organizer_train"] is True
    assert len(exclusion_metadata["provenance_sha256"]) == 64

    binding = future_binding_metadata(
        all_names,
        exclusion_groups=groups,
        roster=first,
    )
    assert binding["plans"]["fit"]["row_count"] == FIT_UPDATES * BATCH_SIZE
    assert binding["plans"]["calibration"]["row_count"] == CAL_SOURCE_COUNT * 2
    assert binding["plans"]["confirmation"]["row_count"] == CONFIRM_SOURCE_COUNT * 2
    assert len(binding["binding_metadata_sha256"]) == 64
    assert binding == future_binding_metadata(
        all_names,
        exclusion_groups=groups,
        roster=first,
    )
    tampered = SelectorAlignedRoster(
        fit_filenames=(first.fit_filenames[1], first.fit_filenames[0], *first.fit_filenames[2:]),
        calibration_filenames=first.calibration_filenames,
        confirmation_filenames=first.confirmation_filenames,
    )
    with pytest.raises(ValueError, match="deterministic joint selection"):
        future_binding_metadata(
            all_names,
            exclusion_groups=groups,
            roster=tampered,
        )


def test_exclusion_provenance_rejects_incomplete_or_external_groups() -> None:
    all_names = _names(2400)
    groups = _exclusion_groups(all_names)
    missing = dict(groups)
    del missing["stage_b_v1_opened_eval32"]
    with pytest.raises(ValueError, match="inventory is incomplete"):
        select_source_disjoint_roster(all_names, exclusion_groups=missing)

    external = dict(groups)
    external["stage_b_v1_fit64"] = (
        "not_in_manifest.png",
        *external["stage_b_v1_fit64"][1:],
    )
    with pytest.raises(ValueError, match="outside organizer train"):
        exclusion_provenance_metadata(all_names, exclusion_groups=external)

    wrong_count = dict(groups)
    wrong_count["active_joint_fit256"] = wrong_count["active_joint_fit256"][:-1]
    with pytest.raises(ValueError, match="must contain 256"):
        exclusion_provenance_metadata(all_names, exclusion_groups=wrong_count)


def test_roster_validation_rejects_cross_phase_overlap() -> None:
    fit = _names(FIT_SOURCE_COUNT)
    calibration = tuple(f"cal_{index:06d}.png" for index in range(CAL_SOURCE_COUNT))
    confirmation = tuple(f"confirm_{index:06d}.png" for index in range(CONFIRM_SOURCE_COUNT))
    valid = SelectorAlignedRoster(fit, calibration, confirmation)
    validate_roster(valid, excluded_filenames=("excluded.png",))
    overlapping = SelectorAlignedRoster(
        fit,
        (fit[0], *calibration[1:]),
        confirmation,
    )
    with pytest.raises(ValueError, match="overlaps"):
        validate_roster(overlapping, excluded_filenames=("excluded.png",))


def test_plans_are_fixed_balanced_and_source_clustered() -> None:
    fit_names = tuple(f"fit_{index:06d}.png" for index in range(FIT_SOURCE_COUNT))
    calibration_names = tuple(f"cal_{index:06d}.png" for index in range(CAL_SOURCE_COUNT))
    confirmation_names = tuple(f"confirm_{index:06d}.png" for index in range(CONFIRM_SOURCE_COUNT))
    fit_rows = fit_plan(fit_names)
    assert len(fit_rows) == FIT_UPDATES * BATCH_SIZE
    assert sum(row.state_family == "solver_replay" for row in fit_rows) == len(fit_rows) // 2
    source_counts = {name: 0 for name in fit_names}
    for row in fit_rows:
        source_counts[row.source_filename] += 1
    assert max(source_counts.values()) - min(source_counts.values()) <= 1
    for step in range(FIT_UPDATES):
        assert len({row.source_filename for row in fit_rows[step * 4 : step * 4 + 4]}) == 4
    assert plan_digest(fit_rows) == plan_digest(fit_plan(fit_names))
    assert PLAN_SEED == 20261002

    for phase, names in (
        ("calibration", calibration_names),
        ("confirmation", confirmation_names),
    ):
        rows = evaluation_plan(names, phase=phase)
        assert len(rows) == len(names) * 2
        for source_index, filename in enumerate(names):
            source_rows = rows[source_index * 2 : source_index * 2 + 2]
            assert [row.source_filename for row in source_rows] == [filename, filename]
            assert [row.state_family for row in source_rows] == [
                "solver_replay",
                "procedural",
            ]


def test_unsigned_template_is_fail_closed() -> None:
    with pytest.raises(PermissionError, match="no-runner scaffold"):
        require_signed_execution(
            {
                "schema": "aiijc-basincycle-selector-aligned-v2-unsigned-template",
                "status": "unsigned-design-data-blocked",
                "authorization": {},
            }
        )


def test_checked_in_config_has_no_execution_or_roster_authority() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["status"] == "unsigned-design-data-blocked"
    assert config["implementation"]["runner"] is None
    assert config["implementation"]["execution_binding"] is None
    assert config["implementation"]["sha256_values"] is None
    assert config["source_protocol"]["actual_filenames"] is None
    assert config["source_protocol"]["actual_digests"] is None
    assert not any(config["authorization"].values())
    with pytest.raises(PermissionError, match="no-runner scaffold"):
        require_signed_execution(config)
    with pytest.raises(PermissionError, match="future separately audited runner"):
        require_signed_execution(
            {
                "schema": "aiijc-basincycle-selector-aligned-v2-signed-execution",
                "status": "signed",
                "authorization": {"review_completed": True},
            }
        )
