from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aiijc_puzzle.structured_decoder_fit_oracle import (
    DirectedEdge,
    evaluate_pair_safe_oracle,
    layout_metrics,
    strict_layout,
    validate_fixed_reciprocal_head,
)
from scripts import audit_structured_decoder_fit_oracle as runner


def _head() -> tuple[DirectedEdge, ...]:
    return (
        DirectedEdge(0, 0, 1, 5.0),
        DirectedEdge(0, 3, 4, 4.0),
        DirectedEdge(1, 0, 3, 3.0),
        DirectedEdge(1, 1, 4, 2.0),
    )


def test_strict_layout_and_absolute_metrics() -> None:
    reference = np.arange(9, dtype=np.int32)
    candidate = reference.copy()
    candidate[[1, 2]] = candidate[[2, 1]]
    metrics = layout_metrics(candidate, reference, grid=3)
    assert metrics.exact_tiles == 7
    assert metrics.satisfied_pairs < 12
    assert metrics.mean_absolute_manhattan == pytest.approx(2 / 9)
    assert metrics.radius2_recall == 1.0
    with pytest.raises(ValueError):
        strict_layout(np.zeros(9), grid=3)


def test_fixed_head_rejects_nonreciprocal_identity_collisions() -> None:
    assert len(
        validate_fixed_reciprocal_head(_head(), grid=3, requested_per_axis=2)
    ) == 4
    duplicate_target = (
        DirectedEdge(0, 0, 1, 5.0),
        DirectedEdge(0, 3, 1, 4.0),
        DirectedEdge(1, 0, 3, 3.0),
        DirectedEdge(1, 1, 4, 2.0),
    )
    with pytest.raises(ValueError, match="uniqueness"):
        validate_fixed_reciprocal_head(
            duplicate_target, grid=3, requested_per_axis=2
        )


def test_pair_safe_oracle_recovers_missing_true_head_without_pair_loss() -> None:
    reference = np.arange(9, dtype=np.int32)
    control = reference.copy()
    control[[1, 2]] = control[[2, 1]]
    result = evaluate_pair_safe_oracle(
        control,
        reference,
        _head(),
        grid=3,
        requested_per_axis=2,
    )
    assert result.selected_true_edge_count == 4
    assert result.compatible_missing_true_edge_headroom == 2
    assert result.initial_pair_safe_action_count >= 1
    assert result.accepted_action_count == 1
    assert result.realised_supplied_true_edge_gain == 2
    assert result.pair_delta > 0
    assert result.exact_delta == 2
    assert result.manhattan_delta < 0
    assert np.array_equal(result.ceiling_layout, reference)
    assert np.array_equal(np.sort(result.ceiling_layout), reference)


def test_stop_fallback_is_exact_control_and_never_pair_negative() -> None:
    reference = np.arange(9, dtype=np.int32)
    result = evaluate_pair_safe_oracle(
        reference,
        reference,
        _head(),
        grid=3,
        requested_per_axis=2,
    )
    assert result.compatible_missing_true_edge_headroom == 0
    assert result.accepted_action_count == 0
    assert result.pair_delta == 0
    assert result.exact_delta == 0
    assert result.manhattan_delta == 0
    assert result.radius2_delta == 0
    assert np.array_equal(result.ceiling_layout, reference)


def test_oracle_is_equivariant_to_tile_identity_relabelling() -> None:
    reference = np.arange(9, dtype=np.int32)
    control = reference.copy()
    control[[1, 2]] = control[[2, 1]]
    initial = evaluate_pair_safe_oracle(
        control, reference, _head(), grid=3, requested_per_axis=2
    )
    relabel = np.asarray([8, 2, 6, 0, 5, 1, 7, 3, 4], dtype=np.int32)
    relabelled_head = tuple(
        DirectedEdge(
            edge.axis,
            int(relabel[edge.source]),
            int(relabel[edge.target]),
            edge.confidence,
        )
        for edge in _head()
    )
    observed = evaluate_pair_safe_oracle(
        relabel[control],
        relabel[reference],
        relabelled_head,
        grid=3,
        requested_per_axis=2,
    )
    assert observed.pair_delta == initial.pair_delta
    assert observed.exact_delta == initial.exact_delta
    assert observed.realised_supplied_true_edge_gain == (
        initial.realised_supplied_true_edge_gain
    )
    assert np.array_equal(observed.ceiling_layout, relabel[initial.ceiling_layout])


def test_fixed_six_by_six_capacity_contract_rejects_strong_distractor() -> None:
    reference = np.arange(36, dtype=np.int32)
    control = reference.copy()
    control[[1, 2]] = control[[2, 1]]
    head = (
        DirectedEdge(0, 0, 1, 5.0),
        DirectedEdge(0, 6, 7, 4.0),
        # This incompatible false edge is locally strongest but never enters the
        # target-assisted true-action roster.
        DirectedEdge(1, 1, 8, 100.0),
        DirectedEdge(1, 0, 6, 3.0),
    )
    recovered = evaluate_pair_safe_oracle(
        control, reference, head, grid=6, requested_per_axis=2
    )
    assert recovered.selected_true_edge_count == 3
    assert all((action.source, action.target) != (1, 8) for action in recovered.actions)
    assert recovered.pair_delta >= 0
    assert np.array_equal(recovered.ceiling_layout, reference)

    relabel = np.random.default_rng(20260831).permutation(36).astype(np.int32)
    relabelled = evaluate_pair_safe_oracle(
        relabel[control],
        relabel[reference],
        tuple(
            DirectedEdge(
                edge.axis,
                int(relabel[edge.source]),
                int(relabel[edge.target]),
                edge.confidence,
            )
            for edge in head
        ),
        grid=6,
        requested_per_axis=2,
    )
    assert np.array_equal(relabelled.ceiling_layout, relabel[reference])
    assert relabelled.pair_delta == recovered.pair_delta

    stopped = evaluate_pair_safe_oracle(
        reference, reference, head, grid=6, requested_per_axis=2
    )
    assert stopped.accepted_action_count == 0
    assert stopped.pair_delta == 0
    assert np.array_equal(stopped.ceiling_layout, reference)


def _write_fake_head(root: Path, *, forbidden_label: bool = False) -> None:
    fit = root / "fit"
    fit.mkdir(parents=True)
    archive = fit / "frozen-target-free-reciprocal-heads.npz"
    metadata = fit / "frozen-target-free-reciprocal-heads.json"
    freeze = fit / "reciprocal-heads-pre-score-freeze.json"
    arrays = {}
    rows = []
    config_sha = "a" * 64
    for index in range(64):
        prefix = f"case_{index:04d}"
        union_digest = f"{index + 2:064x}"
        arrays[f"{prefix}__union_identity_digest_ascii"] = np.frombuffer(
            union_digest.encode("ascii"), dtype=np.uint8
        )
        for axis, name in enumerate(("right", "down")):
            sources = np.arange(29, dtype=np.int32)
            targets = np.arange(29, dtype=np.int32) + 40 + 100 * axis
            arrays[f"{prefix}__selected_sources__{name}"] = sources
            arrays[f"{prefix}__selected_targets__{name}"] = targets
            arrays[f"{prefix}__selected_joint_confidences__{name}"] = np.linspace(
                2.0, 1.0, 29, dtype=np.float32
            )
            arrays[f"{prefix}__requested_count__{name}"] = np.asarray(
                29, dtype=np.int32
            )
            arrays[f"{prefix}__reciprocal_count__{name}"] = np.asarray(
                40, dtype=np.int32
            )
        rows.append(
            {
                "prefix": prefix,
                "case_id": f"synthetic-{index:016x}",
                "source_filename": f"img_{index:06d}.png",
                "draw_index": index % 2,
                "dirty_sha256": f"{index:064x}",
                "fit_cache": {
                    "path": f"fit-cache/case_{index:04d}.npz",
                    "sha256": f"{index + 1:064x}",
                },
                "union_identity_digest": union_digest,
            }
        )
    if forbidden_label:
        arrays["case_0000__target_slots"] = np.zeros((2, 576), dtype=np.int16)
    np.savez_compressed(archive, **arrays)
    metadata.write_text(
        json.dumps(
            {
                "schema": runner.HEAD_SCHEMA,
                "config_sha256": config_sha,
                "contains_target_slots_truth_or_reference_labels": False,
                "contains_pixels": False,
                "tile_id_space": "immutable-shuffled-tile-bag-identity",
                "candidate_identities_immutable": True,
                "fixed_fraction_per_axis_per_board": 0.05,
                "expected_requested_count_for_576_tiles": 29,
                "strict_target_free_loader_schema": runner.STRICT_HEAD_LOADER_SCHEMA,
                "npz_members_materialised": runner.TARGET_FREE_HEAD_INPUT_KEYS,
                "label_members_materialised": [],
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    freeze.write_text(
        json.dumps(
            {
                "schema": runner.HEAD_FREEZE_SCHEMA,
                "created_before_fit_head_label_scoring": True,
                "contains_target_slots_truth_or_reference_labels": False,
                "strict_target_free_loader_schema": runner.STRICT_HEAD_LOADER_SCHEMA,
                "label_cache_members_materialised": False,
                "config_sha256": config_sha,
                "artifacts": {
                    "archive": {"sha256": runner.sha256_file(archive)},
                    "metadata": {"sha256": runner.sha256_file(metadata)},
                    "fit_endpoint": {
                        "path": str(archive),
                        "sha256": runner.sha256_file(archive),
                    },
                    "runner": {
                        "path": str(metadata),
                        "sha256": runner.sha256_file(metadata),
                    },
                    "module": {
                        "path": str(metadata),
                        "sha256": runner.sha256_file(metadata),
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_availability_and_head_schema_are_fail_closed(tmp_path: Path) -> None:
    args = runner.parse_args(
        ["--mode", "availability", "--joint-experiment-dir", str(tmp_path)]
    )
    assert runner.availability(args)["available"] is False
    _write_fake_head(tmp_path)
    assert runner.availability(args)["available"] is True
    archive, metadata, freeze, _payload, rows = runner._load_head(args)
    assert archive.is_file() and metadata.is_file() and freeze.is_file()
    assert len(rows) == 64


def test_head_archive_rejects_any_copied_fit_label(tmp_path: Path) -> None:
    _write_fake_head(tmp_path, forbidden_label=True)
    args = runner.parse_args(
        ["--mode", "availability", "--joint-experiment-dir", str(tmp_path)]
    )
    with pytest.raises(RuntimeError, match="forbidden label"):
        runner._load_head(args)
