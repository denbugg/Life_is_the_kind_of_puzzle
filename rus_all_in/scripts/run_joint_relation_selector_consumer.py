#!/usr/bin/env python3
"""Freeze and score the conservative joint-to-six-arm DEV consumer.

The runner has two irreversible stages.  ``freeze`` reads only two already
target-free, hash-frozen sibling archives (joint verifier evidence and the
existing relation-selector six-arm roster) and freezes one whole-arm choice.
``score`` verifies that freeze before reconstructing any exact reference.

The checked-in config is deliberately unsigned and blocked.  This file does
not authorize opening DEV, local, terminal, or competition-test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from aiijc_puzzle.joint_relation_selector_consumer import (
    FIXED_HEAD_FRACTION,
    SELECTION_RULE,
    FrozenSixArmRoster,
    JointRelationEvidence,
    reject_target_bearing_array_names,
    select_fixed_head_dominant_arm,
)
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import (
    ExactSyntheticReference,
    make_exact_synthetic_case,
    names_digest,
)
from aiijc_puzzle.taska_relation_truth_selector import FEATURE_NAMES
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES, strict_layout
from aiijc_puzzle.tile_position_distance import evaluate_tile_position_distance

try:
    from scripts import run_fullres_boundary_denoiser as boundary
    from scripts import run_joint_reciprocal_tri_emitter_real as joint
except ModuleNotFoundError:
    import run_fullres_boundary_denoiser as boundary
    import run_joint_reciprocal_tri_emitter_real as joint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/joint_relation_selector_consumer_unsigned_template_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/joint-relation-selector-consumer/dev32-development-v1"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "configs/validation-manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"

CONFIG_SCHEMA = "aiijc-joint-relation-selector-consumer-protocol-v1"
SIGNED_STATUS = "signed-fixed-protocol"
BLOCKED_STATUS = "unsigned-template-blocked-awaiting-target-free-sibling-freezes"
RELATION_ROSTER_SCHEMA = "aiijc-taska-relation-selector-target-free-roster-v1"
RELATION_ROSTER_FREEZE_SCHEMA = (
    "aiijc-taska-relation-selector-target-free-roster-freeze-v1"
)
OUTPUT_METADATA_SCHEMA = "aiijc-joint-relation-selector-target-free-selection-v1"
OUTPUT_FREEZE_SCHEMA = "aiijc-joint-relation-selector-pre-score-freeze-v1"
SCORE_SCHEMA = "aiijc-joint-relation-selector-score-v1"

OUTPUT_ARCHIVE = Path("frozen-target-free-selection.npz")
OUTPUT_METADATA = Path("frozen-target-free-selection.json")
OUTPUT_FREEZE = Path("pre-score-freeze.json")
OUTPUT_SCORE = Path("score.json")


@dataclass(frozen=True)
class VerifiedBundle:
    archive: Path
    metadata: Path
    freeze: Path
    rows: tuple[dict[str, Any], ...]
    archive_sha256: str
    metadata_sha256: str


T = TypeVar("T")
R = TypeVar("R")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "score"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    return parser.parse_args(argv)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["case_id"]),
        str(row["source_filename"]),
        int(row["draw_index"]),
        str(row["dirty_sha256"]),
    )


def rule_commitment_sha256(config: Mapping[str, Any]) -> str:
    """Hash only fields that must be fixed before future archive creation."""

    payload = {
        "schema": config.get("schema"),
        "joint_protocol_sha256": config.get("joint_protocol_sha256"),
        "source_protocol": config.get("source_protocol"),
        "selection": config.get("selection"),
        "evaluation_gate": config.get("evaluation_gate"),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_exact_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("joint relation consumer protocol schema changed")
    selection = config.get("selection", {})
    expected = {
        "grid_size": 24,
        "tile_count": 576,
        "relation_count": 1104,
        "fixed_head_fraction": FIXED_HEAD_FRACTION,
        "fixed_head_count_per_axis": math.ceil(FIXED_HEAD_FRACTION * 576),
        "selection_rule": SELECTION_RULE,
        "layout_synthesis": False,
        "allow_only_existing_whole_arm": True,
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise RuntimeError(f"fixed joint relation selection changed: {key}")
    if tuple(selection.get("arm_names", ())) != tuple(FUSION_ARM_NAMES):
        raise RuntimeError("fixed six-arm roster changed")
    gate = config.get("evaluation_gate", {})
    expected_gate = {
        "mean_satisfied_pairs_delta_minimum": 0.0,
        "mean_exact_tiles_delta_minimum": 0.0,
        "mean_absolute_manhattan_delta_maximum": 0.0,
        "mean_radius2_recall_delta_minimum": 0.0,
        "require_at_least_one_changed_case": True,
        "require_at_least_one_strict_aggregate_improvement": True,
    }
    if gate != expected_gate:
        raise RuntimeError("bounded Pareto evaluation gate changed")
    source = config.get("source_protocol", {})
    names = source.get("dev_filenames")
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise RuntimeError("DEV source roster must be an explicit unique list")
    if names_digest(names) != source.get("dev_digest"):
        raise RuntimeError("DEV source roster digest changed")
    if int(source.get("dev_draw_index", -1)) != 0:
        raise RuntimeError("DEV draw index changed")
    if int(source.get("dev_case_seed", -1)) != 20260908:
        raise RuntimeError("DEV case seed changed")
    if config.get("rule_commitment_sha256") != rule_commitment_sha256(config):
        raise RuntimeError("pre-archive rule commitment digest changed")


def _verified_record(config: Mapping[str, Any], name: str) -> Path:
    artifact = config.get("frozen_inputs", {}).get(name)
    if not isinstance(artifact, Mapping):
        raise RuntimeError(f"frozen input is missing: {name}")
    path = _project_path(str(artifact.get("path", "")))
    expected = artifact.get("sha256")
    if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
        raise RuntimeError(f"frozen input changed or is pending: {name}")
    return path


def _load_signed_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    _require_exact_contract(config)
    if config.get("status") == BLOCKED_STATUS:
        raise RuntimeError("joint relation consumer template is intentionally blocked")
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    digest = sha256_file(resolved)
    if (
        config.get("status") != SIGNED_STATUS
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != digest
    ):
        raise RuntimeError("joint relation consumer protocol is not signed/fixed")
    for name in config.get("frozen_inputs", {}):
        _verified_record(config, name)
    joint_config_path = _verified_record(config, "joint_protocol")
    if sha256_file(joint_config_path) != config.get("joint_protocol_sha256"):
        raise RuntimeError("joint protocol digest differs from fixed consumer contract")
    joint_config = json.loads(joint_config_path.read_text(encoding="utf-8"))
    source = config["source_protocol"]
    joint_source = joint_config.get("source_protocol", {})
    for key in ("dev_filenames", "dev_digest", "dev_draw_index", "dev_case_seed"):
        if source[key] != joint_source.get(key):
            raise RuntimeError(f"consumer source protocol differs from joint protocol: {key}")
    return config, digest


def _verify_nested_artifact(
    freeze: Mapping[str, Any], name: str, path: Path, expected_sha256: str
) -> None:
    artifact = freeze.get("artifacts", {}).get(name)
    if not isinstance(artifact, Mapping) or artifact.get("sha256") != expected_sha256:
        raise RuntimeError(f"pre-score freeze does not bind {name}")
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"target-free {name} changed after freeze")


def verify_joint_bundle(config: Mapping[str, Any]) -> VerifiedBundle:
    archive = _verified_record(config, "joint_archive")
    metadata = _verified_record(config, "joint_metadata")
    freeze_path = _verified_record(config, "joint_pre_score_freeze")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "aiijc-joint-reciprocal-pre-score-freeze-v1":
        raise RuntimeError("joint pre-score freeze schema changed")
    if freeze.get("created_before_exact_reference_scoring") is not True:
        raise RuntimeError("joint evidence was not frozen before reference scoring")
    if freeze.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("joint pre-score freeze contains labels")
    config_sha = config["joint_protocol_sha256"]
    if freeze.get("config_sha256") != config_sha:
        raise RuntimeError("joint freeze belongs to another protocol")
    _verify_nested_artifact(freeze, "archive", archive, sha256_file(archive))
    _verify_nested_artifact(freeze, "metadata", metadata, sha256_file(metadata))
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("schema") != "aiijc-joint-reciprocal-target-free-dev-v1":
        raise RuntimeError("joint target-free metadata schema changed")
    if payload.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("joint target-free metadata contains labels")
    if payload.get("contains_clean_dirty_or_output_pixels") is not False:
        raise RuntimeError("joint target-free metadata unexpectedly contains pixels")
    if payload.get("candidate_identities_immutable") is not True:
        raise RuntimeError("joint candidate identities are not immutable")
    if payload.get("config_sha256") != config_sha:
        raise RuntimeError("joint metadata belongs to another protocol")
    rows = tuple(payload.get("rows", ()))
    expected_names = tuple(config["source_protocol"]["dev_filenames"])
    if tuple(row.get("source_filename") for row in rows) != expected_names:
        raise RuntimeError("joint bundle differs from fixed DEV roster/order")
    return VerifiedBundle(
        archive=archive,
        metadata=metadata,
        freeze=freeze_path,
        rows=rows,
        archive_sha256=sha256_file(archive),
        metadata_sha256=sha256_file(metadata),
    )


def verify_relation_roster_bundle(config: Mapping[str, Any]) -> VerifiedBundle:
    archive = _verified_record(config, "relation_roster_archive")
    metadata = _verified_record(config, "relation_roster_metadata")
    freeze_path = _verified_record(config, "relation_roster_pre_score_freeze")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != RELATION_ROSTER_FREEZE_SCHEMA:
        raise RuntimeError("relation-roster pre-score freeze schema changed")
    if freeze.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("relation roster was not frozen before reference reconstruction")
    if freeze.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("relation-roster pre-score freeze contains labels")
    _verify_nested_artifact(freeze, "archive", archive, sha256_file(archive))
    _verify_nested_artifact(freeze, "metadata", metadata, sha256_file(metadata))
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("schema") != RELATION_ROSTER_SCHEMA:
        raise RuntimeError("relation-roster target-free metadata schema changed")
    if payload.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("relation-roster metadata contains labels")
    if payload.get("contains_pixels") is not False:
        raise RuntimeError("relation-roster metadata unexpectedly contains pixels")
    if payload.get("all_layouts_strict_original_upright_tile_permutations") is not True:
        raise RuntimeError("relation roster lost strict upright permutation contract")
    if tuple(payload.get("arm_names", ())) != tuple(FUSION_ARM_NAMES):
        raise RuntimeError("relation-roster arm names changed")
    if tuple(payload.get("relation_feature_names", ())) != tuple(FEATURE_NAMES):
        raise RuntimeError("relation-roster feature contract changed")
    rows = tuple(payload.get("rows", ()))
    expected_names = tuple(config["source_protocol"]["dev_filenames"])
    if tuple(row.get("source_filename") for row in rows) != expected_names:
        raise RuntimeError("relation roster differs from fixed DEV roster/order")
    return VerifiedBundle(
        archive=archive,
        metadata=metadata,
        freeze=freeze_path,
        rows=rows,
        archive_sha256=sha256_file(archive),
        metadata_sha256=sha256_file(metadata),
    )


def _case_arrays(archive: Mapping[str, Any], prefix: str) -> dict[str, np.ndarray]:
    marker = f"{prefix}__"
    return {
        key[len(marker) :]: np.asarray(archive[key])
        for key in archive
        if key.startswith(marker)
    }


def freeze_target_free_selection(
    output_dir: Path,
    config: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    """Consume two verified target-free siblings and freeze whole-arm choices."""

    joint_bundle = verify_joint_bundle(config)
    relation_bundle = verify_relation_roster_bundle(config)
    if len(joint_bundle.rows) != len(relation_bundle.rows):
        raise RuntimeError("joint and relation target-free case counts differ")
    if any(
        _identity(left) != _identity(right)
        for left, right in zip(joint_bundle.rows, relation_bundle.rows, strict=True)
    ):
        raise RuntimeError("joint and relation tile-bag identities differ")

    target = output_dir.resolve()
    target.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    with (
        np.load(joint_bundle.archive, allow_pickle=False) as joint_archive,
        np.load(relation_bundle.archive, allow_pickle=False) as relation_archive,
    ):
        reject_target_bearing_array_names(tuple(joint_archive.files))
        reject_target_bearing_array_names(tuple(relation_archive.files))
        for index, (joint_row, relation_row) in enumerate(
            zip(joint_bundle.rows, relation_bundle.rows, strict=True)
        ):
            prefix = f"case_{index:04d}"
            joint_values = _case_arrays(joint_archive, str(joint_row["prefix"]))
            relation_values = _case_arrays(
                relation_archive, str(relation_row["prefix"])
            )
            evidence = JointRelationEvidence.from_case_arrays(joint_values)
            if evidence.union_identity_digest != joint_row.get("union_identity_digest"):
                raise RuntimeError("joint metadata and archive identity digests differ")
            roster_row = {**relation_row, "arm_names": list(FUSION_ARM_NAMES)}
            roster = FrozenSixArmRoster.from_case_arrays(relation_values, roster_row)
            selection = select_fixed_head_dominant_arm(evidence, roster)
            incumbent = strict_layout(roster.layouts[selection.incumbent_index])
            candidate = strict_layout(selection.layout)
            arrays.update(
                {
                    f"{prefix}__incumbent_layout": incumbent.astype(np.int32),
                    f"{prefix}__candidate_layout": candidate.astype(np.int32),
                    f"{prefix}__arm_head_hits": selection.arm_evidence.head_hits,
                    f"{prefix}__arm_head_confidence_sums": (
                        selection.arm_evidence.head_confidence_sums
                    ),
                    f"{prefix}__arm_head_logit_sums": (
                        selection.arm_evidence.head_logit_sums
                    ),
                    f"{prefix}__arm_union_mapped_counts": (
                        selection.arm_evidence.mapped_union_counts
                    ),
                    f"{prefix}__arm_union_mapped_logit_sums": (
                        selection.arm_evidence.mapped_logit_sums
                    ),
                    f"{prefix}__selected_arm_index": np.asarray(
                        selection.selected_index, dtype=np.int32
                    ),
                    f"{prefix}__incumbent_arm_index": np.asarray(
                        selection.incumbent_index, dtype=np.int32
                    ),
                }
            )
            rows.append(
                {
                    "prefix": prefix,
                    "case_id": joint_row["case_id"],
                    "source_filename": joint_row["source_filename"],
                    "draw_index": int(joint_row["draw_index"]),
                    "dirty_sha256": joint_row["dirty_sha256"],
                    "union_identity_digest": evidence.union_identity_digest,
                    "incumbent_arm": selection.incumbent_arm,
                    "selected_arm": selection.selected_arm,
                    "changed": selection.changed,
                }
            )
    archive_path = target / OUTPUT_ARCHIVE
    metadata_path = target / OUTPUT_METADATA
    freeze_path = target / OUTPUT_FREEZE
    _write_npz(archive_path, arrays)
    _write_json(
        metadata_path,
        {
            "schema": OUTPUT_METADATA_SCHEMA,
            "config_sha256": config_sha256,
            "contains_exact_references_or_labels": False,
            "contains_pixels": False,
            "layout_only_original_upright_tile_identities": True,
            "selection_rule": SELECTION_RULE,
            "arm_names": list(FUSION_ARM_NAMES),
            "changed_case_count": sum(bool(row["changed"]) for row in rows),
            "rows": rows,
        },
    )
    _write_json(
        freeze_path,
        {
            "schema": OUTPUT_FREEZE_SCHEMA,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "config_sha256": config_sha256,
            "input_artifacts": {
                "joint_archive": _record(joint_bundle.archive),
                "joint_metadata": _record(joint_bundle.metadata),
                "joint_pre_score_freeze": _record(joint_bundle.freeze),
                "relation_roster_archive": _record(relation_bundle.archive),
                "relation_roster_metadata": _record(relation_bundle.metadata),
                "relation_roster_pre_score_freeze": _record(relation_bundle.freeze),
            },
            "artifacts": {
                "archive": _record(archive_path),
                "metadata": _record(metadata_path),
                "module": _record(
                    PROJECT_ROOT
                    / "src/aiijc_puzzle/joint_relation_selector_consumer.py"
                ),
                "runner": _record(Path(__file__)),
            },
        },
    )
    return {
        "schema": "aiijc-joint-relation-selector-freeze-result-v1",
        "status": "target-free-selection-frozen-label-scoring-not-run",
        "case_count": len(rows),
        "changed_case_count": sum(bool(row["changed"]) for row in rows),
        "archive": _record(archive_path),
        "metadata": _record(metadata_path),
        "pre_score_freeze": _record(freeze_path),
        "dev_labels_scored": False,
        "terminal_or_competition_test_accessed": False,
    }


def verify_selection_freeze(
    output_dir: Path, config_sha256: str
) -> VerifiedBundle:
    target = output_dir.resolve()
    archive = target / OUTPUT_ARCHIVE
    metadata = target / OUTPUT_METADATA
    freeze_path = target / OUTPUT_FREEZE
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != OUTPUT_FREEZE_SCHEMA:
        raise RuntimeError("selection pre-score freeze schema changed")
    if freeze.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("selection was not frozen before reference reconstruction")
    if freeze.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("selection pre-score freeze contains labels")
    if freeze.get("config_sha256") != config_sha256:
        raise RuntimeError("selection freeze belongs to another protocol")
    for name, path in (("archive", archive), ("metadata", metadata)):
        expected = freeze.get("artifacts", {}).get(name, {}).get("sha256")
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"target-free selection {name} changed after freeze")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("schema") != OUTPUT_METADATA_SCHEMA:
        raise RuntimeError("selection target-free metadata schema changed")
    if payload.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("selection target-free metadata contains labels")
    if payload.get("contains_pixels") is not False:
        raise RuntimeError("selection target-free metadata contains pixels")
    if payload.get("layout_only_original_upright_tile_identities") is not True:
        raise RuntimeError("selection output lost upright original-tile contract")
    if payload.get("config_sha256") != config_sha256:
        raise RuntimeError("selection metadata belongs to another protocol")
    return VerifiedBundle(
        archive=archive,
        metadata=metadata,
        freeze=freeze_path,
        rows=tuple(payload.get("rows", ())),
        archive_sha256=sha256_file(archive),
        metadata_sha256=sha256_file(metadata),
    )


def score_after_verified_freeze(
    output_dir: Path,
    config_sha256: str,
    *,
    reference_loader: Callable[[VerifiedBundle], T],
    scorer: Callable[[VerifiedBundle, T], R],
) -> R:
    """Dependency-injected proof that hashes are checked before any labels."""

    verified = verify_selection_freeze(output_dir, config_sha256)
    references = reference_loader(verified)
    return scorer(verified, references)


def _load_exact_dev_references(
    verified: VerifiedBundle,
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> dict[str, ExactSyntheticReference]:
    names = tuple(config["source_protocol"]["dev_filenames"])
    records = joint._manifest_records(args.manifest, names)
    boards = boundary._prepare_boards(records, args.targets)
    if tuple(row["source_filename"] for row in verified.rows) != names:
        raise RuntimeError("selection freeze differs from fixed DEV roster")
    result: dict[str, ExactSyntheticReference] = {}
    draw_index = int(config["source_protocol"]["dev_draw_index"])
    seed = int(config["source_protocol"]["dev_case_seed"])
    for board, row in zip(boards, verified.rows, strict=True):
        item, reference = make_exact_synthetic_case(
            board.tiles,
            source_filename=board.filename,
            draw_index=draw_index,
            seed=seed,
        )
        observed = (
            item.case_id,
            item.source_filename,
            item.draw_index,
            hashlib.sha256(item.tiles.tobytes()).hexdigest(),
        )
        if observed != _identity(row):
            raise RuntimeError("exact reference reconstruction differs from target-free freeze")
        result[item.case_id] = reference
    return result


def score_frozen_selection(
    verified: VerifiedBundle,
    references: Mapping[str, ExactSyntheticReference],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Score pairs, exact, absolute Manhattan and radius-2 after the freeze."""

    cases: list[dict[str, Any]] = []
    with np.load(verified.archive, allow_pickle=False) as archive:
        reject_target_bearing_array_names(tuple(archive.files))
        for row in verified.rows:
            prefix = str(row["prefix"])
            reference = references[str(row["case_id"])].tile_at_position
            control = strict_layout(archive[f"{prefix}__incumbent_layout"])
            candidate = strict_layout(archive[f"{prefix}__candidate_layout"])
            control_layout = evaluate_layout(control, reference, reference_is_exact=True)
            candidate_layout = evaluate_layout(
                candidate, reference, reference_is_exact=True
            )
            control_distance = evaluate_tile_position_distance(control, reference)
            candidate_distance = evaluate_tile_position_distance(candidate, reference)
            cases.append(
                {
                    "case_id": row["case_id"],
                    "source_filename": row["source_filename"],
                    "draw_index": int(row["draw_index"]),
                    "changed": bool(row["changed"]),
                    "incumbent_arm": row["incumbent_arm"],
                    "selected_arm": row["selected_arm"],
                    "control": {
                        "satisfied_pairs": control_layout.adjacency_correct,
                        "exact_tiles": control_layout.correct_tile_count,
                        "absolute_mean_manhattan": (
                            control_distance.mean_manhattan_cells
                        ),
                        "radius2_recall": control_distance.within_radius_2_recall,
                    },
                    "candidate": {
                        "satisfied_pairs": candidate_layout.adjacency_correct,
                        "exact_tiles": candidate_layout.correct_tile_count,
                        "absolute_mean_manhattan": (
                            candidate_distance.mean_manhattan_cells
                        ),
                        "radius2_recall": candidate_distance.within_radius_2_recall,
                    },
                }
            )
    metric_names = (
        "satisfied_pairs",
        "exact_tiles",
        "absolute_mean_manhattan",
        "radius2_recall",
    )
    aggregate: dict[str, Any] = {}
    for name in metric_names:
        control_values = np.asarray([case["control"][name] for case in cases], dtype=np.float64)
        candidate_values = np.asarray(
            [case["candidate"][name] for case in cases], dtype=np.float64
        )
        aggregate[name] = {
            "control_mean": float(control_values.mean()),
            "candidate_mean": float(candidate_values.mean()),
            "delta": float((candidate_values - control_values).mean()),
        }
    changed_count = sum(bool(case["changed"]) for case in cases)
    deltas = {name: aggregate[name]["delta"] for name in metric_names}
    checks = {
        "pairs_nonnegative": deltas["satisfied_pairs"]
        >= float(gate["mean_satisfied_pairs_delta_minimum"]),
        "exact_nonnegative": deltas["exact_tiles"]
        >= float(gate["mean_exact_tiles_delta_minimum"]),
        "manhattan_nonincreasing": deltas["absolute_mean_manhattan"]
        <= float(gate["mean_absolute_manhattan_delta_maximum"]),
        "radius2_nonnegative": deltas["radius2_recall"]
        >= float(gate["mean_radius2_recall_delta_minimum"]),
        "changed_case": changed_count > 0,
        "strict_aggregate_improvement": (
            deltas["satisfied_pairs"] > 0
            or deltas["exact_tiles"] > 0
            or deltas["absolute_mean_manhattan"] < 0
            or deltas["radius2_recall"] > 0
        ),
    }
    return {
        "case_count": len(cases),
        "changed_case_count": changed_count,
        "aggregate": aggregate,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "cases": cases,
        "freeze": {
            "archive_sha256": verified.archive_sha256,
            "metadata_sha256": verified.metadata_sha256,
            "verified_before_reference_loading": True,
        },
        "all_outputs_strict_576_original_upright_permutations": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_sha = _load_signed_config(args.config)
    if args.mode == "freeze":
        result = freeze_target_free_selection(args.output_dir, config, config_sha)
    else:
        metrics = score_after_verified_freeze(
            args.output_dir,
            config_sha,
            reference_loader=lambda verified: _load_exact_dev_references(
                verified, args=args, config=config
            ),
            scorer=lambda verified, references: score_frozen_selection(
                verified, references, config["evaluation_gate"]
            ),
        )
        result = {
            "schema": SCORE_SCHEMA,
            "status": "complete-bounded-one-shot-no-tuning",
            "config_sha256": config_sha,
            **metrics,
            "terminal_or_competition_test_accessed": False,
        }
        _write_json(args.output_dir.resolve() / OUTPUT_SCORE, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
