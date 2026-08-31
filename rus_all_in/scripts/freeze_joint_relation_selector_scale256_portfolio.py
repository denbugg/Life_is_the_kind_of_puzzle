#!/usr/bin/env python3
"""Fail-closed target-free freeze of a fixed scale256 six-arm portfolio.

The checked-in config is an unsigned, non-executable template.  A later
reviewer may create a separate signed binding only after both target-free
DEV64 sibling bundles exist.  This runner has no scoring/reference mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aiijc_puzzle.joint_relation_selector_consumer import (
    FrozenSixArmRoster,
    JointRelationEvidence,
    reject_target_bearing_array_names,
)
from aiijc_puzzle.joint_relation_selector_portfolio import (
    MISSING_EDGE_OFFSET,
    NORMALIZATION_EPSILON,
    NORMALIZATION_TANH_SCALE,
    PORTFOLIO_MEMBER_NAMES,
    SOURCE_NORMALIZED_RULE,
    UNION_DENSE_RULE,
    build_frozen_selector_portfolio,
)
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import names_digest
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES

try:
    from scripts import run_joint_relation_selector_consumer as legacy
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_joint_relation_selector_consumer as legacy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/joint_relation_selector_scale256_portfolio_unsigned_template_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/joint-relation-selector-portfolio/scale256-dev64-v1"
)

CONFIG_SCHEMA = "aiijc-joint-relation-selector-scale256-portfolio-protocol-v1"
SIGNED_STATUS = "signed-fixed-protocol"
BLOCKED_STATUS = "unsigned-template-blocked-awaiting-target-free-dev64-bundles"
OUTPUT_SCHEMA = "aiijc-joint-relation-selector-scale256-portfolio-v1"
FREEZE_SCHEMA = "aiijc-joint-relation-selector-scale256-portfolio-freeze-v1"
OUTPUT_ARCHIVE = Path("frozen-target-free-portfolio.npz")
OUTPUT_METADATA = Path("frozen-target-free-portfolio.json")
OUTPUT_FREEZE = Path("pre-reference-freeze.json")
DEV_COUNT = 64
DEV_DIGEST = "5c6cb5b9b204a38c78e79936ff34235dae9896cfc13d6edaf12dfad635bcdb8e"
DEV_DRAW_INDEX = 0
DEV_CASE_SEED = 20260908


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _project_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": _project_label(resolved), "sha256": sha256_file(resolved)}


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
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


def _case_arrays(archive: Mapping[str, Any], prefix: str) -> dict[str, np.ndarray]:
    marker = f"{prefix}__"
    return {
        key[len(marker) :]: np.asarray(archive[key])
        for key in archive
        if key.startswith(marker)
    }


def build_case_freeze_payload(
    evidence: JointRelationEvidence,
    roster: FrozenSixArmRoster,
) -> tuple[dict[str, np.ndarray], tuple[int, ...], tuple[str, ...]]:
    """Materialise one transparent portfolio and its target-free diagnostics."""

    portfolio = build_frozen_selector_portfolio(evidence, roster)
    diagnostics = portfolio.union_dense.arm_evidence
    arrays = {
        "portfolio_layouts": np.stack(portfolio.layouts).astype(np.int32),
        "selected_arm_indices": np.asarray(
            portfolio.selected_indices, dtype=np.int32
        ),
        "arm_union_coverage_counts": diagnostics.union_coverage_counts,
        "arm_union_dense_confidence_sums": (
            diagnostics.union_dense_confidence_sums
        ),
        "arm_union_dense_confidence_means": (
            diagnostics.union_dense_confidence_means
        ),
        "arm_normalized_logit_sums": diagnostics.normalized_logit_sums,
        "arm_normalized_confidence_sums": (
            diagnostics.normalized_confidence_sums
        ),
        "arm_normalized_combined_sums": diagnostics.normalized_combined_sums,
        "arm_missing_edge_counts": diagnostics.missing_edge_counts,
        "arm_legacy_head_hits": diagnostics.legacy_head.head_hits,
    }
    selected_arms = tuple(
        FUSION_ARM_NAMES[value] for value in portfolio.selected_indices
    )
    return arrays, portfolio.selected_indices, selected_arms


def selection_contract() -> dict[str, Any]:
    """The complete fixed decision contract committed before DEV64 access."""

    return {
        "grid_size": 24,
        "tile_count": 576,
        "relation_count_per_layout": 1104,
        "arm_names": list(FUSION_ARM_NAMES),
        "portfolio_member_names": list(PORTFOLIO_MEMBER_NAMES),
        "members": {
            "incumbent_keep": "exact relation-selector incumbent layout",
            "fixed_head_comparator": (
                "unchanged axiswise fixed-5%-head dominance comparator"
            ),
            "union_dense_dominance": UNION_DENSE_RULE,
            "source_normalized_dominance": SOURCE_NORMALIZED_RULE,
        },
        "union_dense_eligibility": (
            "union coverage count and mean dense two-sided confidence each "
            "weakly nonregressing versus incumbent on right and down; at least "
            "one strict improvement"
        ),
        "union_dense_tie_break": [
            "total union-coverage delta",
            "minimum per-axis union-coverage delta",
            "total mean-confidence delta",
            "minimum per-axis mean-confidence delta",
            "frozen expected-correct score",
            "frozen arm order",
        ],
        "source_normalization": {
            "within_source_per_axis": True,
            "formula": "tanh(((value-mean)/population_std)/2)",
            "constant_row_value": 0.0,
            "epsilon": NORMALIZATION_EPSILON,
            "logit_weight": 0.5,
            "dense_two_sided_confidence_weight": 0.5,
            "missing_component_floor": "minimum valid normalized component - 1",
            "missing_edge_offset": MISSING_EDGE_OFFSET,
            "tanh_scale": NORMALIZATION_TANH_SCALE,
        },
        "source_normalized_eligibility": (
            "combined full-edge score weakly nonregressing versus incumbent on "
            "right and down separately; at least one strict improvement"
        ),
        "source_normalized_tie_break": [
            "total combined-score delta",
            "minimum per-axis combined-score delta",
            "fewest missing realised edges",
            "frozen expected-correct score",
            "frozen arm order",
        ],
        "layout_synthesis": False,
        "only_exact_existing_whole_arm_layouts": True,
        "tile_id_equivariant": True,
        "labels_or_references_used": False,
        "posthoc_member_selection": False,
    }


def rule_commitment_sha256(config: Mapping[str, Any]) -> str:
    payload = {
        "schema": config.get("schema"),
        "joint_protocol_sha256": config.get("joint_protocol_sha256"),
        "source_protocol": config.get("source_protocol"),
        "selection": config.get("selection"),
        "future_scoring_contract": config.get("future_scoring_contract"),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_exact_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("scale256 portfolio config schema changed")
    source = config.get("source_protocol", {})
    names = source.get("dev_filenames")
    if (
        not isinstance(names, list)
        or len(names) != DEV_COUNT
        or len(set(names)) != DEV_COUNT
        or names_digest(names) != DEV_DIGEST
        or source.get("dev_digest") != DEV_DIGEST
        or source.get("dev_draw_index") != DEV_DRAW_INDEX
        or source.get("dev_case_seed") != DEV_CASE_SEED
    ):
        raise RuntimeError("scale256 portfolio DEV64 roster contract changed")
    if config.get("selection") != selection_contract():
        raise RuntimeError("scale256 portfolio fixed selection contract changed")
    expected_scoring = {
        "status": "not-authorized-by-this-freezer",
        "report_every_member_transparently": True,
        "metrics": [
            "satisfied_pairs",
            "adjusted_pairs",
            "exact_tiles",
            "absolute_mean_manhattan",
            "radius2_recall",
        ],
        "selection_from_dev_labels": False,
    }
    if config.get("future_scoring_contract") != expected_scoring:
        raise RuntimeError("scale256 portfolio future scoring contract changed")
    if config.get("rule_commitment_sha256") != rule_commitment_sha256(config):
        raise RuntimeError("scale256 portfolio rule commitment changed")


def _verified_record(config: Mapping[str, Any], name: str) -> Path:
    record = config.get("frozen_inputs", {}).get(name)
    if not isinstance(record, Mapping):
        raise RuntimeError(f"portfolio frozen input missing: {name}")
    path = _project_path(str(record.get("path", "")))
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"portfolio frozen input hash is pending: {name}")
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(f"portfolio frozen input changed: {name}")
    return path


def _load_signed_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    _require_exact_contract(config)
    if config.get("status") == BLOCKED_STATUS:
        raise RuntimeError("scale256 portfolio template is intentionally blocked")
    digest = sha256_file(resolved)
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if (
        config.get("status") != SIGNED_STATUS
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != digest
    ):
        raise RuntimeError("scale256 portfolio config is not separately signed")
    required = {
        "joint_protocol",
        "joint_archive",
        "joint_metadata",
        "joint_pre_score_freeze",
        "relation_roster_archive",
        "relation_roster_metadata",
        "relation_roster_pre_score_freeze",
        "legacy_module",
        "legacy_runner",
        "portfolio_module",
        "portfolio_runner",
    }
    if set(config.get("frozen_inputs", {})) != required:
        raise RuntimeError("scale256 portfolio frozen-input inventory changed")
    for name in required:
        _verified_record(config, name)
    joint_protocol = json.loads(
        _verified_record(config, "joint_protocol").read_text(encoding="utf-8")
    )
    if sha256_file(_verified_record(config, "joint_protocol")) != config.get(
        "joint_protocol_sha256"
    ):
        raise RuntimeError("scale256 portfolio joint protocol digest changed")
    source = joint_protocol.get("source_contract", {})
    if (
        source.get("reserved_dev_source_count") != DEV_COUNT
        or source.get("reserved_dev_digest") != DEV_DIGEST
        or source.get("dev_draw_index") != DEV_DRAW_INDEX
        or source.get("dev_case_seed") != DEV_CASE_SEED
    ):
        raise RuntimeError("scale256 portfolio differs from joint DEV contract")
    return config, digest


def freeze_target_free_portfolio(
    output_dir: Path,
    config: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    """Freeze all fixed members after verifying both target-free siblings."""

    joint_bundle = legacy.verify_joint_bundle(config)
    relation_bundle = legacy.verify_relation_roster_bundle(config)
    if len(joint_bundle.rows) != DEV_COUNT or len(relation_bundle.rows) != DEV_COUNT:
        raise RuntimeError("scale256 portfolio does not contain exactly DEV64")
    if any(
        _identity(left) != _identity(right)
        for left, right in zip(joint_bundle.rows, relation_bundle.rows, strict=True)
    ):
        raise RuntimeError("joint and relation DEV64 tile-bag identities differ")

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
            evidence = JointRelationEvidence.from_case_arrays(
                _case_arrays(joint_archive, str(joint_row["prefix"]))
            )
            if evidence.union_identity_digest != joint_row.get(
                "union_identity_digest"
            ):
                raise RuntimeError("joint archive/metadata candidate digest mismatch")
            roster = FrozenSixArmRoster.from_case_arrays(
                _case_arrays(relation_archive, str(relation_row["prefix"])),
                {**relation_row, "arm_names": list(FUSION_ARM_NAMES)},
            )
            case_arrays, selected_indices, selected_arms = build_case_freeze_payload(
                evidence, roster
            )
            arrays.update(
                {f"{prefix}__{name}": value for name, value in case_arrays.items()}
            )
            rows.append(
                {
                    "prefix": prefix,
                    "case_id": joint_row["case_id"],
                    "source_filename": joint_row["source_filename"],
                    "draw_index": int(joint_row["draw_index"]),
                    "dirty_sha256": joint_row["dirty_sha256"],
                    "union_identity_digest": evidence.union_identity_digest,
                    "portfolio_member_names": list(PORTFOLIO_MEMBER_NAMES),
                    "selected_arms": list(selected_arms),
                    "changed_from_incumbent": [
                        value != roster.incumbent_index for value in selected_indices
                    ],
                }
            )

    archive_path = target / OUTPUT_ARCHIVE
    metadata_path = target / OUTPUT_METADATA
    freeze_path = target / OUTPUT_FREEZE
    _write_npz_exclusive(archive_path, arrays)
    _write_json_exclusive(
        metadata_path,
        {
            "schema": OUTPUT_SCHEMA,
            "status": "target-free-portfolio-frozen-no-reference-access",
            "config_sha256": config_sha256,
            "contains_exact_references_or_labels": False,
            "contains_pixels": False,
            "all_outputs_exact_existing_strict_upright_layouts": True,
            "arm_names": list(FUSION_ARM_NAMES),
            "portfolio_member_names": list(PORTFOLIO_MEMBER_NAMES),
            "selection_contract": selection_contract(),
            "rows": rows,
        },
    )
    _write_json_exclusive(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_any_dev64_reference_or_label_access": True,
            "contains_exact_references_or_labels": False,
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
                "config": _record(_project_path(config["config_path"])),
                "module": _record(
                    PROJECT_ROOT
                    / "src/aiijc_puzzle/joint_relation_selector_portfolio.py"
                ),
                "runner": _record(Path(__file__)),
            },
        },
    )
    return {
        "schema": "aiijc-joint-relation-selector-scale256-portfolio-result-v1",
        "status": "target-free-portfolio-frozen-reference-scoring-not-authorized",
        "case_count": len(rows),
        "portfolio_member_count": len(PORTFOLIO_MEMBER_NAMES),
        "archive": _record(archive_path),
        "metadata": _record(metadata_path),
        "pre_reference_freeze": _record(freeze_path),
        "dev64_references_or_labels_accessed": False,
        "competition_test_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_sha256 = _load_signed_config(args.config)
    runtime_config = {**config, "config_path": _project_label(args.config)}
    result = freeze_target_free_portfolio(
        args.output_dir, runtime_config, config_sha256
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
