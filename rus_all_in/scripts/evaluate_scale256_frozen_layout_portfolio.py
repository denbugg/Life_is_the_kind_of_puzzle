#!/usr/bin/env python3
"""Fail-closed post-freeze evaluator for the scale256 DEV64 layout roster.

The checked-in protocol is deliberately unsigned and blocked.  A future,
separately signed copy may score only after the joint, six-arm, and portfolio
target-free receipts all verify.  Candidate selection is intentionally absent:
every preregistered layout is reported against the same frozen incumbent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, TypeVar

import numpy as np

from aiijc_puzzle.joint_relation_selector_consumer import (
    reject_target_bearing_array_names,
)
from aiijc_puzzle.joint_relation_selector_portfolio import PORTFOLIO_MEMBER_NAMES
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import ExactSyntheticReference, names_digest
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES
from aiijc_puzzle.tile_position_distance import evaluate_tile_position_distance

try:
    from scripts import run_joint_relation_selector_consumer as legacy
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_joint_relation_selector_consumer as legacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/scale256_frozen_layout_evaluator_unsigned_template_v1.json"
)
REPORT_RELATIVE_PATH = (
    "outputs/joint-relation-selector-portfolio/scale256-dev64-v1/"
    "score-fixed-one-shot-v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / REPORT_RELATIVE_PATH
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"

CONFIG_SCHEMA = "aiijc-scale256-frozen-layout-evaluator-protocol-v1"
REPORT_SCHEMA = "aiijc-scale256-frozen-layout-evaluator-report-v1"
SIGNED_STATUS = "signed-fixed-protocol"
BLOCKED_STATUS = "unsigned-template-blocked-before-dev64-reference-access"
PORTFOLIO_SCHEMA = "aiijc-joint-relation-selector-scale256-portfolio-v1"
PORTFOLIO_FREEZE_SCHEMA = (
    "aiijc-joint-relation-selector-scale256-portfolio-freeze-v1"
)
DEV_COUNT = 64
DEV_DIGEST = "5c6cb5b9b204a38c78e79936ff34235dae9896cfc13d6edaf12dfad635bcdb8e"
DEV_DRAW_INDEX = 0
DEV_CASE_SEED = 20260908
GRID_SIZE = 24
TILE_COUNT = 576
PAIR_COUNT = 1104
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260901
TIE_EPSILON = 1e-12

INCUMBENT_NAME = "incumbent_joint_layout"
RELATION_CANDIDATE_NAMES = tuple(f"relation_arm/{name}" for name in FUSION_ARM_NAMES)
PORTFOLIO_CANDIDATE_NAMES = tuple(
    f"portfolio/{name}" for name in PORTFOLIO_MEMBER_NAMES
)
CANDIDATE_NAMES = (
    INCUMBENT_NAME,
    *RELATION_CANDIDATE_NAMES,
    *PORTFOLIO_CANDIDATE_NAMES,
)

METRIC_DIRECTIONS: dict[str, int] = {
    "exact_count": 1,
    "exact_rate": 1,
    "satisfied_pairs": 1,
    "satisfied_pairs_rate": 1,
    "manhattan_l1_per_tile": -1,
    "radius_le_1_rate": 1,
    "radius_le_2_rate": 1,
}

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class VerifiedInputs:
    """Three hash-verified target-free bundles sharing one DEV64 identity roster."""

    joint: legacy.VerifiedBundle
    relation: legacy.VerifiedBundle
    portfolio: legacy.VerifiedBundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
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


def evaluator_contract() -> dict[str, Any]:
    """Return the complete policy frozen before any future DEV64 truth access."""

    return {
        "grid_size": GRID_SIZE,
        "tile_count": TILE_COUNT,
        "satisfied_pair_denominator": PAIR_COUNT,
        "one_shot_report_path": REPORT_RELATIVE_PATH,
        "incumbent_name": INCUMBENT_NAME,
        "candidate_names_in_report_order": list(CANDIDATE_NAMES),
        "metric_directions": {
            name: "higher_is_better" if direction > 0 else "lower_is_better"
            for name, direction in METRIC_DIRECTIONS.items()
        },
        "adjusted_satisfied_pairs_definition": (
            "paired candidate-minus-incumbent satisfied-pair count on the exact "
            "same synthetic case; no extra penalty or normalization"
        ),
        "distance_definition": (
            "absolute tile-to-true-position Manhattan distance in board cells; "
            "no translation or cyclic alignment"
        ),
        "source_statistics": {
            "cluster_key": "source_filename",
            "expected_source_count": DEV_COUNT,
            "expected_draw_indices": [DEV_DRAW_INDEX],
            "within_source_reducer": "arithmetic mean over registered cases",
            "quantile_method": "linear",
            "reported_distribution_fields": [
                "mean",
                "median",
                "q25",
                "q75",
                "minimum",
                "maximum",
            ],
            "win_tie_loss_basis": "benefit-oriented source-mean delta",
            "tie_epsilon": TIE_EPSILON,
            "bootstrap": {
                "unit": "source cluster",
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "interval": "equal-tail 95 percent",
                "shared_indices_across_candidates_and_metrics": True,
            },
            "tail_diagnostics": [
                "worst source by benefit delta",
                "largest-positive-source share of positive benefit mass",
                "mean after removing largest positive source",
            ],
        },
        "multiplicity_policy": {
            "report_every_declared_candidate_separately": True,
            "deduplicate_for_weighting_or_selection": False,
            "report_per_case_layout_equivalence_classes": True,
            "report_pairwise_equal-case-counts": True,
        },
        "policy": {
            "selection_or_promotion_from_dev64_labels": False,
            "candidate_ranking": False,
            "metric_or_threshold_tuning_after_open": False,
            "one_fixed_report_only": True,
            "competition_test_or_submission_access": False,
        },
    }


def rule_commitment_sha256(config: Mapping[str, Any]) -> str:
    payload = {
        "schema": config.get("schema"),
        "joint_protocol_sha256": config.get("joint_protocol_sha256"),
        "source_protocol": config.get("source_protocol"),
        "evaluation_contract": config.get("evaluation_contract"),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_record_shape(
    record: Any,
    *,
    name: str,
    pending_allowed: bool,
) -> None:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise RuntimeError(f"evaluator frozen input record is invalid: {name}")
    digest = record.get("sha256")
    if pending_allowed and digest == "__PENDING__":
        return
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(f"evaluator frozen input digest is invalid: {name}")


def _require_exact_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("scale256 evaluator protocol schema changed")
    status = config.get("status")
    if status == BLOCKED_STATUS:
        if config.get("execution_authorized") is not False:
            raise RuntimeError("blocked evaluator template cannot authorize execution")
    elif status == SIGNED_STATUS:
        if config.get("execution_authorized") is not True:
            raise RuntimeError("signed evaluator protocol did not authorize execution")
    else:
        raise RuntimeError("scale256 evaluator protocol status changed")
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
        raise RuntimeError("scale256 evaluator DEV64 source contract changed")
    if config.get("evaluation_contract") != evaluator_contract():
        raise RuntimeError("scale256 evaluator metric/statistics policy changed")
    if config.get("rule_commitment_sha256") != rule_commitment_sha256(config):
        raise RuntimeError("scale256 evaluator pre-reference commitment changed")
    required = {
        "joint_protocol",
        "joint_archive",
        "joint_metadata",
        "joint_pre_score_freeze",
        "relation_roster_protocol",
        "relation_roster_archive",
        "relation_roster_metadata",
        "relation_roster_pre_score_freeze",
        "portfolio_protocol",
        "portfolio_archive",
        "portfolio_metadata",
        "portfolio_pre_reference_freeze",
        "portfolio_module",
        "portfolio_runner",
        "legacy_runner",
        "layout_metric",
        "distance_metric",
        "synthetic_reference_helper",
        "board_loader",
        "validation_manifest",
        "evaluator_runner",
    }
    frozen = config.get("frozen_inputs", {})
    if not isinstance(frozen, Mapping) or set(frozen) != required:
        raise RuntimeError("scale256 evaluator frozen-input inventory changed")
    is_blocked = status == BLOCKED_STATUS
    pending_names = {
        "portfolio_protocol",
        "portfolio_archive",
        "portfolio_metadata",
        "portfolio_pre_reference_freeze",
        "portfolio_module",
        "portfolio_runner",
        "evaluator_runner",
    }
    for name, record in frozen.items():
        _require_record_shape(
            record,
            name=name,
            pending_allowed=is_blocked and name in pending_names,
        )
    if config.get("joint_protocol_sha256") != frozen["joint_protocol"].get("sha256"):
        raise RuntimeError("evaluator joint protocol digest changed")


def _verified_record(config: Mapping[str, Any], name: str) -> Path:
    record = config.get("frozen_inputs", {}).get(name)
    _require_record_shape(record, name=name, pending_allowed=False)
    path = _project_path(str(record["path"]))
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"evaluator frozen input changed or is absent: {name}")
    return path


def _verify_signed_json_record(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    path = _verified_record(config, name)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != digest
    ):
        raise RuntimeError(f"signed upstream sidecar changed or is absent: {name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != SIGNED_STATUS:
        raise RuntimeError(f"upstream protocol is not signed/fixed: {name}")
    return value


def _load_signed_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    _require_exact_contract(config)
    if config.get("status") == BLOCKED_STATUS:
        raise RuntimeError("scale256 evaluator template is intentionally blocked")
    digest = sha256_file(resolved)
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if (
        config.get("status") != SIGNED_STATUS
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != digest
    ):
        raise RuntimeError("scale256 evaluator config is not separately signed")
    for name in config["frozen_inputs"]:
        _verified_record(config, name)
    if _verified_record(config, "evaluator_runner") != Path(__file__).resolve():
        raise RuntimeError("signed evaluator runner path changed")
    joint_protocol = _verify_signed_json_record(config, "joint_protocol")
    relation_protocol = _verify_signed_json_record(config, "relation_roster_protocol")
    portfolio_protocol = _verify_signed_json_record(config, "portfolio_protocol")
    joint_source = joint_protocol.get("source_contract", {})
    if (
        joint_source.get("reserved_dev_source_count") != DEV_COUNT
        or joint_source.get("reserved_dev_digest") != DEV_DIGEST
        or joint_source.get("dev_draw_index") != DEV_DRAW_INDEX
        or joint_source.get("dev_case_seed") != DEV_CASE_SEED
    ):
        raise RuntimeError("joint scale256 protocol differs from evaluator DEV64")
    for label, protocol in (
        ("relation", relation_protocol),
        ("portfolio", portfolio_protocol),
    ):
        if protocol.get("source_protocol") != config["source_protocol"]:
            raise RuntimeError(f"{label} protocol differs from evaluator DEV64")
    return config, digest


def _verify_receipt_record(
    receipt: Mapping[str, Any],
    section: str,
    name: str,
    path: Path,
) -> None:
    record = receipt.get(section, {}).get(name)
    if not path.is_file():
        raise RuntimeError(f"target-free receipt artifact is absent: {section}.{name}")
    expected = sha256_file(path)
    if (
        not isinstance(record, Mapping)
        or record.get("path") != _project_label(path)
        or record.get("sha256") != expected
    ):
        raise RuntimeError(f"target-free receipt does not bind {section}.{name}")


def _verify_joint_lineage(bundle: legacy.VerifiedBundle) -> None:
    freeze = json.loads(bundle.freeze.read_text(encoding="utf-8"))
    _verify_receipt_record(freeze, "artifacts", "archive", bundle.archive)
    _verify_receipt_record(freeze, "artifacts", "metadata", bundle.metadata)


def _verify_relation_lineage(
    config: Mapping[str, Any], bundle: legacy.VerifiedBundle
) -> None:
    protocol = _verified_record(config, "relation_roster_protocol")
    freeze = json.loads(bundle.freeze.read_text(encoding="utf-8"))
    protocol_sha = sha256_file(protocol)
    if freeze.get("config_sha256") != protocol_sha:
        raise RuntimeError("relation roster receipt belongs to another protocol")
    _verify_receipt_record(freeze, "artifacts", "archive", bundle.archive)
    _verify_receipt_record(freeze, "artifacts", "metadata", bundle.metadata)
    _verify_receipt_record(freeze, "artifacts", "preregistration", protocol)
    if int(freeze.get("source_case_count", -1)) != DEV_COUNT:
        raise RuntimeError("relation roster receipt is not DEV64")


def verify_portfolio_bundle(
    config: Mapping[str, Any],
    *,
    joint: legacy.VerifiedBundle,
    relation: legacy.VerifiedBundle,
) -> legacy.VerifiedBundle:
    archive = _verified_record(config, "portfolio_archive")
    metadata = _verified_record(config, "portfolio_metadata")
    freeze_path = _verified_record(config, "portfolio_pre_reference_freeze")
    protocol = _verified_record(config, "portfolio_protocol")
    module = _verified_record(config, "portfolio_module")
    runner = _verified_record(config, "portfolio_runner")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != PORTFOLIO_FREEZE_SCHEMA:
        raise RuntimeError("portfolio pre-reference receipt schema changed")
    if freeze.get("created_before_any_dev64_reference_or_label_access") is not True:
        raise RuntimeError("portfolio was not frozen before DEV64 references")
    if freeze.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("portfolio receipt contains references or labels")
    if freeze.get("config_sha256") != sha256_file(protocol):
        raise RuntimeError("portfolio receipt belongs to another protocol")
    for name, path in (
        ("joint_archive", joint.archive),
        ("joint_metadata", joint.metadata),
        ("joint_pre_score_freeze", joint.freeze),
        ("relation_roster_archive", relation.archive),
        ("relation_roster_metadata", relation.metadata),
        ("relation_roster_pre_score_freeze", relation.freeze),
    ):
        _verify_receipt_record(freeze, "input_artifacts", name, path)
    for name, path in (
        ("archive", archive),
        ("metadata", metadata),
        ("config", protocol),
        ("module", module),
        ("runner", runner),
    ):
        _verify_receipt_record(freeze, "artifacts", name, path)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    expected_flags = {
        "schema": PORTFOLIO_SCHEMA,
        "status": "target-free-portfolio-frozen-no-reference-access",
        "config_sha256": sha256_file(protocol),
        "contains_exact_references_or_labels": False,
        "contains_pixels": False,
        "all_outputs_exact_existing_strict_upright_layouts": True,
        "arm_names": list(FUSION_ARM_NAMES),
        "portfolio_member_names": list(PORTFOLIO_MEMBER_NAMES),
    }
    for name, expected in expected_flags.items():
        if payload.get(name) != expected:
            raise RuntimeError(f"portfolio metadata contract changed: {name}")
    rows = tuple(payload.get("rows", ()))
    expected_names = tuple(config["source_protocol"]["dev_filenames"])
    if tuple(row.get("source_filename") for row in rows) != expected_names:
        raise RuntimeError("portfolio metadata differs from evaluator DEV64 order")
    for row in rows:
        if tuple(row.get("portfolio_member_names", ())) != tuple(
            PORTFOLIO_MEMBER_NAMES
        ):
            raise RuntimeError("portfolio row member order changed")
    return legacy.VerifiedBundle(
        archive=archive,
        metadata=metadata,
        freeze=freeze_path,
        rows=rows,
        archive_sha256=sha256_file(archive),
        metadata_sha256=sha256_file(metadata),
    )


def _validate_bundle_identities(
    joint: legacy.VerifiedBundle,
    relation: legacy.VerifiedBundle,
    portfolio: legacy.VerifiedBundle,
) -> None:
    if not all(len(bundle.rows) == DEV_COUNT for bundle in (joint, relation, portfolio)):
        raise RuntimeError("frozen evaluator inputs are not all DEV64")
    for index, rows in enumerate(
        zip(joint.rows, relation.rows, portfolio.rows, strict=True)
    ):
        identities = {_identity(row) for row in rows}
        if len(identities) != 1:
            raise RuntimeError(f"frozen tile-bag identity mismatch at case {index}")


def verify_frozen_inputs(config: Mapping[str, Any]) -> VerifiedInputs:
    """Verify every target-free hash and cross-bundle identity before truth access."""

    joint = legacy.verify_joint_bundle(config)
    _verify_joint_lineage(joint)
    relation = legacy.verify_relation_roster_bundle(config)
    _verify_relation_lineage(config, relation)
    portfolio = verify_portfolio_bundle(config, joint=joint, relation=relation)
    _validate_bundle_identities(joint, relation, portfolio)
    return VerifiedInputs(joint=joint, relation=relation, portfolio=portfolio)


def score_after_verified_inputs(
    config: Mapping[str, Any],
    *,
    reference_loader: Callable[[VerifiedInputs], T],
    scorer: Callable[[VerifiedInputs, T], R],
    verifier: Callable[[Mapping[str, Any]], VerifiedInputs] = verify_frozen_inputs,
) -> R:
    """Dependency-injected proof that all receipts precede reference loading."""

    verified = verifier(config)
    references = reference_loader(verified)
    return scorer(verified, references)


def _strict_layout(value: Any, *, grid: int, name: str) -> np.ndarray:
    count = grid * grid
    layout = np.asarray(value)
    if layout.shape != (count,) or not np.issubdtype(layout.dtype, np.integer):
        raise RuntimeError(f"{name} is not an integer layout of shape {(count,)}")
    result = np.asarray(layout, dtype=np.int32)
    if not np.array_equal(np.sort(result), np.arange(count, dtype=np.int32)):
        raise RuntimeError(f"{name} is not a strict tile permutation")
    return np.ascontiguousarray(result)


def load_case_candidates(
    relation_arrays: Mapping[str, Any],
    portfolio_arrays: Mapping[str, Any],
    portfolio_row: Mapping[str, Any],
    *,
    grid: int = GRID_SIZE,
) -> dict[str, np.ndarray]:
    """Load the fixed declared roster and enforce its canonical incumbent."""

    incumbent = _strict_layout(
        relation_arrays["relation_truth_selector_layout"],
        grid=grid,
        name=INCUMBENT_NAME,
    )
    candidates = {INCUMBENT_NAME: incumbent}
    for arm, candidate_name in zip(
        FUSION_ARM_NAMES, RELATION_CANDIDATE_NAMES, strict=True
    ):
        candidates[candidate_name] = _strict_layout(
            relation_arrays[f"relation_arm_{arm}_layout"],
            grid=grid,
            name=candidate_name,
        )
    names = tuple(portfolio_row.get("portfolio_member_names", ()))
    layouts = np.asarray(portfolio_arrays["portfolio_layouts"])
    selected_indices = np.asarray(portfolio_arrays["selected_arm_indices"])
    if names != tuple(PORTFOLIO_MEMBER_NAMES):
        raise RuntimeError("portfolio row member order changed")
    if layouts.shape != (len(PORTFOLIO_MEMBER_NAMES), grid * grid):
        raise RuntimeError("portfolio layout matrix shape changed")
    if (
        selected_indices.shape != (len(PORTFOLIO_MEMBER_NAMES),)
        or not np.issubdtype(selected_indices.dtype, np.integer)
        or np.any(selected_indices < 0)
        or np.any(selected_indices >= len(FUSION_ARM_NAMES))
    ):
        raise RuntimeError("portfolio selected-arm indices changed")
    selected_indices = np.asarray(selected_indices, dtype=np.int32)
    selected_names = tuple(FUSION_ARM_NAMES[int(index)] for index in selected_indices)
    if selected_names != tuple(portfolio_row.get("selected_arms", ())):
        raise RuntimeError("portfolio selected-arm lineage changed")
    for index, candidate_name in enumerate(PORTFOLIO_CANDIDATE_NAMES):
        candidates[candidate_name] = _strict_layout(
            layouts[index], grid=grid, name=candidate_name
        )
        relation_name = RELATION_CANDIDATE_NAMES[int(selected_indices[index])]
        if not np.array_equal(candidates[candidate_name], candidates[relation_name]):
            raise RuntimeError(
                f"{candidate_name} is not its declared existing relation-arm layout"
            )
    if not np.array_equal(candidates[PORTFOLIO_CANDIDATE_NAMES[0]], incumbent):
        raise RuntimeError("portfolio incumbent_keep differs from canonical incumbent")
    changed = tuple(
        not np.array_equal(candidates[name], incumbent)
        for name in PORTFOLIO_CANDIDATE_NAMES
    )
    if changed != tuple(bool(value) for value in portfolio_row.get("changed_from_incumbent", ())):
        raise RuntimeError("portfolio changed-from-incumbent lineage changed")
    if tuple(candidates) != CANDIDATE_NAMES:
        raise RuntimeError("declared evaluator candidate order changed")
    return candidates


def evaluate_candidate_layout(
    layout: Any,
    exact_reference: Any,
    *,
    grid: int = GRID_SIZE,
) -> dict[str, int | float]:
    """Compute the fixed absolute metric vector for one strict layout."""

    candidate = _strict_layout(layout, grid=grid, name="candidate")
    reference = _strict_layout(exact_reference, grid=grid, name="exact_reference")
    adjacency = evaluate_layout(candidate, reference, reference_is_exact=True)
    distance = evaluate_tile_position_distance(candidate, reference, grid=grid)
    if (
        adjacency.correct_tile_count != distance.exact_tile_count
        or adjacency.direct_placement != distance.within_radius_0_recall
    ):
        raise RuntimeError("trusted exact and distance metric helpers disagree")
    return {
        "exact_count": adjacency.correct_tile_count,
        "exact_rate": adjacency.direct_placement,
        "satisfied_pairs": adjacency.adjacency_correct,
        "satisfied_pairs_rate": adjacency.adjacency,
        "manhattan_l1_per_tile": distance.mean_manhattan_cells,
        "radius_le_1_rate": distance.within_radius_1_recall,
        "radius_le_2_rate": distance.within_radius_2_recall,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    current = np.asarray(values, dtype=np.float64)
    if current.ndim != 1 or len(current) == 0 or not np.isfinite(current).all():
        raise ValueError("distribution requires a non-empty finite vector")
    return {
        "mean": float(current.mean()),
        "median": float(np.quantile(current, 0.50, method="linear")),
        "q25": float(np.quantile(current, 0.25, method="linear")),
        "q75": float(np.quantile(current, 0.75, method="linear")),
        "minimum": float(current.min()),
        "maximum": float(current.max()),
    }


def _source_means(
    source_names: Sequence[str], values: Sequence[float]
) -> tuple[tuple[str, ...], np.ndarray, dict[str, int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for source, value in zip(source_names, values, strict=True):
        grouped[str(source)].append(float(value))
    names = tuple(grouped)
    means = np.asarray(
        [np.mean(grouped[source], dtype=np.float64) for source in names],
        dtype=np.float64,
    )
    multiplicity = {source: len(grouped[source]) for source in names}
    return names, means, multiplicity


def _shared_bootstrap_indices(source_count: int) -> np.ndarray:
    if source_count != DEV_COUNT:
        raise RuntimeError("fixed evaluator bootstrap requires exactly 64 sources")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(
        0,
        source_count,
        size=(BOOTSTRAP_RESAMPLES, source_count),
        dtype=np.int32,
    )


def summarize_absolute_metric(
    source_names: Sequence[str],
    values: Sequence[float],
    *,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    case_values = np.asarray(values, dtype=np.float64)
    sources, source_values, multiplicity = _source_means(source_names, case_values)
    if len(sources) != DEV_COUNT or bootstrap_indices.shape != (
        BOOTSTRAP_RESAMPLES,
        DEV_COUNT,
    ):
        raise RuntimeError("source-clustered absolute contract requires fixed DEV64")
    bootstrap_mean = source_values[bootstrap_indices].mean(axis=1)
    return {
        "case_distribution": _distribution(case_values),
        "source_clustered": {
            "source_count": len(sources),
            "case_multiplicity": multiplicity,
            "source_mean_distribution": _distribution(source_values),
            "mean_bootstrap_95pct_ci": [
                float(np.quantile(bootstrap_mean, 0.025, method="linear")),
                float(np.quantile(bootstrap_mean, 0.975, method="linear")),
            ],
        },
    }


def summarize_delta_metric(
    source_names: Sequence[str],
    raw_deltas: Sequence[float],
    *,
    direction: int,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    """Report fixed source-clustered raw and benefit-oriented delta statistics."""

    if direction not in (-1, 1):
        raise ValueError("metric direction must be -1 or +1")
    raw_case = np.asarray(raw_deltas, dtype=np.float64)
    sources, raw_source, multiplicity = _source_means(source_names, raw_case)
    if len(sources) != DEV_COUNT or bootstrap_indices.shape != (
        BOOTSTRAP_RESAMPLES,
        DEV_COUNT,
    ):
        raise RuntimeError("source-clustered delta contract requires fixed DEV64")
    benefit_case = direction * raw_case
    benefit_source = direction * raw_source
    bootstrap_raw = raw_source[bootstrap_indices].mean(axis=1)
    bootstrap_benefit = benefit_source[bootstrap_indices].mean(axis=1)
    positive = np.maximum(benefit_source, 0.0)
    positive_total = float(positive.sum())
    positive_index = int(np.argmax(positive))
    worst_index = int(np.argmin(benefit_source))
    return {
        "direction": "higher_is_better" if direction > 0 else "lower_is_better",
        "case_raw_delta_distribution": _distribution(raw_case),
        "case_benefit_delta_distribution": _distribution(benefit_case),
        "source_clustered": {
            "source_count": len(sources),
            "case_multiplicity": multiplicity,
            "raw_delta_distribution": _distribution(raw_source),
            "benefit_delta_distribution": _distribution(benefit_source),
            "wins_ties_losses": {
                "wins": int(np.count_nonzero(benefit_source > TIE_EPSILON)),
                "ties": int(np.count_nonzero(np.abs(benefit_source) <= TIE_EPSILON)),
                "losses": int(np.count_nonzero(benefit_source < -TIE_EPSILON)),
            },
            "raw_mean_bootstrap_95pct_ci": [
                float(np.quantile(bootstrap_raw, 0.025, method="linear")),
                float(np.quantile(bootstrap_raw, 0.975, method="linear")),
            ],
            "benefit_mean_bootstrap_95pct_ci": [
                float(np.quantile(bootstrap_benefit, 0.025, method="linear")),
                float(np.quantile(bootstrap_benefit, 0.975, method="linear")),
            ],
            "worst_source": {
                "source_filename": sources[worst_index],
                "raw_delta": float(raw_source[worst_index]),
                "benefit_delta": float(benefit_source[worst_index]),
            },
            "positive_mass": {
                "total": positive_total,
                "largest_source_filename": (
                    sources[positive_index] if positive_total else None
                ),
                "largest_source_benefit_delta": (
                    float(positive[positive_index]) if positive_total else 0.0
                ),
                "largest_source_share": (
                    float(positive[positive_index] / positive_total)
                    if positive_total
                    else 0.0
                ),
                "mean_after_removing_largest_positive_source": (
                    float(
                        (benefit_source.sum() - benefit_source[positive_index])
                        / (len(benefit_source) - 1)
                    )
                    if positive_total and len(benefit_source) > 1
                    else float(benefit_source.mean())
                ),
            },
        },
    }


def layout_sha256(layout: Any) -> str:
    value = np.ascontiguousarray(np.asarray(layout, dtype="<i4"))
    return hashlib.sha256(value.tobytes()).hexdigest()


def layout_equivalence_classes(
    layouts: Mapping[str, np.ndarray],
) -> tuple[list[list[str]], dict[str, str]]:
    digests = {name: layout_sha256(layout) for name, layout in layouts.items()}
    classes: dict[str, list[str]] = {}
    for name in CANDIDATE_NAMES:
        classes.setdefault(digests[name], []).append(name)
    return list(classes.values()), digests


def summarize_multiplicity(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique_counts = np.asarray(
        [len(case["layout_equivalence_classes"]) for case in cases], dtype=np.float64
    )
    same_as_incumbent = {
        name: sum(
            case["layout_sha256"][name]
            == case["layout_sha256"][INCUMBENT_NAME]
            for case in cases
        )
        for name in CANDIDATE_NAMES
    }
    equal_pairs = []
    for left_index, left in enumerate(CANDIDATE_NAMES):
        for right in CANDIDATE_NAMES[left_index + 1 :]:
            equal_pairs.append(
                {
                    "left": left,
                    "right": right,
                    "equal_case_count": sum(
                        case["layout_sha256"][left]
                        == case["layout_sha256"][right]
                        for case in cases
                    ),
                }
            )
    return {
        "declared_candidate_count": len(CANDIDATE_NAMES),
        "unique_layout_count_per_case": _distribution(unique_counts),
        "same_as_incumbent_case_count": same_as_incumbent,
        "pairwise_equal_case_counts": equal_pairs,
        "deduplicated_for_weighting_or_selection": False,
    }


def score_frozen_candidates(
    verified: VerifiedInputs,
    references: Mapping[str, ExactSyntheticReference],
) -> dict[str, Any]:
    """Score all declared candidates once, without choosing among them."""

    case_ids = tuple(str(row["case_id"]) for row in verified.joint.rows)
    if set(references) != set(case_ids):
        raise RuntimeError("exact reference roster differs from frozen DEV64 cases")
    cases: list[dict[str, Any]] = []
    with (
        np.load(verified.relation.archive, allow_pickle=False) as relation_archive,
        np.load(verified.portfolio.archive, allow_pickle=False) as portfolio_archive,
    ):
        reject_target_bearing_array_names(tuple(relation_archive.files))
        reject_target_bearing_array_names(tuple(portfolio_archive.files))
        for relation_row, portfolio_row in zip(
            verified.relation.rows, verified.portfolio.rows, strict=True
        ):
            relation_arrays = _case_arrays(
                relation_archive, str(relation_row["prefix"])
            )
            portfolio_arrays = _case_arrays(
                portfolio_archive, str(portfolio_row["prefix"])
            )
            layouts = load_case_candidates(
                relation_arrays, portfolio_arrays, portfolio_row
            )
            reference = references[str(relation_row["case_id"])].tile_at_position
            metrics = {
                name: evaluate_candidate_layout(layout, reference)
                for name, layout in layouts.items()
            }
            incumbent = metrics[INCUMBENT_NAME]
            deltas = {
                name: {
                    metric: float(values[metric]) - float(incumbent[metric])
                    for metric in METRIC_DIRECTIONS
                }
                for name, values in metrics.items()
            }
            classes, digests = layout_equivalence_classes(layouts)
            cases.append(
                {
                    "case_id": relation_row["case_id"],
                    "source_filename": relation_row["source_filename"],
                    "draw_index": int(relation_row["draw_index"]),
                    "metrics": metrics,
                    "delta_vs_incumbent": deltas,
                    "adjusted_satisfied_pairs": {
                        name: deltas[name]["satisfied_pairs"]
                        for name in CANDIDATE_NAMES
                    },
                    "layout_sha256": digests,
                    "layout_equivalence_classes": classes,
                }
            )
    source_names = [str(case["source_filename"]) for case in cases]
    if len(set(source_names)) != DEV_COUNT or any(
        int(case["draw_index"]) != DEV_DRAW_INDEX for case in cases
    ):
        raise RuntimeError("scored cases do not satisfy fixed 64-source/draw contract")
    bootstrap_indices = _shared_bootstrap_indices(DEV_COUNT)
    summaries: dict[str, Any] = {}
    for candidate in CANDIDATE_NAMES:
        absolute = {}
        comparisons = {}
        for metric, direction in METRIC_DIRECTIONS.items():
            absolute[metric] = summarize_absolute_metric(
                source_names,
                [float(case["metrics"][candidate][metric]) for case in cases],
                bootstrap_indices=bootstrap_indices,
            )
            comparisons[metric] = summarize_delta_metric(
                source_names,
                [
                    float(case["delta_vs_incumbent"][candidate][metric])
                    for case in cases
                ],
                direction=direction,
                bootstrap_indices=bootstrap_indices,
            )
        summaries[candidate] = {
            "absolute": absolute,
            "delta_vs_incumbent": comparisons,
            "adjusted_satisfied_pairs": comparisons["satisfied_pairs"],
        }
    return {
        "case_count": len(cases),
        "source_count": len(set(source_names)),
        "candidate_names_in_report_order": list(CANDIDATE_NAMES),
        "candidate_summaries": summaries,
        "multiplicity": summarize_multiplicity(cases),
        "cases": cases,
        "verified_target_free_inputs": {
            "joint_archive_sha256": verified.joint.archive_sha256,
            "joint_metadata_sha256": verified.joint.metadata_sha256,
            "relation_archive_sha256": verified.relation.archive_sha256,
            "relation_metadata_sha256": verified.relation.metadata_sha256,
            "portfolio_archive_sha256": verified.portfolio.archive_sha256,
            "portfolio_metadata_sha256": verified.portfolio.metadata_sha256,
            "all_receipts_verified_before_reference_loading": True,
        },
        "selection_or_promotion_performed": False,
        "all_outputs_strict_original_upright_tile_permutations": True,
    }


def _open_json_exclusive(path: Path) -> TextIO:
    """Claim the fixed report path before any exact-reference access."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x", encoding="utf-8")


def execute_one_shot_report(
    config: Mapping[str, Any],
    config_sha256: str,
    report_path: Path,
    *,
    reference_loader: Callable[[VerifiedInputs], T],
    scorer: Callable[[VerifiedInputs, T], Mapping[str, Any]],
    verifier: Callable[[Mapping[str, Any]], VerifiedInputs] = verify_frozen_inputs,
) -> dict[str, Any]:
    """Verify target-free inputs, claim output, then open truth exactly once.

    A failure after the exclusive claim deliberately leaves the fixed report
    path present.  Retrying then requires a separately reviewed protocol and
    output path instead of silently reopening DEV64 under the same commitment.
    """

    verified = verifier(config)
    with _open_json_exclusive(report_path) as stream:
        references = reference_loader(verified)
        metrics = dict(scorer(verified, references))
        report = {
            "schema": REPORT_SCHEMA,
            "status": "complete-fixed-one-shot-report-all-no-selection",
            "config_sha256": config_sha256,
            "evaluation_contract": config["evaluation_contract"],
            **metrics,
            "dev64_reference_access_occurred_only_after_all_receipt_checks": True,
            "competition_test_or_submission_accessed": False,
        }
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
    return report


def _require_runtime_paths(config: Mapping[str, Any], args: argparse.Namespace) -> Path:
    expected_manifest = _project_path(
        config["frozen_inputs"]["validation_manifest"]["path"]
    )
    if args.manifest.resolve() != expected_manifest:
        raise RuntimeError("runtime manifest differs from signed evaluator protocol")
    if args.targets.resolve() != DEFAULT_TARGETS.resolve():
        raise RuntimeError("runtime target path differs from signed evaluator protocol")
    expected_report = _project_path(
        config["evaluation_contract"]["one_shot_report_path"]
    )
    if args.report.resolve() != expected_report:
        raise RuntimeError("runtime report path differs from signed evaluator protocol")
    return expected_report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_sha256 = _load_signed_config(args.config)
    expected_report = _require_runtime_paths(config, args)
    report = execute_one_shot_report(
        config,
        config_sha256,
        expected_report,
        reference_loader=lambda verified: legacy._load_exact_dev_references(
            verified.joint, args=args, config=config
        ),
        scorer=lambda verified, references: score_frozen_candidates(
            verified, references
        ),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
