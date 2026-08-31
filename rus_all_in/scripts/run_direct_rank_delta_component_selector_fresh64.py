#!/usr/bin/env python3
"""Preregister and run a source-disjoint fresh64 selector confirmation.

The frozen treatment is the target-blind whole-layout selector already used on
the opened Union-v2 panel.  For every board it compares the Union-v2 and
direct rank-delta translation-component builds lexicographically by
``(consistent redundant constraints, largest component)``; exact evidence
ties keep Union-v2.

``selection`` is metadata-only.  It validates the canonical roster audit,
excludes its complete 3,064-source organizer-train union and the 80 sources
reserved by the learned Union-priority pilot, then freezes a new deterministic
64-source roster and every upstream SHA-256 before any selected target image is
read.  ``run`` first freezes all target-free arm decisions, hard-edge
priorities, and strict layouts, and only then recreates exact references for
scoring.  There is no threshold, budget, seed, or arm sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.direct_hard_edge_production import (
    FROZEN_DIRECT_HARD_EDGE_SHA256,
    infer_direct_hard_edge_priorities,
    load_direct_hard_edge_checkpoint,
)
from aiijc_puzzle.direct_rank_delta_component_selector import (
    select_direct_rank_delta_component_arm,
)
from aiijc_puzzle.direct_residual_union_priority import (
    build_direct_rank_delta_union_priority,
)
from aiijc_puzzle.protocol import (
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
)
from aiijc_puzzle.raw_twin_union_production import (
    FROZEN_UNION_CHECKPOINT_SHA256,
    infer_raw_twin_union_assignments,
    load_fullres_twin_checkpoint,
    load_raw_twin_union_checkpoint,
)
from aiijc_puzzle.socket_decoder import build_translation_components
from aiijc_puzzle.socket_sorter_production import (
    DECODER_EDGE_BUDGET,
    load_socket_checkpoint,
)
from aiijc_puzzle.synthetic_socket_evaluation import names_digest

try:
    from scripts.run_direct_rank_delta_component_selector_opened64 import (
        ARM_NAMES,
        METRIC_NAMES,
        _comparison_metrics,
        _component_evidence,
        _mean,
        _report_path,
    )
    from scripts.run_direct_residual_union_priority_opened64 import (
        COUNT,
        DIRECT_CHECKPOINT,
        GRID,
        PROJECT_ROOT,
        UNION_CONFIG,
        _decode_layout,
        _edge_arrays,
        _fixed_top144_correct,
        _strict_layout,
    )
    from scripts.run_fullres_twin_side_matcher import (
        _atomic_json,
        _prepare_boards,
        _two_view_case,
    )
    from scripts.run_raw_twin_union_reranker_fresh64 import (
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        SOCKET_CHECKPOINT,
        TWIN_CHECKPOINT,
        UNION_CHECKPOINT,
        UNION_SELECTION,
        _case_seeds,
    )
    from scripts.run_raw_twin_union_reranker_v2 import _adjacency_fraction
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_direct_rank_delta_component_selector_opened64 import (
        ARM_NAMES,
        METRIC_NAMES,
        _comparison_metrics,
        _component_evidence,
        _mean,
        _report_path,
    )
    from run_direct_residual_union_priority_opened64 import (
        COUNT,
        DIRECT_CHECKPOINT,
        GRID,
        PROJECT_ROOT,
        UNION_CONFIG,
        _decode_layout,
        _edge_arrays,
        _fixed_top144_correct,
        _strict_layout,
    )
    from run_fullres_twin_side_matcher import _atomic_json, _prepare_boards, _two_view_case
    from run_raw_twin_union_reranker_fresh64 import (
        DEFAULT_MANIFEST,
        DEFAULT_TARGETS,
        SOCKET_CHECKPOINT,
        TWIN_CHECKPOINT,
        UNION_CHECKPOINT,
        UNION_SELECTION,
        _case_seeds,
    )
    from run_raw_twin_union_reranker_v2 import _adjacency_fraction


SCHEMA = "aiijc-direct-rank-delta-component-selector-fresh64-confirmation-v1"
SELECTION_NAMESPACE = "aiijc-direct-rank-delta-component-selector-fresh64-confirmation-v1"
SELECTION_SEED = 721_058_253
SYNTHETIC_SEED = 367_570_216
BOOTSTRAP_SEED = 344_962_880
EXPECTED_SOURCES = 64
BOOTSTRAP_RESAMPLES = 20_000
AUDIT_SHA256 = "2f3105037095c5c1ebc1c116d5fea3689d0dbd540bf8e9b746d5f429659d8dea"
AUDIT_EXCLUDED_TRAIN_COUNT = 3_064
AUDIT_EXCLUDED_TRAIN_DIGEST = "96560e08a123d9d53ffc981388e67bd9c7a8943fa9a59f77347a84b2c31922b6"
PILOT_SOURCE_COUNT = 80
PILOT_FIT_ORDER_DIGEST = "2cafca0d2d231857afb626dc84335dc38740a9fdd3c1f6427376fa1f5a3c78fc"
PILOT_EVAL_ORDER_DIGEST = "13f7fe84262f9c4d0aee7ce80dfdc1edeec3ce7f1b5082f06ae5c6aceda6fa5f"

DEFAULT_AUDIT = PROJECT_ROOT / "outputs/union-hard-edge-priority/roster-audit-v1.json"
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/direct_rank_delta_component_selector_fresh64_confirmation_v1.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/direct-rank-delta-component-selector/fresh64-v1"
OPENED_SELECTOR_REPORT = (
    PROJECT_ROOT / "outputs/direct-rank-delta-component-selector/opened64-v1/report.json"
)
SELECTOR_IMPLEMENTATION = PROJECT_ROOT / "src/aiijc_puzzle/direct_rank_delta_component_selector.py"
RANK_DELTA_IMPLEMENTATION = PROJECT_ROOT / "src/aiijc_puzzle/direct_residual_union_priority.py"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("selection", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _dirty_sha256(tiles: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(tiles).tobytes()).hexdigest()


def _ordered_names(value: Any, *, field: str, expected: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(name, str) and Path(name).name == name and name.endswith(".png")
        for name in value
    ):
        raise ValueError(f"{field} must be a list of PNG basenames")
    names = tuple(value)
    if len(names) != expected or len(set(names)) != expected:
        raise ValueError(f"{field} must contain {expected} unique names")
    return names


def load_roster_audit(path: Path) -> tuple[dict[str, Any], str, set[str], tuple[str, ...]]:
    """Validate the canonical metadata-only audit and return exclusions.

    The audit contains the exact exclusion membership, so this function does
    not rescan mutable output directories or infer roster fields.
    """

    observed = sha256_file(path)
    if observed != AUDIT_SHA256:
        raise ValueError("canonical roster-audit SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "aiijc-union-hard-priority-roster-audit-v1":
        raise ValueError("unsupported roster-audit schema")
    if payload.get("created_before_target_access") is not True:
        raise ValueError("roster audit lacks pre-target timing attestation")
    if payload.get("target_images_accessed") is not False:
        raise ValueError("roster audit is not target blind")

    selection = payload.get("selection")
    exclusion = payload.get("exclusion")
    if not isinstance(selection, Mapping) or not isinstance(exclusion, Mapping):
        raise ValueError("roster audit is missing selection or exclusion")
    excluded_names = _ordered_names(
        exclusion.get("excluded_train_filenames"),
        field="audit excluded_train_filenames",
        expected=AUDIT_EXCLUDED_TRAIN_COUNT,
    )
    if names_digest(excluded_names, sort_names=True) != AUDIT_EXCLUDED_TRAIN_DIGEST:
        raise ValueError("audit excluded-train membership digest mismatch")
    if (
        selection.get("excluded_train_count") != AUDIT_EXCLUDED_TRAIN_COUNT
        or selection.get("excluded_train_digest") != AUDIT_EXCLUDED_TRAIN_DIGEST
    ):
        raise ValueError("audit exclusion summary changed")

    fit = _ordered_names(
        selection.get("fit_source_filenames"),
        field="pilot fit_source_filenames",
        expected=64,
    )
    evaluation = _ordered_names(
        selection.get("eval_source_filenames"),
        field="pilot eval_source_filenames",
        expected=16,
    )
    if names_digest(fit) != PILOT_FIT_ORDER_DIGEST:
        raise ValueError("pilot fit roster order digest mismatch")
    if names_digest(evaluation) != PILOT_EVAL_ORDER_DIGEST:
        raise ValueError("pilot eval roster order digest mismatch")
    pilot = (*fit, *evaluation)
    if len(set(pilot)) != PILOT_SOURCE_COUNT:
        raise ValueError("pilot fit/eval rosters overlap")
    if set(pilot) & set(excluded_names):
        raise ValueError("pilot roster unexpectedly overlaps its exclusion union")
    return payload, observed, set(excluded_names), pilot


def _frozen_input_records(audit_path: Path, manifest_path: Path) -> dict[str, dict[str, str]]:
    paths = {
        "roster_audit": audit_path,
        "manifest": manifest_path,
        "confirmation_runner": Path(__file__).resolve(),
        "selector_implementation": SELECTOR_IMPLEMENTATION,
        "rank_delta_implementation": RANK_DELTA_IMPLEMENTATION,
        "opened_selector_report": OPENED_SELECTOR_REPORT,
        "socket_checkpoint": SOCKET_CHECKPOINT,
        "twin_checkpoint": TWIN_CHECKPOINT,
        "union_checkpoint": UNION_CHECKPOINT,
        "union_config": UNION_CONFIG,
        "union_selection": UNION_SELECTION,
        "direct_checkpoint": DIRECT_CHECKPOINT,
    }
    return {
        name: {"path": _project_path(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _write_config_and_sidecar(path: Path, payload: Mapping[str, Any]) -> str:
    sidecar = path.with_name(f"{path.name}.sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite frozen fresh64 preregistration")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(path)
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def freeze_selection(args: argparse.Namespace) -> None:
    audit, audit_sha, audit_excluded, pilot = load_roster_audit(args.audit)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest mismatch")
    audit_manifest = audit["manifest"]
    if (
        sha256_file(args.manifest) != audit_manifest["sha256"]
        or manifest["protocol_digest"] != audit_manifest["protocol_digest"]
    ):
        raise ValueError("manifest differs from the canonical roster audit")
    train = manifest.get("splits", {}).get("train")
    if not isinstance(train, list):
        raise ValueError("manifest train split is absent")
    train_names = {str(record["filename"]) for record in train}

    excluded = audit_excluded | set(pilot)
    if not excluded <= train_names:
        raise ValueError("fresh64 exclusion includes a non-train source")
    excluded_order = tuple(sorted(excluded))
    excluded_digest = names_digest(excluded_order)
    selector_sha = sha256_file(SELECTOR_IMPLEMENTATION)
    effective_namespace = (
        f"{SELECTION_NAMESPACE}\0manifest={manifest['protocol_digest']}"
        f"\0audit={audit_sha}\0excluded={excluded_digest}\0selector={selector_sha}"
    )
    ranked = select_manifest_records(
        manifest,
        "train",
        limit=len(train),
        seed=SELECTION_SEED,
        namespace=effective_namespace,
    )
    records = tuple(record for record in ranked if str(record["filename"]) not in excluded)[
        :EXPECTED_SOURCES
    ]
    if len(records) != EXPECTED_SOURCES:
        raise ValueError("not enough source-disjoint organizer-train records")
    source_names = tuple(str(record["filename"]) for record in records)
    if set(source_names) & excluded:
        raise RuntimeError("fresh64 roster overlaps the frozen exclusion union")

    frozen_inputs = _frozen_input_records(args.audit, args.manifest)
    if frozen_inputs["roster_audit"]["sha256"] != AUDIT_SHA256:
        raise RuntimeError("roster audit changed during selection")
    payload = {
        "schema": SCHEMA,
        "status": "frozen-before-selected-target-access",
        "registered_before_selected_target_access": True,
        "registered_before_dirty_prediction_generation": True,
        "purpose": "source-disjoint confirmation of the frozen component-geometry selector",
        "frozen_inputs": frozen_inputs,
        "selection": {
            "split": "train",
            "namespace": SELECTION_NAMESPACE,
            "effective_namespace": effective_namespace,
            "selection_seed": SELECTION_SEED,
            "synthetic_seed": SYNTHETIC_SEED,
            "draw_indices": [0],
            "source_filenames": list(source_names),
            "source_order_digest": names_digest(source_names),
            "source_set_digest": names_digest(source_names, sort_names=True),
            "audit_excluded_train_count": len(audit_excluded),
            "audit_excluded_train_digest": names_digest(tuple(sorted(audit_excluded))),
            "learned_priority_pilot_source_count": len(pilot),
            "learned_priority_pilot_source_digest": names_digest(tuple(sorted(pilot))),
            "combined_excluded_train_count": len(excluded),
            "combined_excluded_train_digest": excluded_digest,
            "selected_exclusion_overlap": [],
        },
        "selector": {
            "rule": (
                "lexicographically maximize (consistent_redundant_constraints, "
                "largest_component); exact ties retain union_v2"
            ),
            "arm_granularity": "one complete strict layout per board",
            "candidate_supply": "unchanged Union-v2 hard projection",
            "rank_delta": (
                "frozen direct learned-minus-raw percentile-rank displacement on "
                "identical Union hard-edge identities; Union confidence multiset preserved"
            ),
            "component_edge_budget_per_axis": DECODER_EDGE_BUDGET,
            "threshold_budget_weight_or_seed_sweep": False,
            "retrain_or_recalibration": False,
        },
        "gate": {
            "all_required": {
                "exact_gain_vs_union_minimum_tiles_per_board": 0.25,
                "exact_gain_vs_rank_delta_minimum_tiles_per_board": 0.10,
                "adjacency_delta_vs_union_nonnegative": True,
                "all_three_arms_strict_original_permutations": True,
            },
            "clustered_ci": {
                "cluster": "source; exactly one draw per source",
                "confidence": 0.95,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "reported_not_thresholded": True,
            },
            "promotion_automatic": False,
        },
        "legality": {
            "organizer_train_only": True,
            "target_available_to_inference_or_selector": False,
            "layout": "strict permutation of all 576 original upright dirty tiles",
            "restored_pixels": "matcher-only evidence",
            "tile_rotation_warp_replacement_or_generation": False,
            "holdout_opened": False,
            "competition_test_opened": False,
        },
    }
    digest = _write_config_and_sidecar(args.config, payload)
    print(
        json.dumps(
            {
                "event": "fresh64_selector_preregistered",
                "path": str(args.config),
                "sha256": digest,
                "source_order_digest": names_digest(source_names),
                "combined_excluded": len(excluded),
                "selected_target_access": False,
            }
        ),
        flush=True,
    )


def load_confirmation_config(path: Path) -> tuple[dict[str, Any], str]:
    sidecar = path.with_name(f"{path.name}.sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("fresh64 config/sidecar SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported fresh64 selector schema")
    if payload.get("status") != "frozen-before-selected-target-access":
        raise ValueError("fresh64 selector was not preregistered before target access")
    if payload.get("registered_before_selected_target_access") is not True:
        raise ValueError("fresh64 selector timing contract is absent")
    for name, record in payload.get("frozen_inputs", {}).items():
        if not isinstance(record, Mapping):
            raise ValueError(f"frozen input {name} is malformed")
        source = _resolve_project_path(str(record.get("path", "")))
        if sha256_file(source) != record.get("sha256"):
            raise ValueError(f"frozen input changed after preregistration: {name}")
    return payload, observed


def evaluate_confirmation_gate(
    metrics: Mapping[str, Any],
    *,
    strict_layouts: int,
) -> dict[str, Any]:
    versus_union = metrics["component_selector_vs_union_v2"]
    versus_rank = metrics["component_selector_vs_rank_delta_transfer"]
    exact_union = float(versus_union["exact_tiles_delta"]["mean"])
    exact_rank = float(versus_rank["exact_tiles_delta"]["mean"])
    adjacency_union = float(versus_union["adjacency_delta"]["mean"])
    checks = {
        "exact_gain_vs_union_at_least_quarter_tile": {
            "observed": exact_union,
            "required": 0.25,
            "pass": exact_union >= 0.25,
        },
        "exact_gain_vs_rank_delta_at_least_tenth_tile": {
            "observed": exact_rank,
            "required": 0.10,
            "pass": exact_rank >= 0.10,
        },
        "adjacency_nonnegative_vs_union": {
            "observed": adjacency_union,
            "required": ">=0",
            "pass": adjacency_union >= 0.0,
        },
        "all_three_arms_strict": {
            "observed": strict_layouts,
            "required": 3 * EXPECTED_SOURCES,
            "pass": strict_layouts == 3 * EXPECTED_SOURCES,
        },
    }
    passed = all(bool(check["pass"]) for check in checks.values())
    return {
        "pass": passed,
        "status": "fresh64-confirmed" if passed else "fresh64-not-confirmed",
        "checks": checks,
        "clustered_ci_role": "reported honestly; not an additional threshold",
        "promotion_automatic": False,
        "competition_test_authorized": False,
    }


def _validated_run_roster(
    config: Mapping[str, Any],
    audit_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    _, _, audit_excluded, pilot = load_roster_audit(audit_path)
    selection = config["selection"]
    names = _ordered_names(
        selection.get("source_filenames"),
        field="fresh64 source_filenames",
        expected=EXPECTED_SOURCES,
    )
    if names_digest(names) != selection.get("source_order_digest"):
        raise ValueError("fresh64 source order digest mismatch")
    if names_digest(names, sort_names=True) != selection.get("source_set_digest"):
        raise ValueError("fresh64 source set digest mismatch")
    excluded = audit_excluded | set(pilot)
    if len(excluded) != selection.get("combined_excluded_train_count"):
        raise ValueError("fresh64 combined exclusion count mismatch")
    if names_digest(tuple(sorted(excluded))) != selection.get("combined_excluded_train_digest"):
        raise ValueError("fresh64 combined exclusion digest mismatch")
    if set(names) & excluded:
        raise ValueError("fresh64 roster overlaps the frozen exclusion union")
    train = manifest.get("splits", {}).get("train")
    if not isinstance(train, list):
        raise ValueError("manifest train split is absent")
    if set(names) - {str(record["filename"]) for record in train}:
        raise ValueError("fresh64 roster contains a non-train source")
    return names


def _selection_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    selection_counts = {
        arm: sum(row["selection"]["selected_arm"] == arm for row in rows)
        for arm in ("union_v2", "rank_delta_transfer")
    }
    reasons = (
        "more_consistent_redundant_constraints",
        "consistent_tie_larger_component",
        "union_conservative_fallback",
    )
    reason_counts = {
        reason: sum(row["selection"]["reason"] == reason for row in rows) for reason in reasons
    }
    return selection_counts, reason_counts


def run_confirmation(args: argparse.Namespace) -> None:
    config, config_sha = load_confirmation_config(args.config)
    audit_path = _resolve_project_path(str(config["frozen_inputs"]["roster_audit"]["path"]))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest mismatch")
    if sha256_file(args.manifest) != config["frozen_inputs"]["manifest"]["sha256"]:
        raise ValueError("run manifest differs from preregistration")
    names = _validated_run_roster(config, audit_path, manifest)
    lookup = {str(record["filename"]): dict(record) for record in manifest["splits"]["train"]}
    records = tuple(lookup[name] for name in names)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "frozen-target-free-layouts.npz"
    metadata_path = output_dir / "frozen-target-free-layouts.json"
    report_path = output_dir / "report.json"
    if any(path.exists() for path in (prediction_path, metadata_path, report_path)):
        raise FileExistsError("refusing to overwrite a fresh64 selector run")

    boards = _prepare_boards(records, args.targets)
    device = torch.device("cpu")
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    twin = load_fullres_twin_checkpoint(TWIN_CHECKPOINT, device=device)
    union = load_raw_twin_union_checkpoint(
        UNION_CHECKPOINT,
        config_path=UNION_CONFIG,
        selection_path=UNION_SELECTION,
        device=device,
    )
    direct = load_direct_hard_edge_checkpoint(DIRECT_CHECKPOINT, device=device)
    if union.sha256 != FROZEN_UNION_CHECKPOINT_SHA256:
        raise ValueError("Union-v2 checkpoint identity changed")
    if direct.sha256 != FROZEN_DIRECT_HARD_EDGE_SHA256:
        raise ValueError("direct hard-edge checkpoint identity changed")

    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    synthetic_seed = int(config["selection"]["synthetic_seed"])
    with torch.inference_mode():
        for index, board in enumerate(boards):
            corruption_seed, permutation_seed = _case_seeds(
                synthetic_seed,
                board.filename,
            )
            dirty, unused_second, unused_reference = _two_view_case(
                board.tiles,
                first_seed=corruption_seed,
                second_seed=corruption_seed + 1,
                permutation_seed=permutation_seed,
            )
            del unused_second, unused_reference
            union_inference = infer_raw_twin_union_assignments(
                dirty,
                socket,
                twin,
                union,
                device=device,
            )
            direct_inference = infer_direct_hard_edge_priorities(
                dirty,
                socket,
                direct,
                device=device,
            )
            rank_priority = build_direct_rank_delta_union_priority(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                direct_source=direct_inference.source,
                direct_target=direct_inference.target,
                direct_axis=direct_inference.axis,
                direct_raw_scores=direct_inference.raw_scores,
                direct_learned_scores=direct_inference.learned_scores,
                grid=GRID,
            )
            baseline_build = build_translation_components(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                grid=GRID,
                edge_budget_per_axis=DECODER_EDGE_BUDGET,
            )
            treatment_build = build_translation_components(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                grid=GRID,
                edge_budget_per_axis=DECODER_EDGE_BUDGET,
                component_edge_priority=rank_priority.component_edge_priority,
            )
            decision = select_direct_rank_delta_component_arm(
                _component_evidence(
                    baseline_build.status_counts,
                    baseline_build.components,
                    tile_count=COUNT,
                ),
                _component_evidence(
                    treatment_build.status_counts,
                    treatment_build.components,
                    tile_count=COUNT,
                ),
            )
            baseline = _decode_layout(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                component_edge_priority=None,
            )
            treatment = _decode_layout(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                component_edge_priority=rank_priority.component_edge_priority,
            )
            selected = treatment if decision.treatment_selected else baseline
            prefix = f"case_{index:04d}"
            arrays[f"{prefix}__union_v2_layout"] = baseline
            arrays[f"{prefix}__rank_delta_transfer_layout"] = treatment
            arrays[f"{prefix}__component_selector_layout"] = selected
            arrays[f"{prefix}__selected_arm_index"] = np.asarray(
                1 if decision.treatment_selected else 0,
                dtype=np.int8,
            )
            edge_arrays = _edge_arrays(
                union_inference.learned_right_log_assignment,
                union_inference.learned_down_log_assignment,
                rank_priority.component_edge_priority,
            )
            for axis in (0, 1):
                source = edge_arrays[f"axis_{axis}_source"]
                target = edge_arrays[f"axis_{axis}_target"]
                baseline_priority = edge_arrays[f"axis_{axis}_baseline_priority"]
                treatment_priority = edge_arrays[f"axis_{axis}_treatment_priority"]
                selected_priority = (
                    treatment_priority if decision.treatment_selected else baseline_priority
                )
                arrays[f"{prefix}__axis_{axis}_source"] = source
                arrays[f"{prefix}__axis_{axis}_target"] = target
                arrays[f"{prefix}__axis_{axis}_union_v2_priority"] = baseline_priority
                arrays[f"{prefix}__axis_{axis}_rank_delta_transfer_priority"] = treatment_priority
                arrays[f"{prefix}__axis_{axis}_component_selector_priority"] = selected_priority
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": board.filename,
                    "dirty_sha256": _dirty_sha256(dirty),
                    "corruption_seed": corruption_seed,
                    "permutation_seed": permutation_seed,
                    "priority": rank_priority.report(),
                    "selection": decision.report(),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "freeze",
                        "done": index + 1,
                        "total": EXPECTED_SOURCES,
                        "selected_arm": decision.selected_arm,
                        "reason": decision.reason,
                    }
                ),
                flush=True,
            )

    np.savez_compressed(prediction_path, **arrays)
    selection_counts, reason_counts = _selection_counts(frozen_rows)
    _atomic_json(
        metadata_path,
        {
            "schema": "aiijc-direct-rank-delta-component-selector-fresh64-predictions-v1",
            "panel_role": "source-disjoint frozen confirmation",
            "contains_exact_references": False,
            "contains_dirty_or_clean_pixels": False,
            "contains_target_free_strict_layouts": True,
            "selector": config["selector"],
            "selection_counts": selection_counts,
            "reason_counts": reason_counts,
            "cases": frozen_rows,
        },
    )
    prediction_sha = sha256_file(prediction_path)
    metadata_sha = sha256_file(metadata_path)
    print(
        json.dumps(
            {
                "event": "selector_decisions_and_layouts_frozen_before_scoring",
                "predictions_sha256": prediction_sha,
                "metadata_sha256": metadata_sha,
            }
        ),
        flush=True,
    )

    scored_rows: list[dict[str, Any]] = []
    strict_layouts = 0
    with np.load(prediction_path) as archive:
        for index, board in enumerate(boards):
            corruption_seed, permutation_seed = _case_seeds(
                synthetic_seed,
                board.filename,
            )
            _, _, reference = _two_view_case(
                board.tiles,
                first_seed=corruption_seed,
                second_seed=corruption_seed + 1,
                permutation_seed=permutation_seed,
            )
            reference = _strict_layout(reference)
            prefix = f"case_{index:04d}"
            row: dict[str, Any] = {"source_filename": board.filename}
            for arm in ARM_NAMES:
                layout = _strict_layout(archive[f"{prefix}__{arm}_layout"])
                strict_layouts += 1
                row[arm] = {
                    "exact_tiles": int(np.count_nonzero(layout == reference)),
                    "adjacency": float(_adjacency_fraction(layout, reference)),
                    "top144_correct": _fixed_top144_correct(
                        archive,
                        prefix,
                        reference,
                        arm=arm,
                    ),
                }
            row["selected_arm"] = (
                "rank_delta_transfer"
                if int(archive[f"{prefix}__selected_arm_index"]) == 1
                else "union_v2"
            )
            scored_rows.append(row)

    arms = {
        arm: {metric: _mean(scored_rows, arm, metric) for metric in METRIC_NAMES}
        for arm in ARM_NAMES
    }
    versus_union = _comparison_metrics(
        scored_rows,
        treatment="component_selector",
        baseline="union_v2",
        seed=BOOTSTRAP_SEED,
    )
    versus_rank = _comparison_metrics(
        scored_rows,
        treatment="component_selector",
        baseline="rank_delta_transfer",
        seed=BOOTSTRAP_SEED + 10,
    )
    metrics = {
        "arms": arms,
        "component_selector_vs_union_v2": versus_union,
        "component_selector_vs_rank_delta_transfer": versus_rank,
        "strict_layouts": strict_layouts,
        "selection_counts": selection_counts,
        "reason_counts": reason_counts,
    }
    gate = evaluate_confirmation_gate(metrics, strict_layouts=strict_layouts)
    report = {
        "schema": "aiijc-direct-rank-delta-component-selector-fresh64-report-v1",
        "status": gate["status"],
        "panel_role": "source-disjoint frozen confirmation",
        "config": _report_path(args.config),
        "config_sha256": config_sha,
        "frozen_inputs": config["frozen_inputs"],
        "selection": config["selection"],
        "predictions": {
            "path": _report_path(prediction_path),
            "sha256": prediction_sha,
            "metadata_path": _report_path(metadata_path),
            "metadata_sha256": metadata_sha,
            "selector_decisions_and_layouts_frozen_before_reference_scoring": True,
            "contains_exact_references": False,
        },
        "metrics": metrics,
        "gate": gate,
        "rows": scored_rows,
        "runtime_seconds": perf_counter() - started,
        "organizer_holdout_or_test_opened": False,
        "original_upright_tile_permutations_only": True,
        "whole_layout_arm_selection_only": True,
        "weight_budget_threshold_seed_or_arm_sweep": False,
    }
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "metrics": metrics,
                "gate": gate,
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.mode == "selection":
        freeze_selection(args)
    else:
        run_confirmation(args)


if __name__ == "__main__":
    main()
