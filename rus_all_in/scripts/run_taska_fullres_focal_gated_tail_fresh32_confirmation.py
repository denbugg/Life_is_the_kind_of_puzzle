#!/usr/bin/env python3
"""Run one fixed, truly disjoint fullres+focal-tail confirmation panel.

The three arms share one same-pass target-350 TASKA match, its original dense
cost matrices, current candidate harvest, four solver layouts, and recovered
focal logits.  ``control`` is the current four-arm selector plus all-edge
tail96.  ``fullres`` adds the already-fixed matcher-only full-resolution
candidate supply and fifth arm, then uses all-edge tail96.  ``combo`` reuses
the exact same five-arm pre-tail winner and applies the already-fixed focal
logit-zero protected tail96 to the winner-aligned candidate set.

The registered source roster is selected by deterministic SHA ranking only
after the separate tail192 reservation exists.  All target-free matrices,
edges, logits, layouts, and artifact hashes are frozen before exact synthetic
references are reconstructed.  No threshold, orientation, budget, arm, or
panel sweep is available in this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, solve_raw_tail_global
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_edge_calibrator import (
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_verifier import score_focal_edges
from aiijc_puzzle.taska_fullres_focal_gated_tail import (
    polish_fullres_winner_with_focal_gate,
)
from aiijc_puzzle.taska_fullres_union_voter import (
    FULLRES_ARM_NAME,
    FULLRES_DENOISER_SHA256,
    NEW_EDGE_FOCAL_LOGIT_MINIMUM,
    RESTORED_SCORER_COUNT,
    RESTORED_SUPPORT_MINIMUM,
    accept_focal_proposals,
    compose_fullres_union_focal_arm,
    load_fullres_denoiser,
    restore_fixed_matcher_view,
    restored_mutual_scorer_sets,
    strict_layout,
    supported_absent_edges,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import (
    ARM_NAMES,
    EXPECTED_ARTIFACT_SHA256,
    FOCAL_MODE,
    GRID_SIZE,
    MATCHER_CONFIG,
    PAIR_DENOMINATOR,
    SOLVER_CONFIG,
    TAIL_MAX_SWAPS,
    TAIL_MINIMUM_GAIN,
    TaskaPairArtifactPaths,
    load_taska_pair_pipeline_resources,
)
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail
from aiijc_puzzle.taska_seam_matcher import match_taska_tiles

try:
    from scripts import run_taska_protected_tail_fresh32_confirmation as synthetic
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_protected_tail_fresh32_confirmation as synthetic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/taska_fullres_focal_gated_tail_fresh32_confirmation_v1.json"
)
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/taska-fullres-focal-gated-tail/fresh32-confirmation-v1"
)
FULLRES_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
    "fullres_boundary_denoiser.pt"
)

CONFIG_SCHEMA = (
    "aiijc-taska-fullres-focal-gated-tail-fresh32-confirmation-config-v1"
)
FROZEN_SCHEMA = (
    "aiijc-taska-fullres-focal-gated-tail-fresh32-confirmation-target-free-v1"
)
FREEZE_SCHEMA = (
    "aiijc-taska-fullres-focal-gated-tail-fresh32-confirmation-freeze-v1"
)
REPORT_SCHEMA = (
    "aiijc-taska-fullres-focal-gated-tail-fresh32-confirmation-report-v1"
)

SOURCE_MINIMUM = 6_400
SOURCE_MAXIMUM = 6_699
SOURCE_COUNT = 16
DRAWS = (0, 1)
CASE_COUNT = SOURCE_COUNT * len(DRAWS)
SELECTION_NAMESPACE = (
    "aiijc-taska-fullres-focal-gated-tail-e2e-fresh32-confirmation-v1-"
    "source16xdraw2"
)
SELECTION_SEED = 2_026_083_194
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 94_083_126
PAIR_GATE_MEAN = 2.0
PAIR_GATE_CI95_LOWER = 0.0
RAW_SOLVER_SHA256 = (
    "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
)
ARMS = (
    "control_four_arm_all_edge_tail96",
    "fullres_five_arm_all_edge_tail96",
    "combo_five_arm_focal_gated_tail96",
)
COMPARISONS = {
    "fullres_minus_control": (ARMS[1], ARMS[0]),
    "combo_minus_fullres": (ARMS[2], ARMS[1]),
    "combo_minus_control": (ARMS[2], ARMS[0]),
}
IMAGE_NAME_PATTERN = re.compile(r"^img_\d{6}\.png$")

EXPECTED_ARTIFACT_PATHS = {
    "manifest": "data/interim/validation_manifest.json",
    "tail192_reservation": (
        "configs/taska_focal_gated_tail192_fresh16_capacity_v1.json"
    ),
    "fullres_relation_fusion_decoder_preregistration": (
        "configs/fullres_relation_fusion_decoder_d2_preregistered_v1.json"
    ),
    "fullres_relation_fusion_preregistration": (
        "configs/fullres_relation_fusion_preregistered_v1.json"
    ),
    "fullres_twin_preregistration": (
        "configs/fullres_twin_side_matcher_preregistered_v1.json"
    ),
    "fullres_boundary_report": (
        "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
        "report.json"
    ),
    "fullres_boundary_smoke_report": (
        "outputs/fullres-boundary-denoiser/smoke-train1-s1-eval1-auto/report.json"
    ),
    "fullres_fragment_smoke_report": (
        "outputs/fullres-fusion-fragment-solver/smoke1-mps-v2/report.json"
    ),
    "fullres_union_opened40_report": (
        "outputs/fullres-fusion-union-priority/opened40-mps-v1/report.json"
    ),
    "fullres_union_opened8_report": (
        "outputs/fullres-fusion-union-priority/opened8-mps-v1/report.json"
    ),
    "fullres_union_smoke_report": (
        "outputs/fullres-fusion-union-priority/smoke1-mps-v1/report.json"
    ),
    "fullres_conversion_audit_v2_report": (
        "outputs/fullres-relation-fusion/"
        "conversion-audit-opened-source40-v2/report.json"
    ),
    "fullres_conversion_audit_report": (
        "outputs/fullres-relation-fusion/"
        "conversion-audit-opened-source40/report.json"
    ),
    "fullres_decoder_report": (
        "outputs/fullres-relation-fusion/decoder-d2-source40-draw1/report.json"
    ),
    "fullres_relation_report": (
        "outputs/fullres-relation-fusion/v1-fit32-s300-eval16/report.json"
    ),
    "fullres_twin_report": (
        "outputs/fullres-twin-side-matcher/v1-fit256-s400-eval24/report.json"
    ),
    "fullres_twin_selection_commitment": (
        "outputs/fullres-twin-side-matcher/v1-fit256-s400-eval24/"
        "selection-commitment.json"
    ),
    "fullres_denoiser_checkpoint": (
        "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
        "fullres_boundary_denoiser.pt"
    ),
}
HISTORICAL_FULLRES_ARTIFACTS = tuple(
    name
    for name in EXPECTED_ARTIFACT_PATHS
    if name.startswith("fullres_") and name != "fullres_denoiser_checkpoint"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--inference-batch", type=int, default=576)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": _project_path(resolved), "sha256": sha256_file(resolved)}


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _names_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _cases_digest(names: Sequence[str]) -> str:
    serialized = "\n".join(f"{name}\0{draw}" for name in names for draw in DRAWS)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _full_universe() -> tuple[str, ...]:
    return tuple(
        f"img_{index:06d}.png"
        for index in range(SOURCE_MINIMUM, SOURCE_MAXIMUM + 1)
    )


def _collect_image_names(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, str):
        if IMAGE_NAME_PATTERN.fullmatch(value):
            result.add(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            result.update(_collect_image_names(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_collect_image_names(child))
    return result


def _load_config(path: Path) -> Mapping[str, Any]:
    resolved = path.resolve()
    sidecar = Path(f"{resolved}.sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise ValueError("preregistration JSON and SHA sidecar are both required")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if len(tokens) not in {1, 2} or tokens[0] != sha256_file(resolved):
        raise ValueError("preregistration SHA sidecar does not match")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("preregistration schema changed")
    return config


def _require_record(
    config: Mapping[str, Any], name: str, expected_path: str
) -> Path:
    record = config.get("artifacts", {}).get(name)
    if not isinstance(record, Mapping):
        raise ValueError(f"missing preregistered artifact: {name}")
    if record.get("path") != expected_path or not isinstance(record.get("sha256"), str):
        raise ValueError(f"malformed preregistered artifact: {name}")
    path = (PROJECT_ROOT / expected_path).resolve()
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"preregistered artifact changed: {name}")
    return path


def _json_payload(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"signed roster artifact is not valid JSON: {path}") from error


def _tail192_signed_sources(config: Mapping[str, Any]) -> set[str]:
    path = _require_record(
        config,
        "tail192_reservation",
        EXPECTED_ARTIFACT_PATHS["tail192_reservation"],
    )
    tail = _json_payload(path)
    if tail.get("schema") != "aiijc-taska-focal-gated-tail192-fresh16-capacity-config-v1":
        raise ValueError("tail192 reservation schema changed")
    roster = tail.get("panel", {}).get("source_filenames")
    if not isinstance(roster, list) or len(roster) != 16:
        raise ValueError("tail192 reservation roster changed")
    if _names_digest(roster) != (
        "46818ecfcd4dc5b53ac35548a1fe250a9242b455555a19b2c420738a252adac4"
    ):
        raise ValueError("tail192 reservation order digest changed")
    result = _collect_image_names(tail)
    dependencies = tail.get("artifacts")
    if not isinstance(dependencies, Mapping):
        raise ValueError("tail192 signed dependency registry is absent")
    for name, record in dependencies.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"malformed tail192 dependency: {name}")
        raw_path, digest = record.get("path"), record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            raise ValueError(f"malformed tail192 dependency: {name}")
        dependency = (PROJECT_ROOT / raw_path).resolve()
        if not dependency.is_file() or sha256_file(dependency) != digest:
            raise ValueError(f"tail192 dependency changed: {name}")
        if name != "manifest":
            result.update(_collect_image_names(_json_payload(dependency)))
    return result


def _historical_fullres_sources(config: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for name in HISTORICAL_FULLRES_ARTIFACTS:
        path = _require_record(config, name, EXPECTED_ARTIFACT_PATHS[name])
        result.update(_collect_image_names(_json_payload(path)))
    return result


def _deterministic_roster(eligible: Sequence[str]) -> tuple[str, ...]:
    prefix = f"{SELECTION_NAMESPACE}\0{SELECTION_SEED}\0".encode()
    return tuple(
        sorted(
            eligible,
            key=lambda name: (
                hashlib.sha256(prefix + name.encode()).digest(),
                name,
            ),
        )[:SOURCE_COUNT]
    )


def _validate_preregistration(config: Mapping[str, Any]) -> tuple[str, ...]:
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        EXPECTED_ARTIFACT_PATHS
    ):
        raise ValueError("preregistered artifact registry changed")
    for name, expected_path in EXPECTED_ARTIFACT_PATHS.items():
        _require_record(config, name, expected_path)

    tail_sources = _tail192_signed_sources(config)
    historical_fullres = _historical_fullres_sources(config)
    excluded = tail_sources | historical_fullres
    universe = _full_universe()
    eligible = tuple(name for name in universe if name not in excluded)
    roster = _deterministic_roster(eligible)
    panel = config.get("panel", {})
    fixed = {
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_seed": SELECTION_SEED,
        "universe_minimum": f"img_{SOURCE_MINIMUM:06d}.png",
        "universe_maximum": f"img_{SOURCE_MAXIMUM:06d}.png",
        "universe_count": len(universe),
        "universe_digest": _names_digest(universe),
        "exclusion_union_count": len(excluded),
        "exclusion_union_digest": _names_digest(sorted(excluded)),
        "excluded_in_universe_count": len(set(universe) & excluded),
        "eligible_count": len(eligible),
        "eligible_digest": _names_digest(eligible),
        "source_filenames": list(roster),
        "source_count": SOURCE_COUNT,
        "draws": list(DRAWS),
        "case_count": CASE_COUNT,
        "source_order_digest": _names_digest(roster),
        "cases_digest": _cases_digest(roster),
    }
    for key, value in fixed.items():
        if panel.get(key) != value:
            raise ValueError(f"preregistered panel field changed: {key}")
    exclusion_rosters = panel.get("exclusion_rosters")
    expected_exclusion_rosters = {
        "tail192_signed_dependencies_and_reserved": {
            "count": len(tail_sources),
            "digest": _names_digest(sorted(tail_sources)),
        },
        "historical_fullres_explicit": {
            "count": len(historical_fullres),
            "digest": _names_digest(sorted(historical_fullres)),
        },
    }
    if exclusion_rosters != expected_exclusion_rosters:
        raise ValueError("preregistered exclusion roster manifest changed")
    if set(roster) & excluded:
        raise RuntimeError("registered panel overlaps a signed exclusion roster")

    candidate = config.get("candidate", {})
    fixed_candidate = {
        "matcher_vote_target": 350,
        "current_scorer_count": 12,
        "portfolio_arms": list(ARM_NAMES),
        "restored_scorer_count": RESTORED_SCORER_COUNT,
        "new_edge_support_minimum": RESTORED_SUPPORT_MINIMUM,
        "new_edge_focal_logit_minimum": NEW_EDGE_FOCAL_LOGIT_MINIMUM,
        "focal_mode": FOCAL_MODE,
        "focal_protection_logit_threshold": 0.0,
        "tail_max_swaps": TAIL_MAX_SWAPS,
        "tail_minimum_gain": TAIL_MINIMUM_GAIN,
        "threshold_budget_orientation_arm_or_panel_sweep": False,
        "restored_pixels_matcher_only": True,
        "raw_dense_cost_matrices_unchanged": True,
    }
    for key, value in fixed_candidate.items():
        if candidate.get(key) != value:
            raise ValueError(f"fixed candidate field changed: {key}")
    evaluation = config.get("evaluation", {})
    fixed_evaluation = {
        "primary_metric": (
            "combo_minus_control_satisfied_adjacent_pairs_per_board"
        ),
        "pair_denominator": PAIR_DENOMINATOR,
        "required_ablations": [
            "fullres_minus_control",
            "combo_minus_fullres",
        ],
        "secondary_metrics": ["adjacency_recall", "exact_tiles_per_board"],
        "bootstrap_unit": "source_with_two_draws",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirmation_gate": {
            "combo_minus_control_pair_delta_mean_at_least": PAIR_GATE_MEAN,
            "combo_minus_control_pair_delta_ci95_lower_at_least": (
                PAIR_GATE_CI95_LOWER
            ),
        },
    }
    if evaluation != fixed_evaluation:
        raise ValueError("fixed evaluation protocol changed")
    return roster


def _load_manifest(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    path = _require_record(config, "manifest", EXPECTED_ARTIFACT_PATHS["manifest"])
    manifest = _json_payload(path)
    if compute_protocol_digest(manifest) != manifest.get("protocol_digest"):
        raise ValueError("organizer-train manifest protocol digest is invalid")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("organizer-train manifest split mapping is absent")
    rows = [row for values in splits.values() for row in values]
    if len(rows) != 7_000 or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("organizer-train manifest must contain exactly 7000 rows")
    lookup = {str(row["filename"]): row for row in rows}
    if len(lookup) != 7_000:
        raise ValueError("organizer-train manifest contains duplicate filenames")
    return lookup


def _strict_layout(value: Any) -> np.ndarray:
    return strict_layout(value, grid=GRID_SIZE)


def _edge_evidence(matched: Any) -> tuple[np.ndarray, np.ndarray]:
    by_edge = {record.edge: record for record in matched.vote_records}
    if len(by_edge) != len(matched.vote_records) or set(by_edge) != set(
        matched.candidate_edges
    ):
        raise ValueError("matcher vote records are not aligned to candidate edges")
    margins = np.asarray(
        [by_edge[edge].minimum_margin for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    votes = np.asarray(
        [by_edge[edge].vote_count for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    return margins, votes


def _edge_arrays(
    prefix: str, name: str, edges: Sequence[RawTailEdge]
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__{name}__edge_source": np.asarray(
            [edge.source for edge in edges], dtype=np.int16
        ),
        f"{prefix}__{name}__edge_target": np.asarray(
            [edge.target for edge in edges], dtype=np.int16
        ),
        f"{prefix}__{name}__edge_axis": np.asarray(
            [0 if edge.axis == "right" else 1 for edge in edges], dtype=np.uint8
        ),
    }


def _edges_from_archive(
    archive: Any, prefix: str, name: str
) -> tuple[RawTailEdge, ...]:
    source = np.asarray(archive[f"{prefix}__{name}__edge_source"], dtype=np.int64)
    target = np.asarray(archive[f"{prefix}__{name}__edge_target"], dtype=np.int64)
    axis = np.asarray(archive[f"{prefix}__{name}__edge_axis"], dtype=np.uint8)
    if not (source.ndim == target.ndim == axis.ndim == 1):
        raise ValueError("frozen edge arrays must be one-dimensional")
    if not (len(source) == len(target) == len(axis)) or not np.isin(axis, (0, 1)).all():
        raise ValueError("frozen edge arrays are malformed")
    edges = tuple(
        RawTailEdge(int(left), int(right), "right" if int(direction) == 0 else "down")
        for left, right, direction in zip(source, target, axis, strict=True)
    )
    if len(set(edges)) != len(edges):
        raise ValueError("frozen edge list contains duplicates")
    return edges


def _truth_edges(reference: Any) -> frozenset[RawTailEdge]:
    layout = _strict_layout(reference).reshape(GRID_SIZE, GRID_SIZE)
    result: set[RawTailEdge] = set()
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE - 1):
            result.add(
                RawTailEdge(
                    int(layout[row, column]),
                    int(layout[row, column + 1]),
                    "right",
                )
            )
    for row in range(GRID_SIZE - 1):
        for column in range(GRID_SIZE):
            result.add(
                RawTailEdge(
                    int(layout[row, column]),
                    int(layout[row + 1, column]),
                    "down",
                )
            )
    if len(result) != PAIR_DENOMINATOR:
        raise RuntimeError("truth adjacency denominator changed")
    return frozenset(result)


def _layout_metrics(layout: np.ndarray, exact: np.ndarray) -> dict[str, Any]:
    metrics = evaluate_layout(layout, exact, reference_is_exact=True)
    if metrics.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("pair denominator changed")
    return {
        "satisfied_adjacent_pairs": int(metrics.adjacency_correct),
        "adjacency_recall": float(metrics.adjacency),
        "exact_tiles": int(metrics.correct_tile_count),
        "strict_permutation": True,
    }


def _target_free_case(
    dirty_tiles: np.ndarray,
    *,
    resources: Any,
    denoiser: Any,
    inference_batch: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    matched = match_taska_tiles(
        dirty_tiles,
        resources.matchers,
        config=MATCHER_CONFIG,
        device=resources.device,
        require_verified=True,
    )
    right = np.ascontiguousarray(matched.cost_right, dtype=np.float32)
    down = np.ascontiguousarray(matched.cost_down, dtype=np.float32)
    current = tuple(matched.candidate_edges)
    current_focal = score_focal_edges(
        resources.focal_verifier,
        dirty_tiles,
        right,
        down,
        current,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.device,
    )
    current_logits = np.asarray(current_focal.logits, dtype=np.float32)
    margins, votes = _edge_evidence(matched)
    features = extract_taska_edge_features(
        right,
        down,
        matched.right_log,
        matched.down_log,
        current,
        margins,
        votes,
        grid=GRID_SIZE,
    ).values
    priorities = {
        "logistic": resources.logistic_calibrator.predict_priorities(features),
        "focal_top5": current_logits,
        "nonlinear": resources.nonlinear_calibrator.predict_priorities(features),
    }
    raw = solve_raw_tail_global(
        right,
        down,
        current,
        border_unary=None,
        grid=GRID_SIZE,
        config=SOLVER_CONFIG,
    )
    solved = {
        "raw": raw,
        **{
            name: solve_prioritized_raw_tail_global(
                right,
                down,
                current,
                priorities[name],
                border_unary=None,
                grid=GRID_SIZE,
                config=SOLVER_CONFIG,
            )
            for name in ("logistic", "focal_top5", "nonlinear")
        },
    }
    if tuple(solved) != ARM_NAMES:
        raise RuntimeError("four-arm solver order changed")
    four = {name: _strict_layout(result.layout) for name, result in solved.items()}
    selected_four = select_lowest_taska_seam_cost_layout(
        four,
        right,
        down,
        grid=GRID_SIZE,
    )
    four_pre_tail = _strict_layout(selected_four.layout)
    control = polish_unprotected_taska_tail(
        four_pre_tail,
        right,
        down,
        current,
        grid=GRID_SIZE,
        max_swaps=TAIL_MAX_SWAPS,
        minimum_gain=TAIL_MINIMUM_GAIN,
    )
    control_layout = _strict_layout(control.layout)

    restored = restore_fixed_matcher_view(
        denoiser,
        dirty_tiles,
        device=resources.device,
        batch_size=inference_batch,
    )
    scorer_sets = restored_mutual_scorer_sets(
        restored,
        resources.matchers,
        device=resources.device,
    )
    proposed, support = supported_absent_edges(current, scorer_sets)
    proposed_focal = score_focal_edges(
        resources.focal_verifier,
        dirty_tiles,
        right,
        down,
        proposed,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.device,
    )
    proposed_logits = np.asarray(proposed_focal.logits, dtype=np.float32)
    accepted, accepted_logits = accept_focal_proposals(proposed, proposed_logits)
    accepted_logits = np.asarray(accepted_logits, dtype=np.float32)
    composition = compose_fullres_union_focal_arm(
        cost_right=right,
        cost_down=down,
        current_edges=current,
        current_focal_logits=current_logits,
        accepted_new_edges=accepted,
        accepted_new_logits=accepted_logits,
        four_layouts=four,
        grid=GRID_SIZE,
    )
    fullres_layout = _strict_layout(composition.layout)
    winner_is_fullres = composition.choice == FULLRES_ARM_NAME
    five_pre_tail = (
        _strict_layout(composition.fullres_layout)
        if winner_is_fullres
        else four[composition.choice]
    )
    combo = polish_fullres_winner_with_focal_gate(
        five_pre_tail,
        right,
        down,
        current,
        current_logits,
        accepted,
        accepted_logits,
        winner_is_fullres=winner_is_fullres,
        grid=GRID_SIZE,
    )
    combo_layout = _strict_layout(combo.layout)
    if not winner_is_fullres and not np.array_equal(fullres_layout, control_layout):
        raise RuntimeError("old-arm five-way winner did not replay the same-pass control")
    if set(current) & set(accepted):
        raise RuntimeError("accepted fullres edges are not absent from current harvest")
    if len(scorer_sets) != RESTORED_SCORER_COUNT:
        raise RuntimeError("restored scorer roster changed")
    arrays = {
        "cost_right": right,
        "cost_down": down,
        "current_focal_logits": current_logits,
        "proposed_support": np.asarray(support, dtype=np.uint8),
        "proposed_focal_logits": proposed_logits,
        "accepted_focal_logits": accepted_logits,
        "four_pre_tail_layout": four_pre_tail,
        "five_pre_tail_layout": five_pre_tail,
        "fullres_union_focal_pre_tail_layout": _strict_layout(
            composition.fullres_layout
        ),
        f"{ARMS[0]}_layout": control_layout,
        f"{ARMS[1]}_layout": fullres_layout,
        f"{ARMS[2]}_layout": combo_layout,
        **{f"{name}_layout": layout for name, layout in four.items()},
    }
    diagnostics = {
        "current_candidate_count": len(current),
        "current_vote_threshold": matched.chosen_vote_threshold,
        "four_arm_choice": selected_four.choice,
        "four_arm_total_costs": dict(selected_four.total_costs),
        "five_arm_choice": composition.choice,
        "five_arm_total_costs": dict(composition.total_costs),
        "winner_is_fullres": winner_is_fullres,
        "restored_scorer_edge_counts": [len(edges) for edges in scorer_sets],
        "restored_supported_absent_count": len(proposed),
        "focal_accepted_new_count": len(accepted),
        "same_pass_raw_matrices_and_current_harvest_shared_by_all_arms": True,
        "solver_diagnostics": {
            name: asdict(result.diagnostics) for name, result in solved.items()
        },
        "control_tail": asdict(control.diagnostics),
        "fullres_composition": dict(composition.diagnostics),
        "combo": asdict(combo.diagnostics),
    }
    edge_sets = {
        "current": current,
        "proposed": proposed,
        "accepted": accepted,
        **{
            f"restored_scorer_{index}": tuple(
                sorted(
                    edges,
                    key=lambda edge: (
                        0 if edge.axis == "right" else 1,
                        edge.source,
                        edge.target,
                    ),
                )
            )
            for index, edges in enumerate(scorer_sets)
        },
    }
    for name, edges in edge_sets.items():
        arrays.update(
            {
                key.split("__", 1)[1]: value
                for key, value in _edge_arrays("case", name, edges).items()
            }
        )
    return arrays, diagnostics


def _freeze_target_free(
    *,
    config_path: Path,
    specs: Sequence[tuple[Mapping[str, Any], str, int]],
    targets: Path,
    output_dir: Path,
    device: torch.device,
    inference_batch: int,
) -> tuple[Path, Path, Path, dict[str, Any], float]:
    output_dir.mkdir(parents=True, exist_ok=False)
    archive_path = output_dir / "frozen-target-free-eval.npz"
    metadata_path = output_dir / "frozen-target-free-eval.json"
    freeze_path = output_dir / "pre-score-freeze.json"
    resources = load_taska_pair_pipeline_resources(device=device)
    denoiser = load_fullres_denoiser(FULLRES_CHECKPOINT, device=resources.device)
    cache = synthetic.CleanTileCache(targets.resolve(), maximum_boards=2)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    started = perf_counter()

    for index, (record, source, draw) in enumerate(specs):
        prefix = f"case_{index:03d}"
        dirty = synthetic._dirty_case(cache, record, source, draw)
        dirty_sha = synthetic._dirty_sha256(dirty.dirty_tiles)
        case_arrays, diagnostics = _target_free_case(
            dirty.dirty_tiles,
            resources=resources,
            denoiser=denoiser,
            inference_batch=inference_batch,
        )
        arrays.update(
            {f"{prefix}__{name}": value for name, value in case_arrays.items()}
        )
        row = {
            "prefix": prefix,
            "case_id": dirty.case_id,
            "source_filename": source,
            "draw_index": draw,
            "dirty_sha256": dirty_sha,
            **diagnostics,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "fullres_focal_e2e_target_free_case_ready",
                    "case": index + 1,
                    "case_count": len(specs),
                    "source_filename": source,
                    "draw_index": draw,
                    "current_candidates": diagnostics["current_candidate_count"],
                    "supported_absent": diagnostics[
                        "restored_supported_absent_count"
                    ],
                    "accepted_new": diagnostics["focal_accepted_new_count"],
                    "five_arm_choice": diagnostics["five_arm_choice"],
                }
            ),
            flush=True,
        )

    _write_npz_exclusive(archive_path, arrays)
    target_free_summary = {
        "case_count": len(rows),
        "mean_current_candidates": float(
            np.mean([row["current_candidate_count"] for row in rows])
        ),
        "mean_supported_absent": float(
            np.mean([row["restored_supported_absent_count"] for row in rows])
        ),
        "mean_focal_accepted_new": float(
            np.mean([row["focal_accepted_new_count"] for row in rows])
        ),
        "total_supported_absent": int(
            sum(row["restored_supported_absent_count"] for row in rows)
        ),
        "total_focal_accepted_new": int(
            sum(row["focal_accepted_new_count"] for row in rows)
        ),
        "five_arm_choice_counts": dict(
            Counter(row["five_arm_choice"] for row in rows)
        ),
    }
    _write_json_exclusive(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "same_pass_raw_matrices_and_current_harvest_shared_by_all_arms": True,
            "all_dense_matrices_edges_logits_and_layouts_frozen": True,
            "all_layouts_strict_original_upright_tile_permutations": True,
            "restored_pixels_matcher_only": True,
            "matcher_config": asdict(MATCHER_CONFIG),
            "solver_config": asdict(SOLVER_CONFIG),
            "portfolio_arms": list(ARM_NAMES),
            "restored_scorer_count": RESTORED_SCORER_COUNT,
            "restored_support_minimum": RESTORED_SUPPORT_MINIMUM,
            "new_edge_focal_logit_minimum": NEW_EDGE_FOCAL_LOGIT_MINIMUM,
            "focal_mode": FOCAL_MODE,
            "tail_max_swaps": TAIL_MAX_SWAPS,
            "tail_minimum_gain": TAIL_MINIMUM_GAIN,
            "threshold_budget_orientation_arm_or_panel_sweep": False,
            "target_free_summary": target_free_summary,
            "rows": rows,
        },
    )

    pair_paths = TaskaPairArtifactPaths()
    runtime_sources = {
        "confirmation_runner": Path(__file__).resolve(),
        "union_voter_module": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_fullres_union_voter.py"
        ),
        "combo_module": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_fullres_focal_gated_tail.py"
        ),
        "focal_tail_module": (
            PROJECT_ROOT
            / "src/aiijc_puzzle/taska_focal_gated_protected_tail.py"
        ),
        "pair_pipeline": PROJECT_ROOT / "src/aiijc_puzzle/taska_pair_pipeline.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        "matcher": PROJECT_ROOT / "src/aiijc_puzzle/taska_seam_matcher.py",
        "focal_verifier": PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_verifier.py",
        "edge_calibrator": PROJECT_ROOT / "src/aiijc_puzzle/taska_edge_calibrator.py",
        "layout_portfolio": PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py",
        "protected_tail": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_protected_tail_polish.py"
        ),
        "layout_evaluation": PROJECT_ROOT / "src/aiijc_puzzle/layout_evaluation.py",
        "synthetic_generator": (
            PROJECT_ROOT / "src/aiijc_puzzle/synthetic_socket_evaluation.py"
        ),
    }
    artifacts = {
        "preregistration": _record(config_path),
        "preregistration_sidecar": _record(Path(f"{config_path}.sha256")),
        "frozen_archive": _record(archive_path),
        "frozen_metadata": _record(metadata_path),
        **{name: _record(path) for name, path in runtime_sources.items()},
        "fullres_denoiser_checkpoint": _record(FULLRES_CHECKPOINT),
        "matcher_v3_checkpoint": _record(pair_paths.matcher_v3),
        "matcher_local_checkpoint": _record(pair_paths.matcher_local),
        "logistic_calibrator": _record(pair_paths.logistic_calibrator),
        "focal_checkpoint": _record(pair_paths.focal_verifier),
        "nonlinear_calibrator": _record(pair_paths.nonlinear_calibrator),
    }
    _write_json_exclusive(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "same_pass_raw_matrices_and_current_harvest_shared_by_all_arms": True,
            "all_candidate_layouts_edges_logits_and_dense_costs_frozen": True,
            "fullres_denoiser_sha256": FULLRES_DENOISER_SHA256,
            "verified_taska_artifact_sha256": dict(EXPECTED_ARTIFACT_SHA256),
            "device": str(resources.device),
            "mps_bitwise_reproducibility_claimed": False,
            "artifacts": artifacts,
        },
    )
    return (
        archive_path,
        metadata_path,
        freeze_path,
        target_free_summary,
        perf_counter() - started,
    )


def _validate_freeze(path: Path) -> Mapping[str, Any]:
    payload = _json_payload(path)
    if payload.get("schema") != FREEZE_SCHEMA:
        raise RuntimeError("pre-score freeze schema changed")
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("pre-score freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains labels")
    if payload.get("fullres_denoiser_sha256") != FULLRES_DENOISER_SHA256:
        raise RuntimeError("pre-score denoiser SHA changed")
    if payload.get("verified_taska_artifact_sha256") != dict(
        EXPECTED_ARTIFACT_SHA256
    ):
        raise RuntimeError("pre-score TASKA artifact manifest changed")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("pre-score artifact roster is absent")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"malformed frozen artifact: {name}")
        raw_path, digest = record.get("path"), record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            raise RuntimeError(f"malformed frozen artifact: {name}")
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise RuntimeError(f"pre-score artifact changed: {name}")
    return payload


def _cluster_ci(
    values: Sequence[float], sources: Sequence[str], *, seed: int
) -> dict[str, Any]:
    if len(values) != len(sources) or not values:
        raise ValueError("cluster bootstrap inputs must be aligned and non-empty")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("bootstrap values must be finite")
        grouped[source].append(float(value))
    if len(grouped) != SOURCE_COUNT or any(len(group) != 2 for group in grouped.values()):
        raise ValueError("confirmation bootstrap requires 16 sources with two draws each")
    source_means = np.asarray(
        [np.mean(grouped[name]) for name in sorted(grouped)], dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(
            0,
            len(source_means),
            size=(stop - start, len(source_means)),
        )
        distribution[start:stop] = source_means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "total_sum": float(np.sum(values)),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(source_means),
        "case_count": len(values),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
        "case_wins_ties_losses": {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        },
        "source_wins_ties_losses": {
            "wins": int(np.sum(source_means > 0)),
            "ties": int(np.sum(source_means == 0)),
            "losses": int(np.sum(source_means < 0)),
        },
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != CASE_COUNT:
        raise ValueError("confirmation summary requires exactly 32 cases")
    sources = [str(row["source_filename"]) for row in rows]
    metric_names = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    arm_means = {
        arm: {
            metric: float(np.mean([row[arm][metric] for row in rows]))
            for metric in metric_names
        }
        for arm in ARMS
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for comparison_index, (name, (candidate, control)) in enumerate(
        COMPARISONS.items()
    ):
        comparisons[name] = {
            metric: _cluster_ci(
                [
                    float(row[candidate][metric]) - float(row[control][metric])
                    for row in rows
                ],
                sources,
                seed=BOOTSTRAP_SEED + comparison_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(metric_names)
        }

    supply_fields = (
        "current_candidate_count",
        "current_true_edges",
        "proposed_absent_count",
        "proposed_true_edges",
        "accepted_new_count",
        "accepted_new_true_edges",
        "union_candidate_count",
        "union_true_edges",
    )
    totals = {
        field: int(sum(row["candidate_supply"][field] for row in rows))
        for field in supply_fields
    }
    candidate_supply = {
        "totals": totals,
        "means_per_board": {
            field: float(np.mean([row["candidate_supply"][field] for row in rows]))
            for field in supply_fields
        },
        "current_precision": (
            totals["current_true_edges"] / max(1, totals["current_candidate_count"])
        ),
        "proposed_absent_precision": (
            totals["proposed_true_edges"] / max(1, totals["proposed_absent_count"])
        ),
        "accepted_new_precision": (
            totals["accepted_new_true_edges"] / max(1, totals["accepted_new_count"])
        ),
        "current_candidate_recall": (
            totals["current_true_edges"] / (PAIR_DENOMINATOR * CASE_COUNT)
        ),
        "union_candidate_recall": (
            totals["union_true_edges"] / (PAIR_DENOMINATOR * CASE_COUNT)
        ),
        "mean_true_missing_edges_added_per_board": float(
            np.mean(
                [
                    row["candidate_supply"]["union_true_edges"]
                    - row["candidate_supply"]["current_true_edges"]
                    for row in rows
                ]
            )
        ),
    }
    primary = comparisons["combo_minus_control"]["satisfied_adjacent_pairs"]
    gate = {
        "required_pair_delta_mean": PAIR_GATE_MEAN,
        "required_pair_delta_ci95_lower": PAIR_GATE_CI95_LOWER,
        "observed_pair_delta_mean": primary["mean"],
        "observed_pair_delta_total_sum": primary["total_sum"],
        "observed_pair_delta_ci95_lower": primary["ci95_lower"],
        "passed": (
            primary["mean"] >= PAIR_GATE_MEAN
            and primary["ci95_lower"] >= PAIR_GATE_CI95_LOWER
        ),
    }
    return {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": arm_means,
        "comparisons": comparisons,
        "candidate_supply": candidate_supply,
        "confirmation_gate": gate,
        "four_arm_choice_counts": dict(
            Counter(row["four_arm_choice"] for row in rows)
        ),
        "five_arm_choice_counts": dict(
            Counter(row["five_arm_choice"] for row in rows)
        ),
    }


def _score_after_freeze(
    *,
    specs: Sequence[tuple[Mapping[str, Any], str, int]],
    targets: Path,
    archive_path: Path,
    metadata_path: Path,
    freeze_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_freeze(freeze_path)
    metadata = _json_payload(metadata_path)
    if metadata.get("schema") != FROZEN_SCHEMA:
        raise RuntimeError("frozen metadata schema changed")
    if metadata.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("frozen metadata contains labels")
    frozen_rows = metadata.get("rows")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != CASE_COUNT:
        raise RuntimeError("frozen candidate row roster changed")
    cache = synthetic.CleanTileCache(targets.resolve(), maximum_boards=2)
    scored: list[dict[str, Any]] = []
    with np.load(archive_path, allow_pickle=False) as archive:
        for (record, source, draw), frozen in zip(specs, frozen_rows, strict=True):
            dirty, reference = make_exact_synthetic_case(
                cache.load(record),
                source_filename=source,
                draw_index=draw,
                seed=synthetic.SYNTHETIC_SEED,
            )
            if (
                frozen.get("source_filename") != source
                or int(frozen.get("draw_index", -1)) != draw
                or frozen.get("case_id") != dirty.case_id
                or synthetic._dirty_sha256(dirty.tiles) != frozen.get("dirty_sha256")
                or reference.case_id != dirty.case_id
            ):
                raise RuntimeError("scoring recreated a different registered case")
            prefix = str(frozen["prefix"])
            exact = _strict_layout(reference.tile_at_position)
            truth = _truth_edges(exact)
            current = _edges_from_archive(archive, prefix, "current")
            proposed = _edges_from_archive(archive, prefix, "proposed")
            accepted = _edges_from_archive(archive, prefix, "accepted")
            if set(current) & set(proposed) or not set(accepted) <= set(proposed):
                raise RuntimeError("frozen new-edge set contract changed")
            union = set(current) | set(accepted)
            current_true = len(set(current) & truth)
            proposed_true = len(set(proposed) & truth)
            accepted_true = len(set(accepted) & truth)
            union_true = len(union & truth)
            row: dict[str, Any] = {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source,
                "draw_index": draw,
                "four_arm_choice": str(frozen["four_arm_choice"]),
                "five_arm_choice": str(frozen["five_arm_choice"]),
                "candidate_supply": {
                    "current_candidate_count": len(current),
                    "current_true_edges": current_true,
                    "proposed_absent_count": len(proposed),
                    "proposed_true_edges": proposed_true,
                    "accepted_new_count": len(accepted),
                    "accepted_new_true_edges": accepted_true,
                    "union_candidate_count": len(union),
                    "union_true_edges": union_true,
                    "accepted_edges_are_current_absent": True,
                },
            }
            for arm in ARMS:
                layout = _strict_layout(archive[f"{prefix}__{arm}_layout"])
                row[arm] = _layout_metrics(layout, exact)
            scored.append(row)
    return scored, _summarize(scored)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate, freeze one fixed panel, and score only after the freeze."""

    config_path = args.config.resolve()
    config = _load_config(config_path)
    roster = _validate_preregistration(config)
    manifest = _load_manifest(config)
    specs = [(manifest[name], name, draw) for name in roster for draw in DRAWS]
    if len(specs) != CASE_COUNT:
        raise RuntimeError("registered panel expansion changed")
    if args.inference_batch <= 0:
        raise ValueError("inference_batch must be positive")
    raw_solver = PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
    if sha256_file(raw_solver) != RAW_SOLVER_SHA256:
        raise ValueError("frozen raw solver SHA-256 changed")

    if args.validate_only:
        result = {
            "status": "validated",
            "config": _record(config_path),
            "source_count": SOURCE_COUNT,
            "case_count": CASE_COUNT,
            "source_filenames": list(roster),
            "competition_test_accessed": False,
        }
        print(json.dumps(result, indent=2), flush=True)
        return result

    device = synthetic._select_device(
        args.device,
        allow_nondeterministic_mps=bool(args.allow_nondeterministic_mps),
    )
    started = perf_counter()
    archive, metadata, freeze, target_free_summary, inference_seconds = (
        _freeze_target_free(
            config_path=config_path,
            specs=specs,
            targets=args.targets.resolve(),
            output_dir=args.output_dir.resolve(),
            device=device,
            inference_batch=args.inference_batch,
        )
    )
    print(
        json.dumps(
            {
                "event": "fullres_focal_e2e_all_layouts_frozen_before_scoring",
                "case_count": CASE_COUNT,
                "archive_sha256": sha256_file(archive),
                "metadata_sha256": sha256_file(metadata),
                "pre_score_freeze_sha256": sha256_file(freeze),
                "reference_reconstructed_yet": False,
            }
        ),
        flush=True,
    )
    rows, metrics = _score_after_freeze(
        specs=specs,
        targets=args.targets.resolve(),
        archive_path=archive,
        metadata_path=metadata,
        freeze_path=freeze,
    )
    gate_passed = bool(metrics["confirmation_gate"]["passed"])
    report = {
        "schema": REPORT_SCHEMA,
        "status": "confirmed" if gate_passed else "not-confirmed",
        "panel": {
            "source_count": SOURCE_COUNT,
            "draws": list(DRAWS),
            "case_count": CASE_COUNT,
            "source_filenames": list(roster),
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_seed": SELECTION_SEED,
            "truly_disjoint_under_preregistered_signed_exclusions": True,
            "one_panel_only": True,
        },
        "candidate": {
            "fixed_before_panel_scoring": True,
            "matcher_vote_target": MATCHER_CONFIG.vote_target,
            "current_matcher_views": ["raw", "median", "bilateral"],
            "current_scorer_count": 12,
            "portfolio_arms": list(ARM_NAMES),
            "portfolio_selector": "minimum original TASKA all-1104-bond seam cost",
            "restored_scorers": "v3/local x first two audited orientations",
            "restored_scorer_count": RESTORED_SCORER_COUNT,
            "new_edge_support_minimum": RESTORED_SUPPORT_MINIMUM,
            "new_edge_focal_logit_minimum": NEW_EDGE_FOCAL_LOGIT_MINIMUM,
            "focal_mode": FOCAL_MODE,
            "focal_protection_logit_threshold": 0.0,
            "tail_max_swaps": TAIL_MAX_SWAPS,
            "tail_minimum_gain": TAIL_MINIMUM_GAIN,
            "threshold_budget_orientation_arm_or_panel_sweep": False,
            "same_pass_raw_matrices_and_current_harvest_shared": True,
        },
        "target_free_summary": target_free_summary,
        "frozen_eval": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
            "all_three_strict_layouts_frozen_before_exact_reference_reconstruction": (
                True
            ),
            "contains_exact_references_or_labels": False,
        },
        "metrics": metrics,
        "rows": rows,
        "runtime_seconds": {
            "target_free_matcher_denoiser_focal_and_solver": inference_seconds,
            "total": perf_counter() - started,
        },
        "legality": {
            "organizer_train_sources_only": True,
            "dirty_tiles_only_for_candidate_inference": True,
            "targets_or_exact_references_used_during_candidate_inference": False,
            "restored_pixels_matcher_only": True,
            "raw_dense_cost_matrices_unchanged": True,
            "original_upright_20x20_tile_permutations_only": True,
            "rotated_warped_replaced_or_constant_tiles": False,
            "competition_test_accessed": False,
            "postprocessing_used": False,
        },
    }
    report_path = args.output_dir.resolve() / "report.json"
    _write_json_exclusive(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "metrics": metrics,
                "report": _record(report_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    run(parse_args())
