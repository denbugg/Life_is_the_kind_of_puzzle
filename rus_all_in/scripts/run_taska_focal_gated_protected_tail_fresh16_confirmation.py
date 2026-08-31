#!/usr/bin/env python3
"""Confirm the fixed focal-gated TASKA tail on one new 16-source panel.

The candidate is frozen: current target-350 TASKA matching, the unchanged
raw/logistic/focal-top5/nonlinear four-arm portfolio selected by original
all-bond seam cost, and tail96 protecting only harvested edges whose recovered
``train_exact_top5`` focal logit is non-negative.  The control is the identical
pre-tail layout and tail96 implementation while protecting every harvested
edge.  There is no threshold, budget, arm, or panel search in this runner.

Both strict original-tile layouts and their complete provenance are persisted
before the exact synthetic reference is reconstructed.  The registered panel
is disjoint from the current TASKA lineage named in the preregistration, but is
not claimed to be universally or model fresh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from aiijc_puzzle.raw_tail_global_solver import solve_raw_tail_global
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_edge_calibrator import (
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    FOCAL_PROTECTION_LOGIT_THRESHOLD,
    polish_taska_tail_with_focal_gate,
)
from aiijc_puzzle.taska_focal_verifier import score_focal_edges
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
    / "configs/taska_focal_gated_protected_tail_fresh16_confirmation_v1.json"
)
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/taska-focal-gated-protected-tail/fresh16-confirmation-v1"
)

CONFIG_SCHEMA = "aiijc-taska-focal-gated-protected-tail-fresh16-config-v1"
FROZEN_SCHEMA = "aiijc-taska-focal-gated-protected-tail-fresh16-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-focal-gated-protected-tail-fresh16-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-focal-gated-protected-tail-fresh16-report-v1"
SOURCE_MINIMUM = 6_400
SOURCE_MAXIMUM = 6_699
SOURCE_COUNT = 16
DRAWS = (0, 1)
CASE_COUNT = SOURCE_COUNT * len(DRAWS)
SELECTION_NAMESPACE = (
    "aiijc-taska-focal-gated-protected-tail-fresh16-confirmation-v1-source16xdraw2"
)
SELECTION_SEED = 2_026_083_109
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 1_603_189_331
ARMS = ("control_all_edges_tail96", "candidate_focal_gated_tail96")
PRIMARY_METRIC = "satisfied_adjacent_pairs"
PAIR_GATE_MEAN = 0.5
PAIR_GATE_CI95_LOWER = -0.25
RAW_SOLVER_SHA256 = (
    "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
)

PREREG_INPUTS = {
    "manifest": "data/interim/validation_manifest.json",
    "train256_roster": "outputs/taska-edge-calibrator/train256-v1/roster.json",
    "opened32_recipe": "configs/taska_seam_replay_opened32_v1.json",
    "held32_recipe": "configs/taska_seam_held300_diagnostic_v1.json",
    "prior_fresh32_recipe": "configs/taska_protected_tail_fresh32_confirmation_v1.json",
    "focal_current_recipe": "configs/taska_focal_current_finetune_v1.json",
    "focal_training224_metadata": (
        "outputs/taska-focal-feature-stacker/train224-v1/"
        "training-stacked-features.json"
    ),
    "fullres_boundary_preregistration": (
        "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
        "preregistered-local-gate.json"
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
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
        f"img_{index:06d}.png" for index in range(SOURCE_MINIMUM, SOURCE_MAXIMUM + 1)
    )


def _require_record(config: Mapping[str, Any], name: str, expected_path: str) -> Path:
    record = config.get("artifacts", {}).get(name)
    if not isinstance(record, Mapping):
        raise ValueError(f"missing preregistered artifact: {name}")
    if record.get("path") != expected_path or not isinstance(record.get("sha256"), str):
        raise ValueError(f"malformed preregistered artifact: {name}")
    path = (PROJECT_ROOT / expected_path).resolve()
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"preregistered artifact changed: {name}")
    return path


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


def _registered_rosters(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    paths = {
        name: _require_record(config, name, relative)
        for name, relative in PREREG_INPUTS.items()
    }
    train256 = tuple(
        json.loads(paths["train256_roster"].read_text(encoding="utf-8"))[
            "source_filenames"
        ]
    )
    opened32 = tuple(
        json.loads(paths["opened32_recipe"].read_text(encoding="utf-8"))["panel"][
            "source_filenames"
        ]
    )
    held32 = tuple(
        json.loads(paths["held32_recipe"].read_text(encoding="utf-8"))["panel"][
            "source_filenames"
        ]
    )
    prior_fresh32 = tuple(
        json.loads(paths["prior_fresh32_recipe"].read_text(encoding="utf-8"))[
            "panel"
        ]["source_filenames"]
    )
    focal_recipe = json.loads(paths["focal_current_recipe"].read_text(encoding="utf-8"))
    training_meta = json.loads(
        paths["focal_training224_metadata"].read_text(encoding="utf-8")
    )
    if focal_recipe.get("selection", {}).get("parent_roster", {}).get("source_count") != 256:
        raise ValueError("focal/current recipe no longer derives from train256")
    if training_meta.get("selection", {}).get("train256_indices") != "0:96 + 128:256":
        raise ValueError("focal training224 index selection changed")
    local32 = train256[96:128]
    focal_training224 = train256[:96] + train256[128:256]
    fullres = json.loads(
        paths["fullres_boundary_preregistration"].read_text(encoding="utf-8")
    ).get("selection", {})
    rosters = {
        "train256": train256,
        "local32_96_128": local32,
        "opened32": opened32,
        "held32": held32,
        "prior_fresh32": prior_fresh32,
        "focal_current_training224": focal_training224,
        "fullres_train32": tuple(fullres.get("train_filenames", ())),
        "fullres_eval16": tuple(fullres.get("eval_filenames", ())),
        "fullres_terminal16": tuple(fullres.get("terminal_filenames", ())),
    }
    expected_counts = {
        "train256": 256,
        "local32_96_128": 32,
        "opened32": 16,
        "held32": 16,
        "prior_fresh32": 16,
        "focal_current_training224": 224,
        "fullres_train32": 32,
        "fullres_eval16": 16,
        "fullres_terminal16": 16,
    }
    registered = config.get("panel", {}).get("exclusion_rosters")
    if not isinstance(registered, Mapping):
        raise ValueError("preregistered exclusion roster manifest is absent")
    for name, roster in rosters.items():
        record = registered.get(name)
        if (
            not isinstance(record, Mapping)
            or len(roster) != expected_counts[name]
            or record.get("count") != len(roster)
            or record.get("digest") != _names_digest(roster)
        ):
            raise ValueError(f"preregistered exclusion roster changed: {name}")
    return rosters


def _deterministic_roster(eligible: Sequence[str]) -> tuple[str, ...]:
    prefix = f"{SELECTION_NAMESPACE}\0{SELECTION_SEED}\0".encode()
    return tuple(
        sorted(
            eligible,
            key=lambda name: (hashlib.sha256(prefix + name.encode()).digest(), name),
        )[:SOURCE_COUNT]
    )


def _validate_preregistration(config: Mapping[str, Any]) -> tuple[str, ...]:
    rosters = _registered_rosters(config)
    excluded = set().union(*(set(roster) for roster in rosters.values()))
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
        "eligible_count": len(eligible),
        "eligible_digest": _names_digest(eligible),
        "source_count": SOURCE_COUNT,
        "draws": list(DRAWS),
        "case_count": CASE_COUNT,
        "source_order_digest": _names_digest(roster),
        "cases_digest": _cases_digest(roster),
        "source_filenames": list(roster),
    }
    for key, value in fixed.items():
        if panel.get(key) != value:
            raise ValueError(f"preregistered panel field changed: {key}")
    if set(roster) & excluded:
        raise RuntimeError("registered confirmation panel overlaps an exclusion roster")
    candidate = config.get("candidate", {})
    expected_candidate = {
        "matcher_vote_target": 350,
        "portfolio_arms": list(ARM_NAMES),
        "portfolio_selector": "minimum original all-1104-bond TASKA seam cost",
        "focal_mode": FOCAL_MODE,
        "control_protected_edges": "all harvested candidate edges",
        "candidate_protected_edges": (
            "harvested candidate edges with frozen train_exact_top5 focal logit >= 0.0"
        ),
        "focal_logit_threshold": FOCAL_PROTECTION_LOGIT_THRESHOLD,
        "tail_max_swaps": TAIL_MAX_SWAPS,
        "tail_minimum_gain": TAIL_MINIMUM_GAIN,
        "threshold_budget_arm_sweep": False,
    }
    if candidate != expected_candidate:
        raise ValueError("fixed candidate recipe changed")
    evaluation = config.get("evaluation", {})
    expected_evaluation = {
        "primary_metric": "satisfied_adjacent_pairs_per_board",
        "pair_denominator": PAIR_DENOMINATOR,
        "secondary_metrics": ["adjacency_recall", "exact_tiles_per_board"],
        "bootstrap_unit": "source_with_two_draws",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirmation_gate": {
            "pair_delta_mean_at_least": PAIR_GATE_MEAN,
            "pair_delta_ci95_lower_at_least": PAIR_GATE_CI95_LOWER,
        },
    }
    if evaluation != expected_evaluation:
        raise ValueError("fixed evaluation protocol changed")
    return roster


def _load_manifest(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    manifest_path = _require_record(config, "manifest", PREREG_INPUTS["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    layout = np.ascontiguousarray(value, dtype=np.int32)
    count = GRID_SIZE * GRID_SIZE
    if layout.shape != (count,) or not np.array_equal(np.sort(layout), np.arange(count)):
        raise ValueError("layout is not a strict 576-tile permutation")
    return layout


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


def _freeze_target_free(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    specs: Sequence[tuple[Mapping[str, Any], str, int]],
    targets: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[Path, Path, Path, float]:
    output_dir.mkdir(parents=True, exist_ok=False)
    archive_path = output_dir / "frozen-target-free-eval.npz"
    metadata_path = output_dir / "frozen-target-free-eval.json"
    freeze_path = output_dir / "pre-score-freeze.json"
    resources = load_taska_pair_pipeline_resources(device=device)
    cache = synthetic.CleanTileCache(targets.resolve())
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    started = perf_counter()

    for index, (record, source, draw) in enumerate(specs):
        prefix = f"case_{index:03d}"
        dirty = synthetic._dirty_case(cache, record, source, draw)
        dirty_sha = synthetic._dirty_sha256(dirty.dirty_tiles)
        matched = match_taska_tiles(
            dirty.dirty_tiles,
            resources.matchers,
            config=MATCHER_CONFIG,
            device=resources.device,
            require_verified=True,
        )
        focal = score_focal_edges(
            resources.focal_verifier,
            dirty.dirty_tiles,
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            mode=FOCAL_MODE,
            grid=GRID_SIZE,
            device=resources.device,
        )
        margins, votes = _edge_evidence(matched)
        features = extract_taska_edge_features(
            matched.cost_right,
            matched.cost_down,
            matched.right_log,
            matched.down_log,
            matched.candidate_edges,
            margins,
            votes,
            grid=GRID_SIZE,
        ).values
        priorities = {
            "logistic": resources.logistic_calibrator.predict_priorities(features),
            "focal_top5": focal.logits,
            "nonlinear": resources.nonlinear_calibrator.predict_priorities(features),
        }
        raw = solve_raw_tail_global(
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            border_unary=None,
            grid=GRID_SIZE,
            config=SOLVER_CONFIG,
        )
        solved = {
            "raw": raw,
            **{
                name: solve_prioritized_raw_tail_global(
                    matched.cost_right,
                    matched.cost_down,
                    matched.candidate_edges,
                    priorities[name],
                    border_unary=None,
                    grid=GRID_SIZE,
                    config=SOLVER_CONFIG,
                )
                for name in ("logistic", "focal_top5", "nonlinear")
            },
        }
        if tuple(solved) != ARM_NAMES:
            raise RuntimeError("four-arm order changed")
        layouts = {name: _strict_layout(result.layout) for name, result in solved.items()}
        selected = select_lowest_taska_seam_cost_layout(
            layouts,
            matched.cost_right,
            matched.cost_down,
            grid=GRID_SIZE,
        )
        pre_tail = _strict_layout(selected.layout)
        control = polish_unprotected_taska_tail(
            pre_tail,
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            grid=GRID_SIZE,
            max_swaps=TAIL_MAX_SWAPS,
            minimum_gain=TAIL_MINIMUM_GAIN,
        )
        candidate = polish_taska_tail_with_focal_gate(
            pre_tail,
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            focal.logits,
            grid=GRID_SIZE,
        )
        control_layout = _strict_layout(control.layout)
        candidate_layout = _strict_layout(candidate.layout)
        edge_sources = np.asarray(
            [edge.source for edge in matched.candidate_edges], dtype=np.int16
        )
        edge_targets = np.asarray(
            [edge.target for edge in matched.candidate_edges], dtype=np.int16
        )
        edge_axes = np.asarray(
            [0 if edge.axis == "right" else 1 for edge in matched.candidate_edges],
            dtype=np.uint8,
        )
        arrays[f"{prefix}__edge_source"] = edge_sources
        arrays[f"{prefix}__edge_target"] = edge_targets
        arrays[f"{prefix}__edge_axis"] = edge_axes
        arrays[f"{prefix}__focal_logits"] = np.asarray(focal.logits, dtype=np.float32)
        arrays[f"{prefix}__pre_tail_layout"] = pre_tail
        arrays[f"{prefix}__{ARMS[0]}_layout"] = control_layout
        arrays[f"{prefix}__{ARMS[1]}_layout"] = candidate_layout
        rows.append(
            {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source,
                "draw_index": draw,
                "dirty_sha256": dirty_sha,
                "candidate_edge_count": len(matched.candidate_edges),
                "chosen_vote_threshold": matched.chosen_vote_threshold,
                "portfolio_choice": selected.choice,
                "portfolio_total_costs": dict(selected.total_costs),
                "solver_diagnostics": {
                    name: asdict(result.diagnostics) for name, result in solved.items()
                },
                "control_tail": asdict(control.diagnostics),
                "candidate_focal_gate": asdict(candidate.diagnostics),
            }
        )
        print(
            json.dumps(
                {
                    "event": "fresh16_focal_gate_target_free_case_ready",
                    "case": index + 1,
                    "case_count": len(specs),
                    "source_filename": source,
                    "draw_index": draw,
                    "portfolio_choice": selected.choice,
                    "candidate_edges": len(matched.candidate_edges),
                    "focal_kept_edges": candidate.diagnostics.focal_kept_edge_count,
                    "control_swaps": control.diagnostics.accepted_swap_count,
                    "candidate_swaps": candidate.diagnostics.tail.accepted_swap_count,
                }
            ),
            flush=True,
        )

    _write_npz_exclusive(archive_path, arrays)
    _write_json_exclusive(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "all_layouts_are_strict_original_upright_tile_permutations": True,
            "matcher_config": asdict(MATCHER_CONFIG),
            "solver_config": asdict(SOLVER_CONFIG),
            "portfolio_arms": list(ARM_NAMES),
            "focal_mode": FOCAL_MODE,
            "control_tail_max_swaps": TAIL_MAX_SWAPS,
            "candidate_tail_max_swaps": TAIL_MAX_SWAPS,
            "candidate_focal_logit_threshold": FOCAL_PROTECTION_LOGIT_THRESHOLD,
            "rows": rows,
        },
    )
    runtime_sources = {
        "confirmation_runner": Path(__file__).resolve(),
        "focal_gated_tail": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_gated_protected_tail.py"
        ),
        "pair_pipeline": PROJECT_ROOT / "src/aiijc_puzzle/taska_pair_pipeline.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        "matcher": PROJECT_ROOT / "src/aiijc_puzzle/taska_seam_matcher.py",
        "focal_verifier": PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_verifier.py",
        "edge_calibrator": PROJECT_ROOT / "src/aiijc_puzzle/taska_edge_calibrator.py",
        "nonlinear_calibrator": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_nonlinear_calibrator.py"
        ),
        "layout_portfolio": PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py",
        "protected_tail": PROJECT_ROOT / "src/aiijc_puzzle/taska_protected_tail_polish.py",
        "layout_evaluation": PROJECT_ROOT / "src/aiijc_puzzle/layout_evaluation.py",
        "synthetic_generator": (
            PROJECT_ROOT / "src/aiijc_puzzle/synthetic_socket_evaluation.py"
        ),
    }
    model_paths = TaskaPairArtifactPaths()
    artifacts = {
        "preregistration": _record(config_path),
        "preregistration_sidecar": _record(Path(f"{config_path}.sha256")),
        "frozen_archive": _record(archive_path),
        "frozen_metadata": _record(metadata_path),
        **{name: _record(path) for name, path in runtime_sources.items()},
        "matcher_v3": _record(model_paths.matcher_v3),
        "matcher_local": _record(model_paths.matcher_local),
        "logistic_calibrator": _record(model_paths.logistic_calibrator),
        "focal_checkpoint": _record(model_paths.focal_verifier),
        "nonlinear_calibrator_artifact": _record(model_paths.nonlinear_calibrator),
    }
    _write_json_exclusive(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "panel_is_current_lineage_disjoint_only": True,
            "universal_or_model_freshness_claimed": False,
            "device": str(device),
            "verified_production_artifact_sha256": dict(EXPECTED_ARTIFACT_SHA256),
            "artifacts": artifacts,
        },
    )
    return archive_path, metadata_path, freeze_path, perf_counter() - started


def _validate_freeze(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("pre-score freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains labels")
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
    source_means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(
            0, len(source_means), size=(stop - start, len(source_means))
        )
        distribution[start:stop] = source_means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
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
    sources = [str(row["source_filename"]) for row in rows]
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    arms = {
        arm: {
            metric: float(np.mean([row[arm][metric] for row in rows]))
            for metric in metrics
        }
        for arm in ARMS
    }
    deltas = {
        metric: _cluster_ci(
            [
                float(row[ARMS[1]][metric]) - float(row[ARMS[0]][metric])
                for row in rows
            ],
            sources,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, metric in enumerate(metrics)
    }
    pair = deltas[PRIMARY_METRIC]
    gate = {
        "required_pair_delta_mean": PAIR_GATE_MEAN,
        "required_pair_delta_ci95_lower": PAIR_GATE_CI95_LOWER,
        "observed_pair_delta_mean": pair["mean"],
        "observed_pair_delta_ci95_lower": pair["ci95_lower"],
        "passed": (
            pair["mean"] >= PAIR_GATE_MEAN
            and pair["ci95_lower"] >= PAIR_GATE_CI95_LOWER
        ),
    }
    return {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": arms,
        "candidate_minus_control": deltas,
        "confirmation_gate": gate,
        "portfolio_choice_counts": dict(Counter(row["portfolio_choice"] for row in rows)),
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
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frozen_rows = metadata.get("rows")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != CASE_COUNT:
        raise RuntimeError("frozen candidate row roster changed")
    cache = synthetic.CleanTileCache(targets.resolve())
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
            exact = _strict_layout(reference.tile_at_position)
            prefix = str(frozen["prefix"])
            row: dict[str, Any] = {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source,
                "draw_index": draw,
                "portfolio_choice": str(frozen["portfolio_choice"]),
            }
            for arm in ARMS:
                layout = _strict_layout(archive[f"{prefix}__{arm}_layout"])
                row[arm] = _layout_metrics(layout, exact)
            scored.append(row)
    return scored, _summarize(scored)


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = _load_config(config_path)
    roster = _validate_preregistration(config)
    manifest = _load_manifest(config)
    specs = [(manifest[name], name, draw) for name in roster for draw in DRAWS]
    if len(specs) != CASE_COUNT:
        raise RuntimeError("registered panel expansion changed")
    raw_solver = PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
    if sha256_file(raw_solver) != RAW_SOLVER_SHA256:
        raise ValueError("frozen raw solver SHA-256 changed")
    device = synthetic._select_device(
        args.device,
        allow_nondeterministic_mps=bool(args.allow_nondeterministic_mps),
    )
    started = perf_counter()
    archive, metadata, freeze, inference_seconds = _freeze_target_free(
        config_path=config_path,
        config=config,
        specs=specs,
        targets=args.targets.resolve(),
        output_dir=args.output_dir.resolve(),
        device=device,
    )
    print(
        json.dumps(
            {
                "event": "fresh16_focal_gate_both_layouts_frozen_before_scoring",
                "case_count": len(specs),
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
            "current_lineage_disjoint": True,
            "universal_or_model_freshness_claimed": False,
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_seed": SELECTION_SEED,
        },
        "candidate": {
            "fixed_before_panel_scoring": True,
            "matcher_vote_target": MATCHER_CONFIG.vote_target,
            "portfolio_arms": list(ARM_NAMES),
            "portfolio_selector": "minimum original TASKA all-1104-bond seam cost",
            "focal_mode": FOCAL_MODE,
            "focal_logit_threshold": FOCAL_PROTECTION_LOGIT_THRESHOLD,
            "tail_max_swaps": TAIL_MAX_SWAPS,
            "threshold_budget_arm_changes": False,
        },
        "frozen_eval": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
            "both_strict_layouts_frozen_before_exact_reference_reconstruction": True,
            "contains_exact_references_or_labels": False,
        },
        "metrics": metrics,
        "rows": rows,
        "runtime_seconds": {
            "target_free_matcher_focal_and_solver": inference_seconds,
            "total": perf_counter() - started,
        },
        "legality": {
            "organizer_train_sources_only": True,
            "dirty_tiles_only_for_candidate_inference": True,
            "target_ids_or_exact_references_used_during_candidate_inference": False,
            "original_upright_20x20_tile_permutations_only": True,
            "pixels_rendered_or_replaced": False,
            "competition_test_accessed": False,
        },
    }
    _write_json_exclusive(args.output_dir.resolve() / "report.json", report)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
