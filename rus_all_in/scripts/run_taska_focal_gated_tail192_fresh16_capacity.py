#!/usr/bin/env python3
"""Run one preregistered focal-gated TASKA tail96 -> tail192 capacity step.

Both arms use target350 matching, the same four-arm original-cost selector,
the same focal-logit-zero protected edges, the same non-adjacent greedy tail,
and the same strict pre-tail layout.  Only max_swaps differs: 96 versus 192.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import solve_raw_tail_global
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_edge_calibrator import (
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    FOCAL_PROTECTION_LOGIT_THRESHOLD,
)
from aiijc_puzzle.taska_focal_gated_tail_capacity import (
    FOCAL_GATED_CANDIDATE_MAX_SWAPS,
    FOCAL_GATED_CAPACITY_MINIMUM_GAIN,
    FOCAL_GATED_CONTROL_MAX_SWAPS,
    compare_focal_gated_tail96_to_tail192,
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
    TaskaPairArtifactPaths,
    load_taska_pair_pipeline_resources,
)
from aiijc_puzzle.taska_seam_matcher import match_taska_tiles

try:
    from scripts import run_taska_focal_gated_protected_tail_fresh16_confirmation as previous
    from scripts import run_taska_protected_tail_fresh32_confirmation as synthetic
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_gated_protected_tail_fresh16_confirmation as previous
    import run_taska_protected_tail_fresh32_confirmation as synthetic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/taska_focal_gated_tail192_fresh16_capacity_v1.json"
)
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/taska-focal-gated-tail-capacity/tail192-fresh16-v1"
)

CONFIG_SCHEMA = "aiijc-taska-focal-gated-tail192-fresh16-capacity-config-v1"
FROZEN_SCHEMA = "aiijc-taska-focal-gated-tail192-fresh16-capacity-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-focal-gated-tail192-fresh16-capacity-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-focal-gated-tail192-fresh16-capacity-report-v1"
SOURCE_MINIMUM = 6_400
SOURCE_MAXIMUM = 6_699
SOURCE_COUNT = 16
DRAWS = (0, 1)
CASE_COUNT = SOURCE_COUNT * len(DRAWS)
SELECTION_NAMESPACE = (
    "aiijc-taska-focal-gated-tail192-capacity-fresh16-v1-source16xdraw2"
)
SELECTION_SEED = 2_026_083_193
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 192_160_931
PAIR_GATE_MEAN = 0.5
PAIR_GATE_CI95_LOWER = -0.25
ARMS = ("control_focal_gated_tail96", "candidate_focal_gated_tail192")
PRIMARY_METRIC = "satisfied_adjacent_pairs"
RAW_SOLVER_SHA256 = (
    "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
)

PREREG_INPUTS = {
    "manifest": "data/interim/validation_manifest.json",
    "taska_train256": "outputs/taska-edge-calibrator/train256-v1/roster.json",
    "taska_extension128": (
        "outputs/taska-focal-feature-stacker/train224-v1/"
        "extension128-focal-harvest.json"
    ),
    "taska_opened32": "configs/taska_seam_replay_opened32_v1.json",
    "taska_held32": "configs/taska_seam_held300_diagnostic_v1.json",
    "taska_fresh32": "configs/taska_protected_tail_fresh32_confirmation_v1.json",
    "taska_focal_current": "configs/taska_focal_current_finetune_v1.json",
    "taska_focal_train224": (
        "outputs/taska-focal-feature-stacker/train224-v1/"
        "training-stacked-features.json"
    ),
    "taska_fresh16_confirmation": (
        "configs/taska_focal_gated_protected_tail_fresh16_confirmation_v1.json"
    ),
    "fullres_denoiser": (
        "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
        "preregistered-local-gate.json"
    ),
    "active_train96_local": (
        "outputs/taska-focal-feature-stacker/train96-v1/local32/"
        "frozen-target-free-eval.json"
    ),
    "active_train96_held": (
        "outputs/taska-focal-feature-stacker/train96-v1/held32/"
        "frozen-target-free-eval.json"
    ),
    "active_train96_fresh": (
        "outputs/taska-focal-feature-stacker/train96-v1/fresh32-exact-override/"
        "frozen-target-free-eval.json"
    ),
    "active_fullres_union_local": (
        "outputs/taska-fullres-union-voter/fixed-v1/local32/"
        "frozen-target-free-eval.json"
    ),
    "active_fullres_union_held": (
        "outputs/taska-fullres-union-voter/fixed-v1/held32/"
        "frozen-target-free-eval.json"
    ),
    "active_fullres_union_fresh": (
        "outputs/taska-fullres-union-voter/fixed-v1/fresh32/"
        "frozen-target-free-eval.json"
    ),
    "active_incidence_local": (
        "outputs/taska-incidence-gnn/extension128-v1/local32/"
        "frozen-target-free-eval.json"
    ),
    "active_incidence_held": (
        "outputs/taska-incidence-gnn/extension128-v1/held32/"
        "frozen-target-free-eval.json"
    ),
    "active_incidence_fresh": (
        "outputs/taska-incidence-gnn/extension128-v1/fresh32/"
        "frozen-target-free-eval.json"
    ),
    "active_fullres_focal_local": (
        "outputs/taska-fullres-focal-gated-tail/fixed-v1/local32/"
        "frozen-target-free-eval.json"
    ),
    "active_fullres_focal_held": (
        "outputs/taska-fullres-focal-gated-tail/fixed-v1/held32/"
        "frozen-target-free-eval.json"
    ),
    "active_focal_gate_local": (
        "outputs/taska-focal-gated-protected-tail/logit0-v1/local32/"
        "frozen-target-free-eval.json"
    ),
    "active_focal_gate_held": (
        "outputs/taska-focal-gated-protected-tail/logit0-v1/held32/"
        "frozen-target-free-eval.json"
    ),
    "active_focal_gate_fresh": (
        "outputs/taska-focal-gated-protected-tail/logit0-v1/fresh32/"
        "frozen-target-free-eval.json"
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


def _record(path: Path) -> dict[str, str]:
    return previous._record(path)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    previous._write_json_exclusive(path, payload)


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    previous._write_npz_exclusive(path, arrays)


def _names_digest(names: Sequence[str]) -> str:
    return previous._names_digest(names)


def _cases_digest(names: Sequence[str]) -> str:
    serialized = "\n".join(f"{name}\0{draw}" for name in names for draw in DRAWS)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _unique_names(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _full_universe() -> tuple[str, ...]:
    return tuple(
        f"img_{index:06d}.png" for index in range(SOURCE_MINIMUM, SOURCE_MAXIMUM + 1)
    )


def _require_record(config: Mapping[str, Any], name: str) -> Path:
    expected_path = PREREG_INPUTS[name]
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
        raise ValueError("preregistration JSON and SHA sidecar are required")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if len(tokens) not in {1, 2} or tokens[0] != sha256_file(resolved):
        raise ValueError("preregistration SHA sidecar does not match")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("preregistration schema changed")
    return config


def _panel_names(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _unique_names(payload["panel"]["source_filenames"])


def _row_names(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _unique_names([row["source_filename"] for row in payload["rows"]])


def _registered_rosters(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    paths = {name: _require_record(config, name) for name in PREREG_INPUTS}
    train256 = _unique_names(
        json.loads(paths["taska_train256"].read_text(encoding="utf-8"))[
            "source_filenames"
        ]
    )
    extension_payload = json.loads(
        paths["taska_extension128"].read_text(encoding="utf-8")
    )
    extension128 = _unique_names(
        [row["source_filename"] for row in extension_payload["rows"]]
    )
    if extension128 != train256[128:256]:
        raise ValueError("TASKA extension128 no longer equals train256[128:256]")
    focal_current = json.loads(
        paths["taska_focal_current"].read_text(encoding="utf-8")
    )
    focal_train224 = json.loads(
        paths["taska_focal_train224"].read_text(encoding="utf-8")
    )
    if focal_current.get("selection", {}).get("parent_roster", {}).get(
        "source_count"
    ) != 256:
        raise ValueError("focal/current parent roster changed")
    if focal_train224.get("selection", {}).get("train256_indices") != (
        "0:96 + 128:256"
    ):
        raise ValueError("focal train224 selection changed")
    fullres = json.loads(paths["fullres_denoiser"].read_text(encoding="utf-8"))[
        "selection"
    ]
    rosters: dict[str, tuple[str, ...]] = {
        "taska_train256": train256,
        "taska_extension128": extension128,
        "taska_focal_train224": train256[:96] + train256[128:256],
        "taska_local32": train256[96:128],
        "taska_opened32": _panel_names(paths["taska_opened32"]),
        "taska_held32": _panel_names(paths["taska_held32"]),
        "taska_fresh32": _panel_names(paths["taska_fresh32"]),
        "taska_fresh16_confirmation": _panel_names(
            paths["taska_fresh16_confirmation"]
        ),
        "fullres_denoiser_train32": _unique_names(fullres["train_filenames"]),
        "fullres_denoiser_eval16": _unique_names(fullres["eval_filenames"]),
        "fullres_denoiser_terminal16": _unique_names(fullres["terminal_filenames"]),
    }
    for name, path in paths.items():
        if name.startswith("active_"):
            rosters[name] = _row_names(path)
    registered = config.get("panel", {}).get("exclusion_rosters")
    if not isinstance(registered, Mapping) or set(registered) != set(rosters):
        raise ValueError("preregistered exclusion roster inventory changed")
    for name, roster in rosters.items():
        record = registered[name]
        if (
            not isinstance(record, Mapping)
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
    fixed_panel = {
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
    for name, expected in fixed_panel.items():
        if panel.get(name) != expected:
            raise ValueError(f"preregistered panel field changed: {name}")
    if set(roster) & excluded:
        raise RuntimeError("tail192 roster collides with a signed exclusion roster")
    expected_candidate = {
        "matcher_vote_target": 350,
        "portfolio_arms": list(ARM_NAMES),
        "portfolio_selector": "minimum original all-1104-bond TASKA seam cost",
        "focal_mode": FOCAL_MODE,
        "protected_edges_both_arms": (
            "harvested candidate edges with frozen train_exact_top5 focal logit >= 0.0"
        ),
        "focal_logit_threshold": FOCAL_PROTECTION_LOGIT_THRESHOLD,
        "control_tail_max_swaps": FOCAL_GATED_CONTROL_MAX_SWAPS,
        "candidate_tail_max_swaps": FOCAL_GATED_CANDIDATE_MAX_SWAPS,
        "tail_minimum_gain": FOCAL_GATED_CAPACITY_MINIMUM_GAIN,
        "tail_swap_positions": "non-adjacent only",
        "identical_pre_tail_layout": True,
        "budget_threshold_arm_sweep": False,
    }
    if config.get("candidate") != expected_candidate:
        raise ValueError("fixed tail192 candidate recipe changed")
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
    if config.get("evaluation") != expected_evaluation:
        raise ValueError("fixed tail192 evaluation protocol changed")
    return roster


def _strict_layout(value: Any) -> np.ndarray:
    return previous._strict_layout(value)


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
        margins, votes = previous._edge_evidence(matched)
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
        solved = {
            "raw": solve_raw_tail_global(
                matched.cost_right,
                matched.cost_down,
                matched.candidate_edges,
                border_unary=None,
                grid=GRID_SIZE,
                config=SOLVER_CONFIG,
            ),
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
            layouts, matched.cost_right, matched.cost_down, grid=GRID_SIZE
        )
        pre_tail = _strict_layout(selected.layout)
        capacity = compare_focal_gated_tail96_to_tail192(
            pre_tail,
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            focal.logits,
            grid=GRID_SIZE,
        )
        control_layout = _strict_layout(capacity.control.layout)
        candidate_layout = _strict_layout(capacity.candidate.layout)
        arrays[f"{prefix}__edge_source"] = np.asarray(
            [edge.source for edge in matched.candidate_edges], dtype=np.int16
        )
        arrays[f"{prefix}__edge_target"] = np.asarray(
            [edge.target for edge in matched.candidate_edges], dtype=np.int16
        )
        arrays[f"{prefix}__edge_axis"] = np.asarray(
            [edge.axis == "down" for edge in matched.candidate_edges], dtype=np.uint8
        )
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
                "dirty_sha256": synthetic._dirty_sha256(dirty.dirty_tiles),
                "candidate_edge_count": len(matched.candidate_edges),
                "chosen_vote_threshold": matched.chosen_vote_threshold,
                "portfolio_choice": selected.choice,
                "portfolio_total_costs": dict(selected.total_costs),
                "solver_diagnostics": {
                    name: asdict(result.diagnostics) for name, result in solved.items()
                },
                "capacity": asdict(capacity.diagnostics),
                "control_tail96": asdict(capacity.control.diagnostics.tail),
                "candidate_tail192": asdict(capacity.candidate.diagnostics),
            }
        )
        print(
            json.dumps(
                {
                    "event": "tail192_target_free_case_ready",
                    "case": index + 1,
                    "case_count": len(specs),
                    "source_filename": source,
                    "draw_index": draw,
                    "portfolio_choice": selected.choice,
                    "control_swaps": capacity.diagnostics.control_accepted_swaps,
                    "candidate_swaps": capacity.diagnostics.candidate_accepted_swaps,
                    "additional_swaps": capacity.diagnostics.additional_accepted_swaps,
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
            "focal_logit_threshold": FOCAL_PROTECTION_LOGIT_THRESHOLD,
            "control_tail_max_swaps": FOCAL_GATED_CONTROL_MAX_SWAPS,
            "candidate_tail_max_swaps": FOCAL_GATED_CANDIDATE_MAX_SWAPS,
            "rows": rows,
        },
    )
    runtime_sources = {
        "capacity_runner": Path(__file__).resolve(),
        "capacity_module": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_gated_tail_capacity.py"
        ),
        "focal_gated_tail": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_gated_protected_tail.py"
        ),
        "pair_pipeline": PROJECT_ROOT / "src/aiijc_puzzle/taska_pair_pipeline.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        "matcher": PROJECT_ROOT / "src/aiijc_puzzle/taska_seam_matcher.py",
        "focal_verifier": PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_verifier.py",
        "protected_tail": PROJECT_ROOT / "src/aiijc_puzzle/taska_protected_tail_polish.py",
        "layout_portfolio": PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py",
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
            "collision_audit_passed_before_inference": True,
            "universal_or_historical_model_freshness_claimed": False,
            "device": str(device),
            "verified_production_artifact_sha256": dict(EXPECTED_ARTIFACT_SHA256),
            "artifacts": artifacts,
        },
    )
    return archive_path, metadata_path, freeze_path, perf_counter() - started


def _summarize(
    rows: Sequence[Mapping[str, Any]], frozen_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
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
        metric: previous._cluster_ci(
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
    control_swaps = np.asarray(
        [row["capacity"]["control_accepted_swaps"] for row in frozen_rows]
    )
    candidate_swaps = np.asarray(
        [row["capacity"]["candidate_accepted_swaps"] for row in frozen_rows]
    )
    return {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": arms,
        "candidate_minus_control": deltas,
        "confirmation_gate": {
            "required_pair_delta_mean": PAIR_GATE_MEAN,
            "required_pair_delta_ci95_lower": PAIR_GATE_CI95_LOWER,
            "observed_pair_delta_mean": pair["mean"],
            "observed_pair_delta_ci95_lower": pair["ci95_lower"],
            "passed": (
                pair["mean"] >= PAIR_GATE_MEAN
                and pair["ci95_lower"] >= PAIR_GATE_CI95_LOWER
            ),
        },
        "capacity_diagnostics": {
            "control_cap96_reached_cases": int(np.sum(control_swaps == 96)),
            "candidate_cap192_reached_cases": int(np.sum(candidate_swaps == 192)),
            "mean_control_accepted_swaps": float(control_swaps.mean()),
            "mean_candidate_accepted_swaps": float(candidate_swaps.mean()),
            "mean_additional_accepted_swaps": float(
                (candidate_swaps - control_swaps).mean()
            ),
            "minimum_additional_accepted_swaps": int(
                np.min(candidate_swaps - control_swaps)
            ),
            "maximum_additional_accepted_swaps": int(
                np.max(candidate_swaps - control_swaps)
            ),
        },
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
    previous._validate_freeze(freeze_path)
    frozen_rows = json.loads(metadata_path.read_text(encoding="utf-8"))["rows"]
    if len(frozen_rows) != CASE_COUNT:
        raise RuntimeError("frozen tail192 row roster changed")
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
                frozen["source_filename"] != source
                or int(frozen["draw_index"]) != draw
                or frozen["case_id"] != dirty.case_id
                or synthetic._dirty_sha256(dirty.tiles) != frozen["dirty_sha256"]
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
                row[arm] = previous._layout_metrics(layout, exact)
            scored.append(row)
    return scored, _summarize(scored, frozen_rows)


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = _load_config(config_path)
    roster = _validate_preregistration(config)
    manifest = previous._load_manifest(config)
    specs = [(manifest[name], name, draw) for name in roster for draw in DRAWS]
    if len(specs) != CASE_COUNT:
        raise RuntimeError("registered tail192 panel expansion changed")
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
                "event": "tail96_and_tail192_frozen_before_scoring",
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
            "signed_collision_audit_passed": True,
            "universal_or_historical_model_freshness_claimed": False,
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
            "control_tail_max_swaps": FOCAL_GATED_CONTROL_MAX_SWAPS,
            "candidate_tail_max_swaps": FOCAL_GATED_CANDIDATE_MAX_SWAPS,
            "only_difference_is_swap_capacity": True,
            "budget_threshold_arm_or_roster_sweep": False,
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
            "pixels_rendered_replaced_rotated_or_warped": False,
            "competition_test_accessed": False,
            "postprocess_applied": False,
        },
    }
    _write_json_exclusive(args.output_dir.resolve() / "report.json", report)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
