#!/usr/bin/env python3
"""Run the preregistered independent selective+fullres fusion confirmation.

Each case uses one target500 matcher pass.  The unchanged selective-target500
solver is the exact control.  Four fixed restored scorer views then emit the
already-fixed support>=3/4, focal-logit>=0 supply; edges overlapping current or
selective accepted edges are discarded and only the remaining edges enter one
combined sixth arm.  Original dense all-1104 seam costs and focal-gated tail96
are unchanged.  All target-free evidence is frozen before references exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
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
from aiijc_puzzle.taska_fullres_union_voter import (
    FULLRES_DENOISER_SHA256,
    NEW_EDGE_FOCAL_LOGIT_MINIMUM,
    RESTORED_SCORER_COUNT,
    RESTORED_SUPPORT_MINIMUM,
    accept_focal_proposals,
    load_fullres_denoiser,
    restore_fixed_matcher_view,
    restored_mutual_scorer_sets,
    supported_absent_edges,
)
from aiijc_puzzle.taska_pair_pipeline import (
    ARM_NAMES,
    EXPECTED_ARTIFACT_SHA256,
    FOCAL_MODE,
    GRID_SIZE,
    PAIR_DENOMINATOR,
    SOLVER_CONFIG,
    TAIL_MAX_SWAPS,
    TaskaPairArtifactPaths,
    load_taska_pair_pipeline_resources,
)
from aiijc_puzzle.taska_seam_matcher import match_taska_tiles
from aiijc_puzzle.taska_selective_fullres_fusion import (
    FUSION_ARM_NAMES,
    compose_selective_fullres_fusion,
    strict_layout,
)
from aiijc_puzzle.taska_selective_vote500 import (
    SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD,
    _edge_evidence,
    compose_selective_vote500,
    same_pass_target350,
)
from aiijc_puzzle.taska_vote500 import VOTE500_MATCHER_CONFIG

try:
    from scripts import run_taska_protected_tail_fresh32_confirmation as synthetic
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_protected_tail_fresh32_confirmation as synthetic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/taska_selective_fullres_union_fusion_fresh32_confirmation_v1.json"
)
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/taska-selective-fullres-union-fusion/"
    "fresh32-formal-confirmation-v1"
)
FULLRES_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
    "fullres_boundary_denoiser.pt"
)

CONFIG_SCHEMA = (
    "aiijc-taska-selective-fullres-union-fusion-fresh32-confirmation-config-v1"
)
SNAPSHOT_SCHEMA = "aiijc-taska-selective-vote500-confirmation-exclusions-v1"
FROZEN_SCHEMA = (
    "aiijc-taska-selective-fullres-union-fusion-confirmation-target-free-v1"
)
FREEZE_SCHEMA = "aiijc-taska-selective-fullres-union-fusion-confirmation-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-selective-fullres-union-fusion-confirmation-report-v1"

SOURCE_MINIMUM = 6_000
SOURCE_MAXIMUM = 6_399
SOURCE_COUNT = 16
DRAWS = (0, 1)
CASE_COUNT = SOURCE_COUNT * len(DRAWS)
SELECTION_NAMESPACE = (
    "aiijc-taska-selective-fullres-union-fusion-formal-confirmation-v1-"
    "source16xdraw2"
)
SELECTION_SEED = 2_026_083_202
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 102_160_202
PAIR_GATE_MEAN = 1.0
PAIR_GATE_CI95_LOWER = 0.0
RAW_SOLVER_SHA256 = (
    "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
)
CONTROL_ARM = "selective_target500_focal_gated_tail96"
CANDIDATE_ARM = "selective_unique_fullres_fusion_focal_gated_tail96"
ARMS = (CONTROL_ARM, CANDIDATE_ARM)

EXPECTED_ARTIFACT_PATHS = {
    "manifest": "data/interim/validation_manifest.json",
    "exclusion_snapshot": (
        "configs/taska_selective_vote500_fresh32_confirmation_v1.exclusions.json"
    ),
    "exclusion_snapshot_sidecar": (
        "configs/taska_selective_vote500_fresh32_confirmation_v1."
        "exclusions.json.sha256"
    ),
    "selective_target500_confirmation_reservation": (
        "configs/taska_selective_vote500_fresh32_confirmation_v1.json"
    ),
    "fullres_confirmation_reservation": (
        "configs/taska_fullres_focal_gated_tail_fresh32_confirmation_v1.json"
    ),
    "tail192_reservation": (
        "configs/taska_focal_gated_tail192_fresh16_capacity_v1.json"
    ),
    "frozen_parent_report": (
        "outputs/taska-selective-fullres-union-fusion/fixed-v1/report.json"
    ),
    "fusion_solver": "src/aiijc_puzzle/taska_selective_fullres_fusion.py",
    "selective_solver": "src/aiijc_puzzle/taska_selective_vote500.py",
    "fullres_supply_solver": "src/aiijc_puzzle/taska_fullres_union_voter.py",
    "raw_solver": "src/aiijc_puzzle/raw_tail_global_solver.py",
}


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


def _digest(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _cases_digest(names: Sequence[str]) -> str:
    value = "\n".join(f"{name}\0{draw}" for name in names for draw in DRAWS)
    return hashlib.sha256(value.encode()).hexdigest()


def _load_signed_json(path: Path, *, schema: str) -> Mapping[str, Any]:
    resolved = path.resolve()
    sidecar = Path(f"{resolved}.sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise ValueError(f"signed JSON is absent: {resolved}")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if not tokens or tokens[0] != sha256_file(resolved):
        raise ValueError(f"signed JSON digest mismatch: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ValueError(f"signed JSON schema mismatch: {resolved}")
    return payload


def _require_record(
    config: Mapping[str, Any], name: str, expected_path: str
) -> Path:
    record = config.get("artifacts", {}).get(name)
    if not isinstance(record, Mapping) or record.get("path") != expected_path:
        raise ValueError(f"preregistered artifact path changed: {name}")
    path = (PROJECT_ROOT / expected_path).resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"preregistered artifact digest changed: {name}")
    return path


def _load_config(path: Path) -> Mapping[str, Any]:
    return _load_signed_json(path, schema=CONFIG_SCHEMA)


def _load_manifest(path: Path) -> tuple[dict[str, Mapping[str, Any]], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if compute_protocol_digest(payload) != payload.get("protocol_digest"):
        raise ValueError("organizer-train manifest protocol digest is invalid")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("organizer-train manifest splits are absent")
    rows = [row for values in splits.values() for row in values]
    lookup = {str(row["filename"]): row for row in rows}
    train = {str(row["filename"]) for row in splits.get("train", ())}
    if len(rows) != 7_000 or len(lookup) != 7_000 or len(train) != 5_600:
        raise ValueError("organizer-train manifest roster changed")
    return lookup, train


def _dependency_roster(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("panel", {}).get("source_filenames", ())
    if not isinstance(values, list) or len(values) != 16:
        raise ValueError(f"signed dependency roster changed: {path}")
    return {str(value) for value in values}


def _validate_preregistration(
    config_path: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, Mapping[str, Any]]]:
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        EXPECTED_ARTIFACT_PATHS
    ):
        raise ValueError("preregistered artifact registry changed")
    paths = {
        name: _require_record(config, name, expected)
        for name, expected in EXPECTED_ARTIFACT_PATHS.items()
    }
    if sha256_file(paths["raw_solver"]) != RAW_SOLVER_SHA256:
        raise ValueError("raw solver no longer matches the frozen SHA")

    snapshot = _load_signed_json(paths["exclusion_snapshot"], schema=SNAPSHOT_SCHEMA)
    union = snapshot.get("explicit_source_union")
    if not isinstance(union, Mapping):
        raise ValueError("signed exclusion union is absent")
    base_excluded = tuple(str(value) for value in union.get("source_filenames", ()))
    if (
        base_excluded != tuple(sorted(set(base_excluded)))
        or union.get("count") != len(base_excluded)
        or union.get("digest") != _digest(base_excluded)
    ):
        raise ValueError("signed exclusion union changed")
    prior_artifacts = snapshot.get("artifacts")
    if not isinstance(prior_artifacts, list) or not prior_artifacts:
        raise ValueError("signed exclusion artifact inventory is absent")
    for record in prior_artifacts:
        if not isinstance(record, Mapping):
            raise ValueError("malformed exclusion artifact record")
        artifact = (PROJECT_ROOT / str(record.get("path"))).resolve()
        if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
            raise ValueError(f"frozen prior TASKA artifact changed: {artifact}")

    selective_config = _load_signed_json(
        paths["selective_target500_confirmation_reservation"],
        schema="aiijc-taska-selective-vote500-fresh32-confirmation-config-v1",
    )
    selective_names = tuple(
        str(value)
        for value in selective_config.get("panel", {}).get("source_filenames", ())
    )
    if len(selective_names) != 16 or len(set(selective_names)) != 16:
        raise ValueError("selective confirmation reservation changed")
    excluded = tuple(sorted(set(base_excluded) | set(selective_names)))

    for name in ("fullres_confirmation_reservation", "tail192_reservation"):
        if not _dependency_roster(paths[name]) <= set(excluded):
            raise RuntimeError(f"required dependency roster was not excluded: {name}")

    lookup, train = _load_manifest(paths["manifest"])
    universe = tuple(
        sorted(
            name
            for name in train
            if SOURCE_MINIMUM <= int(name[4:10]) <= SOURCE_MAXIMUM
        )
    )
    excluded_set = set(excluded)
    eligible = tuple(name for name in universe if name not in excluded_set)
    prefix = f"{SELECTION_NAMESPACE}\0{SELECTION_SEED}\0".encode()
    roster = tuple(
        sorted(
            eligible,
            key=lambda name: (
                hashlib.sha256(prefix + name.encode()).digest(),
                name,
            ),
        )[:SOURCE_COUNT]
    )
    panel = config.get("panel", {})
    excluded_in_universe = tuple(sorted(set(universe) & excluded_set))
    fixed_panel = {
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_seed": SELECTION_SEED,
        "universe_minimum": "img_006000.png",
        "universe_maximum": "img_006399.png",
        "organizer_train_universe_count": len(universe),
        "organizer_train_universe_digest": _digest(universe),
        "exclusion_union_count": len(excluded),
        "exclusion_union_digest": _digest(excluded),
        "excluded_in_universe_count": len(excluded_in_universe),
        "excluded_in_universe": list(excluded_in_universe),
        "eligible_count": len(eligible),
        "eligible_digest": _digest(eligible),
        "source_filenames": list(roster),
        "source_count": SOURCE_COUNT,
        "draws": list(DRAWS),
        "case_count": CASE_COUNT,
        "source_order_digest": _digest(roster),
        "cases_digest": _cases_digest(roster),
    }
    for key, expected in fixed_panel.items():
        if panel.get(key) != expected:
            raise ValueError(f"preregistered panel field changed: {key}")
    if set(roster) & excluded_set or not set(roster) <= train:
        raise RuntimeError("confirmation roster is not disjoint organizer-train")

    candidate = config.get("candidate", {})
    fixed_candidate = {
        "entrypoint": (
            "aiijc_puzzle.taska_selective_fullres_fusion."
            "compose_selective_fullres_fusion"
        ),
        "matcher_passes_per_case": 1,
        "matcher_vote_target": 500,
        "same_pass_current_vote_target": 350,
        "selective_new_edges": "target500 minus same-pass current350",
        "selective_new_edge_acceptance": (
            "recovered train_exact_top5 focal logit >= 0.0"
        ),
        "restored_scorers": (
            "fixed fullres boundary denoiser, v3/local x first two audited "
            "orientations"
        ),
        "restored_scorer_count": RESTORED_SCORER_COUNT,
        "fullres_new_edge_rule": (
            "absent from current350 harvest, support >= 3/4, then recovered "
            "train_exact_top5 focal logit >= 0.0"
        ),
        "unique_fullres_rule": (
            "drop every accepted fullres edge already present in current350 or "
            "the selective accepted set"
        ),
        "portfolio_arms": [
            "raw",
            "logistic",
            "focal_top5",
            "nonlinear",
            "selective_vote500_focal",
            "combined_selective_unique_fullres_focal",
        ],
        "standalone_fullres_arm": False,
        "selector": "minimum original TASKA all-1104-bond seam cost",
        "control": (
            "exact unchanged selective-target500 five-arm final layout plus "
            "winner-aligned focal-gated tail96"
        ),
        "candidate_layout": (
            "six-arm winner plus winner-aligned focal-logit>=0 protected tail96"
        ),
        "tail_max_swaps": TAIL_MAX_SWAPS,
        "focal_logit_threshold": SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD,
        "restored_pixels_matcher_only": True,
        "raw_dense_cost_matrices_unchanged": True,
        "threshold_arm_budget_orientation_or_roster_sweep": False,
    }
    if candidate != fixed_candidate:
        raise ValueError("fixed fusion candidate changed")
    fixed_evaluation = {
        "primary_metric": (
            "candidate_minus_control_satisfied_adjacent_pairs_per_board"
        ),
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
    if config.get("evaluation") != fixed_evaluation:
        raise ValueError("fixed confirmation evaluation changed")
    if sha256_file(config_path) != sha256_file(DEFAULT_CONFIG):
        raise ValueError("non-default preregistration bytes are not allowed")
    return roster, lookup


def _edge_arrays(name: str, edges: Sequence[RawTailEdge]) -> dict[str, np.ndarray]:
    return {
        f"{name}__edge_source": np.asarray(
            [edge.source for edge in edges], dtype=np.int16
        ),
        f"{name}__edge_target": np.asarray(
            [edge.target for edge in edges], dtype=np.int16
        ),
        f"{name}__edge_axis": np.asarray(
            [edge.axis == "down" for edge in edges], dtype=np.uint8
        ),
    }


def _edges_from_archive(
    archive: Any, prefix: str, name: str
) -> tuple[RawTailEdge, ...]:
    sources = archive[f"{prefix}__{name}__edge_source"]
    targets = archive[f"{prefix}__{name}__edge_target"]
    axes = archive[f"{prefix}__{name}__edge_axis"]
    edges = tuple(
        RawTailEdge(int(source), int(target), "down" if int(axis) else "right")
        for source, target, axis in zip(sources, targets, axes, strict=True)
    )
    if len(set(edges)) != len(edges):
        raise ValueError("frozen edge list contains duplicates")
    return edges


def _four_layouts(matched: Any, focal: Any, resources: Any) -> dict[str, np.ndarray]:
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
    priorities = (
        resources.logistic_calibrator.predict_priorities(features),
        focal.logits,
        resources.nonlinear_calibrator.predict_priorities(features),
    )
    raw = solve_raw_tail_global(
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        grid=GRID_SIZE,
        config=SOLVER_CONFIG,
    )
    prioritized = tuple(
        solve_prioritized_raw_tail_global(
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            values,
            grid=GRID_SIZE,
            config=SOLVER_CONFIG,
        )
        for values in priorities
    )
    return {
        name: strict_layout(result.layout, grid=GRID_SIZE)
        for name, result in zip(ARM_NAMES, (raw, *prioritized), strict=True)
    }


def _target_free_case(
    dirty_tiles: np.ndarray,
    *,
    resources: Any,
    denoiser: Any,
    inference_batch: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    matched500 = match_taska_tiles(
        dirty_tiles,
        resources.matchers,
        config=VOTE500_MATCHER_CONFIG,
        device=resources.device,
        require_verified=True,
    )
    focal500 = score_focal_edges(
        resources.focal_verifier,
        dirty_tiles,
        matched500.cost_right,
        matched500.cost_down,
        matched500.candidate_edges,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.device,
    )
    selective = compose_selective_vote500(matched500, focal500, resources)
    matched350, focal350 = same_pass_target350(matched500, focal500)
    four = _four_layouts(matched350, focal350, resources)
    supply = selective.supply

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
    fullres_proposed, support = supported_absent_edges(
        supply.current_edges, scorer_sets
    )
    fullres_scores = score_focal_edges(
        resources.focal_verifier,
        dirty_tiles,
        matched350.cost_right,
        matched350.cost_down,
        fullres_proposed,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.device,
    )
    fullres_logits = np.asarray(fullres_scores.logits, dtype=np.float32)
    fullres_accepted, fullres_accepted_logits = accept_focal_proposals(
        fullres_proposed, fullres_logits
    )
    fusion = compose_selective_fullres_fusion(
        cost_right=matched350.cost_right,
        cost_down=matched350.cost_down,
        four_layouts=four,
        frozen_selective_control=selective.candidate_layout,
        current_edges=supply.current_edges,
        current_logits=supply.current_logits,
        selective_new_edges=supply.accepted_new_edges,
        selective_new_logits=supply.accepted_new_logits,
        fullres_accepted_edges=fullres_accepted,
        fullres_accepted_logits=fullres_accepted_logits,
        grid=GRID_SIZE,
    )
    if not np.array_equal(fusion.control_layout, fusion.mechanical_control_layout):
        raise RuntimeError("selective final control replay mismatch")
    if len(scorer_sets) != RESTORED_SCORER_COUNT:
        raise RuntimeError("restored scorer roster changed")

    edge_rosters = {
        "current": supply.current_edges,
        "selective_proposed": supply.proposed_new_edges,
        "selective_accepted": supply.accepted_new_edges,
        "fullres_proposed": fullres_proposed,
        "fullres_accepted": fullres_accepted,
        "unique_fullres": fusion.supply.unique_fullres_edges,
        "combined_union": fusion.supply.combined_union_edges,
    }
    arrays: dict[str, np.ndarray] = {
        "cost_right": np.ascontiguousarray(matched350.cost_right, dtype=np.float32),
        "cost_down": np.ascontiguousarray(matched350.cost_down, dtype=np.float32),
        "current_focal_logits": np.asarray(supply.current_logits, dtype=np.float32),
        "selective_proposed_focal_logits": np.asarray(
            supply.proposed_new_logits, dtype=np.float32
        ),
        "selective_accepted_focal_logits": np.asarray(
            supply.accepted_new_logits, dtype=np.float32
        ),
        "fullres_proposed_support": np.asarray(support, dtype=np.uint8),
        "fullres_proposed_focal_logits": fullres_logits,
        "fullres_accepted_focal_logits": np.asarray(
            fullres_accepted_logits, dtype=np.float32
        ),
        "unique_fullres_focal_logits": np.asarray(
            fusion.supply.unique_fullres_logits, dtype=np.float32
        ),
        "combined_union_focal_logits": np.asarray(
            fusion.supply.combined_union_logits, dtype=np.float32
        ),
        f"{CONTROL_ARM}_layout": strict_layout(
            fusion.control_layout, grid=GRID_SIZE
        ),
        f"{CANDIDATE_ARM}_layout": strict_layout(
            fusion.candidate_layout, grid=GRID_SIZE
        ),
        "mechanical_control_layout": strict_layout(
            fusion.mechanical_control_layout, grid=GRID_SIZE
        ),
        "selective_union_layout": strict_layout(
            fusion.selective_union_layout, grid=GRID_SIZE
        ),
        "combined_union_layout": strict_layout(
            fusion.combined_union_layout, grid=GRID_SIZE
        ),
        **{f"{name}_layout": layout for name, layout in four.items()},
    }
    for name, edges in edge_rosters.items():
        arrays.update(_edge_arrays(name, edges))

    diagnostics = {
        **fusion.diagnostics(),
        "target350_vote_threshold": matched350.chosen_vote_threshold,
        "target500_vote_threshold": matched500.chosen_vote_threshold,
        "target500_candidate_count": len(matched500.candidate_edges),
        "selective_proposed_count": len(supply.proposed_new_edges),
        "fullres_proposed_count": len(fullres_proposed),
        "restored_scorer_edge_counts": [len(edges) for edges in scorer_sets],
        "mechanical_selective_control_replay_matches": True,
        "one_target500_matcher_pass": True,
        "standalone_fullres_arm_used": False,
    }
    return arrays, diagnostics


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
        case_arrays, diagnostics = _target_free_case(
            dirty.dirty_tiles,
            resources=resources,
            denoiser=denoiser,
            inference_batch=inference_batch,
        )
        arrays.update(
            {f"{prefix}__{name}": value for name, value in case_arrays.items()}
        )
        rows.append(
            {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source,
                "draw_index": draw,
                "dirty_sha256": synthetic._dirty_sha256(dirty.dirty_tiles),
                **diagnostics,
            }
        )
        print(
            json.dumps(
                {
                    "event": "fusion_confirmation_target_free_case_ready",
                    "case": index + 1,
                    "case_count": len(specs),
                    "source_filename": source,
                    "draw_index": draw,
                    "selective_new": diagnostics[
                        "selective_accepted_new_count"
                    ],
                    "fullres_new": diagnostics["fullres_accepted_new_count"],
                    "unique_fullres": diagnostics[
                        "unique_fullres_accepted_count"
                    ],
                    "choice": diagnostics["choice"],
                }
            ),
            flush=True,
        )

    _write_npz_exclusive(archive_path, arrays)
    summary_fields = (
        "current_edge_count",
        "selective_accepted_new_count",
        "fullres_accepted_new_count",
        "fullres_overlap_current_count",
        "fullres_overlap_selective_count",
        "unique_fullres_accepted_count",
        "combined_union_edge_count",
    )
    target_free_summary = {
        "case_count": len(rows),
        "means_per_board": {
            field: float(np.mean([row[field] for row in rows]))
            for field in summary_fields
        },
        "fusion_choice_counts": dict(Counter(row["choice"] for row in rows)),
        "selective_replay_choice_counts": dict(
            Counter(row["selective_replay_choice"] for row in rows)
        ),
        "mechanical_control_replay_count": sum(
            bool(row["mechanical_selective_control_replay_matches"])
            for row in rows
        ),
    }
    _write_json_exclusive(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "all_layouts_strict_original_upright_tile_permutations": True,
            "all_dense_matrices_edges_logits_layouts_and_provenance_frozen": True,
            "one_target500_matcher_pass_per_case": True,
            "same_pass_target350_subset": True,
            "restored_pixels_matcher_only": True,
            "restored_scorer_count": RESTORED_SCORER_COUNT,
            "restored_support_minimum": RESTORED_SUPPORT_MINIMUM,
            "fullres_focal_logit_minimum": NEW_EDGE_FOCAL_LOGIT_MINIMUM,
            "selective_focal_logit_minimum": (
                SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD
            ),
            "tail_max_swaps": TAIL_MAX_SWAPS,
            "standalone_fullres_arm": False,
            "portfolio_arms": list(FUSION_ARM_NAMES),
            "target_free_summary": target_free_summary,
            "rows": rows,
        },
    )

    pair_paths = TaskaPairArtifactPaths()
    runtime_sources = {
        "confirmation_runner": Path(__file__).resolve(),
        "fusion_solver": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_selective_fullres_fusion.py"
        ),
        "selective_solver": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_selective_vote500.py"
        ),
        "fullres_supply_solver": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_fullres_union_voter.py"
        ),
        "focal_tail": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_gated_protected_tail.py"
        ),
        "pair_pipeline": PROJECT_ROOT / "src/aiijc_puzzle/taska_pair_pipeline.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        "matcher": PROJECT_ROOT / "src/aiijc_puzzle/taska_seam_matcher.py",
        "focal_verifier": PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_verifier.py",
        "edge_calibrator": PROJECT_ROOT / "src/aiijc_puzzle/taska_edge_calibrator.py",
        "layout_portfolio": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py"
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
            "all_candidate_layouts_edges_logits_and_dense_costs_frozen": True,
            "mechanical_selective_control_replay_count": len(rows),
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
    payload = json.loads(path.read_text(encoding="utf-8"))
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
        artifact = Path(str(record.get("path")))
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
            raise RuntimeError(f"pre-score artifact changed: {name}")
    return payload


def _truth_edges(layout: np.ndarray) -> frozenset[RawTailEdge]:
    board = strict_layout(layout, grid=GRID_SIZE).reshape(GRID_SIZE, GRID_SIZE)
    result: set[RawTailEdge] = set()
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE - 1):
            result.add(
                RawTailEdge(
                    int(board[row, column]),
                    int(board[row, column + 1]),
                    "right",
                )
            )
    for row in range(GRID_SIZE - 1):
        for column in range(GRID_SIZE):
            result.add(
                RawTailEdge(
                    int(board[row, column]),
                    int(board[row + 1, column]),
                    "down",
                )
            )
    if len(result) != PAIR_DENOMINATOR:
        raise RuntimeError("truth adjacency denominator changed")
    return frozenset(result)


def _layout_metrics(layout: Any, exact: np.ndarray) -> dict[str, Any]:
    strict = strict_layout(layout, grid=GRID_SIZE)
    metrics = evaluate_layout(strict, exact, reference_is_exact=True)
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
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("bootstrap values must be finite")
        grouped[source].append(float(value))
    if len(grouped) != SOURCE_COUNT or any(len(group) != 2 for group in grouped.values()):
        raise ValueError("bootstrap requires 16 sources with two draws each")
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
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    arm_means = {
        arm: {
            metric: float(np.mean([row[arm][metric] for row in rows]))
            for metric in metrics
        }
        for arm in ARMS
    }
    sources = [str(row["source_filename"]) for row in rows]
    delta = {
        metric: _cluster_ci(
            [
                float(row[CANDIDATE_ARM][metric])
                - float(row[CONTROL_ARM][metric])
                for row in rows
            ],
            sources,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, metric in enumerate(metrics)
    }
    supply_fields = (
        "current_edge_count",
        "current_true_edges",
        "selective_accepted_new_count",
        "selective_accepted_true_edges",
        "fullres_accepted_new_count",
        "fullres_accepted_true_edges",
        "unique_fullres_accepted_count",
        "unique_fullres_true_edges",
        "combined_union_edge_count",
        "combined_union_true_edges",
    )
    totals = {
        field: int(sum(row["candidate_supply"][field] for row in rows))
        for field in supply_fields
    }
    primary = delta["satisfied_adjacent_pairs"]
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
        "candidate_minus_control": delta,
        "candidate_supply": {
            "totals": totals,
            "means_per_board": {
                field: float(np.mean([row["candidate_supply"][field] for row in rows]))
                for field in supply_fields
            },
            "unique_fullres_precision": (
                totals["unique_fullres_true_edges"]
                / max(1, totals["unique_fullres_accepted_count"])
            ),
            "mean_unique_true_edges_added_per_board": (
                totals["unique_fullres_true_edges"] / CASE_COUNT
            ),
        },
        "confirmation_gate": gate,
        "fusion_choice_counts": dict(Counter(row["fusion_choice"] for row in rows)),
        "selective_choice_counts": dict(
            Counter(row["selective_choice"] for row in rows)
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
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != FROZEN_SCHEMA:
        raise RuntimeError("frozen metadata schema changed")
    if metadata.get("contains_exact_references_or_labels") is not False:
        raise RuntimeError("frozen metadata contains labels")
    frozen_rows = metadata.get("rows")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != CASE_COUNT:
        raise RuntimeError("frozen target-free row roster changed")
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
            exact = strict_layout(reference.tile_at_position, grid=GRID_SIZE)
            truth = _truth_edges(exact)
            current = _edges_from_archive(archive, prefix, "current")
            selective = _edges_from_archive(archive, prefix, "selective_accepted")
            fullres = _edges_from_archive(archive, prefix, "fullres_accepted")
            unique = _edges_from_archive(archive, prefix, "unique_fullres")
            combined = _edges_from_archive(archive, prefix, "combined_union")
            if set(current) & set(selective):
                raise RuntimeError("frozen selective supply overlaps current")
            if set(unique) != set(fullres) - set(current) - set(selective):
                raise RuntimeError("frozen unique fullres supply changed")
            if combined != current + selective + unique:
                raise RuntimeError("frozen combined union order changed")
            row: dict[str, Any] = {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source,
                "draw_index": draw,
                "fusion_choice": str(frozen["choice"]),
                "selective_choice": str(frozen["selective_replay_choice"]),
                "candidate_supply": {
                    "current_edge_count": len(current),
                    "current_true_edges": len(set(current) & truth),
                    "selective_accepted_new_count": len(selective),
                    "selective_accepted_true_edges": len(set(selective) & truth),
                    "fullres_accepted_new_count": len(fullres),
                    "fullres_accepted_true_edges": len(set(fullres) & truth),
                    "unique_fullres_accepted_count": len(unique),
                    "unique_fullres_true_edges": len(set(unique) & truth),
                    "combined_union_edge_count": len(combined),
                    "combined_union_true_edges": len(set(combined) & truth),
                },
            }
            for arm in ARMS:
                row[arm] = _layout_metrics(
                    archive[f"{prefix}__{arm}_layout"], exact
                )
            scored.append(row)
    return scored, _summarize(scored)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = _load_config(config_path)
    roster, manifest = _validate_preregistration(config_path, config)
    specs = [(manifest[name], name, draw) for name in roster for draw in DRAWS]
    if len(specs) != CASE_COUNT:
        raise RuntimeError("registered panel expansion changed")
    if args.inference_batch <= 0:
        raise ValueError("inference_batch must be positive")

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
                "event": "fusion_confirmation_all_target_free_evidence_frozen",
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
            "source_disjoint_under_preregistered_signed_exclusions": True,
            "one_panel_only": True,
        },
        "candidate": {
            "fixed_before_panel_scoring": True,
            "matcher_passes_per_case": 1,
            "matcher_vote_target": 500,
            "same_pass_current_vote_target": 350,
            "portfolio_arms": list(FUSION_ARM_NAMES),
            "standalone_fullres_arm": False,
            "restored_scorer_count": RESTORED_SCORER_COUNT,
            "restored_support_minimum": RESTORED_SUPPORT_MINIMUM,
            "new_edge_focal_logit_minimum": NEW_EDGE_FOCAL_LOGIT_MINIMUM,
            "tail_max_swaps": TAIL_MAX_SWAPS,
            "threshold_arm_budget_orientation_or_roster_sweep": False,
        },
        "target_free_summary": target_free_summary,
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
            "production_or_submission_modified": False,
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
