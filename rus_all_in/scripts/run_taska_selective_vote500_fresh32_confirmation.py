#!/usr/bin/env python3
"""Run the preregistered selective-target500 formal confirmation panel.

The unchanged :func:`solve_selective_vote500` implementation produces both
arms from one target500 matcher pass.  ``control`` is the same-pass current350
four-arm winner plus focal-gated tail96.  ``candidate`` adds only the fixed
focal-accepted target500-minus-current350 supply as a fifth arm, then applies
the same winner-aligned focal-gated tail96.  No tuning parameter is exposed.

Both strict upright layouts, every candidate edge/logit roster, and runtime
provenance are frozen before exact references are reconstructed.
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

from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_pair_pipeline import (
    EXPECTED_ARTIFACT_SHA256,
    GRID_SIZE,
    PAIR_DENOMINATOR,
    TaskaPairArtifactPaths,
    load_taska_pair_pipeline_resources,
)
from aiijc_puzzle.taska_selective_vote500 import (
    SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD,
    SELECTIVE_VOTE500_ARM,
    solve_selective_vote500,
)

try:
    from scripts import (
        run_taska_focal_gated_protected_tail_fresh16_confirmation as common,
    )
    from scripts import run_taska_protected_tail_fresh32_confirmation as synthetic
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_gated_protected_tail_fresh16_confirmation as common
    import run_taska_protected_tail_fresh32_confirmation as synthetic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/taska_selective_vote500_fresh32_confirmation_v1.json"
)
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/taska-selective-vote500/fresh32-formal-confirmation-v1"
)

CONFIG_SCHEMA = "aiijc-taska-selective-vote500-fresh32-confirmation-config-v1"
SNAPSHOT_SCHEMA = "aiijc-taska-selective-vote500-confirmation-exclusions-v1"
FROZEN_SCHEMA = "aiijc-taska-selective-vote500-confirmation-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-selective-vote500-confirmation-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-selective-vote500-confirmation-report-v1"

SOURCE_MINIMUM = 6_700
SOURCE_MAXIMUM = 6_999
SOURCE_COUNT = 16
DRAWS = (0, 1)
CASE_COUNT = SOURCE_COUNT * len(DRAWS)
SELECTION_NAMESPACE = (
    "aiijc-taska-selective-vote500-fresh32-formal-confirmation-v1-"
    "source16xdraw2"
)
SELECTION_SEED = 2_026_083_198
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 500_160_198
PAIR_GATE_MEAN = 2.0
PAIR_GATE_CI95_LOWER = 0.0
ARMS = (
    "samepass_current350_focal_gated_tail96",
    "selective_target500_focal_gated_tail96",
)
PRIMARY_METRIC = "satisfied_adjacent_pairs"
RAW_SOLVER_SHA256 = (
    "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
)
EXPECTED_CONFIG_ARTIFACTS = {
    "manifest": "data/interim/validation_manifest.json",
    "exclusion_snapshot": (
        "configs/taska_selective_vote500_fresh32_confirmation_v1.exclusions.json"
    ),
    "exclusion_snapshot_sidecar": (
        "configs/taska_selective_vote500_fresh32_confirmation_v1.exclusions.json.sha256"
    ),
    "tail192_reservation": (
        "configs/taska_focal_gated_tail192_fresh16_capacity_v1.json"
    ),
    "fullres_combo_confirmation": (
        "configs/taska_fullres_focal_gated_tail_fresh32_confirmation_v1.json"
    ),
    "selective_solver": "src/aiijc_puzzle/taska_selective_vote500.py",
    "raw_solver": "src/aiijc_puzzle/raw_tail_global_solver.py",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, str]:
    return {"path": _project_path(path), "sha256": sha256_file(path.resolve())}


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


def _require_config_artifact(
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


def _validate_preregistration(
    config_path: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, Mapping[str, Any]]]:
    paths = {
        name: _require_config_artifact(config, name, relative)
        for name, relative in EXPECTED_CONFIG_ARTIFACTS.items()
    }
    if sha256_file(paths["raw_solver"]) != RAW_SOLVER_SHA256:
        raise ValueError("raw solver no longer matches the frozen production SHA")
    snapshot = _load_signed_json(paths["exclusion_snapshot"], schema=SNAPSHOT_SCHEMA)
    union = snapshot.get("explicit_source_union")
    if not isinstance(union, Mapping):
        raise ValueError("exclusion union is absent")
    excluded = tuple(str(value) for value in union.get("source_filenames", ()))
    if (
        excluded != tuple(sorted(set(excluded)))
        or union.get("count") != len(excluded)
        or union.get("digest") != _digest(excluded)
    ):
        raise ValueError("frozen exclusion union changed")
    artifacts = snapshot.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("frozen TASKA artifact inventory is absent")
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise ValueError("malformed frozen TASKA artifact record")
        artifact = (PROJECT_ROOT / str(record.get("path"))).resolve()
        if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
            raise ValueError(f"frozen prior TASKA artifact changed: {artifact}")
    required = snapshot.get("required_signed_reservations")
    if not isinstance(required, Mapping):
        raise ValueError("required reservation provenance is absent")
    for name in ("tail192", "fullres_combo_confirmation"):
        record = required.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"reservation record is absent: {name}")
        reservation = (PROJECT_ROOT / str(record.get("path"))).resolve()
        if not reservation.is_file() or sha256_file(reservation) != record.get(
            "sha256"
        ):
            raise ValueError(f"reservation changed: {name}")

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
    fixed_panel = {
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_seed": SELECTION_SEED,
        "universe_minimum": "img_006700.png",
        "universe_maximum": "img_006999.png",
        "organizer_train_universe_count": len(universe),
        "organizer_train_universe_digest": _digest(universe),
        "exclusion_union_count": len(excluded),
        "exclusion_union_digest": _digest(excluded),
        "excluded_in_universe_count": len(set(universe) & excluded_set),
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
    for dependency in (paths["tail192_reservation"], paths["fullres_combo_confirmation"]):
        dependency_payload = json.loads(dependency.read_text(encoding="utf-8"))
        dependency_names = set(
            str(value)
            for value in dependency_payload.get("panel", {}).get(
                "source_filenames", ()
            )
        )
        if not dependency_names <= excluded_set or set(roster) & dependency_names:
            raise RuntimeError("reserved confirmation sources were not excluded")

    expected_candidate = {
        "entrypoint": "aiijc_puzzle.taska_selective_vote500.solve_selective_vote500",
        "matcher_passes_per_case": 1,
        "matcher_vote_target": 500,
        "same_pass_current_vote_target": 350,
        "new_edges": "target500 minus same-pass current350",
        "new_edge_acceptance": "recovered train_exact_top5 focal logit >= 0.0",
        "portfolio_arms": [
            "raw",
            "logistic",
            "focal_top5",
            "nonlinear",
            SELECTIVE_VOTE500_ARM,
        ],
        "selector": "minimum original TASKA all-1104-bond seam cost",
        "control": "same-pass current350 four-arm winner plus focal-gated tail96",
        "candidate_layout": "five-arm winner plus winner-aligned focal-gated tail96",
        "tail_max_swaps": 96,
        "focal_logit_threshold": SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD,
        "threshold_arm_or_budget_sweep": False,
    }
    if config.get("candidate") != expected_candidate:
        raise ValueError("fixed selective target500 candidate changed")
    expected_evaluation = {
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
    if config.get("evaluation") != expected_evaluation:
        raise ValueError("fixed confirmation evaluation changed")
    if sha256_file(config_path) != sha256_file(DEFAULT_CONFIG):
        raise ValueError("non-default preregistration bytes are not allowed")
    return roster, lookup


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
            [edge.axis == "down" for edge in edges], dtype=np.uint8
        ),
    }


def _frozen_edges(
    archive: Mapping[str, np.ndarray], prefix: str, name: str
) -> set[RawTailEdge]:
    sources = archive[f"{prefix}__{name}__edge_source"]
    targets = archive[f"{prefix}__{name}__edge_target"]
    axes = archive[f"{prefix}__{name}__edge_axis"]
    return {
        RawTailEdge(int(source), int(target), "down" if int(axis) else "right")
        for source, target, axis in zip(sources, targets, axes, strict=True)
    }


def _truth_edges(layout: np.ndarray) -> set[RawTailEdge]:
    board = common._strict_layout(layout).reshape(GRID_SIZE, GRID_SIZE)
    result: set[RawTailEdge] = set()
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE - 1):
            result.add(
                RawTailEdge(
                    int(board[row, column]), int(board[row, column + 1]), "right"
                )
            )
    for row in range(GRID_SIZE - 1):
        for column in range(GRID_SIZE):
            result.add(
                RawTailEdge(
                    int(board[row, column]), int(board[row + 1, column]), "down"
                )
            )
    if len(result) != PAIR_DENOMINATOR:
        raise RuntimeError("exact truth edge count changed")
    return result


def _freeze_target_free(
    *,
    config_path: Path,
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
        result = solve_selective_vote500(dirty.dirty_tiles, resources)
        arrays[f"{prefix}__{ARMS[0]}_layout"] = common._strict_layout(
            result.control_layout
        )
        arrays[f"{prefix}__{ARMS[1]}_layout"] = common._strict_layout(
            result.candidate_layout
        )
        supply = result.supply
        for name, edges, logits in (
            ("current", supply.current_edges, supply.current_logits),
            ("proposed_new", supply.proposed_new_edges, supply.proposed_new_logits),
            ("accepted_new", supply.accepted_new_edges, supply.accepted_new_logits),
            ("union", supply.union_edges, supply.union_logits),
        ):
            arrays.update(_edge_arrays(prefix, name, edges))
            arrays[f"{prefix}__{name}_focal_logits"] = np.asarray(
                logits, dtype=np.float32
            )
        rows.append(
            {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source,
                "draw_index": draw,
                "dirty_sha256": synthetic._dirty_sha256(dirty.dirty_tiles),
                **result.diagnostics(),
            }
        )
        print(
            json.dumps(
                {
                    "event": "selective500_confirmation_target_free_case_ready",
                    "case": index + 1,
                    "case_count": len(specs),
                    "source_filename": source,
                    "draw_index": draw,
                    "current_edges": len(supply.current_edges),
                    "proposed_new_edges": len(supply.proposed_new_edges),
                    "accepted_new_edges": len(supply.accepted_new_edges),
                    "candidate_choice": result.candidate_choice,
                }
            ),
            flush=True,
        )
    common._write_npz_exclusive(archive_path, arrays)
    common._write_json_exclusive(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "all_layouts_are_strict_original_upright_tile_permutations": True,
            "entrypoint": (
                "aiijc_puzzle.taska_selective_vote500.solve_selective_vote500"
            ),
            "one_target500_matcher_pass_per_case": True,
            "same_pass_target350_subset": True,
            "new_edge_focal_logit_threshold": SELECTIVE_NEW_EDGE_LOGIT_THRESHOLD,
            "arms": list(ARMS),
            "rows": rows,
        },
    )
    runtime_sources = {
        "confirmation_runner": Path(__file__).resolve(),
        "selective_solver": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_selective_vote500.py"
        ),
        "focal_gated_tail": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_gated_protected_tail.py"
        ),
        "pair_pipeline": PROJECT_ROOT / "src/aiijc_puzzle/taska_pair_pipeline.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        "matcher": PROJECT_ROOT / "src/aiijc_puzzle/taska_seam_matcher.py",
        "focal_verifier": PROJECT_ROOT / "src/aiijc_puzzle/taska_focal_verifier.py",
        "layout_portfolio": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py"
        ),
        "protected_tail": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_protected_tail_polish.py"
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
        "nonlinear_calibrator": _record(model_paths.nonlinear_calibrator),
    }
    common._write_json_exclusive(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "current_taska_lineage_disjoint_only": True,
            "universal_or_model_freshness_claimed": False,
            "device": str(device),
            "verified_production_artifact_sha256": dict(EXPECTED_ARTIFACT_SHA256),
            "artifacts": artifacts,
        },
    )
    return archive_path, metadata_path, freeze_path, perf_counter() - started


def _cluster_ci(
    values: Sequence[float], sources: Sequence[str], *, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("bootstrap values must be finite")
        grouped[str(source)].append(float(value))
    if len(grouped) != SOURCE_COUNT or any(len(values) != 2 for values in grouped.values()):
        raise ValueError("bootstrap requires 16 source clusters with two draws each")
    source_means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(
            0, SOURCE_COUNT, size=(stop - start, SOURCE_COUNT)
        )
        distribution[start:stop] = source_means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": SOURCE_COUNT,
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
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in rows]
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
    supply_fields = (
        "current_edges",
        "proposed_new_edges",
        "accepted_new_edges",
        "union_edges",
        "current_true_edges",
        "proposed_new_true_edges",
        "accepted_new_true_edges",
        "union_true_edges",
    )
    means = {
        field: float(np.mean([row["supply"][field] for row in rows]))
        for field in supply_fields
    }
    return {
        "case_count": len(rows),
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
        "candidate_choice_counts": dict(
            Counter(str(row["candidate_choice"]) for row in rows)
        ),
        "candidate_equals_control_count": sum(
            bool(row["candidate_equals_control"]) for row in rows
        ),
        "supply_mean_per_board": means,
        "supply": {
            "current_recall": means["current_true_edges"] / PAIR_DENOMINATOR,
            "union_recall": means["union_true_edges"] / PAIR_DENOMINATOR,
            "proposed_new_precision": (
                sum(row["supply"]["proposed_new_true_edges"] for row in rows)
                / max(1, sum(row["supply"]["proposed_new_edges"] for row in rows))
            ),
            "accepted_new_precision": (
                sum(row["supply"]["accepted_new_true_edges"] for row in rows)
                / max(1, sum(row["supply"]["accepted_new_edges"] for row in rows))
            ),
        },
    }


def _score_after_freeze(
    *,
    specs: Sequence[tuple[Mapping[str, Any], str, int]],
    targets: Path,
    archive_path: Path,
    metadata_path: Path,
    freeze_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    common._validate_freeze(freeze_path)
    frozen_rows = json.loads(metadata_path.read_text(encoding="utf-8"))["rows"]
    if len(frozen_rows) != CASE_COUNT:
        raise RuntimeError("frozen confirmation row roster changed")
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
                raise RuntimeError("scoring recreated a different frozen case")
            exact = common._strict_layout(reference.tile_at_position)
            truth = _truth_edges(exact)
            prefix = str(frozen["prefix"])
            current = _frozen_edges(archive, prefix, "current")
            proposed = _frozen_edges(archive, prefix, "proposed_new")
            accepted = _frozen_edges(archive, prefix, "accepted_new")
            union = _frozen_edges(archive, prefix, "union")
            if union != current | accepted or current & proposed or not accepted <= proposed:
                raise RuntimeError("frozen selective supply partition changed")
            control_layout = common._strict_layout(
                archive[f"{prefix}__{ARMS[0]}_layout"]
            )
            candidate_layout = common._strict_layout(
                archive[f"{prefix}__{ARMS[1]}_layout"]
            )
            scored.append(
                {
                    "prefix": prefix,
                    "case_id": dirty.case_id,
                    "source_filename": source,
                    "draw_index": draw,
                    "candidate_choice": str(frozen["candidate_choice"]),
                    "candidate_equals_control": bool(
                        np.array_equal(candidate_layout, control_layout)
                    ),
                    ARMS[0]: common._layout_metrics(control_layout, exact),
                    ARMS[1]: common._layout_metrics(candidate_layout, exact),
                    "supply": {
                        "current_edges": len(current),
                        "proposed_new_edges": len(proposed),
                        "accepted_new_edges": len(accepted),
                        "union_edges": len(union),
                        "current_true_edges": len(current & truth),
                        "proposed_new_true_edges": len(proposed & truth),
                        "accepted_new_true_edges": len(accepted & truth),
                        "union_true_edges": len(union & truth),
                    },
                }
            )
    return scored, _summarize(scored)


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = _load_config(config_path)
    roster, lookup = _validate_preregistration(config_path, config)
    specs = [(lookup[source], source, draw) for source in roster for draw in DRAWS]
    if len(specs) != CASE_COUNT:
        raise RuntimeError("confirmation panel expansion changed")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "event": "selective500_confirmation_preregistration_valid",
                    "config": _record(config_path),
                    "source_count": len(roster),
                    "case_count": len(specs),
                    "current_lineage_collision_count": 0,
                },
                indent=2,
            )
        )
        return
    device = synthetic._select_device(
        args.device,
        allow_nondeterministic_mps=bool(args.allow_nondeterministic_mps),
    )
    started = perf_counter()
    archive, metadata, freeze, inference_seconds = _freeze_target_free(
        config_path=config_path,
        specs=specs,
        targets=args.targets.resolve(),
        output_dir=args.output_dir.resolve(),
        device=device,
    )
    print(
        json.dumps(
            {
                "event": "selective500_layouts_edges_and_provenance_frozen",
                "case_count": CASE_COUNT,
                "archive_sha256": sha256_file(archive),
                "metadata_sha256": sha256_file(metadata),
                "pre_score_freeze_sha256": sha256_file(freeze),
                "reference_reconstructed_yet": False,
            }
        ),
        flush=True,
    )
    rows, summary = _score_after_freeze(
        specs=specs,
        targets=args.targets.resolve(),
        archive_path=archive,
        metadata_path=metadata,
        freeze_path=freeze,
    )
    passed = bool(summary["confirmation_gate"]["passed"])
    report = {
        "schema": REPORT_SCHEMA,
        "status": "confirmed" if passed else "not-confirmed",
        "panel": {
            "source_count": SOURCE_COUNT,
            "draws": list(DRAWS),
            "case_count": CASE_COUNT,
            "source_filenames": list(roster),
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_seed": SELECTION_SEED,
            "current_taska_lineage_disjoint_only": True,
            "universal_or_model_freshness_claimed": False,
        },
        "candidate": {
            **config["candidate"],
            "unchanged_selective_solver": True,
            "fixed_before_panel_scoring": True,
        },
        "frozen_eval": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
            "both_strict_layouts_frozen_before_exact_reference_reconstruction": True,
            "contains_exact_references_or_labels": False,
        },
        "summary": summary,
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
    common._write_json_exclusive(args.output_dir.resolve() / "report.json", report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
