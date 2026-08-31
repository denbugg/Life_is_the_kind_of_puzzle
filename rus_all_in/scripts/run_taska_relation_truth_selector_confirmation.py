#!/usr/bin/env python3
"""Run the one signed source-disjoint relation-selector confirmation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_pair_pipeline import (
    EXPECTED_ARTIFACT_SHA256,
    GRID_SIZE,
    PAIR_DENOMINATOR,
    TaskaPairArtifactPaths,
)
from aiijc_puzzle.taska_relation_truth_selector import (
    FEATURE_NAMES,
    MODEL_PARAMETERS,
    PROVENANCE_NAMES,
    expected_correct_scores,
    relation_feature_board,
)
from aiijc_puzzle.taska_selective_fullres_fusion import (
    COMBINED_ARM,
    FUSION_ARM_NAMES,
    SELECTIVE_ARM,
    strict_layout,
)
from aiijc_puzzle.taska_six_arm_learned_selector import (
    prepare_six_arm_target_free_board,
)

try:
    from scripts import (
        run_taska_protected_tail_fresh32_confirmation as synthetic,
    )
    from scripts import (
        run_taska_selective_fullres_union_fusion_fresh32_confirmation as parent,
    )
except ModuleNotFoundError:
    import run_taska_protected_tail_fresh32_confirmation as synthetic
    import run_taska_selective_fullres_union_fusion_fresh32_confirmation as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/taska_relation_truth_selector_confirmation_v1.json"
)
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/taska-relation-truth-selector/formal-confirmation-v1"
)
EXCLUSION_PATH = (
    PROJECT_ROOT
    / "configs/taska_relation_truth_selector_confirmation_exclusions_v1.json"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs/taska-relation-truth-selector/fixed-v1/model-local32-held32/"
    "frozen-relation-classifier.pkl"
)
MODEL_SHA256 = "ec4eca99243cdc6be20104d789b9e5d5598b79fa0d1b7e69bc37314375ad8c6b"
FULLRES_CHECKPOINT = parent.FULLRES_CHECKPOINT

CONFIG_SCHEMA = "aiijc-taska-relation-truth-selector-confirmation-config-v1"
EXCLUSION_SCHEMA = "aiijc-taska-relation-truth-selector-confirmation-exclusions-v1"
FROZEN_SCHEMA = "aiijc-taska-relation-truth-selector-confirmation-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-relation-truth-selector-confirmation-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-relation-truth-selector-confirmation-report-v1"
SOURCE_MINIMUM = 6_400
SOURCE_MAXIMUM = 6_999
SOURCE_COUNT = 16
DRAWS = (0, 1)
CASE_COUNT = SOURCE_COUNT * len(DRAWS)
SELECTION_NAMESPACE = (
    "aiijc-taska-relation-truth-selector-formal-confirmation-v1-source16xdraw2"
)
SELECTION_SEED = 2_026_083_132
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_143
PAIR_GATE_MEAN = 1.0
PAIR_GATE_CI95_LOWER = 0.0
EXACT_GATE_MEAN = -1.0
CONTROL = "confirmed_six_arm_fusion"
CANDIDATE = "relation_truth_selector"


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


def _manifest(path: Path) -> tuple[dict[str, Mapping[str, Any]], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if compute_protocol_digest(payload) != payload.get("protocol_digest"):
        raise ValueError("organizer-train manifest protocol digest changed")
    rows = [row for values in payload["splits"].values() for row in values]
    lookup = {str(row["filename"]): row for row in rows}
    train = {str(row["filename"]) for row in payload["splits"]["train"]}
    if len(rows) != 7_000 or len(lookup) != 7_000 or len(train) != 5_600:
        raise ValueError("organizer-train manifest roster changed")
    return lookup, train


def _validate_preregistration(
    config_path: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, Mapping[str, Any]]]:
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("preregistered artifact registry is missing")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"malformed preregistered artifact: {name}")
        path = (PROJECT_ROOT / str(record.get("path"))).resolve()
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"preregistered artifact changed: {name}")
    if sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise ValueError("frozen relation classifier changed")
    exclusion = _load_signed_json(EXCLUSION_PATH, schema=EXCLUSION_SCHEMA)
    excluded = tuple(str(value) for value in exclusion["excluded_in_confirmation_universe"])
    exclusion_spec = exclusion["confirmation_universe"]
    if (
        excluded != tuple(sorted(set(excluded)))
        or len(excluded) != exclusion_spec["excluded_count"]
        or _digest(excluded) != exclusion_spec["excluded_digest"]
    ):
        raise ValueError("signed confirmation exclusion list changed")
    manifest_path = PROJECT_ROOT / str(artifacts["manifest"]["path"])
    lookup, train = _manifest(manifest_path)
    universe = tuple(
        sorted(
            name
            for name in train
            if SOURCE_MINIMUM <= int(name[4:10]) <= SOURCE_MAXIMUM
        )
    )
    eligible = tuple(name for name in universe if name not in set(excluded))
    prefix = f"{SELECTION_NAMESPACE}\0{SELECTION_SEED}\0".encode()
    roster = tuple(
        sorted(
            eligible,
            key=lambda name: (hashlib.sha256(prefix + name.encode()).digest(), name),
        )[:SOURCE_COUNT]
    )
    fixed_panel = {
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_seed": SELECTION_SEED,
        "universe_minimum": "img_006400.png",
        "universe_maximum": "img_006999.png",
        "organizer_train_universe_count": len(universe),
        "organizer_train_universe_digest": _digest(universe),
        "excluded_in_universe_count": len(excluded),
        "excluded_in_universe_digest": _digest(excluded),
        "eligible_count": len(eligible),
        "eligible_digest": _digest(eligible),
        "source_filenames": list(roster),
        "source_count": SOURCE_COUNT,
        "draws": list(DRAWS),
        "case_count": CASE_COUNT,
        "source_order_digest": _digest(roster),
        "cases_digest": _cases_digest(roster),
    }
    panel = config.get("panel")
    if not isinstance(panel, Mapping):
        raise ValueError("preregistered panel is missing")
    for key, value in fixed_panel.items():
        if panel.get(key) != value:
            raise ValueError(f"preregistered panel field changed: {key}")
    if set(roster) & set(excluded) or not set(roster) <= train:
        raise RuntimeError("confirmation roster is not disjoint organizer-train")
    candidate = config.get("candidate")
    expected_candidate = {
        "parent_inference": (
            "unchanged selective-target500 plus unique-fullres six-arm recipe"
        ),
        "post_tail_arms": list(FUSION_ARM_NAMES),
        "tail": "independent focal-logit>=0 protected non-adjacent tail96 per arm",
        "relation_model_sha256": MODEL_SHA256,
        "relation_feature_names": list(FEATURE_NAMES),
        "whole_layout_score": "sum positive-class probabilities over all 1104 seams",
        "selector": "maximum expected-correct score; exact tie retains confirmed control",
        "returns_one_whole_frozen_arm": True,
        "retraining": False,
        "threshold_or_parameter_sweep": False,
    }
    if candidate != expected_candidate:
        raise ValueError("fixed confirmation candidate changed")
    expected_evaluation = {
        "primary_metric": "candidate_minus_control_satisfied_pairs_per_board",
        "pair_denominator": PAIR_DENOMINATOR,
        "secondary_metrics": ["adjacency_recall", "exact_tiles_per_board"],
        "bootstrap_unit": "source_with_two_draws",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirmation_gate": {
            "pair_delta_mean_at_least": PAIR_GATE_MEAN,
            "pair_delta_ci95_lower_at_least": PAIR_GATE_CI95_LOWER,
            "exact_delta_mean_at_least": EXACT_GATE_MEAN,
            "strict_permutations_required": True,
        },
    }
    if config.get("evaluation") != expected_evaluation:
        raise ValueError("fixed confirmation evaluation changed")
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError("non-default confirmation config is not allowed")
    return roster, lookup


def _edges(case: Mapping[str, np.ndarray], name: str) -> tuple[RawTailEdge, ...]:
    sources = case[f"{name}__edge_source"]
    targets = case[f"{name}__edge_target"]
    axes = case[f"{name}__edge_axis"]
    result = tuple(
        RawTailEdge(int(source), int(target), "down" if int(axis) else "right")
        for source, target, axis in zip(sources, targets, axes, strict=True)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} frozen edge list contains duplicates")
    return result


def _relation_case(
    case: Mapping[str, np.ndarray], diagnostics: Mapping[str, Any], model: Any
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    current = _edges(case, "current")
    selective_new = _edges(case, "selective_accepted")
    unique_fullres = _edges(case, "unique_fullres")
    combined = _edges(case, "combined_union")
    selective = current + selective_new
    if combined != current + selective_new + unique_fullres:
        raise RuntimeError("confirmation combined supply order changed")
    current_logits = np.asarray(case["current_focal_logits"])
    selective_logits = np.concatenate(
        (current_logits, np.asarray(case["selective_accepted_focal_logits"]))
    )
    combined_logits = np.asarray(case["combined_union_focal_logits"])
    pre_tail = {
        **{arm: case[f"{arm}_layout"] for arm in FUSION_ARM_NAMES[:4]},
        SELECTIVE_ARM: case["selective_union_layout"],
        COMBINED_ARM: case["combined_union_layout"],
    }
    arm_edges = {
        arm: (
            selective
            if arm == SELECTIVE_ARM
            else combined
            if arm == COMBINED_ARM
            else current
        )
        for arm in FUSION_ARM_NAMES
    }
    arm_logits = {
        arm: (
            selective_logits
            if arm == SELECTIVE_ARM
            else combined_logits
            if arm == COMBINED_ARM
            else current_logits
        )
        for arm in FUSION_ARM_NAMES
    }
    control_choice = str(diagnostics["choice"])
    frozen_control = case[f"{parent.CANDIDATE_ARM}_layout"]
    board = prepare_six_arm_target_free_board(
        pre_tail_layouts=pre_tail,
        cost_right=case["cost_right"],
        cost_down=case["cost_down"],
        arm_edges=arm_edges,
        arm_logits=arm_logits,
        control_choice=control_choice,
        frozen_control_layout=frozen_control,
    )
    post_tail = dict(zip(FUSION_ARM_NAMES, board.layouts, strict=True))
    relations = relation_feature_board(
        post_tail_layouts=post_tail,
        pre_tail_layouts=pre_tail,
        cost_right=case["cost_right"],
        cost_down=case["cost_down"],
        arm_edges=arm_edges,
        arm_logits=arm_logits,
        provenance={
            PROVENANCE_NAMES[0]: current,
            PROVENANCE_NAMES[1]: selective_new,
            PROVENANCE_NAMES[2]: unique_fullres,
        },
        control_choice=control_choice,
    )
    scores = expected_correct_scores(relations, model)
    maximum = float(np.max(scores))
    tied = np.flatnonzero(scores == maximum)
    control_index = FUSION_ARM_NAMES.index(control_choice)
    choice_index = control_index if control_index in tied else int(tied[0])
    choice = FUSION_ARM_NAMES[choice_index]
    arrays = {
        "relation_features": relations.features.astype(np.float32),
        "relation_expected_correct_scores": scores.astype(np.float64),
        f"{CONTROL}_layout": board.control_layout,
        f"{CANDIDATE}_layout": relations.layouts[choice_index],
        **{
            f"relation_arm_{arm}_layout": layout
            for arm, layout in zip(FUSION_ARM_NAMES, relations.layouts, strict=True)
        },
    }
    return arrays, {
        "control_choice": control_choice,
        "choice": choice,
        "expected_correct_scores": dict(
            zip(FUSION_ARM_NAMES, scores.tolist(), strict=True)
        ),
        "changed_from_control": choice != control_choice,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
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
    resources = parent.load_taska_pair_pipeline_resources(device=device)
    denoiser = parent.load_fullres_denoiser(
        FULLRES_CHECKPOINT, device=resources.device
    )
    with MODEL_PATH.open("rb") as stream:
        model = pickle.load(stream)
    for key, value in MODEL_PARAMETERS.items():
        if model.get_params().get(key) != value:
            raise ValueError(f"frozen relation model parameter changed: {key}")
    cache = synthetic.CleanTileCache(targets.resolve(), maximum_boards=2)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    for index, (record, source, draw) in enumerate(specs):
        prefix = f"case_{index:03d}"
        dirty = synthetic._dirty_case(cache, record, source, draw)
        parent_arrays, parent_diagnostics = parent._target_free_case(
            dirty.dirty_tiles,
            resources=resources,
            denoiser=denoiser,
            inference_batch=inference_batch,
        )
        relation_arrays, relation_diagnostics = _relation_case(
            parent_arrays, parent_diagnostics, model
        )
        arrays.update(
            {
                f"{prefix}__{name}": value
                for name, value in {**parent_arrays, **relation_arrays}.items()
            }
        )
        rows.append(
            {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source,
                "draw_index": draw,
                "dirty_sha256": synthetic._dirty_sha256(dirty.dirty_tiles),
                "parent_diagnostics": parent_diagnostics,
                **relation_diagnostics,
            }
        )
        print(
            json.dumps(
                {
                    "event": "relation_selector_confirmation_target_free_case_ready",
                    "case": index + 1,
                    "case_count": len(specs),
                    "source_filename": source,
                    "draw_index": draw,
                    "control_choice": relation_diagnostics["control_choice"],
                    "choice": relation_diagnostics["choice"],
                    "changed": relation_diagnostics["changed_from_control"],
                }
            ),
            flush=True,
        )
    _write_npz(archive_path, arrays)
    summary = {
        "case_count": len(rows),
        "choice_counts": dict(Counter(row["choice"] for row in rows)),
        "control_choice_counts": dict(
            Counter(row["control_choice"] for row in rows)
        ),
        "changed_from_control_count": sum(row["changed_from_control"] for row in rows),
    }
    _write_json(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "contains_exact_references_or_labels": False,
            "all_six_post_tail_layouts_features_scores_and_choice_frozen": True,
            "all_layouts_strict_original_upright_tile_permutations": True,
            "relation_feature_names": list(FEATURE_NAMES),
            "relation_rows_per_arm": PAIR_DENOMINATOR,
            "relation_model_sha256": MODEL_SHA256,
            "target_free_summary": summary,
            "rows": rows,
        },
    )
    pair_paths = TaskaPairArtifactPaths()
    runtime_sources = {
        "confirmation_runner": Path(__file__).resolve(),
        "parent_confirmation_runner": Path(parent.__file__).resolve(),
        "relation_selector": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_relation_truth_selector.py"
        ),
        "six_arm_preparer": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_six_arm_learned_selector.py"
        ),
        "fusion_solver": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_selective_fullres_fusion.py"
        ),
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    }
    artifacts = {
        "preregistration": _record(config_path),
        "preregistration_sidecar": _record(Path(f"{config_path}.sha256")),
        "exclusion_snapshot": _record(EXCLUSION_PATH),
        "exclusion_snapshot_sidecar": _record(Path(f"{EXCLUSION_PATH}.sha256")),
        "relation_model": _record(MODEL_PATH),
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
    _write_json(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "all_six_layouts_features_expected_scores_and_choice_frozen": True,
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
        summary,
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
    if payload.get("verified_taska_artifact_sha256") != dict(
        EXPECTED_ARTIFACT_SHA256
    ):
        raise RuntimeError("pre-score TASKA artifact manifest changed")
    for name, record in payload["artifacts"].items():
        artifact = Path(str(record["path"]))
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"pre-score artifact changed: {name}")
    return payload


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
            0, len(source_means), size=(stop - start, len(source_means))
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
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in rows]
    arm_means = {
        arm: {
            metric: float(np.mean([row[arm][metric] for row in rows]))
            for metric in metrics
        }
        for arm in (CONTROL, CANDIDATE)
    }
    delta = {
        metric: _cluster_ci(
            [row[CANDIDATE][metric] - row[CONTROL][metric] for row in rows],
            sources,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, metric in enumerate(metrics)
    }
    pair = delta["satisfied_adjacent_pairs"]
    exact = delta["exact_tiles"]
    strict_count = sum(
        row[CONTROL]["strict_permutation"] and row[CANDIDATE]["strict_permutation"]
        for row in rows
    )
    gate = {
        "required_pair_delta_mean": PAIR_GATE_MEAN,
        "required_pair_delta_ci95_lower": PAIR_GATE_CI95_LOWER,
        "required_exact_delta_mean": EXACT_GATE_MEAN,
        "observed_pair_delta_mean": pair["mean"],
        "observed_pair_delta_ci95_lower": pair["ci95_lower"],
        "observed_exact_delta_mean": exact["mean"],
        "strict_pair_count": strict_count,
        "passed": bool(
            pair["mean"] >= PAIR_GATE_MEAN
            and pair["ci95_lower"] >= PAIR_GATE_CI95_LOWER
            and exact["mean"] >= EXACT_GATE_MEAN
            and strict_count == CASE_COUNT
        ),
    }
    return {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": arm_means,
        "candidate_minus_control": delta,
        "confirmation_gate": gate,
        "choice_counts": dict(Counter(row["choice"] for row in rows)),
        "control_choice_counts": dict(
            Counter(row["control_choice"] for row in rows)
        ),
        "changed_from_control_count": sum(row["changed_from_control"] for row in rows),
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
    frozen_rows = metadata["rows"]
    if len(frozen_rows) != CASE_COUNT:
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
                frozen["source_filename"] != source
                or int(frozen["draw_index"]) != draw
                or frozen["case_id"] != dirty.case_id
                or frozen["dirty_sha256"] != synthetic._dirty_sha256(dirty.tiles)
            ):
                raise RuntimeError("scoring recreated a different signed case")
            prefix = str(frozen["prefix"])
            exact = strict_layout(reference.tile_at_position, grid=GRID_SIZE)
            choice = str(frozen["choice"])
            selected = archive[f"{prefix}__{CANDIDATE}_layout"]
            arm = archive[f"{prefix}__relation_arm_{choice}_layout"]
            if not np.array_equal(selected, arm):
                raise RuntimeError("frozen selected layout is not its recorded whole arm")
            scored.append(
                {
                    "prefix": prefix,
                    "case_id": dirty.case_id,
                    "source_filename": source,
                    "draw_index": draw,
                    "control_choice": frozen["control_choice"],
                    "choice": choice,
                    "changed_from_control": bool(frozen["changed_from_control"]),
                    CONTROL: _layout_metrics(
                        archive[f"{prefix}__{CONTROL}_layout"], exact
                    ),
                    CANDIDATE: _layout_metrics(selected, exact),
                }
            )
    return scored, _summarize(scored)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = _load_signed_json(config_path, schema=CONFIG_SCHEMA)
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
            "exclusion_snapshot": _record(EXCLUSION_PATH),
            "source_filenames": list(roster),
            "source_count": SOURCE_COUNT,
            "case_count": CASE_COUNT,
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
                "event": "relation_confirmation_all_target_free_evidence_frozen",
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
            "source_disjoint_under_signed_exclusions": True,
            "one_panel_only": True,
        },
        "candidate": {
            "fixed_before_panel_generation_or_scoring": True,
            "relation_model_sha256": MODEL_SHA256,
            "all_six_post_tail_arms_scored": True,
            "returns_one_whole_frozen_arm": True,
            "retrained_or_swept": False,
        },
        "target_free_summary": target_free_summary,
        "frozen_eval": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
            "all_six_layouts_features_scores_and_choice_frozen_before_reference": True,
            "contains_exact_references_or_labels": False,
        },
        "metrics": metrics,
        "rows": rows,
        "runtime_seconds": {
            "target_free_matcher_denoiser_solver_selector": inference_seconds,
            "total": perf_counter() - started,
        },
        "legality": {
            "organizer_train_sources_only": True,
            "dirty_tiles_only_for_candidate_inference": True,
            "targets_or_exact_references_used_during_candidate_inference": False,
            "restored_pixels_matcher_only": True,
            "output_uses_each_original_upright_tile_once": True,
            "rotated_warped_replaced_or_constant_tiles": False,
            "competition_test_accessed": False,
            "postprocessing_used": False,
            "production_or_submission_modified": False,
        },
    }
    report_path = args.output_dir.resolve() / "report.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {"status": report["status"], "metrics": metrics, "report": _record(report_path)},
            indent=2,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    run(parse_args())
