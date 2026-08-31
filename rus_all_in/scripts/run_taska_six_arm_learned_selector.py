#!/usr/bin/env python3
"""Fit and stage one preregistered learned selector over six TASKA arms."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_selective_fullres_fusion import (
    COMBINED_ARM,
    FUSION_ARM_NAMES,
    SELECTIVE_ARM,
)
from aiijc_puzzle.taska_six_arm_learned_selector import (
    FEATURE_NAMES,
    RIDGE_ALPHA,
    FrozenPairwiseRidgeSelector,
    fit_pairwise_ridge_selector,
    prepare_six_arm_target_free_board,
)

try:
    from scripts import run_taska_confirmed_arm_portfolio as portfolio
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_selective_fullres_fusion as fusion
except ModuleNotFoundError:
    import run_taska_confirmed_arm_portfolio as portfolio
    import run_taska_focal_current_finetune as finetune
    import run_taska_selective_fullres_fusion as fusion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-six-arm-learned-selector/fixed-v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_six_arm_learned_selector_v1.json"
LOCAL_PAIR_GATE = 0.0
LOCAL_EXACT_GATE = -1.0
HELD_PAIR_GATE = 0.5
HELD_EXACT_GATE = -1.0
OOF_SPLITS = 8
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_116
CONTROL = "confirmed_six_arm_fusion"
CANDIDATE = "learned_six_arm_selector"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke-one", action="store_true")
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


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


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed learned-selector preregistration is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("learned-selector preregistration SHA-256 mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "roster": list(FUSION_ARM_NAMES),
        "model": "StandardScaler plus ordered-pair Ridge alpha=1 without intercept",
        "fit_panel": "local32 only",
        "oof_splits": OOF_SPLITS,
        "local_pair_gate": LOCAL_PAIR_GATE,
        "local_exact_gate": LOCAL_EXACT_GATE,
        "held_pair_gate": HELD_PAIR_GATE,
        "held_exact_gate": HELD_EXACT_GATE,
        "no_sweep": True,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"learned-selector preregistration mismatch: {key}")
    if config.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("learned-selector feature roster changed")
    for relative, expected in config["fixed_source_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"signed learned-selector source changed: {relative}")
    return config, digest


def _tail_inputs(
    archive: Any,
    prefix: str,
) -> tuple[dict[str, tuple[Any, ...]], dict[str, np.ndarray]]:
    current = fusion._edges(archive, prefix, "current")
    selective_new = fusion._edges(archive, prefix, "selective_new")
    combined = fusion._edges(archive, prefix, "combined_union")
    selective = current + selective_new
    current_logits = np.asarray(archive[f"{prefix}__current_focal_logits"])
    selective_logits = np.concatenate(
        (
            current_logits,
            np.asarray(archive[f"{prefix}__selective_new_focal_logits"]),
        )
    )
    combined_logits = np.asarray(archive[f"{prefix}__combined_union_focal_logits"])
    edges = {
        arm: current if arm not in {SELECTIVE_ARM, COMBINED_ARM} else (
            selective if arm == SELECTIVE_ARM else combined
        )
        for arm in FUSION_ARM_NAMES
    }
    logits = {
        arm: current_logits if arm not in {SELECTIVE_ARM, COMBINED_ARM} else (
            selective_logits if arm == SELECTIVE_ARM else combined_logits
        )
        for arm in FUSION_ARM_NAMES
    }
    return edges, logits


def _freeze_arm_panel(
    spec: portfolio.PanelSpec,
    *,
    output_dir: Path,
    config_path: Path,
) -> tuple[Path, Path, Path]:
    aligned = portfolio._aligned_rows(spec)
    stage = output_dir / spec.name
    stage.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    parent = spec.parent
    with (
        np.load(parent.layout_archive, allow_pickle=False) as layouts,
        np.load(parent.base_archive, allow_pickle=False) as base,
        np.load(spec.fusion_archive, allow_pickle=False) as fused,
    ):
        for index, records in enumerate(aligned):
            row = records[-1]
            prefix = str(row["prefix"])
            four = fusion._four_layouts(layouts, prefix)
            pre_tail = {
                **four,
                SELECTIVE_ARM: fused[f"{prefix}__selective_union_layout"],
                COMBINED_ARM: fused[f"{prefix}__combined_union_layout"],
            }
            edges, logits = _tail_inputs(fused, prefix)
            board = prepare_six_arm_target_free_board(
                pre_tail_layouts=pre_tail,
                cost_right=fusion._matrix(base, f"{prefix}__cost_right"),
                cost_down=fusion._matrix(base, f"{prefix}__cost_down"),
                arm_edges=edges,
                arm_logits=logits,
                control_choice=str(row["choice"]),
                frozen_control_layout=fused[
                    f"{prefix}__combined_union_candidate_layout"
                ],
            )
            arrays[f"{prefix}__features"] = board.features
            arrays[f"{prefix}__{CONTROL}_layout"] = board.control_layout
            for arm, layout in zip(FUSION_ARM_NAMES, board.layouts, strict=True):
                arrays[f"{prefix}__{arm}_layout"] = layout
            rows.append(
                {
                    "prefix": prefix,
                    "source_filename": str(row["source_filename"]),
                    "draw_index": int(row["draw_index"]),
                    "dirty_sha256": str(row["dirty_sha256"]),
                    "control_choice": board.control_choice,
                    "arm_diagnostics": list(board.diagnostics),
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{spec.name}_six_arm_features_target_free",
                        "case": index + 1,
                        "case_count": len(aligned),
                        "control_choice": board.control_choice,
                    }
                ),
                flush=True,
            )
    archive = stage / "frozen-target-free-arms.npz"
    metadata = stage / "frozen-target-free-arms.json"
    freeze = stage / "pre-arm-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-six-arm-learned-selector-arms-v1",
            "contains_exact_references_or_labels": False,
            "roster": list(FUSION_ARM_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "tail": "independent focal-logit-zero non-adjacent tail96 per arm",
            "rows": rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-six-arm-learned-selector-arm-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "preregistration": _record(config_path),
                "fusion_archive": _record(spec.fusion_archive),
                "fusion_metadata": _record(spec.fusion_metadata),
                "fusion_freeze": _record(spec.fusion_freeze),
                "runner": _record(Path(__file__).resolve()),
                "selector_module": _record(
                    PROJECT_ROOT
                    / "src/aiijc_puzzle/taska_six_arm_learned_selector.py"
                ),
            },
        },
    )
    return archive, metadata, freeze


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("pre-score timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains labels")
    for record in payload["artifacts"].values():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"pre-score artifact changed: {artifact}")


def _score_all_arms(
    archive: Path,
    metadata: Path,
    freeze: Path,
    *,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> list[dict[str, Any]]:
    _validate_freeze(freeze)
    rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    scored: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as frozen:
        for row in rows:
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                raise RuntimeError("scoring recreated different dirty bytes")
            reference = finetune._reference(
                cache, lookup[source], source, draw, dirty.dirty_tiles
            )
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "control_choice": row["control_choice"],
                    "control_metrics": portfolio._layout_metrics(
                        frozen[f"{prefix}__{CONTROL}_layout"], reference
                    ),
                    "arm_metrics": {
                        arm: portfolio._layout_metrics(
                            frozen[f"{prefix}__{arm}_layout"], reference
                        )
                        for arm in FUSION_ARM_NAMES
                    },
                }
            )
    return scored


def _features(archive: Path, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    with np.load(archive, allow_pickle=False) as frozen:
        return np.stack(
            [np.asarray(frozen[f"{row['prefix']}__features"]) for row in rows]
        )


def _pair_ci(values: Sequence[float], sources: Sequence[str], *, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("bootstrap values must be finite")
        grouped[source].append(float(value))
    means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, len(means), size=(stop - start, len(means)))
        distribution[start:stop] = means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(means),
        "case_count": len(values),
        "case_wins_ties_losses": {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        },
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in rows]
    result: dict[str, Any] = {
        "case_count": len(rows),
        "choice_counts": dict(Counter(row["choice"] for row in rows)),
        "arms": {
            name: {
                metric: float(np.mean([row[f"{name}_metrics"][metric] for row in rows]))
                for metric in metrics
            }
            for name in ("control", "candidate")
        },
    }
    result["candidate_minus_control"] = {
        metric: _pair_ci(
            [
                row["candidate_metrics"][metric] - row["control_metrics"][metric]
                for row in rows
            ],
            sources,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, metric in enumerate(metrics)
    }
    pair_oracle_choices = [
        max(
            FUSION_ARM_NAMES,
            key=lambda arm: row["arm_metrics"][arm]["satisfied_adjacent_pairs"],
        )
        for row in rows
    ]
    result["pair_oracle"] = {
        "satisfied_adjacent_pairs": float(
            np.mean(
                [
                    row["arm_metrics"][choice]["satisfied_adjacent_pairs"]
                    for row, choice in zip(rows, pair_oracle_choices, strict=True)
                ]
            )
        ),
        "exact_tiles_of_pair_oracle": float(
            np.mean(
                [
                    row["arm_metrics"][choice]["exact_tiles"]
                    for row, choice in zip(rows, pair_oracle_choices, strict=True)
                ]
            )
        ),
        "choice_counts": dict(Counter(pair_oracle_choices)),
    }
    result["pair_oracle_minus_control"] = (
        result["pair_oracle"]["satisfied_adjacent_pairs"]
        - result["arms"]["control"]["satisfied_adjacent_pairs"]
    )
    return result


def _freeze_model(
    model: FrozenPairwiseRidgeSelector,
    *,
    output_dir: Path,
    config_path: Path,
    local_archive: Path,
    local_metadata: Path,
) -> tuple[Path, Path, Path]:
    stage = output_dir / "model"
    stage.mkdir(parents=True, exist_ok=False)
    archive = stage / "frozen-selector.npz"
    metadata = stage / "frozen-selector.json"
    freeze = stage / "pre-held-freeze.json"
    _write_npz(
        archive,
        {
            "scaler_mean": model.scaler_mean,
            "scaler_scale": model.scaler_scale,
            "coefficients": model.coefficients,
        },
    )
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-six-arm-pairwise-ridge-model-v1",
            "trained_on": "local32 only",
            "feature_names": list(FEATURE_NAMES),
            "arm_names": list(FUSION_ARM_NAMES),
            "alpha": RIDGE_ALPHA,
            "fit_intercept": False,
            "ordered_pair_training": True,
            "scaler_mean": model.scaler_mean.tolist(),
            "scaler_scale": model.scaler_scale.tolist(),
            "coefficients": model.coefficients.tolist(),
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-six-arm-pairwise-ridge-pre-held-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": True,
            "contains_held_or_fresh_references_or_labels": False,
            "artifacts": {
                "model_archive": _record(archive),
                "model_metadata": _record(metadata),
                "local_arm_archive": _record(local_archive),
                "local_arm_metadata": _record(local_metadata),
                "preregistration": _record(config_path),
            },
        },
    )
    return archive, metadata, freeze


def _load_model(path: Path) -> FrozenPairwiseRidgeSelector:
    with np.load(path, allow_pickle=False) as frozen:
        return FrozenPairwiseRidgeSelector(
            scaler_mean=frozen["scaler_mean"],
            scaler_scale=frozen["scaler_scale"],
            coefficients=frozen["coefficients"],
        )


def _freeze_selections(
    *,
    stage: Path,
    arm_archive: Path,
    arm_metadata: Path,
    choices: Sequence[str],
    scores: Sequence[Sequence[float]],
    model_record: Mapping[str, str] | None,
    cross_fitted: bool,
) -> tuple[Path, Path, Path]:
    rows = json.loads(arm_metadata.read_text(encoding="utf-8"))["rows"]
    if len(rows) != len(choices) or len(rows) != len(scores):
        raise ValueError("selection rows do not align")
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    with np.load(arm_archive, allow_pickle=False) as arms:
        for row, choice, values in zip(rows, choices, scores, strict=True):
            prefix = str(row["prefix"])
            arrays[f"{prefix}__{CANDIDATE}_layout"] = arms[
                f"{prefix}__{choice}_layout"
            ]
            arrays[f"{prefix}__{CONTROL}_layout"] = arms[
                f"{prefix}__{CONTROL}_layout"
            ]
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": row["source_filename"],
                    "draw_index": row["draw_index"],
                    "dirty_sha256": row["dirty_sha256"],
                    "control_choice": row["control_choice"],
                    "choice": choice,
                    "scores": {
                        arm: float(value)
                        for arm, value in zip(FUSION_ARM_NAMES, values, strict=True)
                    },
                }
            )
    archive = stage / "frozen-selections.npz"
    metadata = stage / "frozen-selections.json"
    freeze = stage / "pre-selection-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-six-arm-learned-selections-v1",
            "cross_fitted": cross_fitted,
            "contains_current_row_exact_reference_or_label_in_prediction": False,
            "rows": frozen_rows,
        },
    )
    artifacts: dict[str, Mapping[str, str]] = {
        "selection_archive": _record(archive),
        "selection_metadata": _record(metadata),
        "arm_archive": _record(arm_archive),
        "arm_metadata": _record(arm_metadata),
    }
    if model_record is not None:
        artifacts["model"] = dict(model_record)
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-six-arm-learned-selection-freeze-v1",
            "created_before_exact_reference_reconstruction": not cross_fitted,
            "cross_fitted_after_fit_panel_arm_scoring": cross_fitted,
            "contains_evaluation_references_or_labels": cross_fitted,
            "contains_current_row_exact_reference_or_label_in_prediction": False,
            "artifacts": artifacts,
        },
    )
    return archive, metadata, freeze


def _selection_rows(
    scored: Sequence[Mapping[str, Any]],
    selection_metadata: Path,
    selection_archive: Path,
    arm_archive: Path,
) -> list[dict[str, Any]]:
    selected = json.loads(selection_metadata.read_text(encoding="utf-8"))["rows"]
    if len(scored) != len(selected):
        raise ValueError("scored and selected rows do not align")
    result: list[dict[str, Any]] = []
    with (
        np.load(selection_archive, allow_pickle=False) as candidates,
        np.load(arm_archive, allow_pickle=False) as arms,
    ):
        for labels, choice in zip(scored, selected, strict=True):
            if any(
                labels[field] != choice[field]
                for field in ("prefix", "source_filename", "draw_index")
            ):
                raise RuntimeError("selection identity mismatch")
            prefix = str(choice["prefix"])
            if not np.array_equal(
                candidates[f"{prefix}__{CANDIDATE}_layout"],
                arms[f"{prefix}__{choice['choice']}_layout"],
            ):
                raise RuntimeError("frozen selected layout changed")
            result.append(
                {
                    **labels,
                    "choice": choice["choice"],
                    "scores": choice["scores"],
                    "candidate_metrics": labels["arm_metrics"][choice["choice"]],
                }
            )
    return result


def _fit_oof(
    features: np.ndarray,
    scored: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[list[float]], list[int], FrozenPairwiseRidgeSelector]:
    labels = np.asarray(
        [
            [
                row["arm_metrics"][arm]["satisfied_adjacent_pairs"]
                for arm in FUSION_ARM_NAMES
            ]
            for row in scored
        ],
        dtype=np.float64,
    )
    groups = np.asarray([row["source_filename"] for row in scored])
    unique_groups = np.unique(groups)
    if len(unique_groups) < OOF_SPLITS:
        raise RuntimeError("fewer source groups than fixed OOF split count")
    choices = [""] * len(scored)
    scores: list[list[float]] = [[] for _ in scored]
    fold_ids = [-1] * len(scored)
    splitter = GroupKFold(n_splits=OOF_SPLITS)
    for fold, (train, valid) in enumerate(splitter.split(features, groups=groups)):
        model = fit_pairwise_ridge_selector(features[train], labels[train])
        for index in valid:
            values = model.scores(features[index])
            maximum = float(np.max(values))
            tied = np.flatnonzero(values == maximum)
            control_index = FUSION_ARM_NAMES.index(scored[index]["control_choice"])
            choice_index = control_index if control_index in tied else int(tied[0])
            choices[index] = FUSION_ARM_NAMES[choice_index]
            scores[index] = values.tolist()
            fold_ids[index] = fold
    if any(not choice for choice in choices) or any(fold < 0 for fold in fold_ids):
        raise RuntimeError("OOF predictions are incomplete")
    final_model = fit_pairwise_ridge_selector(features, labels)
    return choices, scores, fold_ids, final_model


def _apply_model(
    model: FrozenPairwiseRidgeSelector,
    archive: Path,
    metadata: Path,
) -> tuple[list[str], list[list[float]]]:
    rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    choices: list[str] = []
    scores: list[list[float]] = []
    with np.load(archive, allow_pickle=False) as frozen:
        for row in rows:
            prefix = str(row["prefix"])
            features = np.asarray(frozen[f"{prefix}__features"])
            values = model.scores(features)
            maximum = float(np.max(values))
            tied = np.flatnonzero(values == maximum)
            control_index = FUSION_ARM_NAMES.index(str(row["control_choice"]))
            choice_index = control_index if control_index in tied else int(tied[0])
            choices.append(FUSION_ARM_NAMES[choice_index])
            scores.append(values.tolist())
    return choices, scores


def _run_eval_panel(
    spec: portfolio.PanelSpec,
    *,
    output_dir: Path,
    config_path: Path,
    model: FrozenPairwiseRidgeSelector,
    model_archive: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> dict[str, Any]:
    started = perf_counter()
    arm_archive, arm_metadata, arm_freeze = _freeze_arm_panel(
        spec, output_dir=output_dir, config_path=config_path
    )
    choices, scores = _apply_model(model, arm_archive, arm_metadata)
    selection_archive, selection_metadata, selection_freeze = _freeze_selections(
        stage=output_dir / spec.name,
        arm_archive=arm_archive,
        arm_metadata=arm_metadata,
        choices=choices,
        scores=scores,
        model_record=_record(model_archive),
        cross_fitted=False,
    )
    _validate_freeze(arm_freeze)
    _validate_freeze(selection_freeze)
    scored = _score_all_arms(
        arm_archive,
        arm_metadata,
        arm_freeze,
        lookup=lookup,
        cache=cache,
    )
    rows = _selection_rows(
        scored, selection_metadata, selection_archive, arm_archive
    )
    return {
        "status": "complete",
        "rows": rows,
        "summary": _summarize(rows),
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "arm_archive": _record(arm_archive),
            "arm_metadata": _record(arm_metadata),
            "arm_freeze": _record(arm_freeze),
            "selection_archive": _record(selection_archive),
            "selection_metadata": _record(selection_metadata),
            "selection_freeze": _record(selection_freeze),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    portfolio._require_inputs()
    config, config_sha256 = _load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    local_spec = portfolio.PANELS["local32"]
    if args.smoke_one:
        from dataclasses import replace

        smoke = replace(local_spec, name="smoke1", case_count=1)
        archive, metadata, freeze = _freeze_arm_panel(
            smoke, output_dir=output_dir, config_path=args.config.resolve()
        )
        report = {
            "status": "target-free-smoke",
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "freeze": _record(freeze),
            },
            "competition_test_accessed": False,
        }
        _write_json(output_dir / "report.json", report)
        return report
    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)

    local_started = perf_counter()
    local_archive, local_metadata, local_freeze = _freeze_arm_panel(
        local_spec,
        output_dir=output_dir,
        config_path=args.config.resolve(),
    )
    local_scored = _score_all_arms(
        local_archive,
        local_metadata,
        local_freeze,
        lookup=lookup,
        cache=cache,
    )
    local_features = _features(local_archive, local_scored)
    choices, scores, folds, final_model = _fit_oof(local_features, local_scored)
    for row, fold in zip(local_scored, folds, strict=True):
        row["oof_fold"] = fold
    local_selection_archive, local_selection_metadata, local_selection_freeze = (
        _freeze_selections(
            stage=output_dir / "local32",
            arm_archive=local_archive,
            arm_metadata=local_metadata,
            choices=choices,
            scores=scores,
            model_record=None,
            cross_fitted=True,
        )
    )
    local_rows = _selection_rows(
        local_scored,
        local_selection_metadata,
        local_selection_archive,
        local_archive,
    )
    local = {
        "status": "complete",
        "evaluation": "source-grouped 8-fold out-of-fold",
        "rows": local_rows,
        "summary": _summarize(local_rows),
        "runtime_seconds": perf_counter() - local_started,
        "artifacts": {
            "arm_archive": _record(local_archive),
            "arm_metadata": _record(local_metadata),
            "arm_freeze": _record(local_freeze),
            "selection_archive": _record(local_selection_archive),
            "selection_metadata": _record(local_selection_metadata),
            "selection_freeze": _record(local_selection_freeze),
        },
    }
    model_archive, model_metadata, model_freeze = _freeze_model(
        final_model,
        output_dir=output_dir,
        config_path=args.config.resolve(),
        local_archive=local_archive,
        local_metadata=local_metadata,
    )
    frozen_model = _load_model(model_archive)
    local_pair = local["summary"]["candidate_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    local_exact = local["summary"]["candidate_minus_control"]["exact_tiles"][
        "mean"
    ]
    held: dict[str, Any] = {"status": "skipped_by_local_oof_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_gate"}
    if local_pair >= LOCAL_PAIR_GATE and local_exact >= LOCAL_EXACT_GATE:
        held = _run_eval_panel(
            portfolio.PANELS["held32"],
            output_dir=output_dir,
            config_path=args.config.resolve(),
            model=frozen_model,
            model_archive=model_archive,
            lookup=lookup,
            cache=cache,
        )
        held_pair = held["summary"]["candidate_minus_control"][
            "satisfied_adjacent_pairs"
        ]["mean"]
        held_exact = held["summary"]["candidate_minus_control"]["exact_tiles"][
            "mean"
        ]
        if held_pair >= HELD_PAIR_GATE and held_exact >= HELD_EXACT_GATE:
            fresh = _run_eval_panel(
                portfolio.PANELS["fresh32"],
                output_dir=output_dir,
                config_path=args.config.resolve(),
                model=frozen_model,
                model_archive=model_archive,
                lookup=lookup,
                cache=cache,
            )
        else:
            fresh = {"status": "skipped_by_held_pair_or_exact_gate"}
    report = {
        "schema": "aiijc-taska-six-arm-learned-selector-report-v1",
        "status": "complete",
        "protocol": config,
        "preregistration_sha256": config_sha256,
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "model": {
            "archive": _record(model_archive),
            "metadata": _record(model_metadata),
            "pre_held_freeze": _record(model_freeze),
        },
        "legality": {
            "strict_original_upright_tile_permutations": True,
            "pixels_changed_rotated_warped_replaced_or_postprocessed": False,
            "local_only_fit": True,
            "model_frozen_before_held": True,
            "competition_test_accessed": False,
            "production_modified": False,
        },
    }
    _write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                name: report[name].get("summary", report[name])
                for name in ("local32", "held32", "fresh32")
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    run(parse_args())
