#!/usr/bin/env python3
# ruff: noqa: E501
"""Evaluate one OOF weak-bridge rigid relocation over the frozen six-arm layout."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_context_bridge_relocator import (
    PROBABILITY_THRESHOLD,
    ContextBridgeRelocationResult,
    relocate_one_weak_bridge_subtree,
)
from aiijc_puzzle.taska_context_protector import (
    FEATURE_NAMES,
    ContextProtector,
    fit_context_protector,
)
from aiijc_puzzle.taska_pair_pipeline import PAIR_DENOMINATOR

try:
    from scripts import run_taska_confirmed_arm_portfolio as portfolio
    from scripts import run_taska_context_protector as context
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_selective_fullres_fusion as fusion
except ModuleNotFoundError:
    import run_taska_confirmed_arm_portfolio as portfolio
    import run_taska_context_protector as context
    import run_taska_focal_current_finetune as finetune
    import run_taska_selective_fullres_fusion as fusion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_context_bridge_relocator_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-context-bridge-relocator/fixed-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
CONTROL = context.CONTROL
CANDIDATE = "weak_bridge_rigid_relocator"
OOF_SPLITS = 8
LOCAL_PAIR_GATE = 0.0
LOCAL_EXACT_GATE = -1.0
HELD_PAIR_GATE = 0.25
HELD_EXACT_GATE = -1.0
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_135


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-one", action="store_true")
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed bridge-relocation preregistration is missing")
    digest = sha256_file(path)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("bridge-relocation config SHA mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "probability_threshold": PROBABILITY_THRESHOLD,
        "model": "StandardScaler plus unweighted LogisticRegression(C=1, lbfgs, max_iter=1000, random_state=0)",
        "oof_splits": OOF_SPLITS,
        "local_pair_gate": LOCAL_PAIR_GATE,
        "local_exact_gate": LOCAL_EXACT_GATE,
        "held_pair_gate": HELD_PAIR_GATE,
        "held_exact_gate": HELD_EXACT_GATE,
        "feature_names": list(FEATURE_NAMES),
        "no_sweep": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"preregistration mismatch: {key}")
    for relative, expected in config["fixed_source_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"signed source changed: {relative}")
    return config, digest


def _build_input_panel(name: str, output: Path, config: Path) -> tuple[Path, Path, Path, Path]:
    archive, metadata, context_freeze = context._target_free_panel(
        portfolio.PANELS[name], output, config
    )
    bridge_freeze = archive.parent / "pre-bridge-score-freeze.json"
    _write_json(
        bridge_freeze,
        {
            "schema": "aiijc-taska-context-bridge-relocator-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {
                key: _record(value)
                for key, value in {
                    "context_archive": archive,
                    "context_metadata": metadata,
                    "context_freeze": context_freeze,
                    "preregistration": config,
                    "runner": Path(__file__),
                    "relocator_module": PROJECT_ROOT
                    / "src/aiijc_puzzle/taska_context_bridge_relocator.py",
                    "context_module": PROJECT_ROOT / "src/aiijc_puzzle/taska_context_protector.py",
                }.items()
            },
        },
    )
    return archive, metadata, context_freeze, bridge_freeze


def _validate_bridge_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("bridge input was not frozen before target scoring")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("bridge input freeze contains labels")
    for record in payload["artifacts"].values():
        source = PROJECT_ROOT / record["path"]
        if not source.is_file() or sha256_file(source) != record["sha256"]:
            raise RuntimeError("a frozen bridge input changed")


def _metrics(layout: Any, reference: Any) -> dict[str, Any]:
    result = evaluate_layout(layout, reference, reference_is_exact=True)
    if result.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("pair denominator changed")
    return {
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
        "exact_tiles": int(result.correct_tile_count),
        "strict_original_upright_permutation": True,
    }


def _candidate(
    model: ContextProtector, archive: Any, row: Mapping[str, Any]
) -> ContextBridgeRelocationResult:
    prefix = str(row["prefix"])
    probabilities = model.predict_probability(
        np.asarray(archive[f"{prefix}__features"], dtype=np.float64)
    )
    return relocate_one_weak_bridge_subtree(
        archive[f"{prefix}__{CONTROL}_layout"],
        archive[f"{prefix}__cost_right"],
        archive[f"{prefix}__cost_down"],
        fusion._edges(archive, prefix, "context"),
        probabilities,
    )


def _ci(values: Sequence[float], sources: Sequence[str], seed: int) -> dict[str, Any]:
    by_source: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        by_source[str(source)].append(float(value))
    means = np.asarray([np.mean(group) for _, group in sorted(by_source.items())])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(BOOTSTRAP_RESAMPLES)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(BOOTSTRAP_RESAMPLES, start + 2048)
        indices = rng.integers(0, len(means), size=(stop - start, len(means)))
        bootstrap[start:stop] = means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(bootstrap, 0.025)),
        "ci95_upper": float(np.quantile(bootstrap, 0.975)),
        "case_wins_ties_losses": {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        },
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    result: dict[str, Any] = {
        "case_count": len(rows),
        "changed_count": sum(bool(row["bridge"]["changed"]) for row in rows),
        "mean_eligible_weak_subtrees": float(
            np.mean([row["bridge"]["eligible_weak_subtree_count"] for row in rows])
        ),
        "arms": {
            arm: {metric: float(np.mean([row[arm][metric] for row in rows])) for metric in metrics}
            for arm in ("control", "candidate")
        },
    }
    sources = [str(row["source_filename"]) for row in rows]
    result["candidate_minus_control"] = {
        metric: _ci(
            [row["candidate"][metric] - row["control"][metric] for row in rows],
            sources,
            BOOTSTRAP_SEED + index,
        )
        for index, metric in enumerate(metrics)
    }
    return result


def _oof_rows(
    archive_path: Path,
    metadata_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    scored: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], ContextProtector]:
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))["rows"]
    output: list[dict[str, Any]] = []
    with np.load(archive_path, allow_pickle=False) as archive:
        for fold, (train, valid) in enumerate(
            GroupKFold(n_splits=OOF_SPLITS).split(features, labels, sources)
        ):
            model = fit_context_protector(features[train], labels[train])
            valid_sources = set(sources[valid].tolist())
            for row, score in zip(rows, scored, strict=True):
                if row["source_filename"] not in valid_sources:
                    continue
                candidate = _candidate(model, archive, row)
                output.append(
                    {
                        "prefix": row["prefix"],
                        "source_filename": row["source_filename"],
                        "draw_index": row["draw_index"],
                        "choice": row["choice"],
                        "oof_fold": fold,
                        "bridge": candidate.diagnostics.__dict__,
                        "control": _metrics(
                            archive[f"{row['prefix']}__{CONTROL}_layout"], score["reference"]
                        ),
                        "candidate": _metrics(candidate.layout, score["reference"]),
                    }
                )
    if len(output) != len(rows):
        raise RuntimeError("OOF bridge candidate did not cover every local board")
    return output, fit_context_protector(features, labels)


def _frozen_rows(
    model: ContextProtector,
    archive_path: Path,
    metadata_path: Path,
    scored: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))["rows"]
    output: list[dict[str, Any]] = []
    with np.load(archive_path, allow_pickle=False) as archive:
        for row, score in zip(rows, scored, strict=True):
            candidate = _candidate(model, archive, row)
            output.append(
                {
                    "prefix": row["prefix"],
                    "source_filename": row["source_filename"],
                    "draw_index": row["draw_index"],
                    "choice": row["choice"],
                    "bridge": candidate.diagnostics.__dict__,
                    "control": _metrics(
                        archive[f"{row['prefix']}__{CONTROL}_layout"], score["reference"]
                    ),
                    "candidate": _metrics(candidate.layout, score["reference"]),
                }
            )
    return output


def _freeze_model(
    model: ContextProtector, output: Path, config: Path, local_archive: Path
) -> tuple[Path, Path]:
    stage = output / "model"
    stage.mkdir(parents=True, exist_ok=False)
    archive = stage / "frozen-context-logistic.npz"
    freeze = stage / "pre-held-freeze.json"
    _write_npz(
        archive,
        {
            "mean": model.scaler_mean,
            "scale": model.scaler_scale,
            "coefficient": model.coefficient,
            "intercept": np.asarray([model.intercept]),
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-context-bridge-relocator-pre-held-freeze-v1",
            "created_before_held_or_fresh_reference_reconstruction": True,
            "contains_local_labels": True,
            "contains_held_or_fresh_labels": False,
            "artifacts": {
                key: _record(value)
                for key, value in {
                    "model": archive,
                    "config": config,
                    "local_input": local_archive,
                }.items()
            },
        },
    )
    return archive, freeze


def _load_model(path: Path) -> ContextProtector:
    with np.load(path, allow_pickle=False) as archive:
        return ContextProtector(
            archive["mean"],
            archive["scale"],
            archive["coefficient"],
            float(archive["intercept"][0]),
        )


def _score_panel(
    name: str,
    output: Path,
    config: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    model: ContextProtector | None,
    local_oof: bool,
) -> tuple[dict[str, Any], ContextProtector | None, tuple[Path, Path, Path, Path]]:
    archive, metadata, context_freeze, bridge_freeze = _build_input_panel(name, output, config)
    _validate_bridge_freeze(bridge_freeze)
    labels, scored = context._load_labels(archive, metadata, context_freeze, lookup, cache)
    features, sources = context._feature_matrix(
        archive, json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    )
    if local_oof:
        rows, fitted = _oof_rows(archive, metadata, features, labels, sources, scored)
        return (
            {
                "status": "complete",
                "evaluation": "source-grouped-8-fold-OOF",
                "rows": rows,
                "summary": _summary(rows),
            },
            fitted,
            (archive, metadata, context_freeze, bridge_freeze),
        )
    if model is None:
        raise ValueError("frozen local32 model required")
    rows = _frozen_rows(model, archive, metadata, scored)
    return (
        {
            "status": "complete",
            "evaluation": "frozen-local32-model",
            "rows": rows,
            "summary": _summary(rows),
        },
        None,
        (archive, metadata, context_freeze, bridge_freeze),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    portfolio._require_inputs()
    config, config_sha = _load_config(args.config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.smoke_one:
        archive, metadata, context_freeze, bridge_freeze = _build_input_panel(
            "local32", output, args.config.resolve()
        )
        report = {
            "status": "target-free-smoke",
            "competition_test_accessed": False,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "context_freeze": _record(context_freeze),
                "bridge_freeze": _record(bridge_freeze),
            },
        }
        _write_json(output / "report.json", report)
        return report
    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    started = perf_counter()
    local, model, local_inputs = _score_panel(
        "local32", output, args.config.resolve(), lookup, cache, None, True
    )
    if model is None:
        raise RuntimeError("local OOF did not fit its frozen model")
    model_archive, model_freeze = _freeze_model(
        model, output, args.config.resolve(), local_inputs[0]
    )
    local_pair = local["summary"]["candidate_minus_control"]["satisfied_adjacent_pairs"]["mean"]
    local_exact = local["summary"]["candidate_minus_control"]["exact_tiles"]["mean"]
    held: dict[str, Any] = {"status": "skipped_by_local_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_gate"}
    if local_pair >= LOCAL_PAIR_GATE and local_exact >= LOCAL_EXACT_GATE:
        held, _, _ = _score_panel(
            "held32",
            output,
            args.config.resolve(),
            lookup,
            cache,
            _load_model(model_archive),
            False,
        )
        held_pair = held["summary"]["candidate_minus_control"]["satisfied_adjacent_pairs"]["mean"]
        held_exact = held["summary"]["candidate_minus_control"]["exact_tiles"]["mean"]
        if held_pair >= HELD_PAIR_GATE and held_exact >= HELD_EXACT_GATE:
            fresh, _, _ = _score_panel(
                "fresh32",
                output,
                args.config.resolve(),
                lookup,
                cache,
                _load_model(model_archive),
                False,
            )
        else:
            fresh = {"status": "skipped_by_held_gate"}
    report = {
        "schema": "aiijc-taska-context-bridge-relocator-report-v1",
        "status": "complete",
        "protocol": config,
        "preregistration_sha256": config_sha,
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "model": {"archive": _record(model_archive), "pre_held_freeze": _record(model_freeze)},
        "legality": {
            "strict_original_upright_tile_permutations": True,
            "pixels_modified": False,
            "competition_test_accessed": False,
            "production_modified": False,
            "target_free_bridge_inputs_frozen_before_labels": True,
            "held_fresh_model_frozen_from_local_only": True,
        },
    }
    _write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                key: report[key].get("summary", report[key])
                for key in ("local32", "held32", "fresh32")
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    run(parse_args())
