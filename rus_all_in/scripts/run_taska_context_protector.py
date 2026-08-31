#!/usr/bin/env python3
# ruff: noqa: E501
"""Stage one frozen context-protected tail96 experiment over the six-arm control."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_context_protector import (
    DECISION_THRESHOLD,
    FEATURE_NAMES,
    ContextProtector,
    fit_context_protector,
    realised_context_features,
)
from aiijc_puzzle.taska_focal_gated_protected_tail import polish_taska_tail_with_focal_gate
from aiijc_puzzle.taska_pair_pipeline import ARM_NAMES, PAIR_DENOMINATOR
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail
from aiijc_puzzle.taska_selective_fullres_fusion import (
    COMBINED_ARM,
    SELECTIVE_ARM,
    strict_layout,
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_context_protector_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-context-protector/fixed-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
CONTROL = "confirmed_six_arm_fusion"
CANDIDATE = "context_protected_tail96"
OOF_SPLITS = 8
LOCAL_PAIR_GATE = 0.0
LOCAL_EXACT_GATE = -1.0
HELD_PAIR_GATE = 0.5
HELD_EXACT_GATE = -1.0
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_128


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _write_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not path.is_file()
        or not sidecar.is_file()
        or sidecar.read_text().split()[0] != sha256_file(path)
    ):
        raise ValueError("signed context-protector preregistration is missing or mismatched")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model": "StandardScaler plus unweighted LogisticRegression(C=1, lbfgs, max_iter=1000, random_state=0)",
        "decision_threshold": DECISION_THRESHOLD,
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
    return config, sha256_file(path)


def _tail_supply(archive: Any, prefix: str, choice: str) -> tuple[tuple[Any, ...], np.ndarray]:
    current = fusion._edges(archive, prefix, "current")
    current_logits = np.asarray(archive[f"{prefix}__current_focal_logits"], dtype=np.float64)
    selective_new = fusion._edges(archive, prefix, "selective_new")
    selective_logits = np.asarray(
        archive[f"{prefix}__selective_new_focal_logits"], dtype=np.float64
    )
    unique = fusion._edges(archive, prefix, "unique_fullres")
    unique_logits = np.asarray(archive[f"{prefix}__unique_fullres_focal_logits"], dtype=np.float64)
    if choice in ARM_NAMES:
        return current, current_logits
    if choice == SELECTIVE_ARM:
        return current + selective_new, np.concatenate((current_logits, selective_logits))
    if choice == COMBINED_ARM:
        return current + selective_new + unique, np.concatenate(
            (current_logits, selective_logits, unique_logits)
        )
    raise ValueError("choice is outside fixed six-arm roster")


def _target_free_panel(
    spec: portfolio.PanelSpec, output_dir: Path, config_path: Path
) -> tuple[Path, Path, Path]:
    stage = output_dir / spec.name
    stage.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    fusion_rows = json.loads(spec.fusion_metadata.read_text(encoding="utf-8"))["rows"]
    base_rows = fusion._aligned_rows(spec.parent)
    if len(fusion_rows) != len(base_rows):
        raise RuntimeError("frozen fusion panel length changed")
    with (
        np.load(spec.fusion_archive, allow_pickle=False) as fused,
        np.load(spec.parent.layout_archive, allow_pickle=False) as layouts,
        np.load(spec.parent.base_archive, allow_pickle=False) as base,
    ):
        for fusion_row, records in zip(fusion_rows, base_rows, strict=True):
            row = records[2]
            for field in ("prefix", "source_filename", "draw_index", "dirty_sha256"):
                if fusion_row[field] != row[field]:
                    raise RuntimeError("parent/fusion row identity changed")
            prefix, choice = str(fusion_row["prefix"]), str(fusion_row["choice"])
            pre_tail = {
                **fusion._four_layouts(layouts, prefix),
                SELECTIVE_ARM: strict_layout(fused[f"{prefix}__selective_union_layout"]),
                COMBINED_ARM: strict_layout(fused[f"{prefix}__combined_union_layout"]),
            }
            selected = pre_tail[choice]
            supply, logits = _tail_supply(fused, prefix, choice)
            provenance = {
                "current": fusion._edges(fused, prefix, "current"),
                "selective_new": fusion._edges(fused, prefix, "selective_new"),
                "unique_fullres": fusion._edges(fused, prefix, "unique_fullres"),
            }
            costs = (
                fusion._matrix(base, f"{prefix}__cost_right"),
                fusion._matrix(base, f"{prefix}__cost_down"),
            )
            # Freeze the conventional tail control mechanically before any label.
            replay = polish_taska_tail_with_focal_gate(selected, *costs, supply, logits).layout
            control = strict_layout(fused[f"{prefix}__combined_union_candidate_layout"])
            if not np.array_equal(replay, control):
                raise RuntimeError("confirmed six-arm focal-tail control did not replay")
            evidence = realised_context_features(
                selected_layout=selected,
                selected_arm=choice,
                selected_edges=supply,
                selected_logits=logits,
                provenance=provenance,
                pre_tail_layouts=pre_tail,
                cost_right=costs[0],
                cost_down=costs[1],
            )
            arrays[f"{prefix}__features"] = evidence.features
            arrays.update(fusion._edge_arrays(prefix, "context", evidence.edges))
            arrays[f"{prefix}__selected_pre_tail_layout"] = selected
            arrays[f"{prefix}__{CONTROL}_layout"] = control
            arrays[f"{prefix}__cost_right"] = costs[0]
            arrays[f"{prefix}__cost_down"] = costs[1]
            rows.append(
                {
                    "prefix": prefix,
                    "source_filename": fusion_row["source_filename"],
                    "draw_index": int(fusion_row["draw_index"]),
                    "dirty_sha256": fusion_row["dirty_sha256"],
                    "choice": choice,
                    "context_edge_count": len(evidence.edges),
                }
            )
    archive, metadata, freeze = (
        stage / "frozen-target-free-context.npz",
        stage / "frozen-target-free-context.json",
        stage / "pre-score-freeze.json",
    )
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-context-protector-target-free-v1",
            "contains_exact_references_or_labels": False,
            "rows": rows,
            "feature_names": list(FEATURE_NAMES),
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-context-protector-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {
                name: _record(value)
                for name, value in {
                    "archive": archive,
                    "metadata": metadata,
                    "config": config_path,
                    "runner": Path(__file__),
                    "module": PROJECT_ROOT / "src/aiijc_puzzle/taska_context_protector.py",
                    "fusion_archive": spec.fusion_archive,
                    "fusion_metadata": spec.fusion_metadata,
                }.items()
            },
        },
    )
    return archive, metadata, freeze


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("created_before_exact_reference_reconstruction") is not True
        or payload.get("contains_evaluation_references_or_labels") is not False
    ):
        raise RuntimeError("target-free pre-score freeze contract changed")
    for record in payload["artifacts"].values():
        source = PROJECT_ROOT / record["path"]
        if not source.is_file() or sha256_file(source) != record["sha256"]:
            raise RuntimeError("a frozen target-free artifact changed")


def _truth(reference: Any) -> frozenset[Any]:
    return fusion._truth_edges(reference)


def _load_labels(
    archive: Path, metadata: Path, freeze: Path, lookup: Mapping[str, Mapping[str, Any]], cache: Any
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    _validate_freeze(freeze)
    rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    labels: list[np.ndarray] = []
    scored: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as frozen:
        for row in rows:
            prefix, source, draw = (
                str(row["prefix"]),
                str(row["source_filename"]),
                int(row["draw_index"]),
            )
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                raise RuntimeError("dirty bytes no longer match target-free freeze")
            reference = finetune._reference(cache, lookup[source], source, draw, dirty.dirty_tiles)
            edges = fusion._edges(frozen, prefix, "context")
            truth = _truth(reference)
            labels.append(np.asarray([edge in truth for edge in edges], dtype=np.uint8))
            scored.append({**row, "reference": reference})
    return np.concatenate(labels), scored


def _feature_matrix(
    archive: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    chunks, groups = [], []
    with np.load(archive, allow_pickle=False) as frozen:
        for row in rows:
            values = np.asarray(frozen[f"{row['prefix']}__features"], dtype=np.float64)
            chunks.append(values)
            groups.append(np.repeat(str(row["source_filename"]), len(values)))
    return np.concatenate(chunks), np.concatenate(groups)


def _candidate_layout(
    model: ContextProtector, frozen: Any, row: Mapping[str, Any]
) -> tuple[np.ndarray, int]:
    prefix = str(row["prefix"])
    edges = fusion._edges(frozen, prefix, "context")
    keep = model.keep_mask(np.asarray(frozen[f"{prefix}__features"], dtype=np.float64))
    result = polish_unprotected_taska_tail(
        frozen[f"{prefix}__selected_pre_tail_layout"],
        frozen[f"{prefix}__cost_right"],
        frozen[f"{prefix}__cost_down"],
        tuple(edge for edge, selected in zip(edges, keep, strict=True) if selected),
        max_swaps=96,
        minimum_gain=1e-9,
    )
    return result.layout, int(keep.sum())


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


def _ci(values: Sequence[float], sources: Sequence[str], seed: int) -> dict[str, Any]:
    by_source: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        by_source[str(source)].append(float(value))
    means = np.asarray([np.mean(value) for _, value in sorted(by_source.items())])
    rng = np.random.default_rng(seed)
    values_boot = np.empty(BOOTSTRAP_RESAMPLES)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(BOOTSTRAP_RESAMPLES, start + 2048)
        values_boot[start:stop] = means[
            rng.integers(0, len(means), (stop - start, len(means)))
        ].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(values_boot, 0.025)),
        "ci95_upper": float(np.quantile(values_boot, 0.975)),
        "case_wins_ties_losses": {
            "wins": sum(v > 0 for v in values),
            "ties": sum(v == 0 for v in values),
            "losses": sum(v < 0 for v in values),
        },
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "case_count": len(rows),
        "choice_counts": dict(Counter(row["choice"] for row in rows)),
        "mean_classifier_kept_edges": float(np.mean([row["kept_edges"] for row in rows])),
    }
    output["arms"] = {
        name: {
            metric: float(np.mean([row[name][metric] for row in rows]))
            for metric in ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
        }
        for name in ("control", "candidate")
    }
    output["candidate_minus_control"] = {
        metric: _ci(
            [row["candidate"][metric] - row["control"][metric] for row in rows],
            [row["source_filename"] for row in rows],
            BOOTSTRAP_SEED + index,
        )
        for index, metric in enumerate(
            ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
        )
    }
    return output


def _evaluate_local_oof(
    archive: Path, metadata: Path, labels: np.ndarray, scored: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], ContextProtector, list[int]]:
    rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    features, groups = _feature_matrix(archive, rows)
    if labels.shape != (len(features),):
        raise RuntimeError("frozen labels/features misalign")
    result: list[dict[str, Any]] = []
    folds = [-1] * len(rows)
    with np.load(archive, allow_pickle=False) as frozen:
        splitter = GroupKFold(n_splits=OOF_SPLITS)
        for fold, (train, valid) in enumerate(splitter.split(features, labels, groups)):
            model = fit_context_protector(features[train], labels[train])
            validation_sources = set(groups[valid].tolist())
            for index, (row, score) in enumerate(zip(rows, scored, strict=True)):
                if row["source_filename"] not in validation_sources:
                    continue
                candidate, kept = _candidate_layout(model, frozen, row)
                result.append(
                    {
                        "source_filename": row["source_filename"],
                        "draw_index": row["draw_index"],
                        "prefix": row["prefix"],
                        "choice": row["choice"],
                        "oof_fold": fold,
                        "kept_edges": kept,
                        "control": _metrics(
                            frozen[f"{row['prefix']}__{CONTROL}_layout"], score["reference"]
                        ),
                        "candidate": _metrics(candidate, score["reference"]),
                    }
                )
                folds[index] = fold
    if len(result) != len(rows) or any(value < 0 for value in folds):
        raise RuntimeError("OOF split did not cover every board")
    return result, fit_context_protector(features, labels), folds


def _freeze_model(
    model: ContextProtector, output_dir: Path, config: Path, local_archive: Path
) -> tuple[Path, Path]:
    stage = output_dir / "model"
    stage.mkdir(parents=True, exist_ok=False)
    archive, freeze = stage / "frozen-model.npz", stage / "pre-held-freeze.json"
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
            "schema": "aiijc-taska-context-protector-pre-held-freeze-v1",
            "created_before_held_or_fresh_reference_reconstruction": True,
            "contains_local_labels": True,
            "contains_held_or_fresh_labels": False,
            "artifacts": {
                name: _record(value)
                for name, value in {
                    "model": archive,
                    "config": config,
                    "local_target_free": local_archive,
                }.items()
            },
        },
    )
    return archive, freeze


def _load_model(path: Path) -> ContextProtector:
    with np.load(path, allow_pickle=False) as source:
        return ContextProtector(
            source["mean"], source["scale"], source["coefficient"], float(source["intercept"][0])
        )


def _eval_frozen(
    model: ContextProtector,
    archive: Path,
    metadata: Path,
    labels: np.ndarray,
    scored: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    result = []
    with np.load(archive, allow_pickle=False) as frozen:
        for row, score in zip(rows, scored, strict=True):
            candidate, kept = _candidate_layout(model, frozen, row)
            result.append(
                {
                    "source_filename": row["source_filename"],
                    "draw_index": row["draw_index"],
                    "prefix": row["prefix"],
                    "choice": row["choice"],
                    "kept_edges": kept,
                    "control": _metrics(
                        frozen[f"{row['prefix']}__{CONTROL}_layout"], score["reference"]
                    ),
                    "candidate": _metrics(candidate, score["reference"]),
                }
            )
    return result


def _run_scored_panel(
    name: str,
    output: Path,
    config: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    model: ContextProtector | None,
    local: bool,
) -> tuple[dict[str, Any], ContextProtector | None, tuple[Path, Path, Path]]:
    archive, metadata, freeze = _target_free_panel(portfolio.PANELS[name], output, config)
    labels, scored = _load_labels(archive, metadata, freeze, lookup, cache)
    if local:
        rows, final_model, folds = _evaluate_local_oof(archive, metadata, labels, scored)
        return (
            {
                "status": "complete",
                "evaluation": "source-grouped-8-fold-OOF",
                "rows": rows,
                "summary": _summary(rows),
                "oof_folds": folds,
            },
            final_model,
            (archive, metadata, freeze),
        )
    if model is None:
        raise ValueError("frozen model is required outside local32")
    rows = _eval_frozen(model, archive, metadata, labels, scored)
    return (
        {
            "status": "complete",
            "evaluation": "frozen-local32-model",
            "rows": rows,
            "summary": _summary(rows),
        },
        None,
        (archive, metadata, freeze),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    portfolio._require_inputs()
    config, config_sha = _load_config(args.config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.smoke_one:
        from dataclasses import replace

        spec = replace(portfolio.PANELS["local32"], name="smoke1", case_count=1)
        archive, metadata, freeze = _target_free_panel(spec, output, args.config.resolve())
        report = {
            "status": "target-free-smoke",
            "competition_test_accessed": False,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "freeze": _record(freeze),
            },
        }
        _write_json(output / "report.json", report)
        return report
    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    started = perf_counter()
    local, model, local_artifacts = _run_scored_panel(
        "local32", output, args.config.resolve(), lookup, cache, None, True
    )
    if model is None:
        raise RuntimeError("local model did not fit")
    model_archive, model_freeze = _freeze_model(
        model, output, args.config.resolve(), local_artifacts[0]
    )
    local_pair = local["summary"]["candidate_minus_control"]["satisfied_adjacent_pairs"]["mean"]
    local_exact = local["summary"]["candidate_minus_control"]["exact_tiles"]["mean"]
    held: dict[str, Any] = {"status": "skipped_by_local_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_gate"}
    if local_pair >= LOCAL_PAIR_GATE and local_exact >= LOCAL_EXACT_GATE:
        held, _, _ = _run_scored_panel(
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
            fresh, _, _ = _run_scored_panel(
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
        "schema": "aiijc-taska-context-protector-report-v1",
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
            "target_free_rows_frozen_before_labels": True,
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
