#!/usr/bin/env python3
"""Fit one local32 unique-fullres edge calibrator, then gate held/fresh layouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_fullres_unique_edge_calibrator import (
    DECISION_THRESHOLD,
    FEATURE_NAMES,
    UniqueFullresEdgeCalibrator,
    fit_unique_fullres_edge_calibrator,
    unique_fullres_edge_features,
)
from aiijc_puzzle.taska_selective_fullres_fusion import (
    compose_selective_fullres_fusion,
    strict_layout,
)

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_selective_fullres_fusion as parent
except ModuleNotFoundError:
    import run_taska_focal_current_finetune as finetune
    import run_taska_selective_fullres_fusion as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_fullres_unique_edge_calibrator_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-fullres-unique-edge-calibrator/fixed-v1"
CONFIG_SHA256 = "079238829e55f3719592d6c061b040aedf72dc4d1c68f9454681195546532e26"
FUSION_ROOT = PROJECT_ROOT / "outputs/taska-selective-fullres-union-fusion/fixed-v1"
HELD_PAIR_GATE = 0.0
REPORT_SCHEMA = "aiijc-taska-fullres-unique-edge-calibrator-report-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=parent.DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(resolved),
    }


def _fusion_archive(panel_name: str) -> Path:
    return FUSION_ROOT / panel_name / "frozen-target-free-eval.npz"


def _fusion_metadata(panel_name: str) -> Path:
    return FUSION_ROOT / panel_name / "frozen-target-free-eval.json"


def _edge_support(
    fullres: Any,
    prefix: str,
    edges: Sequence[RawTailEdge],
) -> np.ndarray:
    proposed = parent._edges(fullres, prefix, "proposed")
    support = np.asarray(fullres[f"{prefix}__proposed_support"], dtype=np.uint8)
    if support.shape != (len(proposed),):
        raise RuntimeError("fullres proposal support alignment changed")
    lookup = dict(zip(proposed, support, strict=True))
    try:
        result = np.asarray([lookup[edge] for edge in edges], dtype=np.uint8)
    except KeyError as error:
        raise RuntimeError("unique fullres edge is absent from frozen proposals") from error
    if not np.isin(result, (3, 4)).all():
        raise RuntimeError("accepted unique fullres support is outside 3/4 or 4/4")
    return result


def _truth_for_row(
    *,
    row: Mapping[str, Any],
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> tuple[np.ndarray, frozenset[RawTailEdge]]:
    source = str(row["source_filename"])
    draw = int(row["draw_index"])
    dirty = finetune._dirty_case(cache, lookup[source], source, draw)
    if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
        raise RuntimeError("recreated dirty bytes do not match frozen evidence")
    reference = finetune._reference(cache, lookup[source], source, draw, dirty.dirty_tiles)
    return strict_layout(reference), parent._truth_edges(reference)


def _fit_local32(
    *,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> tuple[UniqueFullresEdgeCalibrator, dict[str, Any], np.ndarray, np.ndarray]:
    spec = parent.PANELS["local32"]
    rows = json.loads(_fusion_metadata("local32").read_text(encoding="utf-8"))["rows"]
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    source_chunks: list[np.ndarray] = []
    with (
        np.load(_fusion_archive("local32"), allow_pickle=False) as fusion,
        np.load(spec.fullres_archive, allow_pickle=False) as fullres,
        np.load(spec.base_archive, allow_pickle=False) as base,
    ):
        for row in rows:
            prefix = str(row["prefix"])
            edges = parent._edges(fusion, prefix, "unique_fullres")
            logits = np.asarray(fusion[f"{prefix}__unique_fullres_focal_logits"])
            support = _edge_support(fullres, prefix, edges)
            features = unique_fullres_edge_features(
                edges=edges,
                focal_logits=logits,
                restored_support=support,
                cost_right=parent._matrix(base, f"{prefix}__cost_right"),
                cost_down=parent._matrix(base, f"{prefix}__cost_down"),
            )
            _, truth = _truth_for_row(row=row, lookup=lookup, cache=cache)
            labels = np.asarray([edge in truth for edge in edges], dtype=np.uint8)
            feature_chunks.append(features)
            label_chunks.append(labels)
            source_chunks.append(np.repeat(str(row["source_filename"]), len(edges)))
    features = np.concatenate(feature_chunks)
    labels = np.concatenate(label_chunks)
    sources = np.concatenate(source_chunks)
    calibrator = fit_unique_fullres_edge_calibrator(features, labels)
    probabilities = calibrator.predict_probability(features)
    keep = probabilities >= DECISION_THRESHOLD
    diagnostics = {
        "panel": "local32",
        "source_count": len(set(sources.tolist())),
        "example_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "kept_count": int(keep.sum()),
        "kept_rate": float(keep.mean()),
        "kept_precision_in_sample": float(labels[keep].mean()) if keep.any() else None,
        "dropped_precision_in_sample": float(labels[~keep].mean()) if (~keep).any() else None,
        "training_accuracy_at_fixed_0_5": float(np.mean(keep == labels.astype(bool))),
        "feature_names": list(FEATURE_NAMES),
        "model": calibrator.diagnostics(),
    }
    return calibrator, diagnostics, features, labels


def _save_model(
    output_dir: Path,
    calibrator: UniqueFullresEdgeCalibrator,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[Path, Path]:
    model_path = output_dir / "calibrator.npz"
    fit_path = output_dir / "local32-fit-cache.npz"
    _write_npz(
        model_path,
        {
            "scaler_mean": calibrator.scaler_mean,
            "scaler_scale": calibrator.scaler_scale,
            "coefficient": calibrator.coefficient,
            "intercept": np.asarray([calibrator.intercept], dtype=np.float64),
            "decision_threshold": np.asarray([DECISION_THRESHOLD], dtype=np.float64),
            "feature_names": np.asarray(FEATURE_NAMES),
        },
    )
    _write_npz(fit_path, {"features": features, "labels": labels})
    return model_path, fit_path


def _edge_arrays(prefix: str, name: str, edges: Sequence[RawTailEdge]) -> dict[str, np.ndarray]:
    return parent._edge_arrays(prefix, name, edges)


def _freeze_panel(
    *,
    panel_name: str,
    calibrator: UniqueFullresEdgeCalibrator,
    output_dir: Path,
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    spec = parent.PANELS[panel_name]
    fusion_rows = json.loads(_fusion_metadata(panel_name).read_text(encoding="utf-8"))["rows"]
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    with (
        np.load(_fusion_archive(panel_name), allow_pickle=False) as fusion,
        np.load(spec.fullres_archive, allow_pickle=False) as fullres,
        np.load(spec.base_archive, allow_pickle=False) as base,
        np.load(spec.layout_archive, allow_pickle=False) as layouts,
    ):
        for index, row in enumerate(fusion_rows):
            prefix = str(row["prefix"])
            unique_edges = parent._edges(fusion, prefix, "unique_fullres")
            unique_logits = np.asarray(
                fusion[f"{prefix}__unique_fullres_focal_logits"], dtype=np.float32
            )
            support = _edge_support(fullres, prefix, unique_edges)
            cost_right = parent._matrix(base, f"{prefix}__cost_right")
            cost_down = parent._matrix(base, f"{prefix}__cost_down")
            features = unique_fullres_edge_features(
                edges=unique_edges,
                focal_logits=unique_logits,
                restored_support=support,
                cost_right=cost_right,
                cost_down=cost_down,
            )
            probabilities = calibrator.predict_probability(features)
            keep = probabilities >= DECISION_THRESHOLD
            filtered_edges = tuple(
                edge for edge, selected in zip(unique_edges, keep, strict=True) if bool(selected)
            )
            filtered_logits = np.ascontiguousarray(unique_logits[keep])
            current = parent._edges(fusion, prefix, "current")
            current_logits = np.asarray(fusion[f"{prefix}__current_focal_logits"])
            selective = parent._edges(fusion, prefix, "selective_new")
            selective_logits = np.asarray(fusion[f"{prefix}__selective_new_focal_logits"])
            selective_control = strict_layout(
                fusion[f"{prefix}__selective_target500_control_layout"]
            )
            result = compose_selective_fullres_fusion(
                cost_right=cost_right,
                cost_down=cost_down,
                four_layouts=parent._four_layouts(layouts, prefix),
                frozen_selective_control=selective_control,
                current_edges=current,
                current_logits=current_logits,
                selective_new_edges=selective,
                selective_new_logits=selective_logits,
                fullres_accepted_edges=filtered_edges,
                fullres_accepted_logits=filtered_logits,
            )
            control = strict_layout(fusion[f"{prefix}__combined_union_candidate_layout"])
            candidate = strict_layout(result.candidate_layout)
            arrays[f"{prefix}__confirmed_fusion_control_layout"] = control
            arrays[f"{prefix}__calibrated_fusion_candidate_layout"] = candidate
            arrays[f"{prefix}__unique_fullres_features"] = features.astype(np.float32)
            arrays[f"{prefix}__unique_fullres_probability"] = probabilities.astype(np.float32)
            arrays[f"{prefix}__unique_fullres_support"] = support
            arrays[f"{prefix}__unique_fullres_keep"] = keep.astype(np.uint8)
            arrays.update(_edge_arrays(prefix, "unique_fullres", unique_edges))
            arrays.update(_edge_arrays(prefix, "filtered_unique_fullres", filtered_edges))
            rows.append(
                {
                    "prefix": prefix,
                    "source_filename": str(row["source_filename"]),
                    "draw_index": int(row["draw_index"]),
                    "dirty_sha256": str(row["dirty_sha256"]),
                    "unique_fullres_count": len(unique_edges),
                    "filtered_unique_fullres_count": len(filtered_edges),
                    "candidate_choice": result.choice,
                    "strict_control": True,
                    "strict_candidate": True,
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{panel_name}_target_free",
                        "case": index + 1,
                        "unique": len(unique_edges),
                        "kept": len(filtered_edges),
                        "choice": result.choice,
                    }
                ),
                flush=True,
            )
    stage = output_dir / panel_name
    stage.mkdir(parents=True, exist_ok=False)
    archive = stage / "frozen-target-free-eval.npz"
    metadata = stage / "frozen-target-free-eval.json"
    freeze = stage / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-fullres-unique-edge-calibrator-target-free-v1",
            "panel": panel_name,
            "contains_exact_references_or_candidate_labels": False,
            "matcher_rerun": False,
            "decision_threshold": DECISION_THRESHOLD,
            "feature_names": list(FEATURE_NAMES),
            "control": "exact frozen confirmed unfiltered fusion final layout",
            "all_layouts_strict_original_upright_permutations": True,
            "rows": rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-fullres-unique-edge-calibrator-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "config": _record(DEFAULT_CONFIG),
                "calibrator": _record(output_dir / "calibrator.npz"),
                "parent_fusion_archive": _record(_fusion_archive(panel_name)),
                "runner": _record(Path(__file__).resolve()),
                "module": _record(
                    PROJECT_ROOT
                    / "src/aiijc_puzzle/taska_fullres_unique_edge_calibrator.py"
                ),
                "raw_solver": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
                ),
            },
        },
    )
    return archive, metadata, freeze, rows


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("candidate was not frozen before scoring")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("target-free freeze contains labels")
    for name, record in payload["artifacts"].items():
        artifact = PROJECT_ROOT / record["path"]
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"frozen artifact changed before scoring: {name}")


def _score_panel(
    *,
    panel_name: str,
    archive: Path,
    metadata: Path,
    freeze: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> dict[str, Any]:
    _validate_freeze(freeze)
    frozen_rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    scored: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as frozen:
        for row in frozen_rows:
            prefix = str(row["prefix"])
            reference, truth = _truth_for_row(row=row, lookup=lookup, cache=cache)
            control = parent._layout_metrics(
                frozen[f"{prefix}__confirmed_fusion_control_layout"], reference
            )
            candidate = parent._layout_metrics(
                frozen[f"{prefix}__calibrated_fusion_candidate_layout"], reference
            )
            unique = parent._edges(frozen, prefix, "unique_fullres")
            filtered = parent._edges(frozen, prefix, "filtered_unique_fullres")
            scored.append(
                {
                    **row,
                    "metrics": {
                        "confirmed_fusion_control": control,
                        "calibrated_fusion_candidate": candidate,
                    },
                    "supply": {
                        "unique_fullres_edges": len(unique),
                        "unique_fullres_true_edges": len(set(unique) & truth),
                        "filtered_unique_fullres_edges": len(filtered),
                        "filtered_unique_fullres_true_edges": len(set(filtered) & truth),
                    },
                }
            )
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in scored]
    summary: dict[str, Any] = {
        "case_count": len(scored),
        "pair_denominator": parent.PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row["metrics"][arm][metric] for row in scored]))
                for metric in metrics
            }
            for arm in ("confirmed_fusion_control", "calibrated_fusion_candidate")
        },
        "candidate_choice_counts": dict(Counter(row["candidate_choice"] for row in scored)),
    }
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["metrics"]["calibrated_fusion_candidate"][metric])
            - float(row["metrics"]["confirmed_fusion_control"][metric])
            for row in scored
        ]
        result = parent._cluster_ci(
            values,
            sources,
            seed=202_608_311_104 + index,
        )
        result["case_wins_ties_losses"] = {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        deltas[metric] = result
    summary["candidate_minus_control"] = deltas
    unique_total = sum(row["supply"]["unique_fullres_edges"] for row in scored)
    unique_true = sum(row["supply"]["unique_fullres_true_edges"] for row in scored)
    filtered_total = sum(row["supply"]["filtered_unique_fullres_edges"] for row in scored)
    filtered_true = sum(row["supply"]["filtered_unique_fullres_true_edges"] for row in scored)
    summary["supply"] = {
        "unique_edges": unique_total,
        "unique_true": unique_true,
        "unique_precision": unique_true / max(1, unique_total),
        "filtered_edges": filtered_total,
        "filtered_true": filtered_true,
        "filtered_precision": filtered_true / max(1, filtered_total),
        "true_edge_retention": filtered_true / max(1, unique_true),
    }
    return {"status": "complete", "rows": scored, "summary": summary}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = perf_counter()
    config_path = args.config.resolve()
    if config_path != DEFAULT_CONFIG.resolve() or sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("preregistered config path or SHA-256 changed")
    parent._require_inputs()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    calibrator, fit_summary, fit_features, fit_labels = _fit_local32(
        lookup=lookup,
        cache=cache,
    )
    model_path, fit_path = _save_model(
        output_dir,
        calibrator,
        fit_features,
        fit_labels,
    )
    held_archive, held_metadata, held_freeze, _ = _freeze_panel(
        panel_name="held32",
        calibrator=calibrator,
        output_dir=output_dir,
    )
    held = _score_panel(
        panel_name="held32",
        archive=held_archive,
        metadata=held_metadata,
        freeze=held_freeze,
        lookup=lookup,
        cache=cache,
    )
    held_delta = held["summary"]["candidate_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    fresh: dict[str, Any] = {"status": "skipped_by_negative_held_pair_gate"}
    if held_delta >= HELD_PAIR_GATE:
        fresh_archive, fresh_metadata, fresh_freeze, _ = _freeze_panel(
            panel_name="fresh32",
            calibrator=calibrator,
            output_dir=output_dir,
        )
        fresh = _score_panel(
            panel_name="fresh32",
            archive=fresh_archive,
            metadata=fresh_metadata,
            freeze=fresh_freeze,
            lookup=lookup,
            cache=cache,
        )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "protocol": json.loads(config_path.read_text(encoding="utf-8")),
        "fit_local32": fit_summary,
        "held32": held,
        "fresh32": fresh,
        "formal_confirmation": "not_opened_in_this_runner",
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "strict_original_upright_tile_permutations": True,
            "targets_absent_from_candidate_inference": True,
            "restored_pixels_matcher_only": True,
            "competition_test_accessed": False,
            "postprocessing_used": False,
            "production_or_submission_modified": False,
        },
        "artifacts": {
            "config": _record(config_path),
            "model": _record(model_path),
            "fit_cache": _record(fit_path),
            "runner": _record(Path(__file__).resolve()),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/taska_fullres_unique_edge_calibrator.py"
            ),
            "raw_solver": _record(PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps({"fit_local32": fit_summary, "held32": held, "fresh32": fresh}, indent=2))
    return report


if __name__ == "__main__":
    run(parse_args())
