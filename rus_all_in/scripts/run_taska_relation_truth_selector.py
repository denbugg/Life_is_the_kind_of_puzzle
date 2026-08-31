#!/usr/bin/env python3
"""Run one fixed relation-level nonlinear selector over six TASKA layouts."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_relation_truth_selector import (
    FEATURE_NAMES,
    MODEL_PARAMETERS,
    PROVENANCE_NAMES,
    RelationFeatureBoard,
    expected_correct_scores,
    fit_relation_truth_classifier,
    realised_edges,
    relation_feature_board,
)
from aiijc_puzzle.taska_selective_fullres_fusion import (
    COMBINED_ARM,
    FUSION_ARM_NAMES,
    SELECTIVE_ARM,
)
from aiijc_puzzle.taska_six_arm_learned_selector import (
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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-relation-truth-selector/fixed-v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_relation_truth_selector_v1.json"
CONTROL = "confirmed_six_arm_fusion"
CANDIDATE = "relation_truth_selector"
HELD_EXACT_GATE = -1.0
FRESH_PAIR_GATE = 0.5
FRESH_EXACT_GATE = -1.0
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_131


@dataclass(frozen=True)
class FrozenPanel:
    name: str
    boards: tuple[RelationFeatureBoard, ...]
    metadata_rows: tuple[Mapping[str, Any], ...]
    archive: Path
    metadata: Path
    freeze: Path


@dataclass(frozen=True)
class ScoredPanel:
    name: str
    frozen: FrozenPanel
    labels: tuple[np.ndarray, ...]
    metric_rows: tuple[Mapping[str, Any], ...]


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
        raise FileNotFoundError("signed relation-selector contract is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("relation-selector contract SHA-256 mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "roster": list(FUSION_ARM_NAMES),
        "model": "HistGradientBoostingClassifier",
        "model_parameters": MODEL_PARAMETERS,
        "feature_names": list(FEATURE_NAMES),
        "fit_sequence": [
            "fit local32; diagnose in-sample and held32 source-disjoint",
            "if held signal: fit local32+held32; evaluate fresh32 development",
        ],
        "held_signal": {
            "pair_delta": "strictly positive",
            "exact_delta_minimum": HELD_EXACT_GATE,
        },
        "fresh_pair_gate": FRESH_PAIR_GATE,
        "fresh_exact_gate": FRESH_EXACT_GATE,
        "no_parameter_sweep": True,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"relation-selector contract mismatch: {key}")
    for relative, expected in config["fixed_source_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"signed relation-selector source changed: {relative}")
    return config, digest


def _tail_inputs(
    archive: Any, prefix: str
) -> tuple[dict[str, tuple[Any, ...]], dict[str, np.ndarray]]:
    current = fusion._edges(archive, prefix, "current")
    selective_new = fusion._edges(archive, prefix, "selective_new")
    combined = fusion._edges(archive, prefix, "combined_union")
    selective = current + selective_new
    current_logits = np.asarray(archive[f"{prefix}__current_focal_logits"])
    selective_logits = np.concatenate(
        (current_logits, np.asarray(archive[f"{prefix}__selective_new_focal_logits"]))
    )
    combined_logits = np.asarray(archive[f"{prefix}__combined_union_focal_logits"])
    edges = {
        arm: (
            selective
            if arm == SELECTIVE_ARM
            else combined
            if arm == COMBINED_ARM
            else current
        )
        for arm in FUSION_ARM_NAMES
    }
    logits = {
        arm: (
            selective_logits
            if arm == SELECTIVE_ARM
            else combined_logits
            if arm == COMBINED_ARM
            else current_logits
        )
        for arm in FUSION_ARM_NAMES
    }
    return edges, logits


def _provenance(archive: Any, prefix: str) -> dict[str, tuple[Any, ...]]:
    return {
        name: fusion._edges(archive, prefix, name) for name in PROVENANCE_NAMES
    }


def _freeze_panel(
    spec: portfolio.PanelSpec,
    *,
    output_dir: Path,
    config_path: Path,
) -> FrozenPanel:
    stage = output_dir / spec.name
    stage.mkdir(parents=True, exist_ok=False)
    aligned = portfolio._aligned_rows(spec)
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    with (
        np.load(spec.parent.layout_archive, allow_pickle=False) as layouts,
        np.load(spec.parent.base_archive, allow_pickle=False) as base,
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
            cost_right = fusion._matrix(base, f"{prefix}__cost_right")
            cost_down = fusion._matrix(base, f"{prefix}__cost_down")
            six_arm = prepare_six_arm_target_free_board(
                pre_tail_layouts=pre_tail,
                cost_right=cost_right,
                cost_down=cost_down,
                arm_edges=edges,
                arm_logits=logits,
                control_choice=str(row["choice"]),
                frozen_control_layout=fused[
                    f"{prefix}__combined_union_candidate_layout"
                ],
            )
            post_tail = dict(zip(FUSION_ARM_NAMES, six_arm.layouts, strict=True))
            relation = relation_feature_board(
                post_tail_layouts=post_tail,
                pre_tail_layouts=pre_tail,
                cost_right=cost_right,
                cost_down=cost_down,
                arm_edges=edges,
                arm_logits=logits,
                provenance=_provenance(fused, prefix),
                control_choice=six_arm.control_choice,
            )
            arrays[f"{prefix}__features"] = relation.features.astype(np.float32)
            for arm, layout in zip(FUSION_ARM_NAMES, relation.layouts, strict=True):
                arrays[f"{prefix}__{arm}_layout"] = layout
            rows.append(
                {
                    "prefix": prefix,
                    "source_filename": str(row["source_filename"]),
                    "draw_index": int(row["draw_index"]),
                    "dirty_sha256": str(row["dirty_sha256"]),
                    "control_choice": relation.control_choice,
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{spec.name}_relation_features_target_free",
                        "case": index + 1,
                        "case_count": len(aligned),
                        "runtime_seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    archive = stage / "frozen-target-free-relations.npz"
    metadata = stage / "frozen-target-free-relations.json"
    freeze = stage / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-relation-truth-target-free-v1",
            "panel": spec.name,
            "contains_exact_references_or_relation_labels": False,
            "arm_names": list(FUSION_ARM_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "rows_per_arm": 1104,
            "rows": rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-relation-truth-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "preregistration": _record(config_path),
                "fusion_archive": _record(spec.fusion_archive),
                "fusion_metadata": _record(spec.fusion_metadata),
            },
        },
    )
    return _load_frozen_panel(spec.name, archive, metadata, freeze)


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("target-free relation freeze timing changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("target-free relation freeze contains labels")
    for record in payload["artifacts"].values():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"frozen artifact changed: {artifact}")


def _load_frozen_panel(
    name: str, archive: Path, metadata: Path, freeze: Path
) -> FrozenPanel:
    _validate_freeze(freeze)
    rows = tuple(json.loads(metadata.read_text(encoding="utf-8"))["rows"])
    boards: list[RelationFeatureBoard] = []
    with np.load(archive, allow_pickle=False) as frozen:
        for row in rows:
            prefix = str(row["prefix"])
            layouts = tuple(
                np.asarray(frozen[f"{prefix}__{arm}_layout"]) for arm in FUSION_ARM_NAMES
            )
            boards.append(
                RelationFeatureBoard(
                    layouts=layouts,
                    edges=tuple(realised_edges(layout) for layout in layouts),
                    features=np.asarray(frozen[f"{prefix}__features"], dtype=np.float64),
                    control_choice=str(row["control_choice"]),
                )
            )
    return FrozenPanel(name, tuple(boards), rows, archive, metadata, freeze)


def _score_panel(
    frozen: FrozenPanel,
    *,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> ScoredPanel:
    _validate_freeze(frozen.freeze)
    labels: list[np.ndarray] = []
    metric_rows: list[dict[str, Any]] = []
    for board, row in zip(frozen.boards, frozen.metadata_rows, strict=True):
        source = str(row["source_filename"])
        draw = int(row["draw_index"])
        dirty = finetune._dirty_case(cache, lookup[source], source, draw)
        if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
            raise RuntimeError("scoring recreated different dirty bytes")
        reference = finetune._reference(
            cache, lookup[source], source, draw, dirty.dirty_tiles
        )
        truth = fusion._truth_edges(reference)
        current_labels = board.labels(truth)
        for arm_index, _arm in enumerate(FUSION_ARM_NAMES):
            metric = portfolio._layout_metrics(board.layouts[arm_index], reference)
            if metric["satisfied_adjacent_pairs"] != int(current_labels[arm_index].sum()):
                raise RuntimeError("relation labels and layout metric disagree")
        labels.append(current_labels)
        metric_rows.append(
            {
                **row,
                "arm_metrics": {
                    arm: portfolio._layout_metrics(board.layouts[index], reference)
                    for index, arm in enumerate(FUSION_ARM_NAMES)
                },
            }
        )
    return ScoredPanel(frozen.name, frozen, tuple(labels), tuple(metric_rows))


def _cluster_ci(
    values: Sequence[float], sources: Sequence[str], *, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("bootstrap value is non-finite")
        grouped[source].append(float(value))
    means = np.asarray([np.mean(grouped[source]) for source in sorted(grouped)])
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


def _edge_diagnostics(scored: ScoredPanel, model: Any) -> dict[str, float]:
    labels = np.concatenate([value.reshape(-1) for value in scored.labels])
    probabilities = np.concatenate(
        [
            model.predict_proba(board.features.reshape(-1, len(FEATURE_NAMES)))[:, 1]
            for board in scored.frozen.boards
        ]
    )
    return {
        "row_count": int(len(labels)),
        "positive_fraction": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_loss": float(brier_score_loss(labels, probabilities)),
    }


def _selection_summary(scored: ScoredPanel, model: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for board, metric_row in zip(
        scored.frozen.boards, scored.metric_rows, strict=True
    ):
        scores = expected_correct_scores(board, model)
        maximum = float(np.max(scores))
        tied = np.flatnonzero(scores == maximum)
        control_index = FUSION_ARM_NAMES.index(board.control_choice)
        choice_index = control_index if control_index in tied else int(tied[0])
        choice = FUSION_ARM_NAMES[choice_index]
        pair_oracle = max(
            FUSION_ARM_NAMES,
            key=lambda arm: metric_row["arm_metrics"][arm][
                "satisfied_adjacent_pairs"
            ],
        )
        rows.append(
            {
                "source_filename": metric_row["source_filename"],
                "draw_index": metric_row["draw_index"],
                "control_choice": board.control_choice,
                "choice": choice,
                "pair_oracle_choice": pair_oracle,
                "scores": dict(zip(FUSION_ARM_NAMES, scores.tolist(), strict=True)),
                "control_metrics": metric_row["arm_metrics"][board.control_choice],
                "candidate_metrics": metric_row["arm_metrics"][choice],
                "pair_oracle_metrics": metric_row["arm_metrics"][pair_oracle],
            }
        )
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in rows]
    result: dict[str, Any] = {
        "case_count": len(rows),
        "choice_counts": dict(Counter(row["choice"] for row in rows)),
        "control_choice_counts": dict(Counter(row["control_choice"] for row in rows)),
        "changed_from_control_count": sum(
            row["choice"] != row["control_choice"] for row in rows
        ),
        "pair_oracle_choice_counts": dict(
            Counter(row["pair_oracle_choice"] for row in rows)
        ),
        "arms": {
            name: {
                metric: float(np.mean([row[f"{name}_metrics"][metric] for row in rows]))
                for metric in metrics
            }
            for name in ("control", "candidate", "pair_oracle")
        },
    }
    result["candidate_minus_control"] = {
        metric: _cluster_ci(
            [
                row["candidate_metrics"][metric] - row["control_metrics"][metric]
                for row in rows
            ],
            sources,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, metric in enumerate(metrics)
    }
    result["pair_oracle_minus_control"] = {
        metric: float(
            result["arms"]["pair_oracle"][metric]
            - result["arms"]["control"][metric]
        )
        for metric in metrics
    }
    result["rows"] = rows
    return result


def _freeze_model(
    model: Any,
    *,
    stage: Path,
    trained_on: Sequence[ScoredPanel],
    config_path: Path,
) -> tuple[Path, Path]:
    stage.mkdir(parents=True, exist_ok=False)
    model_path = stage / "frozen-relation-classifier.pkl"
    try:
        with model_path.open("xb") as stream:
            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {model_path}") from error
    freeze = stage / "pre-evaluation-freeze.json"
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-relation-truth-model-freeze-v1",
            "created_before_next_panel_target_free_generation_or_scoring": True,
            "trained_on": [panel.name for panel in trained_on],
            "contains_next_panel_references_or_labels": False,
            "artifacts": {
                "model": _record(model_path),
                "preregistration": _record(config_path),
                **{
                    f"{panel.name}_target_free_archive": _record(panel.frozen.archive)
                    for panel in trained_on
                },
            },
        },
    )
    return model_path, freeze


def _panel_artifacts(panel: FrozenPanel) -> dict[str, Any]:
    return {
        "archive": _record(panel.archive),
        "metadata": _record(panel.metadata),
        "pre_score_freeze": _record(panel.freeze),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    portfolio._require_inputs()
    config, config_sha256 = _load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    if args.smoke_one:
        spec = replace(portfolio.PANELS["local32"], name="smoke1", case_count=1)
        panel = _freeze_panel(
            spec, output_dir=output_dir, config_path=args.config.resolve()
        )
        report = {
            "status": "target-free-smoke",
            "panel": _panel_artifacts(panel),
            "competition_test_accessed": False,
        }
        _write_json(output_dir / "report.json", report)
        return report
    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)

    local_frozen = _freeze_panel(
        portfolio.PANELS["local32"],
        output_dir=output_dir,
        config_path=args.config.resolve(),
    )
    held_frozen = _freeze_panel(
        portfolio.PANELS["held32"],
        output_dir=output_dir,
        config_path=args.config.resolve(),
    )
    local = _score_panel(local_frozen, lookup=lookup, cache=cache)
    held = _score_panel(held_frozen, lookup=lookup, cache=cache)
    local_model = fit_relation_truth_classifier(local.frozen.boards, local.labels)
    local_model_path, local_model_freeze = _freeze_model(
        local_model,
        stage=output_dir / "model-local32",
        trained_on=(local,),
        config_path=args.config.resolve(),
    )
    local_summary = _selection_summary(local, local_model)
    held_summary = _selection_summary(held, local_model)
    held_pair_delta = held_summary["candidate_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    held_exact_delta = held_summary["candidate_minus_control"]["exact_tiles"]["mean"]
    held_signal = bool(
        held_pair_delta > 0.0
        and held_exact_delta >= HELD_EXACT_GATE
        and held_summary["changed_from_control_count"] > 0
    )
    report: dict[str, Any] = {
        "schema": "aiijc-taska-relation-truth-selector-report-v1",
        "status": "held_signal_pass" if held_signal else "rejected_on_held_signal",
        "preregistration_sha256": config_sha256,
        "development_exposure": (
            "local32/held32/fresh32 are existing opened development panels; "
            "fresh32 is not a formal confirmation"
        ),
        "local32_in_sample": {
            "edge_diagnostics": _edge_diagnostics(local, local_model),
            "selection": local_summary,
            "artifacts": _panel_artifacts(local_frozen),
        },
        "held32_source_disjoint": {
            "edge_diagnostics": _edge_diagnostics(held, local_model),
            "selection": held_summary,
            "signal_gate_passed": held_signal,
            "artifacts": _panel_artifacts(held_frozen),
        },
        "local_model": {
            "model": _record(local_model_path),
            "pre_held_model_freeze": _record(local_model_freeze),
        },
        "fresh32_development": {"status": "skipped_by_held_signal_gate"},
        "formal_confirmation": {
            "status": "not_eligible",
            "new_source_disjoint_roster_generated_or_scored": False,
        },
        "legality": {
            "strict_original_upright_576_tile_permutation": True,
            "selector_returns_one_whole_frozen_arm": True,
            "pixels_changed": False,
            "competition_test_accessed": False,
            "production_or_submission_changed": False,
        },
    }
    if held_signal:
        combined_model = fit_relation_truth_classifier(
            (*local.frozen.boards, *held.frozen.boards),
            (*local.labels, *held.labels),
        )
        combined_model_path, combined_model_freeze = _freeze_model(
            combined_model,
            stage=output_dir / "model-local32-held32",
            trained_on=(local, held),
            config_path=args.config.resolve(),
        )
        fresh_frozen = _freeze_panel(
            portfolio.PANELS["fresh32"],
            output_dir=output_dir,
            config_path=args.config.resolve(),
        )
        fresh = _score_panel(fresh_frozen, lookup=lookup, cache=cache)
        fresh_summary = _selection_summary(fresh, combined_model)
        fresh_pair_delta = fresh_summary["candidate_minus_control"][
            "satisfied_adjacent_pairs"
        ]["mean"]
        fresh_exact_delta = fresh_summary["candidate_minus_control"]["exact_tiles"][
            "mean"
        ]
        fresh_gate = bool(
            fresh_pair_delta >= FRESH_PAIR_GATE and fresh_exact_delta >= FRESH_EXACT_GATE
        )
        report["status"] = (
            "formal_confirmation_required" if fresh_gate else "rejected_on_fresh_gate"
        )
        report["fresh32_development"] = {
            "status": "complete",
            "edge_diagnostics": _edge_diagnostics(fresh, combined_model),
            "selection": fresh_summary,
            "promotion_gate_passed": fresh_gate,
            "artifacts": _panel_artifacts(fresh_frozen),
            "model": _record(combined_model_path),
            "pre_fresh_model_freeze": _record(combined_model_freeze),
        }
        report["formal_confirmation"] = {
            "status": "preregister_new_roster_before_generation" if fresh_gate else "not_eligible",
            "new_source_disjoint_roster_generated_or_scored": False,
        }
    report["runtime_seconds"] = perf_counter() - started
    _write_json(output_dir / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
