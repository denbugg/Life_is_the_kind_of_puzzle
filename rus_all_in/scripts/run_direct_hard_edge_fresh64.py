#!/usr/bin/env python3
"""Strict fresh64 confirmation for the frozen direct hard-edge model."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from run_direct_hard_edge_priority import (
    EDGE_BUDGET_PER_AXIS,
    GRID,
    HARD_EDGES_PER_BOARD,
    TILE_COUNT,
    CleanTileCache,
    _collect_actual_roster_filenames,
    _collect_filename_lists,
    _dirty_sha256,
    _forward_board,
    _load_json_or_checkpoint,
    _make_case,
    _record_lookup,
    _relative,
    _resolve,
    fixed_budget_metrics,
    learned_priority_matrices,
    load_frozen_config,
)

from aiijc_puzzle.direct_hard_edge_priority import DirectHardEdgePriority
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import (
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
)
from aiijc_puzzle.socket_confidence_calibration import HardEdgeFeatures, exact_edge_labels
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import (
    choose_deterministic_device,
    load_socket_checkpoint,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.synthetic_socket_evaluation import names_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT / "configs/direct_hard_edge_board_priority_preregistered_v1.json"
)
DEFAULT_PRIOR_REPORT = (
    PROJECT_ROOT
    / "outputs/direct-hard-edge-priority/v1-fit256-s600-d1-32-cpu/report.json"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/direct_hard_edge_fresh64_confirmation_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/direct-hard-edge-priority/frozen-v1-fresh64-draw0"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
NEW_EXCLUSION_ARTIFACTS = (
    "outputs/fullres-twin-side-matcher/v1-fit256-s400-eval24/selection-commitment.json",
)
SELECTION_NAMESPACE = "aiijc-direct-hard-edge-frozen-fresh64-confirmation-v1"
EXPECTED_SOURCES = 64
SYNTHETIC_SEED = 20260918
BOOTSTRAP_RESAMPLES = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("selection", "run"), default="run")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--prior-report", type=Path, default=DEFAULT_PRIOR_REPORT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--additional-exclusion", type=Path, action="append", default=[])
    return parser.parse_args()


def _base_exclusion_names(base_config: Mapping[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    names: set[str] = set()
    registry: list[dict[str, Any]] = []
    for row in base_config["selection"]["exclusion"]["registry"]:
        path = _resolve(str(row["path"]))
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"base exclusion artifact changed: {path}")
        payload = _load_json_or_checkpoint(path)
        role = str(row["role"])
        if role.startswith("actual-panel-roster-exclusion"):
            found = _collect_actual_roster_filenames(payload)
        else:
            found = _collect_filename_lists(payload)
        if len(found) != int(row["filename_count"]):
            raise ValueError(f"base exclusion roster count changed: {path}")
        names.update(found)
        registry.append(dict(row))
    for key in ("fit_source_filenames", "d1_source_filenames"):
        names.update(Path(value).name for value in base_config["selection"][key])
    registry.append(
        {
            "path": _relative(DEFAULT_BASE_CONFIG),
            "sha256": sha256_file(DEFAULT_BASE_CONFIG),
            "filename_count": len(
                set(base_config["selection"]["fit_source_filenames"])
                | set(base_config["selection"]["d1_source_filenames"])
            ),
            "role": "direct-head fit256 plus opened D1-32",
        }
    )
    return names, registry


def freeze_config(args: argparse.Namespace) -> None:
    path = args.config.resolve()
    digest_path = path.with_name(f"{path.name}.sha256")
    if path.exists() or digest_path.exists():
        raise FileExistsError("refusing to overwrite the fresh64 preregistration")
    base, base_sha = load_frozen_config(args.base_config)
    prior_sha = sha256_file(args.prior_report)
    prior = json.loads(args.prior_report.read_text(encoding="utf-8"))
    if prior.get("config_sha256") != base_sha:
        raise ValueError("prior report/base config mismatch")
    checkpoint_path = Path(prior["checkpoint"]["path"])
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != prior["checkpoint"]["sha256"]:
        raise ValueError("frozen model SHA mismatch")
    forbidden, registry = _base_exclusion_names(base)
    additional = [
        *(PROJECT_ROOT / value for value in NEW_EXCLUSION_ARTIFACTS),
        *args.additional_exclusion,
    ]
    for artifact_path in sorted({value.resolve() for value in additional}):
        payload = _load_json_or_checkpoint(artifact_path)
        found = _collect_filename_lists(payload)
        if not found:
            raise ValueError(f"additional exclusion contains no *_filenames: {artifact_path}")
        forbidden.update(found)
        registry.append(
            {
                "path": _relative(artifact_path),
                "sha256": sha256_file(artifact_path),
                "filename_count": len(found),
                "filename_digest": names_digest(sorted(found)),
                "role": "new frozen/opened panel exclusion",
            }
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest mismatch")
    train = manifest.get("splits", {}).get("train")
    if not isinstance(train, list):
        raise ValueError("manifest train split is missing")
    forbidden_digest = names_digest(sorted(forbidden))
    namespace = (
        f"{SELECTION_NAMESPACE}\0{checkpoint_sha}\0{forbidden_digest}\0{SYNTHETIC_SEED}"
    )
    ranked = select_manifest_records(
        manifest,
        "train",
        limit=len(train),
        namespace=namespace,
    )
    records = tuple(
        record
        for record in ranked
        if Path(str(record["filename"])).name not in forbidden
    )[:EXPECTED_SOURCES]
    if len(records) != EXPECTED_SOURCES:
        raise ValueError("not enough fresh source-disjoint records remain")
    source_names = [str(record["filename"]) for record in records]
    if set(source_names) & forbidden:
        raise RuntimeError("fresh64 selection overlaps its exclusion union")
    payload = {
        "schema": "aiijc-direct-hard-edge-frozen-fresh64-confirmation-v1",
        "registered_before_selected_target_access": True,
        "registered_before_dirty_prediction_generation": True,
        "frozen_inputs": {
            "base_config": _relative(args.base_config),
            "base_config_sha256": base_sha,
            "prior_report": _relative(args.prior_report),
            "prior_report_sha256": prior_sha,
            "direct_hard_edge_checkpoint": _relative(checkpoint_path),
            "direct_hard_edge_checkpoint_sha256": checkpoint_sha,
            "socket_checkpoint": base["frozen_inputs"]["socket_checkpoint"],
            "socket_checkpoint_sha256": base["frozen_inputs"][
                "socket_checkpoint_sha256"
            ],
            "no_retrain_recalibration_or_score_sweep": True,
        },
        "selection": {
            "split": "train",
            "namespace": namespace,
            "synthetic_seed": SYNTHETIC_SEED,
            "draw_indices": [0],
            "source_filenames": source_names,
            "source_order_digest": names_digest(source_names),
            "source_set_digest": names_digest(source_names, sort_names=True),
            "excluded_filename_count": len(forbidden),
            "excluded_filename_digest": forbidden_digest,
            "exclusion_registry": registry,
            "selected_exclusion_overlap": [],
        },
        "gate": {
            "primary_all_required": {
                "top144_per_axis_correct_gain_per_board_minimum": 1.0,
                "decoder144_cyclic5_adjacency_delta_strictly_positive": True,
            },
            "secondary_required": {
                "translation_aligned_tiles_delta_nonnegative": True,
            },
            "clustered_ci": {
                "cluster": "source; exactly one draw per source",
                "confidence": 0.95,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "seed": SYNTHETIC_SEED + 31,
                "used_for_reporting_not_threshold": True,
            },
            "exact": {
                "role": "descriptive",
                "unexpected_strong_root_review": (
                    "mean gain >=0.5 tile/board and clustered CI lower >0"
                ),
            },
            "promotion_authorized": False,
        },
        "legality": {
            "candidate_supply": "existing raw d64 hard edges only",
            "layout": "strict permutation of original upright tiles",
            "holdout_opened": False,
            "competition_test_opened": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(path)
    digest_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "fresh64-preregistered",
                "path": str(path),
                "sha256": digest,
                "source_digest": names_digest(source_names),
                "excluded": len(forbidden),
                "selected_target_access": False,
            }
        ),
        flush=True,
    )


def load_confirmation_config(path: Path) -> tuple[dict[str, Any], str]:
    expected = path.with_name(f"{path.name}.sha256").read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("fresh64 config SHA mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "aiijc-direct-hard-edge-frozen-fresh64-confirmation-v1":
        raise ValueError("unsupported fresh64 config schema")
    if not payload.get("registered_before_selected_target_access"):
        raise ValueError("fresh64 was not registered before target access")
    return payload, observed


def source_clustered_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("cluster values must be a finite non-empty vector")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    generator = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        batch = min(2048, remaining)
        indices = generator.integers(0, len(array), size=(batch, len(array)))
        means.append(array[indices].mean(axis=1))
        remaining -= batch
    bootstrap = np.concatenate(means)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "mean": float(array.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "source_clusters": len(array),
        "bootstrap_resamples": resamples,
    }


def evaluate_confirmation_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    correct = float(metrics["hard_edge_correct_gain"]["mean"])
    adjacency = float(metrics["adjacency_delta"]["mean"])
    aligned = float(metrics["translation_aligned_tiles_delta"]["mean"])
    exact = metrics["exact_tiles_delta"]
    checks = {
        "top144_correct_gain": {"observed": correct, "required": 1.0, "pass": correct >= 1.0},
        "adjacency_strict_positive": {
            "observed": adjacency,
            "required": ">0",
            "pass": adjacency > 0.0,
        },
        "translation_aligned_nonnegative": {
            "observed": aligned,
            "required": ">=0",
            "pass": aligned >= 0.0,
        },
    }
    passed = all(bool(row["pass"]) for row in checks.values())
    unexpectedly_strong_exact = (
        float(exact["mean"]) >= 0.5 and float(exact["ci95_lower"]) > 0.0
    )
    return {
        "pass": passed,
        "status": "structural-confirmation-pass" if passed else "stop",
        "checks": checks,
        "unexpectedly_strong_exact_root_review": unexpectedly_strong_exact,
        "promotion_authorized": False,
        "competition_test_authorized": False,
    }


def _arm_means(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    raw = np.asarray([row[f"raw_{metric}"] for row in rows], dtype=np.float64)
    learned = np.asarray([row[f"learned_{metric}"] for row in rows], dtype=np.float64)
    return {
        "raw_mean": float(raw.mean()),
        "learned_mean": float(learned.mean()),
        "mean_delta": float((learned - raw).mean()),
    }


def run_confirmation(args: argparse.Namespace) -> None:
    config, config_sha = load_confirmation_config(args.config)
    frozen = config["frozen_inputs"]
    for field, hash_field in (
        ("base_config", "base_config_sha256"),
        ("prior_report", "prior_report_sha256"),
        ("direct_hard_edge_checkpoint", "direct_hard_edge_checkpoint_sha256"),
        ("socket_checkpoint", "socket_checkpoint_sha256"),
    ):
        if sha256_file(_resolve(str(frozen[field]))) != frozen[hash_field]:
            raise ValueError(f"frozen input changed: {field}")
    base, base_sha = load_frozen_config(_resolve(str(frozen["base_config"])))
    if base_sha != frozen["base_config_sha256"]:
        raise ValueError("base config lineage mismatch")
    checkpoint_path = _resolve(str(frozen["direct_hard_edge_checkpoint"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = choose_deterministic_device(args.device)
    socket = load_socket_checkpoint(_resolve(str(frozen["socket_checkpoint"])), device=device)
    head = DirectHardEdgePriority(
        int(base["model"]["input_dimension"]),
        hidden_dimension=int(base["model"]["hidden_dimension"]),
    ).to(device)
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_names = list(config["selection"]["source_filenames"])
    if len(source_names) != EXPECTED_SOURCES or names_digest(source_names) != config[
        "selection"
    ]["source_order_digest"]:
        raise ValueError("fresh64 roster contract changed")
    records = _record_lookup(manifest, source_names)
    cache = CleanTileCache(args.targets)
    seed = int(config["selection"]["synthetic_seed"])
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=EDGE_BUDGET_PER_AXIS,
        max_swap_steps=24,
    )
    cyclic_config = CyclicTranslationConfig(border_weight=5.0)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "frozen_predictions.npz"
    report_path = output_dir / "report.json"
    if prediction_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite the fresh64 confirmation")
    print(
        json.dumps(
            {"event": "start", "pid": os.getpid(), "device": str(device), "sources": 64}
        ),
        flush=True,
    )
    started = perf_counter()
    frozen_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, record in enumerate(records):
            dirty, _ = _make_case(cache, record, draw_index=0, seed=seed)
            board, features, scores, output = _forward_board(
                socket,
                head,
                dirty.tiles,
                device=device,
            )
            priorities = learned_priority_matrices(board, scores, grid=GRID)
            raw_decoder = decode_socket_assignments(
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
                config=decoder_config,
            )
            learned_decoder = decode_socket_assignments(
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
                config=decoder_config,
                component_edge_priority=priorities,
            )
            raw_layout = select_global_cyclic_translation(
                raw_decoder.layout,
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
                config=cyclic_config,
            ).layout
            learned_layout = select_global_cyclic_translation(
                learned_decoder.layout,
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
                config=cyclic_config,
            ).layout
            if any(
                not np.array_equal(np.sort(layout), np.arange(TILE_COUNT))
                for layout in (raw_layout, learned_layout)
            ):
                raise RuntimeError("fresh64 output is not a strict original permutation")
            frozen_rows.append(
                {
                    "source_filename": str(record["filename"]),
                    "case_id": dirty.case_id,
                    "dirty_sha256": _dirty_sha256(dirty.tiles),
                    "raw_scores": features.values[:, 0].copy(),
                    "learned_scores": scores.detach().cpu().numpy().copy(),
                    "source": features.source.copy(),
                    "target": features.target.copy(),
                    "axis": features.axis.copy(),
                    "raw_layout": raw_layout.copy(),
                    "learned_layout": learned_layout.copy(),
                }
            )
            print(
                json.dumps(
                    {"event": "freeze", "done": index + 1, "total": EXPECTED_SOURCES}
                ),
                flush=True,
            )
    np.savez_compressed(
        prediction_path,
        source_filenames=np.asarray(source_names, dtype="U64"),
        case_ids=np.asarray([row["case_id"] for row in frozen_rows], dtype="U160"),
        dirty_sha256=np.asarray([row["dirty_sha256"] for row in frozen_rows], dtype="U64"),
        raw_scores=np.stack([row["raw_scores"] for row in frozen_rows]),
        learned_scores=np.stack([row["learned_scores"] for row in frozen_rows]),
        source=np.stack([row["source"] for row in frozen_rows]),
        target=np.stack([row["target"] for row in frozen_rows]),
        axis=np.stack([row["axis"] for row in frozen_rows]),
        raw_layout=np.stack([row["raw_layout"] for row in frozen_rows]),
        learned_layout=np.stack([row["learned_layout"] for row in frozen_rows]),
    )
    prediction_sha = sha256_file(prediction_path)
    print(
        json.dumps(
            {"event": "predictions-frozen", "path": str(prediction_path), "sha256": prediction_sha}
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for index, (record, frozen_row) in enumerate(zip(records, frozen_rows, strict=True)):
        dirty, reference = _make_case(cache, record, draw_index=0, seed=seed)
        if dirty.case_id != frozen_row["case_id"] or _dirty_sha256(dirty.tiles) != frozen_row[
            "dirty_sha256"
        ]:
            raise RuntimeError("fresh64 scoring input differs from frozen prediction")
        proxy = HardEdgeFeatures(
            values=np.zeros((HARD_EDGES_PER_BOARD, 20), dtype=np.float32),
            source=frozen_row["source"],
            target=frozen_row["target"],
            axis=frozen_row["axis"],
        )
        labels = exact_edge_labels(proxy, reference.tile_at_position, grid=GRID)
        raw_edges = fixed_budget_metrics(
            frozen_row["raw_scores"],
            labels,
            frozen_row["axis"],
            edge_budget_per_axis=EDGE_BUDGET_PER_AXIS,
        )
        learned_edges = fixed_budget_metrics(
            frozen_row["learned_scores"],
            labels,
            frozen_row["axis"],
            edge_budget_per_axis=EDGE_BUDGET_PER_AXIS,
        )
        raw_layout = evaluate_layout(
            frozen_row["raw_layout"], reference.tile_at_position, reference_is_exact=True
        )
        learned_layout = evaluate_layout(
            frozen_row["learned_layout"],
            reference.tile_at_position,
            reference_is_exact=True,
        )
        rows.append(
            {
                "source_filename": str(record["filename"]),
                "raw_correct_edges": raw_edges["correct_selected_edges"],
                "learned_correct_edges": learned_edges["correct_selected_edges"],
                "raw_edge_precision": raw_edges["selected_edge_precision"],
                "learned_edge_precision": learned_edges["selected_edge_precision"],
                "raw_adjacency": raw_layout.adjacency,
                "learned_adjacency": learned_layout.adjacency,
                "raw_translation_aligned_tiles": raw_layout.translation_aligned_count,
                "learned_translation_aligned_tiles": learned_layout.translation_aligned_count,
                "raw_exact_tiles": raw_layout.correct_tile_count,
                "learned_exact_tiles": learned_layout.correct_tile_count,
            }
        )
        print(
            json.dumps({"event": "score", "done": index + 1, "total": EXPECTED_SOURCES}),
            flush=True,
        )
    ci_seed = int(config["gate"]["clustered_ci"]["seed"])

    def delta_ci(metric: str, offset: int) -> dict[str, float | int]:
        values = [float(row[f"learned_{metric}"]) - float(row[f"raw_{metric}"]) for row in rows]
        return source_clustered_ci(values, seed=ci_seed + offset)

    metrics = {
        "hard_edge_correct_gain": delta_ci("correct_edges", 0),
        "edge_precision_gain": delta_ci("edge_precision", 1),
        "adjacency_delta": delta_ci("adjacency", 2),
        "translation_aligned_tiles_delta": delta_ci("translation_aligned_tiles", 3),
        "exact_tiles_delta": delta_ci("exact_tiles", 4),
        "arms": {
            "correct_edges": _arm_means(rows, "correct_edges"),
            "edge_precision": _arm_means(rows, "edge_precision"),
            "adjacency": _arm_means(rows, "adjacency"),
            "translation_aligned_tiles": _arm_means(rows, "translation_aligned_tiles"),
            "exact_tiles": _arm_means(rows, "exact_tiles"),
        },
    }
    gate = evaluate_confirmation_gate(metrics)
    report = {
        "schema": "aiijc-direct-hard-edge-frozen-fresh64-confirmation-report-v1",
        "config": str(args.config.resolve()),
        "config_sha256": config_sha,
        "frozen_predictions": str(prediction_path),
        "frozen_predictions_sha256": prediction_sha,
        "predictions_frozen_before_reference_scoring": True,
        "selection": config["selection"],
        "metrics": metrics,
        "gate": gate,
        "rows": rows,
        "runtime_seconds": perf_counter() - started,
        "strict_original_permutations": 2 * EXPECTED_SOURCES,
        "retrain_recalibration_or_sweep": False,
        "promotion_authorized": False,
        "competition_test_opened": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "gate": gate,
                "metrics": metrics,
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.mode == "selection":
        freeze_config(args)
    else:
        run_confirmation(args)


if __name__ == "__main__":
    main()
