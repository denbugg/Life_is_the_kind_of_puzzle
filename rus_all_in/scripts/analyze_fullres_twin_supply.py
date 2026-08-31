#!/usr/bin/env python3
"""Target-aware diagnostic of the frozen fullres-twin top-32 supply panel."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from aiijc_puzzle.fullres_twin_side_matcher import (
    OPPOSITE_SIDE,
    FullResolutionTwinSideMatcher,
)
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file, split_tiles
from aiijc_puzzle.synthetic_socket_evaluation import (
    freeze_topk_candidates,
    make_exact_synthetic_case,
)
from scripts.run_fullres_twin_side_matcher import _load_rgb, _tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT_ROOT / "outputs/fullres-twin-side-matcher/v1-fit256-s400-eval24"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_RESTORED_METADATA = (
    PROJECT_ROOT
    / "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
    "frozen_local_predictions.json"
)
GRID = 24
COUNT = GRID * GRID
SEED = 20320917


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _truth_by_anchor(reference: np.ndarray, *, axis: str) -> np.ndarray:
    positions = np.arange(COUNT)
    valid = positions % GRID != GRID - 1 if axis == "right" else positions < COUNT - GRID
    delta = 1 if axis == "right" else GRID
    truth = np.full(COUNT, -1, dtype=np.int32)
    truth[reference[positions[valid]]] = reference[positions[valid] + delta]
    return truth


def _load_model(checkpoint: Path, device: torch.device) -> FullResolutionTwinSideMatcher:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = payload.get("contract", {})
    expected = {
        "architecture": "fullres-ordered-twin-side-matcher-v1",
        "dimension": 48,
        "field_blocks": 4,
        "sequence_blocks": 2,
        "raw_skip_gain": 0.35,
        "spatial_downsampling": False,
        "ordered_side_positions": 20,
        "pixel_prediction_head": False,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint architecture contract differs from frozen v1")
    model = FullResolutionTwinSideMatcher().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model


def _incoming_ranks(scores: np.ndarray) -> np.ndarray:
    count = len(scores)
    order = np.argsort(-scores, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order, np.arange(count)[None, :]] = np.arange(count, dtype=np.int32)[:, None]
    return ranks


def _union_size(first: np.ndarray, second: np.ndarray) -> int:
    return len(np.union1d(first, second))


def _rank_bin(rank: int) -> str:
    if rank == 0:
        return "rank_1"
    if rank < 5:
        return "rank_2_5"
    if rank < 16:
        return "rank_6_16"
    return "rank_17_32"


def _feature_quality(labels: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    labels = labels[finite].astype(np.int8)
    values = values[finite].astype(np.float64)
    positives = int(labels.sum())
    if not len(labels) or not positives or positives == len(labels):
        return {"count": len(labels), "positives": positives, "defined": False}
    auc = float(roc_auc_score(labels, values))
    orientation = "higher" if auc >= 0.5 else "lower"
    oriented = values if auc >= 0.5 else -values
    oriented_auc = max(auc, 1.0 - auc)
    average_precision = float(average_precision_score(labels, oriented))
    base_precision = positives / len(labels)
    top_tenth = max(1, len(labels) // 10)
    selected = np.argsort(-oriented, kind="stable")[:top_tenth]
    precision_top_tenth = float(labels[selected].mean())
    return {
        "defined": True,
        "count": len(labels),
        "positives": positives,
        "base_precision": base_precision,
        "raw_auc_higher_is_true": auc,
        "descriptive_best_orientation": orientation,
        "oriented_auc": oriented_auc,
        "average_precision": average_precision,
        "average_precision_lift": average_precision / base_precision,
        "top_10pct_precision": precision_top_tenth,
        "top_10pct_precision_lift": precision_top_tenth / base_precision,
    }


def _restored_overlap(metadata_path: Path, current_names: set[str]) -> dict[str, Any]:
    if not metadata_path.is_file():
        return {
            "available": False,
            "same_case_overlap_count": 0,
            "reason": "prior restored-denoiser frozen metadata is absent",
        }
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    restored_names = {str(case["source_filename"]) for case in payload.get("cases", [])}
    overlap = sorted(current_names & restored_names)
    return {
        "available": bool(overlap),
        "same_case_overlap_count": len(overlap),
        "same_case_filenames": overlap,
        "restored_artifact": str(metadata_path.relative_to(PROJECT_ROOT)),
        "restored_artifact_sha256": sha256_file(metadata_path),
        "reason": (
            "same-case frozen restored supply can be compared"
            if overlap
            else "zero overlap by source-disjoint design; cross-panel aggregate overlap is invalid"
        ),
        "new_restored_inference_run": False,
    }


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(args.device)
    run_dir = args.run_dir.resolve()
    output = args.output or (run_dir / "target-aware-supply-diagnostic.json")
    report_path = run_dir / "report.json"
    commitment_path = run_dir / "selection-commitment.json"
    checkpoint_path = run_dir / "fullres-twin-side-matcher.pt"
    arrays_path = run_dir / "frozen-local-predictions.npz"
    metadata_path = run_dir / "frozen-local-predictions.json"
    frozen_report = json.loads(report_path.read_text(encoding="utf-8"))
    if frozen_report["gate"]["passed"] is not False:
        raise ValueError("diagnostic is registered only for the failed frozen D1")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = np.load(arrays_path)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest is invalid")
    records = {record["filename"]: record for record in manifest["splits"]["train"]}
    model = _load_model(checkpoint_path, device)

    rank_bins: defaultdict[str, int] = defaultdict(int)
    board_correct: list[dict[str, Any]] = []
    union_board_rows: list[dict[str, Any]] = []
    feature_labels: list[np.ndarray] = []
    feature_values: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    expected_names = commitment["evaluation_filenames"]
    metadata_names = [case["source_filename"] for case in metadata["cases"]]
    if metadata_names != expected_names:
        raise ValueError("frozen prediction metadata order differs from commitment")

    for case_index, case in enumerate(metadata["cases"]):
        filename = str(case["source_filename"])
        record = records[filename]
        target_path = args.targets / filename
        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"target hash mismatch: {filename}")
        clean = split_tiles(_load_rgb(target_path))
        item, reference = make_exact_synthetic_case(
            clean,
            source_filename=filename,
            draw_index=0,
            seed=SEED,
        )
        if item.case_id != case["case_id"]:
            raise ValueError("recreated exact case differs from frozen metadata")
        with torch.inference_mode():
            model_output = model(_tensor(item.tiles, device))
        scores_all = model_output.scores[0].float().cpu().numpy()
        sides = model_output.sides[0].float().cpu().numpy()
        case_correct = {"right": 0, "down": 0}
        case_bins: defaultdict[str, int] = defaultdict(int)
        case_union: dict[str, Any] = {"source_filename": filename}
        prefix = str(case["prefix"])
        for axis, direction in (("right", 1), ("down", 3)):
            raw = arrays[f"{prefix}__supply__socket_d64_raw__{axis}"]
            twin = arrays[f"{prefix}__supply__fullres_twin__{axis}"]
            scores = scores_all[direction]
            rerun_top32 = freeze_topk_candidates(scores, max_k=32)
            if not np.array_equal(rerun_top32, twin):
                raise RuntimeError("checkpoint rerun differs from frozen twin top32")
            truth = _truth_by_anchor(reference.tile_at_position, axis=axis)
            anchors = np.flatnonzero(truth >= 0)
            incoming_rank = _incoming_ranks(scores)
            top33 = np.argsort(-scores, axis=1, kind="stable")[:, :33]
            row_threshold = scores[np.arange(COUNT), top33[:, 32]]
            column_order = np.argsort(-scores, axis=0, kind="stable")[:33]
            column_threshold = scores[column_order[32], np.arange(COUNT)]
            axis_union_sizes: list[int] = []
            axis_union_correct = 0
            top1_union_correct = 0
            top1_union_edges = 0
            selected_anchors: list[int] = []
            selected_candidates: list[int] = []
            selected_ranks: list[int] = []
            selected_labels: list[bool] = []
            for anchor in anchors:
                raw_row = raw[anchor]
                twin_row = twin[anchor]
                truth_tile = int(truth[anchor])
                raw_hit = bool(np.any(raw_row == truth_tile))
                twin_matches = np.flatnonzero(twin_row == truth_tile)
                if not raw_hit and len(twin_matches):
                    rank = int(twin_matches[0])
                    case_correct[axis] += 1
                    case_bins[_rank_bin(rank)] += 1
                    rank_bins[_rank_bin(rank)] += 1
                union_row = np.union1d(raw_row, twin_row)
                axis_union_sizes.append(len(union_row))
                axis_union_correct += int(truth_tile in union_row)
                top1_union = np.union1d(raw_row[:1], twin_row[:1])
                top1_union_edges += len(top1_union)
                top1_union_correct += int(truth_tile in top1_union)
                raw_set = set(raw_row.tolist())
                for rank, candidate in enumerate(twin_row):
                    if int(candidate) in raw_set:
                        continue
                    selected_anchors.append(int(anchor))
                    selected_candidates.append(int(candidate))
                    selected_ranks.append(rank)
                    selected_labels.append(int(candidate) == truth_tile)
            anchors_array = np.asarray(selected_anchors, dtype=np.int64)
            candidates_array = np.asarray(selected_candidates, dtype=np.int64)
            ranks_array = np.asarray(selected_ranks, dtype=np.int32)
            labels_array = np.asarray(selected_labels, dtype=np.bool_)
            edge_scores = scores[anchors_array, candidates_array]
            incoming = incoming_rank[anchors_array, candidates_array]
            row_margin = edge_scores - row_threshold[anchors_array]
            column_margin = edge_scores - column_threshold[candidates_array]
            mutual_top32 = incoming < 32
            reciprocal_top1 = (ranks_array == 0) & (incoming == 0)
            source_tokens = sides[anchors_array, direction]
            target_tokens = sides[candidates_array, OPPOSITE_SIDE[direction]]
            position_dot = np.sum(source_tokens * target_tokens, axis=2)
            sequence_variance = np.var(position_dot, axis=1)
            tangent_variation = 0.5 * (
                np.mean(np.square(np.diff(source_tokens, axis=1)), axis=(1, 2))
                + np.mean(np.square(np.diff(target_tokens, axis=1)), axis=(1, 2))
            )
            feature_labels.append(labels_array)
            values = {
                "negative_outgoing_rank": -ranks_array.astype(np.float32),
                "negative_incoming_rank": -incoming.astype(np.float32),
                "mutual_top32": mutual_top32.astype(np.float32),
                "reciprocal_top1": reciprocal_top1.astype(np.float32),
                "twin_score": edge_scores,
                "row_margin_to_rank33": row_margin,
                "two_sided_margin_to_rank33": np.minimum(row_margin, column_margin),
                "sequence_dot_variance": sequence_variance,
                "side_tangent_variation": tangent_variation,
            }
            for name, value in values.items():
                feature_values[name].append(np.asarray(value))
            case_union[axis] = {
                "physical_queries": len(anchors),
                "top32_union_edge_count": int(sum(axis_union_sizes)),
                "top32_union_mean_candidates_per_query": float(np.mean(axis_union_sizes)),
                "top32_union_correct_truth_edges": axis_union_correct,
                "top32_oracle_top144_precision_ceiling": min(axis_union_correct, 144) / 144,
                "top1_union_edge_count": top1_union_edges,
                "top1_union_correct_truth_edges": top1_union_correct,
                "top1_oracle_top144_precision_ceiling": min(top1_union_correct, 144) / 144,
                "twin_only_candidate_edges": len(labels_array),
                "twin_only_true_edges": int(labels_array.sum()),
            }
        board_correct.append(
            {
                "source_filename": filename,
                "right": case_correct["right"],
                "down": case_correct["down"],
                "pooled": case_correct["right"] + case_correct["down"],
                "rank_bins": dict(case_bins),
            }
        )
        union_board_rows.append(case_union)
        print(f"diagnosed {case_index + 1}/{len(metadata['cases'])} {filename}", flush=True)

    labels = np.concatenate(feature_labels)
    feature_quality = {
        name: _feature_quality(labels, np.concatenate(values))
        for name, values in feature_values.items()
    }
    ordered_features = sorted(
        feature_quality,
        key=lambda name: float(feature_quality[name].get("average_precision_lift", 0.0)),
        reverse=True,
    )
    pooled_correct = np.asarray([row["pooled"] for row in board_correct])
    right_correct = np.asarray([row["right"] for row in board_correct])
    down_correct = np.asarray([row["down"] for row in board_correct])
    top32_axis_rows = [row[axis] for row in union_board_rows for axis in ("right", "down")]
    top1_ceiling = np.asarray(
        [row["top1_oracle_top144_precision_ceiling"] for row in top32_axis_rows]
    )
    top32_ceiling = np.asarray(
        [row["top32_oracle_top144_precision_ceiling"] for row in top32_axis_rows]
    )
    candidate_counts = np.asarray(
        [row["top32_union_mean_candidates_per_query"] for row in top32_axis_rows]
    )
    board_top32_edge_counts = np.asarray(
        [
            row["right"]["top32_union_edge_count"]
            + row["down"]["top32_union_edge_count"]
            for row in union_board_rows
        ]
    )
    restored_overlap = _restored_overlap(
        DEFAULT_RESTORED_METADATA,
        set(expected_names),
    )
    diagnostic = {
        "schema": "aiijc-fullres-twin-target-aware-supply-diagnostic-v1",
        "status": "descriptive-on-already-opened-frozen-d1-no-new-arm",
        "frozen_inputs": {
            "report": str(report_path.relative_to(PROJECT_ROOT)),
            "report_sha256": sha256_file(report_path),
            "selection_commitment": str(commitment_path.relative_to(PROJECT_ROOT)),
            "selection_commitment_sha256": sha256_file(commitment_path),
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions": str(arrays_path.relative_to(PROJECT_ROOT)),
            "predictions_sha256": sha256_file(arrays_path),
            "evaluation_order_digest": commitment["evaluation_order_digest"],
        },
        "scope": {
            "same_opened_eval24_only": True,
            "new_source_or_target_access": False,
            "new_model_training_or_weight_sweep": False,
            "decoder_layout_holdout_or_test": False,
            "target_assisted_aggregate_diagnostic": True,
        },
        "twin_only_correct_top32": {
            "right_total": int(right_correct.sum()),
            "down_total": int(down_correct.sum()),
            "pooled_total": int(pooled_correct.sum()),
            "right_mean_per_board": float(right_correct.mean()),
            "down_mean_per_board": float(down_correct.mean()),
            "pooled_mean_per_board": float(pooled_correct.mean()),
            "pooled_std_per_board": float(pooled_correct.std()),
            "pooled_min_per_board": int(pooled_correct.min()),
            "pooled_max_per_board": int(pooled_correct.max()),
            "rank_bins": dict(rank_bins),
            "per_board": board_correct,
        },
        "restored_denoiser_overlap": restored_overlap,
        "raw_twin_union": {
            "mean_top32_union_candidates_per_physical_query": float(candidate_counts.mean()),
            "min_board_axis_mean_candidates": float(candidate_counts.min()),
            "max_board_axis_mean_candidates": float(candidate_counts.max()),
            "mean_top32_union_candidate_edges_per_board": float(
                board_top32_edge_counts.mean()
            ),
            "min_top32_union_candidate_edges_per_board": int(
                board_top32_edge_counts.min()
            ),
            "max_top32_union_candidate_edges_per_board": int(
                board_top32_edge_counts.max()
            ),
            "mean_top32_oracle_top144_precision_ceiling": float(top32_ceiling.mean()),
            "minimum_top32_oracle_top144_precision_ceiling": float(top32_ceiling.min()),
            "mean_top1_oracle_top144_precision_ceiling": float(top1_ceiling.mean()),
            "minimum_top1_oracle_top144_precision_ceiling": float(top1_ceiling.min()),
            "per_board_axis": union_board_rows,
        },
        "target_free_feature_separation_on_twin_only_edges": {
            "candidate_edge_count": len(labels),
            "true_edge_count": int(labels.sum()),
            "base_precision": float(labels.mean()),
            "feature_order_by_descriptive_ap_lift": ordered_features,
            "features": feature_quality,
            "warning": (
                "orientation and ordering are target-aware descriptive statistics on the opened "
                "D1; they are not frozen gates or permission to fit this panel"
            ),
        },
        "recommended_next_experiment": {
            "name": "raw-twin-union-reranker-v2",
            "status": "architecture-proposal-only-awaiting-approval-no-roster-or-target-access",
            "candidate_roster": "immutable raw d64 top32 union frozen twin top32",
            "model": (
                "zero-initialised residual over raw d64 using a width64 two-layer "
                "permutation-equivariant candidate-set encoder"
            ),
            "inputs": [
                "raw/twin outgoing and incoming ranks and row-standardised scores",
                "raw/twin reciprocal flags and row/column margins",
                "frozen twin 20-position dot mean/std/min and side tangent variation",
                "query-relative candidate-set context; no tile index or target feature",
            ],
            "training": (
                "new organizer-train fit256 exact shuffles, frozen d64 and twin checkpoints, "
                "listwise CE plus precision-weighted top144 hard-edge auxiliary"
            ),
            "evaluation": (
                "new source-disjoint manifest-train D1 24x1 excluding this eval24 and all "
                "active panels; freeze scores before labels"
            ),
            "proposed_gate": (
                "pooled R1 +0.25pp with R5 nonnegative OR matched top144 precision +2pp "
                "at no lower correct-edge count; no decoder at this gate"
            ),
            "explicit_non_actions": [
                "do not train or choose features on the opened eval24",
                "do not fixed-average raw and twin scores",
                "do not run a layout decoder without a separate D2 gate",
            ],
        },
    }
    _atomic_json(output, diagnostic)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "twin_only_correct_per_board": float(pooled_correct.mean()),
                "top_feature": ordered_features[0],
                "top32_oracle_ceiling": float(top32_ceiling.mean()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
