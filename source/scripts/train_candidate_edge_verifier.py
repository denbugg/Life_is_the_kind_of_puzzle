#!/usr/bin/env python3
"""Train a leakage-safe tabular true-edge verifier and test component assembly.

The training/calibration fixtures are the disjoint contextual-refiner development
sources.  The final evaluation fixtures are the frozen candidate-graph v4
sources.  Source names, rather than individual edges, are the split unit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.components import (
    ProposedEdge,
    _complete_with_hungarian,
    _place_components_beam,
    grow_components_with_edges,
    project_rigid_components_around_reference,
)
from puzzle_assembly.geometry import TILE_COUNT, true_neighbour_slots
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics


GRID = 24
FEATURE_NAMES = (
    "direction_down",
    "cost",
    "row_rank",
    "column_rank",
    "cost_minus_row_min",
    "cost_minus_column_min",
    "row_z",
    "column_z",
    "boundary_mae",
    "boundary_rmse",
    "band4_mae",
    "gradient_mae",
    "edge_bias_abs",
    "tile_mean_mae",
    "tile_std_mae",
)
REFERENCE_WEIGHTS = (4.0, 16.0, 64.0)


@dataclass(frozen=True)
class Record:
    key: str
    source_name: str
    panel: str
    seed: int
    tiles: np.ndarray
    right: np.ndarray
    down: np.ndarray
    truth: np.ndarray
    graph_path: Path | None = None
    clean_target: np.ndarray | None = None
    baseline_layout: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root",
        default="runs/assembly_v1/contextual_refiner/v2_development_phase_a_20260712T1658Z",
    )
    parser.add_argument(
        "--v4-fixture-root",
        default="runs/assembly_v1/candidate_graph_oracle_fixtures_v4_6c0fe4e8524ce39d830d9a5bee118d8b",
    )
    parser.add_argument(
        "--v4-graph-root",
        default="runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_readback/candidate_graph_oracle_v4_phase_a/finalized",
    )
    parser.add_argument(
        "--output-root", default="runs/assembly_v1/candidate_edge_verifier_v1"
    )
    parser.add_argument("--calibration-sources", type=int, default=8)
    parser.add_argument("--hard-negatives-per-record", type=int, default=12000)
    parser.add_argument("--row-top-k", type=int, default=32)
    parser.add_argument("--column-top-k", type=int, default=8)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_from_artifact(path: str) -> str:
    return Path(path).stem.split("__", 1)[0] + ".png"


def truth_from_seed(seed: int) -> np.ndarray:
    # make_exact_panel creates this permutation before any degradation work.
    return np.random.default_rng(seed).permutation(TILE_COUNT).astype(np.int32)


def load_training_records(root: Path) -> list[Record]:
    manifest = json.loads((root / "manifest.json").read_text())
    records = []
    for meta in manifest["records"]:
        artifact = root / meta["artifact"]
        source = source_from_artifact(meta["artifact"])
        with np.load(artifact, allow_pickle=False) as arrays:
            records.append(
                Record(
                    key=artifact.stem,
                    source_name=source,
                    panel=str(meta["panel"]),
                    seed=int(meta["panel_seed"]),
                    tiles=np.ascontiguousarray(arrays["selected_slot_tiles"]),
                    right=np.ascontiguousarray(arrays["w4_right"]),
                    down=np.ascontiguousarray(arrays["w4_down"]),
                    truth=truth_from_seed(int(meta["panel_seed"])),
                )
            )
    if len(records) != 64 or len({r.source_name for r in records}) != 32:
        raise RuntimeError("expected 64 records from 32 training sources")
    return records


def load_v4_records(fixture_root: Path, graph_root: Path) -> list[Record]:
    label_manifest = json.loads(
        (fixture_root / "fixture_label/fixture_label_manifest.json").read_text()
    )
    records = []
    for meta in label_manifest["records"]:
        opaque_id = str(meta["opaque_id"])
        graph_path = graph_root / "artifacts" / f"{opaque_id}.graph.npz"
        label_path = fixture_root / "fixture_label/records" / f"{opaque_id}.npz"
        with np.load(graph_path, allow_pickle=False) as graph, np.load(
            label_path, allow_pickle=False
        ) as label:
            records.append(
                Record(
                    key=opaque_id,
                    source_name=str(meta["source_name"]),
                    panel=str(meta["panel"]),
                    seed=int(meta["panel_seed"]),
                    tiles=np.ascontiguousarray(graph["denoised_tiles"]),
                    right=np.ascontiguousarray(graph["w4_right"]),
                    down=np.ascontiguousarray(graph["w4_down"]),
                    truth=np.ascontiguousarray(label["composed_slot_to_target"]),
                    graph_path=graph_path,
                    clean_target=np.ascontiguousarray(label["clean_target_rgb"]),
                    baseline_layout=np.ascontiguousarray(graph["qap_w4_layout"]),
                )
            )
    if len(records) != 64 or len({r.source_name for r in records}) != 32:
        raise RuntimeError("expected 64 records from 32 v4 sources")
    return records


def candidate_pairs(
    matrix: np.ndarray,
    true_neighbours: np.ndarray,
    *,
    row_top_k: int,
    column_top_k: int,
    include_all_true_edges: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix)
    safe = values.copy()
    np.fill_diagonal(safe, np.inf)
    row_dest = np.argpartition(safe, row_top_k - 1, axis=1)[:, :row_top_k]
    row_source = np.repeat(np.arange(TILE_COUNT, dtype=np.int32), row_top_k)
    row_dest = row_dest.reshape(-1).astype(np.int32)
    col_source = np.argpartition(safe, column_top_k - 1, axis=0)[:column_top_k, :]
    col_dest = np.tile(np.arange(TILE_COUNT, dtype=np.int32), column_top_k)
    col_source = col_source.reshape(-1).astype(np.int32)
    if include_all_true_edges:
        queries = np.flatnonzero(true_neighbours >= 0).astype(np.int32)
        destinations = true_neighbours[queries].astype(np.int32)
        source = np.concatenate([row_source, col_source, queries])
        destination = np.concatenate([row_dest, col_dest, destinations])
    else:
        source = np.concatenate([row_source, col_source])
        destination = np.concatenate([row_dest, col_dest])
    packed = source.astype(np.int64) * TILE_COUNT + destination
    _, indices = np.unique(packed, return_index=True)
    indices.sort()
    source, destination = source[indices], destination[indices]
    labels = (true_neighbours[source] == destination).astype(np.uint8)
    return source, destination, labels


def matrix_feature_cache(matrix: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    order = np.argsort(matrix, axis=1, kind="stable")
    row_rank = np.empty_like(order, dtype=np.int16)
    row_rank[np.arange(TILE_COUNT)[:, None], order] = np.arange(
        TILE_COUNT, dtype=np.int16
    )[None, :]
    col_order = np.argsort(matrix, axis=0, kind="stable")
    column_rank = np.empty_like(col_order, dtype=np.int16)
    column_rank[col_order, np.arange(TILE_COUNT)[None, :]] = np.arange(
        TILE_COUNT, dtype=np.int16
    )[:, None]
    finite = matrix.copy()
    np.fill_diagonal(finite, np.nan)
    return {
        "row_rank": row_rank,
        "column_rank": column_rank,
        "row_min": matrix.min(axis=1),
        "column_min": matrix.min(axis=0),
        "row_mean": np.nanmean(finite, axis=1),
        "column_mean": np.nanmean(finite, axis=0),
        "row_std": np.nanstd(finite, axis=1) + 1e-6,
        "column_std": np.nanstd(finite, axis=0) + 1e-6,
    }


def edge_features(
    tiles: np.ndarray,
    matrix: np.ndarray,
    direction: int,
    source: np.ndarray,
    destination: np.ndarray,
    *,
    chunk_size: int = 16384,
) -> np.ndarray:
    tiles_f = np.asarray(tiles, dtype=np.float32) / 255.0
    tile_mean = tiles_f.mean(axis=(1, 2))
    tile_std = tiles_f.std(axis=(1, 2))
    cache = matrix_feature_cache(matrix)
    outputs = []
    for start in range(0, len(source), chunk_size):
        s = source[start : start + chunk_size]
        d = destination[start : start + chunk_size]
        cost = matrix[s, d].astype(np.float32)
        if direction == 0:
            a0 = tiles_f[s, :, -1, :]
            b0 = tiles_f[d, :, 0, :]
            band_a = tiles_f[s, :, -4:, :][:, :, ::-1, :]
            band_b = tiles_f[d, :, :4, :]
            grad_a = tiles_f[s, :, -1, :] - tiles_f[s, :, -2, :]
            grad_b = tiles_f[d, :, 1, :] - tiles_f[d, :, 0, :]
        else:
            a0 = tiles_f[s, -1, :, :]
            b0 = tiles_f[d, 0, :, :]
            band_a = tiles_f[s, -4:, :, :][:, ::-1, :, :]
            band_b = tiles_f[d, :4, :, :]
            grad_a = tiles_f[s, -1, :, :] - tiles_f[s, -2, :, :]
            grad_b = tiles_f[d, 1, :, :] - tiles_f[d, 0, :, :]
        boundary = a0 - b0
        abs_boundary = np.abs(boundary)
        values = np.column_stack(
            [
                np.full(len(s), float(direction), dtype=np.float32),
                cost,
                cache["row_rank"][s, d].astype(np.float32) / (TILE_COUNT - 1),
                cache["column_rank"][s, d].astype(np.float32) / (TILE_COUNT - 1),
                cost - cache["row_min"][s],
                cost - cache["column_min"][d],
                (cost - cache["row_mean"][s]) / cache["row_std"][s],
                (cost - cache["column_mean"][d]) / cache["column_std"][d],
                abs_boundary.mean(axis=(1, 2)),
                np.sqrt(np.mean(boundary * boundary, axis=(1, 2))),
                np.abs(band_a - band_b).mean(axis=(1, 2, 3)),
                np.abs(grad_a - grad_b).mean(axis=(1, 2)),
                np.abs(boundary.mean(axis=(1, 2))),
                np.abs(tile_mean[s] - tile_mean[d]).mean(axis=1),
                np.abs(tile_std[s] - tile_std[d]).mean(axis=1),
            ]
        ).astype(np.float32)
        outputs.append(values)
    return np.concatenate(outputs, axis=0)


def record_examples(
    record: Record,
    *,
    row_top_k: int,
    column_top_k: int,
    max_negatives: int | None,
    include_all_true_edges: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    right_truth, down_truth = true_neighbour_slots(record.truth)
    feature_parts, label_parts = [], []
    for direction, matrix, neighbours in (
        (0, record.right, right_truth),
        (1, record.down, down_truth),
    ):
        source, destination, labels = candidate_pairs(
            matrix,
            neighbours,
            row_top_k=row_top_k,
            column_top_k=column_top_k,
            include_all_true_edges=include_all_true_edges,
        )
        if max_negatives is not None:
            positives = np.flatnonzero(labels == 1)
            negatives = np.flatnonzero(labels == 0)
            take = min(max_negatives // 2, len(negatives))
            negatives = rng.choice(negatives, size=take, replace=False)
            keep = np.sort(np.concatenate([positives, negatives]))
            source, destination, labels = source[keep], destination[keep], labels[keep]
        feature_parts.append(
            edge_features(record.tiles, matrix, direction, source, destination)
        )
        label_parts.append(labels)
    return np.concatenate(feature_parts), np.concatenate(label_parts)


def v4_graph_examples(record: Record) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if record.graph_path is None:
        raise ValueError("v4 graph path is missing")
    with np.load(record.graph_path, allow_pickle=False) as graph:
        direction = np.asarray(graph["candidate_direction"], dtype=np.uint8)
        source = np.asarray(graph["candidate_source"], dtype=np.int32)
        destination = np.asarray(graph["candidate_destination"], dtype=np.int32)
        origin_mask = np.asarray(graph["candidate_origin_mask"], dtype=np.uint8)
        w4_cost = np.asarray(graph["candidate_w4_cost"], dtype=np.float32)
    features = np.empty((len(direction), len(FEATURE_NAMES)), dtype=np.float32)
    for d, matrix in ((0, record.right), (1, record.down)):
        indices = np.flatnonzero(direction == d)
        features[indices] = edge_features(
            record.tiles, matrix, d, source[indices], destination[indices]
        )
    right_truth, down_truth = true_neighbour_slots(record.truth)
    labels = np.where(
        direction == 0,
        right_truth[source] == destination,
        down_truth[source] == destination,
    ).astype(np.uint8)
    inference_pool = (
        (features[:, 2] < (32.0 / (TILE_COUNT - 1)))
        | (features[:, 3] < (8.0 / (TILE_COUNT - 1)))
    )
    return features, labels, {
        "direction": direction,
        "source": source,
        "destination": destination,
        "origin_mask": origin_mask,
        "w4_cost": w4_cost,
        "inference_pool": inference_pool,
    }


def choose_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    target_precision: float,
    minimum_edges: int = 128,
) -> dict[str, float | int]:
    order = np.argsort(-probabilities, kind="stable")
    ordered_labels = labels[order]
    true_count = np.cumsum(ordered_labels, dtype=np.int64)
    count = np.arange(1, len(order) + 1, dtype=np.int64)
    precision = true_count / count
    eligible = np.flatnonzero((precision >= target_precision) & (count >= minimum_edges))
    if len(eligible) == 0:
        best = int(np.argmax(precision * np.sqrt(count)))
    else:
        # Maximize recovered true edges while satisfying the requested precision.
        best = int(eligible[np.argmax(true_count[eligible])])
    threshold = float(probabilities[order[best]])
    return {
        "threshold": threshold,
        "selected_edges": int(best + 1),
        "true_edges": int(true_count[best]),
        "precision": float(precision[best]),
        "recall_over_all_true_edges": float(true_count[best] / labels.sum()),
    }


def aggregate_binary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "edges": int(len(labels)),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def component_diagnostics(
    record: Record,
    graph: dict[str, np.ndarray],
    probabilities: np.ndarray,
    threshold: float,
    *,
    beam_width: int,
) -> dict[str, object]:
    selected = np.flatnonzero(probabilities >= threshold)
    selected = selected[graph["inference_pool"][selected]]
    selected = selected[np.argsort(-probabilities[selected], kind="stable")]
    proposals = [
        ProposedEdge(
            first=int(graph["source"][index]),
            second=int(graph["destination"][index]),
            dx=1 if int(graph["direction"][index]) == 0 else 0,
            dy=0 if int(graph["direction"][index]) == 0 else 1,
            cost=float(1.0 - probabilities[index]),
            margin=float(probabilities[index]),
            reciprocal=False,
            in_loop=bool(int(graph["origin_mask"][index]).bit_count() >= 2),
        )
        for index in selected.tolist()
    ]
    components, accepted = grow_components_with_edges(proposals)
    right_truth, down_truth = true_neighbour_slots(record.truth)

    def is_true(edge: ProposedEdge) -> bool:
        neighbours = right_truth if edge.dx else down_truth
        return int(neighbours[edge.first]) == edge.second

    true_selected = sum(is_true(proposals[i]) for i in range(len(proposals)))
    true_accepted = sum(is_true(edge) for edge in accepted)
    compatibility = CompatibilityMatrices("w4", record.right, record.down)
    grid, placed = _place_components_beam(
        components,
        compatibility,
        boundary_weight=0.05,
        beam_width=beam_width,
        beam_components=8,
        translations_per_state=8,
    )
    layout, unresolved = _complete_with_hungarian(
        grid.copy(), compatibility, boundary_weight=0.05
    )
    baseline = layout_metrics(record.baseline_layout, record.truth)
    verified = layout_metrics(layout, record.truth)
    baseline_image = predicted_image_metrics(
        record.baseline_layout, record.tiles, record.clean_target
    )
    verified_image = predicted_image_metrics(layout, record.tiles, record.clean_target)
    projections = {}
    for reference_weight in REFERENCE_WEIGHTS:
        projection = project_rigid_components_around_reference(
            components,
            accepted,
            compatibility,
            record.baseline_layout,
            selected_proposals=len(proposals),
            reference_weight=reference_weight,
            beam_width=beam_width,
            beam_components=16,
            translations_per_state=8,
        )
        projection_layout = layout_metrics(projection.position_to_slot, record.truth)
        projection_image = predicted_image_metrics(
            projection.position_to_slot, record.tiles, record.clean_target
        )
        projections[f"weight_{reference_weight:g}"] = {
            "combined_adjacency": projection_layout["combined_adjacency"],
            "largest_correct_component": projection_layout["largest_correct_component"],
            "ssim": projection_image["predicted_layout_ssim"],
            "retained_accepted_edge_fraction": projection.retained_accepted_edge_fraction,
        }
    return {
        "source_name": record.source_name,
        "panel": record.panel,
        "selected_edges": len(proposals),
        "selected_precision": float(true_selected / max(1, len(proposals))),
        "accepted_edges": len(accepted),
        "accepted_precision": float(true_accepted / max(1, len(accepted))),
        "component_sizes": sorted((len(value) for value in components), reverse=True),
        "placed_component_tiles": int(placed),
        "unresolved_before_hungarian": int(unresolved),
        "baseline": {
            "combined_adjacency": baseline["combined_adjacency"],
            "largest_correct_component": baseline["largest_correct_component"],
            "ssim": baseline_image["predicted_layout_ssim"],
        },
        "verified": {
            "combined_adjacency": verified["combined_adjacency"],
            "largest_correct_component": verified["largest_correct_component"],
            "ssim": verified_image["predicted_layout_ssim"],
        },
        "rigid_projections": projections,
    }


def mean(records: Iterable[dict[str, object]], path: tuple[str, ...]) -> float:
    values = []
    for record in records:
        value: object = record
        for key in path:
            value = value[key]  # type: ignore[index]
        values.append(float(value))
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    output = Path(args.output_root)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"output is not empty; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rng = np.random.default_rng(args.seed)
    training_records = load_training_records(Path(args.train_root))
    v4_records = load_v4_records(Path(args.v4_fixture_root), Path(args.v4_graph_root))
    training_sources = list(dict.fromkeys(record.source_name for record in training_records))
    calibration_sources = set(training_sources[-args.calibration_sources :])
    fit_records = [r for r in training_records if r.source_name not in calibration_sources]
    calibration_records = [r for r in training_records if r.source_name in calibration_sources]
    v4_sources = {r.source_name for r in v4_records}
    if ({r.source_name for r in training_records} & v4_sources):
        raise RuntimeError("training/v4 whole-source leakage detected")

    fit_x, fit_y = [], []
    for index, record in enumerate(fit_records, 1):
        x, y = record_examples(
            record,
            row_top_k=args.row_top_k,
            column_top_k=args.column_top_k,
            max_negatives=args.hard_negatives_per_record,
            include_all_true_edges=True,
            rng=rng,
        )
        fit_x.append(x)
        fit_y.append(y)
        print(json.dumps({"stage": "features_fit", "done": index, "total": len(fit_records)}), flush=True)
    fit_x_array, fit_y_array = np.concatenate(fit_x), np.concatenate(fit_y)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=args.max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=64,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(fit_x_array, fit_y_array)

    calibration_x, calibration_y = [], []
    for index, record in enumerate(calibration_records, 1):
        x, y = record_examples(
            record,
            row_top_k=args.row_top_k,
            column_top_k=args.column_top_k,
            max_negatives=None,
            include_all_true_edges=False,
            rng=rng,
        )
        calibration_x.append(x)
        calibration_y.append(y)
        print(json.dumps({"stage": "features_calibration", "done": index, "total": len(calibration_records)}), flush=True)
    calibration_x_array = np.concatenate(calibration_x)
    calibration_y_array = np.concatenate(calibration_y)
    calibration_probability = model.predict_proba(calibration_x_array)[:, 1]
    threshold = choose_threshold(
        calibration_probability,
        calibration_y_array,
        target_precision=args.target_precision,
    )

    v4_feature_metrics, v4_component_records = [], []
    all_v4_labels, all_v4_probability = [], []
    for index, record in enumerate(v4_records, 1):
        x, y, graph = v4_graph_examples(record)
        probability = model.predict_proba(x)[:, 1]
        pool = graph["inference_pool"]
        all_v4_labels.append(y[pool])
        all_v4_probability.append(probability[pool])
        binary = aggregate_binary(y[pool], probability[pool])
        binary.update({"source_name": record.source_name, "panel": record.panel})
        v4_feature_metrics.append(binary)
        v4_component_records.append(
            component_diagnostics(
                record,
                graph,
                probability,
                float(threshold["threshold"]),
                beam_width=args.beam_width,
            )
        )
        print(json.dumps({"stage": "v4", "done": index, "total": len(v4_records)}), flush=True)

    all_v4_y = np.concatenate(all_v4_labels)
    all_v4_p = np.concatenate(all_v4_probability)
    report = {
        "schema_version": 1,
        "kind": "leakage_safe_candidate_edge_verifier_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - started,
        "config": vars(args),
        "feature_names": FEATURE_NAMES,
        "source_split": {
            "fit_sources": sorted({r.source_name for r in fit_records}),
            "calibration_sources": sorted(calibration_sources),
            "v4_sources": sorted(v4_sources),
            "pairwise_disjoint": True,
        },
        "fit": {
            "records": len(fit_records),
            "examples": len(fit_y_array),
            "positives": int(fit_y_array.sum()),
        },
        "calibration": {
            **aggregate_binary(calibration_y_array, calibration_probability),
            "threshold_selection": threshold,
        },
        "v4_binary": aggregate_binary(all_v4_y, all_v4_p),
        "v4_assembly": {
            "records": len(v4_component_records),
            "mean_selected_precision": mean(v4_component_records, ("selected_precision",)),
            "mean_accepted_precision": mean(v4_component_records, ("accepted_precision",)),
            "mean_largest_component": float(
                np.mean([record["component_sizes"][0] for record in v4_component_records])
            ),
            "mean_baseline_adjacency": mean(v4_component_records, ("baseline", "combined_adjacency")),
            "mean_verified_adjacency": mean(v4_component_records, ("verified", "combined_adjacency")),
            "mean_adjacency_delta": mean(v4_component_records, ("verified", "combined_adjacency"))
            - mean(v4_component_records, ("baseline", "combined_adjacency")),
            "mean_baseline_ssim": mean(v4_component_records, ("baseline", "ssim")),
            "mean_verified_ssim": mean(v4_component_records, ("verified", "ssim")),
            "mean_ssim_delta": mean(v4_component_records, ("verified", "ssim"))
            - mean(v4_component_records, ("baseline", "ssim")),
            "rigid_projections": {
                f"weight_{weight:g}": {
                    "mean_adjacency": mean(
                        v4_component_records,
                        ("rigid_projections", f"weight_{weight:g}", "combined_adjacency"),
                    ),
                    "mean_ssim": mean(
                        v4_component_records,
                        ("rigid_projections", f"weight_{weight:g}", "ssim"),
                    ),
                    "mean_retained_accepted_edge_fraction": mean(
                        v4_component_records,
                        (
                            "rigid_projections",
                            f"weight_{weight:g}",
                            "retained_accepted_edge_fraction",
                        ),
                    ),
                }
                for weight in REFERENCE_WEIGHTS
            },
        },
        "v4_binary_records": v4_feature_metrics,
        "v4_assembly_records": v4_component_records,
    }
    model_path = output / "candidate_edge_verifier.joblib"
    report_path = output / "report.json"
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, model_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "calibration_threshold": threshold,
        "v4_binary": report["v4_binary"],
        "v4_assembly": report["v4_assembly"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
