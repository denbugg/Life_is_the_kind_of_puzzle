#!/usr/bin/env python3
"""Leakage-safe dual-sided LambdaRank gate on the seven-origin edge recipe.

This is deliberately a retrieval-only first gate.  It trains one ranker for
outgoing tile-side queries and one for incoming tile-side queries, then combines
their within-query percentile ranks.  No V4 labels or final assembly holdout are
opened unless this cheaper source-disjoint gate succeeds.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from PIL import Image

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
for value in (SCRIPT_ROOT, REPO_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.geometry import TILE_COUNT, true_neighbour_slots, validate_permutation
from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer
from train_binary_edge_verifier import (
    CandidateGraph,
    PreparedSource,
    candidate_features,
    candidate_labels,
    feature_names as legacy_feature_names,
    prepare_source,
)


PANELS = ("primary_kornia", "independent_libjpeg")
GRID = 24
ORIGIN_BITS = {
    "c1_out32": 1,
    "hbt_out32": 2,
    "c1_in8": 4,
    "hbt_in8": 8,
    "softcycle": 16,
    "qap_w4": 32,
    "qap_w1": 64,
}
NEW_ORIGIN_NAMES = ("softcycle", "qap_w4", "qap_w1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument(
        "--denoiser", default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
    )
    parser.add_argument(
        "--embedding-checkpoint",
        default="runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt",
    )
    parser.add_argument("--legacy-hgb", required=True)
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument(
        "--quarantine", default="configs/denoise_validation_quarantine_v1.json"
    )
    parser.add_argument("--fit-sources", type=int, default=24)
    parser.add_argument("--calibration-offset", type=int, default=368)
    parser.add_argument("--calibration-sources", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected RGB shape: {path}: {values.shape}")
    return values


def stable_candidates(
    matrix: np.ndarray, *, outgoing: int = 32, incoming: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix)
    row_order = np.argsort(values, axis=1, kind="stable")
    column_order = np.argsort(values, axis=0, kind="stable")
    rows: list[tuple[int, int]] = []
    columns: list[tuple[int, int]] = []
    for source in range(TILE_COUNT):
        selected = [int(x) for x in row_order[source] if int(x) != source]
        rows.extend((source, destination) for destination in selected[:outgoing])
    for destination in range(TILE_COUNT):
        selected = [
            int(x) for x in column_order[:, destination] if int(x) != destination
        ]
        columns.extend((source, destination) for source in selected[:incoming])
    return np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32)


def layout_edges(layout: np.ndarray, direction: int) -> np.ndarray:
    grid = validate_permutation(layout).reshape(GRID, GRID)
    if direction == 0:
        return np.stack([grid[:, :-1].ravel(), grid[:, 1:].ravel()], axis=1).astype(
            np.int32
        )
    if direction == 1:
        return np.stack([grid[:-1, :].ravel(), grid[1:, :].ravel()], axis=1).astype(
            np.int32
        )
    raise ValueError("direction must be 0 or 1")


def seven_origin_graph(
    scores: dict[str, CompatibilityMatrices], layouts: dict[str, np.ndarray]
) -> CandidateGraph:
    edges: dict[tuple[int, int, int], int] = {}

    def add(direction: int, pairs: np.ndarray, origin: str) -> None:
        bit = ORIGIN_BITS[origin]
        for source, destination in np.asarray(pairs).tolist():
            source, destination = int(source), int(destination)
            if source == destination:
                raise RuntimeError("self-edge reached candidate union")
            key = (direction, source, destination)
            edges[key] = edges.get(key, 0) | bit

    for direction, side in ((0, "right"), (1, "down")):
        for prefix in ("c1", "hbt"):
            outgoing, incoming = stable_candidates(getattr(scores[prefix], side))
            add(direction, outgoing, f"{prefix}_out32")
            add(direction, incoming, f"{prefix}_in8")
        for name in NEW_ORIGIN_NAMES:
            add(direction, layout_edges(layouts[name], direction), name)
    ordered = sorted(edges)
    return CandidateGraph(
        direction=np.asarray([key[0] for key in ordered], dtype=np.uint8),
        source=np.asarray([key[1] for key in ordered], dtype=np.int32),
        destination=np.asarray([key[2] for key in ordered], dtype=np.int32),
        origin_mask=np.asarray([edges[key] for key in ordered], dtype=np.uint8),
    )


def corrected_features(
    scores: dict[str, CompatibilityMatrices], graph: CandidateGraph
) -> tuple[np.ndarray, np.ndarray]:
    legacy = candidate_features(scores, graph)
    popcount_index = list(legacy_feature_names()).index("origin_popcount")
    legacy[:, popcount_index] = np.asarray(
        [int(int(value) & 15).bit_count() / 4.0 for value in graph.origin_mask],
        dtype=np.float32,
    )
    corrected = legacy.copy()
    corrected[:, popcount_index] = np.asarray(
        [int(value).bit_count() / 7.0 for value in graph.origin_mask],
        dtype=np.float32,
    )
    extra = np.column_stack(
        [
            ((graph.origin_mask & ORIGIN_BITS[name]) != 0).astype(np.float32)
            for name in NEW_ORIGIN_NAMES
        ]
    )
    return legacy, np.concatenate([corrected, extra], axis=1).astype(np.float32)


def build_record(
    name: str,
    panel: str,
    *,
    args: argparse.Namespace,
    restorer: Any,
    embedding: Any,
    device: Any,
) -> dict[str, Any]:
    panel_seed = per_source_seed(
        args.seed, f"full-union-tabular-{panel}", name, 0
    )
    prepared = prepare_source(
        name,
        panel,
        panel_seed,
        args=args,
        restorer=restorer,
        embedding_model=embedding,
        device=device,
    )
    soft = soft_cycle_component_solver(
        prepared.scores["hbt"],
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        loop_weight=1.0,
        reciprocal_weight=0.35,
    )
    initial = validate_permutation(soft.position_to_slot)
    qap_seed = per_source_seed(args.seed, "dual-lambdarank-qap", name, panel)
    layouts = {"softcycle": initial.copy()}
    for score_name in ("w4", "w1"):
        result = directional_qap(
            prepared.scores[score_name],
            initial=initial.copy(),
            iterations=25,
            restarts=2,
            seed=int(qap_seed),
            boundary_weight=0.05,
            initial_weight=0.75,
            noisy_components=3,
            noise_scale=1.0,
            refine_swaps=8,
            refine_weak_cells=32,
        )
        layouts[f"qap_{score_name}"] = validate_permutation(result.position_to_slot)
    graph = seven_origin_graph(prepared.scores, layouts)
    legacy, features = corrected_features(prepared.scores, graph)
    labels = candidate_labels(graph, prepared.truth).astype(np.uint8)
    return {
        "name": name,
        "panel": panel,
        "panel_seed": int(panel_seed),
        "graph": graph,
        "features": features,
        "legacy_features": legacy,
        "labels": labels,
        "truth": prepared.truth.copy(),
        "candidate_recall": float(labels.sum() / 1104.0),
        "hbt_cost": np.where(
            graph.direction == 0,
            prepared.scores["hbt"].right[graph.source, graph.destination],
            prepared.scores["hbt"].down[graph.source, graph.destination],
        ).astype(np.float32),
        "w4_cost": np.where(
            graph.direction == 0,
            prepared.scores["w4"].right[graph.source, graph.destination],
            prepared.scores["w4"].down[graph.source, graph.destination],
        ).astype(np.float32),
    }


def grouped_dataset(
    records: list[dict[str, Any]], *, incoming: bool
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    groups: list[int] = []
    for record in records:
        graph: CandidateGraph = record["graph"]
        group_node = graph.destination if incoming else graph.source
        keys = graph.direction.astype(np.int64) * TILE_COUNT + group_node.astype(np.int64)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
        stops = np.r_[starts[1:], len(order)]
        for start, stop in zip(starts, stops, strict=True):
            indices = order[start:stop]
            # Boundary/null groups and candidate-miss groups are counted during
            # evaluation but cannot define a LambdaRank query.
            if int(record["labels"][indices].sum()) != 1:
                continue
            feature_parts.append(record["features"][indices])
            label_parts.append(record["labels"][indices])
            groups.append(int(len(indices)))
    if not groups:
        raise RuntimeError("no positive-present ranking groups")
    return np.concatenate(feature_parts), np.concatenate(label_parts), groups


def train_ranker(
    records: list[dict[str, Any]], *, incoming: bool, args: argparse.Namespace
) -> lgb.LGBMRanker:
    x, y, groups = grouped_dataset(records, incoming=incoming)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_at=[1, 5],
        lambdarank_truncation_level=5,
        label_gain=[0, 1],
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=args.n_estimators,
        min_child_samples=64,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=args.seed + int(incoming),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(x, y, group=groups)
    del x, y, groups
    gc.collect()
    return model


def percentile_by_group(
    values: np.ndarray, direction: np.ndarray, nodes: np.ndarray
) -> np.ndarray:
    output = np.empty(len(values), dtype=np.float32)
    keys = direction.astype(np.int64) * TILE_COUNT + nodes.astype(np.int64)
    for key in np.unique(keys):
        indices = np.flatnonzero(keys == key)
        order = np.argsort(-values[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float32)
        ranks[order] = np.arange(len(indices), dtype=np.float32)
        output[indices] = 1.0 - ranks / max(1, len(indices) - 1)
    return output


def retrieval_metrics(record: dict[str, Any], score: np.ndarray) -> dict[str, float]:
    graph: CandidateGraph = record["graph"]
    truth_indices = np.flatnonzero(record["labels"] > 0)
    rank_lookup: dict[tuple[int, int, int], int] = {}
    for direction in (0, 1):
        for source in range(TILE_COUNT):
            indices = np.flatnonzero(
                (graph.direction == direction) & (graph.source == source)
            )
            order = indices[np.argsort(-score[indices], kind="stable")]
            for rank, index in enumerate(order.tolist(), 1):
                rank_lookup[
                    (
                        int(graph.direction[index]),
                        int(graph.source[index]),
                        int(graph.destination[index]),
                    )
                ] = rank
    right, down = true_neighbour_slots(record["truth"])
    ranks: list[float] = []
    for direction, neighbours in ((0, right), (1, down)):
        for source in np.flatnonzero(neighbours >= 0).tolist():
            ranks.append(
                float(rank_lookup.get((direction, source, int(neighbours[source])), np.inf))
            )
    values = np.asarray(ranks, dtype=np.float64)
    finite = np.isfinite(values)
    top1_destinations = []
    for direction, neighbours in ((0, right), (1, down)):
        chosen = []
        for source in np.flatnonzero(neighbours >= 0).tolist():
            indices = np.flatnonzero(
                (graph.direction == direction) & (graph.source == source)
            )
            chosen.append(int(graph.destination[indices[np.argmax(score[indices])]]))
        top1_destinations.append(len(chosen) - len(set(chosen)))
    return {
        "candidate_recall": float(finite.mean()),
        "recall_at_1": float(np.mean(values <= 1)),
        "recall_at_5": float(np.mean(values <= 5)),
        "mrr": float(np.mean(np.where(finite, 1.0 / values, 0.0))),
        "top1_destination_collisions": float(sum(top1_destinations)),
        "candidate_true_edges": float(len(truth_indices)),
    }


def score_record(
    record: dict[str, Any], outgoing: lgb.LGBMRanker, incoming: lgb.LGBMRanker, hgb: Any
) -> dict[str, Any]:
    graph: CandidateGraph = record["graph"]
    out_raw = outgoing.predict(record["features"])
    in_raw = incoming.predict(record["features"])
    out_rank = percentile_by_group(out_raw, graph.direction, graph.source)
    in_rank = percentile_by_group(in_raw, graph.direction, graph.destination)
    combined = 0.5 * (out_rank + in_rank)
    hgb_probability = hgb.predict_proba(record["legacy_features"])[:, 1]
    scores = {
        "hbt": -record["hbt_cost"],
        "w4": -record["w4_cost"],
        "legacy_hgb": hgb_probability,
        "dual_lambdarank": combined,
    }
    return {name: retrieval_metrics(record, value) for name, value in scores.items()}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for panel in PANELS:
        selected = [record for record in records if record["panel"] == panel]
        output[panel] = {}
        for method in ("hbt", "w4", "legacy_hgb", "dual_lambdarank"):
            output[panel][method] = {
                metric: float(np.mean([record["metrics"][method][metric] for record in selected]))
                for metric in selected[0]["metrics"][method]
            }
    return output


def gate(summary: dict[str, Any]) -> dict[str, Any]:
    per_panel: dict[str, Any] = {}
    for panel in PANELS:
        candidate = summary[panel]["dual_lambdarank"]
        strongest = {
            metric: max(
                summary[panel][baseline][metric]
                for baseline in ("hbt", "w4", "legacy_hgb")
            )
            for metric in ("recall_at_1", "recall_at_5", "mrr")
        }
        per_panel[panel] = {
            "delta_recall_at_1_vs_strongest": candidate["recall_at_1"]
            - strongest["recall_at_1"],
            "delta_recall_at_5_vs_strongest": candidate["recall_at_5"]
            - strongest["recall_at_5"],
            "delta_mrr_vs_strongest": candidate["mrr"] - strongest["mrr"],
            "candidate_recall_identity": abs(
                candidate["candidate_recall"] - summary[panel]["w4"]["candidate_recall"]
            )
            < 1e-12,
            "candidate_recall_ge_0.65": candidate["candidate_recall"] >= 0.65,
            "r1_delta_ge_0.01": candidate["recall_at_1"]
            - strongest["recall_at_1"]
            >= 0.01,
            "r5_nonregression": candidate["recall_at_5"] >= strongest["recall_at_5"],
            "mrr_positive": candidate["mrr"] > strongest["mrr"],
            "collisions_nonincrease_vs_w4": candidate["top1_destination_collisions"]
            <= summary[panel]["w4"]["top1_destination_collisions"],
        }
    passed = all(
        values["candidate_recall_identity"]
        and values["candidate_recall_ge_0.65"]
        and values["r1_delta_ge_0.01"]
        and values["r5_nonregression"]
        and values["mrr_positive"]
        and values["collisions_nonincrease_vs_w4"]
        for values in per_panel.values()
    )
    return {
        "per_panel": per_panel,
        "retrieval_gate_passed": passed,
        "open_external_assembly_gate": passed,
        "safe_for_submission": False,
    }


def main() -> None:
    args = parse_args()
    if min(args.fit_sources, args.calibration_sources, args.n_estimators) <= 0:
        raise SystemExit("source counts and estimator count must be positive")
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit("output root is not empty; pass --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    started = time.time()
    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device, state="ema"
    )
    embedding, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    for model in (restorer, embedding):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    names = source_names_for_split(
        "edge_development",
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )
    fit_names = names[: args.fit_sources]
    calibration_names = names[
        args.calibration_offset : args.calibration_offset + args.calibration_sources
    ]
    if len(fit_names) != args.fit_sources or len(calibration_names) != args.calibration_sources:
        raise RuntimeError("requested source slice unavailable")
    if set(fit_names) & set(calibration_names):
        raise RuntimeError("fit/calibration whole-source overlap")
    fit_records: list[dict[str, Any]] = []
    for index, name in enumerate(fit_names, 1):
        for panel in PANELS:
            fit_records.append(
                build_record(
                    name,
                    panel,
                    args=args,
                    restorer=restorer,
                    embedding=embedding,
                    device=device,
                )
            )
        print(json.dumps({"stage": "fit_features", "done": index, "total": len(fit_names)}), flush=True)
    outgoing = train_ranker(fit_records, incoming=False, args=args)
    incoming = train_ranker(fit_records, incoming=True, args=args)
    del fit_records
    gc.collect()
    hgb_payload = joblib.load(args.legacy_hgb)
    if not isinstance(hgb_payload, dict) or "model" not in hgb_payload:
        raise RuntimeError("legacy HGB artifact is not the pinned producer payload")
    hgb = hgb_payload["model"]
    if not hasattr(hgb, "predict_proba") or getattr(hgb, "n_features_in_", None) != len(
        legacy_feature_names()
    ):
        raise RuntimeError("legacy HGB estimator/schema mismatch")
    if list(hgb_payload.get("feature_names", [])) != list(legacy_feature_names()):
        raise RuntimeError("legacy HGB feature-name order mismatch")
    calibration_records: list[dict[str, Any]] = []
    for index, name in enumerate(calibration_names, 1):
        for panel in PANELS:
            record = build_record(
                name,
                panel,
                args=args,
                restorer=restorer,
                embedding=embedding,
                device=device,
            )
            metrics = score_record(record, outgoing, incoming, hgb)
            calibration_records.append(
                {
                    "name": name,
                    "panel": panel,
                    "panel_seed": record["panel_seed"],
                    "candidate_edges": int(len(record["labels"])),
                    "metrics": metrics,
                }
            )
        print(json.dumps({"stage": "calibration", "done": index, "total": len(calibration_names)}), flush=True)
    outgoing_path = output_root / "outgoing_lambdarank.txt"
    incoming_path = output_root / "incoming_lambdarank.txt"
    outgoing.booster_.save_model(str(outgoing_path))
    incoming.booster_.save_model(str(incoming_path))
    summary = aggregate(calibration_records)
    gate_result = gate(summary)
    report = {
        "schema_version": 1,
        "kind": "dual_sided_seven_origin_lambdarank_retrieval_gate",
        "status": "pass_retrieval_open_assembly" if gate_result["retrieval_gate_passed"] else "stop_retrieval_no_signal",
        "args": vars(args),
        "feature_names": legacy_feature_names() + [
            "origin_softcycle",
            "origin_qap_w4",
            "origin_qap_w1",
        ],
        "origin_bits": ORIGIN_BITS,
        "split": {
            "fit_names": fit_names,
            "calibration_names": calibration_names,
            "calibration_offset": args.calibration_offset,
            "whole_source_disjoint": True,
            "v4_paths_constructed_or_opened": False,
            "screen_role": "fresh source-disjoint retrieval development screen only",
        },
        "models": {
            "outgoing": {"path": str(outgoing_path), "sha256": sha256(outgoing_path)},
            "incoming": {"path": str(incoming_path), "sha256": sha256(incoming_path)},
            "legacy_hgb_sha256": sha256(args.legacy_hgb),
        },
        "denoiser_metadata": denoiser_metadata,
        "embedding_metadata": embedding_metadata,
        "calibration_records": calibration_records,
        "summary": summary,
        "gate": gate_result,
        "seconds": time.time() - started,
        "safe_for_submission": False,
    }
    report_path = output_root / "dual_lambdarank_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "dual_lambdarank_complete",
                "status": report["status"],
                "report": str(report_path),
                "report_sha256": sha256(report_path),
                "gate": gate_result,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
