#!/usr/bin/env python3
"""Evaluate frozen HGB rank residual as a local QAP refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, SCRIPT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.compatibility import CompatibilityMatrices, fuse_ranked_scores
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.qap import directional_qap
from train_binary_edge_verifier import CandidateGraph, candidate_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="runs/assembly_v1/full_union_tabular/v1/full_union_tabular.joblib",
    )
    parser.add_argument("--hgb-weight", type=float, default=0.25)
    parser.add_argument(
        "--fixture-root",
        default="runs/assembly_v1/candidate_graph_oracle_fixtures_v4_6c0fe4e8524ce39d830d9a5bee118d8b",
    )
    parser.add_argument(
        "--graph-root",
        default="runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_readback/candidate_graph_oracle_v4_phase_a/finalized",
    )
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layout-root")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hgb_compatibility(
    model: object,
    graph: CandidateGraph,
    features: np.ndarray,
) -> CompatibilityMatrices:
    probability = model.predict_proba(features)[:, 1]
    matrices = []
    for direction in (0, 1):
        matrix = np.full((576, 576), 1e3, dtype=np.float32)
        indices = np.flatnonzero(graph.direction == direction)
        matrix[graph.source[indices], graph.destination[indices]] = -probability[
            indices
        ].astype(np.float32)
        np.fill_diagonal(matrix, np.inf)
        matrices.append(matrix)
    return CompatibilityMatrices("hgb_candidate_probability", matrices[0], matrices[1])


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    layout_root = Path(args.layout_root) if args.layout_root else None
    if layout_root is not None:
        layout_root.mkdir(parents=True, exist_ok=True)
    payload = joblib.load(args.model)
    model = payload["model"]
    fixture_root = Path(args.fixture_root)
    graph_root = Path(args.graph_root)
    manifest = json.loads(
        (fixture_root / "fixture_label/fixture_label_manifest.json").read_text()
    )
    record_metadata = manifest["records"]
    if args.max_records is not None:
        record_metadata = record_metadata[: args.max_records]
    records = []
    started = time.time()
    for record_index, meta in enumerate(record_metadata, 1):
        opaque_id = str(meta["opaque_id"])
        graph_path = graph_root / "artifacts" / f"{opaque_id}.graph.npz"
        label_path = fixture_root / "fixture_label/records" / f"{opaque_id}.npz"
        input_path = fixture_root / "fixture_input/records" / f"{opaque_id}.npz"
        with np.load(graph_path, allow_pickle=False) as graph_values, np.load(
            label_path, allow_pickle=False
        ) as label_values, np.load(input_path, allow_pickle=False) as input_values:
            graph = CandidateGraph(
                direction=np.asarray(graph_values["candidate_direction"]),
                source=np.asarray(graph_values["candidate_source"], dtype=np.int32),
                destination=np.asarray(
                    graph_values["candidate_destination"], dtype=np.int32
                ),
                origin_mask=np.asarray(graph_values["candidate_origin_mask"]),
            )
            scores = {
                name: CompatibilityMatrices(
                    name,
                    np.asarray(graph_values[f"{name}_right"]),
                    np.asarray(graph_values[f"{name}_down"]),
                )
                for name in ("c1", "hbt", "w1", "w4")
            }
            denoised_tiles = np.asarray(graph_values["denoised_tiles"])
            baseline_layout = np.asarray(graph_values["qap_w4_layout"])
            truth = np.asarray(label_values["composed_slot_to_target"])
            clean = np.asarray(label_values["clean_target_rgb"])
            qap_seed = int(input_values["qap_seed"])
        features = candidate_features(scores, graph)
        hgb = hgb_compatibility(model, graph, features)
        bank = {"c1": scores["c1"], "hbt": scores["hbt"], "hgb": hgb}
        residual_score = fuse_ranked_scores(
            bank,
            names=["c1", "hbt", "hgb"],
            weights={"hbt": 4.0, "hgb": args.hgb_weight},
            name="C1_HBTw4_HGB_residual",
        )
        result = directional_qap(
            residual_score,
            initial=baseline_layout,
            iterations=args.iterations,
            restarts=args.restarts,
            seed=qap_seed,
            boundary_weight=0.05,
            initial_weight=0.75,
            noisy_components=3,
            noise_scale=1.0,
            refine_swaps=8,
            refine_weak_cells=32,
        )
        baseline_layout_metrics = layout_metrics(baseline_layout, truth)
        residual_layout_metrics = layout_metrics(result.position_to_slot, truth)
        baseline_image = predicted_image_metrics(baseline_layout, denoised_tiles, clean)
        residual_image = predicted_image_metrics(
            result.position_to_slot, denoised_tiles, clean
        )
        if layout_root is not None:
            np.save(layout_root / f"{opaque_id}.npy", result.position_to_slot)
        records.append(
            {
                "opaque_id": opaque_id,
                "source_name": meta["source_name"],
                "panel": meta["panel"],
                "baseline": {
                    "adjacency": baseline_layout_metrics["combined_adjacency"],
                    "ssim": baseline_image["predicted_layout_ssim"],
                },
                "residual": {
                    "adjacency": residual_layout_metrics["combined_adjacency"],
                    "ssim": residual_image["predicted_layout_ssim"],
                    "qap_objective": float(result.objective),
                    "qap_restart": int(result.restart),
                },
            }
        )
        print(
            json.dumps(
                {"stage": "qap", "done": record_index, "total": len(record_metadata)}
            ),
            flush=True,
        )

    def mean(kind: str, metric: str) -> float:
        return float(np.mean([record[kind][metric] for record in records]))

    panel_summaries = {}
    for panel in ("primary_kornia", "independent_libjpeg"):
        selected = [record for record in records if record["panel"] == panel]
        panel_summaries[panel] = {
            "records": len(selected),
            "mean_baseline_ssim": float(
                np.mean([record["baseline"]["ssim"] for record in selected])
            ),
            "mean_residual_ssim": float(
                np.mean([record["residual"]["ssim"] for record in selected])
            ),
            "mean_ssim_delta": float(
                np.mean(
                    [
                        record["residual"]["ssim"] - record["baseline"]["ssim"]
                        for record in selected
                    ]
                )
            ),
            "mean_adjacency_delta": float(
                np.mean(
                    [
                        record["residual"]["adjacency"]
                        - record["baseline"]["adjacency"]
                        for record in selected
                    ]
                )
            ),
        }
    report = {
        "schema_version": 1,
        "kind": "full_union_hgb_qap_residual_evaluation",
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "hgb_weight": args.hgb_weight,
        "selection_basis": "edge_development calibration MRR",
        "qap": {
            "iterations": args.iterations,
            "restarts": args.restarts,
            "initial": "frozen qap_w4 layout",
        },
        "aggregate": {
            "records": len(records),
            "mean_baseline_ssim": mean("baseline", "ssim"),
            "mean_residual_ssim": mean("residual", "ssim"),
            "mean_ssim_delta": mean("residual", "ssim")
            - mean("baseline", "ssim"),
            "mean_baseline_adjacency": mean("baseline", "adjacency"),
            "mean_residual_adjacency": mean("residual", "adjacency"),
            "mean_adjacency_delta": mean("residual", "adjacency")
            - mean("baseline", "adjacency"),
            "ssim_wins": sum(
                record["residual"]["ssim"] > record["baseline"]["ssim"]
                for record in records
            ),
            "adjacency_wins": sum(
                record["residual"]["adjacency"]
                > record["baseline"]["adjacency"]
                for record in records
            ),
        },
        "panels": panel_summaries,
        "records": records,
        "seconds": time.time() - started,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
