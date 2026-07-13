#!/usr/bin/env python3
"""Evaluate a frozen binary edge verifier on the disjoint v4 candidate graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puzzle_assembly.binary_edge_verifier import load_binary_edge_verifier
from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.components import (
    ProposedEdge,
    _complete_with_hungarian,
    _place_components_beam,
    grow_components_with_edges,
)
from puzzle_assembly.geometry import TILE_COUNT
from puzzle_assembly.learned import seam_pair_patches
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.qap import directional_qap
from scripts.train_binary_edge_verifier import (
    CandidateGraph,
    binary_metrics,
    candidate_features,
    candidate_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--fixture-root",
        default="runs/assembly_v1/candidate_graph_oracle_fixtures_v4_6c0fe4e8524ce39d830d9a5bee118d8b",
    )
    parser.add_argument(
        "--graph-root",
        default="runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_readback/candidate_graph_oracle_v4_phase_a/finalized",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--qap-iterations", type=int, default=25)
    parser.add_argument("--qap-restarts", type=int, default=2)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensor_tiles(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    )


@torch.inference_mode()
def score_graph(
    model: torch.nn.Module,
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    graph: CandidateGraph,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    side_band: int,
) -> np.ndarray:
    raw = tensor_tiles(raw_tiles, device)
    denoised = tensor_tiles(denoised_tiles, device)
    model.eval()
    output = []
    for start in range(0, len(graph.direction), batch_size):
        stop = min(start + batch_size, len(graph.direction))
        indices = np.arange(start, stop)
        first = torch.as_tensor(graph.source[indices], device=device, dtype=torch.long)
        second = torch.as_tensor(
            graph.destination[indices], device=device, dtype=torch.long
        )
        direction = torch.as_tensor(
            graph.direction[indices], device=device, dtype=torch.long
        )
        raw_patch = seam_pair_patches(
            raw, first, second, direction, side_band=side_band
        )
        denoised_patch = seam_pair_patches(
            denoised, first, second, direction, side_band=side_band
        )
        tabular = torch.as_tensor(features[indices], device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(raw_patch, denoised_patch, tabular)
        output.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(output)


def proposals_from_threshold(
    graph: CandidateGraph, probability: np.ndarray, threshold: float
) -> list[ProposedEdge]:
    indices = np.flatnonzero(probability >= threshold)
    indices = indices[np.argsort(-probability[indices], kind="stable")]
    return [
        ProposedEdge(
            first=int(graph.source[index]),
            second=int(graph.destination[index]),
            dx=1 if int(graph.direction[index]) == 0 else 0,
            dy=0 if int(graph.direction[index]) == 0 else 1,
            cost=float(1.0 - probability[index]),
            margin=float(probability[index]),
            reciprocal=False,
            in_loop=int(graph.origin_mask[index]).bit_count() >= 2,
        )
        for index in indices.tolist()
    ]


def accepted_precision(
    accepted: list[ProposedEdge], graph: CandidateGraph, labels: np.ndarray
) -> float:
    lookup = {
        (int(direction), int(source), int(destination)): bool(label)
        for direction, source, destination, label in zip(
            graph.direction,
            graph.source,
            graph.destination,
            labels,
            strict=True,
        )
    }
    correct = sum(
        lookup[(0 if edge.dx else 1, edge.first, edge.second)] for edge in accepted
    )
    return float(correct / max(1, len(accepted)))


def solve_components(
    components: list[dict[int, tuple[int, int]]],
    compatibility: CompatibilityMatrices,
    *,
    qap_seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    grid, _ = _place_components_beam(
        components,
        compatibility,
        boundary_weight=0.05,
        beam_width=args.beam_width,
        beam_components=8,
        translations_per_state=8,
    )
    initial, _ = _complete_with_hungarian(
        grid.copy(), compatibility, boundary_weight=0.05
    )
    return directional_qap(
        compatibility,
        initial=initial,
        iterations=args.qap_iterations,
        restarts=args.qap_restarts,
        seed=qap_seed,
        boundary_weight=0.05,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=8,
        refine_weak_cells=32,
    ).position_to_slot


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, names, metadata = load_binary_edge_verifier(
        args.checkpoint, device=device
    )
    expected_names = metadata.get("feature_names")
    if expected_names is not None and list(expected_names) != names:
        raise RuntimeError("checkpoint feature metadata drift")
    frontier = metadata.get("best_precision_frontier")
    if not isinstance(frontier, dict) or "threshold" not in frontier:
        raise RuntimeError("checkpoint lacks frozen calibration threshold")
    threshold = float(frontier["threshold"])
    fixture_root = Path(args.fixture_root)
    graph_root = Path(args.graph_root)
    manifest = json.loads(
        (fixture_root / "fixture_label/fixture_label_manifest.json").read_text()
    )
    records = []
    all_labels, all_probability = [], []
    started = time.time()
    record_metadata = manifest["records"]
    if args.max_records is not None:
        if args.max_records <= 0:
            raise SystemExit("max-records must be positive")
        record_metadata = record_metadata[: args.max_records]
    for record_index, meta in enumerate(record_metadata, 1):
        opaque_id = str(meta["opaque_id"])
        input_path = fixture_root / "fixture_input/records" / f"{opaque_id}.npz"
        label_path = fixture_root / "fixture_label/records" / f"{opaque_id}.npz"
        graph_path = graph_root / "artifacts" / f"{opaque_id}.graph.npz"
        with np.load(input_path, allow_pickle=False) as input_values, np.load(
            label_path, allow_pickle=False
        ) as label_values, np.load(graph_path, allow_pickle=False) as graph_values:
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
            raw_tiles = np.asarray(input_values["slot_tiles"])
            denoised_tiles = np.asarray(graph_values["denoised_tiles"])
            truth = np.asarray(label_values["composed_slot_to_target"])
            clean = np.asarray(label_values["clean_target_rgb"])
            baseline_layout = np.asarray(graph_values["qap_w4_layout"])
            qap_seed = int(input_values["qap_seed"])
        features = candidate_features(scores, graph)
        labels = candidate_labels(graph, truth)
        probability = score_graph(
            model,
            raw_tiles,
            denoised_tiles,
            graph,
            features,
            device=device,
            batch_size=args.batch_size,
            side_band=model.side_band,
        )
        proposals = proposals_from_threshold(graph, probability, threshold)
        components, accepted = grow_components_with_edges(proposals)
        solved = solve_components(
            components, scores["w4"], qap_seed=qap_seed, args=args
        )
        baseline_layout_metrics = layout_metrics(baseline_layout, truth)
        solved_layout_metrics = layout_metrics(solved, truth)
        baseline_image = predicted_image_metrics(baseline_layout, denoised_tiles, clean)
        solved_image = predicted_image_metrics(solved, denoised_tiles, clean)
        record = {
            "opaque_id": opaque_id,
            "source_name": meta["source_name"],
            "panel": meta["panel"],
            "binary": binary_metrics(labels, probability),
            "candidate_recall": float(labels.sum() / 1104.0),
            "selected_edges": len(proposals),
            "accepted_edges": len(accepted),
            "accepted_precision": accepted_precision(accepted, graph, labels),
            "component_sizes": sorted(
                (len(component) for component in components), reverse=True
            ),
            "baseline": {
                "adjacency": baseline_layout_metrics["combined_adjacency"],
                "ssim": baseline_image["predicted_layout_ssim"],
            },
            "verified": {
                "adjacency": solved_layout_metrics["combined_adjacency"],
                "ssim": solved_image["predicted_layout_ssim"],
            },
        }
        records.append(record)
        all_labels.append(labels)
        all_probability.append(probability)
        print(
            json.dumps({"stage": "v4", "done": record_index, "total": len(record_metadata)}),
            flush=True,
        )
    pooled = binary_metrics(np.concatenate(all_labels), np.concatenate(all_probability))

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for record in records:
            value: object = record
            for key in path:
                value = value[key]  # type: ignore[index]
            values.append(float(value))
        return float(np.mean(values))

    report = {
        "schema_version": 1,
        "kind": "binary_edge_verifier_v4_evaluation",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "calibration_threshold": threshold,
        "checkpoint_metadata": metadata,
        "pooled_binary": pooled,
        "aggregate": {
            "records": len(records),
            "mean_accepted_precision": mean(("accepted_precision",)),
            "mean_largest_component": float(
                np.mean([record["component_sizes"][0] for record in records])
            ),
            "mean_baseline_adjacency": mean(("baseline", "adjacency")),
            "mean_verified_adjacency": mean(("verified", "adjacency")),
            "mean_baseline_ssim": mean(("baseline", "ssim")),
            "mean_verified_ssim": mean(("verified", "ssim")),
            "mean_ssim_delta": mean(("verified", "ssim"))
            - mean(("baseline", "ssim")),
        },
        "records": records,
        "seconds": time.time() - started,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
