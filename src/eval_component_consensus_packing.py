"""Ensemble component geometries across raw/confidence edge-budget solvers."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from config import NFRAG, WORK_ROOT
from edge_confidence import EdgeConfidenceMLP, standardize
from eval_calibrated_buddies import component_metrics
from eval_seeded_qap import dense_rd
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import (
    build_buddies_components,
    build_directed_components,
    solve_components_from_scores,
)


def component_relations(
    components: list[dict[int, tuple[int, int]]],
) -> list[tuple[int, int, int]]:
    """Return canonical right/down tile relations implied inside components."""
    relations: list[tuple[int, int, int]] = []
    for component in components:
        by_position = {position: tile for tile, position in component.items()}
        for tile, (row, col) in component.items():
            right = by_position.get((row, col + 1))
            down = by_position.get((row + 1, col))
            if right is not None:
                relations.append((int(tile), 3, int(right)))
            if down is not None:
                relations.append((int(tile), 1, int(down)))
    return relations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confidence-checkpoint",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "best.pt",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--images", default="50,51,52,53,54,55")
    parser.add_argument("--budgets", default="64,96,128,192,256,384,512")
    parser.add_argument("--minimum-votes", default="2,3,4,5,6,8")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "component_consensus_packing_gate.json",
    )
    args = parser.parse_args()
    checkpoint = torch.load(args.confidence_checkpoint, map_location="cpu", weights_only=False)
    model = EdgeConfidenceMLP(
        checkpoint["features"],
        hidden=checkpoint["hidden"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    images = [int(value) for value in args.images.split(",")]
    budgets = [int(value) for value in args.budgets.split(",")]
    vote_thresholds = [int(value) for value in args.minimum_votes.split(",")]
    rows: dict[str, list[dict[str, float]]] = {
        str(value): [] for value in vote_thresholds
    }
    for image in images:
        stored = np.load(args.cache_dir / f"image_{image:04d}_k64.npz")
        with torch.inference_mode():
            confidence = torch.sigmoid(
                model(
                    torch.from_numpy(
                        standardize(stored["features"].astype(np.float32), mean, scale)
                    )
                )
            ).numpy()
        candidates = torch.from_numpy(stored["candidate_ids"]).long()
        flat_scores = torch.from_numpy(stored["candidate_scores"]).float()
        scores = flat_scores.reshape(NFRAG, 4, -1).permute(1, 0, 2).contiguous()
        right_t, down_t = dense_rd(candidates, scores)
        right = right_t.numpy()
        down = down_t.numpy()
        anchors = stored["anchors"].astype(np.int64)
        directions = stored["directions"].astype(np.int64)
        predicted = stored["predicted"].astype(np.int64)
        permutation = stored["permutation"].astype(np.int64)
        votes: Counter[tuple[int, int, int]] = Counter()
        for budget in budgets:
            raw_components = build_buddies_components(
                right,
                down,
                max_edges=budget,
                min_margin=0.0,
            )
            votes.update(set(component_relations(raw_components)))
            calibrated_components = build_directed_components(
                anchors,
                directions,
                predicted,
                confidence,
                max_edges=budget,
            )
            votes.update(set(component_relations(calibrated_components)))
        truth = np.argsort(permutation)
        for minimum_votes in vote_thresholds:
            accepted = [
                (relation, count)
                for relation, count in votes.items()
                if count >= minimum_votes
            ]
            edge_anchors = np.asarray([item[0][0] for item in accepted], dtype=np.int64)
            edge_directions = np.asarray([item[0][1] for item in accepted], dtype=np.int64)
            edge_targets = np.asarray([item[0][2] for item in accepted], dtype=np.int64)
            edge_weights = np.asarray([item[1] for item in accepted], dtype=np.float32)
            components = build_directed_components(
                edge_anchors,
                edge_directions,
                edge_targets,
                edge_weights,
                max_edges=len(accepted),
            )
            placement, objective = solve_components_from_scores(
                right,
                down,
                components,
                repair_passes=0,
            )
            placement_acc, _ = placement_accuracy(placement, truth)
            neighbour, right_acc, down_acc = neighbour_accuracy(placement, truth)
            metrics = {
                "placement": float(placement_acc),
                "neighbour": float(neighbour),
                "right": float(right_acc),
                "down": float(down_acc),
                "objective": float(objective),
                "accepted_relations": float(len(accepted)),
                "image": float(image),
                **component_metrics(components, permutation),
            }
            rows[str(minimum_votes)].append(metrics)
            print(
                json.dumps(
                    {
                        "image": image,
                        "minimum_votes": minimum_votes,
                        **metrics,
                    }
                ),
                flush=True,
            )
    summary = {
        threshold: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
            if key != "image"
        }
        for threshold, values in rows.items()
    }
    best_key = max(
        summary,
        key=lambda key: (
            summary[key]["neighbour"],
            summary[key]["placement"],
        ),
    )
    best = summary[best_key]
    thresholds = {
        "neighbour": 0.18,
        "placement": 0.01,
        "internal_edge_precision": 0.70,
    }
    checks = {key: best[key] >= value for key, value in thresholds.items()}
    report = {
        "experiment": "raw_confidence_component_geometry_consensus",
        "status": "pass" if all(checks.values()) else "fail",
        "best_minimum_votes": best_key,
        "best": best,
        "summary": summary,
        "thresholds": thresholds,
        "checks": checks,
        "budgets": budgets,
        "images": images,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate": report, "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
