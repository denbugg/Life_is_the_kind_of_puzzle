"""Gate calibrated edge bonuses inside the discrete global buddies solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from candidate_rank import DOWN, LEFT, RIGHT, UP
from config import NFRAG, WORK_ROOT
from edge_confidence import EdgeConfidenceMLP, standardize
from eval_seeded_qap import dense_rd
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import (
    build_buddies_components,
    build_directed_components,
    solve_components_from_scores,
    solve_buddies_from_scores,
    solve_buddies_multistart_from_scores,
)


def component_metrics(
    components: list[dict[int, tuple[int, int]]],
    permutation: np.ndarray,
) -> dict[str, float]:
    sizes: list[int] = []
    pure_sizes: list[int] = []
    aligned_correct = 0
    internal_edges = 0
    exact_internal_edges = 0
    for component in components:
        size = len(component)
        sizes.append(size)
        offsets: dict[tuple[int, int], int] = {}
        by_position = {position: tile for tile, position in component.items()}
        for tile, position in component.items():
            cell = int(permutation[tile])
            truth = (cell // 24, cell % 24)
            offset = (truth[0] - position[0], truth[1] - position[1])
            offsets[offset] = offsets.get(offset, 0) + 1
            for delta in ((0, 1), (1, 0)):
                target = by_position.get(
                    (position[0] + delta[0], position[1] + delta[1])
                )
                if target is None:
                    continue
                internal_edges += 1
                target_cell = int(permutation[target])
                if target_cell - cell == delta[0] * 24 + delta[1]:
                    exact_internal_edges += 1
        correct = max(offsets.values())
        aligned_correct += correct
        if correct == size:
            pure_sizes.append(size)
    nontrivial = [size for size in sizes if size >= 2]
    return {
        "component_count": float(len(components)),
        "largest_component": float(max(sizes, default=1)),
        "largest_pure_component": float(max(pure_sizes, default=1)),
        "nontrivial_tile_coverage": float(sum(nontrivial) / len(permutation)),
        "translation_aligned_accuracy": float(aligned_correct / len(permutation)),
        "internal_edge_precision": (
            float(exact_internal_edges / internal_edges) if internal_edges else 0.0
        ),
        "internal_edges": float(internal_edges),
    }


def add_seed_bonus(
    right: np.ndarray,
    down: np.ndarray,
    *,
    anchors: np.ndarray,
    directions: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    bonus: float,
) -> int:
    accepted = confidence >= threshold
    for anchor, direction, target, probability in zip(
        anchors[accepted],
        directions[accepted],
        predicted[accepted],
        confidence[accepted],
    ):
        value = bonus * float(probability)
        if direction == RIGHT:
            right[anchor, target] += value
        elif direction == LEFT:
            right[target, anchor] += value
        elif direction == DOWN:
            down[anchor, target] += value
        elif direction == UP:
            down[target, anchor] += value
    return int(accepted.sum())


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
    parser.add_argument("--images", default="50,51,52")
    parser.add_argument("--bonuses", default="0,0.25,0.5,1,2")
    parser.add_argument("--max-edges", default="64,128,256,512")
    parser.add_argument("--repair-passes", type=int, default=0)
    parser.add_argument("--packing-restarts", type=int, default=1)
    parser.add_argument("--packing-temperature", type=float, default=0.05)
    parser.add_argument("--packing-order-jitter", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--component-source",
        choices=("raw", "confidence"),
        default="raw",
    )
    parser.add_argument(
        "--reference-neighbour",
        type=float,
        default=None,
        help="optional recorded reference used only for an explicit breakthrough check",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "calibrated_buddies_gate.json",
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
    threshold = float(checkpoint["threshold"])
    image_ids = [int(value) for value in args.images.split(",")]
    bonuses = [float(value) for value in args.bonuses.split(",")]
    edge_limits = [int(value) for value in args.max_edges.split(",")]
    rows: dict[str, list[dict[str, float]]] = {
        f"{bonus}:{limit}": [] for bonus in bonuses for limit in edge_limits
    }
    for image_id in image_ids:
        cache = args.cache_dir / f"image_{image_id:04d}_k64.npz"
        stored = np.load(cache)
        features = stored["features"]
        with torch.inference_mode():
            confidence = torch.sigmoid(
                model(torch.from_numpy(standardize(features, mean, scale)))
            ).numpy()
        candidates = torch.from_numpy(stored["candidate_ids"]).long()
        flat_scores = torch.from_numpy(stored["candidate_scores"]).float()
        scores = flat_scores.reshape(NFRAG, 4, -1).permute(1, 0, 2).contiguous()
        base_right, base_down = dense_rd(candidates, scores)
        anchors = stored["anchors"]
        directions = stored["directions"]
        predicted = stored["predicted"]
        permutation = stored["permutation"].astype(np.int64)
        truth = np.argsort(permutation)
        for bonus in bonuses:
            for limit in edge_limits:
                right = base_right.numpy().copy()
                down = base_down.numpy().copy()
                seed_count = add_seed_bonus(
                    right,
                    down,
                    anchors=anchors,
                    directions=directions,
                    predicted=predicted,
                    confidence=confidence,
                    threshold=threshold,
                    bonus=bonus,
                )
                if args.component_source == "confidence":
                    components = build_directed_components(
                        anchors,
                        directions,
                        predicted,
                        confidence,
                        max_edges=limit,
                    )
                    placement, objective = solve_components_from_scores(
                        right,
                        down,
                        components,
                        repair_passes=args.repair_passes,
                        restarts=args.packing_restarts,
                        seed=args.seed + image_id * 1009,
                        temperature=args.packing_temperature,
                        order_jitter=args.packing_order_jitter,
                    )
                elif args.packing_restarts > 1:
                    placement, objective = solve_buddies_multistart_from_scores(
                        right,
                        down,
                        max_edges=limit,
                        min_margin=0.0,
                        repair_passes=args.repair_passes,
                        restarts=args.packing_restarts,
                        seed=args.seed + image_id * 1009,
                        temperature=args.packing_temperature,
                        order_jitter=args.packing_order_jitter,
                    )
                else:
                    placement, objective = solve_buddies_from_scores(
                        right,
                        down,
                        max_edges=limit,
                        min_margin=0.0,
                        repair_passes=args.repair_passes,
                    )
                if args.component_source == "raw":
                    components = build_buddies_components(
                        right,
                        down,
                        max_edges=limit,
                        min_margin=0.0,
                    )
                placement_acc, _ = placement_accuracy(placement, truth)
                neighbour, right_acc, down_acc = neighbour_accuracy(placement, truth)
                metrics = {
                    "placement": float(placement_acc),
                    "neighbour": float(neighbour),
                    "right": float(right_acc),
                    "down": float(down_acc),
                    "objective": float(objective),
                    "seeds": float(seed_count),
                    "image": float(image_id),
                    **component_metrics(components, permutation),
                }
                rows[f"{bonus}:{limit}"].append(metrics)
                print(
                    json.dumps(
                        {
                            "bonus": bonus,
                            "max_edges": limit,
                            **metrics,
                        }
                    ),
                    flush=True,
                )
    summary: dict[str, dict[str, float]] = {}
    for operating_point, values in rows.items():
        summary[operating_point] = {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
            if key != "image"
        }
    baseline = max(
        (value for key, value in summary.items() if key.startswith("0.0:")),
        key=lambda value: (value["neighbour"], value["placement"]),
    )
    calibrated_keys = [key for key in summary if not key.startswith("0.0:")]
    calibrated_key = (
        max(
            calibrated_keys,
            key=lambda key: (summary[key]["neighbour"], summary[key]["placement"]),
        )
        if calibrated_keys
        else max(
            summary,
            key=lambda key: (summary[key]["neighbour"], summary[key]["placement"]),
        )
    )
    calibrated = summary[calibrated_key]
    delta = {
        "placement": calibrated["placement"] - baseline["placement"],
        "neighbour": calibrated["neighbour"] - baseline["neighbour"],
    }
    thresholds = {
        "neighbour": 0.05,
        "neighbour_delta": 0.02,
        "placement_delta": 0.005,
    }
    checks = {
        "neighbour": calibrated["neighbour"] >= thresholds["neighbour"],
        "neighbour_delta": delta["neighbour"] >= thresholds["neighbour_delta"],
        "placement_delta": delta["placement"] >= thresholds["placement_delta"],
    }
    solver_thresholds = {
        "neighbour": 0.16,
        "reference_delta": 0.02,
    }
    reference_delta = (
        baseline["neighbour"] - args.reference_neighbour
        if args.reference_neighbour is not None
        else None
    )
    solver_checks = {
        "neighbour": baseline["neighbour"] >= solver_thresholds["neighbour"],
        "reference_delta": (
            reference_delta >= solver_thresholds["reference_delta"]
            if reference_delta is not None
            else True
        ),
    }
    report = {
        "experiment": "calibrated_discrete_global_buddies",
        "status": "pass" if all(solver_checks.values()) else "fail",
        "best_calibrated": calibrated_key,
        "baseline": baseline,
        "calibrated": calibrated,
        "delta": delta,
        "thresholds": thresholds,
        "checks": checks,
        "solver_breakthrough": {
            "status": "pass" if all(solver_checks.values()) else "fail",
            "reference_neighbour": args.reference_neighbour,
            "reference_delta": reference_delta,
            "thresholds": solver_thresholds,
            "checks": solver_checks,
        },
        "summary": summary,
        "packing": {
            "restarts": args.packing_restarts,
            "temperature": args.packing_temperature,
            "order_jitter": args.packing_order_jitter,
            "component_source": args.component_source,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate": report, "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
