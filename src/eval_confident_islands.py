"""Evaluate sparse calibrated edge predictions as translation-invariant islands."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from candidate_rank import neighbor_targets
from canvas_data import CanvasDataset
from config import NFRAG, SEED, WORK_ROOT
from consensus_islands import (
    ConsensusAssembler,
    SelectedEdges,
    edge_metrics,
    island_metrics,
)
from edge_confidence import EdgeConfidenceMLP, standardize
from eval_test_time_adaptation import _load_ranker
from imgio import train_val_split
from train_edge_confidence import collect_one_image
from train_offset_pose import load_frozen_affinity

_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))


def selected_edges(
    anchors: np.ndarray,
    directions: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> SelectedEdges:
    return SelectedEdges(
        anchors=anchors[mask],
        directions=directions[mask],
        targets=predicted[mask],
        margins=probabilities[mask],
        loop=np.zeros(int(mask.sum()), dtype=bool),
        threshold=threshold,
    )


def grow_from_seeds(
    *,
    anchors: np.ndarray,
    directions: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    seed_threshold: float,
    growth_threshold: float,
) -> tuple[ConsensusAssembler, SelectedEdges]:
    """Grow only from non-singleton seed components, strongest edge first."""
    assembler = ConsensusAssembler(NFRAG)
    seed_mask = probabilities >= seed_threshold
    seed = selected_edges(
        anchors,
        directions,
        predicted,
        probabilities,
        seed_mask,
        seed_threshold,
    )
    assembler.add_edges(seed)
    accepted_indices = list(np.flatnonzero(seed_mask))
    growth_order = np.flatnonzero(
        (probabilities >= growth_threshold) & ~seed_mask
    )
    growth_order = growth_order[np.argsort(-probabilities[growth_order])]
    # A newly attached singleton immediately becomes part of the island, so
    # later edges may continue the frontier. Isolated weak pairs are forbidden.
    for index in growth_order:
        anchor = int(anchors[index])
        target = int(predicted[index])
        ca = int(assembler.component_of[anchor])
        cb = int(assembler.component_of[target])
        if max(len(assembler.positions[ca]), len(assembler.positions[cb])) < 2:
            continue
        if assembler.add(anchor, target, int(directions[index])):
            accepted_indices.append(int(index))
    mask = np.zeros(len(probabilities), dtype=bool)
    mask[accepted_indices] = True
    return assembler, selected_edges(
        anchors,
        directions,
        predicted,
        probabilities,
        mask,
        growth_threshold,
    )


def grow_by_component_consensus(
    *,
    anchors: np.ndarray,
    directions: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    seed_threshold: float,
    growth_threshold: float,
    minimum_support: int,
) -> tuple[ConsensusAssembler, SelectedEdges]:
    """Merge components only when distinct tile pairs vote for one translation."""
    assembler = ConsensusAssembler(NFRAG)
    seed_mask = probabilities >= seed_threshold
    assembler.add_edges(
        selected_edges(
            anchors,
            directions,
            predicted,
            probabilities,
            seed_mask,
            seed_threshold,
        )
    )
    accepted_indices = list(np.flatnonzero(seed_mask))
    candidate_indices = np.flatnonzero(
        (probabilities >= growth_threshold) & ~seed_mask
    )
    for _ in range(NFRAG):
        proposals: dict[
            tuple[int, int, int, int], list[int]
        ] = {}
        for raw_index in candidate_indices:
            index = int(raw_index)
            anchor = int(anchors[index])
            target = int(predicted[index])
            ca = int(assembler.component_of[anchor])
            cb = int(assembler.component_of[target])
            if ca == cb:
                continue
            if max(len(assembler.positions[ca]), len(assembler.positions[cb])) < 2:
                # Weak reciprocal pairs may extend a trusted island, but may
                # not manufacture brand-new islands from two singletons.
                continue
            pa = assembler.positions[ca][anchor]
            pb = assembler.positions[cb][target]
            dr, dc = _DELTA[int(directions[index])]
            if ca < cb:
                # Translation applied to component cb in coordinates of ca.
                shift = (pa[0] + dr - pb[0], pa[1] + dc - pb[1])
                key = (ca, cb, shift[0], shift[1])
            else:
                # Canonicalize the component order: translate ca in cb's frame.
                shift = (pb[0] - pa[0] - dr, pb[1] - pa[1] - dc)
                key = (cb, ca, shift[0], shift[1])
            proposals.setdefault(key, []).append(index)
        eligible: list[tuple[int, float, list[int]]] = []
        for indices in proposals.values():
            # Opposite directed decisions are separate ranker evaluations.
            # They may therefore provide the two votes needed to extend a
            # trusted component, while support=3 still demands extra context.
            support_count = len(indices)
            if support_count >= minimum_support:
                eligible.append(
                    (
                        support_count,
                        float(probabilities[indices].mean()),
                        indices,
                    )
                )
        if not eligible:
            break
        _, _, indices = max(eligible, key=lambda item: (item[0], item[1]))
        merged = False
        for index in sorted(indices, key=lambda i: -probabilities[i]):
            if assembler.add(
                int(anchors[index]),
                int(predicted[index]),
                int(directions[index]),
            ):
                accepted_indices.append(index)
                merged = True
                break
        if not merged:
            # Remove a geometrically impossible proposal and continue.
            candidate_indices = np.setdiff1d(
                candidate_indices,
                np.asarray(indices, dtype=np.int64),
                assume_unique=False,
            )
            continue
        # Add the remaining supporting constraints as consistency checks.
        for index in indices:
            if index in accepted_indices:
                continue
            if assembler.add(
                int(anchors[index]),
                int(predicted[index]),
                int(directions[index]),
            ):
                accepted_indices.append(index)
    mask = np.zeros(len(probabilities), dtype=bool)
    mask[np.asarray(accepted_indices, dtype=np.int64)] = True
    return assembler, selected_edges(
        anchors,
        directions,
        predicted,
        probabilities,
        mask,
        growth_threshold,
    )


def grow_by_alternative_consensus(
    *,
    seed_edges: SelectedEdges,
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    alternative_k: int,
    minimum_support: int,
) -> tuple[ConsensusAssembler, SelectedEdges]:
    """Merge seed components using agreement among non-top-1 alternatives."""
    assembler = ConsensusAssembler(NFRAG)
    assembler.add_edges(seed_edges)
    finite_scores = np.where(np.isfinite(candidate_scores), candidate_scores, -1.0e4)
    shifted = finite_scores - finite_scores.max(axis=1, keepdims=True)
    row_probability = np.exp(shifted)
    row_probability /= row_probability.sum(axis=1, keepdims=True)
    slots = np.argpartition(
        -finite_scores,
        kth=min(alternative_k, finite_scores.shape[1]) - 1,
        axis=1,
    )[:, :alternative_k]
    row_index = np.repeat(np.arange(candidate_scores.shape[0]), alternative_k)
    slot_index = slots.reshape(-1)
    alt_anchor = row_index // 4
    alt_direction = row_index % 4
    alt_target = candidate_ids[alt_anchor, slot_index]
    alt_probability = row_probability[row_index, slot_index]
    accepted_alt: list[int] = []
    for _ in range(NFRAG):
        proposals: dict[tuple[int, int, int, int], list[int]] = {}
        for index in range(len(alt_anchor)):
            anchor = int(alt_anchor[index])
            target = int(alt_target[index])
            ca = int(assembler.component_of[anchor])
            cb = int(assembler.component_of[target])
            if ca == cb:
                continue
            if max(len(assembler.positions[ca]), len(assembler.positions[cb])) < 2:
                continue
            pa = assembler.positions[ca][anchor]
            pb = assembler.positions[cb][target]
            dr, dc = _DELTA[int(alt_direction[index])]
            if ca < cb:
                shift = (pa[0] + dr - pb[0], pa[1] + dc - pb[1])
                key = (ca, cb, shift[0], shift[1])
            else:
                shift = (pb[0] - pa[0] - dr, pb[1] - pa[1] - dc)
                key = (cb, ca, shift[0], shift[1])
            proposals.setdefault(key, []).append(index)
        eligible: list[tuple[int, float, list[int]]] = []
        for indices in proposals.values():
            distinct_pairs = {
                tuple(sorted((int(alt_anchor[i]), int(alt_target[i]))))
                for i in indices
            }
            if len(distinct_pairs) < minimum_support:
                continue
            eligible.append(
                (
                    len(distinct_pairs),
                    float(alt_probability[indices].sum()),
                    indices,
                )
            )
        if not eligible:
            break
        _, _, indices = max(eligible, key=lambda item: (item[0], item[1]))
        merged = False
        for index in sorted(indices, key=lambda i: -alt_probability[i]):
            if assembler.add(
                int(alt_anchor[index]),
                int(alt_target[index]),
                int(alt_direction[index]),
            ):
                accepted_alt.append(index)
                merged = True
                break
        if not merged:
            break
        for index in indices:
            if index in accepted_alt:
                continue
            if assembler.add(
                int(alt_anchor[index]),
                int(alt_target[index]),
                int(alt_direction[index]),
            ):
                accepted_alt.append(index)
    # Report seed and alternative constraints together.
    anchors = np.concatenate(
        (seed_edges.anchors, alt_anchor[accepted_alt].astype(np.int64))
    )
    directions = np.concatenate(
        (seed_edges.directions, alt_direction[accepted_alt].astype(np.int64))
    )
    targets = np.concatenate(
        (seed_edges.targets, alt_target[accepted_alt].astype(np.int64))
    )
    margins = np.concatenate(
        (seed_edges.margins, alt_probability[accepted_alt].astype(np.float32))
    )
    return assembler, SelectedEdges(
        anchors=anchors,
        directions=directions,
        targets=targets,
        margins=margins,
        loop=np.zeros(len(anchors), dtype=bool),
        threshold=0.0,
    )


def main() -> None:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confidence-checkpoint",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "best.pt",
    )
    parser.add_argument(
        "--ranker",
        default=str(workspace / "artifacts/candidate_rank/rank_v2w64_best.pt"),
    )
    parser.add_argument(
        "--affinity-ckpt",
        default=str(workspace / "artifacts/macro_affinity/affinity_r1_1200_best.pt"),
    )
    parser.add_argument(
        "--affinity-ckpt2",
        default=str(workspace / "artifacts/macro_affinity/affinity_r3_1000_best.pt"),
    )
    parser.add_argument("--validation-offset", type=int, default=50)
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--pair-batch", type=int, default=4096)
    parser.add_argument("--growth-thresholds", default="0.98,0.95,0.90,0.80")
    parser.add_argument("--consensus-support", default="2,3")
    parser.add_argument("--alternative-k", default="2,3,5")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "confident_islands_gate.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.confidence_checkpoint, map_location=device, weights_only=False)
    model = EdgeConfidenceMLP(
        checkpoint["features"],
        hidden=checkpoint["hidden"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    threshold = float(checkpoint["threshold"])
    ranker = _load_ranker(args.ranker, device)
    affinity, _, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity2, _, _ = load_frozen_affinity(args.affinity_ckpt2, device)
    _, validation_names = train_val_split()
    names = validation_names[
        args.validation_offset : args.validation_offset + args.images
    ]
    dataset = CanvasDataset(names, real_prob=0.0, seed=args.seed + 400_000)
    image_rows: list[dict[str, float]] = []
    growth_thresholds = [
        float(value) for value in args.growth_thresholds.split(",")
    ]
    growth_rows: dict[str, list[dict[str, float]]] = {
        str(value): [] for value in growth_thresholds
    }
    support_values = [int(value) for value in args.consensus_support.split(",")]
    consensus_rows: dict[str, list[dict[str, float]]] = {
        f"{threshold}:{support}": []
        for threshold in growth_thresholds
        for support in support_values
    }
    alternative_values = [int(value) for value in args.alternative_k.split(",")]
    alternative_rows: dict[str, list[dict[str, float]]] = {
        f"{alternative_k}:{support}": []
        for alternative_k in alternative_values
        for support in support_values
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for image_index in range(len(names)):
        sample = dataset[image_index]
        absolute_index = args.validation_offset + image_index
        cache = args.cache_dir / f"image_{absolute_index:04d}_k{args.candidate_k}.npz"
        cache_fields = np.load(cache).files if cache.exists() else []
        if (
            cache.exists()
            and "permutation" in cache_fields
            and "candidate_ids" in cache_fields
            and "candidate_scores" in cache_fields
        ):
            stored = np.load(cache)
            features = stored["features"]
            labels = stored["labels"]
            anchors = stored["anchors"]
            directions = stored["directions"]
            predicted = stored["predicted"]
            permutation = stored["permutation"]
            candidate_ids = stored["candidate_ids"]
            candidate_scores = stored["candidate_scores"]
            feature_names = checkpoint["feature_names"]
        else:
            (
                features,
                labels,
                _,
                feature_names,
                anchors,
                directions,
                predicted,
                candidate_ids,
                candidate_scores,
            ) = (
                collect_one_image(
                image_index=args.validation_offset + image_index,
                sample=sample,
                ranker=ranker,
                affinity=affinity,
                affinity2=affinity2,
                candidate_k=args.candidate_k,
                rows_per_image=NFRAG * 4,
                pair_batch=args.pair_batch,
                device=device,
            )
            )
            np.savez_compressed(
                cache,
                features=features,
                labels=labels,
                anchors=anchors,
                directions=directions,
                predicted=predicted,
                permutation=sample["perm"].numpy(),
                candidate_ids=candidate_ids,
                candidate_scores=candidate_scores,
            )
            permutation = sample["perm"].numpy()
        if feature_names != checkpoint["feature_names"]:
            raise RuntimeError("confidence feature schema differs from checkpoint")
        with torch.inference_mode():
            x = torch.from_numpy(standardize(features, mean, scale)).to(device)
            probabilities = torch.sigmoid(model(x)).cpu().numpy()
        accepted = probabilities >= threshold
        edges = selected_edges(
            anchors,
            directions,
            predicted,
            probabilities,
            accepted,
            threshold,
        )
        assembler = ConsensusAssembler(NFRAG)
        assembler.add_edges(edges)
        perm_tensor = torch.from_numpy(permutation).unsqueeze(0).to(device).long()
        targets, _ = neighbor_targets(perm_tensor)
        metrics = {
            **edge_metrics(edges, targets),
            **island_metrics(assembler, permutation=permutation, grid_side=24),
            "accepted_row_coverage": float(accepted.mean()),
            "sampled_classifier_precision": (
                float(labels[accepted].mean()) if accepted.any() else 0.0
            ),
            "image_index": float(args.validation_offset + image_index),
        }
        image_rows.append(metrics)
        print(json.dumps(metrics), flush=True)
        for growth_threshold in growth_thresholds:
            grown, grown_edges = grow_from_seeds(
                anchors=anchors,
                directions=directions,
                predicted=predicted,
                probabilities=probabilities,
                seed_threshold=threshold,
                growth_threshold=growth_threshold,
            )
            grown_metrics = {
                **edge_metrics(grown_edges, targets),
                **island_metrics(
                    grown,
                    permutation=permutation,
                    grid_side=24,
                ),
                "image_index": float(absolute_index),
            }
            growth_rows[str(growth_threshold)].append(grown_metrics)
            print(
                json.dumps(
                    {
                        "growth_threshold": growth_threshold,
                        **grown_metrics,
                    }
                ),
                flush=True,
            )
        for growth_threshold in growth_thresholds:
            for support in support_values:
                grown, grown_edges = grow_by_component_consensus(
                    anchors=anchors,
                    directions=directions,
                    predicted=predicted,
                    probabilities=probabilities,
                    seed_threshold=threshold,
                    growth_threshold=growth_threshold,
                    minimum_support=support,
                )
                consensus_metrics = {
                    **edge_metrics(grown_edges, targets),
                    **island_metrics(
                        grown,
                        permutation=permutation,
                        grid_side=24,
                    ),
                    "image_index": float(absolute_index),
                }
                consensus_rows[f"{growth_threshold}:{support}"].append(
                    consensus_metrics
                )
                print(
                    json.dumps(
                        {
                            "consensus_threshold": growth_threshold,
                            "minimum_support": support,
                            **consensus_metrics,
                        }
                    ),
                    flush=True,
                )
        for alternative_k in alternative_values:
            for support in support_values:
                alternative_assembler, alternative_edges = (
                    grow_by_alternative_consensus(
                        seed_edges=edges,
                        candidate_ids=candidate_ids,
                        candidate_scores=candidate_scores,
                        alternative_k=alternative_k,
                        minimum_support=support,
                    )
                )
                alternative_metrics = {
                    **edge_metrics(alternative_edges, targets),
                    **island_metrics(
                        alternative_assembler,
                        permutation=permutation,
                        grid_side=24,
                    ),
                    "image_index": float(absolute_index),
                }
                alternative_rows[f"{alternative_k}:{support}"].append(
                    alternative_metrics
                )
                print(
                    json.dumps(
                        {
                            "alternative_k": alternative_k,
                            "minimum_support": support,
                            **alternative_metrics,
                        }
                    ),
                    flush=True,
                )
    keys = [key for key in image_rows[0] if key != "image_index"]
    mean_metrics = {
        key: float(np.mean([row[key] for row in image_rows])) for key in keys
    }
    growth_summary: dict[str, dict[str, float]] = {}
    for growth_threshold, rows in growth_rows.items():
        growth_keys = [key for key in rows[0] if key != "image_index"]
        growth_summary[growth_threshold] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in growth_keys
        }
    consensus_summary: dict[str, dict[str, float]] = {}
    for operating_point, rows in consensus_rows.items():
        consensus_keys = [key for key in rows[0] if key != "image_index"]
        consensus_summary[operating_point] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in consensus_keys
        }
    alternative_summary: dict[str, dict[str, float]] = {}
    for operating_point, rows in alternative_rows.items():
        alternative_keys = [key for key in rows[0] if key != "image_index"]
        alternative_summary[operating_point] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in alternative_keys
        }
    growth_gate_thresholds = {
        "exact_edge_precision": 0.85,
        "pure_nontrivial_tile_coverage": 0.20,
        "largest_pure_component": 8.0,
    }
    best_growth_key = max(
        growth_summary,
        key=lambda key: (
            sum(
                growth_summary[key][metric] >= value
                for metric, value in growth_gate_thresholds.items()
            ),
            growth_summary[key]["pure_nontrivial_tile_coverage"],
            growth_summary[key]["largest_pure_component"],
        ),
    )
    best_growth = growth_summary[best_growth_key]
    growth_checks = {
        key: best_growth[key] >= value
        for key, value in growth_gate_thresholds.items()
    }
    best_consensus_key = max(
        consensus_summary,
        key=lambda key: (
            sum(
                consensus_summary[key][metric] >= value
                for metric, value in growth_gate_thresholds.items()
            ),
            consensus_summary[key]["pure_nontrivial_tile_coverage"],
            consensus_summary[key]["largest_pure_component"],
        ),
    )
    best_consensus = consensus_summary[best_consensus_key]
    consensus_checks = {
        key: best_consensus[key] >= value
        for key, value in growth_gate_thresholds.items()
    }
    best_alternative_key = max(
        alternative_summary,
        key=lambda key: (
            sum(
                alternative_summary[key][metric] >= value
                for metric, value in growth_gate_thresholds.items()
            ),
            alternative_summary[key]["pure_nontrivial_tile_coverage"],
            alternative_summary[key]["largest_pure_component"],
        ),
    )
    best_alternative = alternative_summary[best_alternative_key]
    alternative_checks = {
        key: best_alternative[key] >= value
        for key, value in growth_gate_thresholds.items()
    }
    thresholds = {
        "exact_edge_precision": 0.90,
        "pure_nontrivial_tile_coverage": 0.10,
        "largest_pure_component": 6.0,
    }
    checks = {
        key: mean_metrics[key] >= value for key, value in thresholds.items()
    }
    report = {
        "experiment": "calibrated_sparse_consensus_islands",
        "status": "pass" if all(checks.values()) else "fail",
        "confidence_threshold": threshold,
        "mean": mean_metrics,
        "per_image": image_rows,
        "thresholds": thresholds,
        "checks": checks,
        "seeded_growth": {
            "status": "pass" if all(growth_checks.values()) else "fail",
            "best_threshold": best_growth_key,
            "best": best_growth,
            "summary": growth_summary,
            "thresholds": growth_gate_thresholds,
            "checks": growth_checks,
        },
        "component_consensus_growth": {
            "status": "pass" if all(consensus_checks.values()) else "fail",
            "best_operating_point": best_consensus_key,
            "best": best_consensus,
            "summary": consensus_summary,
            "thresholds": growth_gate_thresholds,
            "checks": consensus_checks,
        },
        "alternative_candidate_consensus": {
            "status": "pass" if all(alternative_checks.values()) else "fail",
            "best_operating_point": best_alternative_key,
            "best": best_alternative,
            "summary": alternative_summary,
            "thresholds": growth_gate_thresholds,
            "checks": alternative_checks,
        },
        "images": len(names),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate": report, "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
