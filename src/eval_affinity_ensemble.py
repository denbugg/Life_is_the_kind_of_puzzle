"""Compare two learned affinity encoders as a directed candidate-set ensemble.

This is intentionally a narrow gate for the hierarchical solver branch.  For
every tile in a freshly generated, exactly labelled synthetic puzzle it asks:
do the top-K candidates produced by either encoder contain the true local
neighbours?  It does *not* attempt to arrange a puzzle or score seams.

The two individual top-K candidate sets, their deduplicated directed union,
and their deduplicated directed intersection are evaluated.  ``mutual``
metrics are stricter: a candidate edge ``i -> j`` counts only if the reverse
edge ``j -> i`` also appears in the same candidate set.  That is useful when a
later offset scorer will only retain mutually plausible tile pairs.

Examples:

    python src/eval_affinity_ensemble.py
    python src/eval_affinity_ensemble.py --n 24 --k 64 --device cuda
    python src/eval_affinity_ensemble.py --ckpt_a path/to/a.pt --ckpt_b path/to/b.pt
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor

from canvas_data import CanvasDataset
from config import NFRAG, SEED
from eval_affinity_graph import (
    _parse_device,
    chebyshev_distance,
    learned_affinity,
    load_model,
    top_neighbours,
)
from imgio import train_val_split
from macro_affinity import count_params


RADII = (1, 3)
DEFAULT_CKPT_A = os.path.join("artifacts", "macro_affinity", "affinity_r1_600_best.pt")
DEFAULT_CKPT_B = os.path.join("artifacts", "macro_affinity", "affinity_r3_1000_best.pt")


def _candidate_matrix(neighbours: Tensor) -> Tensor:
    """Convert per-anchor candidate indices into a directed boolean matrix."""
    if neighbours.ndim != 2 or neighbours.shape[0] != NFRAG:
        raise ValueError(
            f"neighbours must have shape ({NFRAG}, K), got {tuple(neighbours.shape)}"
        )
    if neighbours.shape[1] < 1:
        raise ValueError("neighbours must contain at least one candidate per anchor")
    if neighbours.dtype != torch.long:
        neighbours = neighbours.long()
    if torch.any(neighbours < 0) or torch.any(neighbours >= NFRAG):
        raise ValueError("neighbours contains an out-of-range tile index")
    candidates = torch.zeros((NFRAG, NFRAG), dtype=torch.bool, device=neighbours.device)
    candidates.scatter_(1, neighbours, True)
    candidates.fill_diagonal_(False)
    return candidates


def candidate_metrics(distance: Tensor, candidates: Tensor) -> dict[str, float]:
    """Measure local relation coverage of a directed candidate relation.

    Precision and recall are anchor-balanced, so tiles near a clean-board edge
    do not get drowned out by interior tiles with more spatial neighbours.
    Empty intersection rows count as zero precision/recall rather than being
    silently discarded.  ``mutual_r*_recall`` is the true-neighbour coverage
    after requiring both directions of a candidate edge.
    """
    if tuple(distance.shape) != (NFRAG, NFRAG):
        raise ValueError(f"distance must have shape ({NFRAG}, {NFRAG})")
    if tuple(candidates.shape) != (NFRAG, NFRAG):
        raise ValueError(f"candidates must have shape ({NFRAG}, {NFRAG})")
    if candidates.dtype != torch.bool:
        candidates = candidates.bool()

    off_diagonal = ~torch.eye(NFRAG, dtype=torch.bool, device=distance.device)
    candidates = candidates & off_diagonal
    size = candidates.sum(dim=-1).float()
    mutual = candidates & candidates.transpose(0, 1)
    mutual_size = mutual.sum(dim=-1).float()
    total_candidates = size.sum()

    result = {
        "candidate_mean_size": float(size.mean()),
        "candidate_empty_fraction": float((size == 0).float().mean()),
        "candidate_reciprocal_fraction": float(
            mutual_size.sum() / total_candidates.clamp_min(1.0)
        ),
        "candidate_mean_mutual_size": float(mutual_size.mean()),
    }
    for radius in RADII:
        positive = (distance <= radius) & off_diagonal
        positive_count = positive.sum(dim=-1).float()
        hits = (candidates & positive).sum(dim=-1).float()
        mutual_hits = (mutual & positive).sum(dim=-1).float()
        # Recall is always defined for Chebyshev radii >= 1.  Precision of an
        # empty proposal row is deliberately zero: it cannot help a downstream
        # candidate scorer.
        result[f"r{radius}_precision"] = float(
            torch.where(size > 0.0, hits / size.clamp_min(1.0), torch.zeros_like(size)).mean()
        )
        result[f"r{radius}_recall"] = float((hits / positive_count).mean())
        result[f"mutual_r{radius}_recall"] = float((mutual_hits / positive_count).mean())
    return result


def _mean_metrics(totals: Mapping[str, float], count: int) -> dict[str, float]:
    if count < 1:
        raise ValueError("cannot average zero images")
    return {key: value / count for key, value in totals.items()}


def _checkpoint_step(metadata: Mapping[str, object]) -> str:
    step = metadata.get("step")
    return f" step={step}" if step is not None else ""


def _print_report(label: str, metrics: Mapping[str, float], images: int) -> None:
    """Print all candidate-set quantities needed for the ensemble decision."""
    print(f"[{label}] exact synthetic images={images}")
    print(
        "  candidates: "
        f"mean_size={metrics['candidate_mean_size']:.2f} "
        f"empty={metrics['candidate_empty_fraction']:.4f} "
        f"reciprocal_fraction={metrics['candidate_reciprocal_fraction']:.4f} "
        f"mean_mutual_size={metrics['candidate_mean_mutual_size']:.2f}"
    )
    for radius in RADII:
        print(
            f"  Cheb<={radius}: "
            f"precision={metrics[f'r{radius}_precision']:.4f} "
            f"recall={metrics[f'r{radius}_recall']:.4f} "
            f"mutual_recall={metrics[f'mutual_r{radius}_recall']:.4f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt_a",
        default=DEFAULT_CKPT_A,
        help=f"first MacroAffinityNet checkpoint (default: {DEFAULT_CKPT_A})",
    )
    parser.add_argument(
        "--ckpt_b",
        default=DEFAULT_CKPT_B,
        help=f"second MacroAffinityNet checkpoint (default: {DEFAULT_CKPT_B})",
    )
    parser.add_argument("--n", type=int, default=12, help="held-out synthetic images to evaluate")
    parser.add_argument("--k", type=int, default=64, help="top-K directed candidates per encoder")
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--seed", type=int, default=SEED, help="seed for fresh synthetic corruptions")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    if not 1 <= args.k < NFRAG:
        parser.error(f"--k must be in [1, {NFRAG - 1}]")
    return args


def main() -> None:
    args = _parse_args()
    device = _parse_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model_a, metadata_a = load_model(args.ckpt_a, device)
    model_b, metadata_b = load_model(args.ckpt_b, device)
    print(f"device={device} images={args.n} top_k={args.k}", flush=True)
    print(
        f"model_a={os.path.abspath(args.ckpt_a)} params={count_params(model_a):,}"
        f"{_checkpoint_step(metadata_a)}",
        flush=True,
    )
    print(
        f"model_b={os.path.abspath(args.ckpt_b)} params={count_params(model_b):,}"
        f"{_checkpoint_step(metadata_b)}",
        flush=True,
    )

    train_names, val_names = train_val_split()
    del train_names
    if not val_names:
        raise RuntimeError("held-out validation split is empty")
    if args.n > len(val_names):
        raise ValueError(f"--n={args.n} exceeds held-out split size {len(val_names)}")
    dataset = CanvasDataset(val_names[: args.n], real_prob=0.0, seed=args.seed)

    totals: dict[str, defaultdict[str, float]] = {
        "model_a_topk": defaultdict(float),
        "model_b_topk": defaultdict(float),
        "union_deduplicated": defaultdict(float),
        "intersection_deduplicated": defaultdict(float),
    }
    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("evaluator requires exact synthetic CanvasDataset examples")
        distance = chebyshev_distance(sample["perm"])
        candidates_a = _candidate_matrix(top_neighbours(learned_affinity(model_a, sample["tiles"], device), args.k))
        candidates_b = _candidate_matrix(top_neighbours(learned_affinity(model_b, sample["tiles"], device), args.k))
        # Both models place candidates on the same device; move only compact
        # boolean matrices to the label device if a caller uses a nonstandard
        # implementation that does not preserve it.
        candidates_a = candidates_a.to(distance.device)
        candidates_b = candidates_b.to(distance.device)
        variants = {
            "model_a_topk": candidates_a,
            "model_b_topk": candidates_b,
            "union_deduplicated": candidates_a | candidates_b,
            "intersection_deduplicated": candidates_a & candidates_b,
        }
        for label, candidates in variants.items():
            for key, value in candidate_metrics(distance, candidates).items():
                totals[label][key] += value
        print(f"processed {index + 1}/{args.n}", flush=True)

    print(
        "candidate relation is directed; union/intersection are per-anchor set operations; "
        "mutual_recall additionally requires the reverse directed edge.",
        flush=True,
    )
    for label in ("model_a_topk", "model_b_topk", "union_deduplicated", "intersection_deduplicated"):
        _print_report(label, _mean_metrics(totals[label], args.n), args.n)


if __name__ == "__main__":
    main()
