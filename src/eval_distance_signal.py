"""Audit whether tile affinities encode usable *global* spatial distance.

The affinity encoders were trained only as local candidate generators.  A good
top-K direct-neighbour recall does not by itself imply that their full 576 x
576 score matrix is a noisy metric from which a grid can be recovered.  This
read-only exact-synthetic diagnostic separates those two claims.

For raw z-normalized pixels, the radius-one encoder, the radius-three encoder,
and a rank-calibrated two-encoder ensemble, it reports:

* threshold ROC-AUC for Chebyshev distance <= 1, 3, 6, and 12;
* Spearman association with true Chebyshev and Euclidean grid distance;
* mean score in increasing true-distance bins and monotonicity violations;
* ordinal consistency inside random three-tile distance triangles; and
* mutual-kNN locality, connectivity, and geodesic-distance correlation.

The ensemble deliberately averages *within-anchor affinity percentiles*, not
raw cosine values.  The two encoders have different contrastive objectives, so
their cosine scales are not calibrated; ordinal fusion is the only claim made
by the existing candidate-union branch.

All labels come from ``CanvasDataset(real_prob=0)`` on held-out target images:
every corruption and shuffle is fresh, while ``perm`` remains exact.

Examples:

    python src/eval_distance_signal.py --n 12 --device cuda
    python src/eval_distance_signal.py --n 24 --graph-images 12 --graph-k 8,16,32,64
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import rankdata, spearmanr

from canvas_data import CanvasDataset
from config import GRID, NFRAG, SEED
from eval_affinity_graph import (
    _parse_device,
    chebyshev_distance,
    learned_affinity,
    load_model,
    raw_zpixel_affinity,
)
from imgio import train_val_split


DEFAULT_R1 = os.path.join("artifacts", "macro_affinity", "affinity_r1_1200_best.pt")
DEFAULT_R3 = os.path.join("artifacts", "macro_affinity", "affinity_r3_1000_best.pt")
THRESHOLDS = (1, 3, 6, 12)
DISTANCE_BINS = ((1, 1), (2, 3), (4, 6), (7, 12), (13, GRID - 1))


def _parse_ks(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--graph-k must be comma-separated integers") from exc
    if not parsed or any(k < 2 or k >= NFRAG for k in parsed):
        raise argparse.ArgumentTypeError(f"--graph-k entries must be in [2, {NFRAG - 1}]")
    return tuple(dict.fromkeys(parsed))


def _off_diagonal_mask() -> torch.Tensor:
    return ~torch.eye(NFRAG, dtype=torch.bool)


def euclidean_distance(perm: torch.Tensor) -> torch.Tensor:
    """Exact Euclidean clean-grid distance for every ordered input-tile pair."""
    if perm.ndim != 1 or perm.numel() != NFRAG:
        raise ValueError(f"perm must have shape ({NFRAG},), got {tuple(perm.shape)}")
    rows = torch.div(perm.long(), GRID, rounding_mode="floor")
    cols = torch.remainder(perm.long(), GRID)
    dr = rows[:, None].sub(rows[None, :]).float()
    dc = cols[:, None].sub(cols[None, :]).float()
    return dr.square().add(dc.square()).sqrt()


def row_percentile_affinity(affinity: torch.Tensor) -> torch.Tensor:
    """Map each non-self row to [0, 1] ordinal affinity percentiles.

    A percentile is deliberately invariant to an encoder's cosine scale.  The
    self entry is excluded before ranking and reset to zero afterward.
    """
    if tuple(affinity.shape) != (NFRAG, NFRAG):
        raise ValueError(f"affinity must have shape ({NFRAG}, {NFRAG})")
    values = affinity.detach().float().cpu().clone()
    values.fill_diagonal_(-torch.inf)
    # All practical encoder scores are continuous.  The stable sort simply
    # makes the rare numerical tie deterministic rather than model-dependent.
    order = torch.argsort(values, dim=-1, stable=True)
    ranks = torch.argsort(order, dim=-1, stable=True).float()
    percentiles = ranks.sub(1.0).div(float(NFRAG - 2)).clamp_(0.0, 1.0)
    percentiles.fill_diagonal_(0.0)
    return percentiles


def _auc_from_ranks(scores: np.ndarray, positive: np.ndarray) -> float:
    """Tie-correct ROC-AUC without relying on a classifier package."""
    count_positive = int(positive.sum())
    count_negative = int(positive.size - count_positive)
    if count_positive == 0 or count_negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    numerator = float(ranks[positive].sum()) - count_positive * (count_positive + 1) / 2.0
    return numerator / float(count_positive * count_negative)


def _scalar_metrics(
    affinity: torch.Tensor,
    cheb: torch.Tensor,
    euclid: torch.Tensor,
) -> dict[str, float]:
    """Global all-pair association metrics for one score matrix and image."""
    off_diagonal = _off_diagonal_mask()
    scores = affinity.detach().float().cpu()[off_diagonal].numpy().astype(np.float64, copy=False)
    cheb_values = cheb.detach().cpu()[off_diagonal].numpy().astype(np.float64, copy=False)
    euclid_values = euclid.detach().cpu()[off_diagonal].numpy().astype(np.float64, copy=False)
    metrics: dict[str, float] = {}
    for threshold in THRESHOLDS:
        metrics[f"auc_r{threshold}"] = _auc_from_ranks(scores, cheb_values <= threshold)
    metrics["spearman_cheb"] = float(spearmanr(scores, -cheb_values).statistic)
    metrics["spearman_euclid"] = float(spearmanr(scores, -euclid_values).statistic)

    bin_means: list[float] = []
    for low, high in DISTANCE_BINS:
        selected = scores[(cheb_values >= low) & (cheb_values <= high)]
        mean = float(selected.mean()) if selected.size else float("nan")
        metrics[f"bin_{low}_{high}_mean"] = mean
        bin_means.append(mean)
    finite = np.asarray(bin_means, dtype=np.float64)
    gaps = finite[:-1] - finite[1:]
    metrics["bin_monotone_fraction"] = float(np.mean(gaps >= 0.0))
    metrics["bin_all_monotone"] = float(np.all(gaps >= 0.0))
    metrics["bin_first_last_gap"] = float(finite[0] - finite[-1])
    metrics["bin_min_adjacent_gap"] = float(np.min(gaps))
    return metrics


def _triad_order_metrics(
    affinity: torch.Tensor,
    cheb: torch.Tensor,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Check ordinal distance consistency in random three-tile triangles.

    For every triplet, each unequal true-distance pair has a required affinity
    ordering.  A random score matrix gets 0.5 on this comparison; a usable
    multi-scale geometry should be substantially above that *and* preserve all
    three orderings for many non-degenerate triangles.
    """
    if samples < 1:
        return {}
    anchor = rng.integers(0, NFRAG, size=samples)
    middle = rng.integers(0, NFRAG - 1, size=samples)
    middle += middle >= anchor
    last = rng.integers(0, NFRAG - 2, size=samples)
    # Sample from the N-2 IDs remaining after removing anchor and middle.
    low = np.minimum(anchor, middle)
    high = np.maximum(anchor, middle)
    last += last >= low
    last += last >= high

    affinity_np = affinity.detach().float().cpu().numpy()
    distance_np = cheb.detach().cpu().numpy()
    scores = np.stack(
        (affinity_np[anchor, middle], affinity_np[anchor, last], affinity_np[middle, last]), axis=1
    )
    distance = np.stack(
        (distance_np[anchor, middle], distance_np[anchor, last], distance_np[middle, last]), axis=1
    )
    correct: list[np.ndarray] = []
    eligible: list[np.ndarray] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        valid = distance[:, left] != distance[:, right]
        # Shorter clean-grid distance should receive larger affinity.
        okay = (scores[:, left] - scores[:, right]) * (distance[:, left] - distance[:, right]) < 0.0
        correct.append(okay)
        eligible.append(valid)
    stacked_correct = np.stack(correct, axis=1)
    stacked_eligible = np.stack(eligible, axis=1)
    comparable = stacked_eligible.sum()
    result = {
        "triad_order_accuracy": float(
            stacked_correct[stacked_eligible].mean() if comparable else float("nan")
        ),
    }
    strict = stacked_eligible.all(axis=1)
    result["triad_all_orders_accuracy"] = float(
        stacked_correct[strict].all(axis=1).mean() if strict.any() else float("nan")
    )
    result["triad_strict_fraction"] = float(strict.mean())
    return result


def _topk_directed(affinity: np.ndarray, k: int) -> np.ndarray:
    """Return a boolean directed top-k relation from a CPU affinity matrix."""
    if affinity.shape != (NFRAG, NFRAG):
        raise ValueError("unexpected affinity shape")
    values = affinity.copy()
    np.fill_diagonal(values, -np.inf)
    # Sort only the high end: ordering inside the chosen set is irrelevant for
    # connectivity and much cheaper than ranking all 576 values in every row.
    selected = np.argpartition(values, kth=NFRAG - k, axis=1)[:, -k:]
    graph = np.zeros((NFRAG, NFRAG), dtype=bool)
    graph[np.arange(NFRAG)[:, None], selected] = True
    return graph


def _mutual_graph_metrics(
    affinity: torch.Tensor,
    cheb: torch.Tensor,
    *,
    k: int,
) -> dict[str, float]:
    """Assess local purity and global topology of a reciprocal kNN graph."""
    affinity_np = affinity.detach().float().cpu().numpy()
    distance = cheb.detach().cpu().numpy()
    directed = _topk_directed(affinity_np, k)
    mutual = directed & directed.T
    np.fill_diagonal(mutual, False)
    upper = np.triu(mutual, 1)
    edge_distance = distance[upper]
    edge_count = int(edge_distance.size)
    result: dict[str, float] = {
        f"k{k}_mutual_edges": float(edge_count),
        f"k{k}_mutual_mean_cheb": float(edge_distance.mean()) if edge_count else float("nan"),
    }
    for threshold in THRESHOLDS:
        result[f"k{k}_mutual_p_r{threshold}"] = float(
            (edge_distance <= threshold).mean() if edge_count else 0.0
        )

    sparse = csr_matrix(mutual.astype(np.uint8))
    component_count, labels = connected_components(sparse, directed=False, return_labels=True)
    counts = np.bincount(labels, minlength=component_count)
    largest_label = int(counts.argmax())
    members = np.flatnonzero(labels == largest_label)
    result[f"k{k}_components"] = float(component_count)
    result[f"k{k}_largest_fraction"] = float(members.size / NFRAG)

    # A graph can be connected simply because false semantic shortcuts bridge
    # it.  Its shortest-path distance must still correlate with true grid
    # separation for it to be a plausible input to a geodesic/Gromov solver.
    if members.size >= 16:
        local = sparse[members][:, members]
        hops = shortest_path(local, directed=False, unweighted=True)
        local_distance = distance[np.ix_(members, members)]
        off_diagonal = ~np.eye(members.size, dtype=bool)
        finite = np.isfinite(hops) & off_diagonal
        result[f"k{k}_hop_spearman_cheb"] = float(
            spearmanr(hops[finite], local_distance[finite]).statistic
        )
    else:
        result[f"k{k}_hop_spearman_cheb"] = float("nan")
    return result


def _aggregate(values: Mapping[str, Iterable[float]]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for key, items in values.items():
        array = np.asarray(list(items), dtype=np.float64)
        result[key] = (float(np.nanmean(array)), float(np.nanstd(array)))
    return result


def _fmt(metric: tuple[float, float]) -> str:
    mean, std = metric
    if not np.isfinite(mean):
        return "nan"
    # Keep terminal logs ASCII-clean on Windows PowerShell code pages.
    return f"{mean:.4f}+/-{std:.4f}"


def _print_summary(
    label: str,
    scalar: Mapping[str, tuple[float, float]],
    triangles: Mapping[str, tuple[float, float]],
    graph: Mapping[str, tuple[float, float]],
    ks: tuple[int, ...],
) -> None:
    print(f"[{label}]", flush=True)
    print(
        "  threshold ROC-AUC (Cheb<=1/3/6/12): "
        + " / ".join(_fmt(scalar[f"auc_r{threshold}"]) for threshold in THRESHOLDS),
        flush=True,
    )
    print(
        "  Spearman(score, -distance): "
        f"Cheb={_fmt(scalar['spearman_cheb'])} "
        f"Euclid={_fmt(scalar['spearman_euclid'])}",
        flush=True,
    )
    bin_text = " ".join(
        f"{low}-{high}:{_fmt(scalar[f'bin_{low}_{high}_mean'])}"
        for low, high in DISTANCE_BINS
    )
    print(f"  mean score by true Cheb bin: {bin_text}", flush=True)
    print(
        "  monotonic bins: "
        f"adjacent={_fmt(scalar['bin_monotone_fraction'])} "
        f"all={_fmt(scalar['bin_all_monotone'])} "
        f"first-last-gap={_fmt(scalar['bin_first_last_gap'])} "
        f"min-gap={_fmt(scalar['bin_min_adjacent_gap'])}",
        flush=True,
    )
    print(
        "  triangle ordinal consistency: "
        f"pair-order={_fmt(triangles['triad_order_accuracy'])} "
        f"all-three={_fmt(triangles['triad_all_orders_accuracy'])}",
        flush=True,
    )
    for k in ks:
        purity = "/".join(
            f"r{threshold}={_fmt(graph[f'k{k}_mutual_p_r{threshold}'])}"
            for threshold in THRESHOLDS
        )
        print(
            f"  mutual-kNN K={k}: edges={_fmt(graph[f'k{k}_mutual_edges'])} "
            f"meanCheb={_fmt(graph[f'k{k}_mutual_mean_cheb'])} "
            f"{purity} components={_fmt(graph[f'k{k}_components'])} "
            f"largest={_fmt(graph[f'k{k}_largest_fraction'])} "
            f"hop-vs-Cheb rho={_fmt(graph[f'k{k}_hop_spearman_cheb'])}",
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-r1", default=DEFAULT_R1, help="radius-one MacroAffinityNet checkpoint")
    parser.add_argument("--ckpt-r3", default=DEFAULT_R3, help="radius-three MacroAffinityNet checkpoint")
    parser.add_argument("--n", type=int, default=12, help="held-out exact synthetic images")
    parser.add_argument(
        "--graph-images",
        type=int,
        default=8,
        help="prefix of images that also run sparse graph/geodesic diagnostics",
    )
    parser.add_argument("--graph-k", type=_parse_ks, default=(16, 64), help="mutual kNN degrees")
    parser.add_argument("--triads", type=int, default=50_000, help="random unordered tile triplets per image")
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--seed", type=int, default=SEED + 9107, help="fresh synthetic corruption seed")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    if args.graph_images < 1:
        parser.error("--graph-images must be positive")
    if args.triads < 1:
        parser.error("--triads must be positive")
    return args


def main() -> None:
    args = _parse_args()
    device = _parse_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model_r1, meta_r1 = load_model(args.ckpt_r1, device)
    model_r3, meta_r3 = load_model(args.ckpt_r3, device)
    step_r1 = meta_r1.get("step") if isinstance(meta_r1, Mapping) else None
    step_r3 = meta_r3.get("step") if isinstance(meta_r3, Mapping) else None
    print(
        f"device={device} images={args.n} graph_images={min(args.graph_images, args.n)} "
        f"graph_k={args.graph_k} triads/image={args.triads}",
        flush=True,
    )
    print(f"r1={os.path.abspath(args.ckpt_r1)} step={step_r1}", flush=True)
    print(f"r3={os.path.abspath(args.ckpt_r3)} step={step_r3}", flush=True)
    print(
        "ensemble=mean of per-anchor affinity percentiles; all reported labels are exact synthetic held-out",
        flush=True,
    )

    _, val_names = train_val_split()
    if args.n > len(val_names):
        raise ValueError(f"--n={args.n} exceeds held-out split ({len(val_names)})")
    dataset = CanvasDataset(val_names[: args.n], real_prob=0.0, seed=args.seed)
    labels = ("raw_zpixel", "affinity_r1", "affinity_r3", "rank_ensemble")
    scalar: dict[str, defaultdict[str, list[float]]] = {
        label: defaultdict(list) for label in labels
    }
    triangles: dict[str, defaultdict[str, list[float]]] = {
        label: defaultdict(list) for label in labels
    }
    graph: dict[str, defaultdict[str, list[float]]] = {
        label: defaultdict(list) for label in labels
    }

    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("distance audit requires exact synthetic examples")
        cheb = chebyshev_distance(sample["perm"]).cpu()
        euclid = euclidean_distance(sample["perm"]).cpu()
        raw = raw_zpixel_affinity(sample["tiles"]).cpu()
        r1 = learned_affinity(model_r1, sample["tiles"], device).cpu()
        r3 = learned_affinity(model_r3, sample["tiles"], device).cpu()
        ensemble = row_percentile_affinity(r1).add(row_percentile_affinity(r3)).mul_(0.5)
        affinities = {
            "raw_zpixel": raw,
            "affinity_r1": r1,
            "affinity_r3": r3,
            "rank_ensemble": ensemble,
        }
        for label, affinity in affinities.items():
            for key, value in _scalar_metrics(affinity, cheb, euclid).items():
                scalar[label][key].append(value)
            for key, value in _triad_order_metrics(
                # Resetting the deterministic stream gives every score source
                # exactly the same sampled triangles for a fair comparison.
                affinity,
                cheb,
                samples=args.triads,
                rng=np.random.default_rng(args.seed + 10_000_019 * (index + 1)),
            ).items():
                triangles[label][key].append(value)
            if index < args.graph_images:
                for k in args.graph_k:
                    for key, value in _mutual_graph_metrics(affinity, cheb, k=k).items():
                        graph[label][key].append(value)
        print(f"processed {index + 1}/{args.n}", flush=True)

    print("\n=== Global distance-signal audit (mean+/-std across images) ===", flush=True)
    for label in labels:
        _print_summary(
            label,
            _aggregate(scalar[label]),
            _aggregate(triangles[label]),
            _aggregate(graph[label]),
            args.graph_k,
        )
    print(
        "\nInterpretation: A useful direct distance/Gromov input needs monotone multi-scale scores "
        "and a connected reciprocal graph whose shortest-path distance still tracks clean-grid distance. "
        "High r1 retrieval alone is insufficient because false long-range shortcuts collapse geodesics.",
        flush=True,
    )


if __name__ == "__main__":
    main()
