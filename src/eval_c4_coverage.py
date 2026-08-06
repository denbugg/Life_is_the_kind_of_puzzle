"""Hard coverage gate for a candidate-conditioned oriented 2x2 (C4) branch.

This diagnostic answers one narrow question before a C4 scorer is trained:
can a sparse graph made from the two frozen affinity encoders even *propose*
the real oriented blocks?  It never scores seams, learns weights, or injects
the known permutation into candidate construction.

For every anchor ``a`` the constructor:

1. takes the deduplicated union of the r1_1200 and r3_1000 top-K affinity
   lists and symmetrizes that relation;
2. ranks a symmetrized link by the best of its two directed candidate ranks;
3. ranks ordered hypothetical ``(right=b, down=c)`` pairs by the two anchor
   link ranks, keeps a deterministic oversampled prefix, and finds ``d`` in
   the intersection of ``b`` and ``c`` candidate sets;
4. ranks the resulting four-tile motifs by their four-link rank sum and keeps
   at most ``--motifs_per_anchor`` motifs for that anchor.

Only *after* this graph-only construction is frozen do we use the synthetic
permutation to ask whether the true ``(a, right, down, diagonal)`` tuple was
present.  Thus the reported C4 coverage is a real candidate-coverage gate,
not an oracle-assisted assembly score.

Examples:

    python src/eval_c4_coverage.py --n 8 --k 64 --motifs_per_anchor 128
    python src/eval_c4_coverage.py --n 1 --device cpu
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from canvas_data import CanvasDataset
from config import GRID, NFRAG, SEED
from imgio import train_val_split
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_AFFINITY_A = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r1_1200_best.pt"
)
DEFAULT_AFFINITY_B = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r3_1000_best.pt"
)

# The final candidate budget is the public contract.  The two internal caps
# merely make ranking the sparse C4 product bounded and deterministic.
PAIR_OVERSAMPLE = 8
D_PER_PAIR = 4
RANK_INF = np.int32(1_000_000)


@dataclass
class MotifGeneration:
    """Graph-only candidate motifs, retained separately for each anchor."""

    motifs: list[list[tuple[int, int, int]]]
    pair_prefixes: list[set[tuple[int, int]]]


def _parse_device(value: str | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return device


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _candidate_matrices(candidates: Tensor, valid: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return directed and symmetrized candidate relations plus symmetrized ranks.

    Candidate-list position is a rank (one-based).  A symmetrized edge gets the
    smaller rank of its two directed appearances.  This is deliberately the
    only information used to prune motif candidates.
    """
    if candidates.ndim != 2 or candidates.shape[0] != NFRAG:
        raise ValueError(f"candidates must have shape ({NFRAG},K), got {tuple(candidates.shape)}")
    if valid.shape != candidates.shape:
        raise ValueError("valid mask must have the same shape as candidates")
    candidate_np = candidates.detach().cpu().numpy().astype(np.int64, copy=False)
    valid_np = valid.detach().cpu().numpy().astype(bool, copy=False)
    if np.any(candidate_np < 0) or np.any(candidate_np >= NFRAG):
        raise ValueError("candidate indices lie outside the tile bag")

    directed = np.zeros((NFRAG, NFRAG), dtype=bool)
    directed_rank = np.full((NFRAG, NFRAG), RANK_INF, dtype=np.int32)
    anchors = np.arange(NFRAG, dtype=np.int64)
    for column in range(candidate_np.shape[1]):
        keep = valid_np[:, column]
        source = anchors[keep]
        target = candidate_np[keep, column]
        directed[source, target] = True
        # The union routine already masks duplicate targets.  minimum.at keeps
        # this helper correct even if a caller supplies a duplicate manually.
        np.minimum.at(directed_rank, (source, target), np.int32(column + 1))
    np.fill_diagonal(directed, False)
    np.fill_diagonal(directed_rank, RANK_INF)

    symmetric = directed | directed.T
    symmetric_rank = np.minimum(directed_rank, directed_rank.T)
    np.fill_diagonal(symmetric, False)
    np.fill_diagonal(symmetric_rank, RANK_INF)
    return directed, symmetric, symmetric_rank


def _ranked_neighbours(adjacency: np.ndarray, ranks: np.ndarray, anchor: int) -> np.ndarray:
    """Return an anchor's symmetric neighbours ordered only by graph rank/id."""
    neighbours = np.flatnonzero(adjacency[anchor]).astype(np.int64, copy=False)
    if neighbours.size < 2:
        return neighbours
    order = np.lexsort((neighbours, ranks[anchor, neighbours]))
    return neighbours[order]


def _ordered_pair_prefix(
    neighbours: np.ndarray, ranks_from_anchor: np.ndarray, pair_budget: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the deterministic lowest-rank ordered ``(b,c)`` pair prefix."""
    count = int(neighbours.size)
    if count < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    right = np.repeat(neighbours, count)
    down = np.tile(neighbours, count)
    different = right != down
    right, down = right[different], down[different]
    score = ranks_from_anchor[right].astype(np.int64) + ranks_from_anchor[down].astype(np.int64)
    # The labels right/down are hypothetical at this stage, so preserving both
    # orders is essential: it lets the true oriented C4 appear without a
    # directional oracle.
    order = np.lexsort((down, right, score))
    order = order[: min(pair_budget, order.size)]
    return right[order], down[order], score[order]


def _motifs_for_anchor(
    anchor: int,
    adjacency: np.ndarray,
    ranks: np.ndarray,
    *,
    motifs_per_anchor: int,
) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]]]:
    """Build one bounded ranked C4 list without receiving labels/permutation."""
    neighbours = _ranked_neighbours(adjacency, ranks, anchor)
    pair_budget = max(motifs_per_anchor, PAIR_OVERSAMPLE * motifs_per_anchor)
    right, down, pair_score = _ordered_pair_prefix(neighbours, ranks[anchor], pair_budget)
    pair_prefix = {(int(b), int(c)) for b, c in zip(right, down)}
    if right.size == 0:
        return [], pair_prefix

    # Shape is only (pair_budget, 576), e.g. 1024 x 576 for the default.  The
    # staged rank prefix prevents materializing the cubic/all-pairs universe.
    common = adjacency[right] & adjacency[down]
    row = np.arange(right.size)
    common[row, anchor] = False
    common[row, right] = False
    common[row, down] = False
    if not bool(common.any()):
        return [], pair_prefix

    four_edge_score = (
        pair_score[:, None]
        + ranks[right].astype(np.int64)
        + ranks[down].astype(np.int64)
    )
    four_edge_score[~common] = int(RANK_INF)
    # It is enough to retain a few d choices per (b,c) pair before the final
    # global top-M: all other d choices have the same first two links and a
    # larger last-two-link score.
    choices = min(D_PER_PAIR, NFRAG)
    # Make argpartition's boundary deterministic: the d id is an explicit
    # secondary key, rather than relying on an implementation's tie order.
    d_tiebreak = four_edge_score * (NFRAG + 1) + np.arange(NFRAG, dtype=np.int64)
    d_index = np.argpartition(d_tiebreak, kth=choices - 1, axis=1)[:, :choices]
    d_score = np.take_along_axis(four_edge_score, d_index, axis=1)
    keep = d_score < int(RANK_INF)
    if not bool(keep.any()):
        return [], pair_prefix

    source_index, local_index = np.nonzero(keep)
    candidate_right = right[source_index]
    candidate_down = down[source_index]
    candidate_d = d_index[source_index, local_index]
    candidate_score = d_score[source_index, local_index]
    order = np.lexsort((candidate_d, candidate_down, candidate_right, candidate_score))
    order = order[: min(motifs_per_anchor, order.size)]
    motifs = [
        (int(candidate_right[index]), int(candidate_down[index]), int(candidate_d[index]))
        for index in order
    ]
    return motifs, pair_prefix


def generate_motifs(
    adjacency: np.ndarray, ranks: np.ndarray, *, motifs_per_anchor: int
) -> MotifGeneration:
    """Generate bounded graph-only C4 candidate lists for all 576 anchors."""
    if adjacency.shape != (NFRAG, NFRAG) or ranks.shape != (NFRAG, NFRAG):
        raise ValueError("adjacency and ranks must both be 576x576")
    if adjacency.dtype != bool:
        raise ValueError("adjacency must be boolean")
    if motifs_per_anchor < 1:
        raise ValueError("motifs_per_anchor must be positive")
    motifs: list[list[tuple[int, int, int]]] = []
    pair_prefixes: list[set[tuple[int, int]]] = []
    for anchor in range(NFRAG):
        anchor_motifs, anchor_pairs = _motifs_for_anchor(
            anchor, adjacency, ranks, motifs_per_anchor=motifs_per_anchor
        )
        motifs.append(anchor_motifs)
        pair_prefixes.append(anchor_pairs)
    return MotifGeneration(motifs=motifs, pair_prefixes=pair_prefixes)


def _inverse_permutation(perm: Tensor) -> np.ndarray:
    """Map true clean cell -> input-tile index for one exact synthetic sample."""
    values = perm.detach().cpu().numpy().astype(np.int64, copy=False)
    if values.shape != (NFRAG,) or not np.array_equal(np.sort(values), np.arange(NFRAG)):
        raise ValueError("perm must be one exact input-tile -> clean-cell permutation")
    inverse = np.empty(NFRAG, dtype=np.int64)
    inverse[values] = np.arange(NFRAG, dtype=np.int64)
    return inverse


def true_oriented_c4(perm: Tensor) -> np.ndarray:
    """Return ``(a,right,down,diagonal)`` true C4 tuples in input-tile ids."""
    clean_to_input = _inverse_permutation(perm)
    rows, cols = np.meshgrid(np.arange(GRID - 1), np.arange(GRID - 1), indexing="ij")
    top_left = (rows * GRID + cols).reshape(-1)
    return np.stack(
        (
            clean_to_input[top_left],
            clean_to_input[top_left + 1],
            clean_to_input[top_left + GRID],
            clean_to_input[top_left + GRID + 1],
        ),
        axis=1,
    )


def true_direct_edges(perm: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return each physical direct board edge once, in arbitrary true orientation."""
    clean_to_input = _inverse_permutation(perm)
    cells = np.arange(NFRAG, dtype=np.int64).reshape(GRID, GRID)
    source_cells = np.concatenate((cells[:, :-1].reshape(-1), cells[:-1, :].reshape(-1)))
    target_cells = np.concatenate((cells[:, 1:].reshape(-1), cells[1:, :].reshape(-1)))
    return clean_to_input[source_cells], clean_to_input[target_cells]


def _rank_summary(values: np.ndarray) -> str:
    """Compact rank distribution; input has only finite observed ranks."""
    if values.size == 0:
        return "none"
    return (
        f"mean={values.mean():.1f} median={np.median(values):.0f} "
        f"p90={np.percentile(values, 90):.0f} max={values.max():.0f}"
    )


def _intersection_rank(
    anchor: int,
    right: int,
    down: int,
    diagonal: int,
    adjacency: np.ndarray,
    ranks: np.ndarray,
) -> int:
    """One-based graph-rank of the true d within N(right) intersection N(down)."""
    common = np.flatnonzero(adjacency[right] & adjacency[down]).astype(np.int64, copy=False)
    # Match the constructor exactly: a, b, and c cannot be used again as d.
    common = common[(common != anchor) & (common != right) & (common != down)]
    if common.size == 0 or not np.any(common == diagonal):
        return 0
    score = ranks[right, common].astype(np.int64) + ranks[down, common].astype(np.int64)
    order = np.lexsort((common, score))
    position = np.flatnonzero(common[order] == diagonal)
    return int(position[0] + 1) if position.size else 0


@dataclass
class Totals:
    """Additive counts and rank samples across fresh synthetic images."""

    images: int = 0
    directed_edges: int = 0
    symmetric_edges: int = 0
    motifs: int = 0
    min_motifs: int = NFRAG
    max_motifs: int = 0
    c4_total: int = 0
    c4_all_four_edges: int = 0
    c4_pair_prefix: int = 0
    c4_pair_prefix_given_edges: int = 0
    c4_motif: int = 0
    direct_edge_total: int = 0
    direct_edge_present: int = 0
    direct_edge_ranks: list[int] | None = None
    c4_worst_edge_ranks: list[int] | None = None
    c4_intersection_ranks: list[int] | None = None
    true_motif_ranks: list[int] | None = None

    def __post_init__(self) -> None:
        self.direct_edge_ranks = []
        self.c4_worst_edge_ranks = []
        self.c4_intersection_ranks = []
        self.true_motif_ranks = []


def accumulate_sample(
    totals: Totals,
    perm: Tensor,
    directed: np.ndarray,
    symmetric: np.ndarray,
    ranks: np.ndarray,
    generated: MotifGeneration,
) -> None:
    """Evaluate frozen graph candidates against exact labels after construction."""
    counts = np.asarray([len(motifs) for motifs in generated.motifs], dtype=np.int64)
    totals.images += 1
    totals.directed_edges += int(directed.sum())
    totals.symmetric_edges += int(symmetric.sum())
    totals.motifs += int(counts.sum())
    totals.min_motifs = min(totals.min_motifs, int(counts.min()))
    totals.max_motifs = max(totals.max_motifs, int(counts.max()))

    source, target = true_direct_edges(perm)
    edge_rank = ranks[source, target]
    edge_present = edge_rank < RANK_INF
    totals.direct_edge_total += int(edge_rank.size)
    totals.direct_edge_present += int(edge_present.sum())
    totals.direct_edge_ranks.extend(edge_rank[edge_present].astype(int).tolist())

    truth = true_oriented_c4(perm)
    totals.c4_total += int(truth.shape[0])
    for anchor, right, down, diagonal in truth:
        anchor, right, down, diagonal = map(int, (anchor, right, down, diagonal))
        edge_ranks = np.asarray(
            (
                ranks[anchor, right],
                ranks[anchor, down],
                ranks[right, diagonal],
                ranks[down, diagonal],
            ),
            dtype=np.int32,
        )
        all_four = bool(np.all(edge_ranks < RANK_INF))
        if all_four:
            totals.c4_all_four_edges += 1
            totals.c4_worst_edge_ranks.append(int(edge_ranks.max()))
            intersection_rank = _intersection_rank(
                anchor, right, down, diagonal, symmetric, ranks
            )
            if not intersection_rank:
                raise AssertionError("all four candidate links must place true d in the intersection")
            totals.c4_intersection_ranks.append(intersection_rank)

        if (right, down) in generated.pair_prefixes[anchor]:
            totals.c4_pair_prefix += 1
            if all_four:
                totals.c4_pair_prefix_given_edges += 1
        rank_map = {motif: index + 1 for index, motif in enumerate(generated.motifs[anchor])}
        motif_rank = rank_map.get((right, down, diagonal), 0)
        if motif_rank:
            totals.c4_motif += 1
            totals.true_motif_ranks.append(motif_rank)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12, help="fresh held-out synthetic images")
    parser.add_argument("--k", type=int, default=64, help="top-K candidates per frozen encoder")
    parser.add_argument(
        "--motifs_per_anchor", type=int, default=128, help="final sparse oriented C4 budget per anchor"
    )
    parser.add_argument("--device", default=None, help="cuda when available by default")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    if not 1 <= args.k < NFRAG:
        parser.error(f"--k must lie in [1, {NFRAG - 1}]")
    if args.motifs_per_anchor < 1:
        parser.error("--motifs_per_anchor must be positive")
    return args


def _print_report(totals: Totals, *, k: int, motifs_per_anchor: int) -> None:
    anchors = totals.images * NFRAG
    print("\n[candidate-conditioned oriented C4 coverage]", flush=True)
    print(
        "  graph: "
        f"directed_union_edges/anchor={_ratio(totals.directed_edges, anchors):.2f} "
        f"symmetrized_edges/anchor={_ratio(totals.symmetric_edges, anchors):.2f}",
        flush=True,
    )
    print(
        "  motifs: "
        f"total={totals.motifs:,} per_anchor={_ratio(totals.motifs, anchors):.2f} "
        f"min/max={totals.min_motifs}/{totals.max_motifs} "
        f"budget={motifs_per_anchor} pair_prefix={PAIR_OVERSAMPLE * motifs_per_anchor} "
        f"d_per_pair={D_PER_PAIR}",
        flush=True,
    )
    print(
        "  true direct physical edges (sym graph): "
        f"coverage={_ratio(totals.direct_edge_present, totals.direct_edge_total):.4f} "
        f"rank[{_rank_summary(np.asarray(totals.direct_edge_ranks, dtype=np.int64))}]",
        flush=True,
    )
    print(
        "  true oriented 2x2: "
        f"blocks={totals.c4_total:,} "
        f"all_four_direct_edges={_ratio(totals.c4_all_four_edges, totals.c4_total):.4f} "
        f"pair_in_prefix={_ratio(totals.c4_pair_prefix, totals.c4_total):.4f} "
        f"exact_motif_coverage={_ratio(totals.c4_motif, totals.c4_total):.4f}",
        flush=True,
    )
    print(
        "  conditional on all four true direct edges being candidates: "
        f"pair_in_prefix={_ratio(totals.c4_pair_prefix_given_edges, totals.c4_all_four_edges):.4f} "
        f"exact_motif_coverage={_ratio(totals.c4_motif, totals.c4_all_four_edges):.4f}",
        flush=True,
    )
    print(
        "  true C4 candidate ranks: "
        f"worst_of_4_edges[{_rank_summary(np.asarray(totals.c4_worst_edge_ranks, dtype=np.int64))}] "
        f"d_in_intersection[{_rank_summary(np.asarray(totals.c4_intersection_ranks, dtype=np.int64))}] "
        f"retained_motif_rank[{_rank_summary(np.asarray(totals.true_motif_ranks, dtype=np.int64))}]",
        flush=True,
    )
    print(
        "  Labels were queried only after graph/motif construction; no true edge or tile was added "
        f"to the r1_1200+r3_1000 top{k} candidate relation.",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    device = _parse_device(args.device)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    affinity_a, _, kwargs_a = load_frozen_affinity(DEFAULT_AFFINITY_A, device)
    affinity_b, _, kwargs_b = load_frozen_affinity(DEFAULT_AFFINITY_B, device)
    print(
        f"device={device} images={args.n} top_k_per_encoder={args.k} "
        f"motifs_per_anchor={args.motifs_per_anchor}",
        flush=True,
    )
    print(f"affinity_a={DEFAULT_AFFINITY_A} kwargs={kwargs_a}", flush=True)
    print(f"affinity_b={DEFAULT_AFFINITY_B} kwargs={kwargs_b}", flush=True)

    _, val_names = train_val_split()
    if args.n > len(val_names):
        raise ValueError(f"--n={args.n} exceeds held-out split size {len(val_names)}")
    dataset = CanvasDataset(val_names[: args.n], real_prob=0.0, seed=SEED)

    totals = Totals()
    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("C4 coverage requires exact fresh synthetic CanvasDataset samples")
        tiles = sample["tiles"].unsqueeze(0).to(device, non_blocking=device.type == "cuda")
        with torch.inference_mode():
            candidates, valid = mine_affinity_candidates(
                affinity_a,
                tiles,
                candidate_k=args.k,
                device=device,
                affinity_secondary=affinity_b,
            )
        directed, symmetric, ranks = _candidate_matrices(candidates[0], valid[0])
        # Crucially, generate_motifs does not receive sample["perm"].
        generated = generate_motifs(
            symmetric, ranks, motifs_per_anchor=args.motifs_per_anchor
        )
        accumulate_sample(totals, sample["perm"], directed, symmetric, ranks, generated)
        print(
            f"processed {index + 1}/{args.n}: motifs={sum(map(len, generated.motifs)):,} "
            f"exact_c4_so_far={_ratio(totals.c4_motif, totals.c4_total):.4f}",
            flush=True,
        )

    _print_report(totals, k=args.k, motifs_per_anchor=args.motifs_per_anchor)


if __name__ == "__main__":
    main()
