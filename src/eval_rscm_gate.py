"""Gate reciprocal slot-capacitated matching on sparse PairwiseNet seams.

This is a deliberately small diagnostic for the only remaining useful part of
the historical seam model.  It never evaluates PairwiseNet over all
``576 * 575`` tile pairs.  Instead it reuses the frozen affinity-union
candidate graph from :mod:`eval_pair_affinity_fusion`, scores its proposed
pairs in all four physical layouts, and applies a stricter graph rule:

* a relation ``i --d--> j`` must be present in both affinity candidate rows;
* the reverse row must support ``j --inverse(d)--> i``;
* its physical weight is ``min(row_z[i,j,d], row_z[j,i,inverse(d)])``;
* the two directed views are canonicalised into one physical relation;
* greedy RSCM accepts descending-weight relations only when neither endpoint
  has already consumed the needed U/D/L/R slot, and an unordered tile pair is
  never used twice.

The baseline intentionally omits the last global constraints: it reports raw
reciprocal threshold and row-top selections over exactly the same sparse
relations.  This answers whether slot capacity itself turns the promising
reciprocal seam evidence into cleaner assembly seeds.

The default is one fresh, exact synthetic held-out puzzle.  That is a gate,
not a leaderboard claim: increase the sample count only after explicitly
requesting a broader run.

Examples
--------

    python src/eval_rscm_gate.py --smoke
    python src/eval_rscm_gate.py --device cuda
    python src/eval_rscm_gate.py --thresholds 2.0,2.5,3.0 --tops 1,2,4,8
"""
from __future__ import annotations

import argparse
import math
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from canvas_data import CanvasDataset
from config import GRID, NFRAG, SEED
from direct_pose import DIRECT_CLASS_COUNT, DIRECT_OFFSETS
from eval_pair_affinity_fusion import (
    DEFAULT_AFFINITY_CKPT,
    DEFAULT_AFFINITY_CKPT2,
    _pair_checkpoint_paths,
    _parse_device,
    _row_z,
    load_pair_ensemble,
    score_pairwise_directions,
)
from imgio import train_val_split
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


# DirectPose's fixed class contract: U <-> D and L <-> R.  Keep a plain tuple
# here because this evaluator deliberately handles physical candidate records
# on CPU dictionaries rather than materialising a 576x576 score/rank matrix.
INVERSE_DIRECTION = (1, 0, 3, 2)
TRUE_PHYSICAL_EDGES = 2 * GRID * (GRID - 1)


@dataclass(frozen=True)
class PhysicalRelation:
    """One canonical candidate relation ``anchor --direction--> target``.

    ``anchor`` is always the numerically smaller tile id.  This convention
    collapses the two directed affinity views of one proposed physical edge.
    It intentionally does *not* collapse differing directional hypotheses for
    the same unordered pair; RSCM's pair constraint then prevents a tile pair
    from being used as both, for example, a right and an above relation.
    """

    anchor: int
    target: int
    direction: int
    anchor_rank: int
    target_rank: int
    anchor_z: float
    target_z: float
    weight: float

    @property
    def pair_key(self) -> tuple[int, int]:
        return self.anchor, self.target


@dataclass(frozen=True)
class PhysicalMetrics:
    """Metrics defined over physical, not duplicate-directed, relations."""

    selected: int
    correct: int
    precision: float
    coverage: float
    edges_per_tile: float
    horizontal: int
    vertical: int
    horizontal_fraction: float
    horizontal_precision: float
    vertical_precision: float
    correct_components: int
    correct_component_nodes: int
    correct_largest_nodes: int
    correct_largest_edges: int
    correct_mean_nodes: float


class _UnionFind:
    """Tiny deterministic union-find used only for correct selected edges."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _float_list(values: Sequence[str], *, name: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for raw in values:
        for item in str(raw).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                value = float(item)
            except ValueError as exc:
                raise ValueError(f"invalid {name} value {item!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} values must be finite, got {item!r}")
            parsed.append(value)
    if not parsed:
        raise ValueError(f"at least one {name} value is required")
    return tuple(sorted(set(parsed)))


def _int_list(values: Sequence[str], *, name: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for raw in values:
        for item in str(raw).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                value = int(item)
            except ValueError as exc:
                raise ValueError(f"invalid {name} value {item!r}") from exc
            if value < 1:
                raise ValueError(f"{name} values must be positive, got {value}")
            parsed.append(value)
    if not parsed:
        raise ValueError(f"at least one {name} value is required")
    return tuple(sorted(set(parsed)))


def _require_shapes(candidates: Tensor, valid: Tensor, row_z: Tensor) -> None:
    if candidates.ndim != 2 or candidates.shape[0] != NFRAG:
        raise ValueError(f"candidates must have shape ({NFRAG},K), got {tuple(candidates.shape)}")
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a bool matrix aligned with candidates")
    if row_z.shape != (*candidates.shape, DIRECT_CLASS_COUNT):
        raise ValueError(
            "row_z must have shape "
            f"({NFRAG},K,{DIRECT_CLASS_COUNT}), got {tuple(row_z.shape)}"
        )


def reciprocal_physical_relations(
    candidates: Tensor,
    valid: Tensor,
    row_z: Tensor,
) -> list[PhysicalRelation]:
    """Build weighted canonical relations using only the sparse candidate graph.

    The dictionary stores one rank for each *valid proposed* ordered pair.  It
    is intentionally not a dense rank tensor: absent affinity candidates stay
    absent, and PairwiseNet has never been asked to score them.
    """
    _require_shapes(candidates, valid, row_z)
    candidates_cpu = candidates.detach().to(device="cpu", dtype=torch.long)
    valid_cpu = valid.detach().to(device="cpu", dtype=torch.bool)
    z_cpu = row_z.detach().float().cpu()
    width = int(candidates_cpu.shape[1])

    rank_by_pair: dict[tuple[int, int], int] = {}
    for anchor in range(NFRAG):
        for rank in range(width):
            if not bool(valid_cpu[anchor, rank]):
                continue
            target = int(candidates_cpu[anchor, rank])
            if target == anchor:
                raise RuntimeError("affinity candidate graph contains a self relation")
            key = anchor, target
            if key in rank_by_pair:
                raise RuntimeError(f"duplicate valid affinity candidate {key}")
            rank_by_pair[key] = rank

    canonical: dict[tuple[int, int, int], PhysicalRelation] = {}
    for (source, target), source_rank in rank_by_pair.items():
        reverse_rank = rank_by_pair.get((target, source))
        if reverse_rank is None:
            continue
        for direction in range(DIRECT_CLASS_COUNT):
            inverse = INVERSE_DIRECTION[direction]
            source_z = float(z_cpu[source, source_rank, direction])
            reverse_z = float(z_cpu[target, reverse_rank, inverse])
            if not math.isfinite(source_z) or not math.isfinite(reverse_z):
                continue
            if source < target:
                anchor, other = source, target
                canonical_direction = direction
                anchor_rank, other_rank = source_rank, reverse_rank
                anchor_z, other_z = source_z, reverse_z
            else:
                anchor, other = target, source
                canonical_direction = inverse
                anchor_rank, other_rank = reverse_rank, source_rank
                anchor_z, other_z = reverse_z, source_z
            relation = PhysicalRelation(
                anchor=anchor,
                target=other,
                direction=canonical_direction,
                anchor_rank=anchor_rank,
                target_rank=other_rank,
                anchor_z=anchor_z,
                target_z=other_z,
                weight=min(anchor_z, other_z),
            )
            key = anchor, other, canonical_direction
            previous = canonical.get(key)
            # The reverse directed traversal creates the same physical record.
            # Preserve the stronger copy if malformed input introduces tiny
            # numerical differences, while retaining deterministic semantics.
            if previous is None or relation.weight > previous.weight:
                canonical[key] = relation

    return sorted(
        canonical.values(),
        key=lambda relation: (relation.anchor, relation.target, relation.direction),
    )


def _oriented_top_mask(row_z: Tensor, valid: Tensor, top: int) -> Tensor:
    """Keep each source tile's top oriented seam hypotheses.

    This is deliberately a top-*orientation* baseline rather than the old
    max-over-orientation candidate top-k: reciprocal constraints need an
    explicit inverse U/D/L/R hypothesis at both endpoints.
    """
    if valid.ndim != 2 or valid.shape[0] != NFRAG or valid.dtype != torch.bool:
        raise ValueError(f"valid must have shape ({NFRAG},K) and bool dtype")
    if row_z.shape != (*valid.shape, DIRECT_CLASS_COUNT):
        raise ValueError(
            "row_z must have shape "
            f"({NFRAG},K,{DIRECT_CLASS_COUNT}), got {tuple(row_z.shape)}"
        )
    if top < 1:
        raise ValueError("top must be positive")
    flat = torch.where(
        valid.unsqueeze(-1), row_z, torch.full_like(row_z, -torch.inf)
    ).reshape(NFRAG, -1)
    count = min(top, int(flat.shape[1]))
    indices = flat.topk(count, dim=-1).indices
    selected = torch.zeros_like(flat, dtype=torch.bool)
    selected.scatter_(1, indices, True)
    return selected.reshape_as(row_z) & valid.unsqueeze(-1)


def _top_reciprocal_relations(
    relations: Iterable[PhysicalRelation], row_z: Tensor, valid: Tensor, top: int
) -> list[PhysicalRelation]:
    """Raw reciprocal row-top baseline, with no cross-edge capacity rules."""
    selected = _oriented_top_mask(row_z, valid, top).cpu()
    output: list[PhysicalRelation] = []
    for relation in relations:
        inverse = INVERSE_DIRECTION[relation.direction]
        if bool(selected[relation.anchor, relation.anchor_rank, relation.direction]) and bool(
            selected[relation.target, relation.target_rank, inverse]
        ):
            output.append(relation)
    return output


def _threshold_relations(
    relations: Iterable[PhysicalRelation], threshold: float
) -> list[PhysicalRelation]:
    """Raw reciprocal threshold baseline; min weight means both views pass."""
    return [relation for relation in relations if relation.weight >= threshold]


def rscm_greedy(relations: Iterable[PhysicalRelation]) -> list[PhysicalRelation]:
    """Greedily choose reciprocal relations under endpoint slot capacities.

    Each accepted relation consumes two physical slots: ``anchor.direction``
    and ``target.inverse(direction)``.  ``used_pairs`` additionally forbids
    contradictory directional hypotheses joining the same tile pair.
    """
    ordered = sorted(
        relations,
        key=lambda relation: (
            -relation.weight,
            relation.anchor,
            relation.target,
            relation.direction,
        ),
    )
    used_slots: set[tuple[int, int]] = set()
    used_pairs: set[tuple[int, int]] = set()
    selected: list[PhysicalRelation] = []
    for relation in ordered:
        inverse = INVERSE_DIRECTION[relation.direction]
        anchor_slot = relation.anchor, relation.direction
        target_slot = relation.target, inverse
        if relation.pair_key in used_pairs:
            continue
        if anchor_slot in used_slots or target_slot in used_slots:
            continue
        used_pairs.add(relation.pair_key)
        used_slots.add(anchor_slot)
        used_slots.add(target_slot)
        selected.append(relation)
    return selected


def _expected_direction(perm: Tensor, anchor: int, target: int) -> int | None:
    """Return target's exact direct direction from anchor, if it is cardinal."""
    anchor_cell = int(perm[anchor])
    target_cell = int(perm[target])
    delta = (
        target_cell // GRID - anchor_cell // GRID,
        target_cell % GRID - anchor_cell % GRID,
    )
    try:
        return DIRECT_OFFSETS.index(delta)
    except ValueError:
        return None


def _correct_component_stats(correct: Sequence[PhysicalRelation]) -> tuple[int, int, int, int, float]:
    """Return non-singleton component count/nodes/largest nodes/edges/mean."""
    if not correct:
        return 0, 0, 0, 0, 0.0
    union_find = _UnionFind(NFRAG)
    for relation in correct:
        union_find.union(relation.anchor, relation.target)
    edges_by_root: dict[int, int] = {}
    for relation in correct:
        root = union_find.find(relation.anchor)
        edges_by_root[root] = edges_by_root.get(root, 0) + 1
    component_sizes = [union_find.size[root] for root in edges_by_root]
    component_edges = list(edges_by_root.values())
    return (
        len(component_sizes),
        sum(component_sizes),
        max(component_sizes),
        max(component_edges),
        float(np.mean(component_sizes)),
    )


def physical_metrics(relations: Sequence[PhysicalRelation], perm: Tensor) -> PhysicalMetrics:
    """Score a physical edge set against an exact synthetic permutation."""
    if perm.ndim != 1 or perm.numel() != NFRAG:
        raise ValueError(f"perm must have shape ({NFRAG},), got {tuple(perm.shape)}")
    perm_cpu = perm.detach().to(device="cpu", dtype=torch.long)
    correct = [
        relation
        for relation in relations
        if _expected_direction(perm_cpu, relation.anchor, relation.target) == relation.direction
    ]
    horizontal = sum(relation.direction >= 2 for relation in relations)
    vertical = len(relations) - horizontal
    correct_horizontal = sum(relation.direction >= 2 for relation in correct)
    correct_vertical = len(correct) - correct_horizontal
    (
        component_count,
        component_nodes,
        largest_nodes,
        largest_edges,
        mean_nodes,
    ) = _correct_component_stats(correct)
    selected = len(relations)
    return PhysicalMetrics(
        selected=selected,
        correct=len(correct),
        precision=float(len(correct) / selected) if selected else 0.0,
        coverage=float(len(correct) / TRUE_PHYSICAL_EDGES),
        edges_per_tile=float(selected / NFRAG),
        horizontal=horizontal,
        vertical=vertical,
        horizontal_fraction=float(horizontal / selected) if selected else 0.0,
        horizontal_precision=float(correct_horizontal / horizontal) if horizontal else 0.0,
        vertical_precision=float(correct_vertical / vertical) if vertical else 0.0,
        correct_components=component_count,
        correct_component_nodes=component_nodes,
        correct_largest_nodes=largest_nodes,
        correct_largest_edges=largest_edges,
        correct_mean_nodes=mean_nodes,
    )


def _print_metrics(label: str, metrics: PhysicalMetrics) -> None:
    ratio = float(metrics.horizontal / metrics.vertical) if metrics.vertical else float("inf")
    print(
        f"  [{label}] physical selected={metrics.selected} "
        f"({metrics.edges_per_tile:.4f}/tile) exact p={metrics.precision:.4f} "
        f"true-edge coverage={metrics.coverage:.4f}",
        flush=True,
    )
    print(
        f"    H/V={metrics.horizontal}/{metrics.vertical} "
        f"(H share={metrics.horizontal_fraction:.4f}, H/V={ratio:.4f}; "
        f"exact p H/V={metrics.horizontal_precision:.4f}/{metrics.vertical_precision:.4f})",
        flush=True,
    )
    print(
        f"    correct-only components={metrics.correct_components} "
        f"nodes={metrics.correct_component_nodes}/{NFRAG} "
        f"largest={metrics.correct_largest_nodes} nodes, {metrics.correct_largest_edges} edges "
        f"mean_nontrivial={metrics.correct_mean_nodes:.2f}",
        flush=True,
    )


def _perfect_sparse_inputs(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build a data-free exact reciprocal graph for the RSCM smoke test."""
    anchors = torch.arange(NFRAG, device=device)
    rows = torch.div(anchors, GRID, rounding_mode="floor")
    cols = torch.remainder(anchors, GRID)
    candidates = torch.stack(
        (
            torch.where(rows.gt(0), anchors - GRID, anchors),
            torch.where(rows.lt(GRID - 1), anchors + GRID, anchors),
            torch.where(cols.gt(0), anchors - 1, anchors),
            torch.where(cols.lt(GRID - 1), anchors + 1, anchors),
        ),
        dim=-1,
    )
    valid = torch.ones_like(candidates, dtype=torch.bool)
    valid[rows.eq(0), 0] = False
    valid[rows.eq(GRID - 1), 1] = False
    valid[cols.eq(0), 2] = False
    valid[cols.eq(GRID - 1), 3] = False
    row_z = torch.full((*candidates.shape, DIRECT_CLASS_COUNT), -4.0, device=device)
    for direction in range(DIRECT_CLASS_COUNT):
        row_z[:, direction, direction] = 4.0
    return candidates, valid, row_z, anchors


def smoke(device: torch.device) -> dict[str, float]:
    """Exercise canonicalisation, reciprocal weighting, capacity and metrics."""
    candidates, valid, row_z, perm = _perfect_sparse_inputs(device)
    relations = reciprocal_physical_relations(candidates, valid, row_z)
    high_confidence = _threshold_relations(relations, threshold=0.0)
    raw = physical_metrics(high_confidence, perm)
    selected = rscm_greedy(high_confidence)
    rscm = physical_metrics(selected, perm)
    if raw.selected != TRUE_PHYSICAL_EDGES or raw.precision < 0.999:
        raise AssertionError(f"raw reciprocal smoke failed: {raw}")
    if (
        rscm.selected != TRUE_PHYSICAL_EDGES
        or rscm.precision < 0.999
        or rscm.coverage < 0.999
        or rscm.correct_largest_nodes != NFRAG
    ):
        raise AssertionError(f"RSCM smoke failed: {rscm}")
    # Explicitly check both global constraints: one consumed slot excludes a
    # competing relation, and an unordered pair cannot be reused with a
    # different directional hypothesis.
    conflict = PhysicalRelation(0, 2, 3, 0, 0, 9.0, 9.0, 9.0)
    kept = rscm_greedy([high_confidence[0], conflict])
    slots = {(relation.anchor, relation.direction) for relation in kept}
    if len(kept) != 1 or len(slots) != 1:
        raise AssertionError("RSCM slot-capacity smoke failed")
    pair_conflict = PhysicalRelation(0, 1, 1, 0, 0, 8.0, 8.0, 8.0)
    if len(rscm_greedy([high_confidence[0], pair_conflict])) != 1:
        raise AssertionError("RSCM duplicate-pair smoke failed")
    return {
        "relations": float(len(relations)),
        "exact_precision": rscm.precision,
        "true_coverage": rscm.coverage,
        "largest_correct_component": float(rscm.correct_largest_nodes),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=1,
        help="fresh exact synthetic held-out images; this gate is deliberately restricted to 1",
    )
    parser.add_argument("--top-k", "--top_k", type=int, default=64, help="per-encoder affinity top-K")
    parser.add_argument("--affinity-ckpt", "--affinity_ckpt", default=DEFAULT_AFFINITY_CKPT)
    parser.add_argument(
        "--affinity-ckpt2", "--affinity_ckpt2", default=DEFAULT_AFFINITY_CKPT2,
        help="secondary affinity checkpoint; empty disables the union",
    )
    parser.add_argument(
        "--pair-ckpts", "--pair_ckpts", default="",
        help="comma-separated PairwiseNet checkpoint paths; default is pair0/pair1",
    )
    parser.add_argument("--pair-tag", "--pair_tag", default="pair")
    parser.add_argument("--pair-which", "--pair_which", default="best")
    parser.add_argument(
        "--thresholds", nargs="+", default=["2.0,2.5,3.0"],
        help="raw reciprocal min(row-z) thresholds to compare with RSCM",
    )
    parser.add_argument(
        "--tops", "--topks", nargs="+", default=["1,2,4,8"],
        help="per-source top oriented seams before reciprocal checking",
    )
    parser.add_argument(
        "--pair-batch", "--pair_batch", type=int, default=4096,
        help="maximum raw PairwiseNet seam layouts per inference chunk",
    )
    parser.add_argument("--seed", type=int, default=SEED + 4177, help="fresh synthetic corruption seed")
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--smoke", action="store_true", help="run data-free RSCM checks and exit")
    args = parser.parse_args()
    try:
        args.thresholds = _float_list(args.thresholds, name="threshold")
        args.tops = _int_list(args.tops, name="top")
    except ValueError as exc:
        parser.error(str(exc))
    if args.n != 1:
        parser.error("--n is intentionally fixed at 1 for this diagnostic gate")
    if not 1 <= args.top_k < NFRAG:
        parser.error(f"--top-k must lie in [1,{NFRAG - 1}]")
    if args.pair_batch < DIRECT_CLASS_COUNT:
        parser.error(f"--pair-batch must be at least {DIRECT_CLASS_COUNT}")
    maximum_orientations = DIRECT_CLASS_COUNT * args.top_k * (2 if args.affinity_ckpt2 else 1)
    if args.tops[-1] > maximum_orientations:
        parser.error(
            f"--tops may not exceed {maximum_orientations} oriented candidates per source "
            "for the configured affinity graph"
        )
    return args


def main() -> None:
    args = _parse_args()
    device = _parse_device(args.device)
    if args.smoke:
        print(f"[RSCM smoke] device={device} {smoke(device)}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    affinity, affinity_metadata, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity_secondary: nn.Module | None = None
    affinity_secondary_metadata: Mapping[str, object] | None = None
    if args.affinity_ckpt2:
        affinity_secondary, affinity_secondary_metadata, _ = load_frozen_affinity(
            args.affinity_ckpt2, device
        )
    pair_paths = _pair_checkpoint_paths(args.pair_ckpts, args.pair_tag, args.pair_which)
    pair_models, pair_metadata = load_pair_ensemble(pair_paths, device)

    _, val_names = train_val_split()
    dataset = CanvasDataset(val_names[:1], real_prob=0.0, seed=args.seed)
    sample = dataset[0]
    if not bool(sample["has_perm"]):
        raise RuntimeError("RSCM requires CanvasDataset(real_prob=0.0) exact synthetic labels")
    tiles = sample["tiles"].to(device, non_blocking=device.type == "cuda")
    perm = sample["perm"].to(device, non_blocking=device.type == "cuda").long()

    print(
        f"device={device} exact_fresh_heldout_images=1 affinity_top_k={args.top_k} "
        f"encoders={1 + int(affinity_secondary is not None)} pair_ensemble={len(pair_models)} "
        f"pair_batch(raw-layouts)={args.pair_batch}",
        flush=True,
    )
    print(
        f"affinity_1={os.path.abspath(args.affinity_ckpt)} step={affinity_metadata.get('step')}",
        flush=True,
    )
    if affinity_secondary is not None and affinity_secondary_metadata is not None:
        print(
            f"affinity_2={os.path.abspath(args.affinity_ckpt2)} "
            f"step={affinity_secondary_metadata.get('step')}",
            flush=True,
        )
    for path, metadata in zip(pair_paths, pair_metadata):
        print(
            f"pair={os.path.abspath(path)} step={metadata.get('step')} val={metadata.get('val')}",
            flush=True,
        )
    print(
        "PairwiseNet is evaluated only on frozen affinity candidates; reciprocal lookup uses sparse dictionaries, not dense seam scoring.",
        flush=True,
    )

    candidates_batched, valid_batched = mine_affinity_candidates(
        affinity,
        tiles.unsqueeze(0),
        candidate_k=args.top_k,
        device=device,
        affinity_secondary=affinity_secondary,
    )
    candidates, valid = candidates_batched[0], valid_batched[0]
    orientations = score_pairwise_directions(
        pair_models, tiles, candidates, valid, pair_batch=args.pair_batch, device=device
    )
    row_z = _row_z(orientations, valid)
    relations = reciprocal_physical_relations(candidates, valid, row_z)
    reciprocal_true = sum(
        _expected_direction(perm.detach().cpu(), relation.anchor, relation.target) == relation.direction
        for relation in relations
    )
    candidate_pairs = int(valid.sum())
    raw_layouts = candidate_pairs * DIRECT_CLASS_COUNT
    dense_layouts = NFRAG * (NFRAG - 1) * DIRECT_CLASS_COUNT
    print(
        f"candidate pairs={candidate_pairs} ({candidate_pairs / NFRAG:.2f}/tile); "
        f"PairwiseNet layouts={raw_layouts} ({100.0 * raw_layouts / dense_layouts:.2f}% of dense UDLR universe)",
        flush=True,
    )
    print(
        f"mutual canonical physical hypotheses={len(relations)}; "
        f"true physical candidate coverage={reciprocal_true / TRUE_PHYSICAL_EDGES:.4f}",
        flush=True,
    )

    print("\n=== reciprocal min(row-z) threshold mode: raw vs RSCM ===", flush=True)
    for threshold in args.thresholds:
        raw = _threshold_relations(relations, threshold)
        _print_metrics(f"raw reciprocal threshold={threshold:.3f}", physical_metrics(raw, perm))
        _print_metrics(
            f"RSCM threshold={threshold:.3f}", physical_metrics(rscm_greedy(raw), perm)
        )

    print("\n=== reciprocal row-top oriented mode: raw vs RSCM ===", flush=True)
    for top in args.tops:
        raw = _top_reciprocal_relations(relations, row_z, valid, top)
        _print_metrics(f"raw reciprocal top={top}", physical_metrics(raw, perm))
        _print_metrics(f"RSCM top={top}", physical_metrics(rscm_greedy(raw), perm))


if __name__ == "__main__":
    main()
