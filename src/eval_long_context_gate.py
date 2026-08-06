"""Operational gate for long line contexts from high-confidence seam seeds.

``rank_v2w64`` is not dense enough to solve a whole puzzle by itself, but its
``mutual argmax -> RSCM`` graph at confidence 0.70 contains very accurate
physical seams.  Before spending time on a component-level model, this script
asks a narrower, falsifiable question:

* do those *model-selected* seams contain true oriented 3-tile and 4-tile
  lines (two and three consecutive seed edges)?
* once such a line exists, is its next physical tile still present in the
  frozen dual-affinity candidate list?
* does one extra known tile provide even a cheap raw-profile continuation
  signal, compared with only the last two known tiles?

The procedure is exact synthetic held-out.  ``perm`` is deliberately absent
from candidate mining and seed selection.  It is revealed only afterwards to
measure seed correctness, component geometry, oracle candidate coverage, and
the raw-context ranking diagnostic.  The latter is therefore an *oracle
correct-context information probe*, not an assembly result and not a learned
model.

Examples
--------

    python src/eval_long_context_gate.py --smoke
    python src/eval_long_context_gate.py --device cuda --n 4

The defaults reproduce the documented v2 frozen graph: ``rank_v2w64_best``
and the two affinity encoders recorded in its checkpoint metadata.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from direct_pose import DIRECT_OFFSETS
from eval_candidate_rank import load_ranker, mutual_argmax_relations, score_full_graph
from eval_rscm_gate import INVERSE_DIRECTION, PhysicalRelation, rscm_greedy
from imgio import train_val_split
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


# Number of counter-clockwise 90-degree rotations that makes the indicated
# cardinal continuation point left -> right.  This lets the raw profile
# heuristic use one canonical layout without favouring horizontal seams.
_TURNS_TO_RIGHT = (3, 1, 2, 0)  # U, D, L, R


@dataclass(frozen=True)
class OrientedPath:
    """A label-confirmed straight seed path in one cardinal direction."""

    nodes: tuple[int, ...]
    direction: int

    @property
    def endpoint(self) -> int:
        return self.nodes[-1]


@dataclass
class ContextRankSums:
    """Additive raw-continuation ranks on the same path/candidate queries."""

    eligible: int = 0
    covered: int = 0
    candidate_count: int = 0
    two_r1: float = 0.0
    two_r5: float = 0.0
    two_mrr: float = 0.0
    three_r1: float = 0.0
    three_r5: float = 0.0
    three_mrr: float = 0.0

    def add_ranks(self, two_ranks: Tensor, three_ranks: Tensor) -> None:
        if two_ranks.numel() != three_ranks.numel():
            raise ValueError("two/three-context rank counts differ")
        count = int(two_ranks.numel())
        self.covered += count
        self.two_r1 += float(two_ranks.le(1).sum())
        self.two_r5 += float(two_ranks.le(5).sum())
        self.two_mrr += float(two_ranks.float().reciprocal().sum())
        self.three_r1 += float(three_ranks.le(1).sum())
        self.three_r5 += float(three_ranks.le(5).sum())
        self.three_mrr += float(three_ranks.float().reciprocal().sum())

    def merge(self, other: "ContextRankSums") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def metric(self, name: str) -> float:
        if not self.covered:
            return float("nan")
        return float(getattr(self, name) / self.covered)


@dataclass(frozen=True)
class ImageReport:
    """Numeric held-out result for one puzzle; labels entered only here."""

    selected: int
    correct: int
    component_shapes: Mapping[str, int]
    correct_component_nodes: int
    largest_correct_component: int
    paths_len2: int
    paths_len3: int
    len2_next_eligible: int
    len2_next_covered: int
    len3_next_eligible: int
    len3_next_covered: int
    ranks: ContextRankSums

    def scalar_metrics(self) -> dict[str, float]:
        def ratio(numerator: int | float, denominator: int | float) -> float:
            return float(numerator / denominator) if denominator else float("nan")

        return {
            "selected seeds": float(self.selected),
            "correct seeds": float(self.correct),
            "seed precision": ratio(self.correct, self.selected),
            "correct components": float(self.component_shapes["components"]),
            "correct edge components": float(self.component_shapes["edge"]),
            "correct 3-tile lines": float(self.component_shapes["line3"]),
            "correct 4+ tile lines": float(self.component_shapes["line4plus"]),
            "correct non-line components": float(self.component_shapes["other"]),
            "correct component nodes": float(self.correct_component_nodes),
            "largest correct component": float(self.largest_correct_component),
            "oriented seed paths, 2 edges": float(self.paths_len2),
            "oriented seed paths, 3 edges": float(self.paths_len3),
            "next-list coverage after 3 tiles": ratio(
                self.len2_next_covered, self.len2_next_eligible
            ),
            "next-list coverage after 4 tiles": ratio(
                self.len3_next_covered, self.len3_next_eligible
            ),
            "raw query candidate coverage": ratio(self.ranks.covered, self.ranks.eligible),
            "raw 2-tile R@1": self.ranks.metric("two_r1"),
            "raw 3-tile R@1": self.ranks.metric("three_r1"),
            "raw 3-minus-2 R@1": self.ranks.metric("three_r1") - self.ranks.metric("two_r1"),
            "raw 2-tile MRR": self.ranks.metric("two_mrr"),
            "raw 3-tile MRR": self.ranks.metric("three_mrr"),
        }


def _default_ranker_path(workspace: str) -> str:
    """The long-context gate is intentionally pinned to the v2 capacity run."""
    return os.path.join(workspace, "artifacts", "candidate_rank", "rank_v2w64_best.pt")


def _resolve_affinity_paths(
    payload: Mapping[str, object], args: argparse.Namespace
) -> tuple[str, str, int]:
    """Recover the exact dual frozen candidate graph from the ranker payload."""
    raw_graph = payload.get("candidate_graph", {})
    graph = raw_graph if isinstance(raw_graph, Mapping) else {}
    raw_encoders = graph.get("encoders", ())
    encoders = list(raw_encoders) if isinstance(raw_encoders, (list, tuple)) else []
    raw_args = payload.get("args", {})
    saved_args = raw_args if isinstance(raw_args, Mapping) else {}

    recorded_first = ""
    recorded_second = ""
    if encoders and isinstance(encoders[0], Mapping):
        recorded_first = str(encoders[0].get("path", ""))
    if len(encoders) > 1 and isinstance(encoders[1], Mapping):
        recorded_second = str(encoders[1].get("path", ""))
    primary = str(args.affinity_ckpt or recorded_first or saved_args.get("affinity_ckpt", ""))
    secondary = str(
        args.affinity_ckpt2 or recorded_second or saved_args.get("affinity_ckpt2", "")
    )
    if not primary or not secondary:
        raise RuntimeError(
            "ranker checkpoint did not record a dual affinity union; pass both "
            "--affinity-ckpt and --affinity-ckpt2 explicitly"
        )
    try:
        saved_top_k = int(saved_args.get("candidate_k", 64))
    except (TypeError, ValueError):
        saved_top_k = 64
    top_k = int(args.top_k or saved_top_k)
    if not 1 <= top_k < NFRAG:
        raise ValueError(f"resolved --top-k must lie in [1,{NFRAG - 1}], got {top_k}")
    return primary, secondary, top_k


def _inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """Return clean-cell -> shuffled-input-tile mapping with strict checks."""
    if perm.shape != (NFRAG,):
        raise ValueError(f"perm must have shape ({NFRAG},), got {perm.shape}")
    if np.any(perm < 0) or np.any(perm >= NFRAG) or np.unique(perm).size != NFRAG:
        raise ValueError("synthetic permutation is not a bijection")
    inverse = np.empty(NFRAG, dtype=np.int64)
    inverse[perm] = np.arange(NFRAG, dtype=np.int64)
    return inverse


def _true_direction(perm: np.ndarray, source: int, target: int) -> int | None:
    """Return target's exact direction from source, or ``None`` if non-adjacent."""
    source_row, source_col = divmod(int(perm[source]), GRID)
    target_row, target_col = divmod(int(perm[target]), GRID)
    delta = target_row - source_row, target_col - source_col
    try:
        return DIRECT_OFFSETS.index(delta)
    except ValueError:
        return None


def _next_true_tile(
    perm: np.ndarray, inverse: np.ndarray, source: int, direction: int
) -> int | None:
    """Get the next true input tile in a direction; return ``None`` at a border."""
    row, col = divmod(int(perm[source]), GRID)
    delta_row, delta_col = DIRECT_OFFSETS[direction]
    row += delta_row
    col += delta_col
    if not (0 <= row < GRID and 0 <= col < GRID):
        return None
    return int(inverse[row * GRID + col])


def _candidate_sets(candidates: Tensor, valid: Tensor) -> list[set[int]]:
    """CPU membership sets for oracle reporting; candidate graph stays frozen."""
    if candidates.ndim != 2 or candidates.shape[0] != NFRAG:
        raise ValueError(f"candidates must have shape ({NFRAG},K), got {tuple(candidates.shape)}")
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a bool mask aligned with candidates")
    candidate_cpu = candidates.detach().to(device="cpu", dtype=torch.long).numpy()
    valid_cpu = valid.detach().to(device="cpu", dtype=torch.bool).numpy()
    return [
        {int(tile) for tile, keep in zip(row, mask) if bool(keep)}
        for row, mask in zip(candidate_cpu, valid_cpu)
    ]


def _correct_relations(
    selected: Sequence[PhysicalRelation], perm: np.ndarray
) -> list[PhysicalRelation]:
    """Reveal labels only after model-only seed selection has completed."""
    return [
        relation
        for relation in selected
        if _true_direction(perm, relation.anchor, relation.target) == relation.direction
    ]


def _correct_oriented_arcs(
    correct: Sequence[PhysicalRelation],
) -> dict[tuple[int, int], int]:
    """Materialize both directions of each true selected physical seam."""
    arcs: dict[tuple[int, int], int] = {}
    for relation in correct:
        views = (
            (relation.anchor, relation.direction, relation.target),
            (relation.target, INVERSE_DIRECTION[relation.direction], relation.anchor),
        )
        for source, direction, target in views:
            previous = arcs.get((source, direction))
            if previous is not None and previous != target:
                # RSCM guarantees a source/direction slot is unique.  Failing
                # here would signal an upstream invariant regression, not an
                # ambiguity that a downstream context model should hide.
                raise RuntimeError(
                    f"correct seed graph reused ({source}, direction={direction}) "
                    f"for both {previous} and {target}"
                )
            arcs[source, direction] = target
    return arcs


def _oriented_paths(
    arcs: Mapping[tuple[int, int], int], *, edge_length: int
) -> list[OrientedPath]:
    """Enumerate directed same-direction paths of exactly ``edge_length`` seams."""
    if edge_length < 1:
        raise ValueError("edge_length must be positive")
    paths: list[OrientedPath] = []
    for (source, direction), first_target in sorted(arcs.items()):
        nodes = [source, first_target]
        current = first_target
        for _ in range(1, edge_length):
            target = arcs.get((current, direction))
            if target is None or target in nodes:
                break
            nodes.append(target)
            current = target
        if len(nodes) == edge_length + 1:
            paths.append(OrientedPath(tuple(nodes), direction))
    return paths


def _component_shape_stats(
    correct: Sequence[PhysicalRelation], perm: np.ndarray
) -> tuple[dict[str, int], int, int]:
    """Classify connected correct components by actual clean-grid geometry."""
    adjacency: dict[int, set[int]] = defaultdict(set)
    for relation in correct:
        adjacency[relation.anchor].add(relation.target)
        adjacency[relation.target].add(relation.anchor)

    shapes = Counter({"components": 0, "edge": 0, "line3": 0, "line4plus": 0, "other": 0})
    seen: set[int] = set()
    total_nodes = 0
    largest = 0
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        nodes: set[int] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            nodes.add(node)
            stack.extend(adjacency[node] - seen)

        edge_count = sum(
            1
            for relation in correct
            if relation.anchor in nodes and relation.target in nodes
        )
        coordinates = [divmod(int(perm[node]), GRID) for node in nodes]
        rows = {row for row, _ in coordinates}
        cols = {col for _, col in coordinates}
        straight = False
        if edge_count == len(nodes) - 1 and len(rows) == 1:
            col_values = [col for _, col in coordinates]
            straight = max(col_values) - min(col_values) == len(nodes) - 1
        elif edge_count == len(nodes) - 1 and len(cols) == 1:
            row_values = [row for row, _ in coordinates]
            straight = max(row_values) - min(row_values) == len(nodes) - 1

        shapes["components"] += 1
        total_nodes += len(nodes)
        largest = max(largest, len(nodes))
        if len(nodes) == 2 and edge_count == 1:
            shapes["edge"] += 1
        elif straight and len(nodes) == 3:
            shapes["line3"] += 1
        elif straight and len(nodes) >= 4:
            shapes["line4plus"] += 1
        else:
            shapes["other"] += 1
    return dict(shapes), total_nodes, largest


def _next_coverage(
    paths: Sequence[OrientedPath],
    perm: np.ndarray,
    inverse: np.ndarray,
    candidate_sets: Sequence[set[int]],
) -> tuple[int, int]:
    """Oracle membership ceiling for the physical next tile after every path."""
    eligible = 0
    covered = 0
    for path in paths:
        target = _next_true_tile(perm, inverse, path.endpoint, path.direction)
        if target is None:
            continue
        eligible += 1
        covered += int(target in candidate_sets[path.endpoint])
    return eligible, covered


def _preprocess_tiles(tiles: Tensor, smooth_kernel: int) -> Tensor:
    """Per-tile exposure normalization plus a tiny noise-suppression filter."""
    if tuple(tiles.shape) != (NFRAG, 3, FS, FS):
        raise ValueError(f"tiles must have shape ({NFRAG},3,{FS},{FS}), got {tuple(tiles.shape)}")
    normalized = tiles.float()
    mean = normalized.mean(dim=(-2, -1), keepdim=True)
    rms = (normalized - mean).square().mean(dim=(-2, -1), keepdim=True).add(1.0e-6).sqrt()
    normalized = (normalized - mean) / rms
    if smooth_kernel > 1:
        pad = smooth_kernel // 2
        normalized = F.avg_pool2d(
            F.pad(normalized, (pad, pad, pad, pad), mode="reflect"),
            smooth_kernel,
            stride=1,
        )
    return normalized


def _profile_cost(delta: Tensor, gradient_weight: float) -> Tensor:
    """Cheap color/profile mismatch cost; preserves any leading batch axes."""
    cost = delta.square().mean(dim=(-2, -1))
    if gradient_weight:
        gradient = delta[..., 1:] - delta[..., :-1]
        cost = cost + gradient_weight * gradient.square().mean(dim=(-2, -1))
    return cost


def _raw_context_scores(
    first: Tensor,
    second: Tensor,
    third: Tensor,
    candidates: Tensor,
    *,
    edge_band: int,
    context_weight: float,
    gradient_weight: float,
) -> tuple[Tensor, Tensor]:
    """Score next candidates with the last 2 vs all 3 known line tiles.

    Inputs are already rotated to canonical left-to-right orientation.

    * 2-tile context projects the next border jump as a continuation of the
      last observed jump ``B -> C``.
    * 3-tile context uses ``A -> B -> C`` to linearly extrapolate that jump,
      which introduces a third finite-difference term without any training.

    Both scores include the same raw C/D seam mismatch; their delta is the
    only intended comparison.
    """
    if first.shape != second.shape or second.shape != third.shape or first.ndim != 4:
        raise ValueError("known line tiles must share shape (rows,3,20,20)")
    if candidates.ndim != 5 or candidates.shape[0] != first.shape[0]:
        raise ValueError("candidate tiles must have shape (rows,K,3,20,20)")
    if candidates.shape[2:] != first.shape[1:]:
        raise ValueError("candidate tile shape differs from known line tiles")

    a_right = first[..., -edge_band:].mean(dim=-1)
    b_left = second[..., :edge_band].mean(dim=-1)
    b_right = second[..., -edge_band:].mean(dim=-1)
    c_left = third[..., :edge_band].mean(dim=-1)
    c_right = third[..., -edge_band:].mean(dim=-1)
    d_left = candidates[..., :edge_band].mean(dim=-1)

    jump_ab = b_left - a_right
    jump_bc = c_left - b_right
    jump_cd = d_left - c_right.unsqueeze(1)
    seam_cost = _profile_cost(jump_cd, gradient_weight)
    two_cost = seam_cost + context_weight * _profile_cost(
        jump_cd - jump_bc.unsqueeze(1), gradient_weight
    )
    extrapolated = 2.0 * jump_bc - jump_ab
    three_cost = seam_cost + context_weight * _profile_cost(
        jump_cd - extrapolated.unsqueeze(1), gradient_weight
    )
    return -two_cost, -three_cost


def _ranks(scores: Tensor, target_mask: Tensor) -> Tensor:
    """Return one-based strict ranks for one unique finite target per row."""
    if scores.shape != target_mask.shape:
        raise ValueError("score and target-mask shapes differ")
    if not bool(torch.all(target_mask.sum(dim=-1).eq(1))):
        raise RuntimeError("each covered raw-context row needs exactly one target")
    target = scores[target_mask].reshape(scores.shape[0])
    if not bool(torch.isfinite(target).all()):
        raise RuntimeError("raw-context target has a non-finite score")
    return scores.gt(target.unsqueeze(1)).sum(dim=-1).add(1)


@torch.inference_mode()
def _rank_long_contexts(
    paths: Sequence[OrientedPath],
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    *,
    perm: np.ndarray,
    inverse: np.ndarray,
    candidate_sets: Sequence[set[int]],
    edge_band: int,
    smooth_kernel: int,
    context_weight: float,
    gradient_weight: float,
    row_batch: int,
) -> ContextRankSums:
    """Oracle-rank continuations of true three-tile selected seed paths."""
    report = ContextRankSums()
    by_direction: dict[int, list[tuple[OrientedPath, int]]] = defaultdict(list)
    for path in paths:
        target = _next_true_tile(perm, inverse, path.endpoint, path.direction)
        if target is None:
            continue
        report.eligible += 1
        if target in candidate_sets[path.endpoint]:
            by_direction[path.direction].append((path, target))

    if not by_direction:
        return report

    normalized = _preprocess_tiles(tiles, smooth_kernel)
    for direction, records in sorted(by_direction.items()):
        turns = _TURNS_TO_RIGHT[direction]
        for start in range(0, len(records), row_batch):
            chunk = records[start : start + row_batch]
            first_ids = torch.tensor([record[0].nodes[0] for record in chunk], device=tiles.device)
            second_ids = torch.tensor([record[0].nodes[1] for record in chunk], device=tiles.device)
            third_ids = torch.tensor([record[0].nodes[2] for record in chunk], device=tiles.device)
            target_ids = torch.tensor([record[1] for record in chunk], device=tiles.device)
            endpoint_ids = third_ids

            candidate_ids = candidates[endpoint_ids]
            valid_rows = valid[endpoint_ids]
            target_mask = valid_rows & candidate_ids.eq(target_ids.unsqueeze(1))
            if not bool(torch.all(target_mask.sum(dim=1).eq(1))):
                raise RuntimeError("candidate-set membership did not imply exactly one valid target")

            first = normalized[first_ids]
            second = normalized[second_ids]
            third = normalized[third_ids]
            candidate_tiles = normalized[candidate_ids]
            if turns:
                first = torch.rot90(first, turns, dims=(-2, -1))
                second = torch.rot90(second, turns, dims=(-2, -1))
                third = torch.rot90(third, turns, dims=(-2, -1))
                candidate_tiles = torch.rot90(candidate_tiles, turns, dims=(-2, -1))
            two_score, three_score = _raw_context_scores(
                first,
                second,
                third,
                candidate_tiles,
                edge_band=edge_band,
                context_weight=context_weight,
                gradient_weight=gradient_weight,
            )
            two_score = two_score.masked_fill(~valid_rows, -torch.inf)
            three_score = three_score.masked_fill(~valid_rows, -torch.inf)
            report.candidate_count += int(valid_rows.sum())
            report.add_ranks(_ranks(two_score, target_mask), _ranks(three_score, target_mask))
    return report


def _format_ratio(numerator: int | float, denominator: int | float) -> str:
    return f"{numerator}/{denominator}=n/a" if not denominator else f"{numerator}/{denominator}={numerator / denominator:.4f}"


def _format_raw(ranks: ContextRankSums) -> str:
    if not ranks.covered:
        return f"candidate coverage={_format_ratio(ranks.covered, ranks.eligible)}; no rankable paths"
    return (
        f"candidate coverage={_format_ratio(ranks.covered, ranks.eligible)}; "
        f"mean candidates={ranks.candidate_count / ranks.covered:.1f}; "
        f"2-tile R@1/R@5/MRR={ranks.metric('two_r1'):.4f}/"
        f"{ranks.metric('two_r5'):.4f}/{ranks.metric('two_mrr'):.4f}; "
        f"3-tile R@1/R@5/MRR={ranks.metric('three_r1'):.4f}/"
        f"{ranks.metric('three_r5'):.4f}/{ranks.metric('three_mrr'):.4f}; "
        f"delta R@1={ranks.metric('three_r1') - ranks.metric('two_r1'):+.4f}"
    )


def _print_image_report(index: int, total: int, report: ImageReport) -> None:
    shapes = report.component_shapes
    precision = report.correct / report.selected if report.selected else float("nan")
    print(f"\n=== image {index}/{total}: model-only seeds, then exact labels ===", flush=True)
    print(
        f"selected={report.selected}; correct={report.correct}; seed precision={precision:.4f}",
        flush=True,
    )
    print(
        "correct-component shapes: "
        f"components={shapes['components']} nodes={report.correct_component_nodes}/{NFRAG} "
        f"largest={report.largest_correct_component}; "
        f"edge={shapes['edge']} line3={shapes['line3']} line4+={shapes['line4plus']} "
        f"other={shapes['other']}",
        flush=True,
    )
    print(
        f"true oriented seed paths: 2 edges / 3 tiles={report.paths_len2}; "
        f"3 edges / 4 tiles={report.paths_len3}",
        flush=True,
    )
    print(
        "next true tile in frozen affinity list: "
        f"after 3-tile path {_format_ratio(report.len2_next_covered, report.len2_next_eligible)}; "
        f"after 4-tile path {_format_ratio(report.len3_next_covered, report.len3_next_eligible)}",
        flush=True,
    )
    print("raw oracle continuation on the same 3-tile paths: " + _format_raw(report.ranks), flush=True)


def _print_mean_variance(reports: Sequence[ImageReport]) -> None:
    """Report macro means and population variance across the held-out images."""
    if not reports:
        return
    metric_rows: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        for name, value in report.scalar_metrics().items():
            if math.isfinite(value):
                metric_rows[name].append(value)
    print(f"\n=== per-image macro mean and variance (n={len(reports)}) ===", flush=True)
    print("metric                                      n_eff       mean      variance", flush=True)
    for name, values in metric_rows.items():
        array = np.asarray(values, dtype=np.float64)
        print(
            f"{name:<42} {len(values):>5}  {array.mean():>9.4f}  {array.var(ddof=0):>12.6f}",
            flush=True,
        )


def _print_pooled(reports: Sequence[ImageReport]) -> None:
    """Show count-weighted totals alongside the requested per-image variance."""
    if not reports:
        return
    pooled_ranks = ContextRankSums()
    for report in reports:
        pooled_ranks.merge(report.ranks)
    selected = sum(report.selected for report in reports)
    correct = sum(report.correct for report in reports)
    len2 = sum(report.paths_len2 for report in reports)
    len3 = sum(report.paths_len3 for report in reports)
    len2_eligible = sum(report.len2_next_eligible for report in reports)
    len2_covered = sum(report.len2_next_covered for report in reports)
    len3_eligible = sum(report.len3_next_eligible for report in reports)
    len3_covered = sum(report.len3_next_covered for report in reports)
    print(f"\n=== pooled exact held-out counts (n={len(reports)}) ===", flush=True)
    print(
        f"selected={selected} correct={correct} seed precision="
        f"{correct / selected if selected else float('nan'):.4f}",
        flush=True,
    )
    print(
        f"true oriented paths: length2={len2}; length3={len3}; "
        f"next-list after length2 {_format_ratio(len2_covered, len2_eligible)}; "
        f"after length3 {_format_ratio(len3_covered, len3_eligible)}",
        flush=True,
    )
    print("pooled raw oracle continuation: " + _format_raw(pooled_ranks), flush=True)


def smoke() -> dict[str, int | float]:
    """Data-free topology and raw-score contracts; no checkpoint or labels leak."""
    # Identity perm yields the true horizontal chain 0 -> 1 -> 2 -> 3.
    relation = lambda anchor, target: PhysicalRelation(
        anchor=anchor,
        target=target,
        direction=3,
        anchor_rank=0,
        target_rank=0,
        anchor_z=0.9,
        target_z=0.9,
        weight=0.9,
    )
    correct = [relation(0, 1), relation(1, 2), relation(2, 3)]
    arcs = _correct_oriented_arcs(correct)
    paths2 = _oriented_paths(arcs, edge_length=2)
    paths3 = _oriented_paths(arcs, edge_length=3)
    shapes, nodes, largest = _component_shape_stats(correct, np.arange(NFRAG, dtype=np.int64))
    if len(paths2) != 4 or len(paths3) != 2 or shapes["line4plus"] != 1:
        raise AssertionError(
            f"line topology smoke failed: paths2={len(paths2)} paths3={len(paths3)} shapes={shapes}"
        )
    generator = torch.Generator().manual_seed(123)
    known = torch.rand((2, 3, FS, FS), generator=generator)
    candidate = torch.rand((2, 5, 3, FS, FS), generator=generator)
    two, three = _raw_context_scores(
        known,
        known,
        known,
        candidate,
        edge_band=3,
        context_weight=0.5,
        gradient_weight=0.5,
    )
    if two.shape != (2, 5) or three.shape != (2, 5) or not bool(torch.isfinite(two).all()):
        raise AssertionError("raw long-context smoke produced invalid scores")
    return {
        "paths_len2": len(paths2),
        "paths_len3": len(paths3),
        "line4plus_components": shapes["line4plus"],
        "component_nodes": nodes,
        "largest": largest,
        "raw_delta_mean": float((three - two).mean()),
    }


def _parse_args() -> argparse.Namespace:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranker-ckpt",
        "--ranker_ckpt",
        dest="ranker_ckpt",
        default=_default_ranker_path(workspace),
        help="rank_v2w64 CandidateSeamRanker checkpoint",
    )
    parser.add_argument(
        "--affinity-ckpt",
        "--affinity_ckpt",
        dest="affinity_ckpt",
        default="",
        help="primary affinity checkpoint; default uses ranker metadata",
    )
    parser.add_argument(
        "--affinity-ckpt2",
        "--affinity_ckpt2",
        dest="affinity_ckpt2",
        default="",
        help="secondary affinity checkpoint; default uses ranker metadata",
    )
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=0)
    parser.add_argument("--n", type=int, default=4, help="fresh exact synthetic held-out puzzles")
    parser.add_argument(
        "--confidence",
        "--conf",
        type=float,
        default=0.70,
        help="mutual reciprocal confidence floor before RSCM",
    )
    parser.add_argument("--pair-batch", "--pair_batch", type=int, default=4096)
    parser.add_argument("--edge-band", "--edge_band", type=int, default=3)
    parser.add_argument("--smooth-kernel", "--smooth_kernel", type=int, default=3)
    parser.add_argument("--context-weight", "--context_weight", type=float, default=0.5)
    parser.add_argument("--gradient-weight", "--gradient_weight", type=float, default=0.5)
    parser.add_argument("--row-batch", "--row_batch", type=int, default=64)
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED + 7331,
        help="fresh synthetic corruption/shuffle seed used by the candidate-rank gate",
    )
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--smoke", action="store_true", help="run data-free contracts and exit")
    args = parser.parse_args()
    if args.n < 1 or args.pair_batch < 1 or args.row_batch < 1:
        parser.error("--n, --pair-batch, and --row-batch must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must lie in [0,1]")
    if args.top_k < 0 or args.top_k >= NFRAG:
        parser.error(f"--top-k must lie in [0,{NFRAG - 1}]")
    if not 1 <= args.edge_band <= FS:
        parser.error(f"--edge-band must lie in [1,{FS}]")
    if args.smooth_kernel < 1 or args.smooth_kernel > FS or not args.smooth_kernel % 2:
        parser.error(f"--smooth-kernel must be odd and lie in [1,{FS}]")
    if args.context_weight < 0.0 or args.gradient_weight < 0.0:
        parser.error("--context-weight and --gradient-weight must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    if args.smoke:
        print(f"[long-context gate smoke] {smoke()}", flush=True)
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    model, payload = load_ranker(args.ranker_ckpt, device)
    affinity_path, affinity_path2, top_k = _resolve_affinity_paths(payload, args)
    affinity, _, _ = load_frozen_affinity(affinity_path, device)
    affinity_secondary: nn.Module
    affinity_secondary, _, _ = load_frozen_affinity(affinity_path2, device)

    _, validation_names = train_val_split()
    if args.n > len(validation_names):
        raise ValueError(f"--n={args.n} exceeds held-out split size {len(validation_names)}")
    dataset = CanvasDataset(validation_names[: args.n], real_prob=0.0, seed=args.seed)

    print(
        f"device={device}; ranker={os.path.abspath(args.ranker_ckpt)} step={payload.get('step')}; "
        f"params={sum(parameter.numel() for parameter in model.parameters()):,}",
        flush=True,
    )
    print(
        f"model-only seed rule: mutual argmax -> conf>={args.confidence:.2f} -> RSCM; "
        f"frozen dual affinity union=top{top_k}+top{top_k}",
        flush=True,
    )
    print(
        "Labels are not read until seed selection is complete.  Raw 2-vs-3 context "
        "ranks are conditional on label-confirmed three-tile paths and are only an "
        "information probe, not a deployable score.",
        flush=True,
    )

    reports: list[ImageReport] = []
    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("long-context gate requires exact synthetic CanvasDataset samples")
        tiles = sample["tiles"].to(device, non_blocking=device.type == "cuda")

        # Everything through this line is model-only: image pixels, frozen
        # encoders, ranker, and RSCM.  ``perm`` has not influenced selection.
        candidates_batched, valid_batched = mine_affinity_candidates(
            affinity,
            tiles.unsqueeze(0),
            candidate_k=top_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        candidates, valid = candidates_batched[0], valid_batched[0]
        scores = score_full_graph(
            model, tiles, candidates, valid, pair_batch=args.pair_batch, device=device
        )
        reciprocal = mutual_argmax_relations(candidates, scores)
        selected = rscm_greedy(
            [relation for relation in reciprocal if relation.weight >= args.confidence]
        )

        # Oracle reporting begins only now.
        perm = sample["perm"].detach().to(device="cpu", dtype=torch.long).numpy()
        inverse = _inverse_permutation(perm)
        candidate_sets = _candidate_sets(candidates, valid)
        correct = _correct_relations(selected, perm)
        arcs = _correct_oriented_arcs(correct)
        paths_len2 = _oriented_paths(arcs, edge_length=2)
        paths_len3 = _oriented_paths(arcs, edge_length=3)
        shapes, component_nodes, largest_component = _component_shape_stats(correct, perm)
        len2_eligible, len2_covered = _next_coverage(
            paths_len2, perm, inverse, candidate_sets
        )
        len3_eligible, len3_covered = _next_coverage(
            paths_len3, perm, inverse, candidate_sets
        )
        ranks = _rank_long_contexts(
            paths_len2,
            tiles,
            candidates,
            valid,
            perm=perm,
            inverse=inverse,
            candidate_sets=candidate_sets,
            edge_band=args.edge_band,
            smooth_kernel=args.smooth_kernel,
            context_weight=args.context_weight,
            gradient_weight=args.gradient_weight,
            row_batch=args.row_batch,
        )
        report = ImageReport(
            selected=len(selected),
            correct=len(correct),
            component_shapes=shapes,
            correct_component_nodes=component_nodes,
            largest_correct_component=largest_component,
            paths_len2=len(paths_len2),
            paths_len3=len(paths_len3),
            len2_next_eligible=len2_eligible,
            len2_next_covered=len2_covered,
            len3_next_eligible=len3_eligible,
            len3_next_covered=len3_covered,
            ranks=ranks,
        )
        reports.append(report)
        _print_image_report(index + 1, args.n, report)

    _print_pooled(reports)
    _print_mean_variance(reports)


if __name__ == "__main__":
    main()
