"""Label-free geometric islands from reciprocal directed-neighbor predictions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from candidate_rank import DOWN, LEFT, NUM_DIRECTIONS, RIGHT, UP

_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


@dataclass(frozen=True)
class DirectedGraph:
    predicted: Tensor
    margins: Tensor
    mutual: Tensor
    loop: Tensor


@dataclass(frozen=True)
class SelectedEdges:
    anchors: np.ndarray
    directions: np.ndarray
    targets: np.ndarray
    margins: np.ndarray
    loop: np.ndarray
    threshold: float


def graph_from_scores(candidates: Tensor, valid: Tensor, scores: Tensor) -> DirectedGraph:
    """Convert all 576x4 candidate rows into reciprocal/2x2 graph flags."""
    if candidates.shape[0] != 1 or valid.shape != candidates.shape:
        raise ValueError("expected one candidate graph")
    expected_rows = candidates.shape[1] * NUM_DIRECTIONS
    if scores.shape[0] != expected_rows:
        raise ValueError(f"expected {expected_rows} complete rows")
    masked = scores.float().masked_fill(~valid[0, :, None, :].expand(-1, 4, -1).reshape_as(scores), -torch.inf)
    values, slots = torch.topk(masked, k=2, dim=-1)
    anchors = torch.arange(candidates.shape[1], device=scores.device).repeat_interleave(4)
    predicted = candidates[0, anchors, slots[:, 0]].reshape(-1, 4)
    margins = (values[:, 0] - values[:, 1]).reshape(-1, 4)

    inverse = torch.tensor((DOWN, UP, RIGHT, LEFT), device=scores.device)
    anchor_grid = torch.arange(candidates.shape[1], device=scores.device)[:, None].expand(-1, 4)
    reverse = predicted[predicted, inverse[None, :]]
    mutual = reverse.eq(anchor_grid)

    loop = torch.zeros_like(mutual)
    for anchor in range(candidates.shape[1]):
        right = int(predicted[anchor, RIGHT])
        down = int(predicted[anchor, DOWN])
        corner_from_right = int(predicted[right, DOWN])
        corner_from_down = int(predicted[down, RIGHT])
        if corner_from_right != corner_from_down:
            continue
        if bool(
            mutual[anchor, RIGHT]
            and mutual[anchor, DOWN]
            and mutual[right, DOWN]
            and mutual[down, RIGHT]
        ):
            loop[anchor, RIGHT] = True
            loop[anchor, DOWN] = True
            loop[right, DOWN] = True
            loop[down, RIGHT] = True
    return DirectedGraph(predicted, margins, mutual, loop)


def select_edges(
    graph: DirectedGraph,
    *,
    confidence_quantile: float,
    max_directed_edges: int,
) -> SelectedEdges:
    """Keep loop edges first, then highest-margin reciprocal directed edges."""
    if not 0.0 <= confidence_quantile <= 1.0:
        raise ValueError("confidence_quantile must lie in [0,1]")
    mutual_margins = graph.margins[graph.mutual]
    threshold = (
        float(torch.quantile(mutual_margins, confidence_quantile))
        if mutual_margins.numel()
        else float("inf")
    )
    accepted = graph.loop | (graph.mutual & graph.margins.ge(threshold))
    flat = torch.nonzero(accepted.flatten(), as_tuple=False).flatten()
    loop_flat = graph.loop.flatten()
    margin_flat = graph.margins.flatten()
    if flat.numel() > max_directed_edges:
        loop_indices = flat[loop_flat[flat]]
        other = flat[~loop_flat[flat]]
        room = max(0, max_directed_edges - int(loop_indices.numel()))
        if room:
            other = other[torch.argsort(margin_flat[other], descending=True)[:room]]
        else:
            other = other[:0]
        flat = torch.cat((loop_indices[:max_directed_edges], other))
    anchors = torch.div(flat, NUM_DIRECTIONS, rounding_mode="floor")
    directions = torch.remainder(flat, NUM_DIRECTIONS)
    targets = graph.predicted.flatten()[flat]
    return SelectedEdges(
        anchors.detach().cpu().numpy(),
        directions.detach().cpu().numpy(),
        targets.detach().cpu().numpy(),
        margin_flat[flat].detach().cpu().numpy(),
        loop_flat[flat].detach().cpu().numpy(),
        threshold,
    )


class ConsensusAssembler:
    """Greedily merge translated coordinate maps while rejecting conflicts."""

    def __init__(self, count: int) -> None:
        self.count = int(count)
        self.component_of = np.arange(count, dtype=np.int64)
        self.positions: dict[int, dict[int, tuple[int, int]]] = {
            node: {node: (0, 0)} for node in range(count)
        }
        self.accepted = 0
        self.rejected_geometry = 0
        self.rejected_collision = 0

    def add(self, anchor: int, target: int, direction: int) -> bool:
        if anchor == target:
            self.rejected_collision += 1
            return False
        delta = _DELTA[int(direction)]
        ca = int(self.component_of[anchor])
        cb = int(self.component_of[target])
        pa = self.positions[ca][anchor]
        pb = self.positions[cb][target]
        desired_target = (pa[0] + delta[0], pa[1] + delta[1])
        if ca == cb:
            if pb == desired_target:
                self.accepted += 1
                return True
            self.rejected_geometry += 1
            return False

        shift = (desired_target[0] - pb[0], desired_target[1] - pb[1])
        shifted_b = {
            node: (position[0] + shift[0], position[1] + shift[1])
            for node, position in self.positions[cb].items()
        }
        occupied = set(self.positions[ca].values())
        if occupied.intersection(shifted_b.values()):
            self.rejected_collision += 1
            return False
        self.positions[ca].update(shifted_b)
        for node in shifted_b:
            self.component_of[node] = ca
        del self.positions[cb]
        self.accepted += 1
        return True

    def add_edges(self, edges: SelectedEdges) -> None:
        # Strongest edges first. Loop-certified edges receive strict priority.
        order = sorted(
            range(len(edges.anchors)),
            key=lambda index: (bool(edges.loop[index]), float(edges.margins[index])),
            reverse=True,
        )
        for index in order:
            self.add(
                int(edges.anchors[index]),
                int(edges.targets[index]),
                int(edges.directions[index]),
            )


def island_metrics(
    assembler: ConsensusAssembler,
    *,
    permutation: np.ndarray,
    grid_side: int,
) -> dict[str, float]:
    """Translation-invariant island purity, size, and tile coverage."""
    component_sizes: list[int] = []
    component_purities: list[float] = []
    aligned_correct = 0
    pure_coverage = 0
    pure_sizes: list[int] = []
    for positions in assembler.positions.values():
        size = len(positions)
        component_sizes.append(size)
        offsets: dict[tuple[int, int], int] = {}
        for node, predicted in positions.items():
            cell = int(permutation[node])
            truth = (cell // grid_side, cell % grid_side)
            offset = (truth[0] - predicted[0], truth[1] - predicted[1])
            offsets[offset] = offsets.get(offset, 0) + 1
        correct = max(offsets.values())
        purity = correct / size
        component_purities.append(purity)
        aligned_correct += correct
        if size >= 2 and correct == size:
            pure_coverage += size
            pure_sizes.append(size)
    nontrivial = [size for size in component_sizes if size >= 2]
    return {
        "components": float(len(component_sizes)),
        "nontrivial_components": float(len(nontrivial)),
        "largest_component": float(max(component_sizes, default=1)),
        "largest_pure_component": float(max(pure_sizes, default=1)),
        "tiles_in_nontrivial_components": float(sum(nontrivial)),
        "pure_nontrivial_tile_coverage": pure_coverage / assembler.count,
        "translation_aligned_tile_accuracy": aligned_correct / assembler.count,
        "mean_component_purity": float(np.mean(component_purities)),
        "accepted_directed_constraints": float(assembler.accepted),
        "rejected_geometry": float(assembler.rejected_geometry),
        "rejected_collision": float(assembler.rejected_collision),
    }


def edge_metrics(edges: SelectedEdges, exact_targets: Tensor) -> dict[str, float]:
    if len(edges.anchors) == 0:
        return {"selected_directed_edges": 0.0, "exact_edge_precision": 0.0}
    truth = exact_targets[
        0,
        torch.from_numpy(edges.anchors).to(exact_targets.device),
        torch.from_numpy(edges.directions).to(exact_targets.device),
    ].detach().cpu().numpy()
    exact = edges.targets == truth
    return {
        "selected_directed_edges": float(len(exact)),
        "exact_edge_precision": float(exact.mean()),
        "exact_directed_edges": float(exact.sum()),
        "selected_loop_edges": float(edges.loop.sum()),
    }


def smoke_test() -> dict[str, float]:
    """Small exact geometry contract, including conflict rejection."""
    edges = SelectedEdges(
        anchors=np.array([0, 1, 0, 2, 3]),
        directions=np.array([RIGHT, DOWN, DOWN, RIGHT, LEFT]),
        targets=np.array([1, 3, 2, 3, 0]),
        margins=np.array([5.0, 4.0, 5.0, 4.0, 1.0]),
        loop=np.array([True, True, True, True, False]),
        threshold=0.0,
    )
    assembler = ConsensusAssembler(4)
    assembler.add_edges(edges)
    permutation = np.array([0, 1, 2, 3])
    metrics = island_metrics(assembler, permutation=permutation, grid_side=2)
    if metrics["largest_pure_component"] != 4.0:
        raise AssertionError("exact 2x2 island was not recovered")
    if assembler.rejected_geometry != 1:
        raise AssertionError("inconsistent closing constraint was not rejected")
    return metrics


if __name__ == "__main__":
    print(smoke_test())
