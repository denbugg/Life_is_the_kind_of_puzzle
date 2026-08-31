"""Joint dense-contact pose retrieval for frozen TASKA components.

This module deliberately differs from the earlier independent absolute
component heads.  It builds a target-blind graph from dense top-k TASKA
boundary contacts, contextualises every real component in the current board,
and scores only contact-implied rigid translations.  The learned score is not
used as a raw-seam veto.  A separate strict packer can place several selected
components simultaneously while retaining every original upright tile.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as functional

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_component_relation_anchor import (
    build_realised_focal_components,
)

NODE_FEATURE_DIM = 68
PAIR_FEATURE_DIM = 14
CANDIDATE_FEATURE_DIM = 19


@dataclass(frozen=True)
class JointPoseBoard:
    """One target-blind variable-size component graph."""

    layout: np.ndarray
    component_of_tile: np.ndarray
    component_relative_coordinates: np.ndarray
    component_sizes: np.ndarray
    component_origins: np.ndarray
    node_features: np.ndarray
    pair_index: np.ndarray
    pair_features: np.ndarray
    candidate_component: np.ndarray
    candidate_shift: np.ndarray
    candidate_features: np.ndarray
    candidate_raw_score: np.ndarray


@dataclass(frozen=True)
class JointPoseTargets:
    """Exact synthetic labels kept outside the inference interface."""

    dominant_shift: np.ndarray
    dominant_support: np.ndarray
    purity: np.ndarray
    covered: np.ndarray
    positive_candidate: np.ndarray


@dataclass(frozen=True)
class JointPoseOutput:
    """Model outputs for one board."""

    candidate_score: torch.Tensor
    coverage_logit: torch.Tensor
    purity_logit: torch.Tensor


@dataclass(frozen=True)
class MultiAnchorPackingDiagnostics:
    """Auditable strict-packing diagnostics."""

    requested_anchor_count: int
    placed_anchor_count: int
    preserved_component_count: int
    repacked_component_count: int
    deferred_tile_count: int
    total_tile_l1_displacement: int
    strict_permutation: bool


def _strict_layout(value: Any, *, grid: int) -> np.ndarray:
    count = grid * grid
    raw = np.asarray(value)
    if raw.shape != (count,) or raw.dtype.kind not in {"i", "u"}:
        raise ValueError("layout must be a one-dimensional integer permutation")
    result = np.ascontiguousarray(raw, dtype=np.int32)
    if not np.array_equal(np.sort(result), np.arange(count, dtype=np.int32)):
        raise ValueError("layout must contain every input tile exactly once")
    return result


def _normalise_cost(value: Any, *, count: int) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != (count, count) or not np.isfinite(result).all():
        raise ValueError("directional cost matrix must be finite and square")
    return result


def tile_visible_descriptors(dirty_tiles: Any) -> np.ndarray:
    """Extract compact dirty-visible colour/boundary descriptors per tile."""

    raw = np.asarray(dirty_tiles)
    if raw.ndim != 4 or raw.shape[1:] != (20, 20, 3):
        raise ValueError("dirty_tiles must have shape [N,20,20,3]")
    tiles = np.ascontiguousarray(raw, dtype=np.float32) / 255.0
    mean = tiles.mean(axis=(1, 2))
    std = tiles.std(axis=(1, 2))
    top = tiles[:, :2].mean(axis=(1, 2))
    bottom = tiles[:, -2:].mean(axis=(1, 2))
    left = tiles[:, :, :2].mean(axis=(1, 2))
    right = tiles[:, :, -2:].mean(axis=(1, 2))
    centre = tiles[:, 5:15, 5:15].mean(axis=(1, 2))
    horizontal = np.abs(np.diff(tiles, axis=2)).mean(axis=(1, 2))
    vertical = np.abs(np.diff(tiles, axis=1)).mean(axis=(1, 2))
    result = np.concatenate(
        (mean, std, top, bottom, left, right, centre, horizontal, vertical), axis=1
    )
    if result.shape != (len(tiles), 27) or not np.isfinite(result).all():
        raise RuntimeError("dirty-visible tile descriptor contract failed")
    return np.ascontiguousarray(result, dtype=np.float32)


def _component_arrays(
    layout: np.ndarray,
    components: Sequence[Sequence[int]],
    *,
    grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = grid * grid
    component_of = np.empty(count, dtype=np.int32)
    relative = np.empty((count, 2), dtype=np.int16)
    sizes = np.empty(len(components), dtype=np.int16)
    origins = np.empty((len(components), 2), dtype=np.int16)
    position = np.empty(count, dtype=np.int32)
    position[layout] = np.arange(count, dtype=np.int32)
    rows, columns = divmod(position, grid)
    for index, component in enumerate(components):
        tiles = np.asarray(component, dtype=np.int32)
        minimum_row = int(rows[tiles].min())
        minimum_column = int(columns[tiles].min())
        component_of[tiles] = index
        relative[tiles, 0] = rows[tiles] - minimum_row
        relative[tiles, 1] = columns[tiles] - minimum_column
        sizes[index] = len(tiles)
        origins[index] = (minimum_row, minimum_column)
    flattened = np.concatenate([np.asarray(component) for component in components])
    if not np.array_equal(np.sort(flattened), np.arange(count)):
        raise RuntimeError("components do not partition all tiles")
    return component_of, relative, sizes, origins


def _node_features(
    *,
    layout: np.ndarray,
    components: Sequence[Sequence[int]],
    component_relative: np.ndarray,
    component_sizes: np.ndarray,
    component_origins: np.ndarray,
    descriptors: np.ndarray,
    realised_count: int,
    grid: int,
) -> np.ndarray:
    count = grid * grid
    rows: list[np.ndarray] = []
    for index, component in enumerate(components):
        tiles = np.asarray(component, dtype=np.int32)
        relative = component_relative[tiles].astype(np.float32)
        height = int(relative[:, 0].max()) + 1
        width = int(relative[:, 1].max()) + 1
        size = int(component_sizes[index])
        values = descriptors[tiles]
        geometry = np.asarray(
            [
                size / count,
                np.log1p(size) / np.log1p(count),
                height / grid,
                width / grid,
                size / (height * width),
                component_origins[index, 0] / (grid - 1),
                component_origins[index, 1] / (grid - 1),
                (component_origins[index, 0] + (height - 1) / 2) / (grid - 1),
                (component_origins[index, 1] + (width - 1) / 2) / (grid - 1),
                float(size == 1),
                realised_count / max(1, count - len(components)),
                relative[:, 0].std() / grid,
                relative[:, 1].std() / grid,
                np.mean(relative[:, 0]) / grid,
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate((geometry, values.mean(axis=0), values.max(axis=0))))
    result = np.ascontiguousarray(rows, dtype=np.float32)
    if result.shape != (len(components), NODE_FEATURE_DIM) or not np.isfinite(result).all():
        raise RuntimeError("node feature contract failed")
    return result


def _external_topk(
    costs: np.ndarray,
    component_of: np.ndarray,
    component_index: int,
    tile: int,
    *,
    topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(costs)
    shortlist = min(count, topk + max(32, int(component_of.size // 8)))
    if shortlist == count:
        order = np.argsort(costs, kind="stable")
    else:
        indices = np.argpartition(costs, shortlist - 1)[:shortlist]
        order = indices[np.argsort(costs[indices], kind="stable")]
    selected = [
        int(other)
        for other in order
        if int(other) != tile and int(component_of[other]) != component_index
    ][:topk]
    if len(selected) < topk:
        order = np.argsort(costs, kind="stable")
        selected = [
            int(other)
            for other in order
            if int(other) != tile and int(component_of[other]) != component_index
        ][:topk]
    targets = np.asarray(selected, dtype=np.int32)
    ranks = np.arange(len(targets), dtype=np.float64)
    finite_std = float(costs.std())
    standard = max(finite_std, 1e-6)
    goodness = -(costs[targets] - float(costs.mean())) / standard
    return targets, ranks, goodness


def _aggregate_pair_features(
    evidence: Sequence[tuple[float, float, int, int, int]],
    *,
    grid: int,
) -> np.ndarray:
    rank = np.asarray([item[0] for item in evidence], dtype=np.float64)
    goodness = np.asarray([item[1] for item in evidence], dtype=np.float64)
    direction = np.asarray([item[2] for item in evidence], dtype=np.int32)
    shifts = np.asarray([(item[3], item[4]) for item in evidence], dtype=np.float64)
    fractions = np.bincount(direction, minlength=4).astype(np.float64) / len(evidence)
    return np.asarray(
        [
            np.log1p(len(evidence)),
            rank.max(),
            rank.mean(),
            goodness.max(),
            goodness.mean(),
            *fractions,
            shifts[:, 0].mean() / (grid - 1),
            shifts[:, 1].mean() / (grid - 1),
            shifts[:, 0].std() / grid,
            shifts[:, 1].std() / grid,
            float(len({(item[3], item[4]) for item in evidence}) == 1),
        ],
        dtype=np.float32,
    )


def _aggregate_candidate_features(
    evidence: Sequence[tuple[float, float, int, int]],
    *,
    shift: tuple[int, int],
    origin: np.ndarray,
    height: int,
    width: int,
    grid: int,
) -> tuple[np.ndarray, float]:
    rank = np.asarray([item[0] for item in evidence], dtype=np.float64)
    goodness = np.asarray([item[1] for item in evidence], dtype=np.float64)
    direction = np.asarray([item[2] for item in evidence], dtype=np.int32)
    fractions = np.bincount(direction, minlength=4).astype(np.float64) / len(evidence)
    row = int(origin[0]) + shift[0]
    column = int(origin[1]) + shift[1]
    unique_targets = len({item[3] for item in evidence})
    raw_score = float(rank.sum() + 0.25 * goodness.max() + 0.10 * np.log1p(len(evidence)))
    features = np.asarray(
        [
            np.log1p(len(evidence)),
            np.log1p(unique_targets),
            rank.max(),
            rank.mean(),
            rank.sum() / max(1.0, np.sqrt(len(evidence))),
            goodness.max(),
            goodness.mean(),
            *fractions,
            shift[0] / (grid - 1),
            shift[1] / (grid - 1),
            row / (grid - 1),
            column / (grid - 1),
            (row + height - 1) / (grid - 1),
            (column + width - 1) / (grid - 1),
            (abs(shift[0]) + abs(shift[1])) / (2 * (grid - 1)),
            float(shift == (0, 0)),
        ],
        dtype=np.float32,
    )
    return features, raw_score


def build_joint_pose_board(
    *,
    layout: Any,
    dirty_tiles: Any,
    cost_right: Any,
    cost_down: Any,
    selected_edges: Sequence[RawTailEdge],
    selected_logits: Any,
    grid: int = 24,
    dense_topk: int = 8,
    candidate_cap: int = 128,
    focal_threshold: float = 0.0,
) -> JointPoseBoard:
    """Build one all-component dense-contact graph without exact targets."""

    strict = _strict_layout(layout, grid=grid)
    count = grid * grid
    if isinstance(dense_topk, bool) or not 1 <= dense_topk < count:
        raise ValueError("dense_topk must be between one and tile count minus one")
    if isinstance(candidate_cap, bool) or candidate_cap < 2:
        raise ValueError("candidate_cap must be at least two")
    right = _normalise_cost(cost_right, count=count)
    down = _normalise_cost(cost_down, count=count)
    logits = np.ascontiguousarray(selected_logits, dtype=np.float64)
    components_raw, realised_count = build_realised_focal_components(
        strict,
        selected_edges,
        logits,
        grid=grid,
        focal_threshold=focal_threshold,
    )
    components = tuple(tuple(int(tile) for tile in component) for component in components_raw)
    component_of, relative, sizes, origins = _component_arrays(
        strict, components, grid=grid
    )
    descriptors = tile_visible_descriptors(dirty_tiles)
    if len(descriptors) != count:
        raise ValueError("dirty tile count differs from layout")
    node_features = _node_features(
        layout=strict,
        components=components,
        component_relative=relative,
        component_sizes=sizes,
        component_origins=origins,
        descriptors=descriptors,
        realised_count=realised_count,
        grid=grid,
    )
    position = np.empty(count, dtype=np.int32)
    position[strict] = np.arange(count, dtype=np.int32)
    tile_rows, tile_columns = divmod(position, grid)
    pair_evidence: dict[tuple[int, int], list[tuple[float, float, int, int, int]]] = (
        defaultdict(list)
    )
    candidate_evidence: list[
        dict[tuple[int, int], list[tuple[float, float, int, int]]]
    ] = [defaultdict(list) for _ in components]
    for tile in range(count):
        component_index = int(component_of[tile])
        specifications = (
            (right[tile], 0, 1, 0, False),
            (right[:, tile], 0, 1, 1, True),
            (down[tile], 1, 0, 2, False),
            (down[:, tile], 1, 0, 3, True),
        )
        for costs, delta_row, delta_column, direction, reverse in specifications:
            targets, ranks, goodness = _external_topk(
                costs,
                component_of,
                component_index,
                tile,
                topk=dense_topk,
            )
            for rank_index, (other, zscore) in enumerate(
                zip(targets, goodness, strict=True)
            ):
                target_component = int(component_of[other])
                if reverse:
                    shift = (
                        int(tile_rows[other] + delta_row - tile_rows[tile]),
                        int(tile_columns[other] + delta_column - tile_columns[tile]),
                    )
                else:
                    shift = (
                        int(tile_rows[other] - delta_row - tile_rows[tile]),
                        int(tile_columns[other] - delta_column - tile_columns[tile]),
                    )
                component_tiles = np.where(component_of == component_index)[0]
                destination_rows = tile_rows[component_tiles] + shift[0]
                destination_columns = tile_columns[component_tiles] + shift[1]
                if (
                    np.any(destination_rows < 0)
                    or np.any(destination_rows >= grid)
                    or np.any(destination_columns < 0)
                    or np.any(destination_columns >= grid)
                ):
                    continue
                rank_score = float(np.exp(-float(ranks[rank_index]) / 2.0))
                pair_evidence[(component_index, target_component)].append(
                    (rank_score, float(zscore), direction, shift[0], shift[1])
                )
                candidate_evidence[component_index][shift].append(
                    (rank_score, float(zscore), direction, target_component)
                )
    pair_keys = sorted(pair_evidence)
    pair_index = np.asarray(pair_keys, dtype=np.int32).T
    pair_features = np.asarray(
        [
            _aggregate_pair_features(pair_evidence[key], grid=grid)
            for key in pair_keys
        ],
        dtype=np.float32,
    )
    candidate_component: list[int] = []
    candidate_shift: list[tuple[int, int]] = []
    candidate_features: list[np.ndarray] = []
    candidate_raw: list[float] = []
    for component_index, component in enumerate(components):
        if len(component) < 2:
            continue
        component_relative = relative[np.asarray(component, dtype=np.int32)]
        height = int(component_relative[:, 0].max()) + 1
        width = int(component_relative[:, 1].max()) + 1
        rows: list[tuple[float, tuple[int, int], np.ndarray]] = []
        for shift, evidence in candidate_evidence[component_index].items():
            features, raw_score = _aggregate_candidate_features(
                evidence,
                shift=shift,
                origin=origins[component_index],
                height=height,
                width=width,
                grid=grid,
            )
            rows.append((raw_score, shift, features))
        zero_features = np.zeros(CANDIDATE_FEATURE_DIM, dtype=np.float32)
        zero_features[-1] = 1.0
        if not any(shift == (0, 0) for _, shift, _ in rows):
            rows.append((-1e6, (0, 0), zero_features))
        rows.sort(key=lambda item: (-item[0], item[1]))
        kept = rows[:candidate_cap]
        if not any(shift == (0, 0) for _, shift, _ in kept):
            kept[-1] = (-1e6, (0, 0), zero_features)
            kept.sort(key=lambda item: (-item[0], item[1]))
        for raw_score, shift, features in kept:
            candidate_component.append(component_index)
            candidate_shift.append(shift)
            candidate_features.append(features)
            candidate_raw.append(raw_score)
    result = JointPoseBoard(
        layout=strict,
        component_of_tile=np.ascontiguousarray(component_of, dtype=np.int32),
        component_relative_coordinates=np.ascontiguousarray(relative, dtype=np.int16),
        component_sizes=np.ascontiguousarray(sizes, dtype=np.int16),
        component_origins=np.ascontiguousarray(origins, dtype=np.int16),
        node_features=node_features,
        pair_index=np.ascontiguousarray(pair_index, dtype=np.int32),
        pair_features=np.ascontiguousarray(pair_features, dtype=np.float32),
        candidate_component=np.ascontiguousarray(candidate_component, dtype=np.int32),
        candidate_shift=np.ascontiguousarray(candidate_shift, dtype=np.int16),
        candidate_features=np.ascontiguousarray(candidate_features, dtype=np.float32),
        candidate_raw_score=np.ascontiguousarray(candidate_raw, dtype=np.float32),
    )
    if result.pair_index.shape != (2, len(result.pair_features)):
        raise RuntimeError("pair graph contract failed")
    if not (
        len(result.candidate_component)
        == len(result.candidate_shift)
        == len(result.candidate_features)
        == len(result.candidate_raw_score)
    ):
        raise RuntimeError("candidate graph contract failed")
    return result


def joint_pose_targets(
    board: JointPoseBoard,
    reference_layout: Any,
    *,
    grid: int = 24,
) -> JointPoseTargets:
    """Attach dominant exact translation labels after inference-data freeze."""

    reference = _strict_layout(reference_layout, grid=grid)
    count = grid * grid
    current_position = np.empty(count, dtype=np.int32)
    target_position = np.empty(count, dtype=np.int32)
    current_position[board.layout] = np.arange(count, dtype=np.int32)
    target_position[reference] = np.arange(count, dtype=np.int32)
    current_rows, current_columns = divmod(current_position, grid)
    target_rows, target_columns = divmod(target_position, grid)
    component_count = len(board.component_sizes)
    dominant_shift = np.zeros((component_count, 2), dtype=np.int16)
    support = np.zeros(component_count, dtype=np.int16)
    purity = np.zeros(component_count, dtype=np.float32)
    for component_index in range(component_count):
        tiles = np.where(board.component_of_tile == component_index)[0]
        shifts = Counter(
            zip(
                (target_rows[tiles] - current_rows[tiles]).tolist(),
                (target_columns[tiles] - current_columns[tiles]).tolist(),
                strict=True,
            )
        )
        shift, value = min(shifts.items(), key=lambda item: (-item[1], item[0]))
        dominant_shift[component_index] = shift
        support[component_index] = value
        purity[component_index] = value / len(tiles)
    positive = np.all(
        board.candidate_shift
        == dominant_shift[board.candidate_component],
        axis=1,
    )
    covered = np.zeros(component_count, dtype=bool)
    np.logical_or.at(covered, board.candidate_component, positive)
    return JointPoseTargets(
        dominant_shift=dominant_shift,
        dominant_support=support,
        purity=purity,
        covered=covered,
        positive_candidate=np.ascontiguousarray(positive, dtype=bool),
    )


class _GraphBlock(nn.Module):
    def __init__(self, width: int, pair_dim: int, heads: int) -> None:
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(width * 2 + pair_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.update = nn.Sequential(
            nn.Linear(width * 3, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.graph_norm = nn.LayerNorm(width)
        self.attention = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 3,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        pair_index: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        source, target = pair_index
        messages = self.edge(
            torch.cat((nodes[source], nodes[target], pair_features), dim=-1)
        )
        outgoing = torch.zeros_like(nodes)
        incoming = torch.zeros_like(nodes)
        outgoing.index_add_(0, source, messages)
        incoming.index_add_(0, target, messages)
        out_degree = torch.zeros(len(nodes), device=nodes.device, dtype=nodes.dtype)
        in_degree = torch.zeros_like(out_degree)
        ones = torch.ones(len(messages), device=nodes.device, dtype=nodes.dtype)
        out_degree.index_add_(0, source, ones)
        in_degree.index_add_(0, target, ones)
        outgoing = outgoing / out_degree.clamp_min(1).unsqueeze(-1)
        incoming = incoming / in_degree.clamp_min(1).unsqueeze(-1)
        nodes = self.graph_norm(
            nodes + self.update(torch.cat((nodes, outgoing, incoming), dim=-1))
        )
        return self.attention(nodes.unsqueeze(0)).squeeze(0)


class JointComponentPoseTransformer(nn.Module):
    """Edge-aware all-component Transformer with candidate-shift retrieval."""

    def __init__(
        self,
        *,
        node_dim: int = NODE_FEATURE_DIM,
        pair_dim: int = PAIR_FEATURE_DIM,
        candidate_dim: int = CANDIDATE_FEATURE_DIM,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.blocks = nn.ModuleList(
            [_GraphBlock(width, pair_dim, heads) for _ in range(layers)]
        )
        self.candidate = nn.Sequential(
            nn.Linear(width * 2 + candidate_dim, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.coverage = nn.Linear(width * 2, 1)
        self.purity = nn.Linear(width * 2, 1)
        nn.init.zeros_(self.candidate[-1].weight)
        nn.init.zeros_(self.candidate[-1].bias)

    @staticmethod
    def _normalised_raw(
        raw_score: torch.Tensor,
        component: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.empty_like(raw_score)
        for index in torch.unique(component, sorted=True):
            mask = component == index
            values = raw_score[mask]
            result[mask] = (values - values.mean()) / values.std(unbiased=False).clamp_min(
                1e-4
            )
        return result

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        pair_index: torch.Tensor,
        pair_features: torch.Tensor,
        candidate_component: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_raw_score: torch.Tensor,
    ) -> JointPoseOutput:
        nodes = self.node_encoder(node_features)
        for block in self.blocks:
            nodes = block(nodes, pair_index, pair_features)
        global_token = nodes.mean(dim=0, keepdim=True).expand(len(nodes), -1)
        contextual = torch.cat((nodes, global_token), dim=-1)
        candidate_context = torch.cat(
            (
                contextual[candidate_component],
                candidate_features,
            ),
            dim=-1,
        )
        residual = self.candidate(candidate_context).squeeze(-1)
        score = self._normalised_raw(candidate_raw_score, candidate_component) + residual
        return JointPoseOutput(
            candidate_score=score,
            coverage_logit=self.coverage(contextual).squeeze(-1),
            purity_logit=self.purity(contextual).squeeze(-1),
        )


def joint_pose_loss(
    output: JointPoseOutput,
    *,
    candidate_component: torch.Tensor,
    positive_candidate: torch.Tensor,
    component_sizes: torch.Tensor,
    dominant_support: torch.Tensor,
    purity: torch.Tensor,
    covered: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Listwise covered-shift retrieval plus calibrated coverage/purity heads."""

    retrieval_terms: list[torch.Tensor] = []
    retrieval_weights: list[torch.Tensor] = []
    for component in torch.where(covered & (component_sizes >= 2))[0]:
        mask = candidate_component == component
        positive = positive_candidate[mask]
        if int(positive.sum()) != 1:
            raise RuntimeError("each covered component must have one positive candidate")
        log_probability = functional.log_softmax(output.candidate_score[mask], dim=0)
        retrieval_terms.append(-log_probability[positive][0])
        retrieval_weights.append(torch.sqrt(dominant_support[component].clamp_min(1)))
    if not retrieval_terms:
        raise RuntimeError("board contains no supervised shift candidate")
    terms = torch.stack(retrieval_terms)
    weights = torch.stack(retrieval_weights)
    retrieval = (terms * weights).sum() / weights.sum()
    nontrivial = component_sizes >= 2
    positive_weight = torch.tensor(4.0, device=covered.device)
    coverage_loss = functional.binary_cross_entropy_with_logits(
        output.coverage_logit[nontrivial],
        covered[nontrivial].to(output.coverage_logit.dtype),
        pos_weight=positive_weight,
    )
    purity_loss = functional.binary_cross_entropy_with_logits(
        output.purity_logit[nontrivial],
        purity[nontrivial],
        weight=torch.sqrt(component_sizes[nontrivial].to(purity.dtype)),
    )
    total = retrieval + 0.25 * coverage_loss + 0.10 * purity_loss
    return total, {
        "retrieval": retrieval.detach(),
        "coverage": coverage_loss.detach(),
        "purity": purity_loss.detach(),
    }


def candidate_ranks(
    score: Any,
    board: JointPoseBoard,
    targets: JointPoseTargets,
) -> dict[int, int]:
    """Return one-based positive rank for each covered nontrivial component."""

    values = np.asarray(score, dtype=np.float64)
    if values.shape != (len(board.candidate_component),) or not np.isfinite(values).all():
        raise ValueError("candidate score is malformed")
    result: dict[int, int] = {}
    for component in np.where(targets.covered & (board.component_sizes >= 2))[0]:
        indices = np.where(board.candidate_component == component)[0]
        order = indices[np.argsort(-values[indices], kind="stable")]
        positive = np.where(targets.positive_candidate[order])[0]
        if len(positive) != 1:
            raise RuntimeError("covered component lost its unique positive")
        result[int(component)] = int(positive[0]) + 1
    return result


def select_component_anchors(
    output: JointPoseOutput,
    board: JointPoseBoard,
    *,
    maximum_anchors: int = 4,
    coverage_threshold: float = 0.5,
    purity_threshold: float = 0.5,
    candidate_probability_threshold: float = 0.05,
) -> tuple[tuple[int, tuple[int, int], float], ...]:
    """Select up to a fixed number of confident nonzero component shifts."""

    score = output.candidate_score.detach().cpu()
    coverage = torch.sigmoid(output.coverage_logit).detach().cpu().numpy()
    purity = torch.sigmoid(output.purity_logit).detach().cpu().numpy()
    rows: list[tuple[float, int, tuple[int, int]]] = []
    for component in np.where(board.component_sizes >= 2)[0]:
        indices = np.where(board.candidate_component == component)[0]
        probability = torch.softmax(score[indices], dim=0).numpy()
        best_local = int(np.argmax(probability))
        index = int(indices[best_local])
        shift = tuple(int(value) for value in board.candidate_shift[index])
        if shift == (0, 0):
            continue
        if coverage[component] < coverage_threshold or purity[component] < purity_threshold:
            continue
        if probability[best_local] < candidate_probability_threshold:
            continue
        confidence = float(
            coverage[component]
            * purity[component]
            * probability[best_local]
            * np.sqrt(float(board.component_sizes[component]))
        )
        rows.append((confidence, int(component), shift))
    rows.sort(
        key=lambda row: (-row[0], -int(board.component_sizes[row[1]]), row[1], row[2])
    )
    return tuple(
        (component, shift, confidence)
        for confidence, component, shift in rows[:maximum_anchors]
    )


def pack_multiple_component_anchors(
    board: JointPoseBoard,
    anchors: Sequence[tuple[int, tuple[int, int], float]],
    *,
    grid: int = 24,
) -> tuple[np.ndarray, MultiAnchorPackingDiagnostics]:
    """Place several anchors, then preserve/repack all remaining components."""

    baseline = _strict_layout(board.layout, grid=grid)
    count = grid * grid
    component_count = len(board.component_sizes)
    occupied = np.full((grid, grid), -1, dtype=np.int32)
    selected: set[int] = set()
    placed = 0

    def locations(component: int, origin: tuple[int, int]) -> tuple[tuple[int, int, int], ...]:
        tiles = np.where(board.component_of_tile == component)[0]
        return tuple(
            (
                int(tile),
                int(origin[0] + board.component_relative_coordinates[tile, 0]),
                int(origin[1] + board.component_relative_coordinates[tile, 1]),
            )
            for tile in tiles
        )

    for component, shift, _ in anchors:
        if not 0 <= component < component_count or component in selected:
            continue
        origin = (
            int(board.component_origins[component, 0]) + int(shift[0]),
            int(board.component_origins[component, 1]) + int(shift[1]),
        )
        proposed = locations(component, origin)
        if any(
            row < 0
            or row >= grid
            or column < 0
            or column >= grid
            or occupied[row, column] >= 0
            for _, row, column in proposed
        ):
            continue
        for tile, row, column in proposed:
            occupied[row, column] = tile
        selected.add(component)
        placed += 1
    preserved = 0
    queue: list[int] = []
    for component in range(component_count):
        if component in selected:
            continue
        origin = tuple(int(value) for value in board.component_origins[component])
        proposed = locations(component, origin)
        if all(occupied[row, column] < 0 for _, row, column in proposed):
            for tile, row, column in proposed:
                occupied[row, column] = tile
            preserved += 1
        else:
            queue.append(component)
    repacked = 0
    deferred: list[int] = []
    for component in sorted(
        queue,
        key=lambda index: (-int(board.component_sizes[index]), index),
    ):
        tiles = np.where(board.component_of_tile == component)[0]
        relative = board.component_relative_coordinates[tiles]
        height = int(relative[:, 0].max()) + 1
        width = int(relative[:, 1].max()) + 1
        baseline_origin = board.component_origins[component]
        best: tuple[int, int, int] | None = None
        for row in range(grid - height + 1):
            for column in range(grid - width + 1):
                proposed = locations(component, (row, column))
                if any(occupied[r, c] >= 0 for _, r, c in proposed):
                    continue
                displacement = abs(row - int(baseline_origin[0])) + abs(
                    column - int(baseline_origin[1])
                )
                candidate = (displacement, row, column)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            deferred.extend(int(tile) for tile in tiles)
            continue
        for tile, row, column in locations(component, (best[1], best[2])):
            occupied[row, column] = tile
        repacked += 1
    empty = np.argwhere(occupied < 0)
    if len(empty) != len(deferred):
        raise RuntimeError("multi-anchor packing lost or duplicated tiles")
    if deferred:
        baseline_position = np.empty((count, 2), dtype=np.int32)
        baseline_position[baseline, 0], baseline_position[baseline, 1] = divmod(
            np.arange(count), grid
        )
        cost = np.empty((len(deferred), len(empty)), dtype=np.float64)
        for index, tile in enumerate(deferred):
            cost[index] = np.abs(empty - baseline_position[tile]).sum(axis=1)
        tile_rows, slot_columns = linear_sum_assignment(cost)
        for tile_row, slot_column in zip(tile_rows, slot_columns, strict=True):
            row, column = empty[slot_column]
            occupied[row, column] = deferred[tile_row]
    result = _strict_layout(occupied.reshape(-1), grid=grid)
    baseline_position = np.empty((count, 2), dtype=np.int32)
    result_position = np.empty((count, 2), dtype=np.int32)
    baseline_position[baseline, 0], baseline_position[baseline, 1] = divmod(
        np.arange(count), grid
    )
    result_position[result, 0], result_position[result, 1] = divmod(np.arange(count), grid)
    return result, MultiAnchorPackingDiagnostics(
        requested_anchor_count=len(anchors),
        placed_anchor_count=placed,
        preserved_component_count=preserved,
        repacked_component_count=repacked,
        deferred_tile_count=len(deferred),
        total_tile_l1_displacement=int(np.abs(result_position - baseline_position).sum()),
        strict_permutation=True,
    )


__all__ = [
    "CANDIDATE_FEATURE_DIM",
    "NODE_FEATURE_DIM",
    "PAIR_FEATURE_DIM",
    "JointComponentPoseTransformer",
    "JointPoseBoard",
    "JointPoseOutput",
    "JointPoseTargets",
    "MultiAnchorPackingDiagnostics",
    "build_joint_pose_board",
    "candidate_ranks",
    "joint_pose_loss",
    "joint_pose_targets",
    "pack_multiple_component_anchors",
    "select_component_anchors",
    "tile_visible_descriptors",
]
