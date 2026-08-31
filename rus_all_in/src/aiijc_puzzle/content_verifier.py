"""Content-aware listwise verification over dirty-tile candidate pools.

Candidate generation and model inputs are inference-visible: only corrupted,
shuffled tiles and classical costs derived from them are passed to the model.
Clean targets are used exclusively to recover training/evaluation labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from aiijc_puzzle.candidate_supply import (
    DEFAULT_VIEWS,
    analytic_views,
    classical_costs,
    recover_layout,
    top_candidates,
)


@dataclass(frozen=True)
class CandidateRow:
    """One directed neighbour query and its inference-visible shortlist."""

    anchor: int
    candidates: np.ndarray
    features: np.ndarray
    candidate_rmse: np.ndarray
    candidate_mapping_margin: np.ndarray
    mapping_confidence_cut: float
    exact_candidate: int
    baseline_choices: tuple[int, ...]
    ensemble_choice: int
    direction: int
    trusted: bool

    def positives(self, threshold: float) -> np.ndarray:
        return self.candidate_rmse <= threshold

    def training_positives(self, threshold: float) -> np.ndarray:
        """Return content positives whose recovered candidate mapping is trusted.

        A trusted row's exact neighbour remains positive even if floating-point
        ties put its margin infinitesimally below the board-median cutoff.
        """
        result = self.positives(threshold) & (
            self.candidate_mapping_margin >= self.mapping_confidence_cut
        )
        if self.trusted and self.exact_candidate >= 0:
            result = result.copy()
            result[self.exact_candidate] = True
        return result


@dataclass(frozen=True)
class CandidateBoard:
    """Dirty tiles plus target-assisted labels for one board."""

    filename: str
    tiles: np.ndarray
    rows: tuple[CandidateRow, ...]
    views: tuple[str, ...]
    median_mapping_margin: float


def _candidate_features(
    anchor: int,
    candidates: np.ndarray,
    costs: Sequence[np.ndarray],
    ranked: Sequence[np.ndarray],
    k: int,
) -> np.ndarray:
    """Classical dirty-only scores, shortlist ranks, and emitter flags."""
    columns: list[np.ndarray] = []
    for cost, order in zip(costs, ranked, strict=True):
        # candidate_supply costs are already row median/MAD normalised.  Clip
        # only to keep the small scalar MLP numerically tame.
        columns.append(np.clip(cost[anchor, candidates], -10.0, 20.0) / 10.0)
        rank = np.full(len(candidates), k, dtype=np.float32)
        for position, tile in enumerate(order[anchor, :k]):
            rank[candidates == tile] = position
        columns.append(rank / max(k, 1))
        columns.append((rank < k).astype(np.float32))
    return np.stack(columns, axis=1).astype(np.float32)


def build_candidate_board(
    dirty: np.ndarray,
    clean: np.ndarray,
    *,
    filename: str = "",
    views: Sequence[str] = DEFAULT_VIEWS,
    candidate_k: int = 5,
) -> CandidateBoard:
    """Build the union-top-k pool and target-assisted labels for one board.

    Every query corresponds to a dirty anchor whose recovered clean position
    has a right or down neighbour.  At inference the same emitters can run for
    every dirty anchor; clean positions are needed here only to identify which
    rows have a labelled neighbour and to measure exact/content positives.
    """
    dirty = np.asarray(dirty)
    clean = np.asarray(clean)
    if dirty.shape != clean.shape or dirty.ndim != 4 or dirty.shape[-1] != 3:
        raise ValueError(f"expected equal N x H x W x 3 arrays, got {dirty.shape}, {clean.shape}")
    n = len(dirty)
    grid = round(n**0.5)
    if grid * grid != n:
        raise ValueError(f"tile count must be square, got {n}")
    if not 1 <= candidate_k < n:
        raise ValueError(f"candidate_k must be in [1, {n - 1}], got {candidate_k}")

    recovered = recover_layout(dirty, clean)
    position_of_dirty = recovered.position_of_dirty
    confidence_cut = float(np.median(recovered.margin_at_position))
    view_names = tuple(views)

    direction_costs: list[list[np.ndarray]] = [[], []]
    direction_ranked: list[list[np.ndarray]] = [[], []]
    for view in analytic_views(dirty, view_names).values():
        right_cost, down_cost = classical_costs(view)
        for axis, cost in enumerate((right_cost, down_cost)):
            direction_costs[axis].append(cost)
            direction_ranked[axis].append(top_candidates(cost, candidate_k))

    clean_flat = clean.astype(np.float32).reshape(n, -1)
    rows: list[CandidateRow] = []
    for axis, delta in ((0, 1), (1, grid)):
        costs = direction_costs[axis]
        ranked = direction_ranked[axis]
        for anchor in range(n):
            position = int(position_of_dirty[anchor])
            legal = position % grid != grid - 1 if axis == 0 else position < n - grid
            if not legal:
                continue
            neighbour_position = position + delta
            true_dirty = int(recovered.dirty_at_position[neighbour_position])
            pool = np.unique(
                np.concatenate([order[anchor, :candidate_k] for order in ranked])
            ).astype(np.int64)
            # A deterministic dirty-only ordering makes batches reproducible.
            ensemble_cost = np.mean(
                np.stack([cost[anchor, pool] for cost in costs], axis=1), axis=1
            )
            pool = pool[np.lexsort((pool, ensemble_cost))]
            features = _candidate_features(anchor, pool, costs, ranked, candidate_k)
            candidate_positions = position_of_dirty[pool]
            candidate_mapping_margin = recovered.margin_at_position[candidate_positions].astype(
                np.float32
            )
            delta_pixels = clean_flat[candidate_positions] - clean_flat[neighbour_position]
            candidate_rmse = np.sqrt(np.mean(np.square(delta_pixels), axis=1)).astype(np.float32)
            exact_matches = np.flatnonzero(pool == true_dirty)
            exact_candidate = int(exact_matches[0]) if len(exact_matches) else -1
            baseline_choices = tuple(
                int(np.flatnonzero(pool == order[anchor, 0])[0]) for order in ranked
            )
            ensemble_choice = int(np.argmin(features[:, 0::3].mean(axis=1)))
            trusted = bool(
                recovered.margin_at_position[position] >= confidence_cut
                and recovered.margin_at_position[neighbour_position] >= confidence_cut
            )
            rows.append(
                CandidateRow(
                    anchor=anchor,
                    candidates=pool,
                    features=features,
                    candidate_rmse=candidate_rmse,
                    candidate_mapping_margin=candidate_mapping_margin,
                    mapping_confidence_cut=confidence_cut,
                    exact_candidate=exact_candidate,
                    baseline_choices=baseline_choices,
                    ensemble_choice=ensemble_choice,
                    direction=axis,
                    trusted=trusted,
                )
            )

    return CandidateBoard(
        filename=filename,
        tiles=np.clip(dirty, 0, 255).astype(np.uint8),
        rows=tuple(rows),
        views=view_names,
        median_mapping_margin=confidence_cut,
    )


def tile_channels(tiles: torch.Tensor) -> torch.Tensor:
    """Return raw-colour and per-tile-normalised channels."""
    raw = tiles.float() / 127.5 - 1.0
    mean = raw.mean(dim=(1, 2, 3), keepdim=True)
    std = raw.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    local = ((raw - mean) / std).clamp(-4.0, 4.0) / 4.0
    return torch.cat((raw, local), dim=1)


class ContentListwiseVerifier(nn.Module):
    """Full-tile pair cross-attention followed by shortlist attention.

    Unlike a pooled bi-encoder, each candidate is jointly encoded with the
    anchor: their 5x5 spatial token grids attend to one another before a score
    exists.  A second permutation-equivariant transformer lets candidates be
    compared in the context of their rivals.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        dim: int = 32,
        heads: int = 4,
        pair_layers: int = 1,
        list_layers: int = 1,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.stem = nn.Sequential(
            nn.Conv2d(6, dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.side = nn.Parameter(torch.zeros(2, 1, dim))
        self.spatial_position = nn.Parameter(torch.empty(1, 25, dim))
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        self.direction = nn.Embedding(2, dim)
        pair_layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.pair_mix = nn.TransformerEncoder(pair_layer, pair_layers)
        self.feature_mix = nn.Sequential(
            nn.Linear(feature_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        list_layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.list_mix = nn.TransformerEncoder(list_layer, list_layers)
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _tokens(features: torch.Tensor) -> torch.Tensor:
        return features.flatten(2).transpose(1, 2)

    def forward(
        self,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        classical_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """Score padded candidate rows; ``True`` mask entries are valid."""
        batch, count = candidates.shape[:2]
        anchor_map = self.stem(tile_channels(anchors))
        candidate_map = self.stem(
            tile_channels(candidates.reshape(batch * count, *candidates.shape[2:]))
        )
        anchor_tokens = self._tokens(anchor_map)
        flat_candidate_tokens = self._tokens(candidate_map)
        if anchor_tokens.shape[1] != self.spatial_position.shape[1]:
            raise ValueError(
                "tile stem must produce a 5x5 grid, got "
                f"{anchor_map.shape[-2]}x{anchor_map.shape[-1]}"
            )
        # The same 2-D grid is shared by both sides.  Side embeddings identify
        # anchor versus candidate; spatial positions preserve row/column
        # relationships during their joint attention.
        anchor_tokens = anchor_tokens + self.spatial_position
        flat_candidate_tokens = flat_candidate_tokens + self.spatial_position
        candidate_tokens = flat_candidate_tokens.reshape(batch, count, -1, anchor_map.shape[1])
        anchor_tokens = anchor_tokens[:, None].expand(-1, count, -1, -1)
        pair_count = batch * count
        direction = self.direction(directions)[:, None, None, :].expand(-1, count, 1, -1)
        cls = self.cls.expand(pair_count, -1, -1).reshape(batch, count, 1, -1) + direction
        pair_tokens = torch.cat(
            (
                cls,
                anchor_tokens + self.side[0],
                candidate_tokens + self.side[1],
            ),
            dim=2,
        )
        pair_repr = self.pair_mix(pair_tokens.reshape(pair_count, pair_tokens.shape[2], -1))[
            :, 0
        ].reshape(batch, count, -1)
        pair_repr = pair_repr + self.feature_mix(classical_features)
        mixed = self.list_mix(pair_repr, src_key_padding_mask=~candidate_mask)
        # The frozen classical ensemble is the exact initial behaviour; the
        # model must learn a content-based residual to move away from it.
        prior = -classical_features[..., 0::3].mean(dim=-1) * self.prior_scale
        logits = prior + self.score(mixed).squeeze(-1)
        return logits.masked_fill(~candidate_mask, -1e4)


def multi_positive_listwise_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log probability assigned to any acceptable candidate."""
    if logits.shape != positive_mask.shape or logits.shape != candidate_mask.shape:
        raise ValueError("logits and masks must have identical shapes")
    usable = (positive_mask & candidate_mask).any(dim=1)
    if not bool(usable.any()):
        return logits.sum() * 0.0
    valid_logits = logits.masked_fill(~candidate_mask, -torch.inf)
    positive_logits = logits.masked_fill(~(positive_mask & candidate_mask), -torch.inf)
    losses = torch.logsumexp(valid_logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
    return losses[usable].mean()


def summarize_choices(
    rows: Sequence[CandidateRow],
    choices: Sequence[int],
    *,
    scope: str,
) -> dict[str, float | int]:
    """Summarise top-1 choices under all/query-trusted/strict-trusted labels.

    Exact identity is valid whenever the anchor/true-neighbour query is
    trusted.  In strict ``trusted`` content metrics, a selected non-exact
    candidate also needs a trusted recovered mapping because its clean RMSE is
    otherwise label-noisy.  ``trusted_query`` retains the historical weaker
    diagnostic for explicit comparison.
    """
    if len(rows) != len(choices):
        raise ValueError("rows and choices must have equal length")
    if scope not in {"all", "trusted_query", "trusted"}:
        raise ValueError(f"unknown scope: {scope}")
    selected = [
        (row, int(choice))
        for row, choice in zip(rows, choices, strict=True)
        if scope == "all" or row.trusted
    ]
    if not selected:
        return {"rows": 0, "exact": 0.0, "content_rmse_le_10": 0.0, "content_rmse_le_20": 0.0}
    exact = sum(choice == row.exact_candidate for row, choice in selected)

    def content_hit(row: CandidateRow, choice: int, threshold: float) -> bool:
        mapping_is_trusted = (
            float(row.candidate_mapping_margin[choice]) >= row.mapping_confidence_cut
        )
        label_is_valid = scope != "trusted" or mapping_is_trusted
        return label_is_valid and float(row.candidate_rmse[choice]) <= threshold

    rmse10 = sum(content_hit(row, choice, 10.0) for row, choice in selected)
    rmse20 = sum(content_hit(row, choice, 20.0) for row, choice in selected)
    count = len(selected)
    return {
        "rows": count,
        "exact": exact / count,
        "content_rmse_le_10": rmse10 / count,
        "content_rmse_le_20": rmse20 / count,
    }


def summarize_oracle(rows: Sequence[CandidateRow], *, scope: str) -> dict[str, float | int]:
    """Summarise whether the candidate pool contains an acceptable item."""
    if scope not in {"all", "trusted_query", "trusted"}:
        raise ValueError(f"unknown scope: {scope}")
    selected = [row for row in rows if scope == "all" or row.trusted]
    if not selected:
        return {
            "rows": 0,
            "mean_candidates": 0.0,
            "exact": 0.0,
            "content_rmse_le_10": 0.0,
            "content_rmse_le_20": 0.0,
        }
    count = len(selected)

    def content_oracle(row: CandidateRow, threshold: float) -> bool:
        positives = row.positives(threshold)
        if scope == "trusted":
            positives &= row.candidate_mapping_margin >= row.mapping_confidence_cut
        return bool(positives.any())

    return {
        "rows": count,
        "mean_candidates": float(np.mean([len(row.candidates) for row in selected])),
        "exact": sum(row.exact_candidate >= 0 for row in selected) / count,
        "content_rmse_le_10": sum(content_oracle(row, 10.0) for row in selected) / count,
        "content_rmse_le_20": sum(content_oracle(row, 20.0) for row in selected) / count,
    }
