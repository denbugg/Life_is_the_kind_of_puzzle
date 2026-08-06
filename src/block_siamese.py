"""Direct dirty-to-dirty metric learning for 4x4 macro-block membership.

The earlier block-identity experiment trained a dirty 20x20 tile against a
clean 80x80 macro-block.  That objective was useful as an information gate,
but its clean block prototypes do not exist at inference time.  This module
instead exposes the inference-time relation directly: dirty tiles from the
same 4x4 source block should be close, while tiles from different blocks of
the same image should be far apart.

The encoder remains Siamese (one shared network for every tile).  The loss is
a supervised contrastive loss over sibling tiles.  Importantly, the second
degradation of the *same* tile is excluded from the positive set: otherwise a
network can solve tile identity without learning the coarser same-block
relation needed by clustering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from config import NFRAG
from eval_block_identity import MACRO, NUM_BLOCKS, TILE_BLOCK_ID, TileToBlockEncoder


CAPACITY = MACRO * MACRO


class BlockSiamese(nn.Module):
    """Shared dirty-tile encoder used for same-macro-block metric learning."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.encoder = TileToBlockEncoder(embed_dim=self.embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.08)))

    def forward(self, tiles: Tensor) -> Tensor:
        return self.encoder(tiles)

    def scale(self) -> Tensor:
        return self.logit_scale.exp().clamp(min=1.0, max=100.0)


def sibling_supervised_contrastive_loss(
    embeddings: Tensor,
    block_labels: Tensor,
    tile_ids: Tensor,
    *,
    scale: Tensor | float,
) -> Tensor:
    """Supervised contrastive loss for dirty sibling tiles.

    Parameters
    ----------
    embeddings:
        Unit-normalized embeddings shaped ``(views, tiles, dim)``.
    block_labels:
        Source macro-block id for every sampled tile, shaped ``(tiles,)``.
    tile_ids:
        Stable source tile id for every sampled tile, shaped ``(tiles,)``.
    scale:
        Learned inverse temperature.

    Positives share a block but must be *different source tiles*.  Multiple
    corruptions of the same source tile are therefore neither positives nor
    negatives and are removed from the denominator altogether.
    """
    if embeddings.ndim != 3:
        raise ValueError(f"embeddings must be (views,tiles,dim), got {tuple(embeddings.shape)}")
    views, tiles, _ = embeddings.shape
    if views < 2:
        raise ValueError("at least two independently degraded views are required")
    if block_labels.shape != (tiles,) or tile_ids.shape != (tiles,):
        raise ValueError("block_labels and tile_ids must match the tile dimension")

    flat = F.normalize(embeddings.reshape(views * tiles, -1).float(), dim=-1)
    labels = block_labels.repeat(views)
    ids = tile_ids.repeat(views)
    same_tile = ids[:, None].eq(ids[None, :])
    same_block = labels[:, None].eq(labels[None, :])
    positive = same_block & ~same_tile
    allowed = ~same_tile

    positive_count = positive.sum(dim=1)
    valid = positive_count.gt(0)
    if not bool(valid.all()):
        raise ValueError("every sampled tile must have a different sibling from the same block")

    logits = flat @ flat.t()
    logits = logits * torch.as_tensor(scale, device=logits.device, dtype=logits.dtype)
    logits = logits.masked_fill(~allowed, -torch.inf)
    row_max = logits.max(dim=1, keepdim=True).values.detach()
    log_denom = torch.logsumexp(logits - row_max, dim=1) + row_max.squeeze(1)
    positive_log_prob = logits - log_denom[:, None]
    per_anchor = -(positive_log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive_count)
    return per_anchor[valid].mean()


@dataclass(frozen=True)
class RetrievalMetrics:
    top1: float
    precision_at_5: float
    recall_at_5: float
    reciprocal_precision: float
    reciprocal_edges: int

    def as_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {
            f"{prefix}top1_same_block": self.top1,
            f"{prefix}precision_at_5": self.precision_at_5,
            f"{prefix}recall_at_5": self.recall_at_5,
            f"{prefix}reciprocal_precision": self.reciprocal_precision,
            f"{prefix}reciprocal_edges": self.reciprocal_edges,
        }


def same_block_retrieval_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray = TILE_BLOCK_ID,
) -> RetrievalMetrics:
    """Measure whether nearest dirty tiles belong to the same source block."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError("embeddings and labels must have the same leading dimension")
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-8)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argpartition(-similarity, kth=4, axis=1)[:, :5]
    order_scores = np.take_along_axis(similarity, order, axis=1)
    order = np.take_along_axis(order, np.argsort(-order_scores, axis=1), axis=1)
    matches = labels[order] == labels[:, None]
    nearest = order[:, 0]
    mutual = np.arange(len(labels)) == nearest[nearest]
    reciprocal_anchors = np.flatnonzero(mutual)
    # Each undirected reciprocal pair appears twice.  Precision is unchanged,
    # while the reported edge count is made human-readable by dividing by two.
    reciprocal_precision = (
        float(np.mean(labels[nearest[reciprocal_anchors]] == labels[reciprocal_anchors]))
        if len(reciprocal_anchors)
        else 0.0
    )
    positives_per_anchor = np.bincount(labels, minlength=int(labels.max()) + 1)[labels] - 1
    recall5 = matches.sum(axis=1) / np.maximum(positives_per_anchor, 1)
    return RetrievalMetrics(
        top1=float(matches[:, 0].mean()),
        precision_at_5=float(matches.mean()),
        recall_at_5=float(recall5.mean()),
        reciprocal_precision=reciprocal_precision,
        reciprocal_edges=int(len(reciprocal_anchors) // 2),
    )


def _balanced_assignment(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    cost = 1.0 - embeddings @ centroids.T
    expanded_cost = np.repeat(cost, CAPACITY, axis=1)
    rows, columns = linear_sum_assignment(expanded_cost)
    assignment = np.empty(len(embeddings), dtype=np.int64)
    assignment[rows] = columns // CAPACITY
    return assignment


def _cluster_objective(embeddings: np.ndarray, assignment: np.ndarray, centroids: np.ndarray) -> float:
    return float(np.sum(embeddings * centroids[assignment]))


def balanced_spherical_kmeans(
    embeddings: np.ndarray,
    *,
    iterations: int = 20,
    restarts: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Balanced multi-start spherical k-means with exactly 16 tiles per group."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) != NFRAG:
        raise ValueError(f"expected ({NFRAG},D) embeddings, got {embeddings.shape}")
    if iterations < 1 or restarts < 1:
        raise ValueError("iterations and restarts must be positive")
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-8)
    best_assignment: np.ndarray | None = None
    best_objective = -np.inf

    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart * 104729)
        # Farthest-point seeding is deterministic after the first random point
        # and spreads prototypes over the embedding sphere.
        init = [int(rng.integers(0, NFRAG))]
        nearest_similarity = embeddings @ embeddings[init[0]]
        for _ in range(1, NUM_BLOCKS):
            candidate = int(np.argmin(nearest_similarity))
            init.append(candidate)
            nearest_similarity = np.maximum(nearest_similarity, embeddings @ embeddings[candidate])
        centroids = embeddings[np.asarray(init)].copy()
        assignment = np.zeros(NFRAG, dtype=np.int64)

        for _ in range(iterations):
            new_assignment = _balanced_assignment(embeddings, centroids)
            stable = np.array_equal(new_assignment, assignment)
            assignment = new_assignment
            for cluster in range(NUM_BLOCKS):
                mean = embeddings[assignment == cluster].mean(axis=0)
                norm = np.linalg.norm(mean)
                if norm > 1.0e-8:
                    centroids[cluster] = mean / norm
            if stable:
                break

        objective = _cluster_objective(embeddings, assignment, centroids)
        if objective > best_objective:
            best_objective = objective
            best_assignment = assignment.copy()

    if best_assignment is None:
        raise RuntimeError("balanced clustering produced no assignment")
    return best_assignment, best_objective


def clustering_metrics(
    assignment: np.ndarray,
    true_labels: np.ndarray = TILE_BLOCK_ID,
) -> dict[str, float | int]:
    """Permutation-invariant recovery metrics for 36 balanced source groups."""
    assignment = np.asarray(assignment, dtype=np.int64)
    true_labels = np.asarray(true_labels, dtype=np.int64)
    if assignment.shape != true_labels.shape:
        raise ValueError("assignment and true_labels must have identical shapes")
    overlap = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.int64)
    for cluster in range(NUM_BLOCKS):
        members = true_labels[assignment == cluster]
        overlap[cluster] = np.bincount(members, minlength=NUM_BLOCKS)
    rows, columns = linear_sum_assignment(-overlap)
    correct = overlap[rows, columns]
    return {
        "purity": float(correct.sum() / len(true_labels)),
        "perfect_blocks": int(np.sum(correct == CAPACITY)),
        "near_perfect_blocks": int(np.sum(correct >= CAPACITY - 2)),
        "mean_best_overlap": float(correct.mean()),
        "min_best_overlap": int(correct.min()),
    }


def smoke(seed: int = 7) -> dict[str, float | int]:
    """Data-free checks for the loss and balanced decoder."""
    torch.manual_seed(seed)
    views, tiles, dim = 2, 32, 16
    labels = torch.arange(tiles) // 8
    ids = torch.arange(tiles)
    raw = torch.randn(views, tiles, dim, requires_grad=True)
    embeddings = F.normalize(raw, dim=-1)
    loss = sibling_supervised_contrastive_loss(
        embeddings, labels, ids, scale=torch.tensor(12.5)
    )
    loss.backward()
    if not torch.isfinite(loss) or raw.grad is None or not torch.isfinite(raw.grad).all():
        raise AssertionError("contrastive loss produced non-finite values or gradients")

    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(NUM_BLOCKS, 48)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    toy = centers[TILE_BLOCK_ID] + rng.normal(scale=0.015, size=(NFRAG, 48))
    toy = toy.astype(np.float32)
    toy /= np.linalg.norm(toy, axis=1, keepdims=True)
    assignment, _ = balanced_spherical_kmeans(toy, iterations=15, restarts=3, seed=seed)
    cluster = clustering_metrics(assignment)
    retrieval = same_block_retrieval_metrics(toy)
    if cluster["purity"] < 0.999 or cluster["perfect_blocks"] != NUM_BLOCKS:
        raise AssertionError(f"balanced decoder failed a separable toy problem: {cluster}")
    if retrieval.top1 < 0.999:
        raise AssertionError(f"retrieval failed a separable toy problem: {retrieval}")
    return {
        "loss": float(loss.detach()),
        "toy_purity": float(cluster["purity"]),
        "toy_perfect_blocks": int(cluster["perfect_blocks"]),
        "toy_top1": retrieval.top1,
    }


__all__ = [
    "BlockSiamese",
    "CAPACITY",
    "RetrievalMetrics",
    "balanced_spherical_kmeans",
    "clustering_metrics",
    "same_block_retrieval_metrics",
    "sibling_supervised_contrastive_loss",
    "smoke",
]
