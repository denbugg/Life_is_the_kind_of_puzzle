"""Train a hierarchical direct-neighbour directional model on affinity pairs.

For an ordered candidate pair ``(i, j)``, the only positive labels are the
four cardinal clean-grid offsets ``(-1,0)``, ``(+1,0)``, ``(0,-1)``, and
``(0,+1)``.  Diagonals, radius-two/radius-three pairs, and far pairs all map
to ``NON_DIRECT``.  The default candidate graph is the de-duplicated union of
the affinity checkpoints that worked best in the locality audit:

    affinity_r1_1200_best.pt + affinity_r3_1000_best.pt, top-64 each.

The graph is frozen and mined exactly once per input bag.  Training samples a
balanced direct/non-direct minibatch from that hard candidate distribution;
within direct examples the four directions receive equal quotas (equivalent to
inverse-frequency class weighting).  The model is trained hierarchically:

* binary CE on ``[non_direct_logit, logsumexp(four_direct_logits)]``;
* conditional four-way CE only on true direct pairs.

This prevents a flat five-way classifier from collapsing toward the roughly
97% non-direct candidate class.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from direct_pose import (
    DIRECT_CLASS_COUNT,
    DIRECT_OFFSETS,
    NON_DIRECT_CLASS,
    NUM_CLASSES,
    DirectPoseNet,
    class_offsets_metadata,
    count_params,
    hierarchical_predictions,
    inverse_classes,
    offsets_to_direct_classes,
)
from imgio import train_val_split
# Candidate mining is deliberately shared with the offset-pose experiment so
# this gate sees the exact same frozen, de-duplicated affinity union.
from train_offset_pose import (
    checkpoint_sha256,
    load_frozen_affinity,
    make_loader,
    mine_affinity_candidates,
)


def _autocast(device: torch.device):
    """Use fp16 only where CUDA is available, keeping CPU behavior simple."""
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def _clean_rows_cols(perm: Tensor) -> tuple[Tensor, Tensor]:
    """Convert input-tile -> original-cell labels into exact grid coordinates."""
    if perm.ndim != 2 or perm.shape[1] != NFRAG:
        raise ValueError(f"perm must have shape (B,{NFRAG}), got {tuple(perm.shape)}")
    if torch.any(perm < 0) or torch.any(perm >= NFRAG):
        raise ValueError("perm contains an invalid clean-grid cell")
    cells = perm.long()
    return (
        torch.div(cells, GRID, rounding_mode="floor"),
        torch.remainder(cells, GRID),
    )


def candidate_direct_labels(perm: Tensor, candidates: Tensor) -> Tensor:
    """Label affinity candidates with exact cardinal direction or non-direct.

    Args:
        perm: ``(B,576)`` synthetic input-tile -> clean-grid-cell mapping.
        candidates: ``(B,576,K)`` ordered affinity neighbours for each anchor.
    """
    if candidates.ndim != 3 or candidates.shape[:2] != (perm.shape[0], NFRAG):
        raise ValueError(
            f"candidates must have shape (B,{NFRAG},K) matching perm, got {tuple(candidates.shape)}"
        )
    if torch.any(candidates < 0) or torch.any(candidates >= NFRAG):
        raise ValueError("candidate indices are outside the tile bag")
    rows, cols = _clean_rows_cols(perm)
    target_cells = perm.gather(1, candidates.reshape(perm.shape[0], -1)).reshape_as(candidates)
    target_rows = torch.div(target_cells, GRID, rounding_mode="floor")
    target_cols = torch.remainder(target_cells, GRID)
    return offsets_to_direct_classes(
        target_rows - rows.unsqueeze(-1),
        target_cols - cols.unsqueeze(-1),
    )


def _draw_indices(
    population: Tensor,
    number: int,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    """Draw row indices, using replacement only when a class is undersized."""
    if number <= 0:
        return population.new_empty((0,), dtype=torch.long)
    if population.numel() == 0:
        return population.new_empty((0,), dtype=torch.long)
    if number > population.numel():
        choice = torch.randint(
            population.numel(), (number,), device=population.device, generator=generator
        )
    else:
        choice = torch.randperm(population.numel(), device=population.device, generator=generator)[:number]
    return population[choice]


def _sample_balanced_direct_rows(
    direct_rows: Tensor,
    flat_labels: Tensor,
    number: int,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample direct rows with equal per-direction quotas where possible.

    Equal quotas are a stronger form of inverse-frequency weighting: a rare
    cardinal direction is never drowned out by an over-represented one.  If a
    tiny candidate graph happens to omit a direction, its quota is distributed
    uniformly across directions that are present rather than failing the run.
    """
    if number <= 0 or direct_rows.numel() == 0:
        return direct_rows.new_empty((0,), dtype=torch.long)
    grouped = [
        direct_rows[flat_labels[direct_rows].eq(direction)]
        for direction in range(DIRECT_CLASS_COUNT)
    ]
    active = [direction for direction, rows in enumerate(grouped) if rows.numel()]
    if not active:
        return direct_rows.new_empty((0,), dtype=torch.long)
    base, remainder = divmod(number, len(active))
    quotas = {direction: base for direction in active}
    if remainder:
        order = torch.randperm(len(active), device=direct_rows.device, generator=generator)
        for position in order[:remainder].tolist():
            quotas[active[position]] += 1
    pieces = [
        _draw_indices(grouped[direction], quotas[direction], generator=generator)
        for direction in active
        if quotas[direction]
    ]
    chosen = torch.cat(pieces) if pieces else direct_rows.new_empty((0,), dtype=torch.long)
    if chosen.numel() != number:
        raise AssertionError(f"sampled {chosen.numel()} direct rows, expected {number}")
    return chosen


def sample_candidate_pairs(
    candidates: Tensor,
    labels: Tensor,
    *,
    valid: Tensor | None = None,
    pairs_per_image: int,
    direct_fraction: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample binary-balanced, direction-balanced candidates for training.

    Sampling happens only from the affinity candidate lists, not the full
    576x575 universe.  With the default ``direct_fraction=0.5`` it turns the
    roughly 3% direct pool into an exactly balanced binary learning batch.

    Returns ``(image_ids, anchor_indices, target_indices, labels)`` with
    exactly ``B * pairs_per_image`` entries unless a malformed graph contains
    no valid candidate rows at all.
    """
    if candidates.ndim != 3 or labels.shape != candidates.shape:
        raise ValueError("candidates and labels must have equal (B,576,K) shapes")
    if candidates.shape[1] != NFRAG:
        raise ValueError(f"candidate axis must have {NFRAG} anchors")
    if pairs_per_image < 1:
        raise ValueError("pairs_per_image must be positive")
    if not 0.0 <= direct_fraction <= 1.0:
        raise ValueError("direct_fraction must lie in [0,1]")
    if valid is None:
        valid = torch.ones_like(candidates, dtype=torch.bool)
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask aligned with candidates")

    batch, _, candidate_k = candidates.shape
    anchors_template = torch.arange(NFRAG, device=candidates.device).view(NFRAG, 1)
    anchors_template = anchors_template.expand(-1, candidate_k).reshape(-1)
    requested_direct = int(round(pairs_per_image * direct_fraction))
    pieces: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []

    for image in range(batch):
        flat_labels = labels[image].reshape(-1).long()
        flat_targets = candidates[image].reshape(-1).long()
        flat_valid = valid[image].reshape(-1)
        direct_rows = torch.nonzero(
            flat_valid & flat_labels.ne(NON_DIRECT_CLASS), as_tuple=False
        ).flatten()
        non_direct_rows = torch.nonzero(
            flat_valid & flat_labels.eq(NON_DIRECT_CLASS), as_tuple=False
        ).flatten()

        direct_count = requested_direct if direct_rows.numel() else 0
        non_direct_count = pairs_per_image - direct_count
        # Preserve batch size even for an unexpectedly pathological top-K.  In
        # normal runs both populations are plentiful; this branch is a guard,
        # not a replacement for the intended balanced sampler.
        if non_direct_count and not non_direct_rows.numel():
            direct_count = pairs_per_image
            non_direct_count = 0
        if not direct_count and not non_direct_count:
            raise RuntimeError("affinity candidate graph has no selectable rows")

        chosen_parts: list[Tensor] = []
        if direct_count:
            chosen_parts.append(
                _sample_balanced_direct_rows(
                    direct_rows, flat_labels, direct_count, generator=generator
                )
            )
        if non_direct_count:
            chosen_parts.append(
                _draw_indices(non_direct_rows, non_direct_count, generator=generator)
            )
        chosen = torch.cat(chosen_parts)
        order = torch.randperm(chosen.numel(), device=chosen.device, generator=generator)
        chosen = chosen[order]
        pieces.append(
            (
                torch.full((chosen.numel(),), image, device=candidates.device, dtype=torch.long),
                anchors_template[chosen],
                flat_targets[chosen],
                flat_labels[chosen],
            )
        )

    image_ids, anchors, targets, sampled_labels = (
        torch.cat([piece[index] for piece in pieces], dim=0) for index in range(4)
    )
    expected = batch * pairs_per_image
    if image_ids.numel() != expected:
        raise AssertionError(f"sampled {image_ids.numel()} pairs, expected {expected}")
    return image_ids, anchors, targets, sampled_labels


def _hierarchical_loss_components(
    logits: Tensor,
    labels: Tensor,
    *,
    non_direct_weight: float,
) -> dict[str, Tensor]:
    """Return binary direct/non-direct and conditional-direction loss parts."""
    if logits.shape[:-1] != labels.shape or logits.shape[-1] != NUM_CLASSES:
        raise ValueError("logits must align with labels and end in 5 classes")
    if non_direct_weight <= 0.0:
        raise ValueError("non_direct_weight must be positive")
    flat_logits = logits.float().reshape(-1, NUM_CLASSES)
    flat_labels = labels.long().reshape(-1)
    direct = flat_labels.ne(NON_DIRECT_CLASS)
    binary_logits = torch.stack(
        (flat_logits[:, NON_DIRECT_CLASS], torch.logsumexp(flat_logits[:, :DIRECT_CLASS_COUNT], dim=-1)),
        dim=-1,
    )
    binary_per_pair = F.cross_entropy(binary_logits, direct.long(), reduction="none")
    binary_weights = torch.where(
        direct,
        torch.ones_like(binary_per_pair),
        torch.full_like(binary_per_pair, float(non_direct_weight)),
    )
    binary_numerator = (binary_per_pair * binary_weights).sum()
    binary_denominator = binary_weights.sum().clamp_min(1.0)
    if torch.any(direct):
        direction_numerator = F.cross_entropy(
            flat_logits[direct, :DIRECT_CLASS_COUNT], flat_labels[direct], reduction="sum"
        )
    else:
        # Preserve a differentiable zero for rare all-non-direct minibatches.
        direction_numerator = flat_logits.sum() * 0.0
    return {
        "binary_numerator": binary_numerator,
        "binary_denominator": binary_denominator,
        "direction_numerator": direction_numerator,
        "direct_count": direct.sum().to(binary_per_pair.dtype),
    }


def hierarchical_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    non_direct_weight: float,
    direction_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the intended aggregate-binary plus conditional-direction loss."""
    if direction_weight < 0.0:
        raise ValueError("direction_weight must be non-negative")
    parts = _hierarchical_loss_components(
        logits, labels, non_direct_weight=non_direct_weight
    )
    binary_loss = parts["binary_numerator"] / parts["binary_denominator"]
    direction_loss = parts["direction_numerator"] / parts["direct_count"].clamp_min(1.0)
    return binary_loss + float(direction_weight) * direction_loss, {
        "binary_loss": binary_loss,
        "direction_loss": direction_loss,
        "direct_count": parts["direct_count"],
    }


def _flatten_candidate_indices(candidates: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return flat image/anchor/target indices without materialising pixels."""
    batch, count, candidate_k = candidates.shape
    anchors = torch.arange(count, device=candidates.device).view(1, count, 1)
    anchors = anchors.expand(batch, -1, candidate_k).reshape(-1)
    image_ids = torch.arange(batch, device=candidates.device).view(batch, 1, 1)
    image_ids = image_ids.expand(-1, count, candidate_k).reshape(-1)
    return image_ids, anchors, candidates.reshape(-1).long(), valid.reshape(-1)


@torch.no_grad()
def score_candidate_graph(
    model: DirectPoseNet,
    tiles: Tensor,
    candidates: Tensor,
    *,
    valid: Tensor | None = None,
    pair_batch: int,
    device: torch.device,
) -> Tensor:
    """Score all valid affinity candidates in chunks, never all pair pixels."""
    if pair_batch < 1:
        raise ValueError("pair_batch must be positive")
    if valid is None:
        valid = torch.ones_like(candidates, dtype=torch.bool)
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask aligned with candidates")
    image_ids, anchors, targets, flat_valid = _flatten_candidate_indices(candidates, valid)
    image_ids = image_ids[flat_valid]
    anchors = anchors[flat_valid]
    targets = targets[flat_valid]
    if not image_ids.numel():
        raise RuntimeError("candidate graph contains no valid directed pairs")
    chunks: list[Tensor] = []
    for start in range(0, image_ids.numel(), pair_batch):
        stop = min(start + pair_batch, image_ids.numel())
        with _autocast(device):
            chunks.append(
                model(
                    tiles[image_ids[start:stop], anchors[start:stop]],
                    tiles[image_ids[start:stop], targets[start:stop]],
                ).float()
            )
    scored = torch.cat(chunks, dim=0)
    full = scored.new_zeros((candidates.numel(), NUM_CLASSES))
    full[flat_valid] = scored
    return full.reshape(*candidates.shape, NUM_CLASSES)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def candidate_metric_sums(
    logits: Tensor,
    candidates: Tensor,
    labels: Tensor,
    *,
    valid: Tensor | None = None,
    direct_threshold: float = 0.5,
    non_direct_weight: float = 1.0,
) -> dict[str, float]:
    """Return additive exact-synthetic direct and reciprocal metric counts."""
    if logits.shape[:-1] != labels.shape or logits.shape[-1] != NUM_CLASSES:
        raise ValueError("logits must be (B,576,K,5) aligned with labels")
    if candidates.shape != labels.shape:
        raise ValueError("candidates and labels must have equal shapes")
    if valid is None:
        valid = torch.ones_like(candidates, dtype=torch.bool)
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask aligned with candidates")

    decoded = hierarchical_predictions(logits, direct_threshold=direct_threshold)
    prediction = decoded["classes"]
    conditional_direction = decoded["conditional_direction_class"]
    direct = valid & labels.ne(NON_DIRECT_CLASS)
    non_direct = valid & labels.eq(NON_DIRECT_CLASS)
    predicted_direct = valid & decoded["predicted_direct"]
    detected_direct = predicted_direct & direct
    direct_count = float(direct.sum())
    non_direct_count = float(non_direct.sum())
    pair_count = float(valid.sum())
    sums: dict[str, float] = {
        "pairs": pair_count,
        "direct_pairs": direct_count,
        "non_direct_pairs": non_direct_count,
        "detected_direct": float(detected_direct.sum()),
        "predicted_direct": float(predicted_direct.sum()),
        "detected_non_direct": float((valid & ~decoded["predicted_direct"] & non_direct).sum()),
        "predicted_non_direct": float((valid & ~decoded["predicted_direct"]).sum()),
        "conditional_direction_exact": float((conditional_direction.eq(labels) & direct).sum()),
        "end_to_end_exact_direction": float((prediction.eq(labels) & direct).sum()),
        "direct_probability_direct_sum": float(decoded["direct_probability"].masked_select(direct).sum()),
        "direct_probability_non_direct_sum": float(
            decoded["direct_probability"].masked_select(non_direct).sum()
        ),
        "edge_confidence_direct_sum": float(decoded["confidence"].masked_select(direct).sum()),
        "edge_confidence_non_direct_sum": float(
            decoded["confidence"].masked_select(non_direct).sum()
        ),
    }

    # Reciprocals are queried through a compact 576x576 rank map rather than
    # an O(K^2) search.  A predicted reciprocal edge must appear in both
    # candidate lists, be direct in both directions, and carry inverse labels.
    batch, count, candidate_k = candidates.shape
    rank_map = torch.full(
        (batch, count, count), -1, dtype=torch.long, device=candidates.device
    )
    rank_values = torch.arange(candidate_k, device=candidates.device).view(1, 1, candidate_k)
    rank_values = rank_values.expand(batch, count, -1)
    image_ids = torch.arange(batch, device=candidates.device).view(batch, 1, 1)
    image_ids = image_ids.expand(batch, count, candidate_k)
    anchor_ids = torch.arange(count, device=candidates.device).view(1, count, 1)
    anchor_ids = anchor_ids.expand(batch, count, candidate_k)
    rank_map[image_ids[valid], anchor_ids[valid], candidates.long()[valid]] = rank_values[valid]
    reverse_rank = rank_map[image_ids, candidates.long(), anchor_ids]
    mutual = valid & reverse_rank.ge(0)
    sums["mutual_pairs"] = float(mutual.sum())
    if torch.any(mutual):
        reverse_prediction = prediction[
            image_ids[mutual], candidates[mutual].long(), reverse_rank[mutual]
        ]
        reverse_labels = labels[
            image_ids[mutual], candidates[mutual].long(), reverse_rank[mutual]
        ]
        forward_prediction = prediction[mutual]
        forward_labels = labels[mutual]
        forward_direct = forward_prediction.ne(NON_DIRECT_CLASS)
        reverse_direct = reverse_prediction.ne(NON_DIRECT_CLASS)
        predicted_reciprocal = (
            forward_direct
            & reverse_direct
            & reverse_prediction.eq(inverse_classes(forward_prediction))
        )
        true_direct_mutual = forward_labels.ne(NON_DIRECT_CLASS) & reverse_labels.eq(
            inverse_classes(forward_labels)
        )
        correct_reciprocal = (
            predicted_reciprocal
            & true_direct_mutual
            & forward_prediction.eq(forward_labels)
            & reverse_prediction.eq(reverse_labels)
        )
        sums.update(
            {
                "mutual_true_direct": float(true_direct_mutual.sum()),
                "predicted_reciprocal_inverse": float(predicted_reciprocal.sum()),
                "correct_reciprocal_inverse": float(correct_reciprocal.sum()),
            }
        )
    else:
        sums.update(
            {
                "mutual_true_direct": 0.0,
                "predicted_reciprocal_inverse": 0.0,
                "correct_reciprocal_inverse": 0.0,
            }
        )

    components = _hierarchical_loss_components(
        logits[valid], labels[valid], non_direct_weight=non_direct_weight
    )
    sums["binary_loss_numerator"] = float(components["binary_numerator"])
    sums["binary_loss_denominator"] = float(components["binary_denominator"])
    sums["direction_loss_numerator"] = float(components["direction_numerator"])
    return sums


def finalize_candidate_metrics(
    sums: Mapping[str, float], *, direction_weight: float
) -> dict[str, float]:
    """Turn additive direct-edge statistics into held-out metrics."""
    if direction_weight < 0.0:
        raise ValueError("direction_weight must be non-negative")
    pairs = sums["pairs"]
    direct = sums["direct_pairs"]
    non_direct = sums["non_direct_pairs"]
    direct_precision = _safe_ratio(sums["detected_direct"], sums["predicted_direct"])
    direct_recall = _safe_ratio(sums["detected_direct"], direct)
    direct_f1 = _safe_ratio(2.0 * direct_precision * direct_recall, direct_precision + direct_recall)
    binary_loss = _safe_ratio(sums["binary_loss_numerator"], sums["binary_loss_denominator"])
    direction_loss = _safe_ratio(sums["direction_loss_numerator"], direct)
    metrics = {
        "candidate_hierarchical_loss": binary_loss + float(direction_weight) * direction_loss,
        "candidate_binary_cross_entropy": binary_loss,
        "candidate_direction_cross_entropy": direction_loss,
        "candidate_direct_fraction": _safe_ratio(direct, pairs),
        "direct_detection_precision": direct_precision,
        "direct_detection_recall": direct_recall,
        "direct_detection_f1": direct_f1,
        "binary_accuracy": _safe_ratio(
            sums["detected_direct"] + sums["detected_non_direct"], pairs
        ),
        # Conditional direction is isolated from threshold calibration.  The
        # end-to-end version below is the metric an actual edge graph sees.
        "conditional_direction_accuracy_direct": _safe_ratio(
            sums["conditional_direction_exact"], direct
        ),
        "exact_direction_accuracy_direct": _safe_ratio(
            sums["end_to_end_exact_direction"], direct
        ),
        "non_direct_detection_recall": _safe_ratio(sums["detected_non_direct"], non_direct),
        "mean_direct_probability_direct": _safe_ratio(
            sums["direct_probability_direct_sum"], direct
        ),
        "mean_direct_probability_non_direct": _safe_ratio(
            sums["direct_probability_non_direct_sum"], non_direct
        ),
        "mean_edge_confidence_direct": _safe_ratio(sums["edge_confidence_direct_sum"], direct),
        "mean_edge_confidence_non_direct": _safe_ratio(
            sums["edge_confidence_non_direct_sum"], non_direct
        ),
        "mutual_candidate_fraction": _safe_ratio(sums["mutual_pairs"], pairs),
        "mutual_direct_candidate_coverage": _safe_ratio(sums["mutual_true_direct"], direct),
        "reciprocal_inverse_precision": _safe_ratio(
            sums["correct_reciprocal_inverse"], sums["predicted_reciprocal_inverse"]
        ),
        "reciprocal_inverse_coverage_mutual_direct": _safe_ratio(
            sums["correct_reciprocal_inverse"], sums["mutual_true_direct"]
        ),
        "reciprocal_inverse_coverage_all_direct": _safe_ratio(
            sums["correct_reciprocal_inverse"], direct
        ),
    }
    return metrics


def _ranking_metrics(scores: Tensor, positive: Tensor, *, prefix: str) -> dict[str, float]:
    """Compute rank-only AP/AUC diagnostics without a sklearn dependency."""
    flat_scores = scores.detach().float().reshape(-1).cpu()
    flat_positive = positive.detach().bool().reshape(-1).cpu()
    total = int(flat_positive.numel())
    positives = int(flat_positive.sum())
    negatives = total - positives
    if not total:
        return {
            f"{prefix}_average_precision": 0.0,
            f"{prefix}_roc_auc": 0.0,
            f"{prefix}_precision_at_prevalence": 0.0,
        }
    order_desc = torch.argsort(flat_scores, descending=True)
    sorted_positive = flat_positive[order_desc].to(torch.float32)
    ranks = torch.arange(1, total + 1, dtype=torch.float32)
    precision_at_rank = sorted_positive.cumsum(dim=0) / ranks
    average_precision = (
        float((precision_at_rank * sorted_positive).sum() / positives) if positives else 0.0
    )
    top_k = max(1, positives)
    precision_at_prevalence = float(sorted_positive[:top_k].mean())
    if positives and negatives:
        order_asc = torch.argsort(flat_scores, descending=False)
        positive_ranks = torch.nonzero(flat_positive[order_asc], as_tuple=False).flatten().to(torch.float32) + 1.0
        # Scores are continuous in normal operation, so exact ties are rare.
        # This rank form is memory-safe even for a multi-image candidate graph.
        roc_auc = float(
            (positive_ranks.sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
        )
    else:
        roc_auc = 0.0
    return {
        f"{prefix}_average_precision": average_precision,
        f"{prefix}_roc_auc": roc_auc,
        f"{prefix}_precision_at_prevalence": precision_at_prevalence,
    }


@torch.no_grad()
def evaluate(
    model: DirectPoseNet,
    affinity: nn.Module,
    loader: DataLoader,
    *,
    candidate_k: int,
    max_images: int,
    pair_batch: int,
    device: torch.device,
    affinity_secondary: nn.Module | None = None,
    direct_threshold: float = 0.5,
    non_direct_weight: float = 1.0,
    direction_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluate exact synthetic candidates, ranking, and reciprocal edges."""
    model_was_training = model.training
    model.eval()
    aggregate: defaultdict[str, float] = defaultdict(float)
    direct_scores: list[Tensor] = []
    direct_targets: list[Tensor] = []
    edge_scores: list[Tensor] = []
    edge_targets: list[Tensor] = []
    seen = 0
    for batch in loader:
        if seen >= max_images:
            break
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("direct-pose evaluation requires CanvasDataset(real_prob=0)")
        take = min(max_images - seen, int(batch["tiles"].shape[0]))
        tiles = batch["tiles"][:take].to(device, non_blocking=True)
        perm = batch["perm"][:take].to(device, non_blocking=True).long()
        candidates, valid = mine_affinity_candidates(
            affinity,
            tiles,
            candidate_k=candidate_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        labels = candidate_direct_labels(perm, candidates)
        logits = score_candidate_graph(
            model, tiles, candidates, valid=valid, pair_batch=pair_batch, device=device
        )
        for key, value in candidate_metric_sums(
            logits,
            candidates,
            labels,
            valid=valid,
            direct_threshold=direct_threshold,
            non_direct_weight=non_direct_weight,
        ).items():
            aggregate[key] += value
        decoded = hierarchical_predictions(logits, direct_threshold=direct_threshold)
        direct_scores.append(decoded["direct_probability"].masked_select(valid).cpu())
        direct_targets.append(labels.ne(NON_DIRECT_CLASS).masked_select(valid).cpu())
        edge_scores.append(decoded["confidence"].masked_select(valid).cpu())
        edge_targets.append(
            (labels.ne(NON_DIRECT_CLASS) & decoded["conditional_direction_class"].eq(labels))
            .masked_select(valid)
            .cpu()
        )
        seen += take
    if model_was_training:
        model.train()
    if not seen:
        raise RuntimeError("evaluation loader yielded no images")
    metrics = finalize_candidate_metrics(aggregate, direction_weight=direction_weight)
    metrics.update(
        _ranking_metrics(
            torch.cat(direct_scores), torch.cat(direct_targets), prefix="direct_confidence"
        )
    )
    metrics.update(
        _ranking_metrics(
            torch.cat(edge_scores), torch.cat(edge_targets), prefix="edge_confidence"
        )
    )
    metrics["eval_images"] = float(seen)
    return metrics


def _next_batch(
    iterator: Iterable[dict[str, Tensor]], loader: DataLoader
) -> tuple[dict[str, Tensor], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _format(metrics: Mapping[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def save_checkpoint(
    path: str,
    model: DirectPoseNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    args: argparse.Namespace,
    metrics: Mapping[str, float],
    affinity_provenance: list[Mapping[str, Any]],
) -> None:
    """Save an inference-readable checkpoint with graph/protocol provenance."""
    torch.save(
        {
            "format": "direct_pose_v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "args": vars(args),
            "metrics": dict(metrics),
            "model_kwargs": {
                "tile_size": model.tile_size,
                "width": model.width,
                "dropout": model.dropout,
                "num_classes": model.num_classes,
            },
            "grid": GRID,
            "fragment_size": FS,
            "num_classes": NUM_CLASSES,
            "non_direct_class": NON_DIRECT_CLASS,
            "direct_offsets": class_offsets_metadata(),
            "prediction_mode": "hierarchical_direct",
            "hierarchical_objective": {
                "binary": "cross_entropy([non_direct_logit, logsumexp(direct_logits)])",
                "conditional_direction": "cross_entropy(direct_logits) on true direct pairs only",
                "direction_weight": float(args.direction_weight),
                "non_direct_weight": float(args.non_direct_weight),
                "direct_threshold": float(args.direct_threshold),
            },
            "sampling": {
                "pairs_per_image": int(args.pairs_per_image),
                "requested_direct_fraction": float(args.direct_fraction),
                "direct_direction_sampling": "equal cardinal quotas; fallback over present directions",
            },
            "affinity_checkpoint": str(affinity_provenance[0]["path"]),
            "affinity_checkpoint_sha256": str(affinity_provenance[0]["sha256"]),
            "affinity_model_kwargs": dict(affinity_provenance[0]["model_kwargs"]),
            "affinity_checkpoints": [dict(item) for item in affinity_provenance],
            "candidate_mining": {
                "candidate_k_per_encoder": int(args.candidate_k),
                "encoders": len(affinity_provenance),
                "union_deduplicate": len(affinity_provenance) > 1,
                "source": "frozen MacroAffinityNet cosine top-k excluding self",
            },
        },
        path,
    )


def _tiny_smoke(device: torch.device) -> dict[str, float]:
    """Exercise labels, balanced sampling, hierarchy, metrics, and one CPU step."""
    torch.manual_seed(9876)
    anchors = torch.arange(NFRAG, device=device)
    cols = torch.remainder(anchors, GRID)
    # Each interior tile lists both horizontal cardinal neighbours plus one far
    # target.  Border duplicates are masked, giving a valid reciprocal graph.
    right_or_left = torch.where(cols.lt(GRID - 1), anchors + 1, anchors - 1)
    left_or_right = torch.where(cols.gt(0), anchors - 1, anchors + 1)
    far = torch.remainder(anchors + GRID * 12, NFRAG)
    candidates = torch.stack((right_or_left, left_or_right, far), dim=-1).unsqueeze(0)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    border = cols.eq(0) | cols.eq(GRID - 1)
    valid[0, border, 1] = False
    perm = anchors.unsqueeze(0)
    labels = candidate_direct_labels(perm, candidates)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(2468)
    image_ids, source, target, sampled_labels = sample_candidate_pairs(
        candidates,
        labels,
        valid=valid,
        pairs_per_image=16,
        direct_fraction=0.5,
        generator=generator,
    )
    if int(sampled_labels.ne(NON_DIRECT_CLASS).sum()) != 8:
        raise AssertionError("tiny sampler did not preserve direct/non-direct balance")
    model = DirectPoseNet(tile_size=FS, width=8, dropout=0.0).to(device)
    tiles = torch.rand(1, NFRAG, 3, FS, FS, device=device)
    logits = model(tiles[image_ids, source], tiles[image_ids, target])
    loss, _ = hierarchical_loss(
        logits, sampled_labels, non_direct_weight=1.0, direction_weight=1.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Use deliberately perfect logits for deterministic metric/reciprocity
    # checks rather than treating an untrained CNN as a quality test.
    exact_logits = torch.full((*labels.shape, NUM_CLASSES), -6.0, device=device)
    exact_logits.scatter_(-1, labels.unsqueeze(-1), 6.0)
    metrics = finalize_candidate_metrics(
        candidate_metric_sums(
            exact_logits,
            candidates,
            labels,
            valid=valid,
            direct_threshold=0.5,
            non_direct_weight=1.0,
        ),
        direction_weight=1.0,
    )
    if metrics["direct_detection_f1"] < 0.999 or metrics["reciprocal_inverse_precision"] < 0.999:
        raise AssertionError(f"tiny metric guard failed: {metrics}")
    return {
        "sample_loss": float(loss.detach()),
        "direct_f1": metrics["direct_detection_f1"],
        "reciprocal_precision": metrics["reciprocal_inverse_precision"],
        "reciprocal_coverage": metrics["reciprocal_inverse_coverage_all_direct"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--affinity_ckpt",
        default=os.path.join(workspace, "artifacts", "macro_affinity", "affinity_r1_1200_best.pt"),
        help="primary frozen MacroAffinityNet checkpoint used to mine top-K candidates",
    )
    parser.add_argument(
        "--affinity_ckpt2",
        default=os.path.join(workspace, "artifacts", "macro_affinity", "affinity_r3_1000_best.pt"),
        help="optional secondary frozen affinity checkpoint; pass an empty string to disable the union",
    )
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--bs", type=int, default=2, help="full synthetic puzzle bags per optimization step")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train_n", type=int, default=0, help="0 uses all non-validation targets")
    parser.add_argument("--candidate_k", type=int, default=64, help="frozen affinity neighbours per encoder")
    parser.add_argument("--pairs_per_image", type=int, default=1024, help="sampled directed candidates per puzzle")
    parser.add_argument(
        "--direct_fraction",
        type=float,
        default=0.50,
        help="requested direct share of a sampled minibatch; 0.5 is binary-balanced",
    )
    parser.add_argument(
        "--non_direct_weight",
        type=float,
        default=1.0,
        help="optional non-direct example weight in aggregate binary CE",
    )
    parser.add_argument(
        "--direction_weight",
        type=float,
        default=1.0,
        help="weight on conditional four-way directional CE for true direct pairs",
    )
    parser.add_argument(
        "--direct_threshold",
        type=float,
        default=0.5,
        help="P(direct) threshold used by exact held-out candidate decisions",
    )
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--eval_n", type=int, default=12)
    parser.add_argument("--eval_bs", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--eval_pair_batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="direct_pose")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(workspace, "artifacts", "direct_pose"),
        help="workspace-local output checkpoint directory",
    )
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument(
        "--tiny_smoke",
        action="store_true",
        help="run a data-free labels/loss/metric/optimizer smoke test and exit",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.tiny_smoke:
        print(f"[direct-pose tiny smoke] device={device} {_format(_tiny_smoke(device))}", flush=True)
        return

    if args.steps < 1 or args.bs < 1 or args.eval_n < 1 or args.eval_bs < 1:
        parser.error("--steps, --bs, --eval_n, and --eval_bs must be positive")
    if args.workers < 0 or args.train_n < 0:
        parser.error("--workers and --train_n must be non-negative")
    if not 1 <= args.candidate_k < NFRAG:
        parser.error(f"--candidate_k must lie in [1,{NFRAG - 1}]")
    if args.pairs_per_image < 1 or args.eval_pair_batch < 1:
        parser.error("--pairs_per_image and --eval_pair_batch must be positive")
    if not 0.0 <= args.direct_fraction <= 1.0:
        parser.error("--direct_fraction must lie in [0,1]")
    if (
        args.non_direct_weight <= 0.0
        or args.direction_weight < 0.0
        or args.width < 4
        or args.lr <= 0.0
        or args.weight_decay < 0.0
    ):
        parser.error("invalid --non_direct_weight/--direction_weight/--width/--lr/--weight_decay value")
    if not 0.0 <= args.direct_threshold <= 1.0:
        parser.error("--direct_threshold must lie in [0,1]")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must lie in [0,1)")
    if args.eval_every < 1:
        parser.error("--eval_every must be positive")
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    affinity_path = os.path.abspath(args.affinity_ckpt)
    affinity_path2 = os.path.abspath(args.affinity_ckpt2) if args.affinity_ckpt2 else None
    print(
        f"device={device} classes={NUM_CLASSES} direct_offsets={DIRECT_OFFSETS} "
        f"candidate_k={args.candidate_k}/encoder sampled_pairs/image={args.pairs_per_image} "
        f"direct_fraction={args.direct_fraction:g} hierarchical(direction_weight={args.direction_weight:g}, "
        f"threshold={args.direct_threshold:g})",
        flush=True,
    )
    affinity, affinity_metadata, affinity_kwargs = load_frozen_affinity(affinity_path, device)
    del affinity_metadata
    affinity_provenance: list[Mapping[str, Any]] = [
        {
            "path": affinity_path,
            "sha256": checkpoint_sha256(affinity_path),
            "model_kwargs": dict(affinity_kwargs),
        }
    ]
    affinity_secondary: nn.Module | None = None
    if affinity_path2:
        if os.path.normcase(affinity_path) == os.path.normcase(affinity_path2):
            parser.error("--affinity_ckpt2 must differ from --affinity_ckpt")
        affinity_secondary, secondary_metadata, secondary_kwargs = load_frozen_affinity(
            affinity_path2, device
        )
        del secondary_metadata
        affinity_provenance.append(
            {
                "path": affinity_path2,
                "sha256": checkpoint_sha256(affinity_path2),
                "model_kwargs": dict(secondary_kwargs),
            }
        )
    for ordinal, provenance in enumerate(affinity_provenance, start=1):
        print(
            f"frozen affinity[{ordinal}]={provenance['path']} "
            f"sha256={str(provenance['sha256'])[:12]} kwargs={provenance['model_kwargs']}",
            flush=True,
        )
    if affinity_secondary is not None:
        print("candidate graph=deduplicated union of each encoder's top-K lists", flush=True)

    train_names, validation_names = train_val_split()
    if args.train_n:
        train_names = train_names[: args.train_n]
    if not train_names or not validation_names:
        raise RuntimeError("train/validation split is empty")
    train_dataset = CanvasDataset(train_names, real_prob=0.0, seed=args.seed)
    validation_dataset = CanvasDataset(validation_names, real_prob=0.0, seed=args.seed + 10_000)
    train_loader = make_loader(train_dataset, args.bs, args.workers, shuffle=True, device=device)
    validation_loader = make_loader(
        validation_dataset, args.eval_bs, min(args.workers, 2), shuffle=False, device=device
    )

    model = DirectPoseNet(tile_size=FS, width=args.width, dropout=args.dropout).to(device)
    print(f"DirectPoseNet params={count_params(model):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed + 24680)
    best = -float("inf")
    started = time.time()
    iterator = iter(train_loader)

    for step in range(1, args.steps + 1):
        batch, iterator = _next_batch(iterator, train_loader)
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("direct-pose training requires CanvasDataset(real_prob=0)")
        tiles = batch["tiles"].to(device, non_blocking=True)
        perm = batch["perm"].to(device, non_blocking=True).long()
        candidates, valid = mine_affinity_candidates(
            affinity,
            tiles,
            candidate_k=args.candidate_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        labels = candidate_direct_labels(perm, candidates)
        image_ids, anchors, targets, sampled_labels = sample_candidate_pairs(
            candidates,
            labels,
            valid=valid,
            pairs_per_image=args.pairs_per_image,
            direct_fraction=args.direct_fraction,
            generator=generator,
        )
        with _autocast(device):
            logits = model(tiles[image_ids, anchors], tiles[image_ids, targets])
        loss, loss_parts = hierarchical_loss(
            logits,
            sampled_labels,
            non_direct_weight=args.non_direct_weight,
            direction_weight=args.direction_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % 50 == 0:
            elapsed = time.time() - started
            direct_mask = sampled_labels.ne(NON_DIRECT_CLASS)
            decoded = hierarchical_predictions(logits.detach(), direct_threshold=args.direct_threshold)
            conditional_accuracy = (
                decoded["conditional_direction_class"][direct_mask]
                .eq(sampled_labels[direct_mask])
                .float()
            )
            direction_counts = torch.bincount(
                sampled_labels[direct_mask], minlength=DIRECT_CLASS_COUNT
            ).tolist()
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"bin={float(loss_parts['binary_loss'].detach()):.4f} "
                f"dir={float(loss_parts['direction_loss'].detach()):.4f} "
                f"sample_direct={float(direct_mask.float().mean()):.3f} "
                f"sample_dirs={direction_counts} "
                f"sample_cond_dir={float(conditional_accuracy.mean()) if conditional_accuracy.numel() else 0.0:.3f} "
                f"sample_direct_recall={float(decoded['predicted_direct'][direct_mask].float().mean()) if torch.any(direct_mask) else 0.0:.3f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} {elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                affinity,
                validation_loader,
                candidate_k=args.candidate_k,
                max_images=args.eval_n,
                pair_batch=args.eval_pair_batch,
                device=device,
                affinity_secondary=affinity_secondary,
                direct_threshold=args.direct_threshold,
                non_direct_weight=args.non_direct_weight,
                direction_weight=args.direction_weight,
            )
            print(f"[SYN direct-pose held-out] step={step} {_format(metrics)}", flush=True)
            last_path = os.path.join(args.out_dir, f"{args.tag}_last.pt")
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                step=step,
                args=args,
                metrics=metrics,
                affinity_provenance=affinity_provenance,
            )
            # This is the strict graph-ready gate: correct directed edges that
            # survive reciprocal inverse validation, divided by all available
            # true direct candidate rows (not just the mutual subset).
            selection = metrics["reciprocal_inverse_coverage_all_direct"]
            if selection > best:
                best = selection
                best_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    args=args,
                    metrics=metrics,
                    affinity_provenance=affinity_provenance,
                )
                print(f"saved best reciprocal_inverse_coverage_all_direct={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
