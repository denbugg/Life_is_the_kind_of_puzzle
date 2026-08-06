"""Train a permutation-invariant 4x4 macro-partition model.

The input is an unordered, synthetically corrupted bag of all 576 fragments.
``MacroPartitionNet`` predicts 36 *anonymous* partition slots, rather than a
named 6x6 coordinate.  Since those slots have no stable semantic identity, a
normal cross-entropy against macrocell IDs is invalid: slot zero may represent
a different true macrocell for every image and every optimization step.

For each image this trainer therefore:

1. aggregates each true group's member log-probabilities for every predicted
   slot;
2. uses Hungarian matching to map the 36 anonymous slot columns to the 36
   true groups; and
3. applies cross-entropy to the resulting matched slot target per tile.

A soft capacity penalty keeps every slot close to its required sixteen tiles.
All supervision uses freshly generated synthetic examples from
``CanvasDataset(real_prob=0)``; no recovered/noisy permutation cache enters
this experiment.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from imgio import train_val_split
from partition_model import MacroPartitionNet, count_params


MACRO_SIDE = 4
if GRID % MACRO_SIDE:
    raise RuntimeError(f"fine grid {GRID} must be divisible by macro side {MACRO_SIDE}")
MACRO_GRID = GRID // MACRO_SIDE
MACRO_CELLS = MACRO_GRID * MACRO_GRID
MACRO_CAPACITY = MACRO_SIDE * MACRO_SIDE
if NFRAG != MACRO_CELLS * MACRO_CAPACITY:
    raise RuntimeError(
        f"expected {MACRO_CELLS} macro groups x {MACRO_CAPACITY} tiles, got {NFRAG}"
    )


def _autocast(device: torch.device):
    """Enable fp16 only where it is useful and safe to do so."""
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def make_loader(
    dataset: CanvasDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create a loader without relaxing the exact synthetic-data policy."""
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": shuffle and len(dataset) >= batch_size,
    }
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def macro_group_labels(perm: Tensor) -> Tensor:
    """Map a synthetic input-tile permutation to its non-overlapping 4x4 group.

    ``perm[b, tile]`` is the original row-major clean-cell index of the input
    tile.  The desired target is exactly ``(row // 4, col // 4)`` in row-major
    six-by-six macrocell order.  Each label must occur exactly sixteen times
    for a valid shuffled 24x24 puzzle.
    """
    if perm.ndim < 1 or perm.shape[-1] != NFRAG:
        raise ValueError(f"perm must end in {NFRAG} cells, got {tuple(perm.shape)}")
    if torch.any(perm < 0) or torch.any(perm >= NFRAG):
        raise ValueError("perm contains a clean-cell index outside the fine grid")
    perm = perm.long()
    row = torch.div(perm, GRID, rounding_mode="floor")
    col = torch.remainder(perm, GRID)
    return (
        torch.div(row, MACRO_SIDE, rounding_mode="floor") * MACRO_GRID
        + torch.div(col, MACRO_SIDE, rounding_mode="floor")
    ).long()


def require_assignment_logits(output: Mapping[str, Tensor]) -> Tensor:
    """Validate the narrow ``MacroPartitionNet`` forward contract used here."""
    try:
        logits = output["assignment_logits"]
    except KeyError as exc:
        raise KeyError("MacroPartitionNet forward must return assignment_logits") from exc
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (NFRAG, MACRO_CELLS):
        raise ValueError(
            f"expected assignment_logits (B,{NFRAG},{MACRO_CELLS}), got {tuple(logits.shape)}"
        )
    if not torch.is_floating_point(logits):
        raise TypeError(f"assignment_logits must be floating point, got {logits.dtype}")
    return logits


def aggregate_group_log_probs(logits: Tensor, target_groups: Tensor) -> Tensor:
    """Return true-group x anonymous-slot evidence for every image.

    The value for ``[group, slot]`` is the sum of the log-probabilities that
    the sixteen true members of ``group`` belong to ``slot``.  This is a
    principled per-image matching score: it rewards a slot only when it gives
    coherent probability mass to the entire true group, rather than merely to
    one visually easy member.
    """
    _check_logits_and_labels(logits, target_groups)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    membership = F.one_hot(target_groups.long(), num_classes=MACRO_CELLS).to(log_probs.dtype)
    # (B, group, tile) @ (B, tile, slot) -> (B, group, slot).
    return membership.transpose(1, 2) @ log_probs


def hungarian_group_slot_matching(group_slot_scores: Tensor) -> Tensor:
    """Find a true-group -> anonymous-slot permutation independently per image.

    The assignment is deliberately detached: Hungarian is a discrete target
    construction step, while the subsequent cross-entropy supplies gradients.
    Returned indices are on the same device as ``group_slot_scores`` and have
    shape ``(B, 36)``; ``result[b, true_group]`` is a predicted slot column.
    """
    if group_slot_scores.ndim != 3 or tuple(group_slot_scores.shape[1:]) != (
        MACRO_CELLS,
        MACRO_CELLS,
    ):
        raise ValueError(
            "group_slot_scores must have shape "
            f"(B,{MACRO_CELLS},{MACRO_CELLS}), got {tuple(group_slot_scores.shape)}"
        )
    scores_np = group_slot_scores.detach().float().cpu().numpy()
    true_to_slot = np.empty((scores_np.shape[0], MACRO_CELLS), dtype=np.int64)
    for image, scores in enumerate(scores_np):
        true_groups, slots = linear_sum_assignment(-scores)
        # scipy presently returns sorted rows, but filling by the explicit row
        # indices protects this invariant if that implementation ever changes.
        true_to_slot[image, true_groups] = slots
    return torch.as_tensor(true_to_slot, dtype=torch.long, device=group_slot_scores.device)


def inverse_group_slot_matching(true_to_slot: Tensor) -> Tensor:
    """Invert a true-group -> slot permutation into a slot -> true-group map."""
    if true_to_slot.ndim != 2 or true_to_slot.shape[1] != MACRO_CELLS:
        raise ValueError(
            f"true_to_slot must have shape (B,{MACRO_CELLS}), got {tuple(true_to_slot.shape)}"
        )
    slots = torch.empty_like(true_to_slot)
    groups = torch.arange(MACRO_CELLS, device=true_to_slot.device).view(1, -1)
    slots.scatter_(1, true_to_slot, groups.expand_as(true_to_slot))
    return slots


def matched_slot_targets(target_groups: Tensor, true_to_slot: Tensor) -> Tensor:
    """Translate true group labels into the anonymous slot targets of one batch."""
    if target_groups.ndim != 2 or target_groups.shape[1] != NFRAG:
        raise ValueError(f"target_groups must have shape (B,{NFRAG}), got {tuple(target_groups.shape)}")
    if true_to_slot.shape != (target_groups.shape[0], MACRO_CELLS):
        raise ValueError("matching batch size must agree with target_groups")
    return true_to_slot.gather(1, target_groups.long())


def _check_logits_and_labels(logits: Tensor, target_groups: Tensor) -> None:
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (NFRAG, MACRO_CELLS):
        raise ValueError(f"expected logits (B,{NFRAG},{MACRO_CELLS}), got {tuple(logits.shape)}")
    if target_groups.shape != logits.shape[:2]:
        raise ValueError(
            f"target_groups must have shape {tuple(logits.shape[:2])}, got {tuple(target_groups.shape)}"
        )
    if torch.any(target_groups < 0) or torch.any(target_groups >= MACRO_CELLS):
        raise ValueError("target_groups has an invalid macrocell index")


def partition_loss(
    logits: Tensor,
    target_groups: Tensor,
    *,
    capacity_weight: float,
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    """Permutation-invariant CE plus expected 16-members-per-slot regularizer.

    ``capacity_loss`` uses the softmax expectation, not hard argmax counts, so
    it remains differentiable.  It is zero exactly when each anonymous slot
    receives total probability mass of sixteen tiles.
    """
    if capacity_weight < 0:
        raise ValueError("capacity_weight must be non-negative")
    with torch.no_grad():
        # The discrete matching only chooses a target permutation.  Keeping its
        # log-softmax/matmul graph would waste activation memory without giving
        # any usable gradient through Hungarian.
        evidence = aggregate_group_log_probs(logits, target_groups)
        true_to_slot = hungarian_group_slot_matching(evidence)
        slot_targets = matched_slot_targets(target_groups, true_to_slot)
    ce = F.cross_entropy(logits.float().reshape(-1, MACRO_CELLS), slot_targets.reshape(-1))
    expected_counts = F.softmax(logits.float(), dim=-1).sum(dim=1)
    capacity = ((expected_counts - MACRO_CAPACITY) / MACRO_CAPACITY).square().mean()
    loss = ce + float(capacity_weight) * capacity
    return loss, {"ce": ce.detach(), "capacity": capacity.detach()}, true_to_slot


def _slot_group_contingency(predicted_slots: Tensor, target_groups: Tensor) -> Tensor:
    """Count hard memberships as ``(B, predicted_slot, true_group)``."""
    if predicted_slots.shape != target_groups.shape:
        raise ValueError("predicted_slots and target_groups must share shape")
    predicted = F.one_hot(predicted_slots.long(), num_classes=MACRO_CELLS).float()
    true = F.one_hot(target_groups.long(), num_classes=MACRO_CELLS).float()
    return predicted.transpose(1, 2) @ true


def _accuracy_after_matching(
    predicted_slots: Tensor,
    target_groups: Tensor,
    true_to_slot: Tensor,
) -> Tensor:
    """Per-image membership accuracy after translating slot IDs to true groups."""
    slot_to_true = inverse_group_slot_matching(true_to_slot)
    translated = slot_to_true.gather(1, predicted_slots.long())
    return (translated == target_groups).float().mean(dim=1)


def _purity_from_contingency(contingency: Tensor) -> tuple[Tensor, Tensor]:
    """Return classical weighted purity and mean purity over non-empty slots."""
    if contingency.ndim != 3 or tuple(contingency.shape[1:]) != (MACRO_CELLS, MACRO_CELLS):
        raise ValueError("contingency must have shape (B,36,36)")
    counts = contingency.sum(dim=-1)
    largest = contingency.max(dim=-1).values
    weighted = largest.sum(dim=-1) / float(NFRAG)
    per_slot = largest / counts.clamp_min(1.0)
    nonempty = counts > 0
    mean_nonempty = (per_slot * nonempty).sum(dim=-1) / nonempty.sum(dim=-1).clamp_min(1)
    return weighted, mean_nonempty


def capacity_hungarian_slots(scores: Tensor) -> Tensor:
    """Assign exactly sixteen tiles to each anonymous slot using Hungarian.

    Repeating each of the 36 slot columns sixteen times turns a capacity-constrained
    partition into an ordinary 576-by-576 linear assignment.  ``scores`` may
    be logits or log-probabilities because the row-wise softmax normalizer is
    constant across a tile's candidate slots.
    """
    if scores.ndim != 3 or tuple(scores.shape[1:]) != (NFRAG, MACRO_CELLS):
        raise ValueError(f"scores must have shape (B,{NFRAG},{MACRO_CELLS})")
    scores_np = scores.detach().float().cpu().numpy()
    result = np.empty((scores_np.shape[0], NFRAG), dtype=np.int64)
    for image, score in enumerate(scores_np):
        virtual_scores = np.repeat(score, MACRO_CAPACITY, axis=1)
        tile_rows, virtual_slots = linear_sum_assignment(-virtual_scores)
        result[image, tile_rows] = virtual_slots // MACRO_CAPACITY
    return torch.as_tensor(result, dtype=torch.long, device=scores.device)


def capacity_hungarian_membership_accuracy(scores: Tensor, target_groups: Tensor) -> list[float]:
    """Best membership accuracy after an exact-16-per-slot Hungarian decode."""
    _check_logits_and_labels(scores, target_groups)
    slots = capacity_hungarian_slots(scores)
    contingency = _slot_group_contingency(slots, target_groups)
    # Standard clustering accuracy: align the anonymous decoded slots using
    # their membership contingency, independently for every image.
    best_true_to_slot = hungarian_group_slot_matching(contingency.transpose(1, 2))
    return _accuracy_after_matching(slots, target_groups, best_true_to_slot).detach().cpu().tolist()


def group_coverage(
    logits: Tensor,
    target_groups: Tensor,
    true_to_slot: Tensor,
    ks: Sequence[int],
) -> dict[str, Tensor]:
    """Measure true-member recall in the top-K tile shortlist of each group.

    Predicted slot columns are first aligned to their true group.  For every
    group, the K highest-scoring tiles for the aligned slot are retained; the
    metric is the fraction of that group's sixteen actual members retained.
    The tensors returned here have one value per image, making validation
    averages insensitive to a short final batch.
    """
    _check_logits_and_labels(logits, target_groups)
    if true_to_slot.shape != (logits.shape[0], MACRO_CELLS):
        raise ValueError("true_to_slot must have one 36-slot matching per image")
    values: dict[str, Tensor] = {}
    requested = tuple(dict.fromkeys(int(k) for k in ks))
    aligned_scores = logits.gather(
        2,
        true_to_slot.unsqueeze(1).expand(-1, NFRAG, -1),
    )
    group_ids = torch.arange(MACRO_CELLS, device=logits.device).view(1, 1, -1)
    labels = target_groups.unsqueeze(-1).expand(-1, -1, MACRO_CELLS)
    for k in requested:
        if not 1 <= k <= NFRAG:
            raise ValueError(f"coverage K must be in [1,{NFRAG}], got {k}")
        top_tiles = aligned_scores.topk(k, dim=1).indices
        labels_at_top = labels.gather(1, top_tiles)
        coverage = (labels_at_top == group_ids).sum(dim=1).float() / MACRO_CAPACITY
        values[f"top{k}_group_coverage"] = coverage.mean(dim=1)
        # A mean can hide one unsalvageable group, which matters to a later
        # within-group solver.  Keep compact tail diagnostics at the native
        # 4x4 candidate-set size as well.
        if k == MACRO_CAPACITY:
            values[f"top{k}_group_coverage_min"] = coverage.min(dim=1).values
            values[f"top{k}_group_coverage_p10"] = torch.quantile(coverage, 0.10, dim=1)
    return values


@torch.no_grad()
def partition_metrics(
    logits: Tensor,
    target_groups: Tensor,
    *,
    coverage_ks: Sequence[int],
    include_capacity_hungarian: bool = True,
    matched_true_to_slot: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return per-image diagnostics for anonymous macro partitions.

    ``aligned_partition_acc`` follows the log-probability matching used to
    train the model.  ``best_partition_acc`` is the usual clustering score
    obtained by optimally matching the hard partition contingency itself;
    reporting both makes a bad matching objective visible instead of silently
    conflating it with classifier quality.
    """
    _check_logits_and_labels(logits, target_groups)
    if matched_true_to_slot is None:
        matched_true_to_slot = hungarian_group_slot_matching(
            aggregate_group_log_probs(logits, target_groups)
        )
    if matched_true_to_slot.shape != (logits.shape[0], MACRO_CELLS):
        raise ValueError("matched_true_to_slot has the wrong shape")

    hard_slots = logits.argmax(dim=-1)
    contingency = _slot_group_contingency(hard_slots, target_groups)
    best_true_to_slot = hungarian_group_slot_matching(contingency.transpose(1, 2))
    weighted_purity, mean_slot_purity = _purity_from_contingency(contingency)

    hard_counts = contingency.sum(dim=-1)
    soft_counts = F.softmax(logits.float(), dim=-1).sum(dim=1)
    metrics: dict[str, Tensor] = {
        "aligned_partition_acc": _accuracy_after_matching(
            hard_slots, target_groups, matched_true_to_slot
        ),
        "best_partition_acc": _accuracy_after_matching(
            hard_slots, target_groups, best_true_to_slot
        ),
        "partition_purity": weighted_purity,
        "mean_nonempty_slot_purity": mean_slot_purity,
        "hard_slot_count_std": hard_counts.std(dim=1, unbiased=False) / MACRO_CAPACITY,
        "hard_slot_count_min": hard_counts.min(dim=1).values,
        "hard_slot_count_max": hard_counts.max(dim=1).values,
        "hard_empty_slots": (hard_counts == 0).float().sum(dim=1),
        "soft_usage_std": soft_counts.std(dim=1, unbiased=False) / MACRO_CAPACITY,
        "soft_usage_l1": (soft_counts - MACRO_CAPACITY).abs().mean(dim=1) / MACRO_CAPACITY,
        "soft_usage_min": soft_counts.min(dim=1).values,
        "soft_usage_max": soft_counts.max(dim=1).values,
    }
    metrics.update(group_coverage(logits, target_groups, matched_true_to_slot, coverage_ks))

    if include_capacity_hungarian:
        capacity_slots = capacity_hungarian_slots(logits)
        capacity_contingency = _slot_group_contingency(capacity_slots, target_groups)
        capacity_match = hungarian_group_slot_matching(capacity_contingency.transpose(1, 2))
        cap_weighted_purity, _ = _purity_from_contingency(capacity_contingency)
        metrics["capacity_hungarian_acc"] = _accuracy_after_matching(
            capacity_slots, target_groups, capacity_match
        )
        metrics["capacity_hungarian_purity"] = cap_weighted_purity
    return metrics


@torch.no_grad()
def evaluate(
    model: MacroPartitionNet,
    loader: DataLoader,
    device: torch.device,
    *,
    max_images: int,
    coverage_ks: Sequence[int],
    capacity_weight: float,
) -> dict[str, float]:
    """Evaluate only held-out synthetic examples with exact group labels."""
    if max_images < 1:
        raise ValueError("max_images must be positive")
    was_training = model.training
    model.eval()
    values: defaultdict[str, list[float]] = defaultdict(list)
    seen = 0
    for batch in loader:
        if seen >= max_images:
            break
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("partition evaluation must contain only exact synthetic examples")
        take = min(max_images - seen, int(batch["tiles"].shape[0]))
        tiles = batch["tiles"][:take].to(device, non_blocking=True)
        perm = batch["perm"][:take].to(device, non_blocking=True).long()
        with _autocast(device):
            logits = require_assignment_logits(model(tiles))
        labels = macro_group_labels(perm)
        loss, loss_terms, match = partition_loss(
            logits, labels, capacity_weight=capacity_weight
        )
        metric_tensors = partition_metrics(
            logits.float(),
            labels,
            coverage_ks=coverage_ks,
            matched_true_to_slot=match,
        )
        values["loss"].extend([float(loss.detach().cpu())] * take)
        for name, value in loss_terms.items():
            values[name].extend([float(value.detach().cpu())] * take)
        for name, value in metric_tensors.items():
            values[name].extend(value.detach().float().cpu().tolist())
        seen += take
    if was_training:
        model.train()
    if not seen:
        raise RuntimeError("evaluation loader yielded no examples")
    result = {name: float(np.mean(items)) for name, items in values.items()}
    result["eval_images"] = float(seen)
    return result


def save_checkpoint(
    path: str,
    model: MacroPartitionNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    args: argparse.Namespace,
    metrics: Mapping[str, float],
) -> None:
    """Save a workspace-local, resume-friendly experimental checkpoint."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "args": vars(args),
            "metrics": dict(metrics),
            "macro_side": MACRO_SIDE,
            "macro_grid": MACRO_GRID,
            "macro_cells": MACRO_CELLS,
            "macro_capacity": MACRO_CAPACITY,
        },
        path,
    )


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def _parse_coverage_ks(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--coverage_k must be comma-separated integers") from exc
    if not parsed or any(k < 1 or k > NFRAG for k in parsed):
        raise argparse.ArgumentTypeError(f"--coverage_k entries must be in [1,{NFRAG}]")
    return tuple(dict.fromkeys(parsed))


def _next_batch(
    iterator: Iterable[dict[str, Tensor]], loader: DataLoader
) -> tuple[dict[str, Tensor], Iterable[dict[str, Tensor]]]:
    """Cycle a non-empty training loader indefinitely."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--d", type=int, default=None, help="MacroPartitionNet token width")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--set_layers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--slot_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train_n", type=int, default=0, help="0 uses the whole training split")
    parser.add_argument("--eval_n", type=int, default=12, help="held-out synthetic images per evaluation")
    parser.add_argument("--eval_every", type=int, default=400)
    parser.add_argument("--coverage_k", type=_parse_coverage_ks, default=(16, 24, 32))
    parser.add_argument("--capacity_weight", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="partition")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "artifacts",
            "partition",
        ),
        help="workspace-local checkpoint directory",
    )
    parser.add_argument("--device", default=None, help="cuda when available by default")
    args = parser.parse_args()

    if args.steps < 1 or args.bs < 1 or args.eval_n < 1 or args.eval_every < 1:
        parser.error("--steps, --bs, --eval_n and --eval_every must be positive")
    if args.workers < 0 or args.train_n < 0:
        parser.error("--workers and --train_n must be non-negative")
    if args.d is not None and args.d < 1:
        parser.error("--d must be positive")
    if args.heads < 1:
        parser.error("--heads must be positive")
    if args.slot_layers < 1:
        parser.error("--slot_layers must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0,1)")
    if args.capacity_weight < 0.0:
        parser.error("--capacity_weight must be non-negative")
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        parser.error("--lr must be positive and --weight_decay non-negative")
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"device={device} macro_grid={MACRO_GRID} macro_side={MACRO_SIDE} "
        f"macro_capacity={MACRO_CAPACITY}",
        flush=True,
    )

    train_names, val_names = train_val_split()
    if args.train_n:
        train_names = train_names[: args.train_n]
    if not train_names:
        raise RuntimeError("training split is empty")
    if not val_names:
        raise RuntimeError("validation split is empty")
    # Exact, on-the-fly synthetic permutations are essential here: there is no
    # valid way to train an anonymous partition with noisy recovered labels.
    train_ds = CanvasDataset(train_names, real_prob=0.0, seed=args.seed)
    val_ds = CanvasDataset(val_names, real_prob=0.0, seed=args.seed + 10_000)
    train_loader = make_loader(train_ds, args.bs, args.workers, shuffle=True, device=device)
    val_loader = make_loader(
        val_ds, args.bs, min(args.workers, 2), shuffle=False, device=device
    )

    model_kwargs: dict[str, Any] = {
        "tiles": NFRAG,
        "slots": MACRO_CELLS,
        "tile_size": FS,
        "heads": args.heads,
        "set_layers": args.set_layers,
        "slot_layers": args.slot_layers,
        "dropout": args.dropout,
    }
    if args.d is not None:
        model_kwargs["d"] = args.d
    model = MacroPartitionNet(**model_kwargs).to(device)
    print(f"MacroPartitionNet params={count_params(model):,}", flush=True)
    print(
        "group coverage shortlist K=" + ",".join(str(k) for k in args.coverage_k),
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = -float("inf")
    started = time.time()
    iterator: Iterable[dict[str, Tensor]] = iter(train_loader)
    for step in range(1, args.steps + 1):
        batch, iterator = _next_batch(iterator, train_loader)
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("partition training must use CanvasDataset(real_prob=0)")
        tiles = batch["tiles"].to(device, non_blocking=True)
        perm = batch["perm"].to(device, non_blocking=True).long()
        labels = macro_group_labels(perm)
        with _autocast(device):
            logits = require_assignment_logits(model(tiles))
            loss, loss_terms, _ = partition_loss(
                logits, labels, capacity_weight=args.capacity_weight
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
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"ce={float(loss_terms['ce']):.4f} cap={float(loss_terms['capacity']):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} {elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                val_loader,
                device,
                max_images=args.eval_n,
                coverage_ks=args.coverage_k,
                capacity_weight=args.capacity_weight,
            )
            print(f"[SYN partition held-out] step={step} {_format_metrics(metrics)}", flush=True)
            last_path = os.path.join(args.out_dir, f"{args.tag}_last.pt")
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                step=step,
                args=args,
                metrics=metrics,
            )
            # This is the deployment-relevant metric: every output slot must
            # contain exactly sixteen candidates before the local 4x4 solver.
            score = metrics["capacity_hungarian_acc"]
            if score > best:
                best = score
                best_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    args=args,
                    metrics=metrics,
                )
                print(f"saved best capacity_hungarian_acc={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
