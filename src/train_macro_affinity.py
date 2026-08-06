"""Train local/macro affinity embeddings from exact synthetic puzzle layouts.

This is intentionally a *relative* pre-training task.  ``MacroAffinityNet``
receives a complete unordered 576-tile image and emits one L2-normalized
embedding per tile.  We supervise the full within-image affinity matrix rather
than asking a tile to predict an image-global coordinate.

By default a tile is positive with every other tile whose original clean-grid
location is within Chebyshev radius three.  That relation crosses arbitrary
4x4 macro boundaries and is therefore much less brittle than treating every
4x4 block as an isolated class.  ``--positive_mode macro`` retains the strict
16-member macro-class objective for ablations.  Validation always reports the
fixed 4x4 macro retrieval and group-coverage metrics needed by the downstream
macro solver.

All examples use ``CanvasDataset(real_prob=0)``: ``perm`` is consequently an
exact clean-cell label, never a recovered or noisy pseudo-label.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from imgio import train_val_split
from macro_affinity import MacroAffinityNet, count_params


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
    """Use fp16 for the encoder on CUDA and preserve a simple CPU path."""
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
    """Build a loader without weakening the exact-synthetic-data invariant."""
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
    """Return the fixed 6x6 macrocell ID for each exact clean-grid cell.

    ``perm[b, tile]`` is the original row-major clean-cell location of an
    input tile, so this is exactly ``(row // 4, col // 4)`` in row-major order.
    Every valid full puzzle has sixteen members of each returned label.
    """
    if perm.ndim < 1 or perm.shape[-1] != NFRAG:
        raise ValueError(f"perm must end in {NFRAG} entries, got {tuple(perm.shape)}")
    if torch.any(perm < 0) or torch.any(perm >= NFRAG):
        raise ValueError("perm has a clean-cell index outside the fine grid")
    perm = perm.long()
    rows = torch.div(perm, GRID, rounding_mode="floor")
    cols = torch.remainder(perm, GRID)
    return (
        torch.div(rows, MACRO_SIDE, rounding_mode="floor") * MACRO_GRID
        + torch.div(cols, MACRO_SIDE, rounding_mode="floor")
    ).long()


def _clean_coordinates(perm: Tensor) -> tuple[Tensor, Tensor]:
    """Split row-major exact clean-cell labels into row and column tensors."""
    if perm.ndim < 1 or perm.shape[-1] != NFRAG:
        raise ValueError(f"perm must end in {NFRAG} entries, got {tuple(perm.shape)}")
    if torch.any(perm < 0) or torch.any(perm >= NFRAG):
        raise ValueError("perm has a clean-cell index outside the fine grid")
    perm = perm.long()
    return (
        torch.div(perm, GRID, rounding_mode="floor"),
        torch.remainder(perm, GRID),
    )


def _off_diagonal_mask(batch: int, count: int, device: torch.device) -> Tensor:
    """Create a broadcasted valid-pair mask with self-pairs removed."""
    return ~torch.eye(count, dtype=torch.bool, device=device).unsqueeze(0).expand(batch, -1, -1)


def spatial_positive_mask(
    perm: Tensor,
    *,
    radius: int = 3,
    positive_mode: str = "radius",
) -> Tensor:
    """Construct the exact positive relation for every pair in each image.

    ``radius`` uses Chebyshev distance in the original 24x24 clean grid.  A
    radius of three makes every corner tile have 15 positives and interior
    tiles up to 48 positives.  All non-positive, non-self tiles are negatives
    for the full-image InfoNCE denominator.
    """
    if positive_mode not in {"radius", "macro"}:
        raise ValueError("positive_mode must be 'radius' or 'macro'")
    if radius < 1:
        raise ValueError("radius must be at least one")
    if perm.ndim != 2 or perm.shape[1] != NFRAG:
        raise ValueError(f"perm must have shape (B,{NFRAG}), got {tuple(perm.shape)}")

    batch, count = perm.shape
    off_diagonal = _off_diagonal_mask(batch, count, perm.device)
    if positive_mode == "macro":
        groups = macro_group_labels(perm)
        return groups.unsqueeze(-1).eq(groups.unsqueeze(-2)) & off_diagonal

    rows, cols = _clean_coordinates(perm)
    row_distance = (rows.unsqueeze(-1) - rows.unsqueeze(-2)).abs()
    col_distance = (cols.unsqueeze(-1) - cols.unsqueeze(-2)).abs()
    return torch.maximum(row_distance, col_distance).le(radius) & off_diagonal


def _validate_relation_mask(mask: Tensor, count: int) -> None:
    """Fail early if a relation cannot support contrastive learning."""
    if mask.ndim != 3 or tuple(mask.shape[1:]) != (count, count):
        raise ValueError(f"relation mask must have shape (B,{count},{count}), got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise TypeError(f"relation mask must be bool, got {mask.dtype}")
    if torch.any(torch.diagonal(mask, dim1=1, dim2=2)):
        raise ValueError("a relation mask must exclude self-pairs")
    positives = mask.sum(dim=-1)
    negatives = count - 1 - positives
    if torch.any(positives == 0):
        raise ValueError("every anchor needs at least one positive")
    if torch.any(negatives == 0):
        raise ValueError("every anchor needs at least one negative")


def cosine_affinity(embeddings: Tensor) -> Tensor:
    """Return a numerically safe full within-image cosine affinity matrix."""
    if embeddings.ndim != 3 or embeddings.shape[1] != NFRAG:
        raise ValueError(
            f"embeddings must have shape (B,{NFRAG},D), got {tuple(embeddings.shape)}"
        )
    if embeddings.shape[-1] < 1:
        raise ValueError("embedding width must be positive")
    if not torch.is_floating_point(embeddings):
        raise TypeError(f"embeddings must be floating point, got {embeddings.dtype}")
    # MacroAffinityNet already emits unit vectors.  Normalizing again makes the
    # trainer safe under AMP and leaves that intended representation unchanged.
    unit = F.normalize(embeddings.float(), dim=-1)
    return torch.matmul(unit, unit.transpose(-1, -2))


def _multi_positive_info_nce_from_affinity(
    affinity: Tensor,
    positive_mask: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Stable full-image supervised contrastive loss.

    For each anchor the denominator contains every other tile from *the same
    image*.  The numerator averages the log-probability of every positive,
    rather than rewarding only the easiest one.  For the strict macro relation
    this means 15 positives and 560 negatives per tile exactly.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if affinity.ndim != 3 or affinity.shape[1] != affinity.shape[2] or affinity.shape[1] != NFRAG:
        raise ValueError(f"affinity must have shape (B,{NFRAG},{NFRAG}), got {tuple(affinity.shape)}")
    if not torch.is_floating_point(affinity):
        raise TypeError(f"affinity must be floating point, got {affinity.dtype}")
    _validate_relation_mask(positive_mask, affinity.shape[1])
    if positive_mask.shape[0] != affinity.shape[0]:
        raise ValueError("affinity and positive_mask batch dimensions must agree")

    batch, count, _ = affinity.shape
    valid = _off_diagonal_mask(batch, count, affinity.device)
    logits = affinity.float() / float(temperature)
    logits = logits.masked_fill(~valid, -torch.inf)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    positive_count = positive_mask.sum(dim=-1).to(log_probs.dtype)
    # Avoid ``0 * -inf`` on the masked diagonal.
    positive_log_probs = torch.where(positive_mask, log_probs, torch.zeros_like(log_probs))
    loss = -(positive_log_probs.sum(dim=-1) / positive_count).mean()

    negative_mask = valid & ~positive_mask
    diagnostics = {
        "positive_count": positive_count.mean(),
        "positive_affinity": affinity.masked_select(positive_mask).mean(),
        "negative_affinity": affinity.masked_select(negative_mask).mean(),
    }
    return loss, diagnostics


def multi_positive_info_nce(
    embeddings: Tensor,
    positive_mask: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the full-image multi-positive contrastive loss from embeddings."""
    return _multi_positive_info_nce_from_affinity(
        cosine_affinity(embeddings), positive_mask, temperature=temperature
    )


def _topk_relation_metrics(
    affinity: Tensor,
    positive_mask: Tensor,
    ks: Sequence[int],
    *,
    prefix: str,
) -> dict[str, float]:
    """Compute anchor-level top-K precision and recall for a pair relation."""
    if not ks:
        return {}
    count = affinity.shape[1]
    if any(k < 1 or k >= count for k in ks):
        raise ValueError(f"top-K values must lie in [1, {count - 1}]")
    _validate_relation_mask(positive_mask, count)
    if positive_mask.shape[0] != affinity.shape[0]:
        raise ValueError("affinity and positive_mask batch dimensions must agree")

    batch = affinity.shape[0]
    valid = _off_diagonal_mask(batch, count, affinity.device)
    max_k = max(ks)
    top_indices = affinity.float().masked_fill(~valid, -torch.inf).topk(max_k, dim=-1).indices
    top_is_positive = positive_mask.gather(-1, top_indices)
    cumulative_hits = top_is_positive.to(torch.float32).cumsum(dim=-1)
    positive_count = positive_mask.sum(dim=-1).to(torch.float32)
    metrics: dict[str, float] = {}
    for k in ks:
        hits = cumulative_hits[..., k - 1]
        metrics[f"{prefix}_precision@{k}"] = float((hits / float(k)).mean())
        metrics[f"{prefix}_recall@{k}"] = float((hits / positive_count).mean())
    return metrics


def macro_group_coverage(
    affinity: Tensor,
    macro_labels: Tensor,
    ks: Sequence[int] = (16, 32, 64),
) -> dict[str, float]:
    """Measure 4x4-group candidate retention from affinity-derived scores.

    During evaluation only, every true group acts as an oracle *query set*.
    A candidate tile is scored by its mean affinity to that set; its own
    diagonal affinity is omitted.  For each group, coverage@K is the fraction
    of its sixteen true members retained in the K highest-scoring candidates.
    This measures whether a downstream local solver would receive the correct
    tiles in its shortlist, without pretending that macro IDs are model slots.
    """
    if not ks:
        return {}
    if any(k < 1 or k > NFRAG for k in ks):
        raise ValueError(f"coverage K values must lie in [1, {NFRAG}]")
    if affinity.ndim != 3 or tuple(affinity.shape[1:]) != (NFRAG, NFRAG):
        raise ValueError(f"affinity must have shape (B,{NFRAG},{NFRAG})")
    if macro_labels.shape != affinity.shape[:2]:
        raise ValueError("macro_labels must provide one label per tile")
    if torch.any(macro_labels < 0) or torch.any(macro_labels >= MACRO_CELLS):
        raise ValueError("macro_labels has an invalid macrocell ID")

    membership = F.one_hot(macro_labels.long(), num_classes=MACRO_CELLS).to(affinity.dtype)
    counts = membership.sum(dim=1)
    if not torch.all(counts.eq(MACRO_CAPACITY)):
        raise ValueError("macro coverage expects exactly sixteen tiles per macrocell")

    batch = affinity.shape[0]
    no_self = affinity.float().masked_fill(
        ~_off_diagonal_mask(batch, NFRAG, affinity.device), 0.0
    )
    # (image, candidate, group): sum affinities from candidate to group members.
    group_scores = torch.einsum("bij,bjg->big", no_self, membership)
    # A member has fifteen available peers while an outsider has sixteen; divide
    # by the actual number so candidate membership cannot win through self-score.
    group_scores = group_scores / (MACRO_CAPACITY - membership).clamp_min(1.0)
    group_scores = group_scores.transpose(1, 2)  # (B, group, candidate)
    labels_by_group = macro_labels.long().unsqueeze(1).expand(-1, MACRO_CELLS, -1)
    group_ids = torch.arange(MACRO_CELLS, device=affinity.device).view(1, -1, 1)

    metrics: dict[str, float] = {}
    for k in ks:
        selected = group_scores.topk(k, dim=-1).indices
        recovered_labels = labels_by_group.gather(-1, selected)
        hits = recovered_labels.eq(group_ids).sum(dim=-1).to(torch.float32)
        metrics[f"macro_coverage@{k}"] = float((hits / float(MACRO_CAPACITY)).mean())
    return metrics


def sampled_affinity_auc(
    affinity: Tensor,
    positive_mask: Tensor,
    *,
    samples_per_image: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Estimate pairwise ROC AUC using balanced sampled positive/negative pairs.

    Sampling is anchor-balanced, so a large interior radius neighbourhood cannot
    swamp border anchors.  Direct comparison gives the correct 0.5 contribution
    to exact ties (important for an untrained or collapsed model).
    """
    if samples_per_image < 1:
        raise ValueError("samples_per_image must be positive")
    if affinity.ndim != 3 or tuple(affinity.shape[1:]) != (NFRAG, NFRAG):
        raise ValueError(f"affinity must have shape (B,{NFRAG},{NFRAG})")
    _validate_relation_mask(positive_mask, NFRAG)
    if positive_mask.shape[0] != affinity.shape[0]:
        raise ValueError("affinity and positive_mask batch dimensions must agree")

    batch = affinity.shape[0]
    device = affinity.device
    anchors = torch.randint(
        NFRAG, (batch, samples_per_image), device=device, generator=generator
    )
    image_ids = torch.arange(batch, device=device).unsqueeze(1).expand_as(anchors)
    positive_rows = positive_mask[image_ids, anchors]
    valid = _off_diagonal_mask(batch, NFRAG, device)
    negative_rows = (valid & ~positive_mask)[image_ids, anchors]

    positive_indices = torch.multinomial(
        positive_rows.reshape(-1, NFRAG).float(), 1, generator=generator
    ).reshape(batch, samples_per_image)
    negative_indices = torch.multinomial(
        negative_rows.reshape(-1, NFRAG).float(), 1, generator=generator
    ).reshape(batch, samples_per_image)
    positive_scores = affinity[image_ids, anchors, positive_indices].float()
    negative_scores = affinity[image_ids, anchors, negative_indices].float()

    # P=1024 uses only four MB per image in fp32; it is both more robust to ties
    # than rank shortcuts and still tiny compared with a 576x576 affinity batch.
    differences = positive_scores.unsqueeze(-1) - negative_scores.unsqueeze(-2)
    return (differences.gt(0).to(torch.float32) + 0.5 * differences.eq(0).to(torch.float32)).mean()


def _extract_embeddings(model: MacroAffinityNet, tiles: Tensor) -> Tensor:
    """Use the narrow ``MacroAffinityNet.embed`` API, with a helpful contract error."""
    output = model.embed(tiles)
    # The current model deliberately returns a tensor.  Supporting this small
    # mapping adapter keeps saved experimental variants usable without allowing
    # the trainer to fall back to an unrelated ``forward`` implementation.
    if isinstance(output, Mapping):
        for key in ("embeddings", "tile_embeddings"):
            if key in output:
                output = output[key]
                break
        else:
            raise KeyError("MacroAffinityNet.embed mapping lacks embeddings")
    if not isinstance(output, Tensor):
        raise TypeError(f"MacroAffinityNet.embed must return a Tensor, got {type(output)!r}")
    if output.ndim != 3 or output.shape[0] != tiles.shape[0] or output.shape[1] != NFRAG:
        raise ValueError(
            "MacroAffinityNet.embed must return (B,576,D), got "
            f"{tuple(output.shape)} for tiles {tuple(tiles.shape)}"
        )
    if output.shape[-1] < 1 or not torch.is_floating_point(output):
        raise TypeError("MacroAffinityNet.embed must return non-empty floating embeddings")
    return output


@torch.no_grad()
def evaluate(
    model: MacroAffinityNet,
    loader: DataLoader,
    device: torch.device,
    *,
    max_images: int,
    topk: Sequence[int],
    coverage_ks: Sequence[int],
    positive_mode: str,
    radius: int,
    temperature: float,
    auc_samples: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate exact synthetic held-out images with local and macro diagnostics."""
    if max_images < 1:
        raise ValueError("max_images must be positive")
    was_training = model.training
    model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    seen = 0
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))

    for batch in loader:
        if seen >= max_images:
            break
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("macro-affinity validation must use exact synthetic examples")
        take = min(max_images - seen, batch["tiles"].shape[0])
        tiles = batch["tiles"][:take].to(device, non_blocking=True)
        perm = batch["perm"][:take].to(device, non_blocking=True).long()

        with _autocast(device):
            embeddings = _extract_embeddings(model, tiles)
        affinity = cosine_affinity(embeddings)
        local_positive = spatial_positive_mask(
            perm, radius=radius, positive_mode=positive_mode
        )
        macro_positive = spatial_positive_mask(perm, radius=radius, positive_mode="macro")
        macro_labels = macro_group_labels(perm)
        loss, diagnostics = _multi_positive_info_nce_from_affinity(
            affinity, local_positive, temperature=temperature
        )

        image_metrics: dict[str, float] = {
            "loss": float(loss),
            "train_relation_positive_count": float(diagnostics["positive_count"]),
            "train_relation_positive_affinity": float(diagnostics["positive_affinity"]),
            "train_relation_negative_affinity": float(diagnostics["negative_affinity"]),
            "affinity_auc": float(
                sampled_affinity_auc(
                    affinity,
                    local_positive,
                    samples_per_image=auc_samples,
                    generator=generator,
                )
            ),
            "macro_affinity_auc": float(
                sampled_affinity_auc(
                    affinity,
                    macro_positive,
                    samples_per_image=auc_samples,
                    generator=generator,
                )
            ),
        }
        # These are the requested downstream fixed-4x4 same-group metrics.
        image_metrics.update(
            _topk_relation_metrics(
                affinity, macro_positive, topk, prefix="same_group"
            )
        )
        # Kept separately because the training relation is normally spatial
        # radius, not an arbitrary hard macro partition.
        image_metrics.update(
            _topk_relation_metrics(
                affinity, local_positive, topk, prefix="local"
            )
        )
        image_metrics.update(macro_group_coverage(affinity, macro_labels, coverage_ks))
        for name, value in image_metrics.items():
            totals[name] += value * take
        seen += take

    if was_training:
        model.train()
    if not seen:
        raise RuntimeError("evaluation loader yielded no examples")
    metrics = {name: value / seen for name, value in totals.items()}
    metrics["eval_images"] = float(seen)
    return metrics


def save_checkpoint(
    path: str,
    model: MacroAffinityNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    args: argparse.Namespace,
    metrics: Mapping[str, float],
) -> None:
    """Save self-contained workspace checkpoints for inspection or resumption."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "args": vars(args),
            "metrics": dict(metrics),
            "grid": GRID,
            "macro_side": MACRO_SIDE,
            "macro_grid": MACRO_GRID,
            "macro_capacity": MACRO_CAPACITY,
        },
        path,
    )


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return " ".join(f"{name}={value:.4f}" for name, value in metrics.items())


def _parse_ks(value: str, *, maximum: int, flag: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{flag} must be comma-separated integers") from exc
    if not parsed or any(k < 1 or k > maximum for k in parsed):
        raise argparse.ArgumentTypeError(f"{flag} entries must be in [1, {maximum}]")
    return tuple(dict.fromkeys(parsed))


def _parse_topk(value: str) -> tuple[int, ...]:
    return _parse_ks(value, maximum=NFRAG - 1, flag="--topk")


def _parse_coverage_ks(value: str) -> tuple[int, ...]:
    return _parse_ks(value, maximum=NFRAG, flag="--coverage_k")


def _next_batch(
    iterator: Iterable[dict[str, Tensor]], loader: DataLoader
) -> tuple[dict[str, Tensor], Any]:
    """Cycle a non-empty data loader without retaining old batches."""
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
    parser.add_argument("--embedding_dim", "--d", dest="embedding_dim", type=int, default=128)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--stats_hidden", type=int, default=32)
    parser.add_argument("--no_stats", action="store_true", help="disable tile colour/statistics branch")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--positive_mode",
        choices=("radius", "macro"),
        default="radius",
        help="radius is the default relative objective; macro is a 4x4 ablation",
    )
    parser.add_argument("--radius", type=int, default=3, help="Chebyshev positive radius")
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_n", type=int, default=0, help="0 uses the full train split")
    parser.add_argument("--eval_n", type=int, default=12, help="exact synthetic images per evaluation")
    parser.add_argument("--eval_every", type=int, default=400)
    parser.add_argument("--topk", type=_parse_topk, default=(1, 4, 8, 15, 16, 32, 64))
    parser.add_argument("--coverage_k", type=_parse_coverage_ks, default=(16, 32, 64))
    parser.add_argument("--auc_samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="macro_affinity")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "artifacts",
            "macro_affinity",
        ),
        help="workspace-local checkpoint directory",
    )
    parser.add_argument("--device", default=None, help="cuda when available by default")
    args = parser.parse_args()

    if args.steps < 1 or args.bs < 1 or args.eval_n < 1 or args.eval_every < 1:
        parser.error("--steps, --bs, --eval_n and --eval_every must be positive")
    if args.workers < 0 or args.train_n < 0:
        parser.error("--workers and --train_n must be non-negative")
    if args.embedding_dim < 1 or args.width < 1 or args.stats_hidden < 1:
        parser.error("--embedding_dim, --width and --stats_hidden must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    # At the centre of an even 24x24 board the farthest clean cell is only
    # twelve Chebyshev steps away.  Larger radii would leave some anchors with
    # no negatives and make the contrastive denominator degenerate.
    max_radius_with_negatives = (GRID - 2) // 2
    if args.radius < 1 or (
        args.positive_mode == "radius" and args.radius > max_radius_with_negatives
    ):
        parser.error(
            f"--radius must be in [1, {max_radius_with_negatives}] for the radius relation"
        )
    if args.temperature <= 0.0 or args.lr <= 0.0 or args.weight_decay < 0.0:
        parser.error("--temperature and --lr must be positive; --weight_decay non-negative")
    if args.auc_samples < 1:
        parser.error("--auc_samples must be positive")
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"device={device} relation={args.positive_mode} radius={args.radius} "
        f"macro={MACRO_GRID}x{MACRO_GRID} members={MACRO_CAPACITY}",
        flush=True,
    )

    train_names, val_names = train_val_split()
    if args.train_n:
        train_names = train_names[:args.train_n]
    if not train_names:
        raise RuntimeError("training split is empty")
    if not val_names:
        raise RuntimeError("validation split is empty")
    # Never use real inputs here: exact synthetic distortion/shuffle supplies
    # the clean geometry without the old recovered-permutation cache.
    train_ds = CanvasDataset(train_names, real_prob=0.0, seed=args.seed)
    val_ds = CanvasDataset(val_names, real_prob=0.0, seed=args.seed + 10_000)
    train_loader = make_loader(train_ds, args.bs, args.workers, shuffle=True, device=device)
    val_loader = make_loader(val_ds, args.bs, min(args.workers, 2), shuffle=False, device=device)

    model = MacroAffinityNet(
        tiles=NFRAG,
        tile_size=FS,
        embedding_dim=args.embedding_dim,
        width=args.width,
        use_stats=not args.no_stats,
        stats_hidden=args.stats_hidden,
        dropout=args.dropout,
    ).to(device)
    print(f"MacroAffinityNet params={count_params(model):,}", flush=True)
    print(
        f"same-group topK={','.join(map(str, args.topk))} "
        f"macro-coverage K={','.join(map(str, args.coverage_k))}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -float("inf")
    selection_key = f"macro_coverage@{args.coverage_k[0]}"
    started = time.time()
    iterator = iter(train_loader)

    for step in range(1, args.steps + 1):
        batch, iterator = _next_batch(iterator, train_loader)
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("training must use CanvasDataset(real_prob=0)")
        tiles = batch["tiles"].to(device, non_blocking=True)
        perm = batch["perm"].to(device, non_blocking=True).long()
        positive = spatial_positive_mask(
            perm, radius=args.radius, positive_mode=args.positive_mode
        )
        with _autocast(device):
            embeddings = _extract_embeddings(model, tiles)
        # Keep the 576x576 logits and logsumexp in fp32 even under AMP.
        loss, diagnostics = multi_positive_info_nce(
            embeddings, positive, temperature=args.temperature
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
                f"pos={float(diagnostics['positive_count'].detach()):.1f} "
                f"a+={float(diagnostics['positive_affinity'].detach()):.3f} "
                f"a-={float(diagnostics['negative_affinity'].detach()):.3f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} {elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                val_loader,
                device,
                max_images=args.eval_n,
                topk=args.topk,
                coverage_ks=args.coverage_k,
                positive_mode=args.positive_mode,
                radius=args.radius,
                temperature=args.temperature,
                auc_samples=args.auc_samples,
                seed=args.seed + step,
            )
            print(f"[SYN macro-affinity held-out] step={step} {_format_metrics(metrics)}", flush=True)
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
            if metrics[selection_key] > best:
                best = metrics[selection_key]
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
                print(f"saved best {selection_key}={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
