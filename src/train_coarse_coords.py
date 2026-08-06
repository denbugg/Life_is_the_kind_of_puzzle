"""Train a coarse 6x6 coordinate model before attempting 24x24 placement.

This is deliberately a smaller and better-conditioned version of the failed
absolute-coordinate experiment.  A shuffled synthetic tile is supervised only
with the 6x6 *macrocell* containing its original 24x24 location.  Each
macrocell represents a 4x4 block of fine cells and must receive exactly sixteen
tiles.  The model is still permutation-equivariant: it sees a bag of 576
independently corrupted tiles and has no input-order features.

Validation reports three complementary quantities:

* ``macro_r1`` / ``macro_r3``: per-tile retrieval of the true macrocell;
* ``macro_hungarian_acc``: membership accuracy after a 16-capacity assignment;
* ``top<K>_coverage``: for each macrocell, the fraction of its sixteen true
  tiles retained among its K highest-scoring candidate tiles.

The latter is intentionally a group-retrieval measure, not another spelling of
R@K.  It answers whether a local 4x4 solver would receive the right tiles in a
shortlist for each predicted macrocell.
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
from torch import nn
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from config import GRID, NFRAG, SEED
from coord_model import CoordSetNet, count_params
from imgio import train_val_split


MACRO_GRID = 6
if GRID % MACRO_GRID:
    raise RuntimeError(f"fine grid {GRID} must be divisible by macro grid {MACRO_GRID}")
FINE_PER_MACRO = GRID // MACRO_GRID
MACRO_CAPACITY = FINE_PER_MACRO * FINE_PER_MACRO
MACRO_CELLS = MACRO_GRID * MACRO_GRID
if NFRAG != MACRO_CELLS * MACRO_CAPACITY:
    raise RuntimeError(
        f"expected {MACRO_CELLS} macrocells x {MACRO_CAPACITY} tiles, got NFRAG={NFRAG}"
    )


def _autocast(device: torch.device):
    """Use CUDA fp16 when available and do nothing on CPU."""
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
    """Build a loader without silently changing the synthetic data policy."""
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


def macro_labels(perm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map exact 24x24 clean-cell labels to 6x6 macro row/column labels.

    ``perm[i]`` is the original row-major clean cell for input tile ``i``.
    A macrocell is a contiguous 4x4 fine-cell region, so its row and column
    labels are precisely ``floor(original_coordinate / 4)``.
    """
    if perm.ndim < 1 or perm.shape[-1] != NFRAG:
        raise ValueError(f"perm must end in {NFRAG} clean-cell labels, got {tuple(perm.shape)}")
    if torch.any(perm < 0) or torch.any(perm >= NFRAG):
        raise ValueError("perm contains a clean-cell index outside the fine grid")
    fine_row = torch.div(perm, GRID, rounding_mode="floor")
    fine_col = torch.remainder(perm, GRID)
    macro_row = torch.div(fine_row, FINE_PER_MACRO, rounding_mode="floor")
    macro_col = torch.div(fine_col, FINE_PER_MACRO, rounding_mode="floor")
    macro_cell = macro_row * MACRO_GRID + macro_col
    return macro_row.long(), macro_col.long(), macro_cell.long()


def require_logits(output: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the subset of the ``CoordSetNet`` forward contract used here."""
    try:
        row_logits = output["row_logits"]
        col_logits = output["col_logits"]
    except KeyError as exc:
        raise KeyError("CoordSetNet forward must return row_logits and col_logits") from exc
    expected = (NFRAG, MACRO_GRID)
    if row_logits.ndim != 3 or col_logits.ndim != 3:
        raise ValueError(
            "row/col logits must be rank 3, got "
            f"{tuple(row_logits.shape)} and {tuple(col_logits.shape)}"
        )
    if row_logits.shape != col_logits.shape or tuple(row_logits.shape[1:]) != expected:
        raise ValueError(
            f"expected matching logits (B,{NFRAG},{MACRO_GRID}), got "
            f"{tuple(row_logits.shape)} and {tuple(col_logits.shape)}"
        )
    return row_logits, col_logits


def combined_macro_logits(row_logits: torch.Tensor, col_logits: torch.Tensor) -> torch.Tensor:
    """Create one tile-to-macrocell score in row-major 6x6 macrocell order."""
    if row_logits.shape != col_logits.shape or row_logits.shape[-1] != MACRO_GRID:
        raise ValueError("row and col logits must have matching final macro-grid dimensions")
    # (B, tile, macro_row, macro_col) -> (B, tile, macrocell)
    return (row_logits.unsqueeze(-1) + col_logits.unsqueeze(-2)).flatten(2)


def supervised_loss(
    row_logits: torch.Tensor,
    col_logits: torch.Tensor,
    perm: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Equal-weight exact cross entropy for the two coarse coordinates."""
    macro_row, macro_col, _ = macro_labels(perm)
    row_loss = F.cross_entropy(row_logits.reshape(-1, MACRO_GRID), macro_row.reshape(-1))
    col_loss = F.cross_entropy(col_logits.reshape(-1, MACRO_GRID), macro_col.reshape(-1))
    loss = 0.5 * (row_loss + col_loss)
    return loss, {
        "loss": float(loss.detach()),
        "macro_row_loss": float(row_loss.detach()),
        "macro_col_loss": float(col_loss.detach()),
    }


def _ranks(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return deterministic one-indexed ranks, treating an exact tie as a hit."""
    target_score = logits.gather(-1, target.unsqueeze(-1))
    return 1 + (logits > target_score).sum(dim=-1)


def rank_metrics(ranks: torch.Tensor) -> dict[str, float]:
    """Summarize macrocell ranks over all tiles in a held-out set."""
    ranks = ranks.detach().float().reshape(-1)
    return {
        "macro_r1": float((ranks <= 1).float().mean()),
        "macro_r3": float((ranks <= 3).float().mean()),
        "macro_mean_rank": float(ranks.mean()),
        "macro_median_rank": float(ranks.median()),
    }


def capacity_hungarian_membership_accuracy(
    macro_scores: torch.Tensor,
    target_macro: torch.Tensor,
) -> list[float]:
    """Decode 16 slots per macrocell and return per-image membership accuracy.

    The ordinary Hungarian algorithm accepts one tile per column.  Repeating
    each of the 36 macrocell score columns sixteen times produces 576 virtual
    columns, exactly enforcing the desired 16-tile capacity for every group.
    ``predicted_macro[tile]`` is then the owning macrocell of the assigned
    virtual column.
    """
    if macro_scores.ndim != 3 or tuple(macro_scores.shape[1:]) != (NFRAG, MACRO_CELLS):
        raise ValueError(f"expected macro_scores (B,{NFRAG},{MACRO_CELLS}), got {tuple(macro_scores.shape)}")
    if target_macro.shape != macro_scores.shape[:2]:
        raise ValueError(
            f"target macro labels must have shape {tuple(macro_scores.shape[:2])}, "
            f"got {tuple(target_macro.shape)}"
        )

    score_np = macro_scores.detach().float().cpu().numpy()
    target_np = target_macro.detach().cpu().numpy().astype(np.int64, copy=False)
    values: list[float] = []
    for scores, target in zip(score_np, target_np):
        # (tile, macrocell) -> (tile, virtual macrocell slot).
        virtual_scores = np.repeat(scores, MACRO_CAPACITY, axis=1)
        tile_rows, virtual_slots = linear_sum_assignment(-virtual_scores)
        predicted = np.empty(NFRAG, dtype=np.int64)
        predicted[tile_rows] = virtual_slots // MACRO_CAPACITY
        values.append(float(np.mean(predicted == target)))
    return values


def topk_group_coverage(
    macro_scores: torch.Tensor,
    target_macro: torch.Tensor,
    ks: Sequence[int],
) -> dict[str, float]:
    """Measure how many true members appear in each macrocell's tile shortlist.

    For macrocell ``m``, take the ``K`` input tiles with the largest score for
    ``m`` and count its true sixteen members.  The reported coverage is that
    count divided by 16 and averaged across all images and all 36 macrocells.
    A high value at K=32, for example, means a later local solver can work on a
    two-times-capacity candidate set without losing many correct tiles.
    """
    if macro_scores.ndim != 3 or tuple(macro_scores.shape[1:]) != (NFRAG, MACRO_CELLS):
        raise ValueError(f"expected macro_scores (B,{NFRAG},{MACRO_CELLS}), got {tuple(macro_scores.shape)}")
    if target_macro.shape != macro_scores.shape[:2]:
        raise ValueError("target_macro must have one macrocell label per tile")

    expanded_labels = target_macro.unsqueeze(-1).expand(-1, -1, MACRO_CELLS)
    macro_ids = torch.arange(MACRO_CELLS, device=macro_scores.device).view(1, 1, -1)
    values: dict[str, float] = {}
    for k in ks:
        if not 1 <= k <= NFRAG:
            raise ValueError(f"coverage K must be in [1, {NFRAG}], got {k}")
        top_tiles = macro_scores.topk(k, dim=1).indices
        labels_at_top = expanded_labels.gather(1, top_tiles)
        hits = (labels_at_top == macro_ids).sum(dim=1)
        values[f"top{k}_coverage"] = float((hits.float() / MACRO_CAPACITY).mean().cpu())
    return values


@torch.no_grad()
def evaluate(
    model: CoordSetNet,
    loader: DataLoader,
    device: torch.device,
    *,
    max_images: int,
    coverage_ks: Sequence[int],
) -> dict[str, float]:
    """Evaluate coarse placement only on exact synthetic held-out examples."""
    if max_images < 1:
        raise ValueError("max_images must be positive")
    was_training = model.training
    model.eval()
    ranks: list[torch.Tensor] = []
    hungarian: list[float] = []
    coverage: defaultdict[str, list[float]] = defaultdict(list)
    seen = 0
    for batch in loader:
        if seen >= max_images:
            break
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("coarse validation must contain only exact synthetic examples")
        take = min(max_images - seen, batch["tiles"].shape[0])
        tiles = batch["tiles"][:take].to(device, non_blocking=True)
        perm = batch["perm"][:take].to(device, non_blocking=True).long()
        with _autocast(device):
            output = model(tiles)
        row_logits, col_logits = require_logits(output)
        macro_scores = combined_macro_logits(row_logits.float(), col_logits.float())
        _, _, target_macro = macro_labels(perm)

        ranks.append(_ranks(macro_scores, target_macro).cpu())
        hungarian.extend(capacity_hungarian_membership_accuracy(macro_scores, target_macro))
        for name, value in topk_group_coverage(macro_scores, target_macro, coverage_ks).items():
            coverage[name].append(value)
        seen += take

    if was_training:
        model.train()
    if not ranks:
        raise RuntimeError("evaluation loader yielded no examples")

    metrics = rank_metrics(torch.cat(ranks))
    metrics["macro_hungarian_acc"] = float(np.mean(hungarian))
    metrics.update({name: float(np.mean(values)) for name, values in coverage.items()})
    metrics["eval_images"] = float(seen)
    return metrics


def save_checkpoint(
    path: str,
    model: CoordSetNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    args: argparse.Namespace,
    metrics: Mapping[str, float],
) -> None:
    """Save all state needed to inspect or resume the coarse experiment."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "args": vars(args),
            "metrics": dict(metrics),
            "macro_grid": MACRO_GRID,
            "macro_capacity": MACRO_CAPACITY,
        },
        path,
    )


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def _parse_coverage_ks(value: str) -> tuple[int, ...]:
    """Parse a compact, deterministic comma-separated shortlist configuration."""
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--coverage_k must be comma-separated integers") from exc
    if not parsed or any(k < 1 or k > NFRAG for k in parsed):
        raise argparse.ArgumentTypeError(f"--coverage_k entries must be in [1, {NFRAG}]")
    return tuple(dict.fromkeys(parsed))


def _next_batch(
    iterator: Iterable[dict[str, torch.Tensor]], loader: DataLoader
) -> tuple[dict[str, torch.Tensor], Any]:
    """Cycle a non-empty synthetic loader indefinitely."""
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
    parser.add_argument("--d", type=int, default=None, help="CoordSetNet token width")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--set_layers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train_n", type=int, default=0, help="0 uses the whole training split")
    parser.add_argument("--eval_n", type=int, default=12, help="synthetic held-out images per evaluation")
    parser.add_argument("--eval_every", type=int, default=400)
    parser.add_argument("--coverage_k", type=_parse_coverage_ks, default=(16, 32, 48))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="coarse_coords")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "artifacts",
            "coarse_coords",
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
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
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
        f"device={device} macro_grid={MACRO_GRID} fine_per_macro={FINE_PER_MACRO} "
        f"capacity={MACRO_CAPACITY}",
        flush=True,
    )

    train_names, val_names = train_val_split()
    if args.train_n:
        train_names = train_names[:args.train_n]
    if not train_names:
        raise RuntimeError("training split is empty")
    if not val_names:
        raise RuntimeError("validation split is empty")

    # Both loaders are exact synthetic data.  In particular, no recovered
    # permutation cache and no real/noisy pseudo-label can enter this experiment.
    train_ds = CanvasDataset(train_names, real_prob=0.0, seed=args.seed)
    val_ds = CanvasDataset(val_names, real_prob=0.0, seed=args.seed + 10_000)
    train_loader = make_loader(train_ds, args.bs, args.workers, shuffle=True, device=device)
    val_loader = make_loader(val_ds, args.bs, min(args.workers, 2), shuffle=False, device=device)

    model_kwargs: dict[str, Any] = {
        "grid": MACRO_GRID,
        "heads": args.heads,
        "set_layers": args.set_layers,
        "dropout": args.dropout,
    }
    if args.d is not None:
        model_kwargs["d"] = args.d
    model = CoordSetNet(**model_kwargs).to(device)
    print(f"CoordSetNet(grid={MACRO_GRID}) params={count_params(model):,}", flush=True)
    print(f"top-k group coverage: {','.join(str(k) for k in args.coverage_k)}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = -float("inf")
    started = time.time()
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        batch, iterator = _next_batch(iterator, train_loader)
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("training must be synthetic exact data (CanvasDataset real_prob=0)")
        tiles = batch["tiles"].to(device, non_blocking=True)
        perm = batch["perm"].to(device, non_blocking=True).long()
        with _autocast(device):
            output = model(tiles)
            row_logits, col_logits = require_logits(output)
            loss, loss_metrics = supervised_loss(row_logits, col_logits, perm)
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
                f"step {step}/{args.steps} loss={loss_metrics['loss']:.4f} "
                f"macro_row={loss_metrics['macro_row_loss']:.4f} "
                f"macro_col={loss_metrics['macro_col_loss']:.4f} "
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
            )
            print(f"[SYN coarse held-out] step={step} {_format_metrics(metrics)}", flush=True)
            last_path = os.path.join(args.out_dir, f"{args.tag}_last.pt")
            save_checkpoint(last_path, model, optimizer, scheduler, step=step, args=args, metrics=metrics)
            if metrics["macro_hungarian_acc"] > best:
                best = metrics["macro_hungarian_acc"]
                best_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
                save_checkpoint(best_path, model, optimizer, scheduler, step=step, args=args, metrics=metrics)
                print(f"saved best macro_hungarian_acc={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
