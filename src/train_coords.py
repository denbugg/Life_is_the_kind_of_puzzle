"""Train and gate ``CoordSetNet`` on exact synthetic coordinate labels.

The training task deliberately contains no recovered train-input permutations:

``clean target -> independently distorted 20px tiles -> random tile shuffle``.

For an input tile ``i``, ``perm[i]`` is consequently its exact row-major clean
cell.  ``CoordSetNet`` predicts its row and column independently; their summed
logits define a 576-cell score matrix which is decoded with Hungarian matching.
The held-out evaluation therefore measures the full assignment task rather than
just independent row/column classification.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim
from torch import nn
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from canvas_metrics import decoded_geometry, hard_assignment
from config import GRID, NFRAG, SEED
from coord_model import CoordSetNet, count_params
from imgio import from_frags, train_val_split


def _autocast(device: torch.device):
    """CUDA fp16 context, or a no-op context on CPU."""
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


def _labels(perm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert clean row-major cell labels into independent row/column labels."""
    return torch.div(perm, GRID, rounding_mode="floor"), torch.remainder(perm, GRID)


def _require_logits(output: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the narrow public ``CoordSetNet`` forward contract."""
    try:
        row_logits = output["row_logits"]
        col_logits = output["col_logits"]
    except KeyError as exc:
        raise KeyError("CoordSetNet forward must return row_logits and col_logits") from exc
    if row_logits.ndim != 3 or col_logits.ndim != 3:
        raise ValueError(
            "CoordSetNet logits must be (B,576,24), got "
            f"{tuple(row_logits.shape)} and {tuple(col_logits.shape)}"
        )
    if row_logits.shape != col_logits.shape or row_logits.shape[1:] != (NFRAG, GRID):
        raise ValueError(
            "CoordSetNet logits must both be (B,576,24), got "
            f"{tuple(row_logits.shape)} and {tuple(col_logits.shape)}"
        )
    return row_logits, col_logits


def combined_logits(row_logits: torch.Tensor, col_logits: torch.Tensor) -> torch.Tensor:
    """Make one score for every tile-to-cell pairing in row-major cell order."""
    # (B, tile, row, col) -> (B, tile, row * GRID + col)
    return (row_logits.unsqueeze(-1) + col_logits.unsqueeze(-2)).flatten(2)


def _ranks(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One-indexed rank of each target class, with a deterministic tie convention."""
    target_score = logits.gather(-1, target.unsqueeze(-1))
    # Equal logits are uncommon after initialization.  Treat ties in favour of
    # the target so an all-equal untrained classifier has the expected rank 1.
    return 1 + (logits > target_score).sum(dim=-1)


def _rank_metrics(prefix: str, ranks: torch.Tensor) -> dict[str, float]:
    ranks = ranks.detach().float().reshape(-1)
    return {
        f"{prefix}_r1": float((ranks <= 1).float().mean()),
        f"{prefix}_r3": float((ranks <= 3).float().mean()),
        f"{prefix}_r5": float((ranks <= 5).float().mean()),
        f"{prefix}_mean_rank": float(ranks.mean()),
        f"{prefix}_median_rank": float(ranks.median()),
    }


def supervised_loss(
    row_logits: torch.Tensor,
    col_logits: torch.Tensor,
    perm: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Balanced exact CE for the row and column labels of synthetic examples."""
    row, col = _labels(perm)
    row_loss = F.cross_entropy(row_logits.reshape(-1, GRID), row.reshape(-1))
    col_loss = F.cross_entropy(col_logits.reshape(-1, GRID), col.reshape(-1))
    loss = 0.5 * (row_loss + col_loss)
    return loss, {
        "loss": float(loss.detach()),
        "row_loss": float(row_loss.detach()),
        "col_loss": float(col_loss.detach()),
    }


def _solve_ssim(tiles: torch.Tensor, clean: torch.Tensor, scores: torch.Tensor) -> list[float]:
    """Reassemble a Hungarian solution and score it against the clean target."""
    tiles_np = tiles.detach().float().cpu().permute(0, 1, 3, 4, 2).numpy()
    clean_np = clean.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    values: list[float] = []
    for frags, target, score in zip(tiles_np, clean_np, scores):
        place = hard_assignment(score)
        assembled = from_frags(frags[place])
        values.append(float(sk_ssim(target, assembled, channel_axis=2, data_range=1.0)))
    return values


@torch.no_grad()
def evaluate(
    model: CoordSetNet,
    loader: DataLoader,
    device: torch.device,
    *,
    max_images: int,
) -> dict[str, float]:
    """Evaluate only synthetic held-out puzzles with exact placement labels."""
    model.eval()
    ranks: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    geometry: defaultdict[str, list[float]] = defaultdict(list)
    seen = 0
    for batch in loader:
        if seen >= max_images:
            break
        # Validation is deliberately made synthetic, but retain this check so
        # the evaluator stays sound if a mixed loader is supplied accidentally.
        synthetic_cpu = batch["has_perm"].bool()
        if not bool(synthetic_cpu.any()):
            continue
        # Respect max_images exactly even when it falls in the middle of a
        # DataLoader batch.  Validation itself is synthetic, so this just
        # slices the leading labelled examples deterministically.
        selected_cpu = torch.zeros_like(synthetic_cpu)
        available = max_images - seen
        selected_indices = synthetic_cpu.nonzero(as_tuple=False).flatten()[:available]
        selected_cpu[selected_indices] = True
        if not bool(selected_cpu.any()):
            break
        tiles = batch["tiles"].to(device, non_blocking=True)
        selected = selected_cpu.to(device, non_blocking=True)
        with _autocast(device):
            output = model(tiles)
        row_logits, col_logits = _require_logits(output)
        row_logits, col_logits = row_logits.float()[selected], col_logits.float()[selected]
        perm = batch["perm"][selected_cpu].to(device, non_blocking=True).long()
        row, col = _labels(perm)
        scores = combined_logits(row_logits, col_logits)

        ranks["row"].append(_ranks(row_logits, row).cpu())
        ranks["col"].append(_ranks(col_logits, col).cpu())
        ranks["cell"].append(_ranks(scores, perm).cpu())
        decoded = decoded_geometry(scores, perm)
        geometry["placement"].append(decoded["place_acc"])
        geometry["neighbour"].append(decoded["neighbour_acc"])
        # Index the batch before image-space scoring so only labelled samples
        # contribute should evaluate ever be called with a mixed dataset.
        geometry["solve_ssim"].extend(
            _solve_ssim(tiles[selected], batch["clean"][selected_cpu], scores)
        )
        seen += int(selected_cpu.sum())

    model.train()
    if not ranks["cell"]:
        raise RuntimeError("evaluation loader yielded no synthetic labelled samples")
    metrics: dict[str, float] = {}
    for name in ("row", "col", "cell"):
        metrics.update(_rank_metrics(name, torch.cat(ranks[name])))
    metrics["placement"] = float(np.mean(geometry["placement"]))
    metrics["neighbour"] = float(np.mean(geometry["neighbour"]))
    metrics["solve_ssim"] = float(np.mean(geometry["solve_ssim"]))
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
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "args": vars(args),
            "metrics": dict(metrics),
        },
        path,
    )


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def _next_labelled_batch(iterator: Iterable[dict[str, torch.Tensor]], loader: DataLoader) -> tuple[dict[str, torch.Tensor], Any]:
    """Return a batch containing at least one synthetic example.

    ``real_prob`` exists for future semi-supervised additions.  Until then real
    images have no trusted coordinate labels, so they must not enter CE.
    """
    for _ in range(1_000):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if bool(batch["has_perm"].any()):
            return batch, iterator
    raise RuntimeError("could not draw a synthetic-labelled batch; lower --real_prob")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--d",
        type=int,
        default=None,
        help="CoordSetNet token width; omit to use the model's own default",
    )
    parser.add_argument("--real_prob", type=float, default=0.0,
                        help="optional unlabeled real inputs; default keeps all updates exactly supervised")
    parser.add_argument("--train_n", type=int, default=0, help="0 uses all non-validation targets")
    parser.add_argument("--eval_n", type=int, default=12, help="held-out synthetic images per evaluation")
    parser.add_argument("--eval_every", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="coords")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "coords"),
        help="workspace-local checkpoint directory",
    )
    parser.add_argument("--device", default=None, help="cuda when available by default")
    args = parser.parse_args()
    if args.steps < 1 or args.bs < 1 or args.eval_n < 1 or args.eval_every < 1:
        parser.error("--steps, --bs, --eval_n and --eval_every must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.d is not None and args.d < 1:
        parser.error("--d must be positive")
    if not 0.0 <= args.real_prob < 1.0:
        parser.error("--real_prob must be in [0, 1); real samples have no trusted coordinate labels yet")
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        parser.error("--lr must be positive and --weight_decay non-negative")
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}", flush=True)

    train_names, val_names = train_val_split()
    if args.train_n:
        train_names = train_names[:args.train_n]
    if not train_names:
        raise RuntimeError("training split is empty")
    train_ds = CanvasDataset(train_names, real_prob=args.real_prob, seed=args.seed)
    # Exact synthetic corruption and a fresh input shuffle are intentionally used
    # for validation as well; no recovered permutation cache participates.
    val_ds = CanvasDataset(val_names, real_prob=0.0, seed=args.seed + 10_000)
    train_loader = make_loader(train_ds, args.bs, args.workers, shuffle=True, device=device)
    val_loader = make_loader(val_ds, args.bs, min(args.workers, 2), shuffle=False, device=device)

    model_kwargs = {} if args.d is None else {"d": args.d}
    model = CoordSetNet(**model_kwargs).to(device)
    print(f"CoordSetNet params={count_params(model):,}", flush=True)
    if args.real_prob:
        print(
            f"real_prob={args.real_prob:.3f}: unlabeled real samples are skipped by coordinate CE; "
            "they are not treated as pseudo-labels.",
            flush=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = -float("inf")
    started = time.time()
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        batch, iterator = _next_labelled_batch(iterator, train_loader)
        synthetic = batch["has_perm"].to(device, non_blocking=True).bool()
        tiles = batch["tiles"].to(device, non_blocking=True)
        perm = batch["perm"].to(device, non_blocking=True).long()
        with _autocast(device):
            output = model(tiles)
            row_logits, col_logits = _require_logits(output)
            loss, loss_metrics = supervised_loss(row_logits[synthetic], col_logits[synthetic], perm[synthetic])
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
                f"row={loss_metrics['row_loss']:.4f} col={loss_metrics['col_loss']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} {elapsed / step:.2f}s/it",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val_loader, device, max_images=args.eval_n)
            print(f"[SYN held-out] step={step} {_format_metrics(metrics)}", flush=True)
            last_path = os.path.join(args.out_dir, f"{args.tag}_last.pt")
            save_checkpoint(last_path, model, optimizer, scheduler, step=step, args=args, metrics=metrics)
            if metrics["solve_ssim"] > best:
                best = metrics["solve_ssim"]
                best_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
                save_checkpoint(best_path, model, optimizer, scheduler, step=step, args=args, metrics=metrics)
                print(f"saved best solve_ssim={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
