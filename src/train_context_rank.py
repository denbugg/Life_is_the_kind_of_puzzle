"""Train an oracle context-conditioned A->B->C continuation ranker.

This is a deliberately scoped next gate after pairwise seam ranking plateaued:
the trainer is given an exact synthetic, already-correct A->B link and asks
whether three-tile context can rank the true next C among B's frozen affinity
union candidates.  It never feeds clean coordinates, recovered positions, or
real-image pseudo-labels to the model.

Typical guarded run after the CPU smoke::

    python src/train_context_rank.py --steps 800 --bs 2 --rows-per-image 24 ^
      --eval-every 100 --eval-n 4 --device cuda
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
from torch import Tensor, nn
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from config import FS, NFRAG, SEED
from context_rank import (
    ContextContinuationRanker,
    continuation_rank_metric_sums,
    continuation_target_slots,
    continuation_targets,
    count_params,
    finalize_continuation_metrics,
    listwise_cross_entropy,
    score_continuation_rows,
    select_continuation_rows,
    smoke,
)
from imgio import train_val_split
from train_offset_pose import checkpoint_sha256, load_frozen_affinity, mine_affinity_candidates


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def _make_loader(dataset: CanvasDataset, batch_size: int, workers: int, *, shuffle: bool, device: torch.device) -> DataLoader:
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


def _next_batch(iterator: Iterable[dict[str, Tensor]], loader: DataLoader) -> tuple[dict[str, Tensor], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _append(total: defaultdict[str, float], values: Mapping[str, float]) -> None:
    for key, value in values.items():
        total[key] += float(value)


def _format(metrics: Mapping[str, float]) -> str:
    order = (
        "continuation_candidate_coverage_all_true",
        "continuation_target_r1",
        "continuation_target_r5",
        "continuation_target_mrr",
        "continuation_target_r1_all_true_proxy",
        "continuation_target_r5_all_true_proxy",
        "continuation_target_cross_entropy",
        "continuation_rank_rows",
        "eval_images",
    )
    seen: set[str] = set()
    pieces: list[str] = []
    for key in order:
        if key in metrics:
            pieces.append(f"{key}={metrics[key]:.4f}")
            seen.add(key)
    pieces.extend(f"{key}={value:.4f}" for key, value in metrics.items() if key not in seen)
    return " ".join(pieces)


@torch.inference_mode()
def evaluate(
    model: ContextContinuationRanker,
    affinity: nn.Module,
    loader: DataLoader,
    *,
    candidate_k: int,
    max_images: int,
    rows_per_image: int,
    pair_batch: int,
    device: torch.device,
    affinity_secondary: nn.Module | None,
) -> dict[str, float]:
    """Held-out synthetic oracle continuation metrics.

    Candidate coverage is over *all* valid two-step chains. R@1/R@5/MRR are
    conditional on true C being present in B's frozen hard list and are scored
    on a deterministic direction-balanced subset. Their product with coverage
    is the comparable all-true proxy: it cannot hide candidate misses.
    """
    was_training = model.training
    model.eval()
    total: defaultdict[str, float] = defaultdict(float)
    seen = 0
    for batch in loader:
        if seen >= max_images:
            break
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("context-rank validation requires CanvasDataset(real_prob=0)")
        take = min(max_images - seen, int(batch["tiles"].shape[0]))
        tiles = batch["tiles"][:take].to(device, non_blocking=True)
        perm = batch["perm"][:take].to(device, non_blocking=True).long()
        candidates, valid = mine_affinity_candidates(
            affinity, tiles, candidate_k=candidate_k, device=device, affinity_secondary=affinity_secondary
        )
        middles, targets, exists = continuation_targets(perm)
        slots, available = continuation_target_slots(candidates, valid, middles, targets, exists)
        rows = select_continuation_rows(
            middles, targets, slots, available, rows_per_image=rows_per_image, random_sample=False
        )
        if rows.count:
            with _autocast(device):
                scores = score_continuation_rows(model, tiles, candidates, valid, rows, pair_batch=pair_batch)
            _append(total, continuation_rank_metric_sums(scores.float(), rows.target_slots))
        total["true_chains"] += float(exists.sum())
        total["covered_chains"] += float(available.sum())
        total["selected_rows"] += float(rows.count)
        seen += take
    if was_training:
        model.train()
    if not seen:
        raise RuntimeError("evaluation loader yielded no images")

    metrics = finalize_continuation_metrics(dict(total))
    coverage = _ratio(total["covered_chains"], total["true_chains"])
    metrics.update(
        {
            "continuation_candidate_coverage_all_true": coverage,
            "continuation_target_r1_all_true_proxy": coverage * metrics["continuation_target_r1"],
            "continuation_target_r5_all_true_proxy": coverage * metrics["continuation_target_r5"],
            "continuation_true_chains": total["true_chains"],
            "continuation_covered_chains": total["covered_chains"],
            "continuation_selected_rows": total["selected_rows"],
            "eval_images": float(seen),
        }
    )
    return metrics


def _checkpoint(
    path: str,
    model: ContextContinuationRanker,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    args: argparse.Namespace,
    metrics: Mapping[str, float],
    affinity_provenance: list[Mapping[str, Any]],
) -> None:
    """Persist a self-describing checkpoint; no hidden coordinate contract."""
    torch.save(
        {
            "schema_version": 1,
            "experiment": "context_continuation_ranker",
            "model": model.state_dict(),
            "model_kwargs": {
                "tile_size": model.tile_size,
                "width": model.width,
                "dropout": model.dropout,
                "context_band": model.context_band,
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "args": vars(args),
            "metrics": dict(metrics),
            "candidate_graph": {
                "encoders": [dict(item) for item in affinity_provenance],
                "per_encoder_top_k": int(args.candidate_k),
                "union": len(affinity_provenance) > 1,
                "max_candidates_per_row": int(args.candidate_k) * len(affinity_provenance),
                "candidate_row_owner": "oracle middle B from known-correct A->B",
                "candidate_target": "C, the next same-direction tile after B",
                "objective": "full frozen hard-list cross entropy conditional on C retained",
                "layout": "A-B-C canonical physical 20x60 with cardinal rotation",
            },
            "supervision": {
                "source": "fresh CanvasDataset(real_prob=0) synthetic shuffle/distortion",
                "absolute_coordinates_input_to_model": False,
                "input_position_features": False,
                "known_correct_context": "A->B supplied only to define this oracle gate",
            },
        },
        path,
    )


def _parse_args() -> argparse.Namespace:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--affinity-ckpt", "--affinity_ckpt", dest="affinity_ckpt",
        default=os.path.join(workspace, "artifacts", "macro_affinity", "affinity_r1_1200_best.pt"),
        help="primary frozen MacroAffinityNet checkpoint",
    )
    parser.add_argument(
        "--affinity-ckpt2", "--affinity_ckpt2", dest="affinity_ckpt2",
        default=os.path.join(workspace, "artifacts", "macro_affinity", "affinity_r3_1000_best.pt"),
        help="secondary frozen affinity checkpoint; empty disables the top-K union",
    )
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-n", "--train_n", dest="train_n", type=int, default=0)
    parser.add_argument("--candidate-k", "--candidate_k", dest="candidate_k", type=int, default=64)
    parser.add_argument("--rows-per-image", "--rows_per_image", dest="rows_per_image", type=int, default=24)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--context-band", "--context_band", dest="context_band", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4.0e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--eval-n", "--eval_n", dest="eval_n", type=int, default=4)
    parser.add_argument("--eval-bs", "--eval_bs", dest="eval_bs", type=int, default=1)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=100)
    parser.add_argument("--eval-rows-per-image", "--eval_rows_per_image", dest="eval_rows_per_image", type=int, default=128)
    parser.add_argument("--pair-batch", "--pair_batch", dest="pair_batch", type=int, default=2048)
    parser.add_argument("--train-pair-batch", "--train_pair_batch", dest="train_pair_batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="context_rank")
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default=os.path.join(workspace, "artifacts", "context_rank"))
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--tiny-smoke", "--tiny_smoke", action="store_true", help="run data-free CPU-safe contract smoke and exit")
    args = parser.parse_args()
    if args.steps < 1 or args.bs < 1 or args.eval_n < 1 or args.eval_bs < 1:
        parser.error("--steps, --bs, --eval-n, and --eval-bs must be positive")
    if args.workers < 0 or args.train_n < 0:
        parser.error("--workers and --train-n must be non-negative")
    if not 1 <= args.candidate_k < NFRAG:
        parser.error(f"--candidate-k must lie in [1,{NFRAG - 1}]")
    if args.rows_per_image < 4 or args.eval_rows_per_image < 4:
        parser.error("row counts must be at least four for direction balance")
    if args.width < 4 or args.context_band < 1 or args.pair_batch < 1 or args.train_pair_batch < 1:
        parser.error("invalid width/context-band/pair-batch value")
    if args.lr <= 0.0 or args.weight_decay < 0.0 or not 0.0 <= args.dropout < 1.0 or args.eval_every < 1:
        parser.error("invalid optimizer/dropout/eval value")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.tiny_smoke:
        print(f"[context-rank tiny smoke] device={device} {smoke(device)}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    affinity_path = os.path.abspath(args.affinity_ckpt)
    affinity_path2 = os.path.abspath(args.affinity_ckpt2) if args.affinity_ckpt2 else None
    if affinity_path2 and os.path.normcase(affinity_path) == os.path.normcase(affinity_path2):
        raise ValueError("--affinity-ckpt2 must differ from --affinity-ckpt")
    affinity, _, affinity_kwargs = load_frozen_affinity(affinity_path, device)
    affinity_provenance: list[Mapping[str, Any]] = [{
        "path": affinity_path, "sha256": checkpoint_sha256(affinity_path), "model_kwargs": dict(affinity_kwargs),
    }]
    affinity_secondary: nn.Module | None = None
    if affinity_path2:
        affinity_secondary, _, affinity_kwargs2 = load_frozen_affinity(affinity_path2, device)
        affinity_provenance.append({
            "path": affinity_path2, "sha256": checkpoint_sha256(affinity_path2), "model_kwargs": dict(affinity_kwargs2),
        })

    train_names, validation_names = train_val_split()
    if args.train_n:
        train_names = train_names[:args.train_n]
    if not train_names or not validation_names:
        raise RuntimeError("training or held-out split is empty")
    train_loader = _make_loader(CanvasDataset(train_names, real_prob=0.0, seed=args.seed), args.bs, args.workers, shuffle=True, device=device)
    validation_loader = _make_loader(
        CanvasDataset(validation_names, real_prob=0.0, seed=args.seed + 10_000), args.eval_bs, min(args.workers, 2), shuffle=False, device=device
    )
    model = ContextContinuationRanker(tile_size=FS, width=args.width, dropout=args.dropout, context_band=args.context_band).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"device={device} ContextContinuationRanker params={count_params(model):,} "
        f"top{args.candidate_k}/encoder encoders={len(affinity_provenance)} max_list={args.candidate_k * len(affinity_provenance)} "
        f"rows/image={args.rows_per_image} objective=listwise-CE(full B candidate row; exact synthetic A->B->C)",
        flush=True,
    )
    for index, provenance in enumerate(affinity_provenance, start=1):
        print(f"frozen affinity[{index}]={provenance['path']} sha256={str(provenance['sha256'])[:12]}", flush=True)

    best = -float("inf")
    started = time.time()
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        batch, iterator = _next_batch(iterator, train_loader)
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("context-rank training requires exact synthetic CanvasDataset examples")
        tiles = batch["tiles"].to(device, non_blocking=True)
        perm = batch["perm"].to(device, non_blocking=True).long()
        candidates, valid = mine_affinity_candidates(
            affinity, tiles, candidate_k=args.candidate_k, device=device, affinity_secondary=affinity_secondary
        )
        middles, targets, exists = continuation_targets(perm)
        slots, available = continuation_target_slots(candidates, valid, middles, targets, exists)
        rows = select_continuation_rows(
            middles, targets, slots, available, rows_per_image=args.rows_per_image, random_sample=True
        )
        if not rows.count:
            raise RuntimeError("frozen affinity graph retained no true continuation target in this training batch")
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            scores = score_continuation_rows(
                model, tiles, candidates, valid, rows, pair_batch=args.train_pair_batch, checkpoint_chunks=True
            )
        loss = listwise_cross_entropy(scores, rows.target_slots)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % 25 == 0:
            train_metrics = finalize_continuation_metrics(continuation_rank_metric_sums(scores.detach().float(), rows.target_slots))
            coverage = float(available[exists].float().mean())
            elapsed = time.time() - started
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"train_r1={train_metrics['continuation_target_r1']:.4f} "
                f"train_r5={train_metrics['continuation_target_r5']:.4f} "
                f"train_mrr={train_metrics['continuation_target_mrr']:.4f} "
                f"candidate_coverage={coverage:.4f} rows={rows.count} lr={scheduler.get_last_lr()[0]:.3e} {elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model, affinity, validation_loader, candidate_k=args.candidate_k, max_images=args.eval_n,
                rows_per_image=args.eval_rows_per_image, pair_batch=args.pair_batch, device=device,
                affinity_secondary=affinity_secondary,
            )
            print(f"[SYN context-continuation held-out] step={step} {_format(metrics)}", flush=True)
            last_path = os.path.join(args.out_dir, f"{args.tag}_last.pt")
            _checkpoint(last_path, model, optimizer, scheduler, step=step, args=args, metrics=metrics, affinity_provenance=affinity_provenance)
            gate = metrics["continuation_target_r1_all_true_proxy"]
            if gate > best:
                best = gate
                best_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
                _checkpoint(best_path, model, optimizer, scheduler, step=step, args=args, metrics=metrics, affinity_provenance=affinity_provenance)
                print(f"saved best continuation_target_r1_all_true_proxy={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
