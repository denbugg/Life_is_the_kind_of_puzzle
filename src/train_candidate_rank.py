"""Train a listwise physical-seam ranker on frozen affinity candidate lists.

This is intentionally not another independent edge/non-edge classifier.  For
one synthetic source tile and one cardinal direction, it scores the complete
deduplicated union of the two frozen affinity top-64 lists and optimizes the
true neighbour's position inside that hard list.

Typical guarded run (after the data-free smoke succeeds)::

    python src/train_candidate_rank.py --steps 800 --bs 2 --rows-per-image 48 ^
      --eval-every 100 --eval-n 8 --device cuda

The trainable CNN never receives a 576 x 576 score matrix.  At the default
48 rows/image and a maximum 128 candidates/row, it sees at most 6,144
oriented seams per puzzle bag per optimizer step.
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

from candidate_rank import (
    CandidateSeamRanker,
    ListwiseRows,
    candidate_target_slots,
    count_params,
    finalize_rank_metrics,
    inverse_directions,
    listwise_cross_entropy,
    neighbor_targets,
    rank_metric_sums,
    score_candidate_rows,
    select_listwise_rows,
)
from canvas_data import CanvasDataset
from config import FS, NFRAG, SEED
from imgio import train_val_split
from train_offset_pose import checkpoint_sha256, load_frozen_affinity, mine_affinity_candidates


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def _make_loader(
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


def _next_batch(
    iterator: Iterable[dict[str, Tensor]], loader: DataLoader
) -> tuple[dict[str, Tensor], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _format(metrics: Mapping[str, float]) -> str:
    order = (
        "candidate_target_coverage_all_true",
        "candidate_target_r1",
        "candidate_target_r5",
        "candidate_target_mrr",
        "candidate_target_r1_all_true_proxy",
        "reciprocal_exact_precision",
        "reciprocal_exact_coverage_candidate",
        "reciprocal_true_mutual_candidate_coverage",
        "candidate_target_cross_entropy",
        "candidate_rank_rows",
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


def _append_sums(total: defaultdict[str, float], values: Mapping[str, float]) -> None:
    for key, value in values.items():
        total[key] += float(value)


@torch.inference_mode()
def _reciprocal_counts(
    model: CandidateSeamRanker,
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    rows: ListwiseRows,
    scores: Tensor,
    *,
    pair_batch: int,
) -> dict[str, float]:
    """Measure exact reciprocal cycles for selected candidate-ranked rows.

    The reverse row is built from the *predicted* target, not the known truth.
    Thus a successful cycle is a usable graph edge at inference.  Its precision
    is exact because the held-out synthetic permutation is consulted only for
    the final metric, never as an input to the model.
    """
    if rows.count == 0:
        return {
            "reciprocal_predicted": 0.0,
            "reciprocal_exact": 0.0,
            "reciprocal_true_mutual_candidate": 0.0,
        }
    predicted_slots = scores.argmax(dim=-1)
    predicted_targets = candidates[rows.image_ids, rows.anchors, predicted_slots]
    reverse_directions = inverse_directions(rows.directions)
    reverse_candidates = candidates[rows.image_ids, predicted_targets]
    reverse_valid = valid[rows.image_ids, predicted_targets]
    reverse_match = reverse_valid & reverse_candidates.eq(rows.anchors[:, None])
    reverse_present = reverse_match.any(dim=-1)

    mutual = torch.zeros(rows.count, dtype=torch.bool, device=tiles.device)
    if bool(reverse_present.any()):
        reverse_slots = reverse_match.long().argmax(dim=-1)
        selected = torch.nonzero(reverse_present, as_tuple=False).flatten()
        reverse_rows = ListwiseRows(
            rows.image_ids[selected],
            predicted_targets[selected],
            reverse_directions[selected],
            reverse_slots[selected],
            rows.anchors[selected],
        )
        reverse_scores = score_candidate_rows(
            model, tiles, candidates, valid, reverse_rows, pair_batch=pair_batch
        )
        reverse_prediction = reverse_scores.argmax(dim=-1)
        reverse_target = candidates[
            reverse_rows.image_ids, reverse_rows.anchors, reverse_prediction
        ]
        mutual[selected] = reverse_target.eq(rows.anchors[selected])

    forward_exact = predicted_targets.eq(rows.target_indices)
    true_reverse_candidates = candidates[rows.image_ids, rows.target_indices]
    true_reverse_valid = valid[rows.image_ids, rows.target_indices]
    true_mutual_candidate = (
        true_reverse_valid & true_reverse_candidates.eq(rows.anchors[:, None])
    ).any(dim=-1)
    return {
        "reciprocal_predicted": float(mutual.sum()),
        "reciprocal_exact": float((mutual & forward_exact).sum()),
        "reciprocal_true_mutual_candidate": float(true_mutual_candidate.sum()),
    }


@torch.inference_mode()
def evaluate(
    model: CandidateSeamRanker,
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
    """Evaluate listwise rank and reciprocal exact-edge gates on fresh synthetic bags.

    Evaluation intentionally samples a bounded number of complete hard rows
    per image.  A full 576 x 576 cross-encoder evaluation is neither needed
    nor representative of the candidate-distribution deployment path.
    """
    was_training = model.training
    model.eval()
    total: defaultdict[str, float] = defaultdict(float)
    seen = 0
    for batch in loader:
        if seen >= max_images:
            break
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("candidate-rank validation requires CanvasDataset(real_prob=0)")
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
        targets, exists = neighbor_targets(perm)
        target_slots, available = candidate_target_slots(candidates, valid, targets, exists)
        rows = select_listwise_rows(
            targets,
            target_slots,
            available,
            rows_per_image=rows_per_image,
            random_sample=False,
        )
        if rows.count:
            with _autocast(device):
                scores = score_candidate_rows(
                    model, tiles, candidates, valid, rows, pair_batch=pair_batch
                )
            _append_sums(total, rank_metric_sums(scores.float(), rows.target_slots))
            _append_sums(
                total,
                _reciprocal_counts(
                    model,
                    tiles,
                    candidates,
                    valid,
                    rows,
                    scores.float(),
                    pair_batch=pair_batch,
                ),
            )
        total["candidate_true_rows"] += float(exists.sum())
        total["candidate_covered_rows"] += float(available.sum())
        total["selected_rows"] += float(rows.count)
        seen += take
    if was_training:
        model.train()
    if not seen:
        raise RuntimeError("evaluation loader yielded no images")

    metrics = finalize_rank_metrics(dict(total))
    coverage = _ratio(total["candidate_covered_rows"], total["candidate_true_rows"])
    metrics.update(
        {
            # The frozen graph's recall is reported separately from the seam
            # ranker: R@1/R@5 above are conditional on a target being proposed.
            "candidate_target_coverage_all_true": coverage,
            "candidate_target_r1_all_true_proxy": coverage * metrics["candidate_target_r1"],
            "candidate_target_r5_all_true_proxy": coverage * metrics["candidate_target_r5"],
            "reciprocal_predicted_edges_per_row": _ratio(
                total["reciprocal_predicted"], total["selected_rows"]
            ),
            "reciprocal_exact_precision": _ratio(
                total["reciprocal_exact"], total["reciprocal_predicted"]
            ),
            "reciprocal_exact_coverage_candidate": _ratio(
                total["reciprocal_exact"], total["selected_rows"]
            ),
            "reciprocal_true_mutual_candidate_coverage": _ratio(
                total["reciprocal_true_mutual_candidate"], total["selected_rows"]
            ),
            "candidate_true_rows": total["candidate_true_rows"],
            "candidate_covered_rows": total["candidate_covered_rows"],
            "selected_rows": total["selected_rows"],
            "eval_images": float(seen),
        }
    )
    return metrics


def _checkpoint(
    path: str,
    model: CandidateSeamRanker,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    args: argparse.Namespace,
    metrics: Mapping[str, float],
    affinity_provenance: list[Mapping[str, Any]],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "model_kwargs": {
                "tile_size": model.tile_size,
                "width": model.width,
                "dropout": model.dropout,
                "seam_band": model.seam_band,
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
                "training_objective": "full-hard-list cross entropy for known U/D/L/R target",
                "orientation": "source-left target-right canonical rotations",
            },
        },
        path,
    )


def _perfect_candidate_graph(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Tiny smoke candidate graph containing all true cardinal neighbours."""
    perm = torch.arange(NFRAG, device=device, dtype=torch.long).unsqueeze(0)
    targets, exists = neighbor_targets(perm)
    anchors = torch.arange(NFRAG, device=device, dtype=torch.long)
    candidates = torch.empty((1, NFRAG, 5), device=device, dtype=torch.long)
    valid = torch.zeros_like(candidates, dtype=torch.bool)
    candidates[0, :, :4] = targets[0].clamp_min(0)
    valid[0, :, :4] = exists[0]
    # A safe distant distractor, so every valid list has at least two entries.
    candidates[0, :, 4] = torch.remainder(anchors + NFRAG // 2, NFRAG)
    valid[0, :, 4] = True
    return perm, candidates, valid, targets


def tiny_smoke(device: torch.device) -> dict[str, float]:
    """Data-free labels, full-list loss, gradient, and rank metric smoke test."""
    torch.manual_seed(9876)
    perm, candidates, valid, targets = _perfect_candidate_graph(device)
    _, exists = neighbor_targets(perm)
    slots, available = candidate_target_slots(candidates, valid, targets, exists)
    if float(available[exists].float().mean()) < 0.99:
        raise AssertionError("perfect candidate graph failed to retain direct targets")
    rows = select_listwise_rows(
        targets, slots, available, rows_per_image=16, random_sample=False
    )
    model = CandidateSeamRanker(width=8, dropout=0.0).to(device)
    tiles = torch.rand(1, NFRAG, 3, FS, FS, device=device)
    # checkpoint_chunks is the training path; the smoke must exercise its
    # gradient flow, not only the plain chunk loop used at evaluation.
    scores = score_candidate_rows(
        model, tiles, candidates, valid, rows, pair_batch=13, checkpoint_chunks=True
    )
    loss = listwise_cross_entropy(scores, rows.target_slots)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    perfect_scores = torch.full_like(scores, -8.0)
    perfect_scores.scatter_(1, rows.target_slots[:, None], 8.0)
    metrics = finalize_rank_metrics(rank_metric_sums(perfect_scores, rows.target_slots))
    if metrics["candidate_target_r1"] < 0.999 or metrics["candidate_target_r5"] < 0.999:
        raise AssertionError(f"perfect listwise rank metric failed: {metrics}")
    return {
        "rows": float(rows.count),
        "sample_loss": float(loss.detach()),
        "perfect_target_r1": metrics["candidate_target_r1"],
        "parameters": float(count_params(model)),
    }


def _parse_args() -> argparse.Namespace:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--affinity-ckpt",
        "--affinity_ckpt",
        dest="affinity_ckpt",
        default=os.path.join(workspace, "artifacts", "macro_affinity", "affinity_r1_1200_best.pt"),
        help="primary frozen MacroAffinityNet checkpoint",
    )
    parser.add_argument(
        "--affinity-ckpt2",
        "--affinity_ckpt2",
        dest="affinity_ckpt2",
        default=os.path.join(workspace, "artifacts", "macro_affinity", "affinity_r3_1000_best.pt"),
        help="secondary frozen MacroAffinityNet checkpoint; empty disables the union",
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--bs", type=int, default=2, help="synthetic puzzle bags per step")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-n", "--train_n", dest="train_n", type=int, default=0)
    parser.add_argument("--candidate-k", "--candidate_k", dest="candidate_k", type=int, default=64)
    parser.add_argument(
        "--rows-per-image",
        "--rows_per_image",
        dest="rows_per_image",
        type=int,
        default=48,
        help="direction-balanced full candidate rows sampled per bag per optimizer step",
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seam-band", "--seam_band", dest="seam_band", type=int, default=6)
    parser.add_argument("--lr", type=float, default=4.0e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--eval-n", "--eval_n", dest="eval_n", type=int, default=8)
    parser.add_argument("--eval-bs", "--eval_bs", dest="eval_bs", type=int, default=1)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=200)
    parser.add_argument(
        "--eval-rows-per-image",
        "--eval_rows_per_image",
        dest="eval_rows_per_image",
        type=int,
        default=192,
        help="bounded complete hard rows/image for held-out ranking and reciprocity",
    )
    parser.add_argument("--pair-batch", "--pair_batch", dest="pair_batch", type=int, default=4096)
    parser.add_argument(
        "--train-pair-batch",
        "--train_pair_batch",
        dest="train_pair_batch",
        type=int,
        default=2048,
        help="checkpointed training chunk; also the training-step activation memory cap",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="candidate_rank")
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        default=os.path.join(workspace, "artifacts", "candidate_rank"),
    )
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument(
        "--tiny-smoke",
        "--tiny_smoke",
        action="store_true",
        help="run a CPU/data-free listwise contract test and exit",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.bs < 1 or args.eval_n < 1 or args.eval_bs < 1:
        parser.error("--steps, --bs, --eval-n, and --eval-bs must be positive")
    if args.workers < 0 or args.train_n < 0:
        parser.error("--workers and --train-n must be non-negative")
    if not 1 <= args.candidate_k < NFRAG:
        parser.error(f"--candidate-k must lie in [1,{NFRAG - 1}]")
    if args.rows_per_image < 4 or args.eval_rows_per_image < 4 or args.pair_batch < 1:
        parser.error("row counts must be at least 4 and --pair-batch must be positive")
    if args.train_pair_batch < 1:
        parser.error("--train-pair-batch must be positive")
    if args.width < 4 or args.seam_band < 2 or args.lr <= 0.0 or args.weight_decay < 0.0:
        parser.error("invalid width/seam-band/lr/weight-decay value")
    if not 0.0 <= args.dropout < 1.0 or args.eval_every < 1:
        parser.error("--dropout must lie in [0,1) and --eval-every must be positive")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.tiny_smoke:
        print(f"[candidate-rank tiny smoke] device={device} {_format(tiny_smoke(device))}", flush=True)
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
    affinity, metadata, affinity_kwargs = load_frozen_affinity(affinity_path, device)
    del metadata
    affinity_provenance: list[Mapping[str, Any]] = [
        {
            "path": affinity_path,
            "sha256": checkpoint_sha256(affinity_path),
            "model_kwargs": dict(affinity_kwargs),
        }
    ]
    affinity_secondary: nn.Module | None = None
    if affinity_path2:
        affinity_secondary, metadata2, affinity_kwargs2 = load_frozen_affinity(affinity_path2, device)
        del metadata2
        affinity_provenance.append(
            {
                "path": affinity_path2,
                "sha256": checkpoint_sha256(affinity_path2),
                "model_kwargs": dict(affinity_kwargs2),
            }
        )

    train_names, validation_names = train_val_split()
    if args.train_n:
        train_names = train_names[: args.train_n]
    if not train_names or not validation_names:
        raise RuntimeError("training or held-out split is empty")
    train_dataset = CanvasDataset(train_names, real_prob=0.0, seed=args.seed)
    validation_dataset = CanvasDataset(validation_names, real_prob=0.0, seed=args.seed + 10_000)
    train_loader = _make_loader(train_dataset, args.bs, args.workers, shuffle=True, device=device)
    validation_loader = _make_loader(
        validation_dataset, args.eval_bs, min(args.workers, 2), shuffle=False, device=device
    )
    model = CandidateSeamRanker(
        tile_size=FS, width=args.width, dropout=args.dropout, seam_band=args.seam_band
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"device={device} CandidateSeamRanker params={count_params(model):,} "
        f"top{args.candidate_k}/encoder encoders={len(affinity_provenance)} "
        f"max_list={args.candidate_k * len(affinity_provenance)} rows/image={args.rows_per_image} "
        "objective=listwise-CE(full candidate row; synthetic exact U/D/L/R targets)",
        flush=True,
    )
    for index, item in enumerate(affinity_provenance, start=1):
        print(
            f"frozen affinity[{index}]={item['path']} sha256={str(item['sha256'])[:12]} "
            f"kwargs={item['model_kwargs']}",
            flush=True,
        )

    best = -float("inf")
    started = time.time()
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        batch, iterator = _next_batch(iterator, train_loader)
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("candidate-rank training requires exact synthetic CanvasDataset examples")
        tiles = batch["tiles"].to(device, non_blocking=True)
        perm = batch["perm"].to(device, non_blocking=True).long()
        candidates, valid = mine_affinity_candidates(
            affinity,
            tiles,
            candidate_k=args.candidate_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        targets, exists = neighbor_targets(perm)
        slots, available = candidate_target_slots(candidates, valid, targets, exists)
        rows = select_listwise_rows(
            targets,
            slots,
            available,
            rows_per_image=args.rows_per_image,
            random_sample=True,
        )
        if not rows.count:
            raise RuntimeError("frozen candidate graph retained no true direct rows in this training batch")
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            # Checkpointed chunks keep training memory at one chunk of
            # activations; without this a full step spills past 8 GB VRAM and
            # WDDM shared-memory paging dominates the step time.
            scores = score_candidate_rows(
                model,
                tiles,
                candidates,
                valid,
                rows,
                pair_batch=args.train_pair_batch,
                checkpoint_chunks=True,
            )
        loss = listwise_cross_entropy(scores, rows.target_slots)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % 25 == 0:
            train_metrics = finalize_rank_metrics(rank_metric_sums(scores.detach().float(), rows.target_slots))
            elapsed = time.time() - started
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"train_r1={train_metrics['candidate_target_r1']:.4f} "
                f"train_r5={train_metrics['candidate_target_r5']:.4f} "
                f"train_mrr={train_metrics['candidate_target_mrr']:.4f} "
                f"candidate_coverage={float(available.float().mean()):.4f} "
                f"rows={rows.count} lr={scheduler.get_last_lr()[0]:.3e} {elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                affinity,
                validation_loader,
                candidate_k=args.candidate_k,
                max_images=args.eval_n,
                rows_per_image=args.eval_rows_per_image,
                pair_batch=args.pair_batch,
                device=device,
                affinity_secondary=affinity_secondary,
            )
            print(f"[SYN candidate-rank held-out] step={step} {_format(metrics)}", flush=True)
            last_path = os.path.join(args.out_dir, f"{args.tag}_last.pt")
            _checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                step=step,
                args=args,
                metrics=metrics,
                affinity_provenance=affinity_provenance,
            )
            # A candidate miss cannot be repaired by any seam ranker.  The
            # product is therefore the graph-ready gate, while conditional R@1
            # remains visible as the model-quality diagnostic.
            gate = metrics["candidate_target_r1_all_true_proxy"]
            if gate > best:
                best = gate
                best_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
                _checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    args=args,
                    metrics=metrics,
                    affinity_provenance=affinity_provenance,
                )
                print(f"saved best candidate_target_r1_all_true_proxy={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
