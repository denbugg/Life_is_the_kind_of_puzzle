"""Train the generative-contrastive continuation gate on exact synthetic chains.

This is deliberately not another seam classifier. For an oracle-correct noisy
A -> B chain, the model must predict clean C and retrieve noisy C from the
frozen affinity-union candidates attached to B. The retrieval path is a
query(A,B) dot key(C) listwise InfoNCE objective; candidates are encoded alone,
never cross-encoded with the chain.

Typical guarded GPU run after the data-free smoke:

    python src/train_continuation_predictor.py --steps 1000 --bs 2 ^
      --recon-rows-per-image 48 --rank-rows-per-image 24 --device cuda

The held-out report always separates:

* predicted_clean_l1: clean RGB prediction on all true chains by default;
* continuation_candidate_coverage_all_true: frozen graph ceiling;
* R@1/R@5/MRR: conditional on retaining true C in B's list;
* all-true proxies: coverage multiplied by conditional retrieval.
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
from config import FS, NFRAG, SEED
from context_rank import continuation_target_slots, continuation_targets, select_continuation_rows
from continuation_predictor import (
    ContinuationPredictor,
    clean_targets_for_rows,
    continuation_rank_metric_sums,
    count_params,
    finalize_continuation_metrics,
    listwise_info_nce,
    predict_clean_rows,
    score_candidate_rows,
    select_oracle_chain_rows,
    smoke,
)
from imgio import train_val_split
from train_offset_pose import checkpoint_sha256, load_frozen_affinity, mine_affinity_candidates


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


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
    iterator: Iterable[dict[str, Tensor]],
    loader: DataLoader,
) -> tuple[dict[str, Tensor], Any]:
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
        "predicted_clean_l1",
        "continuation_candidate_coverage_all_true",
        "continuation_target_r1",
        "continuation_target_r5",
        "continuation_target_mrr",
        "continuation_target_r1_all_true_proxy",
        "continuation_target_r5_all_true_proxy",
        "continuation_target_cross_entropy",
        "continuation_rank_rows",
        "clean_prediction_rows",
        "eval_images",
    )
    seen: set[str] = set()
    parts: list[str] = []
    for key in order:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
            seen.add(key)
    parts.extend(f"{key}={value:.4f}" for key, value in metrics.items() if key not in seen)
    return " ".join(parts)


@torch.inference_mode()
def evaluate(
    model: ContinuationPredictor,
    affinity: nn.Module,
    loader: DataLoader,
    *,
    candidate_k: int,
    max_images: int,
    rank_rows_per_image: int,
    reconstruction_rows_per_image: int | None,
    context_batch: int,
    candidate_batch: int,
    device: torch.device,
    affinity_secondary: nn.Module | None,
) -> dict[str, float]:
    """Evaluate generation and retrieval with distinct honest denominators.

    Clean L1 uses all exact oracle chains unless a positive deterministic
    reconstruction_rows_per_image is requested. Retrieval scores only rows for
    which C is actually in B's frozen candidate list. Candidate coverage is
    measured over every true chain before that conditioning.
    """
    was_training = model.training
    model.eval()
    total: defaultdict[str, float] = defaultdict(float)
    seen = 0
    for batch in loader:
        if seen >= max_images:
            break
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("continuation predictor validation requires synthetic CanvasDataset examples")
        take = min(max_images - seen, int(batch["tiles"].shape[0]))
        tiles = batch["tiles"][:take].to(device, non_blocking=True)
        clean = batch["clean"][:take].to(device, non_blocking=True)
        perm = batch["perm"][:take].to(device, non_blocking=True).long()
        candidates, valid = mine_affinity_candidates(
            affinity,
            tiles,
            candidate_k=candidate_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        middles, targets, exists = continuation_targets(perm)
        target_slots, available = continuation_target_slots(candidates, valid, middles, targets, exists)

        # This branch deliberately does not filter on candidate availability.
        reconstruction_rows = select_oracle_chain_rows(
            middles,
            targets,
            exists,
            rows_per_image=reconstruction_rows_per_image,
            random_sample=False,
        )
        with _autocast(device):
            prediction = predict_clean_rows(
                model, tiles, reconstruction_rows, context_batch=context_batch
            )
        target_clean = clean_targets_for_rows(clean, perm, reconstruction_rows)
        total["clean_abs_sum"] += float((prediction.float() - target_clean.float()).abs().sum())
        total["clean_values"] += float(target_clean.numel())
        total["clean_rows"] += float(reconstruction_rows.count)

        # This branch is conditioned only on the fixed candidate graph
        # retaining true C. It uses deterministic direction-balanced rows.
        rank_rows = select_continuation_rows(
            middles,
            targets,
            target_slots,
            available,
            rows_per_image=rank_rows_per_image,
            random_sample=False,
        )
        if rank_rows.count:
            with _autocast(device):
                scores = score_candidate_rows(
                    model,
                    tiles,
                    candidates,
                    valid,
                    rank_rows,
                    candidate_batch=candidate_batch,
                    checkpoint_chunks=False,
                )
            _append(
                total,
                continuation_rank_metric_sums(scores.float(), rank_rows.target_slots),
            )
        total["true_chains"] += float(exists.sum())
        total["covered_chains"] += float(available.sum())
        total["selected_rank_rows"] += float(rank_rows.count)
        seen += take

    if was_training:
        model.train()
    if not seen:
        raise RuntimeError("evaluation loader yielded no images")
    metrics = finalize_continuation_metrics(dict(total))
    coverage = _ratio(total["covered_chains"], total["true_chains"])
    metrics.update(
        {
            "predicted_clean_l1": _ratio(total["clean_abs_sum"], total["clean_values"]),
            "clean_prediction_rows": total["clean_rows"],
            "continuation_candidate_coverage_all_true": coverage,
            "continuation_target_r1_all_true_proxy": coverage * metrics["continuation_target_r1"],
            "continuation_target_r5_all_true_proxy": coverage * metrics["continuation_target_r5"],
            "continuation_true_chains": total["true_chains"],
            "continuation_covered_chains": total["covered_chains"],
            "continuation_selected_rank_rows": total["selected_rank_rows"],
            "eval_images": float(seen),
        }
    )
    return metrics


def _checkpoint(
    path: str,
    model: ContinuationPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    args: argparse.Namespace,
    metrics: Mapping[str, float],
    affinity_provenance: list[Mapping[str, Any]],
) -> None:
    """Save a self-describing checkpoint, including the no-cross-encoder contract."""
    torch.save(
        {
            "schema_version": 1,
            "experiment": "generative_contrastive_continuation_predictor",
            "model": model.state_dict(),
            "model_kwargs": {
                "tile_size": model.tile_size,
                "width": model.width,
                "embedding_dim": model.embedding_dim,
                "dropout": model.dropout,
                "temperature": model.initial_temperature,
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
                "candidate_row_owner": "oracle middle B from exact A->B chain",
                "candidate_target": "same-direction next C after B",
                "coverage_denominator": "all valid A->B->C chains",
            },
            "objective": {
                "clean_generation": "RGB L1(predicted clean C, target clean C) on exact chains independent of graph coverage",
                "retrieval": "InfoNCE/listwise cross entropy over full valid frozen B candidate row only where true C is retained",
                "reconstruction_weight": float(args.reconstruction_weight),
                "retrieval_weight": float(args.retrieval_weight),
                "query": "canonical noisy A->B context encoder",
                "key": "separate canonical noisy C candidate encoder",
                "interaction": "normalized dot product only",
                "candidate_cross_encoder": False,
                "model_input": "raw RGB plus independent per-tile exposure-normalized RGB",
                "direction_handling": "canonical physical rotation; no direction embedding",
            },
            "supervision": {
                "source": "fresh CanvasDataset(real_prob=0) synthetic shuffle/distortion",
                "clean_target_input_to_model": False,
                "absolute_coordinates_input_to_model": False,
                "input_position_features": False,
                "oracle_context": "A->B defines this scoped continuation gate only",
            },
        },
        path,
    )


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
        help="secondary frozen affinity checkpoint; pass an empty string to disable union",
    )
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-n", "--train_n", dest="train_n", type=int, default=0)
    parser.add_argument("--candidate-k", "--candidate_k", dest="candidate_k", type=int, default=64)
    parser.add_argument(
        "--recon-rows-per-image",
        "--recon_rows_per_image",
        dest="recon_rows_per_image",
        type=int,
        default=48,
        help="exact oracle A->B->C rows/image for graph-independent clean L1 training",
    )
    parser.add_argument(
        "--rank-rows-per-image",
        "--rank_rows_per_image",
        dest="rank_rows_per_image",
        type=int,
        default=24,
        help="candidate-covered oracle rows/image for listwise InfoNCE training",
    )
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--embedding-dim", "--embedding_dim", dest="embedding_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=4.0e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--reconstruction-weight",
        "--reconstruction_weight",
        dest="reconstruction_weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--retrieval-weight",
        "--retrieval_weight",
        dest="retrieval_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument("--eval-n", "--eval_n", dest="eval_n", type=int, default=4)
    parser.add_argument("--eval-bs", "--eval_bs", dest="eval_bs", type=int, default=1)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=100)
    parser.add_argument(
        "--eval-rank-rows-per-image",
        "--eval_rank_rows_per_image",
        dest="eval_rank_rows_per_image",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--eval-recon-rows-per-image",
        "--eval_recon_rows_per_image",
        dest="eval_recon_rows_per_image",
        type=int,
        default=0,
        help="0 means all true chains; positive values select a deterministic balanced subset",
    )
    parser.add_argument(
        "--context-batch",
        "--context_batch",
        dest="context_batch",
        type=int,
        default=512,
        help="maximum A/B context rows per decoder forward",
    )
    parser.add_argument(
        "--candidate-batch",
        "--candidate_batch",
        dest="candidate_batch",
        type=int,
        default=2048,
        help="maximum candidate-only keys per encoder chunk",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="continuation_predictor")
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        default=os.path.join(workspace, "artifacts", "continuation_predictor"),
    )
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument(
        "--no-checkpoint-chunks",
        dest="checkpoint_chunks",
        action="store_false",
        help="disable candidate-encoder gradient checkpointing",
    )
    parser.set_defaults(checkpoint_chunks=True)
    parser.add_argument(
        "--tiny-smoke",
        "--tiny_smoke",
        action="store_true",
        help="run the data-free CPU-safe contract smoke and exit",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.bs < 1 or args.eval_n < 1 or args.eval_bs < 1:
        parser.error("--steps, --bs, --eval-n, and --eval-bs must be positive")
    if args.workers < 0 or args.train_n < 0:
        parser.error("--workers and --train-n must be non-negative")
    if not 1 <= args.candidate_k < NFRAG:
        parser.error(f"--candidate-k must lie in [1,{NFRAG - 1}]")
    for value, name in (
        (args.recon_rows_per_image, "--recon-rows-per-image"),
        (args.rank_rows_per_image, "--rank-rows-per-image"),
        (args.eval_rank_rows_per_image, "--eval-rank-rows-per-image"),
    ):
        if value < 4:
            parser.error(f"{name} must be at least four for direction-balanced sampling")
    if args.eval_recon_rows_per_image and args.eval_recon_rows_per_image < 4:
        parser.error("--eval-recon-rows-per-image must be zero or at least four")
    if (
        args.width < 4
        or args.embedding_dim < 8
        or args.context_batch < 1
        or args.candidate_batch < 1
        or args.eval_every < 1
    ):
        parser.error("invalid model/batch/evaluation value")
    if (
        args.temperature <= 0.0
        or args.lr <= 0.0
        or args.weight_decay < 0.0
        or args.reconstruction_weight < 0.0
        or args.retrieval_weight <= 0.0
        or not 0.0 <= args.dropout < 1.0
    ):
        parser.error("invalid optimizer, loss weight, temperature, or dropout")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.tiny_smoke:
        print(f"[continuation-predictor tiny smoke] device={device} {smoke(device)}", flush=True)
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
    affinity_provenance: list[Mapping[str, Any]] = [
        {
            "path": affinity_path,
            "sha256": checkpoint_sha256(affinity_path),
            "model_kwargs": dict(affinity_kwargs),
        }
    ]
    affinity_secondary: nn.Module | None = None
    if affinity_path2:
        affinity_secondary, _, affinity_kwargs2 = load_frozen_affinity(affinity_path2, device)
        affinity_provenance.append(
            {
                "path": affinity_path2,
                "sha256": checkpoint_sha256(affinity_path2),
                "model_kwargs": dict(affinity_kwargs2),
            }
        )

    train_names, validation_names = train_val_split()
    if args.train_n:
        train_names = train_names[:args.train_n]
    if not train_names or not validation_names:
        raise RuntimeError("training or held-out split is empty")
    train_loader = _make_loader(
        CanvasDataset(train_names, real_prob=0.0, seed=args.seed),
        args.bs,
        args.workers,
        shuffle=True,
        device=device,
    )
    validation_loader = _make_loader(
        CanvasDataset(validation_names, real_prob=0.0, seed=args.seed + 10_000),
        args.eval_bs,
        min(args.workers, 2),
        shuffle=False,
        device=device,
    )

    model = ContinuationPredictor(
        tile_size=FS,
        width=args.width,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"device={device} ContinuationPredictor params={count_params(model):,} "
        f"top{args.candidate_k}/encoder encoders={len(affinity_provenance)} "
        f"max_list={args.candidate_k * len(affinity_provenance)} "
        f"recon_rows/image={args.recon_rows_per_image} rank_rows/image={args.rank_rows_per_image} "
        f"objective={args.reconstruction_weight:g}*clean_L1 + {args.retrieval_weight:g}*listwise_InfoNCE",
        flush=True,
    )
    print(
        "contract=canonical noisy A->B query; separate canonical noisy C key; "
        "dot-product-only retrieval; no candidate cross-encoder or coordinates",
        flush=True,
    )
    for index, provenance in enumerate(affinity_provenance, start=1):
        print(
            f"frozen affinity[{index}]={provenance['path']} "
            f"sha256={str(provenance['sha256'])[:12]}",
            flush=True,
        )

    best = -float("inf")
    started = time.time()
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        batch, iterator = _next_batch(iterator, train_loader)
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("training requires exact synthetic CanvasDataset examples")
        tiles = batch["tiles"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        perm = batch["perm"].to(device, non_blocking=True).long()
        candidates, valid = mine_affinity_candidates(
            affinity,
            tiles,
            candidate_k=args.candidate_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        middles, targets, exists = continuation_targets(perm)
        target_slots, available = continuation_target_slots(candidates, valid, middles, targets, exists)

        # Clean reconstruction sees exact chains regardless of whether the
        # frozen graph happened to retain C. Retrieval is separately and
        # correctly conditioned on retained C rows.
        reconstruction_rows = select_oracle_chain_rows(
            middles,
            targets,
            exists,
            rows_per_image=args.recon_rows_per_image,
            random_sample=True,
        )
        rank_rows = select_continuation_rows(
            middles,
            targets,
            target_slots,
            available,
            rows_per_image=args.rank_rows_per_image,
            random_sample=True,
        )
        if not reconstruction_rows.count:
            raise RuntimeError("no exact continuation rows available for clean reconstruction")
        if not rank_rows.count:
            raise RuntimeError("frozen affinity graph retained no C target in this training batch")

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            clean_prediction = predict_clean_rows(
                model, tiles, reconstruction_rows, context_batch=args.context_batch
            )
            clean_target = clean_targets_for_rows(clean, perm, reconstruction_rows)
            reconstruction_loss = F.l1_loss(clean_prediction, clean_target)
            scores = score_candidate_rows(
                model,
                tiles,
                candidates,
                valid,
                rank_rows,
                candidate_batch=args.candidate_batch,
                checkpoint_chunks=args.checkpoint_chunks,
            )
            retrieval_loss = listwise_info_nce(scores, rank_rows.target_slots)
            loss = (
                float(args.reconstruction_weight) * reconstruction_loss
                + float(args.retrieval_weight) * retrieval_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % 25 == 0:
            train_rank = finalize_continuation_metrics(
                continuation_rank_metric_sums(scores.detach().float(), rank_rows.target_slots)
            )
            coverage = float(available[exists].float().mean())
            elapsed = time.time() - started
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"clean_l1={float(reconstruction_loss.detach()):.4f} "
                f"rank_ce={float(retrieval_loss.detach()):.4f} "
                f"train_r1={train_rank['continuation_target_r1']:.4f} "
                f"train_r5={train_rank['continuation_target_r5']:.4f} "
                f"train_mrr={train_rank['continuation_target_mrr']:.4f} "
                f"candidate_coverage={coverage:.4f} recon_rows={reconstruction_rows.count} "
                f"rank_rows={rank_rows.count} lr={scheduler.get_last_lr()[0]:.3e} "
                f"{elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                affinity,
                validation_loader,
                candidate_k=args.candidate_k,
                max_images=args.eval_n,
                rank_rows_per_image=args.eval_rank_rows_per_image,
                reconstruction_rows_per_image=(
                    None if args.eval_recon_rows_per_image <= 0 else args.eval_recon_rows_per_image
                ),
                context_batch=args.context_batch,
                candidate_batch=args.candidate_batch,
                device=device,
                affinity_secondary=affinity_secondary,
            )
            print(f"[SYN generative continuation held-out] step={step} {_format(metrics)}", flush=True)
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
            gate = metrics["continuation_target_r1_all_true_proxy"]
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
                print(
                    f"saved best continuation_target_r1_all_true_proxy={best:.4f} "
                    f"clean_l1={metrics['predicted_clean_l1']:.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
