"""Train the clean-structure auxiliary seam ranker on frozen hard lists.

This deliberately reuses the exact candidate graph and held-out rank metrics
from ``train_candidate_rank.py``.  The only changed variable is the model and
its clean luminance/gradient reconstruction auxiliary objective.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from candidate_rank import (
    candidate_target_slots,
    count_params,
    finalize_rank_metrics,
    listwise_cross_entropy,
    neighbor_targets,
    rank_metric_sums,
    score_candidate_rows,
    select_listwise_rows,
)
from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from imgio import train_val_split
from structural_seam import (
    StructuralSeamRanker,
    clean_structure_target,
    smoke as model_smoke,
    structural_reconstruction_loss,
)
from train_candidate_rank import (
    _make_loader,
    _next_batch,
    evaluate,
)
from train_offset_pose import checkpoint_sha256, load_frozen_affinity, mine_affinity_candidates


def _clean_tiles(clean: Tensor) -> Tensor:
    """Convert ``(B,3,480,480)`` clean canvases to row-major tiles."""
    if clean.ndim != 4 or tuple(clean.shape[1:]) != (3, GRID * FS, GRID * FS):
        raise ValueError(f"unexpected clean canvas shape {tuple(clean.shape)}")
    batch = clean.shape[0]
    return (
        clean.reshape(batch, 3, GRID, FS, GRID, FS)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, NFRAG, 3, FS, FS)
        .contiguous()
    )


def auxiliary_positive_loss(
    model: StructuralSeamRanker,
    dirty_tiles: Tensor,
    clean_canvas: Tensor,
    perm: Tensor,
    rows: Any,
) -> Tensor:
    """Reconstruct clean structure for the exact positive pair in each row."""
    image = rows.image_ids
    anchor = rows.anchors
    target = rows.target_indices
    clean = _clean_tiles(clean_canvas)
    anchor_cell = perm[image, anchor]
    target_cell = perm[image, target]
    clean_source = clean[image, anchor_cell]
    clean_target = clean[image, target_cell]
    dirty_source = dirty_tiles[image, anchor]
    dirty_target = dirty_tiles[image, target]
    _, prediction = model.forward_with_structure(
        dirty_source, dirty_target, rows.directions
    )
    structure = clean_structure_target(
        clean_source, clean_target, rows.directions
    )
    return structural_reconstruction_loss(prediction.float(), structure.float())


def _save(
    path: str,
    model: StructuralSeamRanker,
    *,
    step: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    affinity_provenance: list[dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_kwargs": model.model_kwargs,
            "model_type": "StructuralSeamRanker",
            "step": step,
            "metrics": metrics,
            "config": vars(args),
            "candidate_graph": affinity_provenance,
        },
        path,
    )


def initialize_from_candidate_ranker(
    model: StructuralSeamRanker,
    path: str,
) -> dict[str, Any]:
    """Transfer the old ranker while initially ignoring new structure channels."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload["model"]
    destination = model.state_dict()
    copied: list[str] = []
    for key, value in source.items():
        mapped = f"rank_head.{key[5:]}" if key.startswith("head.") else key
        if mapped == "stem.0.weight":
            if destination[mapped].shape[1] != 9 or value.shape[1] != 6:
                raise ValueError("unexpected old/new stem shapes during transfer")
            destination[mapped].zero_()
            destination[mapped][:, :6].copy_(value)
            copied.append(mapped)
        elif mapped in destination and destination[mapped].shape == value.shape:
            destination[mapped].copy_(value)
            copied.append(mapped)
    model.load_state_dict(destination, strict=True)
    required_prefixes = ("stem.", "block1.", "down.", "block2.", "rank_head.")
    missing = [
        prefix for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in copied)
    ]
    if missing:
        raise RuntimeError(f"candidate-ranker transfer missed modules: {missing}")
    return {
        "path": os.path.abspath(path),
        "source_step": int(payload.get("step", -1)),
        "source_metrics": payload.get("metrics", {}),
        "copied_tensors": len(copied),
    }


def _parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--affinity-ckpt",
        default=str(workspace / "artifacts/macro_affinity/affinity_r1_1200_best.pt"),
    )
    parser.add_argument(
        "--affinity-ckpt2",
        default=str(workspace / "artifacts/macro_affinity/affinity_r3_1000_best.pt"),
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--rows-per-image", type=int, default=48)
    parser.add_argument("--eval-n", type=int, default=8)
    parser.add_argument("--eval-bs", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-rows-per-image", type=int, default=192)
    parser.add_argument("--pair-batch", type=int, default=4096)
    parser.add_argument("--train-pair-batch", type=int, default=2048)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seam-band", type=int, default=6)
    parser.add_argument("--aux-weight", type=float, default=0.50)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="structural_seam")
    parser.add_argument("--out-dir", default=str(workspace / "artifacts/structural_seam"))
    parser.add_argument(
        "--init-candidate-checkpoint",
        default="",
        help="optional old CandidateSeamRanker checkpoint for exact ranking-path transfer",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("E:/pazzle_work/gates/structural_seam_gate.json"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    positive = (
        "steps",
        "bs",
        "candidate_k",
        "rows_per_image",
        "eval_n",
        "eval_bs",
        "eval_every",
        "eval_rows_per_image",
        "pair_batch",
        "train_pair_batch",
        "width",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error(f"{', '.join(positive)} must be positive")
    if args.workers < 0 or args.aux_weight < 0.0 or args.lr <= 0.0:
        parser.error("invalid workers, aux-weight, or lr")
    return args


def gate_result(metrics: dict[str, float]) -> dict[str, Any]:
    thresholds = {
        "candidate_target_r1": 0.35,
        "candidate_target_r5": 0.60,
        "reciprocal_exact_precision": 0.65,
        "candidate_target_r1_all_true_proxy": 0.24,
    }
    checks = {key: metrics[key] >= value for key, value in thresholds.items()}
    return {
        "thresholds": thresholds,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.smoke:
        result = model_smoke(device)
        # Exercise clean-canvas tiling as part of the trainer contract.
        tiled = _clean_tiles(torch.rand(2, 3, GRID * FS, GRID * FS, device=device))
        result["clean_tiles_shape"] = tuple(tiled.shape)
        print(f"[structural-seam trainer smoke] {result}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    affinity_path = os.path.abspath(args.affinity_ckpt)
    affinity, _, affinity_kwargs = load_frozen_affinity(affinity_path, device)
    affinity_secondary = None
    provenance: list[dict[str, Any]] = [
        {
            "path": affinity_path,
            "sha256": checkpoint_sha256(affinity_path),
            "model_kwargs": dict(affinity_kwargs),
        }
    ]
    if args.affinity_ckpt2:
        second_path = os.path.abspath(args.affinity_ckpt2)
        affinity_secondary, _, second_kwargs = load_frozen_affinity(second_path, device)
        provenance.append(
            {
                "path": second_path,
                "sha256": checkpoint_sha256(second_path),
                "model_kwargs": dict(second_kwargs),
            }
        )

    train_names, validation_names = train_val_split()
    train_data = CanvasDataset(train_names, real_prob=0.0, seed=args.seed)
    validation_data = CanvasDataset(
        validation_names, real_prob=0.0, seed=args.seed + 10_000
    )
    train_loader = _make_loader(
        train_data, args.bs, args.workers, shuffle=True, device=device
    )
    validation_loader = _make_loader(
        validation_data,
        args.eval_bs,
        min(args.workers, 2),
        shuffle=False,
        device=device,
    )
    model = StructuralSeamRanker(
        width=args.width,
        dropout=args.dropout,
        seam_band=args.seam_band,
    ).to(device)
    initialization = None
    if args.init_candidate_checkpoint:
        initialization = initialize_from_candidate_ranker(
            model, args.init_candidate_checkpoint
        )
        model.to(device)
        print(
            f"initialized ranking path from {initialization['path']} "
            f"step={initialization['source_step']} copied={initialization['copied_tensors']}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )
    rng = np.random.default_rng(args.seed + 919)
    iterator = iter(train_loader)
    best_selection = -math.inf
    best_metrics: dict[str, float] | None = None
    started = time.time()
    print(
        f"device={device} params={count_params(model):,} rows/image={args.rows_per_image} "
        f"aux_weight={args.aux_weight:g} candidate_union={len(provenance)}x{args.candidate_k}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        batch, iterator = _next_batch(iterator, train_loader)
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
        targets, exists = neighbor_targets(perm)
        target_slots, available = candidate_target_slots(
            candidates, valid, targets, exists
        )
        rows = select_listwise_rows(
            targets,
            target_slots,
            available,
            rows_per_image=args.rows_per_image,
            random_sample=True,
        )
        if not rows.count:
            raise RuntimeError("candidate graph retained no exact target rows")

        optimizer.zero_grad(set_to_none=True)
        # FP32 is intentional: previous FP16 Siamese experiments produced
        # infinite gradients even though their losses remained finite.
        scores = score_candidate_rows(
            model,
            tiles,
            candidates,
            valid,
            rows,
            pair_batch=args.train_pair_batch,
            checkpoint_chunks=True,
        )
        rank_loss = listwise_cross_entropy(scores, rows.target_slots)
        aux_loss = auxiliary_positive_loss(model, tiles, clean, perm, rows)
        loss = rank_loss + args.aux_weight * aux_loss
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 20 == 0:
            train_metrics = finalize_rank_metrics(
                rank_metric_sums(scores.detach(), rows.target_slots)
            )
            print(
                f"step {step}/{args.steps} total={float(loss.detach()):.4f} "
                f"rank={float(rank_loss.detach()):.4f} aux={float(aux_loss.detach()):.4f} "
                f"r1={train_metrics['candidate_target_r1']:.3f} "
                f"r5={train_metrics['candidate_target_r5']:.3f} "
                f"grad={float(gradient_norm):.3f} lr={scheduler.get_last_lr()[0]:.3e} "
                f"{(time.time() - started) / step:.2f}s/it",
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
            selection = metrics["candidate_target_r1_all_true_proxy"]
            print(
                f"[held-out] step={step} conditional_r1={metrics['candidate_target_r1']:.4f} "
                f"r5={metrics['candidate_target_r5']:.4f} "
                f"all_true_r1={selection:.4f} reciprocal_p={metrics['reciprocal_exact_precision']:.4f}",
                flush=True,
            )
            _save(
                os.path.join(args.out_dir, f"{args.tag}_last.pt"),
                model,
                step=step,
                metrics=metrics,
                args=args,
                affinity_provenance=provenance,
            )
            if selection > best_selection:
                best_selection = selection
                best_metrics = metrics
                _save(
                    os.path.join(args.out_dir, f"{args.tag}_best.pt"),
                    model,
                    step=step,
                    metrics=metrics,
                    args=args,
                    affinity_provenance=provenance,
                )
                print(f"saved best all_true_r1={selection:.4f}", flush=True)

    if best_metrics is None:
        raise RuntimeError("training finished without held-out evaluation")
    gate = gate_result(best_metrics)
    report = {
        "experiment": "clean_structure_auxiliary_seam_ranker",
        "best_metrics": best_metrics,
        "gate": gate,
        "initialization": initialization,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"=== structural seam gate {'PASSED' if gate['pass'] else 'FAILED'} ===")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"report saved to {args.report}")


if __name__ == "__main__":
    main()
