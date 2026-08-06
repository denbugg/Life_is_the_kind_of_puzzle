"""Train a bounded neural residual on candidates that fool tile-mean TV.

The fixed TV term remains the dominant board energy.  The neural critic is
passed through ``tanh`` and receives a deliberately small bounded weight, so it
can only reorder candidates whose TV scores are close.  Training mines those
close or incorrectly ranked local permutations from a larger candidate pool.

Examples
--------

    python src/train_tv_residual_critic.py --smoke --device cuda
    python src/train_tv_residual_critic.py --steps 600 --device cuda
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
import torch.nn.functional as F
from torch import Tensor, nn

from config import CKPT_DIR, SEED
from global_critic import (
    HARD_NEAR_FAMILIES,
    GlobalStructuralCritic,
    count_params,
    tile_mean_tv_score,
)
from imgio import train_val_split
from train_global_critic import (
    _autocast,
    _negative_boards,
    _score_boards,
    _synthetic_correct_board,
)


def hybrid_scores(
    neural_scores: Tensor,
    tv_scores: Tensor,
    *,
    tv_scale: float,
    residual_bound: float,
) -> Tensor:
    """Return a higher-is-better score with a strictly bounded learned term."""
    if neural_scores.shape != tv_scores.shape:
        raise ValueError("neural_scores and tv_scores must have identical shapes")
    return tv_scale * tv_scores.float() + residual_bound * torch.tanh(neural_scores.float())


def mine_tv_hard_negatives(
    correct: Tensor,
    *,
    rng: np.random.Generator,
    pool_per_family: int,
    hard_count: int,
) -> tuple[Tensor, Tensor, tuple[str, ...]]:
    """Generate local corruptions and retain the lowest positive-minus-negative TV margins."""
    families = tuple(
        family
        for family in HARD_NEAR_FAMILIES
        for _ in range(pool_per_family)
    )
    candidates = _negative_boards(correct, families, rng)
    with torch.no_grad():
        positive_tv = tile_mean_tv_score(correct.unsqueeze(0))[0]
        candidate_tv = tile_mean_tv_score(candidates)
        margins = positive_tv - candidate_tv
        keep = torch.argsort(margins)[: min(hard_count, len(candidates))]
    selected = candidates.index_select(0, keep)
    selected_tv = candidate_tv.index_select(0, keep)
    selected_families = tuple(families[int(index)] for index in keep.cpu().tolist())
    return selected, selected_tv, selected_families


def residual_ranking_loss(
    neural_scores: Tensor,
    tv_scores: Tensor,
    *,
    tv_scale: float,
    residual_bound: float,
    margin: float,
    residual_penalty: float,
) -> tuple[Tensor, Tensor]:
    combined = hybrid_scores(
        neural_scores,
        tv_scores,
        tv_scale=tv_scale,
        residual_bound=residual_bound,
    )
    ranking = F.softplus(margin + combined[1:] - combined[0]).mean()
    bounded_residual = torch.tanh(neural_scores)
    penalty = residual_penalty * bounded_residual.square().mean()
    return ranking + penalty, combined


def _rates(tv_margin: np.ndarray, hybrid_margin: np.ndarray) -> dict[str, float | int]:
    tv_ok = tv_margin > 0.0
    hybrid_ok = hybrid_margin > 0.0
    failures = ~tv_ok
    return {
        "count": int(len(tv_margin)),
        "tv_accuracy": float(tv_ok.mean()),
        "hybrid_accuracy": float(hybrid_ok.mean()),
        "lift": float(hybrid_ok.mean() - tv_ok.mean()),
        "tv_failures": int(failures.sum()),
        "corrected": int(np.logical_and(failures, hybrid_ok).sum()),
        "broken": int(np.logical_and(tv_ok, ~hybrid_ok).sum()),
        "correction_rate": float(
            np.logical_and(failures, hybrid_ok).sum() / max(1, failures.sum())
        ),
        "break_rate": float(
            np.logical_and(tv_ok, ~hybrid_ok).sum() / max(1, tv_ok.sum())
        ),
        "mean_tv_margin": float(tv_margin.mean()),
        "mean_hybrid_margin": float(hybrid_margin.mean()),
    }


@torch.inference_mode()
def evaluate(
    model: GlobalStructuralCritic,
    names: list[str],
    *,
    device: torch.device,
    seed: int,
    negatives_per_family: int,
    board_batch: int,
    tv_scale: float,
    residual_bound: float,
    amp: bool,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    by_family: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        family: [] for family in HARD_NEAR_FAMILIES
    }

    for image_index, name in enumerate(names):
        rng = np.random.default_rng(seed + image_index * 104729)
        correct = _synthetic_correct_board(name, rng, device)
        positive_neural = _score_boards(
            model, correct.unsqueeze(0), board_batch=board_batch, amp=amp
        )[0]
        positive_tv = tile_mean_tv_score(correct.unsqueeze(0))[0]
        for family_index, family in enumerate(HARD_NEAR_FAMILIES):
            family_rng = np.random.default_rng(
                seed + image_index * 1_000_003 + family_index * 8191
            )
            candidates = _negative_boards(
                correct, (family,) * negatives_per_family, family_rng
            )
            candidate_neural = _score_boards(
                model, candidates, board_batch=board_batch, amp=amp
            )
            candidate_tv = tile_mean_tv_score(candidates)
            neural = torch.cat((positive_neural[None], candidate_neural))
            tv = torch.cat((positive_tv[None], candidate_tv))
            hybrid = hybrid_scores(
                neural,
                tv,
                tv_scale=tv_scale,
                residual_bound=residual_bound,
            )
            by_family[family].append(
                (
                    (positive_tv - candidate_tv).cpu().numpy(),
                    (hybrid[0] - hybrid[1:]).cpu().numpy(),
                )
            )

    if was_training:
        model.train()
    family_metrics: dict[str, dict[str, float | int]] = {}
    all_tv: list[np.ndarray] = []
    all_hybrid: list[np.ndarray] = []
    for family, rows in by_family.items():
        tv_margin = np.concatenate([row[0] for row in rows])
        hybrid_margin = np.concatenate([row[1] for row in rows])
        family_metrics[family] = _rates(tv_margin, hybrid_margin)
        all_tv.append(tv_margin)
        all_hybrid.append(hybrid_margin)
    overall = _rates(np.concatenate(all_tv), np.concatenate(all_hybrid))
    return {
        "families": family_metrics,
        "overall": overall,
        "images": len(names),
        "negatives_per_family": negatives_per_family,
    }


def gate_result(metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    result = {
        "minimum_lift": 0.02,
        "minimum_correction_rate": 0.15,
        "maximum_break_rate": 0.05,
        "lift_pass": bool(overall["lift"] >= 0.02),
        "correction_pass": bool(overall["correction_rate"] >= 0.15),
        "break_pass": bool(overall["break_rate"] <= 0.05),
    }
    result["pass"] = bool(
        result["lift_pass"] and result["correction_pass"] and result["break_pass"]
    )
    return result


def _model(args: argparse.Namespace, device: torch.device) -> GlobalStructuralCritic:
    return GlobalStructuralCritic(
        tile_width=args.tile_width,
        embedding_dim=args.embedding_dim,
        edge_width=args.edge_width,
        edge_dim=args.edge_dim,
        grid_width=args.grid_width,
        stats_dim=args.stats_dim,
        edge_band=args.edge_band,
        dilations=tuple(int(value) for value in args.dilations.split(",")),
        dropout=args.dropout,
    ).to(device)


def _save(
    path: str,
    model: GlobalStructuralCritic,
    *,
    step: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_kwargs": model.model_kwargs,
            "step": step,
            "metrics": metrics,
            "hybrid": {
                "tv_scale": args.tv_scale,
                "residual_bound": args.residual_bound,
            },
            "config": vars(args),
        },
        path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--images-per-step", type=int, default=2)
    parser.add_argument("--pool-per-family", type=int, default=12)
    parser.add_argument("--hard-count", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--eval-images", type=int, default=8)
    parser.add_argument("--eval-negatives", type=int, default=32)
    parser.add_argument("--board-batch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--tv-scale", type=float, default=1000.0)
    parser.add_argument("--residual-bound", type=float, default=0.30)
    parser.add_argument("--residual-penalty", type=float, default=0.01)
    parser.add_argument("--tile-width", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--edge-width", type=int, default=12)
    parser.add_argument("--edge-dim", type=int, default=24)
    parser.add_argument("--grid-width", type=int, default=64)
    parser.add_argument("--stats-dim", type=int, default=12)
    parser.add_argument("--edge-band", type=int, default=3)
    parser.add_argument("--dilations", default="1,2,4,8,12,4")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="tv_residual_critic")
    parser.add_argument("--ckpt-dir", default=CKPT_DIR)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("E:/pazzle_work/gates/tv_residual_critic_gate.json"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    positive = (
        "steps",
        "images_per_step",
        "pool_per_family",
        "hard_count",
        "eval_every",
        "eval_images",
        "eval_negatives",
        "board_batch",
    )
    if any(getattr(args, key) < 1 for key in positive):
        parser.error(f"{', '.join(positive)} must all be positive")
    if args.tv_scale <= 0.0 or args.residual_bound <= 0.0:
        parser.error("--tv-scale and --residual-bound must be positive")
    return args


def _smoke(device: torch.device) -> dict[str, Any]:
    neural = torch.tensor((0.2, -0.3, 0.7), device=device, requires_grad=True)
    tv = torch.tensor((-0.1, -0.1001, -0.101), device=device)
    loss, combined = residual_ranking_loss(
        neural,
        tv,
        tv_scale=1000.0,
        residual_bound=0.3,
        margin=0.05,
        residual_penalty=0.01,
    )
    loss.backward()
    if neural.grad is None or not torch.isfinite(neural.grad).all():
        raise AssertionError("bounded residual backward pass failed")
    # The 0.0009 TV advantage (0.9 after scaling) cannot be overturned by a
    # residual whose maximum pairwise swing is 0.6.
    if not combined[0] > combined[2]:
        raise AssertionError("residual bound failed to protect a confident TV decision")
    return {
        "loss": float(loss.detach()),
        "combined": [float(value) for value in combined.detach().cpu()],
    }


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.smoke:
        print(f"[tv-residual smoke] {_smoke(device)}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_names, val_names = train_val_split()
    eval_names = val_names[: args.eval_images]
    model = _model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    checkpoint = os.path.join(args.ckpt_dir, f"{args.tag}_best.pt")
    last_checkpoint = os.path.join(args.ckpt_dir, f"{args.tag}_last.pt")
    rng = np.random.default_rng(args.seed + 811)
    best_lift = -math.inf
    best_metrics: dict[str, Any] | None = None
    started = time.time()
    print(
        f"device={device} params={count_params(model):,} images/step={args.images_per_step} "
        f"pool={args.pool_per_family * len(HARD_NEAR_FAMILIES)} keep={args.hard_count} "
        f"hybrid={args.tv_scale:g}*TV+{args.residual_bound:g}*tanh(neural)",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        train_tv_margins: list[float] = []
        train_hybrid_margins: list[float] = []
        for _ in range(args.images_per_step):
            name = train_names[int(rng.integers(0, len(train_names)))]
            sample_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
            correct = _synthetic_correct_board(name, sample_rng, device)
            negatives, negative_tv, _ = mine_tv_hard_negatives(
                correct,
                rng=sample_rng,
                pool_per_family=args.pool_per_family,
                hard_count=args.hard_count,
            )
            boards = torch.cat((correct.unsqueeze(0), negatives))
            neural = _score_boards(
                model, boards, board_batch=args.board_batch, amp=args.amp
            )
            positive_tv = tile_mean_tv_score(correct.unsqueeze(0))
            tv = torch.cat((positive_tv, negative_tv)).detach()
            loss, combined = residual_ranking_loss(
                neural,
                tv,
                tv_scale=args.tv_scale,
                residual_bound=args.residual_bound,
                margin=args.margin,
                residual_penalty=args.residual_penalty,
            )
            losses.append(loss)
            train_tv_margins.append(float((tv[0] - tv[1:]).mean()))
            train_hybrid_margins.append(float((combined[0] - combined[1:]).mean().detach()))

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 20 == 0:
            print(
                f"step {step}/{args.steps} loss={float(total_loss.detach()):.4f} "
                f"tv_margin={np.mean(train_tv_margins):+.6f} "
                f"hybrid_margin={np.mean(train_hybrid_margins):+.3f} "
                f"grad={float(gradient_norm):.3f} lr={scheduler.get_last_lr()[0]:.3e} "
                f"{(time.time() - started) / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                eval_names,
                device=device,
                seed=args.seed + 19001,
                negatives_per_family=args.eval_negatives,
                board_batch=args.board_batch,
                tv_scale=args.tv_scale,
                residual_bound=args.residual_bound,
                amp=args.amp,
            )
            overall = metrics["overall"]
            print(
                f"[held-out] step={step} TV={overall['tv_accuracy']:.4f} "
                f"hybrid={overall['hybrid_accuracy']:.4f} lift={overall['lift']:+.4f} "
                f"correct={overall['corrected']}/{overall['tv_failures']} "
                f"break={overall['broken']} ({overall['break_rate']:.3f})",
                flush=True,
            )
            _save(last_checkpoint, model, step=step, metrics=metrics, args=args)
            if overall["lift"] > best_lift:
                best_lift = float(overall["lift"])
                best_metrics = metrics
                _save(checkpoint, model, step=step, metrics=metrics, args=args)
                print(f"saved best lift={best_lift:+.4f} -> {checkpoint}", flush=True)

    if best_metrics is None:
        raise RuntimeError("training completed without evaluation")
    gate = gate_result(best_metrics)
    report = {
        "experiment": "bounded_tv_residual_hard_negative_gate",
        "best_metrics": best_metrics,
        "gate": gate,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== TV residual gate {'PASSED' if gate['pass'] else 'FAILED'} ===", flush=True)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
    print(f"report saved to {args.report}", flush=True)


if __name__ == "__main__":
    main()
