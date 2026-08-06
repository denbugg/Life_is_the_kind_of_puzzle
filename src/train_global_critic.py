"""Train and gate the whole-board structural critic.

The critic never predicts an absolute position and never receives a tile id.
For every training example the positive and all negatives contain *exactly the
same independently degraded tile bag*; only the proposed 24x24 arrangement
changes.  This prevents image/content shortcuts and directly trains the score
that a later discrete repair/search procedure is allowed to query.

Examples
--------

    python src/train_global_critic.py --smoke
    python src/train_global_critic.py --steps 1200 --device cuda
    python src/train_global_critic.py --eval-only --eval-images 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import CKPT_DIR, NFRAG, SEED, TRAIN_TGT
from distort import distort_frags
from global_critic import (
    EVAL_NEGATIVE_FAMILIES,
    HARD_NEAR_FAMILIES,
    TRAIN_NEGATIVE_FAMILIES,
    GlobalStructuralCritic,
    apply_orders,
    count_params,
    sample_negative_orders,
    smoke as critic_smoke,
    tile_mean_tv_score,
)
from imgio import load, to_frags, train_val_split


NEAR_THRESHOLDS: dict[str, float] = {
    "adjacent_swap": 0.65,
    "nearby_swap": 0.65,
    "patch_shuffle_3": 0.80,
    "block_swap_2": 0.80,
}
MACRO_THRESHOLDS: dict[str, float] = {
    "macro_swap_4": 0.90,
    "macro_swap_6": 0.90,
    "random_permutation": 0.99,
}


def _autocast(device: torch.device, enabled: bool):
    return (
        torch.autocast("cuda", dtype=torch.float16)
        if enabled and device.type == "cuda"
        else nullcontext()
    )


def _to_tensor(tiles: np.ndarray, device: torch.device) -> Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(tiles))
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
        .to(device)
    )


def _synthetic_correct_board(
    name: str,
    rng: np.random.Generator,
    device: torch.device,
) -> Tensor:
    clean = load(os.path.join(TRAIN_TGT, name))
    dirty = distort_frags(to_frags(clean), rng)
    return _to_tensor(dirty, device)


def _sample_training_families(
    rng: np.random.Generator,
    count: int,
) -> tuple[str, ...]:
    pool = np.asarray(TRAIN_NEGATIVE_FAMILIES, dtype=object)
    picked = rng.choice(pool, size=count, replace=count > len(pool))
    return tuple(str(value) for value in picked)


def _negative_boards(
    correct: Tensor,
    families: Sequence[str],
    rng: np.random.Generator,
) -> Tensor:
    orders_np = sample_negative_orders(1, families, rng)
    orders = torch.from_numpy(orders_np).to(correct.device)
    return apply_orders(correct.unsqueeze(0), orders).squeeze(0)


def ranking_loss(scores: Tensor, *, margin: float, temperature: float) -> Tensor:
    """Pairwise soft-margin plus listwise positive-board classification."""
    if scores.ndim != 1 or len(scores) < 2:
        raise ValueError("scores must contain one positive and at least one negative")
    pairwise = F.softplus(margin + scores[1:] - scores[0]).mean()
    target = torch.zeros(1, dtype=torch.long, device=scores.device)
    listwise = F.cross_entropy(scores.unsqueeze(0) / temperature, target)
    return pairwise + 0.5 * listwise


def _score_boards(
    model: GlobalStructuralCritic,
    boards: Tensor,
    *,
    board_batch: int,
    amp: bool,
) -> Tensor:
    if boards.ndim != 5:
        raise ValueError(f"boards must be (B,{NFRAG},3,20,20), got {tuple(boards.shape)}")
    rows: list[Tensor] = []
    for start in range(0, len(boards), board_batch):
        with _autocast(boards.device, amp):
            rows.append(model(boards[start : start + board_batch]).float())
    return torch.cat(rows)


def _family_metrics(
    positive: float,
    negatives: np.ndarray,
    tv_positive: float,
    tv_negatives: np.ndarray,
) -> dict[str, float]:
    learned_margin = positive - negatives
    tv_margin = tv_positive - tv_negatives
    return {
        "accuracy": float(np.mean(learned_margin > 0.0)),
        "mean_margin": float(learned_margin.mean()),
        "median_margin": float(np.median(learned_margin)),
        "strict_min_margin": float(learned_margin.min()),
        "tv_accuracy": float(np.mean(tv_margin > 0.0)),
        "tv_mean_margin": float(tv_margin.mean()),
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
    amp: bool,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    aggregate: dict[str, list[dict[str, float]]] = {
        family: [] for family in EVAL_NEGATIVE_FAMILIES
    }
    per_image: list[dict[str, Any]] = []

    for image_index, name in enumerate(names):
        rng = np.random.default_rng(seed + image_index * 104729)
        correct = _synthetic_correct_board(name, rng, device)
        with _autocast(device, amp):
            positive_score = float(model(correct.unsqueeze(0)).float().item())
            tv_positive = float(tile_mean_tv_score(correct.unsqueeze(0)).item())
        image_row: dict[str, Any] = {
            "image": name,
            "positive_score": positive_score,
            "tv_positive_score": tv_positive,
            "families": {},
        }

        for family_index, family in enumerate(EVAL_NEGATIVE_FAMILIES):
            family_rng = np.random.default_rng(
                seed + image_index * 1_000_003 + family_index * 8191
            )
            families = (family,) * negatives_per_family
            negatives = _negative_boards(correct, families, family_rng)
            scores = (
                _score_boards(
                    model,
                    negatives,
                    board_batch=board_batch,
                    amp=amp,
                )
                .cpu()
                .numpy()
            )
            tv_scores = tile_mean_tv_score(negatives).cpu().numpy()
            metrics = _family_metrics(
                positive_score,
                scores,
                tv_positive,
                tv_scores,
            )
            image_row["families"][family] = metrics
            aggregate[family].append(metrics)
        per_image.append(image_row)
        near_text = " ".join(
            f"{family}={image_row['families'][family]['accuracy']:.2f}"
            for family in HARD_NEAR_FAMILIES
        )
        print(f"  {name}: {near_text}", flush=True)

    if was_training:
        model.train()

    families_result: dict[str, dict[str, float]] = {}
    for family, rows in aggregate.items():
        families_result[family] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
    near_accuracy = float(
        np.mean([families_result[family]["accuracy"] for family in HARD_NEAR_FAMILIES])
    )
    macro_accuracy = float(
        np.mean(
            [
                families_result[family]["accuracy"]
                for family in ("macro_swap_4", "macro_swap_6", "random_permutation")
            ]
        )
    )
    overall_accuracy = float(
        np.mean([row["accuracy"] for row in families_result.values()])
    )
    tv_overall_accuracy = float(
        np.mean([row["tv_accuracy"] for row in families_result.values()])
    )
    return {
        "families": families_result,
        "near_accuracy": near_accuracy,
        "macro_accuracy": macro_accuracy,
        "overall_accuracy": overall_accuracy,
        "tv_overall_accuracy": tv_overall_accuracy,
        "learned_lift_over_tv": overall_accuracy - tv_overall_accuracy,
        "images": len(names),
        "negatives_per_family": negatives_per_family,
        "per_image": per_image,
    }


def gate_result(metrics: dict[str, Any]) -> dict[str, Any]:
    thresholds = {**NEAR_THRESHOLDS, **MACRO_THRESHOLDS}
    family_pass = {
        family: metrics["families"][family]["accuracy"] >= threshold
        for family, threshold in thresholds.items()
    }
    lift_pass = metrics["learned_lift_over_tv"] >= 0.10
    return {
        "thresholds": thresholds,
        "family_pass": family_pass,
        "learned_lift_over_tv_threshold": 0.10,
        "learned_lift_over_tv_pass": bool(lift_pass),
        "pass": bool(all(family_pass.values()) and lift_pass),
    }


def _checkpoint(
    path: str,
    model: GlobalStructuralCritic,
    *,
    step: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_kwargs": model.model_kwargs,
            "step": int(step),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[GlobalStructuralCritic, dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = GlobalStructuralCritic(**payload["model_kwargs"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload


def _parse_dilations(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("dilations must be comma-separated positive integers")
    return result


def _jsonable(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: (
            str(value)
            if isinstance(value, Path)
            else list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value in vars(args).items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--negatives-per-step", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-images", type=int, default=4)
    parser.add_argument("--eval-negatives", type=int, default=8)
    parser.add_argument("--board-batch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--tile-width", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--edge-width", type=int, default=12)
    parser.add_argument("--edge-dim", type=int, default=24)
    parser.add_argument("--grid-width", type=int, default=64)
    parser.add_argument("--stats-dim", type=int, default=12)
    parser.add_argument("--edge-band", type=int, default=3)
    parser.add_argument("--dilations", type=_parse_dilations, default=(1, 2, 4, 8, 12, 4))
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="global_critic")
    parser.add_argument("--ckpt-dir", default=CKPT_DIR)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("E:/pazzle_work/gates/global_critic_gate.json"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        return args
    positive = (
        "steps",
        "negatives_per_step",
        "eval_every",
        "eval_images",
        "eval_negatives",
        "board_batch",
    )
    if any(getattr(args, key) < 1 for key in positive):
        parser.error(f"{', '.join(positive)} must all be positive")
    if args.margin < 0.0 or args.score_temperature <= 0.0:
        parser.error("--margin must be nonnegative and --score-temperature positive")
    return args


def _smoke(device: torch.device) -> dict[str, Any]:
    critic_smoke(device)
    scores = torch.tensor([1.0, 0.5, -0.2], device=device, requires_grad=True)
    loss = ranking_loss(scores, margin=0.1, temperature=1.0)
    loss.backward()
    if not torch.isfinite(loss) or scores.grad is None or not torch.isfinite(scores.grad).all():
        raise AssertionError("ranking loss smoke failed")
    rng = np.random.default_rng(17)
    orders = sample_negative_orders(1, EVAL_NEGATIVE_FAMILIES, rng)
    identity = np.arange(NFRAG)
    changed = [int(np.sum(order != identity)) for order in orders[0]]
    # A random permutation may retain a few fixed points by chance.
    if min(changed) < 2 or max(changed) < int(0.90 * NFRAG):
        raise AssertionError(f"negative family smoke produced unexpected changes: {changed}")
    return {"ranking_loss": float(loss.detach()), "changed_cells": changed}


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.smoke:
        print(f"[global-critic smoke] {_smoke(device)}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_names, val_names = train_val_split()
    if len(val_names) < args.eval_images:
        raise ValueError(f"--eval-images exceeds held-out pool ({len(val_names)})")
    eval_names = val_names[: args.eval_images]
    checkpoint = args.checkpoint or os.path.join(args.ckpt_dir, f"{args.tag}_best.pt")

    if args.eval_only:
        model, payload = load_checkpoint(checkpoint, device)
        print(
            f"eval-only checkpoint={checkpoint} step={payload.get('step')} device={device}",
            flush=True,
        )
        metrics = evaluate(
            model,
            eval_names,
            device=device,
            seed=args.seed + 9973,
            negatives_per_family=args.eval_negatives,
            board_batch=args.board_batch,
            amp=args.amp,
        )
        result = {"metrics": metrics, "gate": gate_result(metrics)}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    model = GlobalStructuralCritic(
        tile_width=args.tile_width,
        embedding_dim=args.embedding_dim,
        edge_width=args.edge_width,
        edge_dim=args.edge_dim,
        grid_width=args.grid_width,
        stats_dim=args.stats_dim,
        edge_band=args.edge_band,
        dilations=args.dilations,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    config = _jsonable(args)
    print(
        f"device={device} params={count_params(model):,} amp={args.amp} "
        f"negative_boards/step={args.negatives_per_step} board_batch={args.board_batch} "
        f"dilations={args.dilations}",
        flush=True,
    )
    print(
        "contract: positive and negatives contain exactly the same dirty tile bag; "
        "only board order changes",
        flush=True,
    )

    rng = np.random.default_rng(args.seed + 73)
    best_selection = -math.inf
    best_metrics: dict[str, Any] | None = None
    started = time.time()

    for step in range(1, args.steps + 1):
        name = train_names[int(rng.integers(0, len(train_names)))]
        sample_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        correct = _synthetic_correct_board(name, sample_rng, device)
        families = _sample_training_families(sample_rng, args.negatives_per_step)
        negatives = _negative_boards(correct, families, sample_rng)
        boards = torch.cat((correct.unsqueeze(0), negatives), dim=0)

        optimizer.zero_grad(set_to_none=True)
        scores = _score_boards(
            model,
            boards,
            board_batch=args.board_batch,
            amp=args.amp,
        )
        loss = ranking_loss(
            scores,
            margin=args.margin,
            temperature=args.score_temperature,
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gradients_finite = bool(torch.isfinite(gradient_norm))
        if gradients_finite:
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            update_applied = scaler.get_scale() >= old_scale
        else:
            # Do not let an FP32 optimizer consume non-finite gradients.  For
            # AMP, lowering the scale without an optimizer step is the intended
            # recovery path.
            if scaler.is_enabled():
                scaler.update(new_scale=max(1.0, scaler.get_scale() * 0.5))
            update_applied = False
        if update_applied:
            scheduler.step()
        else:
            optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 20 == 0:
            elapsed = time.time() - started
            deltas = (scores[0] - scores[1:]).detach().cpu().numpy()
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"margin={float(deltas.mean()):+.3f}/{float(deltas.min()):+.3f} "
                f"grad={float(gradient_norm):.3f} update={int(update_applied)} "
                f"lr={scheduler.get_last_lr()[0]:.3e} families={','.join(families)} "
                f"{elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            print(f"[held-out global critic] step={step}", flush=True)
            metrics = evaluate(
                model,
                eval_names,
                device=device,
                seed=args.seed + 9973,
                negatives_per_family=args.eval_negatives,
                board_batch=args.board_batch,
                amp=args.amp,
            )
            selection = metrics["near_accuracy"]
            print(
                f"[mean] near={metrics['near_accuracy']:.4f} "
                f"macro={metrics['macro_accuracy']:.4f} "
                f"overall={metrics['overall_accuracy']:.4f} "
                f"tv={metrics['tv_overall_accuracy']:.4f} "
                f"lift={metrics['learned_lift_over_tv']:+.4f}",
                flush=True,
            )
            _checkpoint(
                os.path.join(args.ckpt_dir, f"{args.tag}_last.pt"),
                model,
                step=step,
                metrics=metrics,
                config=config,
            )
            if selection > best_selection:
                best_selection = selection
                best_metrics = metrics
                _checkpoint(
                    checkpoint,
                    model,
                    step=step,
                    metrics=metrics,
                    config=config,
                )
                print(f"saved best near_accuracy={selection:.4f} -> {checkpoint}", flush=True)

    if best_metrics is None:
        raise RuntimeError("training finished without evaluation")
    gate = gate_result(best_metrics)
    report = {
        "experiment": "global_structural_critic_hard_negative_gate",
        "config": config,
        "best_metrics": best_metrics,
        "gate": gate,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict = "PASSED -> run critic-guided repair gate" if gate["pass"] else (
        "FAILED -> critic score is not yet reliable enough for discrete repair"
    )
    print(f"\n=== global critic gate {verdict} ===", flush=True)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
    print(f"report saved to {args.report}", flush=True)


if __name__ == "__main__":
    main()
