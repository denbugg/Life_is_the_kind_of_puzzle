"""Train and gate the all-pairs directional Siamese puzzle scorer."""
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from candidate_rank import neighbor_targets
from canvas_data import CanvasDataset
from config import CKPT_DIR, NFRAG, SEED, WORK_ROOT
from imgio import train_val_split
from placement_metrics import neighbour_accuracy, placement_accuracy
from siamese_directional import (
    DOWN, LEFT, RIGHT, UP,
    DirectionalSiamese,
    count_parameters,
    load_paired_backbone,
)
from solve_buddies import solve_buddies_from_scores


def _amp(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def siamese_loss(scores: Tensor, permutation: Tensor) -> Tensor:
    targets, exists = neighbor_targets(permutation)
    # scores are B,D,anchor,candidate; targets/existence are B,anchor,D.
    logits = scores.permute(0, 2, 1, 3)
    return F.cross_entropy(logits[exists], targets[exists])


def _probability_rd(scores: Tensor) -> tuple[np.ndarray, np.ndarray]:
    probability = scores.float().softmax(dim=-1)[0].cpu().numpy()
    right = 0.5 * (probability[RIGHT] + probability[LEFT].T)
    down = 0.5 * (probability[DOWN] + probability[UP].T)
    np.fill_diagonal(right, 0.0)
    np.fill_diagonal(down, 0.0)
    return right.astype(np.float32), down.astype(np.float32)


@torch.inference_mode()
def evaluate(
    model: DirectionalSiamese,
    loader: DataLoader,
    device: torch.device,
    maximum_images: int,
    budgets: tuple[int, ...],
) -> dict[str, float]:
    model.eval()
    rank_rows = []
    solver_rows = {budget: [] for budget in budgets}
    seen = 0
    for batch in loader:
        tiles = batch["tiles"].to(device, non_blocking=True)
        permutation = batch["perm"].to(device, non_blocking=True).long()
        with _amp(device):
            scores = model(tiles)
        targets, exists = neighbor_targets(permutation)
        logits = scores.permute(0, 2, 1, 3)
        selected = logits[exists]
        truth = targets[exists]
        true_score = selected.gather(1, truth[:, None])
        ranks = 1 + selected.gt(true_score).sum(dim=1)
        rank_rows.append(
            (
                float(ranks.le(1).float().mean()),
                float(ranks.le(5).float().mean()),
                float(ranks.le(20).float().mean()),
                float(ranks.float().median()),
            )
        )
        for item in range(tiles.shape[0]):
            right, down = _probability_rd(scores[item : item + 1])
            truth_place = np.argsort(permutation[item].cpu().numpy())
            for budget in budgets:
                placement, _ = solve_buddies_from_scores(
                    right, down, max_edges=budget, repair_passes=0
                )
                solver_rows[budget].append(
                    (
                        placement_accuracy(placement, truth_place)[0],
                        neighbour_accuracy(placement, truth_place)[0],
                    )
                )
            seen += 1
            if seen >= maximum_images:
                break
        if seen >= maximum_images:
            break
    model.train()
    ranks = np.asarray(rank_rows)
    result = {
        "r1": float(ranks[:, 0].mean()),
        "r5": float(ranks[:, 1].mean()),
        "r20": float(ranks[:, 2].mean()),
        "median_rank": float(ranks[:, 3].mean()),
    }
    for budget, values in solver_rows.items():
        array = np.asarray(values)
        result[f"b{budget}_placement"] = float(array[:, 0].mean())
        result[f"b{budget}_neighbour"] = float(array[:, 1].mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-images", type=int, default=6)
    parser.add_argument("--budgets", default="128,256,384")
    parser.add_argument(
        "--paired-checkpoint", type=Path,
        default=Path(CKPT_DIR) / "paired_alignment_best.pt",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(WORK_ROOT) / "siamese_directional",
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path(WORK_ROOT) / "gates" / "siamese_directional_gate.json",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.steps < 1 or args.bs < 1 or args.eval_images < 1:
        parser.error("--steps, --bs and --eval-images must be positive")
    if args.channels != 128:
        parser.error("--channels must be 128 when initializing the paired backbone")
    budgets = tuple(int(value) for value in args.budgets.split(","))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = DirectionalSiamese(args.channels, args.embed_dim).to(device)
    paired = torch.load(args.paired_checkpoint, map_location=device, weights_only=False)
    copied = load_paired_backbone(model, paired)
    train_names, val_names = train_val_split()
    train_loader = DataLoader(
        CanvasDataset(train_names, real_prob=0.0, seed=args.seed),
        batch_size=args.bs, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        CanvasDataset(
            val_names[: args.eval_images], real_prob=0.0,
            seed=args.seed + 700_000,
        ),
        batch_size=1, shuffle=False, num_workers=min(args.workers, 1),
        pin_memory=device.type == "cuda",
    )
    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(value) for value in backbone_parameters}
    head_parameters = [
        value for value in model.parameters() if id(value) not in backbone_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.lr * args.backbone_lr_scale},
            {"params": head_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * args.backbone_lr_scale, args.lr],
        total_steps=args.steps,
        pct_start=0.08,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(train_loader)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = -float("inf")
    best_metrics: dict[str, float] = {}
    started = time.time()
    print(
        f"device={device} params={count_parameters(model):,} copied={copied} "
        f"train={len(train_names)} val={args.eval_images}",
        flush=True,
    )
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        tiles = batch["tiles"].to(device, non_blocking=True)
        permutation = batch["perm"].to(device, non_blocking=True).long()
        with _amp(device):
            scores = model(tiles)
            loss = siamese_loss(scores, permutation)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step == 1 or step % 25 == 0:
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"scale={float(model.logit_scale.exp().detach()):.2f} "
                f"lr={scheduler.get_last_lr()[-1]:.2e} "
                f"{(time.time() - started) / step:.2f}s/it",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model, val_loader, device, args.eval_images, budgets
            )
            print(
                "[VAL] " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()),
                flush=True,
            )
            score = metrics["r1"] + max(
                metrics[f"b{budget}_neighbour"] for budget in budgets
            )
            payload = {
                "model": model.state_dict(),
                "model_kwargs": {
                    "channels": args.channels, "embed_dim": args.embed_dim
                },
                "step": step,
                "metrics": metrics,
                "args": vars(args),
            }
            torch.save(payload, args.out_dir / "last.pt")
            if score > best:
                best, best_metrics = score, metrics
                torch.save(payload, args.out_dir / "best.pt")
                print(f"saved best score={best:.4f}", flush=True)

    best_budget = max(
        budgets, key=lambda budget: best_metrics[f"b{budget}_neighbour"]
    )
    report = {
        "experiment": "all_pairs_directional_siamese",
        "checkpoint": str(args.out_dir / "best.pt"),
        "metrics": best_metrics,
        "best_budget": best_budget,
        "thresholds": {"r1": 0.25, "neighbour": 0.18},
        "passed": (
            best_metrics["r1"] >= 0.25
            and best_metrics[f"b{best_budget}_neighbour"] >= 0.18
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
