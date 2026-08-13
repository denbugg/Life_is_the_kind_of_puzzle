"""OH1: cache-free online hard-negative refinement for PairwiseNet.

For each correct-order synthetic/real-recovered board, score only a small random
reservoir per true horizontal/vertical pair, retain the current model's hardest
false candidates, and train a listwise positive-at-zero objective.  No full
576xK candidate cache or board-wide all-pair score is constructed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from config import FS, GRID, NFRAG, SEED
from datasets import CompatDataset
from imgio import train_val_split
from models import PairwiseNet, count_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OH1 online hard-negative PairwiseNet refinement")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--nA", type=int, default=8, help="anchors per direction and board")
    p.add_argument("--pool", type=int, default=32, help="random false reservoir per anchor")
    p.add_argument("--M", type=int, default=16, help="true plus mined false list size")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--real-prob", type=float, default=0.6)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--init", type=Path, default=Path(r"E:\pazzle_work\ckpt\pair0_best.pt"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=240815)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pair_images(frags: Tensor, anchors: Tensor, candidates: Tensor, transpose: bool) -> Tensor:
    value = frags.transpose(-1, -2) if transpose else frags
    n_anchor, n_cand = candidates.shape
    left = value[anchors][:, None].expand(n_anchor, n_cand, 3, FS, FS)
    right = value[candidates]
    return torch.cat((left, right), dim=-1).reshape(n_anchor * n_cand, 3, FS, 2 * FS)


def mined_candidates(
    model: PairwiseNet,
    frags: Tensor,
    anchors: Tensor,
    offset: int,
    transpose: bool,
    pool: int,
    M: int,
    generator: torch.Generator,
) -> Tensor:
    """Return [true, current-model-hard false...], completely within one board."""
    if M < 2 or M > pool + 1:
        raise ValueError("require 2 <= M <= pool+1")
    true = anchors + offset
    random = torch.randint(NFRAG, (anchors.numel(), pool), device=frags.device, generator=generator)
    invalid = random.eq(true[:, None]) | random.eq(anchors[:, None])
    # Deterministic non-true replacements preserve a bounded reservoir without rejection loops.
    random = torch.where(invalid, (random + 1 + (random.eq(anchors[:, None])).long()) % NFRAG, random)
    invalid = random.eq(true[:, None]) | random.eq(anchors[:, None])
    random = torch.where(invalid, (random + 2) % NFRAG, random)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=frags.device.type == "cuda"):
        score = model(pair_images(frags, anchors, random, transpose)).float().reshape(anchors.numel(), pool)
    hardest = random.gather(1, score.topk(M - 1, dim=1).indices)
    return torch.cat((true[:, None], hardest), dim=1)


def one_board_loss(
    model: PairwiseNet,
    frags: Tensor,
    right_anchors: Tensor,
    down_anchors: Tensor,
    nA: int,
    pool: int,
    M: int,
    generator: torch.Generator,
) -> tuple[Tensor, float]:
    chunks: list[Tensor] = []
    for anchors_all, offset, transpose in ((right_anchors, 1, False), (down_anchors, GRID, True)):
        picks = anchors_all[torch.randint(anchors_all.numel(), (nA,), generator=generator, device=frags.device)]
        candidates = mined_candidates(model, frags, picks, offset, transpose, pool, M, generator)
        chunks.append(pair_images(frags, picks, candidates, transpose))
    groups = len(chunks)
    with torch.autocast("cuda", dtype=torch.float16, enabled=frags.device.type == "cuda"):
        logits = model(torch.cat(chunks, dim=0)).reshape(groups, nA, M)
        loss = F.cross_entropy(logits.reshape(groups * nA, M), torch.zeros(groups * nA, dtype=torch.long, device=frags.device))
    accuracy = float(logits.argmax(dim=-1).eq(0).float().mean())
    return loss, accuracy


@torch.inference_mode()
def evaluate(
    model: PairwiseNet,
    loader: DataLoader,
    right_anchors: Tensor,
    down_anchors: Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> float:
    model.eval()
    vals: list[float] = []
    for batch_index, frags in enumerate(loader):
        if batch_index >= args.eval_batches:
            break
        for board in frags.to(args.device, non_blocking=True):
            _, value = one_board_loss(model, board, right_anchors, down_anchors, args.nA, args.pool, args.M, generator)
            vals.append(value)
    model.train()
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.bs < 1 or args.nA < 1 or args.pool < args.M - 1:
        raise ValueError("invalid positive sampling arguments")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    payload = torch.load(args.init, map_location=device, weights_only=False)
    model = PairwiseNet().to(device)
    model.load_state_dict(payload["model"], strict=True)
    print(f"device={device} params={count_params(model):,} init={args.init} init_step={payload.get('step')}", flush=True)
    names, val_names = train_val_split()
    train_loader = DataLoader(CompatDataset(names, real_prob=args.real_prob), batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(CompatDataset(val_names, real_prob=1.0), batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    index = torch.arange(NFRAG, device=device)
    right = index[index.remainder(GRID) != GRID - 1]
    down = index[index.div(GRID, rounding_mode="floor") != GRID - 1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps)
    generator = torch.Generator(device=device).manual_seed(args.seed + 99)
    args.out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history: list[dict[str, float]] = []
    iterator = iter(train_loader)
    started = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        boards = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        accs: list[float] = []
        for board in boards:
            loss, acc = one_board_loss(model, board, right, down, args.nA, args.pool, args.M, generator)
            losses.append(loss)
            accs.append(acc)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0:
            print(f"step {step}/{args.steps} loss={float(loss):.4f} online_hard_acc={np.mean(accs):.4f} lr={scheduler.get_last_lr()[0]:.2e} {(time.time()-started)/step:.2f}s/it", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            value = evaluate(model, val_loader, right, down, args, generator)
            record = {"step": float(step), "train_loss": float(loss), "val_online_hard_acc": value}
            history.append(record)
            print(f"[VAL] step={step} online_hard_acc={value:.4f}", flush=True)
            payload_out = {"model": model.state_dict(), "step": step, "val": value, "args": vars(args)}
            torch.save(payload_out, args.out / "last.pt")
            if value > best:
                best = value
                torch.save(payload_out, args.out / "best.pt")
                print(f"saved best={best:.4f}", flush=True)
    report = {"checkpoint": str(args.out / "best.pt"), "best_online_hard_acc": best, "history": history}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
