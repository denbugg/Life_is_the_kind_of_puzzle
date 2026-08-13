"""OH3: online hard-negative PairwiseNet training on U1 candidate rows.

The U1 graph is built label-blind per streaming board from frozen MacroAffinity
and R2L directional proposal models.  Only sampled rows are scored by PairwiseNet;
there is no saved full-board pairwise hard-negative cache.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from config import FS, GRID, NFRAG
from datasets import CompatDataset
from eval_r2l_affinity_union import DEFAULT_AFFINITY_A, DEFAULT_AFFINITY_B, DEFAULT_R2L, _load_r2, _union_candidates
from imgio import train_val_split
from models import PairwiseNet, count_params
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OH3 online U1-candidate hard PairwiseNet refinement")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--nA", type=int, default=8)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--candidate-k", type=int, default=64)
    p.add_argument("--r2-topk", type=int, default=8)
    p.add_argument("--affinity-ckpt", default=DEFAULT_AFFINITY_A)
    p.add_argument("--affinity-ckpt2", default=DEFAULT_AFFINITY_B)
    p.add_argument("--r2-ckpt", default=DEFAULT_R2L)
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


def autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def pair_images(frags: Tensor, anchors: Tensor, candidates: Tensor, transpose: bool) -> Tensor:
    value = frags.transpose(-1, -2) if transpose else frags
    n_anchor, n_cand = candidates.shape
    left = value[anchors][:, None].expand(n_anchor, n_cand, 3, FS, FS)
    right = value[candidates]
    return torch.cat((left, right), dim=-1).reshape(n_anchor * n_cand, 3, FS, 2 * FS)


@torch.inference_mode()
def build_u1_graph(
    tiles: Tensor,
    affinity: torch.nn.Module,
    affinity2: torch.nn.Module,
    r2: torch.nn.Module,
    candidate_k: int,
    r2_topk: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    base, base_valid = mine_affinity_candidates(
        affinity, tiles.unsqueeze(0), candidate_k=candidate_k, device=device, affinity_secondary=affinity2
    )
    with autocast(device):
        r2_scores = r2(tiles.unsqueeze(0))[0].float()
    return _union_candidates(base[0], base_valid[0], r2_scores, r2_topk)


def sample_u1_rows(
    model: PairwiseNet,
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    anchors_all: Tensor,
    offset: int,
    transpose: bool,
    nA: int,
    M: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    true = anchors_all + offset
    row_true = candidates[anchors_all].eq(true[:, None]) & valid[anchors_all]
    eligible = anchors_all[row_true.any(dim=1)]
    if eligible.numel() < 1:
        raise RuntimeError("U1 graph has no candidate-covered true anchors")
    picks = eligible[torch.randint(eligible.numel(), (nA,), device=tiles.device, generator=generator)]
    choices: list[Tensor] = []
    for anchor in picks:
        target = anchor + offset
        row = candidates[anchor, valid[anchor]]
        false = row[(row != target) & (row != anchor)]
        if false.numel() < M - 1:
            raise RuntimeError("U1 candidate row too short for requested M")
        with torch.no_grad(), autocast(tiles.device):
            score = model(pair_images(tiles, anchor.view(1), false.view(1, -1), transpose)).float().flatten()
        hard = false[score.topk(M - 1).indices]
        choices.append(torch.cat((target.view(1), hard)))
    return picks, torch.stack(choices)


def one_board_loss(
    model: PairwiseNet,
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    right: Tensor,
    down: Tensor,
    nA: int,
    M: int,
    generator: torch.Generator,
) -> tuple[Tensor, float, float]:
    chunks: list[Tensor] = []
    covered: list[float] = []
    for anchors_all, offset, transpose in ((right, 1, False), (down, GRID, True)):
        anchors, chosen = sample_u1_rows(model, tiles, candidates, valid, anchors_all, offset, transpose, nA, M, generator)
        chunks.append(pair_images(tiles, anchors, chosen, transpose))
        truth = anchors_all + offset
        covered.append(float((candidates[anchors_all].eq(truth[:, None]) & valid[anchors_all]).any(dim=1).float().mean()))
    with autocast(tiles.device):
        logits = model(torch.cat(chunks, dim=0)).reshape(2, nA, M)
        loss = F.cross_entropy(logits.reshape(2 * nA, M), torch.zeros(2 * nA, dtype=torch.long, device=tiles.device))
    acc = float(logits.argmax(dim=-1).eq(0).float().mean())
    return loss, acc, float(np.mean(covered))


@torch.inference_mode()
def evaluate(
    model: PairwiseNet,
    loader: DataLoader,
    affinity: torch.nn.Module,
    affinity2: torch.nn.Module,
    r2: torch.nn.Module,
    right: Tensor,
    down: Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    accs: list[float] = []
    covers: list[float] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.eval_batches:
            break
        for tiles in batch.to(device, non_blocking=True):
            candidates, valid = build_u1_graph(tiles, affinity, affinity2, r2, args.candidate_k, args.r2_topk, device)
            _, acc, cover = one_board_loss(model, tiles, candidates, valid, right, down, args.nA, args.M, generator)
            accs.append(acc)
            covers.append(cover)
    model.train()
    return (float(np.mean(accs)) if accs else 0.0, float(np.mean(covers)) if covers else 0.0)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.bs < 1 or args.nA < 1 or args.M < 2:
        raise ValueError("steps, bs, nA must be positive and M must be >=2")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    affinity, _, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity2, _, _ = load_frozen_affinity(args.affinity_ckpt2, device)
    r2 = _load_r2(args.r2_ckpt, device)
    init = torch.load(args.init, map_location=device, weights_only=False)
    model = PairwiseNet().to(device)
    model.load_state_dict(init["model"], strict=True)
    print(f"device={device} pair_params={count_params(model):,} init_step={init.get('step')} U1=affinity{args.candidate_k}+r2x{args.r2_topk}", flush=True)
    train_names, val_names = train_val_split()
    train_loader = DataLoader(CompatDataset(train_names, real_prob=args.real_prob), batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(CompatDataset(val_names, real_prob=1.0), batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    index = torch.arange(NFRAG, device=device)
    right = index[index.remainder(GRID) != GRID - 1]
    down = index[index.div(GRID, rounding_mode="floor") != GRID - 1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps)
    generator = torch.Generator(device=device).manual_seed(args.seed + 123)
    args.out.mkdir(parents=True, exist_ok=True)
    iterator = iter(train_loader)
    best = -1.0
    history: list[dict[str, float]] = []
    started = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        accs: list[float] = []
        covers: list[float] = []
        for tiles in batch.to(device, non_blocking=True):
            candidates, valid = build_u1_graph(tiles, affinity, affinity2, r2, args.candidate_k, args.r2_topk, device)
            loss, acc, cover = one_board_loss(model, tiles, candidates, valid, right, down, args.nA, args.M, generator)
            losses.append(loss)
            accs.append(acc)
            covers.append(cover)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0:
            print(f"step {step}/{args.steps} loss={loss.detach().item():.4f} u1_hard_acc={np.mean(accs):.4f} u1_coverage={np.mean(covers):.4f} lr={scheduler.get_last_lr()[0]:.2e} {(time.time()-started)/step:.2f}s/it", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val_acc, val_cover = evaluate(model, val_loader, affinity, affinity2, r2, right, down, args, generator, device)
            record = {"step": float(step), "train_loss": loss.detach().item(), "val_u1_hard_acc": val_acc, "val_u1_coverage": val_cover}
            history.append(record)
            print(f"[VAL] step={step} u1_hard_acc={val_acc:.4f} u1_coverage={val_cover:.4f}", flush=True)
            saved = {"model": model.state_dict(), "step": step, "val": val_acc, "args": vars(args)}
            torch.save(saved, args.out / "last.pt")
            if val_acc > best:
                best = val_acc
                torch.save(saved, args.out / "best.pt")
                print(f"saved best={best:.4f}", flush=True)
    report = {"checkpoint": str(args.out / "best.pt"), "best_u1_hard_acc": best, "history": history}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
