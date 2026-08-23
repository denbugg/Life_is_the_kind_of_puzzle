"""Ceiling probe: how much does neighbour context buy the restorer?

The context-free restorer is heading for ridge bb_prec ~0.40, which maps to
about 0.30 SSIM -- past the current submission but short of the 0.41 that a
0.64 placement would give.  The proposed way out is an EM loop: solve, then
re-restore each tile using its placed neighbours, then re-solve.

That loop is only worth building if context actually helps.  This trains the
ContextRestorer with ORACLE neighbours (the true layout) and reports the same
held-out seam metrics as the context-free model, so the two are directly
comparable.  Oracle context is an upper bound, not a deployable arm: in the
real loop neighbours come from an imperfect layout and some are wrong.

A neighbour-dropout schedule is applied anyway, so the model never assumes a
full 3x3 and stays usable when the layout is sparse.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import CACHE_DIR, CKPT_DIR, FS, GRID as G, NFRAG as N, TRAIN_INP, TRAIN_TGT, VAL_COUNT
from distort import distort_frags
from restore_context import ContextRestorer, build_blocks
from restore_tile import seam_metrics, to_frags


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


class Boards(Dataset):
    def __init__(self, names, inv, margin, thr, synth_prob, drop, acc_low=0.15):
        self.names, self.inv, self.margin, self.thr = names, inv, margin, thr
        self.synth_prob, self.drop, self.acc_low = synth_prob, drop, acc_low

    def __len__(self):
        return len(self.names)

    def __getitem__(self, k):
        nm = self.names[k]
        clean = to_frags(load_rgb(Path(TRAIN_TGT) / nm)).astype(np.float32)
        if self.synth_prob and np.random.rand() < self.synth_prob:
            tiles = distort_frags(clean.astype(np.uint8), np.random.default_rng()).astype(np.float32)
            good = np.ones(N, bool)
        else:
            dirty = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)
            tiles = dirty[self.inv[k].astype(np.int64)]
            good = self.margin[k] >= self.thr
        # Simulate the layout the loop will actually hand us.  Training only on
        # true neighbours teaches a model that never sees a WRONG one, but the
        # first solve lands around place_acc 0.2, so most neighbours are strangers.
        # Sampling the accuracy per board forces the net to check its context
        # rather than trust it.
        board = np.arange(N)
        acc = np.random.uniform(self.acc_low, 1.0)
        n_wrong = int(N * (1.0 - acc))
        if n_wrong > 1:
            idx = np.random.choice(N, n_wrong, replace=False)
            board[idx] = board[np.random.permutation(idx)]
        keep = good & (np.random.rand(N) > self.drop)
        block, mask = build_blocks(tiles, board, G, keep)
        return (torch.from_numpy(block).permute(0, 3, 1, 2),
                torch.from_numpy(mask).permute(0, 3, 1, 2),
                torch.from_numpy(clean).permute(0, 3, 1, 2),
                torch.from_numpy(good.astype(np.float32)))


@torch.no_grad()
def evaluate(model, names, inv, device, n_boards, drop, metric, layout_acc=1.0):
    """layout_acc<1 rebuilds the context from a deliberately wrong layout, which
    is the condition the EM loop actually runs in; layout_acc=1 is the oracle
    ceiling and is reported alongside for reference."""
    model.eval()
    rng = np.random.default_rng(0)
    got, base = [], []
    for k in range(n_boards):
        dirty = to_frags(load_rgb(Path(TRAIN_INP) / names[k])).astype(np.float32)
        ordered = dirty[inv[k].astype(np.int64)]
        keep = rng.random(N) > drop
        board = np.arange(N)
        n_wrong = int(N * (1.0 - layout_acc))
        if n_wrong > 1:
            idx = rng.choice(N, n_wrong, replace=False)
            board[idx] = board[rng.permutation(idx)]
        block, mask = build_blocks(ordered, board, G, keep)
        b = torch.from_numpy(block).permute(0, 3, 1, 2).to(device)
        m = torch.from_numpy(mask).permute(0, 3, 1, 2).to(device)
        with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            out = torch.cat([model(b[i:i + 128], m[i:i + 128]) for i in range(0, N, 128)])
        rec = out.float().permute(0, 2, 3, 1).clamp(0, 255).cpu().numpy()
        got.append(seam_metrics(rec, metric=metric))
        base.append(seam_metrics(ordered, metric=metric))
    model.train()
    agg = lambda rows, key: float(np.mean([r[key] for r in rows]))
    return ({k: agg(got, k) for k in got[0]}, {k: agg(base, k) for k in base[0]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path(CACHE_DIR) / "restore_labels.npz")
    ap.add_argument("--ckpt", type=Path, default=Path(CKPT_DIR) / "restore_context.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--tile-batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--synth-prob", type=float, default=0.5)
    ap.add_argument("--layout-acc-low", type=float, default=0.15,
                    help="lowest simulated placement accuracy of the context layout")
    ap.add_argument("--neighbour-drop", type=float, default=0.3,
                    help="probability a neighbour is withheld, so the model tolerates sparse layouts")
    ap.add_argument("--margin-quantile", type=float, default=0.5)
    ap.add_argument("--l1-weight", type=float, default=1.0)
    ap.add_argument("--edge-weight", type=float, default=3.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-boards", type=int, default=6)
    ap.add_argument("--eval-layout-acc", type=float, default=0.25,
                    help="placement accuracy of the context layout used by the gate")
    ap.add_argument("--eval-metric", choices=("ridge", "mgc", "both"), default="both")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    blob = np.load(args.labels, allow_pickle=True)
    names, inv, margin = blob["names"], blob["inv"], blob["margin"]
    thr = float(np.quantile(margin, args.margin_quantile))
    n_val = min(VAL_COUNT, len(names) // 4)
    tr, va = slice(0, len(names) - n_val), slice(len(names) - n_val, len(names))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ContextRestorer(args.ch, args.blocks).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  device={device}", flush=True)

    dl = DataLoader(Boards(names[tr], inv[tr], margin[tr], thr, args.synth_prob,
                           args.neighbour_drop, args.layout_acc_low),
                    batch_size=1, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    wmap = torch.ones(1, 1, FS, FS, device=device)
    wmap[:, :, :2, :] = args.edge_weight; wmap[:, :, -2:, :] = args.edge_weight
    wmap[:, :, :, :2] = args.edge_weight; wmap[:, :, :, -2:] = args.edge_weight

    best, step, started = -1.0, 0, time.perf_counter()
    while step < args.steps:
        for block, mask, clean, good in dl:
            if step >= args.steps:
                break
            block, mask, clean = block[0].to(device), mask[0].to(device), clean[0].to(device)
            sel = torch.nonzero(good[0] > 0, as_tuple=True)[0]
            if len(sel) > args.tile_batch:
                sel = sel[torch.randperm(len(sel))[: args.tile_batch]]
            with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                out = model(block[sel], mask[sel])
                loss = ((out.float() - clean[sel]).abs() * wmap).mean() * args.l1_weight
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                got, base = evaluate(model, names[va], inv[va], device, args.eval_boards,
                                     args.neighbour_drop, args.eval_metric, args.eval_layout_acc)
                orac, _ = evaluate(model, names[va], inv[va], device, args.eval_boards,
                                   args.neighbour_drop, args.eval_metric, 1.0)
                flag = ""
                if got["bb_prec"] > best:
                    best = got["bb_prec"]
                    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "args": vars(args),
                                "step": step, "bb_prec": best}, args.ckpt)
                    flag = "  *saved"
                extra = ""
                if "ridge_bb_prec" in got:
                    extra = (f"  [ridge {got['ridge_bb_prec']:.3f} / mgc {got['mgc_bb_prec']:.3f}]")
                extra += f"  oracle {orac['bb_prec']:.3f}"
                print(f"step {step:5d}  l1 {loss.item():6.2f}  "
                      f"bb_prec {got['bb_prec']:.3f} (raw {base['bb_prec']:.3f})"
                      f"{extra}  {(time.perf_counter()-started)/60:.1f}min{flag}", flush=True)

    print(f"best held-out bb_prec = {best:.4f}   ckpt={args.ckpt}", flush=True)


if __name__ == "__main__":
    main()
