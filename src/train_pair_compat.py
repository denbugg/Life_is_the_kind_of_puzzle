"""Train the joint full-pair compatibility scorer on REAL adjacent fragments.

Positives are true grid neighbours of a train board, taken only where BOTH
endpoints exceed the label-margin threshold (top-50% margin -> 0.996 matching
accuracy).  Negatives mix uniform in-board tiles with the hardest candidates
under the cheap ridge seam cost, so the model spends capacity on the confusable
tail rather than on obvious mismatches.

Optionally consumes tiles already restored by train_restore_tile.py, which lets
the two levers be measured separately and then stacked.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import CACHE_DIR, CKPT_DIR, GRID as G, NFRAG as N, TRAIN_INP, VAL_COUNT
from pair_compat import PairCompat, dense_scores, join_h, join_v, pair_metrics
from restore_tile import TileRestorer, ridge_cost, ridge_cost_torch, to_frags


def load_tiles(name: str) -> np.ndarray:
    img = cv2.imread(str(Path(TRAIN_INP) / name), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {name}")
    return to_frags(np.ascontiguousarray(img[:, :, ::-1])).astype(np.float32)


class Boards(Dataset):
    """One item = one board's tiles in true grid order plus a per-position mask."""

    def __init__(self, names, inv, margin, thr):
        self.names, self.inv, self.margin, self.thr = names, inv, margin, thr

    def __len__(self):
        return len(self.names)

    def __getitem__(self, k):
        tiles = load_tiles(self.names[k])[self.inv[k].astype(np.int64)]
        good = (self.margin[k] >= self.thr).astype(np.float32)
        return torch.from_numpy(tiles).permute(0, 3, 1, 2), torch.from_numpy(good)


def sample_rows(good: np.ndarray, axis: str, n_rows: int, rng) -> np.ndarray:
    step = 1 if axis == "h" else G
    valid = [p for p in range(N)
             if ((p % G) != G - 1 if axis == "h" else p < N - G)
             and good[p] > 0 and good[p + step] > 0]
    if not valid:
        return np.empty(0, np.int64)
    return rng.choice(np.array(valid), size=min(n_rows, len(valid)), replace=False)


def build_negatives(cost: np.ndarray, anchors: np.ndarray, pos: np.ndarray,
                    n_neg: int, hard_frac: float, rng) -> np.ndarray:
    """(len(anchors), n_neg) negative tile ids: part hardest by ridge cost, part uniform."""
    n_hard = int(round(n_neg * hard_frac))
    out = np.empty((len(anchors), n_neg), np.int64)
    order = np.argsort(cost[anchors], axis=1)
    for r, (a, p) in enumerate(zip(anchors, pos)):
        hard = [c for c in order[r] if c != a and c != p][:n_hard]
        pool = rng.integers(0, N, size=n_neg * 3)
        rand = [c for c in pool if c != a and c != p][: n_neg - len(hard)]
        out[r] = np.array(hard + rand, np.int64)[:n_neg]
    return out


@torch.no_grad()
def evaluate(model, names, inv, device, n_boards, restorer):
    model.eval()
    rows = []
    for k in range(n_boards):
        tiles = load_tiles(names[k])[inv[k].astype(np.int64)]
        x = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(device)
        if restorer is not None:
            with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                x = torch.cat([restorer(x[i:i + 288]) for i in range(0, len(x), 288)]).float().clamp(0, 255)
        rec = {}
        npt = x.permute(0, 2, 3, 1).cpu().numpy()
        for axis in ("h", "v"):
            # Full 331k-pair dense scoring costs minutes per board; restrict to
            # the ridge top-K, which already carries the recall we can act on.
            sl = np.argsort(ridge_cost(npt, axis=axis), axis=1)[:, :64]
            rec[axis] = pair_metrics(dense_scores(model, x, axis, shortlist=sl), axis)
        rows.append({k2: 0.5 * (rec["h"][k2] + rec["v"][k2]) for k2 in rec["h"]})
    model.train()
    return {k2: float(np.mean([r[k2] for r in rows])) for k2 in rows[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path(CACHE_DIR) / "restore_labels.npz")
    ap.add_argument("--ckpt", type=Path, default=Path(CKPT_DIR) / "pair_compat.pt")
    ap.add_argument("--restorer", type=Path, default=None,
                    help="optional TileRestorer checkpoint applied before pairing")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--boards-per-batch", type=int, default=2)
    ap.add_argument("--rows-per-board", type=int, default=96)
    ap.add_argument("--negatives", type=int, default=48)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--anchor-chunk", type=int, default=32,
                    help="anchors per backward; keeps activations inside 8 GB")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--margin-quantile", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-boards", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    blob = np.load(args.labels, allow_pickle=True)
    names, inv, margin = blob["names"], blob["inv"], blob["margin"]
    thr = float(np.quantile(margin, args.margin_quantile))
    n_val = min(VAL_COUNT, len(names) // 4)
    tr, va = slice(0, len(names) - n_val), slice(len(names) - n_val, len(names))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    restorer = None
    if args.restorer is not None:
        blob_r = torch.load(args.restorer, map_location=device, weights_only=False)
        ra = blob_r["args"]
        restorer = TileRestorer(ra["ch"], ra["blocks"], ra.get("residual", False)).to(device)
        restorer.load_state_dict(blob_r["model"])
        restorer.eval()
        for p in restorer.parameters():
            p.requires_grad_(False)
        print(f"restorer loaded: bb_prec {blob_r.get('bb_prec'):.4f} @ step {blob_r.get('step')}", flush=True)

    model = PairCompat(args.ch).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  device={device}  "
          f"margin_thr={thr:.4f}", flush=True)

    dl = DataLoader(Boards(names[tr], inv[tr], margin[tr], thr), batch_size=args.boards_per_batch,
                    shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best, step, started = -1.0, 0, time.perf_counter()
    while step < args.steps:
        for tiles_b, good_b in dl:
            if step >= args.steps:
                break
            # Gradient is accumulated chunk by chunk.  Holding the graph for the
            # whole step is what kills this box: fwd+bwd costs 0.234 s at 2112
            # pairs and 17.4 s at 6272, because the activations stop fitting in
            # 8 GB and silently spill into WDDM shared memory.
            opt.zero_grad(set_to_none=True)
            losses = []
            n_chunks = max(1, len(tiles_b)) * 2 * max(1, args.rows_per_board // args.anchor_chunk)
            for tiles, good in zip(tiles_b, good_b):
                tiles = tiles.to(device, non_blocking=True)
                if restorer is not None:
                    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                        tiles = torch.cat([restorer(tiles[i:i + 288])
                                           for i in range(0, len(tiles), 288)]).float().clamp(0, 255)
                gnp = good.numpy()
                for axis in ("h", "v"):
                    step_off = 1 if axis == "h" else G
                    anchors = sample_rows(gnp, axis, args.rows_per_board, rng)
                    if len(anchors) == 0:
                        continue
                    pos = anchors + step_off
                    # on GPU: the numpy einsum version costs seconds per step
                    with torch.no_grad():
                        cost = ridge_cost_torch(tiles, axis=axis).cpu().numpy()
                    neg = build_negatives(cost, anchors, pos, args.negatives, args.hard_frac, rng)
                    cand = np.concatenate([pos[:, None], neg], axis=1)          # (R, 1+M)
                    join = join_h if axis == "h" else join_v
                    for s in range(0, len(anchors), args.anchor_chunk):
                        sub = cand[s:s + args.anchor_chunk]
                        rows_a = anchors[s:s + args.anchor_chunk]
                        a_idx = torch.from_numpy(np.repeat(rows_a, sub.shape[1])).to(device)
                        c_idx = torch.from_numpy(sub.reshape(-1)).to(device)
                        with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                            logit = model(join(tiles[a_idx], tiles[c_idx])).view(len(rows_a), -1)
                            loss = F.cross_entropy(
                                logit.float(),
                                torch.zeros(len(rows_a), dtype=torch.long, device=device)) / n_chunks
                        scaler.scale(loss).backward()
                        losses.append(loss.detach() * n_chunks)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(model, names[va], inv[va], device, args.eval_boards, restorer)
                flag = ""
                if m["bb_prec"] > best:
                    best = m["bb_prec"]
                    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "args": vars(args),
                                "step": step, "bb_prec": best}, args.ckpt)
                    flag = "  *saved"
                print(f"step {step:5d}  loss {loss.item():6.3f}  bb_prec {m['bb_prec']:.3f}  "
                      f"R@1 {m['R1']:.3f}  R@20 {m['R20']:.3f}  "
                      f"{(time.perf_counter()-started)/60:.1f}min{flag}", flush=True)

    print(f"best held-out bb_prec = {best:.4f}   ckpt={args.ckpt}", flush=True)


if __name__ == "__main__":
    main()
