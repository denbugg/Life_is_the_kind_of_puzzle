"""Curriculum gate for the recurrent Sinkhorn assembler."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from config import CACHE_DIR, CKPT_DIR
from sinkhorn_assembler import (SinkhornAssembler, decode, edge_loss,
                                permutation_loss)


def crop_batch(bag4, ids, side, rng):
    tiles, target = [], []
    for idx in ids:
        y = int(rng.integers(25 - side))
        x = int(rng.integers(25 - side))
        cells = ((np.arange(side)[:, None] + y) * 24
                 + np.arange(side)[None, :] + x).reshape(-1)
        clean_order = bag4[idx, cells]
        perm = rng.permutation(side * side)
        tiles.append(clean_order[perm])
        inv = np.empty(side * side, np.int64)
        inv[perm] = np.arange(side * side)
        # cell c needs the input-tile index whose original crop index is c.
        target.append(inv)
    return np.stack(tiles), np.stack(target)


def metrics(layout, target, side):
    place = float((layout == target).mean())
    # Convert input tile id -> true crop cell id, then inspect realised bonds.
    vals = np.empty_like(target)
    for b in range(len(target)):
        vals[b, target[b]] = np.arange(side * side)
    board = np.take_along_axis(vals, layout, axis=1).reshape(-1, side, side)
    adj = ((board[:, :, 1:] == board[:, :, :-1] + 1).sum()
           + (board[:, 1:] == board[:, :-1] + side).sum())
    return place, float(adj / (len(board) * 2 * side * (side - 1)))


def edge_metrics(edges, target, side):
    """Board-wide directed-neighbour recall, including candidates off-crop.

    This separates representation quality from the recurrent assignment.  If
    edge R@K is healthy while placement is not, more encoder training cannot
    fix the bottleneck; the decoder has to change.
    """
    e = edges.float().cpu().numpy()
    b, _, n, _ = e.shape
    grid = target.reshape(b, side, side)
    ranks = []
    for axis, anchors, neighbours in (
            (0, grid[:, :, :-1], grid[:, :, 1:]),
            (1, grid[:, :-1], grid[:, 1:])):
        bi = np.broadcast_to(np.arange(b)[:, None, None], anchors.shape)
        score = e[bi, axis, anchors]
        truth = np.take_along_axis(score, neighbours[..., None], axis=-1)[..., 0]
        # Ties count pessimistically.  Self is not a valid candidate.
        self_score = np.take_along_axis(score, anchors[..., None], axis=-1)[..., 0]
        score = score.copy()
        np.put_along_axis(score, anchors[..., None], -np.inf, axis=-1)
        del self_score
        ranks.append((score > truth[..., None]).sum(-1).reshape(-1))
    rank = np.concatenate(ranks)
    return float((rank == 0).mean()), float((rank < 5).mean()), float((rank < 20).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", type=int, default=6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--teacher-start", type=float, default=0.35)
    ap.add_argument("--teacher-end", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-boards", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="sinkhorn_asm_s6.pt")
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    z = np.load(Path(CACHE_DIR) / "field_cache.npz")
    bag4 = z["bag8"]
    train_ids = np.arange(len(bag4) - 300)
    eval_ids = np.arange(len(bag4) - 300, len(bag4))[:a.eval_boards]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SinkhornAssembler(a.d, a.rounds, a.blocks).to(dev)
    if a.resume:
        rp = Path(a.resume)
        if not rp.is_absolute():
            rp = Path(CKPT_DIR) / rp
        model.load_state_dict(torch.load(rp, map_location=dev,
                                         weights_only=False)["model"])
        print(f"resumed {rp}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(dev)
    best = 0.0
    fixed_rng = np.random.default_rng(9876)
    ex, ey = crop_batch(bag4, eval_ids, a.side, fixed_rng)

    def evaluate():
        model.eval()
        ps, aa, ee1, ee5, ee20 = [], [], [], [], []
        with torch.no_grad():
            for k in range(0, len(ex), a.batch):
                x = torch.from_numpy(ex[k:k + a.batch]).to(dev)
                logits, _, edges = model(x, a.side)
                p, q = metrics(decode(logits), ey[k:k + a.batch], a.side)
                e1, e5, e20 = edge_metrics(edges, ey[k:k + a.batch], a.side)
                ps.append(p * len(x)); aa.append(q * len(x))
                ee1.append(e1 * len(x)); ee5.append(e5 * len(x)); ee20.append(e20 * len(x))
        model.train()
        return (sum(ps) / len(ex), sum(aa) / len(ex), sum(ee1) / len(ex),
                sum(ee5) / len(ex), sum(ee20) / len(ex))

    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params; "
          f"side {a.side}; chance {1/(a.side*a.side):.4f}", flush=True)
    t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            p, adj, e1, e5, e20 = evaluate()
            print(f"[{step}] place {p:.4f} adjacency {adj:.4f} "
                  f"edge R@1/5/20 {e1:.3f}/{e5:.3f}/{e20:.3f} "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)
            if p > best:
                best = p
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "place": p, "adjacency": adj},
                           Path(CKPT_DIR) / a.out)
        if step == a.steps:
            break
        ids = rng.choice(train_ids, a.batch, replace=False)
        bx, by = crop_batch(bag4, ids, a.side, rng)
        x = torch.from_numpy(bx).to(dev)
        y = torch.from_numpy(by).to(dev)
        teacher = torch.nn.functional.one_hot(y, a.side * a.side).float()
        frac = step / max(a.steps - 1, 1)
        tw = a.teacher_start * (1 - frac) + a.teacher_end * frac
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev, dtype=torch.float16):
            logits, history, edges = model(x, a.side, teacher=teacher,
                                           teacher_weight=tw)
            ploss = permutation_loss(history, y)
            eloss = edge_loss(edges, y, a.side)
            loss = ploss + eloss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % 50 == 0:
            print(f"  loss {float(loss.detach()):.4f} perm {float(ploss.detach()):.4f} "
                  f"edge {float(eloss.detach()):.4f} teacher {tw:.3f}", flush=True)
    print(f"best placement {best:.4f}")


if __name__ == "__main__":
    main()
