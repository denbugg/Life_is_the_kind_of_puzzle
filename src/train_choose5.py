"""Train the five-candidate chooser, and score it in correct bonds.

The target, from M407 and M409: 450 to 500 correct bonds a board crosses the
percolation knee, the shipping roster's plain top-1 delivers 330 to 348, and the
top-five shortlist holds 524. So the number to watch is not accuracy but CORRECT
BONDS after the choice, and the baseline to beat is the matcher's own top-1 on
the same boards.

M306 fixed the standard of proof: two runs of the same configuration differ by
0.028 in R@1, so nothing below about 0.03 counts, and a single seed proves
nothing. That is why the baseline here is recomputed on the same held-out boards
rather than quoted, and why the run prints the per-board spread.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from choose5 import K, Choose5, choose_loss, seam_patch
from config import CKPT_DIR, GRID as G, TRAIN_INP
from infer_coarse_field import load_rgb
from restore_tile import to_frags

N = G * G


class _NoCache(dict):
    """A dict that forgets, so the cached and uncached paths share one code."""

    def __setitem__(self, k, v):
        pass


class Boards(Dataset):
    """One board per item: its tiles and both directions' shortlists."""

    def __init__(self, files, cache=False):
        self.files = files
        self._cache = {} if cache else _NoCache()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, k):
        if k in self._cache:
            return self._cache[k]
        z = np.load(self.files[k])
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / str(z["name"]))).astype(
            np.float32)[z["inv"].astype(np.int64)]
        item = (torch.from_numpy(tiles),
                {t: (torch.from_numpy(z[f"{t}_idx"].astype(np.int64)),
                     torch.from_numpy(z[f"{t}_val"]),
                     torch.from_numpy(z[f"{t}_lab"].astype(np.int64)))
                 for t in ("h", "v")})
        self._cache[k] = item
        return item


def collate(batch):
    return batch


def board_batch(tiles, idx, val, lab, strip, dev):
    """Every fragment that HAS a true partner, as one batch of shortlists."""
    keep = (lab >= 0).nonzero(as_tuple=True)[0]
    src = keep.repeat_interleave(K)
    dst = idx[keep].reshape(-1)
    return keep, src, dst, val[keep], lab[keep]


def run_board(model, tiles, packs, strip, dev, train, none_weight=0.3):
    loss_sum, rows = 0.0, []
    for axis in ("h", "v"):
        idx, val, lab = packs[axis]
        idx, val, lab = idx.to(dev), val.to(dev), lab.to(dev)
        keep, src, dst, v, y = board_batch(tiles, idx, val, lab, strip, dev)
        if not len(keep):
            continue
        patch = seam_patch(tiles, src, dst, axis, strip).reshape(
            len(keep), K, 3, 20, 2 * strip)
        rank = torch.arange(K, device=dev, dtype=torch.float32)
        z = v - v[:, :1]
        sc = torch.stack([v / 10.0, z, rank.expand(len(keep), K),
                          (z == 0).float()], -1)
        logits = model(patch, sc)
        loss = choose_loss(logits, y, none_weight)
        loss_sum += float(loss.detach())
        if train:
            loss.backward()
        pick = logits.argmax(1)
        rows.append((int((pick == y).sum()),
                     int(((pick < K) & (pick == y)).sum()),
                     int((y == 0).sum()), len(keep)))
    return loss_sum, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--held", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--strip", type=int, default=4)
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--none-weight", type=float, default=0.3,
                    help="how much the NONE class counts in the loss. It is the right answer for 47%% of fragments, so at weight 1 the model learns to abstain and an abstention is worth zero correct bonds; at 0 it is trained only where the truth is in the shortlist")
    ap.add_argument("--encoder", default="cnn",
                    choices=("cnn", "cross"),
                    help="cnn convolves the join; cross makes the two sides of it attend to each other row by row, which no matcher in this project does -- they are all bi-encoders comparing pooled descriptors")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="choose5.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    files = sorted(Path(a.dumps).glob("*.npz"))
    if len(files) <= a.held:
        sys.exit(f"only {len(files)} dumps in {a.dumps}")
    train, held = files[a.held:], files[:a.held]
    print(f"{len(train)} train boards, {len(held)} held out, seed {a.seed}",
          flush=True)

    dev = "cuda"
    model = Choose5(a.ch, a.dim, a.strip, a.layers,
                    encoder=a.encoder).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    dl = DataLoader(Boards(train, cache=True), batch_size=1,
                    shuffle=True, collate_fn=collate, num_workers=0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, total_steps=max(a.epochs * len(train), 1), pct_start=0.15)

    def evaluate():
        model.eval()
        got = base = tot = 0
        per = []
        with torch.no_grad():
            for hb in held_ds:
                tiles, packs = hb
                tiles = tiles.to(dev)
                _l, rows = run_board(model, tiles, packs, a.strip, dev, False)
                g = sum(r[1] for r in rows)
                b0 = sum(r[2] for r in rows)
                n = sum(r[3] for r in rows)
                got += g
                base += b0
                tot += n
                per.append(g - b0)
        model.train()
        return got / len(held), base / len(held), tot / len(held), np.std(per)

    held_ds = [Boards([f])[0] for f in held]
    g, b0, tot, sd = evaluate()
    print(f"[init] chooser {g:.1f} correct bonds, matcher top-1 {b0:.1f}, "
          f"{tot:.0f} fragments with a true partner", flush=True)

    for ep in range(a.epochs):
        run = []
        for batch in dl:
            tiles, packs = batch[0]
            tiles = tiles.to(dev)
            opt.zero_grad(set_to_none=True)
            loss, _rows = run_board(model, tiles, packs, a.strip, dev, True,
                                    a.none_weight)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run.append(loss)
        if ep % a.eval_every and ep != a.epochs - 1:
            print(f"[epoch {ep}] loss {np.mean(run):.4f}", flush=True)
            continue
        g, b0, tot, sd = evaluate()
        print(f"[epoch {ep}] loss {np.mean(run):.4f}  chooser {g:.1f} against "
              f"matcher {b0:.1f} correct bonds  (per-board sd {sd:.1f})",
              flush=True)

    out = Path(CKPT_DIR) / a.out
    torch.save({"model": model.state_dict(),
                "args": {k: getattr(a, k) for k in
                         ("ch", "dim", "strip", "layers", "encoder")}}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
