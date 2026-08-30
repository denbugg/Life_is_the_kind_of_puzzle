"""Train the seam verifier and score it on PRECISION AT THE HARVEST VOLUME.

The number to watch is not accuracy and not AUC. M456 measured that the
connected block runs 350 correct fragments at edge precision 1.00, 186 at 0.99,
65 at 0.95 and 18 at the 0.746 the shipping harvest delivers, so the only
question about any scorer is what precision it holds at the volume actually
harvested. Every epoch here reports PRECISION AT 430 EDGES A BOARD, against the
matcher's own precision at the same volume on the same boards, recomputed rather
than quoted.

The data are the top-5 dumps: 1104 rows a board, five candidates each, labelled
by whether the candidate is the true neighbour. About one in eleven is positive.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from choose5 import K, seam_patch
from config import CKPT_DIR, GRID as G, TRAIN_INP
from restore_tile import to_frags
from verify_pair import (SeamVerifier, focal_bce, precision_at_k,
                         topk_hinge)

N = G * G


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


class Boards(Dataset):
    def __init__(self, files, cache=True):
        self.files = files
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, k):
        hit = None if self._cache is None else self._cache.get(k)
        if hit is None:
            z = np.load(self.files[k])
            tiles = to_frags(load_rgb(Path(TRAIN_INP) / str(z["name"])))[
                z["inv"].astype(np.int64)].astype(np.uint8)
            packs = {t: (z[f"{t}_idx"].astype(np.int64),
                         z[f"{t}_val"].astype(np.float32),
                         z[f"{t}_lab"].astype(np.int64))
                     for t in ("h", "v")}
            hit = (tiles, packs)
            if self._cache is not None:
                self._cache[k] = hit
        return hit


def collate(batch):
    return batch


def board_pairs(tiles, packs, strip, dev):
    """Every shortlisted pair of one board, as patches, features and labels."""
    out = []
    for axis in ("h", "v"):
        idx, val, lab = packs[axis]
        keep = np.nonzero(lab >= 0)[0]
        if not len(keep):
            continue
        ii = torch.from_numpy(idx[keep]).to(dev)
        vv = torch.from_numpy(val[keep]).to(dev)
        src = torch.from_numpy(keep).to(dev).repeat_interleave(K)
        dst = ii.reshape(-1)
        patch = seam_patch(tiles, src, dst, axis, strip)
        rank = torch.arange(K, device=dev, dtype=torch.float32)
        z = vv - vv[:, :1]
        feats = torch.stack([vv / 10.0, z, rank.expand(len(keep), K),
                             (z == 0).float(),
                             vv.mean(1, keepdim=True).expand(-1, K) / 10.0,
                             (vv[:, :1] - vv[:, -1:]).expand(-1, K)], -1)
        y = torch.zeros(len(keep), K, device=dev)
        hit = torch.from_numpy(lab[keep]).to(dev)
        ok = hit < K
        y[torch.arange(len(keep), device=dev)[ok], hit[ok]] = 1.0
        out.append((patch, feats.reshape(-1, feats.shape[-1]), y.reshape(-1)))
    if not out:
        return None
    return (torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out]),
            torch.cat([o[2] for o in out]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--held", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--strip", type=int, default=4)
    ap.add_argument("--loss", choices=("focal", "hinge"), default="hinge",
                    help="hinge puts the loss at the operating point: the k-th "
                         "ranked score is the threshold the harvest uses, so "
                         "only a positive below it or a negative above it "
                         "costs anything. The focal arm moved precision 0.0086 "
                         "in three epochs while its own loss barely fell")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=2048,
                    help="pairs per forward; a whole board does not fit")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--volume", type=int, default=430,
                    help="edges a board the precision is read at (M456)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="verify_pair.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    files = sorted(Path(a.dumps).glob("*.npz"))
    if len(files) <= a.held:
        sys.exit(f"only {len(files)} dumps in {a.dumps}")
    train, held = files[a.held:], files[:a.held]
    print(f"{len(train)} train boards, {len(held)} held out", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeamVerifier(a.ch, a.blocks, 6, a.strip).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    dl = DataLoader(Boards(train), batch_size=1, shuffle=True,
                    collate_fn=collate, num_workers=0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, total_steps=max(a.epochs * len(train), 1), pct_start=0.15)
    held_ds = [Boards([f], cache=False)[0] for f in held]

    def evaluate():
        model.eval()
        ours, base = [], []
        with torch.no_grad():
            for tiles, packs in held_ds:
                t = torch.from_numpy(tiles).float().to(dev)
                got = board_pairs(t, packs, a.strip, dev)
                if got is None:
                    continue
                patch, feats, y = got
                s = []
                for k in range(0, len(patch), a.chunk):
                    s.append(model(patch[k:k + a.chunk],
                                   feats[k:k + a.chunk]))
                s = torch.cat(s)
                ours.append(precision_at_k(s, y, a.volume))
                base.append(precision_at_k(feats[:, 0], y, a.volume))
        model.train()
        return float(np.mean(ours)), float(np.mean(base))

    p, b0 = evaluate()
    print(f"[init] verifier {p:.4f} against matcher {b0:.4f} precision at "
          f"{a.volume} edges a board", flush=True)
    best = p
    for ep in range(a.epochs):
        run = []
        for batch in dl:
            tiles, packs = batch[0]
            t = torch.from_numpy(tiles).float().to(dev)
            got = board_pairs(t, packs, a.strip, dev)
            if got is None:
                continue
            patch, feats, y = got
            opt.zero_grad(set_to_none=True)
            # A board is about eleven thousand pairs and one activation of a
            # 20x16 patch through a 96-channel trunk does not fit; the first
            # attempt died of CUDA OOM on its first batch. The threshold the
            # hinge needs is a property of the WHOLE board, so it is taken once
            # without gradients and then held fixed while the forward and
            # backward run in chunks, which keeps the loss exactly what it was.
            with torch.no_grad():
                scores = torch.cat([model(patch[k:k + a.chunk],
                                          feats[k:k + a.chunk])
                                    for k in range(0, len(patch), a.chunk)])
            kk = min(a.volume, scores.numel() - 1)
            thr = torch.kthvalue(-scores.flatten(), kk).values.neg()
            tot = 0.0
            for k in range(0, len(patch), a.chunk):
                out = model(patch[k:k + a.chunk], feats[k:k + a.chunk])
                yy = y[k:k + a.chunk]
                if a.loss == "hinge":
                    pos = yy > 0.5
                    below = torch.relu(a.margin - (out - thr))[pos]
                    above = torch.relu(a.margin + (out - thr))[~pos]
                    loss = (below.sum() + above.sum()) / max(len(out), 1)
                else:
                    loss = focal_bce(out, yy, a.gamma, a.pos_weight)
                (loss * len(out) / len(patch)).backward()
                tot += float(loss.detach()) * len(out) / len(patch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run.append(tot)
        p, b0 = evaluate()
        print(f"[epoch {ep}] loss {np.mean(run):.4f}  verifier {p:.4f} "
              f"against matcher {b0:.4f}", flush=True)
        torch.save({"model": model.state_dict(),
                    "args": {k: getattr(a, k)
                             for k in ("ch", "blocks", "strip")}},
                   Path(CKPT_DIR) / a.out)
        if p > best:
            best = p
            torch.save({"model": model.state_dict(),
                        "args": {k: getattr(a, k)
                                 for k in ("ch", "blocks", "strip")}},
                       Path(CKPT_DIR) / (a.out[:-3] + "_best.pt"))
            print(f"  best so far {p:.4f}", flush=True)
    print(f"best precision at {a.volume} edges: {best:.4f}")


if __name__ == "__main__":
    main()
