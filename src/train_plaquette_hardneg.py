"""Train the 2x2 verifier on pixels, against hard negatives the matcher supplies.

Why this and not another pairwise scorer
----------------------------------------
Every scorer in this project judges ONE seam.  Three re-rankers have now
plateaued at the retriever's own rate (M105, M107, M113, M164), and the joint
head's gain does not survive into the full cost matrix (M164, and the two
switched-off knobs measured today).  Meanwhile M150 measured that matching 2x2
blocks runs at precision 0.709 where matching single tiles runs at 0.438 -- the
object-size effect is real once the object is correct, and the thing this repo
has never had is a way to MAKE correct 2x2 objects.  Loop closure, the
arithmetic way of making them from pairwise edges, cuts 598 edges at 0.438 down
to 22 at 0.628 (M92): it discards almost everything.

A 2x2 verifier reads evidence no pairwise score can reach.  Four fragments meeting
at a point constrain each other in two dimensions at once, and the junction where
all four corners touch is visible to nothing that scores a single seam.
`plaquette.PlaquetteNet` was written for exactly this and never trained.

Negatives
---------
Random negatives would teach nothing: at inference the candidates come from the
matcher's shortlists, so a negative must be a quad that the matcher finds
plausible.  Each board's calibrated cost path is run once and its top-K right
and down candidates cached; a negative then replaces one to three slots of a
true quad with a highly-ranked wrong candidate for that slot.  Single-slot
errors are the hardest and are sampled most often, because that is what the
verifier will have to reject.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import CACHE_DIR, CKPT_DIR, GRID as G, TRAIN_INP
from plaquette import PlaquetteNet, count_params
from restore_tile import to_frags
from seam_cost import costs_from_model
from seam_embed import SeamEmbed

N = G * G


def load_rgb(path):
    return np.ascontiguousarray(cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1])


def load_matcher(name, dev):
    ck = torch.load(Path(CKPT_DIR) / name, map_location=dev, weights_only=False)
    a = ck["args"]
    m = SeamEmbed(a["ch"], a["blocks"], a["dim"], a["strip"],
                  a.get("head", "global")).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def build_cache(matcher, names, inv, k, dev, out):
    """Per board: the corrupted tiles in TRUE order, and the matcher's shortlists."""
    tiles = np.empty((len(names), N, 20, 20, 3), np.uint8)
    right = np.empty((len(names), N, k), np.int16)
    down = np.empty((len(names), N, k), np.int16)
    t0 = time.time()
    for b, nm in enumerate(names):
        t = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            inv[b].astype(np.int64)]
        tiles[b] = t.astype(np.uint8)
        ch, cv_ = costs_from_model(matcher, t, device=dev)
        for src, dst in ((ch, right), (cv_, down)):
            s = src.copy()
            np.fill_diagonal(s, np.inf)
            dst[b] = np.argpartition(s, k, axis=1)[:, :k].astype(np.int16)
        if (b + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  cached {b+1}/{len(names)}  {el:.0f}s", flush=True)
    np.savez(out, tiles=tiles, right=right, down=down)
    return {"tiles": tiles, "right": right, "down": down}


def quad_ids(rng, cache, n_boards, batch, k):
    """Sample (board, four tile ids, label). Half positive, half hard negative."""
    b = rng.integers(0, n_boards, batch)
    r = rng.integers(0, G - 1, batch)
    c = rng.integers(0, G - 1, batch)
    tl = (r * G + c).astype(np.int64)
    ids = np.stack([tl, tl + 1, tl + G, tl + G + 1], 1)
    lab = (rng.random(batch) < 0.5).astype(np.float32)

    neg = np.nonzero(lab == 0.0)[0]
    # one wrong slot is the hard case and the one inference actually produces;
    # two and three keep the model honest about compound errors
    how_many = rng.choice([1, 2, 3], size=neg.size, p=[0.65, 0.25, 0.10])
    for idx, m in zip(neg, how_many):
        slots = rng.choice([1, 2, 3], size=m, replace=False)
        for s in slots:
            anchor, table = (ids[idx, 0], cache["right"]) if s == 1 else \
                            (ids[idx, 0], cache["down"]) if s == 2 else \
                            (ids[idx, 2], cache["right"])
            pool = table[b[idx], anchor]
            pick = int(pool[rng.integers(0, k)])
            tries = 0
            while pick == ids[idx, s] and tries < 8:
                pick = int(pool[rng.integers(0, k)])
                tries += 1
            ids[idx, s] = pick
    return b, ids, lab


def gather(cache, b, ids, dev):
    t = cache["tiles"][b[:, None], ids]                      # (B,4,20,20,3)
    x = torch.from_numpy(np.ascontiguousarray(t)).to(dev, torch.float32)
    return x.permute(0, 1, 4, 2, 3) / 255.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matcher", default="seam_embed_v1.pt")
    ap.add_argument("--boards", type=int, default=400)
    ap.add_argument("--k", type=int, default=8, help="shortlist depth for negatives")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--cache", default="plaquette_cache.npz")
    ap.add_argument("--out", default="plaquette_v1.pt")
    a = ap.parse_args()

    dev = "cuda"
    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = [str(x) for x in blob["names"][: a.boards]]
    inv = blob["inv"][: a.boards]
    cpath = Path(CACHE_DIR) / a.cache
    if cpath.exists():
        z = np.load(cpath)
        cache = {"tiles": z["tiles"], "right": z["right"], "down": z["down"]}
        print(f"cache loaded: {cache['tiles'].shape[0]} boards", flush=True)
    else:
        print(f"building cache for {len(names)} boards", flush=True)
        cache = build_cache(load_matcher(a.matcher, dev), names, inv, a.k, dev, cpath)
    n_boards = cache["tiles"].shape[0]
    hold = max(1, n_boards // 10)
    train_b, val_b = n_boards - hold, hold

    model = PlaquetteNet(a.width).to(dev)
    print(f"PlaquetteNet width {a.width}, {count_params(model):,} parameters",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(0)
    vrng = np.random.default_rng(99)

    def evaluate():
        model.eval()
        hit, tot = 0, 0
        with torch.no_grad():
            for _ in range(8):
                b, ids, lab = quad_ids(vrng, cache, val_b, 256, a.k)
                b = b + train_b
                with torch.autocast("cuda", torch.float16):
                    p = model(gather(cache, b, ids, dev)).float()
                hit += int(((p > 0).cpu().numpy() == (lab > 0.5)).sum())
                tot += lab.size
        model.train()
        return hit / tot

    run, t0, best = [], time.time(), 0.0
    for step in range(1, a.steps + 1):
        b, ids, lab = quad_ids(rng, cache, train_b, a.batch, a.k)
        y = torch.from_numpy(lab).to(dev)
        with torch.autocast("cuda", torch.float16):
            logit = model(gather(cache, b, ids, dev))
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run.append(float(loss.detach()))
        if step % a.eval_every == 0 or step == a.steps:
            acc = evaluate()
            print(f"step {step:6d}  loss {np.mean(run[-200:]):.4f}  "
                  f"val acc {acc:.4f}  {time.time() - t0:.0f}s", flush=True)
            if acc >= best:
                best = acc
                torch.save({"model": model.state_dict(),
                            "args": vars(a), "step": step,
                            "eval": {"acc": acc}}, Path(CKPT_DIR) / a.out)
    print(json.dumps({"best_val_acc": best, "out": a.out}), flush=True)


if __name__ == "__main__":
    main()
