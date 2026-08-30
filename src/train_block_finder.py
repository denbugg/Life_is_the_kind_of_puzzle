"""Train the small-block finder and read it as PRECISION OF WHOLE BLOCKS.

The gate is not how many blocks come out. M456 prices what converts: a connected
block of correct fragments runs 350 at edge precision 1.00 and 18 at the 0.746
the harvest delivers, so a generator of ISLANDS is worth exactly what its purity
is worth. The two numbers reported every evaluation are therefore

    perfect   the fraction of emitted blocks that are correct ENTIRELY
    bonds     the fraction of a block's internal joins that are true

and a run that emits many nearly-right blocks has failed, because a nearly-right
island is a weld waiting to happen.

Controls. The bag is SHUFFLED every time, so the caches' storage order -- which
happens to be the true cell order -- cannot leak; the neighbour maps are passed
through the same permutation. Chance for a k x k block is the probability that
k*k fragments drawn from 576 happen to form a square in the right arrangement,
which is far below anything that will be printed, so the honest reference is the
BONDS column against the shortlist's own recall of about 0.32 at depth one.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from block_finder import (BlockFinder, adjacency_reward, block_bonds,
                          block_is_perfect, decode_block)
from config import CACHE_DIR, CKPT_DIR, GRID as G

N = G * G
HELD = 300


class Boards:
    def __init__(self, cache, seam, rng):
        z = np.load(Path(CACHE_DIR) / cache)
        self.bag8, self.stats = z["bag8"], z["stats"]
        d = Path(CACHE_DIR) / seam
        self.idx = np.load(d / "idx.npy", mmap_mode="r")
        self.val = np.load(d / "val.npy", mmap_mode="r")
        names_a, names_b = z["names"], np.load(d / "names.npy")
        if not (names_a[:len(names_b)] == names_b).all():
            raise SystemExit("the two caches are not in the same board order")
        self.rng = rng

    def __len__(self):
        return len(self.bag8)

    def draw(self, ids, dev, floor=-12.0, fixed=False):
        view, stat, seam = [], [], np.full((2, len(ids), N, N), floor,
                                           np.float32)
        right, down, okr, okd, perms = [], [], [], [], []
        for n, b in enumerate(ids):
            rng = (np.random.default_rng(5000 + int(b)) if fixed else self.rng)
            perm = rng.permutation(N)          # board cell of each bag slot
            inv = np.empty(N, np.int64)
            inv[perm] = np.arange(N)           # bag slot of each board cell
            view.append(self.bag8[b][perm])
            stat.append(self.stats[b][perm])
            rows = np.arange(N)[:, None]
            for ax in range(2):
                cand = np.asarray(self.idx[b, ax], np.int64)[perm]
                v = np.asarray(self.val[b, ax], np.float32)[perm]
                seam[ax, n][rows, inv[cand]] = v
            r = np.where((perm % G) < G - 1, perm + 1, perm)
            d_ = np.where((perm // G) < G - 1, perm + G, perm)
            right.append(inv[r])
            down.append(inv[d_])
            okr.append(((perm % G) < G - 1).astype(np.float32))
            okd.append(((perm // G) < G - 1).astype(np.float32))
            perms.append(perm)

        def f(a, dt=torch.float32):
            return torch.from_numpy(np.asarray(a)).to(dev, dt)

        return (f(view), f(stat),
                (f(seam[0]), f(seam[1])),
                f(right, torch.long), f(okr), f(down, torch.long), f(okd),
                np.asarray(perms))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="field_cache.npz")
    ap.add_argument("--seam", default="verify_top5_v2")
    ap.add_argument("--side", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--bag-layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--grad-rounds", type=int, default=2)
    ap.add_argument("--assign-rounds", type=int, default=2)
    ap.add_argument("--eval-boards", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--train-boards", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="block_finder.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    data = Boards(a.cache, a.seam, rng)
    n = len(data)
    n_train = min(n - HELD, a.train_boards) if a.train_boards else n - HELD
    tr = np.arange(n_train)
    ev = np.arange(n - HELD, n)[:a.eval_boards]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlockFinder(a.side, a.d, a.bag_layers, a.blocks, a.heads).to(dev)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters, "
          f"block {a.side}x{a.side}, {len(tr)} train boards", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    joins = 2 * a.side * (a.side - 1)

    @torch.no_grad()
    def evaluate():
        model.eval()
        perfect, bonds = [], []
        for i in ev:
            v, s, sm, r, okr, d_, okd, perms = data.draw([i], dev, fixed=True)
            logp = model(v, s, sm, a.rounds, iters=10,
                         assign_rounds=a.assign_rounds,
                         grad_rounds=0)
            slots = decode_block(logp[0])
            cells = perms[0][slots]
            perfect.append(float(block_is_perfect(cells, a.side, G)))
            ok, tot = block_bonds(cells, a.side, G)
            bonds.append(ok / tot)
        model.train()
        return float(np.mean(perfect)), float(np.mean(bonds))

    p0, b0 = evaluate()
    print(f"[init] perfect {p0:.4f}  bonds {b0:.4f}  "
          f"(a block has {joins} internal joins; the shortlist's own top-1 "
          f"recall is about 0.32)", flush=True)
    best = b0
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        nb = len(perm) // a.batch
        if a.max_batches:
            nb = min(nb, a.max_batches)
        run, t0 = [], time.time()
        for k in range(nb):
            ids = np.sort(perm[k * a.batch:(k + 1) * a.batch])
            v, s, sm, r, okr, d_, okd, _p = data.draw(ids, dev)
            opt.zero_grad(set_to_none=True)
            logp = model(v, s, sm, a.rounds, iters=10,
                         assign_rounds=a.assign_rounds,
                         grad_rounds=a.grad_rounds)
            reward = adjacency_reward(logp.exp(), a.side, r, okr, d_, okd)
            loss = -reward / joins
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run.append(float(-loss.detach()))
        msg = (f"[epoch {ep}] expected bonds {np.mean(run):.4f} "
               f"{(time.time()-t0)/60:.1f} min")
        if ep % a.eval_every == 0 or ep == a.epochs - 1:
            p, b = evaluate()
            msg += f"  perfect {p:.4f}  bonds {b:.4f}"
            ck = {"model": model.state_dict(),
                  "args": {k: getattr(a, k) for k in
                           ("side", "d", "blocks", "bag_layers", "heads",
                            "rounds", "assign_rounds")},
                  "perfect": p, "bonds": b}
            torch.save(ck, Path(CKPT_DIR) / a.out)
            if b > best:
                best = b
                torch.save(ck, Path(CKPT_DIR) / (a.out[:-3] + "_best.pt"))
                msg += "  *"
        print(msg, flush=True)
    print(f"best held-out internal bonds: {best:.4f}")


if __name__ == "__main__":
    main()
