"""Decode the real seam evidence inside a permutation, with no training at all.

The argument
-----------
Every assembler this project has shipped GROWS a graph: rank edges, union them,
and hope none is false. M456 prices that exactly -- the connected block holds 350
fragments at edge precision 1.00 and 18 at the 0.746 the harvest delivers -- and
the reason a false edge is so expensive is that it welds two correct islands at
a wrong relative offset, destroying structure rather than merely failing to add
any.

Message passing under a HARD permutation constraint cannot weld. Beliefs are
re-projected onto the doubly stochastic set every round, and that set has
already spent each tile exactly once, so a false edge must beat every other
claim on that tile instead of merely being the best local score. Loopy BP was
tried here and has no such projection; M403's Sinkhorn was a training loss
inside a matcher, never a decoder over these matrices.

There is nothing learned in this file. The evidence is the frozen top-5
candidate cache of two seam matchers -- top-1 recall 0.3045, top-5 0.4669, about
515 true directed links available a board -- and the only question is whether a
global decoder converts more of it than greedy growth does.

Symmetry has to be broken by hand: with uniform beliefs every message is uniform
and the iteration sits at a fixed point, so each board is run from several noisy
starts. Restarts are chosen by realised seam energy, which is legitimate because
the choice is among a handful of COMPLETE layouts and never a free search over
permutations.

Read against the shipping pipeline on held-out boards: placement 0.0251,
adjacency 0.2890.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from config import CACHE_DIR, GRID as G

N = G * G
EPS = 1e-9
VOLUMES = (150, 250, 430, 600)


def load_board(cache, b):
    """Dense right/down weight matrices from the sparse top-5 candidates.

    Each row is a distribution over that tile's five candidates, so a message is
    probability mass moved along plausible neighbours and the whole scheme stays
    in the space of beliefs.
    """
    idx = np.asarray(cache["idx"][b], np.int64)     # (2, N, 5)
    val = np.asarray(cache["val"][b], np.float32)
    out = []
    for ax in range(2):
        w = torch.softmax(torch.from_numpy(val[ax]), dim=1)
        m = torch.zeros(N, N)
        m.scatter_(1, torch.from_numpy(idx[ax]), w)
        out.append(m)
    return out


def score_matrices(cache, b, floor=-20.0):
    """Raw scores with a floor off the shortlist, for ranking whole layouts."""
    idx = np.asarray(cache["idx"][b], np.int64)
    val = np.asarray(cache["val"][b], np.float32)
    out = []
    for ax in range(2):
        m = torch.full((N, N), floor)
        m.scatter_(1, torch.from_numpy(idx[ax]), torch.from_numpy(val[ax]))
        out.append(m)
    return out


def load_dense(path, topk):
    """The pipeline's OWN fused matrices, truncated to the same shortlist depth.

    The top-5 cache carries two matchers; the shipping pipeline fuses eighteen
    scorers, the centred square re-ranking, the chooser and the verifier. Running
    the decoder on that evidence is the only comparison that isolates GREEDY
    GROWTH versus GLOBAL DECODING, with everything upstream held equal.
    """
    z = np.load(path)
    out = []
    for key in ("H", "V"):
        m = torch.from_numpy(z[key].astype(np.float32))
        m.fill_diagonal_(-1e4)
        v, i = m.topk(topk, dim=1)
        w = torch.zeros(N, N)
        w.scatter_(1, i, torch.softmax(v, 1))
        raw = torch.full((N, N), -20.0)
        raw.scatter_(1, i, v)
        out.append((w, raw))
    return [o[0] for o in out], [o[1].numpy() for o in out]


def inject_islands(W, S, keep):
    """Make the globally strongest edges near-deterministic in the messages.

    M449 measures the top 0.02 per cent of a board's pairs at about 0.985
    precision -- roughly 127 directed edges -- and the doc's own rule is that
    such a tail is a CONSTRAINT and not a seed to grow from. Greedy growth has
    to grow from it and welds; the decoder does not, because every round is
    re-projected onto the doubly stochastic set, so a wrong one of these has to
    defeat every other claim on that tile instead of silently fusing two
    islands at a false offset.

    Injection is by row: for a kept edge the source tile's message row becomes a
    one-hot on its chosen partner, so the evidence enters at full strength while
    every other tile stays soft.
    """
    flat = np.concatenate([S[0].reshape(-1), S[1].reshape(-1)])
    if keep <= 0 or keep >= len(flat):
        return W
    thr = np.partition(flat, -keep)[-keep]
    out = []
    for ax in range(2):
        w = W[ax].clone()
        rows, cols = np.nonzero(S[ax] >= thr)
        w[rows] = 0.0
        w[rows, cols] = 1.0
        out.append(w)
    return out


def decode_board(Wr, Wd, rounds, beta, damp, restarts, gen, dev, sink=20):
    """Sinkhorn message passing from several noisy starts; best layout wins."""
    Wr, Wd = Wr.to(dev), Wd.to(dev)
    idx = torch.arange(N, device=dev).reshape(G, G)
    sr, dr = idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)
    sd, dd = idx[:-1].reshape(-1), idx[1:].reshape(-1)

    def sinkhorn(z, iters=sink):
        for _ in range(iters):
            z = z - torch.logsumexp(z, 1, keepdim=True)
            z = z - torch.logsumexp(z, 0, keepdim=True)
        return z

    outs = []
    for _ in range(restarts):
        lp = sinkhorn(torch.randn(N, N, device=dev, generator=gen))
        logit = torch.zeros(N, N, device=dev)
        for _ in range(rounds):
            p = lp.exp()
            # beliefs travel forwards and backwards along both axes; each is a
            # distribution over tiles, so they combine as a product, which is a
            # sum of logs
            new = torch.zeros(N, N, device=dev)
            new[dr] += torch.log(p[sr] @ Wr + EPS)
            new[sr] += torch.log(p[dr] @ Wr.T + EPS)
            new[dd] += torch.log(p[sd] @ Wd + EPS)
            new[sd] += torch.log(p[dd] @ Wd.T + EPS)
            logit = (1 - damp) * logit + damp * new
            lp = sinkhorn(beta * logit)
        outs.append(lp)
    return outs


def hungarian(lp):
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-lp.double().cpu().numpy())
    o = np.empty(N, np.int64)
    o[r] = c
    return o


def energy(order, Sr, Sd):
    g = np.asarray(order).reshape(G, G)
    return float(Sr[g[:, :-1], g[:, 1:]].sum() + Sd[g[:-1], g[1:]].sum())


def best_torus_shift(order):
    """Placement once the global ORIGIN is allowed to be wrong.

    A layout can hold many correct adjacencies while every tile sits in the
    wrong absolute cell, because shifting the whole board on a torus preserves
    every bond except the ones that wrap. Adjacency near 0.21 at placement 0.000
    is that signature, so this separates the two failures: how much of the board
    is assembled, and whether its origin is known.
    """
    g = np.asarray(order).reshape(G, G)
    truth = np.arange(N).reshape(G, G)
    best, arg = 0.0, (0, 0)
    for dy in range(G):
        for dx in range(G):
            v = float((np.roll(g, (-dy, -dx), (0, 1)) == truth).mean())
            if v > best:
                best, arg = v, (dy, dx)
    return best, arg


def adjacency(order):
    g = np.asarray(order).reshape(G, G)
    ok = int(((g[:, 1:] - g[:, :-1]) == 1).sum())
    ok += int(((g[1:] - g[:-1]) == G).sum())
    return ok / (2 * G * (G - 1))


def bond_precision(lp, order, Sr, Sd, volumes):
    """Precision of the decoder's own bonds, ranked by ITS confidence.

    M456 makes precision at volume the only quantity that converts: the block
    holds 350 fragments at edge precision 1.00 and 18 at the 0.746 we harvest,
    and the seam ranking gives 0.944 at 150 edges and 0.673 at 430. A seam score
    judges a pair alone. The decoder's marginal judges it against every other
    claim on both tiles at once, which is different evidence about the same
    edge, so it is worth asking whether it ranks better.
    """
    p = lp.exp().cpu().numpy()
    g = np.asarray(order).reshape(G, G)
    conf = p[np.arange(N), np.asarray(order)].reshape(G, G)
    bonds = []
    for (dy, dx), off in (((0, 1), 1), ((1, 0), G)):
        a = g[:G - dy, :G - dx]
        b = g[dy:, dx:]
        c = np.minimum(conf[:G - dy, :G - dx], conf[dy:, dx:])
        for u, v, w in zip(a.reshape(-1), b.reshape(-1), c.reshape(-1)):
            bonds.append((float(w), int(v) - int(u) == off))
    bonds.sort(reverse=True)
    ok = np.cumsum([t for _w, t in bonds])
    return [float(ok[v - 1] / v) if v <= len(ok) else np.nan for v in volumes]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="verify_top5_v2")
    ap.add_argument("--boards", type=int, default=32)
    ap.add_argument("--start", type=int, default=6700, help="held-out region")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--damp", type=float, default=0.5)
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--dense-dir", default="",
                    help="directory of costs_quad/*.npz; when set "
                         "the decoder eats the pipeline's own "
                         "fused matrices instead of the top-5 cache")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--islands", type=int, default=0,
                    help="how many of the globally strongest "
                         "edges enter as near-certain evidence")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = Path(CACHE_DIR) / a.cache
    cache = {k: np.load(d / f"{k}.npy", mmap_mode="r")
             for k in ("idx", "val")}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    rows = {b: [] for b in a.betas}
    prec = {}
    for k, b in enumerate(range(a.start, a.start + a.boards)):
        if a.dense_dir:
            f = Path(a.dense_dir) / f"{k:03d}.npz"
            if not f.exists():
                break
            (Wr, Wd), (Sr, Sd) = load_dense(f, a.topk)
        else:
            Wr, Wd = load_board(cache, b)
            Sr, Sd = [m.numpy() for m in score_matrices(cache, b)]
        if a.islands:
            Wr, Wd = inject_islands([Wr, Wd], [Sr, Sd], a.islands)
        for beta in a.betas:
            outs = decode_board(Wr, Wd, a.rounds, beta, a.damp, a.restarts,
                                gen, dev)
            best = None
            for lp in outs:
                o = hungarian(lp)
                e = energy(o, Sr, Sd)
                if best is None or e > best[0]:
                    best = (e, o, lp)
            o = best[1]
            shifted, _sh = best_torus_shift(o)
            prec.setdefault(beta, []).append(
                bond_precision(best[2], o, Sr, Sd, VOLUMES))
            rows[beta].append((float((o == np.arange(N)).mean()),
                               adjacency(o), shifted))
        print(f"board {b} done", flush=True)

    print(f"\n{a.boards} held-out boards, {a.restarts} restarts, {a.rounds} "
          f"rounds, damping {a.damp}. Real seam evidence, no training")
    print(f"{'beta':>8} {'placed':>9} {'adjacency':>11} "
          f"{'placed, best origin':>21}")
    for beta in a.betas:
        m = np.mean(rows[beta], axis=0)
        print(f"{beta:8.2f} {m[0]:9.4f} {m[1]:11.4f} {m[2]:21.4f}")
    print(f"\nchance {1/N:.4f}; shipping pipeline 0.0251 placement, "
          f"0.2890 adjacency")
    print("")
    print("PRECISION OF THE DECODER OWN BONDS, ranked by its confidence")
    print(f"{chr(98)+chr(101)+chr(116)+chr(97):>8}"
          + "".join(f"{v:>9}" for v in VOLUMES))
    for beta in a.betas:
        m = np.nanmean(prec[beta], axis=0)
        print(f"{beta:8.2f}" + "".join(f"{x:9.4f}" for x in m))
    print("the seam ranking reads 0.944 / 0.852 / 0.673 / 0.552 at these "
          "volumes; the target is 430 edges a board at 0.98")


if __name__ == "__main__":
    main()
