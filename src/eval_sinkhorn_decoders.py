"""Measure whether discrete QAP refinement converts learned edge evidence.

This is deliberately a decoder-only experiment: the checkpoint and held-out
crop bags are frozen.  It compares the recurrent Hungarian layout with global
edge-objective refinement, so a gain cannot be attributed to more training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import CACHE_DIR
from discrete_field_decoder import solve_discrete
from sinkhorn_assembler import SinkhornAssembler, decode
from solve_anneal import solve_anneal, total_cost
from train_sinkhorn_assembler import crop_batch, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--boards", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=9876)
    ap.add_argument("--unary-weights", default="0.03,0.1,0.3,1.0")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ma = ck["args"]
    side = int(ma["side"])
    model = SinkhornAssembler(int(ma["d"]), int(ma["rounds"]),
                              int(ma["blocks"])).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    bag = np.load(Path(CACHE_DIR) / "field_cache.npz")["bag8"]
    ids = np.arange(len(bag) - 300, len(bag))[:a.boards]
    x, target = crop_batch(bag, ids, side, np.random.default_rng(a.seed))
    weights = [float(v) for v in a.unary_weights.split(",")]
    rows = []
    for begin in range(0, len(x), a.batch):
        xt = torch.from_numpy(x[begin:begin + a.batch]).to(dev)
        with torch.no_grad():
            logits, _, edges = model(xt, side)
        base = decode(logits)
        e = edges.float().cpu().numpy()
        # Log-probabilities remove arbitrary learned temperature/offsets.
        ep = F.log_softmax(edges.float(), dim=-1).cpu().numpy()
        up = F.log_softmax(logits.float(), dim=-1).cpu().numpy()
        for j in range(len(xt)):
            truth = target[begin + j]
            p0, a0 = metrics(base[j:j + 1], truth[None], side)
            ch, cv = -e[j, 0], -e[j, 1]
            true_cost = float(total_cost(truth, ch, cv, side, side * side))
            init_cost = float(total_cost(base[j], ch, cv, side, side * side))
            lay, cost = solve_anneal(
                ch, cv, grid=side, iters=a.iters, restarts=a.restarts,
                seed=a.seed + begin + j, init=base[j], sweeps=8)
            p1, a1 = metrics(lay[None], truth[None], side)
            row = [p0, a0, p1, a1, init_cost / true_cost,
                   float(cost) / true_cost]
            for weight in weights:
                dl, _ = solve_discrete(ep[j, 0], ep[j, 1], up[j], base[j], side,
                                       weight, a.iters, a.restarts,
                                       a.seed + begin + j)
                dp, da = metrics(dl[None], truth[None], side)
                row.extend((dp, da))
            rows.append(row)
        print(f"{min(begin + a.batch, len(x))}/{len(x)}", flush=True)

    z = np.asarray(rows)
    labels = ("base_place", "base_adj", "qap_place", "qap_adj",
              "base_cost/true", "qap_cost/true")
    for name, value in zip(labels, z.mean(0)):
        print(f"{name:18s} {value:.6f}")
    d = z[:, 2] - z[:, 0]
    print(f"placement delta    {d.mean():.6f}  wins/ties/losses "
          f"{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}")
    for wi, weight in enumerate(weights):
        pcol, acol = 6 + 2 * wi, 7 + 2 * wi
        delta = z[:, pcol] - z[:, 0]
        print(f"unary {weight:<6g} place {z[:, pcol].mean():.6f}  "
              f"adj {z[:, acol].mean():.6f}  delta {delta.mean():+.6f}  "
              f"W/T/L {(delta > 0).sum()}/{(delta == 0).sum()}/{(delta < 0).sum()}")


if __name__ == "__main__":
    main()
