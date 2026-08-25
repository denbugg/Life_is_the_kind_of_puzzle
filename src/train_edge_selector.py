"""Train the edge selector that replaces the vote threshold.

The vote threshold answers "how many scorers called this pair mutually best",
which is the only question the mutual-best pool can be asked -- every candidate
in it is rank one from both ends. M317 built the best selector this project has
had on exactly that pool and closed the route: precision 0.951 at 100 edges
falling smoothly to 0.586 at 430, against M316's requirement of 430 at 0.97,
"so there is no threshold hiding a clean set".

M377 moved the ground under that conclusion. Widening each fragment's candidate
list lifts true-edge recall from 0.368 to 0.516 at depth two and 0.645 at depth
eight, so M268's cliff of 552 true edges is now inside the evidence; and the
edges widening adds are the ones mutual best discarded, where one side preferred
somebody else. Their rank on each side, the score margin at that rank, and the
disagreement between the two ends are all features that do not exist until the
pool is widened.

Input is a directory of per-scorer rank dumps (scratchpad/dump_ranks.py). The
model is scored the way it will be used: greedily and exclusively decoded to a
fixed volume, then built into components, because M317 reported precision and
watched placement collapse anyway.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

from config import CKPT_DIR, GRID as G
from edge_selector import FEATURES, _features
from solve_buddies import build_directed_components

N = G * G
DIRS = {(0, 1): 3, (1, 0): 1}


def board_rows(path, depth):
    z = np.load(path)
    tags = [t.split("|") for t in z["tags"]]
    keys, X, y = [], [], []
    for dn, off in (("h", (0, 1)), ("v", (1, 0))):
        per = []
        for s in range(len(tags)):
            mi, vi, _oi = (int(x) for x in tags[s])
            per.append((z[f"{s}_{dn}_fi"].astype(np.int64), z[f"{s}_{dn}_fv"],
                        z[f"{s}_{dn}_bi"].astype(np.int64), z[f"{s}_{dn}_bv"],
                        vi, mi))
        src, dst, x = _features(per, depth)
        keys += [(int(a), int(b), off) for a, b in zip(src, dst)]
        X.append(x)
        y.append(((dst - src == 1) & (src % G != G - 1)) if off == (0, 1)
                 else (dst - src == G))
    return keys, np.vstack(X), np.concatenate(y).astype(np.int8)


def decode(keys, scores, volume):
    """One right-hand and one left-hand partner per fragment, best score first."""
    order = np.argsort(-scores)
    used_src, used_dst, out = set(), set(), []
    for idx in order:
        i, j, off = keys[idx]
        if i == j or (i, off) in used_src or (j, off) in used_dst:
            continue
        used_src.add((i, off))
        used_dst.add((j, off))
        out.append((keys[idx], float(scores[idx])))
        if len(out) >= volume:
            break
    return out


def coherent_block(comps):
    best = 1
    for c in comps:
        sh = defaultdict(int)
        for f, (y, x) in c.items():
            sh[(y - int(f) // G, x - int(f) % G)] += 1
        if sh:
            best = max(best, max(sh.values()))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", required=True,
                    help="directory of per-scorer rank dumps")
    ap.add_argument("--depth", type=int, default=2,
                    help="candidates per fragment per scorer the model sees")
    ap.add_argument("--held", type=int, default=20, help="boards held out")
    ap.add_argument("--held-front", action="store_true",
                    help="hold out the FIRST boards instead of the last, so a "
                         "held-out set can be chosen to match dumps another "
                         "experiment already has")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--volumes", type=int, nargs="+",
                    default=[200, 300, 430, 552])
    ap.add_argument("--objective", default="binary",
                    choices=("binary", "lambdarank"),
                    help="binary treats every candidate independently; "
                         "lambdarank ranks the candidates of one SOURCE "
                         "fragment against each other, which is the question "
                         "exclusive decoding actually asks -- each fragment "
                         "keeps one right-hand and one left-hand partner, so "
                         "what matters is the order within a fragment and not "
                         "the calibration across the board")
    ap.add_argument("--out", default="edge_selector.txt")
    a = ap.parse_args()

    files = sorted(Path(a.dumps).glob("*.npz"))
    if len(files) <= a.held:
        sys.exit(f"only {len(files)} dumps in {a.dumps}")
    train, held = ((files[a.held:], files[:a.held]) if a.held_front
                   else (files[:-a.held], files[-a.held:]))
    print(f"{len(train)} train boards, {len(held)} held out, depth {a.depth}",
          flush=True)

    X, Y, GRP = [], [], []
    for bi, f in enumerate(train):
        k, x, y = board_rows(f, a.depth)
        # one group per (board, source fragment, direction): the candidates
        # that compete for the same slot under exclusive decoding
        g = np.array([hash((bi, i, off)) for i, _j, off in k])
        o = np.argsort(g, kind="stable")
        X.append(x[o])
        Y.append(y[o])
        GRP.append(np.unique(g[o], return_counts=True)[1])
    X, Y = np.vstack(X), np.concatenate(Y)
    GRP = np.concatenate(GRP)
    pos = float(Y.mean())
    print(f"{len(Y)/len(train):.0f} candidates a board, {Y.sum()/len(train):.0f} "
          f"true, base rate {pos:.4f}", flush=True)

    params = {"learning_rate": 0.05, "num_leaves": a.leaves,
              "min_data_in_leaf": 50, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}
    if a.objective == "binary":
        params["objective"] = "binary"
        params["scale_pos_weight"] = ((1 - pos) / max(pos, 1e-9)) ** 0.5
        ds = lgb.Dataset(X, Y, feature_name=FEATURES)
    else:
        params["objective"] = "lambdarank"
        params["lambdarank_truncation_level"] = 8
        ds = lgb.Dataset(X, Y, group=GRP, feature_name=FEATURES)
    model = lgb.train(params, ds, num_boost_round=a.rounds)

    prec, block, adj = defaultdict(list), defaultdict(list), defaultdict(list)
    for f in held:
        k, x, y = board_rows(f, a.depth)
        s = model.predict(x)
        for v in a.volumes:
            sel = decode(k, s, v)
            t = [int((j - i == 1 and i % G != G - 1) if off == (0, 1)
                     else j - i == G) for (i, j, off), _ in sel]
            prec[v].append(np.mean(t))
            comps = build_directed_components(
                [i for (i, _j, _o), _w in sel],
                [DIRS[o] for (_i, _j, o), _w in sel],
                [j for (_i, j, _o), _w in sel],
                [w for _e, w in sel], max_edges=len(sel))
            comps = [dict(c) for c in comps if c]
            block[v].append(coherent_block(comps))
            adj[v].append(sum(t))

    print(f"\n{'volume':>8} {'precision':>10} {'true edges':>11} "
          f"{'largest block':>14}   M317, mutual-best pool")
    ref = {200: 0.833, 300: 0.707, 430: 0.586}
    for v in a.volumes:
        print(f"{v:>8} {np.mean(prec[v]):10.3f} {np.mean(adj[v]):11.1f} "
              f"{np.mean(block[v]):14.1f}   {ref.get(v, '')}")
    print("\nthe shipping harvest reaches 254 true adjacencies and a coherent "
          "block of 33.7 on the same measure")
    imp = sorted(zip(FEATURES, model.feature_importance("gain")),
                 key=lambda kv: -kv[1])[:8]
    print("top features: " + ", ".join(f"{n} {g:.0f}" for n, g in imp))

    out = Path(CKPT_DIR) / a.out
    model.save_model(str(out))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
