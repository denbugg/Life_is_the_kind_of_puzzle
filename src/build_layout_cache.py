"""Precompute the solver's layout for many train boards, once.

The restorer we are about to train has to see the input distribution it will
actually meet, and that input is OUR assembled board -- dirty tiles in the order
our solver puts them, not a correctly ordered corrupted image.  R5 was trained
on the latter and has never seen the former (M133), which is the whole reason
this exists.

Only the permutation is stored: 576 int16 per board, and the image is rebuilt
from the raw tiles at training time in under a millisecond.  Running the solver
costs about 2.9 s per board, almost all of it in the CPU-side loop construction,
so shards run in parallel processes with barely any GPU contention.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config import CACHE_DIR, CKPT_DIR, TRAIN_INP
from restore_tile import to_frags
from seam_cost import costs_from_model
from seam_embed import SeamEmbed
from solve_loop import solve as solve_loop
from solve_relax import solve_relax
from torus_origin import fix_origin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="seam_embed_v1.pt")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1, help="shard stride")
    ap.add_argument("--offset", type=int, default=0, help="shard offset")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda"
    ck = torch.load(Path(CKPT_DIR) / a.ckpt, map_location=dev, weights_only=False)
    ta = ck["args"]
    model = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                      ta.get("head", "global")).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names, inv_all = blob["names"], blob["inv"]
    idx = list(range(a.start, min(a.start + a.count, len(names))))[a.offset::a.stride]

    out_names, out_lay = [], []
    t0 = time.time()
    for n_done, k in enumerate(idx):
        nm = str(names[k])
        img = cv2.imread(str(Path(TRAIN_INP) / nm), cv2.IMREAD_COLOR)
        if img is None:
            continue
        tiles = to_frags(np.ascontiguousarray(img[:, :, ::-1])).astype(np.float32)[
            inv_all[k].astype(np.int64)]
        CH, CV = costs_from_model(model, tiles)
        greedy = fix_origin(solve_loop(CH, CV)[0], tiles, metric="mgc")
        lay = solve_relax(CH, CV, rounds=200, init=greedy)
        out_names.append(nm)
        out_lay.append(np.asarray(lay, dtype=np.int16))
        if (n_done + 1) % 25 == 0:
            el = time.time() - t0
            print(f"{n_done+1}/{len(idx)}  {el:.0f}s  "
                  f"eta {el / (n_done + 1) * (len(idx) - n_done - 1):.0f}s", flush=True)
            np.savez_compressed(a.out, names=np.array(out_names),
                                lay=np.stack(out_lay))
    np.savez_compressed(a.out, names=np.array(out_names), lay=np.stack(out_lay))
    print(f"wrote {len(out_names)} layouts to {a.out} in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
