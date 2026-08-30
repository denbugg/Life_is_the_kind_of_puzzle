"""Cache the bag and the picture for the field diffusion, once.

Training reads 7000 boards many times and each board is two 480x480 images on a
slow disk, so the whole corpus is reduced to what the model actually consumes:

    bag8    every fragment pooled to 8x8, which is the encoder's view
    bag4    every fragment as exact 4x4 block means, which is what the Hungarian
            assignment compares -- 20 divides by 4 and not by 8, so this cannot
            be derived from bag8 and has to be stored
    stats   each fragment's per-channel mean and spread at full resolution
    picture the CLEAN target at 96x96, which is 4x4 a cell, the target of M428

About a gigabyte in total. The fragments are stored in TRUE CELL ORDER because
that is how the caches already hold them; nothing downstream may rely on it, and
the encoder is permutation invariant by construction so it cannot.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from config import CACHE_DIR, GRID as G, TRAIN_INP, TRAIN_TGT

N, S, SUB = G * G, 20, 4


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


def to_frags(img):
    a = img.reshape(G, S, G, S, 3).transpose(0, 2, 1, 3, 4)
    return a.reshape(N, S, S, 3)


def block_mean(frags, d):
    a = frags.astype(np.float32).reshape(len(frags), d, S // d, d, S // d, 3)
    return a.mean((2, 4))


def pool8(frags):
    """8x8 from 20x20 by uneven bins, matching adaptive average pooling."""
    e = np.linspace(0, S, 9).round().astype(int)
    out = np.zeros((len(frags), 8, 8, 3), np.float32)
    f = frags.astype(np.float32)
    for i in range(8):
        for j in range(8):
            out[:, i, j] = f[:, e[i]:e[i + 1], e[j]:e[j + 1]].mean((1, 2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boards", type=int, default=7000)
    ap.add_argument("--out", default="field_cache.npz")
    a = ap.parse_args()

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = [str(x) for x in blob["names"]]
    inv = blob["inv"]
    n = min(a.boards, len(names))

    bag8 = np.zeros((n, N, 8, 8, 3), np.uint8)
    bag4 = np.zeros((n, N, SUB, SUB, 3), np.uint8)
    stats = np.zeros((n, N, 6), np.float16)
    pic = np.zeros((n, G * SUB, G * SUB, 3), np.uint8)

    for b in range(n):
        frags = to_frags(load_rgb(Path(TRAIN_INP) / names[b]))[
            inv[b].astype(np.int64)]
        bag8[b] = np.round(pool8(frags)).astype(np.uint8)
        bag4[b] = np.round(block_mean(frags, SUB)).astype(np.uint8)
        f = frags.astype(np.float32)
        stats[b] = np.concatenate([f.mean((1, 2)), f.std((1, 2))], 1)
        clean = to_frags(load_rgb(Path(TRAIN_TGT) / names[b]))
        m = np.round(block_mean(clean, SUB)).astype(np.uint8)
        pic[b] = m.reshape(G, G, SUB, SUB, 3).transpose(
            0, 2, 1, 3, 4).reshape(G * SUB, G * SUB, 3)
        if b % 200 == 0:
            print(f"{b}/{n}", flush=True)

    out = Path(CACHE_DIR) / a.out
    np.savez(out, bag8=bag8, bag4=bag4, stats=stats, pic=pic,
             names=np.array(names[:n]))
    print(f"wrote {out} -- {out.stat().st_size / 2**30:.2f} GiB, {n} boards")
    print(f"picture spread {pic.reshape(n, -1).astype(np.float32).std():.2f}, "
          f"per board {pic.reshape(n, -1).astype(np.float32).std(1).mean():.2f}")


if __name__ == "__main__":
    main()
