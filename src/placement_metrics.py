"""Placement and adjacency metrics for the 24x24 jigsaw.

`place[p]` and recovered `inv[p]` both mean: fragment id placed at grid position p.
Neighbour accuracy is directed and frame-invariant: a right/down edge is correct if
the two adjacent fragments are true right/down neighbours somewhere in the target.
"""
import numpy as np
from config import GRID, NFRAG


def true_edges(inv):
    inv = np.asarray(inv, dtype=np.int64)
    right, down = set(), set()
    for p in range(NFRAG):
        r, c = divmod(p, GRID)
        if c < GRID - 1:
            right.add((int(inv[p]), int(inv[p + 1])))
        if r < GRID - 1:
            down.add((int(inv[p]), int(inv[p + GRID])))
    return right, down


def pred_edges(place):
    place = np.asarray(place, dtype=np.int64)
    right, down = set(), set()
    for p in range(NFRAG):
        r, c = divmod(p, GRID)
        if c < GRID - 1:
            right.add((int(place[p]), int(place[p + 1])))
        if r < GRID - 1:
            down.add((int(place[p]), int(place[p + GRID])))
    return right, down


def neighbour_accuracy(place, inv):
    tr, td = true_edges(inv)
    pr, pd = pred_edges(place)
    nr = GRID * (GRID - 1)
    nd = (GRID - 1) * GRID
    ar = len(pr & tr) / nr
    ad = len(pd & td) / nd
    return float((len(pr & tr) + len(pd & td)) / (nr + nd)), float(ar), float(ad)


def placement_accuracy(place, inv, conf=None, hi=0.6):
    place = np.asarray(place, dtype=np.int64)
    inv = np.asarray(inv, dtype=np.int64)
    acc = float(np.mean(place == inv))
    if conf is None:
        return acc, None
    conf = np.asarray(conf)
    keep = conf > hi
    hi_acc = float(np.mean(place[keep] == inv[keep])) if keep.any() else 0.0
    return acc, hi_acc


def objective(place, R, D):
    place = np.asarray(place, dtype=np.int64)
    s = 0.0
    for p in range(NFRAG):
        r, c = divmod(p, GRID)
        a = place[p]
        if c < GRID - 1:
            s += float(R[a, place[p + 1]])
        if r < GRID - 1:
            s += float(D[a, place[p + GRID]])
    return s


def local_agreement(place, R, D):
    place = np.asarray(place, dtype=np.int64)
    out = np.zeros(NFRAG, np.float32)
    cnt = np.zeros(NFRAG, np.float32)
    for p in range(NFRAG):
        r, c = divmod(p, GRID)
        a = place[p]
        if c > 0:
            out[p] += R[place[p - 1], a]; cnt[p] += 1
        if c < GRID - 1:
            out[p] += R[a, place[p + 1]]; cnt[p] += 1
        if r > 0:
            out[p] += D[place[p - GRID], a]; cnt[p] += 1
        if r < GRID - 1:
            out[p] += D[a, place[p + GRID]]; cnt[p] += 1
    return out / np.maximum(cnt, 1)

