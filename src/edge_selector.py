"""Score candidate edges with a trained selector instead of a vote threshold.

The vote threshold is a blunt instrument on a wide pool. M377 measured that
widening each fragment's candidate list from mutual-best to the top k lifts
true-edge recall from 0.368 to 0.516 at depth two and 0.645 at depth eight, so
M268's cliff of 552 true edges is inside the evidence -- and that ranking the
wider pool by vote count goes the WRONG way, 253 true in the best 432 at depth
one against 157 at depth two. Volume without a ranking is worthless.

M317 closed per-edge selection, but on the mutual-best pool, where the most
informative property of a candidate is constant: every edge in it is rank one
from both ends. Widening makes RANK a variable, and adds evidence that cannot
exist at depth one -- the two ends rank each other independently, so their
DISAGREEMENT is a feature no single side carries.

The features are built vectorised here because the pipeline pays for them on
every board; scratchpad/rank_features.py holds the reference implementation the
training set was built from, and `_features` reproduces it exactly.
"""
import numpy as np

from config import GRID as G
from harvest_votes import ORIENTATIONS, _calibrated

N = G * G
K = 8
FEATURES = [
    "n_seen", "n_fwd1", "n_bwd1", "n_mutual1", "n_fwd2", "n_bwd2",
    "fwd_rank_mean", "fwd_rank_min", "bwd_rank_mean", "bwd_rank_min",
    "rank_sum", "rank_asym",
    "fwd_margin_mean", "fwd_margin_max", "bwd_margin_mean", "bwd_margin_max",
    "fwd_gap1_mean", "bwd_gap1_mean",
    "src_rivals", "dst_rivals", "src_share", "dst_share", "src_gap", "dst_gap",
    "n_views", "n_archs", "view_spread", "arch_spread",
]


def _topk(M):
    """Top-K partners of every row and of every column, sorted, with scores."""
    D = np.array(M, dtype=np.float32)
    np.fill_diagonal(D, -np.inf)
    fi = np.argpartition(-D, K, axis=1)[:, :K]
    fv = np.take_along_axis(D, fi, axis=1)
    o = np.argsort(-fv, axis=1)
    fi, fv = np.take_along_axis(fi, o, 1), np.take_along_axis(fv, o, 1)
    bi = np.argpartition(-D, K, axis=0)[:K].T
    bv = np.take_along_axis(D.T, bi, axis=1)
    o = np.argsort(-bv, axis=1)
    bi, bv = np.take_along_axis(bi, o, 1), np.take_along_axis(bv, o, 1)
    return fi, fv, bi, bv


def _acc(keys, inv, n, vals=None, how="sum"):
    if how == "count":
        return np.bincount(inv, minlength=n).astype(np.float32)
    if how == "sum":
        return np.bincount(inv, weights=vals, minlength=n).astype(np.float32)
    out = np.full(n, np.inf if how == "min" else -np.inf, np.float32)
    np.minimum.at(out, inv, vals) if how == "min" else np.maximum.at(out, inv, vals)
    return out


def _features(per_scorer, depth):
    """The 28 features of every candidate, for one direction.

    `per_scorer` is a list of (fi, fv, bi, bv, view_index, arch_index).
    """
    kf, rf, mf, gf, vf, af = [], [], [], [], [], []
    kb, rb, mb, gb, vb, ab = [], [], [], [], [], []
    rows = np.arange(N)
    for fi, fv, bi, bv, vi, ai in per_scorer:
        d = min(depth, K)
        kf.append(rows[:, None] * N + fi[:, :d])
        rf.append(np.broadcast_to(np.arange(d), (N, d)))
        mf.append(fv[:, :d] - fv[:, :1])
        gf.append(np.broadcast_to((fv[:, 0] - fv[:, 1])[:, None], (N, d)))
        vf.append(np.full((N, d), 1 << vi, np.int32))
        af.append(np.full((N, d), 1 << ai, np.int32))
        kb.append(bi[:, :d] * N + rows[:, None])
        rb.append(np.broadcast_to(np.arange(d), (N, d)))
        mb.append(bv[:, :d] - bv[:, :1])
        gb.append(np.broadcast_to((bv[:, 0] - bv[:, 1])[:, None], (N, d)))
        vb.append(np.full((N, d), 1 << vi, np.int32))
        ab.append(np.full((N, d), 1 << ai, np.int32))

    def cat(a):
        return np.concatenate([x.ravel() for x in a])

    kf, rf, mf, gf, vf, af = map(cat, (kf, rf, mf, gf, vf, af))
    kb, rb, mb, gb, vb, ab = map(cat, (kb, rb, mb, gb, vb, ab))
    uniq, inv = np.unique(np.concatenate([kf, kb]), return_inverse=True)
    n = len(uniq)
    inv_f, inv_b = inv[:len(kf)], inv[len(kf):]

    cnt_f = _acc(kf, inv_f, n, how="count")
    cnt_b = _acc(kb, inv_b, n, how="count")
    seen = cnt_f + cnt_b
    fwd1 = _acc(kf, inv_f, n, (rf == 0).astype(np.float32))
    bwd1 = _acc(kb, inv_b, n, (rb == 0).astype(np.float32))
    fwd2 = _acc(kf, inv_f, n, (rf < 2).astype(np.float32))
    bwd2 = _acc(kb, inv_b, n, (rb < 2).astype(np.float32))
    safe_f = np.maximum(cnt_f, 1)
    safe_b = np.maximum(cnt_b, 1)
    fr_mean = np.where(cnt_f > 0,
                       _acc(kf, inv_f, n, rf.astype(np.float32)) / safe_f, K)
    br_mean = np.where(cnt_b > 0,
                       _acc(kb, inv_b, n, rb.astype(np.float32)) / safe_b, K)
    fr_min = _acc(kf, inv_f, n, rf.astype(np.float32), "min")
    br_min = _acc(kb, inv_b, n, rb.astype(np.float32), "min")
    fr_min[~np.isfinite(fr_min)] = K
    br_min[~np.isfinite(br_min)] = K
    fm_mean = np.where(cnt_f > 0, _acc(kf, inv_f, n, mf) / safe_f, -9.0)
    bm_mean = np.where(cnt_b > 0, _acc(kb, inv_b, n, mb) / safe_b, -9.0)
    fm_max = _acc(kf, inv_f, n, mf, "max")
    bm_max = _acc(kb, inv_b, n, mb, "max")
    fm_max[~np.isfinite(fm_max)] = -9.0
    bm_max[~np.isfinite(bm_max)] = -9.0
    fg = np.where(cnt_f > 0, _acc(kf, inv_f, n, gf) / safe_f, 0.0)
    bg = np.where(cnt_b > 0, _acc(kb, inv_b, n, gb) / safe_b, 0.0)

    vmask = np.zeros(n, np.int32)
    amask = np.zeros(n, np.int32)
    np.bitwise_or.at(vmask, inv_f, vf)
    np.bitwise_or.at(vmask, inv_b, vb)
    np.bitwise_or.at(amask, inv_f, af)
    np.bitwise_or.at(amask, inv_b, ab)
    n_views = np.array([bin(int(x)).count("1") for x in vmask], np.float32)
    n_archs = np.array([bin(int(x)).count("1") for x in amask], np.float32)

    src, dst = uniq // N, uniq % N
    src_tot = np.bincount(src, weights=seen, minlength=N)[src]
    dst_tot = np.bincount(dst, weights=seen, minlength=N)[dst]
    src_cnt = np.bincount(src, minlength=N)[src].astype(np.float32)
    dst_cnt = np.bincount(dst, minlength=N)[dst].astype(np.float32)
    src_best = np.zeros(N, np.float32)
    dst_best = np.zeros(N, np.float32)
    np.maximum.at(src_best, src, seen)
    np.maximum.at(dst_best, dst, seen)
    src_share = seen / np.maximum(src_tot, 1)
    dst_share = seen / np.maximum(dst_tot, 1)

    X = np.stack([
        seen, fwd1, bwd1, np.minimum(fwd1, bwd1), fwd2, bwd2,
        fr_mean, fr_min, br_mean, br_min, fr_mean + br_mean,
        np.abs(fr_mean - br_mean),
        fm_mean, fm_max, bm_mean, bm_max, fg, bg,
        src_cnt, dst_cnt, src_share, dst_share,
        seen - src_best[src], seen - dst_best[dst],
        n_views, n_archs, n_views * src_share, n_archs * src_share,
    ], axis=1).astype(np.float32)
    return src, dst, X


def selected_edges(booster, matchers, views, device, orientations=2, depth=2,
                   volume=430):
    """The `volume` best candidates the selector can find, as {edge: score}.

    The score is returned as the ORDERING weight, which is all
    `build_directed_components` uses it for, and decoding is exclusive: one
    right-hand and one left-hand partner per fragment, as the grid demands.
    """
    per = {(0, 1): [], (1, 0): []}
    for ai, m in enumerate(matchers):
        for vi, v in enumerate(views):
            for o in ORIENTATIONS[:orientations]:
                H, V = _calibrated(m, v, device, o)
                per[(0, 1)].append(_topk(H) + (vi, ai))
                per[(1, 0)].append(_topk(V) + (vi, ai))
    keys, scores = [], []
    for off, ps in per.items():
        src, dst, X = _features(ps, depth)
        s = booster.predict(X)
        keys += [(int(a), int(b), off) for a, b in zip(src, dst)]
        scores.append(s)
    scores = np.concatenate(scores)
    order = np.argsort(-scores)
    used_src, used_dst, out = set(), set(), {}
    for idx in order:
        i, j, off = keys[idx]
        if i == j or (i, off) in used_src or (j, off) in used_dst:
            continue
        used_src.add((i, off))
        used_dst.add((j, off))
        out[(i, j, off)] = float(scores[idx])
        if len(out) >= volume:
            break
    return out
