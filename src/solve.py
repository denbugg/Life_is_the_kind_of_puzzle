"""Puzzle solver: from compatibility matrices R (right) and D (down), find the
fragment->grid-position assignment maximizing total edge compatibility.
Greedy corner-seeded init + numba simulated annealing over swaps.
"""
import numpy as np
import torch
from numba import njit
from config import GRID, NFRAG, FS


# --------------------- compatibility scoring (GPU) ---------------------
@torch.no_grad()
def compat_scores(model, frags_np, device="cuda", bs=2048):
    """frags_np: (N,20,20,3) uint8. Returns R,D (N,N) float32 (scaled logits)."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(frags_np)).permute(0, 3, 1, 2).float().div_(255)
    eRs, eLs, eTs, eBs = [], [], [], []
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            eR, eL, eT, eB = model.embed(xb)
        eRs.append(eR.float()); eLs.append(eL.float())
        eTs.append(eT.float()); eBs.append(eB.float())
    eR = torch.cat(eRs); eL = torch.cat(eLs); eT = torch.cat(eTs); eB = torch.cat(eBs)
    scale = model.logit_scale.exp().clamp(max=100).float()
    R = (scale * eR @ eL.t()).cpu().numpy()
    D = (scale * eB @ eT.t()).cpu().numpy()
    return R, D


# --------------------- numba SA solver ---------------------
@njit(cache=True, fastmath=True)
def _obj(place, R, D, G):
    N = place.shape[0]; s = 0.0
    for p in range(N):
        c = p % G; r = p // G
        if c < G - 1:
            s += R[place[p], place[p + 1]]
        if r < G - 1:
            s += D[place[p], place[p + G]]
    return s


@njit(cache=True, fastmath=True)
def _local(place, p, R, D, G):
    c = p % G; r = p // G; s = 0.0
    if c > 0:     s += R[place[p - 1], place[p]]
    if c < G - 1: s += R[place[p], place[p + 1]]
    if r > 0:     s += D[place[p - G], place[p]]
    if r < G - 1: s += D[place[p], place[p + G]]
    return s


@njit(cache=True, fastmath=True)
def _shared(place, p, q, R, D, G):
    # edge value between positions p,q if grid-adjacent (current place), else 0
    if q == p + 1 and p % G < G - 1:   return R[place[p], place[q]]
    if q == p - 1 and q % G < G - 1:   return R[place[q], place[p]]
    if q == p + G:                     return D[place[p], place[q]]
    if q == p - G:                     return D[place[q], place[p]]
    return 0.0


@njit(cache=True, fastmath=True)
def _gain(place, p, q, R, D, G):
    before = _local(place, p, R, D, G) + _local(place, q, R, D, G) - _shared(place, p, q, R, D, G)
    t = place[p]; place[p] = place[q]; place[q] = t
    after = _local(place, p, R, D, G) + _local(place, q, R, D, G) - _shared(place, p, q, R, D, G)
    t = place[p]; place[p] = place[q]; place[q] = t
    return after - before


@njit(cache=True, fastmath=True)
def _greedy(R, D, G, seed):
    N = R.shape[0]
    place = -np.ones(N, np.int64)
    used = np.zeros(N, np.uint8)
    place[0] = seed; used[seed] = 1
    for p in range(1, N):
        c = p % G; r = p // G
        best = -1; bestv = -1e18
        for f in range(N):
            if used[f]:
                continue
            v = 0.0
            if c > 0:  v += R[place[p - 1], f]
            if r > 0:  v += D[place[p - G], f]
            if v > bestv:
                bestv = v; best = f
        place[p] = best; used[best] = 1
    return place


@njit(cache=True, fastmath=True)
def _anneal(place, R, D, G, iters, T0, T1, seed):
    np.random.seed(seed)
    N = place.shape[0]
    cur = _obj(place, R, D, G)
    best = place.copy(); bestv = cur
    ratio = T1 / T0
    for it in range(iters):
        p = np.random.randint(N); q = np.random.randint(N)
        if p == q:
            continue
        g = _gain(place, p, q, R, D, G)
        T = T0 * ratio ** (it / iters)
        if g > 0 or np.random.random() < np.exp(g / T):
            t = place[p]; place[p] = place[q]; place[q] = t
            cur += g
            if cur > bestv:
                bestv = cur
                best = place.copy()
    return best, bestv


def solve_from_scores(R, D, iters=3_000_000, restarts=3, T_scale=1.0, seed0=0):
    R = np.ascontiguousarray(R, np.float32)
    D = np.ascontiguousarray(D, np.float32)
    G = GRID
    std = float(np.std(R) + np.std(D)) * 0.5
    T0 = max(1e-3, std * T_scale); T1 = T0 * 0.01
    # corner-ish seed: low chance of having a left/top neighbor
    corner = np.argmin(R.max(0) + D.max(0))
    best = None; bestv = -1e18
    for rs in range(restarts):
        s = int(corner) if rs == 0 else int(np.random.randint(NFRAG))
        place = _greedy(R, D, G, s)
        place, v = _anneal(place, R, D, G, iters, T0, T1, seed0 + rs + 1)
        if v > bestv:
            bestv = v; best = place
    return best, bestv


@torch.no_grad()
def rescore_pairwise(pair_model, frags_np, R, D, K=32, alpha=3.0, device="cuda", bs=16384):
    """Blend standardized siamese scores with pairwise logits on top-K candidates.
    Keeps a smooth objective (non-topK keep siamese order) while topK get the
    more accurate pairwise signal."""
    pair_model.eval()
    x = torch.from_numpy(np.ascontiguousarray(frags_np)).permute(0, 3, 1, 2).float().div_(255).to(device)
    xt = x.transpose(-1, -2)
    N = x.shape[0]

    def zc(M):
        return (M - M.mean()) / (M.std() + 1e-6)

    def do(M, feat):
        base = zc(M).copy()
        Mm = M.copy(); np.fill_diagonal(Mm, -1e30)
        topk = np.argpartition(-Mm, K, axis=1)[:, :K]           # (N,K)
        anchors = np.repeat(np.arange(N), K)
        cands = topk.reshape(-1)
        logits = np.empty(len(anchors), np.float32)
        for s in range(0, len(anchors), bs):
            a = torch.as_tensor(anchors[s:s + bs], device=device)
            c = torch.as_tensor(cands[s:s + bs], device=device)
            pair = torch.cat([feat[a], feat[c]], dim=-1)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = pair_model(pair).float()
            logits[s:s + bs] = lg.cpu().numpy()
        lz = (logits - logits.mean()) / (logits.std() + 1e-6)
        for i in range(N):
            base[i, topk[i]] += alpha * lz[i * K:(i + 1) * K]
        return base.astype(np.float32)

    return do(R, x), do(D, xt)


@torch.no_grad()
def pairwise_scores_full(pair_model, frags_np, device="cuda", bs=16384):
    """Full NxN pairwise compatibility (bypasses any siamese pre-filter).
    pair_model may be a single net or a LIST of nets, whose logits are averaged
    (ensemble of independently-trained scorers → sharper compatibility)."""
    models = pair_model if isinstance(pair_model, (list, tuple)) else [pair_model]
    for m in models:
        m.eval()
    x = torch.from_numpy(np.ascontiguousarray(frags_np)).permute(0, 3, 1, 2).float().div_(255).to(device)
    xt = x.transpose(-1, -2)
    N = x.shape[0]
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    ii = ii.reshape(-1); jj = jj.reshape(-1)

    def do(feat):
        out = np.empty(N * N, np.float32)
        for s in range(0, N * N, bs):
            a = torch.as_tensor(ii[s:s + bs], device=device)
            b = torch.as_tensor(jj[s:s + bs], device=device)
            pair = torch.cat([feat[a], feat[b]], dim=-1)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = sum(m(pair).float() for m in models) / len(models)
            out[s:s + bs] = lg.cpu().numpy()
        return out.reshape(N, N)

    return do(x), do(xt)


def solve_image(frags_np, model, device="cuda", pair_model=None, rescore_kw=None,
                full_pair=False, **kw):
    if full_pair and pair_model is not None:
        R, D = pairwise_scores_full(pair_model, frags_np, device)
    else:
        R, D = compat_scores(model, frags_np, device)
        if pair_model is not None:
            R, D = rescore_pairwise(pair_model, frags_np, R, D, **(rescore_kw or {}))
    place, v = solve_from_scores(R, D, **kw)
    return place, R, D, v
    R, D = compat_scores(model, frags_np, device)
    if pair_model is not None:
        R, D = rescore_pairwise(pair_model, frags_np, R, D, **(rescore_kw or {}))
    place, v = solve_from_scores(R, D, **kw)
    return place, R, D, v
