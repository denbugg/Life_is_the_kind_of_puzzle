"""Best-buddies / component-growth solver for PairwiseNet score matrices.

This solver is intentionally conservative: it uses only high-confidence mutual
edges to build components, places components on the 24x24 board, fills holes
greedily, then runs a small local swap repair. It is meant to replace SA only
after the scorer passes the bb_prec gate.
"""
import os
import argparse
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from config import GRID, NFRAG, TRAIN_INP, TRAIN_TGT, CACHE_DIR
from imgio import load, to_frags, assemble, train_val_split
from pipeline import load_pair
from solve import pairwise_scores_full
from placement_metrics import neighbour_accuracy, placement_accuracy, objective, local_agreement
from match_preprocess import load_match_denoiser, preprocess_frags_np

DEV = "cuda"


def _best_and_margin(M):
    X = M.copy()
    np.fill_diagonal(X, -1e30)
    best = X.argmax(1)
    bval = X[np.arange(NFRAG), best]
    X[np.arange(NFRAG), best] = -1e30
    second = X.max(1)
    return best, bval, bval - second


def _candidate_edges(R, D, max_edges=900, min_margin=0.0):
    out = []
    for M, dy, dx in (
        (R, 0, 1),
        (D, 1, 0),
    ):
        rb, rv, rm = _best_and_margin(M)
        cb = M.copy()
        np.fill_diagonal(cb, -1e30)
        col_best = cb.argmax(0)
        for a in range(NFRAG):
            b = int(rb[a])
            if col_best[b] != a:
                continue
            if rm[a] < min_margin:
                continue
            out.append((float(rv[a]), float(rm[a]), int(a), b, dy, dx))
    out.sort(reverse=True)
    return out[:max_edges]


class _Builder:
    def __init__(self, cap=0):
        self.frag_comp = {}
        self.comps = []
        # M444: an edge that would grow a component past `cap` is refused, so a
        # larger harvest makes MORE components rather than BIGGER ones. Without
        # it the extra edges WELD islands together and one wrong edge makes the
        # whole grown island internally wrong -- at 500 edges the uncapped seed
        # keeps 7 correct islands of 40 where a cap of two keeps 77, and the
        # largest truly-connected group goes 50.4 to 127.8.
        self.cap = int(cap)

    def _span_ok(self, comp):
        ys = [p[0] for p in comp.values()]
        xs = [p[1] for p in comp.values()]
        return max(ys) - min(ys) < GRID and max(xs) - min(xs) < GRID

    def _new_comp(self, a, b, off):
        cid = len(self.comps)
        self.comps.append({a: (0, 0), b: off})
        self.frag_comp[a] = cid
        self.frag_comp[b] = cid

    def _add_to_comp(self, cid, frag, coord):
        comp = self.comps[cid]
        if self.cap and len(comp) + 1 > self.cap:
            return False
        if coord in comp.values():
            return False
        comp[frag] = coord
        if not self._span_ok(comp):
            del comp[frag]
            return False
        self.frag_comp[frag] = cid
        return True

    def _merge(self, ca, cb, shift):
        if ca == cb:
            return True
        A, B = self.comps[ca], self.comps[cb]
        if self.cap and len(A) + len(B) > self.cap:
            return False
        moved = {f: (p[0] + shift[0], p[1] + shift[1]) for f, p in B.items()}
        occ = set(A.values())
        if any(p in occ for p in moved.values()):
            return False
        merged = dict(A)
        merged.update(moved)
        if not self._span_ok(merged):
            return False
        self.comps[ca] = merged
        self.comps[cb] = {}
        for f in moved:
            self.frag_comp[f] = ca
        return True

    def add_edge(self, a, b, dy, dx):
        off = (dy, dx)
        ca = self.frag_comp.get(a)
        cb = self.frag_comp.get(b)
        if ca is None and cb is None:
            self._new_comp(a, b, off)
            return True
        if ca is not None and cb is None:
            ay, ax = self.comps[ca][a]
            return self._add_to_comp(ca, b, (ay + dy, ax + dx))
        if ca is None and cb is not None:
            by, bx = self.comps[cb][b]
            return self._add_to_comp(cb, a, (by - dy, bx - dx))
        if ca == cb:
            ay, ax = self.comps[ca][a]
            by, bx = self.comps[ca][b]
            return (by - ay, bx - ax) == off
        ay, ax = self.comps[ca][a]
        by, bx = self.comps[cb][b]
        shift = (ay + dy - by, ax + dx - bx)
        return self._merge(ca, cb, shift)

    def components(self):
        return [c for c in self.comps if c]


def _shift_score(comp, board, R, D, sy, sx):
    score = 0.0
    for f, (y, x) in comp.items():
        r, c = y + sy, x + sx
        if c > 0 and board[r, c - 1] >= 0:
            score += R[board[r, c - 1], f]
        if c < GRID - 1 and board[r, c + 1] >= 0:
            score += R[f, board[r, c + 1]]
        if r > 0 and board[r - 1, c] >= 0:
            score += D[board[r - 1, c], f]
        if r < GRID - 1 and board[r + 1, c] >= 0:
            score += D[f, board[r + 1, c]]
    return float(score)


def _place_components(comps, R, D):
    board = -np.ones((GRID, GRID), np.int64)
    used = set()
    comps = sorted(comps, key=len, reverse=True)
    for comp0 in comps:
        comp = dict(comp0)
        miny = min(p[0] for p in comp.values())
        minx = min(p[1] for p in comp.values())
        comp = {f: (p[0] - miny, p[1] - minx) for f, p in comp.items()}
        maxy = max(p[0] for p in comp.values())
        maxx = max(p[1] for p in comp.values())
        best = None
        bestv = -1e30
        for sy in range(GRID - maxy):
            for sx in range(GRID - maxx):
                coords = [(y + sy, x + sx) for y, x in comp.values()]
                if any(board[y, x] >= 0 for y, x in coords):
                    continue
                v = _shift_score(comp, board, R, D, sy, sx)
                if best is None or v > bestv:
                    best = (sy, sx)
                    bestv = v
        if best is None:
            continue
        sy, sx = best
        for f, (y, x) in comp.items():
            if f not in used:
                board[y + sy, x + sx] = f
                used.add(f)
    return board, used


def _place_components_randomized(comps, R, D, rng, temperature=0.05, order_jitter=0.25):
    """Randomized greedy polyomino packing for objective-selected restarts."""
    board = -np.ones((GRID, GRID), np.int64)
    used = set()
    ranked = [
        (
            -(len(comp) + order_jitter * rng.gumbel()),
            index,
            comp,
        )
        for index, comp in enumerate(comps)
    ]
    ranked.sort(key=lambda item: (item[0], item[1]))
    for _, _, comp0 in ranked:
        comp = dict(comp0)
        miny = min(p[0] for p in comp.values())
        minx = min(p[1] for p in comp.values())
        comp = {f: (p[0] - miny, p[1] - minx) for f, p in comp.items()}
        maxy = max(p[0] for p in comp.values())
        maxx = max(p[1] for p in comp.values())
        choices = []
        for sy in range(GRID - maxy):
            for sx in range(GRID - maxx):
                coords = [(y + sy, x + sx) for y, x in comp.values()]
                if any(board[y, x] >= 0 for y, x in coords):
                    continue
                value = _shift_score(comp, board, R, D, sy, sx)
                if temperature > 0:
                    value += temperature * rng.gumbel()
                choices.append((value, sy, sx))
        if not choices:
            continue
        _, sy, sx = max(choices)
        for f, (y, x) in comp.items():
            if f not in used:
                board[y + sy, x + sx] = f
                used.add(f)
    return board, used


def _fill_board(board, used, R, D):
    unused = set(range(NFRAG)) - set(used)
    while unused:
        empties = np.argwhere(board < 0)
        if len(empties) == 0:
            break
        # Fill cells with most fixed neighbours first.
        counts = []
        for r, c in empties:
            n = 0
            if c > 0 and board[r, c - 1] >= 0: n += 1
            if c < GRID - 1 and board[r, c + 1] >= 0: n += 1
            if r > 0 and board[r - 1, c] >= 0: n += 1
            if r < GRID - 1 and board[r + 1, c] >= 0: n += 1
            counts.append(n)
        r, c = empties[int(np.argmax(counts))]
        best, bestv = None, -1e30
        for f in unused:
            v = 0.0
            if c > 0 and board[r, c - 1] >= 0:
                v += R[board[r, c - 1], f]
            if c < GRID - 1 and board[r, c + 1] >= 0:
                v += R[f, board[r, c + 1]]
            if r > 0 and board[r - 1, c] >= 0:
                v += D[board[r - 1, c], f]
            if r < GRID - 1 and board[r + 1, c] >= 0:
                v += D[f, board[r + 1, c]]
            if v > bestv:
                best, bestv = f, v
        board[r, c] = best
        unused.remove(best)
    return board.reshape(-1)


def _repair(place, R, D, passes=2, pool=96):
    place = place.copy()
    cur = objective(place, R, D)
    for _ in range(passes):
        bad = np.argsort(local_agreement(place, R, D))[:pool]
        improved = False
        for p in bad:
            bestq, bestv = None, cur
            for q in range(NFRAG):
                if p == q:
                    continue
                place[p], place[q] = place[q], place[p]
                v = objective(place, R, D)
                place[p], place[q] = place[q], place[p]
                if v > bestv:
                    bestq, bestv = q, v
            if bestq is not None:
                place[p], place[bestq] = place[bestq], place[p]
                cur = bestv
                improved = True
        if not improved:
            break
    return place, cur


def build_buddies_components(R, D, max_edges=900, min_margin=0.0):
    R = np.ascontiguousarray(R, np.float32)
    D = np.ascontiguousarray(D, np.float32)
    builder = _Builder()
    for _, _, a, b, dy, dx in _candidate_edges(R, D, max_edges=max_edges, min_margin=min_margin):
        builder.add_edge(a, b, dy, dx)
    return builder.components()


def build_directed_components(anchors, directions, targets, weights,
                              max_edges=256, cap=0):
    """Build conflict-safe components from externally calibrated U/D/L/R edges."""
    anchors = np.asarray(anchors, dtype=np.int64)
    directions = np.asarray(directions, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (anchors.shape == directions.shape == targets.shape == weights.shape):
        raise ValueError("directed edge arrays must have the same shape")
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    builder = _Builder(cap)
    order = np.argsort(-weights)[:max_edges]
    for index in order:
        dy, dx = deltas[int(directions[index])]
        builder.add_edge(
            int(anchors[index]),
            int(targets[index]),
            dy,
            dx,
        )
    return builder.components()


def solve_components_from_scores(
    R,
    D,
    components,
    repair_passes=0,
    restarts=1,
    seed=1234,
    temperature=0.05,
    order_jitter=0.25,
):
    """Pack a prebuilt component set and fill remaining board cells."""
    R = np.ascontiguousarray(R, np.float32)
    D = np.ascontiguousarray(D, np.float32)
    rng = np.random.default_rng(seed)
    best_place = None
    best_objective = -np.inf
    for restart in range(restarts):
        if restart == 0:
            board, used = _place_components(components, R, D)
        else:
            board, used = _place_components_randomized(
                components,
                R,
                D,
                rng,
                temperature=temperature,
                order_jitter=order_jitter,
            )
        place = _fill_board(board, used, R, D)
        if repair_passes > 0:
            place, value = _repair(place, R, D, passes=repair_passes)
        else:
            value = objective(place, R, D)
        if value > best_objective:
            best_place = place.copy()
            best_objective = float(value)
    if best_place is None:
        raise RuntimeError("component packing produced no placement")
    return best_place.astype(np.int64), best_objective


def solve_buddies_from_scores(R, D, max_edges=900, min_margin=0.0, repair_passes=2):
    R = np.ascontiguousarray(R, np.float32)
    D = np.ascontiguousarray(D, np.float32)
    components = build_buddies_components(R, D, max_edges, min_margin)
    return solve_components_from_scores(
        R,
        D,
        components,
        repair_passes=repair_passes,
    )


def solve_buddies_multistart_from_scores(
    R,
    D,
    max_edges=900,
    min_margin=0.0,
    repair_passes=0,
    restarts=16,
    seed=1234,
    temperature=0.05,
    order_jitter=0.25,
):
    """Keep component geometry fixed and search over global component packing."""
    if restarts < 1:
        raise ValueError("restarts must be positive")
    R = np.ascontiguousarray(R, np.float32)
    D = np.ascontiguousarray(D, np.float32)
    components = build_buddies_components(R, D, max_edges, min_margin)
    return solve_components_from_scores(
        R,
        D,
        components,
        repair_passes=repair_passes,
        restarts=restarts,
        seed=seed,
        temperature=temperature,
        order_jitter=order_jitter,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--preprocess", choices=("raw", "norm", "denoise", "denoise_norm"), default="raw")
    ap.add_argument("--denoise_tag", default="matchden")
    ap.add_argument("--pair_tag", default="pair")
    ap.add_argument("--max_edges", type=int, default=900)
    ap.add_argument("--min_margin", type=float, default=0.0)
    ap.add_argument("--repair_passes", type=int, default=2)
    ap.add_argument("--bs_score", type=int, default=4096)
    args = ap.parse_args()

    pair, pck = load_pair(args.pair_tag)
    if pair is None:
        raise FileNotFoundError("no pair checkpoint found")
    print(f"pair step={pck.get('step')} val={pck.get('val')}", flush=True)
    denoiser = None
    if args.preprocess in ("denoise", "denoise_norm"):
        denoiser, _ = load_match_denoiser(args.denoise_tag, device=DEV)
        if denoiser is None:
            raise FileNotFoundError("no matching denoiser checkpoint found")

    z = np.load(os.path.join(CACHE_DIR, "perms.npz"), allow_pickle=True)
    names_, inv_, conf_ = z["names"], z["inv"], z["conf"]  # materialize once; npz is lazy
    gt = {n: (inv_[i].astype(np.int64), conf_[i]) for i, n in enumerate(names_)}
    _, val = train_val_split()
    rows = []
    for nm in val[:args.n]:
        frags = to_frags(load(os.path.join(TRAIN_INP, nm)))
        sf = preprocess_frags_np(frags, args.preprocess, denoiser, DEV)
        R, D = pairwise_scores_full(pair, sf, DEV, bs=args.bs_score)
        place, obj = solve_buddies_from_scores(R, D, args.max_edges, args.min_margin, args.repair_passes)
        inv, conf = gt[nm]
        pacc, hi = placement_accuracy(place, inv, conf)
        nacc, nr, nd = neighbour_accuracy(place, inv)
        tgt = load(os.path.join(TRAIN_TGT, nm))
        ss = sk_ssim(tgt, assemble(frags, place), channel_axis=2, data_range=255)
        rows.append((pacc, hi, nacc, nr, nd, ss))
        print(f"{nm} place={pacc:.3f} hi={hi:.3f} neigh={nacc:.3f} "
              f"R={nr:.3f} D={nd:.3f} SSIM={ss:.3f} obj={obj:.0f}", flush=True)
    a = np.array(rows, np.float32)
    print(f"\n== buddies N={len(rows)} preprocess={args.preprocess} ==")
    print(f"place_acc   {a[:,0].mean():.4f}")
    print(f"hi_acc      {a[:,1].mean():.4f}")
    print(f"neigh_acc   {a[:,2].mean():.4f}  right={a[:,3].mean():.4f} down={a[:,4].mean():.4f}")
    print(f"SSIM_solve  {a[:,5].mean():.4f}")


if __name__ == "__main__":
    main()



