"""Train the learned seam matcher against the 576-way ranking it will face.

Trains on synthetic boards by default: distort.py was verified against real
pairs (residual 13.1 vs 13.3, autocorrelation 0.732 vs 0.735, JPEG blockiness
1.49 vs 1.50) and gives exact labels plus a fresh corruption of each tile every
epoch, where recovered permutations are 0.996 accurate only on the retained
half.  Real boards can be mixed in with --real-prob to guard the domain match.

Validation is always on REAL boards and reports the same R@1 every other scorer
in this repo reports, so the number is directly comparable: MGC on restored
tiles reaches 0.154, and the assembly threshold is about 0.52.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import CACHE_DIR, CKPT_DIR, NFRAG as N, TRAIN_INP, TRAIN_TGT
from distort import distort_frags, distort_frags_scaled
from mgc import mgc_cost
from restore_tile import to_frags
from seam_cost import cycle_consistency
from restore_tile import TileRestorer
from seam_embed import (SeamEmbed, board_logits, infonce, invariance_loss,
                        sinkhorn_loss,
                        predict_loss)


RESTORER = None
INPUT_RESTORER = None
INPUT_FILTER = ""


def load_restorer(name, dev):
    """A frozen per-tile restorer supplying the third input view.

    M167: costs from restored tiles are much worse alone (precision 0.199-0.32
    against the raw view's 0.449) and where the two views agree precision is
    0.95, because denoising and noise fail in different places.  Intersecting
    the two cost matrices discards every edge only one view gets right; this
    hands both to the model instead.
    """
    ck = torch.load(Path(CKPT_DIR) / name, map_location=dev, weights_only=False)
    a = ck.get("args", {})
    m = TileRestorer(a.get("ch", 64), a.get("blocks", 5),
                     a.get("residual", False), False,
                     a.get("ycc", False)).to(dev)
    m.load_state_dict(ck.get("model", ck))
    m.eval()
    for p_ in m.parameters():
        p_.requires_grad_(False)
    return m


@torch.no_grad()
def to_input(x):
    """The tiles the model actually sees.

    `--restored` ADDS the restorer's output as a third view, which M170 measured
    as worth nothing at equal budget.  `--restore-input` REPLACES the input with
    it, which is a different experiment and has never been run here: M66 and M91
    both concluded that restoration hurts matching, and both fed restored tiles
    to a matcher TRAINED ON RAW ones.  M199 then measured that the domain
    mismatch alone costs about 0.15 of edge precision, so that conclusion was
    never tested on its own terms.  A teammate's pair-relation model, trained
    and evaluated on restored tiles throughout, reads accuracy 0.378 against
    0.274 on raw and adjacency 0.483 against 0.311.
    """
    if INPUT_FILTER:
        return _filtered(x)
    if INPUT_RESTORER is None:
        return x
    out = []
    for i in range(0, x.shape[0], 288):
        with torch.autocast("cuda", torch.float16):
            out.append(INPUT_RESTORER(x[i:i + 288]).float())
    return torch.cat(out).clamp(0, 255)


def _filtered(x):
    """One of the shipped analytic views, applied exactly as inference does.

    Every matcher in this repo was trained on RAW fragments, and since M371 the
    pipeline shows each of them three views -- raw, median, bilateral -- two of
    which they have never seen. M199 measured the cost of exactly that kind of
    mismatch at about 0.15 of edge precision when the shift was restoration.
    A filter shifts the distribution far less than a restorer does, so the cost
    should be smaller, but it is not zero and nothing here has ever paid it.
    """
    from analytic_views import ANALYTIC_VIEWS
    fn = ANALYTIC_VIEWS[INPUT_FILTER]
    a = x.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
    out = np.stack([fn(np.ascontiguousarray(t)) for t in a])
    return torch.from_numpy(out).to(x.dtype).permute(0, 3, 1, 2).to(x.device)


@torch.no_grad()
def restored_view(x):
    """(B,3,20,20) raw tiles -> the restorer's output, same shape."""
    if RESTORER is None:
        return None
    out = []
    for i in range(0, x.shape[0], 288):
        with torch.autocast("cuda", torch.float16):
            out.append(RESTORER(x[i:i + 288]).float())
    return torch.cat(out).clamp(0, 255)

G = 24


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


class Boards(Dataset):
    """One item = a whole board of dirty tiles in TRUE grid order.

    The loss softmaxes over all 576 tiles, so a board cannot be subsampled --
    every tile is another tile's negative.  Low-confidence real positions are
    masked out of the loss rather than dropped.
    """

    def __init__(self, names, inv, margin, thr, real_prob=0.0, mix=0.0):
        self.names, self.inv, self.margin = names, inv, margin
        self.thr, self.real_prob, self.mix = thr, real_prob, mix

    def __len__(self):
        return len(self.names)

    def __getitem__(self, k):
        nm = str(self.names[k])
        if np.random.rand() < self.real_prob:
            iv = self.inv[k].astype(np.int64)
            d = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[iv]
            m = (self.margin[k] >= self.thr).astype(np.float32)
        else:
            clean = to_frags(load_rgb(Path(TRAIN_TGT) / nm))
            rng = np.random.default_rng()
            if self.mix and np.random.rand() < self.mix:
                # severity is sampled PER BOARD inside the worker: a schedule
                # mutated in the main process never reaches DataLoader workers,
                # which hold their own copy of the dataset
                d = distort_frags_scaled(clean.astype(np.uint8), rng,
                                         float(np.random.rand())).astype(np.float32)
            else:
                d = distort_frags(clean.astype(np.uint8), rng).astype(np.float32)
            m = np.ones(N, np.float32)
        cl = to_frags(load_rgb(Path(TRAIN_TGT) / nm)).astype(np.float32)
        # a SECOND independent corruption of the same clean board; the pair
        # shares only content, which is what the invariance loss asks for
        d2 = distort_frags(cl.astype(np.uint8),
                           np.random.default_rng()).astype(np.float32)
        return (torch.from_numpy(d).permute(0, 3, 1, 2), torch.from_numpy(m),
                torch.from_numpy(cl).permute(0, 3, 1, 2),
                torch.from_numpy(d2).permute(0, 3, 1, 2))


@torch.no_grad()
def evaluate(model, names, inv_all, n_boards, dev, with_mgc=False, twin_thr=10.0):
    model.eval()
    out = []
    for k in range(n_boards):
        nm = str(names[k]); iv = inv_all[k].astype(np.int64)
        d = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[iv]
        cl = to_frags(load_rgb(Path(TRAIN_TGT) / nm)).astype(np.float32)
        f = cl.reshape(N, -1)
        rms = np.sqrt(np.maximum((f ** 2).sum(1)[:, None] + (f ** 2).sum(1)[None]
                                 - 2 * f @ f.T, 0) / f.shape[1])
        x = torch.from_numpy(d).permute(0, 3, 1, 2).to(dev)
        with torch.autocast("cuda", torch.float16):
            desc = model(to_input(x), restored_view(x))[:4]
        desc = [t.float() for t in desc[:4]]
        # Calibrated scores: log-probability at the model's own temperature,
        # Sinkhorn, then cycle consistency.  Raw cosines understate R@1 by about
        # 0.10 (M99) and are not what any solver consumes.
        lg = []
        for ax in ("h", "v"):
            A = board_logits(desc, ax, getattr(model, "modes", 1),
                             getattr(model, "mode_tau", 0.0)).float()                 * model.logit_scale.exp().detach()
            A.fill_diagonal_(-1e4)
            lg.append(A)
        HH, VV = cycle_consistency(lg[0], lg[1])
        cal = {"h": HH, "v": VV}
        for t in cal.values():
            t.fill_diagonal_(-1e4)
        row = []
        for axis, step, ok in (("h", 1, lambda p: p % G != G - 1),
                               ("v", G, lambda p: p < N - G)):
            S = cal[axis]
            idx = torch.tensor([p for p in range(N) if ok(p)], device=dev)
            o = S[idx].argsort(1, descending=True)
            tgt = (idx + step)[:, None]
            rk = (o == tgt).float().argmax(1)
            # a pick that is a visual twin of the true neighbour is not an
            # assembly error, so report it separately rather than scoring it
            # as a miss (M68 flagged this as a measurement artefact)
            pick = o[:, 0].cpu().numpy()
            tt = tgt[:, 0].cpu().numpy()
            row += [(rk == 0).float().mean().item(), (rk < 20).float().mean().item(),
                    float(np.mean([p == t or rms[t, p] < twin_thr
                                   for p, t in zip(pick, tt)]))]
        if with_mgc:
            for axis, step, ok in (("h", 1, lambda p: p % G != G - 1),
                                   ("v", G, lambda p: p < N - G)):
                C = mgc_cost(d, axis); np.fill_diagonal(C, np.inf)
                i2 = np.array([p for p in range(N) if ok(p)])
                row.append((C[i2].argmin(1) == i2 + step).mean())
        out.append(row)
    model.train()
    v = np.mean(out, axis=0)
    return {"R@1": (v[0] + v[3]) / 2, "R@20": (v[1] + v[4]) / 2,
            "twinR@1": (v[2] + v[5]) / 2,
            **({"mgc_R@1": (v[6] + v[7]) / 2} if with_mgc else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--strip", type=int, default=3)
    ap.add_argument("--rows", default="",
                    help="restrict a local head to these seam rows, as a:b. "
                         "Two matchers on disjoint halves are the only pair of "
                         "scorers in this project whose errors are independent "
                         "(M311: true-edge agreement 0.214 against 0.877 for "
                         "two separately trained networks on the whole seam)")
    ap.add_argument("--modes", type=int, default=1,
                    help="read the descriptor as this many sub-vectors and score "
                         "a pair by the BEST-agreeing one. A maximum is not "
                         "bilinear, which is the one thing capacity, depth and "
                         "dimension cannot change (M197, M306)")
    ap.add_argument("--mode-tau", type=float, default=0.0,
                    help="soften the maximum over modes into a logsumexp at this "
                         "temperature; a hard max trains only the winning mode")
    ap.add_argument("--head", default="global", choices=["global", "local"])
    ap.add_argument("--restore-input", default="",
                    help="replace the input tiles with this restorer's output, "
                         "for both training and evaluation")
    ap.add_argument("--filter-input", default="", choices=[
        "", "median", "bilateral", "bilat_mild", "guided2", "guided4", "nlm",
        "unsharp"],
        help="train and evaluate on this analytic view instead of the raw "
             "fragments, so the matcher sees at training time what the pipeline "
             "shows it at inference. M292 measured that a matcher trained on "
             "RESTORED tiles plateaus at 0.295 against 0.334 on raw, but M372 "
             "found the two transforms differ in kind -- a restorer invents "
             "detail the matcher then believes, a filter can only remove -- so "
             "this has never been tested on its own terms")
    ap.add_argument("--restored", default="",
                    help="per-tile restorer checkpoint supplying a third input "
                         "view; empty keeps the two-view model")
    ap.add_argument("--norm-only", action="store_true",
                    help="feed only the photometry-normalised view")
    ap.add_argument("--invariance-weight", type=float, default=0.0,
                    help="weight of the same-tile-under-fresh-corruption loss")
    ap.add_argument("--photo-jitter", type=float, default=0.0,
                    help="extra per-tile gain/offset applied at train time")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--real-prob", type=float, default=0.0)
    ap.add_argument("--sinkhorn-weight", type=float, default=0.0,
                    help="mix in a cross-entropy on the doubly-stochastic "
                         "projection of the score matrix, which is the shape "
                         "the pipeline actually decodes (M395, M396). M116 "
                         "REPLACED the loss with this at twenty unrolled "
                         "iterations and two consistency rounds and the model "
                         "collapsed, R@1 0.3527 to 0.0009; mixed at a small "
                         "weight with few iterations the raw objective keeps "
                         "anchoring it, which is the standard remedy and was "
                         "never tried")
    ap.add_argument("--sinkhorn-iters", type=int, default=3)
    ap.add_argument("--predict-weight", type=float, default=0.0,
                    help="weight of the continuation-prediction auxiliary loss")
    ap.add_argument("--twin-thr", type=float, default=0.0,
                    help="RMS grey levels below which two tiles count as one")
    ap.add_argument("--mix", type=float, default=0.0,
                    help="fraction of boards at a randomly sampled severity")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-boards", type=int, default=6)
    ap.add_argument("--calibrate", type=int, default=0,
                    help="consistency rounds inside the training loss")
    ap.add_argument("--init", default="",
                    help="warm-start weights from this checkpoint")
    ap.add_argument("--out", default="seam_embed.pt")
    a = ap.parse_args()

    dev = "cuda"
    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names, inv, margin = blob["names"], blob["inv"], blob["margin"]
    thr = float(np.quantile(margin, 0.5))
    cut = len(names) - 300
    tr = Boards(names[:cut], inv[:cut], margin[:cut], thr, a.real_prob, a.mix)
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, persistent_workers=a.workers > 0)

    global RESTORER, INPUT_RESTORER, INPUT_FILTER
    if a.filter_input:
        INPUT_FILTER = a.filter_input
        print(f"input filtered by {a.filter_input}", flush=True)
    if a.restore_input:
        INPUT_RESTORER = load_restorer(a.restore_input, dev)
        print(f"input replaced by {a.restore_input}", flush=True)
    if a.restored:
        RESTORER = load_restorer(a.restored, dev)
        print(f"restored view from {a.restored}", flush=True)
    model = SeamEmbed(a.ch, a.blocks, a.dim, a.strip, a.head,
                      predict=a.predict_weight > 0,
                      norm_only=a.norm_only,
                      restored=bool(a.restored)).to(dev)
    if a.rows:
        lo, hi = (int(v) for v in a.rows.split(":"))
        model.rows = list(range(lo, hi))
        print(f"reading only seam rows {lo}:{hi} of 20", flush=True)
    model.modes = a.modes
    model.mode_tau = a.mode_tau
    if a.modes > 1:
        assert a.dim % a.modes == 0, "dim must divide by modes"
        print(f"scoring by the best of {a.modes} descriptor modes of "
              f"{a.dim // a.modes}", flush=True)
    if a.init:
        prev = torch.load(Path(CKPT_DIR) / a.init, map_location=dev,
                          weights_only=False)
        missing, unexpected = model.load_state_dict(prev["model"], strict=False)
        print(f"warm start from {a.init} step {prev.get('step')}: "
              f"{len(missing)} new tensors, {len(unexpected)} dropped", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    base = evaluate(model, names[cut:], inv[cut:], a.eval_boards, dev, with_mgc=True)
    print(f"[init] R@1 {base['R@1']:.4f}  MGC baseline on the same boards "
          f"{base['mgc_R@1']:.4f}", flush=True)

    step, t0, run = 0, time.time(), []
    while step < a.steps:
        for x, m, cl, x2 in dl:
            if step >= a.steps:
                break
            x = x.to(dev, non_blocking=True)
            if a.photo_jitter > 0:
                # an EXTRA independent affine per tile, on top of the one the
                # generator already applied.  The label does not change, so the
                # only way to keep the loss down is to stop using absolute level
                # one scalar gain and offset PER TILE, matching the generator's
                # own affine; x arrives as (batch, tiles, 3, 20, 20)
                j = a.photo_jitter
                shape = x.shape[:2] + (1, 1, 1)
                g = 1.0 + (torch.rand(shape, device=dev) - 0.5) * 2 * j
                o = (torch.rand(shape, device=dev) - 0.5) * 2 * 60 * j
                m = x.mean((-3, -2, -1), keepdim=True)
                x = ((x - m) * g + m + o).clamp(0, 255)
            need_clean = a.twin_thr > 0 or a.predict_weight > 0
            cl = cl.to(dev, non_blocking=True) if need_clean else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16):
                loss = 0.0
                for b in range(x.shape[0]):          # each board is its own softmax
                    out = model(to_input(x[b]), restored_view(x[b]))
                    desc = out[:4]
                    l, _ = infonce(desc, model.logit_scale.exp(),
                                   clean=None if cl is None else cl[b],
                                   twin_thr=a.twin_thr, calibrate=a.calibrate,
                                   modes=a.modes, mode_tau=a.mode_tau)
                    if a.sinkhorn_weight > 0:
                        l = l + a.sinkhorn_weight * sinkhorn_loss(
                            desc, model.logit_scale.exp(), iters=a.sinkhorn_iters,
                            modes=a.modes, mode_tau=a.mode_tau)
                    if a.predict_weight > 0:
                        l = l + a.predict_weight * predict_loss(
                            out[4], cl[b], strip=a.strip)
                    if a.invariance_weight > 0:
                        xb2 = x2[b].to(dev, non_blocking=True)
                        d2 = model(to_input(xb2), restored_view(xb2))[:4]
                        l = l + a.invariance_weight * invariance_loss(
                            desc, d2, model.logit_scale.exp())
                    loss = loss + l
                loss = loss / x.shape[0]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run.append(float(loss)); step += 1
            if step % 100 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**20
                print(f"step {step:5d}  loss {np.mean(run[-100:]):.4f}  "
                      f"{(time.time()-t0)/step:.2f} s/step  {mem:.0f} MiB", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                e = evaluate(model, names[cut:], inv[cut:], a.eval_boards, dev)
                print(f"  [eval @ {step}] R@1 {e['R@1']:.4f}  R@20 {e['R@20']:.4f}  "
                      f"twin {e['twinR@1']:.4f}  "
                      f"(MGC 0.154, threshold ~0.52)", flush=True)
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "eval": e, "step": step}, Path(CKPT_DIR) / a.out)


if __name__ == "__main__":
    main()
