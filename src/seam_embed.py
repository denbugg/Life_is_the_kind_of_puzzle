"""Learned seam matcher: a tile becomes four directional descriptors.

Why replace MGC
---------------
Every scorer in this repo is MGC or a relative of it, and the restorer was
trained to feed MGC better tiles.  MGC was designed for undamaged puzzles: it
predicts the colour gradient across a seam from each tile's internal gradient
covariance, a statistic whose covariance estimate is inflated by our noise and
which was never optimal under it.  Measurements that pin the blame on MGC
rather than on the tiles:

    clean blurred tiles, MGC              R@1 0.759   (not 1.0 -- MGC loses a
                                                       quarter with no noise)
    measured ring residual 13.4 maps to       ~0.25   from the response curve
    dirty tiles, MGC                          0.050
    dirty tiles, oracle affine removed        0.077   (so the affine is minor)
    restored tiles, MGC                       0.154

The tiles carry roughly 0.25 worth of signal and MGC extracts 0.05-0.15 of it.
This module optimises R@1 directly instead of hoping a hand-designed statistic
correlates with it.

Design
------
Matching against 576 candidates makes joint pair scoring quadratic and
untrainable at full-board scale, which is what sank the earlier pair CNN.  A
siamese formulation keeps it linear: each tile yields a "right" and a "left"
descriptor (and "down"/"up"), and a seam score is one dot product, so a whole
board is a single 576x576 matmul and the InfoNCE softmax runs over exactly the
candidate set the solver will face.

Each tile enters both as raw values and in a noise-aware normalised copy: the
per-tile affine is real but minor, and letting the trunk see both spares it from
having to undo the gain itself.  std(dirty)^2 = a^2 s^2 + n^2, so the noise
variance is subtracted before dividing -- on a flat tile the raw std is mostly
noise, and dividing by it (plain z-norm) drops R@1 from 0.154 to 0.068.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RING_SIGMA = 13.4          # measured on real (input, target) pairs


class Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, ch)
        self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        return x + self.n2(self.c2(h))


class SeamEmbed(nn.Module):
    """(B,3,20,20) uint8-range tiles -> 4 unit descriptors of size `dim` each."""

    def __init__(self, ch=64, blocks=4, dim=128, strip=3, head="global",
                 predict=False, norm_only=False, restored=False):
        super().__init__()
        self.strip = strip
        self.predict = predict
        # Absolute brightness is pure nuisance here: the generator gives every
        # tile its own offset and gain independently, so a tile's mean level
        # carries NO information about who its neighbours are.  The matcher
        # leans on it anyway -- among its wrong picks, the chosen tile is closer
        # in mean brightness than the true neighbour 63.2% of the time, where a
        # random tile manages only 39.5%.  True neighbours differ by 40.8 grey
        # levels on average and the model prefers candidates 26.8 apart.
        # norm_only drops the raw view so the nuisance is simply unavailable.
        self.norm_only = norm_only
        # A third view: the same tile after a per-tile restorer.  M167 measured
        # why it belongs here.  Alone it is much worse than the raw view (edge
        # precision 0.199-0.32 against 0.449) because restoration invents detail
        # the matcher then believes, but where the two views AGREE precision is
        # 0.71 and on the confident half 0.95 -- higher than anything else this
        # project has produced, because denoising and noise fail in different
        # places.  Intersecting the two cost matrices throws away every edge
        # only one view gets right (M169: the detectors overlap almost
        # completely and the union adds nothing); giving the model both and
        # letting it fuse them internally does not.
        self.restored = restored
        n_in = 3 if norm_only else 6
        if restored:
            n_in += 3
        self.stem = nn.Conv2d(n_in, ch, 3, padding=1)
        self.trunk = nn.Sequential(*[Block(ch) for _ in range(blocks)])
        # one head per direction; a seam is directional, so right and left must
        # not share weights or the score collapses to a symmetric similarity
        self.head_kind = head
        if head == "local":
            # One descriptor per position ALONG the seam, concatenated.  The dot
            # product then sums 20 local agreements instead of comparing two
            # summaries of the whole edge, which is what a seam actually is: row r
            # of one tile continues into row r of the other.  A flattened Linear
            # can represent this but has to discover it, and it carries 170x more
            # parameters per head while doing so.
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
                              nn.Conv2d(ch, dim, (1, strip)))
                for _ in range(4)])
        else:
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
                              nn.Flatten(), nn.Linear(ch * strip * 20, dim))
                for _ in range(4)])
        # Which rows of the seam this matcher is allowed to read.  M311 measured
        # that the top half of a seam and the bottom half agree on only 21% of
        # the true edges they find, against 88% for two independently trained
        # networks and 63% for two independent draws of the noise -- because the
        # noise is per-pixel, so different ROWS of one seam carry different
        # draws of it.  That is the only genuinely independent evidence in this
        # project, and M311 tested it unfairly, by slicing a descriptor trained
        # to work as a sum over all twenty rows.  A matcher TRAINED on its half
        # sees half the evidence and is optimised for it.
        self.rows = None
        self.logit_scale = nn.Parameter(torch.tensor(2.5))
        # how many sub-vectors `board_logits` should read the descriptor as
        self.modes = 1
        if predict:
            # Auxiliary task: from a tile alone, guess the CLEAN pixels just
            # beyond its right and lower edges.  The ranking loss delivers about
            # one bit per row; this delivers 180 numbers, and it is the exact
            # notion the descriptors need -- what a seam continuing here would
            # look like.  As a SCORER this failed twice (M47 extrapolation, M56
            # inpainting) because prediction error exceeds the noise floor, but
            # as a shaping signal for the trunk none of that applies.
            self.pred = nn.ModuleList([
                nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
                              nn.Conv2d(ch, 3 * strip, (1, strip)))
                for _ in range(2)])

    def prep(self, x, rest=None):
        s = x.flatten(2)
        mu = s.mean(-1)[:, :, None, None]
        var = s.var(-1)[:, :, None, None] - RING_SIGMA ** 2
        sd = torch.sqrt(torch.clamp(var, min=(0.25 * RING_SIGMA) ** 2))
        z = (x - mu) / sd / 4.0
        views = [z] if self.norm_only else [x / 255.0 - 0.5, z]
        if self.restored:
            if rest is None:
                raise ValueError("this matcher was built with restored=True "
                                 "and needs the restored view")
            views.append(rest / 255.0 - 0.5)
        return views[0] if len(views) == 1 else torch.cat(views, 1)

    def forward(self, x, rest=None):
        f = self.trunk(self.stem(self.prep(x, rest)))
        k = self.strip
        strips = (f[:, :, :, -k:],                       # right edge
                  f[:, :, :, :k],                        # left edge
                  f[:, :, -k:, :].transpose(2, 3),       # bottom edge
                  f[:, :, :k, :].transpose(2, 3))        # top edge
        out = []
        for h, s in zip(self.heads, strips):
            d = h(s)
            if self.head_kind == "local":
                if self.rows is not None:
                    d = d[:, :, self.rows]
                d = d.flatten(1)          # (B, dim*k): k stacked local descriptors
            out.append(F.normalize(d, dim=-1))
        if self.predict:
            # (B, 3, strip, 20) each: the guessed continuation past the right
            # and lower edges, in the same layout as the strips they face
            out.append([p(s).reshape(f.shape[0], 3, k, 20)
                        for p, s in zip(self.pred, (strips[0], strips[2]))])
        return out


def board_logits(desc, axis, modes=1, mode_tau=0.0):
    """Full 576x576 score matrix; entry (i,j) scores tile j placed after tile i.

    With `modes` > 1 the descriptor is read as that many independent sub-vectors
    and the score is the MAXIMUM over them.  The reason is structural: any score
    of the form (descriptor, descriptor) is bilinear and therefore a dot product
    in some feature space, so capacity, depth and dimension all move within one
    family -- which is why M197 closed steps and ch/blocks and M306 closed
    dimension, all of them leaving R@20 where it was.  A maximum is not
    bilinear.  It expresses "these two edges agree under ANY of K patterns",
    which a single inner product cannot, and it costs K matrix products rather
    than the quadratic price of a joint encoder.

    `mode_tau` softens the maximum into a temperature-scaled logsumexp. A hard
    maximum passes gradient only to the winning sub-vector, so with K modes each
    one is trained roughly K times less often -- the soft form keeps the score
    non-bilinear while every mode still learns from every pair.

    modes=1 reproduces the old behaviour exactly, so existing checkpoints are
    unaffected.
    """
    r, l, d, u = desc
    a, b = (r, l) if axis == "h" else (d, u)
    if modes <= 1:
        return a @ b.t()
    n, dim = a.shape
    per = dim // modes
    a = a.reshape(n, modes, per)
    b = b.reshape(n, modes, per)
    # (modes, n, n) then the best-agreeing mode per pair
    per_mode = torch.einsum("ikd,jkd->kij", a, b)
    if mode_tau > 0:
        return mode_tau * torch.logsumexp(per_mode / mode_tau, dim=0)
    return per_mode.amax(0)


def invariance_loss(desc_a, desc_b, scale):
    """Recognise the SAME tile under a second, independent corruption.

    The generator draws a fresh gain, offset, noise field and JPEG quality for
    every tile every epoch, so two corruptions of one clean tile share only the
    underlying content.  Asking the descriptor to identify its own tile across
    that pair forces it to encode what survives the corruption and to discard
    what does not -- including absolute brightness, which carries no information
    about adjacency yet drives a measurable share of the matcher's errors
    (M129: among wrong picks the chosen tile is closer in mean level than the
    true neighbour 63.2% of the time, against 39.5% for a random tile).

    This is a different question from the seam loss: not "who is my neighbour"
    but "which of these 576 noisy tiles is me".  It supervises the trunk without
    touching the ranking objective.
    """
    a = F.normalize(torch.cat(desc_a, 1), dim=-1)
    b = F.normalize(torch.cat(desc_b, 1), dim=-1)
    logits = (a @ b.t()) * scale
    tgt = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def predict_loss(pred, clean, grid=24, strip=3):
    """L1 between the guessed continuation and the neighbour's clean pixels.

    Scored after removing each block's own mean and scale.  The tile carries an
    unknown per-piece gain and offset, so demanding absolute grey levels would
    charge the model for something it cannot recover (M73: the affine is worth
    only 0.027 of matching and is not what we want the trunk to spend capacity
    on).  Structure is the part that transfers.

    pred: [right, down], each (n, 3, strip, 20), as returned by SeamEmbed.
    clean: (n, 3, 20, 20) undamaged tiles in true grid order.
    """
    n = clean.shape[0]
    dev = clean.device
    pos = torch.arange(n, device=dev)
    loss = 0.0
    for p, sel, tgt in (
            (pred[0], pos[pos % grid != grid - 1], 1),
            (pred[1], pos[pos < n - grid], grid)):
        if sel.numel() == 0:
            continue
        nb = clean[sel + tgt]
        t = (nb[:, :, :, :strip].permute(0, 1, 3, 2) if tgt == 1
             else nb[:, :, :strip, :])
        a, b = p[sel], t
        a = (a - a.mean((1, 2, 3), keepdim=True)) / (a.std((1, 2, 3), keepdim=True) + 1e-5)
        b = (b - b.mean((1, 2, 3), keepdim=True)) / (b.std((1, 2, 3), keepdim=True) + 1e-5)
        loss = loss + (a - b).abs().mean()
    return loss / 2.0


def twin_targets(clean, tgt_idx, thr, exclude=None):
    """A soft target that spreads mass over visually identical tiles.

    A board with a large flat region contains tiles that are the same tile as
    far as any pixel measure goes: 27.9% of rows have a true neighbour with a
    twin within 10 grey levels RMS, against a noise floor of 13.4.  A one-hot
    target on those rows demands a distinction that is not present in the data
    and punishes a choice the assembly does not care about -- putting one patch
    of sky where its twin belongs leaves SSIM essentially unchanged.  Mass is
    spread uniformly over the twin set, which always contains the true
    neighbour itself.

    clean: (n, 3, 20, 20) undamaged tiles, used only to define identity.
    """
    n = clean.shape[0]
    f = clean.reshape(n, -1)
    d2 = (f * f).sum(1)[:, None] + (f * f).sum(1)[None] - 2.0 * (f @ f.t())
    rms = torch.sqrt(torch.clamp(d2, min=0.0) / f.shape[1])
    t = (rms[tgt_idx] < thr).float()
    ar = torch.arange(len(tgt_idx), device=clean.device)
    # Cap the twin set.  Spreading mass EVENLY over every twin is what destroyed
    # a run: on a board with a large flat region every patch of sky is every
    # other one's twin, so the target demanded the model divide its belief among
    # a hundred tiles, which costs about 14 nats a row.  The loss started at
    # 11.15 against ln(576) = 6.36 and settled at exactly uniform.  Keep only the
    # closest few, and leave most of the mass on the true neighbour.
    if t.sum(1).max() > 1:
        far = rms[tgt_idx].clone()
        far[t == 0] = float("inf")
        keep = far.argsort(1)[:, :4]
        m = torch.zeros_like(t)
        m.scatter_(1, keep, 1.0)
        t = t * m
    if exclude is not None:
        # Two adjacent sky tiles are twins of EACH OTHER, so a row's own tile
        # routinely lands in its target's twin set -- and that column is the
        # masked diagonal of the score matrix.  Left in, the soft target puts
        # mass on a -1e4 logit and the row contributes ~5000 to the loss.
        t[ar, exclude] = 0.0
    t[ar, tgt_idx] = 0.0
    t = t / t.sum(1, keepdim=True).clamp_min(1e-9) * 0.25
    t[ar, tgt_idx] = 0.75                      # the truth keeps the bulk
    return t / t.sum(1, keepdim=True)


def infonce(desc, scale, grid=24, clean=None, twin_thr=0.0, calibrate=0,
            modes=1, mode_tau=0.0):
    """Softmax over every candidate tile, in both directions, for both axes.

    With `calibrate` rounds the loss is taken on the SAME transformed scores the
    solvers consume -- Sinkhorn plus cycle consistency -- instead of on raw dot
    products.  Training on the raw scores optimises something the pipeline never
    sees: calibration is worth R@1 0.245 -> 0.348 at inference (M99), and none of
    that gain was asked for during training.  Both operations are a handful of
    differentiable matmuls, so the gradient can flow through them.
    """
    n = grid * grid
    dev = desc[0].device
    eye = torch.eye(n, device=dev, dtype=torch.bool)
    if calibrate > 0:
        from seam_cost import cycle_consistency
        # float32 regardless of autocast: Sinkhorn is repeated logsumexp over
        # logits already multiplied by a temperature near 27, and in fp16 that
        # overflows -- the first attempt produced nan within 700 steps
        raw = {}
        for ax in ("h", "v"):
            A = board_logits(desc, ax, modes, mode_tau).float() * scale.float()
            raw[ax] = A.masked_fill(eye, -1e4)
        ch, cv = cycle_consistency(raw["h"], raw["v"], rounds=calibrate,
                                   weight=0.5, iters=10, acyclic=0.0)
        cal = {"h": ch, "v": cv}
    loss, acc = 0.0, []
    for axis, step, mask in (
            ("h", 1, torch.tensor([p % grid != grid - 1 for p in range(n)], device=dev)),
            ("v", grid, torch.tensor([p < n - grid for p in range(n)], device=dev))):
        if calibrate > 0:
            # already log-normalised by Sinkhorn; a temperature on top would
            # fight the normaliser, so it is applied before, not after
            S = cal[axis].masked_fill(eye, -1e4)
        else:
            S = board_logits(desc, axis, modes, mode_tau) * scale
            S = S.masked_fill(eye, -1e4)
        rows = mask.nonzero(as_tuple=True)[0]
        tgt = rows + step
        if clean is not None and twin_thr > 0:
            tf = twin_targets(clean, tgt, twin_thr, exclude=rows)
            tb = twin_targets(clean, rows, twin_thr, exclude=tgt)
            loss = loss - (tf * F.log_softmax(S[rows], -1)).sum(1).mean()
            loss = loss - (tb * F.log_softmax(S.t()[tgt], -1)).sum(1).mean()
        else:
            loss = loss + F.cross_entropy(S[rows], tgt)
            # the reverse lookup shares the descriptors but is a different
            # softmax; training only the forward one leaves columns unpinned
            loss = loss + F.cross_entropy(S.t()[tgt], rows)
        acc.append((S[rows].argmax(1) == tgt).float().mean())
    return loss / 4.0, torch.stack(acc).mean()
