"""The submission: assemble all 576, level the seams, restore against the field.

What changed since `infer_conformant.py`
----------------------------------------
That module assembled the board and then interpolated between two restorers.
Three measurements taken afterwards say the interpolation was the wrong knob.

M178: the flat fill scores 0.3525 on held-out boards and every arm this project
has shipped scores below it, because at full restoration the network emits a
smooth image whose low frequencies are the local means of a SCRAMBLED board.
That structure is uncorrelated with the target, and SSIM charges for
uncorrelated structure while a constant carries none and pays nothing.

M177: the predicted colour field, drawn on its own, beats the flat fill by
0.016 -- it is the only correlated estimate of the board's low frequencies we
have.  With the TRUE field, a colour assignment of all 576 fragments recovers
absolute position at twenty-one times chance and scores above the constant.

M179: the per-fragment brightness the generator hands out can be estimated from
the steps at the seams and removed, with no knowledge of the layout, and it is
worth about 0.005 at equal detail on top of everything else.

So the picture is built as

    field(x, y)  +  alpha * highpass(restored assembly, sigma)

with every fragment present, unaltered in shape or position, and corrected only
in brightness -- which is the restoration the organisers ask for by name.
`alpha` trades detail against score and `sigma` decides at which scale the
surviving texture lives; both are choices about what an expert should see.
alpha 0 is the field alone and is not a submission, since it draws no fragments.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import infer_rank96 as rank96
from coarse_field import CoarseField
from config import (CACHE_DIR, CKPT_DIR, GRID as G, SUB_DIR, TEST_DIR, TRAIN_INP,
                    TRAIN_TGT)
from infer_coarse_field import deterministic_zip, load_rgb
from infer_conformant import (agreed_edges, assemble, detail, load_unet, _net,
                              _restore_tiles)
from border_prior import border_prior, border_scores, content_scores
from harvest_votes import voted_edges, voted_pool
from place_search import anneal, corroborate, fill_rest, fill_seams, search
from level_seams import level
from restore_tile import TileRestorer, to_frags
from seam_cost import costs_from_models
from seam_embed import SeamEmbed
from solve_buddies import build_directed_components, solve_components_from_scores

DIR_RIGHT, DIR_DOWN = 3, 1
N = G * G


def load_field(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["args"]
    m = CoarseField(a["n"], a["ch"], a["dim"], a["hidden"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def predict_field(model, tiles, dev):
    """Bag of tiles -> (480,480,3) and (576,3), both in 0..255."""
    x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2)[None].to(dev)
    f = model(x)
    big = (F.interpolate(f, size=(480, 480), mode="bicubic", align_corners=False)
           .clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0)
    cells = (F.interpolate(f, size=(G, G), mode="bicubic", align_corners=False)
             .clamp(0, 1)[0].permute(1, 2, 0).reshape(N, 3).cpu().numpy() * 255.0)
    return big, cells


def _assign(cell_colour, tile_colour, cells, tiles_idx):
    d = ((cell_colour[cells][:, None, :] - tile_colour[tiles_idx][None, :, :]) ** 2
         ).sum(-1)
    r, c = linear_sum_assignment(d)
    return cells[r], tiles_idx[c]


def component_health(comp, CH, CV):
    """Does a component confirm itself under its own evidence? (M206, M336)

    For every tile with two or more in-component neighbours, each neighbour
    names its best candidate from OUTSIDE the component; the tile is confirmed
    when they agree on the tile that is actually there.  M336 measured this
    correlating with a component's coherent fraction at +0.310 and choosing a
    correct core 0.836 of the time against 0.629 by size -- and found it useless
    as an edge FILTER, because dropping components shrinks the largest coherent
    block, which is what placement tracks.

    M358 supplies the use it was missing.  A component is placed by the annealer
    as a UNIT, so a large one in the wrong place costs more SSIM than the same
    fragments scattered, while fragments left to the fill are placed by COLOUR,
    which SSIM rewards.  So the question is not which components to trust for
    the harvest but which to trust enough to place as a BLOCK.
    """
    at = {(dy, dx): t for t, (dy, dx) in comp.items()}
    inside = set(comp)
    fired = tested = 0
    for t, (dy, dx) in comp.items():
        nb = []
        for (ddy, ddx), (M, fwd) in (((0, -1), (CH, False)), ((0, 1), (CH, True)),
                                     ((-1, 0), (CV, False)), ((1, 0), (CV, True))):
            u = at.get((dy + ddy, dx + ddx))
            if u is not None:
                nb.append((M, u, fwd))
        if len(nb) < 2:
            continue
        votes_ = []
        for M, u, fwd in nb:
            col = (M[:, u] if fwd else M[u, :]).copy()
            col[[x for x in inside if x != t]] = np.inf
            votes_.append(int(np.argmin(col)))
        tested += 1
        fired += int(len(set(votes_)) == 1 and votes_[0] == t)
    return (fired / tested) if tested else None


def solve_layout(matcher, restorers, tiles, dev, cell_colour, fill, votes=8,
                 margin=0.0, orientations=2, frame=0.0, anneal_iters=20000,
                 border_net=None, content=0.0, place="descent",
                 corroboration=4.0, vote_target=0, order="margin",
                 dissolve=0.0, dissolve_min=6, seed=0, jump=1.0, step=3.0):
    """Full 576-fragment bijection. `fill` decides how the leftovers are placed.

    The harvest agrees across architectures AND inputs at once (M212, M214),
    which reaches 0.259 of the board inside internally-correct components where
    every single-axis scheme capped at 0.155. Six votes of nine is the point the
    LAYOUT wants rather than the cleanest one (M215): adjacency 0.234 against
    0.192 for the previous harvest, at equal placement.

    With `frame`, the components are placed by annealing an objective that
    carries the border prior instead of by the greedy packer.  Two measurements
    put them there.  M232 found the packer is PAID to keep components apart --
    it receives R = -CH with CH non-negative, so every contact subtracts, and
    half the boards come back with zero inter-component contacts.  M245 then
    found that fixing the objective is not enough, because it is nearly flat
    near its maximum: the search reaches 88% of the value the true arrangement
    attains while placing thirty times fewer fragments, so what it lacks is a
    second and independent opinion.  The frame is one -- absolute where every
    seam is relative -- and on 24 boards it doubles placement and adds 0.015
    SSIM over the packer (M246, M247).
    """
    pool = None
    if votes:
        CH, CV = costs_from_models(matcher, tiles)
        views = [tiles] + [_restore_tiles(m, tiles, dev) for m in restorers]
        agreed = voted_edges(matcher, views, dev, votes,
                             orientations=orientations, margin=margin,
                             target=vote_target, order=order)
        if frame > 0 and corroboration > 0:
            pool = voted_pool(matcher, views, dev, orientations)
    else:
        CH, CV, agreed = agreed_edges(matcher, restorers, tiles, dev)
    comps = build_directed_components(
        [i for (i, j, o) in agreed],
        [DIR_RIGHT if o == (0, 1) else DIR_DOWN for (i, j, o) in agreed],
        [j for (i, j, o) in agreed], list(agreed.values()), max_edges=len(agreed))
    placed = comps
    if frame > 0:
        prior = border_prior(border_scores(matcher, tiles, dev),
                             content_scores(border_net, tiles, dev)
                             if border_net is not None else None, content)
        placed = [c for c in comps if len(c) > 1]
        if dissolve > 0:
            # a component we do not trust is not placed as a block; its
            # fragments go to the fill, where colour decides (M358)
            keep = []
            for c in placed:
                if len(c) < dissolve_min:
                    keep.append(c)
                    continue
                h = component_health(c, CH, CV)
                if h is None or h >= dissolve:
                    keep.append(c)
            placed = keep
        H, V = (corroborate(placed, pool, CH, CV, corroboration)
                if pool else (CH, CV))
        if place == "anneal":
            board, _ = anneal(placed, H, V, iters=anneal_iters,
                              baseline_q=0.15, prior=prior, lam=frame, seed=seed,
                              jump=jump, step=step, swap=swap)
        else:
            board, _, _ = search(placed, H, V, rounds=6, baseline_q=0.15,
                                 prior=prior, lam=frame)
        lay = fill_seams(board, H, V)
    else:
        lay = np.asarray(solve_components_from_scores(
            (-CH).astype(np.float32), (-CV).astype(np.float32), comps,
            repair_passes=0, restarts=1)[0], np.int64)
    if fill == "field":
        tile_colour = np.mean([_restore_tiles(m, tiles, dev).mean((1, 2))
                               for m in restorers], axis=0)
        # only what was actually PLACED is owned: the annealer positions the
        # multi-tile components and `fill_rest` scatters the rest at random,
        # and those scattered fragments are exactly what this is here to place
        owned = {int(t) for c in placed for t in c}
        free_cell = np.array([c for c in range(N) if int(lay[c]) not in owned],
                             np.int64)
        free_tile = np.array(sorted(set(range(N)) - owned), np.int64)
        if len(free_cell):
            cc, tt = _assign(cell_colour, tile_colour, free_cell, free_tile)
            lay[cc] = tt
    if len(np.unique(lay)) != N:
        raise RuntimeError("layout is not a bijection")
    return lay, len(comps)


def texture(tiles, lay, r5, dev, do_level, nlm, bilateral):
    """The assembly, restored -- the only source of real content in the output.

    Every source was priced on the frontier of score against visible detail
    (M183): NLM is worth a great deal over R5 alone, an edge-preserving pass on
    top is worth another 0.003 to 0.004 at equal detail because it removes grain
    while leaving fragment boundaries and text, and restoring each fragment
    before assembly instead of the board after it is worse than both.
    """
    src = level(tiles, lay) if do_level else tiles
    tex = _net(r5, assemble(src, lay), dev)
    if nlm:
        tex = rank96.fixed_nlm(tex)
    if bilateral:
        tex = cv2.bilateralFilter(tex, 7, 40, 7)
    return tex.astype(np.float32)


def render(tiles, lay, field480, r5, dev, alpha, sigma, do_level, nlm,
           bilateral=True):
    tex = texture(tiles, lay, r5, dev, do_level, nlm, bilateral)
    hi = tex - cv2.GaussianBlur(tex, (0, 0), sigma)
    return np.rint(field480 + alpha * hi).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matcher", nargs="+",
                    default=["seam_embed_v3.pt", "seam_embed_local.pt",
                             "seam_embed_wide.pt"],
                    help="one or more matchers; several are combined by the "
                         "pessimistic minimum of M201, worth about +0.02 edge "
                         "precision when they are of comparable strength")
    ap.add_argument("--restorers", nargs="+",
                    default=["tile_restorer_mgc.pt", "tile_restorer.pt"])
    ap.add_argument("--field", default="coarse_field_v2.pt")
    ap.add_argument("--r5", default="E:/pazzle_work/pazzle_fixed_orientation_20260813/"
                                    "R5_restore_unet/r5_capacity_fp32.pt")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.5],
                    help="how much of the assembly's texture survives; several "
                         "values share one pass, since the layout is the "
                         "expensive part and re-rendering it is nearly free")
    ap.add_argument("--sigma", type=float, default=20.0,
                    help="cut-off of the high-pass, in pixels; one fragment wide, "
                         "which M182 measured as the optimum at equal detail")
    ap.add_argument("--no-field", action="store_true",
                    help="ship the restored assembly itself, with no colour "
                         "substitution at all. Costs about 0.05 of SSIM and "
                         "removes every trace of averaging: M210 measured that "
                         "at full texture the field is worthless at ANY cut-off "
                         "-- sigma 20 is worth +0.005 and every larger sigma is "
                         "worse than not using it -- so there is no middle "
                         "ground between this and the composed picture")
    ap.add_argument("--votes", type=int, default=10,
                    help="how many of the eighteen scorers must agree on an "
                         "edge; 0 falls back to the old two-view filter. TEN "
                         "after M348 swept the bar through this whole path on "
                         "24 boards: against the previous default of eight it "
                         "lifts adjacency 0.240 to 0.242, fragment placement "
                         "0.0012 to 0.0113 and SSIM 0.2626 to 0.2701 at full "
                         "texture; against thirteen it gains 0.024 of adjacency "
                         "for an SSIM wash")
    ap.add_argument("--orientations", type=int, default=2,
                    help="how many of the board's eight symmetries each matcher "
                         "sees; each one makes different heads judge the same "
                         "seam, at one forward pass (M236, M237)")
    ap.add_argument("--swap", type=float, default=0.0,
                    help="share of annealing proposals that SWAP two components "
                         "rather than move one. M361 measured why this is the "
                         "move that matters: of the thousands of single-component "
                         "proposals that fit, not one improved the score, so the "
                         "greedy initialisation is a deep local optimum for that "
                         "move class and the annealer was a no-op")
    ap.add_argument("--jump", type=float, default=1.0,
                    help="share of annealing proposals that TELEPORT a component "
                         "to a uniform random position; the rest displace it "
                         "locally. 1.0 is the historical behaviour, which M360 "
                         "measured as a no-op -- at 43 to 72 per cent occupancy "
                         "a teleport almost never fits, so the annealer never "
                         "improved on its greedy initialisation across 32 "
                         "boards and five seeds")
    ap.add_argument("--step", type=float, default=3.0,
                    help="standard deviation, in cells, of a local displacement")
    ap.add_argument("--seed", type=int, default=0,
                    help="the annealer's seed. M357 found per-board choice "
                         "between CONFIGURATIONS hopeless because board texture "
                         "swamps it; restarts of one configuration differ only "
                         "in the arrangement, so the comparison is clean")
    ap.add_argument("--dissolve", type=float, default=0.0,
                    help="do not place a component as a BLOCK unless it confirms "
                         "itself at least this much; its fragments go to the "
                         "fill instead, where colour decides. M358 measured why "
                         "this could pay: a component is placed as a unit, so a "
                         "large one in the wrong place costs more SSIM than the "
                         "same fragments scattered. 0 disables")
    ap.add_argument("--dissolve-min", type=int, default=6,
                    help="only components of at least this size are ever "
                         "dissolved; small ones are cheap to misplace")
    ap.add_argument("--order", choices=("margin", "votes", "votes_margin"),
                    default="margin",
                    help="what orders the edges as components are built. The "
                         "weight is an ORDERING and nothing else, and M270 "
                         "measured that ordering decides -- the same edges "
                         "shuffled take clean coverage from 0.931 to 0.268 -- "
                         "yet this has always been the minimum margin and was "
                         "never swept")
    ap.add_argument("--vote-target", type=int, default=0,
                    help="choose the vote bar PER BOARD so the harvest reaches "
                         "this many edges, instead of clearing a fixed one. "
                         "OFF by default: it does raise the assembly's "
                         "adjacency, 0.218 to 0.245, but M347 measured the "
                         "whole pipeline and the gain does not survive the "
                         "restorer and the field -- SSIM falls at every alpha, "
                         "0.2711 to 0.2635 at full texture. Kept because the "
                         "adjacency gain is real and reproduces, and because a "
                         "later renderer may convert it")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="the weakest agreeing scorer's margin must clear this; "
                         "raises precision where votes alone cannot (M224)")
    ap.add_argument("--frame", type=float, default=1.0,
                    help="weight of the border prior; the components are then "
                         "placed by annealing rather than by the greedy packer. "
                         "1.0 is the default because assembly is free: honest_v4 "
                         "at 0.2 scored 0.28207 on the platform and honest_v5 at "
                         "1.0 scored 0.28195, a difference of 0.0001, while "
                         "placing three times as many components correctly (M297)")
    ap.add_argument("--place", choices=("descent", "anneal"), default="descent",
                    help="how the components are positioned once the frame "
                         "prior is on. Descent is faster and better at the "
                         "RELATIVE placement M231 prices at +0.145 (M257)")
    ap.add_argument("--corroboration", type=float, default=0.0,
                    help="discount for seams whose implied component offset a "
                         "second, independent seam confirms (M258)")
    ap.add_argument("--border-net", default="border_net_v2.pt",
                    help="content border detector; blank to use the structural "
                         "one alone")
    ap.add_argument("--content", type=float, default=0.0,
                    help="weight of the content border detector relative to the "
                         "structural one. Default off: at 0.5 it lifts every "
                         "side's AUC and still costs 0.004 SSIM on 24 boards "
                         "(M256)")
    ap.add_argument("--fill", choices=("seam", "field"), default="field",
                    help="how the fragments no component owns are placed. FIELD "
                         "after M350: assigning them by colour against the "
                         "predicted field beats the seam fill by 0.0027 of SSIM "
                         "at full texture, reproduced on 24 boards and again on "
                         "48. It costs adjacency, 0.244 to 0.230, because a "
                         "colour fill does not preserve index adjacency, and "
                         "gains score anyway. M226 found every fill rule within "
                         "one ten-thousandth of RANDOM, but measured it with no "
                         "field at all, where the fill had to supply the colour "
                         "rather than place texture beneath one")
    ap.add_argument("--level", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--nlm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bilateral", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--test-dir", default=TEST_DIR)
    ap.add_argument("--out", default=str(Path(SUB_DIR) / "composed_v1"))
    ap.add_argument("--limit", type=int, default=0, help="smoke run; no ZIP")
    ap.add_argument("--validate", type=int, default=24)
    ap.add_argument("--dump-validate", default="",
                    help="save the validation renders and their per-board "
                         "metrics to this directory, so a truth-FREE judge of "
                         "the output can be scored against what happened")
    a = ap.parse_args()

    dev = "cuda"
    matcher = []
    for nm in a.matcher:
        mk = torch.load(Path(CKPT_DIR) / nm, map_location=dev, weights_only=False)
        ta = mk["args"]
        m = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                      ta.get("head", "global"),
                      predict=any(k.startswith("pred.") for k in mk["model"])).to(dev)
        m.load_state_dict(mk["model"])
        m.eval()
        matcher.append(m)
    restorers = []
    for n in a.restorers:
        ck = torch.load(Path(CKPT_DIR) / n, map_location=dev, weights_only=False)
        ar = ck.get("args", {})
        # `residual` and `ycc` add no weights but change the forward pass, so a
        # checkpoint trained with either and loaded without loads silently and
        # returns the wrong thing -- tile_restorer_mgc was trained with residual
        m = TileRestorer(ar.get("ch", 64), ar.get("blocks", 5),
                         ar.get("residual", False), False,
                         ar.get("ycc", False)).to(dev)
        m.load_state_dict(ck.get("model", ck))
        m.eval()
        restorers.append(m)
    field = load_field(Path(CKPT_DIR) / a.field, dev)
    r5 = load_unet(a.r5, dev)
    bnet = None
    if a.frame > 0 and a.border_net:
        from train_border import BorderNet
        bk = torch.load(Path(CKPT_DIR) / a.border_net, map_location=dev,
                        weights_only=False)
        bnet = BorderNet().to(dev)
        bnet.load_state_dict(bk["model"])
        bnet.eval()
    report = {k: getattr(a, k) for k in
              ("matcher", "field", "alpha", "sigma", "fill", "level", "nlm",
               "bilateral", "no_field", "votes", "margin",
               "orientations")}
    print(json.dumps(report), flush=True)

    def one(tiles):
        """One assembly, rendered at every requested alpha."""
        big, cells = predict_field(field, tiles, dev)
        lay, nc = solve_layout(matcher, restorers, tiles, dev, cells, a.fill,
                               a.votes, a.margin, a.orientations, a.frame,
                               20000, bnet, a.content, a.place,
                               a.corroboration, a.vote_target, a.order,
                               a.dissolve, a.dissolve_min, a.seed, a.jump,
                               a.step, a.swap)
        tex = texture(tiles, lay, r5, dev, a.level, a.nlm, a.bilateral)
        if a.no_field:
            out = np.rint(tex).clip(0, 255).astype(np.uint8)
            return {al: out for al in a.alpha}, lay, nc
        hi = tex - cv2.GaussianBlur(tex, (0, 0), a.sigma)
        return {al: np.rint(big + al * hi).clip(0, 255).astype(np.uint8)
                for al in a.alpha}, lay, nc

    if a.validate:
        from skimage.metrics import structural_similarity as ssim_fn
        blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
        names = [str(n) for n in blob["names"][-300:]]
        inv = blob["inv"][-300:]
        rows = {al: [] for al in a.alpha}
        common, per_board = [], []
        for k in range(min(a.validate, len(names))):
            tgt = load_rgb(Path(TRAIN_TGT) / names[k])
            tiles = to_frags(load_rgb(Path(TRAIN_INP) / names[k])).astype(
                np.float32)[inv[k].astype(np.int64)]
            imgs, lay, nc = one(tiles)
            flat = np.zeros_like(tgt)
            flat[:] = np.rint(tiles.reshape(-1, 3).mean(0)).clip(0, 255).astype(np.uint8)
            bd = np.asarray(lay).reshape(G, G)
            # adjacency is the honest measure of an assembly whose absolute
            # placement is near chance: M121 measured that at place_acc ~0 the
            # entire SSIM gain from assembling comes from adjacency
            adj = float((bd[:, 1:] == bd[:, :-1] + 1).sum()
                        + (bd[1:] == bd[:-1] + G).sum()) / (2 * G * (G - 1))
            common.append([float(ssim_fn(flat, tgt, channel_axis=2, data_range=255)),
                           float(np.mean(lay == np.arange(N))), nc, adj])
            for al, img in imgs.items():
                rows[al].append([float(ssim_fn(img, tgt, channel_axis=2,
                                               data_range=255)), detail(img)])
            if a.dump_validate:
                # the rendered board beside its truth-based metrics, so a
                # truth-FREE judge can be scored against what actually happened
                dd = Path(a.dump_validate)
                dd.mkdir(parents=True, exist_ok=True)
                for al, img in imgs.items():
                    cv2.imwrite(str(dd / f"{names[k]}_a{int(al*100):03d}.png"),
                                img[:, :, ::-1])
                per_board.append({
                    "name": names[k], "place": float(np.mean(lay == np.arange(N))),
                    "adjacency": adj,
                    "lay": [int(x) for x in np.asarray(lay).reshape(-1)],
                    "ssim": {f"{al:.2f}": float(ssim_fn(img, tgt, channel_axis=2,
                                                        data_range=255))
                             for al, img in imgs.items()}})
        c = np.mean(common, axis=0)
        report["validation"] = {
            "flat_fill": round(float(c[0]), 4),
            "place_acc": round(float(c[1]), 4),
            "adjacency": round(float(c[3]), 3),
            "components": round(float(c[2]), 1),
            "alpha": {f"{al:.2f}": {"ssim": round(float(np.mean(v, 0)[0]), 4),
                                    "vs_flat": round(float(np.mean(v, 0)[0] - c[0]), 4),
                                    "detail": round(float(np.mean(v, 0)[1]), 1)}
                      for al, v in rows.items()}}
        print(json.dumps(report["validation"], indent=1), flush=True)
        if a.dump_validate:
            (Path(a.dump_validate) / "per_board.json").write_text(
                json.dumps(per_board, indent=1), encoding="utf-8")

    out = Path(a.out)
    dirs = {al: out / f"a{int(round(al * 100)):03d}" for al in a.alpha}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in Path(a.test_dir).glob("*.png"))
    if a.limit:
        names = names[:a.limit]
    print(f"{len(names)} test boards -> {out}", flush=True)
    t0 = time.time()
    for i, nm in enumerate(names):
        tiles = to_frags(load_rgb(Path(a.test_dir) / nm)).astype(np.float32)
        for al, img in one(tiles)[0].items():
            cv2.imwrite(str(dirs[al] / nm), img[:, :, ::-1])
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(names)}  {el:.0f}s  eta "
                  f"{el / (i + 1) * (len(names) - i - 1):.0f}s", flush=True)
    report["boards"] = len(names)
    report["seconds"] = round(time.time() - t0, 1)
    if a.limit:
        print("smoke run: no ZIP written", flush=True)
    else:
        report["zips"] = {}
        for al, d in dirs.items():
            z = out / f"submission_composed_a{int(round(al * 100)):03d}.zip"
            report["zips"][f"{al:.2f}"] = {"zip": str(z),
                                           "sha256": deterministic_zip(d, names, z)}
            print(f"wrote {z}\nsha256 {report['zips'][f'{al:.2f}']['sha256']}",
                  flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
