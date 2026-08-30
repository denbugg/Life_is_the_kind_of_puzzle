"""Assemble all 576 tiles and restore the assembled image.

COMPLIANCE: the output is made only from rearranged input tiles followed by
restoration.  Pixel-field/high-pass composition has been removed from the
output path: no command-line option can enable pixel averaging or substitution.

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

Those measurements describe a rejected historical experiment.  They must not
be used for submission scoring: the only rendered image is now the restored
discrete assembly.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import infer_rank96 as rank96
from analytic_views import ANALYTIC_VIEWS, analytic_view  # noqa: F401
from coarse_field import CoarseField
from config import (CACHE_DIR, CKPT_DIR, GRID as G, SUB_DIR, TEST_DIR, TRAIN_INP,
                    TRAIN_TGT)
from infer_coarse_field import deterministic_zip, load_rgb
from infer_conformant import (agreed_edges, assemble, detail, load_unet, _net,
                              _restore_tiles)
from border_prior import border_prior, border_scores, content_scores
from harvest_votes import voted_edges, voted_pool
from place_search import anneal, corroborate, fill_rest, fill_seams, search
from edge_selector import selected_edges
from frame_classifier import frame_features, frame_unary
from level_seams import level
from deadness import tile_contrast
from chooser_edges import chooser_edges, load_chooser
from verify_edges import load_verifier, verify_harvest
from quad_rerank import quad_rerank
from merge_corroborated import merge_by_contact, merge_corroborated
from path_merge import merge_by_paths
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


def matrix_edges(CH, CV, how, keep=0):
    """Edges read straight off the fused cost matrix, with no harvest.

    `margin` is the PURE seed: each fragment's best partner, ranked by how far
    it leads the runner-up, cut to `keep`. M378 measured that purity comes from
    the margin -- and M416 that growth only pays from a pure seed, because the
    harvest's components are internally correct only a quarter of the time and
    merging compounds their errors.

    M395 made the count of CORRECT bonds the currency and M408 measured what
    each rule yields on the pipeline's own matrix. Taking each fragment's best
    partner and keeping the collisions beats every cleverer set, because
    `build_directed_components` already resolves collisions in score order --
    it walks edges in descending weight and drops whatever conflicts with the
    geometry it has committed to.
    """
    H, V = -np.asarray(CH, np.float64), -np.asarray(CV, np.float64)
    np.fill_diagonal(H, -1e9)
    np.fill_diagonal(V, -1e9)
    out = {}
    for M, off in ((H, (0, 1)), (V, (1, 0))):
        if how == "assignment":
            r, c = linear_sum_assignment(-M)
            for a, b in zip(r, c):
                out[(int(a), int(b), off)] = float(M[a, b])
            continue
        if how == "margin":
            part = np.partition(M, -2, axis=1)
            lead = part[:, -1] - part[:, -2]
            am = M.argmax(1)
            for i in range(N):
                out[(i, int(am[i]), off)] = float(lead[i])
            continue
        am = M.argmax(1)
        bm = M.argmax(0)
        for i in range(N):
            j = int(am[i])
            if how == "mutual" and int(bm[j]) != i:
                continue
            out[(i, j, off)] = float(M[i, j])
    if how == "margin" and keep:
        out = dict(sorted(out.items(), key=lambda kv: -kv[1])[:keep])
    return out


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
                 dissolve=0.0, dissolve_min=6, seed=0, jump=1.0, step=3.0,
                 swap=0.0, depth=1, weighted=False, analytic=(),
                 merge_support=0, selector=None, sel_depth=2, sel_volume=430,
                 sel_decode="greedy", edges_from="votes",
                 merge_contact=0, merge_paths=0, dead_fill=0.0,
                 quad=0.0, seed_cap=0, chooser=None, place_top=0,
                 edge_floor=None, chooser_floor=None, fill_rounds=1,
                 verifier=None,
                 frame_classifier=None, learned_frame_weight=0.0):
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
        if quad > 0:
            CH, CV = quad_rerank(CH, CV, weight=quad)
        views = ([tiles] + [_restore_tiles(m, tiles, dev) for m in restorers]
                 + [analytic_view(n, tiles) for n in analytic])
        if chooser is not None:
            agreed = chooser_edges(chooser, tiles, CH, CV, dev, sel_volume,
                                   chooser_floor)
        elif edges_from in ("top1", "mutual", "assignment", "margin"):
            agreed = matrix_edges(CH, CV, edges_from, sel_volume)
        elif selector is not None:
            agreed = selected_edges(selector, matcher, views, dev, orientations,
                                    sel_depth, sel_volume, sel_decode)
        else:
            agreed = voted_edges(matcher, views, dev, votes,
                                 orientations=orientations, margin=margin,
                                 target=vote_target, order=order, depth=depth,
                                 weighted=weighted, quad=quad)
        if frame > 0 and (corroboration > 0 or merge_support > 0):
            pool = voted_pool(matcher, views, dev, orientations)
    else:
        CH, CV, agreed = agreed_edges(matcher, restorers, tiles, dev)
    if verifier is not None and votes:
        # M456 makes precision at the harvest volume the only figure that
        # decides anything, so the harvest is RE-ORDERED by a scorer trained
        # for exactly that. Nothing is dropped; the weight is an ordering
        # (M270) and `build_directed_components` spends it on conflicts.
        agreed = verify_harvest(verifier, tiles, CH, CV, agreed, dev)
    if edge_floor is not None and votes:
        # M452: keep every harvested edge whose fused score clears a fixed
        # THRESHOLD, so the volume is chosen per board instead of being a fixed
        # count. Boards are nothing alike -- placement is heavy-tailed and the
        # best board carries eighteen times the mean -- so a fixed count makes a
        # weak board scrape the bottom of its evidence and stops a strong board
        # short. At a MATCHED average count of 179 edges the adaptive rule holds
        # a connected group of 71.5 against the fixed rule's 55.4.
        agreed = {e: w for e, w in agreed.items()
                  if -(CH if e[2] == (0, 1) else CV)[e[0], e[1]] >= edge_floor}
    if order == "raw" and votes and verifier is None:
        # the verifier's logit IS the ordering, so the raw re-weighting must not
        # run after it -- it did, and overwrote the verifier's output entirely,
        # which showed as all 24 boards identical to the fourth decimal.
        # M450: order the harvested edges by the FUSED SCORE itself rather than
        # by any margin. The margin is a per-fragment quantity and takes one
        # edge from every fragment whether or not it has anything to say; the
        # raw tail takes the board's best edges wherever they are. On the stand
        # that is 157.2 fragments in correct islands against 143.4 and a
        # connected group of 56.5 against 50.4. M382 is the caution: a purer
        # ordering did not reach the pipeline because the conflict rule dropped
        # the disputed edges anyway.
        agreed = {e: float(-(CH if e[2] == (0, 1) else CV)[e[0], e[1]])
                  for e in agreed}
    comps = build_directed_components(
        [i for (i, j, o) in agreed],
        [DIR_RIGHT if o == (0, 1) else DIR_DOWN for (i, j, o) in agreed],
        [j for (i, j, o) in agreed], list(agreed.values()),
        max_edges=len(agreed), cap=seed_cap)
    if merge_support > 0 and pool:
        comps = merge_corroborated(comps, pool, merge_support)
    if merge_contact > 0:
        comps = merge_by_contact(comps, CH, CV, rounds=merge_contact)
    if merge_paths > 0:
        comps = merge_by_paths(comps, CH, CV, min_paths=merge_paths)
    placed = comps
    if frame > 0:
        prior = border_prior(border_scores(matcher, tiles, dev),
                             content_scores(border_net, tiles, dev)
                             if border_net is not None else None, content)
        if frame_classifier is not None:
            # Bag-relative missing-neighbour probabilities.  Unlike the old
            # content prior, this is trained on seam distributions of entire
            # bags and therefore asks whether a plausible neighbour is absent.
            stats = np.concatenate([tiles.mean((1, 2)), tiles.std((1, 2))], 1) / 255.0
            feat = frame_features(-CH, -CV, stats)
            probability = frame_classifier.predict_proba(feat)[:, 1]
            learned = frame_unary(probability, G).T.reshape(N, G, G)
            # Residual evidence, not a replacement.  Scale matching makes the
            # coefficient portable and preserves the calibrated legacy prior.
            learned = learned - np.mean(learned)
            learned *= np.std(prior) / max(float(np.std(learned)), 1e-8)
            prior = prior + float(learned_frame_weight) * learned
        placed = [c for c in comps if len(c) > 1]
        if place_top:
            # M451: place only the biggest few as BLOCKS and send the rest to
            # the fill. M321 measured that placement follows the largest
            # coherent block alone -- blocks of 19.6, 27.2 and 37.7 fragments
            # all pay about 0.002 where one of 194 pays 0.40 -- so the smaller
            # components buy nothing while giving the search sixty-odd more
            # degrees of freedom, and M449 measured what freedom does to an
            # optimum: it drifts to the extreme of the noise.
            placed = sorted(placed, key=len, reverse=True)[:place_top]
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
        lay = fill_seams(board, H, V, contrast=tile_contrast(tiles),
                         dead_q=dead_fill, rounds=fill_rounds)
    else:
        lay = np.asarray(solve_components_from_scores(
            (-CH).astype(np.float32), (-CV).astype(np.float32), comps,
            repair_passes=0, restarts=1)[0], np.int64)
    if fill == "field":
        tile_colour = (np.mean([_restore_tiles(m, tiles, dev).mean((1, 2))
                                for m in restorers], axis=0) if restorers
                       else tiles.mean((1, 2)))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matcher", nargs="+",
                    default=["seam_embed_v3.pt", "seam_embed_local.pt"],
                    help="one or more matchers; several are combined by the "
                         "pessimistic minimum of M201. The default exactly "
                         "matches the ensemble used to train the top-5 chooser")
    ap.add_argument("--restorers", nargs="+", default=["none"],
                    help="restorer checkpoints, one extra VIEW of the board "
                         "each; pass `none` for the raw view alone")
    ap.add_argument("--field", default="coarse_field_v2.pt")
    ap.add_argument("--r5", default="E:/pazzle_work/pazzle_fixed_orientation_20260813/"
                                    "R5_restore_unet/r5_capacity_fp32.pt")
    ap.add_argument("--no-field", action="store_true", default=True,
                    help="deprecated compatibility flag; strict no-field output "
                         "is unconditional")
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
    ap.add_argument("--analytic", nargs="*", default=["median", "bilateral"],
                    choices=sorted(ANALYTIC_VIEWS),
                    help="analytic filters to add as extra VIEWS. M366 measured "
                         "these at solo precision 0.352, 0.346 and 0.342 for "
                         "median, non-local means and the bilateral, against "
                         "0.396 for the raw fragments and 0.148 to 0.280 for "
                         "every learned restorer")
    ap.add_argument("--weighted", action="store_true",
                    help="weight each scorer's vote by how far it agrees with "
                         "the consensus of the others, estimated without labels. "
                         "M365 measured the views at solo precisions from 0.483 "
                         "down to 0.173 while every one casts an equal vote")
    ap.add_argument("--depth", type=int, default=1,
                    help="how deep into each scorer's candidate list an edge may "
                         "sit and still be voted on. 1 is mutual-best, which is "
                         "all the shipping harvest has ever used; M253 measured "
                         "the top-1 union holding 525 correct edges of 1104 "
                         "against percolation at 552, and the top-2 union 599")
    ap.add_argument("--anneal-iters", type=int, default=20000,
                    help="annealing iterations per board. M360 measured the "
                         "phase as a no-op -- five seeds return bitwise "
                         "identical layouts because the greedy initialisation "
                         "is a local optimum no single-component move can leave "
                         "(M361) -- so this is pure cost unless --swap is on")
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
    ap.add_argument("--order",
                    choices=("margin", "max_margin", "votes",
                             "votes_margin", "raw"),
                    default="raw",
                    help="what orders the edges as components are built. "
                         "DEFAULT raw (M450): the fused score itself, "
                         "rather than any margin. A margin is a per-fragment "
                         "quantity and takes one edge from every fragment "
                         "whether or not it has anything to say; the raw "
                         "tail takes the board's best edges wherever they "
                         "are, and M449 measured that axis as strictly "
                         "monotone in P(adjacent) to the very top -- the "
                         "best 0.02% of pairs are 98.5% right. On 24 PAIRED "
                         "boards it moves ADJACENCY 0.2628 to 0.2714, +3.9 "
                         "sigma with 20 boards up and 3 down, the clearest "
                         "pipeline signal measured in this run; SSIM leans "
                         "up at 0.4 sigma and placement leans down at 0.9, "
                         "which is noise. The older orderings all read a "
                         "MARGIN and differ only in which scorer's. "
                         "ORIGINAL NOTE: The "
                         "weight is an ORDERING and nothing else, and M270 "
                         "measured that ordering decides -- the same edges "
                         "shuffled take clean coverage from 0.931 to 0.268. "
                         "M362 swept the three that existed and found them "
                         "flat, but all three read the MINIMUM margin, the "
                         "least convinced scorer. `max_margin` reads the most "
                         "convinced one instead, which M382 measured as a purer "
                         "prefix at every depth: over 24 boards the strongest "
                         "half of the harvest is 0.883 against 0.849 and the "
                         "strongest three quarters 0.767 against 0.737")
    ap.add_argument("--edges", choices=("votes", "top1", "mutual", "assignment", "margin"),
                    default="votes",
                    help="what becomes an edge. VOTES is the harvest. The other "
                         "three read the fused cost matrix directly, and M408 "
                         "measured them there: top-1 with collisions kept gives "
                         "348.4 correct bonds a board and a coherent block of "
                         "56.8, mutual best 315.7 and 56.6, the assignment "
                         "332.4 and 58.1, against the harvest's 254 and 33.7")
    ap.add_argument("--selector", default="",
                    help="a trained LightGBM edge selector, replacing the vote "
                         "threshold. M317 closed per-edge selection on the "
                         "MUTUAL-BEST pool, where every candidate is rank one "
                         "from both ends and rank carries no information; M377 "
                         "widened the pool and made it a variable")
    ap.add_argument("--sel-depth", type=int, default=2,
                    help="how many candidates per fragment the selector sees")
    ap.add_argument("--sel-volume", type=int, default=430,
                    help="how many edges the selector is allowed to keep; "
                         "M316 set the target at about 430")
    ap.add_argument("--sel-decode", choices=("greedy", "assignment"),
                    default="greedy",
                    help="how the selector's scores become edges. GREEDY walks "
                         "the ranked list and takes an edge when neither end is "
                         "spoken for, which lets a wrong edge block a true one "
                         "that needed the same fragment. ASSIGNMENT solves each "
                         "direction as a 576x576 matching instead, and M396 "
                         "measured 291.7 correct bonds at 600 edges against "
                         "greedy's 267.0 at 430, with a clean block of 45.6 "
                         "against 41.5. With this, --sel-volume is per direction")
    ap.add_argument("--chooser", default="choose5_full_none0_best_bonds.pt",
                    help="a trained five-candidate chooser (M412, M438). The "
                         "default is the full 6,700-board best-bonds checkpoint. It "
                         "picks one of each fragment's five best candidates or "
                         "abstains, and the picks become the harvest. M409 "
                         "sizes the door: the top-5 shortlist holds 543 correct "
                         "bonds once the square is centred, against the "
                         "percolation knee at 450 to 500 and a top-1 harvest of "
                         "about 348, so this is the only lever measured that "
                         "could deliver a multiplier rather than a percentage")
    ap.add_argument("--chooser-floor", type=float, default=None,
                    help="optional absolute confidence floor for the chooser; "
                         "selects a board-dependent edge count, while "
                         "--sel-volume remains a safety cap")
    ap.add_argument("--verifier", default="verify_hinge.pt",
                    help="a trained seam verifier (M459). It re-scores every "
                         "harvested edge with a JOINT model of the two sides of "
                         "the join, where every scorer in the roster is a "
                         "bi-encoder taking a dot product of pooled "
                         "descriptors, and it is trained for PRECISION AT THE "
                         "HARVEST VOLUME rather than by retrieval. M456 is why: "
                         "the connected block runs 350 at edge precision 1.00, "
                         "186 at 0.99 and 18 at the 0.746 we deliver")
    ap.add_argument("--fill-rounds", type=int, default=1,
                    help="how many times to re-solve the leftover assignment. "
                         "The components cover about two hundred fragments, so "
                         "on the first pass most free cells have NO placed "
                         "neighbour and their cost row is flat; after one pass "
                         "every cell has four, and the assignment can be "
                         "re-solved against them. The fill decides about 330 "
                         "of the 576 cells (M226)")
    ap.add_argument("--edge-floor", type=float, default=None,
                    help="keep every harvested edge whose FUSED SCORE clears "
                         "this threshold, letting the volume differ per board "
                         "instead of being the fixed count `--vote-target` "
                         "sets. M449 calibrated that axis -- the best 0.02%% of "
                         "pairs are 98.5%% right -- and M452 measured the rule "
                         "against a fixed count at MATCHED average volume, 179 "
                         "edges a board either way: connected group 71.5 "
                         "against 55.4, so what pays is the adaptivity and not "
                         "the extra edges. Use with a generous --vote-target so "
                         "the floor, not the count, decides")
    ap.add_argument("--place-top", type=int, default=0,
                    help="place only this many of the largest components as "
                         "blocks and send every other one to the fill; 0 keeps "
                         "them all. M321 measured that placement follows the "
                         "largest coherent block ALONE, so the rest buy "
                         "nothing and cost the search sixty degrees of freedom")
    ap.add_argument("--seed-cap", type=int, default=0,
                    help="refuse any harvested edge that would grow a "
                         "component past this many fragments; 0 is off. M444 "
                         "measured what the cap is for: the extra edges of a "
                         "larger harvest WELD islands together and one wrong "
                         "edge makes the whole grown island internally wrong, "
                         "so at 500 edges the uncapped seed keeps 7 correct "
                         "islands of 40 where a cap of two keeps 77. On the "
                         "stand the largest TRULY CONNECTED group of correct "
                         "islands goes 50.4 to 127.8 and the island "
                         "programme's ceiling 0.2885 to 0.3268")
    ap.add_argument("--quad", type=float, default=0.4,
                    help="re-rank every scorer's shortlist by the best 2x2 "
                         "SQUARE a pair can stand in, at this weight; 0 is "
                         "off. If j sits to the right of i then whatever "
                         "sits below i must sit to the left of whatever "
                         "sits below j, and the pairwise score does not "
                         "contain that. M93's cycle consistency, already in "
                         "the cost path, is the linear first-order version; "
                         "M92 and M404 tried the same evidence as a HARD "
                         "filter and as a REPLACEMENT and both collapsed, "
                         "while as a weighted term it pays. The term must "
                         "be CENTRED per row (M441): these are "
                         "log-assignments, so an uncentred bonus sinks "
                         "shortlist members below candidates outside the "
                         "shortlist and corrupts it instead of re-ordering "
                         "it, which cost 24 correct bonds a board at depth "
                         "five and reached nothing downstream. Centred, on "
                         "24 PAIRED boards: recall@1 +0.0121 at 8.4 sigma, "
                         "recall@5 +0.0086 at 6.4 sigma, and through the "
                         "pipeline ADJACENCY 0.2575 to 0.2628 at 2.1 sigma "
                         "with 15 boards up and 7 down. Placement and SSIM "
                         "lean up and are unproven, which M402 predicts "
                         "below the percolation knee. Applied to every "
                         "scorer BEFORE it votes: on the fused matrix alone "
                         "it leaves the component count untouched, because "
                         "the harvest never reads that matrix")
    ap.add_argument("--dead-fill", type=float, default=0.0,
                    help="hold back this share of the leftover fragments -- the "
                         "ones with the least information left in them -- and "
                         "let the textured fragments take their cells first. "
                         "M69 measured that misplacing a flat fragment costs "
                         "2.6x less SSIM than misplacing a textured one, and "
                         "the fill decides about 330 of the 576 cells, so which "
                         "fragment is wrong where is not a free choice")
    ap.add_argument("--merge-paths", type=int, default=0,
                    help="join islands that this many independent, "
                         "VERTEX-DISJOINT paths through free fragments agree "
                         "about, with M248's geometric veto and confidence "
                         "filter. Measured precision 0.034 at one path, 0.206 "
                         "at two and 0.400 at three -- the same corroboration "
                         "M248 found on direct contacts, reaching pairs that "
                         "do not touch. Volume is the wall: 2.5 offers a board "
                         "at three paths, and the global reconciliation that "
                         "should filter them cannot, because the hypotheses "
                         "form a forest and consistency only bites on cycles")
    ap.add_argument("--merge-contact", type=int, default=0,
                    help="after the components are built, join the pair whose "
                         "contact carries the single BEST seam, this many "
                         "times. M415 measured the statistic: ranked by the "
                         "mean -- the rule the island branch stood on since "
                         "August -- growth reaches a clean block of 30.9, and "
                         "by the MAXIMUM it reaches 38.9 with more true "
                         "adjacencies. M409 is the reason this needs the "
                         "pipeline to judge it: below the knee a bigger block "
                         "bought with precision has always lost")
    ap.add_argument("--merge-support", type=int, default=0,
                    help="join two components when this many independent tile "
                         "pairs from the FULL pool imply the same relative "
                         "offset between them. `place_search.corroborate` reads "
                         "the same signal and only discounts the seams, leaving "
                         "the components apart, and M321 measured that "
                         "separateness is what costs -- placement follows the "
                         "largest coherent block alone. M381 measured the merge "
                         "on 24 boards: the block goes 33.7 to 42.9 and true "
                         "adjacencies 254.2 to 271.5 at a support of two")
    ap.add_argument("--vote-target", type=int, default=350,
                    help="choose the vote bar PER BOARD so the harvest reaches "
                         "this many edges, instead of clearing a fixed one. "
                         "OFF by default: it does raise the assembly's "
                         "adjacency, 0.218 to 0.245, but M347 measured the "
                         "whole historical pipeline. Its old pixel-composition "
                         "SSIM measurements are invalid for submission. Kept "
                         "because the "
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
    ap.add_argument("--frame-classifier", default="",
                    help="pickle from train_frame_classifier.py; replaces the "
                         "legacy border prior with learned bag-relative "
                         "missing-neighbour probabilities")
    ap.add_argument("--learned-frame-weight", type=float, default=0.0,
                    help="scale-matched residual weight of --frame-classifier; "
                         "0 preserves the legacy prior exactly")
    ap.add_argument("--content", type=float, default=0.0,
                    help="weight of the content border detector relative to the "
                         "structural one. Default off: at 0.5 it lifts every "
                         "side's AUC and still costs 0.004 SSIM on 24 boards "
                         "(M256)")
    ap.add_argument("--fill", choices=("seam", "field"), default="seam",
                    help="how the fragments no component owns are placed, which "
                         "is most of the board. SEAM after M375. M350 shipped "
                         "the colour fill for 0.0027 of SSIM while recording "
                         "that it cost adjacency, 0.244 to 0.230 -- a trade the "
                         "project rule forbids, since the assembly metric is "
                         "what ranks arms and SSIM is the control. On the "
                         "analytic roster of M371 there is no trade left to "
                         "make: over 48 boards the seam fill gives adjacency "
                         "0.256 against 0.242, identical placement (0.0107 "
                         "against 0.0106) and 0.0013 less SSIM, and it wins "
                         "adjacency on every one of the first six boards "
                         "individually. M226 found every fill rule within one "
                         "ten-thousandth of RANDOM, but measured it with no "
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
    ap.add_argument("--validate-offset", type=int, default=0,
                    help="start inside the frozen final-300 validation split")
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
    for n in ([] if a.restorers == ["none"] else a.restorers):
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
    # A field may supply unary costs to the discrete placement solver, but its
    # pixels never enter the rendered image.
    field = (load_field(Path(CKPT_DIR) / a.field, dev)
             if a.fill == "field" else None)
    r5 = load_unet(a.r5, dev)
    booster = None
    if a.selector:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(Path(CKPT_DIR) / a.selector))
    bnet = None
    if a.frame > 0 and a.border_net:
        from train_border import BorderNet
        bk = torch.load(Path(CKPT_DIR) / a.border_net, map_location=dev,
                        weights_only=False)
        bnet = BorderNet().to(dev)
        bnet.load_state_dict(bk["model"])
        bnet.eval()
    report = {k: getattr(a, k) for k in
              ("matcher", "field", "fill", "level", "nlm",
               "bilateral", "no_field", "votes", "margin", "verifier",
               "orientations", "chooser", "chooser_floor", "sel_volume",
               "frame", "validate_offset")}
    print(json.dumps(report), flush=True)

    # loaded once, not once a board
    chooser_path = Path(a.chooser)
    if a.chooser and not chooser_path.is_file():
        chooser_path = Path(CKPT_DIR) / chooser_path
    booster_chooser = (load_chooser(chooser_path, dev)
                       if a.chooser else None)
    verifier_path = Path(a.verifier)
    if a.verifier and not verifier_path.is_file():
        verifier_path = Path(CKPT_DIR) / verifier_path
    seam_verifier = (load_verifier(verifier_path, dev)
                     if a.verifier else None)
    learned_frame = None
    if a.frame_classifier:
        with open(a.frame_classifier, "rb") as f:
            learned_frame = pickle.load(f)["model"]

    def one(tiles):
        """One discrete assembly followed by restoration."""
        if field is None:
            cells = np.zeros((N, 3), np.float32)
        else:
            _, cells = predict_field(field, tiles, dev)
        lay, nc = solve_layout(matcher, restorers, tiles, dev, cells, a.fill,
                               a.votes, a.margin, a.orientations, a.frame,
                               a.anneal_iters, bnet, a.content, a.place,
                               a.corroboration, a.vote_target, a.order,
                               a.dissolve, a.dissolve_min, a.seed, a.jump,
                               a.step, a.swap, a.depth, a.weighted,
                               a.analytic, a.merge_support, booster,
                               a.sel_depth, a.sel_volume, a.sel_decode,
                               a.edges, a.merge_contact, a.merge_paths,
                               a.dead_fill, a.quad, a.seed_cap,
                               booster_chooser, a.place_top,
                               a.edge_floor, a.chooser_floor, a.fill_rounds,
                               seam_verifier, learned_frame,
                               a.learned_frame_weight)
        tex = texture(tiles, lay, r5, dev, a.level, a.nlm, a.bilateral)
        out = np.rint(tex).clip(0, 255).astype(np.uint8)
        return out, lay, nc

    if a.validate:
        from skimage.metrics import structural_similarity as ssim_fn
        blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
        names = [str(n) for n in blob["names"][-300:]]
        inv = blob["inv"][-300:]
        restored = []
        common, per_board = [], []
        start = max(0, a.validate_offset)
        stop = min(start + a.validate, len(names))
        if start >= stop:
            raise ValueError("validation offset/count selects no boards")
        for k in range(start, stop):
            tgt = load_rgb(Path(TRAIN_TGT) / names[k])
            tiles = to_frags(load_rgb(Path(TRAIN_INP) / names[k])).astype(
                np.float32)[inv[k].astype(np.int64)]
            img, lay, nc = one(tiles)
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
            restored.append([float(ssim_fn(img, tgt, channel_axis=2,
                                            data_range=255)), detail(img)])
            if a.dump_validate:
                # the rendered board beside its truth-based metrics, so a
                # truth-FREE judge can be scored against what actually happened
                dd = Path(a.dump_validate)
                dd.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dd / names[k]), img[:, :, ::-1])
                per_board.append({
                    "name": names[k], "place": float(np.mean(lay == np.arange(N))),
                    "adjacency": adj,
                    "lay": [int(x) for x in np.asarray(lay).reshape(-1)],
                    "ssim": float(ssim_fn(img, tgt, channel_axis=2,
                                           data_range=255))})
        c = np.mean(common, axis=0)
        r = np.mean(restored, axis=0)
        report["validation"] = {
            "flat_fill": round(float(c[0]), 4),
            "place_acc": round(float(c[1]), 4),
            "adjacency": round(float(c[3]), 3),
            "components": round(float(c[2]), 1),
            "restored": {"ssim": round(float(r[0]), 4),
                         "vs_flat": round(float(r[0] - c[0]), 4),
                         "detail": round(float(r[1]), 1)}}
        print(json.dumps(report["validation"], indent=1), flush=True)
        if a.dump_validate:
            (Path(a.dump_validate) / "per_board.json").write_text(
                json.dumps(per_board, indent=1), encoding="utf-8")

    out = Path(a.out)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in Path(a.test_dir).glob("*.png"))
    if a.limit:
        names = names[:a.limit]
    print(f"{len(names)} test boards -> {out}", flush=True)
    t0 = time.time()
    for i, nm in enumerate(names):
        tiles = to_frags(load_rgb(Path(a.test_dir) / nm)).astype(np.float32)
        img = one(tiles)[0]
        cv2.imwrite(str(images_dir / nm), img[:, :, ::-1])
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(names)}  {el:.0f}s  eta "
                  f"{el / (i + 1) * (len(names) - i - 1):.0f}s", flush=True)
    report["boards"] = len(names)
    report["seconds"] = round(time.time() - t0, 1)
    if a.limit:
        print("smoke run: no ZIP written", flush=True)
    else:
        z = out / "submission_assembly_restored.zip"
        report["zip"] = {"path": str(z),
                         "sha256": deterministic_zip(images_dir, names, z)}
        print(f"wrote {z}\nsha256 {report['zip']['sha256']}", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
