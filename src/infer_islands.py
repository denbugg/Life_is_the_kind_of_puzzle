"""Build a submission that shows the picture we actually recovered.

Why this arm exists
-------------------
The coarse field alone scores best of anything we can ship (+0.016 to +0.024
over the flat fill) and looks like a smooth blob: its detail, the spread
surviving a 12 px high-pass, is 0.3 where a real photograph scores 35.  It is
not a fill -- it changes with the board and passes the acceptance test by a wide
margin -- but that distinction does not survive a human looking at it.

This arm pastes the recovered material on top of that base.  Loop closure
harvests 2x2 blocks at precision 0.878 from tile-level edges running at 0.438,
a margin rule grows them into islands averaging 5.5 tiles, and about 16 of them
per board are internally perfect.  Placed where our colour field puts them the
result scores about 0.32 against the field's 0.36, and carries detail 19.9
instead of 0.3.

That trade is the point.  It costs 0.045 SSIM and buys an output that is visibly
an attempt at the task, while still standing about 0.10 above the deployed
0.23748 submission.

Placement is the honest weak point, measured rather than hidden: a 2x2 block
cannot be placed against any field we have (M149), and the islands here are
small.  When placement is solved these same islands are worth +0.095 rather than
-0.025 (M154), which is why the arm is built now and not after.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from coarse_field import CoarseField, render
from config import CACHE_DIR, CKPT_DIR, FS, GRID as G, SUB_DIR, TEST_DIR, TRAIN_INP, TRAIN_TGT
from infer_coarse_field import deterministic_zip, load_rgb
from restore_tile import to_frags
from seam_cost import costs_from_model
from joint_cost import costs_from_joint, load_joint
from seam_embed import SeamEmbed
from solve_agglom import grow, islands_from_edges, seed_quads


def field_place(isl, mu, fgrid):
    """Position the island by sliding its mean colours over the field.

    The per-tile photometric bias is unknown and uninformative, so the offset is
    removed and only the SHAPE of the island's colour map decides.
    """
    kh, kw = isl.height, isl.width
    a = np.zeros((kh, kw, 3), np.float32)
    m = np.zeros((kh, kw, 1), np.float32)
    for (r, c), t in isl.cells.items():
        a[r, c] = mu[t]
        m[r, c] = 1.0
    scores = []
    for r in range(G - kh + 1):
        for c in range(G - kw + 1):
            d = (a - fgrid[r:r + kh, c:c + kw]) * m
            d = d - (d.sum((0, 1)) / m.sum())[None, None, :] * m
            scores.append((float((d ** 2).sum()), (r, c)))
    scores.sort()
    best = scores[0]
    # confidence is the runner-up ratio, and it is not decoration: M148 measured
    # that a partial answer needs placement precision at or above 0.9 and that
    # at 0.7 it scores WORSE than pasting nothing, so an island the field cannot
    # place decisively is better left out
    runner = next((v for v, p in scores[1:]
                   if abs(p[0] - best[1][0]) > 1 or abs(p[1] - best[1][1]) > 1),
                  best[0] * 1e9)
    return best[1], runner / max(best[0], 1e-9)


def _mutual(C):
    D = C.copy()
    np.fill_diagonal(D, np.inf)
    f, b = D.argmin(1), D.argmin(0)
    part = np.partition(D, 1, axis=1)
    return {i: (int(f[i]), float(part[i, 1] / max(part[i, 0], 1e-9)))
            for i in range(C.shape[0]) if b[int(f[i])] == i}


def agreed_edges(RH, RV, JH, JV):
    """Edges two independently trained matchers both call a mutual best match.

    M158: agreement runs at precision 0.57 where either model alone runs at
    0.45, and its top half by margin at 0.84.  Blending their scores instead is
    worth nothing (M157) -- it is the coincidence of two opinions that carries
    the information, exactly as with loop closure.
    """
    out = []
    for RC, JC, off in ((RH, JH, (0, 1)), (RV, JV, (1, 0))):
        r, j = _mutual(RC), _mutual(JC)
        for i, (t, m) in r.items():
            if i in j and j[i][0] == t:
                out.append((i, t, off, m))
    out.sort(key=lambda e: -e[3])
    return out


def paste(base, islands, positions, tiles):
    out = base.copy()
    for isl, pos in zip(islands, positions):
        for (r, c), t in isl.cells.items():
            r1, c1 = int(pos[0]) + r, int(pos[1]) + c
            if 0 <= r1 < G and 0 <= c1 < G:
                out[r1 * FS:(r1 + 1) * FS, c1 * FS:(c1 + 1) * FS] = tiles[t]
    return out


def build_one(tiles, matcher, field_model, dev, margin, rounds, min_tiles,
              joint=None, jfrozen=None, frac=0.40, place_conf=1.0):
    """One board: field base, harvested islands pasted where the field wants."""
    mu = tiles.mean((1, 2))
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2)[None].to(dev)
        base = render(field_model(x))[0].permute(1, 2, 0).cpu().numpy() * 255.0
        fgrid = render(field_model(x), size=G)[0].permute(1, 2, 0).cpu().numpy() * 255.0
    CH, CV = costs_from_model(matcher, tiles)
    if joint is None:
        islands = seed_quads(CH, CV, mutual=True, keep_fraction=1.0)
    else:
        JH, JV = costs_from_joint(joint, jfrozen, tiles, blend=0.7, rounds=3)
        ed = agreed_edges(CH, CV, JH, JV)
        islands = islands_from_edges(ed[: max(1, int(round(frac * len(ed))))],
                                     tiles.shape[0])
    islands = [i for i in grow(islands, CH, CV, rounds=rounds, topk=5, margin=margin)
               if len(i) >= min_tiles]
    placed, pos = [], []
    for i in islands:
        p, conf = field_place(i, mu, fgrid)
        if conf >= place_conf:
            placed.append(i)
            pos.append(p)
    img = paste(base, placed, pos, tiles)
    return np.rint(img).clip(0, 255).astype(np.uint8), len(placed)


def detail(img):
    f = img.astype(np.float32)
    return float((f - cv2.GaussianBlur(f, (0, 0), 12.0)).std())


def validate(matcher, field_model, dev, boards, margin, rounds, min_tiles,
             joint=None, jfrozen=None, frac=0.40, place_conf=1.0):
    """Score both arms on held-out TRAIN boards before touching the test set."""
    from skimage.metrics import structural_similarity as ssim_fn

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = [str(n) for n in blob["names"][-300:]]
    inv = blob["inv"][-300:]
    rows = []
    for k in range(min(boards, len(names))):
        tgt = load_rgb(Path(TRAIN_TGT) / names[k])
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / names[k])).astype(np.float32)[
            inv[k].astype(np.int64)]
        flat = np.zeros_like(tgt)
        flat[:] = np.rint(tiles.reshape(-1, 3).mean(0)).clip(0, 255).astype(np.uint8)
        fs = float(ssim_fn(flat, tgt, channel_axis=2, data_range=255))
        img, n_isl = build_one(tiles, matcher, field_model, dev, margin, rounds,
                               min_tiles, joint, jfrozen, frac, place_conf)
        with torch.no_grad():
            x = torch.from_numpy(tiles).permute(0, 3, 1, 2)[None].to(dev)
            bare = np.rint(render(field_model(x))[0].permute(1, 2, 0).cpu().numpy()
                           * 255.0).clip(0, 255).astype(np.uint8)
        rows.append([float(ssim_fn(img, tgt, channel_axis=2, data_range=255)),
                     detail(img),
                     float(ssim_fn(bare, tgt, channel_axis=2, data_range=255)),
                     detail(bare), fs, n_isl])
    m = np.mean(rows, axis=0)
    return {"islands_ssim": round(m[0], 4), "islands_detail": round(m[1], 1),
            "field_ssim": round(m[2], 4), "field_detail": round(m[3], 1),
            "flat_ssim": round(m[4], 4), "islands_per_board": round(m[5], 1),
            "islands_gain": round(m[0] - m[4], 4),
            "field_gain": round(m[2] - m[4], 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matcher", default="seam_embed_v1.pt")
    ap.add_argument("--field", default="coarse_field_n8.pt")
    ap.add_argument("--margin", type=float, default=1.20)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--min-tiles", type=int, default=4)
    ap.add_argument("--joint", default="seam_joint_unfrozen.pt",
                    help="second matcher for the agreement construction; "
                         "empty falls back to loop closure")
    ap.add_argument("--place-conf", type=float, default=1.0,
                    help="paste an island only if the runner-up position costs "
                         "at least this many times the chosen one")
    ap.add_argument("--frac", type=float, default=0.40,
                    help="fraction of the agreed edge set to trust")
    ap.add_argument("--test-dir", default=TEST_DIR)
    ap.add_argument("--out", default=str(Path(SUB_DIR) / "islands_v1"))
    ap.add_argument("--limit", type=int, default=0, help="smoke run; no ZIP")
    ap.add_argument("--validate", type=int, default=20,
                    help="held-out boards to score first; 0 skips")
    a = ap.parse_args()

    dev = "cuda"
    mk = torch.load(Path(CKPT_DIR) / a.matcher, map_location=dev, weights_only=False)
    ta = mk["args"]
    matcher = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                        ta.get("head", "global")).to(dev)
    matcher.load_state_dict(mk["model"])
    matcher.eval()
    fk = torch.load(Path(CKPT_DIR) / a.field, map_location=dev, weights_only=False)
    fa = fk["args"]
    field_model = CoarseField(fa["n"], fa["ch"], fa["dim"], fa["hidden"]).to(dev)
    field_model.load_state_dict(fk["model"])
    field_model.eval()
    joint = jfrozen = None
    if a.joint:
        joint, jfrozen, jck = load_joint(a.joint, dev)
        print(f"joint {a.joint} step {jck.get('step')}", flush=True)
    print(f"matcher {a.matcher}, field {a.field}, "
          f"construction {'agreement' if joint else 'loop closure'}", flush=True)

    report = {"matcher": a.matcher, "field": a.field, "margin": a.margin,
              "rounds": a.rounds}
    if a.validate:
        v = validate(matcher, field_model, dev, a.validate, a.margin, a.rounds,
                     a.min_tiles, joint, jfrozen, a.frac, a.place_conf)
        report["validation"] = v
        print(json.dumps(v, indent=1), flush=True)
        print(f"islands arm: SSIM {v['islands_ssim']} at detail "
              f"{v['islands_detail']}; field alone {v['field_ssim']} at detail "
              f"{v['field_detail']}; flat fill {v['flat_ssim']}", flush=True)

    out = Path(a.out)
    png = out / "png"
    png.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in Path(a.test_dir).glob("*.png"))
    if a.limit:
        names = names[:a.limit]
    print(f"{len(names)} test boards -> {png}", flush=True)

    t0, counts = time.time(), []
    for i, nm in enumerate(names):
        tiles = to_frags(load_rgb(Path(a.test_dir) / nm)).astype(np.float32)
        img, n_isl = build_one(tiles, matcher, field_model, dev, a.margin,
                               a.rounds, a.min_tiles, joint, jfrozen, a.frac,
                               a.place_conf)
        counts.append(n_isl)
        cv2.imwrite(str(png / nm), img[:, :, ::-1])
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(names)}  {el:.0f}s  eta "
                  f"{el / (i + 1) * (len(names) - i - 1):.0f}s", flush=True)

    report["boards"] = len(names)
    report["islands_per_board"] = round(float(np.mean(counts)), 1)
    report["seconds"] = round(time.time() - t0, 1)
    if a.limit:
        print("smoke run: no ZIP written", flush=True)
    else:
        zpath = out / "submission_islands_v1.zip"
        report["zip"] = str(zpath)
        report["zip_sha256"] = deterministic_zip(png, names, zpath)
        print(f"wrote {zpath}\nsha256 {report['zip_sha256']}", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
