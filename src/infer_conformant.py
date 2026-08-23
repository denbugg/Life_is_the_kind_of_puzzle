"""The submission the rules actually ask for: assemble all 576, then restore.

The organisers restated the requirements on 2026-08-20: every one of the 576
fragments must be assigned a cell of the 24x24 grid, no fragment may be altered
to make it placeable, and the assembled image must then be restored -- noise,
blur, artefacts and brightness are named explicitly.  A solution that fails the
manual check is excluded from the final whatever it scores automatically.

Assembly
--------
Components are built from edges that two views agree on -- the raw tile and the
same tile after a per-tile restorer -- which run at precision 0.91-0.95 where
either view alone runs at 0.45 (M167).  Those components are packed by the
earlier team's R10-A packer, which fills every remaining cell and returns a full
bijection (M171); place_acc rises about elevenfold against our previous solver.

Restoration strength is a dial, and it is the only part the metric sees
----------------------------------------------------------------------
M174 measured four layouts spanning adjacency 0.005 to 0.21 through the same
restoration ladder.  At full restoration they score 0.3433 to 0.3438 -- a spread
of five ten-thousandths.  The layout is worth almost nothing to the metric and
the restoration strength is worth 0.13, so `--blend` is the only number here
that moves the score:

  0.00  R5 alone, the inherited denoiser     ~0.216, detail 30   looks like a photo
  0.50  halfway                              ~0.243, detail 17
  0.75  three quarters                       ~0.286, detail  9   visibly smoothed
  1.00  the restorer fitted to our boards    ~0.344, detail  1   very smooth

Training a network directly under a detail floor does NOT beat this blend at
equal detail (M175): the constraint makes the model invent texture, and invented
texture is uncorrelated with the target and charged for, while the blend's
texture is the real tile content coming through R5.

Choose the point for what an expert should see, not for the number.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import infer_rank96 as rank96
from config import CACHE_DIR, CKPT_DIR, FS, GRID as G, SUB_DIR, TEST_DIR, TRAIN_INP, TRAIN_TGT
from infer_coarse_field import deterministic_zip, load_rgb
from models import RestoreNet
from restore_tile import TileRestorer, to_frags
from seam_cost import costs_from_models
from seam_embed import SeamEmbed
from solve_buddies import build_directed_components, solve_components_from_scores

DIR_RIGHT, DIR_DOWN = 3, 1
N = G * G


def assemble(tiles, lay):
    x = np.clip(tiles[np.asarray(lay)], 0, 255).astype(np.uint8)
    return x.reshape(G, G, FS, FS, 3).transpose(0, 2, 1, 3, 4).reshape(G * FS, G * FS, 3)


def _mutual(C):
    D = C.copy()
    np.fill_diagonal(D, np.inf)
    f, b = D.argmin(1), D.argmin(0)
    part = np.partition(D, 1, axis=1)
    return {i: (int(f[i]), float(part[i, 1] / max(part[i, 0], 1e-9)))
            for i in range(C.shape[0]) if b[int(f[i])] == i}


def _edge_map(CH, CV):
    out = {}
    for C, off in ((CH, (0, 1)), (CV, (1, 0))):
        for i, (j, m) in _mutual(C).items():
            out[(i, j, off)] = m
    return out


@torch.no_grad()
def _restore_tiles(model, tiles, dev):
    x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).to(dev)
    out = []
    for i in range(0, x.shape[0], 256):
        with torch.autocast("cuda", torch.float16):
            out.append(model(x[i:i + 256]).float())
    return torch.cat(out).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy()


def agreed_edges(matcher, restorers, tiles, dev, keep=0.5):
    """Edges the raw view and each restored view independently both call best.

    `matcher` may be a single model or a list; a list is combined by the
    pessimistic minimum of M201, which is worth about +0.02 edge precision over
    the best member when the members are of comparable strength.
    """
    CH, CV = costs_from_models(matcher, tiles)
    base = _edge_map(CH, CV)
    agreed = {}
    for m in restorers:
        other = _edge_map(*costs_from_models(matcher, _restore_tiles(m, tiles, dev)))
        both = {e: max(base[e], other[e]) for e in base if e in other}
        order = sorted(both, key=lambda e: -both[e])
        for e in order[: max(1, int(round(keep * len(order))))]:
            agreed[e] = max(agreed.get(e, 0.0), both[e])
    return CH, CV, agreed


def solve_board(matcher, restorers, tiles, dev):
    CH, CV, agreed = agreed_edges(matcher, restorers, tiles, dev)
    comps = build_directed_components(
        [i for (i, j, o) in agreed], [DIR_RIGHT if o == (0, 1) else DIR_DOWN
                                      for (i, j, o) in agreed],
        [j for (i, j, o) in agreed], list(agreed.values()),
        max_edges=len(agreed))
    lay = solve_components_from_scores((-CH).astype(np.float32),
                                       (-CV).astype(np.float32), comps,
                                       repair_passes=0, restarts=1)[0]
    lay = np.asarray(lay, np.int64)
    if len(np.unique(lay)) != N:
        raise RuntimeError("packer did not return a bijection")
    return lay, len(comps)


def _net(model, img, dev):
    with torch.no_grad():
        t = torch.from_numpy(img).to(dev, torch.float32).permute(2, 0, 1)[None] / 255.0
        o = model(t).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.rint(o * 255.0).clip(0, 255).astype(np.uint8)


def restore(img, r5, ours, blend, dev, nlm):
    a = _net(r5, img, dev).astype(np.float32)
    b = _net(ours, img, dev).astype(np.float32)
    out = np.rint(blend * b + (1.0 - blend) * a).clip(0, 255).astype(np.uint8)
    return rank96.fixed_nlm(out) if nlm else out


def detail(img):
    f = img.astype(np.float32)
    return float((f - cv2.GaussianBlur(f, (0, 0), 12.0)).std())


def load_unet(path, dev, base=None, depth=None):
    pay = torch.load(path, map_location=dev, weights_only=False)
    st = pay.get("model") or pay.get("state_dict") or pay
    b = base or st["stem.weight"].shape[0]
    d = depth or 1 + sum(1 for k in st if k.startswith("down.")
                         and k.endswith(".weight"))
    m = RestoreNet(base=b, depth=d).to(dev)
    m.load_state_dict(st, strict=True)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matcher", default="seam_embed_v1.pt")
    ap.add_argument("--restorers", nargs="+",
                    default=["tile_restorer_mgc.pt", "tile_restorer.pt"])
    ap.add_argument("--ours", default="restore_ours_v1.pt")
    ap.add_argument("--r5", default="E:/pazzle_work/pazzle_fixed_orientation_20260813/"
                                    "R5_restore_unet/r5_capacity_fp32.pt")
    ap.add_argument("--blend", type=float, default=0.75,
                    help="0 = R5 only, 1 = the restorer fitted to our boards")
    ap.add_argument("--nlm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--test-dir", default=TEST_DIR)
    ap.add_argument("--out", default=str(Path(SUB_DIR) / "conformant_v1"))
    ap.add_argument("--limit", type=int, default=0, help="smoke run; no ZIP")
    ap.add_argument("--validate", type=int, default=16)
    a = ap.parse_args()

    dev = "cuda"
    mk = torch.load(Path(CKPT_DIR) / a.matcher, map_location=dev, weights_only=False)
    ta = mk["args"]
    matcher = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                        ta.get("head", "global")).to(dev)
    matcher.load_state_dict(mk["model"])
    matcher.eval()
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
    r5 = load_unet(a.r5, dev)
    ok = torch.load(Path(CKPT_DIR) / a.ours, map_location=dev, weights_only=False)
    ours = RestoreNet(base=ok["args"]["base"], depth=ok["args"]["depth"]).to(dev)
    ours.load_state_dict(ok["model"])
    ours.eval()
    print(f"matcher {a.matcher}; {len(restorers)} tile restorers; blend {a.blend}; "
          f"NLM {a.nlm}", flush=True)

    report = {"matcher": a.matcher, "blend": a.blend, "nlm": bool(a.nlm),
              "ours": a.ours}
    if a.validate:
        from skimage.metrics import structural_similarity as ssim_fn
        blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
        names = [str(n) for n in blob["names"][-300:]]
        inv = blob["inv"][-300:]
        rows = []
        for k in range(min(a.validate, len(names))):
            tgt = load_rgb(Path(TRAIN_TGT) / names[k])
            tiles = to_frags(load_rgb(Path(TRAIN_INP) / names[k])).astype(np.float32)[
                inv[k].astype(np.int64)]
            lay, nc = solve_board(matcher, restorers, tiles, dev)
            img = restore(assemble(tiles, lay), r5, ours, a.blend, dev, a.nlm)
            rows.append([float(ssim_fn(img, tgt, channel_axis=2, data_range=255)),
                         detail(img), float(np.mean(lay == np.arange(N))), nc])
        m = np.mean(rows, axis=0)
        report["validation"] = {"ssim": round(float(m[0]), 4),
                                "detail": round(float(m[1]), 1),
                                "place_acc": round(float(m[2]), 4),
                                "components": round(float(m[3]), 1)}
        print(json.dumps(report["validation"], indent=1), flush=True)

    out = Path(a.out)
    png = out / "png"
    png.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in Path(a.test_dir).glob("*.png"))
    if a.limit:
        names = names[:a.limit]
    print(f"{len(names)} test boards -> {png}", flush=True)

    t0 = time.time()
    for i, nm in enumerate(names):
        tiles = to_frags(load_rgb(Path(a.test_dir) / nm)).astype(np.float32)
        lay, _ = solve_board(matcher, restorers, tiles, dev)
        cv2.imwrite(str(png / nm),
                    restore(assemble(tiles, lay), r5, ours, a.blend, dev,
                            a.nlm)[:, :, ::-1])
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(names)}  {el:.0f}s  eta "
                  f"{el / (i + 1) * (len(names) - i - 1):.0f}s", flush=True)

    report["boards"] = len(names)
    report["seconds"] = round(time.time() - t0, 1)
    if a.limit:
        print("smoke run: no ZIP written", flush=True)
    else:
        z = out / f"submission_conformant_b{int(round(a.blend * 100)):03d}.zip"
        report["zip"] = str(z)
        report["zip_sha256"] = deterministic_zip(png, names, z)
        print(f"wrote {z}\nsha256 {report['zip_sha256']}", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
