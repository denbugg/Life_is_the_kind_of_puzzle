"""Build a submission from the coarse colour field.

Why this arm and not the assembly
---------------------------------
Scored as a gain over the flat fill -- the only honest scale on this task, since
absolute SSIM here mostly reports proximity to a constant (M137) -- the deployed
submission sits at -0.141 and this model sits at about +0.016 to +0.024.  That
is roughly 0.237 to 0.39 on the platform.

It needs no solver at all.  The model reads the unordered bag of 576 tiles, so
the order the pieces arrive in is irrelevant and nothing has to be assembled
first.  Seven hundred test boards take under a minute.

Honesty
-------
This is a trained restoration model, not a fill: its output is a function of the
board, which is the standing acceptance test (M146).  `--swap-check` runs that
test on held-out TRAIN boards and prints both numbers -- feeding a board another
board's tiles has to score clearly worse.  A run whose swapped score matches its
own score is emitting a constant with extra steps and must not be shipped.

What it looks like
------------------
Smooth.  Detail, measured as the spread surviving a 12 px high-pass, is about
0.3 where a real photograph scores 35 (M145).  That is the honest optimum given
that our layout places 0.6% of tiles correctly, and it is also the reason the
island work exists: islands placed correctly reach +0.095 at detail 27.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from coarse_field import CoarseField, render
from config import CACHE_DIR, CKPT_DIR, SUB_DIR, TEST_DIR, TRAIN_INP, TRAIN_TGT
from restore_tile import to_frags


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deterministic_zip(png_dir, names, destination):
    """Byte-identical for identical content, as the existing S1 builder is."""
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as ar:
        for name in names:
            path = png_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"missing output: {path}")
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            ar.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=9)
    os.replace(temp, destination)
    return file_sha256(destination)


@torch.no_grad()
def predict(model, tiles, dev):
    x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2)[None].to(dev)
    img = render(model(x))[0].permute(1, 2, 0).cpu().numpy()
    return np.rint(img * 255.0).clip(0, 255).astype(np.uint8)


def swap_check(model, dev, boards):
    """The M146 acceptance test, on held-out TRAIN boards."""
    from skimage.metrics import structural_similarity as ssim_fn

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = [str(n) for n in blob["names"][-300:]]
    inv = blob["inv"][-300:]
    own, swapped, flats = [], [], []
    n = min(boards, len(names))
    for k in range(n):
        tgt = load_rgb(Path(TRAIN_TGT) / names[k])
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / names[k])).astype(np.float32)[
            inv[k].astype(np.int64)]
        j = (k + 1) % n
        other = to_frags(load_rgb(Path(TRAIN_INP) / names[j])).astype(np.float32)[
            inv[j].astype(np.int64)]
        flat = np.zeros_like(tgt)
        flat[:] = np.rint(tiles.reshape(-1, 3).mean(0)).clip(0, 255).astype(np.uint8)
        base = float(ssim_fn(flat, tgt, channel_axis=2, data_range=255))
        flats.append(base)
        own.append(float(ssim_fn(predict(model, tiles, dev), tgt,
                                 channel_axis=2, data_range=255)) - base)
        swapped.append(float(ssim_fn(predict(model, other, dev), tgt,
                                     channel_axis=2, data_range=255)) - base)
    return float(np.mean(own)), float(np.mean(swapped)), float(np.mean(flats))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="coarse_field_n8.pt")
    ap.add_argument("--test-dir", default=TEST_DIR)
    ap.add_argument("--out", default=str(Path(SUB_DIR) / "coarse_field_v1"))
    ap.add_argument("--limit", type=int, default=0, help="smoke run; no ZIP")
    ap.add_argument("--swap-check", type=int, default=40,
                    help="held-out boards for the acceptance test; 0 skips it")
    a = ap.parse_args()

    dev = "cuda"
    ck = torch.load(Path(CKPT_DIR) / a.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    model = CoarseField(ca["n"], ca["ch"], ca["dim"], ca["hidden"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"{a.ckpt}: step {ck.get('step')}, its eval {ck.get('eval')}", flush=True)

    if a.swap_check:
        own, sw, flat = swap_check(model, dev, a.swap_check)
        print(f"acceptance test on {a.swap_check} held-out boards: "
              f"own input {own:+.4f}, swapped {sw:+.4f}, flat fill {flat:.4f} "
              f"absolute", flush=True)
        if own - sw < 0.002:
            raise SystemExit("REFUSING TO BUILD: the output barely depends on the "
                             "input, which is the definition of a fill (M146)")

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
        img = predict(model, tiles, dev)
        cv2.imwrite(str(png / nm), img[:, :, ::-1])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(names)}  {time.time()-t0:.0f}s", flush=True)

    report = {"checkpoint": a.ckpt, "checkpoint_step": ck.get("step"),
              "checkpoint_eval": ck.get("eval"), "boards": len(names),
              "seconds": round(time.time() - t0, 1)}
    if a.limit:
        print("smoke run: no ZIP written", flush=True)
    else:
        zpath = out / "submission_coarse_field_v1.zip"
        report["zip"] = str(zpath)
        report["zip_sha256"] = deterministic_zip(png, names, zpath)
        print(f"wrote {zpath}\nsha256 {report['zip_sha256']}", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
