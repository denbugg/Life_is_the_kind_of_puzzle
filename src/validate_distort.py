"""Validate that synthetic distortion matches the real degradation statistics."""
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from config import TRAIN_INP, TRAIN_TGT, FS
from imgio import load, to_frags, from_frags
from recover import recover
from distort import distort_image


def stats(clean, dist, tag):
    s = ssim(clean.astype(np.uint8), dist.astype(np.uint8), channel_axis=2, data_range=255)
    fc = to_frags(clean).astype(np.float32)
    fd = to_frags(dist).astype(np.float32)
    # noise std in flat clean regions
    resid = []
    for i in range(0, fc.shape[0], 3):
        c, d = fc[i], fd[i]
        gx = np.abs(np.diff(c, axis=1)).mean(2)
        gy = np.abs(np.diff(c, axis=0)).mean(2)
        flat = np.zeros((FS, FS), bool)
        flat[1:-1, 1:-1] = (gx[1:-1, 1:] + gx[1:-1, :-1] + gy[1:, 1:-1] + gy[:-1, 1:-1] < 8)
        if flat.sum() < 10:
            continue
        for ch in range(3):
            resid.append((d[..., ch][flat] - d[..., ch][flat].mean()).std())
    # JPEG blockiness ratio at x=8,16 vs interior
    jb, ib = [], []
    for i in range(0, fd.shape[0], 3):
        g = fd[i].mean(2)
        for bnd in (8, 16):
            jb.append(np.abs(g[:, bnd] - g[:, bnd - 1]).mean())
        for inr in (4, 12):
            ib.append(np.abs(g[:, inr] - g[:, inr - 1]).mean())
    print(f"  {tag:16s} SSIM={s:.4f} noise_std~{np.median(resid):5.1f} "
          f"blockiness={np.mean(jb)/max(1e-6,np.mean(ib)):.2f}")
    return s


if __name__ == "__main__":
    names = sorted(os.listdir(TRAIN_TGT))[:6]
    rng = np.random.default_rng(0)
    real_s, synth_s = [], []
    for nm in names:
        inp = load(os.path.join(TRAIN_INP, nm))
        tgt = load(os.path.join(TRAIN_TGT, nm))
        perm, inv, conf = recover(inp, tgt)
        real_recon = from_frags(to_frags(inp)[inv])   # real distorted, correct order
        synth = distort_image(tgt, rng)               # synthetic distorted, correct order
        print(nm, f"conf_mean={conf.mean():.3f}")
        real_s.append(stats(tgt, real_recon, "REAL"))
        synth_s.append(stats(tgt, synth, "SYNTH"))
    print(f"\nMean SSIM(clean,distorted):  REAL={np.mean(real_s):.4f}  SYNTH={np.mean(synth_s):.4f}")
    print("(want SYNTH close to REAL; both ~0.43-0.50)")
