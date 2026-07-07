"""Recover ground-truth arrangement of a train input by matching its distorted
fragments to the clean target fragments (normalized 5x5 descriptor + Hungarian).

Gives, per train image:
  perm[i]  = grid position (0..575) where input fragment i belongs
  inv[p]   = input fragment index that belongs at grid position p
  conf[p]  = matching confidence for the fragment at position p (structure corr)
Used for real aligned (distorted<->clean) training data and local placement eval.
"""
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from config import FS, NFRAG, TRAIN_INP, TRAIN_TGT, CACHE_DIR
from imgio import load, to_frags


def _desc(frags, k=5, normalize=True):
    N = frags.shape[0]
    b = FS // k
    x = frags[:, :k * b, :k * b, :].astype(np.float32)
    x = x.reshape(N, k, b, k, b, 3).mean(axis=(2, 4)).reshape(N, -1)
    if normalize:
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    return x


def recover(inp_img, tgt_img):
    fi = to_frags(inp_img)
    ft = to_frags(tgt_img)
    di, dt = _desc(fi), _desc(ft)
    aa = (di * di).sum(1)[:, None]
    bb = (dt * dt).sum(1)[None, :]
    cost = aa + bb - 2.0 * di @ dt.T                  # (input i, target pos j)
    ri, cj = linear_sum_assignment(cost)
    perm = np.empty(NFRAG, dtype=np.int16)
    perm[ri] = cj.astype(np.int16)
    inv = np.empty(NFRAG, dtype=np.int16)
    inv[cj] = ri.astype(np.int16)
    # confidence = normalized-descriptor correlation of the matched pair (by position)
    D = di.shape[1]
    corr_i = (di * dt[perm.astype(int)]).sum(1) / D   # per input fragment
    conf = np.empty(NFRAG, dtype=np.float32)
    conf[perm.astype(int)] = corr_i                   # per grid position
    return perm.astype(np.int64), inv.astype(np.int64), conf


def build_cache(names, out_path):
    perms = np.zeros((len(names), NFRAG), np.int16)
    invs = np.zeros((len(names), NFRAG), np.int16)
    confs = np.zeros((len(names), NFRAG), np.float32)
    for k, nm in enumerate(names):
        inp = load(os.path.join(TRAIN_INP, nm))
        tgt = load(os.path.join(TRAIN_TGT, nm))
        p, iv, cf = recover(inp, tgt)
        perms[k], invs[k], confs[k] = p, iv, cf
        if (k + 1) % 200 == 0:
            print(f"  recovered {k+1}/{len(names)}  conf_mean={confs[:k+1].mean():.3f}", flush=True)
    np.savez(out_path, names=np.array(names), perm=perms, inv=invs, conf=confs)
    print(f"saved {out_path}  conf_mean={confs.mean():.3f} "
          f"frac(conf>0.5)={(confs>0.5).mean():.3f}", flush=True)


if __name__ == "__main__":
    from imgio import list_train
    names = list_train()
    build_cache(names, os.path.join(CACHE_DIR, "perms.npz"))
