"""Training datasets. Data is generated two ways and mixed:
  - SYNTHETIC: clean target -> per-fragment distort (perfect pairing, infinite aug)
  - REAL: real distorted input fragments re-assembled in recovered correct order
Both give (distorted-in-correct-order, clean) pairs and known adjacency.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import GRID, FS, NFRAG, TRAIN_TGT, TRAIN_INP, CACHE_DIR
from imgio import load, to_frags, from_frags
from distort import distort_frags

_PERMS = None
def _perms():
    global _PERMS
    if _PERMS is None:
        p = os.path.join(CACHE_DIR, "perms.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            d = {"names": z["names"], "inv": z["inv"], "conf": z["conf"]}  # materialize once
            _PERMS = {n: i for i, n in enumerate(d["names"])}, d
        else:
            _PERMS = ({}, None)
    return _PERMS


def real_recon(name):
    """Real distorted fragments assembled in recovered correct order + confidence."""
    idx, d = _perms()
    if name not in idx:
        return None
    k = idx[name]
    inv = d["inv"][k].astype(int)
    fi = to_frags(load(os.path.join(TRAIN_INP, name)))
    return fi[inv], d["conf"][k]      # (576,20,20,3) correct order, conf per position


# --------------------------- Restoration ---------------------------
class RestoreDataset(Dataset):
    def __init__(self, names, crop_frags=12, real_prob=0.5, flip=True):
        self.names = names
        self.cf = crop_frags
        self.real_prob = real_prob
        self.flip = flip

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        name = self.names[i]
        use_real = (self.real_prob > 0 and rng.random() < self.real_prob
                    and name in _perms()[0])
        if use_real:
            frags, _ = real_recon(name)
            dist = from_frags(frags)
            clean = load(os.path.join(TRAIN_TGT, name))
        else:
            clean = load(os.path.join(TRAIN_TGT, name))
            dist = None  # distort only the crop below (cheaper)
        m = GRID - self.cf
        r, c = rng.integers(0, m + 1), rng.integers(0, m + 1)
        y0, x0 = r * FS, c * FS
        sz = self.cf * FS
        cl = clean[y0:y0 + sz, x0:x0 + sz]
        if use_real:
            di = dist[y0:y0 + sz, x0:x0 + sz]
        else:
            di = from_frags(distort_frags(to_frags(cl), rng))
        if self.flip:
            if rng.random() < 0.5:
                cl, di = cl[:, ::-1], di[:, ::-1]
            if rng.random() < 0.5:
                cl, di = cl[::-1], di[::-1]
        cl = torch.from_numpy(np.ascontiguousarray(cl)).permute(2, 0, 1).float() / 255
        di = torch.from_numpy(np.ascontiguousarray(di)).permute(2, 0, 1).float() / 255
        return di, cl


# --------------------------- Compatibility ---------------------------
class CompatDataset(Dataset):
    """Yields (576,3,20,20) distorted fragments in CORRECT row-major order,
    so fragment index == grid position (right neighbor = +1, below = +GRID)."""
    def __init__(self, names, real_prob=0.5):
        self.names = names
        self.real_prob = real_prob

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        name = self.names[i]
        use_real = (self.real_prob > 0 and rng.random() < self.real_prob
                    and name in _perms()[0])
        if use_real:
            frags, _ = real_recon(name)                 # correct order, real distortion
        else:
            clean = load(os.path.join(TRAIN_TGT, name))
            frags = distort_frags(to_frags(clean), rng)  # correct order, synthetic
        t = torch.from_numpy(np.ascontiguousarray(frags)).permute(0, 3, 1, 2).float() / 255
        return t
