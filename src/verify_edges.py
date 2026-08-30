"""Re-score a harvest with the seam verifier.

M456 makes precision at the harvest volume the only figure that decides
anything: with the true edges held fixed and false ones added, the connected
block runs 350 correct fragments at edge precision 1.00, 186 at 0.99, 65 at 0.95
and 18 at the 0.746 the shipping harvest delivers. A wrong edge does not merely
fail to help -- it welds two islands at a false offset and destroys correct
structure that already existed.

The verifier is the only scorer in this project that is JOINT over the two sides
of a join rather than a dot product of two pooled descriptors, and the only one
trained for precision at a volume rather than by retrieval. Here it re-orders
what the vote harvested; nothing is dropped, because the weight is an ORDERING
(M270) and `build_directed_components` spends it on conflicts.
"""
import numpy as np
import torch

from choose5 import seam_patch
from config import GRID as G
from verify_pair import SeamVerifier

N = G * G


def load_verifier(path, device="cuda"):
    c = torch.load(path, map_location=device, weights_only=False)
    a = c.get("args", {})
    m = SeamVerifier(a.get("ch", 64), a.get("blocks", 4), 6,
                     a.get("strip", 4)).to(device)
    m.load_state_dict(c["model"])
    m.strip = a.get("strip", 4)
    m.eval()
    return m


def verify_harvest(model, tiles, CH, CV, agreed, device="cuda", chunk=8192):
    """{(i, j, offset): weight} re-weighted by the verifier's logit."""
    if not agreed:
        return agreed
    strip = getattr(model, "strip", 4)
    x = torch.from_numpy(np.ascontiguousarray(tiles)).float().to(device)
    H = -np.asarray(CH, np.float64)
    V = -np.asarray(CV, np.float64)
    np.fill_diagonal(H, -1e9)
    np.fill_diagonal(V, -1e9)
    out = {}
    for axis, off, M in (("h", (0, 1), H), ("v", (1, 0), V)):
        keys = [e for e in agreed if e[2] == off]
        if not keys:
            continue
        src = torch.tensor([e[0] for e in keys], device=device)
        dst = torch.tensor([e[1] for e in keys], device=device)
        # the features the verifier was trained on, rebuilt from this board's
        # own matrix: the score, its lead over the row's best, the rank, whether
        # it IS the best, the row's mean and its spread
        rowsort = np.sort(M, axis=1)[:, ::-1]
        best = rowsort[:, 0]
        mean8 = rowsort[:, :8].mean(1)
        spread = rowsort[:, 0] - rowsort[:, 7]
        rank = np.array([int((M[e[0]] > M[e[0], e[1]]).sum()) for e in keys])
        s = np.array([M[e[0], e[1]] for e in keys])
        b = best[[e[0] for e in keys]]
        feats = torch.tensor(np.stack([
            s / 10.0, s - b, rank.astype(np.float64), (s == b).astype(float),
            mean8[[e[0] for e in keys]] / 10.0,
            spread[[e[0] for e in keys]]], 1), dtype=torch.float32,
            device=device)
        logits = []
        with torch.no_grad():
            for k in range(0, len(keys), chunk):
                p = seam_patch(x, src[k:k + chunk], dst[k:k + chunk], axis,
                               strip)
                logits.append(model(p, feats[k:k + chunk]))
        logits = torch.cat(logits).cpu().numpy()
        for e, w in zip(keys, logits):
            out[e] = float(w)
    for e in agreed:
        out.setdefault(e, float(agreed[e]))
    return out
