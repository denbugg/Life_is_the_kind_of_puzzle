"""Use the strongest full-resolution seam ensemble inside the discrete solver."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from build_field_cache import pool8
from config import CACHE_DIR, CKPT_DIR, TRAIN_INP
from discrete_field_decoder import solve_discrete
from frame_classifier import frame_features, frame_unary
from island_field_decoder import solve_island_field
from seam_cost import costs_from_models
from seam_embed import SeamEmbed
from sinkhorn_assembler import SinkhornAssembler, decode
from train_sinkhorn_assembler import metrics


def load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(image[..., ::-1])


def fragments(image):
    return image.reshape(24, 20, 24, 20, 3).transpose(0, 2, 1, 3, 4).reshape(
        576, 20, 20, 3)


def load_matcher(name, device):
    ck = torch.load(Path(CKPT_DIR) / name, map_location=device, weights_only=False)
    a = ck["args"]
    model = SeamEmbed(a["ch"], a["blocks"], a["dim"], a["strip"],
                      a.get("head", "global"), a.get("predict_weight", 0) > 0,
                      a.get("norm_only", False), bool(a.get("restored", False))).to(device)
    model.load_state_dict(ck["model"])
    model.rows = a.get("rows") or None
    model.modes = int(a.get("modes", 1))
    model.eval()
    return model


def edge_recall(right, down, target, side):
    grid = target.reshape(side, side)
    ranks = []
    for score, anchor, truth in (
            (right, grid[:, :-1].reshape(-1), grid[:, 1:].reshape(-1)),
            (down, grid[:-1].reshape(-1), grid[1:].reshape(-1))):
        s = score[anchor].copy()
        s[np.arange(len(anchor)), anchor] = -np.inf
        tv = s[np.arange(len(anchor)), truth]
        ranks.append((s > tv[:, None]).sum(1))
    rank = np.concatenate(ranks)
    return (float((rank == 0).mean()), float((rank < 5).mean()),
            float((rank < 20).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sinkhorn", required=True)
    ap.add_argument("--side-override", type=int, default=0)
    ap.add_argument("--matchers", default="seam_embed_v3.pt,seam_embed_local.pt")
    ap.add_argument("--boards", type=int, default=64)
    ap.add_argument("--board-offset", type=int, default=0)
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--offsets", type=int, default=0,
                    help="candidate translations per island; 0 means every cell")
    ap.add_argument("--beam", type=int, default=128)
    ap.add_argument("--weights", default="0.3,1,3")
    ap.add_argument("--keeps", default="4,8,12,20,30")
    ap.add_argument("--border-weights", default="0")
    ap.add_argument("--frame", default="")
    ap.add_argument("--frame-weights", default="0")
    ap.add_argument("--seed", type=int, default=9876)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    sc = torch.load(a.sinkhorn, map_location=dev, weights_only=False)
    sa = sc["args"]
    side = int(a.side_override or sa["side"])
    slot = SinkhornAssembler(sa["d"], sa["rounds"], sa["blocks"]).to(dev)
    slot.load_state_dict(sc["model"]); slot.eval()
    matchers = [load_matcher(v.strip(), dev) for v in a.matchers.split(",")]
    print(f"loaded {len(matchers)} matchers on {dev}", flush=True)
    weights = [float(v) for v in a.weights.split(",")]
    keeps = [int(v) for v in a.keeps.split(",")]
    border_weights = [float(v) for v in a.border_weights.split(",")]
    frame_weights = [float(v) for v in a.frame_weights.split(",")]
    frame_model = None
    if a.frame:
        with open(a.frame, "rb") as f:
            frame_model = pickle.load(f)["model"]

    labels = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = labels["names"][-300:][a.board_offset:a.board_offset + a.boards]
    inverses = labels["inv"][-300:][a.board_offset:a.board_offset + a.boards].astype(np.int64)
    print(f"loaded labels; evaluating {len(names)} crops", flush=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for kk, (nm0, inv) in enumerate(zip(names, inverses)):
        y, x = int(rng.integers(25 - side)), int(rng.integers(25 - side))
        cells = ((np.arange(side)[:, None] + y) * 24
                 + np.arange(side)[None, :] + x).reshape(-1)
        perm = rng.permutation(side * side)
        target = np.empty(side * side, np.int64); target[perm] = np.arange(side * side)
        nm = str(nm0)
        raw = fragments(load_rgb(Path(TRAIN_INP) / nm))[inv]
        raw_tiles = raw[cells][perm].astype(np.float32)
        bag_tiles = np.round(pool8(raw_tiles)).astype(np.uint8)

        with torch.no_grad():
            inp = torch.from_numpy(bag_tiles[None]).to(dev)
            logits, _, _ = slot(inp, side)
        base = decode(logits)[0]
        unary = F.log_softmax(logits[0].float(), -1).cpu().numpy()
        ch, cv = costs_from_models(matchers, raw_tiles, device=dev)
        if kk == 0:
            print("first full-resolution score matrix ready", flush=True)
        right, down = -ch, -cv
        learned_frame = np.zeros_like(unary)
        if frame_model is not None:
            stats = np.concatenate([raw_tiles.mean((1, 2)),
                                    raw_tiles.std((1, 2))], 1) / 255.0
            feat = frame_features(right, down, stats)
            probability = frame_model.predict_proba(feat)[:, 1]
            learned_frame = frame_unary(probability, side)
        e1, e5, e20 = edge_recall(right, down, target, side)
        bp, ba = metrics(base[None], target[None], side)
        row = [bp, ba, e1, e5, e20]
        for weight in weights:
            lay, _ = solve_discrete(right, down, unary, base, side, weight,
                                    a.iters, a.restarts, a.seed + kk)
            if kk == 0:
                print(f"first decoder arm ready (weight={weight:g})", flush=True)
            p, adj = metrics(lay[None], target[None], side)
            row.extend((p, adj))
        for keep in keeps:
            for border_weight in border_weights:
                for frame_weight in frame_weights:
                    lay, _, comps = solve_island_field(
                        right, down, unary + frame_weight * learned_frame,
                        side, keep, beam=a.beam,
                        offsets=a.offsets or side * side,
                        border_weight=border_weight)
                    p, adj = metrics(lay[None], target[None], side)
                    row.extend((p, adj, sum(map(len, comps))))
        rows.append(row)
        if (kk + 1) % 8 == 0:
            z = np.asarray(rows)
            print(f"{kk+1}/{len(names)} edgeR1 {z[:,2].mean():.3f} "
                  f"base {z[:,0].mean():.3f} best {z[:,5::2].mean(0).max():.3f}",
                  flush=True)
    z = np.asarray(rows)
    print(f"base place/adj {z[:,0].mean():.6f}/{z[:,1].mean():.6f}")
    print(f"strong edge R@1/5/20 {z[:,2].mean():.6f}/{z[:,3].mean():.6f}/"
          f"{z[:,4].mean():.6f}")
    for wi, weight in enumerate(weights):
        print(f"weight {weight:g} place/adj {z[:,5+2*wi].mean():.6f}/"
              f"{z[:,6+2*wi].mean():.6f}")
    start = 5 + 2 * len(weights)
    arm = 0
    for keep in keeps:
        for border_weight in border_weights:
            for frame_weight in frame_weights:
                print(f"islands {keep:3d} border {border_weight:g} frame "
                      f"{frame_weight:g} place/adj/tiles "
                      f"{z[:,start+3*arm].mean():.6f}/"
                      f"{z[:,start+3*arm+1].mean():.6f}/"
                      f"{z[:,start+3*arm+2].mean():.2f}")
                arm += 1


if __name__ == "__main__":
    main()
