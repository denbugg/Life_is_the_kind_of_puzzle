"""Train the global assignment head on frozen full-resolution seam evidence."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from build_field_cache import pool8
from config import CACHE_DIR, CKPT_DIR, TRAIN_INP
from eval_strong_crop_solver import load_matcher
from seam_cost import costs_from_models
from sinkhorn_assembler import SinkhornAssembler, decode, permutation_loss
from train_sinkhorn_assembler import metrics


def load_rgb(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(im[..., ::-1])


def fragments(image):
    return image.reshape(24, 20, 24, 20, 3).transpose(0, 2, 1, 3, 4).reshape(
        576, 20, 20, 3)


def sample(labels, index, side, rng):
    name = str(labels["names"][index])
    inv = labels["inv"][index].astype(np.int64)
    raw = fragments(load_rgb(Path(TRAIN_INP) / name))[inv]
    y, x = int(rng.integers(25 - side)), int(rng.integers(25 - side))
    cells = ((np.arange(side)[:, None] + y) * 24
             + np.arange(side)[None, :] + x).reshape(-1)
    perm = rng.permutation(side * side)
    tiles = raw[cells][perm].astype(np.float32)
    target = np.empty(side * side, np.int64); target[perm] = np.arange(side * side)
    bag = np.round(pool8(tiles)).astype(np.uint8)
    return bag, tiles, target


def strong_edges(matchers, raw_tiles, device):
    ch, cv = costs_from_models(matchers, raw_tiles, device=device)
    # Per-row log probabilities give the recurrent decoder a stable scale.
    score = torch.from_numpy(np.stack([-ch, -cv])).to(device).float()
    return torch.log_softmax(score, -1)[None]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-boards", type=int, default=16)
    ap.add_argument("--matchers", default="seam_embed_v3.pt,seam_embed_local.pt")
    ap.add_argument("--resume", required=True)
    ap.add_argument("--out", default="strong_sinkhorn_s12.pt")
    ap.add_argument("--seed", type=int, default=2468)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    labels = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    cut = len(labels["names"]) - 300
    matchers = [load_matcher(v.strip(), dev) for v in a.matchers.split(",")]
    for model in matchers:
        for p in model.parameters(): p.requires_grad_(False)

    model = SinkhornAssembler(a.d, a.rounds, a.blocks).to(dev)
    ck = torch.load(a.resume, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    # These heads are bypassed by the frozen seam ensemble.
    for p in list(model.edge_q.parameters()) + list(model.edge_k.parameters()):
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(dev)

    # Freeze evaluation crops and their expensive matcher matrices once.
    eval_data = []
    eval_rng = np.random.default_rng(97531)
    for idx in range(cut, cut + a.eval_boards):
        bag, raw, target = sample(labels, idx, a.side, eval_rng)
        eval_data.append((bag, target, strong_edges(matchers, raw, dev).cpu()))

    @torch.no_grad()
    def evaluate():
        model.eval(); values = []
        for bag, target, edges in eval_data:
            x = torch.from_numpy(bag[None]).to(dev)
            logits, _, _ = model(x, a.side, edge_override=edges.to(dev))
            values.append(metrics(decode(logits), target[None], a.side))
        model.train()
        return np.mean(values, 0)

    best, t0 = -1.0, time.time()
    print(f"strong-edge curriculum side={a.side}, {a.eval_boards} fixed eval boards",
          flush=True)
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            place, adj = evaluate()
            print(f"[{step}] place {place:.5f} adj {adj:.5f} "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)
            if place > best:
                best = float(place)
                out = Path(a.out)
                if not out.is_absolute(): out = Path(CKPT_DIR) / out
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "place": float(place), "adjacency": float(adj)}, out)
        if step == a.steps:
            break
        idx = int(rng.integers(cut))
        bag, raw, target = sample(labels, idx, a.side, rng)
        edges = strong_edges(matchers, raw, dev)
        x = torch.from_numpy(bag[None]).to(dev)
        y = torch.from_numpy(target[None]).to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev, dtype=torch.float16):
            _, history, _ = model(x, a.side, edge_override=edges)
            loss = permutation_loss(history, y)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % 50 == 0:
            print(f"  loss {float(loss.detach()):.4f}", flush=True)
    print(f"best placement {best:.5f}")


if __name__ == "__main__":
    main()
