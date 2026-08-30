"""Fine-tune the joint seam verifier on the all-train top-k cache.

The split is fixed by board index: 0..6699 train, 6700..6999 validation.
Only corrupted input pixels and permutation-derived candidate labels are read.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import CACHE_DIR, CKPT_DIR, TRAIN_INP
from infer_coarse_field import load_rgb
from restore_tile import to_frags
from train_verify import board_pairs
from verify_pair import SeamVerifier, precision_at_k


class Cache:
    def __init__(self, root: Path):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.root = root
        self.manifest = manifest
        self.names = np.load(root / "names.npy")
        self.inverse = np.load(root / "inv.npy", mmap_mode="r")
        self.idx = np.load(root / "idx.npy", mmap_mode="r")
        self.val = np.load(root / "val.npy", mmap_mode="r")
        self.lab = np.load(root / "lab.npy", mmap_mode="r")
        self.done = np.load(root / "done.npy", mmap_mode="r")
        if len(self.names) != manifest["boards"]:
            raise ValueError("cache length disagrees with manifest")

    def board(self, index):
        if not self.done[index]:
            raise ValueError(f"cache board {index} is incomplete")
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / str(self.names[index])))[
            self.inverse[index].astype(np.int64)].astype(np.uint8)
        packs = {
            "h": (np.asarray(self.idx[index, 0], np.int64),
                  np.asarray(self.val[index, 0], np.float32),
                  np.asarray(self.lab[index, 0], np.int64)),
            "v": (np.asarray(self.idx[index, 1], np.int64),
                  np.asarray(self.val[index, 1], np.float32),
                  np.asarray(self.lab[index, 1], np.int64)),
        }
        return tiles, packs


def load_model(path: Path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    model = SeamVerifier(args.get("ch", 64), args.get("blocks", 4), 6,
                         args.get("strip", 4)).to(device)
    model.load_state_dict(ck["model"])
    model.strip = args.get("strip", 4)
    return model


def score_pairs(model, patch, feats, chunk):
    return torch.cat([model(patch[i:i + chunk], feats[i:i + chunk])
                      for i in range(0, len(patch), chunk)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=Path(CACHE_DIR) / "verify_top5_v2")
    parser.add_argument("--init", default="verify_hinge.pt")
    parser.add_argument("--out", default="verify_full_hinge.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--volume", type=int, default=430)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--train-end", type=int, default=6700)
    parser.add_argument("--held-start", type=int, default=6700)
    parser.add_argument("--held-boards", type=int, default=96)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    cache = Cache(args.cache)
    if args.train_end > cache.manifest["train_end"]:
        parser.error("training range crosses the frozen validation boundary")
    train_ids = np.arange(args.train_end)
    held_ids = np.arange(args.held_start,
                         min(len(cache.names), args.held_start + args.held_boards))
    required = np.concatenate([train_ids, held_ids])
    missing = required[~np.asarray(cache.done[required], bool)]
    if len(missing):
        raise RuntimeError(f"cache incomplete: first missing board {missing[0]}, "
                           f"{len(missing)} required boards missing")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init = Path(args.init)
    if not init.is_file():
        init = Path(CKPT_DIR) / init
    model = load_model(init, device)
    strip = model.strip
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    total_steps = args.epochs * len(train_ids)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, total_steps), eta_min=args.lr * 0.05)
    rng = np.random.default_rng(args.seed)

    @torch.no_grad()
    def evaluate(limit=None):
        model.eval()
        ours, base = [], []
        ids = held_ids if limit is None else held_ids[:limit]
        for index in ids:
            tiles, packs = cache.board(int(index))
            got = board_pairs(torch.from_numpy(tiles).float().to(device),
                              packs, strip, device)
            if got is None:
                continue
            patch, feats, labels = got
            scores = score_pairs(model, patch, feats, args.chunk)
            ours.append(precision_at_k(scores, labels, args.volume))
            base.append(precision_at_k(feats[:, 0], labels, args.volume))
        model.train()
        return float(np.mean(ours)), float(np.mean(base))

    before = evaluate()
    print(f"init precision@{args.volume}: verifier {before[0]:.5f}, "
          f"matcher {before[1]:.5f}; train={len(train_ids)} held={len(held_ids)}",
          flush=True)
    best = before[0]
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(train_ids)
        for index in order:
            if args.max_steps and step >= args.max_steps:
                break
            tiles, packs = cache.board(int(index))
            got = board_pairs(torch.from_numpy(tiles).float().to(device),
                              packs, strip, device)
            if got is None:
                continue
            patch, feats, labels = got
            optimizer.zero_grad(set_to_none=True)
            # The operating threshold is a whole-board statistic.  Freeze it,
            # then backpropagate exact top-k hinge chunks without retaining all
            # convolutional activations at once.
            with torch.no_grad():
                frozen = score_pairs(model, patch, feats, args.chunk)
                kth = min(args.volume, frozen.numel())
                threshold = torch.kthvalue(-frozen.flatten(), kth).values.neg()
            loss_value = 0.0
            for start in range(0, len(patch), args.chunk):
                out = model(patch[start:start + args.chunk],
                            feats[start:start + args.chunk])
                target = labels[start:start + args.chunk]
                positive = target > 0.5
                below = torch.relu(args.margin - (out - threshold))[positive]
                above = torch.relu(args.margin + (out - threshold))[~positive]
                loss = (below.sum() + above.sum()) / len(patch)
                loss.backward()
                loss_value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            step += 1
            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"step {step}/{total_steps} loss {loss_value:.5f} "
                      f"{elapsed/step:.3f}s/board", flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                score, base = evaluate()
                print(f"  eval verifier {score:.5f} matcher {base:.5f}", flush=True)
                payload = {"model": model.state_dict(),
                           "args": {"ch": 64, "blocks": 4, "strip": strip},
                           "train": vars(args), "step": step,
                           "eval": {"precision": score, "matcher": base}}
                torch.save(payload, Path(CKPT_DIR) / args.out)
                if score > best:
                    best = score
                    best_name = args.out[:-3] + "_best.pt"
                    torch.save(payload, Path(CKPT_DIR) / best_name)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    print(json.dumps({"initial": before[0], "best": best, "steps": step}))


if __name__ == "__main__":
    main()
