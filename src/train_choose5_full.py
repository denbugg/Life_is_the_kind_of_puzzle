"""Fine-tune the listwise five-candidate chooser on all 6700 train boards."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from choose5 import K, Choose5, seam_patch
from config import CACHE_DIR, CKPT_DIR
from train_choose5 import board_batch, run_board
from train_verify_full import Cache


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    model = Choose5(args.get("ch", 64), args.get("dim", 192),
                    args.get("strip", 4), args.get("layers", 3),
                    encoder=args.get("encoder", "cnn")).to(device)
    model.load_state_dict(ck["model"])
    model.strip = args.get("strip", 4)
    return model, args


def torch_packs(packs):
    return {axis: tuple(torch.from_numpy(np.asarray(value)) for value in values)
            for axis, values in packs.items()}


@torch.no_grad()
def choice_metrics(model, tiles, packs, device, volume=430):
    """Correct choices and precision of the most confident emitted edges."""
    correct, confidence = [], []
    for axis in ("h", "v"):
        idx, val, lab = packs[axis]
        idx, val, lab = idx.to(device), val.to(device), lab.to(device)
        keep, src, dst, values, target = board_batch(
            tiles, idx, val, lab, model.strip, device)
        if not len(keep):
            continue
        patch = seam_patch(tiles, src, dst, axis, model.strip).reshape(
            len(keep), K, 3, 20, 2 * model.strip)
        rank = torch.arange(K, device=device, dtype=torch.float32)
        relative = values - values[:, :1]
        scalars = torch.stack(
            [values / 10.0, relative, rank.expand(len(keep), K),
             (relative == 0).float()], -1)
        logits = model(patch, scalars)
        pick = logits.argmax(1)
        top2 = logits.topk(2, dim=1).values
        emitted = pick < K
        correct.append(((pick == target) & emitted)[emitted].float())
        confidence.append((top2[:, 0] - top2[:, 1])[emitted])
    if not correct:
        return 0, 0, 0.0
    y = torch.cat(correct)
    conf = torch.cat(confidence)
    order = torch.argsort(conf, descending=True)[:min(volume, len(conf))]
    return int(y.sum()), len(y), float(y[order].mean()) if len(order) else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=Path(CACHE_DIR) / "verify_top5_v2")
    parser.add_argument("--init", default="choose5_big.pt")
    parser.add_argument("--out", default="choose5_full.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--none-weight", type=float, default=0.3)
    parser.add_argument("--train-end", type=int, default=6700)
    parser.add_argument("--held-start", type=int, default=6700)
    parser.add_argument("--held-boards", type=int, default=96)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    cache = Cache(args.cache)
    if args.train_end > cache.manifest["train_end"]:
        parser.error("training range crosses frozen validation")
    train_ids = np.arange(args.train_end)
    held_ids = np.arange(args.held_start,
                         min(len(cache.names), args.held_start + args.held_boards))
    required = np.concatenate([train_ids, held_ids])
    missing = required[~np.asarray(cache.done[required], bool)]
    if len(missing):
        raise RuntimeError(f"cache incomplete: {len(missing)} required boards")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init = Path(args.init)
    if not init.is_file():
        init = Path(CKPT_DIR) / init
    model, model_args = load_model(init, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    total_steps = args.epochs * len(train_ids)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, total_steps), eta_min=args.lr * 0.05)
    rng = np.random.default_rng(args.seed)

    @torch.no_grad()
    def evaluate():
        model.eval()
        got = base = rows = emitted = 0
        top_precision = []
        per_board = []
        for index in held_ids:
            tiles, packs = cache.board(int(index))
            tensor = torch.from_numpy(tiles.astype(np.float32)).to(device)
            packed = torch_packs(packs)
            _, result = run_board(model, tensor, packed,
                                  model.strip, device,
                                  False, args.none_weight)
            _, _, top = choice_metrics(model, tensor, packed, device)
            top_precision.append(top)
            g = sum(r[1] for r in result)
            b = sum(r[2] for r in result)
            n = sum(r[3] for r in result)
            e = sum(r[4] for r in result)
            got += g; base += b; rows += n; emitted += e
            per_board.append(g - b)
        model.train()
        count = max(len(held_ids), 1)
        return {"chooser_bonds": got / count, "matcher_bonds": base / count,
                "rows": rows / count, "delta": (got - base) / count,
                "emitted": emitted / count,
                "precision": got / max(emitted, 1),
                "precision_at_430": float(np.mean(top_precision)),
                "delta_sd": float(np.std(per_board))}

    initial = evaluate()
    print("init", json.dumps(initial), flush=True)
    best_bonds = initial["chooser_bonds"]
    best_precision = initial["precision_at_430"]
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        for index in rng.permutation(train_ids):
            if args.max_steps and step >= args.max_steps:
                break
            tiles, packs = cache.board(int(index))
            tensor = torch.from_numpy(tiles.astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = run_board(model, tensor, torch_packs(packs),
                                model.strip, device,
                                True, args.none_weight)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{total_steps} loss {loss:.5f} "
                      f"{(time.time()-t0)/step:.3f}s/board", flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                metrics = evaluate()
                print("  eval", json.dumps(metrics), flush=True)
                payload = {"model": model.state_dict(), "args": model_args,
                           "train": vars(args), "step": step,
                           "eval": metrics}
                torch.save(payload, Path(CKPT_DIR) / args.out)
                if metrics["chooser_bonds"] > best_bonds:
                    best_bonds = metrics["chooser_bonds"]
                    torch.save(payload, Path(CKPT_DIR) /
                               (args.out[:-3] + "_best_bonds.pt"))
                if metrics["precision_at_430"] > best_precision:
                    best_precision = metrics["precision_at_430"]
                    torch.save(payload, Path(CKPT_DIR) /
                               (args.out[:-3] + "_best_precision.pt"))
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    print(json.dumps({"initial": initial, "best_bonds": best_bonds,
                      "best_precision_at_430": best_precision,
                      "steps": step}))


if __name__ == "__main__":
    main()
