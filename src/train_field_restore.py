"""Train the field-anchored restorer and report it on the honest scale.

Every number is a gain over the flat fill at our own tiles' mean colour, which
M137 established as the only meaningful zero on this task.  Reference points:

  deployed submission (R5 + NLM)            -0.141
  our assembled board, unprocessed          -0.253
  restorer anchored on the board (M147)     -0.030,  -0.017 with NLM
  coarse field alone (M144 CAL, 670 boards) +0.0159
  leader 0.40, converted to this scale      ~ +0.02

Step 0 of this run reproduces the field exactly, so anything above +0.0159 is
new.  The final line runs the acceptance test from M146: feed another board's
input and the output must get clearly worse.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader, Dataset

import infer_rank96 as rank96
from coarse_field import CoarseField, render
from config import CACHE_DIR, CKPT_DIR, FS, GRID as G, TRAIN_INP, TRAIN_TGT
from field_restore import FieldRestore
from models import restore_loss
from restore_tile import to_frags


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


def assemble(tiles, lay):
    x = np.clip(tiles[np.asarray(lay)], 0, 255).astype(np.uint8)
    return x.reshape(G, G, FS, FS, 3).transpose(0, 2, 1, 3, 4).reshape(G * FS, G * FS, 3)


def load_layouts(paths):
    names, lays = [], []
    for p in paths:
        b = np.load(p, allow_pickle=True)
        names += [str(x) for x in b["names"]]
        lays.append(b["lay"])
    return names, np.concatenate(lays)


class Boards(Dataset):
    """Returns the raw tiles (for the field), the assembled board, the target."""

    def __init__(self, names, lays, inv_by_name, augment=True):
        self.names, self.lays = names, lays
        self.inv, self.augment = inv_by_name, augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, k):
        nm = self.names[k]
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            self.inv[nm].astype(np.int64)]
        board = assemble(tiles, self.lays[k].astype(np.int64))
        tgt = load_rgb(Path(TRAIN_TGT) / nm)
        if self.augment:
            r = np.random.randint(4)
            if r:
                tiles = np.rot90(tiles, r, axes=(1, 2))
                board, tgt = np.rot90(board, r), np.rot90(tgt, r)
            if np.random.rand() < 0.5:
                tiles, board, tgt = tiles[:, :, ::-1], board[:, ::-1], tgt[:, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).float()
        b = torch.from_numpy(np.ascontiguousarray(board)).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(np.ascontiguousarray(tgt)).permute(2, 0, 1).float() / 255.0
        return t, b, y


@torch.no_grad()
def evaluate(model, field_model, names, lays, inv_by_name, dev, limit,
             nlm=False, swap=False, field_only=False):
    model.eval()
    gains, sd, ss_nlm = [], [], []
    n = min(limit, len(names))
    for k in range(n):
        nm = names[k]
        src = k if not swap else (k + 1) % n
        snm = names[src]
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            inv_by_name[nm].astype(np.int64)]
        stiles = tiles if src == k else to_frags(
            load_rgb(Path(TRAIN_INP) / snm)).astype(np.float32)[
                inv_by_name[snm].astype(np.int64)]
        board = assemble(stiles, lays[src].astype(np.int64))
        tgt = load_rgb(Path(TRAIN_TGT) / nm)

        t = torch.from_numpy(stiles).permute(0, 3, 1, 2)[None].to(dev)
        b = torch.from_numpy(board).to(dev, torch.float32).permute(2, 0, 1)[None] / 255.0
        fld = render(field_model(t))
        o = (fld if field_only else model(b, fld)).clamp(0, 1)
        out = np.rint(o.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
                      ).clip(0, 255).astype(np.uint8)
        flat = np.zeros_like(tgt)
        flat[:] = np.rint(tiles.reshape(-1, 3).mean(0)).clip(0, 255).astype(np.uint8)
        base = float(ssim_fn(flat, tgt, channel_axis=2, data_range=255))
        gains.append(float(ssim_fn(out, tgt, channel_axis=2, data_range=255)) - base)
        sd.append(float(out.astype(np.float32).std()))
        if nlm:
            ss_nlm.append(float(ssim_fn(rank96.fixed_nlm(out), tgt,
                                        channel_axis=2, data_range=255)) - base)
    model.train()
    return (float(np.mean(gains)), float(np.mean(sd)),
            float(np.mean(ss_nlm)) if ss_nlm else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="coarse_field_n8.pt")
    ap.add_argument("--train-layouts", nargs="+",
                    default=[str(Path(CACHE_DIR) / f"layouts_tr_{i}.npz")
                             for i in range(3)])
    ap.add_argument("--val-layouts", default=str(Path(CACHE_DIR) / "layouts_val.npz"))
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.84)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--eval-boards", type=int, default=40)
    ap.add_argument("--out", default="field_restore_v1.pt")
    a = ap.parse_args()

    dev = "cuda"
    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    inv_by_name = {str(n): v for n, v in zip(blob["names"], blob["inv"])}

    fk = torch.load(Path(CKPT_DIR) / a.field, map_location=dev, weights_only=False)
    fa = fk["args"]
    field_model = CoarseField(fa["n"], fa["ch"], fa["dim"], fa["hidden"]).to(dev)
    field_model.load_state_dict(fk["model"])
    field_model.eval()
    for p_ in field_model.parameters():
        p_.requires_grad_(False)
    print(f"field {a.field}: step {fk.get('step')}, its eval {fk.get('eval')}",
          flush=True)

    tr_names, tr_lays = load_layouts(a.train_layouts)
    va_names, va_lays = load_layouts([a.val_layouts])
    model = FieldRestore(base=a.base, depth=a.depth).to(dev)
    print(f"train {len(tr_names)} boards, val {len(va_names)}; FieldRestore "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    base_gain, base_sd, _ = evaluate(model, field_model, va_names, va_lays,
                                     inv_by_name, dev, a.eval_boards,
                                     field_only=True)
    print(f"the field alone on these boards: gain {base_gain:+.4f}, "
          f"out_std {base_sd:.1f} -- this run starts exactly here", flush=True)

    dl = DataLoader(Boards(tr_names, tr_lays, inv_by_name), batch_size=a.batch,
                    shuffle=True, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")

    best, step, t0, run = -9.0, 0, time.time(), []
    while step < a.steps:
        for t, b, y in dl:
            if step >= a.steps:
                break
            t, b, y = t.to(dev), b.to(dev), y.to(dev)
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                fld = render(field_model(t).float())
            with torch.autocast("cuda", torch.float16):
                pred = model(b, fld, clamp=False)
            loss = restore_loss(pred.float(), y, a.alpha)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(loss.item())
            step += 1
            if step % 200 == 0:
                print(f"step {step:6d}  loss {np.mean(run[-200:]):.4f}  "
                      f"{(time.time()-t0)/step:.2f}s/step", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                g, sd, _ = evaluate(model, field_model, va_names, va_lays,
                                    inv_by_name, dev, a.eval_boards)
                print(f"  eval step {step}: gain over flat {g:+.4f}  out_std "
                      f"{sd:.1f}  (field alone {base_gain:+.4f})", flush=True)
                if g > best:
                    best = g
                    torch.save({"model": model.state_dict(), "args": vars(a),
                                "step": step,
                                "eval": {"gain": g, "std": sd,
                                         "field_only": base_gain}},
                               Path(CKPT_DIR) / a.out)
    g, sd, g_nlm = evaluate(model, field_model, va_names, va_lays, inv_by_name,
                            dev, a.eval_boards, nlm=True)
    sw, sw_sd, _ = evaluate(model, field_model, va_names, va_lays, inv_by_name,
                            dev, a.eval_boards, swap=True)
    print(f"\nfinal: gain {g:+.4f}  (+NLM {g_nlm:+.4f})  out_std {sd:.1f}  "
          f"best {best:+.4f}  field alone {base_gain:+.4f}", flush=True)
    print(f"swapped input: gain {sw:+.4f}, out_std {sw_sd:.1f} -- the M146 "
          f"acceptance test, this must be clearly worse than {g:+.4f}", flush=True)


if __name__ == "__main__":
    main()
