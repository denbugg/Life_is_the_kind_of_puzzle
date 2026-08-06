"""Train and gate the Canvas-first, instance-conditioned assignment prototype.

This is intentionally not a replacement for the production pipeline yet.  It is
an experiment with two falsifiable outputs:

1. Can an unordered bag predict its own ordered low-frequency clean canvas?
2. Does that predicted, image-specific canvas turn tile placement into a useful
   unary assignment problem?

Real train pairs supervise only the canvas/prototype target and never use the
noisy recovered permutation cache.  Exact assignment labels come solely from
on-the-fly synthetic corruptions of clean targets.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from canvas_metrics import canvas_patches, decoded_geometry, hard_assignment, rank_summary, symmetric_assignment_ce
from canvas_model import CanvasNet
from config import NFRAG, SEED
from imgio import from_frags, train_val_split


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-6)


def _photometric_descriptor(x: torch.Tensor) -> torch.Tensor:
    """Per-patch contrast-invariant descriptor used by the proven oracle test."""
    return _norm(x - x.mean(dim=-1, keepdim=True))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CanvasSystem(nn.Module):
    """CanvasNet plus the three matching views used by the gate.

    - ``oracle_logits`` compares a corrupted tile to the true clean target cell
      and validates cross-domain matching separately from canvas prediction.
    - ``slot_logits`` compares it to an instance-conditioned slot latent.
    - ``canvas_logits`` compares it to the *predicted RGB canvas* itself; this
      is the deployment-relevant signal.
    """

    def __init__(self, patch: int = 4, d: int = 128, match_dim: int = 64) -> None:
        super().__init__()
        self.patch = int(patch)
        self.backbone = CanvasNet(patch=patch, d=d)
        self.tile_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, match_dim))
        self.slot_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, match_dim))
        raw_dim = 3 * patch * patch
        self.clean_head = nn.Sequential(
            nn.LayerNorm(raw_dim), nn.Linear(raw_dim, d), nn.GELU(), nn.Linear(d, match_dim)
        )
        # A learned inverse temperature makes normalized dot products useful for
        # CE without allowing an unbounded numerical shortcut.
        self.logit_log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def clean_features(self, target_patches: torch.Tensor) -> torch.Tensor:
        # Dataset layout: (B, 576, patch_y, patch_x, RGB).
        raw = target_patches.permute(0, 1, 4, 2, 3).reshape(target_patches.shape[0], NFRAG, -1)
        return _norm(self.clean_head(raw))

    def raw_features_from_patches(self, patches: torch.Tensor) -> torch.Tensor:
        raw = patches.permute(0, 1, 4, 2, 3).reshape(patches.shape[0], NFRAG, -1)
        return _photometric_descriptor(raw)

    def raw_features_from_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = tiles.shape
        side = int(round((h / self.patch)))
        if h != w or h % self.patch:
            raise ValueError(f"tile side {h} must be divisible by canvas patch {self.patch}")
        raw = F.avg_pool2d(tiles.reshape(b * n, c, h, w), side).reshape(b, n, -1)
        return _photometric_descriptor(raw)

    def forward(self, tiles: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(tiles)
        if not {"canvas", "tile_tokens", "slot_tokens"}.issubset(out):
            raise KeyError("CanvasNet must return canvas, tile_tokens and slot_tokens")
        canvas = out["canvas"]
        tile = _norm(self.tile_head(out["tile_tokens"]))
        slot = _norm(self.slot_head(out["slot_tokens"]))
        predicted_raw = canvas_patches(canvas, self.patch)
        canvas_feature = _norm(self.clean_head(predicted_raw))
        raw_tile = self.raw_features_from_tiles(tiles)
        raw_canvas = _photometric_descriptor(predicted_raw)
        scale = self.logit_log_scale.exp().clamp(1.0, 50.0)
        return {
            "canvas": canvas,
            "tile": tile,
            "slot": slot,
            "canvas_feature": canvas_feature,
            "raw_tile": raw_tile,
            "raw_canvas": raw_canvas,
            "scale": scale,
            "slot_logits": scale * (tile @ slot.transpose(1, 2)),
            "canvas_logits": scale * (tile @ canvas_feature.transpose(1, 2)),
            "raw_canvas_logits": scale * (raw_tile @ raw_canvas.transpose(1, 2)),
        }


def loss_terms(system: CanvasSystem, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], args: Any, step: int) -> tuple[torch.Tensor, Dict[str, float]]:
    target_canvas = batch["canvas"].to(output["canvas"].device, non_blocking=True)
    target_patches = batch["target_patches"].to(output["canvas"].device, non_blocking=True)
    clean_feature = system.clean_features(target_patches)
    canvas_loss = F.smooth_l1_loss(output["canvas"], target_canvas)
    proto_loss = 1.0 - (output["slot"] * clean_feature).sum(-1).mean()
    total = args.w_canvas * canvas_loss + args.w_proto * proto_loss
    terms: Dict[str, torch.Tensor] = {
        "canvas": canvas_loss,
        "proto": proto_loss,
        "oracle": canvas_loss.new_zeros(()),
        "slot": canvas_loss.new_zeros(()),
        "pred": canvas_loss.new_zeros(()),
        "raw_pred": canvas_loss.new_zeros(()),
    }
    synthetic = batch["has_perm"].to(output["canvas"].device, non_blocking=True).bool()
    if bool(synthetic.any()):
        perm = batch["perm"].to(output["canvas"].device, non_blocking=True).long()[synthetic]
        oracle_logits = output["scale"] * (output["tile"][synthetic] @ clean_feature[synthetic].transpose(1, 2))
        slot_logits = output["slot_logits"][synthetic]
        pred_logits = output["canvas_logits"][synthetic]
        raw_pred_logits = output["raw_canvas_logits"][synthetic]
        terms["oracle"] = symmetric_assignment_ce(oracle_logits, perm)
        terms["slot"] = symmetric_assignment_ce(slot_logits, perm)
        terms["pred"] = symmetric_assignment_ce(pred_logits, perm)
        terms["raw_pred"] = symmetric_assignment_ce(raw_pred_logits, perm)
        pred_ramp = min(1.0, (step + 1) / max(1, args.pred_warmup))
        total = total + args.w_oracle * terms["oracle"] + args.w_slot * terms["slot"]
        total = total + args.w_pred * pred_ramp * terms["pred"]
        total = total + args.w_raw_pred * pred_ramp * terms["raw_pred"]
    values = {k: float(v.detach()) for k, v in terms.items()}
    values["total"] = float(total.detach())
    return total, values


def make_loader(ds: CanvasDataset, batch_size: int, workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    kw: Dict[str, Any] = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": workers,
                          "pin_memory": device.type == "cuda", "drop_last": shuffle}
    if workers:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(ds, **kw)


def _blend_logits(output: Dict[str, torch.Tensor], slot_blend: float, raw_blend: float) -> torch.Tensor:
    canvas_score = raw_blend * output["raw_canvas_logits"] + (1.0 - raw_blend) * output["canvas_logits"]
    return slot_blend * output["slot_logits"] + (1.0 - slot_blend) * canvas_score


def _solve_ssim(tiles: torch.Tensor, clean: torch.Tensor, logits: torch.Tensor) -> list[float]:
    """Direct image score; works for real pairs without a recovered permutation."""
    tiles_np = tiles.detach().float().cpu().permute(0, 1, 3, 4, 2).numpy()
    clean_np = clean.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    out: list[float] = []
    for frags, target, score in zip(tiles_np, clean_np, logits):
        place = hard_assignment(score)
        assembled = from_frags(frags[place])
        out.append(float(sk_ssim(target, assembled, channel_axis=2, data_range=1.0)))
    return out


@torch.no_grad()
def evaluate(
    system: CanvasSystem,
    loader: DataLoader,
    device: torch.device,
    args: Any,
    *,
    max_images: int,
) -> Dict[str, float]:
    system.eval()
    vals: defaultdict[str, list[float]] = defaultdict(list)
    seen = 0
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    for batch in loader:
        if seen >= max_images:
            break
        tiles = batch["tiles"].to(device, non_blocking=True)
        with amp:
            output = system(tiles)
        score = _blend_logits(output, args.slot_blend, args.raw_blend).float()
        target_canvas = batch["canvas"].to(device, non_blocking=True)
        vals["canvas_l1"].append(float(F.l1_loss(output["canvas"].float(), target_canvas).cpu()))
        vals["solve_ssim"].extend(_solve_ssim(tiles, batch["clean"], score))
        synthetic = batch["has_perm"].bool()
        if bool(synthetic.any()):
            perm = batch["perm"][synthetic].to(device)
            clean_feature = system.clean_features(batch["target_patches"].to(device))[synthetic]
            oracle_logits = output["scale"] * (output["tile"][synthetic] @ clean_feature.transpose(1, 2))
            raw_target = system.raw_features_from_patches(batch["target_patches"].to(device))[synthetic]
            raw_oracle_logits = output["scale"] * (output["raw_tile"][synthetic] @ raw_target.transpose(1, 2))
            # Report both deployment canvas and slot signal, so a good latent
            # cannot hide a poor actual canvas.
            for prefix, logits in (("raw_oracle", raw_oracle_logits.float()), ("oracle", oracle_logits.float()),
                                   ("raw_canvas", output["raw_canvas_logits"][synthetic].float()),
                                   ("pred", score[synthetic]), ("slot", output["slot_logits"][synthetic].float())):
                rank = rank_summary(logits, perm)
                geo = decoded_geometry(logits, perm)
                for k, v in rank.items():
                    vals[f"{prefix}_{k}"].append(v)
                vals[f"{prefix}_place_acc"].append(geo["place_acc"])
                vals[f"{prefix}_neighbour_acc"].append(geo["neighbour_acc"])
        seen += int(tiles.shape[0])
    system.train()
    return {k: float(np.mean(v)) for k, v in vals.items() if v}


def checkpoint(path: str, system: CanvasSystem, optimizer: torch.optim.Optimizer, step: int, args: Any, metrics: Dict[str, float]) -> None:
    torch.save({"model": system.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "args": vars(args), "metrics": metrics}, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8_000)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train_n", type=int, default=0, help="0 = all 6,700 training images")
    ap.add_argument("--eval_n", type=int, default=12)
    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--match_dim", type=int, default=64)
    ap.add_argument("--real_prob", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tag", default="canvas")
    ap.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "canvas"),
        help="checkpoint directory (workspace-local by default)",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--w_canvas", type=float, default=1.0)
    ap.add_argument("--w_proto", type=float, default=0.50)
    ap.add_argument("--w_oracle", type=float, default=0.35)
    ap.add_argument("--w_slot", type=float, default=0.75)
    ap.add_argument("--w_pred", type=float, default=0.50)
    ap.add_argument("--w_raw_pred", type=float, default=0.75)
    ap.add_argument("--pred_warmup", type=int, default=800)
    ap.add_argument("--slot_blend", type=float, default=0.5, help="1=latent slots, 0=predicted RGB canvas")
    ap.add_argument("--raw_blend", type=float, default=0.7, help="within canvas score: 1=fixed raw descriptor")
    args = ap.parse_args()
    if not 0.0 <= args.real_prob <= 1.0:
        ap.error("--real_prob must be in [0, 1]")
    if not 0.0 <= args.slot_blend <= 1.0:
        ap.error("--slot_blend must be in [0, 1]")
    if not 0.0 <= args.raw_blend <= 1.0:
        ap.error("--raw_blend must be in [0, 1]")
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    train_names, val_names = train_val_split()
    if args.train_n:
        train_names = train_names[:args.train_n]
    train_ds = CanvasDataset(train_names, patch=args.patch, real_prob=args.real_prob, seed=args.seed)
    # Synthetic validation is exact; real validation is scoreable by target SSIM
    # but intentionally has no placement labels.
    val_syn = CanvasDataset(val_names, patch=args.patch, real_prob=0.0, seed=args.seed + 10_000)
    val_real = CanvasDataset(val_names, patch=args.patch, real_prob=1.0, seed=args.seed + 20_000)
    train_dl = make_loader(train_ds, args.bs, args.workers, True, device)
    val_syn_dl = make_loader(val_syn, args.bs, min(args.workers, 2), False, device)
    val_real_dl = make_loader(val_real, args.bs, min(args.workers, 2), False, device)

    system = CanvasSystem(args.patch, args.d, args.match_dim).to(device)
    print(f"CanvasSystem params={count_params(system):,}", flush=True)
    opt = torch.optim.AdamW(system.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.08)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()

    best = -float("inf")
    t0 = time.time()
    iterator = iter(train_dl)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_dl)
            batch = next(iterator)
        tiles = batch["tiles"].to(device, non_blocking=True)
        with amp:
            output = system(tiles)
            loss, terms = loss_terms(system, output, batch, args, step)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()

        if step % 50 == 0:
            rate = (time.time() - t0) / max(1, step)
            print("step {}/{} loss {:.4f} canvas {:.4f} proto {:.4f} oracle {:.3f} slot {:.3f} pred {:.3f} raw {:.3f} lr {:.2e} {:.2f}s/it".format(
                step, args.steps, terms["total"], terms["canvas"], terms["proto"], terms["oracle"],
                terms["slot"], terms["pred"], terms["raw_pred"], sched.get_last_lr()[0], rate), flush=True)
        if step > 0 and step % args.eval_every == 0:
            syn = evaluate(system, val_syn_dl, device, args, max_images=args.eval_n)
            real = evaluate(system, val_real_dl, device, args, max_images=args.eval_n)
            print("[SYN] " + " ".join(f"{k}={v:.3f}" for k, v in sorted(syn.items())), flush=True)
            print("[REAL] " + " ".join(f"{k}={v:.3f}" for k, v in sorted(real.items())), flush=True)
            checkpoint(os.path.join(args.out_dir, f"{args.tag}_last.pt"), system, opt, step, args, {"syn": syn, "real": real})
            score = real.get("solve_ssim", -float("inf"))
            if score > best:
                best = score
                checkpoint(os.path.join(args.out_dir, f"{args.tag}_best.pt"), system, opt, step, args, {"syn": syn, "real": real})
                print(f"saved best real solve_ssim={best:.4f}", flush=True)
    print("final synthetic evaluation:", evaluate(system, val_syn_dl, device, args, max_images=args.eval_n), flush=True)
    print("final real evaluation:", evaluate(system, val_real_dl, device, args, max_images=args.eval_n), flush=True)


if __name__ == "__main__":
    main()
