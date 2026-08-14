"""P1/CB1 G1: bounded FIT-only matched-corruption boundary-buddy capacity test.

This harness trains a small directional boundary verifier with exact online
per-tile challenge corruption. Every 32-way list is one true neighbour plus
31 L1-hard false neighbours from the same corrupted bag. Held-out FIT sources
measure only listwise R@20 against the identical L1 hard-negative lists.
No CAL/DEV/test file, layout, restorer, or submission is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from config import FS, NFRAG
from distort import distort_frags
from imgio import to_frags

GRID = 24
FIT_TARGETS = Path(r"E:\pazzle_data\train\targets")
SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g1_capacity")


@dataclass(frozen=True)
class PairBatch:
    bands: np.ndarray       # [queries, 32, 3, 20, 4]
    l1_scores: np.ndarray   # [queries, 32], lower is better


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape != (GRID * FS, GRID * FS, 3) or image.dtype != np.uint8:
        raise ValueError(f"unexpected source image {path}: {image.shape} {image.dtype}")
    return image


def neighbor(anchor: int, direction: int) -> int:
    row, col = divmod(anchor, GRID)
    if direction == 0:  # right
        return anchor + 1 if col + 1 < GRID else -1
    if direction == 1:  # down
        return anchor + GRID if row + 1 < GRID else -1
    raise ValueError(direction)


def seam_l1(tiles: np.ndarray, anchor: int, candidate: np.ndarray, direction: int) -> np.ndarray:
    if direction == 0:
        left = tiles[anchor, :, -2:, :].astype(np.float32)
        right = tiles[candidate, :, :2, :].astype(np.float32)
    else:
        left = np.swapaxes(tiles[anchor, -2:, :, :], 0, 1).astype(np.float32)
        right = np.swapaxes(tiles[candidate, :2, :, :], 1, 2).swapaxes(1, 2).astype(np.float32)
        # Equivalent shape [candidate, 20, 2, 3] after transposition.
        right = np.swapaxes(tiles[candidate, :2, :, :], 1, 2).astype(np.float32)
    return np.abs(left[None] - right).mean(axis=(1, 2, 3))


def pair_band(tiles: np.ndarray, anchor: int, candidate: int, direction: int) -> np.ndarray:
    if direction == 0:
        a = tiles[anchor, :, -2:, :]
        b = tiles[candidate, :, :2, :]
    else:
        a = np.swapaxes(tiles[anchor, -2:, :, :], 0, 1)
        b = np.swapaxes(tiles[candidate, :2, :, :], 0, 1)
    x = np.concatenate((a, b), axis=1).astype(np.float32)
    # Normalize each tile half separately, preserving only local shape/chromatic structure.
    for lo, hi in ((0, 2), (2, 4)):
        half = x[:, lo:hi, :]
        center = np.median(half, axis=(0, 1), keepdims=True)
        scale = np.median(np.abs(half - center), axis=(0, 1), keepdims=True) * 1.4826 + 4.0
        x[:, lo:hi, :] = (half - center) / scale
    return np.transpose(x, (2, 0, 1))


def make_hard_batch(tiles: np.ndarray, rng: np.random.Generator, queries: int) -> PairBatch:
    bands: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for _ in range(queries):
        while True:
            anchor = int(rng.integers(0, NFRAG))
            direction = int(rng.integers(0, 2))
            positive = neighbor(anchor, direction)
            if positive >= 0:
                break
        candidates = np.arange(NFRAG, dtype=np.int32)
        candidates = candidates[candidates != anchor]
        score = seam_l1(tiles, anchor, candidates, direction)
        order = np.argsort(score, kind="stable")
        hard = candidates[order[candidates[order] != positive][:31]]
        if hard.shape != (31,):
            raise RuntimeError("unable to construct 31 hard negatives")
        members = np.concatenate((np.asarray([positive], dtype=np.int32), hard))
        scores = seam_l1(tiles, anchor, members, direction)
        bands.append(np.stack([pair_band(tiles, anchor, int(c), direction) for c in members]))
        all_scores.append(scores)
    return PairBatch(np.stack(bands).astype(np.float32), np.stack(all_scores).astype(np.float32))


class BoundaryBuddyNet(nn.Module):
    def __init__(self, width: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False), nn.GroupNorm(6, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.GroupNorm(6, width), nn.SiLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(width, width * 2, 3, padding=1, bias=False), nn.GroupNorm(6, width * 2), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def ranks(scores: np.ndarray, descending: bool) -> np.ndarray:
    order = np.argsort(-scores if descending else scores, axis=1, kind="stable")
    out = np.empty_like(order)
    out[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[1])[None, :]
    return out


def evaluate(model: BoundaryBuddyNet, names: Iterable[str], device: torch.device, seed: int, queries: int) -> dict[str, float]:
    model.eval()
    cb1_rank, l1_rank, count = [], [], 0
    with torch.no_grad():
        for offset, name in enumerate(names):
            clean = load_rgb(FIT_TARGETS / name)
            tiles = distort_frags(to_frags(clean), np.random.default_rng(seed + offset))
            batch = make_hard_batch(tiles, np.random.default_rng(seed * 1000 + offset), queries)
            tensor = torch.from_numpy(batch.bands).to(device).reshape(-1, 3, FS, 4)
            score = model(tensor).reshape(batch.bands.shape[:2]).cpu().numpy()
            cb1_rank.append(ranks(score, True)[:, 0])
            l1_rank.append(ranks(batch.l1_scores, False)[:, 0])
            count += batch.bands.shape[0]
    a, b = np.concatenate(cb1_rank), np.concatenate(l1_rank)
    return {
        "queries": int(count),
        "cb1_r1": float(np.mean(a < 1)), "cb1_r20": float(np.mean(a < 20)), "cb1_mean_rank": float(a.mean()),
        "l1_r1": float(np.mean(b < 1)), "l1_r20": float(np.mean(b < 20)), "l1_mean_rank": float(b.mean()),
    }


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=Path, default=FIT_TARGETS)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--queries-per-step", type=int, default=12)
    p.add_argument("--eval-scenes", type=int, default=4)
    p.add_argument("--eval-queries", type=int, default=96)
    return p.parse_args()


def main() -> None:
    cfg = args()
    if cfg.steps != 240 or cfg.queries_per_step != 12 or cfg.eval_scenes != 4 or cfg.eval_queries != 96:
        raise ValueError("CB1 G1 is pre-registered at 240 steps, 12 train queries, 4 held-out FIT scenes and 96 eval queries")
    if not cfg.targets.is_dir() or not cfg.split.is_file():
        raise FileNotFoundError("CB1 G1 requires FIT clean targets and the pinned split manifest")
    split = json.loads(cfg.split.read_text(encoding="utf-8"))["splits"]
    fit = list(map(str, split["fit"]))
    heldout = fit[:cfg.eval_scenes]
    train = fit[cfg.eval_scenes:]
    for name in heldout + train[:1]:
        if not (cfg.targets / name).is_file():
            raise FileNotFoundError(cfg.targets / name)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = BoundaryBuddyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(cfg.seed)
    losses = []
    model.train()
    for step in range(cfg.steps):
        name = train[int(rng.integers(0, len(train)))]
        clean = load_rgb(cfg.targets / name)
        tiles = distort_frags(to_frags(clean), np.random.default_rng(cfg.seed * 1_000_000 + step))
        batch = make_hard_batch(tiles, np.random.default_rng(cfg.seed * 10_000_000 + step), cfg.queries_per_step)
        tensor = torch.from_numpy(batch.bands).to(device).reshape(-1, 3, FS, 4)
        logits = model(tensor).reshape(cfg.queries_per_step, 32)
        labels = torch.zeros((cfg.queries_per_step,), dtype=torch.long, device=device)
        listwise = F.cross_entropy(logits, labels)
        binary_labels = torch.zeros_like(logits); binary_labels[:, 0] = 1.0
        binary = F.binary_cross_entropy_with_logits(logits, binary_labels, pos_weight=torch.tensor(6.0, device=device))
        loss = listwise + 0.25 * binary
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CB1 loss")
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    metrics = evaluate(model, heldout, device, cfg.seed + 99, cfg.eval_queries)
    cfg.work.mkdir(parents=True, exist_ok=True)
    checkpoint = cfg.work / "cb1_g1_capacity.pt"
    torch.save({"state_dict": model.state_dict(), "seed": cfg.seed, "steps": cfg.steps}, checkpoint)
    report = {
        "experiment": "P1_CB1_boundary_buddies", "gate": "G1_bounded_FIT_capacity",
        "fit_train_sources": len(train), "heldout_fit_sources": heldout, "training_steps": cfg.steps,
        "queries_per_step": cfg.queries_per_step, "hard_list_width": 32,
        "loss_first": losses[0], "loss_last": losses[-1], "loss_mean": float(np.mean(losses)),
        "metrics": metrics, "cb1_minus_l1_r20": float(metrics["cb1_r20"] - metrics["l1_r20"]),
        "targets_opened": {"FIT_clean_sources_only": True, "CAL": False, "DEV": False, "test": False},
        "layouts_assembled": False, "restorer_used": False,
        "split_manifest_sha256": sha256_file(cfg.split), "checkpoint_sha256": sha256_file(checkpoint),
    }
    report["passes_G1"] = bool(metrics["cb1_r20"] > metrics["l1_r20"])
    report["decision"] = "advance_to_CB1_full_train_and_CAL_graph" if report["passes_G1"] else "reject_CB1_before_full_train"
    destination = cfg.report or cfg.work / "cb1_g1_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
