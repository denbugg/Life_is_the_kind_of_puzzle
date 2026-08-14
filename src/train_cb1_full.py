"""P1/CB1 full FIT-only training after the registered G1 capacity pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from distort import distort_frags
from imgio import to_frags
from train_eval_cb1_g1_capacity import BoundaryBuddyNet, FIT_TARGETS, SPLIT, load_rgb, make_hard_batch, sha256_file

WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\full_fit")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=Path, default=FIT_TARGETS)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--queries-per-step", type=int, default=24)
    return p.parse_args()


def main() -> None:
    cfg = args()
    if cfg.steps != 6000 or cfg.queries_per_step != 24 or cfg.seed != 20260814:
        raise ValueError("CB1 full training is pre-registered at seed=20260814, steps=6000, queries=24")
    if not cfg.targets.is_dir() or not cfg.split.is_file():
        raise FileNotFoundError("CB1 full training requires FIT clean targets and pinned split manifest")
    split = json.loads(cfg.split.read_text(encoding="utf-8"))["splits"]
    fit = list(map(str, split["fit"]))
    if len(fit) != 5360:
        raise ValueError(f"expected 5360 FIT sources, got {len(fit)}")
    for name in fit[:4]:
        if not (cfg.targets / name).is_file():
            raise FileNotFoundError(cfg.targets / name)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = BoundaryBuddyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(cfg.seed)
    running: list[float] = []
    model.train()
    for step in range(cfg.steps):
        name = fit[int(rng.integers(0, len(fit)))]
        clean = load_rgb(cfg.targets / name)
        tiles = distort_frags(to_frags(clean), np.random.default_rng(cfg.seed * 1_000_000 + step))
        batch = make_hard_batch(tiles, np.random.default_rng(cfg.seed * 10_000_000 + step), cfg.queries_per_step)
        tensor = torch.from_numpy(batch.bands).to(device).reshape(-1, 3, 20, 4)
        logits = model(tensor).reshape(cfg.queries_per_step, 32)
        labels = torch.zeros((cfg.queries_per_step,), dtype=torch.long, device=device)
        listwise = F.cross_entropy(logits, labels)
        binary_labels = torch.zeros_like(logits); binary_labels[:, 0] = 1.0
        binary = F.binary_cross_entropy_with_logits(logits, binary_labels, pos_weight=torch.tensor(6.0, device=device))
        loss = listwise + 0.25 * binary
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite CB1 loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        running.append(float(loss.detach().cpu()))
        if (step + 1) % 500 == 0:
            print(json.dumps({"step": step + 1, "mean_loss_last500": float(np.mean(running[-500:]))}), flush=True)
    cfg.work.mkdir(parents=True, exist_ok=True)
    checkpoint = cfg.work / "cb1_full_fit.pt"
    torch.save({"state_dict": model.state_dict(), "seed": cfg.seed, "steps": cfg.steps, "queries_per_step": cfg.queries_per_step, "architecture": "BoundaryBuddyNet(width=48)"}, checkpoint)
    report = {
        "experiment": "P1_CB1_boundary_buddies", "stage": "full_FIT_training",
        "fit_sources": len(fit), "steps": cfg.steps, "queries_per_step": cfg.queries_per_step,
        "loss_first": running[0], "loss_last": running[-1], "loss_mean": float(np.mean(running)),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "split_manifest_sha256": sha256_file(cfg.split),
        "access": {"FIT_clean_sources_only": True, "CAL": False, "DEV": False, "test": False, "targets_outside_FIT": False},
        "layouts_assembled": False, "restorer_used": False,
        "decision": "ready_for_CB1_G2_CAL_candidate_graph_if_metadata_cache_contract_is_available",
    }
    destination = cfg.report or cfg.work / "cb1_full_fit_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
