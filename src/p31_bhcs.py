"""P31 BHCS-24: seam-only boundary hard-contrastive scorer.
G0/G1 are input-only. G2 is bounded FIT-train only and compares against frozen recall@20 baseline.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

from p29_dpcg import N, load_tiles
from p12_loop_consensus import solve_buddies_from_scores
import p13_component_pose as p13

T = 20
EDGE = 4
FIT_EPOCH_CAP = 40
FIT_SAMPLES_PER_EPOCH_CAP = 2_500_000
PAIR_BATCH_CAP = 16_384


class SeamCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 48, 3, padding=1), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def bad(*paths: Path) -> None:
    if "p8" in "\n".join(str(p).lower() for p in paths):
        raise RuntimeError("P8 prohibited")


def seed_from(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFF


def orient_tiles(t: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0:
        return t
    if d == 1:
        return t.transpose(-2, -1).contiguous()
    raise ValueError(d)


def seam_parts(t: torch.Tensor, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = orient_tiles(t, d)
    return x[:, :, :, -EDGE:].contiguous(), x[:, :, :, :EDGE].contiguous()


def seam(t: torch.Tensor, i: int, j: int, d: int) -> torch.Tensor:
    if d == 2:
        return seam(t, j, i, 0)
    if d == 3:
        return seam(t, j, i, 1)
    src, tgt = seam_parts(t, d)
    return torch.cat((src[i], tgt[j]), dim=-1)


def labels(label_dir: Path, source: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(label_dir / (Path(source).stem + ".npz"), allow_pickle=False) as z:
        p = z["target_tile_to_slot"].astype(np.int32)
        src = str(z["source"])
    if src != source or p.shape != (N,) or np.unique(p).size != N:
        raise RuntimeError("invalid cached FIT labels")
    inv = np.empty(N, np.int32)
    inv[p] = np.arange(N, dtype=np.int32)
    return p, inv


def neighbor(tile_to_slot: np.ndarray, slot_to_tile: np.ndarray, i: int, d: int) -> int:
    rr, cc = divmod(int(slot_to_tile[i]), 24)
    if d == 0:
        cc += 1
    elif d == 1:
        rr += 1
    elif d == 2:
        cc -= 1
    else:
        rr -= 1
    return -1 if rr < 0 or rr >= 24 or cc < 0 or cc >= 24 else int(tile_to_slot[rr * 24 + cc])


def frozen_recall20(score_dir: Path, label_dir: Path, source: str) -> float:
    cands, valid, scores = p13.load_score_cache(score_dir, source)
    tile_to_slot, slot_to_tile = labels(label_dir, source)
    hit = 0
    tot = 0
    for d in range(4):
        for i in range(N):
            q = neighbor(tile_to_slot, slot_to_tile, i, d)
            if q < 0:
                continue
            row = cands[i, valid[i]].astype(np.int32)
            ord_idx = np.argsort(-scores[d, i, valid[i]], kind="stable")
            top = row[ord_idx[:20]]
            hit += int(q in top)
            tot += 1
    return hit / tot


def learned_recall20(right: np.ndarray, down: np.ndarray, label_dir: Path, source: str) -> float:
    mats = (right, down, right.T, down.T)
    tile_to_slot, slot_to_tile = labels(label_dir, source)
    hit = 0
    tot = 0
    for d in range(4):
        top = np.argsort(-mats[d], axis=1, kind="stable")[:, :20]
        for i in range(N):
            q = neighbor(tile_to_slot, slot_to_tile, i, d)
            if q < 0:
                continue
            hit += int(q in top[i])
            tot += 1
    return hit / tot


def g0(_: argparse.Namespace) -> dict:
    t = torch.arange(4 * 3 * T * T, dtype=torch.float32).reshape(4, 3, T, T)
    x = seam(t, 0, 1, 0)
    y = seam(t, 0, 1, 1)
    raw = np.full((2, N, N), -20.0, np.float32)
    for r in range(24):
        for c in range(24):
            i = r * 24 + c
            if c < 23:
                raw[0, i, i + 1] = 20.0
            if r < 23:
                raw[1, i, i + 24] = 20.0
    for k in range(2):
        np.fill_diagonal(raw[k], -np.inf)
    p, _ = solve_buddies_from_scores(raw[0], raw[1], max_edges=2208, min_margin=0.0, repair_passes=2)
    p = np.asarray(p, np.int32)
    return {
        "experiment": "P31_BHCS24",
        "gate": "G0",
        "seam_shape": list(x.shape),
        "vertical_shape": list(y.shape),
        "valid_bijection": bool(p.shape == (N,) and np.unique(p).size == N),
        "exact_synthetic": bool(np.array_equal(p, np.arange(N, dtype=np.int32))),
        "finite_seam": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
        "passes_G0": bool(torch.isfinite(x).all() and torch.isfinite(y).all() and np.array_equal(p, np.arange(N, dtype=np.int32))),
    }


def g1(a: argparse.Namespace) -> dict:
    dev = torch.device("cuda")
    m = SeamCNN().to(dev).eval()
    rows = []
    with torch.no_grad():
        for k, n in enumerate(a.sources):
            st = time.perf_counter()
            t = load_tiles(a.inputs, n).to(dev)
            a0 = torch.arange(16, device=dev)
            b0 = (a0 + 1) % N
            src, tgt = seam_parts(t, 0)
            x = torch.cat((src[a0], tgt[b0]), dim=-1)
            z = m(x)
            dt = time.perf_counter() - st
            ok = bool(x.shape == (16, 3, T, 2 * EDGE) and torch.isfinite(z).all() and dt <= 90.0)
            rows.append({"source": n, "seconds": dt, "shape": list(x.shape), "ok": ok})
            if (k + 1) % 4 == 0:
                print(json.dumps({"stage": "g1", "done": k + 1, "total": len(a.sources)}), flush=True)
    return {
        "experiment": "P31_BHCS24",
        "gate": "G1",
        "rows": rows,
        "labels_used": False,
        "targets_opened": False,
        "p8_imported": False,
        "passes_G1": bool(all(r["ok"] for r in rows)),
    }


def prepare_examples(inputs: Path, label_dir: Path, sources: list[str], neg_hard: int, neg_rand: int) -> tuple[dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]], list[tuple[str, int, int, int, np.ndarray]]]:
    caches: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    examples: list[tuple[str, int, int, int, np.ndarray]] = []
    for idx, source in enumerate(sources, start=1):
        t = load_tiles(inputs, source).float().cpu()
        caches[source] = {0: seam_parts(t, 0), 1: seam_parts(t, 1)}
        tile_to_slot, slot_to_tile = labels(label_dir, source)
        for d in (0, 1):
            src_e, tgt_e = caches[source][d]
            for i in range(N):
                q = neighbor(tile_to_slot, slot_to_tile, i, d)
                if q < 0:
                    continue
                dist = ((src_e[i:i+1] - tgt_e) ** 2).mean(dim=(1, 2, 3)).numpy()
                dist[i] = np.inf
                dist[q] = np.inf
                order = np.argsort(dist, kind="stable")
                hard = order[:neg_hard].astype(np.int32)
                pool = order[neg_hard:]
                rng = np.random.default_rng(seed_from(source, d, i))
                rcount = min(neg_rand, len(pool))
                rand = rng.choice(pool, size=rcount, replace=False).astype(np.int32) if rcount else np.empty(0, np.int32)
                neg = np.concatenate((hard, rand)).astype(np.int32)
                examples.append((source, d, i, q, neg))
        if idx % 4 == 0:
            print(json.dumps({"stage": "prepare_examples", "done": idx, "total": len(sources)}), flush=True)
    return caches, examples


def make_batch(caches: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]], batch: list[tuple[str, int, int, int, np.ndarray]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pos = []
    neg = []
    for source, d, i, q, ng in batch:
        src_e, tgt_e = caches[source][d]
        pos.append(torch.cat((src_e[i], tgt_e[q]), dim=-1))
        neg.append(torch.cat((src_e[i].expand(len(ng), -1, -1, -1), tgt_e[torch.from_numpy(ng)]), dim=-1))
    pos_t = torch.stack(pos).to(device)
    neg_t = torch.cat(neg, dim=0).to(device)
    return pos_t, neg_t


def train_fit(a: argparse.Namespace, caches: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]], examples: list[tuple[str, int, int, int, np.ndarray]]) -> tuple[SeamCNN, dict]:
    device = torch.device("cuda")
    model = SeamCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    epochs = min(a.epochs, FIT_EPOCH_CAP)
    if len(examples) * 9 > FIT_SAMPLES_PER_EPOCH_CAP:
        keep = FIT_SAMPLES_PER_EPOCH_CAP // 9
        examples = examples[:keep]
    start = time.perf_counter()
    hist = []
    for epoch in range(epochs):
        rng = np.random.default_rng(seed_from("p31", epoch))
        order = rng.permutation(len(examples))
        model.train()
        loss_sum = 0.0
        seen = 0
        for off in range(0, len(order), a.batch):
            idxs = order[off:off + a.batch]
            batch = [examples[int(i)] for i in idxs]
            pos_t, neg_t = make_batch(caches, batch, device)
            pos_s = model(pos_t)
            neg_s = model(neg_t).reshape(len(batch), -1)
            loss = torch.relu(a.margin - pos_s[:, None] + neg_s).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(batch)
            seen += len(batch)
        hist.append({"epoch": epoch + 1, "loss": loss_sum / max(seen, 1)})
        print(json.dumps({"stage": "fit", "epoch": epoch + 1, "epochs": epochs, "loss": hist[-1]["loss"]}), flush=True)
        if time.perf_counter() - start > a.train_minutes * 60.0:
            break
    return model.eval(), {"epochs_run": len(hist), "history": hist, "train_minutes": (time.perf_counter() - start) / 60.0, "example_count": len(examples)}


def score_direction(model: SeamCNN, src_e: torch.Tensor, tgt_e: torch.Tensor, device: torch.device, row_chunk: int) -> np.ndarray:
    model.eval()
    out = np.empty((N, N), np.float32)
    with torch.no_grad():
        src_e = src_e.to(device)
        tgt_e = tgt_e.to(device)
        for r0 in range(0, N, row_chunk):
            r1 = min(r0 + row_chunk, N)
            left = src_e[r0:r1][:, None].expand(r1 - r0, N, 3, T, EDGE)
            right = tgt_e[None, :].expand(r1 - r0, N, 3, T, EDGE)
            seam_t = torch.cat((left, right), dim=-1).reshape(-1, 3, T, 2 * EDGE)
            vals = []
            for p0 in range(0, seam_t.shape[0], PAIR_BATCH_CAP):
                vals.append(model(seam_t[p0:p0 + PAIR_BATCH_CAP]).float().cpu())
            out[r0:r1] = torch.cat(vals).numpy().reshape(r1 - r0, N)
        np.fill_diagonal(out, -np.inf)
    return out


def g2(a: argparse.Namespace) -> dict:
    bad(a.inputs, a.label_dir, a.score_dir, a.manifest, a.work)
    train_sources, _ = p13.source_lists(a.manifest)
    fit = sorted(train_sources)[:96]
    caches, examples = prepare_examples(a.inputs, a.label_dir, fit, neg_hard=4, neg_rand=4)
    model, fit_info = train_fit(a, caches, examples)
    device = torch.device("cuda")
    learned = []
    frozen = []
    for k, source in enumerate(fit, start=1):
        src0, tgt0 = caches[source][0]
        src1, tgt1 = caches[source][1]
        r = score_direction(model, src0, tgt0, device, a.row_chunk)
        d = score_direction(model, src1, tgt1, device, a.row_chunk)
        learned.append(learned_recall20(r, d, a.label_dir, source))
        frozen.append(frozen_recall20(a.score_dir, a.label_dir, source))
        print(json.dumps({"stage": "g2_eval", "done": k, "total": len(fit), "source": source}), flush=True)
    learned_mean = float(np.mean(learned))
    frozen_mean = float(np.mean(frozen))
    gain_pp = 100.0 * (learned_mean - frozen_mean)
    return {
        "experiment": "P31_BHCS24",
        "gate": "G2",
        "fit_info": fit_info,
        "frozen_recall20": frozen_mean,
        "learned_recall20": learned_mean,
        "gain_pp": gain_pp,
        "labels_used": True,
        "targets_opened": False,
        "p8_imported": False,
        "passes_G2": bool(gain_pp >= 1.0),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("g0", "g1", "g2"), required=True)
    p.add_argument("--inputs", type=Path, default=Path(r"E:\pazzle_data\train\inputs"))
    p.add_argument("--label-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache"))
    p.add_argument("--score-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache"))
    p.add_argument("--manifest", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json"))
    p.add_argument("--work", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P31_bhcs"))
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--train-minutes", type=float, default=20.0)
    p.add_argument("--row-chunk", type=int, default=24)
    p.add_argument("--sources", nargs="*", default=(
        "img_000002.png", "img_000025.png", "img_000098.png", "img_000168.png",
        "img_000172.png", "img_000194.png", "img_000223.png", "img_000243.png",
        "img_000267.png", "img_000304.png", "img_000344.png", "img_000384.png",
        "img_000426.png", "img_000457.png", "img_000480.png", "img_000513.png",
    ))
    a = p.parse_args()
    bad(a.inputs, a.label_dir, a.score_dir, a.manifest, a.work)
    a.work.mkdir(parents=True, exist_ok=True)
    rep = g0(a) if a.mode == "g0" else g1(a) if a.mode == "g1" else g2(a)
    out = a.work / f"p31_{a.mode}_report.json"
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep), flush=True)
    if not rep[f"passes_{a.mode.upper()}"]:
        raise RuntimeError(f"P31 {a.mode} rejected")


if __name__ == "__main__":
    main()
