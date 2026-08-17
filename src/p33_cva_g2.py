"""P33 CVA-24 G2: FIT-train-only cycle-verified agglomeration coverage gate."""
from __future__ import annotations
import argparse, json, time, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p29_dpcg import N, load_tiles, desc, model, topm
import p13_component_pose as p13

RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3
OPP = {RIGHT: LEFT, DOWN: UP, LEFT: RIGHT, UP: DOWN}
GRID = 24
MAX_UNION = 128
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def bad(*x):
    if "p8" in "\n".join(map(lambda q: str(q).lower(), x)):
        raise RuntimeError("P8 prohibited")


def load_label(cache_dir: Path, source: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_dir / f"{Path(source).stem}.npz", allow_pickle=False) as z:
        po = z["target_tile_to_slot"].astype(np.int32, copy=False)
        src = str(z["source"])
    if src != source or po.shape != (N,) or np.unique(po).size != N:
        raise RuntimeError("invalid FIT label cache")
    inv = np.empty(N, np.int32)
    inv[po] = np.arange(N, dtype=np.int32)
    return po, inv


def neighbor_tile(tile_to_slot: np.ndarray, slot_to_tile: np.ndarray, src: int, direction: int) -> int:
    slot = int(tile_to_slot[src])
    r, c = divmod(slot, GRID)
    if direction == RIGHT:
        c += 1
    elif direction == DOWN:
        r += 1
    elif direction == LEFT:
        c -= 1
    elif direction == UP:
        r -= 1
    else:
        raise ValueError(direction)
    if r < 0 or r >= GRID or c < 0 or c >= GRID:
        return -1
    return int(tile_to_slot[r * GRID + c])


def dense_candidates(dino, inputs: Path, source: str, device: torch.device) -> list[np.ndarray]:
    z = desc(dino, load_tiles(inputs, source), device)
    bands = torch.stack([z[:, :, -1, :], z[:, -1, :, :], z[:, :, 0, :], z[:, 0, :, :]])
    return [topm(bands[d], bands[OPP[d]], 64)[0] for d in range(4)]


def build_union(frozen_cands: np.ndarray, frozen_valid: np.ndarray, dense: list[np.ndarray]) -> list[list[list[int]]]:
    out: list[list[list[int]]] = [[[] for _ in range(N)] for _ in range(4)]
    for d in range(4):
        for i in range(N):
            row: list[int] = []
            for q in list(map(int, dense[d][i])) + list(map(int, frozen_cands[i, frozen_valid[i]])):
                if q != i and q not in row:
                    row.append(q)
                if len(row) == MAX_UNION:
                    break
            out[d][i] = row
    return out


def candidate_features(union: list[list[list[int]]], frozen_cands: np.ndarray, frozen_valid: np.ndarray, frozen_scores: np.ndarray, src: int, direction: int, dst: int) -> np.ndarray:
    valid_idx = np.flatnonzero(frozen_valid[src])
    frozen_row = list(map(int, frozen_cands[src, valid_idx]))
    fr_rank = frozen_row.index(dst) + 1 if dst in frozen_row else 999.0
    fr_score = float(frozen_scores[direction, src, valid_idx[frozen_row.index(dst)]]) if dst in frozen_row else -1e3
    u_row = union[direction][src]
    dn_rank = u_row.index(dst) + 1 if dst in u_row else 999.0
    rec = float(src in union[OPP[direction]][dst])
    rec_rank = float(union[OPP[direction]][dst].index(src) + 1) if rec else 999.0
    # Approximate 2x2 closure support: count one-step turn continuations available from both tiles.
    if direction in (RIGHT, LEFT):
        turns = (UP, DOWN)
    else:
        turns = (LEFT, RIGHT)
    cyc = 0.0
    for td in turns:
        a = set(union[td][src][:32])
        b = set(union[td][dst][:32])
        cyc += float(len(a & b))
    return np.asarray([
        fr_score / 100.0,
        math.log1p(fr_rank),
        1.0 / dn_rank,
        rec,
        1.0 / rec_rank,
        cyc / 32.0,
    ], dtype=np.float32)


def collect_examples(inputs: Path, score_dir: Path, label_dir: Path, sources: list[str]) -> tuple[list[np.ndarray], list[float], dict[str, tuple[list[list[list[int]]], np.ndarray, np.ndarray, np.ndarray]]]:
    device = torch.device("cuda")
    dino = model(device)
    feats: list[np.ndarray] = []
    ys: list[float] = []
    board_data = {}
    for k, source in enumerate(sources, start=1):
        c, v, s = p13.load_score_cache(score_dir, source)
        union = build_union(c, v, dense_candidates(dino, inputs, source, device))
        tile_to_slot, slot_to_tile = load_label(label_dir, source)
        for d in range(4):
            for i in range(N):
                gt = neighbor_tile(tile_to_slot, slot_to_tile, i, d)
                for q in union[d][i]:
                    feats.append(candidate_features(union, c, v, s, i, d, q))
                    ys.append(float(q == gt))
        board_data[source] = (union, tile_to_slot, slot_to_tile, c, v, s)
        if k % 4 == 0:
            print(json.dumps({"stage": "prepare", "done": k, "total": len(sources)}), flush=True)
    return feats, ys, board_data


class EdgeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class OffsetUF:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int32)
        self.d = np.zeros((n, 2), dtype=np.int32)
    def find(self, x: int):
        if self.p[x] == x:
            return x, np.zeros(2, dtype=np.int32)
        r, up = self.find(int(self.p[x]))
        self.d[x] += up
        self.p[x] = r
        return r, self.d[x].copy()
    def add(self, a: int, b: int, delta: tuple[int, int]) -> bool:
        ra, da = self.find(a)
        rb, db = self.find(b)
        delta = np.asarray(delta, dtype=np.int32)
        if ra == rb:
            return bool(np.array_equal(db - da, delta))
        rel = delta + da - db
        self.p[rb] = ra
        self.d[rb] = rel
        return True


def edge_delta(direction: int) -> tuple[int, int]:
    return {(RIGHT): (1, 0), (LEFT): (-1, 0), (DOWN): (0, 1), (UP): (0, -1)}[direction]


def baseline_coverage(board_data, source: str) -> float:
    union, tile_to_slot, slot_to_tile, _, _, _ = board_data[source]
    total = 0
    hits = 0
    for d in range(4):
        for i in range(N):
            q = neighbor_tile(tile_to_slot, slot_to_tile, i, d)
            if q < 0:
                continue
            total += 1
            if (q in union[d][i]) and (i in union[OPP[d]][q]):
                hits += 1
    return hits / total if total else 0.0


def verifier_coverage(model: EdgeMLP, board_data, source: str, threshold: float, device: torch.device) -> tuple[float, bool]:
    union, tile_to_slot, slot_to_tile, cands, valid, scores = board_data[source]
    uf = OffsetUF(N)
    total = 0
    hits = 0
    invalid = False
    with torch.no_grad():
        for d in range(4):
            for i in range(N):
                q = neighbor_tile(tile_to_slot, slot_to_tile, i, d)
                if q < 0:
                    continue
                total += 1
                for dst in union[d][i]:
                    if i not in union[OPP[d]][dst]:
                        continue
                    x = torch.from_numpy(candidate_features(union, cands, valid, scores, i, d, dst))[None].to(device)
                    p = torch.sigmoid(model(x)).item()
                    if p >= threshold:
                        ok = uf.add(i, dst, edge_delta(d))
                        if not ok:
                            invalid = True
                        if dst == q:
                            hits += 1
                        break
    return (hits / total if total else 0.0), invalid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, default=Path(r"E:\pazzle_data\train\inputs"))
    p.add_argument("--scores", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache"))
    p.add_argument("--labels", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache"))
    p.add_argument("--manifest", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json"))
    p.add_argument("--work", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P33_cva"))
    p.add_argument("--epochs", type=int, default=10)
    a = p.parse_args()
    bad(a.inputs, a.scores, a.labels, a.manifest, a.work)
    a.work.mkdir(parents=True, exist_ok=True)
    train_sources, _ = p13.source_lists(a.manifest)
    sources = sorted(train_sources)[:96]
    feats, ys, board_data = collect_examples(a.inputs, a.scores, a.labels, sources)
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32))
    y = torch.from_numpy(np.asarray(ys, dtype=np.float32))
    device = torch.device("cuda")
    model = EdgeMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    hist = []
    start = time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(x.shape[0])
        losses = []
        for off in range(0, x.shape[0], 4096):
            idx = perm[off:off+4096]
            xb = x[idx].to(device)
            yb = y[idx].to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        hist.append({"epoch": ep + 1, "loss": float(np.mean(losses))})
        print(json.dumps({"stage": "fit", **hist[-1]}), flush=True)
        if time.perf_counter() - start > 15 * 60:
            break
    base = float(np.mean([baseline_coverage(board_data, s) for s in sources]))
    rows = []
    for th in THRESHOLDS:
        covs = []
        inval = 0
        for s in sources:
            cov, bad_flag = verifier_coverage(model, board_data, s, th, device)
            covs.append(cov)
            inval += int(bad_flag)
        mean_cov = float(np.mean(covs))
        rows.append({"threshold": th, "coverage": mean_cov, "gain_pp": 100.0 * (mean_cov - base), "invalid": inval})
    best = max(rows, key=lambda r: r["gain_pp"])
    rep = {
        "experiment": "P33_CVA24",
        "gate": "G2",
        "baseline_coverage": base,
        "rows": rows,
        "best": best,
        "labels_used": True,
        "targets_opened": False,
        "p8_imported": False,
        "passes_G2": bool(best["gain_pp"] >= 3.0 and best["invalid"] == 0),
    }
    (a.work / "p33_g2_report.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep), flush=True)
    if not rep["passes_G2"]:
        raise RuntimeError("P33 G2 rejected")


if __name__ == "__main__":
    main()
