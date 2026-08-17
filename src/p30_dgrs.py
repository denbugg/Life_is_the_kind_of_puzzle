"""P30 DGRS-24: Dense-score-only reciprocal graph solver.
Frozen rank96 cache is baseline-only and is never an input to dense graph scoring.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p29_dpcg import N, desc, load_tiles, model
from p12_loop_consensus import solve_buddies_from_scores
import p13_component_pose as p13

GRID = (0.0, 0.10, 0.25, 0.50, 1.0)
OPP = (2, 3, 0, 1)


def reject_p8(*paths: Path) -> None:
    if "p8" in "\n".join(str(p).lower() for p in paths):
        raise RuntimeError("P8 artifact/path prohibited")


def dense_scores(m: torch.nn.Module, device: torch.device, inputs: Path, source: str) -> np.ndarray:
    z = desc(m, load_tiles(inputs, source), device)
    bands = torch.stack((z[:, :, -1, :], z[:, -1, :, :], z[:, :, 0, :], z[:, 0, :, :]))
    out = np.empty((4, N, N), np.float32)
    with torch.no_grad():
        for d in range(4):
            s = torch.einsum("idc,jdc->ij", bands[d], bands[OPP[d]]) / bands.shape[2]
            s.fill_diagonal_(-float("inf"))
            out[d] = s.float().cpu().numpy()
    return out


def rank0(s: np.ndarray) -> np.ndarray:
    # Deterministic stable ranks: zero is best. Self is always worst.
    return np.argsort(np.argsort(-s, axis=2, kind="stable"), axis=2, kind="stable").astype(np.float32)


def reciprocal_field(raw: np.ndarray, lam: float) -> np.ndarray:
    if raw.shape != (4, N, N) or not np.isfinite(raw[np.isfinite(raw)]).all():
        raise RuntimeError("bad raw dense score tensor")
    r = rank0(raw) / float(N - 1)
    out = np.empty_like(raw)
    for d in range(4):
        z = raw[d].copy()
        finite = np.isfinite(z)
        mu = float(z[finite].mean()); sd = float(z[finite].std()) + 1e-6
        z[finite] = (z[finite] - mu) / sd
        rev = r[OPP[d]].T
        out[d] = z - float(lam) * (r[d] + rev) * 0.5
        np.fill_diagonal(out[d], -np.inf)
    return out


def neighbor(tile_to_slot: np.ndarray, slot_to_tile: np.ndarray, i: int, d: int) -> int:
    rr, cc = divmod(int(slot_to_tile[i]), 24)
    if d == 0: cc += 1
    elif d == 1: rr += 1
    elif d == 2: cc -= 1
    else: rr -= 1
    return -1 if rr < 0 or rr >= 24 or cc < 0 or cc >= 24 else int(tile_to_slot[rr * 24 + cc])


def labels(label_dir: Path, source: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(label_dir / (Path(source).stem + ".npz"), allow_pickle=False) as z:
        p = z["target_tile_to_slot"].astype(np.int32)
        src = str(z["source"])
    if src != source or p.shape != (N,) or np.unique(p).size != N:
        raise RuntimeError("invalid cached FIT labels")
    inv = np.empty(N, np.int32); inv[p] = np.arange(N, dtype=np.int32)
    return p, inv


def recall20(field: np.ndarray, label_dir: Path, source: str) -> tuple[float, int]:
    p, inv = labels(label_dir, source)
    hit = tot = 0
    for d in range(4):
        top = np.argsort(-field[d], axis=1, kind="stable")[:, :20]
        for i in range(N):
            q = neighbor(p, inv, i, d)
            if q >= 0:
                hit += int(q in top[i]); tot += 1
    return hit / tot, tot


def placement(field: np.ndarray, max_edges: int = 96) -> tuple[np.ndarray, float]:
    # The canonical solver accepts contiguous NumPy FP32 matrices; this call is dense-only by construction.
    R, D = np.ascontiguousarray(field[0], np.float32), np.ascontiguousarray(field[1], np.float32)
    board, obj = solve_buddies_from_scores(R, D, max_edges=max_edges, min_margin=0.0, repair_passes=2)
    # Canonical output is a flat tile->slot permutation, not a 24x24 slot->tile image.
    b = np.asarray(board, np.int32).reshape(-1)
    if b.shape != (N,) or np.unique(b).size != N or b.min() < 0 or b.max() >= N:
        raise RuntimeError("invalid canonical solver permutation")
    return b, float(obj)


def board_accuracy(board: np.ndarray, label_dir: Path, source: str) -> float:
    p, _ = labels(label_dir, source)
    # Canonical board is the tile->slot placement, as is the cached target mapping.
    return float(np.mean(board == p))


def g0(args: argparse.Namespace) -> dict:
    # Direct reciprocal edges form a unique 24x24 score graph.
    raw = np.full((4, N, N), -20.0, np.float32)
    for r in range(24):
        for c in range(24):
            i = r * 24 + c
            for d, (dr, dc) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 24 and 0 <= cc < 24: raw[d, i, rr * 24 + cc] = 20.0
    for d in range(4): np.fill_diagonal(raw[d], -np.inf)
    f = reciprocal_field(raw, 0.50)
    b, obj = placement(f, max_edges=2208)
    expected = np.arange(N, dtype=np.int32)
    rep = {"experiment": "P30_DGRS24", "gate": "G0", "valid_bijection": bool(np.unique(b).size == N), "exact_synthetic": bool(np.array_equal(b, expected)), "objective": obj, "passes_G0": bool(np.array_equal(b, expected))}
    return rep


def g1(args: argparse.Namespace) -> dict:
    device = torch.device("cuda")
    m = model(device)
    rows = []
    for k, source in enumerate(args.sources):
        start = time.perf_counter(); raw = dense_scores(m, device, args.inputs, source); elapsed = time.perf_counter() - start
        reciprocal = rank0(raw); asymmetric = float(np.mean(np.abs(raw[0] - raw[2].T) > 1e-6))
        ok = bool(np.isfinite(raw[np.isfinite(raw)]).all() and elapsed <= 90.0 and asymmetric > 0.01 and np.array_equal(reciprocal, rank0(raw)))
        rows.append({"source": source, "seconds": elapsed, "asymmetric_fraction": asymmetric, "ok": ok})
        if (k + 1) % 4 == 0: print(json.dumps({"stage": "g1", "done": k + 1, "total": len(args.sources)}), flush=True)
    return {"experiment": "P30_DGRS24", "gate": "G1", "rows": rows, "labels_used": False, "targets_opened": False, "p8_imported": False, "passes_G1": bool(all(x["ok"] for x in rows))}


def fit_grid(raws: dict[str, np.ndarray], label_dir: Path, names: list[str]) -> list[dict]:
    result = []
    for lam in GRID:
        vals = []
        for source in names: vals.append(recall20(reciprocal_field(raws[source], lam), label_dir, source)[0])
        result.append({"lambda": lam, "recall20": float(np.mean(vals)), "boards": len(vals)})
    return result


def g2g3(args: argparse.Namespace, mode: str) -> dict:
    reject_p8(args.inputs, args.label_dir, args.score_dir, args.manifest, args.work)
    train, _ = p13.source_lists(args.manifest)
    ns = sorted(train)[:128]
    fit, selection = ns[:96], ns[96:]
    device = torch.device("cuda"); m = model(device); raws = {}
    for k, source in enumerate(ns):
        raws[source] = dense_scores(m, device, args.inputs, source)
        if (k + 1) % 8 == 0: print(json.dumps({"stage": "features", "done": k + 1, "total": 128}), flush=True)
    grid = fit_grid(raws, args.label_dir, fit)
    base = next(x for x in grid if x["lambda"] == 0.0)
    best = max(grid, key=lambda x: (x["recall20"], -x["lambda"]))
    if mode == "g2":
        gain = 100.0 * (best["recall20"] - base["recall20"])
        return {"experiment": "P30_DGRS24", "gate": "G2", "train_grid": grid, "selected": best, "gain_pp": gain, "labels_used": True, "targets_opened": False, "p8_imported": False, "passes_G2": bool(gain >= 1.0)}
    selected = float(best["lambda"])
    vals = []; base_vals = []; bad = 0
    for source in selection:
        f = reciprocal_field(raws[source], selected); f0 = reciprocal_field(raws[source], 0.0)
        vals.append(recall20(f, args.label_dir, source)[0]); base_vals.append(recall20(f0, args.label_dir, source)[0])
        try: b, _ = placement(f); board_accuracy(b, args.label_dir, source)
        except Exception: bad += 1
    gain = 100.0 * (float(np.mean(vals)) - float(np.mean(base_vals)))
    return {"experiment": "P30_DGRS24", "gate": "G3", "selected_lambda": selected, "fit_grid": grid, "selection_recall20": float(np.mean(vals)), "selection_base_recall20": float(np.mean(base_vals)), "gain_pp": gain, "invalid_boards": bad, "labels_used": True, "targets_opened": False, "p8_imported": False, "passes_G3": bool(gain >= 1.0 and bad == 0)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("g0", "g1", "g2", "g3"), required=True)
    p.add_argument("--inputs", type=Path, default=Path(r"E:\pazzle_data\train\inputs"))
    p.add_argument("--label-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache"))
    p.add_argument("--score-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache"))
    p.add_argument("--manifest", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json"))
    p.add_argument("--work", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P30_dgrs"))
    p.add_argument("--sources", nargs="*", default=("img_000002.png", "img_000025.png", "img_000098.png", "img_000168.png", "img_000172.png", "img_000194.png", "img_000223.png", "img_000243.png", "img_000267.png", "img_000304.png", "img_000344.png", "img_000384.png", "img_000426.png", "img_000457.png", "img_000480.png", "img_000513.png"))
    a = p.parse_args(); reject_p8(a.inputs, a.label_dir, a.score_dir, a.manifest, a.work); a.work.mkdir(parents=True, exist_ok=True)
    rep = g0(a) if a.mode == "g0" else g1(a) if a.mode == "g1" else g2g3(a, a.mode)
    out = a.work / f"p30_{a.mode}_report.json"; out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8"); print(json.dumps(rep), flush=True)
    key = f"passes_{a.mode.upper()}"
    if not rep[key]: raise RuntimeError(f"P30 {a.mode} rejected")

if __name__ == "__main__": main()
