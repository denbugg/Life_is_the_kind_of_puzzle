"""F1P: deterministic phase-derivative compatibility diagnostic.

This evaluator is label-blind while scoring.  It removes per-tile photometric
nuisance, compares boundary value and continuation derivatives, and measures
one-dimensional Fourier phase coherence under a Gaussian window. Exact synthetic
permutations are read only after score matrices are frozen for held-out metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from canvas_data import CanvasDataset
from config import GRID, NFRAG
from imgio import train_val_split

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
INVERSE = (DOWN, UP, RIGHT, LEFT)
DIRECT_EDGES_PER_BOARD = 4 * GRID * (GRID - 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F1P phase-derivative boundary compatibility diagnostic")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--band", type=int, default=3)
    p.add_argument("--topks", default="1,4,20")
    p.add_argument("--seed", type=int, default=240815)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def normalize_per_tile(tiles: Tensor) -> Tensor:
    mean = tiles.mean(dim=(-2, -1), keepdim=True)
    std = tiles.std(dim=(-2, -1), keepdim=True).clamp_min(0.06)
    return (tiles - mean) / std


def luminance(tiles: Tensor) -> Tensor:
    return (tiles[:, 0] * 0.299 + tiles[:, 1] * 0.587 + tiles[:, 2] * 0.114)


def zscore_rows(score: Tensor) -> Tensor:
    count_rows, width = score.shape
    row_source = torch.arange(count_rows, device=score.device).remainder(width)
    mask = torch.arange(width, device=score.device).view(1, width).eq(row_source.view(count_rows, 1))
    score = score.masked_fill(mask, -torch.inf)
    finite = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
    count = (~mask).sum(dim=-1, keepdim=True).clamp_min(1)
    mean = finite.sum(dim=-1, keepdim=True) / count
    var = ((finite - mean).square() * (~mask)).sum(dim=-1, keepdim=True) / count
    return (score - mean) / var.sqrt().clamp_min(1e-5)


def directional_scores(tiles: Tensor, band: int) -> dict[str, Tensor]:
    """Return (4,N,N) scores, ordered U,D,L,R, with self pairs excluded."""
    x = normalize_per_tile(tiles)
    y = luminance(x)
    window = torch.hann_window(y.shape[-2], device=tiles.device, dtype=tiles.dtype).clamp_min(0.05)
    # Derive each canonical (left,right) compatibility, then transpose for inverse direction.
    def lr_score(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # a/b each N×H×W; b is proposed right neighbour.
        a_edge = a[:, :, -band:]
        b_edge = b[:, :, :band]
        # N×N×H×band explicit broadcast remains small at tile size 20.
        diff = (a_edge[:, None] - b_edge[None]).square().mean(dim=(-2, -1))
        a_slope = a[:, :, -1] - a[:, :, -2]
        b_slope = b[:, :, 1] - b[:, :, 0]
        slope = (a_slope[:, None] - b_slope[None]).square().mean(dim=-1)
        # Phase coherence of Gaussian-windowed central boundary profiles.
        pa = ((a[:, :, -1] + a[:, :, -2]) * 0.5) * window
        pb = ((b[:, :, 0] + b[:, :, 1]) * 0.5) * window
        fa = torch.fft.rfft(pa, dim=-1)
        fb = torch.fft.rfft(pb, dim=-1)
        phase = (fa[:, None] * fb[None].conj()).real.sum(dim=-1)
        denom = fa.abs().square().sum(dim=-1).sqrt()[:, None] * fb.abs().square().sum(dim=-1).sqrt()[None]
        coherence = phase / denom.clamp_min(1e-6)
        return -diff, -slope, coherence

    raw_lr, slope_lr, phase_lr = lr_score(y, y)
    raw_ud, slope_ud, phase_ud = lr_score(y.transpose(-1, -2), y.transpose(-1, -2))
    def stack(lr: Tensor, ud: Tensor) -> Tensor:
        return torch.stack((ud.transpose(0, 1), ud, lr.transpose(0, 1), lr), dim=0)
    raw = stack(raw_lr, raw_ud)
    slope = stack(slope_lr, slope_ud)
    phase = stack(phase_lr, phase_ud)
    # Combine individually row-normalized terms so nuisance scales cannot dominate.
    fused = 0.55 * zscore_rows(raw.reshape(4 * NFRAG, NFRAG)).reshape(4, NFRAG, NFRAG)
    fused += 0.30 * zscore_rows(slope.reshape(4 * NFRAG, NFRAG)).reshape(4, NFRAG, NFRAG)
    fused += 0.15 * zscore_rows(phase.reshape(4 * NFRAG, NFRAG)).reshape(4, NFRAG, NFRAG)
    eye = torch.eye(NFRAG, dtype=torch.bool, device=tiles.device).unsqueeze(0)
    return {"norm_value": raw.masked_fill(eye, -torch.inf), "derivative": slope.masked_fill(eye, -torch.inf), "phase_fused": fused.masked_fill(eye, -torch.inf)}


def targets_from_perm(perm: Tensor) -> Tensor:
    """Return exact true candidate index per direction; -1 for exterior anchors."""
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(NFRAG, device=perm.device)
    cell = perm
    row, col = torch.div(cell, GRID, rounding_mode="floor"), torch.remainder(cell, GRID)
    out = torch.full((4, NFRAG), -1, dtype=torch.long, device=perm.device)
    good = row.gt(0); out[UP, good] = inverse[cell[good] - GRID]
    good = row.lt(GRID - 1); out[DOWN, good] = inverse[cell[good] + GRID]
    good = col.gt(0); out[LEFT, good] = inverse[cell[good] - 1]
    good = col.lt(GRID - 1); out[RIGHT, good] = inverse[cell[good] + 1]
    return out


def metrics(score: Tensor, target: Tensor, topks: list[int]) -> dict[str, object]:
    valid = target.ge(0)
    result: dict[str, object] = {}
    top1 = score.argmax(dim=-1)
    reciprocal = torch.zeros_like(valid)
    for direction in range(4):
        inverse = INVERSE[direction]
        a = torch.arange(NFRAG, device=score.device)
        b = top1[direction]
        reciprocal[direction] = valid[direction] & top1[inverse, b].eq(a)
    reciprocal_true = reciprocal & top1.eq(target)
    reciprocal_count = reciprocal.sum().item()
    result["reciprocal_top1"] = {
        "selected_edges": float(reciprocal_count),
        "precision": float(reciprocal_true.sum()) / reciprocal_count if reciprocal_count else 0.0,
        "recall_all_true": float(reciprocal_true.sum()) / DIRECT_EDGES_PER_BOARD,
    }
    for k in topks:
        indices = score.topk(k, dim=-1).indices
        hit = indices.eq(target[..., None]).any(dim=-1) & valid
        result[str(k)] = {"recall_all_true": float(hit.sum()) / DIRECT_EDGES_PER_BOARD}
    return result


def mean_records(records: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for mode in records[0]:
        result[mode] = {}
        for key, vals in records[0][mode].items():
            result[mode][key] = {}
            for metric in vals:
                result[mode][key][metric] = float(sum(record[mode][key][metric] for record in records) / len(records))
    return result


def main() -> None:
    args = parse_args()
    if args.n < 1 or args.band < 2:
        raise ValueError("n must be positive and band must be >=2")
    topks = sorted({int(x) for x in args.topks.split(",") if x.strip()})
    if not topks or min(topks) < 1:
        raise ValueError("topks must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    _, val_names = train_val_split()
    dataset = CanvasDataset(val_names[:args.n], real_prob=0.0, seed=args.seed)
    records: list[dict[str, object]] = []
    for index in range(args.n):
        sample = dataset[index]
        tiles = sample["tiles"].to(device)
        target = targets_from_perm(sample["perm"].to(device).long())
        report = {mode: metrics(score, target, topks) for mode, score in directional_scores(tiles, args.band).items()}
        records.append(report)
        fused = report["phase_fused"]
        print(f"image={index+1}/{args.n} fused_r1={fused['1']['recall_all_true']:.4f} fused_r20={fused[str(max(topks))]['recall_all_true']:.4f} reciprocal_p={fused['reciprocal_top1']['precision']:.4f} reciprocal_r={fused['reciprocal_top1']['recall_all_true']:.4f}", flush=True)
    aggregate = mean_records(records)
    fused = aggregate["phase_fused"]
    gate = bool(
        (fused["reciprocal_top1"]["precision"] >= 0.40 and fused["reciprocal_top1"]["recall_all_true"] >= 0.05)
        or fused.get("20", {}).get("recall_all_true", 0.0) >= 0.1489
    )
    output = {"experiment": "F1P_phase_derivative_boundary", "images": args.n, "band": args.band, "metrics": aggregate, "gate_pass": gate}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
