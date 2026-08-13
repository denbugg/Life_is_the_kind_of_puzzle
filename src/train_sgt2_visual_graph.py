"""SGT2-V: visual sparse candidate-graph reranker capacity and transfer harness.

This train/eval program consumes only manifest-aligned visual graph caches built
from corrupted train inputs plus frozen candidate lists.  It never opens targets
test images, changes candidate IDs, or emits a board layout.  Its metric is
covered-neighbour top-1: queries whose true neighbour is absent are excluded
from both loss and accuracy and are reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 24
N = GRID * GRID
DIRECTIONS = 4
OPPOSITE = torch.tensor([1, 0, 3, 2], dtype=torch.long)
DEFAULT_WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\SGT2_visual_graph")
DEFAULT_CACHE = DEFAULT_WORK / "visual_cache"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    values = np.take_along_axis(scores, order, axis=1)
    valid = np.isfinite(values)
    finite_count = valid.sum(axis=1, keepdims=True).clip(min=1)
    mean = np.where(valid, values, 0.0).sum(axis=1, keepdims=True) / finite_count
    variance = np.where(valid, (values - mean) ** 2, 0.0).sum(axis=1, keepdims=True) / finite_count
    zscore = np.where(valid, (values - mean) / np.sqrt(variance + 1e-6), -10.0).astype(np.float32)
    return order, valid, zscore


@dataclass
class Board:
    name: str
    split: str
    tiles: torch.Tensor        # (576,3,20,20), RGB normalized only per tile in model
    candidate_ids: torch.Tensor  # (Q,K)
    score_z: torch.Tensor      # (Q,K)
    candidate_valid: torch.Tensor  # (Q,K)
    target_index: torch.Tensor # (Q,), -1 when true relation absent
    valid_query: torch.Tensor  # (Q,)
    coverage: float


def load_board(path: Path, k: int, device: torch.device) -> Board:
    with np.load(path, allow_pickle=False) as z:
        tiles = np.asarray(z["tiles_rgb"], dtype=np.uint8)
        perm = np.asarray(z["permutation"], dtype=np.int64)
        candidate_ids = np.asarray(z["candidate_ids"], dtype=np.int64)
        scores = np.asarray(z["candidate_scores"], dtype=np.float32)
        anchors = np.asarray(z["anchors"], dtype=np.int64)
        directions = np.asarray(z["directions"], dtype=np.int64)
        split = str(z["split_name"].item())
    if tiles.shape != (N, 20, 20, 3) or perm.shape != (N,) or scores.shape[0] != N * DIRECTIONS:
        raise ValueError(f"invalid visual graph cache schema: {path}")
    if candidate_ids.shape[0] != N or anchors.shape != (N * DIRECTIONS,) or directions.shape != (N * DIRECTIONS,):
        raise ValueError(f"invalid candidate/query schema: {path}")
    if not (1 <= k <= candidate_ids.shape[1]):
        raise ValueError(f"invalid K={k} for {path}")
    order, candidate_valid, score_z = finite_topk(scores, k)
    candidate_by_query = np.take_along_axis(candidate_ids[anchors], order, axis=1)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(N, dtype=np.int64)
    target = np.full(N * DIRECTIONS, -1, dtype=np.int64)
    valid_query = np.zeros(N * DIRECTIONS, dtype=np.bool_)
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for source in range(N):
        row, col = divmod(int(perm[source]), GRID)
        for direction, (dr, dc) in enumerate(deltas):
            query = source * DIRECTIONS + direction
            rr, cc = row + dr, col + dc
            if not (0 <= rr < GRID and 0 <= cc < GRID):
                continue
            valid_query[query] = True
            truth = int(inv[rr * GRID + cc])
            locations = np.flatnonzero((candidate_by_query[query] == truth) & candidate_valid[query])
            if locations.size:
                target[query] = int(locations[0])
    mask = valid_query & (target >= 0)
    coverage = float(mask.sum() / max(1, valid_query.sum()))
    tiles_tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).contiguous().to(device=device, dtype=torch.float32) / 255.0
    return Board(
        name=path.name,
        split=split,
        tiles=tiles_tensor,
        candidate_ids=torch.from_numpy(candidate_by_query).long().to(device),
        score_z=torch.from_numpy(score_z).to(device),
        candidate_valid=torch.from_numpy(candidate_valid).to(device),
        target_index=torch.from_numpy(target).long().to(device),
        valid_query=torch.from_numpy(valid_query).to(device),
        coverage=coverage,
    )


class TileEncoder(nn.Module):
    def __init__(self, width: int, emb: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, emb, 3, padding=1), nn.GELU(),
        )
        self.norm = nn.GroupNorm(1, emb)

    def forward(self, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Affine-normalize every tile to reduce independent brightness/contrast variance.
        mean = tiles.mean(dim=(2, 3), keepdim=True)
        std = tiles.std(dim=(2, 3), keepdim=True).clamp_min(0.03)
        fmap = self.norm(self.body((tiles - mean) / std))
        # Direction order is U,D,L,R and facing bands preserve orientation.
        sides = torch.stack((fmap[:, :, :6, :].mean((2, 3)), fmap[:, :, -6:, :].mean((2, 3)), fmap[:, :, :, :6].mean((2, 3)), fmap[:, :, :, -6:].mean((2, 3))), dim=1)
        global_token = fmap.mean((2, 3))
        return global_token, sides


class VisualSparseRanker(nn.Module):
    def __init__(self, width: int = 32, emb: int = 64, hidden: int = 128, dropout: float = 0.05) -> None:
        super().__init__()
        self.encoder = TileEncoder(width, emb)
        self.pair = nn.Sequential(nn.Linear(emb * 4 + 3, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.query = nn.Sequential(nn.Linear(hidden + emb, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.output = nn.Sequential(nn.Linear(hidden * 2 + emb * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.residual_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, board: Board) -> torch.Tensor:
        _, sides = self.encoder(board.tiles)
        queries, k = board.candidate_ids.shape
        sources = torch.arange(N, device=board.tiles.device).repeat_interleave(DIRECTIONS)
        directions = torch.arange(DIRECTIONS, device=board.tiles.device).repeat(N)
        opposite = OPPOSITE.to(board.tiles.device)[directions]
        src_side = sides[sources, directions][:, None, :].expand(-1, k, -1)
        dst_side = sides[board.candidate_ids, opposite[:, None]]
        raw = board.score_z[..., None]
        rank = (torch.arange(k, device=board.tiles.device, dtype=torch.float32)[None, :, None] / max(1, k - 1)).expand(queries, -1, -1)
        valid = board.candidate_valid.float()[..., None]
        pair_input = torch.cat((src_side, dst_side, (src_side - dst_side).abs(), src_side * dst_side, raw, rank, valid), dim=-1)
        edge = self.pair(pair_input)
        edge = edge * valid
        counts = valid.sum(dim=1).clamp_min(1.0)
        query_context = edge.sum(dim=1) / counts
        source_global = sides[sources].mean(dim=1)
        query_context = self.query(torch.cat((query_context, source_global), dim=-1))
        dst_global = sides[board.candidate_ids].mean(dim=2)
        context = query_context[:, None, :].expand(-1, k, -1)
        logit_residual = self.output(torch.cat((edge, context, src_side, dst_global), dim=-1)).squeeze(-1)
        logits = board.score_z + self.residual_scale * logit_residual
        return logits.masked_fill(~board.candidate_valid, -1e9)


def covered_loss(logits: torch.Tensor, board: Board) -> torch.Tensor:
    mask = board.target_index >= 0
    if not mask.any():
        raise RuntimeError(f"no covered labels in {board.name}")
    return F.cross_entropy(logits[mask], board.target_index[mask])


def top1(logits: torch.Tensor, board: Board) -> tuple[float, float, int]:
    mask = board.target_index >= 0
    predicted = logits.argmax(dim=1)
    correct = (predicted[mask] == board.target_index[mask]).float()
    return float(correct.mean().detach()), board.coverage, int(mask.sum().detach())


def evaluate(model: nn.Module, boards: list[Board]) -> dict[str, Any]:
    model.eval()
    rows = []
    with torch.no_grad():
        for board in boards:
            logits = model(board)
            learned, coverage, count = top1(logits, board)
            frozen, _, _ = top1(board.score_z.masked_fill(~board.candidate_valid, -1e9), board)
            rows.append({"name": board.name, "split": board.split, "covered_top1_frozen": frozen, "covered_top1_sgt2": learned, "delta": learned - frozen, "coverage": coverage, "covered_queries": count})
    values = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    return {"rows": rows, "summary": {"covered_top1_frozen": float(np.mean([row["covered_top1_frozen"] for row in rows])), "covered_top1_sgt2": float(np.mean([row["covered_top1_sgt2"] for row in rows])), "delta": float(values.mean()), "coverage": float(np.mean([row["coverage"] for row in rows])), "min_delta": float(values.min())}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fit", default="image_0000_k64.npz,image_0001_k64.npz")
    parser.add_argument("--dev", default="image_0014_k64.npz,image_0020_k64.npz")
    parser.add_argument("--k", type=int, default=96)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--emb", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=260814)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    fit_paths = [args.cache_dir / name.strip() for name in args.fit.split(",") if name.strip()]
    dev_paths = [args.cache_dir / name.strip() for name in args.dev.split(",") if name.strip()]
    if not all(path.is_file() for path in fit_paths + dev_paths):
        raise FileNotFoundError("requested visual cache unavailable")
    fit_boards = [load_board(path, args.k, device) for path in fit_paths]
    dev_boards = [load_board(path, args.k, device) for path in dev_paths]
    if any(board.split != "fit" for board in fit_boards) or any(board.split != "dev" for board in dev_boards):
        raise RuntimeError("fit/dev visual-cache membership mismatch")
    model = VisualSparseRanker(args.width, args.emb, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        board = fit_boards[(step - 1) % len(fit_boards)]
        optimizer.zero_grad(set_to_none=True)
        loss = covered_loss(model(board), board)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite SGT2 loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 6) == 0 or step == args.steps:
            dev = evaluate(model, dev_boards)
            item = {"step": step, "loss": float(loss.detach()), **dev["summary"]}
            history.append(item)
            print(json.dumps(item), flush=True)
    dev = evaluate(model, dev_boards)
    summary = dev["summary"]
    gate = {"condition": "source-disjoint covered top-1 delta > 0; retain for expanded-cache G2 only if >= +0.01", "passed_positive": bool(summary["delta"] > 0), "passed_expansion": bool(summary["delta"] >= 0.01), "decision": "expand_candidate_caches" if summary["delta"] >= 0.01 else ("diagnose_before_expansion" if summary["delta"] > 0 else "reject_SGT2_visual_ranker")}
    report = {"experiment": "SGT2_visual_sparse_candidate_ranker", "scope": "visual cache from corrupted train inputs plus frozen candidate graphs; only covered candidate labels; no test data; no layout solver", "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "fit": [board.name for board in fit_boards], "dev": [board.name for board in dev_boards], "history": history, "dev": dev, "gate": gate, "cache_manifest_sha256": sha256(args.cache_dir / "visual_cache_manifest.json"), "elapsed_seconds": time.time() - started}
    args.work.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.work / "sgt2_visual_capacity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    checkpoint = args.checkpoint or args.work / "sgt2_visual_capacity.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "report": report}, checkpoint)
    print(json.dumps({"summary": summary, "gate": gate, "report": str(report_path), "checkpoint": str(checkpoint)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
