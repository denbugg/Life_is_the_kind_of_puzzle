"""SGT1: sparse rank96-candidate graph Transformer capacity gate.

This harness never opens test images.  It reads pre-existing candidate-graph
caches, where every directed query has a frozen ranker candidate list.  A graph
Transformer applies a learned residual to edge scores.  Supervision is limited
to labels whose true neighbour is actually present in the frozen list; missing
relations are explicitly reported rather than treated as reranker errors.
"""
from __future__ import annotations

import argparse
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
OPPOSITE = np.array([1, 0, 3, 2], dtype=np.int64)
DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))
DEFAULT_CACHE = Path(r"E:\pazzle_work\edge_confidence\full_graph_cache")
DEFAULT_WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\SGT1_sparse_graph")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_score_summary(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return top-K order, finite mask, safe scores and finite-only row statistics."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    raw = np.take_along_axis(scores, order, axis=1)
    finite = np.isfinite(raw)
    counts = finite.sum(axis=1, keepdims=True).clip(min=1)
    finite_min = np.where(finite, raw, np.inf).min(axis=1, keepdims=True)
    finite_min[~np.isfinite(finite_min)] = -20.0
    safe = np.where(finite, raw, finite_min - 10.0).astype(np.float32)
    mean = np.where(finite, raw, 0.0).sum(axis=1, keepdims=True) / counts
    variance = np.where(finite, (raw - mean) ** 2, 0.0).sum(axis=1, keepdims=True) / counts
    return order, finite, safe, np.concatenate((mean, np.sqrt(variance) + 1e-6), axis=1).astype(np.float32)


def mean_std_features(scores: np.ndarray, k: int) -> np.ndarray:
    _, _, top, _ = finite_score_summary(scores, k)
    first = top[:, 0]
    second = top[:, 1] if k > 1 else top[:, 0]
    return np.stack((first, top.mean(1), top.std(1), first - second, np.median(top, 1), top[:, -1]), axis=1)


@dataclass
class GraphBoard:
    name: str
    node_features: torch.Tensor  # (576, 24)
    src: torch.Tensor            # (Q*K,)
    dst: torch.Tensor            # (Q*K,)
    edge_features: torch.Tensor  # (Q*K, 5)
    direction: torch.Tensor      # (Q*K,)
    q_index: torch.Tensor        # (Q*K,)
    valid_query: torch.Tensor    # (Q,)
    target_index: torch.Tensor   # (Q,), -1 if true edge missing
    candidate_valid: torch.Tensor  # (Q,K), excludes padded -inf candidates
    k: int


def load_board(path: Path, k: int, device: torch.device) -> GraphBoard:
    with np.load(path, allow_pickle=False) as z:
        perm = np.asarray(z["permutation"], dtype=np.int64)
        candidate_ids = np.asarray(z["candidate_ids"], dtype=np.int64)
        candidate_scores = np.asarray(z["candidate_scores"], dtype=np.float32)
    if perm.shape != (N,) or candidate_ids.shape[0] != N or candidate_scores.shape[0] != N * DIRECTIONS:
        raise ValueError(f"unexpected cache schema in {path}")
    if not (1 <= k <= candidate_ids.shape[1]):
        raise ValueError(f"K={k} outside candidate capacity {candidate_ids.shape[1]}")
    width = candidate_ids.shape[1]
    order, candidate_valid, score_by_query, row_stats = finite_score_summary(candidate_scores, k)
    # Each direction query inherits its candidate ids from its source tile.
    source_by_query = np.repeat(np.arange(N), DIRECTIONS)
    dst_by_query = candidate_ids[source_by_query[:, None], order]
    row_mean, row_std = row_stats[:, :1], row_stats[:, 1:]
    zscore = (score_by_query - row_mean) / row_std

    # Dense lookup is bounded: 4*576*576 float32, used only to attach reciprocal evidence.
    lookup = np.full((DIRECTIONS, N, N), -np.inf, dtype=np.float32)
    for direction in range(DIRECTIONS):
        q = np.arange(N) * DIRECTIONS + direction
        lookup[direction, np.arange(N)[:, None], candidate_ids] = candidate_scores[q]
    directions_by_query = np.tile(np.arange(DIRECTIONS), N)
    reciprocal = lookup[OPPOSITE[directions_by_query][:, None], dst_by_query, source_by_query[:, None]]
    reciprocal_present = np.isfinite(reciprocal).astype(np.float32)
    reciprocal_finite = np.where(np.isfinite(reciprocal), reciprocal, score_by_query.min(axis=1, keepdims=True) - 5.0)
    reciprocal_z = (reciprocal_finite - row_mean) / row_std
    rank = np.broadcast_to(np.arange(k, dtype=np.float32)[None], (N * DIRECTIONS, k)) / max(k - 1, 1)

    inv = np.empty_like(perm)
    inv[perm] = np.arange(N)
    valid_query = np.zeros(N * DIRECTIONS, dtype=bool)
    target_index = np.full(N * DIRECTIONS, -1, dtype=np.int64)
    for source in range(N):
        r, c = divmod(int(perm[source]), GRID)
        for direction, (dr, dc) in enumerate(DELTA):
            rr, cc = r + dr, c + dc
            q = source * DIRECTIONS + direction
            if not (0 <= rr < GRID and 0 <= cc < GRID):
                continue
            valid_query[q] = True
            truth = int(inv[rr * GRID + cc])
            where = np.flatnonzero((dst_by_query[q] == truth) & candidate_valid[q])
            if len(where):
                target_index[q] = int(where[0])

    query_stats = mean_std_features(candidate_scores, k).reshape(N, DIRECTIONS * 6)
    src = np.repeat(source_by_query, k)
    dst = dst_by_query.reshape(-1)
    edge_features = np.stack((zscore.reshape(-1), reciprocal_z.reshape(-1), reciprocal_present.reshape(-1), rank.reshape(-1), score_by_query.reshape(-1), candidate_valid.astype(np.float32).reshape(-1)), axis=1).astype(np.float32)
    direction = np.repeat(directions_by_query, k)
    q_index = np.repeat(np.arange(N * DIRECTIONS), k)
    return GraphBoard(
        name=path.name,
        node_features=torch.from_numpy(query_stats.astype(np.float32)).to(device),
        src=torch.from_numpy(src).long().to(device),
        dst=torch.from_numpy(dst).long().to(device),
        edge_features=torch.from_numpy(edge_features).to(device),
        direction=torch.from_numpy(direction).long().to(device),
        q_index=torch.from_numpy(q_index).long().to(device),
        valid_query=torch.from_numpy(valid_query).to(device),
        target_index=torch.from_numpy(target_index).long().to(device),
        candidate_valid=torch.from_numpy(candidate_valid).to(device),
        k=k,
    )


class GraphBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.node_norm = nn.LayerNorm(dim)
        self.node_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.edge_norm = nn.LayerNorm(dim * 3)
        self.edge_mlp = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.node_update = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.node_norm(nodes).unsqueeze(0)
        nodes = nodes + self.node_attn(n, n, n, need_weights=False)[0].squeeze(0)
        e_input = torch.cat((edges, nodes[src], nodes[dst]), dim=1)
        edges = edges + self.edge_mlp(self.edge_norm(e_input))
        out = torch.zeros_like(nodes)
        inc = torch.zeros_like(nodes)
        out.index_add_(0, src, edges)
        inc.index_add_(0, dst, edges)
        out_count = torch.bincount(src, minlength=N).to(nodes.dtype).clamp_min_(1).unsqueeze(1)
        inc_count = torch.bincount(dst, minlength=N).to(nodes.dtype).clamp_min_(1).unsqueeze(1)
        nodes = nodes + self.node_update(torch.cat((nodes, out / out_count, inc / inc_count), dim=1))
        return nodes, edges


class SparseGraphTransformer(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, layers: int = 3, dropout: float = 0.05) -> None:
        super().__init__()
        self.node_in = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, dim), nn.GELU(), nn.Linear(dim, dim))
        self.edge_in = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.direction = nn.Embedding(DIRECTIONS, dim)
        self.blocks = nn.ModuleList([GraphBlock(dim, heads, dropout) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(dim * 3 + 1), nn.Linear(dim * 3 + 1, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, board: GraphBoard) -> torch.Tensor:
        nodes = self.node_in(board.node_features)
        edges = self.edge_in(board.edge_features) + self.direction(board.direction)
        for block in self.blocks:
            nodes, edges = block(nodes, edges, board.src, board.dst)
        base = board.edge_features[:, 4:5]
        residual = self.out(torch.cat((edges, nodes[board.src], nodes[board.dst], base), dim=1)).squeeze(1)
        # The frozen score is deliberately retained; learned residual is the only change.
        return base.squeeze(1) + residual


def scores_to_query(scores: torch.Tensor, board: GraphBoard) -> torch.Tensor:
    query = scores.reshape(N * DIRECTIONS, board.k)
    return query.masked_fill(~board.candidate_valid, float("-inf"))


def metrics(scores: torch.Tensor, board: GraphBoard) -> dict[str, float]:
    query_scores = scores_to_query(scores, board)
    target = board.target_index
    valid = board.valid_query
    covered = target >= 0
    pred = query_scores.argmax(dim=1)
    correct = pred.eq(target) & covered
    return {
        "valid_edges": int(valid.sum().item()),
        "covered_edges": int(covered.sum().item()),
        "coverage": float(covered[valid].float().mean().item()),
        "top1_all": float(correct[valid].float().mean().item()),
        "top1_covered": float(correct[covered].float().mean().item()),
    }


def loss_for(scores: torch.Tensor, board: GraphBoard) -> torch.Tensor:
    q = scores_to_query(scores, board)
    mask = board.target_index >= 0
    return F.cross_entropy(q[mask], board.target_index[mask])


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--train", default="image_0000_k64.npz,image_0001_k64.npz")
    ap.add_argument("--eval", default="image_0000_k64.npz,image_0001_k64.npz")
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2413)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--mode", choices=("capacity", "pilot"), default="capacity", help="capacity enforces covered-edge memorization; pilot only reports held-out deltas")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SGT1 is a local-GPU-only gate")
    if args.dim % args.heads:
        raise ValueError("dim must be divisible by heads")
    seed_all(args.seed)
    device = torch.device("cuda")
    train_paths = [args.cache_dir / n.strip() for n in args.train.split(",") if n.strip()]
    eval_paths = [args.cache_dir / n.strip() for n in args.eval.split(",") if n.strip()]
    if not train_paths or not eval_paths or any(not path.is_file() for path in train_paths + eval_paths):
        raise FileNotFoundError("missing specified candidate cache")
    boards = {path.name: load_board(path, args.k, device) for path in dict.fromkeys(train_paths + eval_paths)}
    model = SparseGraphTransformer(args.dim, args.heads, args.layers, args.dropout).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    start = time.time()
    for step in range(1, args.steps + 1):
        board = boards[train_paths[(step - 1) % len(train_paths)].name]
        optimizer.zero_grad(set_to_none=True)
        scores = model(board)
        loss = loss_for(scores, board)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            with torch.no_grad():
                m = metrics(model(board), board)
            print(json.dumps({"step": step, "loss": float(loss.detach()), **m, "seconds": round(time.time() - start, 2)}), flush=True)
    model.eval()
    per_board: list[dict[str, Any]] = []
    with torch.no_grad():
        for path in eval_paths:
            board = boards[path.name]
            base = board.edge_features[:, 4]
            reranked = model(board)
            row = {"name": path.name, "baseline": metrics(base, board), "reranked": metrics(reranked, board)}
            row["delta_top1_covered"] = row["reranked"]["top1_covered"] - row["baseline"]["top1_covered"]
            per_board.append(row)
    covered_top1 = np.asarray([row["reranked"]["top1_covered"] for row in per_board], dtype=np.float64)
    capacity_pass = bool(np.min(covered_top1) >= 0.95)
    pilot_positive = bool(np.mean([row["delta_top1_covered"] for row in per_board]) > 0.0)
    if args.mode == "capacity":
        gate = {"condition": "each fixed cached board top1_covered >= 0.95", "passed": capacity_pass, "decision": "advance_to_source_disjoint_cache_gate" if capacity_pass else "reject_SGT1_before_cache_expansion"}
    else:
        gate = {"condition": "informational two-board source-disjoint pilot; full gate needs at least eight DEV caches", "passed": pilot_positive, "decision": "pilot_positive_expand_DEV_cache" if pilot_positive else "pilot_negative_stop_before_DEV_cache_expansion"}
    report = {
        "experiment": "SGT1_sparse_candidate_graph_transformer_capacity",
        "scope": "two-board fixed cached candidate graph relative-overfit only; no test access; not a DEV result",
        "args": vars(args) | {"cache_dir": str(args.cache_dir), "work": str(args.work), "report": str(args.report) if args.report else None, "checkpoint": str(args.checkpoint) if args.checkpoint else None},
        "params": params,
        "per_board": per_board,
        "summary": {
            "mean_top1_covered": float(covered_top1.mean()),
            "min_top1_covered": float(covered_top1.min()),
            "mean_delta_top1_covered": float(np.mean([row["delta_top1_covered"] for row in per_board])),
        },
        "gate": gate,
        "elapsed_seconds": time.time() - start,
    }
    destination = args.report or args.work / "sgt1_capacity_report.json"
    save_json(destination, report)
    checkpoint = args.checkpoint or args.work / "sgt1_capacity.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "report": report}, checkpoint)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
