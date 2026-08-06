"""Bounded seeded graduated-assignment QAP smoke evaluator.

It turns the listwise seam ranker's sparse candidate rows into dense, nonnegative
right/down compatibility matrices, optionally boosts high-confidence reciprocal
seeds, and optimizes a soft tile-to-position permutation.  This is deliberately
a small diagnostic, not a production full-board solver.

Examples
--------
python src/eval_seeded_qap.py --oracle
python src/eval_seeded_qap.py --n 1 --steps 24 --device cuda
"""
from __future__ import annotations

import argparse
import math
import os
import random
from collections.abc import Mapping

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from candidate_rank import DOWN, LEFT, RIGHT, UP
from canvas_data import CanvasDataset
from config import GRID, NFRAG, SEED
from eval_candidate_rank import load_ranker, mutual_argmax_relations, score_full_graph
from imgio import train_val_split
from placement_metrics import neighbour_accuracy
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


def _default_ranker(workspace: str) -> str:
    directory = os.path.join(workspace, "artifacts", "candidate_rank")
    v2 = os.path.join(directory, "rank_v2w64_best.pt")
    return v2 if os.path.isfile(v2) else os.path.join(directory, "rank_v1_best.pt")


def _graph_paths(payload: Mapping[str, object], args: argparse.Namespace) -> tuple[str, str, int]:
    graph = payload.get("candidate_graph", {})
    graph = graph if isinstance(graph, Mapping) else {}
    raw_encoders = graph.get("encoders", ())
    encoders = list(raw_encoders) if isinstance(raw_encoders, (list, tuple)) else []
    saved = payload.get("args", {})
    saved = saved if isinstance(saved, Mapping) else {}

    def recorded(index: int) -> str:
        return str(encoders[index].get("path", "")) if (
            index < len(encoders) and isinstance(encoders[index], Mapping)
        ) else ""

    primary = str(args.affinity_ckpt or recorded(0) or saved.get("affinity_ckpt", ""))
    secondary = str(args.affinity_ckpt2 or recorded(1) or saved.get("affinity_ckpt2", ""))
    if not primary:
        raise RuntimeError("no affinity checkpoint recorded; pass --affinity-ckpt")
    top_k = int(args.top_k or saved.get("candidate_k", 64))
    if not 1 <= top_k < NFRAG:
        raise ValueError(f"resolved top-K must lie in [1,{NFRAG - 1}], got {top_k}")
    return primary, secondary, top_k


def dense_rd(candidates: Tensor, scores: Tensor) -> tuple[Tensor, Tensor]:
    """Listwise softmax rows -> dense nonnegative directed right/down scores."""
    finite = torch.isfinite(scores)
    safe = scores.masked_fill(~finite, -torch.inf)
    # A fully invalid row must be zero, not softmax(NaN).
    safe = torch.where(finite.any(dim=-1, keepdim=True), safe, torch.zeros_like(safe))
    probabilities = torch.softmax(safe, dim=-1).masked_fill(~finite, 0.0)

    def direction_matrix(direction: int) -> Tensor:
        out = torch.zeros((NFRAG, NFRAG), dtype=probabilities.dtype, device=probabilities.device)
        out.scatter_add_(1, candidates, probabilities[direction])
        out.fill_diagonal_(0.0)
        return out

    # Average both directions of the same physical claim when present.
    right = 0.5 * (direction_matrix(RIGHT) + direction_matrix(LEFT).transpose(0, 1))
    down = 0.5 * (direction_matrix(DOWN) + direction_matrix(UP).transpose(0, 1))
    right.fill_diagonal_(0.0)
    down.fill_diagonal_(0.0)
    return right.contiguous(), down.contiguous()


def add_reciprocal_seed_bonus(
    right: Tensor, down: Tensor, candidates: Tensor, scores: Tensor, confidence: float, bonus: float
) -> int:
    """Add model-only reciprocal seed evidence in its physical R/D orientation."""
    seeds = [edge for edge in mutual_argmax_relations(candidates, scores) if edge.weight >= confidence]
    if bonus <= 0.0:
        return len(seeds)
    for edge in seeds:
        value = bonus * float(edge.weight)
        if edge.direction == RIGHT:
            right[edge.anchor, edge.target] += value
        elif edge.direction == LEFT:
            right[edge.target, edge.anchor] += value
        elif edge.direction == DOWN:
            down[edge.anchor, edge.target] += value
        elif edge.direction == UP:
            down[edge.target, edge.anchor] += value
    right.fill_diagonal_(0.0)
    down.fill_diagonal_(0.0)
    return len(seeds)


def log_sinkhorn(logits: Tensor, temperature: float, rounds: int) -> Tensor:
    """Numerically stable log-domain projection to a doubly stochastic matrix."""
    log_p = logits / temperature
    for _ in range(rounds):
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
    return log_p.exp()


def qap_value(p: Tensor, right: Tensor, down: Tensor) -> Tensor:
    """Expected mean directed grid-edge compatibility under tile->position P."""
    cells = torch.arange(NFRAG, device=p.device).reshape(GRID, GRID)
    left, rpos = cells[:, :-1].reshape(-1), cells[:, 1:].reshape(-1)
    top, bpos = cells[:-1, :].reshape(-1), cells[1:, :].reshape(-1)

    def edge_sum(matrix: Tensor, first: Tensor, second: Tensor) -> Tensor:
        # Sparse multiplication keeps the gradient path to P while exploiting
        # the zero-filled dense candidate matrix.
        linked = torch.sparse.mm(matrix, p.index_select(1, second))
        return (p.index_select(1, first) * linked).sum()

    return (edge_sum(right, left, rpos) + edge_sum(down, top, bpos)) / (2 * GRID * (GRID - 1))


def optimize_qap(
    right: Tensor, down: Tensor, args: argparse.Namespace, *, init_logits: Tensor | None = None,
    steps: int | None = None, entropy_weight: float | None = None,
) -> tuple[Tensor, float]:
    """Adam ascent through annealed log-Sinkhorn; all inputs are label-free."""
    count = int(steps or args.steps)
    logits = nn.Parameter(
        init_logits.detach().clone() if init_logits is not None
        else torch.randn((NFRAG, NFRAG), device=right.device) * args.init_noise
    )
    optimizer = torch.optim.Adam((logits,), lr=args.lr)
    sparse_right, sparse_down = right.to_sparse_coo().coalesce(), down.to_sparse_coo().coalesce()
    entropy_weight = args.entropy if entropy_weight is None else entropy_weight
    for step in range(count):
        fraction = step / max(count - 1, 1)
        temperature = math.exp(
            math.log(args.temperature_start) * (1.0 - fraction)
            + math.log(args.temperature_end) * fraction
        )
        p = log_sinkhorn(logits, temperature, args.sinkhorn_rounds)
        value = qap_value(p, sparse_right, sparse_down)
        entropy = -(p * p.clamp_min(1.0e-12).log()).sum() / (NFRAG * math.log(NFRAG))
        loss = -value - entropy_weight * (1.0 - fraction) * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    p = log_sinkhorn(logits, args.temperature_end, args.sinkhorn_rounds)
    return p.detach(), float(qap_value(p, sparse_right, sparse_down).detach())


def greedy_decode(p: Tensor) -> np.ndarray:
    """Collision-free descending-P decode; intentionally never calls Hungarian."""
    order = torch.argsort(p.detach().reshape(-1), descending=True).cpu().numpy()
    place = np.full(NFRAG, -1, dtype=np.int64)  # clean position -> shuffled tile
    used_tiles = np.zeros(NFRAG, dtype=bool)
    for flat in order:
        tile, position = divmod(int(flat), NFRAG)
        if place[position] < 0 and not used_tiles[tile]:
            place[position], used_tiles[tile] = tile, True
            if used_tiles.all():
                return place
    raise RuntimeError("greedy decode exhausted candidates before a bijection")


def hungarian_decode(p: Tensor) -> np.ndarray:
    """Maximum-weight collision-free tile-to-position assignment."""
    tile, position = linear_sum_assignment(-p.detach().float().cpu().numpy())
    place = np.full(NFRAG, -1, dtype=np.int64)
    place[position] = tile
    if np.any(place < 0):
        raise RuntimeError("Hungarian decode did not cover the board")
    return place


def report(p: Tensor, perm: Tensor, value: float) -> dict[str, float]:
    place = hungarian_decode(p)
    truth = np.argsort(perm.detach().cpu().numpy().astype(np.int64))
    entropy = -(p * p.clamp_min(1.0e-12).log()).sum(dim=1).mean() / math.log(NFRAG)
    return {
        "qap": value,
        "placement": float(np.mean(place == truth)),
        "neighbour": neighbour_accuracy(place, truth)[0],
        "row_max": float(p.max(dim=1).values.mean()),
        "col_max": float(p.max(dim=0).values.mean()),
        "entropy": float(entropy),
        "ds_error": float(torch.maximum((p.sum(1) - 1).abs().max(), (p.sum(0) - 1).abs().max())),
    }


def perfect_rd(perm: Tensor) -> tuple[Tensor, Tensor]:
    """Synthetic exact R/D matrices for the label-aware `--oracle` contract."""
    right = torch.zeros((NFRAG, NFRAG), device=perm.device)
    down = torch.zeros_like(right)
    tiles, cells = torch.arange(NFRAG, device=perm.device), perm.long()
    inverse = torch.empty_like(cells)
    inverse.scatter_(0, cells, tiles)
    has_right, has_down = cells.remainder(GRID).lt(GRID - 1), cells.lt(NFRAG - GRID)
    right[tiles[has_right], inverse[cells[has_right] + 1]] = 1.0
    down[tiles[has_down], inverse[cells[has_down] + GRID]] = 1.0
    return right, down


def run_oracle(args: argparse.Namespace, device: torch.device) -> None:
    """Recovery gate: perfect relations, but an entirely label-free QAP start."""
    perm = torch.from_numpy(np.random.default_rng(args.seed).permutation(NFRAG)).to(device)
    right, down = perfect_rd(perm)
    # `optimize_qap` draws only the globally seeded noise configured in main;
    # it never receives perm.  The labels above are used solely to construct
    # the perfect oracle R/D and to score recovery after the optimisation.
    p, value = optimize_qap(right, down, args)
    metrics = report(p, perm, value)
    if metrics["placement"] < 0.999 or metrics["neighbour"] < 0.999:
        raise AssertionError(f"unseeded perfect-R/D QAP recovery gate failed: {metrics}")
    print(f"[oracle recovery contract: perfect R/D, unseeded init] {metrics}", flush=True)


def parse_args() -> argparse.Namespace:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker-ckpt", "--ranker_ckpt", default=_default_ranker(workspace))
    parser.add_argument("--affinity-ckpt", "--affinity_ckpt", default="")
    parser.add_argument("--affinity-ckpt2", "--affinity_ckpt2", default="")
    parser.add_argument("--top-k", "--top_k", type=int, default=0)
    parser.add_argument("--n", type=int, default=1, help="fresh synthetic held-out images")
    parser.add_argument("--steps", type=int, default=24, help="bounded Adam steps per image")
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.08)
    parser.add_argument("--sinkhorn-rounds", type=int, default=12)
    parser.add_argument("--entropy", type=float, default=0.02, help="early diffuse-assignment bonus")
    parser.add_argument("--init-noise", type=float, default=0.02)
    parser.add_argument("--seed-confidence", type=float, default=0.70)
    parser.add_argument("--seed-bonus", type=float, default=0.50, help="0 disables reciprocal seed bonus")
    parser.add_argument("--pair-batch", "--pair_batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=SEED + 7411)
    parser.add_argument("--device", default=None)
    parser.add_argument("--oracle", action="store_true", help="unseeded perfect-R/D recovery gate; no checkpoint")
    args = parser.parse_args()
    if args.n < 1 or args.steps < 1 or args.sinkhorn_rounds < 1 or args.pair_batch < 1:
        parser.error("--n, --steps, --sinkhorn-rounds, and --pair-batch must be positive")
    if args.top_k < 0 or args.top_k >= NFRAG:
        parser.error(f"--top-k must lie in [0,{NFRAG - 1}]")
    if args.lr <= 0 or args.temperature_end <= 0 or args.temperature_start < args.temperature_end:
        parser.error("require --lr>0 and 0<temperature-end<=temperature-start")
    if args.entropy < 0 or args.init_noise < 0 or args.seed_bonus < 0 or not 0 <= args.seed_confidence <= 1:
        parser.error("entropy/noise/seed-bonus must be nonnegative and confidence must lie in [0,1]")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    if args.oracle:
        run_oracle(args, device)
        return

    model, payload = load_ranker(args.ranker_ckpt, device)
    affinity_path, affinity_path2, top_k = _graph_paths(payload, args)
    affinity, _, _ = load_frozen_affinity(affinity_path, device)
    affinity2: nn.Module | None = load_frozen_affinity(affinity_path2, device)[0] if affinity_path2 else None
    _, names = train_val_split()
    if args.n > len(names):
        raise ValueError(f"--n exceeds held-out pool ({len(names)})")
    dataset = CanvasDataset(names[:args.n], real_prob=0.0, seed=args.seed)
    print(
        f"seeded QAP: device={device} n={args.n} steps={args.steps} topK={top_k} "
        f"seed conf>={args.seed_confidence:.2f} bonus={args.seed_bonus:.2f}; labels only score output",
        flush=True,
    )
    rows: list[dict[str, float]] = []
    for index in range(args.n):
        sample = dataset[index]
        tiles, perm = sample["tiles"].to(device), sample["perm"].to(device).long()
        candidates_b, valid_b = mine_affinity_candidates(
            affinity, tiles.unsqueeze(0), candidate_k=top_k, device=device, affinity_secondary=affinity2
        )
        candidates, valid = candidates_b[0], valid_b[0]
        scores = score_full_graph(model, tiles, candidates, valid, pair_batch=args.pair_batch, device=device)
        right, down = dense_rd(candidates, scores)
        seed_count = add_reciprocal_seed_bonus(
            right, down, candidates, scores, args.seed_confidence, args.seed_bonus
        )
        p, value = optimize_qap(right, down, args)
        metrics = report(p, perm, value)
        metrics["seeds"] = float(seed_count)
        rows.append(metrics)
        print(f"image {index + 1}/{args.n}: seeds={seed_count} {metrics}", flush=True)
    if len(rows) > 1:
        print("mean: " + str({key: float(np.mean([row[key] for row in rows])) for key in rows[0]}), flush=True)


if __name__ == "__main__":
    main()
