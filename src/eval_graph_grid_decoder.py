"""Procedural gate for a learned global decoder of noisy oriented grid graphs.

The model never sees pixels, node identities, or a recovered puzzle layout.
Every generated sample randomly relabels the 24x24 grid nodes.  Its only useful
signal is a sparse, weighted set of candidate RIGHT/DOWN relations and, during
denoising training, an independently corrupted current permutation.

This is deliberately a mechanism gate.  Passing it justifies extracting real
ranker graphs; failing it closes the learned graph-decoder branch cheaply.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from config import GRID, NFRAG, SEED
from placement_metrics import neighbour_accuracy


K = 16


@dataclass
class GraphBatch:
    true_slot: Tensor
    current_slot: Tensor
    noise: Tensor
    r_dst: Tensor
    r_weight: Tensor
    d_dst: Tensor
    d_weight: Tensor


def random_permutations(batch: int, device: torch.device) -> Tensor:
    return torch.stack([torch.randperm(NFRAG, device=device) for _ in range(batch)])


def true_neighbours(true_slot: Tensor) -> tuple[Tensor, Tensor]:
    """Return node identities immediately right/down of every relabelled node."""
    batch = true_slot.shape[0]
    inverse = torch.empty_like(true_slot)
    nodes = torch.arange(NFRAG, device=true_slot.device).expand(batch, -1)
    inverse.scatter_(1, true_slot, nodes)
    right = torch.full_like(true_slot, -1)
    down = torch.full_like(true_slot, -1)
    has_right = true_slot.remainder(GRID).lt(GRID - 1)
    has_down = true_slot.lt(NFRAG - GRID)
    right[has_right] = inverse.gather(1, (true_slot + 1).clamp_max(NFRAG - 1))[has_right]
    down[has_down] = inverse.gather(1, (true_slot + GRID).clamp_max(NFRAG - 1))[has_down]
    return right, down


def corrupt_slots(true_slot: Tensor, noise: Tensor) -> Tensor:
    """Permute a fraction of target slots while preserving a strict bijection."""
    result = true_slot.clone()
    for row in range(true_slot.shape[0]):
        count = min(NFRAG, max(2, int(round(float(noise[row]) * NFRAG))))
        selected = torch.randperm(NFRAG, device=true_slot.device)[:count]
        shuffled = selected[torch.randperm(count, device=true_slot.device)]
        result[row, selected] = true_slot[row, shuffled]
    return result


def _candidate_rows(
    truth: Tensor,
    *,
    exact: bool,
    recall: float,
    decoy: Tensor | None,
    decoy_fraction: float,
) -> tuple[Tensor, Tensor]:
    """Make fixed-K candidate rows with calibrated conditional true ranks."""
    batch, count = truth.shape
    device = truth.device
    nodes = torch.arange(count, device=device).view(1, count, 1)
    dst = (nodes + torch.randint(1, count, (batch, count, K), device=device)) % count
    weight = torch.zeros((batch, count, K), device=device)
    valid = truth.ge(0)
    if exact:
        rows, sources = valid.nonzero(as_tuple=True)
        dst[rows, sources, 0] = truth[rows, sources]
        weight[rows, sources, 0] = 1.0
        return dst, weight

    rank_logits = -0.35 * torch.arange(K, device=device, dtype=torch.float32)
    rank_logits = rank_logits.view(1, 1, K) + 0.04 * torch.randn_like(weight)
    weight = torch.softmax(rank_logits, dim=-1)
    present = valid & torch.rand((batch, count), device=device).lt(recall)
    u = torch.rand((batch, count), device=device)
    middle = torch.randint(1, min(5, K), (batch, count), device=device)
    tail = torch.randint(min(5, K - 1), K, (batch, count), device=device)
    # The recorded R@1/R@5 are unconditional over all valid directions while
    # ``recall`` controls whether the truth enters the candidate pool at all.
    # Therefore the within-pool rank probabilities must be divided by recall.
    conditional_r1 = min(1.0, 0.27 / max(recall, 1.0e-6))
    conditional_r5 = min(1.0, 0.49 / max(recall, 1.0e-6))
    rank = torch.where(
        u < conditional_r1,
        torch.zeros_like(middle),
        torch.where(u < conditional_r5, middle, tail),
    )

    # Prevent an accidental duplicate truth from making the calibrated rank optimistic.
    collision = dst.eq(truth.clamp_min(0).unsqueeze(-1)) & present.unsqueeze(-1)
    dst = torch.where(collision, (dst + 37) % count, dst)
    collision_self = dst.eq(nodes)
    dst = torch.where(collision_self, (dst + 1) % count, dst)

    rows, sources = present.nonzero(as_tuple=True)
    dst[rows, sources, rank[rows, sources]] = truth[rows, sources]
    if decoy is not None and decoy_fraction > 0.0:
        use_decoy = decoy.ge(0) & torch.rand((batch, count), device=device).lt(decoy_fraction)
        dr = torch.where(present & rank.eq(0), torch.ones_like(rank), torch.zeros_like(rank))
        rows, sources = use_decoy.nonzero(as_tuple=True)
        dst[rows, sources, dr[rows, sources]] = decoy[rows, sources]
    return dst, weight


def make_batch(
    batch: int,
    device: torch.device,
    *,
    mode: str,
    recall: float = 0.67,
    noise: Tensor | None = None,
) -> GraphBatch:
    if mode not in {"exact", "noisy", "decoy"}:
        raise ValueError(f"unknown graph mode {mode!r}")
    true_slot = random_permutations(batch, device)
    if noise is None:
        noise = torch.where(
            torch.rand(batch, device=device) < 0.5,
            torch.ones(batch, device=device),
            0.25 + 0.65 * torch.rand(batch, device=device),
        )
    current_slot = corrupt_slots(true_slot, noise)
    true_r, true_d = true_neighbours(true_slot)
    decoy_r = decoy_d = None
    decoy_fraction = 0.0
    if mode == "decoy":
        decoy_r, decoy_d = true_neighbours(random_permutations(batch, device))
        decoy_fraction = 0.25
    exact = mode == "exact"
    r_dst, r_weight = _candidate_rows(
        true_r, exact=exact, recall=recall, decoy=decoy_r, decoy_fraction=decoy_fraction
    )
    d_dst, d_weight = _candidate_rows(
        true_d, exact=exact, recall=recall, decoy=decoy_d, decoy_fraction=decoy_fraction
    )
    return GraphBatch(true_slot, current_slot, noise, r_dst, r_weight, d_dst, d_weight)


def directional_aggregate(h: Tensor, dst: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    """Weighted outgoing and incoming messages for one physical direction."""
    batch, count, width = h.shape
    # Under AMP the recurrent state is fp16 while procedural edge weights are
    # generated in fp32.  scatter_add requires an exact dtype match.
    weight = weight.to(dtype=h.dtype)
    flat_dst = dst.reshape(batch, -1)
    expanded = flat_dst.unsqueeze(-1).expand(-1, -1, width)
    neighbours = h.gather(1, expanded).reshape(batch, count, K, width)
    out_mass = weight.sum(dim=2, keepdim=True)
    outgoing = (neighbours * weight.unsqueeze(-1)).sum(dim=2) / out_mass.clamp_min(1.0e-6)
    source_messages = (h.unsqueeze(2) * weight.unsqueeze(-1)).reshape(batch, -1, width)
    incoming = torch.zeros_like(h).scatter_add_(1, expanded, source_messages)
    incoming_mass = torch.zeros((batch, count), device=h.device, dtype=h.dtype)
    incoming_mass.scatter_add_(1, flat_dst, weight.reshape(batch, -1))
    incoming = incoming / incoming_mass.unsqueeze(-1).clamp_min(1.0e-6)
    return outgoing, incoming


def edge_statistics(dst: Tensor, weight: Tensor) -> Tensor:
    batch, count, _ = weight.shape
    incoming = torch.zeros((batch, count), device=weight.device, dtype=weight.dtype)
    incoming.scatter_add_(1, dst.reshape(batch, -1), weight.reshape(batch, -1))
    out_sum = weight.sum(-1)
    out_max = weight.max(-1).values
    entropy = -(weight * weight.clamp_min(1.0e-12).log()).sum(-1) / math.log(K)
    return torch.stack((out_sum, out_max, entropy, incoming), dim=-1)


class GraphGridDecoder(nn.Module):
    """Permutation-equivariant recurrent decoder for oriented sparse graphs."""

    def __init__(self, width: int = 48, rounds: int = 24) -> None:
        super().__init__()
        if width < 16 or rounds < 1:
            raise ValueError("width>=16 and rounds>=1 are required")
        self.width, self.rounds = int(width), int(rounds)
        coord = max(8, width // 4)
        self.row_embedding = nn.Embedding(GRID, coord)
        self.col_embedding = nn.Embedding(GRID, coord)
        self.time_embedding = nn.Sequential(nn.Linear(3, coord), nn.GELU(), nn.Linear(coord, coord))
        self.initial = nn.Sequential(
            nn.Linear(3 * coord + 8, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.message = nn.Sequential(
            nn.LayerNorm(7 * width),
            nn.Linear(7 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.update = nn.GRUCell(width, width)
        self.final = nn.LayerNorm(width)
        self.row_head = nn.Linear(width, GRID)
        self.col_head = nn.Linear(width, GRID)

    def forward(self, graph: GraphBatch) -> tuple[Tensor, Tensor]:
        row, col = torch.div(graph.current_slot, GRID, rounding_mode="floor"), graph.current_slot % GRID
        time_features = torch.stack(
            (graph.noise, torch.sin(math.pi * graph.noise), torch.cos(math.pi * graph.noise)), dim=-1
        )
        time = self.time_embedding(time_features).unsqueeze(1).expand(-1, NFRAG, -1)
        stats = torch.cat(
            (edge_statistics(graph.r_dst, graph.r_weight), edge_statistics(graph.d_dst, graph.d_weight)),
            dim=-1,
        )
        h = self.initial(torch.cat((self.row_embedding(row), self.col_embedding(col), time, stats), dim=-1))
        for _ in range(self.rounds):
            r_out, r_in = directional_aggregate(h, graph.r_dst, graph.r_weight)
            d_out, d_in = directional_aggregate(h, graph.d_dst, graph.d_weight)
            mean = h.mean(1, keepdim=True).expand_as(h)
            maximum = h.amax(1, keepdim=True).expand_as(h)
            proposal = self.message(torch.cat((h, r_out, r_in, d_out, d_in, mean, maximum), dim=-1))
            h = self.update(proposal.reshape(-1, self.width), h.reshape(-1, self.width)).reshape_as(h)
        h = self.final(h)
        return self.row_head(h), self.col_head(h)


def assignment_loss(row_logits: Tensor, col_logits: Tensor, true_slot: Tensor) -> tuple[Tensor, dict[str, float]]:
    target_row, target_col = torch.div(true_slot, GRID, rounding_mode="floor"), true_slot % GRID
    row_ce = F.cross_entropy(row_logits.flatten(0, 1), target_row.flatten())
    col_ce = F.cross_entropy(col_logits.flatten(0, 1), target_col.flatten())
    row_count = row_logits.softmax(-1).sum(1) / GRID
    col_count = col_logits.softmax(-1).sum(1) / GRID
    capacity = 0.5 * ((row_count - 1).square().mean() + (col_count - 1).square().mean())
    loss = 0.5 * (row_ce + col_ce) + 0.05 * capacity
    return loss, {"loss": float(loss.detach()), "row_ce": float(row_ce.detach()),
                  "col_ce": float(col_ce.detach()), "capacity": float(capacity.detach())}


def decode(row_logits: Tensor, col_logits: Tensor) -> list[np.ndarray]:
    grid_row = torch.arange(NFRAG, device=row_logits.device) // GRID
    grid_col = torch.arange(NFRAG, device=row_logits.device) % GRID
    scores = row_logits[:, :, grid_row] + col_logits[:, :, grid_col]
    results: list[np.ndarray] = []
    for score in scores.detach().float().cpu().numpy():
        nodes, slots = linear_sum_assignment(-score)
        node_to_slot = np.empty(NFRAG, dtype=np.int64)
        node_to_slot[nodes] = slots
        results.append(node_to_slot)
    return results


def metrics_for(node_to_slot: np.ndarray, true_slot: np.ndarray) -> dict[str, float]:
    place = np.empty(NFRAG, dtype=np.int64)
    truth = np.empty(NFRAG, dtype=np.int64)
    place[node_to_slot] = np.arange(NFRAG)
    truth[true_slot] = np.arange(NFRAG)
    neighbour, right, down = neighbour_accuracy(place, truth)
    return {"placement": float(np.mean(node_to_slot == true_slot)), "neighbour": neighbour,
            "right": right, "down": down}


def shuffled_control(graph: GraphBatch) -> GraphBatch:
    mappings = random_permutations(graph.true_slot.shape[0], graph.true_slot.device)
    def remap(dst: Tensor) -> Tensor:
        return mappings.gather(1, dst.reshape(dst.shape[0], -1)).reshape_as(dst)
    return GraphBatch(graph.true_slot, graph.current_slot, graph.noise, remap(graph.r_dst),
                      graph.r_weight, remap(graph.d_dst), graph.d_weight)


@torch.no_grad()
def evaluate(model: GraphGridDecoder, device: torch.device, mode: str, recall: float,
             samples: int, *, shuffle: bool = False) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for _ in range(samples):
        graph = make_batch(1, device, mode=mode, recall=recall, noise=torch.ones(1, device=device))
        if shuffle:
            graph = shuffled_control(graph)
        row_logits, col_logits = model(graph)
        predicted = decode(row_logits.float(), col_logits.float())[0]
        rows.append(metrics_for(predicted, graph.true_slot[0].cpu().numpy()))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def gate_report(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    checks = {
        "exact": metrics["exact"]["placement"] >= 0.995 and metrics["exact"]["neighbour"] >= 0.995,
        "recall_067": metrics["recall_067"]["placement"] >= 0.02 and metrics["recall_067"]["neighbour"] >= 0.15,
        "recall_050": metrics["recall_050"]["neighbour"] >= 0.08,
        "decoy": metrics["decoy"]["placement"] >= 0.01 and metrics["decoy"]["neighbour"] >= 0.08,
        "shuffled": metrics["shuffled"]["placement"] <= 0.01 and metrics["shuffled"]["neighbour"] <= 0.02,
    }
    return {"thresholds": {"exact": {"placement": 0.995, "neighbour": 0.995},
                            "recall_067": {"placement": 0.02, "neighbour": 0.15},
                            "recall_050": {"neighbour": 0.08},
                            "decoy": {"placement": 0.01, "neighbour": 0.08},
                            "shuffled_max": {"placement": 0.01, "neighbour": 0.02}},
            "checks": checks, "passed": bool(all(checks.values()))}


def relabel(graph: GraphBatch, order: Tensor) -> GraphBatch:
    """Node relabel helper used only by the equivariance smoke contract."""
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(NFRAG, device=order.device)
    def change(dst: Tensor) -> Tensor:
        return inverse[dst[:, order]]
    return GraphBatch(graph.true_slot[:, order], graph.current_slot[:, order], graph.noise,
                      change(graph.r_dst), graph.r_weight[:, order], change(graph.d_dst),
                      graph.d_weight[:, order])


def smoke() -> dict[str, Any]:
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    graph = make_batch(1, device, mode="exact", noise=torch.ones(1))
    assert torch.equal(torch.sort(graph.true_slot[0]).values, torch.arange(NFRAG))
    assert torch.equal(torch.sort(graph.current_slot[0]).values, torch.arange(NFRAG))
    h = torch.arange(NFRAG * 2, dtype=torch.float32).reshape(1, NFRAG, 2)
    dst, weight = torch.zeros((1, NFRAG, K), dtype=torch.long), torch.zeros((1, NFRAG, K))
    dst[0, 0, 0], weight[0, 0, 0] = 1, 1.0
    outgoing, incoming = directional_aggregate(h, dst, weight)
    assert torch.equal(outgoing[0, 0], h[0, 1]) and torch.equal(incoming[0, 1], h[0, 0])
    model = GraphGridDecoder(width=16, rounds=2).eval()
    with torch.no_grad():
        first = model(graph)
        order = torch.randperm(NFRAG)
        second = model(relabel(graph, order))
    equivariance = max(float((second[i] - first[i][:, order]).abs().max()) for i in range(2))
    if equivariance > 2.0e-5:
        raise AssertionError(f"node permutation equivariance failed: {equivariance}")
    true_row, true_col = graph.true_slot // GRID, graph.true_slot % GRID
    row_logits = F.one_hot(true_row, GRID).float() * 20.0
    col_logits = F.one_hot(true_col, GRID).float() * 20.0
    decoded = decode(row_logits, col_logits)[0]
    contract = metrics_for(decoded, graph.true_slot[0].numpy())
    if contract["placement"] != 1.0 or contract["neighbour"] != 1.0:
        raise AssertionError(f"Hungarian metric contract failed: {contract}")
    train_probe = GraphGridDecoder(width=16, rounds=1)
    probe_loss, _ = assignment_loss(*train_probe(graph), graph.true_slot)
    probe_loss.backward()
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in train_probe.parameters()):
        raise AssertionError("non-finite gradient in GraphGRU smoke backward")
    return {"generator": "ok", "directed_aggregation": "ok", "equivariance_max": equivariance,
            "hungarian": contract, "backward_loss": float(probe_loss.detach())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--exact-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-n", type=int, default=8)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=SEED + 7601)
    parser.add_argument("--report", default="E:/pazzle_work/gates/graph_grid_decoder_gate.json")
    parser.add_argument("--out", default="E:/pazzle_work/ckpt/graph_grid_decoder.pt")
    args = parser.parse_args()
    if min(args.train_steps, args.exact_steps, args.batch_size, args.eval_n, args.rounds) < 1:
        parser.error("training, evaluation, and model counts must be positive")
    if args.exact_steps > args.train_steps or args.lr <= 0:
        parser.error("--exact-steps must not exceed --train-steps and --lr must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2), flush=True)
        return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = GraphGridDecoder(args.width, args.rounds).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.time()
    model.train()
    for step in range(1, args.train_steps + 1):
        if step <= args.exact_steps:
            mode, recall = "exact", 1.0
        else:
            progress = (step - args.exact_steps) / max(args.train_steps - args.exact_steps, 1)
            mode = "noisy"
            recall = 0.95 + progress * (0.67 - 0.95)
        graph = make_batch(args.batch_size, device, mode=mode, recall=recall)
        context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with context:
            row_logits, col_logits = model(graph)
            loss, terms = assignment_loss(row_logits.float(), col_logits.float(), graph.true_slot)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer); scaler.update()
        if step == 1 or step % 25 == 0:
            print(f"step={step:04d}/{args.train_steps} mode={mode} recall={recall:.3f} "
                  f"loss={terms['loss']:.4f} row={terms['row_ce']:.4f} col={terms['col_ce']:.4f}", flush=True)

    metrics = {
        "exact": evaluate(model, device, "exact", 1.0, args.eval_n),
        "recall_067": evaluate(model, device, "noisy", 0.67, args.eval_n),
        "recall_050": evaluate(model, device, "noisy", 0.50, args.eval_n),
        "decoy": evaluate(model, device, "decoy", 0.67, args.eval_n),
        "shuffled": evaluate(model, device, "noisy", 0.67, args.eval_n, shuffle=True),
    }
    gate = gate_report(metrics)
    payload = {"schema_version": 1, "experiment": "procedural_graph_grid_decoder",
               "args": vars(args), "elapsed_seconds": time.time() - started, "metrics": metrics, "gate": gate}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    torch.save({"model": model.state_dict(), "model_kwargs": {"width": args.width, "rounds": args.rounds},
                "payload": payload}, args.out)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({"metrics": metrics, "gate": gate, "out": args.out, "report": args.report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
