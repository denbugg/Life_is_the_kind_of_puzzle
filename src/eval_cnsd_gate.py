"""Procedural gate for a Commuting Nilpotent Shift Decoder (CNSD).

No real data are read. Randomly relabelled grids become noisy oriented candidate
graphs; jointly optimised partial shifts R/D must commute, form finite boundary
chains, and induce one tile per boundary-distance pair. The augmented dustbin has
capacity ``grid`` and its self-edge is forbidden. Absolute cells only score the
final result. Use ``--smoke`` for the grid-4 algebra/gradient contract.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

@dataclass(frozen=True)
class ProceduralGraph:
    grid: int
    placement: np.ndarray  # clean cell -> shuffled tile
    true_right: np.ndarray
    true_down: np.ndarray
    candidates: np.ndarray
    weight_right: np.ndarray
    weight_down: np.ndarray
    candidate_recall: float
    conditional_r1: float

def _truth(grid: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = grid * grid
    placement = rng.permutation(n).astype(np.int64)
    targets = np.full((4, n), -1, dtype=np.int64)
    right = np.zeros((n, n), dtype=bool)
    down = np.zeros((n, n), dtype=bool)
    for cell, tile in enumerate(placement):
        row, col = divmod(cell, grid)
        if row:
            targets[UP, tile] = placement[cell - grid]
        if row + 1 < grid:
            target = placement[cell + grid]
            targets[DOWN, tile] = target
            down[tile, target] = True
        if col:
            targets[LEFT, tile] = placement[cell - 1]
        if col + 1 < grid:
            target = placement[cell + 1]
            targets[RIGHT, tile] = target
            right[tile, target] = True
    return placement, targets, right, down

def _sample_rank(rng: np.random.Generator, k: int) -> int:
    bins = [
        (0.272, 1, 1),
        (0.250, 2, min(5, k)),
        (0.230, 6, min(20, k)),
        (0.248, 21, k),
    ]
    available = [(p, lo, hi) for p, lo, hi in bins if lo <= hi]
    probabilities = np.asarray([row[0] for row in available], dtype=np.float64)
    probabilities /= probabilities.sum()
    index = int(rng.choice(len(available), p=probabilities))
    _, low, high = available[index]
    return int(rng.integers(low, high + 1))

def make_graph(
    *, grid: int, k: int, recall: float, mode: str, seed: int, floor: float
) -> ProceduralGraph:
    rng = np.random.default_rng(seed)
    n = grid * grid
    k = min(k, n - 1)
    if k < min(4, n - 1):
        raise ValueError(f"k={k} cannot retain all possible physical neighbours")
    placement, targets, true_right, true_down = _truth(grid, rng)
    candidates = np.zeros((n, n), dtype=bool)
    # Candidate presence is shared by both ends of a physical pair.  This is a
    # useful approximation to an affinity graph, while directional ranks below
    # remain independent.
    physical = np.argwhere(true_right | true_down)
    for source, target in physical:
        if mode == "exact" or rng.random() < recall:
            candidates[source, target] = True
            candidates[target, source] = True
    for source in range(n):
        need = k - int(candidates[source].sum())
        if need:
            allowed = np.flatnonzero(~candidates[source])
            allowed = allowed[allowed != source]
            chosen = rng.choice(allowed, size=need, replace=False)
            candidates[source, chosen] = True
    raw = np.full((4, n, n), floor, dtype=np.float32)
    covered = 0
    rank_one = 0
    valid_true = 0
    rank_scores = -np.log(np.arange(1, k + 1, dtype=np.float32))
    for direction in range(4):
        for source in range(n):
            target = int(targets[direction, source])
            row = np.flatnonzero(candidates[source])
            rng.shuffle(row)
            if target >= 0:
                valid_true += 1
            if target >= 0 and candidates[source, target]:
                covered += 1
                rank = 1 if mode == "exact" else _sample_rank(rng, k)
                rank_one += int(rank == 1)
                others = row[row != target]
                row = np.insert(others, rank - 1, target)
            raw[direction, source, row] = rank_scores
    # Average a directed assertion with the inverse assertion.  One-sided
    # decoys remain possible, but reciprocal evidence is naturally stronger.
    weight_right = 0.5 * (raw[RIGHT] + raw[LEFT].T)
    weight_down = 0.5 * (raw[DOWN] + raw[UP].T)
    np.fill_diagonal(weight_right, floor * 2.0)
    np.fill_diagonal(weight_down, floor * 2.0)
    true_total = int(true_right.sum() + true_down.sum())
    retained = int((true_right & candidates).sum() + (true_down & candidates).sum())
    return ProceduralGraph(
        grid=grid,
        placement=placement,
        true_right=true_right,
        true_down=true_down,
        candidates=candidates,
        weight_right=weight_right,
        weight_down=weight_down,
        candidate_recall=retained / true_total,
        conditional_r1=rank_one / max(covered, 1),
    )

def augmented_sinkhorn(logits: Tensor, *, grid: int, temperature: float, rounds: int) -> Tensor:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("augmented Sinkhorn logits must be square")
    if temperature <= 0.0 or rounds < 1:
        raise ValueError("temperature and rounds must be positive")
    n_aug = logits.shape[0]
    log_marginal = logits.new_zeros(n_aug)
    log_marginal[-1] = math.log(grid)
    work = logits / temperature
    forbidden = torch.zeros_like(work, dtype=torch.bool)
    forbidden[-1, -1] = True
    for _ in range(rounds):
        work = work.masked_fill(forbidden, -torch.inf)
        work = work + log_marginal[:, None] - torch.logsumexp(work, dim=1, keepdim=True)
        work = work.masked_fill(forbidden, -torch.inf)
        work = work + log_marginal[None, :] - torch.logsumexp(work, dim=0, keepdim=True)
    return work.masked_fill(forbidden, -torch.inf).exp()

def boundary_messages(augmented: Tensor, grid: int) -> Tensor:
    n = grid * grid
    shift = augmented[:n, :n]
    values = [augmented[:n, -1]]
    for _ in range(1, grid):
        values.append(shift @ values[-1])
    return torch.stack(values, dim=1)

def geometry(
    augmented_right: Tensor, augmented_down: Tensor, grid: int
) -> tuple[dict[str, Tensor], Tensor]:
    n = grid * grid
    right = augmented_right[:n, :n]
    down = augmented_down[:n, :n]
    commutator = right @ down - down @ right
    commutation = commutator.square().sum() / n
    q_right = boundary_messages(augmented_right, grid)
    q_down = boundary_messages(augmented_down, grid)
    def one_chain(augmented: Tensor, shift: Tensor, q: Tensor) -> Tensor:
        incoming = augmented[-1, :n]
        coverage = (q.sum(dim=1) - 1.0).square().mean()
        endpoint = (q[:, -1] - incoming).square().mean()
        nilpotent_tail = (shift @ q[:, -1]).square().mean()
        return coverage + endpoint + nilpotent_tail
    chain = one_chain(augmented_right, right, q_right) + one_chain(
        augmented_down, down, q_down
    )
    row_probability = q_down.flip(dims=(1,))
    column_probability = q_right.flip(dims=(1,))
    joint = torch.einsum("ir,ic->irc", row_probability, column_probability)
    occupancy = (joint.sum(dim=0) - 1.0).square().mean()
    tile_mass = (joint.sum(dim=(1, 2)) - 1.0).square().mean()
    return {
        "commutation": commutation,
        "chain": chain,
        "occupancy": occupancy,
        "tile_mass": tile_mass,
    }, joint.reshape(n, n)

def exact_augmented(graph: ProceduralGraph, direction: str, device: torch.device) -> Tensor:
    grid = graph.grid
    n = grid * grid
    truth = graph.true_right if direction == "right" else graph.true_down
    matrix = torch.zeros((n + 1, n + 1), dtype=torch.float32, device=device)
    source, target = np.nonzero(truth)
    matrix[torch.as_tensor(source, device=device), torch.as_tensor(target, device=device)] = 1.0
    outgoing = ~truth.any(axis=1)
    incoming = ~truth.any(axis=0)
    matrix[torch.as_tensor(np.flatnonzero(outgoing), device=device), -1] = 1.0
    matrix[-1, torch.as_tensor(np.flatnonzero(incoming), device=device)] = 1.0
    return matrix

def decode(cell_mass: Tensor) -> np.ndarray:
    cost = -cell_mass.detach().float().cpu().numpy()
    tiles, cells = linear_sum_assignment(cost)
    placement = np.full(len(tiles), -1, dtype=np.int64)
    placement[cells] = tiles
    if np.any(placement < 0):
        raise RuntimeError("Hungarian decode did not produce a full placement")
    return placement

def placement_metrics(prediction: np.ndarray, graph: ProceduralGraph) -> dict[str, float]:
    grid = graph.grid
    correct = 0
    missing_correct = 0
    missing_total = 0
    total = 2 * grid * (grid - 1)
    for row in range(grid):
        for col in range(grid):
            cell = row * grid + col
            source = int(prediction[cell])
            if col + 1 < grid:
                target = int(prediction[cell + 1])
                hit = bool(graph.true_right[source, target])
                correct += int(hit)
                missing_correct += int(hit and not graph.candidates[source, target])
            if row + 1 < grid:
                target = int(prediction[cell + grid])
                hit = bool(graph.true_down[source, target])
                correct += int(hit)
                missing_correct += int(hit and not graph.candidates[source, target])
    missing_total = int(
        (graph.true_right & ~graph.candidates).sum()
        + (graph.true_down & ~graph.candidates).sum()
    )
    return {
        "placement": float(np.mean(prediction == graph.placement)),
        "neighbour": correct / total,
        "correct_edges": float(correct),
        "missing_edges": float(missing_total),
        "missing_edges_recovered": float(missing_correct),
        "missing_edge_recall": missing_correct / max(missing_total, 1),
        "completion_fraction_of_correct": missing_correct / max(correct, 1),
    }

def _initial_logits(weight: Tensor, grid: int, boundary_logit: float) -> Tensor:
    n = grid * grid
    logits = weight.new_full((n + 1, n + 1), boundary_logit)
    logits[:n, :n] = weight
    logits[-1, -1] = 0.0  # always masked by augmented_sinkhorn
    return logits

def solve(
    graph: ProceduralGraph, args: argparse.Namespace, device: torch.device, deadline: float
) -> tuple[dict[str, float], list[dict[str, float]]]:
    grid = graph.grid
    n = grid * grid
    wr = torch.as_tensor(graph.weight_right, dtype=torch.float32, device=device)
    wd = torch.as_tensor(graph.weight_down, dtype=torch.float32, device=device)
    logits_right = nn.Parameter(_initial_logits(wr, grid, args.boundary_logit))
    logits_down = nn.Parameter(_initial_logits(wd, grid, args.boundary_logit))
    optimizer = torch.optim.Adam((logits_right, logits_down), lr=args.lr)
    trace: list[dict[str, float]] = []
    final_augmented: tuple[Tensor, Tensor] | None = None
    for step in range(args.steps):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"CNSD exceeded --timeout-seconds={args.timeout_seconds}")
        fraction = step / max(args.steps - 1, 1)
        temperature = math.exp(
            math.log(args.temperature_start) * (1.0 - fraction)
            + math.log(args.temperature_end) * fraction
        )
        ar = augmented_sinkhorn(
            logits_right, grid=grid, temperature=temperature, rounds=args.sinkhorn_rounds
        )
        ad = augmented_sinkhorn(
            logits_down, grid=grid, temperature=temperature, rounds=args.sinkhorn_rounds
        )
        terms, _ = geometry(ar, ad, grid)
        right, down = ar[:n, :n], ad[:n, :n]
        real_mass = float(n - grid)
        unary = ((right * wr).sum() + (down * wd).sum()) / (2.0 * real_mass)
        entropy = -(
            (right * right.clamp_min(1.0e-12).log()).sum()
            + (down * down.clamp_min(1.0e-12).log()).sum()
        ) / (2.0 * real_mass * math.log(n))
        loss = (
            -unary
            + args.lambda_commutation * terms["commutation"]
            + args.lambda_chain * terms["chain"]
            + args.lambda_occupancy * (terms["occupancy"] + terms["tile_mass"])
            - args.entropy * (1.0 - fraction) * entropy
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((logits_right, logits_down), args.grad_clip)
        optimizer.step()
        if step in {0, args.steps // 4, args.steps // 2, 3 * args.steps // 4, args.steps - 1}:
            trace.append({
                "step": float(step + 1), "temperature": temperature,
                "loss": float(loss.detach()), "unary": float(unary.detach()),
                "commutation": float(terms["commutation"].detach()),
                "chain": float(terms["chain"].detach()),
                "occupancy": float(terms["occupancy"].detach()),
                "tile_mass": float(terms["tile_mass"].detach()),
            })
        final_augmented = (ar.detach(), ad.detach())
    assert final_augmented is not None
    ar = augmented_sinkhorn(
        logits_right.detach(), grid=grid, temperature=args.temperature_end,
        rounds=max(args.sinkhorn_rounds, 32),
    )
    ad = augmented_sinkhorn(
        logits_down.detach(), grid=grid, temperature=args.temperature_end,
        rounds=max(args.sinkhorn_rounds, 32),
    )
    terms, cell_mass = geometry(ar, ad, grid)
    result = placement_metrics(decode(cell_mass), graph)
    result.update({f"final_{name}": float(value) for name, value in terms.items()})
    result["candidate_recall"] = graph.candidate_recall
    result["conditional_r1"] = graph.conditional_r1
    result["sinkhorn_row_error"] = float(
        max((ar.sum(1)[:-1] - 1).abs().max(), (ad.sum(1)[:-1] - 1).abs().max())
    )
    return result, trace

def smoke(args: argparse.Namespace, device: torch.device) -> None:
    graph = make_graph(
        grid=4, k=min(args.k, 15), recall=1.0, mode="exact", seed=args.seed, floor=args.floor
    )
    ar = exact_augmented(graph, "right", device)
    ad = exact_augmented(graph, "down", device)
    terms, cell_mass = geometry(ar, ad, 4)
    exact_values = {name: float(value) for name, value in terms.items()}
    if max(exact_values.values()) > 1.0e-7:
        raise AssertionError(f"exact CNSD algebra failed: {exact_values}")
    metrics = placement_metrics(decode(cell_mass), graph)
    if metrics["placement"] != 1.0 or metrics["neighbour"] != 1.0:
        raise AssertionError(f"exact derived-P decode failed: {metrics}")
    wr = torch.as_tensor(graph.weight_right, device=device)
    wd = torch.as_tensor(graph.weight_down, device=device)
    zr = nn.Parameter(_initial_logits(wr, 4, args.boundary_logit))
    zd = nn.Parameter(_initial_logits(wd, 4, args.boundary_logit))
    sr = augmented_sinkhorn(zr, grid=4, temperature=0.7, rounds=8)
    sd = augmented_sinkhorn(zd, grid=4, temperature=0.7, rounds=8)
    soft_terms, _ = geometry(sr, sd, 4)
    objective = sum(soft_terms.values()) - 0.01 * (
        (sr[:16, :16] * wr).sum() + (sd[:16, :16] * wd).sum()
    )
    objective.backward()
    gradients = [zr.grad, zd.grad]
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("CNSD differentiable smoke produced a non-finite gradient")
    gradient_norm = float(sum(gradient.square().sum() for gradient in gradients).sqrt())
    if gradient_norm <= 1.0e-8:
        raise AssertionError("CNSD differentiable smoke produced a zero gradient")
    print(
        json.dumps(
            {"status": "smoke_pass", "grid": 4, "exact_terms": exact_values,
             "exact_metrics": metrics, "gradient_norm": gradient_norm},
            indent=2,
        ),
        flush=True,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="grid-4 exact algebra/gradient check")
    mode.add_argument("--grid24", action="store_true", help="explicitly select the default grid-24 gate")
    add = parser.add_argument
    add("--mode", choices=("exact", "noisy"), default="noisy")
    add("--images", type=int, default=4); add("--k", type=int, default=32)
    add("--recall", type=float, default=0.67); add("--steps", type=int, default=120)
    add("--sinkhorn-rounds", type=int, default=16); add("--lr", type=float, default=0.25)
    add("--temperature-start", type=float, default=1.0)
    add("--temperature-end", type=float, default=0.08)
    add("--lambda-commutation", type=float, default=2.0)
    add("--lambda-chain", type=float, default=2.0)
    add("--lambda-occupancy", type=float, default=2.0); add("--entropy", type=float, default=0.02)
    add("--grad-clip", type=float, default=10.0); add("--boundary-logit", type=float, default=-1.0)
    add("--floor", type=float, default=-8.0); add("--timeout-seconds", type=float, default=480.0)
    add("--min-neighbour", type=float, default=0.50)
    add("--min-image-neighbour", type=float, default=0.40)
    add("--min-placement", type=float, default=0.25)
    add("--min-completion-fraction", type=float, default=0.15)
    add("--seed", type=int, default=17411); add("--device", default=None)
    add("--report", default="E:/pazzle_work/gates/cnsd_gate.json")
    args = parser.parse_args()
    if args.images < 1 or args.k < 4 or args.steps < 1 or args.sinkhorn_rounds < 1:
        parser.error("--images, --steps, --sinkhorn-rounds must be positive and --k must be >=4")
    if not 0.0 <= args.recall <= 1.0:
        parser.error("--recall must lie in [0,1]")
    if args.lr <= 0 or args.temperature_end <= 0 or args.temperature_start < args.temperature_end:
        parser.error("require lr>0 and 0<temperature-end<=temperature-start")
    if args.timeout_seconds <= 0 or args.grad_clip <= 0:
        parser.error("timeout and grad clip must be positive")
    return args

def _write_report(path: str, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, target)

def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    if args.smoke:
        smoke(args, device)
        return
    started = time.monotonic()
    deadline = started + args.timeout_seconds
    rows: list[dict[str, float]] = []
    traces: list[list[dict[str, float]]] = []
    status = "complete"
    error = ""
    try:
        for index in range(args.images):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"CNSD exceeded --timeout-seconds={args.timeout_seconds}")
            graph = make_graph(
                grid=24, k=args.k, recall=args.recall, mode=args.mode,
                seed=args.seed + index, floor=args.floor,
            )
            result, trace = solve(graph, args, device, deadline)
            rows.append(result)
            traces.append(trace)
            print(f"procedural image {index + 1}/{args.images}: {result}", flush=True)
    except TimeoutError as exc:
        status, error = "timeout", str(exc)
    aggregate: dict[str, float] = {}
    if rows:
        for key in rows[0]:
            aggregate[f"mean_{key}"] = float(np.mean([row[key] for row in rows]))
        aggregate["min_neighbour"] = float(min(row["neighbour"] for row in rows))
    recall_sane = bool(
        args.mode == "exact"
        or (rows and all(0.62 <= row["candidate_recall"] <= 0.72 for row in rows))
    )
    rank_sane = bool(
        args.mode == "exact"
        or (rows and all(0.22 <= row["conditional_r1"] <= 0.32 for row in rows))
    )
    completion_pass = bool(
        args.mode == "exact"
        or aggregate.get("mean_completion_fraction_of_correct", 0.0)
        >= args.min_completion_fraction
    )
    gate_pass = bool(
        status == "complete"
        and len(rows) == args.images
        and recall_sane
        and rank_sane
        and aggregate.get("mean_neighbour", 0.0) >= args.min_neighbour
        and aggregate.get("min_neighbour", 0.0) >= args.min_image_neighbour
        and aggregate.get("mean_placement", 0.0) >= args.min_placement
        and completion_pass
    )
    report: dict[str, Any] = {
        "format": "cnsd_procedural_gate_v1",
        "status": status,
        "gate_pass": gate_pass,
        "error": error,
        "runtime_seconds": time.monotonic() - started,
        "config": vars(args),
        "sanity": {"candidate_recall": recall_sane, "conditional_r1": rank_sane},
        "aggregate": aggregate,
        "images": rows,
        "traces": traces,
        "contract": {"mean_neighbour": args.min_neighbour,
                     "min_image_neighbour": args.min_image_neighbour,
                     "mean_placement": args.min_placement,
                     "completion_fraction_of_correct": args.min_completion_fraction},
    }
    _write_report(args.report, report)
    print(json.dumps({"report": args.report, "status": status, "gate_pass": gate_pass,
                      "aggregate": aggregate}, indent=2), flush=True)
    if status == "timeout":
        raise SystemExit(2)
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
