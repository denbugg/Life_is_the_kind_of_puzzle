"""Train a relational message-passing re-ranker on frozen candidate graphs.

Unlike the independent LambdaRank calibrator, this model updates every tile
from four distributions over its possible neighbours, then re-scores an edge
using the updated states of both endpoints.  It is residual around the frozen
candidate-ranker z-score, so epoch zero exactly reproduces the local ranking.
Whole images are kept disjoint for fit, selection, and external evaluation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import NFRAG, SEED, WORK_ROOT
from eval_candidate_calibrator import DELTAS, Graph, edge_features, load_graph, true_target
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import solve_buddies_from_scores


@dataclass
class GraphTensor:
    image: int
    candidates: Tensor       # (2304,K)
    features: Tensor         # (2304,K,F)
    base: Tensor             # (2304,K)
    target_slot: Tensor      # (2304,), -100 when unrankable
    node_features: Tensor    # (576,14)
    permutation: np.ndarray

    def to(self, device: torch.device) -> "GraphTensor":
        return GraphTensor(
            self.image,
            self.candidates.to(device),
            self.features.to(device),
            self.base.to(device),
            self.target_slot.to(device),
            self.node_features.to(device),
            self.permutation,
        )


def tensorize(graph: Graph, top_k: int) -> GraphTensor:
    candidate_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    base_rows: list[np.ndarray] = []
    targets: list[int] = []
    for anchor in range(NFRAG):
        for direction in range(4):
            candidates = np.flatnonzero(graph.valid[direction, anchor])
            order = np.argsort(-graph.raw[direction, anchor, candidates])
            candidates = candidates[order[:top_k]]
            if len(candidates) != top_k:
                raise ValueError(
                    f"image {graph.image} row {(anchor, direction)} has "
                    f"{len(candidates)} candidates, expected at least {top_k}"
                )
            values = edge_features(graph, anchor, direction, candidates)
            candidate_rows.append(candidates)
            feature_rows.append(values)
            # Column one is a complete-row standardized raw ranker score.
            base_rows.append(values[:, 1])
            target = true_target(graph, anchor, direction)
            match = np.flatnonzero(candidates == target)
            targets.append(int(match[0]) if len(match) else -100)
    stats = graph.stats.astype(np.float32)
    z = (stats - graph.scene_mean) / np.maximum(graph.scene_std, 1.0e-4)
    return GraphTensor(
        graph.image,
        torch.from_numpy(np.stack(candidate_rows)).long(),
        torch.from_numpy(np.stack(feature_rows)).float(),
        torch.from_numpy(np.stack(base_rows)).float(),
        torch.tensor(targets, dtype=torch.long),
        torch.from_numpy(np.concatenate((stats, z), axis=1)).float(),
        graph.permutation,
    )


class GraphMessageRefiner(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 48, rounds: int = 2, dropout: float = 0.05) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.rounds = int(rounds)
        self.edge = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.node = nn.Sequential(
            nn.Linear(14, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.message = nn.Linear(hidden, hidden, bias=False)
        self.context = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
        )
        self.update = nn.GRUCell(hidden, hidden)
        self.score = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, 1),
        )
        # Exact raw-ranker residual at initialization.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def _scores(self, edge: Tensor, node: Tensor, candidates: Tensor, base: Tensor) -> Tensor:
        rows, width = candidates.shape
        anchors = torch.arange(NFRAG, device=node.device).repeat_interleave(4)
        source = node[anchors].unsqueeze(1).expand(rows, width, self.hidden)
        target = node[candidates]
        delta = self.score(torch.cat((edge, source, target), dim=-1)).squeeze(-1)
        return base + delta

    def forward(
        self, candidates: Tensor, features: Tensor, base: Tensor, node_features: Tensor
    ) -> Tensor:
        if candidates.shape[:2] != features.shape[:2] or base.shape != candidates.shape:
            raise ValueError("candidate, feature and base row shapes disagree")
        edge = self.edge(features)
        node = self.node(node_features)
        for _ in range(self.rounds):
            logits = self._scores(edge, node, candidates, base)
            weight = F.softmax(logits, dim=1)
            values = self.message(node[candidates])
            aggregate = (weight.unsqueeze(-1) * values).sum(dim=1)
            context = self.context(aggregate.reshape(NFRAG, 4 * self.hidden))
            node = self.update(context, node)
        return self._scores(edge, node, candidates, base)


def _paths(cache_dir: Path, text: str) -> list[Path]:
    paths = []
    for value in text.split(","):
        image = int(value.strip())
        path = cache_dir / f"image_{image:04d}_k64.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def _dense_scores(graph: GraphTensor, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((4, NFRAG, NFRAG), dtype=np.float32)
    candidates = graph.candidates.numpy().reshape(NFRAG, 4, -1)
    logits = logits.reshape(NFRAG, 4, -1)
    for anchor in range(NFRAG):
        for direction in range(4):
            score = logits[anchor, direction]
            probability = np.exp(score - score.max())
            probability /= max(float(probability.sum()), 1.0e-8)
            values[direction, anchor, candidates[anchor, direction]] = probability
    right = 0.5 * (values[3] + values[2].T)
    down = 0.5 * (values[1] + values[0].T)
    np.fill_diagonal(right, 0.0)
    np.fill_diagonal(down, 0.0)
    return right, down


@torch.inference_mode()
def evaluate(
    model: GraphMessageRefiner,
    graphs: list[GraphTensor],
    device: torch.device,
    solver_budget: int,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for graph_cpu in graphs:
        graph = graph_cpu.to(device)
        refined = model(
            graph.candidates, graph.features, graph.base, graph.node_features
        ).float()
        mask = graph.target_slot >= 0
        target = graph.target_slot[mask]
        base_rank = 1 + (graph.base[mask] > graph.base[mask].gather(1, target[:, None])).sum(dim=1)
        refined_rank = 1 + (refined[mask] > refined[mask].gather(1, target[:, None])).sum(dim=1)
        # Every undirected physical seam appears in both directed rows.
        physical = 4 * 24 * 23
        truth = np.argsort(graph.permutation)
        base_right, base_down = _dense_scores(graph_cpu, graph_cpu.base.numpy())
        ref_right, ref_down = _dense_scores(graph_cpu, refined.cpu().numpy())
        base_place, _ = solve_buddies_from_scores(
            base_right, base_down, max_edges=solver_budget, repair_passes=0
        )
        ref_place, _ = solve_buddies_from_scores(
            ref_right, ref_down, max_edges=solver_budget, repair_passes=0
        )
        rows.append(
            {
                "coverage": float(mask.sum().item() / physical),
                "base_r1": float(base_rank.le(1).float().sum().item() / physical),
                "refined_r1": float(refined_rank.le(1).float().sum().item() / physical),
                "base_r5": float(base_rank.le(5).float().sum().item() / physical),
                "refined_r5": float(refined_rank.le(5).float().sum().item() / physical),
                "base_placement": placement_accuracy(base_place, truth)[0],
                "refined_placement": placement_accuracy(ref_place, truth)[0],
                "base_neighbour": neighbour_accuracy(base_place, truth)[0],
                "refined_neighbour": neighbour_accuracy(ref_place, truth)[0],
            }
        )
    model.train()
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _print(label: str, metrics: dict[str, float]) -> None:
    print(label + " " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--fit-images", default="10,11,12,13,14,15,16,17")
    parser.add_argument("--val-images", default="18,19,20,21")
    parser.add_argument("--external-images", default="50,51,52,53,54,55")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--solver-budget", type=int, default=384)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "graph_message_refiner.pt",
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path(WORK_ROOT) / "gates" / "graph_message_refiner_gate.json",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.top_k < 2 or args.epochs < 1 or args.rounds < 1:
        parser.error("--top-k >=2 and positive --epochs/--rounds are required")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def load_many(text: str) -> list[GraphTensor]:
        output = []
        for path in _paths(args.cache_dir, text):
            print(f"tensorize {path.name}", flush=True)
            output.append(tensorize(load_graph(path), args.top_k))
        return output

    fit = load_many(args.fit_images)
    validation = load_many(args.val_images)
    external = load_many(args.external_images)
    feature_dim = int(fit[0].features.shape[-1])
    all_fit_features = torch.cat([graph.features.reshape(-1, feature_dim) for graph in fit])
    mean = all_fit_features.mean(dim=0)
    scale = all_fit_features.std(dim=0, unbiased=False).clamp_min(1.0e-4)
    del all_fit_features
    for collection in (fit, validation, external):
        for graph in collection:
            graph.features = ((graph.features - mean) / scale).clamp(-8.0, 8.0)

    model = GraphMessageRefiner(feature_dim, args.hidden, args.rounds).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_score = -float("inf")
    best_metrics: dict[str, float] = {}
    order_rng = np.random.default_rng(args.seed)
    _print("VAL epoch=0", evaluate(model, validation, device, args.solver_budget))
    for epoch in range(1, args.epochs + 1):
        losses = []
        for index in order_rng.permutation(len(fit)):
            graph = fit[int(index)].to(device)
            logits = model(
                graph.candidates, graph.features, graph.base, graph.node_features
            )
            mask = graph.target_slot >= 0
            loss = F.cross_entropy(logits[mask], graph.target_slot[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        print(
            f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, validation, device, args.solver_budget)
            _print(f"VAL epoch={epoch}", metrics)
            score = metrics["refined_r1"] + 0.25 * metrics["refined_neighbour"]
            if score > best_score:
                best_score, best_metrics = score, metrics
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "feature_dim": feature_dim,
                        "hidden": args.hidden,
                        "rounds": args.rounds,
                        "top_k": args.top_k,
                        "mean": mean,
                        "scale": scale,
                        "epoch": epoch,
                        "validation": metrics,
                    },
                    args.checkpoint,
                )
                print(f"saved best score={score:.4f}", flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    external_metrics = evaluate(model, external, device, args.solver_budget)
    _print("EXTERNAL", external_metrics)
    delta = {
        "r1": external_metrics["refined_r1"] - external_metrics["base_r1"],
        "neighbour": external_metrics["refined_neighbour"] - external_metrics["base_neighbour"],
        "placement": external_metrics["refined_placement"] - external_metrics["base_placement"],
    }
    report = {
        "experiment": "relational_candidate_graph_message_passing",
        "checkpoint": str(args.checkpoint),
        "fit_images": args.fit_images,
        "validation_images": args.val_images,
        "external_images": args.external_images,
        "best_validation": best_metrics,
        "external": external_metrics,
        "delta": delta,
        "thresholds": {"r1": 0.02, "neighbour": 0.01},
        "passed": delta["r1"] >= 0.02 and delta["neighbour"] >= 0.01,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
