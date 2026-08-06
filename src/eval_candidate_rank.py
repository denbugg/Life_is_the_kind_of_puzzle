"""Full-graph reciprocal gate and assembly for the listwise seam ranker.

``train_candidate_rank.py`` measures ranking quality on *sampled* rows.  This
evaluator answers the strategy's actual gate question: if the trained
:class:`~candidate_rank.CandidateSeamRanker` scores **every** anchor's full
frozen candidate list in all four directions, how precise and how large is the
resulting mutual (reciprocal-argmax) edge graph, and what placement does robust
pose synchronization + Hungarian rounding recover from it?

Per image the ranker scores ``576 anchors x 4 directions x ~81 candidates``
oriented seams -- still a small fraction of the dense 576x575x4 universe, and
exactly the distribution it was trained on.  The pipeline is:

1. frozen affinity union proposes candidates (identical builder to training);
2. the ranker produces one score per (anchor, direction, candidate);
3. each row keeps its argmax with a softmax confidence;
4. an edge survives only if both endpoints choose each other with inverse
   directions (reciprocal argmax);
5. RSCM greedy enforces per-tile U/D/L/R slot capacity and pair uniqueness;
6. surviving edges become ``x_j - x_i = offset`` constraints for IRLS pose
   synchronization; the largest component is Hungarian-rounded to the grid.

Strategy gates (see ``DIRECT_POSE_GRAPH_STRATEGY.md``): reciprocal exact-edge
precision ``>= 0.75`` at coverage well above the PairwiseNet baseline
``~0.07``; downstream neighbour accuracy ``>= 0.25`` keeps the branch alive.

Examples
--------

    python src/eval_candidate_rank.py --smoke
    python src/eval_candidate_rank.py --ranker-ckpt artifacts/candidate_rank/rank_v1_best.pt --n 4
"""
from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor, nn

from candidate_rank import (
    DIRECTION_NAMES,
    NUM_DIRECTIONS,
    CandidateSeamRanker,
    neighbor_targets,
)
from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from direct_pose import DIRECT_OFFSETS
from eval_offset_pose import (
    PoseConstraints,
    synchronization_metrics,
    synchronize_coordinates,
)
from eval_rscm_gate import (
    INVERSE_DIRECTION,
    TRUE_PHYSICAL_EDGES,
    PhysicalMetrics,
    PhysicalRelation,
    physical_metrics,
    rscm_greedy,
)
from imgio import train_val_split
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def load_ranker(path: str, device: torch.device) -> tuple[CandidateSeamRanker, Mapping[str, object]]:
    """Load a frozen CandidateSeamRanker checkpoint with its exact architecture."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ranker checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload or "model_kwargs" not in payload:
        raise RuntimeError(f"{path} is not a train_candidate_rank checkpoint")
    model = CandidateSeamRanker(**dict(payload["model_kwargs"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


@torch.inference_mode()
def score_full_graph(
    model: CandidateSeamRanker,
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    *,
    pair_batch: int,
    device: torch.device,
) -> Tensor:
    """Score every valid (anchor, direction, candidate) seam; -inf elsewhere.

    Returns a ``(NUM_DIRECTIONS, NFRAG, K)`` float32 tensor.  Unlike training,
    no row is skipped for lacking a true target: inference must rank exactly
    what the frozen graph proposes.
    """
    if tiles.ndim != 4 or tuple(tiles.shape) != (NFRAG, 3, FS, FS):
        raise ValueError(f"tiles must have shape ({NFRAG},3,{FS},{FS}), got {tuple(tiles.shape)}")
    if candidates.ndim != 2 or candidates.shape[0] != NFRAG:
        raise ValueError(f"candidates must have shape ({NFRAG},K), got {tuple(candidates.shape)}")
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask aligned with candidates")
    if pair_batch < 1:
        raise ValueError("pair_batch must be positive")

    width = int(candidates.shape[1])
    flat_valid = valid.reshape(-1)
    anchor_flat = (
        torch.arange(NFRAG, device=candidates.device)
        .unsqueeze(1)
        .expand(-1, width)
        .reshape(-1)[flat_valid]
    )
    target_flat = candidates.reshape(-1)[flat_valid]
    scores = torch.full(
        (NUM_DIRECTIONS, NFRAG * width), -torch.inf, dtype=torch.float32, device=device
    )
    direction_ids = torch.empty(pair_batch, dtype=torch.long, device=device)
    for direction in range(NUM_DIRECTIONS):
        direction_ids.fill_(direction)
        chunks: list[Tensor] = []
        for start in range(0, int(anchor_flat.numel()), pair_batch):
            stop = min(start + pair_batch, int(anchor_flat.numel()))
            source = tiles[anchor_flat[start:stop]]
            target = tiles[target_flat[start:stop]]
            # Constant chunk shapes keep cudnn.benchmark's autotuning valid
            # across chunks and images; pad rows are sliced off unscored.
            count = stop - start
            if count < pair_batch:
                pad = pair_batch - count
                source = torch.cat((source, source[-1:].expand(pad, -1, -1, -1)), dim=0)
                target = torch.cat((target, target[-1:].expand(pad, -1, -1, -1)), dim=0)
            with _autocast(device):
                chunks.append(model(source, target, direction_ids).float()[:count])
        scores[direction, flat_valid] = torch.cat(chunks, dim=0)
    return scores.reshape(NUM_DIRECTIONS, NFRAG, width)


def mutual_argmax_relations(candidates: Tensor, scores: Tensor) -> list[PhysicalRelation]:
    """Keep edges where both endpoints pick each other with inverse directions.

    Confidence is each row's softmax probability of its argmax; a physical
    relation's weight is the weaker of its two directed confidences, matching
    the strategy's "reciprocal filtering" definition.  The anchor is always the
    smaller tile id, so each mutual pair is emitted exactly once per direction
    hypothesis.
    """
    if candidates.ndim != 2 or candidates.shape[0] != NFRAG:
        raise ValueError(f"candidates must have shape ({NFRAG},K), got {tuple(candidates.shape)}")
    if scores.shape != (NUM_DIRECTIONS, *candidates.shape):
        raise ValueError(
            f"scores must have shape ({NUM_DIRECTIONS},{NFRAG},K), got {tuple(scores.shape)}"
        )
    finite = torch.isfinite(scores)
    row_ok = finite.any(dim=-1)
    safe = torch.where(finite, scores, torch.full_like(scores, -torch.inf))
    probabilities = torch.softmax(safe, dim=-1).masked_fill(~finite, 0.0)
    best_slot = safe.argmax(dim=-1)
    best_probability = probabilities.gather(-1, best_slot.unsqueeze(-1)).squeeze(-1)
    best_target = (
        candidates.unsqueeze(0).expand(NUM_DIRECTIONS, -1, -1)
        .gather(-1, best_slot.unsqueeze(-1))
        .squeeze(-1)
    )

    slot_cpu = best_slot.cpu().numpy()
    target_cpu = best_target.cpu().numpy()
    probability_cpu = best_probability.float().cpu().numpy()
    ok_cpu = row_ok.cpu().numpy()
    relations: list[PhysicalRelation] = []
    for direction in range(NUM_DIRECTIONS):
        inverse = INVERSE_DIRECTION[direction]
        for anchor in range(NFRAG):
            if not ok_cpu[direction, anchor]:
                continue
            target = int(target_cpu[direction, anchor])
            if target <= anchor:
                # The (target, anchor, inverse) traversal emits this pair with
                # the canonical smaller-id anchor; self-pairs cannot occur in a
                # valid affinity graph.
                continue
            if not ok_cpu[inverse, target] or int(target_cpu[inverse, target]) != anchor:
                continue
            anchor_probability = float(probability_cpu[direction, anchor])
            target_probability = float(probability_cpu[inverse, target])
            relations.append(
                PhysicalRelation(
                    anchor=anchor,
                    target=target,
                    direction=direction,
                    anchor_rank=int(slot_cpu[direction, anchor]),
                    target_rank=int(slot_cpu[inverse, target]),
                    anchor_z=anchor_probability,
                    target_z=target_probability,
                    weight=min(anchor_probability, target_probability),
                )
            )
    return relations


def constraints_from_relations(relations: Sequence[PhysicalRelation]) -> PoseConstraints:
    """Turn accepted physical relations into ``x_target - x_anchor`` constraints."""
    if not relations:
        return PoseConstraints(
            source=np.empty(0, dtype=np.int64),
            target=np.empty(0, dtype=np.int64),
            delta=np.empty((0, 2), dtype=np.float64),
            weight=np.empty(0, dtype=np.float64),
        )
    offsets = np.asarray(DIRECT_OFFSETS, dtype=np.float64)
    return PoseConstraints(
        source=np.asarray([relation.anchor for relation in relations], dtype=np.int64),
        target=np.asarray([relation.target for relation in relations], dtype=np.int64),
        delta=offsets[[relation.direction for relation in relations]],
        weight=np.asarray([max(relation.weight, 1.0e-3) for relation in relations], dtype=np.float64),
    )


def _print_relation_metrics(label: str, metrics: PhysicalMetrics) -> None:
    print(
        f"  [{label}] selected={metrics.selected} ({metrics.edges_per_tile:.4f}/tile) "
        f"exact p={metrics.precision:.4f} true-edge coverage={metrics.coverage:.4f} "
        f"H/V p={metrics.horizontal_precision:.4f}/{metrics.vertical_precision:.4f} "
        f"largest correct component={metrics.correct_largest_nodes} nodes",
        flush=True,
    )


def _nan_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _perfect_inputs(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Data-free perfect candidate graph + oracle scores for the smoke test."""
    anchors = torch.arange(NFRAG, device=device)
    rows = torch.div(anchors, GRID, rounding_mode="floor")
    cols = torch.remainder(anchors, GRID)
    exists = torch.stack(
        (rows.gt(0), rows.lt(GRID - 1), cols.gt(0), cols.lt(GRID - 1)), dim=-1
    )
    deltas = torch.tensor((-GRID, GRID, -1, 1), device=device)
    candidates = torch.where(exists, anchors.unsqueeze(-1) + deltas, anchors.unsqueeze(-1))
    scores = torch.full((NUM_DIRECTIONS, NFRAG, NUM_DIRECTIONS), -8.0, device=device)
    for direction in range(NUM_DIRECTIONS):
        scores[direction, :, direction] = 8.0
    invalid = ~exists.unsqueeze(0).expand(NUM_DIRECTIONS, -1, -1).reshape(
        NUM_DIRECTIONS, NFRAG, NUM_DIRECTIONS
    )
    scores = scores.masked_fill(invalid, -torch.inf)
    return candidates, scores, anchors


def smoke(device: torch.device) -> dict[str, float]:
    """Exercise mutual-argmax extraction, RSCM, sync, and Hungarian end to end."""
    candidates, scores, perm = _perfect_inputs(device)
    relations = mutual_argmax_relations(candidates, scores)
    metrics = physical_metrics(relations, perm)
    if metrics.selected != TRUE_PHYSICAL_EDGES or metrics.precision < 0.999:
        raise AssertionError(f"perfect mutual-argmax graph failed: {metrics}")
    selected = rscm_greedy(relations)
    constraints = constraints_from_relations(selected)
    sync = synchronize_coordinates(constraints, iterations=8, huber=0.5)
    sync_metrics = synchronization_metrics(sync, constraints, perm)
    if sync_metrics["hungarian_whole_placement"] < 0.999:
        raise AssertionError(f"perfect graph did not place exactly: {sync_metrics}")

    # Reciprocity must reject a one-sided argmax.  Repoint an interior tile's
    # RIGHT row at its UP neighbour: that neighbour's LEFT row still contains a
    # true +8 target, so the wrong claim cannot become mutual.  (A border tile
    # is deliberately avoided -- its all-tied reverse row would echo any claim.)
    interior = GRID + 1
    corrupted = scores.clone()
    corrupted[3, interior, :] = -torch.inf
    corrupted[3, interior, 0] = 9.0
    broken = mutual_argmax_relations(candidates, corrupted)
    if len(broken) != TRUE_PHYSICAL_EDGES - 1:
        raise AssertionError(
            f"reciprocal filter kept {len(broken)} edges, expected {TRUE_PHYSICAL_EDGES - 1}"
        )
    return {
        "relations": float(metrics.selected),
        "precision": metrics.precision,
        "coverage": metrics.coverage,
        "whole_placement": float(sync_metrics["hungarian_whole_placement"]),
        "whole_neighbour": float(sync_metrics["hungarian_whole_neighbour"]),
    }


def _parse_args() -> argparse.Namespace:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranker-ckpt",
        "--ranker_ckpt",
        dest="ranker_ckpt",
        default=os.path.join(workspace, "artifacts", "candidate_rank", "rank_v1_best.pt"),
        help="trained CandidateSeamRanker checkpoint",
    )
    parser.add_argument(
        "--affinity-ckpt",
        "--affinity_ckpt",
        dest="affinity_ckpt",
        default="",
        help="primary affinity checkpoint; default reuses the ranker's recorded graph",
    )
    parser.add_argument(
        "--affinity-ckpt2",
        "--affinity_ckpt2",
        dest="affinity_ckpt2",
        default="",
        help="secondary affinity checkpoint; default reuses the ranker's recorded graph",
    )
    parser.add_argument("--n", type=int, default=4, help="fresh exact synthetic held-out images")
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=0,
                        help="per-encoder affinity top-K; 0 reuses the training value")
    parser.add_argument(
        "--thresholds",
        default="0.0,0.1,0.2,0.3,0.5,0.7",
        help="comma-separated min reciprocal confidence sweep",
    )
    parser.add_argument(
        "--sync-threshold",
        "--sync_threshold",
        dest="sync_threshold",
        type=float,
        default=0.2,
        help="reciprocal confidence floor for pose-graph constraints",
    )
    parser.add_argument("--sync-iterations", "--sync_iterations", dest="sync_iterations", type=int, default=25)
    parser.add_argument("--sync-huber", "--sync_huber", dest="sync_huber", type=float, default=0.5)
    parser.add_argument("--pair-batch", "--pair_batch", dest="pair_batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=SEED + 7331, help="fresh synthetic corruption seed")
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--smoke", action="store_true", help="run the data-free contract test and exit")
    args = parser.parse_args()
    try:
        args.thresholds = tuple(
            sorted({float(item) for item in str(args.thresholds).split(",") if item.strip()})
        )
    except ValueError:
        parser.error(f"invalid --thresholds value {args.thresholds!r}")
    if not args.thresholds:
        parser.error("at least one threshold is required")
    if any(not math.isfinite(value) or value < 0.0 for value in args.thresholds):
        parser.error("thresholds must be finite and non-negative")
    if args.n < 1 or args.pair_batch < 1 or args.sync_iterations < 1:
        parser.error("--n, --pair-batch, and --sync-iterations must be positive")
    if args.top_k < 0 or args.top_k >= NFRAG:
        parser.error(f"--top-k must lie in [0,{NFRAG - 1}]")
    if not 0.0 <= args.sync_threshold <= 1.0 or args.sync_huber <= 0.0:
        parser.error("--sync-threshold must lie in [0,1] and --sync-huber must be positive")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.smoke:
        print(f"[candidate-rank gate smoke] device={device} {smoke(device)}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    model, payload = load_ranker(args.ranker_ckpt, device)
    recorded = payload.get("candidate_graph", {})
    encoders = list(recorded.get("encoders", ())) if isinstance(recorded, Mapping) else []
    training_args = payload.get("args", {}) if isinstance(payload.get("args"), Mapping) else {}
    affinity_path = args.affinity_ckpt or (str(encoders[0]["path"]) if encoders else "")
    affinity_path2 = args.affinity_ckpt2 or (
        str(encoders[1]["path"]) if len(encoders) > 1 else ""
    )
    if not affinity_path:
        raise RuntimeError("no affinity checkpoint recorded in the ranker; pass --affinity-ckpt")
    top_k = int(args.top_k or training_args.get("candidate_k", 64))

    affinity, _, _ = load_frozen_affinity(affinity_path, device)
    affinity_secondary: nn.Module | None = None
    if affinity_path2:
        affinity_secondary, _, _ = load_frozen_affinity(affinity_path2, device)

    _, validation_names = train_val_split()
    if args.n > len(validation_names):
        raise ValueError(f"--n exceeds the held-out pool ({len(validation_names)})")
    dataset = CanvasDataset(validation_names[: args.n], real_prob=0.0, seed=args.seed)

    print(
        f"device={device} ranker={os.path.abspath(args.ranker_ckpt)} step={payload.get('step')} "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    print(
        f"frozen graph: top{top_k}/encoder encoders={1 + int(affinity_secondary is not None)} "
        f"({os.path.basename(affinity_path)}"
        + (f" + {os.path.basename(affinity_path2)})" if affinity_path2 else ")"),
        flush=True,
    )
    print(
        f"gate targets: reciprocal exact precision >= 0.75 at coverage >> 0.07; "
        f"neighbour accuracy >= 0.25 keeps the branch alive",
        flush=True,
    )

    sweep_selected: defaultdict[float, float] = defaultdict(float)
    sweep_correct: defaultdict[float, float] = defaultdict(float)
    rscm_selected: defaultdict[float, float] = defaultdict(float)
    rscm_correct: defaultdict[float, float] = defaultdict(float)
    sync_rows: defaultdict[str, list[float]] = defaultdict(list)
    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("gate evaluation requires exact synthetic samples")
        tiles = sample["tiles"].to(device, non_blocking=device.type == "cuda")
        perm = sample["perm"].to(device, non_blocking=device.type == "cuda").long()
        candidates_batched, valid_batched = mine_affinity_candidates(
            affinity,
            tiles.unsqueeze(0),
            candidate_k=top_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        candidates, valid = candidates_batched[0], valid_batched[0]
        scores = score_full_graph(
            model, tiles, candidates, valid, pair_batch=args.pair_batch, device=device
        )
        relations = mutual_argmax_relations(candidates, scores)

        # A candidate graph can only propose what it contains; report its own
        # ceiling next to the model so a recall problem is never misread as a
        # ranking problem.
        targets, exists = neighbor_targets(perm.unsqueeze(0))
        proposed = valid.unsqueeze(-1) & candidates.unsqueeze(-1).eq(
            targets[0].clamp_min(0).unsqueeze(1)
        )
        graph_recall = float((proposed.any(dim=1) & exists[0]).sum() / exists[0].sum())
        print(
            f"\nimage {index + 1}/{args.n}: mutual relations={len(relations)} "
            f"graph direct recall={graph_recall:.4f}",
            flush=True,
        )
        for threshold in args.thresholds:
            raw = [relation for relation in relations if relation.weight >= threshold]
            raw_metrics = physical_metrics(raw, perm)
            capped_metrics = physical_metrics(rscm_greedy(raw), perm)
            sweep_selected[threshold] += raw_metrics.selected
            sweep_correct[threshold] += raw_metrics.correct
            rscm_selected[threshold] += capped_metrics.selected
            rscm_correct[threshold] += capped_metrics.correct
            _print_relation_metrics(f"raw conf>={threshold:.2f}", raw_metrics)
            _print_relation_metrics(f"RSCM conf>={threshold:.2f}", capped_metrics)

        kept = rscm_greedy(
            [relation for relation in relations if relation.weight >= args.sync_threshold]
        )
        constraints = constraints_from_relations(kept)
        sync = synchronize_coordinates(
            constraints, iterations=args.sync_iterations, huber=args.sync_huber
        )
        metrics = synchronization_metrics(sync, constraints, perm)
        for key, value in metrics.items():
            sync_rows[key].append(float(value))
        print(
            f"  [sync conf>={args.sync_threshold:.2f}] constraints={int(metrics['sync_constraints'])} "
            f"components={int(metrics['sync_components'])} "
            f"largest={metrics['sync_largest_component_fraction']:.4f} "
            f"R2={metrics['sync_affine_coordinate_r2']:.4f} "
            f"component_place={metrics['hungarian_component_placement']:.4f} "
            f"whole_place={metrics['hungarian_whole_placement']:.4f} "
            f"whole_neighbour={metrics['hungarian_whole_neighbour']:.4f}",
            flush=True,
        )

    total_edges = float(TRUE_PHYSICAL_EDGES * args.n)
    print(f"\n=== pooled gate over {args.n} fresh held-out images ===", flush=True)
    for threshold in args.thresholds:
        raw_precision = (
            sweep_correct[threshold] / sweep_selected[threshold]
            if sweep_selected[threshold]
            else 0.0
        )
        capped_precision = (
            rscm_correct[threshold] / rscm_selected[threshold]
            if rscm_selected[threshold]
            else 0.0
        )
        print(
            f"conf>={threshold:.2f}: raw p={raw_precision:.4f} "
            f"cov={sweep_correct[threshold] / total_edges:.4f} "
            f"({sweep_selected[threshold] / args.n:.0f} edges/img) | "
            f"RSCM p={capped_precision:.4f} cov={rscm_correct[threshold] / total_edges:.4f} "
            f"({rscm_selected[threshold] / args.n:.0f} edges/img)",
            flush=True,
        )
    print(
        f"sync mean: R2={_nan_mean(sync_rows['sync_affine_coordinate_r2']):.4f} "
        f"largest={_nan_mean(sync_rows['sync_largest_component_fraction']):.4f} "
        f"component_place={_nan_mean(sync_rows['hungarian_component_placement']):.4f} "
        f"whole_place={_nan_mean(sync_rows['hungarian_whole_placement']):.4f} "
        f"whole_neighbour={_nan_mean(sync_rows['hungarian_whole_neighbour']):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
