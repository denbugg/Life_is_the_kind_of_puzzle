"""Oracle coverage gate for growing high-confidence seam seed components.

The listwise seam ranker is not dense enough to assemble a whole puzzle, but
``rank_v2w64`` leaves a small set of very reliable reciprocal seams.  This
diagnostic asks a narrower operational question before building a contextual
triple model:

* select high-confidence reciprocal/RSCM seed edges *without* clean labels;
* reveal the exact synthetic labels only afterwards;
* for each true selected ``A -> B`` seed, ask whether the true next tile
  ``C`` in the same direction is present in ``B``'s frozen affinity list;
* repeat one more oracle step to obtain a hard upper bound for directed
  seed-growth chains ``A -> B -> C -> D``.

The result is deliberately optimistic: it grants a future context model an
oracle ranker among candidates.  If the correct continuation is absent from
the affinity union, no triple/context scorer can recover it without expanding
the candidate graph.  Conversely, presence alone is not a model result.

Examples
--------

    python src/eval_seed_growth_coverage.py --smoke
    python src/eval_seed_growth_coverage.py --device cuda --n 1

By default the evaluator uses ``rank_v2w64_best.pt`` and falls back to
``rank_v1_best.pt`` only when the larger checkpoint is unavailable.  The dual
affinity-union paths and top-K are recovered from the ranker checkpoint so the
test exactly matches the candidate graph on which the ranker was trained.
"""
from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from canvas_data import CanvasDataset
from config import GRID, NFRAG, SEED
from direct_pose import DIRECT_OFFSETS
from eval_candidate_rank import load_ranker, mutual_argmax_relations, score_full_graph
from eval_rscm_gate import PhysicalRelation, rscm_greedy
from imgio import train_val_split
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


@dataclass
class _UnionFind:
    """Minimal union-find for reporting seed-component sizes only."""

    parent: list[int]
    size: list[int]

    @classmethod
    def create(cls, count: int) -> "_UnionFind":
        return cls(parent=list(range(count)), size=[1] * count)

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _default_ranker_path(workspace: str) -> str:
    """Prefer the capacity-control checkpoint, with a useful old fallback."""
    directory = os.path.join(workspace, "artifacts", "candidate_rank")
    v2 = os.path.join(directory, "rank_v2w64_best.pt")
    v1 = os.path.join(directory, "rank_v1_best.pt")
    if os.path.isfile(v2):
        return v2
    return v1


def _resolve_affinity_paths(
    payload: Mapping[str, object],
    args: argparse.Namespace,
) -> tuple[str, str, int]:
    """Recover the exact dual frozen graph recorded by train_candidate_rank."""
    recorded = payload.get("candidate_graph", {})
    graph = recorded if isinstance(recorded, Mapping) else {}
    raw_encoders = graph.get("encoders", ())
    encoders = list(raw_encoders) if isinstance(raw_encoders, Sequence) else []
    train_args = payload.get("args", {})
    saved_args = train_args if isinstance(train_args, Mapping) else {}

    recorded_first = ""
    recorded_second = ""
    if encoders and isinstance(encoders[0], Mapping):
        recorded_first = str(encoders[0].get("path", ""))
    if len(encoders) > 1 and isinstance(encoders[1], Mapping):
        recorded_second = str(encoders[1].get("path", ""))

    primary = str(args.affinity_ckpt or recorded_first or saved_args.get("affinity_ckpt", ""))
    secondary = str(
        args.affinity_ckpt2 or recorded_second or saved_args.get("affinity_ckpt2", "")
    )
    if not primary or not secondary:
        raise RuntimeError(
            "this diagnostic requires the dual affinity union recorded by the ranker; "
            "pass both --affinity-ckpt and --affinity-ckpt2 if checkpoint metadata is absent"
        )
    try:
        saved_top_k = int(saved_args.get("candidate_k", 64))
    except (TypeError, ValueError):
        saved_top_k = 64
    top_k = int(args.top_k or saved_top_k)
    if not 1 <= top_k < NFRAG:
        raise ValueError(f"resolved top-K must lie in [1,{NFRAG - 1}], got {top_k}")
    return primary, secondary, top_k


def _true_direction(perm: np.ndarray, source: int, target: int) -> int | None:
    """Return the exact cardinal direction of target from source, if any."""
    source_row, source_col = divmod(int(perm[source]), GRID)
    target_row, target_col = divmod(int(perm[target]), GRID)
    delta = target_row - source_row, target_col - source_col
    try:
        return DIRECT_OFFSETS.index(delta)
    except ValueError:
        return None


def _inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """Map clean-cell id back to shuffled input-tile id, with strict checks."""
    if perm.shape != (NFRAG,):
        raise ValueError(f"perm must have shape ({NFRAG},), got {perm.shape}")
    if np.any(perm < 0) or np.any(perm >= NFRAG) or np.unique(perm).size != NFRAG:
        raise ValueError("perm is not a valid input-tile -> clean-cell permutation")
    inverse = np.empty(NFRAG, dtype=np.int64)
    inverse[perm] = np.arange(NFRAG, dtype=np.int64)
    return inverse


def _next_true_tile(
    perm: np.ndarray,
    inverse: np.ndarray,
    source: int,
    direction: int,
) -> int | None:
    """Return the true next input tile in a direction, or None at the border."""
    row, col = divmod(int(perm[source]), GRID)
    delta_row, delta_col = DIRECT_OFFSETS[direction]
    next_row = row + delta_row
    next_col = col + delta_col
    if not (0 <= next_row < GRID and 0 <= next_col < GRID):
        return None
    return int(inverse[next_row * GRID + next_col])


def _candidate_sets(candidates: Tensor, valid: Tensor) -> list[set[int]]:
    """Turn the de-duplicated sparse union into cheap CPU membership lookups."""
    if candidates.ndim != 2 or candidates.shape[0] != NFRAG:
        raise ValueError(f"candidates must have shape ({NFRAG},K), got {tuple(candidates.shape)}")
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a bool mask aligned with candidates")
    candidate_cpu = candidates.detach().to(device="cpu", dtype=torch.long).numpy()
    valid_cpu = valid.detach().to(device="cpu", dtype=torch.bool).numpy()
    return [
        {int(candidate) for candidate, keep in zip(row, mask) if bool(keep)}
        for row, mask in zip(candidate_cpu, valid_cpu)
    ]


def _component_sizes(relations: Sequence[PhysicalRelation]) -> list[int]:
    """Return non-singleton component sizes, without consulting labels."""
    union_find = _UnionFind.create(NFRAG)
    for relation in relations:
        union_find.union(relation.anchor, relation.target)
    roots_with_edges = {union_find.find(relation.anchor) for relation in relations}
    return sorted(union_find.size[root] for root in roots_with_edges)


def _format_component_distribution(relations: Sequence[PhysicalRelation]) -> str:
    """Compact, stable component distribution for human gate reading."""
    sizes = _component_sizes(relations)
    if not sizes:
        return f"components=0 nodes=0/{NFRAG} largest=0 bins[2=0 3=0 4=0 5-7=0 8-15=0 16+=0]"
    histogram = Counter(
        "2"
        if size == 2
        else "3"
        if size == 3
        else "4"
        if size == 4
        else "5-7"
        if size <= 7
        else "8-15"
        if size <= 15
        else "16+"
        for size in sizes
    )
    return (
        f"components={len(sizes)} nodes={sum(sizes)}/{NFRAG} largest={max(sizes)} "
        f"mean={float(np.mean(sizes)):.2f} p50={float(np.median(sizes)):.1f} "
        f"bins[2={histogram['2']} 3={histogram['3']} 4={histogram['4']} "
        f"5-7={histogram['5-7']} 8-15={histogram['8-15']} 16+={histogram['16+']}]"
    )


def select_seed_relations(
    candidates: Tensor,
    scores: Tensor,
    *,
    confidence: float,
    use_rscm: bool,
) -> list[PhysicalRelation]:
    """Select seeds from model scores alone; no clean labels enter this path."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must lie in [0,1]")
    reciprocal = mutual_argmax_relations(candidates, scores)
    high_confidence = [relation for relation in reciprocal if relation.weight >= confidence]
    return rscm_greedy(high_confidence) if use_rscm else high_confidence


def analyze_oracle_growth(
    seeds: Sequence[PhysicalRelation],
    candidates: Tensor,
    valid: Tensor,
    perm: Tensor,
) -> dict[str, int | float | list[PhysicalRelation]]:
    """Reveal labels *after* seed selection and measure oracle chain ceilings.

    ``one_step_*`` follows each correct selected physical relation in its stored
    direction ``A -> B``.  It checks only whether the actual continuation C is
    in B's affinity list.  ``two_step_*`` requires both B->C and C->D list
    membership, so its reachable-tile count is a directional two-step upper
    bound for a context model grown from these seeds.
    """
    perm_cpu = perm.detach().to(device="cpu", dtype=torch.long).numpy()
    inverse = _inverse_permutation(perm_cpu)
    candidate_sets = _candidate_sets(candidates, valid)

    correct = [
        relation
        for relation in seeds
        if _true_direction(perm_cpu, relation.anchor, relation.target) == relation.direction
    ]
    seed_nodes = {node for relation in correct for node in (relation.anchor, relation.target)}

    continuable = 0
    one_step_chains = 0
    two_step_eligible = 0
    two_step_chains = 0
    one_step_nodes: set[int] = set()
    two_step_nodes: set[int] = set()

    for relation in correct:
        # A -> B is already known to be true.  C/D are retrieved from labels
        # only here, after model-only high-confidence selection above.
        continuation = _next_true_tile(perm_cpu, inverse, relation.target, relation.direction)
        if continuation is None:
            continue
        continuable += 1
        if continuation not in candidate_sets[relation.target]:
            continue
        one_step_chains += 1
        one_step_nodes.add(continuation)

        second = _next_true_tile(perm_cpu, inverse, continuation, relation.direction)
        if second is None:
            continue
        two_step_eligible += 1
        if second in candidate_sets[continuation]:
            two_step_chains += 1
            two_step_nodes.add(second)

    reachable_one = seed_nodes | one_step_nodes
    reachable_two = reachable_one | two_step_nodes
    selected_count = len(seeds)
    correct_count = len(correct)
    return {
        "selected": selected_count,
        "correct": correct_count,
        "precision": float(correct_count / selected_count) if selected_count else 0.0,
        "correct_seed_nodes": len(seed_nodes),
        "continuable_correct_seeds": continuable,
        "one_step_growable_chains": one_step_chains,
        "one_step_coverage": float(one_step_chains / continuable) if continuable else 0.0,
        "two_step_eligible_after_one": two_step_eligible,
        "two_step_growable_chains": two_step_chains,
        "two_step_coverage_given_one": (
            float(two_step_chains / two_step_eligible) if two_step_eligible else 0.0
        ),
        "two_step_coverage_from_continuable": (
            float(two_step_chains / continuable) if continuable else 0.0
        ),
        "one_step_new_tiles": len(one_step_nodes - seed_nodes),
        "two_step_new_tiles": len(two_step_nodes - reachable_one),
        "reachable_after_one": len(reachable_one),
        "reachable_after_two": len(reachable_two),
        "correct_relations": correct,
    }


def _direct_candidate_recall(candidates: Tensor, valid: Tensor, perm: Tensor) -> float:
    """Report the frozen graph ceiling after selection, as an interpretation aid."""
    perm_cpu = perm.detach().to(device="cpu", dtype=torch.long).numpy()
    inverse = _inverse_permutation(perm_cpu)
    candidate_sets = _candidate_sets(candidates, valid)
    available = 0
    total = 0
    for source in range(NFRAG):
        for direction in range(len(DIRECT_OFFSETS)):
            target = _next_true_tile(perm_cpu, inverse, source, direction)
            if target is None:
                continue
            total += 1
            available += int(target in candidate_sets[source])
    return float(available / total) if total else 0.0


def _print_image_summary(
    index: int,
    total: int,
    seeds: Sequence[PhysicalRelation],
    stats: Mapping[str, int | float | list[PhysicalRelation]],
    graph_recall: float,
) -> None:
    correct = stats["correct_relations"]
    if not isinstance(correct, list):  # Defensive runtime guard for the typed mapping.
        raise RuntimeError("internal growth report lost its correct relation list")
    print(f"\n=== image {index}/{total}: model-only seed selection, then oracle labels ===", flush=True)
    print(
        f"seeds={stats['selected']} correct={stats['correct']} p={float(stats['precision']):.4f}; "
        f"frozen direct candidate recall={graph_recall:.4f}",
        flush=True,
    )
    print(f"selected seed components: {_format_component_distribution(seeds)}", flush=True)
    print(f"correct-only components:  {_format_component_distribution(correct)}", flush=True)
    print(
        "oracle directed continuation A->B->C: "
        f"eligible={stats['continuable_correct_seeds']} "
        f"growable={stats['one_step_growable_chains']} "
        f"coverage={float(stats['one_step_coverage']):.4f}",
        flush=True,
    )
    print(
        "oracle two-step A->B->C->D: "
        f"eligible-after-C={stats['two_step_eligible_after_one']} "
        f"growable={stats['two_step_growable_chains']} "
        f"coverage|C={float(stats['two_step_coverage_given_one']):.4f} "
        f"coverage|initial={float(stats['two_step_coverage_from_continuable']):.4f}",
        flush=True,
    )
    print(
        "oracle reachable true tiles (unique, directed chains): "
        f"seed={stats['correct_seed_nodes']}/{NFRAG}; "
        f"after+1={stats['reachable_after_one']}/{NFRAG} "
        f"(+{stats['one_step_new_tiles']} new); "
        f"after+2={stats['reachable_after_two']}/{NFRAG} "
        f"(+{stats['two_step_new_tiles']} further)",
        flush=True,
    )


def smoke() -> dict[str, int | float]:
    """Data-free proof that labels are consumed only by the oracle analyser."""
    # Identity labels make input tile id equal clean cell.  0->1->2->3 is a
    # valid RIGHT chain.  The model-selection part is represented by two
    # preselected physical records, so no model/labels are mixed here.
    candidates = torch.zeros((NFRAG, 2), dtype=torch.long)
    valid = torch.zeros_like(candidates, dtype=torch.bool)
    candidates[1, 0] = 2
    candidates[2, 0] = 3
    valid[1, 0] = True
    valid[2, 0] = True
    seed = PhysicalRelation(0, 1, 3, 0, 0, 0.9, 0.9, 0.9)
    wrong = PhysicalRelation(10, 11, 0, 0, 0, 0.9, 0.9, 0.9)
    report = analyze_oracle_growth([seed, wrong], candidates, valid, torch.arange(NFRAG))
    if (
        report["correct"] != 1
        or report["one_step_growable_chains"] != 1
        or report["two_step_growable_chains"] != 1
        or report["reachable_after_two"] != 4
    ):
        raise AssertionError(f"seed-growth smoke failed: {report}")
    return {
        "selected": int(report["selected"]),
        "correct": int(report["correct"]),
        "one_step_chains": int(report["one_step_growable_chains"]),
        "two_step_chains": int(report["two_step_growable_chains"]),
        "reachable_after_two": int(report["reachable_after_two"]),
    }


def _parse_args() -> argparse.Namespace:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranker-ckpt",
        "--ranker_ckpt",
        dest="ranker_ckpt",
        default="",
        help="rank_v2w64 checkpoint; empty prefers v2 then falls back to rank_v1",
    )
    parser.add_argument(
        "--affinity-ckpt",
        "--affinity_ckpt",
        dest="affinity_ckpt",
        default="",
        help="primary affinity checkpoint; empty reuses ranker metadata",
    )
    parser.add_argument(
        "--affinity-ckpt2",
        "--affinity_ckpt2",
        dest="affinity_ckpt2",
        default="",
        help="secondary affinity checkpoint; empty reuses ranker metadata",
    )
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=0)
    parser.add_argument("--n", type=int, default=1, help="fresh exact synthetic held-out puzzles")
    parser.add_argument(
        "--confidence",
        "--conf",
        type=float,
        default=0.70,
        help="documented high-confidence reciprocal softmax floor",
    )
    parser.add_argument(
        "--no-rscm",
        action="store_true",
        help="diagnostic override: skip the documented slot-capacity seed filter",
    )
    parser.add_argument("--pair-batch", "--pair_batch", type=int, default=4096)
    parser.add_argument(
        "--seed", type=int, default=SEED + 7331,
        help="fresh synthetic distortion/shuffle seed used by the candidate-rank gate",
    )
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--smoke", action="store_true", help="run data-free chain checks and exit")
    args = parser.parse_args()
    if not args.ranker_ckpt:
        args.ranker_ckpt = _default_ranker_path(workspace)
    if args.n < 1:
        parser.error("--n must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must lie in [0,1]")
    if args.pair_batch < 1:
        parser.error("--pair-batch must be positive")
    if args.top_k < 0 or args.top_k >= NFRAG:
        parser.error(f"--top-k must lie in [0,{NFRAG - 1}]")
    return args


def main() -> None:
    args = _parse_args()
    if args.smoke:
        print(f"[seed-growth coverage smoke] {smoke()}", flush=True)
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    model, payload = load_ranker(args.ranker_ckpt, device)
    affinity_path, affinity_path2, top_k = _resolve_affinity_paths(payload, args)
    affinity, _, _ = load_frozen_affinity(affinity_path, device)
    affinity_secondary: nn.Module
    affinity_secondary, _, _ = load_frozen_affinity(affinity_path2, device)

    _, validation_names = train_val_split()
    if args.n > len(validation_names):
        raise ValueError(f"--n exceeds held-out pool ({len(validation_names)})")
    dataset = CanvasDataset(validation_names[: args.n], real_prob=0.0, seed=args.seed)

    print(
        f"device={device} ranker={os.path.abspath(args.ranker_ckpt)} step={payload.get('step')} "
        f"params={sum(parameter.numel() for parameter in model.parameters()):,}",
        flush=True,
    )
    print(
        f"seed rule: mutual argmax, conf>={args.confidence:.2f}, "
        f"RSCM={'on' if not args.no_rscm else 'off'}; "
        f"dual affinity union=top{top_k}+top{top_k}",
        flush=True,
    )
    print(
        f"affinity_1={os.path.abspath(affinity_path)}\n"
        f"affinity_2={os.path.abspath(affinity_path2)}",
        flush=True,
    )
    print(
        "Labels are intentionally absent from candidate mining and seed selection; "
        "they are revealed only for the reports below.",
        flush=True,
    )

    totals: defaultdict[str, float] = defaultdict(float)
    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("seed-growth coverage requires real_prob=0 exact synthetic samples")
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
        # This is the only path that selects a seed.  It has no access to perm.
        seeds = select_seed_relations(
            candidates, scores, confidence=args.confidence, use_rscm=not args.no_rscm
        )
        # All label-dependent work starts here.
        report = analyze_oracle_growth(seeds, candidates, valid, perm)
        graph_recall = _direct_candidate_recall(candidates, valid, perm)
        _print_image_summary(index + 1, args.n, seeds, report, graph_recall)
        for key, value in report.items():
            if key != "correct_relations" and isinstance(value, (int, float)):
                totals[key] += float(value)
        totals["graph_recall"] += graph_recall

    if args.n > 1:
        print(f"\n=== mean over {args.n} independent fresh held-out puzzles ===", flush=True)
        print(
            f"seeds={totals['selected'] / args.n:.1f}; correct={totals['correct'] / args.n:.1f}; "
            f"mean seed precision={totals['correct'] / totals['selected'] if totals['selected'] else 0.0:.4f}; "
            f"mean graph recall={totals['graph_recall'] / args.n:.4f}",
            flush=True,
        )
        print(
            f"growable A->B->C={totals['one_step_growable_chains'] / args.n:.1f}/"
            f"{totals['continuable_correct_seeds'] / args.n:.1f} "
            f"({totals['one_step_growable_chains'] / totals['continuable_correct_seeds'] if totals['continuable_correct_seeds'] else 0.0:.4f}); "
            f"A->B->C->D={totals['two_step_growable_chains'] / args.n:.1f}/"
            f"{totals['two_step_eligible_after_one'] / args.n:.1f} "
            f"({totals['two_step_growable_chains'] / totals['two_step_eligible_after_one'] if totals['two_step_eligible_after_one'] else 0.0:.4f})",
            flush=True,
        )
        print(
            f"reachable unique true tiles: seed={totals['correct_seed_nodes'] / args.n:.1f}; "
            f"after+1={totals['reachable_after_one'] / args.n:.1f}; "
            f"after+2={totals['reachable_after_two'] / args.n:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
