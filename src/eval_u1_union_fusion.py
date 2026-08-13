"""U2: score the U1 R2L∪R3 candidate union with frozen pair and pose models.

Candidate construction is label-blind.  Exact synthetic labels are consulted only
after the graph and no-label pair/pose scores are frozen, solely for held-out
metrics.  This is a ranking diagnostic, not an assignment or SSIM evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from canvas_data import CanvasDataset
from config import GRID, NFRAG
from direct_pose import NON_DIRECT_CLASS
from eval_pair_affinity_fusion import (
    _candidate_scores_from_orientations,
    load_direct_pose,
    load_pair_ensemble,
    score_direct_pose_bundle,
    score_pairwise_directions,
)
from eval_r2l_affinity_union import (
    DEFAULT_AFFINITY_A,
    DEFAULT_AFFINITY_B,
    DEFAULT_R2L,
    _autocast,
    _device,
    _load_r2,
    _union_candidates,
)
from imgio import train_val_split
from train_direct_pose import candidate_direct_labels
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates

DIRECT_EDGES_PER_BOARD = 4 * GRID * (GRID - 1)
DEFAULT_PAIR = r"E:\pazzle_work\ckpt\pair0_best.pt,E:\pazzle_work\ckpt\pair1_best.pt"
DEFAULT_POSE = r"E:\pazzle_work\pazzle_fixed_orientation_20260813\F1_direct_pose\checkpoints\orbit24_f1_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U2 pair/pose fusion on U1 candidate union.")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--affinity-k", type=int, default=64)
    parser.add_argument("--r2-topk", type=int, default=8)
    parser.add_argument("--r2-ckpt", default=DEFAULT_R2L)
    parser.add_argument("--affinity-ckpt", default=DEFAULT_AFFINITY_A)
    parser.add_argument("--affinity-ckpt2", default=DEFAULT_AFFINITY_B)
    parser.add_argument("--pair-ckpts", default=DEFAULT_PAIR)
    parser.add_argument("--direct-pose-ckpt", default=DEFAULT_POSE)
    parser.add_argument("--pair-batch", type=int, default=4096)
    parser.add_argument("--pose-pair-batch", type=int, default=4096)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--topks", default="1,2,4,8,16")
    parser.add_argument("--seed", type=int, default=240815)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _topk_mask(score: Tensor, valid: Tensor, topk: int) -> Tensor:
    if topk < 1:
        raise ValueError("topk must be positive")
    width = min(topk, score.shape[1])
    masked = torch.where(valid, score, torch.full_like(score, -torch.inf))
    index = masked.topk(width, dim=1).indices
    selected = torch.zeros_like(valid)
    selected.scatter_(1, index, True)
    return selected & valid


def _counts(selected: Tensor, labels: Tensor) -> dict[str, float]:
    direct = selected & labels.ne(NON_DIRECT_CLASS)
    direction = torch.arange(4, device=labels.device).view(1, 1, 4)
    # caller supplies prediction tensor through selected direction match separately.
    return {"selected": float(selected.sum()), "direct": float(direct.sum())}


def _topk_metrics(
    score: Tensor,
    direction: Tensor,
    valid: Tensor,
    labels: Tensor,
    images: int,
    topks: list[int],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for topk in topks:
        selected = _topk_mask(score, valid, topk)
        direct = selected & labels.ne(NON_DIRECT_CLASS)
        exact = direct & labels.eq(direction)
        total = float(selected.sum())
        result[str(topk)] = {
            "selected_edges_per_tile": total / float(images * NFRAG),
            "direct_precision": float(direct.sum()) / total if total else 0.0,
            "direct_recall_all_true": float(direct.sum()) / float(images * DIRECT_EDGES_PER_BOARD),
            "exact_direction_precision": float(exact.sum()) / total if total else 0.0,
            "exact_direction_recall_all_true": float(exact.sum()) / float(images * DIRECT_EDGES_PER_BOARD),
        }
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.n < 1 or args.affinity_k < 1 or args.r2_topk < 1:
        raise ValueError("n, affinity-k, and r2-topk must be positive")
    if not 0.0 <= args.pair_weight <= 1.0:
        raise ValueError("pair-weight must be in [0,1]")
    topks = sorted({int(value) for value in args.topks.split(",") if value.strip()})
    if not topks or min(topks) < 1:
        raise ValueError("topks must be positive comma-separated integers")
    device = _device(args.device)
    affinity, affinity_meta, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity2, affinity_meta2, _ = load_frozen_affinity(args.affinity_ckpt2, device)
    r2 = _load_r2(args.r2_ckpt, device)
    pair_paths = [item.strip() for item in args.pair_ckpts.split(",") if item.strip()]
    pair_models, pair_meta = load_pair_ensemble(pair_paths, device)
    pose_model, pose_meta = load_direct_pose(args.direct_pose_ckpt, device)
    _, val_names = train_val_split()
    if args.n > len(val_names):
        raise ValueError(f"n={args.n} exceeds held-out set size {len(val_names)}")
    dataset = CanvasDataset(val_names[: args.n], real_prob=0.0, seed=args.seed)
    fusion_scores: list[Tensor] = []
    fusion_directions: list[Tensor] = []
    valid_rows: list[Tensor] = []
    label_rows: list[Tensor] = []
    candidate_total = 0.0
    candidate_true = 0.0
    print(
        f"device={device} images={args.n} affinity_k={args.affinity_k} r2_topk={args.r2_topk} "
        f"pair_models={len(pair_models)} pair_weight={args.pair_weight:.2f}",
        flush=True,
    )
    print(f"r2={os.path.abspath(args.r2_ckpt)}", flush=True)
    print(f"affinity_a={os.path.abspath(args.affinity_ckpt)} step={affinity_meta.get('step')}", flush=True)
    print(f"affinity_b={os.path.abspath(args.affinity_ckpt2)} step={affinity_meta2.get('step')}", flush=True)
    print(f"pair={pair_paths}", flush=True)
    print(f"pose={os.path.abspath(args.direct_pose_ckpt)} step={pose_meta.get('step')}", flush=True)
    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("U2 requires exact synthetic held-out labels")
        tiles = sample["tiles"].to(device, non_blocking=device.type == "cuda")
        perm = sample["perm"].to(device, non_blocking=device.type == "cuda").long()
        base_batched, base_valid_batched = mine_affinity_candidates(
            affinity, tiles.unsqueeze(0), candidate_k=args.affinity_k, device=device, affinity_secondary=affinity2
        )
        with _autocast(device):
            r2_scores = r2(tiles.unsqueeze(0))[0].float()
        candidates, valid = _union_candidates(base_batched[0], base_valid_batched[0], r2_scores, args.r2_topk)
        labels = candidate_direct_labels(perm.unsqueeze(0), candidates.unsqueeze(0))[0]
        orientations = score_pairwise_directions(pair_models, tiles, candidates, valid, pair_batch=args.pair_batch, device=device)
        _, _, orientation_z = _candidate_scores_from_orientations(candidates, valid, labels, orientations)
        _, pose_orientations = score_direct_pose_bundle(
            pose_model, tiles, candidates, valid, labels, pair_batch=args.pose_pair_batch, device=device
        )
        combined_orientations = (1.0 - args.pair_weight) * pose_orientations + args.pair_weight * torch.sigmoid(orientation_z)
        score, direction = combined_orientations.max(dim=-1)
        fusion_scores.append(score.cpu())
        fusion_directions.append(direction.cpu())
        valid_rows.append(valid.cpu())
        label_rows.append(labels.cpu())
        candidate_total += float(valid.sum())
        candidate_true += float((valid & labels.ne(NON_DIRECT_CLASS)).sum())
        print(
            f"image={index + 1}/{args.n} candidates_per_tile={float(valid.sum()) / NFRAG:.2f} "
            f"candidate_direct_coverage={float((valid & labels.ne(NON_DIRECT_CLASS)).sum()) / DIRECT_EDGES_PER_BOARD:.4f}",
            flush=True,
        )
    # Width can differ per image after deduplication; evaluate each image separately then sum counts by top-k.
    metrics: dict[str, dict[str, float]] = {}
    for topk in topks:
        selected_total = direct_total = exact_total = 0.0
        for score, direction, valid, labels in zip(fusion_scores, fusion_directions, valid_rows, label_rows):
            selected = _topk_mask(score, valid, topk)
            direct = selected & labels.ne(NON_DIRECT_CLASS)
            exact = direct & labels.eq(direction)
            selected_total += float(selected.sum())
            direct_total += float(direct.sum())
            exact_total += float(exact.sum())
        metrics[str(topk)] = {
            "selected_edges_per_tile": selected_total / float(args.n * NFRAG),
            "direct_precision": direct_total / selected_total if selected_total else 0.0,
            "direct_recall_all_true": direct_total / float(args.n * DIRECT_EDGES_PER_BOARD),
            "exact_direction_precision": exact_total / selected_total if selected_total else 0.0,
            "exact_direction_recall_all_true": exact_total / float(args.n * DIRECT_EDGES_PER_BOARD),
        }
    result = {
        "images": args.n,
        "candidate_direct_coverage": candidate_true / float(args.n * DIRECT_EDGES_PER_BOARD),
        "candidate_edges_per_tile": candidate_total / float(args.n * NFRAG),
        "pair_weight": args.pair_weight,
        "topk": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
