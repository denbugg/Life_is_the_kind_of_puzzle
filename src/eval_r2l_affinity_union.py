"""Evaluate whether R2L directional retrieval adds complementary candidates to R3.

This diagnostic is label-blind while building candidates.  Labels are queried only
for held-out reporting after the union has been frozen.  It deliberately does
not score seams or solve an assignment: U1 is only a candidate-recall gate.
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
from config import GRID, NFRAG, SEED
from direct_pose import NON_DIRECT_CLASS
from imgio import train_val_split
from siamese_directional import DirectionalSiamese
from train_direct_pose import candidate_direct_labels
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_AFFINITY_A = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r1_1200_best.pt"
)
DEFAULT_AFFINITY_B = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r3_1000_best.pt"
)
DEFAULT_R2L = r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R2L_siamese\best.pt"
DIRECT_EDGES_PER_BOARD = 4 * GRID * (GRID - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U1 candidate-recall union: R3 MacroAffinity + R2L directional top-K."
    )
    parser.add_argument("--n", type=int, default=8, help="fresh held-out synthetic boards")
    parser.add_argument("--affinity-k", type=int, default=64)
    parser.add_argument("--r2-topk", type=int, default=8, help="per fixed direction")
    parser.add_argument("--r2-ckpt", default=DEFAULT_R2L)
    parser.add_argument("--affinity-ckpt", default=DEFAULT_AFFINITY_A)
    parser.add_argument("--affinity-ckpt2", default=DEFAULT_AFFINITY_B)
    parser.add_argument("--seed", type=int, default=240815)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _device(value: str | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return device


def _load_r2(path: str, device: torch.device) -> DirectionalSiamese:
    payload: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    kwargs = payload.get("model_kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("R2L checkpoint lacks model_kwargs")
    model = DirectionalSiamese(**kwargs).to(device)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("R2L checkpoint lacks model state")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def _union_candidates(
    base_candidates: Tensor,
    base_valid: Tensor,
    directional_scores: Tensor,
    r2_topk: int,
) -> tuple[Tensor, Tensor]:
    """Union R3 candidates with label-blind top-K R2L candidates per direction."""
    if directional_scores.shape != (4, NFRAG, NFRAG):
        raise ValueError(f"unexpected directional score shape {tuple(directional_scores.shape)}")
    rows: list[list[int]] = []
    max_width = 0
    for anchor in range(NFRAG):
        choices = set(base_candidates[anchor, base_valid[anchor]].detach().cpu().tolist())
        for direction in range(4):
            ranked = torch.topk(directional_scores[direction, anchor], k=r2_topk).indices
            choices.update(int(value) for value in ranked.detach().cpu().tolist() if int(value) != anchor)
        ordered = sorted(choices)
        rows.append(ordered)
        max_width = max(max_width, len(ordered))
    candidates = torch.zeros((NFRAG, max_width), dtype=torch.long, device=base_candidates.device)
    valid = torch.zeros((NFRAG, max_width), dtype=torch.bool, device=base_candidates.device)
    for anchor, values in enumerate(rows):
        if values:
            candidates[anchor, : len(values)] = torch.tensor(values, device=base_candidates.device)
            valid[anchor, : len(values)] = True
    return candidates, valid


def _coverage(perm: Tensor, candidates: Tensor, valid: Tensor) -> tuple[float, float]:
    labels = candidate_direct_labels(perm.unsqueeze(0), candidates.unsqueeze(0))[0]
    direct = valid & labels.ne(NON_DIRECT_CLASS)
    return float(direct.sum().item()) / float(DIRECT_EDGES_PER_BOARD), float(valid.sum().item()) / float(NFRAG)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.n < 1 or args.affinity_k < 1 or args.r2_topk < 1:
        raise ValueError("n, affinity-k, and r2-topk must be positive")
    device = _device(args.device)
    affinity, affinity_meta, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity2, affinity_meta2, _ = load_frozen_affinity(args.affinity_ckpt2, device)
    r2 = _load_r2(args.r2_ckpt, device)
    _, val_names = train_val_split()
    if args.n > len(val_names):
        raise ValueError(f"n={args.n} exceeds held-out set size {len(val_names)}")
    dataset = CanvasDataset(val_names[: args.n], real_prob=0.0, seed=args.seed)
    base_coverages: list[float] = []
    union_coverages: list[float] = []
    base_densities: list[float] = []
    union_densities: list[float] = []
    print(
        f"device={device} images={args.n} affinity_k={args.affinity_k} r2_topk_per_direction={args.r2_topk}",
        flush=True,
    )
    print(f"r2={os.path.abspath(args.r2_ckpt)} step={torch.load(args.r2_ckpt, map_location='cpu', weights_only=False).get('step')}", flush=True)
    print(f"affinity_a={os.path.abspath(args.affinity_ckpt)} step={affinity_meta.get('step')}", flush=True)
    print(f"affinity_b={os.path.abspath(args.affinity_ckpt2)} step={affinity_meta2.get('step')}", flush=True)
    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("U1 requires exact synthetic held-out labels")
        tiles = sample["tiles"].to(device, non_blocking=device.type == "cuda")
        perm = sample["perm"].to(device, non_blocking=device.type == "cuda").long()
        base_batched, base_valid_batched = mine_affinity_candidates(
            affinity,
            tiles.unsqueeze(0),
            candidate_k=args.affinity_k,
            device=device,
            affinity_secondary=affinity2,
        )
        base_candidates, base_valid = base_batched[0], base_valid_batched[0]
        with _autocast(device):
            directional = r2(tiles.unsqueeze(0))[0].float()
        union_candidates, union_valid = _union_candidates(
            base_candidates, base_valid, directional, args.r2_topk
        )
        base_coverage, base_density = _coverage(perm, base_candidates, base_valid)
        union_coverage, union_density = _coverage(perm, union_candidates, union_valid)
        base_coverages.append(base_coverage)
        union_coverages.append(union_coverage)
        base_densities.append(base_density)
        union_densities.append(union_density)
        print(
            f"image={index + 1}/{args.n} base_coverage={base_coverage:.4f} union_coverage={union_coverage:.4f} "
            f"base_edges={base_density:.2f} union_edges={union_density:.2f}",
            flush=True,
        )
    result = {
        "images": args.n,
        "affinity_k": args.affinity_k,
        "r2_topk_per_direction": args.r2_topk,
        "base_direct_coverage": sum(base_coverages) / len(base_coverages),
        "union_direct_coverage": sum(union_coverages) / len(union_coverages),
        "coverage_delta": (sum(union_coverages) - sum(base_coverages)) / len(base_coverages),
        "base_edges_per_tile": sum(base_densities) / len(base_densities),
        "union_edges_per_tile": sum(union_densities) / len(union_densities),
        "density_increase": (sum(union_densities) / sum(base_densities)) - 1.0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
