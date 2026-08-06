"""Cheap oracle-context diagnostic for the next component-growth branch.

This is deliberately *not* another learned seam model.  On fresh exact
synthetic held-out puzzles it gives the evaluator the true oriented predecessor
``A -> B`` and asks it to rank ``B -> C`` candidates from B's frozen dual
affinity union.  It compares:

* ``pair``: an exposure-normalized raw border/profile continuity score for
  ``B -> C``;
* ``triple``: the same pair score plus a two-step (A-B-B-C) profile-jump
  consistency term.  All chains are rotated to canonical left-to-right form.

The oracle predecessor is intentional: this is a narrow information gate for
whether even a tiny correct component supplies measurable context beyond a
single seam.  It is not an assembly result and never trains a model.

Examples
--------

    python src/eval_triple_context_baseline.py --smoke
    python src/eval_triple_context_baseline.py --n 1 --device cuda
    python src/eval_triple_context_baseline.py --n 4 --candidate-k 64
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from imgio import train_val_split
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


# (name, row delta, col delta, number of counter-clockwise quarter turns that
# makes B->C point right).  Canonicalizing all directions is important: a
# horizontal-profile score must not silently privilege horizontal seams.
_DIRECTIONS: tuple[tuple[str, int, int, int], ...] = (
    ("up", -1, 0, 3),
    ("down", 1, 0, 1),
    ("left", 0, -1, 2),
    ("right", 0, 1, 0),
)


@dataclass
class RankSums:
    """Additive exact-chain ranking counts, pooled only at reporting time."""

    eligible: int = 0
    covered: int = 0
    candidate_count: int = 0
    pair_r1: float = 0.0
    pair_r5: float = 0.0
    pair_mrr: float = 0.0
    triple_r1: float = 0.0
    triple_r5: float = 0.0
    triple_mrr: float = 0.0

    def merge(self, other: "RankSums") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


def _preprocess(tiles: Tensor, smooth_kernel: int) -> Tensor:
    """Cancel per-tile affine exposure, then lightly suppress independent noise.

    ``tiles`` can be ``(rows,3,H,W)`` or ``(rows,candidates,3,H,W)``.  The
    normalization is intentionally per tile/channel, because brightness and
    contrast were independently randomized for every fragment.
    """
    if tiles.shape[-3:] != (3, FS, FS):
        raise ValueError(f"expected trailing shape (3,{FS},{FS}), got {tuple(tiles.shape)}")
    shape = tuple(tiles.shape)
    flat = tiles.reshape(-1, 3, FS, FS).float()
    mean = flat.mean(dim=(-2, -1), keepdim=True)
    rms = (flat - mean).square().mean(dim=(-2, -1), keepdim=True).add(1.0e-6).sqrt()
    flat = (flat - mean) / rms
    if smooth_kernel > 1:
        pad = smooth_kernel // 2
        flat = F.avg_pool2d(F.pad(flat, (pad, pad, pad, pad), mode="reflect"), smooth_kernel, stride=1)
    return flat.reshape(shape)


def _score_normalized_chain(
    left: Tensor,
    center: Tensor,
    candidates: Tensor,
    *,
    edge_band: int,
    gradient_weight: float,
    context_weight: float,
) -> tuple[Tensor, Tensor]:
    """Return pair and triple scores for canonical A(left)-B(center)-C(right).

    The profile term compares the vertical RGB trace near B's right border to
    the trace near C's left border.  The context term compares the resulting
    B->C jump with the known A->B jump; this is a simple second finite
    difference across the three-tile chain.  Lower costs become higher scores.
    """
    if left.shape != center.shape or left.ndim != 4:
        raise ValueError("left and center must share shape (rows,3,20,20)")
    if candidates.ndim != 5 or candidates.shape[0] != center.shape[0]:
        raise ValueError("candidates must have shape (rows,K,3,20,20)")

    # Average a narrow edge band before measuring its vertical profile.  This
    # reduces JPEG/noise sensitivity without hiding any global information.
    a_right = left[..., -edge_band:].mean(dim=-1)
    b_left = center[..., :edge_band].mean(dim=-1)
    b_right = center[..., -edge_band:].mean(dim=-1)
    c_left = candidates[..., :edge_band].mean(dim=-1)

    bc_jump = c_left - b_right.unsqueeze(1)
    pair_value = bc_jump.square().mean(dim=(-2, -1))
    if gradient_weight:
        pair_gradient = (bc_jump[..., 1:] - bc_jump[..., :-1]).square().mean(dim=(-2, -1))
        pair_value = pair_value + gradient_weight * pair_gradient

    # A correct predecessor makes this an actual second-order continuation
    # check rather than a fixed additive A-B seam score (which would not affect
    # candidate ranking at all).
    ab_jump = b_left - a_right
    second_difference = bc_jump - ab_jump.unsqueeze(1)
    context_value = second_difference.square().mean(dim=(-2, -1))
    if gradient_weight:
        context_gradient = (
            second_difference[..., 1:] - second_difference[..., :-1]
        ).square().mean(dim=(-2, -1))
        context_value = context_value + gradient_weight * context_gradient
    return -pair_value, -(pair_value + context_weight * context_value)


def _inverse_permutation(perm: Tensor) -> Tensor:
    """Return clean-cell -> shuffled-input-tile lookup for one exact sample."""
    if perm.ndim != 1 or perm.numel() != NFRAG:
        raise ValueError(f"perm must have shape ({NFRAG},), got {tuple(perm.shape)}")
    if not torch.equal(perm.sort().values.cpu(), torch.arange(NFRAG)):
        raise ValueError("synthetic permutation is not a bijection")
    inverse = torch.empty_like(perm)
    inverse.scatter_(0, perm.long(), torch.arange(NFRAG, device=perm.device))
    return inverse


def _ranks(scores: Tensor, target_mask: Tensor) -> Tensor:
    """One-based ranks; all true targets must be finite and unique."""
    if scores.shape != target_mask.shape:
        raise ValueError("score and target-mask shapes differ")
    target_count = target_mask.sum(dim=-1)
    if not bool(torch.all(target_count.eq(1))):
        raise RuntimeError("covered candidate rows must contain exactly one true target")
    target = scores[target_mask].reshape(scores.shape[0])
    if not bool(torch.isfinite(target).all()):
        raise RuntimeError("true candidate received a non-finite score")
    # Strict comparison intentionally gives a stable best rank to exact ties;
    # ties are exceedingly rare for the floating-point profiles used here.
    return scores.gt(target.unsqueeze(1)).sum(dim=-1).add(1)


def _add_rank_values(sums: RankSums, pair_rank: Tensor, triple_rank: Tensor) -> None:
    if pair_rank.numel() != triple_rank.numel():
        raise ValueError("pair/triple rank count mismatch")
    count = int(pair_rank.numel())
    sums.covered += count
    sums.pair_r1 += float(pair_rank.le(1).sum())
    sums.pair_r5 += float(pair_rank.le(5).sum())
    sums.pair_mrr += float(pair_rank.float().reciprocal().sum())
    sums.triple_r1 += float(triple_rank.le(1).sum())
    sums.triple_r5 += float(triple_rank.le(5).sum())
    sums.triple_mrr += float(triple_rank.float().reciprocal().sum())


@torch.inference_mode()
def evaluate_one(
    tiles: Tensor,
    perm: Tensor,
    candidates: Tensor,
    valid: Tensor,
    *,
    edge_band: int,
    smooth_kernel: int,
    gradient_weight: float,
    context_weight: float,
    row_batch: int,
) -> RankSums:
    """Evaluate all oracle three-tile chains from one exact shuffled puzzle."""
    if tuple(tiles.shape) != (NFRAG, 3, FS, FS):
        raise ValueError(f"tiles must have shape ({NFRAG},3,{FS},{FS})")
    if candidates.ndim != 2 or candidates.shape[0] != NFRAG or valid.shape != candidates.shape:
        raise ValueError("candidate ids and valid mask must share shape (576,K)")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")

    inverse = _inverse_permutation(perm)
    cells = perm.long()
    rows = torch.div(cells, GRID, rounding_mode="floor")
    cols = torch.remainder(cells, GRID)
    sums = RankSums()

    for _name, dr, dc, turns in _DIRECTIONS:
        # Both A and C must physically exist, hence B lies away from this
        # direction's two opposite image boundaries.
        exists = (
            (rows - dr).ge(0)
            & (rows - dr).lt(GRID)
            & (cols - dc).ge(0)
            & (cols - dc).lt(GRID)
            & (rows + dr).ge(0)
            & (rows + dr).lt(GRID)
            & (cols + dc).ge(0)
            & (cols + dc).lt(GRID)
        )
        anchor = torch.nonzero(exists, as_tuple=False).squeeze(1)
        sums.eligible += int(anchor.numel())
        if not anchor.numel():
            continue

        a_cells = cells[anchor] - dr * GRID - dc
        c_cells = cells[anchor] + dr * GRID + dc
        a_ids = inverse[a_cells]
        true_c = inverse[c_cells]
        candidate_rows = candidates[anchor]
        valid_rows = valid[anchor]
        target_mask = valid_rows & candidate_rows.eq(true_c.unsqueeze(1))
        covered = target_mask.any(dim=1)
        if not bool(covered.any()):
            continue

        anchor = anchor[covered]
        a_ids = a_ids[covered]
        candidate_rows = candidate_rows[covered]
        valid_rows = valid_rows[covered]
        target_mask = target_mask[covered]
        sums.candidate_count += int(valid_rows.sum())

        # Preprocess the oracle A/B context once per orientation.  Candidate
        # tiles are chunked to keep this cheap on an 8GB desktop GPU.
        left = tiles[a_ids]
        center = tiles[anchor]
        if turns:
            left = torch.rot90(left, turns, dims=(-2, -1))
            center = torch.rot90(center, turns, dims=(-2, -1))
        left = _preprocess(left, smooth_kernel)
        center = _preprocess(center, smooth_kernel)

        for start in range(0, int(anchor.numel()), row_batch):
            stop = min(start + row_batch, int(anchor.numel()))
            candidate_tiles = tiles[candidate_rows[start:stop]]
            if turns:
                candidate_tiles = torch.rot90(candidate_tiles, turns, dims=(-2, -1))
            candidate_tiles = _preprocess(candidate_tiles, smooth_kernel)
            pair_score, triple_score = _score_normalized_chain(
                left[start:stop],
                center[start:stop],
                candidate_tiles,
                edge_band=edge_band,
                gradient_weight=gradient_weight,
                context_weight=context_weight,
            )
            mask = valid_rows[start:stop]
            pair_score = pair_score.masked_fill(~mask, -torch.inf)
            triple_score = triple_score.masked_fill(~mask, -torch.inf)
            _add_rank_values(
                sums,
                _ranks(pair_score, target_mask[start:stop]),
                _ranks(triple_score, target_mask[start:stop]),
            )
    return sums


def _summary(label: str, sums: RankSums) -> str:
    if sums.eligible == 0:
        return f"{label}: no eligible oracle chains"
    coverage = sums.covered / sums.eligible
    if sums.covered == 0:
        return f"{label}: candidate coverage=0.0000 (0/{sums.eligible})"
    pair_r1 = sums.pair_r1 / sums.covered
    triple_r1 = sums.triple_r1 / sums.covered
    return (
        f"{label}: candidate coverage={coverage:.4f} ({sums.covered}/{sums.eligible}), "
        f"mean valid candidates={sums.candidate_count / sums.covered:.1f}\n"
        f"  pair   conditional R@1={pair_r1:.4f} R@5={sums.pair_r5 / sums.covered:.4f} "
        f"MRR={sums.pair_mrr / sums.covered:.4f}\n"
        f"  triple conditional R@1={triple_r1:.4f} R@5={sums.triple_r5 / sums.covered:.4f} "
        f"MRR={sums.triple_mrr / sums.covered:.4f} "
        f"(delta R@1={triple_r1 - pair_r1:+.4f})"
    )


def smoke() -> dict[str, float]:
    """Data-free shape/finite-value contract for fast CI-style validation."""
    generator = torch.Generator().manual_seed(7)
    left = torch.rand((3, 3, FS, FS), generator=generator)
    center = torch.rand((3, 3, FS, FS), generator=generator)
    candidate = torch.rand((3, 5, 3, FS, FS), generator=generator)
    pair, triple = _score_normalized_chain(
        _preprocess(left, 3), _preprocess(center, 3), _preprocess(candidate, 3),
        edge_band=3, gradient_weight=0.5, context_weight=0.5,
    )
    if pair.shape != (3, 5) or triple.shape != (3, 5):
        raise RuntimeError("unexpected smoke score shape")
    if not bool(torch.isfinite(pair).all() and torch.isfinite(triple).all()):
        raise RuntimeError("non-finite raw continuation score")
    return {"pair_mean": float(pair.mean()), "triple_mean": float(triple.mean())}


def _parse_args() -> argparse.Namespace:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--affinity-ckpt", "--affinity_ckpt", dest="affinity_ckpt",
        default=os.path.join(workspace, "artifacts", "macro_affinity", "affinity_r1_1200_best.pt"),
        help="primary frozen MacroAffinityNet checkpoint",
    )
    parser.add_argument(
        "--affinity-ckpt2", "--affinity_ckpt2", dest="affinity_ckpt2",
        default=os.path.join(workspace, "artifacts", "macro_affinity", "affinity_r3_1000_best.pt"),
        help="secondary frozen MacroAffinityNet checkpoint; pass an empty string to disable union",
    )
    parser.add_argument("--n", type=int, default=4, help="fresh exact synthetic held-out puzzles")
    parser.add_argument("--candidate-k", "--candidate_k", dest="candidate_k", type=int, default=64)
    parser.add_argument("--edge-band", "--edge_band", dest="edge_band", type=int, default=3)
    parser.add_argument("--smooth-kernel", "--smooth_kernel", dest="smooth_kernel", type=int, default=3)
    parser.add_argument("--gradient-weight", "--gradient_weight", dest="gradient_weight", type=float, default=0.5)
    parser.add_argument("--context-weight", "--context_weight", dest="context_weight", type=float, default=0.5)
    parser.add_argument("--row-batch", "--row_batch", dest="row_batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED + 12121, help="fresh synthetic corruption seed")
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--smoke", action="store_true", help="run only the data-free contract test")
    args = parser.parse_args()
    if args.n < 1 or args.row_batch < 1:
        parser.error("--n and --row-batch must be positive")
    if not 1 <= args.candidate_k < NFRAG:
        parser.error(f"--candidate-k must lie in [1,{NFRAG - 1}]")
    if not 1 <= args.edge_band <= FS:
        parser.error(f"--edge-band must lie in [1,{FS}]")
    if args.smooth_kernel < 1 or args.smooth_kernel > FS or not args.smooth_kernel % 2:
        parser.error(f"--smooth-kernel must be an odd integer in [1,{FS}]")
    if args.gradient_weight < 0.0 or args.context_weight < 0.0:
        parser.error("--gradient-weight and --context-weight must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    if args.smoke:
        print(f"[triple-context baseline smoke] {smoke()}", flush=True)
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    affinity, _, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity_secondary = None
    if args.affinity_ckpt2:
        affinity_secondary, _, _ = load_frozen_affinity(args.affinity_ckpt2, device)
    _, validation_names = train_val_split()
    if args.n > len(validation_names):
        raise ValueError(f"--n={args.n} exceeds held-out split size {len(validation_names)}")
    dataset = CanvasDataset(validation_names[: args.n], real_prob=0.0, seed=args.seed)

    print(
        f"device={device} exact_fresh_heldout_images={args.n} "
        f"dual_affinity={affinity_secondary is not None} top{args.candidate_k}/encoder "
        f"edge_band={args.edge_band} smooth={args.smooth_kernel} "
        f"gradient_weight={args.gradient_weight:g} context_weight={args.context_weight:g}",
        flush=True,
    )
    print(
        "oracle setup: A is the exact predecessor of B; C is ranked only in B's frozen affinity list. "
        "Conditional ranks exclude candidate-graph misses.",
        flush=True,
    )

    total = RankSums()
    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("triple-context diagnostic requires exact synthetic CanvasDataset samples")
        tiles = sample["tiles"].to(device, non_blocking=device.type == "cuda")
        perm = sample["perm"].to(device, non_blocking=device.type == "cuda").long()
        candidates, valid = mine_affinity_candidates(
            affinity,
            tiles.unsqueeze(0),
            candidate_k=args.candidate_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        one = evaluate_one(
            tiles, perm, candidates[0], valid[0], edge_band=args.edge_band,
            smooth_kernel=args.smooth_kernel, gradient_weight=args.gradient_weight,
            context_weight=args.context_weight, row_batch=args.row_batch,
        )
        total.merge(one)
        print(_summary(f"image {index + 1}/{args.n}", one), flush=True)

    print("\n=== pooled oracle triple-context diagnostic ===", flush=True)
    print(_summary(f"pooled n={args.n}", total), flush=True)


if __name__ == "__main__":
    main()
