"""Audit whether frozen per-tile denoising improves *candidate* edge matching.

This is intentionally an evaluation-only experiment.  It creates fresh exact
synthetic puzzles from held-out clean targets, freezes the current raw-tile
affinity union as the candidate graph, and asks whether preprocessing the
tiles before scoring improves the rank of the true right/down neighbour among
those candidates.  It never uses ``perms.npz`` or a recovered real-input
permutation.

The candidate graph remains raw because both frozen affinity encoders were
trained on raw corrupt tiles.  Consequently this answers the useful narrow
question for a possible denoise -> edge-graph pivot: does denoising make a
*second-stage* seam/edge scorer better without invalidating its candidate
distribution?

Examples:

    python src/eval_denoise_matching.py --n 1 --modes raw,denoise
    python src/eval_denoise_matching.py --n 4 --modes raw,norm,denoise,denoise_norm

Metrics whose name ends in ``_all`` count an affinity miss as a failure; those
ending in ``_covered`` condition on the true direct neighbour being present in
the frozen candidate set.  ``border`` is a deliberately simple non-learned
one-pixel RGB seam score.  ``pair`` is the existing PairwiseNet ensemble.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor

from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from imgio import train_val_split
from match_preprocess import load_match_denoiser, photometric_normalize_tensor
from pipeline import load_pair
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_AFFINITY_1 = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r1_1200_best.pt"
)
DEFAULT_AFFINITY_2 = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r3_1000_best.pt"
)


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def parse_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return device


def clean_tiles_in_input_order(sample: dict[str, Tensor]) -> Tensor:
    """Return clean 20x20 targets aligned with the shuffled input tile order."""
    clean = sample["clean"].float()
    perm = sample["perm"].long()
    if tuple(clean.shape) != (3, GRID * FS, GRID * FS):
        raise ValueError(f"unexpected clean canvas shape {tuple(clean.shape)}")
    ordered = (
        clean.reshape(3, GRID, FS, GRID, FS)
        .permute(1, 3, 0, 2, 4)
        .reshape(NFRAG, 3, FS, FS)
    )
    return ordered[perm]


def direct_truth(perm: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build exact right/down input-tile targets from input->clean-cell labels."""
    perm = perm.long().cpu()
    if tuple(perm.shape) != (NFRAG,):
        raise ValueError(f"perm must be ({NFRAG},), got {tuple(perm.shape)}")
    inverse = torch.empty(NFRAG, dtype=torch.long)
    inverse[perm] = torch.arange(NFRAG, dtype=torch.long)
    cells = perm
    has_right = cells.remainder(GRID).ne(GRID - 1)
    has_down = torch.div(cells, GRID, rounding_mode="floor").ne(GRID - 1)
    right = torch.full((NFRAG,), -1, dtype=torch.long)
    down = torch.full((NFRAG,), -1, dtype=torch.long)
    right[has_right] = inverse[cells[has_right] + 1]
    down[has_down] = inverse[cells[has_down] + GRID]
    return right, has_right, down, has_down


def candidate_rows(
    candidates: Tensor, valid: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten the deduplicated directed candidate graph for batched scoring."""
    if candidates.shape != valid.shape or candidates.ndim != 2:
        raise ValueError("candidate and validity tensors must be matching (576,K) arrays")
    anchors = torch.arange(NFRAG, device=candidates.device).view(-1, 1).expand_as(candidates)
    return anchors[valid], candidates[valid].long(), valid


def _rank_metrics(
    scores: Tensor,
    candidates: Tensor,
    valid: Tensor,
    truth: Tensor,
    eligible: Tensor,
) -> dict[str, float]:
    """Rank an exact directional target among this anchor's valid candidates."""
    if scores.shape != candidates.shape or valid.shape != candidates.shape:
        raise ValueError("scores, candidates, and valid must share one shape")
    truth = truth.to(candidates.device)
    eligible = eligible.to(candidates.device)
    match = candidates.eq(truth[:, None]) & valid & eligible[:, None]
    covered = match.any(dim=1) & eligible
    total = int(eligible.sum().item())
    covered_count = int(covered.sum().item())
    result: dict[str, float] = {
        "eligible": float(total),
        "covered": float(covered_count),
        "coverage": covered_count / max(1, total),
    }
    if not covered_count:
        for key in ("r1_all", "r5_all", "r10_all", "r1_covered", "r5_covered", "r10_covered", "mrr_covered", "median_rank_covered"):
            result[key] = 0.0
        return result

    masked = scores.masked_fill(~valid, -torch.inf)
    truth_score = scores.masked_fill(~match, -torch.inf).amax(dim=1)
    # Exact score ties are exceptionally rare here; the strict comparison gives
    # the usual competition-style rank 1 + number of candidates that beat truth.
    rank = (masked > truth_score[:, None]).sum(dim=1).add(1)
    ranked = rank[covered].float()
    for cutoff in (1, 5, 10):
        hits = (ranked <= cutoff).float()
        result[f"r{cutoff}_all"] = float(hits.sum() / max(1, total))
        result[f"r{cutoff}_covered"] = float(hits.mean())
    result["mrr_covered"] = float(ranked.reciprocal().mean())
    result["median_rank_covered"] = float(ranked.median())
    return result


def merge_directions(
    right: dict[str, float], down: dict[str, float]) -> dict[str, float]:
    """Combine directional metrics by their exact number of eligible edges."""
    total = right["eligible"] + down["eligible"]
    covered = right["covered"] + down["covered"]
    out = {
        "edges": total,
        "covered": covered,
        "coverage": covered / max(1.0, total),
    }
    for cutoff in (1, 5, 10):
        # ``r@k_all * eligible`` is exactly the number of correct edges.
        out[f"r{cutoff}_all"] = (
            right[f"r{cutoff}_all"] * right["eligible"]
            + down[f"r{cutoff}_all"] * down["eligible"]
        ) / max(1.0, total)
        out[f"r{cutoff}_covered"] = (
            right[f"r{cutoff}_covered"] * right["covered"]
            + down[f"r{cutoff}_covered"] * down["covered"]
        ) / max(1.0, covered)
    out["mrr_covered"] = (
        right["mrr_covered"] * right["covered"] + down["mrr_covered"] * down["covered"]
    ) / max(1.0, covered)
    out["median_rank_covered"] = 0.5 * (
        right["median_rank_covered"] + down["median_rank_covered"]
    )
    return out


def border_scores(tiles: Tensor, candidates: Tensor, valid: Tensor, *, transpose: bool) -> Tensor:
    """Simple non-learned negative RGB seam-MSE for candidate pairs.

    The score intentionally has no trainable component.  For a horizontal
    seam it compares the last column of the source with the first column of
    the target; transposition produces the analogous vertical score.
    """
    sources, targets, _ = candidate_rows(candidates, valid)
    x = tiles.transpose(-1, -2) if transpose else tiles
    left = x[sources, :, :, -1]
    right = x[targets, :, :, 0]
    flat = -(left - right).square().mean(dim=(1, 2))
    out = torch.full(candidates.shape, -torch.inf, dtype=torch.float32, device=tiles.device)
    out[valid] = flat.float()
    return out


@torch.inference_mode()
def pair_scores(
    models: Iterable[torch.nn.Module],
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    *,
    transpose: bool,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Score only affinity candidates using the frozen existing pair ensemble."""
    source, target, _ = candidate_rows(candidates, valid)
    source = source.long()
    target = target.long()
    x = tiles.transpose(-1, -2) if transpose else tiles
    model_list = list(models)
    output = torch.empty(source.numel(), dtype=torch.float32, device="cpu")
    for begin in range(0, source.numel(), batch_size):
        end = min(source.numel(), begin + batch_size)
        pair = torch.cat((x[source[begin:end]], x[target[begin:end]]), dim=-1)
        with _autocast(device):
            logits = sum(model(pair).float() for model in model_list) / len(model_list)
        output[begin:end] = logits.detach().float().cpu()
    dense = torch.full(candidates.shape, -torch.inf, dtype=torch.float32)
    dense[valid.detach().cpu()] = output
    return dense.to(tiles.device)


@torch.inference_mode()
def denoise_tiles(model: torch.nn.Module, tiles: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
    parts: list[Tensor] = []
    for begin in range(0, tiles.shape[0], batch_size):
        with _autocast(device):
            parts.append(model(tiles[begin : begin + batch_size]).float())
    return torch.cat(parts, dim=0)


def describe_metrics(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label}: cov={metrics['coverage']:.3f} "
        f"R1(all/covered)={metrics['r1_all']:.3f}/{metrics['r1_covered']:.3f} "
        f"R5(all/covered)={metrics['r5_all']:.3f}/{metrics['r5_covered']:.3f} "
        f"MRRcovered={metrics['mrr_covered']:.3f} medrank={metrics['median_rank_covered']:.1f}"
    )


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = set().union(*(row.keys() for row in rows))
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1, help="number of held-out exact synthetic puzzles")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=64, help="top-K from each frozen affinity encoder")
    parser.add_argument("--affinity_ckpt", default=DEFAULT_AFFINITY_1)
    parser.add_argument("--affinity_ckpt2", default=DEFAULT_AFFINITY_2)
    parser.add_argument("--modes", default="raw,denoise")
    parser.add_argument("--denoise_tag", default="matchden")
    parser.add_argument("--pair_tag", default="pair")
    parser.add_argument("--pair_batch", type=int, default=4096)
    parser.add_argument("--denoise_batch", type=int, default=576)
    parser.add_argument(
        "--with_pair",
        action="store_true",
        help="also run the expensive existing PairwiseNet ensemble on candidates",
    )
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    if args.top_k < 1 or args.top_k >= NFRAG:
        parser.error(f"--top_k must be in [1,{NFRAG - 1}]")
    if args.pair_batch < 1 or args.denoise_batch < 1:
        parser.error("batch sizes must be positive")
    modes = tuple(dict.fromkeys(mode.strip() for mode in args.modes.split(",") if mode.strip()))
    allowed = {"raw", "norm", "denoise", "denoise_norm"}
    if not modes or any(mode not in allowed for mode in modes):
        parser.error(f"--modes must be a non-empty subset of {sorted(allowed)}")

    device = parse_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("== DENOISE -> CANDIDATE MATCHING AUDIT ==", flush=True)
    print(
        f"device={device} n={args.n} top_k/encoder={args.top_k} modes={','.join(modes)} "
        f"candidate graph=RAW frozen affinity union",
        flush=True,
    )
    affinity, _, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity2 = None
    if args.affinity_ckpt2:
        affinity2, _, _ = load_frozen_affinity(args.affinity_ckpt2, device)
    print(f"affinity_1={os.path.abspath(args.affinity_ckpt)}", flush=True)
    if affinity2 is not None:
        print(f"affinity_2={os.path.abspath(args.affinity_ckpt2)}", flush=True)

    denoiser = None
    if any(mode.startswith("denoise") for mode in modes):
        denoiser, denoise_ckpt = load_match_denoiser(args.denoise_tag, device=str(device))
        if denoiser is None:
            raise FileNotFoundError(f"no matching denoiser checkpoint for tag={args.denoise_tag}")
        print(
            f"denoiser={args.denoise_tag} step={denoise_ckpt.get('step')} "
            f"val_frag_l1={denoise_ckpt.get('val')}",
            flush=True,
        )
    pair_models = None
    if args.with_pair:
        pair_models, pair_ckpt = load_pair(args.pair_tag)
        if pair_models is None:
            raise FileNotFoundError(f"no existing pair checkpoint for tag={args.pair_tag}")
        print(
            f"pair={args.pair_tag} ensemble={len(pair_models)} step={pair_ckpt.get('step')} "
            f"val={pair_ckpt.get('val')}",
            flush=True,
        )

    _, val_names = train_val_split()
    dataset = CanvasDataset(val_names, real_prob=0.0, seed=args.seed)
    results: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    denoise_l1: list[float] = []
    raw_l1: list[float] = []
    candidate_counts: list[float] = []
    candidate_runtime: list[float] = []
    preprocess_runtime: dict[str, list[float]] = defaultdict(list)
    scorer_runtime: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for index in range(args.n):
        # CanvasDataset deliberately adds fresh worker entropy.  Seed the global
        # draw immediately before __getitem__ so this audit is reproducible.
        np.random.seed(args.seed + index)
        sample = dataset[index]
        name = val_names[index]
        tiles_raw = sample["tiles"].to(device, non_blocking=device.type == "cuda")
        clean = clean_tiles_in_input_order(sample).to(device)
        perm = sample["perm"].long()
        if not bool(sample["has_perm"]):
            raise RuntimeError("audit requires exact synthetic permutation labels")

        start = time.perf_counter()
        with torch.inference_mode():
            candidates_b, valid_b = mine_affinity_candidates(
                affinity,
                tiles_raw.unsqueeze(0),
                candidate_k=args.top_k,
                device=device,
                affinity_secondary=affinity2,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        candidate_runtime.append(time.perf_counter() - start)
        candidates = candidates_b[0]
        valid = valid_b[0]
        candidate_counts.append(float(valid.sum().item()) / NFRAG)

        truth_r, has_r, truth_d, has_d = direct_truth(perm)
        raw_l1.append(float((tiles_raw - clean).abs().mean().item()))
        prepared: dict[str, Tensor] = {"raw": tiles_raw}
        if "norm" in modes:
            prepared["norm"] = photometric_normalize_tensor(tiles_raw)
        if denoiser is not None:
            start = time.perf_counter()
            prepared["denoise"] = denoise_tiles(
                denoiser, tiles_raw, batch_size=args.denoise_batch, device=device
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            preprocess_runtime["denoise"].append(time.perf_counter() - start)
            denoise_l1.append(float((prepared["denoise"] - clean).abs().mean().item()))
            if "denoise_norm" in modes:
                prepared["denoise_norm"] = photometric_normalize_tensor(prepared["denoise"])

        print(
            f"[{index + 1}/{args.n}] {name} candidates/tile={candidate_counts[-1]:.1f} "
            f"raw_L1={raw_l1[-1]:.5f}" + (f" denoise_L1={denoise_l1[-1]:.5f}" if denoise_l1 else ""),
            flush=True,
        )
        for mode in modes:
            tiles = prepared[mode]
            start = time.perf_counter()
            border_r = border_scores(tiles, candidates, valid, transpose=False)
            border_d = border_scores(tiles, candidates, valid, transpose=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            scorer_runtime[mode]["border"].append(time.perf_counter() - start)
            border_metrics = merge_directions(
                _rank_metrics(border_r, candidates, valid, truth_r, has_r),
                _rank_metrics(border_d, candidates, valid, truth_d, has_d),
            )
            results[mode]["border"].append(border_metrics)
            print("  " + describe_metrics(f"{mode}/border", border_metrics), flush=True)

            if pair_models is not None:
                start = time.perf_counter()
                pair_r = pair_scores(
                    pair_models, tiles, candidates, valid, transpose=False,
                    batch_size=args.pair_batch, device=device,
                )
                pair_d = pair_scores(
                    pair_models, tiles, candidates, valid, transpose=True,
                    batch_size=args.pair_batch, device=device,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                scorer_runtime[mode]["pair"].append(time.perf_counter() - start)
                pair_metrics = merge_directions(
                    _rank_metrics(pair_r, candidates, valid, truth_r, has_r),
                    _rank_metrics(pair_d, candidates, valid, truth_d, has_d),
                )
                results[mode]["pair"].append(pair_metrics)
                print("  " + describe_metrics(f"{mode}/pair", pair_metrics), flush=True)

    print("\n== AGGREGATE (exact synthetic held-out) ==", flush=True)
    print(
        f"candidate graph: mean candidates/tile={np.mean(candidate_counts):.2f}, "
        f"build_time={np.mean(candidate_runtime):.2f}s/image",
        flush=True,
    )
    print(f"pixel L1 to clean: raw={np.mean(raw_l1):.5f}", end="", flush=True)
    if denoise_l1:
        print(f"  denoise={np.mean(denoise_l1):.5f}  delta={np.mean(denoise_l1) - np.mean(raw_l1):+.5f}", flush=True)
    else:
        print(flush=True)
    for mode in modes:
        for scorer in ("border", "pair"):
            rows = results[mode].get(scorer, [])
            if not rows:
                continue
            metrics = average_metric_rows(rows)
            print("  " + describe_metrics(f"{mode}/{scorer}", metrics), flush=True)
            print(
                f"    scorer_time={np.mean(scorer_runtime[mode][scorer]):.2f}s/image"
                + (
                    f"; denoise_time={np.mean(preprocess_runtime['denoise']):.2f}s/image"
                    if mode.startswith("denoise") and preprocess_runtime["denoise"]
                    else ""
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
