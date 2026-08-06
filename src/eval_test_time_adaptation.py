"""Label-free per-puzzle adaptation gate for the frozen seam ranker.

Only a tiny ranking-head adapter is optimized for one 576-tile bag.  Pseudo
labels come from high-margin reciprocal predictions and closed 2x2 loops.
Ground-truth permutation labels are accepted only by the metric function and
never by pseudo-label selection or adaptation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from candidate_rank import (
    DOWN,
    LEFT,
    NUM_DIRECTIONS,
    RIGHT,
    UP,
    CandidateSeamRanker,
    ListwiseRows,
    candidate_target_slots,
    finalize_rank_metrics,
    inverse_directions,
    neighbor_targets,
    rank_metric_sums,
    score_candidate_rows,
    select_listwise_rows,
)
from canvas_data import CanvasDataset
from config import NFRAG, SEED
from imgio import train_val_split
from train_candidate_rank import _reciprocal_counts
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


@dataclass(frozen=True)
class PseudoRows:
    rows: ListwiseRows
    mutual_count: int
    loop_edge_count: int
    confidence_threshold: float


def _all_rows(candidates: Tensor, valid: Tensor) -> ListwiseRows:
    """Create every anchor/direction row without consulting puzzle labels."""
    if candidates.shape[0] != 1:
        raise ValueError("per-puzzle adaptation expects one bag")
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("every anchor needs at least one candidate")
    anchors = torch.arange(NFRAG, device=candidates.device).repeat_interleave(
        NUM_DIRECTIONS
    )
    directions = torch.arange(
        NUM_DIRECTIONS, device=candidates.device
    ).repeat(NFRAG)
    image_ids = torch.zeros_like(anchors)
    first = valid[0].long().argmax(dim=-1)
    slots = first[anchors]
    targets = candidates[0, anchors, slots]
    return ListwiseRows(image_ids, anchors, directions, slots, targets)


def _row_predictions(
    candidates: Tensor,
    rows: ListwiseRows,
    scores: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    values, indices = torch.topk(scores.float(), k=2, dim=-1)
    predicted_slots = indices[:, 0]
    predicted_targets = candidates[
        rows.image_ids, rows.anchors, predicted_slots
    ]
    margins = values[:, 0] - values[:, 1]
    return predicted_slots, predicted_targets, margins


@torch.no_grad()
def build_pseudo_rows(
    model: CandidateSeamRanker,
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    *,
    pair_batch: int,
    confidence_quantile: float,
    max_rows: int,
    probe_rows: int,
) -> PseudoRows:
    """Select label-free reciprocal and loop-consistent directed edges."""
    if not 0.0 <= confidence_quantile <= 1.0:
        raise ValueError("confidence_quantile must lie in [0,1]")
    all_rows = _all_rows(candidates, valid)
    if 0 < probe_rows < all_rows.count:
        indices = (
            torch.arange(probe_rows, device=tiles.device)
            * all_rows.count
            // probe_rows
        )
        rows = ListwiseRows(
            all_rows.image_ids[indices],
            all_rows.anchors[indices],
            all_rows.directions[indices],
            all_rows.target_slots[indices],
            all_rows.target_indices[indices],
        )
    else:
        rows = all_rows
    scores = score_candidate_rows(
        model, tiles, candidates, valid, rows, pair_batch=pair_batch
    )
    slots, predicted, margins = _row_predictions(candidates, rows, scores)

    reverse_directions = inverse_directions(rows.directions)
    reverse_candidates = candidates[0, predicted]
    reverse_valid = valid[0, predicted]
    anchor_match = reverse_valid & reverse_candidates.eq(rows.anchors[:, None])
    reverse_present = anchor_match.any(dim=-1)
    mutual = torch.zeros(rows.count, dtype=torch.bool, device=tiles.device)
    if bool(reverse_present.any()):
        selected = torch.nonzero(reverse_present, as_tuple=False).flatten()
        reverse_slots = anchor_match.long().argmax(dim=-1)
        reverse_rows = ListwiseRows(
            torch.zeros_like(selected),
            predicted[selected],
            reverse_directions[selected],
            reverse_slots[selected],
            rows.anchors[selected],
        )
        reverse_scores = score_candidate_rows(
            model,
            tiles,
            candidates,
            valid,
            reverse_rows,
            pair_batch=pair_batch,
        )
        mutual[selected] = reverse_scores.argmax(dim=-1).eq(
            reverse_slots[selected]
        )

    loop_flat = torch.zeros(rows.count, dtype=torch.bool, device=tiles.device)
    # A --right--> B
    # |             |
    # down          down
    # v             v
    # C --right--> D
    if rows.count == NFRAG * NUM_DIRECTIONS:
        prediction_grid = predicted.reshape(NFRAG, NUM_DIRECTIONS)
        mutual_grid = mutual.reshape(NFRAG, NUM_DIRECTIONS)
        loop_edges = loop_flat.reshape(NFRAG, NUM_DIRECTIONS)
        for anchor in range(NFRAG):
            b = int(prediction_grid[anchor, RIGHT])
            c = int(prediction_grid[anchor, DOWN])
            d_from_b = int(prediction_grid[b, DOWN])
            d_from_c = int(prediction_grid[c, RIGHT])
            if d_from_b != d_from_c:
                continue
            if not bool(
                mutual_grid[anchor, RIGHT]
                and mutual_grid[anchor, DOWN]
                and mutual_grid[b, DOWN]
                and mutual_grid[c, RIGHT]
            ):
                continue
            loop_edges[anchor, RIGHT] = True
            loop_edges[anchor, DOWN] = True
            loop_edges[b, DOWN] = True
            loop_edges[c, RIGHT] = True

    mutual_margins = margins[mutual]
    threshold = (
        float(torch.quantile(mutual_margins, confidence_quantile))
        if mutual_margins.numel()
        else math.inf
    )
    accepted = loop_flat | (mutual & margins.ge(threshold))
    chosen = torch.nonzero(accepted, as_tuple=False).flatten()
    if chosen.numel() > max_rows:
        # Loop edges are retained first; remaining capacity takes the largest
        # reciprocal margins.  No truth label participates in this ordering.
        loop_indices = chosen[loop_flat[chosen]]
        other = chosen[~loop_flat[chosen]]
        room = max(0, max_rows - int(loop_indices.numel()))
        if room and other.numel():
            other = other[torch.argsort(margins[other], descending=True)[:room]]
        else:
            other = other[:0]
        chosen = torch.cat((loop_indices[:max_rows], other))
    if not chosen.numel():
        # Safe no-op fallback: adaptation will be skipped and paired metrics
        # remain exactly equal to baseline.
        empty = rows.image_ids[:0]
        selected_rows = ListwiseRows(empty, empty, empty, empty, empty)
    else:
        selected_rows = ListwiseRows(
            rows.image_ids[chosen],
            rows.anchors[chosen],
            rows.directions[chosen],
            slots[chosen],
            predicted[chosen],
        )
    return PseudoRows(
        selected_rows,
        mutual_count=int(mutual.sum()),
        loop_edge_count=int(loop_flat.sum()),
        confidence_threshold=threshold,
    )


def _mild_augment(
    tiles: Tensor,
    generator: torch.Generator,
    *,
    noise_std: float,
    photometric: float,
) -> Tensor:
    """Add a second, mild independent corruption to each observed tile."""
    count = tiles.shape[1]
    contrast = 1.0 + (
        torch.rand(
            (1, count, 1, 1, 1),
            device=tiles.device,
            generator=generator,
        )
        * 2.0
        - 1.0
    ) * photometric
    brightness = (
        torch.rand(
            (1, count, 1, 1, 1),
            device=tiles.device,
            generator=generator,
        )
        * 2.0
        - 1.0
    ) * photometric
    noise = torch.randn(
        tiles.shape,
        device=tiles.device,
        dtype=tiles.dtype,
        generator=generator,
    ) * noise_std
    return (tiles * contrast + brightness + noise).clamp(0.0, 1.0)


def _adapter_parameters(
    model: CandidateSeamRanker,
    mode: str,
) -> list[nn.Parameter]:
    """Expose a small, explicit adapter parameter set."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[nn.Parameter] = []
    if mode == "head":
        for name, parameter in model.named_parameters():
            if (
                name.startswith("head.0.")
                or name.startswith("head.1.")
                or name.startswith("head.4.")
            ):
                parameter.requires_grad_(True)
                selected.append(parameter)
    elif mode == "norm":
        for module in model.modules():
            if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad_(True)
                    selected.append(parameter)
    else:
        raise ValueError("adapter mode must be 'head' or 'norm'")
    if not selected:
        raise RuntimeError("no adapter parameters selected")
    return selected


def adapt_one_puzzle(
    base_model: CandidateSeamRanker,
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    pseudo: PseudoRows,
    *,
    steps: int,
    lr: float,
    pair_batch: int,
    seed: int,
    noise_std: float,
    photometric: float,
    distill_weight: float,
    trust_weight: float,
    temperature: float,
    adapter_mode: str,
) -> tuple[CandidateSeamRanker, dict[str, float]]:
    """Adapt without any permutation/position labels."""
    adapted = copy.deepcopy(base_model).to(tiles.device).eval()
    parameters = _adapter_parameters(adapted, adapter_mode)
    initial = [value.detach().clone() for value in parameters]
    if pseudo.rows.count == 0 or steps == 0:
        return adapted, {
            "pseudo_rows": float(pseudo.rows.count),
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "adapter_l2": 0.0,
        }

    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    with torch.no_grad():
        base_scores = score_candidate_rows(
            base_model,
            tiles,
            candidates,
            valid,
            pseudo.rows,
            pair_batch=pair_batch,
        ).float()
        row_valid = valid[
            pseudo.rows.image_ids, pseudo.rows.anchors
        ]
        base_probability = F.softmax(
            base_scores.masked_fill(~row_valid, -1.0e4) / temperature, dim=-1
        )

    generator = torch.Generator(device=tiles.device)
    generator.manual_seed(seed)
    initial_loss = math.nan
    final_loss = math.nan
    for step in range(steps):
        augmented = _mild_augment(
            tiles,
            generator,
            noise_std=noise_std,
            photometric=photometric,
        )
        optimizer.zero_grad(set_to_none=True)
        scores = score_candidate_rows(
            adapted,
            augmented,
            candidates,
            valid,
            pseudo.rows,
            pair_batch=pair_batch,
            checkpoint_chunks=True,
        ).float()
        finite_scores = scores.masked_fill(~row_valid, -1.0e4)
        pseudo_loss = F.cross_entropy(
            finite_scores / temperature, pseudo.rows.target_slots
        )
        distill = F.kl_div(
            F.log_softmax(finite_scores / temperature, dim=-1),
            base_probability,
            reduction="batchmean",
        )
        trust = torch.stack(
            [
                current.sub(reference).square().mean()
                for current, reference in zip(parameters, initial)
            ]
        ).mean()
        loss = pseudo_loss + distill_weight * distill + trust_weight * trust
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite TTA gradient at step {step + 1}")
        optimizer.step()
        value = float(loss.detach())
        if step == 0:
            initial_loss = value
        final_loss = value
    adapter_l2 = float(
        torch.stack(
            [
                current.detach().sub(reference).square().mean()
                for current, reference in zip(parameters, initial)
            ]
        ).mean()
    )
    return adapted.eval(), {
        "pseudo_rows": float(pseudo.rows.count),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "adapter_l2": adapter_l2,
    }


def _load_ranker(path: str, device: torch.device) -> CandidateSeamRanker:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CandidateSeamRanker(**payload["model_kwargs"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def _paired_metrics(
    model: CandidateSeamRanker,
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    rows: ListwiseRows,
    *,
    pair_batch: int,
) -> dict[str, float]:
    scores = score_candidate_rows(
        model, tiles, candidates, valid, rows, pair_batch=pair_batch
    ).float()
    metrics = finalize_rank_metrics(rank_metric_sums(scores, rows.target_slots))
    reciprocal = _reciprocal_counts(
        model,
        tiles,
        candidates,
        valid,
        rows,
        scores,
        pair_batch=pair_batch,
    )
    predicted = reciprocal["reciprocal_predicted"]
    metrics["reciprocal_exact_precision"] = (
        reciprocal["reciprocal_exact"] / predicted if predicted else 0.0
    )
    metrics["reciprocal_exact_coverage"] = (
        reciprocal["reciprocal_exact"] / rows.count if rows.count else 0.0
    )
    return metrics


def _pseudo_diagnostics(
    pseudo: PseudoRows,
    exact_targets: Tensor,
) -> dict[str, float]:
    if pseudo.rows.count:
        truth = exact_targets[
            pseudo.rows.image_ids,
            pseudo.rows.anchors,
            pseudo.rows.directions,
        ]
        precision = float(pseudo.rows.target_indices.eq(truth).float().mean())
    else:
        precision = 0.0
    return {
        "pseudo_rows": float(pseudo.rows.count),
        "pseudo_exact_precision_diagnostic": precision,
        "mutual_count": float(pseudo.mutual_count),
        "loop_edge_count": float(pseudo.loop_edge_count),
        "confidence_threshold": float(pseudo.confidence_threshold),
    }


def gate_result(result: dict[str, Any]) -> dict[str, Any]:
    delta = result["delta_mean"]
    thresholds = {
        "candidate_target_r1": 0.05,
        "reciprocal_exact_precision": 0.08,
        "reciprocal_exact_coverage": -0.02,
    }
    checks = {
        "r1_improvement": delta["candidate_target_r1"] >= thresholds["candidate_target_r1"],
        "reciprocal_precision_improvement": (
            delta["reciprocal_exact_precision"]
            >= thresholds["reciprocal_exact_precision"]
        ),
        "coverage_preserved": (
            delta["reciprocal_exact_coverage"]
            >= thresholds["reciprocal_exact_coverage"]
        ),
        "stable_across_images": result["delta_std"]["candidate_target_r1"] <= 0.03,
    }
    return {
        "thresholds": thresholds,
        "maximum_r1_delta_std": 0.03,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranker",
        default=str(workspace / "artifacts/candidate_rank/rank_v2w64_best.pt"),
    )
    parser.add_argument(
        "--affinity-ckpt",
        default=str(workspace / "artifacts/macro_affinity/affinity_r1_1200_best.pt"),
    )
    parser.add_argument(
        "--affinity-ckpt2",
        default=str(workspace / "artifacts/macro_affinity/affinity_r3_1000_best.pt"),
    )
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--eval-rows", type=int, default=192)
    parser.add_argument("--max-pseudo-rows", type=int, default=192)
    parser.add_argument(
        "--pseudo-probe-rows",
        type=int,
        default=768,
        help="uniform label-free row probe; >=2304 enables 2x2-loop discovery",
    )
    parser.add_argument("--confidence-quantile", type=float, default=0.85)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--pair-batch", type=int, default=4096)
    parser.add_argument("--train-pair-batch", type=int, default=1024)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--photometric", type=float, default=0.04)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--trust-weight", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--adapter-mode",
        choices=("norm", "head"),
        default="norm",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("E:/pazzle_work/gates/test_time_adaptation_gate.json"),
    )
    args = parser.parse_args()
    positive = (
        "images",
        "candidate_k",
        "eval_rows",
        "max_pseudo_rows",
        "pseudo_probe_rows",
        "steps",
        "pair_batch",
        "train_pair_batch",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error(f"{', '.join(positive)} must be positive")
    if (
        not 0.0 <= args.confidence_quantile <= 1.0
        or args.lr <= 0.0
        or args.temperature <= 0.0
    ):
        parser.error("invalid confidence quantile or learning rate")
    return args


def _smoke(device: torch.device) -> dict[str, Any]:
    # A tiny exact loop graph validates label-free selection without loading
    # any image or permutation tensor.
    torch.manual_seed(123)
    model = CandidateSeamRanker(width=8, dropout=0.0).to(device).eval()
    tiles = torch.rand(1, NFRAG, 3, 20, 20, device=device)
    candidates = torch.stack(
        (
            torch.arange(NFRAG, device=device),
            torch.remainder(torch.arange(NFRAG, device=device) + 1, NFRAG),
        ),
        dim=-1,
    ).unsqueeze(0)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    rows = _all_rows(candidates, valid)
    if rows.count != NFRAG * NUM_DIRECTIONS:
        raise AssertionError("all-row smoke failed")
    augmented = _mild_augment(
        tiles,
        torch.Generator(device=device).manual_seed(7),
        noise_std=0.01,
        photometric=0.02,
    )
    if augmented.shape != tiles.shape or not torch.isfinite(augmented).all():
        raise AssertionError("augmentation smoke failed")
    parameters = _adapter_parameters(copy.deepcopy(model), "norm")
    return {
        "all_rows": rows.count,
        "adapter_parameters": sum(value.numel() for value in parameters),
        "directions": {"up": UP, "down": DOWN, "left": LEFT, "right": RIGHT},
    }


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.smoke:
        print(f"[test-time-adaptation smoke] {_smoke(device)}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    base_model = _load_ranker(args.ranker, device)
    affinity, _, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity_secondary = None
    if args.affinity_ckpt2:
        affinity_secondary, _, _ = load_frozen_affinity(
            args.affinity_ckpt2, device
        )
    _, validation_names = train_val_split()
    dataset = CanvasDataset(
        validation_names[: args.images],
        real_prob=0.0,
        seed=args.seed + 50_000,
    )

    rows_result: list[dict[str, Any]] = []
    metric_keys = (
        "candidate_target_r1",
        "candidate_target_r5",
        "candidate_target_mrr",
        "reciprocal_exact_precision",
        "reciprocal_exact_coverage",
    )
    for image_index in range(args.images):
        sample = dataset[image_index]
        tiles = sample["tiles"].unsqueeze(0).to(device)
        perm = sample["perm"].unsqueeze(0).to(device).long()
        candidates, valid = mine_affinity_candidates(
            affinity,
            tiles,
            candidate_k=args.candidate_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        exact_targets, exists = neighbor_targets(perm)
        exact_slots, available = candidate_target_slots(
            candidates, valid, exact_targets, exists
        )
        eval_rows = select_listwise_rows(
            exact_targets,
            exact_slots,
            available,
            rows_per_image=args.eval_rows,
            random_sample=False,
        )
        base_metrics = _paired_metrics(
            base_model,
            tiles,
            candidates,
            valid,
            eval_rows,
            pair_batch=args.pair_batch,
        )
        pseudo = build_pseudo_rows(
            base_model,
            tiles,
            candidates,
            valid,
            pair_batch=args.pair_batch,
            confidence_quantile=args.confidence_quantile,
            max_rows=args.max_pseudo_rows,
            probe_rows=args.pseudo_probe_rows,
        )
        adapted, adaptation = adapt_one_puzzle(
            base_model,
            tiles,
            candidates,
            valid,
            pseudo,
            steps=args.steps,
            lr=args.lr,
            pair_batch=args.train_pair_batch,
            seed=args.seed + image_index * 1009,
            noise_std=args.noise_std,
            photometric=args.photometric,
            distill_weight=args.distill_weight,
            trust_weight=args.trust_weight,
            temperature=args.temperature,
            adapter_mode=args.adapter_mode,
        )
        adapted_metrics = _paired_metrics(
            adapted,
            tiles,
            candidates,
            valid,
            eval_rows,
            pair_batch=args.pair_batch,
        )
        delta = {
            key: adapted_metrics[key] - base_metrics[key] for key in metric_keys
        }
        pseudo_diag = _pseudo_diagnostics(pseudo, exact_targets)
        row = {
            "image": validation_names[image_index],
            "base": base_metrics,
            "adapted": adapted_metrics,
            "delta": delta,
            "pseudo": pseudo_diag,
            "adaptation": adaptation,
        }
        rows_result.append(row)
        print(
            f"{row['image']}: pseudo={int(pseudo_diag['pseudo_rows'])} "
            f"pseudo_precision={pseudo_diag['pseudo_exact_precision_diagnostic']:.3f} "
            f"R1={base_metrics['candidate_target_r1']:.3f}->{adapted_metrics['candidate_target_r1']:.3f} "
            f"recipP={base_metrics['reciprocal_exact_precision']:.3f}->"
            f"{adapted_metrics['reciprocal_exact_precision']:.3f}",
            flush=True,
        )

    base_mean = {
        key: float(np.mean([row["base"][key] for row in rows_result]))
        for key in metric_keys
    }
    adapted_mean = {
        key: float(np.mean([row["adapted"][key] for row in rows_result]))
        for key in metric_keys
    }
    delta_mean = {
        key: adapted_mean[key] - base_mean[key] for key in metric_keys
    }
    delta_std = {
        key: float(np.std([row["delta"][key] for row in rows_result]))
        for key in metric_keys
    }
    result: dict[str, Any] = {
        "experiment": "label_free_per_puzzle_test_time_adaptation",
        "base_mean": base_mean,
        "adapted_mean": adapted_mean,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "pseudo_mean": {
            key: float(np.mean([row["pseudo"][key] for row in rows_result]))
            for key in (
                "pseudo_rows",
                "pseudo_exact_precision_diagnostic",
                "mutual_count",
                "loop_edge_count",
            )
        },
        "images": rows_result,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    result["gate"] = gate_result(result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[mean] R1={base_mean['candidate_target_r1']:.4f}->"
        f"{adapted_mean['candidate_target_r1']:.4f} "
        f"delta={delta_mean['candidate_target_r1']:+.4f} "
        f"recipP_delta={delta_mean['reciprocal_exact_precision']:+.4f}",
        flush=True,
    )
    print(f"=== TTA gate {'PASSED' if result['gate']['pass'] else 'FAILED'} ===")
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    print(f"report saved to {args.report}")


if __name__ == "__main__":
    main()
