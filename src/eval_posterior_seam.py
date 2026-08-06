"""Gate posterior seam marginalization on frozen held-out candidate rows."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from candidate_rank import (
    _orient_to_canonical,
    candidate_target_slots,
    finalize_rank_metrics,
    neighbor_targets,
    rank_metric_sums,
    score_candidate_rows,
    select_listwise_rows,
)
from canvas_data import CanvasDataset
from config import SEED, WORK_ROOT
from eval_test_time_adaptation import _load_ranker
from imgio import train_val_split
from match_preprocess import load_match_denoiser
from posterior_edge import PosteriorEdgeRestorer
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


def load_posterior(path: Path, device: torch.device) -> PosteriorEdgeRestorer:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PosteriorEdgeRestorer(**payload["model_kwargs"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def restore_hypotheses(
    posterior: PosteriorEdgeRestorer,
    denoiser: torch.nn.Module,
    tiles: Tensor,
    *,
    hypotheses: int,
    seed: int,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Return deterministic mean and K posterior bags, all shape-compatible."""
    means: list[Tensor] = []
    samples: list[Tensor] = []
    generator = torch.Generator(device=tiles.device).manual_seed(seed)
    for start in range(0, tiles.shape[1], batch_size):
        dirty = tiles[0, start : start + batch_size]
        mean = denoiser(dirty).float()
        means.append(mean)
        samples.append(
            posterior.sample(
                dirty,
                mean,
                hypotheses=hypotheses,
                generator=generator,
            )
        )
    deterministic = torch.cat(means, dim=0).unsqueeze(0)
    posterior_bags = torch.cat(samples, dim=1).unsqueeze(1)
    return deterministic, posterior_bags


def calibrated_metrics(
    scores: Tensor,
    target_slots: Tensor,
    valid_rows: Tensor,
) -> dict[str, float]:
    """Ranking plus fixed-temperature multiclass Brier/NLL."""
    masked = scores.float().masked_fill(~valid_rows, -1.0e4)
    probabilities = F.softmax(masked, dim=-1)
    labels = F.one_hot(target_slots.long(), num_classes=scores.shape[-1]).float()
    brier = (probabilities - labels).square().sum(dim=-1).mean()
    nll = F.cross_entropy(masked, target_slots.long())
    entropy = -(probabilities * probabilities.clamp_min(1.0e-9).log()).sum(dim=-1).mean()
    return {
        **finalize_rank_metrics(rank_metric_sums(masked, target_slots)),
        "brier": float(brier),
        "nll": float(nll),
        "entropy": float(entropy),
    }


@torch.inference_mode()
def score_bag(
    ranker: torch.nn.Module,
    bag: Tensor,
    candidates: Tensor,
    valid: Tensor,
    rows,
    *,
    pair_batch: int,
) -> Tensor:
    return score_candidate_rows(
        ranker,
        bag,
        candidates,
        valid,
        rows,
        pair_batch=pair_batch,
    ).float()


@torch.inference_mode()
def posterior_overlap_scores(
    samples: Tensor,
    candidates: Tensor,
    valid: Tensor,
    rows,
    *,
    band: int = 2,
    variance_floor: float = 0.02,
) -> tuple[Tensor, Tensor]:
    """Analytic expected seam error and Gaussian overlap from K clean hypotheses."""
    if samples.ndim != 6 or samples.shape[1] != 1:
        raise ValueError("samples must have shape (K,1,N,3,H,W)")
    row_candidates = candidates[rows.image_ids, rows.anchors]
    row_valid = valid[rows.image_ids, rows.anchors]
    width = row_candidates.shape[-1]
    row_ids, slots = torch.nonzero(row_valid, as_tuple=True)
    sources = rows.anchors[row_ids]
    targets = row_candidates[row_ids, slots]
    directions = rows.directions[row_ids]
    count = sources.numel()
    hypotheses = samples.shape[0]
    source_tiles = samples[:, 0, sources].reshape(-1, 3, 20, 20)
    target_tiles = samples[:, 0, targets].reshape(-1, 3, 20, 20)
    repeated_directions = directions.repeat(hypotheses)
    source_oriented = _orient_to_canonical(source_tiles, repeated_directions).reshape(
        hypotheses, count, 3, 20, 20
    )
    target_oriented = _orient_to_canonical(target_tiles, repeated_directions).reshape(
        hypotheses, count, 3, 20, 20
    )
    source_edge = source_oriented[..., -band:].flip(-1)
    target_edge = target_oriented[..., :band]
    source_mean = source_edge.mean(dim=0)
    target_mean = target_edge.mean(dim=0)
    source_var = source_edge.var(dim=0, unbiased=False)
    target_var = target_edge.var(dim=0, unbiased=False)
    variance = source_var + target_var
    difference = (source_mean - target_mean).square()
    expected = -(difference + variance).mean(dim=(-1, -2, -3))
    total_variance = variance + variance_floor**2
    gaussian = -(
        difference / total_variance + total_variance.log()
    ).mean(dim=(-1, -2, -3))
    expected_dense = samples.new_full((rows.count, width), -torch.inf)
    gaussian_dense = samples.new_full((rows.count, width), -torch.inf)
    expected_dense[row_ids, slots] = expected
    gaussian_dense[row_ids, slots] = gaussian
    return expected_dense, gaussian_dense


def row_standardize(scores: Tensor, valid: Tensor) -> Tensor:
    finite = scores.masked_fill(~valid, 0.0)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = finite.sum(dim=-1, keepdim=True) / count
    variance = (
        (finite - mean).square().masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True)
        / count
    )
    return ((scores - mean) / variance.add(1.0e-6).sqrt()).masked_fill(~valid, -torch.inf)


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def main() -> None:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--posterior",
        type=Path,
        default=Path(WORK_ROOT) / "posterior_edge" / "posterior_edge_best.pt",
    )
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
    parser.add_argument("--hypotheses", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--eval-rows", type=int, default=192)
    parser.add_argument("--pair-batch", type=int, default=4096)
    parser.add_argument("--restore-batch", type=int, default=576)
    parser.add_argument("--marginal-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "posterior_seam_gate.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ranker = _load_ranker(args.ranker, device)
    denoiser, denoiser_payload = load_match_denoiser("matchden", device=str(device))
    if denoiser is None:
        raise FileNotFoundError("matchden checkpoint is required")
    posterior = load_posterior(args.posterior, device)
    affinity, _, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity2, _, _ = load_frozen_affinity(args.affinity_ckpt2, device)
    _, validation_names = train_val_split()
    dataset = CanvasDataset(
        validation_names[: args.images],
        real_prob=0.0,
        seed=args.seed + 110_000,
    )
    modes = (
        "raw",
        "deterministic",
        "posterior_k1",
        "posterior_k4",
        "raw_plus_half_posterior_delta",
        "raw_plus_quarter_expected_seam",
        "raw_plus_quarter_gaussian_overlap",
    )
    per_mode: dict[str, list[dict[str, float]]] = {mode: [] for mode in modes}
    image_rows: list[dict[str, object]] = []
    for image_index in range(args.images):
        sample = dataset[image_index]
        tiles = sample["tiles"].unsqueeze(0).to(device)
        permutation = sample["perm"].unsqueeze(0).to(device).long()
        candidates, valid = mine_affinity_candidates(
            affinity,
            tiles,
            candidate_k=args.candidate_k,
            device=device,
            affinity_secondary=affinity2,
        )
        exact_targets, exists = neighbor_targets(permutation)
        exact_slots, available = candidate_target_slots(
            candidates, valid, exact_targets, exists
        )
        rows = select_listwise_rows(
            exact_targets,
            exact_slots,
            available,
            rows_per_image=args.eval_rows,
            random_sample=False,
        )
        row_valid = valid[rows.image_ids, rows.anchors]
        deterministic, samples = restore_hypotheses(
            posterior,
            denoiser,
            tiles,
            hypotheses=args.hypotheses,
            seed=args.seed + image_index * 1009,
            batch_size=args.restore_batch,
        )
        raw_scores = score_bag(
            ranker, tiles, candidates, valid, rows, pair_batch=args.pair_batch
        )
        deterministic_scores = score_bag(
            ranker, deterministic, candidates, valid, rows, pair_batch=args.pair_batch
        )
        hypothesis_scores = torch.stack(
            [
                score_bag(
                    ranker,
                    samples[index],
                    candidates,
                    valid,
                    rows,
                    pair_batch=args.pair_batch,
                )
                for index in range(args.hypotheses)
            ]
        )
        temperature = args.marginal_temperature
        marginal_scores = temperature * (
            torch.logsumexp(hypothesis_scores / temperature, dim=0)
            - math.log(args.hypotheses)
        )
        expected_seam, gaussian_overlap = posterior_overlap_scores(
            samples, candidates, valid, rows
        )
        raw_z = row_standardize(raw_scores, row_valid)
        score_modes = {
            "raw": raw_scores,
            "deterministic": deterministic_scores,
            "posterior_k1": hypothesis_scores[0],
            "posterior_k4": marginal_scores,
            "raw_plus_half_posterior_delta": (
                raw_scores + 0.5 * (marginal_scores - deterministic_scores)
            ),
            "raw_plus_quarter_expected_seam": (
                raw_z + 0.25 * row_standardize(expected_seam, row_valid)
            ),
            "raw_plus_quarter_gaussian_overlap": (
                raw_z + 0.25 * row_standardize(gaussian_overlap, row_valid)
            ),
        }
        row_result: dict[str, object] = {"image": image_index, "rows": rows.count}
        for mode, scores in score_modes.items():
            metrics = calibrated_metrics(scores, rows.target_slots, row_valid)
            per_mode[mode].append(metrics)
            row_result[mode] = metrics
        image_rows.append(row_result)
        print(json.dumps(row_result), flush=True)

    summary = {mode: average(per_mode[mode]) for mode in modes}
    baseline = summary["deterministic"]
    posterior_metrics = summary["posterior_k4"]
    deltas = {
        "candidate_target_r1": (
            posterior_metrics["candidate_target_r1"]
            - baseline["candidate_target_r1"]
        ),
        "candidate_target_r5": (
            posterior_metrics["candidate_target_r5"]
            - baseline["candidate_target_r5"]
        ),
        "brier_relative_improvement": (
            (baseline["brier"] - posterior_metrics["brier"])
            / max(1.0e-9, baseline["brier"])
        ),
        "nll": posterior_metrics["nll"] - baseline["nll"],
    }
    hybrid = summary["raw_plus_half_posterior_delta"]
    hybrid_vs_raw = {
        "candidate_target_r1": (
            hybrid["candidate_target_r1"] - summary["raw"]["candidate_target_r1"]
        ),
        "candidate_target_r5": (
            hybrid["candidate_target_r5"] - summary["raw"]["candidate_target_r5"]
        ),
        "brier_relative_improvement": (
            (summary["raw"]["brier"] - hybrid["brier"])
            / max(1.0e-9, summary["raw"]["brier"])
        ),
        "nll": hybrid["nll"] - summary["raw"]["nll"],
    }
    analytic_diagnostics = {}
    for mode in ("raw_plus_quarter_expected_seam", "raw_plus_quarter_gaussian_overlap"):
        value = summary[mode]
        analytic_diagnostics[mode] = {
            "candidate_target_r1": value["candidate_target_r1"] - summary["raw"]["candidate_target_r1"],
            "candidate_target_r5": value["candidate_target_r5"] - summary["raw"]["candidate_target_r5"],
            "brier_relative_improvement": (
                (summary["raw"]["brier"] - value["brier"])
                / max(1.0e-9, summary["raw"]["brier"])
            ),
            "nll": value["nll"] - summary["raw"]["nll"],
        }
    thresholds = {
        "candidate_target_r1": 0.05,
        "brier_relative_improvement": 0.10,
        "nll_max_delta": 0.0,
    }
    checks = {
        "r1": deltas["candidate_target_r1"] >= thresholds["candidate_target_r1"],
        "brier": (
            deltas["brier_relative_improvement"]
            >= thresholds["brier_relative_improvement"]
        ),
        "nll": deltas["nll"] <= thresholds["nll_max_delta"],
    }
    report = {
        "experiment": "posterior_seam_marginalization",
        "status": "pass" if all(checks.values()) else "fail",
        "summary": summary,
        "posterior_k4_vs_deterministic": deltas,
        "raw_plus_half_posterior_delta_vs_raw": hybrid_vs_raw,
        "analytic_posterior_vs_raw": analytic_diagnostics,
        "thresholds": thresholds,
        "checks": checks,
        "images": args.images,
        "hypotheses": args.hypotheses,
        "candidate_k_per_affinity": args.candidate_k,
        "denoiser_step": denoiser_payload.get("step"),
        "posterior_checkpoint": str(args.posterior),
        "image_rows": image_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate": report, "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
