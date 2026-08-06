"""Train/gate a scene-conditioned calibrator for frozen top edge predictions."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from candidate_rank import (
    ListwiseRows,
    NUM_DIRECTIONS,
    inverse_directions,
    neighbor_targets,
    score_candidate_rows,
)
from canvas_data import CanvasDataset
from config import NFRAG, SEED, WORK_ROOT
from edge_confidence import (
    EdgeConfidenceMLP,
    choose_precision_threshold,
    fit_standardizer,
    ranking_diagnostics,
    smoke_test,
    standardize,
    threshold_metrics,
)
from eval_test_time_adaptation import _all_rows, _load_ranker
from imgio import train_val_split
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


@dataclass
class FeatureRows:
    features: np.ndarray
    labels: np.ndarray
    image_ids: np.ndarray
    margins: np.ndarray
    feature_names: list[str]


def _uniform_rows(candidates: torch.Tensor, valid: torch.Tensor, count: int) -> ListwiseRows:
    rows = _all_rows(candidates, valid)
    if count >= rows.count:
        return rows
    indices = torch.arange(count, device=candidates.device) * rows.count // count
    return ListwiseRows(
        rows.image_ids[indices],
        rows.anchors[indices],
        rows.directions[indices],
        rows.target_slots[indices],
        rows.target_indices[indices],
    )


def _top_predictions(
    candidates: torch.Tensor,
    valid: torch.Tensor,
    rows: ListwiseRows,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row_valid = valid[rows.image_ids, rows.anchors]
    masked = scores.float().masked_fill(~row_valid, -1.0e4)
    values, slots = torch.topk(masked, k=2, dim=-1)
    predicted = candidates[rows.image_ids, rows.anchors, slots[:, 0]]
    probability = F.softmax(masked, dim=-1)
    top_probability = probability.gather(1, slots[:, :1]).squeeze(1)
    entropy = -(probability * probability.clamp_min(1.0e-9).log()).sum(dim=-1)
    return predicted, values[:, 0], values[:, 0] - values[:, 1], top_probability, entropy


def _tile_statistics(tiles: torch.Tensor) -> torch.Tensor:
    """RGB mean/std and grayscale gradient energy per tile."""
    value = tiles[0].float()
    mean = value.mean(dim=(-1, -2))
    std = value.std(dim=(-1, -2), unbiased=False)
    gray = (
        value[:, 0] * 0.299 + value[:, 1] * 0.587 + value[:, 2] * 0.114
    )
    gradient = 0.5 * (
        (gray[:, :, 1:] - gray[:, :, :-1]).abs().mean(dim=(-1, -2))
        + (gray[:, 1:, :] - gray[:, :-1, :]).abs().mean(dim=(-1, -2))
    )
    return torch.cat((mean, std, gradient[:, None]), dim=-1)


@torch.inference_mode()
def collect_one_image(
    *,
    image_index: int,
    sample: dict[str, torch.Tensor],
    ranker: torch.nn.Module,
    affinity: torch.nn.Module,
    affinity2: torch.nn.Module,
    candidate_k: int,
    rows_per_image: int,
    pair_batch: int,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    tiles = sample["tiles"].unsqueeze(0).to(device)
    permutation = sample["perm"].unsqueeze(0).long().to(device)
    candidates1, valid1 = mine_affinity_candidates(
        affinity, tiles, candidate_k=candidate_k, device=device
    )
    candidates2, valid2 = mine_affinity_candidates(
        affinity2, tiles, candidate_k=candidate_k, device=device
    )
    candidates, valid = mine_affinity_candidates(
        affinity,
        tiles,
        candidate_k=candidate_k,
        device=device,
        affinity_secondary=affinity2,
    )
    rows = _uniform_rows(candidates, valid, rows_per_image)
    scores = score_candidate_rows(
        ranker, tiles, candidates, valid, rows, pair_batch=pair_batch
    )
    predicted, top_score, margin, top_probability, entropy = _top_predictions(
        candidates, valid, rows, scores
    )

    reverse_candidates = candidates[0, predicted]
    reverse_valid = valid[0, predicted]
    anchor_match = reverse_valid & reverse_candidates.eq(rows.anchors[:, None])
    reverse_slots = anchor_match.long().argmax(dim=-1)
    reverse_rows = ListwiseRows(
        torch.zeros_like(rows.anchors),
        predicted,
        inverse_directions(rows.directions),
        reverse_slots,
        rows.anchors,
    )
    reverse_scores = score_candidate_rows(
        ranker, tiles, candidates, valid, reverse_rows, pair_batch=pair_batch
    )
    reverse_predicted, _, reverse_margin, reverse_probability, reverse_entropy = _top_predictions(
        candidates, valid, reverse_rows, reverse_scores
    )
    reciprocal = reverse_predicted.eq(rows.anchors).float()

    exact_targets, exists = neighbor_targets(permutation)
    truth = exact_targets[0, rows.anchors, rows.directions]
    labels = (exists[0, rows.anchors, rows.directions] & predicted.eq(truth)).float()

    primary_match = valid1[0, rows.anchors] & candidates1[0, rows.anchors].eq(predicted[:, None])
    secondary_match = valid2[0, rows.anchors] & candidates2[0, rows.anchors].eq(predicted[:, None])
    in_primary = primary_match.any(dim=-1).float()
    in_secondary = secondary_match.any(dim=-1).float()
    primary_rank = torch.where(
        in_primary.bool(),
        primary_match.long().argmax(dim=-1).float() + 1.0,
        torch.full_like(in_primary, candidate_k + 1.0),
    ) / candidate_k
    secondary_rank = torch.where(
        in_secondary.bool(),
        secondary_match.long().argmax(dim=-1).float() + 1.0,
        torch.full_like(in_secondary, candidate_k + 1.0),
    ) / candidate_k

    stats = _tile_statistics(tiles)
    source_stats = stats[rows.anchors]
    target_stats = stats[predicted]
    global_mean = stats.mean(dim=0)
    global_std = stats.std(dim=0, unbiased=False)
    row_valid_count = valid[rows.image_ids, rows.anchors].sum(dim=-1).float()
    directions = F.one_hot(rows.directions, num_classes=NUM_DIRECTIONS).float()

    base = torch.cat(
        (
            top_score[:, None],
            margin[:, None],
            top_probability[:, None],
            entropy[:, None],
            row_valid_count[:, None] / (2 * candidate_k),
            reciprocal[:, None],
            reverse_margin[:, None],
            reverse_probability[:, None],
            reverse_entropy[:, None],
            in_primary[:, None],
            in_secondary[:, None],
            primary_rank[:, None],
            secondary_rank[:, None],
            directions,
            source_stats,
            target_stats,
            (source_stats - target_stats).abs(),
            global_mean[None].expand(rows.count, -1),
            global_std[None].expand(rows.count, -1),
        ),
        dim=-1,
    )
    # Per-scene ranks/scales capture whether "large" margins are genuinely
    # exceptional for this puzzle rather than merely shifted in raw units.
    scene_columns: list[torch.Tensor] = []
    for value in (top_score, margin, entropy, reverse_margin):
        centered = (value - value.mean()) / value.std(unbiased=False).clamp_min(1.0e-4)
        order = torch.argsort(torch.argsort(value)).float() / max(1, value.numel() - 1)
        scene_columns.extend((centered[:, None], order[:, None]))
    features = torch.cat((base, *scene_columns), dim=-1)
    names = (
        ["top_score", "margin", "top_probability", "entropy", "candidate_fraction",
         "reciprocal", "reverse_margin", "reverse_probability", "reverse_entropy",
         "in_primary", "in_secondary", "primary_rank", "secondary_rank"]
        + [f"direction_{index}" for index in range(4)]
        + [f"source_stat_{index}" for index in range(7)]
        + [f"target_stat_{index}" for index in range(7)]
        + [f"pair_abs_stat_{index}" for index in range(7)]
        + [f"scene_mean_{index}" for index in range(7)]
        + [f"scene_std_{index}" for index in range(7)]
        + [
            "top_score_z", "top_score_percentile",
            "margin_z", "margin_percentile",
            "entropy_z", "entropy_percentile",
            "reverse_margin_z", "reverse_margin_percentile",
        ]
    )
    if features.shape[1] != len(names):
        raise AssertionError("feature name/width mismatch")
    return (
        features.cpu().numpy().astype(np.float32),
        labels.cpu().numpy().astype(np.float32),
        margin.cpu().numpy().astype(np.float32),
        names,
        rows.anchors.cpu().numpy().astype(np.int64),
        rows.directions.cpu().numpy().astype(np.int64),
        predicted.cpu().numpy().astype(np.int64),
        candidates[0].cpu().numpy().astype(np.int64),
        scores.cpu().numpy().astype(np.float32),
    )


def collect_split(
    *,
    names: list[str],
    split_offset: int,
    dataset_seed: int,
    ranker: torch.nn.Module,
    affinity: torch.nn.Module,
    affinity2: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> FeatureRows:
    dataset = CanvasDataset(names, real_prob=0.0, seed=dataset_seed)
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    margin_parts: list[np.ndarray] = []
    image_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    for index in range(len(names)):
        features, labels, margins, feature_names, _, _, _, _, _ = collect_one_image(
            image_index=split_offset + index,
            sample=dataset[index],
            ranker=ranker,
            affinity=affinity,
            affinity2=affinity2,
            candidate_k=args.candidate_k,
            rows_per_image=args.rows_per_image,
            pair_batch=args.pair_batch,
            device=device,
        )
        feature_parts.append(features)
        label_parts.append(labels)
        margin_parts.append(margins)
        image_parts.append(
            np.full(len(labels), split_offset + index, dtype=np.int64)
        )
        print(
            json.dumps(
                {
                    "collect_image": split_offset + index,
                    "rows": len(labels),
                    "top_edge_accuracy": float(labels.mean()),
                    "positive_reciprocal_hint": float(features[:, feature_names.index("reciprocal")].mean()),
                }
            ),
            flush=True,
        )
    return FeatureRows(
        np.concatenate(feature_parts),
        np.concatenate(label_parts),
        np.concatenate(image_parts),
        np.concatenate(margin_parts),
        feature_names,
    )


def train_mlp(
    fit: FeatureRows,
    calibration: FeatureRows,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[EdgeConfidenceMLP, np.ndarray, np.ndarray]:
    mean, scale = fit_standardizer(fit.features)
    x = torch.from_numpy(standardize(fit.features, mean, scale)).to(device)
    y = torch.from_numpy(fit.labels).to(device)
    model = EdgeConfidenceMLP(x.shape[1], hidden=hidden).to(device)
    positives = float(y.sum())
    pos_weight = torch.tensor(
        max(1.0, (len(y) - positives) / max(1.0, positives)), device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = min(1024, len(y))
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y), device=device, generator=generator)
        total = 0.0
        for start in range(0, len(y), batch_size):
            index = order[start : start + batch_size]
            logits = model(x[index])
            loss = F.binary_cross_entropy_with_logits(
                logits, y[index], pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                cal_x = torch.from_numpy(
                    standardize(calibration.features, mean, scale)
                ).to(device)
                cal_loss = F.binary_cross_entropy_with_logits(
                    model(cal_x),
                    torch.from_numpy(calibration.labels).to(device),
                )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "fit_loss": total / len(y),
                        "calibration_loss": float(cal_loss),
                    }
                ),
                flush=True,
            )
    return model.eval(), mean, scale


@torch.inference_mode()
def predict(
    model: EdgeConfidenceMLP,
    rows: FeatureRows,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(standardize(rows.features, mean, scale)).to(device)
    return torch.sigmoid(model(x)).cpu().numpy()


def main() -> None:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fit-images", type=int, default=12)
    parser.add_argument("--calibration-images", type=int, default=4)
    parser.add_argument("--heldout-images", type=int, default=6)
    parser.add_argument("--rows-per-image", type=int, default=256)
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--pair-batch", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--minimum-calibration-edges", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "best.pt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "edge_confidence_gate.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.smoke:
        print(json.dumps(smoke_test(device), indent=2))
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ranker = _load_ranker(args.ranker, device)
    affinity, _, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity2, _, _ = load_frozen_affinity(args.affinity_ckpt2, device)
    _, validation_names = train_val_split()
    needed = args.fit_images + args.calibration_images + args.heldout_images
    if needed > len(validation_names):
        raise ValueError("requested image split exceeds validation names")
    fit_names = validation_names[: args.fit_images]
    cal_start = args.fit_images
    held_start = cal_start + args.calibration_images
    calibration_names = validation_names[cal_start:held_start]
    heldout_names = validation_names[held_start:needed]
    fit = collect_split(
        names=fit_names,
        split_offset=0,
        dataset_seed=args.seed + 200_000,
        ranker=ranker,
        affinity=affinity,
        affinity2=affinity2,
        args=args,
        device=device,
    )
    calibration = collect_split(
        names=calibration_names,
        split_offset=args.fit_images,
        dataset_seed=args.seed + 300_000,
        ranker=ranker,
        affinity=affinity,
        affinity2=affinity2,
        args=args,
        device=device,
    )
    heldout = collect_split(
        names=heldout_names,
        split_offset=held_start,
        dataset_seed=args.seed + 400_000,
        ranker=ranker,
        affinity=affinity,
        affinity2=affinity2,
        args=args,
        device=device,
    )
    model, mean, scale = train_mlp(
        fit,
        calibration,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=device,
    )
    cal_probability = predict(model, calibration, mean, scale, device)
    held_probability = predict(model, heldout, mean, scale, device)
    selection = choose_precision_threshold(
        cal_probability,
        calibration.labels,
        target_precision=args.target_precision,
        minimum_edges=args.minimum_calibration_edges,
    )
    model_metrics = threshold_metrics(
        held_probability,
        heldout.labels,
        heldout.image_ids,
        selection["threshold"],
    )
    margin_selection = choose_precision_threshold(
        calibration.margins,
        calibration.labels,
        target_precision=args.target_precision,
        minimum_edges=args.minimum_calibration_edges,
    )
    margin_metrics = threshold_metrics(
        heldout.margins,
        heldout.labels,
        heldout.image_ids,
        margin_selection["threshold"],
    )
    reciprocal_index = heldout.feature_names.index("reciprocal")
    both_affinities = (
        heldout.features[:, heldout.feature_names.index("in_primary")].astype(bool)
        & heldout.features[:, heldout.feature_names.index("in_secondary")].astype(bool)
    )
    reciprocal = heldout.features[:, reciprocal_index].astype(bool)
    rule_metrics = {
        "reciprocal": threshold_metrics(
            reciprocal.astype(np.float32), heldout.labels, heldout.image_ids, 0.5
        ),
        "both_affinities": threshold_metrics(
            both_affinities.astype(np.float32), heldout.labels, heldout.image_ids, 0.5
        ),
        "reciprocal_and_both_affinities": threshold_metrics(
            (reciprocal & both_affinities).astype(np.float32),
            heldout.labels,
            heldout.image_ids,
            0.5,
        ),
    }
    thresholds = {
        "precision": 0.90,
        "acceptance_coverage": 0.15,
        "worst_image_precision": 0.80,
    }
    checks = {
        key: model_metrics[key] >= value for key, value in thresholds.items()
    }
    report: dict[str, Any] = {
        "experiment": "scene_conditioned_edge_confidence",
        "status": "pass" if all(checks.values()) else "fail",
        "model": model_metrics,
        "raw_margin_baseline": margin_metrics,
        "heldout_ranking_diagnostics": {
            "model": ranking_diagnostics(
                held_probability,
                heldout.labels,
                minimum_edges=args.minimum_calibration_edges,
            ),
            "raw_margin": ranking_diagnostics(
                heldout.margins,
                heldout.labels,
                minimum_edges=args.minimum_calibration_edges,
            ),
        },
        "heldout_label_free_rules": rule_metrics,
        "calibration_selection": selection,
        "margin_calibration_selection": margin_selection,
        "thresholds": thresholds,
        "checks": checks,
        "splits": {
            "fit_images": args.fit_images,
            "calibration_images": args.calibration_images,
            "heldout_images": args.heldout_images,
            "rows_per_image": args.rows_per_image,
        },
        "fit_positive_rate": float(fit.labels.mean()),
        "calibration_positive_rate": float(calibration.labels.mean()),
        "heldout_positive_rate": float(heldout.labels.mean()),
        "feature_names": fit.feature_names,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "features": model.features,
            "hidden": model.hidden,
            "dropout": model.dropout,
            "mean": mean,
            "scale": scale,
            "threshold": selection["threshold"],
            "feature_names": fit.feature_names,
            "report": report,
        },
        args.checkpoint,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate": report, "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
