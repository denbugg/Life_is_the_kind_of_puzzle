"""Evaluate seam compatibility *only* on affinity-mined puzzle candidates.

This is a deliberately narrow diagnostic for the hierarchical assembly branch.
The historical :class:`models.PairwiseNet` was trained as an ordered seam
ranker, but its all-pairs evaluation is too costly and its random-negative
validation does not answer the useful question here: does it separate true
cardinal neighbours from the hard, semantically similar candidates retained by
the learned proximity graph?

For every freshly corrupted, exactly labelled held-out synthetic puzzle this
program does the following:

1. builds the deduplicated union of top-64 neighbours from the r=1 and r=3
   frozen MacroAffinityNet checkpoints;
2. evaluates the old seam ensemble only for those candidate tile pairs;
3. scores all four physical cardinal seam hypotheses (U/D/L/R) for each such
   pair, never the 576 x 575 universe;
4. reports direct-neighbour and exact-direction precision/recall, reciprocal
   inverse checks, score-quantile threshold curves and fixed-top-k curves.

The PairwiseNet is an InfoNCE ranker, so an absolute raw logit is not a
calibrated probability.  The report therefore shows both ``pair_raw`` and
``pair_row_z`` (each orientation z-normalised within its source tile's frozen
candidate list).  The latter is a no-label, per-row calibration that puts the
horizontal and vertical hypotheses on comparable scales.

When a DirectPoseNet checkpoint is supplied, the evaluator also prints its
candidate-graph metrics and an explicitly labelled *heuristic* fusion:

    (1-w) * P_pose(direct and direction) + w * sigmoid(pair_row_z)

No threshold or weight is fitted against the held-out labels; use it only as a
ranking diagnostic, not as a claimed calibrated probability.

Examples
--------

    python src/eval_pair_affinity_fusion.py --smoke
    python src/eval_pair_affinity_fusion.py --n 1 --device cuda
    python src/eval_pair_affinity_fusion.py --n 4 --pair-batch 4096
    python src/eval_pair_affinity_fusion.py --n 2 --no-direct-pose

The default n=1 gate is intentional.  The seam ensemble is much slower than
the affinity encoder, and a larger run is warranted only if the first fresh
held-out image shows a material improvement over the affinity-only candidate
prior.
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from canvas_data import CanvasDataset
from config import CKPT_DIR, FS, GRID, NFRAG, SEED
from direct_pose import DIRECT_CLASS_COUNT, NON_DIRECT_CLASS, hierarchical_probabilities, inverse_classes
from eval_direct_pose import load_direct_pose
from imgio import train_val_split
from models import PairwiseNet
from train_direct_pose import candidate_direct_labels, score_candidate_graph
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_AFFINITY_CKPT = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r1_1200_best.pt"
)
DEFAULT_AFFINITY_CKPT2 = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r3_1000_best.pt"
)
DEFAULT_DIRECT_POSE_CKPT = os.path.join(
    WORKSPACE, "artifacts", "direct_pose", "direct_union_2000_best.pt"
)

_DIRECTION_NAMES = ("U", "D", "L", "R")
DIRECT_EDGES_PER_BOARD = 4 * GRID * (GRID - 1)


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def _torch_load(path: str, device: torch.device) -> object:
    """Load a trusted local checkpoint on the requested inference device."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Torch versions before ``weights_only``.
        return torch.load(path, map_location=device)


def _is_tensor_state_dict(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) and isinstance(item, Tensor) for key, item in value.items())
    )


def _state_dict(payload: object) -> dict[str, Tensor]:
    if isinstance(payload, nn.Module):
        return dict(payload.state_dict())
    if _is_tensor_state_dict(payload):
        return dict(payload)
    if isinstance(payload, Mapping):
        for key in ("model", "model_state_dict", "state_dict", "network", "net"):
            candidate = payload.get(key)
            if isinstance(candidate, nn.Module):
                return dict(candidate.state_dict())
            if _is_tensor_state_dict(candidate):
                return dict(candidate)
    raise RuntimeError("checkpoint does not contain a recognizable PairwiseNet state dict")


def _strip_uniform_prefix(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    result = dict(state)
    for prefix in ("module.", "model.", "net."):
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix) :]: value for key, value in result.items()}
    return result


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    namespace = getattr(value, "__dict__", None)
    return namespace if isinstance(namespace, Mapping) else {}


def _pair_checkpoint_paths(raw_paths: str, tag: str, which: str) -> list[str]:
    """Resolve the same pair0/pair1 ensemble policy as ``pipeline.load_pair``.

    The evaluator keeps a device-aware local loader rather than importing the
    pipeline's CUDA-global loader, so its CPU smoke path remains valid too.
    """
    explicit = [item.strip() for item in raw_paths.split(",") if item.strip()]
    if explicit:
        paths = [os.path.abspath(item) for item in explicit]
    elif tag == "pair":
        paths = []
        for member in ("pair0", "pair1"):
            for filename in (f"{member}_{which}.pt", f"{member}_last.pt", f"{member}_best.pt"):
                path = os.path.join(CKPT_DIR, filename)
                if os.path.isfile(path):
                    paths.append(path)
                    break
        if not paths:
            paths = [os.path.join(CKPT_DIR, f"pair_{which}.pt")]
    else:
        paths = [os.path.join(CKPT_DIR, f"{tag}_{which}.pt")]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("PairwiseNet checkpoint(s) missing: " + ", ".join(missing))
    return paths


def load_pair_ensemble(paths: Sequence[str], device: torch.device) -> tuple[list[PairwiseNet], list[Mapping[str, Any]]]:
    """Load the old seam models, handling regular and DataParallel states."""
    models: list[PairwiseNet] = []
    metadata: list[Mapping[str, Any]] = []
    for path in paths:
        payload = _torch_load(path, device)
        model = PairwiseNet().to(device)
        state = _strip_uniform_prefix(_state_dict(payload))
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"PairwiseNet checkpoint architecture mismatch for {path}: "
                f"missing={incompatible.missing_keys[:6]}, unexpected={incompatible.unexpected_keys[:6]}"
            )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
        metadata.append(_as_mapping(payload))
    return models, metadata


def _parse_device(value: str | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return device


def _parse_floats(values: Sequence[str], *, low: float, high: float, name: str) -> tuple[float, ...]:
    output: list[float] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                number = float(item)
            except ValueError as exc:
                raise ValueError(f"invalid {name} value {item!r}") from exc
            if not low <= number <= high:
                raise ValueError(f"{name} must lie in [{low},{high}], got {number}")
            output.append(number)
    if not output:
        raise ValueError(f"at least one {name} is required")
    return tuple(sorted(set(output)))


def _parse_ints(values: Sequence[str], *, low: int, high: int, name: str) -> tuple[int, ...]:
    output: list[int] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                number = int(item)
            except ValueError as exc:
                raise ValueError(f"invalid {name} value {item!r}") from exc
            if not low <= number <= high:
                raise ValueError(f"{name} must lie in [{low},{high}], got {number}")
            output.append(number)
    if not output:
        raise ValueError(f"at least one {name} is required")
    return tuple(sorted(set(output)))


@dataclass
class CandidateScores:
    """One directed affinity graph with one U/D/L/R prediction per candidate."""

    candidates: Tensor  # (576,K)
    valid: Tensor  # (576,K), bool
    labels: Tensor  # (576,K), U/D/L/R/non-direct
    score: Tensor  # (576,K), higher means more likely to select as an edge
    direction: Tensor  # (576,K), U/D/L/R predicted orientation

    def __post_init__(self) -> None:
        if self.candidates.ndim != 2 or self.candidates.shape[0] != NFRAG:
            raise ValueError(f"candidates must have shape (576,K), got {tuple(self.candidates.shape)}")
        if self.valid.shape != self.candidates.shape or self.valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean matrix aligned with candidates")
        if self.labels.shape != self.candidates.shape:
            raise ValueError("labels must align with candidates")
        if self.score.shape != self.candidates.shape or self.direction.shape != self.candidates.shape:
            raise ValueError("score and direction must align with candidates")
        if torch.any(self.direction < 0) or torch.any(self.direction >= DIRECT_CLASS_COUNT):
            raise ValueError("direction must contain U/D/L/R class ids")


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _exact_board_direction_counts() -> dict[int, float]:
    return {direction: float(GRID * (GRID - 1)) for direction in range(DIRECT_CLASS_COUNT)}


def _reciprocal_counts(scores: CandidateScores, selected: Tensor) -> dict[str, float]:
    """Count selected inverse predictions using only mutually proposed pairs."""
    candidates, valid, labels = scores.candidates, scores.valid, scores.labels
    count, width = candidates.shape
    anchors = torch.arange(count, device=candidates.device).view(count, 1).expand(count, width)
    ranks = torch.full((count, count), -1, dtype=torch.long, device=candidates.device)
    rank_values = torch.arange(width, device=candidates.device).view(1, width).expand(count, width)
    ranks[anchors[valid], candidates.long()[valid]] = rank_values[valid]
    reverse_rank = ranks[candidates.long(), anchors]
    mutual = valid & reverse_rank.ge(0)
    if not bool(mutual.any()):
        return {
            "mutual_pairs": 0.0,
            "mutual_true_direct": 0.0,
            "predicted_reciprocal_inverse": 0.0,
            "correct_reciprocal_inverse": 0.0,
        }
    reverse_selected = selected[candidates.long(), reverse_rank.clamp_min(0)]
    reverse_labels = labels[candidates.long(), reverse_rank.clamp_min(0)]
    reverse_direction = scores.direction[candidates.long(), reverse_rank.clamp_min(0)]
    true_mutual_direct = (
        mutual
        & labels.ne(NON_DIRECT_CLASS)
        & reverse_labels.eq(inverse_classes(labels))
    )
    predicted_reciprocal = (
        mutual
        & selected
        & reverse_selected
        & reverse_direction.eq(inverse_classes(scores.direction))
    )
    correct = (
        predicted_reciprocal
        & true_mutual_direct
        & scores.direction.eq(labels)
        & reverse_direction.eq(reverse_labels)
    )
    return {
        "mutual_pairs": float(mutual.sum()),
        "mutual_true_direct": float(true_mutual_direct.sum()),
        "predicted_reciprocal_inverse": float(predicted_reciprocal.sum()),
        "correct_reciprocal_inverse": float(correct.sum()),
    }


def threshold_counts(scores: CandidateScores, selected: Tensor) -> dict[str, float]:
    """Return additive metrics for any boolean selected-candidate relation."""
    if selected.shape != scores.valid.shape or selected.dtype != torch.bool:
        raise ValueError("selected must be a boolean matrix aligned with candidates")
    valid, labels = scores.valid, scores.labels
    selected = selected & valid
    direct = valid & labels.ne(NON_DIRECT_CLASS)
    direct_hit = selected & direct
    exact = direct_hit & scores.direction.eq(labels)
    output: dict[str, float] = {
        "candidate_pairs": float(valid.sum()),
        "candidate_true_direct": float(direct.sum()),
        "selected": float(selected.sum()),
        "selected_true_direct": float(direct_hit.sum()),
        "selected_exact_direction": float(exact.sum()),
    }
    for direction, name in enumerate(_DIRECTION_NAMES):
        truth = valid & labels.eq(direction)
        predicted = selected & scores.direction.eq(direction)
        output[f"{name}_truth_candidate"] = float(truth.sum())
        output[f"{name}_predicted"] = float(predicted.sum())
        output[f"{name}_correct"] = float((predicted & labels.eq(direction)).sum())
    output.update(_reciprocal_counts(scores, selected))
    return output


def finalize_metrics(counts: Mapping[str, float], images: int) -> dict[str, float]:
    """Turn accumulated metric counts into candidate and end-to-end rates."""
    if images < 1:
        raise ValueError("images must be positive")
    candidate_direct = counts["candidate_true_direct"]
    selected = counts["selected"]
    all_true = float(images * DIRECT_EDGES_PER_BOARD)
    output = {
        "candidate_edges_per_tile": _ratio(counts["candidate_pairs"], float(images * NFRAG)),
        "candidate_direct_coverage_all_true": _ratio(candidate_direct, all_true),
        "selected_edges_per_tile": _ratio(selected, float(images * NFRAG)),
        "direct_edge_precision": _ratio(counts["selected_true_direct"], selected),
        "direct_edge_recall_candidate": _ratio(counts["selected_true_direct"], candidate_direct),
        "direct_edge_recall_all_true": _ratio(counts["selected_true_direct"], all_true),
        "exact_direction_precision": _ratio(counts["selected_exact_direction"], selected),
        "exact_direction_recall_candidate": _ratio(counts["selected_exact_direction"], candidate_direct),
        "exact_direction_recall_all_true": _ratio(counts["selected_exact_direction"], all_true),
        "reciprocal_inverse_precision": _ratio(
            counts["correct_reciprocal_inverse"], counts["predicted_reciprocal_inverse"]
        ),
        "reciprocal_inverse_coverage_mutual_direct": _ratio(
            counts["correct_reciprocal_inverse"], counts["mutual_true_direct"]
        ),
        "reciprocal_inverse_coverage_all_true": _ratio(
            counts["correct_reciprocal_inverse"], all_true
        ),
        "reciprocal_edges_per_tile": _ratio(
            counts["predicted_reciprocal_inverse"], float(images * NFRAG)
        ),
        "mutual_candidate_fraction": _ratio(counts["mutual_pairs"], counts["candidate_pairs"]),
    }
    for direction, name in enumerate(_DIRECTION_NAMES):
        output[f"{name}_precision"] = _ratio(counts[f"{name}_correct"], counts[f"{name}_predicted"])
        output[f"{name}_recall_candidate"] = _ratio(
            counts[f"{name}_correct"], counts[f"{name}_truth_candidate"]
        )
        output[f"{name}_recall_all_true"] = _ratio(
            counts[f"{name}_correct"], images * _exact_board_direction_counts()[direction]
        )
    return output


def _row_z(orientation_scores: Tensor, valid: Tensor) -> Tensor:
    """Calibrate each U/D/L/R seam score within one source candidate row.

    PairwiseNet's training objective compares candidates for one source seam.
    This preserves that intended relative scale while avoiding a global logit
    threshold that is meaningless across images or orientation heads.
    """
    if orientation_scores.ndim != 3 or orientation_scores.shape[:2] != valid.shape:
        raise ValueError("orientation_scores must have shape (576,K,4) aligned with valid")
    mask = valid.unsqueeze(-1)
    values = torch.where(mask, orientation_scores, torch.zeros_like(orientation_scores))
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
    mean = values.sum(dim=1, keepdim=True) / denom
    variance = ((values - mean).square() * mask).sum(dim=1, keepdim=True) / denom
    normalized = (orientation_scores - mean) / variance.add(1.0e-6).sqrt()
    return torch.where(mask, normalized, torch.full_like(normalized, -torch.inf))


@torch.inference_mode()
def score_pairwise_directions(
    models: Sequence[PairwiseNet],
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    *,
    pair_batch: int,
    device: torch.device,
) -> Tensor:
    """Return raw U/D/L/R seam logits for affinity-proposed directed pairs.

    A row ``i -> j`` requires four physical layouts:

    * U: vertical seam ``j | i`` after transposing tile axes;
    * D: vertical seam ``i | j`` after transposing tile axes;
    * L: horizontal seam ``j | i``;
    * R: horizontal seam ``i | j``.

    Thus every forward pass still involves only endpoint pairs proposed by the
    frozen graph.  No unproposed tile pair and no dense N^2 matrix is built.
    ``pair_batch`` is the maximum number of *raw PairwiseNet inputs* per
    chunk; each logical candidate contributes four inputs.
    """
    if not models:
        raise ValueError("at least one PairwiseNet is required")
    if tiles.ndim != 4 or tuple(tiles.shape[1:]) != (3, FS, FS):
        raise ValueError(f"tiles must have shape (576,3,{FS},{FS}), got {tuple(tiles.shape)}")
    if candidates.shape != valid.shape or candidates.ndim != 2:
        raise ValueError("candidates and valid must be aligned two-dimensional matrices")
    if pair_batch < 4:
        raise ValueError("pair_batch must be at least four raw seam inputs")
    count, width = candidates.shape
    anchors = torch.arange(count, device=device).view(count, 1).expand(count, width)
    sources = anchors[valid]
    targets = candidates.long()[valid]
    positions = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    if not sources.numel():
        raise RuntimeError("affinity candidate graph has no valid pairs")
    output = torch.full((count * width, DIRECT_CLASS_COUNT), -torch.inf, device=device)
    logical_batch = max(1, pair_batch // DIRECT_CLASS_COUNT)
    for start in range(0, sources.numel(), logical_batch):
        stop = min(start + logical_batch, sources.numel())
        source = sources[start:stop]
        target = targets[start:stop]
        left = tiles[source]
        right = tiles[target]
        # Original training uses a horizontal seam for right-neighbour labels
        # and a transposed horizontal seam for below-neighbour labels.
        left_vertical = left.transpose(-1, -2)
        right_vertical = right.transpose(-1, -2)
        layouts = torch.cat(
            (
                torch.cat((right_vertical, left_vertical), dim=-1),  # U
                torch.cat((left_vertical, right_vertical), dim=-1),  # D
                torch.cat((right, left), dim=-1),  # L
                torch.cat((left, right), dim=-1),  # R
            ),
            dim=0,
        )
        with _autocast(device):
            logits = sum(model(layouts).float() for model in models) / float(len(models))
        output[positions[start:stop]] = logits.reshape(DIRECT_CLASS_COUNT, stop - start).transpose(0, 1)
    return output.reshape(count, width, DIRECT_CLASS_COUNT)


def _candidate_scores_from_orientations(
    candidates: Tensor,
    valid: Tensor,
    labels: Tensor,
    orientations: Tensor,
) -> tuple[CandidateScores, CandidateScores, Tensor]:
    """Build raw and per-row calibrated seam prediction bundles."""
    if orientations.shape != (*candidates.shape, DIRECT_CLASS_COUNT):
        raise ValueError("orientations must have shape (576,K,4)")
    raw_score, raw_direction = orientations.max(dim=-1)
    normalized = _row_z(orientations, valid)
    z_score, z_direction = normalized.max(dim=-1)
    return (
        CandidateScores(candidates, valid, labels, raw_score, raw_direction),
        CandidateScores(candidates, valid, labels, z_score, z_direction),
        normalized,
    )


@torch.inference_mode()
def score_direct_pose_bundle(
    model: nn.Module,
    tiles: Tensor,
    candidates: Tensor,
    valid: Tensor,
    labels: Tensor,
    *,
    pair_batch: int,
    device: torch.device,
) -> tuple[CandidateScores, Tensor]:
    """Score the frozen candidate graph with a DirectPoseNet, if requested."""
    logits = score_candidate_graph(
        model, tiles.unsqueeze(0), candidates.unsqueeze(0), valid=valid.unsqueeze(0),
        pair_batch=pair_batch, device=device,
    )[0]
    decoded = hierarchical_probabilities(logits)
    direction_confidence = (
        decoded["direct_probability"].unsqueeze(-1)
        * decoded["conditional_direction_probabilities"]
    )
    score, direction = direction_confidence.max(dim=-1)
    return CandidateScores(candidates, valid, labels, score, direction), direction_confidence


def _topk_select(scores: CandidateScores, top_k: int) -> Tensor:
    """Choose at most ``top_k`` valid candidate relations per source tile."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    width = scores.valid.shape[1]
    selected_width = min(top_k, width)
    masked = torch.where(scores.valid, scores.score, torch.full_like(scores.score, -torch.inf))
    index = masked.topk(selected_width, dim=-1).indices
    selected = torch.zeros_like(scores.valid)
    selected.scatter_(1, index, True)
    return selected & scores.valid


def _quantile_threshold(scores: CandidateScores, quantile: float) -> float:
    values = scores.score[scores.valid]
    if not values.numel():
        return float("inf")
    return float(torch.quantile(values.float(), float(quantile)).item())


def _add_counts(total: defaultdict[str, float], values: Mapping[str, float]) -> None:
    for key, value in values.items():
        total[key] += float(value)


def _fmt(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "nan"


def print_metric_line(label: str, metrics: Mapping[str, float], *, detail: bool = True) -> None:
    """Print a compact, graph-relevant metric report in a stable format."""
    print(
        f"[{label}] selected/tile={_fmt(metrics['selected_edges_per_tile'])} "
        f"direct p={_fmt(metrics['direct_edge_precision'])} "
        f"r(cand/all)={_fmt(metrics['direct_edge_recall_candidate'])}/"
        f"{_fmt(metrics['direct_edge_recall_all_true'])} | "
        f"exact UDLR p={_fmt(metrics['exact_direction_precision'])} "
        f"r(cand/all)={_fmt(metrics['exact_direction_recall_candidate'])}/"
        f"{_fmt(metrics['exact_direction_recall_all_true'])}",
        flush=True,
    )
    if detail:
        print(
            "  reciprocal inverse: "
            f"p={_fmt(metrics['reciprocal_inverse_precision'])} "
            f"coverage(mutual/all)={_fmt(metrics['reciprocal_inverse_coverage_mutual_direct'])}/"
            f"{_fmt(metrics['reciprocal_inverse_coverage_all_true'])} "
            f"edges/tile={_fmt(metrics['reciprocal_edges_per_tile'])}; "
            + " ".join(
                f"{name}:p={_fmt(metrics[f'{name}_precision'])},"
                f"r={_fmt(metrics[f'{name}_recall_candidate'])}/"
                f"{_fmt(metrics[f'{name}_recall_all_true'])}"
                for name in _DIRECTION_NAMES
            ),
            flush=True,
        )


def _full_affinity_baseline(scores: CandidateScores, images: int) -> dict[str, float]:
    """Candidate-only direct-neighbour prior; directions are intentionally absent."""
    valid = scores.valid
    labels = scores.labels
    direct = valid & labels.ne(NON_DIRECT_CLASS)
    all_true = float(images * DIRECT_EDGES_PER_BOARD)
    return {
        "candidate_edges_per_tile": _ratio(float(valid.sum()), float(images * NFRAG)),
        "candidate_direct_fraction": _ratio(float(direct.sum()), float(valid.sum())),
        "candidate_direct_coverage_all_true": _ratio(float(direct.sum()), all_true),
        "mutual_candidate_fraction": _ratio(
            _reciprocal_counts(scores, valid)["mutual_pairs"], float(valid.sum())
        ),
        "mutual_direct_coverage_all_true": _ratio(
            _reciprocal_counts(scores, valid)["mutual_true_direct"], all_true
        ),
    }


def print_affinity_baseline(metrics: Mapping[str, float]) -> None:
    print(
        "[affinity-only candidate prior] "
        f"candidates/tile={_fmt(metrics['candidate_edges_per_tile'])} "
        f"direct precision if every candidate were kept={_fmt(metrics['candidate_direct_fraction'])} "
        f"direct coverage(all true)={_fmt(metrics['candidate_direct_coverage_all_true'])} "
        f"mutual fraction={_fmt(metrics['mutual_candidate_fraction'])} "
        f"mutual-direct coverage(all true)={_fmt(metrics['mutual_direct_coverage_all_true'])}.\n"
        "  It has no U/D/L/R prediction; this is the retrieval prior / oracle-candidate ceiling, not a solver.",
        flush=True,
    )


def _aggregate_and_print_topk(
    name: str, per_image: Sequence[CandidateScores], topks: Sequence[int]
) -> None:
    print(f"\n=== {name}: fixed top-k candidates per source tile ===", flush=True)
    for top_k in topks:
        total: defaultdict[str, float] = defaultdict(float)
        for scores in per_image:
            _add_counts(total, threshold_counts(scores, _topk_select(scores, top_k)))
        # top-1 is the most actionable seed-edge gate.  Include its complete
        # reciprocal and per-direction breakdown rather than leaving a caller
        # to infer inverse consistency from aggregate precision alone.
        print_metric_line(
            f"topk={top_k}", finalize_metrics(total, len(per_image)), detail=(top_k == 1)
        )


def _aggregate_and_print_quantiles(
    name: str, per_image: Sequence[CandidateScores], quantiles: Sequence[float]
) -> None:
    """Threshold sweep using a global threshold per image/method at each quantile.

    Thresholds are computed independently per image because a PairwiseNet
    InfoNCE logit has arbitrary image-level shift.  We print the mean actual
    cut so the curve remains auditable while equal quantiles compare equal
    retained edge density.
    """
    print(f"\n=== {name}: per-image global score-quantile threshold sweep ===", flush=True)
    for quantile in quantiles:
        total: defaultdict[str, float] = defaultdict(float)
        cuts: list[float] = []
        for scores in per_image:
            cut = _quantile_threshold(scores, quantile)
            cuts.append(cut)
            _add_counts(total, threshold_counts(scores, scores.valid & scores.score.ge(cut)))
        metrics = finalize_metrics(total, len(per_image))
        print_metric_line(
            f"q={quantile:.3f} mean_cut={float(np.mean(cuts)):.4f}",
            metrics,
            # q=.99 is the sparse-edge graph gate.  At this cut reciprocal
            # inverse precision and each U/D/L/R rate are decision-critical.
            detail=(quantile >= 0.99),
        )


def _perfect_scores(device: torch.device) -> CandidateScores:
    """A data-free guard for metrics, reciprocal lookup and top-k selection."""
    anchors = torch.arange(NFRAG, device=device)
    rows = torch.div(anchors, GRID, rounding_mode="floor")
    cols = torch.remainder(anchors, GRID)
    candidates = torch.stack(
        (
            torch.where(rows.gt(0), anchors - GRID, anchors),
            torch.where(rows.lt(GRID - 1), anchors + GRID, anchors),
            torch.where(cols.gt(0), anchors - 1, anchors),
            torch.where(cols.lt(GRID - 1), anchors + 1, anchors),
        ),
        dim=-1,
    )
    valid = torch.ones_like(candidates, dtype=torch.bool)
    valid[rows.eq(0), 0] = False
    valid[rows.eq(GRID - 1), 1] = False
    valid[cols.eq(0), 2] = False
    valid[cols.eq(GRID - 1), 3] = False
    labels = candidate_direct_labels(anchors.unsqueeze(0), candidates.unsqueeze(0))[0]
    # Perfect data has one candidate per cardinal direction; exact score loses
    # no edge at top-k=4 and all inverse checks must agree.
    score = torch.ones_like(labels, dtype=torch.float32)
    direction = labels.clamp(max=DIRECT_CLASS_COUNT - 1)
    return CandidateScores(candidates, valid, labels, score, direction)


def smoke(device: torch.device) -> dict[str, float]:
    scores = _perfect_scores(device)
    metrics = finalize_metrics(threshold_counts(scores, scores.valid), 1)
    for key in (
        "direct_edge_precision",
        "direct_edge_recall_all_true",
        "exact_direction_precision",
        "exact_direction_recall_all_true",
        "reciprocal_inverse_precision",
        "reciprocal_inverse_coverage_all_true",
    ):
        if metrics[key] < 0.999:
            raise AssertionError(f"metric smoke failed: {key}={metrics[key]}")
    top4 = finalize_metrics(threshold_counts(scores, _topk_select(scores, 4)), 1)
    if top4["exact_direction_recall_all_true"] < 0.999:
        raise AssertionError(f"top-k smoke failed: {top4}")
    return {
        "exact_precision": metrics["exact_direction_precision"],
        "reciprocal_precision": metrics["reciprocal_inverse_precision"],
        "top4_exact_recall": top4["exact_direction_recall_all_true"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1, help="fresh exact synthetic held-out images")
    parser.add_argument("--top-k", "--top_k", type=int, default=64, help="per-encoder affinity top-K")
    parser.add_argument(
        "--affinity-ckpt", "--affinity_ckpt", default=DEFAULT_AFFINITY_CKPT,
        help="primary MacroAffinityNet checkpoint",
    )
    parser.add_argument(
        "--affinity-ckpt2", "--affinity_ckpt2", default=DEFAULT_AFFINITY_CKPT2,
        help="secondary MacroAffinityNet checkpoint; empty disables union",
    )
    parser.add_argument(
        "--pair-ckpts", "--pair_ckpts", default="",
        help="comma-separated explicit PairwiseNet checkpoint paths; default uses pair0/pair1 ensemble",
    )
    parser.add_argument("--pair-tag", "--pair_tag", default="pair", help="fallback PairwiseNet tag")
    parser.add_argument("--pair-which", "--pair_which", default="best", help="checkpoint suffix")
    parser.add_argument(
        "--direct-pose-ckpt", "--direct_pose_ckpt", default=DEFAULT_DIRECT_POSE_CKPT,
        help="optional DirectPoseNet checkpoint; empty or --no-direct-pose disables it",
    )
    parser.add_argument("--no-direct-pose", "--no_direct_pose", action="store_true")
    parser.add_argument(
        "--fusion-pair-weight", "--fusion_pair_weight", type=float, default=0.5,
        help="heuristic row-z seam contribution to direct-pose fusion in [0,1]",
    )
    parser.add_argument(
        "--topks", nargs="+", default=["1,2,4,8,16,32,64"],
        help="fixed candidates per source in ranking curves",
    )
    parser.add_argument(
        "--quantiles", nargs="+", default=["0.80,0.90,0.95,0.975,0.99"],
        help="per-image score quantiles used as threshold curve cuts",
    )
    parser.add_argument(
        "--pair-batch", "--pair_batch", type=int, default=4096,
        help="maximum raw PairwiseNet seam layouts per inference chunk",
    )
    parser.add_argument(
        "--pose-pair-batch", "--pose_pair_batch", type=int, default=4096,
        help="DirectPoseNet pair inference chunk size",
    )
    parser.add_argument("--seed", type=int, default=SEED + 3191, help="fresh synthetic corruption seed")
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--smoke", action="store_true", help="run data-free metric checks and exit")
    args = parser.parse_args()
    try:
        args.topks = _parse_ints(args.topks, low=1, high=NFRAG - 1, name="top-k")
        args.quantiles = _parse_floats(args.quantiles, low=0.0, high=1.0, name="quantile")
    except ValueError as exc:
        parser.error(str(exc))
    if args.n < 1:
        parser.error("--n must be positive")
    if not 1 <= args.top_k < NFRAG:
        parser.error(f"--top-k must be in [1,{NFRAG - 1}]")
    if args.pair_batch < 4 or args.pose_pair_batch < 1:
        parser.error("--pair-batch must be >=4 and --pose-pair-batch must be positive")
    if not 0.0 <= args.fusion_pair_weight <= 1.0:
        parser.error("--fusion-pair-weight must lie in [0,1]")
    return args


def main() -> None:
    args = _parse_args()
    device = _parse_device(args.device)
    if args.smoke:
        print(f"[pair-affinity evaluator smoke] device={device} {smoke(device)}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    affinity, affinity_metadata, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity2: nn.Module | None = None
    affinity_metadata2: Mapping[str, Any] | None = None
    if args.affinity_ckpt2:
        affinity2, affinity_metadata2, _ = load_frozen_affinity(args.affinity_ckpt2, device)
    pair_paths = _pair_checkpoint_paths(args.pair_ckpts, args.pair_tag, args.pair_which)
    pair_models, pair_metadata = load_pair_ensemble(pair_paths, device)

    direct_model: nn.Module | None = None
    direct_metadata: Mapping[str, Any] | None = None
    direct_path = "" if args.no_direct_pose else args.direct_pose_ckpt.strip()
    if direct_path:
        direct_model, direct_metadata = load_direct_pose(direct_path, device)

    _, val_names = train_val_split()
    if args.n > len(val_names):
        raise ValueError(f"--n={args.n} exceeds held-out split size {len(val_names)}")
    dataset = CanvasDataset(val_names[: args.n], real_prob=0.0, seed=args.seed)

    print(
        f"device={device} exact_fresh_heldout_images={args.n} affinity_top_k={args.top_k} "
        f"encoders={1 + int(affinity2 is not None)} pair_ensemble={len(pair_models)} "
        f"pair_batch(raw-layouts)={args.pair_batch}",
        flush=True,
    )
    print(f"affinity_1={os.path.abspath(args.affinity_ckpt)} step={affinity_metadata.get('step')}", flush=True)
    if affinity2 is not None and affinity_metadata2 is not None:
        print(f"affinity_2={os.path.abspath(args.affinity_ckpt2)} step={affinity_metadata2.get('step')}", flush=True)
    for path, metadata in zip(pair_paths, pair_metadata):
        print(f"pair={os.path.abspath(path)} step={metadata.get('step')} val={metadata.get('val')}", flush=True)
    if direct_model is not None and direct_metadata is not None:
        print(f"direct_pose={os.path.abspath(direct_path)} step={direct_metadata.get('step')}", flush=True)
    else:
        print("direct_pose=disabled", flush=True)
    print(
        "Every affinity candidate receives U/D/L/R physical seam layouts; no dense all-pairs scores are computed.",
        flush=True,
    )

    raw_sets: list[CandidateScores] = []
    z_sets: list[CandidateScores] = []
    pose_sets: list[CandidateScores] = []
    fusion_sets: list[CandidateScores] = []
    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("evaluator requires CanvasDataset(real_prob=0.0) exact labels")
        tiles = sample["tiles"].to(device, non_blocking=device.type == "cuda")
        perm = sample["perm"].to(device, non_blocking=device.type == "cuda").long()
        candidates_batched, valid_batched = mine_affinity_candidates(
            affinity, tiles.unsqueeze(0), candidate_k=args.top_k, device=device, affinity_secondary=affinity2
        )
        candidates, valid = candidates_batched[0], valid_batched[0]
        labels = candidate_direct_labels(perm.unsqueeze(0), candidates.unsqueeze(0))[0]
        orientations = score_pairwise_directions(
            pair_models, tiles, candidates, valid, pair_batch=args.pair_batch, device=device
        )
        raw, calibrated, orientation_z = _candidate_scores_from_orientations(
            candidates, valid, labels, orientations
        )
        raw_sets.append(raw)
        z_sets.append(calibrated)

        if direct_model is not None:
            pose, pose_orientations = score_direct_pose_bundle(
                direct_model, tiles, candidates, valid, labels,
                pair_batch=args.pose_pair_batch, device=device,
            )
            pose_sets.append(pose)
            pair_probability = torch.sigmoid(orientation_z)
            fused_orientations = (
                (1.0 - float(args.fusion_pair_weight)) * pose_orientations
                + float(args.fusion_pair_weight) * pair_probability
            )
            fusion_score, fusion_direction = fused_orientations.max(dim=-1)
            fusion_sets.append(CandidateScores(candidates, valid, labels, fusion_score, fusion_direction))

        candidate_pairs = int(valid.sum())
        raw_layouts = candidate_pairs * DIRECT_CLASS_COUNT
        dense_layouts = NFRAG * (NFRAG - 1) * DIRECT_CLASS_COUNT
        print(
            f"processed {index + 1}/{args.n}: union candidates={candidate_pairs} "
            f"({candidate_pairs / NFRAG:.2f}/tile), evaluated seam layouts={raw_layouts} "
            f"({100.0 * raw_layouts / dense_layouts:.2f}% of dense UDLR universe)",
            flush=True,
        )

    print("\n=== candidate graph baseline ===", flush=True)
    # Scores only differ in direction/score; the affinity retrieval baseline is
    # identical for raw/z/pose, so use the first seam bundle.
    aggregate_prior: defaultdict[str, float] = defaultdict(float)
    for scores in raw_sets:
        prior = _full_affinity_baseline(scores, 1)
        # Preserve count semantics by accumulating raw counts directly instead
        # of averaging metrics that have distinct denominators.
        aggregate_prior["candidate_pairs"] += float(scores.valid.sum())
        aggregate_prior["direct"] += float((scores.valid & scores.labels.ne(NON_DIRECT_CLASS)).sum())
        reciprocal = _reciprocal_counts(scores, scores.valid)
        aggregate_prior["mutual_pairs"] += reciprocal["mutual_pairs"]
        aggregate_prior["mutual_direct"] += reciprocal["mutual_true_direct"]
    all_true = float(args.n * DIRECT_EDGES_PER_BOARD)
    print_affinity_baseline(
        {
            "candidate_edges_per_tile": _ratio(aggregate_prior["candidate_pairs"], float(args.n * NFRAG)),
            "candidate_direct_fraction": _ratio(aggregate_prior["direct"], aggregate_prior["candidate_pairs"]),
            "candidate_direct_coverage_all_true": _ratio(aggregate_prior["direct"], all_true),
            "mutual_candidate_fraction": _ratio(aggregate_prior["mutual_pairs"], aggregate_prior["candidate_pairs"]),
            "mutual_direct_coverage_all_true": _ratio(aggregate_prior["mutual_direct"], all_true),
        }
    )

    _aggregate_and_print_quantiles("PairwiseNet raw max seam logit", raw_sets, args.quantiles)
    _aggregate_and_print_topk("PairwiseNet raw max seam logit", raw_sets, args.topks)
    _aggregate_and_print_quantiles("PairwiseNet per-row orientation z calibration", z_sets, args.quantiles)
    _aggregate_and_print_topk("PairwiseNet per-row orientation z calibration", z_sets, args.topks)
    if pose_sets:
        _aggregate_and_print_quantiles("DirectPoseNet candidate confidence", pose_sets, args.quantiles)
        _aggregate_and_print_topk("DirectPoseNet candidate confidence", pose_sets, args.topks)
        _aggregate_and_print_quantiles(
            f"heuristic fusion (pair weight={args.fusion_pair_weight:.2f}, no fitted calibration)",
            fusion_sets,
            args.quantiles,
        )
        _aggregate_and_print_topk(
            f"heuristic fusion (pair weight={args.fusion_pair_weight:.2f}, no fitted calibration)",
            fusion_sets,
            args.topks,
        )


if __name__ == "__main__":
    main()
