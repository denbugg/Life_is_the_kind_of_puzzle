"""Evaluate a direct-neighbour pose classifier on a frozen affinity graph.

This is the strict gate for the direct-pose branch.  It never scores the
``576 x 575`` all-pairs universe: a frozen :class:`MacroAffinityNet` (or the
deduplicated union of two encoders) first proposes a small directed candidate
set for each tile, then :class:`DirectPoseNet` scores only those pairs in
bounded chunks.

The evaluator deliberately uses fresh synthetic corruptions of the held-out
target split.  Consequently every input tile has an exact clean-grid label,
but no recovered real-input permutation is ever consulted.

The confidence swept below is the model's marginal confidence in its chosen
directed edge::

    P(direct) * max_d P(direction=d | direct).

For each threshold we report both candidate-conditioned recall and recall over
*all* 2,208 true directed board edges.  The latter includes affinity-candidate
misses and is the graph-ready quantity.

Examples
--------

    python src/eval_direct_pose.py --smoke --device cpu
    python src/eval_direct_pose.py --n 8 --device cuda
    python src/eval_direct_pose.py --n 2 --sync --sync-threshold 0.7
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from direct_pose import (
    DIRECT_CLASS_COUNT,
    DIRECT_OFFSETS,
    NON_DIRECT_CLASS,
    NUM_CLASSES,
    DirectPoseNet,
    hierarchical_predictions,
    inverse_classes,
)
# The generic offset evaluator already contains the carefully tested robust
# Laplacian/IRLS synchronization and the 24x24 Hungarian decode.  DirectPoseNet
# produces a special case of those constraints: exact cardinal integer deltas.
from eval_offset_pose import (
    build_edge_matrices,
    collapse_constraints,
    synchronize_coordinates,
    synchronization_metrics,
)
from imgio import train_val_split
from train_direct_pose import candidate_direct_labels, score_candidate_graph
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_POSE_CKPT = os.path.join(
    WORKSPACE, "artifacts", "direct_pose", "direct_union_400_best.pt"
)
DEFAULT_AFFINITY_CKPT = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r1_1200_best.pt"
)
DEFAULT_AFFINITY_CKPT2 = os.path.join(
    WORKSPACE, "artifacts", "macro_affinity", "affinity_r3_1000_best.pt"
)
DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DIRECT_EDGES_PER_BOARD = 4 * GRID * (GRID - 1)
_DIRECTION_NAMES = ("U", "D", "L", "R")


def _torch_load(path: str) -> object:
    """Load both current and older CPU-readable trainer checkpoints."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch versions predating ``weights_only``.
        return torch.load(path, map_location="cpu")


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_tensor_state_dict(value: object) -> bool:
    return bool(value) and isinstance(value, Mapping) and all(
        isinstance(key, str) and isinstance(tensor, Tensor) for key, tensor in value.items()
    )


def _checkpoint_state(payload: object) -> dict[str, Tensor]:
    """Extract a state dictionary from sensible trainer / raw-state formats."""
    if _is_tensor_state_dict(payload):
        return dict(payload)  # type: ignore[arg-type]
    mapping = _as_mapping(payload)
    for key in ("model", "model_state_dict", "state_dict", "net", "network"):
        value = mapping.get(key)
        if isinstance(value, nn.Module):
            return dict(value.state_dict())
        if _is_tensor_state_dict(value):
            return dict(value)  # type: ignore[arg-type]
    raise RuntimeError(
        "could not find a DirectPoseNet state dictionary; expected a trainer "
        "checkpoint with a model/state_dict field"
    )


def _checkpoint_module(payload: object) -> nn.Module | None:
    if isinstance(payload, nn.Module):
        return payload
    mapping = _as_mapping(payload)
    for key in ("model", "module", "network", "net"):
        value = mapping.get(key)
        if isinstance(value, nn.Module):
            return value
    return None


def _strip_uniform_prefix(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Accept state dicts saved from DataParallel or a one-field wrapper."""
    result = dict(state)
    for prefix in ("module.", "model.", "net."):
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix) :]: value for key, value in result.items()}
    return result


def _positive_int(*values: object, fallback: int) -> int:
    for value in values:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return fallback


def _float_in_range(*values: object, fallback: float, low: float, high: float) -> float:
    for value in values:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if low <= parsed <= high:
            return parsed
    return fallback


def _state_dim(state: Mapping[str, Tensor], key: str, dimension: int) -> int | None:
    value = state.get(key)
    return int(value.shape[dimension]) if value is not None and value.ndim > dimension else None


def _direct_model_kwargs(payload: object, state: Mapping[str, Tensor]) -> dict[str, Any]:
    """Reconstruct the compact CNN without assuming a particular checkpoint age."""
    metadata = _as_mapping(payload)
    saved = _as_mapping(metadata.get("model_kwargs"))
    args = _as_mapping(metadata.get("args"))
    tile_size = _positive_int(
        saved.get("tile_size"), saved.get("fragment_size"), metadata.get("fragment_size"), FS, fallback=FS
    )
    width = _positive_int(
        saved.get("width"), args.get("width"), _state_dim(state, "stem.0.weight", 0), fallback=48
    )
    dropout = _float_in_range(
        saved.get("dropout"), args.get("dropout"), fallback=0.0, low=0.0, high=0.999
    )
    num_classes = _positive_int(
        saved.get("num_classes"), metadata.get("num_classes"), _state_dim(state, "head.4.weight", 0),
        fallback=NUM_CLASSES,
    )
    return {
        "tile_size": tile_size,
        "width": width,
        # Evaluation has ``model.eval()``, but preserving the saved value makes
        # an architecture mismatch explicit during state loading.
        "dropout": dropout,
        "num_classes": num_classes,
    }


def load_direct_pose(path: str, device: torch.device) -> tuple[DirectPoseNet, Mapping[str, Any]]:
    """Load a trainer checkpoint or an explicitly serialized DirectPoseNet."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"direct-pose checkpoint does not exist: {path}")
    payload = _torch_load(path)
    metadata = _as_mapping(payload)
    model = _checkpoint_module(payload)
    if model is None:
        state = _strip_uniform_prefix(_checkpoint_state(payload))
        kwargs = _direct_model_kwargs(payload, state)
        if kwargs["tile_size"] != FS:
            raise RuntimeError(
                f"checkpoint tile_size={kwargs['tile_size']} is incompatible with fragment size {FS}"
            )
        if kwargs["num_classes"] != NUM_CLASSES:
            raise RuntimeError(
                f"checkpoint has {kwargs['num_classes']} classes; DirectPoseNet requires {NUM_CLASSES}"
            )
        model = DirectPoseNet(**kwargs)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            missing = ", ".join(incompatible.missing_keys[:8]) or "none"
            unexpected = ", ".join(incompatible.unexpected_keys[:8]) or "none"
            raise RuntimeError(
                "direct-pose checkpoint architecture mismatch "
                f"(missing={missing}; unexpected={unexpected}; inferred={kwargs})"
            )
    if not isinstance(model, DirectPoseNet):
        raise RuntimeError(
            f"checkpoint contains {type(model).__name__}, not DirectPoseNet; "
            "this evaluator intentionally does not reinterpret a different pose head"
        )
    if model.tile_size != FS or model.num_classes != NUM_CLASSES:
        raise RuntimeError(
            "loaded DirectPoseNet has incompatible contract "
            f"tile_size={model.tile_size}, classes={model.num_classes}"
        )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata


def _parse_device(value: str | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return device


def _parse_thresholds(values: Sequence[str]) -> tuple[float, ...]:
    parsed: list[float] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                threshold = float(item)
            except ValueError as exc:
                raise ValueError(f"invalid threshold {item!r}") from exc
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"threshold must lie in [0,1], got {threshold}")
            parsed.append(threshold)
    if not parsed:
        raise ValueError("at least one confidence threshold is required")
    return tuple(sorted(set(parsed)))


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _exact_board_direction_counts() -> dict[int, float]:
    # There are 24*23 directed edges in each of U/D/L/R.
    return {direction: float(GRID * (GRID - 1)) for direction in range(DIRECT_CLASS_COUNT)}


@dataclass
class CandidateScores:
    """One image's frozen candidate graph and direct-pose prediction tensors."""

    candidates: Tensor  # (576, K)
    valid: Tensor  # (576, K) bool
    labels: Tensor  # (576, K), U/D/L/R/non-direct
    direct_probability: Tensor  # (576, K)
    direction: Tensor  # (576, K), conditional U/D/L/R argmax
    confidence: Tensor  # (576, K), P(direct)*P(direction|direct)


def _reciprocal_counts(
    scores: CandidateScores,
    selected: Tensor,
) -> dict[str, float]:
    """Count mutually proposed, selected inverse-direction edges exactly once per orientation.

    The two coverage denominators are intentionally distinct.  ``mutual_true``
    asks whether the candidate graph contained both orientations; ``all_true``
    includes candidate mining misses and is the useful end-to-end graph gate.
    """
    candidates, valid, labels = scores.candidates, scores.valid, scores.labels
    if candidates.ndim != 2 or tuple(candidates.shape) != tuple(valid.shape):
        raise ValueError("candidate/valid shapes are inconsistent")
    count, width = candidates.shape
    if count != NFRAG:
        raise ValueError(f"candidate graph must have {NFRAG} anchors")
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
    direct = valid & labels.ne(NON_DIRECT_CLASS)
    # ``inverse_classes`` accepts the non-direct sentinel too, so the boolean
    # direct masks below make this safe on every candidate pair.
    true_mutual_direct = mutual & direct & reverse_labels.eq(inverse_classes(labels))
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


def threshold_counts(scores: CandidateScores, threshold: float) -> dict[str, float]:
    """Return additive direct-edge, directional, and reciprocal counts.

    A selected edge always predicts one cardinal direction.  Therefore direct
    precision asks whether its endpoints are adjacent, while exact precision
    additionally asks whether the predicted U/D/L/R orientation is right.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0,1]")
    valid, labels = scores.valid, scores.labels
    direct = valid & labels.ne(NON_DIRECT_CLASS)
    selected = valid & scores.confidence.ge(float(threshold))
    direct_hit = selected & direct
    exact = direct_hit & scores.direction.eq(labels)
    result: dict[str, float] = {
        "candidate_pairs": float(valid.sum()),
        "candidate_true_direct": float(direct.sum()),
        "selected": float(selected.sum()),
        "selected_true_direct": float(direct_hit.sum()),
        "selected_exact_direction": float(exact.sum()),
    }
    for direction, name in enumerate(_DIRECTION_NAMES):
        truth = valid & labels.eq(direction)
        predicted = selected & scores.direction.eq(direction)
        correct = predicted & labels.eq(direction)
        result[f"{name}_truth_candidate"] = float(truth.sum())
        result[f"{name}_predicted"] = float(predicted.sum())
        result[f"{name}_correct"] = float(correct.sum())
    result.update(_reciprocal_counts(scores, selected))
    return result


def finalize_threshold_metrics(counts: Mapping[str, float], images: int) -> dict[str, float]:
    """Turn additive counts into unambiguous candidate and end-to-end rates."""
    if images < 1:
        raise ValueError("images must be positive")
    candidate_direct = counts["candidate_true_direct"]
    selected = counts["selected"]
    true_direct = float(images * DIRECT_EDGES_PER_BOARD)
    result = {
        "candidate_edges_per_tile": _ratio(counts["candidate_pairs"], float(images * NFRAG)),
        "candidate_direct_coverage_all_true": _ratio(candidate_direct, true_direct),
        "selected_edges_per_tile": _ratio(selected, float(images * NFRAG)),
        "direct_edge_precision": _ratio(counts["selected_true_direct"], selected),
        "direct_edge_recall_candidate": _ratio(counts["selected_true_direct"], candidate_direct),
        "direct_edge_recall_all_true": _ratio(counts["selected_true_direct"], true_direct),
        "exact_direction_precision": _ratio(counts["selected_exact_direction"], selected),
        "exact_direction_recall_candidate": _ratio(counts["selected_exact_direction"], candidate_direct),
        "exact_direction_recall_all_true": _ratio(counts["selected_exact_direction"], true_direct),
        "reciprocal_inverse_precision": _ratio(
            counts["correct_reciprocal_inverse"], counts["predicted_reciprocal_inverse"]
        ),
        "reciprocal_inverse_coverage_mutual_direct": _ratio(
            counts["correct_reciprocal_inverse"], counts["mutual_true_direct"]
        ),
        "reciprocal_inverse_coverage_all_true": _ratio(
            counts["correct_reciprocal_inverse"], true_direct
        ),
        "reciprocal_edges_per_tile": _ratio(
            counts["predicted_reciprocal_inverse"], float(images * NFRAG)
        ),
        "mutual_candidate_fraction": _ratio(counts["mutual_pairs"], counts["candidate_pairs"]),
    }
    all_direction_counts = _exact_board_direction_counts()
    for direction, name in enumerate(_DIRECTION_NAMES):
        correct = counts[f"{name}_correct"]
        result[f"{name}_precision"] = _ratio(correct, counts[f"{name}_predicted"])
        result[f"{name}_recall_candidate"] = _ratio(correct, counts[f"{name}_truth_candidate"])
        result[f"{name}_recall_all_true"] = _ratio(correct, images * all_direction_counts[direction])
    return result


def _flatten_valid_candidates(scores: CandidateScores) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize only actual directed affinity pairs for optional graph sync."""
    count, width = scores.candidates.shape
    anchors = torch.arange(count, device=scores.candidates.device).view(count, 1).expand(count, width)
    valid = scores.valid
    source = anchors[valid].detach().cpu().numpy().astype(np.int64, copy=False)
    target = scores.candidates[valid].detach().cpu().numpy().astype(np.int64, copy=False)
    direction = scores.direction[valid].detach().cpu().numpy().astype(np.int64, copy=False)
    confidence = scores.confidence[valid].detach().cpu().numpy().astype(np.float32, copy=False)
    direct_probability = scores.direct_probability[valid].detach().cpu().numpy().astype(np.float32, copy=False)
    return source, target, direction, confidence, direct_probability


def sync_one(
    scores: CandidateScores,
    perm: Tensor,
    *,
    threshold: float,
    iterations: int,
    huber: float,
) -> dict[str, float]:
    """Synchronize only inverse-consistent confidence-filtered cardinal edges."""
    source, target, direction, confidence, direct_probability = _flatten_valid_candidates(scores)
    table = np.asarray(DIRECT_OFFSETS, dtype=np.float32)
    delta = table[direction]
    # ``confidence >= threshold`` already implies P(direct) >= threshold, but
    # passing the latter explicitly documents the hierarchical admission gate.
    predicted_local = direct_probability >= float(threshold)
    edge = build_edge_matrices(
        source,
        target,
        delta,
        confidence,
        direct_probability,
        predicted_local,
        min_confidence=float(threshold),
        max_predicted_offset=1.0,
        reciprocal_tolerance=1.0e-6,
        require_reciprocal=True,
    )
    constraints = collapse_constraints(edge, reciprocal_tolerance=1.0e-6)
    sync = synchronize_coordinates(constraints, iterations=iterations, huber=huber)
    return synchronization_metrics(sync, constraints, perm)


@torch.inference_mode()
def score_one(
    sample: Mapping[str, Tensor],
    pose_model: DirectPoseNet,
    affinity: nn.Module,
    *,
    affinity_secondary: nn.Module | None,
    candidate_k: int,
    pair_batch: int,
    device: torch.device,
) -> tuple[CandidateScores, Tensor]:
    """Mine one frozen graph and score its valid pairs in chunks only."""
    if not bool(sample["has_perm"]):
        raise RuntimeError("eval_direct_pose requires CanvasDataset(real_prob=0.0) exact labels")
    tiles = sample["tiles"].unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    perm = sample["perm"].to(device, non_blocking=device.type == "cuda").long().unsqueeze(0)
    candidates, valid = mine_affinity_candidates(
        affinity,
        tiles,
        candidate_k=candidate_k,
        device=device,
        affinity_secondary=affinity_secondary,
    )
    labels = candidate_direct_labels(perm, candidates)
    logits = score_candidate_graph(
        pose_model, tiles, candidates, valid=valid, pair_batch=pair_batch, device=device
    )
    decoded = hierarchical_predictions(logits)
    return (
        CandidateScores(
            candidates=candidates[0],
            valid=valid[0],
            labels=labels[0],
            direct_probability=decoded["direct_probability"][0],
            direction=decoded["conditional_direction_class"][0],
            confidence=decoded["confidence"][0],
        ),
        perm[0],
    )


def _add_counts(total: defaultdict[str, float], values: Mapping[str, float]) -> None:
    for key, value in values.items():
        total[key] += float(value)


def _add_finite(
    total: defaultdict[str, float], count: defaultdict[str, int], values: Mapping[str, float]
) -> None:
    for key, value in values.items():
        if np.isfinite(value):
            total[key] += float(value)
            count[key] += 1


def _mean_finite(total: Mapping[str, float], count: Mapping[str, int]) -> dict[str, float]:
    return {key: _ratio(value, float(count[key])) for key, value in total.items() if count[key]}


def _fmt(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "nan"


def print_threshold_report(threshold: float, metrics: Mapping[str, float]) -> None:
    print(
        f"[confidence>={threshold:.3f}] candidates/tile={_fmt(metrics['candidate_edges_per_tile'])} "
        f"candidate_direct_coverage={_fmt(metrics['candidate_direct_coverage_all_true'])} "
        f"selected/tile={_fmt(metrics['selected_edges_per_tile'])}",
        flush=True,
    )
    print(
        "  direct edge: "
        f"precision={_fmt(metrics['direct_edge_precision'])} "
        f"recall(candidate/all)={_fmt(metrics['direct_edge_recall_candidate'])}/"
        f"{_fmt(metrics['direct_edge_recall_all_true'])}",
        flush=True,
    )
    print(
        "  exact U/D/L/R: "
        f"precision={_fmt(metrics['exact_direction_precision'])} "
        f"recall(candidate/all)={_fmt(metrics['exact_direction_recall_candidate'])}/"
        f"{_fmt(metrics['exact_direction_recall_all_true'])}",
        flush=True,
    )
    print(
        "  per direction: "
        + " ".join(
            f"{name}:p={_fmt(metrics[f'{name}_precision'])},"
            f"r={_fmt(metrics[f'{name}_recall_candidate'])}/"
            f"{_fmt(metrics[f'{name}_recall_all_true'])}"
            for name in _DIRECTION_NAMES
        ),
        flush=True,
    )
    print(
        "  reciprocal inverse: "
        f"precision={_fmt(metrics['reciprocal_inverse_precision'])} "
        f"coverage(mutual/all)={_fmt(metrics['reciprocal_inverse_coverage_mutual_direct'])}/"
        f"{_fmt(metrics['reciprocal_inverse_coverage_all_true'])} "
        f"edges/tile={_fmt(metrics['reciprocal_edges_per_tile'])}",
        flush=True,
    )


def _perfect_candidate_graph(device: torch.device) -> tuple[CandidateScores, Tensor]:
    """Build a complete direct-neighbour graph plus one far candidate per tile."""
    anchors = torch.arange(NFRAG, device=device)
    rows = torch.div(anchors, GRID, rounding_mode="floor")
    cols = torch.remainder(anchors, GRID)
    # Slot 0..3 is U/D/L/R; invalid border slots are masked.  Slot 4 is a
    # deterministic non-neighbour so direct/non-direct thresholding is tested.
    candidates = torch.stack(
        (
            torch.where(rows.gt(0), anchors - GRID, anchors),
            torch.where(rows.lt(GRID - 1), anchors + GRID, anchors),
            torch.where(cols.gt(0), anchors - 1, anchors),
            torch.where(cols.lt(GRID - 1), anchors + 1, anchors),
            torch.remainder(anchors + GRID * 12, NFRAG),
        ),
        dim=-1,
    )
    valid = torch.ones_like(candidates, dtype=torch.bool)
    valid[rows.eq(0), 0] = False
    valid[rows.eq(GRID - 1), 1] = False
    valid[cols.eq(0), 2] = False
    valid[cols.eq(GRID - 1), 3] = False
    perm = anchors.clone()
    labels = candidate_direct_labels(perm.unsqueeze(0), candidates.unsqueeze(0))[0]
    logits = torch.full((*labels.shape, NUM_CLASSES), -12.0, device=device)
    logits.scatter_(-1, labels.unsqueeze(-1), 12.0)
    decoded = hierarchical_predictions(logits)
    return (
        CandidateScores(
            candidates=candidates,
            valid=valid,
            labels=labels,
            direct_probability=decoded["direct_probability"],
            direction=decoded["conditional_direction_class"],
            confidence=decoded["confidence"],
        ),
        perm,
    )


def smoke(device: torch.device) -> dict[str, float]:
    """CPU/GPU-free-data guard for threshold metrics, reciprocity, and sync."""
    scores, perm = _perfect_candidate_graph(device)
    metrics = finalize_threshold_metrics(threshold_counts(scores, 0.9), images=1)
    for key in (
        "candidate_direct_coverage_all_true",
        "direct_edge_precision",
        "direct_edge_recall_all_true",
        "exact_direction_precision",
        "exact_direction_recall_all_true",
        "reciprocal_inverse_precision",
        "reciprocal_inverse_coverage_all_true",
    ):
        if metrics[key] < 0.999:
            raise AssertionError(f"perfect direct-pose metric guard failed: {key}={metrics[key]}")
    sync_metrics = sync_one(scores, perm, threshold=0.9, iterations=2, huber=0.5)
    if sync_metrics["sync_largest_component_fraction"] < 0.999:
        raise AssertionError(f"perfect direct graph did not synchronize: {sync_metrics}")
    if sync_metrics["hungarian_component_placement"] < 0.999:
        raise AssertionError(f"perfect direct graph Hungarian decode failed: {sync_metrics}")
    return {
        "exact_precision": metrics["exact_direction_precision"],
        "reciprocal_precision": metrics["reciprocal_inverse_precision"],
        "sync_component": sync_metrics["sync_largest_component_fraction"],
        "hungarian_component_placement": sync_metrics["hungarian_component_placement"],
    }


def _checkpoint_step(metadata: Mapping[str, Any]) -> str:
    step = metadata.get("step")
    return f" step={step}" if step is not None else ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_ckpt", default=DEFAULT_POSE_CKPT, help="DirectPoseNet trainer checkpoint")
    parser.add_argument(
        "--affinity_ckpt", default=DEFAULT_AFFINITY_CKPT, help="primary frozen MacroAffinityNet"
    )
    parser.add_argument(
        "--affinity_ckpt2",
        default=DEFAULT_AFFINITY_CKPT2,
        help="optional secondary frozen MacroAffinityNet; pass an empty string to disable union",
    )
    parser.add_argument("--n", type=int, default=8, help="fresh held-out synthetic images")
    parser.add_argument("--top_k", type=int, default=64, help="per-encoder frozen affinity top-K")
    parser.add_argument("--pair_batch", type=int, default=4096, help="DirectPoseNet inference pairs per chunk")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=[",".join(str(value) for value in DEFAULT_THRESHOLDS)],
        help="confidence thresholds, comma-separated and/or space-separated (default: 0.3 ... 0.9)",
    )
    parser.add_argument("--seed", type=int, default=SEED + 9173, help="fresh synthetic corruption seed")
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="also synchronize reciprocal direct edges and Hungarian-decode the largest component",
    )
    parser.add_argument(
        "--sync-threshold",
        "--sync_threshold",
        type=float,
        default=0.7,
        help="confidence threshold used by optional synchronization",
    )
    parser.add_argument("--sync-iterations", "--sync_iterations", type=int, default=5)
    parser.add_argument("--sync-huber", "--sync_huber", type=float, default=0.5)
    parser.add_argument(
        "--smoke", "--tiny_smoke", action="store_true", help="run data-free metric/sync checks and exit"
    )
    args = parser.parse_args()
    try:
        args.thresholds = _parse_thresholds(args.thresholds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.n < 1 or args.pair_batch < 1:
        parser.error("--n and --pair_batch must be positive")
    if not 1 <= args.top_k < NFRAG:
        parser.error(f"--top_k must lie in [1,{NFRAG - 1}]")
    if not 0.0 <= args.sync_threshold <= 1.0:
        parser.error("--sync-threshold must lie in [0,1]")
    if args.sync_iterations < 1 or args.sync_huber <= 0.0:
        parser.error("--sync-iterations must be positive and --sync-huber must be positive")
    return args


def main() -> None:
    args = _parse_args()
    device = _parse_device(args.device)
    if args.smoke:
        print(f"[direct-pose evaluator smoke] device={device} {smoke(device)}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    pose_model, pose_metadata = load_direct_pose(args.pose_ckpt, device)
    affinity, affinity_metadata, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity_secondary: nn.Module | None = None
    affinity_metadata2: Mapping[str, Any] | None = None
    if args.affinity_ckpt2:
        affinity_secondary, affinity_metadata2, _ = load_frozen_affinity(args.affinity_ckpt2, device)

    _, val_names = train_val_split()
    if args.n > len(val_names):
        raise ValueError(f"--n={args.n} exceeds held-out split size {len(val_names)}")
    # A new loader and a seed intentionally distinct from the trainer default
    # generate fresh distortions while staying within the held-out image split.
    dataset = CanvasDataset(val_names[: args.n], real_prob=0.0, seed=args.seed)
    print(
        f"device={device} exact_fresh_heldout_images={args.n} top_k={args.top_k} "
        f"encoders={1 + int(affinity_secondary is not None)} pair_batch={args.pair_batch}",
        flush=True,
    )
    print(
        f"pose={os.path.abspath(args.pose_ckpt)} width={pose_model.width}"
        f"{_checkpoint_step(pose_metadata)}",
        flush=True,
    )
    print(
        f"affinity_1={os.path.abspath(args.affinity_ckpt)}{_checkpoint_step(affinity_metadata)}",
        flush=True,
    )
    if affinity_secondary is not None and affinity_metadata2 is not None:
        print(
            f"affinity_2={os.path.abspath(args.affinity_ckpt2)}{_checkpoint_step(affinity_metadata2)}",
            flush=True,
        )
    print(
        "confidence=P(direct)*max P(U/D/L/R|direct); all-true recall includes frozen affinity misses.",
        flush=True,
    )

    totals: dict[float, defaultdict[str, float]] = {
        threshold: defaultdict(float) for threshold in args.thresholds
    }
    sync_total: defaultdict[str, float] = defaultdict(float)
    sync_count: defaultdict[str, int] = defaultdict(int)
    for index in range(args.n):
        scores, perm = score_one(
            dataset[index],
            pose_model,
            affinity,
            affinity_secondary=affinity_secondary,
            candidate_k=args.top_k,
            pair_batch=args.pair_batch,
            device=device,
        )
        for threshold in args.thresholds:
            _add_counts(totals[threshold], threshold_counts(scores, threshold))
        if args.sync:
            _add_finite(
                sync_total,
                sync_count,
                sync_one(
                    scores,
                    perm,
                    threshold=args.sync_threshold,
                    iterations=args.sync_iterations,
                    huber=args.sync_huber,
                ),
            )
        print(f"processed {index + 1}/{args.n}", flush=True)

    print("\n=== direct-pose confidence sweep ===", flush=True)
    for threshold in args.thresholds:
        print_threshold_report(threshold, finalize_threshold_metrics(totals[threshold], args.n))

    if args.sync:
        metrics = _mean_finite(sync_total, sync_count)
        print(
            f"\n=== reciprocal pose-graph sync at confidence>={args.sync_threshold:.3f} "
            f"(finite-image mean) ===",
            flush=True,
        )
        if not metrics:
            print("  no finite synchronization metrics (no usable reciprocal component)", flush=True)
        else:
            print(
                "  "
                f"constraints={_fmt(metrics.get('sync_constraints', float('nan')))} "
                f"largest_component={_fmt(metrics.get('sync_largest_component_fraction', float('nan')))} "
                f"affine_r2={_fmt(metrics.get('sync_affine_coordinate_r2', float('nan')))} "
                f"hungarian_component_place={_fmt(metrics.get('hungarian_component_placement', float('nan')))} "
                f"hungarian_whole_place={_fmt(metrics.get('hungarian_whole_placement', float('nan')))} "
                f"whole_neighbour={_fmt(metrics.get('hungarian_whole_neighbour', float('nan')))}",
                flush=True,
            )


if __name__ == "__main__":
    main()
