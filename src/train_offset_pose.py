"""Train a directional local-offset classifier on affinity-mined tile pairs.

The old pair scorer asks only whether two seams look compatible.  This branch
instead asks a geometric question about an *ordered* pair: is tile ``j`` at one
of the forty-eight exact offsets in ``[-3,3] x [-3,3]`` from tile ``i``, or is
it ``far``?  The pose model sees the full two 20x20 tiles rather than edge
strips.

Candidate pairs deliberately come from a frozen ``MacroAffinityNet`` top-K
graph.  That supplies the hard, nearby-looking false positives encountered at
inference, while avoiding an expensive trainable pass over all 576 x 575
ordered pairs.  Exact labels originate only from newly distorted synthetic
``CanvasDataset(real_prob=0)`` examples.

Typical run after affinity pretraining:

    python src/train_offset_pose.py ^
      --affinity_ckpt artifacts/macro_affinity/affinity_r1_1200_best.pt ^
      --affinity_ckpt2 artifacts/macro_affinity/affinity_r3_1000_best.pt ^
      --steps 4000 --candidate_k 64 --pairs_per_image 1024 --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from imgio import train_val_split
from macro_affinity import MacroAffinityNet
from offset_pose import (
    CLASS_OFFSETS,
    FAR_CLASS,
    LOCAL_CLASS_COUNT,
    NUM_CLASSES,
    OFFSET_RADIUS,
    OffsetPoseNet,
    aggregate_local_logit,
    class_offsets_metadata,
    count_params,
    hierarchical_predictions,
    inverse_classes,
    offsets_to_classes,
)


def _autocast(device: torch.device):
    """Use fp16 only where CUDA is available, keeping CPU behavior simple."""
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def _torch_load(path: str) -> object:
    """Load a trusted local experiment checkpoint on old and new PyTorch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before ``weights_only``.
        return torch.load(path, map_location="cpu")


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    attributes = getattr(value, "__dict__", None)
    return attributes if isinstance(attributes, Mapping) else {}


def _looks_like_state_dict(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) and isinstance(tensor, Tensor) for key, tensor in value.items())
    )


def _checkpoint_state(payload: object) -> dict[str, Tensor]:
    """Extract an affinity state dictionary from common local checkpoint layouts."""
    if isinstance(payload, nn.Module):
        return dict(payload.state_dict())
    if _looks_like_state_dict(payload):
        return dict(payload)
    if isinstance(payload, Mapping):
        for key in ("model", "model_state_dict", "state_dict", "network", "net"):
            candidate = payload.get(key)
            if isinstance(candidate, nn.Module):
                return dict(candidate.state_dict())
            if _looks_like_state_dict(candidate):
                return dict(candidate)
    raise RuntimeError(
        "affinity checkpoint has no recognizable model state dictionary "
        "(expected raw state or model/model_state_dict/state_dict)"
    )


def _strip_uniform_prefix(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Remove a wrapper prefix only when every key uses it."""
    cleaned = dict(state)
    for prefix in ("module.", "model."):
        if cleaned and all(key.startswith(prefix) for key in cleaned):
            cleaned = {key[len(prefix) :]: value for key, value in cleaned.items()}
    return cleaned


def _first_positive_int(*values: object, fallback: int) -> int:
    for value in values:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return fallback


def _state_dim(state: Mapping[str, Tensor], key: str, dimension: int) -> int | None:
    tensor = state.get(key)
    if tensor is None or tensor.ndim <= dimension:
        return None
    return int(tensor.shape[dimension])


def _affinity_model_kwargs(payload: object, state: Mapping[str, Tensor]) -> dict[str, Any]:
    """Recover a MacroAffinityNet architecture from metadata, then state shapes."""
    metadata = _as_mapping(payload)
    args = _as_mapping(metadata.get("args"))
    saved = _as_mapping(metadata.get("model_kwargs"))
    embedding_dim = _first_positive_int(
        saved.get("embedding_dim"),
        saved.get("d"),
        args.get("embedding_dim"),
        args.get("d"),
        _state_dim(state, "backbone.head.5.weight", 0),
        _state_dim(state, "backbone.head.4.weight", 0),
        fallback=128,
    )
    width = _first_positive_int(
        saved.get("width"), args.get("width"), _state_dim(state, "backbone.stem.0.weight", 0), fallback=48
    )
    stats_hidden = _first_positive_int(
        saved.get("stats_hidden"), args.get("stats_hidden"), _state_dim(state, "stats.net.1.weight", 0), fallback=32
    )
    # Dropout has no learned parameters and the frozen model always runs in
    # eval mode, so only the module shape-relevant fields matter here.
    return {
        "tiles": NFRAG,
        "tile_size": FS,
        "embedding_dim": embedding_dim,
        "width": width,
        "use_stats": any(key.startswith("stats.") for key in state),
        "stats_hidden": stats_hidden,
        "dropout": 0.0,
    }


def load_frozen_affinity(
    path: str, device: torch.device
) -> tuple[MacroAffinityNet, Mapping[str, Any], dict[str, Any]]:
    """Load, freeze, and validate the affinity encoder used for candidate mining."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"affinity checkpoint does not exist: {path}")
    payload = _torch_load(path)
    state = _strip_uniform_prefix(_checkpoint_state(payload))
    kwargs = _affinity_model_kwargs(payload, state)
    model = MacroAffinityNet(**kwargs)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        missing = ", ".join(incompatible.missing_keys[:8]) or "none"
        unexpected = ", ".join(incompatible.unexpected_keys[:8]) or "none"
        raise RuntimeError(
            "affinity checkpoint architecture mismatch "
            f"(missing={missing}; unexpected={unexpected}; inferred={kwargs})"
        )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, _as_mapping(payload), kwargs


def checkpoint_sha256(path: str) -> str:
    """Compute a compact provenance fingerprint for the frozen candidate graph."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def make_loader(
    dataset: CanvasDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Build a loader while keeping synthetic labels exact and CPU-resident."""
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": shuffle and len(dataset) >= batch_size,
    }
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def _clean_rows_cols(perm: Tensor) -> tuple[Tensor, Tensor]:
    """Convert input-tile -> original-cell labels into exact grid coordinates."""
    if perm.ndim != 2 or perm.shape[1] != NFRAG:
        raise ValueError(f"perm must have shape (B,{NFRAG}), got {tuple(perm.shape)}")
    if torch.any(perm < 0) or torch.any(perm >= NFRAG):
        raise ValueError("perm contains an invalid clean-grid cell")
    labels = perm.long()
    return (
        torch.div(labels, GRID, rounding_mode="floor"),
        torch.remainder(labels, GRID),
    )


def candidate_offset_labels(perm: Tensor, candidates: Tensor) -> Tensor:
    """Label affinity-mined directed pairs as an exact local delta or ``far``.

    Args:
        perm: ``(B,576)`` exact synthetic input-tile -> clean-cell mapping.
        candidates: ``(B,576,K)`` ordered top-K targets for every anchor.
    """
    if candidates.ndim != 3 or candidates.shape[:2] != (perm.shape[0], NFRAG):
        raise ValueError(
            f"candidates must have shape (B,{NFRAG},K) matching perm, got {tuple(candidates.shape)}"
        )
    if torch.any(candidates < 0) or torch.any(candidates >= NFRAG):
        raise ValueError("candidate indices are outside the tile bag")
    rows, cols = _clean_rows_cols(perm)
    target_cells = perm.gather(1, candidates.reshape(perm.shape[0], -1)).reshape_as(candidates)
    target_rows = torch.div(target_cells, GRID, rounding_mode="floor")
    target_cols = torch.remainder(target_cells, GRID)
    return offsets_to_classes(
        target_rows - rows.unsqueeze(-1), target_cols - cols.unsqueeze(-1)
    )


@torch.inference_mode()
def mine_affinity_candidates(
    affinity: MacroAffinityNet,
    tiles: Tensor,
    *,
    candidate_k: int,
    device: torch.device,
    affinity_secondary: MacroAffinityNet | None = None,
) -> tuple[Tensor, Tensor]:
    """Return frozen-affinity candidates and a validity mask for every tile.

    This builds the 576x576 *frozen* cosine graph once per image.  Crucially,
    the trainable pose CNN is never run over that full 332k-pair universe; it
    sees only the sampled candidate minibatch below.  When a secondary encoder
    is supplied, this returns the ordered de-duplicated *union* of each
    encoder's top-K lists.  Its rectangular storage width is ``2*K`` while
    ``valid`` excludes duplicate entries, so no artificial duplicated pair is
    ever trained on or scored during evaluation.
    """
    if tiles.ndim != 5 or tuple(tiles.shape[1:]) != (NFRAG, 3, FS, FS):
        raise ValueError(
            f"tiles must have shape (B,{NFRAG},3,{FS},{FS}), got {tuple(tiles.shape)}"
        )
    if not 1 <= candidate_k < NFRAG:
        raise ValueError(f"candidate_k must be in [1,{NFRAG - 1}], got {candidate_k}")
    def one_encoder(model: MacroAffinityNet) -> Tensor:
        with _autocast(device):
            embeddings = model.embed(tiles)
        if tuple(embeddings.shape[:2]) != tuple(tiles.shape[:2]):
            raise RuntimeError(
                "frozen MacroAffinityNet.embed did not return one embedding per input tile: "
                f"got {tuple(embeddings.shape)}"
            )
        unit = F.normalize(embeddings.float(), p=2, dim=-1, eps=1.0e-6)
        scores = unit @ unit.transpose(-1, -2)
        torch.diagonal(scores, dim1=1, dim2=2).fill_(-torch.inf)
        return scores.topk(candidate_k, dim=-1).indices

    primary = one_encoder(affinity)
    if affinity_secondary is None:
        return primary, torch.ones_like(primary, dtype=torch.bool)
    secondary = one_encoder(affinity_secondary)
    candidates = torch.cat((primary, secondary), dim=-1)
    width = candidates.shape[-1]
    # Keep the first appearance (the primary graph is first) and mask later
    # repeats.  K is <=575 and normally 64, so this tiny 128x128 equality test
    # is vastly cheaper than the frozen 576x576 affinity itself.
    same = candidates.unsqueeze(-1).eq(candidates.unsqueeze(-2))
    prior = torch.tril(
        torch.ones((width, width), dtype=torch.bool, device=candidates.device), diagonal=-1
    )
    valid = ~(same & prior).any(dim=-1)
    return candidates, valid


def _draw_indices(
    population: Tensor,
    number: int,
    *,
    weights: Tensor | None,
    generator: torch.Generator | None,
) -> Tensor:
    """Draw candidate-row indices, using replacement only when needed."""
    if number <= 0:
        return population.new_empty((0,), dtype=torch.long)
    if population.numel() == 0:
        return population.new_empty((0,), dtype=torch.long)
    replacement = number > population.numel()
    if weights is None:
        choice = torch.randint(
            population.numel(), (number,), device=population.device, generator=generator
        ) if replacement else torch.randperm(
            population.numel(), device=population.device, generator=generator
        )[:number]
    else:
        choice = torch.multinomial(
            weights.float(), number, replacement=replacement, generator=generator
        )
    return population[choice]


def sample_candidate_pairs(
    candidates: Tensor,
    labels: Tensor,
    *,
    valid: Tensor | None = None,
    pairs_per_image: int,
    local_fraction: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample a class-balanced local/far minibatch from affinity candidates.

    Local candidate rows are weighted inversely by their exact delta-class
    frequency, so common offsets do not dominate.  Far rows are sampled
    uniformly.  The loop is only over batch images; each image considers its
    already-mined ``576*K`` candidate rows, never the all-pairs space.

    Returns ``(image_ids, anchor_indices, target_indices, class_labels)`` with
    exactly ``B * pairs_per_image`` entries unless every candidate list is
    malformed (which is treated as an error).
    """
    if candidates.ndim != 3 or labels.shape != candidates.shape:
        raise ValueError("candidates and labels must have equal (B,576,K) shapes")
    if candidates.shape[1] != NFRAG:
        raise ValueError(f"candidate axis must have {NFRAG} anchors")
    if pairs_per_image < 1:
        raise ValueError("pairs_per_image must be positive")
    if not 0.0 <= local_fraction <= 1.0:
        raise ValueError("local_fraction must lie in [0,1]")
    if valid is None:
        valid = torch.ones_like(candidates, dtype=torch.bool)
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask aligned with candidates")

    batch, _, candidate_k = candidates.shape
    anchors_template = torch.arange(NFRAG, device=candidates.device).view(NFRAG, 1)
    anchors_template = anchors_template.expand(-1, candidate_k).reshape(-1)
    requested_local = int(round(pairs_per_image * local_fraction))
    pieces: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []

    for image in range(batch):
        flat_labels = labels[image].reshape(-1).long()
        flat_targets = candidates[image].reshape(-1).long()
        flat_valid = valid[image].reshape(-1)
        local_rows = torch.nonzero(
            flat_valid & flat_labels.ne(FAR_CLASS), as_tuple=False
        ).flatten()
        far_rows = torch.nonzero(
            flat_valid & flat_labels.eq(FAR_CLASS), as_tuple=False
        ).flatten()
        local_count = requested_local if local_rows.numel() else 0
        far_count = pairs_per_image - local_count
        # If an unusually small top-K contains no far candidates, preserve the
        # fixed per-image batch size by moving that quota to available locals.
        if far_count and not far_rows.numel():
            local_count = pairs_per_image
            far_count = 0
        if not local_count and not far_count:
            raise RuntimeError("affinity candidate graph has no selectable rows")

        chosen_parts: list[Tensor] = []
        if local_count:
            local_labels = flat_labels[local_rows]
            frequencies = torch.bincount(local_labels, minlength=LOCAL_CLASS_COUNT).clamp_min(1)
            local_weights = frequencies.reciprocal().to(torch.float32)[local_labels]
            chosen_parts.append(
                _draw_indices(
                    local_rows, local_count, weights=local_weights, generator=generator
                )
            )
        if far_count:
            chosen_parts.append(_draw_indices(far_rows, far_count, weights=None, generator=generator))
        chosen = torch.cat(chosen_parts)
        # Interleave local/far samples rather than presenting the classifier an
        # ordered block of each label type.  This does not alter their weights.
        order = torch.randperm(chosen.numel(), device=chosen.device, generator=generator)
        chosen = chosen[order]
        pieces.append(
            (
                torch.full((chosen.numel(),), image, device=candidates.device, dtype=torch.long),
                anchors_template[chosen],
                flat_targets[chosen],
                flat_labels[chosen],
            )
        )

    image_ids, anchors, targets, target_labels = (
        torch.cat([piece[index] for piece in pieces], dim=0) for index in range(4)
    )
    expected = batch * pairs_per_image
    if image_ids.numel() != expected:
        raise AssertionError(f"sampled {image_ids.numel()} pairs, expected {expected}")
    return image_ids, anchors, targets, target_labels


def _hierarchical_loss_components(
    logits: Tensor, labels: Tensor, *, far_weight: float
) -> dict[str, Tensor]:
    """Return binary local/far and conditional-direction loss components.

    The raw classifier still emits 49 logits, but it is not trained with a
    flat 49-way CE: the aggregate local logit competes against far as one
    balanced binary task, then the 48-way direction CE is evaluated only for
    true local pairs.  This removes the artificial 48-to-1 class-count bias.
    """
    if logits.shape[:-1] != labels.shape or logits.shape[-1] != NUM_CLASSES:
        raise ValueError("logits must align with labels and end in 49 classes")
    if far_weight <= 0.0:
        raise ValueError("far_weight must be positive")
    flat_logits = logits.float().reshape(-1, NUM_CLASSES)
    flat_labels = labels.long().reshape(-1)
    local = flat_labels.ne(FAR_CLASS)
    aggregate = aggregate_local_logit(flat_logits)
    binary_logits = torch.stack((flat_logits[:, FAR_CLASS], aggregate), dim=-1)
    binary_targets = local.long()  # class 0=far, class 1=local
    binary_per_pair = F.cross_entropy(binary_logits, binary_targets, reduction="none")
    binary_weights = torch.where(local, 1.0, far_weight).to(binary_per_pair.dtype)
    binary_numerator = (binary_per_pair * binary_weights).sum()
    binary_denominator = binary_weights.sum().clamp_min(1.0)
    if torch.any(local):
        direction_per_pair = F.cross_entropy(
            flat_logits[local, :LOCAL_CLASS_COUNT], flat_labels[local], reduction="none"
        )
        direction_numerator = direction_per_pair.sum()
    else:
        # Keep a differentiable zero in rare all-far minibatches.
        direction_numerator = flat_logits.sum() * 0.0
    return {
        "binary_numerator": binary_numerator,
        "binary_denominator": binary_denominator,
        "direction_numerator": direction_numerator,
        "local_count": local.sum().to(binary_per_pair.dtype),
    }


def _hierarchical_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    far_weight: float,
    direction_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the balanced binary plus conditional directional training loss."""
    if direction_weight < 0.0:
        raise ValueError("direction_weight must be non-negative")
    components = _hierarchical_loss_components(logits, labels, far_weight=far_weight)
    binary_loss = components["binary_numerator"] / components["binary_denominator"]
    direction_loss = components["direction_numerator"] / components["local_count"].clamp_min(1.0)
    total = binary_loss + float(direction_weight) * direction_loss
    return total, {
        "binary_loss": binary_loss,
        "direction_loss": direction_loss,
        "local_count": components["local_count"],
    }


def _flatten_candidate_indices(candidates: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return flat image/anchor/target index tensors without materialising pixels."""
    batch, count, candidate_k = candidates.shape
    anchors = torch.arange(count, device=candidates.device).view(1, count, 1)
    anchors = anchors.expand(batch, -1, candidate_k).reshape(-1)
    image_ids = torch.arange(batch, device=candidates.device).view(batch, 1, 1)
    image_ids = image_ids.expand(-1, count, candidate_k).reshape(-1)
    return image_ids, anchors, candidates.reshape(-1).long(), valid.reshape(-1)


@torch.no_grad()
def score_candidate_graph(
    model: OffsetPoseNet,
    tiles: Tensor,
    candidates: Tensor,
    *,
    valid: Tensor | None = None,
    pair_batch: int,
    device: torch.device,
) -> Tensor:
    """Score all mined candidates in chunks, never materialising all pair pixels."""
    if pair_batch < 1:
        raise ValueError("pair_batch must be positive")
    if valid is None:
        valid = torch.ones_like(candidates, dtype=torch.bool)
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask aligned with candidates")
    image_ids, anchors, targets, flat_valid = _flatten_candidate_indices(candidates, valid)
    image_ids = image_ids[flat_valid]
    anchors = anchors[flat_valid]
    targets = targets[flat_valid]
    if not image_ids.numel():
        raise RuntimeError("candidate graph contains no valid directed pairs")
    logits: list[Tensor] = []
    for start in range(0, image_ids.numel(), pair_batch):
        stop = min(start + pair_batch, image_ids.numel())
        with _autocast(device):
            chunk = model(
                tiles[image_ids[start:stop], anchors[start:stop]],
                tiles[image_ids[start:stop], targets[start:stop]],
            )
        logits.append(chunk.float())
    scored = torch.cat(logits, dim=0)
    full = scored.new_zeros((candidates.numel(), NUM_CLASSES))
    full[flat_valid] = scored
    return full.reshape(*candidates.shape, NUM_CLASSES)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def candidate_metric_sums(
    logits: Tensor,
    candidates: Tensor,
    labels: Tensor,
    *,
    valid: Tensor | None = None,
    local_threshold: float = 0.5,
    far_weight: float = 1.0,
) -> dict[str, float]:
    """Return additive hierarchical local/far, direction, and reciprocity metrics."""
    if logits.shape[:-1] != labels.shape or logits.shape[-1] != NUM_CLASSES:
        raise ValueError("logits must be (B,576,K,49) aligned with labels")
    if candidates.shape != labels.shape:
        raise ValueError("candidates and labels must have equal shapes")
    if valid is None:
        valid = torch.ones_like(candidates, dtype=torch.bool)
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask aligned with candidates")

    decoded = hierarchical_predictions(logits, local_threshold=local_threshold)
    probabilities = decoded["raw_probabilities"]
    prediction = decoded["classes"]
    conditional_direction = decoded["conditional_offset_class"]
    predicted_local = valid & decoded["predicted_local"]
    predicted_far = valid & ~decoded["predicted_local"]
    target_probability = probabilities.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    local = valid & labels.ne(FAR_CLASS)
    far = valid & labels.eq(FAR_CLASS)
    local_detected = predicted_local & local
    far_detected = predicted_far & far
    local_count = float(local.sum())
    far_count = float(far.sum())
    pair_count = float(valid.sum())
    sums: dict[str, float] = {
        "pairs": pair_count,
        "local_pairs": local_count,
        "far_pairs": far_count,
        # Two different directional reports matter: conditional direction
        # quality does not punish a calibrated local/far threshold, while
        # exact local edge accuracy reflects the end-to-end graph decision.
        "conditional_direction_exact": float((conditional_direction.eq(labels) & local).sum()),
        "local_exact": float((prediction.eq(labels) & local).sum()),
        "local_detected": float(local_detected.sum()),
        "predicted_local": float(predicted_local.sum()),
        "far_detected": float(far_detected.sum()),
        "predicted_far": float(predicted_far.sum()),
        "edge_confidence_sum": float(decoded["confidence"].masked_select(valid).sum()),
        "local_probability_sum": float(decoded["local_probability"].masked_select(valid).sum()),
        "target_probability_sum": float(target_probability.masked_select(valid).sum()),
        "local_edge_confidence_sum": float(decoded["confidence"].masked_select(local).sum()),
        "far_edge_confidence_sum": float(decoded["confidence"].masked_select(far).sum()),
        "local_local_probability_sum": float(decoded["local_probability"].masked_select(local).sum()),
        "far_local_probability_sum": float(decoded["local_probability"].masked_select(far).sum()),
        "local_target_probability_sum": float(target_probability.masked_select(local).sum()),
        "far_target_probability_sum": float(target_probability.masked_select(far).sum()),
    }

    # Determine whether a directed top-K edge is present in both directions.
    # The temporary 576x576 integer rank map is small and lets us check
    # reciprocity without an O(K^2) search per candidate.
    batch, count, candidate_k = candidates.shape
    rank_map = torch.full(
        (batch, count, count), -1, dtype=torch.long, device=candidates.device
    )
    rank_values = torch.arange(candidate_k, device=candidates.device).view(1, 1, candidate_k)
    rank_values = rank_values.expand(batch, count, -1)
    image_ids = torch.arange(batch, device=candidates.device).view(batch, 1, 1)
    image_ids = image_ids.expand(-1, count, candidate_k)
    anchor_ids = torch.arange(count, device=candidates.device).view(1, count, 1)
    anchor_ids = anchor_ids.expand(batch, -1, candidate_k)
    rank_map[image_ids[valid], anchor_ids[valid], candidates.long()[valid]] = rank_values[valid]
    reverse_rank = rank_map[image_ids, candidates.long(), anchor_ids]
    mutual = valid & reverse_rank.ge(0)
    sums["mutual_pairs"] = float(mutual.sum())
    if torch.any(mutual):
        reverse_prediction = prediction[
            image_ids[mutual], candidates[mutual].long(), reverse_rank[mutual]
        ]
        forward_prediction = prediction[mutual]
        both_predicted_local = forward_prediction.ne(FAR_CLASS) & reverse_prediction.ne(FAR_CLASS)
        sums["mutual_both_predicted_local"] = float(both_predicted_local.sum())
        if torch.any(both_predicted_local):
            sums["mutual_inverse_consistent"] = float(
                reverse_prediction[both_predicted_local]
                .eq(inverse_classes(forward_prediction[both_predicted_local]))
                .sum()
            )
        else:
            sums["mutual_inverse_consistent"] = 0.0
        mutual_true_local = local[mutual]
        sums["mutual_true_local"] = float(mutual_true_local.sum())
        if torch.any(mutual_true_local):
            forward_labels = labels[mutual][mutual_true_local]
            reverse_labels = inverse_classes(forward_labels)
            sums["mutual_true_local_both_exact"] = float(
                forward_prediction[mutual_true_local]
                .eq(forward_labels)
                .logical_and(reverse_prediction[mutual_true_local].eq(reverse_labels))
                .sum()
            )
        else:
            sums["mutual_true_local_both_exact"] = 0.0
    else:
        sums.update(
            {
                "mutual_both_predicted_local": 0.0,
                "mutual_inverse_consistent": 0.0,
                "mutual_true_local": 0.0,
                "mutual_true_local_both_exact": 0.0,
            }
        )
    components = _hierarchical_loss_components(
        logits[valid], labels[valid], far_weight=far_weight
    )
    sums["binary_loss_numerator"] = float(components["binary_numerator"])
    sums["binary_loss_denominator"] = float(components["binary_denominator"])
    sums["direction_loss_numerator"] = float(components["direction_numerator"])
    return sums


def finalize_candidate_metrics(
    sums: Mapping[str, float], *, direction_weight: float
) -> dict[str, float]:
    """Turn additive hierarchical candidate statistics into held-out metrics."""
    if direction_weight < 0.0:
        raise ValueError("direction_weight must be non-negative")
    pairs = sums["pairs"]
    local = sums["local_pairs"]
    far = sums["far_pairs"]
    local_precision = _safe_ratio(sums["local_detected"], sums["predicted_local"])
    local_recall = _safe_ratio(sums["local_detected"], local)
    local_f1 = _safe_ratio(2.0 * local_precision * local_recall, local_precision + local_recall)
    precision = _safe_ratio(sums["far_detected"], sums["predicted_far"])
    recall = _safe_ratio(sums["far_detected"], far)
    binary_loss = _safe_ratio(sums["binary_loss_numerator"], sums["binary_loss_denominator"])
    direction_loss = _safe_ratio(sums["direction_loss_numerator"], local)
    hierarchical_loss = binary_loss + float(direction_weight) * direction_loss
    metrics = {
        # Keep candidate_cross_entropy as a compatibility alias for old log
        # consumers; it now denotes the intended hierarchical objective.
        "candidate_cross_entropy": hierarchical_loss,
        "candidate_hierarchical_loss": hierarchical_loss,
        "candidate_binary_cross_entropy": binary_loss,
        "candidate_direction_cross_entropy": direction_loss,
        "candidate_local_fraction": _safe_ratio(local, pairs),
        "binary_local_precision": local_precision,
        "binary_local_recall": local_recall,
        "binary_local_f1": local_f1,
        "binary_local_accuracy": _safe_ratio(sums["local_detected"] + sums["far_detected"], pairs),
        "conditional_direction_accuracy_local": _safe_ratio(
            sums["conditional_direction_exact"], local
        ),
        "exact_delta_accuracy_local": _safe_ratio(sums["local_exact"], local),
        "local_not_far_recall": local_recall,
        "far_detection_recall": recall,
        "far_detection_precision": precision,
        "far_detection_f1": _safe_ratio(2.0 * precision * recall, precision + recall),
        "far_vs_local_accuracy": _safe_ratio(sums["local_detected"] + sums["far_detected"], pairs),
        "mean_confidence": _safe_ratio(sums["edge_confidence_sum"], pairs),
        "mean_edge_confidence": _safe_ratio(sums["edge_confidence_sum"], pairs),
        "mean_local_probability": _safe_ratio(sums["local_probability_sum"], pairs),
        "mean_true_class_probability": _safe_ratio(sums["target_probability_sum"], pairs),
        "local_mean_confidence": _safe_ratio(sums["local_edge_confidence_sum"], local),
        "far_mean_confidence": _safe_ratio(sums["far_edge_confidence_sum"], far),
        "local_mean_local_probability": _safe_ratio(sums["local_local_probability_sum"], local),
        "far_mean_local_probability": _safe_ratio(sums["far_local_probability_sum"], far),
        "local_true_class_probability": _safe_ratio(sums["local_target_probability_sum"], local),
        "far_true_class_probability": _safe_ratio(sums["far_target_probability_sum"], far),
        "mutual_candidate_fraction": _safe_ratio(sums["mutual_pairs"], pairs),
        "mutual_predicted_local_fraction": _safe_ratio(
            sums["mutual_both_predicted_local"], sums["mutual_pairs"]
        ),
        "reciprocity_inverse_consistency": _safe_ratio(
            sums["mutual_inverse_consistent"], sums["mutual_both_predicted_local"]
        ),
        "reciprocity_both_exact_local": _safe_ratio(
            sums["mutual_true_local_both_exact"], sums["mutual_true_local"]
        ),
    }
    return metrics


@torch.no_grad()
def evaluate(
    model: OffsetPoseNet,
    affinity: MacroAffinityNet,
    loader: DataLoader,
    *,
    candidate_k: int,
    max_images: int,
    pair_batch: int,
    device: torch.device,
    affinity_secondary: MacroAffinityNet | None = None,
    local_threshold: float = 0.5,
    far_weight: float = 1.0,
    direction_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluate hierarchical local/far, direction, confidence, and reciprocity."""
    model_was_training = model.training
    model.eval()
    aggregate: defaultdict[str, float] = defaultdict(float)
    seen = 0
    for batch in loader:
        if seen >= max_images:
            break
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("offset-pose evaluation requires CanvasDataset(real_prob=0)")
        take = min(max_images - seen, int(batch["tiles"].shape[0]))
        tiles = batch["tiles"][:take].to(device, non_blocking=True)
        perm = batch["perm"][:take].to(device, non_blocking=True).long()
        candidates, valid = mine_affinity_candidates(
            affinity,
            tiles,
            candidate_k=candidate_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        labels = candidate_offset_labels(perm, candidates)
        logits = score_candidate_graph(
            model, tiles, candidates, valid=valid, pair_batch=pair_batch, device=device
        )
        for key, value in candidate_metric_sums(
            logits,
            candidates,
            labels,
            valid=valid,
            local_threshold=local_threshold,
            far_weight=far_weight,
        ).items():
            aggregate[key] += value
        seen += take
    if model_was_training:
        model.train()
    if not seen:
        raise RuntimeError("evaluation loader yielded no images")
    metrics = finalize_candidate_metrics(aggregate, direction_weight=direction_weight)
    metrics["eval_images"] = float(seen)
    return metrics


def _next_batch(
    iterator: Iterable[dict[str, Tensor]], loader: DataLoader
) -> tuple[dict[str, Tensor], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _format(metrics: Mapping[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def save_checkpoint(
    path: str,
    model: OffsetPoseNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    args: argparse.Namespace,
    metrics: Mapping[str, float],
    affinity_provenance: list[Mapping[str, Any]],
) -> None:
    """Save a self-describing pose checkpoint plus frozen-graph provenance."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "args": vars(args),
            "metrics": dict(metrics),
            "model_kwargs": {
                "tile_size": model.tile_size,
                "width": model.width,
                "dropout": model.dropout,
                "num_classes": model.num_classes,
            },
            "grid": GRID,
            "fragment_size": FS,
            "offset_radius": OFFSET_RADIUS,
            "num_classes": NUM_CLASSES,
            "far_class": FAR_CLASS,
            "class_offsets": class_offsets_metadata(),
            "prediction_mode": "hierarchical",
            "hierarchical": True,
            "hierarchical_objective": {
                "binary": "cross_entropy([far_logit, logsumexp(local_logits)])",
                "conditional_direction": "cross_entropy(local_logits) on true local pairs only",
                "direction_weight": float(args.direction_weight),
                "far_weight": float(args.far_weight),
                "local_threshold": float(args.local_threshold),
            },
            # Keep the first three fields for one-encoder consumers, while the
            # complete list records optional top-K union candidate mining.
            "affinity_checkpoint": str(affinity_provenance[0]["path"]),
            "affinity_checkpoint_sha256": str(affinity_provenance[0]["sha256"]),
            "affinity_model_kwargs": dict(affinity_provenance[0]["model_kwargs"]),
            "affinity_checkpoints": [dict(item) for item in affinity_provenance],
            "candidate_mining": {
                "candidate_k_per_encoder": int(args.candidate_k),
                "encoders": len(affinity_provenance),
                "union_deduplicate": len(affinity_provenance) > 1,
                "source": "frozen MacroAffinityNet cosine top-k excluding self",
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--affinity_ckpt",
        default=os.path.join(workspace, "artifacts", "macro_affinity", "affinity_r1_1200_best.pt"),
        help="frozen MacroAffinityNet checkpoint used to mine top-K candidates",
    )
    parser.add_argument(
        "--affinity_ckpt2",
        default=None,
        help="optional second frozen affinity checkpoint; unique top-K union is used when supplied",
    )
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--bs", type=int, default=2, help="full synthetic puzzle bags per optimization step")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train_n", type=int, default=0, help="0 uses all non-validation targets")
    parser.add_argument("--candidate_k", type=int, default=64, help="frozen affinity neighbours per anchor")
    parser.add_argument("--pairs_per_image", type=int, default=1024, help="sampled directed candidates per puzzle")
    parser.add_argument(
        "--local_fraction",
        type=float,
        default=0.50,
        help="fraction of sampled candidates reserved for exact local offsets",
    )
    parser.add_argument(
        "--far_weight",
        type=float,
        default=1.0,
        help="optional far weight in the balanced binary local-vs-far loss",
    )
    parser.add_argument(
        "--direction_weight",
        type=float,
        default=1.0,
        help="weight on conditional 48-way directional CE for true local pairs",
    )
    parser.add_argument(
        "--local_threshold",
        type=float,
        default=0.5,
        help="P(local) threshold used by held-out hierarchical decisions",
    )
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_n", type=int, default=12)
    parser.add_argument("--eval_bs", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--eval_pair_batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="offset_pose")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(workspace, "artifacts", "offset_pose"),
        help="workspace-local output checkpoint directory",
    )
    parser.add_argument("--device", default=None, help="cuda when available by default")
    args = parser.parse_args()

    if args.steps < 1 or args.bs < 1 or args.eval_n < 1 or args.eval_bs < 1:
        parser.error("--steps, --bs, --eval_n, and --eval_bs must be positive")
    if args.workers < 0 or args.train_n < 0:
        parser.error("--workers and --train_n must be non-negative")
    if not 1 <= args.candidate_k < NFRAG:
        parser.error(f"--candidate_k must lie in [1,{NFRAG - 1}]")
    if args.pairs_per_image < 1 or args.eval_pair_batch < 1:
        parser.error("--pairs_per_image and --eval_pair_batch must be positive")
    if not 0.0 <= args.local_fraction <= 1.0:
        parser.error("--local_fraction must lie in [0,1]")
    if (
        args.far_weight <= 0.0
        or args.direction_weight < 0.0
        or args.width < 4
        or args.lr <= 0.0
        or args.weight_decay < 0.0
    ):
        parser.error("invalid --far_weight/--direction_weight/--width/--lr/--weight_decay value")
    if not 0.0 <= args.local_threshold <= 1.0:
        parser.error("--local_threshold must lie in [0,1]")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must lie in [0,1)")
    if args.eval_every < 1:
        parser.error("--eval_every must be positive")
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    affinity_path = os.path.abspath(args.affinity_ckpt)
    affinity_path2 = os.path.abspath(args.affinity_ckpt2) if args.affinity_ckpt2 else None
    print(
        f"device={device} radius={OFFSET_RADIUS} classes={NUM_CLASSES} "
        f"candidate_k={args.candidate_k} sampled_pairs/image={args.pairs_per_image} "
        f"hierarchical(direction_weight={args.direction_weight:g}, threshold={args.local_threshold:g})",
        flush=True,
    )
    affinity, affinity_metadata, affinity_kwargs = load_frozen_affinity(affinity_path, device)
    del affinity_metadata  # Metadata is represented in checkpoint provenance below.
    affinity_provenance: list[Mapping[str, Any]] = [
        {
            "path": affinity_path,
            "sha256": checkpoint_sha256(affinity_path),
            "model_kwargs": dict(affinity_kwargs),
        }
    ]
    affinity_secondary: MacroAffinityNet | None = None
    if affinity_path2:
        if os.path.normcase(affinity_path) == os.path.normcase(affinity_path2):
            parser.error("--affinity_ckpt2 must differ from --affinity_ckpt")
        affinity_secondary, secondary_metadata, secondary_kwargs = load_frozen_affinity(
            affinity_path2, device
        )
        del secondary_metadata
        affinity_provenance.append(
            {
                "path": affinity_path2,
                "sha256": checkpoint_sha256(affinity_path2),
                "model_kwargs": dict(secondary_kwargs),
            }
        )
    for ordinal, provenance in enumerate(affinity_provenance, start=1):
        print(
            f"frozen affinity[{ordinal}]={provenance['path']} "
            f"sha256={str(provenance['sha256'])[:12]} kwargs={provenance['model_kwargs']}",
            flush=True,
        )
    if affinity_secondary is not None:
        print("candidate graph=deduplicated union of each encoder's top-K lists", flush=True)

    train_names, validation_names = train_val_split()
    if args.train_n:
        train_names = train_names[:args.train_n]
    if not train_names or not validation_names:
        raise RuntimeError("train/validation split is empty")
    # Exact synthetic only: the relative labels below are never recovered-cache
    # pseudo labels and never use the real corrupted input folder.
    train_dataset = CanvasDataset(train_names, real_prob=0.0, seed=args.seed)
    validation_dataset = CanvasDataset(validation_names, real_prob=0.0, seed=args.seed + 10_000)
    train_loader = make_loader(train_dataset, args.bs, args.workers, shuffle=True, device=device)
    validation_loader = make_loader(
        validation_dataset, args.eval_bs, min(args.workers, 2), shuffle=False, device=device
    )

    model = OffsetPoseNet(tile_size=FS, width=args.width, dropout=args.dropout).to(device)
    print(f"OffsetPoseNet params={count_params(model):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed + 54321)
    best = -float("inf")
    started = time.time()
    iterator = iter(train_loader)

    for step in range(1, args.steps + 1):
        batch, iterator = _next_batch(iterator, train_loader)
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("offset-pose training requires CanvasDataset(real_prob=0)")
        tiles = batch["tiles"].to(device, non_blocking=True)
        perm = batch["perm"].to(device, non_blocking=True).long()
        candidates, valid = mine_affinity_candidates(
            affinity,
            tiles,
            candidate_k=args.candidate_k,
            device=device,
            affinity_secondary=affinity_secondary,
        )
        labels = candidate_offset_labels(perm, candidates)
        image_ids, anchors, targets, sampled_labels = sample_candidate_pairs(
            candidates,
            labels,
            valid=valid,
            pairs_per_image=args.pairs_per_image,
            local_fraction=args.local_fraction,
            generator=generator,
        )
        with _autocast(device):
            logits = model(tiles[image_ids, anchors], tiles[image_ids, targets])
        loss, loss_parts = _hierarchical_loss(
            logits,
            sampled_labels,
            far_weight=args.far_weight,
            direction_weight=args.direction_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % 50 == 0:
            elapsed = time.time() - started
            local_rate = float(sampled_labels.ne(FAR_CLASS).float().mean())
            local_mask = sampled_labels.ne(FAR_CLASS)
            decoded = hierarchical_predictions(
                logits.detach(), local_threshold=args.local_threshold
            )
            conditional_exact = (
                decoded["conditional_offset_class"][local_mask]
                .eq(sampled_labels[local_mask])
                .float()
            )
            local_recall = decoded["predicted_local"][local_mask].float()
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"bin={float(loss_parts['binary_loss'].detach()):.4f} "
                f"dir={float(loss_parts['direction_loss'].detach()):.4f} "
                f"sample_local={local_rate:.3f} "
                f"sample_cond_dir={float(conditional_exact.mean()) if conditional_exact.numel() else 0.0:.3f} "
                f"sample_local_recall={float(local_recall.mean()) if local_recall.numel() else 0.0:.3f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} {elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                affinity,
                validation_loader,
                candidate_k=args.candidate_k,
                max_images=args.eval_n,
                pair_batch=args.eval_pair_batch,
                device=device,
                affinity_secondary=affinity_secondary,
                local_threshold=args.local_threshold,
                far_weight=args.far_weight,
                direction_weight=args.direction_weight,
            )
            print(f"[SYN offset-pose held-out] step={step} {_format(metrics)}", flush=True)
            last_path = os.path.join(args.out_dir, f"{args.tag}_last.pt")
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                step=step,
                args=args,
                metrics=metrics,
                affinity_provenance=affinity_provenance,
            )
            selection = metrics["exact_delta_accuracy_local"]
            if selection > best:
                best = selection
                best_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    args=args,
                    metrics=metrics,
                    affinity_provenance=affinity_provenance,
                )
                print(f"saved best exact_delta_accuracy_local={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
