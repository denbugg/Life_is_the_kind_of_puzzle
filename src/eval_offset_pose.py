"""Evaluate the directed relative-offset pose-graph experiment.

This program deliberately evaluates only *fresh exact-synthetic* puzzles.  It
never reads the recovered permutation cache from the real inputs: a clean
held-out target is newly shuffled and degraded by :class:`CanvasDataset`, so
``perm[input_tile] -> clean grid cell`` is available only for diagnostics.

The inference path is deliberately staged:

1. frozen ``MacroAffinityNet`` embeds the 576 unordered tiles and keeps the
   directed top-K candidates for every source tile;
2. ``OffsetPoseNet`` scores only those candidates, predicting a directed
   ``(delta_row, delta_col)`` plus an edge confidence;
3. optional reciprocal checks and short pose cycles reject geometrically
   inconsistent edges;
4. a small robust IRLS pose synchronization estimates continuous coordinates;
5. the largest synchronized component is translated, rounded with Hungarian,
   and scored by placement / neighbour accuracy when meaningful.

The metric printout is intended as a hard gate.  In particular, a pretty
affine probe alone is not evidence of a usable solver: inspect directed edge
exactness, reciprocal precision, cycle consistency, component coverage and the
Hungarian decode together.

Typical command::

    python src/eval_offset_pose.py ^
      --affinity-ckpt artifacts/macro_affinity/affinity_best.pt ^
      --pose-ckpt artifacts/offset_pose/offset_best.pt --n 8 --device cuda ^
      --require-reciprocal

The expected native pose-model contract is ``OffsetPoseNet(left, right)`` with
``left,right=(pairs,3,20,20)`` and logits over local offsets plus a final
``far`` class.  Loading and output parsing also accept common checkpoint and
mapping variants so interrupted experiments remain evaluable.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from eval_affinity_graph import learned_affinity, load_model as load_affinity_model, top_neighbours
from imgio import train_val_split
from placement_metrics import neighbour_accuracy


EPS = 1.0e-8


def _autocast(device: torch.device):
    """Use inexpensive fp16 inference on CUDA without complicating CPU runs."""
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def _parse_device(value: str | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return device


def _is_tensor_state_dict(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) and isinstance(item, Tensor) for key, item in value.items())
    )


def _torch_load(path: str) -> object:
    """Load trusted local checkpoints across recent torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before ``weights_only``.
        return torch.load(path, map_location="cpu")


def _checkpoint_state(payload: object) -> dict[str, Tensor]:
    """Extract a state dict from the handful of local checkpoint conventions."""
    if isinstance(payload, nn.Module):
        return dict(payload.state_dict())
    if _is_tensor_state_dict(payload):
        return dict(payload)
    if isinstance(payload, Mapping):
        for key in ("model", "model_state_dict", "state_dict", "network", "net", "pose"):
            candidate = payload.get(key)
            if isinstance(candidate, nn.Module):
                return dict(candidate.state_dict())
            if _is_tensor_state_dict(candidate):
                return dict(candidate)
    raise RuntimeError(
        "checkpoint does not contain a recognizable state dict; expected a raw "
        "state_dict or one under model/model_state_dict/state_dict"
    )


def _checkpoint_module(payload: object) -> nn.Module | None:
    if isinstance(payload, nn.Module):
        return payload
    if isinstance(payload, Mapping):
        for key in ("model", "network", "net", "pose"):
            candidate = payload.get(key)
            if isinstance(candidate, nn.Module):
                return candidate
    return None


def _strip_uniform_prefix(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Strip only unambiguous wrapper prefixes (DDP / saved wrapper modules)."""
    cleaned = dict(state)
    for prefix in ("module.", "model.", "network.", "net.", "pose."):
        keys = tuple(cleaned)
        if keys and all(key.startswith(prefix) for key in keys):
            cleaned = {key[len(prefix) :]: value for key, value in cleaned.items()}
    return cleaned


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    namespace = getattr(value, "__dict__", None)
    return namespace if isinstance(namespace, Mapping) else {}


def _metadata_sources(payload: object) -> list[Mapping[str, Any]]:
    """Return checkpoint metadata maps in decreasing precedence order."""
    root = _as_mapping(payload)
    return [
        _as_mapping(root.get("model_kwargs")),
        _as_mapping(root.get("pose_kwargs")),
        _as_mapping(root.get("args")),
        root,
    ]


def _metadata_value(payload: object, *names: str) -> object | None:
    for source in _metadata_sources(payload):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return None


def _safe_int(value: object, fallback: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _constructor_kwargs(
    cls: type[nn.Module], payload: object, radius: int,
) -> dict[str, Any]:
    """Recover only constructor keys that ``OffsetPoseNet`` actually accepts.

    The trainer checkpoint is expected to store ``model_kwargs``.  The aliases
    make the evaluator tolerant of an early checkpoint that stored arguments
    under CLI-facing names such as ``pose_width`` instead.
    """
    signature = inspect.signature(cls.__init__)
    parameters = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    aliases: dict[str, tuple[str, ...]] = {
        "radius": ("offset_radius", "pose_radius", "max_offset"),
        "offset_radius": ("radius", "pose_radius", "max_offset"),
        "max_offset": ("radius", "offset_radius", "pose_radius"),
        "width": ("pose_width", "model_width"),
        "hidden": ("hidden_dim", "pose_hidden"),
        "hidden_dim": ("hidden", "pose_hidden"),
        "dropout": ("pose_dropout",),
        "tile_size": ("fs", "fragment_size"),
        "fs": ("tile_size", "fragment_size"),
    }
    kwargs: dict[str, Any] = {}
    sources = _metadata_sources(payload)
    for name in parameters:
        value: object | None = None
        search_names = (name,) + aliases.get(name, ())
        for source in sources:
            for candidate in search_names:
                if candidate in source and source[candidate] is not None:
                    value = source[candidate]
                    break
            if value is not None:
                break
        if value is not None:
            kwargs[name] = value

    for name in ("radius", "offset_radius", "max_offset"):
        if name in parameters and name not in kwargs:
            kwargs[name] = radius
    for name in ("tile_size", "fs", "fragment_size"):
        if name in parameters and name not in kwargs:
            # Do not override an explicit model default unless the constructor
            # makes this argument required.
            parameter = parameters[name]
            if parameter.default is inspect.Parameter.empty:
                kwargs[name] = FS
    return kwargs


def load_pose_model(
    path: str, device: torch.device, radius: int,
) -> tuple[nn.Module, object, Any]:
    """Load ``OffsetPoseNet`` plus checkpoint metadata.

    The implementation intentionally does not import the trainer: inference
    should remain lightweight and never accidentally construct datasets or
    optimizers.  It only imports the model module requested by this stage.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"pose checkpoint does not exist: {path}")
    try:
        pose_module = importlib.import_module("offset_pose")
    except ImportError as exc:
        raise RuntimeError(
            "cannot import src/offset_pose.py; add OffsetPoseNet before evaluating "
            "a pose checkpoint"
        ) from exc

    payload = _torch_load(path)
    model = _checkpoint_module(payload)
    if model is None:
        cls = getattr(pose_module, "OffsetPoseNet", None)
        if cls is None or not inspect.isclass(cls):
            raise RuntimeError("offset_pose.py must export an OffsetPoseNet class")
        kwargs = _constructor_kwargs(cls, payload, radius)
        try:
            model = cls(**kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "could not reconstruct OffsetPoseNet from checkpoint metadata; "
                f"attempted kwargs={kwargs}. Save model_kwargs in the checkpoint."
            ) from exc
        state = _strip_uniform_prefix(_checkpoint_state(payload))
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            missing = ", ".join(incompatible.missing_keys[:8]) or "none"
            unexpected = ", ".join(incompatible.unexpected_keys[:8]) or "none"
            raise RuntimeError(
                "pose checkpoint architecture does not match OffsetPoseNet "
                f"(missing: {missing}; unexpected: {unexpected}; kwargs={kwargs})"
            )
    model.to(device).eval()
    return model, payload, pose_module


def _coerce_delta_table(value: object) -> np.ndarray | None:
    """Convert exported class-offset metadata to a contiguous ``(classes,2)`` table."""
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    # ``offset_pose.class_offsets_metadata()`` intentionally represents the
    # non-geometric far class as ``None``.  Turn only that sentinel into a zero
    # placeholder; ``far_class`` controls whether it is ever used as an edge.
    if isinstance(value, (list, tuple)):
        try:
            value = [([0.0, 0.0] if item is None else item) for item in value]
        except TypeError:
            pass
    try:
        table = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] == 0:
        return None
    if not np.isfinite(table).all():
        return None
    return np.ascontiguousarray(table)


def _query_offset_table(model: nn.Module, pose_module: Any, payload: object, radius: int) -> np.ndarray | None:
    """Find a model-exported offset table before falling back to row-major order."""
    names = (
        "class_offsets",
        "offsets",
        "offset_table",
        "delta_table",
        "class_deltas",
        "deltas",
    )
    for source in _metadata_sources(payload):
        for name in names:
            table = _coerce_delta_table(source.get(name))
            if table is not None:
                return table
    for owner in (model, pose_module):
        for name in names + ("class_offsets_metadata", "offsets_for_radius", "make_offset_table"):
            candidate = getattr(owner, name, None)
            if callable(candidate):
                for call_args in ((radius,), tuple()):
                    try:
                        table = _coerce_delta_table(candidate(*call_args))
                    except (TypeError, ValueError):
                        continue
                    if table is not None:
                        return table
            else:
                table = _coerce_delta_table(candidate)
                if table is not None:
                    return table
    return None


def _parse_far_index(value: str | int | None, classes: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in ("", "auto", "none", "null"):
            return None
        if lower == "last":
            return classes - 1
        if lower == "first":
            return 0
        try:
            value = int(lower)
        except ValueError as exc:
            raise ValueError("--far-index must be auto, first, last, or an integer") from exc
    result = int(value)
    if not 0 <= result < classes:
        raise ValueError(f"far class {result} is outside [0,{classes - 1}]")
    return result


def _resolve_far_index(
    requested: str | int | None,
    classes: int,
    table: np.ndarray | None,
    model: nn.Module,
    pose_module: Any,
    payload: object,
    radius: int,
) -> int | None:
    explicit = _parse_far_index(requested, classes)
    if explicit is not None:
        return explicit
    for source in _metadata_sources(payload):
        for name in ("far_class", "far_index", "nonlocal_class", "background_class"):
            if name in source:
                parsed = _safe_int(source[name])
                if parsed is not None and 0 <= parsed < classes:
                    return parsed
    for owner in (model, pose_module):
        for name in ("far_class", "far_index", "nonlocal_class", "background_class"):
            parsed = _safe_int(getattr(owner, name, None))
            if parsed is not None and 0 <= parsed < classes:
                return parsed

    nozero_count = (2 * radius + 1) ** 2 - 1
    if table is not None and table.shape[0] == classes:
        zero = np.flatnonzero(np.max(np.abs(table), axis=1) < 0.25)
        if zero.size == 1:
            return int(zero[0])
    # Native OffsetPoseNet uses the final class for far.  This also makes the
    # fallback unsurprising for a 48-local-offset + 1-far classifier.
    if classes == nozero_count + 1:
        return classes - 1
    return None


def _row_major_offsets(radius: int, *, include_zero: bool) -> np.ndarray:
    values = [
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if include_zero or dr != 0 or dc != 0
    ]
    return np.asarray(values, dtype=np.float32)


def _class_delta_table(
    classes: int,
    *,
    model: nn.Module,
    pose_module: Any,
    payload: object,
    radius: int,
    requested_far: str | int | None,
) -> tuple[np.ndarray, int | None]:
    """Return a class-indexed delta table and the inferred far-class index."""
    table = _query_offset_table(model, pose_module, payload, radius)
    far = _resolve_far_index(requested_far, classes, table, model, pose_module, payload, radius)
    if table is not None:
        if table.shape[0] == classes:
            return table, far
        if table.shape[0] == classes - 1:
            if far is None:
                far = classes - 1
            full = np.zeros((classes, 2), dtype=np.float32)
            full[np.arange(classes) != far] = table
            return full, far

    nozero = _row_major_offsets(radius, include_zero=False)
    full_grid = _row_major_offsets(radius, include_zero=True)
    if classes == nozero.shape[0]:
        return nozero, far
    if classes == nozero.shape[0] + 1:
        if far is None:
            far = classes - 1
        full = np.zeros((classes, 2), dtype=np.float32)
        full[np.arange(classes) != far] = nozero
        return full, far
    if classes == full_grid.shape[0]:
        if far is None:
            zero = np.flatnonzero(np.max(np.abs(full_grid), axis=1) < 0.25)
            far = int(zero[0]) if zero.size == 1 else None
        return full_grid, far
    raise RuntimeError(
        "cannot infer class-to-offset mapping: output has "
        f"{classes} classes, radius={radius}, and the checkpoint exports no usable "
        "class_offsets. Save class_offsets in the checkpoint or pass a compatible radius."
    )


def _as_probability(value: Tensor, count: int, name: str) -> Tensor:
    value = value.reshape(count, -1)
    if value.shape[1] != 1:
        raise RuntimeError(f"{name} must have one value per pair, got {tuple(value.shape)}")
    value = value[:, 0].float()
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains non-finite values")
    if float(value.min()) < -1.0e-4 or float(value.max()) > 1.0001:
        value = torch.sigmoid(value)
    return value.clamp(0.0, 1.0)


def _output_mapping(output: object) -> Mapping[str, Any]:
    if isinstance(output, Mapping):
        return output
    mapping = getattr(output, "_asdict", None)
    if callable(mapping):
        candidate = mapping()
        if isinstance(candidate, Mapping):
            return candidate
    namespace = getattr(output, "__dict__", None)
    return namespace if isinstance(namespace, Mapping) else {}


def _mapping_tensor(mapping: Mapping[str, Any], names: Sequence[str]) -> Tensor | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, Tensor):
            return value
    return None


@dataclass
class PosePrediction:
    delta: Tensor  # (pairs, 2), float (integer-valued for a classifier)
    confidence: Tensor  # (pairs,), estimated probability that this exact edge is right
    local_probability: Tensor  # (pairs,), probability that pair is not far when available
    predicted_local: Tensor  # (pairs,), whether this edge is geometrically admitted


def _checkpoint_decode_mode(
    payload: object,
    pose_module: Any,
    requested: str,
) -> str:
    """Choose hierarchical decoding for new checkpoints, flat decoding for old ones.

    In hierarchical mode the forty-eight local logits form one aggregate
    ``local`` alternative against the single ``far`` logit.  A legacy flat CE
    checkpoint instead used a literal 49-way global argmax, so retaining that
    fallback keeps its evaluation semantics reproducible.  New trainers should
    persist a mode marker; source-level markers make manually saved native
    models work as well.
    """
    if requested != "auto":
        return requested
    for source in _metadata_sources(payload):
        for name in (
            "prediction_mode",
            "decode_mode",
            "logit_mode",
            "loss_mode",
            "objective",
            "training_objective",
        ):
            value = source.get(name)
            if isinstance(value, str):
                lower = value.lower()
                if "hier" in lower or "aggregate" in lower or "local_vs_far" in lower:
                    return "hierarchical"
                if "flat" in lower or "49way" in lower or "cross_entropy" in lower:
                    return "flat"
        for name in ("hierarchical", "hierarchical_loss", "local_vs_far"):
            if source.get(name) is True:
                return "hierarchical"
    for name in ("PREDICTION_MODE", "DECODE_MODE", "HIERARCHICAL_OUTPUTS", "HIERARCHICAL_LOSS"):
        value = getattr(pose_module, name, None)
        if isinstance(value, str) and ("hier" in value.lower() or "aggregate" in value.lower()):
            return "hierarchical"
        if value is True:
            return "hierarchical"
    return "flat"


def _hierarchical_logits(
    logits: Tensor,
    *,
    far_index: int,
    table: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Decode native logits as local-vs-far plus a conditional local offset.

    Returns ``(delta, p_local, p_best_conditional_offset, class_index)``.  This
    must not use a global 49-way argmax: a large far logit competes with the
    *logsumexp* of all local alternatives, not any one individual direction.
    """
    classes = logits.shape[1]
    local_mask = torch.ones(classes, dtype=torch.bool, device=logits.device)
    local_mask[far_index] = False
    local_indices = torch.arange(classes, device=logits.device)[local_mask]
    local_logits = logits.float()[:, local_mask]
    far_logit = logits.float()[:, far_index]
    local_logsumexp = torch.logsumexp(local_logits, dim=-1)
    local_probability = torch.sigmoid(local_logsumexp - far_logit)
    conditional_probability = torch.softmax(local_logits, dim=-1)
    conditional_confidence, local_choice = conditional_probability.max(dim=-1)
    class_index = local_indices[local_choice]
    return table[class_index].float(), local_probability, conditional_confidence, class_index


def _prediction_from_output(
    output: object,
    *,
    count: int,
    model: nn.Module,
    pose_module: Any,
    payload: object,
    radius: int,
    far_index: str | int | None,
    local_threshold: float,
    logit_decode: str,
) -> PosePrediction:
    """Normalize several sensible pose-model output styles to one prediction.

    Hierarchical native checkpoints use ``p_local`` from the aggregate local
    evidence, then select an offset conditional on being local.  The returned
    geometric confidence is ``p_local * max p(offset | local)``: the exact
    marginal probability of the chosen directed offset.
    """
    if not 0.0 <= local_threshold <= 1.0:
        raise ValueError("local_threshold must lie in [0,1]")
    if logit_decode not in ("flat", "hierarchical"):
        raise ValueError("logit_decode must be flat or hierarchical")
    mapping = _output_mapping(output)
    logits = _mapping_tensor(
        mapping,
        ("logits", "offset_logits", "class_logits", "pose_logits", "classification_logits"),
    )
    delta = _mapping_tensor(
        mapping,
        ("delta", "offset", "offsets", "pred_delta", "predicted_delta", "mean_delta"),
    )
    confidence_tensor = _mapping_tensor(
        mapping,
        ("confidence", "conf", "edge_confidence", "geometric_confidence", "score"),
    )
    explicit_local_probability = _mapping_tensor(
        mapping,
        ("p_local", "local_probability", "local_prob", "probability_local"),
    )

    # A native logits tensor is the expected contract.  Tuple fallbacks make
    # old small experiments usable without making their forward methods a hard
    # dependency of this evaluator.
    if isinstance(output, Tensor):
        if output.ndim == 2 and output.shape[-1] == 2:
            delta = output
        else:
            logits = output
    elif not mapping and isinstance(output, (tuple, list)):
        tensors = [item for item in output if isinstance(item, Tensor)]
        for item in tensors:
            if item.ndim >= 2 and item.shape[0] == count and item.shape[-1] == 2 and delta is None:
                delta = item
            elif item.ndim >= 2 and item.shape[0] == count and logits is None:
                logits = item
            elif item.numel() == count and confidence_tensor is None:
                confidence_tensor = item

    if logits is not None:
        if logits.ndim != 2 or logits.shape[0] != count:
            raise RuntimeError(f"pose logits must have shape (pairs, classes), got {tuple(logits.shape)}")
        if logits.shape[1] == 2 and delta is None:
            # A two-vector output is much more plausibly direct regression than
            # a binary classifier in this pose context.
            delta = logits
            logits = None

    class_confidence: Tensor | None = None
    local_probability: Tensor | None = None
    classifier_has_far = False
    predicted_local: Tensor | None = None
    if logits is not None:
        table, inferred_far = _class_delta_table(
            int(logits.shape[1]),
            model=model,
            pose_module=pose_module,
            payload=payload,
            radius=radius,
            requested_far=far_index,
        )
        table_t = torch.as_tensor(table, device=logits.device, dtype=logits.dtype)
        probabilities = torch.softmax(logits.float(), dim=-1)
        if inferred_far is not None:
            classifier_has_far = True
            if logit_decode == "hierarchical":
                class_delta, local_probability, conditional_confidence, class_index = _hierarchical_logits(
                    logits,
                    far_index=inferred_far,
                    table=table_t,
                )
                class_confidence = local_probability * conditional_confidence
                predicted_local = local_probability >= local_threshold
            else:
                # Legacy flat CE fallback: preserve its original global class
                # decision.  It remains useful for inspecting old checkpoints,
                # while new hierarchical checkpoints should use the branch above.
                class_confidence, class_index = probabilities.max(dim=-1)
                local_probability = 1.0 - probabilities[:, inferred_far]
                predicted_local = class_index.ne(inferred_far)
                class_delta = table_t[class_index].float()
        else:
            class_confidence, class_index = probabilities.max(dim=-1)
            local_probability = torch.ones_like(class_confidence)
            predicted_local = torch.ones_like(class_confidence, dtype=torch.bool)
            class_delta = table_t[class_index].float()
        # Native hierarchical logits are the scoring contract for this stage;
        # decode their conditional direction ourselves rather than trusting an
        # auxiliary/global-argmax field from a generic wrapper.
        if delta is None or (classifier_has_far and logit_decode == "hierarchical"):
            delta = class_delta

    if delta is None:
        available = ", ".join(mapping.keys()) if mapping else type(output).__name__
        raise RuntimeError(
            "cannot find pose logits or a (pairs,2) delta in model output; "
            f"available output fields: {available}"
        )
    if delta.ndim != 2 or tuple(delta.shape) != (count, 2):
        raise RuntimeError(f"pose delta must have shape ({count},2), got {tuple(delta.shape)}")
    delta = delta.float()
    if not torch.isfinite(delta).all():
        raise RuntimeError("pose delta contains non-finite values")

    external_confidence = (
        _as_probability(confidence_tensor, count, "pose confidence")
        if confidence_tensor is not None
        else None
    )
    external_local_probability = (
        _as_probability(explicit_local_probability, count, "local probability")
        if explicit_local_probability is not None
        else None
    )
    if external_local_probability is not None:
        # Native ``predict`` helpers may expose p_local directly.  Prefer it
        # over a recomputation when the raw logits are absent; when raw logits
        # are present, both routes should agree up to numerical roundoff.
        if local_probability is None:
            local_probability = external_local_probability
        elif not classifier_has_far:
            # A mapping with only 48 conditional offset logits plus an explicit
            # p_local is another valid hierarchical representation.
            local_probability = external_local_probability
            if class_confidence is not None:
                class_confidence = class_confidence * external_local_probability
        if logit_decode == "hierarchical":
            predicted_local = local_probability >= local_threshold

    if class_confidence is not None and external_confidence is not None:
        # A separate binary local confidence plus conditional-offset logits
        # needs multiplication.  A full classifier with an explicit far class
        # already gives a joint class probability and should not be multiplied
        # again.
        confidence = class_confidence if classifier_has_far else class_confidence * external_confidence
    elif class_confidence is not None:
        confidence = class_confidence
    elif external_confidence is not None:
        confidence = external_confidence
    else:
        # Regression-only checkpoints have no calibrated uncertainty.  Keep
        # them evaluable but make the fact visible in the report via all edges
        # receiving confidence 1.0.
        confidence = torch.ones(count, device=delta.device, dtype=torch.float32)

    if local_probability is None:
        local_probability = torch.ones_like(confidence)
    if predicted_local is None:
        # Regression / generic mapping-only fallback.  If it exports p_local,
        # honour the same threshold; otherwise confidence is the only gate.
        predicted_local = local_probability >= local_threshold
    return PosePrediction(
        delta=delta,
        confidence=confidence.float(),
        local_probability=local_probability.float(),
        predicted_local=predicted_local.bool(),
    )


def _call_pose_model(
    model: nn.Module,
    left: Tensor,
    right: Tensor,
    *,
    local_threshold: float,
) -> object:
    """Call several conventional pair interfaces, preferring the native one."""
    attempts: list[tuple[str, Any]] = []
    for name in ("score_pairs", "predict_pairs", "forward_pairs"):
        method = getattr(model, name, None)
        if callable(method):
            attempts.append((f"{name}(left,right)", lambda method=method: method(left, right)))
    predict = getattr(model, "predict", None)
    if callable(predict):
        # New native OffsetPoseNet exposes this exact semantic helper.  Keep a
        # no-keyword fallback for older generic models with a simpler predict.
        attempts.append(
            (
                "predict(left,right,local_threshold=...)",
                lambda: predict(left, right, local_threshold=local_threshold),
            )
        )
        attempts.append(("predict(left,right)", lambda: predict(left, right)))
    attempts.extend(
        [
            ("model(left,right)", lambda: model(left, right)),
            ("model(stack(left,right))", lambda: model(torch.stack((left, right), dim=1))),
            ("model(cat(left,right))", lambda: model(torch.cat((left, right), dim=1))),
        ]
    )
    errors: list[str] = []
    for label, call in attempts:
        try:
            return call()
        except (TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:180]}")
    detail = " | ".join(errors[-3:])
    raise RuntimeError(
        "could not call OffsetPoseNet on tile pairs. Expected OffsetPoseNet(left,right) "
        f"or a score_pairs/predict_pairs method. Last attempts: {detail}"
    )


def mine_directed_candidates(
    affinity_models: Sequence[nn.Module],
    tiles: Tensor,
    *,
    top_k: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the de-duplicated directed union of one or two top-K graphs.

    The secondary encoder is optional but deliberately contributes another
    *top-K*, not a larger top-K from the first encoder.  This mirrors the pose
    trainer's candidate-distribution contract and preserves the primary graph's
    first appearance whenever a pair is present in both lists.
    """
    if not 1 <= top_k < NFRAG:
        raise ValueError(f"top_k must be in [1,{NFRAG - 1}], got {top_k}")
    if not affinity_models:
        raise ValueError("at least one frozen affinity model is required")
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for affinity_model in affinity_models:
        affinity = learned_affinity(affinity_model, tiles, device)
        neighbours = top_neighbours(affinity, top_k)
        source_parts.append(
            torch.arange(NFRAG, device=neighbours.device).repeat_interleave(top_k).cpu().numpy()
        )
        target_parts.append(neighbours.reshape(-1).cpu().numpy())
    source = np.concatenate(source_parts)
    target = np.concatenate(target_parts)
    encoded = source.astype(np.int64) * NFRAG + target.astype(np.int64)
    # np.unique returns first positions but sorts by key; sorting those indices
    # restores deterministic primary-first order before the pair CNN sees them.
    _, first = np.unique(encoded, return_index=True)
    first.sort()
    return source[first], target[first]


def score_candidates(
    pose_model: nn.Module,
    pose_module: Any,
    payload: object,
    tiles: Tensor,
    source: np.ndarray,
    target: np.ndarray,
    *,
    device: torch.device,
    radius: int,
    far_index: str | int | None,
    local_threshold: float,
    logit_decode: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the cross-encoder in bounded chunks over directed affinity candidates."""
    if source.ndim != 1 or target.ndim != 1 or source.shape != target.shape:
        raise ValueError("source and target must be equally shaped one-dimensional arrays")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    count = int(source.size)
    if count == 0:
        return (
            np.empty((0, 2), np.float32),
            np.empty(0, np.float32),
            np.empty(0, np.float32),
            np.empty(0, bool),
        )
    if tuple(tiles.shape) != (NFRAG, 3, FS, FS):
        raise ValueError(f"tiles must be ({NFRAG},3,{FS},{FS}), got {tuple(tiles.shape)}")

    source_t = torch.from_numpy(source.astype(np.int64, copy=False)).to(device)
    target_t = torch.from_numpy(target.astype(np.int64, copy=False)).to(device)
    tile_bank = tiles.to(device, non_blocking=device.type == "cuda")
    deltas: list[Tensor] = []
    confidences: list[Tensor] = []
    local_probabilities: list[Tensor] = []
    predicted_locals: list[Tensor] = []
    with torch.inference_mode(), _autocast(device):
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            output = _call_pose_model(
                pose_model,
                tile_bank[source_t[start:stop]],
                tile_bank[target_t[start:stop]],
                local_threshold=local_threshold,
            )
            prediction = _prediction_from_output(
                output,
                count=stop - start,
                model=pose_model,
                pose_module=pose_module,
                payload=payload,
                radius=radius,
                far_index=far_index,
                local_threshold=local_threshold,
                logit_decode=logit_decode,
            )
            deltas.append(prediction.delta.detach().float().cpu())
            confidences.append(prediction.confidence.detach().float().cpu())
            local_probabilities.append(prediction.local_probability.detach().float().cpu())
            predicted_locals.append(prediction.predicted_local.detach().bool().cpu())
    return (
        torch.cat(deltas).numpy(),
        torch.cat(confidences).numpy(),
        torch.cat(local_probabilities).numpy(),
        torch.cat(predicted_locals).numpy(),
    )


@dataclass
class EdgeMatrices:
    candidate: np.ndarray  # (N,N) bool
    delta: np.ndarray  # (N,N,2) float, nan for not-candidate
    confidence: np.ndarray  # (N,N) float
    local_probability: np.ndarray  # (N,N) float
    predicted_local: np.ndarray  # (N,N) bool; classifier says this pair is local
    selected: np.ndarray  # thresholded directed edge
    reciprocal: np.ndarray  # thresholded and inverse-consistent directed edge
    used: np.ndarray  # selected, or reciprocal when requested


def build_edge_matrices(
    source: np.ndarray,
    target: np.ndarray,
    delta: np.ndarray,
    confidence: np.ndarray,
    local_probability: np.ndarray,
    predicted_local: np.ndarray,
    *,
    min_confidence: float,
    max_predicted_offset: float,
    reciprocal_tolerance: float,
    require_reciprocal: bool,
) -> EdgeMatrices:
    """Materialize the sparse directed candidate graph as small 576x576 arrays."""
    count = int(source.size)
    if (
        target.shape != (count,)
        or confidence.shape != (count,)
        or local_probability.shape != (count,)
        or predicted_local.shape != (count,)
        or delta.shape != (count, 2)
    ):
        raise ValueError("candidate score arrays have inconsistent shapes")
    if np.any(source < 0) or np.any(source >= NFRAG) or np.any(target < 0) or np.any(target >= NFRAG):
        raise ValueError("candidate indices fall outside the puzzle")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must lie in [0,1]")
    if reciprocal_tolerance < 0.0:
        raise ValueError("reciprocal_tolerance must be non-negative")

    candidate = np.zeros((NFRAG, NFRAG), dtype=bool)
    all_delta = np.full((NFRAG, NFRAG, 2), np.nan, dtype=np.float32)
    all_confidence = np.zeros((NFRAG, NFRAG), dtype=np.float32)
    all_local_probability = np.zeros((NFRAG, NFRAG), dtype=np.float32)
    all_predicted_local = np.zeros((NFRAG, NFRAG), dtype=bool)
    candidate[source, target] = True
    all_delta[source, target] = delta
    all_confidence[source, target] = np.clip(confidence, 0.0, 1.0)
    all_local_probability[source, target] = np.clip(local_probability, 0.0, 1.0)
    all_predicted_local[source, target] = predicted_local.astype(bool, copy=False)

    finite = np.isfinite(all_delta).all(axis=-1)
    magnitude = np.max(np.abs(np.nan_to_num(all_delta, nan=0.0)), axis=-1)
    selected = (
        candidate
        & finite
        & all_predicted_local
        & (all_confidence >= min_confidence)
        & (magnitude >= 0.5)
        & (magnitude <= max_predicted_offset)
    )
    np.fill_diagonal(selected, False)
    both = selected & selected.T
    inverse_error = np.max(
        np.abs(np.nan_to_num(all_delta, nan=1.0e9) + np.nan_to_num(all_delta.transpose(1, 0, 2), nan=1.0e9)),
        axis=-1,
    )
    reciprocal = both & (inverse_error <= reciprocal_tolerance)
    used = reciprocal.copy() if require_reciprocal else selected.copy()
    return EdgeMatrices(
        candidate=candidate,
        delta=all_delta,
        confidence=all_confidence,
        local_probability=all_local_probability,
        predicted_local=all_predicted_local,
        selected=selected,
        reciprocal=reciprocal,
        used=used,
    )


def _truth_geometry(perm: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return input-tile coordinates, directed true deltas and direct-neighbour mask."""
    values = perm.detach().cpu().numpy().astype(np.int64, copy=False)
    if values.shape != (NFRAG,) or not np.array_equal(np.sort(values), np.arange(NFRAG)):
        raise ValueError("synthetic perm is not a valid input-tile -> clean-cell bijection")
    coords = np.stack((values // GRID, values % GRID), axis=-1).astype(np.float32)
    delta = coords[None, :, :] - coords[:, None, :]
    direct = np.abs(delta).sum(axis=-1) == 1.0
    return coords, delta, direct


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


def directed_edge_metrics(edge: EdgeMatrices, perm: Tensor, radius: int) -> dict[str, float]:
    """Label-aware diagnostics; labels are never fed back into inference."""
    _, truth, direct = _truth_geometry(perm)
    offdiag = ~np.eye(NFRAG, dtype=bool)
    true_cheb = np.max(np.abs(truth), axis=-1)
    local = offdiag & (true_cheb <= float(radius))
    exact = edge.candidate & (np.max(np.abs(edge.delta - truth), axis=-1) <= 0.25)

    candidate_count = float(edge.candidate.sum())
    selected_count = float(edge.selected.sum())
    reciprocal_count = float(edge.reciprocal.sum())
    used_count = float(edge.used.sum())
    direct_count = float(direct.sum())
    local_count = float(local.sum())

    def precision(mask: np.ndarray, condition: np.ndarray) -> float:
        return _ratio(float((mask & condition).sum()), float(mask.sum()))

    metrics = {
        "candidates_per_tile": _ratio(candidate_count, float(NFRAG)),
        "candidate_direct_coverage": _ratio(float((edge.candidate & direct).sum()), direct_count),
        "candidate_local_precision": precision(edge.candidate, local),
        "candidate_local_recall": _ratio(float((edge.candidate & local).sum()), local_count),
        "predicted_local_fraction_of_candidates": _ratio(
            float((edge.candidate & edge.predicted_local).sum()), candidate_count
        ),
        "predicted_local_true_local_precision": precision(edge.candidate & edge.predicted_local, local),
        "pose_exact_on_local_candidates": _ratio(
            float((exact & local).sum()), float((edge.candidate & local).sum())
        ),
        "selected_fraction_of_candidates": _ratio(selected_count, candidate_count),
        # This is the primary directed edge-exactness gate: selected means the
        # model itself considered the edge usable; exact compares its full
        # predicted vector only after inference is complete.
        "edge_exact_precision": precision(edge.selected, exact),
        "edge_direct_precision": precision(edge.selected, direct),
        "edge_exact_direct_recall": _ratio(float((edge.selected & exact & direct).sum()), direct_count),
        "reciprocal_coverage_of_selected": _ratio(reciprocal_count, selected_count),
        "reciprocal_edge_exact_precision": precision(edge.reciprocal, exact),
        "reciprocal_direct_precision": precision(edge.reciprocal, direct),
        "reciprocal_exact_direct_recall": _ratio(
            float((edge.reciprocal & exact & direct).sum()), direct_count
        ),
        "used_edges_per_tile": _ratio(used_count, float(NFRAG)),
        "used_edge_exact_precision": precision(edge.used, exact),
        "used_direct_precision": precision(edge.used, direct),
        "used_exact_direct_recall": _ratio(float((edge.used & exact & direct).sum()), direct_count),
    }
    return metrics


def cycle_metrics(edge: EdgeMatrices, *, degree: int, tolerance: float) -> dict[str, float]:
    """Measure inferred short cycles without consulting any clean coordinates."""
    if degree < 1:
        raise ValueError("cycle degree must be positive")
    if tolerance < 0.0:
        raise ValueError("cycle tolerance must be non-negative")
    usable = edge.used
    neighbours: list[np.ndarray] = []
    for source in range(NFRAG):
        target = np.flatnonzero(usable[source])
        if target.size:
            target = target[np.argsort(edge.confidence[source, target])[::-1][:degree]]
        neighbours.append(target)

    errors: list[float] = []
    for source, middle_nodes in enumerate(neighbours):
        for middle in middle_nodes:
            for target in neighbours[int(middle)]:
                if target == source or not usable[source, target]:
                    continue
                residual = edge.delta[source, middle] + edge.delta[middle, target] - edge.delta[source, target]
                errors.append(float(np.max(np.abs(residual))))
    if not errors:
        return {
            "cycle_triangles": 0.0,
            "cycle_consistent_fraction": float("nan"),
            "cycle_mean_linf_error": float("nan"),
            "cycle_median_linf_error": float("nan"),
        }
    values = np.asarray(errors, dtype=np.float64)
    return {
        "cycle_triangles": float(values.size),
        "cycle_consistent_fraction": float(np.mean(values <= tolerance)),
        "cycle_mean_linf_error": float(values.mean()),
        "cycle_median_linf_error": float(np.median(values)),
    }


@dataclass
class PoseConstraints:
    source: np.ndarray
    target: np.ndarray
    delta: np.ndarray
    weight: np.ndarray


def collapse_constraints(edge: EdgeMatrices, *, reciprocal_tolerance: float) -> PoseConstraints:
    """Collapse directed edges to one robust undirected pose constraint per pair."""
    sources: list[int] = []
    targets: list[int] = []
    deltas: list[np.ndarray] = []
    weights: list[float] = []
    for source in range(NFRAG):
        # Consider either orientation of every unordered pair.  Restricting the
        # loop to outgoing edges would accidentally discard a valid reverse-only
        # constraint when reciprocal filtering is disabled.
        forward_tail = edge.used[source, source + 1 :]
        backward_tail = edge.used[source + 1 :, source]
        for target in np.flatnonzero(forward_tail | backward_tail) + source + 1:
            target = int(target)
            forward = bool(edge.used[source, target])
            backward = bool(edge.used[target, source])
            if not forward and not backward:
                continue
            if forward and backward:
                inverse_error = float(np.max(np.abs(edge.delta[source, target] + edge.delta[target, source])))
                if inverse_error <= reciprocal_tolerance:
                    observation = 0.5 * (edge.delta[source, target] - edge.delta[target, source])
                    confidence = math.sqrt(
                        float(edge.confidence[source, target]) * float(edge.confidence[target, source])
                    )
                elif edge.confidence[source, target] >= edge.confidence[target, source]:
                    observation = edge.delta[source, target]
                    confidence = float(edge.confidence[source, target])
                else:
                    observation = -edge.delta[target, source]
                    confidence = float(edge.confidence[target, source])
            elif forward:
                observation = edge.delta[source, target]
                confidence = float(edge.confidence[source, target])
            else:
                observation = -edge.delta[target, source]
                confidence = float(edge.confidence[target, source])
            sources.append(source)
            targets.append(target)
            deltas.append(np.asarray(observation, dtype=np.float64))
            weights.append(max(float(confidence), 1.0e-3))
    if not sources:
        return PoseConstraints(
            source=np.empty(0, dtype=np.int64),
            target=np.empty(0, dtype=np.int64),
            delta=np.empty((0, 2), dtype=np.float64),
            weight=np.empty(0, dtype=np.float64),
        )
    return PoseConstraints(
        source=np.asarray(sources, dtype=np.int64),
        target=np.asarray(targets, dtype=np.int64),
        delta=np.asarray(deltas, dtype=np.float64),
        weight=np.asarray(weights, dtype=np.float64),
    )


def _components(constraints: PoseConstraints) -> list[np.ndarray]:
    parent = np.arange(NFRAG, dtype=np.int64)
    size = np.ones(NFRAG, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for source, target in zip(constraints.source, constraints.target):
        left, right = find(int(source)), find(int(target))
        if left == right:
            continue
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]
    groups: dict[int, list[int]] = defaultdict(list)
    for node in range(NFRAG):
        groups[find(node)].append(node)
    return [np.asarray(group, dtype=np.int64) for group in sorted(groups.values(), key=len, reverse=True)]


def _solve_weighted_pose(
    constraints: PoseConstraints,
    weights: np.ndarray,
    components: Sequence[np.ndarray],
) -> np.ndarray:
    """Solve a two-coordinate weighted graph Laplacian with one anchor/component."""
    laplacian = np.zeros((NFRAG, NFRAG), dtype=np.float64)
    rhs = np.zeros((NFRAG, 2), dtype=np.float64)
    for source, target, delta, weight in zip(
        constraints.source, constraints.target, constraints.delta, weights
    ):
        source, target = int(source), int(target)
        laplacian[source, source] += weight
        laplacian[target, target] += weight
        laplacian[source, target] -= weight
        laplacian[target, source] -= weight
        rhs[source] -= weight * delta
        rhs[target] += weight * delta

    # Every connected component needs a gauge anchor.  Anchoring a maximum
    # degree node is slightly better conditioned than always anchoring index 0.
    degree = np.diag(laplacian)
    for component in components:
        anchor = int(component[np.argmax(degree[component])])
        laplacian[anchor, anchor] += 1.0
    try:
        return np.linalg.solve(laplacian, rhs)
    except np.linalg.LinAlgError:
        # Should occur only with pathological zero / NaN weights, but a least
        # squares fallback gives a diagnostic instead of killing a long run.
        solution, *_ = np.linalg.lstsq(laplacian, rhs, rcond=None)
        return solution


@dataclass
class SyncResult:
    coordinates: np.ndarray
    components: list[np.ndarray]
    largest_component: np.ndarray
    residual: np.ndarray
    robust_weight: np.ndarray


def synchronize_coordinates(
    constraints: PoseConstraints,
    *,
    iterations: int,
    huber: float,
) -> SyncResult:
    """Robust IRLS synchronization for ``x_j - x_i = delta_ij`` constraints."""
    if iterations < 1:
        raise ValueError("sync iterations must be positive")
    if huber <= 0.0:
        raise ValueError("sync huber must be positive")
    components = _components(constraints)
    largest = components[0] if components else np.empty(0, dtype=np.int64)
    if constraints.source.size == 0:
        return SyncResult(
            coordinates=np.full((NFRAG, 2), np.nan, dtype=np.float64),
            components=components,
            largest_component=largest,
            residual=np.empty(0, dtype=np.float64),
            robust_weight=np.empty(0, dtype=np.float64),
        )

    base_weight = constraints.weight.astype(np.float64, copy=True)
    robust_weight = base_weight.copy()
    coordinates = np.zeros((NFRAG, 2), dtype=np.float64)
    residual = np.empty(constraints.source.size, dtype=np.float64)
    for _ in range(iterations):
        coordinates = _solve_weighted_pose(constraints, robust_weight, components)
        error = (
            coordinates[constraints.target]
            - coordinates[constraints.source]
            - constraints.delta
        )
        residual = np.linalg.norm(error, axis=1)
        robust_weight = base_weight * np.minimum(1.0, huber / np.maximum(residual, EPS))
    return SyncResult(
        coordinates=coordinates,
        components=components,
        largest_component=largest,
        residual=residual,
        robust_weight=robust_weight,
    )


def affine_coordinate_r2(coordinates: np.ndarray, truth_coordinates: np.ndarray, indices: np.ndarray) -> float:
    """Affine diagnostic only: fit estimated coordinates to true rows/columns."""
    if indices.size < 4:
        return float("nan")
    estimate = coordinates[indices]
    target = truth_coordinates[indices].astype(np.float64)
    if not np.isfinite(estimate).all():
        return float("nan")
    design = np.column_stack((np.ones(indices.size), estimate))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    residual = np.square(target - prediction).sum(axis=0)
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum(axis=0)
    if np.any(total <= EPS):
        return float("nan")
    return float(np.mean(1.0 - residual / total))


def _translation_for_grid(coordinates: np.ndarray) -> np.ndarray:
    """Choose a label-free integer translation that best keeps points in 0..23."""
    shift = np.zeros(2, dtype=np.float64)
    for axis in range(2):
        values = coordinates[:, axis]
        lo = int(math.floor(-float(np.max(values)))) - 2
        hi = int(math.ceil((GRID - 1) - float(np.min(values)))) + 2
        # Very bad early models can produce enormous coordinates.  There is no
        # value in scanning their entire range; the centered fallback remains a
        # deterministic diagnostic.
        lo, hi = max(lo, -96), min(hi, 96)
        candidates = np.arange(lo, hi + 1, dtype=np.float64)
        shifted = values[:, None] + candidates[None, :]
        outside = np.maximum(0.0, -shifted) + np.maximum(0.0, shifted - (GRID - 1))
        loss = np.square(outside).sum(axis=0)
        best = np.flatnonzero(loss <= loss.min() + 1.0e-10)
        if best.size > 1:
            # If a partial component fits at several translations, center it;
            # the report retains component coverage so this is never mistaken
            # for an absolute-coordinate oracle.
            centered = np.abs(shifted[:, best].mean(axis=0) - (GRID - 1) / 2.0)
            best_index = int(best[int(np.argmin(centered))])
        else:
            best_index = int(best[0])
        shift[axis] = candidates[best_index]
    return shift


@dataclass
class DecodeResult:
    place: np.ndarray
    assigned_tile: np.ndarray
    assigned_position: np.ndarray
    shift: np.ndarray


def hungarian_round_largest_component(sync: SyncResult) -> DecodeResult | None:
    """Assign the largest coherent relative component to the 24x24 grid.

    Other components have independent translation gauges, so forcing them into
    the same assignment would manufacture a misleading global score.  They are
    filled deterministically only to keep ``place`` a valid permutation for the
    conservative whole-puzzle neighbour metric.
    """
    tiles = sync.largest_component
    if tiles.size < 2:
        return None
    coords = sync.coordinates[tiles]
    if not np.isfinite(coords).all():
        return None
    shift = _translation_for_grid(coords)
    shifted = coords + shift[None, :]
    grid = np.stack(np.divmod(np.arange(NFRAG, dtype=np.int64), GRID), axis=-1).astype(np.float64)
    cost = np.square(shifted[:, None, :] - grid[None, :, :]).sum(axis=-1)
    rows, positions = linear_sum_assignment(cost)
    assigned_tile = tiles[rows]
    place = np.full(NFRAG, -1, dtype=np.int64)
    place[positions] = assigned_tile
    remaining_positions = np.flatnonzero(place < 0)
    remaining_tiles = np.setdiff1d(np.arange(NFRAG, dtype=np.int64), assigned_tile, assume_unique=False)
    place[remaining_positions] = remaining_tiles
    if not np.array_equal(np.sort(place), np.arange(NFRAG)):
        raise AssertionError("Hungarian rounding did not produce a tile permutation")
    return DecodeResult(
        place=place,
        assigned_tile=assigned_tile,
        assigned_position=positions.astype(np.int64),
        shift=shift,
    )


def synchronization_metrics(
    sync: SyncResult,
    constraints: PoseConstraints,
    perm: Tensor,
) -> dict[str, float]:
    truth_coordinates, _, _ = _truth_geometry(perm)
    result: dict[str, float] = {
        "sync_constraints": float(constraints.source.size),
        "sync_components": float(len(sync.components)),
        "sync_largest_component_fraction": _ratio(float(sync.largest_component.size), float(NFRAG)),
        "sync_residual_mean": float(sync.residual.mean()) if sync.residual.size else float("nan"),
        "sync_residual_median": float(np.median(sync.residual)) if sync.residual.size else float("nan"),
        "sync_affine_coordinate_r2": affine_coordinate_r2(
            sync.coordinates, truth_coordinates, sync.largest_component
        ),
    }
    decoded = hungarian_round_largest_component(sync)
    if decoded is None:
        result.update(
            {
                "hungarian_component_placement": float("nan"),
                "hungarian_whole_placement": float("nan"),
                "hungarian_whole_neighbour": float("nan"),
                "hungarian_shift_row": float("nan"),
                "hungarian_shift_col": float("nan"),
            }
        )
        return result
    inverse = np.argsort(perm.detach().cpu().numpy().astype(np.int64, copy=False))
    result.update(
        {
            "hungarian_component_placement": float(
                np.mean(decoded.assigned_position == perm.detach().cpu().numpy()[decoded.assigned_tile])
            ),
            "hungarian_whole_placement": float(np.mean(decoded.place == inverse)),
            "hungarian_whole_neighbour": float(neighbour_accuracy(decoded.place, inverse)[0]),
            "hungarian_shift_row": float(decoded.shift[0]),
            "hungarian_shift_col": float(decoded.shift[1]),
        }
    )
    return result


def evaluate_one(
    affinity_models: Sequence[nn.Module],
    pose_model: nn.Module,
    pose_module: Any,
    pose_payload: object,
    sample: Mapping[str, Tensor],
    *,
    device: torch.device,
    top_k: int,
    radius: int,
    far_index: str | int | None,
    local_threshold: float,
    logit_decode: str,
    chunk_size: int,
    min_confidence: float,
    max_predicted_offset: float,
    reciprocal_tolerance: float,
    require_reciprocal: bool,
    cycle_degree: int,
    cycle_tolerance: float,
    sync_iterations: int,
    sync_huber: float,
) -> dict[str, float]:
    """Run all non-training diagnostics for one exact synthetic puzzle."""
    if not bool(sample["has_perm"]):
        raise RuntimeError("offset-pose evaluator requires exact synthetic examples only")
    tiles = sample["tiles"]
    perm = sample["perm"]
    source, target = mine_directed_candidates(
        affinity_models, tiles, top_k=top_k, device=device
    )
    delta, confidence, local_probability, predicted_local = score_candidates(
        pose_model,
        pose_module,
        pose_payload,
        tiles,
        source,
        target,
        device=device,
        radius=radius,
        far_index=far_index,
        local_threshold=local_threshold,
        logit_decode=logit_decode,
        chunk_size=chunk_size,
    )
    edge = build_edge_matrices(
        source,
        target,
        delta,
        confidence,
        local_probability,
        predicted_local,
        min_confidence=min_confidence,
        max_predicted_offset=max_predicted_offset,
        reciprocal_tolerance=reciprocal_tolerance,
        require_reciprocal=require_reciprocal,
    )
    constraints = collapse_constraints(edge, reciprocal_tolerance=reciprocal_tolerance)
    sync = synchronize_coordinates(
        constraints, iterations=sync_iterations, huber=sync_huber
    )
    metrics = directed_edge_metrics(edge, perm, radius)
    metrics.update(cycle_metrics(edge, degree=cycle_degree, tolerance=cycle_tolerance))
    metrics.update(synchronization_metrics(sync, constraints, perm))
    return metrics


def _mean_metrics(totals: Mapping[str, float], counts: Mapping[str, int]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, total in totals.items():
        count = counts.get(name, 0)
        result[name] = float(total / count) if count else float("nan")
    return result


def _add_metrics(totals: dict[str, float], counts: dict[str, int], metrics: Mapping[str, float]) -> None:
    for name, value in metrics.items():
        if np.isfinite(value):
            totals[name] = totals.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1
        else:
            totals.setdefault(name, 0.0)
            counts.setdefault(name, 0)


def _fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def print_image_report(index: int, metrics: Mapping[str, float]) -> None:
    print(
        f"[{index:03d}] candidate direct_cov={_fmt(metrics['candidate_direct_coverage'])} "
        f"local_p/r={_fmt(metrics['candidate_local_precision'])}/{_fmt(metrics['candidate_local_recall'])} "
        f"pred_local={_fmt(metrics['predicted_local_fraction_of_candidates'])} "
        f"edge_exact={_fmt(metrics['edge_exact_precision'])} "
        f"selected/tile={_fmt(metrics['used_edges_per_tile'])}",
        flush=True,
    )
    print(
        f"      reciprocal coverage={_fmt(metrics['reciprocal_coverage_of_selected'])} "
        f"exact={_fmt(metrics['reciprocal_edge_exact_precision'])} "
        f"cycle={_fmt(metrics['cycle_consistent_fraction'])} "
        f"triangles={metrics['cycle_triangles']:.0f}",
        flush=True,
    )
    print(
        f"      sync largest={_fmt(metrics['sync_largest_component_fraction'])} "
        f"R2={_fmt(metrics['sync_affine_coordinate_r2'])} "
        f"resid={_fmt(metrics['sync_residual_median'])} "
        f"Hungarian component/full place={_fmt(metrics['hungarian_component_placement'])}/"
        f"{_fmt(metrics['hungarian_whole_placement'])} "
        f"neigh={_fmt(metrics['hungarian_whole_neighbour'])}",
        flush=True,
    )


def print_summary(metrics: Mapping[str, float], *, images: int, require_reciprocal: bool) -> None:
    print(f"\n[offset-pose summary] exact synthetic images={images}", flush=True)
    print(
        "  candidates: "
        f"direct-neighbour coverage={_fmt(metrics['candidate_direct_coverage'])} "
        f"local precision/recall={_fmt(metrics['candidate_local_precision'])}/"
        f"{_fmt(metrics['candidate_local_recall'])}",
        flush=True,
    )
    print(
        "  directed edges: "
        f"predicted-local={_fmt(metrics['predicted_local_fraction_of_candidates'])} "
        f"predicted-local true-local precision={_fmt(metrics['predicted_local_true_local_precision'])} "
        f"exact precision={_fmt(metrics['edge_exact_precision'])} "
        f"direct precision={_fmt(metrics['edge_direct_precision'])} "
        f"exact-direct recall={_fmt(metrics['edge_exact_direct_recall'])} "
        f"selected fraction={_fmt(metrics['selected_fraction_of_candidates'])}",
        flush=True,
    )
    print(
        "  reciprocal: "
        f"coverage-of-selected={_fmt(metrics['reciprocal_coverage_of_selected'])} "
        f"exact precision={_fmt(metrics['reciprocal_edge_exact_precision'])} "
        f"direct precision={_fmt(metrics['reciprocal_direct_precision'])} "
        f"(solver {'requires' if require_reciprocal else 'does not require'} reciprocal)",
        flush=True,
    )
    print(
        "  cycles: "
        f"triangles={_fmt(metrics['cycle_triangles'])} "
        f"consistent={_fmt(metrics['cycle_consistent_fraction'])} "
        f"median Linf error={_fmt(metrics['cycle_median_linf_error'])}",
        flush=True,
    )
    print(
        "  synchronization: "
        f"constraints={_fmt(metrics['sync_constraints'])} "
        f"largest component={_fmt(metrics['sync_largest_component_fraction'])} "
        f"affine R2={_fmt(metrics['sync_affine_coordinate_r2'])} "
        f"median residual={_fmt(metrics['sync_residual_median'])}",
        flush=True,
    )
    print(
        "  Hungarian (largest component only before deterministic fill): "
        f"component placement={_fmt(metrics['hungarian_component_placement'])} "
        f"whole placement={_fmt(metrics['hungarian_whole_placement'])} "
        f"whole neighbour={_fmt(metrics['hungarian_whole_neighbour'])}",
        flush=True,
    )


def smoke() -> dict[str, float]:
    """CPU-only check of reciprocal graph, synchronization and Hungarian rounding.

    A perfect Chebyshev-radius-one graph should synchronize exactly.  This
    validates the geometry code without a checkpoint or the large image data.
    """
    values = np.arange(NFRAG, dtype=np.int64)
    coordinates = np.stack((values // GRID, values % GRID), axis=-1).astype(np.float32)
    truth = coordinates[None, :, :] - coordinates[:, None, :]
    cheb = np.max(np.abs(truth), axis=-1)
    mask = (cheb == 1.0) & ~np.eye(NFRAG, dtype=bool)
    source, target = np.nonzero(mask)
    delta = truth[source, target]
    confidence = np.ones(source.size, dtype=np.float32)
    edge = build_edge_matrices(
        source,
        target,
        delta,
        confidence,
        confidence,
        np.ones(source.size, dtype=bool),
        min_confidence=0.5,
        max_predicted_offset=1.5,
        reciprocal_tolerance=0.01,
        require_reciprocal=True,
    )
    constraints = collapse_constraints(edge, reciprocal_tolerance=0.01)
    sync = synchronize_coordinates(constraints, iterations=2, huber=0.5)
    decoded = hungarian_round_largest_component(sync)
    if decoded is None:
        raise AssertionError("perfect graph did not produce a Hungarian decode")
    placement = float(np.mean(decoded.place == values))
    r2 = affine_coordinate_r2(sync.coordinates, coordinates, sync.largest_component)
    if placement < 0.999 or r2 < 0.999:
        raise AssertionError(f"pose smoke failed: placement={placement}, r2={r2}")
    # Hierarchical decode guard: one local direction may lose to the far logit
    # individually, while the aggregate of all local alternatives is still the
    # correct local-vs-far decision.  The evaluator must not regress to a
    # global 49-way argmax here.
    logits = torch.full((1, 49), -10.0)
    logits[0, 0] = 0.0
    logits[0, 48] = 1.0
    table = torch.from_numpy(
        np.concatenate((_row_major_offsets(3, include_zero=False), np.zeros((1, 2), np.float32)))
    )
    _, local_probability, conditional_confidence, _ = _hierarchical_logits(
        logits, far_index=48, table=table
    )
    # With 48 directions near -10 this particular sample is genuinely far.
    # Flipping all local logits to zero makes their logsumexp beat far=1.
    aggregate_probe = torch.zeros((1, 49))
    aggregate_probe[0, 48] = 1.0
    _, aggregate_local_probability, _, _ = _hierarchical_logits(
        aggregate_probe, far_index=48, table=table
    )
    if not (float(local_probability[0]) < 0.5 and float(aggregate_local_probability[0]) > 0.5):
        raise AssertionError("hierarchical local-vs-far decoding is malformed")
    return {
        "constraints": float(constraints.source.size),
        "largest_fraction": float(sync.largest_component.size / NFRAG),
        "affine_r2": r2,
        "placement": placement,
        "hierarchical_aggregate_p_local": float(aggregate_local_probability[0]),
        "hierarchical_conditional_confidence": float(conditional_confidence[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affinity-ckpt", "--affinity_ckpt", default=None, dest="affinity_ckpt")
    parser.add_argument(
        "--affinity-ckpt2",
        "--affinity_ckpt2",
        default=None,
        dest="affinity_ckpt2",
        help="optional second frozen encoder; union its top-K candidates with the primary",
    )
    parser.add_argument("--pose-ckpt", "--pose_ckpt", default=None, dest="pose_ckpt")
    parser.add_argument("--n", type=int, default=8, help="held-out exact synthetic images")
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--seed", type=int, default=SEED, help="fresh synthetic distortion seed")
    parser.add_argument("--top-k", type=int, default=64, help="directed affinity candidates per tile")
    parser.add_argument("--radius", type=int, default=None, help="offset radius; checkpoint value when omitted")
    parser.add_argument(
        "--far-index",
        default="auto",
        help="far class: auto (default), first, last, or integer",
    )
    parser.add_argument("--pose-chunk", type=int, default=4096, help="pair scorer batch size")
    parser.add_argument(
        "--local-threshold",
        "--local_threshold",
        type=float,
        default=0.5,
        help="hierarchical p(local) threshold that admits a geometric edge",
    )
    parser.add_argument(
        "--logit-decode",
        choices=("auto", "hierarchical", "flat"),
        default="auto",
        help="49-way decode: checkpoint-marked hierarchical by default; flat preserves legacy CE behavior",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="geometric confidence gate p(local) * p(offset | local)",
    )
    parser.add_argument(
        "--max-predicted-offset",
        type=float,
        default=None,
        help="discard regressions farther than this; radius+0.5 by default",
    )
    parser.add_argument(
        "--require-reciprocal",
        action="store_true",
        help="use only selected i->j edges supported by selected inverse j->i",
    )
    parser.add_argument("--reciprocal-tolerance", type=float, default=0.25)
    parser.add_argument("--cycle-degree", type=int, default=8, help="strong outgoing edges per node in cycle probe")
    parser.add_argument("--cycle-tolerance", type=float, default=0.25)
    parser.add_argument("--sync-iters", type=int, default=4, help="robust IRLS iterations")
    parser.add_argument("--sync-huber", type=float, default=0.75, help="IRLS Huber radius in cells")
    parser.add_argument(
        "--json-out",
        default=None,
        help="optional local JSON summary path (does not save puzzle predictions)",
    )
    parser.add_argument("--smoke", action="store_true", help="run geometry smoke instead of loading checkpoints")
    args = parser.parse_args()

    if args.smoke:
        print(json.dumps(smoke(), indent=2, sort_keys=True), flush=True)
        return
    if not args.affinity_ckpt:
        parser.error("--affinity-ckpt is required unless --smoke is used")
    if not args.pose_ckpt:
        parser.error("--pose-ckpt is required unless --smoke is used")
    if args.n < 1:
        parser.error("--n must be positive")
    if not 1 <= args.top_k < NFRAG:
        parser.error(f"--top-k must be in [1,{NFRAG - 1}]")
    if args.pose_chunk < 1:
        parser.error("--pose-chunk must be positive")
    if args.radius is not None and args.radius < 1:
        parser.error("--radius must be positive")
    if not 0.0 <= args.local_threshold <= 1.0:
        parser.error("--local-threshold must lie in [0,1]")

    device = _parse_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Affinity is explicitly frozen: only encoder inference mines candidates.
    affinity_model, affinity_metadata = load_affinity_model(args.affinity_ckpt, device)
    for parameter in affinity_model.parameters():
        parameter.requires_grad_(False)
    affinity_models: list[nn.Module] = [affinity_model]
    affinity_metadata2: Mapping[str, Any] | None = None
    if args.affinity_ckpt2:
        affinity_model2, affinity_metadata2 = load_affinity_model(args.affinity_ckpt2, device)
        for parameter in affinity_model2.parameters():
            parameter.requires_grad_(False)
        affinity_models.append(affinity_model2)
    checkpoint_radius = _safe_int(_metadata_value(_torch_load(args.pose_ckpt), "offset_radius", "radius", "pose_radius"))
    radius = args.radius if args.radius is not None else (checkpoint_radius or 3)
    pose_model, pose_payload, pose_module = load_pose_model(args.pose_ckpt, device, radius)
    for parameter in pose_model.parameters():
        parameter.requires_grad_(False)
    max_offset = args.max_predicted_offset if args.max_predicted_offset is not None else radius + 0.5
    if max_offset <= 0.0:
        parser.error("--max-predicted-offset must be positive")
    logit_decode = _checkpoint_decode_mode(pose_payload, pose_module, args.logit_decode)

    affinity_step = _as_mapping(affinity_metadata).get("step")
    pose_step = _as_mapping(pose_payload).get("step")
    print(
        f"device={device} topK={args.top_k} radius={radius} "
        f"decode={logit_decode} "
        f"p_local>={args.local_threshold:.3f} reciprocal={args.require_reciprocal} "
        f"confidence>={args.min_confidence:.3f}",
        flush=True,
    )
    print(
        f"affinity={os.path.abspath(args.affinity_ckpt)}" + (f" step={affinity_step}" if affinity_step is not None else ""),
        flush=True,
    )
    if args.affinity_ckpt2:
        affinity_step2 = _as_mapping(affinity_metadata2).get("step")
        print(
            f"affinity2={os.path.abspath(args.affinity_ckpt2)}"
            + (f" step={affinity_step2}" if affinity_step2 is not None else ""),
            flush=True,
        )
    print(
        f"pose={os.path.abspath(args.pose_ckpt)}" + (f" step={pose_step}" if pose_step is not None else ""),
        flush=True,
    )

    _, val_names = train_val_split()
    if not val_names:
        raise RuntimeError("held-out validation split is empty")
    if args.n > len(val_names):
        parser.error(f"--n={args.n} exceeds held-out split size {len(val_names)}")
    dataset = CanvasDataset(val_names[: args.n], real_prob=0.0, seed=args.seed)

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for index in range(args.n):
        metrics = evaluate_one(
            affinity_models,
            pose_model,
            pose_module,
            pose_payload,
            dataset[index],
            device=device,
            top_k=args.top_k,
            radius=radius,
            far_index=args.far_index,
            local_threshold=args.local_threshold,
            logit_decode=logit_decode,
            chunk_size=args.pose_chunk,
            min_confidence=args.min_confidence,
            max_predicted_offset=max_offset,
            reciprocal_tolerance=args.reciprocal_tolerance,
            require_reciprocal=args.require_reciprocal,
            cycle_degree=args.cycle_degree,
            cycle_tolerance=args.cycle_tolerance,
            sync_iterations=args.sync_iters,
            sync_huber=args.sync_huber,
        )
        _add_metrics(totals, counts, metrics)
        print_image_report(index + 1, metrics)

    summary = _mean_metrics(totals, counts)
    print_summary(summary, images=args.n, require_reciprocal=args.require_reciprocal)
    if args.json_out:
        parent = os.path.dirname(os.path.abspath(args.json_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "args": vars(args),
                    "summary": summary,
                    "finite_counts": counts,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        print(f"wrote {os.path.abspath(args.json_out)}", flush=True)


if __name__ == "__main__":
    main()
