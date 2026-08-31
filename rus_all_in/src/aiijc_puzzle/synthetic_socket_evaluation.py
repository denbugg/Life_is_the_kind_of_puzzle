"""Exact-label synthetic evaluation primitives for :mod:`socket_matcher`.

The organizer training targets are clean, ordered canvases.  They can therefore
produce an exact evaluation panel without recovering a permutation from noisy
pixels: corrupt every tile independently, shuffle the corrupted tiles, and
retain the inverse shuffle only for the later scoring stage.

This module keeps three boundaries explicit:

* source selection is restricted to the frozen manifest's ``train`` split and
  excludes the complete checkpoint ancestry;
* synthetic inputs and exact references are returned as separate objects;
* local candidate lists are frozen before they are compared with a reference.

It deliberately has no path or API for calibration, holdout, or competition
test images.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.protocol import compute_protocol_digest, select_manifest_records
from aiijc_puzzle.restoration_r6 import distort_tiles

DEFAULT_SYNTHETIC_NAMESPACE = "aiijc-socket-exact-synthetic-v1"


@dataclass(frozen=True)
class CheckpointLineage:
    """Verified source names collected over a checkpoint continuation chain."""

    filenames: tuple[str, ...]
    checkpoint_paths: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticSocketInput:
    """A corrupted shuffled board that is safe to pass to a predictor."""

    case_id: str
    source_filename: str
    draw_index: int
    corruption_seed: int
    permutation_seed: int
    tiles: np.ndarray


@dataclass(frozen=True)
class ExactSyntheticReference:
    """The exact label kept outside the dirty-only prediction interface."""

    case_id: str
    tile_at_position: np.ndarray


def names_digest(names: Sequence[str], *, sort_names: bool = False) -> str:
    """Return the newline-delimited digest used by SocketMatcher checkpoints."""

    values = sorted(names) if sort_names else list(names)
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _validated_names(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(name, str) and name for name in value):
        raise ValueError(f"checkpoint {field} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"checkpoint {field} contains duplicate filenames")
    return tuple(value)


def _selection_lineage(payload: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("checkpoint has no valid selection mapping")
    train_names = _validated_names(selection.get("train_filenames"), field="train_filenames")
    train_digest = selection.get("train_digest")
    if not isinstance(train_digest, str) or train_digest != names_digest(train_names):
        raise ValueError("checkpoint train filename digest does not match its names")

    declared_value = selection.get("lineage_train_filenames")
    if declared_value is None:
        training = set(train_names)
        explicit_training: set[str] = set()
    else:
        declared = _validated_names(declared_value, field="lineage_train_filenames")
        lineage_digest = selection.get("lineage_train_digest")
        if lineage_digest is None:
            raise ValueError("checkpoint declares lineage names without a lineage digest")
        if lineage_digest != names_digest(declared, sort_names=True):
            raise ValueError("checkpoint lineage filename digest does not match its names")
        if not set(train_names).issubset(declared):
            raise ValueError("checkpoint lineage omits names used by its current training run")
        training = set(declared)
        explicit_training = set(declared)

    exposed_value = selection.get("lineage_exposed_filenames")
    if exposed_value is None:
        return training, explicit_training
    exposed = _validated_names(exposed_value, field="lineage_exposed_filenames")
    exposed_digest = selection.get("lineage_exposed_digest")
    if exposed_digest is None:
        raise ValueError("checkpoint declares exposure names without an exposure digest")
    if exposed_digest != names_digest(exposed, sort_names=True):
        raise ValueError("checkpoint exposure filename digest does not match its names")
    if not training.issubset(exposed):
        raise ValueError("checkpoint exposure lineage omits a training-lineage source")
    return set(exposed), set(exposed)


def _resolve_ancestor_path(
    value: str,
    *,
    current_path: Path,
    project_root: Path,
) -> Path:
    candidate = Path(value)
    candidates = (
        candidate,
        project_root / candidate,
        current_path.parent / candidate,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "cannot verify checkpoint ancestry because continued_from is missing: "
        f"{value}"
    )


def load_checkpoint_with_lineage(
    checkpoint_path: Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], CheckpointLineage]:
    """Load a checkpoint and fail closed if its complete source lineage is unverifiable."""

    root_path = checkpoint_path.resolve()
    project_root = project_root.resolve()
    if not root_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {root_path}")

    visited: set[Path] = set()
    ordered_paths: list[str] = []

    def visit(path: Path) -> tuple[dict[str, Any], set[str]]:
        path = path.resolve()
        if path in visited:
            raise ValueError(f"checkpoint continuation cycle detected at {path}")
        visited.add(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"checkpoint payload is not a dictionary: {path}")
        own_names, explicit_lineage = _selection_lineage(payload)
        ancestor_names: set[str] = set()
        continued_from = payload.get("continued_from")
        if continued_from is not None:
            if not isinstance(continued_from, str) or not continued_from:
                raise ValueError("checkpoint continued_from must be null or a non-empty path")
            ancestor_path = _resolve_ancestor_path(
                continued_from,
                current_path=path,
                project_root=project_root,
            )
            _, ancestor_names = visit(ancestor_path)
            if explicit_lineage and not ancestor_names.issubset(explicit_lineage):
                raise ValueError("checkpoint declared lineage omits a verified ancestor source")
        ordered_paths.append(str(path))
        return payload, own_names | ancestor_names

    root_payload, filenames = visit(root_path)
    return root_payload, CheckpointLineage(
        filenames=tuple(sorted(filenames)),
        checkpoint_paths=tuple(ordered_paths),
    )


def select_source_disjoint_train_records(
    manifest: Mapping[str, Any],
    *,
    excluded_filenames: Sequence[str],
    limit: int,
    seed: int,
    namespace: str = DEFAULT_SYNTHETIC_NAMESPACE,
) -> tuple[Mapping[str, Any], ...]:
    """Select deterministic manifest-train sources outside checkpoint lineage."""

    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("validation manifest protocol digest is invalid")
    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise ValueError("validation manifest has no splits mapping")
    train = raw_splits.get("train")
    if not isinstance(train, Sequence) or isinstance(train, (str, bytes)):
        raise ValueError("validation manifest train split is not a record sequence")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")

    excluded = set(_validated_names(list(excluded_filenames), field="excluded_filenames"))
    ranked = select_manifest_records(
        manifest,
        "train",
        limit=len(train),
        seed=seed,
        namespace=namespace,
    )
    selected = tuple(record for record in ranked if record["filename"] not in excluded)[:limit]
    if len(selected) != limit:
        raise ValueError(
            f"only {len(selected)} source-disjoint train records remain, requested {limit}"
        )
    if any(record["filename"] in excluded for record in selected):
        raise RuntimeError("source-disjoint selection invariant failed")
    return selected


def _case_seeds(*, seed: int, filename: str, draw_index: int) -> tuple[int, int]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index < 0:
        raise ValueError("draw_index must be a non-negative integer")
    digest = hashlib.sha256(
        f"{DEFAULT_SYNTHETIC_NAMESPACE}\0{seed}\0{filename}\0{draw_index}".encode()
    ).digest()
    corruption_seed = int.from_bytes(digest[:8], "little")
    permutation_seed = int.from_bytes(digest[8:16], "little")
    return corruption_seed, permutation_seed


def make_exact_synthetic_case(
    clean_tiles: np.ndarray,
    *,
    source_filename: str,
    draw_index: int,
    seed: int,
) -> tuple[SyntheticSocketInput, ExactSyntheticReference]:
    """Independently corrupt and shuffle one clean square board with an exact label."""

    clean = np.asarray(clean_tiles)
    if clean.ndim != 4 or clean.shape[1:] != (20, 20, 3) or clean.dtype != np.uint8:
        raise ValueError(
            f"clean_tiles must be uint8 N x 20 x 20 x 3, got {clean.dtype} {clean.shape}"
        )
    grid = round(len(clean) ** 0.5)
    if grid < 2 or grid * grid != len(clean):
        raise ValueError("clean tile count must be a square board with grid >= 2")
    if not isinstance(source_filename, str) or not source_filename:
        raise ValueError("source_filename must be a non-empty string")

    corruption_seed, permutation_seed = _case_seeds(
        seed=seed,
        filename=source_filename,
        draw_index=draw_index,
    )
    corrupted = distort_tiles(clean, np.random.default_rng(corruption_seed))
    permutation = np.random.default_rng(permutation_seed).permutation(len(clean))
    shuffled = np.ascontiguousarray(corrupted[permutation])
    tile_at_position = np.ascontiguousarray(np.argsort(permutation).astype(np.int32))
    case_digest = hashlib.sha256(
        f"{source_filename}\0{draw_index}\0{seed}".encode()
    ).hexdigest()[:16]
    case_id = f"synthetic-{case_digest}"
    return (
        SyntheticSocketInput(
            case_id=case_id,
            source_filename=source_filename,
            draw_index=draw_index,
            corruption_seed=corruption_seed,
            permutation_seed=permutation_seed,
            tiles=shuffled,
        ),
        ExactSyntheticReference(case_id=case_id, tile_at_position=tile_at_position),
    )


def freeze_topk_candidates(scores: np.ndarray, *, max_k: int) -> np.ndarray:
    """Freeze deterministic row-wise candidate indices without seeing labels."""

    value = np.asarray(scores)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"scores must be a square matrix, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("scores contain non-finite values")
    if isinstance(max_k, bool) or not isinstance(max_k, int) or not 1 <= max_k < len(value):
        raise ValueError("max_k must be an integer in [1, tile_count - 1]")
    # Stable sorting makes exact ties reproducible across runs and platforms.
    return np.ascontiguousarray(np.argsort(-value, axis=1, kind="stable")[:, :max_k]).astype(
        np.int32,
        copy=False,
    )


def exact_local_retrieval_metrics(
    right_candidates: np.ndarray,
    down_candidates: np.ndarray,
    reference_tile_at_position: np.ndarray,
    *,
    ks: tuple[int, ...] = (1, 5, 16, 32),
) -> dict[str, float | int]:
    """Score frozen outgoing candidate lists against an exact permutation."""

    reference = np.asarray(reference_tile_at_position, dtype=np.int64)
    if reference.ndim != 1:
        raise ValueError("reference_tile_at_position must be one-dimensional")
    count = len(reference)
    grid = round(count**0.5)
    if grid < 2 or grid * grid != count or not np.array_equal(
        np.sort(reference), np.arange(count)
    ):
        raise ValueError("reference_tile_at_position must be a strict square permutation")
    candidates = {
        "right": np.asarray(right_candidates),
        "down": np.asarray(down_candidates),
    }
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    max_k = max(ks)
    for name, value in candidates.items():
        if value.ndim != 2 or value.shape[0] != count or value.shape[1] < max_k:
            raise ValueError(f"{name}_candidates must have shape ({count}, >= {max_k})")
        if not np.issubdtype(value.dtype, np.integer) or np.any((value < 0) | (value >= count)):
            raise ValueError(f"{name}_candidates contain invalid tile indices")

    result: dict[str, float | int] = {}
    pooled_hits = {k: 0 for k in ks}
    pooled_total = 0
    for name, delta in (("right", 1), ("down", grid)):
        position = np.arange(count)
        valid = position % grid != grid - 1 if name == "right" else position < count - grid
        position = position[valid]
        anchor = reference[position]
        truth = reference[position + delta]
        total = len(position)
        result[f"{name}_total"] = total
        for k in ks:
            hits = int(
                np.count_nonzero(
                    np.any(candidates[name][anchor, :k] == truth[:, None], axis=1)
                )
            )
            result[f"{name}_hits_at_{k}"] = hits
            result[f"{name}_r{k}"] = hits / total
            pooled_hits[k] += hits
        pooled_total += total
    result["pooled_total"] = pooled_total
    for k in ks:
        result[f"pooled_hits_at_{k}"] = pooled_hits[k]
        result[f"pooled_r{k}"] = pooled_hits[k] / pooled_total
    return result


__all__ = [
    "CheckpointLineage",
    "DEFAULT_SYNTHETIC_NAMESPACE",
    "ExactSyntheticReference",
    "SyntheticSocketInput",
    "exact_local_retrieval_metrics",
    "freeze_topk_candidates",
    "load_checkpoint_with_lineage",
    "make_exact_synthetic_case",
    "names_digest",
    "select_source_disjoint_train_records",
]
