"""Inference-only helpers for the raw edge-ranker manual-layout audit.

This module deliberately contains no optimiser or training entrypoint.  It
loads the already frozen raw pairwise checkpoint, verifies its exact semantic
contract, and provides target-assisted diagnostics that are called only after
all input-only predictions have been committed to disk.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.candidate_supply import RecoveredLayout
from aiijc_puzzle.edge_ranker import PairwiseEdgeRanker
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
)

RAW_CHECKPOINT_SHA256 = "d18ff864c63170d5fcdb868d672a60515d10ac600afa2ed0424000921ecbb21a"
RAW_CHECKPOINT_EVAL_OFFSET = 52
RAW_CHECKPOINT_EVAL_COUNT = 12
EXPECTED_CONTRACT = {
    "architecture": "joint-seam-context-cnn-v1",
    "views": ["raw", "tile_z", "bilateral", "gray"],
    "candidate_k": 5,
    "view_mode": "raw",
    "feature_dim": 12,
    "width": 24,
    "hidden": 48,
    "label_policy": "exact recovered neighbour; trusted-query training only",
    "teacher_policy": "trusted candidate clean symmetric extrapolation listwise CE",
    "teacher_weight": 0.15,
    "selector_namespace": EXPERIMENT_SUBSET_NAMESPACE,
    "selector_seed": EXPERIMENT_SUBSET_SEED,
    "train_limit": 64,
}


def names_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash a deterministic ordered panel roster."""

    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode("utf-8")
    ).hexdigest()


def load_verified_raw_checkpoint(
    checkpoint_path: Path,
    *,
    manifest: Mapping[str, Any],
    project_root: Path,
    device: torch.device,
) -> tuple[PairwiseEdgeRanker, Mapping[str, Any]]:
    """Load the immutable scale-raw checkpoint and fail closed on drift."""

    if sha256_file(checkpoint_path) != RAW_CHECKPOINT_SHA256:
        raise ValueError("raw edge-ranker checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint has no mapping contract")
    for key, expected in EXPECTED_CONTRACT.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"checkpoint contract mismatch for {key}: {contract.get(key)!r} != {expected!r}"
            )
    if contract.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("checkpoint validation protocol digest mismatch")
    train_records = select_manifest_records(manifest, "train", limit=64)
    if contract.get("train_filenames") != [record["filename"] for record in train_records]:
        raise ValueError("checkpoint train roster mismatch")
    if contract.get("train_selection_digest") != names_digest(train_records):
        raise ValueError("checkpoint train roster digest mismatch")

    semantic_paths = {
        "edge_ranker": project_root / "src" / "aiijc_puzzle" / "edge_ranker.py",
        "candidate_supply": project_root / "src" / "aiijc_puzzle" / "candidate_supply.py",
        "legacy_upgrade": project_root / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        "protocol": project_root / "src" / "aiijc_puzzle" / "protocol.py",
    }
    semantic_hashes = contract.get("semantic_code_sha256")
    if not isinstance(semantic_hashes, Mapping):
        raise ValueError("checkpoint has no semantic code hashes")
    for name, path in semantic_paths.items():
        if semantic_hashes.get(name) != sha256_file(path):
            raise ValueError(f"checkpoint semantic source drift: {name}")

    model = PairwiseEdgeRanker(
        feature_dim=int(contract["feature_dim"]),
        view_mode=str(contract["view_mode"]),
        width=int(contract["width"]),
        hidden=int(contract["hidden"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def layout_metrics(layout: np.ndarray, recovered: RecoveredLayout) -> dict[str, float]:
    """Measure placement and true neighbour recovery for one strict layout."""

    value = np.asarray(layout, dtype=np.int64)
    n = len(value)
    grid = round(n**0.5)
    if value.shape != (n,) or grid * grid != n:
        raise ValueError("layout must be a flat square-grid permutation")
    if not np.array_equal(np.sort(value), np.arange(n)):
        raise ValueError("layout must be a complete permutation")
    truth = np.asarray(recovered.dirty_at_position, dtype=np.int64)
    if truth.shape != value.shape:
        raise ValueError("recovered layout size mismatch")
    position_of_dirty = recovered.position_of_dirty
    predicted_position = np.empty(n, dtype=np.int64)
    predicted_position[value] = np.arange(n)
    shifts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for tile, predicted in enumerate(predicted_position):
        true = int(position_of_dirty[tile])
        predicted_row, predicted_column = divmod(int(predicted), grid)
        true_row, true_column = divmod(true, grid)
        shifts[(true_row - predicted_row, true_column - predicted_column)] += 1
    board = value.reshape(grid, grid)
    left = position_of_dirty[board[:, :-1]]
    right = position_of_dirty[board[:, 1:]]
    top = position_of_dirty[board[:-1]]
    bottom = position_of_dirty[board[1:]]
    right_accuracy = np.mean((right - left == 1) & (right // grid == left // grid))
    down_accuracy = np.mean(bottom - top == grid)
    return {
        "direct_placement": float(np.mean(value == truth)),
        "translation_aligned_placement": float(max(shifts.values()) / n),
        "right_adjacency": float(right_accuracy),
        "down_adjacency": float(down_accuracy),
        "adjacency": float(0.5 * (right_accuracy + down_accuracy)),
    }


def paired_bootstrap_ci(
    differences: Sequence[float] | np.ndarray,
    *,
    seed: int,
    replicates: int = 20_000,
) -> dict[str, float | int]:
    """Deterministic paired percentile-bootstrap interval for a mean delta."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("differences must be a finite vector with at least two values")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise ValueError("replicates must be an integer >= 100")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    bootstrapped = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_lower": float(np.quantile(bootstrapped, 0.025)),
        "ci95_upper": float(np.quantile(bootstrapped, 0.975)),
        "replicates": replicates,
        "seed": seed,
    }


def dual_manual_gate(
    adjacency_differences: Sequence[float] | np.ndarray,
    final_ssim_differences: Sequence[float] | np.ndarray,
    *,
    seed: int = 20260830,
    replicates: int = 20_000,
) -> dict[str, Any]:
    """Apply the preregistered geometry and final-tail noninferiority gate."""

    adjacency = paired_bootstrap_ci(
        adjacency_differences,
        seed=seed,
        replicates=replicates,
    )
    final_ssim = paired_bootstrap_ci(
        final_ssim_differences,
        seed=seed + 1,
        replicates=replicates,
    )
    conditions = [
        {
            "metric": "adjacency_delta_ci95_lower",
            "observed": adjacency["ci95_lower"],
            "required": "> 0",
            "passed": bool(adjacency["ci95_lower"] > 0.0),
        },
        {
            "metric": "final_ssim_delta_ci95_lower",
            "observed": final_ssim["ci95_lower"],
            "required": ">= -0.003",
            "passed": bool(final_ssim["ci95_lower"] >= -0.003),
        },
    ]
    return {
        "passed": all(condition["passed"] for condition in conditions),
        "conditions": conditions,
        "adjacency_delta": adjacency,
        "final_ssim_delta": final_ssim,
    }


__all__ = [
    "EXPECTED_CONTRACT",
    "RAW_CHECKPOINT_EVAL_COUNT",
    "RAW_CHECKPOINT_EVAL_OFFSET",
    "RAW_CHECKPOINT_SHA256",
    "dual_manual_gate",
    "layout_metrics",
    "load_verified_raw_checkpoint",
    "names_digest",
    "paired_bootstrap_ci",
]
