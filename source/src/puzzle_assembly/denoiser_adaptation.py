"""Leakage-safe metrics and a warm-start multi-view side encoder.

This module is deliberately independent from the candidate-graph oracle.  It
only consumes synthetic exact-panel permutations and ordinary scorer views:
dirty tiles, the production TileNAF output, and a candidate denoiser output.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE, TILE_COUNT
from .learned import SideEmbeddingNet


VIEW_NAMES = ("dirty", "old_denoised", "new_denoised")
FORBIDDEN_ORACLE_LABEL_FRAGMENT = "fixture_label"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        values: list[str] = []
        for key, nested in value.items():
            values.extend(_walk_strings(key))
            values.extend(_walk_strings(nested))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = []
        for nested in value:
            values.extend(_walk_strings(nested))
        return values
    return []


def validate_protocol_safety(config: Mapping[str, Any]) -> None:
    """Fail closed on label-oracle references and protocol shape drift."""

    strings = _walk_strings(config)
    if any(FORBIDDEN_ORACLE_LABEL_FRAGMENT in value.lower() for value in strings):
        raise ValueError("solver adaptation protocol must not reference oracle labels")
    if config.get("kind") != "solver_denoiser_adaptation_protocol":
        raise ValueError("unexpected solver adaptation protocol kind")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported solver adaptation protocol schema")
    arms = config.get("comparison_arms", {})
    if tuple(sorted(arms)) != ("A_renderer_only", "B_new_scoring_view", "C_retrained_multiview"):
        raise ValueError("protocol must define exactly the fixed A/B/C arms")
    production = config.get("production_layout", {})
    required = {
        "soft_cycle_top_k": 8,
        "qap_iterations": 25,
        "qap_restarts": 2,
        "hbt_weight": 4.0,
    }
    for key, expected in required.items():
        if production.get(key) != expected:
            raise ValueError(f"production layout drift for {key}: {production.get(key)!r}")
    if config.get("optional_higher_order", {}).get("enabled_before_pair_gate") is not False:
        raise ValueError("higher-order verification must be disabled before the pair gate")


@dataclass(frozen=True)
class SuccessorTruth:
    right: np.ndarray
    down: np.ndarray
    true_edge_count: int


def successor_truth(
    slot_to_target: np.ndarray,
    *,
    grid: int = GRID,
) -> SuccessorTruth:
    """Return true right/down successor slot for every input slot, or -1."""

    values = np.asarray(slot_to_target)
    count = grid * grid
    if values.shape != (count,):
        raise ValueError(f"slot_to_target must have shape {(count,)}")
    if sorted(values.astype(int).tolist()) != list(range(count)):
        raise ValueError("slot_to_target must be a permutation")
    position_to_slot = np.empty(count, dtype=np.int32)
    position_to_slot[values.astype(np.int64)] = np.arange(count, dtype=np.int32)
    right = np.full(count, -1, dtype=np.int32)
    down = np.full(count, -1, dtype=np.int32)
    for position, slot in enumerate(position_to_slot.tolist()):
        row, column = divmod(position, grid)
        if column + 1 < grid:
            right[slot] = position_to_slot[position + 1]
        if row + 1 < grid:
            down[slot] = position_to_slot[position + grid]
    return SuccessorTruth(
        right=right,
        down=down,
        true_edge_count=2 * grid * (grid - 1),
    )


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def largest(self) -> int:
        return max(self.size[self.find(index)] for index in range(len(self.parent)))


def _stable_order(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("compatibility matrix must be square")
    np.fill_diagonal(values, np.inf)
    return np.argsort(values, axis=1, kind="stable")


def _direction_diagnostics(
    matrix: np.ndarray,
    truth: np.ndarray,
    *,
    ks: tuple[int, ...],
    graph_k: int,
) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(values, np.inf)
    order = _stable_order(values)
    count = len(truth)
    if order.shape != (count, count):
        raise ValueError("truth and compatibility matrix sizes differ")
    eligible = np.flatnonzero(truth >= 0)
    ranks = np.empty(len(eligible), dtype=np.int32)
    for index, query in enumerate(eligible.tolist()):
        hit = np.flatnonzero(order[query] == truth[query])
        if len(hit) != 1:
            raise RuntimeError("true successor is missing from candidate order")
        ranks[index] = int(hit[0]) + 1

    best_out = order[:, 0]
    reverse_order = np.argsort(values, axis=0, kind="stable")
    best_in = reverse_order[0]
    mutual = np.asarray(
        [best_in[candidate] == query for query, candidate in enumerate(best_out)],
        dtype=bool,
    )
    mutual_queries = np.flatnonzero(mutual)
    mutual_correct = int(
        sum(truth[query] >= 0 and best_out[query] == truth[query] for query in mutual_queries)
    )

    k = min(graph_k, count - 1)
    graph_candidates = order[:, :k]
    candidate_hits = int(
        sum(truth[query] in graph_candidates[query] for query in eligible.tolist())
    )
    connected = _DisjointSet(count)
    for query in range(count):
        for candidate in graph_candidates[query].tolist():
            connected.union(query, int(candidate))

    result = {
        f"top{k_value}": float(np.mean(ranks <= k_value)) for k_value in ks
    }
    result.update(
        {
            "mrr": float(np.mean(1.0 / ranks)),
            "mutual_precision": mutual_correct / max(len(mutual_queries), 1),
            "mutual_correct_coverage": mutual_correct / max(len(eligible), 1),
            "mutual_proposal_coverage": len(mutual_queries) / max(len(eligible), 1),
            f"candidate_recall_at_{graph_k}": candidate_hits / max(len(eligible), 1),
            f"candidate_lcc_at_{graph_k}": float(connected.largest()),
        }
    )
    return result


def retrieval_diagnostics(
    compatibility: CompatibilityMatrices,
    slot_to_target: np.ndarray,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    graph_k: int = 32,
    grid: int = GRID,
) -> dict[str, float]:
    """Directed retrieval, mutual-pair, and candidate-graph diagnostics."""

    if not ks or any(value <= 0 for value in ks):
        raise ValueError("ks must contain positive integers")
    truth = successor_truth(slot_to_target, grid=grid)
    directional = {
        "right": _direction_diagnostics(
            compatibility.right, truth.right, ks=ks, graph_k=graph_k
        ),
        "down": _direction_diagnostics(
            compatibility.down, truth.down, ks=ks, graph_k=graph_k
        ),
    }
    result: dict[str, float] = {}
    for direction, metrics in directional.items():
        for key, value in metrics.items():
            result[f"{direction}_{key}"] = float(value)
    shared = sorted(set(directional["right"]) & set(directional["down"]))
    for key in shared:
        result[key] = 0.5 * (
            float(directional["right"][key]) + float(directional["down"][key])
        )
    result["true_directed_edges"] = float(truth.true_edge_count)
    return result


def mean_numeric(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("cannot aggregate an empty record list")
    keys = sorted(
        key
        for key in records[0]
        if all(isinstance(record.get(key), (int, float)) for record in records)
    )
    return {
        key: float(np.mean([float(record[key]) for record in records])) for key in keys
    }


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired bootstrap needs at least two source deltas")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


class MultiViewSideEmbeddingNet(nn.Module):
    """Warm-start HBT shared across dirty, old-denoised, and new-denoised views.

    The production old-denoised representation is the fixed residual anchor.
    Two learned scalar gates can add evidence from dirty and candidate-denoised
    views.  Initial gates are deliberately tiny, so a newly constructed model
    starts close to the production HBT instead of destroying it.
    """

    def __init__(
        self,
        *,
        encoder_config: Mapping[str, Any],
        dirty_gate_init: float = -4.0,
        new_gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.encoder = SideEmbeddingNet(**dict(encoder_config))
        self.dirty_gate_init = float(dirty_gate_init)
        self.new_gate_init = float(new_gate_init)
        self.residual_logits = nn.Parameter(
            torch.tensor([dirty_gate_init, new_gate_init], dtype=torch.float32)
        )

    @property
    def temperature(self) -> float:
        return float(self.encoder.temperature)

    @classmethod
    def from_production_encoder(
        cls,
        encoder: SideEmbeddingNet,
        *,
        dirty_gate_init: float = -4.0,
        new_gate_init: float = -4.0,
    ) -> "MultiViewSideEmbeddingNet":
        if not isinstance(encoder, SideEmbeddingNet):
            raise TypeError("multi-view warm start requires a pooled SideEmbeddingNet")
        model = cls(
            encoder_config=encoder.config(),
            dirty_gate_init=dirty_gate_init,
            new_gate_init=new_gate_init,
        )
        model.encoder.load_state_dict(encoder.state_dict(), strict=True)
        return model

    def config(self) -> dict[str, Any]:
        return {
            "encoder_config": self.encoder.config(),
            "dirty_gate_init": self.dirty_gate_init,
            "new_gate_init": self.new_gate_init,
        }

    def _validate_views(self, views: Mapping[str, torch.Tensor]) -> None:
        if tuple(sorted(views)) != tuple(sorted(VIEW_NAMES)):
            raise ValueError(f"views must contain exactly {VIEW_NAMES}")
        reference = views[VIEW_NAMES[0]]
        if reference.ndim != 4 or reference.shape[1:] != (3, TILE, TILE):
            raise ValueError("view tensors must be NCHW RGB 20x20 tiles")
        if any(value.shape != reference.shape for value in views.values()):
            raise ValueError("all views must have identical tensor shapes")

    def forward(
        self,
        views: Mapping[str, torch.Tensor],
        *,
        view_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_views(views)
        encoded = {name: self.encoder(views[name]) for name in VIEW_NAMES}
        gates = torch.sigmoid(self.residual_logits)
        if view_mask is not None:
            mask = torch.as_tensor(view_mask, dtype=gates.dtype, device=gates.device)
            if mask.shape != (2,):
                raise ValueError("view_mask must have shape (2,) for dirty/new residuals")
            if torch.any((mask < 0) | (mask > 1)):
                raise ValueError("view_mask values must lie in [0,1]")
            gates = gates * mask
        old = encoded["old_denoised"]
        dirty = encoded["dirty"]
        new = encoded["new_denoised"]
        outputs: dict[str, torch.Tensor] = {}
        embedding_names = ("q_right", "k_left", "q_down", "k_up")
        for name in embedding_names:
            fused = (
                old[name]
                + gates[0] * (dirty[name] - old[name])
                + gates[1] * (new[name] - old[name])
            )
            outputs[name] = F.normalize(fused, dim=1)
            outputs[f"raw_{name}"] = fused
        outputs["outside_logits"] = (
            old["outside_logits"]
            + gates[0] * (dirty["outside_logits"] - old["outside_logits"])
            + gates[1] * (new["outside_logits"] - old["outside_logits"])
        )
        outputs["residual_gates"] = gates
        return outputs


@torch.inference_mode()
def multiview_compatibility(
    model: MultiViewSideEmbeddingNet,
    views: Mapping[str, np.ndarray],
    *,
    device: torch.device | str,
    name: str = "dirty_old_new_multiview_hbt",
) -> tuple[CompatibilityMatrices, np.ndarray, np.ndarray]:
    if tuple(sorted(views)) != tuple(sorted(VIEW_NAMES)):
        raise ValueError(f"views must contain exactly {VIEW_NAMES}")
    tensors: dict[str, torch.Tensor] = {}
    for view_name, values in views.items():
        array = np.asarray(values)
        if array.shape != (TILE_COUNT, TILE, TILE, 3) or array.dtype != np.uint8:
            raise ValueError(f"{view_name} must be uint8 (576,20,20,3)")
        tensors[view_name] = torch.from_numpy(np.ascontiguousarray(array)).permute(
            0, 3, 1, 2
        ).to(device=device, dtype=torch.float32)
    model.eval()
    outputs = model(tensors)
    right = -(outputs["q_right"] @ outputs["k_left"].T).float().cpu().numpy()
    down = -(outputs["q_down"] @ outputs["k_up"].T).float().cpu().numpy()
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return (
        CompatibilityMatrices(name, right, down),
        outputs["outside_logits"].float().cpu().numpy(),
        outputs["residual_gates"].float().cpu().numpy(),
    )


def save_multiview_checkpoint(
    path: str | Path,
    model: MultiViewSideEmbeddingNet,
    *,
    metadata: Mapping[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "puzzle_multiview_side_embedding_hbt",
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "metadata": dict(metadata),
        },
        output,
    )


def load_multiview_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[MultiViewSideEmbeddingNet, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "puzzle_multiview_side_embedding_hbt"
    ):
        raise ValueError("unsupported multi-view HBT checkpoint")
    model = MultiViewSideEmbeddingNet(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload.get("metadata", {}))
