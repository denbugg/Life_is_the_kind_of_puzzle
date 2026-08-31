"""Baseline-guided causal repair for a trained :mod:`border_pointer_sorter`.

The repair never constructs a layout from scratch.  It starts from a strict
Socket decoder permutation and processes its current sequence in raster order.
A confident pointer proposal is applied as a swap with the tile's unique future
position, and only when a frozen-Socket top-k seam guard does not lose supported
adjacencies.  Thus every intermediate and returned layout remains a permutation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from aiijc_puzzle.border_pointer_sorter import BorderPointerOutput, BorderPointerSorter


@dataclass(frozen=True)
class BaselineRepairConfig:
    """Frozen knobs for one bounded repair invocation."""

    logit_margin: float = 1.0
    budgets: tuple[int, ...] = (4, 16)
    socket_support_topk: int = 16

    def validate(self, *, count: int) -> None:
        if not math.isfinite(self.logit_margin) or self.logit_margin < 0:
            raise ValueError("logit_margin must be finite and non-negative")
        if (
            not self.budgets
            or tuple(sorted(set(self.budgets))) != self.budgets
            or self.budgets[0] <= 0
            or self.budgets[-1] >= count
        ):
            raise ValueError("budgets must be sorted unique integers in [1, count - 1]")
        if not 1 <= self.socket_support_topk < count:
            raise ValueError("socket_support_topk must be in [1, count - 1]")


@dataclass(frozen=True)
class BaselinePrefixTrace:
    """Target-free candidate rosters for later exact conditional diagnostics."""

    prefix_topk: np.ndarray
    no_prefix_topk: np.ndarray


@dataclass(frozen=True)
class BaselineRepairResult:
    """Strict layouts and target-free proposal diagnostics."""

    layouts: dict[int, np.ndarray]
    proposals: tuple[dict[str, Any], ...]
    trace: BaselinePrefixTrace


def _strict_layout(layout: Any, *, count: int) -> np.ndarray:
    value = np.asarray(layout, dtype=np.int32)
    if value.shape != (count,) or not np.array_equal(np.sort(value), np.arange(count)):
        raise ValueError("baseline layout must be a strict permutation")
    return np.ascontiguousarray(value)


def _real_score_matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.shape == (count + 1, count + 1):
        result = result[:count, :count]
    if result.shape != (count, count) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite {count}x{count} real scores")
    return np.ascontiguousarray(result)


def _topk_support(scores: np.ndarray, *, topk: int) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")[:, :topk]
    support = np.zeros(scores.shape, dtype=bool)
    support[np.arange(len(scores))[:, None], order] = True
    return support


def _affected_edges(*, grid: int, positions: tuple[int, int]) -> tuple[tuple[str, int, int], ...]:
    edges: set[tuple[str, int, int]] = set()
    for position in positions:
        row, column = divmod(position, grid)
        if column:
            edges.add(("right", position - 1, position))
        if column < grid - 1:
            edges.add(("right", position, position + 1))
        if row:
            edges.add(("down", position - grid, position))
        if row < grid - 1:
            edges.add(("down", position, position + grid))
    return tuple(sorted(edges))


def _edge_diagnostics(
    layout: np.ndarray,
    *,
    edges: tuple[tuple[str, int, int], ...],
    support: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> tuple[int, float]:
    supported = 0
    energy = 0.0
    for axis, source_position, target_position in edges:
        source = int(layout[source_position])
        target = int(layout[target_position])
        supported += int(support[axis][source, target])
        energy += float(scores[axis][source, target])
    return supported, energy


def _no_prefix_topk(
    model: BorderPointerSorter,
    output: BorderPointerOutput,
    baseline: torch.Tensor,
    *,
    grid: int,
    max_k: int,
) -> torch.Tensor:
    pointer = model.pointer
    memory = output.memory
    batch, count, _ = memory.shape
    if batch != 1:
        raise ValueError("no-prefix diagnostic requires batch size one")
    slots = pointer._slot_tokens(grid).to(memory)  # noqa: SLF001
    keys = F.normalize(pointer.key(memory), dim=2)
    initial = pointer.initial(memory.mean(1))
    border = pointer._border_logits(memory, grid=grid)  # noqa: SLF001
    start = pointer.start.unsqueeze(0)
    used = torch.zeros(1, count, dtype=torch.bool, device=memory.device)
    rows = []
    for position in range(count):
        recurrent_input = pointer.input_projection(start + slots[position])
        for recurrent in pointer.recurrent:
            recurrent_input = recurrent(recurrent_input, initial)
        query = F.normalize(pointer.query(recurrent_input), dim=1)
        scale = pointer.log_scale.exp().clamp(1.0, 100.0)
        logits = scale * torch.einsum("bd,bnd->bn", query, keys)
        logits = (logits + border[:, position]).masked_fill(used, -1e4)
        rows.append(logits.topk(max_k, dim=1).indices[0])
        chosen = baseline[:, position]
        used = used.scatter(1, chosen[:, None], True)
    return torch.stack(rows)


@torch.no_grad()
def baseline_guided_pointer_repair(
    model: BorderPointerSorter,
    tiles: torch.Tensor,
    baseline_layout: Any,
    socket_right_scores: Any,
    socket_down_scores: Any,
    *,
    grid: int,
    config: BaselineRepairConfig | None = None,
    trace_topk: int = 5,
) -> BaselineRepairResult:
    """Return at most two preregistered strict baseline-repair layouts."""

    count = grid * grid
    config = BaselineRepairConfig() if config is None else config
    config.validate(count=count)
    if not 1 <= trace_topk < count:
        raise ValueError("trace_topk must be in [1, count - 1]")
    baseline_numpy = _strict_layout(baseline_layout, count=count)
    baseline = torch.from_numpy(baseline_numpy.astype(np.int64)).unsqueeze(0).to(tiles.device)
    model.eval()
    output = model(tiles, teacher_layout=baseline, grid=grid)
    if output.pointer_logits is None:
        raise RuntimeError("baseline-prefix pointer trace was not produced")
    prefix_topk = output.pointer_logits[0].topk(trace_topk, dim=1).indices.cpu().numpy()
    no_prefix_topk = _no_prefix_topk(
        model,
        output,
        baseline,
        grid=grid,
        max_k=trace_topk,
    ).cpu().numpy()

    score_matrices = {
        "right": _real_score_matrix(socket_right_scores, count=count, name="right scores"),
        "down": _real_score_matrix(socket_down_scores, count=count, name="down scores"),
    }
    support = {
        axis: _topk_support(scores, topk=config.socket_support_topk)
        for axis, scores in score_matrices.items()
    }
    pointer = model.pointer
    memory = output.memory
    slots = pointer._slot_tokens(grid).to(memory)  # noqa: SLF001
    keys = F.normalize(pointer.key(memory), dim=2)
    initial = pointer.initial(memory.mean(1))
    hidden = [initial for _ in pointer.recurrent]
    previous = pointer.start.unsqueeze(0)
    used = torch.zeros(1, count, dtype=torch.bool, device=memory.device)
    border_logits = pointer._border_logits(memory, grid=grid)  # noqa: SLF001
    prefix: list[torch.Tensor] = []
    current = baseline_numpy.copy()
    snapshots: dict[int, np.ndarray] = {}
    proposals: list[dict[str, Any]] = []
    accepted = 0
    maximum_budget = config.budgets[-1]
    for position in range(count):
        recurrent_input = pointer.input_projection(previous + slots[position])
        for layer, recurrent in enumerate(pointer.recurrent):
            hidden[layer] = recurrent(recurrent_input, hidden[layer])
            recurrent_input = hidden[layer]
        logits = pointer._step_logits(  # noqa: SLF001
            hidden[-1],
            keys,
            used,
            position=position,
            grid=grid,
            prefix=prefix,
            right_logits=output.right_logits,
            down_logits=output.down_logits,
            border_logits=border_logits,
        )[0]
        current_tile = int(current[position])
        proposed_tile = int(logits.argmax())
        margin = float(logits[proposed_tile] - logits[current_tile])
        record: dict[str, Any] | None = None
        if (
            accepted < maximum_budget
            and proposed_tile != current_tile
            and margin >= config.logit_margin
        ):
            future_matches = np.flatnonzero(current[position + 1 :] == proposed_tile)
            if len(future_matches) != 1:
                raise RuntimeError("unused pointer proposal has no unique future position")
            future_position = position + 1 + int(future_matches[0])
            edges = _affected_edges(grid=grid, positions=(position, future_position))
            supported_before, energy_before = _edge_diagnostics(
                current,
                edges=edges,
                support=support,
                scores=score_matrices,
            )
            candidate = current.copy()
            candidate[position], candidate[future_position] = (
                candidate[future_position],
                candidate[position],
            )
            supported_after, energy_after = _edge_diagnostics(
                candidate,
                edges=edges,
                support=support,
                scores=score_matrices,
            )
            guard_pass = supported_after >= supported_before
            if guard_pass:
                current = candidate
                current_tile = proposed_tile
                accepted += 1
                if accepted in config.budgets:
                    snapshots[accepted] = current.copy()
            record = {
                "raster_position": position,
                "proposed_tile": proposed_tile,
                "future_position": future_position,
                "logit_margin": margin,
                "supported_edges_before": supported_before,
                "supported_edges_after": supported_after,
                "socket_energy_delta": energy_after - energy_before,
                "guard_pass": guard_pass,
                "accepted": guard_pass,
                "accepted_count_after": accepted,
            }
        if record is not None:
            proposals.append(record)
        chosen = torch.tensor([current_tile], device=memory.device)
        prefix.append(chosen)
        used = used.scatter(1, chosen[:, None], True)
        previous = memory[:, current_tile]

    for budget in config.budgets:
        if budget not in snapshots:
            snapshots[budget] = current.copy()
        _strict_layout(snapshots[budget], count=count)
    return BaselineRepairResult(
        layouts={budget: snapshots[budget] for budget in config.budgets},
        proposals=tuple(proposals),
        trace=BaselinePrefixTrace(
            prefix_topk=np.ascontiguousarray(prefix_topk.astype(np.int32)),
            no_prefix_topk=np.ascontiguousarray(no_prefix_topk.astype(np.int32)),
        ),
    )


__all__ = [
    "BaselinePrefixTrace",
    "BaselineRepairConfig",
    "BaselineRepairResult",
    "baseline_guided_pointer_repair",
]
