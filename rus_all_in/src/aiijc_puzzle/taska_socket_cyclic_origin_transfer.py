"""Frozen Socket cyclic-origin transfer for a strict TASKA layout.

This is a deliberately thin integration layer.  It does not learn a new
origin model and it does not inspect raw seams or a clean/reference layout.
The only candidate family is the 576 whole-board cyclic rolls already scored
by :mod:`aiijc_puzzle.socket_translation_placer`; the input layout may come
from the stronger TASKA six-arm solver rather than the Socket decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    CyclicTranslationDiagnostics,
    select_global_cyclic_translation,
)


@dataclass(frozen=True)
class TaskaSocketOriginTransferResult:
    """A strict rolled layout plus frozen Socket diagnostics."""

    layout: np.ndarray
    diagnostics: CyclicTranslationDiagnostics


def transfer_socket_cyclic_origin(
    control_layout: Any,
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    grid: int = 24,
) -> TaskaSocketOriginTransferResult:
    """Roll a TASKA control with the confirmed Socket border5 objective.

    The weight and minimum-gain values are intentionally not arguments.  This
    prevents this transfer experiment from silently tuning the independently
    confirmed ``border_weight=5`` primitive on an opened panel.
    """

    result = select_global_cyclic_translation(
        control_layout,
        right_log_assignment,
        down_log_assignment,
        grid=grid,
        config=CyclicTranslationConfig(border_weight=5.0, minimum_gain=1e-9),
    )
    count = grid * grid
    layout = np.asarray(result.layout, dtype=np.int32)
    if layout.shape != (count,) or not np.array_equal(
        np.sort(layout), np.arange(count, dtype=np.int32)
    ):
        raise RuntimeError("Socket origin transfer violated strict permutation")
    return TaskaSocketOriginTransferResult(
        layout=np.ascontiguousarray(layout), diagnostics=result.diagnostics
    )


__all__ = [
    "TaskaSocketOriginTransferResult",
    "transfer_socket_cyclic_origin",
]
