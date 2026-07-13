"""Geometry and permutation conventions for the fixed 24x24 puzzle."""

from __future__ import annotations

import numpy as np

GRID = 24
TILE = 20
TILE_COUNT = GRID * GRID


def validate_permutation(values: np.ndarray, *, name: str = "permutation") -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (TILE_COUNT,):
        raise ValueError(f"{name} must have shape {(TILE_COUNT,)}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be integral, got {array.dtype}")
    array = array.astype(np.int32, copy=False)
    if int(array.min()) != 0 or int(array.max()) != TILE_COUNT - 1:
        raise ValueError(f"{name} values must cover [0, {TILE_COUNT - 1}]")
    if len(np.unique(array)) != TILE_COUNT:
        raise ValueError(f"{name} must contain every value exactly once")
    return array


def inverse_permutation(slot_to_target: np.ndarray) -> np.ndarray:
    """Return ``position_to_slot`` for a validated ``slot_to_target`` mapping."""
    slot_to_target = validate_permutation(slot_to_target, name="slot_to_target")
    position_to_slot = np.empty(TILE_COUNT, dtype=np.int32)
    position_to_slot[slot_to_target] = np.arange(TILE_COUNT, dtype=np.int32)
    return position_to_slot


def target_positions(position_to_slot: np.ndarray, slot_to_target: np.ndarray) -> np.ndarray:
    position_to_slot = validate_permutation(position_to_slot, name="position_to_slot")
    slot_to_target = validate_permutation(slot_to_target, name="slot_to_target")
    return slot_to_target[position_to_slot]


def true_neighbour_slots(slot_to_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return true right/down neighbour slot for each query slot, or -1 outside."""
    slot_to_target = validate_permutation(slot_to_target, name="slot_to_target")
    position_to_slot = inverse_permutation(slot_to_target)
    right = np.full(TILE_COUNT, -1, dtype=np.int32)
    down = np.full(TILE_COUNT, -1, dtype=np.int32)
    for slot, position in enumerate(slot_to_target.tolist()):
        row, column = divmod(position, GRID)
        if column + 1 < GRID:
            right[slot] = position_to_slot[position + 1]
        if row + 1 < GRID:
            down[slot] = position_to_slot[position + GRID]
    return right, down
