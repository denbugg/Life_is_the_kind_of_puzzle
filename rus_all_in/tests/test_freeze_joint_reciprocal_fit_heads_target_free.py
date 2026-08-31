from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.tri_emitter_edge_verifier import (
    AUXILIARY_DIM,
    DINO_PROJECTION_DIM,
    EMITTERS,
    TOP_K,
)
from scripts import freeze_joint_reciprocal_fit_heads_target_free as freezer


def _target_free_arrays(count: int = 4) -> dict[str, np.ndarray]:
    width = count - 1
    candidates = np.empty((2, count, width), dtype=np.int32)
    for axis in range(2):
        for source in range(count):
            candidates[axis, source] = [
                target for target in range(count) if target != source
            ]
    emitter = np.broadcast_to(
        candidates[None, ...],
        (len(EMITTERS), 2, count, min(TOP_K, count - 1)),
    ).copy()
    return {
        "raw_sides": np.zeros((4, count, 20, 6), dtype=np.float16),
        "dino_sides": np.zeros(
            (4, count, 14, DINO_PROJECTION_DIM), dtype=np.float16
        ),
        "candidates": candidates,
        "valid": np.ones((2, count, width), dtype=bool),
        "auxiliary": np.zeros(
            (2, count, width, AUXILIARY_DIM), dtype=np.float16
        ),
        "raw_baseline": np.zeros((2, count, width), dtype=np.float16),
        "emitter_topk": emitter,
    }


def _write_cache(path: Path, *, extra: dict[str, np.ndarray] | None = None) -> None:
    arrays = _target_free_arrays()
    # Object dtype is deliberate: allow_pickle=False raises if this payload is
    # ever indexed, so successful loading proves the label member stayed closed.
    arrays["target_slots"] = np.asarray([["secret"] * 4] * 2, dtype=object)
    arrays.update(extra or {})
    np.savez_compressed(path, **arrays)


def test_strict_loader_never_materialises_target_slots(tmp_path: Path) -> None:
    cache = tmp_path / "fit-cache.npz"
    _write_cache(cache)
    values, materialised = freezer.load_target_free_fit_cache(
        cache,
        expected_sha256=sha256_file(cache),
        expected_tile_count=4,
    )
    assert set(values) == freezer.TARGET_FREE_CACHE_KEYS
    assert set(materialised) == freezer.TARGET_FREE_CACHE_KEYS
    assert "target_slots" not in materialised
    with (
        np.load(cache, allow_pickle=False) as archive,
        pytest.raises(ValueError, match="Object arrays"),
    ):
        _ = archive["target_slots"]


def test_strict_loader_fails_closed_on_member_inventory_drift(tmp_path: Path) -> None:
    cache = tmp_path / "fit-cache-extra-label.npz"
    _write_cache(cache, extra={"truth": np.zeros((2, 4), dtype=np.int16)})
    with pytest.raises(RuntimeError, match="member-name inventory"):
        freezer.load_target_free_fit_cache(
            cache,
            expected_sha256=sha256_file(cache),
            expected_tile_count=4,
        )


def test_strict_loader_checks_cache_hash_before_open(tmp_path: Path) -> None:
    cache = tmp_path / "fit-cache.npz"
    _write_cache(cache)
    with pytest.raises(RuntimeError, match="bytes changed"):
        freezer.load_target_free_fit_cache(
            cache,
            expected_sha256="0" * 64,
            expected_tile_count=4,
        )
