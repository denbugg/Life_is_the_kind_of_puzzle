from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from aiijc_puzzle.direct_hard_edge_production import (
    FROZEN_DIRECT_CONFIG_SHA256,
    FROZEN_DIRECT_HARD_EDGE_SHA256,
    FROZEN_SOCKET_SHA256,
    load_direct_hard_edge_checkpoint,
    predict_direct_hard_edge_variants,
)
from aiijc_puzzle.protocol import IMAGE_SIZE, sha256_file, split_tiles
from aiijc_puzzle.socket_sorter_production import (
    IDENTITY_PIXEL_TAIL,
    choose_deterministic_device,
    load_socket_checkpoint,
    predict_socket_sorter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOCKET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
DIRECT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/direct-hard-edge-priority/v1-fit256-s600-d1-32-cpu"
    / "direct_hard_edge_priority.pt"
)


def _fixture_image(seed: int = 20260919) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0,
        256,
        size=(IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )


def _tile_multiset(image: np.ndarray) -> list[str]:
    return sorted(
        hashlib.sha256(np.ascontiguousarray(tile).tobytes()).hexdigest()
        for tile in split_tiles(image)
    )


def test_frozen_direct_loader_binds_sha_contract_and_lineage(tmp_path: Path) -> None:
    device = torch.device("cpu")
    loaded = load_direct_hard_edge_checkpoint(DIRECT_CHECKPOINT, device=device)
    assert loaded.sha256 == FROZEN_DIRECT_HARD_EDGE_SHA256
    assert loaded.config_sha256 == FROZEN_DIRECT_CONFIG_SHA256
    assert loaded.socket_checkpoint_sha256 == FROZEN_SOCKET_SHA256
    assert loaded.lineage.fit_count == 256
    assert loaded.lineage.d1_count == 32
    assert sum(parameter.numel() for parameter in loaded.model.parameters()) == 47_057
    assert not loaded.model.training
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())

    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_direct_hard_edge_checkpoint(
            DIRECT_CHECKPOINT,
            device=device,
            expected_sha256="0" * 64,
        )

    payload = torch.load(DIRECT_CHECKPOINT, map_location="cpu", weights_only=True)
    payload["contract"]["input_dimension"] = 295
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="architecture contract changed"):
        load_direct_hard_edge_checkpoint(
            tampered,
            device=device,
            expected_sha256=sha256_file(tampered),
        )


def test_no_direct_checkpoint_falls_back_to_existing_baseline_bit_for_bit() -> None:
    device = choose_deterministic_device("cpu")
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    image = _fixture_image(20260920)
    adapter = predict_direct_hard_edge_variants(image, socket, device=device)
    baseline = predict_socket_sorter(
        image,
        socket,
        device=device,
        cyclic_border5=True,
        pixel_tail=IDENTITY_PIXEL_TAIL,
    )
    assert adapter.selected_variant == "baseline"
    assert adapter.fallback_reason == "direct-checkpoint-not-configured"
    assert adapter.learned is None
    assert adapter.selected is adapter.baseline
    assert np.array_equal(adapter.baseline.layout, baseline.layout)
    assert np.array_equal(adapter.baseline.raw, baseline.raw)
    assert np.array_equal(adapter.baseline.output, baseline.output)
    assert adapter.baseline.decoder_report["layout_sha256"] == (
        baseline.decoder_report["layout_sha256"]
    )
    assert adapter.baseline.cyclic_report["layout_sha256"] == (
        baseline.cyclic_report["layout_sha256"]
    )


def test_learned_adapter_is_deterministic_and_preserves_original_upright_tiles() -> None:
    device = choose_deterministic_device("cpu")
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    direct = load_direct_hard_edge_checkpoint(DIRECT_CHECKPOINT, device=device)
    image = _fixture_image(20260921)
    first = predict_direct_hard_edge_variants(
        image,
        socket,
        device=device,
        direct=direct,
    )
    second = predict_direct_hard_edge_variants(
        image,
        socket,
        device=device,
        direct=direct,
    )
    assert first.selected_variant == "direct-hard-edge"
    assert first.fallback_reason is None
    assert first.learned is not None and second.learned is not None
    assert np.array_equal(first.baseline.layout, second.baseline.layout)
    assert np.array_equal(first.learned.layout, second.learned.layout)
    assert np.array_equal(first.selected.layout, first.learned.layout)
    expected_tiles = _tile_multiset(image)
    for arm in (first.baseline, first.learned):
        assert arm.audit.passed
        assert np.array_equal(arm.raw, arm.output)
        assert _tile_multiset(arm.raw) == expected_tiles
        assert np.array_equal(np.sort(arm.layout), np.arange(24 * 24))
    assert first.priority_report is not None
    assert first.priority_report["hard_edges"] == 1104
    assert first.priority_report["hard_edges_per_axis"] == {"right": 552, "down": 552}
    assert first.report()["policy"] == {
        "default_without_direct_checkpoint": "baseline",
        "targets_or_manifest_labels_used": False,
        "restored_only_candidates_used": False,
        "cyclic_border_weight": 5.0,
        "all_original_upright_tiles_used_exactly_once": True,
    }


def test_direct_checkpoint_socket_mismatch_fails_closed() -> None:
    device = choose_deterministic_device("cpu")
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    direct = load_direct_hard_edge_checkpoint(DIRECT_CHECKPOINT, device=device)
    incompatible = type(direct)(
        path=direct.path,
        sha256=direct.sha256,
        model=direct.model,
        contract=direct.contract,
        config_sha256=direct.config_sha256,
        socket_checkpoint_sha256="f" * 64,
        lineage=direct.lineage,
    )
    with pytest.raises(ValueError, match="not trained against this Socket checkpoint"):
        predict_direct_hard_edge_variants(
            _fixture_image(20260922),
            socket,
            device=device,
            direct=incompatible,
        )
