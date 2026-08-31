from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from aiijc_puzzle.protocol import IMAGE_SIZE, sha256_file, split_tiles
from aiijc_puzzle.raw_twin_union_production import (
    FROZEN_SOCKET_SHA256,
    FROZEN_TWIN_SHA256,
    FROZEN_UNION_CHECKPOINT_SHA256,
    FROZEN_UNION_CONFIG_SHA256,
    FROZEN_UNION_SELECTION_SHA256,
    LoadedFullResolutionTwinCheckpoint,
    LoadedRawTwinUnionCheckpoint,
    _compatible_device,
    load_fullres_twin_checkpoint,
    load_raw_twin_union_checkpoint,
    predict_raw_twin_union_variants,
)
from aiijc_puzzle.socket_sorter_production import (
    IDENTITY_PIXEL_TAIL,
    LoadedSocketCheckpoint,
    choose_deterministic_device,
    load_socket_checkpoint,
    predict_socket_sorter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOCKET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
TWIN_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-twin-side-matcher/v1-fit256-s400-eval24"
    / "fullres-twin-side-matcher.pt"
)
UNION_DIR = PROJECT_ROOT / "outputs/raw-twin-union-reranker/v2-fit256-s400-eval24"
UNION_CHECKPOINT = UNION_DIR / "raw-twin-union-reranker-v2.pt"
UNION_SELECTION = UNION_DIR / "selection-commitment.json"
UNION_CONFIG = PROJECT_ROOT / "configs/raw_twin_union_reranker_v2_preregistered.json"


def test_device_normalization_accepts_default_mps_index_only() -> None:
    assert _compatible_device(torch.device("mps:0"), torch.device("mps"))
    assert _compatible_device(torch.device("cpu"), torch.device("cpu"))
    assert not _compatible_device(torch.device("mps:1"), torch.device("mps"))
    assert not _compatible_device(torch.device("cpu"), torch.device("mps"))


@pytest.fixture(scope="module")
def loaded_models() -> tuple[
    torch.device,
    LoadedSocketCheckpoint,
    LoadedFullResolutionTwinCheckpoint,
    LoadedRawTwinUnionCheckpoint,
]:
    device = choose_deterministic_device("cpu")
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    twin = load_fullres_twin_checkpoint(TWIN_CHECKPOINT, device=device)
    union = load_raw_twin_union_checkpoint(
        UNION_CHECKPOINT,
        config_path=UNION_CONFIG,
        selection_path=UNION_SELECTION,
        device=device,
    )
    return device, socket, twin, union


def _fixture_image(seed: int = 20260921) -> np.ndarray:
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


def test_frozen_loaders_bind_all_sha_contract_and_lineage(
    loaded_models: tuple[
        torch.device,
        LoadedSocketCheckpoint,
        LoadedFullResolutionTwinCheckpoint,
        LoadedRawTwinUnionCheckpoint,
    ],
    tmp_path: Path,
) -> None:
    device, socket, twin, union = loaded_models
    assert socket.sha256 == FROZEN_SOCKET_SHA256
    assert twin.sha256 == FROZEN_TWIN_SHA256
    assert union.sha256 == FROZEN_UNION_CHECKPOINT_SHA256
    assert union.config_sha256 == FROZEN_UNION_CONFIG_SHA256
    assert union.selection_sha256 == FROZEN_UNION_SELECTION_SHA256
    assert union.socket_checkpoint_sha256 == socket.sha256
    assert union.twin_checkpoint_sha256 == twin.sha256
    assert twin.lineage.fit_count == union.lineage.fit_count == 256
    assert twin.lineage.evaluation_count == union.lineage.evaluation_count == 24
    assert sum(parameter.numel() for parameter in twin.model.parameters()) == 61_970
    assert sum(parameter.numel() for parameter in union.model.parameters()) == 54_449
    assert all(not model.training for model in (socket.model, twin.model, union.model))
    assert all(
        not parameter.requires_grad
        for model in (socket.model, twin.model, union.model)
        for parameter in model.parameters()
    )

    payload = torch.load(UNION_CHECKPOINT, map_location="cpu", weights_only=True)
    payload["contract"]["raw_topk"] = 31
    tampered = tmp_path / "tampered-union.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="architecture contract changed"):
        load_raw_twin_union_checkpoint(
            tampered,
            config_path=UNION_CONFIG,
            selection_path=UNION_SELECTION,
            device=device,
            expected_checkpoint_sha256=sha256_file(tampered),
        )


def test_no_union_artifacts_fall_back_to_existing_baseline_bit_for_bit(
    loaded_models: tuple[
        torch.device,
        LoadedSocketCheckpoint,
        LoadedFullResolutionTwinCheckpoint,
        LoadedRawTwinUnionCheckpoint,
    ],
) -> None:
    device, socket, _, _ = loaded_models
    image = _fixture_image(20260922)
    adapter = predict_raw_twin_union_variants(image, socket, device=device)
    baseline = predict_socket_sorter(
        image,
        socket,
        device=device,
        cyclic_border5=True,
        pixel_tail=IDENTITY_PIXEL_TAIL,
    )
    assert adapter.selected_variant == "baseline"
    assert adapter.fallback_reason == "union-artifacts-not-configured"
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


def test_union_full_board_is_deterministic_and_preserves_original_upright_tiles(
    loaded_models: tuple[
        torch.device,
        LoadedSocketCheckpoint,
        LoadedFullResolutionTwinCheckpoint,
        LoadedRawTwinUnionCheckpoint,
    ],
) -> None:
    device, socket, twin, union = loaded_models
    image = _fixture_image(20260923)
    first = predict_raw_twin_union_variants(
        image,
        socket,
        device=device,
        twin=twin,
        union=union,
    )
    second = predict_raw_twin_union_variants(
        image,
        socket,
        device=device,
        twin=twin,
        union=union,
    )
    assert first.selected_variant == "raw-twin-union-v2"
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
    assert first.inference_report is not None
    assert first.inference_report["candidate_snapshot"]["count"] > 0
    assert len(first.inference_report["candidate_snapshot"]["sha256"]) == 64
    assert not first.inference_report["candidate_snapshot"][
        "contains_targets_or_absolute_slots"
    ]
    assert first.inference_report["hard_projection_edges_per_axis"] == 552
    assert first.inference_report["hard_projection_inside_immutable_union"]
    assert first.inference_report["candidate_row_minimum"] >= 32
    assert first.inference_report["candidate_row_maximum"] <= 65
    assert first.report()["policy"] == {
        "default_without_union_artifacts": "baseline",
        "targets_manifest_or_filenames_accepted": False,
        "restored_or_generated_pixels_used": False,
        "candidate_union": "raw32+twin32+frozen-raw-hard-projection",
        "restricted_partial_ot": True,
        "decoder_edge_budget_per_axis": 144,
        "cyclic_border_weight": 5.0,
        "all_original_upright_tiles_used_exactly_once": True,
    }


def test_partial_union_activation_and_lineage_mismatch_fail_closed(
    loaded_models: tuple[
        torch.device,
        LoadedSocketCheckpoint,
        LoadedFullResolutionTwinCheckpoint,
        LoadedRawTwinUnionCheckpoint,
    ],
) -> None:
    device, socket, twin, union = loaded_models
    image = _fixture_image(20260924)
    with pytest.raises(ValueError, match="requires both Twin and reranker"):
        predict_raw_twin_union_variants(
            image,
            socket,
            device=device,
            twin=twin,
        )
    incompatible = replace(union, socket_checkpoint_sha256="f" * 64)
    with pytest.raises(ValueError, match="not trained against this Socket checkpoint"):
        predict_raw_twin_union_variants(
            image,
            socket,
            device=device,
            twin=twin,
            union=incompatible,
        )
