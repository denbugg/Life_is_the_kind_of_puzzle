from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from aiijc_puzzle.compliant_submission import (
    _apply_frozen_harmonizers,
    _proper_rgb_nlm_h20_once,
)
from aiijc_puzzle.legacy_upgrade import atomic_write_png
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_COUNT, split_tiles
from aiijc_puzzle.socket_matcher import (
    BORDER_HEAD_EMBEDDING_V2,
    BORDER_HEAD_SCORE_STATS_V3,
    SocketMatcher,
)
from aiijc_puzzle.socket_sorter_production import (
    HISTORICAL_RGB_LUMA_NLM_H20_TAIL,
    IDENTITY_PIXEL_TAIL,
    LoadedSocketCheckpoint,
    assemble_audited_original_tiles,
    choose_deterministic_device,
    load_socket_checkpoint,
    run_socket_sorter_directory,
)


def _lineage_digest(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _write_checkpoint(path: Path, *, version: int = 2) -> None:
    torch.manual_seed(701 + version)
    border_version = (
        BORDER_HEAD_EMBEDDING_V2 if version == 2 else BORDER_HEAD_SCORE_STATS_V3
    )
    architecture = f"board-conditioned-partial-socket-matcher-v{version}"
    model = SocketMatcher(
        dimension=4,
        heads=1,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=2,
        border_head_version=border_version,
    )
    lineage_train = ["train_a.png"]
    lineage_exposed = ["train_a.png", "train_b.png"]
    contract: dict[str, object] = {
        "architecture": architecture,
        "dimension": 4,
        "heads": 1,
        "board_layers": 1,
        "socket_layers": 1,
        "sinkhorn_iterations": 2,
        "synthetic_grid": 24,
        "input_index_position_embedding": False,
    }
    if version == 3:
        contract["border_head_version"] = border_version
    torch.save(
        {
            "contract": contract,
            "state_dict": model.state_dict(),
            "selection": {
                "lineage_train_filenames": lineage_train,
                "lineage_train_digest": _lineage_digest(lineage_train),
                "lineage_exposed_filenames": lineage_exposed,
                "lineage_exposed_digest": _lineage_digest(lineage_exposed),
            },
        },
        path,
    )


def _fixture_image(seed: int = 709) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(
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


def test_strict_checkpoint_loader_accepts_v2_v3_and_rejects_v1(tmp_path: Path) -> None:
    device = choose_deterministic_device("cpu")
    v2 = tmp_path / "v2.pt"
    v3 = tmp_path / "v3.pt"
    _write_checkpoint(v2, version=2)
    _write_checkpoint(v3, version=3)
    loaded_v2 = load_socket_checkpoint(v2, device=device)
    loaded_v3 = load_socket_checkpoint(v3, device=device)
    assert isinstance(loaded_v2, LoadedSocketCheckpoint)
    assert loaded_v2.resolved_border_head_version == BORDER_HEAD_EMBEDDING_V2
    assert loaded_v3.resolved_border_head_version == BORDER_HEAD_SCORE_STATS_V3
    assert loaded_v2.lineage.train_count == 1
    assert loaded_v2.lineage.exposed_count == 2
    assert loaded_v2.lineage.train_filenames == ("train_a.png",)
    assert loaded_v2.lineage.exposed_filenames == ("train_a.png", "train_b.png")
    assert "exposed_filenames" not in loaded_v2.lineage.as_dict()
    assert all(not parameter.requires_grad for parameter in loaded_v3.model.parameters())

    payload = torch.load(v2, map_location="cpu", weights_only=True)
    payload["contract"]["architecture"] = "board-conditioned-partial-socket-matcher-v1"
    v1 = tmp_path / "v1.pt"
    torch.save(payload, v1)
    with pytest.raises(ValueError, match="unsupported SocketMatcher architecture"):
        load_socket_checkpoint(v1, device=device)


def test_checkpoint_loader_rejects_position_embedding_and_bad_lineage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.pt"
    _write_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["contract"]["input_index_position_embedding"] = True
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="shuffled-index embeddings"):
        load_socket_checkpoint(checkpoint, device=torch.device("cpu"))

    _write_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["selection"]["lineage_exposed_digest"] = "0" * 64
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="lineage_exposed_digest"):
        load_socket_checkpoint(checkpoint, device=torch.device("cpu"))


def test_audited_assembly_fails_on_missing_or_duplicate_tile_identity() -> None:
    image = _fixture_image()
    layout = np.arange(TILE_COUNT, dtype=np.int32)
    raw, audit = assemble_audited_original_tiles(image, layout)
    assert np.array_equal(raw, image)
    assert audit.passed

    invalid = layout.copy()
    invalid[-1] = invalid[0]
    with pytest.raises(ValueError, match="every integer tile identity exactly once"):
        assemble_audited_original_tiles(image, invalid)
    with pytest.raises(ValueError, match="every integer tile identity exactly once"):
        assemble_audited_original_tiles(image, layout.astype(np.float64))
    with pytest.raises(ValueError, match="every integer tile identity exactly once"):
        assemble_audited_original_tiles(image, layout.astype(bool))


def test_identity_tail_is_separate_and_preserves_pixels() -> None:
    image = _fixture_image(711)
    restored = IDENTITY_PIXEL_TAIL.apply(image)
    assert np.array_equal(restored, image)
    assert restored is not image
    assert IDENTITY_PIXEL_TAIL.target_blind
    assert IDENTITY_PIXEL_TAIL.post_layout_only


def test_historical_h20_tail_is_post_layout_and_content_bound() -> None:
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 127, dtype=np.uint8)
    restored = HISTORICAL_RGB_LUMA_NLM_H20_TAIL.apply(image)
    assert restored.shape == image.shape
    assert restored.dtype == np.uint8
    assert np.array_equal(restored, image)
    assert HISTORICAL_RGB_LUMA_NLM_H20_TAIL.target_blind
    assert HISTORICAL_RGB_LUMA_NLM_H20_TAIL.post_layout_only
    evidence = HISTORICAL_RGB_LUMA_NLM_H20_TAIL.evidence
    assert evidence is not None
    assert evidence["layout_changed"] is False
    assert evidence["nlm"]["passes"] == 1
    assert evidence["rgb_config_sha256"] == (
        "4adfd9b614e8556b7de5c1f527d759d15d29c0f74e20aa26ff87900dd773ec9a"
    )
    assert evidence["luma_config_sha256"] == (
        "7488cad2ae7cc75792d6ff0ff2ea0a38fa778979083ffd5c161c857b68fd550f"
    )

    nonconstant = _fixture_image(712)
    historical = HISTORICAL_RGB_LUMA_NLM_H20_TAIL.apply(nonconstant)
    frozen_production = _proper_rgb_nlm_h20_once(_apply_frozen_harmonizers(nonconstant))
    assert np.array_equal(historical, frozen_production)


def test_one_fixture_directory_smoke_is_compliant_and_resumable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "socket.pt"
    _write_checkpoint(checkpoint)
    source = tmp_path / "source"
    source.mkdir()
    filename = "img_000001.png"
    input_image = _fixture_image(713)
    atomic_write_png(source / filename, input_image)
    output = tmp_path / "output"

    first = run_socket_sorter_directory(
        checkpoint_path=checkpoint,
        source_dir=source,
        output_dir=output,
        device_name="cpu",
        cyclic_border5=True,
        pixel_tail_name="identity",
    )
    assert first["processed"] == 1
    assert first["resumed"] == 0
    with Image.open(output / filename) as image:
        image.load()
        predicted = np.asarray(image, dtype=np.uint8)
    assert _tile_multiset(predicted) == _tile_multiset(input_image)

    record = json.loads((output / "records" / f"{filename}.json").read_text())
    assert record["raw_assembly"]["audit"]["passed"] is True
    assert record["layout"]["all_576_original_tiles_used_exactly_once"] is True
    assert record["pixel_tail"]["name"] == "identity"
    assert record["diagnostics"]["cyclic_translation"]["placer"] == (
        "socket-global-cyclic-translation-v1"
    )
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "COMPLETE"
    assert run["pipeline"]["policy"] == {
        "all_original_tiles_used_exactly_once_before_tail": True,
        "constant_canvas_used": False,
        "external_templates_used": False,
        "source_lookup_used": False,
        "targets_or_manifest_labels_used": False,
        "tile_warp_or_resize_used": False,
    }

    second = run_socket_sorter_directory(
        checkpoint_path=checkpoint,
        source_dir=source,
        output_dir=output,
        device_name="cpu",
        cyclic_border5=True,
        pixel_tail_name="identity",
    )
    assert second["processed"] == 0
    assert second["resumed"] == 1

    (output / "rogue.txt").write_text("foreign artifact", encoding="utf-8")
    with pytest.raises(ValueError, match="foreign artifact"):
        run_socket_sorter_directory(
            checkpoint_path=checkpoint,
            source_dir=source,
            output_dir=output,
            device_name="cpu",
            cyclic_border5=True,
            pixel_tail_name="identity",
        )


def test_resume_fails_closed_when_output_is_tampered(tmp_path: Path) -> None:
    checkpoint = tmp_path / "socket.pt"
    _write_checkpoint(checkpoint)
    source = tmp_path / "source"
    source.mkdir()
    filename = "img_000002.png"
    atomic_write_png(source / filename, _fixture_image(719))
    output = tmp_path / "output"
    run_socket_sorter_directory(
        checkpoint_path=checkpoint,
        source_dir=source,
        output_dir=output,
        cyclic_border5=False,
    )
    atomic_write_png(output / filename, np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="output PNG hash mismatch"):
        run_socket_sorter_directory(
            checkpoint_path=checkpoint,
            source_dir=source,
            output_dir=output,
            cyclic_border5=False,
        )
