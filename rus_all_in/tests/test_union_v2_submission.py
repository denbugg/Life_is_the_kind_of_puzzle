from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from aiijc_puzzle.compliant_submission import array_sha256
from aiijc_puzzle.legacy_upgrade import atomic_write_png
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_COUNT
from aiijc_puzzle.socket_pixel_tails import historical_rgb_luma_nlm_h20_once
from aiijc_puzzle.socket_sorter_production import assemble_audited_original_tiles
from aiijc_puzzle.union_v2_submission import (
    FROZEN_PRODUCTION_CONFIG_SHA256,
    _board_record,
    _output_state,
    _record_layout,
    _validate_material,
    load_union_submission_config,
)
from aiijc_puzzle.union_v2_submission_validation import (
    _independent_raw_assembly,
    _independent_tail,
)


def _synthetic_board() -> np.ndarray:
    rows, columns = np.indices((IMAGE_SIZE, IMAGE_SIZE))
    return np.stack(
        (
            (3 * rows + columns) % 256,
            (rows + 5 * columns) % 256,
            (7 * rows + 11 * columns) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def test_frozen_union_v2_config_and_schema_are_self_consistent() -> None:
    config = load_union_submission_config()
    assert config.sha256 == FROZEN_PRODUCTION_CONFIG_SHA256
    schema = __import__("json").loads(
        config.artifacts["attestation_schema"].read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    assert config.payload["policy"]["constant_or_near_flat_tile_substitution_used"] is False
    assert {"opencv", "pillow"}.issubset(
        schema["$defs"]["pipeline"]["properties"]["runtime"]["required"]
    )


def test_independent_union_v2_raw_and_tail_match_production() -> None:
    image = _synthetic_board()
    layout = np.random.default_rng(97).permutation(TILE_COUNT).astype(np.int32)
    raw, audit = assemble_audited_original_tiles(image, layout)
    assert audit.passed
    assert np.array_equal(_independent_raw_assembly(image, layout), raw)
    assert np.array_equal(_independent_tail(raw), historical_rgb_luma_nlm_h20_once(raw))


def test_union_v2_resume_layout_rejects_non_bijection() -> None:
    duplicate = np.arange(TILE_COUNT, dtype=np.int32)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="strict permutation"):
        _record_layout(
            {"layout": {"tile_at_position": duplicate.tolist()}},
            filename="img_000001.png",
        )


def test_union_v2_material_validation_detects_png_tamper(tmp_path: Path) -> None:
    image = _synthetic_board()
    layout = np.arange(TILE_COUNT, dtype=np.int32)
    raw, audit = assemble_audited_original_tiles(image, layout)
    output = historical_rgb_luma_nlm_h20_once(raw)
    output_path = tmp_path / "img_000001.png"
    output_png_sha256 = atomic_write_png(output_path, output)
    pipeline = {"pipeline_digest": "1" * 64}
    record = _board_record(
        filename=output_path.name,
        input_sha256="2" * 64,
        input_image=image,
        layout=layout,
        raw=raw,
        audit=audit,
        output=output,
        output_png_sha256=output_png_sha256,
        pipeline=pipeline,
        diagnostics={},
    )
    _validate_material(
        record=record,
        filename=output_path.name,
        input_sha256="2" * 64,
        input_image=image,
        layout=layout,
        raw=raw,
        output=output,
        output_path=output_path,
        pipeline=pipeline,
    )
    tampered = output.copy()
    tampered[0, 0, 0] ^= 1
    atomic_write_png(output_path, tampered)
    with pytest.raises(ValueError, match="output PNG mismatch"):
        _validate_material(
            record=record,
            filename=output_path.name,
            input_sha256="2" * 64,
            input_image=image,
            layout=layout,
            raw=raw,
            output=output,
            output_path=output_path,
            pipeline=pipeline,
        )
    assert record["raw_assembly"]["array_sha256"] == array_sha256(raw)


def test_union_v2_output_state_rejects_foreign_files(tmp_path: Path) -> None:
    output = tmp_path / "predictions"
    records = output / "records"
    records.mkdir(parents=True)
    (output / "run.json").write_text("{}", encoding="utf-8")
    (output / ".DS_Store").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="foreign file"):
        _output_state(output, records, ["img_000001.png"], require_complete=False)
