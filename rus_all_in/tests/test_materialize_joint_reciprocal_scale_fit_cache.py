from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aiijc_puzzle.protocol import compute_protocol_digest
from aiijc_puzzle.synthetic_socket_evaluation import (
    names_digest,
    select_source_disjoint_train_records,
)
from scripts import materialize_joint_reciprocal_scale_fit_cache as cache


def _template() -> dict[str, object]:
    return json.loads(cache.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _populate_real_rosters(config: dict[str, object]) -> dict[str, object]:
    manifest = json.loads(cache.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    socket = json.loads(
        (
            cache.PROJECT_ROOT
            / cache.EXPECTED_FROZEN_PATHS["socket_parent_report"]
        ).read_text(encoding="utf-8")
    )
    adapter = json.loads(
        (
            cache.PROJECT_ROOT
            / cache.EXPECTED_FROZEN_PATHS["adapter_parent_report"]
        ).read_text(encoding="utf-8")
    )
    groups = cache._parent_source_groups(socket, adapter)
    excluded = sorted(set().union(*(set(value) for value in groups.values())))
    selected = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=excluded,
        limit=cache.FIT_SOURCE_COUNT + cache.RESERVED_DEV_SOURCE_COUNT,
        seed=cache.SELECTION_SEED,
        namespace=cache.SELECTION_NAMESPACE,
    )
    names = [str(record["filename"]) for record in selected]
    source = config["source_protocol"]
    source["fit_filenames"] = names[: cache.FIT_SOURCE_COUNT]
    source["fit_digest"] = names_digest(source["fit_filenames"])
    source["reserved_dev_filenames"] = names[cache.FIT_SOURCE_COUNT :]
    source["reserved_dev_digest"] = names_digest(source["reserved_dev_filenames"])
    return config


def _fake_parent_reports() -> tuple[dict[str, object], dict[str, object]]:
    socket = json.loads(
        (
            cache.PROJECT_ROOT
            / cache.EXPECTED_FROZEN_PATHS["socket_parent_report"]
        ).read_text(encoding="utf-8")
    )
    adapter = json.loads(
        (
            cache.PROJECT_ROOT
            / cache.EXPECTED_FROZEN_PATHS["adapter_parent_report"]
        ).read_text(encoding="utf-8")
    )
    return socket, adapter


def _fake_manifest(records: list[dict[str, str]]) -> dict[str, object]:
    manifest: dict[str, object] = {"splits": {"train": records}}
    manifest["protocol_digest"] = compute_protocol_digest(manifest)
    return manifest


def _populated_fake_contract() -> tuple[
    dict[str, object], dict[str, object], dict[str, object], dict[str, object]
]:
    config = _template()
    socket, adapter = _fake_parent_reports()
    groups = cache._parent_source_groups(socket, adapter)
    excluded = sorted(set().union(*(set(value) for value in groups.values())))
    available = [f"available_{index:04d}.png" for index in range(400)]
    records = [
        {
            "filename": name,
            "input_sha256": "1" * 64,
            "target_sha256": "2" * 64,
        }
        for name in [*excluded, *available]
    ]
    manifest = _fake_manifest(records)
    selected = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=excluded,
        limit=cache.FIT_SOURCE_COUNT + cache.RESERVED_DEV_SOURCE_COUNT,
        seed=cache.SELECTION_SEED,
        namespace=cache.SELECTION_NAMESPACE,
    )
    names = [str(record["filename"]) for record in selected]
    source = config["source_protocol"]
    source["fit_filenames"] = names[: cache.FIT_SOURCE_COUNT]
    source["fit_digest"] = names_digest(source["fit_filenames"])
    source["reserved_dev_filenames"] = names[cache.FIT_SOURCE_COUNT :]
    source["reserved_dev_digest"] = names_digest(source["reserved_dev_filenames"])
    return config, manifest, socket, adapter


def test_unsigned_template_blocks_before_roster_or_data_access() -> None:
    assert not cache.DEFAULT_CONFIG.with_suffix(".json.sha256").exists()
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        cache.load_signed_config(cache.DEFAULT_CONFIG)


def test_exact_contract_rejects_schema_roster_and_checkpoint_drift() -> None:
    config, _, _, _ = _populated_fake_contract()
    cache.require_exact_contract(config)

    changed = json.loads(json.dumps(config))
    changed["source_protocol"]["fit_filenames"] = changed["source_protocol"][
        "fit_filenames"
    ][:-1]
    with pytest.raises(RuntimeError, match="exactly 256"):
        cache.require_exact_contract(changed)

    changed = json.loads(json.dumps(config))
    changed["cache_contract"]["candidate_roster"] = "raw-only"
    with pytest.raises(RuntimeError, match="schema/identity"):
        cache.require_exact_contract(changed)

    changed = json.loads(json.dumps(config))
    changed["frozen_inputs"]["capacity_checkpoint"] = changed["frozen_inputs"][
        "socket_checkpoint"
    ]
    with pytest.raises(RuntimeError, match="inventory"):
        cache.require_exact_contract(changed)


def test_metadata_selection_is_exact_parent_disjoint_and_reserves_dev() -> None:
    config, manifest, socket, adapter = _populated_fake_contract()
    records, audit = cache.validate_metadata_rosters(config, manifest, socket, adapter)
    fit = {str(record["filename"]) for record in records}
    reserved = set(config["source_protocol"]["reserved_dev_filenames"])
    groups = cache._parent_source_groups(socket, adapter)
    excluded = set().union(*(set(value) for value in groups.values()))

    assert len(records) == cache.FIT_SOURCE_COUNT
    assert len(reserved) == cache.RESERVED_DEV_SOURCE_COUNT
    assert not fit & reserved
    assert not (fit | reserved) & excluded
    assert audit["reserved_dev_opened"] is False
    assert audit["parent_exclusion_count"] == cache.PARENT_EXCLUSION_COUNT

    changed = json.loads(json.dumps(config))
    changed["source_protocol"]["fit_filenames"][0] = next(iter(excluded))
    changed["source_protocol"]["fit_digest"] = names_digest(
        changed["source_protocol"]["fit_filenames"]
    )
    with pytest.raises(RuntimeError, match="not the fixed selection"):
        cache.validate_metadata_rosters(changed, manifest, socket, adapter)


def test_generated_cache_schema_rejects_invalid_identity_and_target_slot() -> None:
    values = {
        "raw_sides": np.zeros((4, 576, 20, 6), dtype=np.float16),
        "dino_sides": np.zeros((4, 576, 14, 16), dtype=np.float16),
        "candidates": np.zeros((2, 576, 96), dtype=np.int32),
        "valid": np.zeros((2, 576, 96), dtype=bool),
        "auxiliary": np.zeros((2, 576, 96, 19), dtype=np.float16),
        "raw_baseline": np.zeros((2, 576, 96), dtype=np.float16),
        "emitter_topk": np.zeros((3, 2, 576, 32), dtype=np.int32),
    }
    targets = np.full((2, 576), -1, dtype=np.int16)
    cache._validate_case_arrays(values, targets)

    values["valid"][0, 0, 0] = True
    values["candidates"][0, 0, 0] = 576
    with pytest.raises(RuntimeError, match="out of range"):
        cache._validate_case_arrays(values, targets)

    values["valid"][0, 0, 0] = False
    targets[0, 0] = 96
    with pytest.raises(RuntimeError, match="target slot"):
        cache._validate_case_arrays(values, targets)


def test_cache_only_run_writes_512_rows_and_no_next_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _populate_real_rosters(_template())
    output = tmp_path / "scale-cache"
    args = argparse.Namespace(
        config=cache.DEFAULT_CONFIG,
        output_dir=output,
        manifest=cache.DEFAULT_MANIFEST,
        targets=cache.DEFAULT_TARGETS,
        socket_checkpoint=cache.DEFAULT_SOCKET_CHECKPOINT,
        device="cpu",
        allow_nondeterministic_mps=False,
    )
    model_loads = 0
    board_loads = 0

    def fake_models(
        _args: argparse.Namespace,
    ) -> tuple[object, object, object, np.ndarray, torch.device]:
        nonlocal model_loads
        model_loads += 1
        return object(), object(), object(), np.zeros((1, 1)), torch.device("cpu")

    def fake_board(record: dict[str, str], _targets: Path) -> SimpleNamespace:
        nonlocal board_loads
        board_loads += 1
        return SimpleNamespace(filename=record["filename"], tiles=None)

    def fake_case(
        *, board: SimpleNamespace, draw_index: int, cache_path: Path, **_: object
    ) -> dict[str, object]:
        cache_path.write_bytes(b"cache-only")
        return {
            "path": cache._path_label(cache_path),
            "sha256": "0" * 64,
            "source_filename": board.filename,
            "draw_index": draw_index,
            "case_id": f"case-{board.filename}-{draw_index}",
            "dirty_sha256": "1" * 64,
            "eligible_queries": 1,
            "candidate_union_identity_digest": "2" * 64,
            "cache_bytes": cache_path.stat().st_size,
            "runtime": {
                "total_seconds": 0.01,
                "cache_write_seconds": 0.001,
                "materializer_case_seconds": 0.02,
            },
        }

    monkeypatch.setattr(cache.prior, "_make_models", fake_models)
    monkeypatch.setattr(cache, "_load_one_board", fake_board)
    monkeypatch.setattr(cache, "_materialize_one_case", fake_case)

    report = cache.run_materialization(args, config, "signed-config-sha")
    assert model_loads == 1
    assert board_loads == cache.FIT_SOURCE_COUNT
    assert report["schema"] == cache.COMPATIBLE_REPORT_SCHEMA
    assert report["producer_schema"] == cache.PRODUCER_SCHEMA
    assert report["fit_cache"]["case_count"] == cache.FIT_CASE_COUNT
    assert len(report["fit_cache"]["rows"]) == cache.FIT_CASE_COUNT
    assert report["scope"]["training_run"] is False
    assert report["scope"]["reserved_dev_pixels_or_labels_opened"] is False
    assert report["next_transition_authorized"] is False
    assert {path.name for path in output.iterdir()} == {"fit-cache", "report.json"}
    assert not list(output.rglob("*.pt"))
    assert not (output / "dev").exists()
    assert not (output / "score.json").exists()

    with pytest.raises(FileExistsError):
        cache.run_materialization(args, config, "signed-config-sha")
    assert model_loads == 1


def test_cli_exposes_no_fit_dev_or_score_mode(tmp_path: Path) -> None:
    args = cache.parse_args(["--output-dir", str(tmp_path / "out"), "--device", "cpu"])
    assert not hasattr(args, "mode")
