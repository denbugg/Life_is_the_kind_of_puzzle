from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import names_digest
from scripts import (
    freeze_taska_relation_selector_roster_scale256_target_free as wrapper,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALE_CONFIG = (
    PROJECT_ROOT / "configs/joint_reciprocal_scale256_real_preregistered_v1.json"
)
SCALE_CACHE_CONFIG = (
    PROJECT_ROOT
    / "configs/joint_reciprocal_scale256_fit_cache_preregistered_v1.json"
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_blocked_template_derives_exact_dev64_from_both_scale_protocols() -> None:
    config = _json(wrapper.DEFAULT_CONFIG)
    scale = _json(SCALE_CONFIG)
    cache = _json(SCALE_CACHE_CONFIG)
    wrapper._require_exact_contract(config)
    derived = wrapper.derive_source_protocol(config, scale, cache)

    assert len(derived["dev_filenames"]) == 64
    assert len(set(derived["dev_filenames"])) == 64
    assert names_digest(derived["dev_filenames"]) == wrapper.DEV_DIGEST
    assert derived["dev_filenames"] == cache["source_protocol"][
        "reserved_dev_filenames"
    ]
    assert derived["dev_digest"] == scale["source_contract"][
        "reserved_dev_digest"
    ]
    assert derived["dev_draw_index"] == 0
    assert derived["dev_case_seed"] == 20260908
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        wrapper._load_signed_config(wrapper.DEFAULT_CONFIG)


def test_source_bridge_fails_closed_on_either_upstream_contract() -> None:
    config = _json(wrapper.DEFAULT_CONFIG)
    scale = _json(SCALE_CONFIG)
    cache = _json(SCALE_CACHE_CONFIG)

    changed_scale = copy.deepcopy(scale)
    changed_scale["source_contract"]["dev_case_seed"] += 1
    with pytest.raises(RuntimeError, match="digest/draw/seed"):
        wrapper.derive_source_protocol(config, changed_scale, cache)

    changed_cache = copy.deepcopy(cache)
    changed_cache["source_protocol"]["reserved_dev_filenames"].reverse()
    with pytest.raises(RuntimeError, match="explicit DEV64 roster"):
        wrapper.derive_source_protocol(config, scale, changed_cache)


def test_template_binds_existing_lineage_and_both_freezer_bytes() -> None:
    config = _json(wrapper.DEFAULT_CONFIG)
    frozen = config["frozen_inputs"]
    assert set(frozen) == wrapper.REQUIRED_FROZEN_INPUTS
    for name, (path, digest) in wrapper.EXISTING_FROZEN_INPUTS.items():
        assert frozen[name] == {"path": path, "sha256": digest}
    assert sha256_file(wrapper._project_path(frozen["base_freezer"]["path"])) == (
        frozen["base_freezer"]["sha256"]
    )
    assert sha256_file(wrapper._project_path(frozen["wrapper_freezer"]["path"])) == (
        frozen["wrapper_freezer"]["sha256"]
    )
    assert frozen["joint_protocol"]["sha256"] == sha256_file(SCALE_CONFIG)
    assert frozen["scale_cache_protocol"]["sha256"] == sha256_file(
        SCALE_CACHE_CONFIG
    )


def test_wrapper_delegates_without_reimplementing_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _json(wrapper.DEFAULT_CONFIG)
    digest = "a" * 64
    args = argparse.Namespace(
        config=wrapper.DEFAULT_CONFIG,
        output_dir=tmp_path / "output",
        manifest=wrapper._project_path(
            config["frozen_inputs"]["validation_manifest"]["path"]
        ),
        targets=wrapper.base.DEFAULT_TARGETS,
    )
    calls: list[tuple[argparse.Namespace, dict[str, Any], str]] = []
    monkeypatch.setattr(wrapper, "_load_signed_config", lambda _path: (config, digest))

    def fake_freeze(
        received_args: argparse.Namespace,
        received_config: dict[str, Any],
        received_digest: str,
    ) -> dict[str, Any]:
        calls.append((received_args, received_config, received_digest))
        return {"status": "delegated"}

    monkeypatch.setattr(wrapper.base, "freeze_target_free_roster", fake_freeze)
    assert wrapper.run(args) == {"status": "delegated"}
    assert calls == [(args, config, digest)]


def test_wrapper_rejects_runtime_source_path_drift(tmp_path: Path) -> None:
    config = _json(wrapper.DEFAULT_CONFIG)
    args = argparse.Namespace(
        manifest=wrapper._project_path(
            config["frozen_inputs"]["validation_manifest"]["path"]
        ),
        targets=tmp_path / "other-targets",
    )
    with pytest.raises(RuntimeError, match="targets directory"):
        wrapper._validate_runtime_paths(args, config)


def test_wrapper_contract_rejects_pixels_labels_or_non_strict_output() -> None:
    config = _json(wrapper.DEFAULT_CONFIG)
    for key in (
        "contains_pixels",
        "contains_exact_references_or_labels",
        "strict_original_upright_permutations",
    ):
        changed = copy.deepcopy(config)
        changed["output_contract"][key] = not changed["output_contract"][key]
        with pytest.raises(RuntimeError, match="output contract"):
            wrapper._require_exact_contract(changed)
