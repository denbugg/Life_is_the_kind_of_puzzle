from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.synthetic_socket_evaluation import names_digest
from scripts import run_joint_reciprocal_scale256_real as scale

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNED_CONFIG = (
    PROJECT_ROOT / "configs/joint_reciprocal_scale256_real_preregistered_v1.json"
)


def _args(tmp_path: Path, mode: str = "fit") -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        config=SIGNED_CONFIG,
        experiment_dir=tmp_path / "new-experiment",
        manifest=scale._project_path(scale.IMMUTABLE_BASE_ARTIFACTS["manifest"][0]),
        targets=scale.base.prior.roster.DEFAULT_TARGETS,
        socket_checkpoint=scale._project_path(
            scale.IMMUTABLE_BASE_ARTIFACTS["socket_checkpoint"][0]
        ),
        device="cpu",
        allow_nondeterministic_mps=False,
    )


def test_signed_scale_contract_derives_exact_rosters_from_metadata_only() -> None:
    runtime, digest, rosters = scale.load_signed_runtime_config(SIGNED_CONFIG)
    assert digest == sha256_file(SIGNED_CONFIG)
    assert len(rosters["fit"]) == 256
    assert names_digest(rosters["fit"]) == scale.FIT_DIGEST
    assert len(rosters["dev"]) == 64
    assert names_digest(rosters["dev"]) == scale.DEV_DIGEST
    excluded = (
        set(rosters["opened_local16_owned"])
        | set(rosters["terminal16_owned"])
        | set(rosters["source_audit_excluded"])
    )
    assert not ((set(rosters["fit"]) | set(rosters["dev"])) & excluded)
    assert runtime["training"]["optimizer_updates"] == 1752
    assert runtime["training"]["from_scratch"] is True
    assert runtime["fixed_model"]["capacity_checkpoint_reuse"] is False


def test_scale_contract_rejects_update_or_cache_hash_drift() -> None:
    config = json.loads(SIGNED_CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config)
    changed["training"]["optimizer_updates"] = 1751
    with pytest.raises(RuntimeError, match="training contract"):
        scale.require_exact_contract(changed)
    changed = copy.deepcopy(config)
    changed["frozen_inputs"]["fit_cache_report"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="fit_cache_report"):
        scale.require_exact_contract(changed)


def test_frozen_v2_engine_bytes_are_unchanged() -> None:
    for key in ("base_runner", "base_module", "base_runner_test"):
        path, expected = scale.IMMUTABLE_BASE_ARTIFACTS[key]
        assert sha256_file(scale._project_path(path)) == expected


def test_fit_dispatch_delegates_only_to_base_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(tmp_path)
    runtime = {"training": {"from_scratch": True}}
    rosters = {"fit": ("fit.png",), "dev": ("reserved.png",)}
    calls: list[str] = []
    monkeypatch.setattr(scale, "_validate_runtime_paths", lambda *_: None)
    monkeypatch.setattr(
        scale,
        "load_signed_runtime_config",
        lambda _path: (runtime, "a" * 64, rosters),
    )

    def fake_fit(*_args: Any) -> dict[str, str]:
        calls.append("fit")
        return {"status": "ok"}

    def forbidden(*_args: Any) -> dict[str, str]:
        raise AssertionError("FIT dispatch touched a reserved DEV stage")

    monkeypatch.setattr(scale.base, "run_fit", fake_fit)
    monkeypatch.setattr(scale.base, "run_freeze_dev", forbidden)
    monkeypatch.setattr(scale.base, "run_score_dev", forbidden)
    assert scale.run_mode(args) == {"status": "ok"}
    assert calls == ["fit"]


def test_fit_refuses_resume_or_overwrite(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.experiment_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="resume/overwrite"):
        scale._validate_runtime_paths(args, "fit")
